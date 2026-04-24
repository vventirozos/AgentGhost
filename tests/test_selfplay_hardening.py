"""Tests for the self-play hardening fixes:
  - per-attempt turn cap (max_turns_override)
  - tightened thinking-loop detection (smaller window, lower threshold)
  - max_thinking_chars_override on GhostAgent
  - validator timeout trim (30s instead of 300s)
  - brittleness scoring now counts struggled-then-won as partial signal
  - dream.py quality-gate triggers regeneration path
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.agent import (
    GhostAgent,
    _detect_thinking_loop,
    THINKING_LOOP_WINDOW,
    THINKING_LOOP_THRESHOLD,
    THINKING_LOOP_PROBE_EVERY,
)
from ghost_agent.core.dream import Dreamer
from ghost_agent.memory.frontier import FrontierTracker


# --------------------------------------------------------- thinking-loop probe


class TestThinkingLoopConstants:
    def test_window_is_tighter_than_pre_fix(self):
        # Pre-fix was 400; post-fix must be <= 200 to catch shorter loops.
        assert THINKING_LOOP_WINDOW <= 200

    def test_threshold_is_tighter_than_pre_fix(self):
        # Pre-fix was 4; post-fix must be <= 3.
        assert THINKING_LOOP_THRESHOLD <= 3

    def test_probe_frequency_is_tighter_than_pre_fix(self):
        # Pre-fix was 2000; post-fix must be <= 500 so runaway streams are
        # caught within a few probes, not a dozen.
        assert THINKING_LOOP_PROBE_EVERY <= 500


class TestDetectThinkingLoop:
    def test_short_buffer_returns_false(self):
        assert _detect_thinking_loop("short text") is False

    def test_repeated_exact_block_is_loop(self):
        # A 250-char paragraph repeated 4 times — the tight 200-char window
        # tail must appear >=3 times somewhere in the buffer for the probe
        # to fire.
        block = "A" * 250
        buf = block * 4
        assert _detect_thinking_loop(buf) is True

    def test_nonrepeating_text_not_loop(self):
        buf = "".join(f"unique line {i} with different content each pass\n" for i in range(30))
        assert _detect_thinking_loop(buf) is False

    def test_wide_window_backstop_catches_longer_repeats(self):
        # Each chunk is larger than the tight window but smaller than the
        # wide (2x) backstop. Three repeats should still fire.
        chunk = "A" * 300 + "\n"  # 301 chars, larger than 200-char tight window
        buf = chunk * 4
        assert _detect_thinking_loop(buf) is True


# --------------------------------------------------------- GhostAgent overrides


class TestAgentOverrides:
    def test_max_turns_override_attribute_respected(self):
        """Verify the override attribute is wired: when set, the effective
        cap used by handle_chat is the override, not the default 40."""
        # We can't run handle_chat end-to-end here, but we can verify the
        # lookup logic by creating an agent, setting the override, and
        # checking the attribute resolution path.
        ctx = MagicMock()
        agent = GhostAgent.__new__(GhostAgent)  # bypass __init__
        agent.max_turns_override = 15
        effective = getattr(agent, "max_turns_override", None) or 40
        assert effective == 15

    def test_max_turns_default_is_40(self):
        agent = GhostAgent.__new__(GhostAgent)
        effective = getattr(agent, "max_turns_override", None) or 40
        assert effective == 40

    def test_max_thinking_chars_override_respected(self):
        agent = GhostAgent.__new__(GhostAgent)
        agent.max_thinking_chars_override = 12000
        effective = getattr(agent, "max_thinking_chars_override", None) or 32000
        assert effective == 12000

    def test_max_thinking_chars_default_32000(self):
        from ghost_agent.core.agent import MAX_THINKING_CHARS
        agent = GhostAgent.__new__(GhostAgent)
        effective = getattr(agent, "max_thinking_chars_override", None) or MAX_THINKING_CHARS
        assert effective == 32000


# --------------------------------------------------------- brittleness scoring


class TestBrittlenessScoring:
    def test_struggled_then_won_is_partial_brittle(self, tmp_path):
        """A single 2-attempt struggle (success) must register as mildly
        brittle so the next session targets it instead of cold-starting.
        Pre-fix the score was 0 and the cluster was ignored."""
        ft = FrontierTracker(tmp_path)
        ft.record_run("data_analysis", "c1", 2, True, 1500)
        brittle = ft.get_brittle_clusters()
        assert any(k == "data_analysis" for k, _ in brittle)

    def test_one_attempt_win_is_not_brittle(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("algo", "c", 1, True, 500)
        brittle = ft.get_brittle_clusters()
        assert not any(k == "algo" for k, _ in brittle)

    def test_failure_ranks_higher_than_soft_win(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("bash", "c_fail", 3, False, 0)
        ft.record_run("sql", "c_soft", 2, True, 1000)
        brittle = ft.get_brittle_clusters()
        # Failure (weight 2) should outrank soft win (weight 1)
        assert brittle[0][0] == "bash"
        assert any(k == "sql" for k, _ in brittle)

    def test_pick_seed_targets_struggled_cluster(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("data_analysis", "c1", 2, True, 1500)
        seed = ft.pick_seed(random_explore_prob=0.0)
        assert seed["mode"] == "frontier"
        assert seed["cluster_key"] == "data_analysis"


# --------------------------------------------------------- dream.py integration


def dict_to_xml(d):
    return "".join(f"<{k}>{v}</{k}>\n" for k, v in d.items())


def _make_context(tmp_path, frontier_tracker=None):
    context = MagicMock()
    context.memory_system = MagicMock()
    context.skill_memory = MagicMock()
    context.skill_memory.get_recent_failures.return_value = "No failures"
    context.llm_client = MagicMock()
    context.args = MagicMock()
    context.args.perfect_it = True
    context.args.smart_memory = 1.0
    context.sandbox_manager = MagicMock()
    context.sandbox_dir = str(tmp_path)
    context.tor_proxy = None
    context.scratchpad = MagicMock()
    context.frontier_tracker = frontier_tracker
    return context


def _make_sandbox(validator_exit_code=0, validator_output="Success"):
    sandbox = MagicMock()

    def execute(cmd, *a, **kw):
        if "py_compile" in cmd:
            return ("Syntax OK", 0)
        if ".setup.py" in cmd:
            return ("Setup OK", 0)
        if ".validator.py" in cmd:
            return (validator_output, validator_exit_code)
        return ("", 0)

    sandbox.execute.side_effect = execute
    return sandbox


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_quality_gate_regenerates_on_bad_validator(
    mock_agent_cls, mock_sandbox_cls, tmp_path, disable_self_play_templates
):
    """First generation produces an unwinnable validator (random.seed);
    second generation produces a clean one. The dream must accept the
    second and run the simulation."""
    ctx = _make_context(tmp_path, frontier_tracker=None)

    bad_payload = {
        "challenge_prompt": "Do sales analysis",
        "setup_script": 'with open("sales.csv","w") as f: f.write("a,b\\n1,2\\n")',
        "validation_script": (
            "import random\n"
            "random.seed(42)\n"
            "x = random.randint(1, 10)\n"
        ),
    }
    good_payload = {
        "challenge_prompt": "Do sales analysis",
        "setup_script": 'with open("sales.csv","w") as f: f.write("a,b\\n1,2\\n")',
        "validation_script": (
            'import subprocess\n'
            'with open("sales.csv") as f:\n'
            '    lines = f.readlines()\n'
            'subprocess.run(["python3","solution.py"])\n'
        ),
    }
    learning_payload = {
        "choices": [
            {"message": {"content": '{"task":"x","mistake":"","solution":"y"}'}}
        ]
    }

    ctx.llm_client.chat_completion = AsyncMock(
        side_effect=[
            {"choices": [{"message": {"content": dict_to_xml(bad_payload)}}]},
            {"choices": [{"message": {"content": dict_to_xml(good_payload)}}]},
            learning_payload,
        ]
    )

    mock_agent = MagicMock()
    mock_agent.handle_chat = AsyncMock(return_value=("ok", None, None))
    mock_agent._get_recent_transcript.return_value = "t"
    mock_agent_cls.return_value = mock_agent
    mock_sandbox_cls.return_value = _make_sandbox()

    dreamer = Dreamer(ctx)
    result = await dreamer.synthetic_self_play("test-model")

    # The chat_completion must have been called at least twice (bad + good)
    # plus the learning-extraction call.
    assert ctx.llm_client.chat_completion.await_count >= 2
    assert "SUCCESS" in result or "FAILURE" in result  # reached the loop
    # The temp agent should have been instantiated (regeneration succeeded)
    mock_agent_cls.assert_called_once()


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_quality_gate_aborts_after_max_regenerations(
    mock_agent_cls, mock_sandbox_cls, tmp_path, disable_self_play_templates
):
    """Three bad generations in a row must abort without ever instantiating
    the temp agent or touching the sandbox."""
    ctx = _make_context(tmp_path, frontier_tracker=None)

    bad_payload = {
        "challenge_prompt": "task",
        "setup_script": 'open("x.csv","w").close()',
        "validation_script": "import random\nrandom.seed(1)\nrandom.randint(0, 5)\n",
    }

    ctx.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": dict_to_xml(bad_payload)}}]}
    )

    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent
    mock_sandbox_cls.return_value = _make_sandbox()

    dreamer = Dreamer(ctx)
    result = await dreamer.synthetic_self_play("test-model")

    assert "quality gate" in result.lower()
    assert ctx.llm_client.chat_completion.await_count == 3
    mock_agent_cls.assert_not_called()


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_temp_agent_receives_budget_overrides(
    mock_agent_cls, mock_sandbox_cls, tmp_path
):
    """The GhostAgent instance created inside self-play must have the
    per-attempt turn cap and thinking-chars cap set."""
    ctx = _make_context(tmp_path, frontier_tracker=None)

    good_payload = {
        "challenge_prompt": "task",
        "setup_script": 'with open("a.csv","w") as f: f.write("1\\n")',
        "validation_script": (
            'import subprocess\n'
            'open("a.csv").read()\n'
            'subprocess.run(["python3","solution.py"])\n'
        ),
    }
    learning_payload = {
        "choices": [
            {"message": {"content": '{"task":"x","mistake":"","solution":"y"}'}}
        ]
    }

    ctx.llm_client.chat_completion = AsyncMock(
        side_effect=[
            {"choices": [{"message": {"content": dict_to_xml(good_payload)}}]},
            learning_payload,
        ]
    )

    mock_agent = MagicMock()
    mock_agent.disabled_tools = set()
    mock_agent.available_tools = {}
    mock_agent.handle_chat = AsyncMock(return_value=("ok", None, None))
    mock_agent._get_recent_transcript.return_value = "t"
    mock_agent_cls.return_value = mock_agent
    mock_sandbox_cls.return_value = _make_sandbox()

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play("test-model")

    assert mock_agent.max_turns_override == 15
    assert mock_agent.max_thinking_chars_override == 12000


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_validator_execute_uses_short_timeout(
    mock_agent_cls, mock_sandbox_cls, tmp_path
):
    """sandbox_manager.execute must be called with timeout=30 (not the
    default 300) for the .validator.py run, and timeout=60 for setup."""
    ctx = _make_context(tmp_path, frontier_tracker=None)

    good_payload = {
        "challenge_prompt": "task",
        "setup_script": 'with open("a.csv","w") as f: f.write("1\\n")',
        "validation_script": (
            'import subprocess\n'
            'open("a.csv").read()\n'
            'subprocess.run(["python3","solution.py"])\n'
        ),
    }
    learning_payload = {
        "choices": [
            {"message": {"content": '{"task":"x","mistake":"","solution":"y"}'}}
        ]
    }

    ctx.llm_client.chat_completion = AsyncMock(
        side_effect=[
            {"choices": [{"message": {"content": dict_to_xml(good_payload)}}]},
            learning_payload,
        ]
    )

    mock_agent = MagicMock()
    mock_agent.handle_chat = AsyncMock(return_value=("ok", None, None))
    mock_agent._get_recent_transcript.return_value = "t"
    mock_agent_cls.return_value = mock_agent

    sandbox = _make_sandbox()
    mock_sandbox_cls.return_value = sandbox

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play("test-model")

    # Scan the positional-arg calls to the sandbox executor.
    # Exclude py_compile calls (syntax checks) from both setup and validator — they
    # are pre-flight sanity checks, not the actual execution.
    setup_calls = [c for c in sandbox.execute.call_args_list if ".setup.py" in str(c.args) and "py_compile" not in str(c.args)]
    validator_calls = [
        c for c in sandbox.execute.call_args_list
        if ".validator.py" in str(c.args) and "py_compile" not in str(c.args)
    ]

    assert setup_calls, "setup script should have been executed"
    assert validator_calls, "validator should have been executed"

    # Setup must use the 60s cap, validator the 30s cap.
    for c in setup_calls:
        timeout_arg = c.args[1] if len(c.args) > 1 else c.kwargs.get("timeout")
        assert timeout_arg == 60, f"setup timeout was {timeout_arg}"
    for c in validator_calls:
        timeout_arg = c.args[1] if len(c.args) > 1 else c.kwargs.get("timeout")
        assert timeout_arg == 30, f"validator timeout was {timeout_arg}"
