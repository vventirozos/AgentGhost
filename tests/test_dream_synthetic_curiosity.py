"""Integration tests for the curiosity / frontier-driven self-play loop.

These tests use a real FrontierTracker (on a tmp dir) attached to a
MagicMock context, so the gate / classifier / delta logic runs against a
real JSON store rather than MagicMock-returning-MagicMock.
"""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from ghost_agent.core.dream import Dreamer
from ghost_agent.memory.frontier import FrontierTracker


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


def _make_llm_response(challenge, validator, setup=""):
    payload = {
        "challenge_prompt": challenge,
        "validation_script": validator,
    }
    if setup:
        payload["setup_script"] = setup
    return {"choices": [{"message": {"content": dict_to_xml(payload)}}]}


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
async def test_frontier_seed_injected_into_challenge_prompt(
    mock_agent_cls, mock_sandbox_cls, tmp_path, disable_self_play_templates
):
    """When the tracker has a brittle cluster, the challenge-gen LLM call
    must include a FRONTIER SEED section in the system message."""
    ft = FrontierTracker(tmp_path)
    ft.record_run("sql", "prior challenge", 3, False, 0, mistake="bad join")
    ft.record_run("sql", "prior 2", 3, False, 0, mistake="bad join")

    ctx = _make_context(tmp_path, frontier_tracker=ft)
    ctx.llm_client.chat_completion = AsyncMock(
        return_value=_make_llm_response("Write SQL", "assert True")
    )

    mock_agent = MagicMock()
    mock_agent.handle_chat = AsyncMock(return_value=("ok", None, None))
    mock_agent._get_recent_transcript.return_value = "transcript"
    mock_agent_cls.return_value = mock_agent
    mock_sandbox_cls.return_value = _make_sandbox()

    # Force deterministic frontier selection (no random exploration)
    with patch("ghost_agent.memory.frontier.random.random", return_value=1.0):
        dreamer = Dreamer(ctx)
        await dreamer.synthetic_self_play("test-model")

    # The first chat_completion call is the challenge generator
    first_call = ctx.llm_client.chat_completion.await_args_list[0]
    payload = first_call.args[0]
    system_msg = payload["messages"][1]["content"]  # user message carries the system_message
    assert "FRONTIER SEED" in system_msg
    assert "sql" in system_msg.lower()
    assert "bad join" in system_msg


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_frontier_records_first_try_success(
    mock_agent_cls, mock_sandbox_cls, tmp_path, disable_self_play_templates
):
    """A new-cluster first-try pass must be recorded in the tracker and
    must trigger a learn_lesson write (new cluster is always worth it)."""
    ft = FrontierTracker(tmp_path)
    ctx = _make_context(tmp_path, frontier_tracker=ft)
    ctx.llm_client.chat_completion = AsyncMock(
        return_value=_make_llm_response("Write a bash awk pipeline", "assert True")
    )
    # Learning extraction call returns a valid lesson JSON
    ctx.llm_client.chat_completion.side_effect = [
        _make_llm_response("Write a bash awk pipeline", "assert True"),
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"trigger": "transforming tab-delimited rows on the command line", '
                            '"anti_pattern": "", '
                            '"correct_pattern": "from subprocess import run; run([\\"awk\\", \\"-F\\\\\\\\t\\", \\"{print $NF}\\"], check=True)", '
                            '"domains": ["bash"], "confidence": 0.7, '
                            '"task": "bash awk", "mistake": "", "solution": "use awk"}'
                        )
                    }
                }
            ]
        },
    ]

    mock_agent = MagicMock()

    # S1: description_length is now a tool-invocation count taken
    # from body["messages"], not a transcript byte length. Simulate a
    # realistic 2-tool-call solve so the tracker records a non-zero
    # length for a passing run.
    async def fake_handle_chat(body, **kw):
        body.setdefault("messages", []).extend([
            {"role": "assistant", "tool_calls": [{"id": "1"}, {"id": "2"}]},
            {"role": "tool", "content": "ok"},
            {"role": "tool", "content": "ok"},
        ])
        return ("done", None, None)

    mock_agent.handle_chat = AsyncMock(side_effect=fake_handle_chat)
    mock_agent._get_recent_transcript.return_value = "a" * 500
    mock_agent_cls.return_value = mock_agent
    mock_sandbox_cls.return_value = _make_sandbox()

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play("test-model")

    stats = ft.get_cluster_stats("bash")
    assert stats["runs"] == 1
    # Tool-invocation count > 0 for a passing run with tool calls.
    assert stats["last_length"] > 0
    # New cluster → lesson should be written
    assert ctx.skill_memory.learn_lesson.called


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_frontier_failure_records_negative_delta(
    mock_agent_cls, mock_sandbox_cls, tmp_path, disable_self_play_templates
):
    ft = FrontierTracker(tmp_path)
    ctx = _make_context(tmp_path, frontier_tracker=ft)
    ctx.llm_client.chat_completion = AsyncMock(
        side_effect=[
            _make_llm_response("Write SQL query", "assert False"),
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"task": "sql", "mistake": "x", "solution": "y"}'
                        }
                    }
                ]
            },
        ]
    )

    mock_agent = MagicMock()
    mock_agent.handle_chat = AsyncMock(return_value=("tried", None, None))
    mock_agent._get_recent_transcript.return_value = "transcript"
    mock_agent_cls.return_value = mock_agent
    mock_sandbox_cls.return_value = _make_sandbox(
        validator_exit_code=1, validator_output="AssertionError: mismatch"
    )

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play("test-model")

    stats = ft.get_cluster_stats("sql")
    assert stats["runs"] == 1
    assert stats["last_compression"] == -1.0
    assert dreamer.last_compression_delta == -1.0


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_repeat_failure_on_known_cluster_suppresses_skill_write(
    mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch
):
    """After a cluster already has a prior failure, a second failure must
    NOT call learn_lesson — the skill gate suppresses duplicates."""
    # Force the LLM-generated challenge path so the cluster the test
    # primes (sql) is the cluster actually classified for the new run.
    # Pre-2026-05 the template fast path was non-deterministic across
    # test orderings — _LAST_TEMPLATE_KEY pollution from prior tests
    # could route this to bash, turning the test's primed-sql scenario
    # into a "first failure on new cluster" case that legitimately
    # opens the new write gate. Disabling both template entry points
    # makes the cluster classification deterministic.
    monkeypatch.setattr(
        "ghost_agent.core.challenge_templates.try_template",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "ghost_agent.core.challenge_templates.pick_random_template",
        lambda *a, **kw: None,
    )
    ft = FrontierTracker(tmp_path)
    # Prime the tracker: cluster already known with one failure
    ft.record_run("sql", "prior", 3, False, 0, mistake="prior fail")

    ctx = _make_context(tmp_path, frontier_tracker=ft)
    ctx.llm_client.chat_completion = AsyncMock(
        side_effect=[
            _make_llm_response("Write SQL join", "assert False"),
            # Learning extraction should never be reached, but provide
            # a fallback in case the gate logic regresses.
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"task": "x", "mistake": "y", "solution": "z"}'
                        }
                    }
                ]
            },
        ]
    )

    mock_agent = MagicMock()
    mock_agent.handle_chat = AsyncMock(return_value=("tried", None, None))
    mock_agent._get_recent_transcript.return_value = "t"
    mock_agent_cls.return_value = mock_agent
    mock_sandbox_cls.return_value = _make_sandbox(
        validator_exit_code=1, validator_output="AssertionError: mismatch"
    )

    dreamer = Dreamer(ctx)
    result = await dreamer.synthetic_self_play("test-model")

    # Skill gate must suppress
    ctx.skill_memory.learn_lesson.assert_not_called()
    assert "suppress" in result or "gate" in result.lower() or "FAILURE" in result


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_validator_called_process_error_is_not_validator_crash(
    mock_agent_cls, mock_sandbox_cls, tmp_path
):
    """A CalledProcessError traceback inside .validator.py must NOT be
    flagged as a validator crash (that was the bug) — the loop should
    retry all 3 attempts."""
    ft = FrontierTracker(tmp_path)
    ctx = _make_context(tmp_path, frontier_tracker=ft)
    ctx.llm_client.chat_completion = AsyncMock(
        side_effect=[
            _make_llm_response("Python task", "assert True"),
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"task": "x", "mistake": "y", "solution": "z"}'
                        }
                    }
                ]
            },
        ]
    )

    called_process_trace = (
        'Traceback (most recent call last):\n'
        '  File ".validator.py", line 5, in <module>\n'
        '    subprocess.check_output(["python3", "solution.py"])\n'
        'subprocess.CalledProcessError: Command failed with exit status 1'
    )

    attempt_counter = {"n": 0}

    def execute(cmd, *a, **kw):
        if "py_compile" in cmd:
            return ("Syntax OK", 0)
        if ".validator.py" in cmd:
            attempt_counter["n"] += 1
            return (called_process_trace, 1)
        return ("", 0)

    mock_sandbox = MagicMock()
    mock_sandbox.execute.side_effect = execute
    mock_sandbox_cls.return_value = mock_sandbox

    mock_agent = MagicMock()
    mock_agent.handle_chat = AsyncMock(return_value=("tried", None, None))
    mock_agent._get_recent_transcript.return_value = "t"
    mock_agent_cls.return_value = mock_agent

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play("test-model")

    # All 3 attempts must run — NOT early-aborted by the crash heuristic
    assert attempt_counter["n"] == 3


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_validator_syntax_error_still_aborts(
    mock_agent_cls, mock_sandbox_cls, tmp_path
):
    """The tighter heuristic must still catch unambiguous validator syntax
    errors — they should not silently retry 3 times."""
    ft = FrontierTracker(tmp_path)
    ctx = _make_context(tmp_path, frontier_tracker=ft)
    ctx.llm_client.chat_completion = AsyncMock(
        side_effect=[
            _make_llm_response("Python task", "assert True"),
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"task": "x", "mistake": "y", "solution": "z"}'
                        }
                    }
                ]
            },
        ]
    )

    # Note: py_compile is bypassed here to let the runtime path hit.
    syntax_trace = (
        'Traceback (most recent call last):\n'
        '  File ".validator.py", line 12\n'
        '    if x = 5:\n'
        '         ^\n'
        'SyntaxError: invalid syntax'
    )

    attempt_counter = {"n": 0}

    def execute(cmd, *a, **kw):
        if "py_compile" in cmd:
            return ("Syntax OK", 0)  # pre-check passes, runtime still fails
        if ".validator.py" in cmd:
            attempt_counter["n"] += 1
            return (syntax_trace, 1)
        return ("", 0)

    mock_sandbox = MagicMock()
    mock_sandbox.execute.side_effect = execute
    mock_sandbox_cls.return_value = mock_sandbox

    mock_agent = MagicMock()
    mock_agent.handle_chat = AsyncMock(return_value=("tried", None, None))
    mock_agent._get_recent_transcript.return_value = "t"
    mock_agent_cls.return_value = mock_agent

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play("test-model")

    # Runtime SyntaxError should abort on attempt 1
    assert attempt_counter["n"] == 1


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_last_compression_delta_exposed_on_success(
    mock_agent_cls, mock_sandbox_cls, tmp_path, disable_self_play_templates
):
    """dreamer.last_compression_delta must be populated after a run so
    the biological watchdog can read it."""
    ft = FrontierTracker(tmp_path)
    # Prime with a previous passing run so the new run can show improvement
    ft.record_run("data_analysis", "prior", 1, True, 2000)

    ctx = _make_context(tmp_path, frontier_tracker=ft)
    ctx.llm_client.chat_completion = AsyncMock(
        side_effect=[
            _make_llm_response("Use pandas dataframe to parse", "assert True"),
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"task": "x", "mistake": "", "solution": "use pandas"}'
                        }
                    }
                ]
            },
        ]
    )

    mock_agent = MagicMock()
    mock_agent.handle_chat = AsyncMock(return_value=("ok", None, None))
    mock_agent._get_recent_transcript.return_value = "a" * 400  # much shorter than 2000
    mock_agent_cls.return_value = mock_agent
    mock_sandbox_cls.return_value = _make_sandbox()

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play("test-model")

    assert dreamer.last_compression_delta > 0
    stats = ft.get_cluster_stats("data_analysis")
    assert stats["runs"] == 2
    assert stats["best_length"] < 2000


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_no_frontier_tracker_falls_back_gracefully(
    mock_agent_cls, mock_sandbox_cls, tmp_path
):
    """If the context has no frontier_tracker (cold boot / tests), the
    function must still run end-to-end without crashing."""
    ctx = _make_context(tmp_path, frontier_tracker=None)
    ctx.llm_client.chat_completion = AsyncMock(
        side_effect=[
            _make_llm_response("Simple task", "assert True"),
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"task": "x", "mistake": "", "solution": "y"}'
                        }
                    }
                ]
            },
        ]
    )

    mock_agent = MagicMock()
    mock_agent.handle_chat = AsyncMock(return_value=("ok", None, None))
    mock_agent._get_recent_transcript.return_value = "transcript"
    mock_agent_cls.return_value = mock_agent
    mock_sandbox_cls.return_value = _make_sandbox()

    dreamer = Dreamer(ctx)
    result = await dreamer.synthetic_self_play("test-model")

    assert "SUCCESS" in result
    assert dreamer.last_compression_delta == 0.0
