"""Tests for the post-log fixes:
  1. create_skill / manage_skills disabled in self-play
  2. tool_create_skill tolerates unknown kwargs (filename rescue + pass-through)
  3. suppress_meta_task_nudges flag blocks the checklist nudge
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.dream import Dreamer
from ghost_agent.tools.acquired_skills import tool_create_skill


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


# ------------------------------------------------ 1. disabled tools in self-play


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_create_skill_disabled_in_self_play(
    mock_agent_cls, mock_sandbox_cls, tmp_path
):
    """create_skill and manage_skills must be added to temp_agent.disabled_tools
    so the isolated simulation cannot touch real acquired_skills state."""
    ctx = _make_context(tmp_path)
    good = {
        "challenge_prompt": "task",
        "setup_script": 'with open("a.csv","w") as f: f.write("1\\n")',
        "validation_script": 'import subprocess\nopen("a.csv").read()\nsubprocess.run(["python3","solution.py"])\n',
    }
    ctx.llm_client.chat_completion = AsyncMock(
        side_effect=[
            {"choices": [{"message": {"content": dict_to_xml(good)}}]},
            {"choices": [{"message": {"content": '{"task":"x","mistake":"","solution":"y"}'}}]},
        ]
    )

    captured = {}

    class StubAgent:
        def __init__(self, ctx):
            self.disabled_tools = set()
            self.available_tools = {
                "create_skill": lambda **kw: "ok",
                "manage_skills": lambda **kw: "ok",
                "learn_skill": lambda **kw: "ok",
                "file_system": lambda **kw: "ok",
            }
            self.max_turns_override = None
            self.max_thinking_chars_override = None
            self.suppress_meta_task_nudges = False

        async def handle_chat(self, body, *a, **kw):
            captured["disabled"] = set(self.disabled_tools)
            captured["available"] = dict(self.available_tools)
            captured["max_turns"] = self.max_turns_override
            captured["max_think"] = self.max_thinking_chars_override
            captured["suppress_nudge"] = self.suppress_meta_task_nudges
            return ("ok", None, None)

        def _get_recent_transcript(self, messages):
            return "t"

    mock_agent_cls.side_effect = StubAgent
    mock_sandbox_cls.return_value = _make_sandbox()

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play("test-model")

    assert "create_skill" in captured["disabled"]
    assert "manage_skills" in captured["disabled"]
    assert "learn_skill" in captured["disabled"]
    # After disabling, they must also be removed from available_tools
    assert "create_skill" not in captured["available"]
    assert "manage_skills" not in captured["available"]
    # Budget overrides also applied
    assert captured["max_turns"] == 15
    assert captured["max_think"] == 12000
    assert captured["suppress_nudge"] is True


# ------------------------------------------------------ 2. tool_create_skill


class TestCreateSkillKwargTolerance:
    @pytest.mark.asyncio
    async def test_unknown_kwarg_no_typeerror(self, tmp_path):
        """Pre-fix this raised TypeError('unexpected keyword argument filename').
        Post-fix it must return a structured SYSTEM ERROR string."""
        result = await tool_create_skill(
            sandbox_dir=tmp_path,
            memory_system=None,
            sandbox_manager=None,
            filename="something.py",  # unknown kwarg
            description="d",
            parameters_schema='{"type":"object"}',
            python_code="print('x')",
            test_payload="{}",
        )
        assert isinstance(result, str)
        # Should not have raised — check it got through the signature
        # The filename rescue populated name → validation succeeds past
        # the MANDATORY check, so the result will go further. Either way:
        # no TypeError.

    @pytest.mark.asyncio
    async def test_filename_rescued_to_name(self, tmp_path):
        """When the LLM sends `filename` but no `name`, the rescue path
        should strip .py and use it as the skill name so validation
        proceeds past the MANDATORY check."""
        # Fake sandbox_manager so we don't spin up Docker
        fake_sandbox = MagicMock()
        fake_sandbox.execute.return_value = ("passed", 0)

        result = await tool_create_skill(
            sandbox_dir=tmp_path,
            memory_system=None,
            sandbox_manager=fake_sandbox,
            filename="my_skill.py",  # should become name="my_skill"
            description="desc",
            parameters_schema='{"type":"object"}',
            python_code="if __name__ == '__main__':\n    print('ok')",
            test_payload="{}",
        )
        # "name is mandatory" should NOT appear because filename was rescued
        assert "MANDATORY" not in result or "name" not in result.lower()

    @pytest.mark.asyncio
    async def test_missing_name_still_errors_cleanly(self, tmp_path):
        """Unknown kwarg that is NOT filename → name stays None → clean
        SYSTEM ERROR, not a TypeError."""
        result = await tool_create_skill(
            sandbox_dir=tmp_path,
            memory_system=None,
            sandbox_manager=None,
            weird_param="ignored",
            description="d",
            parameters_schema='{"type":"object"}',
            python_code="print('x')",
            test_payload="{}",
        )
        assert "MANDATORY" in result
        assert "name" in result.lower()

    @pytest.mark.asyncio
    async def test_filename_without_py_extension(self, tmp_path):
        """filename='foo' (no .py) should still become name='foo'."""
        fake_sandbox = MagicMock()
        fake_sandbox.execute.return_value = ("passed", 0)

        result = await tool_create_skill(
            sandbox_dir=tmp_path,
            memory_system=None,
            sandbox_manager=fake_sandbox,
            filename="foo",
            description="d",
            parameters_schema='{"type":"object"}',
            python_code="if __name__ == '__main__':\n    print('ok')",
            test_payload="{}",
        )
        assert "MANDATORY" not in result

    @pytest.mark.asyncio
    async def test_explicit_name_wins_over_filename(self, tmp_path):
        """If both name and filename are present, name is authoritative."""
        fake_sandbox = MagicMock()
        fake_sandbox.execute.return_value = ("passed", 0)

        result = await tool_create_skill(
            sandbox_dir=tmp_path,
            memory_system=None,
            sandbox_manager=fake_sandbox,
            name="real_name",
            filename="bogus.py",
            description="d",
            parameters_schema='{"type":"object"}',
            python_code="if __name__ == '__main__':\n    print('ok')",
            test_payload="{}",
        )
        assert "MANDATORY" not in result


# ---------------------------------------------------- 3. suppress_meta_task_nudges


class TestSuppressMetaTaskNudges:
    def test_default_is_false(self):
        from ghost_agent.core.agent import GhostAgent
        agent = GhostAgent.__new__(GhostAgent)
        assert getattr(agent, "suppress_meta_task_nudges", False) is False

    def test_override_attribute_respected(self):
        from ghost_agent.core.agent import GhostAgent
        agent = GhostAgent.__new__(GhostAgent)
        agent.suppress_meta_task_nudges = True
        assert getattr(agent, "suppress_meta_task_nudges", False) is True

    def test_nudge_block_guards_on_attribute(self):
        """Verify the agent.py source wires the suppress flag into the
        checklist-nudge conditional (structural test so we don't have to
        run the full handle_chat loop)."""
        import inspect
        from ghost_agent.core.agent import GhostAgent
        src = inspect.getsource(GhostAgent)
        assert "suppress_meta_task_nudges" in src
        # Must be checked before the nudge fires
        nudge_idx = src.index('"Checklist Nudge"')
        suppress_idx = src.index("suppress_meta_task_nudges")
        assert suppress_idx < nudge_idx

    def test_readonly_meta_tools_discharge_nudge(self):
        """`list_lessons` / `recall` are READ-ONLY surface tools — when
        the user asks "what have you learned today" the agent should
        call `list_lessons` and the nudge must treat that as meta-task
        satisfied. Without this exemption the nudge kept firing in
        production and the model eventually wrote a noop deduplicated
        skill to silence it (trace 15:17, request 5C)."""
        import inspect
        from ghost_agent.core.agent import GhostAgent
        src = inspect.getsource(GhostAgent)
        # The exemption set must mention at least `list_lessons`.
        assert "list_lessons" in src
        # And it must be used to set meta_tools_called = True before
        # the nudge check. We assert the read-only exemption lines
        # appear textually before the nudge log line.
        exempt_idx = src.index("read_only_meta_tools_called")
        nudge_idx = src.index('"Checklist Nudge"')
        assert exempt_idx < nudge_idx
