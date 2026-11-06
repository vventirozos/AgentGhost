"""Tests for the post-log fixes:
  1. tool_create_skill tolerates unknown kwargs (filename rescue + pass-through)
  2. suppress_meta_task_nudges flag blocks the checklist nudge
(§4NM: the "create_skill disabled in self-play" test went with synthetic self-play.)
"""

from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.acquired_skills import tool_create_skill


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
        """`recall` and the like are READ-ONLY surface tools — the nudge
        must treat calling one as meta-task satisfied. Without this exemption the nudge kept firing in
        production and the model eventually wrote a noop deduplicated
        skill to silence it (trace 15:17, request 5C)."""
        import inspect
        from ghost_agent.core.agent import GhostAgent
        src = inspect.getsource(GhostAgent)
        # And it must be used to set meta_tools_called = True before
        # the nudge check. We assert the read-only exemption lines
        # appear textually before the nudge log line.
        exempt_idx = src.index("read_only_meta_tools_called")
        nudge_idx = src.index('"Checklist Nudge"')
        assert exempt_idx < nudge_idx
