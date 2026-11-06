"""Unit tests for tool registry / sandbox audit fixes.

Covers fixes #1-#13 from the audit. The sandbox-only fixes (#14, #15)
live in ``test_sandbox_audit_fixes.py``.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure src is on the path (tests run with PYTHONPATH=src per CLAUDE.md but
# guard anyway so a bare `pytest` invocation still works).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Fix #1: acquired-skills closure captures correct skill name per iteration
# ---------------------------------------------------------------------------
class TestAcquiredSkillsClosure:
    @pytest.mark.asyncio
    async def test_each_runner_calls_its_own_skill(self, tmp_path, monkeypatch):
        """Register two acquired skills, invoke each, and assert each closure
        runs the *correct* underlying skill name (not the last one registered).
        """
        from ghost_agent.tools import registry as registry_mod
        from ghost_agent.tools.acquired_skills import AcquiredSkillManager

        # Build a context that enables the dynamic acquired-skills branch.
        ctx = MagicMock()
        ctx.sandbox_dir = tmp_path
        ctx.memory_dir = tmp_path
        ctx.sandbox_manager = MagicMock()
        ctx.memory_system = MagicMock()
        ctx.memory_system.collection = MagicMock()
        # Pretend the semantic-routing query returns nothing (so all active
        # skills are advertised).
        ctx.memory_system.collection.query.return_value = {"metadatas": [[]]}
        ctx.llm_client = MagicMock()
        ctx.llm_client.image_gen_clients = None
        ctx.tor_proxy = None
        ctx.scratchpad = MagicMock()
        ctx.scheduler = MagicMock()
        ctx.profile_memory = MagicMock()
        ctx.args = MagicMock()
        ctx.args.anonymous = True
        ctx.args.max_context = 8192
        ctx.args.default_db = None
        ctx.args.model = "test-model"

        # Materialize two real skills on disk via the manager.
        mgr = AcquiredSkillManager(tmp_path, ctx.memory_system)
        mgr.save_skill("alpha_skill", "alpha desc", {"type": "object"}, "print('alpha')")
        mgr.save_skill("beta_skill", "beta desc", {"type": "object"}, "print('beta')")

        # Capture which filename `tool_execute` was invoked with each call.
        captured_filenames: list[str] = []
        captured_ctx: list = []

        async def fake_tool_execute(*, sandbox_dir, sandbox_manager, memory_dir,
                                     filename, args, **_kw):
            captured_filenames.append(filename)
            captured_ctx.append(_kw.get("run_context"))
            return f"--- EXECUTION RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\n{filename}"

        monkeypatch.setattr(registry_mod, "tool_execute", fake_tool_execute)

        tools = registry_mod.get_available_tools(ctx)
        assert "alpha_skill" in tools, "alpha_skill must be registered"
        assert "beta_skill" in tools, "beta_skill must be registered"

        # Invoke each registered runner.
        await tools["alpha_skill"](foo="bar")
        await tools["beta_skill"](baz="qux")

        # Each call must have used its own filename — NOT the last one.
        assert captured_filenames == [
            "acquired_skills/alpha_skill.py",
            "acquired_skills/beta_skill.py",
        ], f"closure scope leaked: {captured_filenames}"
        # 11-05 r19: the runner hands its context on, so the no-DB-server note knows whether labs exist
        assert captured_ctx == [ctx, ctx]


# ---------------------------------------------------------------------------
# Fix #2: n_results bumped to 15 in semantic skill routing
# ---------------------------------------------------------------------------
class TestSemanticRoutingNResults:
    def test_uses_n_results_15(self, tmp_path, monkeypatch):
        # See test_registry_skills.py — routing only runs above the
        # small-catalogue threshold; this test is about the routing call.
        # -1, not 0: this fixture registers NO real skills, so the active
        # count is 0 and `0 > 0` would still skip routing.
        monkeypatch.setattr(
            'ghost_agent.tools.registry._SKILL_ROUTING_MIN_SKILLS', -1)
        from ghost_agent.tools import registry as registry_mod

        ctx = MagicMock()
        ctx.sandbox_dir = tmp_path
        ctx.memory_dir = tmp_path
        ctx.memory_system = MagicMock()
        ctx.memory_system.collection = MagicMock()
        ctx.memory_system.collection.query.return_value = {"metadatas": [[]]}
        ctx.llm_client = MagicMock()
        ctx.llm_client.image_gen_clients = None
        ctx.args = MagicMock()
        ctx.args.default_db = None

        registry_mod.get_active_tool_definitions(ctx, query="anything triggers semantic routing")

        # Inspect the kwargs of the .query call
        ctx.memory_system.collection.query.assert_called()
        kwargs = ctx.memory_system.collection.query.call_args.kwargs
        assert kwargs.get("n_results") == 15, f"expected n_results=15, got {kwargs.get('n_results')}"


# ---------------------------------------------------------------------------
# Fix #3: intent filter no longer drops image/vision aggressively
# ---------------------------------------------------------------------------
class TestIntentFilterPermissive:
    def test_keeps_vision_when_query_is_unrelated(self, tmp_path):
        from ghost_agent.tools import registry as registry_mod

        ctx = MagicMock()
        ctx.sandbox_dir = tmp_path
        ctx.memory_dir = tmp_path
        ctx.memory_system = MagicMock()
        ctx.memory_system.collection = MagicMock()
        ctx.memory_system.collection.query.return_value = {"metadatas": [[]]}
        ctx.llm_client = MagicMock()
        ctx.llm_client.image_gen_clients = ["fake-client"]  # image_gen advertised
        ctx.args = MagicMock()
        ctx.args.default_db = None  # postgres dropped (configured-required)

        defs = registry_mod.get_active_tool_definitions(ctx, query="hello, what time is it?")

        names = {d["function"]["name"] for d in defs}
        # Vision and image_gen MUST stay even though the query mentions neither.
        assert "vision_analysis" in names, "vision_analysis was wrongly dropped"
        assert "image_generation" in names, "image_generation was wrongly dropped"
        # postgres_admin is dropped because no DB URI is configured.
        assert "postgres_admin" not in names, "postgres_admin should be dropped when default_db is unset"


# ---------------------------------------------------------------------------
# Fix #5: execute.py output truncation keeps head AND tail
# ---------------------------------------------------------------------------
class TestExecuteTruncation:
    @pytest.mark.asyncio
    async def test_oversized_output_delegated_to_sandbox_spill(self, tmp_path):
        """2026-07-07 (#10): truncation moved from execute.py to the sandbox
        layer (spill mode). execute.py now requests spill and passes the
        sandbox's already-trimmed result through — head+tail preserved. The
        real head+tail/spill behavior is covered by test_execute_output_spill.py
        against the actual DockerSandbox."""
        from ghost_agent.tools import execute as execute_mod

        head_marker = "HEAD_MARKER_START"
        tail_marker = "TAIL_MARKER_END"
        # Simulate what the sandbox layer returns after spill-truncation:
        # a small head+tail view with the banner.
        trimmed = (head_marker + "\n[... 600000 chars truncated (run output) ...]\n"
                   + tail_marker + "\n[Full output saved to '.ghost_runs/run_1.log' ...]")
        sm = MagicMock()
        sm.execute = MagicMock(return_value=(trimmed, 0))

        script_dir = tmp_path
        script_dir.mkdir(exist_ok=True)
        (script_dir / "test_script.py").write_text("print('hi')")

        result = await execute_mod.tool_execute(
            filename="test_script.py",
            sandbox_dir=script_dir,
            sandbox_manager=sm,
        )
        # execute.py requested spill mode and passed the trimmed view through.
        assert sm.execute.call_args.kwargs.get("spill_large_output") is True
        assert head_marker in result and tail_marker in result
        assert "truncated" in result.lower()
        assert ".ghost_runs/run_" in result

    @pytest.mark.asyncio
    async def test_small_output_passes_through_unchanged(self, tmp_path):
        from ghost_agent.tools import execute as execute_mod

        small = "Hello, world."
        sm = MagicMock()
        sm.execute = MagicMock(return_value=(small, 0))
        script_path = tmp_path / "tiny.py"
        script_path.write_text("print('hi')")

        result = await execute_mod.tool_execute(
            filename="tiny.py",
            sandbox_dir=tmp_path,
            sandbox_manager=sm,
        )
        assert small in result
        assert "truncated" not in result.lower()


# ---------------------------------------------------------------------------
# Fix #6: file_system search output truncation keeps head AND tail
# ---------------------------------------------------------------------------
class TestFileSearchTruncation:
    @pytest.mark.asyncio
    async def test_search_oversized_keeps_head_and_tail(self, tmp_path):
        from ghost_agent.tools import file_system as fs_mod

        head_marker = "FS_HEAD_MARKER"
        tail_marker = "FS_TAIL_MARKER"
        big = head_marker + ("y" * (60 * 1024)) + tail_marker

        sm = MagicMock()
        sm.execute = MagicMock(return_value=(big, 0))

        result = await fs_mod.tool_file_search(
            "pattern", tmp_path, filename=None, sandbox_manager=sm
        )
        assert head_marker in result
        assert tail_marker in result
        assert "truncated" in result.lower()


# ---------------------------------------------------------------------------
# Fix #7: database row count marker
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fix #8: search _clean_for_cpp leaves valid JSON unchanged
# ---------------------------------------------------------------------------
class TestCleanForCpp:
    def test_passes_valid_json_unchanged(self):
        from ghost_agent.tools.search import _clean_for_cpp

        payload = '{"a": 1, "b": [1, 2, 3], "c": {"nested": true}}'
        out = _clean_for_cpp(payload)
        # Must still parse as JSON after the call
        parsed = json.loads(out)
        assert parsed == {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}

    def test_non_json_text_gets_brace_substitution_only_while_the_grammar_runs(self, monkeypatch):
        from ghost_agent.tools.search import _clean_for_cpp
        from ghost_agent.core import tool_grammar as TG

        # §4PD: the rewrite exists for the llama.cpp grammar parser; with that
        # parser off (the default) titles and snippets keep their braces and
        # angle brackets — `std::vector<int>` is what the owner searched for
        monkeypatch.setattr(TG, "grammar_enabled", lambda: False)
        assert _clean_for_cpp("std::vector<int> {x}") == "std::vector<int> {x}"
        monkeypatch.setattr(TG, "grammar_enabled", lambda: True)
        out = _clean_for_cpp("hello {world}")
        assert "[" in out and "]" in out
        assert "{" not in out and "}" not in out


# ---------------------------------------------------------------------------
# Fix #11: tool_failure.get_fallback_hint returns expected hints
# ---------------------------------------------------------------------------
class TestFallbackHint:
    def test_known_tool_known_error_returns_hint(self):
        from ghost_agent.tools.tool_failure import get_fallback_hint

        h = get_fallback_hint("execute", "ModuleNotFoundError: no module named 'xyz'")
        assert h is not None
        assert "pip install" in h

    def test_unknown_tool_returns_none(self):
        from ghost_agent.tools.tool_failure import get_fallback_hint
        assert get_fallback_hint("totally_made_up", "anything") is None

    def test_known_tool_unknown_error_returns_none(self):
        from ghost_agent.tools.tool_failure import get_fallback_hint
        assert get_fallback_hint("execute", "some unrelated noise") is None

    def test_empty_inputs_return_none(self):
        from ghost_agent.tools.tool_failure import get_fallback_hint
        assert get_fallback_hint("", "x") is None
        assert get_fallback_hint("execute", "") is None
        assert get_fallback_hint("execute", None) is None  # type: ignore


# (Fix #12 — composed_skills.register_composed_skills mutates the list — went with the macro engine, §4TA)


# ---------------------------------------------------------------------------
# Fix #13: acquired-skills retirement triggers on 3 consecutive failures
# ---------------------------------------------------------------------------
class TestRetirementThreshold:
    def test_three_consecutive_failures_retires(self, tmp_path):
        from ghost_agent.tools.acquired_skills import AcquiredSkillManager

        mgr = AcquiredSkillManager(tmp_path, memory_system=None)
        mgr.save_skill("flaky_tool", "flaky", {}, "print('x')")

        # Drive the failure_count to exactly 3 (the new consecutive threshold).
        # We can't easily simulate a real "consecutive" streak via log_telemetry
        # without 3 sequential calls — do that explicitly to mirror reality.
        mgr.log_telemetry("flaky_tool", success=False)
        mgr.log_telemetry("flaky_tool", success=False)
        mgr.log_telemetry("flaky_tool", success=False)

        registry = mgr._load_registry()
        assert registry["flaky_tool"]["failure_count"] == 3

        retired = mgr.retire_degraded_skills()
        assert "flaky_tool" in retired

    def test_two_consecutive_failures_does_not_retire(self, tmp_path):
        from ghost_agent.tools.acquired_skills import AcquiredSkillManager

        mgr = AcquiredSkillManager(tmp_path, memory_system=None)
        mgr.save_skill("ok_tool", "ok", {}, "print('y')")
        mgr.log_telemetry("ok_tool", success=False)
        mgr.log_telemetry("ok_tool", success=False)

        retired = mgr.retire_degraded_skills()
        assert retired == []

    def test_success_after_failures_resets_streak(self, tmp_path):
        from ghost_agent.tools.acquired_skills import AcquiredSkillManager

        mgr = AcquiredSkillManager(tmp_path, memory_system=None)
        mgr.save_skill("recovers_tool", "ok", {}, "print('z')")
        mgr.log_telemetry("recovers_tool", success=False)
        mgr.log_telemetry("recovers_tool", success=False)
        mgr.log_telemetry("recovers_tool", success=True)  # resets streak
        mgr.log_telemetry("recovers_tool", success=False)

        retired = mgr.retire_degraded_skills()
        assert retired == [], "successful call should have reset the streak"
