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
        ctx.skill_memory = MagicMock()
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

        async def fake_tool_execute(*, sandbox_dir, sandbox_manager, memory_dir,
                                     filename, args, **_kw):
            captured_filenames.append(filename)
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


# ---------------------------------------------------------------------------
# Fix #2: n_results bumped to 15 in semantic skill routing
# ---------------------------------------------------------------------------
class TestSemanticRoutingNResults:
    def test_uses_n_results_15(self, tmp_path):
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

    def test_keeps_postgres_when_db_uri_is_set(self, tmp_path):
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
        ctx.args.default_db = "postgresql://user@host/db"

        defs = registry_mod.get_active_tool_definitions(ctx, query="totally unrelated query")
        names = {d["function"]["name"] for d in defs}
        assert "postgres_admin" in names, "postgres_admin must stay when DB URI is configured"


# ---------------------------------------------------------------------------
# Fix #4: swarm await_results=True returns aggregated results
# ---------------------------------------------------------------------------
class TestSwarmAwaitResults:
    @pytest.mark.asyncio
    async def test_await_results_true_blocks_and_aggregates(self, monkeypatch):
        from ghost_agent.tools import swarm as swarm_mod

        completed = {}

        async def fake_worker(instruction, input_data, output_key, llm_client,
                              fallback_model_name, scratchpad, worker_persona=None,
                              target_model=None, preselected_node=None):
            await asyncio.sleep(0.01)
            scratchpad.set(output_key, f"OK::{output_key}")
            completed[output_key] = True

        monkeypatch.setattr(swarm_mod, "_swarm_worker", fake_worker)

        # Build a fake llm_client and scratchpad
        llm = MagicMock()
        llm.swarm_clients = ["x"]  # truthy + non-empty
        llm.get_swarm_node = MagicMock(return_value={"client": MagicMock(), "model": "m", "url": "u"})

        scratch = MagicMock()
        seen = {}

        def _set(k, v):
            seen[k] = v

        scratch.set = _set

        result = await swarm_mod.tool_delegate_to_swarm(
            llm_client=llm,
            model_name="m",
            scratchpad=scratch,
            tasks=[
                {"instruction": "i1", "input_data": "d1", "output_key": "out1"},
                {"instruction": "i2", "input_data": "d2", "output_key": "out2"},
            ],
            await_results=True,
        )
        assert "completed (await_results=True)" in result
        assert "out1" in seen and seen["out1"] == "OK::out1"
        assert "out2" in seen and seen["out2"] == "OK::out2"
        assert completed == {"out1": True, "out2": True}

    @pytest.mark.asyncio
    async def test_await_results_false_returns_dispatched_marker(self, monkeypatch):
        from ghost_agent.tools import swarm as swarm_mod

        async def fake_worker(*a, **kw):
            await asyncio.sleep(0.01)

        monkeypatch.setattr(swarm_mod, "_swarm_worker", fake_worker)

        llm = MagicMock()
        llm.swarm_clients = ["x"]
        llm.get_swarm_node = MagicMock(return_value={"client": MagicMock(), "model": "m", "url": "u"})

        scratch = MagicMock()
        scratch.set = MagicMock()

        result = await swarm_mod.tool_delegate_to_swarm(
            llm_client=llm,
            model_name="m",
            scratchpad=scratch,
            tasks=[{"instruction": "i", "input_data": "d", "output_key": "out"}],
            await_results=False,
        )
        # Default fire-and-forget marker
        assert "dispatched" in result.lower()
        # Task ID should be reported back
        assert "swarm-" in result
        # Drain background task to keep the test runner clean
        await asyncio.sleep(0.05)


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
class TestDatabaseRowMarker:
    @pytest.mark.asyncio
    async def test_query_under_300_rows_reports_count(self, monkeypatch):
        # We need to mock psycopg2 + tabulate. Inject before import.
        import sys
        import types

        fake_psycopg2 = types.ModuleType("psycopg2")
        fake_psycopg2_extras = types.ModuleType("psycopg2.extras")

        class _FakeCursor:
            description = [("id",)]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, *a, **k):
                return None

            def fetchmany(self, n):
                return [{"id": i} for i in range(50)]  # under 300

            def fetchall(self):
                return []

        class _FakeConn:
            closed = 0
            autocommit = True

            def cursor(self, **k):
                return _FakeCursor()

            def close(self):
                pass

        fake_psycopg2.connect = lambda uri: _FakeConn()
        fake_psycopg2.extras = fake_psycopg2_extras
        fake_psycopg2_extras.RealDictCursor = object

        fake_tabulate = types.ModuleType("tabulate")
        fake_tabulate.tabulate = lambda rows, headers="keys", tablefmt="pipe": "FAKE_TABLE"

        sys.modules["psycopg2"] = fake_psycopg2
        sys.modules["psycopg2.extras"] = fake_psycopg2_extras
        sys.modules["tabulate"] = fake_tabulate

        # Now import (after patching), force fresh module load if needed
        if "ghost_agent.tools.database" in sys.modules:
            del sys.modules["ghost_agent.tools.database"]
        from ghost_agent.tools import database as db_mod

        # Reset the connection pool
        db_mod._connection_pool.clear()

        result = await db_mod.tool_postgres_admin(
            action="query",
            connection_string="postgresql://x",
            query="SELECT 1",
        )
        assert "50" in result and "rows" in result.lower()

    @pytest.mark.asyncio
    async def test_query_over_300_rows_reports_truncation_marker(self, monkeypatch):
        import sys
        import types

        fake_psycopg2 = types.ModuleType("psycopg2")
        fake_psycopg2_extras = types.ModuleType("psycopg2.extras")

        class _FakeCursor:
            description = [("id",)]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, *a, **k):
                return None

            def fetchmany(self, n):
                # Return 301 rows so the truncated branch fires
                return [{"id": i} for i in range(301)]

            def fetchall(self):
                return []

        class _FakeConn:
            closed = 0
            autocommit = True

            def cursor(self, **k):
                return _FakeCursor()

            def close(self):
                pass

        fake_psycopg2.connect = lambda uri: _FakeConn()
        fake_psycopg2.extras = fake_psycopg2_extras
        fake_psycopg2_extras.RealDictCursor = object

        fake_tabulate = types.ModuleType("tabulate")
        fake_tabulate.tabulate = lambda rows, headers="keys", tablefmt="pipe": "FAKE_TABLE"

        sys.modules["psycopg2"] = fake_psycopg2
        sys.modules["psycopg2.extras"] = fake_psycopg2_extras
        sys.modules["tabulate"] = fake_tabulate

        if "ghost_agent.tools.database" in sys.modules:
            del sys.modules["ghost_agent.tools.database"]
        from ghost_agent.tools import database as db_mod

        db_mod._connection_pool.clear()

        result = await db_mod.tool_postgres_admin(
            action="query",
            connection_string="postgresql://y",
            query="SELECT 1",
        )
        assert "300" in result
        assert ("max display" in result.lower()) or ("more rows" in result.lower())


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

    def test_non_json_text_still_gets_brace_substitution(self):
        from ghost_agent.tools.search import _clean_for_cpp

        # Non-JSON text containing braces should still get the rewrite to
        # avoid llama.cpp grammar parser crashes.
        out = _clean_for_cpp("hello {world}")
        assert "[" in out and "]" in out
        assert "{" not in out and "}" not in out


# ---------------------------------------------------------------------------
# Fix #10: qwen_bridge passes through extra params
# ---------------------------------------------------------------------------
class TestQwenBridgeKwargPassthrough:
    def test_extra_params_reach_native_handler(self, monkeypatch):
        # The qwen_agent dependency is heavy; mock it before import.
        import sys
        import types

        if "qwen_agent.tools.base" not in sys.modules:
            fake_qwen_tools_base = types.ModuleType("qwen_agent.tools.base")

            class _FakeBaseTool:
                pass

            def _register_tool(name):
                def deco(cls):
                    return cls
                return deco

            fake_qwen_tools_base.BaseTool = _FakeBaseTool
            fake_qwen_tools_base.register_tool = _register_tool

            fake_qwen = types.ModuleType("qwen_agent")
            fake_qwen_tools = types.ModuleType("qwen_agent.tools")
            sys.modules["qwen_agent"] = fake_qwen
            sys.modules["qwen_agent.tools"] = fake_qwen_tools
            sys.modules["qwen_agent.tools.base"] = fake_qwen_tools_base

        # Force fresh import of the bridge against our fakes
        if "ghost_agent.tools.qwen_bridge" in sys.modules:
            del sys.modules["ghost_agent.tools.qwen_bridge"]
        from ghost_agent.tools import qwen_bridge as qb

        captured = {}

        async def fake_tool_file_system(**kwargs):
            captured.update(kwargs)
            return "ok"

        # Patch the imported native handler
        monkeypatch.setattr(qb, "tool_file_system", fake_tool_file_system)

        # Bind a context
        ctx = MagicMock()
        ctx.sandbox_dir = "/tmp/anywhere"
        ctx.tor_proxy = None
        qb.set_context(ctx)

        # Build the bridge tool and call it
        tool = qb.GhostFileSystem()
        tool.call({
            "operation": "list_files",
            "path": "/x",
            "filename": "extra-alias",   # extra/non-named param
            "weird_extra": "should-pass-through",
        })

        assert captured.get("operation") == "list_files"
        assert captured.get("path") == "/x"
        # Extra params should land in kwargs
        assert captured.get("filename") == "extra-alias", captured
        assert captured.get("weird_extra") == "should-pass-through", captured


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


# ---------------------------------------------------------------------------
# Fix #12: composed_skills.register_composed_skills mutates the list
# ---------------------------------------------------------------------------
class TestRegisterComposedSkills:
    def test_register_appends_definitions(self, tmp_path):
        from ghost_agent.tools.composed_skills import (
            ComposedSkill,
            ComposedSkillRegistry,
            SkillStep,
            register_composed_skills,
        )

        # Build a context with a pre-populated composed-skill registry
        storage = tmp_path / "composed_skills"
        reg = ComposedSkillRegistry(storage_dir=storage)
        reg.register(ComposedSkill(
            name="my_macro",
            trigger_description="do the thing",
            steps=[
                SkillStep(tool_name="execute", description="step1",
                          param_template={"command": "$cmd"}),
            ],
        ))

        ctx = MagicMock()
        ctx.sandbox_dir = tmp_path
        # Pre-cache the registry on the context so the helper finds it.
        ctx._composed_skill_registry = reg

        defs = []
        added = register_composed_skills(defs, ctx)
        assert added >= 1
        names = {d["function"]["name"] for d in defs}
        assert "my_macro" in names

    def test_register_skips_shadowed_names(self, tmp_path):
        from ghost_agent.tools.composed_skills import (
            ComposedSkill,
            ComposedSkillRegistry,
            SkillStep,
            register_composed_skills,
        )

        storage = tmp_path / "composed_skills"
        reg = ComposedSkillRegistry(storage_dir=storage)
        reg.register(ComposedSkill(
            name="shadow_me",
            trigger_description="ignored",
            steps=[SkillStep(tool_name="execute", description="x")],
        ))

        ctx = MagicMock()
        ctx.sandbox_dir = tmp_path
        ctx._composed_skill_registry = reg

        defs = [{"type": "function", "function": {"name": "shadow_me"}}]
        added = register_composed_skills(defs, ctx)
        # No new entries should be added; the only entry is the original.
        assert added == 0
        assert len(defs) == 1


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
