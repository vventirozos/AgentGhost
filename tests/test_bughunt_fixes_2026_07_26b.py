"""Bug-hunt fixes from the 2026-07-26 (later) six-agent review.

Covers the verified findings fixed in that batch:
* calibration composite-leak — ConfidenceReading.pre_penalty_composite
* coding_executor _op_ok / _looks_like_write_error anchored classifiers
* workspace_cleanup data-deliverable partial-recovery
* search formatter None-trap
* project_advancer inter-project dependency deadlock
* work_log streamed-drain snapshot (concurrency)

The streamed-path calibration/post_mortem gate fixes are exercised by the
existing streaming + calibration suites; the deep drain paths are covered
here at the helper level where they can be driven deterministically.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# --------------------------------------------------------------------------
# Calibration composite-leak: pre_penalty_composite
# --------------------------------------------------------------------------

class TestConfidencePrePenalty:
    def _scorer(self):
        from ghost_agent.core.confidence import CompositeConfidence
        return CompositeConfidence(w_entropy=0.5, w_competence=0.5,
                                   threshold=0.89, lambda_uncertainty=0.0)

    def test_pre_penalty_excludes_outcome_penalty(self):
        c = self._scorer()
        r = c.score(normalised_entropy=0.5, competence_p_success=0.95,
                    n_observations=50, outcome_penalty=0.8)
        # The penalized composite is pulled way down (below threshold);
        # the pre-penalty prediction is NOT — that's what calibration must
        # record (a prediction can't know its own outcome).
        assert r.composite < r.pre_penalty_composite
        assert r.below_threshold is True          # decision uses penalized
        assert r.pre_penalty_composite > 0.5      # honest prediction stays high

    def test_no_penalty_pre_equals_composite(self):
        c = self._scorer()
        r = c.score(normalised_entropy=0.3, competence_p_success=0.7,
                    n_observations=20, outcome_penalty=0.0)
        assert abs(r.pre_penalty_composite - r.composite) < 1e-9

    def test_positional_reading_defaults_pre_to_composite(self):
        from ghost_agent.core.confidence import ConfidenceReading
        r = ConfidenceReading(0.7, 0.5, 0.6, 0.89, False)
        assert r.pre_penalty_composite == 0.7


# --------------------------------------------------------------------------
# coding_executor anchored classifiers
# --------------------------------------------------------------------------

class TestOpOkAnchored:
    def test_failed_replace_on_success_named_file_is_not_ok(self):
        from ghost_agent.core.coding_executor import _op_ok
        out = ("SYSTEM INSTRUCTION: The search block was NOT found in "
               "'payment_success.html'. Your remembered old_text does not "
               "byte-match the current file.")
        assert _op_ok(out) is False

    def test_real_success_is_ok(self):
        from ghost_agent.core.coding_executor import _op_ok
        assert _op_ok("SUCCESS: Applied 2 SEARCH/REPLACE blocks to 'x.py'.")
        assert _op_ok("SUCCESS: Wrote 40 chars to 'login_success.js'.")

    def test_rejected_is_not_ok(self):
        from ghost_agent.core.coding_executor import _op_ok
        assert _op_ok("REJECTED: that replace would have written markers") is False

    def test_write_error_security_is_anchored(self):
        from ghost_agent.core.coding_executor import _looks_like_write_error
        # A SUCCESS write to a file whose NAME contains the words is not an error.
        assert _looks_like_write_error(
            "SUCCESS: Wrote 12 chars to 'security error handler.py'.") is False
        # A real security-error head IS an error.
        assert _looks_like_write_error(
            "Security Error: Path '../x' attempts to access outside sandbox.")


# --------------------------------------------------------------------------
# workspace_cleanup data-deliverable recovery
# --------------------------------------------------------------------------

class TestDataDeliverableRecovery:
    def test_classifier(self):
        from ghost_agent.core.workspace_cleanup import (
            _is_data_deliverable, _is_source_like)
        assert _is_data_deliverable("model.pt")
        assert _is_data_deliverable("data/app.db")
        assert _is_data_deliverable("weights.safetensors")
        assert _is_data_deliverable("train.parquet")
        assert not _is_data_deliverable("screenshot.png")   # media scratch
        assert not _is_data_deliverable("render.mp4")
        # source-like stays source-like
        assert _is_source_like("app.py")

    def test_partial_recovery_rescues_unregistered_db(self, tmp_path):
        from ghost_agent.core import workspace_cleanup as wc
        store = MagicMock()
        store.list_tasks.return_value = []
        root = tmp_path / "proj"
        (root / "sub").mkdir(parents=True)
        (root / "app.py").write_text("x")
        (root / "data.db").write_bytes(b"\x00sqlite")
        (root / "shot.png").write_bytes(b"\x89PNG")
        # app.py registered → keep-set is partial.
        found = wc._recover_unregistered_sources(
            store, "pid", root, keep={"app.py"}, dry_run=True)
        assert "data.db" in found          # binary deliverable rescued
        assert "shot.png" not in found     # media scratch stays deletable


# (format_search_results is a nested function inside the search tool and is
# not importable; the `.get(k) or default` None-trap fix is a one-line
# change covered by reading + the existing search suite.)


# --------------------------------------------------------------------------
# inter-project dependency deadlock (real advance_once)
# --------------------------------------------------------------------------

class TestInterProjectDepGate:
    def _ctx(self, tmp_path):
        from types import SimpleNamespace
        from ghost_agent.memory.projects import ProjectStore
        from ghost_agent.memory.scratchpad import Scratchpad
        store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
        ctx = SimpleNamespace(project_store=store,
                              scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                              graph_memory=None, current_project_id=None)
        return ctx, store

    async def _dep_result(self, tmp_path, dep_status_or_delete):
        from ghost_agent.core.project_advancer import advance_once
        ctx, store = self._ctx(tmp_path)
        dep = store.create_project("Dep")
        store.add_task(dep, "dep task")
        a = store.create_project("A")
        store.add_task(a, "a task")
        store.update_project(a, metadata={"depends_on_projects": [dep]})
        if dep_status_or_delete == "__delete__":
            store.delete_project(dep, hard=True)
        else:
            store.update_project(dep, status=dep_status_or_delete)
        return await advance_once(ctx, a, tool_runner=None)

    @pytest.mark.asyncio
    async def test_deleted_dep_does_not_block_on_dependency(self, tmp_path):
        # The dep gate clears (it may still block LATER for an unrelated
        # reason like "no tool_runner"); the point is the block reason must
        # NOT be the stale dependency.
        res = await self._dep_result(tmp_path, "__delete__")
        assert "dependency" not in (res.summary or "").lower()

    @pytest.mark.asyncio
    async def test_archived_dep_does_not_block_on_dependency(self, tmp_path):
        res = await self._dep_result(tmp_path, "ARCHIVED")
        assert "dependency" not in (res.summary or "").lower()

    @pytest.mark.asyncio
    async def test_paused_dep_still_blocks(self, tmp_path):
        res = await self._dep_result(tmp_path, "PAUSED")
        assert res.classification == "blocked"
        assert "dependency" in (res.summary or "").lower()


# --------------------------------------------------------------------------
# Streamed-drain snapshot: work_log concurrency + calibration helper
# --------------------------------------------------------------------------

def _agent():
    from ghost_agent.core.agent import GhostAgent, GhostContext
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.7
    ctx.args.max_context = 8000
    ctx.args.smart_memory = 0.5
    ctx.args.use_planning = False
    ctx.args.model = "Qwen-Test"
    ctx.llm_client = MagicMock()
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_context_string.return_value = ""
    ctx.memory_system = MagicMock()
    ctx.cached_sandbox_state = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.verifier = None
    return GhostAgent(ctx)


class TestWorkLogSnapshotConcurrency:
    @pytest.mark.asyncio
    async def test_snapshot_files_project_under_snapshot_pid_not_live(self):
        """The streamed drain runs after the semaphore is released. If turn
        B re-points current_project_id, the drain must still file A's work
        under A's SNAPSHOT pid, and must NOT clear B's live accumulators."""
        agent = _agent()
        store = MagicMock()
        store.get_project.return_value = {"title": "Project A build calendar"}
        store.list_tasks.return_value = []
        agent.context.project_store = store
        # A's snapshot, tagged with A's req_id.
        agent.context._project_work_pending = ("reqA", {
            "pid": "projA", "files": {"projA/app.py"}, "tools": {"execute": 1},
            "failed": {}, "cmds": [], "failure_texts": [],
        })
        # B has re-pointed the live context.
        agent.context.current_project_id = "projB"
        agent.context._project_work_files = {"projB/other.py"}
        agent.context._project_work_tools = {"browser": 1}
        agent.context._project_work_failed_tools = {}
        agent.context._project_work_cmds = []
        agent.context._turn_failure_texts = []

        await agent._write_project_work_log_safe(
            last_user_content="build the calendar",
            final_ai_content="done", execution_failure_count=0,
            verifier_backfill=None, req_id="reqA")

        # Filed under A's project, with A's files.
        assert store.add_work_log.call_count == 1
        args, kwargs = store.add_work_log.call_args
        assert args[0] == "projA"
        assert "projA/app.py" in kwargs["files"]
        # B's live accumulators were NOT cleared by A's drain.
        assert agent.context._project_work_files == {"projB/other.py"}
        # A's pending snapshot was consumed.
        assert agent.context._project_work_pending is None

    @pytest.mark.asyncio
    async def test_foreign_reqid_snapshot_not_consumed(self):
        agent = _agent()
        store = MagicMock()
        agent.context.project_store = store
        agent.context._project_work_pending = ("reqA", {"pid": "projA",
            "files": {"projA/x"}, "tools": {}, "failed": {}, "cmds": [],
            "failure_texts": []})
        # A different request's drain must not consume A's snapshot.
        await agent._write_project_work_log_safe(
            last_user_content="q", final_ai_content="a",
            execution_failure_count=0, verifier_backfill=None, req_id="reqB")
        assert agent.context._project_work_pending == (
            "reqA", agent.context._project_work_pending[1])


# --------------------------------------------------------------------------
# Deferred LOWs (2026-07-26c)
# --------------------------------------------------------------------------

class TestExecuteRerunGuard:
    def test_compound_with_mutating_prefix_is_unsafe(self):
        from ghost_agent.tools.execute import _rerun_unsafe
        assert _rerun_unsafe("mkdir out && cat out/x.txt") is True
        assert _rerun_unsafe("git clone https://x && cd x") is True
        assert _rerun_unsafe("echo hi > f.txt && cat g.txt") is True

    def test_safe_compounds_and_simple_are_ok(self):
        from ghost_agent.tools.execute import _rerun_unsafe
        assert _rerun_unsafe("cd /workspace && python3 game.py") is False
        assert _rerun_unsafe("pwd && ls && cat x.py") is False
        assert _rerun_unsafe("python3 game.py") is False  # simple


class TestDepsCycle:
    def _plan(self, tmp_path):
        from ghost_agent.memory.projects import ProjectStore
        from ghost_agent.core.planning import ProjectPlan
        store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
        pid = store.create_project("P")
        return ProjectPlan(store, pid)

    def test_dep_cycle_does_not_starve_plan(self, tmp_path):
        plan = self._plan(tmp_path)
        a = plan.add_task("task A")
        b = plan.add_task("task B")
        plan.tree.nodes[a].depends_on = [b]
        plan.tree.nodes[b].depends_on = [a]   # cycle
        # Before the fix: both permanently ineligible → next_ready_leaf None
        # → advance_many falsely reports "project_done". Now it drains.
        leaf = plan.next_ready_leaf()
        assert leaf is not None
        assert leaf.id in (a, b)

    def test_normal_dep_still_blocks(self, tmp_path):
        plan = self._plan(tmp_path)
        a = plan.add_task("task A")
        b = plan.add_task("task B")
        plan.tree.nodes[b].depends_on = [a]   # B waits on A (not done)
        leaf = plan.next_ready_leaf()
        assert leaf is not None and leaf.id == a   # A ready, B blocked


class TestIdlessToolCallPairing:
    def test_two_idless_same_name_calls_both_get_results(self):
        from ghost_agent.core.agent import GhostAgent
        # Reconstruct a message list with two id-less `execute` calls and
        # two results; both must pair (neither result dropped).
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "execute", "arguments": '{"command": "a"}'}},
                {"function": {"name": "execute", "arguments": '{"command": "b"}'}},
            ]},
            {"role": "tool", "name": "execute", "content": "result-A"},
            {"role": "tool", "name": "execute", "content": "result-B"},
        ]
        agent = _agent()
        tcs = agent._reconstruct_tool_calls(msgs)
        results = [t.result for t in tcs if getattr(t, "result", None)]
        assert "result-A" in results and "result-B" in results
        assert len([t for t in tcs if t.name == "execute"]) == 2


class TestSearchCacheKeyNormalization:
    def test_trivial_variants_share_a_key(self):
        from ghost_agent.tools.search import _norm_cache_key
        a = _norm_cache_key("python asyncio")
        assert _norm_cache_key("Python  asyncio") == a
        assert _norm_cache_key("  python asyncio?  ") == a
        assert _norm_cache_key("python asyncio.") == a

    def test_distinct_queries_differ(self):
        from ghost_agent.tools.search import _norm_cache_key
        assert _norm_cache_key("python asyncio") != _norm_cache_key("python threading")
