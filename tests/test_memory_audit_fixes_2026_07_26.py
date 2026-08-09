"""Memory-substrate audit fixes (2026-07-26 six-agent scan).

Covers the splitter separator-drop corruption and the learning-loop
attribution cluster:

* ``recursive_split_text`` dropped whitespace separators on rejoin (every
  ingested chunk had words fused across line breaks — live: all 7,131
  postgresql-manual chunks) and honored ``chunk_overlap`` only on the
  no-separator hard split.
* ``get_playbook_context`` never reset ``last_playbook_triggers`` on empty
  branches, so outcome arms booked turns against stale/foreign lesson sets;
  the dream sim and the read-only façade stamped the production attribute
  mid-turn.
* ``ReadOnlySkillMemory`` leaked ``quarantine_lesson`` and retrieval-stat
  bumps to delegated sub-agents.
* ``record_helpful_retrieval`` (async judge) re-credited retrievals already
  credited by the inline ``credit_recent_retrievals`` pass.
* PASSED late verdicts returned before the lesson-outcome stash drained
  (FAILED never did), biasing the arms toward failure.
* Bus-tier lessons were invisible to the outcome arms.
* Belief-revision / project-miss probes phantom-bumped retrieval stats.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict

import pytest
from unittest.mock import MagicMock, AsyncMock

from ghost_agent.utils.helpers import recursive_split_text
from ghost_agent.core.agent import GhostAgent, GhostContext


# --------------------------------------------------------------------------
# recursive_split_text
# --------------------------------------------------------------------------

class TestSplitterSeparatorPreservation:
    def test_newline_separator_words_do_not_fuse(self):
        text = "\n".join(f"line{i} tail{i}" for i in range(60))
        chunks = recursive_split_text(text, chunk_size=120, chunk_overlap=0)
        for c in chunks:
            for i in range(60):
                assert f"tail{i}line" not in c.replace(" ", "")

    def test_all_source_words_survive(self):
        text = "\n\n".join(f"para{i} alpha{i} omega{i}" for i in range(40))
        chunks = recursive_split_text(text, chunk_size=100, chunk_overlap=0)
        out_words = set(w for c in chunks for w in c.split())
        assert set(text.split()) <= out_words

    def test_overlap_applies_on_separator_path(self):
        text = "\n".join(f"line{i:03d} content words here" for i in range(50))
        chunks = recursive_split_text(text, chunk_size=150, chunk_overlap=40)
        assert len(chunks) > 2
        overlapping = sum(
            1 for a, b in zip(chunks, chunks[1:]) if b[:12] and b[:12] in a)
        assert overlapping >= len(chunks) // 2

    def test_zero_overlap_still_valid(self):
        text = ". ".join(f"sentence {i}" for i in range(80))
        chunks = recursive_split_text(text, chunk_size=90, chunk_overlap=0)
        assert all(len(c) <= 90 for c in chunks)

    def test_pathological_input_terminates_bounded(self):
        text = ("y" * 700) + "\n" + ("z" * 700)
        chunks = recursive_split_text(text, chunk_size=100, chunk_overlap=20)
        assert chunks and all(len(c) <= 100 for c in chunks)
        joined = "".join(chunks)
        assert joined.count("y") >= 700 and joined.count("z") >= 700

    def test_empty_and_short(self):
        assert recursive_split_text("", 100, 10) == []
        assert recursive_split_text("abc", 100, 10) == ["abc"]


# --------------------------------------------------------------------------
# last_playbook_triggers hygiene (SkillMemory)
# --------------------------------------------------------------------------

@pytest.fixture
def skill_memory(tmp_path):
    from ghost_agent.memory.skills import SkillMemory
    d = tmp_path / "mem"
    d.mkdir()
    return SkillMemory(d)


class TestTriggerStampHygiene:
    def test_empty_retrieval_resets_stale_triggers(self, skill_memory, monkeypatch):
        skill_memory.last_playbook_triggers = ["stale-1", "stale-2"]
        monkeypatch.setattr(
            skill_memory, "_playbook_items_and_branch",
            lambda *a, **k: ([], "vector_empty"))
        out = skill_memory.get_playbook_context("hello", memory_system=MagicMock())
        assert out == ""
        assert skill_memory.last_playbook_triggers == []

    def test_nonempty_retrieval_stamps_fresh(self, skill_memory, monkeypatch):
        skill_memory.last_playbook_triggers = ["stale"]
        items = [{"text": "1. x", "trigger": "fresh-trigger"}]
        monkeypatch.setattr(
            skill_memory, "_playbook_items_and_branch",
            lambda *a, **k: (items, "vector"))
        monkeypatch.setattr(
            skill_memory, "_filter_quarantined", lambda it: it)
        skill_memory.get_playbook_context("q", memory_system=MagicMock(),
                                          record_retrievals=False)
        assert skill_memory.last_playbook_triggers == ["fresh-trigger"]

    def test_stamp_triggers_false_uses_sim_channel(self, skill_memory, monkeypatch):
        skill_memory.last_playbook_triggers = ["operator-turn"]
        items = [{"text": "1. x", "trigger": "sim-trigger"}]
        monkeypatch.setattr(
            skill_memory, "_playbook_items_and_branch",
            lambda *a, **k: (items, "vector"))
        monkeypatch.setattr(
            skill_memory, "_filter_quarantined", lambda it: it)
        skill_memory.get_playbook_context(
            "q", memory_system=MagicMock(),
            record_retrievals=False, stamp_triggers=False)
        # Operator record untouched; sim channel carries the sim's set.
        assert skill_memory.last_playbook_triggers == ["operator-turn"]
        assert skill_memory.last_sim_triggers == ["sim-trigger"]


# --------------------------------------------------------------------------
# ReadOnlySkillMemory façade
# --------------------------------------------------------------------------

class TestReadOnlySkillFacade:
    def test_quarantine_lesson_blocked(self):
        from ghost_agent.memory.readonly import ReadOnlySkillMemory
        real = MagicMock()
        ro = ReadOnlySkillMemory(real)
        ro.quarantine_lesson("some trigger")
        real.quarantine_lesson.assert_not_called()

    def test_get_playbook_context_forces_pure_read(self):
        from ghost_agent.memory.readonly import ReadOnlySkillMemory
        real = MagicMock()
        real.get_playbook_context.return_value = "## LESSONS"
        ro = ReadOnlySkillMemory(real)
        out = ro.get_playbook_context("query", memory_system=None)
        assert out == "## LESSONS"
        kwargs = real.get_playbook_context.call_args.kwargs
        assert kwargs["record_retrievals"] is False
        assert kwargs["stamp_triggers"] is False


# --------------------------------------------------------------------------
# helpful-retrieval credit idempotency
# --------------------------------------------------------------------------

class TestHelpfulCreditIdempotency:
    def _lesson(self, skill_memory, trigger="deploy fails on port bind"):
        skill_memory.learn_lesson(
            trigger, "killed wrong pid", "use lsof -ti :8000")
        return trigger

    def _helpful(self, skill_memory, trigger):
        pb = skill_memory._load_playbook()
        for p in pb:
            if (p.get("trigger") or "").lower() == trigger.lower():
                return int(p.get("helpful_retrievals") or 0)
        return None

    def test_judge_after_inline_credit_no_double(self, skill_memory):
        trig = self._lesson(skill_memory)
        skill_memory.record_retrieval(trig)
        # Inline pass credits first (legacy credit-everything form).
        assert skill_memory.credit_recent_retrievals(300) == 1
        assert self._helpful(skill_memory, trig) == 1
        # Async judge lands seconds later — must be a no-op now.
        assert skill_memory.record_helpful_retrieval(trig) is False
        assert self._helpful(skill_memory, trig) == 1

    def test_new_retrieval_reopens_credit(self, skill_memory):
        trig = self._lesson(skill_memory)
        skill_memory.record_retrieval(trig)
        assert skill_memory.record_helpful_retrieval(trig) is True
        assert self._helpful(skill_memory, trig) == 1
        # A NEW retrieval re-arms the credit.
        skill_memory.record_retrieval(trig)
        assert skill_memory.record_helpful_retrieval(trig) is True
        assert self._helpful(skill_memory, trig) == 2


# --------------------------------------------------------------------------
# Agent-level: flush ordering + bus-tier trigger merge + probe purity
# --------------------------------------------------------------------------

@pytest.fixture
def agent():
    # Mirrors the harness in test_edit_churn_and_refute_followups.py.
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.7
    ctx.args.max_context = 8000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    ctx.args.model = "Qwen-Test"
    ctx.llm_client = MagicMock()
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_context_string.return_value = ""
    ctx.memory_system = MagicMock()
    ctx.memory_system.search = MagicMock(return_value="")
    ctx.cached_sandbox_state = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.verifier = None
    return GhostAgent(ctx)


class TestPassedFlushOrdering:
    @pytest.mark.asyncio
    async def test_passed_verdict_drains_stash_even_without_cached_traj(self, agent):
        """A CONFIRMED (PASSED) late verdict whose trajectory fell out of the
        correction cache used to return before the stash drained — the
        success tick was silently lost while FAILED always got through."""
        agent.context.trajectory_collector = MagicMock()
        agent.context._recent_trajectories_for_correction = {}  # evicted
        agent.context._surfaced_triggers_by_traj = OrderedDict(
            {"traj-1": ["lesson-a", "lesson-b"]})
        rec = MagicMock()
        agent.context.skill_memory.record_surfaced_outcomes = rec

        agent._backfill_trajectory_outcome("traj-1", "passed")
        # The flush schedules record_surfaced_outcomes via spawn_bg +
        # to_thread; give the loop a tick to run it.
        for _ in range(20):
            if rec.call_count:
                break
            await asyncio.sleep(0.02)

        rec.assert_called_once_with(["lesson-a", "lesson-b"], True)
        assert "traj-1" not in agent.context._surfaced_triggers_by_traj


class TestSurfacedTriggerMerge:
    def test_merges_playbook_and_bus_skill_tier(self, agent):
        sm = MagicMock()
        sm.last_playbook_triggers = ["pb-1", "shared"]
        sm._playbook_turn_key = "T1"          # §4L R3 turn-key guards
        sm.last_bus_triggers = []
        sm._bus_delivered_turn_key = ""
        agent.context.memory_bus = MagicMock()
        agent.context.memory_bus.last_hydration = {
            "turn_id": "T1",
            "survivors": [
                {"source": "skill", "trigger": "bus-1"},
                {"source": "skill", "trigger": "shared"},
                {"source": "vector", "trigger": "not-a-lesson"},
            ],
        }
        got = agent._surfaced_lesson_triggers(sm, turn_id="T1")
        assert got == ["pb-1", "shared", "bus-1"]

    def test_foreign_turn_stash_excluded(self, agent):
        sm = MagicMock()
        sm.last_playbook_triggers = ["pb-1"]
        sm._playbook_turn_key = "T1"
        sm.last_bus_triggers = []
        sm._bus_delivered_turn_key = ""
        agent.context.memory_bus = MagicMock()
        agent.context.memory_bus.last_hydration = {
            "turn_id": "OTHER",
            "survivors": [{"source": "skill", "trigger": "bus-1"}],
        }
        assert agent._surfaced_lesson_triggers(sm, turn_id="T1") == ["pb-1"]

    def test_no_bus_no_crash(self, agent):
        sm = MagicMock()
        sm.last_playbook_triggers = ["pb-1"]
        sm._playbook_turn_key = "T1"
        sm.last_bus_triggers = []
        sm._bus_delivered_turn_key = ""
        agent.context.memory_bus = None
        assert agent._surfaced_lesson_triggers(sm, turn_id="T1") == ["pb-1"]


class TestProbePurity:
    @pytest.mark.asyncio
    async def test_project_miss_probe_does_not_bump_stats(self):
        from ghost_agent.tools.projects import _not_found_with_recall
        ctx = MagicMock()
        ctx.memory_system.search_advanced = MagicMock(return_value=[])
        await _not_found_with_recall(ctx, "ghost-project")
        kwargs = ctx.memory_system.search_advanced.call_args.kwargs
        assert kwargs.get("record_retrievals") is False


# --------------------------------------------------------------------------
# Vector-integrity cluster
# --------------------------------------------------------------------------

class TestBusIdentitySmartUpdate:
    @pytest.mark.asyncio
    async def test_identity_fact_routes_through_smart_update(self):
        from ghost_agent.core.bus import MemoryBus
        vec = MagicMock()
        bus = MemoryBus(vector_memory=vec)
        await bus.publish_fact("update_profile", {
            "text": "User car is BMW",
            "metadata": {"timestamp": "2026-07-26T00:00:00Z", "type": "identity"},
        })
        vec.smart_update.assert_called_once_with("User car is BMW", "identity")
        vec.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_identity_fact_still_plain_add(self):
        from ghost_agent.core.bus import MemoryBus
        vec = MagicMock()
        bus = MemoryBus(vector_memory=vec)
        await bus.publish_fact("insert_fact", {
            "text": "The moon is far away",
            "metadata": {"timestamp": "2026-07-26T00:00:00Z", "type": "manual"},
        })
        vec.add.assert_called_once()
        vec.smart_update.assert_not_called()


class TestForgetSweepScoping:
    @pytest.mark.asyncio
    async def test_semantic_sweep_query_is_type_scoped(self):
        from ghost_agent.tools.memory import tool_unified_forget, _FORGET_PROTECTED_TYPES
        ms = MagicMock()
        ms.collection.query.return_value = {"ids": [[]], "distances": [[]],
                                            "documents": [[]], "metadatas": [[]]}
        ms.search_advanced.return_value = []
        await tool_unified_forget(target="wal", memory_system=ms)
        assert ms.collection.query.called
        for call in ms.collection.query.call_args_list:
            where = call.kwargs.get("where")
            assert where == {"type": {"$nin": _FORGET_PROTECTED_TYPES}}
        from chromadb.api.types import validate_where
        validate_where({"type": {"$nin": _FORGET_PROTECTED_TYPES}})

    @pytest.mark.asyncio
    async def test_literal_match_on_protected_type_not_deleted(self):
        """Belt-and-braces: even if the where scope were bypassed, a
        document-typed row that literally mentions the target must survive."""
        from ghost_agent.tools.memory import tool_unified_forget
        ms = MagicMock()
        ms.collection.query.return_value = {
            "ids": [["doc-1"]],
            "distances": [[0.2]],
            "documents": [["chapter about wal_level tuning"]],
            "metadatas": [[{"type": "document"}]],
        }
        ms.search_advanced.return_value = []
        await tool_unified_forget(target="wal", memory_system=ms)
        deleted_ids = [c.kwargs.get("ids") or (c.args[0] if c.args else None)
                       for c in ms.collection.delete.call_args_list]
        assert ["doc-1"] not in deleted_ids


class TestIdentityProbeGate:
    def _vm_with_selection(self, selection):
        from ghost_agent.memory.vector import VectorMemory
        vm = VectorMemory.__new__(VectorMemory)
        vm._search_selection = lambda q, inject_identity=True: selection
        return vm

    def test_probe_distances_do_not_defeat_gate(self):
        vm = self._vm_with_selection([
            {"dist": 0.2, "from_identity_probe": True, "mem_id": "a",
             "combined_score": 0.0, "meta": {}, "doc": "x", "p_score": 1},
            {"dist": 0.55, "from_identity_probe": False, "mem_id": "b",
             "combined_score": 0.0, "meta": {}, "doc": "y", "p_score": 1},
        ])
        # Best QUERY-batch dist is 0.55 > 0.42 → off-topic → nothing injected.
        assert vm.search_items("should i restart", min_relevance_dist=0.42) == []

    def test_on_topic_query_batch_passes_gate(self):
        vm = self._vm_with_selection([
            {"dist": 0.30, "from_identity_probe": False, "mem_id": "b",
             "combined_score": 0.0, "meta": {"type": "auto", "timestamp": "?"},
             "doc": "y", "p_score": 1},
        ])
        out = vm.search_items("restart procedure", min_relevance_dist=0.42)
        assert len(out) == 1


class TestBusPartialReporting:
    @pytest.mark.asyncio
    async def test_update_profile_surfaces_bus_failures(self):
        from ghost_agent.tools.memory import tool_update_profile
        bus = MagicMock()
        bus.publish_fact = AsyncMock(return_value={
            "vector": "error: store down", "profile": "ok",
            "graph": "ok", "skill": "skip"})
        out = await tool_update_profile(
            category="identity", key="car", value="BMW", memory_bus=bus)
        assert out.startswith("PARTIAL")
        assert "vector" in out

    @pytest.mark.asyncio
    async def test_update_profile_clean_report_is_success(self):
        from ghost_agent.tools.memory import tool_update_profile
        bus = MagicMock()
        bus.publish_fact = AsyncMock(return_value={
            "vector": "ok", "profile": "ok", "graph": "ok", "skill": "skip"})
        out = await tool_update_profile(
            category="identity", key="car", value="BMW", memory_bus=bus)
        assert out.startswith("SUCCESS")


# --------------------------------------------------------------------------
# Graph/episodes cluster
# --------------------------------------------------------------------------

@pytest.fixture
def graph_memory(tmp_path):
    from ghost_agent.memory.graph import GraphMemory
    return GraphMemory(tmp_path)


class TestGraphTripletRobustness:
    def test_malformed_triplet_does_not_abort_batch(self, graph_memory):
        """One list-shaped triplet used to raise AttributeError out of the
        whole batch — at the smart-memory call site that also lost the
        turn's fact embed and profile write (graph runs first)."""
        added = graph_memory.add_triplets([
            ["user", "REL", "thing"],           # list-shaped
            "not even a container",              # string
            {"subject": "xhost", "predicate": "REL", "object": "yservice"},
        ])
        assert added == 1
        import sqlite3
        with sqlite3.connect(graph_memory.db_path) as conn:
            rows = conn.execute(
                "SELECT subject, object FROM triplets WHERE subject='xhost'"
            ).fetchall()
        assert rows == [("xhost", "yservice")]

    def test_relation_key_accepted_as_predicate(self, graph_memory):
        added = graph_memory.add_triplets([
            {"subject": "a", "relation": "OWNS_KIND_OF", "object": "b"},
        ])
        assert added == 1

    def test_generic_subject_skips_functional_expiry(self, graph_memory):
        """`project HAS_STATUS x` rows are per-project facts under an
        aggregate subject — expiring siblings would erase OTHER projects'
        statuses. Specific subjects keep the expiry behavior."""
        import sqlite3
        graph_memory.add_triplets([
            {"subject": "project", "predicate": "HAS_STATUS", "object": "done"}])
        graph_memory.add_triplets([
            {"subject": "project", "predicate": "HAS_STATUS", "object": "active"}])
        graph_memory.add_triplets([
            {"subject": "chess-v9", "predicate": "HAS_STATUS", "object": "dead"}])
        graph_memory.add_triplets([
            {"subject": "chess-v9", "predicate": "HAS_STATUS", "object": "running"}])
        with sqlite3.connect(graph_memory.db_path) as conn:
            generic_live = conn.execute(
                "SELECT COUNT(*) FROM triplets WHERE subject='project' "
                "AND predicate='HAS_STATUS' AND valid_until IS NULL").fetchone()[0]
            specific_live = conn.execute(
                "SELECT object FROM triplets WHERE subject='chess-v9' "
                "AND predicate='HAS_STATUS' AND valid_until IS NULL").fetchall()
        assert generic_live == 2          # both project statuses coexist
        assert specific_live == [("running",)]  # old pid/status superseded


class TestEpisodeEvictionPreference:
    def test_eviction_prefers_spent_over_pending(self, tmp_path, monkeypatch):
        from ghost_agent.memory.episodes import EpisodicMemory
        em = EpisodicMemory(tmp_path)
        monkeypatch.setattr(EpisodicMemory, "MAX_EPISODES", 4)
        import sqlite3
        # 3 consolidated (spent) + 1 pending, oldest first.
        ids = [em.record_episode(f"trigger {i}", outcome="ok", success=True)
               for i in range(4)]
        with sqlite3.connect(em.db_path) as conn:
            conn.execute(
                f"UPDATE episodes SET consolidated = 1 WHERE id IN "
                f"({ids[0]}, {ids[1]}, {ids[2]})")
            conn.commit()
        # 5th insert overflows the cap: the OLDEST SPENT row must go,
        # never the pending one (id 4).
        em.record_episode("trigger 4 fresh", outcome="ok", success=True)
        with sqlite3.connect(em.db_path) as conn:
            remaining = {r[0] for r in conn.execute(
                "SELECT id FROM episodes").fetchall()}
        assert ids[0] not in remaining          # oldest consolidated evicted
        assert ids[3] in remaining              # pending survives


class TestFrontierDuplicateStats:
    def test_duplicate_reroll_does_not_inflate_template_runs(self, tmp_path):
        from ghost_agent.memory.frontier import FrontierTracker
        ft = FrontierTracker(tmp_path)
        kw = dict(cluster_key="sql|execute", challenge="SELECT the same thing",
                  attempts_used=1, passed=True, description_length=40,
                  template_key="sql.groupby")
        first = ft.record_run(**kw)
        second = ft.record_run(**kw)  # byte-identical re-roll
        assert not first.get("duplicate")
        assert second.get("duplicate") is True
        state = ft._load()
        tstats = state["clusters"]["sql|execute"]["templates"]["sql.groupby"]
        assert tstats["runs"] == 1


# --------------------------------------------------------------------------
# Skill-store cluster
# --------------------------------------------------------------------------

class TestComposedMintSafety:
    def _registry(self, tmp_path):
        from ghost_agent.tools.composed_skills import ComposedSkillRegistry
        return ComposedSkillRegistry(storage_dir=tmp_path / "composed")

    def test_compile_from_pattern_sanitizes_dotted_names(self, tmp_path):
        reg = self._registry(tmp_path)
        sk = reg.compile_from_pattern(
            "auto.generic.manage_services_manage_services.c73e69",
            [{"tool": "workspace", "description": "", "params": {}}],
            "desc")
        assert "." not in sk.name
        import re
        assert re.match(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$", sk.name)

    def test_reminting_does_not_demote_approved_macro(self, tmp_path):
        reg = self._registry(tmp_path)
        sk = reg.compile_from_pattern(
            "auto_seq_x",
            [{"tool": "workspace", "description": "", "params": {}}],
            "first mint")
        sk.status = "active"
        sk.usage_count = 7
        reg.save()
        again = reg.compile_from_pattern(
            "auto_seq_x",
            [{"tool": "workspace", "description": "", "params": {}}],
            "re-mint after re-graduation")
        assert again is reg.skills["auto_seq_x"]
        assert reg.skills["auto_seq_x"].status == "active"
        assert reg.skills["auto_seq_x"].usage_count == 7


class TestAcquiredResultClassification:
    def test_infra_vs_fail_vs_ok(self):
        from ghost_agent.tools.registry import _acquired_skill_result_class
        assert _acquired_skill_result_class(
            "--- EXECUTION RESULT ---\nEXIT CODE: 0\nSTDOUT:\nfine") == "ok"
        assert _acquired_skill_result_class(
            "--- EXECUTION RESULT ---\nEXIT CODE: 1\nSTDOUT:\nboom") == "fail"
        assert _acquired_skill_result_class(
            "--- EXECUTION RESULT ---\nEXIT CODE: 124\nSTDOUT:") == "infra"
        assert _acquired_skill_result_class("[SYSTEM ERROR] sandbox down") == "infra"

    def test_echoed_banner_does_not_mask_failure(self):
        from ghost_agent.tools.registry import _acquired_skill_result_class
        # The skill's own stdout echoes a subprocess banner; the harness
        # banner (first) says exit 1 — must classify as fail.
        result = ("--- EXECUTION RESULT ---\nEXIT CODE: 1\nSTDOUT:\n"
                  "inner run said EXIT CODE: 0 but then crashed")
        assert _acquired_skill_result_class(result) == "fail"


class TestDegradedRecovery:
    def _mgr(self, tmp_path, with_memory=False):
        from ghost_agent.tools.acquired_skills import AcquiredSkillManager
        return AcquiredSkillManager(tmp_path, MagicMock() if with_memory else None)

    def test_success_clears_degraded(self, tmp_path):
        mgr = self._mgr(tmp_path)
        mgr.save_skill("flaky", "desc", {}, "code")
        for _ in range(3):
            mgr.log_telemetry("flaky", success=False)
        assert mgr.get_all_skills()["flaky"]["status"] == "degraded"
        mgr.log_telemetry("flaky", success=True)
        info = mgr.get_all_skills()["flaky"]
        assert info["status"] == "active"
        assert info["failure_count"] == 0

    def test_recreate_with_new_code_resets_status(self, tmp_path):
        mgr = self._mgr(tmp_path)
        mgr.save_skill("flaky", "desc", {}, "code v1")
        for _ in range(3):
            mgr.log_telemetry("flaky", success=False)
        assert mgr.get_all_skills()["flaky"]["status"] == "degraded"
        mgr.save_skill("flaky", "desc", {}, "code v2 rewritten")
        info = mgr.get_all_skills()["flaky"]
        assert info["status"] == "active"
        assert info["failure_count"] == 0

    def test_retire_catches_legacy_zombie(self, tmp_path):
        """Pre-fix zombie shape: status stuck 'degraded' while a success
        reset failure_count to 0 — retirement used to key only on the
        counter, so the entry was unretirable AND invisible."""
        mgr = self._mgr(tmp_path)
        mgr.save_skill("zombie", "desc", {}, "code")
        reg = mgr._load_registry()
        reg["zombie"]["status"] = "degraded"
        reg["zombie"]["failure_count"] = 0
        mgr._save_registry(reg)
        retired = mgr.retire_degraded_skills()
        assert retired == ["zombie"]


class TestGraduatedStoreLifecycle:
    def _cand(self, name="auto.x.y", sig="sig123", conf=0.6, support=4):
        c = MagicMock()
        c.signature_hash = sig
        c.name = name
        c.cluster = "x"
        c.tool_sequence = ["workspace", "introspect"]
        c.support = support
        c.confidence = conf
        c.trigger_examples = []
        c.exemplar_trajectory_id = ""
        return c

    def test_confidence_can_go_down(self, tmp_path):
        from ghost_agent.skills_auto.store import GraduatedSkillStore
        store = GraduatedSkillStore(tmp_path)
        store.graduate(self._cand(), confidence=0.9)
        store.graduate(self._cand(), confidence=0.4)
        assert store.all_skills()[0]["confidence"] == 0.4

    def test_remove_deletes_entry(self, tmp_path):
        from ghost_agent.skills_auto.store import GraduatedSkillStore
        store = GraduatedSkillStore(tmp_path)
        store.graduate(self._cand(), confidence=0.9)
        assert store.remove("sig123") is True
        assert store.count() == 0
        assert store.remove("sig123") is False


class TestMinerSupportAndParams:
    def _traj(self, tid, outcome, calls):
        from types import SimpleNamespace
        return SimpleNamespace(
            id=tid, outcome=outcome,
            tool_calls=[SimpleNamespace(name=n, arguments=a) for n, a in calls])

    def test_unknown_outcome_does_not_count_as_support(self):
        from ghost_agent.core.dream import mine_recurring_tool_sequences
        seq = [("workspace", {"action": "summary"}),
               ("introspect", {"action": "summary"})]
        trajs = [
            self._traj("t1", "passed", seq),
            self._traj("t2", "unknown", seq),
            self._traj("t3", "", seq),
        ]
        proposals = mine_recurring_tool_sequences(trajs, min_support=2)
        assert proposals == []  # only 1 PASSED trajectory → below support

    def test_mutating_tool_params_not_baked(self):
        from ghost_agent.core.dream import mine_recurring_tool_sequences
        seq = [("workspace", {"action": "summary"}),
               ("manage_projects", {"action": "task_update",
                                    "project_id": "f36f04",
                                    "task_id": "eed65d",
                                    "status": "DONE"})]
        trajs = [self._traj(f"t{i}", "passed", seq) for i in range(3)]
        proposals = mine_recurring_tool_sequences(trajs, min_support=2)
        assert proposals
        steps = proposals[0]["steps"]
        by_tool = {s["tool"]: s for s in steps}
        assert by_tool["workspace"]["params"] == {"action": "summary"}
        assert by_tool["manage_projects"]["params"] == {}


# --------------------------------------------------------------------------
# Small-stores cluster
# --------------------------------------------------------------------------

class TestJournalCorruptionHandling:
    def _journal(self, tmp_path):
        from ghost_agent.memory.journal import MemoryJournal
        return MemoryJournal(tmp_path)

    def test_undecodable_inflight_is_quarantined_not_emptied(self, tmp_path):
        j = self._journal(tmp_path)
        j.inflight_path.write_bytes(b"\xff\xfe garbage \x80")
        assert j._read_inflight() == []
        sidecars = list(tmp_path.glob("memory_journal.inflight.corrupt-*.json"))
        assert len(sidecars) == 1
        assert not j.inflight_path.exists()

    def test_undecodable_overflow_is_quarantined_not_emptied(self, tmp_path):
        j = self._journal(tmp_path)
        j.overflow_path.write_bytes(b"\xff\xfe spill \x80")
        assert j._read_overflow() == []
        sidecars = list(tmp_path.glob("memory_journal.overflow.corrupt-*.json"))
        assert len(sidecars) == 1

    def test_retries_mutation_does_not_defeat_recovery_dedup(self, tmp_path):
        j = self._journal(tmp_path)
        j.append("post_mortem", {"user": "x", "ai": "y"})
        staged_entry = j.load()[0]
        requeued = dict(staged_entry, retries=2)
        # Recovery folds a staged copy that gained a retries counter — the
        # canonical key must treat it as the SAME item.
        folded = j._prepend_overflow([requeued])
        assert folded == 0
        assert j.pending_count() == 1


class TestBeliefChangeMatching:
    def test_multiword_message_matches_related_entry(self, tmp_path):
        from ghost_agent.memory.contradiction_log import ContradictionLog
        cl = ContradictionLog(tmp_path)
        cl.record("User drives a BMW",
                  [{"id": "1", "text": "User drives a Nissan"}],
                  ["1"], reason="test")
        # "what car do I drive now?" reduces to ONE content token ({drive};
        # "car"/"now" are <4 chars), and §4R R2 raised the bar to two shared
        # tokens after measuring a 61% false-fire rate on 240 real user turns
        # — single-token queries were the hole. The test's INTENT (a multiword
        # message matches a related entry, vs the old whole-substring rule
        # that never matched) is preserved with a query that carries two.
        out = cl.explain_belief_change("does the user still drive a Nissan?")
        assert "BELIEF REVISION HISTORY" in out
        assert "BMW" in out

    def test_unrelated_message_matches_nothing(self, tmp_path):
        from ghost_agent.memory.contradiction_log import ContradictionLog
        cl = ContradictionLog(tmp_path)
        cl.record("User drives a BMW",
                  [{"id": "1", "text": "User drives a Nissan"}],
                  ["1"], reason="test")
        assert cl.explain_belief_change("weather forecast tomorrow?") == ""


class TestProfileFailClosed:
    def test_transient_read_error_blocks_writes(self, tmp_path, monkeypatch):
        from ghost_agent.memory.profile import ProfileMemory
        pm = ProfileMemory(tmp_path)
        pm.save({"root": {"name": "Vasilis"}, "relationships": {},
                 "interests": {}, "assets": {}})
        original = pm.file_path.read_text()

        real_read_text = type(pm.file_path).read_text

        def _sick(self_path, *a, **k):
            if self_path == pm.file_path:
                raise OSError(5, "I/O error")
            return real_read_text(self_path, *a, **k)

        monkeypatch.setattr(type(pm.file_path), "read_text", _sick)
        data = pm.load()
        assert data["root"]["name"] == "User"      # default served
        assert pm._degraded is True
        pm.save({"root": {"name": "CLOBBER"}})     # must be refused
        monkeypatch.undo()
        assert pm.file_path.read_text() == original  # intact on disk
        # A successful read clears the degradation.
        assert pm.load()["root"]["name"] == "Vasilis"
        assert pm._degraded is False


class TestScratchpadHardening:
    def test_sentinel_survives_lru_eviction(self, tmp_path):
        from ghost_agent.memory.scratchpad import Scratchpad
        sp = Scratchpad(max_entries=3, persist_path=tmp_path / "sp.db")
        sp.set("__current_project__", "pid-1", namespace=None)
        for i in range(5):
            sp.set(f"key{i}", f"v{i}", namespace=None)
        assert sp.get("__current_project__") == "pid-1"

    def test_export_restore_preserves_namespaces(self, tmp_path):
        from ghost_agent.memory.scratchpad import Scratchpad
        sp = Scratchpad(max_entries=10, persist_path=tmp_path / "sp.db")
        sp.set("global_key", "g", namespace=None)
        sp.set("proj_key", "p", namespace="proj-123")
        snap = sp.export_state()
        assert snap["proj_key"]["namespace"] == "proj-123"
        sp2 = Scratchpad(max_entries=10, persist_path=tmp_path / "sp2.db")
        sp2.restore_state(snap)
        assert sp2._scopes.get("proj_key") == "proj-123"
        assert sp2._scopes.get("global_key") is None

    def test_restore_legacy_flat_shape_lands_global(self, tmp_path):
        from ghost_agent.memory.scratchpad import Scratchpad
        sp = Scratchpad(max_entries=10, persist_path=tmp_path / "sp.db")
        sp.active_namespace = "active-proj"
        sp.restore_state({"old_key": "old_value"})
        assert sp._scopes.get("old_key") is None  # global, not active-proj


class TestAdaptiveThresholdEffectiveBar:
    def test_cleared_bar_uses_effective_threshold(self, tmp_path):
        from ghost_agent.memory.adaptive_threshold import AdaptiveThreshold
        at = AdaptiveThreshold(tmp_path)
        at.threshold = 0.6
        at.record(score=0.8, was_useful=False, effective_threshold=0.9)
        # score 0.8 did NOT clear the effective 0.9 bar.
        assert list(at.window)[-1][2] is False
        at.record(score=0.95, was_useful=False, effective_threshold=0.9)
        assert list(at.window)[-1][2] is True


# --------------------------------------------------------------------------
# Projects/work-log cluster
# --------------------------------------------------------------------------

@pytest.fixture
def project_store(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    return ProjectStore(tmp_path)


class TestProjectReopenAndEvents:
    def test_update_task_reopens_done_project(self, project_store):
        pid = project_store.create_project("Demo Site")
        tid = project_store.add_task(pid, "build the demo page")
        project_store.update_task(tid, status="DONE")
        assert (project_store.get_project(pid)["status"] or "").upper() == "DONE"
        # Reviving the EXISTING task must reopen the project (add_task
        # already did; this tuple was missing DONE).
        project_store.update_task(tid, status="PENDING")
        assert (project_store.get_project(pid)["status"] or "").upper() == "ACTIVE"

    def test_artifact_added_event_carries_path(self, project_store):
        pid = project_store.create_project("Files Project")
        tid = project_store.add_task(pid, "produce report")
        project_store.register_file_artifact(tid, "report.pdf", "the report")
        events = [e for e in project_store.list_events(pid, limit=20)
                  if e.get("type") == "artifact_added"]
        assert events
        payload = events[0].get("payload") or {}
        if isinstance(payload, str):
            import json as _j
            payload = _j.loads(payload)
        assert payload.get("path") == "report.pdf"

    def test_event_retention_caps_high_churn_types(self, project_store):
        pid = project_store.create_project("Churny")
        cap = project_store._EVENTS_RETAIN_PER_TYPE
        for i in range(cap + 40):
            project_store.log_event(pid, None, "task_updated", {"n": i})
        import sqlite3
        with sqlite3.connect(project_store.db_path) as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM project_events WHERE project_id=? "
                "AND type='task_updated'", (pid,)).fetchone()[0]
        assert n == cap

    def test_deleted_similar_requires_strong_overlap(self, project_store):
        pid = project_store.create_project("Chess Game")
        project_store.delete_project(pid, hard=True)
        # Single shared token ("chess") must NOT link.
        assert project_store.find_deleted_similar("Chess Tutorial") is None
        # Identical token set MUST link.
        assert project_store.find_deleted_similar("Chess Game") is not None


class TestWorkLogRelevanceGate:
    def test_off_topic_request_rejected(self, agent, project_store):
        pid = project_store.create_project("Jiu Jitsu Calendar")
        project_store.add_task(pid, "render the calendar grid")
        agent.context._project_work_cmds = []
        assert agent._request_relevant_to_project(
            project_store, pid, "get me the news please.") is False

    def test_on_topic_request_accepted(self, agent, project_store):
        pid = project_store.create_project("Jiu Jitsu Calendar")
        project_store.add_task(pid, "render the calendar grid")
        agent.context._project_work_cmds = []
        assert agent._request_relevant_to_project(
            project_store, pid, "fix the calendar rendering bug") is True

    def test_command_naming_project_dir_accepted(self, agent, project_store):
        pid = project_store.create_project("Opaque Title")
        agent.context._project_work_cmds = [f"ls projects/{pid}/src"]
        assert agent._request_relevant_to_project(
            project_store, pid, "totally unrelated words here") is True


# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# §4B residual: shared AcquiredSkillManager lock (concurrent telemetry)
# --------------------------------------------------------------------------

class TestSharedAcquiredManager:
    def test_get_shared_returns_same_instance(self, tmp_path):
        from ghost_agent.tools.acquired_skills import AcquiredSkillManager
        AcquiredSkillManager._SHARED.clear()
        a = AcquiredSkillManager.get_shared(tmp_path, None)
        b = AcquiredSkillManager.get_shared(tmp_path, None)
        assert a is b

    def test_get_shared_upgrades_none_memory_system(self, tmp_path):
        from ghost_agent.tools.acquired_skills import AcquiredSkillManager
        AcquiredSkillManager._SHARED.clear()
        a = AcquiredSkillManager.get_shared(tmp_path, None)
        assert a.memory_system is None
        ms = MagicMock()
        b = AcquiredSkillManager.get_shared(tmp_path, ms)
        assert b is a and a.memory_system is ms

    def test_concurrent_telemetry_loses_no_increments(self, tmp_path):
        """The per-call fresh-manager-per-RLock pattern lost failure_count
        increments under concurrency (each manager serialized only itself).
        A shared instance serializes all writers."""
        import threading
        from ghost_agent.tools.acquired_skills import AcquiredSkillManager
        AcquiredSkillManager._SHARED.clear()
        mgr = AcquiredSkillManager.get_shared(tmp_path, None)
        mgr.save_skill("busy", "d", {}, "code")

        N = 60
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            for _ in range(N):
                # A fresh get_shared each call (mirrors the real call sites)
                # must resolve to the same locked instance.
                AcquiredSkillManager.get_shared(tmp_path, None).log_telemetry(
                    "busy", success=False)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # failure_count is capped at 3 by degrade logic? No — it keeps
        # incrementing; each call adds 1. 8*N increments, none lost.
        assert mgr.get_all_skills()["busy"]["failure_count"] == 8 * N


# --------------------------------------------------------------------------
# #2: episode field population + competence wiring
# --------------------------------------------------------------------------

class TestEpisodeFieldPopulation:
    def _agent(self):
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

    @pytest.mark.asyncio
    async def test_cluster_and_context_populated(self):
        agent = self._agent()
        em = MagicMock()
        agent.context.episodic_memory = em
        agent.context._turn_failure_texts = []
        tools = [
            {"name": "execute", "content": "Error: boom"},
            {"name": "execute", "content": "EXIT CODE: 0\nok"},
        ]
        await agent._record_episode_safe("run the build", tools, "Done, build passed.")
        kwargs = em.record_episode.call_args.kwargs
        # execute → "shell" domain (via metacog._domain_for_tool).
        assert kwargs["cluster_id"] == "shell"
        assert "tools:" in kwargs["context"]
        # first execute failed, turn succeeded → recovery signal present.
        assert "recovered after failures" in kwargs["context"]
        assert kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_empty_tools_no_crash_empty_cluster(self):
        agent = self._agent()
        em = MagicMock()
        agent.context.episodic_memory = em
        agent.context._turn_failure_texts = []
        await agent._record_episode_safe("just chatting", [], "hello there")
        kwargs = em.record_episode.call_args.kwargs
        assert kwargs["cluster_id"] == ""
        assert "no tools" in kwargs["context"]


class TestCompetenceContextString:
    def test_renders_per_domain_rollup(self, tmp_path):
        from ghost_agent.memory.competence import CompetenceProfile
        cp = CompetenceProfile(tmp_path)
        for _ in range(8):
            cp.record("shell", "execute", True)
        for _ in range(3):
            cp.record("code", "file_system", False)
        s = cp.get_context_string()
        assert "Competence" in s
        assert "shell" in s and "code" in s

    def test_empty_profile_renders_nothing(self, tmp_path):
        from ghost_agent.memory.competence import CompetenceProfile
        cp = CompetenceProfile(tmp_path)
        assert cp.get_context_string() == ""
