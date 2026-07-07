"""Per-item fusion + deferred retrieval credit (IMPROVEMENTS.md #13/#14/#16).

Three coupled fixes to the MemoryBus read path (2026-07-07):

#13 — episodic semantic recall WIRED: `_fetch_episodic` now passes the bus's
     vector store to `search_similar`, activating the embedding-based episode
     retrieval path that the write side has been populating all along
     (without it, recall always fell back to substring matching over the 100
     most recent episodes).

#14 — per-item RRF: the skill and episodic tiers used to enter fusion as ONE
     monolithic blob each — always rank 1, immune to the per-item relevance
     floor and _PER_SOURCE_CAP, and duplicated per sub-query. They now emit
     one item per lesson / per episode.

#16 — deferred retrieval credit: fetchers no longer bump retrieval counters
     for every candidate of every sub-query (which inflated `retrievals` ~4x
     per turn and crushed hit_rate, mis-flagging good lessons as stale, at a
     cost of up to ~20 playbook rewrites per turn). Credit now happens ONCE
     per turn, post-fusion, only for items that actually entered the prompt:
     `VectorMemory.bump_retrievals(ids)` + `SkillMemory.record_retrievals_bulk`.
"""
import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.core.bus import MemoryBus
from ghost_agent.memory.skills import SkillMemory


# ------------------------------------------------------------ #13 episodic


async def test_fetch_episodic_passes_vector_memory():
    """The bus must hand its vector store to search_similar so the semantic
    episode-recall path (episodes.py _vector_search) can fire."""
    ep = MagicMock()
    seen = {}

    def search_similar(query, limit, vector_memory=None):
        seen["vector_memory"] = vector_memory
        return []

    ep.search_similar = search_similar
    vec = MagicMock()
    bus = MemoryBus(vector_memory=vec, episodic_memory=ep)
    await bus._fetch_episodic("how did I fix the docker probe")
    assert seen["vector_memory"] is vec


async def test_fetch_episodic_emits_one_item_per_episode():
    ep = MagicMock()
    ep.search_similar = MagicMock(return_value=[
        {"trigger": "t1", "outcome": "o1", "outcome_success": True},
        {"trigger": "t2", "outcome": "o2", "outcome_success": False},
    ])
    ep.format_episode = lambda e: f"- Trigger: {e['trigger']}"
    bus = MemoryBus(episodic_memory=ep)
    items = await bus._fetch_episodic("query")
    assert len(items) == 2
    assert all(it["source"] == "episodic" for it in items)
    assert items[0]["text"] == "- Trigger: t1"
    assert items[1]["text"] == "- Trigger: t2"


async def test_fetch_episodic_blob_fallback_for_legacy_stubs():
    """Stores without format_episode keep the old single-blob contract."""
    ep = MagicMock()
    ep.search_similar = MagicMock(return_value=[{"trigger": "t"}])
    ep.format_episode = None  # not callable → fallback
    ep.format_for_context = MagicMock(return_value="blob text")
    bus = MemoryBus(episodic_memory=ep)
    items = await bus._fetch_episodic("query")
    assert items == [{"source": "episodic", "text": "blob text"}]


def test_episodes_format_episode_matches_blob_lines():
    """format_episode is the extracted per-line renderer — format_for_context
    must be built from it so the two can never drift."""
    from ghost_agent.memory.episodes import EpisodicMemory
    ep = {"cluster_id": "general", "trigger": "T" * 150,
          "outcome": "worked", "outcome_success": True, "lesson": "L"}
    line = EpisodicMemory.format_episode(ep)
    assert "Trigger:" in line and "SUCCESS" in line and "Lesson: L" in line
    assert len(line) < 400  # per-field truncation still applies


# ------------------------------------------------------------- #14 per-item


async def test_fetch_skill_emits_one_item_per_lesson():
    skill = MagicMock()
    skill.get_playbook_items = MagicMock(return_value=[
        {"text": "lesson A", "trigger": "trig-a"},
        {"text": "lesson B", "trigger": "trig-b"},
        {"text": "lesson C", "trigger": "trig-c"},
    ])
    bus = MemoryBus(skill_memory=skill)
    items = await bus._fetch_skill("query")
    assert [it["text"] for it in items] == ["lesson A", "lesson B", "lesson C"]
    assert [it["trigger"] for it in items] == ["trig-a", "trig-b", "trig-c"]
    assert all(it["source"] == "skill" for it in items)


async def test_fetch_vector_items_carry_mem_id():
    vec = MagicMock()
    vec.search_items = MagicMock(return_value=[
        {"id": "chroma-1", "text": "doc one", "score": 0.1},
        {"id": "chroma-2", "text": "doc two", "score": 0.4},
    ])
    bus = MemoryBus(vector_memory=vec)
    items = await bus._fetch_vector("query")
    assert [it["mem_id"] for it in items] == ["chroma-1", "chroma-2"]


def test_format_markdown_with_survivors_returns_emitted_items():
    fused = [
        ({"source": "vector", "text": "kept-1", "mem_id": "a"}, 0.9),
        ({"source": "skill", "text": "kept-2", "trigger": "t"}, 0.8),
        ({"source": "vector", "text": "dropped-too-big", "mem_id": "b"}, 0.7),
    ]
    # Budget fits the first two items + headers but not the third.
    header_cost = len("### MEMORY CONTEXT:") + 1 + len(
        "### SKILL PLAYBOOK (lessons from prior runs — follow to avoid repeats):") + 1
    budget = header_cost + len("kept-1") + 1 + len("kept-2") + 1
    out, survivors = MemoryBus._format_markdown_with_survivors(fused, max_chars=budget)
    assert "kept-1" in out and "kept-2" in out
    assert "dropped-too-big" not in out
    assert [s.get("mem_id") or s.get("trigger") for s in survivors] == ["a", "t"]


def test_format_markdown_wrapper_unchanged_contract():
    fused = [({"source": "vector", "text": "hello"}, 0.9)]
    out = MemoryBus._format_markdown(fused, max_chars=500)
    assert isinstance(out, str) and "hello" in out


# ------------------------------------------------------- #16 deferred credit


async def test_credit_goes_only_to_surviving_items():
    """End-to-end hydrate: candidates beyond the per-source cap must NOT be
    credited; survivors are credited exactly once via the bulk APIs."""
    vec = MagicMock()
    vec.search_items = MagicMock(return_value=[
        {"id": f"id-{i}", "text": f"vector doc {i}", "score": 0.1 * i}
        for i in range(8)  # _PER_SOURCE_CAP is 6 → ids 6,7 must not be credited
    ])
    vec.bump_retrievals = MagicMock()
    skill = MagicMock()
    skill.get_playbook_items = MagicMock(return_value=[
        {"text": "lesson A", "trigger": "trig-a"},
        {"text": "lesson B", "trigger": "trig-b"},
    ])
    skill.record_retrievals_bulk = MagicMock()

    bus = MemoryBus(vector_memory=vec, skill_memory=skill)
    out = await bus.hydrate_context("what did I learn about docker probes")
    assert out  # something was injected

    vec.bump_retrievals.assert_called_once()
    bumped = vec.bump_retrievals.call_args.args[0]
    assert set(bumped).issubset({f"id-{i}" for i in range(6)})
    assert "id-6" not in bumped and "id-7" not in bumped

    skill.record_retrievals_bulk.assert_called_once()
    credited = skill.record_retrievals_bulk.call_args.args[0]
    assert set(credited) == {"trig-a", "trig-b"}


async def test_no_credit_calls_when_nothing_survives():
    vec = MagicMock()
    vec.search_items = MagicMock(return_value=[])
    vec.bump_retrievals = MagicMock()
    bus = MemoryBus(vector_memory=vec)
    out = await bus.hydrate_context("query with no hits")
    assert out == ""
    vec.bump_retrievals.assert_not_called()


# ------------------------------------------------- SkillMemory bulk credit


@pytest.fixture
def skill_store(tmp_path):
    sm = SkillMemory(tmp_path)
    sm.save_playbook([
        {"task": "docker probe", "mistake": "m1", "solution": "s1",
         "retrievals": 0, "helpful_retrievals": 0},
        {"task": "tor circuit", "mistake": "m2", "solution": "s2",
         "retrievals": 5, "helpful_retrievals": 2},
        {"task": "untouched", "mistake": "m3", "solution": "s3",
         "retrievals": 1, "helpful_retrievals": 1},
    ])
    return sm


def test_record_retrievals_bulk_single_write(skill_store, monkeypatch):
    """N lessons credited → exactly ONE playbook save (the old per-lesson
    record_retrieval path cost one full-file rewrite per lesson)."""
    saves = []
    real_save = skill_store._save_playbook_unlocked
    monkeypatch.setattr(
        skill_store, "_save_playbook_unlocked",
        lambda pb: (saves.append(1), real_save(pb)))
    updated = skill_store.record_retrievals_bulk(["docker probe", "tor circuit"])
    assert updated == 2
    assert len(saves) == 1

    pb = skill_store._load_playbook()
    by_task = {(p.get("trigger") or p.get("task")): p for p in pb}
    assert by_task["docker probe"]["retrievals"] == 1
    assert by_task["tor circuit"]["retrievals"] == 6
    assert by_task["untouched"]["retrievals"] == 1  # not credited


def test_record_retrievals_bulk_dedups_and_ignores_empty(skill_store):
    updated = skill_store.record_retrievals_bulk(
        ["docker probe", "DOCKER PROBE", "", None, "no-such-lesson"])
    assert updated == 1
    pb = skill_store._load_playbook()
    lesson = next(p for p in pb if (p.get("trigger") or p.get("task")) == "docker probe")
    assert lesson["retrievals"] == 1  # bumped once despite the duplicate


def test_get_playbook_items_has_no_side_effects(skill_store):
    before = json.dumps(skill_store._load_playbook(), sort_keys=True)
    items = skill_store.get_playbook_items("docker probe failed again")
    assert items and all("text" in it and "trigger" in it for it in items)
    after = json.dumps(skill_store._load_playbook(), sort_keys=True)
    assert before == after  # NO counter bumps from the items API


def test_get_playbook_context_contracts_preserved(skill_store, tmp_path):
    # query + BM25 hit → relevant-lessons header, counters bumped in ONE write
    out = skill_store.get_playbook_context(query="docker probe broke")
    assert out.startswith("## RELEVANT LESSONS LEARNED")
    pb = skill_store._load_playbook()
    lesson = next(p for p in pb if (p.get("trigger") or p.get("task")) == "docker probe")
    assert lesson["retrievals"] == 1

    # query with zero keyword overlap → "" (no recency dump)
    assert skill_store.get_playbook_context(query="zzz qqq xyzzy") == ""

    # no query → recency header
    out = skill_store.get_playbook_context()
    assert out.startswith("## RECENT LESSONS LEARNED")

    # empty playbook → legacy sentinel string
    (tmp_path / "empty").mkdir()
    empty = SkillMemory(tmp_path / "empty")
    assert empty.get_playbook_context() == "No lessons learned yet."


# ------------------------------------------------- VectorMemory deferral


def _vm_stub():
    from ghost_agent.memory.vector import VectorMemory
    vm = MagicMock()
    vm._lock = threading.RLock()
    vm._get_lock = lambda: vm._lock
    # Selection stub: two items shaped like _search_selection output.
    selection = [
        {"mem_id": "id-1", "doc": "d1", "meta": {"type": "auto", "timestamp": "t"},
         "dist": 0.2, "p_score": 1, "combined_score": 10.2},
        {"mem_id": "id-2", "doc": "d2", "meta": {"type": "manual", "timestamp": "t"},
         "dist": 0.3, "p_score": 0, "combined_score": 0.3},
    ]
    vm._search_selection = MagicMock(return_value=selection)
    vm._render_item = lambda item: VectorMemory._render_item(item)
    vm.bump_retrievals = MagicMock()
    return vm, selection


def test_vector_search_items_returns_ids_and_never_bumps():
    from ghost_agent.memory.vector import VectorMemory
    vm, _sel = _vm_stub()
    items = VectorMemory.search_items(vm, "query")
    assert [it["id"] for it in items] == ["id-1", "id-2"]
    assert all(isinstance(it["text"], str) and it["text"] for it in items)
    vm.bump_retrievals.assert_not_called()


def test_vector_search_default_still_bumps_once():
    from ghost_agent.memory.vector import VectorMemory
    vm, _sel = _vm_stub()
    out = VectorMemory.search(vm, "query")
    assert "d1" in out and "d2" in out
    vm.bump_retrievals.assert_called_once_with(["id-1", "id-2"])


def test_vector_search_record_retrievals_false_skips_bump():
    from ghost_agent.memory.vector import VectorMemory
    vm, _sel = _vm_stub()
    out = VectorMemory.search(vm, "query", record_retrievals=False)
    assert "d1" in out
    vm.bump_retrievals.assert_not_called()


def test_vector_bump_retrievals_dedups():
    from ghost_agent.memory.vector import VectorMemory
    vm = MagicMock()
    vm._bump_retrieval_stats = MagicMock()
    VectorMemory.bump_retrievals(vm, ["a", "b", "a", None, "b"])
    vm._bump_retrieval_stats.assert_called_once_with(["a", "b"])


# --------------------------------------------- #15 tier scoring / new types


def _run_selection(monkeypatch, batch):
    """Drive the real _search_selection against a stubbed Chroma collection.

    `batch` is a list of (id, doc, meta, dist) tuples returned for the
    single (non-identity) query batch.
    """
    from ghost_agent.memory.vector import VectorMemory
    vm = MagicMock()
    vm._lock = threading.RLock()
    vm._get_lock = lambda: vm._lock
    vm.collection = MagicMock()
    vm.collection.query.return_value = {
        "ids": [[b[0] for b in batch]],
        "documents": [[b[1] for b in batch]],
        "metadatas": [[b[2] for b in batch]],
        "distances": [[b[3] for b in batch]],
    }
    return VectorMemory._search_selection(vm, "some technical query about docker",
                                          inject_identity=False)


def test_identity_and_synthesis_types_scored_from_metadata(monkeypatch):
    """type=identity / type=synthesis must rank at their curated tiers, not
    fall into the generic else (auto tier, 0.55 threshold)."""
    sel = _run_selection(monkeypatch, [
        ("i1", "prefers dark mode", {"type": "identity", "timestamp": "2026-01-01T00:00:00Z"}, 0.75),
        ("s1", "consolidated insight", {"type": "synthesis", "timestamp": "2026-01-01T00:00:00Z"}, 0.72),
        ("a1", "random auto chunk", {"type": "auto", "timestamp": "2026-01-01T00:00:00Z"}, 0.75),
    ])
    kept = {item["mem_id"]: item for item in sel}
    # identity at dist .75 passes its .8 threshold; auto at .75 fails .55
    assert "i1" in kept and kept["i1"]["p_score"] == -10
    assert "s1" in kept and kept["s1"]["p_score"] == -15
    assert "a1" not in kept


def test_tier_prior_no_longer_absolute(monkeypatch):
    """A decisively closer match must be able to outrank an adjacent-tier
    item — the old ×10 multiplier made category priority absolute."""
    sel = _run_selection(monkeypatch, [
        # manual tier (p_score 0), barely relevant
        ("m1", "vaguely related note", {"type": "manual", "timestamp": "2026-01-01T00:00:00Z"}, 0.62),
        # auto tier (p_score 1, one tier lower), exact match
        ("a1", "exact answer chunk", {"type": "auto", "timestamp": "2026-01-01T00:00:00Z"}, 0.05),
    ])
    order = [item["mem_id"] for item in sel]
    # auto item: 1*0.3 + 0.05 ≈ 0.35+penalty; manual: 0*0.3 + 0.62 ≈ 0.62+penalty
    assert order.index("a1") < order.index("m1")


def test_distant_tiers_still_dominated_by_prior(monkeypatch):
    """The prior must still separate DISTANT tiers: an identity hit at its
    threshold edge outranks an auto chunk at the same distance."""
    sel = _run_selection(monkeypatch, [
        ("i1", "user's name is Vasilis", {"type": "identity", "timestamp": "2026-01-01T00:00:00Z"}, 0.5),
        ("a1", "some auto chunk", {"type": "auto", "timestamp": "2026-01-01T00:00:00Z"}, 0.05),
    ])
    order = [item["mem_id"] for item in sel]
    assert order.index("i1") < order.index("a1")


def test_synthesis_is_prunable_identity_is_not():
    from ghost_agent.memory.vector import VectorMemory
    assert "synthesis" in VectorMemory._PRUNABLE_TYPES
    assert "auto" in VectorMemory._PRUNABLE_TYPES
    assert "identity" not in VectorMemory._PRUNABLE_TYPES
