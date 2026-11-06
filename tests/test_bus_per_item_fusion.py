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


# ------------------------------------------------------------ #13 episodic


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


def test_format_markdown_with_survivors_returns_emitted_items():
    fused = [
        ({"source": "vector", "text": "kept-1", "mem_id": "a"}, 0.9),
        ({"source": "graph", "text": "kept-2", "mem_id": "t"}, 0.8),
        ({"source": "vector", "text": "dropped-too-big", "mem_id": "b"}, 0.7),
    ]
    # Budget fits the first two items + headers but not the third.
    header_cost = len("### MEMORY CONTEXT:") + 1 + len(
        "### TOPOLOGICAL KNOWLEDGE GRAPH:") + 1
    budget = header_cost + len("kept-1") + 1 + len("kept-2") + 1
    out, survivors = MemoryBus._format_markdown_with_survivors(fused, max_chars=budget)
    assert "kept-1" in out and "kept-2" in out
    assert "dropped-too-big" not in out
    assert [s.get("mem_id") for s in survivors] == ["a", "t"]


def test_format_markdown_wrapper_unchanged_contract():
    fused = [({"source": "vector", "text": "hello"}, 0.9)]
    out = MemoryBus._format_markdown(fused, max_chars=500)
    assert isinstance(out, str) and "hello" in out


# ------------------------------------------------------- #16 deferred credit


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
    # §4KZ: the profile's mirror (identity) is the CURRENT owner fact — above a
    # synthesis of older fragments
    assert "i1" in kept and kept["i1"]["p_score"] == -15
    assert "s1" in kept and kept["s1"]["p_score"] == -10
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
