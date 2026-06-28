"""Memory/retrieval relevance-gating regression tests.

Covers the review fixes that stopped the memory bus injecting ~12k chars of
often-irrelevant context every turn:

* `MemoryBus._format_markdown` now emits items in DESCENDING fused-score order
  under a single global budget (the fused RRF ranking is no longer discarded by
  fixed per-source budgets), with a small per-source cap and a cross-tier
  relevance floor.
* `MemoryBus._extract_query_terms` no longer auto-seeds the literal "user" term
  (which returned the user ego-graph on every turn).
* `VectorMemory.search` gates name-memory / summary rows on a RELAXED distance
  threshold instead of injecting them unconditionally.
* `EpisodicMemory._vector_search` derives relevance from distance and drops hits
  below a floor (was a flat 1.0 for every hit).

Pure-python: no docker, no live embedding model — collections / vector memories
are mocked and instances built via ``__new__`` to skip heavy ``__init__``.
"""

import pytest

from ghost_agent.core.bus import MemoryBus
from ghost_agent.memory.vector import VectorMemory
from ghost_agent.memory.episodes import EpisodicMemory


# ----------------------------------------------------------------------------
# 1. _format_markdown — fused-score ordering, per-source cap, relevance floor
# ----------------------------------------------------------------------------

def _text_lines(md: str):
    """Return the emitted content lines (headers / blanks stripped)."""
    return [ln for ln in md.splitlines() if ln.strip() and not ln.startswith("### ")]


def test_format_markdown_emits_in_descending_fused_score_order():
    # Interleave sources; the highest fused score must lead regardless of source.
    fused = [
        ({"source": "graph", "text": "G_HIGH"}, 0.90),
        ({"source": "vector", "text": "V_MID"}, 0.50),
        ({"source": "skill", "text": "S_LOW"}, 0.10),
    ]
    md = MemoryBus._format_markdown(fused, max_chars=6000)
    lines = _text_lines(md)
    assert lines == ["G_HIGH", "V_MID", "S_LOW"], lines


def test_format_markdown_does_not_regroup_low_score_above_high_score():
    # A low-scored graph item must NOT be hoisted above a high-scored vector
    # item just because graph used to be its own section.
    fused = [
        ({"source": "vector", "text": "V_TOP"}, 0.99),
        ({"source": "graph", "text": "G_BOTTOM"}, 0.01),
    ]
    md = MemoryBus._format_markdown(fused, max_chars=6000)
    lines = _text_lines(md)
    assert lines.index("V_TOP") < lines.index("G_BOTTOM")


def test_format_markdown_per_source_cap():
    fused = [({"source": "graph", "text": f"G{i}"}, 1.0 - i * 0.01) for i in range(12)]
    md = MemoryBus._format_markdown(fused, max_chars=100000)
    lines = _text_lines(md)
    assert len(lines) == MemoryBus._PER_SOURCE_CAP  # capped at 6, not 12


def test_format_markdown_relevance_floor_drops_low_signal(monkeypatch):
    # With a floor of 0.5, an item normalised to 0.4 must be dropped.
    monkeypatch.setattr(MemoryBus, "_RELEVANCE_FLOOR", 0.5)
    fused = [
        ({"source": "vector", "text": "KEEP"}, 1.0),   # normalised 1.0
        ({"source": "vector", "text": "DROP"}, 0.4),   # normalised 0.4 < 0.5
    ]
    md = MemoryBus._format_markdown(fused, max_chars=6000)
    lines = _text_lines(md)
    assert "KEEP" in lines
    assert "DROP" not in lines


def test_format_markdown_empty_returns_empty_string():
    assert MemoryBus._format_markdown([], max_chars=6000) == ""


# ----------------------------------------------------------------------------
# 2. _extract_query_terms — no auto-seeded "user"
# ----------------------------------------------------------------------------

def test_extract_query_terms_no_user_seed():
    terms = MemoryBus._extract_query_terms("design the database schema for orders")
    assert "user" not in terms
    assert "database" in terms or "schema" in terms


def test_extract_query_terms_trivial_query_no_user_fallback():
    # Trivial query yields no >3-char non-stopword terms; fallback returns raw
    # tokens as-is and must NOT re-introduce "user".
    terms = MemoryBus._extract_query_terms("hi")
    assert "user" not in terms
    assert terms == ["hi"]


# ----------------------------------------------------------------------------
# 3. VectorMemory.search — name/summary gated on RELAXED distance threshold
# ----------------------------------------------------------------------------

class _DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _make_vector_memory(results):
    vm = VectorMemory.__new__(VectorMemory)

    class _Coll:
        def query(self, *a, **k):
            return results

    vm.collection = _Coll()
    vm._get_lock = lambda: _DummyLock()
    vm._bump_retrieval_stats = lambda *a, **k: None
    return vm


def _single_batch(doc, m_type, dist):
    return {
        "ids": [["id1"]],
        "documents": [[doc]],
        "metadatas": [[{"type": m_type, "timestamp": "2026-01-01T00:00:00Z"}]],
        "distances": [[dist]],
    }


def test_name_memory_dropped_beyond_relaxed_threshold():
    # name threshold = 1.0 -> relaxed = 1.5; dist 1.8 must now be EXCLUDED
    # (previously injected unconditionally).
    vm = _make_vector_memory(_single_batch("My name is Bob", "manual", 1.8))
    out = vm.search("database schema design", inject_identity=False)
    assert "My name is Bob" not in out


def test_name_memory_kept_within_relaxed_threshold():
    # dist 1.3 is between strict (1.0) and relaxed (1.5) -> still injected.
    vm = _make_vector_memory(_single_batch("My name is Bob", "manual", 1.3))
    out = vm.search("database schema design", inject_identity=False)
    assert "My name is Bob" in out


def test_summary_dropped_beyond_relaxed_threshold():
    # summary threshold = 0.75 -> relaxed = 1.125; dist 2.0 excluded.
    vm = _make_vector_memory(_single_batch("Document summary here", "document_summary", 2.0))
    out = vm.search("unrelated quantum topic", inject_identity=False)
    assert "Document summary here" not in out


def test_summary_kept_within_relaxed_threshold():
    vm = _make_vector_memory(_single_batch("Document summary here", "document_summary", 1.0))
    out = vm.search("document summary topic", inject_identity=False)
    assert "Document summary here" in out


# ----------------------------------------------------------------------------
# 4. EpisodicMemory._vector_search — distance-derived score + threshold drop
# ----------------------------------------------------------------------------

class _FakeVectorForEpisodes:
    def __init__(self, hits):
        self._hits = hits

    def search_advanced(self, trigger, limit=5):
        return self._hits


def _make_episodic():
    ep = EpisodicMemory.__new__(EpisodicMemory)
    ep.get_episode = lambda eid: {"id": eid, "trigger": f"ep{eid}"}
    return ep


def test_episodic_distance_derived_score_and_threshold_drop():
    hits = [
        # close hit -> relevance 1 - 0.1 = 0.9 (kept)
        {"metadata": {"type": "episode", "episode_id": 1}, "score": 0.1},
        # far hit  -> relevance 1 - 0.95 = 0.05 < 0.2 floor (dropped)
        {"metadata": {"type": "episode", "episode_id": 2}, "score": 0.95},
        # non-episode hit -> filtered out before scoring
        {"metadata": {"type": "document"}, "score": 0.0},
    ]
    ep = _make_episodic()
    results = ep._vector_search("some trigger", limit=10,
                                vector_memory=_FakeVectorForEpisodes(hits))
    ids = [r["id"] for r in results]
    assert ids == [1]                                   # ep 2 dropped, doc filtered
    assert results[0]["relevance_score"] == pytest.approx(0.9)
    assert results[0]["relevance_score"] != 1.0          # no longer a flat 1.0


def test_episodic_no_hits_returns_empty():
    ep = _make_episodic()
    out = ep._vector_search("x", limit=5,
                            vector_memory=_FakeVectorForEpisodes([]))
    assert out == []
