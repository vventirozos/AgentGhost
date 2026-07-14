"""Dream-cycle graph compression wiring.

`execute_graph_compression` was hardened 2026-07-07 (temporal merge
semantics, test_graph_compression_temporal.py) but had no live caller —
journal §4B tracked it as "hardened before wiring". The dream cycle now
runs a compression step after `prune_stale_edges`:

- `GraphMemory.propose_merge_candidates` produces deterministic candidate
  pairs in two tiers: "safe" (identical after stripping punctuation /
  whitespace) and "fuzzy" (high-similarity lexicographic neighbors —
  plurals, trailing typos).
- `Dreamer._compress_graph_nodes` applies safe pairs directly; fuzzy pairs
  only when the worker model confirms both names refer to the same entity.
  Capped per cycle; a worker failure skips fuzzy pairs but keeps safe ones;
  the step never raises out of the dream.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.dream import Dreamer
from ghost_agent.memory.graph import GraphMemory


@pytest.fixture
def gm(tmp_path):
    return GraphMemory(tmp_path)


def _add(gm, s, p, o):
    gm.add_triplets([{"subject": s, "predicate": p, "object": o}])


# ------------------------------------------------- propose_merge_candidates


class TestProposeMergeCandidates:
    def test_punctuation_variants_are_safe_pairs(self, gm):
        _add(gm, "new york", "HOSTS", "un hq")
        _add(gm, "new york", "LOCATED_IN", "usa")
        _add(gm, "new-york", "HAS", "subway")

        cands = gm.propose_merge_candidates()
        safe = [c for c in cands if c["kind"] == "safe"]
        assert len(safe) == 1
        # Higher-degree node ("new york", 2 edges) survives as canonical.
        assert safe[0]["old_node"] == "new-york"
        assert safe[0]["new_node"] == "new york"

    def test_near_duplicates_are_fuzzy_pairs(self, gm):
        _add(gm, "kubernetes", "RUNS_ON", "ghost")
        _add(gm, "kubernete", "HAS", "pods")

        cands = gm.propose_merge_candidates()
        fuzzy = [c for c in cands if c["kind"] == "fuzzy"]
        assert len(fuzzy) == 1
        # Degree tie → longer name survives.
        assert fuzzy[0]["old_node"] == "kubernete"
        assert fuzzy[0]["new_node"] == "kubernetes"

    def test_distinct_nodes_not_proposed(self, gm):
        _add(gm, "apple", "IS_A", "fruit")
        _add(gm, "banana", "IS_A", "fruit")
        _add(gm, "carrot", "IS_A", "vegetable")

        assert gm.propose_merge_candidates() == []

    def test_short_and_numeric_nodes_excluded(self, gm):
        _add(gm, "ab", "IS", "ab-")   # <=2 chars: excluded outright
        _add(gm, "1234", "IS", "12345")  # digits: excluded

        assert gm.propose_merge_candidates() == []

    def test_max_candidates_cap(self, gm):
        for i in range(6):
            _add(gm, f"topic {i}", "REL", f"topic-{i}")
        assert len(gm.propose_merge_candidates(max_candidates=3)) == 3

    def test_read_only(self, gm):
        _add(gm, "new york", "HOSTS", "un hq")
        _add(gm, "new-york", "HAS", "subway")
        before = sorted(gm.nx_graph.nodes())
        gm.propose_merge_candidates()
        assert sorted(gm.nx_graph.nodes()) == before


# ------------------------------------------------- Dreamer._compress_graph_nodes


class TestDreamCompressionStep:
    async def test_safe_merges_applied_without_llm(self, gm):
        _add(gm, "new york", "HOSTS", "un hq")
        _add(gm, "new york", "LOCATED_IN", "usa")
        _add(gm, "new-york", "HAS", "subway")
        ctx = MagicMock()
        ctx.graph_memory = gm
        ctx.llm_client = None  # no worker: fuzzy skipped, safe still applied

        merged = await Dreamer(ctx)._compress_graph_nodes("test-model")

        assert merged == 1
        assert "new-york" not in gm.nx_graph.nodes
        # The variant's edge migrated onto the canonical node.
        assert gm.nx_graph.has_edge("new york", "subway")

    async def test_fuzzy_merge_applied_when_worker_confirms(self, gm):
        _add(gm, "kubernetes", "RUNS_ON", "ghost")
        _add(gm, "kubernete", "HAS", "pods")
        ctx = MagicMock()
        ctx.graph_memory = gm
        ctx.llm_client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": '{"same_entity": [1]}'}}]
        })

        merged = await Dreamer(ctx)._compress_graph_nodes("test-model")

        assert merged == 1
        assert "kubernete" not in gm.nx_graph.nodes
        assert gm.nx_graph.has_edge("kubernetes", "pods")

    async def test_fuzzy_merge_skipped_when_worker_rejects(self, gm):
        _add(gm, "kubernetes", "RUNS_ON", "ghost")
        _add(gm, "kubernete", "HAS", "pods")
        ctx = MagicMock()
        ctx.graph_memory = gm
        ctx.llm_client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": '{"same_entity": []}'}}]
        })

        merged = await Dreamer(ctx)._compress_graph_nodes("test-model")

        assert merged == 0
        assert "kubernete" in gm.nx_graph.nodes
        assert "kubernetes" in gm.nx_graph.nodes

    async def test_worker_failure_keeps_safe_merges(self, gm):
        _add(gm, "new york", "HOSTS", "un hq")
        _add(gm, "new-york", "HAS", "subway")
        _add(gm, "kubernetes", "RUNS_ON", "ghost")
        _add(gm, "kubernete", "HAS", "pods")
        ctx = MagicMock()
        ctx.graph_memory = gm
        ctx.llm_client.chat_completion = AsyncMock(side_effect=RuntimeError("worker down"))

        merged = await Dreamer(ctx)._compress_graph_nodes("test-model")

        assert merged == 1  # the safe pair only
        assert "new-york" not in gm.nx_graph.nodes
        assert "kubernete" in gm.nx_graph.nodes  # fuzzy untouched on failure

    async def test_merge_cap_respected(self, gm):
        for i in range(5):
            _add(gm, f"topic {i}", "REL", f"filler{i}")
            _add(gm, f"topic-{i}", "REL", f"other{i}")
        ctx = MagicMock()
        ctx.graph_memory = gm
        ctx.llm_client = None

        merged = await Dreamer(ctx)._compress_graph_nodes("test-model", max_merges=2)

        assert merged == 2

    async def test_no_graph_memory_returns_zero(self):
        ctx = MagicMock()
        ctx.graph_memory = None

        assert await Dreamer(ctx)._compress_graph_nodes("test-model") == 0

    async def test_sqlite_and_mirror_stay_consistent(self, gm):
        _add(gm, "new york", "HOSTS", "un hq")
        _add(gm, "new-york", "HAS", "subway")
        ctx = MagicMock()
        ctx.graph_memory = gm
        ctx.llm_client = None

        await Dreamer(ctx)._compress_graph_nodes("test-model")

        current = {(t["subject"], t["predicate"], t["object"])
                   for t in gm.get_recent_triplets()}
        assert ("new york", "HAS", "subway") in current
        assert not any("new-york" in (s, o) for s, _, o in current)
