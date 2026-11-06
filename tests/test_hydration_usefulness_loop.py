"""The hydration stash and the vector helpful-credit (2026-07-14).

`VectorMemory.bump_helpful` is the double-half-life credit for an item a
reply actually drew on. (§4NM: the post-turn usefulness judge, its
observations ledger and the dream's RRF refit were removed — the stash
and the credit primitive are what remain.)
"""

from unittest.mock import MagicMock

from ghost_agent.core.bus import MemoryBus
from ghost_agent.memory.vector import VectorMemory


class TestStash:
    async def test_hydrate_context_returns_the_survivors(self):
        graph = MagicMock()
        graph.owner_facts_matching = MagicMock(return_value=["(Elena) -[LIVES_IN]-> (Kalopetres)"])
        bus = MemoryBus(graph_memory=graph)

        out = await bus.hydrate_context("where do I live")

        assert "Kalopetres" in out


class TestVectorBumpHelpful:
    def test_bump_helpful_increments_and_touches(self):
        vm = VectorMemory.__new__(VectorMemory)  # bypass heavy __init__
        vm.collection = MagicMock()
        vm.collection.get.return_value = {
            "ids": ["v1"], "metadatas": [{"helpful_count": 2}],
        }
        vm.bump_helpful(["v1", "v1", None])
        vm.collection.get.assert_called_once_with(ids=["v1"], include=["metadatas"])
        meta = vm.collection.update.call_args.kwargs["metadatas"][0]
        assert meta["helpful_count"] == 3
        assert "last_accessed" in meta

    def test_bump_helpful_empty_is_noop(self):
        vm = VectorMemory.__new__(VectorMemory)
        vm.collection = MagicMock()
        vm.bump_helpful([])
        vm.collection.get.assert_not_called()
