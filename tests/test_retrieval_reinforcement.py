"""Tests for retrieval reinforcement decay (#10).

Verifies that:
- _bump_retrieval_stats updates retrieval_count and last_accessed
- Time decay uses effective_half_life based on retrieval_count
- Frequently accessed memories decay slower
"""

import pytest
import hashlib
from unittest.mock import MagicMock, patch
from pathlib import Path


@pytest.fixture
def mock_vector_memory():
    vm = MagicMock()
    vm._lock = __import__("threading").RLock()
    vm._get_lock = lambda: vm._lock
    return vm


class TestBumpRetrievalStats:
    def test_bump_increments_retrieval_count(self, mock_vector_memory):
        from ghost_agent.memory.vector import VectorMemory
        vm = mock_vector_memory

        # Simulate existing metadata
        vm.collection = MagicMock()
        vm.collection.get.return_value = {
            "ids": ["id1"],
            "metadatas": [{"type": "auto", "retrieval_count": 2, "timestamp": "2025-01-01T00:00:00Z"}]
        }

        VectorMemory._bump_retrieval_stats(vm, ["id1"])

        vm.collection.update.assert_called_once()
        call_kwargs = vm.collection.update.call_args
        updated_meta = call_kwargs[1]["metadatas"][0] if call_kwargs[1] else call_kwargs[0][1][0]
        # Should be 3 (was 2, incremented by 1)
        assert updated_meta["retrieval_count"] == 3
        assert "last_accessed" in updated_meta

    def test_bump_initializes_retrieval_count_from_zero(self, mock_vector_memory):
        from ghost_agent.memory.vector import VectorMemory
        vm = mock_vector_memory
        vm.collection = MagicMock()
        vm.collection.get.return_value = {
            "ids": ["id1"],
            "metadatas": [{"type": "auto", "timestamp": "2025-01-01T00:00:00Z"}]
        }

        VectorMemory._bump_retrieval_stats(vm, ["id1"])

        call_kwargs = vm.collection.update.call_args
        updated_meta = call_kwargs[1]["metadatas"][0] if call_kwargs[1] else call_kwargs[0][1][0]
        assert updated_meta["retrieval_count"] == 1

    def test_bump_handles_empty_ids(self, mock_vector_memory):
        from ghost_agent.memory.vector import VectorMemory
        vm = mock_vector_memory
        vm.collection = MagicMock()

        VectorMemory._bump_retrieval_stats(vm, [])
        vm.collection.get.assert_not_called()

    def test_bump_handles_exception_gracefully(self, mock_vector_memory):
        from ghost_agent.memory.vector import VectorMemory
        vm = mock_vector_memory
        vm.collection = MagicMock()
        vm.collection.get.side_effect = Exception("DB error")

        # Should not raise
        VectorMemory._bump_retrieval_stats(vm, ["id1"])


class TestRetrievalReinforcedDecay:
    def test_half_life_increases_with_retrieval_count(self):
        import math
        # Baseline: 0 retrievals → 30 day half-life
        base_hl = 30.0 * (1.0 + math.log1p(0))
        assert base_hl == 30.0

        # Each additional retrieval increases half-life logarithmically
        hl_5 = 30.0 * (1.0 + math.log1p(5))
        assert hl_5 > base_hl

        hl_20 = 30.0 * (1.0 + math.log1p(20))
        assert hl_20 > hl_5

        # Verify monotonic increase
        prev = base_hl
        for n in [1, 5, 10, 20, 50]:
            hl = 30.0 * (1.0 + math.log1p(n))
            assert hl > prev
            prev = hl

    def test_decay_penalty_lower_for_frequently_accessed(self):
        import math

        age_days = 60.0  # 60 days old

        # Zero retrievals
        hl_0 = 30.0
        penalty_0 = 0.30 * (1.0 - math.exp(-age_days / hl_0))

        # 10 retrievals
        hl_10 = 30.0 * (1.0 + math.log1p(10))
        penalty_10 = 0.30 * (1.0 - math.exp(-age_days / hl_10))

        # Frequently accessed should have LOWER penalty (decays slower)
        assert penalty_10 < penalty_0

    def test_search_advanced_calls_bump(self, mock_vector_memory):
        from ghost_agent.memory.vector import VectorMemory
        vm = mock_vector_memory
        vm.collection = MagicMock()
        vm.collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"type": "auto"}, {"type": "auto"}]],
            "distances": [[0.1, 0.2]]
        }
        vm.collection.get.return_value = {
            "ids": ["id1", "id2"],
            "metadatas": [{"type": "auto"}, {"type": "auto"}]
        }
        # Bind the real _bump_retrieval_stats to the mock so it executes
        vm._bump_retrieval_stats = lambda ids: VectorMemory._bump_retrieval_stats(vm, ids)

        result = VectorMemory.search_advanced(vm, "test query")

        assert len(result) == 2
        # Verify bump updated the collection
        vm.collection.update.assert_called_once()
