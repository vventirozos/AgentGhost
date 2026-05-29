"""Tests for the Episodic Replay enhancements (vector search + recovery search)."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from ghost_agent.memory.episodes import EpisodicMemory


@pytest.fixture
def memory_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    import shutil
    shutil.rmtree(d)


@pytest.fixture
def ep_mem(memory_dir):
    return EpisodicMemory(memory_dir)


class TestSearchSimilarWithVector:
    def test_falls_back_to_substring_without_vector(self, ep_mem):
        ep_mem.record_episode(
            trigger="Python import error",
            outcome="Fixed by installing package",
            success=True,
            lesson="Always check pip freeze first",
        )
        results = ep_mem.search_similar("import error")
        assert len(results) >= 1

    def test_uses_vector_when_provided(self, ep_mem):
        ep_mem.record_episode(
            trigger="CSS layout broken",
            outcome="Fixed with flexbox",
            success=True,
        )
        mock_vector = MagicMock()
        # search_advanced is the real raw-hits API (the old `search_raw`
        # existed nowhere). Hits need type=="episode" + episode_id metadata.
        mock_vector.search_advanced = MagicMock(return_value=[
            {"metadata": {"type": "episode", "episode_id": 1}, "score": 0.3},
        ])
        results = ep_mem.search_similar(
            "styling issue with HTML page",
            vector_memory=mock_vector,
        )
        mock_vector.search_advanced.assert_called_once()

    def test_falls_back_when_vector_fails(self, ep_mem):
        ep_mem.record_episode(
            trigger="database connection timeout",
            outcome="Increased pool size",
            success=True,
        )
        mock_vector = MagicMock()
        mock_vector.search_advanced = MagicMock(side_effect=Exception("vector down"))
        results = ep_mem.search_similar(
            "database timeout",
            vector_memory=mock_vector,
        )
        # Should fall back to substring and find it
        assert len(results) >= 1

    def test_falls_back_when_vector_returns_empty(self, ep_mem):
        ep_mem.record_episode(
            trigger="memory leak detected",
            outcome="Fixed circular reference",
            success=True,
        )
        mock_vector = MagicMock()
        mock_vector.search_advanced = MagicMock(return_value=[])
        results = ep_mem.search_similar(
            "memory leak",
            vector_memory=mock_vector,
        )
        assert len(results) >= 1


class TestSearchByOutcome:
    def test_search_successes(self, ep_mem):
        ep_mem.record_episode(trigger="Task A", outcome="Done", success=True)
        ep_mem.record_episode(trigger="Task B", outcome="Failed", success=False)
        ep_mem.record_episode(trigger="Task C", outcome="Done", success=True)

        successes = ep_mem.search_by_outcome(success=True)
        assert len(successes) == 2
        assert all(ep["outcome_success"] == 1 for ep in successes)

    def test_search_failures(self, ep_mem):
        ep_mem.record_episode(trigger="Task A", outcome="Done", success=True)
        ep_mem.record_episode(trigger="Task B", outcome="Failed", success=False)

        failures = ep_mem.search_by_outcome(success=False)
        assert len(failures) == 1
        assert failures[0]["outcome_success"] == 0


class TestSearchRecoveries:
    def test_finds_recovery_episodes(self, ep_mem):
        # A successful recovery episode
        ep_mem.record_episode(
            trigger="Sandbox execution timeout during data processing",
            outcome="Recovered by chunking the data",
            success=True,
            lesson="Split large datasets into chunks before processing",
        )
        # A failure without lesson
        ep_mem.record_episode(
            trigger="Sandbox execution timeout",
            outcome="Could not recover",
            success=False,
        )

        recoveries = ep_mem.search_recoveries("execution timeout")
        assert len(recoveries) >= 1
        assert recoveries[0]["lesson"] != ""
        assert recoveries[0]["outcome_success"] == 1

    def test_no_recoveries_found(self, ep_mem):
        ep_mem.record_episode(trigger="unrelated", outcome="done", success=True)
        recoveries = ep_mem.search_recoveries("completely different problem")
        assert len(recoveries) == 0

    def test_recovery_search_with_vector(self, ep_mem):
        ep_mem.record_episode(
            trigger="API rate limit exceeded",
            outcome="Added exponential backoff",
            success=True,
            lesson="Use exponential backoff for rate limits",
        )
        mock_vector = MagicMock()
        mock_vector.search_advanced = MagicMock(return_value=[
            {"metadata": {"type": "episode", "episode_id": 1}, "score": 0.2},
        ])
        recoveries = ep_mem.search_recoveries(
            "rate limiting error",
            vector_memory=mock_vector,
        )
        # Should have used vector search
        mock_vector.search_advanced.assert_called()


class TestExistingFunctionality:
    """Ensure the original functionality still works."""

    def test_record_and_retrieve(self, ep_mem):
        ep_id = ep_mem.record_episode(
            trigger="test trigger",
            context="test context",
            actions=[{"tool": "execute", "args": {"cmd": "ls"}, "result": "files", "success": True}],
            outcome="success",
            success=True,
            lesson="test lesson",
        )
        ep = ep_mem.get_episode(ep_id)
        assert ep is not None
        assert ep["trigger"] == "test trigger"
        assert len(ep["actions"]) == 1

    def test_count(self, ep_mem):
        assert ep_mem.count() == 0
        ep_mem.record_episode(trigger="test")
        assert ep_mem.count() == 1

    def test_format_for_context(self, ep_mem):
        ep_mem.record_episode(
            trigger="debug task",
            outcome="fixed the bug",
            success=True,
            lesson="check imports first",
        )
        episodes = ep_mem.get_recent_episodes()
        formatted = ep_mem.format_for_context(episodes)
        assert "RELEVANT PAST EPISODES" in formatted
        assert "debug task" in formatted
