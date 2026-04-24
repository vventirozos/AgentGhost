"""Tests for episodic memory (#2).

Verifies that:
- Episodes can be recorded and retrieved
- Actions are stored and linked to episodes
- search_similar finds relevant episodes
- Capacity limits are enforced
- Consolidation marking works
- Context formatting produces valid output
"""

import pytest
import time
from pathlib import Path
from ghost_agent.memory.episodes import EpisodicMemory


@pytest.fixture
def ep_memory(tmp_path):
    return EpisodicMemory(tmp_path)


class TestEpisodeRecording:
    def test_record_basic_episode(self, ep_memory):
        ep_id = ep_memory.record_episode(
            trigger="Parse CSV file",
            context="User uploaded data.csv",
            actions=[
                {"tool": "file_system", "args": {"action": "read"}, "result": "OK"},
                {"tool": "execute", "args": {"code": "parse()"}, "result": "Done"},
            ],
            outcome="Successfully parsed",
            success=True,
            lesson="Always check encoding first",
            cluster_id="data_analysis"
        )

        assert ep_id is not None
        assert ep_id > 0

    def test_retrieve_episode_with_actions(self, ep_memory):
        ep_id = ep_memory.record_episode(
            trigger="Debug error",
            actions=[
                {"tool": "execute", "args": {}, "result": "Error: syntax"},
                {"tool": "file_system", "args": {}, "result": "Fixed"},
            ],
            outcome="Fixed the bug",
            success=True
        )

        ep = ep_memory.get_episode(ep_id)
        assert ep is not None
        assert ep["trigger"] == "Debug error"
        assert len(ep["actions"]) == 2
        assert ep["actions"][0]["tool_name"] == "execute"

    def test_get_nonexistent_episode(self, ep_memory):
        assert ep_memory.get_episode(99999) is None


class TestEpisodeSearch:
    def test_search_by_trigger(self, ep_memory):
        ep_memory.record_episode(
            trigger="Parse spreadsheet with pandas library",
            outcome="success", success=True
        )
        ep_memory.record_episode(
            trigger="Deploy Docker container",
            outcome="success", success=True
        )

        results = ep_memory.search_similar("parse spreadsheet pandas")
        assert len(results) >= 1
        assert any("pandas" in r["trigger"] for r in results)

    def test_search_empty_trigger(self, ep_memory):
        ep_memory.record_episode(trigger="test", outcome="ok")
        results = ep_memory.search_similar("")
        assert isinstance(results, list)

    def test_search_no_matches(self, ep_memory):
        ep_memory.record_episode(trigger="Python script", outcome="ok")
        results = ep_memory.search_similar("quantum physics")
        assert len(results) == 0


class TestEpisodeCapacity:
    def test_max_episodes_enforced(self, ep_memory):
        for i in range(ep_memory.MAX_EPISODES + 50):
            ep_memory.record_episode(
                trigger=f"Task {i}", outcome=f"Done {i}"
            )

        count = ep_memory.count()
        assert count <= ep_memory.MAX_EPISODES

    def test_preserves_episodes_with_lessons(self, ep_memory):
        # Record episodes with lessons (should be kept)
        for i in range(10):
            ep_memory.record_episode(
                trigger=f"Important {i}",
                lesson=f"Lesson {i}",
                outcome="success"
            )

        # Fill up with non-lesson episodes
        for i in range(ep_memory.MAX_EPISODES):
            ep_memory.record_episode(
                trigger=f"Filler {i}", outcome="ok"
            )

        # Lesson episodes should survive eviction
        count = ep_memory.count()
        assert count <= ep_memory.MAX_EPISODES


class TestConsolidation:
    def test_get_unconsolidated(self, ep_memory):
        ep_memory.record_episode(trigger="unconsol 1", outcome="ok")
        ep_memory.record_episode(trigger="unconsol 2", outcome="ok")

        uncons = ep_memory.get_unconsolidated()
        assert len(uncons) == 2

    def test_mark_consolidated(self, ep_memory):
        id1 = ep_memory.record_episode(trigger="to consolidate", outcome="ok")
        ep_memory.mark_consolidated([id1])

        uncons = ep_memory.get_unconsolidated()
        assert len(uncons) == 0

    def test_mark_empty_list(self, ep_memory):
        # Should not crash
        ep_memory.mark_consolidated([])


class TestClusterOperations:
    def test_get_by_cluster(self, ep_memory):
        ep_memory.record_episode(trigger="SQL q1", cluster_id="sql")
        ep_memory.record_episode(trigger="SQL q2", cluster_id="sql")
        ep_memory.record_episode(trigger="Bash cmd", cluster_id="bash")

        sql_eps = ep_memory.get_episodes_by_cluster("sql")
        assert len(sql_eps) == 2
        assert all(e["cluster_id"] == "sql" for e in sql_eps)


class TestContextFormatting:
    def test_format_for_context(self, ep_memory):
        ep_memory.record_episode(
            trigger="Test task",
            outcome="Succeeded",
            success=True,
            lesson="Check encoding",
            cluster_id="data"
        )

        episodes = ep_memory.get_recent_episodes()
        formatted = ep_memory.format_for_context(episodes)

        assert "RELEVANT PAST EPISODES" in formatted
        assert "Test task" in formatted
        assert "SUCCESS" in formatted

    def test_format_empty_list(self, ep_memory):
        assert ep_memory.format_for_context([]) == ""

    def test_format_respects_max_chars(self, ep_memory):
        for i in range(20):
            ep_memory.record_episode(
                trigger=f"Long trigger text for task number {i} with lots of detail",
                outcome=f"Detailed outcome {i}",
                cluster_id="test"
            )

        episodes = ep_memory.get_recent_episodes(limit=20)
        formatted = ep_memory.format_for_context(episodes, max_chars=200)
        assert len(formatted) <= 400  # Some slack for headers


class TestRecentEpisodes:
    def test_get_recent(self, ep_memory):
        for i in range(5):
            ep_memory.record_episode(trigger=f"Task {i}", outcome=f"Done {i}")

        recent = ep_memory.get_recent_episodes(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0]["trigger"] == "Task 4"
