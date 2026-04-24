"""Tests for self-play curriculum scaffolding (#7).

Verifies that:
- Difficulty tiers progress from basic to expert
- Tiers unlock after mastering enough challenges
- Difficulty hints are included in pick_seed output
"""

import pytest
from pathlib import Path
from ghost_agent.memory.frontier import (
    FrontierTracker,
    DIFFICULTY_TIERS,
    DIFFICULTY_HINTS,
    TIER_UNLOCK_THRESHOLD,
)


@pytest.fixture
def tracker(tmp_path):
    return FrontierTracker(tmp_path)


class TestDifficultyTiers:
    def test_new_cluster_starts_at_basic(self, tracker):
        tier = tracker.get_difficulty_tier("sql")
        assert tier == "basic"

    def test_tier_progresses_with_first_try_wins(self, tracker):
        # Record enough first-try wins to unlock intermediate
        for i in range(TIER_UNLOCK_THRESHOLD):
            tracker.record_run(
                "sql", f"Challenge {i}", attempts_used=1,
                passed=True, description_length=100
            )

        tier = tracker.get_difficulty_tier("sql")
        assert tier == "intermediate"

    def test_tier_progresses_to_advanced(self, tracker):
        for i in range(TIER_UNLOCK_THRESHOLD * 2):
            tracker.record_run(
                "sql", f"Challenge {i}", attempts_used=1,
                passed=True, description_length=100
            )

        tier = tracker.get_difficulty_tier("sql")
        assert tier == "advanced"

    def test_tier_caps_at_expert(self, tracker):
        for i in range(TIER_UNLOCK_THRESHOLD * 10):
            tracker.record_run(
                "sql", f"Challenge {i}", attempts_used=1,
                passed=True, description_length=100
            )

        tier = tracker.get_difficulty_tier("sql")
        assert tier == "expert"

    def test_failures_dont_count_for_progression(self, tracker):
        # Mix of wins and failures
        tracker.record_run("bash", "C1", attempts_used=1, passed=True, description_length=100)
        tracker.record_run("bash", "C2", attempts_used=3, passed=False, description_length=0)
        tracker.record_run("bash", "C3", attempts_used=2, passed=True, description_length=100)

        tier = tracker.get_difficulty_tier("bash")
        # Only 2 first-try wins, not enough for intermediate
        assert tier == "basic"


class TestDifficultyHints:
    def test_get_difficulty_hint(self, tracker):
        hint = tracker.get_difficulty_hint("new_cluster")
        assert "BASIC" in hint

    def test_all_tiers_have_hints(self):
        for tier in DIFFICULTY_TIERS:
            assert tier in DIFFICULTY_HINTS


class TestPickSeedWithDifficulty:
    def test_pick_seed_includes_difficulty(self, tracker):
        # Create a brittle cluster so pick_seed returns a frontier seed
        for i in range(3):
            tracker.record_run(
                "sql", f"Challenge {i}", attempts_used=3,
                passed=False, description_length=0,
                mistake="Failed badly"
            )

        seed = tracker.pick_seed(random_explore_prob=0.0)

        if seed["mode"] == "frontier":
            assert "DIFFICULTY TIER" in seed["hint"]
            assert "difficulty_tier" in seed
            assert seed["difficulty_tier"] in DIFFICULTY_TIERS
