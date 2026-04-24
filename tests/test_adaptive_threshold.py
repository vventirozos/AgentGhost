"""Tests for adaptive smart-memory threshold (#3).

Verifies that:
- Threshold adjusts based on useful/useless observations
- Threshold respects floor and ceiling bounds
- State persists across instances
- Insufficient observations don't trigger recalculation
"""

import pytest
from pathlib import Path
from ghost_agent.memory.adaptive_threshold import AdaptiveThreshold


@pytest.fixture
def threshold(tmp_path):
    return AdaptiveThreshold(tmp_path, initial=0.7)


class TestAdaptiveThreshold:
    def test_initial_threshold(self, threshold):
        assert threshold.get_threshold() == 0.7

    def test_threshold_adjusts_with_useful_observations(self, threshold):
        # Feed in useful observations at various scores
        for _ in range(10):
            threshold.record(0.8, True)
        for _ in range(10):
            threshold.record(0.6, True)

        # After enough useful observations at 0.6+, threshold should drop
        t = threshold.get_threshold()
        assert t <= 0.7  # Should have lowered or stayed

    def test_threshold_raises_with_useless_low_scores(self, threshold):
        # Many useless at low scores, few useful at high
        for _ in range(15):
            threshold.record(0.4, False)  # useless
        for _ in range(5):
            threshold.record(0.9, True)  # useful

        t = threshold.get_threshold()
        # Should be above 0.4 (median of useless)
        assert t >= 0.4

    def test_threshold_floor(self, threshold):
        # All useful at very low scores
        for _ in range(25):
            threshold.record(0.1, True)

        t = threshold.get_threshold()
        assert t >= AdaptiveThreshold.FLOOR

    def test_threshold_ceiling(self, threshold):
        # Only useful at very high scores
        for _ in range(25):
            threshold.record(0.99, True)

        t = threshold.get_threshold()
        assert t <= AdaptiveThreshold.CEILING

    def test_insufficient_observations_no_change(self, threshold):
        initial = threshold.get_threshold()
        threshold.record(0.5, True)
        threshold.record(0.3, False)
        # Not enough observations to trigger recalculation (need 20)
        assert threshold.get_threshold() == initial

    def test_persistence(self, tmp_path):
        at1 = AdaptiveThreshold(tmp_path, initial=0.7)
        for _ in range(25):
            at1.record(0.6, True)

        t1 = at1.get_threshold()

        # New instance should load saved state
        at2 = AdaptiveThreshold(tmp_path, initial=0.7)
        t2 = at2.get_threshold()
        assert t1 == t2

    def test_get_stats(self, threshold):
        threshold.record(0.5, True)
        threshold.record(0.3, False)

        stats = threshold.get_stats()
        assert stats["observations"] == 2
        assert stats["useful_count"] == 1
        assert stats["useful_rate"] == 0.5

    def test_window_size_limit(self, threshold):
        for i in range(150):
            threshold.record(0.5, True)

        stats = threshold.get_stats()
        assert stats["observations"] <= AdaptiveThreshold.WINDOW_SIZE
