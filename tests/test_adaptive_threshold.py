"""Tests for adaptive smart-memory threshold (#3).

Verifies that:
- Threshold adjusts based on useful/useless observations
- Threshold respects floor and ceiling bounds
- State persists across instances
- Insufficient observations don't trigger recalculation
"""

import json

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


def _closed_loop_record(at, score):
    """Feed one observation the way core.agent wires the gate: was_useful
    is approximated by 'cleared the current bar' (the self-referential
    signal that produced the 2026-07 ratchet lockup)."""
    accepted = score >= at.get_threshold()
    at.record(score, was_useful=accepted)
    return accepted


class TestRatchetLockup:
    """A burst of high-scoring facts failing CONTENT gates used to push
    median_useless ~0.9, snap the threshold near CEILING in one window,
    and lock it there: every later fact scored below the new bar, was
    recorded as high-score useless, and sustained the clamp. Recovery
    needed ~100 consecutive useful-free records."""

    def test_content_reject_burst_does_not_ratchet_gate(self, threshold):
        # Warm-up: a healthy stream of 0.75 facts that pass everything.
        for _ in range(25):
            assert _closed_loop_record(threshold, 0.75)
        warm = threshold.get_threshold()
        assert warm < 0.75

        # Burst: high-scoring facts rejected by content gates downstream.
        for _ in range(15):
            threshold.record(0.92, False)

        after = threshold.get_threshold()
        # Old code snapped to ~0.92 on the first recalculation here.
        assert after < 0.9
        assert after <= warm + 15 * AdaptiveThreshold.MAX_STEP_UP + 1e-9
        # The healthy stream is still accepted — no lockup.
        assert _closed_loop_record(threshold, 0.75)

    def test_upward_move_rate_limited_per_update(self, threshold):
        for _ in range(19):
            threshold.record(0.99, True)
        prev = threshold.get_threshold()
        for _ in range(6):  # recalculation starts at MIN_OBSERVATIONS
            threshold.record(0.99, True)
            cur = threshold.get_threshold()
            assert cur - prev <= AdaptiveThreshold.MAX_STEP_UP + 1e-9
            prev = cur
        # It rose (adaptation works), just not in one snap to 0.891.
        assert 0.7 < prev < 0.99 * 0.9

    def test_starvation_decay_reachable_with_stale_useful_in_window(self, threshold):
        # Stale useful entries at 0.9 hold min_useful high…
        for _ in range(25):
            threshold.record(0.9, True)
        elevated = threshold.get_threshold()
        assert elevated > 0.7

        # …then the scorer only emits below-bar facts. Old code kept the
        # clamp branch in charge until ALL useful entries flushed from the
        # 100-slot window; now STARVATION_RUN consecutive useless records
        # reach the decay path.
        for _ in range(40):
            threshold.record(0.5, False)
        assert threshold.get_threshold() < 0.7
        assert threshold.get_threshold() >= AdaptiveThreshold.FLOOR

    def test_locked_gate_recovers_in_closed_loop(self, tmp_path):
        # Simulate a gate already ratcheted near CEILING (pre-fix damage),
        # with stale useful entries still in the window.
        at = AdaptiveThreshold(tmp_path, initial=0.7)
        at.threshold = 0.93
        at.window.extend([(0.95, True, True)] * 10 + [(0.85, False, False)] * 20)

        accepted_at = None
        for i in range(60):
            if _closed_loop_record(at, 0.8):
                accepted_at = i
                break
        # Old code needed ~100 useful-free records before decay could start.
        assert accepted_at is not None and accepted_at < 60

    def test_legacy_two_tuple_state_still_loads(self, tmp_path):
        legacy = {
            "threshold": 0.65,
            "window": [[0.8, True], [0.5, False], [0.9, False]],
        }
        (tmp_path / "adaptive_threshold.json").write_text(json.dumps(legacy))

        at = AdaptiveThreshold(tmp_path, initial=0.7)
        assert at.get_threshold() == 0.65
        stats = at.get_stats()
        assert stats["observations"] == 3 and stats["useful_count"] == 1
        # Mixed 2-/3-tuple windows must recalculate without raising.
        for _ in range(25):
            at.record(0.7, True)
        assert AdaptiveThreshold.FLOOR <= at.get_threshold() <= AdaptiveThreshold.CEILING
