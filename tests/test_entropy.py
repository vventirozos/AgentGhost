"""Unit tests for ghost_agent.core.entropy."""

import math

import pytest

from ghost_agent.core.entropy import (
    EntropyReading,
    EntropyTracker,
    compute_token_entropy,
    extract_top_logprobs,
    normalise_entropy,
)


# ──────────────────────────────────────────────────────────────────────
# compute_token_entropy
# ──────────────────────────────────────────────────────────────────────

class TestComputeTokenEntropy:
    def test_empty_returns_zero(self):
        assert compute_token_entropy([]) == 0.0

    def test_single_top_logprob_is_zero(self):
        # Only one option → deterministic → entropy 0
        assert compute_token_entropy([0.0]) == pytest.approx(0.0, abs=1e-9)

    def test_uniform_distribution_maxes_out(self):
        # 5 equal logprobs → entropy ~ log(5)
        # All equal in log space → equal in prob space after renorm
        lps = [-1.6094] * 5  # log(1/5) = -1.609
        h = compute_token_entropy(lps)
        assert h == pytest.approx(math.log(5), abs=1e-3)

    def test_peaked_distribution_lower_than_uniform(self):
        # One dominant token, four long tails
        peaked = [-0.01, -10.0, -10.0, -10.0, -10.0]
        uniform = [-1.6094] * 5
        assert compute_token_entropy(peaked) < compute_token_entropy(uniform)

    def test_non_finite_inputs_skipped(self):
        # NaN/inf must not crash
        h = compute_token_entropy([float("nan"), float("inf"), -1.0, -1.0])
        assert math.isfinite(h)

    def test_clamps_extreme_logprobs(self):
        # A -1e9 logprob would overflow exp without the floor — verify safe
        h = compute_token_entropy([-1e9, -1.0])
        assert math.isfinite(h)

    def test_non_numeric_inputs_skipped(self):
        h = compute_token_entropy(["nope", None, -1.0, -1.0])  # type: ignore
        assert math.isfinite(h)


# ──────────────────────────────────────────────────────────────────────
# normalise_entropy
# ──────────────────────────────────────────────────────────────────────

class TestNormaliseEntropy:
    def test_zero_maps_to_zero(self):
        assert normalise_entropy(0.0, k=5) == 0.0

    def test_log_k_maps_to_one(self):
        assert normalise_entropy(math.log(5), k=5) == pytest.approx(1.0)

    def test_clamps_to_unit_interval(self):
        assert normalise_entropy(99.0, k=5) == 1.0
        assert normalise_entropy(-1.0, k=5) == 0.0

    def test_k_coerced_to_two(self):
        # Degenerate K should not divide by zero
        assert 0.0 <= normalise_entropy(0.5, k=1) <= 1.0


# ──────────────────────────────────────────────────────────────────────
# EntropyTracker
# ──────────────────────────────────────────────────────────────────────

class TestEntropyTracker:
    def test_empty_reading(self):
        tracker = EntropyTracker(window=4)
        r = tracker.reading()
        assert r == EntropyReading(raw=0.0, norm=0.0, n=0)

    def test_rolling_window_size(self):
        tracker = EntropyTracker(window=3)
        for _ in range(10):
            tracker.observe([-1.6094] * 5)
        assert tracker.reading().n == 3

    def test_running_mean_includes_all_observations(self):
        tracker = EntropyTracker(window=2)
        for _ in range(5):
            tracker.observe([-1.6094] * 5)
        # Running mean is over 5 observations, window only over 2
        assert tracker._total_observed == 5
        assert tracker.running_mean() == pytest.approx(math.log(5), abs=1e-3)

    def test_observe_returns_normalised_value(self):
        tracker = EntropyTracker(window=4, top_k=5)
        v = tracker.observe([-1.6094] * 5)
        assert v == pytest.approx(1.0, abs=1e-3)

    def test_reset_clears_state(self):
        tracker = EntropyTracker(window=4)
        for _ in range(3):
            tracker.observe([-1.6094] * 5)
        tracker.reset()
        assert tracker._total_observed == 0
        assert tracker.reading().n == 0

    def test_high_entropy_vs_low_entropy(self):
        """A peaked distribution should produce a lower mean than a
        uniform one, even on the same number of tokens."""
        peaked_tracker = EntropyTracker(window=10)
        uniform_tracker = EntropyTracker(window=10)
        for _ in range(8):
            peaked_tracker.observe([-0.01, -10.0, -10.0, -10.0, -10.0])
            uniform_tracker.observe([-1.6094] * 5)
        assert peaked_tracker.reading().norm < uniform_tracker.reading().norm


# ──────────────────────────────────────────────────────────────────────
# extract_top_logprobs
# ──────────────────────────────────────────────────────────────────────

class TestExtractTopLogprobs:
    def test_standard_openai_shape(self):
        chunk = {
            "choices": [{
                "logprobs": {
                    "content": [{
                        "token": "x",
                        "logprob": -0.1,
                        "top_logprobs": [
                            {"token": "x", "logprob": -0.1},
                            {"token": "y", "logprob": -2.0},
                        ],
                    }],
                },
            }]
        }
        out = extract_top_logprobs(chunk)
        assert out == [-0.1, -2.0]

    def test_llama_cpp_flat_shape(self):
        chunk = {
            "choices": [{
                "logprobs": {"top_logprobs": [[-0.1, -2.0, -3.0]]},
            }]
        }
        out = extract_top_logprobs(chunk)
        assert out == [-0.1, -2.0, -3.0]

    def test_missing_returns_none(self):
        assert extract_top_logprobs({"choices": [{}]}) is None
        assert extract_top_logprobs({}) is None
        assert extract_top_logprobs(None) is None  # type: ignore

    def test_malformed_does_not_raise(self):
        # Random shape — should return None, not crash
        out = extract_top_logprobs({"choices": [{"logprobs": "broken"}]})
        assert out is None
