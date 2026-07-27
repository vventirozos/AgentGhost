"""Second-pass review of the same day's calibration work — 2026-07-27 (later 7).

Six defects, all introduced earlier the same session and all invisible to a
142-green calibration suite because each needs a corpus shape the tests never
built (linearly separable, single-valued, or a fit that actually gets
adopted).

The theme: guards written from intuition rather than measurement. The slope
floor rejected good maps for a reason that was false; the "ridge" was
Levenberg damping and bounded nothing; the penalty/map ordering silently
made a documented guarantee unreachable.
"""

import tempfile
from pathlib import Path

import pytest

from ghost_agent.core.calibration import (
    CalibrationTracker, _best_threshold, _fit_platt,
)
from ghost_agent.core.confidence import CompositeConfidence


# ──────────────────────────────────────────────────────────────────────
# D1 — the fit must be bounded on separable data
# ──────────────────────────────────────────────────────────────────────

def _separable():
    return ([(0.30 + 0.01 * i, 0.0) for i in range(20)]
            + [(0.61 + 0.01 * i, 1.0) for i in range(20)])


class TestFitIsRegularised:
    def test_answer_does_not_depend_on_iteration_budget(self):
        """The docstring promises this. With ridge applied only to the
        Hessian it was FALSE on separable data: a grew 51 → 207 → 233 at
        5 → 50 → 200 iterations, because damping changes the step size but
        not the fixed point."""
        a5, b5 = _fit_platt(_separable(), iters=5)
        a50, b50 = _fit_platt(_separable(), iters=50)
        a200, b200 = _fit_platt(_separable(), iters=200)
        assert a5 == pytest.approx(a50, rel=1e-6)
        assert a50 == pytest.approx(a200, rel=1e-6)
        assert b50 == pytest.approx(b200, rel=1e-6)

    def test_separable_slope_stays_finite(self):
        a, b = _fit_platt(_separable())
        assert 0.0 < a < 50.0, f"separable fit diverged to a={a}"

    @pytest.mark.parametrize("pairs", [
        [(0.49, 0.0), (0.51, 1.0)],                       # 2 samples
        [(0.5, 1.0)] * 200 + [(0.1, 0.0)],                # single negative
        [(0.0, 0.0)] * 20 + [(0.0, 1.0)] * 20,            # all-identical
        [(1.0, 1.0)] * 20 + [(1.0, 0.0)] * 20,
    ])
    def test_pathological_corpora_stay_finite(self, pairs):
        a, b = _fit_platt(pairs)
        assert abs(a) < 1e3 and abs(b) < 1e3


# ──────────────────────────────────────────────────────────────────────
# D2/D3 — adoption guards
# ──────────────────────────────────────────────────────────────────────

class TestAdoptionGuards:
    def test_near_step_function_is_rejected(self):
        """A separable batch fitted in-sample scores a perfect Brier and
        would be adopted, collapsing confidence to a step on one competence
        value."""
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=10)
            for c, y in _separable():
                t.record(composite=c, entropy_component=0.5,
                         competence_component=c, outcome=y)
            p = t.fit()
            assert p.platt_a <= 50.0

    def test_small_positive_slope_is_NOT_rejected(self):
        """The old `slope >= 0.5` floor discarded a measured Brier
        improvement. Platt is monotone for a > 0 and the threshold is refit
        on the same scale, so the below-threshold decision set is unchanged
        — only calibration quality improves."""
        import random
        random.seed(4)
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=50)
            for _ in range(600):
                c = random.random()
                t.record(composite=c, entropy_component=0.5,
                         competence_component=c,
                         outcome=1.0 if random.random() < (0.40 + 0.12 * c) else 0.0)
            p = t.fit()
            if 0.0 < p.platt_a < 0.5:
                assert p.brier <= p.brier_raw, (
                    "a small positive slope that improves Brier must be kept")

    def test_inverting_map_still_rejected(self):
        """The one case that IS unsafe: a <= 0 flips the ordering."""
        import random
        random.seed(9)
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=50)
            for _ in range(400):
                c = random.random()
                t.record(composite=c, entropy_component=0.5,
                         competence_component=c,
                         outcome=1.0 if random.random() > c else 0.0)
            p = t.fit()
            assert p.platt_a > 0.0
            assert (p.platt_a, p.platt_b) == (1.0, 0.0)


# ──────────────────────────────────────────────────────────────────────
# D4 — w_effort == 1.0 must be adopted, not silently dropped
# ──────────────────────────────────────────────────────────────────────

class TestWeightAdoption:
    @pytest.mark.parametrize("w_e,w_c,w_eff", [
        (0.0, 0.0, 1.0),        # the dropped case
        (0.0, 0.5, 0.5),
        (0.6, 0.0, 0.4),
        (0.0, 0.1, 0.9),
    ])
    def test_weights_round_trip_and_sum_to_one(self, w_e, w_c, w_eff):
        cc = CompositeConfidence()
        cc.w_effort = 0.3            # a stale prior value
        cc.apply_fitted(type("P", (), {
            "w_entropy": w_e, "w_competence": w_c, "threshold": 0.5,
            "lambda_uncertainty": 0.0, "w_effort": w_eff,
            "platt_a": 1.0, "platt_b": 0.0})())
        assert cc.w_effort == pytest.approx(w_eff)
        total = cc.w_entropy + cc.w_competence + cc.w_effort
        assert total == pytest.approx(1.0, abs=1e-9)


# ──────────────────────────────────────────────────────────────────────
# D5 — the outcome penalty must always be able to cross the threshold
# ──────────────────────────────────────────────────────────────────────

class TestOutcomePenaltyReachesZero:
    def test_penalty_applies_after_the_map(self):
        """Applying it BEFORE let the map's intercept floor the result at
        sigmoid(b): with b = 0.27 even a full penalty bottomed out at 0.566,
        so `below_threshold` was unreachable for every possible input — while
        the code comment promised the opposite."""
        cc = CompositeConfidence()
        cc.platt_a, cc.platt_b, cc.threshold = 1.187, 0.2675, 0.5
        full = cc.score(competence_p_success=0.95, n_observations=1000,
                        outcome_penalty=1.0)
        assert full.composite == pytest.approx(0.0, abs=1e-9)
        assert full.below_threshold is True

    def test_refuted_verdict_crosses_the_threshold(self):
        cc = CompositeConfidence()
        cc.platt_a, cc.platt_b, cc.threshold = 6.0, -3.0, 0.5
        clean = cc.score(competence_p_success=0.95, n_observations=1000)
        refuted = cc.score(competence_p_success=0.95, n_observations=1000,
                           outcome_penalty=0.8)
        assert clean.below_threshold is False
        assert refuted.below_threshold is True

    def test_raw_scale_is_preserved_for_calibration(self):
        """The recorded column must stay on ONE scale across refits."""
        cc = CompositeConfidence()
        raw = cc.score(competence_p_success=0.8, n_observations=1000)
        cc.platt_a, cc.platt_b = 6.0, -3.0
        mapped = cc.score(competence_p_success=0.8, n_observations=1000)
        assert mapped.composite != pytest.approx(raw.composite)
        assert mapped.raw_pre_penalty_composite == pytest.approx(
            raw.raw_pre_penalty_composite, abs=1e-9)


# ──────────────────────────────────────────────────────────────────────
# D5b — threshold fallbacks must be scale-free
# ──────────────────────────────────────────────────────────────────────

class TestThresholdFallbacks:
    def test_single_class_fallback_sits_inside_the_observed_range(self):
        """A hardcoded 0.55 is meaningless when the caller is on the mapped
        probability scale."""
        pairs = [(0.90 + 0.001 * i, 1.0) for i in range(20)]
        tau = _best_threshold(pairs)
        assert 0.85 <= tau <= 1.0

    def test_uncorrelated_fallback_sits_inside_the_observed_range(self):
        pairs = [(0.90, float(i % 2)) for i in range(40)]
        tau = _best_threshold(pairs)
        assert 0.85 <= tau <= 0.95


# ──────────────────────────────────────────────────────────────────────
# D7 — stats() must not mix windows
# ──────────────────────────────────────────────────────────────────────

class TestStatsWindow:
    def test_brier_and_ece_describe_the_same_population(self):
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=10,
                                   max_history=50)
            # 300 well-calibrated, then 50 badly-calibrated at the tail
            for _ in range(300):
                t.record(composite=0.9, entropy_component=0.5,
                         competence_component=0.9, outcome=1.0)
            for _ in range(50):
                t.record(composite=0.9, entropy_component=0.5,
                         competence_component=0.9, outcome=0.0)
            st = t.stats()
            assert st["samples"] == 50
            # ECE over the same 50-tail must be large, not diluted by the
            # 300 older well-calibrated rows.
            assert st["ece"] > 0.5, (
                f"ece {st['ece']} looks computed over the whole file")
