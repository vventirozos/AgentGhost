"""Probability recalibration + feature-health diagnosis — 2026-07-27 (later 4).

The confidence composite was measured against the only baseline that
matters — always predicting the base rate — and lost:

    raw composite  Brier 0.0656
    base rate      Brier 0.0374

The weight search decides how to COMBINE signals but cannot set their
level, so a Platt stage was added. On the live corpus it is REJECTED, and
that rejection is the point: the leak-free AUC is 0.473, i.e. the composite
carries no discrimination, so the optimiser converges to a ≈ −0.08 —
a negative slope that would INVERT the agent's confidence ordering, and a
near-flat one that would collapse every turn onto the base rate and leave
`below_threshold` permanently inert. Calibrating noise is not an
improvement; the guard refuses and says so.

The machinery still engages the moment a predictive feature exists — pinned
below by a corpus that is informative but badly scaled.
"""

import random
import tempfile
from pathlib import Path

import pytest

from ghost_agent.core.calibration import (
    CalibrationTracker, _fit_platt, apply_platt,
)
from ghost_agent.core.confidence import CompositeConfidence, _apply_platt
from ghost_agent.core.learning_health import _feature_health


def _fit(build, *, seed=5, floor=50):
    random.seed(seed)
    with tempfile.TemporaryDirectory() as d:
        t = CalibrationTracker(Path(d), min_samples_for_fit=floor)
        build(t)
        return t.fit()


# ──────────────────────────────────────────────────────────────────────
# The two implementations of the map must agree
# ──────────────────────────────────────────────────────────────────────

class TestMapConsistency:
    @pytest.mark.parametrize("c", [0.0, 0.13, 0.5, 0.87, 1.0])
    @pytest.mark.parametrize("ab", [(1.0, 0.0), (6.35, -1.6), (47.6, -23.8)])
    def test_scorer_and_fitter_compute_the_same_function(self, c, ab):
        """`confidence._apply_platt` is a local copy so the scorer doesn't
        import the fitting module. If they ever diverge the agent evaluates
        a different function than the one that was fitted."""
        a, b = ab
        assert _apply_platt(c, a, b) == pytest.approx(apply_platt(c, a, b), abs=1e-12)

    def test_map_is_overflow_safe(self):
        for a, b in ((1e6, 1e6), (-1e6, -1e6), (1e9, 0.0)):
            v = apply_platt(0.5, a, b)
            assert 0.0 <= v <= 1.0

    def test_map_is_monotonic_for_positive_slope(self):
        prev = -1.0
        for i in range(11):
            v = apply_platt(i / 10.0, 6.0, -3.0)
            assert v >= prev
            prev = v


# ──────────────────────────────────────────────────────────────────────
# Convergence must not depend on the iteration budget
# ──────────────────────────────────────────────────────────────────────

class TestFitConvergence:
    def test_converges_regardless_of_budget(self):
        """Gradient descent was still at a=1.28 after 4000 steps and only
        reached its optimum (a≈-0.08) after ~60000 — values on OPPOSITE
        sides of the safety guard. Newton must land in the same place from
        any budget above a handful of steps."""
        random.seed(11)
        pairs = [(random.random(), 1.0 if random.random() < 0.9 else 0.0)
                 for _ in range(400)]
        a20, b20 = _fit_platt(pairs, iters=20)
        a200, b200 = _fit_platt(pairs, iters=200)
        assert a20 == pytest.approx(a200, abs=1e-6)
        assert b20 == pytest.approx(b200, abs=1e-6)

    def test_single_class_returns_identity(self):
        assert _fit_platt([(0.5, 1.0)] * 40) == (1.0, 0.0)
        assert _fit_platt([(0.5, 0.0)] * 40) == (1.0, 0.0)

    def test_empty_input_is_safe(self):
        assert _fit_platt([]) == (1.0, 0.0)


# ──────────────────────────────────────────────────────────────────────
# Adoption / rejection policy
# ──────────────────────────────────────────────────────────────────────

class TestAdoptionPolicy:
    def test_rejected_when_the_score_does_not_discriminate(self):
        """The live shape: composite varies but is unrelated to outcome.
        The optimum is to ignore it, which the guard must refuse."""
        def build(t):
            for _ in range(400):
                comp = random.random()
                t.record(composite=comp, entropy_component=0.5,
                         competence_component=comp,
                         outcome=1.0 if random.random() < 0.96 else 0.0,
                         entropy_observed=False)
        p = _fit(build)
        assert (p.platt_a, p.platt_b) == (1.0, 0.0), "must not calibrate noise"

    def test_never_adopts_an_inverting_map(self):
        """A negative slope would report LOWER confidence on turns the score
        rates higher — actively worse than doing nothing."""
        def build(t):
            for _ in range(400):
                comp = random.random()
                # deliberately anti-correlated
                t.record(composite=comp, entropy_component=0.5,
                         competence_component=comp,
                         outcome=1.0 if random.random() > comp else 0.0,
                         entropy_observed=False)
        p = _fit(build)
        assert p.platt_a >= 0.0
        assert (p.platt_a, p.platt_b) == (1.0, 0.0)

    def test_adopted_when_informative_but_miscalibrated(self):
        """Real signal compressed into [0.45, 0.55] — exactly what the stage
        exists to fix."""
        def build(t):
            for _ in range(800):
                true_p = random.random()
                comp = 0.45 + 0.10 * true_p
                t.record(composite=comp, entropy_component=0.5,
                         competence_component=comp,
                         outcome=1.0 if random.random() < true_p else 0.0,
                         entropy_observed=False)
        p = _fit(build)
        assert p.platt_a > 1.0, "should stretch the compressed range"
        assert p.brier < p.brier_raw
        assert p.brier < p.brier_base_rate, "must beat the base rate"

    def test_baseline_brier_is_always_recorded(self):
        def build(t):
            for _ in range(200):
                t.record(composite=0.9, entropy_component=0.5,
                         competence_component=0.9,
                         outcome=1.0 if random.random() < 0.9 else 0.0,
                         entropy_observed=False)
        p = _fit(build)
        assert p.brier_base_rate >= 0.0
        assert p.brier_raw >= 0.0


# ──────────────────────────────────────────────────────────────────────
# Scorer integration
# ──────────────────────────────────────────────────────────────────────

class TestScorerIntegration:
    def test_identity_params_leave_scoring_untouched(self):
        cc = CompositeConfidence()
        before = cc.score(competence_p_success=0.9, n_observations=100).composite
        cc.apply_fitted(type("P", (), {
            "w_entropy": 0.0, "w_competence": 1.0, "threshold": 0.7,
            "lambda_uncertainty": 0.0, "platt_a": 1.0, "platt_b": 0.0})())
        after = cc.score(competence_p_success=0.9, n_observations=100).composite
        assert before == pytest.approx(after)

    def test_fitted_map_is_applied(self):
        cc = CompositeConfidence()
        raw = cc.score(competence_p_success=0.9, n_observations=100).composite
        cc.apply_fitted(type("P", (), {
            "w_entropy": 0.0, "w_competence": 1.0, "threshold": 0.9,
            "lambda_uncertainty": 0.0, "platt_a": 6.0, "platt_b": -3.0})())
        mapped = cc.score(competence_p_success=0.9, n_observations=100).composite
        assert mapped != pytest.approx(raw)
        assert mapped == pytest.approx(apply_platt(raw, 6.0, -3.0), abs=1e-9)

    def test_map_preserves_ordering(self):
        """Monotonic by construction — a recalibration must never reorder
        two turns' relative confidence."""
        cc = CompositeConfidence()
        cc.apply_fitted(type("P", (), {
            "w_entropy": 0.0, "w_competence": 1.0, "threshold": 0.9,
            "lambda_uncertainty": 0.0, "platt_a": 6.0, "platt_b": -3.0})())
        lo = cc.score(competence_p_success=0.3, n_observations=100).composite
        hi = cc.score(competence_p_success=0.95, n_observations=100).composite
        assert hi > lo


# ──────────────────────────────────────────────────────────────────────
# Feature health
# ──────────────────────────────────────────────────────────────────────

class TestFeatureHealth:
    def test_constant_feature_is_dead(self):
        s = [{"entropy_component": 0.5, "competence_component": 0.5,
              "uncertainty_pressure": 0.0, "outcome": float(i % 2)}
             for i in range(50)]
        fh = _feature_health(s)["feature_health"]
        assert fh["uncertainty_pressure"]["dead"] is True
        assert fh["entropy_component"]["dead"] is True

    def test_varying_but_non_separating_feature_is_dead(self):
        """The live competence case: many distinct values, separation
        ≈0 — varies plenty, predicts nothing. Built with the SAME value
        spread in both outcome classes so separation is exactly zero
        rather than merely small by chance."""
        vals = [i / 100.0 for i in range(1, 100)]
        s = ([{"entropy_component": 0.5, "competence_component": v,
               "uncertainty_pressure": 0.0, "outcome": 1.0} for v in vals]
             + [{"entropy_component": 0.5, "competence_component": v,
                 "uncertainty_pressure": 0.0, "outcome": 0.0} for v in vals])
        fh = _feature_health(s)["feature_health"]
        assert fh["competence_component"]["distinct"] > 50
        assert fh["competence_component"]["separation"] == pytest.approx(0.0, abs=1e-9)
        assert fh["competence_component"]["dead"] is True

    def test_separating_feature_is_live(self):
        s = ([{"entropy_component": 0.5, "competence_component": 0.9,
               "uncertainty_pressure": 0.0, "outcome": 1.0}] * 100
             + [{"entropy_component": 0.5, "competence_component": 0.2,
                 "uncertainty_pressure": 0.0, "outcome": 0.0}] * 100)
        out = _feature_health(s)
        assert out["feature_health"]["competence_component"]["dead"] is False
        assert "competence_component" in out["live_features"]

    def test_empty_and_junk_are_safe(self):
        assert _feature_health([]) == {}
        _feature_health([{"outcome": "x", "competence_component": None}])
