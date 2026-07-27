"""Turn-effort: the first per-TURN confidence feature — 2026-07-27 (later 5).

Before this, every input to the composite was dead: entropy had 2 distinct
values, uncertainty pressure had 1, and competence — the only one that
varied — separated success from failure by −0.0008. Competence is a
per-DOMAIN historical average, so it is near-constant within a domain and
structurally cannot discriminate individual turns. The composite's
leak-free AUC was 0.473, below chance, and the fit correctly refused to
calibrate it.

Turn SHAPE does discriminate. Measured over 296 labelled trajectories:
passed turns averaged 2.4 tool calls with a longest same-tool run of 1.4;
failed turns averaged 11.7 and 5.8. The combined feature scores AUC 0.670
with separation +0.232, and end-to-end it takes the fitted model from
losing to the base rate by 75% to BEATING it by 12%.
"""

import json
import tempfile
from pathlib import Path

import pytest

from ghost_agent.core.calibration import CalibrationTracker, _composite_for
from ghost_agent.core.confidence import CompositeConfidence, effort_component


# ──────────────────────────────────────────────────────────────────────
# The feature itself
# ──────────────────────────────────────────────────────────────────────

class TestEffortComponent:
    def test_short_clean_turn_scores_high(self):
        assert effort_component(["file_system"]) > 0.85

    def test_sprawling_turn_scores_low(self):
        many = [f"tool_{i}" for i in range(14)]
        assert effort_component(many) < 0.5

    def test_spinning_on_one_tool_scores_low(self):
        assert effort_component(["browser"] * 8) < 0.3

    def test_monotonic_in_call_count(self):
        prev = 1.1
        for n in (1, 3, 6, 9, 12, 20):
            v = effort_component([f"t{i}" for i in range(n)])
            assert v <= prev
            prev = v

    def test_repetition_is_penalised_beyond_raw_count(self):
        """Six calls to the SAME tool is a worse signal than six distinct
        ones — spinning, not breadth."""
        distinct = effort_component([f"t{i}" for i in range(6)])
        same = effort_component(["t"] * 6)
        assert same < distinct

    def test_bounded_and_safe(self):
        for arg in ([], None, [None, ""], ["a"] * 500):
            v = effort_component(arg)
            assert 0.0 <= v <= 1.0

    def test_matches_the_measured_regimes(self):
        """Sanity-check against the corpus means the thresholds came from:
        a typical PASSED turn (≈2 calls, no repeats) must score well above a
        typical FAILED one (≈12 calls with a run of ≈6)."""
        passed = effort_component(["file_system", "execute"])
        failed = effort_component(["browser"] * 6 + [f"t{i}" for i in range(6)])
        assert passed - failed > 0.3


# ──────────────────────────────────────────────────────────────────────
# Scorer integration
# ──────────────────────────────────────────────────────────────────────

class TestScorerIntegration:
    def test_absent_effort_leaves_scoring_unchanged(self):
        """A turn that ran no tools has no effort EVIDENCE. It must not be
        recorded as a measured 0.5 — that is the fabricated-neutral bug that
        made entropy unfittable."""
        cc = CompositeConfidence()
        cc.w_effort = 0.5
        cc.w_competence = 0.5
        r = cc.score(competence_p_success=0.9, n_observations=100)
        assert r.effort_observed is False
        assert r.composite == pytest.approx(r.competence_component, abs=1e-9)

    def test_observed_effort_is_blended(self):
        cc = CompositeConfidence()
        cc.w_effort, cc.w_competence, cc.w_entropy = 0.5, 0.5, 0.0
        r = cc.score(competence_p_success=0.9, n_observations=10_000, effort=0.1)
        assert r.effort_observed is True
        assert r.effort_component == pytest.approx(0.1)
        # halfway between competence (~0.9) and effort (0.1)
        assert r.composite == pytest.approx(0.5, abs=0.02)

    def test_weights_renormalise_over_available_components(self):
        """With entropy unobserved, competence+effort must renormalise to 1
        rather than leaving a third of the mass on a missing feature."""
        cc = CompositeConfidence()
        cc.w_entropy, cc.w_competence, cc.w_effort = 0.4, 0.3, 0.3
        r = cc.score(competence_p_success=0.8, n_observations=10_000, effort=0.8)
        # Both AVAILABLE components sit at ~0.8 (competence is very slightly
        # shrunk toward 0.5 by the n/(n+5) prior), so the composite must land
        # on that same value — NOT dragged toward 0.5 by the 0.4 of weight
        # belonging to the absent entropy term, which is what would happen
        # without renormalisation.
        expected = (0.3 * r.competence_component + 0.3 * 0.8) / 0.6
        assert r.composite == pytest.approx(expected, abs=1e-9)
        # Prove the renormalisation is load-bearing: WITHOUT it the 0.4 of
        # weight belonging to the absent entropy term would be spent on the
        # neutral 0.5 stand-in and drag the result well below 0.79.
        unrenormalised = 0.4 * 0.5 + 0.3 * r.competence_component + 0.3 * 0.8
        assert unrenormalised < 0.68
        assert r.composite > 0.79

    def test_default_weight_is_zero_until_fitted(self):
        cc = CompositeConfidence()
        assert cc.w_effort == 0.0
        before = cc.score(competence_p_success=0.9, n_observations=100).composite
        after = cc.score(competence_p_success=0.9, n_observations=100,
                         effort=0.0).composite
        assert before == pytest.approx(after), "unfitted effort must not move the score"


# ──────────────────────────────────────────────────────────────────────
# Fit / scorer agreement and the evidence gate
# ──────────────────────────────────────────────────────────────────────

def _sample(**kw):
    from ghost_agent.core.calibration import CalibrationSample
    base = dict(composite=0.5, entropy_component=0.5, competence_component=0.8,
                uncertainty_pressure=0.0, outcome=1.0, entropy_observed=False,
                effort_component=0.5, effort_observed=False)
    base.update(kw)
    return CalibrationSample(**base)


class TestFitAgreement:
    def test_composite_for_matches_scorer(self):
        cc = CompositeConfidence()
        cc.w_entropy, cc.w_competence, cc.w_effort = 0.0, 0.6, 0.4
        r = cc.score(competence_p_success=0.8, n_observations=10_000, effort=0.2)
        s = _sample(competence_component=r.competence_component,
                    effort_component=0.2, effort_observed=True)
        assert _composite_for(s, 0.0, 0.0, 0.4) == pytest.approx(r.composite, abs=1e-6)

    def test_unobserved_effort_excluded_from_blend(self):
        s = _sample(competence_component=0.8, effort_component=0.0,
                    effort_observed=False)
        # w_eff would otherwise drag this toward 0
        assert _composite_for(s, 0.0, 0.0, 0.9) == pytest.approx(0.8, abs=1e-9)

    def test_weight_pinned_until_enough_observations(self):
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=10)
            for i in range(100):
                t.record(composite=0.8, entropy_component=0.5,
                         competence_component=0.8,
                         outcome=1.0 if i % 5 else 0.0,
                         effort_component=0.9 if i % 5 else 0.1,
                         effort_observed=i < 5)      # only 5 observed
            p = t.fit()
            assert p.w_effort == 0.0, "must not fit a weight on 5 samples"

    def test_weight_earned_when_effort_predicts(self):
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=50)
            for i in range(400):
                ok = i % 4 != 0
                t.record(composite=0.8, entropy_component=0.5,
                         competence_component=0.8,          # constant, useless
                         outcome=1.0 if ok else 0.0,
                         effort_component=0.9 if ok else 0.1,  # informative
                         effort_observed=True)
            p = t.fit()
            assert p.w_effort > 0.0
            assert p.n_effort_observed == 400
            assert p.brier < p.brier_base_rate, "should now beat the base rate"

    def test_persisted_and_reloaded(self):
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=50)
            for i in range(200):
                ok = i % 3 != 0
                t.record(composite=0.8, entropy_component=0.5,
                         competence_component=0.8, outcome=1.0 if ok else 0.0,
                         effort_component=0.9 if ok else 0.1, effort_observed=True)
            fitted = t.fit()
            loaded = t.load_params()
            assert loaded.w_effort == fitted.w_effort
            assert loaded.n_effort_observed == fitted.n_effort_observed

    def test_legacy_sample_without_effort_fields_loads(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "calibration.jsonl").write_text(json.dumps({
                "composite": 0.9, "entropy_component": 0.5,
                "competence_component": 0.9, "uncertainty_pressure": 0.0,
                "outcome": 1.0, "domain": "", "ts": "2026-07-01T00:00:00Z",
            }) + "\n")
            s = CalibrationTracker(p, min_samples_for_fit=1)._load_samples()
            assert s[0].effort_observed is False
            assert s[0].effort_component == pytest.approx(0.5)
