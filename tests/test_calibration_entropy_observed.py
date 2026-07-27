"""The entropy signal was structurally unfittable — 2026-07-27 (later 3).

Live measurement: 1200 of 1201 stored calibration samples had
``entropy_component`` exactly 0.5 and the fitted ``w_entropy`` was 0.0, so
the confidence composite ran on competence alone.

Root cause (confirmed by instrumenting the running agent): llama-server
rejects ``logprobs`` on ``tools + stream`` payloads, and the turn loop
attaches tools on every non-final generation — so the overwhelming majority
of turns observe NO token logprobs. Those turns recorded the neutral 0.5
stand-in as if it were a measurement. With zero variance in the entropy
column, any ``w_e > 0`` could only drag composites toward 0.5, so the grid
search was guaranteed to pick 0 forever.

The fix has three parts, pinned below:
  1. readings/samples carry ``entropy_observed``;
  2. an unobserved sample scores on competence ALONE (missing-feature
     renormalisation) instead of being blended with the stand-in;
  3. the weight only moves once enough real observations of both outcome
     classes exist.
"""

import random
import tempfile
from pathlib import Path

import pytest

from ghost_agent.core.calibration import (
    CalibrationTracker, _MIN_ENTROPY_SAMPLES, _composite_for, CalibrationSample,
)
from ghost_agent.core.confidence import CompositeConfidence


def _sample(*, ent, comp, outcome, observed, pressure=0.0):
    return CalibrationSample(
        composite=0.5, entropy_component=ent, competence_component=comp,
        uncertainty_pressure=pressure, outcome=outcome,
        entropy_observed=observed,
    )


# ──────────────────────────────────────────────────────────────────────
# The reading itself must distinguish measured from fabricated
# ──────────────────────────────────────────────────────────────────────

class TestReadingMarksObservation:
    def test_none_entropy_is_flagged_unobserved(self):
        r = CompositeConfidence().score(
            normalised_entropy=None, competence_p_success=0.9,
            n_observations=100)
        assert r.entropy_observed is False
        # Still scored (neutral), never crashes.
        assert 0.0 <= r.composite <= 1.0

    def test_float_entropy_is_flagged_observed(self):
        r = CompositeConfidence().score(
            normalised_entropy=0.2, competence_p_success=0.9,
            n_observations=100)
        assert r.entropy_observed is True

    def test_unobserved_reading_scores_on_competence_alone(self):
        """The stand-in must not dilute the composite. With w_entropy > 0 a
        blended neutral pulled every logprob-less turn toward 0.5."""
        cc = CompositeConfidence()
        cc.w_entropy, cc.w_competence = 0.5, 0.5
        unobs = cc.score(normalised_entropy=None, competence_p_success=0.9,
                         n_observations=1000)
        # competence ~0.9 (shrink is negligible at n=1000), NOT ~0.7
        assert unobs.composite == pytest.approx(unobs.competence_component, abs=1e-9)
        assert unobs.composite > 0.85

    def test_observed_reading_still_blends(self):
        cc = CompositeConfidence()
        cc.w_entropy, cc.w_competence = 0.5, 0.5
        obs = cc.score(normalised_entropy=1.0, competence_p_success=0.9,
                       n_observations=1000)
        # entropy_component = 1 - 1.0 = 0.0 → blend pulls it down
        assert obs.composite < obs.competence_component

    def test_explicit_override_is_respected(self):
        r = CompositeConfidence().score(
            normalised_entropy=0.3, competence_p_success=0.9,
            entropy_observed=False)
        assert r.entropy_observed is False


# ──────────────────────────────────────────────────────────────────────
# Fit and scorer must agree on the formula
# ──────────────────────────────────────────────────────────────────────

class TestFitScorerAgreement:
    def test_composite_for_matches_scorer_when_unobserved(self):
        cc = CompositeConfidence()
        cc.w_entropy, cc.w_competence = 0.4, 0.6
        r = cc.score(normalised_entropy=None, competence_p_success=0.8,
                     n_observations=10_000)
        s = _sample(ent=r.entropy_component, comp=r.competence_component,
                    outcome=1.0, observed=False)
        assert _composite_for(s, 0.4, 0.0) == pytest.approx(r.composite, abs=1e-6)

    def test_composite_for_matches_scorer_when_observed(self):
        cc = CompositeConfidence()
        cc.w_entropy, cc.w_competence = 0.4, 0.6
        r = cc.score(normalised_entropy=0.25, competence_p_success=0.8,
                     n_observations=10_000)
        s = _sample(ent=r.entropy_component, comp=r.competence_component,
                    outcome=1.0, observed=True)
        assert _composite_for(s, 0.4, 0.0) == pytest.approx(r.composite, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────
# End-to-end fit behaviour
# ──────────────────────────────────────────────────────────────────────

def _fit(build, seed=7):
    random.seed(seed)
    with tempfile.TemporaryDirectory() as d:
        t = CalibrationTracker(Path(d), min_samples_for_fit=10)
        build(t)
        return t.fit()


class TestFitBehaviour:
    def test_all_unobserved_pins_weight_to_zero(self):
        """Today's live data shape — must not invent a weight."""
        def build(t):
            for _ in range(200):
                ok = random.random() < 0.9
                t.record(composite=0.89, entropy_component=0.5,
                         competence_component=0.89,
                         outcome=1.0 if ok else 0.0, entropy_observed=False)
        p = _fit(build)
        assert p.w_entropy == 0.0
        assert p.n_entropy_observed == 0

    def test_informative_entropy_becomes_learnable(self):
        """THE regression: before the fix this could never move off 0."""
        def build(t):
            for _ in range(200):
                ok = random.random() < 0.5
                t.record(composite=0.5,
                         entropy_component=0.9 if ok else 0.1,
                         competence_component=0.5,
                         outcome=1.0 if ok else 0.0, entropy_observed=True)
        p = _fit(build)
        assert p.w_entropy > 0.0, "informative entropy must earn weight"
        assert p.n_entropy_observed == 200

    def test_uninformative_observed_entropy_earns_no_weight(self):
        """Observed but pure noise → the grid should still prefer competence."""
        def build(t):
            for _ in range(200):
                ok = random.random() < 0.9
                t.record(composite=0.89, entropy_component=random.random(),
                         competence_component=0.89,
                         outcome=1.0 if ok else 0.0, entropy_observed=True)
        p = _fit(build)
        assert p.w_entropy == 0.0

    def test_observed_minority_does_not_degrade_the_majority(self):
        """The naive 'fit on the subset, apply to everything' version scored
        Brier 0.219 here because unobserved samples got blended with the
        stand-in. Renormalisation keeps the population honest."""
        def build(t):
            for _ in range(400):
                ok = random.random() < 0.9
                t.record(composite=0.89, entropy_component=0.5,
                         competence_component=0.89,
                         outcome=1.0 if ok else 0.0, entropy_observed=False)
            for _ in range(60):
                ok = random.random() < 0.5
                t.record(composite=0.5,
                         entropy_component=0.9 if ok else 0.1,
                         competence_component=0.5,
                         outcome=1.0 if ok else 0.0, entropy_observed=True)
        p = _fit(build)
        assert p.w_entropy > 0.0
        assert p.brier < 0.15, f"mixture Brier regressed: {p.brier}"

    def test_below_floor_stays_pinned(self):
        def build(t):
            for _ in range(200):
                ok = random.random() < 0.9
                t.record(composite=0.89, entropy_component=0.5,
                         competence_component=0.89,
                         outcome=1.0 if ok else 0.0, entropy_observed=False)
            for i in range(_MIN_ENTROPY_SAMPLES - 5):
                t.record(composite=0.5, entropy_component=0.9 if i % 2 else 0.1,
                         competence_component=0.5,
                         outcome=1.0 if i % 2 else 0.0, entropy_observed=True)
        p = _fit(build)
        assert p.w_entropy == 0.0, "must not fit a weight below the floor"

    def test_single_outcome_class_among_observed_stays_pinned(self):
        def build(t):
            for _ in range(200):
                ok = random.random() < 0.9
                t.record(composite=0.89, entropy_component=0.5,
                         competence_component=0.89,
                         outcome=1.0 if ok else 0.0, entropy_observed=False)
            for i in range(60):          # all successes → no contrast
                t.record(composite=0.5, entropy_component=random.random(),
                         competence_component=0.5, outcome=1.0,
                         entropy_observed=True)
        p = _fit(build)
        assert p.w_entropy == 0.0


# ──────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_flag_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=1)
            t.record(composite=0.5, entropy_component=0.3,
                     competence_component=0.7, outcome=1.0,
                     entropy_observed=True)
            t.record(composite=0.5, entropy_component=0.5,
                     competence_component=0.7, outcome=0.0,
                     entropy_observed=False)
            loaded = t._load_samples()
            assert [s.entropy_observed for s in loaded] == [True, False]

    def test_legacy_record_without_the_field_reads_as_unobserved(self):
        """Every pre-fix record WAS the fabricated neutral, so defaulting to
        False is the truthful interpretation — not a lossy assumption."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "calibration.jsonl").write_text(
                '{"composite":0.9,"entropy_component":0.5,'
                '"competence_component":0.9,"uncertainty_pressure":0.0,'
                '"outcome":1.0,"domain":"","ts":"2026-07-01T00:00:00Z"}\n')
            t = CalibrationTracker(p, min_samples_for_fit=1)
            loaded = t._load_samples()
            assert len(loaded) == 1
            assert loaded[0].entropy_observed is False
