"""Graded outcome labels + sample provenance — 2026-07-27 (later 9).

The binary label (`0.0 if anything broke else 1.0`) had two defects: it was
96.1% one class, and it measured the wrong thing. "Nothing visibly broke" is
not "the answer was good", and a turn that hit one tool error, recovered,
and answered correctly was labelled a FAILURE — identical to a refuted one.

Grading fixes the mislabel and gives every turn a value instead of only the
4% that break. The constants are measured: across 302 verdict-bearing
trajectories the agent passed 251, so an unverified-but-clean turn scores
0.83 — the observed P(good | checkable) — rather than an asserted 1.0.

`source` exists because the graded label is a PROXY while user corrections
are ground truth; mixing signal tiers without provenance is irreversible.
"""

import json
import tempfile
from pathlib import Path

import pytest

from ghost_agent.core.calibration import (
    CalibrationTracker, _UNVERIFIED_PRIOR, _outcome_variance,
    grade_turn_outcome,
)


class TestGrading:
    def test_verifier_verdicts_are_the_hard_anchors(self):
        assert grade_turn_outcome(verifier_verdict="failed") == 0.0
        assert grade_turn_outcome(verifier_verdict="passed") == 1.0

    def test_unverified_clean_turn_uses_the_measured_prior(self):
        """NOT 1.0 — claiming certainty for a turn nothing checked is the
        verification theatre this project keeps rediscovering."""
        assert grade_turn_outcome() == pytest.approx(_UNVERIFIED_PRIOR)
        assert 0.5 < _UNVERIFIED_PRIOR < 1.0

    def test_recovered_turn_is_not_scored_as_a_failure(self):
        """THE mislabel: one tool error followed by a correct answer used to
        score 0.0, the same as a refuted answer."""
        g = grade_turn_outcome(execution_failure_count=1)
        assert g > 0.5
        assert g < _UNVERIFIED_PRIOR

    def test_more_failures_score_worse_monotonically(self):
        grades = [grade_turn_outcome(execution_failure_count=n)
                  for n in range(0, 6)]
        assert grades == sorted(grades, reverse=True)
        assert grades[-1] > 0.0, "reserve 0.0 for a verdict of WRONG"

    def test_a_refuted_verdict_overrides_everything(self):
        assert grade_turn_outcome(verifier_verdict="failed",
                                  execution_failure_count=0) == 0.0
        assert grade_turn_outcome(verifier_verdict="failed",
                                  execution_failure_count=9) == 0.0

    def test_budget_exhaustion_is_a_partial(self):
        g = grade_turn_outcome(budget_exhausted=True)
        assert 0.0 < g < 0.5

    @pytest.mark.parametrize("kw", [
        {"execution_failure_count": "x"},
        {"execution_failure_count": None},
        {"verifier_verdict": object()},
        {"execution_failure_count": -5},
    ])
    def test_never_raises_and_stays_in_range(self, kw):
        assert 0.0 <= grade_turn_outcome(**kw) <= 1.0


class TestGradedStorage:
    def test_graded_values_are_not_binarised(self):
        """`1.0 if outcome >= 0.5 else 0.0` on both the write and read paths
        crushed every graded label back to two values — exactly the constant
        column the grading exists to remove."""
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=1)
            for v in (0.0, 0.15, 0.53, 0.68, 0.83, 1.0):
                t.record(composite=0.8, entropy_component=0.5,
                         competence_component=0.8, outcome=v)
            got = sorted(s.outcome for s in t._load_samples())
            assert got == [0.0, 0.15, 0.53, 0.68, 0.83, 1.0]

    def test_out_of_range_labels_are_clamped(self):
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=1)
            t.record(composite=0.8, entropy_component=0.5,
                     competence_component=0.8, outcome=42.0)
            assert t._load_samples()[0].outcome == 1.0


class TestProvenance:
    def test_source_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=1)
            t.record(composite=0.8, entropy_component=0.5,
                     competence_component=0.8, outcome=0.83, source="turn")
            t.record(composite=0.9, entropy_component=0.5,
                     competence_component=0.9, outcome=0.0,
                     source="user_correction")
            assert [s.source for s in t._load_samples()] == [
                "turn", "user_correction"]

    def test_legacy_rows_default_to_turn(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "calibration.jsonl").write_text(json.dumps({
                "composite": 0.9, "entropy_component": 0.5,
                "competence_component": 0.9, "uncertainty_pressure": 0.0,
                "outcome": 1.0, "ts": "2026-07-01T00:00:00Z"}) + "\n")
            s = CalibrationTracker(p, min_samples_for_fit=1)._load_samples()[0]
            assert s.source == "turn"

    def test_ground_truth_stays_separable_from_the_proxy(self):
        """The whole point: if the two tiers were indistinguishable you
        could never drop or reweight one without discarding the corpus."""
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=1)
            for _ in range(5):
                t.record(composite=0.8, entropy_component=0.5,
                         competence_component=0.8, outcome=0.83, source="turn")
            t.record(composite=0.9, entropy_component=0.5,
                     competence_component=0.9, outcome=0.0,
                     source="user_correction")
            samples = t._load_samples()
            truth = [s for s in samples if s.source == "user_correction"]
            assert len(truth) == 1 and truth[0].outcome == 0.0


class TestVarianceGate:
    def test_binary_single_class_still_bails(self):
        """Backward compatibility: for 0/1 labels, zero variance is exactly
        'one class missing', so the old behaviour is preserved."""
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=10)
            for _ in range(50):
                t.record(composite=0.8, entropy_component=0.5,
                         competence_component=0.8, outcome=1.0)
            assert t.fit() is None

    def test_graded_corpus_above_half_still_fits(self):
        """The old class-presence gate refused to fit a graded corpus whose
        every sample sat above 0.5, even though it carries real signal."""
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=10)
            for i in range(120):
                t.record(composite=0.5 + (i % 5) / 10.0,
                         entropy_component=0.5,
                         competence_component=0.5 + (i % 5) / 10.0,
                         outcome=0.6 + (i % 5) * 0.08)
            p = t.fit()
            assert p is not None, "graded corpus must be fittable"

    def test_variance_helper_matches_the_binary_intuition(self):
        from ghost_agent.core.calibration import CalibrationSample

        def mk(o):
            return CalibrationSample(composite=0.5, entropy_component=0.5,
                                     competence_component=0.5,
                                     uncertainty_pressure=0.0, outcome=o)
        assert _outcome_variance([mk(1.0)] * 10) == 0.0
        assert _outcome_variance([mk(1.0), mk(0.0)]) > 0.0
        assert _outcome_variance([mk(0.83), mk(0.68)]) > 0.0
