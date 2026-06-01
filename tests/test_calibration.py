"""Tests for the closed-loop confidence calibration spine.

Covers core/calibration.py (record / Brier / ECE / reliability table /
grid-search refit / persistence / fail-safety) and the
core/confidence.py integration (apply_fitted + uncertainty-pressure
fusion) and core/uncertainty.py pressure().
"""

import json
import os

import pytest

from ghost_agent.core.calibration import (
    CalibrationTracker,
    FittedParams,
    SCHEMA_VERSION,
    _best_threshold,
)
from ghost_agent.core.confidence import CompositeConfidence, ConfidenceReading
from ghost_agent.core.uncertainty import UncertaintyTracker


# ──────────────────────────────────────────────────────────────────────
# recording + metrics
# ──────────────────────────────────────────────────────────────────────

def test_record_and_brier(tmp_path):
    t = CalibrationTracker(tmp_path / "calib")
    # Perfectly calibrated extremes → Brier 0.
    t.record(composite=1.0, entropy_component=1.0, competence_component=1.0, outcome=1.0)
    t.record(composite=0.0, entropy_component=0.0, competence_component=0.0, outcome=0.0)
    assert t.sample_count() == 2
    assert t.brier_score() == pytest.approx(0.0)


def test_brier_none_when_empty(tmp_path):
    t = CalibrationTracker(tmp_path / "calib")
    assert t.brier_score() is None
    assert t.ece() is None
    assert t.reliability_table() and all(b.count == 0 for b in t.reliability_table())


def test_brier_worst_case(tmp_path):
    t = CalibrationTracker(tmp_path / "calib")
    # Confidently wrong both ways → Brier 1.0.
    t.record(composite=1.0, entropy_component=1.0, competence_component=1.0, outcome=0.0)
    t.record(composite=0.0, entropy_component=0.0, competence_component=0.0, outcome=1.0)
    assert t.brier_score() == pytest.approx(1.0)


def test_reliability_table_bins_and_ece(tmp_path):
    t = CalibrationTracker(tmp_path / "calib")
    # Overconfident: composite 0.9 but only 50% actually succeed.
    for i in range(10):
        t.record(composite=0.9, entropy_component=0.9, competence_component=0.9,
                 outcome=1.0 if i < 5 else 0.0)
    table = t.reliability_table(bins=10)
    populated = [b for b in table if b.count]
    assert len(populated) == 1
    b = populated[0]
    assert b.count == 10
    assert b.mean_confidence == pytest.approx(0.9)
    assert b.mean_outcome == pytest.approx(0.5)
    # ECE is the gap between confidence and outcome (0.4 here).
    assert t.ece(bins=10) == pytest.approx(0.4, abs=1e-6)


def test_window_limits_samples(tmp_path):
    t = CalibrationTracker(tmp_path / "calib")
    for _ in range(5):
        t.record(composite=0.0, entropy_component=0.0, competence_component=0.0, outcome=0.0)
    for _ in range(5):
        t.record(composite=1.0, entropy_component=1.0, competence_component=1.0, outcome=1.0)
    # Last 5 are all the perfect-1.0 records → Brier 0.
    assert t.brier_score(window=5) == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────
# fitting
# ──────────────────────────────────────────────────────────────────────

def test_fit_bails_on_thin_data(tmp_path):
    t = CalibrationTracker(tmp_path / "calib", min_samples_for_fit=40)
    for i in range(10):
        t.record(composite=0.5, entropy_component=0.5, competence_component=0.5,
                 outcome=1.0 if i % 2 == 0 else 0.0)
    assert t.fit() is None
    assert not t.params_path.exists()  # no checkpoint written on bail


def test_fit_bails_on_single_class(tmp_path):
    t = CalibrationTracker(tmp_path / "calib", min_samples_for_fit=10)
    for _ in range(50):
        t.record(composite=0.7, entropy_component=0.7, competence_component=0.7, outcome=1.0)
    assert t.fit() is None
    assert not t.params_path.exists()


def test_fit_prefers_predictive_signal(tmp_path):
    """When competence predicts the outcome and entropy is noise, the
    fit should up-weight competence."""
    t = CalibrationTracker(tmp_path / "calib", min_samples_for_fit=20)
    for i in range(100):
        comp = 0.9 if i % 2 == 0 else 0.1
        t.record(
            composite=comp,
            entropy_component=0.5,          # uninformative constant
            competence_component=comp,      # carries the signal
            outcome=1.0 if comp > 0.5 else 0.0,
        )
    params = t.fit()
    assert params is not None
    assert params.schema == SCHEMA_VERSION
    assert params.w_competence > params.w_entropy
    assert params.brier < 0.05
    # Threshold should land between the two composite clusters.
    assert 0.1 < params.threshold < 0.9
    assert t.params_path.exists()


def test_fit_learns_uncertainty_penalty(tmp_path):
    """When verbalised-uncertainty pressure predicts failure and the
    objective components are flat, the fit should pick λ > 0."""
    t = CalibrationTracker(tmp_path / "calib", min_samples_for_fit=20)
    for i in range(100):
        high_pressure = (i % 2 == 0)
        t.record(
            composite=0.8,
            entropy_component=0.8,
            competence_component=0.8,
            uncertainty_pressure=1.0 if high_pressure else 0.0,
            outcome=0.0 if high_pressure else 1.0,
        )
    params = t.fit()
    assert params is not None
    assert params.lambda_uncertainty > 0.0


def test_fit_roundtrip_persistence(tmp_path):
    t = CalibrationTracker(tmp_path / "calib", min_samples_for_fit=20)
    for i in range(60):
        comp = 0.8 if i % 2 == 0 else 0.2
        t.record(composite=comp, entropy_component=comp, competence_component=comp,
                 outcome=1.0 if comp > 0.5 else 0.0)
    params = t.fit()
    assert params is not None
    loaded = t.load_params()
    assert loaded is not None
    assert loaded.w_entropy == pytest.approx(params.w_entropy)
    assert loaded.threshold == pytest.approx(params.threshold)
    assert loaded.lambda_uncertainty == pytest.approx(params.lambda_uncertainty)


# ──────────────────────────────────────────────────────────────────────
# persistence / fail-safety
# ──────────────────────────────────────────────────────────────────────

def test_load_params_absent(tmp_path):
    t = CalibrationTracker(tmp_path / "calib")
    assert t.load_params() is None


def test_load_params_wrong_schema(tmp_path):
    t = CalibrationTracker(tmp_path / "calib")
    t.dir.mkdir(parents=True, exist_ok=True)
    t.params_path.write_text(json.dumps({"schema": "ghost.calibration.v999",
                                         "w_entropy": 0.3, "w_competence": 0.7,
                                         "threshold": 0.5}))
    assert t.load_params() is None  # wrong schema → degrade to defaults


def test_load_params_corrupt(tmp_path):
    t = CalibrationTracker(tmp_path / "calib")
    t.dir.mkdir(parents=True, exist_ok=True)
    t.params_path.write_text("{not valid json")
    assert t.load_params() is None


def test_malformed_history_lines_skipped(tmp_path):
    t = CalibrationTracker(tmp_path / "calib")
    t.record(composite=0.6, entropy_component=0.6, competence_component=0.6, outcome=1.0)
    with t.history_path.open("a") as fh:
        fh.write("garbage not json\n")
        fh.write("\n")
    t.record(composite=0.4, entropy_component=0.4, competence_component=0.4, outcome=0.0)
    assert t.sample_count() == 2  # two valid, garbage skipped


def test_record_never_raises_on_bad_dir(tmp_path):
    t = CalibrationTracker(tmp_path / "calib")
    # Make the history path itself a directory so open('a') fails.
    t.dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(t.history_path)
    # Must not raise.
    t.record(composite=0.5, entropy_component=0.5, competence_component=0.5, outcome=1.0)
    assert t.sample_count() == 0


def test_record_clamps_and_binarizes(tmp_path):
    t = CalibrationTracker(tmp_path / "calib")
    t.record(composite=5.0, entropy_component=-1.0, competence_component=0.5, outcome=0.7)
    s = t._load_samples()[0]
    assert s.composite == 1.0
    assert s.entropy_component == 0.0
    assert s.outcome == 1.0  # 0.7 ≥ 0.5 → success


def test_stats_shape(tmp_path):
    t = CalibrationTracker(tmp_path / "calib", min_samples_for_fit=20)
    for i in range(40):
        comp = 0.8 if i % 2 == 0 else 0.2
        t.record(composite=comp, entropy_component=comp, competence_component=comp,
                 outcome=1.0 if comp > 0.5 else 0.0)
    t.fit()
    s = t.stats()
    assert s["samples"] == 40
    assert s["fitted"] is True
    assert 0.0 <= s["threshold"] <= 1.0


# ──────────────────────────────────────────────────────────────────────
# _best_threshold
# ──────────────────────────────────────────────────────────────────────

def test_best_threshold_separable():
    pairs = [(0.9, 1.0)] * 10 + [(0.1, 0.0)] * 10
    tau = _best_threshold(pairs)
    assert 0.1 < tau < 0.9


def test_best_threshold_single_class():
    assert _best_threshold([(0.9, 1.0)] * 5) == 0.55  # neutral fallback


# ──────────────────────────────────────────────────────────────────────
# CompositeConfidence integration
# ──────────────────────────────────────────────────────────────────────

def test_apply_fitted_swaps_params():
    cc = CompositeConfidence()
    assert cc.threshold == pytest.approx(0.55)
    params = FittedParams(w_entropy=0.2, w_competence=0.8, threshold=0.42,
                          lambda_uncertainty=0.3, brier=0.1, n_samples=100,
                          fitted_at="")
    cc.apply_fitted(params)
    assert cc.threshold == pytest.approx(0.42)
    assert cc.w_competence == pytest.approx(0.8)
    assert cc.lambda_uncertainty == pytest.approx(0.3)


def test_apply_fitted_defensive_on_garbage():
    cc = CompositeConfidence()
    before = (cc.w_entropy, cc.threshold)
    cc.apply_fitted(object())  # no attributes → untouched
    assert (cc.w_entropy, cc.threshold) == before


def test_uncertainty_pressure_penalizes_composite():
    # λ=0 → no-op (back-compat); λ>0 → pressure pulls composite down.
    base = CompositeConfidence(w_entropy=1.0, w_competence=0.0, lambda_uncertainty=0.0)
    r0 = base.score(normalised_entropy=0.0, competence_p_success=0.5,
                    uncertainty_pressure=1.0)
    assert r0.composite == pytest.approx(1.0)  # 1-e, no penalty

    pen = CompositeConfidence(w_entropy=1.0, w_competence=0.0, lambda_uncertainty=0.5)
    r1 = pen.score(normalised_entropy=0.0, competence_p_success=0.5,
                   uncertainty_pressure=1.0)
    assert r1.composite == pytest.approx(0.5)  # 1.0 * (1 - 0.5*1.0)
    assert r1.uncertainty_pressure == pytest.approx(1.0)


def test_confidence_reading_backcompat_positional():
    # Existing call sites build the reading positionally without the
    # new field — must still work with the default.
    r = ConfidenceReading(0.6, 0.7, 0.5, 0.55, False)
    assert r.uncertainty_pressure == 0.0


def test_neutral_entropy_is_competence_driven():
    """Logprob-optional path (calibration #1): when the entropy term is
    neutral (0.5, the value the agent passes when no/sparse logprobs),
    competence must still move the composite across the threshold — so a
    confidence reading (and thus a calibration sample) is produced on
    MTP/no-logprobs upstreams instead of being suppressed."""
    cc = CompositeConfidence(w_entropy=0.5, w_competence=0.5, threshold=0.55)
    # Strong competence (well-observed) with neutral entropy → above τ.
    hi = cc.score(normalised_entropy=0.5, competence_p_success=0.9, n_observations=50)
    # Weak competence with neutral entropy → below τ.
    lo = cc.score(normalised_entropy=0.5, competence_p_success=0.1, n_observations=50)
    assert hi.composite > lo.composite
    assert hi.below_threshold is False
    assert lo.below_threshold is True


# ──────────────────────────────────────────────────────────────────────
# UncertaintyTracker.pressure
# ──────────────────────────────────────────────────────────────────────

def test_pressure_zero_when_nothing_flagged():
    ut = UncertaintyTracker()
    assert ut.pressure() == 0.0


def test_pressure_monotone_in_impact():
    low = UncertaintyTracker()
    low.flag_unknown("x", impact=1)
    high = UncertaintyTracker()
    high.flag_unknown("y", impact=5)
    assert high.pressure() > low.pressure()
    assert 0.0 <= low.pressure() <= 1.0
    assert 0.0 <= high.pressure() <= 1.0


def test_pressure_drops_when_resolved():
    ut = UncertaintyTracker()
    u = ut.flag_unknown("z", impact=5)
    before = ut.pressure()
    ut.resolve_unknown(u, "answered")
    assert ut.pressure() < before
