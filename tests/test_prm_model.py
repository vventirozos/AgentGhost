"""Tests for prm.model.StepValueModel."""

import json
from pathlib import Path

import numpy as np
import pytest

from ghost_agent.prm.features import (
    PRM_FEATURE_NAMES,
    ActionFeatures,
    PlanState,
    extract_step_features,
)
from ghost_agent.prm.model import PRMTrainingReport, StepValueModel


# ──────────────────────────────────────────────────────────────────────
# Helpers — synthetic separable data
# ──────────────────────────────────────────────────────────────────────

def _state(**kw):
    base = dict(
        user_request="",
        steps_so_far=0,
        failures_so_far=0,
        pending_count=0,
        plan_depth=0,
        tools_used_this_turn=(),
        tools_failed_this_turn=(),
    )
    base.update(kw)
    return PlanState(**base)


def _action(tool_name="", description="", tool_args=None):
    return ActionFeatures(
        description=description,
        tool_name=tool_name,
        tool_args=tool_args or {},
    )


def _good_step_features():
    """Lightweight tool, no failures, fresh state — 'looks good'."""
    return extract_step_features(
        _state(user_request="hi"),
        _action(tool_name="scratchpad"),
    )


def _bad_step_features():
    """Heavyweight tool already failed this turn — 'looks bad'."""
    return extract_step_features(
        _state(
            user_request="x" * 500,
            failures_so_far=3,
            tools_failed_this_turn=("execute",),
        ),
        _action(tool_name="execute"),
    )


def _training_data():
    """Deliberately separable: good vs bad shapes."""
    pos = [_good_step_features() for _ in range(8)]
    neg = [_bad_step_features() for _ in range(8)]
    X = pos + neg
    y = [1] * len(pos) + [0] * len(neg)
    return X, y


# ──────────────────────────────────────────────────────────────────────
# Fit
# ──────────────────────────────────────────────────────────────────────

def test_fit_returns_self():
    X, y = _training_data()
    m = StepValueModel(epochs=200)
    out = m.fit(X, y)
    assert out is m


def test_fit_populates_report():
    X, y = _training_data()
    m = StepValueModel(epochs=200).fit(X, y)
    assert isinstance(m.report_, PRMTrainingReport)
    assert m.report_.n_samples == len(y)
    assert m.report_.n_features == len(PRM_FEATURE_NAMES)
    assert m.report_.class_counts["positive"] == 8
    assert m.report_.class_counts["negative"] == 8
    assert set(m.report_.weights.keys()) == set(PRM_FEATURE_NAMES)


def test_fit_separates_clearly_distinct_classes():
    X, y = _training_data()
    m = StepValueModel(epochs=500).fit(X, y)
    # On clearly separable synthetic data, the model should at least
    # learn the gradient. We're not chasing 100% — just "better than
    # random by a wide margin."
    assert m.report_.train_accuracy >= 0.9


def test_fit_empty_raises():
    with pytest.raises(ValueError):
        StepValueModel().fit([], [])


def test_fit_single_class_binary_raises():
    fv = _good_step_features()
    with pytest.raises(ValueError):
        StepValueModel().fit([fv, fv], [1, 1])


def test_fit_accepts_continuous_soft_labels():
    """Soft labels (e.g., MC-discounted values from labels.derive_step_labels)
    must be accepted — that's how the trainer's continuous-mode works."""
    X, _ = _training_data()
    # Mix of soft labels in [0, 1].
    y_soft = [0.9, 0.95, 1.0, 0.85, 0.7, 0.8, 1.0, 0.75,
              0.0, 0.05, 0.1, 0.0, 0.15, 0.0, 0.0, 0.05]
    m = StepValueModel(epochs=200).fit(X, y_soft)
    # Just survives without raising and produces finite weights.
    assert np.all(np.isfinite(m.weights_))
    assert np.isfinite(m.bias_)


def test_fit_clamps_labels_outside_unit_interval():
    """Label values outside [0, 1] would yield malformed cross-entropy.
    Trainer-side clamp keeps them in range silently — confirm the
    fit still produces finite weights."""
    X, _ = _training_data()
    y = [1.5] * 8 + [-0.3] * 8
    m = StepValueModel(epochs=100).fit(X, y)
    assert np.all(np.isfinite(m.weights_))


# ──────────────────────────────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────────────────────────────

def test_predict_proba_bounded_in_unit_interval():
    X, y = _training_data()
    m = StepValueModel(epochs=200).fit(X, y)
    for fv in X:
        p = m.predict_proba(fv)
        assert 0.0 <= p <= 1.0


def test_predict_value_via_state_action():
    X, y = _training_data()
    m = StepValueModel(epochs=300).fit(X, y)
    p_good = m.predict_value(
        _state(user_request="hi"),
        _action(tool_name="scratchpad"),
    )
    p_bad = m.predict_value(
        _state(failures_so_far=3, tools_failed_this_turn=("execute",)),
        _action(tool_name="execute"),
    )
    assert 0.0 <= p_good <= 1.0
    assert 0.0 <= p_bad <= 1.0
    # Direction: the "good" shape should score higher than the "bad" shape.
    assert p_good > p_bad


def test_predict_proba_accepts_list_input():
    X, y = _training_data()
    m = StepValueModel(epochs=200).fit(X, y)
    raw = list(X[0].values)
    p = m.predict_proba(raw)
    assert 0.0 <= p <= 1.0


def test_predict_proba_accepts_ndarray_input():
    X, y = _training_data()
    m = StepValueModel(epochs=200).fit(X, y)
    raw = np.array(X[0].values, dtype=float)
    p = m.predict_proba(raw)
    assert 0.0 <= p <= 1.0


def test_untrained_predict_raises():
    m = StepValueModel()
    with pytest.raises(RuntimeError):
        m.predict_proba(_good_step_features())


def test_untrained_predict_value_raises():
    m = StepValueModel()
    with pytest.raises(RuntimeError):
        m.predict_value(_state(), _action())


def test_predict_unsupported_input_type_raises():
    X, y = _training_data()
    m = StepValueModel(epochs=200).fit(X, y)
    with pytest.raises(TypeError):
        m.predict_proba(object())


# ──────────────────────────────────────────────────────────────────────
# Sigmoid clipping (overflow safety)
# ──────────────────────────────────────────────────────────────────────

def test_sigmoid_clipping_stable_for_extreme_weights():
    m = StepValueModel()
    m.weights_ = np.ones(len(PRM_FEATURE_NAMES)) * 1e6
    m.bias_ = -1e6
    p = m.predict_proba(_good_step_features())
    assert 0.0 <= p <= 1.0


# ──────────────────────────────────────────────────────────────────────
# Save / load
# ──────────────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path: Path):
    X, y = _training_data()
    m = StepValueModel(epochs=300, random_state=42).fit(X, y)
    path = tmp_path / "prm.json"
    m.save(path)
    assert path.exists()

    payload = json.loads(path.read_text())
    assert payload["schema"] == "ghost.prm.logreg.v1"
    assert payload["feature_names"] == list(PRM_FEATURE_NAMES)
    assert "weights" in payload
    assert "bias" in payload

    m2 = StepValueModel.load(path)
    # Predictions must match exactly.
    test_inputs = [
        _good_step_features(),
        _bad_step_features(),
        extract_step_features(_state(steps_so_far=2), _action(tool_name="browser")),
    ]
    for fv in test_inputs:
        assert m.predict_proba(fv) == pytest.approx(m2.predict_proba(fv), abs=1e-12)


def test_save_untrained_raises(tmp_path: Path):
    m = StepValueModel()
    with pytest.raises(RuntimeError):
        m.save(tmp_path / "x.json")


def test_load_unknown_schema_raises(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({
        "schema": "something.else",
        "weights": [],
        "bias": 0.0,
        "feature_names": list(PRM_FEATURE_NAMES),
    }))
    with pytest.raises(ValueError, match="unknown PRM schema"):
        StepValueModel.load(p)


def test_load_feature_drift_raises(tmp_path: Path):
    """A checkpoint trained against an older feature set must NOT
    silently load — that would mis-align weights against the current
    features and produce confidently-wrong scores."""
    p = tmp_path / "drift.json"
    drifted = ("only_one_feature",)
    p.write_text(json.dumps({
        "schema": "ghost.prm.logreg.v1",
        "feature_names": list(drifted),
        "weights": [0.5],
        "bias": 0.0,
    }))
    with pytest.raises(ValueError, match="feature schema drift"):
        StepValueModel.load(p)


def test_save_uses_atomic_replace(tmp_path: Path):
    """Save writes through a .tmp path then replaces, so an interrupt
    can't leave a half-written checkpoint."""
    X, y = _training_data()
    m = StepValueModel(epochs=200).fit(X, y)
    path = tmp_path / "atomic.json"
    m.save(path)
    assert path.exists()
    # No leftover .tmp
    assert not (tmp_path / "atomic.json.tmp").exists()
