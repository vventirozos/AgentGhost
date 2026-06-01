"""Tests for online PRM/router updates (partial_fit + guarded online_update).

The math is exercised with full-dimension synthetic vectors (the fit()
report indexes weights by the frozen feature-name list, so the vector
width must match: 25 for the PRM, 17 for the router). Only the first
feature carries signal; the rest are zero.
"""

import numpy as np
import pytest

from ghost_agent.prm.model import StepValueModel
from ghost_agent.prm.scorer import PRMScorer
from ghost_agent.prm.trainer import samples_to_xy
from ghost_agent.prm.features import PRM_FEATURE_NAMES
from ghost_agent.router.model import ComplexityClassifier
from ghost_agent.router.features import FEATURE_NAMES
from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome

PRM_DIM = len(PRM_FEATURE_NAMES)
R_DIM = len(FEATURE_NAMES)


def _pv(first):
    return [float(first)] + [0.0] * (PRM_DIM - 1)


def _rv(first):
    return [float(first)] + [0.0] * (R_DIM - 1)


def _fit_prm():
    X = [_pv(0), _pv(1)] * 20
    y = [0.0, 1.0] * 20
    m = StepValueModel(epochs=200)
    m.fit(X, y)
    return m


# ──────────────────────────────────────────────────────────────────────
# partial_fit / clone / bce_loss
# ──────────────────────────────────────────────────────────────────────

def test_partial_fit_requires_fitted_model():
    m = StepValueModel()
    with pytest.raises(RuntimeError):
        m.partial_fit([_pv(1)], [1.0])


def test_partial_fit_moves_prediction():
    m = _fit_prm()
    before = m.predict_proba(_pv(1))
    m.partial_fit([_pv(1)], [0.0], lr=0.5, steps=30)
    after = m.predict_proba(_pv(1))
    assert after < before


def test_clone_is_independent():
    m = _fit_prm()
    w_before = m.weights_.copy()
    c = m.clone()
    c.partial_fit([_pv(1)], [0.0], lr=0.5, steps=30)
    assert np.allclose(m.weights_, w_before)        # original untouched
    assert not np.allclose(c.weights_, w_before)    # clone moved


def test_bce_loss_finite_and_discriminative():
    m = _fit_prm()
    good = m.bce_loss([_pv(0), _pv(1)], [0.0, 1.0])
    bad = m.bce_loss([_pv(0), _pv(1)], [1.0, 0.0])
    assert good < bad
    assert np.isfinite(good)


# ──────────────────────────────────────────────────────────────────────
# PRMScorer.online_update guard
# ──────────────────────────────────────────────────────────────────────

def test_online_update_no_model_returns_false():
    s = PRMScorer()  # empty
    assert s.online_update([_pv(1)], [1.0]) is False


def test_online_update_commits_consistent_step():
    s = PRMScorer(model=_fit_prm())
    holdout_X = [_pv(0), _pv(1)] * 5
    holdout_y = [0.0, 1.0] * 5
    ok = s.online_update([_pv(1)], [1.0],
                         holdout_X=holdout_X, holdout_y=holdout_y)
    assert ok is True


def test_online_update_rejects_forgetting_step():
    s = PRMScorer(model=_fit_prm())
    model_before = s.model
    holdout_X = [_pv(0), _pv(1)] * 10
    holdout_y = [0.0, 1.0] * 10
    # Contradicts the holdout, pushed hard → worsens holdout BCE → rejected.
    ok = s.online_update([_pv(1)], [0.0],
                         holdout_X=holdout_X, holdout_y=holdout_y,
                         lr=0.8, max_steps=40)
    assert ok is False
    assert s.model is model_before  # live model not swapped


def test_online_update_without_holdout_applies_bounded_step():
    s = PRMScorer(model=_fit_prm())
    assert s.online_update([_pv(1)], [1.0]) is True


# ──────────────────────────────────────────────────────────────────────
# router mirror (string labels via LABEL_TO_INT)
# ──────────────────────────────────────────────────────────────────────

def test_router_partial_fit_and_clone():
    X = [_rv(0), _rv(1)] * 20
    y = ["easy", "hard"] * 20
    m = ComplexityClassifier(epochs=200)
    m.fit(X, y)
    before = m.predict_proba(_rv(1))  # p(hard)
    c = m.clone()
    c.partial_fit([_rv(1)], ["easy"], lr=0.5, steps=30)
    assert c.predict_proba(_rv(1)) < before
    assert m.predict_proba(_rv(1)) == pytest.approx(before)  # original untouched


def test_router_bce_loss_string_labels():
    X = [_rv(0), _rv(1)] * 20
    y = ["easy", "hard"] * 20
    m = ComplexityClassifier(epochs=200)
    m.fit(X, y)
    loss = m.bce_loss([_rv(0), _rv(1)], ["easy", "hard"])
    assert np.isfinite(loss) and loss >= 0.0


# ──────────────────────────────────────────────────────────────────────
# samples_to_xy
# ──────────────────────────────────────────────────────────────────────

def test_samples_to_xy_from_trajectories():
    failed = Trajectory(
        user_request="parse the log",
        tool_calls=[ToolCall(name="execute"), ToolCall(name="file_system")],
        outcome=Outcome.FAILED.value,
    )
    passed = Trajectory(
        user_request="count words",
        tool_calls=[ToolCall(name="execute")],
        outcome=Outcome.PASSED.value,
    )
    X, y = samples_to_xy([failed, passed])
    assert len(X) == len(y) > 0
    assert min(y) == pytest.approx(0.0, abs=1e-6)  # FAILED steps → ~0
    assert max(y) > 0.5                            # PASSED terminal step → ~1
