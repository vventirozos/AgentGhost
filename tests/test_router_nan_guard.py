"""Router NaN guard (log-audit fix, 2026-06-11).

The production log showed router/model.py matmul overflow → divide-by-zero
→ invalid-value during the idle retrain, i.e. training diverging to
non-finite weights, which were then hot-swapped into the live dispatcher
(agent.py) with no finiteness check. A NaN classifier returns NaN
confidences and poisons routing. Guards now sit at every chokepoint:
fit() raises on divergence, partial_fit() reverts, load() rejects a
corrupt checkpoint, predict_proba() never emits NaN, and the hot-swap is
gated on is_finite().
"""

import numpy as np
import pytest

from ghost_agent.router.model import ComplexityClassifier, FEATURE_NAMES

NF = len(FEATURE_NAMES)


def _row(base):
    # a full-width (NF) feature row, separable by the first feature
    return [base] + [base * 0.5] * (NF - 1)


def _trainable():
    # finite samples, both classes present, correct feature width
    X = [_row(0.1), _row(0.9), _row(0.2), _row(0.8)]
    y = ["easy", "hard", "easy", "hard"]
    return X, y


def test_normal_fit_is_finite():
    clf = ComplexityClassifier(epochs=50)
    clf.fit(*_trainable())
    assert clf.is_finite()
    p = clf.predict_proba(_row(0.5))
    assert 0.0 <= p <= 1.0


def test_unfitted_is_not_finite():
    assert ComplexityClassifier().is_finite() is False


def test_fit_raises_on_divergence_with_huge_lr():
    # an absurd learning rate with the (default-style) L2 term drives the
    # exponential `(1 - lr·l2)·w` blowup to non-finite weights; fit must
    # raise rather than persist a NaN model (this is the production failure)
    clf = ComplexityClassifier(learning_rate=1e30, epochs=300, l2=1e-3)
    with pytest.raises(ValueError):
        clf.fit(*_trainable())


def test_fit_sanitises_non_finite_features():
    # an inf feature value must not poison training (mapped to 0)
    X = [_row(0.1), _row(0.9), _row(0.2), _row(0.8)]
    X[0][0] = float("inf")
    y = ["easy", "hard", "easy", "hard"]
    clf = ComplexityClassifier(epochs=50)
    clf.fit(X, y)
    assert clf.is_finite()


def test_predict_proba_neutral_on_nan_weights():
    clf = ComplexityClassifier(epochs=20)
    clf.fit(*_trainable())
    # force a poisoned state and confirm predict never emits NaN
    poisoned = clf.weights_.copy()
    poisoned[0] = np.nan
    clf.weights_ = poisoned
    assert clf.predict_proba(_row(0.5)) == 0.5


def test_partial_fit_reverts_on_divergence():
    clf = ComplexityClassifier(epochs=50, l2=1e-3)
    clf.fit(*_trainable())
    good = clf.weights_.copy()
    # a runaway online step (huge lr × the L2 feedback) goes non-finite;
    # partial_fit must leave the prior finite weights intact, never poison
    clf.partial_fit([_row(1e6)], ["hard"], lr=1e30, steps=50)
    assert clf.is_finite()
    assert np.allclose(clf.weights_, good)


def test_load_rejects_nonfinite_checkpoint(tmp_path):
    clf = ComplexityClassifier(epochs=20)
    clf.fit(*_trainable())
    p = clf.save(tmp_path / "ckpt.json")
    # corrupt the on-disk checkpoint to non-finite weights
    import json
    raw = json.loads(p.read_text())
    raw["weights"] = [float("nan")] * len(raw["weights"])
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        ComplexityClassifier.load(p)


def test_load_accepts_finite_checkpoint(tmp_path):
    clf = ComplexityClassifier(epochs=20)
    clf.fit(*_trainable())
    p = clf.save(tmp_path / "ckpt.json")
    loaded = ComplexityClassifier.load(p)
    assert loaded.is_finite()
