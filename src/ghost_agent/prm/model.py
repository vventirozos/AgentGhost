"""Numpy-only logistic regression for the Process Reward Model.

Mirrors ``router/model.py`` deliberately: same hyperparameters, same
JSON checkpoint format (with a different schema string), same
fail-safe behaviours. The two differ in shape only:

  * ``router/`` predicts request difficulty (binary easy/hard) from
    request features alone.
  * This module predicts step VALUE from a (state, action) feature
    vector. The label is binary at training time (continuous values
    can be passed in and are interpreted as soft labels — the binary
    cross-entropy degrades gracefully).

The choice to stick with logistic regression rather than reach for a
deep MLP is the same as the router's: the corpus is small, the
features are hand-crafted, and an inspectable model is more useful at
this stage than an opaque one. When the trajectory store grows past a
few thousand samples the model can be swapped — the file format is
versioned (``ghost.prm.logreg.v1``) and a future MLP would land at
``ghost.prm.mlp.v1``.

Saves to JSON — no pickle — so the persisted model is human-diffable
and safe to transfer (no code-execution risk on load).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .features import (
    PRM_FEATURE_NAMES,
    FeatureVector,
    PlanState,
    ActionFeatures,
    extract_step_features,
)


_SCHEMA = "ghost.prm.logreg.v1"


@dataclass
class PRMTrainingReport:
    """Snapshot of a fit. Stored alongside the weights in the saved JSON
    so a stale checkpoint doesn't surprise its loader."""

    n_samples: int = 0
    n_features: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    final_loss: float = 0.0
    train_accuracy: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    epochs_run: int = 0
    converged: bool = False


class StepValueModel:
    """Binary logistic regression on PRM features.

    API:
        m.fit(X, y)              — X: iterable of FeatureVector or
                                   raw float sequences; y: iterable of
                                   ints in {0, 1} OR floats in [0, 1]
                                   (soft labels are accepted).
        m.predict_value(state, action) — returns p(success | s, a).
                                          Convenience wrapper that
                                          builds the feature vector
                                          internally.
        m.predict_proba(x)       — returns p(success|x) for a vector
                                   the caller already has.
        m.save(path) / .load(path) — JSON round-trip.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        l2: float = 1e-3,
        epochs: int = 300,
        tol: float = 1e-5,
        random_state: int = 0,
    ):
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.epochs = int(epochs)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.feature_names_: Tuple[str, ...] = PRM_FEATURE_NAMES
        self.report_: Optional[PRMTrainingReport] = None

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------

    def fit(
        self,
        X: Iterable[Any],
        y: Iterable[Any],
    ) -> "StepValueModel":
        """Train on ``(X, y)``.

        Labels can be 0/1 ints or float values in [0, 1]. Float labels
        are interpreted as soft targets — useful when callers feed in
        the discount-weighted continuous values from
        ``derive_step_labels`` and want the model to fit the gradient
        rather than the threshold.
        """
        X_arr, y_arr = self._to_arrays(X, y)
        n_samples = X_arr.shape[0]
        if n_samples == 0:
            raise ValueError("fit called with no samples")
        if n_samples < 2:
            raise ValueError("fit needs at least 2 samples")

        # For soft-label training we don't strictly need both classes
        # represented, but for binary 0/1 labels we do — otherwise the
        # gradient is one-sided and the bias will run off to ±∞.
        unique_floats = np.unique(y_arr)
        is_binary = bool(np.all((unique_floats == 0.0) | (unique_floats == 1.0)))
        if is_binary and len(unique_floats) < 2:
            raise ValueError(
                f"fit needs both classes present; saw only "
                f"{sorted(unique_floats.tolist())}"
            )

        rng = np.random.default_rng(self.random_state)
        n_features = X_arr.shape[1]
        w = rng.normal(0.0, 0.01, size=n_features)
        b = 0.0

        prev_loss = math.inf
        converged = False
        epochs_run = 0
        final_loss = 0.0
        # On highly-separable synthetic data the weights can grow large
        # enough that the matmul step itself emits overflow / invalid /
        # divide-by-zero warnings — the _sigmoid clip downstream
        # produces correct 0/1 outputs but numpy still complains during
        # the intermediate. Suppress for the fit window only; predict
        # paths still pay the standard sigmoid clip.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for epoch in range(self.epochs):
                epochs_run = epoch + 1
                logits = X_arr @ w + b
                probs = _sigmoid(logits)
                err = probs - y_arr
                grad_w = (X_arr.T @ err) / n_samples + self.l2 * w
                grad_b = float(np.mean(err))
                w -= self.learning_rate * grad_w
                b -= self.learning_rate * grad_b

                eps = 1e-9
                loss = -float(np.mean(
                    y_arr * np.log(np.clip(probs, eps, 1 - eps))
                    + (1 - y_arr) * np.log(np.clip(1 - probs, eps, 1 - eps))
                )) + 0.5 * self.l2 * float(np.dot(w, w))
                final_loss = loss
                if abs(prev_loss - loss) < self.tol:
                    converged = True
                    break
                prev_loss = loss

        self.weights_ = w
        self.bias_ = b

        # Train-accuracy is meaningful only for binary labels. For
        # soft labels we threshold both pred and target at 0.5 so the
        # number is at least comparable.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            preds = (_sigmoid(X_arr @ w + b) >= 0.5).astype(float)
        targets = (y_arr >= 0.5).astype(float)
        train_acc = float(np.mean(preds == targets))
        class_counts = {
            "positive": int(np.sum(targets == 1)),
            "negative": int(np.sum(targets == 0)),
        }

        self.report_ = PRMTrainingReport(
            n_samples=int(n_samples),
            n_features=int(n_features),
            class_counts=class_counts,
            final_loss=float(final_loss),
            train_accuracy=train_acc,
            weights={
                name: float(w[i]) for i, name in enumerate(self.feature_names_)
            },
            bias=float(b),
            epochs_run=epochs_run,
            converged=converged,
        )
        return self

    # -----------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------

    def predict_proba(self, x: Any) -> float:
        """Return p(success | x) for a feature vector / sequence.

        Raises ``RuntimeError`` if called before fit. Prediction is
        bounded in [0, 1] even with pathological weights — the
        sigmoid clip guards against overflow."""
        if self.weights_ is None:
            raise RuntimeError("model not fitted")
        vec = self._vectorize(x)
        logit = float(np.dot(self.weights_, vec) + self.bias_)
        return float(_sigmoid(logit))

    def predict_value(
        self,
        state: PlanState,
        action: ActionFeatures,
    ) -> float:
        """Convenience: build features from ``(state, action)`` and
        return the model's success probability. This is the API the
        ``PRMScorer`` and the MCTS integration call into."""
        fv = extract_step_features(state, action)
        return self.predict_proba(fv)

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        if self.weights_ is None:
            raise RuntimeError("cannot save untrained model")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        payload = {
            "schema": _SCHEMA,
            "feature_names": list(self.feature_names_),
            "weights": self.weights_.tolist(),
            "bias": float(self.bias_),
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "l2": self.l2,
                "epochs": self.epochs,
                "tol": self.tol,
                "random_state": self.random_state,
            },
            "report": self.report_.__dict__ if self.report_ else None,
        }
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(p)
        return p

    @classmethod
    def load(cls, path: Path | str) -> "StepValueModel":
        p = Path(path)
        raw = json.loads(p.read_text())
        if raw.get("schema") != _SCHEMA:
            raise ValueError(
                f"unknown PRM schema: {raw.get('schema')!r} "
                f"(expected {_SCHEMA!r})"
            )
        # Detect feature drift: if the saved feature_names don't match
        # the current PRM_FEATURE_NAMES, the checkpoint is stale and
        # silently using it would mis-align weights against features.
        saved_names = tuple(raw.get("feature_names") or ())
        if saved_names and saved_names != PRM_FEATURE_NAMES:
            raise ValueError(
                "PRM feature schema drift — saved checkpoint's feature "
                "list does not match the current PRM_FEATURE_NAMES. "
                "Retrain before loading."
            )
        hp = raw.get("hyperparameters") or {}
        m = cls(
            learning_rate=float(hp.get("learning_rate", 0.1)),
            l2=float(hp.get("l2", 1e-3)),
            epochs=int(hp.get("epochs", 300)),
            tol=float(hp.get("tol", 1e-5)),
            random_state=int(hp.get("random_state", 0)),
        )
        m.weights_ = np.array(raw["weights"], dtype=float)
        m.bias_ = float(raw["bias"])
        m.feature_names_ = tuple(saved_names) if saved_names else PRM_FEATURE_NAMES
        if raw.get("report"):
            m.report_ = PRMTrainingReport(**raw["report"])
        return m

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _vectorize(self, x: Any) -> np.ndarray:
        if isinstance(x, FeatureVector):
            arr = np.array(x.values, dtype=float)
        elif isinstance(x, np.ndarray):
            arr = x.astype(float)
        elif isinstance(x, (list, tuple)):
            arr = np.array(x, dtype=float)
        else:
            raise TypeError(f"cannot vectorize {type(x).__name__}")
        # Sanitise: NaN → 0 (zero contribution), ±inf → finite bound.
        # The sigmoid clip downstream caps logits at ±60 anyway, so
        # any individual feature pushed toward ±1e6 just saturates the
        # output toward 0 or 1 — never NaN. This keeps a single bad
        # input value from poisoning the prediction.
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr

    def _to_arrays(
        self,
        X: Iterable[Any],
        y: Iterable[Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_list = [self._vectorize(x) for x in X]
        if not X_list:
            return np.zeros((0, len(PRM_FEATURE_NAMES))), np.zeros((0,))
        X_arr = np.stack(X_list, axis=0)
        y_list = [float(v) for v in y]
        y_arr = np.array(y_list, dtype=float)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y length mismatch")
        # Sanitise + clamp: NaN labels poison the cross-entropy gradient
        # (one NaN → all weights NaN within an epoch), so they're
        # neutralised to 0.5 (the "no useful signal" prior). ±inf is
        # similarly clamped before the unit-interval clip. Same
        # philosophy as input sanitisation — bad data must not corrupt
        # the model, but it must not silently train on it either; the
        # neutral value contributes near-zero gradient.
        y_arr = np.nan_to_num(y_arr, nan=0.5, posinf=1.0, neginf=0.0)
        y_arr = np.clip(y_arr, 0.0, 1.0)
        return X_arr, y_arr


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
