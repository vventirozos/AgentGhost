"""Numpy-only logistic regression for the complexity router.

Why not sklearn? Two reasons:
  1. `scikit-learn` isn't a hard dependency of Ghost; pulling it in
     for a single classifier inflates the install footprint.
  2. The fit is trivial (~20 features, ~10k samples in the realistic
     case); a ~40-line gradient descent is as fast as the overhead of
     calling into sklearn and keeps the whole thing inspectable.

Model:
    p(hard|x) = sigmoid(w·x + b)
    Loss: binary cross-entropy with L2 regularization.

Saves to JSON — no pickle — so the persisted model is human-diffable
and safe to transfer (no code-execution risk when loading).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .features import FEATURE_NAMES, FeatureVector, extract_features


LABEL_TO_INT = {"easy": 0, "hard": 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


@dataclass
class TrainingReport:
    """Snapshot of training outcome. Returned from `fit`; also stored
    inside the saved model JSON so a future read knows what it's
    looking at."""

    n_samples: int = 0
    n_features: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    final_loss: float = 0.0
    train_accuracy: float = 0.0
    # Per-feature weight snapshot (name → float). Useful for explaining
    # a prediction without re-running the model.
    weights: Dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    epochs_run: int = 0
    converged: bool = False


class ComplexityClassifier:
    """Binary logistic regression.

    API (mirrors scikit's familiar shape, but deliberately small):
        clf.fit(X, y)          — X: iterable of feature vectors;
                                y: iterable of 'easy'/'hard' labels.
        clf.predict_proba(x)   — returns p(hard|x).
        clf.predict(x)         — returns ('easy'|'hard', confidence).
        clf.save(path)         — JSON dump.
        ComplexityClassifier.load(path) — JSON load.
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
        self.feature_names_: Tuple[str, ...] = FEATURE_NAMES
        self.report_: Optional[TrainingReport] = None

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------

    def fit(
        self,
        X: Iterable[Any],
        y: Iterable[str],
    ) -> "ComplexityClassifier":
        """Train on `(X, y)`. Accepts either FeatureVector instances or
        raw sequences of floats for X."""
        X_arr, y_arr = self._to_arrays(X, y)
        if X_arr.shape[0] == 0:
            raise ValueError("fit called with no samples")
        if X_arr.shape[0] < 2:
            raise ValueError("fit needs at least 2 samples")
        # Require at least one of each class for a meaningful binary fit.
        if len(set(y_arr.tolist())) < 2:
            raise ValueError(
                f"fit needs both classes present; saw only "
                f"{sorted(set(y_arr.tolist()))}"
            )

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X_arr.shape
        w = rng.normal(0.0, 0.01, size=n_features)
        b = 0.0

        # Sanitise the design matrix: a single non-finite feature value
        # (e.g. an inf slipping through extract_features) contaminates the
        # whole matmul and the model diverges to NaN. A non-finite feature
        # is always a bug, never signal, so map it to the neutral 0.
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        prev_loss = math.inf
        converged = False
        epochs_run = 0
        final_loss = 0.0
        # errstate: divergence shows up as over/invalid/divide warnings on
        # the matmul; we DETECT it explicitly below and bail, so suppress
        # the warning spam (the production log filled with these). Mirrors
        # partial_fit / bce_loss.
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

                # Divergence guard: an LR too large for the feature scale
                # blows the weights up to inf/NaN (the production failure was
                # the L2 feedback term `(1 - lr·l2)·w` going < -1 → exponential
                # blowup). Once any weight is non-finite every subsequent
                # matmul is poisoned and the model would be hot-swapped into
                # the live router returning NaN confidences. Check the weights
                # directly each epoch and bail loudly rather than persist
                # garbage — RouterTrainer.run() catches this into bail_reason
                # and the router stays in its safe escalate-all pass-through.
                if not (np.all(np.isfinite(w)) and math.isfinite(b)):
                    raise ValueError(
                        f"router training diverged to non-finite weights at "
                        f"epoch {epoch + 1} (learning_rate={self.learning_rate} "
                        "is likely too large for the feature scale); refusing "
                        "to persist a NaN model"
                    )

                # Loss with L2 term
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

        # Final backstop: never expose a non-finite model even if the loss
        # check above was somehow skipped (0 epochs, etc.).
        if not (np.all(np.isfinite(w)) and math.isfinite(b)):
            raise ValueError(
                "router training produced non-finite weights; refusing to "
                "persist a NaN model"
            )

        self.weights_ = w
        self.bias_ = b

        preds = (_sigmoid(X_arr @ w + b) >= 0.5).astype(float)
        train_acc = float(np.mean(preds == y_arr))
        class_counts = {
            "easy": int(np.sum(y_arr == 0)),
            "hard": int(np.sum(y_arr == 1)),
        }

        self.report_ = TrainingReport(
            n_samples=int(n_samples),
            n_features=int(n_features),
            class_counts=class_counts,
            final_loss=float(final_loss),
            train_accuracy=train_acc,
            weights={name: float(w[i]) for i, name in enumerate(self.feature_names_)},
            bias=float(b),
            epochs_run=epochs_run,
            converged=converged,
        )
        return self

    # -----------------------------------------------------------------
    # Online update (mirrors prm.model.StepValueModel)
    # -----------------------------------------------------------------

    def is_finite(self) -> bool:
        """True iff the model is fitted with all-finite weights and bias.

        A guard for callers that hot-swap a freshly-trained classifier into
        the live router (``core.agent`` idle retrain) or load one from disk:
        a diverged NaN/inf model must NEVER be installed, because
        ``predict_proba`` would then return NaN and every routing decision
        would be garbage. Returns False for an unfitted model too."""
        if self.weights_ is None:
            return False
        return bool(np.all(np.isfinite(self.weights_))) and math.isfinite(
            float(self.bias_)
        )

    def clone(self) -> "ComplexityClassifier":
        """Copy with the same hyperparameters and weights — used by a
        guarded online-update path so a candidate step is applied to a
        throwaway model first."""
        m = ComplexityClassifier(
            learning_rate=self.learning_rate, l2=self.l2,
            epochs=self.epochs, tol=self.tol, random_state=self.random_state,
        )
        m.weights_ = None if self.weights_ is None else self.weights_.copy()
        m.bias_ = float(self.bias_)
        m.feature_names_ = tuple(self.feature_names_)
        m.report_ = self.report_
        return m

    def partial_fit(self, X: Iterable[Any], y: Iterable[Any], *,
                    lr: Optional[float] = None, steps: int = 1) -> "ComplexityClassifier":
        """Apply ``steps`` gradient steps to the existing weights (online
        counterpart to batch ``fit``). Requires an already-fitted model;
        small ``lr`` + few ``steps`` + the existing L2 bound the change."""
        if self.weights_ is None:
            raise RuntimeError(
                "partial_fit requires an already-fitted model — online "
                "updates refine the batch model, they don't bootstrap it"
            )
        X_arr, y_arr = self._to_arrays(X, y)
        n = X_arr.shape[0]
        if n == 0:
            return self
        rate = float(self.learning_rate if lr is None else lr)
        w = self.weights_.astype(float).copy()
        b = float(self.bias_)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for _ in range(max(1, int(steps))):
                probs = _sigmoid(X_arr @ w + b)
                err = probs - y_arr
                grad_w = (X_arr.T @ err) / n + self.l2 * w
                grad_b = float(np.mean(err))
                w -= rate * grad_w
                b -= rate * grad_b
        # Reject a diverged online step rather than poisoning the live
        # model: keep the prior (finite) weights if the update went
        # non-finite. The caller's holdout BCE gate would also reject it,
        # but this keeps the model self-consistent regardless of caller.
        if not (np.all(np.isfinite(w)) and math.isfinite(b)):
            return self
        self.weights_ = w
        self.bias_ = b
        return self

    def bce_loss(self, X: Iterable[Any], y: Iterable[Any]) -> float:
        """Mean BCE of the current model on ``(X, y)`` — the holdout
        metric a guarded online update compares before/after a step."""
        if self.weights_ is None:
            raise RuntimeError("model not fitted")
        X_arr, y_arr = self._to_arrays(X, y)
        if X_arr.shape[0] == 0:
            return 0.0
        eps = 1e-9
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            probs = _sigmoid(X_arr @ self.weights_ + self.bias_)
            return -float(np.mean(
                y_arr * np.log(np.clip(probs, eps, 1 - eps))
                + (1 - y_arr) * np.log(np.clip(1 - probs, eps, 1 - eps))
            ))

    # -----------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------

    def predict_proba(self, x: Any) -> float:
        """Return p(hard|x)."""
        if self.weights_ is None:
            raise RuntimeError("classifier not fitted")
        vec = self._vectorize(x)
        logit = float(np.dot(self.weights_, vec) + self.bias_)
        if not math.isfinite(logit):
            # Defensive: a non-finite model should never reach here (fit /
            # load / partial_fit all reject one), but if it does, return the
            # neutral 0.5 so the dispatcher escalates rather than acting on
            # a NaN. Never let NaN reach a routing decision.
            return 0.5
        return float(_sigmoid(logit))

    def predict(self, x: Any, *, decision_threshold: float = 0.5) -> Tuple[str, float]:
        """Return (label, confidence). Confidence is
        `|p - 0.5| * 2` → 0 at 50/50, 1 at 0 or 1."""
        p_hard = self.predict_proba(x)
        label = "hard" if p_hard >= decision_threshold else "easy"
        conf = abs(p_hard - 0.5) * 2.0
        return label, float(conf)

    def predict_from_text(self, text: str, prior_turn_text: str = "") -> Tuple[str, float]:
        """Convenience: extract features and predict in one call."""
        fv = extract_features(text, prior_turn_text=prior_turn_text)
        return self.predict(fv)

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        if self.weights_ is None:
            raise RuntimeError("cannot save untrained classifier")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        payload = {
            "schema": "ghost.router.logreg.v1",
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
    def load(cls, path: Path | str) -> "ComplexityClassifier":
        p = Path(path)
        raw = json.loads(p.read_text())
        if raw.get("schema") != "ghost.router.logreg.v1":
            raise ValueError(f"unknown model schema: {raw.get('schema')}")
        hp = raw.get("hyperparameters") or {}
        clf = cls(
            learning_rate=float(hp.get("learning_rate", 0.1)),
            l2=float(hp.get("l2", 1e-3)),
            epochs=int(hp.get("epochs", 300)),
            tol=float(hp.get("tol", 1e-5)),
            random_state=int(hp.get("random_state", 0)),
        )
        clf.weights_ = np.array(raw["weights"], dtype=float)
        clf.bias_ = float(raw["bias"])
        names = tuple(raw.get("feature_names") or FEATURE_NAMES)
        clf.feature_names_ = names
        # Validate the persisted feature schema against the CURRENT one. Only
        # the schema string was checked before, so a checkpoint written under
        # a reordered/renamed feature set (same length) loaded clean and
        # predict_proba dotted old-order weights against new-order feature
        # vectors → silently wrong routing. A length mismatch would instead
        # raise deep inside np.dot at serve time. Fail loud here so the boot
        # loader falls back to the safe escalate-all dispatcher and retrains.
        if names != tuple(FEATURE_NAMES) or clf.weights_.shape[0] != len(FEATURE_NAMES):
            raise ValueError(
                f"router checkpoint at {p} was trained on a different feature "
                f"schema (checkpoint has {len(names)} features "
                f"{'in a different order ' if len(names) == len(FEATURE_NAMES) else ''}"
                f"vs current {len(FEATURE_NAMES)}) — refusing to load a "
                "misaligned model; it will be retrained."
            )
        if raw.get("report"):
            clf.report_ = TrainingReport(**raw["report"])
        # Reject a persisted NaN/inf checkpoint (e.g. one written by a
        # pre-guard training run that diverged). Loading it would silently
        # poison routing; the boot loader catches this and falls back to
        # the safe escalate-all pass-through dispatcher.
        if not clf.is_finite():
            raise ValueError(
                f"router checkpoint at {p} has non-finite weights — refusing "
                "to load a corrupt (diverged) model"
            )
        return clf

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _vectorize(self, x: Any) -> np.ndarray:
        if isinstance(x, FeatureVector):
            return np.array(x.values, dtype=float)
        if isinstance(x, np.ndarray):
            return x.astype(float)
        if isinstance(x, (list, tuple)):
            return np.array(x, dtype=float)
        if isinstance(x, str):
            return np.array(extract_features(x).values, dtype=float)
        raise TypeError(f"cannot vectorize {type(x).__name__}")

    def _to_arrays(
        self,
        X: Iterable[Any],
        y: Iterable[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_list = [self._vectorize(x) for x in X]
        if not X_list:
            return np.zeros((0, 0)), np.zeros((0,))
        X_arr = np.stack(X_list, axis=0)
        y_arr = np.array(
            [LABEL_TO_INT[label] for label in y],
            dtype=float,
        )
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y length mismatch")
        return X_arr, y_arr


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
