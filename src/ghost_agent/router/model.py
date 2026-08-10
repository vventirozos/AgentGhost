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
        # ⚠ RAISED FROM 300/1e-5 (audit 2026-08-10). At 300 epochs the fit
        # reported `converged: False`, and an under-converged logistic
        # regression has weights too small in magnitude, which COMPRESSES its
        # sigmoid outputs. Measured on 800 live requests: max confidence
        # **0.710**, so the planner-skip gate at 0.75 was UNREACHABLE and the
        # router's only live consumer could never fire — it trained, gated and
        # deployed correctly every idle cycle while changing nothing.
        #
        # 20000/1e-6 converges. Measured against the 300-epoch incumbent on the
        # SAME held-out split (n=409):
        #   accuracy      0.687 -> 0.704   (paired McNemar p=0.52 — a WASH)
        #   false-easy    0.0386 -> 0.0558 (gate max 0.25, so 22% of budget)
        #   max confidence 0.710 -> 0.912  (0 -> 123 of 800 usable skips)
        #   fit cost      19ms -> 95ms     (irrelevant on an idle retrain)
        #
        # THE POINT IS NOT ACCURACY — that did not move. It is that the model
        # is now CALIBRATED enough for its consumer to act on. 100k epochs
        # scored marginally higher accuracy but WORSE false-easy (0.0687) and
        # fewer usable skips, so it was rejected.
        epochs: int = 20000,
        tol: float = 1e-6,
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

        # errstate: X_arr is sanitized (nan_to_num above) and w/b are
        # guaranteed finite (the per-epoch guard raises otherwise), so this
        # matmul cannot see non-finite data — yet on macOS the Accelerate
        # BLAS raises spurious divide/overflow/invalid FPE flags on matmul
        # even with finite inputs, and this was the one matmul left outside
        # the suppression (the boot/idle-retrain warning spam at this line).
        # Real divergence is still detected explicitly by the guards above.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
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

    #: Features whose presence should raise p(hard), not lower it. A model
    #: that has learned NEGATIVE weights on these is inverted — it routes
    #: jargon-dense technical/coding requests to "easy" (§4O C-MAJOR-1, the
    #: n_steps-counts-history bug produced exactly this). Names must exist
    #: in FEATURE_NAMES.
    _MONOTONE_HARD_FEATURES = (
        "technical_jargon_count_log1p", "coding_language_mentions",
        "has_uppercase_acronym", "has_numeric_density",
    )
    #: The LOAD-BEARING inversion signal: "more jargon / more coding ⇒
    #: harder" is the monotonicity that actually matters. §4O R3 MAJOR-1:
    #: a pure net-sum over all four features had a blind spot — a model
    #: strongly inverted on jargon+coding could PASS if acronym/numeric
    #: carried compensating positive weight (repro: jargon -0.86 + coding
    #: -1.43 but net +0.66). Gate on the jargon+coding SUBTOTAL directly.
    _CORE_HARD_FEATURES = (
        "technical_jargon_count_log1p", "coding_language_mentions",
    )
    #: reject only a CLEARLY inverted model, not noise near zero — a fine
    #: model can carry a slightly-negative numeric weight (numbers ≠ hard).
    #: A healthy fixture's jargon+coding subtotal sits ≈ 0; a truly-inverted
    #: one ≈ -0.9 or worse. The margin sits between.
    _INVERSION_CORE_MARGIN = -0.5
    _INVERSION_NET_MARGIN = -1.0

    #: Held-out gate thresholds (§4AA, 2026-08-09). Asymmetric BY DESIGN:
    #: predicting "easy" for a genuinely hard request SKIPS THE PLANNER on
    #: a hard task (harmful); predicting "hard" for an easy one only wastes
    #: compute. So accuracy alone is not enough — the false-easy rate on
    #: hard requests carries its own ceiling.
    _GATE_MIN_HELDOUT = 60          # below this a "score" is noise
    _GATE_MAX_FALSE_EASY = 0.25     # measured: good 0.132, sign-flipped 0.868

    def weights_fingerprint(self) -> str:
        """Hash of the exact weights+bias the evidence was measured on.

        ⚠ Without this the gate is FORGEABLE (found by mutation-testing the
        gate itself, 2026-08-09): `looks_sane()` reads a stored report, so a
        checkpoint carrying real weights beside a passing report — a
        hand-edited file, a partial write, a mutated in-memory copy — would
        be waved through. The OLD prior-based gate could not be fooled that
        way because it inspected the weights directly. Binding the evidence
        to a fingerprint restores that property: change the weights and the
        evidence stops applying.
        """
        import hashlib
        if self.weights_ is None:
            return ""
        blob = ",".join(f"{float(w):.10g}" for w in self.weights_)
        blob += f"|{float(self.bias_):.10g}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _deploy_confidence_threshold() -> float:
        """The threshold the LIVE dispatcher escalates below.

        ⚠ Read from `ComplexityDispatcher`, never copied. It is also a CLI
        flag (`--router-confidence-threshold`), so the operator can move it;
        a hardcoded copy here would silently score an operating point the
        router does not run — the same two-copies-drift defect fixed in the
        bench pool earlier today. Callers that KNOW the live value pass it
        explicitly; this is only the fallback.

        ⚠ AUDIT 2026-08-10 — THE FALLBACK USED TO COINCIDE WITH THE TRUTH.
        It returned a bare 0.3, which is ALSO the dispatcher's default and
        ALSO the live value, so a broken read produced the RIGHT answer and
        was undetectable — right up until the operator moved
        `--router-confidence-threshold`, at which point the gate would
        silently score an operating point the router does not run. "Could a
        disconnected instrument produce this same output?" was yes.

        The numeric fallback is unchanged (a plausible threshold beats a
        wrong one), but `_deploy_threshold_probe` now reports WHICH path was
        taken, and `evaluate()` records it in the gate evidence so the
        coincidence can never hide again.
        """
        return ComplexityClassifier._deploy_threshold_probe()[0]

    @staticmethod
    def _deploy_threshold_probe() -> tuple:
        """(threshold, source) where source is 'dispatcher' or 'fallback'."""
        try:
            from .dispatch import ComplexityDispatcher
            import inspect
            v = float(inspect.signature(ComplexityDispatcher.__init__)
                      .parameters["confidence_threshold"].default)
            return v, "dispatcher"
        except Exception:
            return 0.3, "fallback"

    def evaluate(self, X, y, *, confidence_threshold: float | None = None) -> dict:
        """Score this model on held-out data. Returns the gate's evidence.

        `accuracy` is compared against `baseline` = always-predict-hard,
        which is exactly what the router does when it is escalate-all —
        so "beats baseline" literally means "better than doing nothing".

        ⚠ SCORES THE DEPLOYED DECISION, not the raw label (fresh-eye review,
        2026-08-09). `ComplexityDispatcher` escalates when confidence < 0.3
        WHATEVER the label says, so a low-confidence "easy" never actually
        skips the planner in production. A first version of this method read
        the bare label and therefore measured an operating point the router
        does not ship — overstating false-easy, i.e. gating on a system that
        does not exist. Simulating the escalation here is what makes "beats
        escalate-all" a statement about the deployed router.
        """
        thr = (self._deploy_confidence_threshold()
               if confidence_threshold is None else float(confidence_threshold))
        n = len(y)
        if n == 0:
            return {"n": 0, "accuracy": 0.0, "baseline": 0.0,
                    "false_easy_on_hard": 1.0, "classes": 0}
        hard = [i for i, lab in enumerate(y) if lab == "hard"]
        ok = fe = 0
        for i, lab in enumerate(y):
            pred, conf = self.predict(X[i])
            if conf < thr:
                pred = "hard"       # the dispatcher escalates — safe path
            ok += (pred == lab)
            if lab == "hard" and pred == "easy":
                fe += 1
        return {
            "n": n,
            "accuracy": ok / n,
            "baseline": len(hard) / n,
            "false_easy_on_hard": (fe / len(hard)) if hard else 0.0,
            "classes": len({lab for lab in y}),
            # Binds this evidence to the weights it was measured on.
            "weights_sha": self.weights_fingerprint(),
            "confidence_threshold": thr,
            # ⚠ WHERE the threshold came from. "fallback" means the read of
            # ComplexityDispatcher FAILED and this evidence describes an
            # operating point that may not be the live one. Recorded because
            # the fallback value (0.3) currently equals the truth, so the
            # failure is otherwise invisible — and a gate that scores the
            # wrong operating point is the §4AA defect all over again.
            "confidence_threshold_source": (
                "caller" if confidence_threshold is not None
                else self._deploy_threshold_probe()[1]),
        }

    @classmethod
    def gate_verdict(cls, ev: dict) -> tuple:
        """(passed, reason) for a held-out evidence dict. FAIL-CLOSED."""
        if not ev:
            return False, "no held-out evidence recorded"
        if int(ev.get("n", 0)) < cls._GATE_MIN_HELDOUT:
            return False, f"held-out n={ev.get('n')} < {cls._GATE_MIN_HELDOUT}"
        if int(ev.get("classes", 0)) < 2:
            return False, "held-out split has only one class"
        if ev["accuracy"] <= ev["baseline"]:
            return False, (f"accuracy {ev['accuracy']:.3f} does not beat "
                           f"escalate-all {ev['baseline']:.3f}")
        if ev["false_easy_on_hard"] > cls._GATE_MAX_FALSE_EASY:
            return False, (f"skips the planner on "
                           f"{ev['false_easy_on_hard']:.1%} of hard requests "
                           f"(max {cls._GATE_MAX_FALSE_EASY:.0%})")
        return True, (f"acc {ev['accuracy']:.3f} > escalate-all "
                      f"{ev['baseline']:.3f}, false-easy "
                      f"{ev['false_easy_on_hard']:.1%} (n={ev['n']})")

    def looks_sane(self) -> bool:
        """True iff finite AND its HELD-OUT evidence clears the gate.

        ⚠ REPLACES A PRIOR-BASED TEST THAT WAS ANTI-CORRELATED WITH QUALITY
        (§4AA, 2026-08-09). The previous gate rejected a model whose
        technical/coding weights were negative, encoding the prior "more
        jargon ⇒ harder". Measured on 1354 labelled trajectories (92% real
        user_request turns; self-play contamination ruled out), THIS
        agent's traffic says the opposite — jargon is 4.1x and coding
        mentions 6.8x MORE common in EASY turns, and even LENGTH inverts
        (longer ⇒ easier). Vagueness, not technicality, predicts work here.
        No labelling bug makes longer requests easier; the signal is real.

        Consequences of the old gate, both measured on a 70/30 split
        (seed 7, n=407 held out):
          * the FITTED model — accuracy 0.695 vs escalate-all 0.560,
            skipping the planner on 13.2% of hard requests — was REJECTED,
            so the router sat escalate-all permanently (~29h and counting,
            and forever, since every fit reproduces the same signs);
          * a SIGN-FLIPPED model — accuracy 0.305, skipping the planner on
            86.8% of hard requests, i.e. exactly the §4O catastrophe the
            gate exists to prevent — was ACCEPTED.
        A gate that rejects the good model and admits the catastrophic one
        is worse than no gate, because it is trusted.

        The replacement asks the only question that matters: on data the
        model did NOT train on, does it beat doing nothing, without
        skipping the planner too often? FAIL-CLOSED — a model carrying no
        held-out evidence (a legacy checkpoint, a hand-built model) is
        rejected, and rejection leaves the router escalate-all, which is
        the safe default and today's behaviour.
        """
        if not self.is_finite():
            return False
        ev = getattr(self, "gate_report_", None) or {}
        # The evidence must describe THESE weights, not some earlier ones.
        if ev.get("weights_sha") != self.weights_fingerprint():
            return False
        passed, _ = self.gate_verdict(ev)
        return passed

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
            # §4AA: the gate's EVIDENCE travels with the model. A checkpoint
            # that cannot say why it was accepted is rejected on load —
            # fail-closed, so a legacy or hand-built file cannot slip past.
            "gate_report": getattr(self, "gate_report_", None),
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
        # §4AA: the held-out evidence that justified deploying this model.
        # Absent on a legacy checkpoint → `looks_sane()` fails closed and the
        # router stays escalate-all until a gated model trains.
        clf.gate_report_ = raw.get("gate_report") or None
        try:
            clf.weights_ = np.array(raw["weights"], dtype=float)
            clf.bias_ = float(raw["bias"])
        except (KeyError, TypeError, ValueError) as e:
            # Normalise malformed-payload failures (missing keys, wrong
            # types, ragged arrays) into ValueError so the boot loader has
            # ONE clean, catchable failure mode: fall back to the
            # escalate-all dispatcher and retrain over the bad file.
            raise ValueError(
                f"router checkpoint at {p} is malformed ({e}) — refusing to "
                "load; it will be retrained."
            ) from e
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
            try:
                clf.report_ = TrainingReport(**raw["report"])
            except TypeError as e:
                # Field drift in the diagnostics-only report (older/newer
                # TrainingReport shape) — same clean ValueError contract
                # as above so the boot loader falls back and retrains.
                raise ValueError(
                    f"router checkpoint at {p} has an incompatible training "
                    f"report ({e}) — refusing to load; it will be retrained."
                ) from e
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
            vec = np.array(x.values, dtype=float)
        elif isinstance(x, np.ndarray):
            vec = x.astype(float)
        elif isinstance(x, (list, tuple)):
            vec = np.array(x, dtype=float)
        elif isinstance(x, str):
            vec = np.array(extract_features(x).values, dtype=float)
        else:
            raise TypeError(f"cannot vectorize {type(x).__name__}")
        # Sanitise here, the single choke point every path flows through
        # (fit/partial_fit/bce_loss via _to_arrays, predict_proba directly):
        # a non-finite feature is always a bug, never signal, and one NaN
        # otherwise poisons the dot product → a 0.5 blanket fallback at
        # predict, a silently-dropped partial_fit step, or a NaN holdout
        # loss. Mapping just the bad component to 0 keeps the remaining
        # features informative.
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

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
