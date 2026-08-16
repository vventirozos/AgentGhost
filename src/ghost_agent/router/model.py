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

from .features import (
    EMBED_DIM,
    EMBED_FEATURE_NAMES,
    FEATURE_NAMES,
    FeatureVector,
    extract_features,
    model_feature_names,
)


def _binom_tail_ge(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p) — the gate's one-sided p-value.

    EXACT, with no approximation branch. The normal approximation this
    replaces was anti-conservative exactly where it mattered — 3 successes
    in 3 trials scores z=1.73 and "passes" a 5% test whose true p is
    0.125, which measurably deployed label-independent models. A first fix
    kept the approximation for n > 1000; that branch was unreachable by
    any test (it needs >1000 discordant pairs) and so was UNPINNED code
    guarding the most safety-critical decision in the router. One path
    instead.

    Summed in log space via lgamma: the direct product overflows the
    binomial coefficient into `inf` for large n, which would silently
    return a garbage p-value rather than failing.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    log_fact_n = math.lgamma(n + 1)
    total = 0.0
    for i in range(k, n + 1):
        log_c = log_fact_n - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        total += math.exp(log_c + i * log_p + (n - i) * log_q)
    return min(1.0, total)


LABEL_TO_INT = {"easy": 0, "hard": 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


class FeatureSchemaMismatch(ValueError):
    """A vector's width does not match the fitted model's.

    Subclasses ValueError deliberately: every existing caller that
    wraps scoring in `except ValueError` (the trainer's fit path, the
    checkpoint loader) keeps its current behaviour, while the
    dispatcher can catch this specific type and escalate.
    """


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
        # §4BQ flip (vi): True once fitted on (or loaded from) the
        # lexical+embedding schema. The MODEL, not an env flag, decides
        # whether an embedding is required at serve time — so toggling
        # GHOST_ROUTER_EMBED can never mismatch a checkpoint on disk.
        self.uses_embeddings_: bool = False
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
        # §4BQ: adopt the schema the DATA arrived in, before anything
        # (notably the TrainingReport's per-name weight map) reads
        # feature_names_. Width is the only thing that can distinguish
        # them, and an unrecognised width is a caller bug — bail rather
        # than fit a model whose saved names would misdescribe its own
        # weights, which is exactly what the load guard exists to catch.
        if n_features == len(FEATURE_NAMES) + EMBED_DIM:
            self.uses_embeddings_ = True
        elif n_features == len(FEATURE_NAMES):
            self.uses_embeddings_ = False
        else:
            raise ValueError(
                f"fit got {n_features}-wide rows; expected "
                f"{len(FEATURE_NAMES)} (lexical) or "
                f"{len(FEATURE_NAMES) + EMBED_DIM} (lexical+embedding)"
            )
        self.feature_names_ = model_feature_names(self.uses_embeddings_)
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
    #: ⚠ 60, and it STAYS 60. I raised this to 150 twice and was wrong
    #: both times, by the same mechanism, the second time while quoting a
    #: comment warning about that mechanism:
    #:
    #:   1. "a real fit deploys 0/20 at corpus 200" — measured with 150
    #:      already installed, so the runs were rejected by THIS check.
    #:   2. Re-measured after reverting... by sampling 200 RAW
    #:      trajectories. Labelling drops the ambiguous ones, so the
    #:      labelled corpus fell under `_gate_min_trajectories()` and the
    #:      runs bailed at the floor without ever reaching the gate. Same
    #:      self-confirming artifact, different disguise.
    #:
    #: Measured correctly (production trainer, real bge, real corpus,
    #: LABELLED sizes, floor at 60): a real fit deploys at every corpus
    #: size from 200 to 499 — chronological prefixes 7/7, random
    #: subsamples 26/40 at n=200 and 37/40 at n=300. And 6 wins / 0 losses
    #: at baseline 0.546 gives p = 0.0087, so 60 held-out turns can
    #: demonstrate significance perfectly well.
    #:
    #: The raise would have cost a fresh box ~3 extra months of
    #: escalate-all (at ~3.5 turns/day, first deploy 200 → 500 labelled
    #: turns) and bought nothing: label-randomised corpora leak at the
    #: same rate under either floor. A held-out-size floor cannot lower a
    #: significance test's type-I rate; only the test can.
    _GATE_MIN_HELDOUT = 60          # below this a "score" is noise
    _GATE_MAX_FALSE_EASY = 0.25     # measured: good 0.132, sign-flipped 0.868

    #: Minimum SHARE of held-out turns the model must route differently
    #: from escalate-all, as a fraction, with a small absolute floor.
    #:
    #: ⚠ This was a FIXED 30 for one round, and measuring it is what
    #: killed it. A fixed count is an absolute bar against a held-out set
    #: that may be 60 turns, so it demanded the model reroute half of
    #: them. Ablated, 100 trials/cell, embedding arm:
    #:
    #:     arm                n=200      n=300      n=500     noise
    #:     alpha .01, min 0   40/100     84/100    100/100    1/300
    #:     alpha .01, min 30   0/100      3/100     88/100    0/300
    #:
    #: It bought one avoided noise deploy in 300 runs and cost ALL real
    #: deployment at corpus 200 and 96% at 300 — §4AA's original sin
    #: (a gate that rejects the good model) in milder form. Its motivating
    #: case, `win=4 lose=0`, is already rejected by the alpha below
    #: (p=0.043 > 0.01), so the guard was redundant where it mattered and
    #: harmful where it did not. Kept only as a scale-free floor against
    #: the genuinely degenerate case — a model that changes almost no
    #: decisions has nothing to be right about.
    #:
    #: ⚠ AND IT CHANGES NO VERDICTS AT THIS BOX'S HARD-RATE. Across 8,801
    #: production-trainer fits it rejected 3,000 — every one of which the
    #: alpha below would also have rejected (0 exceptions) — and rejected
    #: 0 of 801 real fits. It is behaviourally live only at hard-rates
    #: above ~0.70 with exactly 4 clean wins. Kept because that case is
    #: real and cheap to exclude, NOT because it reduces the noise rate:
    #: an earlier comment here claimed it delivered "noise 0/300 … none of
    #: its leak", and the leak is identical with and without it.
    #: Minimum corpus GROWTH before the gate is allowed to look again.
    #:
    #: A 1% test leaks ~1% per look, and this gate is re-run on EVERY
    #: corpus change while its verdict is sticky — so the number that
    #: matters is P(ever deploys noise), not the per-run rate. Measured
    #: over 184 simulated 200→400-turn growth histories with signal-free
    #: labels, re-gating at every size: **19 deployed a signal-free model
    #: at least once (10.3%)**, in clusters of 3-7 consecutive sizes,
    #: because adjacent looks share most of their held-out split.
    #:
    #: Controlling the number of LOOKS is the honest lever — the last two
    #: thresholds added to this gate were redundant or harmful, so this
    #: does not add a third bar to the decision. Requiring 20% more
    #: labelled turns than the deployed model was gated on turns ~200
    #: looks over that window into ~4.
    #:
    #: Overridden when the REPRESENTATION changes (a lexical↔embedding
    #: switch must be allowed to re-gate immediately), so a flip is never
    #: blocked waiting for the corpus to grow.
    #: Held-out overlap (Jaccard) above which a re-test is the SAME test.
    #: 0.8 means a look re-opens once ~20% of the evidence is new. Replaces
    #: a corpus-growth ratio and an absolute slack, both of which were
    #: proxies with bypass channels — bench rows, a flapping embedder and
    #: an orphaned ledger each re-opened looks without the evidence
    #: changing at all (measured 101/201, 10/10 and 201/201 looks).
    #: 0.85 => a look re-opens once the labelled corpus is ~18% new,
    #: i.e. at about the 1.2x growth the old ratio asked for (600 vs 720
    #: overlaps 0.833). Compared on the CORPUS, not the held-out split:
    #: the split is a positional shuffle, so n and n+1 barely overlap and
    #: every added turn read as new evidence.
    _GATE_MAX_HELDOUT_OVERLAP = 0.85

    _GATE_MIN_CORPUS_GROWTH = 1.2

    #: ...OR this many additional labelled turns, whichever comes first.
    #: The ratio alone compounds: 1,483 → 1,780 → 2,136 → 2,564 → 3,077 is
    #: +10, +11, +14, +16 days at this box's measured 31.1 labelled
    #: turns/day, so a router that keeps being rejected waits ever longer
    #: for its next chance. A multiple-testing control may delay a look;
    #: it must not be able to postpone one indefinitely.
    _GATE_LOOK_ABSOLUTE_SLACK = 250

    _GATE_MIN_DISCORDANT_FRACTION = 0.02
    _GATE_MIN_DISCORDANT_FLOOR = 5

    #: One-sided significance bar. 0.01 rather than 0.05 because the gate
    #: is run REPEATEDLY (every corpus change) and its verdict is sticky,
    #: so the per-run rate is not the rate that matters.
    #:
    #: ⚠ MEASURED, because an earlier version of this comment asserted the
    #: repeated-testing argument and never quantified it, while a
    #: neighbouring comment claimed the resulting noise rate was ZERO.
    #: It is not zero — a 1% test leaks about 1% of the time by
    #: construction. Pooled over 8,000 label-randomised production fits at
    #: hard-rates 0.30-0.75 and under label permutation: 6 deploys,
    #: 0.075% [0.034%, 0.164%]. ⚠ A later, larger run measured **0.224%
    #: [0.157%, 0.310%]** over 16,080 label-randomised production fits —
    #: the CI excludes the first figure, so take 0.224% as the number.
    #: It is load-bearing (it justifies alpha 0.01 and the looks budget),
    #: and the first estimate was quoted for months of this work off 8,000
    #: fits at a single hard-rate mix. Re-running the gate as a corpus grows
    #: compounds that: over 184 simulated 200→400-turn growth histories at
    #: this box's hard-rate, **19 (10.3% [6.7%, 15.6%]) deployed a
    #: signal-free model at least once**, and a deploy is sticky.
    #:
    #: That residual is a property of significance testing under repeated
    #: looks, not a bug to be patched with another threshold — the last
    #: two thresholds added here were redundant or harmful. Reducing it
    #: further needs a design change (alpha spending, or re-gating only on
    #: a materially different corpus), which is registered as open work
    #: rather than guessed at. Consequence today is bounded: the router
    #: has no live consumer (`_MCTS_TURNSTART_ENABLED = False`).
    _GATE_ALPHA = 0.01

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
        # DISCORDANT PAIRS vs the escalate-all baseline. The baseline says
        # "hard" for everything, so the two predictors can only differ where
        # this model says "easy": `disc_win` = it was genuinely easy (we are
        # right, baseline wrong), `disc_lose` = it was hard (we are wrong,
        # baseline right). Everything else is a tie. Hence, EXACTLY:
        #     accuracy - baseline == (disc_win - disc_lose) / n
        # which is what makes McNemar the correct significance test in
        # `gate_verdict` — the counts have to travel with the evidence for
        # the gate to use the paired statistic instead of a proxy for it.
        disc_win = disc_lose = 0
        for i, lab in enumerate(y):
            pred, conf = self.predict(X[i])
            if conf < thr:
                pred = "hard"       # the dispatcher escalates — safe path
            ok += (pred == lab)
            if pred == "easy":
                if lab == "easy":
                    disc_win += 1
                else:
                    disc_lose += 1
            if lab == "hard" and pred == "easy":
                fe += 1
        return {
            "n": n,
            "accuracy": ok / n,
            "baseline": len(hard) / n,
            "discordant_win": disc_win,
            "discordant_lose": disc_lose,
            # The MAJORITY class, which is NOT the same question as
            # escalate-all whenever the held-out hard-rate is below 0.5.
            # Beating escalate-all is the OPERATIONAL test; beating the
            # majority class is the "does this model have any signal at
            # all" test, and only the second one is safe against a
            # label-independent model on an easy-dominated corpus.
            "majority_baseline": max(len(hard), n - len(hard)) / n,
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
        if not isinstance(ev, dict) or not ev:
            return False, "no held-out evidence recorded"
        # EVERY field, INCLUDING n and classes, coerced inside the guard.
        # Round 6 wrote "every numeric field is coerced HERE, once" and
        # left these two above it on a bare int(), so `n: None` raised
        # TypeError, `n: "abc"` ValueError, `n: inf` OverflowError — the
        # exact fail-CRASH class that round said it had fixed, contained
        # only by all three callers' try/except. OverflowError is caught
        # too: it is not a subclass of ValueError, so `inf` counts slipped
        # past the inner guard as well.
        try:
            _n_raw = int(ev.get("n", 0))
            _classes = int(ev.get("classes", 0))
        except (TypeError, ValueError, OverflowError):
            return False, "held-out evidence is malformed or incomplete"
        if _n_raw < cls._GATE_MIN_HELDOUT:
            return False, f"held-out n={_n_raw} < {cls._GATE_MIN_HELDOUT}"
        if _classes < 2:
            return False, "held-out split has only one class"
        # BEATING THE BASELINE MUST BE SIGNIFICANT, NOT MERELY TRUE.
        #
        # This was a bare `accuracy <= baseline` rejection, and its
        # protection against a noise fit turned out to be INCIDENTAL: an
        # 18-feature model on signal-free labels degenerates to
        # constant-predict, so accuracy == baseline exactly and the tie
        # rejected it. §4BQ took the model to 402 features, and a
        # 402-parameter fit on ~140 training rows has enough prediction
        # variance to land a hair above baseline about half the time.
        # Measured on shuffled-label corpora through the production
        # trainer: lexical-18 deployed noise 1/14 runs, combined-402
        # deployed it 6/14 at the corpus floor and 7/14 at n=600, with
        # winning margins of 0.003-0.033 — one to three held-out turns.
        #
        # The margin required here is a one-sided 95% normal bound on the
        # baseline rate at this held-out n, so it TIGHTENS automatically as
        # a corpus grows and cannot be outrun by adding parameters. At the
        # live n=445/baseline 0.544 it demands +0.039; the shipped model
        # clears it by 0.184. §4AA's own thesis is that a gate which admits
        # what it exists to exclude is worse than no gate, because it is
        # trusted.
        _n = max(1, int(ev.get("n", 0)))
        # EVERY numeric field is coerced HERE, once, and only the coerced
        # values are used below. The first version of this guard coerced
        # three fields, then re-read the RAW dict at the false-easy check —
        # so a string survived the guard and crashed there, and a NaN
        # sailed through entirely (`nan > 0.25` is False), which is
        # FAIL-OPEN in the one place that must never be. Evidence comes
        # straight from checkpoint JSON, and a hand-edited or
        # partially-written file is the stated threat model.
        try:
            _base = float(ev["baseline"])
            _acc = float(ev["accuracy"])
            _fe = float(ev["false_easy_on_hard"])
            _win_raw, _lose_raw = ev.get("discordant_win"), ev.get("discordant_lose")
            _w = None if _win_raw is None else int(_win_raw)
            _l = None if _lose_raw is None else int(_lose_raw)
        except (KeyError, TypeError, ValueError, OverflowError):
            # FAIL-CLOSED means fail-closed, not fail-crash. The docstring
            # promised this; the code raised KeyError and relied on every
            # caller's try/except to look like it was working.
            return False, "held-out evidence is malformed or incomplete"
        # INTERNAL CONSISTENCY. `accuracy` was coerced, finiteness-checked
        # and then used only in message strings, so evidence could PASS
        # while its own reason line was arithmetically false
        # (`acc 0.000 > escalate-all 0.900`). And `baseline` was unbounded:
        # at 5.0 the null easy-rate clamps to 1e-9 and everything looks
        # significant — fail-OPEN. The paired test is valid only because
        # `accuracy - baseline == (win - lose)/n`; enforce that here, where
        # the decision is made, not only in the test that checks
        # `evaluate()` produces it.
        # This also rejects NaN and ±inf — every comparison against NaN is
        # False, so `0.0 <= nan <= 1.0` fails. A separate isfinite() check
        # stood here for one round and was pure redundancy: mutation
        # testing showed deleting it changed no verdict. NaN reaching this
        # point at all was the original fail-open (`nan > 0.25` is False,
        # so a NaN false-easy rate passed the capability check).
        if not (0.0 <= _base <= 1.0 and 0.0 <= _acc <= 1.0 and 0.0 <= _fe <= 1.0):
            return False, "held-out evidence has out-of-range rates"

        # ── THE SIGNIFICANCE TEST ────────────────────────────────────────
        # Against always-predict-hard, the two predictors differ ONLY where
        # this model says "easy", so `accuracy - baseline == (win - lose)/n`
        # exactly. The question "is that edge real?" is therefore a question
        # about the discordant pairs alone.
        #
        # THE NULL IS NOT 0.5. Under "this model's easy-calls are
        # independent of the true label", an easy-call is genuinely easy
        # with probability (1 - hard_rate) — the population rate, not a coin
        # flip. Two earlier versions of this guard got that wrong in
        # opposite directions:
        #
        #   v1  1.645*sqrt(p(1-p)/n): substituted a CONSTANT for the number
        #       of discordant pairs, so it ignored how aggressive the model
        #       is — measured up to 3.6x too strict on a cautious model and
        #       1.5-1.9x too lax on an aggressive one.
        #   v2  McNemar at null 0.5, normal approximation: anti-conservative
        #       at small counts (3 wins vs 0 losses gives z=1.73 and passes,
        #       exact p=0.125) — measured deploying label-independent models
        #       1.9% of fits in THIS box's regime. And at hard-rates below
        #       0.5 its null is simply wrong: noise then beats always-hard
        #       by q(1-2p) in expectation, so it passes given enough n. The
        #       majority-class guard bolted on to patch that rejected
        #       genuinely significant models (McNemar z~8, false-easy inside
        #       the cap) — the §4AA failure the gate exists to prevent.
        #
        # Testing `win ~ Binomial(win+lose, 1-hard_rate)` is EXACT, needs no
        # approximation, is correctly calibrated at EVERY hard-rate, and so
        # replaces both the McNemar test and the majority-class guard.
        if _w is None or _l is None:
            # No counts, no evidence. A proxy fallback here would be a
            # SECOND standard for the same decision — and the branch is
            # reachable, since every pre-§4BQ checkpoint lacks the counts.
            return False, ("held-out evidence predates the significance "
                           "test (no discordant counts) — retraining")
        _w, _l = max(0, _w), max(0, _l)
        if (_w + _l) > _n_raw:
            return False, (f"evidence claims {_w + _l} differing turns out "
                           f"of {_n_raw} held-out — inconsistent")
        # Tolerance of ONE held-out turn (floored at 1e-3): `accuracy` and
        # `baseline` are stored rounded in checkpoint JSON, so an exact
        # comparison rejects genuine evidence — the shipped model's own
        # report is off by 3e-5 from rounding alone. This check is for
        # GROSS inconsistency (fabricated or mismatched fields), which is
        # off by whole turns, not for float equality.
        _tol = max(1e-3, 1.0 / max(1, _n_raw))
        if abs((_acc - _base) - (_w - _l) / max(1, _n_raw)) > _tol:
            return False, (
                f"evidence is internally inconsistent: accuracy-baseline "
                f"{_acc - _base:+.4f} != (win-lose)/n "
                f"{(_w - _l) / max(1, _n_raw):+.4f}")
        # A MINIMUM EFFECT SIZE. Significance alone let `win=4, lose=0`
        # deploy a model that differs from escalate-all on 4 of 445 turns
        # (p=0.043) — operationally meaningless, and measured to be the
        # regime where label-independent fits slip through (winning
        # margins 0.009-0.033). The router exists to change decisions; a
        # model that changes almost none has nothing to be right about.
        _min_disc = max(cls._GATE_MIN_DISCORDANT_FLOOR,
                        math.ceil(cls._GATE_MIN_DISCORDANT_FRACTION * _n))
        if (_w + _l) < _min_disc:
            return False, (
                f"only {_w + _l} of {_n} held-out turns are routed "
                f"differently from escalate-all (min {_min_disc}) — too "
                f"small a change to justify deploying")
        _p_easy = min(max(1.0 - _base, 1e-9), 1.0 - 1e-9)
        _pval = _binom_tail_ge(_w, _w + _l, _p_easy)
        if _pval >= cls._GATE_ALPHA:
            return False, (
                f"accuracy {_acc:.3f} vs escalate-all {_base:.3f} is not "
                f"significant: {_w} turns fixed vs {_l} broken, one-sided "
                f"p={_pval:.3f} against a {_p_easy:.3f} chance rate (n={_n})")

        if _fe > cls._GATE_MAX_FALSE_EASY:
            return False, (f"skips the planner on "
                           f"{_fe:.1%} of hard requests "
                           f"(max {cls._GATE_MAX_FALSE_EASY:.0%})")
        # COERCED values only — the raw dict is not touched past the
        # guard above, which is what let a string field crash the success
        # path after every rejection path had been made safe.
        return True, (f"acc {_acc:.3f} > escalate-all {_base:.3f}, "
                      f"{_w} fixed vs {_l} broken (p={_pval:.4f}), "
                      f"false-easy {_fe:.1%} (n={_n})")

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
        m.uses_embeddings_ = bool(self.uses_embeddings_)
        m.report_ = self.report_
        # Without this the copy carries no held-out evidence, so
        # `clone().looks_sane()` is False for ANY model — the gate is
        # fail-closed, so a clone could never be installed.
        m.gate_report_ = getattr(self, "gate_report_", None)
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
        X_arr, y_arr = self._to_arrays(
            X, y, expect=int(self.weights_.shape[0]))
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
        X_arr, y_arr = self._to_arrays(
            X, y, expect=int(self.weights_.shape[0]))
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
        vec = self._vectorize(x, expect=int(self.weights_.shape[0]))
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

    @staticmethod
    def _embed_model_name() -> str:
        """Live embedder identity, or "" when it cannot be determined.

        Imported lazily: `router/model.py` must stay importable without
        pulling the vector store (and chromadb) in behind it.
        """
        try:
            from .embedding import current_embed_model_name
            return current_embed_model_name()
        except Exception:  # noqa: BLE001
            return ""

    def predict_from_text(
        self,
        text: str,
        prior_turn_text: str = "",
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> Tuple[str, float]:
        """Convenience: extract features and predict in one call.

        An embedding model REQUIRES `embedding=`; without it the width
        guard raises FeatureSchemaMismatch rather than scoring a
        lexical-only vector against embedding weights. Production routes
        through ComplexityDispatcher, which supplies it and escalates on
        failure — this stays a convenience for tests and scripts.
        """
        fv = extract_features(text, prior_turn_text=prior_turn_text,
                              embedding=embedding)
        return self.predict(fv)

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        if self.weights_ is None:
            raise RuntimeError("cannot save untrained classifier")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # UNIQUE tmp name, for the reason `router/trainer._record_look`
        # documents next door: with a shared one, concurrent writers
        # rename it out from under each other. Measured here, 40 trials
        # of 8 concurrent saves lost 210 of 320 writes (66%), each with
        # a "router classifier save failed" warning. Concurrent trainers
        # are reachable — they clear the looks block via TOCTOU and then
        # collide on this name.
        import os as _os, uuid as _uuid
        tmp = p.with_suffix(
            f"{p.suffix}.{_os.getpid()}.{_uuid.uuid4().hex[:8]}.tmp")
        payload = {
            "schema": "ghost.router.logreg.v1",
            "feature_names": list(self.feature_names_),
            # Diagnostic only — `feature_names` above is authoritative and
            # is what load() matches on. Recorded so an operator reading
            # the JSON can see the representation without counting names.
            "uses_embeddings": bool(self.uses_embeddings_),
            # NOT diagnostic: WHICH embedder produced the training
            # vectors. Width alone cannot tell bge-small from MiniLM
            # (both 384-d, both in the local HF cache), so a
            # GHOST_EMBED_MODEL migration would otherwise leave this
            # checkpoint scoring one model's weights against another
            # model's vectors — silently, and confidently wrong.
            "embed_model": (self._embed_model_name()
                            if self.uses_embeddings_ else None),
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
        # §4BQ: TWO schemas are legitimate — lexical-only (pre-flip, and
        # what the kill switch trains) and lexical+embedding. Both are
        # matched EXACTLY, in full, including order: a checkpoint whose
        # names were reordered or renamed at the same length used to load
        # clean and then dot old-order weights against new-order vectors,
        # routing silently wrongly. Anything else fails loud here so the
        # boot loader falls back to the safe escalate-all dispatcher and
        # retrains over the bad file.
        _schemas = {
            model_feature_names(False): False,
            model_feature_names(True): True,
        }
        if names not in _schemas or clf.weights_.shape[0] != len(names):
            _known = " or ".join(str(len(k)) for k in _schemas)
            raise ValueError(
                f"router checkpoint at {p} was trained on a different feature "
                f"schema (checkpoint has {len(names)} names / "
                f"{clf.weights_.shape[0]} weights vs a known {_known}) — "
                "refusing to load a misaligned model; it will be retrained."
            )
        clf.uses_embeddings_ = _schemas[names]
        # EMBEDDER IDENTITY. Fail-closed: a checkpoint that cannot name
        # its embedder (one written before this guard, or hand-built) is
        # rejected exactly like a mismatched one, because "unknown" is
        # not evidence of "the same". Rejection costs a retrain; the
        # alternative costs confidently wrong routing that nothing
        # detects.
        if clf.uses_embeddings_:
            saved_model = raw.get("embed_model")
            current = clf._embed_model_name()
            if not saved_model or str(saved_model) != current:
                raise ValueError(
                    f"router checkpoint at {p} was trained with embedder "
                    f"{saved_model!r} but {current!r} is live — refusing to "
                    "score one model's weights against another model's "
                    "vectors; it will be retrained."
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

    def _vectorize(self, x: Any, *, expect: Optional[int] = None) -> np.ndarray:
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
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        # §4BQ WIDTH GUARD, applied only where a width is EXPECTED — i.e.
        # scoring against existing weights. Deliberately not during fit:
        # `fit` is entitled to replace the representation, and checking
        # against the outgoing weights made re-fitting a classifier onto a
        # different schema impossible.
        #
        # np.dot would raise on a plain mismatch anyway, but only
        # sometimes with a comprehensible message — and the `str` branch
        # above silently produces a LEXICAL-ONLY vector, which an
        # embedding model would otherwise consume as though the caller had
        # supplied an embedding. One named, catchable error instead, so
        # the dispatcher escalates rather than routing on a short vector.
        # `ndim` is checked too: a (402, 1) column matched on shape[0],
        # produced a shape-(1,) dot product, and float() accepted it — a
        # wrong-shaped input scoring cleanly through the one guard that
        # exists to stop exactly that.
        if expect is not None and (vec.ndim != 1 or vec.shape[0] != expect):
            raise FeatureSchemaMismatch(
                f"model expects a 1-D vector of {expect} "
                f"features, got shape {tuple(vec.shape)}"
                + (" — this model needs an embedding (see "
                   "router/embedding.py)" if self.uses_embeddings_ else "")
            )
        return vec

    def _to_arrays(
        self,
        X: Iterable[Any],
        y: Iterable[str],
        *,
        expect: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """`expect` enforces a width; pass it from every path that REFINES
        existing weights (partial_fit, bce_loss) and omit it from `fit`,
        which is entitled to adopt a new representation."""
        X_list = [self._vectorize(x, expect=expect) for x in X]
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
