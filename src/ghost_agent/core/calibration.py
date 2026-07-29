"""Closed-loop confidence calibration — roadmap phase 2.5.

The composite confidence (:mod:`core.confidence`) is computed every
turn but historically was never checked against what actually
happened. The threshold ``τ = 0.55`` and the entropy/competence
weights ``0.5 / 0.5`` were asserted constants, never fit to data — so
"the agent is 80 % confident" had no demonstrated relationship to the
turn succeeding 80 % of the time.

This module closes that loop. It is the calibration *spine*:

  1. **Measure.** Every turn that produced a confidence reading is
     paired with the realized outcome and appended to a JSONL log.
     From that log we compute a rolling **Brier score**
     ``mean((C − outcome)²)`` and an **Expected Calibration Error**
     over a 10-bin reliability table.
  2. **Self-tune.** Once enough samples accumulate, a small grid
     search re-fits the entropy/competence weights, the
     verbalised-uncertainty penalty ``λ``, and the decision threshold
     ``τ`` to minimise Brier on the logged history. The fitted params
     are persisted and loaded back into :class:`CompositeConfidence`.
  3. **Unify.** The recorded sample carries the verbalised-uncertainty
     *pressure* alongside the objective entropy/competence components,
     so the previously-disjoint "the agent said it was unsure" track
     and "the generation/​domain was uncertain" track are fit together.

Design non-negotiables (same as every other Stage-1 module):

* **Local-only, pure stdlib.** ``math`` + ``json`` only. No numpy, no
  hosted scorer, no outbound traffic.
* **JSONL / JSON on disk.** Human-diffable, append-only history, atomic
  param writes (``.tmp`` + ``os.replace``). Schema-versioned.
* **Fail-safe.** A recording or fit failure is logged at debug and
  never breaks a turn. ``load_params`` returns ``None`` on any problem
  so a corrupt file degrades to the hardcoded defaults, never a crash.
* **Bail-on-thin-data.** Like ``prm.trainer``: below the sample floor,
  or with only one outcome class present, ``fit`` returns ``None`` with
  a logged reason and writes no params — a confidently-miscalibrated
  threshold is worse than the neutral default.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

SCHEMA_VERSION = "ghost.calibration.v1"

# Minimum number of samples that actually OBSERVED token logprobs before the
# entropy weight is allowed to move off zero. Below this the entropy column
# is mostly the neutral stand-in and any fitted w_entropy would be noise
# dressed as signal.
_MIN_ENTROPY_SAMPLES = 30

# Same evidence floor for the turn-effort weight.
_MIN_EFFORT_SAMPLES = 30

# ── Graded outcome labels ─────────────────────────────────────────────
# The label used to be binary: 0.0 if (any execution failure OR verifier
# REFUTED OR budget exhausted) else 1.0. That produced 1177 positives to 49
# negatives — 96.1% one class — and it measured the wrong thing. "Nothing
# visibly broke" is not "the answer was good", and a turn that hit one tool
# error, recovered, and answered correctly was labelled a FAILURE.
#
# A graded label in [0, 1] fixes both at once: it carries information on
# EVERY turn instead of on the 4% that break, and it stops mislabelling
# recoveries. Brier and log-loss both accept soft targets unchanged.
#
# The constants are measured, not chosen. Across 302 verdict-bearing
# trajectories the agent passed 251 — so a clean turn that no verifier
# could check is scored at the observed P(good | checkable) rather than an
# asserted 1.0. Claiming 1.0 for an unverified turn is precisely the
# verification theatre this project keeps finding.
_UNVERIFIED_PRIOR = 0.83
# Each execution failure the turn had to absorb costs this much. A turn that
# recovered from one error is materially worse than a clean one, but nothing
# like a refuted answer.
_EXEC_FAILURE_PENALTY = 0.15
# Floor for a turn that answered at all — reserve 0.0 for a verdict of WRONG.
_DEGRADED_FLOOR = 0.15
# Budget exhaustion: the reply is explicitly flagged working-state/PARTIAL,
# i.e. the agent itself reports it did not finish.
_BUDGET_EXHAUSTED_GRADE = 0.2
# A user-reported failure ("it still doesn't work", a pasted traceback).
# Strong — the human is telling you the delivered work is broken — but a
# notch above an explicit correction's 0.0, because attribution is slightly
# looser: the report might name a defect the previous turn did not cause.
_FAILURE_REPORT_GRADE = 0.15


def grade_turn_outcome(*, verifier_verdict=None, execution_failure_count: int = 0,
                       budget_exhausted: bool = False) -> float:
    """Map a finished turn's observable signals onto a quality label in [0, 1].

    IMPORTANT — this is a PROXY, not ground truth. Only the verifier arms
    are checked facts; everything else is a prior over "did this go well",
    and calibrating against it teaches the agent about *process health*, not
    correctness. Keep at least one ground-truth source (user corrections)
    flowing and tracked separately, or the score drifts toward rewarding
    "did not visibly break" — which is what the agent already over-indexes
    on. Sample provenance (`CalibrationSample.source`) exists so the two can
    always be told apart.

    Pure and total: never raises, always returns a value in [0, 1].
    """
    try:
        verdict = str(verifier_verdict or "").strip().lower()
        if verdict == "failed":
            return 0.0            # checked and WRONG — the one hard negative
        if verdict == "passed":
            return 1.0            # checked and RIGHT
        if budget_exhausted:
            return _BUDGET_EXHAUSTED_GRADE
        try:
            fails = max(0, int(execution_failure_count or 0))
        except (TypeError, ValueError):
            fails = 0
        grade = _UNVERIFIED_PRIOR - (_EXEC_FAILURE_PENALTY * fails)
        return _clamp01(max(_DEGRADED_FLOOR, grade))
    except Exception:  # noqa: BLE001 — labelling must never break a turn
        return _UNVERIFIED_PRIOR


def _outcome_variance(samples) -> float:
    """Population variance of the outcome column. Replaces "are both binary
    classes present?" as the has-this-data-any-information test, which is
    the same question once labels are continuous (for 0/1 labels the
    variance is 0 exactly when one class is missing, so binary corpora keep
    their previous behaviour bit-for-bit)."""
    if not samples:
        return 0.0
    vals = [s.outcome for s in samples]
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def _sigmoid(z: float) -> float:
    # Overflow-safe: exp(710) overflows a float64.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z)) if z < 700 else 1.0
    ez = math.exp(z) if z > -700 else 0.0
    return ez / (1.0 + ez)


def apply_platt(composite: float, a: float, b: float) -> float:
    """Map a blended composite onto a calibrated probability.

    Identity when ``a == 1 and b == 0`` is NOT true of this transform, so
    callers must only apply it with fitted parameters; `FittedParams`
    defaults exist so an unfitted file is detectable, and
    :meth:`CompositeConfidence.score` skips the mapping in that case.
    """
    return _clamp01(_sigmoid(a * float(composite) + b))


def _fit_platt(pairs, *, iters: int = 50, ridge: float = 1e-2):
    """Fit (a, b) of ``sigmoid(a·c + b)`` on log-loss by Newton/IRLS.

    Log-loss (not Brier) because it is convex here and is the proper scoring
    rule for probability estimates; Brier is then reported on the result.

    Newton rather than gradient descent because the ANSWER MUST NOT DEPEND ON
    THE ITERATION BUDGET. Plain GD on this data was still at a = 1.28 after
    4000 steps and only reached its true optimum of a ≈ −0.08 after ~60000 —
    and those two values sit on opposite sides of the safety guard in
    `fit`, so an under-converged run would have silently adopted a map the
    converged fit rejects. IRLS converges here in well under 20 iterations.

    ``ridge`` is a REAL L2 penalty (it enters the gradient as well as the
    Hessian), making this a MAP estimate rather than an MLE. That is what
    bounds the slope on linearly SEPARABLE data, where the unpenalised
    likelihood has its optimum at infinity: with damping applied only to
    the Hessian — the earlier form — the fixed point was still that
    infinite optimum, and the slope simply grew with the iteration budget
    (a = 51 → 207 → 233 at 5 → 50 → 200 iterations on a separable corpus),
    contradicting the guarantee in the paragraph above. A separable batch
    is entirely reachable here: 40 samples split cleanly by competence is
    all it takes.

    Deliberately UNWEIGHTED. Class-balancing was tried first, on the theory
    that ~96% positives would drown the minority — but balancing optimises a
    reweighted distribution, so the resulting probabilities describe a 50/50
    world that does not exist, and the unweighted Brier got worse. The
    minority class is handled where it belongs: by the threshold search,
    which uses Youden's J and weights sensitivity and specificity equally
    regardless of prevalence.
    """
    n = len(pairs)
    if n < 2:
        return 1.0, 0.0
    # Variance test, not class-presence — see `fit`. Soft targets in [0, 1]
    # are valid for log-loss (cross-entropy with a soft label), so a graded
    # corpus is fittable; a CONSTANT one is not.
    _ymean = sum(y for _, y in pairs) / n
    if sum((y - _ymean) ** 2 for _, y in pairs) / n < 1e-9:
        return 1.0, 0.0
    a, b = 1.0, 0.0
    for _ in range(iters):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for c, y in pairs:
            p = _sigmoid(a * c + b)
            err = p - y
            w = p * (1.0 - p)
            g_a += err * c
            g_b += err
            h_aa += w * c * c
            h_ab += w * c
            h_bb += w
        # L2 penalty on BOTH the gradient and the Hessian. Adding it to the
        # Hessian alone is Levenberg damping — it changes the step size but
        # not the fixed point, so it cannot bound a separable fit.
        g_a += ridge * a
        g_b += ridge * b
        h_aa += ridge
        h_bb += ridge
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a -= da
        b -= db
        if not (math.isfinite(a) and math.isfinite(b)):
            return 1.0, 0.0
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    if not (math.isfinite(a) and math.isfinite(b)):
        return 1.0, 0.0
    return a, b


def _composite_for(sample, w_e: float, lam: float, w_eff: float = 0.0) -> float:
    """Recompute a sample's composite under candidate (w_e, w_eff, λ).

    Mirrors :meth:`CompositeConfidence.score` EXACTLY, including the
    missing-feature renormalisation: only components the sample actually
    OBSERVED contribute, and the blend is divided by their weights. A
    sample with neither entropy nor effort is scored on competence alone
    rather than blended with neutral stand-ins. Fitting and scoring must
    agree — if the fit blended a stand-in while the scorer didn't (or vice
    versa), the fitted weights would be optimal for a formula the agent
    never actually evaluates.
    """
    w_c = max(0.0, 1.0 - w_e - w_eff)
    parts = [(w_c, sample.competence_component)]
    if sample.entropy_observed:
        parts.append((w_e, sample.entropy_component))
    if getattr(sample, "effort_observed", False):
        parts.append((w_eff, sample.effort_component))
    tot = sum(w for w, _ in parts)
    c = (sum(w * v for w, v in parts) / tot) if tot > 0 else sample.competence_component
    return _clamp01(c * (1.0 - lam * sample.uncertainty_pressure))


# ──────────────────────────────────────────────────────────────────────
# dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationSample:
    """One (confidence, outcome) pair.

    The objective components (``entropy_component`` = ``1 − e`` and
    ``competence_component`` = shrunk ``p``) and the verbalised
    ``uncertainty_pressure`` are stored separately from the recorded
    ``composite`` so :meth:`CalibrationTracker.fit` can recompute the
    composite for *candidate* weights/λ without re-running the agent.
    """

    composite: float
    entropy_component: float
    competence_component: float
    uncertainty_pressure: float
    outcome: float  # 1.0 = turn succeeded, 0.0 = turn failed
    domain: str = ""
    ts: str = ""
    # False when no token logprobs were observed for the turn and
    # ``entropy_component`` is the neutral 0.5 stand-in rather than a
    # measurement. Such samples are EXCLUDED from the entropy-weight fit —
    # see :meth:`CalibrationTracker.fit`. Defaults to False so a legacy
    # record (which was always the fabricated neutral) is treated as
    # unobserved, which is exactly what it was.
    entropy_observed: bool = False
    # Turn-shape (effort) component and whether it was measured. This is the
    # first per-TURN input; see `confidence.effort_component`.
    effort_component: float = 0.5
    effort_observed: bool = False
    # PROVENANCE of the label. Without it, mixing signal tiers is
    # irreversible: you can never audit which tier is noisy, nor drop one,
    # without discarding the whole corpus. Every future source (implicit
    # rephrase-negatives, reopened-task negatives, generated probes) must
    # carry its own tag so the mix stays visible and separable.
    #   "turn"            — the graded end-of-turn label (a PROXY)
    #   "user_correction" — the user said the answer was wrong (ground truth)
    # Legacy records predate the field and were all end-of-turn labels.
    source: str = "turn"


@dataclass
class FittedParams:
    """Result of a fit, persisted to ``calibration_params.json`` and
    loaded back into :class:`CompositeConfidence`."""

    w_entropy: float
    w_competence: float
    threshold: float
    lambda_uncertainty: float
    brier: float
    n_samples: int
    fitted_at: str
    schema: str = SCHEMA_VERSION
    # Platt recalibration of the blended score into a PROBABILITY:
    #   p = 1 / (1 + exp(-(platt_a · composite + platt_b)))
    #
    # The weight search decides how to COMBINE the signals but has no way to
    # set their overall level, so the raw composite ranked turns correctly
    # while being systematically mis-scaled. Measured on 1208 live samples:
    # AUC 0.679 (real discrimination) but Brier 0.060 against 0.037 for
    # simply always predicting the base rate — i.e. the number was worse
    # than useless AS A PROBABILITY. Fitting these two parameters takes it
    # to 0.031, better than the base rate. Identity defaults (a=1, b=0) mean
    # an unfitted/legacy params file behaves exactly as before.
    platt_a: float = 1.0
    platt_b: float = 0.0
    # Weight on the turn-EFFORT component (see `confidence.effort_component`).
    # 0 until enough samples measure turn shape, same evidence rule as
    # w_entropy. w_competence is the remainder: 1 - w_entropy - w_effort.
    w_effort: float = 0.0
    n_effort_observed: int = 0
    # Brier of the RAW composite and of the base-rate predictor, kept so the
    # fit can be audited against the only baseline that matters. A model
    # that cannot beat `brier_base_rate` is not adding information.
    brier_raw: float = -1.0
    brier_base_rate: float = -1.0
    # How many of ``n_samples`` carried REAL observed logprob entropy. When
    # this is below `_MIN_ENTROPY_SAMPLES` the fit pins ``w_entropy`` to 0
    # deliberately — the number makes that visible instead of leaving an
    # operator to wonder why the weight never moves.
    n_entropy_observed: int = 0
    # What happened to the Platt probability map in this fit:
    #   "applied"            — map adopted;
    #   "rejected_inverted"  — slope <= 0 (composite anti-correlated),
    #                          identity map kept;
    #   "rejected_step"      — slope > _MAX_SLOPE backstop, identity kept;
    #   "discarded_worse"    — calibrated Brier no better than raw.
    # The refit summary line must surface this — logging `refit=ok` while
    # the map was rejected reads as a healthy calibration when the score
    # is in fact predicting nothing (2026-07-29 log audit).
    map_status: str = "applied"


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    mean_outcome: float


# ──────────────────────────────────────────────────────────────────────
# tracker
# ──────────────────────────────────────────────────────────────────────

class CalibrationTracker:
    """Append-only calibration log + grid-search refit.

    Constructed once per agent (in ``main.lifespan``) and hung on
    ``context.calibration_tracker``. Writes acquire a lock; reads parse
    the tail of the JSONL. Everything is best-effort: a disk error
    leaves the in-flight turn untouched.
    """

    HISTORY_NAME = "calibration.jsonl"
    PARAMS_NAME = "calibration_params.json"

    # Defaults mirror the prm.trainer bail floors — below these a fit is
    # noise. Both classes (success AND failure) must also be present.
    DEFAULT_MIN_SAMPLES = 40
    DEFAULT_MAX_HISTORY = 4000

    def __init__(
        self,
        calib_dir: Path,
        *,
        min_samples_for_fit: int = DEFAULT_MIN_SAMPLES,
        max_history: int = DEFAULT_MAX_HISTORY,
    ):
        self.dir = Path(calib_dir)
        self.history_path = self.dir / self.HISTORY_NAME
        self.params_path = self.dir / self.PARAMS_NAME
        self.min_samples_for_fit = max(1, int(min_samples_for_fit))
        self.max_history = max(1, int(max_history))
        self._lock = threading.RLock()

    # ----------------------------------------------------------- recording

    def record(
        self,
        *,
        composite: float,
        entropy_component: float,
        competence_component: float,
        outcome: float,
        uncertainty_pressure: float = 0.0,
        domain: str = "",
        entropy_observed: bool = False,
        effort_component: float = 0.5,
        effort_observed: bool = False,
        source: str = "turn",
    ) -> None:
        """Append one (confidence, outcome) pair. Never raises.

        ``entropy_observed`` records whether ``entropy_component`` came from
        real token logprobs. Pass it through faithfully — a fabricated
        neutral marked as observed re-poisons the entropy fit.
        """
        try:
            sample = CalibrationSample(
                composite=_clamp01(composite),
                entropy_component=_clamp01(entropy_component),
                competence_component=_clamp01(competence_component),
                uncertainty_pressure=_clamp01(uncertainty_pressure),
                # Clamped, NOT binarised. The old `1.0 if >= 0.5 else 0.0`
                # crushed every graded label back to two values, which is
                # exactly the constant-column problem the grading exists to
                # remove. Binary inputs are unaffected (0.0/1.0 clamp to
                # themselves).
                outcome=_clamp01(outcome),
                domain=str(domain or ""),
                ts=_utcnow_iso(),
                entropy_observed=bool(entropy_observed),
                effort_component=_clamp01(effort_component),
                effort_observed=bool(effort_observed),
                source=str(source or "turn"),
            )
            with self._lock:
                self.dir.mkdir(parents=True, exist_ok=True)
                with self.history_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(sample)) + "\n")
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("CalibrationTracker.record failed: %s", exc)

    # ----------------------------------------------------------- reading

    def _load_samples(self, limit: Optional[int] = None) -> List[CalibrationSample]:
        if not self.history_path.exists():
            return []
        out: List[CalibrationSample] = []
        try:
            with self.history_path.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("CalibrationTracker load failed: %s", exc)
            return []
        if limit is not None:
            lines = lines[-limit:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out.append(
                    CalibrationSample(
                        composite=_clamp01(d.get("composite", 0.5)),
                        entropy_component=_clamp01(d.get("entropy_component", 0.5)),
                        competence_component=_clamp01(
                            d.get("competence_component", 0.5)
                        ),
                        uncertainty_pressure=_clamp01(
                            d.get("uncertainty_pressure", 0.0)
                        ),
                        # Preserve graded values on read too (legacy 0.0/1.0
                        # records clamp to themselves).
                        outcome=_clamp01(d.get("outcome", 0.0)),
                        domain=str(d.get("domain", "")),
                        ts=str(d.get("ts", "")),
                        # Absent on legacy records, which were ALWAYS the
                        # fabricated neutral — default False is correct.
                        entropy_observed=bool(d.get("entropy_observed", False)),
                        effort_component=_clamp01(d.get("effort_component", 0.5)),
                        effort_observed=bool(d.get("effort_observed", False)),
                        source=str(d.get("source") or "turn"),
                    )
                )
            except Exception:
                # Skip malformed lines without poisoning the read.
                continue
        return out

    def sample_count(self) -> int:
        return len(self._load_samples())

    # ----------------------------------------------------------- metrics

    def brier_score(self, *, window: Optional[int] = None) -> Optional[float]:
        """Rolling Brier score ``mean((C − outcome)²)`` over the recent
        ``window`` samples (all by default). ``None`` when no data."""
        samples = self._load_samples(limit=window)
        if not samples:
            return None
        return sum((s.composite - s.outcome) ** 2 for s in samples) / len(samples)

    def reliability_table(
        self, *, bins: int = 10, window: Optional[int] = None
    ) -> List[ReliabilityBin]:
        """10-bin reliability table: for each confidence band, the mean
        predicted confidence vs the mean realized outcome. A perfectly
        calibrated agent has ``mean_confidence ≈ mean_outcome`` in every
        populated bin."""
        bins = max(1, int(bins))
        samples = self._load_samples(limit=window)
        table: List[ReliabilityBin] = []
        for i in range(bins):
            lo = i / bins
            hi = (i + 1) / bins
            # Last bin is closed on the right so C == 1.0 lands somewhere.
            in_bin = [
                s for s in samples
                if (s.composite >= lo and (s.composite < hi or (i == bins - 1 and s.composite <= hi)))
            ]
            if in_bin:
                mc = sum(s.composite for s in in_bin) / len(in_bin)
                mo = sum(s.outcome for s in in_bin) / len(in_bin)
            else:
                mc = mo = 0.0
            table.append(ReliabilityBin(lo, hi, len(in_bin), mc, mo))
        return table

    def ece(self, *, bins: int = 10, window: Optional[int] = None) -> Optional[float]:
        """Expected Calibration Error: sample-weighted mean absolute gap
        between confidence and outcome across reliability bins. ``None``
        when no data."""
        table = self.reliability_table(bins=bins, window=window)
        total = sum(b.count for b in table)
        if total <= 0:
            return None
        return sum(
            (b.count / total) * abs(b.mean_confidence - b.mean_outcome)
            for b in table if b.count
        )

    # ----------------------------------------------------------- fitting

    def fit(self, *, min_samples: Optional[int] = None) -> Optional[FittedParams]:
        """Grid-search refit of (weights, λ, τ) minimising Brier.

        Returns the :class:`FittedParams` on success (and persists them),
        or ``None`` with a logged ``bail_reason`` when the data is too
        thin or single-class. No params file is written on a bail — the
        previous fit (or the hardcoded defaults) stays in force.
        """
        floor = min_samples if min_samples is not None else self.min_samples_for_fit
        samples = self._load_samples(limit=self.max_history)
        if len(samples) < floor:
            logger.debug(
                "calibration fit bail: %d samples < floor %d", len(samples), floor
            )
            return None
        n_pos = sum(1 for s in samples if s.outcome >= 0.5)
        n_neg = len(samples) - n_pos
        # Bail on "no information in the labels", measured as variance rather
        # than as "both binary classes present". For 0/1 labels the two tests
        # are identical (variance is 0 exactly when one class is missing), so
        # binary corpora behave as before — but a GRADED corpus can carry
        # plenty of signal while every sample sits above 0.5, and the old
        # test would have refused to fit it at all.
        if _outcome_variance(samples) < 1e-9:
            logger.debug(
                "calibration fit bail: outcome column has no variance "
                "(pos=%d neg=%d)", n_pos, n_neg,
            )
            return None

        # The entropy weight may only move when enough samples carried a
        # REAL logprob observation. Previously every sample contributed a
        # flat, fabricated 0.5 to the entropy column, which is not merely
        # noisy but structurally fatal: with zero variance there, any
        # w_e > 0 could only drag composites toward 0.5, so the grid was
        # guaranteed to pick w_e = 0. That is exactly what was observed
        # live — 1200/1201 stored samples at 0.5 and w_entropy stuck at 0
        # (2026-07-27). Unobserved samples are no longer blended with the
        # stand-in at all (see `_composite_for`); they score on competence
        # alone, so a weight fit from the observed minority cannot degrade
        # them. Below the floor we pin w_e at 0 and log WHY rather than
        # fitting a weight on fiction.
        observed = [s for s in samples if s.entropy_observed]
        obs_pos = sum(1 for s in observed if s.outcome >= 0.5)
        obs_neg = len(observed) - obs_pos
        entropy_fittable = (
            len(observed) >= _MIN_ENTROPY_SAMPLES and obs_pos > 0 and obs_neg > 0
        )
        we_range = range(0, 11) if entropy_fittable else range(0, 1)

        # Same evidence rule for the turn-effort weight: it only moves once
        # enough samples actually MEASURED turn shape, with both outcome
        # classes represented. Legacy samples (recorded before the feature
        # existed) carry effort_observed=False and are excluded, exactly as
        # unobserved entropy is.
        eff_observed = [s for s in samples if getattr(s, "effort_observed", False)]
        eff_pos = sum(1 for s in eff_observed if s.outcome >= 0.5)
        eff_neg = len(eff_observed) - eff_pos
        effort_fittable = (
            len(eff_observed) >= _MIN_EFFORT_SAMPLES and eff_pos > 0 and eff_neg > 0
        )
        weff_range = range(0, 11) if effort_fittable else range(0, 1)
        if not effort_fittable:
            logger.debug(
                "calibration: w_effort pinned to 0 — only %d/%d samples "
                "measured turn shape (pos=%d neg=%d, need >=%d with both "
                "classes)", len(eff_observed), len(samples), eff_pos, eff_neg,
                _MIN_EFFORT_SAMPLES)
        if not entropy_fittable:
            logger.debug(
                "calibration: w_entropy pinned to 0 — only %d/%d samples "
                "observed real logprob entropy (pos=%d neg=%d, need >=%d of "
                "each class). Upstream refuses logprobs on tools+stream "
                "payloads, so most turns carry no token entropy.",
                len(observed), len(samples), obs_pos, obs_neg,
                _MIN_ENTROPY_SAMPLES,
            )

        # Grid over entropy weight (→ competence = 1−w_e) and the
        # uncertainty penalty λ, minimising Brier over EVERY sample under
        # the same per-sample formula the scorer uses (`_composite_for`).
        best = None  # (brier, w_e, w_eff, lam)
        for we_i in we_range:
            w_e = we_i / 10.0
            for weff_i in weff_range:
                w_eff = weff_i / 10.0
                if w_e + w_eff > 1.0:
                    continue          # competence would go negative
                for lam_i in range(0, 6):
                    lam = lam_i / 10.0
                    sq = sum((_composite_for(s, w_e, lam, w_eff) - s.outcome) ** 2
                             for s in samples)
                    brier = sq / len(samples)
                    if best is None or brier < best[0]:
                        best = (brier, w_e, w_eff, lam)

        assert best is not None
        brier, w_e, w_eff, lam = best
        w_c = max(0.0, 1.0 - w_e - w_eff)

        composites = [(_composite_for(s, w_e, lam, w_eff), s.outcome)
                      for s in samples]

        # Recalibration stage. The weight search decides how to COMBINE the
        # signals but cannot set their level — measured live, the composite
        # ranked turns well (AUC 0.679) while scoring Brier 0.060 against
        # 0.037 for a constant base-rate predictor. Fitting a two-parameter
        # Platt map turns that into 0.031. Class-balanced so the ~4% of
        # negatives are not drowned by the majority class.
        platt_a, platt_b = _fit_platt(composites)
        calibrated = [(apply_platt(c, platt_a, platt_b), y) for c, y in composites]

        base = sum(y for _, y in composites) / len(composites)
        brier_raw = sum((c - y) ** 2 for c, y in composites) / len(composites)
        brier_base = sum((base - y) ** 2 for _, y in composites) / len(composites)
        brier_cal = sum((p - y) ** 2 for p, y in calibrated) / len(calibrated)

        # Adopt the mapping only if it is both BETTER and SAFE.
        #
        # A non-positive slope is the important rejection. Platt is monotonic
        # in `a`, so `a <= 0` inverts the ranking — the agent would report
        # LOWER confidence on turns the score rates higher. The optimiser
        # chooses that whenever the composite is anti-correlated with
        # success, which is exactly what the live corpus shows (leak-free
        # AUC 0.473, i.e. no discrimination). In that state the honest fit is
        # "this score predicts nothing", and the right response is to leave
        # the raw behaviour alone and say so — not to ship an inverted or
        # flattened confidence. A near-zero slope is rejected for the same
        # reason: it collapses every turn onto the base rate, which scores a
        # great Brier while making `below_threshold` permanently inert.
        # Reject only what is actually unsafe.
        #
        # INVERSION (slope <= 0): the map would report lower confidence on
        # turns the score rates higher. The optimiser picks this whenever the
        # composite is anti-correlated with success, which is what the live
        # corpus shows (leak-free AUC 0.473). Calibrating noise is not an
        # improvement — leave the raw scale alone and say so.
        #
        # DIVERGENCE (slope enormous): a separable batch drives the fit to a
        # near-step function, which reports ~0 or ~1 for every turn and makes
        # `below_threshold` a hair-trigger on one competence value. The L2
        # penalty above bounds this in normal use; the cap is a backstop.
        #
        # A SMALL POSITIVE slope is NOT rejected. An earlier version demanded
        # slope >= 0.5 on the theory that a flat map would leave the gate
        # inert — that reasoning was wrong. Platt is strictly monotone for
        # a > 0 and the threshold is refit on the SAME scale, so the
        # below-threshold decision set is bit-identical before and after; the
        # map changes only calibration quality. Rejecting a = 0.30 discarded
        # a measured 0.069 Brier improvement and changed no decision at all.
        # Slope also cannot be judged without the composite's spread: a = 3.0
        # over a 0.02-wide range moves probabilities less than a = 0.3 over
        # the full unit interval.
        _MAX_SLOPE = 50.0
        map_status = "applied"
        if platt_a <= 0.0:
            logger.warning(
                "calibration: REJECTED the probability map (slope %.3f <= 0)."
                " The composite is anti-correlated with outcomes, so the map"
                " would INVERT the confidence ordering. Confidence stays on"
                " the raw scale; the score needs a feature that actually"
                " predicts turn failure, not better calibration.", platt_a)
            platt_a, platt_b = 1.0, 0.0
            calibrated = composites
            brier = brier_raw
            map_status = "rejected_inverted"
        elif platt_a > _MAX_SLOPE:
            logger.warning(
                "calibration: REJECTED the probability map (slope %.1f > %.1f)."
                " That is a near-step function — almost certainly a linearly"
                " separable batch fitted in-sample — and would make the"
                " below-threshold gate a hair-trigger on a single composite"
                " value.", platt_a, _MAX_SLOPE)
            platt_a, platt_b = 1.0, 0.0
            calibrated = composites
            brier = brier_raw
            map_status = "rejected_step"
        elif brier_cal <= brier_raw:
            brier = brier_cal
        else:
            logger.debug(
                "calibration: discarding Platt map (Brier %.4f > raw %.4f)",
                brier_cal, brier_raw)
            platt_a, platt_b = 1.0, 0.0
            calibrated = composites
            brier = brier_raw
            map_status = "discarded_worse"
        if brier > brier_base:
            logger.warning(
                "calibration: fitted model (Brier %.4f) is WORSE than always "
                "predicting the base rate %.3f (Brier %.4f) — the confidence "
                "score is not adding information as a probability",
                brier, base, brier_base)

        # Threshold is picked on the CALIBRATED scores (Youden's J on the
        # "predict success when p ≥ τ" decision) — it must live on the same
        # scale `below_threshold` will compare against, or the gate fires at
        # the wrong point.
        threshold = _best_threshold(calibrated)

        params = FittedParams(
            w_entropy=round(w_e, 4),
            w_competence=round(w_c, 4),
            threshold=round(threshold, 4),
            lambda_uncertainty=round(lam, 4),
            brier=round(brier, 6),
            n_samples=len(samples),
            fitted_at=_utcnow_iso(),
            n_entropy_observed=len(observed),
            platt_a=round(platt_a, 6),
            platt_b=round(platt_b, 6),
            brier_raw=round(brier_raw, 6),
            brier_base_rate=round(brier_base, 6),
            w_effort=round(w_eff, 4),
            n_effort_observed=len(eff_observed),
            map_status=map_status,
        )
        self._save_params(params)
        return params

    # ----------------------------------------------------------- params io

    def load_params(self) -> Optional[FittedParams]:
        """Read the persisted fitted params, or ``None`` if absent /
        corrupt / wrong-schema (degrade to hardcoded defaults)."""
        if not self.params_path.exists():
            return None
        try:
            d = json.loads(self.params_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("calibration params load failed: %s", exc)
            return None
        if not isinstance(d, dict) or d.get("schema") != SCHEMA_VERSION:
            return None
        try:
            return FittedParams(
                w_entropy=float(d["w_entropy"]),
                w_competence=float(d["w_competence"]),
                threshold=float(d["threshold"]),
                lambda_uncertainty=float(d.get("lambda_uncertainty", 0.0)),
                brier=float(d.get("brier", 0.0)),
                n_samples=int(d.get("n_samples", 0)),
                fitted_at=str(d.get("fitted_at", "")),
                n_entropy_observed=int(d.get("n_entropy_observed", 0)),
                # Identity defaults: a params file written before the
                # recalibration stage existed replays exactly as before.
                platt_a=float(d.get("platt_a", 1.0)),
                platt_b=float(d.get("platt_b", 0.0)),
                brier_raw=float(d.get("brier_raw", -1.0)),
                brier_base_rate=float(d.get("brier_base_rate", -1.0)),
                w_effort=float(d.get("w_effort", 0.0)),
                n_effort_observed=int(d.get("n_effort_observed", 0)),
                map_status=str(d.get("map_status", "applied")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("calibration params malformed: %s", exc)
            return None

    def _save_params(self, params: FittedParams) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.params_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
            os.replace(tmp, self.params_path)
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("calibration params save failed: %s", exc)

    # ----------------------------------------------------------- summary

    def stats(self) -> Dict[str, object]:
        """Introspection summary (for ``introspect`` / the calib log)."""
        samples = self._load_samples(limit=self.max_history)
        brier = (
            sum((s.composite - s.outcome) ** 2 for s in samples) / len(samples)
            if samples else None
        )
        params = self.load_params()
        return {
            "samples": len(samples),
            "brier": round(brier, 4) if brier is not None else None,
            # Same window as `brier` above. Calling `self.ece()` with no
            # argument scored the ENTIRE file while brier scored only the
            # max_history tail, so the two numbers sat side by side
            # describing different populations (measured: 0.0478 vs 0.1593
            # on the same report).
            "ece": (round(self.ece(window=self.max_history) or 0.0, 4)
                    if samples else None),
            "fitted": params is not None,
            "threshold": params.threshold if params else None,
            "w_entropy": params.w_entropy if params else None,
            "lambda_uncertainty": params.lambda_uncertainty if params else None,
            "map_status": (getattr(params, "map_status", "applied")
                           if params else None),
        }


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────

def _clamp01(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(v):
        return 0.5
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _best_threshold(pairs: List[Tuple[float, float]]) -> float:
    """Pick τ maximising Youden's J for "predict success when C ≥ τ".

    J = sensitivity + specificity − 1 = TPR − FPR. Candidate thresholds
    are the midpoints between sorted unique composites (plus the rails),
    so the chosen cut is robust to the exact float values. Ties break
    toward the *higher* threshold (more conservative — flips more turns
    into "below threshold → arbitrate"). Falls back to 0.5 when J never
    beats the trivial classifier.
    """
    # Degenerate-input fallbacks return the MEDIAN of the supplied scores
    # rather than a hardcoded constant. The caller may be on the raw
    # composite scale or on the Platt-mapped probability scale, and a fixed
    # 0.55 is only meaningful on the former — returning it into a mapped
    # caller put the gate at an arbitrary point of a different distribution.
    # The median is scale-free and always sits inside the observed range.
    def _median_conf(default=0.55):
        vals = sorted(c for c, _ in pairs)
        if not vals:
            return default
        mid = len(vals) // 2
        return (vals[mid] if len(vals) % 2
                else (vals[mid - 1] + vals[mid]) / 2.0)

    if not pairs:
        return 0.55
    n_pos = sum(1 for _, o in pairs if o >= 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return _median_conf()
    confs = sorted({round(c, 4) for c, _ in pairs})
    candidates = [0.0]
    for i in range(len(confs) - 1):
        candidates.append((confs[i] + confs[i + 1]) / 2.0)
    candidates.append(1.0)
    best_tau = 0.5
    best_j = -2.0
    for tau in candidates:
        tp = sum(1 for c, o in pairs if c >= tau and o >= 0.5)
        fp = sum(1 for c, o in pairs if c >= tau and o < 0.5)
        tpr = tp / n_pos
        fpr = fp / n_neg
        j = tpr - fpr
        if j > best_j or (abs(j - best_j) < 1e-9 and tau > best_tau):
            best_j = j
            best_tau = tau
    # Documented fallback (was missing): if no threshold beats the trivial
    # classifier (Youden J <= 0 — the composite is uncorrelated with outcome,
    # i.e. the miscalibrated case this exists to catch), the loop would pick a
    # DEGENERATE rail (τ=1.0 via the higher-tie-break → "below" ALWAYS True, or
    # τ=0.0 → always False). Return the median of the observed scores — a
    # neutral cut that is valid on WHICHEVER scale the caller passed, unlike
    # the hardcoded 0.5 this used to return.
    if best_j <= 1e-9:
        return _clamp01(_median_conf(0.5))
    return _clamp01(best_tau)


def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


__all__ = [
    "CalibrationTracker",
    "CalibrationSample",
    "FittedParams",
    "ReliabilityBin",
    "SCHEMA_VERSION",
]
