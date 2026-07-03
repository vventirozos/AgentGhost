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
    ) -> None:
        """Append one (confidence, outcome) pair. Never raises."""
        try:
            sample = CalibrationSample(
                composite=_clamp01(composite),
                entropy_component=_clamp01(entropy_component),
                competence_component=_clamp01(competence_component),
                uncertainty_pressure=_clamp01(uncertainty_pressure),
                outcome=1.0 if float(outcome) >= 0.5 else 0.0,
                domain=str(domain or ""),
                ts=_utcnow_iso(),
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
                        outcome=1.0 if float(d.get("outcome", 0.0)) >= 0.5 else 0.0,
                        domain=str(d.get("domain", "")),
                        ts=str(d.get("ts", "")),
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
        if n_pos == 0 or n_neg == 0:
            logger.debug(
                "calibration fit bail: single outcome class (pos=%d neg=%d)",
                n_pos, n_neg,
            )
            return None

        # Grid over entropy weight (→ competence = 1−w_e) and the
        # uncertainty penalty λ. Composite is recomputed per candidate;
        # we minimise Brier of that recomputed composite vs outcome.
        best: Optional[Tuple[float, float, float]] = None  # (brier, w_e, lam)
        for we_i in range(0, 11):
            w_e = we_i / 10.0
            w_c = 1.0 - w_e
            for lam_i in range(0, 6):
                lam = lam_i / 10.0
                sq = 0.0
                for s in samples:
                    c = w_e * s.entropy_component + w_c * s.competence_component
                    c = c * (1.0 - lam * s.uncertainty_pressure)
                    c = _clamp01(c)
                    sq += (c - s.outcome) ** 2
                brier = sq / len(samples)
                if best is None or brier < best[0]:
                    best = (brier, w_e, lam)

        assert best is not None
        brier, w_e, lam = best
        w_c = 1.0 - w_e

        # With the winning weights/λ, recompute composites and pick the
        # threshold that best separates pass from fail (Youden's J on the
        # "predict success when C ≥ τ" decision). This is what
        # ``below_threshold`` keys off, so it should be the empirically
        # best cut, not a hand-picked 0.55.
        composites = []
        for s in samples:
            c = w_e * s.entropy_component + w_c * s.competence_component
            c = _clamp01(c * (1.0 - lam * s.uncertainty_pressure))
            composites.append((c, s.outcome))
        threshold = _best_threshold(composites)

        params = FittedParams(
            w_entropy=round(w_e, 4),
            w_competence=round(w_c, 4),
            threshold=round(threshold, 4),
            lambda_uncertainty=round(lam, 4),
            brier=round(brier, 6),
            n_samples=len(samples),
            fitted_at=_utcnow_iso(),
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
            "ece": round(self.ece() or 0.0, 4) if samples else None,
            "fitted": params is not None,
            "threshold": params.threshold if params else None,
            "w_entropy": params.w_entropy if params else None,
            "lambda_uncertainty": params.lambda_uncertainty if params else None,
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
    if not pairs:
        return 0.55
    n_pos = sum(1 for _, o in pairs if o >= 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.55
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
    # τ=0.0 → always False). Return the neutral 0.5 instead.
    if best_j <= 1e-9:
        return 0.5
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
