"""Training pipeline: trajectories → step samples → fit → save.

Entry point for both the manual CLI workflow (a script that points at
``$GHOST_HOME/trajectories`` and produces a checkpoint) and the idle
biological retrain phase (phase 2.7 in ``core/agent.py``).

Design notes:

  * Pure CPU pass — no LLM call, no embedder. Mirrors ``skills_auto/``
    so it's safe to run inside the watchdog without busying the
    user-facing model.
  * Data source is anything that yields ``Trajectory`` objects. In
    production that's ``TrajectoryCollector.iter_trajectories()``;
    tests pass synthetic lists. The trainer doesn't import the
    collector to keep the dependency one-way.
  * The trainer is RESILIENT: a minimum-sample floor and a class-balance
    floor cause it to BAIL out (returning an unfit model + a report
    explaining why) rather than fitting on degenerate data and shipping
    a confidently-wrong scorer. Retraining a fresh model is cheap; a
    bad model poisons every subsequent plan score until someone
    notices.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from ..distill.schema import Trajectory
from .features import extract_step_features

# Features that VARY across training samples (drawn mid-turn) but are ALWAYS 0
# at the live scoring site — the PRM scores at TURN START, where no step has run
# yet (steps_so_far=0, failures_so_far=0, no tool used/failed this turn). A fit
# that leans on them reports a train accuracy the deployed model can't reproduce.
# We surface this skew (see PRMTrainer.run); a full fix is a training-signal
# redesign (score at turn start, or drop these columns). Names must exist in
# PRM_FEATURE_NAMES (guarded by a unit test).
SERVE_TURN_START_INERT_FEATURES: Tuple[str, ...] = (
    "plan_steps_so_far_log1p",
    "plan_failures_so_far_log1p",
    "plan_has_any_failure",
    "tool_already_used_this_turn",
    "tool_failed_this_turn",
)
from .labels import (
    StepLabelSpec,
    StepSample,
    class_balance,
    iter_step_samples,
)
from .model import StepValueModel, PRMTrainingReport


logger = logging.getLogger("GhostAgent")


def samples_to_xy(
    trajectories: Iterable[Trajectory],
    *,
    spec: Optional[StepLabelSpec] = None,
    use_continuous: bool = True,
) -> Tuple[List, List[float]]:
    """Build ``(X, y)`` feature/label lists from trajectories.

    Shared by the batch trainer and the online-update path
    (``PRMScorer.online_update``) so both featurise identically. ``X`` is
    a list of ``FeatureVector``; ``y`` is the discount-weighted continuous
    value (or the 0/1 binary when ``use_continuous`` is False)."""
    spec = spec or StepLabelSpec()
    samples = list(iter_step_samples(list(trajectories), spec))
    X = [extract_step_features(s.state, s.action) for s in samples]
    y = [s.value if use_continuous else s.binary for s in samples]
    return X, y


@dataclass
class TrainerReport:
    """Summary of what the trainer did. Returned from ``run`` and also
    exposed on the saved JSON via ``StepValueModel.report_``."""

    n_trajectories_seen: int = 0
    n_samples_total: int = 0
    n_samples_positive: int = 0
    n_samples_negative: int = 0
    fit_attempted: bool = False
    fit_succeeded: bool = False
    bail_reason: str = ""
    saved_to: str = ""
    model_report: Optional[PRMTrainingReport] = None
    # Non-empty when the fit leaned on features that are inert at serve time
    # (see SERVE_TURN_START_INERT_FEATURES) — a caveat on train accuracy.
    feature_skew_warning: str = ""

    def summary(self) -> str:
        if not self.fit_attempted:
            return f"no fit attempted: {self.bail_reason or 'unknown'}"
        if not self.fit_succeeded:
            return f"fit failed: {self.bail_reason or 'unknown'}"
        base = (
            f"trained on {self.n_samples_total} samples "
            f"({self.n_samples_positive}+/{self.n_samples_negative}-) "
            f"from {self.n_trajectories_seen} trajectories; "
            f"saved to {self.saved_to or '<unsaved>'}"
        )
        if self.feature_skew_warning:
            base += f"  ⚠ {self.feature_skew_warning}"
        return base


class PRMTrainer:
    """Orchestrates the trajectory → fit pipeline.

    Usage:
        trainer = PRMTrainer()
        report = trainer.run(
            trajectories=collector.iter_trajectories(),
            save_path=Path("/path/to/prm.json"),
        )
        # report.model_report has weights / loss / accuracy
    """

    # Below this many samples, training is skipped — the model would
    # overfit hard and provide misleading scores on real plans.
    DEFAULT_MIN_SAMPLES: int = 20

    # Below this many distinct trajectories, training is also skipped:
    # samples drawn from a single trajectory are nearly identical and
    # don't span the input distribution.
    DEFAULT_MIN_TRAJECTORIES: int = 5

    # Each class must have at least this fraction of total samples,
    # else the model's bias just memorises the prior. Applies to the BINARY
    # label mode only (see the continuous-mode floor below).
    DEFAULT_MIN_CLASS_FRACTION: float = 0.05

    # Continuous-label mode fits soft targets, so the viability floor is on
    # label VARIANCE, not binary class balance: an all-but-constant target has
    # nothing to learn but the mean. std below this ⇒ bail.
    DEFAULT_MIN_LABEL_STD: float = 0.02

    def __init__(
        self,
        *,
        spec: Optional[StepLabelSpec] = None,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        min_trajectories: int = DEFAULT_MIN_TRAJECTORIES,
        min_class_fraction: float = DEFAULT_MIN_CLASS_FRACTION,
        min_label_std: float = DEFAULT_MIN_LABEL_STD,
        learning_rate: float = 0.1,
        l2: float = 1e-3,
        epochs: int = 300,
        random_state: int = 0,
        use_continuous_labels: bool = True,
    ):
        self.spec = spec or StepLabelSpec()
        self.min_samples = int(min_samples)
        self.min_trajectories = int(min_trajectories)
        self.min_class_fraction = float(min_class_fraction)
        self.min_label_std = float(min_label_std)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.epochs = int(epochs)
        self.random_state = int(random_state)
        # When True, fit on the discount-weighted continuous values
        # (preserves how-close-to-success the step was). When False,
        # threshold to 0/1 via the spec's binary_threshold first.
        self.use_continuous_labels = bool(use_continuous_labels)
        # The trained model from the most recent successful run.
        # Exposed so the biological retrain phase can hot-swap it
        # into the live ``PRMScorer`` without paying a disk round-trip.
        # ``None`` until ``run`` succeeds at least once.
        self.model: Optional[StepValueModel] = None

    def run(
        self,
        trajectories: Iterable[Trajectory],
        save_path: Optional[Path | str] = None,
    ) -> TrainerReport:
        """Execute the pipeline. Returns a ``TrainerReport`` describing
        what happened — either a successful fit or the reason the
        trainer bailed."""
        traj_list = list(trajectories)
        report = TrainerReport(n_trajectories_seen=len(traj_list))

        if len(traj_list) < self.min_trajectories:
            report.bail_reason = (
                f"need ≥{self.min_trajectories} trajectories, "
                f"have {len(traj_list)}"
            )
            return report

        samples: List[StepSample] = list(iter_step_samples(traj_list, self.spec))
        balance = class_balance(samples)
        report.n_samples_total = balance["total"]
        report.n_samples_positive = balance["positive"]
        report.n_samples_negative = balance["negative"]

        if balance["total"] < self.min_samples:
            report.bail_reason = (
                f"need ≥{self.min_samples} step samples, "
                f"have {balance['total']}"
            )
            return report

        pos_frac = (
            balance["positive"] / balance["total"]
            if balance["total"] else 0.0
        )
        neg_frac = (
            balance["negative"] / balance["total"]
            if balance["total"] else 0.0
        )
        # Training-viability gate — matched to what the fit ACTUALLY consumes.
        # A binary classifier needs BOTH classes present (a one-sided gradient
        # sends the bias to ±∞); a continuous regressor only needs the soft
        # targets to VARY (all-constant y ⇒ nothing to learn but the mean).
        # Gating the DEFAULT continuous fit on the BINARY class balance wrongly
        # bailed on a perfectly trainable set whose discount-weighted values all
        # sit on one side of the 0.5 binary threshold (e.g. 0.1‥0.48 reads as
        # 100% "negative" in the binary view, yet carries a clean gradient).
        if self.use_continuous_labels:
            # Both regimes must be REPRESENTED (≥1 success-side and ≥1
            # failure-side sample) — a value model that never saw a failure
            # (or never a success) can't discriminate, so an all-PASSED or
            # all-FAILED corpus still bails. But we deliberately do NOT
            # re-impose the binary FRACTION floor: a few high-value anchors
            # among many lows is a legitimate soft-target gradient, and gating
            # that on binary balance is exactly the bug this fix removes.
            if balance["positive"] < 1 or balance["negative"] < 1:
                report.bail_reason = (
                    f"class imbalance (single-regime): "
                    f"{balance['positive']} success-side / "
                    f"{balance['negative']} failure-side sample(s), need ≥1 of each"
                )
                return report
            # Even with both regimes present, near-constant soft targets carry
            # no gradient to fit (a safety net for tightly-clustered values).
            _vals = [float(s.value) for s in samples]
            _std = statistics.pstdev(_vals) if len(_vals) > 1 else 0.0
            if _std < self.min_label_std:
                report.bail_reason = (
                    f"continuous labels near-constant: std={_std:.4f} "
                    f"(need ≥{self.min_label_std}); "
                    f"range=[{min(_vals):.3f}, {max(_vals):.3f}]"
                )
                return report
        else:
            if min(pos_frac, neg_frac) < self.min_class_fraction:
                report.bail_reason = (
                    f"class imbalance: positive={pos_frac:.2%}, "
                    f"negative={neg_frac:.2%} (need ≥{self.min_class_fraction:.0%} of each)"
                )
                return report

        # Build training arrays.
        X = [extract_step_features(s.state, s.action) for s in samples]
        if self.use_continuous_labels:
            y = [s.value for s in samples]
        else:
            y = [s.binary for s in samples]

        # Train↔serve skew check: flag serve-inert features that nonetheless
        # vary across the training samples, so a caller doesn't read the model's
        # train accuracy as deployed discrimination (the live scorer sees these
        # as 0). Cheap, best-effort — never blocks a fit.
        try:
            skewed = [
                name for name in SERVE_TURN_START_INERT_FEATURES
                if statistics.pstdev([fv.by_name.get(name, 0.0) for fv in X]) > 1e-6
            ] if len(X) > 1 else []
            if skewed:
                report.feature_skew_warning = (
                    "serve-inert features vary in training but read 0 at "
                    "turn-start scoring: " + ", ".join(sorted(skewed))
                )
                logger.warning("PRM %s", report.feature_skew_warning)
        except Exception as exc:
            logger.debug("PRM feature-skew check skipped: %s", exc)

        report.fit_attempted = True
        try:
            model = StepValueModel(
                learning_rate=self.learning_rate,
                l2=self.l2,
                epochs=self.epochs,
                random_state=self.random_state,
            )
            model.fit(X, y)
        except Exception as exc:
            report.bail_reason = f"fit raised: {type(exc).__name__}: {exc}"
            logger.warning("PRM fit failed: %s", exc)
            return report

        report.fit_succeeded = True
        report.model_report = model.report_
        # Expose the freshly-trained model so callers (notably the
        # biological retrain phase) can hot-swap it into the live
        # PRMScorer without a disk round-trip.
        self.model = model

        if save_path is not None:
            try:
                p = Path(save_path)
                model.save(p)
                report.saved_to = str(p)
            except Exception as exc:
                report.bail_reason = f"save failed: {exc}"
                logger.warning("PRM save failed: %s", exc)

        return report
