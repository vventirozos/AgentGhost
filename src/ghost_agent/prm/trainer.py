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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from ..distill.schema import Trajectory
from .features import extract_step_features
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

    def summary(self) -> str:
        if not self.fit_attempted:
            return f"no fit attempted: {self.bail_reason or 'unknown'}"
        if not self.fit_succeeded:
            return f"fit failed: {self.bail_reason or 'unknown'}"
        return (
            f"trained on {self.n_samples_total} samples "
            f"({self.n_samples_positive}+/{self.n_samples_negative}-) "
            f"from {self.n_trajectories_seen} trajectories; "
            f"saved to {self.saved_to or '<unsaved>'}"
        )


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
    # else the model's bias just memorises the prior.
    DEFAULT_MIN_CLASS_FRACTION: float = 0.05

    def __init__(
        self,
        *,
        spec: Optional[StepLabelSpec] = None,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        min_trajectories: int = DEFAULT_MIN_TRAJECTORIES,
        min_class_fraction: float = DEFAULT_MIN_CLASS_FRACTION,
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
