"""Router classifier trainer.

Mirrors ``prm/trainer.py``: turns the trajectory log into labeled training
data and fits the ``ComplexityClassifier`` so the router stops shipping
untrained (an untrained classifier escalates EVERY request to the full swarm,
which is exactly the cost the router exists to avoid).

Bail floors are essential here because ``ComplexityClassifier.fit`` *raises*
(rather than bailing gracefully) on too-few-samples or single-class data, and
``label_trajectories`` skews toward "hard" (every failed trajectory labels
hard). So we gate on a minimum labeled count AND require both classes present
before calling ``fit``.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

logger = logging.getLogger("GhostAgent")


# Startup bootstrap-train gate (mirrors the idle-retrain floors, but
# higher: a one-time boot train should only fire when there's enough
# real signal to beat the safe escalate-all pass-through). Below this many
# LABELED, multi-class samples we stay pass-through and wait for an idle
# retrain to produce a checkpoint later.
BOOTSTRAP_MIN_SAMPLES = 50
# Cap the raw trajectory stream read at boot so a giant log can't stall
# startup. Labeling drops the ambiguous middle, so we read generously
# more than BOOTSTRAP_MIN_SAMPLES before the cap bites.
BOOTSTRAP_MAX_TRAJECTORIES = 20000


@dataclass
class RouterTrainerReport:
    fit_succeeded: bool = False
    bail_reason: str = ""
    n_samples: int = 0
    easy: int = 0
    hard: int = 0

    def summary(self) -> str:
        if not self.fit_succeeded:
            return f"bailed: {self.bail_reason}"
        return f"fit on {self.n_samples} samples (easy={self.easy}, hard={self.hard})"


class RouterTrainer:
    """Label trajectories → extract features → fit ComplexityClassifier.

    On success ``self.classifier`` holds the fitted model (the hot-swap
    handle) and ``run()`` returns a report with ``fit_succeeded=True``.
    """

    def __init__(self, min_trajectories: int = 20, min_per_class: int = 1):
        # Require ≥ this many LABELED (non-ambiguous) trajectories before
        # training — below it the classifier would overfit noise.
        self.min_trajectories = int(min_trajectories)
        self.min_per_class = int(min_per_class)
        self.classifier = None

    def run(self, trajectories: Iterable, save_path: Optional[Path] = None) -> RouterTrainerReport:
        from .features import extract_features
        from .labels import label_trajectories, class_balance
        from .model import ComplexityClassifier

        report = RouterTrainerReport()
        try:
            pairs = label_trajectories(list(trajectories))
        except Exception as e:
            report.bail_reason = f"labeling failed: {e}"
            return report

        if len(pairs) < self.min_trajectories:
            report.bail_reason = (
                f"too few labeled trajectories ({len(pairs)} < {self.min_trajectories})"
            )
            return report

        y = [label for _, label in pairs]
        bal = class_balance(y)
        report.n_samples = int(bal.get("total", len(y)))
        report.easy = int(bal.get("easy", 0))
        report.hard = int(bal.get("hard", 0))
        if report.easy < self.min_per_class or report.hard < self.min_per_class:
            report.bail_reason = (
                f"single-class data (easy={report.easy}, hard={report.hard})"
            )
            return report

        try:
            X = [extract_features((getattr(t, "user_request", "") or "")) for t, _ in pairs]
            clf = ComplexityClassifier()
            clf.fit(X, y)
        except Exception as e:
            report.bail_reason = f"fit failed: {e}"
            return report

        if save_path is not None:
            try:
                clf.save(save_path)
            except Exception as e:
                logger.warning("router classifier save failed: %s", e)

        self.classifier = clf
        report.fit_succeeded = True
        return report


def bootstrap_router(
    trajectories: Iterable,
    *,
    save_path: Optional[Path] = None,
    min_samples: int = BOOTSTRAP_MIN_SAMPLES,
    max_trajectories: int = BOOTSTRAP_MAX_TRAJECTORIES,
) -> Tuple[Optional["ComplexityClassifier"], RouterTrainerReport]:  # noqa: F821
    """One-time startup bootstrap-train from the trajectory log.

    Why this exists: the router ships UNTRAINED, and a trained model is
    otherwise only ever produced by an IDLE retrain (needs a long-lived idle
    process). A busy server or a benchmark never idles, so the dispatcher
    stays escalate-all forever. This trains ONCE at boot from the existing
    trajectory log so the router starts routing immediately when enough
    labeled, multi-class data already exists.

    Reuses RouterTrainer (same labeling + feature extraction + fit as the
    idle path); only difference is a higher min-sample floor and a cap on
    how much of the log we read, so a huge log can't stall startup.

    NEVER raises — any failure (malformed log, fit divergence, IO error) is
    logged and returns ``(None, report)`` so the caller falls back to the
    safe escalate-all pass-through. Returns ``(classifier, report)``; the
    classifier is None whenever training bailed.
    """
    report = RouterTrainerReport()
    try:
        # Cap the raw stream so a giant log doesn't stall boot. islice is
        # lazy, so we never materialise the whole log.
        capped = itertools.islice(trajectories, max(0, int(max_trajectories)))
        trainer = RouterTrainer(min_trajectories=int(min_samples), min_per_class=1)
        report = trainer.run(trajectories=capped, save_path=save_path)
        if report.fit_succeeded and trainer.classifier is not None:
            return trainer.classifier, report
        return None, report
    except Exception as e:  # never crash boot
        logger.warning("router bootstrap-train failed (staying pass-through): %s", e)
        report.fit_succeeded = False
        report.bail_reason = f"bootstrap exception: {e}"
        return None, report
