"""Adaptive smart-memory scoring threshold.

Tracks memory write acceptance/rejection rates and adjusts the threshold
dynamically. Power users who share lots of context need a higher bar;
new users need aggressive retention.

The "was_useful" signal: a memory was retrieved and included in a response
the user didn't correct (positive), or the user corrected a response that
used a stored memory (negative).
"""

import json
import logging
import os
import threading
from collections import deque
from pathlib import Path

logger = logging.getLogger("GhostAgent")


class AdaptiveThreshold:
    """Self-tuning memory acceptance threshold.

    Maintains a sliding window of (score, was_useful) observations and
    adjusts the threshold to maximise the precision/recall trade-off.
    Persists state to JSON so the threshold survives restarts.
    """

    WINDOW_SIZE = 100
    MIN_OBSERVATIONS = 20
    # Hard bounds — never go below 0.3 (spam) or above 0.95 (nothing saved)
    FLOOR = 0.3
    CEILING = 0.95
    # Max the threshold may RISE per update. Bursts once snapped the gate
    # near CEILING in a single recalculation; everything after scored
    # below the new bar, was recorded as high-score useless, and sustained
    # the clamp. Downward moves stay instant — over-rejection is the
    # harmful direction.
    MAX_STEP_UP = 0.02
    # Consecutive useless observations after which the gate is considered
    # starving: decay toward exploration even while stale useful entries
    # linger in the window (previously decay needed the WHOLE window to be
    # useless, i.e. ~WINDOW_SIZE records to flush first).
    STARVATION_RUN = 20

    def __init__(self, memory_dir: Path, initial: float = 0.7):
        self.file_path = memory_dir / "adaptive_threshold.json"
        self._lock = threading.RLock()
        self._initial = initial
        self.threshold = initial
        self.window: deque = deque(maxlen=self.WINDOW_SIZE)
        self._load()

    def _load(self):
        if not self.file_path.exists():
            return
        try:
            data = json.loads(self.file_path.read_text())
            self.threshold = float(data.get("threshold", self._initial))
            for obs in data.get("window", []):
                self.window.append(tuple(obs))
        except Exception:
            pass

    def _save(self):
        try:
            data = {
                "threshold": self.threshold,
                "window": list(self.window),
            }
            tmp = self.file_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, self.file_path)
        except Exception as e:
            logger.debug(f"Adaptive threshold save failed: {e}")

    def record(self, score: float, was_useful: bool):
        """Record an observation: a memory with this score was (not) useful."""
        with self._lock:
            # Third element: the score CLEARED the bar at record time. A
            # useless observation that cleared the bar was rejected by the
            # downstream content gates — raising the score bar cannot help
            # reject its kind, so it must not feed the median-useless
            # clamp. (That feedback was the 2026-07 ratchet: a burst of
            # high-scoring content-rejects pushed median_useless ~0.9,
            # the gate snapped up, and everything after was recorded as
            # high-score useless, sustaining the lock near CEILING.)
            cleared_bar = score >= self.threshold
            self.window.append((score, was_useful, cleared_bar))
            self._recalculate()
            self._save()

    def _recalculate(self):
        """Adjust threshold based on accumulated observations.

        Strategy: set threshold to 90% of the minimum score among useful
        memories. This keeps the bar slightly below the weakest useful signal
        to avoid filtering out borderline-but-valuable facts.
        """
        if len(self.window) < self.MIN_OBSERVATIONS:
            return

        useful_scores = [obs[0] for obs in self.window if obs[1]]

        # Starvation guard: nothing accepted for STARVATION_RUN straight
        # records means the bar is above everything the scorer emits; the
        # stale useful entries earlier in the window would otherwise keep
        # the clamp branch below in charge until the whole window flushed.
        recent = list(self.window)[-self.STARVATION_RUN:]
        starving = (
            len(recent) >= self.STARVATION_RUN
            and not any(obs[1] for obs in recent)
        )

        if useful_scores and not starving:
            # Target: accept everything that was useful, reject the rest
            min_useful = min(useful_scores)
            candidate = min_useful * 0.9

            # If we have useless data, ensure we're not below their median.
            # Rejections that cleared the then-current bar are excluded
            # (see record()); legacy 2-tuples predate the flag and keep
            # their old (counted) semantics.
            useless_scores = [
                obs[0] for obs in self.window
                if not obs[1] and not (len(obs) > 2 and obs[2])
            ]
            if useless_scores:
                useless_sorted = sorted(useless_scores)
                median_useless = useless_sorted[len(useless_sorted) // 2]
                # Don't drop below median of useless scores
                candidate = max(candidate, median_useless)

            # Rate-limit upward moves (see MAX_STEP_UP).
            if candidate > self.threshold:
                candidate = min(candidate, self.threshold + self.MAX_STEP_UP)

            self.threshold = max(self.FLOOR, min(self.CEILING, candidate))
        else:
            # No useful observations yet — slowly lower the bar to explore
            self.threshold = max(self.FLOOR, self.threshold * 0.98)

    def get_threshold(self) -> float:
        """Return the current adaptive threshold."""
        with self._lock:
            return self.threshold

    def get_stats(self) -> dict:
        """Return diagnostic stats about the threshold state."""
        with self._lock:
            useful = sum(1 for obs in self.window if obs[1])
            total = len(self.window)
            return {
                "threshold": self.threshold,
                "observations": total,
                "useful_count": useful,
                "useful_rate": useful / total if total > 0 else 0.0,
            }
