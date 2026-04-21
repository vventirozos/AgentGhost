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
            self.window.append((score, was_useful))
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

        useful_scores = [s for s, u in self.window if u]
        useless_scores = [s for s, u in self.window if not u]

        if useful_scores:
            # Target: accept everything that was useful, reject the rest
            min_useful = min(useful_scores)
            candidate = min_useful * 0.9

            # If we have useless data, ensure we're not below their median
            if useless_scores:
                useless_sorted = sorted(useless_scores)
                median_useless = useless_sorted[len(useless_sorted) // 2]
                # Don't drop below median of useless scores
                candidate = max(candidate, median_useless)

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
            useful = sum(1 for _, u in self.window if u)
            total = len(self.window)
            return {
                "threshold": self.threshold,
                "observations": total,
                "useful_count": useful,
                "useful_rate": useful / total if total > 0 else 0.0,
            }
