"""Adaptive smart-memory scoring threshold.

Tracks memory write acceptance/rejection rates and adjusts the threshold
dynamically. Power users who share lots of context need a higher bar;
new users need aggressive retention.

The "was_useful" signal: a memory was retrieved and included in a response
the user didn't correct (positive), or the user corrected a response that
used a stored memory (negative).

KNOWN LIMITATION — the learned threshold is currently inert in prod
--------------------------------------------------------------------
Read this before "fixing" a threshold that never seems to move:

1. ``was_useful`` is SELF-REFERENTIAL as wired in ``core.agent``: the
   signal fed back is "the score cleared the bar", not "the memory was
   later retrieved and helped". The loop therefore measures its own gate,
   not utility.
2. The update rule can effectively only ratchet DOWN. The upward
   candidate is ``min(useful_scores) * 0.9``, and a score is at most 1.0,
   so the candidate is at most 0.9 — while ``MAX_STEP_UP`` caps each rise
   at 0.02. The floor branch (``threshold * 0.98``) has no such cap.
3. The consumer takes ``max(cli_selectivity, learned)``. Under the live
   flag ``--smart-memory 0.9`` the effective bar is therefore >= 0.9 and
   the learned value — which cannot exceed 0.9 and decays toward
   ``FLOOR`` (0.3) whenever the window starves — is DEAD WEIGHT: it can
   never raise the bar and can never lower it either.

Consequence: with ``--smart-memory 0.9`` this class is bookkeeping only.
It becomes load-bearing again only for a selectivity below ~0.9. The
2026-07-22 durability pass deliberately did NOT change the numbers here:
altering the retention rule changes what the agent remembers, which is an
operator-visible behaviour change and needs to be decided, not smuggled
in with a persistence fix. See ``_recalculate`` for the mechanics.
"""

import json
import logging
import os
import threading
import time
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
        # Set when the state file is PRESENT but could not be read. While
        # set, _save() refuses to write — see _load().
        self._degraded = False
        self._load()

    def _quarantine_corrupt(self, why: str) -> None:
        """Corrupt state: preserve the raw file as a timestamped sidecar
        BEFORE the next _save() overwrites it, then start from the
        defaults. Same policy as journal.py / profile.py."""
        try:
            sidecar = self.file_path.with_suffix(f".corrupt-{int(time.time())}.json")
            os.replace(self.file_path, sidecar)
            logger.warning(
                "adaptive_threshold.json was corrupt (%s); preserved to %s and "
                "restarted from the initial threshold.", why, sidecar.name,
            )
        except Exception:
            logger.warning(
                "adaptive_threshold.json was corrupt (%s) and could not be "
                "preserved; restarting from the initial threshold.", why,
            )

    def _load(self):
        # A missing file is the normal cold start — nothing to preserve.
        try:
            content = self.file_path.read_text()
        except FileNotFoundError:
            return
        except UnicodeDecodeError:
            self._quarantine_corrupt("undecodable bytes")
            return
        except OSError as exc:
            # PRESENT but unreadable (EIO / EACCES / ENFILE …). The old
            # `except Exception: pass` silently started from the initial
            # threshold and the very next record() atomically OVERWROTE
            # the intact file — the whole learned window gone, at no log
            # level at all. Refuse to write instead: fail closed, loudly.
            self._degraded = True
            logger.error(
                "adaptive_threshold.json is present but unreadable (%s: %s). "
                "Running on the initial threshold and REFUSING to overwrite "
                "the on-disk state until it can be read.",
                type(exc).__name__, exc,
            )
            return
        if not content.strip():
            return
        try:
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError(
                    f"state is a {type(data).__name__}, expected object")
            threshold = float(data.get("threshold", self._initial))
            window = data.get("window", [])
            if not isinstance(window, list):
                raise ValueError("window is not a list")
        except Exception as exc:
            self._quarantine_corrupt(f"{type(exc).__name__}: {exc}")
            return
        self.threshold = threshold
        for obs in window:
            try:
                self.window.append(tuple(obs))
            except TypeError:
                continue

    def _save(self):
        if self._degraded:
            # See _load(): the on-disk state exists but could not be read,
            # so writing would destroy it.
            logger.warning(
                "Adaptive threshold save skipped: on-disk state is unreadable "
                "and must not be overwritten."
            )
            return
        try:
            data = {
                "threshold": self.threshold,
                "window": list(self.window),
            }
            tmp = self.file_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, self.file_path)
        except Exception as e:
            logger.warning(f"Adaptive threshold save failed: {e}")

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

        NOTE (see the module docstring): ``min_useful * 0.9 <= 0.9`` for any
        score in [0, 1], so this rule cannot push the threshold above 0.9,
        while the no-useful-observations branch decays it toward FLOOR
        without limit. Combined with the consumer's
        ``max(cli_selectivity, learned)`` that makes the learned value inert
        at the live ``--smart-memory 0.9``. Changing that is a retention
        decision, not a bug fix, so the arithmetic is left as-is here.
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
