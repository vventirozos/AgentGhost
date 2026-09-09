"""Fail-closed stores must be able to come BACK (§4FP, 2026-09-09).

FOUR memory stores refuse to write after a read of their own file fails
with an OSError — the file is PRESENT but unreadable (EIO, EACCES,
ENFILE/EMFILE …), so the on-disk state is probably intact and overwriting
it with whatever is in memory would destroy it. That guard is right, and
it was measured: without it `adaptive_threshold`'s very next `record()`
atomically overwrote the whole learned window.

The half that was missing is the way back. `contradiction_log` and
`profile` re-read on every operation, so a successful read clears the flag
by itself — documented there as "cleared automatically as soon as a read
succeeds". `adaptive_threshold` and `competence` read only in `__init__`,
so their flag was cleared only by a RESTART: one transient file-descriptor
exhaustion stopped that store learning for the life of the process, which
for this agent is days, announced by a single log line.

This module is the one home for the policy the four now share:

  * **retry the read that armed it**, not more often than
    ``RETRY_EVERY_S`` (a store that saves on every observation must not
    stat a sick disk thousands of times a minute);
  * **recover only on a real read** — the flag clears because the file was
    read, never because time passed;
  * **lose nothing**: whatever the store accumulated while blind is merged
    onto the history it could not see, by a hook the store owns, because
    only the store knows what merging its own state means.

⚠ `tests/test_failclosed_recovery.py` enumerates the CLASS from the AST: a
store that arms a fail-closed flag and cannot clear it outside `__init__`
fails the suite. Fixing this shape site-by-site is what let two of the four
drift apart in the first place.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger("GhostAgent")

#: Seconds between read retries while degraded. Small enough that a blip
#: costs a handful of observations, large enough that a genuinely sick
#: disk is not hammered by a store that saves on every record.
RETRY_EVERY_S = 30.0


class FailClosedStore:
    """Mixin for a store that blocks writes after an unreadable read.

    The store owns two things: it calls `_fc_arm` where it used to set
    `self._degraded = True`, and it implements `_fc_reload_from_disk`.
    Everything else — the retry cadence, the "only a read clears it" rule,
    the logging — lives here so the four stores cannot drift apart again.
    """

    #: Set by `_fc_arm`; read by the store's own `_save`.
    _degraded = False
    _fc_last_retry = 0.0
    #: True while a retry is in flight. A retry clears `_degraded` so the
    #: reload can re-arm it and thereby REPORT failure — without this the
    #: re-arm would log a fresh ERROR every 30 s for as long as the disk is
    #: sick, which is how one fault becomes a log flood.
    _fc_reloading = False

    def _fc_arm(self, why: str) -> None:
        """Block writes because the file is present but unreadable.

        Logged once per arming, not once per attempt: a store that saves on
        every observation would otherwise turn one sick disk into thousands
        of identical ERROR lines.
        """
        if not self._degraded and not self._fc_reloading:
            logger.error(
                "%s: %s. Serving in-memory state and REFUSING to overwrite "
                "the file until a read succeeds (retrying every %.0fs).",
                type(self).__name__, why, RETRY_EVERY_S,
            )
        self._degraded = True
        # Retry on the very next save: a blip that lasts one write should
        # cost one write, so the clock starts EXPIRED rather than now.
        #
        # ⚠ Only on a FRESH arming. A failed retry re-arms through this same
        # method, and resetting the clock there would expire it every time —
        # the rate limit would be dead and a sick disk would be re-read on
        # every observation (measured: 21 reads for 20 records).
        if not self._fc_reloading:
            self._fc_last_retry = 0.0

    def _fc_reload_from_disk(self) -> bool:  # pragma: no cover - overridden
        """Re-read the file and merge anything accumulated while blind.

        Returns True when the read succeeded (and the store is whole
        again), False when the file is still unreadable. Implemented by
        each store: only it knows what merging its own state means.
        """
        raise NotImplementedError

    def _fc_ready_to_write(self) -> bool:
        """May the caller write? True when healthy, or when a retried read
        has just succeeded. False leaves the file untouched."""
        if not self._degraded:
            return True
        now = time.monotonic()
        if (now - float(self._fc_last_retry or 0.0)) < RETRY_EVERY_S:
            return False
        self._fc_last_retry = now
        self._fc_reloading = True
        try:
            recovered = bool(self._fc_reload_from_disk())
        except Exception as exc:  # noqa: BLE001 — recovery must never raise
            logger.debug("%s: fail-closed recovery attempt raised: %s",
                         type(self).__name__, exc)
            return False
        finally:
            self._fc_reloading = False
        if recovered:
            self._degraded = False
            logger.info(
                "%s: the file is readable again — history reloaded, the "
                "observations recorded while it was unreadable were merged "
                "in, and writes are ENABLED.", type(self).__name__,
            )
        return recovered
