"""Active-turn registry + real turn cancellation (2026-07-11).

Turns are globally serialized (``GhostAgent.agent_semaphore`` is a
``Semaphore(1)`` — see #22), so ONE wedged turn blocks the web UI, Slack,
AND the autonomous project-advance ticks simultaneously. Until this module
there was no way out: the interface's ``/api/chat/cancel`` only cancels the
*proxy's* buffered stream, so the agent kept working and kept the lock.

This registry gives every in-flight turn an identity (its ``req_id``), a
handle on its asyncio task, and a cooperative cancel flag, so a caller can:

* **see** what is running and for how long (``/api/turns``);
* **cancel cooperatively** (the default) — the turn loop notices at its next
  boundary, stops cleanly, and returns whatever partial work it has;
* **cancel hard** — the asyncio task is cancelled outright, which unwinds
  the ``async with agent_semaphore`` and RELEASES the lock even if the turn
  is wedged inside a long upstream call that never reaches a boundary.

A turn registers itself BEFORE acquiring the semaphore, so a request still
queued behind a runaway turn can be cancelled too (there is nothing to
preserve for a turn that never started, so those are always hard-cancelled).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GhostAgent")

# Reason strings surfaced to the operator / stored on the turn.
REASON_USER = "cancelled by user"


class TurnCancelled(Exception):
    """Raised at a turn-loop boundary when a cooperative cancel was
    requested. Caught by ``handle_chat``, which returns the partial work
    instead of an error — so cancelling is a clean stop, not a crash."""

    def __init__(self, req_id: str = "", reason: str = REASON_USER):
        self.req_id = req_id
        self.reason = reason
        super().__init__(f"turn {req_id} cancelled: {reason}")


@dataclass
class ActiveTurn:
    req_id: str
    started_at: float = field(default_factory=time.time)
    preview: str = ""
    session_id: str = ""
    running: bool = False          # False while queued on the semaphore
    task: Optional[asyncio.Task] = None
    _cancel: bool = False
    reason: str = ""

    @property
    def age_s(self) -> float:
        return time.time() - self.started_at

    @property
    def cancelled(self) -> bool:
        return self._cancel

    def to_dict(self) -> dict:
        return {
            "request_id": self.req_id,
            "age_s": round(self.age_s, 1),
            "running": self.running,
            "queued": not self.running,
            "cancelled": self._cancel,
            "preview": self.preview,
            "session_id": self.session_id,
        }


class TurnRegistry:
    """Thread-safe registry of in-flight turns. Never raises into a caller."""

    def __init__(self):
        self._turns: Dict[str, ActiveTurn] = {}
        self._lock = threading.Lock()

    def register(self, req_id: str, *, preview: str = "",
                 session_id: str = "") -> ActiveTurn:
        base = str(req_id)
        turn = ActiveTurn(
            req_id=base,
            preview=" ".join(str(preview or "").split())[:120],
            session_id=str(session_id or ""),
        )
        try:
            turn.task = asyncio.current_task()
        except RuntimeError:
            turn.task = None
        with self._lock:
            key = base
            if key in self._turns:
                # Collision. req_id can be a CLIENT-supplied X-Request-ID
                # (routes.py reads the header), so two overlapping requests
                # can carry the same id. Overwriting the live entry made
                # /api/turns hide the first turn and made cancel/unregister
                # hit the WRONG one (cancelling B killed running A). Never
                # clobber a live entry — uniquify the new turn's key instead.
                n = 2
                while f"{base}#{n}" in self._turns:
                    n += 1
                key = f"{base}#{n}"
                turn.req_id = key
            self._turns[key] = turn
        return turn

    def unregister(self, req_id: str, turn: Optional["ActiveTurn"] = None) -> None:
        """Remove a turn. When ``turn`` is given the pop is IDENTITY-CHECKED —
        a turn's ``finally`` must not evict a different turn that happens to
        share the key (see register's collision handling)."""
        with self._lock:
            cur = self._turns.get(str(req_id))
            if cur is None:
                return
            if turn is not None and cur is not turn:
                return
            self._turns.pop(str(req_id), None)

    def mark_running(self, req_id: str) -> None:
        """Called once the turn has actually ACQUIRED the semaphore — before
        that it is merely queued, and a cancel can just kill it outright."""
        with self._lock:
            turn = self._turns.get(str(req_id))
            if turn is not None:
                turn.running = True

    def get(self, req_id: str) -> Optional[ActiveTurn]:
        with self._lock:
            return self._turns.get(str(req_id))

    def list(self) -> List[ActiveTurn]:
        with self._lock:
            return sorted(self._turns.values(), key=lambda t: t.started_at)

    def current(self) -> Optional[ActiveTurn]:
        """The turn holding the semaphore (there is at most one)."""
        with self._lock:
            running = [t for t in self._turns.values() if t.running]
        return min(running, key=lambda t: t.started_at) if running else None

    def is_cancelled(self, req_id: str) -> bool:
        turn = self.get(req_id)
        return bool(turn is not None and turn._cancel)

    def cancel(self, req_id: Optional[str] = None, *, hard: bool = False,
               reason: str = REASON_USER) -> dict:
        """Cancel a turn (or the currently-running one when ``req_id`` is
        omitted).

        Cooperative by default: the flag is set and the turn stops at its
        next boundary, keeping partial work. A QUEUED turn is always
        hard-cancelled (nothing to preserve, and it would otherwise sit on
        the semaphore queue until the runaway turn ahead of it finishes).
        ``hard=True`` cancels the asyncio task outright — the guaranteed
        lock release for a turn wedged inside a long upstream call.
        """
        turn = self.get(req_id) if req_id else self.current()
        if turn is None:
            return {"cancelled": False,
                    "error": (f"no active turn {req_id!r}" if req_id
                              else "no turn is currently running")}
        turn._cancel = True
        turn.reason = reason
        # A queued turn has no partial work and no boundary to reach —
        # killing it is strictly better than letting it start.
        do_hard = bool(hard) or not turn.running
        if do_hard and turn.task is not None and not turn.task.done():
            turn.task.cancel()
        return {
            "cancelled": True,
            "request_id": turn.req_id,
            "mode": "hard" if do_hard else "cooperative",
            "was_running": turn.running,
            "age_s": round(turn.age_s, 1),
        }


def get_turn_registry(agent) -> TurnRegistry:
    """Get-or-create the registry on the agent instance."""
    reg = getattr(agent, "turn_registry", None)
    if not isinstance(reg, TurnRegistry):
        reg = TurnRegistry()
        try:
            agent.turn_registry = reg
        except Exception:  # noqa: BLE001 — a mock may refuse attributes
            pass
    return reg


__all__ = [
    "REASON_USER", "TurnCancelled", "ActiveTurn", "TurnRegistry",
    "get_turn_registry",
]
