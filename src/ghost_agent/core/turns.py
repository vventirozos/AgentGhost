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

#: §4RD-c: how often a generation in flight reads its cancel flag (seconds).
#: A cooperative Stop used to wait for the whole generation — minutes on
#: the 27B — because the flag was read only between model calls.
CANCEL_POLL_S = 0.2


class TurnCancelled(Exception):
    """Raised at a turn-loop boundary when a cooperative cancel was
    requested. Caught by ``handle_chat``, which returns the partial work
    instead of an error — so cancelling is a clean stop, not a crash."""

    def __init__(self, req_id: str = "", reason: str = REASON_USER,
                 partial: Optional[str] = None, lead: str = ""):
        self.req_id = req_id
        self.reason = reason
        # §4RD-c: raised MID-generation, the text the client was shown
        # (None at a loop boundary: the turn's last reply text stands)
        self.partial = partial
        # the tap's opening block (owner-rule banner, a staged correction),
        # which the partial no longer starts with: it goes back at the head
        self.lead = lead
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
    # Queued messages (core/queued_messages.py, 2026-10-28). `kind` is WHO
    # the turn serves (owner / probe / member…) — an inbox post must match.
    kind: str = ""
    inbox: List[dict] = field(default_factory=list)
    accepting: bool = True         # False once the loop has exited
    # progress, for the system's answer to a status question
    step: int = 0
    tools_now: List[str] = field(default_factory=list)
    tools_since: float = 0.0
    tools_done: int = 0
    # §4RZ step 5: the running batch's ToolStop (utils/tool_stop.py), or None
    tool_stop: Optional[object] = None

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
            "step": self.step,
            "tools_now": list(self.tools_now),
            "queued_pending": len(self.inbox),
        }


class TurnRegistry:
    """Thread-safe registry of in-flight turns. Never raises into a caller."""

    def __init__(self):
        self._turns: Dict[str, ActiveTurn] = {}
        self._lock = threading.Lock()
        # session_id -> [(ts, text)]: queued messages a turn accepted but
        # ended before reading; the client takes them and sends them normally
        self._leftovers: Dict[str, List[dict]] = {}

    def register(self, req_id: str, *, preview: str = "",
                 session_id: str = "", kind: str = "") -> ActiveTurn:
        base = str(req_id)
        turn = ActiveTurn(
            req_id=base,
            preview=" ".join(str(preview or "").split())[:120],
            session_id=str(session_id or ""),
            kind=str(kind or ""),
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
            self._close_locked(cur)

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
        with self._lock:                       # a stopped turn reads no more messages
            turn.accepting = False
            undelivered_items = [dict(m) for m in turn.inbox]
            undelivered = [m["text"] for m in undelivered_items]
            turn.inbox.clear()
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
            "undelivered": undelivered,
            "undelivered_items": undelivered_items,      # with accept times (routes pop it)
        }

    def interrupt_tools(self, req_id: str) -> List[str]:
        """§4RZ step 5: stop the tool call(s) this running turn is waiting
        on; the turn carries on. Returns the tools running when the stop was
        sent ([] when none is running, or a stop was already sent for this
        batch)."""
        turn = self.get(req_id)
        if turn is None or not turn.running or turn.cancelled:
            return []
        stop = turn.tool_stop
        names = stop.running_tools() if stop is not None else []
        if not names or not stop.request():
            return []                    # nothing running: a finished call is never "stopped"
        return names

    # ---- queued messages (core/queued_messages.py) -------------------

    def post(self, req_id: str, text: str) -> bool:
        """Put a routed message in the turn's inbox. False when the turn is
        gone, no longer accepting, or full — the caller sends it normally."""
        from .queued_messages import MAX_PENDING, MAX_TEXT
        with self._lock:
            turn = self._turns.get(str(req_id))
            if (turn is None or not turn.accepting or turn._cancel or len(turn.inbox) >= MAX_PENDING
                    or len(str(text)) > MAX_TEXT):          # never cut: the router queues long text
                return False
            turn.inbox.append({"text": str(text), "ts": time.time()})
            return True

    def drain(self, req_id: str) -> List[dict]:
        """The turn's pending messages, oldest first; the inbox is emptied."""
        with self._lock:
            turn = self._turns.get(str(req_id))
            if turn is None or not turn.inbox:
                return []
            out = list(turn.inbox)
            turn.inbox.clear()
            return out

    def close_inbox(self, req_id: str, turn: Optional["ActiveTurn"] = None) -> None:
        """The loop has exited: no message can be read any more. Messages
        still pending become the conversation's leftovers."""
        with self._lock:
            cur = turn if turn is not None else self._turns.get(str(req_id))
            self._close_locked(cur)

    def _close_locked(self, turn: Optional["ActiveTurn"]) -> None:
        if turn is None:
            return
        turn.accepting = False
        if turn.inbox and turn.session_id and not turn._cancel:
            self._add_leftovers_locked(turn.session_id, turn.kind, turn.inbox, tag=turn.req_id)
        turn.inbox.clear()

    def add_leftovers(self, session_id: str, kind: str, items: List[dict]) -> int:
        """§4SB: messages a turn DRAINED but never read (cancelled, or an
        upstream error after the drain) go to the conversation's leftovers
        like unread inbox rows do — ``items`` are ``{"text", "ts"}`` dicts;
        no cancel check (a cancelled turn's drained rows are exactly the
        case). Returns how many were kept."""
        items = [m for m in (items or []) if isinstance(m, dict) and str(m.get("text") or "").strip()]
        if not session_id or not items:
            return 0
        with self._lock:
            return self._add_leftovers_locked(str(session_id), str(kind or ""), items, tag="")

    def _add_leftovers_locked(self, session_id: str, kind: str, items, *, tag: str) -> int:
        """Append ``items`` to the session's leftovers: TTL-pruned, tagged
        with ``kind``, newest MAX_PENDING kept per session, LEFTOVER_MAX_SESSIONS
        bound. Logs what moved and what was dropped (§4SB: the cap dropped the
        oldest silently — two turns ending with unread rows lost the first's)."""
        from .queued_messages import MAX_PENDING, LEFTOVER_MAX_SESSIONS
        self._prune_leftovers_locked()
        kept = self._leftovers.pop(session_id, [])     # re-inserted: newest last
        # §4SB r2: the TTL counts from NOW (`kept_at`); the accept `ts` stays as it
        # was — a re-sent row's start is backdated to it (a hold typed after the
        # row still withdraws it), so it must never be refreshed
        now = time.time()
        kept.extend(dict(m, kind=kind, kept_at=now) for m in items)
        dropped = max(0, len(kept) - MAX_PENDING)
        self._leftovers[session_id] = kept[-MAX_PENDING:]
        while len(self._leftovers) > LEFTOVER_MAX_SESSIONS:
            self._leftovers.pop(next(iter(self._leftovers)))
        n = len(items)
        try:                              # a log line never raises into the registry
            from ..utils.logging import Icons, pretty_log
            head = f"[{(tag or session_id)[:8]}]"
            pretty_log("Queued Message", f"{head} {n} unread → leftovers", icon=Icons.QUEUED_MSG)
            if dropped:
                pretty_log("Queued Message",
                           f"{head} {dropped} older unread message(s) dropped (leftovers full)",
                           icon=Icons.QUEUED_MSG)
        except Exception:  # noqa: BLE001
            pass
        return min(n, MAX_PENDING)        # the cap drops the OLDEST: earlier rows first

    def _prune_leftovers_locked(self) -> None:
        from .queued_messages import LEFTOVER_TTL_S
        cutoff = time.time() - LEFTOVER_TTL_S
        for sid in list(self._leftovers):
            fresh = [m for m in self._leftovers[sid] if m.get("kept_at", m.get("ts", 0)) >= cutoff]
            if fresh:
                self._leftovers[sid] = fresh
            else:
                self._leftovers.pop(sid)

    def take_leftovers(self, session_id: str, *, kind: str) -> List[str]:
        """The conversation's leftovers that THIS kind of sender posted;
        others stay."""
        return [m["text"] for m in self.take_leftover_items(session_id, kind=kind)]

    def take_leftover_items(self, session_id: str, *, kind: str) -> List[dict]:
        """As take_leftovers, with each message's accept time (``ts``)."""
        with self._lock:
            self._prune_leftovers_locked()
            mine, rest = [], []
            for m in self._leftovers.pop(str(session_id or ""), []):
                (mine if m.get("kind") == kind else rest).append(m)
            if rest:
                self._leftovers[str(session_id)] = rest
            return [dict(m) for m in mine]

    def find_running(self, session_id: str, kind: str) -> Optional[ActiveTurn]:
        """The running turn of this conversation for this kind of sender — by
        session, never by `current()`. It may no longer accept messages (its
        streamed final is draining): a stop word or a status question still
        reaches it, and the router refuses an inject."""
        hits = self.find_running_all(session_id, kind)
        return hits[-1] if hits else None

    def find_running_all(self, session_id: str, kind: str) -> List[ActiveTurn]:
        """§4SB: EVERY running turn of this sender, oldest first — a draining
        streamed tail and a newer turn of the same conversation can both be
        running, and a "stop" must reach both (find_running alone hit the
        newest)."""
        if not session_id:
            return []
        with self._lock:
            hits = [t for t in self._turns.values()
                    if t.running and t.session_id == str(session_id) and t.kind == kind]
        return sorted(hits, key=lambda t: t.started_at)


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
