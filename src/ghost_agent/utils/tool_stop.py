"""§4RZ step 5: the owner stops the tool call(s) running right now, and the
turn carries on.

One ``ToolStop`` per tool batch. Each call of the batch runs inside
`agent._stoppable_call`, which binds the stop in a ContextVar for that call
alone (the loop's own context never carries it), notes when the call really
started and while it runs. Tasks and ``asyncio.to_thread`` copy the context,
so code that runs a tool in a worker thread (the sandbox job supervisor,
yt-dlp, the browser runner's exec) sees the stop and kills what it started.

A killer ARMS for its tool while it can still act, and reports a confirmed
kill (``note_killed``); the batch waits for armed killers to settle before it
writes what the model reads, which says only what is certain: stopped (a
confirmed kill), discarded (its work may still finish in the background), or
not run.

No stop exists when queued messages are off, so every reader is a no-op.
"""
from __future__ import annotations

import asyncio
import collections
import contextvars
import logging
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Set

logger = logging.getLogger("GhostAgent")

#: how long the batch waits, after a stop, for its cancelled calls to unwind
#: (a browser call holds its profile lock until the killed runner returns)
STOP_GRACE_S = 20.0
#: the batch's poll for a stop
STOP_POLL_S = 0.2
#: how long the batch waits for armed killers to confirm (or give up)
KILL_CONFIRM_S = 8.0
#: the reason code an interrupted call's outcome carries
REASON = "user_interrupted"


class ToolStop:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        # keyed by CALL (the batch index), never by tool name (s5-r2: two
        # knowledge_base calls, one killed, both read "it was stopped")
        self._killed: Set[int] = set()
        self._callbacks: List[Callable[[], None]] = []
        self._started: Set[int] = set()
        self._running: Dict[int, str] = {}
        self._hit: Set[int] = set()
        self._armed: collections.Counter = collections.Counter()
        self._killers = 0

    # -- the stop -----------------------------------------------------------
    def requested(self) -> bool:
        return self._event.is_set()

    def request(self) -> bool:
        """Set the stop; run each registered killer in its own thread (a
        killer may block on a docker exec). False when already set."""
        with self._lock:
            if self._event.is_set():
                return False
            self._event.set()
            # the calls the stop reached: stopped, whatever they return
            # (s5-r2: one killed inside the poll gap came back as its raw error)
            self._hit = set(self._running)
            callbacks = list(self._callbacks)
            self._killers += len(callbacks)
        for fn in callbacks:
            self._start_killer(fn)
        return True

    def on_stop(self, fn: Callable[[], None]) -> Callable[[], None]:
        """Register a killer while a call is running; returns its removal.
        A stop already requested runs it at once — in a thread, never on the
        caller's (event-loop) thread."""
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(fn)
                return lambda: self._remove(fn)
            self._killers += 1
        self._start_killer(fn)
        return lambda: None

    def _start_killer(self, fn) -> None:
        def _run():
            try:
                _run_quietly(fn)
            finally:
                with self._lock:
                    self._killers -= 1
        threading.Thread(target=_run, daemon=True, name="ghost-tool-stop").start()

    def _remove(self, fn) -> None:
        with self._lock:
            try:
                self._callbacks.remove(fn)
            except ValueError:
                pass

    # -- the calls ----------------------------------------------------------
    def enter(self, idx: int, tool: str) -> None:
        with self._lock:
            self._started.add(idx)
            self._running[idx] = str(tool)

    def leave(self, idx: int) -> None:
        with self._lock:
            self._running.pop(idx, None)

    def started(self, idx: int) -> bool:
        with self._lock:
            return idx in self._started

    def hit(self, idx: int) -> bool:
        with self._lock:
            return idx in self._hit

    def running_tools(self) -> List[str]:
        """The tools of the calls running right now, in batch order."""
        with self._lock:
            return [self._running[k] for k in sorted(self._running)]

    # -- the killers --------------------------------------------------------
    def arm(self, key) -> None:
        with self._lock:
            self._armed[key] += 1

    def disarm(self, key) -> None:
        with self._lock:
            self._armed[key] -= 1

    def note_killed(self, key) -> None:
        with self._lock:
            self._killed.add(key)

    def killed(self, key) -> bool:
        with self._lock:
            return key in self._killed

    def settled(self, keys: Iterable = ()) -> bool:
        with self._lock:
            return self._killers <= 0 and all(self._armed[k] <= 0 for k in keys)


def _run_quietly(fn) -> None:
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — a killer never raises into a tool
        logger.warning("tool stop: a killer failed: %s: %s", type(e).__name__, e)


async def settle(stop: ToolStop, keys: Iterable, timeout: float = KILL_CONFIRM_S) -> None:
    """Wait until no killer for the calls ``keys`` is still acting (a worker
    thread notices the stop at its next poll, after the batch moved on)."""
    keys = list(keys)
    deadline = time.monotonic() + timeout
    while not stop.settled(keys) and time.monotonic() < deadline:
        await asyncio.sleep(0.05)


_CURRENT: contextvars.ContextVar[Optional[ToolStop]] = contextvars.ContextVar(
    "ghost_tool_stop", default=None)
#: the batch index of the call running in this context
_CALL: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "ghost_tool_stop_call", default=None)


def bind_call(stop: Optional[ToolStop], idx: Optional[int]):
    return _CURRENT.set(stop), _CALL.set(idx)


def unbind_call(tokens) -> None:
    for var, tok in zip((_CURRENT, _CALL), tokens):
        try:
            var.reset(tok)
        except (ValueError, LookupError):
            var.set(None)


def call_key() -> Optional[int]:
    return _CALL.get()


def bind(stop: Optional[ToolStop]):
    return _CURRENT.set(stop)


def unbind(token) -> None:
    try:
        _CURRENT.reset(token)
    except (ValueError, LookupError):
        _CURRENT.set(None)


def current() -> Optional[ToolStop]:
    return _CURRENT.get()


def requested() -> bool:
    s = _CURRENT.get()
    return bool(s is not None and s.requested())


def note_killed(tool: str = "") -> None:
    """A killer confirmed it stopped THIS call's work (``tool`` names it in
    logs only)."""
    s = _CURRENT.get()
    if s is not None:
        s.note_killed(_CALL.get())


def arm(tool: str = ""):
    """Arm the bound stop's killer for this call; returns a token for
    `disarm`, or None when nothing is bound."""
    s = _CURRENT.get()
    if s is None:
        return None
    key = _CALL.get()
    s.arm(key)
    return (s, key)


def disarm(token, tool: str = "") -> None:
    if token is not None:
        token[0].disarm(token[1])


#: the fixed heads of `interrupted_text` — a stored row keeps only its text
_HEADS = ("INTERRUPTED BY THE USER: the user stopped this ",
          "NOT RUN: the user interrupted the tool batch before this ")


def is_user_interrupted(result) -> bool:
    """Was this tool result a §4RZ step-5 stop? Not a success, not a
    failure, and not still running."""
    if getattr(result, "reason_code", None) == REASON:
        return True
    return str(result or "").startswith(_HEADS)


class Interrupted:
    """A batch result slot the stop left without a result. ``started``:
    the call was running (else it never began)."""
    __slots__ = ("started",)

    def __init__(self, started: bool) -> None:
        self.started = started


def interrupted_text(tool: str, *, started: bool, killed: bool) -> str:
    """The tool result the model reads. Says only what is certain."""
    tail = (" Do not run it again unless the user asks for it; go on with the rest of the "
            "request, or ask the user what they want instead.")
    if not started:
        return (f"NOT RUN: the user interrupted the tool batch before this {tool} call started."
                + tail)
    if killed:
        return (f"INTERRUPTED BY THE USER: the user stopped this {tool} call while it was "
                "running, and it was stopped. Its output is not available." + tail)
    return (f"INTERRUPTED BY THE USER: the user stopped this {tool} call while it was "
            "running. Its output was discarded; work it started may still finish in the "
            "background." + tail)


def uncut_note(tool: str) -> str:
    """Appended to a call the stop reached but never cuts (a file
    operation): it finished, and the model still reads that the user asked
    to stop (s5-r9: a lone write's stop answered 200 and left no trace in
    the next prompt). Added to the model's tool message only, never to the
    result (s5-r12)."""
    return (f"\n\n(The user asked to stop the tools while this {tool} call was running. "
            f"{tool} calls are never cut, so it finished and the result above is real. "
            "Do not start further work the user may not want; go on with the rest of "
            "the request, or ask the user what they want instead.)")


def interrupted_outcome(tool: str, *, started: bool, killed: bool):
    """The result as a `ToolOutcome`: UNRESOLVED (not a success — never
    memoised, never recorded as applied, never a passing run — and not a
    failure — no strike), reason ``user_interrupted``; a call that ran may
    have changed something before it stopped."""
    from ..tools.outcome import OutcomeStatus, ToolOutcome
    return ToolOutcome(interrupted_text(tool, started=started, killed=killed),
                       status=OutcomeStatus.UNRESOLVED, reason_code=REASON,
                       world_changed=bool(started))
