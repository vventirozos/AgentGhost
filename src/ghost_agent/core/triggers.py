"""Trigger taxonomy & replan bus — roadmap phase 2.5/2.6.

Today's agent has triggers, but they are scattered: thinking-loop
detection in ``agent.py``, strike counters next to tool dispatch,
``NodeCircuitBreaker`` in ``llm.py``, ad-hoc checks for free RAM in
``tools/system.py``. Each one logs and aborts in place. There is no
single signal-bus the planner can subscribe to, so when a trigger
fires the planner can't *replan* — it can only retry or abort.

This module promotes the three trigger classes the meta-cognition doc
calls out into first-class events:

  * ``LoopDetected``        — same tool / command / error 3× in a row,
                              or thinking-loop n-gram repetition.
  * ``ResourceExhausted``   — host telemetry breached a hard threshold
                              (RAM > 85%, free < 800MB, CPU sustained).
  * ``ExecutionAnomaly``    — a tool invocation exceeded ``p95 × 3``
                              of its historical runtime budget.

Producers publish to ``TriggerBus``; subscribers (the planner, the
arbiter, the watchdog) react. Each event carries enough context for
the planner's ``request_revision`` to be called with a meaningful
``reason``. ``ToolRuntimeBudget`` is a small helper that maintains a
p95 window per tool so producers can detect anomalies without
re-implementing rolling-percentile logic.

The bus is asyncio-native because the agent's I/O loop is asyncio,
but subscribers can be sync callables — the bus awaits coroutines and
calls sync callables directly. This mirrors the contract on
``HostTelemetry.subscribe`` so the two integrate cleanly.
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("GhostAgent")


# ──────────────────────────────────────────────────────────────────────
# Event types
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TriggerEvent:
    """Base event. Subclasses freeze additional fields on top."""

    ts: float
    kind: str            # "loop" | "resource" | "anomaly"
    severity: str        # "info" | "warning" | "critical"
    reason: str          # human-readable
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoopDetected(TriggerEvent):
    repeated_item: str = ""
    count: int = 0


@dataclass(frozen=True)
class ResourceExhausted(TriggerEvent):
    metric: str = ""           # "ram" | "cpu" | "ram_floor" | "disk"
    observed: float = 0.0
    threshold: float = 0.0


@dataclass(frozen=True)
class ExecutionAnomaly(TriggerEvent):
    tool_name: str = ""
    duration_s: float = 0.0
    budget_s: float = 0.0


TriggerHandler = Callable[[TriggerEvent], Optional[Awaitable[None]]]


# ──────────────────────────────────────────────────────────────────────
# Bus
# ──────────────────────────────────────────────────────────────────────

class TriggerBus:
    """Lightweight async pub/sub bus for trigger events.

    Subscribers can register either for a specific event kind ("loop")
    or for the wildcard "*". The bus never raises out of ``publish``;
    a misbehaving subscriber is isolated to its own try/except so the
    rest of the chain still fires.
    """

    KINDS = ("loop", "resource", "anomaly", "*")

    def __init__(self):
        self._subs: Dict[str, List[TriggerHandler]] = {k: [] for k in self.KINDS}
        self._history: List[TriggerEvent] = []
        self._history_cap = 64

    def subscribe(self, kind: str, handler: TriggerHandler) -> None:
        bucket = self._subs.setdefault(kind, [])
        if handler not in bucket:
            bucket.append(handler)

    def unsubscribe(self, kind: str, handler: TriggerHandler) -> None:
        bucket = self._subs.get(kind, [])
        if handler in bucket:
            bucket.remove(handler)

    async def publish(self, event: TriggerEvent) -> None:
        self._history.append(event)
        if len(self._history) > self._history_cap:
            self._history = self._history[-self._history_cap:]
        # Fire kind-specific then wildcard
        for handler in list(self._subs.get(event.kind, [])):
            await self._call(handler, event)
        for handler in list(self._subs.get("*", [])):
            await self._call(handler, event)

    def history(self, kind: Optional[str] = None) -> List[TriggerEvent]:
        if kind is None:
            return list(self._history)
        return [e for e in self._history if e.kind == kind]

    async def _call(self, handler: TriggerHandler, event: TriggerEvent) -> None:
        try:
            r = handler(event)
            if asyncio.iscoroutine(r):
                await r
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("TriggerBus handler %s failed: %s", handler, exc)


# ──────────────────────────────────────────────────────────────────────
# Runtime budget tracker
# ──────────────────────────────────────────────────────────────────────

class ToolRuntimeBudget:
    """Maintain a rolling p95-runtime window per tool name.

    The doc's Execution Anomaly Trigger is "tool exceeded its expected
    runtime". This class is the cheap, dependency-free percentile
    tracker that feeds that detection. Cold start: until a tool has
    been called ``min_samples`` times, the budget is unbounded (no
    anomaly can fire — preventing false positives during warm-up).
    """

    def __init__(self, *, window: int = 50, min_samples: int = 10,
                 multiplier: float = 3.0):
        self.window = int(max(2, window))
        self.min_samples = int(max(3, min_samples))
        self.multiplier = float(max(1.0, multiplier))
        self._samples: Dict[str, List[float]] = {}

    def record(self, tool_name: str, duration_s: float) -> None:
        if not tool_name or duration_s < 0:
            return
        bucket = self._samples.setdefault(tool_name, [])
        bucket.append(float(duration_s))
        if len(bucket) > self.window:
            del bucket[: len(bucket) - self.window]

    def budget(self, tool_name: str) -> Optional[float]:
        """Return the current anomaly budget for ``tool_name`` — i.e.
        ``p95 * multiplier``. ``None`` during cold start."""
        bucket = self._samples.get(tool_name)
        if not bucket or len(bucket) < self.min_samples:
            return None
        sorted_b = sorted(bucket)
        idx = int(0.95 * (len(sorted_b) - 1))
        p95 = sorted_b[idx]
        return p95 * self.multiplier

    def is_anomalous(self, tool_name: str, duration_s: float) -> bool:
        budget = self.budget(tool_name)
        return budget is not None and duration_s > budget


# ──────────────────────────────────────────────────────────────────────
# Repetition counter
# ──────────────────────────────────────────────────────────────────────

class RepetitionCounter:
    """Tracks recurrences of the same key in a short rolling history.

    The doc's Loop Detection Trigger is "same item N times in
    succession" — a slightly different shape from the existing
    ``thinking_loop`` and ``cross_turn_repeat`` detectors in
    ``agent.py``. This counter is sequence-aware (resets on a
    different key, not on a window slide), which is exactly the
    semantics ``request_revision`` needs to attribute a loop to a
    specific failing action.
    """

    def __init__(self, *, threshold: int = 3):
        self.threshold = int(max(2, threshold))
        self._last: Optional[str] = None
        self._streak: int = 0

    def observe(self, key: str) -> int:
        """Record one observation. Returns the current streak length."""
        if not key:
            return 0
        if key == self._last:
            self._streak += 1
        else:
            self._last = key
            self._streak = 1
        return self._streak

    def tripped(self) -> bool:
        return self._streak >= self.threshold

    def reset(self) -> None:
        self._last = None
        self._streak = 0


# ──────────────────────────────────────────────────────────────────────
# Pre-flight repeat-failure guard
# ──────────────────────────────────────────────────────────────────────

class RecentFailureGuard:
    """Rolling memory of recently-FAILED tool calls, consulted *before*
    dispatch.

    The offline post-mortem (``reflection/postmortem.py``) mines a
    ``TranscriptSignature`` that already names the dominant pathology —
    "the SAME error from the same tool recurred Nx" — but it only runs in
    the idle watchdog, *after* a failed run has burned its turns. This is
    the live counterpart: before a tool is dispatched the agent asks
    "have I already failed this exact action the same way?", and if so the
    call is blocked and the model is handed the prior error instead of
    re-running a known failure.

    Keying mirrors the offline detector so the two agree on "the same
    action": ``(tool, primary-target)`` where the target comes from
    ``primary_target_from_args`` (path / url / selector / query / …). The
    error text is normalised to a short lowercased prefix so two failures
    that differ only in a trailing variable (an offset, a pid) count as the
    same recurring failure.

    The window is bounded (``window``) and entries age out as new failures
    push them off the back, so a one-off failure that isn't repeated soon
    simply falls out of memory and never blocks anything — the guard only
    fires on a *sustained* identical loop.

    Lifecycle (2026-07-30, solar-sim postmortem): a blocked call never
    dispatches, so it can never demonstrate that an intervening fix worked
    — without an external reset, two stale failures block a verifiably
    remediated action forever (observed live: the port holder was killed
    and the port confirmed free, yet three consecutive requests were boxed
    in by failures recorded before the fix). The dispatch loop therefore
    (a) calls :meth:`note_world_changed` after any SUCCESSFUL state-
    mutating call, and (b) calls :meth:`reset` at the start of each
    request — cross-request pathology stays covered by the offline
    post-mortem fingerprint, which mines whole transcripts.
    """

    def __init__(self, *, window: int = 24, repeat_threshold: int = 2):
        self.window = int(max(2, window))
        # Number of PRIOR identical (same tool, target, normalised error)
        # failures that must already be on record before the next attempt
        # is blocked. Default 2 → the action is allowed to fail twice the
        # SAME way (proving it isn't transient and wasn't fixed between
        # attempts) before the third identical re-issue is intercepted.
        #
        # Why not 1: keying on (tool, target) alone would block a LEGITIMATE
        # retry — e.g. re-running `execute` on a script after editing it to
        # fix the bug (the same pathology the idempotency guard deliberately
        # exempts `execute`/`file_system.write` for). Requiring the identical
        # ERROR to recur means a post-fix re-run that now succeeds, or fails a
        # NEW way, never trips the guard — only a genuine stuck loop does.
        self.repeat_threshold = int(max(1, repeat_threshold))
        # Each entry: ((tool, target, op), normalised-error).
        self._history: Deque[Tuple[Tuple[str, str], str]] = deque(maxlen=self.window)

    @staticmethod
    def _norm_err(text: str) -> str:
        """Normalised prefix of an error string — same shape as the offline
        ``_error_key`` so the live and offline notions of "the same error"
        line up."""
        return (text or "").strip()[:80].lower()

    def record(self, tool: str, target: str, error_text: str,
               op: str = "") -> None:
        """Remember one FAILED call. No-op for an empty tool or error (a
        successful call has no error and must never seed the guard).

        ``op`` is the tool's operation/action discriminator: for multi-op
        tools (file_system, manage_projects, browser) two calls on the same
        target with DIFFERENT operations are different actions — observed
        live 2026-07-08: keying on (tool, target) alone blocked the model's
        correct replace→write recovery three times in a row."""
        err = self._norm_err(error_text)
        if not tool or not err:
            return
        self._history.append(((tool, target or "", op or ""), err))

    def would_repeat(self, tool: str, target: str, op: str = "") -> Optional[str]:
        """If dispatching ``tool`` against ``target`` would re-run an action
        that has already failed identically at least ``repeat_threshold``
        times in the window, return the offending error text (for the
        corrective message); otherwise ``None``.

        "Identically" = same ``(tool, target)`` AND same normalised error.
        The most recent error recorded for the key anchors the match, so a
        target that failed two *different* ways does not trip the guard —
        only a genuine repeat of one failure does.
        """
        if not tool:
            return None
        key = (tool, target or "", op or "")
        last_err = None
        for k, err in reversed(self._history):
            if k == key:
                last_err = err
                break
        if last_err is None:
            return None
        count = sum(1 for k, err in self._history if k == key and err == last_err)
        if count >= self.repeat_threshold:
            return last_err
        return None

    def note_world_changed(self) -> int:
        """Forget every recorded failure because a SUCCESSFUL state-mutating
        call just ran (a file write, a service start/stop, a process kill).

        The guard's premise is "re-running the recorded call UNCHANGED will
        fail the same way" — a successful mutation invalidates that premise
        for everything on record, since the failure's cause may be exactly
        what was just changed (the live pathology: `manage_services start`
        failed twice on an occupied port, the model killed the holder via
        shell, and the now-correct retry was blocked pre-dispatch, so the
        guard could never learn the world had moved). Clearing is deliberately
        global rather than per-key: the guard cannot know which keys the
        mutation affected, a false block costs far more than a false allow,
        and ``repeat_threshold`` fresh identical failures re-arm it quickly
        if the change fixed nothing.

        Returns the number of entries dropped so the call site can log only
        when the reset actually mattered."""
        n = len(self._history)
        self._history.clear()
        return n

    def reset(self) -> None:
        self._history.clear()


def guard_key_target(primary_target: str, call_hash: str) -> str:
    """The target half of a :class:`RecentFailureGuard` key.

    When the call has a recognised primary target (path / url / query / …
    via ``primary_target_from_args``) that IS the key, matching the offline
    post-mortem's notion of "the same action". When it has none — e.g.
    ``manage_services``, whose identity lives in ``name``/``port``/
    ``command`` — fall back to a short signature of the FULL canonical call
    (``call_hash`` is the dispatch loop's ``fname:json(args)`` string), so
    a retry with ANY changed argument is a DIFFERENT action. This makes the
    guard's message ("re-running it UNCHANGED will fail the same way")
    literally true: before this fix every ``manage_services start`` — any
    service, any port, any command — collapsed into one key, and legitimately
    changed retries (new port, new command) were blocked (2026-07-30)."""
    if primary_target:
        return primary_target
    if not call_hash:
        return ""
    return "args#" + hashlib.sha1(
        call_hash.encode("utf-8", "replace")).hexdigest()[:12]


# World-mutation shell heuristic for the failure guard's world-changed
# reset. Deliberately narrower than execute.py's _SHELL_MUTATION_RE (whose
# bare `>>?` counts every `2>/dev/null` probe as a mutation — safe for a
# refusal gate, but here it would clear the guard on nearly every probe and
# nullify it) and wider on process mutations, which that regex lacks and
# which were the live remediation (`kill -9`, `fuser -k`). `kill -0` /
# `pkill -0` are liveness PROBES and are excluded.
#
# Two precision rails (2026-07-30 review — probe traffic dominates this
# agent's shell use, so frequent false positives would keep the guard
# permanently disarmed):
#   * mutation VERBS must sit at command position (start of string, after
#     ;|&|(|`|$( separators, or after a runner prefix like sudo/xargs/
#     timeout) — `grep -rn 'mkdir' src/` and `ls release.tar.gz` mention
#     verbs as DATA, not commands;
#   * quoted segments are stripped before matching (`awk '$3 > 100'` is
#     not a redirect), EXCEPT that a `sh -c '…'` payload is itself a
#     command and gets its own scan.
# Residual false positives merely allow one extra real attempt (the
# threshold re-arms on fresh failures); false negatives preserve the
# block — the failure mode being fixed.
_WM_CMD_POS = (
    r"(?:^|[;&|(`]\s*|\$\(\s*"
    r"|\b(?:sudo|nohup|setsid|xargs|exec|env|command|do|then)\s+"
    r"|-exec\s+"
    r"|\btimeout\s+(?:-\S+\s+)*\S+\s+)"
)
_WM_VERBS = (
    r"(?:kill(?:all)?\s+(?!-0\b)"                      # kill/killall, not `kill -0` probes
    r"|pkill\s+(?!-0\b)"                               # pkill, not `pkill -0` probes
    r"|fuser\s+-[A-Za-z]*k\b"                          # fuser -k / -sk … (bare fuser = probe)
    r"|(?:rm|mv|cp|tee|touch|mkdir|rmdir|chmod|chown|truncate|dd|ln|unzip|tar|patch)\b"
    r"|sed\s+-i\b|perl\s+-i\b"
    # service managers only with a mutating subcommand nearby (either order:
    # `systemctl restart nginx` / `service nginx restart`) — `systemctl
    # status` and prose mentions of "service" are observations.
    r"|(?:systemctl|launchctl|service|supervisorctl)\s+(?:[\w./@:-]+\s+){0,2}"
    r"(?:start|stop|restart|kill|reload|enable|disable|bootout|bootstrap|kickstart)\b)"
)
_WORLD_MUTATING_CMD_RE = re.compile(
    _WM_CMD_POS + _WM_VERBS
    + r"|(?<![\d&>])>(?!&)"                            # bare redirect; not 2>/dev/null, 2>&1, or >>'s 2nd char
)
_WM_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
# A nested shell's -c payload IS a command string (the model often wraps:
# `bash -c 'kill -9 …'`). python/node -c payloads are code, not shell —
# only sh/bash/zsh/dash qualify.
_WM_SHELL_C_RE = re.compile(
    r"\b(?:ba|z|da)?sh\s+(?:-\S+\s+)*-c\s+(?:'([^']*)'|\"([^\"]*)\")"
)


def looks_mutating_command(command: str) -> bool:
    """True when a shell command plausibly mutates world state (kills a
    process, moves/removes files, drives a service manager, redirects into
    a file) rather than merely observing it. Consumed by the dispatch loop
    to decide whether a SUCCESSFUL ``execute`` call should clear the
    pre-flight failure guard via ``note_world_changed``."""
    if not command:
        return False
    cmd = str(command)
    for m in _WM_SHELL_C_RE.finditer(cmd):
        inner = m.group(1) or m.group(2) or ""
        if inner and _WORLD_MUTATING_CMD_RE.search(inner):
            return True
    return bool(_WORLD_MUTATING_CMD_RE.search(_WM_QUOTED_RE.sub(" ", cmd)))


# ──────────────────────────────────────────────────────────────────────
# Replan bridge
# ──────────────────────────────────────────────────────────────────────

class ReplanBridge:
    """Glue that turns trigger events into ``request_revision`` calls
    on a ``ProjectPlan`` instance. The plan/tree is held weakly — the
    bridge has no opinion about plan lifecycle and is happy when
    ``current_task_id`` is unset (no live task → silent drop).

    Wire it once at startup::

        bus = TriggerBus()
        bridge = ReplanBridge(bus, plan_getter=lambda: agent.current_plan)
        bridge.attach()
    """

    def __init__(
        self,
        bus: TriggerBus,
        *,
        plan_getter: Optional[Callable[[], Any]] = None,
        current_task_getter: Optional[Callable[[], Optional[str]]] = None,
        counter_hook: Optional[Callable[..., None]] = None,
    ):
        self.bus = bus
        self.plan_getter = plan_getter
        self.current_task_getter = current_task_getter
        # Optional lifetime-counter hook, called with keyword flags that
        # match ``MetacogBundle.count`` (``replan_attempt=True`` when a
        # ``request_revision`` call is issued, ``replan_succeeded=True``
        # when it returns ok) so the construction site can pass
        # ``bundle.count`` directly. Default ``None`` = no accounting,
        # behaviour identical to before the hook existed.
        self.counter_hook = counter_hook
        self._attached = False
        # Bounded: every published event (incl. recurring host-telemetry
        # heartbeats) appends a record here — an unbounded list grows for
        # the daemon's whole uptime. The bus's own history is capped at 64;
        # keep this audit log capped too.
        self._revisions: Deque[Dict[str, Any]] = deque(maxlen=256)

    def attach(self) -> None:
        if self._attached:
            return
        self.bus.subscribe("*", self._on_event)
        self._attached = True

    def detach(self) -> None:
        if not self._attached:
            return
        self.bus.unsubscribe("*", self._on_event)
        self._attached = False

    @property
    def revisions(self) -> List[Dict[str, Any]]:
        """Audit log of replan attempts. Tests and observability use this."""
        return list(self._revisions)

    async def _on_event(self, event: TriggerEvent) -> None:
        plan = self._safe_call(self.plan_getter)
        task_id = self._safe_call(self.current_task_getter)
        record = {
            "ts": event.ts, "kind": event.kind, "severity": event.severity,
            "reason": event.reason, "task_id": task_id,
        }
        # Only critical / warning events trigger replan; info events
        # are observability-only.
        if event.severity == "info":
            record["action"] = "noop:info"
            self._revisions.append(record)
            return  # silent — audit lives in `self._revisions`
        if plan is None or task_id is None:
            record["action"] = "noop:no_plan"
            self._revisions.append(record)
            return  # silent
        try:
            req = getattr(plan, "request_revision", None)
            tree_req = getattr(getattr(plan, "tree", None), "request_revision",
                               None)
            fn = req or tree_req
            if fn is None:
                record["action"] = "noop:no_request_revision"
                self._revisions.append(record)
                return  # silent
            reason = f"{event.kind}/{event.severity}: {event.reason}"
            self._count(replan_attempt=True)
            ok = fn(task_id, reason)
            if ok:
                self._count(replan_succeeded=True)
            record["action"] = "revised" if ok else "revision_rejected"
            self._revisions.append(record)
            self._emit(record["action"], event, task_id)
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("ReplanBridge request_revision failed: %s", exc)
            record["action"] = f"error:{type(exc).__name__}"
            self._revisions.append(record)
            self._emit(record["action"], event, task_id)

    def _emit(self, action: str, event: TriggerEvent,
              task_id: Optional[str]) -> None:
        """Structured log line per ACTIONABLE replan attempt.

        Noop cases (no_plan, no_request_revision, info severity) are
        deliberately silent — they fired before pretty_log existed for
        metacog and producing one line per 1Hz host signal turned the
        log into spam (`action=noop:no_plan ... reason="RAM 96%"`
        repeated forever on a host where the LLM server pins memory
        as steady state). The full audit still lives in
        ``self._revisions`` for programmatic inspection.
        """
        try:
            from .metacog_log import (
                emit as _mc_emit, Subsystem as _mc_ss,
                LEVEL_INFO, LEVEL_WARN,
            )
            lvl = LEVEL_INFO if action == "revised" else LEVEL_WARN
            _mc_emit(
                _mc_ss.REPLAN, level=lvl,
                action=action,
                trigger=f"{event.kind}/{event.severity}",
                task=task_id,
                reason=event.reason,
            )
        except Exception as exc:
            logger.debug("ReplanBridge log emit failed: %s", exc)

    def _count(self, **counters: bool) -> None:
        """Invoke the optional counter hook. Never raises — accounting
        must not break a replan."""
        if self.counter_hook is None:
            return
        try:
            self.counter_hook(**counters)
        except Exception as exc:
            logger.debug("ReplanBridge counter hook failed: %s", exc)

    @staticmethod
    def _safe_call(fn):
        if fn is None:
            return None
        try:
            return fn()
        except Exception:
            return None


# ──────────────────────────────────────────────────────────────────────
# Convenience factories
# ──────────────────────────────────────────────────────────────────────

def loop_event(reason: str, *, key: str = "", count: int = 0,
               severity: str = "warning") -> LoopDetected:
    return LoopDetected(
        ts=time.time(), kind="loop", severity=severity, reason=reason,
        repeated_item=key, count=int(count),
    )


def resource_event(reason: str, *, metric: str, observed: float,
                   threshold: float, severity: str = "warning") -> ResourceExhausted:
    return ResourceExhausted(
        ts=time.time(), kind="resource", severity=severity, reason=reason,
        metric=metric, observed=float(observed), threshold=float(threshold),
    )


def anomaly_event(reason: str, *, tool_name: str, duration_s: float,
                  budget_s: float, severity: str = "warning") -> ExecutionAnomaly:
    return ExecutionAnomaly(
        ts=time.time(), kind="anomaly", severity=severity, reason=reason,
        tool_name=tool_name, duration_s=float(duration_s), budget_s=float(budget_s),
    )


__all__ = [
    "TriggerBus",
    "TriggerEvent",
    "LoopDetected",
    "ResourceExhausted",
    "ExecutionAnomaly",
    "ToolRuntimeBudget",
    "RepetitionCounter",
    "RecentFailureGuard",
    "guard_key_target",
    "looks_mutating_command",
    "ReplanBridge",
    "loop_event",
    "resource_event",
    "anomaly_event",
]
