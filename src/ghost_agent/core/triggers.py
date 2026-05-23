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
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

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
    ):
        self.bus = bus
        self.plan_getter = plan_getter
        self.current_task_getter = current_task_getter
        self._attached = False
        self._revisions: List[Dict[str, Any]] = []

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
            ok = fn(task_id, reason)
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
    "ReplanBridge",
    "loop_event",
    "resource_event",
    "anomaly_event",
]
