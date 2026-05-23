"""Host telemetry poller — grounded monitoring for the agent's own
runtime. Roadmap phase 1.1/1.2 of the metacognition uplift.

The agent already exposes `tool_check_health` as a callable tool, but
that path is opt-in and snapshots-on-demand. The poller defined here
runs in the background of the asyncio event loop at a low fixed cadence
(default 1 Hz), keeps a bounded ring buffer of the last ``N`` snapshots,
and emits ``HostSignal`` events on the registered ``TriggerBus`` when
configured thresholds are tripped.

Design constraints (intentional):
  * Lock-free read path. ``snapshot()`` returns the most recent reading
    by index lookup — no locks. Writers append-and-bump under a single
    asyncio task, so a torn read is impossible: the consumer either
    reads the previous complete snapshot or the new one.
  * Never raises. psutil glitches (missing /proc on a stripped container,
    permission errors on macOS) are logged and the affected field is set
    to NaN — that way callers can compare with ``math.isfinite`` and
    treat the sample as "unknown" rather than "healthy".
  * Cheap. ``psutil.cpu_percent(interval=None)`` returns the cached
    delta since the last call; we sample once per second so the cost is
    sub-millisecond on a hot path that already runs.

This module deliberately stays independent of ``triggers.py`` so it can
be imported and used as a thin probe in tests without dragging in the
trigger taxonomy. The bridge to triggers happens in the daemon caller
(``GhostAgent.biological_watchdog``) which subscribes to host signals.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Deque, List, Optional

logger = logging.getLogger("GhostAgent")


# ──────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HostSnapshot:
    """One reading of the host's vital signs.

    Frozen so callers can pass snapshots around without worrying about
    a producer mutating them mid-read.
    """

    ts: float                # wall-clock time of the sample
    cpu_percent: float       # 0-100; NaN when psutil refused
    mem_percent: float       # 0-100; NaN when unavailable
    mem_available_mb: float  # MB available; NaN when unavailable
    disk_percent: float      # 0-100 on '/'; NaN when unavailable
    proc_count: int          # active processes; -1 when unavailable

    @property
    def healthy(self) -> bool:
        """Quick boolean: nothing tripped a soft threshold yet.

        Used by call sites that just want a "is the host okay right now"
        check without a full ``HostSignal`` subscription. The thresholds
        here mirror the defaults on ``HostTelemetry`` so behaviour is
        consistent between the two access patterns.
        """
        if math.isfinite(self.cpu_percent) and self.cpu_percent >= 85.0:
            return False
        if math.isfinite(self.mem_percent) and self.mem_percent >= 85.0:
            return False
        if math.isfinite(self.mem_available_mb) and self.mem_available_mb < 800.0:
            return False
        return True


@dataclass(frozen=True)
class HostSignal:
    """A trigger fired by the poller.

    ``severity`` is "info" (one threshold crossed once), "warning"
    (sustained crossing or two thresholds simultaneously), or
    "critical" (free RAM below the hard floor, where OOM is imminent).
    The agent's halt logic reads severity, not the raw thresholds, so
    the policy can evolve without changing call sites.
    """

    ts: float
    severity: str            # info | warning | critical
    reason: str              # human-readable, used by pretty_log
    snapshot: HostSnapshot


# Type alias for subscribers; sync or async callables both work.
HostSignalHandler = Callable[[HostSignal], Optional[Awaitable[None]]]


# ──────────────────────────────────────────────────────────────────────
# Poller
# ──────────────────────────────────────────────────────────────────────

class HostTelemetry:
    """Background asyncio poller maintaining a ring buffer of snapshots.

    Lifecycle::

        telemetry = HostTelemetry()
        telemetry.subscribe(on_signal)
        await telemetry.start()
        ...
        await telemetry.stop()

    ``snapshot()`` is safe to call without ``start()``; it returns a
    single on-demand reading. ``start()`` is what wires the periodic
    poll loop and signal dispatch.

    Tests can inject a synthetic ``probe`` to avoid taking a dependency
    on the host's actual psutil readings. Production code leaves it
    None so the default psutil probe is used.
    """

    # Threshold defaults are conservative for an 8GB edge device (Jetson
    # Nano Orin target). Override at construction for beefier hardware.
    DEFAULT_CPU_HIGH = 85.0
    DEFAULT_MEM_HIGH = 85.0
    DEFAULT_MEM_FLOOR_MB = 800.0
    DEFAULT_DISK_HIGH = 90.0
    # Sustained-crossing window: severity escalates to warning when the
    # threshold is crossed in N consecutive samples.
    DEFAULT_SUSTAIN_SAMPLES = 3

    def __init__(
        self,
        *,
        interval_s: float = 1.0,
        history: int = 60,
        cpu_high: float = DEFAULT_CPU_HIGH,
        mem_high: float = DEFAULT_MEM_HIGH,
        mem_floor_mb: float = DEFAULT_MEM_FLOOR_MB,
        disk_high: float = DEFAULT_DISK_HIGH,
        sustain_samples: int = DEFAULT_SUSTAIN_SAMPLES,
        heartbeat_s: float = 300.0,
        probe: Optional[Callable[[], HostSnapshot]] = None,
    ):
        self.interval_s = max(0.1, float(interval_s))
        self._history: Deque[HostSnapshot] = deque(maxlen=int(max(1, history)))
        self.cpu_high = float(cpu_high)
        self.mem_high = float(mem_high)
        self.mem_floor_mb = float(mem_floor_mb)
        self.disk_high = float(disk_high)
        self.sustain_samples = int(max(1, sustain_samples))
        # Dedup window. When (metric-set, severity) is unchanged from the
        # previous emission, suppress the signal — but re-emit every
        # ``heartbeat_s`` seconds so the operator still sees a periodic
        # "still degraded" trail in the log. Set to 0 to disable dedup
        # entirely (every poll that crosses thresholds emits).
        self.heartbeat_s = float(max(0.0, heartbeat_s))
        self._probe = probe or _psutil_probe
        self._subscribers: List[HostSignalHandler] = []
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        # Per-axis consecutive-crossing counters. Reset to 0 on a healthy
        # sample; an axis "sustains" when it hits ``sustain_samples``.
        self._cpu_run = 0
        self._mem_run = 0
        self._mem_floor_run = 0
        self._disk_run = 0
        # Dedup state: ``(frozenset_of_metric_tokens, severity)`` of the
        # last emitted signal, plus its timestamp. ``None`` until the
        # first emission.
        self._last_emit_key: Optional[tuple] = None
        self._last_emit_ts: float = 0.0

    # -------------------------------------------------------------- API

    def subscribe(self, handler: HostSignalHandler) -> None:
        """Register a signal handler. Idempotent; same handler added
        twice is treated as one subscription."""
        if handler not in self._subscribers:
            self._subscribers.append(handler)

    def unsubscribe(self, handler: HostSignalHandler) -> None:
        if handler in self._subscribers:
            self._subscribers.remove(handler)

    def snapshot(self) -> HostSnapshot:
        """Return the latest snapshot. Triggers an on-demand probe if
        the poll loop hasn't produced any sample yet — that way callers
        get usable data even if ``start()`` was never invoked."""
        if self._history:
            return self._history[-1]
        return self._probe()

    def history(self, n: Optional[int] = None) -> List[HostSnapshot]:
        """Return the last ``n`` snapshots (most recent last). With
        ``n=None``, returns the whole ring buffer copy."""
        if n is None:
            return list(self._history)
        if n <= 0:
            return []
        return list(self._history)[-int(n):]

    async def start(self) -> None:
        """Spawn the background polling task. No-op if already running."""
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run())

    async def stop(self) -> None:
        """Cancel the polling task and await its teardown."""
        self._stopping = True
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass
        self._task = None

    async def poll_once(self) -> HostSnapshot:
        """Take exactly one sample, append it to the buffer, run signal
        dispatch. Exposed for tests that need deterministic timing —
        production code uses ``start`` instead."""
        snap = self._probe()
        self._history.append(snap)
        await self._dispatch_signals(snap)
        return snap

    # ---------------------------------------------------------- internal

    async def _run(self) -> None:
        while not self._stopping:
            try:
                await self.poll_once()
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug("HostTelemetry poll failed: %s", exc)
            try:
                await asyncio.sleep(self.interval_s)
            except asyncio.CancelledError:
                break

    async def _dispatch_signals(self, snap: HostSnapshot) -> None:
        """Translate a snapshot into zero-or-one ``HostSignal`` events.

        The single-event-per-poll rule keeps the subscriber stream
        manageable. Additionally, when ``heartbeat_s > 0`` we suppress
        signals whose ``(metric-set, severity)`` is unchanged from the
        last emission — re-emitting only every ``heartbeat_s`` seconds
        so a deployment whose LLM server pins RAM at 95% as steady
        state produces ONE warning line every 5 minutes, not 1Hz spam.

        State transitions (a metric clears, or severity escalates from
        warning to critical) always emit immediately regardless of the
        heartbeat window — those are operationally interesting.
        """
        reasons: List[str] = []
        metric_tokens: List[str] = []
        severity = "info"

        # CPU
        if math.isfinite(snap.cpu_percent) and snap.cpu_percent >= self.cpu_high:
            self._cpu_run += 1
            reasons.append(f"CPU {snap.cpu_percent:.0f}%")
            metric_tokens.append("cpu")
            if self._cpu_run >= self.sustain_samples:
                severity = _escalate(severity, "warning")
        else:
            self._cpu_run = 0

        # Memory pct
        if math.isfinite(snap.mem_percent) and snap.mem_percent >= self.mem_high:
            self._mem_run += 1
            reasons.append(f"RAM {snap.mem_percent:.0f}%")
            metric_tokens.append("ram")
            if self._mem_run >= self.sustain_samples:
                severity = _escalate(severity, "warning")
        else:
            self._mem_run = 0

        # Memory floor (critical)
        if math.isfinite(snap.mem_available_mb) and snap.mem_available_mb < self.mem_floor_mb:
            self._mem_floor_run += 1
            reasons.append(f"free<{int(self.mem_floor_mb)}MB ({snap.mem_available_mb:.0f}MB)")
            metric_tokens.append("ram_floor")
            severity = _escalate(severity, "critical")
        else:
            self._mem_floor_run = 0

        # Disk
        if math.isfinite(snap.disk_percent) and snap.disk_percent >= self.disk_high:
            self._disk_run += 1
            reasons.append(f"disk {snap.disk_percent:.0f}%")
            metric_tokens.append("disk")
            if self._disk_run >= self.sustain_samples:
                severity = _escalate(severity, "warning")
        else:
            self._disk_run = 0

        if not reasons:
            # Health restored — reset dedup state so the NEXT crossing
            # emits immediately (it's a state change worth surfacing).
            self._last_emit_key = None
            return

        emit_key = (frozenset(metric_tokens), severity)

        # Dedup: same (metric-set, severity) AND within heartbeat window
        # → suppress. heartbeat_s == 0 disables the dedup entirely.
        if (
            self.heartbeat_s > 0.0
            and self._last_emit_key == emit_key
            and (snap.ts - self._last_emit_ts) < self.heartbeat_s
        ):
            return

        signal = HostSignal(
            ts=snap.ts,
            severity=severity,
            reason="; ".join(reasons),
            snapshot=snap,
        )
        self._last_emit_key = emit_key
        self._last_emit_ts = snap.ts
        await self._fanout(signal)

    async def _fanout(self, signal: HostSignal) -> None:
        for handler in list(self._subscribers):
            try:
                result = handler(signal)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug("HostSignal handler failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────
# Default probe
# ──────────────────────────────────────────────────────────────────────

def _psutil_probe() -> HostSnapshot:
    """Default snapshot source. Returns a HostSnapshot with NaN fields
    when psutil is unavailable or a particular reading fails."""
    ts = time.time()
    cpu = math.nan
    mem_pct = math.nan
    mem_avail = math.nan
    disk_pct = math.nan
    proc_count = -1
    try:
        import psutil
    except Exception:
        return HostSnapshot(ts, cpu, mem_pct, mem_avail, disk_pct, proc_count)
    try:
        cpu = float(psutil.cpu_percent(interval=None))
    except Exception:
        pass
    try:
        m = psutil.virtual_memory()
        mem_pct = float(m.percent)
        mem_avail = float(m.available) / (1024 * 1024)
    except Exception:
        pass
    try:
        d = psutil.disk_usage("/")
        disk_pct = float(d.percent)
    except Exception:
        pass
    try:
        proc_count = int(len(psutil.pids()))
    except Exception:
        pass
    return HostSnapshot(ts, cpu, mem_pct, mem_avail, disk_pct, proc_count)


def _escalate(current: str, candidate: str) -> str:
    """Return the higher of two severity labels."""
    order = {"info": 0, "warning": 1, "critical": 2}
    return candidate if order.get(candidate, 0) > order.get(current, 0) else current
