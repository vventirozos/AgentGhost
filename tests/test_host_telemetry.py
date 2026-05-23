"""Unit tests for ghost_agent.utils.telemetry — host telemetry poller."""

import asyncio
import math
import time

import pytest

from ghost_agent.utils.telemetry import (
    HostSnapshot,
    HostSignal,
    HostTelemetry,
    _escalate,
    _psutil_probe,
)


# ──────────────────────────────────────────────────────────────────────
# HostSnapshot.healthy
# ──────────────────────────────────────────────────────────────────────

def _snap(*, cpu=10.0, mem=30.0, avail=4000.0, disk=20.0, procs=100, ts=None):
    return HostSnapshot(
        ts=ts if ts is not None else time.time(),
        cpu_percent=cpu, mem_percent=mem,
        mem_available_mb=avail, disk_percent=disk,
        proc_count=procs,
    )


def test_snapshot_healthy_when_below_thresholds():
    assert _snap().healthy is True


def test_snapshot_unhealthy_on_high_cpu():
    assert _snap(cpu=90.0).healthy is False


def test_snapshot_unhealthy_on_high_mem():
    assert _snap(mem=90.0).healthy is False


def test_snapshot_unhealthy_on_low_free_mem():
    assert _snap(avail=500.0).healthy is False


def test_snapshot_ignores_nan_axes():
    # NaN means "psutil refused"; we must NOT treat that as a failure
    assert _snap(cpu=math.nan, mem=math.nan, avail=math.nan).healthy is True


# ──────────────────────────────────────────────────────────────────────
# Threshold dispatch
# ──────────────────────────────────────────────────────────────────────

async def _collect_signals(telemetry: HostTelemetry, snapshots):
    """Drive the poller through a scripted sequence of snapshots and
    return every signal it emits."""
    received: list = []

    async def handler(sig: HostSignal):
        received.append(sig)

    telemetry.subscribe(handler)
    for snap in snapshots:
        # poke the probe to return our scripted snapshot
        telemetry._probe = lambda s=snap: s  # noqa: E731
        await telemetry.poll_once()
    return received


@pytest.mark.asyncio
async def test_no_signal_on_healthy_snapshot():
    telemetry = HostTelemetry(sustain_samples=1)
    signals = await _collect_signals(telemetry, [_snap()])
    assert signals == []


@pytest.mark.asyncio
async def test_signal_fires_when_cpu_high():
    telemetry = HostTelemetry(sustain_samples=1)
    signals = await _collect_signals(telemetry, [_snap(cpu=92.0)])
    assert len(signals) == 1
    assert "CPU" in signals[0].reason


@pytest.mark.asyncio
async def test_severity_escalates_to_critical_on_floor():
    telemetry = HostTelemetry(sustain_samples=1)
    signals = await _collect_signals(telemetry, [_snap(avail=300.0)])
    assert signals[0].severity == "critical"


@pytest.mark.asyncio
async def test_severity_warning_only_on_sustain():
    """Three consecutive crossings still escalate severity from info to
    warning. After the dedup uplift (2026-05-23) the run of identical
    info signals is suppressed — only the state change to ``warning``
    emits a second signal. The previous shape (3 signals, info/info/warn)
    produced 1Hz log spam on hosts where the LLM server pins the
    threshold as steady state."""
    telemetry = HostTelemetry(sustain_samples=3, heartbeat_s=300.0)
    snaps = [_snap(cpu=92.0), _snap(cpu=92.0), _snap(cpu=92.0)]
    signals = await _collect_signals(telemetry, snaps)
    assert len(signals) == 2
    assert signals[0].severity == "info"
    assert signals[-1].severity == "warning"


@pytest.mark.asyncio
async def test_run_counter_resets_on_healthy_sample():
    """Healthy sample resets the per-axis sustain counter AND clears
    the dedup key — so the next unhealthy sample emits afresh as
    ``info`` rather than being suppressed (the dedup is on the LAST
    emitted state, which gets cleared by the healthy gap)."""
    telemetry = HostTelemetry(sustain_samples=3, heartbeat_s=300.0)
    snaps = [_snap(cpu=92.0), _snap(cpu=92.0), _snap(cpu=20.0),
             _snap(cpu=92.0)]
    signals = await _collect_signals(telemetry, snaps)
    severities = [s.severity for s in signals]
    # No 'warning' — the run counter reset on the healthy sample so
    # the 4th unhealthy sample is just streak=1, still info.
    assert "warning" not in severities
    # 2 emitted info signals (first crossing, and re-crossing after
    # the healthy gap). The duplicate-92.0 in slot 2 is dedup'd.
    assert len(signals) == 2


# ──────────────────────────────────────────────────────────────────────
# Ring buffer
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_bounded():
    telemetry = HostTelemetry(history=4, sustain_samples=1)
    telemetry._probe = lambda: _snap()
    for _ in range(10):
        await telemetry.poll_once()
    assert len(telemetry.history()) == 4


@pytest.mark.asyncio
async def test_snapshot_on_demand_without_start():
    telemetry = HostTelemetry()
    telemetry._probe = lambda: _snap(cpu=42.0)
    snap = telemetry.snapshot()
    assert snap.cpu_percent == 42.0


# ──────────────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_stop_clean():
    telemetry = HostTelemetry(interval_s=0.1, sustain_samples=1)
    telemetry._probe = lambda: _snap()
    await telemetry.start()
    await asyncio.sleep(0.25)
    await telemetry.stop()
    # At least one sample should have landed in the buffer
    assert len(telemetry.history()) >= 1


@pytest.mark.asyncio
async def test_handler_error_does_not_break_poll():
    telemetry = HostTelemetry(sustain_samples=1)

    def bad_handler(sig):
        raise RuntimeError("simulated")

    telemetry.subscribe(bad_handler)
    telemetry._probe = lambda: _snap(cpu=99.0)
    # Should not raise
    await telemetry.poll_once()


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def test_escalate_order():
    assert _escalate("info", "warning") == "warning"
    assert _escalate("warning", "info") == "warning"
    assert _escalate("warning", "critical") == "critical"
    assert _escalate("critical", "warning") == "critical"


def test_psutil_probe_returns_finite_or_nan():
    """The default probe must never raise; numeric fields must be
    finite or NaN (so callers can use math.isfinite as a sentinel)."""
    snap = _psutil_probe()
    for v in (snap.cpu_percent, snap.mem_percent,
              snap.mem_available_mb, snap.disk_percent):
        assert math.isnan(v) or math.isfinite(v)
