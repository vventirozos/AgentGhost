"""Tests for the log-noise polish pass:

  1. Telemetry CLI thresholds plumbed through the bundle.
  2. HostTelemetry dedup — steady-state crossings emit ONE signal, not 1Hz.
  3. ReplanBridge noop:* actions DO NOT emit log lines.
  4. metacog_log assigns a distinct icon per subsystem (not BRAIN_THINK).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.core.metacog import MetacogBundle
from ghost_agent.core.metacog_log import (
    _DEFAULT_ICON,
    _SUBSYSTEM_ICONS,
    Subsystem,
    emit,
)
from ghost_agent.core.triggers import (
    ReplanBridge,
    TriggerBus,
    loop_event,
    resource_event,
)
from ghost_agent.utils.telemetry import (
    HostSnapshot,
    HostTelemetry,
)


# ──────────────────────────────────────────────────────────────────────
# CLI threshold flag plumbing
# ──────────────────────────────────────────────────────────────────────

def _args(**overrides):
    a = argparse.Namespace(
        enable_metacog=True,
        metacog_confidence_threshold=0.55,
        metacog_disable_logprobs=False,
        metacog_disable_arbiter=False,
        metacog_cpu_high=85.0,
        metacog_mem_high=85.0,
        metacog_mem_floor_mb=800.0,
        metacog_disk_high=90.0,
        metacog_host_heartbeat_s=300.0,
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


@pytest.fixture
def fake_context(tmp_path):
    ctx = MagicMock()
    ctx.memory_dir = tmp_path
    ctx.args = MagicMock()
    ctx.args.model = "test"
    ctx.llm_client = MagicMock()
    if hasattr(ctx.llm_client, "get_embeddings"):
        del ctx.llm_client.get_embeddings
    ctx.project_store = None
    ctx.current_project_id = None
    return ctx


class TestThresholdPlumbing:
    def test_defaults_plumbed(self, fake_context):
        bundle = MetacogBundle.from_args(fake_context, _args())
        assert bundle.telemetry.cpu_high == 85.0
        assert bundle.telemetry.mem_high == 85.0
        assert bundle.telemetry.mem_floor_mb == 800.0
        assert bundle.telemetry.disk_high == 90.0
        assert bundle.telemetry.heartbeat_s == 300.0

    def test_overrides_plumbed(self, fake_context):
        bundle = MetacogBundle.from_args(fake_context, _args(
            metacog_mem_high=99.0,
            metacog_cpu_high=95.0,
            metacog_mem_floor_mb=500.0,
            metacog_disk_high=98.0,
            metacog_host_heartbeat_s=600.0,
        ))
        assert bundle.telemetry.mem_high == 99.0
        assert bundle.telemetry.cpu_high == 95.0
        assert bundle.telemetry.mem_floor_mb == 500.0
        assert bundle.telemetry.disk_high == 98.0
        assert bundle.telemetry.heartbeat_s == 600.0

    def test_bundle_survives_missing_threshold_attrs(self, fake_context):
        """Tests that don't set the new metacog_* args (older fixtures)
        must still construct the bundle without crashing — defaults
        fall through to HostTelemetry class defaults."""
        args = argparse.Namespace(
            enable_metacog=True,
            metacog_confidence_threshold=0.55,
            metacog_disable_logprobs=False,
            metacog_disable_arbiter=False,
        )
        bundle = MetacogBundle.from_args(fake_context, args)
        # Class defaults
        assert bundle.telemetry.cpu_high == HostTelemetry.DEFAULT_CPU_HIGH
        assert bundle.telemetry.mem_high == HostTelemetry.DEFAULT_MEM_HIGH


# ──────────────────────────────────────────────────────────────────────
# HostTelemetry dedup
# ──────────────────────────────────────────────────────────────────────

def _snap(*, cpu=10.0, mem=30.0, avail=4000.0, disk=20.0, procs=100, ts=None):
    return HostSnapshot(
        ts=ts if ts is not None else time.time(),
        cpu_percent=cpu, mem_percent=mem,
        mem_available_mb=avail, disk_percent=disk,
        proc_count=procs,
    )


async def _drive(telemetry: HostTelemetry, snapshots) -> list:
    received: list = []

    async def handler(sig):
        received.append(sig)

    telemetry.subscribe(handler)
    for snap in snapshots:
        telemetry._probe = lambda s=snap: s
        await telemetry.poll_once()
    return received


class TestHostTelemetryDedup:
    @pytest.mark.asyncio
    async def test_steady_state_emits_once(self):
        """The user's actual scenario: RAM pinned at 96% by the LLM
        process. Pre-dedup the bridge got hammered with one signal per
        second. With dedup we should see exactly one warning."""
        telemetry = HostTelemetry(sustain_samples=1, heartbeat_s=300.0)
        snaps = [_snap(mem=96.0, ts=1000.0 + i) for i in range(10)]
        signals = await _drive(telemetry, snaps)
        assert len(signals) == 1, f"expected 1 dedup'd signal, got {len(signals)}"

    @pytest.mark.asyncio
    async def test_severity_change_re_emits(self):
        """RAM 96% (warning) → free<800MB (critical) is a state change,
        not a dedup. Both should emit."""
        telemetry = HostTelemetry(sustain_samples=1, heartbeat_s=300.0)
        snaps = [
            _snap(mem=96.0, avail=4000.0, ts=1000.0),  # warning
            _snap(mem=96.0, avail=4000.0, ts=1001.0),  # dedup'd
            _snap(mem=96.0, avail=400.0,  ts=1002.0),  # CRITICAL — new state
        ]
        signals = await _drive(telemetry, snaps)
        assert len(signals) == 2
        assert signals[0].severity == "warning"
        assert signals[1].severity == "critical"

    @pytest.mark.asyncio
    async def test_heartbeat_re_emits_after_window(self):
        """Even if state hasn't changed, re-emit every heartbeat_s
        seconds so the operator sees a periodic 'still degraded' line."""
        telemetry = HostTelemetry(sustain_samples=1, heartbeat_s=60.0)
        snaps = [
            _snap(mem=96.0, ts=1000.0),
            _snap(mem=96.0, ts=1030.0),  # 30s — within heartbeat
            _snap(mem=96.0, ts=1090.0),  # 90s — heartbeat expired
        ]
        signals = await _drive(telemetry, snaps)
        assert len(signals) == 2

    @pytest.mark.asyncio
    async def test_healthy_then_unhealthy_re_emits(self):
        """Crossing back into a threshold after a healthy period is a
        state change — must emit even though severity might match."""
        telemetry = HostTelemetry(sustain_samples=1, heartbeat_s=300.0)
        snaps = [
            _snap(mem=96.0, ts=1000.0),  # warning
            _snap(mem=30.0, ts=1001.0),  # healthy — resets dedup
            _snap(mem=96.0, ts=1002.0),  # warning again — NEW signal
        ]
        signals = await _drive(telemetry, snaps)
        assert len(signals) == 2

    @pytest.mark.asyncio
    async def test_heartbeat_zero_disables_dedup(self):
        """heartbeat_s=0 should restore the old behaviour: every poll
        that crosses thresholds emits. Useful as a debugging knob."""
        telemetry = HostTelemetry(sustain_samples=1, heartbeat_s=0.0)
        snaps = [_snap(mem=96.0, ts=1000.0 + i) for i in range(5)]
        signals = await _drive(telemetry, snaps)
        assert len(signals) == 5

    @pytest.mark.asyncio
    async def test_changing_metric_set_re_emits(self):
        """RAM alone → RAM+CPU is a new metric tuple even at same
        severity. Operationally interesting — must emit."""
        telemetry = HostTelemetry(sustain_samples=1, heartbeat_s=300.0)
        snaps = [
            _snap(mem=96.0, cpu=10.0, ts=1000.0),       # ram only
            _snap(mem=96.0, cpu=92.0, ts=1001.0),       # ram + cpu
        ]
        signals = await _drive(telemetry, snaps)
        assert len(signals) == 2


# ──────────────────────────────────────────────────────────────────────
# ReplanBridge noop suppression
# ──────────────────────────────────────────────────────────────────────

class TestReplanNoopSuppression:
    @pytest.mark.asyncio
    async def test_no_plan_does_not_emit_log(self):
        """The user's actual log spam: every host signal hit the bridge
        with no plan attached, producing one `metacog replan
        action=noop:no_plan ...` line per signal. Must be silent."""
        bus = TriggerBus()
        bridge = ReplanBridge(
            bus, plan_getter=lambda: None,
            current_task_getter=lambda: None,
        )
        bridge.attach()
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            await bus.publish(resource_event(
                "RAM 96%", metric="ram", observed=96.0,
                threshold=85.0, severity="warning",
            ))
        replan_calls = [c for c in pl.call_args_list
                        if c.args and c.args[0] == "Metacog Replan"]
        assert replan_calls == [], (
            f"expected silent noop, got {len(replan_calls)} log calls"
        )
        # Audit still captured though
        assert bridge.revisions[-1]["action"] == "noop:no_plan"

    @pytest.mark.asyncio
    async def test_info_severity_does_not_emit_log(self):
        bus = TriggerBus()

        class Plan:
            def request_revision(self, t, r):
                return True

        bridge = ReplanBridge(
            bus, plan_getter=lambda: Plan(),
            current_task_getter=lambda: "t1",
        )
        bridge.attach()
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            await bus.publish(loop_event("noisy", severity="info"))
        replan_calls = [c for c in pl.call_args_list
                        if c.args and c.args[0] == "Metacog Replan"]
        assert replan_calls == []
        assert bridge.revisions[-1]["action"] == "noop:info"

    @pytest.mark.asyncio
    async def test_revised_still_emits(self):
        """The suppression must not break the actionable case."""
        bus = TriggerBus()

        class Plan:
            def request_revision(self, t, r):
                return True

        bridge = ReplanBridge(
            bus, plan_getter=lambda: Plan(),
            current_task_getter=lambda: "t1",
        )
        bridge.attach()
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            await bus.publish(loop_event("looped", severity="warning"))
        replan_calls = [c for c in pl.call_args_list
                        if c.args and c.args[0] == "Metacog Replan"]
        assert len(replan_calls) == 1
        assert "action=revised" in replan_calls[0].args[1]


# ──────────────────────────────────────────────────────────────────────
# Per-subsystem distinct icons
# ──────────────────────────────────────────────────────────────────────

class TestSubsystemIcons:
    def test_every_subsystem_has_distinct_icon(self):
        # All seven subsystems are mapped
        subsystems = {
            Subsystem.BOOT, Subsystem.SUMMARY, Subsystem.CONF,
            Subsystem.ARBITER, Subsystem.VALID, Subsystem.HOST,
            Subsystem.REPLAN, Subsystem.GATE,
        }
        assert set(_SUBSYSTEM_ICONS.keys()) == subsystems
        # Icons distinct from each other
        icons = list(_SUBSYSTEM_ICONS.values())
        assert len(set(icons)) == len(icons), (
            f"duplicate icons in subsystem map: {icons}"
        )

    def test_metacog_icons_distinct_from_brain_think(self):
        """The whole point of this pass — operators couldn't tell
        metacog lines apart from agent brain-thinking lines because
        both used 💭."""
        from ghost_agent.utils.logging import Icons
        for sub, icon in _SUBSYSTEM_ICONS.items():
            assert icon != Icons.BRAIN_THINK, (
                f"subsystem {sub} still uses BRAIN_THINK 💭"
            )
        assert _DEFAULT_ICON != Icons.BRAIN_THINK

    def test_emit_uses_subsystem_icon(self):
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            emit(Subsystem.HOST, severity="warning")
        assert pl.called
        kwargs = pl.call_args.kwargs
        assert kwargs["icon"] == _SUBSYSTEM_ICONS[Subsystem.HOST]

    def test_emit_caller_override_respected(self):
        """Caller explicitly passing icon= overrides the subsystem map."""
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            emit(Subsystem.HOST, icon="🚨", severity="critical")
        assert pl.call_args.kwargs["icon"] == "🚨"

    def test_unknown_subsystem_falls_back(self):
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            emit("brand_new_subsystem", foo="bar")
        assert pl.call_args.kwargs["icon"] == _DEFAULT_ICON
