"""Unit tests for ghost_agent.core.triggers."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from ghost_agent.core.triggers import (
    ExecutionAnomaly,
    LoopDetected,
    RepetitionCounter,
    ReplanBridge,
    ResourceExhausted,
    ToolRuntimeBudget,
    TriggerBus,
    anomaly_event,
    loop_event,
    resource_event,
)


# ──────────────────────────────────────────────────────────────────────
# TriggerBus
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_subscribe_and_publish_kind():
    bus = TriggerBus()
    received: list = []
    bus.subscribe("loop", lambda ev: received.append(ev))
    await bus.publish(loop_event("test"))
    assert len(received) == 1


@pytest.mark.asyncio
async def test_wildcard_subscriber_receives_all():
    bus = TriggerBus()
    received: list = []
    bus.subscribe("*", lambda ev: received.append(ev))
    await bus.publish(loop_event("a"))
    await bus.publish(resource_event("b", metric="ram", observed=90, threshold=85))
    await bus.publish(anomaly_event("c", tool_name="t", duration_s=10, budget_s=3))
    assert len(received) == 3


@pytest.mark.asyncio
async def test_async_handler_is_awaited():
    bus = TriggerBus()
    received: list = []

    async def handler(ev):
        await asyncio.sleep(0.001)
        received.append(ev)

    bus.subscribe("*", handler)
    await bus.publish(loop_event("test"))
    assert len(received) == 1


@pytest.mark.asyncio
async def test_handler_error_isolated():
    bus = TriggerBus()
    received: list = []

    def bad(ev):
        raise RuntimeError("boom")

    bus.subscribe("*", bad)
    bus.subscribe("*", lambda ev: received.append(ev))
    await bus.publish(loop_event("test"))
    # Good handler still ran despite bad one raising
    assert len(received) == 1


@pytest.mark.asyncio
async def test_history_is_bounded():
    bus = TriggerBus()
    bus._history_cap = 5
    for i in range(10):
        await bus.publish(loop_event(f"x{i}"))
    assert len(bus.history()) == 5
    # Most recent retained
    assert bus.history()[-1].reason == "x9"


@pytest.mark.asyncio
async def test_history_by_kind():
    bus = TriggerBus()
    await bus.publish(loop_event("a"))
    await bus.publish(resource_event("b", metric="ram", observed=90, threshold=85))
    assert len(bus.history("loop")) == 1
    assert len(bus.history("resource")) == 1


@pytest.mark.asyncio
async def test_unsubscribe():
    bus = TriggerBus()
    received: list = []

    def h(ev):
        received.append(ev)

    bus.subscribe("loop", h)
    bus.unsubscribe("loop", h)
    await bus.publish(loop_event("test"))
    assert received == []


# ──────────────────────────────────────────────────────────────────────
# ToolRuntimeBudget
# ──────────────────────────────────────────────────────────────────────

class TestToolRuntimeBudget:
    def test_cold_start_no_budget(self):
        b = ToolRuntimeBudget(min_samples=5)
        for _ in range(3):
            b.record("x", 1.0)
        assert b.budget("x") is None
        assert not b.is_anomalous("x", 999.0)

    def test_p95_after_warmup(self):
        b = ToolRuntimeBudget(min_samples=5, multiplier=2.0)
        for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            b.record("x", float(v))
        budget = b.budget("x")
        # p95 of [1..10] ~ 10; multiplier 2 → 20
        assert budget is not None
        assert 15.0 <= budget <= 25.0

    def test_is_anomalous(self):
        b = ToolRuntimeBudget(min_samples=5, multiplier=2.0)
        for _ in range(10):
            b.record("x", 1.0)
        assert b.is_anomalous("x", 100.0) is True
        assert b.is_anomalous("x", 1.5) is False

    def test_window_bounded(self):
        b = ToolRuntimeBudget(window=4, min_samples=3)
        for v in range(10):
            b.record("x", float(v))
        # Last 4 retained: [6,7,8,9]
        assert len(b._samples["x"]) == 4

    def test_unknown_tool_returns_none(self):
        b = ToolRuntimeBudget()
        assert b.budget("never") is None


# ──────────────────────────────────────────────────────────────────────
# RepetitionCounter
# ──────────────────────────────────────────────────────────────────────

class TestRepetitionCounter:
    def test_streak_grows(self):
        c = RepetitionCounter(threshold=3)
        assert c.observe("x") == 1
        assert c.observe("x") == 2
        assert c.observe("x") == 3
        assert c.tripped()

    def test_different_key_resets(self):
        c = RepetitionCounter(threshold=3)
        c.observe("x"); c.observe("x")
        assert c.observe("y") == 1
        assert not c.tripped()

    def test_reset(self):
        c = RepetitionCounter(threshold=2)
        c.observe("x"); c.observe("x")
        assert c.tripped()
        c.reset()
        assert not c.tripped()
        assert c.observe("x") == 1


# ──────────────────────────────────────────────────────────────────────
# ReplanBridge
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_replan_bridge_calls_request_revision():
    bus = TriggerBus()

    class FakePlan:
        def __init__(self):
            self.calls: list = []

        def request_revision(self, task_id, reason):
            self.calls.append((task_id, reason))
            return True

    plan = FakePlan()
    bridge = ReplanBridge(
        bus, plan_getter=lambda: plan,
        current_task_getter=lambda: "t123",
    )
    bridge.attach()
    await bus.publish(loop_event("looped", severity="warning"))
    assert plan.calls == [("t123", "loop/warning: looped")]


@pytest.mark.asyncio
async def test_replan_bridge_silent_when_no_plan():
    bus = TriggerBus()
    bridge = ReplanBridge(bus, plan_getter=lambda: None,
                          current_task_getter=lambda: None)
    bridge.attach()
    await bus.publish(loop_event("looped"))
    assert bridge.revisions[-1]["action"] == "noop:no_plan"


@pytest.mark.asyncio
async def test_replan_bridge_ignores_info_severity():
    bus = TriggerBus()

    class FakePlan:
        def __init__(self):
            self.calls: list = []

        def request_revision(self, task_id, reason):
            self.calls.append((task_id, reason))
            return True

    plan = FakePlan()
    bridge = ReplanBridge(bus, plan_getter=lambda: plan,
                          current_task_getter=lambda: "t1")
    bridge.attach()
    await bus.publish(loop_event("ignored", severity="info"))
    assert plan.calls == []
    assert bridge.revisions[-1]["action"] == "noop:info"


@pytest.mark.asyncio
async def test_replan_bridge_records_revision_rejected():
    bus = TriggerBus()

    class FakePlan:
        def request_revision(self, task_id, reason):
            return False  # max revisions reached

    bridge = ReplanBridge(bus, plan_getter=lambda: FakePlan(),
                          current_task_getter=lambda: "t1")
    bridge.attach()
    await bus.publish(loop_event("x", severity="warning"))
    assert bridge.revisions[-1]["action"] == "revision_rejected"


@pytest.mark.asyncio
async def test_replan_bridge_handles_tree_indirection():
    """Some callers hold a TaskTree (not a ProjectPlan), which exposes
    request_revision via .tree. Bridge should work with both shapes."""
    bus = TriggerBus()

    class Tree:
        def __init__(self):
            self.calls = []

        def request_revision(self, task_id, reason):
            self.calls.append((task_id, reason))
            return True

    class WrappedPlan:
        def __init__(self):
            self.tree = Tree()

    plan = WrappedPlan()
    bridge = ReplanBridge(bus, plan_getter=lambda: plan,
                          current_task_getter=lambda: "t99")
    bridge.attach()
    await bus.publish(resource_event("ram blew up", metric="ram",
                                     observed=99, threshold=85,
                                     severity="critical"))
    assert plan.tree.calls == [("t99", "resource/critical: ram blew up")]


# ──────────────────────────────────────────────────────────────────────
# Event factories
# ──────────────────────────────────────────────────────────────────────

def test_factories_set_defaults():
    e1 = loop_event("x")
    assert isinstance(e1, LoopDetected)
    assert e1.kind == "loop"
    e2 = resource_event("y", metric="ram", observed=90, threshold=85)
    assert isinstance(e2, ResourceExhausted)
    assert e2.kind == "resource"
    e3 = anomaly_event("z", tool_name="t", duration_s=10, budget_s=3)
    assert isinstance(e3, ExecutionAnomaly)
    assert e3.kind == "anomaly"
