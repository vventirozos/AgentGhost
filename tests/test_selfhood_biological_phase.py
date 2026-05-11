"""Integration tests for biological watchdog phase 2.8 (selfhood
narrative consolidation). Mirrors the shape of
test_reflection_biological_tick.py: drive _biological_tick directly
with a mock context.
"""

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import GhostAgent


def _make_ctx(*, idle_secs: float, self_model=None):
    """Build a mock GhostContext shaped to trigger phase 2.8.

    Only populates what _biological_tick reads. Other phases'
    dependencies are nulled out so they short-circuit cleanly."""
    ctx = MagicMock()
    ctx.memory_system = MagicMock()
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    # Phase 1: journal None
    ctx.journal = None
    # Phase 2: empty dream-eligible auto memories
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=idle_secs)
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    ctx.args.prm_train_cooldown = None
    ctx.args.self_narrative_cooldown = None
    ctx.frontier_tracker = None

    # Disable other phases by missing wirings:
    ctx.reflector = None
    ctx.trajectory_collector = None
    ctx.prm_scorer = None  # type-check filters this out in phase 2.7

    ctx.self_model = self_model
    return ctx


async def _tick(ctx):
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    await agent._biological_tick()
    return agent


async def test_narrative_phase_fires_when_wired_and_cooldown_elapsed():
    sm = MagicMock()
    sm.enabled = True
    sm.consolidate_narrative = AsyncMock(return_value="A diary entry.")

    ctx = _make_ctx(idle_secs=1200, self_model=sm)
    agent = await _tick(ctx)
    sm.consolidate_narrative.assert_awaited_once()
    assert agent._last_narrative_at > datetime.datetime.min


async def test_narrative_phase_skipped_when_self_model_missing():
    ctx = _make_ctx(idle_secs=1200, self_model=None)
    # Must not raise
    await _tick(ctx)


async def test_narrative_phase_skipped_when_disabled():
    sm = MagicMock()
    sm.enabled = False
    sm.consolidate_narrative = AsyncMock(return_value="should not be called")
    ctx = _make_ctx(idle_secs=1200, self_model=sm)
    await _tick(ctx)
    sm.consolidate_narrative.assert_not_called()


async def test_narrative_phase_does_not_fire_below_15min_idle():
    sm = MagicMock()
    sm.enabled = True
    sm.consolidate_narrative = AsyncMock()
    ctx = _make_ctx(idle_secs=400, self_model=sm)
    await _tick(ctx)
    sm.consolidate_narrative.assert_not_called()


async def test_narrative_phase_does_not_fire_over_60min_idle():
    """Phase 2.8 is scoped to 900-3600s, same as reflection. Beyond
    that, phase 3 (self-play) takes over."""
    sm = MagicMock()
    sm.enabled = True
    sm.consolidate_narrative = AsyncMock()
    ctx = _make_ctx(idle_secs=5000, self_model=sm)
    await _tick(ctx)
    sm.consolidate_narrative.assert_not_called()


async def test_narrative_phase_respects_cooldown():
    sm = MagicMock()
    sm.enabled = True
    sm.consolidate_narrative = AsyncMock(return_value="x")
    ctx = _make_ctx(idle_secs=1200, self_model=sm)
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    # Anchor: just fired.
    agent._last_narrative_at = datetime.datetime.now()
    await agent._biological_tick()
    sm.consolidate_narrative.assert_not_called()


async def test_narrative_phase_anchor_advances_even_on_exception():
    """The anchor-before-await invariant: a crash during the LLM call
    must NOT leave the cooldown un-advanced (otherwise the failing
    consolidation re-fires every 60-second tick)."""
    sm = MagicMock()
    sm.enabled = True
    sm.consolidate_narrative = AsyncMock(side_effect=RuntimeError("boom"))
    ctx = _make_ctx(idle_secs=1200, self_model=sm)
    agent = await _tick(ctx)
    # Anchor advanced despite the exception.
    assert agent._last_narrative_at > datetime.datetime.min


async def test_narrative_phase_honours_cooldown_override():
    """Custom --self-narrative-cooldown is respected."""
    sm = MagicMock()
    sm.enabled = True
    sm.consolidate_narrative = AsyncMock(return_value="x")
    ctx = _make_ctx(idle_secs=1200, self_model=sm)
    # Set the override to 10 seconds — should fire even if the anchor
    # is 30 seconds ago.
    ctx.args.self_narrative_cooldown = 10
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    agent._last_narrative_at = datetime.datetime.now() - datetime.timedelta(seconds=30)
    await agent._biological_tick()
    sm.consolidate_narrative.assert_awaited_once()


async def test_narrative_phase_does_not_reset_user_activity_clock():
    """Internal phases must NOT reset ctx.last_activity_time — that
    clock is the user's, and phase 3 (self-play) gates on it."""
    sm = MagicMock()
    sm.enabled = True
    sm.consolidate_narrative = AsyncMock(return_value="diary text")
    ctx = _make_ctx(idle_secs=1200, self_model=sm)
    original_activity = ctx.last_activity_time
    await _tick(ctx)
    assert ctx.last_activity_time == original_activity
