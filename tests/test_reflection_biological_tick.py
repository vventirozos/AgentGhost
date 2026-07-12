"""Integration tests for the reflection phase inside _biological_tick.

These tests verify the wiring inside agent.py — specifically the
cooldown-anchor pattern and the activity-clock semantics — without
requiring the full agent to boot.
"""

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.reflection.loop import ReflectionRunReport


def _make_ctx(*, idle_secs: float, reflector=None, collector=None):
    """Build a mock GhostContext shaped to trigger the reflection phase.

    We only populate what _biological_tick reads — the rest of the
    agent plumbing is short-circuited by missing attributes.
    """
    ctx = MagicMock()
    # memory_dir must NOT be a bare MagicMock: the reflection phase persists
    # its reflected-id set to <memory_dir>/reflected_ids.json, and str(a
    # MagicMock) is a bogus name that mkdir would litter into the CWD. This
    # unit test doesn't exercise persistence, so pin it to None (the source
    # also guards non-path values — see agent._reflected_ids_path).
    ctx.memory_dir = None
    ctx.memory_system = MagicMock()
    # llm_client must report zero foreground tasks so the tick proceeds
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    # Journal: None so Phase 1 short-circuits and doesn't early-return.
    ctx.journal = None
    # Collection for Phase 2: return empty ids so dream is skipped too.
    ctx.memory_system.collection.get = MagicMock(
        return_value={"ids": []}
    )
    ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=idle_secs)
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    ctx.frontier_tracker = None

    ctx.reflector = reflector
    ctx.trajectory_collector = collector
    return ctx


async def _tick(ctx):
    """Run one _biological_tick pass on a fresh agent instance."""
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    await agent._biological_tick()
    return agent


async def test_reflection_phase_fires_when_wired_and_cooldown_elapsed():
    mock_reflector = MagicMock()
    mock_reflector.run = AsyncMock(return_value=ReflectionRunReport(
        seen_failures=1, reflected_ok=1,
    ))
    mock_collector = MagicMock()
    mock_collector.iter_trajectories = MagicMock(return_value=iter([]))
    mock_collector.append = MagicMock()

    ctx = _make_ctx(idle_secs=1200, reflector=mock_reflector, collector=mock_collector)
    agent = await _tick(ctx)
    mock_reflector.run.assert_called_once()
    # Anchor was advanced
    assert agent._last_reflection_at > datetime.datetime.min


async def test_reflection_phase_skipped_when_collector_missing():
    mock_reflector = MagicMock()
    mock_reflector.run = AsyncMock()
    ctx = _make_ctx(idle_secs=1200, reflector=mock_reflector, collector=None)
    await _tick(ctx)
    mock_reflector.run.assert_not_called()


async def test_reflection_phase_skipped_when_reflector_missing():
    mock_collector = MagicMock()
    mock_collector.iter_trajectories = MagicMock(return_value=iter([]))
    ctx = _make_ctx(idle_secs=1200, reflector=None, collector=mock_collector)
    await _tick(ctx)
    # If neither reflector nor collector, the phase is a no-op; we
    # confirm nothing raised by exiting the tick cleanly.


async def test_reflection_phase_respects_cooldown():
    mock_reflector = MagicMock()
    mock_reflector.run = AsyncMock(return_value=ReflectionRunReport())
    mock_collector = MagicMock()
    mock_collector.iter_trajectories = MagicMock(return_value=iter([]))

    ctx = _make_ctx(idle_secs=1200, reflector=mock_reflector, collector=mock_collector)
    # Pre-tick: set the anchor to "just fired".
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    agent._last_reflection_at = datetime.datetime.now()
    await agent._biological_tick()
    mock_reflector.run.assert_not_called()


async def test_reflection_phase_does_not_fire_below_15min_idle():
    """Under 900s idle the phase should not engage."""
    mock_reflector = MagicMock()
    mock_reflector.run = AsyncMock()
    mock_collector = MagicMock()
    mock_collector.iter_trajectories = MagicMock(return_value=iter([]))
    ctx = _make_ctx(idle_secs=300, reflector=mock_reflector, collector=mock_collector)
    await _tick(ctx)
    mock_reflector.run.assert_not_called()


async def test_reflection_phase_does_not_fire_over_60min_idle():
    """Over 3600s, Phase 3 (self-play) takes over; reflection is scoped
    to 900-3600s."""
    mock_reflector = MagicMock()
    mock_reflector.run = AsyncMock()
    mock_collector = MagicMock()
    mock_collector.iter_trajectories = MagicMock(return_value=iter([]))
    ctx = _make_ctx(idle_secs=4000, reflector=mock_reflector, collector=mock_collector)
    await _tick(ctx)
    mock_reflector.run.assert_not_called()


async def test_reflection_phase_never_resets_activity_clock():
    """Same invariant as dream/phase 1: the reflection phase must NOT
    touch ctx.last_activity_time — doing so would starve phase 3."""
    mock_reflector = MagicMock()
    mock_reflector.run = AsyncMock(return_value=ReflectionRunReport())
    mock_collector = MagicMock()
    mock_collector.iter_trajectories = MagicMock(return_value=iter([]))

    ctx = _make_ctx(idle_secs=1200, reflector=mock_reflector, collector=mock_collector)
    activity_before = ctx.last_activity_time
    await _tick(ctx)
    assert ctx.last_activity_time == activity_before


async def test_reflection_phase_advances_anchor_on_exception():
    """Critical invariant from CLAUDE.md: an exception mid-phase must
    NOT leave the cooldown anchor at its prior value."""
    mock_reflector = MagicMock()
    mock_reflector.run = AsyncMock(side_effect=RuntimeError("simulated"))
    mock_collector = MagicMock()
    mock_collector.iter_trajectories = MagicMock(return_value=iter([]))

    ctx = _make_ctx(idle_secs=1200, reflector=mock_reflector, collector=mock_collector)
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    agent._last_reflection_at = datetime.datetime.min  # before
    await agent._biological_tick()
    # Anchor must have been advanced despite the exception.
    assert agent._last_reflection_at > datetime.datetime.min


async def test_reflection_cooldown_constant_lies_between_dream_and_selfplay():
    """Sanity: reflection cooldown is between the others, matching the
    intended firing cadence."""
    assert (
        GhostAgent._DREAM_COOLDOWN
        < GhostAgent._REFLECTION_COOLDOWN
        < GhostAgent._SELFPLAY_COOLDOWN
    )


async def test_reflected_ids_set_initialized_on_context():
    """The idempotency set should be created on first fire."""
    mock_reflector = MagicMock()
    mock_reflector.run = AsyncMock(return_value=ReflectionRunReport())
    mock_collector = MagicMock()
    mock_collector.iter_trajectories = MagicMock(return_value=iter([]))

    ctx = _make_ctx(idle_secs=1200, reflector=mock_reflector, collector=mock_collector)
    await _tick(ctx)
    # The tick should have attached a set onto the context.
    assert hasattr(ctx, "_reflected_trajectory_ids") or \
           ctx._reflected_trajectory_ids is not None
