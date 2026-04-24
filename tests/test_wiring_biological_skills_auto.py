"""Integration tests for the skills_auto phase inside _biological_tick.

Mirrors the shape of test_reflection_biological_tick.py — confirms the
phase fires only when the trajectory collector is wired, respects its
own cooldown, and stays inside the 15-60 min idle window.
"""

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.core.agent import GhostAgent


def _make_ctx(*, idle_secs: float, collector=None):
    ctx = MagicMock()
    ctx.memory_system = MagicMock()
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    ctx.journal = None
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=idle_secs)
    ctx.args = MagicMock()
    ctx.args.model = "m"
    ctx.frontier_tracker = None
    ctx.reflector = None
    ctx.trajectory_collector = collector
    return ctx


async def _tick(ctx):
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    await agent._biological_tick()
    return agent


async def test_skills_auto_skipped_without_collector():
    ctx = _make_ctx(idle_secs=1200, collector=None)
    agent = await _tick(ctx)
    # With no collector, anchor should still be at its initial value.
    assert agent._last_skills_auto_at == datetime.datetime.min


async def test_skills_auto_fires_when_collector_present():
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter([]))
    ctx = _make_ctx(idle_secs=1200, collector=collector)
    agent = await _tick(ctx)
    collector.iter_trajectories.assert_called_once()
    assert agent._last_skills_auto_at > datetime.datetime.min


async def test_skills_auto_respects_cooldown():
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter([]))
    ctx = _make_ctx(idle_secs=1200, collector=collector)
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    agent._last_skills_auto_at = datetime.datetime.now()  # just fired
    await agent._biological_tick()
    collector.iter_trajectories.assert_not_called()


async def test_skills_auto_outside_idle_window_skips():
    """Under 15 min idle, the phase should not engage even when a
    collector is present."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter([]))
    ctx = _make_ctx(idle_secs=300, collector=collector)
    await _tick(ctx)
    collector.iter_trajectories.assert_not_called()


async def test_skills_auto_cooldown_constant_is_longer_than_reflection():
    """Sanity: skills auto-extraction runs rarer than reflection,
    because the result is CPU-only analysis that doesn't need the
    fidelity of a single-idle-session sweep."""
    assert GhostAgent._SKILLS_AUTO_COOLDOWN > GhostAgent._REFLECTION_COOLDOWN


async def test_skills_auto_advances_anchor_on_exception():
    """Same anchor-advances-on-exception invariant as dream/reflection —
    a broken extractor must not re-fire every tick."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(side_effect=RuntimeError("boom"))
    ctx = _make_ctx(idle_secs=1200, collector=collector)
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    agent._last_skills_auto_at = datetime.datetime.min
    await agent._biological_tick()
    assert agent._last_skills_auto_at > datetime.datetime.min
