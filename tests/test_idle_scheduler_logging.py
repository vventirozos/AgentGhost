"""Idle-scheduler (biological tick) instrumentation (2026-07-24).

The nightly idle loop was hard to reconstruct from logs: self-play fires on a
silent 20% dice roll gated together with the cooldown, so "self-play never
fired" looked identical to a crash, a lost roll, or "not idle enough". And no
consolidated per-cycle line said what ran.

These tests pin: (1) an eligible-but-dice-missed self-play tick logs a distinct
line and does NOT run; (2) a self-play tick that runs emits the per-cycle
"idle cycle: ran …" summary naming self-play.
"""
from __future__ import annotations

import datetime
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ghost_agent.core.agent as agent_mod
from ghost_agent.memory.frontier import FrontierTracker


def _make_deeply_idle_agent(tmp_path):
    """Agent idle >60min (only the self-play phase is eligible; the 10–60min
    phases are all past their upper window), cooldown elapsed."""
    ctx = MagicMock()
    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    ctx.llm_client = MagicMock(foreground_tasks=0, foreground_requests=0)
    ctx.frontier_tracker = FrontierTracker(tmp_path)
    ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(hours=2)
    ctx.args = MagicMock(model="default", no_self_play=False, no_dream=False)
    agent = agent_mod.GhostAgent.__new__(agent_mod.GhostAgent)
    agent.context = ctx
    far_past = datetime.datetime.now() - datetime.timedelta(hours=3)
    agent._last_selfplay_at = far_past
    agent._current_selfplay_cooldown = 3600
    return agent


@pytest.mark.asyncio
async def test_selfplay_dice_miss_is_logged_and_does_not_run(tmp_path, caplog):
    agent = _make_deeply_idle_agent(tmp_path)
    # random 0.99 → _bio_roll(0.2) is False → the dice-miss branch.
    with patch("ghost_agent.core.agent.random.random", return_value=0.99):
        with patch("ghost_agent.core.dream.Dreamer") as D:
            with caplog.at_level(logging.INFO, logger="GhostAgent"):
                await agent._biological_tick()
            D.assert_not_called()  # eligible, but skipped this tick
    assert any("eligible, skipped this tick" in r.getMessage()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_selfplay_run_emits_idle_cycle_summary(tmp_path, caplog):
    agent = _make_deeply_idle_agent(tmp_path)
    # random 0.0 → all _bio_roll gates pass → self-play runs.
    inst = MagicMock()
    inst.synthetic_self_play = AsyncMock(return_value=None)
    with patch("ghost_agent.core.agent.random.random", return_value=0.0):
        with patch("ghost_agent.core.dream.Dreamer", return_value=inst):
            with caplog.at_level(logging.INFO, logger="GhostAgent"):
                await agent._biological_tick()
    msgs = [r.getMessage() for r in caplog.records]
    assert any("idle cycle: ran" in m and "self-play" in m for m in msgs)
