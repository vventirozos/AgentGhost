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
from unittest.mock import MagicMock, patch

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
    ctx.args = MagicMock(model="default", no_dream=False)
    agent = agent_mod.GhostAgent.__new__(agent_mod.GhostAgent)
    agent.context = ctx
    far_past = datetime.datetime.now() - datetime.timedelta(hours=3)
    agent._last_selfplay_at = far_past
    agent._current_selfplay_cooldown = 3600
    return agent

