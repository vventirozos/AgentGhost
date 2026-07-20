"""Tests for the adaptive self-play cooldown in _biological_tick.

After every self-play run the watchdog must recompute the next cooldown
from the FrontierTracker's last compression delta: shorter when the
run made progress, longer when it was wasted.
"""

import datetime
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.memory.frontier import FrontierTracker


@pytest.fixture(autouse=True)
def _isolate_ghost_home(monkeypatch):
    """_biological_tick's counterfactual batch resolves its replay
    ledger from $GHOST_HOME. On a dev box with GHOST_HOME exported and
    a populated ledger, ticks here replayed REAL pending challenges
    through the mocked dreamer (extra synthetic_self_play awaits) and
    appended results to the operator's ledger. Keep the tests hermetic."""
    monkeypatch.delenv("GHOST_HOME", raising=False)


def _make_agent_with_tracker(tmp_path, idle_seconds=4000):
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    ctx.args.temperature = 0.5
    ctx.args.max_context = 4000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False

    ctx.llm_client = MagicMock()
    ctx.llm_client.foreground_tasks = 0
    ctx.llm_client.foreground_requests = 0  # HARD LOCK also gates on this now

    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get.return_value = {"ids": []}

    ctx.profile_memory = MagicMock()
    ctx.scratchpad = MagicMock()
    ctx.skill_memory = None
    ctx.graph_memory = None
    ctx.sandbox_dir = str(tmp_path)
    ctx.journal = None
    ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=idle_seconds)

    ctx.frontier_tracker = FrontierTracker(tmp_path)
    return GhostAgent(ctx)


@pytest.mark.asyncio
async def test_default_cooldown_before_any_run(tmp_path):
    agent = _make_agent_with_tracker(tmp_path)
    # Trigger the lazy init path via one tick (force random gate to skip so
    # the actual dreamer is never invoked).
    with patch("ghost_agent.core.dream.Dreamer"), \
         patch("ghost_agent.core.agent.random.random", return_value=0.9):
        await agent._biological_tick()
    assert agent._current_selfplay_cooldown == GhostAgent._SELFPLAY_COOLDOWN


@pytest.mark.asyncio
async def test_cooldown_halves_on_compression_progress(tmp_path):
    agent = _make_agent_with_tracker(tmp_path)
    # Prime the tracker: first run establishes baseline, second run is a
    # big compression improvement.
    ft = agent.context.frontier_tracker
    ft.record_run("sql", "c1", 1, True, 2000)

    mock_dreamer = MagicMock()
    mock_dreamer.dream = AsyncMock()

    async def fake_self_play(*a, **kw):
        # Simulate a passing compression-improvement run inside the mock.
        ft.record_run("sql", "c2", 1, True, 1000)

    mock_dreamer.synthetic_self_play = AsyncMock(side_effect=fake_self_play)

    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer), \
         patch("ghost_agent.core.agent.random.random", return_value=0.05):
        await agent._biological_tick()

    # last delta = 0.5 → cooldown = base/2 = 1800
    assert agent._current_selfplay_cooldown == 1800


@pytest.mark.asyncio
async def test_cooldown_doubles_on_failure(tmp_path):
    agent = _make_agent_with_tracker(tmp_path)
    ft = agent.context.frontier_tracker

    async def fake_self_play(*a, **kw):
        ft.record_run("sql", "cfail", 3, False, 0, mistake="broken")

    mock_dreamer = MagicMock()
    mock_dreamer.dream = AsyncMock()
    mock_dreamer.synthetic_self_play = AsyncMock(side_effect=fake_self_play)

    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer), \
         patch("ghost_agent.core.agent.random.random", return_value=0.05):
        await agent._biological_tick()

    # Failure → 2 * base = 7200, capped at ceiling
    assert agent._current_selfplay_cooldown == 7200


@pytest.mark.asyncio
async def test_cooldown_gate_respects_adaptive_value(tmp_path):
    """After a failure, the cooldown doubles — a subsequent tick at only
    3700s since last self-play must NOT trigger another run."""
    agent = _make_agent_with_tracker(tmp_path)
    ft = agent.context.frontier_tracker

    async def fake_self_play(*a, **kw):
        ft.record_run("sql", "fail", 3, False, 0)

    mock_dreamer = MagicMock()
    mock_dreamer.dream = AsyncMock()
    mock_dreamer.synthetic_self_play = AsyncMock(side_effect=fake_self_play)

    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer), \
         patch("ghost_agent.core.agent.random.random", return_value=0.05):
        await agent._biological_tick()

    assert mock_dreamer.synthetic_self_play.await_count == 1
    assert agent._current_selfplay_cooldown == 7200

    # Simulate 3700s of additional idle (>60 min, but well under the new
    # 7200s adaptive cooldown). Another tick should NOT fire self-play.
    agent._last_selfplay_at = datetime.datetime.now() - datetime.timedelta(seconds=3700)
    agent.context.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=4000)

    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer), \
         patch("ghost_agent.core.agent.random.random", return_value=0.05):
        await agent._biological_tick()

    assert mock_dreamer.synthetic_self_play.await_count == 1  # unchanged


@pytest.mark.asyncio
async def test_missing_frontier_tracker_uses_baseline(tmp_path):
    """If context has no frontier_tracker, the cooldown stays at baseline."""
    agent = _make_agent_with_tracker(tmp_path)
    agent.context.frontier_tracker = None

    async def fake_self_play(*a, **kw):
        pass

    mock_dreamer = MagicMock()
    mock_dreamer.dream = AsyncMock()
    mock_dreamer.synthetic_self_play = AsyncMock(side_effect=fake_self_play)

    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer), \
         patch("ghost_agent.core.agent.random.random", return_value=0.05):
        await agent._biological_tick()

    assert agent._current_selfplay_cooldown == GhostAgent._SELFPLAY_COOLDOWN
