"""Unit tests for the native asyncio biological_watchdog daemon on GhostAgent."""
import asyncio
import datetime
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.agent import GhostAgent, GhostContext


def _make_agent(idle_seconds: int = 0,
                foreground_tasks: int = 0,
                foreground_requests: int = 0,
                with_journal: bool = False,
                journal_items: int = 0,
                memory_ids: int = 0):
    """Build a minimally-mocked GhostAgent suitable for tick-level testing."""
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    ctx.args.temperature = 0.5
    ctx.args.max_context = 4000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    # idle-loop ablation toggles default OFF (production dreams / self-plays).
    # Must be real bools, not the auto-vivified MagicMock children, so the
    # phase gates (`is not True`) see them as disabled=False.
    ctx.args.no_dream = False
    ctx.args.no_self_play = False

    ctx.llm_client = MagicMock()
    ctx.llm_client.foreground_tasks = foreground_tasks
    # The HARD LOCK now defers on an in-flight user REQUEST too, not just an
    # in-flight LLM call. A bare MagicMock would auto-return a truthy
    # foreground_requests and trip the gate — set it explicitly.
    ctx.llm_client.foreground_requests = foreground_requests

    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get.return_value = {
        "ids": [str(i) for i in range(memory_ids)]
    }

    ctx.profile_memory = MagicMock()
    ctx.scratchpad = MagicMock()
    ctx.skill_memory = None
    ctx.graph_memory = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=idle_seconds)

    if with_journal:
        ctx.journal = MagicMock()
        ctx.journal._lock = MagicMock()
        ctx.journal._lock.__enter__ = MagicMock(return_value=None)
        ctx.journal._lock.__exit__ = MagicMock(return_value=False)
        _items = [{"type": "smart_memory", "data": {"text": "x", "model": "m"}}] * journal_items
        ctx.journal.file_path = MagicMock()
        ctx.journal.file_path.read_text.return_value = json.dumps(_items)
        # The watchdog now routes through the guarded journal.load() (so a
        # corrupt journal sidecars instead of silently skipping consolidation).
        ctx.journal.load.return_value = _items
        # Phase-1 'has work?' uses pending_count() (hot + overflow), not
        # len(load()), so overflow-only work isn't mistaken for empty.
        ctx.journal.pending_count.return_value = journal_items
    else:
        ctx.journal = None

    return GhostAgent(ctx)


# ---------------------------------------------------------------- guard rails


@pytest.mark.asyncio
async def test_tick_skips_when_no_memory_system():
    agent = _make_agent(idle_seconds=4000)
    agent.context.memory_system = None
    # Should silently no-op; no exception, no Dreamer instantiation.
    with patch("ghost_agent.core.dream.Dreamer") as MockDreamer:
        await agent._biological_tick()
        MockDreamer.assert_not_called()


@pytest.mark.asyncio
async def test_tick_skips_when_foreground_active():
    """Hard lock: never interrupt an active LLM generation."""
    agent = _make_agent(idle_seconds=4000, foreground_tasks=2, memory_ids=5)
    with patch("ghost_agent.core.dream.Dreamer") as MockDreamer:
        await agent._biological_tick()
        MockDreamer.assert_not_called()
    # last_activity_time untouched
    assert (datetime.datetime.now() - agent.context.last_activity_time).total_seconds() > 3000


@pytest.mark.asyncio
async def test_tick_does_nothing_when_not_idle_enough():
    """Less than 120s idle → no journal/dream/self-play triggered."""
    agent = _make_agent(idle_seconds=30, with_journal=True, journal_items=5,
                        memory_ids=10)
    agent.process_journal_queue = AsyncMock()
    with patch("ghost_agent.core.dream.Dreamer") as MockDreamer:
        await agent._biological_tick()
        agent.process_journal_queue.assert_not_called()
        MockDreamer.assert_not_called()


# ------------------------------------------------------------ phase 1: journal


@pytest.mark.asyncio
async def test_tick_phase1_processes_journal_when_items_present():
    agent = _make_agent(idle_seconds=200, with_journal=True, journal_items=3)
    agent.process_journal_queue = AsyncMock()
    before = agent.context.last_activity_time
    await agent._biological_tick()
    agent.process_journal_queue.assert_awaited_once()
    # last_activity_time must NOT be reset — that clock tracks user
    # idleness, and resetting it on internal journal work used to
    # starve phase 3 (self-play). Per-phase gates control refire.
    assert agent.context.last_activity_time == before


@pytest.mark.asyncio
async def test_tick_phase1_skips_empty_journal():
    agent = _make_agent(idle_seconds=200, with_journal=True, journal_items=0)
    agent.process_journal_queue = AsyncMock()
    await agent._biological_tick()
    agent.process_journal_queue.assert_not_called()


# -------------------------------------------------------------- phase 2: dream


@pytest.mark.asyncio
async def test_tick_phase2_triggers_rem_dream():
    """10–60 min idle with ≥3 auto memories should fire a REM dream."""
    agent = _make_agent(idle_seconds=900, memory_ids=4)
    mock_dreamer = MagicMock()
    mock_dreamer.dream = AsyncMock()
    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer) as MockDreamer, \
         patch("ghost_agent.core.agent.random.random", return_value=0.1):
        await agent._biological_tick()
    MockDreamer.assert_called_once_with(agent.context)
    mock_dreamer.dream.assert_awaited_once()
    mock_dreamer.dream.assert_awaited_with(model_name="test-model")


@pytest.mark.asyncio
async def test_tick_phase2_skips_when_no_dream():
    """--no-dream ablates JUST the REM dream loop: even fully eligible (idle,
    ≥3 auto memories, random gate open), Dreamer.dream must not fire. The
    Track-B earn-keep dream-off arm depends on this."""
    agent = _make_agent(idle_seconds=900, memory_ids=4)
    agent.context.args.no_dream = True
    mock_dreamer = MagicMock()
    mock_dreamer.dream = AsyncMock()
    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer), \
         patch("ghost_agent.core.agent.random.random", return_value=0.1):
        await agent._biological_tick()
    mock_dreamer.dream.assert_not_called()


@pytest.mark.asyncio
async def test_tick_phase2_random_gate_can_skip():
    agent = _make_agent(idle_seconds=900, memory_ids=4)
    mock_dreamer = MagicMock()
    mock_dreamer.dream = AsyncMock()
    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer), \
         patch("ghost_agent.core.agent.random.random", return_value=0.99):
        await agent._biological_tick()
    mock_dreamer.dream.assert_not_called()


@pytest.mark.asyncio
async def test_tick_phase2_skips_when_not_enough_memories():
    agent = _make_agent(idle_seconds=900, memory_ids=1)
    with patch("ghost_agent.core.dream.Dreamer") as MockDreamer, \
         patch("ghost_agent.core.agent.random.random", return_value=0.1):
        await agent._biological_tick()
    MockDreamer.assert_not_called()


@pytest.mark.asyncio
async def test_tick_phase2_advances_cooldown_when_dream_raises():
    """Regression: a crashing REM dream must still advance `_last_dream_at`
    so the watchdog doesn't re-fire the failing dream every tick.
    `last_activity_time` is deliberately NOT touched — it tracks user
    idleness, not agent internal work."""
    agent = _make_agent(idle_seconds=900, memory_ids=4)
    mock_dreamer = MagicMock()
    mock_dreamer.dream = AsyncMock(side_effect=RuntimeError("llm boom"))
    before_tick = datetime.datetime.now()
    activity_before = agent.context.last_activity_time

    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer), \
         patch("ghost_agent.core.agent.random.random", return_value=0.1):
        with pytest.raises(RuntimeError):
            await agent._biological_tick()

    mock_dreamer.dream.assert_awaited_once()
    # Cooldown anchor must have advanced past the start of the tick.
    assert agent._last_dream_at >= before_tick
    # User-activity clock must be untouched.
    assert agent.context.last_activity_time == activity_before


@pytest.mark.asyncio
async def test_tick_phase2_preserves_idle_clock_for_phase3():
    """A successful REM dream must leave `last_activity_time` untouched so
    `idle_secs` can continue growing into the phase-3 (self-play) window.
    Resetting the clock here was what made self-play unreachable on an
    AFK agent."""
    agent = _make_agent(idle_seconds=900, memory_ids=4)
    mock_dreamer = MagicMock()
    mock_dreamer.dream = AsyncMock(return_value="Dream Complete.")
    activity_before = agent.context.last_activity_time

    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer), \
         patch("ghost_agent.core.agent.random.random", return_value=0.1):
        await agent._biological_tick()

    mock_dreamer.dream.assert_awaited_once()
    assert agent.context.last_activity_time == activity_before
    # And the per-phase cooldown still advanced, so phase 2 won't
    # immediately refire on the next tick.
    assert agent._last_dream_at > activity_before


@pytest.mark.asyncio
async def test_tick_phase2_does_not_refire_after_failed_dream():
    """End-to-end of the regression: two back-to-back ticks where the
    first dream raises must result in exactly one Dreamer.dream() call,
    because the cooldown anchor blocks the second tick."""
    agent = _make_agent(idle_seconds=900, memory_ids=4)
    mock_dreamer = MagicMock()
    mock_dreamer.dream = AsyncMock(side_effect=RuntimeError("llm boom"))

    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer), \
         patch("ghost_agent.core.agent.random.random", return_value=0.1):
        # Tick 1 — dream fires and raises. Watchdog would catch this
        # exception in production; here we let it bubble to confirm the
        # finally-clause still ran.
        with pytest.raises(RuntimeError):
            await agent._biological_tick()

        # Simulate a full minute elapsing before the next tick — still
        # well inside the 30-min dream cooldown.
        agent.context.last_activity_time = (
            datetime.datetime.now() - datetime.timedelta(seconds=900)
        )

        # Tick 2 — cooldown must suppress the second fire even though
        # idle_secs is back above 600.
        await agent._biological_tick()

    assert mock_dreamer.dream.await_count == 1


# ---------------------------------------------------------- phase 3: self-play


@pytest.mark.asyncio
async def test_tick_phase3_triggers_self_play():
    """>60 min idle should fire synthetic_self_play (under random gate)."""
    agent = _make_agent(idle_seconds=4000)
    mock_dreamer = MagicMock()
    mock_dreamer.dream = AsyncMock()
    mock_dreamer.synthetic_self_play = AsyncMock()
    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer) as MockDreamer, \
         patch("ghost_agent.core.agent.random.random", return_value=0.05):
        await agent._biological_tick()
    MockDreamer.assert_called_once_with(agent.context)
    mock_dreamer.synthetic_self_play.assert_awaited_once_with(
        model_name="test-model", is_background=True
    )
    mock_dreamer.dream.assert_not_called()


@pytest.mark.asyncio
async def test_tick_phase3_skips_when_no_self_play():
    """--no-self-play ablates JUST the self-play loop: even deeply idle with
    the random gate open, synthetic_self_play must not fire. The Track-B
    earn-keep self-play-off arm depends on this."""
    agent = _make_agent(idle_seconds=4000)
    agent.context.args.no_self_play = True
    mock_dreamer = MagicMock()
    mock_dreamer.synthetic_self_play = AsyncMock()
    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer), \
         patch("ghost_agent.core.agent.random.random", return_value=0.05):
        await agent._biological_tick()
    mock_dreamer.synthetic_self_play.assert_not_called()


@pytest.mark.asyncio
async def test_tick_phase2_only_no_dream_leaves_self_play(monkeypatch):
    """Isolation guarantee: --no-dream must NOT disable self-play (and vice
    versa). A deeply-idle agent with only dream ablated still self-plays."""
    agent = _make_agent(idle_seconds=4000)
    agent.context.args.no_dream = True
    agent.context.args.no_self_play = False
    mock_dreamer = MagicMock()
    mock_dreamer.dream = AsyncMock()
    mock_dreamer.synthetic_self_play = AsyncMock()
    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer), \
         patch("ghost_agent.core.agent.random.random", return_value=0.05):
        await agent._biological_tick()
    mock_dreamer.synthetic_self_play.assert_awaited_once()


@pytest.mark.asyncio
async def test_tick_phase3_random_gate_can_skip():
    agent = _make_agent(idle_seconds=4000)
    mock_dreamer = MagicMock()
    mock_dreamer.synthetic_self_play = AsyncMock()
    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer), \
         patch("ghost_agent.core.agent.random.random", return_value=0.9):
        await agent._biological_tick()
    mock_dreamer.synthetic_self_play.assert_not_called()


# ---------------------------------------------------- daemon loop and lifespan


_REAL_SLEEP = asyncio.sleep


async def _yield_sleep(_secs):
    """Replacement for asyncio.sleep that yields immediately so the test
    loop can drive multiple watchdog iterations without waiting 60s."""
    await _REAL_SLEEP(0)


@pytest.mark.asyncio
async def test_biological_watchdog_loops_and_cancels_cleanly():
    """The infinite loop must propagate cancellation without raising."""
    agent = _make_agent(idle_seconds=10)
    tick_calls = []

    async def fake_tick():
        tick_calls.append(1)

    agent._biological_tick = fake_tick

    with patch("ghost_agent.core.agent.asyncio.sleep", new=_yield_sleep):
        task = asyncio.create_task(agent.biological_watchdog())
        for _ in range(5):
            await _REAL_SLEEP(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(tick_calls) >= 1


@pytest.mark.asyncio
async def test_biological_watchdog_swallows_tick_exceptions():
    """A broken tick must not crash the daemon loop."""
    agent = _make_agent(idle_seconds=10)
    calls = {"n": 0}

    async def boom_tick():
        calls["n"] += 1
        raise RuntimeError("boom")

    agent._biological_tick = boom_tick

    with patch("ghost_agent.core.agent.asyncio.sleep", new=_yield_sleep):
        task = asyncio.create_task(agent.biological_watchdog())
        for _ in range(5):
            await _REAL_SLEEP(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    # Loop survived past the first failure.
    assert calls["n"] >= 2


@pytest.mark.asyncio
async def test_lifespan_starts_and_cancels_biological_daemon():
    """The FastAPI lifespan must spin up the daemon and cancel it on shutdown."""
    from ghost_agent.main import lifespan

    mock_app = MagicMock()
    mock_app.state.args = MagicMock()
    mock_app.state.args.no_memory = True
    mock_app.state.args.swarm_nodes_parsed = []
    mock_app.state.args.worker_nodes_parsed = []
    mock_app.state.args.visual_nodes_parsed = []
    mock_app.state.args.coding_nodes_parsed = []
    mock_app.state.args.image_gen_nodes_parsed = []
    mock_app.state.args.upstream_url = "http://mock"

    mock_context = MagicMock()
    mock_context.tor_proxy = None
    mock_context.memory_dir = "/tmp/memory"
    mock_app.state.context = mock_context

    fake_agent = MagicMock()
    fake_agent.biological_watchdog = AsyncMock(side_effect=asyncio.sleep)

    with patch("ghost_agent.main.LLMClient") as MockLLM, \
         patch("ghost_agent.main.importlib.util.find_spec", return_value=False), \
         patch("ghost_agent.main.ProfileMemory"), \
         patch("ghost_agent.main.GraphMemory"), \
         patch("ghost_agent.main.GhostAgent", return_value=fake_agent):
        mock_llm_instance = MagicMock()
        mock_llm_instance.close = AsyncMock()
        MockLLM.return_value = mock_llm_instance

        async with lifespan(mock_app):
            # Daemon should be a live asyncio.Task attached to app.state
            bio_task = mock_app.state.biological_task
            assert isinstance(bio_task, asyncio.Task)
            assert not bio_task.done()
            fake_agent.biological_watchdog.assert_called_once()

        # After exit: cancelled and LLM client closed
        assert bio_task.cancelled() or bio_task.done()
        mock_llm_instance.close.assert_awaited_once()
    # Scheduler attribute is now a live APScheduler AsyncIOScheduler
    # instance (re-enabled for user-facing manage_tasks). Post-lifespan
    # it should be shut down. The previous assertion that it stayed
    # `None` only held while apscheduler was deliberately removed.
    sched = mock_context.scheduler
    assert sched is not None, "scheduler should be initialized at boot"
    # Post-shutdown: AsyncIOScheduler's `running` flag flips to False.
    # Use getattr with default in case the test env mocks out the attribute.
    assert getattr(sched, "running", False) is False, (
        "scheduler should be stopped after lifespan exits"
    )


def test_main_module_has_no_global_context_or_agent():
    """Defensive regression test: globals must be fully removed."""
    import ghost_agent.main as m
    assert not hasattr(m, "GLOBAL_CONTEXT")
    assert not hasattr(m, "GLOBAL_AGENT")
    assert not hasattr(m, "idle_dream_watchdog")
    assert not hasattr(m, "proactive_runner")


def test_main_module_does_not_import_apscheduler():
    """Defensive regression test: apscheduler must be gone from main.py."""
    import ghost_agent.main as m
    assert not hasattr(m, "AsyncIOScheduler")
    assert not hasattr(m, "SQLAlchemyJobStore")
