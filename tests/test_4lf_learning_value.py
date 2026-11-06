"""§4LF (2026-10-04): background learning that re-ran on unchanged input, and
the idle clock that only self-play could roll. Each test names the world it
fails in. §4NM: the --no-self-play / bench / adaptive-cooldown tests removed."""
import asyncio
import datetime
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ── "has the input changed?" survives a restart ──────────────────────────────
@pytest.mark.parametrize("modpath,cls", [("ghost_agent.selfhood.narrative", "NarrativeSummariser"),
                                         ("ghost_agent.workspace.narrative", "WorkspaceNarrative")])
def test_a_narratives_input_key_survives_a_restart(tmp_path, modpath, cls):
    """Fails where the key lived in memory only: 213 of 354 main-model
    narrative rewrites were the first after one of 285 boots."""
    import importlib
    C = getattr(importlib.import_module(modpath), cls)
    first = C(tmp_path)
    first._remember_input_key("abc123")
    assert C(tmp_path)._last_input_key == "abc123"
    assert C(tmp_path / "fresh")._last_input_key == ""


# ── the idle window without self-play ────────────────────────────────────────
def _tick_ctx(idle_secs):
    """Minimal mock context for `_biological_tick` (moved here from the
    derived-mood tests when §4NM deleted that idle phase)."""
    ctx = MagicMock()
    ctx.memory_system = MagicMock()
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    ctx.journal = None
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    ctx.last_activity_time = (datetime.datetime.now()
                              - datetime.timedelta(seconds=idle_secs))
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    ctx.args.self_narrative_cooldown = None
    ctx.frontier_tracker = None
    ctx.trajectory_collector = MagicMock()
    ctx.self_model = None
    return ctx


async def _bio_tick(ctx):
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    await agent._biological_tick()
    return agent


@pytest.mark.asyncio
async def test_the_window_also_rolls_while_self_play_cools_down(monkeypatch):
    from ghost_agent.core.agent import GhostAgent
    ctx = _tick_ctx(4000)
    before = ctx.last_activity_time
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    agent._last_selfplay_at = datetime.datetime.now()       # just ran: in its cooldown
    await agent._biological_tick()
    assert getattr(agent, "_idle_window_at", datetime.datetime.min) > before


@pytest.mark.asyncio
async def test_inside_the_window_the_clock_is_left_alone():
    ctx = _tick_ctx(1200)
    agent = await _bio_tick(ctx)
    assert getattr(agent, "_idle_window_at", datetime.datetime.min) == datetime.datetime.min


# ── experiments and graduated skills ─────────────────────────────────────────
@pytest.mark.asyncio
async def test_a_rolled_window_reopens_the_mid_phases_on_the_next_tick():
    """The point of the roll: the 15–60 min phases run again an hour later
    instead of never."""
    from ghost_agent.core.agent import GhostAgent
    ctx = _tick_ctx(4000)
    from unittest.mock import AsyncMock
    ctx.self_model = MagicMock(enabled=True)
    ctx.self_model.consolidate_narrative = AsyncMock(return_value="")
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    agent._last_selfplay_at = datetime.datetime.now()       # §4NM: no --no-self-play; the slot cools down
    await agent._biological_tick()
    # (§4TB: the witness was the stale-questions job; it is removed, and the
    # diary — the same (900, 3600] window — is the mid phase observed now)
    assert agent._last_narrative_at == datetime.datetime.min      # idle 4000 s: outside the window
    assert ctx.self_model.consolidate_narrative.await_count == 0
    agent._idle_window_at = datetime.datetime.now() - datetime.timedelta(seconds=1200)
    await agent._biological_tick()                      # window idle now 1200 s: inside (900, 3600]
    assert agent._last_narrative_at > datetime.datetime.min
    assert ctx.self_model.consolidate_narrative.await_count == 1


# ── the operator's frequency decisions (2026-10-04) ──────────────────────────
@pytest.mark.parametrize("hour,quiet", [(23, True), (2, True), (6, True), (7, False), (13, False), (22, False)])
def test_quiet_hours_wrap_midnight(hour, quiet):
    from ghost_agent.core.autonomous_activity import in_quiet_hours
    assert in_quiet_hours(datetime.datetime(2026, 10, 4, hour, 30), "23-07") is quiet
    assert in_quiet_hours(datetime.datetime(2026, 10, 4, hour, 30), "off") is False


def test_quiet_hours_hold_notices_without_losing_them(monkeypatch, tmp_path):
    """Fails where self-play regressions paged at 03:41 and 04:55: inside the
    window nothing is served and the watermark stays, so the first poll
    after it delivers them."""
    import ghost_agent.api.routes as routes
    import ghost_agent.core.autonomous_activity as aa
    log = MagicMock()
    log.read_since.return_value = ([], 9)
    log.current_offset.return_value = 9
    agent = SimpleNamespace(context=SimpleNamespace(memory_dir=str(tmp_path / "memory")))
    monkeypatch.setattr(routes, "get_agent", lambda r: agent)
    monkeypatch.setattr(aa, "get_activity_log", lambda ctx: log)
    monkeypatch.setattr(aa, "load_consumer_offset", lambda path, consumer: 5)
    monkeypatch.setattr(aa, "in_quiet_hours", lambda *a, **k: True)
    resp = asyncio.run(routes.notifications_pending(MagicMock(), consumer="slack"))
    body = json.loads(resp.body)
    assert body["quiet_hours"] is True and body["records"] == [] and body["watermark"] == 5   # nothing acked away
    monkeypatch.setattr(aa, "in_quiet_hours", lambda *a, **k: False)
    body = json.loads(asyncio.run(routes.notifications_pending(MagicMock(), consumer="slack")).body)
    assert "quiet_hours" not in body


def test_the_adaptive_threshold_is_opt_in(monkeypatch, tmp_path):
    import ghost_agent.main as M
    ctx = SimpleNamespace(memory_dir=tmp_path)
    monkeypatch.delenv("GHOST_ADAPTIVE_THRESHOLD", raising=False)
    M._maybe_adaptive_threshold(ctx)
    assert not hasattr(ctx, "adaptive_threshold")
    monkeypatch.setenv("GHOST_ADAPTIVE_THRESHOLD", "1")
    M._maybe_adaptive_threshold(ctx)
    assert ctx.adaptive_threshold is not None


def test_quiet_hours_never_hold_what_the_owner_asked_for(monkeypatch, tmp_path):
    """§4LI (final review): a "notify me when done", a scheduled reminder or
    a correction to the owner's answer waited until 07:00 — and so did
    everything while the owner was at the console."""
    import datetime as _dt
    from ghost_agent.core.autonomous_activity import owner_awaits
    rec = lambda phase, **meta: SimpleNamespace(phase=phase, meta=meta)
    assert owner_awaits([rec("agent_message", auto="job finished")])
    assert owner_awaits([rec("scheduled_task")])
    assert owner_awaits([rec("project", auto="project notify promise")])
    assert not owner_awaits([rec("experiment_verdict"), rec("evolve_proposal")])
    now = _dt.datetime(2026, 10, 4, 23, 40)
    assert owner_awaits([rec("evolve_proposal")], last_activity=now - _dt.timedelta(minutes=10), now=now)
    assert not owner_awaits([rec("evolve_proposal")], last_activity=now - _dt.timedelta(hours=2), now=now)
