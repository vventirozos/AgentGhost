"""§4LF (2026-10-04): background learning that re-ran on unchanged input, and
the idle clock that only self-play could roll. Each test names the world it
fails in."""
import asyncio
import datetime
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


def _engine(tmp_path):
    from ghost_agent.reflection.postmortem import DefectQueue, PostMortemEngine
    return PostMortemEngine(AsyncMock(), queue=DefectQueue(tmp_path))


def test_a_failed_postmortem_analysis_is_not_retried_forever_across_restarts(tmp_path):
    """Fails where every restart re-tried the same failed analyses (~230
    main-model calls, mostly 120 s timeouts, over ~129 boots)."""
    from ghost_agent.reflection.postmortem import _FAILED_ANALYSIS_MAX_TRIES
    for _ in range(_FAILED_ANALYSIS_MAX_TRIES - 1):
        e = _engine(tmp_path)                       # one boot …
        assert "sig-1" not in e._failed_analysis_sigs  # … a retry is still allowed
        e._note_failed_analysis("sig-1")
    e = _engine(tmp_path)
    e._note_failed_analysis("sig-1")                # the last allowed try fails too
    assert "sig-1" in _engine(tmp_path)._failed_analysis_sigs   # skipped from the next boot on


# ── the idle window without self-play ────────────────────────────────────────
def _tick_ctx(idle_secs, no_self_play):
    from tests.test_selfhood_derived_mood import _make_bio_ctx
    ctx = _make_bio_ctx(idle_secs=idle_secs, self_model=None)
    ctx.args.no_self_play = no_self_play
    return ctx


@pytest.mark.asyncio
async def test_the_idle_window_rolls_over_with_self_play_off():
    """Fails where self-play's finally was the only idle-time writer of the
    clock: --no-self-play left idle past 3600 s for a whole away stretch and
    silently stopped reflection, postmortem, skills, router, calibration,
    the narratives and autoadvance."""
    from tests.test_selfhood_derived_mood import _bio_tick
    ctx = _tick_ctx(4000, True)
    before = ctx.last_activity_time
    agent = await _bio_tick(ctx)
    assert getattr(agent, "_idle_window_at", datetime.datetime.min) > before
    assert ctx.last_activity_time == before          # the USER clock is never touched


@pytest.mark.asyncio
async def test_the_window_also_rolls_while_self_play_cools_down(monkeypatch):
    from ghost_agent.core.agent import GhostAgent
    from tests.test_selfhood_derived_mood import _bio_tick
    ctx = _tick_ctx(4000, False)
    before = ctx.last_activity_time
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    agent._last_selfplay_at = datetime.datetime.now()       # just ran: in its cooldown
    await agent._biological_tick()
    assert getattr(agent, "_idle_window_at", datetime.datetime.min) > before


@pytest.mark.asyncio
async def test_inside_the_window_the_clock_is_left_alone():
    from tests.test_selfhood_derived_mood import _bio_tick
    ctx = _tick_ctx(1200, True)
    agent = await _bio_tick(ctx)
    assert getattr(agent, "_idle_window_at", datetime.datetime.min) == datetime.datetime.min


# ── experiments and graduated skills ─────────────────────────────────────────
def test_every_stamped_trigger_is_analysed():
    """Fails where imagine_preflight and search_yield_steer stamped a
    trigger the report never read (only their diluted all-turns block)."""
    from ghost_agent.core.experiments import TRIGGER_KEYS
    assert TRIGGER_KEYS["imagine_preflight"] == "imagine_preflight_fired"
    assert TRIGGER_KEYS["search_yield_steer"] == "search_yield_steer_fired"


def test_only_new_evidence_counts_as_a_verification(tmp_path):
    """Fails where every idle run re-graduated every skill on an unchanged
    corpus (950 "verifications" on one), refreshing last_verified_at and so
    disabling the 14-day staleness rule."""
    from ghost_agent.skills_auto.store import GraduatedSkillStore
    st = GraduatedSkillStore(tmp_path)
    cand = SimpleNamespace(signature_hash="h1", name="svc", cluster=None, tool_sequence=["a"],
                           support=5, trigger_examples=["stop the chess service"], exemplar_trajectory_id="")
    st.graduate(cand, confidence=0.9)
    raw = json.loads(st.path.read_text())
    v0 = raw["h1"].get("verifications", 0)
    raw["h1"]["last_verified_at"] = "2026-09-01T00:00:00Z"
    st.path.write_text(json.dumps(raw))
    st.graduate(cand, confidence=0.9)                      # same corpus
    after = json.loads(st.path.read_text())["h1"]
    assert after["last_verified_at"] == "2026-09-01T00:00:00Z" and after.get("verifications", 0) == v0
    cand.support = 6
    st.graduate(cand, confidence=0.9)                      # new evidence
    assert json.loads(st.path.read_text())["h1"]["last_verified_at"] != "2026-09-01T00:00:00Z"


@pytest.mark.asyncio
async def test_a_rolled_window_reopens_the_mid_phases_on_the_next_tick():
    """The point of the roll: with self-play off, the 15–60 min phases run
    again an hour later instead of never."""
    from ghost_agent.core.agent import GhostAgent
    ctx = _tick_ctx(4000, True)
    ctx.self_model = MagicMock(enabled=True)
    ctx.self_model.stale_open_questions.return_value = []
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    await agent._biological_tick()
    assert agent._last_stale_questions_at == datetime.datetime.min      # idle 4000 s: outside the window
    agent._idle_window_at = datetime.datetime.now() - datetime.timedelta(seconds=1200)
    await agent._biological_tick()                      # window idle now 1200 s: inside (900, 3600]
    assert agent._last_stale_questions_at > datetime.datetime.min


# ── the operator's frequency decisions (2026-10-04) ──────────────────────────
def test_self_play_and_bench_run_less_often_and_are_tunable(monkeypatch):
    from ghost_agent.core.agent import GhostAgent, _env_cooldown_s
    assert GhostAgent._SELFPLAY_COOLDOWN == 14400 and GhostAgent._BENCH_COOLDOWN == 21600
    monkeypatch.setenv("GHOST_X_COOLDOWN", "7200")
    assert _env_cooldown_s("GHOST_X_COOLDOWN", 1) == 7200
    for bad in ("-5", "abc", "0"):
        monkeypatch.setenv("GHOST_X_COOLDOWN", bad)
        assert _env_cooldown_s("GHOST_X_COOLDOWN", 99) == 99


@pytest.mark.asyncio
async def test_a_failed_self_play_run_never_shortens_a_long_cooldown(monkeypatch):
    """Fails where the adaptive cooldown's fixed 2 h ceiling turned a 4 h
    base into 2 h after a FAILED run."""
    from unittest.mock import patch
    from tests.test_evolve_mutator import _idle_agent
    agent = _idle_agent(quiet_siblings=False)
    agent._last_bench_at = datetime.datetime.now()
    agent._last_dream_replay_at = datetime.datetime.now()
    agent._bio_deterministic = True
    tracker = MagicMock()
    tracker.adaptive_cooldown = MagicMock(return_value=28800)
    agent.context.frontier_tracker = tracker
    dreamer = MagicMock()
    dreamer.synthetic_self_play = AsyncMock(return_value="ok")
    dreamer.last_bench_result = None
    with patch("ghost_agent.core.dream.Dreamer", return_value=dreamer), \
            patch("ghost_agent.core.counterfactual.load_replay_candidates", return_value=[]):
        await agent._biological_tick()
    assert tracker.adaptive_cooldown.called
    kw = tracker.adaptive_cooldown.call_args.kwargs
    assert kw["ceiling"] >= 2 * kw["base"]


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
