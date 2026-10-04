"""§4LD (2026-10-04): what the agent does on its own — scheduled tasks, jobs,
idle phases, the messages it sends unprompted, and how it stops. Each test
names the world it fails in."""
import asyncio
import datetime as dt
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.utils.logging import (client_deadline_context, reply_surface_context, request_id_context,
                                       request_origin_context, requester_role_context, ORIGIN_PROBE)


# ── a scheduled turn's context ───────────────────────────────────────────────
def test_an_autonomous_turn_runs_as_the_owner_on_no_surface():
    """Fails where APScheduler's wake-up copied the last task-creating turn's
    contextvars into every later fire: a task created in a Slack channel made
    every scheduled run a PUBLIC reply, a web client's deadline cut it short."""
    import ghost_agent.main as M
    seen = {}

    async def _hc(body, bg, request_id=None):
        seen.update(role=requester_role_context.get(), surface=reply_surface_context.get(),
                    deadline=client_deadline_context.get())
        return "ok", None, None
    ctx = SimpleNamespace(agent=SimpleNamespace(handle_chat=_hc), llm_client=SimpleNamespace(foreground_requests=0))
    toks = [(v, v.set(x)) for v, x in ((requester_role_context, "member"), (reply_surface_context, "public"),
                                       (client_deadline_context, 30.0))]
    try:
        asyncio.run(M._handle_chat_foreground(ctx, {"messages": []}, "sched-x"))
        assert seen == {"role": "owner", "surface": "", "deadline": 0.0}
        assert reply_surface_context.get() == "public"           # the caller's context is restored
    finally:
        for v, t in reversed(toks):
            v.reset(t)


@pytest.mark.asyncio
async def test_an_internal_turn_is_not_activity(monkeypatch, tmp_path):
    """Fails where a task every <15 min kept `last_activity_time` fresh and
    silently starved every idle phase."""
    from tests.test_requester_role import _agent, _resp
    from tests.helpers import FakeBgTasks
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("ok"))
    old = dt.datetime(2026, 1, 1)
    ctx.last_activity_time = old
    await agent.handle_chat({"messages": [{"role": "user", "content": "check the disk"}]},
                            FakeBgTasks(), request_id="sched-task_1")
    assert ctx.last_activity_time == old
    await agent.handle_chat({"messages": [{"role": "user", "content": "hi"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="owner")
    assert ctx.last_activity_time > old


# ── scheduling bounds ────────────────────────────────────────────────────────
def _sched(n=0, ids=()):
    s = MagicMock()
    s.get_jobs.return_value = [SimpleNamespace(id=i) for i in ids] or [SimpleNamespace(id=f"j{i}") for i in range(n)]
    return s


def _create(monkeypatch, sched, rid="web-1", origin="", **kw):
    import ghost_agent.tools.tasks as T
    monkeypatch.setattr(T, "run_proactive_task_fn", AsyncMock())
    t1, t2 = request_id_context.set(rid), request_origin_context.set(origin)
    try:
        args = dict(action="create", scheduler=sched, task_name="disk check", cron_expression="interval:3600",
                    prompt="check the disk")
        args.update(kw)
        return asyncio.run(T.tool_manage_tasks(**args))
    finally:
        request_id_context.reset(t1)
        request_origin_context.reset(t2)


@pytest.mark.parametrize("rid,origin,action", [("sched-task_1", "", "create"), ("job-abc", "", "create"),
                                               ("sched-task_1", "", "stop_all"), ("probe-1", "", "create"),
                                               ("web-9", ORIGIN_PROBE, "create")])
def test_an_unattended_run_or_a_probe_never_schedules(monkeypatch, rid, origin, action):
    """Fails where a scheduled run could schedule more tasks (or wipe the
    owner's), and a probe left a task firing forever."""
    s = _sched()
    out = _create(monkeypatch, s, rid=rid, origin=origin, action=action)
    assert out.startswith("Error") and not s.add_job.called and not s.remove_all_jobs.called


def test_the_owner_can_still_schedule(monkeypatch):
    s = _sched()
    assert _create(monkeypatch, s).startswith("SUCCESS") and s.add_job.called


def test_a_task_runs_at_most_once_a_minute(monkeypatch):
    """Fails where `interval:1` was accepted (300 tasks fired 600 turns in 2.2 s)."""
    s = _sched()
    assert "at least 60 seconds" in _create(monkeypatch, s, cron_expression="interval:5")
    assert _create(monkeypatch, s, cron_expression="interval:60").startswith("SUCCESS")


def test_the_number_of_tasks_is_bounded(monkeypatch):
    from ghost_agent.tools.tasks import MAX_TASKS
    s = _sched(n=MAX_TASKS)
    assert "limit" in _create(monkeypatch, s) and not s.add_job.called


def test_a_task_name_is_not_silently_replaced(monkeypatch):
    """Fails where the id was the name's hash and `replace_existing` overwrote
    the old prompt with a SUCCESS reply."""
    import hashlib
    jid = f"task_{hashlib.md5(b'disk check').hexdigest()[:10]}"
    s = _sched(ids=[jid])
    assert "already exists" in _create(monkeypatch, s) and not s.add_job.called


def test_a_probe_cannot_confirm_a_deletion():
    from ghost_agent.tools.memory import _not_the_user
    assert _not_the_user("probe-123")
    t = request_origin_context.set(ORIGIN_PROBE)
    try:
        assert _not_the_user("web-5")
    finally:
        request_origin_context.reset(t)
    assert not _not_the_user("web-5")


def test_sub_agents_are_capped_across_calls():
    """Fails where the per-call cap let a turn delegate again and again."""
    from ghost_agent.core.jobs import get_job_registry
    from ghost_agent.tools.delegate import tool_delegate, MAX_RUNNING_DELEGATES
    ctx = SimpleNamespace(llm_client=SimpleNamespace(foreground_requests=0))
    reg = get_job_registry(ctx)
    for i in range(MAX_RUNNING_DELEGATES):
        reg.register("subagent", f"t{i}")
    out = asyncio.run(tool_delegate(task="research X", context=ctx))
    assert "already running" in out


# ── unprompted messages ──────────────────────────────────────────────────────
def test_a_launched_job_is_not_reported_done(monkeypatch, tmp_path):
    """Fails where "notify me when it's done" sent "Done — I've started the
    benchmark as background job job-7f3a" while it ran."""
    from ghost_agent.core import agent as A
    from ghost_agent.core.jobs import get_job_registry
    ctx = SimpleNamespace()
    log = MagicMock()
    log.record.return_value = True
    import ghost_agent.core.autonomous_activity as aa
    monkeypatch.setattr(aa, "get_activity_log", lambda c: log)
    job = get_job_registry(ctx).register("subagent", "run the benchmark")
    tools = [{"name": "delegate", "content": f"Started {job.id} (running in the background)"}]
    fired = A._notify_promise_backstop(ctx, last_user_content="run the benchmark and notify me when it's done",
                                       tools_run=tools, final_content=f"I've started {job.id}.", req_id="web-1",
                                       had_failures=False)
    assert fired is False and not log.record.called
    get_job_registry(ctx).finish(job.id, result="benchmark: 412 req/s")
    assert log.record.called
    msg = log.record.call_args.args[1]
    assert msg.startswith("Done — run the benchmark") and "412 req/s" in msg


def test_a_failed_job_says_so():
    from ghost_agent.core.jobs import JobRegistry, STATUS_FAILED
    reg = JobRegistry()
    got = []
    reg.notifier = got.append
    job = reg.register("subagent", "nightly export")
    job.meta["notify_owner_req"] = "web-1"
    reg.finish(job.id, status=STATUS_FAILED, error="disk full")
    assert got and got[0].status == STATUS_FAILED
    plain = reg.register("subagent", "unwatched")
    reg.finish(plain.id, result="x")
    assert len(got) == 1                     # only the jobs the owner asked about


def test_a_finished_turn_is_still_reported_done(monkeypatch):
    from ghost_agent.core import agent as A
    log = MagicMock()
    log.record.return_value = True
    import ghost_agent.core.autonomous_activity as aa
    monkeypatch.setattr(aa, "get_activity_log", lambda c: log)
    import ghost_agent.tools.notify_tool as nt
    monkeypatch.setattr(nt, "_rate_limited", lambda: False)
    assert A._notify_promise_backstop(SimpleNamespace(), last_user_content="notify me when you're done",
                                      tools_run=[{"name": "web_search", "content": "results"}],
                                      final_content="Postgres 18 adds async I/O.", req_id="web-2",
                                      had_failures=False)
    assert log.record.call_args.args[1].startswith("Done — ")


def test_a_scheduled_task_pages_on_change_and_at_a_bounded_rate():
    """Fails where every fire paged — a task failing every cycle paged
    "FAILED" every cycle."""
    from ghost_agent.core import autonomous_activity as aa
    aa._SCHED_LAST_OK.clear()
    aa._SCHED_PAGES.clear()
    sev = [aa._scheduled_severity("j", False, now=t) for t in (0, 60, 120)]
    assert sev == ["notify", "info", "info"]                     # the transition into failure only
    assert aa._scheduled_severity("j", True, now=180) == "notify"  # recovery
    got = [aa._scheduled_severity("k", True, now=200 + i) for i in range(aa.SCHEDULED_NOTIFY_PER_HOUR + 2)]
    assert got.count("notify") == aa.SCHEDULED_NOTIFY_PER_HOUR
    assert aa._scheduled_severity("k", True, now=200 + 3700) == "notify"   # a new hour


@pytest.mark.parametrize("text,asked", [
    ("Stop pinging me so often, but notify me when the build is done", True),
    ("don't notify me", False), ("run it, no need to notify me", False),
    ("notify me in slack when you're done", True)])
def test_a_negation_cancels_only_its_own_clause(text, asked):
    from ghost_agent.core.agent import _user_asked_for_notification
    assert _user_asked_for_notification(text) is asked


# ── idle phases ──────────────────────────────────────────────────────────────
def test_idle_cooldowns_survive_a_restart(tmp_path):
    """Fails where every boot reset the anchors to `datetime.min` — 37/73
    postmortem runs fell inside their cooldown on a box booted 50×."""
    from ghost_agent.core.agent import _sync_idle_anchors
    (tmp_path / "memory").mkdir()
    ctx = SimpleNamespace(memory_dir=str(tmp_path / "memory"))
    first = SimpleNamespace(_last_postmortem_at=dt.datetime(2026, 10, 4, 1, 0),
                            _last_dream_at=dt.datetime.min)
    _sync_idle_anchors(first, ctx)
    booted = SimpleNamespace(_last_postmortem_at=dt.datetime.min, _last_dream_at=dt.datetime.min)
    _sync_idle_anchors(booted, ctx)
    assert booted._last_postmortem_at == dt.datetime(2026, 10, 4, 1, 0)
    assert booted._last_dream_at == dt.datetime.min


def test_a_failing_idle_phase_is_said_once_an_hour_and_not_called_ran(caplog):
    from ghost_agent.core.agent import _idle_phase_failed
    agent, ran = SimpleNamespace(), ["stale-questions"]
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        _idle_phase_failed(agent, ran, "stale-questions", RuntimeError("db locked"))
        _idle_phase_failed(agent, ran, "stale-questions", RuntimeError("db locked"))
    assert ran == ["stale-questions(failed)"]
    assert sum("stale-questions failed" in r.message for r in caplog.records) == 1


# ── stopping ─────────────────────────────────────────────────────────────────
def test_the_server_stops_within_launchds_kill(monkeypatch):
    """Fails where uvicorn waited for every open request with NO limit and
    launchd's SIGKILL (20 s) landed before the drains, the abort record and
    `sched.shutdown` ever ran (45 of 49 stops)."""
    import ghost_agent.main as M
    seen = {}
    monkeypatch.setattr(M.uvicorn, "run", lambda app, **kw: seen.update(kw))
    M._serve(object(), SimpleNamespace(host="127.0.0.1", port=1))
    assert seen["timeout_graceful_shutdown"] == M._HTTP_SHUTDOWN_GRACE_S
    assert M._HTTP_SHUTDOWN_GRACE_S + M._BIO_SHUTDOWN_GRACE_S <= 12      # leaves the drains time inside 20 s


def test_the_scheduled_result_writer_uses_the_rule():
    from ghost_agent.core import autonomous_activity as aa
    aa._SCHED_LAST_OK.clear()
    aa._SCHED_PAGES.clear()
    log = MagicMock()
    for _ in range(3):
        aa.record_scheduled_result(log, job_id="task_x", task_name="disk check", content="disk full", ok=False)
    sev = [c.kwargs["severity"] for c in log.record.call_args_list]
    assert sev == ["notify", "info", "info"]
