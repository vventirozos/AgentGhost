"""The sandbox-job reaper stops on request at shutdown: at once when asleep, after its pass when
working, and never starts a wake turn once asked."""
import ast
import asyncio
import inspect
import time
import types
from unittest.mock import patch

import pytest

import ghost_agent.main as M
import ghost_agent.sandbox.jobs as J
import ghost_agent.tools.delegate as D
import ghost_agent.tools.runs as R
import ghost_agent.utils.logging as glog

pytestmark = pytest.mark.asyncio


def _ctx():
    return types.SimpleNamespace(sandbox_manager=object(), _reaper_stop=asyncio.Event())


async def test_asleep_it_exits_at_once_and_the_drain_is_silent(monkeypatch):
    monkeypatch.setattr(M, "_SANDBOX_JOB_REAP_EVERY_S", 60)
    ctx = _ctx()
    task = glog.spawn_bg(M._reap_sandbox_jobs(ctx), name="sandbox-job-reaper")
    await asyncio.sleep(0.02)
    ctx._reaper_stop.set()
    t0 = time.monotonic()
    with patch.object(glog, "pretty_log") as plog:
        await glog.drain_background_tasks(timeout=3.0)
    assert task.done() and not task.cancelled() and time.monotonic() - t0 < 1.0
    assert not plog.called


def _sup(monkeypatch, wakes, on_take=lambda: None):
    class _Sup:
        def reap(self):
            return []

        def take_unreported(self):
            on_take()
            return [{"id": "j1", "state": "done", "command": "x"}]

        def pending_wakes(self):
            return [{"id": "j1"}, {"id": "j2"}]

        def mark_woken(self, _jid):
            pass
    recorded = []
    monkeypatch.setattr(J, "get_job_supervisor", lambda m: _Sup())
    monkeypatch.setattr(R, "end_landed_job_runs", lambda c, ch: None)
    monkeypatch.setattr(D, "record_landings", lambda c, ch: recorded.append(len(ch)))
    monkeypatch.setattr(M, "_collect_hint", lambda e, c: "jobs", raising=False)
    monkeypatch.setattr(M, "_SANDBOX_JOB_REAP_EVERY_S", 0.01)

    async def _wake(_c, e):
        wakes.append(e["id"])
        return True
    monkeypatch.setattr(M, "_resume_after_job", _wake)
    return recorded


async def test_mid_pass_it_finishes_the_write_and_starts_no_wake(monkeypatch):
    ctx, wakes = _ctx(), []
    recorded = _sup(monkeypatch, wakes, on_take=lambda: time.sleep(0.2))
    task = asyncio.ensure_future(M._reap_sandbox_jobs(ctx))
    await asyncio.sleep(0.1)                          # inside take_unreported
    ctx._reaper_stop.set()
    await asyncio.wait_for(task, 2)
    assert recorded == [1] and wakes == []


async def test_without_a_stop_the_wakes_run(monkeypatch):
    ctx, wakes = _ctx(), []
    _sup(monkeypatch, wakes)
    task = asyncio.ensure_future(M._reap_sandbox_jobs(ctx))
    for _ in range(100):
        await asyncio.sleep(0.01)
        if len(wakes) >= 2:
            break
    ctx._reaper_stop.set()
    await asyncio.wait_for(task, 2)
    assert wakes[:2] == ["j1", "j2"]


async def test_a_cancel_while_asleep_still_cancels(monkeypatch):
    monkeypatch.setattr(M, "_SANDBOX_JOB_REAP_EVERY_S", 60)
    task = asyncio.ensure_future(M._reap_sandbox_jobs(_ctx()))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_stop_during_the_first_wake_prevents_the_second(monkeypatch):
    ctx, wakes = _ctx(), []
    _sup(monkeypatch, wakes)

    async def _wake(_c, e):
        wakes.append(e["id"])
        ctx._reaper_stop.set()
        return True
    monkeypatch.setattr(M, "_resume_after_job", _wake)
    await asyncio.wait_for(M._reap_sandbox_jobs(ctx), 2)
    assert wakes == ["j1"]


async def test_the_pause_leaks_no_waiter():
    stop = asyncio.Event()
    before = len(asyncio.all_tasks())
    for _ in range(5):
        assert await M._reaper_pause(stop, 0.001) is False
    await asyncio.sleep(0.01)
    assert len(asyncio.all_tasks()) == before


async def test_a_context_without_a_real_event_keeps_the_plain_sleep():
    from unittest.mock import MagicMock
    ctx = MagicMock()
    assert await M._reaper_pause(ctx._reaper_stop, 0.001) is False
    assert M._reaper_stopping(ctx) is False
    M._ask_reaper_to_stop(ctx)


async def test_arm_and_ask_meet_on_the_same_object():
    ctx = types.SimpleNamespace()
    M._ask_reaper_to_stop(ctx)                        # never armed: nothing to do
    assert M._reaper_stopping(ctx) is False
    M._arm_reaper_stop(ctx)
    assert M._reaper_stopping(ctx) is False
    M._ask_reaper_to_stop(ctx)
    assert M._reaper_stopping(ctx) and ctx._reaper_stop.is_set()


async def test_the_lifespan_arms_before_the_spawn_and_asks_first_in_its_finally():
    tree = ast.parse(inspect.getsource(M))

    def _is(n, name):
        return (isinstance(n, ast.Call) and getattr(n.func, "id", "") == name
                and [getattr(a, "id", "") for a in n.args] == ["context"])
    life = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
            and any(_is(c, "_arm_reaper_stop") for c in ast.walk(n))]
    assert [n.name for n in life] == ["lifespan"]
    arm = next(c for c in ast.walk(life[0]) if _is(c, "_arm_reaper_stop"))
    spawn = next(c for c in ast.walk(life[0]) if isinstance(c, ast.Call)
                 and any(k.arg == "name" and getattr(k.value, "value", "") == "sandbox-job-reaper"
                         for k in c.keywords))
    assert arm.lineno < spawn.lineno and _is(spawn.args[0], "_reap_sandbox_jobs")
    tries = [t for t in ast.walk(life[0]) if isinstance(t, ast.Try)
             and any(isinstance(s, ast.Expr) and isinstance(s.value, ast.Yield) for s in t.body)]
    assert len(tries) == 1
    fb = tries[0].finalbody                           # [0] is the synchronous "System Shutdown" line
    assert "System Shutdown" in ast.dump(fb[0]) and _is(fb[1].value, "_ask_reaper_to_stop")
    assert not any(isinstance(n, ast.Await) for n in ast.walk(fb[0]))


async def test_a_wake_being_prepared_when_shutdown_starts_runs_no_turn_and_is_not_stamped(monkeypatch):
    from unittest.mock import AsyncMock
    ctx = _ctx()
    ctx.args = types.SimpleNamespace(model="m")
    ctx.llm_client = None
    ctx.activity_log = None
    ctx.agent = types.SimpleNamespace(handle_chat=AsyncMock(return_value=("x", None, None)))
    import ghost_agent.tools.tasks as TT
    monkeypatch.setattr(TT, "should_defer_scheduled_task", lambda c: False)
    monkeypatch.setattr(J, "get_job_supervisor", lambda m: None)
    monkeypatch.setattr(M, "_RESUMED_JOBS", set())
    monkeypatch.setattr(M, "_resume_times", [])
    import ghost_agent.api.routes as AR
    monkeypatch.setattr(AR, "_mark_foreground", lambda *a: None)

    def _scope(_c, _p):                               # the LAST await before handle_chat
        ctx._reaper_stop.set()
        return None, ""
    monkeypatch.setattr(M, "_project_turn_scope", _scope)
    entry = {"id": "job-0000aaaa", "state": "done", "exit_code": 0, "command": "x"}
    assert await M._resume_after_job(ctx, entry) is False
    assert not ctx.agent.handle_chat.called
    assert M._RESUMED_JOBS == set() and M._resume_times == []
    # control: without a stop the same wake runs its turn
    ctx._reaper_stop = asyncio.Event()
    monkeypatch.setattr(M, "_project_turn_scope", lambda c, p: (None, ""))
    assert await M._resume_after_job(ctx, entry) is True
    assert ctx.agent.handle_chat.called
