"""Reactive condition-watching scheduler (2026-07-16).

`manage_tasks(action='watch')` registers a task that POLLS a shell condition
on an interval and fires its reaction only when the condition first becomes
true (edge-triggered), instead of firing on a clock like 'create'.
"""
import asyncio
import hashlib
import tempfile
from pathlib import Path

import pytest

from ghost_agent.tools import tasks as T


class _FakeSched:
    def __init__(self):
        self.jobs = {}

    def add_job(self, fn, trigger, seconds=None, args=None, id=None,
                name=None, replace_existing=None, **kw):
        self.jobs[id] = {"fn": fn, "args": args, "name": name,
                         "seconds": seconds, "trigger": trigger}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "task_store_path", str(tmp_path / "sched.json"))
    monkeypatch.setattr(T, "run_proactive_task_fn", lambda j, p: None)
    monkeypatch.setattr(T, "run_watch_condition_fn", lambda j: None)
    return tmp_path


def _wid(name):
    return "watch_" + hashlib.md5(name.encode()).hexdigest()[:10]


def _run(coro):
    return asyncio.run(coro)


class TestWatchRegistration:
    def test_registers_and_persists(self, store):
        sched = _FakeSched()
        r = _run(T.tool_manage_tasks(
            action="watch", scheduler=sched, memory_system=None,
            task_name="net-down", check_command="! curl -sf http://x/health",
            prompt="tell me the net is down", interval_secs=30))
        assert r.startswith("SUCCESS")
        jid = _wid("net-down")
        assert jid in sched.jobs and sched.jobs[jid]["seconds"] == 30
        rec = T.get_watch_record(jid)
        assert rec["kind"] == "watch"
        assert rec["check_command"] == "! curl -sf http://x/health"
        assert rec["prompt"] == "tell me the net is down"
        assert rec["last_fired"] is False

    def test_missing_params_rejected(self, store):
        sched = _FakeSched()
        r = _run(T.tool_manage_tasks(action="watch", scheduler=sched,
                                     task_name="x", interval_secs=30))
        assert r.startswith("Error") and "check_command" in r

    def test_watch_rejects_cron_expression(self, store):
        # A watch polls — it must use interval:N, not a cron expression.
        err = T._add_job(_FakeSched(), _wid("w"), "w", "react", "0 9 * * *",
                         kind="watch", check_command="true")
        assert err and "interval" in err.lower()

    def test_watch_rejects_too_frequent(self, store):
        err = T._add_job(_FakeSched(), _wid("w"), "w", "react", "interval:5",
                         kind="watch", check_command="true")
        assert err and ">= 10" in err

    def test_uninitialized_runner_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(T, "task_store_path", str(tmp_path / "s.json"))
        monkeypatch.setattr(T, "run_watch_condition_fn", None)
        err = T._add_job(_FakeSched(), "w", "w", "react", "interval:60",
                         kind="watch", check_command="true")
        assert err and "not initialized" in err


class TestEdgeState:
    def test_set_and_get_last_fired(self, store):
        sched = _FakeSched()
        _run(T.tool_manage_tasks(
            action="watch", scheduler=sched, memory_system=None,
            task_name="w", check_command="true", prompt="p", interval_secs=60))
        jid = _wid("w")
        assert T.get_watch_record(jid)["last_fired"] is False
        T.set_watch_state(jid, True)
        assert T.get_watch_record(jid)["last_fired"] is True
        T.set_watch_state(jid, False)
        assert T.get_watch_record(jid)["last_fired"] is False

    def test_absent_record_is_empty(self, store):
        assert T.get_watch_record("nope") == {}


class TestRestore:
    def test_watch_survives_restart(self, store):
        sched = _FakeSched()
        _run(T.tool_manage_tasks(
            action="watch", scheduler=sched, memory_system=None,
            task_name="net-down", check_command="! curl -sf http://x/health",
            prompt="react", interval_secs=30))
        _run(T.tool_manage_tasks(
            action="create", scheduler=sched, memory_system=None,
            task_name="daily", cron_expression="interval:3600", prompt="digest"))
        fresh = _FakeSched()
        n = T.restore_persisted_tasks(fresh)
        assert n == 2
        assert _wid("net-down") in fresh.jobs   # watch re-registered
        # the watch job points at the watch runner, not the proactive one
        assert fresh.jobs[_wid("net-down")]["fn"] is T.run_watch_condition_fn
