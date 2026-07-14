"""Scheduled-task persistence — 2026-07-14 audit.

The AsyncIOScheduler jobstore is IN-MEMORY and the operator deploys by
killing the agent, so every deploy silently wiped all user cron tasks —
while the "task X is running" vector-memory note kept asserting they were
alive. Scheduled tasks run with nobody watching, so the wipe was invisible.
These tests pin the JSON store contract: create persists, stop/stop_all
unpersist, and boot-time restore re-registers exactly what was stored
(dropping malformed records without aborting the rest).
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import MagicMock

from ghost_agent.tools import tasks as tasks_mod
from ghost_agent.tools.tasks import (
    restore_persisted_tasks,
    tool_list_tasks,
    tool_schedule_task,
    tool_stop_all_tasks,
    tool_stop_task,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def store(tmp_path, monkeypatch):
    p = tmp_path / "system" / "scheduled_tasks.json"
    monkeypatch.setattr(tasks_mod, "task_store_path", p)
    monkeypatch.setattr(tasks_mod, "run_proactive_task_fn", lambda *a: None)
    return p


def _sched():
    m = MagicMock()
    m.get_jobs.return_value = []
    return m


def _stored(p):
    return json.loads(p.read_text())["tasks"]


async def test_create_persists(store):
    sched = _sched()
    out = await tool_schedule_task("daily brief", "summarize the news",
                                   "interval:3600", sched, None)
    assert out.startswith("SUCCESS")
    recs = _stored(store)
    assert len(recs) == 1
    rec = next(iter(recs.values()))
    assert rec["task_name"] == "daily brief"
    assert rec["prompt"] == "summarize the news"
    assert rec["cron_expression"] == "interval:3600"


async def test_malformed_schedule_not_persisted(store):
    sched = _sched()
    out = await tool_schedule_task("bad", "p", "interval:5m", sched, None)
    assert out.startswith("Error")
    assert not store.exists() or _stored(store) == {}
    sched.add_job.assert_not_called()


async def test_stop_unpersists_only_that_job(store):
    sched = _sched()
    await tool_schedule_task("keep", "p1", "interval:60", sched, None)
    await tool_schedule_task("drop", "p2", "interval:60", sched, None)
    assert len(_stored(store)) == 2

    drop_id = next(j for j, r in _stored(store).items()
                   if r["task_name"] == "drop")
    job = MagicMock()
    job.id = drop_id
    job.name = "drop"
    sched.get_jobs.return_value = [job]
    out = await tool_stop_task("drop", sched)
    assert out.startswith("SUCCESS")
    remaining = _stored(store)
    assert len(remaining) == 1
    assert next(iter(remaining.values()))["task_name"] == "keep"


async def test_stop_all_unpersists_everything(store):
    sched = _sched()
    await tool_schedule_task("a", "p", "interval:60", sched, None)
    await tool_schedule_task("b", "p", "interval:60", sched, None)
    job = MagicMock()
    sched.get_jobs.return_value = [job, job]
    out = await tool_stop_all_tasks(sched)
    assert out.startswith("SUCCESS")
    assert _stored(store) == {}


async def test_restore_reregisters_persisted_tasks(store):
    sched = _sched()
    await tool_schedule_task("interval task", "p1", "interval:300", sched, None)
    await tool_schedule_task("cron task", "p2", "0 9 * * *", sched, None)

    fresh = _sched()  # a new scheduler, as after a restart
    n = restore_persisted_tasks(fresh)
    assert n == 2
    assert fresh.add_job.call_count == 2
    names = {c.kwargs.get("name") for c in fresh.add_job.call_args_list}
    assert names == {"interval task", "cron task"}


async def test_restore_drops_malformed_record_keeps_rest(store):
    sched = _sched()
    await tool_schedule_task("good", "p", "interval:60", sched, None)
    recs = _stored(store)
    recs["task_rotten"] = {"task_name": "rotten", "prompt": "p",
                           "cron_expression": "interval:banana"}
    store.write_text(json.dumps({"tasks": recs}))

    fresh = _sched()
    n = restore_persisted_tasks(fresh)
    assert n == 1
    # The rotten record is dropped from the store, the good one kept.
    after = _stored(store)
    assert "task_rotten" not in after
    assert len(after) == 1


async def test_no_store_path_is_clean_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks_mod, "task_store_path", None)
    monkeypatch.setattr(tasks_mod, "run_proactive_task_fn", lambda *a: None)
    sched = _sched()
    out = await tool_schedule_task("x", "p", "interval:60", sched, None)
    assert out.startswith("SUCCESS")           # scheduling still works
    assert restore_persisted_tasks(sched) == 0


async def test_corrupt_store_reads_as_empty(store):
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("{not json")
    assert restore_persisted_tasks(_sched()) == 0  # no raise


async def test_list_declares_utc(store):
    sched = _sched()
    job = MagicMock()
    job.id = "task_abc"
    job.name = "morning"
    job.next_run_time = "2026-07-15 06:00:00+00:00"
    sched.get_jobs.return_value = [job]
    out = await tool_list_tasks(sched)
    assert "UTC" in out
    assert "morning" in out
