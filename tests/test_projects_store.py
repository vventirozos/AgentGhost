"""Unit tests for memory/projects.py — the persistent project store."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from ghost_agent.memory.projects import (
    ProjectStore, ProjectKind, ProjectStatus,
)


@pytest.fixture
def store(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    memdir = tmp_path / "memory"
    return ProjectStore(memdir, sandbox_root=sandbox)


# --------------------------------------------------------------------- projects

def test_create_project_persists_row_and_logs_event(store, tmp_path):
    pid = store.create_project("Build CLI", kind="CODING", goal="Ship v1")
    proj = store.get_project(pid)
    assert proj is not None
    assert proj["title"] == "Build CLI"
    assert proj["kind"] == ProjectKind.CODING.value
    assert proj["status"] == ProjectStatus.ACTIVE.value
    assert proj["goal"] == "Ship v1"
    assert proj["metadata"] == {}
    events = store.list_events(pid)
    assert any(e["type"] == "project_created" for e in events)


def test_create_project_rejects_empty_title(store):
    with pytest.raises(ValueError):
        store.create_project("", kind="GENERAL")
    with pytest.raises(ValueError):
        store.create_project("   ", kind="GENERAL")


def test_create_project_rejects_bad_kind(store):
    with pytest.raises(ValueError):
        store.create_project("Oops", kind="NOT_A_KIND")


def test_create_project_creates_workspace_dir(store, tmp_path):
    pid = store.create_project("Web scraper", kind="CODING")
    proj = store.get_project(pid)
    wsd = proj["workspace_dir"]
    assert wsd is not None
    assert Path(wsd).exists()
    assert pid in wsd


def test_create_project_without_sandbox_root_has_no_workspace(tmp_path):
    s = ProjectStore(tmp_path / "mem", sandbox_root=None)
    pid = s.create_project("No WS")
    proj = s.get_project(pid)
    assert proj["workspace_dir"] is None


def test_create_project_with_explicit_workspace_dir(store, tmp_path):
    custom = str(tmp_path / "custom_ws")
    pid = store.create_project("Custom", workspace_dir=custom)
    proj = store.get_project(pid)
    assert proj["workspace_dir"] == custom
    assert Path(custom).exists()


def test_create_project_with_metadata(store):
    meta = {"budget": 100, "owner": "alice"}
    pid = store.create_project("Meta Proj", metadata=meta)
    proj = store.get_project(pid)
    assert proj["metadata"] == meta


def test_list_projects_orders_by_updated_desc(store):
    a = store.create_project("A")
    b = store.create_project("B")
    c = store.create_project("C")
    # Touch A so it becomes most recent
    store.update_project(a, goal="bumped")
    ids = [p["id"] for p in store.list_projects()]
    assert ids[0] == a
    assert set(ids) == {a, b, c}


def test_list_projects_filters_by_status(store):
    active = store.create_project("Active")
    archived = store.create_project("Archived")
    store.update_project(archived, status=ProjectStatus.ARCHIVED.value)
    actives = store.list_projects(status_filter="ACTIVE")
    assert [p["id"] for p in actives] == [active]
    archs = store.list_projects(status_filter="ARCHIVED")
    assert [p["id"] for p in archs] == [archived]


def test_update_project_changes_fields_and_logs(store):
    pid = store.create_project("X")
    assert store.update_project(pid, goal="new goal", status="PAUSED")
    proj = store.get_project(pid)
    assert proj["goal"] == "new goal"
    assert proj["status"] == ProjectStatus.PAUSED.value
    evs = store.list_events(pid, event_type="project_updated")
    assert evs and "goal" in evs[0]["payload"]["fields"]


def test_update_project_rejects_unknown_field(store):
    pid = store.create_project("X")
    with pytest.raises(ValueError):
        store.update_project(pid, does_not_exist="x")


def test_update_project_returns_false_when_missing(store):
    assert store.update_project("missing", goal="x") is False


def test_update_project_merges_metadata_as_replacement(store):
    pid = store.create_project("X", metadata={"a": 1})
    store.update_project(pid, metadata={"b": 2})
    # Replacement semantics (not merge) — keeps the store simple
    assert store.get_project(pid)["metadata"] == {"b": 2}


def test_soft_delete_marks_archived(store):
    pid = store.create_project("X")
    assert store.delete_project(pid) is True
    assert store.get_project(pid)["status"] == ProjectStatus.ARCHIVED.value


def test_hard_delete_removes_row_and_cascades(store):
    pid = store.create_project("X")
    tid = store.add_task(pid, "root")
    store.add_artifact(tid, "note", "hello")
    assert store.delete_project(pid, hard=True) is True
    assert store.get_project(pid) is None
    assert store.get_task(tid) is None


def test_ensure_workspace_creates_directory(store, tmp_path):
    pid = store.create_project("WS")
    proj = store.get_project(pid)
    # Delete the dir so ensure_workspace has to recreate it
    wsd = Path(proj["workspace_dir"])
    if wsd.exists():
        wsd.rmdir()
    out = store.ensure_workspace(pid)
    assert out is not None and out.exists()


def test_ensure_workspace_returns_none_when_missing(tmp_path):
    s = ProjectStore(tmp_path / "mem", sandbox_root=None)
    pid = s.create_project("X")
    assert s.ensure_workspace(pid) is None


# --------------------------------------------------------------------- tasks

def test_add_task_root(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Do the thing")
    t = store.get_task(tid)
    assert t["description"] == "Do the thing"
    assert t["project_id"] == pid
    assert t["parent_id"] is None
    assert t["depth"] == 0
    assert t["position"] == 0
    assert t["status"] == "PENDING"


def test_add_task_subtask_increments_depth_and_position(store):
    pid = store.create_project("P")
    root = store.add_task(pid, "root")
    c1 = store.add_task(pid, "child1", parent_id=root)
    c2 = store.add_task(pid, "child2", parent_id=root)
    gc = store.add_task(pid, "grand", parent_id=c1)
    assert store.get_task(c1)["depth"] == 1
    assert store.get_task(c1)["position"] == 0
    assert store.get_task(c2)["position"] == 1
    assert store.get_task(gc)["depth"] == 2


def test_add_task_rejects_empty_description(store):
    pid = store.create_project("P")
    with pytest.raises(ValueError):
        store.add_task(pid, "")


def test_add_task_rejects_missing_parent(store):
    pid = store.create_project("P")
    with pytest.raises(ValueError):
        store.add_task(pid, "x", parent_id="no-such-id")


def test_add_task_rejects_cross_project_parent(store):
    p1 = store.create_project("P1")
    p2 = store.create_project("P2")
    t1 = store.add_task(p1, "p1 root")
    with pytest.raises(ValueError):
        store.add_task(p2, "bad", parent_id=t1)


def test_list_tasks_order_is_depth_then_position(store):
    pid = store.create_project("P")
    r = store.add_task(pid, "r")
    a = store.add_task(pid, "a", parent_id=r)
    b = store.add_task(pid, "b", parent_id=r)
    descs = [t["description"] for t in store.list_tasks(pid)]
    assert descs == ["r", "a", "b"]


def test_list_tasks_filters_by_status(store):
    pid = store.create_project("P")
    t1 = store.add_task(pid, "t1")
    t2 = store.add_task(pid, "t2")
    store.update_task(t2, status="DONE")
    done = store.list_tasks(pid, status_filter="DONE")
    assert [t["id"] for t in done] == [t2]


def test_update_task_persists_and_logs(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "x")
    assert store.update_task(tid, status="DONE", result_summary="ok",
                             actual_tool_used="execute")
    t = store.get_task(tid)
    assert t["status"] == "DONE"
    assert t["result_summary"] == "ok"
    assert t["actual_tool_used"] == "execute"
    evs = store.list_events(pid, event_type="task_updated")
    assert evs


def test_update_task_lists_are_json_round_tripped(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "x")
    store.update_task(tid, alternatives=["alt1", "alt2"],
                      postconditions=["file exists", "exit_code==0"])
    t = store.get_task(tid)
    assert t["alternatives"] == ["alt1", "alt2"]
    assert t["postconditions"] == ["file exists", "exit_code==0"]


def test_update_task_rejects_unknown_field(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "x")
    with pytest.raises(ValueError):
        store.update_task(tid, nope=1)


def test_delete_task_cascades_descendants(store):
    pid = store.create_project("P")
    r = store.add_task(pid, "r")
    c = store.add_task(pid, "c", parent_id=r)
    gc = store.add_task(pid, "gc", parent_id=c)
    assert store.delete_task(r) is True
    assert store.get_task(r) is None
    assert store.get_task(c) is None
    assert store.get_task(gc) is None


def test_delete_task_missing_returns_false(store):
    assert store.delete_task("nope") is False


# --------------------------------------------------------------------- artifacts

def test_add_artifact_and_list(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "t")
    a1 = store.add_artifact(tid, "note", "first note")
    a2 = store.add_artifact(tid, "url", "https://example.com")
    arts = store.list_artifacts(task_id=tid)
    assert [a["id"] for a in arts] == [a1, a2]
    assert arts[0]["kind"] == "note"
    assert arts[1]["payload"] == "https://example.com"


def test_add_artifact_rejects_bad_kind(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "t")
    with pytest.raises(ValueError):
        store.add_artifact(tid, "binary", "...")


def test_add_artifact_rejects_missing_task(store):
    with pytest.raises(ValueError):
        store.add_artifact("no-task", "note", "x")


def test_list_artifacts_by_project(store):
    pid = store.create_project("P")
    t1 = store.add_task(pid, "t1")
    t2 = store.add_task(pid, "t2")
    store.add_artifact(t1, "note", "a")
    store.add_artifact(t2, "note", "b")
    arts = store.list_artifacts(project_id=pid)
    assert len(arts) == 2


def test_list_artifacts_requires_scope(store):
    with pytest.raises(ValueError):
        store.list_artifacts()


# --------------------------------------------------------------------- events

def test_log_event_and_list(store):
    pid = store.create_project("P")
    store.log_event(pid, None, "custom", {"x": 1})
    evs = store.list_events(pid, event_type="custom")
    assert len(evs) == 1
    assert evs[0]["payload"] == {"x": 1}


def test_list_events_limit(store):
    pid = store.create_project("P")
    for i in range(10):
        store.log_event(pid, None, "note", {"i": i})
    evs = store.list_events(pid, limit=3, event_type="note")
    assert len(evs) == 3
    # newest first
    assert evs[0]["payload"]["i"] == 9


# --------------------------------------------------------------------- concurrency & persistence

def test_store_persists_across_instances(tmp_path):
    mem = tmp_path / "mem"
    s1 = ProjectStore(mem)
    pid = s1.create_project("Persist")
    t = s1.add_task(pid, "root")
    s2 = ProjectStore(mem)
    assert s2.get_project(pid) is not None
    assert s2.get_task(t) is not None


def test_concurrent_task_additions_are_safe(store):
    pid = store.create_project("P")
    ids: list = []

    def worker(n):
        for i in range(5):
            ids.append(store.add_task(pid, f"t-{n}-{i}"))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(ids) == 20
    assert len(set(ids)) == 20
    tasks = store.list_tasks(pid)
    # All 20 are root tasks with distinct positions
    positions = sorted(t["position"] for t in tasks)
    assert positions == list(range(20))


def test_schema_uses_foreign_keys(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "t")
    # Hard-delete project → tasks should cascade
    store.delete_project(pid, hard=True)
    assert store.get_task(tid) is None
