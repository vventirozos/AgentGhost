"""Unit tests for memory/projects.py — the persistent project store."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
import sqlite3
import threading
import time
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


def test_update_project_merges_metadata_shallow(store):
    # MERGE semantics (2026-07-20): the blob carries system state
    # (design_ledger, config, budgets) — a whole-dict replace wiped it.
    pid = store.create_project("X", metadata={"a": 1})
    store.update_project(pid, metadata={"b": 2})
    assert store.get_project(pid)["metadata"] == {"a": 1, "b": 2}


def test_update_project_metadata_budget_raise_preserves_system_state(store):
    # The documented budget-raise (metadata={"steps_cap": 100}) must not
    # destroy the ledger/config/counter siblings that live in the same blob.
    pid = store.create_project("X")
    store.append_ledger(pid, "src/app.py: entrypoint")
    store.set_config_value(pid, "PORT", "8080")
    store.update_project(pid, metadata={"steps_used": 40, "defect_reopens": 2})
    store.update_project(pid, metadata={"steps_cap": 100})
    meta = store.get_project(pid)["metadata"]
    assert meta["steps_cap"] == 100
    assert meta["steps_used"] == 40
    assert meta["defect_reopens"] == 2
    assert meta["design_ledger"] == "src/app.py: entrypoint"
    assert meta["config"] == {"PORT": "8080"}


def test_update_project_metadata_replace_opt_in(store):
    pid = store.create_project("X", metadata={"a": 1, "b": 2})
    store.update_project(pid, metadata={"c": 3}, metadata_replace=True)
    assert store.get_project(pid)["metadata"] == {"c": 3}


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


def test_hard_delete_removes_workspace_dir(store):
    pid = store.create_project("X")
    ws = store.sandbox_root / "projects" / pid
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "a.txt").write_text("x")
    assert store.delete_project(pid, hard=True) is True
    assert not ws.exists()  # on-disk files removed too


def test_hard_delete_never_escapes_sandbox_root(store, tmp_path):
    """A custom workspace_dir OUTSIDE the sandbox root is left untouched —
    hard delete only removes paths strictly inside the root."""
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "important.txt").write_text("keep me")
    pid = store.create_project("X", workspace_dir=str(outside))
    assert store.delete_project(pid, hard=True) is True
    assert outside.exists() and (outside / "important.txt").exists()


def test_soft_delete_keeps_workspace_dir(store):
    pid = store.create_project("X")
    ws = store.sandbox_root / "projects" / pid
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "a.txt").write_text("x")
    store.delete_project(pid, hard=False)  # archive
    assert (ws / "a.txt").exists()  # preserved


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


# --------------------------------------------------------------- 2026-07-20 fixes

def _set_project_status_raw(store, pid, status):
    """Simulate a concurrent writer flipping status outside the API."""
    with store._lock, store._connect() as conn:
        conn.execute("UPDATE projects SET status = ? WHERE id = ?", (status, pid))
        conn.commit()


def test_reaper_resets_stale_in_progress(store):
    pid = store.create_project("P")
    stale = store.add_task(pid, "stale", status="IN_PROGRESS")
    fresh = store.add_task(pid, "fresh", status="IN_PROGRESS")
    open_task = store.add_task(pid, "open")
    with store._lock, store._connect() as conn:
        conn.execute("UPDATE tasks SET updated_at = ? WHERE id = ?",
                     (time.time() - 3600, stale))
        conn.commit()
    count = store.reset_orphaned_in_progress(older_than_seconds=600)
    assert count == 1
    assert store.get_task(stale)["status"] == "READY"
    assert store.get_task(fresh)["status"] == "IN_PROGRESS"
    assert store.get_task(open_task)["status"] == "PENDING"
    evs = store.list_events(pid, event_type="task_reset_orphaned")
    assert len(evs) == 1
    assert evs[0]["task_id"] == stale
    assert evs[0]["payload"]["to"] == "READY"


def test_reaper_safe_on_empty_store(store):
    # Boot-time call with nothing in flight must be a clean no-op.
    assert store.reset_orphaned_in_progress() == 0


def test_rollup_noops_when_archive_interleaves(store, monkeypatch):
    # A manual ARCHIVE landing between the rollup's read and its write must
    # not be stomped back to DONE (which would fire the destructive cleanup
    # on an archived project).
    pid = store.create_project("P")
    tid = store.add_task(pid, "t")
    fired = []
    store.on_project_done = fired.append

    orig_list_tasks = store.list_tasks

    def racy_list_tasks(project_id, status_filter=None):
        rows = orig_list_tasks(project_id, status_filter)
        _set_project_status_raw(store, pid, "ARCHIVED")
        return rows

    monkeypatch.setattr(store, "list_tasks", racy_list_tasks)
    store.update_task(tid, status="DONE")  # triggers the rollup
    assert store.get_project(pid)["status"] == ProjectStatus.ARCHIVED.value
    assert fired == []
    assert not store.list_events(pid, event_type="project_auto_rollup")


def test_add_task_reopens_failed_project(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "t")
    store.update_task(tid, status="FAILED")
    assert store.get_project(pid)["status"] == ProjectStatus.FAILED.value
    store.add_task(pid, "retry the build")
    assert store.get_project(pid)["status"] == ProjectStatus.ACTIVE.value
    evs = store.list_events(pid, event_type="project_reopened")
    assert evs and evs[0]["payload"]["from_status"] == "FAILED"


def test_add_task_reopens_paused_project(store):
    pid = store.create_project("P")
    store.add_task(pid, "t")
    store.update_project(pid, status="PAUSED")
    store.add_task(pid, "new work")
    assert store.get_project(pid)["status"] == ProjectStatus.ACTIVE.value


def test_add_task_never_reopens_archived_or_needs_user(store):
    a = store.create_project("A")
    store.add_task(a, "t")
    store.update_project(a, status="ARCHIVED")
    store.add_task(a, "more")
    assert store.get_project(a)["status"] == ProjectStatus.ARCHIVED.value

    n = store.create_project("N")
    tid = store.add_task(n, "ask user")
    store.update_task(tid, status="NEEDS_USER")
    assert store.get_project(n)["status"] == ProjectStatus.NEEDS_USER.value
    store.add_task(n, "more")
    assert store.get_project(n)["status"] == ProjectStatus.NEEDS_USER.value


def test_update_task_reopen_reactivates_failed_project(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "t")
    store.update_task(tid, status="FAILED")
    assert store.get_project(pid)["status"] == ProjectStatus.FAILED.value
    # Reviving the failed task must give the project a path back to ACTIVE.
    store.update_task(tid, status="PENDING")
    assert store.get_project(pid)["status"] == ProjectStatus.ACTIVE.value


def test_delete_task_of_last_open_triggers_rollup(store):
    pid = store.create_project("P")
    t1 = store.add_task(pid, "done work")
    t2 = store.add_task(pid, "abandoned")
    store.update_task(t1, status="DONE")
    assert store.get_project(pid)["status"] == ProjectStatus.ACTIVE.value
    store.delete_task(t2)
    assert store.get_project(pid)["status"] == ProjectStatus.DONE.value


def test_delete_task_leaving_no_tasks_does_not_complete_project(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "only")
    store.delete_task(tid)
    assert store.get_project(pid)["status"] == ProjectStatus.ACTIVE.value


def test_register_file_artifact_normalizes_absolute_sandbox_path(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "t")
    aid = store.register_file_artifact(tid, f"/workspace/projects/{pid}/out/x.png")
    assert aid is not None
    arts = store.list_artifacts(project_id=pid)
    assert [a["payload"] for a in arts if a["kind"] == "file"] == ["out/x.png"]
    # The bare relative form dedupes against the absolute registration.
    assert store.register_file_artifact(tid, "out/x.png") == aid
    assert store.register_file_artifact(tid, f"projects/{pid}/out/x.png") == aid
    assert store.register_file_artifact(tid, "workspace/projects/" + pid + "/out/x.png") == aid
    assert len(store.list_deliverables(pid)) == 1


def test_register_file_artifact_rejects_traversal(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "t")
    assert store.register_file_artifact(tid, "../../etc/passwd") is None
    assert store.register_file_artifact(tid, f"projects/{pid}/../other/x") is None
    assert store.list_deliverables(pid) == []


def test_list_events_clamps_negative_and_huge_limits(store):
    pid = store.create_project("P")
    for i in range(5):
        store.log_event(pid, None, "note", {"i": i})
    # Negative limit is "no limit" to SQLite — must not dump the whole log.
    assert len(store.list_events(pid, limit=-1, event_type="note")) == 1
    assert len(store.list_events(pid, limit=0, event_type="note")) == 1
    assert len(store.list_events(pid, limit=10**9, event_type="note")) == 5
    assert store.EVENTS_MAX_LIMIT >= 50
