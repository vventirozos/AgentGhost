"""Tests for the project-management hardening pass.

Covers, in order:
  * ProjectStatus terminal states (FAILED / BLOCKED / NEEDS_USER) and the
    corrected status rollup (a wholly-failed project no longer reports DONE).
  * Durable alternatives/fallback edges (ProjectPlan.update_status now
    persists re-parenting + alternative consumption).
  * Sibling-DAG prerequisites (`depends_on`) end-to-end: store column,
    migration, next_ready_leaf gating, sequential decompose, tool surface.
  * SQLite WAL + busy_timeout pragmas.
  * scratchpad_snapshot event pruning.
  * Advancer hardening: round-robin stamp + stricter completion.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import sqlite3
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore, ProjectStatus
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.core.planning import ProjectPlan, TaskStatus, TaskTree
from ghost_agent.core.project_advancer import (
    advance_once, _looks_like_failure,
)
from ghost_agent.tools.projects import tool_manage_projects


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def context(tmp_path, store):
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None,
        contradiction_log=None,
        current_project_id=None,
    )


# ===================================================================== status

def test_project_status_terminal_members_exist():
    assert ProjectStatus.FAILED.value == "FAILED"
    assert ProjectStatus.BLOCKED.value == "BLOCKED"
    assert ProjectStatus.NEEDS_USER.value == "NEEDS_USER"


def test_rollup_all_done_marks_project_done(store):
    pid = store.create_project("P")
    t1 = store.add_task(pid, "a")
    t2 = store.add_task(pid, "b")
    store.update_task(t1, status="DONE")
    store.update_task(t2, status="DONE")
    assert store.get_project(pid)["status"] == ProjectStatus.DONE.value


def test_rollup_any_failed_marks_project_failed_not_done(store):
    """Regression: a project whose tasks all reached a terminal state with
    at least one FAILED used to roll up to DONE (ProjectStatus had no
    FAILED member). It must report FAILED."""
    pid = store.create_project("P")
    t1 = store.add_task(pid, "a")
    t2 = store.add_task(pid, "b")
    store.update_task(t1, status="DONE")
    store.update_task(t2, status="FAILED")
    assert store.get_project(pid)["status"] == ProjectStatus.FAILED.value


def test_rollup_blocked_counts_as_failure(store):
    pid = store.create_project("P")
    t1 = store.add_task(pid, "a")
    store.update_task(t1, status="BLOCKED")
    assert store.get_project(pid)["status"] == ProjectStatus.FAILED.value


def test_rollup_only_needs_user_open_marks_project_needs_user(store):
    pid = store.create_project("P")
    t1 = store.add_task(pid, "a")
    t2 = store.add_task(pid, "b")
    store.update_task(t1, status="DONE")
    store.update_task(t2, status="NEEDS_USER")
    assert store.get_project(pid)["status"] == ProjectStatus.NEEDS_USER.value


def test_rollup_does_not_fire_with_open_work(store):
    pid = store.create_project("P")
    t1 = store.add_task(pid, "a")
    store.add_task(pid, "b")  # left PENDING
    store.update_task(t1, status="DONE")
    assert store.get_project(pid)["status"] == ProjectStatus.ACTIVE.value


def test_failed_project_not_locked_rolls_forward_to_done(store):
    """FAILED is not a locked terminal — fixing the failed task rolls the
    project forward to DONE."""
    pid = store.create_project("P")
    t1 = store.add_task(pid, "a")
    store.update_task(t1, status="FAILED")
    assert store.get_project(pid)["status"] == ProjectStatus.FAILED.value
    store.update_task(t1, status="DONE")
    assert store.get_project(pid)["status"] == ProjectStatus.DONE.value


def test_done_project_is_locked(store):
    pid = store.create_project("P")
    t1 = store.add_task(pid, "a")
    store.update_task(t1, status="DONE")
    assert store.get_project(pid)["status"] == ProjectStatus.DONE.value
    # A later spurious FAILED on the (already-done) task must not un-done it.
    store.update_task(t1, status="FAILED")
    assert store.get_project(pid)["status"] == ProjectStatus.DONE.value


# ===================================================== alternatives persistence

def test_alternative_consumption_is_durable(store):
    """When a child fails and the parent's alternative is consumed, the
    re-parenting + cleared alternatives list must survive a reload.

    Before the fix, ProjectPlan.update_status diffed only status fields, so
    the alternative's new parent_id and the parent's emptied alternatives
    list were never written back — the fallback edge vanished on rehydrate.
    """
    pid = store.create_project("P")
    parent = store.add_task(pid, "parent")
    child = store.add_task(pid, "child", parent_id=parent)
    alt = store.add_task(pid, "alternative")  # top-level, to be reparented
    store.update_task(parent, alternatives=[alt])

    plan = ProjectPlan(store, pid)
    plan.update_status(child, TaskStatus.FAILED, failure_reason="boom")

    # Reload from the store — the durable state is what matters.
    plan2 = ProjectPlan(store, pid)
    alt_node = plan2.tree.nodes[alt]
    assert alt_node.parent_id == parent          # reparented + persisted
    assert alt_node.status == TaskStatus.READY    # promoted + persisted
    assert alt in plan2.tree.nodes[parent].children
    assert plan2.tree.nodes[parent].alternatives == []  # consumed + persisted


# =================================================================== depends_on

def test_depends_on_persists_through_store(store):
    pid = store.create_project("P")
    a = store.add_task(pid, "first")
    b = store.add_task(pid, "second", depends_on=[a])
    assert store.get_task(b)["depends_on"] == [a]
    assert store.get_task(a)["depends_on"] == []


def test_depends_on_update_via_update_task(store):
    pid = store.create_project("P")
    a = store.add_task(pid, "first")
    b = store.add_task(pid, "second")
    store.update_task(b, depends_on=[a])
    assert store.get_task(b)["depends_on"] == [a]
    store.update_task(b, depends_on=[])
    assert store.get_task(b)["depends_on"] == []


def test_next_ready_leaf_blocked_until_prereq_done(store):
    pid = store.create_project("P")
    a = store.add_task(pid, "first")
    b = store.add_task(pid, "second", depends_on=[a])
    plan = ProjectPlan(store, pid)
    # Only `a` is eligible while `b` waits on it.
    leaf = plan.next_ready_leaf()
    assert leaf is not None and leaf.id == a
    plan.update_status(a, TaskStatus.DONE, result="done")
    # Now `b` is unblocked.
    plan2 = ProjectPlan(store, pid)
    leaf2 = plan2.next_ready_leaf()
    assert leaf2 is not None and leaf2.id == b


def test_unknown_dependency_does_not_deadlock(store):
    pid = store.create_project("P")
    b = store.add_task(pid, "task", depends_on=["deadbeefdead"])
    plan = ProjectPlan(store, pid)
    leaf = plan.next_ready_leaf()
    assert leaf is not None and leaf.id == b  # stale dep id is ignored


def test_decompose_sequential_chains_dependencies(store):
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    root = plan.add_task("root")
    ids = plan.decompose(root, ["step 1", "step 2", "step 3"], sequential=True)
    assert len(ids) == 3
    # Each step depends on the previous one.
    assert store.get_task(ids[0])["depends_on"] == []
    assert store.get_task(ids[1])["depends_on"] == [ids[0]]
    assert store.get_task(ids[2])["depends_on"] == [ids[1]]
    # next_ready_leaf yields them strictly in order.
    plan2 = ProjectPlan(store, pid)
    assert plan2.next_ready_leaf().id == ids[0]
    plan2.update_status(ids[0], TaskStatus.DONE, result="ok")
    assert ProjectPlan(store, pid).next_ready_leaf().id == ids[1]


def test_decompose_non_sequential_all_eligible(store):
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    root = plan.add_task("root")
    ids = plan.decompose(root, ["a", "b"])  # sequential defaults to False
    for i in ids:
        assert store.get_task(i)["depends_on"] == []


def test_tasktree_depends_on_json_roundtrip():
    # to_json serializes the root's subtree, so depends_on must live on a
    # node reachable from the root.
    tree = TaskTree()
    root = tree.add_task("root")
    a = tree.add_task("a", parent_id=root)
    b = tree.add_task("b", parent_id=root, depends_on=[a])
    data = tree.to_json()
    tree2 = TaskTree()
    tree2.load_from_json(data)
    assert tree2.nodes[b].depends_on == [a]


# ========================================================== sqlite robustness

def test_wal_mode_enabled(store):
    conn = store._connect()
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert str(mode).lower() == "wal"


def test_migration_adds_depends_on_to_legacy_db(tmp_path):
    """A tasks table created before depends_on_json existed gets the column
    added by the additive migration, and reads back as []."""
    mem = tmp_path / "mem"
    mem.mkdir(parents=True)
    db = mem / "projects.db"
    # Hand-build a legacy `tasks` table WITHOUT depends_on_json.
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            parent_id TEXT,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            dependency_type TEXT NOT NULL DEFAULT 'ALL',
            alternatives_json TEXT NOT NULL DEFAULT '[]',
            postconditions_json TEXT NOT NULL DEFAULT '[]',
            result_summary TEXT NOT NULL DEFAULT '',
            failure_reason TEXT NOT NULL DEFAULT '',
            revision_count INTEGER NOT NULL DEFAULT 0,
            actual_tool_used TEXT,
            estimated_cost REAL NOT NULL DEFAULT 0.0,
            actual_cost REAL NOT NULL DEFAULT 0.0,
            depth INTEGER NOT NULL DEFAULT 0,
            position INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        """
    )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "depends_on_json" not in cols
    conn.commit()
    conn.close()

    # Opening the store runs the migration.
    store = ProjectStore(mem, sandbox_root=tmp_path / "sb")
    pid = store.create_project("P")
    tid = store.add_task(pid, "task")
    assert store.get_task(tid)["depends_on"] == []


# ===================================================== event-log pruning

def test_scratchpad_snapshot_pruning_keeps_only_latest(store):
    pid = store.create_project("P")
    store.log_event(pid, None, "scratchpad_snapshot", {"keys": {"a": 1}})
    store.log_event(pid, None, "scratchpad_snapshot", {"keys": {"a": 2}})
    store.log_event(pid, None, "scratchpad_snapshot", {"keys": {"a": 3}})
    snaps = store.list_events(pid, event_type="scratchpad_snapshot", limit=50)
    assert len(snaps) == 1
    assert snaps[0]["payload"]["keys"] == {"a": 3}


def test_pruning_does_not_touch_other_event_types(store):
    pid = store.create_project("P")
    store.log_event(pid, None, "custom_a", {})
    store.log_event(pid, None, "scratchpad_snapshot", {"keys": {}})
    store.log_event(pid, None, "custom_b", {})
    store.log_event(pid, None, "scratchpad_snapshot", {"keys": {}})
    # Both customs survive; only one snapshot remains.
    assert len(store.list_events(pid, event_type="custom_a")) == 1
    assert len(store.list_events(pid, event_type="custom_b")) == 1
    assert len(store.list_events(pid, event_type="scratchpad_snapshot")) == 1


# ===================================================== advancer hardening

@pytest.mark.parametrize("out,expected", [
    ("", True),
    ("   ", True),
    (None, True),
    ("ERROR: boom", True),
    ("error: boom", True),
    ("Traceback (most recent call last):", True),
    ("Here are the search results about errors in Python", False),
    ("normal output", False),
])
def test_looks_like_failure(out, expected):
    assert _looks_like_failure(out) is expected


async def test_advance_error_output_marks_failed_not_done(context, store):
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    tid = plan.add_task("Research the thing")

    async def runner(name, args):
        return "ERROR: search backend unreachable"

    r = await advance_once(context, pid, tool_runner=runner)
    assert store.get_task(tid)["status"] == "FAILED"
    assert "no usable result" in r.summary.lower()
    events = store.list_events(pid, event_type="autoadvance_failed")
    assert events and events[0]["payload"]["tool"] == "web_search"


async def test_advance_empty_output_marks_failed(context, store):
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    tid = plan.add_task("Research the thing")

    async def runner(name, args):
        return ""

    await advance_once(context, pid, tool_runner=runner)
    assert store.get_task(tid)["status"] == "FAILED"


async def test_advance_stamps_last_autoadvance_ts(context, store):
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    plan.add_task("Research the thing")

    async def runner(name, args):
        return "good search results here"

    await advance_once(context, pid, tool_runner=runner)
    meta = store.get_project(pid)["metadata"]
    assert "last_autoadvance_ts" in meta
    assert float(meta["last_autoadvance_ts"]) > 0


# ===================================================== tool surface

async def test_tool_task_decompose_sequential(context, store):
    await tool_manage_projects(context, action="create", title="Seq Proj")
    pid = context.current_project_id
    out = await tool_manage_projects(
        context, action="task_decompose",
        subtasks=["one", "two", "three"], sequential=True,
    )
    import json as _json
    created = _json.loads(out)["created"]
    assert len(created) == 3
    assert store.get_task(created[1])["depends_on"] == [created[0]]
    assert store.get_task(created[2])["depends_on"] == [created[1]]


async def test_tool_task_add_depends_on(context, store):
    await tool_manage_projects(context, action="create", title="Dep Proj")
    pid = context.current_project_id
    import json as _json
    first = _json.loads(
        await tool_manage_projects(context, action="task_add",
                                   description="first")
    )["task_id"]
    second = _json.loads(
        await tool_manage_projects(context, action="task_add",
                                   description="second", depends_on=[first])
    )["task_id"]
    assert store.get_task(second)["depends_on"] == [first]
