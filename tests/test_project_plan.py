"""Unit tests for ProjectPlan + the PAUSED / NEEDS_USER task statuses."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.planning import (
    ProjectPlan, TaskStatus, TaskTree, DependencyType,
)
from ghost_agent.memory.projects import ProjectStore


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def plan(store):
    pid = store.create_project("P")
    return ProjectPlan(store, pid)


# --------------------------------------------------------------------- new statuses

def test_new_status_values_exist():
    assert TaskStatus.PAUSED.value == "PAUSED"
    assert TaskStatus.NEEDS_USER.value == "NEEDS_USER"


def test_paused_status_does_not_propagate_failure():
    """PAUSED must NOT mark the parent as BLOCKED — it is in-flight."""
    tree = TaskTree()
    root = tree.add_task("root")
    c1 = tree.add_task("c1", parent_id=root)
    c2 = tree.add_task("c2", parent_id=root)
    tree.update_status(c1, TaskStatus.PAUSED)
    assert tree.nodes[root].status == TaskStatus.PENDING
    # Completing c2 should not complete the parent (c1 still PAUSED)
    tree.update_status(c2, TaskStatus.DONE)
    assert tree.nodes[root].status == TaskStatus.PENDING


def test_needs_user_status_does_not_propagate_failure():
    tree = TaskTree()
    root = tree.add_task("root")
    c = tree.add_task("c", parent_id=root)
    tree.update_status(c, TaskStatus.NEEDS_USER)
    assert tree.nodes[root].status == TaskStatus.PENDING


def test_render_uses_icon_for_new_statuses():
    tree = TaskTree()
    root = tree.add_task("root")
    tree.update_status(root, TaskStatus.PAUSED)
    assert "⏸" in tree.render()


# --------------------------------------------------------------------- ProjectPlan

def test_plan_starts_empty(plan):
    assert plan.tree.root_id is None
    assert plan.next_ready_leaf() is None


def test_plan_add_task_persists(plan, store):
    tid = plan.add_task("root task")
    assert plan.tree.root_id == tid
    # Rehydrate from the store to confirm persistence
    plan2 = ProjectPlan(store, plan.project_id)
    assert plan2.tree.root_id == tid
    assert plan2.tree.nodes[tid].description == "root task"


def test_plan_add_subtask_links_parent(plan):
    r = plan.add_task("root")
    c = plan.add_task("child", parent_id=r)
    assert c in plan.tree.nodes[r].children
    assert plan.tree.nodes[c].parent_id == r


def test_plan_decompose_creates_ordered_subtasks(plan):
    r = plan.add_task("root")
    ids = plan.decompose(r, ["one", "two", "three"])
    assert len(ids) == 3
    descs = [plan.tree.nodes[i].description for i in ids]
    assert descs == ["one", "two", "three"]
    for i in ids:
        assert plan.tree.nodes[i].parent_id == r


def test_plan_decompose_rejects_unknown_task(plan):
    with pytest.raises(ValueError):
        plan.decompose("no-id", ["x"])


def test_plan_decompose_skips_blank_descriptions(plan):
    r = plan.add_task("root")
    ids = plan.decompose(r, ["real", "", "  ", "also real"])
    assert len(ids) == 2


def test_plan_update_status_persists_cascade(plan, store):
    r = plan.add_task("root")
    c1 = plan.add_task("c1", parent_id=r)
    c2 = plan.add_task("c2", parent_id=r)
    plan.update_status(c1, TaskStatus.DONE, result="ok")
    plan.update_status(c2, TaskStatus.DONE, result="ok")
    # Parent cascades to DONE via TaskTree — assert both tree + store reflect it
    assert plan.tree.nodes[r].status == TaskStatus.DONE
    assert store.get_task(r)["status"] == "DONE"
    assert store.get_task(c1)["status"] == "DONE"


def test_plan_next_ready_leaf_skips_paused_branches(plan):
    r = plan.add_task("root")
    c1 = plan.add_task("c1", parent_id=r)
    c2 = plan.add_task("c2", parent_id=r)
    plan.update_status(c1, TaskStatus.PAUSED)
    leaf = plan.next_ready_leaf()
    assert leaf is not None and leaf.id == c2


def test_plan_next_ready_leaf_skips_needs_user(plan):
    r = plan.add_task("root")
    c1 = plan.add_task("c1", parent_id=r)
    plan.update_status(c1, TaskStatus.NEEDS_USER)
    assert plan.next_ready_leaf() is None


def test_plan_next_ready_leaf_returns_pending_leaf(plan):
    r = plan.add_task("root")
    c = plan.add_task("c", parent_id=r)
    leaf = plan.next_ready_leaf()
    assert leaf is not None and leaf.id == c


def test_plan_rehydrates_entire_tree(store):
    pid = store.create_project("P")
    p = ProjectPlan(store, pid)
    r = p.add_task("root")
    c1 = p.add_task("c1", parent_id=r)
    c2 = p.add_task("c2", parent_id=r)
    gc = p.add_task("gc", parent_id=c1)
    p.update_status(c1, TaskStatus.IN_PROGRESS)

    fresh = ProjectPlan(store, pid)
    assert fresh.tree.root_id == r
    assert set(fresh.tree.nodes[r].children) == {c1, c2}
    assert fresh.tree.nodes[c1].status == TaskStatus.IN_PROGRESS
    assert fresh.tree.nodes[gc].parent_id == c1


def test_plan_update_status_pause_then_resume(plan, store):
    r = plan.add_task("root")
    plan.update_status(r, TaskStatus.PAUSED)
    assert store.get_task(r)["status"] == "PAUSED"
    plan.update_status(r, TaskStatus.READY)
    assert store.get_task(r)["status"] == "READY"


def test_plan_respects_postconditions_through_cascade(plan):
    r = plan.add_task("root", postconditions=["must contain success marker"])
    plan.update_status(r, TaskStatus.DONE, result="irrelevant output")
    # TaskTree should flip back to FAILED because postconditions unmet
    assert plan.tree.nodes[r].status == TaskStatus.FAILED
    # And that's persisted
    assert plan.store.get_task(r)["status"] == "FAILED"


def test_plan_generate_retrospective_roundtrip(plan):
    r = plan.add_task("root")
    c = plan.add_task("c", parent_id=r)
    plan.update_status(c, TaskStatus.DONE, result="done ok")
    retro = plan.generate_retrospective()
    assert retro["completed"] >= 1
    assert retro["total_tasks"] == 2
