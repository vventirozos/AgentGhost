"""§4VB: every task of a project is in its task tree.

Live 11-20: a project planned as 3 top-level tasks (the usual shape — the owner's live projects have 8 and 9
and no common parent) showed ONE in `manage_projects get`'s ``task_tree``: `TaskTree.render` walked from
``root_id``, the first root ever seen. The model answered "tasks done 1 of 1". `next_ready_leaf` had been
fixed for this class long ago; render / to_json / get_active_node had not.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import planning as P
from ghost_agent.core.planning import ProjectPlan, TaskStatus, TaskTree
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import tool_manage_projects


@pytest.fixture
def store(tmp_path):
    sb = tmp_path / "sandbox"
    sb.mkdir()
    return ProjectStore(tmp_path / "memory", sandbox_root=sb)


def _three(store):
    pid = store.create_project("tempconv", kind="CODING", goal="a converter")
    a = store.add_task(pid, "Implement conv.py")
    b = store.add_task(pid, "Write test_conv.py", depends_on=[a])
    c = store.add_task(pid, "Run the tests", depends_on=[a, b])
    return pid, (a, b, c)


def test_render_lists_every_top_level_task_in_plan_order(store):
    pid, (a, b, c) = _three(store)
    lines = ProjectPlan(store, pid).render().splitlines()
    assert [ln.split("]")[0].split("[")[1] for ln in lines] == [a, b, c]


def test_a_child_stays_under_its_parent_and_the_next_root_follows(store):
    pid, (a, b, c) = _three(store)
    kid = store.add_task(pid, "a sub-step", parent_id=a)
    lines = ProjectPlan(store, pid).render().splitlines()
    assert lines[1].startswith("  ") and kid in lines[1]
    assert [ln for ln in lines if not ln.startswith(" ")] == [ln for ln in lines if any(t in ln for t in (a, b, c))]


def test_all_done_shows_every_task_done(store):
    pid, ids = _three(store)
    for t in ids:
        store.update_task(t, status="DONE")
    out = ProjectPlan(store, pid).render()
    assert out.count("(DONE)") == 3 and all(t in out for t in ids)


def test_an_orphan_whose_parent_is_gone_is_still_shown():
    tree = TaskTree()
    tree.nodes["x1"] = P.TaskNode(id="x1", description="orphan", status=TaskStatus.PENDING, parent_id="gone")
    assert "orphan" in tree.render() and tree.roots() == ["x1"]
    assert tree.generate_retrospective()["total_tasks"] == 1      # not "No plan executed."


def test_the_active_task_is_found_under_a_later_root():
    tree = TaskTree()
    first = tree.add_task("first")
    second = tree.add_task("second")
    tree.update_status(first, TaskStatus.DONE, "ok")
    assert tree.get_active_node().id == second


def test_a_running_task_under_a_later_root_beats_a_pending_one_under_the_first():
    """The order survives several roots: anything IN_PROGRESS anywhere comes before any PENDING task."""
    tree = TaskTree()
    tree.add_task("first, not started")
    running = tree.add_task("second, running")
    tree.update_status(running, TaskStatus.IN_PROGRESS)
    assert tree.get_active_node().id == running


def test_an_orphans_only_tree_still_has_an_active_task():
    tree = TaskTree()
    tree.nodes["x1"] = P.TaskNode(id="x1", description="orphan", status=TaskStatus.PENDING, parent_id="gone")
    assert tree.root_id is None and tree.get_active_node().id == "x1"


def test_to_json_keeps_the_one_root_shape_and_lists_every_root_when_several():
    one = TaskTree()
    r = one.add_task("only")
    assert one.to_json()["id"] == r
    many = TaskTree()
    ids = [many.add_task(f"t{i}") for i in range(3)]
    assert [n["id"] for n in many.to_json()["roots"]] == ids


def test_the_several_roots_shape_reads_back_whole():
    """Review r1: `load_from_json` reads what `to_json` writes — every root and its children."""
    src = TaskTree()
    a = src.add_task("a")
    src.add_task("a1", parent_id=a)
    src.add_task("b")
    back = TaskTree()
    back.load_from_json(src.to_json())
    assert sorted(n.description for n in back.nodes.values()) == ["a", "a1", "b"]
    assert {back.nodes[r].description for r in back.roots()} == {"a", "b"}


@pytest.mark.asyncio
async def test_manage_projects_get_shows_the_model_every_task(store, tmp_path):
    """R5 — one project, one story: `get`'s task_tree names every task `task_list` names."""
    pid, ids = _three(store)
    ctx = SimpleNamespace(project_store=store, scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                          graph_memory=None, contradiction_log=None, current_project_id=None,
                          llm_client=None, sandbox_manager=None)
    tree = json.loads(await tool_manage_projects(ctx, action="get", project_id=pid))["briefing"]["task_tree"]
    listed = await tool_manage_projects(ctx, action="task_list", project_id=pid)
    assert all(t in tree for t in ids) and all(t in listed for t in ids)


# ───────────────────────────── R1 enumeration ─────────────────────────────

#: the only functions that may READ ``root_id``: the bookkeeping that records the first root once
_ROOT_ID_SETTERS = {"add_task", "traverse", "_hydrate"}


def test_no_walk_starts_from_root_id():
    """Outside the setters, nothing in planning.py reads ``root_id`` — a walk over the tree starts at
    `TaskTree.roots()`. A new method anchored at ``root_id`` would hide every other top-level task again."""
    tree = ast.parse(Path(P.__file__).read_text(encoding="utf-8"))
    readers = set()
    seen = 0

    def visit(node, fn=""):
        nonlocal seen
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if isinstance(child, ast.Attribute) and child.attr == "root_id":
                seen += 1
                if isinstance(child.ctx, ast.Load):
                    readers.add(name)
            visit(child, name)
    visit(tree)
    assert seen >= 4, "the enumeration found no root_id — the instrument is broken"
    assert readers <= _ROOT_ID_SETTERS, readers - _ROOT_ID_SETTERS
