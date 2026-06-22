"""Project pacing: after create/decompose the agent must STOP and advance
ONE task per user go-ahead, not grind the whole tree in one turn (which
floods the context window on large projects).

Guards the prompt directive in core.prompts.build_project_briefing and the
stop-and-await `agent_instruction` returned by the create / task_decompose
actions of manage_projects.
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.prompts import build_project_briefing
from ghost_agent.core.planning import ProjectPlan
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
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


# ------------------------------------------------------------- briefing prompt

def test_briefing_directs_one_task_at_a_time(store):
    pid = store.create_project("Big Proj", goal="ship a large thing")
    plan = ProjectPlan(store, pid)
    plan.add_task("task a")
    plan.add_task("task b")

    briefing = build_project_briefing(store, pid)

    # The new pacing directive is present and unambiguous.
    assert "ONE TASK AT A TIME" in briefing
    assert "context window" in briefing
    # It tells the agent to stop and await direction.
    assert "STOP" in briefing
    low = briefing.lower()
    assert "go-ahead" in low or "proceed" in low or "wait for" in low


def test_briefing_drops_old_parallel_directive(store):
    pid = store.create_project("P")
    ProjectPlan(store, pid).add_task("t")
    briefing = build_project_briefing(store, pid)
    # The previous behaviour (stack every task into one turn) must be gone.
    assert "PARALLEL TASK EXECUTION" not in briefing
    assert "stack ALL" not in briefing


def test_briefing_shows_next_task_by_default(store):
    pid = store.create_project("P", goal="g")
    plan = ProjectPlan(store, pid)
    plan.add_task("first task")
    plan.add_task("second task")
    briefing = build_project_briefing(store, pid)
    assert "NEXT TASK:" in briefing


def test_briefing_suppresses_next_task_after_a_close(store):
    # Once a task has been closed this request, the NEXT TASK pointer is
    # replaced by a hard stop so the model doesn't roll into the next task.
    pid = store.create_project("P", goal="g")
    plan = ProjectPlan(store, pid)
    plan.add_task("first task")
    plan.add_task("second task")
    briefing = build_project_briefing(store, pid, suppress_next_task=True)
    assert "NEXT TASK:" not in briefing
    assert "completed a task this turn" in briefing
    assert "STOP" in briefing
    # the pending tasks are still listed (situational awareness), just not
    # advertised as the thing to start now
    assert "OPEN TASKS" in briefing


# ------------------------------------------------------------- create / decompose

@pytest.mark.asyncio
async def test_create_with_subtasks_creates_the_tasks(context):
    """Live bug: the model routinely passes `subtasks` to action=create
    expecting them to become tasks. They used to be silently dropped — the
    project had ZERO tasks while the model believed they existed, so a later
    'proceed' found no plan and the flow derailed (observed twice). create
    now decomposes them."""
    store = context.project_store
    out = json.loads(await tool_manage_projects(
        context, action="create", title="Mini OS",
        goal="a multi-module desktop app project",
        subtasks=["Core shell", "File Explorer", "Snake game"]))
    pid = out["created"]
    assert len(out["tasks_created"]) == 3
    descs = [t["description"] for t in store.list_tasks(pid)]
    assert descs == ["Core shell", "File Explorer", "Snake game"]
    # instruction reflects that tasks already exist (don't tell it to decompose)
    assert "WITH 3 task" in out["agent_instruction"]


@pytest.mark.asyncio
async def test_create_with_sequential_subtasks_chains_them(context):
    store = context.project_store
    out = json.loads(await tool_manage_projects(
        context, action="create", title="Seq", subtasks=["a", "b", "c"],
        sequential=True))
    pid = out["created"]
    ids = out["tasks_created"]
    assert store.get_task(ids[1])["depends_on"] == [ids[0]]
    assert store.get_task(ids[2])["depends_on"] == [ids[1]]


@pytest.mark.asyncio
async def test_create_then_decompose_does_not_duplicate_tasks(context):
    """Live bug: the model calls create-WITH-subtasks AND then task_decompose
    (or decomposes twice), piling up duplicate tasks (two Core Shells, two
    File Explorers) that then fail. Decompose now drops subtasks whose feature
    already exists."""
    store = context.project_store
    out = json.loads(await tool_manage_projects(
        context, action="create", title="OS", goal="a multi-module desktop app",
        subtasks=["Core Shell: skeleton", "File Explorer: vfs"]))
    pid = out["created"]
    assert len(store.list_tasks(pid)) == 2
    # agent now decomposes again with dup descriptions + one genuinely new task
    await tool_manage_projects(
        context, action="task_decompose",
        subtasks=["Core Shell: HTML structure", "File Explorer: folders",
                  "Terminal: shell"])
    descs = [t["description"] for t in store.list_tasks(pid)]
    assert len(descs) == 3                      # only Terminal added
    assert sum("core shell" in d.lower() for d in descs) == 1
    assert sum("file explorer" in d.lower() for d in descs) == 1


@pytest.mark.asyncio
async def test_single_file_goal_collapses_to_one_task(context):
    """A cohesive single-file deliverable must NOT be split into per-feature
    tasks (they'd merge into one file and collide — observed live as a page
    that throws on load). create collapses subtasks to ONE build task and
    steers to a one-turn build."""
    store = context.project_store
    out = json.loads(await tool_manage_projects(
        context, action="create", title="Browser OS",
        goal="Create a single-file browser OS with desktop, taskbar, 5 apps",
        subtasks=["Core Shell", "File Explorer", "Snake", "Terminal"]))
    pid = out["created"]
    assert len(out["tasks_created"]) == 1                      # collapsed
    assert "single-file" in store.list_tasks(pid)[0]["description"].lower()
    assert "SINGLE-FILE" in out["agent_instruction"]
    assert "do NOT call autoadvance" in out["agent_instruction"]


@pytest.mark.asyncio
async def test_single_file_project_refuses_decompose_split(context):
    store = context.project_store
    await tool_manage_projects(
        context, action="create", title="OS",
        goal="a single-file browser OS in one index.html")
    pid = context.current_project_id
    before = len(store.list_tasks(pid))
    dec = json.loads(await tool_manage_projects(
        context, action="task_decompose", subtasks=["App A", "App B", "App C"]))
    assert dec.get("refused") is True
    assert len(store.list_tasks(pid)) == before               # nothing fanned out


@pytest.mark.asyncio
async def test_multi_file_goal_still_decomposes_normally(context):
    # a NON-single-file build goal must still decompose per-file as before
    store = context.project_store
    out = json.loads(await tool_manage_projects(
        context, action="create", title="CSV tool",
        goal="build a python CSV stats CLI with tests",
        subtasks=["src/parser.py: parse", "src/stats.py: compute", "tests/test_stats.py"]))
    assert len(out["tasks_created"]) == 3                      # not collapsed


@pytest.mark.asyncio
async def test_create_without_subtasks_still_asks_to_decompose(context):
    out = json.loads(await tool_manage_projects(
        context, action="create", title="Bare"))
    assert out["tasks_created"] == []
    assert "task_decompose" in out["agent_instruction"]


@pytest.mark.asyncio
async def test_create_returns_stop_and_await_instruction(context):
    out = json.loads(
        await tool_manage_projects(context, action="create", title="New Effort")
    )
    instr = out.get("agent_instruction", "")
    assert "STOP" in instr
    assert "ONE task" in instr or "one task" in instr.lower()
    # explicitly steers away from auto-executing
    assert "DO NOT begin executing" in instr or "do not begin executing" in instr.lower()


@pytest.mark.asyncio
async def test_decompose_returns_stop_and_await_instruction(context):
    await tool_manage_projects(context, action="create", title="Effort 2")
    out = json.loads(
        await tool_manage_projects(
            context, action="task_decompose",
            subtasks=["alpha", "beta", "gamma"],
        )
    )
    # the created list is unchanged (additive return key)
    assert len(out["created"]) == 3
    instr = out.get("agent_instruction", "")
    assert "STOP" in instr
    assert "3 task" in instr  # mentions the count
    assert "proceed to next task" in instr.lower()
