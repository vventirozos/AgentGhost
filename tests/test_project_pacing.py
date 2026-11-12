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

    # §4UI: ONE pacing rule, with the go-ahead case spelled out — a message
    # that asks for the project AND the work is itself the go-ahead
    assert "*** PACING (the one rule; tool results defer to it)" in briefing
    assert "that message IS the go-ahead" in briefing
    assert "If it asked ONLY for a project or a plan, show the task list and stop" in briefing
    assert "no 'shall I start?'" in briefing   # §4UK: no asking for a go-ahead
    assert "context window" in briefing
    assert "HARD RULE" not in briefing and "honor any pacing" not in briefing


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
    # §4UI: advice, not a stop — the next task stays visible for a user who
    # asked for more than one
    assert "You closed a task this turn. If the user asked for more" in briefing
    assert "NEXT TASK:" in briefing
    assert "STOP now" not in briefing
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
        subtasks=["Core Shell: skeleton → index.html", "File Explorer: vfs → js/files.js"]))
    pid = out["created"]
    assert len(store.list_tasks(pid)) == 2
    # agent now decomposes again with the same deliverables + one genuinely new
    # task (§4UI: duplicates are keyed on the declared file, not the head words)
    await tool_manage_projects(
        context, action="task_decompose",
        subtasks=["Core Shell: HTML structure → index.html", "File Explorer: folders → js/files.js",
                  "Terminal: shell → js/term.js"])
    descs = [t["description"] for t in store.list_tasks(pid)]
    assert len(descs) == 3                      # only Terminal added
    assert sum("core shell" in d.lower() for d in descs) == 1
    assert sum("file explorer" in d.lower() for d in descs) == 1


@pytest.mark.asyncio
async def test_a_goal_that_says_one_file_keeps_its_plan(context):
    """§4UI regression (req 8646c6c8): the model's own goal said "adding a new
    app = one registry entry + one file"; a goal-text regex collapsed the
    7-file webOS to ONE task, told the model to STOP, and refused every later
    split. The classifier is gone: the plan is what the model asked for."""
    store = context.project_store
    out = json.loads(await tool_manage_projects(
        context, action="create", title="webOS",
        goal="Build a modular webOS: each app lives in its own JS file under js/apps/, "
             "so adding a new app = one registry entry + one file.",
        subtasks=["Core shell: index.html + css/style.css + js/core.js",
                  "Apps: js/apps/notes.js + js/apps/calculator.js"]))
    assert len(out["tasks_created"]) == 2
    assert "SINGLE-FILE" not in out["agent_instruction"]
    dec = json.loads(await tool_manage_projects(
        context, action="task_decompose", subtasks=["js/apps/terminal.js", "js/apps/settings.js"]))
    assert not dec.get("refused") and len(dec["created"]) == 2
    rp = str(await tool_manage_projects(context, action="replan",
                                        subtasks=["Shell", "Apps", "Polish"]))
    assert "SINGLE-FILE" not in rp and "NOT replanned" not in rp


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
    # §4UI: the result defers to the ONE pacing rule instead of its own STOP
    assert "follow the PACING rule in the CURRENT PROJECT block" in instr
    assert "STOP" not in instr and "DO NOT begin executing" not in instr


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
    assert "3 task" in instr  # mentions the count
    # §4UI: defers to the one pacing rule (a split mid-work is not a new plan
    # to stop on)
    assert "follow the PACING rule in the CURRENT" in instr
    assert "STOP here" not in instr
