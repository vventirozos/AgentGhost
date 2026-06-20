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


# ------------------------------------------------------------- create / decompose

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
