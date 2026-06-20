"""Cross-turn project working memory (rec 1) + per-file decomposition
guidance (rec 2).

Rec 1: the briefing must carry what makes a fresh turn cheap — the DESIGN
LEDGER (durable facts the agent recorded) and DONE SO FAR (recently-completed
tasks + their one-line result) — so the next turn inherits "what exists and
how it works" instead of re-reading files.

Rec 2: the decompose surface must steer toward per-file/bounded-function
tasks instead of N tasks that all edit one file.
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.prompts import build_project_briefing
from ghost_agent.core.planning import ProjectPlan, TaskStatus
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import tool_manage_projects, MANAGE_PROJECTS_TOOL_DEF


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


# --------------------------------------------------------------- ledger store

def test_append_ledger_accumulates_and_dedups(store):
    pid = store.create_project("P")
    store.append_ledger(pid, "single HTML file at browser-os.html")
    store.append_ledger(pid, "windows are .window divs, opened via openApp(id)")
    store.append_ledger(pid, "windows are .window divs, opened via openApp(id)")  # dup
    led = store.get_ledger(pid)
    assert led.count("openApp(id)") == 1
    assert "browser-os.html" in led
    assert len(led.splitlines()) == 2


def test_append_ledger_collapses_whitespace(store):
    pid = store.create_project("P")
    store.append_ledger(pid, "line  with\nnewline\tand   spaces")
    assert store.get_ledger(pid) == "line with newline and spaces"


def test_ledger_is_bounded(store):
    pid = store.create_project("P")
    for i in range(100):
        store.append_ledger(pid, f"fact number {i} " + "x" * 50)
    led = store.get_ledger(pid)
    assert len(led) <= store.LEDGER_MAX_CHARS
    assert len(led.splitlines()) <= store.LEDGER_MAX_LINES
    # newest fact survives, oldest evicted
    assert "fact number 99" in led
    assert "fact number 0 " not in led


def test_set_ledger_replaces(store):
    pid = store.create_project("P")
    store.append_ledger(pid, "old")
    store.set_ledger(pid, "fresh ledger content")
    assert store.get_ledger(pid) == "fresh ledger content"


# --------------------------------------------------------------- briefing surfacing

def test_briefing_surfaces_ledger(store):
    pid = store.create_project("P", goal="build a thing")
    store.append_ledger(pid, "entrypoint is main.py; config in settings.toml")
    b = build_project_briefing(store, pid)
    assert "DESIGN LEDGER" in b
    assert "entrypoint is main.py" in b


def test_briefing_surfaces_done_tasks_with_results(store):
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    t1 = plan.add_task("parse the CSV")
    t2 = plan.add_task("compute stats")
    plan.update_status(t1, TaskStatus.DONE,
                       result="wrote parser.py; parse_csv(path) -> list[dict]")

    b = build_project_briefing(store, pid)
    # match the SECTION HEADER, not the mention inside the ONE TASK rule text
    assert "DONE SO FAR (" in b
    assert "parser.py" in b
    assert "parse_csv(path)" in b
    # the still-open task's id must NOT appear as a DONE digest bullet. Isolate
    # the DONE block (header → next section) to avoid matching RECENT EVENTS.
    done_block = b.split("DONE SO FAR (")[1].split("RECENT EVENTS")[0]
    assert f"[{t1}]" in done_block
    assert f"[{t2}]" not in done_block


def test_briefing_omits_done_section_when_nothing_done(store):
    pid = store.create_project("P")
    ProjectPlan(store, pid).add_task("only task")
    assert "DONE SO FAR (" not in build_project_briefing(store, pid)


# --------------------------------------------------------------- tool: ledger action

@pytest.mark.asyncio
async def test_ledger_action_append_and_read(context, store):
    await tool_manage_projects(context, action="create", title="Ledger Proj")
    pid = context.current_project_id
    out = json.loads(await tool_manage_projects(
        context, action="ledger", ledger="db schema lives in schema.sql"))
    assert out["action_taken"] == "appended"
    assert "schema.sql" in out["ledger"]
    # read back (no ledger text)
    read = json.loads(await tool_manage_projects(context, action="ledger"))
    assert "schema.sql" in read["ledger"]


@pytest.mark.asyncio
async def test_task_update_done_appends_ledger(context, store):
    await tool_manage_projects(context, action="create", title="P2")
    pid = context.current_project_id
    await tool_manage_projects(context, action="task_decompose", subtasks=["build x"])
    tid = store.list_tasks(pid)[0]["id"]
    await tool_manage_projects(
        context, action="task_update", task_id=tid, status="DONE",
        result="built x", ledger="x is implemented in x.py as build_x()")
    assert "build_x()" in store.get_ledger(pid)


# --------------------------------------------------------------- rec 2: decompose guidance

def test_decompose_schema_steers_to_per_file_tasks():
    # The subtasks param description must carry the per-file granularity rule.
    props = MANAGE_PROJECTS_TOOL_DEF["function"]["parameters"]["properties"]
    sub = props["subtasks"]["description"].lower()
    assert "file" in sub and ("bounded" in sub or "function" in sub)
    assert "same file" in sub or "one file into n" in sub
    # the ledger param is documented too
    assert "ledger" in props
    assert "briefing" in props["ledger"]["description"].lower()


def test_ledger_action_is_registered():
    enum = MANAGE_PROJECTS_TOOL_DEF["function"]["parameters"]["properties"]["action"]["enum"]
    assert "ledger" in enum


@pytest.mark.asyncio
async def test_create_instruction_mentions_per_file_and_ledger(context):
    out = json.loads(await tool_manage_projects(context, action="create", title="P3"))
    instr = out["agent_instruction"]
    assert "file" in instr.lower()
    assert "ledger" in instr.lower()
