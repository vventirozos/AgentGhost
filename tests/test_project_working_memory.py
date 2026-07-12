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
    # Match the SECTION HEADER, not the mention inside the ONE TASK rule text.
    # The header was "DONE SO FAR (N of M…)" until 2026-07-11 — that parses as
    # a PROGRESS FRACTION ("N of M tasks done") rather than the display
    # truncation it actually is, and the model misread it live. It now leads
    # with the completed count: "DONE SO FAR — M task(s) complete…".
    assert "DONE SO FAR — 1 task(s) complete" in b
    assert "parser.py" in b
    assert "parse_csv(path)" in b
    # the still-open task's id must NOT appear as a DONE digest bullet. Isolate
    # the DONE block (header → next section) to avoid matching RECENT EVENTS.
    done_block = b.split("DONE SO FAR —")[1].split("RECENT EVENTS")[0]
    assert f"[{t1}]" in done_block
    assert f"[{t2}]" not in done_block


def test_briefing_omits_done_section_when_nothing_done(store):
    pid = store.create_project("P")
    ProjectPlan(store, pid).add_task("only task")
    assert "DONE SO FAR —" not in build_project_briefing(store, pid)


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


# --------------------------------------------------------------- config store (1B)

def test_set_config_value_upsert_and_read(store):
    pid = store.create_project("P")
    store.set_config_value(pid, "GHOST_MODEL", "qwen-3.6-35b-a3")
    store.set_config_value(pid, "port", "8000")
    cfg = store.get_config(pid)
    assert cfg["GHOST_MODEL"] == "qwen-3.6-35b-a3"
    assert cfg["port"] == "8000"


def test_set_config_value_last_write_wins(store):
    pid = store.create_project("P")
    store.set_config_value(pid, "port", "8000")
    store.set_config_value(pid, "port", "9000")
    assert store.get_config(pid) == {"port": "9000"}


def test_empty_config_value_deletes_key(store):
    pid = store.create_project("P")
    store.set_config_value(pid, "port", "8000")
    store.set_config_value(pid, "port", "")
    assert "port" not in store.get_config(pid)


def test_config_collapses_whitespace_and_bounds_value(store):
    pid = store.create_project("P")
    store.set_config_value(pid, "  db  uri ", "postgresql://" + "x" * 500)
    cfg = store.get_config(pid)
    assert "db uri" in cfg  # key whitespace-collapsed
    assert len(cfg["db uri"]) <= store.CONFIG_MAX_VALUE_CHARS


def test_config_key_count_is_bounded(store):
    pid = store.create_project("P")
    for i in range(store.CONFIG_MAX_KEYS + 10):
        store.set_config_value(pid, f"key{i}", f"v{i}")
    cfg = store.get_config(pid)
    assert len(cfg) == store.CONFIG_MAX_KEYS
    # Oldest keys dropped first, newest retained.
    assert f"key{store.CONFIG_MAX_KEYS + 9}" in cfg
    assert "key0" not in cfg


def test_briefing_surfaces_config(store):
    pid = store.create_project("P", goal="ship it")
    store.set_config_value(pid, "GHOST_MODEL", "qwen-3.6-35b-a3")
    b = build_project_briefing(store, pid)
    assert "CONFIG (" in b
    assert "GHOST_MODEL = qwen-3.6-35b-a3" in b


def test_briefing_omits_config_when_empty(store):
    pid = store.create_project("P")
    assert "CONFIG (" not in build_project_briefing(store, pid)


# --------------------------------------------------------------- tool: config action (1B)

@pytest.mark.asyncio
async def test_config_action_set_and_read(context, store):
    await tool_manage_projects(context, action="create", title="Cfg Proj")
    out = json.loads(await tool_manage_projects(
        context, action="config", config_key="port", config_value="8000"))
    assert out["action_taken"] == "set"
    assert out["config"]["port"] == "8000"
    read = json.loads(await tool_manage_projects(context, action="config"))
    assert read["config"]["port"] == "8000"


@pytest.mark.asyncio
async def test_config_action_delete(context, store):
    await tool_manage_projects(context, action="create", title="Cfg Proj 2")
    await tool_manage_projects(
        context, action="config", config_key="port", config_value="8000")
    out = json.loads(await tool_manage_projects(
        context, action="config", config_key="port", config_value=""))
    assert out["action_taken"] == "deleted"
    assert "port" not in out["config"]


def test_config_action_is_registered():
    props = MANAGE_PROJECTS_TOOL_DEF["function"]["parameters"]["properties"]
    enum = props["action"]["enum"]
    assert "config" in enum
    assert "config_key" in props and "config_value" in props
