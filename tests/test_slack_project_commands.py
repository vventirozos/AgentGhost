"""Tests for the Slack slash-command parser (no Slack SDK required)."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from types import SimpleNamespace

from ghost_agent.memory.projects import ProjectStore
from interface.slack_project_commands import (
    SlackContext, SlackResponse, parse_command, route, advance_async,
)


@pytest.fixture
def ctx(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    return SlackContext(store=store)


# --------------------------------------------------------------------- tokenizer

def test_parse_command_handles_quotes():
    assert parse_command('new "Build CLI tool"') == ["new", "Build CLI tool"]


def test_parse_command_falls_back_on_bad_quotes():
    # Unclosed quote — shlex raises; we fall back to whitespace split
    out = parse_command('new "unclosed')
    assert out[0] == "new"


def test_parse_command_empty_returns_empty():
    assert parse_command("") == []
    assert parse_command("   ") == []


# --------------------------------------------------------------------- help / unknown

def test_empty_input_shows_usage(ctx):
    r = route("", ctx)
    assert "project commands" in r.text.lower()


def test_unknown_command_hints_help(ctx):
    r = route("frobnicate", ctx)
    assert "unknown" in r.text.lower()


def test_help_alias_works(ctx):
    r = route("help", ctx)
    assert "project commands" in r.text.lower()


# --------------------------------------------------------------------- list / new

def test_list_empty(ctx):
    r = route("list", ctx)
    assert "no projects" in r.text.lower()


def test_new_creates_and_activates(ctx):
    r = route('new "Research BGE embeddings"', ctx)
    assert "created" in r.text.lower()
    assert ctx.current_project_id is not None
    projs = ctx.store.list_projects()
    assert len(projs) == 1
    assert projs[0]["title"] == "Research BGE embeddings"


def test_new_requires_title(ctx):
    r = route("new", ctx)
    assert "usage" in r.text.lower()


def test_list_shows_projects_and_current(ctx):
    route('new "Alpha"', ctx)
    r = route("list", ctx)
    assert "Alpha" in r.text
    assert "current" in r.text.lower()


def test_list_with_status_filter(ctx):
    route('new "A"', ctx)
    route("exit", ctx)
    r = route("list ACTIVE", ctx)
    assert "A" in r.text


# --------------------------------------------------------------------- switch / exit / resume

def test_switch_sets_current(ctx):
    pid = ctx.store.create_project("X")
    r = route(f"switch {pid}", ctx)
    assert ctx.current_project_id == pid
    assert "X" in r.text


def test_switch_missing_is_soft_error(ctx):
    r = route("switch nope", ctx)
    assert "no project" in r.text.lower()


def test_exit_clears_current(ctx):
    pid = ctx.store.create_project("X")
    ctx.current_project_id = pid
    r = route("exit", ctx)
    assert ctx.current_project_id is None
    assert "left project" in r.text.lower()


def test_exit_when_not_in_project(ctx):
    r = route("exit", ctx)
    assert "not in a project" in r.text.lower()


def test_resume_lists_open_tasks(ctx):
    pid = ctx.store.create_project("X")
    ctx.store.add_task(pid, "alpha")
    ctx.store.add_task(pid, "beta")
    r = route(f"resume {pid}", ctx)
    assert "alpha" in r.text
    assert "beta" in r.text
    # project_resumed event logged
    evs = ctx.store.list_events(pid, event_type="project_resumed")
    assert evs and evs[0]["payload"].get("via") == "slack"


# --------------------------------------------------------------------- status

def test_status_free_chat(ctx):
    r = route("status", ctx)
    assert "free chat" in r.text.lower()


def test_status_project_mode(ctx):
    pid = ctx.store.create_project("X", goal="Ship")
    ctx.current_project_id = pid
    ctx.store.add_task(pid, "t1")
    r = route("status", ctx)
    assert "X" in r.text
    assert "Open: 1" in r.text


# --------------------------------------------------------------------- task sub

def test_task_add_requires_active_project(ctx):
    r = route("task add Do the thing", ctx)
    assert "no active project" in r.text.lower()


def test_task_add_creates(ctx):
    pid = ctx.store.create_project("X")
    ctx.current_project_id = pid
    r = route("task add Build the widget", ctx)
    assert "added task" in r.text.lower()
    tasks = ctx.store.list_tasks(pid)
    assert len(tasks) == 1
    assert tasks[0]["description"] == "Build the widget"


def test_task_done_marks_completion(ctx):
    pid = ctx.store.create_project("X")
    ctx.current_project_id = pid
    tid = ctx.store.add_task(pid, "root")
    r = route(f"task done {tid} shipped it", ctx)
    assert "DONE" in r.text
    assert ctx.store.get_task(tid)["status"] == "DONE"


def test_task_done_cross_project_is_rejected(ctx):
    p1 = ctx.store.create_project("X")
    p2 = ctx.store.create_project("Y")
    tid = ctx.store.add_task(p1, "root")
    ctx.current_project_id = p2
    r = route(f"task done {tid}", ctx)
    assert "not found" in r.text.lower()


def test_task_unknown_subcommand(ctx):
    pid = ctx.store.create_project("X")
    ctx.current_project_id = pid
    r = route("task bogus", ctx)
    assert "unknown" in r.text.lower()


# --------------------------------------------------------------------- events / advance

def test_events_limit(ctx):
    pid = ctx.store.create_project("X")
    ctx.current_project_id = pid
    for i in range(5):
        ctx.store.log_event(pid, None, "custom", {"i": i})
    r = route("events 3", ctx)
    # The response text should contain at most 3 event rows
    assert r.text.count("•") <= 3 + 1  # header + rows


def test_events_requires_project(ctx):
    r = route("events", ctx)
    assert "no active project" in r.text.lower()


def test_advance_sync_is_stub(ctx):
    pid = ctx.store.create_project("X")
    ctx.current_project_id = pid
    r = route("advance", ctx)
    assert "async" in r.text.lower()


async def test_advance_async_runs_tick(ctx):
    pid = ctx.store.create_project("X")
    ctx.current_project_id = pid
    ctx.store.add_task(pid, "Research foo")
    agent_ctx = SimpleNamespace(project_store=ctx.store)
    r = await advance_async(ctx, agent_ctx)
    assert r.text  # non-empty message
    assert "research" in r.text.lower() or "idle" in r.text.lower()


async def test_advance_async_without_project(ctx):
    agent_ctx = SimpleNamespace(project_store=ctx.store)
    r = await advance_async(ctx, agent_ctx)
    assert "no active project" in r.text.lower()


# --------------------------------------------------------------------- prefix handling

def test_optional_project_prefix(ctx):
    r = route("project list", ctx)
    assert r.text  # same as `list`


def test_exception_in_handler_returns_error_text(ctx):
    class BoomStore:
        def list_projects(self, status_filter=None):
            raise RuntimeError("db down")

    bad = SlackContext(store=BoomStore())
    r = route("list", bad)
    assert "error" in r.text.lower() or "warning" in r.text.lower()
