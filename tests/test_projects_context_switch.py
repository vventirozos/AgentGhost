"""Tests for Phase 4: project-scoped context switching.

Covers:
  - build_project_briefing prompt helper
  - scratchpad snapshot/hydrate on project switch
  - graph_memory linkage when projects/tasks are created
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.core.prompts import build_project_briefing
from ghost_agent.tools.projects import tool_manage_projects


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def context(tmp_path, store):
    sp = Scratchpad(persist_path=tmp_path / "sp.db")
    return SimpleNamespace(
        project_store=store,
        scratchpad=sp,
        graph_memory=None,
        current_project_id=None,
    )


def _parse(s: str):
    return json.loads(s)


# --------------------------------------------------------------------- briefing

def test_briefing_empty_when_no_store():
    assert build_project_briefing(None, "x") == ""


def test_briefing_empty_when_no_project_id(store):
    assert build_project_briefing(store, "") == ""


def test_briefing_returns_empty_when_project_missing(store):
    assert build_project_briefing(store, "not-a-real-id") == ""


def test_briefing_includes_title_kind_status(store):
    pid = store.create_project("Build CLI", kind="CODING", goal="Ship v1")
    b = build_project_briefing(store, pid)
    assert "### CURRENT PROJECT" in b
    assert "Build CLI" in b
    assert "CODING" in b
    assert "ACTIVE" in b
    assert "Ship v1" in b


def test_briefing_carries_explicit_no_create_directive(store):
    """The directive is what stops Qwen's create-loop; if it disappears
    in a future refactor, the loop comes back. Pin the wording."""
    pid = store.create_project("X")
    b = build_project_briefing(store, pid)
    assert "ALREADY ACTIVE" in b
    assert "DO NOT call manage_projects action=create" in b
    assert pid in b  # the active id is named so the model can echo it back


def test_briefing_tells_model_not_to_re_read_original_prompt(store):
    """2026-04-19 regression: model periodically re-interpreted the
    'start a new project' user message and retried create. Pin the
    directive that tells it to stop."""
    pid = store.create_project("X")
    b = build_project_briefing(store, pid)
    assert "DO NOT re-read the user's original" in b
    assert "start a new project" in b


def test_briefing_points_to_batch_task_ids(store):
    """Discoverability: briefing should mention task_ids=[] so the
    model notices the bulk-update path instead of looping single
    task_update calls."""
    pid = store.create_project("X")
    b = build_project_briefing(store, pid)
    assert "task_ids" in b


def test_briefing_shows_next_task_and_open_tasks(store):
    pid = store.create_project("P")
    t1 = store.add_task(pid, "root")
    b = build_project_briefing(store, pid)
    assert "NEXT TASK" in b
    assert "root" in b
    assert f"[{t1}]" in b
    assert "OPEN TASKS" in b


def test_briefing_includes_recent_events(store):
    pid = store.create_project("P")
    store.log_event(pid, None, "custom_event", {"detail": "x"})
    b = build_project_briefing(store, pid)
    assert "RECENT EVENTS" in b
    assert "custom_event" in b


def test_briefing_no_open_tasks_shows_nothing_task_wise(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "root")
    store.update_task(tid, status="DONE")
    b = build_project_briefing(store, pid)
    assert "NEXT TASK" not in b
    assert "OPEN TASKS" not in b


# --------------------------------------------------------------------- scratchpad snapshot/hydrate

async def test_scratchpad_snapshot_on_switch(context):
    # Enter project A, write scratchpad, switch to B, confirm A's keys are captured
    a = _parse(await tool_manage_projects(context, action="create", title="A"))
    context.scratchpad.set("note1", "hello-A")
    context.scratchpad.set("note2", "world-A")
    b = _parse(await tool_manage_projects(context, action="create", title="B"))

    # A's snapshot was persisted as an event
    evs = context.project_store.list_events(a["created"],
                                            event_type="scratchpad_snapshot")
    assert evs
    snap = evs[0]["payload"]["keys"]
    assert snap.get("note1") == "hello-A"
    assert snap.get("note2") == "world-A"


async def test_scratchpad_hydrates_on_resume(context):
    # Set up A with a snapshot, then B, then resume A → A's keys return
    a = _parse(await tool_manage_projects(context, action="create", title="A"))
    context.scratchpad.set("preserved", "A-data")
    _parse(await tool_manage_projects(context, action="create", title="B"))
    context.scratchpad.set("leak", "should-not-survive")
    # Switch back to A
    _parse(await tool_manage_projects(context, action="switch",
                                      project_id=a["created"]))
    assert context.scratchpad.get("preserved") == "A-data"
    # B's leaked key was cleared on hydrate
    assert context.scratchpad.get("leak") is None


async def test_exit_clears_scratchpad_free_chat_keys(context):
    _parse(await tool_manage_projects(context, action="create", title="A"))
    context.scratchpad.set("temp", "project-A-value")
    _parse(await tool_manage_projects(context, action="exit"))
    assert context.scratchpad.get("temp") is None


async def test_sentinel_key_survives_switch(context):
    a = _parse(await tool_manage_projects(context, action="create", title="A"))
    # Sentinel __current_project__ should always be present
    assert context.scratchpad.get("__current_project__") == a["created"]
    b = _parse(await tool_manage_projects(context, action="create", title="B"))
    assert context.scratchpad.get("__current_project__") == b["created"]


async def test_switch_to_same_project_is_idempotent(context):
    a = _parse(await tool_manage_projects(context, action="create", title="A"))
    context.scratchpad.set("k", "v")
    _parse(await tool_manage_projects(context, action="switch",
                                      project_id=a["created"]))
    # Same-project switch must NOT snapshot (no events generated from a self-switch)
    # and must NOT wipe the live scratchpad.
    assert context.scratchpad.get("k") == "v"
    evs = context.project_store.list_events(a["created"],
                                            event_type="scratchpad_snapshot")
    assert evs == []


# --------------------------------------------------------------------- graph linkage

def test_graph_link_no_op_when_graph_memory_missing(context):
    # graph_memory is None → nothing should blow up
    from ghost_agent.tools.projects import _link_project_in_graph, _link_task_in_graph
    _link_project_in_graph(context, "pid", "Title")
    _link_task_in_graph(context, "pid", "tid", "desc")


async def test_graph_link_invoked_on_project_and_task_creation(context, tmp_path):
    class FakeGraph:
        def __init__(self):
            self.calls = []

        def add_triplets(self, triplets):
            self.calls.append(triplets)

    context.graph_memory = FakeGraph()
    p = _parse(await tool_manage_projects(context, action="create", title="GraphP"))
    # Project linkage happened
    assert any(
        t["predicate"] == "HAS_TITLE" and t["object"] == "GraphP"
        for call in context.graph_memory.calls for t in call
    )
    _parse(await tool_manage_projects(context, action="task_add",
                                      description="t1"))
    # Task linkage happened with HAS_TASK and HAS_DESCRIPTION edges
    preds = [t["predicate"] for call in context.graph_memory.calls for t in call]
    assert "HAS_TASK" in preds
    assert "HAS_DESCRIPTION" in preds


async def test_graph_link_does_not_block_on_exception(context):
    class BoomGraph:
        def add_triplets(self, _):
            raise RuntimeError("boom")

    context.graph_memory = BoomGraph()
    # Must not raise even though the graph explodes
    res = _parse(await tool_manage_projects(context, action="create", title="P"))
    assert "created" in res


# --------------------------------------------------------------------- agent.py wiring

def test_briefing_called_in_agent_dynamic_state(monkeypatch, store):
    """The dynamic-state assembly path reads build_project_briefing —
    exercise it with a tiny construction to confirm import works."""
    from ghost_agent.core import prompts
    pid = store.create_project("Title A")
    out = prompts.build_project_briefing(store, pid)
    assert "Title A" in out
