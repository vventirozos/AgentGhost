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
    assert "DO NOT call manage_projects action=create" in b
    # §4UI r1 M2: on a LATER request the original message is fulfilled history
    assert "has been fulfilled" in b
    assert "Creating it is done" not in b


def test_briefing_in_the_creating_request_says_the_rest_is_still_to_do(store):
    """§4UI: in the request that created it, the create part is done and the
    REST of that message (\"create and build X\") is not."""
    from ghost_agent.utils.logging import request_id_context
    tok = request_id_context.set("c0ffee01")
    try:
        pid = store.create_project("X")
        b = build_project_briefing(store, pid)
        assert "Creating it is done; anything else the user asked for" in b
        assert "has been fulfilled" not in b
        request_id_context.set("c0ffee02")              # the next request
        assert "has been fulfilled" in build_project_briefing(store, pid)
    finally:
        request_id_context.reset(tok)


def test_briefing_points_to_batch_path(store):
    """Discoverability: for a MULTI-task request the briefing should point
    the model at the bounded batch path (autoadvance count=N|all). The old
    task_ids bulk-update hint was removed when one-task-at-a-time became the
    default — bulk now goes through the autonomous loop, not a stacked
    task_update."""
    pid = store.create_project("X")
    b = build_project_briefing(store, pid)
    assert "autoadvance count" in b


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
    # match the SECTION HEADER ("NEXT TASK: [id] …"), not the mention of
    # "the NEXT TASK shown below" inside the ONE TASK rule text.
    assert "NEXT TASK: [" not in b
    assert "OPEN TASKS (" not in b


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


async def test_the_focus_row_follows_each_create(context):
    """§4SX B0: the conversation's focus is its project-store row (the old
    `__current_project__` scratchpad sentinel is gone)."""
    from tests.focus_helpers import start_request, end_request
    start_request(context, "session:s1")
    try:
        a = _parse(await tool_manage_projects(context, action="create", title="A"))
        assert context.project_store.get_focus("session:s1") == a["created"]
        b = _parse(await tool_manage_projects(context, action="create", title="B"))
        assert context.project_store.get_focus("session:s1") == b["created"]
        assert context.scratchpad.get("__current_project__") is None
    finally:
        end_request()


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


def test_briefing_says_where_files_live(store):
    """§4QD: an already-active project's briefing named the id but not the
    path rule, and the model read '<id>/report/final-report.md'."""
    pid = store.create_project("X")
    b = build_project_briefing(store, pid)
    assert "FILES: this project's folder IS your working directory" in b
    assert f"never '{pid}/…'" in b and f"'projects/{pid}/…'" in b
    # every turn: right under the ALREADY ACTIVE line, not after the task list
    assert b.index("FILES:") < b.index("*** PACING")


def test_released_briefing_says_where_files_live_too(store, monkeypatch):
    pid = store.create_project("X")
    real = store.get_project
    monkeypatch.setattr(store, "get_project", lambda p: {**real(p), "status": "RELEASED"})
    monkeypatch.setattr(store, "get_release", lambda p: {}, raising=False)
    b = build_project_briefing(store, pid)
    assert "### PROJECT (RELEASED)" in b
    assert f"never '{pid}/…'" in b
