"""Round-1 correctness fixes from the 2026-07-25 projects re-evaluation.

R1a: NEEDS_USER/BLOCKED are no longer traps (rollup rolls BACK to ACTIVE
when open work reappears; reopen tuples cover them); RELEASED and PAUSED
are rollup-LOCKED (a store-level task write can no longer roll a released
project to DONE and fire the sweep); archive remembers the prior status
and resume RESTORES it (no more silent RELEASED→ACTIVE strip).
R1b: same-title create over RELEASED steers to create_version; a second
create_version returns the existing fork idempotently.
R1c: unregister_file removes stale deliverable rows + manifest entries
(the permanent-rehearsal-block repair path).
R1d: digest surfaces the release/version/failure milestone events.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore, ProjectStatus
from ghost_agent.tools.projects import tool_manage_projects
from ghost_agent.core.project_digest import summarize_since, render_digest
from ghost_agent.memory.scratchpad import Scratchpad


@pytest.fixture
def store(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return ProjectStore(tmp_path / "memory", sandbox_root=sandbox)


@pytest.fixture
def context(store, tmp_path):
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None, contradiction_log=None,
        current_project_id=None, llm_client=None, sandbox_manager=None,
    )


def _done_project(store, title="App"):
    pid = store.create_project(title, kind="GENERAL", goal="ship")
    tid = store.add_task(pid, "build")
    ws = Path(store.get_project(pid)["workspace_dir"])
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "out.html").write_text("<h1>x</h1>", encoding="utf-8")
    store.register_file_artifact(tid, "out.html")
    store.update_task(tid, status="DONE")
    store.update_project(pid, status="DONE")
    return pid


def _release_ctx(store, pid):
    return SimpleNamespace(project_store=store, scratchpad=None,
                           graph_memory=None, contradiction_log=None,
                           current_project_id=pid, llm_client=None,
                           sandbox_manager=None)


_REL_DIRS = "Open out.html in a browser to view the deliverable."


def _released_project(store, title="App"):
    """Sync-test helper (no running loop)."""
    import asyncio
    pid = _done_project(store, title)
    res = asyncio.run(tool_manage_projects(
        _release_ctx(store, pid), action="release", project_id=pid,
        directions=_REL_DIRS))
    assert json.loads(res)["status"] == "RELEASED"
    return pid


async def _released_project_async(store, title="App"):
    """Async-test helper (awaits inside the test's own loop)."""
    pid = _done_project(store, title)
    res = await tool_manage_projects(
        _release_ctx(store, pid), action="release", project_id=pid,
        directions=_REL_DIRS)
    assert json.loads(res)["status"] == "RELEASED"
    return pid


# ------------------------------------------------------------------- R1a

def test_needs_user_rolls_back_to_active_when_task_revived(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "ask the user something")
    store.update_task(tid, status="NEEDS_USER")
    assert store.get_project(pid)["status"] == ProjectStatus.NEEDS_USER.value
    # The user answered → the task is revived → project must be reachable.
    store.update_task(tid, status="READY")
    assert store.get_project(pid)["status"] == ProjectStatus.ACTIVE.value


def test_released_is_rollup_locked(store):
    pid = _released_project(store)
    # A store-level task write must NOT roll RELEASED → DONE (which would
    # fire the destructive sweep on a human-attested workspace).
    t2 = store.add_task(pid, "sneaky")  # add_task doesn't reopen RELEASED
    store.update_task(t2, status="DONE")
    assert store.get_project(pid)["status"] == ProjectStatus.RELEASED.value


def test_paused_is_rollup_locked(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "t")
    store.update_project(pid, status="PAUSED")
    store.update_task(tid, status="DONE")
    assert store.get_project(pid)["status"] == ProjectStatus.PAUSED.value


@pytest.mark.asyncio
async def test_archive_resume_restores_released(context, store):
    pid = await _released_project_async(store)
    res = await tool_manage_projects(context, action="archive", project_id=pid)
    assert json.loads(res).get("archived")
    assert store.get_project(pid)["status"] == "ARCHIVED"
    res2 = await tool_manage_projects(context, action="resume", project_id=pid)
    data = json.loads(res2)
    # Restored to RELEASED — attestation and guards intact, not ACTIVE.
    assert store.get_project(pid)["status"] == "RELEASED"
    assert "RELEASED" in str(data.get("note", ""))


@pytest.mark.asyncio
async def test_archive_resume_plain_project_still_activates(context, store):
    pid = store.create_project("Plain")
    await tool_manage_projects(context, action="archive", project_id=pid)
    await tool_manage_projects(context, action="resume", project_id=pid)
    assert store.get_project(pid)["status"] == "ACTIVE"


# ------------------------------------------------------------------- R1b

@pytest.mark.asyncio
async def test_same_title_create_over_released_steers_to_version(context, store):
    pid = await _released_project_async(store, title="Journal")
    res = await tool_manage_projects(context, action="create",
                                     title="Journal", goal="again")
    assert "create_version" in res and pid in res
    # No new project was minted, and the released one is untouched.
    assert len(store.list_projects()) == 1
    assert store.get_project(pid)["status"] == "RELEASED"


@pytest.mark.asyncio
async def test_double_fork_returns_existing_fork(context, store):
    pid = await _released_project_async(store)
    r1 = await tool_manage_projects(context, action="create_version",
                                    project_id=pid, description="change A")
    v2 = json.loads(r1)["project"]["id"]
    r2 = await tool_manage_projects(context, action="create_version",
                                    project_id=pid, description="change B")
    d2 = json.loads(r2)
    assert d2.get("existing_fork") is True
    assert d2["project"]["id"] == v2
    # Exactly one fork exists.
    assert len(store.list_children(pid)) == 1


# ------------------------------------------------------------------- R1c

@pytest.mark.asyncio
async def test_unregister_file_repairs_renamed_deliverable(context, store):
    pid = store.create_project("R", kind="GENERAL", goal="g")
    tid = store.add_task(pid, "build")
    ws = Path(store.get_project(pid)["workspace_dir"])
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "old.html").write_text("x", encoding="utf-8")
    store.register_file_artifact(tid, "old.html", description="the page")
    # Rename on disk → stale record would block release forever.
    (ws / "old.html").rename(ws / "new.html")
    res = await tool_manage_projects(context, action="unregister_file",
                                     project_id=pid, file_path="old.html")
    data = json.loads(res)
    assert data["artifacts_removed"] == 1
    assert data["manifest_removed"] == 1
    assert "old.html" not in store.list_deliverables(pid)
    assert "old.html" not in store.get_file_manifest(pid)


# ------------------------------------------------------------------- R1d

def test_digest_surfaces_release_and_failure_milestones(store):
    pid = _released_project(store, title="Shipped")
    # The release logged project_released; add a fork + a failure event.
    store.log_event(pid, None, "version_forked", {"child": "abc", "version": 2})
    p2 = store.create_project("Worker")
    store.log_event(p2, None, "autoadvance_failed", {"reason": "build died"})
    res = summarize_since(store, 0)
    text = render_digest(res)
    assert "RELEASED" in text
    assert "forked a new development version" in text
    assert "FAILED build during autoadvance" in text
