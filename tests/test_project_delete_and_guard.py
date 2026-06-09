"""Tests for two project-management correctness fixes:

  1. The duplicate-create guard only reuses an IN-FLIGHT same-title project
     (ACTIVE / PAUSED / NEEDS_USER). A terminal one (DONE / FAILED / BLOCKED
     / ARCHIVED) is superseded by a fresh `create` — so asking to start a
     "new" project after the old one finished no longer resurrects the old
     one (with its stale tasks + on-disk files).

  2. `delete` / `archive` resolve a project by TITLE when a name is passed
     where an id is expected, and fail LOUDLY (ERROR) when nothing matched —
     instead of silently no-op'ing and letting the caller report success.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import tool_manage_projects, _resolve_project_ref


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


async def _create(context, title, **kw):
    return json.loads(await tool_manage_projects(context, action="create",
                                                 title=title, **kw))


# ===================================================== fix 1: duplicate guard

async def test_create_reuses_active_same_title(context, store):
    pid1 = (await _create(context, "Alpha"))["created"]
    r2 = await _create(context, "Alpha")
    assert r2.get("refused") is True
    assert r2["existing_project_id"] == pid1


async def test_create_reuses_paused_same_title(context, store):
    pid1 = (await _create(context, "Delta"))["created"]
    store.update_project(pid1, status="PAUSED")
    r2 = await _create(context, "Delta")
    assert r2.get("refused") is True and r2["existing_project_id"] == pid1


@pytest.mark.parametrize("terminal", ["DONE", "FAILED", "BLOCKED"])
async def test_create_after_terminal_starts_fresh(context, store, terminal):
    pid1 = (await _create(context, "Beta"))["created"]
    store.update_project(pid1, status=terminal)
    r2 = await _create(context, "Beta")
    # Fresh project, NOT a reuse of the finished one.
    assert r2.get("refused") is not True
    assert r2.get("created") and r2["created"] != pid1


async def test_create_after_archive_starts_fresh(context, store):
    pid1 = (await _create(context, "Gamma"))["created"]
    await tool_manage_projects(context, action="archive", project_id=pid1)
    r2 = await _create(context, "Gamma")
    assert r2.get("created") and r2["created"] != pid1


# ===================================================== fix 2: delete resolution

async def test_resolve_project_ref_by_title(store):
    pid = store.create_project("My Cool Project")
    # name passed where an id is expected
    rid, err = _resolve_project_ref(store, "my cool project", "")
    assert rid == pid and err is None
    # explicit title arg
    rid2, err2 = _resolve_project_ref(store, None, "My Cool Project")
    assert rid2 == pid and err2 is None
    # nothing matches
    rid3, err3 = _resolve_project_ref(store, "nope", "")
    assert rid3 is None and err3 is None


async def test_delete_with_name_passed_as_id(context, store):
    pid1 = (await _create(context, "Epsilon"))["created"]
    await tool_manage_projects(context, action="exit")
    out = json.loads(await tool_manage_projects(
        context, action="delete", project_id="Epsilon"))  # the NAME
    assert out["deleted"] is True and out["project_id"] == pid1
    assert store.get_project(pid1) is None


async def test_delete_by_explicit_title(context, store):
    pid1 = (await _create(context, "Zeta"))["created"]
    await tool_manage_projects(context, action="exit")
    out = json.loads(await tool_manage_projects(
        context, action="delete", title="Zeta"))
    assert out["deleted"] is True
    assert store.get_project(pid1) is None


async def test_delete_nonexistent_fails_loudly(context, store):
    out = await tool_manage_projects(context, action="delete",
                                     project_id="does-not-exist")
    assert out.startswith("ERROR") and "NOTHING was" in out


async def test_delete_ambiguous_title_errors_and_keeps_both(context, store):
    pid1 = (await _create(context, "Eta"))["created"]
    store.update_project(pid1, status="DONE")     # terminal → next is fresh
    pid2 = (await _create(context, "Eta"))["created"]
    assert pid1 != pid2
    await tool_manage_projects(context, action="exit")
    out = await tool_manage_projects(context, action="delete", title="Eta")
    assert out.startswith("ERROR") and "ambiguous" in out
    # Nothing deleted — both survive.
    assert store.get_project(pid1) and store.get_project(pid2)


async def test_archive_by_title(context, store):
    pid1 = (await _create(context, "Theta"))["created"]
    await tool_manage_projects(context, action="exit")
    out = json.loads(await tool_manage_projects(
        context, action="archive", title="Theta"))
    assert out["archived"] is True
    assert store.get_project(pid1)["status"] == "ARCHIVED"


async def test_archive_nonexistent_fails_loudly(context, store):
    out = await tool_manage_projects(context, action="archive",
                                     project_id="ghost-id")
    assert out.startswith("ERROR") and "NOTHING was" in out
