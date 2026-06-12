"""Eligibility gate for unconfirmed hard deletes.

Production failure: one "i don't really like it, delete it an make
something else" cascaded into SIX successive hard deletes — five of them
of projects the agent had created seconds earlier in the same request and
the user never saw — because each new turn re-read the instruction as
unfulfilled. Hard delete is permanent (row + workspace), so the tool now
refuses targets the user cannot plausibly mean by "delete it":

  eligible = request-start active project (the snapshot taken in
  handle_chat) OR a project the user's own message names (title or id).
  Contexts without the snapshot attribute (direct tool use, tests, web
  API) bypass the gate entirely.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from types import SimpleNamespace

import pytest

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
        current_project_id=None,
    )


async def _create(context, title):
    res = await tool_manage_projects(context, action="create",
                                     title=title, kind="CODING", goal="x")
    return json.loads(res)["created"]


async def test_gate_inactive_without_request_snapshot(context, store):
    """No `request_start_project_id` attribute → legacy behavior (direct
    tool use / non-chat surfaces are not gated)."""
    pid = await _create(context, "Old Thing")
    res = await tool_manage_projects(context, action="delete", project_id=pid)
    assert json.loads(res)["deleted"] is True


async def test_request_start_project_is_deletable(context, store):
    """The project that was active when the user's message arrived is
    exactly what a bare 'delete it' refers to."""
    pid = await _create(context, "The Infinite Archive")
    context.request_start_project_id = pid
    context.last_user_content = "i don't really like it, delete it an make something else"
    res = await tool_manage_projects(context, action="delete", project_id=pid)
    assert json.loads(res)["deleted"] is True


async def test_self_created_project_refused(context, store):
    """The E8 loop: after deleting the start project and creating a new
    one mid-request, deleting the NEW one must be refused."""
    start_pid = await _create(context, "The Infinite Archive")
    context.request_start_project_id = start_pid
    context.last_user_content = "i don't really like it, delete it an make something else"
    res = await tool_manage_projects(context, action="delete", project_id=start_pid)
    assert json.loads(res)["deleted"] is True

    new_pid = await _create(context, "The Algorithmic Garden")
    res = await tool_manage_projects(context, action="delete", project_id=new_pid)
    assert res.startswith("ERROR")
    assert "REFUSED" in res
    assert "BUILD" in res
    # the project survives
    assert store.get_project(new_pid) is not None


async def test_user_named_title_is_deletable(context, store):
    """A project the user names explicitly is fair game even when it was
    not active at request start."""
    pid = await _create(context, "Memory Garden")
    context.request_start_project_id = None
    context.last_user_content = "please delete the memory garden project"
    res = await tool_manage_projects(context, action="delete", project_id=pid)
    assert json.loads(res)["deleted"] is True


async def test_user_named_id_is_deletable(context, store):
    pid = await _create(context, "Some Project")
    context.request_start_project_id = None
    context.last_user_content = f"delete project {pid} now"
    res = await tool_manage_projects(context, action="delete", project_id=pid)
    assert json.loads(res)["deleted"] is True


async def test_bare_delete_it_with_no_start_project_refused(context, store):
    """'delete it' when nothing was active at request start and the
    message names nothing → there is no 'it'; refuse."""
    pid = await _create(context, "Phantom")
    context.request_start_project_id = None
    context.last_user_content = "delete it and make something else"
    res = await tool_manage_projects(context, action="delete", project_id=pid)
    assert res.startswith("ERROR") and "REFUSED" in res
    assert store.get_project(pid) is not None


async def test_archive_is_not_gated(context, store):
    """Archive is soft/reversible and stays available as the escape
    hatch the refusal message points to."""
    pid = await _create(context, "Side Quest")
    context.request_start_project_id = None
    context.last_user_content = "delete it"
    res = await tool_manage_projects(context, action="archive", project_id=pid)
    assert json.loads(res)["archived"] is True
