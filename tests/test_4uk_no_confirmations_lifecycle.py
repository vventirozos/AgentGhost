"""§4UK (operator 11-12: "i want all confirmations, everywhere to be removed … that's why everything is
sandboxed"; "keep clarifying questions"): the model-facing manage_projects release / unrelease / delete run in
the turn that asks — no preview, no token. Only the walls on WHO may act stay."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.projects import tool_manage_projects_for_model


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def context(tmp_path, store):
    from ghost_agent.memory.scratchpad import Scratchpad
    return SimpleNamespace(project_store=store, scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                           graph_memory=None, workspace_model=None, current_project_id=None,
                           last_user_content="")


def _as(rid):
    from ghost_agent.utils.logging import request_id_context
    return request_id_context.set(rid)


async def test_the_owners_delete_runs_at_once(context, store):
    from ghost_agent.utils.logging import request_id_context
    pid = store.create_project("Old app")
    tok = _as("req-uj-owner-1")
    try:
        res = str(await tool_manage_projects_for_model(context, action="delete", project_id=pid))
    finally:
        request_id_context.reset(tok)
    assert "confirmation_needed" not in res and "confirm_token" not in res
    assert not store.get_project(pid)


async def test_the_owners_unrelease_runs_and_release_never_previews(context, store):
    from ghost_agent.utils.logging import request_id_context
    pid = store.create_project("Shipped app")
    store.update_project(pid, status="RELEASED")
    tok = _as("req-uj-owner-2")
    try:
        res = str(await tool_manage_projects_for_model(context, action="unrelease", project_id=pid))
        assert "confirmation_needed" not in res
        assert str(store.get_project(pid).get("status")).upper() != "RELEASED"
        store.update_project(pid, status="DONE")
        res = str(await tool_manage_projects_for_model(
            context, action="release", project_id=pid,
            directions="Open index.html in a browser; nothing else is needed to run it."))
        # it runs (its own rehearsal decides) — never a preview with a token
        assert "confirmation_needed" not in res and "confirm_token" not in res
    finally:
        request_id_context.reset(tok)


async def test_a_probe_cannot_delete_an_owner_project(context, store):
    from ghost_agent.utils.logging import request_id_context
    pid = store.create_project("Owner app")
    tok = _as("probe-uj-1")
    try:
        res = str(await tool_manage_projects_for_model(context, action="delete", project_id=pid))
    finally:
        request_id_context.reset(tok)
    assert "NOT done" in res and store.get_project(pid)


async def test_a_probe_may_delete_its_own_project(context, store):
    from ghost_agent.utils.logging import request_id_context
    pid = store.create_project("Probe scratch")
    store.update_project(pid, metadata={"probe_created": True})
    tok = _as("probe-uj-2")
    try:
        await tool_manage_projects_for_model(context, action="delete", project_id=pid)
    finally:
        request_id_context.reset(tok)
    assert not store.get_project(pid)


async def test_a_request_that_read_outside_content_cannot_delete(context, store):
    from ghost_agent.utils.logging import request_id_context
    from ghost_agent.utils.provenance import note_untrusted_content
    pid = store.create_project("Owner app 2")
    tok = _as("req-uj-owner-3")
    try:
        note_untrusted_content("web_fetch", req_id="req-uj-owner-3")
        res = str(await tool_manage_projects_for_model(context, action="delete", project_id=pid))
    finally:
        request_id_context.reset(tok)
    assert "NOT done" in res and "outside content" in res and store.get_project(pid)


def test_the_prompt_says_no_confirmations_but_keeps_clarifying_questions():
    from ghost_agent.core.prompts import SYSTEM_PROMPT
    assert "NO CONFIRMATIONS" in SYSTEM_PROMPT
    assert "clarifying question, never a request for approval" in SYSTEM_PROMPT


def test_the_execute_description_no_longer_asks_for_a_confirm():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    desc = next(t["function"]["description"] for t in TOOL_DEFINITIONS if t["function"]["name"] == "execute")
    assert "and confirmed" not in desc and "no confirmation" in desc


def test_a_lifecycle_result_no_longer_reads_as_a_go_ahead_wait():
    from ghost_agent.core.agent import _awaits_owner_go_ahead
    row = {"name": "manage_projects", "content": json.dumps({"confirmation_needed": True})}
    assert not _awaits_owner_go_ahead(row)
    row = {"name": "manage_projects", "content": json.dumps({"plan_shown": True, "tasks": []})}
    assert _awaits_owner_go_ahead(row)


def test_the_web_console_has_no_confirm_dialogs():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "interface" / "static"
    for name in ("sessions.js", "app.js"):
        txt = (root / name).read_text()
        assert "window.confirm(" not in txt and "!confirm(" not in txt, name
