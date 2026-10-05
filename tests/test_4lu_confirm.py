"""§4LU — a preview → confirm flow finds the conversation's preview when the
user's "yes" arrives without the token (the chat API carries reply text, not
tool results, so the model no longer has it)."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.utils.logging import conversation_key_context, request_id_context


def _turn(rid, conv):
    return request_id_context.set(rid), conversation_key_context.set(conv)


def _end(toks):
    request_id_context.reset(toks[0])
    conversation_key_context.reset(toks[1])


def _plan(kind, rid, **extra):
    from ghost_agent.tools.memory import _store_plan
    return _store_plan({"kind": kind, "rid": rid, "ts": time.time(), **extra})


def test_a_yes_finds_this_conversations_preview_and_not_anothers():
    from ghost_agent.tools.memory import _resolve_plan
    t = _turn("req-1", "convA")
    try:
        tok = _plan("forget", "req-1", items=[])
    finally:
        _end(t)
    t = _turn("req-2", "convA")
    try:
        got_tok, plan = _resolve_plan("yes", "forget")
    finally:
        _end(t)
    assert got_tok == tok and plan is not None
    t = _turn("req-3", "convB")
    try:
        assert _resolve_plan("yes", "forget")[1] is None
    finally:
        _end(t)


def test_without_a_conversation_there_is_no_fallback():
    from ghost_agent.tools.memory import _resolve_plan
    t = _turn("req-1", "")
    try:
        _plan("forget", "req-1", items=[])
        assert _resolve_plan("yes", "forget")[1] is None
    finally:
        _end(t)


async def test_a_forget_confirmed_with_yes_runs_in_a_later_turn_only():
    from ghost_agent.tools.memory import forget_execute
    t = _turn("req-10", "convF")
    try:
        _plan("forget", "req-10", items=[])
        same = await forget_execute("yes", "all")
    finally:
        _end(t)
    assert "has not answered yet" in same                     # same turn: still refused
    t = _turn("req-11", "convF")
    try:
        later = await forget_execute("yes", "all")
    finally:
        _end(t)
    assert "unknown or expired" not in later and "has not answered" not in later


async def test_a_reset_all_confirmed_with_yes_finds_its_preview(monkeypatch):
    import ghost_agent.tools.memory as M
    t = _turn("req-20", "convR")
    try:
        _plan("reset_all", "req-20")
    finally:
        _end(t)
    t = _turn("req-21", "convR")
    try:
        out = await M.tool_knowledge_base(action="reset_all", confirm="yes",
                                          sandbox_dir=None, memory_system=MagicMock())
    finally:
        _end(t)
    assert "unknown or expired" not in str(out)


async def test_a_project_delete_confirmed_with_yes(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    ctx = SimpleNamespace(project_store=store, scratchpad=None, graph_memory=None,
                          workspace_model=None, current_project_id=None, last_user_content="")
    pid = store.create_project("Old app")
    t = _turn("req-30", "convP")
    try:
        prev = json.loads(await tool_manage_projects_for_model(ctx, action="delete", project_id=pid))
        assert prev.get("confirmation_needed")
    finally:
        _end(t)
    t = _turn("req-31", "convP")
    try:
        await tool_manage_projects_for_model(ctx, action="delete", project_id=pid, confirm_token="yes")
    finally:
        _end(t)
    assert not store.get_project(pid)


async def test_a_yes_never_confirms_another_projects_preview(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    ctx = SimpleNamespace(project_store=store, scratchpad=None, graph_memory=None,
                          workspace_model=None, current_project_id=None, last_user_content="")
    a = store.create_project("App A")
    b = store.create_project("App B")
    t = _turn("req-40", "convQ")
    try:
        await tool_manage_projects_for_model(ctx, action="delete", project_id=a)
    finally:
        _end(t)
    t = _turn("req-41", "convQ")
    try:
        out = await tool_manage_projects_for_model(ctx, action="delete", project_id=b, confirm_token="yes")
    finally:
        _end(t)
    assert store.get_project(b) and store.get_project(a) and "mismatched" in out


async def test_a_yes_finds_the_right_project_among_two_previews(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    ctx = SimpleNamespace(project_store=store, scratchpad=None, graph_memory=None,
                          workspace_model=None, current_project_id=None, last_user_content="")
    a = store.create_project("App A")
    b = store.create_project("App B")
    for rid, pid in (("req-50", a), ("req-51", b)):
        t = _turn(rid, "convM")
        try:
            await tool_manage_projects_for_model(ctx, action="delete", project_id=pid)
        finally:
            _end(t)
    t = _turn("req-52", "convM")
    try:
        await tool_manage_projects_for_model(ctx, action="delete", project_id=a, confirm_token="yes")
    finally:
        _end(t)
    assert not store.get_project(a) and store.get_project(b)


def test_the_request_records_its_conversation():
    from ghost_agent.tools.projects import reconcile_conversation
    ctx = SimpleNamespace(scratchpad=None, current_project_id=None)
    tok = conversation_key_context.set("")
    try:
        reconcile_conversation(ctx, "convZ")
        assert conversation_key_context.get() == "convZ"
    finally:
        conversation_key_context.reset(tok)
