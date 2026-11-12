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




def test_without_a_conversation_there_is_no_fallback():
    from ghost_agent.tools.memory import _resolve_plan
    t = _turn("req-1", "")
    try:
        _plan("forget", "req-1", items=[])
        assert _resolve_plan("yes", "forget")[1] is None
    finally:
        _end(t)




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








async def test_the_request_records_its_conversation():
    """§4SX B0: handle_chat sets the conversation contextvar from the
    request's scope (its session id) — the preview → confirm pairing key."""
    from unittest.mock import AsyncMock
    from tests.helpers import make_agent, FakeBgTasks
    agent = make_agent()
    seen = []

    async def _llm(payload, *a, **kw):
        seen.append(conversation_key_context.get())
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=_llm)
    tok = conversation_key_context.set("")
    try:
        await agent.handle_chat({"messages": [{"role": "user", "content": "hello"}],
                                 "session_id": "zz9"}, FakeBgTasks(), request_id="web-1")
    finally:
        conversation_key_context.reset(tok)
    assert seen and seen[0] == "session:zz9"