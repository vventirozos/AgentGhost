"""The pixel check back on the main model (planned 2026-11-07: Gemma replaced by a
text-only model on the worker node, so `GHOST_VISUAL_CHECK_POOL` goes away).

The check runs on the vision pool, which is the main server, up to twice per
picture turn: inside the turn, and as a deferred verdict after the reply. The
question each test answers is what the conversation's NEXT request starts from,
compared with the same request and no check at all.

Each flow uses the real hooks and the real `note_main_slot_prompt` for the
bookkeeping; only llama's HTTP and the files are the §4RO/§4RV harness's fakes.
These pin which file comes back, not the tokens llama then re-reads.
"""
from __future__ import annotations

import pytest

from ghost_agent.core import head_saves as H
from ghost_agent.core import llm as L
from ghost_agent.utils.logging import request_id_context
from tests.test_4ro_head_saves import _switch_env, _turn, _warm_with_file, warm  # noqa: F401
from tests.test_4rv_conv_keep import _conv_env, conv, cwarm  # noqa: F401

CHECK = {"messages": [{"role": "user", "content": [
    {"type": "text", "text": "Does this picture show what the reply says?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}
#: (calls, first prompt, largest prompt) of the request that made the picture
SHORT = (3, 30000, 33000)                  # llama still has the checkpoint its follow-up resumes from
LONG = (21, 30000, 70000)                  # §4SO: that checkpoint is gone
FLOWS = ["no_check", "in_turn", "deferred", "in_turn_and_deferred"]


async def _request_holds_the_slot(agent, llm, st, monkeypatch, span):
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    st["restores"].clear()
    st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch, key="sess-a", slot=70000)
    _conv_env(agent, llm, st, monkeypatch, slot=70000)
    llm._slot_request = "req-1"
    llm._slot_request_calls, llm._slot_request_start, llm._slot_request_end = span
    agent._switch_done = {"req-1": {"head"}}


async def _in_turn_check(agent, llm):
    await agent._side_call_before(CHECK)
    L.note_main_slot_prompt(llm, CHECK, "verify-visual", usage=(2000, 50))


async def _deferred_check(agent, llm):
    await agent._conv_before_background(CHECK, "deferred-verdict")
    L.note_main_slot_prompt(llm, CHECK, "deferred-verdict", background=True)


async def _next_request_after(agent, llm, st, monkeypatch, span, flow):
    """req-1 makes its picture, its checks run, then req-2 opens the conversation."""
    tok = request_id_context.set("req-1")
    try:
        await _request_holds_the_slot(agent, llm, st, monkeypatch, span)
        if "in_turn" in flow:
            await _in_turn_check(agent, llm)
        if "deferred" in flow:
            llm.foreground_requests = 0    # the reply has gone out
            await _deferred_check(agent, llm)
    finally:
        request_id_context.reset(tok)
    tok = request_id_context.set("req-2")
    try:
        st["restores"].clear()
        await agent._slot_switch_before(_turn(st))
    finally:
        request_id_context.reset(tok)
    return list(st["restores"])


@pytest.mark.parametrize("flow", ["in_turn", "deferred", "in_turn_and_deferred"])
async def test_after_a_short_request_the_check_costs_a_restore_not_a_re_read(cwarm, monkeypatch, flow):
    agent, llm, st = cwarm
    restores = await _next_request_after(agent, llm, st, monkeypatch, SHORT, flow)
    assert restores == [H.conv_key("id", "sess-a")]
    assert any("restored from disk in" in m for m in st["lines"])


async def test_a_short_request_without_a_check_resumes_from_the_slot(cwarm, monkeypatch):
    agent, llm, st = cwarm
    assert await _next_request_after(agent, llm, st, monkeypatch, SHORT, "no_check") == []


@pytest.mark.parametrize("flow", FLOWS)
async def test_after_a_long_request_the_check_changes_nothing_the_head_comes_first(cwarm, monkeypatch, flow):
    """§4SO: the follow-up parts from a long request below its surviving
    checkpoints, in the slot and in the file alike, so restoring either would be
    a re-read from token 0. The head first is the good outcome, with or without a
    check: a fix sending this follow-up to the file would make it worse."""
    agent, llm, st = cwarm
    restores = await _next_request_after(agent, llm, st, monkeypatch, LONG, flow)
    assert len(restores) == 1 and restores[0] != H.conv_key("id", "sess-a")


@pytest.mark.parametrize("background,want", [(False, ["side"]), (True, ["background"])])
async def test_the_client_saves_first_on_both_paths_of_the_check(background, want):
    """In the turn it is a foreground call, after the reply a background one:
    each reaches its own save hook when the vision pool is the main server."""
    from tests.test_4rp_slot_keep import _bare_client
    c = _bare_client()
    c.vision_clients = [{"url": "http://127.0.0.1:8088"}]
    seen = []

    async def side(payload):
        seen.append("side")

    async def in_background(payload, label):
        seen.append("background")
        return False

    async def do(payload, *a, **k):
        return {"choices": [{"message": {"content": "x"}}], "_ghost_leg": {"served_by": "vision"}}
    c.before_side_main_dispatch, c.before_background_dispatch, c._do_chat_completion = side, in_background, do
    await c.chat_completion(CHECK, use_vision=True, is_background=background, task_label="verify-visual")
    assert seen == want


async def test_a_failed_save_before_the_check_still_brings_the_head_back(cwarm, monkeypatch):
    """The cost is the conversation re-read, never the head."""
    agent, llm, st = cwarm
    tok = request_id_context.set("req-1")
    try:
        await _request_holds_the_slot(agent, llm, st, monkeypatch, SHORT)
        monkeypatch.setattr(H, "save", lambda *a, **k: (False, "HTTP 500"))
        await _in_turn_check(agent, llm)
    finally:
        request_id_context.reset(tok)
    assert st["conv_saves"] == []
    tok = request_id_context.set("req-2")
    try:
        st["restores"].clear()
        await agent._slot_switch_before(_turn(st))
    finally:
        request_id_context.reset(tok)
    assert len(st["restores"]) == 1 and st["restores"][0] != H.conv_key("id", "sess-a")
