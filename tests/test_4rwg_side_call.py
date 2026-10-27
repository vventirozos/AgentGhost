"""§4RW-g (2026-10-27): a request's own picture look keeps its conversation.

The render look runs on the vision pool, which is the main server
(http://127.0.0.1:8088). The client ran the agent's switch only for calls
marked main, so nothing saved the conversation before the look; and the
request's next call found its restore "already decided" for the request. Live
probe: the reply after the look re-read 30,112 tokens from token 0 (179 s).
Now the look saves the conversation first, and the next call that opens with
the head restores it.
"""
from __future__ import annotations

import pathlib
import time

import pytest

from ghost_agent.core import head_saves as H
from ghost_agent.core import llm as L
from ghost_agent.utils.logging import request_id_context, slot_session_context
from tests.test_4ro_head_saves import _store, _switch_env, _turn, _warm_with_file, warm  # noqa: F401
from tests.test_4rv_conv_keep import _conv_env, conv, cwarm  # noqa: F401

LOOK = {"messages": [{"role": "user", "content": [
    {"type": "text", "text": "What does this picture show?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}


@pytest.fixture
def rid():
    tok = request_id_context.set("req-1")
    yield "req-1"
    request_id_context.reset(tok)


async def _held(agent, llm, st, monkeypatch, slot=40000):
    """sess-a's request req-1 holds the slot and already decided its restore."""
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    st["restores"].clear()
    st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch, key="sess-a", slot=slot)
    _conv_env(agent, llm, st, monkeypatch, slot=slot)
    llm._slot_request = "req-1"
    agent._switch_done = {"req-1": {"head"}}


def _look_ran(llm):
    """What `note_main_slot_prompt` leaves after the look (no head)."""
    llm._head_displaced = True
    llm._last_main_call_end = llm._slot_session_used = time.monotonic()


async def test_the_reply_after_a_look_gets_its_conversation_back(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch)
    await agent._side_call_before(LOOK)
    ckey = H.conv_key("id", "sess-a")
    assert [k for k, _n, _a in st["conv_saves"]] == [ckey]
    assert llm._slot_session == "sess-a"                 # saved, still this request's
    _look_ran(llm)
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == [ckey]
    assert llm._slot_session == "sess-a" and llm._head_displaced is False
    assert any("restored from disk in" in m for m in st["lines"])


async def test_without_a_look_the_request_still_decides_once(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch)
    _look_ran(llm)                                       # displaced by something else of its own
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == []


async def test_a_conversation_too_short_to_save_gets_the_head_back(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch, slot=28213 + 100)
    await agent._side_call_before(LOOK)
    assert st["conv_saves"] == []
    _look_ran(llm)
    await agent._slot_switch_before(_turn(st))
    assert len(st["restores"]) == 1 and st["restores"][0] != H.conv_key("id", "sess-a")
    assert any("restored the head before this request after its own picture look" in m for m in st["lines"])


async def test_two_looks_in_one_request_each_get_it_back(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch)
    for _ in range(2):
        llm._last_main_call_end = time.monotonic() - 5   # the head call claimed it again
        llm._slot_session_used = time.monotonic()
        await agent._side_call_before(LOOK)
        _look_ran(llm)
        await agent._slot_switch_before(_turn(st))
    assert len(st["conv_saves"]) == 2 and len(st["restores"]) == 2


@pytest.mark.parametrize("case", ["other_owner", "displaced", "head_call", "off"])
async def test_the_look_saves_nothing_when_it_is_not_this_requests_slot(cwarm, monkeypatch, rid, case):
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch)
    payload = LOOK
    if case == "other_owner":
        llm._slot_session = "sess-z"
    elif case == "displaced":
        llm._head_displaced = True
    elif case == "head_call":
        payload = _turn(st)
    else:
        monkeypatch.delenv(L.CONV_KEEP_ENV)
    await agent._side_call_before(payload)
    assert st["conv_saves"] == [] and getattr(agent, "_side_call_rid", None) is None


async def test_the_hook_is_installed_with_the_head_plan(warm):
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    assert llm.before_side_main_dispatch == agent._side_call_before


# ── the client ───────────────────────────────────────────────────────────────

def _client(vision_url):
    from tests.test_4rp_slot_keep import _bare_client
    c = _bare_client()
    c.vision_clients = [{"url": vision_url}]
    c.critic_clients = [{"url": "http://10.0.0.5:8080"}]
    seen = []

    async def side(payload):
        seen.append("side")

    async def main(payload):
        seen.append("main")

    async def do(payload, *a, **k):
        return {"choices": [{"message": {"content": "x"}}], "_ghost_leg": {"served_by": "vision"}}
    c.before_side_main_dispatch, c.before_main_dispatch, c._do_chat_completion = side, main, do
    return c, seen


@pytest.mark.parametrize("url,kw,want", [
    ("http://127.0.0.1:8088", {"use_vision": True}, ["side"]),          # live: vision = the main server
    ("http://127.0.0.1:8088/v1/", {"use_vision": True}, ["side"]),
    ("http://10.0.0.9:8088", {"use_vision": True}, []),                 # a vision node of its own
    ("http://127.0.0.1:8088", {"use_critic": True}, []),                # critic on another box
    ("http://127.0.0.1:8088", {"use_vision": True, "off_main_only": True}, []),
    ("http://127.0.0.1:8088", {}, ["main"]),
])
async def test_the_client_runs_the_side_hook_for_a_pool_on_the_main_server(url, kw, want):
    c, seen = _client(url)
    await c.chat_completion(LOOK, **kw)
    assert seen == want


async def test_a_failing_side_hook_never_fails_the_call():
    c, _seen = _client("http://127.0.0.1:8088")

    async def boom(payload):
        raise RuntimeError("boom")
    c.before_side_main_dispatch = boom
    out = await c.chat_completion(LOOK, use_vision=True)
    assert out["choices"][0]["message"]["content"] == "x"


# ── review r1: the marker ────────────────────────────────────────────────────

async def test_after_the_restore_another_displacement_does_not_restore_again(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch)
    await agent._side_call_before(LOOK)
    _look_ran(llm)
    await agent._slot_switch_before(_turn(st))
    assert len(st["restores"]) == 1
    _look_ran(llm)                                       # displaced by something that is not a look
    await agent._slot_switch_before(_turn(st))
    assert len(st["restores"]) == 1


async def test_another_requests_marker_does_not_restore(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch)
    agent._side_call_rid = "req-0"                       # an earlier request's look
    _look_ran(llm)
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == []


async def test_the_marker_alone_restores_nothing(cwarm, monkeypatch, rid):
    """The agent's side only: an undisplaced slot is left alone. (Through the
    client, a look that raised counts as displacing — see the failed-look test:
    it then restores the file just saved, ~1 s.)"""
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch)
    await agent._side_call_before(LOOK)
    await agent._slot_switch_before(_turn(st))           # not displaced: the conversation is there
    assert st["restores"] == []


async def test_a_look_while_another_turn_holds_the_lock_saves_nothing(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch)
    agent._turn_lock_rid = "req-other"
    await agent._side_call_before(LOOK)
    assert st["conv_saves"] == [] and getattr(agent, "_side_call_rid", None) is None


async def test_a_failed_look_counts_as_displacing_the_slot():
    c, _seen = _client("http://127.0.0.1:8088")
    L.note_head_reference(c, {"messages": [{"role": "system", "content": "S"}], "tools": [{"t": 1}]})
    c._head_displaced = False

    async def timeout(payload, *a, **k):
        raise TimeoutError("look timed out mid-prefill")
    c._do_chat_completion = timeout
    with pytest.raises(TimeoutError):
        await c.chat_completion(LOOK, use_vision=True)
    assert c._head_displaced is True


async def test_a_failed_main_call_is_not_marked_by_the_side_path():
    c, _seen = _client("http://127.0.0.1:8088")
    L.note_head_reference(c, {"messages": [{"role": "system", "content": "S"}], "tools": [{"t": 1}]})
    c._head_displaced = False

    async def boom(payload, *a, **k):
        raise TimeoutError("x")
    c._do_chat_completion = boom
    with pytest.raises(TimeoutError):
        await c.chat_completion(LOOK)
    assert c._head_displaced is False


async def test_a_marker_on_an_unowned_slot_that_was_not_displaced_restores_nothing(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch)
    await agent._side_call_before(LOOK)
    llm._slot_session = None                             # released, and the look never ran
    st["lines"].clear()
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == [] and not any("restored" in m for m in st["lines"])
    assert agent._side_call_rid == "req-1"               # (still waiting for its look)


# ── §4RW-g b: a request without a conversation key ──────────────────────────
# Live 2026-10-27: probes and scheduled tasks carry no key — the apple probe's
# reply after its look re-read 30,109 tokens from token 0 (179 s).

async def _keyless(agent, llm, st, monkeypatch):
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    st["restores"].clear()
    st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="", slot=30000)
    _conv_env(agent, llm, st, monkeypatch, slot=30000)
    llm._slot_session = None                             # (the real flow writes no decision for it)


async def test_a_keyless_requests_reply_after_its_look_gets_the_head_back(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _keyless(agent, llm, st, monkeypatch)
    await agent._side_call_before(LOOK)
    assert st["conv_saves"] == []                        # nothing to file it under
    assert agent._side_call_rid == "req-1"
    _look_ran(llm)
    await agent._slot_switch_before(_turn(st))
    assert len(st["restores"]) == 1 and st["restores"][0] != H.conv_key("id", "")
    assert any("restored the head before this request after its own picture look" in m
               and "re-reads its part past the head" in m for m in st["lines"])
    assert agent._side_call_rid is None


async def test_a_keyless_request_without_a_look_is_left_alone(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _keyless(agent, llm, st, monkeypatch)
    _look_ran(llm)                                       # displaced, but not by its own look
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == []


async def test_a_keyless_marker_from_another_request_is_ignored(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _keyless(agent, llm, st, monkeypatch)
    agent._side_call_rid = "req-0"
    _look_ran(llm)
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == []


async def test_a_keyless_look_on_a_displaced_slot_sets_no_marker(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _keyless(agent, llm, st, monkeypatch)
    llm._head_displaced = True
    await agent._side_call_before(LOOK)
    assert getattr(agent, "_side_call_rid", None) is None


async def test_a_keyless_request_never_restores_a_conversation_file(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _keyless(agent, llm, st, monkeypatch)
    _store(pathlib.Path(H.head_dir()), H.conv_key("id", ""), 41000)   # (none is ever written; pinned anyway)
    await agent._side_call_before(LOOK)
    _look_ran(llm)
    await agent._slot_switch_before(_turn(st))
    assert H.conv_key("id", "") not in st["restores"] and len(st["restores"]) == 1


async def test_a_keyless_marker_on_a_slot_that_was_not_displaced_restores_nothing(cwarm, monkeypatch, rid):
    """Review: a look that never reached the slot, or an earlier run of the
    same sched- id, left the marker — the request's own conversation stays."""
    agent, llm, st = cwarm
    await _keyless(agent, llm, st, monkeypatch)
    agent._side_call_rid = "req-1"
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == [] and not any("restored" in m for m in st["lines"])


# ── §4RW-g c: a member's look ────────────────────────────────────────────────
# Live 2026-10-27 slack-7e: the member's call after its render look re-read
# 16.6k tokens (94 s) — the members' head is restored once per request.

def _member_env(agent, llm, st, monkeypatch, restores):
    from ghost_agent.core import agent as A
    hs = {"base": "http://127.0.0.1:8088", "dir": pathlib.Path(H.head_dir()), "key": "mm",
          "identity": "id", "have": True, "n_tokens": 9196, "shape": None, "payload": {}}

    async def plan(_llm):
        return hs
    monkeypatch.setattr(agent, "_member_head_plan", plan)
    monkeypatch.setattr(L, "opens_with_head", lambda p, ref: p is not LOOK)
    monkeypatch.setattr(A, "requester_is_member", lambda: True)

    async def riw(_llm, _hs, late, *, skip_n, park=False):
        restores.append(_hs["key"])
        return (True, "9196 tokens", 0.3, 9196)
    monkeypatch.setattr(agent, "_restore_when_idle", riw)
    llm._slot_session = None


async def _member_call(agent, llm):
    await agent._member_head_before(llm, {"messages": []}, time.monotonic() + 30)


async def test_a_members_call_after_its_look_gets_the_members_head_back(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    restores = []
    _member_env(agent, llm, st, monkeypatch, restores)
    await _member_call(agent, llm)
    await _member_call(agent, llm)                       # its own next call keeps what it has
    assert restores == ["mm"]
    await agent._side_call_before(LOOK)
    llm._last_main_call_end = time.monotonic() + 1       # the look ran
    st["lines"].clear()
    await _member_call(agent, llm)
    assert restores == ["mm", "mm"]
    assert any("after its own picture look" in m for m in st["lines"])
    await _member_call(agent, llm)                       # once per look
    assert restores == ["mm", "mm"]


async def test_a_members_look_that_never_ran_restores_nothing(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    restores = []
    _member_env(agent, llm, st, monkeypatch, restores)
    await _member_call(agent, llm)
    llm._last_main_call_end = time.monotonic() - 5
    await agent._side_call_before(LOOK)                  # nothing ended on the slot since
    await _member_call(agent, llm)
    assert restores == ["mm"]


async def test_another_member_requests_look_restores_nothing_here(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    restores = []
    _member_env(agent, llm, st, monkeypatch, restores)
    await _member_call(agent, llm)
    agent._member_side = ("m-other", time.monotonic() - 10)
    llm._last_main_call_end = time.monotonic()
    await _member_call(agent, llm)
    assert restores == ["mm"]


async def test_a_members_look_while_another_turn_holds_the_lock_marks_nothing(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _member_env(agent, llm, st, monkeypatch, [])
    agent._turn_lock_rid = "req-other"
    await agent._side_call_before(LOOK)
    assert getattr(agent, "_member_side", None) is None


async def test_a_members_look_saves_nothing(cwarm, monkeypatch, rid):
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch)
    _member_env(agent, llm, st, monkeypatch, [])
    await agent._side_call_before(LOOK)
    assert st["conv_saves"] == [] and getattr(agent, "_side_call_rid", None) is None
