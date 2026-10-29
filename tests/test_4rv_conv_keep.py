"""§4RV (2026-10-27): a long conversation's follow-up resumes where it ended.

llama b11669 checkpoints at the start of a prompt's last user message — the
trailing state block — so the next request can resume right after the
previous request's message, if (1) nothing before it changed and (2) the
conversation is still in the slot. Neither held: the first message carried
each request's memory, persona and thinking depth, and its label flipped;
and every deferred verdict, member turn or other conversation replaced the
slot (llama's RAM cache is 2 GB, a conversation with its checkpoints is
larger). With GHOST_CONV_KEEP=1:
  * the first message is the same on every request (`_split_request_context`,
    one label), the per-request parts open the state block;
  * the conversation that owns the slot is saved to disk before any call
    replaces it, and restored before its next request;
  * no keep window: a conversation keeps the slot until another call needs it;
    gap work (a missing head file) saves it first.
"""
from __future__ import annotations

import json
import pathlib
import time

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import head_saves as H
from ghost_agent.core import llm as L
from ghost_agent.core.agent import GhostAgent
from ghost_agent.utils.logging import Icons
from tests.test_4ro_head_saves import (_clean, _dir, _store, _switch_env, _turn,  # noqa: F401
                                       _warm_with_file, warm)

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def conv(monkeypatch):
    for k in (L.CONV_KEEP_ENV, H.ENV, A.STATE_ANCHOR_ENV):
        monkeypatch.setenv(k, "1")


@pytest.fixture
def cwarm(warm, conv, monkeypatch):
    """§4RO's warm-up harness with the switch on and nobody owning the slot
    (its client is a mock: an unset owner would read as a kept conversation)."""
    _agent, llm, _st = warm
    monkeypatch.setattr(A, "CONV_SAVE_MIN_S", 0.0)       # §4RW: the harness's deadlines are short
    llm._slot_session = None
    llm._slot_request = None
    llm._conv_saved = None
    return warm


# ── the switch ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("v,want", [("1", True), ("0", False), ("", False), ("yes", False)])
def test_the_switch(monkeypatch, conv, v, want):
    monkeypatch.setenv(L.CONV_KEEP_ENV, v)
    assert L.conv_keep_enabled() is want


@pytest.mark.parametrize("missing", ["GHOST_HEAD_SAVES", "GHOST_STATE_ANCHOR"])
def test_one_gate_needs_the_files_and_the_anchor(monkeypatch, conv, missing):
    """Review: on without its parts it wrote files that could never match
    (no anchor) or kept the slot forever (no files)."""
    assert L.conv_keep_enabled() and A._stable_first_message()
    monkeypatch.setenv(missing, "0")
    assert not L.conv_keep_enabled() and not A._stable_first_message()


# ── conversation files ───────────────────────────────────────────────────────

def test_a_conversation_key_is_per_session_and_server():
    a = H.conv_key("id", "sess-a")
    assert a == H.conv_key("id", "sess-a")
    assert a != H.conv_key("id", "sess-b") and a != H.conv_key("id2", "sess-a")
    assert a.startswith("c") and len(a) == 16


def test_a_conversation_save_takes_any_count_from_the_minimum(monkeypatch, tmp_path):
    d = _dir(tmp_path)

    def http(url, body=None, timeout=5.0):
        (d / body["filename"]).write_bytes(b"state")
        return {"n_saved": 40100, "n_written": 3_000_000_000}
    monkeypatch.setattr(H, "_http", http)
    ok, detail = H.save("b", d, "cx", 29237, "id", H.CONV, 9.0, True)
    assert ok and "40100 tokens" in detail
    meta = json.loads((d / "head-cx.json").read_text())
    assert meta["kind"] == H.CONV and meta["n_tokens"] == 40100
    ok, detail = H.save("b", d, "cy", 50000, "id", H.CONV, 9.0, True)
    assert not ok and "expected 50000" in detail
    ok, _ = H.save("b", d, "cz", 29237, "id", H.CONV, 9.0)      # exact unless asked otherwise
    assert not ok


def test_conversations_are_pruned_apart_from_heads(tmp_path, monkeypatch):
    d = _dir(tmp_path)
    for i, (key, kind) in enumerate([("o1", H.OWNER), ("c1", H.CONV), ("c2", H.CONV),
                                     ("c3", H.CONV), ("c4", H.CONV), ("o2", H.OWNER)]):
        _store(d, key)
        (d / f"head-{key}.json").write_text(json.dumps({"n_tokens": 1, "kind": kind}))
        t = time.time() - 100 + i
        import os
        os.utime(d / f"head-{key}.bin", (t, t))
        os.utime(d / f"head-{key}.json", (t, t))
    monkeypatch.setattr(H, "KEEP_CONV", 2)               # its own count, not the heads' (3)
    H.prune(d)
    left = sorted(p.name[5:-4] for p in d.glob("head-*.bin"))
    assert left == ["c3", "c4", "o1", "o2"]


def test_llama_start_never_restores_a_conversation(tmp_path):
    d = _dir(tmp_path)
    _store(d, "cc")
    (d / "head-cc.json").write_text(json.dumps({"n_tokens": 5, "identity": "id", "kind": H.CONV,
                                                "saved_at": time.time()}))
    assert H.newest_head_for(d, "id") is None


# ── the switch: save before another call, restore before the next request ────

def _conv_env(agent, llm, st, monkeypatch, *, slot=40000, used_after_end=True):
    """The slot holds sess-a's conversation (its call was the last one)."""
    st.setdefault("conv_saves", [])

    def save(base, d, key, n, ident="", kind=H.OWNER, timeout=H.SAVE_TIMEOUT_S, at_least=False,
             extra=None):
        st["order"].append(f"save:{kind}")
        st["conv_saves" if kind == H.CONV else "saves"].append((key, n, at_least))
        st.setdefault("extras", []).append(extra)
        _store(d, key, n)
        (d / f"head-{key}.json").write_text(json.dumps({"n_tokens": n, "kind": kind, **(extra or {})}))
        return (True, f"{n} tokens, 2.50 GB")
    monkeypatch.setattr(H, "save", save)
    st["slot"] = slot
    now = time.monotonic()
    llm._last_main_call_end = now - 5
    llm._slot_session_used = now - (1 if used_after_end else 9)
    llm._head_displaced = False


async def test_another_conversation_saves_the_kept_one_first(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    st["lines"].clear()
    st["logged"].clear()
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == [f"save:{H.CONV}", "restore"]
    key, n, at_least = st["conv_saves"][0]
    assert key == H.conv_key("id", "sess-a") and n >= 28213 + A.conv_min_past_tokens() and at_least
    assert llm._slot_session is None
    assert any("saved to disk before" in m for m in st["lines"])
    # §4RW: one icon per event — the conversation's save, then the head's restore
    assert [(t, i) for t, m, i in st["logged"] if i in (Icons.CONV_SAVE, Icons.HEAD_RESTORE)] == [
        ("Conversation", Icons.CONV_SAVE), ("Head Disk", Icons.HEAD_RESTORE)]


async def test_the_same_conversations_next_request_keeps_the_slot(cwarm, monkeypatch):
    """§4RP-b restored the head here (the label flip); the first message no
    longer changes, so the follow-up resumes from llama's checkpoint."""
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, key="sess-a")
    _conv_env(agent, llm, st, monkeypatch)
    llm._slot_request = "req-1"
    tok = request_id_context.set("req-2")
    try:
        await agent._slot_switch_before(_turn(st))
    finally:
        request_id_context.reset(tok)
    assert st["order"] == [] and llm._slot_session == "sess-a"


async def test_a_returning_conversation_is_restored_not_the_head(cwarm, monkeypatch):
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-b")
    _store(d, ckey, 41000)
    st["order"].clear()
    st["restores"].clear()
    st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-b", slot=30000)
    tok = request_id_context.set("req-9")
    try:
        await agent._slot_switch_before(_turn(st))
    finally:
        request_id_context.reset(tok)
    assert st["restores"] == [ckey]
    assert llm._slot_session == "sess-b" and llm._slot_request == "req-9"
    assert llm._conv_saved == ("sess-b", llm._slot_session_used)       # on disk as it is
    assert any("restored from disk in" in m for m in st["lines"])
    assert [i for t, m, i in st["logged"] if "restored from disk in" in m] == [Icons.CONV_RESTORE]   # §4RW


async def test_a_failed_conversation_restore_falls_back_to_the_head(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-b")
    _store(d, ckey, 41000)
    calls = []

    def restore(base, d, key):
        calls.append(key)
        return (False, "Unable to restore", 0.1) if key == ckey else (True, "28213 tokens", 0.4)
    monkeypatch.setattr(H, "restore", restore)
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-b", slot=30000)
    await agent._slot_switch_before(_turn(st))
    assert calls[0] == ckey and calls[1] != ckey


@pytest.mark.parametrize("case", ["other_call_since", "displaced", "too_short", "no_owner", "off"])
async def test_no_save_when_the_slot_is_not_that_conversation(cwarm, monkeypatch, case):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch, used_after_end=(case != "other_call_since"),
              slot=28213 + A.conv_min_past_tokens() - 1 if case == "too_short" else 40000)
    if case == "displaced":
        llm._head_displaced = llm._slot_moved_on = True
    if case == "no_owner":
        llm._slot_session = None
    if case == "off":
        monkeypatch.delenv(L.CONV_KEEP_ENV)
    assert await agent._conv_save(llm, "x") is False
    assert st["conv_saves"] == []


async def test_a_state_is_saved_once(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    assert await agent._conv_save(llm, "x") is True
    assert await agent._conv_save(llm, "y") is True
    assert len(st["conv_saves"]) == 1
    llm._slot_session_used = time.monotonic()          # the conversation ran again
    assert await agent._conv_save(llm, "z") is True
    assert len(st["conv_saves"]) == 2


async def test_a_failed_plan_still_saves_the_conversation(cwarm, monkeypatch):
    """§4RW-e (review): a tick whose plan failed left _head_plan_last None,
    and a displacement in that window dropped the conversation unsaved."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    agent._conv_plan_last, agent._head_plan_last = agent._head_plan_last, None
    monkeypatch.setattr(H, "server_identity", lambda base: agent._conv_plan_last["identity"])
    assert await agent._conv_save(llm, "x") is True and len(st["conv_saves"]) == 1


async def test_a_save_finished_after_a_request_started_keeps_the_record(cwarm, monkeypatch):
    """§4RW-e round 2 MAJOR: the owner's message arrived during the save; a
    release would make that request's own call look stale (released after it
    opened), so its end could not record the slot."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    owner = llm._slot_session
    assert await agent._conv_displace(llm, "x", stop=lambda: True) is True
    assert llm._slot_session == owner and len(st["conv_saves"]) == 1
    assert await agent._conv_displace(llm, "y", stop=lambda: False) is True
    assert llm._slot_session is None


async def test_an_unreadable_identity_still_saves_with_the_kept_plan(cwarm, monkeypatch):
    """r3: /props down (what failed the plan) is not another model — the
    check must not drop the conversation it exists to save."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    agent._conv_plan_last, agent._head_plan_last = agent._head_plan_last, None

    def down(base):
        raise TimeoutError("timed out")
    monkeypatch.setattr(H, "server_identity", down)
    assert await agent._conv_save(llm, "x") is True and len(st["conv_saves"]) == 1


async def test_the_kept_plan_is_not_used_under_another_model(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    agent._conv_plan_last, agent._head_plan_last = agent._head_plan_last, None
    monkeypatch.setattr(H, "server_identity", lambda base: "another-model|b1|t")
    assert await agent._conv_save(llm, "x") is False and st["conv_saves"] == []


async def test_a_busy_slot_is_not_saved(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000, busy=True)
    _conv_env(agent, llm, st, monkeypatch)
    assert await agent._conv_save(llm, "x", time.monotonic() + 0.2) is False
    assert st["conv_saves"] == []


async def test_a_member_saves_the_owners_conversation_and_does_not_park(cwarm, monkeypatch):
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    hs = {"base": "http://127.0.0.1:8088", "dir": pathlib.Path(H.head_dir()), "key": "mm",
          "identity": "id", "have": True, "n_tokens": 8000, "shape": None, "payload": {}}

    async def plan(_llm):
        return hs
    monkeypatch.setattr(agent, "_member_head_plan", plan)
    monkeypatch.setattr(L, "opens_with_head", lambda p, ref: True)
    seen = {}

    async def riw(_llm, _hs, late, *, skip_n, park=False):
        seen["park"] = park
        return None
    monkeypatch.setattr(agent, "_restore_when_idle", riw)
    tok = request_id_context.set("m-1")
    try:
        await agent._member_head_before(llm, {"messages": []}, time.monotonic() + 30)
    finally:
        request_id_context.reset(tok)
    assert st["order"] == [f"save:{H.CONV}"] and seen["park"] is False
    assert llm._slot_session is None


# ── background calls ─────────────────────────────────────────────────────────

async def test_a_background_call_saves_and_frees_the_slot(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    assert llm.before_background_dispatch == agent._conv_before_background
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    await agent._conv_before_background({}, "deferred-verdict")
    assert len(st["conv_saves"]) == 1 and llm._slot_session is None
    assert any("before a background 'deferred-verdict' call" in m for m in st["lines"])


async def test_the_client_runs_the_hook_except_for_head_warmups():
    c = L.LLMClient.__new__(L.LLMClient)
    seen = []

    async def hook(payload, label):
        seen.append(label)
    c.before_background_dispatch = hook
    await c._before_background_main({}, "deferred-verdict")
    await c._before_background_main({}, "main-prefix-rewarm")
    await c._before_background_main({}, "stream")
    assert seen == ["deferred-verdict", "stream"]
    c.before_background_dispatch = None
    await c._before_background_main({}, "x")              # no hook: nothing


async def test_a_failing_hook_never_fails_the_call():
    c = L.LLMClient.__new__(L.LLMClient)

    async def hook(payload, label):
        raise RuntimeError("boom")
    c.before_background_dispatch = hook
    await c._before_background_main({}, "deferred-verdict")


async def test_a_head_shaped_background_call_runs_after_the_save_not_deferred(conv):
    """Rule 5 deferred such a call while a conversation kept the slot; with
    no window it would wait forever — the hook saves and frees the slot first."""
    from tests.test_4rp_slot_keep import _bare_client, _head
    c = _bare_client()
    L.note_head_reference(c, _head())
    c._slot_session, c._slot_session_used = "sess-a", time.monotonic() - 3600
    order = []

    async def hook(payload, label):
        order.append(f"hook:{label}")
        c._slot_session = None

    async def do(payload, *a, **k):
        order.append("call")
        return {"choices": [{"message": {"content": "x"}}], "_ghost_leg": {"served_by": "main"}}
    c.before_background_dispatch = hook
    c._do_chat_completion = do
    await c.chat_completion(_head("job"), is_background=True, task_label="selfhood narrative")
    assert order == ["hook:selfhood narrative", "call"]


async def test_without_the_hook_the_slot_is_not_held_forever(conv):
    from tests.test_4rp_slot_keep import _bare_client, _head
    c = _bare_client()
    L.note_head_reference(c, _head())
    c._slot_session, c._slot_session_used = "sess-a", time.monotonic() - 3600
    c.before_background_dispatch = None                  # no head-save plan
    sent = []

    async def do(payload, *a, **k):
        sent.append(payload)
        return {"choices": [{"message": {"content": "x"}}], "_ghost_leg": {"served_by": "main"}}
    c._do_chat_completion = do
    await c.chat_completion(_head("job"), is_background=True, task_label="selfhood narrative")
    assert len(sent) == 1                                 # the 600 s window applies, long past


async def test_a_call_let_in_during_an_image_wait_passes(conv):
    """Review r2 MAJOR: `_wait_for_foreground_clear` lets one background call
    in while a request waits on the image node; the deferral is only for a
    request that STARTED during the hook's save."""
    from tests.test_4rp_slot_keep import _bare_client, _head
    c = _bare_client()
    L.note_head_reference(c, _head())
    sent = []

    async def hook(payload, label):
        return False                                      # did nothing (a request is live)

    async def do(payload, *a, **k):
        sent.append(payload)
        return {"choices": [{"message": {"content": "x"}}], "_ghost_leg": {"served_by": "main"}}
    c.before_background_dispatch = hook
    c._do_chat_completion = do

    async def clear(label):
        c.foreground_requests = 1                         # admitted with one request live
    c._wait_for_foreground_clear = clear
    await c.chat_completion(_head("v", system="Judge."), is_background=True, task_label="deferred-verdict")
    assert len(sent) == 1


async def test_the_hook_leaves_a_live_requests_slot_alone(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    llm.foreground_requests = 1
    assert await agent._conv_before_background({}, "deferred-verdict") is False
    assert llm._slot_session == "sess-a" and st["conv_saves"] == []


async def test_a_save_stopped_by_a_new_request_keeps_the_owner(cwarm, monkeypatch):
    """Review r2 MAJOR: a background save that gave up released the slot —
    the conversation's next request then restored an older file (or the
    head) instead of resuming from the slot."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000, busy=True)
    _conv_env(agent, llm, st, monkeypatch)
    llm.foreground_requests = 0

    async def idle(base, late, stop=None):
        llm.foreground_requests = 1                       # the next request arrived
        return not stop()
    monkeypatch.setattr(agent, "_slot_idle", idle)
    assert await agent._conv_before_background({}, "deferred-verdict") is True
    assert llm._slot_session == "sess-a" and st["conv_saves"] == []


async def test_the_conversations_own_call_ending_during_the_wait_is_still_saved(cwarm, monkeypatch):
    """§4RW review MAJOR (was: "a save that found the slot moved on does not
    release it"): the owner's reply ended while the save waited — its own call
    refreshed the record. Still this conversation: saved (its newer state)
    and released. Before, nothing was saved, the slot was not released, and
    a member's turn then overwrote a slot still recorded as the owner's."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)

    async def idle(base, late, stop=None):
        llm._last_main_call_end = time.monotonic()         # its reply ended meanwhile
        llm._slot_session_used = llm._last_main_call_end + 1e-6
        return True
    monkeypatch.setattr(agent, "_slot_idle", idle)
    await agent._conv_before_background({}, "deferred-verdict")
    assert [k for k, n, a in st["conv_saves"]] == [H.conv_key("id", "sess-a")]
    assert llm._slot_session is None


async def test_another_conversation_on_the_slot_meanwhile_is_not_saved_under_this_name(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)

    async def idle(base, late, stop=None):
        llm._slot_session = "sess-z"                         # another conversation took it
        llm._slot_session_used = llm._last_main_call_end = time.monotonic()
        return True
    monkeypatch.setattr(agent, "_slot_idle", idle)
    await agent._conv_before_background({}, "deferred-verdict")
    assert st["conv_saves"] == [] and llm._slot_session == "sess-z"


async def test_a_save_with_too_little_time_is_not_posted(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    monkeypatch.setattr(A, "CONV_SAVE_MIN_S", 1e6)
    assert await agent._conv_save(llm, "a test") is False and st["conv_saves"] == []


async def test_a_request_that_started_during_the_save_goes_first(conv):
    from tests.test_4rp_slot_keep import _bare_client, _head
    c = _bare_client()
    L.note_head_reference(c, _head())
    sent = []

    async def hook(payload, label):
        c.foreground_requests = 1                         # arrived while the save ran
        return True

    async def do(payload, *a, **k):
        sent.append(payload)
        return {"choices": [{"message": {"content": "x"}}], "_ghost_leg": {"served_by": "main"}}
    c.before_background_dispatch = hook
    c._do_chat_completion = do
    c._slot_session = "sess-a"
    with pytest.raises(L.BackgroundDeferred):
        await c.chat_completion(_head("v", system="Judge."), is_background=True, task_label="deferred-verdict")
    assert sent == [] and c._bg_queue_sem._value == 3


async def test_a_background_stream_saves_first(conv):
    from tests.test_4rp_slot_keep import _bare_client, _head
    c = _bare_client()
    c._main_streams_open = 0
    L.note_head_reference(c, _head())
    order = []

    async def hook(payload, label):
        order.append(f"hook:{label}")

    async def stream(payload, use_coding):
        order.append("stream")
        yield b"x"
    c.before_background_dispatch = hook
    c._do_stream_chat_completion = stream
    async for _ in c.stream_chat_completion(_head("q"), is_background=True):
        pass
    assert order == ["hook:stream", "stream"]


async def test_off_a_background_call_leaves_the_owner(cwarm, monkeypatch):
    """Review MAJOR: the hook is installed whatever the switch; off, it must
    change nothing (rule 5 and the window decide, as before)."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    monkeypatch.delenv(L.CONV_KEEP_ENV)
    llm._slot_session = "sess-a"
    await agent._conv_before_background({}, "deferred-verdict")
    assert llm._slot_session == "sess-a"
    assert await agent._conv_displace(llm, "x") is False and llm._slot_session == "sess-a"


async def test_a_save_that_lost_the_slot_while_it_waited_is_not_made(cwarm, monkeypatch):
    """Review MAJOR: a save waiting for the slot to go idle while another
    request's switch moved it on wrote B's state under A's name."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)

    async def idle(base, late, stop=None):
        llm._slot_session = "sess-b"                      # another request took it meanwhile
        llm._slot_session_used = time.monotonic()
        return True
    monkeypatch.setattr(agent, "_slot_idle", idle)
    assert await agent._conv_save(llm, "x") is False
    assert st["conv_saves"] == []


async def test_saves_run_one_at_a_time(cwarm, monkeypatch):
    import asyncio
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    live = {"n": 0, "max": 0}

    async def idle(base, late, stop=None):
        live["n"] += 1
        live["max"] = max(live["max"], live["n"])
        await asyncio.sleep(0.05)
        live["n"] -= 1
        return True
    monkeypatch.setattr(agent, "_slot_idle", idle)
    r = await asyncio.gather(agent._conv_save(llm, "a"), agent._conv_save(llm, "b"))
    assert r == [True, True] and live["max"] == 1 and len(st["conv_saves"]) == 1


async def test_a_conversation_still_in_the_slot_is_claimed_not_restored(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-b")
    _store(d, ckey, 41000)
    st["restores"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-b", slot=41000)
    st["slot"] = 41000
    llm._last_main_call_end = time.monotonic() - 50
    llm._conv_last_saved = ("sess-c", 41000, time.monotonic())   # another conversation, same size
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == [ckey]                       # review r2: not claimed on a count alone
    st["restores"].clear()
    agent._switch_done = {}
    llm._slot_session = None
    llm._conv_last_saved = ("sess-b", 41000, time.monotonic() - 60)
    llm._last_main_call_end = time.monotonic() - 30        # a call ran on the slot since that save
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == [ckey]
    st["restores"].clear()
    agent._switch_done = {}
    llm._slot_session = None
    llm._conv_last_saved = ("sess-b", 41000, time.monotonic())
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == [] and llm._slot_session == "sess-b"


# ── no keep window ───────────────────────────────────────────────────────────

def test_a_conversation_keeps_the_slot_however_long_ago(monkeypatch, conv):
    from types import SimpleNamespace
    llm = SimpleNamespace(_slot_session="s", _slot_session_used=time.monotonic() - 10 * 3600,
                          before_background_dispatch=lambda p, label: None)
    assert L.slot_kept(llm) is True
    llm._slot_session = None
    assert L.slot_kept(llm) is False
    monkeypatch.delenv(L.CONV_KEEP_ENV)
    llm._slot_session = "s"
    assert L.slot_kept(llm) is False                      # the 600 s window when off


async def test_the_tick_leaves_a_kept_conversation_and_does_not_park_junk(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    monkeypatch.setattr(agent, "_head_restore_due", lambda hs: _ret("park"))
    monkeypatch.setattr(agent, "_head_slot_quiet", lambda llm, hs: _ret(True))
    llm._slot_session = "sess-a"
    llm._slot_session_used = time.monotonic() - 7200
    st["order"].clear()
    await agent.warm_up_main_prefix(quiet=True)
    assert st["order"] == []                              # kept: no park, no restore
    llm._slot_session = None
    await agent.warm_up_main_prefix(quiet=True)
    assert st["order"] == ["restore"]                     # nobody's: restored, not parked


async def _ret(v):
    return v


@pytest.mark.parametrize("idle,want_saved", [(10_000, True), (5, False)])
async def test_a_missing_head_file_is_made_in_a_gap_after_saving(cwarm, monkeypatch, idle, want_saved):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    import datetime
    agent.context.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=idle)
    for p in pathlib.Path(H.head_dir()).glob("head-*"):      # a profile change: no file for the head
        p.unlink()
    deferred = []
    monkeypatch.setattr(agent, "_head_defer", lambda hs, why, **kw: deferred.append(why))
    await agent.warm_up_main_prefix(quiet=True)
    assert bool(st["conv_saves"]) is want_saved
    assert bool(deferred) is not want_saved


# ── the stable first message ─────────────────────────────────────────────────

def test_the_request_parts_leave_the_header():
    tb = A.render_think_budget_guidance("extended")
    header = f"# Tools\nADAPT YOUR THINKING DEPTH TO THE TASK:\n{tb}\nCRITICAL RULES"
    first, secs = A._split_request_context(header, tb, "PERSONA\n\n", "MEMORY: dog = Rex", "WS notes")
    assert tb not in first and A._THINK_DEPTH_POINTER in first
    tight = A.render_think_budget_guidance("tight")
    assert first == A._split_request_context(header.replace(tb, tight), tight, "", "", "")[0]
    assert list(secs) == ["request_context", "rc_think_depth", "rc_persona", "rc_memory", "rc_continuity"]
    assert secs["request_context"].startswith(A._REQUEST_CONTEXT_HEAD)
    assert secs["rc_think_depth"] == "THINKING DEPTH FOR THIS REQUEST:\n" + tb.strip() + "\n\n"
    assert secs["rc_memory"] == "MEMORY: dog = Rex\n\n"
    assert A._split_request_context("H", "", "", "", "") == ("H", {})


def test_a_depth_change_mid_request_resends_only_the_depth():
    """Review MAJOR: the depth is re-picked every turn (a project opened
    mid-request changes it); one section re-sent the persona every turn."""
    tb1, tb2 = A.render_think_budget_guidance("extended"), A.render_think_budget_guidance("tight")
    _h, s1 = A._split_request_context(f"x{tb1}y", tb1, "P" * 5000, "M" * 3000, "")
    _h, s2 = A._split_request_context(f"x{tb2}y", tb2, "P" * 5000, "M" * 3000, "")
    delta = A._state_changes(s1, s2)
    assert tb2.strip() in delta and "P" * 100 not in delta and "M" * 100 not in delta


def test_one_label_on_every_request():
    for msgs, pending in ([{"role": "user", "content": "q1"}], "q1"),\
            ([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a"},
              {"role": "user", "content": "q2"}], "q2"):
        r = GhostAgent._compose_injection([dict(m) for m in msgs], "HDR", "STATE", True,
                                          pending_request=pending, first_user_is_request=len(msgs) == 1,
                                          stable_label=True)
        assert A._CONVERSATION_START_LABEL in r[0]["content"]
        assert A._INSTRUCTION_LABEL not in r[0]["content"]


def _request(msgs, pending, memory, stable):
    tb = A.render_think_budget_guidance("extended" if memory else "tight")
    header = f"# Tools\nADAPT YOUR THINKING DEPTH TO THE TASK:\n{tb}\nRULES"
    secs = {"time": "CURRENT TIME: now\n\n"}
    if stable:
        hdr, ctx = A._split_request_context(header, tb, "", memory, "")
        secs = {**ctx, **secs}
    else:
        hdr = f"{header}\n\n{memory}"
    return GhostAgent._compose_injection([dict(m) for m in msgs], hdr, "".join(secs.values()), True,
                                         pending_request=pending, first_user_is_request=len(msgs) == 2,
                                         sections=secs, anchor={}, stable_label=stable)


@pytest.mark.parametrize("stable", [True, False])
def test_the_follow_up_resumes_right_after_the_previous_message(stable):
    """The property in llama's terms: the follow-up's rendered prompt agrees
    with the previous request's up to the start of its last user message
    (the state block, where llama checkpoints) — and before §4RV it did not."""
    from tests.test_4rc_state_anchor import _template
    tpl = _template()
    sys = {"role": "system", "content": "SYS"}
    r1 = _request([sys, {"role": "user", "content": "find my dog's vet"}], "find my dog's vet",
                  "MEMORY: dog = Rex", stable)
    r2 = _request([sys, {"role": "user", "content": "find my dog's vet"},
                   {"role": "assistant", "content": "Dr Smith."},
                   {"role": "user", "content": "and their hours?"}], "and their hours?",
                  "MEMORY: Dr Smith, Athens", stable)
    p1 = tpl.render(messages=r1, tools=None, add_generation_prompt=True, reasoning_effort="medium")
    p2 = tpl.render(messages=r2, tools=None, add_generation_prompt=True, reasoning_effort="medium")
    last_user_start = len(tpl.render(messages=r1[:-1], tools=None, add_generation_prompt=False,
                                     reasoning_effort="medium"))
    agree = 0
    while agree < min(len(p1), len(p2)) and p1[agree] == p2[agree]:
        agree += 1
    assert (agree >= last_user_start) is stable


async def _two_requests(monkeypatch):
    from unittest.mock import MagicMock
    from tests.test_kv_pin_stable_prefix import _make_agent
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", "1")
    agent = _make_agent()

    class Bus:
        async def hydrate_context(self, q, **kw):
            return f"### RECALLED MEMORY\nabout: {q}\n"
    agent._get_memory_bus = lambda: Bus()
    firsts, lasts = [], []
    q1 = "Write a python script that parses my csv file and prints totals"
    for msgs in ([{"role": "user", "content": q1}],
                 [{"role": "user", "content": q1}, {"role": "assistant", "content": "Done: totals.py"},
                  {"role": "user", "content": "what is my dog's name?"}]):
        await agent.handle_chat({"messages": msgs, "model": "m", "stream": False}, MagicMock())
        p = agent.context.llm_client.chat_completion.await_args.args[0]["messages"]
        firsts.append(next(m for m in p if m["role"] == "user")["content"])
        lasts.append(p[-1]["content"])
    return firsts, lasts


async def test_the_real_turn_loop_sends_the_same_first_message(monkeypatch, conv):
    firsts, lasts = await _two_requests(monkeypatch)
    assert firsts[0] == firsts[1]
    assert A._CONVERSATION_START_LABEL in firsts[0] and "RECALLED MEMORY" not in firsts[0]
    assert "about: what is my dog's name?" in lasts[1] and A._REQUEST_CONTEXT_HEAD in lasts[1]
    assert "THINKING DEPTH FOR THIS REQUEST" in lasts[1]


async def test_off_the_first_message_still_differs(monkeypatch):
    monkeypatch.setenv(A.STATE_ANCHOR_ENV, "1")
    monkeypatch.delenv(L.CONV_KEEP_ENV, raising=False)
    firsts, _ = await _two_requests(monkeypatch)
    assert firsts[0] != firsts[1]


async def test_the_idle_wait_stops_when_told(cwarm, monkeypatch):
    agent, llm, st = cwarm
    _switch_env(agent, llm, st, monkeypatch, busy=True)
    t0 = time.monotonic()
    assert await agent._slot_idle("b", time.monotonic() + 30, lambda: True) is False
    assert time.monotonic() - t0 < 5


async def test_a_saved_conversation_that_is_still_in_the_slot_is_claimed(cwarm, monkeypatch):
    """The save records what a later claim may trust; the conversation's
    next request then finds it in the slot and restores nothing."""
    from ghost_agent.utils.logging import slot_session_context
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    assert await agent._conv_displace(llm, "a gap job") is True and llm._slot_session is None
    st["restores"].clear()
    slot_session_context.set("sess-a")
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == [] and llm._slot_session == "sess-a"


# ── §4RW review fixes ─────────────────────────────────────────────────────

def test_a_restored_conversation_counts_as_used_when_pruning(tmp_path, monkeypatch):
    """Review: prune sorted by the .bin's save time; a restore only touches the
    .json — with 4+ conversations in rotation the one about to be used went."""
    import os
    d = _dir(tmp_path)
    for i, key in enumerate(["c1", "c2", "c3"]):
        _store(d, key)
        (d / f"head-{key}.json").write_text(json.dumps({"n_tokens": 1, "kind": H.CONV}))
        t = time.time() - 100 + i
        os.utime(d / f"head-{key}.bin", (t, t))
        os.utime(d / f"head-{key}.json", (t, t))
    H.touch(d, "c1")                                      # restored just now
    monkeypatch.setattr(H, "KEEP_CONV", 2)
    H.prune(d)
    assert sorted(p.name[5:-4] for p in d.glob("head-*.bin")) == ["c1", "c3"]


def test_six_conversations_are_kept():
    assert H.KEEP_CONV == 6


def test_the_restore_bound_grows_with_the_file(tmp_path):
    d = _dir(tmp_path)
    (d / "head-small.bin").write_bytes(b"x")
    with open(d / "head-big.bin", "wb") as f:
        f.truncate(6 * 10**9)                             # sparse: 6 GB, no disk used
    assert H.restore_timeout_s(d, "small") == H.RESTORE_TIMEOUT_S
    assert H.restore_timeout_s(d, "big") == pytest.approx(60.0)
    assert H.restore_timeout_s(d, "missing") == H.RESTORE_TIMEOUT_S


async def test_a_conversation_restore_that_timed_out_but_landed_is_kept(cwarm, monkeypatch):
    """Review MAJOR: the client gave up at 30 s, llama finished — the head was
    then restored over the conversation it had just loaded."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-b")
    _store(d, ckey, 41000)
    calls = []

    def restore(base, d, key):
        calls.append(key)
        if len(calls) == 1:                                # llama loaded it after the client gave up
            return (False, "TimeoutError: timed out", 30.0)
        return (True, "41000 tokens", 0.8)                 # §4RW-e: the second post, from page cache
    monkeypatch.setattr(H, "restore", restore)
    monkeypatch.setattr(H, "restore_timeout_s", lambda d, k: 0.1)   # the deadline fits a retry

    async def idle(base, late, stop=None):
        return True
    monkeypatch.setattr(agent, "_slot_idle", idle)
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-b", slot=30000)
    await agent._slot_switch_before(_turn(st))
    assert calls == [ckey, ckey] and llm._slot_session == "sess-b"


async def test_a_timed_out_restore_that_did_not_land_falls_back_to_the_head(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-b")
    _store(d, ckey, 41000)
    calls = []

    def restore(base, d, key):
        calls.append(key)
        if key != ckey:
            return (True, "28213 tokens", 0.4)
        if calls.count(ckey) == 1:
            return (False, "TimeoutError: timed out", 30.0)
        return (False, "Unable to restore slot", 0.1)       # the retry: refused outright
    monkeypatch.setattr(H, "restore", restore)
    monkeypatch.setattr(H, "restore_timeout_s", lambda d, k: 0.1)

    async def idle(base, late, stop=None):
        return True
    monkeypatch.setattr(agent, "_slot_idle", idle)
    st["slot"] = 30000
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-b", slot=30000)
    await agent._slot_switch_before(_turn(st))
    # a timeout, then a retry that llama refused outright: the head
    assert calls[:2] == [ckey, ckey] and calls[2] != ckey


@pytest.mark.parametrize("fits", [True, False])
async def test_a_second_timeout_is_a_landing_and_a_short_deadline_no_retry(cwarm, monkeypatch, fits):
    """§4RW-e round 2: the retry goes to an idle slot, so a second timeout is
    llama still reading the file — not a reason to put the head over it; and
    no retry is posted that the request's deadline cannot wait for."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-b")
    _store(d, ckey, 41000)
    calls = []

    def restore(base, d, key):
        calls.append(key)
        return (False, "TimeoutError: timed out", 30.0) if key == ckey else (True, "28213 tokens", 0.4)
    monkeypatch.setattr(H, "restore", restore)
    monkeypatch.setattr(H, "restore_timeout_s", lambda d, k: 0.1 if fits else 3600.0)

    waits = []

    async def idle(base, late, stop=None):
        waits.append(late)
        return True
    monkeypatch.setattr(agent, "_slot_idle", idle)
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-b", slot=30000)
    t_before = time.monotonic()
    await agent._slot_switch_before(_turn(st))
    if fits:
        # r3: the wait before the retry leaves the retry its bound inside the deadline
        assert waits and waits[-1] <= t_before + 1.0 - 0.1 + 0.05
        assert calls == [ckey, ckey] and llm._slot_session == "sess-b"
    else:
        assert calls[0] == ckey and calls[1] != ckey


async def test_a_replaced_slot_does_not_count_as_the_owners_conversation(cwarm, monkeypatch):
    """Review MAJOR: a member's turn replaced the slot while the owner record
    stayed (its save had given up) — the owner's next request skipped every
    restore and re-processed from token 0."""
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-a")
    _store(d, ckey, 41000)
    calls = []

    def restore(base, d, key):
        calls.append(key)
        return (True, "41000 tokens", 0.8)
    monkeypatch.setattr(H, "restore", restore)
    _switch_env(agent, llm, st, monkeypatch, owner="sess-a", key="sess-a", slot=9500)
    llm._head_displaced = llm._slot_moved_on = True        # a member's call ran since
    tok = request_id_context.set("req-next")
    try:
        await agent._slot_switch_before(_turn(st))
    finally:
        request_id_context.reset(tok)
    assert calls == [ckey]
    calls.clear()
    llm._head_displaced = False                            # the slot holds it: nothing
    llm._slot_session = "sess-a"
    tok = request_id_context.set("req-after")
    try:
        await agent._slot_switch_before(_turn(st))
    finally:
        request_id_context.reset(tok)
    assert calls == []


async def test_a_members_call_without_the_members_head_saves_the_owners_conversation(cwarm, monkeypatch):
    """Review MAJOR: the member path saved the owner's conversation only after
    its own checks — a member call without the members' head overwrote it unsaved."""
    from ghost_agent.utils.logging import requester_role_context
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000, owner="sess-a", key="slack-thread")
    _conv_env(agent, llm, st, monkeypatch)
    tok = requester_role_context.set("member")
    try:
        await agent._slot_switch_before({"messages": [{"role": "system", "content": "other"},
                                                      {"role": "user", "content": "hi"}]})
    finally:
        requester_role_context.reset(tok)
    assert [k for k, n, a in st["conv_saves"]] == [H.conv_key("id", "sess-a")]
    assert llm._slot_session is None


def test_a_llama_restart_forgets_the_slot_owner():
    """§4RW-e (review): llama restarted under a live agent — the record still
    named the owner, whose next request then skipped its file's restore."""
    from types import SimpleNamespace
    from ghost_agent.core.agent import GhostAgent
    llm = SimpleNamespace(_slot_session="sess-x", _slot_request="r1", _head_displaced=False)
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace(llm_client=llm)
    agent._note_llama_restart(50_000)
    agent._note_llama_restart(80_000)
    agent._note_llama_restart(None)                     # unreadable: no verdict
    assert llm._slot_session == "sess-x" and llm._head_displaced is False
    agent._note_llama_restart(120)                      # the counter fell
    assert llm._slot_session is None and llm._head_displaced is True
    assert llm._head_displaced_by == "llama restarted"


# ── §4RW-f: an effort change inside a conversation ───────────────────────────
# Levels are read from what is rendered and sent: a plan's from its render, a
# call's from its payload (thinking off = the medium head; Default = the
# server's own level; unknown = nothing moves).

@pytest.fixture
def default_medium(monkeypatch):
    from ghost_agent.core import effort as E
    monkeypatch.setitem(E._server_default, "level", "medium")


def _at(st, level=None, thinking=True):
    p = _turn(st)
    kw = {}
    if level:
        kw["reasoning_effort"] = level
    if not thinking:
        kw["enable_thinking"] = False
    if kw:
        p["chat_template_kwargs"] = kw
    return p


def _level_env(agent, monkeypatch, *, slot_level, plan_level, new_level=None, have=True):
    """The slot was built at ``slot_level``; the warm-up's plan renders ``plan_level``."""
    agent._slot_level = slot_level
    agent._head_plan_last["effort"] = plan_level
    planned = []
    if new_level:
        d = pathlib.Path(H.head_dir())
        new = {**agent._head_plan_last, "key": "lvl" + new_level, "effort": new_level,
               "have": have, "n_tokens": 28251 if have else None}
        if have:
            _store(d, new["key"], 28251)

        async def plan(llm, level, rid):
            planned.append(level)
            return dict(new)
        monkeypatch.setattr(agent, "_head_plan_for_level", plan)
    return planned


async def test_the_same_conversation_at_a_new_level_gets_that_levels_head(cwarm, monkeypatch, default_medium):
    """Operator item 5: the effort sentence closes the system block — the
    conversation in the slot parts from the new prompt ~28.2k tokens in, past
    its checkpoints. The new level's head comes back; only the conversation
    is re-read (before: llama re-read all of it from token 0)."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear(); st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch, key="sess-a")      # the owner's own conversation
    _conv_env(agent, llm, st, monkeypatch)
    planned = _level_env(agent, monkeypatch, slot_level="medium", plan_level="medium", new_level="xhigh")
    await agent._slot_switch_before(_at(st, "xhigh"))
    assert planned == ["xhigh"] and agent._head_plan_last["key"] == "lvlxhigh"
    assert st["order"] == ["restore"] and agent._slot_level == "xhigh"
    assert any("this conversation at xhigh effort" in m and "re-reads its part past the head" in m
               for m in st["lines"])


@pytest.mark.parametrize("call", [dict(level="medium"), dict(level=None), dict(level="xhigh", thinking=False)])
async def test_what_renders_the_same_head_keeps_the_conversation(cwarm, monkeypatch, default_medium, call):
    """Default is the server's medium; a thinking-off turn writes no sentence —
    the medium head, whatever the pick."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, key="sess-a")
    _conv_env(agent, llm, st, monkeypatch)
    planned = _level_env(agent, monkeypatch, slot_level="medium", plan_level="medium", new_level="x")
    await agent._slot_switch_before(_at(st, **call))
    assert planned == [] and st["order"] == [] and llm._slot_session == "sess-a"


async def test_an_unknown_server_level_moves_nothing(cwarm, monkeypatch):
    from ghost_agent.core import effort as E
    monkeypatch.setitem(E._server_default, "level", "")
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, key="sess-a")
    _conv_env(agent, llm, st, monkeypatch)
    planned = _level_env(agent, monkeypatch, slot_level="low", plan_level="medium", new_level="x")
    await agent._slot_switch_before(_at(st))
    assert planned == [] and st["order"] == [] and agent._slot_level == "low"


async def test_a_thinking_off_turn_after_xhigh_calls_restores_in_the_same_request(cwarm, monkeypatch,
                                                                                  default_medium):
    """Review MAJOR: the report turn (thinking off) renders the medium head;
    the request's xhigh conversation parts from it at the sentence. One
    decision per request and LEVEL — the first call's must not block it."""
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-a", slot=30000)
    _conv_env(agent, llm, st, monkeypatch)
    agent._head_plan_last["effort"] = "xhigh"
    tok = request_id_context.set("req-x")
    try:
        await agent._slot_switch_before(_at(st, "xhigh"))   # the request's first call
        assert agent._slot_level == "xhigh"
        llm._slot_session = "sess-a"                       # it ran and owns the slot
        st["order"].clear()
        _level_env(agent, monkeypatch, slot_level="xhigh", plan_level="xhigh", new_level="medium")
        await agent._slot_switch_before(_at(st, "xhigh", thinking=False))
    finally:
        request_id_context.reset(tok)
    assert st["order"] == ["restore"] and agent._slot_level == "medium"


async def test_a_new_level_without_its_file_restores_the_nearest_head(cwarm, monkeypatch, default_medium):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear(); st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch, key="sess-a")
    _conv_env(agent, llm, st, monkeypatch)
    _level_env(agent, monkeypatch, slot_level="medium", plan_level="medium", new_level="low", have=False)
    d = pathlib.Path(H.head_dir())                      # the medium head, as a real save writes it
    _store(d, "medhead", 28213)
    (d / "head-medhead.json").write_text(json.dumps({
        "n_tokens": 28213, "kind": H.OWNER, "identity": agent._head_plan_last["identity"],
        "saved_at": time.time()}))
    await agent._slot_switch_before(_at(st, "low"))
    assert st["order"] == ["restore"]
    assert any("the nearest saved head before this conversation at low effort" in m for m in st["lines"])


@pytest.mark.parametrize("saved,call,left", [("medium", "xhigh", True), ("xhigh", "xhigh", False),
                                             (None, "xhigh", False)])
async def test_a_conversation_saved_at_another_level_is_left_on_disk(cwarm, monkeypatch, default_medium,
                                                                     saved, call, left):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear(); st["lines"].clear()
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-b")
    _store(d, ckey, 41000)
    meta = {"n_tokens": 41000, "kind": H.CONV, **({"effort": saved} if saved else {})}
    (d / f"head-{ckey}.json").write_text(json.dumps(meta))
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-b", slot=30000)
    _level_env(agent, monkeypatch, slot_level=None, plan_level=call)
    calls = []
    monkeypatch.setattr(H, "restore", lambda b, dd, k: (calls.append(k), (True, "41000 tokens", 0.5))[1])
    await agent._slot_switch_before(_at(st, call))
    assert (ckey not in calls) is left
    if left:
        assert any("not restored — saved at medium effort, this request runs at xhigh" in m
                   for m in st["lines"])
        assert (st.get("logged") is None) or any(i == Icons.CONV_LEFT for _t, _m, i in st["logged"])


@pytest.mark.parametrize("slot_level,extra", [("low", {"effort": "low"}), (None, None)])
async def test_a_conversation_is_saved_with_the_level_it_was_built_at(cwarm, monkeypatch, slot_level, extra):
    """The displacing request may already run at a new level — the file
    carries the level of what the slot holds; unknown is left out."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    agent._slot_level = slot_level
    assert await agent._conv_save(llm, "x") is True
    assert st["extras"][-1].get("effort") == (extra or {}).get("effort")
    assert st["extras"][-1]["head_in_slot"] is True              # §4SA: the state opens with the head


@pytest.mark.parametrize("kw,default,want", [
    ({}, "medium", "medium"), ({"reasoning_effort": "xhigh"}, "medium", "xhigh"),
    ({"reasoning_effort": "xhigh", "enable_thinking": False}, "medium", "medium"),
    ({"reasoning_effort": "turbo"}, "medium", "medium"), ({"reasoning_effort": "xhigh"}, "", None)])
def test_a_calls_level_is_what_it_renders(monkeypatch, kw, default, want):
    from ghost_agent.core import effort as E
    monkeypatch.setitem(E._server_default, "level", default)
    assert GhostAgent._call_level({"messages": [], "chat_template_kwargs": kw}) == want


async def test_a_plans_level_is_read_from_its_render(cwarm, monkeypatch):
    agent, llm, st = cwarm
    monkeypatch.setattr(H, "render", lambda base, payload: "S… Reasoning effort is set to low. …")
    monkeypatch.setattr(H, "server_identity", lambda base: "id")
    monkeypatch.setattr(H, "supported", lambda base: True)
    hs = await agent._head_save_plan(llm, {"messages": []})
    assert hs["effort"] == "low"


async def test_the_re_plan_sets_the_level_on_the_warm_ups_own_payload_and_fails_once(cwarm, monkeypatch):
    agent, llm, st = cwarm
    agent._head_payload_base = {"messages": [{"role": "system", "content": "S"}],
                                "chat_template_kwargs": {"reasoning_effort": "low", "x": 1}}
    seen = []

    async def plan(llm, payload):
        seen.append(payload["chat_template_kwargs"])
        return {"effort": "medium", "key": "k"}           # the server rendered something else
    monkeypatch.setattr(agent, "_head_save_plan", plan)
    assert await agent._head_plan_for_level(llm, "xhigh", "r1") is None
    assert await agent._head_plan_for_level(llm, "xhigh", "r1") is None
    assert seen == [{"x": 1, "reasoning_effort": "xhigh"}]   # one try per request and level

    async def good(llm, payload):
        return {"effort": payload["chat_template_kwargs"]["reasoning_effort"], "key": "k"}
    monkeypatch.setattr(agent, "_head_save_plan", good)
    assert (await agent._head_plan_for_level(llm, "xhigh", "r2"))["effort"] == "xhigh"


async def test_a_level_that_comes_back_in_the_request_gets_its_head_again(cwarm, monkeypatch, default_medium):
    """Round 2 MAJOR: xhigh → a thinking-off (medium) turn → xhigh again in one
    request; the third call parts from the medium slot at the sentence."""
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-a", slot=30000)
    _conv_env(agent, llm, st, monkeypatch)
    agent._head_plan_last["effort"] = "xhigh"
    tok = request_id_context.set("req-aba")
    try:
        await agent._slot_switch_before(_at(st, "xhigh"))
        llm._slot_session = "sess-a"
        _level_env(agent, monkeypatch, slot_level="xhigh", plan_level="xhigh", new_level="medium")
        await agent._slot_switch_before(_at(st, "xhigh", thinking=False))
        llm._slot_session = "sess-a"
        st["order"].clear()
        _level_env(agent, monkeypatch, slot_level="medium", plan_level="medium", new_level="xhigh")
        await agent._slot_switch_before(_at(st, "xhigh"))
        assert st["order"] == ["restore"] and agent._slot_level == "xhigh"
        st["order"].clear()                                # and the same level again: nothing
        llm._slot_session = "sess-a"
        await agent._slot_switch_before(_at(st, "xhigh"))
        assert st["order"] == []
    finally:
        request_id_context.reset(tok)


def test_a_probe_arms_level_is_the_calls_level(monkeypatch):
    from ghost_agent.core import effort as E
    from ghost_agent.utils import logging as glog
    monkeypatch.setitem(E._server_default, "level", "medium")
    monkeypatch.setattr(glog, "prompt_arm", lambda: "effort_low")
    assert GhostAgent._call_level({"messages": []}) == "low"
    assert GhostAgent._call_level({"messages": [], "chat_template_kwargs": {"enable_thinking": False}}) == "medium"
