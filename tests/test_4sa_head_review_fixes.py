"""§4SA (2026-10-29): the end-to-end review of the head/conversation
save-restore chain and its seven fixes.

1. the agent's shutdown saves the slot's conversation (the record died with
   the process; the boot warm-up restored the head over an unsaved state);
2. a NEW request of the same conversation at another effort level saves the
   slot's state at the OLD level first (the head went over it unsaved, and the
   level's return restored a file three requests old);
3. the owner on Slack — a call that shares the head's TOOLS but not its system
   text — owns the slot and is saved/restored like any conversation; a channel
   member never owns it, whatever its tools (`llm._slot_moved_on`: whose state
   the slot holds, apart from `_head_displaced`: is the head in it);
4. probes do not reset the owner idle clock; the tick's deferral line is one
   INFO per distinct reason and says "holds the slot" under conversation keep;
5. CONV_MIN_PAST_TOKENS 1024 → 3072, env-overridable;
6. the save bound grows with the state like the restore's;
7. a same-conversation follow-up restores the head first when the previous
   request made ≥ GHOST_CONV_HEAD_FIRST_CALLS main calls (0 = off).
"""
from __future__ import annotations

import ast
import json
import logging
import pathlib
import time

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import head_saves as H
from ghost_agent.core import llm as L
from ghost_agent.utils.logging import (Icons, request_id_context, requester_role_context,
                                       slot_session_context)
from tests.test_4ro_head_saves import (_clean, _dir, _resp, _store, _switch_env,  # noqa: F401
                                       _turn, _warm_with_file, _with_tools, warm)
from tests.test_4rp_slot_keep import _bare_client, _head
from tests.test_4rv_conv_keep import _conv_env, conv, cwarm  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _ctx():
    t1 = slot_session_context.set("")
    t2 = requester_role_context.set("")
    yield
    requester_role_context.reset(t2)
    slot_session_context.reset(t1)


def _tools_turn(llm, st, text="hello from slack"):
    """A call with the head's tools but another system text (the owner on
    Slack). The harness's head has no tools, so the reference is re-taken
    with them, as test_4ro's Slack test does."""
    head = _with_tools(st["payload"])
    L.note_head_reference(llm, head)
    p = dict(head)
    p["messages"] = [{"role": "system", "content": "Public Ghost, no profile."},
                     {"role": "user", "content": text}]
    return p


# ── 1. shutdown ──────────────────────────────────────────────────────────────

async def test_the_shutdown_saves_the_slots_conversation(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    assert await agent.save_slot_conversation_at_shutdown() is True
    key, n, at_least = st["conv_saves"][0]
    assert key == H.conv_key("id", "sess-a") and n == 40000 and at_least
    assert any("saved to disk before the agent's shutdown" in m for m in st["lines"])
    assert [i for _t, m, i in st["logged"] if "the agent's shutdown" in m] == [Icons.CONV_SAVE]


@pytest.mark.parametrize("case", ["turn_running", "request_live", "no_owner", "off", "streaming", "moved_on"])
async def test_the_shutdown_leaves_a_turn_or_an_unowned_slot_alone(cwarm, monkeypatch, case):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    if case == "turn_running":
        agent._turn_lock_rid = "req-live"
    elif case == "request_live":
        llm.foreground_requests = 1
    elif case == "no_owner":
        llm._slot_session = None
    elif case == "streaming":
        llm._main_streams_open = 1                         # r2: a background stream still on the slot
    elif case == "moved_on":
        llm._slot_moved_on = True                          # r2: the slot's last call was not the conversation's
    else:
        monkeypatch.delenv(L.CONV_KEEP_ENV)
    assert await agent.save_slot_conversation_at_shutdown() is False
    assert st["conv_saves"] == []


async def test_the_shutdown_save_is_bounded(cwarm, monkeypatch):
    """The deadline handed to the save sits inside SHUTDOWN_CONV_SAVE_S (launchd
    SIGKILLs 60 s after SIGTERM, and other drains follow)."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    seen = {}

    async def save(llm_, why, late=None, stop=None):
        seen["why"], seen["late"] = why, late
        return True
    monkeypatch.setattr(agent, "_conv_save", save)
    t0 = time.monotonic()
    assert await agent.save_slot_conversation_at_shutdown() is True
    assert seen["why"] == "the agent's shutdown"
    assert t0 + 1.0 < seen["late"] <= t0 + A.SHUTDOWN_CONV_SAVE_S + 0.5
    assert A.SHUTDOWN_CONV_SAVE_S + 15.0 <= 60.0            # room for the other drains


def test_main_calls_the_shutdown_save_before_the_llm_client_closes():
    """The wiring: inside lifespan's finally, after the aborted-turn stamping
    and before `llm_client.close()` (llama must still be reachable)."""
    tree = ast.parse((ROOT / "src/ghost_agent/main.py").read_text())
    lifespan = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.AsyncFunctionDef) and n.name == "lifespan")
    final = next(t.finalbody for t in ast.walk(lifespan) if isinstance(t, ast.Try) and t.finalbody
                 and "System Shutdown" in ast.dump(t.finalbody[0]))
    dumps = [ast.dump(stmt) for stmt in final]
    i_save = next(i for i, d in enumerate(dumps) if "save_slot_conversation_at_shutdown" in d)
    i_close = next(i for i, d in enumerate(dumps) if "llm_client" in d and "close" in d)
    i_bio = next(i for i, d in enumerate(dumps) if "_stop_biological_watchdog" in d)
    i_rw = next(i for i, d in enumerate(dumps) if "_rw" in d and "cancel" in d)
    assert i_bio < i_save < i_close                        # r2: after the idle loop stops
    assert i_rw < i_save                                   # r2: and after the re-warm tick is cancelled
    assert isinstance(final[i_save], ast.Try)              # never fails the shutdown
    assert A.SHUTDOWN_CONV_SAVE_S + 6 + 5 + 5 + 5 + 5 <= 60   # uvicorn + watchdog + the drains + the sandbox


# ── 2. an effort change in a NEW request saves the old level first ───────────

async def test_a_new_request_at_another_level_saves_the_old_level_first(cwarm, monkeypatch):
    from ghost_agent.core import effort as E
    monkeypatch.setitem(E._server_default, "level", "medium")
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear(); st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch, key="sess-a", slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    agent._slot_level = "medium"
    agent._head_plan_last["effort"] = "medium"
    d = pathlib.Path(H.head_dir())
    new = {**agent._head_plan_last, "key": "lvlxhigh", "effort": "xhigh", "have": True, "n_tokens": 28251}
    _store(d, "lvlxhigh", 28251)

    async def plan(llm_, level, rid):
        return dict(new)
    monkeypatch.setattr(agent, "_head_plan_for_level", plan)
    llm._slot_request = "req-1"
    tok = request_id_context.set("req-2")
    try:
        p = _turn(st)
        p["chat_template_kwargs"] = {"reasoning_effort": "xhigh"}
        await agent._slot_switch_before(p)
    finally:
        request_id_context.reset(tok)
    assert st["order"] == [f"save:{H.CONV}", "restore"]
    assert st["extras"][-1]["effort"] == "medium"             # the level the state was built at
    assert st["conv_saves"][0][0] == H.conv_key("id", "sess-a", "medium")     # r2: that level's own file
    assert any("this conversation's change to xhigh effort" in m for m in st["lines"])
    assert any("this conversation at xhigh effort" in m for m in st["lines"])


async def test_a_level_flip_inside_one_request_saves_nothing(cwarm, monkeypatch):
    """One file per conversation: two multi-GB saves per thinking-off flip
    would cost more than the re-read they could spare."""
    from ghost_agent.core import effort as E
    monkeypatch.setitem(E._server_default, "level", "medium")
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, key="sess-a", slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    agent._slot_level = "xhigh"
    agent._head_plan_last["effort"] = "xhigh"

    async def plan(llm_, level, rid):
        return {**agent._head_plan_last, "key": "lvlmedium", "effort": "medium", "have": True, "n_tokens": 28213}
    monkeypatch.setattr(agent, "_head_plan_for_level", plan)
    _store(pathlib.Path(H.head_dir()), "lvlmedium", 28213)
    llm._slot_request = "req-1"
    tok = request_id_context.set("req-1")
    try:
        p = _turn(st)
        p["chat_template_kwargs"] = {"enable_thinking": False}
        await agent._slot_switch_before(p)
    finally:
        request_id_context.reset(tok)
    assert st["order"] == ["restore"] and st["conv_saves"] == []


# ── 3. the owner on Slack owns the slot; a member never does ─────────────────

def test_a_call_sharing_the_heads_tools_takes_the_slot_unless_a_member_made_it():
    c = _bare_client()
    L.note_head_reference(c, _head())
    slot_session_context.set("slack:t1:owner")
    L.note_main_slot_prompt(c, _head("hi", system="Public Ghost."), "stream")
    assert c._slot_session == "slack:t1:owner"
    assert c._head_displaced is True and c._slot_moved_on is False        # no head, still its own
    # a member's call with the same tools: the role says no
    slot_session_context.set("slack:t2:member")
    tok = requester_role_context.set("member")
    try:
        L.note_main_slot_prompt(c, _head("yo", system="Member Ghost."), "stream")
    finally:
        requester_role_context.reset(tok)
    assert c._slot_session == "slack:t1:owner"
    assert c._slot_moved_on is True                 # review: the role gates the flag too — members never hold
    # a call with other tools, no tools, a background call: moved on
    slot_session_context.set("slack:t1:owner")
    L.note_main_slot_prompt(c, _head("e4", tools=None), "game")
    assert c._slot_moved_on is True and c._slot_session == "slack:t1:owner"
    L.note_main_slot_prompt(c, _head("ok"), "deferred-verdict", background=True)
    assert c._slot_moved_on is True
    L.note_main_slot_prompt(c, _head("ok"), "stream")
    assert c._slot_moved_on is False and c._head_displaced is False


def test_slot_moved_on_reads_the_flag():
    c = _bare_client()
    assert L.slot_moved_on(c) is False
    c._slot_moved_on = True
    assert L.slot_moved_on(c) is True


async def test_the_slack_owners_conversation_is_saved_without_the_head(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, owner="slack:t1:U0", key="sess-b", slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    llm._head_displaced = True                   # the slot holds the owner's Slack thread: no head
    llm._slot_moved_on = False
    assert await agent._conv_save(llm, "x") is True
    assert st["conv_saves"][0][0] == H.conv_key("id", "slack:t1:U0")
    assert st["extras"][-1]["head_in_slot"] is False


async def test_a_slack_owner_request_restores_its_conversation_file(cwarm, monkeypatch):
    """Before §4SA a "tools" call only ever got the bare head back."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "slack:t1:U0")
    _store(d, ckey, 36000)
    (d / f"head-{ckey}.json").write_text(json.dumps({"n_tokens": 36000, "kind": H.CONV, "head_in_slot": False}))
    st["order"].clear(); st["restores"].clear(); st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="slack:t1:U0", slot=30000)
    tok = request_id_context.set("req-s1")
    try:
        await agent._slot_switch_before(_tools_turn(llm, st))
    finally:
        request_id_context.reset(tok)
    assert st["restores"] == [ckey]
    assert llm._slot_session == "slack:t1:U0" and llm._slot_request == "req-s1"
    assert llm._head_displaced is True and llm._slot_moved_on is False     # from the file's meta
    assert any("restored from disk" in m for m in st["lines"])


async def test_a_conversation_file_with_the_head_clears_the_mark(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-b")
    _store(d, ckey, 41000)
    (d / f"head-{ckey}.json").write_text(json.dumps({"n_tokens": 41000, "kind": H.CONV, "head_in_slot": True}))
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-b", slot=30000)
    llm._head_displaced = llm._slot_moved_on = True
    await agent._slot_switch_before(_turn(st))
    assert st["restores"][-1] == ckey and llm._head_displaced is False and llm._slot_moved_on is False


async def test_the_slack_owners_next_message_resumes_from_the_slot(cwarm, monkeypatch):
    """Owner == key, the head is not in the slot but the state is this
    conversation's own — nothing is restored over it."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner="slack:t1:U0", key="slack:t1:U0", slot=36000)
    _conv_env(agent, llm, st, monkeypatch)
    llm._head_displaced = True
    llm._slot_moved_on = False
    await agent._slot_switch_before(_tools_turn(llm, st))
    assert st["order"] == [] and llm._slot_session == "slack:t1:U0"


async def test_a_slot_that_moved_on_is_not_this_conversation_any_more(cwarm, monkeypatch):
    """The §4RW review case kept: a member's turn after the owner's reply."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner="sess-a", key="sess-a", slot=9500)
    _conv_env(agent, llm, st, monkeypatch)
    llm._head_displaced = llm._slot_moved_on = True
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == ["restore"]


async def test_a_look_during_the_slack_owners_request_saves_its_thread(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, owner="slack:t1:U0", key="slack:t1:U0", slot=36000)
    _conv_env(agent, llm, st, monkeypatch)
    llm._head_displaced = True
    llm._slot_moved_on = False
    tok = request_id_context.set("req-s2")
    try:
        await agent._side_call_before({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})
    finally:
        request_id_context.reset(tok)
    assert [k for k, _n, _a in st["conv_saves"]] == [H.conv_key("id", "slack:t1:U0")]
    assert agent._side_call_rid == "req-s2"


def test_forgetting_the_owner_marks_the_slot_moved_on(cwarm):
    agent, llm, st = cwarm
    llm._slot_session = "sess-a"
    llm._slot_moved_on = False
    agent._forget_slot_owner("llama restarted")
    assert llm._slot_session is None and llm._slot_moved_on is True and llm._head_displaced is True


# ── 4. probes and the idle clock; the deferral line ──────────────────────────

@pytest.mark.parametrize("rid,idle_moves,sched_moves", [("abc12345", True, True), ("probe-4q-7", False, True),
                                                        ("sched-1", False, False), ("job-2", False, False),
                                                        ("sub-3", False, False), ("slack-7e", True, True)])
def test_only_a_real_request_moves_the_slots_gap_clock(cwarm, rid, idle_moves, sched_moves):
    """Review: the idle SCHEDULER's clock still counts probes (its phases must
    stay off during the operator's probe runs); the slot's gap clock does not."""
    import datetime
    agent, llm, st = cwarm
    old = datetime.datetime(2026, 1, 1)
    agent.context.last_activity_time = old
    agent.context.last_real_activity_time = old
    tok = request_id_context.set(rid)
    try:
        agent._heartbeat()
    finally:
        request_id_context.reset(tok)
    assert (agent.context.last_real_activity_time != old) is idle_moves
    assert (agent.context.last_activity_time != old) is sched_moves
    assert (agent._idle_s() < 3600.0) is idle_moves


def test_the_gap_clock_falls_back_to_the_schedulers(cwarm):
    """A context without the real clock yet (the first request not in), or a
    test double: the old clock, never 0."""
    import datetime
    agent, llm, st = cwarm
    agent.context.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=700)
    agent.context.last_real_activity_time = None
    assert 690 < agent._idle_s() < 800


def test_the_deferral_line_is_said_once_per_reason(cwarm, caplog):
    agent, llm, st = cwarm
    hs = {"detail": ""}
    with caplog.at_level(logging.DEBUG, logger="GhostAgent"):
        for why in ("a conversation holds the slot", "a conversation holds the slot", "the main slot is in use",
                    "the main slot is in use"):
            with pytest.raises(L.BackgroundDeferred):
                agent._head_defer(hs, why, loud=False, count=False)
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO and "deferred" in r.getMessage()]
    debugs = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG and "deferred" in r.getMessage()]
    assert infos == ["head restore deferred: a conversation holds the slot",
                     "head restore deferred: the main slot is in use"]
    assert len(debugs) == 2
    assert agent._head_rewarm_retry is True                # the next tick still retries


async def test_a_dispatched_plan_resets_the_deferral_memory(cwarm, caplog):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    hs = dict(agent._head_plan_last)
    with pytest.raises(L.BackgroundDeferred):
        agent._head_defer(hs, "a conversation holds the slot", loud=False, count=False)
    await agent._head_save_after(llm, hs, _resp(28213, 28213), 0.1)
    assert agent._head_defer_last_why is None
    with caplog.at_level(logging.INFO, logger="GhostAgent"):
        with pytest.raises(L.BackgroundDeferred):
            agent._head_defer(hs, "a conversation holds the slot", loud=False, count=False)
    assert any(r.levelno == logging.INFO and "head restore deferred" in r.getMessage() for r in caplog.records)


def test_the_kept_reason_names_the_disk_under_conversation_keep(monkeypatch, conv):
    assert "holds the slot" in A.GhostAgent._kept_why() and "disk" in A.GhostAgent._kept_why()
    monkeypatch.delenv(L.CONV_KEEP_ENV)
    assert "idle window" in A.GhostAgent._kept_why()


async def test_the_tick_says_a_conversation_holds_the_slot(cwarm, monkeypatch, caplog):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    llm._slot_session, llm._slot_session_used = "sess-a", time.monotonic()
    st["slot"] = 40000
    with caplog.at_level(logging.INFO, logger="GhostAgent"):
        await agent.warm_up_main_prefix(quiet=True)
    assert any("holds the slot" in r.getMessage() for r in caplog.records)
    assert not any("idle window" in r.getMessage() for r in caplog.records)


# ── 5. the save threshold ────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,want", [(None, 3072), ("4096", 4096), ("0", 0), ("-7", 0), ("junk", 3072)])
def test_the_save_threshold_comes_from_the_env(monkeypatch, raw, want):
    monkeypatch.delenv("GHOST_CONV_MIN_PAST_TOKENS", raising=False)
    if raw is not None:
        monkeypatch.setenv("GHOST_CONV_MIN_PAST_TOKENS", raw)
    assert A.conv_min_past_tokens() == want
    assert A.CONV_MIN_PAST_TOKENS == 3072


@pytest.mark.parametrize("past,saved", [(3071, False), (3072, True)])
async def test_a_save_needs_the_threshold_past_the_head(cwarm, monkeypatch, past, saved):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    monkeypatch.delenv("GHOST_CONV_MIN_PAST_TOKENS", raising=False)
    _switch_env(agent, llm, st, monkeypatch, slot=28213 + past)
    _conv_env(agent, llm, st, monkeypatch, slot=28213 + past)
    assert await agent._conv_save(llm, "x") is saved
    assert bool(st["conv_saves"]) is saved


async def test_the_env_threshold_is_the_one_used(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    monkeypatch.setenv("GHOST_CONV_MIN_PAST_TOKENS", "500")
    _switch_env(agent, llm, st, monkeypatch, slot=28213 + 600)
    _conv_env(agent, llm, st, monkeypatch, slot=28213 + 600)
    assert await agent._conv_save(llm, "x") is True


# ── 6. the save bound ────────────────────────────────────────────────────────

def test_the_save_bound_grows_with_the_state():
    assert H.save_timeout_s(0) == H.SAVE_TIMEOUT_S
    assert H.save_timeout_s(30000) == H.SAVE_TIMEOUT_S
    big = H.save_timeout_s(174000)
    assert big > H.SAVE_TIMEOUT_S
    assert abs(big - H.RESTORE_S_PER_GB * (174000 * H.STATE_BYTES_PER_TOKEN + H.CHECKPOINTS_BYTES) / 1e9) < 1e-6
    assert H.save_timeout_s(None) == H.SAVE_TIMEOUT_S and H.save_timeout_s("x") == H.SAVE_TIMEOUT_S


async def test_a_big_conversations_save_gets_the_bigger_bound(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=170000)
    _conv_env(agent, llm, st, monkeypatch, slot=170000)
    bounds = []

    def save(base, d, key, n, ident="", kind=H.OWNER, timeout=H.SAVE_TIMEOUT_S, at_least=False, extra=None):
        bounds.append(timeout)
        _store(d, key, n)
        return (True, f"{n} tokens, 7.9 GB")
    monkeypatch.setattr(H, "save", save)
    assert await agent._conv_save(llm, "x", time.monotonic() + 500.0) is True
    # §4SC-b: a caller that GIVES a window rules (the side hook sizes its own);
    # the size estimate is only the bound when no window is given
    assert bounds == [pytest.approx(500.0, abs=1.0)]
    assert bounds[0] > H.SAVE_TIMEOUT_S


# ── 7. a long request's follow-up gets the head first (gated) ────────────────

@pytest.mark.parametrize("raw,want", [(None, 0), ("", 0), ("8", 8), ("-1", 0), ("x", 0)])
def test_the_head_first_gate_comes_from_the_env(monkeypatch, raw, want):
    monkeypatch.delenv(L.CONV_HEAD_FIRST_CALLS_ENV, raising=False)
    if raw is not None:
        monkeypatch.setenv(L.CONV_HEAD_FIRST_CALLS_ENV, raw)
    assert L.conv_head_first_calls() == want


def test_the_record_counts_a_requests_main_calls():
    c = _bare_client()
    L.note_head_reference(c, _head())
    slot_session_context.set("sess-a")
    for rid, want in (("r1", 1), ("r1", 2), ("r1", 3), ("r2", 1), ("r2", 2)):
        tok = request_id_context.set(rid)
        try:
            L.note_main_slot_prompt(c, _head("q"), "stream")
        finally:
            request_id_context.reset(tok)
        assert (c._slot_request, c._slot_request_calls) == (rid, want)


async def _followup(agent, llm, st, monkeypatch, *, calls, gate, with_file=True):
    monkeypatch.delenv(L.CONV_HEAD_FIRST_CALLS_ENV, raising=False)
    if gate is not None:
        monkeypatch.setenv(L.CONV_HEAD_FIRST_CALLS_ENV, str(gate))
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-a")
    if with_file:
        _store(d, ckey, 35000)             # an OLDER save of this conversation
    st["order"].clear(); st["restores"].clear(); st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner="sess-a", key="sess-a", slot=42000)
    _conv_env(agent, llm, st, monkeypatch)
    llm._slot_request, llm._slot_request_calls = "req-1", calls
    tok = request_id_context.set("req-2")
    try:
        await agent._slot_switch_before(_turn(st))
    finally:
        request_id_context.reset(tok)
    return ckey


async def test_off_by_default_a_follow_up_resumes_from_the_slot(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _followup(agent, llm, st, monkeypatch, calls=40, gate=None)
    assert st["order"] == [] and llm._slot_session == "sess-a"


@pytest.mark.parametrize("calls,restored", [(4, False), (5, True), (12, True)])
async def test_past_the_gate_the_head_comes_back_first(cwarm, monkeypatch, calls, restored):
    agent, llm, st = cwarm
    ckey = await _followup(agent, llm, st, monkeypatch, calls=calls, gate=5)
    if not restored:
        assert st["order"] == [] and llm._slot_session == "sess-a"
        return
    assert st["order"] == ["restore"] and ckey not in st["restores"]      # the head, never the older file
    assert st["restores"] == [agent._head_plan_last["key"]]
    assert llm._slot_session is None                                       # the request's own call re-takes it
    assert any(f"this conversation's next request (its last made {calls} calls" in m
               and "re-reads its part past the head" in m for m in st["lines"])
    assert st["conv_saves"] == []                                          # nothing to save: the slot IS it


async def test_the_gate_counts_only_a_new_request(cwarm, monkeypatch):
    """The same request's later calls never trigger it."""
    agent, llm, st = cwarm
    monkeypatch.setenv(L.CONV_HEAD_FIRST_CALLS_ENV, "3")
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner="sess-a", key="sess-a", slot=42000)
    _conv_env(agent, llm, st, monkeypatch)
    llm._slot_request, llm._slot_request_calls = "req-1", 9
    tok = request_id_context.set("req-1")
    try:
        await agent._slot_switch_before(_turn(st))
    finally:
        request_id_context.reset(tok)
    assert st["order"] == []


# ── review round 1 (gate): the counter and the file ──────────────────────────

def test_the_call_count_never_outlives_its_request():
    """Review MAJOR: r1 made 9 calls; a displacement released the slot; r2 was
    restored from its file and made ONE call — the counter read 10 and the gate
    fired on a one-call request."""
    c = _bare_client()
    L.note_head_reference(c, _head())
    slot_session_context.set("sess-a")
    tok = request_id_context.set("r1")
    try:
        for _ in range(9):
            L.note_main_slot_prompt(c, _head("q"), "stream")
    finally:
        request_id_context.reset(tok)
    assert c._slot_request_calls == 9
    L.release_slot_session(c)
    assert c._slot_request_calls == 0


async def test_a_restored_conversation_starts_its_count_afresh(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-b")
    _store(d, ckey, 41000)
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-b", slot=30000)
    llm._slot_request_calls = 9
    tok = request_id_context.set("req-9")
    try:
        await agent._slot_switch_before(_turn(st))
    finally:
        request_id_context.reset(tok)
    assert st["restores"][-1] == ckey and llm._slot_request_calls == 0


async def test_a_save_records_the_requests_call_count(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    llm._slot_request_calls = 7
    assert await agent._conv_save(llm, "x") is True
    assert st["extras"][-1]["calls"] == 7


@pytest.mark.parametrize("calls,gate,restored", [(9, 0, True), (4, 5, True), (5, 5, False), (12, 5, False),
                                                 (None, 5, True), ("junk", 5, True)])
async def test_a_file_saved_after_a_long_request_is_left_for_the_head(cwarm, monkeypatch, calls, gate, restored):
    """Review MAJOR: the file holds the previous request's last 12 checkpoints
    exactly as the slot did — the same gate applies to a restore."""
    agent, llm, st = cwarm
    monkeypatch.delenv(L.CONV_HEAD_FIRST_CALLS_ENV, raising=False)
    if gate:
        monkeypatch.setenv(L.CONV_HEAD_FIRST_CALLS_ENV, str(gate))
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-b")
    _store(d, ckey, 41000)
    meta = {"n_tokens": 41000, "kind": H.CONV, "head_in_slot": True}
    if calls is not None:
        meta["calls"] = calls
    (d / f"head-{ckey}.json").write_text(json.dumps(meta))
    st["restores"].clear(); st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-b", slot=30000)
    await agent._slot_switch_before(_turn(st))
    if restored:
        assert st["restores"] == [ckey]
    else:
        assert st["restores"] == [agent._head_plan_last["key"]]
        assert any(f"not restored — its last request made {calls} calls" in m for m in st["lines"])
        assert [i for _t, m, i in st["logged"] if "its last request made" in m] == [Icons.CONV_LEFT]


async def test_a_file_without_the_head_mark_is_an_old_one_with_the_head(cwarm, monkeypatch):
    """Files saved before §4SA have no `head_in_slot`; they all opened with the head."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-b")
    _store(d, ckey, 41000)                       # meta: n_tokens only
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-b", slot=30000)
    llm._head_displaced = llm._slot_moved_on = True
    await agent._slot_switch_before(_turn(st))
    assert st["restores"][-1] == ckey and llm._head_displaced is False


# ── review round 1 (ownership) ───────────────────────────────────────────────

async def test_the_slack_owners_reply_after_its_look_gets_its_thread_back(cwarm, monkeypatch):
    """Review MAJOR: the lift past the once-per-request record and the marker
    clear were head-only; the owner on Slack is a "tools" call."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "slack:t1:U0")
    _store(d, ckey, 36000)
    (d / f"head-{ckey}.json").write_text(json.dumps({"n_tokens": 36000, "kind": H.CONV, "head_in_slot": False}))
    st["restores"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner="slack:t1:U0", key="slack:t1:U0", slot=36400)
    _conv_env(agent, llm, st, monkeypatch)
    agent._switch_done = {"req-s3": {"tools"}}            # the request's first call decided already
    agent._side_call_rid = "req-s3"                        # … then its look ran on the slot
    llm._head_displaced = llm._slot_moved_on = True
    llm._slot_request = "req-s3"
    tok = request_id_context.set("req-s3")
    try:
        await agent._slot_switch_before(_tools_turn(llm, st))
    finally:
        request_id_context.reset(tok)
    assert st["restores"] == [ckey] and agent._side_call_rid is None


async def test_a_keyless_marker_needs_a_slot_that_moved_on(cwarm, monkeypatch):
    """The keyless gate reads the moved-on mark, not the head mark."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="", slot=30000)
    agent._side_call_rid = "probe-1"
    llm._head_displaced, llm._slot_moved_on = True, False
    tok = request_id_context.set("probe-1")
    try:
        await agent._slot_switch_before(_turn(st))
    finally:
        request_id_context.reset(tok)
    assert st["order"] == []


def test_a_stream_that_opened_before_the_slot_changed_hands_does_not_take_it():
    """Review MAJOR: the old guard needed an owner, so after a release the
    draining reply re-took a slot another conversation now held."""
    c = _bare_client()
    L.note_head_reference(c, _head())
    slot_session_context.set("slack:t1:U0")
    opened = time.monotonic()
    L.note_main_slot_prompt(c, _head("hi", system="Public Ghost."), "stream", opened_at=opened)
    assert c._slot_session == "slack:t1:U0"
    time.sleep(0.002)
    L.release_slot_session(c)                               # another conversation's switch saved + released
    assert c._slot_changed_at > opened
    L.note_main_slot_prompt(c, _head("hi", system="Public Ghost."), "stream", opened_at=opened)
    assert c._slot_session is None                          # the drain does not take it back
    slot_session_context.set("sess-w")
    L.note_main_slot_prompt(c, _head("q"), "stream", opened_at=time.monotonic())
    assert c._slot_session == "sess-w"                      # a stream opened after the change takes it
    first_change = c._slot_changed_at
    L.note_main_slot_prompt(c, _head("q2"), "stream", opened_at=time.monotonic())
    assert c._slot_changed_at == first_change               # the same owner again is not a change


async def test_a_conversation_restore_stamps_the_change_and_clears_the_early_marks(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-b")
    _store(d, ckey, 41000)
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-b", slot=30000)
    llm._head_displaced_early, llm._head_displaced_by, llm._slot_changed_at = True, "deferred-verdict", 1.0
    t0 = time.monotonic()
    await agent._slot_switch_before(_turn(st))
    assert st["restores"][-1] == ckey
    assert llm._head_displaced_early is False and llm._head_displaced_by == "conversation restore"
    assert llm._slot_changed_at >= t0


async def test_a_kept_conversation_without_the_head_is_not_named_every_tick(cwarm, monkeypatch):
    """Review MINOR: with the owner on Slack holding the slot the head mark is
    True for hours; every periodic tick re-planned for a 'stream' displacement."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    llm._head_displaced, llm._head_displaced_by, llm._head_displaced_early = True, "stream", False
    llm._slot_session, llm._slot_session_used = "slack:t1:U0", time.monotonic()
    llm.before_background_dispatch = agent._conv_before_background

    async def sleep(_s):
        return None
    assert await agent._rewarm_wait(0.01, sleep) == ""
    assert await agent._rewarm_wait(0.01, sleep, early=False) == ""
    llm._slot_session = None                                # nobody holds it: a displacement to name
    assert await agent._rewarm_wait(0.01, sleep) == "stream"


# ── review round 2 ───────────────────────────────────────────────────────────

async def test_the_shutdown_save_does_not_wait_on_a_held_lock_past_its_budget(cwarm, monkeypatch):
    """r2 MAJOR: a background call's save can hold the conversation lock for a
    minute or more; launchd SIGKILLs 60 s after SIGTERM."""
    import asyncio
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    agent._conv_lock = asyncio.Lock()
    await agent._conv_lock.acquire()                       # another save holds it
    monkeypatch.setattr(A, "SHUTDOWN_CONV_SAVE_S", 0.3)
    t0 = time.monotonic()
    try:
        # r3 review: a hang is the failure here — bound the pin itself
        assert await asyncio.wait_for(agent.save_slot_conversation_at_shutdown(), 3.0) is False
    except asyncio.TimeoutError:
        pytest.fail("the shutdown save waited on the held lock past its budget")
    assert time.monotonic() - t0 < 3.0 and st["conv_saves"] == []
    agent._conv_lock.release()


async def test_a_level_file_is_restored_before_the_plain_one(cwarm, monkeypatch):
    """r2 MAJOR (pin gap): the restore must read the LEVEL's file."""
    from ghost_agent.core import effort as E
    monkeypatch.setitem(E._server_default, "level", "medium")
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    plain, lvl = H.conv_key("id", "sess-b"), H.conv_key("id", "sess-b", "xhigh")
    assert plain != lvl
    _store(d, plain, 39000)
    (d / f"head-{plain}.json").write_text(json.dumps({"n_tokens": 39000, "kind": H.CONV, "effort": "xhigh"}))
    _store(d, lvl, 41000)
    (d / f"head-{lvl}.json").write_text(json.dumps({"n_tokens": 41000, "kind": H.CONV, "effort": "xhigh"}))
    st["restores"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-b", slot=30000)
    agent._head_plan_last["effort"] = "xhigh"
    p = _turn(st); p["chat_template_kwargs"] = {"reasoning_effort": "xhigh"}
    await agent._slot_switch_before(p)
    assert st["restores"] == [lvl]
    # without the level's file, the plain one (a pre-§4SA save) still serves
    H.discard(d, lvl)
    llm._slot_session = None; st["restores"].clear(); agent._switch_done = {}
    await agent._slot_switch_before(p)
    assert st["restores"] == [plain]


def test_a_refused_late_stream_leaves_the_marks_alone():
    """r2 MAJOR: r1 refused the ownership but still wrote the head/moved-on
    marks and the call-end time from the stale payload."""
    c = _bare_client()
    L.note_head_reference(c, _head())
    slot_session_context.set("sess-w")
    L.note_main_slot_prompt(c, _head("q"), "stream", opened_at=time.monotonic())
    assert c._slot_session == "sess-w" and c._head_displaced is False
    end_before = c._last_main_call_end
    opened_long_ago = c._slot_changed_at - 1.0
    slot_session_context.set("slack:t1:U0")
    L.note_main_slot_prompt(c, _head("hi", system="Public Ghost."), "stream", opened_at=opened_long_ago)
    assert c._slot_session == "sess-w"
    assert c._head_displaced is False and c._slot_moved_on is False     # the slot still holds W's state
    assert c._last_main_call_end == end_before


async def test_the_file_gate_exempts_the_request_that_saved_it(cwarm, monkeypatch):
    """r2 MAJOR: a request past the gate saves its state before a picture look;
    its own next call extends that prompt whole — no checkpoint is needed."""
    agent, llm, st = cwarm
    monkeypatch.setenv(L.CONV_HEAD_FIRST_CALLS_ENV, "5")
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-a")
    _store(d, ckey, 41000)
    (d / f"head-{ckey}.json").write_text(json.dumps({"n_tokens": 41000, "kind": H.CONV, "head_in_slot": True,
                                                     "calls": 7, "rid": "req-a"}))
    st["restores"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner="sess-a", key="sess-a", slot=41400)
    _conv_env(agent, llm, st, monkeypatch)
    agent._side_call_rid = "req-a"
    llm._head_displaced = llm._slot_moved_on = True
    llm._slot_request = "req-a"
    tok = request_id_context.set("req-a")
    try:
        await agent._slot_switch_before(_turn(st))
    finally:
        request_id_context.reset(tok)
    assert st["restores"] == [ckey]                        # its own file, gate or not
    # another request of the conversation: gated
    (d / f"head-{ckey}.json").write_text(json.dumps({"n_tokens": 41000, "kind": H.CONV, "head_in_slot": True,
                                                     "calls": 7, "rid": "req-a"}))
    st["restores"].clear(); agent._switch_done = {}
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-a", slot=30000)
    tok = request_id_context.set("req-b")
    try:
        await agent._slot_switch_before(_turn(st))
    finally:
        request_id_context.reset(tok)
    assert st["restores"] == [agent._head_plan_last["key"]]


async def test_a_saves_meta_names_its_request_and_conversation(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    llm._slot_request = "req-z"
    agent._slot_level = "low"
    assert await agent._conv_save(llm, "x") is True
    assert st["extras"][-1]["rid"] == "req-z"
    assert st["extras"][-1]["conv"] == H.conv_key("id", "sess-a")          # the conversation, level-less
    assert st["conv_saves"][0][0] == H.conv_key("id", "sess-a", "low")


def test_prune_counts_conversations_not_their_level_files(tmp_path, monkeypatch):
    """r2 MINOR: per-level files shrank the rotation to two conversations."""
    import os
    d = _dir(tmp_path)
    monkeypatch.setattr(H, "KEEP_CONV", 3)
    t = time.time() - 100
    files = [("c1a", "conv1"), ("c1b", "conv1"), ("c1c", "conv1"), ("c2", "conv2"), ("c3", "conv3"), ("c4", "conv4")]
    for i, (key, conv) in enumerate(files):           # oldest first: c4 is the newest
        _store(d, key)
        (d / f"head-{key}.json").write_text(json.dumps({"n_tokens": 1, "kind": H.CONV, "conv": conv}))
        os.utime(d / f"head-{key}.bin", (t + i, t + i)); os.utime(d / f"head-{key}.json", (t + i, t + i))
    H.prune(d)
    left = sorted(p.name[5:-4] for p in d.glob("head-*.bin"))
    assert left == ["c2", "c3", "c4"]                 # three conversations; conv1's three files went together
    for key in ("c2", "c3", "c4"):
        H.discard(d, key)
    for i, (key, conv) in enumerate([("c1a", "conv1"), ("c1b", "conv1"), ("c2", "conv2"), ("c3", "conv3")]):
        _store(d, key)
        (d / f"head-{key}.json").write_text(json.dumps({"n_tokens": 1, "kind": H.CONV, "conv": conv}))
        os.utime(d / f"head-{key}.bin", (t + i, t + i)); os.utime(d / f"head-{key}.json", (t + i, t + i))
    H.prune(d)
    assert sorted(p.name[5:-4] for p in d.glob("head-*.bin")) == ["c1a", "c1b", "c2", "c3"]   # all four files, 3 conversations


def test_the_rewarms_release_does_not_stamp_a_change_of_hands():
    c = _bare_client()
    L.note_head_reference(c, _head())
    slot_session_context.set("sess-a")
    L.note_main_slot_prompt(c, _head("q"), "stream")
    before = c._slot_changed_at
    time.sleep(0.002)
    L.release_slot_session(c, stamp=False)
    assert c._slot_session is None and c._slot_changed_at == before
    L.release_slot_session(c)
    assert c._slot_changed_at > before


async def test_a_keyless_look_over_an_owned_slot_saves_the_owner_first(cwarm, monkeypatch):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, owner="sess-a", key="", slot=40000)
    _conv_env(agent, llm, st, monkeypatch)
    tok = request_id_context.set("probe-9")
    try:
        await agent._side_call_before({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})
    finally:
        request_id_context.reset(tok)
    assert [k for k, _n, _a in st["conv_saves"]] == [H.conv_key("id", "sess-a")]
    assert llm._slot_session is None and agent._side_call_rid == "probe-9"


async def test_a_deadline_less_save_gets_the_states_bound(cwarm, monkeypatch):
    """r2: the background hook passes no deadline; the 60 s idle-wait cap had
    cut an 80 s save."""
    agent, llm, st = cwarm
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", 150.0)      # the harness shortens it to 33 s
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=170000)
    _conv_env(agent, llm, st, monkeypatch, slot=170000)
    bounds = []

    def save(base, d, key, n, ident="", kind=H.OWNER, timeout=H.SAVE_TIMEOUT_S, at_least=False, extra=None):
        bounds.append(timeout)
        _store(d, key, n)
        return (True, f"{n} tokens, 7.9 GB")
    monkeypatch.setattr(H, "save", save)
    assert await agent._conv_save(llm, "a background call") is True
    want = min(H.save_timeout_s(170000), L.BEFORE_MAIN_DISPATCH_S - A.CONV_SAVE_WAIT_S - 15.0)
    assert bounds == [pytest.approx(want, abs=0.5)] and want > H.SAVE_TIMEOUT_S


async def test_the_first_tick_leaves_a_kept_conversation_alone(cwarm):
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    ref = llm._head_ref
    llm._head_ref = None
    llm._head_displaced, llm._head_displaced_by = True, "conversation restore"
    llm._slot_session, llm._slot_session_used = "slack:t1:U0", time.monotonic()
    llm.before_background_dispatch = agent._conv_before_background

    async def sleep(_s):
        llm._head_ref = ref                              # the boot warm-up registered the head meanwhile
    assert await agent._rewarm_wait(0.01, sleep) == ""


# ── review round 3 (pins) ────────────────────────────────────────────────────

async def test_the_rewarms_head_restore_releases_without_a_change_of_hands(cwarm, monkeypatch):
    """r3: the pin for the re-warm's release went through `release_slot_session`
    directly; this one goes through `_head_save_after` (what runs live)."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    llm._slot_session, llm._slot_changed_at = "sess-a", 5.0
    hs = {**agent._head_plan_last, "restored": True, "restore": True, "restore_s": 0.4, "parked": False,
          "detail": "28213 tokens"}
    await agent._head_save_after(llm, hs, _resp(28213, 28209), 0.5)
    assert llm._slot_session is None and llm._slot_changed_at == 5.0


async def test_a_claim_needs_the_same_file_as_the_last_save(cwarm, monkeypatch):
    """r3: the claim "already in the slot" matched any file of the session with
    the slot's token count — another level's file with an equal count."""
    agent, llm, st = cwarm
    await _warm_with_file(agent, st, quiet=True)
    d = pathlib.Path(H.head_dir())
    ckey = H.conv_key("id", "sess-b")
    _store(d, ckey, 41000)
    st["restores"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-b", slot=41000)
    st["slot"] = 41000                                     # what `slot_tokens` reports to the claim
    llm._last_main_call_end = time.monotonic() - 50
    llm._conv_last_saved = ("sess-b", 41000, time.monotonic() - 10, "someotherkey")
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == [ckey]                        # not claimed: the count matched another file
    st["restores"].clear(); agent._switch_done = {}; llm._slot_session = None
    llm._conv_last_saved = ("sess-b", 41000, time.monotonic() - 10, ckey)
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == []                            # this file: already in the slot
