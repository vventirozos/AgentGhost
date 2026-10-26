"""§4RP (2026-10-25): the main llama slot is kept for the conversation that
last used it, and the head is put back when another conversation starts.

Operator's rule (agreed after review of the gaps): a conversation keeps the
slot until it has been idle GHOST_SLOT_KEEP_S (600 s); a request of a
different conversation restores the saved head before its first call (the
previous one, if resumed, re-reads only its part past the head); a background
call that opens with the head waits out the window; nothing touches the slot
while a request runs (the re-warm already refuses); after a restart the owner
is unknown and the head comes back as before.

Why the switch is the agent's job and not llama's: with --ctx-checkpoints 4 a
conversation's own turns push the checkpoint at the head's end out, and the
next conversation then re-processes from token 0 (~166 s plus its own text).
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import time
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import llm as L
from ghost_agent.utils.logging import slot_session_context

pytestmark = pytest.mark.asyncio

TOOLS = [{"type": "function", "function": {"name": f"t{i}", "parameters": {}}} for i in range(8)]
SYS = "You are Ghost. " * 300


def _head(user="ok", tools=TOOLS, system=SYS):
    p = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if tools:
        p["tools"] = tools
    return p


def _bare_client():
    c = L.LLMClient.__new__(L.LLMClient)
    c.foreground_tasks = 0
    c.foreground_requests = 0
    c.main_slot_idle_windows = 0
    c._foreground_lock = asyncio.Lock()
    c._bg_queue_sem = asyncio.Semaphore(3)
    c.upstream_url = "http://127.0.0.1:8088"
    c.worker_clients = c.critic_clients = c.vision_clients = c.swarm_clients = c.coding_clients = []
    c._maybe_record_call = lambda *a, **k: None
    c.before_main_dispatch = None
    return c


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("GHOST_SLOT_KEEP_S", raising=False)
    tok = slot_session_context.set("")
    yield
    slot_session_context.reset(tok)


# ── the window ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,want", [(None, 600.0), ("", 600.0), ("0", 0.0), ("900", 900.0),
                                      ("-5", 0.0), ("x", 600.0)])
def test_the_window_comes_from_the_env(monkeypatch, raw, want):
    if raw is not None:
        monkeypatch.setenv("GHOST_SLOT_KEEP_S", raw)
    assert L.slot_keep_s() == want


def test_a_known_conversation_keeps_the_slot_for_the_window(monkeypatch):
    c = _bare_client()
    now = 1000.0
    c._slot_session, c._slot_session_used = "sess-a", now - 599
    assert L.slot_kept(c, now) is True
    c._slot_session_used = now - 601
    assert L.slot_kept(c, now) is False
    c._slot_session, c._slot_session_used = None, now
    assert L.slot_kept(c, now) is False                 # no owner: nothing kept
    c._slot_session = "sess-a"
    monkeypatch.setenv("GHOST_SLOT_KEEP_S", "0")
    assert L.slot_kept(c, now) is False                 # off
    assert L.slot_kept(object(), now) is False


def test_only_a_requests_own_call_takes_the_slot():
    c = _bare_client()
    L.note_head_reference(c, _head())
    slot_session_context.set("sess-a")
    L.note_main_slot_prompt(c, _head("q"), "stream", background=False)
    assert c._slot_session == "sess-a" and time.monotonic() - c._slot_session_used < 5
    slot_session_context.set("")
    L.note_main_slot_prompt(c, _head("bg"), "deferred-verdict", background=True)
    assert c._slot_session == "sess-a"                  # a background call owns nothing
    L.note_main_slot_prompt(c, _head("move"), "game", background=False)
    assert c._slot_session == "sess-a"                  # review r1: no conversation key leaves it
    slot_session_context.set("sess-b")
    L.note_main_slot_prompt(c, _head("hi", tools=None, system="Greet."), "greeting fast path",
                            background=False)
    assert c._slot_session is None                      # review r1 MAJOR: it replaced the head
    L.note_main_slot_prompt(c, _head("q"), "stream", background=False)
    assert c._slot_session == "sess-b"
    L.release_slot_session(c)
    assert c._slot_session is None


def test_a_call_without_the_head_leaves_the_owner():
    """Review r2 MAJOR: a game move (no system, no tools), a truncation
    continuation (tools dropped), a vision call served on main, a member's
    turn — none of them may end the owner's window."""
    c = _bare_client()
    L.note_head_reference(c, _head())
    slot_session_context.set("sess-a")
    L.note_main_slot_prompt(c, _head("q"), "stream")
    t_a = c._slot_session_used
    for key, p in (("", {"messages": [{"role": "user", "content": "e4"}]}),
                   ("member:U2", _head("hi", system="Member Ghost.")),
                   ("sess-c", _head("look", tools=None))):
        slot_session_context.set(key)
        L.note_main_slot_prompt(c, p, "vision", background=False)
        assert c._slot_session == "sess-a" and c._slot_session_used == t_a, key
    slot_session_context.set("sess-a")
    L.note_main_slot_prompt(c, _head("cont", tools=None), "continuation", background=False)
    assert c._slot_session == "sess-a" and c._slot_session_used > t_a   # its own call refreshes


def test_a_stream_ending_after_another_conversation_took_the_slot_leaves_it():
    """Review r2 MINOR: A's final drains while B's first call already ran."""
    c = _bare_client()
    L.note_head_reference(c, _head())
    opened = time.monotonic()
    slot_session_context.set("sess-b")
    L.note_main_slot_prompt(c, _head("b1"), "tool loop")
    slot_session_context.set("sess-a")
    L.note_main_slot_prompt(c, _head("a-final"), "stream", opened_at=opened)
    assert c._slot_session == "sess-b"
    # a stream that opened after the owner's last call takes the slot as usual
    L.note_main_slot_prompt(c, _head("a-next"), "stream", opened_at=time.monotonic())
    assert c._slot_session == "sess-a"


async def test_the_stream_reports_when_it_opened():
    c = _bare_client()
    c._main_streams_open = 0
    seen = {}
    L.note_head_reference(c, _head())

    async def stream(payload, use_coding):
        yield b"x"
    c._do_stream_chat_completion = stream
    real = L.note_main_slot_prompt
    t0 = time.monotonic()

    def spy(llm, payload, label="", background=False, opened_at=None):
        seen["opened_at"] = opened_at
        return real(llm, payload, label, background, opened_at)
    L.note_main_slot_prompt = spy
    try:
        async for _ in c.stream_chat_completion(_head("q")):
            pass
    finally:
        L.note_main_slot_prompt = real
    assert seen["opened_at"] is not None and seen["opened_at"] >= t0


# ── rule 5: a head-shaped background call waits out the window ─────────

async def test_a_background_call_opening_with_the_head_is_deferred_inside_the_window():
    c = _bare_client()
    L.note_head_reference(c, _head())
    c._slot_session, c._slot_session_used = "sess-a", time.monotonic()
    sent = []

    async def do(payload, *a, **k):
        sent.append(payload)
        return {"choices": [{"message": {"content": "x"}}], "_ghost_leg": {"served_by": "main"}}
    c._do_chat_completion = do
    with pytest.raises(L.BackgroundDeferred):
        await c.chat_completion(_head("job"), is_background=True, task_label="selfhood narrative")
    assert sent == []
    assert c._bg_queue_sem._value == 3                  # the lane gave its permit back
    # a call that does not open with the head runs: llama parks the conversation itself
    await c.chat_completion(_head("v", system="Judge."), is_background=True, task_label="deferred-verdict")
    assert len(sent) == 1
    # the head warm-up is never held by this rule (its own rules decide)
    await c.chat_completion(_head(), is_background=True, task_label="main-prefix-rewarm")
    assert len(sent) == 2
    # outside the window it runs
    c._slot_session_used = time.monotonic() - 601
    await c.chat_completion(_head("job"), is_background=True, task_label="selfhood narrative")
    assert len(sent) == 3


# ── rule 3: the hook before a foreground main call ─────────────────────

async def test_the_hook_runs_before_a_plain_main_call_only():
    c = _bare_client()
    seen, order = [], []

    async def hook(p):
        seen.append(p)
        order.append("hook")

    async def do(*a, **k):
        order.append("call")
        return {"choices": [{"message": {"content": "x"}}]}
    c.before_main_dispatch = hook
    c._do_chat_completion = do
    await c.chat_completion({"messages": [1]})
    assert seen == [{"messages": [1]}] and order == ["hook", "call"]
    for kw in ("use_worker", "use_critic", "use_vision", "use_swarm", "use_coding", "off_main_only"):
        seen.clear()
        await c.chat_completion({"messages": [2]}, **{kw: True})
        assert seen == [], kw
    seen.clear()
    await c.chat_completion({"messages": [3]}, is_background=True, task_label="x")
    assert seen == []                                   # background calls: never


async def test_the_hook_runs_before_a_foreground_main_stream():
    c = _bare_client()
    c._main_streams_open = 0
    order = []

    async def hook(p):
        order.append("hook")

    async def stream(payload, use_coding):
        order.append("stream")
        yield b"x"
    c.before_main_dispatch = hook
    c._do_stream_chat_completion = stream
    async for _ in c.stream_chat_completion({"messages": []}):
        pass
    assert order == ["hook", "stream"]


async def test_a_failing_or_hung_hook_never_fails_the_call(monkeypatch):
    c = _bare_client()

    async def boom(p):
        raise RuntimeError("llama down")
    c.before_main_dispatch = boom
    await c._before_main_dispatch({})
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", 0.05)

    async def hang(p):
        await asyncio.sleep(10)
    c.before_main_dispatch = hang
    t0 = time.monotonic()
    await c._before_main_dispatch({})
    assert time.monotonic() - t0 < 2


def test_the_hook_budget_sits_above_its_http_bounds():
    from ghost_agent.core import head_saves as H
    assert 3.0 + H.RESTORE_TIMEOUT_S < L.BEFORE_MAIN_DISPATCH_S


# ── the request's conversation key ─────────────────────────────────────

def test_the_key_is_the_session_id_else_the_first_message_fingerprint():
    from ghost_agent.core.agent import _set_slot_session
    from ghost_agent.tools.projects import conversation_fingerprint
    _set_slot_session({"session_id": "s-123", "messages": [{"role": "user", "content": "hi"}]})
    assert slot_session_context.get() == "s-123"
    _set_slot_session({"session_id": "slack:1712.5:U1", "messages": []})
    assert slot_session_context.get() == "slack:1712.5:U1"
    msgs = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "a"},
            {"role": "user", "content": "second"}]
    _set_slot_session({"messages": msgs})
    assert slot_session_context.get() == "fp:" + conversation_fingerprint(msgs)
    _set_slot_session({"messages": []})
    assert slot_session_context.get() == ""
    _set_slot_session(None)
    assert slot_session_context.get() == ""


def test_handle_chat_sets_the_key_before_any_call():
    import ast
    import textwrap
    from ghost_agent.core.agent import GhostAgent
    tree = ast.parse(textwrap.dedent(inspect.getsource(GhostAgent.handle_chat)))
    lines = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (
                f"{f.value.id}.{f.attr}" if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else "")
            lines.setdefault(name, []).append(node.lineno)
    # at the top, next to the session id the request carries — before the
    # requester role, the origin and any turn work
    assert lines["_set_slot_session"]
    assert min(lines["_set_slot_session"]) < min(lines["request_origin_context.set"])
    # review r1: a probe or replay request (an idle job's handle_chat) owns nothing
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and isinstance(c.func, ast.Name) and c.func.id == "_set_slot_session"]
    arg = calls[0].args[0]
    assert isinstance(arg, ast.IfExp) and isinstance(arg.body, ast.Constant) and arg.body.value is None
    names = {n.id for n in ast.walk(arg.test) if isinstance(n, ast.Name)}
    assert {"_is_probe_rid", "_UNOWNED_RID_PREFIXES", "req_id"} <= names


def test_unattended_turns_are_named_by_their_request_ids():
    """Review r2: scheduled tasks ("sched-<job>") and job resumes ("sub-<id>")
    run handle_chat with nobody present — they must not own the slot."""
    from ghost_agent.core import agent as A
    assert {"sched-", "sub-", "job-"} <= set(A._UNOWNED_RID_PREFIXES)   # r3: owner-started job wakes
    from ghost_agent.core.failure_census import _NOT_OWNER_PREFIXES
    assert all(any(p.startswith(n) or n.startswith(p) for n in _NOT_OWNER_PREFIXES)
               for p in A._UNOWNED_RID_PREFIXES)
    src = inspect.getsource(__import__("ghost_agent.main", fromlist=["x"]))
    tree = ast.parse(src)
    ids = [n for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)
           and isinstance(n.values[0], ast.Constant) and str(n.values[0].value) in ("sched-", "sub-")]
    assert len({n.values[0].value for n in ids}) == 2   # main.py still mints both
