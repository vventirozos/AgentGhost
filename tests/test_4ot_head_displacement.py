"""§4OT (2026-10-13): keep short main-model calls from costing the owner the
head. §4OS measured it: since --cache-ram 0 every big re-read (>15k tokens,
30-45 s) followed a short main-model call that replaced the 28k-token head on
the single llama slot — 13 of 20 in one run were a deferred verdict's
escalation to the main model. Two fixes:
1. a deferred verdict's main-model calls park behind any live request;
2. the re-warm loop re-sends the head as soon as the slot is quiet after a
   call displaced it, instead of at the next 180 s tick."""
from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core import llm as L
from ghost_agent.core import verifier as V
from tests import pin_ast as P

pytestmark = pytest.mark.asyncio

TOOLS = [{"type": "function", "function": {"name": f"t{i}", "parameters": {}}} for i in range(8)]
SYS = "You are Ghost. " * 300


def _head(user="ok", tools=TOOLS, system=SYS):
    p = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if tools:
        p["tools"] = tools
    return p


# ── 1. a deferred verdict parks its main-model calls ───────────────────

class _Stub:
    critic_clients = None

    def __init__(self, fg):
        self.foreground_requests = fg
        self.calls = []

    async def chat_completion(self, payload, **kw):
        self.calls.append(kw)
        return {"choices": [{"message": {"content": '{"verdict": "CONFIRMED", "confidence": 0.9}'}}]}


def _with_mark(mark, fn):
    tok = V.deferred_verdict_context.set(mark)
    try:
        return fn()
    finally:
        V.deferred_verdict_context.reset(tok)


@pytest.mark.parametrize("fg", [0, 1, 3])
def test_a_deferred_verdict_is_background_even_while_a_request_is_live(fg):
    kw = _with_mark({"nova_busy": False, "internal": False}, lambda: V._bounded_fallback_kwargs(_Stub(fg)))
    assert kw["is_background"] is True
    assert kw["task_label"] == "deferred-verdict"
    assert kw["timeout"] == V._VERIFY_FALLBACK_TIMEOUT_S


def test_an_in_turn_verdict_stays_foreground_while_a_request_is_live():
    """The awaited verdict runs inside the turn: parking it would wait on
    its own request (the self-stall the old rule exists for)."""
    kw = _with_mark(False, lambda: V._bounded_fallback_kwargs(_Stub(1)))
    assert "is_background" not in kw and "task_label" not in kw
    kw = _with_mark(False, lambda: V._bounded_fallback_kwargs(_Stub(0)))
    assert kw["is_background"] is True and "task_label" not in kw


def test_a_client_without_the_keywords_gets_none_of_them():
    class Legacy:
        critic_clients = None

        async def chat_completion(self, payload, timeout=None):
            return {}
    kw = _with_mark({"nova_busy": False}, lambda: V._bounded_fallback_kwargs(Legacy()))
    assert kw == {"timeout": V._VERIFY_FALLBACK_TIMEOUT_S}


async def test_the_escalation_call_carries_the_park():
    stub = _Stub(1)
    v = V.Verifier(llm_client=stub)
    tok = V.deferred_verdict_context.set({"nova_busy": False, "internal": False})
    try:
        out = await v._call_llm("judge this", force_main=True)
    finally:
        V.deferred_verdict_context.reset(tok)
    assert out.get("verdict") == "CONFIRMED"
    assert stub.calls and all(c.get("is_background") is True for c in stub.calls)


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
    return c


async def test_end_to_end_the_escalation_waits_for_the_request_to_end(monkeypatch):
    c = _bare_client()
    c.foreground_requests = 1
    ran = []

    async def do(payload, *a, **k):
        ran.append(time.monotonic())
        return {"choices": [{"message": {"content": '{"verdict": "CONFIRMED", "confidence": 0.9}'}}],
                "_ghost_leg": {"served_by": "main"}}
    c._do_chat_completion = do
    real = asyncio.sleep

    async def fast(_s):
        await real(0)
    monkeypatch.setattr(asyncio, "sleep", fast)
    v = V.Verifier(llm_client=c)
    tok = V.deferred_verdict_context.set({"nova_busy": False, "internal": False})
    try:
        task = asyncio.ensure_future(v._call_llm("judge this", force_main=True))
        for _ in range(50):
            await real(0)
        assert not ran and not task.done()            # parked behind the live request
        c.foreground_requests = 0
        out = await task
    finally:
        V.deferred_verdict_context.reset(tok)
    assert ran and out.get("verdict") == "CONFIRMED"


async def test_a_park_that_runs_out_is_information_not_a_failure(caplog):
    class Deferring(_Stub):
        async def chat_completion(self, payload, **kw):
            raise L.BackgroundDeferred("background call deferred: deferred-verdict")
    v = V.Verifier(llm_client=Deferring(1))
    route = {}
    tok = V.deferred_verdict_context.set({"nova_busy": False, "internal": False})
    try:
        with caplog.at_level(logging.INFO):
            out = await v._call_llm("judge this", force_main=True, route_out=route)
    finally:
        V.deferred_verdict_context.reset(tok)
    assert out == {} and route["route"] == "failed"
    recs = [r for r in caplog.records if "Verifier" in r.getMessage()]
    assert recs and all(r.levelno == logging.INFO for r in recs)
    assert any("whole wait" in r.getMessage() for r in recs)


async def test_a_real_failure_still_warns(caplog):
    class Broken(_Stub):
        async def chat_completion(self, payload, **kw):
            raise RuntimeError("upstream down")
    v = V.Verifier(llm_client=Broken(0))
    with caplog.at_level(logging.INFO):
        assert await v._call_llm("judge this", force_main=True) == {}
    assert any(r.levelno == logging.WARNING and "upstream down" in r.getMessage() for r in caplog.records)


# ── 2. the client marks a displaced head ───────────────────────────────

def test_a_call_opens_with_the_head_when_both_warmed_parts_are_its_prefix():
    ref = L.head_shape(_head())
    assert L.opens_with_head(_head(user="a different question"), ref)
    assert L.opens_with_head(_head(system=SYS + "volatile tail " * 50), ref)          # the live tail follows
    assert not L.opens_with_head(_head(tools=TOOLS + [{"x": "routed skill"}]), ref)   # r2: tools render first
    assert L.opens_with_head(_head(system=SYS[:500] + "\r" + SYS[500:]), ref)         # \r stripped, as the live slot
    assert not L.opens_with_head(_head(tools=None), ref)                              # the verifier's shape
    assert not L.opens_with_head(_head(system="lite greeting prompt"), ref)
    assert not L.opens_with_head(_head(tools=list(reversed(TOOLS))), ref)
    assert not L.opens_with_head(_head(tools=TOOLS[:7]), ref)                         # self-play drops a tool
    assert not L.opens_with_head(_head(system=SYS[:-5] + "other"), ref)               # r1: the profile is at the END
    assert not L.opens_with_head({"messages": [{"role": "user", "content": SYS}], "tools": TOOLS}, ref)
    assert not L.opens_with_head(_head(), None)
    assert L.head_shape(None) is None


def test_no_reference_no_mark():
    c = SimpleNamespace()
    L.note_main_slot_prompt(c, _head(tools=None), "x")
    assert not hasattr(c, "_head_displaced")
    c._head_ref = "not a tuple"
    L.note_main_slot_prompt(c, _head(tools=None), "x")
    assert not hasattr(c, "_head_displaced")


def test_a_short_call_marks_and_a_head_call_clears():
    c = SimpleNamespace()
    L.note_head_reference(c, _head())
    L.note_main_slot_prompt(c, _head(tools=None, system="judge"), "deferred-verdict")
    assert c._head_displaced is True and c._head_displaced_by == "deferred-verdict"
    t = c._last_main_call_end
    L.note_main_slot_prompt(c, _head(user="the owner's next step"), "")
    assert c._head_displaced is False and c._last_main_call_end >= t
    L.note_main_slot_prompt(c, _head(tools=None), "")
    assert c._head_displaced_by == "unlabelled"


def _due_client(**kw):
    c = SimpleNamespace(foreground_requests=0, foreground_tasks=0, _main_streams_open=0)
    L.note_head_reference(c, _head())
    L.note_main_slot_prompt(c, _head(tools=None), "deferred-verdict", background=True)
    c._last_main_call_end = time.monotonic() - 60
    for k, v in kw.items():
        setattr(c, k, v)
    return c


@pytest.mark.parametrize("kw,due", [
    ({}, True),
    ({"foreground_requests": 1}, False),
    ({"foreground_tasks": 1}, False),
    ({"_main_streams_open": 1}, False),
    ({"_head_displaced": False}, False),
    ({"_last_main_call_end": None}, True),                 # unreadable end → long ago
    ({"foreground_requests": "n/a"}, False),               # unreadable counter → not due
])
def test_due_needs_a_mark_a_quiet_slot_and_nothing_live(kw, due):
    assert L.head_rewarm_due(_due_client(**kw), 10.0) is due


def test_due_waits_for_the_slot_to_settle():
    c = _due_client()
    c._last_main_call_end = time.monotonic() - 3
    assert L.head_rewarm_due(c, 10.0) is False
    assert L.head_rewarm_due(c, 2.0) is True
    assert L.head_rewarm_due(SimpleNamespace(), 0.0) is False


@pytest.mark.parametrize("served,marked", [("main", True), ("critic", False), ("worker", False), ("", False)])
@pytest.mark.parametrize("bg", [False, True])
async def test_chat_completion_marks_only_a_main_served_call(served, marked, bg):
    c = _bare_client()
    L.note_head_reference(c, _head())

    async def do(payload, *a, **k):
        return {"choices": [{"message": {"content": "x"}}], "_ghost_leg": {"served_by": served}}
    c._do_chat_completion = do
    await c.chat_completion(_head(tools=None), is_background=bg, task_label="deferred-verdict")
    assert (getattr(c, "_head_displaced", None) is True) is marked
    if marked:
        assert c._head_displaced_by == "deferred-verdict"
        assert c._head_displaced_early is bg              # r2: only a call nobody waits for


async def test_a_stream_counts_while_open_and_marks_at_its_end():
    c = _bare_client()
    L.note_head_reference(c, _head())
    seen = []

    async def do(payload, use_coding=False):
        seen.append(c._main_streams_open)
        yield b"data: {}"
        yield b"data: [DONE]"
    c._do_stream_chat_completion = do
    async for _ in c.stream_chat_completion(_head(tools=None, system="member prompt")):
        pass
    assert seen == [1] and c._main_streams_open == 0
    assert c._head_displaced is True and c._head_displaced_by == "stream"
    async for _ in c.stream_chat_completion(_head()):
        pass
    assert c._head_displaced is False


async def test_a_stream_closed_early_releases_both_counters():
    c = _bare_client()

    async def do(payload, use_coding=False):
        yield b"data: {}"
        yield b"data: {}"
    c._do_stream_chat_completion = do
    gen = c.stream_chat_completion(_head())
    await gen.__anext__()
    assert c._main_streams_open == 1 and c.foreground_tasks == 1
    await gen.aclose()
    assert c._main_streams_open == 0 and c.foreground_tasks == 0


async def test_a_coding_pool_stream_is_not_the_main_slot():
    c = _bare_client()
    c.coding_clients = [{"url": "http://coder:8088"}]
    L.note_head_reference(c, _head())

    async def do(payload, use_coding=False):
        yield b"data: {}"
    c._do_stream_chat_completion = do
    async for _ in c.stream_chat_completion(_head(tools=None), use_coding=True):
        pass
    assert getattr(c, "_head_displaced", None) is None


# ── 3. the re-warm loop answers the mark ───────────────────────────────

def _agent(llm):
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.context = SimpleNamespace(llm_client=llm)
    return a


async def test_a_displaced_head_is_rewarmed_on_the_next_poll_not_the_next_tick():
    from ghost_agent.core import agent as A
    llm = _due_client()
    a = _agent(llm)
    warmed, slept = [], []

    async def warm(*, quiet=False, displaced_by=""):
        warmed.append(displaced_by)
        L.note_main_slot_prompt(llm, _head(), "main-prefix-rewarm")     # the re-warm IS the head

    async def sleep(s):
        slept.append(s)
        if len(slept) > 3:
            raise asyncio.CancelledError()

    async def metrics():
        return 0, len(slept)
    a.warm_up_main_prefix = warm
    with pytest.raises(asyncio.CancelledError):
        await a.rewarm_main_prefix_loop(180, sleep=sleep, metrics=metrics)
    assert warmed == ["deferred-verdict"]
    assert slept[0] == A.HEAD_POLL_S                     # re-warmed after one poll, not 180 s
    assert sum(slept[1:]) <= 180                         # then back to the period, in polls


async def test_nothing_happens_while_a_request_is_live():
    llm = _due_client(foreground_requests=1)
    a = _agent(llm)
    warmed, slept = [], []

    async def warm(**kw):
        warmed.append(kw)

    async def sleep(s):
        slept.append(s)
        if len(slept) > 40:
            raise asyncio.CancelledError()

    async def metrics():
        return 0, len(slept)
    a.warm_up_main_prefix = warm
    with pytest.raises(asyncio.CancelledError):
        await a.rewarm_main_prefix_loop(180, sleep=sleep, metrics=metrics)
    assert warmed == []


async def test_a_rewarm_that_did_not_land_waits_the_full_period():
    """A failing node must not get a 5 s retry loop."""
    llm = _due_client()
    a = _agent(llm)
    warmed, slept = [], []

    async def warm(**kw):
        warmed.append(kw.get("displaced_by", ""))         # fails: the mark stays

    async def sleep(s):
        slept.append(s)
        if len(slept) > 2:
            raise asyncio.CancelledError()

    async def metrics():
        return 0, len(slept)
    a.warm_up_main_prefix = warm
    with pytest.raises(asyncio.CancelledError):
        await a.rewarm_main_prefix_loop(180, sleep=sleep, metrics=metrics)
    assert slept[:2] == [5.0, 180.0]
    assert warmed[:2] == ["deferred-verdict", "deferred-verdict"]


async def test_a_busy_slot_is_polled_not_rewarmed():
    llm = _due_client()
    a = _agent(llm)
    warmed, slept = [], []

    async def warm(**kw):
        warmed.append(kw)

    async def sleep(s):
        slept.append(s)
        if len(slept) > 4:
            raise asyncio.CancelledError()

    async def metrics():
        return 1, len(slept)                               # a background call is running
    a.warm_up_main_prefix = warm
    with pytest.raises(asyncio.CancelledError):
        await a.rewarm_main_prefix_loop(180, sleep=sleep, metrics=metrics)
    assert warmed == [] and set(slept) == {5.0}


async def test_a_client_without_a_reference_sleeps_the_period_as_before():
    a = _agent(SimpleNamespace(foreground_requests=0))
    slept = []

    async def sleep(s):
        slept.append(s)
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await a._rewarm_wait(180.0, sleep)
    assert slept == [180.0]


# ── 4. the warm-up registers the head and says what happened ───────────

@pytest.fixture
def warm_agent(mock_context):
    from unittest.mock import MagicMock
    from ghost_agent.core.agent import GhostAgent
    llm = MagicMock()
    llm.upstream_url = "http://127.0.0.1:8088"
    mock_context.llm_client = llm
    return GhostAgent(mock_context), llm


def _resp(prompt, cached):
    return {"choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": prompt, "prompt_tokens_details": {"cached_tokens": cached}}}


async def test_the_warmup_payload_becomes_the_reference(warm_agent):
    agent, llm = warm_agent
    llm.chat_completion = AsyncMock(return_value=_resp(28000, 26000))
    await agent.warm_up_main_prefix(quiet=True)
    sent = llm.chat_completion.call_args.args[0]
    assert llm._head_ref == L.head_shape(sent)


@pytest.mark.parametrize("cached,needle", [(0, "re-warmed the head after a 'deferred-verdict' call"),
                                           (27000, "found the head still cached")])
async def test_a_displacement_rewarm_logs_information(warm_agent, cached, needle, monkeypatch):
    from ghost_agent.core import agent as A
    agent, llm = warm_agent
    llm.chat_completion = AsyncMock(return_value=_resp(28000, cached))
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda title, msg, **kw: lines.append((msg, kw)))
    await agent.warm_up_main_prefix(quiet=True, displaced_by="deferred-verdict")
    hit = [kw for msg, kw in lines if needle in msg]
    assert hit and hit[0].get("level", "INFO") == "INFO"
    assert not any("would have paid it" in msg for msg, _ in lines)


async def test_a_periodic_rewarm_that_finds_an_eviction_still_warns(warm_agent, monkeypatch):
    from ghost_agent.core import agent as A
    agent, llm = warm_agent
    llm.chat_completion = AsyncMock(return_value=_resp(28000, 0))
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda title, msg, **kw: lines.append(msg))
    await agent.warm_up_main_prefix(quiet=True)
    assert any("would have paid it" in m for m in lines)


# ── 5. wiring ──────────────────────────────────────────────────────────

def test_no_greeting_fast_path_is_left():
    """§4RP-c: the greeting fast path (its own lite prompt on the main slot)
    was removed; nothing labels a call with it any more."""
    from ghost_agent.core.agent import GhostAgent
    t = P.tree(GhostAgent)
    labels = [P.U(P.call_kw(c, "task_label")) for c in P.calls(t, "chat_completion")]
    assert "'greeting fast path'" not in labels
    assert not hasattr(L, "EARLY_REWARM_LABELS")


def test_the_loop_waits_through_the_displacement_wait():
    from ghost_agent.core.agent import GhostAgent
    t = P.tree(GhostAgent.rewarm_main_prefix_loop)
    assert P.calls(t, "_rewarm_wait")
    assert not [c for c in P.calls(t, "_sleep") if c.args and P.U(c.args[0]) == "float(period_s)"]


# ── r1 review fixes ────────────────────────────────────────────────────

async def test_a_redisplaced_rewarm_whose_mark_cleared_while_it_waited_is_refused():
    """r1 MAJOR: a request started and ended while the re-warm waited; its
    conversation is in the slot now and the re-warm would cut it back."""
    c = _bare_client()
    L.note_head_reference(c, _head())
    c._head_displaced = False
    sent = []

    async def do(payload, *a, **k):
        sent.append(1)
        return {"_ghost_leg": {"served_by": "main"}}
    c._do_chat_completion = do
    with pytest.raises(L.BackgroundDeferred):
        await c.chat_completion(_head(), is_background=True, task_label="main-prefix-redisplaced")
    assert not sent
    c._head_displaced = True
    await c.chat_completion(_head(), is_background=True, task_label="main-prefix-redisplaced")
    assert sent and c._head_displaced is False


async def test_the_displacement_rewarm_carries_its_own_label(warm_agent):
    agent, llm = warm_agent
    llm.chat_completion = AsyncMock(return_value=_resp(28000, 0))
    await agent.warm_up_main_prefix(quiet=True, displaced_by="deferred-verdict")
    assert llm.chat_completion.call_args.kwargs["task_label"] == "main-prefix-redisplaced"
    await agent.warm_up_main_prefix(quiet=True)
    assert llm.chat_completion.call_args.kwargs["task_label"] == "main-prefix-rewarm"


@pytest.mark.parametrize("vision_url,marked", [("http://127.0.0.1:8088", True), ("http://127.0.0.1:8088/v1/", True),
                                               ("http://nova:8088", False)])
async def test_a_pool_served_on_the_main_box_marks_too(vision_url, marked):
    """r1 MAJOR: live, the vision pool IS the main URL."""
    c = _bare_client()
    c.vision_clients = [{"url": vision_url}]
    L.note_head_reference(c, _head())

    async def do(payload, *a, **k):
        return {"_ghost_leg": {"served_by": "vision"}}
    c._do_chat_completion = do
    await c.chat_completion(_head(tools=None), use_vision=True)
    assert (getattr(c, "_head_displaced", None) is True) is marked


def test_main_box_needs_every_node_on_the_main_url():
    c = _bare_client()
    c.vision_clients = [{"url": "http://127.0.0.1:8088"}, {"url": "http://nova:8088"}]
    assert c._served_on_main_box("main") and not c._served_on_main_box("vision")
    assert not c._served_on_main_box("") and not c._served_on_main_box("worker")
    c.upstream_url = ""
    c.vision_clients = [{"url": ""}]
    assert not c._served_on_main_box("vision")


async def test_a_deferred_verdict_giving_up_is_information(monkeypatch):
    c = _bare_client()
    c.foreground_requests = 1
    lines = []
    monkeypatch.setattr(L, "pretty_log", lambda t, m, **kw: lines.append((m, kw.get("level", "INFO"))))
    real = asyncio.sleep

    async def fast(_s):
        await real(0)
    monkeypatch.setattr(asyncio, "sleep", fast)
    with pytest.raises(L.BackgroundDeferred):
        await c._wait_for_foreground_clear("deferred-verdict")
    assert lines[-1][1] == "INFO" and "gave up" in lines[-1][0]
    with pytest.raises(L.BackgroundDeferred):
        await c._wait_for_foreground_clear("selfplay-judge")
    assert lines[-1][1] == "WARNING"


async def test_a_render_window_never_admits_a_deferred_verdict(monkeypatch):
    c = _bare_client()
    c.foreground_requests = 1
    c.main_slot_idle_windows = 1
    c._idle_window_admits = 1
    monkeypatch.setattr(L, "_bg_during_image_wait_enabled", lambda: True)
    real = asyncio.sleep
    n = {"i": 0}

    async def fast(_s):
        n["i"] += 1
        if n["i"] > 20:
            c.foreground_requests = 0
        await real(0)
    monkeypatch.setattr(asyncio, "sleep", fast)
    await c._wait_for_foreground_clear("deferred-verdict")
    assert n["i"] > 20 and c._idle_window_admits == 1      # waited for the request, not the window
    c.foreground_requests = 1
    n["i"] = 0
    await c._wait_for_foreground_clear("memory extract")
    assert n["i"] == 0 and c._idle_window_admits == 0       # other background work still uses it


async def test_deferred_verdict_calls_take_the_main_slot_one_at_a_time():
    c = _bare_client()
    live, peak = [0], [0]

    async def do(payload, *a, **k):
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        await asyncio.sleep(0.01)
        live[0] -= 1
        return {"_ghost_leg": {"served_by": "main"}}
    c._do_chat_completion = do
    await asyncio.gather(*[c.chat_completion(_head(tools=None), is_background=True, task_label="deferred-verdict")
                           for _ in range(3)])
    assert peak[0] == 1
    live[0] = peak[0] = 0
    await asyncio.gather(*[c.chat_completion(_head(tools=None), is_background=True, task_label="memory extract")
                           for _ in range(3)])
    assert peak[0] == 3


async def test_a_parked_stream_is_not_open_and_a_refused_one_marks_nothing(monkeypatch):
    c = _bare_client()
    L.note_head_reference(c, _head())
    c.foreground_requests = 1
    seen = []

    async def fake_wait(label=""):
        seen.append(c._main_streams_open if hasattr(c, "_main_streams_open") else 0)
        raise L.BackgroundDeferred("x")
    c._wait_for_foreground_clear = fake_wait
    with pytest.raises(L.BackgroundDeferred):
        async for _ in c.stream_chat_completion(_head(tools=None), is_background=True):
            pass
    assert seen == [0] and int(getattr(c, "_main_streams_open", 0) or 0) == 0
    assert getattr(c, "_head_displaced", None) is None

    async def broken(payload, use_coding=False):
        raise RuntimeError("connect error")
        yield b""
    c._do_stream_chat_completion = broken
    with pytest.raises(RuntimeError):
        async for _ in c.stream_chat_completion(_head(tools=None)):
            pass
    assert getattr(c, "_head_displaced", None) is None and c._main_streams_open == 0


async def test_a_rewarm_refused_for_an_arriving_request_keeps_the_fast_path():
    """r1 minor: the refusal is not a failure; the request settles the mark."""
    llm = _due_client()
    a = _agent(llm)
    slept = []

    async def warm(**kw):
        llm.foreground_requests = 1                     # a request arrived; the re-warm was refused

    async def sleep(s):
        slept.append(s)
        if len(slept) == 2:
            llm.foreground_requests = 0
        if len(slept) > 2:
            raise asyncio.CancelledError()

    async def metrics():
        return 0, len(slept)
    a.warm_up_main_prefix = warm
    with pytest.raises(asyncio.CancelledError):
        await a.rewarm_main_prefix_loop(180, sleep=sleep, metrics=metrics)
    assert slept[:2] == [5.0, 5.0]


def test_a_member_verdict_awaited_inline_is_never_deferred():
    """r1 minor: a deferred call parks behind live requests — this one too."""
    from ghost_agent.core.agent import GhostAgent
    t = P.tree(GhostAgent._gate_verifier_verdict) if hasattr(GhostAgent, "_gate_verifier_verdict") else P.tree(GhostAgent)
    vals = [P.U(P.call_kw(c, "deferred")) for c in P.calls(t, "_compute_verifier_verdict")]
    assert "gate <= 0 and (not _awaited_inline)" in vals
    assert "requester_is_member() and (not _member_research)" in P.assigned_values(t, "_awaited_inline")
    assert P.ifs(t, "_awaited_inline")


async def test_the_vision_judge_skips_quietly_on_a_park_that_ran_out(caplog, tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
    class Deferring(_Stub):
        async def chat_completion(self, payload, **kw):
            raise L.BackgroundDeferred("x")
    v = V.Verifier(llm_client=Deferring(1))
    with caplog.at_level(logging.INFO):
        out = await v._call_llm_vision("is the bug visible?", [str(img)])
    assert not [r for r in caplog.records if "skipped" not in r.getMessage() and r.levelno >= logging.INFO and "Visual" in r.getMessage()]
    assert out == {}
    assert any("Visual verifier call skipped" in r.getMessage() for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ── r2 review fixes ────────────────────────────────────────────────────

@pytest.mark.parametrize("label,bg,early", [
    ("deferred-verdict", True, True), ("stream", True, True),
    ("stream", False, False),                         # a member's / an owner's own turn: the 180 s tick
    ("unlabelled", False, False),
    ("greeting fast path", False, False),             # §4RP-c: no label arms it any more
])
def test_only_calls_nobody_waits_for_arm_the_early_rewarm(label, bg, early):
    c = SimpleNamespace(foreground_requests=0, foreground_tasks=0)
    L.note_head_reference(c, _head())
    L.note_main_slot_prompt(c, _head(tools=None, system="member prompt"), label, background=bg)
    c._last_main_call_end = time.monotonic() - 60
    assert c._head_displaced is True
    assert L.head_rewarm_due(c, 10.0) is early
    L.note_main_slot_prompt(c, _head(), "", background=bg)
    assert c._head_displaced is False and c._head_displaced_early is False


async def test_a_stream_passes_its_priority_to_the_mark():
    c = _bare_client()
    L.note_head_reference(c, _head())

    async def do(payload, use_coding=False):
        yield b"data: {}"
    c._do_stream_chat_completion = do
    async for _ in c.stream_chat_completion(_head(tools=None)):
        pass
    assert c._head_displaced is True and c._head_displaced_early is False
    async for _ in c.stream_chat_completion(_head(tools=None), is_background=True):
        pass
    assert c._head_displaced_early is True


async def test_not_due_while_a_deferred_verdict_holds_the_lane():
    c = _due_client()
    sem = asyncio.Semaphore(1)
    c._deferred_verdict_sem_obj = sem
    assert L.head_rewarm_due(c, 10.0)
    await sem.acquire()
    assert not L.head_rewarm_due(c, 10.0)
    sem.release()


async def test_a_redisplaced_rewarm_is_refused_while_the_lane_is_held():
    c = _bare_client()
    L.note_head_reference(c, _head())
    c._head_displaced = True
    await c._deferred_verdict_sem().acquire()

    async def do(payload, *a, **k):
        raise AssertionError("must not run")
    c._do_chat_completion = do
    with pytest.raises(L.BackgroundDeferred):
        await c.chat_completion(_head(), is_background=True, task_label="main-prefix-redisplaced")


async def test_the_lane_wait_is_bounded_and_never_leaks_a_permit(monkeypatch):
    c = _bare_client()
    monkeypatch.setattr(L, "DEFERRED_LANE_WAIT_S", 0.05)
    lines = []
    monkeypatch.setattr(L, "pretty_log", lambda t, m, **kw: lines.append((m, kw.get("level", "INFO"))))
    sem = c._deferred_verdict_sem()
    await sem.acquire()                                   # another verdict holds it

    async def do(payload, *a, **k):
        return {"_ghost_leg": {"served_by": "main"}}
    c._do_chat_completion = do
    with pytest.raises(L.LaneDeferred):
        await c.chat_completion(_head(tools=None), is_background=True, task_label="deferred-verdict")
    assert lines and lines[-1][1] == "INFO" and "gave up" in lines[-1][0]
    sem.release()
    await asyncio.sleep(0)
    assert not sem.locked()                               # the abandoned acquire gave nothing away
    await c.chat_completion(_head(tools=None), is_background=True, task_label="deferred-verdict")
    assert not sem.locked()


async def test_a_cancelled_lane_wait_releases_nothing_it_does_not_hold():
    c = _bare_client()
    sem = c._deferred_verdict_sem()
    await sem.acquire()

    async def do(payload, *a, **k):
        return {"_ghost_leg": {"served_by": "main"}}
    c._do_chat_completion = do
    t = asyncio.ensure_future(c.chat_completion(_head(tools=None), is_background=True, task_label="deferred-verdict"))
    await asyncio.sleep(0.01)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    sem.release()
    assert not sem.locked()
    assert sem._value == 1


async def test_after_one_give_up_the_verdicts_later_main_calls_stop_at_once():
    calls = []

    class Deferring(_Stub):
        async def chat_completion(self, payload, **kw):
            calls.append(1)
            raise L.BackgroundDeferred("x")
    v = V.Verifier(llm_client=Deferring(1))
    mark = {"nova_busy": False, "internal": False}
    tok = V.deferred_verdict_context.set(mark)
    try:
        r1, r2 = {}, {}
        assert await v._call_llm("a", force_main=True, route_out=r1) == {}
        assert await v._call_llm("b", force_main=True, route_out=r2) == {}
    finally:
        V.deferred_verdict_context.reset(tok)
    assert calls == [1] and mark["main_busy"] is True and r2["route"] == "failed"
    calls.clear()
    assert await v._call_llm("c", force_main=True) == {}  # outside the verdict: nothing remembered
    assert calls == [1]


async def test_the_periodic_tick_names_a_known_displacement():
    llm = _due_client()
    llm._head_displaced_early = False                   # a member's turn: no early re-warm
    a = _agent(llm)
    slept = []

    async def sleep(s):
        slept.append(s)
    assert await a._rewarm_wait(20.0, sleep) == "deferred-verdict"
    assert sum(slept) == 20.0


async def test_an_idle_tick_never_turns_the_early_rewarm_off():
    """r3 MAJOR: the idle "nothing ran" skip disabled the 5 s polling for good."""
    from ghost_agent.core.agent import GhostAgent
    llm = SimpleNamespace(foreground_requests=0, foreground_tasks=0)
    L.note_head_reference(llm, _head())
    a = _agent(llm)
    warmed, slept = [], []

    async def warm(**kw):
        warmed.append(kw.get("displaced_by", ""))
        L.note_main_slot_prompt(llm, _head(), "main-prefix-rewarm", background=True)

    async def sleep(s):
        slept.append(s)
        if sum(slept) == 180.0 + 5.0:                     # after an idle tick: a verdict displaces the head
            L.note_main_slot_prompt(llm, _head(tools=None), "deferred-verdict", background=True)
            llm._last_main_call_end = time.monotonic() - 60
        if len(slept) > 80:
            raise asyncio.CancelledError()

    async def metrics():
        return 0, 100                                    # never moves
    a.warm_up_main_prefix = warm
    with pytest.raises(asyncio.CancelledError):
        await a.rewarm_main_prefix_loop(180, sleep=sleep, metrics=metrics)
    assert warmed[:2] == ["", "deferred-verdict"]        # the first tick, then the displacement…
    assert set(slept) == {5.0}                           # …each found within a 5 s poll


async def test_a_refusal_for_a_held_lane_is_not_a_failure_and_keeps_last_total():
    llm = _due_client()
    a = _agent(llm)
    sem = asyncio.Semaphore(1)
    llm._deferred_verdict_sem_obj = sem
    totals, slept = [], []

    async def warm(**kw):
        await sem.acquire()                              # a verdict took the lane: refused
    calls = {"n": 0}

    async def metrics():
        calls["n"] += 1
        totals.append(calls["n"])
        return 0, calls["n"]

    async def sleep(s):
        slept.append(s)
        if len(slept) > 3:
            raise asyncio.CancelledError()
    a.warm_up_main_prefix = warm
    with pytest.raises(asyncio.CancelledError):
        await a.rewarm_main_prefix_loop(180, sleep=sleep, metrics=metrics)
    assert 180.0 not in slept                            # still on the 5 s polls
    assert calls["n"] == 1                               # no post-warm read: last_total untouched


async def test_a_parked_main_call_stamps_its_cause(monkeypatch):
    lines, rows = [], []
    from ghost_agent.utils import logging as UL
    monkeypatch.setattr(UL, "pretty_log", lambda t, m, **kw: lines.append((m, kw.get("level", "INFO"))))
    monkeypatch.setattr(V, "record_escalation", lambda **kw: rows.append(kw), raising=False)

    class Deferring(_Stub):
        async def chat_completion(self, payload, **kw):
            raise L.BackgroundDeferred("x")
    v = V.Verifier(llm_client=Deferring(1))
    mark = {"nova_busy": False, "internal": False, "main_busy": True}
    tok = V.deferred_verdict_context.set(mark)
    try:
        assert await v._call_llm("judge", force_main=True) == {}
    finally:
        V.deferred_verdict_context.reset(tok)
    assert v._last_main_call == {"finish_reason": "main_busy", "content_chars": 0}


def test_the_refute_escalation_reads_main_busy_as_expected():
    from ghost_agent.core.verifier import Verifier
    t = P.tree(Verifier)
    assert any(P.U(n.value) == "_ro_l.get('finish_reason') in ('main_busy', 'lane_busy')" for n in P.assigns(t, "_parked"))
    lvl = [P.U(P.call_kw(c, "level")) for c in P.calls(t, "pretty_log")]
    assert "'INFO' if _parked else 'WARNING'" in lvl


async def test_a_permit_won_as_the_lane_wait_times_out_is_handed_back(monkeypatch):
    """The acquire completes in the same step the wait gives up: nobody owns
    that permit, so it must go back or the lane stays shut for good."""
    c = _bare_client()
    sem = c._deferred_verdict_sem()
    real_wait = asyncio.wait

    async def wait_then_time_out(fs, timeout=None, **kw):
        await real_wait(fs)                               # the acquire wins…
        return set(), set(fs)                             # …but the wait reports a timeout
    monkeypatch.setattr(asyncio, "wait", wait_then_time_out)
    monkeypatch.setattr(L, "pretty_log", lambda *a, **k: None)
    with pytest.raises(L.BackgroundDeferred):
        await c._take_deferred_verdict_lane(None)
    monkeypatch.setattr(asyncio, "wait", real_wait)
    await asyncio.sleep(0)
    assert not sem.locked()


async def test_a_vision_pool_spelled_with_v1_is_still_the_main_box(monkeypatch):
    """r3 minor: the park and the mark must agree on what the main box is."""
    c = _bare_client()
    c.vision_clients = [{"url": "http://127.0.0.1:8088/v1"}]
    c.foreground_requests = 1
    ran = []

    async def do(payload, *a, **k):
        ran.append(1)
        return {"_ghost_leg": {"served_by": "vision"}}
    c._do_chat_completion = do
    real = asyncio.sleep

    async def fast(_s):
        await real(0)
    monkeypatch.setattr(asyncio, "sleep", fast)
    t = asyncio.ensure_future(c.chat_completion(_head(tools=None), use_vision=True, is_background=True,
                                                task_label="deferred-verdict"))
    for _ in range(30):
        await real(0)
    assert not ran                                        # parked like any main-box call
    c.foreground_requests = 0
    await t
    assert ran



# ── r4 review fixes ────────────────────────────────────────────────────

async def test_a_rewarm_displaced_again_at_once_is_not_a_node_failure():
    """r4 MAJOR: the re-warm landed, a queued background call displaced the
    head again — the loop read that as a failure and slept 180 s."""
    llm = _due_client()
    a = _agent(llm)
    slept = []

    async def warm(**kw):
        L.note_main_slot_prompt(llm, _head(), "main-prefix-redisplaced", background=True)   # landed
        L.note_main_slot_prompt(llm, _head(tools=None), "memory extract", background=True)  # displaced again
        llm._last_main_call_end = time.monotonic() - 60

    async def sleep(s):
        slept.append(s)
        if len(slept) > 2:
            raise asyncio.CancelledError()

    async def metrics():
        return 0, len(slept)
    a.warm_up_main_prefix = warm
    with pytest.raises(asyncio.CancelledError):
        await a.rewarm_main_prefix_loop(180, sleep=sleep, metrics=metrics)
    assert slept == [5.0, 5.0, 5.0]


async def test_a_slot_busy_for_a_whole_period_stops_the_5s_polls(monkeypatch):
    from ghost_agent.core import agent as A
    llm = _due_client()
    a = _agent(llm)
    slept = []
    clock = {"t": 1000.0}
    monkeypatch.setattr(A.time, "monotonic", lambda: clock["t"])
    llm._last_main_call_end = 0.0

    async def warm(**kw):
        raise AssertionError("never while busy")

    async def sleep(s):
        slept.append(s)
        clock["t"] += s
        if len(slept) > 40:
            raise asyncio.CancelledError()

    async def metrics():
        return 1, len(slept)
    a.warm_up_main_prefix = warm
    with pytest.raises(asyncio.CancelledError):
        await a.rewarm_main_prefix_loop(30, sleep=sleep, metrics=metrics)
    assert 30.0 in slept and slept.index(30.0) <= 8      # polled ~a period, then the period


async def test_a_lane_give_up_is_named_as_such(caplog):
    class Lane(_Stub):
        async def chat_completion(self, payload, **kw):
            raise L.LaneDeferred("x")
    v = V.Verifier(llm_client=Lane(0))
    tok = V.deferred_verdict_context.set({"nova_busy": False})
    try:
        with caplog.at_level(logging.INFO):
            assert await v._call_llm("judge", force_main=True) == {}
    finally:
        V.deferred_verdict_context.reset(tok)
    assert v._last_main_call["finish_reason"] == "lane_busy"
    assert any("behind another verdict" in r.getMessage() and r.levelno == logging.INFO for r in caplog.records)
    assert issubclass(L.LaneDeferred, L.BackgroundDeferred)



async def test_the_first_tick_after_boot_names_a_displacement_seen_while_it_slept():
    """Live 2026-10-14: the loop's first sleep began before the boot warm-up
    registered the head, so a greeting inside it re-warmed as "churn"."""
    llm = SimpleNamespace(foreground_requests=0)
    a = _agent(llm)

    async def sleep(s):
        L.note_head_reference(llm, _head())
        L.note_main_slot_prompt(llm, _head(tools=None), "greeting fast path")
    assert await a._rewarm_wait(180.0, sleep) == "greeting fast path"
    llm2 = SimpleNamespace(foreground_requests=0)

    async def quiet(s):
        L.note_head_reference(llm2, _head())
    assert await _agent(llm2)._rewarm_wait(180.0, quiet) == ""
