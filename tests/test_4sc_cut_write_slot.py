"""§4SC (2026-10-29): the cut-write continuation keeps the conversation.

Req 72c2f05d: Turn 1's write was cut at 16k; the §4OB continuation ran a
fresh 1.2k-token prompt on the main server; llama could not park the
51,604-token state (3.4 GB > --cache-ram 2 GB) and dropped it; nothing had
saved the conversation (the continuation took the MAIN dispatch branch, so
the §4RW-g side hook never ran) and the next turn's restore was "already
decided" for the request — Turn 2 re-prefilled 41,742 tokens in 282 s.
Now the continuation is a side call: saved first, restored by the request's
next head call. Every test names the world it fails in."""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import head_saves as H
from ghost_agent.core import llm as L
from ghost_agent.core import write_continuation as W
from ghost_agent.utils.logging import request_id_context
from tests.test_4ro_head_saves import _store, _switch_env, _turn, _warm_with_file, warm  # noqa: F401
from tests.test_4rv_conv_keep import _conv_env, conv, cwarm  # noqa: F401
from tests.test_4rwg_side_call import LOOK as LOOK_MAIN, _client, _held, _look_ran, rid  # noqa: F401

CONT = {"messages": [{"role": "system", "content": "Continue the file."},
                     {"role": "user", "content": "head…tail"}], "stream": False, "max_tokens": 4000}


async def test_the_continuation_call_is_dispatched_as_a_side_call():
    """Fails in the world where `_call_watching_stop` sends a plain main call."""
    seen = {}

    class _Llm:
        async def chat_completion(self, payload, **kw):
            seen.update(kw)
            return {"choices": [{"message": {"content": W.END_MARKER}, "finish_reason": "stop"}]}
    out = await W._call_watching_stop(_Llm(), CONT, None)
    assert out["choices"][0]["finish_reason"] == "stop"
    assert seen.get("side_main") is True and seen.get("task_label") == "write-continuation"


@pytest.mark.parametrize("kw,want", [
    ({"side_main": True}, ["side"]),                       # the continuation: saved first
    ({"side_main": True, "off_main_only": True}, []),      # off main: nothing in the slot to save
    ({}, ["main"]),                                        # an ordinary main call: unchanged
])
async def test_the_client_runs_the_side_hook_for_a_side_main_call(kw, want):
    """Fails where `side_main` is ignored (the main hook runs, no save)."""
    c, seen = _client("http://10.0.0.9:8088")
    await c.chat_completion(CONT, **kw)
    assert seen == want


async def test_the_client_names_the_continuation_for_the_hook_and_clears_it():
    c, _seen = _client("http://10.0.0.9:8088")
    names = []

    async def side(payload):
        names.append(L.side_call_what())
    c.before_side_main_dispatch = side
    await c.chat_completion(CONT, side_main=True)
    assert names == ["this request's cut-write continuation"] and L.side_call_what() == ""


async def test_the_turn_after_a_continuation_gets_its_conversation_back(cwarm, monkeypatch, rid):
    """The live failure: fails where the continuation saves nothing (the
    request's next head call finds its restore already decided → llama
    re-processes from token 0)."""
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch)
    tok = L._side_call_what_ctx.set("this request's cut-write continuation")
    try:
        await agent._side_call_before(CONT)
    finally:
        L._side_call_what_ctx.reset(tok)
    ckey = H.conv_key("id", "sess-a")
    assert [k for k, _n, _a in st["conv_saves"]] == [ckey]
    assert any("cut-write continuation" in m for m in st["lines"]), st["lines"]
    _look_ran(llm)                                   # what note_main_slot_prompt leaves after it
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == [ckey]
    assert llm._slot_session == "sess-a" and llm._head_displaced is False


async def test_a_look_still_says_look(cwarm, monkeypatch, rid):
    """Control: without the client's name the hook's wording is the look's."""
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch)
    from tests.test_4rwg_side_call import LOOK
    await agent._side_call_before(LOOK)
    assert any("picture look" in m for m in st["lines"]) and not any("continuation" in m for m in st["lines"])


# ── §4SC-b (review r1): the same hole on the other foreground calls ──────────
# Every call of a request that runs on the main server without the head
# displaces its conversation exactly as the continuation did. Each pin fails
# in the world where that call is still a plain main call (or off-head).

from unittest.mock import AsyncMock, patch                          # noqa: E402

from ghost_agent.tools import search as S                           # noqa: E402
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome    # noqa: E402
from tests.test_truncation_continuation import _stream_factory, mock_agent  # noqa: F401,E402


class _Recorder:
    """A client that records every call's kwargs."""
    def __init__(self, content="fastest sorting algorithm", raise_=None):
        self.calls, self.content, self.raise_ = [], content, raise_
        self.foreground_requests = 1          # a live request: not an idle invocation

    async def chat_completion(self, payload, **kw):
        self.calls.append((payload, kw))
        if self.raise_ is not None:
            raise self.raise_
        return {"choices": [{"message": {"content": self.content}, "finish_reason": "stop"}]}


async def test_neutralize_query_runs_off_main():
    """MAJOR-1: fails where `neutralize_query` sends a plain main call (it runs
    inline on every anonymous deep_research/darkweb_search)."""
    from ghost_agent.utils.stylometry import neutralize_query
    llm = _Recorder()
    out = await neutralize_query("Please find me the fastest sorting algorithm!", llm_client=llm, model="m")
    assert out == "fastest sorting algorithm" and len(llm.calls) == 1
    kw = llm.calls[0][1]
    assert kw.get("use_worker") is True and kw.get("off_main_only") is True
    # §4SC-c: bounded like `route()`, healthy nodes only
    from ghost_agent.utils import stylometry as ST
    assert ST._NEUTRALIZE_TIMEOUT_S == L._ROUTE_TIMEOUT_S == 12.0
    assert kw.get("require_healthy") is True
    # §4SC-d: the POST bound stays the route's; the WHOLE budget (permit gate
    # + POST) is 6 s — a saturated Nova is not a breaker fault, so the gate
    # is what bounds it (9 s at a 12 s budget, ≤ 3 s at 6 s)
    assert kw.get("timeout") == 12.0 and kw.get("total_budget") == ST._NEUTRALIZE_BUDGET_S == 6.0


async def test_neutralize_query_without_a_worker_is_the_lexical_scrub():
    """The fallback: no worker → the client raises → the lexical scrub, never
    the main model (fails where the raise escapes or the call retries main)."""
    from ghost_agent.utils.stylometry import neutralize_query, scrub_query
    q = "Could you please look up CRDT conflict resolution?"
    llm = _Recorder(raise_=L.OffMainNodeUnavailable("no worker"))
    assert await neutralize_query(q, llm_client=llm) == scrub_query(q)
    assert len(llm.calls) == 1 and llm.calls[0][1].get("off_main_only") is True


async def test_fact_check_verify_is_a_side_call_on_main():
    """MAJOR-2: fails where the verify call is a plain main call."""
    llm = _Recorder(content="TRUE")
    out = await S.tool_fact_check(
        query="x", llm_client=llm,
        deep_research_callable=AsyncMock(return_value=ToolOutcome.ok(
            "--- DEEP RESEARCH RESULT ---\n### SOURCE: https://a\nfacts\n")))
    assert isinstance(out, ToolOutcome) and out.status is OutcomeStatus.OK
    kw = llm.calls[-1][1]
    assert kw.get("side_main") is True and not kw.get("use_worker") and not kw.get("off_main_only")
    assert kw.get("timeout") == S._FACTCHECK_VERIFY_S                  # §4PD's bound kept


@pytest.mark.parametrize("fg,node,want", [
    (1, "main", True),        # a request's own probe on main: side call
    (0, "main", False),       # an idle (background) probe: parks, not a side call
    (1, "worker", False),     # the worker leg never touches main
])
async def test_probe_model_on_main_is_a_side_call(tmp_path, fg, node, want):
    """MAJOR-4a: fails where a foreground probe on main is a plain main call."""
    from ghost_agent.tools.probe_model import tool_probe_model
    from tests.test_4qa_probe_quotes import _ctx
    llm = _Recorder(content="ANSWER")
    llm.foreground_requests = fg
    await tool_probe_model(_ctx(tmp_path, llm), prompts=["a"], node=node)
    assert llm.calls, "no call recorded"
    kw = llm.calls[0][1]
    assert kw.get("side_main") is want and kw.get("is_background") is (fg == 0)


@pytest.mark.parametrize("bg", [False, True])
async def test_research_summary_inline_is_a_side_call(bg):
    """MAJOR-4b: fails where the inline research summary is a plain main call."""
    from ghost_agent.core.project_research import _llm_complete
    llm = _Recorder(content="brief")
    ctx = SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))
    assert await _llm_complete(ctx, "sum", max_tokens=10, is_background=bg) == "brief"
    kw = llm.calls[0][1]
    assert kw.get("side_main") is (not bg) and kw.get("is_background") is bg


_TOOLS = [{"type": "function", "function": {"name": "check_weather", "description": "w",
                                            "parameters": {"type": "object", "properties": {}}}}]


async def test_the_truncation_continuation_opens_with_the_head(mock_agent):
    """MAJOR-3: fails where the §4MD continuation pops `tools` — a non-head
    call (`opens_with_head` hashes the tool list) that displaced the
    conversation AND re-prefilled the whole prompt itself."""
    streamed = []

    def _stream_recording(*args, **kwargs):
        streamed.append(args[0] if args else kwargs.get("payload"))
        return _stream_factory(["The honest answer is that I was functionally"], "length")(*args, **kwargs)
    mock_agent.context.args.native_tools = True
    mock_agent.context.llm_client.stream_chat_completion = _stream_recording
    cont = AsyncMock(return_value={"choices": [{"message": {"content": " self-aware."},
                                                "finish_reason": "stop"}]})
    mock_agent.context.llm_client.chat_completion = cont
    body = {"messages": [{"role": "user", "content": "were you self-aware?"}], "stream": False}
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=list(_TOOLS)), \
            patch("ghost_agent.tools.registry.get_active_tool_definitions", return_value=list(_TOOLS)):
        final, _created, _req = await mock_agent.handle_chat(body, background_tasks=MagicMock(),
                                                             request_id="req-len-tools")
    assert cont.await_count == 1 and "self-aware" in final
    head = streamed[0]
    assert head.get("tools"), "the world this pin needs: a head with tools"
    sent = cont.await_args.args[0]
    assert sent.get("tools") == head["tools"] and sent.get("tool_choice") == "none"
    assert L.opens_with_head(sent, L.head_shape(head))
    assert not cont.await_args.kwargs.get("side_main")      # it IS a head call: the main path


async def test_the_continuation_without_tools_still_carries_none(mock_agent):
    """Control (the §4MD pin's world): no tools on the head → none on the
    continuation, no stray tool_choice."""
    mock_agent.context.llm_client.stream_chat_completion = _stream_factory(["cut"], "length")
    cont = AsyncMock(return_value={"choices": [{"message": {"content": " end."}, "finish_reason": "stop"}]})
    mock_agent.context.llm_client.chat_completion = cont
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        await mock_agent.handle_chat({"messages": [{"role": "user", "content": "q"}], "stream": False},
                                     background_tasks=MagicMock(), request_id="req-len-plain")
    sent = cont.await_args.args[0]
    assert "tools" not in sent and "tool_choice" not in sent and "parallel_tool_calls" not in sent


async def test_a_given_window_is_the_saves_bound(cwarm, monkeypatch, rid):
    """MINOR-2: fails where `_conv_save_locked` caps a caller's window at the
    state's estimate (the side hook sized the window already; a 60 s cap on a
    118 s window timed out a multi-GB save)."""
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch)
    timeouts = []
    _save = H.save

    def save(base, d, key, n, ident="", kind=H.OWNER, timeout=H.SAVE_TIMEOUT_S, at_least=False, extra=None):
        timeouts.append(timeout)
        return _save(base, d, key, n, ident, kind, timeout, at_least, extra)
    monkeypatch.setattr(H, "save", save)
    est = H.save_timeout_s(st["slot"])
    assert est < 100, "the world this pin needs: a window wider than the estimate"
    assert await agent._conv_save(llm, "x", time.monotonic() + 118.0) is True
    assert 100.0 < timeouts[-1] <= 118.0
    # control: no window → the estimate
    llm._conv_saved = None
    assert await agent._conv_save(llm, "x") is True
    assert timeouts[-1] == pytest.approx(est, abs=1.0)


@pytest.mark.parametrize("label,want", [
    ("", "this request's cut-write continuation"),
    ("write-continuation", "this request's cut-write continuation"),
    ("fact-check-verify", "this request's fact-check-verify call"),
])
async def test_the_side_call_is_named_by_its_label(label, want):
    """§4SC-b: a fact-check verify saved 'before this request's cut-write
    continuation' is a false log line."""
    c, _seen = _client("http://10.0.0.9:8088")
    names = []

    async def side(payload):
        names.append(L.side_call_what())
    c.before_side_main_dispatch = side
    await c.chat_completion(CONT, side_main=True, task_label=label)
    assert names == [want]


async def test_the_side_calls_name_is_per_task_not_per_client():
    """NIT-1: fails where the name is a client attribute (two requests' side
    calls on the one client read each other's name across the hook's await)."""
    c, _seen = _client("http://10.0.0.9:8088")
    seen = {}

    async def side(payload):
        mine = L.side_call_what()
        await asyncio.sleep(0.01)                  # the other task sets its own meanwhile
        seen[mine] = L.side_call_what()
    c.before_side_main_dispatch = side
    await asyncio.gather(c.chat_completion(CONT, side_main=True, task_label="fact-check-verify"),
                         c.chat_completion(CONT, side_main=True, task_label="probe-model"))
    assert seen == {"this request's fact-check-verify call": "this request's fact-check-verify call",
                    "this request's probe-model call": "this request's probe-model call"}
    assert L.side_call_what() == ""


# ── §4SC-c (review r2) ───────────────────────────────────────────────────────

from ghost_agent.core import agent as A                              # noqa: E402

Q_ANON = "Could you please look up CRDT conflict resolution?"


async def _hung_server():
    """A worker that accepts and never answers (the reviewer's repro_a)."""
    async def h(_r, _w):
        await asyncio.sleep(3600)
    srv = await asyncio.start_server(h, "127.0.0.1", 0)
    return srv, srv.sockets[0].getsockname()[1]


async def test_a_hung_worker_costs_an_anonymous_search_the_route_bound_not_20_minutes():
    """MAJOR (r2): fails where `neutralize_query` passes no timeout — the
    worker client's 1200 s default; through the REAL client against a socket
    that never answers, the lexical scrub comes back within the 12 s bound."""
    from ghost_agent.utils.stylometry import neutralize_query, scrub_query
    srv, port = await _hung_server()
    c = L.LLMClient("http://127.0.0.1:9/", worker_nodes=[{"url": f"http://127.0.0.1:{port}", "model": "gemma"}])
    node = c.worker_clients[0]
    real_post, seen = node["client"].post, {}

    async def post(*a, **k):
        seen["timeout"] = k.get("timeout")
        return await real_post(*a, **k)
    node["client"].post = post
    try:
        t0 = time.monotonic()
        out = await asyncio.wait_for(neutralize_query(Q_ANON, llm_client=c, model="m"), 40.0)
        dt = time.monotonic() - t0
    finally:
        srv.close()
        await c.close()
    assert out == scrub_query(Q_ANON)
    assert dt < 20.0, f"took {dt:.1f}s"
    # §4SC-d: with a 6 s budget the 5 s /props probe of a hung node leaves no
    # room for a POST at all (node not contacted); if one does go out, it
    # carries a budget, never the client's 1200 s default
    assert "timeout" not in seen or (seen["timeout"] is not None and float(seen["timeout"]) <= 12.0)


async def test_without_a_worker_pool_the_scrub_is_lexical_and_main_is_never_called():
    """MINOR (r2): fails where the client's "requested but unconfigured pool"
    arm runs the call on MAIN with no hook."""
    from ghost_agent.utils.stylometry import neutralize_query, scrub_query
    c = L.LLMClient("http://127.0.0.1:9/")
    assert c.worker_clients == []
    posts, hooks = [], []

    async def post(*a, **k):
        posts.append(k.get("timeout"))
        raise AssertionError("main was called")

    async def side(p):
        hooks.append("side")

    async def mainh(p):
        hooks.append("main")
    c.http_client.post, c.before_side_main_dispatch, c.before_main_dispatch = post, side, mainh
    try:
        out = await neutralize_query(Q_ANON, llm_client=c, model="m")
    finally:
        await c.close()
    assert out == scrub_query(Q_ANON) and posts == [] and hooks == []


async def test_a_continuation_that_answers_with_a_call_leaves_the_cut_answer_cut(mock_agent, monkeypatch):
    """MINOR (r2, repro_c): fails where the continuation's `<tool_call>` text
    is appended — the schemas are in view now, and a cut prose answer became
    a tool turn. The reply is the cut text + the note; no tool turn."""
    streams, lines = [], []

    def _rec(*a, **k):
        streams.append(1)
        return _stream_factory(["The weather in Athens is"], "length")(*a, **k)
    _orig = A.pretty_log

    def _log(title, msg="", *a, **k):
        lines.append(f"{title}: {msg}")
        return _orig(title, msg, *a, **k)
    monkeypatch.setattr(A, "pretty_log", _log)
    mock_agent.context.args.native_tools = True
    mock_agent.context.llm_client.stream_chat_completion = _rec
    cont = AsyncMock(return_value={"choices": [{"message": {"content":
        ' currently unknown to me; let me check.\n<tool_call>\n{"name": "check_weather", '
        '"arguments": {"city": "Athens"}}\n</tool_call>', "tool_calls": []}, "finish_reason": "stop"}]})
    mock_agent.context.llm_client.chat_completion = cont
    body = {"messages": [{"role": "user", "content": "weather in athens?"}], "stream": False}
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=list(_TOOLS)), \
            patch("ghost_agent.tools.registry.get_active_tool_definitions", return_value=list(_TOOLS)):
        final, _c, _r = await mock_agent.handle_chat(body, background_tasks=MagicMock(), request_id="req-xml")
    assert cont.await_count == 1 and len(streams) == 1
    assert final.startswith("The weather in Athens is") and A.TRUNCATED_ANSWER_NOTE in final
    assert "<tool_call" not in final and "let me check" not in final
    assert not any("Extracting XML tool call" in m for m in lines), lines


async def test_the_fact_check_wait_covers_the_side_hooks_save(monkeypatch):
    """MINOR (r2): fails where the outer wait is `_FACTCHECK_VERIFY_S + 5` —
    it wraps the side hook's save (up to BEFORE_MAIN_DISPATCH_S) too."""
    from ghost_agent.utils import aio as AIO
    waits, _orig = [], AIO.wait_for

    async def wait_for(coro, timeout, *a, **k):
        waits.append(timeout)
        return await _orig(coro, timeout, *a, **k)
    monkeypatch.setattr(AIO, "wait_for", wait_for)
    llm = _Recorder(content="TRUE")
    out = await S.tool_fact_check(
        query="x", llm_client=llm,
        deep_research_callable=AsyncMock(return_value=ToolOutcome.ok(
            "--- DEEP RESEARCH RESULT ---\n### SOURCE: https://a\nfacts\n")))
    assert out.status is OutcomeStatus.OK
    assert waits[-1] == S._FACTCHECK_VERIFY_S + L.BEFORE_MAIN_DISPATCH_S


async def test_the_restore_line_names_the_side_call(cwarm, monkeypatch, rid):
    """NIT (r2): fails where the restore line says "picture look" after a
    fact-check verify (or the continuation)."""
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch, slot=28213 + 100)     # too short to save: the head comes back
    tok = L._side_call_what_ctx.set("this request's fact-check-verify call")
    try:
        await agent._side_call_before(CONT)
    finally:
        L._side_call_what_ctx.reset(tok)
    _look_ran(llm)
    await agent._slot_switch_before(_turn(st))
    assert len(st["restores"]) == 1
    assert any("restored the head before this request after its own fact-check-verify call" in m
               for m in st["lines"]), st["lines"]
    assert not any("picture look" in m for m in st["lines"])


async def test_a_tool_less_continuation_is_a_side_call(mock_agent):
    """NIT (r2): fails where a continuation on a head without tools is a plain
    main call. The kwarg is the pin: on a tool-less WARMED head the continuation
    is a prefix extension (`opens_with_head` True — r3), so the side hook
    returns before saving; `side_main` costs nothing there and saves when the
    head in the slot carries tools the continuation's payload does not."""
    mock_agent.context.llm_client.stream_chat_completion = _stream_factory(["cut"], "length")
    cont = AsyncMock(return_value={"choices": [{"message": {"content": " end."}, "finish_reason": "stop"}]})
    mock_agent.context.llm_client.chat_completion = cont
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        await mock_agent.handle_chat({"messages": [{"role": "user", "content": "q"}], "stream": False},
                                     background_tasks=MagicMock(), request_id="req-len-plain-side")
    assert "tools" not in cont.await_args.args[0]
    assert cont.await_args.kwargs.get("side_main") is True


# ── §4SC-d (review r3) ───────────────────────────────────────────────────────

_PROPS = json.dumps({"total_slots": 1}).encode()
_ANSWER = json.dumps({"choices": [{"message": {"content": "CRDT conflict resolution"}}]}).encode()


async def _worker_server():
    """A worker that answers /props (1 slot) and every POST at once."""
    async def h(r, w):
        try:
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = await r.read(65536)
                if not chunk:
                    return
                head += chunk
            body = _PROPS if b"/props" in head.split(b"\r\n", 1)[0] else _ANSWER
            w.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n"
                    b"Connection: close\r\n\r\n" % len(body) + body)
            await w.drain()
            w.close()
        except Exception:  # noqa: BLE001
            pass
    srv = await asyncio.start_server(h, "127.0.0.1", 0)
    return srv, srv.sockets[0].getsockname()[1]


def _worker_client(url):
    c = L.LLMClient("http://127.0.0.1:9/", worker_nodes=[{"url": url, "model": "gemma"}])
    node, posts = c.worker_clients[0], []
    real_post = node["client"].post

    async def post(*a, **k):
        posts.append(k.get("timeout"))
        return await real_post(*a, **k)
    node["client"].post = post
    return c, posts


async def test_a_saturated_worker_costs_the_scrub_its_permit_gate_not_the_route_bound():
    """MINOR-1 (r3): fails where the budget is the route's 12 s — the permit
    gate alone then costs 9 s (the reviewer's "0 free permits" shape); at a
    6 s budget it is ≤ 3 s, no POST, the lexical scrub."""
    from ghost_agent.utils.stylometry import neutralize_query
    srv, port = await _worker_server()
    url = f"http://127.0.0.1:{port}"
    c, posts = _worker_client(url)
    try:
        cap = await c._node_capacity(c.worker_clients[0])
        c._node_slots[url], c._node_slot_built_cap[url] = asyncio.Semaphore(0), cap
        t0 = time.monotonic()
        out = await asyncio.wait_for(neutralize_query(Q_ANON, llm_client=c, model="m"), 30.0)
        dt = time.monotonic() - t0
    finally:
        srv.close()
        await c.close()
    from ghost_agent.utils.stylometry import scrub_query
    assert out == scrub_query(Q_ANON) and posts == []
    assert dt < 4.0, f"gate took {dt:.1f}s"


async def test_an_open_breaker_means_no_call_at_all():
    """MINOR-4 (r3): fails where `require_healthy` is off — a node with its
    breaker open (3 failures) is still POSTed to; here: no POST, < 1 s."""
    from ghost_agent.utils.stylometry import neutralize_query, scrub_query
    srv, port = await _worker_server()
    url = f"http://127.0.0.1:{port}"
    c, posts = _worker_client(url)
    try:
        for _ in range(3):
            c.circuit_breaker.record_failure(url)
        assert not c.circuit_breaker.is_available(url)
        t0 = time.monotonic()
        out = await neutralize_query(Q_ANON, llm_client=c, model="m")
        dt = time.monotonic() - t0
    finally:
        srv.close()
        await c.close()
    assert out == scrub_query(Q_ANON) and posts == [] and dt < 1.0


def _cont_world(mock_agent, first, conts):
    mock_agent.context.args.native_tools = True
    mock_agent.context.llm_client.stream_chat_completion = _stream_factory(first, "length")
    cont = AsyncMock(side_effect=conts)
    mock_agent.context.llm_client.chat_completion = cont
    return cont


async def _run_cont_world(mock_agent, rid):
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=list(_TOOLS)), \
            patch("ghost_agent.tools.registry.get_active_tool_definitions", return_value=list(_TOOLS)):
        final, _c, _r = await mock_agent.handle_chat(
            {"messages": [{"role": "user", "content": "weather?"}], "stream": False},
            background_tasks=MagicMock(), request_id=rid)
    return final


async def test_a_cut_inside_an_opening_tag_is_not_continued(mock_agent):
    """MINOR-2 (r3, the reviewer's d2): fails where the pre-loop gate reads
    "…\n<tool_ca" as prose — the continuation returned "ll>\n{…}</tool_call>"
    and the reply shipped the seam and the call JSON."""
    cont = _cont_world(mock_agent, ["Let me look that up.\n<tool_ca"], [
        {"choices": [{"message": {"content": 'll>\n{"name": "check_weather", "arguments": {"city": "Athens"}}'
                                             '\n</tool_call>'}, "finish_reason": "stop"}]}])
    final = await _run_cont_world(mock_agent, "req-d2")
    assert cont.await_count == 0
    assert "ll>" not in final and "</tool_call" not in final and '"name"' not in final
    # the follow-up: the dangling tag is stripped and the cut is said — never
    # a fragment of markup (fails where "Let me look that up.\n<tool_ca" ships)
    assert "<tool_ca" not in final and "<" not in final
    assert final.startswith("Let me look that up.") and A.TRUNCATED_ANSWER_NOTE in final
    assert final.split(A.TRUNCATED_ANSWER_NOTE)[0].strip() == "Let me look that up."


async def test_a_continuation_that_closes_a_call_leaves_the_cut_answer_cut(mock_agent):
    """MINOR-2 (r3): fails where only the OPENING forms are gated — a
    continuation of '…{"name": …}\n</tool_call>' (no '<tool_call') was appended."""
    cont = _cont_world(mock_agent, ["Let me look that up."], [
        {"choices": [{"message": {"content": '\n{"name": "check_weather", "arguments": {}}\n</tool_call>'},
                      "finish_reason": "stop"}]}])
    final = await _run_cont_world(mock_agent, "req-d3")
    assert cont.await_count == 1
    assert final.startswith("Let me look that up.") and A.TRUNCATED_ANSWER_NOTE in final
    assert "</tool_call" not in final and '"name"' not in final


async def test_an_off_main_only_call_to_a_pool_on_the_main_box_still_saves():
    """MINOR-3 (r3): fails where `off_main_only` skips the side hook — the
    flag governs the fallback, not where the pool lives; a vision call on
    the main box displaces the conversation either way."""
    c, seen = _client("http://127.0.0.1:8088")
    await c.chat_completion(LOOK_MAIN, use_vision=True, off_main_only=True)
    assert seen == ["side"]


# ── §4SC-e (review r4): the tail rule is the tag's own line ──────────────────

@pytest.mark.parametrize("tail,want", [
    ("Let me look that up.\n<tool_ca", True), ("<tool_ca", True), ("…\n<func", True), ("…\n<", True),
    ("…\n<tool_call", True), ("…\n<function", True),
    ("std::vector<int", False), ("x<y", False), ("if a <", False), ("#include <stdio", False),
    ("\n<br", False), ("…\n<Tool_ca", False), ("…\n<tool_calls", False), ("…\n<tool_ca>", False),
])
def test_the_call_tag_tail_is_a_line_anchored_prefix_of_the_templates_tags(tail, want):
    """MAJOR (r4): fails where the tail rule is `<[a-z_]{0,9}$` — any `<` near
    the end of a legitimate cut answer."""
    assert bool(A._CALL_TAG_TAIL.search(tail.rstrip())) is want


@pytest.mark.parametrize("first,cont_text,want", [
    ("Use a vector:\n```cpp\nstd::vector<int", "> v = {1, 2, 3};\n```\nThat is all.",
     "Use a vector:\n```cpp\nstd::vector<int> v = {1, 2, 3};\n```\nThat is all."),
    ("The condition holds when x<y", " and y<z, so x<z.", "The condition holds when x<y and y<z, so x<z."),
    ("The loop runs while i <", " n.", "The loop runs while i < n."),
])
async def test_a_cut_answer_ending_in_a_bare_angle_bracket_is_continued(mock_agent, first, cont_text, want):
    """MAJOR (r4): fails where `std::vector<int` / `x<y` / `i <` is read as a
    cut call tag — not continued, the tail deleted, the note appended."""
    cont = _cont_world(mock_agent, [first], [
        {"choices": [{"message": {"content": cont_text}, "finish_reason": "stop"}]}])
    final = await _run_cont_world(mock_agent, "req-r4-" + str(abs(hash(first)) % 1000))
    assert cont.await_count == 1
    assert A.TRUNCATED_ANSWER_NOTE not in final
    assert final.strip() == want


async def test_a_cut_note_closes_an_open_code_fence_first(mock_agent):
    """NIT (r4): fails where the note lands inside an open fence (renders as
    code) — a cut at "```cpp\nstd::vector<int" whose continuation returns
    nothing."""
    cont = _cont_world(mock_agent, ["Use a vector:\n```cpp\nstd::vector<int"], [
        {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}])
    final = await _run_cont_world(mock_agent, "req-r4-fence")
    assert cont.await_count == 1 and A.TRUNCATED_ANSWER_NOTE in final
    before = final.split(A.TRUNCATED_ANSWER_NOTE)[0]
    assert before.count("```") % 2 == 0 and before.rstrip().endswith("```")
    assert before.rstrip().startswith("Use a vector:\n```cpp\nstd::vector<int")


# ── §4SC-f (review r5) ───────────────────────────────────────────────────────

@pytest.mark.parametrize("tag", ["<tool_call", "<function"])
async def test_a_cut_at_exactly_the_full_tag_name_is_said_not_shipped(mock_agent, tag):
    """MINOR-1 (r5, S3/S3c): fails where the substring exclusion reads the
    tail itself — "…\n<tool_call" (no `>`) excluded its own cut: no note, the
    fragment shipped raw, booked ok."""
    cont = _cont_world(mock_agent, ["Let me look.\n" + tag], [
        {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}])
    final = await _run_cont_world(mock_agent, "req-r5-" + tag.strip("<"))
    assert cont.await_count == 0
    assert A.TRUNCATED_ANSWER_NOTE in final and tag not in final and "<" not in final
    assert final.split(A.TRUNCATED_ANSWER_NOTE)[0].strip() == "Let me look."


async def test_a_tag_tail_with_trailing_blank_lines_still_strips_to_the_prose(mock_agent):
    """Guards the load-bearing rstrip (r5, S2): "…\n<tool_ca\n\n" is the same cut."""
    cont = _cont_world(mock_agent, ["Let me look.\n<tool_ca\n\n"], [
        {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}])
    final = await _run_cont_world(mock_agent, "req-r5-s2")
    assert cont.await_count == 0 and "<" not in final and A.TRUNCATED_ANSWER_NOTE in final
    assert final.split(A.TRUNCATED_ANSWER_NOTE)[0].strip() == "Let me look."


@pytest.mark.parametrize("first,cont_text,want", [
    ("x", "> 5", "x > 5"),                                   # prose: a space
    ("a -", "> b", "a -> b"),                                # an arrow: glued
    ("Use a vector:\n```cpp\nstd::vector<int", "> v;\n```", "Use a vector:\n```cpp\nstd::vector<int> v;\n```"),
])
async def test_a_leading_angle_bracket_is_glued_only_to_an_open_bracket_or_an_arrow(mock_agent, first, cont_text, want):
    """MINOR-2 (r5): fails where `>` is glued unconditionally ("x" + "> 5" →
    "x> 5") or never (the vector seam gets a space)."""
    cont = _cont_world(mock_agent, [first], [
        {"choices": [{"message": {"content": cont_text}, "finish_reason": "stop"}]}])
    final = await _run_cont_world(mock_agent, "req-r5-glue-" + str(abs(hash(first + cont_text)) % 1000))
    assert cont.await_count == 1 and A.TRUNCATED_ANSWER_NOTE not in final
    assert final.strip() == want


@pytest.mark.parametrize("first", ["```html\n<div>\n<t", "Run:\n```\n<tool_ca"])
async def test_a_cut_inside_an_open_code_fence_is_code_and_is_continued(mock_agent, first):
    """MINOR-3 (r5, S5/S8): fails where a `<t…` line inside an open fence is
    read as a call tag — not continued, the code's tail deleted."""
    cont = _cont_world(mock_agent, [first], [
        {"choices": [{"message": {"content": "able>\n```\nDone."}, "finish_reason": "stop"}]}])
    final = await _run_cont_world(mock_agent, "req-r5-fence-" + str(abs(hash(first)) % 1000))
    assert cont.await_count == 1 and A.TRUNCATED_ANSWER_NOTE not in final
    assert final.strip().startswith(first) and final.strip().endswith("```\nDone.")


# ── §4SC-g (review r6) ───────────────────────────────────────────────────────

async def test_a_continuation_round_cut_inside_a_tag_is_said_not_completed(mock_agent):
    """MINOR-1 (r6): fails where the tail rule runs only pre-loop — round 1
    "… and part two.\n<tool_ca" (length) was continued and round 2 completed
    the markup ("…<tool_ca ll>\nignored" shipped, no note)."""
    cont = _cont_world(mock_agent, ["Answer part one"], [
        {"choices": [{"message": {"content": " and part two.\n<tool_ca"}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": "ll>\nignored"}, "finish_reason": "stop"}]}])
    final = await _run_cont_world(mock_agent, "req-r6-cont-cut")
    assert cont.await_count == 1
    assert "<" not in final and "ignored" not in final and A.TRUNCATED_ANSWER_NOTE in final
    assert final.split(A.TRUNCATED_ANSWER_NOTE)[0].strip() == "Answer part one and part two."


async def test_a_dangling_tail_after_a_complete_call_block_never_reaches_the_parser(mock_agent, monkeypatch):
    """MINOR-2 (r6): fails where a matched tail whose rest holds a complete
    call block goes to the parser with the fragment still in the text (it
    reached the pre-tool narration)."""
    seen = []
    _orig = mock_agent._parse_assistant_tool_calls

    def _parse(content, msg):
        seen.append(content)
        return [], content, None                      # record, run no tool
    monkeypatch.setattr(mock_agent, "_parse_assistant_tool_calls", _parse)
    cont = _cont_world(mock_agent, ['<tool_call>\n{"name": "check_weather", "arguments": {}}\n</tool_call>\n'
                                    'Then prose.\n<tool_ca'], [])
    final = await _run_cont_world(mock_agent, "req-r6-dangling")
    assert cont.await_count == 0 and seen, "the parser must have been given the turn"
    # (the complete `<tool_call>` block legitimately stays; the DANGLING tail must not)
    assert all(A._CALL_TAG_TAIL.search(c.rstrip()) is None and c.rstrip().endswith("Then prose.")
               for c in seen), seen
    assert A._CALL_TAG_TAIL.search(final.rstrip()) is None and "<tool_ca\n" not in final


async def test_a_mid_line_fence_mention_is_not_an_open_fence(mock_agent):
    """MINOR-3 (r6): fails where the fence count reads a mid-line ``` mention
    — the tag rule was skipped (continued, shipped raw) and a spurious
    closing fence was appended before the note."""
    cont = _cont_world(mock_agent, ["Open a block with ``` and then write.\n<tool_ca"], [
        {"choices": [{"message": {"content": "ll>\nignored"}, "finish_reason": "stop"}]}])
    final = await _run_cont_world(mock_agent, "req-r6-fence-mention")
    assert cont.await_count == 0 and A.TRUNCATED_ANSWER_NOTE in final
    before = final.split(A.TRUNCATED_ANSWER_NOTE)[0]
    assert before.strip() == "Open a block with ``` and then write." and "<" not in final


@pytest.mark.parametrize("text,want", [
    ("```py\nx = 1", True), ("a\n   ```\nx", True), ("```\nx\n```\n", False),
    ("mention ``` inline", False), ("<think>\n```\n</think>\nprose", False), ("    ```not a fence", False),
])
def test_an_open_fence_is_a_line_start_fence_on_think_stripped_text(text, want):
    assert A._in_open_fence(text) is want
