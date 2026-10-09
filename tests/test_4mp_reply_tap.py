"""§4MP (2026-10-08): the tools-on generation's text streams as it is made.

Since the planner arm was concluded (10-04) every answer came from the
tools-on call, which streamed from llama-server internally while the client
got the finished reply in one burst — first content 4–13 s late (p90 ~18 s).
The `ReplyTap` forwards it: held for HOLD_CHARS, latched off by any call or
think markup, committed or retracted when the request ends."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.reply_tap import HOLD_CHARS, ReplyTap, reply_tap_context


def _frames(tap):
    out = []
    while not tap.queue.empty():
        out.append(tap.queue.get_nowait())
    return out


def _parse(frames):
    objs = []
    for f in frames:
        for line in f.decode().splitlines():
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                objs.append(json.loads(line[6:]))
    return objs


def _text(frames):
    return "".join(((o.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                   for o in _parse(frames))


def _retracts(frames):
    return sum(1 for o in _parse(frames) if (o.get("ghost") or {}).get("retract") is True)


ANSWER = ("PostgreSQL streaming replication ships WAL records from the primary to each standby as they "
          "are written, so a standby can be promoted within seconds. Logical replication instead decodes "
          "the WAL into row changes per table, which lets you replicate a subset of tables or across "
          "major versions, at the cost of DDL not being replicated.")


def _feed_stream(tap, text, step=7):
    acc = ""
    for i in range(0, len(text), step):
        acc += text[i:i + step]
        tap.feed(acc)


# ── the tap ───────────────────────────────────────────────────────────

def test_nothing_is_sent_inside_the_hold_and_everything_after_it():
    tap = ReplyTap("r1", "m", 1)
    tap.begin_generation()
    tap.feed(ANSWER[:HOLD_CHARS - 10])
    assert _text(_frames(tap)) == ""
    _feed_stream(tap, ANSWER)
    assert _text(_frames(tap)) == ANSWER and tap.sent == ANSWER


@pytest.mark.parametrize("call", [
    "<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"x\"}}\n</tool_call>",
    "<function=web_search>\n<parameter=query>x</parameter>\n</function>",
])
def test_a_call_inside_the_hold_sends_nothing(call):
    """84% of tool-call generations write no text first; the rest p90 238
    chars — a call inside the hold must never reach the client."""
    tap = ReplyTap("r1", "m", 1)
    tap.begin_generation()
    _feed_stream(tap, "I'll look that up for you.\n\n" + call + ANSWER, step=3)
    assert _frames(tap) == [] and tap.sent == ""


def test_markup_split_across_chunks_is_never_sent():
    tap = ReplyTap("r1", "m", 1)
    tap.begin_generation()
    _feed_stream(tap, ANSWER)
    sent_before = tap.sent
    acc = ANSWER + "\n\nNow the details <tool"
    tap.feed(acc)
    assert "<tool" not in _text(_frames(tap))
    tap.feed(acc + "_call>{}")
    assert "<" not in tap.sent[len(sent_before):] and "_call" not in tap.sent


def test_a_native_tool_call_delta_latches_the_tap():
    tap = ReplyTap("r1", "m", 1)
    tap.begin_generation()
    tap.tool_call_seen()
    _feed_stream(tap, ANSWER)
    assert tap.sent == ""


def test_a_think_block_in_the_content_is_never_sent():
    tap = ReplyTap("r1", "m", 1)
    tap.begin_generation()
    _feed_stream(tap, "<think>" + ANSWER)
    assert tap.sent == ""


def test_the_finished_reply_that_extends_the_stream_sends_only_the_rest():
    tap = ReplyTap("r1", "m", 1)
    tap.begin_generation()
    _feed_stream(tap, ANSWER)
    streamed = _frames(tap)
    final = ANSWER + "\n\nℹ️ A note appended by finalize."
    closing = tap.finish(final)
    assert tap.outcome == "committed" and _retracts(closing) == 0
    assert _text(streamed) + _text(closing) == final
    assert closing[-1] == b"data: [DONE]\n\n"
    assert _parse(closing)[-1]["choices"][0]["finish_reason"] == "stop"


def test_a_rewritten_reply_is_retracted_and_sent_whole():
    tap = ReplyTap("r1", "m", 1)
    tap.begin_generation()
    _feed_stream(tap, ANSWER)
    _frames(tap)
    final = "**Correction:** an earlier answer was wrong.\n\n" + ANSWER
    closing = tap.finish(final)
    assert tap.outcome == "retracted"
    objs = _parse(closing)
    assert (objs[0].get("ghost") or {}).get("retract") is True       # the retract leads
    assert _text(closing) == final


def test_narration_followed_by_a_call_is_retracted_at_once():
    """Narration over the hold followed by a tool call is retracted when the
    call appears (r2: it stayed on screen for the whole tool phase), and the
    next generation's answer replaces it."""
    tap = ReplyTap("r1", "m", 1)
    tap.begin_generation()
    _feed_stream(tap, ANSWER)
    _frames(tap)
    tap.tool_call_seen()
    assert _retracts(_frames(tap)) == 1 and tap.sent == ""
    tap.begin_generation()
    assert _retracts(_frames(tap)) == 0
    _feed_stream(tap, ANSWER)
    closing = tap.finish(ANSWER)
    assert tap.outcome == "committed" and _retracts(closing) == 0


def test_a_streamed_forced_final_retracts_what_was_shown():
    tap = ReplyTap("r1", "m", 1)
    tap.begin_generation()
    _feed_stream(tap, ANSWER)
    assert _retracts(tap.retract_before_stream()) == 1
    assert ReplyTap("r2", "m", 1).retract_before_stream() == []


def test_frames_carry_the_agents_request_id():
    tap = ReplyTap("abcd1234", "m", 1)
    tap.begin_generation()
    _feed_stream(tap, ANSWER)
    assert {o["id"] for o in _parse(_frames(tap) + tap.finish(ANSWER))} == {"chatcmpl-abcd1234"}


# ── the route ─────────────────────────────────────────────────────────

def _make_request(body, headers=None):
    from fastapi import Request
    req = MagicMock(spec=Request)
    req.method = "POST"
    req.headers = {"content-type": "application/json", **(headers or {})}
    req.body = AsyncMock(return_value=json.dumps(body).encode("utf-8"))
    req.json = AsyncMock(return_value=body)
    return req


async def _drive(fake_handle_chat, headers=None, stream=True):
    from ghost_agent.api.routes import chat_proxy
    agent = MagicMock()
    agent.handle_chat = fake_handle_chat
    agent.context.args.model = "test-model"
    agent.is_trivial_reply = MagicMock(return_value=False)

    async def _stream_openai(model, content, created, req_id, extra=None):
        tap = ReplyTap(req_id, model, created)
        for f in tap.finish(content, extra=extra):
            yield f
    agent.context.llm_client.stream_openai = _stream_openai
    req = _make_request({"stream": stream, "messages": [{"role": "user", "content": "hi"}]}, headers)
    req.app = MagicMock()
    req.app.state.agent = agent
    with patch("ghost_agent.api.routes.get_agent", return_value=agent):
        resp = await chat_proxy(req, MagicMock())
    out = []
    async for chunk in resp.body_iterator:
        out.append(chunk if isinstance(chunk, (bytes, bytearray)) else str(chunk).encode())
    return out


async def test_the_first_tokens_reach_the_client_before_the_turn_ends():
    first_seen = asyncio.Event()
    order = []

    async def fake_handle_chat(body, bg, request_id=None):
        tap = reply_tap_context.get()
        assert tap is not None
        tap.begin_generation()
        _feed_stream(tap, ANSWER)
        # the turn is still running: wait until the client has the text
        await asyncio.wait_for(first_seen.wait(), 5)
        order.append("turn ends")
        return ANSWER, 1, request_id

    from ghost_agent.api.routes import chat_proxy
    agent = MagicMock()
    agent.handle_chat = fake_handle_chat
    agent.context.args.model = "test-model"
    agent.is_trivial_reply = MagicMock(return_value=False)
    req = _make_request({"stream": True, "messages": [{"role": "user", "content": "hi"}]})
    req.app = MagicMock()
    req.app.state.agent = agent
    with patch("ghost_agent.api.routes.get_agent", return_value=agent):
        resp = await chat_proxy(req, MagicMock())
    got = []
    async for chunk in resp.body_iterator:
        got.append(chunk)
        if _text([chunk]) and not first_seen.is_set():
            order.append("client has text")
            first_seen.set()
    assert order == ["client has text", "turn ends"]
    assert _text(got) == ANSWER and _retracts(got) == 0
    assert got[-1] == b"data: [DONE]\n\n"


async def test_a_reply_that_never_released_is_sent_as_before():
    async def fake_handle_chat(body, bg, request_id=None):
        tap = reply_tap_context.get()
        tap.begin_generation()
        tap.feed("Hi!")                 # under the hold
        return "Hi! How can I help?", 1, request_id
    got = await _drive(fake_handle_chat)
    assert _text(got) == "Hi! How can I help?" and _retracts(got) == 0


@pytest.mark.parametrize("headers", [{"X-Ghost-Stream": "final"}, {"X-Ghost-Requester": "member"}])
async def test_no_tap_for_a_client_that_asks_for_the_final_reply_or_a_member(headers):
    seen = []

    async def fake_handle_chat(body, bg, request_id=None):
        seen.append(reply_tap_context.get())
        return ANSWER, 1, request_id or "x"
    got = await _drive(fake_handle_chat, headers=headers)
    assert seen == [None] and _text(got) == ANSWER


async def test_the_kill_switch_turns_the_tap_off(monkeypatch):
    monkeypatch.setenv("GHOST_STREAM_TAP", "0")
    seen = []

    async def fake_handle_chat(body, bg, request_id=None):
        seen.append(reply_tap_context.get())
        return ANSWER, 1, request_id or "x"
    await _drive(fake_handle_chat)
    assert seen == [None]


async def test_a_streamed_forced_final_after_released_text_is_preceded_by_a_retract():
    async def fake_handle_chat(body, bg, request_id=None):
        tap = reply_tap_context.get()
        tap.begin_generation()
        _feed_stream(tap, ANSWER)

        async def forced():
            yield ('data: ' + json.dumps({"id": "chatcmpl-x", "choices": [{"index": 0, "delta": {"content": "Final."}}]})
                   + "\n\n").encode()
            yield b"data: [DONE]\n\n"
        return forced(), 1, request_id
    got = await _drive(fake_handle_chat)
    objs = _parse(got)
    i_retract = next(i for i, o in enumerate(objs) if (o.get("ghost") or {}).get("retract"))
    i_final = next(i for i, o in enumerate(objs)
                   if ((o.get("choices") or [{}])[0].get("delta") or {}).get("content") == "Final.")
    assert i_retract < i_final


# ── the agent feeds the tap from the real tools-on loop ───────────────

def _sse(delta):
    return f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()


async def _run_turn(script):
    """Drive the REAL handle_chat (stream:true) with a scripted upstream;
    returns (reply, tap)."""
    from tests.test_finalize_stream_pins import make_stream_agent
    a = make_stream_agent()
    calls = {"n": 0}

    def make_stream(*_a, **_k):
        n = calls["n"]
        calls["n"] += 1
        deltas = script[min(n, len(script) - 1)]

        async def gen():
            for d in deltas:
                yield _sse(d)
            yield b"data: [DONE]\n\n"
        return gen()
    a.context.llm_client.stream_chat_completion = make_stream
    tap = ReplyTap("req-tap", "m", 1)
    tok = reply_tap_context.set(tap)
    try:
        out = await a.handle_chat({"messages": [{"role": "user", "content": "explain replication"}],
                                   "stream": True}, background_tasks=MagicMock(), request_id="req-tap")
    finally:
        reply_tap_context.reset(tok)
    return out, tap


async def test_the_real_loop_streams_a_tool_free_answer_through_the_tap():
    chunks = [ANSWER[i:i + 9] for i in range(0, len(ANSWER), 9)]
    out, tap = await _run_turn([[{"reasoning_content": "Explain both kinds."}] + [{"content": c} for c in chunks]])
    reply = out[0] if isinstance(out, tuple) else out
    assert isinstance(reply, str)
    streamed = _text(_frames(tap))
    assert streamed and reply.startswith(streamed.rstrip())
    closing = tap.finish(reply)
    assert tap.outcome == "committed" and streamed + _text(closing) == reply


async def test_the_real_loop_latches_on_a_native_tool_call():
    call = {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "web_search",
                                                                   "arguments": "{\"query\": \"x\"}"}}]}
    pre = "Let me look that up in the current documentation before answering. " * 2      # under the hold
    post = " (searching the replication chapter and the release notes for the version)" * 3  # crosses it AFTER the call
    out, tap = await _run_turn([[{"content": pre}, call, {"content": post}],
                                [{"content": c} for c in [ANSWER[i:i + 9] for i in range(0, len(ANSWER), 9)]]])
    frames = _frames(tap)
    assert len(pre + post) > HOLD_CHARS
    assert "Let me look that up" not in _text(frames) and "searching the replication" not in _text(frames)


# ── the clients ───────────────────────────────────────────────────────
# (the same loading convention as test_4ml_interfaces: the JS is EXECUTED
# under node, not matched as text)
from pathlib import Path as _Path  # noqa: E402

STATIC = _Path(__file__).resolve().parents[1] / "interface" / "static"
APP = (STATIC / "app.js").read_text(encoding="utf-8")

def test_the_web_client_drops_the_retracted_text_and_its_speech():
    from tests.helpers import eval_js, extract_js_function
    fn = extract_js_function(APP, "_applyStreamRetract")
    prelude = ("let currentAccumulatedContent = 'Narration before a call.'; let currentStreamPrefixLen = 4;"
               "let currentTTSMutedLength = 9; let currentAgentMessageDiv = {innerHTML: '<p>x</p>'};"
               "let currentSpeechHold = true;"
               "let stopped = 0; let cancelled = 0; function stopTTS() { stopped++; }"
               "function _cancelScheduledStreamRender() { cancelled++; }\n")
    got = eval_js(prelude + fn, "[_applyStreamRetract(), currentAccumulatedContent, currentStreamPrefixLen,"
                                " currentTTSMutedLength, currentAgentMessageDiv.innerHTML, stopped, cancelled,"
                                " _applyStreamRetract(), stopped, currentSpeechHold]")
    assert got == [True, "", 0, 0, "", 1, 1, False, 1, False]          # a second retract with nothing shown is a no-op


def test_the_web_stream_loop_calls_the_retract_on_the_frame():
    """Executed shape: the frame loop routes a `ghost.retract` frame to the
    helper and reads no content from it (AST-free: the node parser checks
    the call sits inside the retract branch)."""
    from tests.helpers import eval_js
    i = APP.index("data.ghost.retract === true")
    branch = APP[i:APP.index("}", i) + 1]
    got = eval_js("let n = 0; function _applyStreamRetract() { n++; }\n"
                  "function step(data) { for (const _ of [0]) { if (data.ghost && " + branch
                  + " return 'content'; } return 'skipped'; }",
                  "[step({ghost: {retract: true}}), n, step({choices: []}), n]")
    assert got == ["skipped", 1, "content", 1]


def _cw():
    from tests.test_4kp_think_split_and_clients import _clockwork_send
    return _clockwork_send


def test_the_clockwork_client_clears_and_silences_on_a_retract():
    silenced = []
    narr = 'data: {"choices":[{"delta":{"content":"I will check the weather first. "}}]}'
    retract = 'data: {"choices":[{"delta":{}}],"ghost":{"retract":true}}'
    answer = 'data: {"choices":[{"delta":{"content":"It is sunny. "}}]}'
    shown, spoken, events = _cw()([narr, retract, answer, "data: [DONE]"],
                                  _silence=lambda: silenced.append(1))
    assert ("retract_response", "") in events and silenced == [1]
    after = events[events.index(("retract_response", "")) + 1:]
    assert "".join(a[1] for a in after if a[0] == "update_response") == "It is sunny. "
    assert spoken[-1] == "It is sunny."


def test_a_retract_with_nothing_shown_changes_nothing_on_the_clockwork():
    shown, spoken, events = _cw()(['data: {"choices":[{"delta":{}}],"ghost":{"retract":true}}',
                                   'data: {"choices":[{"delta":{"content":"Hi there. "}}]}', "data: [DONE]"],
                                  _silence=lambda: (_ for _ in ()).throw(AssertionError("silenced")))
    assert shown == "Hi there. " and ("retract_response", "") not in events


def test_the_cli_marks_a_retract_and_keeps_only_the_replacement(monkeypatch):
    import io
    from tests.test_ghost_cli import _FakeSSE
    from tests.test_4kp_think_split_and_clients import _cli
    cli = _cli()
    captured = io.StringIO()
    monkeypatch.setattr(cli, "console", cli.Console(file=captured, force_terminal=True, width=70, height=24))
    api = cli.GhostAPI("http://localhost:9", "k")
    frames = [b'data: {"choices":[{"delta":{"content":"I will check the weather first.\\n\\n"}}]}',
              b'data: {"choices":[{"delta":{}}],"ghost":{"retract":true}}',
              b'data: {"choices":[{"delta":{"content":"It is sunny."}}]}', b"data: [DONE]"]
    monkeypatch.setattr(api, "chat_stream", lambda *a, **kw: _FakeSSE(frames))
    assert cli.GhostCLI(api).stream_reply("r1") == "It is sunny."
    assert "revised" in captured.getvalue()


def test_piped_cli_output_asks_for_the_final_reply_only(monkeypatch):
    import io
    from tests.test_ghost_cli import _FakeSSE
    from tests.test_4kp_think_split_and_clients import _cli
    cli = _cli()
    buf = io.StringIO()
    buf.isatty = lambda: False
    monkeypatch.setattr(cli.sys, "stdout", buf)
    api = cli.GhostAPI("http://localhost:9", "k")
    seen = {}

    def post(*a, **k):
        seen.update(k.get("headers") or {})
        return _FakeSSE([b'data: {"choices":[{"delta":{"content":"Done."}}]}', b"data: [DONE]"])
    monkeypatch.setattr(api.http, "post", post)
    cli.one_shot(api, "q", None)
    assert seen.get("X-Ghost-Stream") == "final" and buf.getvalue().strip() == "Done."



# ── r2 (fresh reader) ─────────────────────────────────────────────────

@pytest.mark.parametrize("tail", [
    "\n\nDYNAMIC SYSTEM STATE\nUser profile: lives at 12 Example St\n",       # prompt bleed
    '\n\n<tool name="file_system">{"operation": "write"}</tool>',              # the <tool …> dialect
    "\n\n<function_name=file_system>\n<parameter=path>x</parameter>",       # <function_name=…>
])
def test_the_shared_detectors_stop_what_the_parsers_treat_as_a_call_or_bleed(tail):
    """r2 M3: a private regex missed four shapes the parsers accept."""
    from ghost_agent.core.agent import reply_tap_unsafe
    tap = ReplyTap("r1", "m", 1)
    tap.begin_generation(unsafe=reply_tap_unsafe)
    _feed_stream(tap, ANSWER + tail + "x" * 60, step=5)
    shown = "".join(_text([f]) for f in _frames(tap))
    assert tap.sent == "" and "DYNAMIC" not in tap.sent and "file_system" not in tap.sent
    assert ANSWER[:40] in shown                     # it streamed, then was retracted at the markup


def test_a_raw_json_call_is_never_sent():
    tap = ReplyTap("r1", "m", 1)
    tap.begin_generation()
    _feed_stream(tap, json.dumps({"name": "file_system", "arguments": {"operation": "write",
                                                                      "content": "key: value\n" * 40}}))
    assert _frames(tap) == []


def test_content_before_any_reasoning_is_held_while_thinking_is_on():
    """r2 M2: with thinking on, content that arrives with no reasoning
    channel may BE the reasoning (§4KO) — it was shown and spoken."""
    tap = ReplyTap("r1", "m", 1)
    tap.begin_generation()
    acc = ""
    for i in range(0, len(ANSWER), 7):
        acc += ANSWER[i:i + 7]
        tap.feed(acc, may_release=False)
    assert _frames(tap) == []


def test_the_seam_matches_the_final_reply_exactly():
    """r2: a generation ending in blank lines plus an appended note showed
    four newlines where the stored reply has two."""
    tap = ReplyTap("r1", "m", 1)
    tap.begin_generation()
    _feed_stream(tap, ANSWER + "\n\n")
    streamed = _text(_frames(tap))
    final = ANSWER + "\n\nNote: appended by finalize."
    assert streamed + _text(tap.finish(final)) == final


def test_a_bound_turn_id_stamps_every_later_frame():
    tap = ReplyTap("dup1", "m", 1)
    tap.bind("dup1#2")
    tap.begin_generation()
    _feed_stream(tap, ANSWER)
    assert {o["id"] for o in _parse(_frames(tap) + tap.finish(ANSWER))} == {"chatcmpl-dup1#2"}


async def test_an_error_after_streamed_text_retracts_it():
    """r2 M4: the partial text became the stored reply of a failed turn."""
    async def fake_handle_chat(body, bg, request_id=None):
        tap = reply_tap_context.get()
        tap.begin_generation()
        _feed_stream(tap, ANSWER)
        raise RuntimeError("boom")
    got = await _drive(fake_handle_chat)
    objs = _parse(got)
    i_retract = next(i for i, o in enumerate(objs) if (o.get("ghost") or {}).get("retract"))
    i_err = next(i for i, o in enumerate(objs) if o.get("error"))
    assert i_retract < i_err


_PROBE_CV = __import__("contextvars").ContextVar("probe_4mp_cv", default="")


async def test_the_turns_context_variables_reach_the_streamed_drain():
    """r2 M1: run as a task, the turn's context variables (trajectory id,
    event project, conversation key) no longer reached the streamed drain."""
    seen = []

    async def fake_handle_chat(body, bg, request_id=None):
        _PROBE_CV.set("turn-value")

        async def drain():
            seen.append(_PROBE_CV.get())
            yield b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
            yield b"data: [DONE]\n\n"
        return drain(), 1, request_id
    await _drive(fake_handle_chat)
    assert seen == ["turn-value"]


async def test_the_real_loop_holds_content_that_came_without_reasoning():
    """r2 M2 at the agent: thinking on, no reasoning channel — the content
    may be the reasoning itself, so nothing streams (finish sends the
    finalized reply, as before)."""
    chunks = [ANSWER[i:i + 9] for i in range(0, len(ANSWER), 9)]
    out, tap = await _run_turn([[{"content": c} for c in chunks]])
    assert _frames(tap) == [] and tap.released_generations == 0



# ── §4MR (fresh-eye verification) ─────────────────────────────────────

def _cancel_agent(gate):
    from tests.test_finalize_stream_pins import make_stream_agent
    a = make_stream_agent()

    def make_stream(*_a, **_k):
        async def gen():
            yield _sse({"reasoning_content": "think"})
            for i in range(0, len(ANSWER), 9):
                yield _sse({"content": ANSWER[i:i + 9]})
            await gate.wait()                     # the turn is mid-generation
            yield b"data: [DONE]\n\n"
        return gen()
    a.context.llm_client.stream_chat_completion = make_stream
    a.is_trivial_reply = MagicMock(return_value=False)
    a.context.args.model = "test-model"
    return a


async def _cancel_resp(a, rid):
    from ghost_agent.api.routes import chat_proxy
    req = _make_request({"stream": True, "messages": [{"role": "user", "content": "explain"}]},
                        {"X-Request-ID": rid})
    req.app = MagicMock()
    req.app.state.agent = a
    with patch("ghost_agent.api.routes.get_agent", return_value=a):
        return await chat_proxy(req, MagicMock())


async def _consume_until_text(resp, got, err=None):
    async def consume():
        try:
            async for c in resp.body_iterator:
                got.append(c)
        except BaseException as e:  # noqa: BLE001
            if err is not None:
                err.append(type(e).__name__)
            raise
    t = asyncio.create_task(consume())
    for _ in range(1500):
        await asyncio.sleep(0.01)
        if _text(got):
            break
    assert _text(got), "no live text"
    return t


async def test_a_client_that_leaves_mid_turn_cancels_the_turn_and_frees_the_lock():
    """§4MR: the turn is a TASK now — only the route's cancel stops it when
    the client drops; without it a closed tab held the global turn lock."""
    gate = asyncio.Event()
    a = _cancel_agent(gate)
    from ghost_agent.core.turns import get_turn_registry
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        resp = await _cancel_resp(a, "disc0001")
        got = []
        t = await _consume_until_text(resp, got)
        t.cancel()                                  # starlette cancels the send task on disconnect
        with pytest.raises(asyncio.CancelledError):
            await t
        for _ in range(100):
            await asyncio.sleep(0.01)
            if not a.agent_semaphore.locked():
                break
        assert not a.agent_semaphore.locked() and get_turn_registry(a).list() == []


async def test_a_hard_stop_after_live_text_retracts_it_and_ends_the_stream():
    """§4MR: a hard cancel hit the turn task; CancelledError skipped the error
    retract and the streamed text stayed as the stopped turn's reply."""
    gate = asyncio.Event()
    a = _cancel_agent(gate)
    from ghost_agent.core.turns import get_turn_registry
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        resp = await _cancel_resp(a, "hard0001")
        got, err = [], []
        t = await _consume_until_text(resp, got, err)
        get_turn_registry(a).cancel("hard0001", hard=True)
        await asyncio.wait_for(t, 5)
    assert not err and _retracts(got) == 1 and got[-1] == b"data: [DONE]\n\n"


def test_a_crlf_answer_commits():
    tap = ReplyTap("r1", "m", 1)
    tap.begin_generation()
    crlf = ANSWER.replace(". ", ".\r\n")
    _feed_stream(tap, crlf)
    _frames(tap)
    closing = tap.finish(crlf.replace("\r", "") + "\n\nNote.")
    assert tap.outcome == "committed" and _retracts(closing) == 0
