"""§4RD-c — a Stop lands inside a live-streamed generation.

Live probe probe-4rdstop (2026-10-25): an owner reply streaming through the
tap (§4MP) took a cooperative cancel ("cancelled": true) and ran on, because
the flag was read only between model calls. The proxy's 5 s grace ran out,
its hang-up cancelled the task, and the Stop was booked as an interruption
with nothing kept in the session.

World where each test fails: the generation stops reading its cancel flag
(the stream runs to its end), the Stop's partial is not what the client was
shown, the stream's `except Exception` swallows the cancel as an LLM error,
or the route's hang-up keeps nothing of the shown text.
"""
import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.core.reply_tap import ReplyTap, reply_tap_context
from tests.test_4mp_reply_tap import ANSWER, _make_request, _sse, _text, _retracts


def _endless_agent(served, deltas_before=None):
    """The model keeps producing tokens until the turn stops reading."""
    from tests.test_finalize_stream_pins import make_stream_agent
    a = make_stream_agent()

    def make_stream(*_a, **_k):
        async def gen():
            for d in (deltas_before or [{"reasoning_content": "think"}]):
                yield _sse(d)
            i = 0
            while True:
                served["n"] += 1
                yield _sse({"content": (ANSWER + " ")[i % len(ANSWER):][:9] or "x"})
                i += 9
                await asyncio.sleep(0.005)
        return gen()
    a.context.llm_client.stream_chat_completion = make_stream
    a.is_trivial_reply = MagicMock(return_value=False)
    a.context.args.model = "test-model"
    return a


async def _turn_with_tap(a, rid, tap, cancel_after_s=0.4):
    from ghost_agent.core.turns import get_turn_registry

    async def stop_later():
        await asyncio.sleep(cancel_after_s)
        return get_turn_registry(a).cancel(rid)
    tok = reply_tap_context.set(tap) if tap is not None else None
    try:
        with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
            stopper = asyncio.ensure_future(stop_later())
            out = await asyncio.wait_for(a.handle_chat(
                {"messages": [{"role": "user", "content": "explain replication"}], "stream": True},
                background_tasks=MagicMock(), request_id=rid), 10)
    finally:
        if tok is not None:
            reply_tap_context.reset(tok)
    return out, await stopper


async def test_a_cooperative_stop_ends_the_generation_and_keeps_what_was_shown():
    served = {"n": 0}
    a = _endless_agent(served)
    seen = []
    a._record_aborted_turn = lambda **kw: seen.append(kw)
    tap = ReplyTap("live0001", "m", 1)
    out, verdict = await _turn_with_tap(a, "live0001", tap)
    assert verdict.get("cancelled") is True
    reply = out[0] if isinstance(out, tuple) else out
    shown = _text([tap.queue.get_nowait() for _ in range(tap.queue.qsize())])
    assert shown.strip(), "nothing was streamed before the Stop"
    assert reply.startswith(shown.rstrip()) and "Turn cancelled" in reply
    # the shown text extends exactly: the route commits, nothing is retracted
    closing = tap.finish(reply)
    assert tap.outcome == "committed" and not _retracts(closing)
    # booked as a Stop (the default abort marker), with the shown text
    assert len(seen) == 1 and seen[0]["reason"].startswith("cancelled")
    assert seen[0].get("marker", "[ATTEMPT_ABORTED_TURN]") == "[ATTEMPT_ABORTED_TURN]"
    assert seen[0]["partial"].startswith(shown.strip()[:40])
    n_at_stop = served["n"]
    await asyncio.sleep(0.1)
    assert served["n"] == n_at_stop, "the upstream is still being read after the Stop"
    assert not a.agent_semaphore.locked()


async def test_without_a_tap_the_partial_is_the_generations_reply_text():
    served = {"n": 0}
    a = _endless_agent(served)
    seen = []
    a._record_aborted_turn = lambda **kw: seen.append(kw)
    out, _ = await _turn_with_tap(a, "live0002", None)
    reply = out[0] if isinstance(out, tuple) else out
    assert reply.startswith(ANSWER[:30]) and "Turn cancelled" in reply
    assert seen and seen[0]["partial"].startswith(ANSWER[:30])


async def test_a_stop_during_a_tool_call_keeps_no_call_markup():
    served = {"n": 0}
    call = {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "web_search",
                                                                   "arguments": "{\"query\": \"x\"}"}}]}
    a = _endless_agent(served, deltas_before=[{"reasoning_content": "think"}, call])
    a._record_aborted_turn = lambda **kw: None
    out, _ = await _turn_with_tap(a, "live0003", None)
    reply = out[0] if isinstance(out, tuple) else out
    assert "No output was produced" in reply and "web_search" not in reply


async def test_a_stop_is_not_reported_as_an_llm_failure():
    served = {"n": 0}
    a = _endless_agent(served)
    a._record_aborted_turn = lambda **kw: None
    out, _ = await _turn_with_tap(a, "live0004", ReplyTap("live0004", "m", 1))
    reply = out[0] if isinstance(out, tuple) else out
    assert "CRITICAL" not in reply and "communicating with the LLM" not in reply


async def test_a_hang_up_mid_turn_keeps_the_shown_text_in_the_session_marked_cut():
    from ghost_agent.api.routes import chat_proxy, STREAM_CUT_NOTE
    served = {"n": 0}
    a = _endless_agent(served)
    a._record_aborted_turn = lambda **kw: None
    store = MagicMock()
    store.get.return_value = None
    req = _make_request({"stream": True, "session_id": "s1",
                         "messages": [{"role": "user", "content": "explain"}]}, {"X-Request-ID": "live0005"})
    req.app = MagicMock()
    req.app.state.agent = a
    with patch("ghost_agent.api.routes.get_agent", return_value=a), \
            patch("ghost_agent.core.sessions.get_session_store", return_value=store), \
            patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        resp = await chat_proxy(req, MagicMock())
        got = []
        it = resp.body_iterator
        async for c in it:
            got.append(c)
            if len(_text(got)) > 200:
                break
        await it.aclose()                              # the hang-up
        for _ in range(100):
            if store.append_turn.called:
                break
            await asyncio.sleep(0.02)
    assert store.append_turn.called
    stored = store.append_turn.call_args.args[2]
    assert stored.startswith(_text(got).strip()[:60]) and stored.endswith(STREAM_CUT_NOTE)


async def test_the_partial_is_what_the_tap_showed_not_what_it_held_back():
    """A raw-JSON call shape latches the tap (nothing shown); the Stop must
    not hand that text back as the reply."""
    served = {"n": 0}
    a = _endless_agent(served, deltas_before=[
        {"reasoning_content": "think"},
        {"content": '{"name": "web_search", "arguments": {"query": "'}])
    a._record_aborted_turn = lambda **kw: None
    tap = ReplyTap("live0006", "m", 1)
    out, _ = await _turn_with_tap(a, "live0006", tap)
    reply = out[0] if isinstance(out, tuple) else out
    assert tap.sent == "" and "web_search" not in reply and "No output was produced" in reply


# ── review r1 ─────────────────────────────────────────────────────────

def _scripted_agent(script):
    """One generator per model call, from `script` (the last repeats)."""
    from tests.test_finalize_stream_pins import make_stream_agent
    a = make_stream_agent()
    calls = {"n": 0}

    def make_stream(*_a, **_k):
        gen = script[min(calls["n"], len(script) - 1)]
        calls["n"] += 1
        return gen()
    a.context.llm_client.stream_chat_completion = make_stream
    a.is_trivial_reply = MagicMock(return_value=False)
    a.context.args.model = "test-model"
    return a, calls


async def test_a_reply_that_finished_generating_is_not_cut_by_a_late_stop():
    """r1 m3: the Stop arrived between the finish frame and [DONE]."""
    async def gen():
        yield _sse({"reasoning_content": "think"})
        for i in range(0, len(ANSWER), 9):
            yield _sse({"content": ANSWER[i:i + 9]})
        yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode()
        await asyncio.sleep(0.8)                 # the Stop lands here
        yield f"data: {json.dumps({'choices': [{'delta': {}}], 'usage': {}})}\n\n".encode()
        yield b"data: [DONE]\n\n"
    a, _ = _scripted_agent([gen])
    a._record_aborted_turn = lambda **kw: None
    out, _ = await _turn_with_tap(a, "live0007", ReplyTap("live0007", "m", 1), cancel_after_s=0.3)
    reply = out[0] if isinstance(out, tuple) else out
    assert "Turn cancelled" not in reply and ANSWER[-40:].strip() in reply


async def _stop_then_hang_up(a, rid, flag):
    """The proxy's sequence when no check is reached: (a cooperative Stop,
    then) the hang-up cancels the turn task."""
    from ghost_agent.core.turns import get_turn_registry
    seen = []
    a._record_aborted_turn = lambda **kw: seen.append(kw)
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        t = asyncio.ensure_future(a.handle_chat(
            {"messages": [{"role": "user", "content": "explain replication"}], "stream": True},
            background_tasks=MagicMock(), request_id=rid))
        await asyncio.sleep(0.3)
        if flag:
            assert get_turn_registry(a).cancel(rid)["mode"] == "cooperative"
        await asyncio.sleep(0.5)                  # no chunk comes: no check is reached
        assert not t.done()
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
    return seen


async def _prefill_forever():
    await asyncio.sleep(3600)                     # a prefill that never yields
    yield b"data: [DONE]\n\n"


async def test_a_stop_that_no_check_reached_is_booked_as_a_stop():
    """r1 M2: the Stop landed during a prefill; the proxy's hang-up after its
    grace is that Stop, not an interruption."""
    a, _ = _scripted_agent([_prefill_forever])
    seen = await _stop_then_hang_up(a, "live0008", flag=True)
    assert len(seen) == 1 and seen[0]["marker"] == "[ATTEMPT_ABORTED_TURN]"
    assert "stopped" in seen[0]["reason"]


async def test_a_hang_up_with_no_stop_is_still_an_interruption():
    a, _ = _scripted_agent([_prefill_forever])
    seen = await _stop_then_hang_up(a, "live0009", flag=False)
    assert len(seen) == 1 and seen[0]["marker"] == "[TURN_INTERRUPTED]"


async def test_a_stop_in_the_next_round_keeps_the_answer_an_earlier_round_kept():
    """r1 M1: §4PO kept the answer and asked for a continuation; the Stop
    landed while the continuation was inside the tap's hold (nothing shown)."""
    first = ("No — Bloodborne is a PS4 exclusive and has never been released for Mac.\n\n"
             "Let me check if there are any workarounds like cloud gaming or emulation:")

    async def round1():
        yield _sse({"reasoning_content": "answer it"})
        for i in range(0, len(first), 9):
            yield _sse({"content": first[i:i + 9]})
        yield b"data: [DONE]\n\n"

    async def round2():                           # thinks on; never leaves the hold
        while True:
            yield _sse({"reasoning_content": "more "})
            await asyncio.sleep(0.005)
    a, calls = _scripted_agent([round1, round2])
    seen = []
    a._record_aborted_turn = lambda **kw: seen.append(kw)
    out, _ = await _turn_with_tap(a, "live0010", ReplyTap("live0010", "m", 1), cancel_after_s=0.5)
    reply = out[0] if isinstance(out, tuple) else out
    assert calls["n"] >= 2, "the §4PO continuation did not run"
    assert reply.startswith("No — Bloodborne is a PS4 exclusive") and "Turn cancelled" in reply
    assert seen and seen[0]["partial"].startswith("No — Bloodborne")


# ── review r2 ─────────────────────────────────────────────────────────

async def test_the_taps_lead_stays_at_the_head_of_a_stopped_reply():
    """r2 m2: the lead (owner-rule banner, staged correction) opens every
    generation's first piece; joined after kept text it landed mid-reply."""
    first = ("No — Bloodborne is a PS4 exclusive and has never been released for Mac.\n\n"
             "Let me check if there are any workarounds like cloud gaming or emulation:")

    async def round1():
        yield _sse({"reasoning_content": "answer it"})
        for i in range(0, len(first), 9):
            yield _sse({"content": first[i:i + 9]})
        yield b"data: [DONE]\n\n"

    async def round2():
        yield _sse({"reasoning_content": "go on"})
        i = 0
        while True:
            yield _sse({"content": (ANSWER + " ")[i % len(ANSWER):][:9] or "x"})
            i += 9
            await asyncio.sleep(0.005)
    a, _ = _scripted_agent([round1, round2])
    a._record_aborted_turn = lambda **kw: None
    tap = ReplyTap("live0011", "m", 1)
    tap.set_lead("> BANNER\n\n---\n\n")
    out, _ = await _turn_with_tap(a, "live0011", tap, cancel_after_s=0.8)
    reply = out[0] if isinstance(out, tuple) else out
    assert reply.startswith("> BANNER\n\n---\n\nNo — Bloodborne"), reply[:80]
    assert reply.count("BANNER") == 1


async def test_a_stop_before_anything_showed_keeps_the_question_in_the_session():
    """r2 m3: a Stop the turn never reached a check for (a prefill), then the
    proxy's hang-up: the session still gets the turn — the request row alone
    (§4SB: a reply that is ONLY the "_(Turn cancelled…)" stub is no reply;
    one was consolidated into smart memory on 10-29). Until §4SB this pinned
    the stub itself as the stored reply."""
    from ghost_agent.api.routes import chat_proxy
    from ghost_agent.core.turns import get_turn_registry
    a, _ = _scripted_agent([_prefill_forever])
    a._record_aborted_turn = lambda **kw: None
    store = MagicMock()
    store.get.return_value = None
    req = _make_request({"stream": True, "session_id": "s1",
                         "messages": [{"role": "user", "content": "explain"}]}, {"X-Request-ID": "live0012"})
    req.app = MagicMock()
    req.app.state.agent = a
    with patch("ghost_agent.api.routes.get_agent", return_value=a), \
            patch("ghost_agent.core.sessions.get_session_store", return_value=store), \
            patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        resp = await chat_proxy(req, MagicMock())
        async def consume():
            async for _ in resp.body_iterator:
                pass
        t = asyncio.ensure_future(consume())
        await asyncio.sleep(0.3)                       # the turn is in its prefill
        assert get_turn_registry(a).cancel("live0012")["mode"] == "cooperative"
        await asyncio.sleep(0.3)
        t.cancel()                                     # the proxy's hang-up after its grace
        with pytest.raises(asyncio.CancelledError):
            await t
        for _ in range(100):
            if store.append_turn.called:
                break
            await asyncio.sleep(0.02)
    assert store.append_turn.called
    assert [m["content"] for m in store.append_turn.call_args.args[1]] == ["explain"]   # the request row
    assert store.append_turn.call_args.args[2] == ""                                  # no stub stored (§4SB)


async def test_a_plain_stop_commits_without_a_retract_when_the_shown_text_ends_in_a_space():
    """r3: the shown text ended "… the " and was stripped: the closing frames
    retracted it and sent it again (the reply blinked)."""
    async def gen():
        yield _sse({"reasoning_content": "think"})
        words = (ANSWER + " ") * 50
        for w in words.split(" "):
            yield _sse({"content": w + " "})
            await asyncio.sleep(0.005)
    a, _ = _scripted_agent([gen])
    a._record_aborted_turn = lambda **kw: None
    tap = ReplyTap("live0013", "m", 1)
    out, _ = await _turn_with_tap(a, "live0013", tap)
    reply = out[0] if isinstance(out, tuple) else out
    assert tap.sent.endswith(" ")
    closing = tap.finish(reply)
    assert tap.outcome == "committed" and not _retracts(closing)
