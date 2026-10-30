"""§4SD review — a streamed turn sends an SSE keepalive while it runs.

With no time ceiling a build can run for hours of tool calls that stream no
text (84 % of tool turns write none). The web interface reads the agent with
a 4-hour per-gap timeout (GHOST_CHAT_TIMEOUT): with no byte in that time it
hung up and the route cancelled the build. Both stream paths (the reply tap,
and `X-Ghost-Stream: final`) now send a comment frame every
STREAM_KEEPALIVE_S; SSE parsers skip comment lines.
"""
import asyncio

import pytest

import ghost_agent.api.routes as R
from tests.test_feedback_stream_id_restamp import LLAMA_FRAMES, REQ, _sse_delta_text
from tests.test_stream_turn_usage import _agent, _drive


def _slow_agent(seconds):
    agent, _ = _agent()
    fast = agent.handle_chat

    async def slow_handle_chat(*a, **kw):
        await asyncio.sleep(seconds)          # a long tool phase: no text
        return await fast(*a, **kw)
    agent.handle_chat = slow_handle_chat
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [None, {"X-Ghost-Stream": "final"}], ids=["tap", "final"])
async def test_a_silent_turn_sends_keepalives_and_then_the_reply(monkeypatch, headers):
    # world: no byte reaches the client until the turn ends — a 4-h tool
    # phase trips the interface's read timeout and the build is cancelled
    monkeypatch.setattr(R, "STREAM_KEEPALIVE_S", 0.05)
    body = await _drive(_slow_agent(0.4), headers)
    assert body.count(R.STREAM_KEEPALIVE_FRAME) >= 3, body[:300]
    assert _sse_delta_text(body) == "Hey Vasilis!"          # the reply is untouched
    assert body.rstrip().endswith(b"data: [DONE]")


@pytest.mark.asyncio
async def test_a_fast_turn_sends_none():
    # world: the keepalive is sent unconditionally (noise on every reply)
    body = await _drive(_agent()[0])
    assert R.STREAM_KEEPALIVE_FRAME not in body
    assert _sse_delta_text(body) == "Hey Vasilis!"


def test_the_keepalive_is_a_comment_frame():
    # world: a data frame — every client would render it
    assert R.STREAM_KEEPALIVE_FRAME.startswith(b":") and R.STREAM_KEEPALIVE_FRAME.endswith(b"\n\n")
    assert 0 < R.STREAM_KEEPALIVE_S < 600


@pytest.mark.asyncio
async def test_tap_frames_around_keepalives_all_reach_the_client(monkeypatch):
    # world: each keepalive tick makes a NEW queue.get while the old one is
    # still pending — the old one takes the next frame and it is never sent
    from ghost_agent.core.reply_tap import reply_tap_context
    monkeypatch.setattr(R, "STREAM_KEEPALIVE_S", 0.03)
    agent, _ = _agent()
    fast = agent.handle_chat

    async def handle_chat(*a, **kw):
        tap = reply_tap_context.get()
        for i in range(4):
            await asyncio.sleep(0.1)          # a silent stretch: keepalives
            tap.queue.put_nowait(f": frame-{i}\n\n".encode())
        await asyncio.sleep(0.1)
        return await fast(*a, **kw)
    agent.handle_chat = handle_chat
    body = await _drive(agent)
    assert body.count(R.STREAM_KEEPALIVE_FRAME) >= 4
    for i in range(4):
        assert f": frame-{i}".encode() in body, (i, body[:400])


# ── second review: the plain path's context, a frame taken during a keepalive ─

async def _resp(agent, headers=None):
    from unittest.mock import MagicMock, patch
    from tests.test_4mp_reply_tap import _make_request
    req = _make_request({"stream": True, "messages": [{"role": "user", "content": "hi"}]}, headers)
    req.app = MagicMock()
    req.app.state.agent = agent
    with patch("ghost_agent.api.routes.get_agent", return_value=agent):
        return await R.chat_proxy(req, MagicMock())


@pytest.mark.asyncio
async def test_the_plain_paths_turn_context_reaches_the_streamed_drain():
    # world: the turn runs as a task on a COPY of the context and the drain
    # never sees what it set (the slot key: the next turn re-reads the head)
    import contextvars
    from tests.test_4mp_reply_tap import _drive as _tap_drive
    cv = contextvars.ContextVar("probe_4sd_plain_cv", default="")
    seen = []

    async def handle_chat(body, bg, request_id=None):
        cv.set("turn-value")

        async def drain():
            seen.append(cv.get())
            yield b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
            yield b"data: [DONE]\n\n"
        return drain(), 1, request_id
    await _tap_drive(handle_chat, headers={"X-Ghost-Stream": "final"})
    assert seen == ["turn-value"]


@pytest.mark.asyncio
async def test_a_frame_taken_while_a_keepalive_was_sent_is_still_sent(monkeypatch):
    # world: the turn ends while the generator sits at a keepalive yield; the
    # pending get already holds the last frame, the loop exits, it is dropped
    from ghost_agent.core.reply_tap import reply_tap_context
    monkeypatch.setattr(R, "STREAM_KEEPALIVE_S", 0.05)
    agent, _ = _agent()
    fast = agent.handle_chat

    async def handle_chat(*a, **k):
        tap = reply_tap_context.get()
        await asyncio.sleep(0.07)                      # one keepalive goes out at 0.05
        tap.queue.put_nowait(b": frame-LAST\n\n")
        return await fast(*a, **k)
    agent.handle_chat = handle_chat
    resp = await _resp(agent)
    got = []
    async for c in resp.body_iterator:
        got.append(c if isinstance(c, bytes) else c.encode())
        if got[-1] == R.STREAM_KEEPALIVE_FRAME:
            await asyncio.sleep(0.1)                   # a send held by flow control
    assert b"frame-LAST" in b"".join(got)


@pytest.mark.asyncio
async def test_a_hang_up_on_the_plain_path_cancels_the_turn(monkeypatch):
    # world: the client went away and the turn runs on with nobody reading
    monkeypatch.setattr(R, "STREAM_KEEPALIVE_S", 0.02)
    agent, _ = _agent()
    ev = []

    async def handle_chat(*a, **k):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            ev.append("cancelled")
            raise
    agent.handle_chat = handle_chat
    resp = await _resp(agent, {"X-Ghost-Stream": "final"})

    async def consume():
        async for _ in resp.body_iterator:
            pass
    t = asyncio.ensure_future(consume())
    await asyncio.sleep(0.1)
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.05)
    assert ev == ["cancelled"]


@pytest.mark.asyncio
async def test_a_hard_stop_on_the_plain_path_ends_the_stream_properly():
    # world: the registry cancels the turn task and the stream ends with no [DONE]
    agent, _ = _agent()

    async def handle_chat(*a, **k):
        asyncio.current_task().cancel()
        await asyncio.sleep(1)
    agent.handle_chat = handle_chat
    resp = await _resp(agent, {"X-Ghost-Stream": "final"})
    got = []
    async for c in resp.body_iterator:
        got.append(c if isinstance(c, bytes) else c.encode())
    assert b"".join(got).rstrip().endswith(b"data: [DONE]")
