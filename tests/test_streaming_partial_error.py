"""Regression: a mid-stream exception must NOT emit a `delta.content`
chunk if real content has already shipped to the client.

The old SSE error path always emitted both an `event: error` frame
AND a `data: {choices:[{delta:{content: "CRITICAL SERVER ERROR..."}}]}`
chunk. SSE clients (the Slack bot, the web UI in `interface/server.py`)
accumulate `delta.content` into the visible bubble. So a mid-stream
ConnectionResetError that fires after several chunks have streamed
produces a user-visible:

    "Here is the answ" + "CRITICAL SERVER ERROR: ConnectionResetError"

mashed end-to-end. The error-event frame alone is enough for
programmatic detection once content has started; emitting an extra
content chunk corrupts the visible reply.

Fix: track `content_started` in the streaming generator. Always emit
the `event: error` frame. Only emit the `delta.content` chunk when
nothing visible has streamed yet (so a pre-stream error still shows
something to clients that don't listen for `event: error`).
"""
import json
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, BackgroundTasks
from fastapi.testclient import TestClient

from ghost_agent.api.routes import router


def _make_app(handle_chat_impl):
    """Build a minimal FastAPI app with the chat router and a stub agent.

    `handle_chat_impl` returns either:
      - (str, int, str) for non-streaming content
      - (async-iterator, int, str) for streaming
    """
    app = FastAPI()

    agent = MagicMock()
    agent.handle_chat = AsyncMock(side_effect=handle_chat_impl)
    args = MagicMock()
    args.api_key = ""  # disables verify_api_key
    args.model = "test-model"
    agent.context = MagicMock()
    agent.context.args = args
    agent.context.llm_client = MagicMock()

    # Default: never used by these tests; raise if accidentally called.
    async def _no_stream(*a, **k):
        raise AssertionError("stream_openai should not be called in these tests")
        yield  # pragma: no cover
    agent.context.llm_client.stream_openai = _no_stream

    app.state.agent = agent
    app.state.args = args
    app.include_router(router)
    return app, agent


def _parse_sse(text):
    """Parse SSE text into a list of (event, data) tuples."""
    events = []
    cur_event = "message"
    cur_data = []
    for line in text.split("\n"):
        if line == "":
            if cur_data:
                events.append((cur_event, "\n".join(cur_data)))
            cur_event = "message"
            cur_data = []
        elif line.startswith("event: "):
            cur_event = line[7:]
        elif line.startswith("data: "):
            cur_data.append(line[6:])
        elif line.startswith(": "):
            pass  # SSE comment
    if cur_data:
        events.append((cur_event, "\n".join(cur_data)))
    return events


def test_error_before_any_content_emits_both_error_event_and_content_chunk():
    """Pre-stream failure: handle_chat itself raises before any
    chunk ships. In that case the visible-display content chunk
    is still useful for clients that don't listen for events."""
    async def fail_immediately(body, bg, request_id=None):
        raise RuntimeError("upstream unreachable")

    app, _ = _make_app(fail_immediately)
    client = TestClient(app)

    resp = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    text = resp.content.decode()
    events = _parse_sse(text)

    error_events = [d for ev, d in events if ev == "error"]
    content_events = [d for ev, d in events if ev == "message"]

    assert len(error_events) == 1, f"expected one error event, got {events}"
    err = json.loads(error_events[0])
    assert "upstream unreachable" in err["error"]["message"]

    # Pre-stream: visible content chunk SHOULD be emitted (clients
    # without event-listeners need to see something).
    visible = [c for c in content_events if c != "[DONE]"]
    assert len(visible) == 1
    payload = json.loads(visible[0])
    assert "CRITICAL SERVER ERROR" in payload["choices"][0]["delta"]["content"]


def test_error_after_partial_content_does_not_double_render():
    """Mid-stream failure: real chunks have already shipped. The
    visible-display content chunk must be SUPPRESSED so the client
    doesn't render "Here is the answ" + "CRITICAL SERVER ERROR..."
    concatenated together. The error event itself still fires."""

    async def stream_then_fail(body, bg, request_id=None):
        async def chunks():
            yield b'data: {"choices":[{"delta":{"content":"Here is the ans"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"wer to your"}}]}\n\n'
            raise ConnectionResetError("mid-stream connection drop")
        return chunks(), 0, "req-1"

    app, _ = _make_app(stream_then_fail)
    client = TestClient(app)
    resp = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    text = resp.content.decode()
    events = _parse_sse(text)

    error_events = [d for ev, d in events if ev == "error"]
    assert len(error_events) == 1, "the error SSE frame must still fire"
    err = json.loads(error_events[0])
    assert "ConnectionResetError" in err["error"]["type"]

    # Real content was streamed before the failure; the visible
    # error-as-delta-content chunk must NOT appear.
    content_events = [d for ev, d in events if ev == "message" and d != "[DONE]"]
    error_content_chunks = [
        c for c in content_events
        if "CRITICAL SERVER ERROR" in c
    ]
    assert error_content_chunks == [], (
        "After real content has streamed, the error chunk must not be "
        "emitted as `delta.content` — it would concat to the visible "
        f"reply. Got: {error_content_chunks!r}"
    )

    # The original partial content must still be visible.
    assert any("Here is the ans" in c for c in content_events)
    assert any("wer to your" in c for c in content_events)
