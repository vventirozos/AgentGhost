"""Targeted unit tests for the API audit fixes.

Covers:
- catch_all proxy: SSE-shaped responses get the right media_type and the
  X-Accel-Buffering: no header (both when the upstream Content-Type says
  text/event-stream AND when the request body opts in via {"stream": true}).
- chat stream_generator yields the SSE keep-alive comment BEFORE awaiting
  the agent's handle_chat (so headers reach the client immediately).
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure src/ is importable when tests are run from the repo root.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(method="POST", json_body=None, content_type="application/json", stream_body=False):
    """Build a MagicMock that behaves enough like fastapi.Request for our routes."""
    from fastapi import Request

    raw = b"" if json_body is None else json.dumps(json_body).encode("utf-8")

    req = MagicMock(spec=Request)
    req.method = method
    req.headers = {"content-type": content_type}
    req.body = AsyncMock(return_value=raw)

    async def _stream():
        yield raw
    req.stream = MagicMock(side_effect=_stream)

    return req


def _make_agent(response_status=200, response_headers=None, body_chunks=(b"hello",)):
    """A MagicMock agent whose llm_client.http_client behaves like httpx."""
    response_headers = response_headers or {}

    fake_resp = MagicMock()
    fake_resp.status_code = response_status
    fake_resp.headers = response_headers
    fake_resp.aclose = AsyncMock()

    async def _aiter_bytes():
        for c in body_chunks:
            yield c
    fake_resp.aiter_bytes = MagicMock(side_effect=_aiter_bytes)

    fake_http_client = MagicMock()
    fake_http_client.build_request = MagicMock(return_value=MagicMock())
    fake_http_client.send = AsyncMock(return_value=fake_resp)

    agent = MagicMock()
    agent.context.llm_client.http_client = fake_http_client

    # The catch_all gets the agent via request.app.state.agent
    return agent


# ---------------------------------------------------------------------------
# catch_all SSE detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catchall_marks_sse_when_upstream_is_event_stream():
    from ghost_agent.api.routes import catch_all

    req = _make_request(method="POST", json_body={"prompt": "hi"})
    agent = _make_agent(
        response_status=200,
        response_headers={"content-type": "text/event-stream; charset=utf-8"},
        body_chunks=(b"data: hello\n\n",),
    )
    req.app = MagicMock()
    req.app.state.agent = agent

    # Patch get_agent so it sees our agent without going through DI.
    with patch("ghost_agent.api.routes.get_agent", return_value=agent):
        resp = await catch_all(req, "v1/chat/completions")

    assert resp.media_type == "text/event-stream"
    assert resp.headers.get("x-accel-buffering", "").lower() == "no"


@pytest.mark.asyncio
async def test_catchall_marks_sse_when_request_body_opts_in():
    from ghost_agent.api.routes import catch_all

    # Upstream Content-Type is plain JSON, but the request body asks for
    # streaming. The catch-all must still flag SSE in the response.
    req = _make_request(method="POST", json_body={"stream": True, "prompt": "x"})
    agent = _make_agent(
        response_status=200,
        response_headers={"content-type": "application/json"},
        body_chunks=(b"{\"ok\": true}",),
    )

    with patch("ghost_agent.api.routes.get_agent", return_value=agent):
        resp = await catch_all(req, "anything")

    assert resp.media_type == "text/event-stream"
    assert resp.headers.get("x-accel-buffering", "").lower() == "no"


@pytest.mark.asyncio
async def test_catchall_non_sse_request_keeps_upstream_content_type():
    from ghost_agent.api.routes import catch_all

    req = _make_request(method="POST", json_body={"stream": False, "prompt": "x"})
    agent = _make_agent(
        response_status=200,
        response_headers={"content-type": "application/json"},
        body_chunks=(b"{\"ok\": true}",),
    )

    with patch("ghost_agent.api.routes.get_agent", return_value=agent):
        resp = await catch_all(req, "anything")

    # NOT marked as event-stream.
    assert "event-stream" not in (resp.media_type or "").lower()
    # And the no-buffering header should not be present.
    assert "x-accel-buffering" not in {k.lower() for k in resp.headers.keys()}


# ---------------------------------------------------------------------------
# chat stream_generator flushes a keep-alive BEFORE awaiting handle_chat
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_stream_yields_keepalive_before_handle_chat_completes():
    """The first byte the client sees MUST arrive before handle_chat awaits.

    We model handle_chat as a long-blocking coroutine; if the generator
    were awaiting it before yielding, asyncio would block the very first
    `__anext__` call. Instead, we expect to receive an SSE comment frame
    on the first iteration WITHOUT handle_chat ever finishing.
    """
    from ghost_agent.api.routes import chat_proxy

    handle_chat_started = []
    handle_chat_finished = []

    async def fake_handle_chat(*args, **kwargs):
        handle_chat_started.append(True)
        # Block effectively forever — the first generator iteration must
        # not depend on this returning.
        import asyncio as _asyncio
        await _asyncio.sleep(60)
        handle_chat_finished.append(True)
        return ("never", 0, "noreq")

    agent = MagicMock()
    agent.handle_chat = fake_handle_chat
    agent.context.args.model = "test-model"

    req = _make_request(method="POST", json_body={"stream": True, "messages": []})
    req.app = MagicMock()
    req.app.state.agent = agent
    # `chat_proxy` calls `await request.json()` directly.
    req.json = AsyncMock(return_value={"stream": True, "messages": []})

    bg_tasks = MagicMock()

    with patch("ghost_agent.api.routes.get_agent", return_value=agent):
        resp = await chat_proxy(req, bg_tasks)

    # Grab the FIRST chunk — must be the SSE comment, even though
    # handle_chat is still blocked.
    body_iter = resp.body_iterator
    first = await body_iter.__anext__()

    assert first.startswith(b":"), f"expected SSE comment first, got {first!r}"
    assert b"\n\n" in first
    assert handle_chat_finished == []
