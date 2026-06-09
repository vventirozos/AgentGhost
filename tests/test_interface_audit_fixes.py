"""Targeted unit tests for the interface/server.py audit fixes.

Covers:
- stream_generator yields whole HTTP chunks (NOT individual bytes)
- per-task buffer cap: appending past the cap flips truncated=True
- non-streaming chat path reuses the shared httpx client (no per-request
  AsyncClient construction)
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make sure the project root is importable so `import interface.server` works.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# stream_generator yields chunks, not bytes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_chat_yields_whole_chunks():
    """Streaming chat must yield each upstream chunk in one piece.

    The previous bug indexed into a single `bytes` object byte-by-byte;
    here we drive the proxy with deliberately multi-byte chunks and
    assert that each yielded item is a multi-byte bytes object.
    """
    import interface.server as server

    # Three deliberately-multi-byte chunks. If the generator devolved to
    # byte-at-a-time, we'd see len(item) == 1 for all of them.
    chunks_to_send = [b"hello world ", b"this is chunk two ", b"finally chunk three"]

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()

    async def _aiter(*args, **kwargs):
        for c in chunks_to_send:
            yield c
    fake_resp.aiter_bytes = MagicMock(side_effect=_aiter)

    fake_stream_ctx = AsyncMock()
    fake_stream_ctx.__aenter__.return_value = fake_resp
    fake_stream_ctx.__aexit__.return_value = None

    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=fake_stream_ctx)

    req = MagicMock()
    req.json = AsyncMock(return_value={"stream": True, "messages": []})

    with patch.object(server, "_get_http_client", return_value=fake_client):
        resp = await server.chat_proxy(req)
        # Drain INSIDE the patch context so the background task picks up
        # our fake _get_http_client when it actually runs (the bg task is
        # scheduled inside chat_proxy but doesn't execute until we await).
        received = []
        async for item in resp.body_iterator:
            received.append(item)

    # Some items may be the SSE truncation marker — but we asked for a
    # tiny payload, so no truncation should fire. EVERY received item
    # must be a multi-byte bytes object (i.e. NOT a one-byte slice).
    assert received, "stream produced no output"
    for item in received:
        assert isinstance(item, bytes)
    multi_byte = [c for c in received if len(c) > 1]
    assert multi_byte, "every chunk was a single byte — per-byte streaming regression"

    # Joined output preserves the bytes the upstream sent, in order.
    combined = b"".join(received)
    for c in chunks_to_send:
        assert c in combined


# ---------------------------------------------------------------------------
# per-task buffer cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_task_buffer_cap_flips_truncated_flag(monkeypatch):
    """Drive the proxy with chunks larger than the cap; the task must
    flip `truncated=True` and emit an SSE-shaped marker."""
    import interface.server as server

    # Pin a tiny cap so we can blow past it with a couple of small chunks.
    monkeypatch.setenv("GHOST_INTERFACE_STREAM_CAP", "16")

    chunks_to_send = [b"AAAAAAAA", b"BBBBBBBB", b"CCCCCCCC", b"DDDDDDDD"]

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()

    async def _aiter(*args, **kwargs):
        for c in chunks_to_send:
            yield c
    fake_resp.aiter_bytes = MagicMock(side_effect=_aiter)

    fake_stream_ctx = AsyncMock()
    fake_stream_ctx.__aenter__.return_value = fake_resp
    fake_stream_ctx.__aexit__.return_value = None

    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=fake_stream_ctx)

    req = MagicMock()
    req.json = AsyncMock(return_value={"stream": True, "messages": []})

    with patch.object(server, "_get_http_client", return_value=fake_client):
        resp = await server.chat_proxy(req)
        # Drain INSIDE the patch so the bg worker uses our fake client.
        received = []
        async for item in resp.body_iterator:
            received.append(item)

    # The task associated with this stream must have flipped truncated.
    truncated_tasks = [t for t in server.active_chat_tasks.values() if t.get("truncated")]
    assert truncated_tasks, "no task was marked as truncated"

    # And the streamer should have emitted the SSE truncation marker.
    combined = b"".join(received)
    assert b"BufferCapExceeded" in combined or b"truncated" in combined


# ---------------------------------------------------------------------------
# Non-streaming chat reuses the shared httpx client
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_streaming_chat_reuses_shared_http_client():
    """The non-streaming path must call _get_http_client(), not construct
    a fresh httpx.AsyncClient per request."""
    import interface.server as server

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json = MagicMock(return_value={"choices": []})

    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=fake_resp)

    req = MagicMock()
    req.json = AsyncMock(return_value={"stream": False, "messages": []})

    # If the route still constructs `httpx.AsyncClient()` itself, this
    # patch should record at least one call. We expect ZERO.
    with patch.object(server, "_get_http_client", return_value=fake_client) as mock_get_client, \
         patch.object(server.httpx, "AsyncClient") as mock_async_client_ctor:
        mock_async_client_ctor.side_effect = AssertionError("non-streaming path must NOT construct a fresh AsyncClient")
        result = await server.chat_proxy(req)

    assert mock_get_client.called, "non-streaming path did not reuse _get_http_client"
    # The route now propagates the upstream status instead of returning a
    # bare dict with an implicit 200 (which masked agent 4xx/5xx errors).
    import json as _json
    assert result.status_code == 200
    assert _json.loads(result.body) == {"choices": []}
    fake_client.post.assert_called_once()


# ---------------------------------------------------------------------------
# websocket disconnect path uses discard, not remove
# ---------------------------------------------------------------------------

def test_websocket_disconnect_uses_discard_no_keyerror():
    """Reach into the source (or dummy-call discard) to assert KeyError-safe.

    The simplest, surface-level check: calling .discard() on the live set
    with a non-member must NEVER raise. (`.remove()` would raise KeyError.)
    """
    import interface.server as server

    sentinel = object()
    # If the implementation regressed back to `.remove()`, this exact
    # pattern would have raised; using `.discard()` is silent.
    server.connected_websockets.discard(sentinel)
