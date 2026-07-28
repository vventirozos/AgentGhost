"""interface/server.py review fixes (2026-07-28).

Covers the four bugs + improvements from the code review:
  1. log streamer: invalid UTF-8 must not kill the loop, and the tail
     subprocess must be reaped on EVERY exit path (not just cancellation).
  2. body-size limits enforced at the ASGI layer BEFORE parsing
     (declared Content-Length and chunked bodies alike).
  3. download proxy URL-quotes the filename when rebuilding the upstream URL.
  4. malformed JSON is a 400 client error, not a 502 "upstream" error.
  5. websocket broadcast evicts stalled clients instead of freezing.
  6. lifespan holds hard refs to the background tasks and cancels them.
  7. janitor stamps missing finished_at (setdefault) so done tasks expire.
  8. cancel wakes parked readers and stamps finished_at immediately.
  9. global buffer ceiling across all chat tasks.
 10. upstream 4xx/5xx error frames include a body snippet.
 11. tts passes through the Pi's content-type.
 12. /ws evicts the socket on ANY exit path.
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# interface.server raises at import unless GHOST_API_KEY is set.
os.environ.setdefault("GHOST_API_KEY", "test-ghost-key")

import interface.server as server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _FakeStdout:
    def __init__(self, items):
        self._items = list(items)

    async def readline(self):
        item = self._items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeProc:
    def __init__(self, items):
        self.stdout = _FakeStdout(items)
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    async def wait(self):
        return 0


class _FakeWS:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.messages = []
        self.closed = False

    async def send_text(self, message):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.messages.append(message)

    async def close(self, code=1000):
        self.closed = True


def _patch_tail(monkeypatch, fake_proc):
    async def fake_exec(*args, **kwargs):
        return fake_proc
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)


def _fake_stream_client(chunks=None, status_code=200, raise_exc=None):
    """Build a fake pooled client whose .stream() yields `chunks`."""
    fake_resp = MagicMock()
    fake_resp.status_code = status_code
    if raise_exc is not None:
        fake_resp.raise_for_status = MagicMock(side_effect=raise_exc)
    else:
        fake_resp.raise_for_status = MagicMock()

    async def _aiter(*args, **kwargs):
        for c in (chunks or []):
            yield c
    fake_resp.aiter_bytes = MagicMock(side_effect=_aiter)

    ctx = AsyncMock()
    ctx.__aenter__.return_value = fake_resp
    ctx.__aexit__.return_value = None

    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=ctx)
    return fake_client


async def _drain(resp):
    received = []
    async for item in resp.body_iterator:
        received.append(item)
    return b"".join(received)


# ---------------------------------------------------------------------------
# 1. log streamer: decode resilience + tail reaping on every exit path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_utf8_log_line_is_broadcast_not_fatal(monkeypatch):
    fake = _FakeProc([b"\xff\xfeERROR boom\n", b""])
    _patch_tail(monkeypatch, fake)
    ws = _FakeWS()
    server.connected_websockets.add(ws)
    try:
        await server._log_streamer_once()
    finally:
        server.connected_websockets.discard(ws)

    assert ws.messages, "invalid-UTF8 line was dropped instead of replaced"
    payload = json.loads(ws.messages[0])
    assert "�" in payload["content"]      # replacement chars, no raise
    assert payload["is_error"] is True         # "ERROR" still detected
    assert fake.terminated                     # reaped on normal EOF too


@pytest.mark.asyncio
async def test_tail_is_reaped_when_read_loop_errors(monkeypatch):
    """Cleanup used to live only in the CancelledError handler: any other
    escape leaked a live tail while the restart loop spawned another."""
    fake = _FakeProc([RuntimeError("reader exploded")])
    _patch_tail(monkeypatch, fake)
    with pytest.raises(RuntimeError):
        await server._log_streamer_once()
    assert fake.terminated, "tail subprocess leaked on non-cancel error"


@pytest.mark.asyncio
async def test_broadcast_evicts_stalled_client_and_delivers_to_others(monkeypatch):
    fake = _FakeProc([b"hello from the log\n", b""])
    _patch_tail(monkeypatch, fake)
    monkeypatch.setattr(server, "WS_SEND_TIMEOUT_S", 0.05)

    slow = _FakeWS(delay=5.0)
    fast = _FakeWS()
    server.connected_websockets.update({slow, fast})
    try:
        await server._log_streamer_once()
        assert fast.messages, "healthy client starved by a stalled one"
        assert slow not in server.connected_websockets
        assert slow.closed, "stalled socket not closed after eviction"
    finally:
        server.connected_websockets.discard(slow)
        server.connected_websockets.discard(fast)


# ---------------------------------------------------------------------------
# 2. body-size limits at the ASGI layer
# ---------------------------------------------------------------------------

def test_declared_oversized_json_body_is_413(monkeypatch):
    monkeypatch.setattr(server, "MAX_JSON_BYTES", 64)
    client = TestClient(server.app)
    r = client.post(
        "/api/chat",
        content=b"x" * 200,
        headers={"Content-Type": "application/json",
                 "X-Ghost-Key": server.GHOST_API_KEY},
    )
    assert r.status_code == 413


def test_declared_oversized_upload_is_413(monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 16)
    monkeypatch.setattr(server, "_UPLOAD_CAP_SLACK_BYTES", 0)
    client = TestClient(server.app)
    r = client.post(
        "/api/upload",
        files={"file": ("a.bin", b"A" * 1024, "application/octet-stream")},
        headers={"X-Ghost-Key": server.GHOST_API_KEY},
    )
    assert r.status_code == 413


@pytest.mark.asyncio
async def test_chunked_body_without_content_length_is_capped(monkeypatch):
    """Chunked bodies declare no Content-Length; the middleware must count
    the bytes as they arrive and turn overflow into a 413."""
    monkeypatch.setattr(server, "MAX_JSON_BYTES", 64)

    sent = []

    async def send(message):
        sent.append(message)

    body_chunks = [b"A" * 40, b"B" * 40, b""]
    state = {"i": 0}

    async def receive():
        chunk = body_chunks[state["i"]]
        state["i"] += 1
        return {"type": "http.request", "body": chunk, "more_body": bool(chunk)}

    async def inner_app(scope, receive_, send_):
        # Simulate FastAPI reading the whole body before responding.
        while True:
            message = await receive_()
            if not message.get("more_body"):
                break
        await send_({"type": "http.response.start", "status": 200, "headers": []})
        await send_({"type": "http.response.body", "body": b"ok"})

    mw = server.BodySizeLimitMiddleware(inner_app)
    scope = {"type": "http", "method": "POST", "path": "/api/chat", "headers": []}
    await mw(scope, receive, send)

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413


# ---------------------------------------------------------------------------
# 4. malformed JSON → 400, not 502
# ---------------------------------------------------------------------------

def test_malformed_json_chat_body_is_400():
    client = TestClient(server.app)
    r = client.post(
        "/api/chat",
        content=b"{definitely not json",
        headers={"Content-Type": "application/json",
                 "X-Ghost-Key": server.GHOST_API_KEY},
    )
    assert r.status_code == 400
    # Interface error shape, not FastAPI's {"detail": ...} — the bundled
    # client only reads `.error`.
    assert "error" in r.json()


def test_malformed_json_tts_body_is_400():
    client = TestClient(server.app)
    r = client.post(
        "/api/tts",
        content=b"{nope",
        headers={"Content-Type": "application/json",
                 "X-Ghost-Key": server.GHOST_API_KEY},
    )
    assert r.status_code == 400
    assert "error" in r.json()


# ---------------------------------------------------------------------------
# 3. download proxy quotes the filename
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_download_filename_is_url_quoted():
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.headers = {}

    async def _aiter(*args, **kwargs):
        yield b"data"
    fake_resp.aiter_bytes = MagicMock(side_effect=_aiter)
    fake_resp.aclose = AsyncMock()

    fake_client = MagicMock()
    fake_client.build_request = MagicMock(return_value=MagicMock())
    fake_client.send = AsyncMock(return_value=fake_resp)

    with patch.object(server, "_get_http_client", return_value=fake_client):
        resp = await server.download_proxy("report?v=2.pdf")
        async for _ in resp.body_iterator:
            pass

    url = fake_client.build_request.call_args[0][1]
    assert url.endswith("/api/download/report%3Fv%3D2.pdf")
    assert "?" not in url


# ---------------------------------------------------------------------------
# 7. janitor sweep: setdefault stamps finished_at; hard cap evicts done first
# ---------------------------------------------------------------------------

def test_sweep_stamps_missing_finished_at_then_expires():
    saved = dict(server.active_chat_tasks)
    server.active_chat_tasks.clear()
    try:
        server.active_chat_tasks["t1"] = {"done": True, "buffer": [], "buffer_size": 0}
        now = 1000.0
        server._sweep_active_chat_tasks(now)
        # First sweep stamps (does NOT evict, and does NOT keep resetting
        # the clock like the old `or now` did).
        assert server.active_chat_tasks["t1"]["finished_at"] == now
        server._sweep_active_chat_tasks(now + server.ACTIVE_TASK_TTL_SECONDS + 1)
        assert "t1" not in server.active_chat_tasks
    finally:
        server.active_chat_tasks.clear()
        server.active_chat_tasks.update(saved)


def test_sweep_hard_cap_evicts_done_before_live():
    saved = dict(server.active_chat_tasks)
    server.active_chat_tasks.clear()
    try:
        now = 5000.0
        # Oldest entry is LIVE; overflow must come out of the done pool.
        server.active_chat_tasks["live"] = {
            "done": False, "background_task": None,
            "new_data_event": asyncio.Event(), "buffer": [], "buffer_size": 0,
        }
        for i in range(server.ACTIVE_TASK_HARD_CAP + 5):
            server.active_chat_tasks[f"done-{i}"] = {
                "done": True, "finished_at": now, "buffer": [], "buffer_size": 0,
            }
        server._sweep_active_chat_tasks(now)
        assert len(server.active_chat_tasks) <= server.ACTIVE_TASK_HARD_CAP
        assert "live" in server.active_chat_tasks
    finally:
        server.active_chat_tasks.clear()
        server.active_chat_tasks.update(saved)


# ---------------------------------------------------------------------------
# 8. cancel wakes parked readers and stamps finished_at
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_wakes_reader_and_stamps_finished_at():
    tid = "cancel-test-task"
    ev = asyncio.Event()
    bg = MagicMock()
    server.active_chat_tasks[tid] = {
        "done": False, "background_task": bg, "new_data_event": ev,
        "buffer": [], "buffer_size": 0,
    }
    try:
        result = await server.chat_cancel_proxy(tid)
        assert result == {"status": "cancelled"}
        bg.cancel.assert_called_once()
        t = server.active_chat_tasks[tid]
        assert t["done"] is True
        assert "finished_at" in t          # janitor TTL can now expire it
        assert ev.is_set()                 # parked readers wake immediately
    finally:
        server.active_chat_tasks.pop(tid, None)


# ---------------------------------------------------------------------------
# 9. global buffer ceiling across all tasks
# ---------------------------------------------------------------------------

def test_total_buffered_bytes_sums_all_tasks():
    saved = dict(server.active_chat_tasks)
    server.active_chat_tasks.clear()
    try:
        server.active_chat_tasks["a"] = {"buffer_size": 10}
        server.active_chat_tasks["b"] = {"buffer_size": 32}
        assert server._total_buffered_bytes() == 42
    finally:
        server.active_chat_tasks.clear()
        server.active_chat_tasks.update(saved)


@pytest.mark.asyncio
async def test_global_buffer_cap_truncates_stream(monkeypatch):
    saved = dict(server.active_chat_tasks)
    server.active_chat_tasks.clear()
    try:
        monkeypatch.setenv("GHOST_INTERFACE_TOTAL_STREAM_CAP", "16")
        fake_client = _fake_stream_client(chunks=[b"A" * 8, b"B" * 8, b"C" * 8])
        req = MagicMock()
        req.json = AsyncMock(return_value={"stream": True, "messages": []})

        with patch.object(server, "_get_http_client", return_value=fake_client):
            resp = await server.chat_proxy(req)
            combined = await _drain(resp)

        assert b"BufferCapExceeded" in combined
        assert b"global buffer cap exceeded" in combined
        t = next(iter(server.active_chat_tasks.values()))
        assert t["truncated"] is True
        assert t["truncated_reason"] == "global buffer cap exceeded"
    finally:
        server.active_chat_tasks.clear()
        server.active_chat_tasks.update(saved)


@pytest.mark.asyncio
async def test_global_cap_reclaims_done_buffers_before_truncating(monkeypatch):
    """A finished stream's resume buffer must not brown-out new streams:
    the worker evicts done buffers to make room and only truncates if live
    streams alone exceed the ceiling."""
    saved = dict(server.active_chat_tasks)
    server.active_chat_tasks.clear()
    try:
        monkeypatch.setenv("GHOST_INTERFACE_TOTAL_STREAM_CAP", "16")
        # A completed stream is hogging the entire ceiling.
        server.active_chat_tasks["old-done"] = {
            "done": True, "finished_at": 1.0,
            "buffer": [b"X" * 16], "buffer_size": 16,
        }
        fake_client = _fake_stream_client(chunks=[b"A" * 8])
        req = MagicMock()
        req.json = AsyncMock(return_value={"stream": True, "messages": []})

        with patch.object(server, "_get_http_client", return_value=fake_client):
            resp = await server.chat_proxy(req)
            combined = await _drain(resp)

        assert b"A" * 8 in combined
        assert b"BufferCapExceeded" not in combined
        assert "old-done" not in server.active_chat_tasks  # buffer reclaimed
    finally:
        server.active_chat_tasks.clear()
        server.active_chat_tasks.update(saved)


# ---------------------------------------------------------------------------
# 10. upstream 4xx/5xx error frame carries a body snippet
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upstream_error_frame_includes_status_and_body_snippet():
    saved = dict(server.active_chat_tasks)
    server.active_chat_tasks.clear()
    try:
        err = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=MagicMock())
        fake_client = _fake_stream_client(
            chunks=[b'{"detail": "upstream exploded"}'],
            status_code=500, raise_exc=err)
        req = MagicMock()
        req.json = AsyncMock(return_value={"stream": True, "messages": []})

        with patch.object(server, "_get_http_client", return_value=fake_client):
            resp = await server.chat_proxy(req)
            combined = await _drain(resp)

        assert b"HTTP 500" in combined
        assert b"upstream exploded" in combined
    finally:
        server.active_chat_tasks.clear()
        server.active_chat_tasks.update(saved)


@pytest.mark.asyncio
async def test_upstream_error_snippet_read_is_time_bounded(monkeypatch):
    """A drip-feeding failed upstream must not hold the worker: the snippet
    read is wall-clock bounded, and whatever arrived before the deadline
    still makes it into the error frame."""
    saved = dict(server.active_chat_tasks)
    server.active_chat_tasks.clear()
    try:
        monkeypatch.setattr(server, "UPSTREAM_ERROR_SNIPPET_TIMEOUT_S", 0.05)

        err = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=MagicMock())
        fake_resp = MagicMock()
        fake_resp.status_code = 503
        fake_resp.raise_for_status = MagicMock(side_effect=err)

        async def _dripping_aiter(*args, **kwargs):
            yield b"partial detail"
            await asyncio.sleep(30)   # never finishes on its own
            yield b"never delivered"
        fake_resp.aiter_bytes = MagicMock(side_effect=_dripping_aiter)

        ctx = AsyncMock()
        ctx.__aenter__.return_value = fake_resp
        ctx.__aexit__.return_value = None
        fake_client = MagicMock()
        fake_client.stream = MagicMock(return_value=ctx)

        req = MagicMock()
        req.json = AsyncMock(return_value={"stream": True, "messages": []})

        with patch.object(server, "_get_http_client", return_value=fake_client):
            resp = await server.chat_proxy(req)
            combined = await asyncio.wait_for(_drain(resp), timeout=5.0)

        assert b"HTTP 503" in combined
        assert b"partial detail" in combined
    finally:
        server.active_chat_tasks.clear()
        server.active_chat_tasks.update(saved)


# ---------------------------------------------------------------------------
# 11. tts content-type passthrough
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tts_passes_through_pi_content_type():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "audio/ogg"}

    async def _aiter(*args, **kwargs):
        yield b"OggS"
    fake_resp.aiter_bytes = MagicMock(side_effect=_aiter)
    fake_resp.aclose = AsyncMock()

    fake_client = MagicMock()
    fake_client.build_request = MagicMock(return_value=MagicMock())
    fake_client.send = AsyncMock(return_value=fake_resp)

    req = MagicMock()
    req.json = AsyncMock(return_value={"text": "hello"})

    with patch.object(server, "_get_http_client", return_value=fake_client):
        resp = await server.tts_proxy(req)
        async for _ in resp.body_iterator:
            pass

    assert resp.media_type == "audio/ogg"


# ---------------------------------------------------------------------------
# 6. lifespan holds hard refs to background tasks and cancels them
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lifespan_holds_and_cancels_background_tasks(monkeypatch):
    async def stub():
        await asyncio.sleep(60)

    monkeypatch.setattr(server, "log_streamer", stub)
    monkeypatch.setattr(server, "_active_chat_tasks_janitor", stub)

    async with server._lifespan(server.app):
        await asyncio.sleep(0)
        assert len(server._BACKGROUND_TASKS) == 2
        assert all(not t.done() for t in server._BACKGROUND_TASKS)
        refs = list(server._BACKGROUND_TASKS)

    assert server._BACKGROUND_TASKS == []      # cleared for a clean restart
    assert all(t.cancelled() for t in refs)    # actually torn down


# ---------------------------------------------------------------------------
# 12. /ws evicts the socket on ANY exit path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ws_endpoint_discards_socket_on_unexpected_error():
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.receive_text = AsyncMock(side_effect=RuntimeError("boom"))
    ws.close = AsyncMock()

    with pytest.raises(RuntimeError):
        await server.websocket_endpoint(ws, key=server.GHOST_API_KEY)

    assert ws not in server.connected_websockets
