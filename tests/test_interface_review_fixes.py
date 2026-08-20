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
 11. tts returns locally-synthesised WAV audio (was: Pi content-type passthrough).
 12. /ws evicts the socket on ANY exit path.
"""
import asyncio
import json
import os
import time
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
async def test_DELETE_bodies_are_capped_too(monkeypatch):
    """`_proxy_agent_api` explicitly reads DELETE bodies
    (`method in ("POST", "DELETE")`), but the middleware only gated POST /
    PUT / PATCH — so the one route that reads a DELETE body slurped it
    unbounded into RAM while the identical POST was capped (R2 lens A)."""
    monkeypatch.setattr(server, "MAX_JSON_BYTES", 64)
    sent, reads = [], []

    async def send(message):
        sent.append(message)

    async def receive():
        reads.append(1)
        return {"type": "http.request", "body": b"A" * 5000, "more_body": False}

    async def inner_app(scope, receive_, send_):  # pragma: no cover
        raise AssertionError("an oversized DELETE body reached the app")

    mw = server.BodySizeLimitMiddleware(inner_app)
    scope = {"type": "http", "method": "DELETE", "path": "/api/sessions/abc",
             "headers": [(b"content-length", b"5000")]}
    await mw(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413
    assert reads == [], "the DELETE body was read before being refused"


@pytest.mark.asyncio
async def test_an_upstream_4xx_keeps_its_STATUS_and_BODY(monkeypatch):
    """`raise_for_status()` inside a blanket `except Exception` collapsed
    every upstream status into a 502 and discarded the body: "unsupported
    file type: .exe" reached the operator as `Client error '422 …' for url
    '…'`, and a permanent 404 was delivered as a RETRYABLE 502 (R2 lens A,
    with three live log lines to match)."""
    import httpx
    resp = httpx.Response(
        422, json={"error": "unsupported file type: .exe"},
        request=httpx.Request("POST", "http://localhost:8000/api/upload"))
    exc = httpx.HTTPStatusError("Client error '422 Unprocessable Entity'",
                                request=resp.request, response=resp)
    out = server._upstream_failure("Upload proxy error", exc)
    assert out.status_code == 422, "a 4xx is still reported as a retryable 502"
    assert b"unsupported file type" in out.body, out.body


def test_an_empty_exception_message_never_becomes_the_diagnosis():
    """`str(e)` is "" for httpx's whole timeout family, and every consumer
    of `task["error"]` tests it for TRUTHINESS — so a stalled agent produced
    no error frame at all and the client persisted its partial text as a
    finished reply. 16 causeless `… proxy error:` lines are already in the
    live log, one of them a dropped human feedback label."""
    import httpx
    assert str(httpx.ReadTimeout("")) == "", (
        "the premise changed — re-check the consumers")
    out = server._upstream_failure("TTS proxy error", httpx.ReadTimeout(""))
    body = out.body.decode()
    assert "ReadTimeout" in body, f"empty diagnosis reached the client: {body}"
    assert out.status_code == 502


@pytest.mark.asyncio
async def test_declared_length_rejects_WITHOUT_reading_the_body(monkeypatch):
    """The two size guards are not interchangeable, and the tests above
    could not tell them apart: delete the declared-Content-Length branch
    and `counting_receive` still produces a 413, so both status assertions
    stay green while the middleware has started SLURPING the multi-GB body
    it exists to refuse (review R1 M12c).

    What separates them is observable: on a declared overflow the body must
    never be read at all. Pin that, not the status code."""
    monkeypatch.setattr(server, "MAX_JSON_BYTES", 64)
    sent, reads = [], []

    async def send(message):
        sent.append(message)

    async def receive():
        reads.append(1)
        return {"type": "http.request", "body": b"A" * 200, "more_body": False}

    async def inner_app(scope, receive_, send_):  # pragma: no cover
        raise AssertionError("the request reached the app despite overflowing")

    mw = server.BodySizeLimitMiddleware(inner_app)
    scope = {"type": "http", "method": "POST", "path": "/api/chat",
             "headers": [(b"content-length", b"200")]}
    await mw(scope, receive, send)

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413
    assert reads == [], (
        f"the body was read {len(reads)}x before the declared-length "
        "rejection — the up-front branch is gone and only the byte counter "
        "is left")

    # And the upload path must get the UPLOAD cap, not the JSON one: a
    # declared 1 MB upload is legal, and rejecting it here would break every
    # real file send while every assertion above stayed green.
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 1024 * 1024)
    monkeypatch.setattr(server, "_UPLOAD_CAP_SLACK_BYTES", 0)
    passed = []

    async def ok_app(scope_, receive_, send_):
        passed.append(scope_["path"])
        await send_({"type": "http.response.start", "status": 200, "headers": []})
        await send_({"type": "http.response.body", "body": b"ok"})

    up_scope = {"type": "http", "method": "POST",
                "path": next(iter(server._UPLOAD_PATHS)),
                "headers": [(b"content-length", b"500000")]}
    await server.BodySizeLimitMiddleware(ok_app)(up_scope, receive, send)
    assert passed, "a legal-sized upload was rejected by the JSON cap"


def test_chunked_overflow_is_413_THROUGH_THE_REAL_STACK(monkeypatch):
    """⚠ The class's central design claim, previously untested.

    `_BodyTooLarge` derives from BaseException specifically so it sails
    through Starlette's `ExceptionMiddleware` (`except Exception`) and
    FastAPI's body parsing, reaching this middleware's own handler. The
    sibling test below hand-rolls an `inner_app` that catches nothing — so
    the layer BaseException exists to bypass is never in the picture, and
    changing the base class to `Exception` survived all 390 server-touching
    tests (R3 lens A).

    Through the real stack that mutation turns a chunked overflow into
    `502 {"error": "Chat proxy error: 64"}`, and a chunked multipart into
    `400 {"detail": "There was an error parsing the body"}` — neither of
    which is a 413."""
    monkeypatch.setattr(server, "MAX_JSON_BYTES", 64)
    client = TestClient(server.app)

    def _chunks():
        yield b"A" * 40
        yield b"B" * 40

    r = client.post("/api/chat", content=_chunks(),
                    headers={"Content-Type": "application/json",
                             "X-Ghost-Key": server.GHOST_API_KEY})
    assert r.status_code == 413, (
        f"chunked overflow returned {r.status_code} {r.text[:160]!r} — the "
        "BaseException escape hatch is not working through FastAPI")


def test_body_too_large_is_a_BaseException_on_purpose():
    """Stated directly, because the behavioural test above can only fail
    AFTER someone changes it, and the reason is easy to lose."""
    assert issubclass(server._BodyTooLarge, BaseException)
    assert not issubclass(server._BodyTooLarge, Exception), (
        "as an Exception it is swallowed by Starlette's ExceptionMiddleware "
        "and FastAPI's body parser before the middleware can answer 413")


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
# 11. tts returns locally-synthesised WAV
#
# REPLACED 2026-08-02. The original test asserted that /api/tts passed the
# Raspberry-Pi voice server's content-type through (it could serve ogg/mp3).
# That host no longer exists — `raspberrypi.local` doesn't resolve and the
# launcher's `disorder` override listens on nothing — so TTS now synthesises
# locally with macOS `say` and always returns WAV. The passthrough behaviour
# it guarded is gone by design, not by regression.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tts_returns_wav_from_local_synthesis():
    from interface import voice

    req = MagicMock()
    req.json = AsyncMock(return_value={"text": "hello"})

    with patch.object(voice, "synthesize", AsyncMock(return_value=b"RIFF....WAVE")) as synth:
        resp = await server.tts_proxy(req)

    synth.assert_awaited_once_with("hello")
    assert resp.media_type == "audio/wav"
    assert resp.body == b"RIFF....WAVE"


@pytest.mark.asyncio
async def test_tts_voice_error_keeps_its_own_status():
    """A VoiceError must surface its own status, not a blanket 502 — an
    empty-text request is the caller's fault and should read as one."""
    from interface import voice

    req = MagicMock()
    req.json = AsyncMock(return_value={"text": ""})

    with patch.object(voice, "synthesize",
                      AsyncMock(side_effect=voice.VoiceError("No text to speak.", 400))):
        resp = await server.tts_proxy(req)

    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 6. lifespan holds hard refs to background tasks and cancels them
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lifespan_holds_and_cancels_background_tasks(monkeypatch):
    async def stub():
        await asyncio.sleep(60)

    monkeypatch.setattr(server, "log_streamer", stub)
    monkeypatch.setattr(server, "_active_chat_tasks_janitor", stub)
    monkeypatch.setattr(server, "_notify_push_poller", stub)

    async with server._lifespan(server.app):
        await asyncio.sleep(0)
        assert len(server._BACKGROUND_TASKS) == 3
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


# ---------------------------------------------------------------------------
# Round 3 (§4BU): the reporting paths, again — plus the fixes that fixed
# nothing.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_allowlist_proxy_reports_a_TIMEOUT_with_a_cause():
    """`f"{e or type(e).__name__}"` was a NO-OP: an exception OBJECT is
    always truthy, so the `or` never fired and the f-string still rendered
    an empty `str(e)`. The R2 fix looked right and changed nothing, on the
    one proxy family that was never converted to `_upstream_failure` — and
    it carries /api/turn/cancel and /api/feedback, i.e. Stop and the human
    label channel (R3 lens A)."""
    import httpx
    from unittest.mock import AsyncMock, MagicMock, patch

    fake = MagicMock()
    fake.request = AsyncMock(side_effect=httpx.ReadTimeout(""))
    req = MagicMock()
    req.query_params = {}
    req.headers = {}
    req.body = AsyncMock(return_value=b"")
    with patch.object(server, "_get_http_client", return_value=fake):
        out = await server._proxy_agent_api(req, "POST", "/api/feedback")
    body = out.body.decode()
    assert out.status_code == 502
    assert "ReadTimeout" in body, (
        f"the client is told the proxy failed, with no cause: {body}")


def test_env_numbers_never_crash_the_process_at_import():
    """A typo'd or EMPTY numeric env made `float()` raise during module
    import; under a KeepAlive LaunchDaemon that is a permanent silent
    crash-loop with no endpoint left to report it (R3 lens A)."""
    import os
    for raw, want in (("abc", 1800.0), ("", 1800.0), ("  ", 1800.0),
                      ("2.5", 2.5)):
        os.environ["GHOST_TEST_NUM"] = raw
        try:
            assert server._env_num("GHOST_TEST_NUM", 1800.0) == want, raw
        finally:
            os.environ.pop("GHOST_TEST_NUM", None)
    assert server._env_num("GHOST_TEST_ABSENT", 7, int) == 7
    # And the constants themselves must be routed through it.
    import inspect
    src = inspect.getsource(server)
    for name in ("GHOST_CHAT_TIMEOUT", "GHOST_CHAT_CONNECT_TIMEOUT",
                 "GHOST_WS_SEND_TIMEOUT"):
        assert f'float(os.environ.get("{name}"' not in src, (
            f"{name} is back on an unguarded float() at import")


def test_the_active_task_cap_is_enforced_at_INSERT():
    """Enforcement was sweep-only (every 60s), so the dict could reach any
    size in between — measured 3000 entries. Each holds an open stream on
    the shared httpx client, whose pool is 100 with a 1800s pool timeout, so
    the 101st chat parks for THIRTY MINUTES (R3 lens A)."""
    import inspect
    assert "_enforce_active_task_cap()" in inspect.getsource(server.chat_proxy), (
        "the cap is only applied by the 60s sweep")
    server.active_chat_tasks.clear()
    try:
        for i in range(server.ACTIVE_TASK_HARD_CAP + 25):
            server.active_chat_tasks[f"t{i}"] = {"done": True, "buffer": []}
        server._enforce_active_task_cap()
        assert len(server.active_chat_tasks) <= server.ACTIVE_TASK_HARD_CAP
    finally:
        server.active_chat_tasks.clear()


def test_workspace_save_gets_a_bigger_but_BOUNDED_json_cap(monkeypatch):
    """A workspace you can LOAD you must be able to SAVE — but save takes a
    JSON body (`{chat_history: [...]}`; the zip is the RESPONSE), so the R3
    fix of adding it to `_UPLOAD_PATHS` handed `request.json()` a ~101 MB
    input, ~2.3x that in peak RAM, re-serialised to an agent with no body cap
    of its own (R4 lens A).

    Behaviour, not membership: a body over the workspace ceiling must 413,
    and one under it must reach the handler."""
    assert "/api/workspace/save" not in server._UPLOAD_PATHS, (
        "a JSON route is back on the multipart upload ceiling")
    cap = server._JSON_CAP_OVERRIDES["/api/workspace/save"]()
    assert server.MAX_JSON_BYTES < cap < server.MAX_UPLOAD_BYTES, (
        f"the workspace cap ({cap}) is not between the chat cap and the "
        f"upload cap — it is either useless or unbounded")

    monkeypatch.setitem(server._JSON_CAP_OVERRIDES,
                        "/api/workspace/save", lambda: 64)
    client = TestClient(server.app)
    r = client.post("/api/workspace/save", content=b"x" * 200,
                    headers={"Content-Type": "application/json",
                             "X-Ghost-Key": server.GHOST_API_KEY})
    assert r.status_code == 413, (
        f"an oversized workspace save was accepted: {r.status_code}")
    # And a chat body is still on the SMALL cap, i.e. the override is scoped.
    monkeypatch.setattr(server, "MAX_JSON_BYTES", 64)
    r2 = client.post("/api/chat", content=b"x" * 200,
                     headers={"Content-Type": "application/json",
                              "X-Ghost-Key": server.GHOST_API_KEY})
    assert r2.status_code == 413


@pytest.mark.asyncio
async def test_shutdown_cancels_in_flight_chat_workers():
    """In-flight workers live only in `active_chat_tasks[tid]["background_task"]`
    and were never cancelled: the shared httpx client closed underneath them,
    and their `finally` scheduled a reply-ready push AFTER the probe set was
    cleared — producing the "Task was destroyed but it is pending" the
    lifespan says it prevents (R3 lens A).

    Also pins the loop-safety the first version of that fix lacked: a task
    left behind by an earlier request may belong to a CLOSED loop, and
    cancelling it raised straight out of shutdown (this test passed alone and
    failed after its neighbours ran — the full suite caught it)."""
    started = asyncio.Event()

    async def _worker():
        started.set()
        await asyncio.sleep(3600)

    task = asyncio.ensure_future(_worker())
    await started.wait()
    server.active_chat_tasks["t-shutdown"] = {"background_task": task,
                                              "done": False, "buffer": []}
    # A stale entry whose "task" is not a Task at all must not break teardown.
    server.active_chat_tasks["t-junk"] = {"background_task": object(),
                                          "done": True, "buffer": []}
    try:
        async with server._lifespan(server.app):
            pass
        assert task.cancelled() or task.done(), (
            "an in-flight chat worker survived shutdown")
    finally:
        server.active_chat_tasks.pop("t-shutdown", None)
        server.active_chat_tasks.pop("t-junk", None)
        task.cancel()


# ---------------------------------------------------------------------------
# R4 lens A: the gate with no test, and pins that could be neutralised.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_log_stream_websocket_REQUIRES_the_key():
    """⚠ This gate had ZERO tests. Deleting the whole `compare_digest` check
    on `/ws` passed 235 interface tests and 697 audit tests (R4 lens A) — and
    it is the one gate whose removal is an exfiltration channel: the live log
    carries tool output and file contents, and WebSockets bypass CORS, so any
    site the operator visits could connect and read it.

    ⚠ DRIVEN DIRECTLY, not through TestClient. Two earlier versions of this
    test HUNG THE SUITE for 37 minutes: an accepted socket parks the endpoint
    in `await websocket.receive_text()` and the context manager's exit waits
    on it, and a REJECTED one (closed before `accept`) hangs inside
    `TestClient.websocket_connect` itself. A fake peer makes both paths
    deterministic and instant."""
    from starlette.websockets import WebSocketDisconnect as _Disc

    class _WS:
        def __init__(self):
            self.events = []

        async def accept(self):
            self.events.append("accept")

        async def receive_text(self):
            self.events.append("receive")
            raise _Disc(code=1000)

        async def close(self, code=None):
            self.events.append(f"close:{code}")

    # ---- rejected: no key, empty key, wrong key ----
    for key in (None, "", "wrong-key", server.GHOST_API_KEY + "x",
                server.GHOST_API_KEY[:-1]):
        ws = _WS()
        await server.websocket_endpoint(ws, key=key)
        assert ws.events == ["close:1008"], (
            f"key={key!r} was not rejected: {ws.events}")
        assert ws not in server.connected_websockets

    # ---- accepted: the real key. Without this the assertions above would
    # pass against a permanently broken endpoint. ----
    ws = _WS()
    await server.websocket_endpoint(ws, key=server.GHOST_API_KEY)
    assert ws.events[0] == "accept", (
        f"the correct key was REJECTED, so the rejections above prove "
        f"nothing: {ws.events}")
    assert "receive" in ws.events
    assert ws not in server.connected_websockets, (
        "the socket was not evicted from the registry on disconnect")


def test_the_websocket_key_check_is_CONSTANT_TIME():
    """`verify_interface_key` and this gate both compare a secret; the
    codebase mandates `secrets.compare_digest` and nothing enforced it here
    (mutation to `!=` survived)."""
    import inspect
    src = inspect.getsource(server.websocket_endpoint)
    assert "compare_digest" in src, (
        "the /ws key comparison is not constant-time")




@pytest.mark.asyncio
async def test_chat_proxy_enforces_the_cap_before_inserting(monkeypatch):
    monkeypatch.setattr(server, "ACTIVE_TASK_HARD_CAP", 5)
    server.active_chat_tasks.clear()
    for i in range(20):
        server.active_chat_tasks[f"old{i}"] = {"done": True, "buffer": []}

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()

    async def _aiter(*a, **k):
        yield b'data: {}\n\n'

    fake_resp.aiter_bytes = MagicMock(side_effect=_aiter)
    ctx = AsyncMock()
    ctx.__aenter__.return_value = fake_resp
    ctx.__aexit__.return_value = None
    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=ctx)
    req = MagicMock()
    req.json = AsyncMock(return_value={"stream": True, "messages": []})
    try:
        with patch.object(server, "_get_http_client", return_value=fake_client):
            resp = await server.chat_proxy(req)
            async for _ in resp.body_iterator:
                pass
        assert len(server.active_chat_tasks) <= server.ACTIVE_TASK_HARD_CAP + 1, (
            f"the insert path did not enforce the cap: "
            f"{len(server.active_chat_tasks)} entries")
    finally:
        server.active_chat_tasks.clear()


def test_no_module_level_env_number_is_parsed_UNGUARDED():
    """Hostile numeric env values must not crash a module at IMPORT — under a
    KeepAlive LaunchDaemon that is a permanent silent restart loop.

    ⚠ TWO WRONG VERSIONS PRECEDED THIS ONE. The first forbade one exact
    spelling (`float(os.environ.get("X"…`), so
    `float(os.environ["X"]) if … else 1800.0` — which restores the crash-loop
    verbatim — passed it (R4 lens A). The second spawned six subprocesses to
    import the module for real, and **that HUNG the whole suite**: forking
    from a pytest process carrying ~110 threads deadlocks before exec, and
    `subprocess.run`'s timeout covers `communicate()`, not the fork. It passed
    alone in 3s and hung forever as test 36 of 40.

    So: parse the module and look for the SHAPE — `float()`/`int()` applied
    directly to an `os.environ` access at module scope — which is what makes
    an import crashable, whichever way it is spelled. No fork, and no literal
    to dodge."""
    import ast

    ROOT = Path(__file__).resolve().parents[1]
    offenders = []
    for rel in ("interface/server.py", "interface/voice.py",
                "interface/webpush_notify.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))

        def _is_environ_access(node):
            """os.environ["X"] / os.environ.get("X") / environ.get("X") /
            os.getenv("X") — and anything wrapping one of those, so
            `.strip()`, `or 10`, and a walrus are all still seen.

            ⚠ `os.getenv` was missing from the first version, and it is this
            codebase's DOMINANT idiom (91 uses in src/) — the check guarded a
            spelling, not the property (R5 lens A)."""
            if isinstance(node, ast.NamedExpr):          # walrus
                return _is_environ_access(node.value)
            if isinstance(node, ast.BoolOp):             # `os.getenv(…) or 10`
                return any(_is_environ_access(v) for v in node.values)
            if isinstance(node, ast.Subscript):
                return _is_environ_access(node.value)
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute):
                    if fn.attr in ("get", "getenv"):
                        return _is_environ_access(fn.value)
                    # `os.environ.get("X").strip()` and friends
                    return _is_environ_access(fn.value)
                if isinstance(fn, ast.Name) and fn.id == "getenv":
                    return True
                return False
            if isinstance(node, ast.Attribute):
                return node.attr in ("environ", "os")
            if isinstance(node, ast.Name):
                return node.id in ("environ", "os")
            return False

        # MODULE SCOPE ONLY. `ast.walk` descends into function bodies, and a
        # parse inside a function is lazy and usually try/except-guarded —
        # flagging those was a false positive that named four innocent lines
        # (`_max_json_bytes`, `_stream_cap_bytes`, …). What crash-loops a
        # daemon is a parse that runs when the module is IMPORTED.
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                continue
            if isinstance(stmt, ast.Try):
                continue        # a module-level parse inside try/except IS
                                # guarded — flagging it was a false positive
            for node in ast.walk(stmt):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if not (isinstance(fn, ast.Name)
                        and fn.id in ("float", "int", "complex", "Decimal")):
                    continue
                if node.args and _is_environ_access(node.args[0]):
                    offenders.append(f"{rel}:{node.lineno} "
                                     f"{fn.id}(<os.environ …>)")
    assert not offenders, (
        "these run at IMPORT and raise ValueError on a typo'd or empty value, "
        "which crash-loops the daemon — route them through a guarded helper "
        "(_env_num):\n  " + "\n  ".join(offenders))


def test_env_numbers_are_CLAMPED_to_positive(monkeypatch):
    """`_env_num` generalised `_push_timeout_s` and dropped its `v > 0`
    clamp, then guarded nine sites. `GHOST_WS_SEND_TIMEOUT=0` makes every
    broadcast time out on a HEALTHY client, so the log stream discards and
    closes every peer on the first line (R4 lens A)."""
    for raw in ("0", "0.0", "-1", "-0.5"):
        monkeypatch.setenv("GHOST_TEST_CLAMP", raw)
        got = server._env_num("GHOST_TEST_CLAMP", 2.0)
        assert got == 2.0, (
            f"{raw!r} was accepted as {got} — a non-positive timeout or "
            f"ceiling is worse than a typo")
    monkeypatch.setenv("GHOST_TEST_CLAMP", "0.25")
    assert server._env_num("GHOST_TEST_CLAMP", 2.0) == 0.25, (
        "the clamp is rejecting legitimate small values")


@pytest.mark.asyncio
async def test_shutdown_ignores_tasks_from_ANOTHER_live_loop():
    """The previous filter claimed "only tasks from THIS loop" and never
    checked the loop: it relied on `cancel()` raising for a CLOSED foreign
    loop, leaving the OPEN foreign loop case — where cancel() succeeds and
    then `asyncio.gather` raises "attached to a different loop" at attach
    time, which `return_exceptions=True` cannot catch (R4 lens A).

    Same teardown-fix-breaks-teardown class as the bug it replaced."""
    import threading

    holder = {}
    ready = threading.Event()
    stop = threading.Event()

    def _other_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _sleep_forever():
            await asyncio.sleep(3600)

        async def _main():
            holder["task"] = asyncio.ensure_future(_sleep_forever())
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(0.02)

        loop.run_until_complete(_main())
        holder["task"].cancel()
        loop.run_until_complete(asyncio.gather(holder["task"],
                                               return_exceptions=True))
        loop.close()

    t = threading.Thread(target=_other_loop, daemon=True)
    t.start()
    assert ready.wait(5), "helper loop never started"
    server.active_chat_tasks["t-foreign"] = {
        "background_task": holder["task"], "done": False, "buffer": []}
    try:
        # Must not raise out of shutdown.
        async with server._lifespan(server.app):
            pass
    finally:
        server.active_chat_tasks.pop("t-foreign", None)
        stop.set()
        t.join(timeout=5)


@pytest.mark.asyncio
async def test_the_error_body_SURVIVES_a_timeout():
    """⚠ THE ASSIGNMENT MUST BE IN A `finally`.

    It used to sit inside the `try` after the await, so the very timeout this
    function exists to impose discarded the body it had already collected —
    the complete error message could arrive in the first chunk and still be
    thrown away, leaving the client with "upstream returned HTTP N", the
    causeless message this helper family exists to remove (R5 lens A).

    The old test asserted only ELAPSED TIME, which is true with and without
    the bug."""
    class _StallsAfterTheBody:
        status_code = 502

        async def aiter_bytes(self, *a, **k):
            yield b'{"error": "agent refused: unsupported file type .exe"}'
            await asyncio.sleep(30)          # the gap that never closes
            yield b"never arrives"

    r = _StallsAfterTheBody()
    t0 = time.perf_counter()
    await server._buffer_error_body(r, total_timeout_s=0.3)
    dt = time.perf_counter() - t0
    assert dt < 5.0, f"the read is unbounded ({dt:.1f}s)"
    assert r._content and b"unsupported file type" in r._content, (
        "the body collected before the timeout was discarded")


@pytest.mark.asyncio
async def test_the_error_body_survives_a_peer_RESET():
    """Same defect, other trigger: an exception mid-body (RemoteProtocolError,
    ReadTimeout) also skipped the assignment."""
    class _Resets:
        status_code = 500

        async def aiter_bytes(self, *a, **k):
            yield b'{"error": "no such file: report.pdf"}'
            raise ConnectionResetError("peer went away")

    r = _Resets()
    await server._buffer_error_body(r)
    assert r._content and b"no such file" in r._content, (
        "a mid-body reset discarded the part that had already arrived")


@pytest.mark.asyncio
async def test_the_error_body_read_STOPS_at_the_limit():
    """⚠ Not the same as "the stored value is short". The old assertion
    (`len(_content) <= 1024`) is guaranteed by the `[:limit]` slice whether or
    not the READ stops — so removing the size break survived it, and the
    helper could pull a multi-GB error body into RAM before slicing
    (R5 lens A). Count what the generator actually yielded."""
    pulled = {"n": 0}

    class _Huge:
        status_code = 502
        _content = None

        async def aiter_bytes(self, *a, **k):
            for _ in range(1000):
                pulled["n"] += 1
                yield b"x" * 4096

    r = _Huge()
    await server._buffer_error_body(r, limit=1024)
    assert r._content is not None and len(r._content) <= 1024, r._content
    assert pulled["n"] <= 2, (
        f"read {pulled['n']} chunks for a 1024-byte limit — the read is "
        f"unbounded and only the SLICE is short")


@pytest.mark.parametrize("mod_name", ["interface.server", "interface.voice"])
def test_every_env_helper_survives_hostile_values(mod_name):
    """The AST check above sees SHAPES, so it cannot see a helper that stops
    catching — deleting `except (TypeError, ValueError)` from voice's
    `_env_num` restored the crash-loop and survived all 316 tests (R5 lens A).
    Exercise both twins directly, including the positivity clamp that landed
    on one copy and not the other."""
    import importlib
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    mod = importlib.import_module(mod_name)
    fn = mod._env_num
    for raw, default, cast, want in (
            ("abc", 5.0, float, 5.0), ("", 5.0, float, 5.0),
            ("  ", 5.0, float, 5.0), ("0", 5.0, float, 5.0),
            ("-3", 5.0, float, 5.0), ("2.5", 5.0, float, 2.5),
            ("abc", 7, int, 7), ("0", 7, int, 7), ("9", 7, int, 9)):
        os.environ["GHOST_TEST_ENVNUM"] = raw
        try:
            got = fn("GHOST_TEST_ENVNUM", default, cast)
        finally:
            os.environ.pop("GHOST_TEST_ENVNUM", None)
        assert got == want and isinstance(got, cast), (
            f"{mod_name}._env_num({raw!r}) -> {got!r}, want {want!r}")
    assert fn("GHOST_TEST_ABSENT_VAR", 4, int) == 4


@pytest.mark.parametrize("helper,var,default", [
    ("_max_json_bytes", "GHOST_INTERFACE_MAX_JSON_BYTES", 10 * 1024 * 1024),
    ("_stream_cap_bytes", "GHOST_INTERFACE_STREAM_CAP", 50_000_000),
    ("_total_stream_cap_bytes", "GHOST_INTERFACE_TOTAL_STREAM_CAP", 200_000_000),
    ("_push_ack_grace_s", "GHOST_PUSH_ACK_GRACE", 12),
])
def test_every_env_ceiling_rejects_zero_and_garbage(helper, var, default):
    """These four are the helpers `_env_num`'s own docstring cites as the
    guarded idiom it generalised — and they guarded TYPOS only, not zero or
    negative. `GHOST_INTERFACE_STREAM_CAP=0` truncates every chat reply to
    zero bytes with a BufferCapExceeded frame (R5 lens A).

    Behavioural, because the AST check deliberately does not look inside
    functions (a lazy parse there is not a crash-loop) — so reverting one of
    these to a raw `int(os.environ.get(...))` was invisible to it."""
    fn = getattr(server, helper)
    for raw in ("0", "-1", "abc", ""):
        os.environ[var] = raw
        try:
            got = fn()
        finally:
            os.environ.pop(var, None)
        assert got == default, (
            f"{helper}() with {var}={raw!r} -> {got!r}; a non-positive or "
            f"unparseable ceiling degrades the whole subsystem")
    os.environ[var] = "1234"
    try:
        assert fn() == 1234
    finally:
        os.environ.pop(var, None)


@pytest.mark.asyncio
async def test_a_cancel_during_error_buffering_still_closes_the_stream():
    """R4 widened the window between "upstream failed" and `aclose()` from
    zero to up to 5s (the error-body buffer). A client disconnect inside that
    window used to skip `aclose()` entirely and strand a pooled connection —
    100 connections with a 1800s pool timeout, so the 101st request parks for
    half an hour (R5 lens A). The close is shielded; prove it survives a
    cancellation."""
    closed = []

    class _SlowBody:
        status_code = 502

        async def aiter_bytes(self, *a, **k):
            await asyncio.sleep(30)
            yield b"never"

        async def aclose(self):
            closed.append(True)

    resp = _SlowBody()

    async def _handler():
        try:
            await _buffer_and_close(resp)
        except Exception:
            raise

    async def _buffer_and_close(r):
        # The exact shape both streamed proxies use.
        try:
            await server._buffer_error_body(r)
        finally:
            await asyncio.shield(r.aclose())

    task = asyncio.ensure_future(_handler())
    await asyncio.sleep(0.05)          # inside the buffer window
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0)             # let the shielded close finish
    assert closed == [True], (
        "the upstream stream was left OPEN after a cancel — that connection "
        "never returns to the pool")

    # And the shape must actually be in both proxies, not just in this test.
    import inspect
    for name in ("workspace_save_proxy", "download_proxy"):
        src = inspect.getsource(getattr(server, name))
        assert "asyncio.shield(resp.aclose())" in src, (
            f"{name} closes the stream unshielded during error buffering")
