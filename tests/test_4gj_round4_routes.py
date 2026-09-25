"""§4GJ round 4 (2026-09-13) — the defects a read-only reviewer confirmed by
driving the REAL ASGI app over raw sockets, each pinned in the world where it
fails:

  * the chat SSE path released its foreground mark inside the body generator,
    so a mid-stream disconnect left it marked until the cyclic GC ran — round
    3's own CRITICAL, unfixed on the path that carries the user traffic;
  * `_store_call` abandoned a worker of the PROCESS-WIDE default executor per
    timeout, so operator retries of one wedged route stopped every tool call;
  * `_main_node_request(hold_lock=True)` counted a request that the held main
    lock already counts, which is the "Stream Stall (Self-Queued)" condition;
  * the proxy forwarded `proxy-authorization`, `cookie` and the hop-by-hop
    set verbatim, and re-framed a chunked POST with two framings;
  * two `_store_call` sites had no `StoreCallTimeout` handler;
  * the workspace ZIP lost modes and mtimes, dropped empty directories,
    dropped unreadable files SILENTLY, capped itself at 65,535 members and
    materialised every member in RAM;
  * `/api/generate` answered 500 for an upstream 5xx;
  * `catch_all` released its booking on `Exception` but not on the
    BaseException this app deliberately raises through the receive channel.

Finding 10 of that round ("the 504 leaves the upstream generating") is
answered by measurement rather than a fix — see
`test_the_generate_504_has_already_closed_the_upstream_connection`.
"""
import asyncio
import concurrent.futures
import gc
import json
import os
import stat as _stat
import threading
import time
import tracemalloc
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ghost_agent.api import routes as R
from ghost_agent.core.llm import LLMClient


# ── doubles ──────────────────────────────────────────────────────────────────

class _Req:
    """A request with REAL headers and a real body — the routes read both."""

    def __init__(self, body=None, headers=None, raw=b"", method="POST"):
        self._body = body
        self.headers = headers if headers is not None else {}
        self._raw = raw
        self.method = method
        self.query_params = {}

    async def json(self):
        if self._body is None:
            raise json.JSONDecodeError("no body", "", 0)
        return self._body

    async def body(self):
        return self._raw

    def stream(self):
        raise AssertionError("not used by these tests")


def _chat_agent():
    agent = MagicMock()
    agent.context.args.model = "ghost-model"
    agent.context.args.api_key = ""
    agent.context.llm_client.foreground_requests = 0
    return agent


class _InflightLlm:
    """The in-flight bookkeeping of the real client, with the REAL methods —
    `_own_inflight` is the consumer whose answer the defect corrupted, so the
    pin must ask it, not re-derive it."""

    _inflight_map = LLMClient._inflight_map
    _inflight_inc = LLMClient._inflight_inc
    _inflight_dec = LLMClient._inflight_dec
    _own_inflight = LLMClient._own_inflight

    def __init__(self):
        self.upstream_url = "http://main:8000"
        self._main_node_lock = asyncio.Lock()
        self._foreground_lock = asyncio.Lock()
        self.foreground_tasks = 0
        self.http_client = MagicMock()


# ── 1. the chat SSE path releases its mark around the SEND ───────────────────

async def _drive_until_first_chunk(resp):
    """Run the response as a real ASGI app whose client goes away mid-stream:
    `send` never returns after the first body chunk, exactly as a wedged /
    vanished socket behaves, and the surrounding task is then cancelled —
    which is what Starlette does to `stream_response` on a disconnect."""
    first = asyncio.Event()

    async def send(message):
        if message["type"] == "http.response.body":
            first.set()
            await asyncio.sleep(3600)

    async def receive():
        await asyncio.sleep(3600)

    task = asyncio.create_task(resp({"type": "http", "method": "POST",
                                     "path": "/api/chat", "headers": []},
                                    receive, send))
    await asyncio.wait_for(first.wait(), 5)
    task.cancel()
    with pytest.raises(BaseException):
        await task
    await asyncio.sleep(0.05)


async def test_the_chat_stream_releases_its_foreground_mark_on_a_disconnect():
    """The CRITICAL. `foreground_requests > 0` is read by `core/llm` as "a
    user is active" and parks every background LLM call; a mark released only
    by a generator `finally` survives a mid-stream disconnect because nothing
    ever resumes or closes that generator — the cyclic GC does, eventually.
    GC is disabled here so the pin measures the release, not the collector:
    in the pre-fix world the counter stays at 1 forever."""
    agent = _chat_agent()

    async def _gen():
        for i in range(50):
            yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
            await asyncio.sleep(0.01)

    agent.handle_chat = AsyncMock(return_value=(_gen(), 123, "req1"))
    req = _Req({"messages": [{"role": "user", "content": "hi"}], "stream": True})

    gc.disable()
    try:
        with patch.object(R, "get_agent", return_value=agent):
            resp = await R.chat_proxy(req, MagicMock())
        assert agent.context.llm_client.foreground_requests == 1, "never marked"
        await _drive_until_first_chunk(resp)
        assert agent.context.llm_client.foreground_requests == 0, (
            "the foreground mark leaked past the client disconnect")
    finally:
        gc.enable()


async def test_the_chat_stream_mark_is_released_exactly_once_on_a_clean_read():
    """Control: the release is idempotent, so the generator's own `finally`
    and the response's cannot double-decrement a shared counter."""
    agent = _chat_agent()
    agent.handle_chat = AsyncMock(return_value=("hello", 123, "req1"))

    async def _stream(*a, **k):
        yield b"data: {}\n\n"

    agent.context.llm_client.stream_openai = _stream
    req = _Req({"messages": [{"role": "user", "content": "hi"}], "stream": True})
    with patch.object(R, "get_agent", return_value=agent):
        resp = await R.chat_proxy(req, MagicMock())
    assert agent.context.llm_client.foreground_requests == 1
    body = b"".join([c async for c in resp.body_iterator])
    assert b"data:" in body
    assert agent.context.llm_client.foreground_requests == 0
    # a second release (the response's `finally`, after the generator's) does
    # not drive the counter negative or below another request's mark
    await resp._release()
    assert agent.context.llm_client.foreground_requests == 0


# ── 2. a wedged store cannot starve the rest of the process ──────────────────

async def test_a_wedged_store_call_does_not_consume_the_default_executor():
    """`asyncio.to_thread` runs on the loop's DEFAULT executor — the one
    shared by 359 `to_thread`/`run_in_executor(None)` sites across 38 modules.
    A store call that times out is deliberately not cancelled, so every
    timeout abandons one of those workers until the store unwedges. Measured
    pre-fix: 20 wedged store calls blocked an unrelated `to_thread` for 3 s.

    The default executor is shrunk to 2 workers here so the pin states the
    mechanism rather than this box's core count; pre-fix it hangs the
    unrelated call until the wait_for below gives up."""
    loop = asyncio.get_running_loop()
    tiny = concurrent.futures.ThreadPoolExecutor(max_workers=2,
                                                 thread_name_prefix="default-pool")
    loop.set_default_executor(tiny)
    gate = threading.Event()

    def wedged():
        gate.wait(20)

    calls = [asyncio.create_task(R._store_call(wedged, timeout=10)) for _ in range(6)]
    await asyncio.sleep(0.3)
    try:
        got = await asyncio.wait_for(
            asyncio.to_thread(lambda: "the rest of the process"), 2.0)
    finally:
        gate.set()
        await asyncio.gather(*calls, return_exceptions=True)
        tiny.shutdown(wait=False)
    assert got == "the rest of the process"


async def test_a_store_call_runs_on_the_stores_own_pool():
    """The identity of the fix: the work lands on `_STORE_EXECUTOR`, not on
    whatever `to_thread` happens to share."""
    names = []

    def who():
        names.append(threading.current_thread().name)
        return "done"

    assert await R._store_call(who) == "done"
    assert names and names[0].startswith("ghost-store"), names
    assert R._STORE_EXECUTOR._max_workers == R._STORE_EXECUTOR_WORKERS


# ── 3. the held main lock is already worth one in-flight request ─────────────

async def test_a_locked_main_request_is_not_counted_twice():
    """`LLMClient._own_inflight` adds one for a HELD `_main_node_lock` — its
    documented contract, and why llm.py's own locked paths do not increment.
    Booking `/api/generate` with BOTH made one real stream plus one generate
    read as 3 against a truth of 2; on a 2-slot node that is exactly the
    `_conc > _cap` condition that aborts the USER's live stream as
    "Stream Stall (Self-Queued)"."""
    llm = _InflightLlm()
    base = llm.upstream_url
    llm._inflight_inc(base)                    # the user's live stream
    async with R._main_node_request(llm, hold_lock=True):
        conc, cap = llm._own_inflight(base), 2
        assert conc == 2, "the locked path counted itself twice"
        assert not conc > cap, "this is the self-queued abort condition"
    llm._inflight_dec(base)
    assert llm._own_inflight(base) == 0


async def test_the_streaming_path_still_counts_itself():
    """Control: `hold_lock=False` holds no lock, so it is the ONLY thing that
    can make the stream visible — removing the count there would undercount."""
    llm = _InflightLlm()
    base = llm.upstream_url
    async with R._main_node_request(llm, hold_lock=False):
        assert llm._own_inflight(base) == 1
    assert llm._own_inflight(base) == 0


# ── 4. the proxy strips the whole credential + hop-by-hop class ──────────────

def test_the_proxy_forwards_no_credential_and_no_hop_by_hop_header():
    """Confirmed forwarded verbatim pre-fix: `proxy-authorization`, `cookie`,
    `connection`, `te`, `upgrade`. `authorization` was stripped as a
    credential while its two siblings were not, and with a remote
    `--upstream-url` they leave the machine."""
    sent = {
        "X-Ghost-Key": "sekrit", "Authorization": "Bearer t",
        "Proxy-Authorization": "Basic dXNlcjpwdw==", "Cookie": "session=abc",
        "Host": "ghost.local", "Content-Length": "9",
        "Transfer-Encoding": "chunked", "Connection": "keep-alive",
        "Keep-Alive": "timeout=5", "Proxy-Connection": "keep-alive",
        "TE": "trailers", "Trailer": "Expires", "Trailers": "x",
        "Proxy-Authenticate": "Basic", "Upgrade": "websocket",
        "Content-Type": "application/json", "User-Agent": "curl/8",
        "X-Request-ID": "r-1",
    }
    fwd = R._forwardable_headers(sent)
    assert {k.lower() for k in fwd} == {"content-type", "user-agent", "x-request-id"}
    assert "dXNlcjpwdw==" not in json.dumps(fwd) and "session=abc" not in json.dumps(fwd)


async def test_a_chunked_json_post_is_not_forwarded_with_two_framings():
    """The JSON peek re-feeds the body to httpx as BYTES, so httpx sets its
    own `Content-Length`. Forwarding the client's `Transfer-Encoding: chunked`
    alongside it 502'd here (confirmed) and is a CL.TE desync primitive
    against a laxer upstream."""
    llm = _InflightLlm()
    seen = {}

    def build(method, url, headers=None, content=None):
        seen["headers"] = headers
        return MagicMock()

    llm.http_client.build_request = MagicMock(side_effect=build)
    r = MagicMock()
    r.status_code = 200
    r.headers = {"content-type": "application/json"}
    r.aclose = AsyncMock()

    async def _aiter():
        yield b"{}"
    r.aiter_bytes = MagicMock(side_effect=_aiter)
    llm.http_client.send = AsyncMock(return_value=r)
    agent = MagicMock()
    agent.context.llm_client = llm
    req = _Req(headers={"content-type": "application/json",
                        "transfer-encoding": "chunked",
                        "cookie": "session=abc"},
               raw=b'{"stream": false}')
    with patch.object(R, "get_agent", return_value=agent):
        resp = await R.catch_all(req, "v1/chat/completions")
    lower = {k.lower() for k in seen["headers"]}
    assert "transfer-encoding" not in lower and "cookie" not in lower, lower
    assert resp.status_code == 200
    await resp._release()


# ── 5. every store call answers 504, not a bare 500 ──────────────────────────

def _wedged_store():
    started = threading.Event()

    def stuck(*a, **k):
        started.set()
        time.sleep(3.0)
    return started, stuck


@pytest.mark.parametrize("stream", [True, False])
async def test_a_wedged_session_store_answers_504_on_both_chat_paths(monkeypatch, stream):
    """`_sess_store.get` sits ABOVE the `if stream:` split, so the missing
    handler took BOTH chat paths down with a bare text/plain 500 — no
    `error.message`, nothing the web UI or the Slack bot can render, and a
    status that blames the agent for a store that is merely busy."""
    monkeypatch.setattr(R, "_STORE_CALL_TIMEOUT_S", 0.2)
    started, stuck = _wedged_store()
    from ghost_agent.core import sessions as S
    monkeypatch.setattr(S, "get_session_store",
                        lambda ctx: SimpleNamespace(get=stuck, append_turn=stuck))
    agent = _chat_agent()
    req = _Req({"messages": [{"role": "user", "content": "hi"}],
                "stream": stream, "session_id": "s1"})
    with patch.object(R, "get_agent", return_value=agent):
        resp = await R.chat_proxy(req, MagicMock())
    assert started.is_set()
    assert resp.status_code == 504, resp.status_code
    assert json.loads(bytes(resp.body))["error"]["type"] == "StoreCallTimeout"
    # and the request is not left marked as an active user turn
    assert agent.context.llm_client.foreground_requests == 0


async def test_a_wedged_activity_log_answers_504(monkeypatch, tmp_path):
    """`log.read_since` was wrapped without its `except`, so a wedged ledger
    reached the Slack bot's poll loop as a bare 500 — indistinguishable from
    "the agent crashed" — while every other store-backed route answers a
    JSON 504."""
    monkeypatch.setattr(R, "_STORE_CALL_TIMEOUT_S", 0.2)
    started, stuck = _wedged_store()
    from ghost_agent.core import autonomous_activity as AA
    fake_log = SimpleNamespace(current_offset=lambda: 10, read_since=stuck)
    monkeypatch.setattr(AA, "get_activity_log", lambda ctx: fake_log)
    monkeypatch.setattr(AA, "load_consumer_offset", lambda p, c: 0)
    agent = MagicMock()
    agent.context.memory_dir = str(tmp_path / "memory")
    with patch.object(R, "get_agent", return_value=agent):
        resp = await R.notifications_pending(_Req(), consumer="slack", limit=5)
    assert started.is_set()
    assert resp.status_code == 504
    assert "did not return within" in json.loads(bytes(resp.body))["error"]


# ── 6-9. the workspace archive ───────────────────────────────────────────────

def _ws_agent(sandbox: Path):
    agent = MagicMock()
    agent.context.sandbox_dir = sandbox
    agent.context.scratchpad = None
    return agent


async def _save(sandbox: Path):
    agent = _ws_agent(sandbox)
    with patch.object(R, "get_agent", return_value=agent):
        return await R.save_workspace(_Req(raw=b""))


def _sandbox(tmp_path) -> Path:
    sandbox = tmp_path / "sandbox"
    (sandbox / "nested" / "deep").mkdir(parents=True)
    (sandbox / "empty_dir").mkdir()
    run = sandbox / "run.sh"
    run.write_text("#!/bin/sh\necho hi\n")
    run.chmod(0o755)
    os.utime(run, (1_600_000_000, 1_600_000_000))
    (sandbox / "notes.txt").write_text("plain")
    return sandbox


async def test_the_archive_keeps_modes_mtimes_and_empty_directories(tmp_path):
    """`writestr(str, ...)` stamps every member with the archive time and a
    hardcoded `0o600`; the `zip_file.write(path)` it replaced read both off
    the file. Confirmed pre-fix: `run.sh` came back `0o600` (nothing in a
    restored sandbox is executable), every mtime was the moment of the save,
    and `empty_dir`, `nested` and `nested/deep` were absent entirely.

    ⚠ THIS MEASURES THE ARCHIVE, NOT THE WORLD ITS FIRST PARAGRAPH NAMES
    (§4GK round 5). It passed while the restore still applied neither mode
    nor mtime, so "nothing in the restored sandbox is executable any more"
    stayed TRUE for everyone restoring through the agent — only an external
    `unzip` ever saw this fix. The property is pinned end to end by
    `test_4gk_round5_routes.py::test_a_restored_workspace_is_still_executable_and_still_dated`;
    this one keeps the archive half honest."""
    sandbox = _sandbox(tmp_path)
    resp = await _save(sandbox)
    with zipfile.ZipFile(resp.path) as zf:
        names = set(zf.namelist())
        run = zf.getinfo("sandbox/run.sh")
        notes = zf.getinfo("sandbox/notes.txt")
    assert "sandbox/empty_dir/" in names, names
    assert "sandbox/nested/" in names and "sandbox/nested/deep/" in names, names
    assert _stat.S_IMODE(run.external_attr >> 16) == 0o755
    assert _stat.S_IMODE(notes.external_attr >> 16) == 0o644
    assert run.date_time[:3] == time.localtime(1_600_000_000)[:3], run.date_time
    os.unlink(resp.path)


async def test_an_unreadable_member_is_reported_not_silently_dropped(tmp_path):
    """A 200 carrying an archive the operator believes is their whole
    workspace: a restore from it wipes the sandbox and writes back only the
    readable subset, so the silent `continue` turns a permissions glitch into
    data loss."""
    sandbox = _sandbox(tmp_path)
    secret = sandbox / "locked.bin"
    secret.write_text("cannot read me")
    secret.chmod(0o000)
    try:
        resp = await _save(sandbox)
    finally:
        secret.chmod(0o600)
    assert resp.headers.get("x-ghost-archive-omitted") == "1", dict(resp.headers)
    with zipfile.ZipFile(resp.path) as zf:
        assert "sandbox/locked.bin" not in zf.namelist()
        omitted = json.loads(zf.read("omitted.json"))
    assert [o["path"] for o in omitted] == ["locked.bin"], omitted
    assert "PermissionError" in omitted[0]["reason"]
    assert omitted[0]["partial"] is False        # nothing of it reached the zip
    os.unlink(resp.path)


async def test_a_read_that_dies_mid_copy_is_reported_as_a_partial_member(tmp_path, monkeypatch):
    """The streaming path can fail AFTER bytes have landed in the archive
    (EIO on a flaky mount). Calling that "omitted" would be a small lie about
    a member that is physically in the file and short: whoever restores it
    needs to know the difference, so the record says which it was."""
    # sizes chosen so `os.read(fd, 33333)` is a signature only the streaming
    # copy produces — the buffered whole-file read never asks for that number
    monkeypatch.setattr(R, "_ZIP_INLINE_MEMBER_BYTES", 1000)
    monkeypatch.setattr(R, "_ZIP_COPY_CHUNK_BYTES", 33333)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "flaky.bin").write_bytes(b"z" * 100_000)
    real_read, seen = os.read, {"n": 0}

    def flaky(fd, n):
        if n == R._ZIP_COPY_CHUNK_BYTES:          # only `_stream_member` asks this
            seen["n"] += 1
            if seen["n"] > 1:
                raise OSError(5, "Input/output error")
        return real_read(fd, n)

    with patch.object(os, "read", flaky):
        resp = await _save(sandbox)
    assert resp.headers.get("x-ghost-archive-omitted") == "1"
    with zipfile.ZipFile(resp.path) as zf:
        record = json.loads(zf.read("omitted.json"))[0]
        member = zf.getinfo("sandbox/flaky.bin")
    assert record["partial"] is True and "OSError" in record["reason"]
    assert 0 < member.file_size < 100_000
    os.unlink(resp.path)


async def test_a_clean_archive_carries_no_omission_marker(tmp_path):
    """Control: the marker means something only if it is absent when nothing
    was omitted."""
    resp = await _save(_sandbox(tmp_path))
    assert "x-ghost-archive-omitted" not in {k.lower() for k in resp.headers}
    with zipfile.ZipFile(resp.path) as zf:
        assert "omitted.json" not in zf.namelist()
    os.unlink(resp.path)


async def test_the_archive_is_not_capped_at_65535_members(tmp_path, monkeypatch):
    """`ZipFile(path, "w", ZIP_DEFLATED, False)` — the 4th positional is
    `allowZip64`, whose own default is True. A sandbox with a `node_modules`
    clears 65,535 members far below the 500 MB byte cap; `LargeZipFile` is
    raised at CLOSE, after the whole tree has been compressed, and this route
    answered an opaque 500 where the byte path answers an honest 413.

    The member ceiling is the real `ZIP_FILECOUNT_LIMIT`, lowered so the pin
    exercises the same `_write_end_record` branch without writing 65k files.
    """
    monkeypatch.setattr(zipfile, "ZIP_FILECOUNT_LIMIT", 2)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    for i in range(6):
        (sandbox / f"f{i}.txt").write_text(str(i))
    resp = await _save(sandbox)
    assert isinstance(resp, R.FileResponse), getattr(resp, "body", resp)
    with zipfile.ZipFile(resp.path) as zf:
        assert len(zf.namelist()) > 2
    os.unlink(resp.path)


async def test_one_large_member_is_streamed_not_materialised(tmp_path):
    """Per-member RAM was the largest FILE, not the buffer: the read helper
    was called with `max_bytes=<the 500 MB archive cap>`, so one member was
    materialised whole and then deflated whole — measured 215 MB → 369 MB RSS
    for a single 150 MB file, on a box that already runs at 94% memory.

    Two statements, because either alone is weak: no single read returns more
    than the inline ceiling, and the traced peak stays far below the member.
    """
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    big = sandbox / "big.bin"
    payload = (b"ghost-payload-0123456789" * 512)          # ~12 KiB, compressible
    with open(big, "wb") as fh:
        for _ in range(1024):                              # 12 MiB
            fh.write(payload)
    size = big.stat().st_size
    assert size > R._ZIP_INLINE_MEMBER_BYTES

    from ghost_agent.tools import file_system as FS
    real_read = FS.read_bytes_nofollow_fd
    returned = []

    def recording(name, *, dir_fd=None, max_bytes=0):
        data = real_read(name, dir_fd=dir_fd, max_bytes=max_bytes)
        returned.append(len(data))
        return data

    tracemalloc.start()
    try:
        with patch.object(FS, "read_bytes_nofollow_fd", recording):
            tracemalloc.reset_peak()
            resp = await _save(sandbox)
            peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert max(returned or [0]) <= R._ZIP_INLINE_MEMBER_BYTES, returned
    assert peak < size // 2, f"peak {peak} for a {size}-byte member"
    with zipfile.ZipFile(resp.path) as zf:
        assert zf.getinfo("sandbox/big.bin").file_size == size
        assert len(zf.read("sandbox/big.bin")) == size
    os.unlink(resp.path)


async def test_the_byte_ceiling_still_stops_a_runaway_archive(tmp_path, monkeypatch):
    """Control for the streaming path: the 500 MB cap is enforced DURING the
    copy, not only on the whole-file read it replaced, and the route still
    answers the honest 413."""
    monkeypatch.setattr(R, "_MAX_WORKSPACE_SAVE_BYTES", 2 * 1024 * 1024)
    monkeypatch.setattr(R, "_ZIP_INLINE_MEMBER_BYTES", 64 * 1024)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "big.bin").write_bytes(b"x" * (4 * 1024 * 1024))
    resp = await _save(sandbox)
    assert resp.status_code == 413, getattr(resp, "body", resp)
    assert json.loads(bytes(resp.body))["error"]["type"] == "WorkspaceTooLarge"


# ── 10. the 504 has already closed the connection (measurement) ──────────────

async def test_the_generate_504_has_already_closed_the_upstream_connection():
    """NOT a pin — the counter-evidence for round 4's finding 10, which said
    the 504 "leaves the upstream generating … 0 cancelled".

    Driven against a real `httpx.AsyncClient` and a real listening socket:
    by the time the 504 is returned, `wait_for` has cancelled the POST, httpx
    has torn the connection down and the upstream reads EOF. What this route
    cannot do is make a NON-STREAMING llama-server abandon a generation it
    has already started — no HTTP-level cancel exists for it — so the
    residual cost is upstream behaviour, not a release this handler skipped.
    """
    seen = {}

    async def handle(reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        n = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                n = int(line.split(b":")[1])
        if n:
            await reader.readexactly(n)
        seen["request_at"] = time.monotonic()
        try:
            seen["eof"] = (await asyncio.wait_for(reader.read(1), 5.0)) == b""
        except asyncio.TimeoutError:
            seen["eof"] = False
        seen["eof_at"] = time.monotonic()
        writer.close()

    srv = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    llm = _InflightLlm()
    llm.http_client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}")
    agent = MagicMock()
    agent.context.llm_client = llm
    try:
        with patch.object(R, "get_agent", return_value=agent), \
                patch("ghost_agent.core.llm._MAIN_FALLBACK_TIMEOUT_S", 0.3):
            resp = await R.api_generate(_Req({"prompt": "p", "model": "m"}))
            done_at = time.monotonic()
        assert resp.status_code == 504
        await asyncio.sleep(0.2)
        assert seen.get("eof") is True, "the upstream never saw the disconnect"
        assert seen["eof_at"] <= done_at + 0.2, "the close lagged the 504"
    finally:
        await llm.http_client.aclose()
        srv.close()
        await srv.wait_closed()


# ── 11. an upstream fault is the upstream's status ───────────────────────────

async def test_an_upstream_5xx_on_generate_is_a_502_not_a_traceback():
    """`raise_for_status()` sat under the generic `except Exception`, so a 503
    from llama-server came back as a 500 with an error id and a logged
    traceback — blaming the agent, and burying the one fact that matters.
    `catch_all` has always answered 502 for the same condition."""
    llm = _InflightLlm()
    request = httpx.Request("POST", "http://main:8000/v1/chat/completions")
    upstream = httpx.Response(503, request=request, text="slot unavailable")

    async def post(url, json=None):
        return upstream

    llm.http_client.post = AsyncMock(side_effect=post)
    agent = MagicMock()
    agent.context.llm_client = llm
    with patch.object(R, "get_agent", return_value=agent):
        resp = await R.api_generate(_Req({"prompt": "p", "model": "m"}))
    assert resp.status_code == 502, resp.status_code
    assert "503" in json.loads(bytes(resp.body))["error"]
    # the slot is free either way
    assert llm.foreground_tasks == 0 and not llm._main_node_lock.locked()


async def test_an_agent_side_fault_on_generate_is_still_a_500():
    """Control: only an UPSTREAM status becomes a 502. A fault on this side
    keeps the opaque 500 + error id, so the two cannot be confused."""
    llm = _InflightLlm()
    llm.http_client.post = AsyncMock(side_effect=RuntimeError("our bug"))
    agent = MagicMock()
    agent.context.llm_client = llm
    with patch.object(R, "get_agent", return_value=agent):
        resp = await R.api_generate(_Req({"prompt": "p"}))
    assert resp.status_code == 500


# ── 12. the proxy booking survives a BaseException ───────────────────────────

async def test_the_proxy_booking_is_released_on_a_base_exception():
    """This app raises a BaseException through the RECEIVE channel on
    purpose — `api/body_limit.BodyTooLarge` is one so it can sail past
    FastAPI's and these handlers' `except Exception` and reach the middleware
    as a 413. It is thrown from inside httpx on the `request.stream()` branch,
    past a release that only covers `Exception`, leaving `foreground_tasks`
    at 1 and the in-flight slot booked forever."""
    from ghost_agent.api.body_limit import BodyTooLarge
    assert not issubclass(BodyTooLarge, Exception), "this pin's premise"
    llm = _InflightLlm()
    llm.http_client.build_request = MagicMock(return_value=MagicMock())
    llm.http_client.send = AsyncMock(side_effect=BodyTooLarge(1024))
    agent = MagicMock()
    agent.context.llm_client = llm
    req = _Req(headers={"content-type": "application/json"}, raw=b'{"stream": true}')
    with patch.object(R, "get_agent", return_value=agent):
        with pytest.raises(BodyTooLarge):
            await R.catch_all(req, "v1/chat/completions")
    assert llm.foreground_tasks == 0, "the booking leaked past the BaseException"
    assert llm._own_inflight(llm.upstream_url) == 0


@pytest.mark.parametrize("role,consulted", [("owner", True), ("member", False)])
async def test_a_member_request_never_reads_the_stored_session(monkeypatch, role, consulted):
    """§4KJ R9: a stored session is the owner's history; a member-role request
    carrying `session_id` must not have it merged in (nor its turn appended).
    The wedged store is the probe: consulting it answers 504."""
    monkeypatch.setattr(R, "_STORE_CALL_TIMEOUT_S", 0.2)
    started, stuck = _wedged_store()
    from ghost_agent.core import sessions as S
    monkeypatch.setattr(S, "get_session_store",
                        lambda ctx: SimpleNamespace(get=stuck, append_turn=stuck))
    agent = _chat_agent()
    agent.handle_chat = AsyncMock(return_value=("hi", 0, "r1"))
    req = _Req({"messages": [{"role": "user", "content": "hi"}], "stream": False, "session_id": "s1"},
               headers={"X-Ghost-Requester": role})
    with patch.object(R, "get_agent", return_value=agent):
        resp = await R.chat_proxy(req, MagicMock())
    assert started.is_set() is consulted, role
    assert (resp.status_code == 504) is consulted, resp.status_code


async def test_a_refused_member_label_is_final_not_retryable(monkeypatch):
    """§4KJ R10: `member_turn` mapped to 503, so the bot retried every such
    reaction after 5 s and logged a WARNING each time."""
    from ghost_agent.core import feedback as FB
    monkeypatch.setattr(FB, "apply_human_label",
                        lambda *a, **k: {"ok": False, "code": "member_turn", "error": "x"})
    agent = _chat_agent()
    req = _Req({"request_id": "r1", "signal": "positive"})
    with patch.object(R, "get_agent", return_value=agent):
        resp = await R.feedback(req)
    assert resp.status_code == 403
