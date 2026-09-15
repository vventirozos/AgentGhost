"""§4GK round 5 (2026-09-13) — defects a read-only reviewer confirmed by
driving the REAL ASGI app under uvicorn over raw sockets, most of them INSIDE
§4GJ round 4's own fixes:

  * a far-future mtime made the whole workspace save 500 — `_member_info`
    clamped the stamp UP to the 1980 zip epoch and never DOWN from the DOS
    ceiling, and `struct.error` is not in the member loop's except set;
  * the DIRECTORY member was written outside the per-member `try`, so an
    unarchivable file was an `omitted` record + 200 while an unarchivable
    directory was a 500 — the two halves of one walk disagreeing;
  * the restore applied neither mode nor mtime, so round 4's stated defect
    ("nothing in the restored sandbox is executable any more") was still true
    for everyone who restores THROUGH THE AGENT;
  * `/api/generate` answered 500 where `catch_all` answers 502 for the common
    upstream failure — refused connection, RST mid-response;
  * `X-Ghost-Archive-Omitted` died in the interface proxy's fresh header dict
    and the web client said "saved successfully" on any `response.ok`;
  * `_stream_member`'s open dropped `O_NONBLOCK`, so a non-regular final
    component parked the archiver thread forever — `asyncio.to_thread` has no
    timeout;
  * `_store_call` documented "the thread is NOT cancelled on timeout" for a
    case where the call had not started and WAS cancelled, nothing ever shut
    the pool down, and its width was the one constant of the pair that could
    not be tuned.

Nothing here asserts on source text: every pin drives the real handler, the
real helper, or (for app.js) the real listener body under node.
"""
import asyncio
import concurrent.futures
import json
import os
import stat as _stat
import sys
import threading
import time
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ghost_agent.api import routes as R

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.helpers import eval_js  # noqa: E402


# ── doubles ──────────────────────────────────────────────────────────────────

class _Req:
    """A request with a real body — `save_workspace` reads it."""

    def __init__(self, body=None, raw=b"", headers=None):
        self._body = body
        self._raw = raw
        self.headers = headers if headers is not None else {}
        self.query_params = {}

    async def json(self):
        if self._body is None:
            raise json.JSONDecodeError("no body", "", 0)
        return self._body

    async def body(self):
        return self._raw


class _Upload:
    """The UploadFile shape `_read_capped` actually uses."""

    def __init__(self, data: bytes):
        self._data = memoryview(data)

    async def read(self, n: int = -1) -> bytes:
        chunk = bytes(self._data[:n])
        self._data = self._data[n:]
        return chunk


def _ws_agent(sandbox: Path):
    agent = MagicMock()
    agent.context.sandbox_dir = sandbox
    agent.context.scratchpad = None
    return agent


async def _save(sandbox: Path):
    with patch.object(R, "get_agent", return_value=_ws_agent(sandbox)):
        return await R.save_workspace(_Req(raw=b""))


async def _load(sandbox: Path, zip_bytes: bytes):
    with patch.object(R, "get_agent", return_value=_ws_agent(sandbox)):
        return await R.load_workspace(_Req(raw=b""), _Upload(zip_bytes))


# ═══════════════════════════════════════════════════════════════════════════
# 1. a far-future mtime must not take the whole save down
# ═══════════════════════════════════════════════════════════════════════════

_YEAR_2200 = 7258118400          # os.utime(f, (7258118400, ...)) → 2200-01-01


@pytest.mark.parametrize("on_a_directory", [False, True])
async def test_a_far_future_mtime_does_not_500_the_whole_save(tmp_path, on_a_directory):
    """A REGRESSION round 4 shipped with its own fix. `FileHeader` packs the
    date as `(year-1980) << 9 | month << 5 | day` into a ushort, so a year
    ≥ 2108 raises `struct.error` — a plain `Exception`, which the member
    loop's `except (OSError, ValueError)` does not catch. It escaped
    `_build_zip` and the route answered an opaque 500, so ONE file with a
    bogus stamp made EVERY workspace save fail until someone found it — and
    the sandbox is model-writable (`touch -d 2200-01-01`, `os.utime`, an
    unpacked archive, clock skew). `writestr(str, data)` stamped the archive
    time and could not hit this; the ZipInfo that reads the file's own mtime
    can.

    Both halves of the walk are driven: the reviewer confirmed the 500 on a
    file AND on a directory."""
    sandbox = tmp_path / "sandbox"
    (sandbox / "deep").mkdir(parents=True)
    (sandbox / "deep" / "keep.txt").write_text("still here")
    target = (sandbox / "deep") if on_a_directory else (sandbox / "future.txt")
    if not on_a_directory:
        target.write_text("from the year 2200")
    os.utime(target, (_YEAR_2200, _YEAR_2200))

    resp = await _save(sandbox)
    assert isinstance(resp, R.FileResponse), (
        f"the save answered {getattr(resp, 'status_code', resp)} — a single "
        f"far-future stamp took the whole archive down")
    with zipfile.ZipFile(resp.path) as zf:
        names = set(zf.namelist())
        stamped = zf.getinfo("sandbox/deep/" if on_a_directory
                             else "sandbox/future.txt")
    assert "sandbox/deep/keep.txt" in names, names
    # clamped to the last instant a DOS date can hold, not silently dropped
    assert stamped.date_time == (2107, 12, 31, 23, 59, 58), stamped.date_time
    os.unlink(resp.path)


async def test_a_pre_1980_mtime_still_clamps_up(tmp_path):
    """Control for the other end: round 4's floor is not what round 5 moved.
    A 1906 stamp is still the zip epoch, not the archive time."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    old = sandbox / "ancient.txt"
    old.write_text("1906")
    os.utime(old, (-2_020_000_000, -2_020_000_000))      # 1906
    resp = await _save(sandbox)
    with zipfile.ZipFile(resp.path) as zf:
        assert zf.getinfo("sandbox/ancient.txt").date_time == (1980, 1, 1, 0, 0, 0)
    os.unlink(resp.path)


# ═══════════════════════════════════════════════════════════════════════════
# 2. an unarchivable DIRECTORY is an omitted record, like an unarchivable file
# ═══════════════════════════════════════════════════════════════════════════

async def test_an_unarchivable_directory_is_omitted_not_a_500(tmp_path):
    """Only the `fstat` was inside a `try`: `_member_info` and `writestr`
    were bare, so the two halves of ONE walk disagreed about what an
    unarchivable member is — an unreadable FILE became an `omitted` record
    and a 200, an unwritable DIRECTORY aborted the build and the route
    answered 500. That asymmetry is the mechanism behind the directory half
    of finding 1.

    The unarchivable-ness here is a name the zip format cannot hold — what a
    directory created with non-UTF-8 bytes produces on the Linux mount the
    sandbox actually runs on (APFS refuses to create one, so the host cannot
    show it; `UnicodeEncodeError` IS a `ValueError`)."""
    sandbox = tmp_path / "sandbox"
    (sandbox / "sub").mkdir(parents=True)
    (sandbox / "sub" / "file.txt").write_text("readable")
    (sandbox / "top.txt").write_text("also readable")

    real_writestr = zipfile.ZipFile.writestr

    def refuse_directories(self, zinfo_or_arcname, data, *a, **k):
        name = getattr(zinfo_or_arcname, "filename", zinfo_or_arcname)
        if name.startswith("sandbox/") and name.endswith("/"):
            raise UnicodeEncodeError("utf-8", name, 0, 1, "surrogates not allowed")
        return real_writestr(self, zinfo_or_arcname, data, *a, **k)

    with patch.object(zipfile.ZipFile, "writestr", refuse_directories):
        resp = await _save(sandbox)
    assert isinstance(resp, R.FileResponse), (
        f"an unarchivable directory answered {getattr(resp, 'status_code', resp)} "
        f"where an unarchivable file answers 200 + a record")
    assert resp.headers.get("x-ghost-archive-omitted") == "1", dict(resp.headers)
    with zipfile.ZipFile(resp.path) as zf:
        record = json.loads(zf.read("omitted.json"))[0]
        names = set(zf.namelist())
    assert record["path"] == "sub/" and "UnicodeEncodeError" in record["reason"]
    # and the rest of the walk still landed
    assert {"sandbox/top.txt", "sandbox/sub/file.txt"} <= names, names
    os.unlink(resp.path)


# ═══════════════════════════════════════════════════════════════════════════
# 3. save → load ROUND TRIP: the property, not the archive
# ═══════════════════════════════════════════════════════════════════════════

def _round_trip_sandbox(tmp_path) -> Path:
    sandbox = tmp_path / "sandbox"
    (sandbox / "nested").mkdir(parents=True)
    run = sandbox / "run.sh"
    run.write_text("#!/bin/sh\necho hi\n")
    run.chmod(0o755)
    os.utime(run, (1_600_000_000, 1_600_000_000))
    notes = sandbox / "notes.txt"
    notes.write_text("plain")
    notes.chmod(0o644)
    os.utime(notes, (1_500_000_000, 1_500_000_000))
    empty = sandbox / "empty_dir"
    empty.mkdir()
    empty.chmod(0o700)
    os.utime(empty, (1_400_000_000, 1_400_000_000))
    return sandbox


async def test_a_restored_workspace_is_still_executable_and_still_dated(tmp_path):
    """⚠ THE PIN ROUND 4 WROTE MEASURED THE ARCHIVE, NOT THE PROPERTY.
    `test_the_archive_keeps_modes_mtimes_and_empty_directories` asserts on
    `external_attr`/`date_time` inside the zip and passes — while the world
    its docstring names ("nothing in the restored sandbox is executable any
    more", "every mtime rewritten to the moment of the save") was still the
    real world for anyone restoring THROUGH THE AGENT: the restore did
    `write_bytes` + `mkdir` and applied neither. Only an external `unzip` saw
    the fix. Measured on this exact round trip pre-fix: `run.sh` 0o755 in the
    archive, 0o644 on disk, mtime = now.

    So the pin is a ROUND TRIP through both real handlers."""
    src = _round_trip_sandbox(tmp_path)
    resp = await _save(src)
    zip_bytes = Path(resp.path).read_bytes()
    os.unlink(resp.path)

    dst = tmp_path / "restored"
    dst.mkdir()
    result = await _load(dst, zip_bytes)
    assert result["status"] == "success"

    run = dst / "run.sh"
    notes = dst / "notes.txt"
    empty = dst / "empty_dir"
    assert run.is_file() and notes.is_file() and empty.is_dir()
    assert _stat.S_IMODE(run.stat().st_mode) == 0o755, (
        f"restored {oct(_stat.S_IMODE(run.stat().st_mode))} — nothing in the "
        f"restored sandbox is executable")
    assert _stat.S_IMODE(notes.stat().st_mode) == 0o644
    assert _stat.S_IMODE(empty.stat().st_mode) == 0o700
    # DOS stamps are 2-second granular; anything within that is the file's own
    # mtime and anything near `now` is the defect.
    assert abs(run.stat().st_mtime - 1_600_000_000) <= 2, run.stat().st_mtime
    assert abs(notes.stat().st_mtime - 1_500_000_000) <= 2
    assert abs(empty.stat().st_mtime - 1_400_000_000) <= 2
    assert run.read_text() == "#!/bin/sh\necho hi\n"


async def test_the_restore_still_refuses_to_write_through_a_symlink(tmp_path):
    """Control for the fix above: the mode/mtime work must not become the
    call that follows a link. The destination is already a symlink pointing
    OUT of the sandbox — the member is refused, the outside file keeps its
    bytes and its mode, and the rest of the archive still lands (§4GJ R3)."""
    src = tmp_path / "sandbox"
    src.mkdir()
    (src / "payload.txt").write_text("attacker bytes")
    (src / "innocent.txt").write_text("ok")
    resp = await _save(src)
    zip_bytes = Path(resp.path).read_bytes()
    os.unlink(resp.path)

    outside = tmp_path / "outside.txt"
    outside.write_text("host secret")
    outside.chmod(0o600)
    dst = tmp_path / "restored"
    dst.mkdir()

    real_iterdir, planted = Path.iterdir, []

    def plant_the_link(self):
        # planted where the wipe cannot remove it: after the wipe listed
        items = list(real_iterdir(self))
        if not planted and self.name == "restored":
            planted.append((dst / "payload.txt").symlink_to(outside))
        return iter(items)

    with patch.object(Path, "iterdir", plant_the_link):
        result = await _load(dst, zip_bytes)
    assert result["status"] == "success"
    assert planted and (dst / "payload.txt").is_symlink(), (
        "the link was never planted — this control is vacuous")
    assert outside.read_text() == "host secret", "the restore wrote through the link"
    assert _stat.S_IMODE(outside.stat().st_mode) == 0o600, "the chmod followed the link"
    assert (dst / "innocent.txt").read_text() == "ok"


# ═══════════════════════════════════════════════════════════════════════════
# 4. an upstream that never answered is the upstream's failure
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_refused_upstream_on_generate_is_a_502_not_a_traceback():
    """Round 4 closed only `HTTPStatusError` — llama-server up enough to
    reply. The COMMON case is the node being down or restarting, and that
    came back from `/api/generate` as a 500 with an error id and a logged
    traceback while `catch_all`, the other route onto the SAME upstream,
    answered 502. Driven against a real socket on a closed port."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                   # nothing listens here now

    llm = MagicMock()
    llm.upstream_url = f"http://127.0.0.1:{port}"
    llm.http_client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}")
    agent = MagicMock()
    agent.context.llm_client = llm
    try:
        with patch.object(R, "get_agent", return_value=agent):
            resp = await R.api_generate(_Req({"prompt": "p", "model": "m"}))
    finally:
        await llm.http_client.aclose()
    assert resp.status_code == 502, resp.status_code
    body = json.loads(bytes(resp.body))["error"]
    assert "ConnectError" in body, body


async def test_an_upstream_that_drops_mid_response_on_generate_is_a_502():
    """The other half of the same class: the node dies while answering, so
    httpx raises `RemoteProtocolError` instead of returning a status. Same
    verdict — 502, like `catch_all`."""
    llm = MagicMock()
    llm.upstream_url = "http://main:8000"
    llm.http_client.post = AsyncMock(
        side_effect=httpx.RemoteProtocolError("server disconnected"))
    agent = MagicMock()
    agent.context.llm_client = llm
    with patch.object(R, "get_agent", return_value=agent):
        resp = await R.api_generate(_Req({"prompt": "p", "model": "m"}))
    assert resp.status_code == 502, resp.status_code
    assert "RemoteProtocolError" in json.loads(bytes(resp.body))["error"]


async def test_a_generate_timeout_is_still_a_504():
    """Control: `TimeoutException` is a `RequestError` too, so the new 502
    handler must sit BELOW the 504 one or every slow node starts reading as
    a broken one."""
    llm = MagicMock()
    llm.upstream_url = "http://main:8000"
    llm.http_client.post = AsyncMock(side_effect=httpx.ReadTimeout("slow"))
    agent = MagicMock()
    agent.context.llm_client = llm
    with patch.object(R, "get_agent", return_value=agent):
        resp = await R.api_generate(_Req({"prompt": "p", "model": "m"}))
    assert resp.status_code == 504, resp.status_code


# ═══════════════════════════════════════════════════════════════════════════
# 5. the incomplete-archive marker has to reach a human
# ═══════════════════════════════════════════════════════════════════════════

def _interface_server():
    os.environ.setdefault("GHOST_API_KEY", "test-ghost-key")
    import interface.server as server
    return server


@pytest.mark.parametrize("omitted, expected", [("3", "3"), (None, None)])
def test_the_omitted_marker_survives_the_interface_proxy(omitted, expected):
    """The proxy built a fresh header dict and copied only
    `content-disposition`, so the one signal that the zip is SHORT died at
    the interface server on every browser save. `omitted.json` inside the
    archive survives — but nobody opens a zip they were told was fine."""
    server = _interface_server()
    from fastapi.testclient import TestClient

    def upstream(request):
        headers = {"content-disposition": 'attachment; filename="workspace.zip"'}
        if omitted is not None:
            headers["X-Ghost-Archive-Omitted"] = omitted
        return httpx.Response(200, content=b"PK\x03\x04zipbytes", headers=headers)

    fake = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    with patch.object(server, "_get_http_client", lambda: fake):
        client = TestClient(server.app)
        r = client.post("/api/workspace/save", json={"chat_history": []},
                        headers={"X-Ghost-Key": server.GHOST_API_KEY})
    assert r.status_code == 200, r.text
    assert r.headers.get("x-ghost-archive-omitted") == expected, dict(r.headers)
    assert "workspace.zip" in r.headers.get("content-disposition", "")
    assert r.content == b"PK\x03\x04zipbytes"


def _save_click_handler() -> str:
    src = (_ROOT / "interface" / "static" / "app.js").read_text()
    needle = "workspaceSaveBtn.addEventListener('click'"
    i = src.index(needle)
    j = src.index("{", src.index("=>", i))
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return "async function handler() " + src[j:k + 1]
    raise AssertionError("unbalanced braces in the workspace-save listener")


@pytest.mark.parametrize("header, wanted", [
    ("'2'", "MISSING"),
    ("null", "successfully"),
])
def test_the_web_client_does_not_call_a_short_archive_a_clean_save(header, wanted):
    """`if (!response.ok) throw …` then "Workspace saved successfully." — so
    a 200 carrying `X-Ghost-Archive-Omitted: 2` told the operator their whole
    workspace was on disk. They restore from it weeks later, the restore
    wipes the sandbox and writes back the readable subset, and the omission
    is what turned a permissions glitch into data loss.

    The listener is EXTRACTED AND RUN: a text assertion cannot see that the
    header is read from the response the handler actually got."""
    pre = """
let isProcessingRequest = false;
let chatHistory = [{ role: 'user', content: 'hi' }];
const messages = [];
function addMessage(role, text) { messages.push([role, text]); }
function toWireMessage(m) { return m; }
function toggleSendButtonUI() {}
async function _httpError(r, m) { return new Error(m); }
const activeFace = { setWorkingState() {}, triggerSpike() {} };
const _hdr = %s;
globalThis.fetch = async () => ({
    ok: true,
    headers: { get: (k) => (k.toLowerCase() === 'x-ghost-archive-omitted' ? _hdr : null) },
    blob: async () => ({ size: 9 }),
});
globalThis.window = { URL: { createObjectURL: () => 'blob:x', revokeObjectURL() {} } };
globalThis.document = {
    createElement: () => ({ click() {}, remove() {} }),
    body: { appendChild() {} },
};
""" % header
    out = eval_js(pre + _save_click_handler(),
                  "await (async () => { await handler(); return messages; })()")
    assert len(out) == 1, out
    role, text = out[0]
    assert role == "system"
    assert wanted in text, text
    if wanted == "MISSING":
        assert "2" in text and "omitted.json" in text, text


# ═══════════════════════════════════════════════════════════════════════════
# 6. the archiver's open cannot be parked by a non-regular file
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_swapped_in_fifo_cannot_park_the_archiver_forever(tmp_path, monkeypatch):
    """`_stream_member`'s docstring claims parity with the read helper's open
    — `O_RDONLY | O_NOFOLLOW | O_NONBLOCK` — but it dropped `O_NONBLOCK`,
    and that is the only flag that stops a non-regular final component
    blocking BEFORE the `S_ISREG` check can reject it. Measured: the routes
    open was still parked after 1.5 s on a FIFO the helper opened and
    rejected at once.

    It is reachable as a TOCTOU the model owns on its own mount: listed and
    stat'd as a regular file over the inline ceiling, then swapped for a FIFO
    or a device node before `os.open`. The walk here is wrapped to list the
    name the way it would after that swap. The cost is total — `_build_zip`
    runs inside `asyncio.to_thread`, which has no timeout, so the request
    never completes and a default-executor worker is gone for the life of
    the process."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "notes.txt").write_text("a real file")
    fifo = sandbox / "swapped.bin"
    os.mkfifo(fifo)

    from ghost_agent.tools import file_system as FS
    real_walk = FS.walk_nofollow

    def walk_that_listed_it_before_the_swap(base):
        for dirpath, files, dir_fd in real_walk(base):
            if Path(dirpath) == sandbox:
                files = sorted(files + ["swapped.bin"])
            yield dirpath, files, dir_fd

    monkeypatch.setattr(FS, "walk_nofollow", walk_that_listed_it_before_the_swap)
    # the TOCTOU that reaches this open is a member listed OVER the inline
    # ceiling, i.e. the streaming branch
    monkeypatch.setattr(R, "_ZIP_INLINE_MEMBER_BYTES", -1)
    try:
        resp = await asyncio.wait_for(_save(sandbox), 5)
    finally:
        # release a pre-fix build parked in open(), so a red run does not
        # leave a wedged thread behind for the rest of the session
        try:
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass
    assert isinstance(resp, R.FileResponse), getattr(resp, "status_code", resp)
    with zipfile.ZipFile(resp.path) as zf:
        omitted = json.loads(zf.read("omitted.json"))
        assert zf.read("sandbox/notes.txt") == b"a real file"
    assert [o["path"] for o in omitted] == ["swapped.bin"], omitted
    assert "not a regular file" in omitted[0]["reason"]
    os.unlink(resp.path)


# ═══════════════════════════════════════════════════════════════════════════
# 7. the store pool: which outcome, a shutdown, and a knob
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_504_says_whether_the_mutation_ran_at_all(monkeypatch):
    """`_store_call`'s docstring said "the thread is NOT cancelled on timeout
    (a half-applied Chroma write is worse than a late one)". That held only
    while a worker was free: once every worker is busy, `wait_for` cancels a
    still-QUEUED `concurrent.futures.Future` successfully and the mutation
    never runs at all. One 504 covered two opposite facts — "your edit
    probably landed, late" and "your edit did not happen" — and the operator
    could not tell them apart."""
    tiny = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                 thread_name_prefix="ghost-store")
    monkeypatch.setattr(R, "_STORE_EXECUTOR", tiny)
    monkeypatch.setattr(R, "_STORE_EXECUTOR_WORKERS", 1)
    gate, running = threading.Event(), threading.Event()

    def wedged():
        running.set()
        gate.wait(10)

    held = asyncio.create_task(R._store_call(wedged, timeout=5))
    await asyncio.to_thread(running.wait, 5)

    ran = []

    def queued_mutation():
        ran.append(1)

    try:
        with pytest.raises(R.StoreCallTimeout) as queued:
            await R._store_call(queued_mutation, timeout=0.3)
        msg = str(queued.value)
        assert "did not return within" in msg, msg
        assert "never started" in msg and "did NOT take effect" in msg, msg
    finally:
        gate.set()
        await asyncio.gather(held, return_exceptions=True)
        tiny.shutdown(wait=False)
    await asyncio.sleep(0.1)
    assert ran == [], "a cancelled mutation ran anyway, after its own 504"


async def test_a_running_store_call_is_still_not_cancelled(monkeypatch):
    """Control, and the half of the docstring that was always true: a call
    that HAS started keeps running — a half-applied Chroma write is worse
    than a late one — and its 504 says so."""
    tiny = concurrent.futures.ThreadPoolExecutor(max_workers=2,
                                                 thread_name_prefix="ghost-store")
    monkeypatch.setattr(R, "_STORE_EXECUTOR", tiny)
    landed = []
    started = threading.Event()

    def slow_mutation():
        started.set()
        time.sleep(0.6)
        landed.append(1)

    try:
        with pytest.raises(R.StoreCallTimeout) as exc:
            await R._store_call(slow_mutation, timeout=0.2)
        assert started.is_set()
        assert "still running and may still take effect" in str(exc.value)
        await asyncio.to_thread(time.sleep, 1.0)
        assert landed == [1], "a started mutation was cancelled"
    finally:
        tiny.shutdown(wait=False)


# `test_the_store_pool_drops_its_queue_at_exit_without_waiting` lived here
# until §4GK round 6. It called `R._shutdown_store_executor(ex)` DIRECTLY on a
# private executor, mid-test — so it measured what the function does when you
# call it, and never entered interpreter shutdown, which is the only world
# where the defect lives. It passed identically with the `atexit.register`
# line deleted, and it passed while that registration was DEAD CODE: the
# executor's own hook (registered through `threading._register_atexit`, run
# inside `wait_for_thread_shutdown()`) drained the queue first and ours
# arrived afterwards with nothing to cancel. Measured: 3 of 3 queued
# mutations ran at shutdown. The replacement drives a real interpreter to
# exit — `tests/test_4gl_round6_routes.py::test_the_store_pool_drops_its_
# queue_at_interpreter_exit`, with `test_the_shutdown_hook_is_registered_
# where_it_runs_first` for the mechanism.


def test_the_store_pool_width_is_tunable_like_its_sibling_timeout(monkeypatch):
    """`_STORE_EXECUTOR_WORKERS = 8` was hard-coded next to an
    env-configurable `_STORE_CALL_TIMEOUT_S`, and one `notifications_pending`
    poll issues up to 50 sequential store calls — the pool width is exactly
    the number an operator watching store 504s reaches for."""
    assert R._STORE_EXECUTOR._max_workers == R._STORE_EXECUTOR_WORKERS
    monkeypatch.delenv("GHOST_STORE_WORKERS", raising=False)
    assert R._store_worker_count() == 8
    monkeypatch.setenv("GHOST_STORE_WORKERS", "3")
    assert R._store_worker_count() == 3
    # a typo must not raise at module import and stop the agent booting
    monkeypatch.setenv("GHOST_STORE_WORKERS", "eight")
    assert R._store_worker_count() == 8
