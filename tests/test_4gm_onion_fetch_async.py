"""§4GM (2026-09-14): the onion fetch that aborted the whole agent.

The agent died twice in twelve hours — 2026-09-13 22:19:51 (pid 38307) and
2026-09-14 09:07:40 (pid 65198), the second one in the middle of request
319d0b37, one second after an onion engine's post-deadline fetch was still
running. Both crash reports carry the same stack:

    abort ← malloc_report ← ...POINTER_BEING_FREED_WAS_NOT_ALLOCATED
          ← Curl_freeset ← curl_easy_reset        (libcurl-impersonate)
          ← _cffi_f_curl_easy_reset               (curl_cffi)
          ← ...Python... ← thread_run             (a WORKER thread)

`_fetch_raw_html` was the only synchronous curl_cffi `Session` in the tree
and it ran on a thread pool. With ``stream=True`` that path duplicates the
easy handle (`curl_easy_duphandle`), resets the parent immediately, performs
on curl_cffi's own pool and resets the DUPLICATE from a done-callback on yet
another thread; libcurl-impersonate does not deep-copy that option set, so
the two resets free the same pointer and libmalloc aborts the process.
Reproduced locally in ~40s with that call shape on 8 threads.

The fix moves the fetch onto `AsyncSession` — pooled handle, reset on the
loop thread, no `duphandle` anywhere — which is what every other fetch in
the tree already uses, and keeps the old 16-way bound as `_onion_gate`.

The world each pin fails in: any tree where the onion fetch is built on a
synchronous curl_cffi Session again, where the body cap or the redirect
bookkeeping is lost in the move, or where the concurrency bound is dropped.
"""
import ast
import asyncio
import http.server
import inspect
import os
import socketserver
import subprocess
import sys
import threading
import time

import pytest

from ghost_agent.tools import darkweb_search as D

CYR = "<html><body>Привет мир</body></html>".encode("windows-1251")
CHUNK = b"z" * 16384

#: Bytes the server actually managed to push, across every connection. The
#: cap is only real if the READ stops — truncating a 64 MB body after
#: swallowing it whole leaves the OOM window wide open, and a length
#: assertion on the RESULT cannot tell those two worlds apart.
SERVED = {"bytes": 0}
_SERVED_LOCK = threading.Lock()


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves the shapes an onion engine really produces: an unbounded
    chunked body (no Content-Length), a redirect, a non-utf-8 page."""
    protocol_version = "HTTP/1.1"

    def do_GET(self):                                   # noqa: N802
        try:
            if self.path == "/redir":
                self.send_response(302)
                self.send_header("Location", "/small")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path == "/small":
                body = b"<html>ok</html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/cyr":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=windows-1251")
                self.send_header("Content-Length", str(len(CYR)))
                self.end_headers()
                self.wfile.write(CYR)
                return
            slow = self.path == "/slow"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for _ in range(4000):
                if slow:
                    # A trickling engine: slow enough that a 0.25s deadline
                    # always lands mid-body, and the byte counter below
                    # keeps growing for as long as the transfer is alive.
                    time.sleep(0.02)
                self.wfile.write(b"%x\r\n" % len(CHUNK) + CHUNK + b"\r\n")
                with _SERVED_LOCK:
                    SERVED["bytes"] += len(CHUNK)
        except Exception:                               # noqa: BLE001
            pass                                        # client hung up: expected

    def log_message(self, *a):                          # noqa: N802
        pass


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, *a):
        pass


@pytest.fixture
def http_base():
    with _SERVED_LOCK:
        SERVED["bytes"] = 0
    srv = _Server(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


# ── the shape itself ───────────────────────────────────────────────────────

def _module_tree():
    return ast.parse(inspect.getsource(D))


def _fetch_tree():
    fn = next(n for n in ast.walk(_module_tree())
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "_fetch_raw_html")
    return fn


def _called_attrs(node):
    return [n.func.attr for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]


def test_no_synchronous_curl_session_is_constructed_anywhere_in_the_module():
    """The defect was a SHAPE, not a value: `curl_cffi.requests.Session` with
    stream=True on a worker thread. Read it from the AST — a comment saying
    "async only" is not a guarantee."""
    built = [a for a in _called_attrs(_module_tree())
             if a in ("Session", "AsyncSession")]
    assert built, "no curl_cffi session is constructed — re-point this pin"
    assert "Session" not in built, (
        "a SYNCHRONOUS curl_cffi Session is back: that path duplicates the "
        "easy handle and resets it from two threads, which aborts the "
        "process inside libcurl (§4GM)")
    assert "AsyncSession" in built, built


def test_the_fetch_never_hands_the_request_to_a_thread():
    """`run_in_executor` / `to_thread` around the REQUEST is what made the
    crash reachable — a curl handle used from a thread. `_strip_html` may
    still use one; this looks inside `_fetch_raw_html` only."""
    called = _called_attrs(_fetch_tree())
    assert "run_in_executor" not in called, called
    assert "to_thread" not in called, called
    assert "AsyncSession" in called, called
    assert "aiter_content" in called, (
        "the async response must be drained with aiter_content: the sync "
        "iterator yields un-awaited coroutines, not bytes")


def test_an_abandoned_transfer_is_reaped_through_the_shared_helper():
    """The house contract for a curl-async stream is quit_now → aclose, and
    `utils.helpers.aclose_curl_response` is the one implementation of it —
    the sync `close()` is a silent no-op on these responses. Today the
    session's own `__aexit__` would abort the transfer a moment later, which
    is why no behavioural pin here can see the difference; the day this
    session is hoisted or reused, the un-reaped body streams on. Pin the
    contract, in the FINALLY that has to hold it."""
    fn = _fetch_tree()
    reaped = [n for n in ast.walk(fn)
              if isinstance(n, ast.Try)
              for h in n.finalbody
              for c in ast.walk(h)
              if isinstance(c, ast.Call)
              and getattr(c.func, "id", getattr(c.func, "attr", ""))
              == "aclose_curl_response"]
    assert reaped, (
        "nothing reaps the streamed response in a finally: an abandoned or "
        "CANCELLED fetch leaves the transfer running and hands a busy handle "
        "back to the pool")


# ── behaviour that had to survive the move ─────────────────────────────────

async def test_an_unbounded_body_is_abandoned_at_the_cap(http_base):
    """The server streams ~64 MB with no Content-Length. Reading it whole
    OOMs the host, which is why this fetch streams and stops."""
    status, body = await D._fetch_raw_html(f"{http_base}/big", None, 20.0)
    assert status == 200
    assert 0 < len(body) <= D._MAX_ONION_BODY_BYTES + 8192, len(body)


async def test_the_READ_stops_at_the_cap_not_just_the_result(http_base):
    """The endpoint offers ~64 MB. Truncating the decoded text afterwards
    satisfies a length assertion while the whole body has already been
    pulled into RAM — the exact OOM window the streaming cap was added to
    close. Ask the SERVER how much it got to send."""
    await D._fetch_raw_html(f"{http_base}/big", None, 20.0)
    await asyncio.sleep(0.5)                      # let a lingering write land
    with _SERVED_LOCK:
        served = SERVED["bytes"]
    assert served < 4 * D._MAX_ONION_BODY_BYTES, (
        f"the server pushed {served} bytes for a {D._MAX_ONION_BODY_BYTES}-byte "
        "cap: the read is not bounded, or the transfer was never aborted")


async def test_a_deadline_cancellation_really_aborts_the_transfer(http_base, monkeypatch):
    """The per-engine deadline is the production path that carried the crash:
    `asyncio.wait_for` used to cancel the AWAIT while the worker thread ran
    on to curl's own timeout — a lingering fetch with nobody watching it.
    On the loop the cancellation reaches the transfer itself. Raise the cap
    so the body cannot finish inside the deadline, then watch the server."""
    monkeypatch.setattr(D, "_MAX_ONION_BODY_BYTES", 60 * 1024 * 1024)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            D._fetch_raw_html(f"{http_base}/slow", None, 30.0), timeout=0.25)
    await asyncio.sleep(0.4)
    with _SERVED_LOCK:
        first = SERVED["bytes"]
    await asyncio.sleep(0.8)                       # ~40 more chunks if alive
    with _SERVED_LOCK:
        second = SERVED["bytes"]
    assert second == first, (
        f"the cancelled fetch is still streaming: {second - first} more bytes "
        "arrived after it was abandoned")
    # and the pool is still usable — a handle was not left mid-transfer
    status, body = await D._fetch_raw_html(f"{http_base}/small", None, 10.0)
    assert status == 200 and "ok" in body


async def test_the_url_the_body_really_came_from_is_recorded(http_base):
    """A dead endpoint that 302s to a working page is otherwise
    indistinguishable from a live one — the Ahmia misdiagnosis."""
    meta = {}
    status, body = await D._fetch_raw_html(f"{http_base}/redir", None, 20.0,
                                           meta=meta)
    assert status == 200 and "ok" in body
    assert meta.get("final_url", "").endswith("/small"), meta


async def test_a_declared_charset_still_beats_utf8(http_base):
    status, body = await D._fetch_raw_html(f"{http_base}/cyr", None, 20.0)
    assert status == 200
    assert "Привет мир" in body, repr(body[:80])


async def test_a_transport_failure_still_raises_for_the_caller_to_catch(http_base):
    """`_form_token` and `_run_engine` both catch — a swallowed failure here
    would look like an empty page and be cached as one."""
    with pytest.raises(Exception):
        await D._fetch_raw_html(f"{http_base}/small",
                                "socks5h://127.0.0.1:9999", 5.0)


# ── the bound the thread pool used to provide ──────────────────────────────

async def test_the_gate_bounds_concurrent_fetches_and_follows_the_loop():
    gate = D._onion_gate()
    assert gate is D._onion_gate(), "a fresh semaphore per call bounds nothing"
    for _ in range(D._MAX_CONCURRENT_ONION_FETCHES):
        await asyncio.wait_for(gate.acquire(), timeout=1.0)
    assert gate.locked(), "the bound is gone: an unbounded fan-out at Tor"
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(gate.acquire(), timeout=0.05)
    for _ in range(D._MAX_CONCURRENT_ONION_FETCHES):
        gate.release()

    # A semaphore that has waited on one loop cannot be awaited on another —
    # every test in this suite runs on a loop of its own, and so does the
    # agent after a restart. Take the gate from a SECOND loop (in a thread,
    # since a loop cannot be nested inside a running one) and it must be a
    # different object.
    seen = []
    t = threading.Thread(target=lambda: seen.append(asyncio.run(_gate_here())))
    t.start()
    t.join(10)
    assert seen and seen[0] is not gate, (
        "the gate is pinned to a dead loop: the first await on it after a "
        "loop change raises instead of bounding anything")


async def _gate_here():
    return D._onion_gate()


# ── the regression itself: the interpreter must survive the load ───────────

_DRIVER = r'''
import asyncio, os, sys, threading
sys.path.insert(0, {src!r})
sys.path.insert(0, {tests!r})
from test_4gm_onion_fetch_async import _Handler, _Server
from ghost_agent.tools import darkweb_search as D

D._MAX_CONCURRENT_ONION_FETCHES = 8
D._MAX_ONION_BODY_BYTES = 64 * 1024        # abandon every body early

srv = _Server(("127.0.0.1", 0), _Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = "http://127.0.0.1:%d" % srv.server_address[1]

async def main():
    for _ in range(3):
        tasks = []
        for i in range(48):
            if i % 3 == 2:                  # a dead SOCKS proxy: engines die
                tasks.append(D._fetch_raw_html(base + "/small",
                                               "socks5h://127.0.0.1:9999", 5.0))
            else:                           # abandoned mid-body at the cap
                tasks.append(D._fetch_raw_html(base + "/big", None, 20.0))
        res = await asyncio.gather(*tasks, return_exceptions=True)
        ok = sum(1 for r in res if not isinstance(r, BaseException))
        assert ok == 32, ok
    print("OK")

asyncio.run(main())
sys.stdout.flush()
os._exit(0)
'''


def test_a_hundred_abandoned_onion_fetches_do_not_abort_the_interpreter(tmp_path):
    """THE pin for the crash. 144 fetches — two thirds abandoned at the cap,
    one third failing through a dead SOCKS proxy — in one process. On the
    synchronous-Session tree this aborts inside libcurl (exit -6); the fetch
    is deliberately driven in a CHILD process so that abort is a test
    failure here rather than the death of the suite."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = tmp_path / "drive_onion.py"
    script.write_text(_DRIVER.format(src=os.path.join(root, "src"),
                                     tests=os.path.join(root, "tests")))
    env = dict(os.environ, GHOST_API_KEY="x", PYTHONPATH=os.path.join(root, "src"))
    proc = subprocess.run([sys.executable, "-u", str(script)], env=env,
                          capture_output=True, text=True, timeout=150)
    assert proc.returncode == 0, (
        f"the onion fetch killed its process: returncode {proc.returncode} "
        f"(-6 is SIGABRT — libcurl's heap)\nstdout: {proc.stdout[-800:]}\n"
        f"stderr: {proc.stderr[-800:]}")
    # Measured: the fixed tree finishes this in ~10s. The pre-fix shape
    # either aborts or WEDGES — the same lingering post-deadline fetches that
    # carried the crash — so the timeout above is part of the pin, not just
    # harness hygiene.
    assert "OK" in proc.stdout, proc.stdout[-800:]
