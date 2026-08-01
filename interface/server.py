import asyncio
import re
import json
import logging
import signal
import argparse
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote
import uuid
from fastapi import (
    FastAPI, BackgroundTasks, WebSocket, Request, WebSocketDisconnect,
    UploadFile, File, HTTPException, Header, Depends,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn
import os
import secrets

try:
    # Runtime: uvicorn runs with cwd=interface/, plain module on sys.path.
    import webpush_notify
except ModuleNotFoundError:
    # Tests import this file as `interface.server` from the repo root.
    from interface import webpush_notify

# GHOST_API_KEY is required. A hardcoded default (e.g. "ghost-secret-123")
# turns the interface server into an open relay for anyone who knows the
# default — if someone forgets to set the env var in prod, the fallback
# silently accepts anyone's requests. Fail loudly at import instead.
try:
    GHOST_API_KEY = os.environ["GHOST_API_KEY"]
except KeyError as _e:
    raise RuntimeError(
        "GHOST_API_KEY environment variable is required. "
        "Set it to the shared secret the agent's main API also uses."
    ) from _e
if not GHOST_API_KEY.strip():
    # An EMPTY/whitespace key passed the presence check above but made
    # `compare_digest` unmatchable, so every route — including `/` — answered
    # 401 while launchd's KeepAlive kept the dead service "up". The agent
    # treats an empty key as auth-OFF, so the two also disagreed. Fail loudly
    # at import instead of booting into a silent all-401 outage.
    raise RuntimeError(
        "GHOST_API_KEY is set but EMPTY. An empty key cannot authenticate any "
        "request — the interface would answer 401 to everything. Set the real "
        "shared secret (see ~/Data/AI/.ghost_api_key)."
    )
PI_VOICE_URL = os.environ.get("PI_VOICE_URL", "http://raspberrypi.local:8000")

# Hard limit on inbound request body sizes for upload paths. Enforced at the
# ASGI layer by BodySizeLimitMiddleware (see below) BEFORE the body is parsed,
# plus a handler-level backstop in _read_capped_upload.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

# JSON bodies (chat, tts, workspace/save) are far smaller than file uploads,
# so they get their own, much lower ceiling. Without it, `request.json()`
# slurped an arbitrarily large body into RAM — MAX_CHAT_MESSAGES capped the
# message COUNT, but a single message carrying a multi-GB string sailed
# through.
def _max_json_bytes() -> int:
    try:
        return int(os.environ.get("GHOST_INTERFACE_MAX_JSON_BYTES", str(10 * 1024 * 1024)))
    except (TypeError, ValueError):
        return 10 * 1024 * 1024

MAX_JSON_BYTES = _max_json_bytes()

# Hard limit on the number of messages in a chat body. The previous
# version had no cap upstream of the agent.
MAX_CHAT_MESSAGES = 500

# Wall-clock ceiling for a single chat request proxied to the agent
# backend. A long agent turn (deep browser automation, many sequential
# tool calls) can legitimately run for many minutes without flushing any
# SSE bytes to the HTTP response. With the old flat `timeout=600.0`,
# httpx's *read* timeout (the max gap between received bytes) fired at
# 600s, aborted the stream, and the UI rendered a bare "No response"
# because no assistant tokens had arrived yet. Raise the default and make
# it tunable via env. A separate, short connect timeout still fails fast
# when the backend is actually down rather than merely slow.
CHAT_TIMEOUT_S = float(os.environ.get("GHOST_CHAT_TIMEOUT", "1800"))  # 30 min
CHAT_CONNECT_TIMEOUT_S = float(os.environ.get("GHOST_CHAT_CONNECT_TIMEOUT", "10"))

# Per-send ceiling for websocket log broadcasts. A stalled client (lid-closed
# laptop, flaky Wi-Fi) doesn't raise on send — it back-pressures, i.e. the
# await simply never returns — so sends must be bounded or one dead peer
# freezes the whole broadcast (and, transitively, the tail pipe).
WS_SEND_TIMEOUT_S = float(os.environ.get("GHOST_WS_SEND_TIMEOUT", "2.0"))

# Wall-clock bound on reading an upstream ERROR body snippet for the SSE
# error frame. httpx's read timeout is per-gap, not total, so without this
# a drip-feeding failed upstream could hold the worker for the whole chat
# window while the client's reader sat parked on an empty stream.
UPSTREAM_ERROR_SNIPPET_TIMEOUT_S = 5.0


def _chat_timeout() -> "httpx.Timeout":
    """httpx Timeout for chat proxying: fail fast on connect, but allow a
    long read/write/pool window so a slow-but-alive agent turn isn't cut
    mid-flight. Passing the positional default sets read/write/pool to
    CHAT_TIMEOUT_S while `connect` is overridden explicitly."""
    return httpx.Timeout(CHAT_TIMEOUT_S, connect=CHAT_CONNECT_TIMEOUT_S)


def _proxy_timeout(total: float) -> "httpx.Timeout":
    """Timeout for the short-lived proxies (upload/download/save/load/voice):
    bounded total, fast connect failure."""
    return httpx.Timeout(total, connect=CHAT_CONNECT_TIMEOUT_S)


async def verify_interface_key(x_ghost_key: str | None = Header(default=None)) -> None:
    """Auth dependency for state-mutating interface proxies. Mirrors the
    main API's `verify_api_key` shape — clients send the same `X-Ghost-Key`
    header. Without this, anyone who can reach the interface server has
    full agent access (the proxies bake the upstream key into outgoing
    requests, so they were effectively open relays)."""
    # Constant-time compare (encode to bytes so a non-ASCII header can't
    # raise) — the main API mandates this (api/routes.py + a regression test);
    # the interface must not diverge to a timing-leaky `!=`.
    if not x_ghost_key or not secrets.compare_digest(
        x_ghost_key.encode("utf-8"), GHOST_API_KEY.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Missing or invalid X-Ghost-Key header.")


def _sse_error_frame(message: str, err_type: str) -> bytes:
    """SSE-shaped error frame the UI understands. One helper for ALL the
    terminal markers (buffer-cap, eviction, upstream failure) so the copies
    can't drift apart — the inline duplicates used to be patched one at a
    time."""
    payload = json.dumps({"error": {"message": message, "type": err_type}})
    return b"event: error\ndata: " + payload.encode("utf-8") + b"\n\n"


def _upstream_error_frame(msg: str) -> bytes:
    """SSE error frame for an upstream chat failure."""
    return _sse_error_frame(f"upstream chat failed: {str(msg)[:300]}", "UpstreamError")


def _err_json(status: int, msg: str) -> JSONResponse:
    """Helper to return a proper HTTP error code + body. Replaces the
    previous pattern of returning 200 with `{"error": "..."}` which broke
    every client-side retry/backoff loop."""
    return JSONResponse({"error": msg}, status_code=status)


class _JSONBodyError(HTTPException):
    """Malformed-JSON client error rendered in the interface's standard
    {"error": ...} shape (see the app-level handler below) instead of
    FastAPI's default {"detail": ...} — the bundled client (app.js) only
    reads `.error`, so a detail-shaped 400 showed a generic "HTTP 400"."""


async def _json_body_error_handler(request: Request, exc: "_JSONBodyError"):
    return _err_json(exc.status_code, exc.detail)


async def _parse_json_body(request: Request):
    """Parse a JSON request body, mapping malformed JSON to a 400.

    Previously a garbage body raised out of `request.json()` into the
    handler's generic except and surfaced as a 502 "upstream error" — which
    sent client retry/backoff loops chasing a request that can never
    succeed. Size is enforced upstream by BodySizeLimitMiddleware, so this
    only has to classify, not cap."""
    try:
        return await request.json()
    except (ValueError, UnicodeDecodeError):
        # json.JSONDecodeError subclasses ValueError.
        raise _JSONBodyError(status_code=400, detail="Request body is not valid JSON")

# Parse arguments.
# NB: in the production deployment this module is loaded BY UVICORN
# (`uvicorn server:app …`), so sys.argv belongs to uvicorn and --agent-log
# can never actually arrive here — the env var is the real override. The
# old default was ALSO wrong (`/Users/vasilis/AI/…`, missing `Data/`), so
# the live-log tail followed a nonexistent file and the UI's log stream
# (face pulses, planner monologue) was silently dead in prod.
_DEFAULT_AGENT_LOG = os.environ.get(
    "GHOST_AGENT_LOG", "/Users/vasilis/Data/AI/Logs/ghost-agent.log")
parser = argparse.ArgumentParser(description="Ghost Interface Server")
parser.add_argument("--agent-log", default=_DEFAULT_AGENT_LOG, help="Path to the agent log file")
args, unknown = parser.parse_known_args()
AGENT_LOG_PATH = args.agent_log

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GhostInterface")

# Strong references to the startup background tasks. The event loop holds
# only WEAK references to tasks — an unreferenced create_task() result can
# be garbage-collected mid-flight (documented asyncio footgun), silently
# killing the log stream or the janitor (whose death would let
# active_chat_tasks grow unbounded again).
_BACKGROUND_TASKS: "list[asyncio.Task]" = []


@asynccontextmanager
async def _lifespan(_app: "FastAPI"):
    """Startup/shutdown lifecycle (replaces the deprecated @app.on_event
    handlers): spawn the log streamer + janitor holding hard references,
    and tear both down — plus the shared httpx client — on shutdown."""
    global SHARED_HTTP_CLIENT
    _BACKGROUND_TASKS.append(asyncio.create_task(log_streamer()))
    _BACKGROUND_TASKS.append(asyncio.create_task(_active_chat_tasks_janitor()))
    _BACKGROUND_TASKS.append(asyncio.create_task(_notify_push_poller()))
    try:
        yield
    finally:
        for _task in _BACKGROUND_TASKS:
            _task.cancel()
        await asyncio.gather(*_BACKGROUND_TASKS, return_exceptions=True)
        _BACKGROUND_TASKS.clear()
        # Pending reply-push probes (12s grace sleeps) must not outlive
        # the loop — "Task was destroyed but it is pending" otherwise.
        for _task in list(_push_probe_tasks):
            _task.cancel()
        await asyncio.gather(*_push_probe_tasks, return_exceptions=True)
        _push_probe_tasks.clear()
        if SHARED_HTTP_CLIENT is not None and not SHARED_HTTP_CLIENT.is_closed:
            try:
                await SHARED_HTTP_CLIENT.aclose()
            except Exception:
                pass


app = FastAPI(lifespan=_lifespan)
app.add_exception_handler(_JSONBodyError, _json_body_error_handler)

# Shared module-level httpx client. Without this every chat request created
# a fresh AsyncClient (no connection reuse), which exhausts ephemeral ports
# under load. The client is closed in the lifespan shutdown path.
SHARED_HTTP_CLIENT: "httpx.AsyncClient | None" = None

def _get_http_client() -> httpx.AsyncClient:
    global SHARED_HTTP_CLIENT
    if SHARED_HTTP_CLIENT is None or SHARED_HTTP_CLIENT.is_closed:
        SHARED_HTTP_CLIENT = httpx.AsyncClient(timeout=_chat_timeout())
    return SHARED_HTTP_CLIENT

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Routes whose bodies are legitimately file-sized; everything else is JSON.
# NB: spelled without the "/api/" prefix so each full route string appears
# exactly once in this file — in its decorator — where the auth regression
# test (test_interface_proxies_require_auth) locates it and checks for the
# verify_interface_key dependency.
_UPLOAD_PATHS = {"/api/" + _suffix for _suffix in ("upload", "workspace/load", "stt")}
# Multipart framing (boundaries, part headers) rides on top of the file
# bytes, so upload paths get a little slack above MAX_UPLOAD_BYTES —
# otherwise a file at exactly the cap is rejected for its envelope.
_UPLOAD_CAP_SLACK_BYTES = 1024 * 1024


class _BodyTooLarge(BaseException):
    """Raised by the middleware's counting receive when a body overflows.

    Deliberately a BaseException: it must sail through BOTH FastAPI's
    body-parsing `except Exception` (which would remap it to a generic 400)
    and the handlers' own broad `except Exception` blocks (which would
    remap it to a 502) so the middleware can convert it to a proper 413."""
    def __init__(self, cap: int):
        self.cap = cap


class BodySizeLimitMiddleware:
    """Reject oversized request bodies BEFORE any parsing happens.

    A handler-level cap alone (`_read_capped_upload`) is not enough:
    FastAPI/Starlette parses the ENTIRE multipart body (spooling it to
    disk) before the endpoint runs, and `request.json()` slurps the whole
    body into RAM — by the time a handler checks sizes, a multi-GB request
    has already been received. This middleware rejects on the declared
    Content-Length up front, and for chunked bodies that declare nothing it
    counts bytes as they arrive and aborts on overflow."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return

        if scope.get("path") in _UPLOAD_PATHS:
            cap = MAX_UPLOAD_BYTES + _UPLOAD_CAP_SLACK_BYTES
        else:
            cap = MAX_JSON_BYTES

        declared = None
        for name, value in scope.get("headers") or ():
            if name == b"content-length":
                try:
                    declared = int(value)
                except (TypeError, ValueError):
                    declared = None
                break
        if declared is not None and declared > cap:
            await JSONResponse(
                {"error": f"Request body too large ({declared} > {cap} byte cap)"},
                status_code=413,
            )(scope, receive, send)
            return

        received = 0
        response_started = False

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body") or b"")
                if received > cap:
                    raise _BodyTooLarge(cap)
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge as exc:
            # A clean 413 is only possible if the app hasn't started
            # responding; otherwise re-raise and let the connection drop
            # (the alternative is a corrupt half-response).
            if response_started:
                raise
            await JSONResponse(
                {"error": f"Request body too large (> {exc.cap} byte cap)"},
                status_code=413,
            )(scope, receive, send)


# NB: middleware added LAST wraps the others, so adding the body limit
# BEFORE CORS keeps CORS outermost — its headers still decorate 413s.
app.add_middleware(BodySizeLimitMiddleware)

# CORS for development.
# `allow_origins=["*"]` + `allow_credentials=True` is forbidden by the CORS
# spec (browsers reject it). Auth here flows via the `X-Ghost-Key` header,
# not cookies, so credentials=False is the right setting.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global set of connected websockets
connected_websockets = set()

async def log_streamer():
    """Reads agent logs and broadcasts them to connected clients.

    Wrapped in a restart loop: if the tail subprocess ever EXITS (killed,
    binary error, resource pressure), `readline()` returns b'' and the old
    version just returned — the live log stream stayed dead until the whole
    interface server was restarted. Now it logs and respawns with backoff.
    """
    while True:
        try:
            await _log_streamer_once()
            logger.warning(
                "log tail for %s exited — restarting in 5s", AGENT_LOG_PATH)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("log streamer error (%s) — restarting in 5s", e)
        await asyncio.sleep(5)


async def _log_streamer_once():
    """One tail lifetime: spawn, broadcast lines until the process exits."""
    # stderr → DEVNULL: tail -F writes rotation notices ("file truncated",
    # "has appeared") to stderr, and nothing ever drains that pipe — once
    # the 64KB buffer fills, tail blocks and the live log stream freezes.
    process = await asyncio.create_subprocess_exec(
        "tail", "-n", "10", "-F", AGENT_LOG_PATH,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )

    logger.info(f"Started log streamer for {AGENT_LOG_PATH}")

    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                break

            # errors="replace": a log line with invalid UTF-8 (binary tool
            # output dumped into the log) used to raise UnicodeDecodeError
            # out of this loop — and because cleanup only ran on
            # CancelledError, every escape leaked a live tail while the
            # restart loop spawned another, 5 seconds apart, forever.
            decoded_line = line.decode('utf-8', errors='replace').strip()
            if decoded_line:
                # Detect error for reactivity
                is_error = "ERROR" in decoded_line or "Exception" in decoded_line

                message = json.dumps({
                    "type": "log",
                    "content": decoded_line,
                    "is_error": is_error
                })

                # Snapshot the set BEFORE iterating (a disconnect handler
                # mutating it mid-broadcast raised "Set changed size during
                # iteration"), and send CONCURRENTLY with a per-send
                # timeout: one stalled client (TCP backpressure never
                # raises, its send just never returns) used to freeze the
                # serial loop, stop draining the tail pipe, and wedge the
                # log stream for every other client.
                targets = []
                sends = []
                for ws in list(connected_websockets):
                    targets.append(ws)
                    sends.append(asyncio.wait_for(ws.send_text(message), WS_SEND_TIMEOUT_S))
                results = await asyncio.gather(*sends, return_exceptions=True)

                for ws, result in zip(targets, results):
                    if isinstance(result, BaseException):
                        # discard() instead of remove() — a concurrent
                        # disconnect handler may have already evicted it.
                        connected_websockets.discard(ws)
                        # Close so the endpoint's receive loop tears down
                        # too; bounded — close() on a wedged socket can
                        # itself hang.
                        try:
                            await asyncio.wait_for(ws.close(), timeout=1.0)
                        except Exception:
                            pass
    finally:
        # Reap the tail on EVERY exit path — normal EOF, cancellation, or
        # an unexpected error. Cleanup used to live only in a
        # CancelledError handler, so any other escape left the subprocess
        # running (and `process.wait()` with no timeout let a hung tail
        # block shutdown forever).
        try:
            process.terminate()
        except Exception:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                process.kill()
            except Exception:
                pass
        except asyncio.CancelledError:
            # A second cancellation can land while we're parked on the
            # wait above (uvicorn abandoning shutdown); still make sure a
            # wedged tail doesn't outlive us.
            try:
                process.kill()
            except Exception:
                pass
            raise

@app.get("/")
async def get(key: str | None = None):
    # The page itself must be gated, otherwise we'd be handing out the
    # injected API key to anyone who can reach the server. Accept the key
    # via `?key=...` so a user can bookmark the URL once.
    if not key or not secrets.compare_digest(
        key.encode("utf-8"), GHOST_API_KEY.encode("utf-8")
    ):
        return HTMLResponse(
            content="<h1>401 Unauthorized</h1><p>Append <code>?key=YOUR_KEY</code> to the URL.</p>",
            status_code=401,
        )
    html = (static_dir / "index.html").read_text()
    # Inject the key as a global the JS reads to attach X-Ghost-Key on every API call.
    # The manifest link is injected too (not static markup): its URL must
    # carry the key — iOS installs the PWA from the manifest's start_url,
    # and an unkeyed start_url would land the installed app on the 401
    # page. Keeping it out of index.html keeps the key out of the
    # unauthenticated /static mount.
    injected = (
        f'<script>window.GHOST_API_KEY={json.dumps(GHOST_API_KEY)};</script>\n'
        f'<link rel="manifest" href="/manifest.webmanifest?key={quote(GHOST_API_KEY)}">\n'
    )
    html = html.replace("</head>", f"{injected}</head>", 1)
    # `no-cache` forces the browser to revalidate the document with the
    # server on every load instead of serving a stale copy from disk
    # cache. Without this, an edit that bumps an asset cache-buster (e.g.
    # `style.css?v=2.7`) never takes effect until the user manually hard-
    # refreshes, because the cached HTML still references the old `?v=`.
    # The document itself carries the injected API key, so we also mark it
    # `private` to keep shared proxies from caching it.
    return HTMLResponse(
        content=html,
        status_code=200,
        headers={"Cache-Control": "no-cache, must-revalidate, private"},
    )

@app.get("/manifest.webmanifest")
async def get_manifest(key: str | None = None):
    """PWA manifest (key-gated like `/`). iOS requires a manifest for
    Add-to-Home-Screen installation, which in turn is the ONLY context
    Safari delivers web push to. start_url embeds the key so the
    installed app opens authenticated."""
    if not key or not secrets.compare_digest(
        key.encode("utf-8"), GHOST_API_KEY.encode("utf-8")
    ):
        return _err_json(401, "key required")
    manifest = {
        "name": "Ghost",
        "short_name": "Ghost",
        "description": "Ghost agent console",
        "start_url": f"/?key={quote(GHOST_API_KEY)}",
        "scope": "/",
        "display": "standalone",
        "background_color": "#05060a",
        "theme_color": "#05060a",
        "icons": [
            # ?v= busts icon caches on artwork swaps (ghost since
            # 2026-08-01); an installed PWA still shows the OLD icon until
            # removed and re-added — iOS snapshots it at install time.
            {"src": "/static/icons/icon-192.png?v=2", "sizes": "192x192",
             "type": "image/png"},
            {"src": "/static/icons/icon-512.png?v=2", "sizes": "512x512",
             "type": "image/png"},
        ],
    }
    return JSONResponse(
        manifest,
        media_type="application/manifest+json",
        headers={"Cache-Control": "no-cache, must-revalidate, private"},
    )

@app.get("/sw.js")
async def get_sw():
    # no-cache: a stale service worker is the worst kind of stale — it
    # intercepts push events with old code for up to 24h. Revalidate on
    # every load (the file is tiny).
    return FileResponse(
        static_dir / "sw.js", media_type="application/javascript",
        headers={"Cache-Control": "no-cache, must-revalidate"})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, key: str | None = None):
    # Gate the live log stream exactly like the `/` UI. The log carries tool
    # outputs and file contents; without this, ANY LAN host — or any website
    # the operator visits, since WebSockets bypass CORS (cross-site WebSocket
    # hijacking) — could connect and exfiltrate the stream. The browser client
    # passes the injected key as `?key=` (it can't set custom WS headers).
    if not key or not secrets.compare_digest(
        str(key).encode("utf-8"), GHOST_API_KEY.encode("utf-8")
    ):
        await websocket.close(code=1008)  # policy violation
        return
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        # discard() in a finally: ANY exit path must evict the socket —
        # catching only WebSocketDisconnect left sockets parked in the set
        # after unexpected receive errors until a later broadcast send
        # happened to fail. discard (not remove) because the broadcaster
        # may have already evicted it.
        connected_websockets.discard(websocket)

active_chat_tasks = {}
ACTIVE_TASK_TTL_SECONDS = 600  # 10 minutes
ACTIVE_TASK_HARD_CAP = 200     # absolute ceiling regardless of TTL

# Per-task cumulative buffer cap. Without a ceiling, a runaway upstream
# producer (or a client that opens a stream and never reads) lets
# `task["buffer"]` grow without bound and OOMs the interface server.
# Override via env if needed (e.g. for very large model outputs).
def _stream_cap_bytes() -> int:
    try:
        return int(os.environ.get("GHOST_INTERFACE_STREAM_CAP", "50000000"))
    except (TypeError, ValueError):
        return 50_000_000

# Global ceiling across ALL task buffers. The per-task cap alone still
# allowed ACTIVE_TASK_HARD_CAP × per-task cap (~10 GB) in the worst case.
# When the global cap is hit, the offending stream is truncated exactly
# like a per-task overflow (marker and all).
def _total_stream_cap_bytes() -> int:
    try:
        return int(os.environ.get("GHOST_INTERFACE_TOTAL_STREAM_CAP", "200000000"))
    except (TypeError, ValueError):
        return 200_000_000

def _total_buffered_bytes() -> int:
    """Bytes currently buffered across ALL chat tasks. The dict is bounded
    by ACTIVE_TASK_HARD_CAP entries, so the per-chunk sum stays cheap."""
    return sum(t.get("buffer_size", 0) for t in active_chat_tasks.values())

def _reclaim_buffered_bytes(needed: int, cap: int) -> None:
    """Evict the OLDEST done tasks' buffers until `needed` more bytes fit
    under `cap`. Done tasks keep their buffers ~10 min for resume replay;
    counting those against the global ceiling would punish brand-new
    streams (truncated on their first chunk) for memory only the janitor
    TTL would eventually free. Live tasks are never touched. Readers
    mid-replay hold the task dict by reference, so popping the entry
    cannot break them — a later resume of the evicted stream just 404s."""
    for tid, t in list(active_chat_tasks.items()):
        if _total_buffered_bytes() + needed <= cap:
            return
        if t.get("done") and t.get("buffer_size", 0) > 0:
            active_chat_tasks.pop(tid, None)

def _sweep_active_chat_tasks(now: float) -> None:
    """One janitor sweep: evict done tasks past their TTL and trim the dict
    to the hard cap. Extracted from the loop so it's directly testable."""
    stale = []
    for tid, t in list(active_chat_tasks.items()):
        if t.get("done"):
            # setdefault, not `or now`: computing a fallback WITHOUT
            # storing it reset the TTL clock on every sweep, so a done
            # task that was never stamped (e.g. cancelled before its
            # worker's first step — the CancelledError lands before the
            # try/finally is even entered) NEVER expired.
            finished_at = t.setdefault("finished_at", now)
            if now - finished_at > ACTIVE_TASK_TTL_SECONDS:
                stale.append(tid)
    for tid in stale:
        active_chat_tasks.pop(tid, None)
    # Hard cap evict-oldest. Done tasks go first; a live stream is
    # never silently dropped — popping a still-running entry left
    # its worker raising KeyError and its reader parked on
    # new_data_event forever (the client connection hung). If we
    # must evict a live task, cancel it and wake its reader first.
    overflow = len(active_chat_tasks) - ACTIVE_TASK_HARD_CAP
    if overflow > 0:
        for tid, t in list(active_chat_tasks.items()):
            if overflow <= 0:
                break
            if t.get("done"):
                active_chat_tasks.pop(tid, None)
                overflow -= 1
        for tid, t in list(active_chat_tasks.items()):
            if overflow <= 0:
                break
            bg = t.get("background_task")
            if bg is not None:
                bg.cancel()
            t["error"] = t.get("error") or "evicted: active task cap exceeded"
            t["done"] = True
            ev = t.get("new_data_event")
            if ev is not None:
                ev.set()
            active_chat_tasks.pop(tid, None)
            overflow -= 1

async def _active_chat_tasks_janitor():
    """Background sweeper that evicts done tasks past their TTL and trims
    the dict to a hard cap. Without this `active_chat_tasks` grows
    unbounded and the interface server OOMs on long uptimes."""
    while True:
        try:
            await asyncio.sleep(60)
            _sweep_active_chat_tasks(time.time())
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.warning(f"active_chat_tasks janitor error: {e}")

# ── Web push (2026-08-01) ──────────────────────────────────────────────
# Reply-ready pushes for turns that finish while the phone is locked/away,
# plus the agent's notify-severity ledger events. Transport lives in
# webpush_notify.py; iOS prerequisites (installed PWA + trusted cert) are
# documented in docs/interfaces/web_server.html.

def _push_ack_grace_s() -> float:
    try:
        return float(os.environ.get("GHOST_PUSH_ACK_GRACE", "12"))
    except (TypeError, ValueError):
        return 12.0

# Strong refs: an unreferenced create_task() result can be GC'd mid-run
# (same reason _BACKGROUND_TASKS exists).
_push_probe_tasks: set = set()

def _schedule_reply_push(task_id: str, user_text: str) -> None:
    t = asyncio.create_task(_push_if_unacked(task_id, user_text))
    _push_probe_tasks.add(t)
    t.add_done_callback(_push_probe_tasks.discard)

async def _push_if_unacked(task_id: str, user_text: str) -> None:
    """After the grace window, push unless the PAGE confirmed delivery.
    The ack (POST /api/chat/ack/{id}) can only come from live JS — a
    locked iPhone can't send it, which is exactly the case that needs
    the push."""
    try:
        await asyncio.sleep(_push_ack_grace_s())
        task = active_chat_tasks.get(task_id)
        if task is None or task.get("client_acked") or task.get("cancelled"):
            return
        if webpush_notify.subscription_count() == 0:
            return
        preview = (user_text or "your request").strip()
        if len(preview) > 120:
            preview = preview[:117] + "…"
        body = (f"Failed: {task['error'][:200]}" if task.get("error")
                else f"Reply ready — {preview}")
        await webpush_notify.broadcast_async(
            "Ghost", body, url=f"/?key={quote(GHOST_API_KEY)}",
            tag=f"ghost-turn-{task_id[:8]}")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"reply-ready push failed: {e}")

def _last_user_text(payload) -> str:
    try:
        for msg in reversed(payload.get("messages") or []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    except Exception:
        pass
    return ""

async def _notify_push_poller():
    """Forward the agent's notify-severity ledger records as web pushes.
    Own consumer name ('web-push') per the watermark contract — never
    steals records from the Slack bot or the in-page bell. Ack AFTER
    pushing (the bell's crash-re-serve convention); at most 5 pushes per
    cycle — overflow records are acked unpushed and stay visible in the
    bell rather than storming the lock screen."""
    consumer = "web-push"
    last_acked = None  # skip re-acking an unchanged watermark: an idle
    # ledger otherwise gets notify_consumers.json rewritten every 30s
    # around the clock, killing that file's mtime staleness diagnostic
    # (the Slack poller documents removing exactly this churn).
    while True:
        try:
            await asyncio.sleep(30)
            if webpush_notify.subscription_count() == 0:
                continue
            client = _get_http_client()
            r = await client.get(
                "http://localhost:8000/api/notifications/pending",
                params={"consumer": consumer, "limit": 20},
                headers={"X-Ghost-Key": GHOST_API_KEY}, timeout=10.0)
            if r.status_code != 200:
                continue
            data = r.json()
            if not data.get("enabled"):
                continue
            records = [] if data.get("baseline") else (data.get("records") or [])
            for rec in records[:5]:
                await webpush_notify.broadcast_async(
                    f"Ghost — {str(rec.get('phase', 'event')).upper()[:40]}",
                    str(rec.get("summary", ""))[:300],
                    url=f"/?key={quote(GHOST_API_KEY)}", tag="ghost-notify")
            watermark = data.get("watermark", 0)
            if records or data.get("baseline") or watermark != last_acked:
                resp = await client.post(
                    "http://localhost:8000/api/notifications/ack",
                    json={"consumer": consumer, "watermark": watermark},
                    headers={"X-Ghost-Key": GHOST_API_KEY}, timeout=10.0)
                if resp.status_code == 200:
                    last_acked = watermark
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.warning(f"notify push poller error: {e}")

@app.post("/api/chat", dependencies=[Depends(verify_interface_key)])
async def chat_proxy(request: Request):
    """Proxies chat requests to the Ghost Agent."""
    try:
        body = await _parse_json_body(request)
        # A valid-JSON but non-object body ([], "x", 5) would make body.get
        # below raise AttributeError → caught as a 502 + leaked str(e). Reject
        # it cleanly as a 400 client error.
        if not isinstance(body, dict):
            return _err_json(400, "Request body must be a JSON object.")
        # Reject grossly oversized chat bodies upfront — no point shipping
        # 100K messages downstream just to have the agent trim them.
        msgs = body.get("messages", [])
        if isinstance(msgs, list) and len(msgs) > MAX_CHAT_MESSAGES:
            return _err_json(413, f"Too many messages ({len(msgs)} > {MAX_CHAT_MESSAGES} cap)")
        is_streaming = body.get("stream", False)

        if is_streaming:
            task_id = str(uuid.uuid4())
            stream_cap = _stream_cap_bytes()
            active_chat_tasks[task_id] = {
                "buffer": [],
                "buffer_size": 0,
                "stream_cap": stream_cap,
                "truncated": False,
                "done": False,
                "error": None,
                "new_data_event": asyncio.Event(),
                "background_task": None
            }

            async def background_stream_worker(t_id, payload):
                client = _get_http_client()  # reuse pooled client
                total_cap = _total_stream_cap_bytes()
                # Hold the entry directly: if the janitor evicts this task
                # from the dict mid-stream, `active_chat_tasks[t_id]` would
                # raise KeyError in the handlers/finally below, leaving
                # done/new_data_event unset and the reader hung.
                t = active_chat_tasks.get(t_id)
                if t is None:
                    return
                try:
                    async with client.stream("POST", "http://localhost:8000/api/chat", json=payload, headers={"X-Ghost-Key": GHOST_API_KEY}, timeout=_chat_timeout()) as response:
                        try:
                            response.raise_for_status()
                        except httpx.HTTPStatusError:
                            # Read a bounded slice of the error body while
                            # the stream is still open — the bare status
                            # line ("Server error '500'") gave the UI
                            # nothing actionable. Bounded in TIME as well
                            # as size: httpx's read timeout is per-gap, so
                            # a drip-feeding upstream could otherwise hold
                            # the worker (and its parked reader) far past
                            # the point where the old raise_for_status
                            # failed instantly.
                            snippet = b""

                            async def _read_snippet():
                                nonlocal snippet
                                async for part in response.aiter_bytes():
                                    snippet += part
                                    if len(snippet) >= 512:
                                        break

                            try:
                                await asyncio.wait_for(
                                    _read_snippet(), timeout=UPSTREAM_ERROR_SNIPPET_TIMEOUT_S)
                            except Exception:
                                pass
                            detail = snippet.decode("utf-8", errors="replace").strip()
                            msg = f"agent returned HTTP {response.status_code}"
                            if detail:
                                msg += f": {detail[:300]}"
                            raise RuntimeError(msg) from None
                        async for chunk in response.aiter_bytes(chunk_size=None):
                            # Enforce buffer caps. A stuck client + runaway
                            # producer would otherwise OOM us; drop further
                            # chunks and flag the task as truncated.
                            chunk_len = len(chunk) if chunk else 0
                            # Global ceiling: ALL buffers together must
                            # stay under the total cap. Free finished
                            # streams' resume buffers first — otherwise a
                            # few large completed turns would brown-out
                            # every NEW stream until the janitor TTL fired.
                            if _total_buffered_bytes() + chunk_len > total_cap:
                                _reclaim_buffered_bytes(chunk_len, total_cap)
                            if _total_buffered_bytes() + chunk_len > total_cap:
                                t["truncated"] = True
                                t["truncated_reason"] = "global buffer cap exceeded"
                                t["new_data_event"].set()
                                break
                            if t["buffer_size"] + chunk_len > t["stream_cap"]:
                                # break, not continue: continuing kept draining
                                # the upstream for the rest of the turn (up to
                                # 30 min) while throwing every byte away.
                                # Breaking exits the `async with`, which closes
                                # the upstream connection promptly.
                                t["truncated"] = True
                                t["truncated_reason"] = "per-task buffer cap exceeded"
                                t["new_data_event"].set()
                                break
                            t["buffer"].append(chunk)
                            t["buffer_size"] += chunk_len
                            t["new_data_event"].set()
                    t["done"] = True
                except asyncio.CancelledError:
                    t["done"] = True
                    raise
                except Exception as e:
                    t["error"] = str(e)
                    t["done"] = True
                finally:
                    t["finished_at"] = time.time()
                    t["new_data_event"].set()
                    # Reply-ready push: fires after the grace window unless
                    # live page JS acked delivery (locked phones can't).
                    try:
                        _schedule_reply_push(t_id, _last_user_text(payload))
                    except Exception:
                        pass

            # Start detached generation task
            bg_task = asyncio.create_task(background_stream_worker(task_id, body))
            active_chat_tasks[task_id]["background_task"] = bg_task

            async def stream_generator():
                task = active_chat_tasks.get(task_id)
                if task is None:
                    # The janitor's hard-cap path can cancel+pop a live task
                    # before this generator's first iteration; dereferencing
                    # None then aborted the stream with a TypeError instead
                    # of the intended error marker.
                    yield _sse_error_frame(
                        "stream evicted before it started (server at capacity)",
                        "TaskEvicted",
                    )
                    return
                offset = 0  # CHUNK index, NOT byte index. task["buffer"]
                # is a list of bytes objects; yielding `task["buffer"][i]`
                # ships an entire HTTP chunk per iteration. The previous
                # version used a byte index which yielded one byte at a
                # time — devastating to throughput.
                while True:
                    # Clear BEFORE reading the buffer length. The previous
                    # order (drain, then clear, then wait) had a race: the
                    # worker could append + set between the drain check and
                    # the clear, wiping the set and stranding the reader
                    # until the *next* chunk. Clearing first means any set
                    # that races the drain survives the wait.
                    task["new_data_event"].clear()
                    while offset < len(task["buffer"]):
                        yield task["buffer"][offset]
                        offset += 1
                    if task["done"]:
                        # Surface truncation to the client as an SSE-shaped
                        # marker so the UI can warn the user that output
                        # was cut. Harmless when the consumer is plain text.
                        if task.get("truncated"):
                            yield _sse_error_frame(
                                "stream truncated: "
                                + task.get("truncated_reason", "per-task buffer cap exceeded"),
                                "BufferCapExceeded",
                            )
                        # An upstream failure (agent down, 5xx) was recorded
                        # in task["error"] but never emitted: the client got
                        # HTTP 200 with an EMPTY stream and rendered a bare
                        # "No response" with no diagnostic.
                        elif task.get("error"):
                            yield _upstream_error_frame(task["error"])
                        break
                    await task["new_data_event"].wait()

            headers = {
                "Cache-Control": "no-cache, no-store, must-revalidate, private",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Task-ID": task_id,
                "Access-Control-Expose-Headers": "X-Task-ID"
            }
            return StreamingResponse(stream_generator(), media_type="text/event-stream; charset=utf-8", headers=headers)
        else:
            # Reuse the shared pooled client so non-streaming chat doesn't
            # spin up a fresh AsyncClient (and a fresh connection pool) per
            # request, exhausting ephemeral ports under load.
            client = _get_http_client()
            response = await client.post(
                "http://localhost:8000/api/chat",
                json=body,
                headers={"X-Ghost-Key": GHOST_API_KEY},
                timeout=_chat_timeout(),
            )
            # Propagate the upstream status — returning the body with an
            # implicit 200 masks agent 4xx/5xx errors and defeats the
            # client's retry/backoff contract (_err_json).
            try:
                payload = response.json()
            except Exception:
                if response.status_code >= 400:
                    return _err_json(response.status_code, response.text[:500])
                raise
            return JSONResponse(payload, status_code=response.status_code)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat proxy error: {e}")
        return _err_json(502, f"Chat proxy error: {e}")

@app.get("/api/chat/resume/{task_id}", dependencies=[Depends(verify_interface_key)])
async def chat_resume_proxy(task_id: str, offset: int = 0):
    """Resume a buffered chat stream.

    ``offset`` is a CHUNK index into the task's buffer (the i-th upstream
    HTTP chunk), NOT a byte offset. The bundled client always sends 0
    (full replay); a client passing a byte count would silently skip
    whole chunks, so clamp to the valid range instead of trusting it.
    """
    task = active_chat_tasks.get(task_id)
    if not task:
        return _err_json(404, "Task not found or expired")

    async def stream_generator():
        client_offset = max(0, min(offset, len(task["buffer"])))
        while True:
            # See chat_proxy.stream_generator — clear before draining so a
            # set() racing the drain is preserved across the wait().
            task["new_data_event"].clear()
            while client_offset < len(task["buffer"]):
                yield task["buffer"][client_offset]
                client_offset += 1
            if task["done"]:
                # Replay the TERMINAL status too — a resumed stream that
                # dropped the truncation/error marker looked like a clean
                # completion to the client.
                if task.get("truncated"):
                    yield _sse_error_frame(
                        "stream truncated: "
                        + task.get("truncated_reason", "per-task buffer cap exceeded"),
                        "BufferCapExceeded",
                    )
                elif task.get("error"):
                    yield _upstream_error_frame(task["error"])
                break
            await task["new_data_event"].wait()

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate, private",
        "Pragma": "no-cache",
        "Expires": "0",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }
    return StreamingResponse(stream_generator(), media_type="text/event-stream; charset=utf-8", headers=headers)

@app.post("/api/chat/ack/{task_id}", dependencies=[Depends(verify_interface_key)])
async def chat_ack_proxy(task_id: str):
    """Client delivery confirmation (2026-08-01). Server-side "the stream
    drained" is NOT proof the user saw the reply — a locked iPhone's dead
    socket can swallow the tail into TCP buffers without an error. Only
    JS actually running in the page can confirm receipt; the absence of
    this ack within the grace window is what triggers the reply-ready
    web push."""
    task = active_chat_tasks.get(task_id)
    if not task:
        return _err_json(404, "not_found_or_done")
    task["client_acked"] = True
    return {"ok": True}

@app.get("/api/push/vapid", dependencies=[Depends(verify_interface_key)])
async def push_vapid_key():
    """VAPID public key for pushManager.subscribe. `enabled:false` when no
    keypair is provisioned (push feature off, not an error)."""
    key = webpush_notify.public_key()
    return {"enabled": bool(key), "key": key}

@app.post("/api/push/subscribe", dependencies=[Depends(verify_interface_key)])
async def push_subscribe(request: Request):
    body = await _parse_json_body(request)
    if not isinstance(body, dict):
        return _err_json(400, "Request body must be a JSON object.")
    sub = body.get("subscription")
    if not isinstance(sub, dict) or not webpush_notify.add_subscription(sub):
        return _err_json(400, "Malformed push subscription.")
    return {"ok": True, "count": webpush_notify.subscription_count()}

@app.post("/api/push/unsubscribe", dependencies=[Depends(verify_interface_key)])
async def push_unsubscribe(request: Request):
    body = await _parse_json_body(request)
    if not isinstance(body, dict):
        return _err_json(400, "Request body must be a JSON object.")
    endpoint = body.get("endpoint")
    if not isinstance(endpoint, str):
        return _err_json(400, "endpoint required")
    return {"ok": True, "removed": webpush_notify.remove_subscription(endpoint)}

@app.get("/api/chat/task/{task_id}/state", dependencies=[Depends(verify_interface_key)])
async def chat_task_state(task_id: str):
    """Cheap liveness probe for the in-flight-turn resume flow (2026-08-01).

    The client persists its task id across page kills (iOS lock/close) and
    asks here BEFORE reattaching: streaming a full replay just to discover
    a 404 would waste the buffer's bandwidth, and a plain resume 404 was
    indistinguishable from a network error in the UI. `exists:false` is a
    normal answer (TTL swept / proxy restarted), not an error — the client
    then falls back to session-store reconciliation."""
    task = active_chat_tasks.get(task_id)
    if not task:
        return {"exists": False}
    return {
        "exists": True,
        "done": bool(task.get("done")),
        "error": task.get("error"),
        "truncated": bool(task.get("truncated")),
        "chunks": len(task.get("buffer", [])),
    }

@app.post("/api/chat/cancel/{task_id}", dependencies=[Depends(verify_interface_key)])
async def chat_cancel_proxy(task_id: str):
    task = active_chat_tasks.get(task_id)
    if task and not task["done"]:
        if task["background_task"]:
            task["background_task"].cancel()
        task["done"] = True
        # The user ENDED this turn — a "Reply ready" push 12s after they
        # hit Stop would be noise. The push probe checks this flag.
        task["cancelled"] = True
        # Stamp + wake HERE, not only in the worker's finally: a reader
        # parked on new_data_event otherwise stayed parked until the
        # cancelled worker got around to its finally — and a worker
        # cancelled before its first step never runs that finally at all,
        # which also left the task without a finished_at for the janitor.
        task.setdefault("finished_at", time.time())
        ev = task.get("new_data_event")
        if ev is not None:
            ev.set()
        return {"status": "cancelled"}
    return _err_json(404, "not_found_or_done")

@app.post("/api/workspace/save", dependencies=[Depends(verify_interface_key)])
async def workspace_save_proxy(request: Request):
    """Proxies workspace save request and returns the zipped state."""
    try:
        body = await _parse_json_body(request)
        # Shared pooled client (consistent with chat) — the response is
        # closed by the stream generator; the client itself must stay open.
        client = _get_http_client()
        # Close resp on any failure BEFORE the StreamingResponse takes
        # ownership — otherwise every upstream error leaks an open
        # streamed connection.
        resp = None
        try:
            req = client.build_request(
                "POST", "http://localhost:8000/api/workspace/save", json=body,
                headers={"X-Ghost-Key": GHOST_API_KEY},
                timeout=_proxy_timeout(120.0),
            )
            resp = await client.send(req, stream=True)
            resp.raise_for_status()

            headers = {}
            if "content-disposition" in resp.headers:
                headers["content-disposition"] = resp.headers["content-disposition"]
        except Exception:
            if resp is not None:
                await resp.aclose()
            raise

        async def stream_generator():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
        return StreamingResponse(stream_generator(), media_type="application/zip", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workspace save proxy error: {e}")
        return _err_json(502, f"Workspace save proxy error: {e}")


async def _read_capped_upload(upload: UploadFile) -> bytes:
    """Read an UploadFile body in chunks and refuse anything bigger than
    MAX_UPLOAD_BYTES. BodySizeLimitMiddleware rejects oversized requests
    before parsing; this is the handler-level backstop (and the per-FILE
    cap — the middleware only sees the whole multipart body).

    Defensive notes:
      * Chunks must be `bytes`/`bytearray`. A `MagicMock(spec=UploadFile)`
        in tests returns AsyncMock-wrapped non-bytes values which would
        otherwise loop forever (MagicMock.__len__ is 0, and the truthy
        check passes), so we raise on any non-bytes chunk explicitly.
      * Hard iteration cap as a belt-and-braces guard against a hostile
        UploadFile implementation that returns endless empty-ish chunks.
    """
    total = 0
    chunks: list[bytes] = []
    max_iters = (MAX_UPLOAD_BYTES // 65536) + 32
    for _ in range(max_iters):
        chunk = await upload.read(65536)
        if not chunk:
            break
        if not isinstance(chunk, (bytes, bytearray)):
            raise HTTPException(
                status_code=400,
                detail=f"Upload returned non-bytes chunk ({type(chunk).__name__})",
            )
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB cap",
            )
        chunks.append(bytes(chunk))
    else:
        # Loop hit max_iters without breaking — pathological producer.
        raise HTTPException(
            status_code=413,
            detail="Upload iteration cap exceeded — refusing pathological producer.",
        )
    return b"".join(chunks)


@app.post("/api/workspace/load", dependencies=[Depends(verify_interface_key)])
async def workspace_load_proxy(file: UploadFile = File(...)):
    """Proxies workspace zip to unpack and load state."""
    try:
        body = await _read_capped_upload(file)
        client = _get_http_client()
        response = await client.post(
            "http://localhost:8000/api/workspace/load",
            files={"file": (file.filename, body, file.content_type)},
            headers={"X-Ghost-Key": GHOST_API_KEY},
            timeout=_proxy_timeout(120.0),
        )
        response.raise_for_status()
        return response.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workspace load proxy error: {e}")
        return _err_json(502, f"Workspace load proxy error: {e}")

@app.post("/api/upload", dependencies=[Depends(verify_interface_key)])
async def upload_proxy(file: UploadFile = File(...)):
    """Proxies file upload to the Ghost Agent."""
    try:
        body = await _read_capped_upload(file)
        client = _get_http_client()
        response = await client.post(
            "http://localhost:8000/api/upload",
            files={"file": (file.filename, body, file.content_type)},
            headers={"X-Ghost-Key": GHOST_API_KEY},
            timeout=_proxy_timeout(120.0),
        )
        response.raise_for_status()
        return response.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload proxy error: {e}")
        return _err_json(502, f"Upload proxy error: {e}")

@app.get("/api/download/{filename:path}", dependencies=[Depends(verify_interface_key)])
async def download_proxy(filename: str):
    """Proxies file download from the Ghost Agent."""
    # Reject path traversal upfront. The route matcher uses `:path` so it
    # would accept `..` segments. The downstream API also validates, but
    # we should fail fast.
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    try:
        client = _get_http_client()
        # Same leak guard as workspace_save: close on pre-stream failure.
        resp = None
        try:
            # quote(): the path param arrives DECODED, so a filename
            # containing "?", "#" or "%" pasted raw into the upstream URL
            # changed the request's meaning ("report?v=2.pdf" turned
            # everything after "?" into a query string). safe="/" keeps
            # subdirectory paths working.
            req = client.build_request(
                "GET", f"http://localhost:8000/api/download/{quote(filename)}",
                headers={"X-Ghost-Key": GHOST_API_KEY},
                timeout=_proxy_timeout(120.0),
            )
            resp = await client.send(req, stream=True)
            resp.raise_for_status()

            headers = {}
            if "content-type" in resp.headers:
                headers["content-type"] = resp.headers["content-type"]
            if "content-disposition" in resp.headers:
                headers["content-disposition"] = resp.headers["content-disposition"]
            # Agent-authored files serve from the SAME origin as the app and
            # its injected key. The UI only ever inlines images and PDFs —
            # but a pasted /api/download/x.html URL would render agent HTML
            # same-origin. Force HTML-ish types to download instead, and
            # never let the browser sniff its way to the same place.
            headers["x-content-type-options"] = "nosniff"
            ctype = headers.get("content-type", "").lower()
            if "html" in ctype or "xml" in ctype or "svg" in ctype:
                headers["content-disposition"] = "attachment"
        except Exception:
            if resp is not None:
                await resp.aclose()
            raise

        async def stream_generator():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(stream_generator(), headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download proxy error: {e}")
        return _err_json(502, f"Download proxy error: {e}")

@app.post("/api/stt", dependencies=[Depends(verify_interface_key)])
async def stt_proxy(request: Request):
    """Proxies audio blobs to the Raspberry Pi for Speech-to-Text."""
    try:
        form = await request.form()
        file = form.get("file")
        if file is None:
            return _err_json(400, "Missing 'file' part")
        # A text form field named "file" arrives as a plain str — calling
        # .read() on it raised AttributeError and surfaced as a misleading
        # 502. It's a client error: say so.
        if not hasattr(file, "read"):
            return _err_json(400, "'file' must be an uploaded file, not a text field")
        file_content = await _read_capped_upload(file)
        client = _get_http_client()
        files = {"file": (file.filename, file_content, file.content_type)}
        response = await client.post(
            f"{PI_VOICE_URL}/stt", files=files, timeout=_proxy_timeout(60.0))
        if response.status_code != 200:
            logger.error(f"PI STT Error Body: {response.text}")
            return _err_json(response.status_code, f"PI STT failed: {response.text[:200]}")
        return response.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT proxy error: {e}")
        return _err_json(502, f"STT proxy error: {e}")

@app.post("/api/tts", dependencies=[Depends(verify_interface_key)])
async def tts_proxy(request: Request):
    """Proxies text chunks to the Raspberry Pi for Text-to-Speech."""
    try:
        body = await _parse_json_body(request)
        if not isinstance(body, dict):
            return _err_json(400, "Request body must be a JSON object.")
        payload = {"text": body.get("text", "")}
        client = _get_http_client()
        # Same leak guard as workspace_save: close on pre-stream failure.
        resp = None
        try:
            req = client.build_request(
                "POST", f"{PI_VOICE_URL}/tts", json=payload,
                timeout=_proxy_timeout(60.0),
            )
            resp = await client.send(req, stream=True)
            if resp.status_code != 200:
                await resp.aread()
                logger.error(f"PI TTS Error Body: {resp.text}")
                # Explicit raise: raise_for_status() is a no-op for non-200
                # SUCCESS codes (e.g. 204), which would fall through and try
                # to stream an already-consumed body.
                raise RuntimeError(
                    f"PI TTS returned HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception:
            if resp is not None:
                await resp.aclose()
            raise

        async def stream_generator():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(
            stream_generator(),
            # Pass through what the Pi actually returned instead of
            # hardcoding audio/wav (the voice node may serve ogg/mp3).
            media_type=resp.headers.get("content-type", "audio/wav"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS proxy error: {e}")
        return _err_json(502, f"TTS proxy error: {e}")

# ── Agent API passthrough proxies (2026-07-28) ─────────────────────────
# The workspace UI (sessions rail, notifications, status strip, memory
# correction) consumes agent read/manage endpoints the interface never
# exposed — the browser only ever talks to THIS server (which injects the
# shared key), so each needs an authenticated passthrough. This is an
# explicit ALLOWLIST, not a /{path:path} catch-all: a generic forwarder
# would silently widen the agent surface reachable from the LAN every
# time the agent grows a route.

async def _proxy_agent_api(request: Request, method: str, upstream_path: str,
                           timeout_s: float = 30.0):
    """Forward a JSON request to the agent API, preserving query params and
    raw body, and propagate the upstream status + JSON body back. Bodies are
    already capped by BodySizeLimitMiddleware; forwarding the raw bytes
    (rather than re-parsing) keeps upstream error semantics intact."""
    client = _get_http_client()
    body = await request.body() if method in ("POST", "DELETE") else None
    headers = {"X-Ghost-Key": GHOST_API_KEY}
    if body:
        headers["Content-Type"] = request.headers.get(
            "content-type", "application/json")
    try:
        resp = await client.request(
            method, f"http://localhost:8000{upstream_path}",
            params=dict(request.query_params) or None,
            content=body if body else None,
            headers=headers,
            timeout=_proxy_timeout(timeout_s),
        )
    except Exception as e:
        logger.error(f"Agent proxy error ({upstream_path}): {e}")
        return _err_json(502, f"Agent proxy error: {e}")
    try:
        payload = resp.json()
    except Exception:
        return _err_json(resp.status_code if resp.status_code >= 400 else 502,
                         resp.text[:300])
    return JSONResponse(payload, status_code=resp.status_code)


@app.get("/api/health", dependencies=[Depends(verify_interface_key)])
async def health_proxy(request: Request):
    """Agent runtime health for the status strip/panel."""
    return await _proxy_agent_api(request, "GET", "/api/health")

@app.get("/api/sessions", dependencies=[Depends(verify_interface_key)])
async def sessions_list_proxy(request: Request):
    return await _proxy_agent_api(request, "GET", "/api/sessions")

@app.post("/api/sessions", dependencies=[Depends(verify_interface_key)])
async def sessions_create_proxy(request: Request):
    return await _proxy_agent_api(request, "POST", "/api/sessions")

def _safe_session_id(session_id: str) -> str:
    """Reject ids that could step OUT of the /api/sessions/ segment.

    quote(safe='') does not encode dots, and httpx normalizes dot
    segments — so a raw `/api/sessions/..` would reach `/api` upstream
    and land in the agent's /{path:path} catch-all, defeating the
    explicit-allowlist design by one segment. Session ids are uuids (or
    our web-* mints); a tight charset costs nothing."""
    sid = str(session_id or "")
    if not sid or ".." in sid or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", sid):
        raise HTTPException(status_code=400, detail="Invalid session id")
    return sid


@app.get("/api/sessions/{session_id}", dependencies=[Depends(verify_interface_key)])
async def sessions_get_proxy(request: Request, session_id: str):
    # quote(): same lesson as download_proxy — the decoded path param must
    # be re-encoded before being pasted into the upstream URL.
    return await _proxy_agent_api(
        request, "GET", f"/api/sessions/{quote(_safe_session_id(session_id), safe='')}")

@app.delete("/api/sessions/{session_id}", dependencies=[Depends(verify_interface_key)])
async def sessions_delete_proxy(request: Request, session_id: str):
    return await _proxy_agent_api(
        request, "DELETE", f"/api/sessions/{quote(_safe_session_id(session_id), safe='')}")

@app.get("/api/turns", dependencies=[Depends(verify_interface_key)])
async def turns_proxy(request: Request):
    return await _proxy_agent_api(request, "GET", "/api/turns")

@app.post("/api/turn/cancel", dependencies=[Depends(verify_interface_key)])
async def turn_cancel_proxy(request: Request):
    """The REAL turn cancel (releases the agent's turn lock) — the
    interface's /api/chat/cancel only stops the proxy-side stream."""
    return await _proxy_agent_api(request, "POST", "/api/turn/cancel")

@app.get("/api/notifications/pending", dependencies=[Depends(verify_interface_key)])
async def notifications_pending_proxy(request: Request):
    return await _proxy_agent_api(request, "GET", "/api/notifications/pending")

@app.post("/api/notifications/ack", dependencies=[Depends(verify_interface_key)])
async def notifications_ack_proxy(request: Request):
    return await _proxy_agent_api(request, "POST", "/api/notifications/ack")

@app.post("/api/memory/correct", dependencies=[Depends(verify_interface_key)])
async def memory_correct_proxy(request: Request):
    return await _proxy_agent_api(request, "POST", "/api/memory/correct")

@app.post("/api/memory/delete", dependencies=[Depends(verify_interface_key)])
async def memory_delete_proxy(request: Request):
    return await _proxy_agent_api(request, "POST", "/api/memory/delete")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
