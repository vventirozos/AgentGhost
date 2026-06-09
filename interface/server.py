import asyncio
import re
import json
import logging
import signal
import argparse
import sys
from pathlib import Path
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
PI_VOICE_URL = os.environ.get("PI_VOICE_URL", "http://raspberrypi.local:8000")

# Hard limit on inbound request body sizes for upload paths. The previous
# version had no cap — a multi-GB upload would exhaust disk and memory.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

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


def _chat_timeout() -> "httpx.Timeout":
    """httpx Timeout for chat proxying: fail fast on connect, but allow a
    long read/write/pool window so a slow-but-alive agent turn isn't cut
    mid-flight. Passing the positional default sets read/write/pool to
    CHAT_TIMEOUT_S while `connect` is overridden explicitly."""
    return httpx.Timeout(CHAT_TIMEOUT_S, connect=CHAT_CONNECT_TIMEOUT_S)


async def verify_interface_key(x_ghost_key: str | None = Header(default=None)) -> None:
    """Auth dependency for state-mutating interface proxies. Mirrors the
    main API's `verify_api_key` shape — clients send the same `X-Ghost-Key`
    header. Without this, anyone who can reach the interface server has
    full agent access (the proxies bake the upstream key into outgoing
    requests, so they were effectively open relays)."""
    if not x_ghost_key or x_ghost_key != GHOST_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Ghost-Key header.")


def _err_json(status: int, msg: str) -> JSONResponse:
    """Helper to return a proper HTTP error code + body. Replaces the
    previous pattern of returning 200 with `{"error": "..."}` which broke
    every client-side retry/backoff loop."""
    return JSONResponse({"error": msg}, status_code=status)

# Parse arguments
parser = argparse.ArgumentParser(description="Ghost Interface Server")
parser.add_argument("--agent-log", default="/Users/vasilis/AI/Logs/ghost-agent.log", help="Path to the agent log file")
args, unknown = parser.parse_known_args()
AGENT_LOG_PATH = args.agent_log

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GhostInterface")

app = FastAPI()

# Shared module-level httpx client. Without this every chat request created
# a fresh AsyncClient (no connection reuse), which exhausts ephemeral ports
# under load. The client is closed in the shutdown handler.
SHARED_HTTP_CLIENT: "httpx.AsyncClient | None" = None

def _get_http_client() -> httpx.AsyncClient:
    global SHARED_HTTP_CLIENT
    if SHARED_HTTP_CLIENT is None or SHARED_HTTP_CLIENT.is_closed:
        SHARED_HTTP_CLIENT = httpx.AsyncClient(timeout=_chat_timeout())
    return SHARED_HTTP_CLIENT

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

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
    """Reads agent logs and broadcasts them to connected clients."""
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

            decoded_line = line.decode('utf-8').strip()
            if decoded_line:
                # Detect error for reactivity
                is_error = "ERROR" in decoded_line or "Exception" in decoded_line

                message = json.dumps({
                    "type": "log",
                    "content": decoded_line,
                    "is_error": is_error
                })

                # Snapshot the set BEFORE iterating. The previous version
                # iterated `connected_websockets` directly; if a client
                # disconnected mid-broadcast and the disconnect handler
                # mutated the set, Python raised RuntimeError("Set changed
                # size during iteration").
                to_remove = set()
                for ws in list(connected_websockets):
                    try:
                        await ws.send_text(message)
                    except Exception:
                        to_remove.add(ws)

                for ws in to_remove:
                    # discard() instead of remove() — a concurrent
                    # disconnect handler may have already evicted it.
                    connected_websockets.discard(ws)

    except asyncio.CancelledError:
        # Clean up the tail subprocess on cancellation. The previous
        # version called `process.wait()` with no timeout, so a hung tail
        # blocked shutdown forever and old subprocesses accumulated as
        # zombies across server reloads.
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
        raise

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(log_streamer())
    asyncio.create_task(_active_chat_tasks_janitor())

@app.on_event("shutdown")
async def shutdown_event():
    global SHARED_HTTP_CLIENT
    if SHARED_HTTP_CLIENT is not None and not SHARED_HTTP_CLIENT.is_closed:
        try:
            await SHARED_HTTP_CLIENT.aclose()
        except Exception:
            pass

@app.get("/")
async def get(key: str | None = None):
    # The page itself must be gated, otherwise we'd be handing out the
    # injected API key to anyone who can reach the server. Accept the key
    # via `?key=...` so a user can bookmark the URL once.
    if key != GHOST_API_KEY:
        return HTMLResponse(
            content="<h1>401 Unauthorized</h1><p>Append <code>?key=YOUR_KEY</code> to the URL.</p>",
            status_code=401,
        )
    html = (static_dir / "index.html").read_text()
    # Inject the key as a global the JS reads to attach X-Ghost-Key on every API call.
    injected = (
        f'<script>window.GHOST_API_KEY={json.dumps(GHOST_API_KEY)};</script>\n'
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

@app.get("/sw.js")
async def get_sw():
    return FileResponse(static_dir / "sw.js", media_type="application/javascript")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        # discard() avoids KeyError if the broadcaster already evicted the
        # socket due to a send failure mid-broadcast.
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

async def _active_chat_tasks_janitor():
    """Background sweeper that evicts done tasks past their TTL and trims
    the dict to a hard cap. Without this `active_chat_tasks` grows
    unbounded and the interface server OOMs on long uptimes."""
    import time
    while True:
        try:
            await asyncio.sleep(60)
            now = time.time()
            stale = []
            for tid, t in list(active_chat_tasks.items()):
                if t.get("done"):
                    finished_at = t.get("finished_at") or now
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
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.warning(f"active_chat_tasks janitor error: {e}")

@app.post("/api/chat", dependencies=[Depends(verify_interface_key)])
async def chat_proxy(request: Request):
    """Proxies chat requests to the Ghost Agent."""
    try:
        body = await request.json()
        # Reject grossly oversized chat bodies upfront — no point shipping
        # 100K messages downstream just to have the agent trim them.
        msgs = body.get("messages", []) if isinstance(body, dict) else []
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
                import time as _time
                client = _get_http_client()  # reuse pooled client
                # Hold the entry directly: if the janitor evicts this task
                # from the dict mid-stream, `active_chat_tasks[t_id]` would
                # raise KeyError in the handlers/finally below, leaving
                # done/new_data_event unset and the reader hung.
                t = active_chat_tasks.get(t_id)
                if t is None:
                    return
                try:
                    async with client.stream("POST", "http://localhost:8000/api/chat", json=payload, headers={"X-Ghost-Key": GHOST_API_KEY}, timeout=_chat_timeout()) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes(chunk_size=None):
                            # Enforce per-task buffer cap. A stuck client +
                            # runaway producer would otherwise OOM us; drop
                            # further chunks and flag the task as truncated.
                            chunk_len = len(chunk) if chunk else 0
                            if t["buffer_size"] + chunk_len > t["stream_cap"]:
                                t["truncated"] = True
                                t["new_data_event"].set()
                                continue
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
                    t["finished_at"] = _time.time()
                    t["new_data_event"].set()

            # Start detached generation task
            bg_task = asyncio.create_task(background_stream_worker(task_id, body))
            active_chat_tasks[task_id]["background_task"] = bg_task

            async def stream_generator():
                task = active_chat_tasks.get(task_id)
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
                            yield (
                                b"event: error\n"
                                b"data: {\"error\": {\"message\": \"stream truncated: per-task buffer cap exceeded\", \"type\": \"BufferCapExceeded\"}}\n\n"
                            )
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
                
    except Exception as e:
        logger.error(f"Chat proxy error: {e}")
        return _err_json(502, f"Chat proxy error: {e}")

@app.get("/api/chat/resume/{task_id}", dependencies=[Depends(verify_interface_key)])
async def chat_resume_proxy(task_id: str, offset: int = 0):
    task = active_chat_tasks.get(task_id)
    if not task:
        return _err_json(404, "Task not found or expired")
        
    async def stream_generator():
        client_offset = offset
        while True:
            # See chat_proxy.stream_generator — clear before draining so a
            # set() racing the drain is preserved across the wait().
            task["new_data_event"].clear()
            while client_offset < len(task["buffer"]):
                yield task["buffer"][client_offset]
                client_offset += 1
            if task["done"]:
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

@app.post("/api/chat/cancel/{task_id}", dependencies=[Depends(verify_interface_key)])
async def chat_cancel_proxy(task_id: str):
    task = active_chat_tasks.get(task_id)
    if task and not task["done"]:
        if task["background_task"]:
            task["background_task"].cancel()
        task["done"] = True
        return {"status": "cancelled"}
    return _err_json(404, "not_found_or_done")

@app.post("/api/workspace/save", dependencies=[Depends(verify_interface_key)])
async def workspace_save_proxy(request: Request):
    """Proxies workspace save request and returns the zipped state."""
    try:
        body = await request.json()
        client = httpx.AsyncClient(timeout=120.0)
        # Close resp/client on any failure BEFORE the StreamingResponse
        # takes ownership — otherwise every upstream error leaks an open
        # streamed connection and a client pool (socket/FD exhaustion).
        resp = None
        try:
            req = client.build_request("POST", "http://localhost:8000/api/workspace/save", json=body, headers={"X-Ghost-Key": GHOST_API_KEY})
            resp = await client.send(req, stream=True)
            resp.raise_for_status()

            headers = {}
            if "content-disposition" in resp.headers:
                headers["content-disposition"] = resp.headers["content-disposition"]
        except Exception:
            if resp is not None:
                await resp.aclose()
            await client.aclose()
            raise
            
        async def stream_generator():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()
        return StreamingResponse(stream_generator(), media_type="application/zip", headers=headers)
    except Exception as e:
        logger.error(f"Workspace save proxy error: {e}")
        return _err_json(502, f"Workspace save proxy error: {e}")


async def _read_capped_upload(upload: UploadFile) -> bytes:
    """Read an UploadFile body in chunks and refuse anything bigger than
    MAX_UPLOAD_BYTES. Without this the proxy would happily slurp a multi-GB
    upload into memory before forwarding.

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
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/workspace/load",
                files={"file": (file.filename, body, file.content_type)},
                headers={"X-Ghost-Key": GHOST_API_KEY},
                timeout=120.0
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
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/upload",
                files={"file": (file.filename, body, file.content_type)},
                headers={"X-Ghost-Key": GHOST_API_KEY},
                timeout=120.0
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
        client = httpx.AsyncClient(timeout=120.0)
        # Same leak guard as workspace_save: close on pre-stream failure.
        resp = None
        try:
            req = client.build_request("GET", f"http://localhost:8000/api/download/{filename}", headers={"X-Ghost-Key": GHOST_API_KEY})
            resp = await client.send(req, stream=True)
            resp.raise_for_status()

            headers = {}
            if "content-type" in resp.headers:
                headers["content-type"] = resp.headers["content-type"]
            if "content-disposition" in resp.headers:
                headers["content-disposition"] = resp.headers["content-disposition"]
        except Exception:
            if resp is not None:
                await resp.aclose()
            await client.aclose()
            raise

        async def stream_generator():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

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
        file_content = await _read_capped_upload(file)
        async with httpx.AsyncClient() as client:
            files = {"file": (file.filename, file_content, file.content_type)}
            response = await client.post(f"{PI_VOICE_URL}/stt", files=files, timeout=60.0)
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
        body = await request.json()
        payload = {"text": body.get("text", "")}
        client = httpx.AsyncClient(timeout=60.0)
        # Same leak guard as workspace_save: close on pre-stream failure.
        resp = None
        try:
            req = client.build_request("POST", f"{PI_VOICE_URL}/tts", json=payload)
            resp = await client.send(req, stream=True)
            if resp.status_code != 200:
                await resp.aread()
                logger.error(f"PI 500 Error Body: {resp.text}")
                resp.raise_for_status()
        except Exception:
            if resp is not None:
                await resp.aclose()
            await client.aclose()
            raise
        
        async def stream_generator():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()
        
        return StreamingResponse(stream_generator(), media_type="audio/wav")
    except Exception as e:
        logger.error(f"TTS proxy error: {e}")
        return _err_json(502, f"TTS proxy error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
