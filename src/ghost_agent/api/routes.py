import asyncio
import datetime
import json
import secrets
import shutil
import uuid
import httpx
import os
import io
import zipfile
from pathlib import Path
from fastapi import APIRouter, Request, BackgroundTasks, Security, HTTPException, Depends, Response, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.security.api_key import APIKeyHeader
from starlette.background import BackgroundTask
from ..utils.helpers import get_utc_timestamp
import logging
from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

router = APIRouter()

# Hard ceiling on messages in ONE /api/chat body. The durable store caps a
# session at 400; a request carrying more than this is a client bug or an
# attack, and the merge's DP table is O(stored x incoming).
MAX_REQUEST_MESSAGES = 2000


def _log_internal_error(context_label: str) -> str:
    """Log the current exception under a short correlation id and return the
    id. The wire response gets only the id — never the raw exception text —
    so a malformed-input repro can't reveal internal paths / URLs / Python
    error internals to the client."""
    eid = uuid.uuid4().hex[:8]
    logger.error("[err_id=%s] internal error in %s", eid, context_label, exc_info=True)
    return eid

API_KEY_NAME = "X-Ghost-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def _mark_foreground(agent, delta: int) -> None:
    """Adjust the llm client's active-user-request counter (see
    LLMClient.foreground_requests — it parks background LLM work for the
    whole life of a user request, not just per LLM call). Tolerant of
    mocked/missing clients: instrumentation must never break a request,
    so it only touches a counter that is actually an int."""
    try:
        llm = getattr(getattr(agent, "context", None), "llm_client", None)
        cur = getattr(llm, "foreground_requests", None)
        if isinstance(cur, int):
            llm.foreground_requests = max(0, cur + delta)
    except Exception:
        pass


def _restamp_sse_request_id(chunk, req_id: str):
    """Rewrite the ``id`` of every JSON ``data:`` frame in ``chunk`` to
    ``chatcmpl-<req_id>`` — the agent's OWN request id.

    ⚠ WHY (2026-09-05). On the streamed-final-generation path the agent
    forwards the upstream llama-server's SSE frames verbatim, and those
    carry llama's completion id (``chatcmpl-`` + 32 random characters).
    Every client that labels a turn (web UI thumbs, CLI) reads the request
    id off the frames — the contract `core/feedback.normalize_request_id`
    documents — so it POSTed llama's id, while the trajectory was filed
    under the agent's ``req_id`` (``4f6dc15d``). ``/api/feedback`` then
    answered "no trajectory found" for EVERY streamed turn, and the
    operator's labels — the scarcest signal the learning stack has — were
    lost silently. Slack was immune only because it mints its own
    ``X-Request-ID``. This is the one choke point every client-bound
    streaming frame passes through, so it is the place to fix the id.

    Bytes in, bytes out; ``[DONE]``, SSE comments, ``event:`` lines and
    anything unparseable pass through untouched. A chunk may carry several
    frames (split on the blank line); each JSON object frame gets the id,
    including ones that had none — the client captures the first ``id`` it
    sees, and an id-less first frame would leave it holding llama's from
    the second."""
    if not req_id:
        return chunk
    try:
        raw = chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
    except Exception:  # noqa: BLE001 — undecodable bytes: not ours to touch
        return chunk
    if '"id"' not in raw and "data:" not in raw:
        return chunk
    stamped = f"chatcmpl-{req_id}"
    out = []
    changed = False
    for frame in raw.split("\n\n"):
        stripped = frame.strip()
        if stripped.startswith("data:"):
            payload = stripped[len("data:"):].strip()
            # Only JSON OBJECTS are candidates: `[DONE]`, arrays and prose
            # fail the `{` gate (a separate `[DONE]` test was dead code —
            # the mutation batch proved it equivalent, §4EX).
            if payload.startswith("{"):
                try:
                    d = json.loads(payload)
                except ValueError:
                    d = None
                if isinstance(d, dict) and d.get("id") != stamped:
                    d["id"] = stamped
                    frame = "data: " + json.dumps(d, ensure_ascii=False)
                    changed = True
        out.append(frame)
    if not changed:
        return chunk
    result = "\n\n".join(out)
    return result.encode("utf-8") if isinstance(chunk, (bytes, bytearray)) else result


def _sse_delta_text(chunk) -> str:
    """Extract the assistant text from one SSE chunk the agent's streamer
    emitted (``data: {"choices":[{"delta":{"content": "..."}}]}``).

    Used to reconstruct what the user saw on the streamed-final-generation
    path, which bypasses the finalize tail — without this a session would
    record the user's message and no reply. Anything unparseable (the
    ``[DONE]`` sentinel, an SSE comment, an error frame) contributes "".
    """
    try:
        if isinstance(chunk, (bytes, bytearray)):
            chunk = chunk.decode("utf-8", "replace")
        if not isinstance(chunk, str):
            return ""
        out = []
        for line in chunk.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            d = json.loads(payload)
            for choice in (d.get("choices") or []):
                text = (choice.get("delta") or {}).get("content")
                if isinstance(text, str):
                    out.append(text)
        return "".join(out)
    except Exception:  # noqa: BLE001 — accounting must never break the stream
        return ""


# User-Agent marker the agent's own functional suite sends on its
# deliberate auth-rejection probes. Recognised ONLY from loopback (see
# verify_api_key) — it lowers a log level, it never grants access and never
# suppresses a line.
_SELF_TEST_UA = "ghost-functional-test"


def get_agent(request: Request):
    return request.app.state.agent

async def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
    agent = get_agent(request)
    configured = agent.context.args.api_key
    if configured:
        # Constant-time comparison to avoid a timing side-channel that
        # could let an attacker recover the key byte-by-byte. Compare as
        # BYTES: compare_digest raises TypeError on a non-ASCII str, which
        # would surface as a 500 (and a logged traceback) instead of a clean
        # 403 for an attacker-controlled header value.
        _supplied = str(api_key or "").encode("utf-8", "ignore")
        _expected = str(configured).encode("utf-8", "ignore")
        if not api_key or not secrets.compare_digest(_supplied, _expected):
            # Surface auth failures on the monitored stream (brute-force /
            # misconfigured client). Never log the key bytes.
            #
            # The agent's OWN functional suite deliberately probes with a
            # missing and a wrong key to prove auth works, so every run left
            # WARNING lines that are indistinguishable from a real intruder —
            # which is how a security signal gets learned-ignored. Those
            # probes are re-levelled to INFO, never suppressed, and only when
            # BOTH conditions hold:
            #   * the request came from LOOPBACK, and
            #   * it carries the self-test User-Agent marker.
            #
            # Deliberately NOT UA-only: a header is attacker-controlled, so
            # keying on it alone would hand anyone a switch to mute their own
            # probes. Requiring loopback means an attacker must already be on
            # this host to lower the level — and the line is still emitted
            # either way, so nothing can be made to disappear. The `ua` and
            # `ip` fields are always logged so a real hit stays identifiable.
            _ua = (request.headers.get("user-agent") or "")[:80]
            try:
                _ip = request.client.host if request.client else ""
            except Exception:  # noqa: BLE001
                _ip = ""
            _loopback = _ip in ("127.0.0.1", "::1", "localhost")
            _self_test = _loopback and _SELF_TEST_UA in _ua
            # The self-test tag goes FIRST: the log truncates long messages,
            # and a marker appended at the end was cut off — leaving the one
            # field that says "this is self-inflicted" invisible, which is
            # the whole point of the line.
            pretty_log(
                "Auth Rejected",
                (f"[own functional suite] path={request.url.path} ip={_ip}"
                 if _self_test else
                 f"path={request.url.path} ip={_ip or '?'} ua={_ua or '?'}"),
                icon=Icons.SHIELD,
                level="INFO" if _self_test else "WARNING",
            )
            raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@router.get("/")
async def root_check():
    return Response(content="Ollama is running", media_type="text/plain")

@router.head("/")
async def root_head():
    return Response(content="OK", media_type="text/plain")

@router.get("/api/version")
async def api_version():
    return {"version": "0.1.24"}

@router.post("/api/show")
async def api_show(request: Request):
    return {
        "modelfile": "# Modelfile generated by Ghost Agent\nFROM ghost-model",
        "parameters": "stop \"<|im_start|>\"\nstop \"<|im_end|>\"",
        "template": "<|im_start|>system\n{{ .System }}<|im_end|>\n<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n",
        "details": {
            "format": "gguf", "family": "qwen3", "families": ["qwen3"], "parameter_size": "35B-A3B", "quantization_level": "Q4_0"
        }
    }

@router.get("/api/tags")
async def list_models(request: Request):
    model_name = request.app.state.args.model
    return {
        "models": [
            {
                "name": model_name, "model": model_name, "modified_at": get_utc_timestamp(),
                "size": 1000000000, "digest": "sha256:qwen-3.6-35b-a3",
                "details": {"format": "gguf", "family": "qwen3", "families": ["qwen3"], "parameter_size": "35B-A3B", "quantization_level": "Q4_0"}
            }
        ]
    }

@router.get("/v1/models", dependencies=[Security(verify_api_key)])
async def list_openai_models(request: Request):
    model_name = request.app.state.args.model
    return {
        "object": "list",
        "data": [
            {"id": model_name, "object": "model", "created": int(datetime.datetime.now().timestamp()), "owned_by": "ghost-system"}
        ]
    }

def _bench_drain_int(agent) -> int:
    """§4BO health field, read defensively.

    R3 m-2 corrected the claim that used to be here: `int(MagicMock())`
    returns 1, it does not raise — the two real 500s came from
    `_bench_drain_banks` and `_bench_last_item_iso`. The guard stays
    because a non-numeric budget is still possible on a partial agent,
    but it is belt-and-braces, not the fix for the observed failure.
    """
    try:
        return int(getattr(agent, "_bench_drain_remaining", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _bench_drain_banks(agent):
    v = getattr(agent, "_bench_drain_banks", None)
    if not isinstance(v, (list, tuple)):
        return None
    return [str(b)[:64] for b in v if isinstance(b, str)]


def _safe_log(*args, **kwargs) -> None:
    """`pretty_log` that cannot fail a request.

    R3 MAJOR-3: the arm handler's narration ran AFTER the budget was
    written and was unwrapped, so a log write that raised turned a
    SUCCESSFUL arm into an HTTP 500. The operator reads that as "nothing
    happened" and walks away while the box drains 200 items behind it.
    """
    try:
        pretty_log(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.debug("bench-drain log suppressed: %s", exc)


def _iso_or_none(v):
    iso = getattr(v, "isoformat", None)
    if not callable(iso):
        return None
    try:
        out = iso()
    except Exception:  # noqa: BLE001
        return None
    return out if isinstance(out, str) else None


def _bench_started_iso(agent):
    return _iso_or_none(getattr(agent, "_bench_item_started_at", None))


def _bench_last_item_iso(agent):
    v = getattr(agent, "_bench_last_item_at", None)
    iso = getattr(v, "isoformat", None)
    if not callable(iso):
        return None
    try:
        out = iso()
    except Exception:  # noqa: BLE001
        return None
    return out if isinstance(out, str) else None


@router.get("/api/health", dependencies=[Security(verify_api_key)])
async def api_health(request: Request):
    """Runtime introspection for the operator + NetMon + the RSS supervisor
    (IMPROVEMENTS.md #21). Everything here is cheap and read-only. MUST stay
    registered ABOVE the /{path:path} catch-all so it isn't proxied upstream.

    Two silent-failure detectors it surfaces: `memory_system_loaded=false`
    means a degraded boot that disabled ALL biological phases while HTTP keeps
    answering; `biological_watchdog_alive=false` means the self-improvement
    daemon died."""
    import time as _time
    app = request.app
    agent = getattr(app.state, "agent", None)
    context = getattr(agent, "context", None)
    llm = getattr(context, "llm_client", None)

    rss_mb = None
    try:
        import psutil
        rss_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        pass

    try:
        live_tasks = len([t for t in asyncio.all_tasks() if not t.done()])
    except Exception:
        live_tasks = None

    bio_task = getattr(app.state, "biological_task", None)
    boot_mono = getattr(app.state, "boot_monotonic", None)

    sched = getattr(context, "scheduler", None)
    try:
        sched_jobs = len(sched.get_jobs()) if sched is not None else 0
    except Exception:
        sched_jobs = None

    # Off-main node pools (2026-07-11). Auxiliary LLM work (verifier,
    # conversation compaction, constraint audit, task classification…) can be
    # offloaded to secondary boxes so it never occupies the single main
    # inference slot. Without this block there was NO WAY to confirm from
    # outside whether a configured node was actually wired — you'd have to
    # read the boot log. Empty pool = that work runs on the main model.
    _pool_names = ("worker", "critic", "swarm", "coding", "vision", "image_gen")
    nodes = {}
    for _p in _pool_names:
        try:
            _clients = getattr(llm, f"{_p}_clients", None) or []
            nodes[_p] = [str(c.get("url") or "") for c in _clients
                         if isinstance(c, dict)]
        except Exception:  # noqa: BLE001 — health must never raise
            nodes[_p] = []

    # Circuit-breaker view of those nodes (2026-07-22). `nodes` above lists
    # what is CONFIGURED; the breaker tracks what is actually ANSWERING.
    # Without this, a tripped/dead node is indistinguishable from a healthy
    # one on the very endpoint used to verify node offload. Shape:
    # url → {state: closed|open|half_open, failures: int, open_since: float|None}.
    # Includes every URL the breaker has tracked (a configured node with no
    # traffic yet simply has no entry). getattr-guarded so a mock/partial
    # llm_client can never 500 the health endpoint.
    node_health = {}
    try:
        _get_status = getattr(getattr(llm, "circuit_breaker", None), "get_status", None)
        _status = _get_status() if callable(_get_status) else {}
        if isinstance(_status, dict):
            node_health = {
                str(_u): {
                    "state": _s.get("state"),
                    "failures": _s.get("failures"),
                    "open_since": _s.get("open_since"),
                }
                for _u, _s in _status.items() if isinstance(_s, dict)
            }
    except Exception:  # noqa: BLE001 — health must never raise
        node_health = {}

    return JSONResponse({
        "status": "ok",
        "rss_mb": rss_mb,
        "rss_limit_mb": float(os.environ.get("GHOST_MAX_RSS_MB", "0") or "0"),
        "uptime_s": round(_time.monotonic() - boot_mono, 1) if boot_mono else None,
        "asyncio_tasks": live_tasks,
        "foreground_requests": getattr(llm, "foreground_requests", None),
        "foreground_tasks": getattr(llm, "foreground_tasks", None),
        "biological_watchdog_alive": (bio_task is not None and not bio_task.done()),
        "memory_system_loaded": getattr(context, "memory_system", None) is not None,
        # §4BO: the drain budget is in-memory by design, so without this
        # the operator's only view of "did my drain run, how much is
        # left" is watching the live log stream.
        # Read DEFENSIVELY, per this endpoint's standing contract: a
        # partial or mocked agent must never 500 the health probe. Both
        # of these went in un-coerced first and did exactly that —
        # `int(MagicMock)` and `MagicMock.isoformat()` are not JSON
        # serializable, and ten health tests turned red.
        "bench_drain_remaining": _bench_drain_int(agent),
        "bench_drain_banks": _bench_drain_banks(agent),
        # `biological_watchdog_alive` reports the task OBJECT, so it
        # cannot tell "deferring on the idle floor" from "parked inside a
        # wedged solve". A timestamp that stops advancing can.
        "bench_last_item_at": _bench_last_item_iso(agent),
        # The START stamp is what separates "deferring on the idle floor"
        # (never advances, nothing running) from "parked inside a wedged
        # solve" (advanced, then froze while an item is still in flight).
        "bench_item_started_at": _bench_started_iso(agent),
        "scheduler_jobs": sched_jobs,
        "nodes": nodes,
        "node_health": node_health,
        "config": getattr(app.state, "resolved_config", {}),
    })


@router.post("/api/pull")
async def api_pull(request: Request):
    return {"status": "success"}

@router.delete("/api/delete", dependencies=[Security(verify_api_key)])
async def api_delete(request: Request):
    # Ollama-compatible: DELETE /api/delete removes a pulled model. We
    # serve a single canonical model and never let the user actually
    # delete it, but returning 200/success for ANY name was misleading —
    # clients couldn't distinguish "deleted my model" from "no such
    # model." Validate against the configured model name.
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        body = {}
    name = (body or {}).get("model") or (body or {}).get("name")
    configured = request.app.state.args.model
    if name and name != configured:
        return JSONResponse(
            {"error": {"message": f"model '{name}' not found", "type": "NotFound"}},
            status_code=404,
        )
    return {"status": "success"}

@router.post("/api/generate", dependencies=[Security(verify_api_key)])
async def api_generate(request: Request):
    agent = get_agent(request)
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, 400)

    prompt = body.get("prompt", "")
    model = body.get("model", "default")
    stream = body.get("stream", False)

    # Always request a non-streaming upstream completion: this handler
    # parses the body with resp.json(), and the local "stream" emulation
    # below sends the full response as a single NDJSON frame anyway.
    # Forwarding stream=True made the upstream return SSE frames, so
    # resp.json() raised and every Ollama-style streaming client got a 500.
    chat_payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    try:
        resp = await agent.context.llm_client.http_client.post("/v1/chat/completions", json=chat_payload)
        resp.raise_for_status()
        llm_resp = resp.json()
        content = llm_resp["choices"][0]["message"]["content"]
    except Exception:
        _eid = _log_internal_error("api_generate")
        return JSONResponse({"error": f"internal error (error_id={_eid})"}, 500)

    if stream:
         async def generator():
             yield json.dumps({
                 "model": model,
                 "created_at": get_utc_timestamp(),
                 "response": content,
                 "done": True
             }).encode('utf-8') + b"\n"
         return StreamingResponse(generator(), media_type="application/x-ndjson")
    else:
        return {
            "model": model,
            "created_at": get_utc_timestamp(),
            "response": content,
            "done": True
        }

@router.post("/chat", dependencies=[Security(verify_api_key)])
@router.post("/v1/chat/completions", dependencies=[Security(verify_api_key)])
@router.post("/api/chat", dependencies=[Security(verify_api_key)])
async def chat_proxy(request: Request, background_tasks: BackgroundTasks):
    agent = get_agent(request)
    # Body parse failures used to bubble up as a raw 500 with no JSON
    # payload — clients then saw an HTML error page instead of a
    # parseable error. Wrap the parse so malformed JSON / empty body
    # produces an OpenAI-shaped error JSON with a 400 status, matching
    # the error contract the rest of the route already provides for
    # handler exceptions.
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
        return JSONResponse(
            {
                "error": {
                    "message": f"Invalid JSON in request body: {e}",
                    "type": type(e).__name__,
                }
            },
            status_code=400,
        )
    if not isinstance(body, dict):
        return JSONResponse(
            {
                "error": {
                    "message": (
                        "Request body must be a JSON object, got "
                        f"{type(body).__name__}"
                    ),
                    "type": "InvalidRequestShape",
                }
            },
            status_code=400,
        )

    # ---- request validation (HTTP boundary) ---------------------------
    # Stops three classes of garbage from reaching `agent.handle_chat`:
    #   1. `messages: []` (or missing) — used to silently produce
    #      fabricated content from injected system state.
    #   2. `messages: "not a list"` / each-message-not-a-dict — used to
    #      crash `handle_chat` with `'str' object has no attribute 'get'`
    #      leaking Python internals as a 500.
    #   3. Unknown role values — passed through to the upstream LLM and
    #      surfaced as an "upstream error 400 template parser failed"
    #      string masquerading as an assistant reply.
    # Errors here are 422 with OpenAI-style `{error:{message,type}}`.
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse(
            {"error": {
                "message": "`messages` must be a non-empty list of message objects",
                "type": "InvalidRequestShape",
            }},
            status_code=422,
        )
    # ⚠ The AGENT'S OWN route has no body-size middleware (api/app.py adds
    # only CORS), so nothing bounded `messages` here — the interface's
    # 500-message cap protects proxied traffic only, and any key holder (the
    # Slack bot, bin/ scripts) reaches this route raw. `merge_history_detail`
    # then allocates a 401 x (m+1) DP table synchronously on the event loop:
    # measured 3.57s and +332 MB at m=100,000, from a ~3 MB body (R3 lens C).
    if len(messages) > MAX_REQUEST_MESSAGES:
        return JSONResponse(
            {"error": {
                "message": (f"too many messages ({len(messages)} > "
                            f"{MAX_REQUEST_MESSAGES} cap)"),
                "type": "InvalidRequestShape",
            }},
            status_code=413,
        )
    allowed_roles = {"system", "user", "assistant", "tool", "function"}
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return JSONResponse(
                {"error": {
                    "message": f"messages[{i}] must be an object, got {type(m).__name__}",
                    "type": "InvalidRequestShape",
                }},
                status_code=422,
            )
        role = m.get("role")
        if role not in allowed_roles:
            return JSONResponse(
                {"error": {
                    "message": (
                        f"messages[{i}].role={role!r} is not one of "
                        f"{sorted(allowed_roles)}"
                    ),
                    "type": "InvalidRequestShape",
                }},
                status_code=422,
            )

        # Content shape. `null`/absent content is valid ONLY on an assistant
        # message that carries tool_calls (the standard tool-calling turn).
        # Everywhere else a message with no usable content used to slip
        # through to the agent, which then fabricates a reply from nothing
        # (e.g. a `user` turn with content=null produced a greeting instead
        # of a 422). Reject null / wrong-typed / empty-user content here.
        content = m.get("content")
        if content is None:
            if not (role == "assistant" and m.get("tool_calls")):
                return JSONResponse(
                    {"error": {
                        "message": (
                            f"messages[{i}].content is required (null is allowed "
                            "only on an assistant message with tool_calls)"
                        ),
                        "type": "InvalidRequestShape",
                    }},
                    status_code=422,
                )
        elif not isinstance(content, (str, list)):
            return JSONResponse(
                {"error": {
                    "message": (
                        f"messages[{i}].content must be a string or list, got "
                        f"{type(content).__name__}"
                    ),
                    "type": "InvalidRequestShape",
                }},
                status_code=422,
            )
        elif role == "user" and isinstance(content, str) and not content.strip():
            return JSONResponse(
                {"error": {
                    "message": f"messages[{i}].content is empty; a user message must carry content",
                    "type": "InvalidRequestShape",
                }},
                status_code=422,
            )

    # `model` is optional in many clients (Ollama leaves it implicit).
    # If supplied AND it doesn't match the configured model, return 404
    # rather than silently rerouting to the upstream. We don't 404 a
    # missing key — that preserves Ollama compatibility.
    requested_model = body.get("model")
    configured_model = agent.context.args.model
    if requested_model and requested_model != configured_model:
        return JSONResponse(
            {"error": {
                "message": (
                    f"model {requested_model!r} not found; configured "
                    f"model is {configured_model!r}"
                ),
                "type": "ModelNotFound",
            }},
            status_code=404,
        )
    model = requested_model or configured_model
    stream = body.get("stream", False)

    # Extract Request ID if provided (for Slack Bot correlation)
    request_id = request.headers.get("X-Request-ID")

    # ---- durable sessions (2026-07-11) --------------------------------
    # With `session_id`, the SERVER is the source of truth for history: the
    # stored conversation is merged into this request, and the turn is
    # appended after it completes. `merge_history` tolerates both client
    # styles — a thin client sending only the new message, and a fat client
    # (today's web UI) replaying the whole conversation — so a fat client
    # can't double the history. Absent `session_id`, behaviour is exactly as
    # before (fully client-carried), so every existing client keeps working.
    _session_id = body.get("session_id")
    _sess_store = None
    _new_msgs = []
    if _session_id:
        from ..core.sessions import get_session_store, merge_history_detail
        _sess_store = get_session_store(agent.context)
        if _sess_store is not None:
            _existing = _sess_store.get(str(_session_id))
            _stored = _existing.messages if _existing is not None else []
            # The ALIGNMENT reports what this turn adds — the caller no
            # longer guesses it from `_merged[len(_stored):]` (§4BU C2).
            # That slice is only correct when merged is stored PLUS
            # something, and every fat-replay branch REPLACES stored: past
            # the 400-message cap it re-appended already-stored messages
            # every turn, and when the lengths matched it produced an empty
            # tail, so `_persist_session` early-returned and the user's
            # message and the reply were SILENTLY NEVER PERSISTED.
            _merged, _new_msgs = merge_history_detail(_stored, messages)
            body["messages"] = _merged
            messages = _merged

    def _persist_session(assistant_text: str) -> None:
        """Append this turn (new user messages + the reply) to the session.
        Runs AFTER the turn, so a failed turn never leaves a dangling user
        message with no reply. Never raises into the response path."""
        # An EMPTY tail is not a reason to drop the reply. When a client
        # re-sends a history whose last message is already stored (a prior
        # turn that produced no assistant content leaves the store ending in
        # a bare user message), the tail is empty — and this early return
        # then threw away the reply the agent had just generated, leaving
        # that user message unanswered in the durable file forever, so every
        # later replay repeated the same turn (R2 lens C).
        if _sess_store is None or not _session_id:
            return
        if not _new_msgs and not str(assistant_text or "").strip():
            return
        try:
            _sess_store.append_turn(str(_session_id), _new_msgs,
                                    str(assistant_text or ""))
        except Exception:  # noqa: BLE001
            _log_internal_error("session append")

    if stream:
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
        
        async def stream_generator():
            # Track whether any real content chunk has shipped to the
            # client. If yes, an additional `delta.content` error chunk
            # would APPEND to the partial response the user already
            # sees — "Here is the answ" + "CRITICAL SERVER ERROR:
            # ConnectionResetError" mashed together. Once content has
            # started, the `event: error` SSE frame is enough for
            # programmatic detection; emitting an extra content chunk
            # corrupts the visible reply.
            content_started = False
            # Mark a user request active for its WHOLE lifecycle (agent
            # loop + final-answer streaming) so background LLM work parks
            # instead of stealing the inference slot between this
            # request's tool calls. See LLMClient.foreground_requests.
            _mark_foreground(agent, +1)
            try:
                # Yield an SSE comment to send HTTP headers instantly and keep reverse proxies alive
                yield b": processing request...\n\n"

                content, created_time, req_id = await agent.handle_chat(body, background_tasks, request_id=request_id)

                if hasattr(content, '__aiter__'):
                    # Streamed final generation: accumulate the deltas so the
                    # session records what the user actually saw (this path
                    # bypasses the finalize tail, so nothing else knows the
                    # text). Parse-failure of a chunk just contributes nothing.
                    _acc = []
                    async for chunk in content:
                        content_started = True
                        _acc.append(_sse_delta_text(chunk))
                        # The frames are llama-server's, with llama's id;
                        # the feedback contract is OUR id (see the helper).
                        yield _restamp_sse_request_id(chunk, req_id)
                    _persist_session("".join(_acc))
                else:
                    _persist_session(content)
                    # A trivial-fast-path reply has NO trajectory (by
                    # design), so no label can ever land on it: say so on
                    # the wire and the UI renders no thumbs (§4EX). `is
                    # True` — a MagicMock agent answers truthy to anything.
                    _unlabelable = (getattr(agent, "is_trivial_reply", None) is not None
                                    and agent.is_trivial_reply(req_id) is True)
                    async for chunk in agent.context.llm_client.stream_openai(
                            model, content, created_time, req_id,
                            extra={"ghost": {"labelable": False}} if _unlabelable else None):
                        content_started = True
                        yield chunk
            except Exception as e:
                # Opaque error id on the wire (matches the non-streaming path) —
                # the raw str(e) leaked upstream URLs / file paths / Python
                # internals to the client on the DEFAULT (streaming) chat path.
                _eid = _log_internal_error("chat_proxy (streaming)")
                err_msg = f"internal server error (error_id={_eid})"
                error_event = {"error": {"message": err_msg, "type": "InternalError"}}
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode('utf-8')
                # Only emit a content chunk for visual display when
                # NO content has streamed yet — otherwise it gets
                # concatenated to the partial reply the client has
                # already rendered.
                if not content_started:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': err_msg}}]})}\n\n".encode('utf-8')
                yield b"data: [DONE]\n\n"
                return
            finally:
                _mark_foreground(agent, -1)

        return StreamingResponse(stream_generator(), media_type="text/event-stream", headers=headers)
    
    # Non-streaming fallback. Any exception in handle_chat used to bubble
    # up as a raw 500 with no JSON body — clients then saw an HTML error
    # page instead of a parseable error. Mirror the streaming path's
    # error-shape (OpenAI-style ``error.message``/``error.type``) so the
    # Slack bot and web UI can render a useful message to the user.
    _mark_foreground(agent, +1)
    try:
        content, created_time, req_id = await agent.handle_chat(body, background_tasks, request_id=request_id)
    except Exception as e:
        # The detailed stack lives in the log. The wire response gets a
        # generic message + an opaque exception type so a malformed-body
        # repro doesn't reveal Python internals (e.g. "'str' object has
        # no attribute 'get'"). The error-id lets you correlate to logs.
        err_id = uuid.uuid4().hex[:8]
        logger.error(
            f"[err_id={err_id}] Non-streaming error in chat_proxy: "
            f"{type(e).__name__}: {e}",
            exc_info=True,
        )
        return JSONResponse(
            {
                "error": {
                    "message": (
                        "internal error while handling chat request "
                        f"(error_id={err_id})"
                    ),
                    "type": "InternalError",
                }
            },
            status_code=500,
        )
    finally:
        _mark_foreground(agent, -1)

    _persist_session(content)

    # Token cost for the whole turn, summed across every upstream call it
    # made (tool rounds + verifier) — NOT one completion's usage. Surfaced
    # because the eval harness had a `tokens_used` field whose only producer
    # was a hardcoded 0, so every SuiteResult reported `total_tokens: 0`.
    # Omitted entirely when unknown: an absent key is honest, a zero is not.
    _payload = {
        "id": f"chatcmpl-{req_id}", "object": "chat.completion", "created": created_time, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "message": {"role": "assistant", "content": content},
        "done": True, "created_at": get_utc_timestamp()
    }
    try:
        _llm = getattr(getattr(agent, "context", None), "llm_client", None)
        _usage = _llm.usage_for(req_id) if _llm is not None else None
        # `isinstance(dict)` + int() are load-bearing, not defensive noise: a
        # MagicMock client returns a truthy mock whose .get() is also a mock,
        # which serialises to a 500 rather than a reply. Real values only.
        if isinstance(_usage, dict) and _usage:
            _in = int(_usage.get("tokens_in") or 0)
            _out = int(_usage.get("tokens_out") or 0)
            _payload["usage"] = {
                "prompt_tokens": _in,
                "completion_tokens": _out,
                "total_tokens": _in + _out,
                "prompt_tokens_details": {
                    "cached_tokens": int(_usage.get("cached_tokens") or 0)},
                "ghost_llm_calls": int(_usage.get("calls") or 0),
            }
    except Exception:  # noqa: BLE001 — never fail a reply over accounting
        pass
    return JSONResponse(_payload)

def _scratchpad_snapshot(sp) -> dict:
    """Namespace-preserving scratchpad snapshot for session export.

    Prefers the lock-held `export_state()` (real Scratchpad); degrades to
    the legacy flat `_data` read for scratchpad-like doubles that only
    carry `_data` (and returns {} when neither yields a dict)."""
    if sp is None:
        return {}
    try:
        state = sp.export_state()
        if isinstance(state, dict):
            return state
    except Exception:
        pass
    try:
        flat = dict(getattr(sp, "_data", {}) or {})
        return flat if isinstance(flat, dict) else {}
    except Exception:
        return {}


@router.post("/api/workspace/save", dependencies=[Security(verify_api_key)])
async def save_workspace(request: Request):
    """Packages the current chat history, scratchpad, and sandbox into a downloadable zip."""
    agent = get_agent(request)
    # Empty / malformed body used to surface as a bare 500
    # `Internal Server Error` with no JSON envelope. Accept "no body"
    # as "save with empty chat history" and reject malformed bodies
    # with a structured 400.
    raw = await request.body()
    if not raw:
        body = {}
    else:
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
            return JSONResponse(
                {"error": {"message": f"Invalid JSON in request body: {e}",
                           "type": type(e).__name__}},
                status_code=400,
            )
        if not isinstance(body, dict):
            return JSONResponse(
                {"error": {"message": f"Request body must be a JSON object, got {type(body).__name__}",
                           "type": "InvalidRequestShape"}},
                status_code=400,
            )
    chat_history = body.get("chat_history", [])
    session_data = {
        "chat_history": chat_history,
        # export_state(): lock-held, namespace-preserving snapshot — the
        # raw _data read raced live mutation and lost scope tags. Falls
        # back to the legacy flat read for scratchpad-like test doubles
        # that only carry `_data`.
        "scratchpad": _scratchpad_snapshot(getattr(agent.context, 'scratchpad', None))
    }
    sandbox_dir = agent.context.sandbox_dir

    # The zip is built in a WORKER THREAD to a SPOOL FILE, not inline in the
    # coroutine to an in-memory buffer. The old version did a synchronous
    # os.walk + deflate of the entire sandbox on the event loop — freezing
    # every other request/SSE stream for its duration — and then held the
    # whole archive in RAM with no ceiling (the load side caps at 500 MB; the
    # save side had none), a self-inflicted OOM vector on a 94%-RAM box. Now
    # the walk/compress runs off-loop, a byte ceiling aborts a runaway
    # archive, and FileResponse streams the spool + deletes it on completion.
    class _WorkspaceTooLarge(Exception):
        pass

    def _build_zip() -> str:
        import tempfile
        fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="ws_save_")
        os.close(fd)
        try:
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
                zip_file.writestr("session.json", json.dumps(session_data, indent=2))
                total = 0
                for root, dirs, files in os.walk(sandbox_dir):
                    if "acquired_skills" in Path(root).parts:
                        continue
                    for file_name in files:
                        file_path = Path(root) / file_name
                        try:
                            total += file_path.stat().st_size
                        except OSError:
                            continue
                        if total > _MAX_WORKSPACE_SAVE_BYTES:
                            raise _WorkspaceTooLarge()
                        arcname = file_path.relative_to(sandbox_dir)
                        zip_file.write(file_path, f"sandbox/{arcname}")
            return tmp_path
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    try:
        tmp_path = await asyncio.to_thread(_build_zip)
    except _WorkspaceTooLarge:
        return JSONResponse(
            {"error": {"message": f"Workspace exceeds the {_MAX_WORKSPACE_SAVE_BYTES // (1024*1024)} MB save limit.",
                       "type": "WorkspaceTooLarge"}},
            status_code=413,
        )
    except Exception:
        eid = _log_internal_error("workspace_save")
        return JSONResponse({"error": f"internal error (error_id={eid})"}, status_code=500)

    filename = f"workspace_{get_utc_timestamp().replace(':', '')}.zip"

    def _cleanup(path=tmp_path):
        try:
            os.unlink(path)
        except OSError:
            pass

    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(_cleanup),
    )

_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB hard ceiling on inbound uploads
# Ceiling on a workspace SAVE archive (uncompressed input bytes). Mirrors the
# load-side 500 MB inflate cap so save and load are symmetric.
_MAX_WORKSPACE_SAVE_BYTES = 500 * 1024 * 1024


def _is_within(base: Path, candidate: Path) -> bool:
    """True iff resolved ``candidate`` is ``base`` or strictly inside it.

    Robust against the prefix-``startswith`` sibling-directory bypass (e.g.
    ``/data/sandbox`` matching ``/data/sandbox_evil``) that a bare
    ``str(p).startswith(str(base))`` check allows.
    """
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except (ValueError, OSError):
        return False


async def _read_capped(upload: UploadFile) -> bytes:
    """Read an UploadFile body with a hard size cap. Without this, a
    multi-GB upload exhausts memory and disk.

    Chunks must be `bytes`/`bytearray`. Test fixtures often pass
    `MagicMock(spec=UploadFile)` whose `read()` returns a MagicMock — that
    MagicMock has `__len__` defaulting to 0, so a naive byte-counting
    loop would never terminate. We require real bytes and additionally
    cap iterations as a belt-and-braces guard.
    """
    total = 0
    chunks: list[bytes] = []
    max_iters = (_MAX_UPLOAD_BYTES // 65536) + 32
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
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds {_MAX_UPLOAD_BYTES // (1024*1024)} MB cap",
            )
        chunks.append(bytes(chunk))
    else:
        raise HTTPException(
            status_code=413,
            detail="Upload iteration cap exceeded — refusing pathological producer.",
        )
    return b"".join(chunks)


@router.post("/api/workspace/load", dependencies=[Security(verify_api_key)])
async def load_workspace(request: Request, file: UploadFile = File(...)):
    """Restores a workspace by cleaning the sandbox and unpacking the provided zip."""
    agent = get_agent(request)
    sandbox_dir = agent.context.sandbox_dir.resolve()
    zip_bytes = await _read_capped(file)
    # Uncompressed-output ceiling — the 100 MB _read_capped bound is on the
    # COMPRESSED input; without this a ~100 MB zip-of-zeroes inflates to tens
    # of GB and OOMs the process / fills the disk (decompression bomb).
    _MAX_UNCOMPRESSED = 500 * 1024 * 1024

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_ref:
            # VALIDATE FIRST, wipe second. Previously the sandbox was cleared
            # before session.json was parsed, so a malformed session.json (or
            # any mid-extraction error) destroyed the workspace with no
            # rollback. Do all fallible parsing/size-checking up front.
            total_uncompressed = sum(max(0, zi.file_size) for zi in zip_ref.infolist())
            if total_uncompressed > _MAX_UNCOMPRESSED:
                raise HTTPException(
                    status_code=413,
                    detail=f"Archive inflates to {total_uncompressed // (1024*1024)} MB; "
                           f"cap is {_MAX_UNCOMPRESSED // (1024*1024)} MB.",
                )
            session_data = None
            for zip_info in zip_ref.infolist():
                if zip_info.filename == "session.json":
                    try:
                        session_data = json.loads(zip_ref.read(zip_info.filename).decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        raise HTTPException(status_code=400, detail="session.json is not valid JSON")
                    break

            # 1. Clear sandbox safely (Preserve permanent skills) — only AFTER
            # the archive validated above.
            for item in sandbox_dir.iterdir():
                if item.name == "acquired_skills":
                    continue
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)

            # 2. Restore session state.
            chat_history = []
            if session_data is not None:
                chat_history = session_data.get("chat_history", [])
                scratchpad_data = session_data.get("scratchpad", {})
                if getattr(agent.context, 'scratchpad', None):
                    agent.context.scratchpad.clear()
                    if isinstance(scratchpad_data, dict):
                        # Namespace-preserving restore (handles the legacy
                        # flat shape too — those land in the GLOBAL scope,
                        # not whatever namespace is currently active).
                        agent.context.scratchpad.restore_state(scratchpad_data)

            # 3. Extract files.
            for zip_info in zip_ref.infolist():
                if not zip_info.filename.startswith("sandbox/"):
                    continue
                rel_path = zip_info.filename[len("sandbox/"):]
                if not rel_path:
                    continue
                extracted_path = (sandbox_dir / rel_path).resolve()
                # Prevent zip-slip traversal (relative_to-based; rejects
                # sibling-dir prefix escapes).
                if not _is_within(sandbox_dir, extracted_path):
                    continue
                if zip_info.is_dir():
                    extracted_path.mkdir(parents=True, exist_ok=True)
                else:
                    extracted_path.parent.mkdir(parents=True, exist_ok=True)
                    extracted_path.write_bytes(zip_ref.read(zip_info.filename))

        return {"status": "success", "chat_history": chat_history}
    except HTTPException:
        raise
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file format")
    except Exception:
        # Don't leak internal exception text to the client.
        _eid = _log_internal_error("workspace/load")
        raise HTTPException(status_code=500, detail=f"Workspace load failed (error id {_eid}).")

@router.get("/api/sessions", dependencies=[Security(verify_api_key)])
async def sessions_list(request: Request, limit: int = 50):
    """Durable server-side conversations, most recently updated first.

    Summaries only (id / title / timestamps / message_count) — a session list
    must stay cheap. Fetch bodies with GET /api/sessions/{id}."""
    agent = get_agent(request)
    from ..core.sessions import get_session_store
    store = get_session_store(agent.context)
    if store is None:
        return JSONResponse({"enabled": False, "sessions": []})
    return JSONResponse({"enabled": True, "sessions": store.list(limit=limit)})


@router.post("/api/sessions", status_code=201,
             dependencies=[Security(verify_api_key)])
async def sessions_create(request: Request):
    """Create an empty session; returns its id. Optional body {"title": ...}.
    (Clients may also just POST /api/chat with a fresh `session_id` — the
    session is created on first append.)"""
    agent = get_agent(request)
    from ..core.sessions import get_session_store
    store = get_session_store(agent.context)
    if store is None:
        raise HTTPException(status_code=503, detail="sessions are not enabled")
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        body = {}
    title = (body or {}).get("title") if isinstance(body, dict) else ""
    sess = store.create(title=str(title or ""))
    if sess is None:
        raise HTTPException(status_code=500, detail="session create failed")
    return JSONResponse(sess.summary(), status_code=201)


@router.get("/api/sessions/{session_id}", dependencies=[Security(verify_api_key)])
async def sessions_get(request: Request, session_id: str):
    """Full conversation (including messages) — used to RESUME a session on a
    different client than the one that started it."""
    agent = get_agent(request)
    from ..core.sessions import get_session_store
    store = get_session_store(agent.context)
    if store is None:
        raise HTTPException(status_code=503, detail="sessions are not enabled")
    sess = store.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")
    return JSONResponse(sess.to_dict())


@router.delete("/api/sessions/{session_id}", dependencies=[Security(verify_api_key)])
async def sessions_delete(request: Request, session_id: str):
    agent = get_agent(request)
    from ..core.sessions import get_session_store
    store = get_session_store(agent.context)
    if store is None:
        raise HTTPException(status_code=503, detail="sessions are not enabled")
    if not store.delete(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return JSONResponse({"deleted": True, "id": session_id})


@router.get("/api/turns", dependencies=[Security(verify_api_key)])
async def turns_list(request: Request):
    """In-flight turns — the one holding the global turn lock plus anything
    queued behind it. Turns are serialized (Semaphore(1)), so this is how you
    see WHY the agent appears unresponsive."""
    agent = get_agent(request)
    from ..core.turns import get_turn_registry
    reg = get_turn_registry(agent)
    turns = reg.list()
    current = reg.current()
    return JSONResponse({
        "turns": [t.to_dict() for t in turns],
        "running": current.req_id if current is not None else None,
        "queued": sum(1 for t in turns if not t.running),
    })


@router.post("/api/turn/cancel", dependencies=[Security(verify_api_key)])
async def turn_cancel(request: Request):
    """Cancel an in-flight turn and RELEASE the global turn lock.

    Body: ``{"request_id": "...", "hard": false}`` — ``request_id`` defaults
    to the currently-running turn. Cooperative by default (the turn stops at
    its next boundary and returns partial work); ``hard=true`` cancels the
    asyncio task outright, which is the guaranteed release for a turn wedged
    inside a long upstream call. A QUEUED turn is always hard-cancelled.

    This is the real thing: the interface's /api/chat/cancel only stops the
    proxy's buffered stream, leaving the agent working and the lock held.
    """
    agent = get_agent(request)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        body = {}
    from ..core.turns import get_turn_registry
    result = get_turn_registry(agent).cancel(
        body.get("request_id") or None,
        hard=bool(body.get("hard", False)),
    )
    return JSONResponse(result, status_code=200 if result.get("cancelled")
                        else 404)


@router.post("/api/feedback", dependencies=[Security(verify_api_key)])
async def feedback(request: Request):
    """Explicit human outcome label for a completed turn (Track-1a of the
    outcome-supply plan, 2026-08-13).

    Body: ``{"request_id": "chatcmpl-<id>"|"<id>", "signal":
    "positive"|"negative", "note": "...", "source": "slack:U…"|"web"}`` —
    ``note`` and ``source`` optional. Writes the corrections sidecar
    (last-write-wins per trajectory), so a thumbs-up/-down from Slack or
    the web UI resolves a turn's ``outcome=unknown`` the moment the human
    who read the reply judges it. 404 when the request_id has no recorded
    trajectory (still being written → the client may retry once).
    """
    agent = get_agent(request)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        body = {}
    rid = str(body.get("request_id") or "").strip()
    signal = str(body.get("signal") or "").strip().lower()
    from ..core.feedback import apply_human_label, VALID_SIGNALS
    # One body shape on every path ({ok, error, code}) — a client reading
    # `error` must never get None just because the status was a 400.
    if not rid:
        return JSONResponse({"ok": False, "error": "request_id is required",
                             "code": "bad_request"}, status_code=400)
    if signal not in VALID_SIGNALS:
        return JSONResponse(
            {"ok": False, "code": "bad_request",
             "error": f"signal must be one of {list(VALID_SIGNALS)}"},
            status_code=400)
    # The day-partition scan is file I/O — keep it off the event loop.
    result = await asyncio.to_thread(
        apply_human_label, agent, rid, signal,
        str(body.get("note") or "")[:500],
        str(body.get("source") or "")[:80],
    )
    if result.get("ok"):
        # Lesson-outcome flush HERE, on the event loop — the helper spawns
        # loop-bound background work, so calling it from the worker thread
        # popped the stash and then lost the write (R1 review). Called on
        # EVERY ok including idempotent repeats (R2): the flush is
        # sign-aware and no-op-safe, and repeating it heals the case where
        # the first attempt's post-write path failed after the label
        # landed. A human label also revokes any queued machine correction
        # banner for the turn — the verdict-then-label ordering is the
        # common one, and the banner would otherwise apologize on the next
        # reply for an answer the human just endorsed.
        try:
            flush = getattr(agent, "_flush_stashed_lesson_outcome", None)
            if callable(flush):
                flush(result.get("trajectory_id"),
                      result.get("outcome") == "passed")
            drop = getattr(agent, "_drop_pending_corrections_for", None)
            if callable(drop):
                drop(result.get("trajectory_id"))
        except Exception:  # noqa: BLE001 — credit must not fail the label
            pass
        return JSONResponse(result)
    # Status from the machine-readable code, not the error prose — rewording
    # a message must never reroute a client's retry logic.
    status = {"bad_request": 400, "not_found": 404}.get(
        str(result.get("code") or ""), 503)
    return JSONResponse(result, status_code=status)


#: Hard ceiling on one drain request. A bench item is a full isolated
#: solve (minutes of solver tokens), so a typo'd count must not queue
#: days of work on the operator's only inference slot.
_BENCH_DRAIN_MAX = 200


@router.post("/api/bench/drain", dependencies=[Security(verify_api_key)])
async def bench_drain(request: Request):
    """Arm an operator-requested bench drain (§4BO, 2026-08-15).

    Body: ``{"count": N, "banks": ["mbpp", …]}``. ``count: 0`` cancels an
    armed drain. Returns the resulting budget and an honest wall-clock
    estimate — 200 items is 5–13 h of the box's only inference slot.

    This endpoint ARMS; it never runs an item. The biological watchdog
    spends the budget from its own tick, which is what keeps it the single
    caller of ``banks.pick_next_item`` — that function's cursor is an
    unsynchronized read-modify-write, and "run bench from a second
    process" is the race its docstring already names.

    Arming REPLACES any previous budget rather than adding to it. The
    remainder is readable from ``GET /api/health``
    (``bench_drain_remaining``), because the budget is in-memory by design
    and would otherwise be visible only in the live log stream.

    ⚠ Cancelling does NOT abort the item already solving; it stops the
    ones after it. The running item ends on its own or hits the per-item
    timeout (``GhostAgent._BENCH_ITEM_TIMEOUT``).
    """
    agent = get_agent(request)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        body = {}
    # `count` must be PRESENT and an honest integer. A missing key used to
    # fall through to 0 = cancel, so a typo'd arm request and "stop the
    # running drain" were the same 200 response (R3 review m2). Bools are
    # rejected explicitly because `int(True) == 1` would arm one item.
    if "count" not in body:
        return JSONResponse(
            {"ok": False, "code": "bad_request",
             "error": "count is required (use 0 to cancel an armed drain)"},
            status_code=400)
    raw_count = body.get("count")
    if isinstance(raw_count, bool) or not isinstance(raw_count, (int, float)):
        return JSONResponse({"ok": False, "code": "bad_request",
                             "error": "count must be a number"},
                            status_code=400)
    try:
        count = int(raw_count)
    except (TypeError, ValueError, OverflowError):
        # OverflowError is the `1e999` → float('inf') case, which is not a
        # subclass of either of the others and used to 500.
        return JSONResponse({"ok": False, "code": "bad_request",
                             "error": "count must be a finite integer"},
                            status_code=400)
    # CANCEL FIRST, before every other check (R2 review m4). Validation
    # used to run ahead of this, so `{"count": 0, "banks": "typo"}` 400'd
    # and the drain kept running — an operator watching the box thrash
    # must never be refused a stop because of an unrelated field.
    if count == 0:
        prior = int(getattr(agent, "_bench_drain_remaining", 0) or 0)
        agent._bench_drain_remaining = 0
        agent._bench_drain_banks = None      # else health shows a filter
                                             # for a drain that is over
        return JSONResponse({"ok": True, "remaining": 0, "cancelled": prior,
                             "note": "the item already solving is NOT "
                                     "aborted; it runs to completion or "
                                     "hits the per-item timeout"})
    if count < 0 or count > _BENCH_DRAIN_MAX:
        return JSONResponse(
            {"ok": False, "code": "bad_request",
             "error": f"count must be between 0 and {_BENCH_DRAIN_MAX}"},
            status_code=400)
    from ..eval import banks as _banks
    # A filter that silently means "everything" is the defect class
    # `pick_next_item` refuses one layer down — the endpoint must not
    # reintroduce it above that guard. A bare string (what a hand-typed
    # curl produces), a dict, or a non-string element is a 400, not an
    # accidental all-banks drain (R3 review M3).
    raw_banks = body.get("banks")
    want_banks = None
    if raw_banks is not None:
        if not isinstance(raw_banks, list) or not raw_banks:
            return JSONResponse(
                {"ok": False, "code": "bad_request",
                 "error": "banks must be a non-empty list of bank names "
                          "(omit it to drain every bank)"},
                status_code=400)
        if len(raw_banks) > 32:
            return JSONResponse(
                {"ok": False, "code": "bad_request",
                 "error": "banks accepts at most 32 names"},
                status_code=400)
        if not all(isinstance(b, str) for b in raw_banks):
            return JSONResponse(
                {"ok": False, "code": "bad_request",
                 "error": "banks entries must be strings"}, status_code=400)
        want_banks = [b[:64] for b in raw_banks]


    # ── Refuse to arm something that cannot run, at ARM time. ──────────
    # The whole point of the endpoint is supply the operator can schedule;
    # "armed but permanently inert" is the failure this project keeps
    # finding, and it is cheap to rule out here instead of leaving a
    # WARNING in a log nobody is watching.
    if getattr(getattr(agent, "context", None), "args", None) is not None \
            and getattr(agent.context.args, "no_bench", False) is True:
        return JSONResponse(
            {"ok": False, "code": "disabled",
             "error": "the agent runs with --no-bench; the drain would "
                      "never execute. Restart without it to use this."},
            status_code=409)
    # The CONSUMER must be alive. The watchdog is the only thing that
    # spends the budget, so arming while it is dead returns a cheerful 200
    # for work that can never run — the precise outcome this block exists
    # to rule out (R3 review m2). /api/health already reports this signal.
    _bio = getattr(request.app.state, "biological_task", None)
    if _bio is not None and _bio.done():
        return JSONResponse(
            {"ok": False, "code": "no_consumer",
             "error": "the biological watchdog task is not running, so "
                      "nothing would spend this budget. Restart the agent."},
            status_code=409)
    present = await asyncio.to_thread(_banks.list_banks)
    if not present:
        return JSONResponse(
            {"ok": False, "code": "no_banks",
             "error": "no bench banks on disk — import them with "
                      "scripts/import_bench_banks.py first"},
            status_code=409)
    if want_banks:
        missing = [b for b in want_banks if b not in present]
        if missing:
            return JSONResponse(
                {"ok": False, "code": "no_banks",
                 "error": f"unknown bank(s) {sorted(missing)[:8]}; "
                          f"on disk: {sorted(present)}"},
                status_code=409)

    agent._bench_drain_banks = want_banks
    agent._bench_drain_remaining = count
    # An honest wall-clock estimate at ARM time. 200 items is 5–13 h of
    # the box's only inference slot; the operator should see that before
    # walking away, not discover it from the log stream.
    _lo = round(count * 92 / 3600.0, 1)     # ~32 s solve + 60 s tick gap
    _hi = round(count * 240 / 3600.0, 1)    # 3 attempts on a hard item
    _safe_log(
        "Bench Drain",
        f"operator armed {count} item(s), "
        f"banks={sorted(want_banks) if want_banks else 'all'} — the "
        f"watchdog will run them back-to-back while the box is quiet "
        f"(~{_lo}-{_hi}h of the inference slot)",
        icon=Icons.BRAIN_AIM,
    )
    from ..core.agent import GhostAgent as _GA
    # timeout + the 60 s tick gap + the ~60 s teardown tail `wait_for`
    # waits out after cancelling (docker remove + rmtree). Omitting the
    # tail understated a 200-item worst case by ~6%.
    _worst = round(count * (_GA._BENCH_ITEM_TIMEOUT + 120) / 3600.0, 1)
    _note = ("arming REPLACES any previous budget; GET /api/health "
             "reports the remainder. estimated_hours is the MEASURED "
             f"range; worst case if every item wedges to the per-item "
             f"timeout is ~{_worst}h.")
    # R3 MAJOR-2: a live bench-scoped experiment arm accrues from the same
    # population a drain floods. `tts_bon`'s pre-registered rule reasons
    # in "bench nights" and gates on "no confound annotation" — a 200-item
    # drain delivers ~50 nights of accrual in one, under a different
    # machine regime, and NO analysis path reads the regime tag yet. The
    # gate depends on a human noticing, so say it to the human who is
    # about to arm it rather than leaving it to be discovered later.
    _arms = []
    try:
        from ..core.experiments import (load_registry as _lr,
                                        SCOPE_BENCH as _sb)
        _arms = list(_lr().names_for_scope(_sb))
    except Exception as exc:  # noqa: BLE001 — advisory only
        logger.debug("bench-scoped arm lookup skipped: %s", exc)
    if _arms:
        _note += (f" ⚠ live bench-scoped arm(s) {sorted(_arms)} accrue "
                  f"from this population: {count} drained items enroll "
                  f"like organic ones and nothing yet stratifies on the "
                  f"regime. Record the drain window, or pause the arm.")
    if want_banks:
        # R2 MAJOR: the cursor is shared, so draining one bank advances
        # only that bank and the lowest-cursor rotation will not return
        # to it until the others catch up — measured at ~28 days of
        # organic cadence after a 200-item single-bank drain. Say it
        # here, where the operator is choosing.
        _note += (" ⚠ a bank-scoped drain advances only that bank's "
                  "cursor, so the organic rotation will not revisit it "
                  "until the other banks catch up.")
    return JSONResponse({"ok": True, "remaining": count,
                         "banks": sorted(want_banks or present),
                         "estimated_hours": [_lo, _hi],
                         "worst_case_hours": _worst,
                         "bench_scoped_arms": sorted(_arms),
                         "note": _note})


@router.get("/api/notifications/pending", dependencies=[Security(verify_api_key)])
async def notifications_pending(request: Request, consumer: str = "default",
                                limit: int = 50):
    """Undelivered notify-severity autonomous-activity records for an
    external deliverer (e.g. the Slack bot), watermarked per ``consumer``.

    Contract: poll this, deliver the records, then POST the returned
    ``watermark`` to ``/api/notifications/ack`` — records are only
    considered delivered once acked, so a deliverer crash re-serves them.
    A consumer that has never acked is BASELINED to end-of-ledger (no
    historical replay on first contact — mirrors the digest's silent
    first-run baseline)."""
    agent = get_agent(request)
    from ..core.autonomous_activity import (
        get_activity_log, load_consumer_offset, SEVERITY_NOTIFY,
    )
    log = get_activity_log(agent.context)
    if log is None:
        return JSONResponse({"enabled": False, "records": [], "watermark": 0})
    from pathlib import Path as _Path
    consumers_path = (_Path(str(agent.context.memory_dir)).parent
                      / "notify_consumers.json")
    offset = load_consumer_offset(consumers_path, consumer)
    if offset is None:
        # First contact: baseline silently. The caller acks this watermark
        # and subsequent polls return only NEW records.
        return JSONResponse({"enabled": True, "records": [],
                             "watermark": log.current_offset(),
                             "baseline": True})
    try:
        limit = max(1, min(int(limit), 200))
    except (TypeError, ValueError):
        limit = 50
    # Scan PAST non-notify noise (bounded). ``read_since``'s own limit
    # bounds SCANNED LINES per chunk, not returned records — a single
    # call from a stale watermark into an info-heavy ledger (dream /
    # self-play spam) can scan its whole window without meeting one
    # notify record and return []. Combined with a client that only
    # acks non-empty responses, that WEDGED the slack consumer at its
    # Jul-11 watermark: every 30s poll re-scanned the same 20 info
    # lines forever and nothing was ever delivered again (found
    # 2026-07-13). Loop until we have ``limit`` notify records or EOF,
    # capped at 50×200 = 10k lines per poll so one call can't scan an
    # unbounded ledger. ``limit`` is a SOFT bound (whole chunks are
    # kept): truncating mid-chunk would advance the watermark past
    # records we never returned, silently dropping them.
    records = []
    cursor = offset
    for _ in range(50):
        chunk, new_cursor = log.read_since(cursor, limit=200,
                                           severity=SEVERITY_NOTIFY)
        records.extend(chunk)
        if new_cursor <= cursor:  # EOF / no progress
            # A new_cursor BELOW the request cursor is read_since's
            # shrunk-ledger re-baseline (the file was truncated/replaced
            # and this consumer's offset points past the new EOF). Adopt
            # it so the RETURNED watermark heals the consumer: returning
            # the stale offset instead used to rely on the bot blindly
            # re-acking + the ack route's clamp to converge — a loop the
            # bot's idle-identity ack skip (2026-08-01) removed. With the
            # stale value echoed forever, a truncated ledger meant no
            # notification was ever served again for this consumer.
            if new_cursor < cursor:
                cursor = new_cursor
            break
        cursor = new_cursor
        if len(records) >= limit:
            break
    return JSONResponse({
        "enabled": True,
        "records": [r.to_dict() for r in records],
        "watermark": cursor,
    })


@router.post("/api/notifications/ack", dependencies=[Security(verify_api_key)])
async def notifications_ack(request: Request):
    """Advance a consumer's delivery watermark (see /api/notifications/pending)."""
    agent = get_agent(request)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
        consumer = str(body.get("consumer") or "default")
        watermark = int(body.get("watermark"))
    except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
        return JSONResponse(
            {"error": {"message": f"invalid ack body: {e}",
                       "type": "InvalidRequestShape"}},
            status_code=400,
        )
    from ..core.autonomous_activity import (
        save_consumer_offset, get_activity_log)
    from pathlib import Path as _Path
    # Clamp to [0, EOF]. A watermark past EOF (client bug, or a stale large
    # value replayed after a ledger truncation) permanently wedges the
    # consumer: every /pending then reads nothing and re-returns the huge
    # watermark, so no future record is ever served. Clamp so the consumer
    # can only ever be caught up to real EOF, never beyond it.
    _log = get_activity_log(agent.context)
    if _log is not None:
        watermark = max(0, min(watermark, _log.current_offset()))
    else:
        watermark = max(0, watermark)
    consumers_path = (_Path(str(agent.context.memory_dir)).parent
                      / "notify_consumers.json")
    save_consumer_offset(consumers_path, consumer, watermark)
    return JSONResponse({"ok": True, "consumer": consumer,
                         "watermark": watermark})


@router.post("/api/memory/correct", dependencies=[Security(verify_api_key)])
async def memory_correct(request: Request):
    """Surgically rewrite ONE vector-memory fragment's text.

    The safe path for fixing a poisoned auto-memory: Chroma's persist dir
    must only ever be touched by the process that owns it (a second
    PersistentClient risks HNSW corruption), so the correction runs
    in-process via VectorMemory.correct_fragment. Body:
    {"match": <exact text or unique substring>, "replacement": <new text>}.
    """
    agent = get_agent(request)
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, 400)
    match = str(body.get("match") or "")
    replacement = str(body.get("replacement") or "")
    memory = getattr(getattr(agent, "context", None), "memory_system", None)
    if memory is None or not hasattr(memory, "correct_fragment"):
        return JSONResponse({"error": "memory system unavailable"}, 503)
    ok, detail = memory.correct_fragment(match, replacement)
    if not ok:
        return JSONResponse({"ok": False, "error": detail}, 409)
    return {"ok": True, **detail}


@router.post("/api/memory/delete", dependencies=[Security(verify_api_key)])
async def memory_delete(request: Request):
    """Surgically DELETE one vector-memory fragment.

    Companion to /api/memory/correct for a fragment that is WHOLLY false
    (nothing true to rewrite it into). Runs in-process for the same reason
    (a second PersistentClient against the live Chroma dir risks HNSW
    corruption). Body: {"match": <exact text or unique substring>}.
    """
    agent = get_agent(request)
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, 400)
    match = str(body.get("match") or "")
    memory = getattr(getattr(agent, "context", None), "memory_system", None)
    if memory is None or not hasattr(memory, "delete_fragment"):
        return JSONResponse({"error": "memory system unavailable"}, 503)
    ok, detail = memory.delete_fragment(match)
    if not ok:
        return JSONResponse({"ok": False, "error": detail}, 409)
    return {"ok": True, **detail}


@router.post("/api/memory/delete_skill_twin", dependencies=[Security(verify_api_key)])
async def memory_delete_skill_twin(request: Request):
    """Delete the vector TWINS of named skill lessons (type=skill + trigger).

    Cleanup companion for a JSON-playbook prune: the JSON is canonical, but a
    lesson removed from it leaves its embedded twin behind. Runs in-process
    (a second PersistentClient against the live Chroma dir risks HNSW
    corruption). Body: {"triggers": [<trigger str>, ...]} (or {"trigger": …}).
    """
    agent = get_agent(request)
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, 400)
    triggers = body.get("triggers")
    if not triggers and body.get("trigger"):
        triggers = [body["trigger"]]
    if not isinstance(triggers, list) or not triggers:
        return JSONResponse({"error": "body needs 'triggers' (list) or 'trigger'"}, 400)
    memory = getattr(getattr(agent, "context", None), "memory_system", None)
    if memory is None or not hasattr(memory, "delete_skill_twins"):
        return JSONResponse({"error": "memory system unavailable"}, 503)
    removed, detail = memory.delete_skill_twins(triggers)
    return {"ok": True, "requested": len(triggers), "removed": removed, **detail}


@router.post("/api/upload", dependencies=[Security(verify_api_key)])
async def upload_file(request: Request, file: UploadFile = File(...)):
    agent = get_agent(request)
    # When a project is active, land the upload in its scoped dir so it joins
    # the working set the model sees and the (scoped) file_system can read it.
    # A client may pass ?project_id=<id> to scope RACE-FREE (the process-global
    # current_project_id can be pointed at another conversation's project by a
    # concurrent switch/reconcile); absent that, the global is the fallback.
    from ..tools.file_system import project_scoped_sandbox
    _qp = getattr(request, "query_params", None)
    _explicit_pid = _qp.get("project_id") if _qp else None
    sandbox_dir = project_scoped_sandbox(agent.context, explicit_project_id=_explicit_pid)[0]
    # Reject path traversal AND the absent-filename case explicitly.
    if not file.filename or ".." in file.filename or file.filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = (sandbox_dir / file.filename).resolve()
    # ⚠ Containment is checked against the TRUE sandbox root, not against
    # `sandbox_dir`. `sandbox_dir` is DERIVED from the client-supplied
    # `?project_id=`, so checking the file against it validated an
    # already-escaped base and always passed — an absolute project id
    # replaced the root outright and `mkdir(parents=True)` then created the
    # directory. `project_scoped_sandbox` now refuses an unsafe id; this is
    # the second lock, because the caller of a scoping helper must never
    # have to trust that the scope it got back is inside the sandbox.
    _root = getattr(agent.context, "sandbox_dir", None)
    if _root is not None and not _is_within(Path(_root), file_path):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not _is_within(sandbox_dir, file_path):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Read with size cap before writing — same DoS guard as workspace/load.
    body = await _read_capped(file)

    def _write_file():
        with open(file_path, "wb") as buffer:
            buffer.write(body)
    await asyncio.to_thread(_write_file)
    return {"status": "success", "filename": file.filename}

@router.get("/api/download/{filename:path}", dependencies=[Security(verify_api_key)])
async def download_file(request: Request, filename: str):
    agent = get_agent(request)
    sandbox_dir = agent.context.sandbox_dir
    # Reject traversal explicitly (parity with /api/upload) then containment-check.
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=404, detail="File not found")
    file_path = (sandbox_dir / filename).resolve()
    # Fallback: when a project is active, artefact tools (image_generation,
    # execute-saved plots) write into <sandbox>/projects/<id>/, but the model
    # often emits a BARE link (`/api/download/plot.png`). If the bare path
    # isn't at the root, retry under the active project's scoped dir so the
    # link still resolves. Containment is re-checked below.
    if not file_path.exists() and "/" not in filename:
        # Prefer an explicit ?project_id= (race-free) over the process-global,
        # which a concurrent conversation's switch could have moved.
        _qp = getattr(request, "query_params", None)
        pid = (_qp.get("project_id") if _qp else None) or getattr(agent.context, "current_project_id", None)
        if isinstance(pid, str) and pid.strip():
            cand = (sandbox_dir / "projects" / pid.strip().lower() / filename).resolve()
            if _is_within(sandbox_dir, cand) and cand.exists():
                file_path = cand
    if not _is_within(sandbox_dir, file_path) or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), filename=file_path.name)

@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"], dependencies=[Security(verify_api_key)])
async def catch_all(request: Request, path: str):
    agent = get_agent(request)
    url = f"/{path}"
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    # Sniff whether this is a streaming/SSE request so we can set the right
    # response media type AND disable buffering on intermediate proxies.
    # Without `X-Accel-Buffering: no`, nginx (and any sane reverse proxy)
    # will buffer SSE chunks — the client sees the stream arrive in one
    # batch at the end instead of in real time.
    body_says_stream = False
    try:
        # Best-effort peek at the JSON body. The body has not been read yet
        # because we forward `request.stream()` straight to the upstream
        # client below; once we call `await request.body()` here the stream
        # is exhausted, so we re-feed it to httpx via `content=...`.
        if request.method in ("POST", "PUT") and "json" in request.headers.get("content-type", "").lower():
            raw = await request.body()
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict) and parsed.get("stream") is True:
                        body_says_stream = True
                except (json.JSONDecodeError, ValueError):
                    pass
            forward_content = raw
        else:
            forward_content = request.stream()
    except Exception:
        forward_content = request.stream()

    try:
        req = agent.context.llm_client.http_client.build_request(
            request.method, url, headers=headers, content=forward_content
        )
        r = await agent.context.llm_client.http_client.send(req, stream=True)

        upstream_ct = r.headers.get("content-type", "") or ""
        is_event_stream = "text/event-stream" in upstream_ct.lower() or body_says_stream

        async def stream_generator():
            try:
                async for chunk in r.aiter_bytes():
                    yield chunk
            finally:
                await r.aclose()

        response_headers = {}
        if is_event_stream:
            # Tell nginx / any reverse proxy NOT to buffer this response.
            response_headers["X-Accel-Buffering"] = "no"
            response_headers["Cache-Control"] = "no-cache"

        return StreamingResponse(
            stream_generator(),
            status_code=r.status_code,
            media_type=("text/event-stream" if is_event_stream else upstream_ct or None),
            headers=response_headers,
        )
    except Exception as e:
        pretty_log("Proxy Failed", f"{request.method} /{path}: {type(e).__name__}: {e}",
                   icon=Icons.FAIL, level="ERROR")
        return JSONResponse({"error": f"Proxy Error: {e}"}, 502)