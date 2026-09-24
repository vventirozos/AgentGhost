import asyncio
import concurrent.futures
import datetime
import errno
import functools
import json
import secrets
import shutil
import stat as _stat
import sys
import threading
import time
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
from ..utils.helpers import env_positive
import logging
from ..utils.logging import (Icons, pretty_log, ORIGIN_PROBE, PROBE_REQUEST_PREFIX, is_probe_request_id,
                             ORIGIN_SLACK, SLACK_REQUEST_PREFIX, is_slack_request_id,
                             client_deadline_context)

logger = logging.getLogger("GhostAgent")

# ── Store calls off the event loop (§4GI, 2026-09-13) ─────────────────────────
# The operator "surgery" routes (memory correct/delete, skill-twin delete,
# lesson quarantine) called the vector store and the playbook DIRECTLY from
# an `async def` handler. `VectorMemory.correct_fragment` takes the store's
# `threading.RLock` — the same lock the biological tick's consolidation
# holds from a worker thread for seconds — and `quarantine_lesson` takes an
# `fcntl.flock(LOCK_EX)` with no timeout. Either one froze the event loop:
# every SSE stream, `/api/health` and the reaper stalled for the duration,
# and past `GHOST_STREAM_IDLE_TIMEOUT` the user's turn was aborted as an
# "upstream stall". ONE helper now runs every such call in a worker thread
# under a bounded wait; `tests/test_4gi_api_routes.py` enumerates the
# handlers from the AST and fails if a store call bypasses it.
# §4GI: `env_positive`, not a bare float() — a typo in the env var would
# otherwise raise at MODULE IMPORT and the agent would not boot
# (tests/test_env_timeout_constants.py caught the first version).
_STORE_CALL_TIMEOUT_S = max(1.0, env_positive("GHOST_STORE_CALL_TIMEOUT", 30.0))
#: Context attributes that are persistent STORES (a lock, a file, a DB):
#: a route may only call them through `_store_call`.
STORE_CONTEXT_ATTRS = frozenset({
    "memory_system", "skill_memory", "trajectory_collector", "profile_memory",
    "graph_memory", "self_model", "calibration_tracker",
})


class StoreCallTimeout(Exception):
    """A store call did not return inside `_STORE_CALL_TIMEOUT_S`."""


# ⚠ STORE CALLS GET THEIR OWN POOL (§4GJ round 4, 2026-09-13). The first
# version used `asyncio.to_thread`, which runs on the loop's DEFAULT
# executor — the one every other `run_in_executor(None, ...)` in this
# process shares (359 call sites across 38 modules at last count). A store
# call that times out is deliberately NOT cancelled (see below), so each
# timeout ABANDONS a worker of that shared pool for as long as the store
# stays wedged, and nothing bounds how many can pile up. Measured: 20
# wedged store calls blocked an unrelated `asyncio.to_thread` for 3 s, and
# 18 of the abandoned mutations landed after their 504 — i.e. 18 operator
# retries of one stuck route stop every tool call in the agent. A private,
# bounded pool moves the blast radius back inside this module: when it is
# full the next store call queues and answers 504 on its own budget, and the
# sandbox, the file tools and the embedder keep their threads.
def _store_worker_count() -> int:
    """How wide the store pool is. Env-tunable, like its sibling timeout.

    ⚠ IT WAS A HARD-CODED 8 (§4GK round 5) sitting next to an
    env-configurable `_STORE_CALL_TIMEOUT_S`. One `notifications_pending`
    poll issues up to 50 sequential store calls, so the pool width is
    exactly the number an operator watching store 504s wants to move — and
    it was the one number in this pair they could not move without editing
    the source. `env_positive`, not `int(os.getenv(...))`: a typo must not
    raise at module import and stop the agent from booting.
    """
    return max(1, int(env_positive("GHOST_STORE_WORKERS", 8)))


_STORE_EXECUTOR_WORKERS = _store_worker_count()
_STORE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=_STORE_EXECUTOR_WORKERS, thread_name_prefix="ghost-store")
#: Submitted store calls that have not finished. Only used to tell a RUNNING
#: call from a QUEUED one at interpreter exit — see `_shutdown_store_executor`.
_STORE_INFLIGHT = set()


def _shutdown_store_executor(executor=None) -> None:
    """Drop QUEUED store calls at interpreter exit, and never wait.

    ⚠ NOTHING EVER SHUT THIS POOL DOWN (§4GK round 5). Its threads are
    non-daemon — `ThreadPoolExecutor` has not made daemon threads since 3.9
    — so `concurrent.futures.thread`'s own atexit hook joins them, and that
    hook DRAINS the queue first: every store mutation still queued behind a
    wedged one runs during interpreter shutdown, against half-finalised
    modules. Cancelling the queue is the half we control. A call already
    RUNNING in a wedged thread still blocks exit and nothing in-process can
    change that, so name it in the log rather than leave the operator
    staring at a process that will not die.

    ⚠ AND MOVING IT OFF `atexit` GAVE UP `atexit`'s ISOLATION (§4GK round 7).
    `atexit._run_exitfuncs` wraps every callback in its own try/except and
    carries on; `threading._shutdown` does NOT — it walks
    `_threading_atexits` with nothing around the call, so ONE exception out
    of this function skips every hook registered before it, including
    `concurrent.futures`' `_python_exit`, and then the whole non-daemon join
    loop underneath it. The body is not exception-free: it runs at
    finalisation, where `logger.warning` reaches handlers whose streams may
    already be closed (`ValueError: I/O operation on closed file` is the
    ordinary one) — a store-pool warning could take the interpreter's thread
    shutdown with it. So the body carries the isolation the channel no
    longer provides, and the last-resort report goes to the raw stream
    because that is what is left when logging is gone.
    """
    try:
        ex = executor if executor is not None else _STORE_EXECUTOR
        running = [f for f in list(_STORE_INFLIGHT) if f.running()]
        ex.shutdown(wait=False, cancel_futures=True)
        if running:
            logger.warning(
                "%d store call(s) still running at exit — the interpreter "
                "cannot exit until they return (the store is wedged)",
                len(running))
    except BaseException as exc:          # noqa: BLE001 - see the docstring
        try:
            sys.stderr.write(
                f"store-pool shutdown hook failed ({type(exc).__name__}: "
                f"{exc}); continuing interpreter shutdown\n")
        except Exception:                 # pragma: no cover - stderr is gone
            pass


import atexit as _atexit


def _register_store_shutdown(hook=_shutdown_store_executor) -> str:
    """Register `hook` so it runs BEFORE the executor's own shutdown hook.

    ⚠ `atexit.register` MADE THE WHOLE THING DEAD CODE (§4GK round 6).
    `concurrent.futures.thread` registers `_python_exit` through
    `threading._register_atexit`, and `threading._shutdown()` runs those
    hooks — and then JOINS every non-daemon thread — inside
    `wait_for_thread_shutdown()`, which precedes `_PyAtExit_Call`. So the
    executor's hook drained the queue and joined the workers FIRST and
    `_shutdown_store_executor` arrived afterwards with nothing left to
    cancel. Measured on this interpreter against this module: one wedged
    store call, three queued behind it, interpreter exits — all three
    queued mutations RAN during shutdown against half-finalised modules,
    the exact behaviour the docstring above says it prevents. The
    "store is wedged" warning could not fire either (`running` is 0 by
    then), which is why nothing ever looked wrong.

    `threading._register_atexit` runs its hooks in REVERSE registration
    order, before the join loop. `concurrent.futures.thread` is already
    imported by the `ThreadPoolExecutor(...)` above, so its hook is
    registered first and ours runs ahead of it. The name is private, so
    the plain `atexit` path stays as a fallback: a hook that runs late is
    what we had, and it is better than an import error at boot.
    """
    register = getattr(threading, "_register_atexit", None)
    if register is not None:
        register(hook)
        return "threading"
    _atexit.register(hook)           # pragma: no cover - 3.9+ always has it
    return "atexit"


#: Which channel the hook above actually got — read by the pin, and by an
#: operator wondering why queued store calls landed after a shutdown.
_STORE_SHUTDOWN_CHANNEL = _register_store_shutdown()


async def _store_call(fn, *args, timeout: float = None, **kwargs):
    """Run a blocking store method in a worker thread with a bounded wait.

    A call that has STARTED is not cancelled on timeout (a half-applied
    Chroma write is worse than a late one): the handler stops waiting and
    answers 504 while the thread finishes on its own. A call still QUEUED is
    the opposite outcome and gets the opposite treatment — it is cancelled,
    so it cannot land minutes after its own 504. It is a thread of
    `_STORE_EXECUTOR`, never the process-wide default one.

    ⚠ THE DOCSTRING USED TO CLAIM THE FIRST FOR BOTH (§4GK round 5), which
    held only while a worker was free. Once all `_STORE_EXECUTOR_WORKERS`
    are busy, `asyncio.wait_for` cancels a still-queued
    `concurrent.futures.Future` SUCCESSFULLY (measured) and the mutation
    never runs at all. One 504 covered two opposite facts — "your edit
    probably landed, late" and "your edit did not happen" — with nothing on
    the wire to tell them apart. The exception now says which it was.
    """
    budget = float(timeout if timeout is not None else _STORE_CALL_TIMEOUT_S)
    cf = _STORE_EXECUTOR.submit(functools.partial(fn, *args, **kwargs))
    _STORE_INFLIGHT.add(cf)
    cf.add_done_callback(_STORE_INFLIGHT.discard)
    try:
        return await asyncio.wait_for(asyncio.wrap_future(cf), timeout=budget)
    except asyncio.TimeoutError as e:
        # `cancel()` succeeds only for a call that never started — which is
        # precisely the case the old docstring got wrong.
        never_started = cf.cancel()
        raise StoreCallTimeout(
            f"{getattr(fn, '__name__', 'store call')} did not return within "
            f"{budget:.0f}s "
            + (f"(all {_STORE_EXECUTOR_WORKERS} store workers were busy: this "
               f"call never started and has been cancelled — it did NOT take "
               f"effect)"
               if never_started else
               "(the store is busy — a consolidation may hold its lock; the "
               "call is still running and may still take effect)")
        ) from e


def _store_timeout_response(exc: Exception) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, 504)


# ── Proxied inference goes through the main-slot bookkeeping (§4GI) ───────────
# `api_generate` and the catch-all proxy posted straight through
# `http_client`. A chat turn increments `foreground_tasks` (the background
# scheduler's "a user is active" signal), takes `_main_node_lock` for a
# non-streaming main call, and counts a stream in `_inflight_by_url` — the
# stream watchdog's stall verdict reads `_own_inflight(base)` to tell "the
# node is busy with OUR other request" from "the upstream stalled". A proxied
# completion was invisible to all three: a user turn queued behind it was
# aborted as an "Upstream Stream Stall (sole in-flight request)". The two
# routes now do the same bookkeeping, in the same order, through one helper.
import contextlib as _contextlib


@_contextlib.asynccontextmanager
async def _main_node_request(llm, *, hold_lock: bool):
    """Book a request against the MAIN node the way a chat turn does.

    ``hold_lock=True`` mirrors the non-streaming main path (serialised on
    `_main_node_lock`); ``False`` mirrors the streaming path (in-flight
    counted, no lock). Every step is optional on the client (test doubles,
    older clients) — a missing attribute is skipped, never raised on.
    """
    base = str(getattr(llm, "upstream_url", "") or "")
    # isinstance, not truthiness: a MagicMock client (the API tests) answers
    # every getattr with a MagicMock, which is neither a lock nor a counter
    fg_lock = getattr(llm, "_foreground_lock", None)
    count_fg = (isinstance(fg_lock, asyncio.Lock)
                and isinstance(getattr(llm, "foreground_tasks", None), int))
    # ⚠ THE RELEASE MUST NOT AWAIT (round 3, 2026-09-13). The first version
    # mutated the counter under `async with fg_lock` at BOTH ends. The
    # decrement lives in a `finally` that runs on client disconnect, and
    # `Lock.acquire()` there is an await point: with the lock contended (a
    # background call polls it about once a second) and a second cancellation
    # delivered — uvicorn's disconnect followed by shutdown — the acquire
    # raises `CancelledError`, the decrement never runs, and
    # `foreground_tasks` stays at 1 for the life of the process. `core/agent`
    # hard-gates the biological tick on that counter, so all idle work stops.
    # Reproduced: 2 cancels + a contended lock leaks; 1 cancel does not.
    #
    # `+=` / `-=` on an int has NO await point, so on one event loop it is
    # already atomic with respect to every other coroutine — the lock buys
    # nothing for a single-statement mutation and costs the release its
    # cancellation-safety. `llm.py`'s own compound reader holds the lock
    # across a two-field read with no await between them, which a synchronous
    # mutation here cannot interleave with. (This is why the in-flight
    # release, a plain `stack.callback`, survived the same cancellation while
    # the counter did not.) `llm.py` keeps its awaited form: those decrements
    # are not on a path a client can cancel.
    if count_fg:
        llm.foreground_tasks = int(llm.foreground_tasks) + 1
    try:
        async with _contextlib.AsyncExitStack() as stack:
            main_lock = getattr(llm, "_main_node_lock", None)
            if hold_lock and isinstance(main_lock, asyncio.Lock):
                await stack.enter_async_context(main_lock)
            # ⚠ COUNT ONLY ON THE PATH THAT DOES NOT HOLD THE LOCK (§4GJ
            # round 4). The first version did both: it took `_main_node_lock`
            # AND `_inflight_inc(base)`. `LLMClient._own_inflight` already
            # adds one for a HELD main lock — that is the documented contract
            # ("a held lock is worth exactly one more in-flight request
            # against the main URL and needs no second counter"), and llm.py's
            # own locked paths deliberately do not increment. So a proxied
            # `/api/generate` counted itself twice: one real stream plus one
            # generate read as 3 against a truth of 2, which on a 2-slot node
            # is exactly the "Stream Stall (Self-Queued)" condition — the
            # watchdog aborted the USER's live stream to make room for a
            # request that did not exist.
            inc, dec = getattr(llm, "_inflight_inc", None), getattr(llm, "_inflight_dec", None)
            if not hold_lock and callable(inc) and callable(dec):
                inc(base)
                stack.callback(dec, base)
            yield
    finally:
        if count_fg:
            llm.foreground_tasks = max(0, int(llm.foreground_tasks) - 1)


#: Client headers that must never reach the upstream: every CREDENTIAL the
#: client sent us, and the full HOP-BY-HOP / body-framing set (RFC 9110 §7.6.1)
#: that belongs to THIS connection and that httpx recomputes for its own.
#:
#: ⚠ FOUR NAMES WAS NOT THE CLASS (§4GJ round 4, 2026-09-13). The first
#: version stripped `x-ghost-key`, `authorization`, `host`, `content-length`
#: and forwarded everything else verbatim — confirmed by driving the real
#: app: `proxy-authorization`, `cookie`, `connection`, `te` and `upgrade`
#: all reached the upstream. `authorization` was stripped because it is a
#: credential, and `proxy-authorization` and `cookie` are the SAME
#: credential class: with a remote `--upstream-url` they leave the machine
#: to a third party. `transfer-encoding` is the framing half: the JSON peek
#: re-feeds the body to httpx as BYTES, so httpx sets its own
#: `Content-Length` — a chunked JSON POST then carried both framings, which
#: 502'd here (confirmed) and hands a laxer upstream a CL.TE desync
#: primitive. Strip the class, not the names that happened to hurt.
_PROXY_STRIP_HEADERS = frozenset({
    # credentials the client presented to THIS API
    "x-ghost-key", "authorization", "proxy-authorization", "cookie",
    # body framing httpx recomputes for its own request
    "host", "content-length", "transfer-encoding",
    # hop-by-hop: they describe this connection, not the proxied one
    "connection", "keep-alive", "proxy-authenticate", "proxy-connection",
    "te", "trailer", "trailers", "upgrade",
})


def _forwardable_headers(headers) -> dict:
    return {k: v for k, v in dict(headers).items() if str(k).lower() not in _PROXY_STRIP_HEADERS}

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


def _youtube_route_health(context) -> dict:
    try:
        from ..memory.youtube_canary import health_view
        return health_view(context)
    except Exception:  # noqa: BLE001 — health must never raise
        return {"state": "unknown"}


@router.post("/api/youtube-canary/run", dependencies=[Security(verify_api_key)])
async def youtube_canary_run(request: Request):
    """Operator trigger (§4KH): start one canary probe now, off-loop, ignoring
    the cadence, the boot delay AND the foreground gate (an explicit request).
    Used after `bin/update-youtube-stack.sh` to get the RECOVERED notice
    without waiting a day. Returns whether a probe STARTED plus the view as it
    was BEFORE this probe; poll `/api/health` → `youtube_route` for the
    result (a probe takes 5–60 s)."""
    agent = get_agent(request)
    try:
        from ..memory.youtube_canary import health_view, maybe_run
        started = bool(maybe_run(agent, force=True))
        return JSONResponse({"started": started, "youtube_route_before": health_view(agent.context),
                             "poll": "/api/health → youtube_route"})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"started": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)


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

    # Functional mood (2026-09-11, for the web face's slow baseline tint):
    # label + provenance + when. getattr-chained so a mock or a boot
    # without a self-model can never 500 the health endpoint.
    mood = None
    try:
        _sm = getattr(context, "self_model", None)
        _st = getattr(_sm, "state", None)
        _m = getattr(_st, "mood", None)
        _m = _m() if callable(_m) else _m
        if _m is not None and getattr(_m, "label", None):
            mood = {"label": str(_m.label), "source": str(getattr(_m, "source", "") or ""),
                    "set_at": str(getattr(_m, "set_at", "") or "")}
    except Exception:  # noqa: BLE001 — health must never raise
        mood = None

    return JSONResponse({
        "status": "ok",
        "mood": mood,
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
        # §4KH: the YouTube route canary's last verdict (ok / walled /
        # helper_down / tor_down / error / not_yet_run) — read from its
        # state file, so a dead tick shows as a stale `last_run`.
        "youtube_route": _youtube_route_health(context),
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
    # Bound BEFORE the try: the except handler below reports with it, and a
    # name bound only inside the try makes that handler raise NameError
    # instead (the `_wrote` class the §4GJ enumeration catches — committed
    # here inside §4GJ's own round-3 fix, and caught by its own gate).
    from ..core.llm import _MAIN_FALLBACK_TIMEOUT_S as _main_budget
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
        _llm = agent.context.llm_client
        # a non-streaming MAIN call: serialised on the node lock like a turn's
        #
        # ⚠ BOUNDED (round 3, 2026-09-13). This POST passed no `timeout=`, so
        # it inherited httpx's 1200s default WHILE HOLDING the process-wide
        # `_main_node_lock` — one wedged upstream parks every chat turn and
        # every embedding for twenty minutes, and Starlette does not cancel
        # the handler when the client disconnects. `core/llm` bounds every
        # other locked main POST for exactly this reason; reuse ITS budget
        # (`GHOST_MAIN_FALLBACK_TIMEOUT`, 300s) rather than inventing a
        # second number that can drift from it.
        async with _main_node_request(_llm, hold_lock=True):
            resp = await asyncio.wait_for(
                _llm.http_client.post("/v1/chat/completions", json=chat_payload),
                timeout=_main_budget,
            )
        resp.raise_for_status()
        llm_resp = resp.json()
        content = llm_resp["choices"][0]["message"]["content"]
    except (asyncio.TimeoutError, httpx.TimeoutException):
        # The lock is already released (the `async with` unwound) — say so
        # with the status that means it, not a 500 the client will retry.
        #
        # The CONNECTION is gone by the time this line runs: `wait_for`
        # cancels the POST and waits for httpx's teardown, and a raw-socket
        # upstream sees EOF in the same millisecond as the 504 (measured
        # against a real httpx client, §4GJ round 4 — the round-4 report's
        # "0 cancelled" is the upstream ignoring the disconnect, not a
        # connection left open here). What we CANNOT do from this side is
        # make a non-streaming llama-server drop the generation it has
        # already started, so say out loud that the slot may still be warm:
        # the operator reading the stream is the only one who can tell a
        # queue from a stall.
        pretty_log("Proxy Timeout",
                   f"/api/generate: upstream did not answer within "
                   f"{_main_budget:.0f}s — connection closed, main slot "
                   f"released (a non-streaming upstream may still be "
                   f"generating this answer into the void)",
                   icon=Icons.WARN, level="WARNING")
        return JSONResponse(
            {"error": f"upstream did not answer within {_main_budget:.0f}s"}, 504)
    except httpx.HTTPStatusError as e:
        # ⚠ AN UPSTREAM FAULT IS NOT OUR FAULT (§4GJ round 4). The
        # `raise_for_status()` above sat under the generic `except
        # Exception`, so a 503 from llama-server came back as a 500 with an
        # error id and a logged traceback — telling the client "the agent
        # broke" and burying the one fact that matters in the log. The
        # catch-all proxy has always answered 502 for exactly this
        # condition; these two routes front the same upstream and must not
        # disagree about whose failure it is.
        _status = getattr(getattr(e, "response", None), "status_code", "?")
        pretty_log("Proxy Upstream Error",
                   f"/api/generate: upstream answered {_status}",
                   icon=Icons.FAIL, level="ERROR")
        return JSONResponse(
            {"error": f"upstream error (status {_status})"}, 502)
    except httpx.RequestError as e:
        # ⚠ AND AN UPSTREAM THAT NEVER ANSWERED IS ALSO NOT OUR FAULT (§4GK
        # round 5). Round 4 closed only `HTTPStatusError` — the case where
        # llama-server is up enough to reply. The COMMON case is the one it
        # left open: the node is down, restarting, or drops the connection
        # mid-response, i.e. `ConnectError` / `RemoteProtocolError` /
        # `ReadError`. Measured against a real socket: a refused connect and
        # a mid-response RST both came back from `/api/generate` as a 500
        # with an error id and a logged traceback, while `catch_all` — the
        # OTHER route onto the same upstream — answered 502 for both. Round
        # 4's own rationale is the standard: these two must not disagree
        # about whose failure it is. `TimeoutException` is a `RequestError`
        # too, so its 504 handler stays ABOVE this one.
        pretty_log("Proxy Upstream Error",
                   f"/api/generate: upstream unreachable — "
                   f"{type(e).__name__}: {e}",
                   icon=Icons.FAIL, level="ERROR")
        return JSONResponse(
            {"error": f"upstream unreachable ({type(e).__name__})"}, 502)
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
    # §4FB (2026-09-06): `X-Ghost-Origin: probe` marks a DIAGNOSTIC turn —
    # an operator/Claude probe exercising the live path. It runs exactly
    # like a user turn but must never teach (no lesson, reflection,
    # calibration or credit). The mark travels as a request-id PREFIX so
    # every population reader agrees — see `core/agent.turn_origin`.
    if (request.headers.get("X-Ghost-Origin") or "").strip().lower() == ORIGIN_PROBE:
        if not is_probe_request_id(request_id):
            request_id = PROBE_REQUEST_PREFIX + (
                str(request_id or "").strip() or uuid.uuid4().hex[:8])
        # (§4FF's `X-Ghost-Prompt-Variant` probe header was retired here on
        # 2026-09-21, §4JG — no prompt variant exists to select.)
    # §4KD: `X-Ghost-Origin: slack` — a Slack client that did not mint the
    # prefix itself. The bot does (so its feedback correlation keeps the id);
    # this is the fallback for any other Slack-side caller.
    elif (request.headers.get("X-Ghost-Origin") or "").strip().lower() == ORIGIN_SLACK:
        if not is_slack_request_id(request_id):
            request_id = SLACK_REQUEST_PREFIX + (
                str(request_id or "").strip() or uuid.uuid4().hex[:8])

    # §4JP: the client's own timeout, when it says so (the web interface
    # sends GHOST_CHAT_TIMEOUT). The turn loop reserves the last minutes of
    # it for a state report instead of being cut mid-tool. Absent or
    # malformed → 0.0 = no deadline (Slack, CLI, probes without the header).
    try:
        _cdl = float(request.headers.get("X-Ghost-Client-Timeout") or 0.0)
    except (TypeError, ValueError):
        _cdl = 0.0
    client_deadline_context.set(_cdl if _cdl > 0 else 0.0)

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
            # ⚠ THE 504 IS THE PROMISE, NOT THE EXCEPTION (§4GJ round 4).
            # This was the one `_store_call` on the chat hot path with no
            # handler — and it sits ABOVE the `if stream:` split, so a
            # wedged session store took BOTH chat paths down with a bare
            # text/plain `500 Internal Server Error` from Starlette's
            # handler: no `error.message`, no `error.type`, nothing the web
            # UI or the Slack bot can render, and a status that says "the
            # agent is broken" about a store that is merely busy.
            try:
                _existing = await _store_call(_sess_store.get, str(_session_id))
            except StoreCallTimeout as e:
                return JSONResponse(
                    {"error": {"message": str(e), "type": "StoreCallTimeout"}},
                    status_code=504)
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

    async def _persist_session(assistant_text: str) -> None:
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
            # `append_turn` writes and fsyncs — on the loop it stalled every
            # other request for the duration of a disk sync (§4GJ round 3).
            await _store_call(_sess_store.append_turn, str(_session_id),
                              _new_msgs, str(assistant_text or ""))
        except Exception:  # noqa: BLE001 — a persist failure never breaks the reply
            _log_internal_error("session append")

    if stream:
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }

        # ⚠ THE MARK IS RELEASED AROUND THE SEND, NOT INSIDE THE GENERATOR
        # (§4GJ round 4, 2026-09-13). This is round 3's CRITICAL again, on
        # the path that carries the actual user traffic: the proxy path was
        # given `_BookedStreamingResponse` and this one was left with a
        # `finally` inside the body generator. A `finally` in a generator
        # only runs when something advances or closes it — and on a
        # mid-stream client disconnect Starlette CANCELS `stream_response`
        # while it is suspended in `await send(...)`, so nothing is ever
        # thrown into the generator. It is finalised by the CYCLIC garbage
        # collector, whenever that happens to run. Measured on the real
        # ASGI app over raw sockets: 20 mid-stream disconnects left
        # `foreground_requests` at 11, still 11 after five seconds, 0 only
        # after an explicit `gc.collect()` — and with `gc.disable()` it
        # never came back at all. `core/llm` reads
        # `foreground_requests > 0` as "a user is active" and parks EVERY
        # background LLM call for up to ten minutes on that reading, so a
        # handful of users closing their browser tabs silently switches the
        # whole background stack off.
        #
        # The release is idempotent and lives in both places on purpose:
        # the response's `finally` covers the send (including a client that
        # was gone before the first chunk, where the generator never runs at
        # all), and the generator's own `finally` covers a consumer that
        # iterates the body without going through the ASGI call.
        _fg_released = False

        async def _release_foreground():
            nonlocal _fg_released
            if not _fg_released:
                _fg_released = True
                _mark_foreground(agent, -1)

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
                    await _persist_session("".join(_acc))
                else:
                    await _persist_session(content)
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
                await _release_foreground()

        # Mark a user request active for its WHOLE lifecycle (agent loop +
        # final-answer streaming) so background LLM work parks instead of
        # stealing the inference slot between this request's tool calls.
        # See LLMClient.foreground_requests. Taken HERE, not in the
        # generator, so that the release above has something to pair with
        # even when the generator is never entered.
        _mark_foreground(agent, +1)
        return _BookedStreamingResponse(
            stream_generator(), media_type="text/event-stream", headers=headers,
            release=_release_foreground)
    
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

    await _persist_session(content)

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

    #: Members this build could not read, as {path, reason}. An unreadable
    #: file used to be a bare `continue` (§4GJ round 4): nothing logged,
    #: nothing on the wire, and a 200 carrying an archive the operator
    #: believes is their whole workspace. A restore from it then DELETES the
    #: sandbox and writes back the subset that happened to be readable — so
    #: the silent omission is what turns a permissions glitch into data loss.
    #: The list rides back on the response header and, so it survives the
    #: download, as a member of the archive itself.
    omitted = []

    def _member_info(arcname: str, st, *, is_dir: bool = False) -> zipfile.ZipInfo:
        """A ZipInfo carrying the file's OWN mode and mtime.

        ⚠ `writestr(str, ...)` mints its own ZipInfo, stamps it with the
        archive time and hardcodes `external_attr = 0o600 << 16` — the
        earlier `zip_file.write(path, arcname)` read both off the file.
        Confirmed after the §4GJ R3 rewrite: `run.sh` came back `0o600`
        instead of `0o755` (nothing in the restored sandbox is executable
        any more) and every mtime was rewritten to the moment of the save,
        so every incremental tool that compares timestamps sees the whole
        tree as just-changed. The fix is to hand `writestr` a ZipInfo we
        built ourselves.
        """
        if is_dir and not arcname.endswith("/"):
            arcname += "/"
        if st is not None:
            try:
                dt = time.localtime(st.st_mtime)[:6]
            except (OSError, OverflowError, ValueError):
                # An mtime the platform cannot even break down — and WHICH
                # WAY it cannot matters (§4GK round 6). Round 5 clamped every
                # unrepresentable stamp to the MAXIMUM date, so a far-PAST one
                # was archived as 2107-12-31: measured, `time.localtime(-1e18)`
                # raises OSError [Errno 84] and the file came back stamped 127
                # years in the future, on the same line that exists to clamp
                # "BOTH ends". The sign of the stamp is the direction:
                # anything below the epoch is past (the small negatives that
                # DO break down land in 1969 and clamp up to 1980 below),
                # anything above it that fails is future
                # (`time.localtime(1e18)` → OSError EINVAL).
                dt = (_ZIP_DOS_MIN_DATE_TIME if st.st_mtime < 0
                      else _ZIP_DOS_MAX_DATE_TIME)
            mode = _stat.S_IMODE(st.st_mode)
        else:
            dt = time.localtime()[:6]
            mode = 0o755 if is_dir else 0o644
        # ⚠ THE DOS STAMP HAS A CEILING TOO (§4GK round 5). Round 4 clamped
        # an mtime UP to the 1980 zip epoch and stopped there.
        # `ZipInfo.FileHeader` packs the date as `(year-1980) << 9 | month << 5
        # | day` into a USHORT, so a year ≥ 2108 raises `struct.error` — a
        # plain `Exception`, which the member loop's `except (OSError,
        # ValueError)` below does not catch. It escaped `_build_zip` entirely
        # and the route's generic handler answered an opaque 500: ONE file
        # with a far-future stamp made EVERY workspace save 500 until someone
        # found it. The sandbox is model-writable (`touch -d 2200-01-01`,
        # `os.utime`, an unpacked archive carrying bogus stamps, clock skew),
        # so this is reachable on purpose as well as by accident — confirmed
        # with `os.utime(f, (7258118400, 7258118400))` on a file AND on a
        # directory, both 500. The round-4 rewrite is what introduced it:
        # `writestr(str, data)` stamped the archive time and could never hit
        # the ceiling; the ZipInfo that reads the file's OWN mtime can.
        if dt[0] < 1980:                       # the zip epoch; older stamps clamp
            dt = _ZIP_DOS_MIN_DATE_TIME
        elif dt[0] > _ZIP_DOS_MAX_DATE_TIME[0]:
            dt = _ZIP_DOS_MAX_DATE_TIME        # …and later ones clamp down
        zi = zipfile.ZipInfo(arcname, dt)
        zi.compress_type = zipfile.ZIP_STORED if is_dir else zipfile.ZIP_DEFLATED
        zi.external_attr = (mode & 0xFFFF) << 16
        if is_dir:
            zi.external_attr |= 0x10           # FILE_ATTRIBUTE_DIRECTORY
        elif st is not None:
            zi.file_size = st.st_size          # so the zip64 decision is right
        return zi

    def _build_zip() -> str:
        import tempfile
        fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="ws_save_")
        os.close(fd)
        try:
            # ⚠ allowZip64 STAYS TRUE (§4GJ round 4). The 4th positional of
            # `ZipFile` is `allowZip64`, whose own default is True; passing
            # False capped the archive at 65,535 members, far below the
            # 500 MB byte ceiling — one `node_modules` under the sandbox
            # clears that on its own. `LargeZipFile` is then raised at
            # CLOSE, after the whole tree has been compressed, and this
            # route answered an opaque 500 where the byte path answers an
            # honest 413.
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED,
                                 allowZip64=True) as zip_file:
                zip_file.writestr("session.json", json.dumps(session_data, indent=2))
                total = 0
                # §4GJ R3: never archive through a symlink the model planted
                # under the mount. `os.walk` refuses linked DIRECTORIES but
                # lists linked FILES, and `zip_file.write` follows them — so
                # a link to a host secret was archived as a regular member
                # and `restore` wrote those bytes back into the sandbox.
                # `walk_nofollow` lists regular files only and hands back a
                # directory fd, so the read is atomic with the listing (a
                # per-entry `is_symlink()` pre-check leaves a swap window);
                # the restore half refuses a member landing on a link.
                from ..tools.file_system import (read_bytes_nofollow_fd,
                                                 walk_nofollow)
                for dirpath, file_names, _dir_fd in walk_nofollow(sandbox_dir):
                    if "acquired_skills" in Path(dirpath).parts:
                        continue
                    # The DIRECTORY itself is a member (§4GJ round 4). The
                    # file-only loop dropped every directory that held no
                    # files: `empty_dir`, and the interior nodes of a deep
                    # tree, simply vanished from the archive — a restore
                    # recreated leaf parents implicitly and nothing else, so
                    # a build tree came back missing the empty directories
                    # its tooling expects (confirmed: `empty_dir`, `nested`
                    # and `nested/deep` were all absent).
                    _rel_dir = Path(dirpath).relative_to(sandbox_dir)
                    if str(_rel_dir) != ".":
                        # ⚠ THE WHOLE DIRECTORY MEMBER IS INSIDE THE TRY
                        # (§4GK round 5). Only the `fstat` was guarded; the
                        # `_member_info` + `writestr` that follow it were
                        # bare, so the two halves of ONE walk disagreed about
                        # what an unarchivable member is: an unarchivable
                        # FILE became an `omitted` record and a 200, while an
                        # unarchivable DIRECTORY aborted the build and the
                        # route answered 500. That asymmetry is how the
                        # far-future mtime above took the whole save down
                        # from a directory as well as from a file.
                        try:
                            try:
                                _dst = os.fstat(_dir_fd)
                            except OSError:
                                _dst = None
                            zip_file.writestr(
                                _member_info(f"sandbox/{_rel_dir}", _dst, is_dir=True), b"")
                        except (OSError, ValueError) as exc:
                            omitted.append({
                                "path": f"{_rel_dir}/",
                                "reason": f"{type(exc).__name__}: {exc}",
                                "partial": False,
                            })
                    for file_name in file_names:
                        file_path = Path(dirpath) / file_name
                        arcname = f"sandbox/{file_path.relative_to(sandbox_dir)}"
                        try:
                            # `follow_symlinks=False`: the listing already
                            # excluded links, and this must not become the
                            # one call that follows one.
                            st = os.stat(file_name, dir_fd=_dir_fd,
                                         follow_symlinks=False)
                        except OSError:
                            st = None
                        # bytes already written into the archive for this
                        # member, so a read that dies MID-COPY is reported as
                        # a truncated member rather than as a missing one —
                        # the two need different action from whoever restores
                        # it, and "omitted" would be a small lie about a
                        # member that is physically in the file.
                        _progress = [0]
                        try:
                            data = None
                            if st is None or st.st_size <= _ZIP_INLINE_MEMBER_BYTES:
                                data = read_bytes_nofollow_fd(
                                    file_name, dir_fd=_dir_fd,
                                    max_bytes=_ZIP_INLINE_MEMBER_BYTES + 1)
                                if len(data) > _ZIP_INLINE_MEMBER_BYTES:
                                    # it grew between the stat and the read:
                                    # stream it rather than ship a truncated
                                    # member
                                    data = None
                            if data is None:
                                total += _stream_member(zip_file, arcname, st,
                                                        file_name, _dir_fd, total,
                                                        _progress)
                            else:
                                total += len(data)
                                if total > _MAX_WORKSPACE_SAVE_BYTES:
                                    raise _WorkspaceTooLarge()
                                zip_file.writestr(_member_info(arcname, st), data)
                        except (OSError, ValueError) as exc:
                            omitted.append({
                                "path": str(file_path.relative_to(sandbox_dir)),
                                "reason": f"{type(exc).__name__}: {exc}",
                                "partial": _progress[0] > 0,
                            })
                            total += _progress[0]
                            continue
                if omitted:
                    # Inside the archive too: the header below is gone the
                    # moment the browser has saved the file, and the person
                    # who restores it months later is the one who needs to
                    # know what is not in here.
                    zip_file.writestr("omitted.json", json.dumps(omitted, indent=2))
            return tmp_path
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _stream_member(zip_file, arcname: str, st, name, dir_fd, total: int,
                       progress: list) -> int:
        """Compress one member straight from its descriptor, in chunks.

        ⚠ PER-MEMBER RAM WAS UNBOUNDED (§4GJ round 4). The read helper takes
        `max_bytes=<the 500 MB archive cap>`, so a single large member was
        materialised whole and then handed to the deflater whole: measured
        215 MB → 369 MB RSS while archiving ONE 150 MB file, on a box that
        already runs at 94% memory. The byte ceiling bounded the ARCHIVE,
        never the member. Small files still go through the nofollow read
        helper (one syscall, atomic with the listing, and no per-member zip
        handle for the thousands of tiny files that make up a sandbox);
        anything large is copied through a fixed buffer instead.

        The symlink guarantee is unchanged: the open is `O_NOFOLLOW`
        relative to the walk's directory fd — the same open the read helper
        does — and the fstat re-checks that this is a regular file, so the
        bytes can only come from the file the walk listed.
        """
        # ⚠ `O_NONBLOCK` IS PART OF THAT PARITY (§4GK round 5). The read
        # helper opens `O_RDONLY | O_NOFOLLOW | O_NONBLOCK`; this open
        # dropped the third flag, and it is the only one that stops a
        # non-regular final component from blocking BEFORE the `S_ISREG`
        # check below can reject it. Measured: on a FIFO the helper opened
        # and rejected immediately while this open was still parked after
        # 1.5 s. It is reachable as a TOCTOU the model owns on its own
        # mount — listed and stat'd as a regular file over the inline
        # ceiling, then swapped for a FIFO or a device node before this
        # line — and the cost is total: `_build_zip` runs inside
        # `asyncio.to_thread`, which has NO timeout, so the request never
        # completes and a default-executor worker is gone for the life of
        # the process.
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(name, flags, dir_fd=dir_fd) if dir_fd is not None \
            else os.open(str(name), flags)
        try:
            fst = os.fstat(fd)
            if not _stat.S_ISREG(fst.st_mode):
                raise ValueError(f"{name}: not a regular file")
            written = 0
            with zip_file.open(_member_info(arcname, st if st is not None else fst),
                               "w") as dst:
                while True:
                    chunk = os.read(fd, _ZIP_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if total + written > _MAX_WORKSPACE_SAVE_BYTES:
                        raise _WorkspaceTooLarge()
                    dst.write(chunk)
                    progress[0] = written
            return written
        finally:
            os.close(fd)

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

    # An incomplete archive says so, on the response and on the stream. A 200
    # with a silently short archive is the one answer this route must never
    # give (§4GJ round 4).
    _resp_headers = {}
    if omitted:
        _resp_headers["X-Ghost-Archive-Omitted"] = str(len(omitted))
        pretty_log("Workspace Save Incomplete",
                   f"{len(omitted)} file(s) could not be read and are NOT in "
                   f"the archive (see omitted.json inside it): "
                   f"{', '.join(o['path'] for o in omitted[:5])}"
                   + (" …" if len(omitted) > 5 else ""),
                   icon=Icons.WARN, level="WARNING")

    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=filename,
        headers=_resp_headers or None,
        background=BackgroundTask(_cleanup),
    )

_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB hard ceiling on inbound uploads
# Ceiling on a workspace SAVE archive (uncompressed input bytes). Mirrors the
# load-side 500 MB inflate cap so save and load are symmetric.
_MAX_WORKSPACE_SAVE_BYTES = 500 * 1024 * 1024
# A member at or below this is read whole (one atomic nofollow read); anything
# larger is streamed through `_ZIP_COPY_CHUNK_BYTES` at a time, so the archiver's
# peak RAM is set by THIS number and not by the largest file in the sandbox.
_ZIP_INLINE_MEMBER_BYTES = 4 * 1024 * 1024
_ZIP_COPY_CHUNK_BYTES = 1024 * 1024
#: The last instant a DOS date field can hold: 7 bits of year above 1980, and
#: seconds in 2-second units. Anything later cannot be written into a zip
#: header at all — see `_member_info`.
_ZIP_DOS_MAX_DATE_TIME = (2107, 12, 31, 23, 59, 58)
#: ...and the first. Both ends are named because the fallback for an mtime
#: that will not break down has to pick ONE of them, and picking the wrong
#: one stamps a 1930s file as 2107 (§4GK round 6).
_ZIP_DOS_MIN_DATE_TIME = (1980, 1, 1, 0, 0, 0)


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


# ── the restore side of the mode/mtime fix (§4GK round 5) ────────────────────
# Round 4 taught the ARCHIVE to carry each member's own mode and mtime and
# stopped there: `write_bytes()` + `mkdir()` applied neither, so the defect it
# reported — "nothing in the restored sandbox is executable any more", "every
# mtime rewritten to the moment of the save" — stayed true for everyone who
# restores THROUGH THE AGENT. Measured on a real save→load round trip: the zip
# carried `run.sh` at 0o755 (round 4's own pin passes) and the restored file
# was 0o644 with an mtime of now. Only an external `unzip` ever saw the fix.
#
# Both helpers are fd-based at the call site: the tree is model-writable and
# the write sits behind a check-then-write, so the mode and the mtime must
# land on the descriptor we opened `O_NOFOLLOW`, never on a path that can be
# swapped for a link between the check and the chmod.

def _archived_mode(zip_info) -> int | None:
    """The member's own permission bits, or None if the archive has none.

    A DOS-only archive (anything not written by a unix zipper) carries no
    mode in the high half of `external_attr` — restoring the 0 it decodes to
    would make every member unreadable, so "no mode" means "leave the
    default". setuid/setgid/sticky are dropped: the archive is model-supplied
    input and nothing in a restored sandbox needs them.
    """
    raw = (zip_info.external_attr >> 16) & 0xFFFF
    if raw == 0:
        return None
    return raw & 0o777


def _archived_mtime(zip_info) -> float | None:
    """The member's stamp as an epoch float, or None if it will not convert.

    `date_time` is a DOS stamp clamped into 1980..2107 by the save side, but
    this archive can come from anywhere — an unconvertible tuple means "leave
    the mtime alone", not "abort the restore".
    """
    try:
        return time.mktime(tuple(zip_info.date_time) + (0, 0, -1))
    except (ValueError, OverflowError, OSError):
        return None


def _restore_file_member(path: Path, data: bytes, zip_info) -> None:
    """Write one archived file back with its own mode and mtime.

    `O_NOFOLLOW` is the enforcement behind the caller's symlink refusal, not
    a replacement for it: the caller still refuses a destination (or parent)
    that is already a link, and this open refuses one planted in the window
    between that check and the write.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        mode = _archived_mode(zip_info)
        if mode is not None:
            os.fchmod(fd, mode)
        mtime = _archived_mtime(zip_info)
        if mtime is not None and os.utime in os.supports_fd:
            os.utime(fd, (mtime, mtime))
    finally:
        os.close(fd)


def _restore_dir_metadata(path: Path, zip_info) -> None:
    """Apply an archived DIRECTORY's mode and mtime — after its contents.

    Deferred on purpose: a directory archived 0o500 (or 0o555) applied at
    `mkdir` time makes every member inside it unwritable and the restore
    dies half-way through a sandbox it has already wiped. Extracting the
    whole archive first and stamping the directories afterwards also gets
    the mtimes right, since writing a member into a directory bumps it.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return                       # gone, or swapped for a link: not ours
    try:
        mode = _archived_mode(zip_info)
        if mode is not None:
            os.fchmod(fd, mode)
        mtime = _archived_mtime(zip_info)
        if mtime is not None and os.utime in os.supports_fd:
            os.utime(fd, (mtime, mtime))
    except OSError:
        pass                         # a mode we cannot set is not a failed restore
    finally:
        os.close(fd)


# ── the restored mode must not TRAP the workspace (§4GK round 6) ─────────────
# Round 5's deferred-directory comment reasons about exactly one hazard — a
# 0o555 directory applied before its contents land — and solves it WITHIN one
# restore. It does not survive one call later. Measured on a plain save→load
# round trip with no crafted archive, on a sandbox the model can chmod:
#
#   save Monday   (dist/ 0o555, holding bundle.js)
#   save Tuesday  (the same, plus dist/vendor.js)
#   restore Monday  -> 200, and the live sandbox now has a 0o555 dist/
#   restore Tuesday -> HTTP 500, PermissionError .../live/dist/vendor.js
#
# Three faults compound into that 500, and the third is why it is permanent:
#   1. the wipe called `shutil.rmtree(item, ignore_errors=True)`, which
#      CANNOT unlink inside a non-writable directory — so the frozen tree
#      survived the "clean" wipe and its stale contents were carried into the
#      "restored" workspace, silently;
#   2. `os.open(O_CREAT)` for a NEW member inside it fails EACCES, and
#      `_restore_file_member`'s caller re-raised every errno but ELOOP — into
#      the route's generic handler, i.e. an opaque 500 AFTER the sandbox had
#      already been wiped;
#   3. and nothing repaired the directory, so every later archive carrying a
#      new member under it 500s at the same point, for ever.
#
# The helpers below are the repair. They are fd-based and `O_NOFOLLOW` for the
# same reason the two above are: this tree is model-writable and every one of
# these paths was checked a moment earlier.

def _member_reason(exc: OSError) -> str:
    """Why one member could not be restored, in a form that can go on the
    wire: the errno NAME and the OS's own text, never the exception's repr.

    ⚠ `str(exc)` PUT ABSOLUTE HOST PATHS IN THE RESPONSE (§4GK round 7).
    `PermissionError(13, 'Permission denied', '/Users/.../sandboxes/<id>/
    live/dist/vendor.js')` renders every one of those characters into
    `unrestored[].reason`, four lines above the handler whose comment reads
    "Don't leak internal exception text to the client" — the sandbox's real
    location on the host, its id, and the server's user, handed to whoever
    uploaded the archive. The member's own RELATIVE path is already in
    `unrestored[].path`, which is the part the operator needs.
    """
    name = errno.errorcode.get(exc.errno, str(exc.errno))
    return f"{name}: {exc.strerror or type(exc).__name__}"


def _add_owner_access(path: Path) -> bool:
    """Give the OWNER rwx on one directory. True if it now has it.

    Never follows a link (a symlinked component is not ours to chmod) and
    never touches anything but the owner bits: this is "can the server still
    manage its own sandbox", not a permissions rewrite.

    ⚠ IT COULD NOT REPAIR WHAT IT COULD NOT OPEN (§4GK round 7). The whole
    helper hangs off `os.open(O_RDONLY)`, so it repaired exactly the modes
    that still grant the owner READ: 0o555 and 0o444 heal, and 0o333, 0o111,
    0o000 — a `chmod 111 dist` from inside the sandbox, or an archive
    carrying that mode on a directory — fail at the `open` and return False
    before a single bit is changed. Measured end to end: a top-level `dist/`
    at 0o111 survives the wipe (`rmtree` cannot unlink inside it) and its
    stale contents are carried into the "restored" workspace, which is fault
    1 of the round-6 comment above, unfixed for a whole class of modes.

    The fallback repairs through the PARENT's descriptor: `lstat` first, so a
    symlink or a plain file is never touched, and `follow_symlinks=False` so
    that even a link swapped in during the window is chmod'ed as a link
    rather than through to whatever it points at. That is the same property
    `O_NOFOLLOW` gives the fast path, kept by a different mechanism.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            return _chmod_owner_via_parent(path)
        return False
    try:
        mode = _stat.S_IMODE(os.fstat(fd).st_mode)
        if mode & 0o700 != 0o700:
            os.fchmod(fd, mode | 0o700)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def _chmod_owner_via_parent(path: Path) -> bool:
    """`_add_owner_access` for a directory we cannot open for reading.

    Everything here is about NOT following a link: `lstat` decides it is a
    real directory (a symlink lstats as a link and is refused outright), the
    chmod goes through the parent's own descriptor by NAME, and
    `follow_symlinks=False` means a link planted in the window is chmod'ed
    as a link instead of reaching its target. Where the platform cannot do
    that, the repair is declined rather than performed unsafely — an
    unrepaired directory is reported by name; a chmod through a link is a
    hole in the sandbox.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if not _stat.S_ISDIR(st.st_mode):
        return False                     # a link or a file: not ours to chmod
    if os.chmod not in os.supports_dir_fd \
            or os.chmod not in os.supports_follow_symlinks:
        return False                     # pragma: no cover - POSIX has both
    pflags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        pfd = os.open(path.parent, pflags)
    except OSError:
        return False
    try:
        os.chmod(path.name, _stat.S_IMODE(st.st_mode) | 0o700,
                 dir_fd=pfd, follow_symlinks=False)
        return True
    except OSError:
        return False
    finally:
        os.close(pfd)


def _unfreeze_tree(root: Path) -> None:
    """Make every real directory under `root` removable, top-down.

    `os.walk` needs r+x to list a directory and `rmtree` needs w to unlink
    inside it; a 0o555 directory has the first and not the second, which is
    the whole of fault 1 above. Top-down matters: each directory is
    unfrozen from its parent's listing BEFORE the walk descends into it.
    `followlinks=False` plus the `O_NOFOLLOW` in `_add_owner_access` means a
    symlinked subdirectory is listed and skipped, never chmod'ed through.
    """
    _add_owner_access(root)
    for dirpath, dirnames, _files in os.walk(root, topdown=True, followlinks=False):
        for name in dirnames:
            _add_owner_access(Path(dirpath) / name)


def _unfreeze_chain(path: Path, stop: Path) -> bool:
    """Unfreeze every existing directory from `stop` down to `path`.

    Used when a member cannot be written: the frozen directory may be the
    parent, or any ancestor between it and the sandbox root (a member under
    `acquired_skills/` reaches this with the whole chain skipped by the
    wipe). Walks DOWN from `stop` so it can never chmod outside the sandbox,
    and returns True if at least one directory was reachable — the caller
    retries once on that.

    ⚠ IT SKIPPED THE ROOT, WHICH IS THE ONLY ANCESTOR A TOP-LEVEL MEMBER HAS
    (§4GK round 7). `rel.parts` is EMPTY for a member sitting directly in the
    sandbox — the overwhelmingly common case — so `touched` stayed False, the
    caller's one retry never ran, and the frozen directory the helper exists
    to repair was the one it could not reach. `chmod 555 /workspace` from
    inside the container was enough: every top-level member came back
    unwritable, with a repair that had never been attempted. `stop` is the
    sandbox root itself, so unfreezing it is still inside the mount.
    """
    try:
        root = stop.resolve()
        rel = path.resolve().relative_to(root)
    except (ValueError, OSError):
        return False
    cur, touched = root, _add_owner_access(root)
    for part in rel.parts:
        cur = cur / part
        touched = _add_owner_access(cur) or touched
    return touched


def _mkdir_unfreezing(path: Path, stop: Path) -> str | None:
    """`mkdir -p`, repairing a frozen ancestor once. The reason it could
    not, or None.

    The directory half of the same trap: `mkdir` inside a 0o555 ancestor
    raises EACCES exactly as `os.open(O_CREAT)` does, and it sat OUTSIDE the
    restore's try, so it reached the generic 500 without even the errno
    check (§4GK round 6).

    ⚠ "REPORT, NEVER RAISE" WAS WRITTEN FOR TWO ERRNOS (§4GK round 7).
    Everything else still re-raised — into the route's generic handler,
    AFTER the wipe, which is the precise outcome the rule exists to
    eliminate. Confirmed by driving the real route: a zip holding both
    `sandbox/a` (a file) and `sandbox/a/b.txt` gives EEXIST here on `a/`
    (`mkdir(exist_ok=True)` raises when the path exists and is NOT a
    directory) and ENOTDIR one level deeper — a 500 with the workspace
    already gone and no `unrestored` to say which member did it. One
    upload, and the whole sandbox is destroyed for an archive that a
    zipper can produce by accident. The retry stays EACCES/EPERM-only:
    unfreezing an ancestor cannot fix EEXIST, and pretending to repair is
    how a second failure gets a misleading reason.
    """
    for attempt in (1, 2):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return None
        except OSError as exc:
            repairable = exc.errno in (errno.EACCES, errno.EPERM)
            if attempt == 2 or not repairable or not _unfreeze_chain(path, stop):
                return _member_reason(exc)
    return None                          # pragma: no cover - loop returns first


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
            #: What the wipe could not remove. `ignore_errors=True` hid this:
            #: a directory the restore itself froze at 0o555 (see
            #: `_unfreeze_tree`) survived the wipe with its old contents and
            #: was then presented as the restored workspace (§4GK round 6).
            not_cleared = []
            for item in sandbox_dir.iterdir():
                if item.name == "acquired_skills":
                    continue
                # ⚠ UNFREEZING IS NOT UNCONDITIONAL (§4GK round 7). Round 6
                # chmods a frozen tree writable so the wipe can clear a
                # directory the ARCHIVE froze — and `projects/` is frozen for
                # a different reason entirely: `set_workspace_readonly` makes a
                # RELEASED project 0o555/0o444, and the OS half of that
                # immutability was the last thing standing between a restore
                # and the human-attested deliverables inside it. Round 5's
                # `rmtree(ignore_errors=True)` could not delete through it;
                # round 6 could, silently, reporting `not_cleared: []` and
                # `{"status": "success"}` while the project rows pointed at a
                # deleted directory. A restore replaces the WORKSPACE, and a
                # released project is not part of what the archive owns.
                if item.name == "projects":
                    not_cleared.append(
                        f"{item.name} (released projects are never wiped by a "
                        f"restore)")
                    continue
                try:
                    if item.is_dir() and not item.is_symlink():
                        _unfreeze_tree(item)
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                except OSError as _exc:
                    # ⚠ AND THE UNLINK BRANCH HAD NO HANDLER (§4GK round 7).
                    # One `chmod 555 /workspace` from inside the container made
                    # a PermissionError abort the whole loop, so nothing was
                    # extracted and the client got an opaque 500 — the outcome
                    # the "report, never raise" rule at the restore below
                    # exists to eliminate, left open on its own wipe.
                    not_cleared.append(f"{item.name} ({type(_exc).__name__})")
                    continue
                if item.exists() or item.is_symlink():
                    not_cleared.append(item.name)

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

            # 3. Extract files — WITH the modes and mtimes the archive
            # carries (§4GK round 5). `write_bytes` + `mkdir` applied
            # neither, so round 4's user-visible defect was only half
            # fixed: through the agent, `run.sh` still came back 0o644 and
            # every mtime was still the moment of the restore.
            _dir_members = []
            #: Members this restore could not write, as {path, reason}. A
            #: PermissionError used to escape to the route's generic handler
            #: — an opaque 500 handed to the operator AFTER the sandbox had
            #: been wiped, with no way to tell which member failed (§4GK
            #: round 6). The wipe and `_unfreeze_chain` between them make a
            #: frozen directory recoverable; anything still unwritable after
            #: that is reported by name instead of destroying the response.
            unrestored = []
            for zip_info in zip_ref.infolist():
                if not zip_info.filename.startswith("sandbox/"):
                    continue
                rel_path = zip_info.filename[len("sandbox/"):]
                if not rel_path:
                    continue
                # ⚠ AND THE MEMBER LOOP ITSELF COULD RAISE (§4GK round 7).
                # Not only the two calls that had handlers: `.resolve()` and
                # `.is_symlink()` are `os.lstat` underneath, and `lstat` on a
                # 300-byte component answers ENAMETOOLONG — which `pathlib`
                # re-raises rather than treating it as "not a link". Measured
                # against the real route: `OSError [Errno 63] File name too
                # long` out of the symlink CHECK, into the generic handler —
                # a 500 with the sandbox already wiped. Everything a single
                # member can do to this loop is THAT MEMBER's failure, and by
                # here the old workspace is gone, so the whole per-member body
                # reports and only the loop carries on.
                try:
                    extracted_path = (sandbox_dir / rel_path).resolve()
                    # Prevent zip-slip traversal (relative_to-based; rejects
                    # sibling-dir prefix escapes).
                    #
                    # ⚠ AND A REFUSED MEMBER IS STILL A MISSING MEMBER (§4GK
                    # round 7). Four `continue`s in this loop dropped a member
                    # without recording it, so the route answered
                    # `{"status": "success", "unrestored": []}` over an archive
                    # whose files are NOT on disk — and every client believes
                    # that shape: app.js prints "Workspace loaded successfully",
                    # the handheld printed "workspace restored." A refusal is a
                    # decision the operator has to be able to see, especially
                    # this one: a zip-slip member is the loudest thing an
                    # archive can contain and it was the quietest thing in the
                    # response.
                    if not _is_within(sandbox_dir, extracted_path):
                        unrestored.append({
                            "path": rel_path,
                            "reason": "REFUSED: escapes the sandbox (zip slip)"})
                        continue
                    if zip_info.is_dir():
                        _why = _mkdir_unfreezing(extracted_path, sandbox_dir)
                        if _why:
                            unrestored.append({"path": rel_path, "reason": _why})
                            continue
                        # mode/mtime after the whole archive lands — see
                        # `_restore_dir_metadata`.
                        _dir_members.append((extracted_path, zip_info))
                    else:
                        _why = _mkdir_unfreezing(extracted_path.parent, sandbox_dir)
                        if _why:
                            unrestored.append({"path": rel_path, "reason": _why})
                            continue
                        # A member whose target (or whose parent) is a symlink
                        # would write THROUGH it, outside the mount — the
                        # `_is_within` check above resolves the path but a link
                        # planted between that check and this write, or a link
                        # already at the destination, still redirects the bytes.
                        # Refuse both rather than follow (§4GJ R3) — and SAY so:
                        # the bytes are not on disk, whatever the reason.
                        if extracted_path.is_symlink() or extracted_path.parent.is_symlink():
                            unrestored.append({
                                "path": rel_path,
                                "reason": "REFUSED: destination (or its parent) is "
                                          "a symlink"})
                            continue
                        _payload = zip_ref.read(zip_info.filename)
                        try:
                            _restore_file_member(extracted_path, _payload, zip_info)
                        except OSError as exc:
                            if exc.errno == errno.ELOOP:
                                # a link planted between the check above and the
                                # open: the same refusal, one race window later
                                unrestored.append({
                                    "path": rel_path,
                                    "reason": "REFUSED: a symlink appeared at the "
                                              "destination during the restore"})
                                continue
                            if exc.errno not in (errno.EACCES, errno.EPERM):
                                # ⚠ AND THE OTHER ERRNOS STILL RAISED (§4GK round
                                # 7). Confirmed against the real route: EISDIR (a
                                # member whose name is an existing directory),
                                # ENAMETOOLONG (a >255-byte component, which a zip
                                # can carry and this filesystem cannot) and ENOSPC
                                # (end to end, on a full disk) each reached the
                                # generic handler — a 500 with an error id, AFTER
                                # the wipe, for a fault that concerns ONE member.
                                # By this line the old workspace is already gone,
                                # so there is no failure mode left that is better
                                # than reporting the member by name.
                                unrestored.append({"path": rel_path,
                                                   "reason": _member_reason(exc)})
                                continue
                            # ⚠ A FROZEN ANCESTOR IS NOT A BROKEN ARCHIVE (§4GK
                            # round 6). This used to `raise` — into the route's
                            # generic handler, i.e. an opaque 500 AFTER the
                            # sandbox had been wiped, which is how one 0o555
                            # directory made every later restore fail for ever.
                            # Repair the chain and retry ONCE; a second failure
                            # is REPORTED, never raised, because by here the old
                            # workspace is already gone.
                            if _unfreeze_chain(extracted_path.parent, sandbox_dir):
                                try:
                                    _restore_file_member(extracted_path, _payload,
                                                         zip_info)
                                    continue
                                except OSError as retry_exc:
                                    if retry_exc.errno == errno.ELOOP:
                                        # the same refusal as above, one retry
                                        # later — and equally worth saying
                                        unrestored.append({
                                            "path": rel_path,
                                            "reason": "REFUSED: a symlink appeared "
                                                      "at the destination during "
                                                      "the restore"})
                                        continue
                                    exc = retry_exc
                            unrestored.append({"path": rel_path,
                                               "reason": _member_reason(exc)})
                except OSError as _member_exc:
                    unrestored.append({"path": rel_path,
                                       "reason": _member_reason(_member_exc)})
            # Deepest first: nothing here needs its parent writable, and a
            # directory's mtime is only final once everything inside it is.
            for _dir_path, _dir_info in sorted(
                    _dir_members, key=lambda pair: len(pair[0].parts), reverse=True):
                _restore_dir_metadata(_dir_path, _dir_info)

        if not_cleared or unrestored:
            # Said out loud, on the wire, for the same reason the save side
            # reports `omitted`: a restore that silently dropped members is
            # how a permissions glitch turns into data loss the operator
            # only discovers from the missing file (§4GJ round 4, §4GK
            # round 6).
            pretty_log("workspace/load",
                       f"incomplete restore: {len(not_cleared)} entr(ies) "
                       f"survived the wipe {not_cleared[:5]}, "
                       f"{len(unrestored)} member(s) unwritable "
                       f"{[u['path'] for u in unrestored[:5]]}",
                       icon=Icons.WARN, level="WARNING")
        return {"status": "success", "chat_history": chat_history,
                "not_cleared": not_cleared, "unrestored": unrestored}
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
    # §4GJ round 3: the session store is a STORE (files + fsync), reached via
    # a helper rather than a context attribute — which is why the first
    # enumeration could not see it. `list()` reads up to `limit` session
    # files; measured at 117 ms of blocked loop at the 200-session cap.
    try:
        sessions = await _store_call(store.list, limit=limit)
    except StoreCallTimeout as e:
        return _store_timeout_response(e)
    return JSONResponse({"enabled": True, "sessions": sessions})


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
    try:
        sess = await _store_call(store.create, title=str(title or ""))
    except StoreCallTimeout as e:
        return _store_timeout_response(e)
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
    try:
        sess = await _store_call(store.get, session_id)
    except StoreCallTimeout as e:
        return _store_timeout_response(e)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")
    # `sess` is a VALUE returned BY the store, not the store: `to_dict()` is
    # an in-memory shape change with no IO, so it stays on the loop.
    return JSONResponse(sess.to_dict())


@router.delete("/api/sessions/{session_id}", dependencies=[Security(verify_api_key)])
async def sessions_delete(request: Request, session_id: str):
    agent = get_agent(request)
    from ..core.sessions import get_session_store
    store = get_session_store(agent.context)
    if store is None:
        raise HTTPException(status_code=503, detail="sessions are not enabled")
    try:
        _deleted = await _store_call(store.delete, session_id)
    except StoreCallTimeout as e:
        return _store_timeout_response(e)
    if not _deleted:
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
        # §4GJ round 3: the activity log is a file-backed store reached via a
        # helper — the same class as the session store. This loop runs up to
        # 50 times, so parsing on the loop blocked it 50 x a file read.
        # ⚠ …AND THE HANDLER FOR IT (§4GJ round 4). The wrap went in without
        # its `except`, so a wedged ledger (a truncation racing a reader, an
        # NFS stall) came back to the Slack bot as a bare text/plain 500 —
        # which its poll loop cannot tell from "the agent crashed", while
        # every other store-backed route on this API answers a JSON 504.
        try:
            chunk, new_cursor = await _store_call(
                log.read_since, cursor, limit=200, severity=SEVERITY_NOTIFY)
        except StoreCallTimeout as e:
            return _store_timeout_response(e)
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
    try:
        ok, detail = await _store_call(memory.correct_fragment, match, replacement)
    except StoreCallTimeout as e:
        return _store_timeout_response(e)
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
    try:
        ok, detail = await _store_call(memory.delete_fragment, match)
    except StoreCallTimeout as e:
        return _store_timeout_response(e)
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
    try:
        removed, detail = await _store_call(memory.delete_skill_twins, triggers)
    except StoreCallTimeout as e:
        return _store_timeout_response(e)
    return {"ok": True, "requested": len(triggers), "removed": removed, **detail}


@router.post("/api/lessons/quarantine", dependencies=[Security(verify_api_key)])
async def lessons_quarantine(request: Request):
    """Quarantine a playbook lesson by trigger — IN-PROCESS (§4FB, 2026-09-06).

    The playbook has a single-writer contract (memory/skills.py
    `_crossproc_lock`): an external script must not write it while the agent
    runs, and launchd respawns the agent immediately on exit, so there is no
    process-free window either. This is the designed, reversible path
    (`SkillMemory.quarantine_lesson`: excluded from injection, kept on disk
    with the reason + timestamp, announced on the stream). First use: lesson
    49, a bench-probe prompt minted by reflection (see turn_origin "probe").
    Body: {"trigger": <exact trigger text>, "reason": <why>}.
    """
    agent = get_agent(request)
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, 400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, 400)
    trigger = str(body.get("trigger") or "").strip()
    reason = str(body.get("reason") or "").strip()
    if not trigger:
        return JSONResponse({"error": "body needs a non-empty 'trigger'"}, 400)
    if not reason:
        return JSONResponse({"error": "body needs a non-empty 'reason' (it is written on the record)"}, 400)
    sm = getattr(getattr(agent, "context", None), "skill_memory", None)
    if sm is None or not callable(getattr(sm, "quarantine_lesson", None)) \
            or getattr(sm, "is_read_only", False) is True:
        return JSONResponse({"error": "skill memory unavailable or read-only"}, 503)
    try:
        n = int(await _store_call(sm.quarantine_lesson, trigger, reason) or 0)
    except StoreCallTimeout as e:
        return _store_timeout_response(e)
    return {"ok": n > 0, "quarantined": n, "trigger": trigger[:120]}


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

class _BookedStreamingResponse(StreamingResponse):
    """A streaming response that releases its main-node booking in a
    `finally` around the WHOLE send — including the case where the client is
    already gone when the response starts, so the body generator (and its
    own `finally`) never runs. Without this a pre-iteration disconnect left
    `foreground_tasks` at 1 and parked the biological scheduler as "user
    active" forever (R3 review). `release` must be idempotent
    (`AsyncExitStack.aclose` is)."""

    def __init__(self, *args, release=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._release = release

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            if self._release is not None:
                await self._release()


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"], dependencies=[Security(verify_api_key)])
async def catch_all(request: Request, path: str):
    agent = get_agent(request)
    url = f"/{path}"
    # this API's own key and the hop-by-hop headers never go upstream (§4GI)
    headers = _forwardable_headers(request.headers)

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

    # Booked against the main node like a streamed turn (in-flight counted,
    # `foreground_tasks` held) for the WHOLE life of the proxied stream: the
    # booking is entered here and released by the generator's `finally`, or
    # right here when the upstream send fails.
    _llm = agent.context.llm_client
    _booking = _contextlib.AsyncExitStack()
    await _booking.enter_async_context(_main_node_request(_llm, hold_lock=False))
    try:
        req = _llm.http_client.build_request(
            request.method, url, headers=headers, content=forward_content
        )
        r = await _llm.http_client.send(req, stream=True)

        upstream_ct = r.headers.get("content-type", "") or ""
        is_event_stream = "text/event-stream" in upstream_ct.lower() or body_says_stream

        async def stream_generator():
            try:
                async for chunk in r.aiter_bytes():
                    yield chunk
            finally:
                try:
                    await r.aclose()
                finally:
                    await _booking.aclose()

        response_headers = {}
        if is_event_stream:
            # Tell nginx / any reverse proxy NOT to buffer this response.
            response_headers["X-Accel-Buffering"] = "no"
            response_headers["Cache-Control"] = "no-cache"

        return _BookedStreamingResponse(
            stream_generator(),
            status_code=r.status_code,
            media_type=("text/event-stream" if is_event_stream else upstream_ct or None),
            headers=response_headers,
            release=_booking.aclose,
        )
    except Exception as e:
        await _booking.aclose()
        pretty_log("Proxy Failed", f"{request.method} /{path}: {type(e).__name__}: {e}",
                   icon=Icons.FAIL, level="ERROR")
        return JSONResponse({"error": f"Proxy Error: {e}"}, 502)
    except BaseException:
        # ⚠ `except Exception` IS NOT THE WHOLE EXIT SET HERE (§4GJ round 4).
        # This app raises a BaseException through the RECEIVE channel on
        # purpose: `api/body_limit.BodyTooLarge` is a BaseException so it
        # sails past FastAPI's and these handlers' own `except Exception`
        # and reaches the middleware as a 413. The `forward_content =
        # request.stream()` branch reads the body from inside httpx, so that
        # exception is thrown right here — past the release above, leaving
        # `foreground_tasks` at 1 and the in-flight slot booked for a
        # request that no longer exists. It only ever self-healed when the
        # abandoned async generator was finalised; with anything holding the
        # traceback (a logger, a debugger, an `except` frame) the counter
        # stays at 1 and the biological tick never runs again (confirmed).
        # Release and re-raise: the exception's whole point is to travel.
        await _booking.aclose()
        raise