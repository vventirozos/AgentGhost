import json
import asyncio
import time
import logging
import copy
import os
from contextlib import asynccontextmanager
from functools import partial
from typing import List, Dict, Any, Optional
import httpx
from ..utils.logging import (Icons, pretty_log, request_id_context,
                             verify_purpose_context)
from ..utils.helpers import get_utc_timestamp, env_positive

logger = logging.getLogger("GhostAgent")

# --- Upstream streaming idle timeouts ---------------------------------------
# The per-chunk read is guarded so a genuinely hung upstream can't hold the
# event-loop slot forever (keeping foreground_tasks > 0 parks the biological
# watchdog). But the FIRST token of a turn can be legitimately slow: the
# upstream must prefill (prompt-eval) the whole context before emitting any
# bytes, and on a large context (e.g. 120k tokens) or a loaded/CPU node that
# prefill routinely exceeds the old flat 30s — with ZERO bytes during it —
# which tripped a false "stall" and forced a wasteful full re-prefill retry.
# So we split the budget: a generous time-to-FIRST-byte (covers prefill) and
# a tighter inter-token gap (catches a real mid-stream hang). Both env-tunable
# for slow/fast deployments. The httpx client timeout (1200s) bounds the whole
# request above these.
# Embeddings are a single forward pass over short text — seconds, not
# minutes. Unbounded, this call blocks every main completion behind it.
# How long to sit on a failed capacity probe before trying the node again.
_NODE_CAP_RETRY_S = env_positive("GHOST_NODE_CAP_RETRY_S", 300.0)
_EMBEDDINGS_TIMEOUT_S = env_positive("GHOST_EMBEDDINGS_TIMEOUT", 120.0)
_STREAM_FIRST_BYTE_TIMEOUT = env_positive("GHOST_STREAM_FIRST_BYTE_TIMEOUT", 180.0)
_STREAM_IDLE_TIMEOUT = env_positive("GHOST_STREAM_IDLE_TIMEOUT", 60.0)
# How long we will spend reading the BODY of a >=400 response before giving up
# on it. The body is only used to make an error message better; the stream is
# already doomed. Unbounded (the pre-R7 shape) it inherited the client's 1200s
# default, so a wedged upstream that returns headers-then-nothing parked the
# turn for twenty minutes on the error path (§4BV R7).
_STREAM_ERROR_BODY_TIMEOUT = env_positive(
    "GHOST_STREAM_ERROR_BODY_TIMEOUT", 15.0)


# Hostname suffixes that denote LAN infrastructure (never globally routable).
# Kept in sync with utils/notify.py's `url_needs_tor`, which makes the same
# call for outbound push targets.
_LAN_SUFFIXES = (".local", ".lan", ".home", ".internal", ".arpa")


def _socks5h(tor_proxy: str) -> str:
    """socks5h → DNS resolves inside Tor, never on the host."""
    return tor_proxy.replace("socks5://", "socks5h://")


# Ceiling for a `route()` call. It is awaited on the user's CRITICAL PATH
# (query expansion runs before the memory bus hydrates) and its fallback is
# free — a legacy string concat — so it must fail fast.
#
# Sizing (measured 2026-07-12 on the live Gemma-4-E4B worker, an M4 Mini):
# a WARM query-expansion call is ~2.3s uncontended. Two earlier causes are now
# handled elsewhere: COLD START (co-restart) by `warm_up_workers()` at boot, and
# a re-cooling network path by `keepalive_workers()`. What's LEFT is SLOT
# CONTENTION: the worker runs a small `-np` (measured -np 2 — 4 concurrent calls
# returned [2.7, 2.8, 5.3, 5.3]s), so a route() call that queues behind one other
# worker call (query expansion + classifiers/gates fire together at request
# start/finalize) lands at ~5.3s — just over a 5s ceiling, producing the residual
# `Nova: ReadTimeout` lines even on the LAN. 8s absorbed ONE queued call, but a
# call that queues behind TWO (verify fired mid-request alongside the
# start/finalize burst) still died at exactly 8.0s — observed live 2026-07-14:
#     +37.7s  worker compute      verify → Worker Node (Nova)
#     +45.7s  worker node failed  Nova: ReadTimeout — trying next
# 12s clears that double-queued case (~2×5.3s) while still failing fast on a
# genuinely sick node (the circuit breaker trips after 3 strikes). The real fix
# is more worker slots (operator: bump nova's -np); this is the margin that
# keeps the value flowing until then. Losing the expansion only costs a
# slightly cruder retrieval query, never correctness.
# When a POOL call falls back to the main model, the main model gets THIS
# budget — not the caller's node-sized one (too short for a 35B) and not
# httpx's 1200s default (an unbounded foreground occupation). Overridable for
# the rare operator who runs a much slower main model.
def _main_fallback_timeout_s() -> float:
    """Budget for a pool call that has fallen back to the MAIN model.

    ⚠ Clamped, and not merely parsed. `GHOST_MAIN_FALLBACK_TIMEOUT=0` made
    `max(timeout, 0.0)` return the CALLER's node-sized budget — a 6s route
    timeout on the 35B, i.e. the exact 2026-07-11 failure this bound exists
    to prevent. And "<=0 disables" is this codebase's convention elsewhere
    (GHOST_WORKER_KEEPALIVE_S), so the obvious operator reading produced the
    worst outcome (R2 lens A)."""
    raw = os.environ.get("GHOST_MAIN_FALLBACK_TIMEOUT")
    try:
        val = float(raw) if raw not in (None, "") else 300.0
    except (TypeError, ValueError):
        return 300.0
    return val if val > 0 else 300.0


_MAIN_FALLBACK_TIMEOUT_S = _main_fallback_timeout_s()

_ROUTE_TIMEOUT_S = 12.0


class OffMainNodeUnavailable(Exception):
    """Every off-main node for this call failed AND the caller forbade the
    main-model fallback (``off_main_only=True``).

    Raised instead of silently re-running the request on the foreground model
    — see `route()`, whose whole purpose is to keep small sub-tasks OFF the
    single main inference slot.
    """


def _disable_thinking(node_payload: Dict[str, Any]) -> None:
    """Turn OFF chain-of-thought for a WORKER-routed call (2026-07-11).

    Worker-pool work is mechanical by definition — rewrite a query, classify a
    task into one word, extract JSON, summarise. Hidden reasoning buys nothing
    and costs everything.

    MEASURED on the live worker (Gemma 4 E4B, a reasoning model with thinking
    ON by default) for the exact query-expansion call ``route`` makes:

        as sent before this fix : 7.0s, 128/128 tokens, 472 chars of hidden
                                  reasoning, and **content == ""**
        enable_thinking=False   : 0.5s, 5 tokens, correct answer

    i.e. the model burned its ENTIRE token budget thinking, returned an EMPTY
    answer, and the caller fell back to its legacy path anyway — so the offload
    was adding ~13.7s to the front of every user request (measured in prod: the
    worker call fires at +0.01s and the memory bus doesn't hydrate until
    +13.8s) in exchange for NOTHING, and periodically tripped the 15s timeout
    (`Nova: ReadTimeout`). A 14x latency regression that also didn't work.

    Applied to ``node_payload`` (a copy), so a fallback to the main model keeps
    the caller's original payload untouched. ``setdefault`` semantics: an
    explicit caller preference always wins. NOTE: ``reasoning_effort="none"``
    was also measured and does NOT suppress thinking on this template — only
    the chat-template kwarg does.
    """
    kw = node_payload.get("chat_template_kwargs")
    if not isinstance(kw, dict):
        kw = {}
    else:
        kw = dict(kw)
    kw.setdefault("enable_thinking", False)
    node_payload["chat_template_kwargs"] = kw


def compute_tor_proxy(url: str, tor_proxy: Optional[str]) -> Optional[str]:
    """Decide whether traffic to ``url`` must egress via Tor.

    Returns the (socks5h-normalised) proxy URL for genuinely public
    destinations, or ``None`` when ``url`` is local/LAN infrastructure that
    should be reached directly.

    "Local" is defined as *not globally routable* — the same predicate
    ``egress_guard.is_allowed_host`` uses. This deliberately covers more than
    RFC1918+loopback: it also exempts CGNAT / Tailscale (100.64.0.0/10),
    link-local, and IPv6 ULA. The older `is_private or is_loopback` test
    missed those, so a tailnet compute node (e.g. an image-gen GPU at
    ``100.x.x.x``) was forced through a Tor exit that cannot route a tailnet
    address — every connect failed with "All connection attempts failed".
    """
    if not tor_proxy:
        return None
    try:
        import urllib.parse
        import ipaddress

        # Robustly handle URLs missing the http:// scheme
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url

        # Check the parsed HOSTNAME, not a substring of the whole URL — the
        # old `"localhost" in url` shortcut bypassed Tor for a PUBLIC host like
        # `http://localhost.attacker.example/` (real-IP leak).
        hostname = (urllib.parse.urlparse(url).hostname or "").lower()
        if hostname:
            if hostname == "localhost" or hostname.endswith(_LAN_SUFFIXES):
                return None  # local name → bypass Tor (can't route via exit anyway)
            # IP literals are classified by the address itself. This MUST be
            # tried before the dotless-hostname rule below: an IPv6 literal
            # (`2606:4700:4700::1111`) has colons and no dots, and would
            # otherwise be misread as a LAN name and leak OUTSIDE Tor.
            try:
                ip = ipaddress.ip_address(hostname)
            except ValueError:
                pass
            else:
                # loopback / RFC1918 / CGNAT-Tailscale / link-local / ULA
                return None if not ip.is_global else _socks5h(tor_proxy)

            # Not an IP literal. A DOTLESS hostname cannot be a public DNS
            # name — a globally resolvable name needs a TLD. So `nova`,
            # `ghost`, `raspberrypi` are LAN infrastructure resolved via
            # /etc/hosts, mDNS or the LAN search domain, and a Tor exit can no
            # more route to them than to a 192.168.x address.
            #
            # This was a REAL, SILENT bug (found live 2026-07-11): a worker
            # node configured as `--worker-nodes http://nova:8088|Nova` was
            # forced through the SOCKS proxy, so EVERY offloaded call died with
            # `ProxyError` and fell back to the main model — the log said
            # "Routing background task to Worker Node (Nova)" and then "All
            # worker nodes failed", so offloading appeared configured while
            # silently doing nothing. Same for an image-gen node at
            # `http://ghost:8000`. The IP branch above already covered this
            # class for Tailscale/RFC1918 ADDRESSES; bare hostnames were the
            # remaining hole.
            if "." not in hostname:
                return None

            # A dotted, non-IP hostname: assume PUBLIC → route via Tor. We
            # deliberately do NOT resolve it to check for a private answer:
            # that would leak a cleartext DNS query for every node URL, which
            # is exactly what mandatory-tor exists to prevent. A LAN node on a
            # dotted custom domain must be given as an IP or a _LAN_SUFFIXES
            # name.
    except Exception:
        pass
    return tor_proxy.replace("socks5://", "socks5h://")


class NodeSaturated(Exception):
    """We could not get a permit for this node before the deadline.

    ⚠ THIS IS NOT A NODE FAULT AND MUST NEVER REACH THE CIRCUIT BREAKER. The
    request never left this process — the node was not asked and cannot have
    failed. Before the per-node gate existed the same over-subscription showed
    up as a ReadTimeout from llama-server's queue, which `_is_node_fault()`
    (correctly, for what it could see) counted as illness: our own fan-out
    ejected a healthy node for 60s. Moving the wait from the server's queue
    into our own gate is what makes the distinction PROVABLE rather than
    heuristic — the difference between "we never asked" and "it did not
    answer".
    """


def _err_text(exc) -> str:
    """A non-empty description of an exception, always.

    ⚠ `str(e)` is the EMPTY STRING for httpx's entire timeout family —
    ReadTimeout, ConnectTimeout, PoolTimeout, WriteTimeout — and for
    ReadError/WriteError/RemoteProtocolError/ProxyError. Those are precisely
    the failures a node or the main model produces under load, so the
    operator-visible line degraded to "Upstream Failed: Failed after 2
    attempts: " with no cause at all, and the image path handed the MODEL a
    blank diagnosis to reason about. This module already knew the answer in
    two places (`str(e) or repr(e)` and `_node_error_detail`); the fix had
    been applied at 1 of 11 sites (LLM review 2026-08-18, two lenses)."""
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 — a __str__ that raises is still a fault
        text = ""
    return text or f"{type(exc).__name__} (no message)"


# Floor for the per-node permit wait. A caller-supplied budget below this
# is treated as this — see the `_slot_wait` comment in `_do_chat_completion`
# for why a 0 here is indistinguishable from "no gate at all".
_MIN_SLOT_WAIT = 5.0

# Reserved out of the pool budget so a caller cannot spend everything queueing
# and then POST with nothing left. Spending a scarce node permit on a request
# that provably cannot finish is worse than never asking for it.
_MIN_HTTP_FLOOR = 3.0

# A permit request is never made for less than this. Zero is not a fast path:
# `asyncio.wait_for(sem.acquire(), 0.0)` rejects even a completely FREE
# semaphore, so a 0 budget means "never ask this node" (R6).
_MIN_ACQUIRE = 0.05

# Sentinel: the caller's total is spent, so no request can complete. Distinct
# from `None`, which means "the caller named no timeout, do not clip".
_BUDGET_BLOWN = object()

# The smallest budget worth giving the 35B on a last-resort fallback. Below
# this a main call is a guaranteed ReadTimeout (2026-07-11), so a caller's
# remaining total is honoured only down to here.
_MAIN_FALLBACK_MIN_S = 60.0


# Canonical definition lives in utils.helpers so the timeout constants in
# agent.py / build_gates.py / verifier.py can share it — R4 lens A found
# THREE of them reproducing both traps this guard exists to close, two of
# them written in the very round that introduced the guard.
_env_positive = env_positive


def _is_node_fault(exc) -> bool:
    """True if ``exc`` indicates the NODE itself is unhealthy (→ count it
    toward the circuit breaker), False for a caller-fault that would repeat
    identically on any node — an HTTP 4xx (bad/oversized payload, unknown
    model). Counting 4xx as node failures trips the breaker on a perfectly
    HEALTHY node for a deterministic caller bug (e.g. a verify payload
    exceeding Nova's n_ctx), taking the node out of rotation for 60s and
    forcing every routed call onto the main slot. Timeouts / connection
    errors / 5xx stay node faults.

    ⚠ AND SATURATION IS OURS, NOT THE NODE'S (2026-08-11). `NodeSaturated`
    means we never sent the request — no permit was free within the wait
    budget. A node cannot fail a request it was never asked. This is the
    second half of the req-0fb69c5f defect: before the per-node gate, that
    same over-subscription arrived as a ReadTimeout from llama-server's own
    queue, which is indistinguishable from real slowness, so our fan-out
    ejected a healthy Nova for 60s. The gate does not merely prevent the
    flood — it relocates the wait to a place where the cause is KNOWN."""
    if isinstance(exc, NodeSaturated):
        return False
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    if isinstance(code, int) and 400 <= code < 500:
        return False
    return True


def _node_error_detail(exc) -> str:
    """One-line diagnosis for a node-failure log. For an HTTPStatusError the
    bare class name hides the actual cause (a 400 'unsupported image format'
    reads identically to a 500 crash), so include the status code and a
    bounded body snippet; every other exception keeps its class name."""
    if isinstance(exc, NodeSaturated):
        # Do not let this read as node illness in the log either — the node was
        # never contacted. Names the cause so an operator sees over-subscription
        # rather than hunting a node that is fine.
        return f"SATURATED — {exc} (node not contacted; not a node fault)"
    if isinstance(exc, httpx.HTTPStatusError):
        detail = f"HTTP {exc.response.status_code}"
        try:
            body = " ".join((exc.response.text or "").split())[:160]
        except Exception:
            body = ""
        return f"{detail} — {body}" if body else detail
    return type(exc).__name__


class NodeCircuitBreaker:
    """Circuit breaker for LLM nodes.

    Tracks consecutive failures per node. After ``failure_threshold``
    consecutive failures, the node is marked "open" (unavailable) for
    ``cooldown_seconds``. After cooldown the state becomes half_open, and
    the next recorded outcome decides: a success resets the breaker, a
    failure starts another cooldown.

    ⚠ half_open is not single-flight. Every caller arriving during it is
    admitted (measured: 50/50), so under a fan-out the whole burst reaches a
    node that has just been declared sick — see the note at the half_open
    return in ``is_available``.

    States: CLOSED (healthy) → OPEN (unavailable) → HALF_OPEN (probing)
    """
    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        # node_url → {"failures": int, "open_since": float|None, "state": str}
        self._states: dict = {}

    def _get_state(self, node_url: str) -> dict:
        if node_url not in self._states:
            self._states[node_url] = {"failures": 0, "open_since": None, "state": "closed"}
        return self._states[node_url]

    def is_available(self, node_url: str) -> bool:
        """Check if a node is available for requests."""
        import time
        state = self._get_state(node_url)
        if state["state"] == "closed":
            return True
        if state["state"] == "open":
            elapsed = time.time() - (state["open_since"] or 0)
            if elapsed >= self.cooldown_seconds:
                state["state"] = "half_open"
                # Durable node-health timeline: the OPEN trip logged but the
                # probe/recovery never did, so the operator saw nodes die and
                # never heal. Probe is diagnostic (grep-log); recovery is
                # operator-facing (pretty stream, below).
                logger.info(
                    "Circuit breaker HALF_OPEN for node %s — probing after "
                    "%.0fs cooldown", node_url, self.cooldown_seconds)
                return True  # Allow probe request
            return False
        # ⚠ half_open admits EVERY caller until the next recorded outcome —
        # it is not a single probe, despite what this comment and the class
        # docstring used to say. There is no counter, token or in-flight
        # flag: R4 lens B put 50 consecutive calls through this branch and
        # got 50 True, with the state still half_open. Under a fan-out (the
        # req-0fb69c5f shape: 9-20 concurrent calls) all of them are admitted
        # to a node the breaker has just decided is sick.
        #
        # Left as-is deliberately. Single-flight here needs a `probing` flag
        # cleared by the next record_success/record_failure, and a probe that
        # is never recorded (a task cancelled mid-flight) would then wedge the
        # node closed forever — a worse failure than the one it fixes, and a
        # shape this codebase has already been bitten by. Fix it with an
        # expiring token, or not at all; do not fix it with a bare bool.
        return True

    def record_success(self, node_url: str):
        """Record a successful request — reset the breaker."""
        state = self._get_state(node_url)
        prev = state["state"]
        state["failures"] = 0
        state["open_since"] = None
        state["state"] = "closed"
        # Surface RECOVERY on the operator's stream, symmetric with the OPEN
        # trip warning — a node healing is exactly as newsworthy as it tripping.
        if prev != "closed":
            pretty_log("Node Recovered",
                       f"{node_url} circuit CLOSED (was {prev})", icon=Icons.OK)

    def record_failure(self, node_url: str):
        """Record a failed request — potentially trip the breaker."""
        import time
        state = self._get_state(node_url)
        _was_open = state["failures"] >= self.failure_threshold
        state["failures"] += 1
        if state["failures"] >= self.failure_threshold:
            state["state"] = "open"
            state["open_since"] = time.time()
            # ⚠ ANNOUNCE THE TRANSITION, not the state. This logged on every
            # failure at or past the threshold, and `failures` is unbounded —
            # so 20 consecutive failures produced 18 WARNINGs and a node that
            # stays down logs forever (46 lines per 46 POSTs, measured).
            # Combined with the keepalive cadence that is ~4 WARNINGs every
            # 45s from one sick node (R2 lens B, NEW-2).
            if not _was_open:
                logger.warning(f"Circuit breaker OPEN for node {node_url} after {state['failures']} consecutive failures (cooldown {self.cooldown_seconds}s)")

    def get_status(self) -> dict:
        """Return the current state of all tracked nodes."""
        return {url: dict(s) for url, s in self._states.items()}


class RoutingTask:
    """Stable string labels for the `route()` calls that USE one.

    ⚠ Not a closed set, and not type-checked. `route()` takes any string —
    `verifier.py` passes a bare `"VERIFY"` — so a call site that invents a
    label is valid by construction. That is fine, and deliberate:
    `_TASK_DISPLAY_LABELS` falls back to the identity, and
    `test_failure_dimension.py` pins that contract. What is NOT fine is a
    docstring claiming otherwise, which the previous one did twice ("so call
    sites are type-checked", "centralised so a future routing-model swap
    lands in one place" — contradicted by the highest-traffic caller).

    Deliberately plain strings, not Enum members: `str(task)` on an Enum
    yields `RoutingTask.EXPAND_QUERY`, which lowercases into the operator
    stream as "routingtask.expand query".

    Members are the labels with a live caller. Four more
    (`VALIDATE_TOOL_ARGS`, `CLASSIFY_INTENT`, `SCORE_RELEVANCE`,
    `REPAIR_JSON`) were removed in R5 after a tree-wide search found zero
    production uses — an aspirational list read as an inventory of what the
    routing layer does.
    """
    EXPAND_QUERY = "EXPAND_QUERY"          # bus.py, agent.py
    CLASSIFY_FAILURE = "CLASSIFY_FAILURE"  # failure_dimension.py
    DISTILL_PATTERN = "DISTILL_PATTERN"    # failure_distill.py


# Human-readable stream labels. The ROUTING label stays the canonical string
# above (tests and docs assert on it, and it is what a timeout logs) — this map
# only changes what the operator reads in the live stream.
#
# Why it exists: `CLASSIFY_FAILURE` mechanically lowercases to "classify
# failure", which renders as
#     🔧  worker compute      classify failure → Worker Node (Nova)
# and reads as "the classify call FAILED" — the operator asked exactly that
# about seven consecutive INFO dispatch lines (2026-08-04). A genuine node
# failure is a different line entirely: "worker node failed  Nova: ReadTimeout
# — trying next", WARNING level, ⚠ icon. Any task label whose words could be
# read as an outcome belongs here; the fallback stays the 2026-07-12 rule of
# echoing the real task rather than a hardcoded guess.
_TASK_DISPLAY_LABELS = {
    "CLASSIFY_FAILURE": "tag failure-dimension",
    "DISTILL_PATTERN": "distill failure-pattern",
}


def _task_display_label(task) -> str:
    key = str(task)
    return _TASK_DISPLAY_LABELS.get(key, key.replace("_", " ").lower())



def _stamp_leg(result, leg: str, fell_back_from: str = "",
                requested: str = ""):
    """Record WHICH node actually served a result.

    ⚠ Without this the caller cannot tell a pool answer from a main-model
    fallback: `_do_chat_completion` catches every pool failure (and our own
    NodeSaturated), silently re-runs on the 35B, and returns a dict that is
    byte-identical in shape. The verifier then stamped `route="critic"` on a
    verdict the MAIN model produced, so §4BR's degradation guard — which
    aborts a self-consistency vote when `route in ("main","failed")` — could
    never fire, and every sample piled onto the single foreground slot. The
    same blindness mis-attributes the model in `_maybe_record_call`, which
    feeds the §4BG fixture corpus (three independent lenses, 2026-08-18).

    Non-invasive by construction: a private key on the response dict, ignored
    by every existing consumer."""
    try:
        if isinstance(result, dict):
            result["_ghost_leg"] = {"served_by": leg,
                                    "fell_back_from": fell_back_from,
                                    "requested": requested or leg}
    except Exception:  # noqa: BLE001 — telemetry must never break a reply
        pass
    return result


def served_leg(result) -> dict:
    """`{"served_by": …, "fell_back_from": …}` for a chat_completion result.

    `served_by` is one of: main, worker, critic, vision, coding, swarm.
    `fell_back_from` names the pool that was REQUESTED and failed (empty when
    the requested leg served it)."""
    if isinstance(result, dict):
        leg = result.get("_ghost_leg")
        if isinstance(leg, dict):
            return leg
    return {"served_by": "", "fell_back_from": "", "requested": ""}


class LLMClient:
    def __init__(self, upstream_url: str, tor_proxy: Optional[str] = None, swarm_nodes: Optional[list] = None, worker_nodes: Optional[list] = None, visual_nodes: Optional[list] = None, coding_nodes: Optional[list] = None, image_gen_nodes: Optional[list] = None, critic_nodes: Optional[list] = None, node_api_key: Optional[str] = None):
        self.upstream_url = upstream_url
        limits = httpx.Limits(max_keepalive_connections=3, max_connections=15, keepalive_expiry=30.0)

        def get_proxy(url: str) -> Optional[str]:
            return compute_tor_proxy(url, tor_proxy)
        # Determine if we need to route through Tor
        # If upstream is NOT localhost, we force Tor usage
        proxy_url = get_proxy(upstream_url)
        if proxy_url:
            pretty_log("LLM Connection", f"Routing upstream traffic via Tor ({proxy_url})", icon=Icons.SHIELD)

        self.circuit_breaker = NodeCircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)
        self.foreground_tasks = 0
        # url -> monotonic deadline before which not to re-probe capacity
        self._node_cap_retry_at: Dict[str, float] = {}
        # url -> the capacity each live semaphore was sized from
        self._node_slot_built_cap: Dict[str, int] = {}
        # Per-URL, so one unreachable node cannot stall dispatch
        # to every other node while its capacity is probed.
        self._node_slot_locks: Dict[str, asyncio.Lock] = {}
        # Active USER REQUESTS (handle_chat in flight at the API layer), as
        # opposed to in-flight foreground LLM calls. A user turn spends much
        # of its wall-clock BETWEEN LLM calls (tools, file I/O, browser);
        # `foreground_tasks` drops to 0 in those gaps, a background turn
        # grabs the single llama slot, and the user's NEXT turn queues
        # behind a full background generation — the post-req-70 "no prompt
        # for 12 minutes" starvation. The API layer increments this for the
        # whole life of a user request; background callers wait on BOTH
        # counters. Plain int (not lock-guarded): all writers live on the
        # one event loop and the readers tolerate one-tick staleness.
        self.foreground_requests = 0
        # Guards mutations of `foreground_tasks`. Without this the biological
        # watchdog could observe a stale (negative or stuck) value and either
        # spin forever or fire mid-request. Asyncio.Lock is sufficient because
        # all readers/writers live on the same event loop.
        self._foreground_lock = asyncio.Lock()
        self._bg_queue_sem = asyncio.Semaphore(3)  # Allow up to 3 concurrent background tasks
        # ── PER-NODE CONCURRENCY GATE (2026-08-11) ───────────────────────────
        # One budget per node URL, shared by EVERY caller and every role.
        #
        # WHY IT HAS TO LIVE HERE. Tools capped themselves: `search.py` builds
        # an `asyncio.Semaphore(3)` INSIDE deep_research, so the cap is per
        # CALL. Live on req 0fb69c5f the model issued THREE deep_research calls
        # in one batch — 3 semaphores × 3 permits = **9 concurrent requests at
        # a node advertising 4 slots** (20 worker calls in 74s). The excess
        # queued on llama-server past the route timeout, and every one of those
        # ReadTimeouts counted as a NODE fault, so 3 in a row tripped the
        # breaker and ejected a perfectly healthy Nova for 60s. It "recovered"
        # 20s later because nothing was ever wrong with it.
        #
        # No tool can fix this from where it stands: Nova serves the WORKER and
        # CRITIC roles at once on this deployment, and query-expansion, web
        # summaries, fact distillation and the verifier all reach it by
        # different paths. Keying on URL is what makes the budget authoritative
        # — role-keyed limits would each be individually polite and still
        # collectively flood one box.
        self._node_slots: Dict[str, asyncio.Semaphore] = {}
        self._node_slot_caps: Dict[str, int] = {}
        self._node_slots_lock = asyncio.Lock()
        # Fallback when a node will not tell us its capacity. 3 matches the old
        # per-call value, so an unprobeable node behaves exactly as it did
        # before rather than becoming unbounded — a gate whose failure mode is
        # "no gate" is the shape this whole change exists to remove.
        self._node_slot_default = max(1, int(
            os.getenv("GHOST_NODE_SLOTS_DEFAULT", "3") or 3))
        self._main_node_lock = asyncio.Lock()
        # ── OUR-OWN-TRAFFIC COUNTER (§4BV R7) ──────────────────────────────
        # node url -> number of OUR OWN streamed requests currently on the
        # wire to it. Its ONLY consumer is the stall watchdog's attribution:
        # "No bytes for 180s" is a statement about the UPSTREAM only if we
        # were not the ones queueing in front of ourselves. Measured before
        # this existed: 3 streams + 1 POST at a total_slots=1 llama-server,
        # and 2 of the 3 streams reported "Upstream Stream Stall" for a
        # prefill queue WE created. That is the same diagnosis corruption
        # `NodeSaturated` exists to remove one layer down.
        #
        # Keyed on the CONFIGURED url string (`node["url"]` /
        # `self.upstream_url`), never `str(client.base_url)` — httpx
        # normalises the latter (`http://h:8088/v1` -> `.../v1/`,
        # `http://h:80` -> `http://h`), and `_node_slot_caps` is keyed on the
        # configured string, so a normalised key would silently never match
        # and the whole verdict would be dead code.
        self._inflight_by_url: Dict[str, int] = {}
        self.http_client = httpx.AsyncClient(
            base_url=upstream_url,
            timeout=1200.0,
            limits=limits,
            proxy=proxy_url,
            trust_env=False,
            follow_redirects=True,
            http2=False
        )

        self.swarm_clients = []
        self._swarm_index = 0

        if swarm_nodes:
            for node in swarm_nodes:
                client = httpx.AsyncClient(
                    base_url=node["url"],
                    timeout=1200.0,
                    limits=limits,
                    proxy=get_proxy(node["url"]),
                    trust_env=False,
                    follow_redirects=True,
                    http2=False
                )
                self.swarm_clients.append({
                    "client": client,
                    "url": node["url"],
                    "model": node["model"]
                })

        self.worker_clients = []
        self._worker_index = 0

        if worker_nodes:
            for node in worker_nodes:
                client = httpx.AsyncClient(
                    base_url=node["url"],
                    timeout=1200.0,
                    limits=limits,
                    proxy=get_proxy(node["url"]),
                    trust_env=False,
                    follow_redirects=True,
                    http2=False
                )
                self.worker_clients.append({
                    "client": client,
                    "url": node["url"],
                    "model": node["model"]
                })

        self.vision_clients = []
        self._vision_index = 0

        if visual_nodes:
            for node in visual_nodes:
                client = httpx.AsyncClient(
                    base_url=node["url"],
                    timeout=1200.0,
                    limits=limits,
                    proxy=get_proxy(node["url"]),
                    trust_env=False,
                    follow_redirects=True,
                    http2=False
                )
                self.vision_clients.append({
                    "client": client,
                    "url": node["url"],
                    "model": node["model"]
                })

        self.coding_clients = []
        self._coding_index = 0
        if coding_nodes:
            for node in coding_nodes:
                client = httpx.AsyncClient(
                    base_url=node["url"],
                    timeout=1200.0,
                    limits=limits,
                    proxy=get_proxy(node["url"]),
                    trust_env=False,
                    follow_redirects=True,
                    http2=False
                )
                self.coding_clients.append({
                    "client": client,
                    "url": node["url"],
                    "model": node["model"]
                })

        self.image_gen_clients = []
        self._image_gen_index = 0
        if image_gen_nodes:
            for node in image_gen_nodes:
                client = httpx.AsyncClient(
                    base_url=node["url"],
                    timeout=1200.0,
                    limits=limits,
                    proxy=get_proxy(node["url"]),
                    trust_env=False,
                    follow_redirects=True,
                    http2=False,
                    # The image node checks the fleet key (it binds 0.0.0.0 on
                    # the LAN); llama.cpp pools don't, so only this pool sends it.
                    headers={"X-Ghost-Key": node_api_key} if node_api_key else None,
                )
                self.image_gen_clients.append({
                    "client": client,
                    "url": node["url"],
                    "model": node["model"]
                })

        # Dedicated CRITIC pool. A separate node pool (typically a slower,
        # off-host model — e.g. a 9B on a spare Mac Mini) reserved for the
        # self-evaluation verifier. Kept distinct from `worker_clients` on
        # purpose: the worker pool is for fast, latency-sensitive cognitive
        # chores (routing, query expansion, arg validation) that must NOT
        # queue behind a slow critic. Routing the verifier here also keeps
        # its calls off the foreground inference slot, so a verdict never
        # competes with the Studio's main model for the KV-cache.
        self.critic_clients = []
        self._critic_index = 0
        if critic_nodes:
            for node in critic_nodes:
                client = httpx.AsyncClient(
                    base_url=node["url"],
                    timeout=1200.0,
                    limits=limits,
                    proxy=get_proxy(node["url"]),
                    trust_env=False,
                    follow_redirects=True,
                    http2=False
                )
                self.critic_clients.append({
                    "client": client,
                    "url": node["url"],
                    "model": node["model"]
                })

    # ====================================================================
    # ARCHITECTURAL OPTIMISATION #2: TWO-TIER MODEL ROUTING
    # --------------------------------------------------------------------
    # Many cognitive sub-tasks (intent classification, query expansion,
    # tool-arg validation, relevance scoring) don't need the big foreground
    # model. `route()` dispatches them to the worker pool with a tiny
    # canned prompt and a hard token cap, so they stay cheap and never
    # block the foreground inference slot.
    #
    # Use `RoutingTask` enum values (defined in this module) so call sites
    # are type-checked and a future routing-model swap is one place.
    # ====================================================================
    async def route(self,
                    task: str,
                    payload: Dict[str, Any],
                    max_tokens: int = 128,
                    temperature: float = 0.0,
                    fallback: Any = None,
                    timeout: Optional[float] = None,
                    total_budget: Optional[float] = None) -> Any:
        """Route a small cognitive sub-task to the worker pool.

        `task` is a short string label (e.g. ``"EXPAND_QUERY"``) used
        for logging; the actual prompt is in `payload`. `fallback` is
        returned if no worker pool exists or the call fails — callers
        should always pass a sensible default so they degrade silently.

        `timeout` overrides the default `_ROUTE_TIMEOUT_S` ceiling. The
        default is sized for sub-second routing chores (expand/classify);
        a judged call routed through here (VERIFY: 7–11s uncontended on
        the live worker) MUST pass its own budget or any node contention
        kills it at the routing ceiling (observed 2026-07-16: every
        finalize-burst verify died at exactly 12.0s).
        """
        # Worker pool absent → cheap fallback. We do NOT want a router
        # call to ever fall back to the foreground model: that would
        # inflate latency on the very tasks routing was meant to avoid.
        #
        # That intent was only ENFORCED for the no-pool case. When a pool
        # existed but every node FAILED, `_do_chat_completion` fell through to
        # the main upstream — carrying this call's short worker timeout — so
        # the 35B got a 6s budget and died (observed live 2026-07-11):
        #     worker node failed  Nova: ReadTimeout
        #     falling back to main upstream
        #     upstream fatal      ReadTimeout('')
        # `off_main_only=True` below now makes the failure path raise
        # OffMainNodeUnavailable, which we catch → the free fallback.
        if not getattr(self, "worker_clients", None):
            return fallback

        sized_payload = dict(payload)
        sized_payload.setdefault("temperature", temperature)
        sized_payload.setdefault("max_tokens", max_tokens)
        sized_payload["stream"] = False

        try:
            data = await self.chat_completion(
                sized_payload,
                use_worker=True,
                is_background=True,
                # A routing call is AWAITED ON THE USER'S CRITICAL PATH (query
                # expansion runs before the memory bus hydrates) and its
                # fallback is free — a legacy string concat. So a slow worker
                # must degrade FAST rather than stall the user. The old 15s
                # ceiling meant a thinking-happy worker added ~13.7s to every
                # request and still timed out (`Nova: ReadTimeout`); with
                # thinking disabled the same call takes 0.5s, so 6s is a
                # generous ceiling that bounds the damage if a node is sick.
                # Callers with genuinely slow tasks (VERIFY) override it.
                timeout=(timeout if timeout is not None else _ROUTE_TIMEOUT_S),
                # ⚠ THE ONE CALLER THAT ASKS TO BE QUEUE-DEADLINED, and it
                # asks by name. The permit wait happens BEFORE the HTTP
                # budget applies, so without this a 12s routing call could
                # sit 90s on the user's critical path — the fail-fast
                # contract defeated 7.5x over by a sibling budget nobody
                # sized against it. R2 inferred this from `timeout` for
                # EVERY caller; R4 lens B measured that the two budgets add
                # rather than cap, and that seven other callers were being
                # silently re-deadlined by a number that meant something
                # else. This is that fix, aimed only where it belongs.
                # ⚠ THE QUEUE BOUND IS ROUTE'S, NOT THE CALLER'S. A caller
                # raising `timeout` is asking for a longer GENERATION, not
                # for permission to sit in a queue longer — routing is on
                # the user's critical path either way. Deriving the queue
                # bound from `timeout` let a 45s VERIFY or a 60s
                # DISTILL_PATTERN queue for that long.
                slot_wait=_ROUTE_TIMEOUT_S,
                # ...and a TOTAL only for the default contract. route() is
                # awaited on the user's critical path, so its own 12s must
                # cover queueing AND the request — with only `slot_wait` the
                # two still added (a stated 12s measured 19.95s, R6 lens A).
                #
                # ⚠ But a caller that OVERRODE `timeout` did not ask for a
                # total. R6 forced one anyway, so a VERIFY that queued 20s
                # had its 45s generation clipped to 25s — and `verifier.py`'s
                # own comments still say the two budgets add (R7 lens A,
                # MAJOR-6). Such callers keep an unclipped request unless
                # they state `total_budget` themselves.
                total_budget=(total_budget if total_budget is not None
                              else (_ROUTE_TIMEOUT_S if timeout is None
                                    else None)),
                # NEVER re-run a routing sub-task on the main model.
                off_main_only=True,
                # A sick node costs `route()` its whole slot wait for a value
                # whose fallback is FREE (a string concat). It is the one
                # caller that should refuse a tripped node outright — see the
                # selectors' `require_healthy` note (R2 lens B, item 1).
                require_healthy=True,
                # The log label is the ACTUAL routed task ("decompose query",
                # "expand query", …). A hardcoded "query expansion" here made
                # a DECOMPOSE_QUERY timeout read as the anaphora expander and
                # sent the debugging down the wrong path (2026-07-12).
                task_label=_task_display_label(task),
            )
        except OffMainNodeUnavailable:
            logger.debug(f"route({task}): worker pool down — using fallback")
            return fallback
        except Exception as e:
            logger.debug(f"route({task}) worker call failed: {e}")
            return fallback

        try:
            # ⚠ DO NOT COUNT OR RECORD HERE. `route()` delegates to
            # `chat_completion()`, which already folds the response into the
            # usage ring and writes the recording — so doing it again made
            # every worker-routed call (query expansion, decompose, classify,
            # VERIFY) report exactly 2x its tokens into `Trajectory.tokens_in/
            # out` and into the OpenAI `usage` block returned to API clients,
            # and wrote every §4BG fixture TWICE (once kind="chat_completion",
            # once kind="route") into the mining corpus.
            #
            # The comment that used to sit here asserted the opposite —
            # "counted here too, not just in chat_completion" — and the test
            # that guarded it was an AST check that each of `route` /
            # `chat_completion` / `_do_stream_chat_completion` CONTAINS a
            # `_note_usage` call, which a redundant call satisfies. A
            # structural proxy for a semantic property, and it forced the
            # defect in (LLM review 2026-08-18, two independent lenses).
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return content if content else fallback
        except Exception:
            return fallback

    async def warm_up_workers(self) -> None:
        """Fire a tiny throwaway generation at every worker/critic node so its
        model weights + Metal/CUDA state and per-slot KV are hot BEFORE the
        first real (user-critical-path) call (2026-07-12).

        Why: measured on the live worker (Gemma 4 E4B on an M4 Mini) — a warm
        query-expansion call is ~1.9s, but the FIRST call after a (co-)restart
        pays model-load / prefill latency, blowing the short `route()` timeout
        and falling back for no reason. That cold miss happens on EVERY restart
        (and the operator restarts a lot while iterating), which is the bulk of
        the `Nova: ReadTimeout` lines. Paying it here, in the BACKGROUND at
        startup, moves the cost off the user's first request. Best-effort:
        never raises, and each node warms with one request per slot the node
        actually advertises, once per distinct node URL.
        """
        # ⚠ DE-DUPLICATE BY URL ACROSS POOLS. `--worker-nodes` and
        # `--critic-nodes` are byte-identical in the shipping topology, so
        # iterating both pools unconditionally warmed the SAME box twice —
        # six generations at boot against one node, three of them pure waste
        # competing with the three that mattered (R4 lens B, NEW-7).
        _seen_urls = set()
        for pool_attr, label in (("worker_clients", "worker"),
                                  ("critic_clients", "critic")):
            clients = getattr(self, pool_attr, None) or []
            for node in clients:
                _url = node.get("url") or ""
                if _url in _seen_urls:
                    logger.debug("warm_up: %s node %s already warmed via "
                                 "another pool — skipping", label, _url)
                    continue
                _seen_urls.add(_url)
                payload = {
                    "model": node.get("model", "default"),
                    "messages": [{"role": "user", "content": "ok"}],
                    "max_tokens": 1, "temperature": 0.0,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                # Fire the -np per-slot warmups CONCURRENTLY. Awaiting them in
                # series (the old loop) just re-grabs the one slot that freed
                # first, leaving the other slots cold — the opposite of the
                # stated "all -np slots get hot" intent. `off_main_only=True` is
                # essential: without it a DOWN node makes each warmup fall back
                # to the main 35B (3 pings × a dead node burned 3 main-slot
                # generations AND could evict the freshly-warmed main prefix
                # cache — the sibling keepalive_workers already passes it).
                # ⚠ THE NODE'S OWN `-np`, not a hardcoded 3. The docstring
                # below promised "a per-slot request so all `-np` slots get
                # hot" and then fired exactly three regardless — over-warming
                # a 1-slot box by 3x and under-warming a 4-slot one (R4 lens
                # B, NEW-7). `_node_capacity` already probes this number.
                _slots = max(1, await self._node_capacity(node))

                def _one():
                    return self.chat_completion(
                        dict(payload), use_worker=(label == "worker"),
                        use_critic=(label == "critic"),
                        is_background=True, timeout=30.0,
                        off_main_only=True, task_label="warmup")

                # ⚠ WARMUP FAILURES DO NOT REACH THE BREAKER (see the
                # `task_label != "warmup"` guard in the pool branches). R5
                # fixed this by narrowing the fan-out to `_slots - 1`, on
                # the reasoning that the default `_slots` of 3 equalled
                # `failure_threshold`. But Nova advertises FOUR slots, so
                # `_slots - 1` is 3 again — the fix restored the exact
                # collision for the only multi-slot node we actually run
                # (R6 lens A). It also never covered the live shape for
                # this box: a node that answers the first ping and then
                # fails the batch under Metal OOM.
                #
                # The real error was treating a boot ping as evidence. A
                # warmup against a node that is still coming up carries no
                # information the breaker should act on, and the cost of
                # acting on it is 60s of degraded query expansion at exactly
                # the moment warmup exists to prevent it. Probing with one
                # first is kept anyway — it stops us firing N doomed
                # requests at a node we already know is down.
                # ⚠ PROBE WITH ONE, THEN FAN OUT. Firing all `_slots` at once
                # is how boot warmup TRIPPED THE BREAKER IT EXISTS TO SERVE:
                # an unprobeable node yields the default 3, and 3 is exactly
                # `NodeCircuitBreaker.failure_threshold`, so a node still
                # coming up recorded three concurrent failures and opened its
                # own breaker — 60s of degraded query expansion on every
                # co-restart, at precisely the moment warmup is supposed to
                # prevent it (R5 lens A). One failure for a down node is
                # information; three is a self-inflicted outage.
                first = await asyncio.gather(_one(), return_exceptions=True)
                results = list(first)
                if not isinstance(first[0], BaseException) and _slots > 1:
                    results += list(await asyncio.gather(
                        *[_one() for _ in range(_slots - 1)],
                        return_exceptions=True))
                elif isinstance(first[0], BaseException):
                    # The node was down when we sized it, so the capacity we
                    # just cached is a guess with a 300s TTL. Boot is the
                    # WORST moment to freeze that guess — a node that comes
                    # up ten seconds later would stay mis-sized for five
                    # minutes. Let the next dispatch re-probe.
                    self._node_cap_retry_at.pop(node.get("url") or "", None)
                ok = sum(1 for r in results if not isinstance(r, BaseException))
                if ok:
                    pretty_log(
                        "Node Warmup",
                        f"{label} node {node.get('model')} pre-warmed "
                        f"({ok}/{len(results)} slots)",
                        icon=Icons.NODE_WORKER,
                    )
                else:
                    # Every warmup failed — log at debug (a dead node at boot is
                    # expected during a co-restart), do NOT claim "pre-warmed".
                    logger.debug("warm_up %s %s: all warmups failed (%s)",
                                 label, node.get("url"),
                                 next((r for r in results
                                       if isinstance(r, BaseException)), "?"))

    async def keepalive_workers(self, interval_s: float = 45.0) -> None:
        """Long-lived loop that keeps each worker/critic node's network path
        warm (2026-07-12). ``warm_up_workers`` only covers the FIRST request
        after boot; a Tailscale/WireGuard peer's direct path re-cools after an
        idle period, so a node that sits idle between requests — OR during a
        long tool-execution phase WITHIN a request — pays path-establishment
        again and trips the short ``route()`` timeout at BOTH ends of a request
        (observed: front-of-request query expansion AND the finalize route both
        ReadTimeout at exactly 5s, on a request whose worker sat idle ~105s
        during sandbox work). A tiny ping every ``interval_s`` keeps the path
        (and one slot) hot so ``route()`` stays on its ~0.6-1.9s warm path.

        Best-effort and self-contained: a per-node failure is logged at debug
        and never escapes the loop; a task cancel (shutdown) ends it cleanly.
        No worker/critic pool ⇒ returns immediately (no idle spin, no main-node
        traffic — this only ever touches off-main nodes). Interval is tunable
        via ``GHOST_WORKER_KEEPALIVE_S`` (≤0 disables; wired in main.py)."""
        if not (getattr(self, "worker_clients", None)
                or getattr(self, "critic_clients", None)):
            return
        # Heartbeats log TRANSITIONS, not ticks (a ping line every 45s was
        # pure spam in the live stream): one WARNING when a node stops
        # answering, one line when it comes back — silence in between.
        down: set = set()
        while True:
            try:
                await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                return
            # ⚠ ONE PING PER PHYSICAL NODE. worker and critic are the same
            # box in the shipping topology, so this pinged Nova TWICE every
            # 45s — and both pings bypass the concurrency gate
            # (`_slot_wait = None` for keepalive), so it was two un-gated
            # requests per interval against a node of unknown `-np`
            # (R4 lens B, NEW-7/NEW-8).
            _pinged = set()
            for pool_attr, label in (("worker_clients", "worker"),
                                     ("critic_clients", "critic")):
                for node in getattr(self, pool_attr, None) or []:
                    url = node.get("url")
                    if url in _pinged:
                        continue
                    _pinged.add(url)
                    try:
                        payload = {
                            "model": node.get("model", "default"),
                            "messages": [{"role": "user", "content": "ok"}],
                            "max_tokens": 1, "temperature": 0.0,
                            "stream": False,
                            "chat_template_kwargs": {"enable_thinking": False},
                        }
                        await self.chat_completion(
                            payload, use_worker=(label == "worker"),
                            use_critic=(label == "critic"),
                            is_background=True, timeout=30.0,
                            # A failed ping must NEVER burn the single main
                            # slot as a "fallback" — a max_tokens=1 hit on the
                            # 35B every 45s for as long as a node stays down.
                            off_main_only=True,
                            task_label="keepalive",
                        )
                    except Exception as e:  # noqa: BLE001 — best-effort
                        logger.debug("keepalive %s %s failed: %s",
                                     label, url, e)
                        if url not in down:
                            down.add(url)
                            pretty_log(
                                "Node Keepalive",
                                f"{label} node {node.get('model')} stopped "
                                f"answering ({type(e).__name__}) — pings "
                                f"continue silently; recovery will be logged",
                                level="WARNING", icon=Icons.WARN,
                            )
                    else:
                        if url in down:
                            down.discard(url)
                            pretty_log(
                                "Node Keepalive",
                                f"{label} node {node.get('model')} recovered",
                                icon=Icons.NODE_WORKER,
                            )

    async def close(self):
        """Close every client. ⚠ A raise anywhere used to leak the rest:
        `self.http_client.aclose()` was the first statement, so one bad
        upstream socket left all six node pools open (R4 lens B, NEW-10).
        Shutdown is exactly where partial completion is worthless."""
        import contextlib
        with contextlib.suppress(Exception):
            await self.http_client.aclose()
        for attr in ('swarm_clients', 'worker_clients', 'vision_clients',
                     'coding_clients', 'image_gen_clients', 'critic_clients'):
            for node in getattr(self, attr, []) or []:
                client = node.get("client") if isinstance(node, dict) else None
                if client is None:
                    continue
                with contextlib.suppress(Exception):
                    await client.aclose()
        # The gate's state is per-client-instance; leaving it populated makes
        # a reused object account for permits against clients that are gone.
        self._node_slots.clear()
        self._node_slot_caps.clear()
        self._node_slot_built_cap.clear()
        self._node_slot_locks.clear()
        self._node_cap_retry_at.clear()

    def get_swarm_node(self, target_model: Optional[str] = None, *,
                              require_healthy: bool = False) -> Optional[Dict[str, Any]]:
        if not getattr(self, 'swarm_clients', []):
            return None

        # Consult the circuit breaker on the target-model match too — the
        # ONLY thing get_swarm_node was missing vs the vision/worker/coding
        # selectors. Without it a dead swarm node stayed in round-robin
        # rotation forever, eating a full 300 s timeout on every cycle. (The
        # round-robin fallback when a target_model isn't matched is deliberate
        # best-effort — same as the sibling selectors.)
        if target_model:
            target_lower = target_model.lower()
            for node in self.swarm_clients:
                if target_lower in node["model"].lower() and self.circuit_breaker.is_available(node["url"]):
                    return node

        for _ in range(len(self.swarm_clients)):
            node = self.swarm_clients[self._swarm_index]
            self._swarm_index = (self._swarm_index + 1) % len(self.swarm_clients)
            if self.circuit_breaker.is_available(node["url"]):
                return node
        # All nodes tripped. See the sibling selectors: returning a sick node
        # anyway is deliberate for callers whose alternative is worse, and
        # `require_healthy=True` is the opt-in for callers with a free
        # fallback (LLM review R2 lens B, item 1).
        if require_healthy:
            return None
        return self.swarm_clients[self._swarm_index]

    def get_vision_node(self, target_model: Optional[str] = None, *,
                               require_healthy: bool = False) -> Optional[Dict[str, Any]]:
        vision_clients = getattr(self, 'vision_clients', [])
        if not vision_clients:
            return None

        if target_model:
            target_lower = target_model.lower()
            for node in vision_clients:
                if target_lower in node["model"].lower() and self.circuit_breaker.is_available(node["url"]):
                    return node

        if not hasattr(self, '_vision_index'):
            self._vision_index = 0

        # Round-robin with circuit-breaker filtering. Without this a dead
        # vision node stayed in rotation forever, eating 600 s timeouts on
        # every request before the agent fell back.
        for _ in range(len(vision_clients)):
            node = vision_clients[self._vision_index]
            self._vision_index = (self._vision_index + 1) % len(vision_clients)
            if self.circuit_breaker.is_available(node["url"]):
                return node
        # All nodes tripped — return first anyway (the call will fail and
        # the breaker cooldown will extend).
        # ⚠ WITH ONE NODE PER POOL — the shipping topology — this line is
        # what makes the breaker inert: `is_available()` can say NO for
        # every node and the request still goes out. That is DELIBERATE
        # for callers whose alternative is worse than a sick node:
        # `keepalive` is the most FREQUENT recovery detector, and
        # warmup/verify have expensive fallbacks. (It is not the only
        # one, and refusal is not permanent: `is_available` promotes
        # open -> half_open after the cooldown and then returns True to
        # EVERY caller, `require_healthy` included, and any success
        # re-closes the breaker. R4 lens B measured this — an earlier
        # version of this comment claimed "a pool that refuses can
        # never observe healing", which is false and was repeated
        # verbatim at five sites and in a test docstring. The real cost
        # of refusing is a recovery delayed by at most one cooldown.) Callers with a FREE fallback pass
        # `require_healthy=True` and handle None — they get the
        # fail-fast; nobody else pays for it (LLM review R2 lens B,
        # item 1: refuse-and-degrade globally would turn a sick node
        # into a main-model dogpile).
        if require_healthy:
            return None
        return vision_clients[0]

    def get_worker_node(self, target_model: Optional[str] = None, *,
                               require_healthy: bool = False) -> Optional[Dict[str, Any]]:
        worker_clients = getattr(self, 'worker_clients', [])
        if not worker_clients:
            return None

        if target_model:
            target_lower = target_model.lower()
            for node in worker_clients:
                if target_lower in node["model"].lower() and self.circuit_breaker.is_available(node["url"]):
                    return node

        if not hasattr(self, '_worker_index'):
            self._worker_index = 0

        # Round-robin with circuit breaker filtering
        for _ in range(len(worker_clients)):
            node = worker_clients[self._worker_index]
            self._worker_index = (self._worker_index + 1) % len(worker_clients)
            if self.circuit_breaker.is_available(node["url"]):
                return node
        # All nodes tripped — return first anyway (will fail and extend cooldown)
        # ⚠ WITH ONE NODE PER POOL — the shipping topology — this line is
        # what makes the breaker inert: `is_available()` can say NO for
        # every node and the request still goes out. That is DELIBERATE
        # for callers whose alternative is worse than a sick node:
        # `keepalive` is the most FREQUENT recovery detector, and
        # warmup/verify have expensive fallbacks. (It is not the only
        # one, and refusal is not permanent: `is_available` promotes
        # open -> half_open after the cooldown and then returns True to
        # EVERY caller, `require_healthy` included, and any success
        # re-closes the breaker. R4 lens B measured this — an earlier
        # version of this comment claimed "a pool that refuses can
        # never observe healing", which is false and was repeated
        # verbatim at five sites and in a test docstring. The real cost
        # of refusing is a recovery delayed by at most one cooldown.) Callers with a FREE fallback pass
        # `require_healthy=True` and handle None — they get the
        # fail-fast; nobody else pays for it (LLM review R2 lens B,
        # item 1: refuse-and-degrade globally would turn a sick node
        # into a main-model dogpile).
        if require_healthy:
            return None
        return worker_clients[0]

    def get_coding_node(self, target_model: Optional[str] = None, *,
                               require_healthy: bool = False) -> Optional[Dict[str, Any]]:
        coding_clients = getattr(self, 'coding_clients', [])
        if not coding_clients:
            return None

        if target_model:
            target_lower = target_model.lower()
            for node in coding_clients:
                if target_lower in node["model"].lower() and self.circuit_breaker.is_available(node["url"]):
                    return node

        if not hasattr(self, '_coding_index'):
            self._coding_index = 0

        # Round-robin with circuit-breaker filtering — same rationale as
        # `get_vision_node`. Coding requests are slow; hitting a dead node
        # was burning the user's request budget on guaranteed timeouts.
        for _ in range(len(coding_clients)):
            node = coding_clients[self._coding_index]
            self._coding_index = (self._coding_index + 1) % len(coding_clients)
            if self.circuit_breaker.is_available(node["url"]):
                return node
        # ⚠ WITH ONE NODE PER POOL — the shipping topology — this line is
        # what makes the breaker inert: `is_available()` can say NO for
        # every node and the request still goes out. That is DELIBERATE
        # for callers whose alternative is worse than a sick node:
        # `keepalive` is the most FREQUENT recovery detector, and
        # warmup/verify have expensive fallbacks. (It is not the only
        # one, and refusal is not permanent: `is_available` promotes
        # open -> half_open after the cooldown and then returns True to
        # EVERY caller, `require_healthy` included, and any success
        # re-closes the breaker. R4 lens B measured this — an earlier
        # version of this comment claimed "a pool that refuses can
        # never observe healing", which is false and was repeated
        # verbatim at five sites and in a test docstring. The real cost
        # of refusing is a recovery delayed by at most one cooldown.) Callers with a FREE fallback pass
        # `require_healthy=True` and handle None — they get the
        # fail-fast; nobody else pays for it (LLM review R2 lens B,
        # item 1: refuse-and-degrade globally would turn a sick node
        # into a main-model dogpile).
        if require_healthy:
            return None
        return coding_clients[0]

    def get_critic_node(self, target_model: Optional[str] = None, *,
                               require_healthy: bool = False) -> Optional[Dict[str, Any]]:
        """Round-robin pick from the critic pool, skipping tripped nodes.

        Mirrors `get_coding_node`. Returns None when no critic pool is
        configured so callers fall back to their existing path (worker
        route → foreground) without special-casing.
        """
        critic_clients = getattr(self, 'critic_clients', [])
        if not critic_clients:
            return None

        if target_model:
            target_lower = target_model.lower()
            for node in critic_clients:
                if target_lower in node["model"].lower() and self.circuit_breaker.is_available(node["url"]):
                    return node

        if not hasattr(self, '_critic_index'):
            self._critic_index = 0

        for _ in range(len(critic_clients)):
            node = critic_clients[self._critic_index]
            self._critic_index = (self._critic_index + 1) % len(critic_clients)
            if self.circuit_breaker.is_available(node["url"]):
                return node
        # ⚠ WITH ONE NODE PER POOL — the shipping topology — this line is
        # what makes the breaker inert: `is_available()` can say NO for
        # every node and the request still goes out. That is DELIBERATE
        # for callers whose alternative is worse than a sick node:
        # `keepalive` is the most FREQUENT recovery detector, and
        # warmup/verify have expensive fallbacks. (It is not the only
        # one, and refusal is not permanent: `is_available` promotes
        # open -> half_open after the cooldown and then returns True to
        # EVERY caller, `require_healthy` included, and any success
        # re-closes the breaker. R4 lens B measured this — an earlier
        # version of this comment claimed "a pool that refuses can
        # never observe healing", which is false and was repeated
        # verbatim at five sites and in a test docstring. The real cost
        # of refusing is a recovery delayed by at most one cooldown.) Callers with a FREE fallback pass
        # `require_healthy=True` and handle None — they get the
        # fail-fast; nobody else pays for it (LLM review R2 lens B,
        # item 1: refuse-and-degrade globally would turn a sick node
        # into a main-model dogpile).
        if require_healthy:
            return None
        return critic_clients[0]

    def get_image_gen_node(self, target_model: Optional[str] = None, *,
                                  require_healthy: bool = False) -> Optional[Dict[str, Any]]:
        image_gen_clients = getattr(self, 'image_gen_clients', [])
        if not image_gen_clients:
            return None

        if target_model:
            target_lower = target_model.lower()
            for node in image_gen_clients:
                if target_lower in node["model"].lower() and self.circuit_breaker.is_available(node["url"]):
                    return node

        if not hasattr(self, '_image_gen_index'):
            self._image_gen_index = 0

        for _ in range(len(image_gen_clients)):
            node = image_gen_clients[self._image_gen_index]
            self._image_gen_index = (self._image_gen_index + 1) % len(image_gen_clients)
            if self.circuit_breaker.is_available(node["url"]):
                return node
        # ⚠ WITH ONE NODE PER POOL — the shipping topology — this line is
        # what makes the breaker inert: `is_available()` can say NO for
        # every node and the request still goes out. That is DELIBERATE
        # for callers whose alternative is worse than a sick node:
        # `keepalive` is the most FREQUENT recovery detector, and
        # warmup/verify have expensive fallbacks. (It is not the only
        # one, and refusal is not permanent: `is_available` promotes
        # open -> half_open after the cooldown and then returns True to
        # EVERY caller, `require_healthy` included, and any success
        # re-closes the breaker. R4 lens B measured this — an earlier
        # version of this comment claimed "a pool that refuses can
        # never observe healing", which is false and was repeated
        # verbatim at five sites and in a test docstring. The real cost
        # of refusing is a recovery delayed by at most one cooldown.) Callers with a FREE fallback pass
        # `require_healthy=True` and handle None — they get the
        # fail-fast; nobody else pays for it (LLM review R2 lens B,
        # item 1: refuse-and-degrade globally would turn a sick node
        # into a main-model dogpile).
        if require_healthy:
            return None
        return image_gen_clients[0]

    async def generate_image(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates an image by posting to an image generation node.
        """
        image_gen_clients = getattr(self, 'image_gen_clients', [])
        if not image_gen_clients:
            raise Exception("No image generation nodes available")

        node = self.get_image_gen_node(payload.get("model"))
        if not node:
            raise Exception("Could not find a valid image generation node")

        # Image generation is slow by nature; give the permit wait room, but
        # keep it bounded like every other node path.
        # ⚠ A DEADLINE ACROSS ALL THREE ATTEMPTS, not a fresh budget each
        # time. `for attempt in range(3)` each spending the full permit wait
        # meant 3 x 90s plus backoffs — roughly 273 SECONDS of pure queueing
        # for one image request (R5 lens B). Same defect as the pool loop in
        # `_do_chat_completion`, one function over; fixed the same way.
        _img_budget = _env_positive("GHOST_NODE_SLOT_WAIT_S", 90.0)
        _img_deadline = time.monotonic() + _img_budget

        def _img_slot_wait_now() -> float:
            # ⚠ `_MIN_ACQUIRE`, NOT 0.0 — and passed as a CALLABLE below.
            # This function never received either half of the R6 `_node_slot`
            # rework, 200 lines above it. Both defects were live here:
            #   * `max(0.0, …)` handed attempts 2 and 3 a hard 0.0 once
            #     attempt 1 had consumed the budget, and
            #     `wait_for(sem.acquire(), 0.0)` REFUSES even a completely
            #     free semaphore — measured a node that became free at t=3.0
            #     still failing at 5.01s. That is the exact defect
            #     `_MIN_ACQUIRE` was created for, one function away from the
            #     constant.
            #   * as an argument expression it was evaluated BEFORE
            #     `_node_slot` ran the `/props` probe, so the probe was spent
            #     outside the budget (stated 4.00s, elapsed 5.01s). The
            #     image-gen node is a Jetson that never answers `/props`, so
            #     that is the live path (R7 lens A).
            return max(_MIN_ACQUIRE, _img_deadline - time.monotonic())

        for attempt in range(3):
            try:
                pretty_log("Image Compute", f"Routing to Image Node ({node['model']})", level="INFO", icon=Icons.IMAGE_GEN)
                async with self._node_slot(node, wait_timeout=_img_slot_wait_now):
                    resp = await node["client"].post("/v1/images/generations", json=payload)
                resp.raise_for_status()
                # Record breaker success/failure like every other node path —
                # without it the image-gen breaker never trips, so
                # get_image_gen_node's is_available() filtering was dead code
                # and a dead image node stayed selected across every request.
                if node.get("url"):
                    self.circuit_breaker.record_success(node["url"])
                return resp.json()
            except Exception as e:
                # ⚠ ONLY a NODE fault trips the breaker. This recorded every
                # exception, so one caller-side 400 ("bad prompt") tripped a
                # healthy image node in three attempts — exactly the confusion
                # `_is_node_fault` exists to prevent, applied at 4 of 5 sites
                # and missed here (R2 lens B, item 5).
                if node.get("url") and _is_node_fault(e):
                    self.circuit_breaker.record_failure(node["url"])
                if attempt < 2:
                    # A 503 is the node WARMING UP (it binds its port ~1-2s
                    # after a restart but loads the model for ~5-10s more) or
                    # GPU-busy — a fixed 1-2s backoff expired before either
                    # cleared. Wait long enough for the warmup to finish.
                    _is_503 = getattr(getattr(e, "response", None),
                                      "status_code", None) == 503
                    pretty_log("Image Node Retry", f"Attempt {attempt+1} failed: {type(e).__name__}: {e}", level="WARNING", icon=Icons.WARN)
                    await asyncio.sleep(8.0 if _is_503 else 2 ** attempt)
                    # Try to get next node if possible
                    node = self.get_image_gen_node()
                else:
                    raise Exception(f"Image generation failed after 3 attempts: {_err_text(e)}")

    async def _node_capacity(self, node: Dict[str, Any]) -> int:
        """How many requests this node can genuinely serve at once.

        Read from the node's OWN `/props` (`total_slots`) rather than
        configured by hand: the number that matters is llama-server's `-np`,
        and a hand-set copy drifts the moment the operator restarts a node with
        different flags.

        ⚠ A SUCCESSFUL probe is cached forever; a FAILED one is cached only
        briefly. Both used to be permanent, so a node that happened to be down
        at first touch — a co-restart, which happens constantly while
        iterating — was pinned at the default 3 for the entire process
        lifetime. On a `-np 1` node that is 3x over-subscription applied by
        the gate whose only job is to prevent over-subscription: a gate whose
        failure mode is "no gate" (R4 lens B, NEW-6).
        """
        url = node.get("url") or ""
        cap = self._node_slot_caps.get(url)
        if cap is not None:
            return cap
        retry_at = self._node_cap_retry_at.get(url)
        if retry_at is not None and time.monotonic() < retry_at:
            return self._node_slot_default
        cap = self._node_slot_default
        try:
            client = node.get("client")
            if client is not None:
                r = await client.get("/props", timeout=5.0)
                r.raise_for_status()
                data = r.json() or {}
                raw = data.get("total_slots") or data.get("n_parallel")
                if not (isinstance(raw, int) and raw > 0):
                    # Reached the node but learned nothing — a backend with no
                    # `total_slots`. Retry later rather than pinning the
                    # default forever.
                    self._node_cap_retry_at[url] = (
                        time.monotonic() + _NODE_CAP_RETRY_S)
                    return cap
                cap = raw
                pretty_log("Node Capacity",
                           f"{node.get('model')} advertises {cap} slot(s) "
                           f"— per-node gate sized to match",
                           level="INFO", icon=Icons.NODE_WORKER)
            else:
                # No client to probe — nothing will ever change that.
                self._node_slot_caps[url] = cap
                return cap
        except Exception as e:                                  # noqa: BLE001
            logger.debug("node capacity probe failed for %s (%s) — using "
                         "default %d, re-probing in %ds", url,
                         type(e).__name__, cap, int(_NODE_CAP_RETRY_S))
            self._node_cap_retry_at[url] = time.monotonic() + _NODE_CAP_RETRY_S
            return cap
        self._node_slot_caps[url] = cap
        self._node_cap_retry_at.pop(url, None)
        return cap

    async def _absorb_permits(self, sem: "asyncio.Semaphore",
                              n: int, url: str) -> None:
        """Permanently remove ``n`` permits from ``sem`` (a capacity shrink).

        Acquired and never released — the point is that the node's true `-np`
        is lower than we guessed, so those permits must stop existing. Waiting
        here is correct and unbounded on purpose: the permits become
        unavailable as soon as their current holders finish, and until then
        the node is no more loaded than it already was.
        """
        try:
            for _ in range(max(0, n)):
                await sem.acquire()
        except asyncio.CancelledError:                        # shutdown
            raise
        except Exception as e:                                # noqa: BLE001
            logger.debug("permit absorb failed for %s: %s", url, _err_text(e))

    @asynccontextmanager
    async def _node_slot(self, node: Dict[str, Any],
                         wait_timeout: Optional[float] = None):
        """Hold one of ``node``'s slots for the duration of a request.

        Raises :class:`NodeSaturated` if no permit arrives within
        ``wait_timeout`` — the caller then knows the node was never asked, so
        nothing about this failure says anything about the node's health.

        ``wait_timeout=None`` BYPASSES the gate (health probes only). It
        deliberately does not mean "wait forever": an unbounded wait here would
        park callers on a wedged node until their own timeouts fired, which is
        the failure mode this gate replaces, not one to reintroduce.
        """
        if wait_timeout is None:
            yield
            return
        url = node.get("url") or ""
        # ⚠ A CALLABLE WAIT IS RESOLVED AFTER THE PROBE, not before. R5
        # claimed "a wall-clock deadline absorbs the /props probe for free";
        # false — `wait_timeout=_permit_wait(n)` is an ARGUMENT EXPRESSION,
        # evaluated before this function is even entered, so a 5s probe was
        # still spent entirely outside the attempt's budget (R6 lens A
        # measured a stated 12s budget taking 14.01s). Callers that pass a
        # callable get it evaluated below, once the capacity is known.
        _wait_fn = wait_timeout if callable(wait_timeout) else None
        sem = self._node_slots.get(url)
        # ⚠ FAST PATH ONLY ONCE THE CAPACITY IS AUTHORITATIVE. `_node_slots`
        # is written in exactly one place and never resized, so a semaphore
        # built while the node was unreachable — from the provisional default
        # — used to be permanent. That made the re-probe TTL added alongside
        # this comment INERT: `_node_capacity` was simply never called again.
        # A fix that cannot take effect is the defect class this review keeps
        # finding, so the two halves ship together.
        #
        # `url in self._node_slot_caps` means a probe SUCCEEDED; a failed one
        # records only a retry deadline. So the slow path keeps running —
        # cheaply, without probing, until the deadline elapses — until we
        # have a real number, then never again.
        if sem is None or url not in self._node_slot_caps:
            # ⚠ PROBE OUTSIDE ANY LOCK. `_node_capacity` issues a `/props` GET
            # with a 5s timeout; doing that while holding `_node_slots_lock` —
            # ONE lock shared by every node — blocked dispatch to every OTHER
            # node for the duration. R4 made that worse rather than better: it
            # added a 300s re-probe TTL, turning a once-per-process 5s stall
            # into a recurring one, forever, for any node that is unreachable
            # or does not serve `/props`. Measured: a healthy Nova dispatch
            # stalled 5.0s behind the image-gen Jetson's probe, which is not
            # a llama-server and never answers it (R5 lens A).
            cap = await self._node_capacity(node)
            lock = self._node_slot_locks.get(url)
            if lock is None:
                # `setdefault` is atomic here — no await between the miss and
                # the insert, so two coroutines cannot install rival locks.
                lock = self._node_slot_locks.setdefault(url, asyncio.Lock())
            async with lock:
                # ⚠ RE-READ THE AUTHORITY. Probing outside the lock (R5's
                # correct fix for the cross-node stall) means two coroutines
                # racing first touch apply their caps in probe-COMPLETION
                # order, not freshness order: a slow FAILING probe (default 3)
                # could land after a fast SUCCEEDING one (real 1) and win. And
                # because a successful probe populates `_node_slot_caps`, the
                # fast path then locks that wrong value in FOREVER — measured
                # a gate stuck at 3 against a `total_slots=1` node, which
                # live is `--visual-nodes`, i.e. the main 35B's single slot
                # (R6 lens A). A succeeded probe always outranks a default.
                cap = self._node_slot_caps.get(url, cap)
                sem = self._node_slots.get(url)          # re-check under lock
                if sem is None:
                    sem = asyncio.Semaphore(cap)
                    self._node_slots[url] = sem
                    self._node_slot_built_cap[url] = cap
                else:
                    built = self._node_slot_built_cap.get(url, cap)
                    if cap != built:
                        # ⚠ RESIZE, NEVER REPLACE. R4 swapped in a new
                        # Semaphore and claimed the over-subscription window
                        # was "the duration of those requests". False in the
                        # direction that matters: QUEUED WAITERS stay parked
                        # on the old object and keep being admitted through
                        # it, so the window is queue_depth / old_cap x
                        # request_duration — minutes under a fan-out, not
                        # "brief". Measured 4 concurrent in flight against a
                        # total_slots=1 node (R5 lens A).
                        #
                        # Shrinking by permanently acquiring the difference
                        # keeps ONE semaphore, so every waiter — parked or
                        # future — is bounded by the corrected capacity at
                        # every instant. Growing just releases the difference.
                        pretty_log(
                            "Node Capacity",
                            f"{node.get('model') or url}: gate resized "
                            f"{built} -> {cap} (capacity learned after a "
                            f"failed first probe)",
                            level="INFO", icon=Icons.NODE_WORKER)
                        self._node_slot_built_cap[url] = cap
                        if cap > built:
                            for _ in range(cap - built):
                                sem.release()
                        else:
                            # Absorbed in the background: acquiring the
                            # surplus may have to wait for in-flight requests
                            # to finish, and blocking this dispatch on that
                            # would be a self-inflicted stall.
                            asyncio.ensure_future(
                                self._absorb_permits(sem, built - cap, url))
        if _wait_fn is not None:
            wait_timeout = _wait_fn()
            if wait_timeout is None:
                # A callable that resolves to None means "no gate" — the
                # same contract as passing None directly. Without this an
                # `asyncio.wait_for(..., timeout=None)` waits forever, which
                # is the unbounded park this gate exists to replace.
                yield
                return
        try:
            await asyncio.wait_for(sem.acquire(), timeout=wait_timeout)
        except asyncio.TimeoutError:
            # ⚠ Report the capacity the gate is ACTUALLY enforcing.
            # `_node_slot_caps` holds only probes that SUCCEEDED, so after a
            # failed probe this printed "cap None" while the gate was really
            # sized 3 — an operator debugging saturation was shown a number
            # that does not exist (R5 lens A). `:.2f` because sub-second
            # waits rendered as "0s", which reads like a disabled gate.
            raise NodeSaturated(
                f"no free slot on {node.get('model') or url} within "
                f"{wait_timeout:.2f}s "
                f"(cap {self._node_slot_built_cap.get(url, '?')})")
        try:
            yield
        finally:
            sem.release()

    # ── STALL ATTRIBUTION HELPERS (§4BV R7) ─────────────────────────────
    # ⚠ EVERY ONE OF THESE IS TOTAL. They are reached from the streaming
    # path, and several suites build a client with `LLMClient.__new__(...)`
    # and never run `__init__` — tests/test_stream_idle_timeout.py sets
    # exactly two attributes and then drives `_do_stream_chat_completion`
    # directly. Unguarded instance state reached from the stream path turns
    # a missing attribute into a DEAD USER TURN. This is pure diagnostics;
    # it must never be able to break a stream.
    #
    # (The `_BackgroundOnlyLLM` shims in subagent.py / dream.py are NOT a
    # reason: they delegate `stream_chat_completion` to the inner REAL
    # client, so `self` here is never the shim. Checked, not assumed.)
    def _inflight_map(self) -> Dict[str, int]:
        m = getattr(self, "_inflight_by_url", None)
        if m is None:
            m = {}
            self._inflight_by_url = m
        return m

    def _inflight_inc(self, url: str) -> None:
        try:
            m = self._inflight_map()
            m[url] = m.get(url, 0) + 1
        except Exception:  # noqa: BLE001 — diagnostics never break a stream
            pass

    def _inflight_dec(self, url: str) -> None:
        try:
            m = self._inflight_map()
            n = m.get(url, 0) - 1
            if n > 0:
                m[url] = n
            else:
                m.pop(url, None)
        except Exception:  # noqa: BLE001
            pass

    def _own_inflight(self, url: str) -> int:
        """How many requests THIS PROCESS has on the wire to ``url``.

        Streams are counted explicitly. The non-streaming MAIN path and
        `get_embeddings` are both serialised by `_main_node_lock`, so a held
        lock is worth exactly one more in-flight request against the main URL
        and needs no second counter — holding it is what makes that true, so
        if it is ever dropped from those paths this undercounts.

        ⚠ IT CAN ONLY UNDERCOUNT, NEVER OVERCOUNT. Non-streaming POOL traffic
        is not counted here at all. That is the safe direction: the verdict
        below only ACCUSES US when the count provably exceeds a probed
        capacity, so undercounting can lose an accusation but can never
        manufacture one."""
        try:
            n = self._inflight_map().get(url, 0)
        except Exception:  # noqa: BLE001
            return 0
        try:
            if url and url == str(getattr(self, "upstream_url", "") or "") \
                    and self._main_node_lock.locked():
                n += 1
        except Exception:  # noqa: BLE001 — attribution must never raise
            pass
        return n

    def _known_slots(self, url: str):
        """The node's advertised `total_slots`, or None if never probed.

        ⚠ ONLY A SUCCESSFUL `/props` PROBE LANDS IN `_node_slot_caps` — a
        failed one records a retry deadline and nothing else — so this is
        never a guess, and "unknown" is never confused with "1". On the
        shipping topology `--visual-nodes` is byte-identical to the upstream
        URL, so main's capacity (1) is populated for free by the first vision
        call: no extra probe, and no verdict at all until a real number
        exists."""
        try:
            return self._node_slot_caps.get(url)
        except Exception:  # noqa: BLE001
            return None

    async def _do_chat_completion(self, payload: Dict[str, Any], use_swarm: bool = False, use_worker: bool = False, use_vision: bool = False, use_coding: bool = False, use_critic: bool = False, timeout: Optional[float] = None, off_main_only: bool = False, task_label: str = "", require_healthy: bool = False, slot_wait: Optional[float] = None, total_budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Sends a chat completion request to the upstream LLM with robust retry logic.
        """
        # How long a caller will queue for a node permit before giving up.
        # Generous relative to a request, because WAITING is the desired
        # behaviour — the alternative it replaces is not "go faster", it is
        # "flood the node, time out anyway, and take it out of rotation for
        # 60s". Bounded so a wedged node cannot park callers forever.
        _slot_wait = _env_positive("GHOST_NODE_SLOT_WAIT_S", 90.0)
        if slot_wait is not None and task_label != "keepalive":
            # ⚠ EXPLICIT, NOT INFERRED FROM `timeout`. R2 derived the permit
            # budget as `min(_slot_wait, timeout)`, arguing that a caller
            # saying "worthless to me after N seconds" had thereby stated how
            # long it would queue. R4 lens B measured that claim and it is
            # false in both directions:
            #   * the two budgets ADD (timeout=2s, permit freed at 1.9s, HTTP
            #     2.0s -> 3.91s total), so the "total" it described never
            #     existed at any pool size;
            #   * the wait is re-spent PER NODE, because it is computed once
            #     before the retry loop — a 3-node pool spent 3.01s against a
            #     1.0s stated budget, and the overrun grows linearly as nodes
            #     are added.
            # It also silently re-deadlined eight callers to serve one. The
            # mechanism is right for `route()` — whose fallback is a string
            # concat on the user's critical path — so `route()` now asks for
            # it by name, and nobody else is quietly re-timed. The env var
            # stays the CEILING, not the value.
            #
            # ⚠ THE FLOOR IS ON THE TOTAL, and the reserve comes out of the
            # same budget. Flooring at `_MIN_SLOT_WAIT` alone leaves only
            # `5.0 - _MIN_HTTP_FLOOR` = 2.0s of real queueing, which quietly
            # guts the anti-disable guarantee this floor exists to provide
            # (R5 lens B).
            _slot_wait = min(_slot_wait, max(float(slot_wait),
                                             _MIN_SLOT_WAIT + _MIN_HTTP_FLOOR))
        # ⚠ THE HEALTH PROBE MUST NOT QUEUE BEHIND THE TRAFFIC IT IS WATCHING.
        # `keepalive_workers()` is what prints "node X stopped answering"; if
        # its ping waits on the same permits as a research fan-out, a BUSY node
        # reports as a DEAD one — the identical false alarm this change exists
        # to remove, reintroduced one layer down. It skips the gate entirely:
        # one extra in-flight request against a 4-slot node is a rounding
        # error, and an unstarvable probe is the whole point of a probe.
        if task_label == "keepalive":
            _slot_wait = None
        # ⚠ TWO BUDGETS, DELIBERATELY SEPARATE. R5 collapsed them into one
        # and that was a live regression: `_http_cap` clipped EVERY pool
        # caller's generation budget to the queue ceiling, so the verifier's
        # 120s critic budget silently became 30s and dream's 180s became 90s.
        # Measured against the live log (n=39 verdicts): median 24.4s, p90
        # 56.7s, and 28.2% over 30s — so about a quarter of all verdicts
        # would now time out AND be charged to the node as a fault, because
        # a ReadTimeout is a node fault. `GHOST_NODE_SLOT_WAIT_S` documents
        # itself as "how long a caller will queue"; it must not also be the
        # maximum length of a generation (R6 lens A, CRITICAL-2).
        #
        #   _queue_deadline — how long we may spend WAITING for permits,
        #     shared across the whole pool loop. This is R5's genuine fix:
        #     the wait used to be re-spent in full by every node (12/24/36s
        #     at 1/2/3 nodes against a stated 12s).
        #   _total_deadline — queueing AND the request, and ONLY for callers
        #     that opt in with `total_budget`. Those are the callers with an
        #     outer deadline of their own: route() on the critical path, and
        #     the per-URL research distillations, whose `asyncio.wait_for`
        #     cancels fetch+distill and LOSES THE URL if we overrun it.
        _queue_deadline = (None if _slot_wait is None
                           else time.monotonic() + _slot_wait)
        _total_deadline = (None if not total_budget
                           else time.monotonic() + float(total_budget))

        def _permit_wait(untried: int = 1) -> Optional[float]:
            """Share what remains across the attempts still to come.

            ⚠ `untried` INCLUDES the current attempt. It is computed after
            the node is appended to `tried_nodes`, so the caller adds one
            back; without that the final node of every pool was handed
            exactly 0.0s and the gate refused it — even completely idle,
            with every permit free. That is the precise defect the sharing
            exists to prevent, moved from node 2 to node N (R6, both lenses).
            """
            if _queue_deadline is None:
                return None
            remaining = _queue_deadline - time.monotonic()
            if _total_deadline is not None:
                # Never queue into the reserve the request itself needs.
                remaining = min(
                    remaining,
                    _total_deadline - time.monotonic() - _MIN_HTTP_FLOOR)
            return max(_MIN_ACQUIRE, remaining / max(1, untried))

        def _gate_wait(untried: int):
            """The `wait_timeout` to hand the gate for this attempt.

            ⚠ RETURNS A LITERAL `None` FOR KEEPALIVE, not a callable that
            returns None. `_node_slot`'s gate bypass is `if wait_timeout is
            None`, and it is checked BEFORE the callable is resolved — a
            `functools.partial` is truthy, so passing one unconditionally
            defeated the bypass and left the health probe waiting FOREVER on
            a saturated node. That wedges the very detector that reports
            nodes as recovered (caught by the keepalive test hanging).
            """
            if _queue_deadline is None:
                return None
            return partial(_permit_wait, untried)

        def _main_fallback_budget(t):
            """The budget for the LAST-RESORT main call after the pool failed.

            ⚠ A STATED TOTAL MUST SURVIVE THE FALLBACK. Both fallback arms
            used the caller's raw `timeout` and then raised it to
            `_MAIN_FALLBACK_TIMEOUT_S` (300s), so a caller that stated a 60s
            TOTAL could queue 57s on the pool and then hand the 35B 300s —
            357s for a "60s" budget, on the build's critical path, holding
            the single foreground inference slot (R7 lens A).
            
            The 300s exists because a node-sized budget on the 35B is a
            guaranteed ReadTimeout (2026-07-11), so it cannot simply be
            replaced by what remains — that would reintroduce the very
            regression the comment below documents. Instead: honour the
            remaining total, but never go below a 35B-sized floor.
            """
            floor = max(float(t or 0.0), _MAIN_FALLBACK_MIN_S)
            if _total_deadline is None:
                return max(float(t or 0.0), _MAIN_FALLBACK_TIMEOUT_S)
            remaining = _total_deadline - time.monotonic()
            return max(floor, min(_MAIN_FALLBACK_TIMEOUT_S, remaining))

        def _http_budget(t):
            """The POST budget, or `_BUDGET_BLOWN` if the request cannot finish.

            ⚠ RETURNING THE FLOOR HERE WAS A NODE-FAULT FACTORY. The old
            version clamped to `max(_MIN_HTTP_FLOOR, …)`, so a permit
            acquired with 0.5s left still POSTed — with 3s. That request
            almost always ReadTimeouts, and a ReadTimeout IS a node fault,
            so OUR OWN queueing was charged to the node's breaker. Measured:
            search.py's live parameters, permit freed at 41.5s of a 42s
            window -> POST budget 3.497s -> ReadTimeout -> breaker failure
            recorded (R7 lens A). `deep_research` fans out three at a time,
            which is enough to open it for 60s — the req-0fb69c5f outcome
            through a new door.
            
            `NodeSaturated` exists precisely so "we never asked" is not a
            node fault. When the reserve cannot be met, decline the permit
            and say so, rather than spending it on a doomed request.

            ⚠ EVALUATED AFTER THE PERMIT IS HELD. R5 computed this on the
            line ABOVE the `async with`, i.e. at t=0 before any queueing, so
            it never actually clipped anything: `route()` still measured
            19.95s against its stated 12s, and search.py still lost URLs —
            the very outcome the change was written to prevent.
            """
            if _total_deadline is None:
                return t
            remaining = _total_deadline - time.monotonic()
            if remaining < _MIN_HTTP_FLOOR:
                return _BUDGET_BLOWN
            if t is None:
                # The deadline covers the request too, so a caller that
                # named no timeout still gets one — otherwise the node
                # client's 1200s default applies and the "total" is fiction.
                return remaining
            return min(float(t), remaining)

        # Request prefix-cache reuse on the upstream. llama.cpp's server
        # honours this as an OpenAI-compatible extension; other backends
        # (vLLM, OpenAI-proper) silently ignore unknown fields. Setting
        # it explicitly is insurance: llama.cpp's default is `true` but
        # it can be flipped off globally, and being explicit documents
        # intent for the reader.
        payload.setdefault("cache_prompt", True)
        # True once an off-main pool was tried and every node in it failed.
        # LOCAL, not instance state — concurrent calls must not see each
        # other's fallback status.
        fell_back_from_node = False
        _pool_leg = ""
        # WHICH pool the caller asked for — independent of whether one is
        # configured. `fell_back_from_node` only becomes True when a
        # CONFIGURED pool fails, so a pool that is simply absent skipped every
        # branch: `off_main_only` never raised (silently running on the 35B
        # the caller forbade), and `served_leg` reported `{"served_by":
        # "main", "fell_back_from": ""}` — indistinguishable from a deliberate
        # main call. That is exactly the blindness the stamp was added to
        # remove, and it is why the planner's `use_swarm=True` against an
        # empty pool is invisible today (R2 lenses A and C).
        # ⚠ SET THE FLAG FROM THE REQUEST, not from inside `if node:`. It used
        # to be assigned only after a selector returned a node, so a selector
        # that returns None (which any future "refuse while the breaker is
        # open" change would introduce) left it False — and then
        # `off_main_only` was never consulted and the timeout was stripped to
        # httpx's 1200s default. A trap armed under the next edit (R2 lens B,
        # NEW-4).
        _requested_pool = ("vision" if use_vision else
                           "worker" if use_worker else
                           "critic" if use_critic else
                           "coding" if use_coding else
                           "swarm" if use_swarm else "")
        # Read by nothing since R4 removed the unconfigured-pool raise; kept
        # only because the log line below reads better with a name than with
        # the expression. If you find yourself branching on it, re-read the
        # arm at the bottom of this function first — that is where an
        # unconfigured pool is handled now.
        _pool_configured = bool(
            getattr(self, "vision_clients", None) if use_vision else
            getattr(self, "worker_clients", None) if use_worker else
            getattr(self, "critic_clients", None) if use_critic else
            getattr(self, "coding_clients", None) if use_coding else
            getattr(self, "swarm_clients", None) if use_swarm else True)
        # Heartbeat traffic logs transitions, not ticks (see keepalive_workers)
        # — its per-ping routing/failure lines stay at debug.
        # Heartbeat-class callers: their whole point is to run constantly, so
        # their failures are a STATE, reported once on transition by
        # `keepalive_workers`, not an event to log per tick. This gated only
        # the WORKER branch — critic, vision, coding and swarm logged
        # unconditionally, so a down Nova emitted a spurious WARNING every 45s
        # forever from the critic leg, and boot warmup produced 15 lines for
        # one unreachable node. The critic branch already special-cased this
        # tuple for its DISPATCH line twenty lines above: the announcement was
        # fixed and the failure was missed (R2 lens B, items 8 + NEW-3).
        _quiet = task_label in ("keepalive", "warmup")
        if use_vision:
            _pool_leg = "vision"
            if getattr(self, 'vision_clients', None):
                target_model = payload.get("model")
                tried_nodes = []

                node = self.get_vision_node(target_model, require_healthy=require_healthy)
                if node is None:
                    # Same guard as worker/critic/coding: a pool that yields
                    # nothing must set the fallback flag, or `off_main_only`
                    # is skipped and the timeout is stripped. R2 added this
                    # to three branches and missed the two its explanatory
                    # comment did not sit next to (found independently by R3
                    # lenses A and B).
                    fell_back_from_node = True

                if node:
                    for _ in range(len(self.vision_clients)):
                        if not node:
                            break

                        if node in tried_nodes:
                            target_model = None
                            node = self.get_vision_node(target_model, require_healthy=require_healthy)

                        loop_breaker = 0
                        while node in tried_nodes and loop_breaker < len(self.vision_clients):
                            node = self.get_vision_node(None, require_healthy=require_healthy)
                            loop_breaker += 1

                        # Every vision node has been exhausted — break the
                        # outer retry loop instead of re-appending and
                        # hammering the same dead node again.
                        if node in tried_nodes:
                            break

                        tried_nodes.append(node)

                        pretty_log("Vision Compute", f"Routing request to Vision Node ({node['model']})", level="INFO", icon=Icons.TOOL_DEEP)
                        try:
                            node_payload = payload.copy()
                            node_payload["model"] = node["model"]

                            import json

                            body_bytes = json.dumps(node_payload, ensure_ascii=True).encode('ascii', errors='ignore')

                            kwargs = {}
                            _untried = max(1, len(self.vision_clients) - len(tried_nodes) + 1)
                            async with self._node_slot(
                                    node, wait_timeout=_gate_wait(_untried)):
                                _t = _http_budget(timeout)
                                if _t is _BUDGET_BLOWN:
                                    raise NodeSaturated(
                                        f"{node.get('model') or node.get('url')}: "
                                        f"the caller's total budget is spent; "
                                        f"declining the permit rather than "
                                        f"POSTing a request that cannot finish")
                                if _t is not None:
                                    kwargs["timeout"] = _t
                                resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                            resp.raise_for_status()
                            self.circuit_breaker.record_success(node["url"])
                            return _stamp_leg(resp.json(), _pool_leg)
                        except Exception as e:
                            if _is_node_fault(e) and task_label != "warmup":
                                self.circuit_breaker.record_failure(node["url"])
                            if not _quiet:
                                pretty_log("Vision Node Failed", f"{node['model']}: {_node_error_detail(e)} — trying next", level="WARNING", icon=Icons.WARN)
                            target_model = None
                            node = self.get_vision_node(target_model, require_healthy=require_healthy)
                            continue

                pretty_log("Vision Compute Failed", "All vision nodes failed.", level="ERROR", icon=Icons.FAIL)

            # ⚠ Name which of the two it is. This said "the dedicated vision
            # node is offline or returned an error" even when NO vision pool
            # was ever configured — sending the operator to check a node that
            # does not exist (R4 lens B, NEW-13).
            _why = ("no vision nodes are configured"
                    if not getattr(self, "vision_clients", None)
                    else "the dedicated vision node is offline or returned "
                         "an error")
            raise Exception(
                f"Vision analysis failed: {_why}, and the main upstream "
                f"model does not support image inputs.")

        if use_worker and getattr(self, 'worker_clients', None):
            _pool_leg = "worker"
            target_model = payload.get("model")
            tried_nodes = []

            node = self.get_worker_node(target_model, require_healthy=require_healthy)
            if node is None:
                # ⚠ The pool was REQUESTED and CONFIGURED but yielded
                # nothing (every node sick, and the caller asked for a
                # healthy one). This flag used to be set only inside
                # `if node:`, so a None selector left it False —
                # `off_main_only` was never consulted and the timeout
                # was stripped to httpx's 1200s default. A trap armed
                # under exactly the `require_healthy` change that now
                # makes None reachable (R2 lens B, NEW-4).
                fell_back_from_node = True

            if node:
                for _ in range(len(self.worker_clients)):
                    if not node:
                        break

                    if node in tried_nodes:
                        target_model = None
                        node = self.get_worker_node(target_model, require_healthy=require_healthy)

                    loop_breaker = 0
                    while node in tried_nodes and loop_breaker < len(self.worker_clients):
                        node = self.get_worker_node(None, require_healthy=require_healthy)
                        loop_breaker += 1

                    # Every worker node has been exhausted — break the outer
                    # retry loop instead of re-appending and hammering the
                    # same dead node again (mirrors the vision/coding guard).
                    if node in tried_nodes:
                        break

                    tried_nodes.append(node)

                    if _quiet:
                        logger.debug("keepalive → worker node %s", node.get("model"))
                    else:
                        _vp = verify_purpose_context.get()
                        _wl = task_label or 'background task'
                        if _vp:
                            _wl = f"{_wl} ({_vp})"
                        # Same rule as the critic line below: a background
                        # worker dispatch (failure-dimension tagging, REM,
                        # self-play) is plumbing nobody reads — 211 of 4000
                        # lines. Request-scoped work stays visible.
                        _bg = request_id_context.get() == "SYSTEM"
                        pretty_log("Worker Compute", f"{_wl} → Worker Node ({node['model']})", level="DEBUG" if _bg else "INFO", icon=Icons.NODE_WORKER)
                    try:
                        node_payload = payload.copy()
                        node_payload["model"] = node["model"]
                        _disable_thinking(node_payload)

                        import json

                        body_bytes = json.dumps(node_payload, ensure_ascii=True).encode('ascii', errors='ignore')

                        kwargs = {}
                        _untried = max(1, len(self.worker_clients) - len(tried_nodes) + 1)
                        async with self._node_slot(
                                node, wait_timeout=_gate_wait(_untried)):
                            _t = _http_budget(timeout)
                            if _t is _BUDGET_BLOWN:
                                raise NodeSaturated(
                                f"{node.get('model') or node.get('url')}: "
                                f"the caller's total budget is spent; "
                                f"declining the permit rather than "
                                f"POSTing a request that cannot finish")
                            if _t is not None:
                                kwargs["timeout"] = _t
                            resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                        resp.raise_for_status()
                        self.circuit_breaker.record_success(node["url"])
                        return _stamp_leg(resp.json(), _pool_leg)
                    except Exception as e:
                        if _is_node_fault(e) and task_label != "warmup":
                            self.circuit_breaker.record_failure(node["url"])
                        if _quiet:
                            logger.debug("keepalive worker %s failed: %s",
                                         node.get("model"), type(e).__name__)
                        else:
                            pretty_log("Worker Node Failed", f"{node['model']}: {_node_error_detail(e)} — trying next", level="WARNING", icon=Icons.WARN)
                        target_model = None
                        node = self.get_worker_node(target_model, require_healthy=require_healthy)
                        continue

                # Say what will ACTUALLY happen. With off_main_only (route()),
                # there is no main-model fallback — the caller degrades to its
                # own cheap fallback. The old unconditional "falling back to
                # main upstream" text was a lie in that case and cost real
                # debugging time when reading the live log.
                if not _quiet:
                    pretty_log(
                        "Worker Compute Failed",
                        "All worker nodes failed — "
                        + ("caller will use its local fallback (no main-model "
                           "retry)" if off_main_only
                           else "falling back to main upstream"),
                        level="WARNING", icon=Icons.WARN,
                    )
                fell_back_from_node = True

        elif use_critic and getattr(self, 'critic_clients', None):
            _pool_leg = "critic"
            target_model = payload.get("model")
            tried_nodes = []

            node = self.get_critic_node(target_model, require_healthy=require_healthy)
            if node is None:
                # ⚠ The pool was REQUESTED and CONFIGURED but yielded
                # nothing (every node sick, and the caller asked for a
                # healthy one). This flag used to be set only inside
                # `if node:`, so a None selector left it False —
                # `off_main_only` was never consulted and the timeout
                # was stripped to httpx's 1200s default. A trap armed
                # under exactly the `require_healthy` change that now
                # makes None reachable (R2 lens B, NEW-4).
                fell_back_from_node = True

            if node:
                for _ in range(len(self.critic_clients)):
                    if not node:
                        break

                    if node in tried_nodes:
                        target_model = None
                        node = self.get_critic_node(target_model, require_healthy=require_healthy)

                    loop_breaker = 0
                    while node in tried_nodes and loop_breaker < len(self.critic_clients):
                        node = self.get_critic_node(None, require_healthy=require_healthy)
                        loop_breaker += 1

                    if node in tried_nodes:
                        break

                    tried_nodes.append(node)

                    _vp = verify_purpose_context.get()
                    # ⚠ LEVEL DEPENDS ON WHO IS ASKING (2026-08-09 log audit).
                    # This line announces an INTENT, and it fired 1648 times
                    # in a 4000-line window — 41% of the operator's log —
                    # while only 6 lines in the same window carried a verdict.
                    # The log was 274:1 announcements-to-outcomes.
                    #
                    # Background callers (self-play, REM, failure-dimension
                    # tagging) route verification constantly while idle and
                    # nobody reads those; a real user turn is exactly what the
                    # operator IS watching. `request_id_context` already
                    # distinguishes them ("SYSTEM" = not request-scoped), so
                    # background plumbing drops to DEBUG and request-scoped
                    # routing stays visible. Failures are untouched — they are
                    # WARNING below and always were.
                    _bg = request_id_context.get() == "SYSTEM"
                    # ⚠ A HEARTBEAT IS NOT A VERIFICATION (2026-08-10). The
                    # keepalive loop (45s) and the node warmup both reach this
                    # branch via `use_critic=(label == "critic")`, so every
                    # ping announced itself as "Routing verification" — a line
                    # describing work that never happened. The WORKER branch
                    # has had `_quiet` for exactly this since the heartbeat was
                    # added; the critic branch never got it, so the mislabel
                    # survived. Found while checking why real verifications
                    # produced no outcome line: they had not run at all, and
                    # these pings were what made it look as if they had.
                    if task_label in ("keepalive", "warmup"):
                        logger.debug("critic %s ping → %s",
                                     task_label, node["model"])
                    else:
                        pretty_log("Critic Compute", f"Routing verification{f' ({_vp})' if _vp else ''} to Critic Node ({node['model']})", level="DEBUG" if _bg else "INFO", icon=Icons.VERIFIER_LAB)
                    try:
                        import copy as _copy, json
                        node_payload = _copy.deepcopy(payload)
                        node_payload["model"] = node["model"]

                        body_bytes = json.dumps(node_payload, ensure_ascii=True).encode('utf-8')

                        kwargs = {}
                        _untried = max(1, len(self.critic_clients) - len(tried_nodes) + 1)
                        async with self._node_slot(
                                node, wait_timeout=_gate_wait(_untried)):
                            _t = _http_budget(timeout)
                            if _t is _BUDGET_BLOWN:
                                raise NodeSaturated(
                                f"{node.get('model') or node.get('url')}: "
                                f"the caller's total budget is spent; "
                                f"declining the permit rather than "
                                f"POSTing a request that cannot finish")
                            if _t is not None:
                                kwargs["timeout"] = _t
                            resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                        resp.raise_for_status()
                        self.circuit_breaker.record_success(node["url"])
                        return _stamp_leg(resp.json(), _pool_leg)
                    except Exception as e:
                        if _is_node_fault(e) and task_label != "warmup":
                            self.circuit_breaker.record_failure(node["url"])
                        if not _quiet:
                            pretty_log("Critic Node Failed", f"{node['model']}: {_node_error_detail(e)} — trying next", level="WARNING", icon=Icons.WARN)
                        target_model = None
                        node = self.get_critic_node(target_model, require_healthy=require_healthy)
                        continue

                if off_main_only:
                    # ⚠ NOT a fallback: this caller forbids the main model, so
                    # the call raises below and main is never touched. Saying
                    # "falling back to main upstream" here is simply false —
                    # keepalive alone emitted it every 45s, ~2,880 times a day
                    # (LLM review 2026-08-18). The worker branch fixed this;
                    # the other three pools kept the unconditional line.
                    if not _quiet:
                        pretty_log("Critic Nodes Unavailable",
                                   "all critic nodes failed; this caller is "
                                   "off-main-only, so nothing runs on the 35B",
                                   level="WARNING", icon=Icons.WARN)
                elif not _quiet:
                    pretty_log("Critic Compute Failed", "All critic nodes failed, falling back to main upstream", level="WARNING", icon=Icons.WARN)
                fell_back_from_node = True

        elif use_coding and getattr(self, 'coding_clients', None):
            _pool_leg = "coding"
            target_model = payload.get("model")
            tried_nodes = []

            node = self.get_coding_node(target_model, require_healthy=require_healthy)
            if node is None:
                # ⚠ The pool was REQUESTED and CONFIGURED but yielded
                # nothing (every node sick, and the caller asked for a
                # healthy one). This flag used to be set only inside
                # `if node:`, so a None selector left it False —
                # `off_main_only` was never consulted and the timeout
                # was stripped to httpx's 1200s default. A trap armed
                # under exactly the `require_healthy` change that now
                # makes None reachable (R2 lens B, NEW-4).
                fell_back_from_node = True

            if node:
                for _ in range(len(self.coding_clients)):
                    if not node:
                        break

                    if node in tried_nodes:
                        target_model = None
                        node = self.get_coding_node(target_model, require_healthy=require_healthy)

                    loop_breaker = 0
                    while node in tried_nodes and loop_breaker < len(self.coding_clients):
                        node = self.get_coding_node(None, require_healthy=require_healthy)
                        loop_breaker += 1

                    if node in tried_nodes:
                        break

                    tried_nodes.append(node)

                    pretty_log("Coding Compute", f"Routing request to Coding Node ({node['model']})", level="INFO", icon=Icons.TOOL_CODE)
                    try:
                        import copy as _copy, json
                        node_payload = _copy.deepcopy(payload)
                        node_payload["model"] = node["model"]

                        body_bytes = json.dumps(node_payload, ensure_ascii=True).encode('utf-8')

                        kwargs = {}
                        _untried = max(1, len(self.coding_clients) - len(tried_nodes) + 1)
                        async with self._node_slot(
                                node, wait_timeout=_gate_wait(_untried)):
                            _t = _http_budget(timeout)
                            if _t is _BUDGET_BLOWN:
                                raise NodeSaturated(
                                f"{node.get('model') or node.get('url')}: "
                                f"the caller's total budget is spent; "
                                f"declining the permit rather than "
                                f"POSTing a request that cannot finish")
                            if _t is not None:
                                kwargs["timeout"] = _t
                            resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                        resp.raise_for_status()
                        self.circuit_breaker.record_success(node["url"])
                        return _stamp_leg(resp.json(), _pool_leg)
                    except Exception as e:
                        if _is_node_fault(e) and task_label != "warmup":
                            self.circuit_breaker.record_failure(node["url"])
                        if not _quiet:
                            pretty_log("Coding Node Failed", f"{node['model']}: {_node_error_detail(e)} — trying next", level="WARNING", icon=Icons.WARN)
                        target_model = None
                        node = self.get_coding_node(target_model, require_healthy=require_healthy)
                        continue

                if off_main_only:
                    # ⚠ NOT a fallback: this caller forbids the main model, so
                    # the call raises below and main is never touched. Saying
                    # "falling back to main upstream" here is simply false —
                    # keepalive alone emitted it every 45s, ~2,880 times a day
                    # (LLM review 2026-08-18). The worker branch fixed this;
                    # the other three pools kept the unconditional line.
                    if not _quiet:
                        pretty_log("Coding Nodes Unavailable",
                                   "all coding nodes failed; this caller is "
                                   "off-main-only, so nothing runs on the 35B",
                                   level="WARNING", icon=Icons.WARN)
                elif not _quiet:
                    pretty_log("Coding Compute Failed", "All coding nodes failed, falling back to main upstream", level="WARNING", icon=Icons.WARN)
                fell_back_from_node = True

        elif use_swarm and self.swarm_clients:
            _pool_leg = "swarm"
            target_model = payload.get("model")
            tried_nodes = []

            node = self.get_swarm_node(target_model, require_healthy=require_healthy)
            if node is None:
                # Same guard as worker/critic/coding: a pool that
                # yields nothing must set the fallback flag, or
                # `off_main_only` is skipped and the timeout is
                # stripped. R2 added this to three branches and
                # missed the two the comment did not sit next to
                # (found independently by R3 lenses A and B).
                fell_back_from_node = True

            if node:
                for _ in range(len(self.swarm_clients)):
                    if not node:
                        break

                    if node in tried_nodes:
                        target_model = None
                        node = self.get_swarm_node(target_model, require_healthy=require_healthy)

                    loop_breaker = 0
                    while node in tried_nodes and loop_breaker < len(self.swarm_clients):
                        node = self.get_swarm_node(None, require_healthy=require_healthy)
                        loop_breaker += 1

                    # Every swarm node has been exhausted — break the outer
                    # retry loop instead of re-appending and hammering the
                    # same dead node again (mirrors the vision/coding guard).
                    if node in tried_nodes:
                        break

                    tried_nodes.append(node)

                    pretty_log("Edge Compute", f"Routing request to Swarm Node ({node['model']})", level="INFO", icon=Icons.NODE_EDGE)
                    try:
                        import copy as _copy, json
                        node_payload = _copy.deepcopy(payload)
                        node_payload["model"] = node["model"]

                        body_bytes = json.dumps(node_payload, ensure_ascii=True).encode('utf-8')

                        kwargs = {}
                        _untried = max(1, len(self.swarm_clients) - len(tried_nodes) + 1)
                        async with self._node_slot(
                                node, wait_timeout=_gate_wait(_untried)):
                            _t = _http_budget(timeout)
                            if _t is _BUDGET_BLOWN:
                                raise NodeSaturated(
                                f"{node.get('model') or node.get('url')}: "
                                f"the caller's total budget is spent; "
                                f"declining the permit rather than "
                                f"POSTing a request that cannot finish")
                            if _t is not None:
                                kwargs["timeout"] = _t
                            resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                        resp.raise_for_status()
                        self.circuit_breaker.record_success(node["url"])
                        return _stamp_leg(resp.json(), _pool_leg)
                    except Exception as e:
                        if _is_node_fault(e) and task_label != "warmup":
                            self.circuit_breaker.record_failure(node["url"])
                        if not _quiet:
                            pretty_log("Swarm Node Failed", f"{node['model']}: {_node_error_detail(e)} — trying next", level="WARNING", icon=Icons.WARN)
                        target_model = None
                        node = self.get_swarm_node(target_model, require_healthy=require_healthy)
                        continue

                if off_main_only:
                    if not _quiet:
                        pretty_log("Edge Nodes Unavailable",
                                   "all swarm nodes failed; this caller is "
                                   "off-main-only, so nothing runs on the 35B",
                                   level="WARNING", icon=Icons.WARN)
                elif not _quiet:
                    pretty_log("Edge Compute Failed", "All swarm nodes failed, falling back to main upstream", level="WARNING", icon=Icons.WARN)
                fell_back_from_node = True

        # ---- main-upstream fallback -------------------------------------
        # Reached either because no off-main pool was requested/configured, or
        # because every node in the requested pool FAILED.
        # ⚠ NO RAISE FOR AN UNCONFIGURED POOL. R2 added one here, reasoning
        # that "off_main_only means never run on main". That reading is wrong
        # about WHY the flag exists: §4O A-MAJOR-2 introduced it to relieve
        # QUEUE PRESSURE on the single main slot. With no pool configured
        # there is no queue to relieve and no node that could ever serve the
        # call — main is the only way the subsystem exists at all.
        #
        # Measured blast radius of the raise (R4 lens B, 12 subsystems, none
        # of which guards on the pool and only two of which catch this
        # exception by name): smart-memory consolidation and post-mortems
        # LOSE THEIR JOURNAL ITEM PERMANENTLY (the message does not match
        # `is_upstream_transient`, so it never requeues); a user-triggered
        # dream replies "Dream error: no worker nodes are configured…";
        # lesson graduation reports "0 lessons graduated" — a success-shaped
        # string for a dead subsystem; and both web-summary paths swallow it
        # in a bare `except` with no log at all.
        #
        # The correct handling already exists twenty lines below: the
        # "requested but unconfigured" arm bounds the main call to
        # `_MAIN_FALLBACK_TIMEOUT_S`. The raise merely shadowed it. Zero live
        # impact (production sets --worker-nodes), which is exactly why it
        # went unnoticed — it only breaks a deployment without a worker box.
        if fell_back_from_node:
            if off_main_only:
                # The caller (e.g. `route()`) exists precisely to keep this
                # work OFF the single main slot. Re-running it on the 35B is
                # worse than not doing it at all — the caller has a free
                # fallback. Raise; the caller degrades silently.
                raise OffMainNodeUnavailable(
                    "all off-main nodes failed; main-model fallback is "
                    "disabled for this call")
            # A node-sized timeout MUST NOT be applied to the main model
            # (2026-07-11). `timeout` here was sized for a small, fast worker
            # (route() uses 12s; measured 0.5s on the worker). The main model is
            # slower BY CONSTRUCTION — a 35B answering a real prompt takes tens
            # of seconds — so handing it the worker's budget guarantees a
            # ReadTimeout. Observed live: a worker hiccup produced
            #   worker node failed  Nova: ReadTimeout      (at the 6s budget)
            #   falling back to main upstream
            #   upstream fatal      ReadTimeout('')        (6s later — the 35B)
            # i.e. one slow worker call turned into a HARD upstream error.
            #
            # ⚠ BUT NOT UNBOUNDED. `timeout = None` handed the call httpx's
            # 1200-SECOND default, on the FOREGROUND path, for callers whose
            # own budget was 45s or 120s — and our OWN saturation gate
            # (NodeSaturated) reaches here too, so a busy worker converts into
            # a twenty-minute main-slot occupation. Give the 35B a budget
            # sized for the 35B, not the caller's node budget and not
            # infinity (three independent lenses, 2026-08-18).
            timeout = _main_fallback_budget(timeout)
        elif use_worker or use_critic or use_coding or use_swarm:
            # ⚠ NO `timeout is not None` GUARD. R4 removed the
            # unconfigured-pool raise on the stated grounds that "the correct
            # handling already exists twenty lines below: this arm bounds the
            # main call to _MAIN_FALLBACK_TIMEOUT_S". That claim was FALSE
            # for every caller that passes no timeout — this arm was gated on
            # `timeout is not None`, so those callers fell through to httpx's
            # 1200s client default. R5 lens A measured three of them
            # (tools/memory.py smart-memory x2, agent.py perfect-it) running
            # on MAIN with no bound. That traded "smart-memory loses its
            # journal item" for "smart-memory occupies the single main
            # inference slot for twenty minutes" — a worse bug than the one
            # R4 set out to fix, hidden behind a comment asserting the
            # opposite. `max(float(timeout or 0.0), ...)` already handles
            # None, so the guard bought nothing.
            # A pool was REQUESTED but none is configured, so the pool branch
            # was skipped entirely (fell_back_from_node stayed False) and we
            # are about to run on the MAIN model carrying a timeout that was
            # sized for a small, fast off-main node. Same hazard as the
            # fell_back_from_node reset above — a 6s route budget on the 35B is
            # a guaranteed ReadTimeout — so it must not be applied.
            #
            # ⚠ …and this arm must be bounded exactly like its sibling. The
            # comment above already claimed "same hazard as the
            # fell_back_from_node reset", but that reset was bounded to
            # _MAIN_FALLBACK_TIMEOUT_S while this one still said `None`, i.e.
            # httpx's 1200s default — so the parity the comment asserted was
            # false, in the direction that costs a foreground slot (R2 lens A).
            timeout = _main_fallback_budget(timeout)

        for attempt in range(2):
            try:
                kwargs = {}
                if timeout is not None:
                    kwargs["timeout"] = timeout
                async with self._main_node_lock:
                    resp = await self.http_client.post("/v1/chat/completions", json=payload, **kwargs)
                resp.raise_for_status()
                # A 200 with an empty / non-JSON body crashed here as
                # `json.JSONDecodeError: Expecting value: line 1 column 1 (char
                # 0)` → "Upstream Fatal", turning a recoverable state into a hard
                # failure (observed right after a context overflow: the server
                # returned 0 bytes while the emergency-prune retry was in flight).
                # Treat it as a transient upstream glitch: retry once, then raise
                # a clean, explanatory error instead of a bare decoder traceback.
                try:
                    return _stamp_leg(
                        resp.json(), "main",
                        fell_back_from=(_pool_leg if fell_back_from_node else ""),
                        requested=_requested_pool or "main")
                except ValueError as je:   # JSONDecodeError subclasses ValueError
                    # NB: do not reference `json` here — conditional `import json`
                    # in the swarm branches above makes the name function-local
                    # and thus unbound on this (no-swarm) path.
                    body_len = len(resp.text or "")
                    if attempt < 1:
                        # A LARGE non-JSON 200 body that starts with "data:"
                        # is SSE — the caller passed a payload that still
                        # carries the main loop's `stream: true` into this
                        # NON-streaming API (observed 2026-07-18: the
                        # context-overflow recovery reused the turn payload
                        # and got 102 KB of SSE frames, read as "empty/
                        # non-JSON" → fatal). Strip the flag for the retry;
                        # harmless when the body was genuinely empty.
                        _note = ""
                        if payload.get("stream"):
                            payload["stream"] = False
                            if (resp.text or "").lstrip().startswith("data:"):
                                _note = " (SSE body on a non-streaming call — stripped stream flag)"
                        pretty_log("Upstream Empty Body",
                                   f"HTTP {resp.status_code} with non-JSON body ({body_len} B) — retrying{_note}",
                                   level="WARNING", icon=Icons.RETRY)
                        await asyncio.sleep(2)
                        continue
                    raise RuntimeError(
                        f"Upstream returned an empty/non-JSON response "
                        f"(HTTP {resp.status_code}, {body_len} bytes) after retry. "
                        f"This typically follows a context overflow or an upstream "
                        f"restart; the request did not complete."
                    ) from je
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError,
                    httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
                # ConnectTimeout / PoolTimeout mean the request was NEVER put on
                # the wire (connection or pool-slot acquisition failed before
                # send), so retrying is idempotent and worth it right after a
                # llama-server restart. ReadTimeout is deliberately NOT here —
                # the request may already be executing upstream, so a retry
                # could double-run a non-idempotent generation.
                if attempt < 1:
                    wait_time = 2
                    pretty_log("Upstream Retry", f"[{attempt+1}/2] {type(e).__name__}. Retrying in {wait_time}s...", icon=Icons.RETRY)
                    await asyncio.sleep(wait_time)
                else:
                    pretty_log("Upstream Failed", f"Failed after 2 attempts: {_err_text(e)}", level="ERROR", icon=Icons.FAIL)
                    raise
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 or "parse input" in e.response.text.lower():
                    if attempt < 1:
                        wait_time = 2
                        pretty_log("Upstream Retry", f"[{attempt+1}/2] HTTP {e.response.status_code} Server Glitch. Retrying in {wait_time}s...", icon=Icons.RETRY)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        pretty_log("Upstream Failed", f"Failed after 2 attempts: {_err_text(e)}", level="ERROR", icon=Icons.FAIL)
                        raise
                pretty_log("Upstream Error", f"HTTP {e.response.status_code}: {e.response.text}", level="ERROR", icon=Icons.FAIL)
                raise
            except Exception as e:
                err_str = str(e) or repr(e)
                pretty_log("Upstream Fatal", err_str, level="ERROR", icon=Icons.FAIL)
                raise

        raise Exception("Max retries exceeded")

    async def _wait_for_foreground_clear(self):
        """Park a background caller until the foreground is idle.

        Two signals, two budgets:
        - ``foreground_tasks`` (an LLM call in flight): wait up to 30s,
          then proceed — a slightly stale background result beats no
          result, and the in-flight call may be long.
        - ``foreground_requests`` (a USER REQUEST anywhere in its
          handle_chat lifecycle): wait essentially as long as it takes
          (10-minute hard ceiling against a leaked counter). While a user
          is actively being served, background work has NO claim on the
          single inference slot — letting it sneak in between the user
          turn's tool calls is exactly what starved the prompt for ~12
          minutes after req 70.
        """
        waited = 0.0
        while waited < 600.0:
            async with self._foreground_lock:
                request_active = self.foreground_requests > 0
                if self.foreground_tasks <= 0 and not request_active:
                    return
            if not request_active and waited >= 30.0:
                return
            # One visibility line per long park: a background call waiting
            # minutes for the slot is normal under load, but the operator
            # watching the stream should be able to SEE it — a silent
            # multi-minute gap is indistinguishable from a hang.
            if waited == 120.0:
                pretty_log(
                    "BG Queue Wait",
                    "Background LLM call parked 120s waiting for the "
                    "foreground to clear (user request active). Will keep "
                    "waiting up to 600s.",
                    icon=Icons.RETRY,
                )
            await asyncio.sleep(1.0)
            waited += 1.0

    async def chat_completion(self, payload: Dict[str, Any], use_swarm: bool = False, use_worker: bool = False, use_vision: bool = False, use_coding: bool = False, use_critic: bool = False, is_background: bool = False, timeout: Optional[float] = None, off_main_only: bool = False, task_label: str = "", require_healthy: bool = False, slot_wait: Optional[float] = None, total_budget: Optional[float] = None) -> Dict[str, Any]:
        if is_background:
            # The foreground wait protects exactly one resource: the MAIN
            # inference slot. A call that will be served by an off-main
            # pool (worker/critic/vision/swarm) doesn't contend for it, so
            # parking it behind an active user request is pure added
            # latency — and for calls awaited inline it was a deadlock-
            # shaped self-stall (the request waits on a call that waits on
            # the request). Only fall back to the wait when the call will
            # actually land on the main node. (If every off-main node
            # fails, _do_chat_completion may still fall back to main — a
            # rare edge; §4O A-MAJOR-2 makes background pool callers pass
            # off_main_only=True so that fallback RAISES instead of
            # dogpiling the main slot, since §4O A-MAJOR-1 means this
            # off-main branch no longer holds the _bg_queue_sem.)
            # ⚠ COMPARE URLS, not pool membership. This asked "is a pool
            # configured?" — but `--visual-nodes` is byte-identical to
            # `--upstream-url` in production (both http://127.0.0.1:8088), so
            # a background VISION call answered "not main", skipped both the
            # foreground wait and the 3-permit background semaphore, and
            # landed on the single main slot mid-turn. That is the post-req-70
            # starvation `foreground_requests` exists to prevent (R2 lens B,
            # NEW-1). A pool whose node IS the main box does not stop being
            # the main box because it has a different label.
            _main_url = str(getattr(self, "upstream_url", "") or "").rstrip("/")

            def _pool_is_off_main(clients) -> bool:
                if not clients:
                    return False
                if not _main_url:
                    # Unknown main URL (a client built without __init__, as
                    # several tests do): fall back to the old membership
                    # answer rather than guessing.
                    return True
                return any(str(n.get("url", "")).rstrip("/") != _main_url
                           for n in clients)

            targets_main_node = not (
                (use_worker and _pool_is_off_main(getattr(self, "worker_clients", None)))
                or (use_critic and _pool_is_off_main(getattr(self, "critic_clients", None)))
                or (use_vision and _pool_is_off_main(getattr(self, "vision_clients", None)))
                or (use_swarm and _pool_is_off_main(getattr(self, "swarm_clients", None)))
                or (use_coding and _pool_is_off_main(getattr(self, "coding_clients", None)))
            )
            # §4O A-MAJOR-1: only MAIN-targeted background calls contend for
            # the single foreground slot, so only they wait for the
            # foreground-clear AND queue on the 3-permit `_bg_queue_sem`. An
            # OFF-main background call (worker/critic/…) does neither — the
            # old code acquired the sem unconditionally, so a critical-path
            # off-main route() (query-expansion/decompose/verify, run before
            # hydration) could park behind up to 3 long background STREAM
            # holders with NO timeout, defeating route()'s documented
            # fail-fast (which bounds only the HTTP call, not the sem).
            import contextlib as _ctxlib
            async with _ctxlib.AsyncExitStack() as _bg_stack:
                if targets_main_node:
                    await self._wait_for_foreground_clear()
                    await _bg_stack.enter_async_context(self._bg_queue_sem)
                _result = await self._do_chat_completion(payload, use_swarm, use_worker, use_vision, use_coding, use_critic, timeout, off_main_only, task_label, require_healthy, slot_wait, total_budget)
                self._note_usage(_result)
                _leg = served_leg(_result)
                _served = _leg.get("served_by") or ""
                self._maybe_record_call(payload, _result,
                                        use_worker=(_served == "worker"),
                                        use_vision=(_served == "vision"),
                                        use_critic=(_served == "critic"),
                                        served_by=_served,
                                        requested_pool=_leg.get("requested") or "",
                                        task_label=task_label,
                                        background=True)
                return _result
        else:
            async with self._foreground_lock:
                self.foreground_tasks += 1
            try:
                _result = await self._do_chat_completion(payload, use_swarm, use_worker, use_vision, use_coding, use_critic, timeout, off_main_only, task_label, require_healthy, slot_wait, total_budget)
                self._note_usage(_result)
                # ⚠ Record WHICH LEG SERVED IT, not which was requested. The
                # meta used to carry the caller's flags verbatim, so a
                # main-served critic call was filed in the §4BG corpus as
                # `use_critic=True` — wrong model provenance on exactly the
                # degraded turns worth studying. `_stamp_leg`'s docstring
                # already named this; the fix had landed only in the verifier
                # (R2 lenses A and C).
                _leg = served_leg(_result)
                _served = _leg.get("served_by") or ""
                self._maybe_record_call(payload, _result,
                                        use_worker=(_served == "worker"),
                                        use_vision=(_served == "vision"),
                                        use_critic=(_served == "critic"),
                                        served_by=_served,
                                        requested_pool=_leg.get("requested") or "",
                                        task_label=task_label)
                return _result
            finally:
                async with self._foreground_lock:
                    self.foreground_tasks -= 1
                    if self.foreground_tasks < 0:
                        self.foreground_tasks = 0

    # ---- Upstream token accounting -----------------------------------
    # `Trajectory.tokens_in/out` and `eval.TaskResult.tokens_used` both
    # existed as fields for months and read 0 on every record, because
    # nothing ever read the upstream's `usage` block. A turn makes MANY
    # calls (each tool round-trip, plus the verifier), so the per-turn
    # number is a SUM — keyed on request id, not stored on the client,
    # because concurrent requests interleave here.
    #
    # Same ring discipline as `core/turn_facts.py`: bounded, best-effort,
    # never raises. A failure to count tokens must never break a turn.
    _USAGE_RING_MAX = 32

    def _usage_ring(self):
        ring = getattr(self, "_usage_by_req", None)
        if ring is None:
            from collections import OrderedDict
            ring = OrderedDict()
            self._usage_by_req = ring
        return ring

    def _note_usage(self, result: Any) -> None:
        """Fold one response's `usage` into the current request's running
        total. Tolerant by contract — `result` may be a str (the `route`
        path), None, or a dict with no usage at all."""
        try:
            if not isinstance(result, dict):
                return
            usage = result.get("usage")
            if not isinstance(usage, dict):
                return
            from ..utils.logging import request_id_context
            req_id = request_id_context.get()
            if not req_id:
                return
            ring = self._usage_ring()
            slot = ring.get(req_id)
            if slot is None:
                slot = {"tokens_in": 0, "tokens_out": 0,
                        "cached_tokens": 0, "calls": 0}
                ring[req_id] = slot
                while len(ring) > self._USAGE_RING_MAX:
                    ring.popitem(last=False)
            ring.move_to_end(req_id)
            slot["tokens_in"] += int(usage.get("prompt_tokens") or 0)
            slot["tokens_out"] += int(usage.get("completion_tokens") or 0)
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                # Prefill-cache hits. The log reports the system prompt's
                # CHARACTER count today; this is the first real measure of
                # whether that cache is actually being hit.
                slot["cached_tokens"] += int(details.get("cached_tokens") or 0)
            slot["calls"] += 1
        except Exception:  # noqa: BLE001 — accounting must never break a turn
            pass

    def _note_usage_from_sse(self, line: str) -> None:
        """Fold the usage block out of one raw SSE line. Separate from
        `_stream_rec_accumulate` because that one is gated on the opt-in
        recorder; token accounting has to run on every stream."""
        try:
            if isinstance(line, (bytes, bytearray)):
                line = line.decode("utf-8", "replace")
            if not line or not line.startswith("data:"):
                return
            body = line[5:].strip()
            if not body or body == "[DONE]":
                return
            chunk = json.loads(body)
            if isinstance(chunk, dict) and isinstance(chunk.get("usage"), dict):
                self._note_usage(chunk)
        except Exception:  # noqa: BLE001 — a malformed chunk is not fatal
            pass

    def usage_for(self, req_id: str) -> Dict[str, int]:
        """Running token totals for one request. Empty dict when unknown —
        an absent entry and a zero-token turn are different, and callers
        must be able to tell them apart."""
        try:
            if not req_id:
                return {}
            return dict(self._usage_ring().get(req_id) or {})
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _maybe_record_call(payload, result, kind: str = "chat_completion",
                          **meta) -> None:
        """LLM-boundary recording hook (GHOST_LLM_RECORD=1; off by
        default — payloads carry unredacted memory/profile text). Best-
        effort by contract; see core/llm_recording.py."""
        try:
            from .llm_recording import maybe_record
            maybe_record(kind, payload, result, **meta)
        except Exception:
            pass

    @staticmethod
    def _stream_rec_accumulate(line: str, acc: Dict[str, Any]) -> None:
        """Fold one SSE line into the stream-recording accumulator
        (§4F Phase 2b: the main tool loop STREAMS, so without this the
        recorder never saw a tool-bearing call — 7 fixtures in 21 h).
        Tolerant by contract: any unparseable line is ignored.

        ⚠ INCLUDING THE TYPE GUARD, which sat OUTSIDE the `try` and so
        made that contract false: `bytes.startswith("data:")` raises
        `TypeError`, and its only production caller is the streaming chunk
        loop, whose surrounding `except Exception` re-raises — i.e. with
        `GHOST_LLM_RECORD=1` a bytes chunk killed the user's turn outright.
        Latent for the same reason as the repr bug next to its call site
        (httpx yields str), and its documented sibling `_note_usage_from_sse`
        has always decoded first (§4BV R7)."""
        if isinstance(line, (bytes, bytearray)):
            line = line.decode("utf-8", "replace")
        if not isinstance(line, str) or not line.startswith("data:"):
            return
        body = line[5:].strip()
        if not body or body == "[DONE]":
            return
        try:
            chunk = json.loads(body)
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                acc["content"].append(str(delta["content"]))
            if delta.get("reasoning_content"):
                acc["reasoning"].append(str(delta["reasoning_content"]))
            # Native-tools streaming: llama-server emits the PARSED call as
            # indexed delta.tool_calls fragments (name once, arguments in
            # pieces) with finish_reason "tool_calls" — content stays empty,
            # so without this branch tool-choice fixtures record blank.
            for tc in (delta.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                idx = int(tc.get("index") or 0)
                slot = acc["tool_calls"].setdefault(
                    idx, {"id": "", "name": "", "arguments": []})
                if tc.get("id"):
                    slot["id"] = str(tc["id"])
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = str(fn["name"])
                if fn.get("arguments"):
                    slot["arguments"].append(str(fn["arguments"]))
            if choice.get("finish_reason"):
                acc["finish"] = choice["finish_reason"]
        except Exception:
            pass

    @staticmethod
    def _stream_rec_response(acc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Reassemble the accumulator into an OpenAI-shaped response dict,
        or None when nothing was captured (error/empty streams are not
        recorded — fixture mining wants complete generations)."""
        if (not acc["content"] and not acc["reasoning"]
                and not acc["tool_calls"]):
            return None
        msg: Dict[str, Any] = {"role": "assistant",
                               "content": "".join(acc["content"])}
        if acc["reasoning"]:
            msg["reasoning_content"] = "".join(acc["reasoning"])
        if acc["tool_calls"]:
            msg["tool_calls"] = [
                {"id": slot["id"], "type": "function",
                 "function": {"name": slot["name"],
                              "arguments": "".join(slot["arguments"])}}
                for _, slot in sorted(acc["tool_calls"].items())]
        return {"object": "chat.completion.stream-reassembled",
                "choices": [{"message": msg,
                             "finish_reason": acc.get("finish")}]}

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Fetches embeddings from the upstream LLM with robust retry logic.
        """
        # Empty input short-circuit. Without this the function fires a
        # network call with `{"input": []}` which wastes a round-trip and
        # some upstreams reject the payload with a 400.
        if not texts:
            return []
        payload = {"input": texts, "model": "default"}
        for attempt in range(2):
            try:
                # ⚠ THIS HOLDS THE SAME MUTEX AS EVERY MAIN CHAT CALL, and it
                # is the sibling `_do_chat_completion` on that lock that got
                # four rounds of hardening while this one got none (R4 lens B,
                # NEW-2). Two consequences, both measured:
                #   * no `timeout` meant httpx's 1200s default, so ONE wedged
                #     embeddings call blocked every main completion for up to
                #     twenty minutes;
                #   * `foreground_tasks` was never incremented, so through the
                #     whole block `_wait_for_foreground_clear` read an idle
                #     agent and waved background work straight into the mutex,
                #     and the biological watchdog saw an idle slot.
                # The counter is what makes this call VISIBLE; the timeout is
                # what makes it BOUNDED. Neither substitutes for the other.
                async with self._foreground_lock:
                    self.foreground_tasks += 1
                try:
                    async with self._main_node_lock:
                        resp = await self.http_client.post(
                            "/v1/embeddings", json=payload,
                            timeout=_EMBEDDINGS_TIMEOUT_S)
                finally:
                    async with self._foreground_lock:
                        self.foreground_tasks -= 1
                        if self.foreground_tasks < 0:
                            self.foreground_tasks = 0
                resp.raise_for_status()
                data = resp.json()
                return [item["embedding"] for item in data["data"]]
            # ⚠ `httpx.TimeoutException` covers Read/Connect/Pool/Write and
            # was NOT retried here, so a single slow response was fatal on
            # attempt 1 while every other main-node caller retried it. It is
            # also the family whose `str(e)` is the EMPTY STRING — hence
            # `_err_text`, without which this logged "Failed after 2
            # attempts: " and the operator learned nothing.
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError,
                    httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < 1:
                    wait_time = 2
                    await asyncio.sleep(wait_time)
                else:
                    pretty_log("Embedding Failed", f"Failed after 2 attempts: {_err_text(e)}", level="ERROR", icon=Icons.FAIL)
                    raise
            except Exception as e:
                pretty_log("Embedding Fatal", _err_text(e), level="ERROR", icon=Icons.FAIL)
                raise

        raise Exception("Max retries exceeded")

    # ⚠ OPEN, DELIBERATELY NOT FIXED (LLM review R2, decision recorded
    # 2026-08-18). This path holds NEITHER `_main_node_lock` (which
    # `_do_chat_completion` and `get_embeddings` take) NOR a `_node_slot`, and
    # it is the DOMINANT main-node caller — every user turn and every
    # sub-agent turn. Measured peak against the 1-slot llama-server: 5
    # simultaneous in-flight requests (3 streams + embeddings + vision). The
    # excess queues inside llama-server, invisible to our gate.
    #
    # Two fixes were tried and rejected here:
    #   * Taking the lock around `send()` only SERIALISES THE HANDSHAKE, not
    #     the generation — llama.cpp returns headers immediately and generates
    #     during body streaming. It looks like a fix and is worth ~nothing.
    #   * Holding it across the whole generator is the correct shape, but it
    #     means a mutex held across `yield`s on a live inference path: a
    #     consumer that abandons the generator without closing it wedges every
    #     main-node caller until GC. That trade needs an operator's eyes, not
    #     a reviewer's inference.
    #
    # The clean end state is to make the main URL an ordinary node in
    # `_node_slots` (its capacity is already probed — `/props` says 1) so all
    # three entry points share one budget, which would also subsume the
    # vision-pool-is-the-main-URL overlap. That is a structural change with
    # its own review.
    async def _do_stream_chat_completion(self, payload: Dict[str, Any], use_coding: bool = False):
        """
        Streams a chat completion request from the upstream LLM directly to the client.
        """
        import contextlib as _contextlib
        import copy as _copy
        payload = _copy.deepcopy(payload)
        payload["stream"] = True
        # Mirror the non-streaming path: ask llama.cpp to reuse any
        # matching prefix in its KV cache. See _do_chat_completion for
        # rationale.
        payload.setdefault("cache_prompt", True)

        # ── CODING-POOL ROUTING (§4BV R7) ────────────────────────────────
        # ⚠ THE NODE IS SELECTED PER ATTEMPT. Until R7 it was picked ONCE,
        # above the retry loop, with no gate, no breaker bookkeeping and no
        # fallback. Measured on a 2-node pool, node0 dead / node1 healthy,
        # 3 calls (§4BV R7 harness, `measure_coding.py`):
        #
        #                    nodes contacted        permits  breaker(dead)   main
        #   non-streaming    DEAD,GOOD x3           6        failures=3 open  n/a
        #   streaming        DEAD,DEAD,GOOD,DEAD,   0        failures=0 CLOSED never
        #                    DEAD  (2 of 3 calls raised ConnectError)
        #
        # Four consequences, every one of which the non-streaming sibling
        # had already been fixed for:
        #   * the breaker never learns, which makes `get_coding_node`'s
        #     `is_available()` filtering PROVABLY dead code on this path —
        #     verbatim the defect this file records as fixed at
        #     `generate_image`;
        #   * `client_to_use` was bound once, so the retry re-hit the SAME
        #     dead node (the DEAD,DEAD pairs above);
        #   * the pool sat outside the per-node concurrency gate, so a
        #     stream fan-out floods the node the gate exists to protect;
        #   * there was no main fallback, so one dead coding node is a hard
        #     failure for EVERY coding stream, forever.
        # Latent in production only because `--coding-nodes` is empty. With
        # an empty pool this block is a no-op and the loop below is
        # byte-equivalent to the pre-R7 two-attempt main-model retry.
        _coding_pool = (list(getattr(self, "coding_clients", None) or [])
                        if use_coding else [])
        _orig_model = payload.get("model")
        _has_model = "model" in payload
        _tried: List[Dict[str, Any]] = []
        _pool_done = not _coding_pool
        _fallback_logged = False
        # ⚠ ONE WALL-CLOCK QUEUE DEADLINE FOR THE WHOLE POOL, shared across
        # attempts — `_do_chat_completion`'s `_queue_deadline`, minus the
        # `_total_deadline`/`_http_budget` half.
        #
        # That omission is deliberate and re-derived here, not inherited: a
        # STREAM HAS NO MEANINGFUL TOTAL BUDGET. `_total_deadline` exists for
        # callers under an outer `asyncio.wait_for` that would lose work on
        # overrun (route(), the per-URL research distillations); it clips the
        # POST so queueing cannot eat the request. Nothing can pass a budget
        # here — `stream_chat_completion` takes no `timeout` — and a long
        # answer is not a fault, so clipping generation would be a
        # self-inflicted truncation. The stream's real bound is the per-chunk
        # watchdog below (`_STREAM_FIRST_BYTE_TIMEOUT` / `_STREAM_IDLE_TIMEOUT`).
        # `_MIN_HTTP_FLOOR` is likewise NOT reserved out of the wait, for the
        # same reason `_permit_wait` does not reserve it when no total budget
        # was stated: the reserve exists so a permit is never spent on a POST
        # that provably cannot finish, and here the POST is never clipped.
        _slot_wait = (_env_positive("GHOST_NODE_SLOT_WAIT_S", 90.0)
                      if _coding_pool else None)
        _queue_deadline = (None if _slot_wait is None
                           else time.monotonic() + _slot_wait)

        def _permit_wait(untried: int = 1) -> Optional[float]:
            """Share what remains of the queue budget across the attempts
            still to come.

            ⚠ FLOORED AT `_MIN_ACQUIRE`, and ⚠ `untried` INCLUDES THE CURRENT
            ATTEMPT — both of these are the non-streaming path's hard-won
            shape, and both were absent from the R6 draft of this function.
            `asyncio.wait_for(sem.acquire(), 0.0)` refuses a COMPLETELY FREE
            semaphore, so a 0 budget means "never ask this node": without the
            floor plus the +1, node 1 of a 2-node pool took the entire 90s
            and node 2 was handed exactly 0.0s and refused while idle. That
            is the precise defect `_MIN_ACQUIRE` was created for, reproduced
            inside the fix for it."""
            if _queue_deadline is None:
                return None
            remaining = _queue_deadline - time.monotonic()
            return max(_MIN_ACQUIRE, remaining / max(1, untried))

        def _gate_wait(untried: int):
            """The `wait_timeout` to hand the gate.

            ⚠ A CALLABLE, NOT A NUMBER. `_node_slot` resolves a callable
            AFTER its `/props` capacity probe; an argument EXPRESSION is
            evaluated before the context manager is even entered, so a 5s
            probe would be spent entirely outside the budget it is supposed
            to be inside. `_node_slot`'s own comment records this being
            measured (a stated 12s budget taking 14.01s) — and the R6 draft
            of this function passed a plain float anyway."""
            if _queue_deadline is None:
                return None
            return partial(_permit_wait, untried)

        def _pick_coding_node():
            """Next untried, breaker-permitted coding node, or None.

            Same loop-breaker shape as the non-streaming branch: with the
            breaker filtering, the selector can hand back an already-tried
            node, and re-appending it is what made the old code hammer one
            dead box."""
            n = self.get_coding_node(_orig_model if not _tried else None)
            guard = 0
            while n is not None and n in _tried and guard < len(_coding_pool):
                n = self.get_coding_node(None)
                guard += 1
            if n is None or n in _tried:
                return None
            return n

        # We wrap in a generic retry similar to the non-streaming one if it fails at the start.
        # But once bytes are yielded, if it fails mid-stream, it breaks.
        yielded_any = False
        # Stream-side recording (§4F Phase 2b): checked ONCE per call so the
        # off path costs one import + one getenv. When on, deltas are folded
        # into `_rec_acc` as they pass through and ONE reassembled record is
        # written on clean completion (never on error/stall paths).
        try:
            from .llm_recording import recording_enabled as _rec_enabled
            _rec_on = _rec_enabled()
        except Exception:
            _rec_on = False
        _rec_acc: Dict[str, Any] = {"content": [], "reasoning": [],
                                    "tool_calls": {}, "finish": None}
        # Token accounting is NOT gated on `_rec_on` — recording is an opt-in
        # fixture-mining feature, while usage is needed on every turn. An
        # OpenAI-compatible server only emits the usage block on a stream when
        # asked, and it arrives in a FINAL chunk whose `choices` is empty.
        try:
            _so = payload.get("stream_options")
            if not isinstance(_so, dict):
                _so = {}
            _so.setdefault("include_usage", True)
            payload["stream_options"] = _so
        except Exception:  # noqa: BLE001 — a payload that rejects the key
            pass
        # One attempt per coding node, then the main model's usual two. The
        # last two attempts are therefore ALWAYS main (a node attempt appends
        # to `_tried`, and `_pick_coding_node` returns None once the pool is
        # exhausted), which is what keeps the `Max retries exceeded`
        # fall-through below unreachable.
        _max_attempts = (len(_coding_pool) + 2) if _coding_pool else 2
        for attempt in range(_max_attempts):
            node = None
            if _coding_pool and not _pool_done:
                node = _pick_coding_node()
                if node is None:
                    _pool_done = True
            client_to_use = self.http_client
            if node is not None:
                _tried.append(node)
                payload["model"] = node["model"]
                client_to_use = node["client"]
                pretty_log("Coding Compute", f"Routing request to Coding Node ({node['model']})", level="INFO", icon=Icons.TOOL_CODE)
            else:
                # ⚠ RESTORE THE CALLER'S MODEL BEFORE TOUCHING MAIN. The
                # payload is rewritten in place per node, so falling back
                # with `model` still naming a coding node is a 404 on the 35B.
                if _has_model:
                    payload["model"] = _orig_model
                else:
                    payload.pop("model", None)
                if _coding_pool and not _fallback_logged:
                    pretty_log("Coding Compute Failed", "All coding nodes failed, falling back to main upstream", level="WARNING", icon=Icons.WARN)
                    _fallback_logged = True
            # ⚠ THE CONFIGURED URL, NOT `str(client.base_url)`. `_node_slot_caps`
            # — the only source of a real capacity — is keyed on the configured
            # string, and httpx normalises its `base_url` (`.../v1` gains a
            # trailing slash, `:80` disappears, the host lowercases). Keying the
            # verdict on the normalised form would make it silently never match
            # on exactly those topologies: a guard that cannot fire.
            _base = (node["url"] if node is not None
                     else str(getattr(self, "upstream_url", "") or ""))
            # Whether the >=400 body was actually read. Both reads below are
            # bounded; this stops the SECOND one from re-paying the same
            # timeout for a body that has already proven it will not arrive.
            _body_read_ok = True
            try:
                # ⚠ THE PERMIT IS HELD ACROSS `yield`s, AND THAT IS THE POINT.
                # A node slot is occupied for the whole GENERATION, not the
                # handshake — releasing after `send()` returns is the
                # "serialise the handshake" non-fix already rejected for
                # `_main_node_lock` in the comment above this function. The
                # trade differs from that rejected one in the two ways that
                # decide it:
                #   * ACQUISITION IS BOUNDED (`wait_timeout` -> NodeSaturated),
                #     so a holder can never park anyone indefinitely;
                #   * a LEAKED permit costs one of that node's `-np` slots,
                #     where a leaked `_main_node_lock` — a 1-permit mutex
                #     shared with every non-streaming call and every embedding
                #     — wedges the whole process.
                # And the reliance on cross-`yield` finalisation is NOT NEW:
                # `resp.aclose()` below has always been in a `finally` on the
                # far side of these same yields. If an abandoned generator did
                # not finalise, this function would already be leaking httpx
                # connections. The permit rides the mechanism the stream
                # already depends on. (Measured through the real four-wrapper
                # consumer chain with a consumer `break` and no `gc.collect()`:
                # permit, `foreground_tasks` and `aclose()` all released.)
                async with _contextlib.AsyncExitStack() as _stack:
                    if node is not None:
                        _untried = max(1, len(_coding_pool) - len(_tried) + 1)
                        await _stack.enter_async_context(
                            self._node_slot(node,
                                            wait_timeout=_gate_wait(_untried)))
                    # We use stream() to keep the connection open and read chunks.
                    req = client_to_use.build_request("POST", "/v1/chat/completions", json=payload)
                    self._inflight_inc(_base)
                    _stack.callback(self._inflight_dec, _base)
                    _conc_at_send = self._own_inflight(_base)
                    resp = await client_to_use.send(req, stream=True)
                    # The aclose() MUST cover raise_for_status() too — otherwise a
                    # 4xx/5xx leaks the streamed connection (the `except` handlers
                    # never closed it), and repeated upstream errors exhaust the
                    # httpx pool (max_connections=15).
                    try:
                        _sc = getattr(resp, "status_code", None)
                        if isinstance(_sc, int) and _sc >= 400:
                            # Read the error body NOW (stream still open) so the
                            # HTTPStatusError handler's heuristic has content; httpx
                            # caches it, so the handler's aread() returns it again.
                            #
                            # ⚠ BOUNDED. This inherited the client's 1200s
                            # timeout, so an upstream that answers with
                            # headers and then nothing parked the turn for
                            # twenty minutes to decorate an error message.
                            # Losing the body only costs the "parse input"
                            # heuristic below; `raise_for_status()` still
                            # raises on the status alone.
                            try:
                                await asyncio.wait_for(
                                    resp.aread(),
                                    timeout=_STREAM_ERROR_BODY_TIMEOUT)
                            except asyncio.TimeoutError:
                                _body_read_ok = False
                                pretty_log("Upstream Stream Error",
                                           f"HTTP {_sc}: error body did not "
                                           f"arrive within "
                                           f"{_STREAM_ERROR_BODY_TIMEOUT:g}s "
                                           f"— continuing without it",
                                           level="WARNING", icon=Icons.WARN)
                        resp.raise_for_status()

                        # Per-chunk read guard (see module-level constants). The
                        # FIRST byte gets a generous budget to cover prompt prefill
                        # on large contexts / slow nodes; subsequent bytes get a
                        # tighter gap so a real mid-stream hang is still caught.
                        chunk_iter = resp.aiter_lines().__aiter__()
                        awaiting_first_byte = True
                        while True:
                            _timeout = _STREAM_FIRST_BYTE_TIMEOUT if awaiting_first_byte else _STREAM_IDLE_TIMEOUT
                            try:
                                chunk = await asyncio.wait_for(chunk_iter.__anext__(), timeout=_timeout)
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError:
                                _phase = "prefill/first token" if awaiting_first_byte else "mid-stream"
                                # ⚠ ATTRIBUTE THE WAIT BEFORE BLAMING THE
                                # UPSTREAM. Measured: 3 streams + 1 POST at a
                                # 1-slot node put 2 of the 3 streams over the
                                # first-byte budget waiting in the SERVER's
                                # queue behind OUR OWN traffic, and both
                                # printed "Upstream Stream Stall" — our
                                # over-subscription misattributed to the
                                # upstream, which is the diagnosis corruption
                                # the node gate exists to remove.
                                #
                                # A SELF-QUEUED VERDICT IS ISSUED ONLY ON
                                # PROOF: a `/props` probe must have SUCCEEDED
                                # (so the capacity is a real advertised
                                # number, never a default or a guess) and our
                                # own in-flight count must provably exceed it.
                                # Otherwise the line states the concurrency as
                                # a FACT and leaves the diagnosis open — an
                                # unknown capacity is not evidence of
                                # innocence either. `_conc_at_send` is kept
                                # because a stall that ended alone but had
                                # company for most of the wait is still ours.
                                # `:g` not `:.0f`: sub-second budgets rendered
                                # as "0s", which reads like a disabled guard.
                                _conc = max(_conc_at_send,
                                            self._own_inflight(_base))
                                _cap = self._known_slots(_base)
                                if isinstance(_cap, int) and _cap > 0 and _conc > _cap:
                                    pretty_log("Stream Stall (Self-Queued)",
                                               f"No bytes for {_timeout:g}s ({_phase}) — "
                                               f"{_conc} of OUR OWN requests were in flight "
                                               f"against a {_cap}-slot node; self-inflicted "
                                               f"queueing, not an upstream fault — aborting",
                                               level="WARNING", icon=Icons.WARN)
                                    error_data = {"error": f"Stream aborted after {_timeout:g}s without data "
                                                           f"({_phase}); {_conc} concurrent requests from this "
                                                           f"process against a {_cap}-slot node."}
                                elif _conc > 1:
                                    # ⚠ "capacity unknown" ONLY WHEN IT IS.
                                    # A first draft printed it whenever the
                                    # verdict did not fire, so 3 requests
                                    # against a PROBED 4-slot node — the case
                                    # that exonerates us — was reported as if
                                    # we had never asked. The two reasons the
                                    # verdict did not fire are opposites and
                                    # must not share a sentence.
                                    _capnote = (f"a {_cap}-slot node, within capacity"
                                                if isinstance(_cap, int) and _cap > 0
                                                else "capacity unknown")
                                    pretty_log("Upstream Stream Stall",
                                               f"No bytes for {_timeout:g}s ({_phase}) — aborting "
                                               f"({_conc} concurrent requests from this process "
                                               f"against {_base or 'upstream'}; {_capnote})",
                                               level="WARNING", icon=Icons.WARN)
                                    error_data = {"error": f"Upstream stalled ({_phase}, {_timeout:g}s without data, "
                                                           f"{_conc} concurrent requests in flight)."}
                                else:
                                    pretty_log("Upstream Stream Stall",
                                               f"No bytes for {_timeout:g}s ({_phase}) — aborting "
                                               f"(sole in-flight request)",
                                               level="WARNING", icon=Icons.WARN)
                                    error_data = {"error": f"Upstream stalled ({_phase}, {_timeout:g}s without data)."}
                                yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
                                yield b"data: [DONE]\n\n"
                                return
                            # Any line received — including SSE keepalive/blank
                            # lines — counts as activity and ends the prefill wait.
                            awaiting_first_byte = False
                            if chunk:
                                yielded_any = True
                                if _rec_on:
                                    self._stream_rec_accumulate(chunk, _rec_acc)
                                # Cheap pre-filter: only the final chunk carries
                                # usage, so the common chunk costs one substring
                                # test rather than a JSON parse. `chunk` is str on
                                # the real `aiter_lines` path but bytes on others,
                                # so the test must not assume either.
                                if b'"usage"' in chunk if isinstance(chunk, (bytes, bytearray)) \
                                        else '"usage"' in str(chunk):
                                    self._note_usage_from_sse(chunk)
                                # ⚠ NOT `f"{chunk}"` WHEN `chunk` IS BYTES. The
                                # pre-filter two lines up already says a chunk
                                # "is str on the real `aiter_lines` path but
                                # bytes on others", and then this line
                                # interpolated the bytes object into an
                                # f-string — emitting its Python REPR,
                                # `b'data: {...}'`, a frame no SSE parser
                                # accepts. Latent in production (httpx yields
                                # str) and invisible in tests, because a repr
                                # still CONTAINS the substring they assert on.
                                if isinstance(chunk, (bytes, bytearray)):
                                    yield bytes(chunk) + b"\n\n"
                                else:
                                    yield f"{chunk}\n\n".encode('utf-8')
                        if _rec_on:
                            _rec_resp = self._stream_rec_response(_rec_acc)
                            if _rec_resp is not None:
                                self._maybe_record_call(
                                    payload, _rec_resp,
                                    kind="chat_completion_stream")
                    finally:
                        await resp.aclose()
                if node is not None:
                    self.circuit_breaker.record_success(node["url"])
                return
            except NodeSaturated as e:
                # ⚠ NEVER THE NODE'S FAULT — the request never left this
                # process, so the breaker is not touched (see
                # `_is_node_fault`). Straight on to the next node, or main.
                pretty_log("Coding Node Saturated", f"{_err_text(e)} — trying next",
                           level="WARNING", icon=Icons.WARN)
                continue
            # ⚠ THE TIMEOUT FAMILY BELONGS HERE TOO. This was the only one
            # of three retry sites that omitted it — `_do_chat_completion`
            # retries `ConnectTimeout`/`PoolTimeout` with a written
            # rationale, and `get_embeddings` catches the whole
            # `TimeoutException` family. Here they fell through to the
            # generic `except Exception`, which re-raises WITHOUT emitting
            # `data: [DONE]`: measured one attempt instead of two, and a
            # truncated SSE stream (R6 lens B, MAJOR-1). This is the path
            # every user turn streams through.
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError,
                    httpx.ConnectError, httpx.ConnectTimeout,
                    httpx.PoolTimeout) as e:
                if node is not None and _is_node_fault(e):
                    self.circuit_breaker.record_failure(node["url"])
                if yielded_any:
                    # Bytes already reached the client — retrying would
                    # replay the ENTIRE completion after the partial one
                    # (duplicated/garbled text in the UI). Surface the
                    # break instead, honoring the contract in the comment
                    # above the loop.
                    pretty_log("Upstream Stream Broke", f"Mid-stream {type(e).__name__} after output started — not retrying", level="WARNING", icon=Icons.WARN)
                    error_data = {"error": f"Stream broke mid-response: {_err_text(e)}"}
                    yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"
                    return
                if attempt < _max_attempts - 1:
                    if node is not None:
                        # Moving to a DIFFERENT box — a backoff buys nothing
                        # and burns the caller's latency (the non-streaming
                        # branch does not sleep between nodes either).
                        pretty_log("Coding Node Failed", f"{node['model']}: {_node_error_detail(e)} — trying next", level="WARNING", icon=Icons.WARN)
                    else:
                        wait_time = 2
                        pretty_log("Upstream Stream Retry", f"[{attempt+1}/{_max_attempts}] {type(e).__name__}. Retrying in {wait_time}s...", icon=Icons.RETRY)
                        await asyncio.sleep(wait_time)
                else:
                    pretty_log("Upstream Stream Failed", f"Failed after {_max_attempts} attempts: {_err_text(e)}", level="ERROR", icon=Icons.FAIL)
                    # Yield an error event to the client if the stream failed to connect
                    error_data = {"error": f"Stream failed after {_max_attempts} attempts: {_err_text(e)}"}
                    yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"
                    raise
            except httpx.HTTPStatusError as e:
                # ⚠ BOUNDED HERE TOO, AND SKIPPED IF THE FIRST READ ALREADY
                # TIMED OUT. Bounding only the read above left this one
                # inheriting the client's 1200s default, so the fix moved the
                # twenty-minute park four lines down instead of removing it —
                # caught by the very test written for the first half.
                # `except Exception`, not a bare `except`: the bare form also
                # swallowed `CancelledError`, so a cancelled turn kept going.
                err_text = ""
                if _body_read_ok:
                    try:
                        err_text = (await asyncio.wait_for(
                            e.response.aread(),
                            timeout=_STREAM_ERROR_BODY_TIMEOUT)
                        ).decode('utf-8').lower()
                    except Exception:  # noqa: BLE001 — a better message, not a fault
                        pass
                if node is not None and _is_node_fault(e):
                    self.circuit_breaker.record_failure(node["url"])
                _transient = e.response.status_code >= 500 or "parse input" in err_text
                # A 4xx is a CALLER fault on the MAIN model and repeats
                # identically, so it still raises there — `handle_chat`'s
                # internal drain catches `HTTPStatusError` and runs an
                # emergency context prune on 400/"context", which only works
                # because this raises. On a POOL node a 4xx says nothing
                # about the next box: `payload["model"]` is rewritten per
                # node, so a 404 "unknown model" from one is not a verdict on
                # another — which is exactly why the non-streaming branch
                # advances on ANY node exception.
                if node is not None or _transient:
                    if attempt < _max_attempts - 1:
                        if node is not None:
                            pretty_log("Coding Node Failed", f"{node['model']}: {_node_error_detail(e)} — trying next", level="WARNING", icon=Icons.WARN)
                        else:
                            wait_time = 2
                            pretty_log("Upstream Stream Retry", f"[{attempt+1}/{_max_attempts}] HTTP {e.response.status_code} Server Glitch. Retrying in {wait_time}s...", icon=Icons.RETRY)
                            await asyncio.sleep(wait_time)
                        continue
                    pretty_log("Upstream Stream Failed", f"Failed after {_max_attempts} attempts: {_err_text(e)}", level="ERROR", icon=Icons.FAIL)
                    raise
                pretty_log("Upstream Stream Error", f"HTTP {e.response.status_code}: {err_text}", level="ERROR", icon=Icons.FAIL)
                raise
            except Exception as e:
                if node is not None and _is_node_fault(e):
                    self.circuit_breaker.record_failure(node["url"])
                pretty_log("Upstream Stream Fatal", _err_text(e), level="ERROR", icon=Icons.FAIL)
                raise

        raise Exception("Max retries exceeded")

    async def stream_chat_completion(self, payload: Dict[str, Any], use_coding: bool = False, is_background: bool = False):
        if is_background:
            await self._wait_for_foreground_clear()
            async with self._bg_queue_sem:
                async for chunk in self._do_stream_chat_completion(payload, use_coding):
                    yield chunk
        else:
            async with self._foreground_lock:
                self.foreground_tasks += 1
            try:
                async for chunk in self._do_stream_chat_completion(payload, use_coding):
                    yield chunk
            finally:
                async with self._foreground_lock:
                    self.foreground_tasks -= 1
                    if self.foreground_tasks < 0:
                        self.foreground_tasks = 0

    async def stream_openai(self, model: str, content: str, created_time: int, req_id: str):
        chunk_id = f"chatcmpl-{req_id}"
        start_chunk = {
            "id": chunk_id, "object": "chat.completion.chunk", "created": created_time,
            "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(start_chunk)}\n\n".encode('utf-8')

        for i in range(0, len(content), 15):
            slice_str = content[i:i+15]
            content_chunk = {
                "id": chunk_id, "object": "chat.completion.chunk", "created": created_time,
                "model": model, "choices": [{"index": 0, "delta": {"content": slice_str}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(content_chunk)}\n\n".encode('utf-8')
            # NOTE: previously this slept 10ms per chunk, adding ~1 second of
            # artificial latency to a 1500-char trivial-fast-path response
            # (100 chunks × 10ms). The sleep was never load-bearing — there is
            # no upstream backpressure here, we're just chunking an already-
            # complete string for SSE delivery — so it's been removed.

        stop_chunk = {
            "id": chunk_id, "object": "chat.completion.chunk", "created": created_time,
            "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(stop_chunk)}\n\n".encode('utf-8')
        yield b"data: [DONE]\n\n"