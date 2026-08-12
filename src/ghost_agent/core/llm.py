import json
import asyncio
import logging
import copy
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import httpx
from ..utils.logging import (Icons, pretty_log, request_id_context,
                             verify_purpose_context)
from ..utils.helpers import get_utc_timestamp

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
_STREAM_FIRST_BYTE_TIMEOUT = float(os.getenv("GHOST_STREAM_FIRST_BYTE_TIMEOUT", "180"))
_STREAM_IDLE_TIMEOUT = float(os.getenv("GHOST_STREAM_IDLE_TIMEOUT", "60"))


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
    ``cooldown_seconds``. After cooldown, the next request is a probe —
    if it succeeds, the breaker resets; if it fails, another cooldown starts.

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
        # half_open — allow one probe
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
        state["failures"] += 1
        if state["failures"] >= self.failure_threshold:
            state["state"] = "open"
            state["open_since"] = time.time()
            logger.warning(f"Circuit breaker OPEN for node {node_url} after {state['failures']} consecutive failures (cooldown {self.cooldown_seconds}s)")

    def get_status(self) -> dict:
        """Return the current state of all tracked nodes."""
        return {url: dict(s) for url, s in self._states.items()}


class RoutingTask:
    """Stable string labels for `LLMClient.route()` calls. Centralised here
    so a future routing-model swap or instrumentation hook lands in one
    place. These are deliberately strings (not Enum members) so they're
    cheap to log and serialise."""
    VALIDATE_TOOL_ARGS = "VALIDATE_TOOL_ARGS"
    EXPAND_QUERY = "EXPAND_QUERY"
    CLASSIFY_INTENT = "CLASSIFY_INTENT"
    SCORE_RELEVANCE = "SCORE_RELEVANCE"
    REPAIR_JSON = "REPAIR_JSON"
    CLASSIFY_FAILURE = "CLASSIFY_FAILURE"
    DISTILL_PATTERN = "DISTILL_PATTERN"


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
                    timeout: Optional[float] = None) -> Any:
        """Route a small cognitive sub-task to the worker pool.

        `task` is a short string label (e.g. ``"VALIDATE_TOOL_ARGS"``) used
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
                # NEVER re-run a routing sub-task on the main model.
                off_main_only=True,
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
            # Counted here too, not just in chat_completion: the worker route
            # serves verify / decompose / classification, so omitting it would
            # under-report the turn's real cost — silently, and in the
            # direction that flatters it.
            self._note_usage(data)
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            self._maybe_record_call(payload, content, kind="route",
                                    task=str(task))
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
        never raises, and each node warms with a per-slot request so all
        `-np` slots (not just one) get hot.
        """
        for pool_attr, label in (("worker_clients", "worker"),
                                  ("critic_clients", "critic")):
            clients = getattr(self, pool_attr, None) or []
            for node in clients:
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
                results = await asyncio.gather(
                    *[self.chat_completion(
                        dict(payload), use_worker=(label == "worker"),
                        use_critic=(label == "critic"),
                        is_background=True, timeout=30.0,
                        off_main_only=True, task_label="warmup")
                      for _ in range(3)],
                    return_exceptions=True)
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
            for pool_attr, label in (("worker_clients", "worker"),
                                     ("critic_clients", "critic")):
                for node in getattr(self, pool_attr, None) or []:
                    url = node.get("url")
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
        await self.http_client.aclose()
        for node in getattr(self, 'swarm_clients', []):
            await node["client"].aclose()
        for node in getattr(self, 'worker_clients', []):
            await node["client"].aclose()
        for node in getattr(self, 'vision_clients', []):
            await node["client"].aclose()
        for node in getattr(self, 'coding_clients', []):
            await node["client"].aclose()
        for node in getattr(self, 'image_gen_clients', []):
            await node["client"].aclose()
        for node in getattr(self, 'critic_clients', []):
            await node["client"].aclose()

    def get_swarm_node(self, target_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
        # All nodes tripped — return the current one anyway (call fails, cooldown extends).
        return self.swarm_clients[self._swarm_index]

    def get_vision_node(self, target_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
        return vision_clients[0]

    def get_worker_node(self, target_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
        return worker_clients[0]

    def get_coding_node(self, target_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
        return coding_clients[0]

    def get_critic_node(self, target_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
        return critic_clients[0]

    def get_image_gen_node(self, target_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
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

        for attempt in range(3):
            try:
                pretty_log("Image Compute", f"Routing to Image Node ({node['model']})", level="INFO", icon=Icons.IMAGE_GEN)
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
                if node.get("url"):
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
                    raise Exception(f"Image generation failed after 3 attempts: {str(e)}")

    async def _node_capacity(self, node: Dict[str, Any]) -> int:
        """How many requests this node can genuinely serve at once.

        Read from the node's OWN `/props` (`total_slots`) rather than
        configured by hand: the number that matters is llama-server's `-np`,
        and a hand-set copy drifts the moment the operator restarts a node with
        different flags. Probed once per node, cached, and never retried in the
        hot path — a probe failure falls back to the conservative default
        instead of blocking dispatch.
        """
        url = node.get("url") or ""
        cap = self._node_slot_caps.get(url)
        if cap is not None:
            return cap
        cap = self._node_slot_default
        try:
            client = node.get("client")
            if client is not None:
                r = await client.get("/props", timeout=5.0)
                r.raise_for_status()
                data = r.json() or {}
                raw = data.get("total_slots") or data.get("n_parallel")
                if isinstance(raw, int) and raw > 0:
                    cap = raw
                    pretty_log("Node Capacity",
                               f"{node.get('model')} advertises {cap} slot(s) "
                               f"— per-node gate sized to match",
                               level="INFO", icon=Icons.NODE_WORKER)
        except Exception as e:                                  # noqa: BLE001
            logger.debug("node capacity probe failed for %s (%s) — using "
                         "default %d", url, type(e).__name__, cap)
        self._node_slot_caps[url] = cap
        return cap

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
        sem = self._node_slots.get(url)
        if sem is None:
            async with self._node_slots_lock:
                sem = self._node_slots.get(url)          # re-check under lock
                if sem is None:
                    cap = await self._node_capacity(node)
                    sem = asyncio.Semaphore(cap)
                    self._node_slots[url] = sem
        try:
            await asyncio.wait_for(sem.acquire(), timeout=wait_timeout)
        except asyncio.TimeoutError:
            raise NodeSaturated(
                f"no free slot on {node.get('model') or url} within "
                f"{wait_timeout:.0f}s (cap {self._node_slot_caps.get(url)})")
        try:
            yield
        finally:
            sem.release()

    async def _do_chat_completion(self, payload: Dict[str, Any], use_swarm: bool = False, use_worker: bool = False, use_vision: bool = False, use_coding: bool = False, use_critic: bool = False, timeout: Optional[float] = None, off_main_only: bool = False, task_label: str = "") -> Dict[str, Any]:
        """
        Sends a chat completion request to the upstream LLM with robust retry logic.
        """
        # How long a caller will queue for a node permit before giving up.
        # Generous relative to a request, because WAITING is the desired
        # behaviour — the alternative it replaces is not "go faster", it is
        # "flood the node, time out anyway, and take it out of rotation for
        # 60s". Bounded so a wedged node cannot park callers forever.
        _slot_wait = float(os.getenv("GHOST_NODE_SLOT_WAIT_S", "90") or 90)
        # ⚠ THE HEALTH PROBE MUST NOT QUEUE BEHIND THE TRAFFIC IT IS WATCHING.
        # `keepalive_workers()` is what prints "node X stopped answering"; if
        # its ping waits on the same permits as a research fan-out, a BUSY node
        # reports as a DEAD one — the identical false alarm this change exists
        # to remove, reintroduced one layer down. It skips the gate entirely:
        # one extra in-flight request against a 4-slot node is a rounding
        # error, and an unstarvable probe is the whole point of a probe.
        if task_label == "keepalive":
            _slot_wait = None
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
        # Heartbeat traffic logs transitions, not ticks (see keepalive_workers)
        # — its per-ping routing/failure lines stay at debug.
        _quiet = task_label == "keepalive"
        if use_vision:
            if getattr(self, 'vision_clients', None):
                target_model = payload.get("model")
                tried_nodes = []

                node = self.get_vision_node(target_model)

                if node:
                    for _ in range(len(self.vision_clients)):
                        if not node:
                            break

                        if node in tried_nodes:
                            target_model = None
                            node = self.get_vision_node(target_model)

                        loop_breaker = 0
                        while node in tried_nodes and loop_breaker < len(self.vision_clients):
                            node = self.get_vision_node(None)
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
                            if timeout is not None:
                                kwargs["timeout"] = timeout
                            async with self._node_slot(node, wait_timeout=_slot_wait):
                                resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                            resp.raise_for_status()
                            self.circuit_breaker.record_success(node["url"])
                            return resp.json()
                        except Exception as e:
                            if _is_node_fault(e):
                                self.circuit_breaker.record_failure(node["url"])
                            pretty_log("Vision Node Failed", f"{node['model']}: {_node_error_detail(e)} — trying next", level="WARNING", icon=Icons.WARN)
                            target_model = None
                            node = self.get_vision_node(target_model)
                            continue

                pretty_log("Vision Compute Failed", "All vision nodes failed.", level="ERROR", icon=Icons.FAIL)

            raise Exception("Vision analysis failed: The dedicated vision node is offline or returned an error, and the main upstream model does not support image inputs.")

        if use_worker and getattr(self, 'worker_clients', None):
            target_model = payload.get("model")
            tried_nodes = []

            node = self.get_worker_node(target_model)

            if node:
                for _ in range(len(self.worker_clients)):
                    if not node:
                        break

                    if node in tried_nodes:
                        target_model = None
                        node = self.get_worker_node(target_model)

                    loop_breaker = 0
                    while node in tried_nodes and loop_breaker < len(self.worker_clients):
                        node = self.get_worker_node(None)
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
                        if timeout is not None:
                            kwargs["timeout"] = timeout
                        async with self._node_slot(node, wait_timeout=_slot_wait):
                            resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                        resp.raise_for_status()
                        self.circuit_breaker.record_success(node["url"])
                        return resp.json()
                    except Exception as e:
                        if _is_node_fault(e):
                            self.circuit_breaker.record_failure(node["url"])
                        if _quiet:
                            logger.debug("keepalive worker %s failed: %s",
                                         node.get("model"), type(e).__name__)
                        else:
                            pretty_log("Worker Node Failed", f"{node['model']}: {_node_error_detail(e)} — trying next", level="WARNING", icon=Icons.WARN)
                        target_model = None
                        node = self.get_worker_node(target_model)
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
            target_model = payload.get("model")
            tried_nodes = []

            node = self.get_critic_node(target_model)

            if node:
                for _ in range(len(self.critic_clients)):
                    if not node:
                        break

                    if node in tried_nodes:
                        target_model = None
                        node = self.get_critic_node(target_model)

                    loop_breaker = 0
                    while node in tried_nodes and loop_breaker < len(self.critic_clients):
                        node = self.get_critic_node(None)
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
                        if timeout is not None:
                            kwargs["timeout"] = timeout
                        async with self._node_slot(node, wait_timeout=_slot_wait):
                            resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                        resp.raise_for_status()
                        self.circuit_breaker.record_success(node["url"])
                        return resp.json()
                    except Exception as e:
                        if _is_node_fault(e):
                            self.circuit_breaker.record_failure(node["url"])
                        pretty_log("Critic Node Failed", f"{node['model']}: {_node_error_detail(e)} — trying next", level="WARNING", icon=Icons.WARN)
                        target_model = None
                        node = self.get_critic_node(target_model)
                        continue

                pretty_log("Critic Compute Failed", "All critic nodes failed, falling back to main upstream", level="WARNING", icon=Icons.WARN)
                fell_back_from_node = True

        elif use_coding and getattr(self, 'coding_clients', None):
            target_model = payload.get("model")
            tried_nodes = []

            node = self.get_coding_node(target_model)

            if node:
                for _ in range(len(self.coding_clients)):
                    if not node:
                        break

                    if node in tried_nodes:
                        target_model = None
                        node = self.get_coding_node(target_model)

                    loop_breaker = 0
                    while node in tried_nodes and loop_breaker < len(self.coding_clients):
                        node = self.get_coding_node(None)
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
                        if timeout is not None:
                            kwargs["timeout"] = timeout
                        async with self._node_slot(node, wait_timeout=_slot_wait):
                            resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                        resp.raise_for_status()
                        self.circuit_breaker.record_success(node["url"])
                        return resp.json()
                    except Exception as e:
                        if _is_node_fault(e):
                            self.circuit_breaker.record_failure(node["url"])
                        pretty_log("Coding Node Failed", f"{node['model']}: {_node_error_detail(e)} — trying next", level="WARNING", icon=Icons.WARN)
                        target_model = None
                        node = self.get_coding_node(target_model)
                        continue

                pretty_log("Coding Compute Failed", "All coding nodes failed, falling back to main upstream", level="WARNING", icon=Icons.WARN)
                fell_back_from_node = True

        elif use_swarm and self.swarm_clients:
            target_model = payload.get("model")
            tried_nodes = []

            node = self.get_swarm_node(target_model)

            if node:
                for _ in range(len(self.swarm_clients)):
                    if not node:
                        break

                    if node in tried_nodes:
                        target_model = None
                        node = self.get_swarm_node(target_model)

                    loop_breaker = 0
                    while node in tried_nodes and loop_breaker < len(self.swarm_clients):
                        node = self.get_swarm_node(None)
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
                        if timeout is not None:
                            kwargs["timeout"] = timeout
                        async with self._node_slot(node, wait_timeout=_slot_wait):
                            resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                        resp.raise_for_status()
                        self.circuit_breaker.record_success(node["url"])
                        return resp.json()
                    except Exception as e:
                        if _is_node_fault(e):
                            self.circuit_breaker.record_failure(node["url"])
                        pretty_log("Swarm Node Failed", f"{node['model']}: {_node_error_detail(e)} — trying next", level="WARNING", icon=Icons.WARN)
                        target_model = None
                        node = self.get_swarm_node(target_model)
                        continue

                pretty_log("Edge Compute Failed", "All swarm nodes failed, falling back to main upstream", level="WARNING", icon=Icons.WARN)
                fell_back_from_node = True

        # ---- main-upstream fallback -------------------------------------
        # Reached either because no off-main pool was requested/configured, or
        # because every node in the requested pool FAILED.
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
            # (route() uses 6s; measured 0.5s on the worker). The main model is
            # slower BY CONSTRUCTION — a 35B answering a real prompt takes tens
            # of seconds — so handing it the worker's budget guarantees a
            # ReadTimeout. Observed live: a worker hiccup produced
            #   worker node failed  Nova: ReadTimeout      (at the 6s budget)
            #   falling back to main upstream
            #   upstream fatal      ReadTimeout('')        (6s later — the 35B)
            # i.e. one slow worker call turned into a HARD upstream error. Let
            # the main client use its own (1200s) default instead.
            timeout = None
        elif timeout is not None and (
                use_worker or use_critic or use_coding or use_swarm):
            # A pool was REQUESTED but none is configured, so the pool branch
            # was skipped entirely (fell_back_from_node stayed False) and we
            # are about to run on the MAIN model carrying a timeout that was
            # sized for a small, fast off-main node. Same hazard as the
            # fell_back_from_node reset above — a 6s route budget on the 35B is
            # a guaranteed ReadTimeout — so drop it and let the main client use
            # its own default. (use_vision has no such path: it raises above.)
            timeout = None

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
                    return resp.json()
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
                    pretty_log("Upstream Failed", f"Failed after 2 attempts: {str(e)}", level="ERROR", icon=Icons.FAIL)
                    raise
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 or "parse input" in e.response.text.lower():
                    if attempt < 1:
                        wait_time = 2
                        pretty_log("Upstream Retry", f"[{attempt+1}/2] HTTP {e.response.status_code} Server Glitch. Retrying in {wait_time}s...", icon=Icons.RETRY)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        pretty_log("Upstream Failed", f"Failed after 2 attempts: {str(e)}", level="ERROR", icon=Icons.FAIL)
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

    async def chat_completion(self, payload: Dict[str, Any], use_swarm: bool = False, use_worker: bool = False, use_vision: bool = False, use_coding: bool = False, use_critic: bool = False, is_background: bool = False, timeout: Optional[float] = None, off_main_only: bool = False, task_label: str = "") -> Dict[str, Any]:
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
            targets_main_node = not (
                (use_worker and getattr(self, "worker_clients", None))
                or (use_critic and getattr(self, "critic_clients", None))
                or (use_vision and getattr(self, "vision_clients", None))
                or (use_swarm and getattr(self, "swarm_clients", None))
                or (use_coding and getattr(self, "coding_clients", None))
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
                _result = await self._do_chat_completion(payload, use_swarm, use_worker, use_vision, use_coding, use_critic, timeout, off_main_only, task_label)
                self._note_usage(_result)
                self._maybe_record_call(payload, _result,
                                        use_worker=use_worker,
                                        use_vision=use_vision,
                                        use_critic=use_critic,
                                        task_label=task_label,
                                        background=True)
                return _result
        else:
            async with self._foreground_lock:
                self.foreground_tasks += 1
            try:
                _result = await self._do_chat_completion(payload, use_swarm, use_worker, use_vision, use_coding, use_critic, timeout, off_main_only, task_label)
                self._note_usage(_result)
                self._maybe_record_call(payload, _result,
                                        use_worker=use_worker,
                                        use_vision=use_vision,
                                        use_critic=use_critic,
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
        Tolerant by contract: any unparseable line is ignored."""
        if not line or not line.startswith("data:"):
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
                async with self._main_node_lock:
                    resp = await self.http_client.post("/v1/embeddings", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return [item["embedding"] for item in data["data"]]
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError, httpx.ConnectError) as e:
                if attempt < 1:
                    wait_time = 2
                    await asyncio.sleep(wait_time)
                else:
                    pretty_log("Embedding Failed", f"Failed after 2 attempts: {str(e)}", level="ERROR", icon=Icons.FAIL)
                    raise
            except Exception as e:
                pretty_log("Embedding Fatal", str(e), level="ERROR", icon=Icons.FAIL)
                raise

        raise Exception("Max retries exceeded")

    async def _do_stream_chat_completion(self, payload: Dict[str, Any], use_coding: bool = False):
        """
        Streams a chat completion request from the upstream LLM directly to the client.
        """
        import copy as _copy
        payload = _copy.deepcopy(payload)
        payload["stream"] = True
        # Mirror the non-streaming path: ask llama.cpp to reuse any
        # matching prefix in its KV cache. See _do_chat_completion for
        # rationale.
        payload.setdefault("cache_prompt", True)

        client_to_use = self.http_client
        if use_coding and getattr(self, 'coding_clients', None):
            node = self.get_coding_node(payload.get("model"))
            if node:
                payload["model"] = node["model"]
                client_to_use = node["client"]
                pretty_log("Coding Compute", f"Routing request to Coding Node ({node['model']})", level="INFO", icon=Icons.TOOL_CODE)

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
        for attempt in range(2):
            try:
                # We use stream() to keep the connection open and read chunks.
                req = client_to_use.build_request("POST", "/v1/chat/completions", json=payload)
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
                        await resp.aread()
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
                            pretty_log("Upstream Stream Stall", f"No bytes for {_timeout:.0f}s ({_phase}) — aborting", level="WARNING", icon=Icons.WARN)
                            error_data = {"error": f"Upstream stalled ({_phase}, {_timeout:.0f}s without data)."}
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
                            yield f"{chunk}\n\n".encode('utf-8')
                    if _rec_on:
                        _rec_resp = self._stream_rec_response(_rec_acc)
                        if _rec_resp is not None:
                            self._maybe_record_call(
                                payload, _rec_resp,
                                kind="chat_completion_stream")
                finally:
                    await resp.aclose()
                return
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError, httpx.ConnectError) as e:
                if yielded_any:
                    # Bytes already reached the client — retrying would
                    # replay the ENTIRE completion after the partial one
                    # (duplicated/garbled text in the UI). Surface the
                    # break instead, honoring the contract in the comment
                    # above the loop.
                    pretty_log("Upstream Stream Broke", f"Mid-stream {type(e).__name__} after output started — not retrying", level="WARNING", icon=Icons.WARN)
                    error_data = {"error": f"Stream broke mid-response: {str(e)}"}
                    yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"
                    return
                if attempt < 1:
                    wait_time = 2
                    pretty_log("Upstream Stream Retry", f"[{attempt+1}/2] {type(e).__name__}. Retrying in {wait_time}s...", icon=Icons.RETRY)
                    await asyncio.sleep(wait_time)
                else:
                    pretty_log("Upstream Stream Failed", f"Failed after 2 attempts: {str(e)}", level="ERROR", icon=Icons.FAIL)
                    # Yield an error event to the client if the stream failed to connect
                    error_data = {"error": f"Stream failed after 2 attempts: {str(e)}"}
                    yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"
                    raise
            except httpx.HTTPStatusError as e:
                err_text = ""
                try: err_text = (await e.response.aread()).decode('utf-8').lower()
                except: pass
                if e.response.status_code >= 500 or "parse input" in err_text:
                    if attempt < 1:
                        wait_time = 2
                        pretty_log("Upstream Stream Retry", f"[{attempt+1}/2] HTTP {e.response.status_code} Server Glitch. Retrying in {wait_time}s...", icon=Icons.RETRY)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        pretty_log("Upstream Stream Failed", f"Failed after 2 attempts: {str(e)}", level="ERROR", icon=Icons.FAIL)
                        raise
                pretty_log("Upstream Stream Error", f"HTTP {e.response.status_code}: {err_text}", level="ERROR", icon=Icons.FAIL)
                raise
            except Exception as e:
                pretty_log("Upstream Stream Fatal", str(e), level="ERROR", icon=Icons.FAIL)
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