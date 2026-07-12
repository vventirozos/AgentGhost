import json
import asyncio
import logging
import copy
import os
from typing import List, Dict, Any, Optional
import httpx
from ..utils.logging import Icons, pretty_log
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
# `Nova: ReadTimeout` lines even on the LAN. 8s absorbs ONE queued call while
# still failing fast on a genuinely sick node (the circuit breaker trips after 3
# strikes). The real fix is more worker slots (operator: bump nova's -np); this
# is the margin that keeps the value flowing until then. Losing the expansion
# only costs a slightly cruder retrieval query, never correctness.
_ROUTE_TIMEOUT_S = 8.0


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
                return True  # Allow probe request
            return False
        # half_open — allow one probe
        return True

    def record_success(self, node_url: str):
        """Record a successful request — reset the breaker."""
        state = self._get_state(node_url)
        state["failures"] = 0
        state["open_since"] = None
        state["state"] = "closed"

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



class LLMClient:
    def __init__(self, upstream_url: str, tor_proxy: Optional[str] = None, swarm_nodes: Optional[list] = None, worker_nodes: Optional[list] = None, visual_nodes: Optional[list] = None, coding_nodes: Optional[list] = None, image_gen_nodes: Optional[list] = None, critic_nodes: Optional[list] = None):
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
                    http2=False
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
                    fallback: Any = None) -> Any:
        """Route a small cognitive sub-task to the worker pool.

        `task` is a short string label (e.g. ``"VALIDATE_TOOL_ARGS"``) used
        for logging; the actual prompt is in `payload`. `fallback` is
        returned if no worker pool exists or the call fails — callers
        should always pass a sensible default so they degrade silently.
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
                timeout=_ROUTE_TIMEOUT_S,
                # NEVER re-run a routing sub-task on the main model.
                off_main_only=True,
                # The log label is the ACTUAL routed task ("decompose query",
                # "expand query", …). A hardcoded "query expansion" here made
                # a DECOMPOSE_QUERY timeout read as the anaphora expander and
                # sent the debugging down the wrong path (2026-07-12).
                task_label=str(task).replace("_", " ").lower(),
            )
        except OffMainNodeUnavailable:
            logger.debug(f"route({task}): worker pool down — using fallback")
            return fallback
        except Exception as e:
            logger.debug(f"route({task}) worker call failed: {e}")
            return fallback

        try:
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
        never raises, and each node warms with a per-slot request so all
        `-np` slots (not just one) get hot.
        """
        for pool_attr, label in (("worker_clients", "worker"),
                                  ("critic_clients", "critic")):
            clients = getattr(self, pool_attr, None) or []
            for node in clients:
                # One warmup per slot: with -np N the first call only hydrates
                # one slot; fire a few so the pool is broadly hot. Bounded so a
                # dead node can't stall startup.
                for _ in range(3):
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
                            task_label="warmup",
                        )
                    except Exception as e:  # noqa: BLE001 — warmup is best-effort
                        logger.debug("warm_up %s %s failed: %s",
                                     label, node.get("url"), e)
                        break  # node unreachable — don't hammer it
                if clients:
                    pretty_log(
                        "Node Warmup",
                        f"{label} node {node.get('model')} pre-warmed",
                        icon=Icons.NODE_WORKER,
                    )

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

    async def _do_chat_completion(self, payload: Dict[str, Any], use_swarm: bool = False, use_worker: bool = False, use_vision: bool = False, use_coding: bool = False, use_critic: bool = False, timeout: Optional[float] = None, off_main_only: bool = False, task_label: str = "") -> Dict[str, Any]:
        """
        Sends a chat completion request to the upstream LLM with robust retry logic.
        """
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
                            resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                            resp.raise_for_status()
                            self.circuit_breaker.record_success(node["url"])
                            return resp.json()
                        except Exception as e:
                            self.circuit_breaker.record_failure(node["url"])
                            pretty_log("Vision Node Failed", f"{node['model']}: {type(e).__name__} — trying next", level="WARNING", icon=Icons.WARN)
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
                        pretty_log("Worker Compute", f"{task_label or 'background task'} → Worker Node ({node['model']})", level="INFO", icon=Icons.NODE_WORKER)
                    try:
                        node_payload = payload.copy()
                        node_payload["model"] = node["model"]
                        _disable_thinking(node_payload)

                        import json

                        body_bytes = json.dumps(node_payload, ensure_ascii=True).encode('ascii', errors='ignore')

                        kwargs = {}
                        if timeout is not None:
                            kwargs["timeout"] = timeout
                        resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                        resp.raise_for_status()
                        self.circuit_breaker.record_success(node["url"])
                        return resp.json()
                    except Exception as e:
                        self.circuit_breaker.record_failure(node["url"])
                        if _quiet:
                            logger.debug("keepalive worker %s failed: %s",
                                         node.get("model"), type(e).__name__)
                        else:
                            pretty_log("Worker Node Failed", f"{node['model']}: {type(e).__name__} — trying next", level="WARNING", icon=Icons.WARN)
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

                    pretty_log("Critic Compute", f"Routing verification to Critic Node ({node['model']})", level="INFO", icon=Icons.VERIFIER_LAB)
                    try:
                        import copy as _copy, json
                        node_payload = _copy.deepcopy(payload)
                        node_payload["model"] = node["model"]

                        body_bytes = json.dumps(node_payload, ensure_ascii=True).encode('utf-8')

                        kwargs = {}
                        if timeout is not None:
                            kwargs["timeout"] = timeout
                        resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                        resp.raise_for_status()
                        self.circuit_breaker.record_success(node["url"])
                        return resp.json()
                    except Exception as e:
                        self.circuit_breaker.record_failure(node["url"])
                        pretty_log("Critic Node Failed", f"{node['model']}: {type(e).__name__} — trying next", level="WARNING", icon=Icons.WARN)
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
                        resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                        resp.raise_for_status()
                        self.circuit_breaker.record_success(node["url"])
                        return resp.json()
                    except Exception as e:
                        self.circuit_breaker.record_failure(node["url"])
                        pretty_log("Coding Node Failed", f"{node['model']}: {type(e).__name__} — trying next", level="WARNING", icon=Icons.WARN)
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
                        resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                        resp.raise_for_status()
                        self.circuit_breaker.record_success(node["url"])
                        return resp.json()
                    except Exception as e:
                        self.circuit_breaker.record_failure(node["url"])
                        pretty_log("Swarm Node Failed", f"{node['model']}: {type(e).__name__} — trying next", level="WARNING", icon=Icons.WARN)
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
                        pretty_log("Upstream Empty Body",
                                   f"HTTP {resp.status_code} with non-JSON body ({body_len} B) — retrying",
                                   level="WARNING", icon=Icons.RETRY)
                        await asyncio.sleep(2)
                        continue
                    raise RuntimeError(
                        f"Upstream returned an empty/non-JSON response "
                        f"(HTTP {resp.status_code}, {body_len} bytes) after retry. "
                        f"This typically follows a context overflow or an upstream "
                        f"restart; the request did not complete."
                    ) from je
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError, httpx.ConnectError) as e:
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
            # rare, already-bounded edge, same as proceeding after the
            # 600s ceiling.)
            targets_main_node = not (
                (use_worker and getattr(self, "worker_clients", None))
                or (use_critic and getattr(self, "critic_clients", None))
                or (use_vision and getattr(self, "vision_clients", None))
                or (use_swarm and getattr(self, "swarm_clients", None))
            )
            if targets_main_node:
                await self._wait_for_foreground_clear()
            async with self._bg_queue_sem:
                return await self._do_chat_completion(payload, use_swarm, use_worker, use_vision, use_coding, use_critic, timeout, off_main_only, task_label)
        else:
            async with self._foreground_lock:
                self.foreground_tasks += 1
            try:
                return await self._do_chat_completion(payload, use_swarm, use_worker, use_vision, use_coding, use_critic, timeout, off_main_only, task_label)
            finally:
                async with self._foreground_lock:
                    self.foreground_tasks -= 1
                    if self.foreground_tasks < 0:
                        self.foreground_tasks = 0

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
                            yield f"{chunk}\n\n".encode('utf-8')
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