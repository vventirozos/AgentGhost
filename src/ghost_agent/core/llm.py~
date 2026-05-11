import json
import asyncio
import logging
import copy
from typing import List, Dict, Any, Optional
import httpx
from ..utils.logging import Icons, pretty_log
from ..utils.helpers import get_utc_timestamp

logger = logging.getLogger("GhostAgent")


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
    def __init__(self, upstream_url: str, tor_proxy: Optional[str] = None, swarm_nodes: Optional[list] = None, worker_nodes: Optional[list] = None, visual_nodes: Optional[list] = None, coding_nodes: Optional[list] = None, image_gen_nodes: Optional[list] = None):
        self.upstream_url = upstream_url
        limits = httpx.Limits(max_keepalive_connections=3, max_connections=15, keepalive_expiry=30.0)

        def get_proxy(url: str) -> Optional[str]:
            if not tor_proxy:
                return None
            if "127.0.0.1" in url or "localhost" in url:
                return None
            try:
                import urllib.parse
                import ipaddress
                
                # Robustly handle URLs missing the http:// scheme
                if not url.startswith("http://") and not url.startswith("https://"):
                    url = "http://" + url
                    
                hostname = urllib.parse.urlparse(url).hostname
                if hostname:
                    if hostname.endswith(".local"):
                        return None
                    ip = ipaddress.ip_address(hostname)
                    if ip.is_private or ip.is_loopback:
                        return None
            except ValueError:
                pass
            return tor_proxy.replace("socks5://", "socks5h://")
        # Determine if we need to route through Tor
        # If upstream is NOT localhost, we force Tor usage
        proxy_url = get_proxy(upstream_url)
        if proxy_url:
            pretty_log("LLM Connection", f"Routing upstream traffic via Tor ({proxy_url})", icon=Icons.SHIELD)

        self.circuit_breaker = NodeCircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)
        self.foreground_tasks = 0
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
                timeout=15.0,
            )
        except Exception as e:
            logger.debug(f"route({task}) worker call failed: {e}")
            return fallback

        try:
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return content if content else fallback
        except Exception:
            return fallback

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

    def get_swarm_node(self, target_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not getattr(self, 'swarm_clients', []):
            return None
            
        if target_model:
            target_lower = target_model.lower()
            for node in self.swarm_clients:
                if target_lower in node["model"].lower():
                    return node
                    
        node = self.swarm_clients[self._swarm_index]
        self._swarm_index = (self._swarm_index + 1) % len(self.swarm_clients)
        return node

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
                pretty_log("Image Compute", f"Routing to Image Node ({node['model']})", level="INFO", icon="🎨")
                resp = await node["client"].post("/v1/images/generations", json=payload)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt < 2:
                    pretty_log("Image Node Retry", f"Attempt {attempt+1} failed: {type(e).__name__}, retrying...", level="WARNING", icon=Icons.WARN)
                    await asyncio.sleep(2 ** attempt)
                    # Try to get next node if possible
                    node = self.get_image_gen_node()
                else:
                    raise Exception(f"Image generation failed after 3 attempts: {str(e)}")

    async def _do_chat_completion(self, payload: Dict[str, Any], use_swarm: bool = False, use_worker: bool = False, use_vision: bool = False, use_coding: bool = False, timeout: Optional[float] = None) -> Dict[str, Any]:
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
                            pretty_log(f"Vision node ({node['model']}) failed: {type(e).__name__}, trying next...", level="WARNING", icon=Icons.WARN)
                            target_model = None
                            node = self.get_vision_node(target_model)
                            continue

                pretty_log("Vision Compute Failed", "All vision nodes failed.", level="ERROR", icon=Icons.WARN)
                
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
                        
                    tried_nodes.append(node)
                    
                    pretty_log("Worker Compute", f"Routing background task to Worker Node ({node['model']})", level="INFO", icon="⚙️")
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
                        pretty_log(f"Worker node ({node['model']}) failed: {type(e).__name__}, trying next...", level="WARNING", icon=Icons.WARN)
                        target_model = None
                        node = self.get_worker_node(target_model)
                        continue
                        
                pretty_log("Worker Compute Failed", "All worker nodes failed, falling back to main upstream", level="WARNING", icon=Icons.WARN)

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
                        pretty_log(f"Coding node ({node['model']}) failed: {type(e).__name__}, trying next...", level="WARNING", icon=Icons.WARN)
                        target_model = None
                        node = self.get_coding_node(target_model)
                        continue
                        
                pretty_log("Coding Compute Failed", "All coding nodes failed, falling back to main upstream", level="WARNING", icon=Icons.WARN)

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
                        
                    tried_nodes.append(node)
                    
                    pretty_log("Edge Compute", f"Routing request to Swarm Node ({node['model']})", level="INFO", icon="⚡")
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
                        pretty_log(f"Swarm node ({node['model']}) failed: {type(e).__name__}, trying next...", level="WARNING", icon=Icons.WARN)
                        target_model = None
                        node = self.get_swarm_node(target_model)
                        continue
                        
                pretty_log("Edge Compute Failed", "All swarm nodes failed, falling back to main upstream", level="WARNING", icon=Icons.WARN)

        for attempt in range(2): 
            try:
                kwargs = {}
                if timeout is not None:
                    kwargs["timeout"] = timeout
                async with self._main_node_lock:
                    resp = await self.http_client.post("/v1/chat/completions", json=payload, **kwargs)
                resp.raise_for_status()
                return resp.json()
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

    async def chat_completion(self, payload: Dict[str, Any], use_swarm: bool = False, use_worker: bool = False, use_vision: bool = False, use_coding: bool = False, is_background: bool = False, timeout: Optional[float] = None) -> Dict[str, Any]:
        if is_background:
            # Background tasks wait for foreground to clear, but with a
            # maximum wait time to prevent indefinite starvation. After
            # 30 seconds of waiting, proceed anyway — a slightly stale
            # background result is better than no result at all.
            # Wait for foreground to clear, then acquire a semaphore
            # slot (allows up to 3 concurrent background tasks).
            waited = 0.0
            max_bg_wait = 30.0
            while waited < max_bg_wait:
                async with self._foreground_lock:
                    if self.foreground_tasks <= 0:
                        break
                await asyncio.sleep(1.0)
                waited += 1.0
            async with self._bg_queue_sem:
                return await self._do_chat_completion(payload, use_swarm, use_worker, use_vision, use_coding, timeout)
        else:
            async with self._foreground_lock:
                self.foreground_tasks += 1
            try:
                return await self._do_chat_completion(payload, use_swarm, use_worker, use_vision, use_coding, timeout)
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
        for attempt in range(2): 
            try:
                # We use stream() to keep the connection open and read chunks
                req = client_to_use.build_request("POST", "/v1/chat/completions", json=payload)
                resp = await client_to_use.send(req, stream=True)
                resp.raise_for_status()
                
                try:
                    # Per-chunk timeout — without this a stalled upstream
                    # (no bytes for minutes) holds the event-loop slot
                    # forever and keeps `foreground_tasks > 0`, which in
                    # turn parks the biological watchdog. 30s is generous
                    # for a healthy stream and small enough to recover.
                    chunk_iter = resp.aiter_lines().__aiter__()
                    while True:
                        try:
                            chunk = await asyncio.wait_for(chunk_iter.__anext__(), timeout=30.0)
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            pretty_log("Upstream Stream Stall", "No bytes for 30s — aborting", level="WARNING", icon=Icons.WARN)
                            error_data = {"error": "Upstream stalled mid-stream (30s without data)."}
                            yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
                            yield b"data: [DONE]\n\n"
                            return
                        if chunk:
                            yield f"{chunk}\n\n".encode('utf-8')
                finally:
                    await resp.aclose()
                return
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError, httpx.ConnectError) as e:
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
            waited = 0.0
            max_bg_wait = 30.0
            while waited < max_bg_wait:
                async with self._foreground_lock:
                    if self.foreground_tasks <= 0:
                        break
                await asyncio.sleep(1.0)
                waited += 1.0
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