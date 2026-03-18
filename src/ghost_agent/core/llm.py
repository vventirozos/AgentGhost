import json
import asyncio
import logging
import copy
from typing import List, Dict, Any, Optional
import httpx
from ..utils.logging import Icons, pretty_log
from ..utils.helpers import get_utc_timestamp

logger = logging.getLogger("GhostAgent")



class LLMClient:
    def __init__(self, upstream_url: str, tor_proxy: Optional[str] = None, swarm_nodes: Optional[list] = None, worker_nodes: Optional[list] = None, visual_nodes: Optional[list] = None, coding_nodes: Optional[list] = None, image_gen_nodes: Optional[list] = None):
        self.upstream_url = upstream_url
        limits = httpx.Limits(max_keepalive_connections=0, max_connections=15, keepalive_expiry=0.0)
        
        def get_proxy(url: str) -> Optional[str]:
            if not tor_proxy:
                return None
            if "127.0.0.1" in url or "localhost" in url:
                return None
            try:
                import urllib.parse
                import ipaddress
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

        self.foreground_tasks = 0
        self._bg_queue_lock = asyncio.Lock()
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
                if target_lower in node["model"].lower():
                    return node
                    
        if not hasattr(self, '_vision_index'):
            self._vision_index = 0
            
        node = vision_clients[self._vision_index]
        self._vision_index = (self._vision_index + 1) % len(vision_clients)
        return node

    def get_worker_node(self, target_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        worker_clients = getattr(self, 'worker_clients', [])
        if not worker_clients:
            return None
            
        if target_model:
            target_lower = target_model.lower()
            for node in worker_clients:
                if target_lower in node["model"].lower():
                    return node
                    
        if not hasattr(self, '_worker_index'):
            self._worker_index = 0
            
        node = worker_clients[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(worker_clients)
        return node

    def get_coding_node(self, target_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        coding_clients = getattr(self, 'coding_clients', [])
        if not coding_clients:
            return None
            
        if target_model:
            target_lower = target_model.lower()
            for node in coding_clients:
                if target_lower in node["model"].lower():
                    return node
                    
        if not hasattr(self, '_coding_index'):
            self._coding_index = 0
            
        node = coding_clients[self._coding_index]
        self._coding_index = (self._coding_index + 1) % len(coding_clients)
        return node

    def get_image_gen_node(self, target_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        image_gen_clients = getattr(self, 'image_gen_clients', [])
        if not image_gen_clients:
            return None
            
        if target_model:
            target_lower = target_model.lower()
            for node in image_gen_clients:
                if target_lower in node["model"].lower():
                    return node
                    
        if not hasattr(self, '_image_gen_index'):
            self._image_gen_index = 0
            
        node = image_gen_clients[self._image_gen_index]
        self._image_gen_index = (self._image_gen_index + 1) % len(image_gen_clients)
        return node

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
                            return resp.json()
                        except Exception as e:
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
                        return resp.json()
                    except Exception as e:
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
                        
                    tried_nodes.append(node)
                    
                    pretty_log("Coding Compute", f"Routing request to Coding Node ({node['model']})", level="INFO", icon=Icons.TOOL_CODE)
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
                        return resp.json()
                    except Exception as e:
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
                        node_payload = payload.copy()
                        node_payload["model"] = node["model"]
                        
                        import json
                        
                        body_bytes = json.dumps(node_payload, ensure_ascii=True).encode('ascii', errors='ignore')
                        
                        kwargs = {}
                        if timeout is not None:
                            kwargs["timeout"] = timeout
                        resp = await node["client"].post("/v1/chat/completions", content=body_bytes, headers={"Content-Type": "application/json", "Connection": "close"}, **kwargs)
                        resp.raise_for_status()
                        return resp.json()
                    except Exception as e:
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
            async with self._bg_queue_lock:
                while self.foreground_tasks > 0:
                    import asyncio
                    await asyncio.sleep(1.0)
                return await self._do_chat_completion(payload, use_swarm, use_worker, use_vision, use_coding, timeout)
        else:
            self.foreground_tasks += 1
            try:
                return await self._do_chat_completion(payload, use_swarm, use_worker, use_vision, use_coding, timeout)
            finally:
                self.foreground_tasks -= 1

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Fetches embeddings from the upstream LLM with robust retry logic.
        """
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
        payload["stream"] = True
        
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
                    async for chunk in resp.aiter_lines():
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
            async with self._bg_queue_lock:
                while self.foreground_tasks > 0:
                    import asyncio
                    await asyncio.sleep(1.0)
                async for chunk in self._do_stream_chat_completion(payload, use_coding):
                    yield chunk
        else:
            self.foreground_tasks += 1
            try:
                async for chunk in self._do_stream_chat_completion(payload, use_coding):
                    yield chunk
            finally:
                self.foreground_tasks -= 1

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
            await asyncio.sleep(0.01)

        stop_chunk = {
            "id": chunk_id, "object": "chat.completion.chunk", "created": created_time,
            "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(stop_chunk)}\n\n".encode('utf-8')
        yield b"data: [DONE]\n\n"