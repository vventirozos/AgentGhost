"""2026-07-22 LLM-stack review — core/llm.py contention + reliability fixes.

- `_is_node_fault`: HTTP 4xx is a caller fault, must NOT trip the circuit
  breaker (a bad/oversized payload would take a healthy node out of rotation).
- `warm_up_workers`: fires its per-slot warmups CONCURRENTLY and with
  `off_main_only=True` so a dead node at boot can't burn the main slot.
- retry classification: ConnectTimeout/PoolTimeout (nothing sent) are retried.
- `targets_main_node` includes the coding pool.
"""
import asyncio
import inspect

import httpx
import pytest

from ghost_agent.core.llm import LLMClient, _is_node_fault, OffMainNodeUnavailable


def _resp(code):
    return httpx.Response(code, request=httpx.Request("POST", "http://n"))


class TestIsNodeFault:
    def test_4xx_is_not_a_node_fault(self):
        for code in (400, 404, 413, 422, 499):
            err = httpx.HTTPStatusError("x", request=None, response=_resp(code))
            assert _is_node_fault(err) is False, code

    def test_5xx_is_a_node_fault(self):
        for code in (500, 502, 503):
            err = httpx.HTTPStatusError("x", request=None, response=_resp(code))
            assert _is_node_fault(err) is True, code

    def test_timeout_and_connect_are_node_faults(self):
        assert _is_node_fault(httpx.ReadTimeout("x")) is True
        assert _is_node_fault(httpx.ConnectError("x")) is True
        assert _is_node_fault(RuntimeError("boom")) is True


class TestWarmupOffMain:
    def _client(self):
        calls = []

        async def worker_post(url, content=None, **kw):
            calls.append(kw)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]},
                request=httpx.Request("POST", "http://nova.invalid"))

        c = LLMClient(upstream_url="http://main.invalid",
                      worker_nodes=[{"url": "http://nova.invalid", "model": "Nova"}])
        c.worker_clients[0]["client"].post = worker_post
        return c, calls

    def test_warmup_passes_off_main_only(self):
        # The main client must NEVER be hit during warmup — assert via the
        # source that off_main_only rides the warmup call (behavioral proof is
        # the dead-node test below).
        src = inspect.getsource(LLMClient.warm_up_workers)
        assert "off_main_only=True" in src
        assert "asyncio.gather" in src  # concurrent, not serial

    def test_dead_worker_never_falls_back_to_main(self):
        c, _ = self._client()
        main_hits = []

        async def dead_worker(*a, **kw):
            raise httpx.ConnectError("nova down")

        async def main_post(*a, **kw):
            main_hits.append(1)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "x"}}]},
                request=httpx.Request("POST", "http://main.invalid"))

        c.worker_clients[0]["client"].post = dead_worker
        c.http_client.post = main_post
        asyncio.run(c.warm_up_workers())  # must not raise, must not hang
        assert main_hits == []  # off_main_only kept warmup off the main slot


class TestRetryClassification:
    def test_connect_and_pool_timeouts_are_retried(self):
        # These mean the request was never put on the wire → safe to retry.
        src = inspect.getsource(LLMClient._do_chat_completion)
        # The retry-except tuple must include the connect/pool timeouts.
        assert "httpx.ConnectTimeout" in src
        assert "httpx.PoolTimeout" in src
        # ReadTimeout must NOT be retried (may have executed upstream).
        retry_line = next(l for l in src.splitlines()
                          if "httpx.RemoteProtocolError" in l and "except" in l)
        assert "ReadTimeout" not in retry_line


class TestTargetsMainNodeCoding:
    def test_coding_pool_counts_as_off_main(self):
        # A background call bound for a configured coding pool must not be
        # classified main-targeted (which would park it 600s for a slot it
        # never uses).
        src = inspect.getsource(LLMClient.chat_completion)
        assert 'use_coding and getattr(self, "coding_clients"' in src
