"""Regression: a NODE-sized timeout must never be applied to the MAIN model
(found live 2026-07-11 from the operator's trace).

    +274s worker compute     Routing background task to Worker Node (Nova)
    +280s worker node failed Nova: ReadTimeout        <- the 6s worker budget
    +280s worker compute fa… falling back to main upstream
    +286s upstream fatal     ReadTimeout('')          <- the 35B, ALSO at 6s

`_do_chat_completion` applied the caller's `timeout` to the main-upstream
fallback too. That timeout was sized for a small, fast worker (route() uses
6s; measured 0.5s on the node), but the main model is slower BY CONSTRUCTION —
so one slow worker call turned into a HARD `upstream fatal` error. The same
shape appeared at 15s before the route timeout was tightened, so this predates
that change.

Two fixes:
  * a node timeout is dropped when falling back to main;
  * `route()` passes `off_main_only=True` — it must NEVER re-run a routing
    sub-task on the single main slot (its own docstring said so; only the
    no-pool case enforced it).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
from pathlib import Path

import httpx
import pytest

from ghost_agent.core.llm import (
    LLMClient, OffMainNodeUnavailable, _ROUTE_TIMEOUT_S,
)


def _client(worker_ok: bool, main_capture: dict, main_ok: bool = True):
    """LLMClient whose worker node fails (or not) and whose main upstream
    records the timeout it was handed."""
    c = LLMClient(upstream_url="http://main.invalid",
                  worker_nodes=[{"url": "http://worker.invalid",
                                 "model": "W"}])

    async def worker_post(*a, **kw):
        if worker_ok:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "FROM_WORKER"}}]},
                request=httpx.Request("POST", "http://worker.invalid"))
        raise httpx.ReadTimeout("worker slow")

    async def main_post(*a, **kw):
        main_capture["called"] = True
        main_capture["timeout"] = kw.get("timeout", "<unset>")
        if not main_ok:
            raise httpx.ReadTimeout("main slow")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "FROM_MAIN"}}]},
            request=httpx.Request("POST", "http://main.invalid"))

    c.worker_clients[0]["client"].post = worker_post
    c.http_client.post = main_post
    return c


def _content(res):
    return res["choices"][0]["message"]["content"]


# ══════════════════════════════════════════════════════════════════════
# The node timeout must not leak onto the main model
# ══════════════════════════════════════════════════════════════════════

class TestMainFallbackTimeout:
    def test_worker_failure_does_not_hand_its_timeout_to_main(self):
        cap = {}
        c = _client(worker_ok=False, main_capture=cap)
        res = asyncio.run(c.chat_completion(
            {"model": "m", "messages": []}, use_worker=True, timeout=6.0))
        assert _content(res) == "FROM_MAIN"      # fallback still happens…
        assert cap["called"] is True
        # …but WITHOUT the 6s worker budget, which would kill the 35B.
        assert cap["timeout"] == "<unset>"

    def test_worker_success_never_touches_main(self):
        cap = {}
        c = _client(worker_ok=True, main_capture=cap)
        res = asyncio.run(c.chat_completion(
            {"model": "m", "messages": []}, use_worker=True, timeout=6.0))
        assert _content(res) == "FROM_WORKER"
        assert cap == {}

    def test_direct_main_call_still_honours_its_timeout(self):
        # No node pool requested ⇒ the caller's timeout is for the main model
        # and must be applied. (Don't over-correct the fix.)
        cap = {}
        c = _client(worker_ok=True, main_capture=cap)
        asyncio.run(c.chat_completion(
            {"model": "m", "messages": []}, timeout=90.0))
        assert cap["timeout"] == 90.0


# ══════════════════════════════════════════════════════════════════════
# route() must never re-run on the main model
# ══════════════════════════════════════════════════════════════════════

class TestRouteNeverHitsMain:
    def test_worker_down_returns_fallback_without_calling_main(self):
        cap = {}
        c = _client(worker_ok=False, main_capture=cap)
        out = asyncio.run(c.route(
            "EXPAND", {"messages": [{"role": "user", "content": "x"}]},
            fallback="LEGACY"))
        assert out == "LEGACY"
        assert cap == {}, "route() must NOT fall back to the main model"

    def test_worker_up_returns_the_worker_answer(self):
        cap = {}
        c = _client(worker_ok=True, main_capture=cap)
        out = asyncio.run(c.route(
            "EXPAND", {"messages": [{"role": "user", "content": "x"}]},
            fallback="LEGACY"))
        assert out == "FROM_WORKER"
        assert cap == {}

    def test_no_worker_pool_returns_fallback(self):
        c = LLMClient(upstream_url="http://main.invalid")
        out = asyncio.run(c.route("EXPAND", {"messages": []},
                                  fallback="LEGACY"))
        assert out == "LEGACY"

    def test_route_timeout_is_short(self):
        assert _ROUTE_TIMEOUT_S <= 12.0


# ══════════════════════════════════════════════════════════════════════
# off_main_only plumbing
# ══════════════════════════════════════════════════════════════════════

class TestOffMainOnly:
    def test_raises_when_all_nodes_fail(self):
        cap = {}
        c = _client(worker_ok=False, main_capture=cap)
        with pytest.raises(OffMainNodeUnavailable):
            asyncio.run(c.chat_completion(
                {"model": "m", "messages": []},
                use_worker=True, off_main_only=True))
        assert cap == {}

    def test_default_still_falls_back_to_main(self):
        # off_main_only defaults False — existing callers keep their fallback.
        cap = {}
        c = _client(worker_ok=False, main_capture=cap)
        res = asyncio.run(c.chat_completion(
            {"model": "m", "messages": []}, use_worker=True))
        assert _content(res) == "FROM_MAIN"

    def test_fallback_flag_is_not_shared_across_concurrent_calls(self):
        """`fell_back_from_node` must be a LOCAL. As instance state, a failing
        worker call would poison a concurrent main-only call's timeout."""
        src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "core" / "llm.py").read_text()
        assert "fell_back_from_node = False" in src
        assert "self._fell_back_from_node" not in src

        async def both():
            cap_a, cap_b = {}, {}
            c1 = _client(worker_ok=False, main_capture=cap_a)   # fails over
            c2 = _client(worker_ok=True, main_capture=cap_b)
            await asyncio.gather(
                c1.chat_completion({"model": "m", "messages": []},
                                   use_worker=True, timeout=6.0),
                c2.chat_completion({"model": "m", "messages": []},
                                   timeout=90.0),
            )
            return cap_a, cap_b

        cap_a, cap_b = asyncio.run(both())
        assert cap_a["timeout"] == "<unset>"   # node fallback: dropped
        assert cap_b["timeout"] == 90.0        # direct main call: preserved
