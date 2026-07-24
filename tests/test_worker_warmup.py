"""Worker-node pre-warm + route-timeout sizing (2026-07-12).

Why: the `Nova: ReadTimeout` lines are a NETWORK-PATH warmup issue, not a
compute/thread/concurrency one. nova is a Tailscale peer; its inference is
~0.6s warm, but the FIRST request after the agent (co-)restarts pays Tailscale
path-establishment (~1-3s), which the tight 3s route timeout clipped — so the
call fell back for no reason, on every restart. Fix: warm the path/connection/
slot in the background at boot, and give route() enough margin (5s) to absorb a
cold path after an idle period. nova's `-np 4` slots already parallelise fine
(measured 4 concurrent = 1.0s) — the bottleneck was never concurrency.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio

import httpx
import pytest

from ghost_agent.core.llm import LLMClient, _ROUTE_TIMEOUT_S


def _client_with_recording_worker():
    calls = []

    async def worker_post(*a, **kw):
        calls.append(kw)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]},
            request=httpx.Request("POST", "http://nova.invalid"))

    c = LLMClient(upstream_url="http://main.invalid",
                  worker_nodes=[{"url": "http://nova.invalid", "model": "Nova"}])
    c.worker_clients[0]["client"].post = worker_post
    return c, calls


class TestWarmUp:
    def test_warms_the_worker_pool(self):
        c, calls = _client_with_recording_worker()
        asyncio.run(c.warm_up_workers())
        # One warmup per slot (3), tiny + thinking-off.
        assert len(calls) == 3
        # It never raises and doesn't touch the main model.

    def test_warmup_payload_is_tiny_and_thinkless(self):
        c, calls = _client_with_recording_worker()
        # Capture the payload the dispatch sends to the node.
        seen = {}

        async def worker_post(url, content=None, **kw):
            import json
            seen.update(json.loads(content))
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]},
                request=httpx.Request("POST", "http://nova.invalid"))
        c.worker_clients[0]["client"].post = worker_post
        asyncio.run(c.warm_up_workers())
        assert seen["max_tokens"] == 1
        assert seen["chat_template_kwargs"]["enable_thinking"] is False

    def test_no_worker_pool_is_a_noop(self):
        c = LLMClient(upstream_url="http://main.invalid")   # no nodes
        # Must not raise, must not call anything.
        asyncio.run(c.warm_up_workers())

    def test_dead_node_does_not_hang_startup(self):
        c, _ = _client_with_recording_worker()

        async def dead_post(*a, **kw):
            raise httpx.ConnectError("nova down")
        c.worker_clients[0]["client"].post = dead_post
        # Breaks out after the first failure — no exception, no hang.
        asyncio.run(c.warm_up_workers())


class TestRouteTimeoutSizing:
    def test_timeout_absorbs_a_queued_worker_call_but_stays_bounded(self):
        # Warm query-expansion is ~2.3s uncontended, but on a small-`-np`
        # worker a route() call that queues behind one other lands at ~5.3s
        # (measured -np 2), and behind TWO at just over 8s (verify died at
        # exactly 8.0s live, 2026-07-14). The ceiling must clear that
        # double-queued case (>8s) yet still fail fast on a genuinely sick
        # node — 12s does both.
        assert 8.0 < _ROUTE_TIMEOUT_S <= 12.0

    def test_startup_wires_the_warmup_non_blocking(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "main.py").read_text()
        # spawn_bg = fire-and-forget; boot must NOT await the warmup.
        assert "warm_up_workers()" in src
        assert 'name="node-warmup"' in src


class TestKeepAlive:
    """Periodic keepalive (2026-07-12). Boot warmup only covers the FIRST
    request; a Tailscale peer's path re-cools when the node idles between
    requests or during a long tool phase, so the front-of-request AND finalize
    route() calls both ReadTimeout at 5s. A tiny ping every interval keeps the
    path warm."""

    def test_pings_each_node_then_stops_on_cancel(self):
        c, calls = _client_with_recording_worker()

        async def run():
            task = asyncio.ensure_future(c.keepalive_workers(interval_s=0.01))
            await asyncio.sleep(0.05)          # allow a few cycles
            fired = len(calls)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return fired

        assert asyncio.run(run()) >= 1         # at least one keepalive ping

    def test_no_pool_returns_immediately(self):
        c = LLMClient(upstream_url="http://main.invalid")   # no nodes
        # Would loop forever if it didn't early-return on an empty pool.
        asyncio.run(asyncio.wait_for(
            c.keepalive_workers(interval_s=999), timeout=1.0))

    def test_dead_node_does_not_crash_the_loop(self):
        c, _ = _client_with_recording_worker()

        async def dead_post(*a, **kw):
            raise httpx.ConnectError("nova down")
        c.worker_clients[0]["client"].post = dead_post

        async def run():
            task = asyncio.ensure_future(c.keepalive_workers(interval_s=0.01))
            await asyncio.sleep(0.05)
            still_running = not task.done()    # survived the dead-node pings
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return still_running

        assert asyncio.run(run()) is True

    def test_never_touches_the_main_node(self):
        # Keepalive is off-main only: a ping must go to the worker pool, never
        # the main upstream (which would disturb the single main slot's cache).
        c, calls = _client_with_recording_worker()
        main_hits = {"n": 0}
        orig = c.http_client.post

        async def counting_main_post(*a, **kw):
            main_hits["n"] += 1
            return await orig(*a, **kw)
        c.http_client.post = counting_main_post

        async def run():
            task = asyncio.ensure_future(c.keepalive_workers(interval_s=0.01))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run())
        assert len(calls) >= 1 and main_hits["n"] == 0

    def test_startup_wires_keepalive_non_blocking(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "main.py").read_text()
        assert "keepalive_workers(" in src
        assert 'name="node-keepalive"' in src
        assert "GHOST_WORKER_KEEPALIVE_S" in src   # tunable / disable-able


class TestKeepAliveQuietLogging:
    """Heartbeats log TRANSITIONS, not ticks (2026-07-12): the 45s
    'keepalive → Worker Node (Nova)' line spammed the live stream. Healthy
    pings are silent; a node going down logs ONE warning, coming back logs
    ONE recovery line. And a failed ping must never fall back to main."""

    def _run_keepalive(self, c, secs=0.05, mid=None):
        async def run():
            task = asyncio.ensure_future(c.keepalive_workers(interval_s=0.01))
            await asyncio.sleep(secs)
            if mid is not None:
                mid()
                await asyncio.sleep(secs)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        asyncio.run(run())

    def _capture_logs(self, monkeypatch):
        import ghost_agent.core.llm as llm_mod
        logs = []
        monkeypatch.setattr(
            llm_mod, "pretty_log",
            lambda title, msg, **kw: logs.append((title, msg)))
        return logs

    def test_healthy_pings_are_silent(self, monkeypatch):
        logs = self._capture_logs(monkeypatch)
        c, calls = _client_with_recording_worker()
        self._run_keepalive(c)
        assert len(calls) >= 1                                   # pings fired
        assert not [t for t, _ in logs if t == "Worker Compute"]  # no spam
        assert not [t for t, _ in logs if t == "Node Keepalive"]  # no noise

    def test_down_and_recovery_are_logged_once_each(self, monkeypatch):
        logs = self._capture_logs(monkeypatch)
        c, _ = _client_with_recording_worker()
        real_post = c.worker_clients[0]["client"].post
        state = {"dead": True}

        async def flaky_post(*a, **kw):
            if state["dead"]:
                raise httpx.ConnectError("nova down")
            return await real_post(*a, **kw)
        c.worker_clients[0]["client"].post = flaky_post

        self._run_keepalive(c, mid=lambda: state.update(dead=False))
        downs = [m for t, m in logs
                 if t == "Node Keepalive" and "stopped answering" in m]
        ups = [m for t, m in logs
               if t == "Node Keepalive" and "recovered" in m]
        assert len(downs) == 1, downs    # one warning, not one per tick
        assert len(ups) == 1, ups
        # And the repeated per-tick failures stayed out of the pretty stream.
        assert not [t for t, _ in logs if t == "Worker Node Failed"]
        assert not [t for t, _ in logs if t == "Worker Compute Failed"]

    def test_failed_ping_never_falls_back_to_main(self):
        c, _ = _client_with_recording_worker()

        async def dead_post(*a, **kw):
            raise httpx.ConnectError("down")
        c.worker_clients[0]["client"].post = dead_post
        main_hits = {"n": 0}

        async def main_post(*a, **kw):
            main_hits["n"] += 1
            raise AssertionError("main model must not serve keepalive")
        c.http_client.post = main_post
        self._run_keepalive(c)
        assert main_hits["n"] == 0


class TestWorkerTaskLabel:
    """Worker-routing log lines must say WHAT the task is (2026-07-12) — the
    old generic 'Routing background task to Worker Node (Nova)' told the
    operator nothing when several fire back-to-back."""

    def _capture_worker_logs(self, monkeypatch, **cc_kwargs):
        import ghost_agent.core.llm as llm_mod
        logs = []
        monkeypatch.setattr(
            llm_mod, "pretty_log",
            lambda title, msg, **kw: logs.append((title, msg)))
        c, _ = _client_with_recording_worker()
        payload = {"model": "Nova", "messages": [{"role": "user", "content": "hi"}]}
        asyncio.run(c.chat_completion(payload, use_worker=True,
                                      is_background=True, **cc_kwargs))
        return [m for (t, m) in logs if t == "Worker Compute"]

    def test_label_appears_in_worker_log(self, monkeypatch):
        wlogs = self._capture_worker_logs(monkeypatch, task_label="query expansion")
        assert any("query expansion" in m for m in wlogs)
        # And the old uninformative phrasing is gone.
        assert all("Routing background task" not in m for m in wlogs)

    def test_missing_label_falls_back_to_generic(self, monkeypatch):
        wlogs = self._capture_worker_logs(monkeypatch)   # no task_label
        assert any("background task" in m for m in wlogs)

    def test_route_label_is_the_actual_task(self, monkeypatch):
        # route() must log the task it actually runs. A hardcoded "query
        # expansion" made a DECOMPOSE_QUERY (RAG-fusion) timeout read as the
        # anaphora expander and misdirected a live debug (2026-07-12).
        import ghost_agent.core.llm as llm_mod
        logs = []
        monkeypatch.setattr(
            llm_mod, "pretty_log",
            lambda title, msg, **kw: logs.append((title, msg)))
        c, _ = _client_with_recording_worker()
        payload = {"model": "Nova",
                   "messages": [{"role": "user", "content": "q"}]}
        asyncio.run(c.route("DECOMPOSE_QUERY", payload, fallback=None))
        wc = [m for t, m in logs if t == "Worker Compute"]
        assert any("decompose query" in m for m in wc), wc

    def test_internal_requests_skip_decomposition_and_smart_memory(self):
        # sub-/sched- requests are machine sub-calls (chess moves, jobs):
        # (a) hydration must not run the LLM DECOMPOSE_QUERY round-trip on
        # their critical path, and (b) their prompts must never be queued
        # into smart memory (retrieval pollution + worker-side extract load
        # that timed out the NEXT request's routing calls).
        import inspect
        from ghost_agent.core.agent import GhostAgent
        # #5 step 4a moved the streaming smart-mem gate into _stream_final_generation.
        src = (inspect.getsource(GhostAgent.handle_chat)
               + inspect.getsource(GhostAgent._stream_final_generation))
        assert "_is_int_req_h(req_id)" in src          # hydration gate
        assert "_is_int_req_m1(req_id)" in src         # smart-mem (stream)
        assert "_is_int_req_m2(req_id)" in src         # smart-mem (non-stream)
