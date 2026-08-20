"""§4BV R4. The runtime defects R4 found, and the coverage holes beneath them.

Two of these pin things that were never pinned at all: the circuit breaker's
only real input on the production pool (removing BOTH `record_success` and
`record_failure` from the worker branch survived 158 tests), and
`get_embeddings`' relationship to the lock it shares with every main chat
call. Both are load-bearing for the live topology, and neither had a test.
"""

import asyncio

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.llm import LLMClient


def _client(fail=True):
    c = LLMClient("http://main.invalid:8088")
    cl = MagicMock()
    if fail:
        cl.post = AsyncMock(side_effect=RuntimeError("node down"))
    else:
        r = MagicMock()
        r.json = lambda: {"choices": [{"message": {"content": "ok"}}]}
        r.raise_for_status = lambda: None
        r.status_code, r.text = 200, "{}"
        cl.post = AsyncMock(return_value=r)
    c.worker_clients = [{"url": "http://W", "model": "m",
                         "client": cl, "name": "W"}]
    c._worker_index = 0

    async def _main(*a, **k):
        r = MagicMock()
        r.json = lambda: {"choices": [{"message": {"content": "main"}}]}
        r.raise_for_status = lambda: None
        r.status_code, r.text = 200, "{}"
        return r

    c.http_client = MagicMock()
    c.http_client.post = AsyncMock(side_effect=_main)
    return c


class TestTheBreakerActuallyReceivesOutcomes:
    """⚠ THE INSTRUMENT'S ONLY INPUT. Every `is_available()` decision, every
    `require_healthy` refusal, the node status on `/api/health` and the
    recovery log all read state that ONLY `record_success`/`record_failure`
    write. Deleting both from the worker branch — the branch that carries all
    live off-main traffic — left 158 tests green (R4 lens B). A breaker that
    is never told anything is not a conservative breaker; it is a permanently
    closed one, and it reports "healthy" for a node that has been dead for
    hours."""

    def test_repeated_failures_trip_the_breaker(self):
        c = _client(fail=True)
        for _ in range(3):
            asyncio.run(c._do_chat_completion(
                {"model": "m", "messages": []}, use_worker=True, timeout=5))
        st = c.circuit_breaker.get_status()["http://W"]
        assert st["state"] == "open", (
            f"three consecutive node failures left the breaker {st} — "
            f"nothing is feeding it outcomes")
        assert not c.circuit_breaker.is_available("http://W")

    def test_a_success_closes_a_half_open_breaker(self):
        c = _client(fail=False)
        c.circuit_breaker.cooldown_seconds = 0.0
        for _ in range(3):
            c.circuit_breaker.record_failure("http://W")
        assert c.circuit_breaker.get_status()["http://W"]["state"] == "open"
        asyncio.run(c._do_chat_completion(
            {"model": "m", "messages": []}, use_worker=True, timeout=5))
        st = c.circuit_breaker.get_status()["http://W"]
        assert st["state"] == "closed", (
            f"a successful call did not reset the breaker: {st} — the node "
            f"heals in reality and stays sick in our model of it")
        assert st["failures"] == 0

    def test_a_4xx_does_not_trip_the_breaker(self):
        """The complement, so nobody "fixes" the above by recording every
        exception: a bad payload repeats identically on any node, and
        counting it would take a healthy node out of rotation for 60s."""
        c = LLMClient("http://main.invalid:8088")
        cl = MagicMock()
        req = httpx.Request("POST", "http://W/v1/chat/completions")
        resp = httpx.Response(400, request=req, text="bad payload")
        cl.post = AsyncMock(side_effect=httpx.HTTPStatusError(
            "400", request=req, response=resp))
        c.worker_clients = [{"url": "http://W", "model": "m",
                             "client": cl, "name": "W"}]
        c._worker_index = 0

        async def _main(*a, **k):
            r = MagicMock()
            r.json = lambda: {"choices": [{"message": {"content": "main"}}]}
            r.raise_for_status = lambda: None
            r.status_code, r.text = 200, "{}"
            return r

        c.http_client = MagicMock()
        c.http_client.post = AsyncMock(side_effect=_main)
        for _ in range(5):
            asyncio.run(c._do_chat_completion(
                {"model": "m", "messages": []}, use_worker=True, timeout=5))
        assert c.circuit_breaker.is_available("http://W"), (
            "a caller-fault 4xx took a healthy node out of rotation")


class TestEmbeddingsAreVisibleAndBounded:
    """R4 lens B, NEW-2. `get_embeddings` takes `_main_node_lock` — the same
    mutex every main chat completion needs — and did so unbounded and
    uncounted."""

    def _client(self, hang=False):
        c = LLMClient("http://main.invalid:8088")
        seen = {}

        async def _post(path, **kw):
            seen["timeout"] = kw.get("timeout", "<unset>")
            seen["foreground_during"] = c.foreground_tasks
            if hang:
                raise httpx.ReadTimeout("too slow")
            r = MagicMock()
            r.json = lambda: {"data": [{"embedding": [0.1, 0.2]}]}
            r.raise_for_status = lambda: None
            return r

        c.http_client = MagicMock()
        c.http_client.post = AsyncMock(side_effect=_post)
        return c, seen

    def test_the_call_is_bounded(self):
        c, seen = self._client()
        asyncio.run(c.get_embeddings(["hello"]))
        t = seen["timeout"]
        assert isinstance(t, (int, float)), (
            f"embeddings posts with timeout={t!r} — httpx's 1200s default "
            f"means one wedged call blocks every main completion for 20 min")
        assert 0 < t <= 600

    def test_the_call_is_counted_as_foreground_while_it_holds_the_lock(self):
        """⚠ THE INVISIBILITY IS THE DEFECT. `_wait_for_foreground_clear`
        reads `foreground_tasks` to decide whether the main slot is free; at
        0 it waves background work straight into a mutex this call is
        holding."""
        c, seen = self._client()
        assert c.foreground_tasks == 0
        asyncio.run(c.get_embeddings(["hello"]))
        assert seen["foreground_during"] == 1, (
            "an in-flight embeddings call reports the agent as idle")
        assert c.foreground_tasks == 0, "the counter leaked"

    def test_the_counter_is_released_when_the_call_fails(self):
        c, _ = self._client(hang=True)
        with pytest.raises(Exception):
            asyncio.run(c.get_embeddings(["hello"]))
        assert c.foreground_tasks == 0, (
            "a failed embeddings call left foreground_tasks pinned above "
            "zero — the biological watchdog never sees an idle agent again")

    def test_a_timeout_is_retried_like_every_other_main_caller(self):
        c, _ = self._client(hang=True)
        with pytest.raises(Exception):
            asyncio.run(c.get_embeddings(["hello"]))
        assert c.http_client.post.await_count == 2, (
            f"a ReadTimeout was fatal on attempt 1 "
            f"({c.http_client.post.await_count} attempt(s)) — every other "
            f"main-node caller retries this family")


class TestTheSameBoxIsNotServicedTwice:
    """R4 lens B, NEW-7. `--worker-nodes` and `--critic-nodes` are the SAME
    URL in the shipping topology, and both the warmup and the keepalive loop
    iterate the two pools unconditionally. Live, that meant six warmup
    generations at boot against one box and two un-gated pings every 45s."""

    def _dual_pool_client(self):
        c = LLMClient("http://main.invalid:8088")
        posts = []

        async def _post(path, **kw):
            posts.append(path)
            r = MagicMock()
            r.json = lambda: {"choices": [{"message": {"content": "ok"}}]}
            r.raise_for_status = lambda: None
            r.status_code, r.text = 200, "{}"
            return r

        cl = MagicMock()
        cl.post = AsyncMock(side_effect=_post)
        props = MagicMock()
        props.json = lambda: {"total_slots": 1}
        props.raise_for_status = lambda: None
        cl.get = AsyncMock(return_value=props)
        node = {"url": "http://NOVA", "model": "Nova", "client": cl,
                "name": "Nova"}
        c.worker_clients = [node]
        c.critic_clients = [node]      # byte-identical, as live
        c._worker_index = c._critic_index = 0
        c.http_client = MagicMock()
        c.http_client.post = AsyncMock(side_effect=_post)
        return c, posts

    def test_warmup_fires_one_request_per_advertised_slot_not_three(self):
        c, posts = self._dual_pool_client()
        asyncio.run(c.warm_up_workers())
        assert len(posts) == 1, (
            f"a 1-slot node advertising total_slots=1, present in two pools, "
            f"received {len(posts)} warmup generations — the docstring "
            f"promises one per advertised slot, once per node")

    def test_keepalive_pings_each_physical_node_once_per_interval(self):
        """⚠ EXACTLY ONE PASS, not a wall-clock window. The first version of
        this test slept 0.08s at a 0.01s interval and asserted a bound of 4 —
        which is ~8 passes, so the number it compared against meant nothing.
        Drive the loop a known number of times instead."""
        from unittest.mock import patch

        c, posts = self._dual_pool_client()
        calls = {"n": 0}

        async def _sleep(_s):
            calls["n"] += 1
            if calls["n"] > 1:          # let exactly one pass complete
                raise asyncio.CancelledError

        with patch("ghost_agent.core.llm.asyncio.sleep", _sleep):
            asyncio.run(c.keepalive_workers(interval_s=0.01))

        assert calls["n"] == 2, f"the loop did not run exactly once: {calls}"
        assert len(posts) == 1, (
            f"one interval produced {len(posts)} pings against a single "
            f"physical node — worker and critic are the same box, and every "
            f"keepalive ping bypasses the concurrency gate, so a duplicate "
            f"is un-gated load on a node of unknown -np")


class TestTheCapacityGateRecoversFromAFailedFirstProbe:
    """R4 lens B, NEW-6 — and the second half of that fix, which I initially
    shipped without.

    `_node_capacity` used to cache a FAILED probe permanently, pinning a node
    at the provisional default 3 for the whole process. The obvious fix is a
    retry TTL on failures. That fix is INERT on its own: `_node_slots` is
    written in exactly one place and never resized, so the semaphore built
    from the guess outlives the corrected number and `_node_capacity` is
    never consulted again. Both halves are required, so both are pinned."""

    def _node(self, up):
        cl = MagicMock()
        if up:
            r = MagicMock()
            r.json = lambda: {"total_slots": 1}
            r.raise_for_status = lambda: None
            cl.get = AsyncMock(return_value=r)
        else:
            cl.get = AsyncMock(side_effect=RuntimeError("node down"))
        return {"url": "http://N", "model": "Nova", "client": cl}

    def test_a_failed_probe_is_not_cached_as_an_answer(self):
        c = LLMClient("http://main.invalid:8088")
        node = self._node(up=False)
        assert asyncio.run(c._node_capacity(node)) == c._node_slot_default
        assert "http://N" not in c._node_slot_caps, (
            "a failed probe was recorded as the node's capacity — a node "
            "that was merely restarting is now permanently mis-sized")

    def test_the_gate_is_resized_once_the_real_capacity_is_known(self):
        """⚠ THE HALF THAT WAS MISSING. Without it the TTL above changes a
        dict nobody reads again."""
        c = LLMClient("http://main.invalid:8088")
        node = self._node(up=False)

        async def _touch():
            async with c._node_slot(node, wait_timeout=1):
                pass

        asyncio.run(_touch())
        assert c._node_slot_built_cap["http://N"] == c._node_slot_default

        node["client"] = self._node(up=True)["client"]
        c._node_cap_retry_at.clear()          # the retry window elapses
        asyncio.run(_touch())
        assert c._node_slot_built_cap["http://N"] == 1, (
            "the node advertised 1 slot but the gate is still sized from the "
            "guess — 3x over-subscription applied by the gate whose only job "
            "is to prevent over-subscription")
        assert c._node_slots["http://N"]._value == 1

    def test_an_authoritative_capacity_is_not_re_probed(self):
        """The complement: once we have a real number the fast path must
        take over, or every dispatch serialises on the global slots lock."""
        c = LLMClient("http://main.invalid:8088")
        node = self._node(up=True)

        async def _touch():
            async with c._node_slot(node, wait_timeout=1):
                pass

        for _ in range(5):
            asyncio.run(_touch())
        assert node["client"].get.await_count == 1, (
            f"/props was probed {node['client'].get.await_count} times for a "
            f"node whose capacity is already known")
