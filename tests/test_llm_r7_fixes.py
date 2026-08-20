"""§4BV R7 — the guards R6 left unpinned, and R7's own new behaviour.

R7's mutation audit found 20 survivors. Most were second-order: pairs of
redundant guards where each half masks the other's removal — which is
precisely how R5 shipped its regressions. Each half is pinned separately
here, so deleting either one fails.
"""

import asyncio
import time
from contextlib import asynccontextmanager

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.llm import (LLMClient, NodeSaturated, _MIN_ACQUIRE,
                                  _MIN_HTTP_FLOOR, _MAIN_FALLBACK_MIN_S)


def _client(pool_size=1, slots=1, post=None, probe_delay=0.0):
    c = LLMClient("http://main.invalid:8088")
    posts = []

    async def _default_post(path, **kw):
        posts.append(kw.get("timeout"))
        r = MagicMock()
        r.json = lambda: {"choices": [{"message": {"content": "node"}}]}
        r.raise_for_status = lambda: None
        r.status_code, r.text = 200, "{}"
        return r

    c.worker_clients = []
    for i in range(pool_size):
        cl = MagicMock()
        cl.post = AsyncMock(side_effect=post or _default_post)

        async def _get(*a, _d=probe_delay, **kw):
            if _d:
                await asyncio.sleep(_d)
            r = MagicMock()
            r.json = lambda: {"total_slots": slots}
            r.raise_for_status = lambda: None
            return r

        cl.get = AsyncMock(side_effect=_get)
        c.worker_clients.append({"url": f"http://N{i}", "model": "m",
                                 "client": cl, "name": f"N{i}"})
    c._worker_index = 0

    async def _main(*a, **kw):
        r = MagicMock()
        r.json = lambda: {"choices": [{"message": {"content": "MAIN"}}]}
        r.raise_for_status = lambda: None
        r.status_code, r.text = 200, "{}"
        return r

    c.http_client = MagicMock()
    c.http_client.post = AsyncMock(side_effect=_main)
    return c, posts


class TestTheKeepaliveBypassHasTwoHalvesAndBothMatter:
    """R7 lens C, M13/M14/M40. The health probe's gate bypass is implemented
    TWICE — `_gate_wait` returns a literal None, and `_node_slot` bypasses
    when a callable resolves to None. Each was individually unpinned;
    removing BOTH hung the suite forever (there is no pytest-timeout), which
    is the worst possible failure mode for CI."""

    def test_the_probe_does_not_even_wait_for_a_capacity_probe(self):
        """`_gate_wait`'s half. If it returns a truthy callable, the health
        probe pays the `/props` probe before the other bypass catches it —
        and the image-gen node never answers `/props` at all."""
        c, _ = _client(probe_delay=1.0)

        async def _run():
            t0 = time.monotonic()
            await c._do_chat_completion(
                {"model": "m", "messages": []}, use_worker=True,
                timeout=30, slot_wait=30, task_label="keepalive")
            return time.monotonic() - t0

        dt = asyncio.run(_run())
        assert dt < 0.5, (
            f"the keepalive probe took {dt:.2f}s — it is entering the gate "
            f"and paying a capacity probe. A probe that queues behind the "
            f"traffic it watches reports a BUSY node as a DEAD one.")

    def test_a_callable_resolving_to_None_bypasses_the_gate(self):
        """`_node_slot`'s half, pinned directly. Without it a callable that
        resolves to None reaches `wait_for(..., timeout=None)` and waits
        FOREVER — the unbounded park this gate exists to replace."""
        c, _ = _client()
        node = c.worker_clients[0]

        async def _run():
            async with c._node_slot(node, wait_timeout=5):
                cm = c._node_slot(node, wait_timeout=lambda: None)
                await asyncio.wait_for(cm.__aenter__(), timeout=1.0)
                await cm.__aexit__(None, None, None)

        asyncio.run(_run())     # a hang here IS the failure


def test_the_permit_wait_is_resolved_AFTER_the_capacity_probe():
    """R7 lens C, M38/M39. R6's own measured defect — a 5s `/props` probe
    spent outside the caller's budget — had no pin.

    ⚠ Captures the wait ACTUALLY HANDED to the semaphore, not elapsed time.
    An elapsed-time bound could not discriminate: with the probe inside the
    call either way, correct and broken differ by roughly the probe length,
    and any band wide enough to be stable was wide enough to pass on the
    defect. The budget the gate receives is the mechanism itself."""
    c, _ = _client(probe_delay=2.0)
    node = c.worker_clients[0]
    seen = []

    real_wait_for = asyncio.wait_for

    class _Shim:
        def __getattr__(self, name):
            return getattr(asyncio, name)

        async def wait_for(self, aw, timeout=None):
            seen.append(timeout)
            return await real_wait_for(aw, timeout)

    from ghost_agent.core import llm as _llm

    async def _run():
        # ⚠ Hold the permit WITHOUT priming the capacity cache. Acquiring it
        # through `_node_slot` would run the probe first and cache the
        # result, so the measured call would never probe at all — which is
        # how the first version of this test measured 3.00s and proved
        # nothing.
        url = node["url"]
        sem = asyncio.Semaphore(1)
        c._node_slots[url] = sem
        c._node_slot_built_cap[url] = 1
        await sem.acquire()
        assert url not in c._node_slot_caps, "the probe would be skipped"

        saved = _llm.asyncio
        _llm.asyncio = _Shim()
        try:
            await c._do_chat_completion(
                {"model": "m", "messages": []}, use_worker=True,
                timeout=6, slot_wait=6, total_budget=6)
        finally:
            _llm.asyncio = saved

    asyncio.run(_run())
    assert seen, "the gate was never asked to acquire"
    # The probe burns 2s of a 6s total, so a wait resolved AFTERWARDS must
    # reflect what is left (~1s after the 3s reserve), not the ~1.5s that a
    # pre-probe resolution would have computed against the full budget.
    assert seen[0] <= 1.2, (
        f"the gate was handed {seen[0]:.2f}s after a 2s capacity probe had "
        f"already run — the wait is being resolved as an argument "
        f"expression, before the probe, so the probe is spent outside the "
        f"caller's budget")


def test_a_spent_budget_declines_the_permit_instead_of_POSTing():
    """R7 lens A, MAJOR-3. Clamping the POST budget up to the floor meant a
    permit acquired with 0.5s left still POSTed — with 3s — and the
    resulting ReadTimeout IS a node fault, so our own queueing opened the
    node's breaker. `deep_research` fans out three at a time, which is
    enough to trip it for 60s."""
    seen = []

    async def _post(path, **kw):
        seen.append(kw.get("timeout"))
        raise Exception("ReadTimeout")

    c, _ = _client(post=_post)
    node = c.worker_clients[0]

    async def _run():
        await c._node_capacity(node)
        cm = c._node_slot(node, wait_timeout=30)
        await cm.__aenter__()

        async def _free():
            await asyncio.sleep(5.6)          # frees as the total expires
            await cm.__aexit__(None, None, None)

        asyncio.ensure_future(_free())
        await c._do_chat_completion(
            {"model": "m", "messages": []}, use_worker=True,
            timeout=45, slot_wait=6, total_budget=6)

    asyncio.run(_run())
    assert seen == [], (
        f"POSTed with {seen}s of budget left — a request that cannot finish, "
        f"whose ReadTimeout is then charged to the node")
    st = c.circuit_breaker.get_status().get("http://N0", {})
    assert not st.get("failures"), (
        f"our own queueing was recorded as a node fault: {st}")


def test_a_stated_total_survives_the_fallback_to_main():
    """R7 lens A, MAJOR-4. Both fallback arms used the caller's raw timeout
    and raised it to `_MAIN_FALLBACK_TIMEOUT_S`, so a stated 60s total could
    queue 57s on the pool and hand the 35B 300s — 357s for a "60s" budget,
    holding the single foreground slot."""
    async def _dead(path, **kw):
        raise RuntimeError("node down")

    c, _ = _client(post=_dead)
    got = {}

    async def _main(*a, **kw):
        got["t"] = kw.get("timeout")
        r = MagicMock()
        r.json = lambda: {"choices": [{"message": {"content": "MAIN"}}]}
        r.raise_for_status = lambda: None
        r.status_code, r.text = 200, "{}"
        return r

    c.http_client.post = AsyncMock(side_effect=_main)
    asyncio.run(c._do_chat_completion(
        {"model": "m", "messages": []}, use_worker=True,
        timeout=60, slot_wait=60, total_budget=60))
    assert got["t"] <= 60.0 + 0.01, (
        f"the main fallback got {got['t']}s against a stated 60s total")
    assert got["t"] >= _MAIN_FALLBACK_MIN_S - 0.01, (
        f"the main fallback got {got['t']}s — below a 35B-sized floor is a "
        f"guaranteed ReadTimeout (2026-07-11)")


def test_no_total_leaves_the_main_fallback_generous():
    """The complement: without a stated total the 300s bound stands."""
    async def _dead(path, **kw):
        raise RuntimeError("node down")

    c, _ = _client(post=_dead)
    got = {}

    async def _main(*a, **kw):
        got["t"] = kw.get("timeout")
        r = MagicMock()
        r.json = lambda: {"choices": [{"message": {"content": "MAIN"}}]}
        r.raise_for_status = lambda: None
        r.status_code, r.text = 200, "{}"
        return r

    c.http_client.post = AsyncMock(side_effect=_main)
    asyncio.run(c._do_chat_completion(
        {"model": "m", "messages": []}, use_worker=True, timeout=12))
    assert got["t"] >= 300.0, (
        f"a node-sized {got['t']}s budget on the 35B is a guaranteed "
        f"ReadTimeout — the 2026-07-11 regression")


def test_warmup_records_ZERO_breaker_failures():
    """R7 lens C, M16. The shipped test asserted "1 failure, not 3", which a
    breaker that records every warmup still satisfies (1 < threshold 3). The
    runtime now records none at all — a boot ping against a booting node is
    not evidence."""
    c = LLMClient("http://main.invalid:8088")
    cl = MagicMock()
    cl.post = AsyncMock(side_effect=RuntimeError("still booting"))
    cl.get = AsyncMock(side_effect=RuntimeError("still booting"))
    c.worker_clients = [{"url": "http://NOVA", "model": "Nova",
                         "client": cl, "name": "Nova"}]
    c.critic_clients = list(c.worker_clients)
    c._worker_index = c._critic_index = 0
    asyncio.run(c.warm_up_workers())
    st = c.circuit_breaker.get_status().get("http://NOVA", {})
    assert st.get("failures", 0) == 0, (
        f"boot warmup recorded {st} against the node it was warming — at a "
        f"4-slot node the fan-out is exactly the breaker threshold")


def test_a_stale_failed_probe_cannot_beat_a_fresh_successful_one():
    """R7 lens C, M15. Probing outside the lock (correct, it removes a
    cross-node stall) means two first-touches apply their caps in
    COMPLETION order. Without re-reading the authority under the lock, a
    slow FAILING probe lands after a fast SUCCEEDING one and wins — and
    because success populates `_node_slot_caps`, the fast path then locks
    that wrong value in forever."""
    c = LLMClient("http://main.invalid:8088")
    order = []

    async def _slow_fail(*a, **kw):
        await asyncio.sleep(0.30)
        order.append("fail")
        raise RuntimeError("probe failed")

    async def _fast_ok(*a, **kw):
        await asyncio.sleep(0.05)
        order.append("ok")
        r = MagicMock()
        r.json = lambda: {"total_slots": 1}
        r.raise_for_status = lambda: None
        return r

    slow = {"url": "http://N", "model": "m", "client": MagicMock()}
    slow["client"].get = AsyncMock(side_effect=_slow_fail)
    fast = {"url": "http://N", "model": "m", "client": MagicMock()}
    fast["client"].get = AsyncMock(side_effect=_fast_ok)

    async def _touch(n):
        async with c._node_slot(n, wait_timeout=5):
            pass

    async def _both():
        await asyncio.gather(_touch(fast), _touch(slow))

    asyncio.run(_both())
    assert order == ["ok", "fail"], f"probe order not as constructed: {order}"
    assert c._node_slot_built_cap["http://N"] == 1, (
        f"the gate is sized {c._node_slot_built_cap['http://N']} against a "
        f"node that advertises 1 — a stale failed probe overwrote a fresh "
        f"successful one, and the fast path will keep that value forever")


def test_a_caller_stating_a_total_but_no_timeout_still_gets_one():
    """R7 lens C, M06. `_http_budget`'s `t is None` arm was unreachable by
    any test, so returning None there — handing the node client its 1200s
    default and making the 'total' fiction — survived."""
    c, posts = _client()
    asyncio.run(c._do_chat_completion(
        {"model": "m", "messages": []}, use_worker=True, total_budget=6))
    assert posts and posts[0] is not None, (
        "a caller that stated a total but no timeout POSTed unbounded")
    assert posts[0] <= 6.0 + 0.01, posts


def test_MIN_ACQUIRE_is_a_backstop_and_is_documented_as_one():
    """⚠ AN HONEST NON-TEST, and the reason is worth recording.

    `_MIN_ACQUIRE` exists because `wait_for(sem.acquire(), 0.0)` refuses even
    a completely free semaphore, which is how R5 shipped "the last node of
    every pool is never asked". But after the `+1` divisor landed, every
    share is positive, and the one remaining path to a zero share — a total
    down to its reserve — now returns `_BUDGET_BLOWN` and declines the
    permit before the wait is ever computed. R7 measured the constant
    binding ZERO times across the whole suite.

    So it is unreachable defensive code today, and a behavioural test for it
    would have to construct a state no caller can produce — which is how
    harnesses start guaranteeing their own results. It is pinned
    structurally instead, and this docstring is the record of why. If a
    future change makes a zero share reachable again, replace this with a
    real driven test rather than trusting the constant."""
    import ast
    import inspect
    src = inspect.getsource(LLMClient._do_chat_completion)
    fn = ast.parse(src.lstrip()).body[0]
    helper = [n for n in ast.walk(fn)
              if isinstance(n, ast.FunctionDef) and n.name == "_permit_wait"]
    assert helper, "_permit_wait is gone"
    names = {n.id for n in ast.walk(helper[0]) if isinstance(n, ast.Name)}
    assert "_MIN_ACQUIRE" in names, (
        "_permit_wait can return 0.0 again — and a 0.0 wait rejects a node "
        "whose permits are all free")
    assert _MIN_ACQUIRE > 0.0


@pytest.mark.parametrize("override,expect_clipped", [
    (None, True),      # route's own 12s contract: a total, clipped
    (45.0, False),     # VERIFY asked for 45s of GENERATION
    (60.0, False),     # DISTILL_PATTERN likewise
])
def test_route_bounds_its_own_contract_without_re_timing_its_callers(
        override, expect_clipped):
    """R7 lens A, MAJOR-6. `route()`'s 12s fail-fast is a TOTAL because it is
    awaited on the user's critical path. But R6 forced that total on every
    caller, so a caller raising `timeout` to buy a longer GENERATION had it
    clipped by however long it queued — a VERIFY that queued 20s ran on 25s
    of its stated 45s. `verifier.py`'s own comments still describe the two
    budgets as adding."""
    c, posts = _client()
    kw = {} if override is None else {"timeout": override}
    asyncio.run(c.route("T", {"model": "m", "messages": []},
                        fallback="fb", **kw))
    assert posts and posts[0] is not None
    if expect_clipped:
        from ghost_agent.core.llm import _ROUTE_TIMEOUT_S
        assert posts[0] <= _ROUTE_TIMEOUT_S + 0.01, (
            f"route's own contract is not bounded: {posts[0]}")
    else:
        assert posts[0] >= override - 0.01, (
            f"a caller asking for {override}s of generation got "
            f"{posts[0]:.2f}s — routing's queue bound is being applied to "
            f"the request")
