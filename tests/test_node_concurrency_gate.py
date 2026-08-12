"""Per-node concurrency gate — one budget per node URL, shared by every caller.

THE LIVE DEFECT (req 0fb69c5f, 2026-08-11). Deep research capped itself with
an `asyncio.Semaphore(3)` built INSIDE the function, so the cap was per CALL.
The model issued THREE deep_research calls in one batch: 3 semaphores × 3
permits = **9 concurrent requests at a node advertising 4 slots** (20 worker
calls in 74s). The excess queued on llama-server past the route timeout, every
ReadTimeout counted as a NODE fault, and 3 consecutive tripped the breaker —
ejecting a perfectly healthy Nova for 60s. It "recovered" 20s later because
nothing had ever been wrong with it. The research came back degraded: the
model's own next thought was "mostly timeout errors from the scraper backends".

No tool could fix this from where it stood. Nova serves the WORKER and CRITIC
roles simultaneously on this deployment, and query-expansion, web summaries,
fact distillation and the verifier all reach it by different paths. Keying the
budget on the node URL is what makes it authoritative — role-scoped limits are
each individually polite and still collectively flood one box.

⚠ The gate also RELOCATES the queue wait from llama-server into this process,
which is what turns "is this node sick or did we flood it?" from a heuristic
into a fact: `NodeSaturated` means the request was never sent.
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.core.llm import (  # noqa: E402
    LLMClient, NodeSaturated, _is_node_fault, _node_error_detail,
)


def _client() -> LLMClient:
    """A client with no nodes — the gate is exercised directly, so no network
    and no constructor-time node probing."""
    return LLMClient(upstream_url="http://127.0.0.1:9")


class _FakeHTTP:
    """Stands in for httpx: /props reports a slot count, nothing else."""

    def __init__(self, slots):
        self.slots = slots
        self.props_calls = 0

    async def get(self, path, timeout=None):
        assert path == "/props"
        self.props_calls += 1

        class _R:
            status_code = 200

            def raise_for_status(_self):
                return None

            def json(_self):
                return ({"total_slots": self.slots}
                        if self.slots is not None else {})
        return _R()


def _node(slots=4, url="http://nova:8088", model="Nova"):
    return {"url": url, "model": model, "client": _FakeHTTP(slots)}


@pytest.mark.asyncio
async def test_capacity_comes_from_the_NODE_not_a_hand_set_number():
    """`-np` is the number that matters and it lives on the node. A configured
    copy drifts the moment the operator restarts a node with new flags."""
    c = _client()
    n = _node(slots=4)
    assert await c._node_capacity(n) == 4
    # cached: the hot path must not re-probe
    assert await c._node_capacity(n) == 4
    assert n["client"].props_calls == 1


@pytest.mark.asyncio
async def test_an_unprobeable_node_falls_back_to_the_OLD_per_call_value():
    """⚠ A gate whose failure mode is 'no gate' is the defect this removes. A
    node that will not report capacity behaves exactly as it did before (3),
    never unbounded."""
    c = _client()
    assert await c._node_capacity(_node(slots=None)) == 3

    class _Dead:
        async def get(self, *a, **k):
            raise OSError("connection refused")
    assert await c._node_capacity(
        {"url": "http://dead:1", "model": "d", "client": _Dead()}) == 3


@pytest.mark.asyncio
async def test_the_gate_actually_CAPS_concurrency_at_the_node_count():
    """The load-bearing behaviour: 9 simultaneous callers against a 4-slot node
    must never put more than 4 in flight — the live 3×3 fan-out."""
    c = _client()
    n = _node(slots=4)
    live = 0
    peak = 0

    async def one():
        nonlocal live, peak
        async with c._node_slot(n, wait_timeout=5):
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.02)
            live -= 1

    await asyncio.gather(*[one() for _ in range(9)])
    assert peak == 4, f"peak in-flight {peak}, node advertises 4"


@pytest.mark.asyncio
async def test_one_budget_is_SHARED_across_roles_on_the_same_url():
    """Nova is both the worker and the critic node here. Two role-shaped node
    dicts pointing at ONE url must draw on ONE budget, or each role is polite
    and the box still floods."""
    c = _client()
    worker = _node(slots=2, url="http://nova:8088", model="Nova-worker")
    critic = _node(slots=2, url="http://nova:8088", model="Nova-critic")
    live = 0
    peak = 0

    async def one(n):
        nonlocal live, peak
        async with c._node_slot(n, wait_timeout=5):
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.02)
            live -= 1

    await asyncio.gather(*([one(worker) for _ in range(3)]
                           + [one(critic) for _ in range(3)]))
    assert peak == 2, f"peak {peak} — the two roles did not share a budget"


@pytest.mark.asyncio
async def test_saturation_raises_and_is_NOT_a_node_fault():
    """The second half of the defect. A request that never left this process
    cannot be evidence about the node — the breaker must not see it."""
    c = _client()
    n = _node(slots=1)

    async def hold():
        async with c._node_slot(n, wait_timeout=5):
            await asyncio.sleep(0.5)

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0.05)
    with pytest.raises(NodeSaturated):
        async with c._node_slot(n, wait_timeout=0.05):
            pass                                   # pragma: no cover
    await holder

    exc = NodeSaturated("no free slot")
    assert _is_node_fault(exc) is False
    assert "not a node fault" in _node_error_detail(exc)
    # and a real timeout is STILL a node fault — the narrowing must not
    # swallow the case the breaker exists for
    import httpx
    assert _is_node_fault(httpx.ReadTimeout("x")) is True


@pytest.mark.asyncio
async def test_the_permit_is_RELEASED_when_the_request_raises():
    """A leaked permit would shrink the node's usable capacity a little at a
    time until everything queued — a slow-motion version of the same outage."""
    c = _client()
    n = _node(slots=1)
    for _ in range(3):
        with pytest.raises(ValueError):
            async with c._node_slot(n, wait_timeout=1):
                raise ValueError("boom")
    async with c._node_slot(n, wait_timeout=0.2):   # still acquirable
        pass


@pytest.mark.asyncio
async def test_the_health_PROBE_bypasses_the_gate_entirely():
    """⚠ `keepalive_workers()` prints "node X stopped answering". If its ping
    queued behind the traffic it watches, a BUSY node would report as a DEAD
    one — the exact false alarm, reintroduced one layer down."""
    c = _client()
    n = _node(slots=1)

    async def hold():
        async with c._node_slot(n, wait_timeout=5):
            await asyncio.sleep(0.4)

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0.05)
    # wait_timeout=None is BYPASS, not "wait forever" — it must return at once
    # even with every permit held.
    await asyncio.wait_for(_probe(c, n), timeout=0.15)
    await holder


async def _probe(c, n):
    async with c._node_slot(n, wait_timeout=None):
        return True


def test_the_dispatch_path_actually_USES_the_gate():
    """⚠ THE SEAM. Every helper above can be perfect while `_do_chat_completion`
    still posts un-gated — which is precisely the state this file describes.
    Pins that every node POST is wrapped, and that the probe is exempted."""
    import inspect
    src = inspect.getsource(LLMClient._do_chat_completion)
    posts = src.count('node["client"].post("/v1/chat/completions"')
    gated = src.count("self._node_slot(node, wait_timeout=_slot_wait)")
    assert posts >= 5, f"expected the node POST sites, found {posts}"
    assert gated == posts, (
        f"{posts} node POSTs but only {gated} behind the gate — an un-gated "
        "path is a hole in a budget that only works if it is total")
    assert 'task_label == "keepalive"' in src and "_slot_wait = None" in src, (
        "the health probe must be exempt, or a busy node reads as a dead one")
