"""§4BV R5 — the node-gate faults introduced or missed by R4.

Every test here corresponds to a measurement, not a code reading. R4's own
`_node_slot` rework introduced two of these; R4 edited past a third on the
exact line it changed.
"""

import asyncio
import time

from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.llm import LLMClient


def _probe_client(slots=None, fail=False, hang=0.0):
    cl = MagicMock()
    if fail or hang:
        async def _get(*a, **kw):
            if hang:
                await asyncio.sleep(hang)
            raise RuntimeError("no /props here")
        cl.get = AsyncMock(side_effect=_get)
    else:
        r = MagicMock()
        r.json = lambda: {"total_slots": slots}
        r.raise_for_status = lambda: None
        cl.get = AsyncMock(return_value=r)
    return cl


class TestOneSickNodeDoesNotStallTheOthers:
    """R5 lens A, MAJOR-3. The capacity probe ran INSIDE `_node_slots_lock` —
    a single lock shared by every node — so a 5s `/props` timeout on one node
    blocked dispatch to all of them. R4 made it recurring rather than
    once-per-process by adding a 300s re-probe TTL.

    Live shape: the image-gen node is a Jetson that does not serve `/props`
    at all, and Nova goes down on every co-restart."""

    def test_a_hanging_probe_does_not_block_another_nodes_dispatch(self):
        c = LLMClient("http://main.invalid:8088")
        dead = {"url": "http://JETSON", "model": "Ghost",
                "client": _probe_client(hang=1.0)}
        ok = {"url": "http://NOVA", "model": "Nova",
              "client": _probe_client(slots=4)}

        async def _touch(n):
            t0 = time.monotonic()
            async with c._node_slot(n, wait_timeout=8):
                pass
            return time.monotonic() - t0

        async def _both():
            return await asyncio.gather(_touch(dead), _touch(ok))

        slow, fast = asyncio.run(_both())
        assert slow >= 0.9, "the hanging probe did not actually hang"
        assert fast < 0.5, (
            f"a healthy node's dispatch took {fast:.2f}s because an "
            f"unrelated node was being probed — one lock for all nodes "
            f"couples every pool to the sickest box in the fleet")

    def test_each_node_gets_its_own_lock(self):
        c = LLMClient("http://main.invalid:8088")
        assert hasattr(c, "_node_slot_locks"), (
            "capacity probing is back on a single global lock")


class TestAResizeBoundsTheWaitersAlreadyQueued:
    """R5 lens A, MAJOR-4. R4 REPLACED the semaphore on a capacity
    correction and documented the over-subscription window as "the duration
    of those requests". Queued waiters stay parked on the OLD object and keep
    being admitted through it, so the window is
    `queue_depth / old_cap x request_duration` — minutes under a fan-out.
    Measured 4 concurrent against a `total_slots=1` node."""

    def test_no_waiter_is_admitted_beyond_the_corrected_capacity(self):
        c = LLMClient("http://main.invalid:8088")
        node = {"url": "http://N", "model": "m",
                "client": _probe_client(fail=True)}
        live = {"now": 0, "peak": 0}

        async def _work():
            async with c._node_slot(node, wait_timeout=30):
                live["now"] += 1
                live["peak"] = max(live["peak"], live["now"])
                await asyncio.sleep(0.25)
                live["now"] -= 1

        async def _drive():
            tasks = [asyncio.ensure_future(_work()) for _ in range(12)]
            await asyncio.sleep(0.05)
            guessed = c._node_slot_built_cap["http://N"]
            before = c._node_slots["http://N"]
            node["client"] = _probe_client(slots=1)     # node comes up, -np 1
            c._node_cap_retry_at.clear()
            async with c._node_slot(node, wait_timeout=30):
                pass
            # ⚠ WORK MUST ARRIVE AFTER THE RESIZE. The first version of this
            # test queued everything up front, so all 12 waiters were parked
            # on the pre-resize semaphore and drained through it at the old
            # cap either way — peak 3 whether the object was replaced or
            # resized, and the replacement mutant survived. The stranded-
            # waiter fault only shows when NEW arrivals are governed by a
            # different object than the ones already queued.
            tasks += [asyncio.ensure_future(_work()) for _ in range(6)]
            await asyncio.gather(*tasks)
            return guessed, before

        guessed, before = asyncio.run(_drive())
        assert before is c._node_slots["http://N"], (
            "the semaphore object was swapped — everything already queued is "
            "now governed by an object the gate has forgotten")
        assert guessed == c._node_slot_default
        assert c._node_slot_built_cap["http://N"] == 1, "the gate never resized"
        # In-flight holders cannot have their permits revoked, so the guessed
        # capacity is the floor. What must NOT happen is a waiter being let
        # through on top of them via a stranded second semaphore.
        assert live["peak"] <= guessed, (
            f"peak {live['peak']} concurrent against a total_slots=1 node "
            f"exceeded even the pre-correction guess of {guessed} — waiters "
            f"were admitted through a semaphore that no longer governs")

    def test_growing_the_capacity_releases_rather_than_replaces(self):
        c = LLMClient("http://main.invalid:8088")
        node = {"url": "http://G", "model": "m",
                "client": _probe_client(fail=True)}

        async def _drive():
            async with c._node_slot(node, wait_timeout=5):
                pass
            first = c._node_slots["http://G"]
            node["client"] = _probe_client(slots=8)
            c._node_cap_retry_at.clear()
            async with c._node_slot(node, wait_timeout=5):
                pass
            return first is c._node_slots["http://G"]

        assert asyncio.run(_drive()), (
            "the semaphore object was swapped instead of resized — anything "
            "already queued on the old one is no longer governed by the gate")
        assert c._node_slot_built_cap["http://G"] == 8


class TestWarmupDoesNotTripTheBreakerItServes:
    """R5 lens A, MAJOR-5. `_slots` defaults to 3 for an unprobeable node and
    `failure_threshold` is 3, so boot warmup against a node still coming up
    opened that node's own breaker — 60s of degraded query expansion on every
    co-restart, at exactly the moment warmup exists to prevent it."""

    def _client(self):
        c = LLMClient("http://main.invalid:8088")
        cl = MagicMock()
        cl.post = AsyncMock(side_effect=RuntimeError("still booting"))
        cl.get = AsyncMock(side_effect=RuntimeError("still booting"))
        c.worker_clients = [{"url": "http://NOVA", "model": "Nova",
                             "client": cl, "name": "Nova"}]
        c.critic_clients = list(c.worker_clients)
        c._worker_index = c._critic_index = 0
        return c, cl

    def test_a_node_still_booting_records_one_failure_not_three(self):
        c, cl = self._client()
        asyncio.run(c.warm_up_workers())
        st = c.circuit_breaker.get_status().get("http://NOVA", {})
        assert st.get("state") != "open", (
            f"boot warmup opened the breaker on the node it was warming: "
            f"{st} — route() will now refuse a node that is up, for 60s")
        assert cl.post.await_count == 1, (
            f"{cl.post.await_count} doomed warmups against a down node; one "
            f"failure is information, three is the breaker threshold")

    def test_a_failed_boot_probe_does_not_freeze_the_capacity_guess(self):
        """Boot is the worst moment to cache a capacity guess for 300s — the
        node is most likely to be mid-restart, and it may be up seconds
        later."""
        c, _ = self._client()
        asyncio.run(c.warm_up_workers())
        assert "http://NOVA" not in c._node_cap_retry_at, (
            "a node that comes up ten seconds after boot stays mis-sized for "
            "five minutes")

    def test_a_healthy_node_still_warms_every_slot(self):
        c, cl = self._client()
        r = MagicMock()
        r.json = lambda: {"choices": [{"message": {"content": "ok"}}]}
        r.raise_for_status = lambda: None
        r.status_code, r.text = 200, "{}"
        cl.post = AsyncMock(return_value=r)
        props = MagicMock()
        props.json = lambda: {"total_slots": 4}
        props.raise_for_status = lambda: None
        cl.get = AsyncMock(return_value=props)
        asyncio.run(c.warm_up_workers())
        assert cl.post.await_count == 4, (
            f"a healthy 4-slot node warmed {cl.post.await_count} slot(s) — "
            f"probing with one must not cost the fan-out when it succeeds")


def test_image_generation_is_bounded_across_all_three_attempts():
    """R5 lens B. `for attempt in range(3)` each spent the full permit wait:
    3 x 90s plus backoffs, ~273s of pure queueing for one image."""
    import inspect
    src = inspect.getsource(LLMClient.generate_image)
    assert "_img_slot_wait_now()" in src, (
        "the image path takes a fresh permit budget on every retry")
    assert "_img_deadline" in src


def test_saturation_reports_the_capacity_actually_enforced():
    """R5 lens A. `_node_slot_caps` holds only SUCCESSFUL probes, so after a
    failed one the saturation error said "cap None" while the gate was really
    sized from the default — an operator debugging saturation was shown a
    number that does not exist."""
    from ghost_agent.core.llm import NodeSaturated
    import pytest

    c = LLMClient("http://main.invalid:8088")
    node = {"url": "http://N", "model": "m", "client": _probe_client(fail=True)}

    async def _drive():
        async with c._node_slot(node, wait_timeout=5):      # take the permits
            async with c._node_slot(node, wait_timeout=5):
                async with c._node_slot(node, wait_timeout=5):
                    async with c._node_slot(node, wait_timeout=0.05):
                        pass

    with pytest.raises(NodeSaturated) as ei:
        asyncio.run(_drive())
    msg = str(ei.value)
    assert "cap None" not in msg, f"reports a capacity that does not exist: {msg}"
    assert f"cap {c._node_slot_default}" in msg, msg


def test_close_survives_a_bad_upstream_socket():
    """R5 lens C, P11. R4 wrapped every `aclose()` in `suppress` because the
    upstream one runs FIRST and a raise there leaked all six node pools.
    Nothing pinned it."""
    c = LLMClient("http://main.invalid:8088")
    c.http_client = MagicMock()
    c.http_client.aclose = AsyncMock(side_effect=RuntimeError("bad socket"))
    node_client = MagicMock()
    node_client.aclose = AsyncMock()
    c.worker_clients = [{"url": "http://W", "model": "m",
                         "client": node_client, "name": "W"}]
    c._node_slots["http://W"] = asyncio.Semaphore(1)
    c._node_slot_built_cap["http://W"] = 1

    asyncio.run(c.close())
    node_client.aclose.assert_awaited(), (
        "one failing upstream socket leaked every node pool's client")
    assert not c._node_slots and not c._node_slot_built_cap, (
        "gate state survived close() — a reused client would account permits "
        "against clients that no longer exist")


def test_the_corpus_records_which_leg_SERVED_not_which_was_requested():
    """R5 lens C, N18. The §4BG training corpus filed a main-served critic
    call as `use_critic=True` — wrong model provenance on exactly the
    degraded turns worth studying. The fix had no test at all."""
    c = LLMClient("http://main.invalid:8088")
    seen = {}
    c._maybe_record_call = lambda *a, **kw: seen.update(kw)

    dead = MagicMock()
    dead.post = AsyncMock(side_effect=RuntimeError("worker down"))
    c.worker_clients = [{"url": "http://W", "model": "m",
                         "client": dead, "name": "W"}]
    c._worker_index = 0

    async def _main(*a, **kw):
        r = MagicMock()
        r.json = lambda: {"choices": [{"message": {"content": "ok"}}]}
        r.raise_for_status = lambda: None
        r.status_code, r.text = 200, "{}"
        return r

    c.http_client = MagicMock()
    c.http_client.post = AsyncMock(side_effect=_main)
    asyncio.run(c.chat_completion({"model": "m", "messages": []},
                                  use_worker=True, timeout=5))
    assert seen.get("use_worker") is False, (
        "a call the MAIN model answered was filed in the corpus as "
        "worker-served — the provenance is wrong on precisely the degraded "
        "turns the corpus exists to study")
    assert seen.get("served_by") == "main"
    assert seen.get("requested_pool") == "worker", (
        "the request is still worth recording — it is the pair that carries "
        "the information")
