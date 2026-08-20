"""§4BV R7 — the STREAMING coding path joins the discipline the NON-STREAMING
path has had since 2026-08-11, plus the stall-attribution and byte-frame
fixes in the same function.

Every test here was mutation-checked: the mutation it kills is named in its
docstring. `_do_stream_chat_completion` is the path every user turn and every
sub-agent turn streams through, so these are deliberately BEHAVIOURAL — they
run the generator and read the effects, never the source text.
"""
import asyncio
import contextlib
import json

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core import llm as llm_mod
from ghost_agent.core.llm import LLMClient

MAIN = "http://main.invalid:8088"
N1 = "http://cnode-1.invalid:8000"
N2 = "http://cnode-2.invalid:8000"


# ── fixtures ────────────────────────────────────────────────────────────────
def _props(slots):
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json = MagicMock(return_value={"total_slots": slots})
    return r


def _sse(text="hi", lines=None, closed=None):
    items = lines if lines is not None else [
        'data: {"choices": [{"delta": {"content": "%s"}}]}' % text,
        'data: [DONE]',
    ]

    async def _gen():
        for it in items:
            yield it

    r = MagicMock()
    r.status_code = 200
    r.raise_for_status = MagicMock()
    r.aiter_lines = _gen

    async def _aclose():
        if closed is not None:
            closed.append(1)
    r.aclose = _aclose
    return r


def _never_speaks():
    async def _gen():
        await asyncio.sleep(3600)
        yield "data: [DONE]"
    r = MagicMock()
    r.status_code = 200
    r.raise_for_status = MagicMock()
    r.aiter_lines = _gen
    r.aclose = AsyncMock()
    return r


def _wire(node_client, *, slots=2, behaviour=None):
    """Give a node client a working /props probe and a `send`."""
    node_client.get = AsyncMock(return_value=_props(slots))
    node_client.build_request = MagicMock(return_value=MagicMock())
    node_client.send = AsyncMock(side_effect=behaviour)
    return node_client


def _client(nodes, *, upstream=MAIN):
    return LLMClient(upstream_url=upstream, coding_nodes=nodes)


async def _drain(c, **kw):
    out = []
    async for ch in c.stream_chat_completion(
            {"messages": [{"role": "user", "content": "x"}], "model": "caller-model"},
            use_coding=True, **kw):
        out.append(ch.decode("utf-8") if isinstance(ch, (bytes, bytearray)) else ch)
    return "".join(out)


# ── 1. the pool is inside the per-node concurrency gate ─────────────────────
@pytest.mark.asyncio
async def test_the_streaming_coding_path_takes_a_node_permit():
    """MUTATION KILLED: drop the `_node_slot` acquisition from the streaming
    coding branch (its pre-R7 state — measured 0 permits over 3 calls while
    the non-streaming sibling took 6)."""
    c = _client([{"url": N1, "model": "coder"}])
    entered = []

    async def send(req, **kw):
        return _sse()
    _wire(c.coding_clients[0]["client"], slots=2, behaviour=send)

    real = type(c)._node_slot

    @contextlib.asynccontextmanager
    async def counting(node, wait_timeout=None):
        async with real(c, node, wait_timeout=wait_timeout):
            entered.append(node["url"])
            yield
    c._node_slot = counting

    await _drain(c)
    assert entered == [N1], (
        "the streaming coding path never entered the per-node gate — the "
        "pool is outside the budget every other caller shares")
    await c.close()


# ── 2. the permit covers GENERATION, not just the handshake ────────────────
@pytest.mark.asyncio
async def test_the_permit_is_held_across_the_yields_not_just_the_handshake():
    """MUTATION KILLED: release the permit as soon as `send()` returns
    (`async with self._node_slot(...): resp = await send(...)`) — the
    "serialise the handshake" non-fix already rejected for `_main_node_lock`,
    which looks like a gate and bounds nothing."""
    c = _client([{"url": N1, "model": "coder"}])

    async def send(req, **kw):
        return _sse()
    _wire(c.coding_clients[0]["client"], slots=1, behaviour=send)

    seen = []
    async for _ch in c.stream_chat_completion(
            {"messages": [], "model": "m"}, use_coding=True):
        sem = c._node_slots.get(N1)
        seen.append(None if sem is None else sem._value)
    assert seen and all(v == 0 for v in seen), (
        f"the node's only permit was free while its own generation was still "
        f"streaming (values seen: {seen}) — the gate bounds the handshake, "
        f"not the slot")
    assert c._node_slots[N1]._value == 1, "permit not released on completion"
    await c.close()


# ── 3. the permit comes back when the consumer walks away ──────────────────
@pytest.mark.asyncio
async def test_the_permit_comes_back_when_the_consumer_abandons_the_stream():
    """MUTATION KILLED: acquire the permit with a bare `__aenter__()` whose
    context manager is stored on `self` (so it is never dropped and never
    finalised) — i.e. any acquisition that does not ride the same
    finalisation `resp.aclose()` already depends on.

    ⚠ NO `gc.collect()`. The point is that plain refcounting + asyncio's
    async-generator finalizer hooks are enough."""
    c = _client([{"url": N1, "model": "coder"}])
    closed = []

    async def send(req, **kw):
        async def _forever():
            yield 'data: {"choices": [{"delta": {"content": "a"}}]}'
            await asyncio.sleep(3600)
        r = MagicMock()
        r.status_code = 200
        r.raise_for_status = MagicMock()
        r.aiter_lines = _forever

        async def _aclose():
            closed.append(1)
        r.aclose = _aclose
        return r
    _wire(c.coding_clients[0]["client"], slots=1, behaviour=send)

    async def wrapper(inner):                     # agent.stream_wrapper shape
        async for ch in inner:
            yield ch

    async def unregister(gen):                    # _stream_then_unregister
        try:
            async for ch in gen:
                yield ch
        finally:
            pass

    held = {}

    async def consume():                          # routes.stream_generator
        gen = unregister(wrapper(c.stream_chat_completion(
            {"messages": [], "model": "m"}, use_coding=True)))
        async for _ch in gen:
            held["value"] = c._node_slots[N1]._value
            break
        del gen

    await consume()
    assert held.get("value") == 0, "permit was not held while streaming"

    sem = c._node_slots[N1]
    for _ in range(200):
        if sem._value == 1 and closed and c.foreground_tasks == 0:
            break
        await asyncio.sleep(0)
    assert sem._value == 1, (
        "an abandoned stream leaked one of the node's slots forever")
    assert closed, "resp.aclose() did not run either — check the harness"
    assert c.foreground_tasks == 0
    assert not c._inflight_by_url, (
        f"the in-flight counter leaked an abandoned stream: "
        f"{c._inflight_by_url}")
    await c.close()


# ── 4. the retry moves to the NEXT node ────────────────────────────────────
@pytest.mark.asyncio
async def test_the_retry_moves_to_the_next_node_not_the_same_dead_one():
    """MUTATION KILLED: hoist the node selection above the retry loop (its
    pre-R7 state) — `client_to_use` was bound once, so both attempts hit the
    same dead box (measured DEAD,DEAD)."""
    hit = []
    c = _client([{"url": N1, "model": "coder-1"}, {"url": N2, "model": "coder-2"}])

    def mk(name, dead):
        async def send(req, **kw):
            hit.append(name)
            if dead:
                raise httpx.ConnectError("refused")
            return _sse()
        return send
    _wire(c.coding_clients[0]["client"], behaviour=mk("coder-1", True))
    _wire(c.coding_clients[1]["client"], behaviour=mk("coder-2", False))

    body = await _drain(c)
    assert hit == ["coder-1", "coder-2"], hit
    assert '"content": "hi"' in body
    await c.close()


# ── 5. the breaker learns ──────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_a_failing_coding_node_is_recorded_against_the_circuit_breaker():
    """MUTATION KILLED: drop `circuit_breaker.record_failure` from the
    streaming branch (its pre-R7 state — measured `failures=0 closed` after 3
    calls to a dead node, which makes `get_coding_node`'s `is_available()`
    filtering provably dead code on this path)."""
    c = _client([{"url": N1, "model": "coder-1"}, {"url": N2, "model": "coder-2"}])

    async def dead(req, **kw):
        raise httpx.ConnectError("refused")

    async def alive(req, **kw):
        return _sse()
    _wire(c.coding_clients[0]["client"], behaviour=dead)
    _wire(c.coding_clients[1]["client"], behaviour=alive)

    for _ in range(3):
        await _drain(c)
    st = c.circuit_breaker._states.get(N1, {})
    assert st.get("failures", 0) >= 3 and st.get("state") == "open", (
        f"the dead node was contacted repeatedly and the breaker still reads "
        f"{st} — it can never filter it out")
    await c.close()


@pytest.mark.asyncio
async def test_a_healthy_coding_node_records_a_success():
    """MUTATION KILLED: drop `circuit_breaker.record_success` — without it a
    node that recovers is never re-closed and stays half_open forever."""
    c = _client([{"url": N1, "model": "coder"}])

    async def send(req, **kw):
        return _sse()
    _wire(c.coding_clients[0]["client"], behaviour=send)
    c.circuit_breaker.record_failure(N1)
    c.circuit_breaker.record_failure(N1)
    await _drain(c)
    assert c.circuit_breaker._states[N1]["failures"] == 0
    await c.close()


# ── 6. saturation is OURS, never the node's ────────────────────────────────
@pytest.mark.asyncio
async def test_saturation_is_never_charged_to_the_node():
    """MUTATION KILLED: remove the `except NodeSaturated` arm so it falls into
    the generic handler and calls `record_failure` — a node cannot fail a
    request it was never asked (see `_is_node_fault`)."""
    c = _client([{"url": N1, "model": "coder-1"}, {"url": N2, "model": "coder-2"}])
    reached = []

    async def send1(req, **kw):
        reached.append("coder-1")
        return _sse()

    async def send2(req, **kw):
        reached.append("coder-2")
        return _sse()
    _wire(c.coding_clients[0]["client"], slots=1, behaviour=send1)
    _wire(c.coding_clients[1]["client"], slots=1, behaviour=send2)

    # Occupy node 1's only permit for longer than any budget we allow.
    c._node_slots[N1] = asyncio.Semaphore(1)
    c._node_slot_caps[N1] = 1
    c._node_slot_built_cap[N1] = 1
    await c._node_slots[N1].acquire()

    import os
    os.environ["GHOST_NODE_SLOT_WAIT_S"] = "0.2"
    try:
        body = await _drain(c)
    finally:
        os.environ.pop("GHOST_NODE_SLOT_WAIT_S", None)

    assert reached == ["coder-2"], reached
    assert '"content": "hi"' in body
    assert c.circuit_breaker._states.get(N1, {}).get("failures", 0) == 0, (
        "a node we never asked was charged with a failure")
    await c.close()


# ── 7. the queue budget is SHARED and never zero ───────────────────────────
@pytest.mark.asyncio
async def test_the_last_node_of_a_pool_still_gets_a_usable_permit_budget():
    """MUTATION KILLED (two of them):
      * `max(0.0, ...)` instead of `max(_MIN_ACQUIRE, ...)` in `_permit_wait`;
      * `len(_coding_pool) - len(_tried)` instead of `... + 1` (`untried` must
        INCLUDE the current attempt).
    Either one hands the LAST node of the pool exactly 0.0s, and
    `asyncio.wait_for(sem.acquire(), 0.0)` refuses a COMPLETELY FREE
    semaphore — so the final, idle, healthy node is never asked. Both are the
    non-streaming path's documented scars, and both were absent from the R6
    draft of this function."""
    import os
    c = _client([{"url": N1, "model": "coder-1"}, {"url": N2, "model": "coder-2"}])
    reached = []
    offered = []

    async def dead(req, **kw):
        reached.append("coder-1")
        # ⚠ BURN THE WHOLE POOL BUDGET DELIBERATELY. Without this the test is
        # a race — a sub-millisecond mock finishes with budget to spare and
        # the `max(0.0, ...)` mutant passes for the wrong reason. Node 2 must
        # be reached with the deadline PROVABLY in the past.
        await asyncio.sleep(0.05)
        raise httpx.ConnectError("refused")

    async def alive(req, **kw):
        reached.append("coder-2")
        return _sse()
    _wire(c.coding_clients[0]["client"], slots=1, behaviour=dead)
    _wire(c.coding_clients[1]["client"], slots=1, behaviour=alive)

    real = type(c)._node_slot

    @contextlib.asynccontextmanager
    async def spy(node, wait_timeout=None):
        async with real(c, node, wait_timeout=wait_timeout):
            yield
    # resolve what the gate was actually offered, at the moment it was offered
    @contextlib.asynccontextmanager
    async def recording(node, wait_timeout=None):
        if callable(wait_timeout):
            offered.append((node["url"], wait_timeout()))
        async with real(c, node, wait_timeout=wait_timeout):
            yield
    c._node_slot = recording

    os.environ["GHOST_NODE_SLOT_WAIT_S"] = "0.02"
    try:
        body = await _drain(c)
    finally:
        os.environ.pop("GHOST_NODE_SLOT_WAIT_S", None)
    assert reached == ["coder-1", "coder-2"], (
        f"the second node was refused a permit while completely idle, with "
        f"every one of its slots free (contacted: {reached}; budgets offered: "
        f"{offered})")
    assert offered[-1][0] == N2
    assert offered[-1][1] == pytest.approx(llm_mod._MIN_ACQUIRE), (
        f"node 2 was offered {offered[-1][1]!r}s after the pool deadline had "
        f"passed — `asyncio.wait_for(sem.acquire(), 0.0)` refuses a "
        f"COMPLETELY FREE semaphore, so 0 means 'never ask this node'")
    assert '"content": "hi"' in body
    await c.close()


@pytest.mark.asyncio
async def test_the_gate_wait_is_a_callable_resolved_after_the_capacity_probe():
    """MUTATION KILLED: pass a plain float (`wait_timeout=_permit_wait(n)`).
    An argument EXPRESSION is evaluated before `_node_slot` is entered, so its
    `/props` probe is spent entirely OUTSIDE the budget it is meant to be
    inside — `_node_slot`'s own comment records this being measured (a stated
    12s budget taking 14.01s), and the R6 draft reintroduced it."""
    c = _client([{"url": N1, "model": "coder"}])
    kinds = []

    async def send(req, **kw):
        return _sse()
    _wire(c.coding_clients[0]["client"], behaviour=send)

    real = type(c)._node_slot

    @contextlib.asynccontextmanager
    async def spy(node, wait_timeout=None):
        kinds.append(callable(wait_timeout))
        async with real(c, node, wait_timeout=wait_timeout):
            yield
    c._node_slot = spy

    await _drain(c)
    assert kinds == [True], (
        "the streaming path handed the gate a pre-resolved number, so the "
        "capacity probe is charged to nobody's budget")
    await c.close()


# ── 8. main fallback ───────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_every_coding_node_dead_falls_back_to_main():
    """MUTATION KILLED: remove the main-model fallback leg (its pre-R7 state —
    one dead coding node was a HARD FAILURE for every coding stream, forever,
    while the non-streaming sibling has always fallen back)."""
    c = _client([{"url": N1, "model": "coder-1"}, {"url": N2, "model": "coder-2"}])

    async def dead(req, **kw):
        raise httpx.ConnectError("refused")
    _wire(c.coding_clients[0]["client"], behaviour=dead)
    _wire(c.coding_clients[1]["client"], behaviour=dead)

    sent = {}

    def build_request(method, url, json=None, **kw):
        sent["model"] = (json or {}).get("model", "<absent>")
        return MagicMock()
    c.http_client.build_request = MagicMock(side_effect=build_request)

    async def main_send(req, **kw):
        return _sse("main")
    c.http_client.send = AsyncMock(side_effect=main_send)

    body = await _drain(c)
    assert '"content": "main"' in body, (
        "a dead coding pool is still a hard failure on the streaming path")
    assert sent["model"] == "caller-model", (
        f"the fallback POSTed model={sent['model']!r} to the 35B — the "
        f"per-node model rewrite was never undone, so main gets a 404")
    await c.close()


# ── 9. an EMPTY pool must be indistinguishable from the pre-R7 code ────────
@pytest.mark.asyncio
async def test_an_empty_coding_pool_is_still_exactly_two_main_attempts():
    """MUTATION KILLED: `_max_attempts = len(_coding_pool) + 2` written
    unconditionally (so an empty pool becomes 2 anyway — equivalent) OR any
    change that alters the main-path attempt count. This is the live
    topology: `--coding-nodes` is EMPTY, so this test is what says the whole
    change is a no-op in production."""
    c = LLMClient(upstream_url=MAIN)
    c.http_client = MagicMock()
    c.http_client.base_url = MAIN
    c.http_client.build_request = MagicMock(return_value=MagicMock())
    c.http_client.send = AsyncMock(side_effect=httpx.ConnectError("boom"))

    chunks = []
    with pytest.raises(Exception):
        async for ch in c._do_stream_chat_completion({"model": "m", "messages": []},
                                                     use_coding=True):
            chunks.append(ch)
    assert c.http_client.send.await_count == 2, c.http_client.send.await_count
    assert any(b"[DONE]" in ch for ch in chunks)
    assert any(b"2 attempts" in ch for ch in chunks), (
        "the operator-facing count no longer matches the attempts actually made")


# ── 10. stall attribution ──────────────────────────────────────────────────
def _stall_client(cap, upstream=MAIN):
    # ⚠ KEEP THE REAL httpx client and mock only its METHODS. Replacing it
    # with a MagicMock and hand-setting `base_url` destroys the very property
    # under test: httpx NORMALISES base_url in its constructor, and a
    # hand-set attribute does not. A mutant keying the verdict on
    # `str(client.base_url)` survived against a mocked client and dies
    # against this one.
    c = LLMClient(upstream_url=upstream)
    c.http_client.build_request = MagicMock(return_value=MagicMock())

    async def _send(req, **kw):
        return _never_speaks()
    c.http_client.send = AsyncMock(side_effect=_send)
    c.http_client.get = (AsyncMock(return_value=_props(cap)) if cap
                         else AsyncMock(side_effect=RuntimeError("no /props")))
    return c


async def _stall(c, n, monkeypatch, cap, hold_main_lock=False):
    if cap:
        # exactly what a vision call to the byte-identical --visual-nodes URL
        # does; nothing here probes on main's behalf
        await c._node_capacity({"url": str(c.upstream_url),
                                "client": c.http_client, "model": "main"})
    monkeypatch.setattr(llm_mod, "_STREAM_FIRST_BYTE_TIMEOUT", 0.25)
    log = []
    monkeypatch.setattr(llm_mod, "pretty_log",
                        lambda title, msg="", **kw: log.append((title, msg)))
    frames = []

    async def one():
        buf = []
        async for ch in c._do_stream_chat_completion({"model": "m", "messages": []}):
            buf.append(ch.decode())
        frames.append("".join(buf))

    holder = None
    if hold_main_lock:
        async def hold():
            async with c._main_node_lock:
                await asyncio.sleep(5)
        holder = asyncio.create_task(hold())
        await asyncio.sleep(0)
    await asyncio.gather(*[one() for _ in range(n)])
    if holder:
        holder.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await holder
    return log, frames


@pytest.mark.asyncio
async def test_a_self_inflicted_prefill_queue_is_not_blamed_on_the_upstream(monkeypatch):
    """MUTATION KILLED: remove the attribution and log "Upstream Stream Stall"
    unconditionally (its pre-R7 state — measured 3 streams + 1 POST at a
    1-slot node, and 2 of 3 streams reported the UPSTREAM as stalled for a
    queue we created)."""
    c = _stall_client(1)
    log, frames = await _stall(c, 3, monkeypatch, cap=1, hold_main_lock=True)
    titles = {t for t, _ in log}
    assert titles == {"Stream Stall (Self-Queued)"}, titles
    assert all("OUR OWN" in m for _, m in log)
    assert all("1-slot node" in f for f in frames)
    await c.close()


@pytest.mark.asyncio
async def test_a_sole_caller_stall_is_still_the_upstreams_fault(monkeypatch):
    """FALSE-POSITIVE CONTROL. MUTATION KILLED: issue the self-queued verdict
    whenever a capacity is known (`_conc >= _cap` instead of `>`), which
    accuses us on every single-caller stall against the 1-slot main node —
    i.e. on essentially every real stall in production."""
    c = _stall_client(1)
    log, frames = await _stall(c, 1, monkeypatch, cap=1)
    assert [t for t, _ in log] == ["Upstream Stream Stall"], log
    assert "sole in-flight request" in log[0][1]
    assert "OUR OWN" not in log[0][1]
    assert "concurrent" not in frames[0]
    await c.close()


@pytest.mark.asyncio
async def test_an_unprobed_node_never_produces_a_self_queued_verdict(monkeypatch):
    """FALSE-POSITIVE CONTROL. MUTATION KILLED: read the capacity from
    `_node_slot_built_cap` / `_node_slot_default` instead of `_node_slot_caps`
    — those hold the GUESS (3) that a FAILED probe leaves behind, so 4
    concurrent requests against a node we know nothing about would be
    announced as a proven self-inflicted queue."""
    c = _stall_client(None)                       # /props raises
    log, frames = await _stall(c, 3, monkeypatch, cap=None)
    assert {t for t, _ in log} == {"Upstream Stream Stall"}, log
    assert all("capacity unknown" in m for _, m in log)
    assert all("OUR OWN" not in m for _, m in log)
    await c.close()


@pytest.mark.asyncio
async def test_concurrency_within_a_known_capacity_says_so(monkeypatch):
    """MUTATION KILLED: print "capacity unknown" whenever the self-queued
    verdict did not fire. The two reasons it can not fire are opposites —
    "we never probed" and "we probed and we are within it" — and reporting
    the second as the first sends the operator to look for a probe that
    already succeeded."""
    c = _stall_client(4)
    log, _ = await _stall(c, 3, monkeypatch, cap=4)
    assert all("within capacity" in m for _, m in log), log
    assert all("capacity unknown" not in m for _, m in log)
    await c.close()


@pytest.mark.asyncio
async def test_the_attribution_is_keyed_on_the_configured_url(monkeypatch):
    """MUTATION KILLED: key the verdict on `str(client.base_url)`.
    `_node_slot_caps` — the only source of a real capacity — is keyed on the
    CONFIGURED url, and httpx normalises its `base_url`: measured
    `http://h:8088/v1` -> `http://h:8088/v1/`, `http://h:80` -> `http://h`,
    `HTTP://H:8088` -> `http://h:8088`. On any of those topologies the
    normalised key never matches and the whole verdict is dead code — the
    silent-inoperative-guard shape this project keeps re-finding."""
    upstream = "http://main.invalid:8088/v1"
    assert str(httpx.AsyncClient(base_url=upstream).base_url) != upstream, (
        "httpx stopped normalising — this test's premise is gone")
    c = _stall_client(1, upstream=upstream)
    log, _ = await _stall(c, 3, monkeypatch, cap=1, hold_main_lock=True)
    assert {t for t, _ in log} == {"Stream Stall (Self-Queued)"}, log
    await c.close()


@pytest.mark.asyncio
async def test_a_subsecond_stall_budget_is_not_rendered_as_zero_seconds(monkeypatch):
    """MUTATION KILLED: `{_timeout:.0f}` (its pre-R7 state). A sub-second
    budget printed "No bytes for 0s", which reads like a disabled guard —
    the identical defect `_node_slot`'s `NodeSaturated` message was fixed for.
    Observable in the existing suite: test_stream_idle_timeout.py runs this
    path at 0.2s."""
    c = _stall_client(None)
    log, frames = await _stall(c, 1, monkeypatch, cap=None)
    assert "0.25s" in log[0][1], log[0][1]
    assert "for 0s" not in log[0][1]
    assert "0.25s" in frames[0]
    await c.close()


# ── 11. a bytes chunk is a frame, not a repr ───────────────────────────────
@pytest.mark.asyncio
async def test_a_bytes_chunk_is_forwarded_as_a_frame_not_a_python_repr():
    """MUTATION KILLED: `yield f"{chunk}\\n\\n".encode('utf-8')` for every
    chunk (its pre-R7 state). The usage pre-filter two lines above already
    documents that a chunk "is str on the real `aiter_lines` path but bytes on
    others", and the emit interpolated the bytes object — shipping
    `b'data: {...}'`, a frame no SSE parser accepts. Invisible to the existing
    test, whose assertion is a SUBSTRING that a repr still contains."""
    c = _client([{"url": N1, "model": "coder"}])

    async def send(req, **kw):
        return _sse(lines=[b'data: {"choices": [{"delta": {"content": "b"}}]}',
                           b'data: [DONE]'])
    _wire(c.coding_clients[0]["client"], behaviour=send)

    out = []
    async for ch in c.stream_chat_completion({"messages": [], "model": "m"},
                                             use_coding=True):
        out.append(ch)
    assert out[0].startswith(b"data: "), out[0]
    assert not out[0].startswith(b"b'"), (
        f"a bytes chunk was emitted as its Python repr: {out[0]!r}")
    # and it must actually parse as SSE
    payload = out[0].decode().split("data: ", 1)[1].strip()
    assert json.loads(payload)["choices"][0]["delta"]["content"] == "b"
    await c.close()


# ── 12. the error body read is bounded ─────────────────────────────────────
@pytest.mark.asyncio
async def test_the_error_body_read_cannot_park_the_turn(monkeypatch):
    """MUTATION KILLED: `await resp.aread()` with no bound (its pre-R7 state).
    It inherited the node client's 1200s default, so an upstream that answers
    with headers and then nothing parked the turn for twenty minutes to
    decorate an error message."""
    monkeypatch.setattr(llm_mod, "_STREAM_ERROR_BODY_TIMEOUT", 0.2)
    c = LLMClient(upstream_url=MAIN)
    c.http_client = MagicMock()
    c.http_client.base_url = httpx.URL(MAIN)
    c.http_client.build_request = MagicMock(return_value=MagicMock())

    resp = MagicMock()
    resp.status_code = 500

    async def _hang():
        await asyncio.sleep(3600)
    resp.aread = _hang
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("boom", request=MagicMock(), response=resp))
    resp.aclose = AsyncMock()

    async def _send(req, **kw):
        return resp
    c.http_client.send = AsyncMock(side_effect=_send)

    async def go():
        async for _ in c._do_stream_chat_completion({"model": "m", "messages": []}):
            pass

    t0 = asyncio.get_event_loop().time()
    with pytest.raises(Exception):
        await asyncio.wait_for(go(), timeout=10)
    elapsed = asyncio.get_event_loop().time() - t0
    assert elapsed < 5, f"the error path took {elapsed:.1f}s — the body read is unbounded"
    await c.close()


# ── 13. the diagnostics can never break a stream ───────────────────────────
@pytest.mark.asyncio
async def test_the_counters_survive_a_client_that_never_ran_init():
    """MUTATION KILLED: reach `self._inflight_by_url` / `self._node_slot_caps`
    / `self._main_node_lock` directly instead of through the total helpers.
    Several suites build a client with `LLMClient.__new__(LLMClient)` and set
    two attributes (tests/test_stream_idle_timeout.py), and the live
    `_BackgroundOnlyLLM` shim delegates by attribute — unguarded instance
    state reached from this path turns a missing attribute into a DEAD USER
    TURN."""
    c = LLMClient.__new__(LLMClient)
    c.coding_clients = None
    c.http_client = MagicMock()
    c.http_client.build_request = MagicMock(return_value=MagicMock())

    async def _send(req, **kw):
        return _sse("bare")
    c.http_client.send = AsyncMock(side_effect=_send)

    out = []
    async for ch in c._do_stream_chat_completion({"model": "m", "messages": []}):
        out.append(ch.decode())
    assert '"content": "bare"' in "".join(out)
    # the helpers must be total on this object AND still FUNCTION — a
    # try/except that swallows the AttributeError keeps the turn alive but
    # leaves the attribution permanently reading 0, which is a guard that
    # cannot fire rather than a guard that is off.
    assert c._known_slots("anything") is None
    assert c._own_inflight("anything") == 0
    c._inflight_inc("http://x.invalid")
    assert c._own_inflight("http://x.invalid") == 1, (
        "the in-flight counter is inert on a client built with __new__ — "
        "the stall attribution would silently never fire there")
    c._inflight_dec("http://x.invalid")
    assert c._own_inflight("http://x.invalid") == 0


# ── 14. the queue budget is SHARED, not re-spent per node ──────────────────
@pytest.mark.asyncio
async def test_the_queue_budget_is_divided_across_the_nodes_still_to_try():
    """MUTATION KILLED: `untried = len(_coding_pool) - len(_tried)` (without
    the `+ 1`). `_untried` is computed AFTER the node is appended to
    `_tried`, so the caller must add one back — otherwise node 1 of a 2-node
    pool is handed the ENTIRE deadline and the sharing that the whole
    `_queue_deadline` design exists for does not happen. `_MIN_ACQUIRE` hides
    the symptom (the last node still gets 0.05s), which is exactly why this
    test resolves the budget rather than watching for a refusal.

    ⚠ It resolves the CALLABLE production actually handed the gate."""
    import os
    c = _client([{"url": N1, "model": "coder-1"}, {"url": N2, "model": "coder-2"}])
    budgets = []

    async def send(req, **kw):
        return _sse()
    _wire(c.coding_clients[0]["client"], behaviour=send)
    _wire(c.coding_clients[1]["client"], behaviour=send)

    real = type(c)._node_slot

    @contextlib.asynccontextmanager
    async def spy(node, wait_timeout=None):
        budgets.append(wait_timeout() if callable(wait_timeout) else wait_timeout)
        async with real(c, node, wait_timeout=wait_timeout):
            yield
    c._node_slot = spy

    os.environ["GHOST_NODE_SLOT_WAIT_S"] = "10"
    try:
        await _drain(c)
    finally:
        os.environ.pop("GHOST_NODE_SLOT_WAIT_S", None)
    assert budgets, "the gate was never asked for a budget"
    assert 4.0 < budgets[0] < 5.5, (
        f"the FIRST of two nodes was offered {budgets[0]:.2f}s of a 10s pool "
        f"budget — it should get about half, or the last node's share is only "
        f"rescued by the _MIN_ACQUIRE floor")
    await c.close()


# ── 15. a 4xx from a node is a CALLER fault, not the node's ────────────────
@pytest.mark.asyncio
async def test_a_4xx_from_a_coding_node_is_not_charged_to_the_breaker():
    """MUTATION KILLED: `record_failure` without the `_is_node_fault(e)`
    guard. A 4xx repeats identically on any node (bad/oversized payload,
    unknown model), so counting it as illness ejects a perfectly HEALTHY node
    for 60s over a deterministic caller bug — the exact rule
    `_is_node_fault` exists to enforce, applied at every non-streaming site
    and at none of the streaming ones before R7."""
    c = _client([{"url": N1, "model": "coder-1"}, {"url": N2, "model": "coder-2"}])

    def _four_oh_four():
        r = MagicMock()
        r.status_code = 404
        r.aread = AsyncMock(return_value=b'{"error":"unknown model"}')
        r.aclose = AsyncMock()
        r.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
            "404", request=MagicMock(), response=r))
        return r

    async def four_oh_four(req, **kw):
        return _four_oh_four()

    async def alive(req, **kw):
        return _sse()
    _wire(c.coding_clients[0]["client"], behaviour=four_oh_four)
    _wire(c.coding_clients[1]["client"], behaviour=alive)

    for _ in range(4):
        await _drain(c)
    st = c.circuit_breaker._states.get(N1, {})
    assert st.get("failures", 0) == 0 and st.get("state", "closed") == "closed", (
        f"a 404 from a coding node tripped the breaker on a healthy box: {st}")
    await c.close()


# ── 16. the main lock counts toward our own in-flight total ────────────────
@pytest.mark.asyncio
async def test_a_held_main_lock_counts_as_one_of_our_own_requests():
    """MUTATION KILLED: drop the `_main_node_lock.locked()` term from
    `_own_inflight`. The non-streaming main path and `get_embeddings` are
    serialised by that lock and are counted NOWHERE else, so without it a
    stream that is queued behind a 35B `chat_completion` reads as the sole
    caller and the upstream takes the blame for our own POST."""
    c = LLMClient(upstream_url=MAIN)
    assert c._own_inflight(MAIN) == 0
    async with c._main_node_lock:
        assert c._own_inflight(MAIN) == 1, (
            "a held main lock is invisible to the attribution")
        c._inflight_inc(MAIN)
        assert c._own_inflight(MAIN) == 2
        c._inflight_dec(MAIN)
    assert c._own_inflight(MAIN) == 0
    # a DIFFERENT url must never inherit main's lock
    c2 = LLMClient(upstream_url=MAIN)
    async with c2._main_node_lock:
        assert c2._own_inflight(N1) == 0
    await c.close()
    await c2.close()


# ── 17. output already shipped => never retry ──────────────────────────────
@pytest.mark.asyncio
async def test_a_break_after_output_started_is_surfaced_not_retried():
    """MUTATION KILLED: drop the `if yielded_any:` guard from the transport
    handler. Retrying after bytes have reached the client replays the ENTIRE
    completion after the partial one — duplicated/garbled text in the UI.
    The R7 restructuring moved this guard into a longer loop, so it gets its
    own regression."""
    c = _client([{"url": N1, "model": "coder-1"}, {"url": N2, "model": "coder-2"}])
    hits = []

    async def half(req, **kw):
        hits.append("coder-1")

        async def _gen():
            yield 'data: {"choices": [{"delta": {"content": "partial"}}]}'
            raise httpx.RemoteProtocolError("peer went away")
        r = MagicMock()
        r.status_code = 200
        r.raise_for_status = MagicMock()
        r.aiter_lines = _gen
        r.aclose = AsyncMock()
        return r

    async def alive(req, **kw):
        hits.append("coder-2")
        return _sse()
    _wire(c.coding_clients[0]["client"], behaviour=half)
    _wire(c.coding_clients[1]["client"], behaviour=alive)

    body = await _drain(c)
    assert hits == ["coder-1"], (
        f"the stream was replayed on another node after output had already "
        f"reached the client (contacted: {hits})")
    assert "partial" in body
    assert "broke mid-response" in body
    assert body.rstrip().endswith("data: [DONE]")
    await c.close()


# ── 18. every node saturated still ends in a terminal frame ────────────────
@pytest.mark.asyncio
async def test_every_node_saturated_still_reaches_main_and_terminates():
    """PINS AN INVARIANT, not a line: `_max_attempts = len(pool) + 2` and node
    attempts form a PREFIX (once `_pick_coding_node` returns None,
    `_pool_done` latches), so the LAST attempt is always the main model and
    the `except NodeSaturated: continue` can never fall out of the loop into
    the `Max retries exceeded` raise — which emits no `[DONE]` and would hang
    the client. If someone later changes the attempt budget, this is what
    catches it."""
    import os
    c = _client([{"url": N1, "model": "coder-1"}, {"url": N2, "model": "coder-2"}])

    async def unreachable(req, **kw):        # must never be called
        raise AssertionError("a saturated node was contacted")
    _wire(c.coding_clients[0]["client"], slots=1, behaviour=unreachable)
    _wire(c.coding_clients[1]["client"], slots=1, behaviour=unreachable)
    for url in (N1, N2):
        c._node_slots[url] = asyncio.Semaphore(1)
        c._node_slot_caps[url] = 1
        c._node_slot_built_cap[url] = 1
        await c._node_slots[url].acquire()    # every permit taken, forever

    c.http_client.build_request = MagicMock(return_value=MagicMock())

    async def main_send(req, **kw):
        return _sse("main")
    c.http_client.send = AsyncMock(side_effect=main_send)

    os.environ["GHOST_NODE_SLOT_WAIT_S"] = "0.05"
    try:
        body = await _drain(c)
    finally:
        os.environ.pop("GHOST_NODE_SLOT_WAIT_S", None)
    assert '"content": "main"' in body
    assert body.rstrip().endswith("data: [DONE]")
    assert all(s.get("failures", 0) == 0
               for s in c.circuit_breaker._states.values()), (
        "saturation reached the breaker")
    await c.close()


# ── 19. the recorder cannot kill a turn ────────────────────────────────────
@pytest.mark.asyncio
async def test_a_bytes_chunk_does_not_kill_the_turn_when_recording_is_on(
        monkeypatch):
    """MUTATION KILLED: restore `_stream_rec_accumulate`'s type guard to
    `if not line or not line.startswith("data:")`. That guard sits OUTSIDE
    the function's `try`, so `bytes.startswith("data:")` raises `TypeError`
    — which the streaming loop's `except Exception` turns into a re-raise
    with no `[DONE]`. The docstring already promised "Tolerant by contract:
    any unparseable line is ignored", and the existing tolerance test only
    ever feeds it `str`. Third instance of the same "a chunk may be bytes"
    inconsistency inside one function, and the only one that KILLS the turn
    rather than garbling a frame."""
    monkeypatch.setattr(llm_mod.LLMClient, "_maybe_record_call",
                        staticmethod(lambda *a, **k: None))
    import ghost_agent.core.llm_recording as _recmod
    monkeypatch.setattr(_recmod, "recording_enabled", lambda: True)

    c = _client([{"url": N1, "model": "coder"}])

    async def send(req, **kw):
        return _sse(lines=[b'data: {"choices": [{"delta": {"content": "b"}}]}',
                           b'data: [DONE]'])
    _wire(c.coding_clients[0]["client"], behaviour=send)

    out = []
    async for ch in c.stream_chat_completion({"messages": [], "model": "m"},
                                             use_coding=True):
        out.append(ch)
    assert b"[DONE]" in b"".join(out)
    assert b'"content": "b"' in b"".join(out)
    await c.close()


def test_the_recorder_accumulator_is_total_on_any_line_type():
    """Direct unit form of the same mutation — the guard must not raise for
    ANY input type, which is what "tolerant by contract" has to mean."""
    acc = {"content": [], "reasoning": [], "tool_calls": {}, "finish": None}
    LLMClient._stream_rec_accumulate(
        b'data: {"choices":[{"delta":{"content":"x"}}]}', acc)
    assert acc["content"] == ["x"], (
        "a bytes SSE line was dropped by the accumulator")
    for junk in (None, 123, b"", b": keepalive", object()):
        LLMClient._stream_rec_accumulate(junk, acc)   # must not raise
    assert acc["content"] == ["x"]
