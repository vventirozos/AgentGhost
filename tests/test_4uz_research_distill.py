"""§4UZ — how a research call spends the worker's time.

THE DEFECT (probe of 2026-11-19, a deep_research of 8 pages). The first 3 pages were fetched and sent to
the worker node TOGETHER; each came back after ~25 s. The node does not scale with parallel requests
(bench, same prompt: 1 request 17.5 s; 3 at once all done at ~48 s; 5 at once done at 61-81 s), so the
fan-out bought nothing — and it taught the planner an interleaved "decode" of a few tok/s. The next 5
plans read "distill declined (46.0s available but the smallest useful distill (6,000 chars + 128 tok)
needs 75.1s at 192/2 tok/s)", all five pages went into the report as raw text, and the tool was back
after ~54 s of a 190 s phase.

NOW. Pages are still FETCHED in parallel; their distills go through ONE queue per client
(`core/node_throughput.py`: `DistillQueue` / `DistillWave`), `GHOST_DISTILL_WIDTH` at a time (default
1), best search rank first among the pages whose text is ready, each sized from the time the phase has
left shared among the pages still waiting.

HOW THESE TESTS RUN. On a VIRTUAL clock: an event loop whose time jumps to the next timer, with
`time.monotonic` pointed at it — a 190 s phase takes milliseconds and every figure is exact. The worker
is a fake SERIAL node: N requests at once each take N x their solo time, and its `timings` are what the
real node reported (see `SerialNode`). The planner is the real `NodeThroughput`.
"""
import asyncio
import heapq
import time

import pytest

from ghost_agent.core import node_throughput as NT
from ghost_agent.core.node_throughput import NodeThroughput
from ghost_agent.tools import search as S
from ghost_agent.utils.helpers import FetchError

URL = "http://worker.test"
UTIL = NT.BUDGET_UTILISATION


# ── the virtual clock ───────────────────────────────────────────────────────

class VirtualLoop(asyncio.SelectorEventLoop):
    """Nothing ready → the clock jumps to the next timer instead of sleeping."""

    def __init__(self):
        super().__init__()
        self.vnow = 1000.0

    def time(self):
        return self.vnow

    def _run_once(self):
        while self._scheduled and self._scheduled[0]._cancelled:
            h = heapq.heappop(self._scheduled)
            h._scheduled = False
            self._timer_cancelled_count -= 1
        if not self._ready and not self._scheduled:
            # no test here does real I/O: nothing ready and no timer is a waiter nobody will wake
            raise RuntimeError("deadlock: every task is waiting and no timer is set")
        if not self._ready and self._scheduled[0]._when > self.vnow:
            self.vnow = self._scheduled[0]._when
        super()._run_once()


@pytest.fixture
def vloop(monkeypatch):
    loop = VirtualLoop()
    monkeypatch.setattr(time, "monotonic", lambda: loop.vnow)

    async def _inline(fn, *a, **k):         # a worker thread would let the clock run ahead of it
        return fn(*a, **k)
    monkeypatch.setattr(asyncio, "to_thread", _inline)
    monkeypatch.delenv("GHOST_DISTILL_WIDTH", raising=False)
    yield loop
    try:
        left = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in left:
            t.cancel()
        if left:
            loop.run_until_complete(asyncio.gather(*left, return_exceptions=True))
    finally:
        loop.close()


# ── the fake worker ─────────────────────────────────────────────────────────

class SerialNode:
    """A worker client like the live node: it does NOT scale with parallel requests.

    Time: a request needs `prompt/prefill + max_tokens/decode` seconds of the node; N requests at once
    share it, so each takes N x as long (the bench: 17.5 s alone, ~48 s each for three).
    `timings`: alone, the true rates. With company the node reported an interleaved decode far below
    1/N (bench: 4-40 tok/s at five; the probe's planner line "192/2 tok/s" = 2 tok/s per request at
    three) — `loaded_decode` is that figure.
    `plan_distill` / the `observe` after each answer are what `LLMClient` does: plan at
    max(declared, busy + 1), learn at the average concurrency the request ran at.
    """

    def __init__(self, prefill=372.0, decode=48.0, loaded_decode=2.0, slowdown=1.0, deaf=False):
        self.nt = NodeThroughput()
        self.deaf = deaf                  # ignores the caller's time limit (a hung connection)
        self.prefill, self.decode, self.loaded_decode = prefill, decode, loaded_decode
        self.slowdown = slowdown          # > 1: the node is this much slower than it reports
        self.active = 0
        self.peak = 0
        self.plans = []                   # dict(t, budget, declared, plan)
        self.calls = []                   # dict(url, t0, t1, timeout, ok)
        self.notes = []                   # the burned seconds of each timeout the wave reported

    def plan_distill(self, budget_s, **kw):
        declared = int(kw.pop("concurrency", 1) or 1)
        plan = self.nt.plan(budget_s, URL, concurrency=max(1, declared, self.active + 1), **kw)
        self.plans.append({"t": time.monotonic(), "budget": budget_s, "declared": declared, "plan": plan})
        return plan

    def note_distill_density(self, *a, **k):
        return False

    def note_distill_timeout(self, prompt_tokens, max_tokens, burned_s, concurrency=1):
        self.notes.append(round(burned_s, 1))               # as `LLMClient.note_distill_timeout`
        return self.nt.observe_timeout(URL, prompt_tokens, max_tokens, burned_s, concurrency=concurrency)

    async def chat_completion(self, payload, **kw):
        text = payload["messages"][0]["content"]
        mark = text.split("PAGE-MARK ", 1)[1].split()[0] if "PAGE-MARK " in text else "?"
        p_n, d_n = len(text) / 3.2, float(payload["max_tokens"])
        remaining = (p_n / self.prefill + d_n / self.decode) * self.slowdown
        t0 = time.monotonic()
        rec = {"url": mark, "t0": t0, "t1": None, "timeout": kw.get("timeout"), "ok": False}
        self.calls.append(rec)
        self.active += 1
        self.peak = max(self.peak, self.active)
        conc_s = 0.0
        try:
            while remaining > 1e-9:
                n = self.active
                step = min(remaining * n, 0.25)
                left = (kw.get("timeout") or 1e9) - (time.monotonic() - t0)
                if left <= 1e-6 and not self.deaf:      # (a step below the clock's resolution never passes)
                    raise asyncio.TimeoutError("the worker did not answer in time")
                step = min(step, left) if not self.deaf else step
                await asyncio.sleep(step)
                remaining -= step / n
                conc_s += step * n
        finally:
            self.active -= 1
            rec["t1"] = time.monotonic()
        wall = max(1e-6, rec["t1"] - t0)
        avg = max(1, int(round(conc_s / wall)))
        decode = self.decode if avg == 1 else self.loaded_decode
        data = {"choices": [{"message": {"content": f"- a fact from {mark}"}, "finish_reason": "stop"}],
                "timings": {"prompt_n": p_n, "prompt_ms": 1000.0 * p_n / (self.prefill / avg),
                            "predicted_n": d_n, "predicted_ms": 1000.0 * d_n / decode}}
        self.nt.observe(URL, data, concurrency=avg)
        rec["ok"] = True
        return data


def _page(i):
    return (f"PAGE-MARK p{i} The Kosmas lighthouse log, entry {i}: the lamp was lit by Elena and the "
            f"keeper Markos wrote the hour down. ") * 400          # ~50k chars


def _urls(n):
    return [f"https://lighthouse{i}.example/log" for i in range(n)]


def _fetcher(delays=None, fail=None, hang=None):
    """`helper_fetch_url_content` stand-in: page i after `delays[i]` s (default 1 s)."""
    async def _fetch(url, **kw):
        i = int(url.split("lighthouse", 1)[1].split(".", 1)[0])
        if hang and i in hang:
            await asyncio.sleep(10_000)
        await asyncio.sleep((delays or {}).get(i, 1.0))
        if fail and i in fail:
            return FetchError(f"Error: Received status 403 from {url}")
        return _page(i)
    return _fetch


async def _research(node, n=8, fetch=None, **kw):
    kw.setdefault("max_context", 240_000)
    return await S._deep_research_core(urls=_urls(n), llm_client=node, tor_proxy="socks5h://127.0.0.1:9050",
                                       question="When was the Kosmas lighthouse lamp lit?", **kw)


def _run(vloop, monkeypatch, node, **kw):
    monkeypatch.setattr(S, "helper_fetch_url_content", kw.pop("fetch", None) or _fetcher())
    t0 = vloop.vnow
    out = str(vloop.run_until_complete(_research(node, **kw)))
    return out, vloop.vnow - t0


def _blocks(report):
    """url index -> "distilled" | "raw" | "failed", from the report's own blocks."""
    out = {}
    for part in report.split("### SOURCE: ")[1:]:
        i = int(part.split("lighthouse", 1)[1].split(".", 1)[0])
        if S.source_block_failed("### SOURCE: " + part):
            out[i] = "failed"
        else:
            out[i] = "distilled" if "[EDGE EXTRACTED FACTS]" in part else "raw"
    return out


# ── (a) the live case ───────────────────────────────────────────────────────

def test_the_live_case_eight_pages_are_all_distilled(vloop, monkeypatch):
    """Before: 3 distilled at once, the interleaved reading learned, the other 5 declined → raw."""
    node = SerialNode()
    out, took = _run(vloop, monkeypatch, node)
    got = _blocks(out)
    assert got == {i: "distilled" for i in range(8)}, got
    assert "could not be distilled" not in out
    assert all(p["plan"].feasible for p in node.plans), [p["plan"].reason for p in node.plans]
    assert took <= S._RESEARCH_PHASE_TIMEOUT_S


def test_a_short_phase_distills_the_best_ranked_and_keeps_the_rest_raw(vloop, monkeypatch):
    """60 s for 8 pages: an equal share (~7 s) is below the smallest useful distill (~11 s here), so
    the best-ranked pages get that smallest size while it fits and the rest stay raw — never
    "all declined", and never a source lost."""
    monkeypatch.setattr(S, "_RESEARCH_PHASE_TIMEOUT_S", 60.0)
    node = SerialNode()
    out, took = _run(vloop, monkeypatch, node)
    got = _blocks(out)
    assert sorted(got) == list(range(8)) and "failed" not in got.values(), got
    k = sum(1 for v in got.values() if v == "distilled")
    assert k >= 5, got                                           # the old code: 3
    assert [got[i] for i in range(8)] == ["distilled"] * k + ["raw"] * (8 - k), got
    assert 8 - k >= 1 and "could not be distilled" in out
    assert took <= 60.0


# ── (b) width 1: one at a time; (i) the readings it leaves ───────────────────

def test_with_width_one_no_two_distills_overlap(vloop, monkeypatch):
    node = SerialNode()
    _run(vloop, monkeypatch, node)
    assert node.peak == 1 and len(node.calls) == 8
    spans = sorted((c["t0"], c["t1"]) for c in node.calls)
    assert all(a[1] <= b[0] + 1e-9 for a, b in zip(spans, spans[1:])), spans
    assert all(p["declared"] == 1 for p in node.plans)


def test_a_serial_wave_leaves_solo_rates(vloop, monkeypatch):
    """What `observe` holds afterwards is the node alone (372 / 48 tok/s) — both the mean and the
    figure plans are made from. Before: the 3-way readings (decode 2 tok/s per request)."""
    node = SerialNode()
    _run(vloop, monkeypatch, node)
    prefill, decode, samples = node.nt.rates(URL)
    assert samples == 8
    assert prefill == pytest.approx(372.0, rel=0.02) and decode == pytest.approx(48.0, rel=0.02)
    p_prefill, p_decode, _ = node.nt.plan_rates(URL)
    assert p_prefill == pytest.approx(372.0, rel=0.02) and p_decode == pytest.approx(48.0, rel=0.02)


# ── (c) rank order ──────────────────────────────────────────────────────────

def test_ready_pages_are_distilled_best_rank_first(vloop, monkeypatch):
    """Page 0 takes 20 s to load, the others 1-3 s. It does not hold the queue (page 1 goes first),
    and once its text is there it goes before every lower-ranked page still waiting."""
    node = SerialNode()
    out, _ = _run(vloop, monkeypatch, node, fetch=_fetcher(delays={0: 20.0}))
    order = [c["url"] for c in node.calls]
    assert order[0] == "p1", order
    ready0 = 20.0
    t_start = node.calls[0]["t0"] - 1.0           # the fetches began 1 s before the first distill
    after = [c["url"] for c in node.calls if c["t0"] - t_start >= ready0]
    assert after and after[0] == "p0", (order, after)
    rest = [u for u in order if u != "p0"]
    assert rest == [f"p{i}" for i in range(1, 8)], order
    assert _blocks(out) == {i: "distilled" for i in range(8)}


def test_the_queue_orders_waiters_by_rank_not_arrival(vloop):
    async def main():
        q = NT.DistillQueue()
        order = []

        async def page(rank):
            async with q.slot(rank) as mine:
                assert mine
                order.append(rank)
                await asyncio.sleep(1.0)
        first = asyncio.ensure_future(page(5))          # takes the free turn
        await asyncio.sleep(0)
        rest = [asyncio.ensure_future(page(r)) for r in (7, 3, 6, 1)]
        await asyncio.gather(first, *rest)
        return order, q.running
    order, running = vloop.run_until_complete(main())
    assert order == [5, 1, 3, 6, 7] and running == 0


# ── (d) the budget of one page ──────────────────────────────────────────────

def test_a_page_budget_is_the_time_left_shared_among_those_waiting(vloop, monkeypatch):
    """budget = (phase left - the report's reserve) / pages still waiting, never above the ceiling."""
    node = SerialNode()
    t0 = vloop.vnow
    _run(vloop, monkeypatch, node)
    deadline = t0 + S._RESEARCH_PHASE_TIMEOUT_S
    assert len(node.plans) == 8
    for i, p in enumerate(node.plans):
        share = (deadline - S.RESEARCH_ASSEMBLY_S - p["t"]) / (8 - i)
        assert share < S._WEB_SUMMARY_TIMEOUT_S
        # the plan is sized for BUDGET_UTILISATION of the share (headroom for a node slower than its
        # figures); the call's own hard limit is the whole share (one at a time: no queue allowance)
        assert p["budget"] == pytest.approx(share * UTIL, abs=0.05), (i, p["budget"], share)
        assert node.calls[i]["timeout"] == pytest.approx(share, abs=0.05)


def test_a_page_budget_never_exceeds_the_ceiling(vloop, monkeypatch):
    """Two pages and 190 s: the share (93 s) is above the per-page ceiling (65 s), which holds."""
    node = SerialNode()
    out, _ = _run(vloop, monkeypatch, node, n=2)
    assert [p["budget"] for p in node.plans] == pytest.approx([S._WEB_SUMMARY_TIMEOUT_S * UTIL] * 2)
    assert [c["timeout"] for c in node.calls] == pytest.approx([S._WEB_SUMMARY_TIMEOUT_S] * 2)
    assert _blocks(out) == {0: "distilled", 1: "distilled"}


def test_the_budget_shrinks_as_the_phase_runs_out(vloop, monkeypatch):
    """The same 8 pages with 190 s, 160 s and 130 s (the node's rates known): each page is offered
    less, and less of it is sent — smaller, never declined."""
    firsts, sizes = [], []
    for phase in (190.0, 160.0, 130.0):
        monkeypatch.setattr(S, "_RESEARCH_PHASE_TIMEOUT_S", phase)
        node = SerialNode()
        node.nt.observe(URL, _timings(372.0, 48.0), concurrency=1)
        out, took = _run(vloop, monkeypatch, node)
        assert _blocks(out) == {i: "distilled" for i in range(8)}, (phase, _blocks(out))
        assert took <= phase
        firsts.append(node.plans[0]["budget"])
        sizes.append(node.plans[0]["plan"].char_limit)
    assert firsts[0] > firsts[1] > firsts[2] > 0, firsts
    assert sizes[0] > sizes[1] > sizes[2] >= NT.MIN_CHARS, sizes


def test_a_share_below_the_planners_minimum_is_floored_at_it(vloop):
    """4 pages, 36 s: a share of 9 s is refused by the planner (it needs ~11.3 s at the priors'
    rates here), so each page in turn gets exactly that minimum — two fit, the others are declined."""
    async def main():
        node = SerialNode()
        node.nt._rates[URL] = [372.0, 48.0, 1, 372.0, 48.0, time.monotonic(), time.monotonic()]
        wave = NT.DistillWave(node, ["a", "b", "c", "d"], deadline=time.monotonic() + 36.0,
                              ceiling_s=65.0, reserve_s=0.0)
        least = node.nt.plan(1.0, URL).min_budget_s
        assert 7.5 < least < 15.0
        got = []
        for key in "abcd":
            async with wave.turn(key) as turn:
                got.append((turn.plan.feasible, turn.budget_s))
                if turn.plan.feasible:
                    await asyncio.sleep(turn.budget_s)
        return least, got
    least, got = vloop.run_until_complete(main())
    assert [g[0] for g in got] == [True, True, False, False], got
    floor = (least + 0.05) / UTIL                  # the plan gets `least`; the call's limit keeps the headroom
    assert got[0][1] == pytest.approx(floor) and got[1][1] == pytest.approx(floor)


# ── (e) the bounds ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("deadline_s", [None, 70.0])
def test_the_phase_and_the_callers_deadline_are_honoured(vloop, monkeypatch, deadline_s):
    """A node 6x slower than it reports: every distill runs into its own time limit. The tool is
    still back inside the phase (190 s) — or inside the caller's `deadline_s` (fact_check passes
    one) — no distill is allowed past the bound less the report's reserve, and no source is lost."""
    node = SerialNode(slowdown=6.0)
    t0 = vloop.vnow
    kw = {} if deadline_s is None else {"deadline_s": deadline_s}
    out, took = _run(vloop, monkeypatch, node, **kw)
    bound = S._RESEARCH_PHASE_TIMEOUT_S if deadline_s is None else deadline_s - S.RESEARCH_ASSEMBLY_S
    assert took <= bound + 1e-6, took
    for c in node.calls:
        assert c["t0"] + c["timeout"] <= t0 + bound - S.RESEARCH_ASSEMBLY_S + 0.06, (c, t0 + bound)
        assert c["t1"] <= t0 + bound + 1e-6
    got = _blocks(out)
    assert sorted(got) == list(range(8)) and "failed" not in got.values(), got
    assert node.calls, "nothing was sent to the worker at all"


def test_a_worker_that_ignores_its_time_limit_cannot_hold_the_tool(vloop, monkeypatch):
    """The call to the worker has a hard limit of its own (2 s above its budget): a request that
    never comes back is cut there, the page stays as raw text, and the next page gets its turn."""
    node = SerialNode(slowdown=50.0, deaf=True)
    out, took = _run(vloop, monkeypatch, node, n=3)
    assert took <= S._RESEARCH_PHASE_TIMEOUT_S, took
    assert _blocks(out) == {0: "raw", 1: "raw", 2: "raw"}
    assert 2 <= len(node.calls) <= 3
    for c in node.calls:
        assert c["t1"] - c["t0"] <= c["timeout"] + 2.0 + 1e-6, c
    assert node.active == 0


def test_a_call_waiting_behind_another_calls_distill_still_returns_in_time(vloop, monkeypatch):
    """Two research calls on one client. The first one's distill holds the worker for a minute; the
    second has a 30 s bound (fact_check's). Its pages cannot have the worker in time: it returns
    inside its bound with every page as raw text — it does not wait for the queue."""
    monkeypatch.setattr(S, "helper_fetch_url_content", _fetcher())
    node = SerialNode(slowdown=50.0)                 # every distill runs to its 65 s limit

    async def main():
        first = asyncio.ensure_future(_research(node, n=2))
        await asyncio.sleep(3.0)
        t0 = time.monotonic()
        second = str(await S._deep_research_core(
            urls=[f"https://lighthouse{i}.example/log" for i in (5, 6)], llm_client=node,
            tor_proxy="socks5h://127.0.0.1:9050", deadline_s=30.0, max_context=240_000))
        took = time.monotonic() - t0
        return second, took, str(await first)
    second, took, first = vloop.run_until_complete(main())
    assert took <= 30.0, took
    assert _blocks(second) == {5: "raw", 6: "raw"}
    assert sorted(_blocks(first)) == [0, 1] and node.peak == 1


def test_a_deadline_too_short_for_everything_distills_the_best_ranked(vloop, monkeypatch):
    out, took = _run(vloop, monkeypatch, node := SerialNode(), deadline_s=45.0)
    got = _blocks(out)
    k = sum(1 for v in got.values() if v == "distilled")
    assert took <= 45.0 and sorted(got) == list(range(8))
    assert k >= 3 and [got[i] for i in range(8)] == ["distilled"] * k + ["raw"] * (8 - k), got
    assert node.peak == 1


# ── (f) cancel ──────────────────────────────────────────────────────────────

def test_cancel_mid_wave_stops_everything_and_frees_the_queue(vloop, monkeypatch):
    """The owner stops the turn while the 2nd page is being distilled: the tool's task ends
    cancelled, nothing of it keeps running, no further distill starts, and the queue is free — the
    next research on the same client distills normally."""
    monkeypatch.setattr(S, "helper_fetch_url_content", _fetcher())
    node = SerialNode()

    async def main():
        task = asyncio.ensure_future(_research(node))
        for _ in range(400):
            if len(node.calls) >= 2:
                break
            await asyncio.sleep(0.5)
        assert len(node.calls) == 2, "the second page never got its turn"
        await asyncio.sleep(1.0)
        assert node.active == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(5):
            await asyncio.sleep(0)
        started = len(node.calls)
        others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        await asyncio.sleep(300.0)                      # nothing wakes up later either
        q = NT.distill_queue(node)
        state = (started, len(node.calls), node.active, len(others), q.running)
        return state, str(await _research(node, n=3))
    (started, later, active, others, running), again = vloop.run_until_complete(main())
    assert started == 2 and later == 2, (started, later)
    assert active == 0 and others == 0 and running == 0
    assert _blocks(again) == {i: "distilled" for i in range(3)}


def test_a_waiter_cancelled_in_the_queue_never_gets_the_turn(vloop):
    async def main():
        q = NT.DistillQueue()
        ran = []

        async def page(rank, hold):
            async with q.slot(rank):
                ran.append(rank)
                await asyncio.sleep(hold)
        a = asyncio.ensure_future(page(0, 5.0))
        await asyncio.sleep(0)
        b = asyncio.ensure_future(page(1, 5.0))
        c = asyncio.ensure_future(page(2, 5.0))
        await asyncio.sleep(1.0)
        b.cancel()
        await asyncio.gather(a, b, c, return_exceptions=True)
        return ran, q.running, b.cancelled()
    assert vloop.run_until_complete(main()) == ([0, 2], 0, True)


def test_a_turn_granted_and_cancelled_in_one_step_goes_back(vloop):
    async def main():
        q = NT.DistillQueue()
        assert await q.acquire(0)
        waiter = asyncio.ensure_future(q.acquire(1))
        await asyncio.sleep(0)
        q.release()                 # the turn passes to the waiter ...
        waiter.cancel()             # ... which is cancelled before it runs again
        await asyncio.gather(waiter, return_exceptions=True)
        return q.running, await q.acquire(2, timeout=1.0)
    assert vloop.run_until_complete(main()) == (0, True)


def test_a_queue_wait_ends_at_its_timeout(vloop):
    async def main():
        q = NT.DistillQueue()
        assert await q.acquire(0)
        t0 = time.monotonic()
        got = await q.acquire(1, timeout=7.0)
        waited = time.monotonic() - t0
        q.release()
        return got, waited, q.running
    assert vloop.run_until_complete(main()) == (False, 7.0, 0)


# ── (g) a page that fails or hangs ──────────────────────────────────────────

def test_a_failed_and_a_hung_fetch_do_not_stall_the_queue(vloop, monkeypatch):
    """Page 0 is refused (403), page 1 never answers (two 22 s attempts). The other six are
    distilled meanwhile — the first distill starts as soon as the first text is there — and the two
    stop counting as waiting, so the pages after them are offered more."""
    node = SerialNode()
    t0 = vloop.vnow
    out, took = _run(vloop, monkeypatch, node, fetch=_fetcher(fail={0}, hang={1}))
    got = _blocks(out)
    assert got == {0: "failed", 1: "failed", **{i: "distilled" for i in range(2, 8)}}, got
    assert node.calls[0]["t0"] - t0 <= 2.0, node.calls[0]["t0"] - t0
    assert [c["url"] for c in node.calls] == [f"p{i}" for i in range(2, 8)]
    deadline = t0 + S._RESEARCH_PHASE_TIMEOUT_S - S.RESEARCH_ASSEMBLY_S
    hung_until = t0 + 2 * S._FETCH_ATTEMPT_TIMEOUT
    for i, p in enumerate(node.plans):
        waiting = (6 - i) + (1 if p["t"] < hung_until else 0)      # the hung page counts until it fails
        want = min(S._WEB_SUMMARY_TIMEOUT_S, (deadline - p["t"]) / waiting)
        assert p["budget"] == pytest.approx(want * UTIL, abs=0.05), (i, p["budget"], want, waiting)
    assert took <= S._RESEARCH_PHASE_TIMEOUT_S


def test_a_planner_that_raises_keeps_every_page(vloop, monkeypatch):
    class Broken(SerialNode):
        def plan_distill(self, budget_s, **kw):
            raise RuntimeError("no planner")
    node = Broken()
    out, _ = _run(vloop, monkeypatch, node, n=3)
    assert _blocks(out) == {0: "raw", 1: "raw", 2: "raw"} and node.calls == []
    assert NT._QUEUES[node].running == 0

    async def one_turn():
        wave = NT.DistillWave(node, ["a"], deadline=time.monotonic() + 100.0, ceiling_s=65.0)
        async with wave.turn("a") as turn:
            return turn
    turn = vloop.run_until_complete(one_turn())       # the wave hands the error over: the caller says why
    assert turn.plan is None and isinstance(turn.error, RuntimeError)


def test_a_recheck_that_cannot_have_the_worker_in_time_sends_nothing(vloop):
    """The "nothing relevant" recheck takes the same queue. With the worker held by another call past
    the recheck's deadline it asks nothing (it used to start at once, beside the running distill)."""
    node = SerialNode()
    urls = _urls(2)
    blocks = [f"### SOURCE: {u}\n[EDGE EXTRACTED FACTS]:\nNONE\n" for u in urls]
    texts = {u: (_page(i)[:6000], 128, 5.0) for i, u in enumerate(urls)}

    async def main():
        q = NT.distill_queue(node)
        assert await q.acquire(0)
        t0 = time.monotonic()
        r = await S.recheck_all_none(blocks, urls, texts, "lighthouse lamp", node, "m",
                                     deadline=t0 + 40.0, slot_wait=1.0)
        waited = time.monotonic() - t0
        q.release()
        return r, waited
    r, waited = vloop.run_until_complete(main())
    assert node.calls == [] and r.fixed == 0 and r.contents == blocks
    assert waited <= 40.0


# ── (h) a wider queue keeps the declared-concurrency planning ────────────────

def test_width_three_runs_three_at_once_and_plans_for_three(vloop, monkeypatch):
    monkeypatch.setenv("GHOST_DISTILL_WIDTH", "3")
    node = SerialNode(loaded_decode=16.0)            # a node that does give each of three 1/3
    t0 = vloop.vnow
    out, took = _run(vloop, monkeypatch, node)
    assert node.peak == 3
    assert all(p["declared"] == 3 for p in node.plans), [p["declared"] for p in node.plans]
    first = node.plans[0]
    share = (t0 + S._RESEARCH_PHASE_TIMEOUT_S - S.RESEARCH_ASSEMBLY_S - first["t"]) * 3 / 8
    assert first["budget"] == pytest.approx(
        (min(S._WEB_SUMMARY_TIMEOUT_S, share) - S._QUEUE_ALLOWANCE_S) * UTIL, abs=0.05)
    assert first["plan"].decode_tok_s == pytest.approx(NT.DEFAULT_DECODE_TOK_S / 3)
    assert _blocks(out) == {i: "distilled" for i in range(8)}
    assert took <= S._RESEARCH_PHASE_TIMEOUT_S


@pytest.mark.parametrize("raw,width", [("", 1), ("1", 1), ("3", 3), ("0", 1), ("-2", 1), ("x", 1), ("99", 8)])
def test_the_width_is_read_at_call_time(monkeypatch, raw, width):
    monkeypatch.setenv("GHOST_DISTILL_WIDTH", raw)
    assert NT.distill_width() == width


def test_widening_the_queue_takes_effect_on_the_running_queue(vloop, monkeypatch):
    async def main():
        q = NT.DistillQueue()
        peak = [0]

        async def page(rank):
            async with q.slot(rank):
                peak[0] = max(peak[0], q.running)
                await asyncio.sleep(1.0)
        await asyncio.gather(*[page(r) for r in range(4)])
        one = peak[0]
        monkeypatch.setenv("GHOST_DISTILL_WIDTH", "2")
        peak[0] = 0
        await asyncio.gather(*[page(r) for r in range(4)])
        return one, peak[0]
    assert vloop.run_until_complete(main()) == (1, 2)


# ── design point 4: a reading taken under parallel load ──────────────────────

def _timings(prefill, decode):
    return {"timings": {"prompt_n": 4900, "prompt_ms": 4900 / prefill * 1000.0,
                        "predicted_n": 200, "predicted_ms": 200 / decode * 1000.0}}


def test_a_reading_from_parallel_load_does_not_veto_the_solo_plan(monkeypatch):
    """The probe's numbers: the node alone does 372 / 48; one 3-way reading said 2 tok/s of decode
    per request. A plan for ONE distill in 46 s must still be made (it was refused: "needs 75.1s")."""
    monkeypatch.delenv("GHOST_DISTILL_WIDTH", raising=False)
    nt = NodeThroughput()
    for _ in range(3):
        assert nt.observe(URL, _timings(372.0, 48.0), concurrency=1)
    assert nt.observe(URL, _timings(110.0, 2.0), concurrency=3)
    plan = nt.plan(46.0, URL, concurrency=1)
    assert plan.feasible, plan.reason
    assert plan.decode_tok_s > 30.0 and plan.max_tokens > NT.MIN_TOKENS
    # the reading is not thrown away: the mean moved
    assert nt.rates(URL)[1] < 48.0


def test_a_node_slow_on_its_own_is_still_planned_for_at_once(monkeypatch):
    """The conservative rule stays for what it was made for: ONE slow solo reading governs the next plan."""
    monkeypatch.delenv("GHOST_DISTILL_WIDTH", raising=False)
    nt = NodeThroughput()
    for _ in range(3):
        nt.observe(URL, _timings(372.0, 48.0), concurrency=1)
    nt.observe(URL, _timings(372.0, 6.0), concurrency=1)
    assert nt.plan_rates(URL)[1] == pytest.approx(6.0, rel=0.01)
    assert not nt.plan(20.0, URL, concurrency=1).feasible


def test_at_width_three_a_three_way_reading_is_the_planned_regime(monkeypatch):
    monkeypatch.setenv("GHOST_DISTILL_WIDTH", "3")
    nt = NodeThroughput()
    for _ in range(3):
        nt.observe(URL, _timings(372.0, 48.0), concurrency=1)
    nt.observe(URL, _timings(110.0, 2.0), concurrency=3)
    assert nt.plan_rates(URL)[1] == pytest.approx(6.0, rel=0.01)      # 2 tok/s x 3, as before


# ── the class: every tool that distills on the worker takes the same queue ──

class _CountingClient:
    """Real clock, tiny sleeps: how many worker distills ran at once, across tools."""

    def __init__(self):
        self.now = self.peak = 0
        self.labels = []

    def plan_distill(self, budget_s, **kw):
        return NT.DistillPlan(6_000, 128, True, "", 300.0, 30.0, 3, 1.0)

    def note_distill_density(self, *a, **k):
        return False

    async def chat_completion(self, payload, **kw):
        self.now += 1
        self.peak = max(self.peak, self.now)
        self.labels.append(kw.get("task_label"))
        try:
            await asyncio.sleep(0.03)
        finally:
            self.now -= 1
        return {"choices": [{"message": {"content": "NONE"}, "finish_reason": "stop"}]}


@pytest.mark.parametrize("width", [1, 2])
async def test_every_distilling_tool_shares_the_one_queue(monkeypatch, width):
    """deep_research, darkweb_research and the "nothing relevant" recheck, at the same time on one
    client: never more than the width at the worker. (Every answer is NONE, so both tools go on to
    their recheck as well.) A tool that sends its distills past the queue shows up as a higher peak."""
    from unittest.mock import AsyncMock, patch
    from ghost_agent.tools import darkweb_search as D
    from tests.onion_v3 import v3_label
    monkeypatch.setenv("GHOST_DISTILL_WIDTH", str(width))
    # real clock: a 40 s phase (a fetch needs 24 s of it left, a recheck 25 s), so a queue that stops
    # passing turns fails this test in under a minute instead of holding it for the full 190 s
    monkeypatch.setattr(S, "_RESEARCH_PHASE_TIMEOUT_S", 40.0)
    monkeypatch.setattr(D, "_ONION_PAGE_TIMEOUT", 3.0)
    client = _CountingClient()

    async def _fetch(url, **kw):
        await asyncio.sleep(0.01)
        return _page(1)
    monkeypatch.setattr(S, "helper_fetch_url_content", _fetch)
    html = "".join(f'<a href="http://{v3_label(c)}.onion/">Kosmas lighthouse log {c}</a>' for c in "abcd")

    async def _engine(url, proxy, timeout, **kwargs):
        return 200, html
    with patch.object(D, "_fetch_raw_html", side_effect=_engine), \
            patch.object(D, "_fetch_onion_text", new_callable=AsyncMock) as onion:
        onion.return_value = _page(2)
        deep, deep2, dark = await asyncio.wait_for(asyncio.gather(
            S._deep_research_core(urls=_urls(4), llm_client=client, tor_proxy="socks5h://127.0.0.1:9050"),
            S._deep_research_core(urls=_urls(8)[4:], llm_client=client, tor_proxy="socks5h://127.0.0.1:9050"),
            D.tool_darkweb_research("Kosmas lighthouse log", tor_proxy="socks5://127.0.0.1:9050",
                                    llm_client=client)), 60.0)
    assert "web summary" in client.labels
    assert any("recheck" in str(x) for x in client.labels), client.labels
    assert len(client.labels) >= 9 and "lighthouse" in str(deep) and "onion" in str(dark)
    assert client.peak == width, (client.peak, client.labels)


async def test_darkweb_pages_that_fail_stop_counting_as_waiting(monkeypatch):
    """darkweb_research, 4 onions, 2 of them refused (scaled: a 14 s phase, 3 s kept for the report):
    the first of the two that load is offered half of the ~11 s left, not a quarter."""
    from unittest.mock import patch
    from ghost_agent.tools import darkweb_search as D
    from tests.onion_v3 import v3_label
    monkeypatch.delenv("GHOST_DISTILL_WIDTH", raising=False)
    monkeypatch.setattr(S, "RESEARCH_BACKFILL_PER_SLOT", 0)
    monkeypatch.setattr(D, "_ONION_PAGE_TIMEOUT", 3.0)
    monkeypatch.setattr(S, "_RESEARCH_PHASE_TIMEOUT_S", 14.0)
    labels = [v3_label(c) for c in "abcd"]
    html = "".join(f'<a href="http://{lab}.onion/">Kosmas lighthouse log {i}</a>' for i, lab in enumerate(labels))
    budgets = []

    class Client(_CountingClient):
        def plan_distill(self, budget_s, **kw):
            budgets.append(budget_s)
            return super().plan_distill(budget_s, **kw)

    async def _engine(url, proxy, timeout, **kwargs):
        return 200, html

    async def _onion(url, proxy, *a, **k):
        await asyncio.sleep(0.01)
        if labels[0] in url or labels[1] in url:
            return FetchError(f"Error: Received status 403 from {url}")
        return _page(3)
    with patch.object(D, "_fetch_raw_html", side_effect=_engine), patch.object(D, "_fetch_onion_text", _onion):
        out = str(await asyncio.wait_for(D.tool_darkweb_research(
            "Kosmas lighthouse log", tor_proxy="socks5://127.0.0.1:9050", llm_client=Client()), 30.0))
    assert out.count("[EDGE EXTRACTED FACTS]") == 2, out[:600]
    assert 4.5 < budgets[0] <= 5.5, budgets


# ── review r1 ───────────────────────────────────────────────────────────────

def _llm_wrapper_classes():
    """Every class in src/ that wraps an LLM client: it answers `chat_completion` itself and hands
    every other attribute to the client inside it (`__getattr__`)."""
    import ast
    import importlib
    from pathlib import Path
    root = Path(NT.__file__).resolve().parents[1]
    found = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.ClassDef):
                continue
            names = {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if {"__getattr__", "chat_completion"} <= names:
                mod = "ghost_agent." + ".".join(path.relative_to(root).with_suffix("").parts)
                found.append(getattr(importlib.import_module(mod), node.name))
    return found


def test_every_wrapper_of_the_client_shares_its_queue(vloop, monkeypatch):
    """A background or isolated turn WRAPS the client. The queue belongs to the worker node, so the
    wrapped client must reach the same one — by the node's URL for a real client, by the client
    under the wrapper for one without nodes. Walks every wrapper class in src/: a new one that hides
    the client some other way fails here."""
    wrappers = _llm_wrapper_classes()
    assert len(wrappers) >= 2, wrappers

    class Pooled(SerialNode):
        def __init__(self, url):
            super().__init__()
            self.worker_clients = [{"url": url, "model": "worker"}]

    async def identities():
        bare, pooled = SerialNode(), Pooled("http://worker-a.test:8080")
        for cls in wrappers:
            assert NT.distill_queue(cls(bare)) is NT.distill_queue(bare), cls
            assert NT.distill_queue(cls(cls(bare))) is NT.distill_queue(bare), cls
            assert NT.distill_queue(cls(pooled)) is NT.distill_queue(pooled), cls
        # the node is what is shared: another client object on the same node, the same queue
        assert NT.distill_queue(Pooled("http://worker-a.test:8080")) is NT.distill_queue(pooled)
        assert NT.distill_queue(Pooled("http://worker-b.test:8080")) is not NT.distill_queue(pooled)
        assert NT.distill_queue(SerialNode()) is not NT.distill_queue(bare)
    vloop.run_until_complete(identities())

    # and what it is for: a foreground call and a wrapped (background) one never distil together
    monkeypatch.setattr(S, "helper_fetch_url_content", _fetcher())
    for cls in wrappers:
        for node in (SerialNode(), Pooled("http://worker-c.test:8080")):
            async def both():
                return await asyncio.gather(
                    _research(node, n=4),
                    S._deep_research_core(urls=_urls(8)[4:], llm_client=cls(node),
                                          tor_proxy="socks5h://127.0.0.1:9050", max_context=240_000))
            a, b = vloop.run_until_complete(both())
            assert node.peak == 1, (cls, node.peak)
            assert list(_blocks(str(a)).values()) + list(_blocks(str(b)).values()) == ["distilled"] * 8


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("slowdown,expect", [(1.0, "all"), (1.4, "all"), (1.5, "all"), (1.75, "most"),
                                             (2.0, "most"), (2.5, "most"), (3.0, "none"), (4.0, "none")])
def test_a_node_slower_than_its_own_figures(vloop, monkeypatch, warm, slowdown, expect):
    """The reviewer's table. The node's wall time is `slowdown` x what its timings say (cold, and
    with its rates already known).
      up to 1.5x  — a plan is sized for 85% of its budget: every page is distilled.
      1.75-2.5x   — a node that is merely busy. One page runs into its limit; the tracker is told,
                    every later plan is smaller, and at least 6 of 8 are distilled at reduced size.
      3x and more — even the SMALLEST plan times out. With the node's rates known (warm) its limit
                    has real slack, so two in a row end the wave: back after ~71 s, pages as raw text.
                    COLD the plan is sized from the conservative priors and a 23 s share is only
                    1.5x what they predict — no slack to judge by — so the cap ends it: four
                    timeouts in a row of any kind, back after ~94 s (it was all 8 shares, 187 s).
    Before these rules: at 1.4-1.5x one page was distilled and seven each burned ~23 s (187 s)."""
    node = SerialNode(slowdown=slowdown)
    if warm:
        node.nt.observe(URL, _timings(372.0, 48.0), concurrency=1)
    out, took = _run(vloop, monkeypatch, node)
    got = _blocks(out)
    assert sorted(got) == list(range(8)) and "failed" not in got.values(), got
    oks = [c["ok"] for c in node.calls]
    distilled = list(got.values()).count("distilled")
    assert distilled == oks.count(True)
    if expect == "all":
        assert distilled == 8 and all(oks), (got, oks)
    elif expect == "most":
        assert distilled >= 6 and oks.count(False) <= 2, (got, oks)
        first_bad = oks.index(False)
        sizes = [p["plan"].char_limit for p in node.plans if p["plan"].feasible]
        assert len(sizes) == len(oks)
        assert sizes[first_bad + 1] < 0.7 * sizes[first_bad], sizes                    # really smaller
        assert all(oks[first_bad + 1:]), oks                                           # and it worked
        assert took <= S._RESEARCH_PHASE_TIMEOUT_S
    elif warm:
        assert distilled == 0 and oks[-2:] == [False, False] and len(oks) <= 3, oks
        assert took <= 80.0, took
    else:
        assert distilled == 0 and oks == [False] * NT.WAVE_MAX_ANY_TIMEOUTS, oks
        assert 85.0 <= took <= 100.0, took
        assert all(c["timeout"] <= 24.0 for c in node.calls), [c["timeout"] for c in node.calls]   # no limit grows


def test_every_timeout_makes_the_next_plan_smaller_down_to_the_minimum(vloop):
    """A timeout returns no timings. The wave tells the tracker what was asked and how long it did
    not fit in (the plan rates drop at once) and halves the size of what it asks next — again after
    every further timeout, until the plan is the planner's smallest. Any other failure, however
    late (a reset connection at the very end of the limit), changes nothing. A success does not put
    the size back within the wave."""
    async def main():
        node = SerialNode()
        node.nt.observe(URL, _timings(372.0, 48.0), concurrency=1)
        keys = list("abcdefgh")
        wave = NT.DistillWave(node, keys, deadline=time.monotonic() + 10_000.0, ceiling_s=65.0)
        outcomes = dict(zip(keys, ["ok", "late-error", "timeout", "ok", "timeout", "ok", "ok", "ok"]))
        seen = []
        for key in keys:
            async with wave.turn(key) as turn:
                seen.append({"chars": turn.plan.char_limit, "tokens": turn.plan.max_tokens,
                             "decode": node.nt.plan_rates(URL)[1], "bound": turn.budget_s,
                             "minimum": turn.minimum, "feasible": turn.plan.feasible})
                if outcomes[key] == "ok":
                    wave.report(turn, True)
                else:       # a late error burned as much of the limit as a timeout; only the timeout counts
                    wave.report(turn, False, timed_out=outcomes[key] == "timeout", burned_s=turn.budget_s,
                                prompt_chars=turn.plan.char_limit, max_tokens=turn.plan.max_tokens)
        return seen, wave.gave_up
    seen, gave_up = vloop.run_until_complete(main())
    a, b, c, d, e, f, g, h = seen
    assert all(x["feasible"] for x in seen) and not gave_up
    assert all(x["bound"] == pytest.approx(65.0) for x in seen)      # the hard limit stays the page's budget
    assert (a["chars"], a["decode"]) == (b["chars"], b["decode"]) == (c["chars"], c["decode"])   # not a timeout: nothing
    assert not c["minimum"] and not d["minimum"]
    assert d["decode"] < 0.9 * c["decode"]                    # the timeout lowered the rate plans use
    assert d["chars"] < 0.6 * c["chars"] and d["tokens"] < 0.6 * c["tokens"]      # ... and the next plan is smaller
    assert e["chars"] == pytest.approx(d["chars"], rel=0.1)   # a success does not grow it back within the wave
    # the 2nd timeout: smaller again — here already the planner's smallest, and it stays there
    assert f["minimum"] and (f["chars"], f["tokens"]) == (NT.MIN_CHARS, NT.MIN_TOKENS)
    assert f["decode"] < 0.9 * e["decode"]
    assert g["minimum"] and h["minimum"]


def test_only_two_minimum_size_timeouts_in_a_row_end_the_wave(vloop):
    class Untaught(SerialNode):                # the wave's own rule, without the tracker's reaction
        def note_distill_timeout(self, *a, **k):
            return False

    async def run(pattern, ceiling=40.0):
        node = Untaught()
        node.nt.observe(URL, _timings(372.0, 48.0), concurrency=1)
        n = len(pattern) + 2
        wave = NT.DistillWave(node, list(range(n)), deadline=time.monotonic() + 10_000.0, ceiling_s=ceiling)
        got = []
        for i in range(n):
            t0 = time.monotonic()
            async with wave.turn(i) as turn:
                got.append(("min" if turn.minimum else "plan") if turn.plan is not None else turn.note)
                if turn.plan is not None and i < len(pattern):
                    wave.report(turn, pattern[i], timed_out=not pattern[i], burned_s=turn.budget_s,
                                prompt_chars=turn.plan.char_limit, max_tokens=turn.plan.max_tokens)
            assert time.monotonic() == t0               # a page of a wave that gave up does not wait
        return got
    F, T = False, True
    # larger plans timing out shrink the next one; they never end the wave by themselves
    got = vloop.run_until_complete(run([F, F, F]))
    assert got[0] == "plan" and "min" in got and NT.GAVE_UP_NOTE not in got[:got.index("min") + 1]
    # the smallest plan timing out twice in a row does
    got = vloop.run_until_complete(run([F, F, F, F, F, F]))
    first_min = got.index("min")
    assert got[first_min:first_min + 2] == ["min", "min"]
    assert got[first_min + 2:] == [NT.GAVE_UP_NOTE] * (len(got) - first_min - 2)
    # ... and a success between two of them starts the count again
    k = first_min
    got = vloop.run_until_complete(run([F] * k + [F, T, F, T]))
    assert NT.GAVE_UP_NOTE not in got and got[k:] == ["min"] * (len(got) - k)
    # a halved budget that buys exactly the smallest plan (no refusal on the way) is the smallest plan too:
    # 28 s -> 11.9 s to plan with, which pays the floors and nothing more
    got = vloop.run_until_complete(run([F, F, F], ceiling=28.0))
    assert got == ["plan", "min", "min", NT.GAVE_UP_NOTE, NT.GAVE_UP_NOTE]
    # no slack, no end: a 15 s limit is under twice the 8.3 s the smallest plan should take, however
    # often it times out (and with a tracker that learned nothing, the yardstick does not move)
    got = vloop.run_until_complete(run([F, F, F], ceiling=15.0))      # (three: the cap of four is its own test)
    assert NT.GAVE_UP_NOTE not in got and got.count("min") >= 3, got


def test_readings_above_the_width_never_set_a_plan_rate_on_their_own(monkeypatch):
    """A tracker that has heard ONLY above-width readings plans from the priors: a cold 3-way reading
    (2 tok/s) must not refuse the first solo plan, and a 5-way history must not plan the node faster
    than anything it showed alone. The first solo reading then governs."""
    monkeypatch.delenv("GHOST_DISTILL_WIDTH", raising=False)
    cold = NodeThroughput()
    assert cold.observe(URL, _timings(110.0, 2.0), concurrency=3)
    assert cold.plan_rates(URL)[:2] == (NT.DEFAULT_PREFILL_TOK_S, NT.DEFAULT_DECODE_TOK_S)
    assert cold.plan(46.0, URL, concurrency=1).feasible
    fast = NodeThroughput()
    for _ in range(6):
        fast.observe(URL, _timings(300.0, 40.0), concurrency=5)      # "solo equivalent" 671 / 200 tok/s
    assert fast.rates(URL)[1] > 150.0
    assert fast.plan_rates(URL)[:2] == (NT.DEFAULT_PREFILL_TOK_S, NT.DEFAULT_DECODE_TOK_S)
    assert fast.plan(46.0, URL, concurrency=1).max_tokens < NT.MAX_TOKENS
    fast.observe(URL, _timings(372.0, 48.0), concurrency=1)
    assert fast.plan_rates(URL)[1] == pytest.approx(48.0, rel=0.01)
    seeded = NodeThroughput()                                         # rates set directly keep working
    seeded._rates[URL] = [300.0, 30.0, 4]
    assert seeded.plan_rates(URL)[:2] == (300.0, 30.0)


def test_a_page_with_no_time_left_says_so(vloop, monkeypatch, capsys):
    """Its log line names the cause, not an exception class."""
    monkeypatch.setattr(S, "RESEARCH_ASSEMBLY_S", 5.0)
    node = SerialNode()            # a 26 s phase; the page takes 21.5 s to load: nothing is left
    out, _ = _run(vloop, monkeypatch, node, n=1, deadline_s=31.0, fetch=_fetcher(delays={0: 21.5}))
    assert _blocks(out) == {0: "raw"} and node.plans == [] and node.calls == []
    log = " ".join(capsys.readouterr().out.split())
    assert "no time left to distil after the page" in log, log[-1500:]
    assert "could not size the distill" not in log


def test_a_recheck_checks_its_fit_again_after_waiting_for_the_worker(vloop):
    """The recheck needs 25 s. It fits when asked (40 s left), waits 20 s for the worker, and then
    does not: nothing is sent (it used to be posted with whatever was left)."""
    node = SerialNode()
    urls = _urls(2)
    blocks = [f"### SOURCE: {u}\n[EDGE EXTRACTED FACTS]:\nNONE\n" for u in urls]
    texts = {u: (_page(i)[:6000], 128, 5.0) for i, u in enumerate(urls)}

    async def main():
        q = NT.distill_queue(node)
        assert await q.acquire(0)
        asyncio.get_running_loop().call_later(20.0, q.release)
        return await S.recheck_all_none(blocks, urls, texts, "lighthouse lamp", node, "m",
                                        deadline=time.monotonic() + 40.0, slot_wait=1.0)
    r = vloop.run_until_complete(main())
    assert node.calls == [] and r.fixed == 0


async def test_a_cancel_landing_as_an_onion_fetch_completes_is_not_swallowed(monkeypatch):
    """darkweb_research, python 3.10: `asyncio.wait_for` returns the result and DROPS a cancel that
    lands in the loop step its inner call finishes in — the research then went on to the worker."""
    from unittest.mock import patch
    from ghost_agent.tools import darkweb_search as D
    from tests.onion_v3 import v3_label
    monkeypatch.delenv("GHOST_DISTILL_WIDTH", raising=False)
    html = "".join(f'<a href="http://{v3_label(c)}.onion/">Kosmas lighthouse log {c}</a>' for c in "abcd")
    client, holder = _CountingClient(), {}

    async def _engine(url, proxy, timeout, **kwargs):
        return 200, html

    async def _onion(url, proxy, *a, **k):
        await asyncio.sleep(0.01)
        return _page(3)
    real = D._neutralise

    def _last_step_of_the_fetch_stage(text):
        holder["task"].cancel()                     # the owner stops the turn exactly as the page is ready
        return real(text)
    monkeypatch.setattr(D, "_neutralise", _last_step_of_the_fetch_stage)
    with patch.object(D, "_fetch_raw_html", side_effect=_engine), patch.object(D, "_fetch_onion_text", _onion):
        holder["task"] = asyncio.ensure_future(D.tool_darkweb_research(
            "Kosmas lighthouse log", tor_proxy="socks5://127.0.0.1:9050", llm_client=client))
        done, _ = await asyncio.wait({holder["task"]}, timeout=30.0)
        await asyncio.sleep(0.2)
    assert done and holder["task"].cancelled()
    assert client.labels == [], client.labels


def test_the_real_client_tells_every_worker_node_of_a_timeout():
    from ghost_agent.core.llm import LLMClient
    c = LLMClient.__new__(LLMClient)
    c.node_throughput = NodeThroughput()
    c.worker_clients = [{"url": "http://worker-a.test:8080"}, {"url": "http://worker-b.test:8080"}]
    for n in c.worker_clients:
        c.node_throughput.observe(n["url"], _timings(372.0, 48.0), concurrency=1)
    # 4,000 prompt tokens + 300 out were believed to take 17 s and did not fit in 30 s
    assert c.note_distill_timeout(4000, 300, 30.0, concurrency=1) is True
    for n in c.worker_clients:
        prefill, decode, _ = c.node_throughput.plan_rates(n["url"])
        assert prefill == pytest.approx(372.0 * 17.0 / 30.0, rel=0.02)
        assert decode == pytest.approx(48.0 * 17.0 / 30.0, rel=0.02)
    assert LLMClient.__new__(LLMClient).note_distill_timeout(4000, 300, 30.0) is False     # never raises


# ── review r2 ───────────────────────────────────────────────────────────────

class _Flaky(SerialNode):
    """Page 0's call fails with `exc` after `frac` of its time limit; the others are served."""
    exc, frac = None, 0.95

    async def chat_completion(self, payload, **kw):
        if self.exc is not None and "PAGE-MARK p0 " in payload["messages"][0]["content"]:
            await asyncio.sleep(kw["timeout"] * self.frac)
            raise self.exc
        return await super().chat_completion(payload, **kw)


def _named_timeout():
    class ReadTimeout(Exception):               # a client library's own timeout class
        pass
    return ReadTimeout("read timed out")


def _folded_timeout(flag):
    from ghost_agent.core.llm import OffMainNodeUnavailable
    e = OffMainNodeUnavailable("all off-main nodes failed")
    e.timed_out = flag
    return e


@pytest.mark.parametrize("make_exc,is_timeout", [
    (lambda: ConnectionError("reset by peer"), False),
    (lambda: ValueError("malformed answer"), False),
    (lambda: _folded_timeout(False), False),
    (lambda: asyncio.TimeoutError(), True),
    (_named_timeout, True),
    (lambda: _folded_timeout(True), True),
])
def test_only_a_time_limit_is_read_as_a_slow_node(vloop, monkeypatch, make_exc, is_timeout):
    """Page 0 fails at 95 % of its limit. A reset connection, a malformed answer or a refused node
    say nothing about speed: the tracker keeps 372 / 48 and the next plans keep their size. A
    timeout — the stdlib's, a client's own class, or the real client's folded
    `OffMainNodeUnavailable.timed_out` — lowers the rates and shrinks what follows."""
    node = _Flaky()
    node.exc = make_exc()
    for _ in range(3):
        node.nt.observe(URL, _timings(372.0, 48.0), concurrency=1)
    out, _ = _run(vloop, monkeypatch, node)
    got = _blocks(out)
    assert got[0] == "raw" and list(got.values()).count("distilled") == 7, got
    assert NT.is_distill_timeout(node.exc) is is_timeout
    sizes = [p["plan"].char_limit for p in node.plans if p["plan"].feasible]
    if is_timeout:
        assert len(node.notes) == 1
        assert sizes[1] < 0.7 * sizes[0], sizes
    else:
        assert node.notes == []
        assert sizes[1] >= sizes[0], sizes
        assert node.nt.plan_rates(URL)[1] == pytest.approx(48.0, rel=0.02)


@pytest.mark.parametrize("fault,want", [("timeout", True), ("reset", False), ("saturated", False)])
def test_the_real_client_says_when_a_worker_ran_into_the_time_limit(monkeypatch, fault, want):
    import httpx
    from unittest.mock import MagicMock
    import ghost_agent.core.llm as L
    exc = {"timeout": httpx.ReadTimeout("read timed out"), "reset": httpx.ConnectError("reset"),
           "saturated": L.NodeSaturated("no permit")}[fault]
    c = L.LLMClient.__new__(L.LLMClient)
    node = {"url": "http://worker-a.test:8080", "model": "Worker", "client": MagicMock()}
    c.worker_clients, c.critic_clients = [node], []
    c.get_worker_node = lambda *a, **k: node
    c.circuit_breaker = MagicMock()

    class _Gate:
        async def __aenter__(self):
            raise exc

        async def __aexit__(self, *a):
            return False
    c._node_slot = lambda *a, **k: _Gate()
    monkeypatch.setattr(L, "_charge_node_fault", lambda *a, **k: False)
    with pytest.raises(L.OffMainNodeUnavailable) as ei:
        asyncio.run(c._do_chat_completion(
            {"messages": [{"role": "user", "content": "x"}]},
            use_worker=True, off_main_only=True, task_label="web summary", timeout=5.0, slot_wait=5.0))
    assert ei.value.timed_out is want and NT.is_distill_timeout(ei.value) is want


def test_a_busy_moment_in_a_short_phase_does_not_end_the_wave(vloop, monkeypatch):
    """The reviewer's case: a 100 s bound (fact_check), the node 4x slow for the first 40 s, then
    healthy. A share is below the planner's minimum, so the first pages get the smallest plan on a
    limit with no slack; two of those timing out meant "gave up", 0 of 8, back at 34 s with a
    minute unused. Timeouts without slack still shrink and teach — the wave goes on."""
    monkeypatch.setattr(S, "helper_fetch_url_content", _fetcher())
    node = SerialNode(slowdown=4.0)
    for _ in range(3):
        node.nt.observe(URL, _timings(372.0, 48.0), concurrency=1)

    async def main():
        async def heal():
            await asyncio.sleep(40.0)
            node.slowdown = 1.0
        h = asyncio.ensure_future(heal())
        t0 = time.monotonic()
        out = str(await _research(node, deadline_s=100.0))
        await h
        return out, time.monotonic() - t0
    out, took = vloop.run_until_complete(main())
    got = _blocks(out)
    assert sorted(got) == list(range(8)) and "failed" not in got.values()
    assert list(got.values()).count("distilled") >= 5, got
    assert node.notes and not any(c["ok"] for c in node.calls[:len(node.notes)])     # it began with timeouts
    assert len(node.notes) < NT.WAVE_MAX_ANY_TIMEOUTS and took <= 100.0


@pytest.mark.parametrize("warm,calls,by", [(True, 3, 80.0), (False, 4, 100.0)])
def test_a_hung_worker_still_ends_the_wave_early(vloop, monkeypatch, capsys, warm, calls, by):
    """Nothing ever comes back. With the node's rates known, the smallest plan's limit has real
    slack and two of them timing out end the wave (3 calls, ~71 s). Cold there is no slack to judge
    by, and the cap does it: four timeouts in a row (~94 s). Either way every page is raw text and
    half the phase is not spent."""
    node = SerialNode(slowdown=1000.0)
    if warm:
        for _ in range(3):
            node.nt.observe(URL, _timings(372.0, 48.0), concurrency=1)
    out, took = _run(vloop, monkeypatch, node)
    assert _blocks(out) == {i: "raw" for i in range(8)}
    assert len(node.calls) == calls and took <= by, (len(node.calls), took)
    assert "kept running into its time limit" in " ".join(capsys.readouterr().out.split())


def test_four_timeouts_in_a_row_end_the_wave_and_a_success_starts_the_row_again(vloop):
    class Untaught(SerialNode):
        def note_distill_timeout(self, *a, **k):
            return False

    async def run(pattern):
        wave = NT.DistillWave(Untaught(), list(range(len(pattern) + 1)),
                              deadline=time.monotonic() + 10_000.0, ceiling_s=15.0)     # 15 s: never any slack
        got = []
        for i in range(len(pattern) + 1):
            async with wave.turn(i) as turn:
                got.append(turn.plan is not None)
                if turn.plan is not None and i < len(pattern):
                    wave.report(turn, pattern[i], timed_out=not pattern[i], burned_s=turn.budget_s,
                                prompt_chars=turn.plan.char_limit, max_tokens=turn.plan.max_tokens)
        return got
    F, T = False, True
    assert vloop.run_until_complete(run([F, F, F])) == [True] * 4
    assert vloop.run_until_complete(run([F, F, F, F, F])) == [True] * 4 + [False] * 2
    assert vloop.run_until_complete(run([F, F, F, T, F, F, F])) == [True] * 8
    assert vloop.run_until_complete(run([F, F, F, T, F, F, F, F])) == [True] * 8 + [False]
