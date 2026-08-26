"""deep_research / darkweb_research distill budgeting (2026-08-25, req 08766aa1).

Three defects shipped together and produced one symptom (`Nova: ReadTimeout`
that read as node sickness):

1. the distill prompt was sized from `args.max_context` — the MAIN model's
   240,000 — so it pinned to its 40,000-char ceiling on every call, which on
   Nova is 41s of prefill against a 45s budget;
2. every `_bounded()` per-URL clock started at `asyncio.gather`, so with
   `Semaphore(3)` URLs 4-8 reached the distiller with seconds left — the 4th
   was handed 6s for a 27s job and posted it anyway;
3. the shortfall check `raise`d OUTSIDE the try, so a URL it was supposed to
   degrade to raw text was dropped from the report entirely, while its own
   message claimed it was "taking the raw-text path".

And the circuit breaker could not see any of it: the 45s keepalive
(`max_tokens=1`, "ok") closed the breaker 7s into a 60s cooldown while every
real request was still failing.
"""
import asyncio
import time
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.llm import LLMClient, NodeCircuitBreaker
from ghost_agent.core.node_throughput import DistillPlan, NodeThroughput
from ghost_agent.tools import search as search_mod
from ghost_agent.tools.search import tool_deep_research

PAGE = "Qwen 3.6 35B-A3B runs at 16GB in Q4_K_M. " * 3000  # ~126k chars


def _plan(chars=12_000, tokens=320, feasible=True, reason=""):
    return DistillPlan(chars, tokens, feasible, reason, 220.0, 12.0, 5, 40.0)


class StubClient:
    """Records what the tool asked for and what it was sent."""

    def __init__(self, plan=None):
        self._plan = plan if plan is not None else _plan()
        self.budgets = []      # budget_s handed to plan_distill, per URL
        self.payloads = []     # payloads that actually reached the node
        self.max_chars_args = []
        self.fanouts = []

    def plan_distill(self, budget_s, **kwargs):
        self.budgets.append(budget_s)
        self.max_chars_args.append(kwargs.get("max_chars"))
        self.fanouts.append(kwargs.get("concurrency"))
        return self._plan

    def note_distill_density(self, *a, **k):
        return False

    async def chat_completion(self, payload, **kwargs):
        self.payloads.append((payload, kwargs))
        return {"choices": [{"message": {"content": "FACT: 16GB is enough."}}]}


def _ddgs(mock_ddgs, hrefs):
    inst = MagicMock()
    mock_ddgs.return_value.__enter__.return_value = inst
    inst.text.return_value = [{"href": h} for h in hrefs]
    return inst


async def _run(client, hrefs=("https://a.com/1",), fetch=None,
               max_context=240_000):
    with patch("ddgs.DDGS") as mock_ddgs, \
         patch("ghost_agent.tools.search.importlib.util.find_spec",
               return_value=True), \
         patch("ghost_agent.tools.search.helper_fetch_url_content",
               new_callable=AsyncMock) as mock_fetch:
        _ddgs(mock_ddgs, list(hrefs))
        mock_fetch.side_effect = fetch or (lambda url, **kw: PAGE)
        return await tool_deep_research(
            "best local llm", False, "socks5://127.0.0.1:9050",
            llm_client=client, max_context=max_context)


class TestTheRequestIsSizedByThePlan:
    @pytest.mark.asyncio
    async def test_payload_uses_the_plans_numbers_not_the_old_constants(self):
        client = StubClient(_plan(chars=9_000, tokens=300))
        out = await _run(client)
        assert len(client.payloads) == 1
        payload, kwargs = client.payloads[0]
        # The two constants that shipped and could not fit the budget.
        assert payload["max_tokens"] == 300
        assert payload["max_tokens"] != 2048
        content = payload["messages"][0]["content"]
        source_text = content.split("Source text:\n", 1)[1]
        assert len(source_text) <= 9_000
        assert len(source_text) < 40_000
        assert "FACT: 16GB is enough." in out

    @pytest.mark.asyncio
    async def test_the_mains_240k_context_does_not_size_the_worker(self):
        # Same page, wildly different main-model windows: the prompt the
        # WORKER receives must be identical, because the worker's own
        # throughput decides it. Under the old arithmetic an 8k main context
        # shrank the worker prompt and a 240k one pinned it to 40k.
        sizes = []
        for ctx in (8_192, 240_000):
            client = StubClient(_plan(chars=9_000, tokens=300))
            await _run(client, max_context=ctx)
            body = client.payloads[0][0]["messages"][0]["content"]
            sizes.append(len(body.split("Source text:\n", 1)[1]))
        assert sizes[0] == sizes[1]

    @pytest.mark.asyncio
    async def test_the_report_share_ceiling_comes_from_the_main_context(self):
        # `max_context` keeps ONE legitimate job: bounding each source's share
        # of the report that is read back into the main window.
        client = StubClient()
        await _run(client, hrefs=[f"https://a.com/{i}" for i in range(4)],
                   max_context=8_192)
        assert all(m is not None for m in client.max_chars_args)
        assert max(client.max_chars_args) < 40_000

    @pytest.mark.asyncio
    async def test_the_summary_budget_is_bounded_by_the_url_deadline(self):
        client = StubClient()
        await _run(client)
        assert client.budgets
        assert all(0 < b <= search_mod._WEB_SUMMARY_TIMEOUT_S
                   for b in client.budgets)


class TestARefusedPlanDegradesInsteadOfLosingTheURL:
    @pytest.mark.asyncio
    async def test_infeasible_plan_never_posts_and_keeps_the_source(self):
        client = StubClient(_plan(feasible=False, reason="only 3.0s left"))
        out = await _run(client)
        # The doomed POST is the thing being prevented.
        assert client.payloads == []
        # ...and the URL survives as raw text rather than vanishing, which is
        # what the escaped `raise TimeoutError` used to do.
        assert "https://a.com/1" in out
        assert "Qwen 3.6 35B-A3B" in out
        assert "[...truncated...]" in out

    @pytest.mark.asyncio
    async def test_a_failing_distill_still_degrades_to_raw_text(self):
        class Boom(StubClient):
            async def chat_completion(self, payload, **kwargs):
                raise TimeoutError("ReadTimeout")

        out = await _run(Boom())
        assert "https://a.com/1" in out
        assert "Qwen 3.6 35B-A3B" in out

    @pytest.mark.asyncio
    async def test_raw_fallback_is_capped_independently_of_the_plan(self):
        client = StubClient(_plan(feasible=False, reason="no budget"))
        out = await _run(client)
        body = out.split("### SOURCE: https://a.com/1\n", 1)[1]
        raw = body.split("\n[...truncated...]", 1)[0]
        assert len(raw) <= search_mod._RAW_FALLBACK_CHARS

    @pytest.mark.asyncio
    async def test_no_client_still_produces_a_source_block(self):
        out = await _run(None)
        assert "https://a.com/1" in out
        assert "Qwen 3.6 35B-A3B" in out


class TestThePerURLClockStartsWhenTheURLStarts:
    @pytest.mark.asyncio
    async def test_every_url_gets_the_same_budget_regardless_of_its_wave(
            self, monkeypatch):
        # Eight URLs through `Semaphore(3)` is three waves. With the clock
        # started at `gather`, wave 3 lost ~2 of its 6 seconds before it ran;
        # started at acquisition, every wave gets the same.
        monkeypatch.setattr(search_mod, "_PER_URL_TIMEOUT_S", 6.0)
        monkeypatch.setattr(search_mod, "_WEB_SUMMARY_TIMEOUT_S", 1000.0)
        monkeypatch.setattr(search_mod, "_MIN_URL_BUDGET_S", 0.5)
        # Scaled with the rest of the clock — the allowance is subtracted from
        # what the planner is offered, so at production size it would swamp a
        # deliberately tiny per-URL window.
        monkeypatch.setattr(search_mod, "_QUEUE_ALLOWANCE_S", 0.5)

        async def slow(url, **kw):
            await asyncio.sleep(1.0)
            return PAGE

        client = StubClient()
        await _run(client, hrefs=[f"https://a.com/{i}" for i in range(8)],
                   fetch=slow)
        assert len(client.budgets) == 8
        spread = max(client.budgets) - min(client.budgets)
        assert spread < 0.5, f"budgets drifted across waves: {client.budgets}"
        assert min(client.budgets) > 1.5
        # ⚠ A CEILING TOO. With only a floor and a spread bound, replacing the
        # per-URL deadline with a CONSTANT (`monotonic() + (budget_s and 55.0)`)
        # passes both — every budget is then a fixed ~52.5s, uniform and large.
        # The budget must be bounded by the window the semaphore issued.
        assert max(client.budgets) < search_mod._PER_URL_TIMEOUT_S, client.budgets

    @pytest.mark.asyncio
    async def test_the_phase_deadline_declines_late_urls_rather_than_posting(
            self, monkeypatch):
        monkeypatch.setattr(search_mod, "_RESEARCH_PHASE_TIMEOUT_S", 2.0)
        monkeypatch.setattr(search_mod, "_MIN_URL_BUDGET_S", 1.0)

        async def slow(url, **kw):
            await asyncio.sleep(1.2)
            return PAGE

        client = StubClient()
        out = await _run(client, hrefs=[f"https://a.com/{i}" for i in range(8)],
                         fetch=slow)
        # The later waves are refused outright — no Tor circuit, no worker slot.
        assert len(client.payloads) < 8
        assert "research phase deadline" in out
        # And the model is TOLD its coverage was cut, not silently shortchanged.
        assert "SOURCE FAILURES" in out

    @pytest.mark.asyncio
    async def test_a_generous_phase_lets_every_url_through(self):
        client = StubClient()
        await _run(client, hrefs=[f"https://a.com/{i}" for i in range(8)])
        assert len(client.payloads) == 8


class TestBreakerAndThroughputRecording:
    """A heartbeat is evidence of sickness, never evidence of health."""

    def _client(self):
        c = LLMClient.__new__(LLMClient)
        c.circuit_breaker = NodeCircuitBreaker(failure_threshold=1,
                                               cooldown_seconds=60.0)
        c.node_throughput = NodeThroughput()
        return c

    def _resp(self, prompt_n=12000, prompt_ms=40000,
               predicted_n=400, predicted_ms=30000):
        return types.SimpleNamespace(json=lambda: {
            "choices": [{"message": {"content": "x"}}],
            "timings": {"prompt_n": prompt_n, "prompt_ms": prompt_ms,
                        "predicted_n": predicted_n,
                        "predicted_ms": predicted_ms}})

    def test_a_keepalive_success_cannot_close_an_open_breaker(self):
        # Live shape: three real timeouts opened the breaker at +101s and a
        # keepalive closed it at +108s — 7s into a 60s cooldown — while every
        # real request was still failing.
        c = self._client()
        node = {"url": "http://nova:8088", "model": "E4B"}
        c.circuit_breaker.record_failure(node["url"])
        assert c.circuit_breaker.get_status()[node["url"]]["state"] == "open"
        c._on_node_success(node, self._resp(), "worker", "keepalive")
        assert c.circuit_breaker.get_status()[node["url"]]["state"] == "open"
        c._on_node_success(node, self._resp(), "worker", "warmup")
        assert c.circuit_breaker.get_status()[node["url"]]["state"] == "open"

    def test_a_real_success_does_close_it(self):
        c = self._client()
        node = {"url": "http://nova:8088", "model": "E4B"}
        c.circuit_breaker.record_failure(node["url"])
        c._on_node_success(node, self._resp(), "worker", "web summary")
        assert c.circuit_breaker.get_status()[node["url"]]["state"] == "closed"

    def test_throughput_is_learned_from_real_calls_of_any_label(self):
        c = self._client()
        node = {"url": "http://nova:8088", "model": "E4B"}
        c._on_node_success(node, self._resp(), "worker", "web summary")
        assert c.node_throughput.rates(node["url"])[2] == 1

    def test_a_keepalive_sized_response_teaches_nothing(self):
        c = self._client()
        node = {"url": "http://nova:8088", "model": "E4B"}
        # The real heartbeat shape: `max_tokens=1`, so BOTH halves are
        # below the sampling floors. (An earlier version of this test passed
        # 400 decoded tokens and "passed" for the wrong reason.)
        c._on_node_success(node, self._resp(prompt_n=1, prompt_ms=70,
                                            predicted_n=1, predicted_ms=30),
                           "worker", "keepalive")
        assert c.node_throughput.rates(node["url"])[2] == 0

    def test_the_leg_stamp_survives_the_refactor(self):
        # `_stamp_leg` used to be called inline at each of the five pool
        # branches; collapsing them into one helper must not drop it.
        c = self._client()
        node = {"url": "http://nova:8088", "model": "E4B"}
        data = c._on_node_success(node, self._resp(), "critic", "verify")
        assert data["_ghost_leg"]["served_by"] == "critic"


class TestPlanDistillPoolSelection:
    def _client(self, pools):
        c = LLMClient.__new__(LLMClient)
        c.node_throughput = NodeThroughput()
        c.worker_clients = pools
        c.critic_clients = []
        c._node_slots, c._node_slot_built_cap, c._node_run_tasks = {}, {}, {}
        return c

    def test_an_empty_pool_still_plans_on_defaults(self):
        c = self._client([])
        assert c.plan_distill(45.0).feasible is True

    def test_a_mixed_pool_is_sized_for_the_slowest_node(self):
        fast, slow = {"url": "http://fast"}, {"url": "http://slow"}
        c = self._client([fast, slow])
        # Rates chosen so BOTH are feasible and NEITHER hits the 40k ceiling
        # — at the ceiling the two plans are equal and the test proves
        # nothing about which node was sized for.
        c.node_throughput._rates["http://fast"] = [400.0, 25.0, 5]
        c.node_throughput._rates["http://slow"] = [200.0, 12.0, 5]
        only_fast = self._client([fast])
        only_fast.node_throughput._rates["http://fast"] = [400.0, 25.0, 5]
        assert (c.plan_distill(45.0).char_limit
                < only_fast.plan_distill(45.0).char_limit)


class TestTelemetryCannotDestroyTheDegradation:
    """The handler that SAVES the URL must not be able to lose it.

    Found by the suite while shipping this fix: a `{plan.char_limit:,}` format
    spec in the degradation log raised against a non-numeric plan, escaped
    `process_url`, was swallowed by `gather(return_exceptions=True)`, and
    dropped the source — re-creating the exact loss the rewrite removed. Two
    previously-passing tests went red. The lesson is the codebase's own:
    telemetry must never break the call it is describing.
    """

    @pytest.mark.asyncio
    async def test_an_unformattable_plan_still_degrades_to_raw_text(self):
        class Hostile:
            """Answers `feasible` but explodes on every description."""
            feasible = True
            char_limit = 5_000
            max_tokens = 300

            def __format__(self, spec):
                raise TypeError("unsupported format string")

            def describe(self):
                raise RuntimeError("no")

        class Client(StubClient):
            def plan_distill(self, budget_s, **kwargs):
                p = Hostile()
                p.char_limit = Hostile()      # blows up the `:,` format spec
                p.max_tokens = 300
                return p

            async def chat_completion(self, payload, **kwargs):
                raise TimeoutError("ReadTimeout")

        out = await _run(Client())
        assert "https://a.com/1" in out
        assert "Qwen 3.6 35B-A3B" in out          # the source survived

    @pytest.mark.asyncio
    async def test_a_describe_that_raises_does_not_lose_a_good_distill(self):
        class Client(StubClient):
            def plan_distill(self, budget_s, **kwargs):
                p = _plan(chars=5_000, tokens=300)
                object.__setattr__(p, "describe", None)   # not callable
                return p

        out = await _run(Client())
        assert "FACT: 16GB is enough." in out

    @pytest.mark.asyncio
    async def test_a_client_without_plan_distill_degrades_instead_of_raising(self):
        # An older or hand-rolled client. Before the guard this raised
        # AttributeError inside process_url and the source vanished.
        class NoPlanner:
            async def chat_completion(self, payload, **kwargs):
                raise AssertionError("must not be reached")

        out = await _run(NoPlanner())
        assert "https://a.com/1" in out
        assert "Qwen 3.6 35B-A3B" in out


class TestTheQueueAllowance:
    """The plan assumes it starts soon; the permit wait must enforce that."""

    @pytest.mark.asyncio
    async def test_the_plan_is_sized_below_the_budget_by_the_allowance(self):
        client = StubClient()
        await _run(client)
        # What the planner was offered is the wall budget MINUS the time the
        # call is allowed to spend queueing for a node permit. Sizing for the
        # full budget is how a queued request ends up POSTing work that its
        # remaining time cannot pay for.
        offered = client.budgets[0]
        _, kwargs = client.payloads[0]
        assert offered == pytest.approx(
            kwargs["total_budget"] - search_mod._QUEUE_ALLOWANCE_S, abs=0.5)

    @pytest.mark.asyncio
    async def test_the_permit_wait_is_bounded_by_the_allowance(self):
        client = StubClient()
        await _run(client)
        _, kwargs = client.payloads[0]
        assert kwargs["slot_wait"] == search_mod._QUEUE_ALLOWANCE_S
        # ...and is strictly less than the total, so queueing can never eat
        # the whole budget the way `slot_wait == total_budget` allowed.
        assert kwargs["slot_wait"] < kwargs["total_budget"]


# ── the onion sibling ──────────────────────────────────────────────────────
# Review finding: every guard in `darkweb_search.process_url` was UNPINNED —
# this file's docstring named both tools and only ever imported the clearnet
# one, and the two modules have a documented history of drifting apart.
from ghost_agent.tools import darkweb_search as dw_mod          # noqa: E402

V3 = "a" * 56


def _onion_stub(html):
    async def _stub(url, proxy, timeout, **kwargs):
        return 200, html
    return _stub


async def _run_dw(client, page=PAGE):
    html = f'<a href="http://{V3}.onion/">Target</a>'
    with patch.object(dw_mod, "_fetch_raw_html",
                      side_effect=_onion_stub(html)), \
         patch.object(dw_mod, "_fetch_onion_text",
                      new_callable=AsyncMock) as mfetch:
        mfetch.return_value = page
        return await dw_mod.tool_darkweb_research(
            "target topic", tor_proxy="socks5://127.0.0.1:9050",
            llm_client=client)


class TestDarkwebSharesTheGuards:
    @pytest.mark.asyncio
    async def test_onion_distill_uses_the_plan(self):
        client = StubClient(_plan(chars=7_000, tokens=280))
        out = await _run_dw(client)
        assert client.payloads, "the onion distill never ran"
        payload, _ = client.payloads[0]
        assert payload["max_tokens"] == 280
        assert len(payload["messages"][0]["content"]
                   .split("Source text:\n", 1)[1]) <= 7_000
        assert "FACT: 16GB is enough." in out

    @pytest.mark.asyncio
    async def test_an_infeasible_onion_plan_keeps_the_source(self):
        client = StubClient(_plan(feasible=False, reason="no budget"))
        out = await _run_dw(client)
        assert client.payloads == []
        assert V3 in out
        assert "Qwen 3.6 35B-A3B" in out

    @pytest.mark.asyncio
    async def test_an_unformattable_onion_plan_still_degrades(self):
        class Hostile:
            feasible = True
            max_tokens = 300

            def __format__(self, spec):
                raise TypeError("nope")

        class Client(StubClient):
            def plan_distill(self, budget_s, **kwargs):
                p = Hostile()
                p.char_limit = Hostile()
                return p

            async def chat_completion(self, payload, **kwargs):
                raise TimeoutError("ReadTimeout")

        out = await _run_dw(Client())
        assert V3 in out and "Qwen 3.6 35B-A3B" in out

    @pytest.mark.asyncio
    async def test_a_non_string_onion_fetch_does_not_drop_the_source(self):
        # The clearnet sibling coerces; this one returned the awaited value
        # verbatim and then indexed it — one `return None` from losing the
        # onion (review, CONFIRMED by injection).
        client = StubClient()
        out = await _run_dw(client, page=None)
        assert V3 in out
        # ⚠ AND IT MUST BE COERCED, NOT MERELY CAUGHT. The `except Exception`
        # net in `_bounded` also keeps the source block alive, so an assertion
        # that only checks the URL survived passes in BOTH worlds — it cannot
        # distinguish "the page was kept as text" from "the coroutine crashed
        # and we reported it". The sibling coerces; this must too.
        assert "internal error" not in out

    @pytest.mark.asyncio
    async def test_an_onion_run_reports_its_coverage(self):
        # The failure banner existed only on the clearnet side, so a
        # partly-failed onion run rendered as a confident SHORT report.
        client = StubClient(_plan(feasible=False, reason="no budget"))
        out = await _run_dw(client)
        assert "DARK-WEB RESEARCH RESULT" in out
        assert "could not be distilled" in out or "SOURCE FAILURES" in out

    def test_the_two_modules_share_one_definition_of_every_budget(self):
        # Import-time value binding silently un-linked them once already: a
        # constant added to search.py and not to darkweb's import list was a
        # NameError inside the onion distill path.
        import ghost_agent.tools.search as s
        for name in ("_RESEARCH_PHASE_TIMEOUT_S", "_MIN_URL_BUDGET_S",
                     "_QUEUE_ALLOWANCE_S", "_WEB_SUMMARY_TIMEOUT_S"):
            assert hasattr(s, name), f"search.py lost {name}"
        assert dw_mod._budgets is s


class TestTheFanOutIsDeclared:
    """The wave is planned before any of it is in flight."""

    @pytest.mark.asyncio
    async def test_deep_research_declares_its_semaphore_width(self):
        # Without this the node looks IDLE to every plan in the wave and each
        # is sized as if it had the box to itself — measured live as four
        # ~19,936-char plans built at ~306/32 tok/s that then ran at 115/11.
        client = StubClient()
        await _run(client, hrefs=[f"https://a.com/{i}" for i in range(4)])
        assert client.fanouts, "no plan was requested"
        assert all(f == 3 for f in client.fanouts), client.fanouts

    @pytest.mark.asyncio
    async def test_darkweb_declares_its_own_narrower_width(self):
        client = StubClient(_plan(chars=6_000, tokens=200))
        await _run_dw(client)
        assert client.fanouts and all(f == 2 for f in client.fanouts), client.fanouts


class TestPlanDistillRaisesForBusyNodes:
    def _client(self, url="http://n", cap=4, held=0):
        import asyncio as _a
        c = LLMClient.__new__(LLMClient)
        c.node_throughput = NodeThroughput()
        c.worker_clients = [{"url": url}]
        c.critic_clients = []
        sem = _a.Semaphore(cap)
        for _ in range(held):
            sem._value -= 1
        c._node_slots = {url: sem}
        c._node_slot_built_cap = {url: cap}
        return c

    def test_an_idle_node_uses_the_declared_fan_out(self):
        c = self._client(held=0)
        c.node_throughput._rates["http://n"] = [304.0, 31.6, 9]
        idle = c.plan_distill(37.0, concurrency=3)
        assert idle.decode_tok_s == pytest.approx(31.6 / 3, rel=0.02)
        # An IDLE node with no declaration must plan at 1, not 2.
        solo = c.plan_distill(37.0, concurrency=1)
        assert solo.decode_tok_s == pytest.approx(31.6, rel=0.02)

    def test_a_busy_node_raises_it_above_the_declaration(self):
        # --worker-nodes and --critic-nodes point at the same box in this
        # deployment, so a turn-gate verify is sharing these slots. A plan
        # that believes the caller's number and ignores the traffic already
        # there is sized for a node that does not exist.
        c = self._client(held=3)
        c.node_throughput._rates["http://n"] = [304.0, 31.6, 9]
        busy = c.plan_distill(37.0, concurrency=1)
        assert busy.decode_tok_s == pytest.approx(31.6 / 4, rel=0.02)

    def test_it_never_plans_above_the_gate_s_own_cap(self):
        # The gate admits at most `cap`, so a larger figure plans for a node
        # that cannot exist — and the over-count produced blanket refusals.
        c = self._client(cap=4, held=4)
        c.node_throughput._rates["http://n"] = [304.0, 31.6, 9]
        p = c.plan_distill(37.0, concurrency=8)
        assert p.decode_tok_s == pytest.approx(31.6 / 4, rel=0.02)

    def test_inflight_reports_OTHERS_and_is_total(self):
        # ⚠ 0, NOT 1. It used to return `max(1, ...)`, conflating "idle" with
        # "cannot tell" — and `plan_distill` adds one for the caller, so an
        # idle node could never be planned at concurrency 1 and every
        # single-shot distill had its decode halved for absent company.
        c = LLMClient.__new__(LLMClient)
        c.node_throughput = NodeThroughput()
        c._node_slots = {}
        c._node_slot_built_cap = {}
        assert c._node_inflight("http://nowhere") == 0
        assert c._node_inflight("") == 0


class TestConcurrencyIsMeasuredOnTheLivePath:
    """⚠ THE PIN MUST SIT WHERE PRODUCTION READS IT.

    The previous version sampled `_run_concurrency()` INSIDE
    `async with _node_slot(...)` and asserted [3, 3, 3]. It passed — and said
    nothing, because `_on_node_success` runs AFTER that block, by which point
    the gate's `finally` has discarded the bookkeeping. Measured: 3 inside the
    permit, 1 after it. So `observe()` was handed 1 on every real call, loaded
    rates were stored as solo rates and then divided a second time at plan
    time, and the distiller read ~4,600 chars of a page instead of ~15,000 —
    less than the raw-text fallback it exists to beat. This drives the real
    dispatch path instead.
    """

    def _client(self, url="http://n", slots=4, decode_ms=600.0):
        import asyncio as _a
        from unittest.mock import AsyncMock as _AM, MagicMock as _MM
        c = LLMClient.__new__(LLMClient)
        c.node_throughput = NodeThroughput()
        c.circuit_breaker = NodeCircuitBreaker(failure_threshold=3,
                                               cooldown_seconds=60.0)
        c._node_slots = {url: _a.Semaphore(slots)}
        c._node_slot_built_cap = {url: slots}
        c._node_slot_caps = {url: slots}
        c._node_run_tasks = {}
        c._node_cap_retry_at, c._node_slot_locks = {}, {}
        c._node_slot_default = slots

        async def _post(*a, **k):
            await _a.sleep(0.30)
            r = _MM()
            r.raise_for_status = lambda: None
            r.json = lambda: {
                "choices": [{"message": {"content": "x"}}],
                # 12,000 prompt tokens in 54.5s and 600 tokens in 50s: the
                # rate a request measures WHILE SHARING the node three ways.
                "timings": {"prompt_n": 12000, "prompt_ms": 54500,
                            "predicted_n": 600, "predicted_ms": 50000}}
            return r
        node = {"url": url, "model": "m", "client": _MM(), "name": "N"}
        node["client"].post = _AM(side_effect=_post)
        props = _MM()
        props.json = lambda: {"total_slots": slots,
                              "default_generation_settings": {"n_ctx": 32768}}
        props.raise_for_status = lambda: None
        node["client"].get = _AM(return_value=props)
        return c, node

    @pytest.mark.asyncio
    async def test_a_three_way_wave_is_stored_as_a_solo_equivalent_rate(self):
        # ⚠ DRIVEN THROUGH `_do_chat_completion`, NOT A HAND-ROLLED IMITATION.
        # A first version of this test acquired the permit itself and called
        # `_on_node_success` directly — and a mutant that moved the capture
        # back OUTSIDE the permit (the original bug, verbatim) SURVIVED it.
        # A pin that reproduces the production path by hand cannot fail when
        # the production path changes. This one dispatches for real.
        import asyncio as _a
        c, node = self._client()
        c.worker_clients = [node]
        c._worker_index = 0
        c.foreground_tasks = 0
        await c._node_capacity(node)

        async def one():
            return await c._do_chat_completion(
                {"model": "m", "messages": []}, use_worker=True,
                timeout=30, slot_wait=10, total_budget=30,
                task_label="web summary")

        await _a.gather(one(), one(), one())
        prefill, decode, n = c.node_throughput.rates(node["url"])
        assert n == 3
        # Raw per-request readings are 220 / 12. Normalised for the 3-way it
        # actually ran at, the SOLO-equivalent must come back near 304 / 36.
        assert decode == pytest.approx(36.0, rel=0.15), decode
        assert prefill == pytest.approx(381.0, rel=0.15), prefill
        # ...and re-planning at that same fan-out returns the measurement.
        p = c.node_throughput.plan(60.0, node["url"], concurrency=3)
        assert p.decode_tok_s == pytest.approx(12.0, rel=0.15)

    @pytest.mark.asyncio
    async def test_a_solo_request_is_not_inflated(self):
        c, node = self._client()
        c.worker_clients = [node]
        c._worker_index = 0
        c.foreground_tasks = 0
        await c._node_capacity(node)
        await c._do_chat_completion(
            {"model": "m", "messages": []}, use_worker=True,
            timeout=30, slot_wait=10, total_budget=30, task_label="web summary")
        assert c.node_throughput.rates(node["url"])[1] == pytest.approx(12.0, rel=0.05)

    @pytest.mark.asyncio
    async def test_attribution_is_time_weighted_not_peak(self):
        # A request that ran alone for ~95% of its life and had three others
        # join at the very end must NOT be attributed 4 — the solo-equivalent
        # is `rate * N**exp`, so an over-attributed peak inflates the learned
        # rate permanently.
        import asyncio as _a
        c, node = self._client()
        got = []

        async def lonely():
            async with c._node_slot(node, wait_timeout=5):
                await _a.sleep(1.0)
                got.append(c._run_concurrency(node["url"]))

        async def latecomer():
            await _a.sleep(0.95)
            async with c._node_slot(node, wait_timeout=5):
                await _a.sleep(0.10)

        await _a.gather(lonely(), latecomer(), latecomer(), latecomer())
        assert got == [1], got



class TestTheBudgetsContainEachOther:
    """Structural pins. Nothing in the suite failed when the phase clock and
    the per-URL leash drifted apart — which is exactly how a container that
    grew 25% while its consumer grew 45% came to ship."""

    def test_the_phase_funds_the_waves_it_must_run(self):
        import math
        import ghost_agent.tools.search as s
        # 8 URLs is what `tool_deep_research` takes, through Semaphore(3).
        waves = math.ceil(8 / s._DISTILL_FANOUT_FOR_TESTS)
        # A URL is only STARTED if the phase has `_MIN_URL_BUDGET_S` left, so
        # the last wave needs at least that much beyond the preceding ones.
        need = (waves - 1) * s._PER_URL_TIMEOUT_S + s._MIN_URL_BUDGET_S
        assert s._RESEARCH_PHASE_TIMEOUT_S >= need, (
            f"phase {s._RESEARCH_PHASE_TIMEOUT_S:.0f}s cannot fund {waves} "
            f"waves of {s._PER_URL_TIMEOUT_S:.0f}s — the last wave is starved "
            f"and its sources fall back to raw HTML")

    def test_the_per_url_leash_funds_a_fetch_and_a_distill(self):
        import ghost_agent.tools.search as s
        # One fetch attempt, the 2s reserve, and the queue allowance must all
        # fit inside the per-URL window with the distill's own floor left over.
        overhead = s._FETCH_ATTEMPT_TIMEOUT + 2.0 + s._QUEUE_ALLOWANCE_S
        assert s._PER_URL_TIMEOUT_S > overhead, (
            f"per-URL {s._PER_URL_TIMEOUT_S}s leaves nothing after "
            f"{overhead}s of fetch+overhead")

    def test_the_queue_allowance_is_not_below_the_clients_own_floor(self):
        # `llm.py` clamps slot_wait up to `_MIN_SLOT_WAIT + _MIN_HTTP_FLOOR`,
        # so a smaller value here is silently ignored there while still being
        # subtracted from the planner's budget — the plan would over-reserve.
        import ghost_agent.core.llm as l
        import ghost_agent.tools.search as s
        floor = getattr(l, "_MIN_SLOT_WAIT", 5.0) + getattr(l, "_MIN_HTTP_FLOOR", 3.0)
        assert s._QUEUE_ALLOWANCE_S >= floor, (s._QUEUE_ALLOWANCE_S, floor)

    def test_the_onion_floor_covers_the_onion_fetch_not_the_clearnet_one(self):
        # The clearnet floor is derived from a 22s attempt; the onion fetch's
        # own wait_for is `_ONION_PAGE_TIMEOUT + 5`. Sharing the clearnet
        # number admitted onions that the outer wait_for then killed mid-fetch.
        import inspect
        import ghost_agent.tools.darkweb_search as dw
        src = inspect.getsource(dw.tool_darkweb_research)
        assert "_MIN_URL_BUDGET_S = _ONION_PAGE_TIMEOUT" in src, (
            "the onion path is back on the clearnet fetch floor")
        assert "_budgets._MIN_URL_BUDGET_S" not in src

    def test_a_floor_sized_plan_is_defended_on_signal_not_coverage(self):
        # MIN_CHARS < _RAW_FALLBACK_CHARS means a floor-sized distill READS
        # less of the page than declining would hand over. That is allowed —
        # the extract wins on signal — but the comment must not claim coverage.
        import ghost_agent.core.node_throughput as nt
        import ghost_agent.tools.search as s
        if nt.MIN_CHARS < s._RAW_FALLBACK_CHARS:
            assert "SIGNAL, NOT COVERAGE" in inspect_source(nt), (
                "the floor reads less of the page than the fallback and no "
                "longer says why that is acceptable")


def inspect_source(mod):
    import inspect
    return inspect.getsource(mod)


class TestSurvivingMutants:
    """Kill recipes from the mutation audit — each names the world it fails in."""

    @pytest.mark.asyncio
    async def test_a_declined_plan_whose_describe_raises_still_keeps_the_source(self):
        # `log_plan` wraps its whole body, INCLUDING `plan.describe()`, in
        # try/except. Nothing exercised it: both hostile-plan tests set
        # feasible=True, and `log_distill_plan` is only reached on the
        # NOT-feasible branch. Without the guard the exception escapes
        # `process_url` and `gather(return_exceptions=True)` drops the source.
        class Angry:
            feasible = False
            char_limit = 0
            max_tokens = 0
            reason = "no"

            def describe(self):
                raise RuntimeError("telemetry exploded")

        class Client(StubClient):
            def plan_distill(self, budget_s, **kwargs):
                return Angry()

        out = await _run(Client())
        assert "https://a.com/1" in out
        assert "Qwen 3.6 35B-A3B" in out

    @pytest.mark.asyncio
    async def test_the_onion_path_survives_a_client_without_plan_distill(self):
        # The clearnet twin of this dies; the onion one survived, because the
        # guard tests were written against `_run` with no `_run_dw` mirror —
        # in a file whose own comment says these two modules drift.
        class NoPlanner:
            async def chat_completion(self, payload, **kwargs):
                raise AssertionError("must not be reached")

        out = await _run_dw(NoPlanner())
        assert V3 in out and "Qwen 3.6 35B-A3B" in out

    @pytest.mark.asyncio
    async def test_the_onion_path_survives_a_describe_that_raises(self):
        class Client(StubClient):
            def plan_distill(self, budget_s, **kwargs):
                p = _plan(chars=6_000, tokens=200)
                object.__setattr__(p, "describe", None)   # not callable
                return p

        out = await _run_dw(Client())
        assert "FACT: 16GB is enough." in out
