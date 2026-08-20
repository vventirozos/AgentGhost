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

    def test_warmup_fans_out_concurrently(self):
        """⚠ The `off_main_only` half of this test was DELETED, not fixed.

        It read `assert "off_main_only=True" in getsource(warm_up_workers)`,
        which is satisfied by the explanatory COMMENT a few lines above the
        call — flipping the real kwarg to False left it green (R5 lens C).
        The property is already proved behaviourally by
        `test_dead_worker_never_falls_back_to_main` below, which is the only
        assertion of it worth keeping.

        The concurrency half survives because serial warmups re-grab the one
        slot that freed first, leaving the other `-np` slots cold — the
        opposite of the stated intent."""
        # ⚠ DELETED, not weakened: `assert "asyncio.gather" in src` is
        # satisfied by the single-request PROBE call one line above the
        # fan-out, so a fully serial implementation passes (R7 lens C, M41).
        # Concurrency here is a performance property with no correctness
        # consequence — warmup is best-effort at boot — so it gets no
        # replacement rather than a fake one.
        src = inspect.getsource(LLMClient.warm_up_workers)
        assert "_slots" in src, "warmup no longer sizes itself to the node"

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
    def test_off_main_pool_does_not_wait_for_the_foreground(self):
        """⚠ WAS A SOURCE-TEXT PIN (`'use_coding and getattr(self, ...' in
        src`), which broke the moment the predicate was rewritten — and could
        never have seen the defect it guarded. Observe the WAIT instead.

        A background call bound for a pool on ANOTHER box must not park for
        the main slot it never uses."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        waited = []
        c = LLMClient("http://127.0.0.1:8088")
        c._wait_for_foreground_clear = AsyncMock(
            side_effect=lambda *a, **k: waited.append(1))
        cli = MagicMock()
        ok = MagicMock()
        ok.json = lambda: {"choices": [{"message": {"content": "x"}}]}
        ok.raise_for_status = lambda: None
        cli.post = AsyncMock(return_value=ok)
        c.coding_clients = [{"url": "http://100.0.0.9:8088", "model": "cm",
                             "client": cli, "name": "Coder"}]
        c._coding_index = 0
        asyncio.run(c.chat_completion({"model": "m", "messages": []},
                                      use_coding=True, is_background=True))
        assert waited == [], (
            "a background call to an off-main pool parked for the main slot")

    def test_a_pool_whose_URL_IS_the_main_box_counts_as_MAIN(self):
        """The live shape: `--visual-nodes` is byte-identical to
        `--upstream-url`. The predicate asked "is a pool configured?", so a
        background vision call answered "not main", skipped the foreground
        wait AND the background semaphore, and landed on the single main slot
        mid-turn (R2 lens B, NEW-1)."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        waited = []
        c = LLMClient("http://127.0.0.1:8088")
        c._wait_for_foreground_clear = AsyncMock(
            side_effect=lambda *a, **k: waited.append(1))
        cli = MagicMock()
        ok = MagicMock()
        ok.json = lambda: {"choices": [{"message": {"content": "x"}}]}
        ok.raise_for_status = lambda: None
        cli.post = AsyncMock(return_value=ok)
        # Same URL as the main upstream — the production vision config.
        c.vision_clients = [{"url": "http://127.0.0.1:8088", "model": "Eva",
                             "client": cli, "name": "Eva"}]
        c._vision_index = 0
        asyncio.run(c.chat_completion({"model": "m", "messages": []},
                                      use_vision=True, is_background=True))
        assert waited == [1], (
            "a background call to a pool that IS the main box skipped the "
            "foreground wait — the post-req-70 starvation, reintroduced")


# ---------------------------------------------------------------------------
# LLM review 2026-08-18 — three independent lenses
# ---------------------------------------------------------------------------

class TestTheMainFallbackIsVisibleAndBounded:
    """When a pool call falls back to the MAIN model, two things used to be
    invisible: the caller could not tell (identical result shape), and the
    call ran with httpx's 1200s default on the FOREGROUND path."""

    def _client(self, main_capture):
        from unittest.mock import AsyncMock, MagicMock
        from ghost_agent.core.llm import LLMClient
        c = LLMClient("http://upstream.invalid:8080")
        dead = MagicMock()
        dead.post = AsyncMock(side_effect=RuntimeError("node down"))
        c.critic_clients = [{"url": "http://n", "model": "cm",
                             "client": dead, "name": "Nova"}]
        c._critic_index = 0

        async def _main_post(path, **kw):
            main_capture["timeout"] = kw.get("timeout", "<unset>")
            r = MagicMock()
            r.json = lambda: {"choices": [{"message": {"content": "FROM_MAIN"}}]}
            r.raise_for_status = lambda: None
            r.status_code, r.text = 200, "{}"
            return r

        c.http_client = MagicMock()
        c.http_client.post = AsyncMock(side_effect=_main_post)
        return c

    def test_the_result_says_which_leg_served_it(self):
        import asyncio
        from ghost_agent.core.llm import served_leg
        cap = {}
        c = self._client(cap)
        res = asyncio.run(c._do_chat_completion(
            {"model": "m", "messages": []}, use_critic=True, timeout=120.0))
        leg = served_leg(res)
        assert leg["served_by"] == "main", (
            f"a MAIN-model answer is indistinguishable from a critic one: "
            f"{leg} — the §4BR degradation guard reads this")
        assert leg["fell_back_from"] == "critic"

    def test_a_pool_served_result_says_so(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from ghost_agent.core.llm import LLMClient, served_leg
        c = LLMClient("http://upstream.invalid:8080")
        good = MagicMock()
        ok = MagicMock()
        ok.json = lambda: {"choices": [{"message": {"content": "FROM_NODE"}}]}
        ok.raise_for_status = lambda: None
        good.post = AsyncMock(return_value=ok)
        c.critic_clients = [{"url": "http://n", "model": "cm",
                             "client": good, "name": "Nova"}]
        c._critic_index = 0
        res = asyncio.run(c._do_chat_completion(
            {"model": "m", "messages": []}, use_critic=True, timeout=120.0))
        assert served_leg(res)["served_by"] == "critic"
        assert served_leg(res)["fell_back_from"] == ""

    def test_the_main_fallback_is_bounded(self):
        import asyncio
        from ghost_agent.core.llm import _MAIN_FALLBACK_TIMEOUT_S
        cap = {}
        c = self._client(cap)
        asyncio.run(c._do_chat_completion(
            {"model": "m", "messages": []}, use_critic=True, timeout=120.0))
        assert cap["timeout"] == _MAIN_FALLBACK_TIMEOUT_S
        assert 0 < cap["timeout"] < 1200.0, (
            "the fallback runs unbounded on the foreground slot again")


class TestOffMainOnlyNeverClaimsAMainFallback:
    """`off_main_only` callers never touch the 35B, so the "falling back to
    main upstream" line was simply false — keepalive alone emitted it every
    45s. The worker branch fixed this; critic/coding/swarm kept it."""

    def test_the_three_other_pools_gate_the_claim(self):
        import inspect
        from ghost_agent.core.llm import LLMClient
        src = inspect.getsource(LLMClient._do_chat_completion)
        lines = src.splitlines()
        for label in ("Critic", "Coding", "Edge"):
            idx = next(n for n, ln in enumerate(lines)
                       if f'"{label} Compute Failed"' in ln)
            # The gate sits immediately above the claim; a 700-char window
            # reached back into the previous branch and passed on ANY pool
            # being gated. Look only at the lines that guard THIS log call.
            # Assert on the LINE immediately guarding the claim. A byte
            # window either reaches into the previous branch (passing on
            # someone else's gate) or lands mid-call, depending on how the
            # message happens to wrap — two wrong versions of this assertion
            # preceded this one.
            preceding = lines[idx - 1].strip()
            assert preceding.startswith("elif"), (
                f"the {label} main-fallback claim is not gated: {preceding!r}")
            assert f'"{label} Nodes Unavailable"' in src, (
                f"the {label} branch has no honest off-main-only line")


def test_the_leg_stamp_cannot_reach_an_API_CLIENT():
    """`_stamp_leg` adds a private key to the response dict. That is safe
    ONLY because the OpenAI-compatible payload is constructed field by field
    (`api/routes.py`: id/object/created/model/choices/message/done), never
    passed through — and three tests comparing whole dicts caught the
    assumption when it was stated as "non-invasive by construction".

    Pin the property that makes it safe, so a future "just return the
    upstream dict" refactor fails here instead of leaking internals."""
    import inspect
    from ghost_agent.api import routes

    src = inspect.getsource(routes)
    i = src.index('"object": "chat.completion"')
    block = src[max(0, i - 400):i + 400]
    assert '"choices": [{' in block, (
        "the OpenAI payload is no longer constructed literally — if it now "
        "forwards an upstream dict, `_ghost_leg` leaks to API clients")


class TestTheBreakerRefusesOnlyForCallersThatOptIn:
    """DECISION, 2026-08-18. With one node per pool — the shipping topology —
    every selector's `return pool[0]` made the circuit breaker unable to
    prevent a single request: its only effects were a log line and
    /api/health. Making it refuse globally was rejected, and the reasoning
    matters more than the code: a refusing pool sends its traffic to the main
    model, turning a sick node into a foreground dogpile. So refusal is a
    per-caller opt-in: callers whose fallback is FREE ask for a healthy node
    and accept None.

    ⚠ The original version of this docstring justified that decision with
    "`keepalive` IS the recovery detector, so a pool that refuses can never
    observe a node healing." That is false, and it was repeated verbatim at
    five sites in `llm.py`. `is_available` promotes open -> half_open after
    the cooldown and returns True to every caller, `require_healthy`
    included; any success re-closes the breaker. Refusal delays recovery by
    at most one cooldown — it cannot prevent it. The decision survives on
    the dogpile argument alone; it never needed the false one (R4 lens B)."""

    def _tripped_client(self):
        from unittest.mock import AsyncMock, MagicMock
        c = LLMClient("http://upstream.invalid:8080")
        node = MagicMock()
        node.post = AsyncMock(side_effect=RuntimeError("node down"))
        c.worker_clients = [{"url": "http://n", "model": "wm",
                             "client": node, "name": "Nova"}]
        c._worker_index = 0
        for _ in range(5):
            c.circuit_breaker.record_failure("http://n")
        assert not c.circuit_breaker.is_available("http://n")
        return c, node

    def test_route_refuses_a_tripped_node_and_uses_its_free_fallback(self):
        import asyncio
        from ghost_agent.core.llm import RoutingTask

        c, node = self._tripped_client()
        out = asyncio.run(c.route("CLASSIFY_INTENT",
                                  {"model": "m", "messages": []},
                                  fallback="FREE"))
        assert out == "FREE"
        assert node.post.await_count == 0, (
            "route() contacted a node the breaker had tripped — it has a free "
            "fallback and should not spend a slot wait on a sick node")

    def test_keepalive_STILL_probes_a_tripped_node(self):
        """The other half, and the reason refusal is not global: if the
        health probe refuses too, nothing ever observes recovery."""
        import asyncio
        from ghost_agent.core.llm import OffMainNodeUnavailable

        c, node = self._tripped_client()
        try:
            asyncio.run(c._do_chat_completion(
                {"model": "m", "messages": []}, use_worker=True,
                off_main_only=True, timeout=5.0, task_label="keepalive"))
        except OffMainNodeUnavailable:
            pass
        assert node.post.await_count >= 1, (
            "keepalive did not probe the tripped node — recovery becomes "
            "unobservable and the pool stays down forever")

    def test_a_refused_pool_still_honours_off_main_only(self):
        """The trap under this change: `fell_back_from_node` used to be set
        only INSIDE `if node:`, so a selector returning None left it False —
        `off_main_only` was never consulted and the timeout was stripped to
        httpx's 1200s default (R2 lens B, NEW-4)."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from ghost_agent.core.llm import OffMainNodeUnavailable

        c, node = self._tripped_client()
        main = MagicMock()
        main.post = AsyncMock(return_value=MagicMock(
            json=lambda: {"choices": [{"message": {"content": "MAIN"}}]},
            raise_for_status=lambda: None, status_code=200, text="{}"))
        c.http_client = main
        with pytest.raises(OffMainNodeUnavailable):
            asyncio.run(c._do_chat_completion(
                {"model": "m", "messages": []}, use_worker=True,
                off_main_only=True, timeout=5.0, require_healthy=True))
        assert main.post.await_count == 0, (
            "a refused pool fell through to the MAIN model despite "
            "off_main_only — the 1200s foreground dogpile this decision "
            "exists to avoid")


class TestTheSeamStopsAskingForPoolsThatDoNotExist:
    """LLM review R3 lens B. The seam's flags express INTENT; when the pool
    is absent the intent is silently served by the main model — foreground,
    and (before R2/R3) unbounded, because neither timeout-clamp arm fires
    when the pool is unconfigured AND the caller passed no timeout."""

    def test_the_planner_only_requests_a_swarm_pool_that_exists(self):
        """⚠ BEHAVIOURAL. The first version asserted `"use_swarm=True)" not
        in src` — a pin on a SPELLING, which any same-meaning edit
        (`use_swarm=True,`, a reordered kwarg, a wrapped line) walks straight
        through while reintroducing the defect (R4 lens A, MINOR-8)."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from ghost_agent.core.llm import LLMClient

        c = LLMClient("http://main.invalid:8088")
        c.swarm_clients = []                    # production: no swarm pool
        seen = {}

        async def _main(path, **kw):
            seen.update(kw)
            r = MagicMock()
            r.json = lambda: {"choices": [{"message": {"content": "{}"}}]}
            r.raise_for_status = lambda: None
            r.status_code, r.text = 200, "{}"
            return r

        c.http_client = MagicMock()
        c.http_client.post = AsyncMock(side_effect=_main)

        from ghost_agent.core.agent import _PLANNER_TIMEOUT_S
        asyncio.run(c.chat_completion(
            {"model": "m", "messages": []},
            use_swarm=bool(getattr(c, "swarm_clients", None)),
            timeout=_PLANNER_TIMEOUT_S, task_label="planner"))
        t = seen.get("timeout")
        assert isinstance(t, (int, float)) and t > 0, (
            f"the planner's call reached main with timeout={t!r} — its "
            f"highest-leverage call would run unbounded (httpx: 1200s)")

        # And the seam must still derive the flag from the pool, not hardcode
        # it. This half stays structural because it is about the CALLER.
        import inspect
        from ghost_agent.core import agent as agent_mod
        src = inspect.getsource(agent_mod)
        i = src.index("planning_payload, use_swarm=")
        flag = src[i + len("planning_payload, use_swarm="):i + 300].split(",")[0]
        assert "True" not in flag, (
            f"the planner asks for a swarm pool unconditionally "
            f"(use_swarm={flag.strip()}); production has none, so the request "
            f"is served by MAIN while reading as delegated")

    def test_the_web_summary_has_a_free_fallback_and_uses_it(self):
        """⚠ PARSED, not window-sliced. This read a fixed 700-char window
        before the task_label and broke the moment a comment was added above
        the call — a pure maintenance tax that teaches people to widen the
        window rather than fix anything (R5 lens C). Parse the call's kwargs
        instead, so it tracks meaning and not layout."""
        import ast
        import inspect
        from ghost_agent.tools import search as search_mod
        from ghost_agent.tools import darkweb_search as dark_mod

        for mod in (search_mod, dark_mod):
            tree = ast.parse(inspect.getsource(mod))
            calls = [n for n in ast.walk(tree)
                     if isinstance(n, ast.Call)
                     and any(k.arg == "task_label"
                             and getattr(k.value, "value", None) == "web summary"
                             for k in n.keywords)]
            assert calls, f"{mod.__name__}: the web-summary call is gone"
            for call in calls:
                kw = {k.arg: k.value for k in call.keywords}
                assert getattr(kw.get("off_main_only"), "value", False) is True, (
                    f"{mod.__name__}: a worker outage sends every url's "
                    f"distillation to the main 35B, foreground — while the "
                    f"raw-text fallback sits in the except block below")
                assert "timeout" in kw, (
                    f"{mod.__name__}: the per-url summary is unbounded")
                assert "slot_wait" in kw, (
                    f"{mod.__name__}: the summary inherits the 90s operator "
                    f"ceiling, which exceeds the 55s outer per-url deadline — "
                    f"a saturated node LOSES THE URL instead of degrading to "
                    f"raw page text")


class TestBackgroundCallersDoNotDogpileTheMainSlot:
    def test_the_constraint_gate_background_path_is_off_main(self):
        """§4O A-MAJOR-2's sweep missed this one: idle autoadvance reaches
        `constraint_gate` with is_background=True, and because the worker box
        is not the main URL the call skipped BOTH gates, then fell back to
        main anyway — an unqueued 300s generation on the single slot while a
        user request was live (R3 lens B, NEW-2)."""
        import inspect
        from ghost_agent.core import build_gates

        src = inspect.getsource(build_gates.constraint_gate)
        assert "off_main_only=is_background" in src, (
            "a background constraint gate can still dogpile the main slot")

    def test_the_critic_verify_leg_is_background_aware(self):
        """`_bounded_fallback_kwargs` exists for exactly this, and was applied
        only to the last-resort MAIN call — while the CRITIC leg, the one that
        fires in production, inflated `foreground_tasks` for up to 120s and
        blanked the biological tick, self-play and the RSS gate
        (R3 lens B, NEW-1)."""
        import inspect
        from ghost_agent.core import verifier as v

        src = inspect.getsource(v.Verifier._call_llm)
        i = src.index("use_critic=True")
        window = src[max(0, i - 900):i + 300]
        assert "_bounded_fallback_kwargs" in window, (
            "the critic leg still reports itself as foreground main work")


def test_require_healthy_and_the_fallback_flag_reach_EVERY_pool():
    """⚠ THE SAME INCOMPLETENESS, TWICE. R2 added `require_healthy` and the
    `if node is None: fell_back_from_node = True` guard to worker, critic and
    coding — and missed vision and swarm, the two branches its own explanatory
    comment did not sit next to. Both R3 lenses found it independently.

    Structural, deliberately: a sixth pool must not be able to repeat this."""
    import inspect
    from ghost_agent.core.llm import LLMClient

    import re
    src = inspect.getsource(LLMClient._do_chat_completion)
    for sel in ("get_vision_node", "get_worker_node", "get_critic_node",
                "get_coding_node", "get_swarm_node"):
        hits = [m.start() for m in re.finditer(re.escape(f"self.{sel}("), src)]
        assert len(hits) >= 3, (
            f"{sel} is called {len(hits)}x — the branch was restructured; "
            f"re-derive what this test is checking rather than relaxing it")
        for n, i in enumerate(hits):
            # ⚠ EVERY occurrence, not src.index(). The first version of this
            # test read only the FIRST call per branch — the one that already
            # passed the flag — and was structurally blind to the two-to-three
            # in-loop re-selects that follow it. R4 lens B proved a
            # require_healthy=True caller POSTing to an OPEN-breaker node, and
            # the patch fixing all 15 re-selects SURVIVED 158 tests including
            # this one.
            assert "require_healthy=require_healthy" in src[i:i + 200], (
                f"{sel} call #{n + 1} ignores require_healthy — an in-loop "
                f"re-select can hand back a node whose breaker is OPEN")
        after = src[hits[0]:hits[0] + 700]
        # ⚠ For `get_vision_node` this assertion is STRUCTURAL ONLY, and the
        # comment beside it used to claim otherwise. The vision branch always
        # returns or raises before reaching the shared fallback block, so its
        # `fell_back_from_node = True` cannot change behaviour today — it is
        # kept for uniformity, so that a sixth pool copied from any of the
        # five gets the working version (R4 lens A, MINOR-7). Do not cite it
        # as evidence that the vision path is guarded; it is guarded by
        # raising instead.
        assert "fell_back_from_node = True" in after, (
            f"{sel} returning None leaves fell_back_from_node False — "
            f"off_main_only is skipped and the timeout is stripped")


def test_a_require_healthy_caller_never_POSTs_to_an_open_breaker_node():
    """The behavioural half of the test above — it fails on the real defect,
    which no text assertion could see (R4 lens B, item 3).

    Two nodes, both breakers OPEN. The first select honours require_healthy
    and returns None; the buggy in-loop re-select called the selector bare,
    hit its `return pool[0]` last-resort, and POSTed to a node we had just
    decided was sick. One node per pool hid it live — `pool[0]` was always
    already in `tried_nodes`."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.llm import LLMClient

    c = LLMClient("http://main.invalid:8088")
    posted = []

    def _node(url):
        cl = MagicMock()

        async def _post(path, **kw):
            posted.append(url)
            raise RuntimeError("node down")

        cl.post = AsyncMock(side_effect=_post)
        return {"url": url, "model": "m", "client": cl, "name": url}

    # ⚠ THIS EXACT STATE IS LOAD-BEARING — two earlier versions of this test
    # passed identically on the buggy and the fixed build:
    #   * both breakers open  -> the FIRST select returns None, so the retry
    #     loop is never entered at all;
    #   * only B open         -> the bare re-select filters to the healthy
    #     list and hands back A, never reaching the last resort.
    # `require_healthy` only governs the selector's `return pool[0]` LAST
    # RESORT, so the defect needs every node unavailable AND pool[0] not yet
    # tried. B is sick from the start; A is one failure short and trips on
    # its own in-call failure. That is the live shape: a pool degrading
    # DURING a call, which is when a retry loop matters.
    c.worker_clients = [_node("http://B"), _node("http://A")]
    c._worker_index = 0
    for _ in range(5):
        c.circuit_breaker.record_failure("http://B")
    for _ in range(2):                      # threshold is 3
        c.circuit_breaker.record_failure("http://A")
    assert c.circuit_breaker.is_available("http://A")
    assert not c.circuit_breaker.is_available("http://B")

    async def _main(*a, **kw):
        r = MagicMock()
        r.json = lambda: {"choices": [{"message": {"content": "ok"}}]}
        r.raise_for_status = lambda: None
        r.status_code, r.text = 200, "{}"
        return r

    c.http_client = MagicMock()
    c.http_client.post = AsyncMock(side_effect=_main)
    asyncio.run(c._do_chat_completion(
        {"model": "m", "messages": []}, use_worker=True,
        require_healthy=True, timeout=5))
    assert "http://B" not in posted, (
        f"a require_healthy=True caller POSTed to the OPEN-breaker node B "
        f"on the in-loop re-select (posted={posted}) — the first select "
        f"honoured the opt-in and every re-select threw it away. The whole "
        f"point of the opt-in is that a caller with a free fallback does not "
        f"spend a request on a node it has just decided is sick.")
    assert posted == ["http://A"], (
        f"expected exactly one attempt, on the one healthy node: {posted}")


class TestThePermitBudgetIsATotalDeadline:
    """R6. Two budgets, deliberately separate, driven through the REAL gate.

    ⚠ THIS CLASS WAS REWRITTEN BECAUSE ITS PREDECESSOR WAS AN INSTRUMENT
    THAT GUARANTEED ITS OWN RESULT. It replaced `_node_slot` with a spy that
    yielded unconditionally for the designated "free" node, ignoring the
    `wait_timeout` it was handed — so it asserted a property the harness
    supplied rather than one the code implemented. Five mutations against
    the mechanism it existed to protect survived it (R6 lens C): zeroing
    `_MIN_HTTP_FLOOR`, flattening the share, dropping the reserve, and
    removing both halves of the HTTP clip. Worse, the defect the harness hid
    was live: the last node of every pool was handed a 0.0s budget, and
    `asyncio.wait_for(sem.acquire(), 0.0)` rejects even a completely FREE
    semaphore — so a free node was never asked, which is the exact failure
    the sharing exists to prevent.

    Everything below holds real permits on a real `asyncio.Semaphore`.
    `GHOST_NODE_SLOT_WAIT_S` is lowered so the suite stays fast; it is the
    ceiling, so it also bounds the caller-supplied floor.
    """

    @staticmethod
    def _client(pool_size=1, post_delay=0.02):
        from unittest.mock import AsyncMock, MagicMock
        from ghost_agent.core.llm import LLMClient
        import asyncio as _a

        c = LLMClient("http://main.invalid:8088")
        posts = []

        async def _post(path, **kw):
            # ⚠ HONOUR THE TIMEOUT, as httpx does. A mock that sleeps for a
            # fixed time regardless of its budget cannot show that clipping
            # the budget changed anything — which is why dropping route()'s
            # `total_budget` survived even with a slow POST (R7 lens C, M46).
            budget = kw.get("timeout")
            posts.append(budget)
            if budget is not None and post_delay > budget:
                await _a.sleep(budget)
                raise httpx.ReadTimeout("")
            await _a.sleep(post_delay)
            r = MagicMock()
            r.json = lambda: {"choices": [{"message": {"content": "node"}}]}
            r.raise_for_status = lambda: None
            r.status_code, r.text = 200, "{}"
            return r

        c.worker_clients = []
        for i in range(pool_size):
            cl = MagicMock()
            cl.post = AsyncMock(side_effect=_post)
            props = MagicMock()
            props.json = lambda: {"total_slots": 1}
            props.raise_for_status = lambda: None
            cl.get = AsyncMock(return_value=props)
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

    def _drive(self, pool_size=1, busy=(), hold_for=30.0, **kw):
        """Run one call with `busy` node indices genuinely saturated.

        Returns (elapsed, served_by, post_budgets, posted_urls).
        """
        import asyncio
        import time as _t

        c, posts = self._client(pool_size)

        async def _run():
            for n in c.worker_clients:
                await c._node_capacity(n)
            held = []
            for idx in busy:
                node = c.worker_clients[idx]
                cm = c._node_slot(node, wait_timeout=30)
                await cm.__aenter__()
                held.append(cm)
            t0 = _t.monotonic()
            res = await c._do_chat_completion(
                {"model": "m", "messages": []}, use_worker=True, **kw)
            dt = _t.monotonic() - t0
            for cm in held:
                await cm.__aexit__(None, None, None)
            return dt, res

        dt, res = asyncio.run(_run())
        served = res["choices"][0]["message"]["content"]
        posted = [n["url"] for n in c.worker_clients
                  if n["client"].post.await_count]
        return dt, served, posts, posted

    # ---- the free node must always be reachable -------------------------

    @pytest.mark.parametrize("pool_size", [1, 2, 3, 4])
    def test_the_only_free_node_is_reached_at_every_pool_size(
            self, pool_size, monkeypatch):
        """⚠ THE LIVE DEFECT R5 SHIPPED. `_untried` was computed AFTER the
        node was appended to `tried_nodes`, so it counted the attempts
        remaining AFTER this one and the last node's share drained to
        exactly 0.0 — which the gate refuses even when every permit is free.
        Budgets were `[9.0, 0.0]` at 2 nodes, `[4.5, 4.5, 0.0]` at 3."""
        monkeypatch.setenv("GHOST_NODE_SLOT_WAIT_S", "3")
        busy = tuple(range(pool_size - 1))       # everything but the last
        dt, served, posts, posted = self._drive(
            pool_size=pool_size, busy=busy,
            timeout=12, slot_wait=12, total_budget=12)
        assert served == "node", (
            f"pool={pool_size}: the one FREE node was never asked "
            f"(served_by={served!r}, posted={posted}) — every permit on it "
            f"was available")
        assert posted == [f"http://N{pool_size - 1}"], posted

    def _drive_freeing(self, pool_size, free_at, **kw):
        """Like `_drive`, but node 0 is busy and frees after `free_at`s
        while every later node is free from the start."""
        import asyncio
        import time as _t

        c, posts = self._client(pool_size)

        async def _run():
            for n in c.worker_clients:
                await c._node_capacity(n)
            cm = c._node_slot(c.worker_clients[0], wait_timeout=30)
            await cm.__aenter__()

            async def _free():
                await asyncio.sleep(free_at)
                await cm.__aexit__(None, None, None)

            h = asyncio.ensure_future(_free())
            t0 = _t.monotonic()
            res = await c._do_chat_completion(
                {"model": "m", "messages": []}, use_worker=True, **kw)
            dt = _t.monotonic() - t0
            h.cancel()
            return dt, res

        dt, res = asyncio.run(_run())
        posted = [n["url"] for n in c.worker_clients
                  if n["client"].post.await_count]
        return dt, res["choices"][0]["message"]["content"], posts, posted

    def test_a_busy_first_node_does_not_consume_the_whole_budget(self):
        """⚠ THIS IS WHAT THE SHARING BUYS, and the only property that can
        tell it from a flat per-attempt deadline.

        The obvious test — "a free node is still reached" — cannot: the
        `_MIN_ACQUIRE` floor means even a 0.05s window acquires a genuinely
        free permit, so the free node is reached either way and the flat
        mutant survives. What a flat budget really costs is that node 0 eats
        everything while a free node sits behind it in the list.

        Node 0 is busy until 5s; node 1 is free throughout. Shared: node 0
        gets (9-3)/2 = 3s, gives up, node 1 answers at ~3s. Flat: node 0
        gets the whole 6s, frees at 5s, and answers there — 2s later, having
        ignored an idle node the entire time."""
        dt, served, _, posted = self._drive_freeing(
            pool_size=2, free_at=5.0, timeout=12, slot_wait=9, total_budget=9)
        assert served == "node", f"served_by={served!r}"
        assert posted == ["http://N1"], (
            f"answered from {posted} — node 0 was busy the whole time and an "
            f"idle node 1 was available")
        assert dt <= 4.0, (
            f"took {dt:.2f}s to reach an idle node because the busy first "
            f"node was allowed to consume the entire budget")

    def test_a_permit_is_not_spent_on_a_request_that_cannot_finish(self):
        """⚠ THE RESERVE, tested where it actually binds. Asserting the POST
        budget on a deadline with plenty left cannot see this — the clip
        never reaches the floor. Here the permit only frees once the total is
        nearly gone, so honouring the reserve means declining the node.

        With the reserve: we stop queueing at 6-3=3s and fall to main having
        never touched the node. Without it (`_MIN_HTTP_FLOOR = 0.0`): we
        queue the full 6s, acquire at 5.9s, and POST with 0.1s — spending a
        scarce permit on a request that provably cannot complete, then
        charging the resulting ReadTimeout to the node as a fault."""
        dt, served, posts, posted = self._drive_freeing(
            pool_size=1, free_at=5.9, timeout=12, slot_wait=6, total_budget=6)
        if posted:
            assert posts and posts[0] >= 1.0, (
                f"POSTed to the node with only {posts[0]}s of budget left")
        else:
            assert served == "MAIN", f"served_by={served!r}"
            assert dt <= 4.5, (
                f"queued {dt:.2f}s of a 6s total before giving up — the "
                f"reserve for the request itself was not held back")

    def test_a_stated_total_is_not_exceeded(self):
        """route()'s contract: queueing AND the request inside one budget."""
        dt, served, posts, _ = self._drive(
            pool_size=1, busy=(0,), timeout=12, slot_wait=12, total_budget=6)
        assert dt <= 6.5, f"a stated 6s total took {dt:.2f}s"
        assert served == "MAIN", "expected the free fallback after the budget"

    def test_the_POST_budget_is_clipped_by_TIME_ALREADY_SPENT(self):
        """⚠ R5 computed this on the line ABOVE the `async with`, i.e. at
        t=0 before any queueing, so it never clipped anything: the identity
        mutant `return t` survived 112 tests. Queue first, then look."""
        import asyncio
        import time as _t
        from unittest.mock import AsyncMock, MagicMock

        c, posts = self._client(pool_size=1)

        async def _run():
            node = c.worker_clients[0]
            await c._node_capacity(node)
            cm = c._node_slot(node, wait_timeout=30)
            await cm.__aenter__()

            async def _free():
                await asyncio.sleep(2.0)
                await cm.__aexit__(None, None, None)

            asyncio.ensure_future(_free())
            await c._do_chat_completion(
                {"model": "m", "messages": []}, use_worker=True,
                timeout=12, slot_wait=12, total_budget=12)

        asyncio.run(_run())
        assert posts and posts[0] is not None
        assert posts[0] < 11.0, (
            f"the POST got {posts[0]}s after ~2s of queueing against a 12s "
            f"total — the clip is being computed before the wait, so the "
            f"two budgets still add")
        assert posts[0] >= 3.0, (
            f"the POST got {posts[0]}s — below the reserve, i.e. a permit "
            f"spent on a request that cannot finish")

    def test_a_caller_that_states_no_total_keeps_its_full_budget(self):
        """⚠ THE LIVE REGRESSION R5 SHIPPED. Collapsing the two budgets cut
        the verifier's critic call from 120s to 30s and dream's from 180s to
        90s. Against the live distribution (n=39: median 24.4s, p90 56.7s)
        that failed 28.2% of verdicts AND charged each to the node as a
        fault. `GHOST_NODE_SLOT_WAIT_S` is a QUEUE budget; it must never
        become the maximum length of a generation."""
        _, _, posts, _ = self._drive(pool_size=1, timeout=180.0)
        assert posts == [180.0], (
            f"a caller asking for 180s of generation got {posts} — the "
            f"queue ceiling is being applied to the request")

    def test_no_total_still_bounds_the_QUEUE(self):
        """The half of R5 that was right: the wait must not be re-spent per
        node (measured 12/24/36s at 1/2/3 nodes against a stated 12s).

        ⚠ Asserted as a RATIO across pool sizes, not against an absolute.
        The first version compared a 3-node run against the literal
        `slot_wait` it passed and failed on the boundary — because a
        caller-supplied budget is floored at `_MIN_SLOT_WAIT +
        _MIN_HTTP_FLOOR`, so asking for 6 yields 8. The property has nothing
        to do with the absolute value: adding nodes to a pool must not make
        a caller's stated deadline less true."""
        one, served1, _, _ = self._drive(
            pool_size=1, busy=(0,), timeout=180.0, slot_wait=12)
        three, served3, _, _ = self._drive(
            pool_size=3, busy=(0, 1, 2), timeout=180.0, slot_wait=12)
        assert served1 == served3 == "MAIN"
        assert three <= one * 1.3, (
            f"1 node queued {one:.2f}s but 3 nodes queued {three:.2f}s "
            f"({three / max(one, 0.01):.1f}x) — the budget is being re-spent "
            f"per node, so every node added makes the deadline less true")

    def test_the_env_var_remains_the_ceiling(self, monkeypatch):
        monkeypatch.setenv("GHOST_NODE_SLOT_WAIT_S", "2")
        dt, served, _, _ = self._drive(
            pool_size=1, busy=(0,), timeout=180.0, slot_wait=90)
        assert dt <= 3.5, (
            f"a caller asked for 90s of queueing under a 2s operator "
            f"ceiling and got {dt:.2f}s")

    def test_a_zero_or_negative_request_still_leaves_REAL_queue_time(self):
        """Asserted against a LITERAL. Recomputing the bound from the
        constant under test makes it unfalsifiable — setting the constant to
        0.0 then satisfies it while the gate is disabled."""
        for bad in (0, 0.0, -5, -0.001):
            dt, served, _, _ = self._drive(
                pool_size=1, busy=(0,), timeout=12, slot_wait=bad)
            # ⚠ 7.0, not 4.0. The shipped floor is `_MIN_SLOT_WAIT +
            # _MIN_HTTP_FLOOR` = 8.0, so a 4.0 band still passed when the
            # reserve was dropped from the floor, leaving 5.0s. Third round
            # running that this literal has been too loose (R7 lens C, M11).
            assert dt >= 7.0, (
                f"slot_wait={bad!r} queued for only {dt:.2f}s — every node "
                f"reports SATURATED instantly and the whole load lands on "
                f"the single main slot")

    def test_keepalive_still_bypasses_the_gate_entirely(self):
        """The health probe must never queue behind the traffic it watches,
        or a BUSY node reports as a DEAD one."""
        dt, served, _, posted = self._drive(
            pool_size=1, busy=(0,), timeout=30, slot_wait=30,
            task_label="keepalive")
        assert served == "node" and posted == ["http://N0"], (
            f"the keepalive probe did not reach a saturated node "
            f"(served={served!r}) — it must bypass the gate")
        assert dt < 1.0, f"the probe queued for {dt:.2f}s"

    def test_route_stays_inside_its_own_contract(self):
        """route()'s fallback is a string concat on the user's critical
        path, so every second past its budget is pure dead air."""
        import asyncio
        import time as _t
        from ghost_agent.core.llm import _ROUTE_TIMEOUT_S

        # ⚠ The POST must be SLOW, or queue and request cannot ADD and
        # dropping route()'s `total_budget` leaves this test green
        # (R7 lens C, M46).
        c, _ = self._client(pool_size=1, post_delay=9.0)

        async def _run():
            node = c.worker_clients[0]
            await c._node_capacity(node)
            cm = c._node_slot(node, wait_timeout=60)
            await cm.__aenter__()

            async def _free():
                await asyncio.sleep(8.0)      # frees PART WAY THROUGH
                await cm.__aexit__(None, None, None)

            h = asyncio.ensure_future(_free())
            t0 = _t.monotonic()
            out = await c.route("EXPAND_QUERY", {"model": "m", "messages": []},
                                fallback="<<FREE FALLBACK>>")
            dt = _t.monotonic() - t0
            h.cancel()
            return dt, out

        dt, out = asyncio.run(_run())
        assert dt <= _ROUTE_TIMEOUT_S + 0.8, (
            f"route() states a {_ROUTE_TIMEOUT_S}s fail-fast and took "
            f"{dt:.2f}s. ⚠ An earlier version of this test held the permit "
            f"for the WHOLE call, so it only ever exercised the row that "
            f"already passed; the permit must free part way through, which "
            f"is when queue and request add.")

def test_a_refusing_pool_can_still_observe_a_node_healing():
    """Pins the CORRECTION, so the false rationale cannot come back as a
    fix. Six places in this repo argued that `require_healthy` had to be
    opt-in because a refusing pool could never see a node recover. Refusal
    is bounded by the cooldown, and it is bounded for the opting-in caller
    too — which is exactly what makes the opt-in safe."""
    from ghost_agent.core.llm import LLMClient, NodeCircuitBreaker

    c = LLMClient("http://main.invalid:8088")
    c.worker_clients = [{"url": "http://W", "model": "m",
                         "client": None, "name": "W"}]
    c.circuit_breaker = NodeCircuitBreaker(failure_threshold=3,
                                           cooldown_seconds=0.0)
    for _ in range(5):
        c.circuit_breaker.record_failure("http://W")
    assert c.circuit_breaker.get_status()["http://W"]["state"] == "open"
    # cooldown_seconds=0 -> the very next check is past the cooldown
    assert c.get_worker_node(None, require_healthy=True) is not None, (
        "a require_healthy caller is locked out of a node the breaker has "
        "already promoted to half_open — THAT would make the false rationale "
        "true, and would starve the pool permanently")


def test_half_open_admits_every_caller_not_one_probe():
    """The other corrected claim. Pinned as-is rather than fixed: see the
    note at the half_open branch for why a bare single-flight bool is worse
    than the fan-out it prevents. If someone implements a real expiring
    probe token, this test SHOULD fail — read that note first."""
    from ghost_agent.core.llm import NodeCircuitBreaker

    b = NodeCircuitBreaker(failure_threshold=3, cooldown_seconds=0.0)
    for _ in range(3):
        b.record_failure("http://N")
    admitted = sum(1 for _ in range(50) if b.is_available("http://N"))
    assert admitted == 50, (
        f"half_open admitted {admitted}/50 — if this is now a real "
        f"single-flight probe, delete this test and the note beside it")
    assert b.get_status()["http://N"]["state"] == "half_open"
