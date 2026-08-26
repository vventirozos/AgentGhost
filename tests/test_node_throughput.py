"""Throughput-derived sizing for off-main work (2026-08-25, req 08766aa1).

`deep_research` posted a flat 40,000-char / `max_tokens=2048` distillation on
a flat 45s budget. On Nova that is ~12,500 prompt tokens at ~300 tok/s
prefill — 41s before the first output token — so it could not finish even
with the node idle and a slot free. Measured: 42.7s solo, 135/258/258s at the
live concurrency, 1 success against 7 degradations in the log window. The
guard that was supposed to prevent it read `args.max_context` (the MAIN
model's 240,000), so it pinned to its own 40k ceiling and never bound.

These pin the replacement: rates learned from llama.cpp `timings`, both knobs
derived from them, and an explicit refusal when the budget cannot buy a
useful answer.
"""
import pytest

from ghost_agent.core.node_throughput import (
    CHARS_PER_TOKEN,
    MAX_CHARS,
    MIN_CHARS,
    MIN_TOKENS,
    MIN_PREFILL_SAMPLE_TOKENS,
    DistillPlan,
    NodeThroughput,
    env_float,
)

URL = "http://nova:8088"

# Rates measured on the live box 2026-08-25 (see the module docstring).
NOVA_SOLO = (304.0, 31.6)
NOVA_3WAY = (220.0, 12.0)


def _timings(p_n, p_ms, d_n, d_ms):
    return {"timings": {"prompt_n": p_n, "prompt_ms": p_ms,
                        "predicted_n": d_n, "predicted_ms": d_ms}}


def _teach(nt, url, prefill, decode):
    """Pin the estimator to a known regime without going through EWMA."""
    nt._rates[url] = [prefill, decode, 5]


class TestSamplingGuards:
    """A small sample is not a slow node — the trap that would invert this."""

    def test_keepalive_ping_cannot_teach(self):
        # The live 45s heartbeat: `max_tokens=1`, content "ok". llama.cpp
        # reports 13-25 tok/s for it on a node that really does 300, because
        # the rate is dominated by fixed per-request overhead. Learning from
        # it would collapse every subsequent plan.
        nt = NodeThroughput()
        assert nt.observe(URL, _timings(1, 70, 1, 30)) is False
        assert nt.rates(URL) == (nt.default_prefill_tok_s,
                                 nt.default_decode_tok_s, 0)

    def test_prompt_cache_hit_cannot_teach(self):
        # Measured shape: 4 prompt tokens in 6.2s (~1 tok/s) because the count
        # is the cache MISS while the clock is the whole wait.
        nt = NodeThroughput()
        assert nt.observe(URL, _timings(4, 6200, 0, 0)) is False
        assert nt.rates(URL)[2] == 0

    def test_a_real_prefill_teaches_the_measured_rate(self):
        nt = NodeThroughput()
        assert nt.observe(URL, _timings(12512, 41100, 40, 1300)) is True
        prefill, decode, samples = nt.rates(URL)
        assert prefill == pytest.approx(304, abs=2)   # the live solo number
        assert decode == pytest.approx(30.8, abs=2)
        assert samples == 1

    def test_decode_can_be_learned_without_prefill(self):
        # A cache-hit request teaches decode only; the prefill side must fall
        # back independently rather than dragging in a garbage rate.
        nt = NodeThroughput()
        assert nt.observe(URL, _timings(4, 6200, 200, 10000)) is True
        prefill, decode, _ = nt.rates(URL)
        assert prefill == nt.default_prefill_tok_s
        assert decode == pytest.approx(20.0, abs=0.1)

    def test_ewma_tracks_a_regime_change(self):
        # Idle node, then 3-way contention: the estimate must move toward the
        # slower rate rather than staying pinned to the optimistic one.
        nt = NodeThroughput()
        nt.observe(URL, _timings(12000, 40000, 500, 16000))   # 300 / 31
        before = nt.rates(URL)[0]
        for _ in range(6):
            nt.observe(URL, _timings(12000, 60000, 500, 42000))  # 200 / 12
        after = nt.rates(URL)[0]
        assert after < before
        # EWMA(0.3) closes 1 - 0.7**6 = 88% of the gap in six samples, so the
        # assertion is "tracked the new regime", not "reached it exactly".
        assert abs(after - 200) < abs(after - 300)
        assert after == pytest.approx(200, rel=0.10)

    def test_garbage_and_missing_timings_are_ignored(self):
        nt = NodeThroughput()
        for bad in ({}, {"timings": None}, {"timings": {}},
                    {"timings": {"prompt_n": "x", "prompt_ms": 1}},
                    {"timings": {"prompt_n": 9999, "prompt_ms": 0}},
                    None, "not-a-dict"):
            assert nt.observe(URL, bad) is False
        assert nt.observe("", _timings(9999, 1000, 99, 1000)) is False


class TestPlanFitsTheBudget:
    """The invariant: never ask for what the node cannot finish."""

    @pytest.mark.parametrize("prefill,decode", [
        NOVA_SOLO, NOVA_3WAY, (150.0, 10.0), (1000.0, 80.0), (60.0, 4.0)])
    @pytest.mark.parametrize("budget", [30.0, 45.0, 90.0, 300.0])
    def test_predicted_cost_never_exceeds_the_budget(self, prefill, decode,
                                                     budget):
        nt = NodeThroughput()
        _teach(nt, URL, prefill, decode)
        plan = nt.plan(budget, URL)
        if not plan.feasible:
            return
        # Recomputed from the plan's own numbers, not from `predicted_s` —
        # a field that agrees with itself proves nothing.
        cost = ((plan.char_limit / CHARS_PER_TOKEN) / prefill
                + plan.max_tokens / decode)
        assert cost <= budget, f"{plan.describe()} costs {cost:.1f}s"

    def test_a_free_node_is_sized_larger_than_a_contended_one(self):
        idle, busy = NodeThroughput(), NodeThroughput()
        _teach(idle, URL, *NOVA_SOLO)
        _teach(busy, URL, *NOVA_3WAY)
        assert idle.plan(45.0, URL).char_limit > busy.plan(45.0, URL).char_limit

    def test_floors_and_ceilings_hold(self):
        nt = NodeThroughput()
        _teach(nt, URL, 100_000.0, 100_000.0)      # absurdly fast
        plan = nt.plan(600.0, URL)
        assert plan.char_limit <= MAX_CHARS
        assert plan.max_tokens <= 768
        assert plan.char_limit >= MIN_CHARS and plan.max_tokens >= MIN_TOKENS


class TestRefusal:
    """`feasible=False` is load-bearing — it is what stops the doomed POST."""

    def test_the_starved_fourth_url_is_refused(self):
        # req 08766aa1: the 4th URL reached the distiller with ~6s of deadline
        # left, posted anyway, and ReadTimeout'd in 6s. At the node's real
        # rates that job needs ~27s.
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_3WAY)
        plan = nt.plan(6.0, URL)
        assert plan.feasible is False
        assert "6" in plan.reason or "3.0s" in plan.reason
        assert plan.char_limit == 0 and plan.max_tokens == 0

    def test_a_zero_or_negative_budget_is_refused(self):
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_SOLO)
        assert nt.plan(0.0, URL).feasible is False
        assert nt.plan(-10.0, URL).feasible is False

    def test_a_generous_budget_is_granted(self):
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_3WAY)
        plan = nt.plan(45.0, URL)
        assert plan.feasible is True
        assert plan.char_limit >= MIN_CHARS
        assert plan.max_tokens >= MIN_TOKENS


class TestTheShippedDefect:
    """Arithmetic on the constants that actually shipped, at measured rates."""

    def test_the_old_40k_2048_request_did_not_fit_45s(self):
        # 40,000 chars is ~12,500 tokens of real page text; 2048 output tokens
        # at the solo decode rate. This is the request the agent was posting.
        prefill, decode = NOVA_SOLO
        old_cost = (40_000 / CHARS_PER_TOKEN) / prefill + 2048 / decode
        assert old_cost > 45.0
        # And under the live 3-way concurrency it was not close.
        prefill, decode = NOVA_3WAY
        assert (40_000 / CHARS_PER_TOKEN) / prefill + 2048 / decode > 100.0

    def test_the_new_plan_does_fit_where_the_old_one_did_not(self):
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_3WAY)
        plan = nt.plan(45.0, URL)
        assert plan.feasible
        cost = ((plan.char_limit / CHARS_PER_TOKEN) / NOVA_3WAY[0]
                + plan.max_tokens / NOVA_3WAY[1])
        assert cost <= 45.0
        # It is genuinely smaller than what shipped — not a relabelling.
        assert plan.char_limit < 40_000
        assert plan.max_tokens < 2048

    def test_the_main_models_context_cannot_size_the_worker(self):
        # The old line was `max(4000, min(40000, (max_context - 2560) * 4))`
        # with max_context=240000 — i.e. the ceiling, unconditionally. The
        # replacement must not depend on the main model's window at all.
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_3WAY)
        assert nt.plan(45.0, URL).char_limit < 40_000


class TestContextWindow:
    """The constraint the original code claimed to apply and never did."""

    def test_a_small_context_node_caps_the_prompt(self):
        nt = NodeThroughput()
        _teach(nt, URL, 5000.0, 5000.0)          # fast enough to want 40k
        nt.note_context(URL, 4096)
        plan = nt.plan(300.0, URL)
        assert plan.feasible
        assert plan.char_limit <= int(4096 * CHARS_PER_TOKEN)
        assert plan.char_limit < MAX_CHARS

    def test_a_context_too_small_to_be_useful_is_refused(self):
        nt = NodeThroughput()
        _teach(nt, URL, 5000.0, 5000.0)
        nt.note_context(URL, 512)
        assert nt.plan(300.0, URL).feasible is False

    def test_novas_32k_window_does_not_bind(self):
        # The live worker: the time budget, not the context, is the limit.
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_SOLO)
        nt.note_context(URL, 32768)
        assert nt.plan(45.0, URL).feasible
        assert nt.context_tokens(URL) == 32768

    def test_bad_context_values_are_ignored(self):
        nt = NodeThroughput()
        for bad in (None, 0, -1, "32768", 3.5):
            nt.note_context(URL, bad)
        assert nt.context_tokens(URL) is None


class TestEnvOverride:
    def test_garbage_falls_back_to_the_measured_default(self, monkeypatch):
        for bad in ("", "   ", "abc", "0", "-5"):
            monkeypatch.setenv("GHOST_TEST_RATE", bad)
            assert env_float("GHOST_TEST_RATE", 150.0) == 150.0
        monkeypatch.setenv("GHOST_TEST_RATE", "42.5")
        assert env_float("GHOST_TEST_RATE", 150.0) == 42.5

    def test_unset_uses_the_default(self, monkeypatch):
        monkeypatch.delenv("GHOST_TEST_RATE", raising=False)
        assert env_float("GHOST_TEST_RATE", 7.0) == 7.0


class TestDescribe:
    def test_describe_says_the_numbers(self):
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_3WAY)
        text = nt.plan(45.0, URL).describe()
        assert "chars" in text and "tok" in text and "tok/s" in text

    def test_a_refusal_describes_itself_as_declined(self):
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_3WAY)
        assert "declined" in nt.plan(6.0, URL).describe()

    def test_clear_forgets_rates_and_context(self):
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_SOLO)
        nt.note_context(URL, 32768)
        nt.clear()
        assert nt.rates(URL)[2] == 0
        assert nt.context_tokens(URL) is None


class TestCallerCeilingsOnlyTighten:
    """A caller's limit is a further restriction, never a licence."""

    def test_a_large_caller_max_chars_cannot_exceed_the_module_ceiling(self):
        # Reachable shape: `_report_share_chars` is `max_context * 3.2 * 0.4 /
        # len(urls)`, which for a single-source research call is ~307,000. On a
        # fast enough node that would have sized the prompt ABOVE the 40k cap
        # the module exists to hold.
        nt = NodeThroughput()
        _teach(nt, URL, 5_000.0, 500.0)
        plan = nt.plan(600.0, URL, max_chars=307_200, max_tokens=99_999)
        # ⚠ LITERALS. `assert char_limit <= MAX_CHARS` imports the constant it
        # is supposed to pin, so raising MAX_CHARS moves both sides together
        # and the assertion cannot fail.
        assert plan.char_limit <= 40_000
        assert plan.max_tokens <= 768
        assert MAX_CHARS == 40_000, (
            "the cap exists so a fast node is never sized ABOVE the 40k "
            "behaviour this replaced")
        assert MIN_TOKENS == 128 and MIN_CHARS == 6_000

    def test_a_small_caller_max_chars_still_binds(self):
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_SOLO)
        assert nt.plan(45.0, URL, max_chars=6_000).char_limit <= 6_000

    def test_a_caller_ceiling_below_the_floor_is_refused(self):
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_SOLO)
        assert nt.plan(45.0, URL, max_chars=100).feasible is False


class TestPlansUseTheWorstRecentRate:
    """A mean is the wrong statistic for a deadline.

    Measured live: a 23,397-char / 592-token plan built while Nova was near
    idle (decode 32.7 tok/s) then ran under 3-way contention (decode 12) and
    blew its 45s budget. Decode degrades -62% under load where prefill
    degrades -28%, so the volatile half must be planned pessimistically.
    """

    def _fast(self):   # ~306 tok/s prefill, ~32 tok/s decode
        return _timings(12000, 39200, 600, 18700)

    def _slow(self):   # a genuine solo-equivalent slowdown (~220 / ~20 tok/s)
        # ⚠ Sized to sit ABOVE `PLAN_RATE_FLOOR_FRACTION` of the fast sample.
        # A deeper drop is clipped by the floor BY DESIGN: rates are stored
        # solo-equivalent now, so a solo-equivalent halving is far more likely
        # to be concurrency we failed to observe (traffic that bypasses the
        # permit gate is invisible to the counter) than a node that genuinely
        # became twice as slow on its own.
        return _timings(12000, 54500, 600, 30000)

    def test_one_slow_sample_immediately_governs_the_plan(self):
        nt = NodeThroughput()
        nt.observe(URL, self._fast())
        nt.observe(URL, self._slow())
        mean_prefill, mean_decode, _ = nt.rates(URL)
        plan_prefill, plan_decode, _ = nt.plan_rates(URL)
        assert plan_decode < mean_decode          # planned pessimistically
        assert plan_decode == pytest.approx(20, abs=1)
        assert plan_prefill == pytest.approx(220, abs=5)

    def test_a_plan_survives_the_regime_it_was_measured_in(self):
        nt = NodeThroughput()
        nt.observe(URL, self._fast())
        nt.observe(URL, self._slow())
        plan = nt.plan(45.0, URL)
        assert plan.feasible
        # Cost it at the SLOW regime — the one that broke the live call.
        cost = ((plan.char_limit / CHARS_PER_TOKEN) / 220.0
                + plan.max_tokens / 20.0)
        assert cost <= 45.0, f"{plan.describe()} costs {cost:.1f}s when loaded"

    def test_recovery_is_gradual_not_instant(self):
        nt = NodeThroughput()
        nt.observe(URL, self._fast())
        nt.observe(URL, self._slow())
        after_slow = nt.plan_rates(URL)[1]
        nt.observe(URL, self._fast())
        after_one_fast = nt.plan_rates(URL)[1]
        # One good call must not undo the evidence of contention.
        assert after_one_fast < after_slow * 3
        for _ in range(200):
            nt.observe(URL, self._fast())
        assert nt.plan_rates(URL)[1] > after_slow * 1.4  # but it does recover

    def test_an_unmeasured_node_still_plans_on_the_priors(self):
        nt = NodeThroughput()
        assert nt.plan_rates(URL) == (nt.default_prefill_tok_s,
                                      nt.default_decode_tok_s, 0)


class TestPriorsAndFloorsStayCompatible:
    """The two halves of one identity — pin BOTH, or a retune breaks it."""

    def test_the_priors_can_afford_the_floors(self):
        # ⚠ THE INVARIANT THE 8s QUEUE ALLOWANCE BROKE. With priors of 150/10
        # the smallest permitted request cost 34.9s against the ~34s a live
        # distill has, so a COLD agent refused every distillation and then
        # learned nothing from the calls it never made — the original defect
        # reached by refusal instead of timeout. Any retune of the priors, the
        # floors, the safety margin or the queue allowance must keep this true.
        from ghost_agent.tools.search import (
            _WEB_SUMMARY_TIMEOUT_S, _QUEUE_ALLOWANCE_S)
        budget = _WEB_SUMMARY_TIMEOUT_S - _QUEUE_ALLOWANCE_S
        plan = NodeThroughput().plan(budget, "never-seen-node")
        assert plan.feasible, (
            f"a cold agent cannot distil anything: {plan.reason}")
        assert plan.char_limit >= MIN_CHARS
        assert plan.max_tokens >= MIN_TOKENS

    def test_priors_stay_below_the_measured_node(self):
        # The other half: a prior above real hardware blows the first budget.
        # ⚠ COMPARED AGAINST THE *SOLO* RATES. The priors are solo-equivalent
        # now that `plan()` divides by the fan-out explicitly; comparing them
        # to a 3-way per-request rate is the unit confusion that made them
        # get divided twice and decline everything.
        nt = NodeThroughput()
        assert nt.default_prefill_tok_s < NOVA_SOLO[0]
        assert nt.default_decode_tok_s < NOVA_SOLO[1]

    def test_the_priors_afford_the_floors_at_every_live_fan_out(self):
        # The callers declare 3 (clearnet) and 2 (onion), and `plan_distill`
        # raises that to `in-flight + 1` when the node is already busy — so a
        # cold agent must be able to distil at least up to the node's slot
        # count, or the first research call after a restart declines
        # everything and never learns.
        from ghost_agent.tools.search import (
            _WEB_SUMMARY_TIMEOUT_S, _QUEUE_ALLOWANCE_S)
        budget = _WEB_SUMMARY_TIMEOUT_S - _QUEUE_ALLOWANCE_S
        # 1..4 is the reachable range: `plan_distill` clamps the fan-out to
        # the gate's own `cap`, because the gate admits at most `cap` at once
        # and anything larger plans for a node that cannot exist. A node with
        # more slots widens this, so the clamp is pinned separately.
        for fan in (1, 2, 3, 4):
            plan = NodeThroughput().plan(budget, "cold", concurrency=fan)
            assert plan.feasible, f"cold agent declines at fan-out {fan}: {plan.reason}"


class TestLearnedDensity:
    """chars/token was the last hand-tuned number in the sizing path."""

    def test_density_is_learned_from_real_prompts(self):
        nt = NodeThroughput()
        assert nt.observe_density(40_000, 12_500) is True     # 3.2
        assert nt.density() == pytest.approx(3.2 * 0.9, rel=0.01)

    def test_a_dense_corpus_shrinks_the_prompt(self):
        # CJK / minified assets run ~1.3 chars/token. A prompt sized on 3.2
        # would be 2.5x the tokens the budget paid for.
        loose, dense = NodeThroughput(), NodeThroughput()
        for nt in (loose, dense):
            _teach(nt, URL, *NOVA_SOLO)
        for _ in range(12):
            loose.observe_density(50_000, 10_000)     # 5.0 ch/tok
            dense.observe_density(13_000, 10_000)     # 1.3 ch/tok
        assert dense.plan(45.0, URL).char_limit < loose.plan(45.0, URL).char_limit

    def test_a_dense_plan_still_fits_its_budget_in_tokens(self):
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_SOLO)
        for _ in range(12):
            nt.observe_density(13_000, 10_000)        # 1.3 ch/tok
        plan = nt.plan(45.0, URL)
        real_tokens = plan.char_limit / 1.3           # what it ACTUALLY costs
        cost = real_tokens / NOVA_SOLO[0] + plan.max_tokens / NOVA_SOLO[1]
        assert cost <= 45.0

    def test_junk_density_samples_are_ignored(self):
        nt = NodeThroughput()
        for chars, toks in ((40_000, 10), (0, 5000), (-1, 5000),
                            (400_000, 5000), (5_000, 5000),
                            ("x", 5000), (40_000, None)):
            nt.observe_density(chars, toks)
        assert nt.density() == pytest.approx(CHARS_PER_TOKEN * 0.9, rel=0.01)


class TestSanityBounds:
    def test_an_infinite_rate_cannot_poison_the_estimator(self):
        nt = NodeThroughput()
        nt.observe(URL, _timings(float("inf"), 1000, 600, 20000))
        nt.observe(URL, _timings(12000, 40000, 600, 20000))
        prefill, _, _ = nt.rates(URL)
        assert prefill == pytest.approx(300, abs=20)

    def test_a_boolean_timing_cannot_teach_a_512000_tok_s_rate(self):
        nt = NodeThroughput()
        nt.observe(URL, _timings(512, True, 600, 20000))
        assert nt.plan_rates(URL)[0] <= nt.default_prefill_tok_s * 10

    def test_a_zero_default_cannot_reach_the_divisor(self):
        nt = NodeThroughput(default_prefill_tok_s=0.0, default_decode_tok_s=0.0)
        plan = nt.plan(45.0, "unknown")          # must not ZeroDivisionError
        assert isinstance(plan, DistillPlan)


class TestTheSlowTrackerExpires:
    def test_a_stale_slow_reading_relaxes_toward_the_mean(self, monkeypatch):
        # Without expiry one contended sample latches the node infeasible
        # forever: a refused plan posts no request, so the distill path never
        # produces the sample that would clear it.
        import ghost_agent.core.node_throughput as m
        clock = {"t": 1000.0}
        monkeypatch.setattr(m.NodeThroughput, "_now",
                            staticmethod(lambda: clock["t"]))
        nt = m.NodeThroughput()
        nt.observe(URL, _timings(12000, 39200, 600, 18700))    # fast
        nt.observe(URL, _timings(12000, 250000, 600, 150000))  # badly contended
        latched = nt.plan_rates(URL)[1]
        clock["t"] += m.SLOW_HALF_LIFE_S * 8
        relaxed = nt.plan_rates(URL)[1]
        assert relaxed > latched
        assert relaxed == pytest.approx(nt.rates(URL)[1], rel=0.05)

    def test_it_still_governs_while_fresh(self, monkeypatch):
        import ghost_agent.core.node_throughput as m
        clock = {"t": 500.0}
        monkeypatch.setattr(m.NodeThroughput, "_now",
                            staticmethod(lambda: clock["t"]))
        nt = m.NodeThroughput()
        nt.observe(URL, _timings(12000, 39200, 600, 18700))
        nt.observe(URL, _timings(12000, 54500, 600, 50000))
        clock["t"] += 1.0
        assert nt.plan_rates(URL)[1] < nt.rates(URL)[1]


class TestPoolWorstCase:
    def test_a_non_dominated_pool_is_sized_for_both_nodes(self):
        # Review's measured counterexample to `min(plans, key=char_limit)`:
        # A is fast to read and slow to write, B the reverse, so the
        # smallest-chars plan is the LARGEST-tokens plan and overran on A by
        # 2.4x. The plan must fit whichever node answers.
        nt = NodeThroughput()
        nt._rates["A"] = [1000.0, 8.0, 5]
        nt._rates["B"] = [120.0, 60.0, 5]
        plan = nt.plan_worst_of(["A", "B"], 45.0)
        if plan.feasible:
            for prefill, decode in ((1000.0, 8.0), (120.0, 60.0)):
                cost = ((plan.char_limit / plan.density) / prefill
                        + plan.max_tokens / decode)
                assert cost <= 45.0, f"{plan.describe()} costs {cost:.1f}s"

    def test_the_smallest_context_in_the_pool_wins(self):
        nt = NodeThroughput()
        nt._rates["A"] = [5000.0, 500.0, 5]
        nt._rates["B"] = [5000.0, 500.0, 5]
        nt.note_context("A", 131072)
        nt.note_context("B", 4096)
        plan = nt.plan_worst_of(["A", "B"], 300.0)
        assert plan.char_limit <= int(4096 * plan.density)


class TestConcurrencyScaling:
    """A rate without its concurrency is not a measurement.

    Measured on Nova going from 1 to 3 concurrent requests:
        prefill 304 -> 220 tok/s each  (AGGREGATE x2.17 — llama.cpp batches it)
        decode  31.6 -> 12.0 tok/s each (AGGREGATE x1.14 — bandwidth-bound)
    so per-request prefill ~ solo/sqrt(N) and decode ~ solo/N.

    This is the defect the slow tracker could NOT fix: a research wave plans
    every URL before any of them is in flight, so all of them are sized as if
    they had the node to themselves. Live (req b86cdd59): four plans of
    ~19,936 chars / 596 tok built at ~306/32 then ran at 115/11 — 126s of work
    against a 45s budget.
    """

    def test_the_scaling_law_matches_the_measurement(self):
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_SOLO)
        p3 = nt.plan(45.0, URL, concurrency=3)
        # Predicted per-request rates at N=3, and they must be CONSERVATIVE
        # relative to what the node really delivered (220 / 12.0).
        assert p3.prefill_tok_s == pytest.approx(304 / 3 ** 0.5, rel=0.02)
        assert p3.decode_tok_s == pytest.approx(31.6 / 3, rel=0.02)
        assert p3.prefill_tok_s < NOVA_3WAY[0]
        assert p3.decode_tok_s < NOVA_3WAY[1]

    def test_a_declared_fan_out_shrinks_the_plan(self):
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_SOLO)
        sizes = [nt.plan(45.0, URL, concurrency=c).char_limit
                 for c in (1, 2, 3, 4)]
        assert sizes == sorted(sizes, reverse=True), sizes
        # Strictly monotone, and materially smaller — but the MIN_CHARS floor
        # compresses the bottom of the range, so this is 0.6 rather than 0.5.
        assert sizes[3] < sizes[0] * 0.6, sizes

    def test_the_wave_that_timed_out_would_now_fit(self):
        # The exact live failure, re-planned. It must come out small enough to
        # finish at the rates the node was ACTUALLY running (115/11 per
        # request, text at ~2.4 chars/token).
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_SOLO)
        from ghost_agent.tools.search import (
            _WEB_SUMMARY_TIMEOUT_S, _QUEUE_ALLOWANCE_S)
        plan = nt.plan(_WEB_SUMMARY_TIMEOUT_S - _QUEUE_ALLOWANCE_S, URL,
                       concurrency=3)
        assert plan.feasible
        actual = plan.char_limit / 2.4 / 115 + plan.max_tokens / 11
        assert actual <= _WEB_SUMMARY_TIMEOUT_S, (
            f"{plan.describe()} would still cost {actual:.0f}s")
        # ...and it is genuinely smaller than the plan that failed.
        assert plan.char_limit < 19_936 and plan.max_tokens < 596

    def test_a_sample_is_normalised_by_the_concurrency_it_ran_at(self):
        # The same physical node measured alone and 3-way must land on
        # roughly the same solo-equivalent rate — otherwise a loaded sample
        # teaches "this node is slow" and gets divided by the fan-out AGAIN.
        alone, busy = NodeThroughput(), NodeThroughput()
        alone.observe(URL, _timings(12000, 39474, 600, 18987), concurrency=1)
        busy.observe(URL, _timings(12000, 54545, 600, 50000), concurrency=3)
        assert alone.rates(URL)[0] == pytest.approx(busy.rates(URL)[0], rel=0.30)
        assert alone.rates(URL)[1] == pytest.approx(busy.rates(URL)[1], rel=0.30)

    def test_round_tripping_a_loaded_sample_is_stable(self):
        # Observe at N, plan at N -> you get back what you measured. Without
        # that, every wave would shrink the next one without bound.
        nt = NodeThroughput()
        for _ in range(6):
            nt.observe(URL, _timings(12000, 54545, 600, 50000), concurrency=3)
        p = nt.plan(45.0, URL, concurrency=3)
        assert p.prefill_tok_s == pytest.approx(220, rel=0.05)
        assert p.decode_tok_s == pytest.approx(12, rel=0.05)

    def test_concurrency_one_is_a_no_op(self):
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_SOLO)
        assert (nt.plan(45.0, URL).char_limit
                == nt.plan(45.0, URL, concurrency=1).char_limit)

    def test_a_bogus_concurrency_cannot_widen_the_plan(self):
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_SOLO)
        base = nt.plan(45.0, URL, concurrency=1).char_limit
        for bad in (0, -3, None):
            assert nt.plan(45.0, URL, concurrency=bad).char_limit <= base


class TestNoPermanentRateFloor:
    """Pin the DELETION: a downgraded node must stay plannable.

    A "never plan below half the best rate ever seen" floor was added while a
    poisoned estimator was learning ~4 tok/s on a node that does 31.6. The
    poison was a wiring bug, and a floor derived from the same corrupted
    estimator was measured to change nothing. What it DID do was refuse to
    decay: swap in a heavier model and the node is pinned at a speed it can no
    longer reach, forever — a timeout returns no `timings`, so no sample can
    ever move the high-water mark.
    """

    def test_a_permanently_slower_node_is_tracked_all_the_way_down(self):
        nt = NodeThroughput()
        for _ in range(4):                       # fast era
            nt.observe(URL, _timings(12000, 39474, 600, 18987), concurrency=1)
        fast = nt.plan_rates(URL)[1]
        for _ in range(40):                      # 4x heavier model swapped in
            nt.observe(URL, _timings(12000, 157896, 600, 75948), concurrency=1)
        slow = nt.plan_rates(URL)[1]
        assert slow < fast / 3, (fast, slow)
        # ...and a plan built on it must actually fit the new hardware.
        plan = nt.plan(60.0, URL)
        if plan.feasible:
            cost = (plan.char_limit / plan.density) / nt.plan_rates(URL)[0] \
                + plan.max_tokens / slow
            assert cost <= 60.0

    def test_no_floor_constant_survives(self):
        import ghost_agent.core.node_throughput as m
        assert not hasattr(m, "PLAN_RATE_FLOOR_FRACTION")


class TestPerRateDecay:
    def test_a_prefill_dip_does_not_re_arm_a_decayed_decode_dip(self, monkeypatch):
        # One shared `slow_at` let an unrelated prefill dip resurrect a decode
        # dip that had already decayed (review, CONFIRMED).
        import ghost_agent.core.node_throughput as m
        clock = {"t": 1000.0}
        monkeypatch.setattr(m.NodeThroughput, "_now", staticmethod(lambda: clock["t"]))
        nt = m.NodeThroughput()
        nt.observe(URL, _timings(12000, 39474, 600, 18987), concurrency=1)
        nt.observe(URL, _timings(12000, 39474, 600, 120000), concurrency=1)  # decode dip
        clock["t"] += m.SLOW_HALF_LIFE_S * 3                                  # let it decay
        decayed = nt.plan_rates(URL)[1]
        nt.observe(URL, _timings(12000, 300000, 600, 18987), concurrency=1)   # PREFILL dip only
        after = nt.plan_rates(URL)[1]
        assert after >= decayed * 0.98, (decayed, after)




class TestMoreSurvivingMutants:
    def test_the_operator_knobs_use_their_real_names(self, monkeypatch):
        # `env_float` was only ever tested through a throwaway GHOST_TEST_RATE,
        # so renaming either real knob silently disabled it.
        monkeypatch.setenv("GHOST_NODE_PREFILL_TOK_S", "77")
        monkeypatch.setenv("GHOST_NODE_DECODE_TOK_S", "9")
        nt = NodeThroughput()
        assert nt.default_prefill_tok_s == 77.0
        assert nt.default_decode_tok_s == 9.0

    def test_predicted_s_is_the_recomputed_cost_not_a_constant(self):
        # `describe()` prints it on the operator's stream. A plan that reports
        # "~0s" for a 40s job is a broken instrument — pin it against a value
        # recomputed from the plan's own numbers, not against itself.
        nt = NodeThroughput()
        _teach(nt, URL, *NOVA_SOLO)
        p = nt.plan(60.0, URL, concurrency=3)
        assert p.feasible
        expect = ((p.char_limit / p.density) / p.prefill_tok_s
                  + p.max_tokens / p.decode_tok_s)
        assert p.predicted_s == pytest.approx(expect, rel=0.01)
        assert p.predicted_s > 1.0

    def test_headroom_never_vetoes_the_floor(self):
        # The deliberate split: the FLOOR is judged against the whole budget,
        # only the SURPLUS against the utilised portion. Merging them made an
        # unmeasured node decline everything and then learn nothing from calls
        # it never made. Pinned at the boundary, where a merge flips it.
        nt = NodeThroughput(default_prefill_tok_s=150.0, default_decode_tok_s=15.0)
        plan = nt.plan(57.0, "cold", concurrency=3)
        floor_cost = ((MIN_CHARS / plan.density) / plan.prefill_tok_s
                      + MIN_TOKENS / plan.decode_tok_s)
        usable = 57.0 - 3.0
        assert floor_cost <= usable, "fixture no longer sits at the boundary"
        assert floor_cost > usable * 0.85, "fixture no longer sits at the boundary"
        assert plan.feasible, (
            "the headroom vetoed the floor — the original defect, by refusal")
