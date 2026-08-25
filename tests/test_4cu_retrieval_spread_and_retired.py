"""§4CU — two axes the LOOP YIELD view could not express.

**1. Retrieval CONCENTRATION.** `invoked` is a SUM over per-item counters,
and a sum cannot tell a store whose 50 items are all being drawn from a
store where one item takes every retrieval and 49 never surface. Those are
opposite findings with opposite remedies: the first is healthy, the second
means the RETRIEVER is broken and minting more cannot help. The failure is
measured in arXiv:2604.27003 ("When Continual Learning Moves to Memory"),
where 88.5% of queries retrieved the identical top item despite high
key-level diversity — their conclusion, "pool size alone predicts nothing
about retrieval effectiveness", is a direct statement that `minted` and
`invoked` cannot see it.

**2. RETIRED.** `foresight.gate` read BARREN — "produces artifacts nobody
invokes", sorted FIRST in a worst-news-first view, remedy "find it a
consumer" — while the real answer was that §4CS item G had SETTLED it: the
index is anti-predictive, and `_evaluate_bucket` checks precision before
the interval test, so no bucket can enable at any n. BARREN reads as owed
work. RETIRED says measured dead, nothing owed, and is DERIVED so it
retracts itself if the sign flips.

The tests below drive real store files through the real probes. Two
properties carry most of the weight:

* `spread.total` must EQUAL the row's `invoked` — a concentration computed
  over a different population than its own row is the plausible lie §4CE
  found three instruments telling, and it would be invisible by reading.
* a verdict is WITHHELD under the power floor while the numbers are still
  printed. §4CE is exactly the failure of reporting a verdict a
  denominator cannot support.
"""

import json

import pytest

from ghost_agent.core import liveness as L
from ghost_agent.core.liveness import (
    BARREN, CONCENTRATED, GATED, RETIRED, SPREAD_OK, UNDEFINED, UNDERPOWERED,
    YIELDING, Spread, YieldResult, _anti_predictive, _eligible_counts,
    _spread, _yield_status, render_yield, yield_all,
)


@pytest.fixture
def home(tmp_path):
    for rel in ("system/memory/composed_skills",
                "system/memory/acquired_skills",
                "system/foresight", "system/evolve", "system/optim"):
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)
    return tmp_path


def _w(home, rel, obj):
    p = home / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))
    return p


def _row(rows, name):
    return next(r for r in rows if r["name"] == name)


# ══════════════════════════════════════════════════════════════════════
# 1. The statistic itself
# ══════════════════════════════════════════════════════════════════════
class TestSpreadArithmetic:
    def test_perfectly_even_reads_evenness_one_and_distributed(self):
        sp = _spread([10] * 10)
        assert sp.verdict == SPREAD_OK
        assert sp.top1_share == pytest.approx(0.1)
        assert sp.coverage == pytest.approx(1.0)
        assert sp.entropy_ratio == pytest.approx(1.0)

    def test_all_on_one_item_reads_evenness_zero_and_concentrated(self):
        sp = _spread([100, 0, 0, 0, 0])
        assert sp.verdict == CONCENTRATED
        assert sp.top1_share == pytest.approx(1.0)
        assert sp.entropy_ratio == pytest.approx(0.0)

    def test_entropy_is_never_pushed_past_one_by_float_error(self):
        """A '1.0000000002 uniform' reads as a bug in the instrument.

        ⚠ The first version of this test swept n ∈ {3,7,11,50,97} and a
        mutation showed that DELETING the clamp changed nothing — none of
        those n overflow. The clamp looked like dead defence. It is not:
        the smallest overflowing uniform store is n=5, and the test now
        carries its own NEGATIVE CONTROL (the unclamped value, recomputed
        here, must actually exceed 1.0) so it can never again pass while
        asserting nothing.
        """
        import math
        for n in (5, 12, 13, 18, 24, 26):
            p = 1.0 / n
            raw = -sum(p * math.log(p) for _ in range(n)) / math.log(n)
            assert raw > 1.0, f"n={n} no longer overflows; pick another"
            assert 0.0 <= _spread([7] * n).entropy_ratio <= 1.0

    def test_the_clamp_does_not_flatten_a_real_ratio(self):
        """The other half: clamping must not quietly rewrite a legitimate
        interior value to 1.0."""
        assert _spread([9, 1]).entropy_ratio == pytest.approx(0.469, abs=1e-3)

    def test_the_published_collapse_signature_is_caught(self):
        """arXiv:2604.27003's condition: 88.5% of queries on one item."""
        sp = _spread([885] + [115 // 9] * 9)
        assert sp.verdict == CONCENTRATED
        assert sp.top1_share > 0.88

    def test_a_wide_store_where_most_never_surfaces_is_concentrated(self):
        """The SECOND signature: no single item dominates, but the tail is
        dead. 20 items share the draws evenly and 80 never surface —
        top-1 is 5%, so the top-1 bar alone would call this healthy."""
        sp = _spread([50] * 20 + [0] * 80)
        assert sp.top1_share < 0.1               # top-1 bar does NOT fire
        assert sp.coverage == pytest.approx(0.2)
        assert sp.verdict == CONCENTRATED
        assert "never surfaced" in sp.why


class TestPowerFloor:
    """§4CE: a verdict a denominator cannot support IS the instrument
    failure, not a conservative reading of it."""

    def test_under_the_floor_the_verdict_is_withheld(self):
        sp = _spread([3, 0, 0, 0, 0, 0, 0, 0, 0, 0])   # 3 draws, 10 items
        assert sp.verdict == UNDERPOWERED
        assert sp.verdict != CONCENTRATED

    def test_but_the_NUMBERS_are_still_reported(self):
        """Withholding the numbers too would hide the thing the axis was
        added to expose. Only the JUDGEMENT needs a denominator."""
        sp = _spread([3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert sp.top1_share == pytest.approx(1.0)
        assert sp.total == 3 and sp.nonzero == 1 and sp.n == 10

    def test_the_floor_is_five_per_eligible_item(self):
        assert _spread([5] * 4).verdict != UNDERPOWERED        # 20 == 5*4
        assert _spread([4] * 4 + [3]).verdict == UNDERPOWERED  # 19 < 5*5

    @pytest.mark.parametrize("counts", [
        [2, 1] + [0] * 8,          # total 3 < n 10  -> floor 1/3
        [24, 0, 0, 0, 0],          # total 24 > n 5  -> floor 5/24, NOT 1/24
        [1] * 100 + [399],         # total 499, n 101 -> floor 5/499
        [4] * 3,                   # total 12, n 3   -> floor 4/12
    ])
    def test_the_why_states_the_REAL_arithmetic_bound(self, counts):
        """⚠ THIS WAS A TOKEN PIN AND A REVIEWER PROVED IT. The first
        version asserted only that the string "cannot fall below" was
        present; rewriting the sentence to "cannot fall below 99%" left
        it green. And the number WAS wrong — `1/total` is the pigeonhole
        floor only when n >= total, so the live `skills.acquired` row
        printed "cannot fall below 4%" where the true floor is 20.8%,
        understating it 5x in the direction that makes the observed
        concentration look more meaningful.

        Recomputed here from the counts, so the assertion cannot pass on
        a wrong number."""
        import math
        import re
        sp = _spread(counts)
        assert sp.verdict == UNDERPOWERED
        want = math.ceil(sp.total / sp.n) / sp.total
        m = re.search(r"cannot fall below (\d+)%", sp.why)
        assert m, sp.why
        assert int(m.group(1)) == round(want * 100), (
            f"printed {m.group(1)}%, pigeonhole floor is {want:.1%}")

    def test_the_bound_is_NOT_one_over_total_when_n_is_small(self):
        """The negative control that makes the test above meaningful: the
        two formulas must actually DISAGREE on this input, or the pin
        cannot distinguish them."""
        import math
        sp = _spread([24, 0, 0, 0, 0])
        assert math.ceil(sp.total / sp.n) / sp.total != 1.0 / sp.total

    def test_a_UNIFORM_TWO_ITEM_store_is_DISTRIBUTED_not_concentrated(self):
        """⚠ `>=` made DISTRIBUTED structurally UNREACHABLE at n == 2 —
        exhaustively, for every total from 1 to 2000 — so a perfectly
        even two-item store was reported as a broken retriever with a
        claim that is arithmetically false ("more than every other item
        combined": 5 is not more than 5). Two active skills or two
        approved macros is an ordinary shape."""
        sp = _spread([5, 5])
        assert sp.top1_share == pytest.approx(0.5)
        assert sp.verdict == SPREAD_OK

    def test_but_a_SKEWED_two_item_store_is_still_caught(self):
        assert _spread([9, 1]).verdict == CONCENTRATED

    def test_exactly_half_is_not_MORE_than_the_rest_at_any_n(self):
        assert _spread([50, 25, 25]).verdict == SPREAD_OK
        assert _spread([51, 25, 24]).verdict == CONCENTRATED

    def test_a_single_eligible_item_is_UNDEFINED_not_concentrated(self):
        """100% top-1 over one item is true by construction — reporting it
        as concentration is the derived-zero error on the new axis."""
        sp = _spread([99])
        assert sp.verdict == UNDEFINED
        assert "by construction" in sp.why

    def test_nothing_retrieved_is_UNDEFINED_not_a_finding(self):
        """The row's own BARREN/EMPTY status is the finding; repeating it
        as a concentration verdict double-counts it."""
        sp = _spread([0, 0, 0])
        assert sp.verdict == UNDEFINED
        assert sp.top1_share is None


# ══════════════════════════════════════════════════════════════════════
# 2. The eligible population — one formula, four probes
# ══════════════════════════════════════════════════════════════════════
class TestEligiblePopulation:
    def test_ineligible_and_unused_items_are_excluded(self):
        rows = [{"c": 4, "on": True}, {"c": 0, "on": True}, {"c": 0, "on": False}]
        assert sorted(_eligible_counts(
            rows, lambda r: r["c"], lambda r: r["on"])) == [0, 4]

    def test_an_item_with_a_COUNT_stays_in_even_when_no_longer_eligible(self):
        """THE INVARIANT. A macro un-approved after being run, or a skill
        degraded after being used, still contributed to `invoked`. Dropping
        it from the denominator would make `total` disagree with the row
        it describes."""
        rows = [{"c": 7, "on": False}, {"c": 1, "on": True}]
        got = _eligible_counts(rows, lambda r: r["c"], lambda r: r["on"])
        assert sorted(got) == [1, 7]

    def test_none_and_missing_counts_read_as_zero(self):
        rows = [{"c": None, "on": True}, {"on": True}]
        assert _eligible_counts(
            rows, lambda r: r.get("c"), lambda r: r["on"]) == [0, 0]


# ══════════════════════════════════════════════════════════════════════
# 3. total == invoked, driven through the REAL probes
# ══════════════════════════════════════════════════════════════════════
class TestSpreadMatchesItsOwnRow:
    """`pin identity, not a property`: assert the spread's total EQUALS a
    separately-computed `invoked`, not merely that it is 'reasonable'."""

    def _stores(self, home):
        _w(home, "system/memory/skills_playbook.json", [
            {"retrievals": 9, "helpful_retrievals": 4},
            {"retrievals": 3},
            {"retrievals": 0},
            {"retrievals": 41, "quarantined": True},   # excluded BOTH sides
        ])
        _w(home, "system/memory/auto_skills.json", {
            "a": {"retrievals": 6}, "b": {"retrievals": 0},
        })
        _w(home, "system/memory/acquired_skills/skills_registry.json", {
            "x": {"usage_count": 8, "status": "active"},
            "y": {"usage_count": 0, "status": "active"},
            "z": {"usage_count": 5, "status": "degraded"},   # used → IN
            # Degraded AND never used → OUT. Without this row the eligible
            # gate is undetectable: every other skill here is admitted by
            # the union clause whether or not the gate is honoured, so a
            # mutation deleting it left the suite green.
            "w": {"usage_count": 0, "status": "degraded"},
        })
        _w(home, "system/memory/composed_skills/composed_skills.json", {
            "auto_p": {"usage_count": 2, "status": "active",
                       "trigger_description": "auto-mined from"},
            "auto_q": {"usage_count": 0, "status": "proposed",
                       "trigger_description": "auto-mined from"},
        })

    @pytest.mark.parametrize("name", [
        "lessons.playbook", "skills.graduated",
        "skills.acquired", "macros.auto_mined",
    ])
    def test_total_equals_invoked(self, home, monkeypatch, name):
        monkeypatch.setattr(L, "_macro_marks", lambda: ("auto-mined from", "gr"))
        self._stores(home)
        row = _row(yield_all(home)["rows"], name)
        assert row["spread"] is not None, f"{name} lost its spread"
        assert row["spread"]["total"] == row["invoked"], (
            f"{name}: spread describes a different population than its row")

    def test_a_quarantined_lesson_is_in_NEITHER_total_nor_n(self, home):
        """Quarantine is retention without service — `_filter_quarantined`
        runs at both retrieval surfaces, so a quarantined lesson can never
        be drawn again. Its 41 retrievals are dead history."""
        self._stores(home)
        row = _row(yield_all(home)["rows"], "lessons.playbook")
        assert row["spread"]["total"] == 12          # 9 + 3, NOT 53
        assert row["spread"]["n"] == 3               # NOT 4

    def test_a_proposed_macro_is_not_in_the_denominator(self, home, monkeypatch):
        """26 of 29 live macros are `proposed`; scoring over all of them
        would report 100% top-1 about a store with one runnable item."""
        monkeypatch.setattr(L, "_macro_marks", lambda: ("auto-mined from", "gr"))
        self._stores(home)
        row = _row(yield_all(home)["rows"], "macros.auto_mined")
        assert row["spread"]["n"] == 1               # only auto_p is active
        assert row["spread"]["verdict"] == UNDEFINED

    def test_a_degraded_but_USED_skill_stays_in(self, home):
        """It contributed to `invoked`; dropping it breaks the invariant."""
        self._stores(home)
        row = _row(yield_all(home)["rows"], "skills.acquired")
        # x, y active + z (degraded but USED). `w` is degraded and unused —
        # not advertised, not dispatchable, never drawn — so it is not a
        # place a retrieval could have gone.
        assert row["spread"]["n"] == 3
        assert row["spread"]["total"] == 13


class TestSpreadIsAbsentWhereThereIsNoPerItemChannel:
    def test_gepa_foresight_evolve_carry_no_spread(self, home):
        _w(home, "system/foresight/gate.json", {"buckets": {}})
        rows = yield_all(home)["rows"]
        for name in ("prompts.gepa", "foresight.gate", "evolve.candidates"):
            assert _row(rows, name)["spread"] is None, (
                f"{name} has no per-item counter — a spread here would be "
                f"fabricated")


# ══════════════════════════════════════════════════════════════════════
# 4. RETIRED
# ══════════════════════════════════════════════════════════════════════
class TestAntiPredictiveTestsThePRECISION_withPOWER:
    """⚠ REWRITTEN AFTER ROUND 1, WHICH TWO REVIEWERS DISQUALIFIED.

    Version 1 was `spread <= 0` with both denominators merely truthy. On
    the LIVE gate that retired a loop on 14 predicted-fail rows of which
    ONE failed — Fisher exact p = 1.00, and the interval for the spread
    contains both zero AND the +0.10 bar the verdict cites as the thing
    it fails. A terminal "nothing owed" verdict on an arithmetically
    undetectable difference is §4CE VERBATIM, committed inside the
    instrument added to prevent it.

    It was also a PROXY. The sign of a pooled spread fails both ways: a
    Simpson reversal (every bucket +0.10, pooled -0.47) retired a healthy
    loop, and precision 0.07 at fail_n 500 with a hair-positive spread was
    NOT retired — reported as owed work forever.

    The rule now tests the THING with power: `_evaluate_bucket` rejects on
    `precision < min_fail_precision` BEFORE the interval test, so a bucket
    under the bar cannot enable at ANY n. That is established only when
    the precision interval's UPPER bound is below the bar.
    """

    def test_the_live_shape_retires_ON_PRECISION(self):
        """1/14: Wilson upper 0.31, below the 0.60 bar."""
        assert _anti_predictive(
            {"fail_n": 14, "fail_hits": 1, "ok_n": 564, "ok_hits": 60}) is True

    def test_ONE_ROW_PER_ARM_DOES_NOT_RETIRE(self):
        """The single worst property of version 1."""
        assert _anti_predictive(
            {"spread": 0.0, "fail_n": 1, "fail_hits": 0, "ok_n": 1, "ok_hits": 1}) is False
        assert _anti_predictive(
            {"spread": -0.5, "fail_n": 1, "fail_hits": 0, "ok_n": 2, "ok_hits": 1}) is False

    def test_a_MEASURABLY_dead_index_retires_even_with_a_POSITIVE_spread(self):
        """Version 1's other failure direction: precision 0.07 at n=500
        is decisively under the bar, and a hair-positive pooled spread
        kept it reading as 'waiting for data' forever."""
        assert _anti_predictive(
            {"fail_n": 500, "fail_hits": 35, "ok_n": 500, "ok_hits": 200}) is True

    def test_a_GOOD_index_does_not_retire(self):
        assert _anti_predictive(
            {"fail_n": 40, "fail_hits": 34, "ok_n": 500, "ok_hits": 25}) is False

    def test_the_bar_comes_from_the_GATE_not_a_literal(self):
        disc = {"fail_n": 30, "fail_hits": 12, "ok_n": 300,
                "ok_hits": 150}   # precision 0.40 < ok_fail 0.50
        assert _anti_predictive(disc, bar=0.60) is True
        assert _anti_predictive(disc, bar=0.30) is False

    def test_gate_bars_are_read_from_the_artifact(self):
        from ghost_agent.core.liveness import _gate_bars
        assert _gate_bars({"params": {"min_fail_precision": 0.8,
                                      "min_fail_n": 25}}) == (0.8, 25)
        assert _gate_bars({}) == (0.60, 10)
        assert _gate_bars({"params": {"min_fail_precision": "x",
                                      "min_fail_n": True}}) == (0.60, 10)

    @pytest.mark.parametrize("disc", [
        # ⚠ THE BOUNDS CHECK WAS UNPINNED and a mutation replacing it
        # with `pass` survived the whole suite — every malformed case in
        # the original list was behaviourally IDENTICAL under the mutant
        # (a >1 precision fails the sign test anyway; a negative count
        # makes `_wilson_upper` return None). The inputs that actually
        # DECIDE it are the ones where an out-of-range `ok_hits` inflates
        # `ok_fail_rate` past a real precision, so the sign test passes
        # and a tight interval then retires a live loop off a schema slip.
        {"fail_n": 10, "fail_hits": 0, "ok_n": 100, "ok_hits": 101},
        {"fail_n": 10, "fail_hits": 1, "ok_n": 700, "ok_hits": 900},
        {"fail_n": 14, "fail_hits": 1, "ok_n": 700, "ok_hits": -5},
        {"fail_n": 0, "fail_hits": 0, "ok_n": 700, "ok_hits": 70},
        {"fail_n": 14, "fail_hits": 1, "ok_n": 0, "ok_hits": 0},
        {"fail_n": 14, "ok_n": 700, "ok_hits": 70},
        {"fail_n": 14, "fail_hits": 99, "ok_n": 700, "ok_hits": 70},
        {"fail_n": True, "fail_hits": 1, "ok_n": 5, "ok_hits": 3},
        {"fail_n": "14", "fail_hits": 1, "ok_n": 5, "ok_hits": 3},
        {}, None, "anti-predictive", 3,
    ])
    def test_a_missing_or_malformed_denominator_is_UNMEASURED(self, disc):
        assert _anti_predictive(disc) is False

    def test_it_does_NOT_read_the_verdict_PROSE(self):
        assert _anti_predictive(
            {"verdict": "⚠ ANTI-PREDICTIVE, pooled over ALL buckets",
             "fail_n": 14, "ok_n": 747}) is False


class TestAPoweredLiveBucketVetoesRetirement:
    """An aggregate can be dead while a stratum is alive."""

    def _fn(self):
        from ghost_agent.core.liveness import _live_bucket_can_still_qualify
        return _live_bucket_can_still_qualify

    def test_a_POWERED_bucket_reaching_the_bar_counts_as_alive(self):
        assert self._fn()({"b": {"fail_n": 30, "fail_hits": 27}}) is True

    def test_an_UNDERPOWERED_bucket_does_NOT(self):
        """⚠ THE SECOND-ORDER TRAP. Measured on the live gate: 4 of 64
        buckets have a precision interval reaching the bar, and EVERY one
        sits at fail_n 1-3, where the Wilson upper bound is wide BY
        CONSTRUCTION (0/1 reads 0.79). Counting those as alive is absence
        of evidence read as evidence, and it makes RETIRED unreachable
        forever, since some one-row bucket always exists."""
        assert self._fn()({"b": {"fail_n": 1, "fail_hits": 1}}) is False
        assert self._fn()({"b": {"fail_n": 3, "fail_hits": 1}}) is False

    def test_the_threshold_is_the_gates_own_min_fail_n(self):
        b = {"b": {"fail_n": 12, "fail_hits": 11}}
        assert self._fn()(b, min_fail_n=10) is True
        assert self._fn()(b, min_fail_n=30) is False

    def test_a_powered_bucket_UNDER_the_bar_is_not_alive(self):
        assert self._fn()({"b": {"fail_n": 100, "fail_hits": 5}}) is False

    @pytest.mark.parametrize("b", [None, 7, "x", {}, {"b": None},
                                   {"b": {"fail_n": "30", "fail_hits": 27}}])
    def test_junk_is_not_alive(self, b):
        assert self._fn()(b) is False


class TestForesightGateRetires:
    """⚠ THIS WHOLE CLASS WAS ACCIDENTALLY DELETED by a patch that
    replaced the span between two anchors and did not notice a third
    class sitting between them. Nothing failed — the deletion of a test
    class is invisible to a test run. The §4CU mutation harness caught
    it: `"retired": [...]` -> `[]` suddenly SURVIVED, because the only
    test reading that bucket had ceased to exist.

    `pin-the-deletion` is the standing lesson: a removal needs a
    regression test for ABSENCE, and a mutation harness is the only
    thing that notices a pin has silently gone.
    """

    def _gate(self, home, disc, buckets=None):
        _w(home, "system/foresight/gate.json",
           {"ledger_rows": 763,
            "params": {"min_fail_precision": 0.60, "min_fail_n": 10},
            "buckets": buckets or {f"b{i}": {"enabled": False}
                                   for i in range(64)},
            "discrimination": disc})

    def test_measured_backwards_reads_RETIRED(self, home):
        self._gate(home, {"fail_n": 14, "fail_hits": 1, "ok_n": 747,
                          "ok_hits": 80, "verdict": "precision 7.1%"})
        assert _row(yield_all(home)["rows"], "foresight.gate")["status"] \
            == RETIRED

    def test_the_same_gate_with_no_verdict_yet_stays_BARREN(self, home):
        """The retirement must be EARNED by a measurement. Without one the
        row keeps saying 'blocked upstream', which is still true."""
        self._gate(home, {"spread": None, "fail_n": 0, "fail_hits": 0,
                          "ok_n": 0, "ok_hits": 0})
        assert _row(yield_all(home)["rows"], "foresight.gate")["status"] \
            == BARREN

    def test_it_UN_RETIRES_when_the_PRECISION_recovers(self, home):
        """Derived, not asserted — no code change, no operator action."""
        self._gate(home, {"fail_n": 14, "fail_hits": 1, "ok_n": 747,
                          "ok_hits": 80})
        assert _row(yield_all(home)["rows"],
                    "foresight.gate")["status"] == RETIRED
        self._gate(home, {"fail_n": 40, "fail_hits": 34, "ok_n": 700,
                          "ok_hits": 35})
        assert _row(yield_all(home)["rows"],
                    "foresight.gate")["status"] == BARREN

    def test_a_POWERED_LIVE_BUCKET_keeps_it_out_of_RETIRED(self, home):
        """A dead pool does not settle the question if a stratum with
        real data can still qualify."""
        self._gate(home, {"fail_n": 500, "fail_hits": 35, "ok_n": 500,
                          "ok_hits": 200},
                   buckets={"alive": {"enabled": False, "fail_n": 30,
                                      "fail_hits": 27}})
        assert _row(yield_all(home)["rows"],
                    "foresight.gate")["status"] == BARREN

    def test_UNDERPOWERED_buckets_do_NOT_veto_retirement(self, home):
        """Otherwise RETIRED is unreachable forever — some one-row bucket
        always exists, and its interval is wide by construction."""
        self._gate(home, {"fail_n": 500, "fail_hits": 35, "ok_n": 500,
                          "ok_hits": 200},
                   buckets={"thin": {"enabled": False, "fail_n": 1,
                                     "fail_hits": 1}})
        assert _row(yield_all(home)["rows"],
                    "foresight.gate")["status"] == RETIRED

    def test_an_ENABLED_bucket_outranks_the_pooled_verdict(self, home):
        """If a bucket is actually open the loop is live whatever the
        pooled number says — retiring it would hide a steering site."""
        _w(home, "system/foresight/gate.json",
           {"buckets": {"b": {"enabled": True}},
            "discrimination": {"fail_n": 40, "fail_hits": 1, "ok_n": 40, "ok_hits": 20}})
        assert _row(yield_all(home)["rows"],
                    "foresight.gate")["status"] != RETIRED

    def test_retired_is_NOT_counted_as_barren_or_blocked(self, home):
        """The whole point: it must stop appearing on the actionable
        lists an operator triages from."""
        self._gate(home, {"fail_n": 14, "fail_hits": 1, "ok_n": 747,
                          "ok_hits": 80})
        r = yield_all(home)
        assert "foresight.gate" not in r["barren"]
        assert "foresight.gate" not in r["blocked"]
        assert r["retired"] == ["foresight.gate"]

    def test_it_keeps_the_pooled_ARITHMETIC_in_the_note(self, home):
        """A retirement with no numbers beside it is an assertion."""
        self._gate(home, {"fail_n": 14, "fail_hits": 1, "ok_n": 747, "ok_hits": 80,
                          "verdict": "rows it claims will FAIL fail 7.1%"})
        note = _row(yield_all(home)["rows"], "foresight.gate")["note"]
        assert "7.1%" in note and "RETIRED" in note
        # ⚠ AND THE INTERVAL. A verdict with no interval beside it cannot
        # be contradicted by any number on the row — which is how the
        # round-1 version presented an undetectable difference as settled.
        assert "1/14" in note and "interval tops out at" in note

    def test_the_note_does_not_double_punctuate(self, home):
        self._gate(home, {"fail_n": 14, "fail_hits": 1, "ok_n": 747, "ok_hits": 80,
                          "verdict": "only a better index does."})
        assert ".. RETIRED" not in \
            _row(yield_all(home)["rows"], "foresight.gate")["note"]


class TestRetiredSortsWithTheSettledStates:
    def test_status_is_a_known_state(self):
        assert _yield_status(YieldResult(minted=1, invoked=0,
                                         status=RETIRED)) == RETIRED

    def test_it_sorts_below_every_actionable_state(self, home):
        _w(home, "system/foresight/gate.json",
           {"buckets": {"b": {"enabled": False}},
            "discrimination": {"fail_n": 40, "fail_hits": 1, "ok_n": 40, "ok_hits": 20}})
        rows = yield_all(home)["rows"]
        names = [r["name"] for r in rows]
        actionable = [r["name"] for r in rows
                      if r["status"] not in (GATED, RETIRED, YIELDING)]
        for a in actionable:
            assert names.index(a) < names.index("foresight.gate")


# ══════════════════════════════════════════════════════════════════════
# 5. The rendered view
# ══════════════════════════════════════════════════════════════════════
class TestRender:
    def test_a_spread_line_is_printed_for_stores_that_have_one(self, home):
        _w(home, "system/memory/skills_playbook.json",
           [{"retrievals": 30} for _ in range(6)])
        assert "retrieval spread:" in render_yield(home)

    def test_no_spread_line_is_invented_for_stores_without_one(self, home):
        _w(home, "system/foresight/gate.json", {"buckets": {}})
        out = [ln for ln in render_yield(home).splitlines()
               if "retrieval spread" in ln]
        assert out == []

    def test_the_all_clear_names_loops_that_are_NOT_in_service(
            self, home, monkeypatch):
        """A summary that contradicts the rows above it is worse than none.

        Driven through `yield_all` rather than seven store files, because
        the property under test is the RENDER's closing line: before
        RETIRED existed the settled states never coincided with an
        otherwise-empty finding list, so 'every probed loop has a live
        consumer' was printable on a view whose top two rows were a
        retired loop and a switched-off one.
        """
        def _fake(_home=None):
            def _r(name, status):
                return {"name": name, "source": "s", "status": status,
                        "minted": 1, "activated": 1, "invoked": 1,
                        "age_h": None, "note": "", "derived_zero": "",
                        "activated_means": "a", "invoked_means": "i",
                        "spread": None}
            return {"rows": [_r("foresight.gate", RETIRED),
                             _r("evolve.candidates", GATED),
                             _r("lessons.playbook", YIELDING)],
                    "n_probes": 3, "barren": [], "blocked": [],
                    "unmeasured": [], "gaps": [], "empty": [],
                    "retired": ["foresight.gate"], "concentrated": []}
        monkeypatch.setattr(L, "yield_all", _fake)
        out = render_yield(home)
        assert "every probed loop has a live consumer" not in out
        assert "not in service" in out
        assert "foresight.gate" in out and "evolve.candidates" in out

    def test_the_all_clear_still_fires_when_everything_IS_in_service(
            self, home, monkeypatch):
        """The negative control: without it the branch above could be an
        unconditional rewrite that never prints the all-clear at all."""
        def _fake(_home=None):
            return {"rows": [{"name": "lessons.playbook", "source": "s",
                              "status": YIELDING, "minted": 1,
                              "activated": 1, "invoked": 1, "age_h": None,
                              "note": "", "derived_zero": "",
                              "activated_means": "a", "invoked_means": "i",
                              "spread": None}],
                    "n_probes": 1, "barren": [], "blocked": [],
                    "unmeasured": [], "gaps": [], "empty": [],
                    "retired": [], "concentrated": []}
        monkeypatch.setattr(L, "yield_all", _fake)
        assert "every probed loop has a live consumer" in render_yield(home)

    def test_a_concentrated_loop_says_the_remedy_is_the_RETRIEVER(self, home):
        _w(home, "system/memory/skills_playbook.json",
           [{"retrievals": 200}] + [{"retrievals": 0} for _ in range(9)])
        out = render_yield(home)
        assert "CONSUMED BUT COLLAPSED" in out
        assert "the RETRIEVER, not more" in out

    def test_a_collapsed_loop_SUPPRESSES_the_all_clear(self, home, monkeypatch):
        """A view cannot say 'every probed loop has a live consumer' while
        a row above it reads `concentrated`. Pinned separately from the
        line itself: a mutation removing `concentrated` from the finding
        list left every other test green, because the concentrated LINE
        still printed — the contradiction was two lines apart."""
        def _fake(_home=None):
            return {"rows": [{"name": "lessons.playbook", "source": "s",
                              "status": YIELDING, "minted": 9,
                              "activated": 1, "invoked": 90, "age_h": None,
                              "note": "", "derived_zero": "",
                              "activated_means": "a", "invoked_means": "i",
                              "spread": {"n": 9, "total": 90, "top1": 90,
                                         "nonzero": 1, "top1_share": 1.0,
                                         "coverage": 1 / 9,
                                         "entropy_ratio": 0.0,
                                         "verdict": CONCENTRATED,
                                         "why": "one item takes all"}}],
                    "n_probes": 1, "barren": [], "blocked": [],
                    "unmeasured": [], "gaps": [], "empty": [],
                    "retired": [], "concentrated": ["lessons.playbook"]}
        monkeypatch.setattr(L, "yield_all", _fake)
        out = render_yield(home)
        assert "CONSUMED BUT COLLAPSED" in out
        assert "every probed loop has a live consumer" not in out

    def test_an_underpowered_loop_is_NOT_called_collapsed(self, home):
        """The §4CE failure, rebuilt here, would be exactly this."""
        _w(home, "system/memory/skills_playbook.json",
           [{"retrievals": 3}] + [{"retrievals": 0} for _ in range(9)])
        out = render_yield(home)
        assert "CONSUMED BUT COLLAPSED" not in out
        assert UNDERPOWERED in out

    def test_the_retired_mark_is_eight_chars_like_every_other(self, home):
        """A short mark shifts the whole row's columns left."""
        from ghost_agent.core.liveness import BARREN as _B
        _w(home, "system/foresight/gate.json",
           {"buckets": {"b": {"enabled": False}},
            "discrimination": {"fail_n": 40, "fail_hits": 1, "ok_n": 40, "ok_hits": 20}})
        line = next(ln for ln in render_yield(home).splitlines()
                    if "foresight.gate" in ln and RETIRED in ln)
        assert line.index("foresight.gate") == 8

    def test_render_never_raises_on_a_bare_home(self, tmp_path):
        assert "LOOP YIELD" in render_yield(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# 6. The feature cannot be silently deleted
# ══════════════════════════════════════════════════════════════════════
class TestAbsenceIsDetectable:
    """§4CS item B shipped a suite where emptying YIELD_PROBES left all 20
    tests green. A test that cannot fail when the feature is gone is not a
    test of the feature."""

    def test_at_least_four_probes_carry_a_real_spread(self, home):
        _w(home, "system/memory/skills_playbook.json",
           [{"retrievals": 30} for _ in range(6)])
        _w(home, "system/memory/auto_skills.json",
           {str(i): {"retrievals": 30} for i in range(6)})
        _w(home, "system/memory/acquired_skills/skills_registry.json",
           {str(i): {"usage_count": 30, "status": "active"} for i in range(6)})
        _w(home, "system/memory/composed_skills/composed_skills.json",
           {f"auto_{i}": {"usage_count": 30, "status": "active",
                          "trigger_description": "auto-mined from"}
            for i in range(6)})
        with_spread = [r["name"] for r in yield_all(home)["rows"]
                       if r["spread"] is not None]
        assert len(with_spread) >= 4, with_spread

    def test_the_concentrated_bucket_is_reachable(self, home):
        _w(home, "system/memory/skills_playbook.json",
           [{"retrievals": 500}] + [{"retrievals": 1} for _ in range(9)])
        assert yield_all(home)["concentrated"] == ["lessons.playbook"]

    def test_the_spread_verdicts_are_all_reachable(self, home):
        seen = {
            _spread([10] * 10).verdict,
            _spread([100, 0, 0, 0, 0]).verdict,
            _spread([1, 1]).verdict,
            _spread([5]).verdict,
        }
        assert seen == {SPREAD_OK, CONCENTRATED, UNDERPOWERED, UNDEFINED}


class TestTop1IsTheMAXNotTheFirst:
    """⚠ EVERY FIXTURE IN THIS FILE PUT THE BUSIEST ITEM FIRST, so
    `top1 = max(counts)` -> `counts[0]` survived all 71 tests. Real
    stores are appended in MINT order, not retrieval order, so the
    busiest item is in an arbitrary position — and under the mutant the
    published arXiv:2604.27003 collapse signature is silently
    reclassified as healthy."""

    def test_the_busiest_item_LAST_is_still_found(self):
        sp = _spread([1] * 9 + [100])
        assert sp.top1 == 100
        assert sp.top1_share == pytest.approx(100 / 109, abs=1e-3)
        assert sp.verdict == CONCENTRATED, (
            "a 92%-collapsed store read as healthy because the busiest "
            "item was not first")

    def test_the_busiest_item_in_the_MIDDLE_is_still_found(self):
        sp = _spread([1, 1, 200, 1, 1])
        assert sp.top1 == 200 and sp.verdict == CONCENTRATED

    @pytest.mark.parametrize("pos", range(6))
    def test_position_does_not_change_the_verdict(self, pos):
        counts = [2] * 6
        counts[pos] = 300
        sp = _spread(counts)
        assert sp.top1 == 300 and sp.verdict == CONCENTRATED


class TestTimestampAndLabelChannelRegressions:
    """Two guards a mutation showed were unpinned."""

    def test_a_macro_with_an_ISO_last_used_does_not_break_its_row(self, home):
        """⚠ `_yield_macros` used a raw `float()` where every peer probe
        uses `_parse_ts`. Fed the ISO stamp that `auto_skills.json`
        writes for the same concept, the probe raised ValueError and a
        populated 29-macro store rendered `no_source` — a live store
        reported as a MISSING one."""
        import ghost_agent.core.liveness as _L
        _w(home, "system/memory/composed_skills/composed_skills.json",
           {"auto_a": {"usage_count": 4, "status": "active",
                       "trigger_description": "auto-mined from",
                       "last_used": "2026-08-24T10:00:00Z"}})
        import unittest.mock as _m
        with _m.patch.object(_L, "_macro_marks",
                             lambda: ("auto-mined from", "gr")):
            row = _row(yield_all(home)["rows"], "macros.auto_mined")
        assert row["status"] != "no_source", row["note"]
        assert row["minted"] == 1 and row["invoked"] == 4
        assert row["age_h"] is not None, "the ISO stamp was not parsed"

    def test_a_float_last_used_still_works(self, home):
        """Negative control — the fix must not break the live shape."""
        import time
        import unittest.mock as _m
        import ghost_agent.core.liveness as _L
        _w(home, "system/memory/composed_skills/composed_skills.json",
           {"auto_a": {"usage_count": 4, "status": "active",
                       "trigger_description": "auto-mined from",
                       "last_used": time.time() - 3600}})
        with _m.patch.object(_L, "_macro_marks",
                             lambda: ("auto-mined from", "gr")):
            row = _row(yield_all(home)["rows"], "macros.auto_mined")
        assert row["age_h"] is not None and 0.5 < row["age_h"] < 2.0

    def test_operator_overlay_counts_as_a_HUMAN_label(self, home):
        """`operator_overlay` is documented as a manual out-of-process
        edit — a human channel that failed both the prefix test and the
        allow-list, silently dropping rows from the only ground truth
        the rubric-shadow row has."""
        from ghost_agent.core.liveness import _human_labels
        p = home / "system" / "trajectories" / "corrections.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(json.dumps(r) for r in [
            {"trajectory_id": "a", "outcome": "passed",
             "source": "operator_overlay"},
            {"trajectory_id": "b", "outcome": "failed", "source": "feedback"},
            {"trajectory_id": "c", "outcome": "passed",
             "source": "machine_verifier"},
        ]) + "\n")
        got = _human_labels(home)
        assert got == {"a": "passed", "b": "failed"}, got

    def test_a_MACHINE_verdict_is_still_excluded(self, home):
        """The other half: scoring a judge against another judge measures
        their shared blind spot."""
        from ghost_agent.core.liveness import _human_labels
        p = home / "system" / "trajectories" / "corrections.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"trajectory_id": "z", "outcome": "passed",
             "source": "verifier"}) + "\n")
        assert _human_labels(home) == {}


class TestRetirementNeedsBOTH_SignAndPower:
    """⚠ ROUND 1 HAD THE SIGN AND NO POWER; ROUND 2's FIX HAD POWER AND
    DROPPED THE SIGN. Both were wrong, in opposite directions, and a
    reviewer executed each.

    Round 1: `spread <= 0` with both denominators merely truthy retired a
    loop on ONE row per arm — a terminal 'nothing owed' verdict on an
    arithmetically undetectable difference, §4CE verbatim.

    Round 2's fix: a precision-only test RETIRED an index where
    predicted-fail rows fail 40% and predicted-ok rows fail 2% — spread
    +0.38, hugely predictive — and the row printed the builder's verdict
    ("discriminates in the right direction… but pooled precision is under
    the bar") directly above "SETTLED, nothing is owed". Two sentences in
    one string saying opposite things.

    They are DIFFERENT FINDINGS with DIFFERENT REMEDIES:
        spread <= 0  -> the INDEX is backwards; nothing to do but build a
                        better one. RETIRED.
        spread  > 0  -> the index works and the BAR is wrong for it. That
                        is owed work, so NOT retired.
    """

    def test_predictive_but_UNDER_THE_BAR_is_not_retired(self):
        """precision 0.40 vs ok_fail 0.02 — spread +0.38."""
        assert _anti_predictive(
            {"fail_n": 100, "fail_hits": 40,
             "ok_n": 500, "ok_hits": 10}) is False

    def test_anti_predictive_WITH_power_is_retired(self):
        assert _anti_predictive(
            {"fail_n": 100, "fail_hits": 5,
             "ok_n": 500, "ok_hits": 50}) is True

    def test_the_power_floor_applies_to_BOTH_legs(self):
        """Round 2 measured the asymmetry: retirement needed 3
        predicted-fail rows (0/3 has a Wilson upper of 0.562, under the
        0.60 bar) while a bucket needed 10 before it could OBJECT. Every
        tie broke toward 'settled'."""
        thin = {"fail_n": 3, "fail_hits": 0, "ok_n": 500, "ok_hits": 250}
        assert _anti_predictive(thin, min_fail_n=10) is False
        assert _anti_predictive(thin, min_fail_n=3) is True

    def test_the_live_gate_still_retires(self):
        """The corrected rule must not undo the real finding."""
        assert _anti_predictive(
            {"fail_n": 14, "fail_hits": 1, "ok_n": 564, "ok_hits": 60},
            min_fail_n=10) is True

    def test_ok_hits_is_REQUIRED_now(self):
        """The sign cannot be computed without it, and guessing would
        reintroduce the proxy."""
        assert _anti_predictive(
            {"fail_n": 100, "fail_hits": 5, "ok_n": 500}) is False

    def test_the_note_does_not_claim_an_UNRUN_bucket_check(self, home):
        """⚠ 'No bucket with at least 10 predicted-fail rows can still
        reach the bar' read as 'we checked and none qualifies' — but the
        live gate has ZERO buckets with 10+ rows, so none was ELIGIBLE to
        be checked. An unrun check presented as a passed one."""
        _w(home, "system/foresight/gate.json",
           {"params": {"min_fail_precision": 0.60, "min_fail_n": 10},
            "buckets": {"thin": {"enabled": False, "fail_n": 2,
                                 "fail_hits": 0}},
            "discrimination": {"fail_n": 100, "fail_hits": 5,
                               "ok_n": 500, "ok_hits": 50}})
        note = _row(yield_all(home)["rows"], "foresight.gate")["note"]
        assert "No bucket has 10+ predicted-fail rows yet" in note
        assert "could be assessed individually" in note

    def test_the_note_SAYS_SO_when_buckets_WERE_assessed(self, home):
        """The other half — otherwise the sentence above could be
        unconditional."""
        _w(home, "system/foresight/gate.json",
           {"params": {"min_fail_precision": 0.60, "min_fail_n": 10},
            "buckets": {"real": {"enabled": False, "fail_n": 40,
                                 "fail_hits": 2}},
            "discrimination": {"fail_n": 100, "fail_hits": 5,
                               "ok_n": 500, "ok_hits": 50}})
        note = _row(yield_all(home)["rows"], "foresight.gate")["note"]
        assert "Of the buckets with 10+ predicted-fail rows" in note


class TestTheFloorNeverPrintsAVacuousZero:
    """⚠ A brute force found 816 (n, total) pairs in the underpowered
    region still printing "cannot fall below 0%" — the exact vacuous
    sentence the comment block above the code cites as the defect,
    reintroduced by ROUNDING. `f"{0.005:.0%}"` is "0%"."""

    @pytest.mark.parametrize("counts", [
        [1] * 200 + [0],          # n=201, total=200 -> floor 0.5%
        [1] * 200,                # n=200, total=200 -> floor 0.5%
        [1] * 300 + [0] * 50,     # n=350, total=300
    ])
    def test_a_sub_one_percent_floor_is_not_printed_as_zero(self, counts):
        sp = _spread(counts)
        assert sp.verdict == UNDERPOWERED
        assert "cannot fall below 0%" not in sp.why, sp.why
        assert "cannot fall below <1%" in sp.why

    def test_a_REAL_percentage_is_still_printed_normally(self):
        """Negative control — otherwise "<1%" could be unconditional."""
        sp = _spread([24, 0, 0, 0, 0])
        assert "cannot fall below 21%" in sp.why

    def test_no_underpowered_shape_prints_zero_percent(self):
        """The brute force, as a pin: sweep the region and assert the
        vacuous sentence is unreachable."""
        for n in range(2, 60):
            for total in range(1, 5 * n):
                sp = _spread([total] + [0] * (n - 1))
                if sp.verdict == UNDERPOWERED:
                    assert "below 0%" not in sp.why, (n, total, sp.why)


class TestTheINTERVAL_IsWhatMakesRetirementHonest:
    """⚠ TWO LOAD-BEARING MUTANTS SURVIVED THE WHOLE SUITE, and a
    reviewer found them by re-running the harness on the CURRENT tree
    after the round-3 edits. My "42/42" was measured BEFORE those edits
    and restated afterwards — `restated-is-not-checked`, on the number
    that certifies everything else.

    Both survivors attack the same property: retirement must rest on an
    INTERVAL, not a point. That is the entire v1→v3 argument (a terminal
    "nothing owed" verdict on an undetectable difference is §4CE), and
    nothing tested it — because the only LIVE input, 1/14, gives 0.315
    by Wilson, 0.206 by Wald and 0.071 as a point estimate, all three
    under the 0.60 bar. `verify-cannot-distinguish`: one input cannot
    separate three formulas that agree on it.
    """

    def test_the_POINT_estimate_is_not_enough_to_retire(self):
        """5/10 = 0.50 is under the 0.60 bar, but its interval reaches
        0.76 — the index may yet clear the bar, so retiring is a verdict
        the data does not support."""
        disc = {"fail_n": 10, "fail_hits": 5, "ok_n": 100, "ok_hits": 60}
        assert disc["fail_hits"] / disc["fail_n"] < 0.60, "point IS under"
        from ghost_agent.core.liveness import _wilson_upper
        assert _wilson_upper(5, 10) > 0.60, "but the interval is NOT"
        assert _anti_predictive(disc, min_fail_n=10) is False

    def test_and_a_TIGHT_interval_under_the_bar_still_retires(self):
        """Negative control: the guard must not refuse every retirement,
        or the state becomes unreachable."""
        assert _anti_predictive(
            {"fail_n": 100, "fail_hits": 5, "ok_n": 500, "ok_hits": 100},
            min_fail_n=10) is True

    def test_WILSON_not_WALD(self):
        """At 3/10 a Wald upper bound is 0.584 — under the bar, so Wald
        would RETIRE — while Wilson gives 0.603, over it. Wald runs off
        the end of the scale at small k, which is exactly where these
        denominators live; `_wilson_upper`'s docstring says so and
        nothing tested it."""
        from ghost_agent.core.liveness import _wilson_upper
        p = 3 / 10
        wald = p + 1.96 * ((p * (1 - p) / 10) ** 0.5)
        assert wald < 0.60 < _wilson_upper(3, 10), (
            "this input no longer separates Wald from Wilson — pick "
            "another, or the pin below proves nothing")
        assert _anti_predictive(
            {"fail_n": 10, "fail_hits": 3, "ok_n": 100, "ok_hits": 40},
            min_fail_n=10) is False

    def test_wilson_is_wider_than_wald_at_the_extremes(self):
        """The property in general, not just at one input."""
        from ghost_agent.core.liveness import _wilson_upper
        for k, n in [(0, 5), (0, 14), (1, 14), (1, 20), (2, 30)]:
            p = k / n
            wald = min(1.0, p + 1.96 * ((p * (1 - p) / n) ** 0.5))
            assert _wilson_upper(k, n) > wald, (k, n)


class TestARetiredArtifactIsNotAnEmptyLoop:
    """⚠ Retiring `planning.decompose` (2026-08-24, measured worse than
    its own seed) left zero live `.json` files, and the row rendered
    `empty · minted 0 · invoked 135` — "this loop has minted NOTHING
    yet" printed beside 135 recorded loads of the thing it minted.

    The suffix filter is right: a retired artifact must not count as
    live output. The STATE was wrong. EMPTY's own contract is "produced
    nothing", and "the gate withdrew everything it produced" is the
    opposite finding with the opposite remedy."""

    def _row(self, home):
        return _row(yield_all(home)["rows"], "prompts.gepa")

    def _log(self, home, n):
        import time
        lg = home / "system" / "ghost-agent.log"
        lg.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lg.write_text("\n".join(
            f"{stamp} - GhostAgent - INFO - GEPA: loaded tuned instruction x"
            for _ in range(n)) + "\n")

    def test_only_withdrawn_artifacts_reads_RETIRED_not_EMPTY(self, home):
        d = home / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        (d / "planning.decompose.json.retired-4cw").write_text("{}")
        (d / "verifier.enumerate.json.noop.retired").write_text("{}")
        self._log(home, 3)
        r = self._row(home)
        assert r["status"] == RETIRED, r["status"]
        assert "GATE WORKING" in r["note"]
        assert "minted nothing" in r["note"]

    def test_a_GENUINELY_empty_optim_dir_still_reads_EMPTY(self, home):
        """Negative control: a fresh box has produced nothing, and that
        must not be dressed up as a working gate."""
        (home / "system" / "optim").mkdir(parents=True, exist_ok=True)
        assert self._row(home)["status"] == "empty"

    def test_a_LIVE_artifact_outranks_the_withdrawn_ones(self, home):
        import json as _j
        d = home / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        (d / "planning.decompose.json.retired-4cw").write_text("{}")
        (d / "planning.decompose.json").write_text(
            _j.dumps({"optimized_instruction": "do the thing"}))
        r = self._row(home)
        assert r["status"] != RETIRED
        assert r["minted"] == 1 and r["activated"] == 1

    def test_the_note_names_the_withdrawn_files(self, home):
        d = home / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        (d / "planning.decompose.json.retired-4cw").write_text("{}")
        self._log(home, 2)
        assert "planning.decompose.json.retired-4cw" in self._row(home)["note"]


class TestTwoViewsMustNotContradictEachOther:
    """⚠ After the 2026-08-24 retirement, ONE SCREEN said both things
    about ONE loop: SUBSYSTEM LIVENESS printed
    `gepa.applies fired n=28 last 12.3h ago` fifty lines above LOOP
    YIELD's `† END prompts.gepa retired`. Both were literally true — the
    window is 168h and the loads are real history — and the older probe
    would keep reading FIRED for another seven days about a loop that is
    settled dead. Neither view knew about the other.

    A count inside a window is not wrong. A count presented without the
    fact that its subject has been WITHDRAWN is."""

    def _probe(self, home):
        from ghost_agent.core.liveness import probe_all
        return next(r for r in probe_all(home)["rows"]
                    if r["name"] == "gepa.applies")

    def _log(self, home, n):
        p = home / "system" / "ghost-agent.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        # Lines carry a leading 'YYYY-MM-DD HH:MM:SS' stamp — `_log_probe`
        # parses it to apply the window, so a bare HH:MM:SS fixture is
        # silently outside every window and counts zero.
        import time
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        p.write_text("\n".join(
            f"{stamp} - GhostAgent - INFO - GEPA: loaded tuned instruction "
            f"for 'x' (10 chars, sha deadbeef)" for _ in range(n)) + "\n")

    def test_loads_of_a_WITHDRAWN_artifact_are_annotated(self, home):
        d = home / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        (d / "planning.decompose.json.retired-4cw").write_text("{}")
        self._log(home, 3)
        note = self._probe(home)["note"] or ""
        assert "HISTORICAL" in note
        assert "no longer served" in note
        assert "RETIRED" in note

    def test_loads_of_a_LIVE_artifact_are_NOT_annotated(self, home):
        """Negative control — the annotation must not fire while an
        artifact is actually being served."""
        import json as _j
        d = home / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        (d / "planning.decompose.json.retired-4cw").write_text("{}")
        (d / "planning.decompose.json").write_text(
            _j.dumps({"optimized_instruction": "live"}))
        self._log(home, 3)
        assert "HISTORICAL" not in (self._probe(home)["note"] or "")

    def test_no_annotation_when_there_are_no_loads(self, home):
        d = home / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        (d / "x.json.retired-4cw").write_text("{}")
        self._log(home, 0)
        assert "HISTORICAL" not in (self._probe(home)["note"] or "")

    def test_the_probe_still_counts_correctly(self, home):
        """The annotation must not disturb the number it annotates."""
        d = home / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        (d / "x.json.retired-4cw").write_text("{}")
        self._log(home, 7)
        assert self._probe(home)["count"] == 7


class TestTheDerivedZeroReachesTheOperator:
    """⚠ `derived_zero` reached the row MARK and the `blocked` summary and
    NOTHING ELSE, so on any non-BARREN row the one line explaining the
    zero was computed and thrown away. `mining.failure_envs` knew
    "staging is deliberately not arming" and never said it. An
    explanation that survives only in the JSON payload is one the
    operator does not get."""

    def test_an_UNMEASURED_row_prints_why_its_zero_is_derived(self, home):
        _w(home, "system/optim/mined_envs/ghost_failures.jsonl", {})
        p = home / "system" / "optim" / "mined_envs" / "ghost_failures.jsonl"
        p.write_text(json.dumps({
            "bank": "ghost_failures", "item_id": "m1", "cluster": "c",
            "challenge": "q", "setup_script": "", "validation_script": "v",
            "graded_on": "final_response", "mining_epoch": "e1",
            "reference_answer": "42"}) + "\n")
        out = render_yield(home)
        assert "why the zero:" in out
        assert "deliberately not arming" in out

    def test_a_BARREN_row_does_NOT_double_print_it(self, home):
        """BARREN already surfaces it through the ⊘ mark and the
        `blocked` summary; printing it a third time is noise."""
        _w(home, "system/foresight/gate.json",
           {"params": {"min_fail_precision": 0.60, "min_fail_n": 10},
            "buckets": {"b": {"enabled": False}},
            "discrimination": {"fail_n": 2, "fail_hits": 0, "ok_n": 5,
                               "ok_hits": 1}})
        lines = [l for l in render_yield(home).splitlines()
                 if "no steering site exists" in l]
        assert len(lines) <= 2, lines

    def test_a_row_with_no_derived_zero_prints_no_such_line(self, home):
        _w(home, "system/memory/skills_playbook.json",
           [{"retrievals": 30} for _ in range(6)])
        out = [l for l in render_yield(home).splitlines()
               if "lessons.playbook" in l or "why the zero" in l]
        assert not any("why the zero" in l for l in out)

    def test_a_retired_loop_with_an_UNREADABLE_log_claims_no_count(self, home):
        """⚠ The RETIRED branch returned BEFORE the NO_SOURCE check, so an
        unreadable log became `invoked 0` with the note "the 0 loads are
        historical" — a FABRICATED zero. That is the missing-vs-empty
        conflation this module's own docstring calls "the finding", and a
        live artifact with no log already handles it correctly."""
        d = home / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        (d / "planning.decompose.json.retired-4cw").write_text("{}")
        # no ghost-agent.log at all
        r = _row(yield_all(home)["rows"], "prompts.gepa")
        assert r["status"] == RETIRED
        assert r["invoked"] is None, (
            "an unreadable log was reported as a measured zero")
        assert "no load count is claimed" in r["note"]

    def test_but_a_READABLE_log_still_gives_the_count(self, home):
        d = home / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        (d / "planning.decompose.json.retired-4cw").write_text("{}")
        import time
        lg = home / "system" / "ghost-agent.log"
        lg.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lg.write_text("\n".join(
            f"{stamp} - GhostAgent - INFO - GEPA: loaded tuned instruction x"
            for _ in range(4)) + "\n")
        r = _row(yield_all(home)["rows"], "prompts.gepa")
        assert r["status"] == RETIRED and r["invoked"] == 4
