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

**2. RETIRED.** the (since removed) foresight gate read BARREN — "produces artifacts nobody
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
    YIELDING, Spread, YieldResult, _eligible_counts,
    _spread, _yield_status, render_yield, yield_all,
)


@pytest.fixture
def home(tmp_path):
    for rel in ("system/memory/acquired_skills",
                "system/evolve", "system/optim"):
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
        combined": 5 is not more than 5). Two active skills is an
        ordinary shape."""
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
        """THE INVARIANT. A skill
        degraded after being used still contributed to `invoked`. Dropping
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

    @pytest.mark.parametrize("name", [
        "skills.acquired",          # ("macros.auto_mined" went with the macro engine, §4TA)
    ])
    def test_total_equals_invoked(self, home, name):
        self._stores(home)
        row = _row(yield_all(home)["rows"], name)
        assert row["spread"] is not None, f"{name} lost its spread"
        assert row["spread"]["total"] == row["invoked"], (
            f"{name}: spread describes a different population than its row")


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
    def test_gepa_carries_no_spread(self, home):
        # (the evolve.candidates probe left with ghost_agent/evolve)
        rows = yield_all(home)["rows"]
        assert _row(rows, "prompts.gepa")["spread"] is None, (
            "prompts.gepa has no per-item counter — a spread here would be "
            "fabricated")


class TestRetiredSortsWithTheSettledStates:
    def test_status_is_a_known_state(self):
        assert _yield_status(YieldResult(minted=1, invoked=0,
                                         status=RETIRED)) == RETIRED

# ══════════════════════════════════════════════════════════════════════
# 5. The rendered view
# ══════════════════════════════════════════════════════════════════════
class TestRender:
    def test_a_spread_line_is_printed_for_stores_that_have_one(self, home):
        _w(home, "system/memory/acquired_skills/skills_registry.json",
           {str(i): {"usage_count": 30, "status": "active"} for i in range(6)})
        assert "retrieval spread:" in render_yield(home)

    def test_no_spread_line_is_invented_for_stores_without_one(self, home):
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
            return {"rows": [_r("prompts.gepa", RETIRED),
                             _r("evolve.candidates", GATED),
                             _r("skills.acquired", YIELDING)],
                    "n_probes": 3, "barren": [], "blocked": [],
                    "unmeasured": [], "gaps": [], "empty": [],
                    "retired": ["prompts.gepa"], "concentrated": []}
        monkeypatch.setattr(L, "yield_all", _fake)
        out = render_yield(home)
        assert "every probed loop has a live consumer" not in out
        assert "not in service" in out
        assert "prompts.gepa" in out and "evolve.candidates" in out

    def test_the_all_clear_still_fires_when_everything_IS_in_service(
            self, home, monkeypatch):
        """The negative control: without it the branch above could be an
        unconditional rewrite that never prints the all-clear at all."""
        def _fake(_home=None):
            return {"rows": [{"name": "skills.acquired", "source": "s",
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
        _w(home, "system/memory/acquired_skills/skills_registry.json",
           {str(i): {"usage_count": 200 if i == 0 else 0, "status": "active"} for i in range(10)})
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
            return {"rows": [{"name": "skills.acquired", "source": "s",
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
                    "retired": [], "concentrated": ["skills.acquired"]}
        monkeypatch.setattr(L, "yield_all", _fake)
        out = render_yield(home)
        assert "CONSUMED BUT COLLAPSED" in out
        assert "every probed loop has a live consumer" not in out

    def test_an_underpowered_loop_is_NOT_called_collapsed(self, home):
        """The §4CE failure, rebuilt here, would be exactly this."""
        _w(home, "system/memory/acquired_skills/skills_registry.json",
           {str(i): {"usage_count": 3 if i == 0 else 0, "status": "active"} for i in range(10)})
        out = render_yield(home)
        assert "CONSUMED BUT COLLAPSED" not in out
        assert UNDERPOWERED in out

    def test_the_retired_mark_is_eight_chars_like_every_other(self, home):
        """A short mark shifts the whole row's columns left."""
        d = home / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        (d / "planning.decompose.json.retired-4cw").write_text("{}")
        line = next(ln for ln in render_yield(home).splitlines()
                    if "prompts.gepa" in ln and RETIRED in ln)
        assert line.index("prompts.gepa") == 8

    def test_render_never_raises_on_a_bare_home(self, tmp_path):
        assert "LOOP YIELD" in render_yield(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# 6. The feature cannot be silently deleted
# ══════════════════════════════════════════════════════════════════════
class TestAbsenceIsDetectable:
    """§4CS item B shipped a suite where emptying YIELD_PROBES left all 20
    tests green. A test that cannot fail when the feature is gone is not a
    test of the feature."""

    def test_both_per_item_probes_carry_a_real_spread(self, home):
        # (three until the macro row was removed, §4TA; a macro registry left
        # on disk is not read and adds no row)
        _w(home, "system/memory/acquired_skills/skills_registry.json",
           {str(i): {"usage_count": 30, "status": "active"} for i in range(6)})
        _w(home, "system/memory/acquired_skills/skills_registry.json",
           {str(i): {"usage_count": 30, "status": "active"} for i in range(6)})
        _w(home, "system/memory/composed_skills/composed_skills.json",
           {f"auto_{i}": {"usage_count": 30, "status": "active",
                          "trigger_description": "auto-mined from"}
            for i in range(6)})
        with_spread = sorted(r["name"] for r in yield_all(home)["rows"]
                             if r["spread"] is not None)
        assert with_spread == ["skills.acquired"], with_spread

    def test_the_concentrated_bucket_is_reachable(self, home):
        _w(home, "system/memory/acquired_skills/skills_registry.json",
           {str(i): {"usage_count": 500 if i == 0 else 1, "status": "active"} for i in range(10)})
        assert yield_all(home)["concentrated"] == ["skills.acquired"]

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

    def test_an_UNMEASURED_row_prints_why_its_zero_is_derived(self, home, monkeypatch):
        # §4NM: the `mining.failure_envs` probe that carried this derived
        # zero left with env mining; the render path is pinned through a
        # registered probe whose zero is derived the same way.
        probe = L.YieldProbe(
            "test.derived", "system/test.json",
            lambda _home: YieldResult(
                minted=3, activated=0, invoked=None,
                derived_zero="staging is deliberately not arming"),
            activated_means="promoted", invoked_means="consumed")
        monkeypatch.setattr(L, "YIELD_PROBES", list(L.YIELD_PROBES) + [probe])
        out = render_yield(home)
        assert "why the zero:" in out
        assert "deliberately not arming" in out

    def test_a_row_with_no_derived_zero_prints_no_such_line(self, home):
        _w(home, "system/memory/acquired_skills/skills_registry.json",
           {str(i): {"usage_count": 30, "status": "active"} for i in range(6)})
        out = [l for l in render_yield(home).splitlines()
               if "skills.acquired" in l or "why the zero" in l]
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
