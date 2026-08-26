"""§4DA — the significance requirement, ported to the tool-description gate.

§4CY fixed `optim/ab_eval.py`'s ship rule and left two siblings carrying the
one it replaced. This is the second of the three:
`scripts/optimize_tool_descriptions.py:634` read
`ships = valid and aggregate_ok and delta > args.min_delta` — a margin with no
significance test, which under the null promotes 25-40% of the time because the
smallest swing clearing the bar is one or two flipped replays.

⚠ McNEMAR IS RIGHT HERE AND WRONG IN §4CZ, and the difference is the point.
Both arms replay the SAME fixture list in order, so fixture i is a matched
pair. §4CZ's live arms are different requests — unpaired — and need Fisher.
Reaching for whichever test is to hand produces a number for the wrong
question.

These pins drive `_ship_decision` and `_significance_floor` directly. Every
§4CY/§4CZ round that went wrong went wrong inside a harness that stubbed out
the component under test; there is nothing to stub here.
"""

import importlib.util
import sys

import pytest

from ghost_agent.optim import ab_eval


LAST_ADAPTER = {}


def _otd(register=True):
    """Load the script the way its consumers do.

    ⚠ `register=False` IS THE REAL CONSUMER SHAPE. Every existing test
    that touches this script does `module_from_spec` + `exec_module`
    WITHOUT putting it in `sys.modules`. Adding an `@dataclass` here made
    that import fail — `dataclasses` resolves annotations through
    `sys.modules[cls.__module__].__dict__`, which is then None — breaking
    3 tests and 14 collection errors elsewhere. My own helper registered
    the module, so my 23 pins all passed while the suite broke; only the
    full run caught it. The default stays True for convenience, and
    `test_it_imports_WITHOUT_sys_modules_registration` drives the other.
    """
    spec = importlib.util.spec_from_file_location(
        "otd_gate", "scripts/optimize_tool_descriptions.py")
    mod = importlib.util.module_from_spec(spec)
    if register:
        sys.modules["otd_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestItImportsTheWayItsConsumersLoadIt:
    def test_it_imports_WITHOUT_sys_modules_registration(self):
        """The import contract this script already had, which my own
        change broke and my own pins could not see."""
        mod = _otd(register=False)
        assert hasattr(mod, "_ship_decision")
        d = mod._ship_decision(
            [{"score": 1.0}], [{"score": 0.0}], min_delta=0.0,
            valid=True, aggregate_ok=True)
        assert d.incumbent_wins == 1


def _arms(n, inc_pass, cand_pass, *, overlap=0):
    """Two trajectory lists of length `n`, paired by index.

    `overlap` fixtures pass in BOTH arms (ties, which carry no information
    and must not move the statistic).
    """
    inc, cand = [], []
    for i in range(n):
        if i < overlap:
            inc.append({"score": 1.0}); cand.append({"score": 1.0})
        elif i < overlap + inc_pass:
            inc.append({"score": 1.0}); cand.append({"score": 0.0})
        elif i < overlap + inc_pass + cand_pass:
            inc.append({"score": 0.0}); cand.append({"score": 1.0})
        else:
            inc.append({"score": 0.0}); cand.append({"score": 0.0})
    return inc, cand


def _decide(mod, inc, cand, *, min_delta=0.02, valid=True,
            aggregate_ok=True, allow=False, min_usable=0):
    """⚠ NO `delta` ARGUMENT. `_ship_decision` derives the margin from the
    arms, so a test can no longer state a margin its own trajectories do
    not support — the round-1 helper defaulted to `delta=0.30` beside arms
    built for something else, and every pin that relied on the margin being
    cleared was asserting a number nothing had produced."""
    return mod._ship_decision(inc, cand, min_delta=min_delta,
                              valid=valid, aggregate_ok=aggregate_ok,
                              min_usable=min_usable,
                              allow_insignificant=allow)


class TestTheMarginAloneNoLongerShips:
    def test_a_TWO_replay_swing_does_not_ship(self):
        """The defect, driven. Two flipped replays out of 40 clears a 0.02
        margin five times over and is p=0.25 — noise."""
        mod = _otd()
        inc, cand = _arms(40, 0, 2, overlap=10)
        d = _decide(mod, inc, cand)
        assert d.cleared_margin is True, "the margin is cleared — the point"
        assert d.p_value == pytest.approx(0.25)
        assert d.ships is False

    def test_a_SIX_ZERO_sweep_ships(self):
        mod = _otd()
        inc, cand = _arms(40, 0, 6, overlap=10)
        d = _decide(mod, inc, cand)
        assert d.p_value == pytest.approx(0.015625)
        assert d.ships is True

    def test_the_FIVE_ZERO_boundary_ships(self):
        """One-sided, so 5-0 is exactly p=0.03125 — the smallest sweep the
        bar admits. Its admit side, not just its reject side."""
        mod = _otd()
        inc, cand = _arms(40, 0, 5, overlap=10)
        d = _decide(mod, inc, cand)
        assert d.p_value == pytest.approx(0.03125)
        assert d.ships is True

    def test_the_FOUR_ZERO_boundary_does_not(self):
        mod = _otd()
        inc, cand = _arms(40, 0, 4, overlap=10)
        d = _decide(mod, inc, cand)
        assert d.p_value == pytest.approx(0.0625)
        assert d.ships is False

    def test_a_LOSING_candidate_never_ships(self):
        mod = _otd()
        inc, cand = _arms(40, 8, 0, overlap=10)
        d = _decide(mod, inc, cand)
        assert d.ships is False and d.cleared_margin is False

    def test_TIES_do_not_move_the_statistic(self):
        """Only the replays where exactly one arm passed carry
        information — a fixture both arms get right must not dilute it."""
        mod = _otd()
        a = _decide(mod, *_arms(40, 0, 6, overlap=0))
        b = _decide(mod, *_arms(90, 0, 6, overlap=60))
        assert a.p_value == b.p_value
        assert a.discordant == b.discordant == 6


class TestTheOtherConditionsStillBind:
    def test_an_INVALID_candidate_never_ships_however_strong(self):
        mod = _otd()
        inc, cand = _arms(40, 0, 20, overlap=10)
        d = _decide(mod, inc, cand, valid=False)
        assert d.significant is True, "evidence is overwhelming"
        assert d.ships is False, "an invalid description shipped"

    def test_an_AGGREGATE_reject_never_ships_however_strong(self):
        mod = _otd()
        inc, cand = _arms(40, 0, 20, overlap=10)
        d = _decide(mod, inc, cand, aggregate_ok=False)
        assert d.significant is True
        assert d.ships is False

    def test_the_margin_bound_is_EXCLUSIVE(self):
        """A margin exactly equal to the bar is not more than it. The arms
        carry a 20/40 = 0.500 paired margin and overwhelming evidence, so
        only the boundary can decide the two calls apart."""
        mod = _otd()
        inc, cand = _arms(40, 0, 20, overlap=10)
        assert _decide(mod, inc, cand, min_delta=0.5).paired_delta == 0.5
        assert _decide(mod, inc, cand, min_delta=0.5).ships is False
        assert _decide(mod, inc, cand, min_delta=0.4999).ships is True

    def test_the_margin_is_read_from_the_ARGUMENT(self):
        """Twin-value guard: the arms' 0.500 margin is above a 0.05 bar and
        below a 0.60 one, so the parameter must be what decides."""
        mod = _otd()
        inc, cand = _arms(40, 0, 20, overlap=10)
        assert _decide(mod, inc, cand, min_delta=0.05).ships
        assert not _decide(mod, inc, cand, min_delta=0.60).ships


class TestTheBarIsTheSHARED_constant:
    def test_the_decision_FOLLOWS_ship_alpha(self, monkeypatch):
        """Move the constant. A 3-0 sweep is p=0.125: it must ship at a
        0.2 bar and not at the live 0.05 one — a hardcoded literal cannot
        do both."""
        mod = _otd()
        inc, cand = _arms(40, 0, 3, overlap=10)
        assert _decide(mod, inc, cand).ships is False
        monkeypatch.setattr(ab_eval, "SHIP_ALPHA", 0.2)
        assert _decide(mod, inc, cand).ships is True

    def test_the_floor_FOLLOWS_ship_alpha(self, monkeypatch):
        """Derivation, not the number: `assert floor == 5` cannot tell a
        derived 5 from a literal."""
        mod = _otd()
        assert mod._significance_floor() == 5
        monkeypatch.setattr(ab_eval, "SHIP_ALPHA", 0.2)
        assert mod._significance_floor() == 3

    def test_a_p_EXACTLY_at_the_bar_ships(self, monkeypatch):
        """⚠ THE ADMIT SIDE OF THE ALPHA BOUND, and it needs a DYADIC
        alpha to be reachable at all: exact McNemar p is always m/2^k, and
        0.05 = 1/20 carries a factor of 5, so `p == SHIP_ALPHA` can never
        happen at the live constant and `<=` vs `<` is indistinguishable
        there. At 0.03125 a 5-0 sweep lands exactly on the bar."""
        mod = _otd()
        monkeypatch.setattr(ab_eval, "SHIP_ALPHA", 0.03125)
        inc, cand = _arms(40, 0, 5, overlap=10)
        d = _decide(mod, inc, cand)
        assert d.p_value == pytest.approx(0.03125)
        assert d.ships is True, (
            "p exactly equal to the bar was treated as failing it")

    def test_an_unreachable_alpha_RAISES(self, monkeypatch):
        mod = _otd()
        monkeypatch.setattr(ab_eval, "SHIP_ALPHA", 0.0)
        with pytest.raises(ValueError, match="unreachable"):
            mod._significance_floor()


class TestMainDerivesTheFloorRatherThanHardcodingIt:
    """⚠ THE PRE-FLIGHT CALL SITE IS NOT REACHED BY THESE PINS, because
    driving `main()` would mean stubbing the whole optimizer — the exact
    harness shape that hid four defects across §4CY/§4CZ. An AST walk over
    the real source checks the property without pretending to run it: it
    survives reformatting, and it fails on the hardcode."""

    def _preflight_calls(self):
        import ast as _ast
        from pathlib import Path
        tree = _ast.parse(
            Path("scripts/optimize_tool_descriptions.py").read_text())
        found = []
        for node in _ast.walk(tree):
            if (isinstance(node, _ast.Assign)
                    and any(getattr(t, "id", "") == "_min_discordant"
                            for t in node.targets)):
                found.append(_ast.dump(node.value))
        return found

    def test_the_floor_is_computed_not_written(self):
        calls = self._preflight_calls()
        assert calls, "_min_discordant is never assigned in main()"
        for value in calls:
            assert "_significance_floor" in value, (
                f"the pre-flight floor is a literal, so it cannot follow "
                f"SHIP_ALPHA: {value}")
            assert "Constant" not in value, (
                f"the pre-flight floor is hardcoded: {value}")


class TestUnpairedArmsRefuseRatherThanGuess:
    def test_different_lengths_produce_NO_verdict(self):
        """⚠ Pairing by position across lists of different lengths would
        compare fixture i against fixture j and still yield a p."""
        mod = _otd()
        inc, cand = _arms(40, 0, 6, overlap=10)
        d = _decide(mod, inc, cand[:-1])
        assert d.unpaired is True
        assert d.ships is False and d.p_value is None

    def test_empty_arms_produce_NO_verdict(self):
        mod = _otd()
        d = _decide(mod, [], [])
        assert d.unpaired is True and d.ships is False

    def test_EQUAL_lengths_are_paired(self):
        """The admit side — a guard that refuses everything passes the
        test above."""
        mod = _otd()
        d = _decide(mod, *_arms(40, 0, 6, overlap=10))
        assert d.unpaired is False and d.ships is True


class TestNoPIsFoldedWhenItIsUnknown:
    def test_zero_discordant_replays_give_None_not_one(self):
        """`verdict-without-power`: nothing disagreed is an absence of
        evidence, never evidence of equality."""
        mod = _otd()
        d = _decide(mod, *_arms(40, 0, 0, overlap=40))
        assert d.p_value is None
        assert d.significant is False and d.ships is False


class TestTheOverrideIsDeliberateAndBounded:
    def test_it_promotes_a_borderline_sweep(self):
        mod = _otd()
        inc, cand = _arms(40, 0, 4, overlap=10)
        d = _decide(mod, inc, cand, allow=True)
        assert d.ships is True and d.overridden is True

    def test_it_does_NOT_lift_the_margin(self):
        """It lifts the significance bar only, or it is a blanket
        --ship-anything. These arms are 4-0 — insignificant, so the
        override is live — and carry a 0.10 paired margin against a 0.20
        bar, so ONLY the margin can be what refuses."""
        mod = _otd()
        inc, cand = _arms(40, 0, 4, overlap=10)
        d = _decide(mod, inc, cand, min_delta=0.20, allow=True)
        assert d.significant is False, "the override must be reachable"
        assert d.paired_delta == pytest.approx(0.10)
        assert d.ships is False and d.overridden is False
        assert _decide(mod, inc, cand, min_delta=0.05,
                       allow=True).ships is True

    def test_it_does_NOT_lift_validity_or_the_aggregate_ceiling(self):
        mod = _otd()
        inc, cand = _arms(40, 0, 4, overlap=10)
        assert not _decide(mod, inc, cand, valid=False,
                           allow=True).ships
        assert not _decide(mod, inc, cand, aggregate_ok=False,
                           allow=True).ships

    def test_an_HONEST_promotion_is_not_stamped_as_overridden(self):
        mod = _otd()
        d = _decide(mod, *_arms(40, 0, 6, overlap=10))
        assert d.ships is True and d.overridden is False


class TestTheReplayPassTestMatchesTheAdapter:
    def test_full_fidelity_only(self):
        mod = _otd()
        assert mod._replay_passed({"score": 1.0}) is True
        assert mod._replay_passed({"score": 0.99}) is False
        assert mod._replay_passed({"score": 0.0}) is False

    def test_a_missing_or_junk_score_is_not_a_pass(self):
        """An unreplayable fixture carries `score: 0.0`, but a malformed
        one must not read as a pass by accident."""
        mod = _otd()
        for bad in ({}, {"score": None}, {"score": "x"}, {"score": []}):
            assert mod._replay_passed(bad) is False


# ══════════════════════════════════════════════════════════════════════
# The CALL SITE — where the fix and the bug still agreed
# ══════════════════════════════════════════════════════════════════════
class TestTheDecisionIsActuallyUSED:
    """⚠ THE RULE WAS PINNED; ITS ONLY CALL SITE WAS NOT. Two independent
    reviewers found the same thing: `ships = _dec.ships` reverted to
    `_dec.cleared_margin` — the exact pre-§4DA rule — passed the whole
    16k-test suite. So did `allow_insignificant=True`, `valid=True,
    aggregate_ok=True`, swapped trajectory lists, `if True:` on the live
    artifact write, and every one of the seven fields §4DA added to the
    payload.

    I extracted `_ship_decision` precisely to avoid the harness trap that
    hid four defects across §4CY/§4CZ — and then pinned only the extracted
    part. The region where the fix and the bug agree moved one level out.

    This drives the real `main()`. Only the OPTIMIZER and the REPLAY
    TRANSPORT are stubbed; the decision, the payload, the artifact writer
    and the printed line are all real.
    """

    def _run(self, tmp_path, monkeypatch, *, cand_wins=0, inc_wins=0,
             extra_argv=(), min_delta="0.02", n_fixtures=70,
             transport=0, inflate=0, other_err=0, gap=0,
             transport_arm="incumbent", mutate=True, n_tools=1,
             age_days=None, on_optimize=None, on_module=None,
             seed_wins=0):
        """Drive the real `main()`.

        ⚠ THE SCORES ARE DERIVED FROM THE ACTUAL BATCH, not from fixed
        lists. There is no `--private-pct`: the tier is hashed per request
        inside the script, so a hard-coded score list would silently
        mismatch `len(priv)` — a corpus the run cannot produce, which is
        the shape that hid four defects across §4CY/§4CZ. `n_fixtures` is
        sized so the private tier clears the pre-flight at the 0.02
        default (which needs 50), and small enough that a 4-replay swing
        CLEARS the margin while staying insignificant — the region where
        the override matters. At a 250-fixture tier, 4 wins is delta 0.016
        and the MARGIN blocks it before significance is consulted, so the
        override branch is never reached.
        """
        import json as _json
        import types
        mod = _otd()
        # ⚠ THE MODULE IS RE-IMPORTED PER CALL, so a test cannot patch
        # anything inside it from the outside — it would patch a module
        # object this run never touches. `on_module` receives the one the
        # run will actually execute. §4DA round 16 needed it to freeze
        # the clock: the re-draw guard's `_age >= cap` boundary cannot be
        # reached with a wall clock (the age is always a few microseconds
        # over), so `>=` -> `>` took the same branch and survived.
        if on_module is not None:
            on_module(mod)
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("GHOST_HOME", str(home))

        base = mod._baseline_descriptions()
        # ⚠ `n_tools` EXISTS BECAUSE A ONE-TOOL CORPUS CANNOT SEE A
        # SET-LEVEL DEFECT. With a single component `best` and the
        # CHANGED subset are the same dict, so promoting the whole set
        # instead of the changed part is invisible.
        _tools = sorted(base)[:n_tools]
        tool = _tools[0]
        fx = home / "fx.jsonl"
        # The tier is an explicit field (not a hash) and the tool comes
        # from `chosen_tools`, so the corpus is fully controllable — the
        # private tier is sized to clear the pre-flight, and `pub` is
        # non-empty because the optimizer's call budget is
        # `iterations * len(pub)`.
        # ⚠ AND A REAL RECORDINGS DAY-FILE BEHIND THEM. The pre-flight
        # now PROBES replayability (`_load_recorded_payload`), so a
        # fixture with no `source` pointer is honestly unreplayable and
        # the run refuses before the gate — which is the correct
        # behaviour and made every artifact test here vacuous. The
        # harness carries the recordings the real corpus carries.
        rec = home / "system" / "llm_recordings"
        rec.mkdir(parents=True, exist_ok=True)
        recs, rows = [], []
        for i in range(n_fixtures):
            rows.append({"req_id": f"r{i}", "label": 1.0,
                         "tier": "private" if i < n_fixtures - 10
                                 else "public",
                         "chosen_tools": [{"name": _tools[i % len(_tools)]}],
                         "source": {"file": str(rec
                                                / "2026-08-01.jsonl"),
                                    "ordinal": i,
                                    "session_id": "s1"},
                         "payload": {
                             "messages": [{"role": "user",
                                           "content": "q"}],
                             "tools": [], "max_tokens": 256}})
            recs.append({"ordinal": i, "session_id": "s1",
                         "payload": {
                             "messages": [{"role": "user",
                                           "content": "q"}],
                             "tools": [{"type": "function",
                                        "function": {
                                            "name": _t2,
                                            "description": base[_t2],
                                            "parameters": {}}}
                                       for _t2 in _tools],
                             "max_tokens": 256}})
        (rec / "2026-08-01.jsonl").write_text(
            "\n".join(_json.dumps(r) for r in recs))
        fx.write_text("\n".join(_json.dumps(r) for r in rows))

        calls = {"n": 0}
        seen = {}

        def _fake_eval(self, batch, candidate, capture_traces=False):
            from gepa.core.adapter import EvaluationBatch
            # ⚠ The adapter the RUN built, observed from inside it. The
            # script is re-imported per call (`_otd()`), so a test that
            # patches its own copy of `ToolDescAdapter` patches a class
            # object the run never instantiates — §4DA round 15's first
            # attempt at pinning `--timeout` did exactly that and saw an
            # empty dict.
            LAST_ADAPTER["timeout"] = self.timeout
            calls["n"] += 1
            n = len(batch)
            seen["n"] = n
            # ⚠ THE THIRD PASS IS THE §4CW SEED ARM. It runs only when a
            # LIVE artifact seeded the A/B, i.e. on a RE-promotion — the
            # state in which the hand-written text is in NEITHER main arm.
            # Its `seed_wins` band scores 0.0 in both MAIN arms, so it is
            # concordant there and cannot move the main gate, and 1.0 for
            # the hand-written arm — the only shape that lets a run ship
            # against the incumbent and still lose to the baseline.
            _is_seed = calls["n"] >= 3
            # First `cand_wins` fixtures: incumbent fails, candidate passes.
            # Next `inc_wins`: the mirror. The rest tie as passes.
            scores, errs = [], []
            for i in range(n):
                # `transport` rows fail to REACH the model in the
                # incumbent pass only — the measured one-arm outage. They
                # score 0.0 like a wrong answer, which is the whole
                # problem: indistinguishable in the raw accuracy.
                # ⚠ THE ARM MATTERS. `transport` used to hit the
                # INCUMBENT pass only, so `cand_acc` always equalled the
                # paired candidate rate and five mutants that swap one
                # for the other were indistinguishable. The candidate arm
                # is also the one that runs LAST, hours later, which is
                # the arm an upstream restart is most likely to hit.
                # ⚠ THE SEED ARM IS ITS OWN ARM. It ran as "candidate"
                # here, so `transport_arm="candidate"` starved the MAIN
                # candidate pass too and the run refused before the seed
                # arm was ever consulted — which made
                # `transport_arm="seed"`, the only shape that can show a
                # SUPPRESSED veto, unreachable. §4DA round 16.
                _this_arm = ("incumbent" if calls["n"] == 1
                             else "seed" if calls["n"] >= 3
                             else "candidate")
                if transport and _this_arm == transport_arm \
                        and i >= n - transport:
                    scores.append(0.0); errs.append("transport"); continue
                # A CORPUS GAP: no recorded payload, so it fails in BOTH
                # arms deterministically. Without it `_transport_failed`
                # and `_outage` agree on every row the harness can build.
                if gap and n - transport - gap <= i < n - transport:
                    scores.append(0.0); errs.append("unreplayable")
                    continue
                # A THIRD err state: the adapter marks a candidate the
                # per-tool validator refused. It is neither a corpus gap
                # nor an outage, and counting the two by SUBTRACTION
                # labelled it "unreplayable".
                # ⚠ CANDIDATE ARM ONLY. The incumbent arm evaluates
                # `seed_candidate`, and `_validate_tool_description(n, b,
                # b)` is True for every real tool — so an incumbent-arm
                # cap rejection is a state the pipeline cannot produce,
                # and a pin asserting one asserts nothing. Offset past
                # `cand_wins` so the two cannot silently overlap.
                if (seed_wins
                        and cand_wins + inc_wins <= i
                        < cand_wins + inc_wins + seed_wins):
                    errs.append("")
                    scores.append(1.0 if _is_seed else 0.0)
                    continue
                if (other_err and calls["n"] != 1
                        and cand_wins <= i < cand_wins + other_err):
                    scores.append(0.0)
                    errs.append("candidate over per-tool cap")
                    continue
                errs.append("")
                if i < cand_wins:
                    scores.append(0.0 if calls["n"] == 1 else 1.0)
                elif i < cand_wins + inc_wins:
                    scores.append(1.0 if calls["n"] == 1 else 0.0)
                else:
                    scores.append(1.0)
            trajs = [{"fx": b, "picked": None, "truth": None,
                      "score": s, "err": e}
                     for b, s, e in zip(batch, scores, errs)]
            return EvaluationBatch(
                outputs=[None] * n, scores=scores,
                trajectories=trajs if capture_traces else None)

        monkeypatch.setattr(mod.ToolDescAdapter, "evaluate", _fake_eval)

        import gepa as _gepa

        def _fake_optimize(**kw):
            # ⚠ THE HOOK EXISTS BECAUSE A TEST-LOCAL SPY CANNOT RUN.
            # This harness patches `gepa.optimize` AFTER any patch the
            # test installed, so an abort pin that asserted
            # `calls["n"] == 0` on its own spy was asserting on a
            # function that was never wired in — green whether the abort
            # fired or not. Demonstrated: with the outage abort put back
            # under `--smoke`, the run PAID for the optimizer and
            # `calls["n"] == 0` still passed. Pass `on_optimize=` to be
            # called from the stub that is actually installed.
            if on_optimize is not None:
                on_optimize(**kw)
            # `inflate` lengthens ONE component so the REAL
            # `_aggregate_inflation` computes a real number — the ceiling
            # branch is otherwise unreachable, since an unchanged
            # candidate inflates by 0.
            # ⚠ A REAL OPTIMIZER CHANGES THE TEXT. Returning the seed
            # verbatim made every component "unchanged", and §4DA round
            # 13 stopped promoting those — an unchanged component has
            # nothing to promote, and stamping it with the set's win is a
            # false audit record. A harness that cannot produce a changed
            # candidate cannot exercise the promotion path at all.
            # `mutate=False` returns the seed verbatim — an optimizer
            # that produced no candidate, which §4DA round 13 refuses to
            # promote.
            # `mutate=True` changes every component, `"one"` changes
            # exactly one (what a real proposal does), `False` returns
            # the seed verbatim (no candidate at all).
            _seed = dict(kw["seed_candidate"])
            # ⚠ `"whitespace"` and `"internal_ws"` exist because the
            # `_changed` comparison `.strip()`s BOTH sides, and dropping
            # both strips survived §4DA round 14's battery. The read site
            # (`loader.py`) stores `opt.strip()`, so surrounding
            # whitespace renders byte-identically in production
            # (-> unchanged) and internal whitespace does not
            # (-> a genuinely different served prompt).
            # ⚠ AN INTEGER CHANGES EXACTLY N. `True`/`"one"`/`False` cover
            # all / 1 / 0, which leaves the ONLY interesting boundary for
            # a `len(_changed) == 1` test — two — unreachable, and
            # `== 1 -> <= 2` survived §4DA round 15's battery because of
            # it. `isinstance(mutate, bool)` first: bool IS an int.
            if isinstance(mutate, int) and not isinstance(mutate, bool):
                cand = dict(_seed)
                for _k in sorted(cand)[:mutate]:
                    cand[_k] = cand[_k] + " Prefer it for current events."
            elif mutate == "whitespace":
                cand = {k: "  " + v + "\n" for k, v in _seed.items()}
            elif mutate == "internal_ws":
                _k = sorted(_seed)[0]
                cand = dict(_seed)
                cand[_k] = cand[_k].replace(" ", "  ", 1)
            elif mutate == "one":
                cand = dict(_seed)
                _k = sorted(cand)[0]
                cand[_k] = cand[_k] + " Prefer it for current events."
            elif mutate:
                cand = {k: v + " Prefer it for current events."
                        for k, v in _seed.items()}
            else:
                cand = _seed
            if inflate:
                k = sorted(cand)[0]
                cand[k] = cand[k] + " " + ("x" * inflate)
            return types.SimpleNamespace(best_candidate=cand)
        monkeypatch.setattr(_gepa, "optimize", _fake_optimize)

        # ⚠ `--min-promotion-age-days 0`. §4DA round 13 ported the
        # sibling's re-draw guard, and a harness that promotes twice into
        # one tmp home is refused by it — correctly. Tests that exercise
        # the PROMOTION path disable it; a dedicated pin drives the guard
        # itself.
        argv = ["otd", "--fixtures", str(fx), "--min-delta", min_delta,
                "--upstream-url", "http://127.0.0.1:9",
                *([] if age_days == "DEFAULT" else
                  ["--min-promotion-age-days",
                   ("0" if age_days is None else str(age_days))]),
                "--min-fixtures", "1", "--max-iterations", "1", *extra_argv]
        old = sys.argv
        try:
            sys.argv = argv
            rc = mod.main()
        finally:
            sys.argv = old
        live = sorted((home / "system" / "optim").glob("*.json"))
        rejected = sorted((home / "system" / "optim")
                          .glob("*.candidate.rejected"))
        return rc, live, rejected, seen.get("n", 0)

    def test_a_significant_sweep_REACHES_the_live_artifact(self, tmp_path,
                                                           monkeypatch):
        """The admit side, first: if this cannot ship, the refusals below
        prove nothing."""
        rc, live, rejected, n = self._run(
            tmp_path, monkeypatch, cand_wins=6)
        assert n >= 50, f"the private tier was only {n}"
        assert rc == 0, "a 6-0 sweep did not ship"
        assert live and not rejected

    def test_a_MARGIN_WITHOUT_SIGNIFICANCE_never_reaches_it(self, tmp_path,
                                                            monkeypatch):
        """⚠ THE §4DA DEFECT, DRIVEN THROUGH main(). Two flipped replays
        of the 60-row private tier is a +0.033 margin — clearing the 0.02
        bar — and p=0.25."""
        rc, live, rejected, _n = self._run(
            tmp_path, monkeypatch, cand_wins=2)
        assert rc == 1, "a p=0.25 candidate shipped"
        assert not live, "a rejected candidate reached the LIVE path"
        assert rejected, "the rejected candidate was not recorded"

    def test_the_artifact_records_the_evidence_it_decided_on(self, tmp_path,
                                                             monkeypatch):
        """⚠ DELIBERATELY UNEQUAL WIN COUNTS (2 vs 6), so a crossed pair
        cannot satisfy both assertions — all seven fields §4DA added were
        unpinned, including `p_value` and the two win columns."""
        import json as _json
        rc, live, rejected, _n = self._run(
            tmp_path, monkeypatch, cand_wins=6, inc_wins=2)
        art = _json.loads((live or rejected)[0].read_text())
        # ⚠ NESTED UNDER `gate`, WITH `run_gepa.py`'S KEY NAMES. §4DA
        # first stamped these flat and called the count
        # `discordant_replays`; `recheck_gepa_incumbent.py:107` reads
        # `art["gate"]["discordant_pairs"]`, so the audit trail added
        # specifically so an override could be re-examined was written in
        # a shape its only reader cannot open.
        g = art["gate"]
        assert g["incumbent_wins"] == 2
        assert g["candidate_wins"] == 6
        assert g["discordant_pairs"] == 8
        assert g["ship_alpha"] == ab_eval.SHIP_ALPHA
        assert g["min_delta"] == 0.02
        assert g["p_value"] is not None
        assert g["significance_overridden"] is False
        # ⚠ ONLY ON A PROMOTION. `gate_arm` is the loader's proxy for
        # "this artifact has gate provenance", so stamping it on the
        # `.candidate.rejected` file too meant a rejected candidate
        # renamed into place loaded as a GATED artifact. Rejected files
        # carry `gate_arm_candidate` instead.
        if live:
            assert art["gate_arm"], \
                "the loader warns 'predates the gate schema' without it"
            assert art["gate"]["promoted_utc"]
        else:
            assert "gate_arm" not in art
            assert "promoted_utc" not in art["gate"]
            assert art["gate_arm_candidate"]

    def test_the_OVERRIDE_flag_is_read_from_the_command_line(self, tmp_path,
                                                             monkeypatch):
        """`allow_insignificant=True` hardcoded at the call site survived —
        the flag was never driven through argparse anywhere."""
        import json as _json
        rc_off, live_off, _r, _h = self._run(
            tmp_path / "off", monkeypatch, cand_wins=4)
        assert rc_off == 1 and not live_off, "p=0.0625 shipped without the flag"
        rc_on, live_on, _r2, _h2 = self._run(
            tmp_path / "on", monkeypatch, cand_wins=4,
            extra_argv=("--allow-insignificant-ship",))
        assert rc_on == 0 and live_on, "the flag did not promote"
        art = _json.loads(live_on[0].read_text())
        assert art["gate"]["significance_overridden"] is True
        assert "SIGNIFICANCE OVERRIDDEN" in art["gate_arm"]

    def test_the_MARGIN_is_read_from_the_command_line(self, tmp_path,
                                                      monkeypatch):
        """Swapping `delta` and `min_delta` at the call site survived."""
        rc, live, _r, _h = self._run(
            tmp_path / "hi", monkeypatch, cand_wins=6, min_delta="0.5")
        assert rc == 1 and not live, (
            "a +0.10 delta shipped against a 0.5 margin")

    def test_the_return_code_DISTINGUISHES_promotion_from_rejection(
            self, tmp_path, monkeypatch):
        """`return 0 if ships else 1` -> `return 0` survived; a caller or
        cron job then cannot tell the two apart."""
        good = self._run(tmp_path / "g", monkeypatch, cand_wins=6)[0]
        bad = self._run(tmp_path / "b", monkeypatch, cand_wins=2)[0]
        assert (good, bad) == (0, 1)

    def test_the_printed_line_carries_the_statistic(self, tmp_path,
                                                    monkeypatch, capsys):
        """Removing p and the discordant count from the A/B line survived —
        the operator's only view of why it shipped."""
        self._run(tmp_path, monkeypatch, cand_wins=6)
        out = capsys.readouterr().out
        assert "McNemar p=0.0156" in out
        assert "6 discordant replays (6 candidate / 0 incumbent)" in out

    def test_an_UNKNOWN_p_is_never_printed_as_a_number(self, tmp_path,
                                                       monkeypatch, capsys):
        """`verdict-without-power`, in the operator's face: rendering an
        absent p as `0.0000` reads as overwhelming significance."""
        self._run(tmp_path, monkeypatch, cand_wins=0)
        out = capsys.readouterr().out
        assert "McNemar p=n/a" in out
        assert "p=0.0000" not in out


class TestTheRemainingCallSiteAndGuards:
    """The second batch: every branch the first battery left alive."""

    def _h(self):
        return TestTheDecisionIsActuallyUSED()

    def test_an_INVALID_candidate_is_refused_through_main(self, tmp_path,
                                                          monkeypatch):
        """⚠ `valid=True, aggregate_ok=True` hardcoded at the call site
        survived: nothing drove a candidate the validator rejects, so the
        per-tool and aggregate guards were decorative from main()'s side."""
        mod = _otd()
        monkeypatch.setattr(mod.registry_mod, "_validate_tool_description",
                            lambda *a, **k: False)
        rc, live, rejected, _n = self._h()._run(
            tmp_path, monkeypatch, cand_wins=20)
        assert rc == 1 and not live, "an invalid description shipped"
        assert rejected

    def test_the_artifact_stamps_the_REAL_constant(self, tmp_path,
                                                   monkeypatch):
        """`ship_alpha == 0.05` compares two copies of one value. Move the
        constant and the stamp must follow."""
        import json as _json
        monkeypatch.setattr(ab_eval, "SHIP_ALPHA", 0.2)
        rc, live, rejected, _n = self._h()._run(
            tmp_path, monkeypatch, cand_wins=6)
        art = _json.loads((live or rejected)[0].read_text())
        assert art["gate"]["ship_alpha"] == 0.2

    def test_the_offered_min_delta_ACTUALLY_clears_the_refusal(
            self, tmp_path, monkeypatch, capsys):
        """⚠ AT n=60 THE ROUNDED-DOWN AND ROUNDED-UP OFFERS AGREE, so the
        default tier sits inside the region where the fix and the bug are
        the same. n=45 discriminates: 1/45 = 0.0222 renders DOWN to 0.022,
        and ceil(1/0.022) = 46 > 45 — the fixed point §4CY fixed. This
        parses the offer out of the refusal and re-runs with it."""
        import re as _re
        h = self._h()
        rc, _l, _r, _n = h._run(tmp_path / "a", monkeypatch, cand_wins=6,
                                n_fixtures=55)          # 45 private
        assert rc == 2, "the pre-flight did not refuse"
        err = capsys.readouterr().err
        m = _re.search(r"raise --min-delta to at least ([0-9.]+)", err)
        assert m, f"no remedy offered; stderr: {err}"
        rc2, _l2, _r2, _n2 = h._run(tmp_path / "b", monkeypatch,
                                    cand_wins=6, n_fixtures=55,
                                    min_delta=m.group(1))
        assert rc2 != 2, (
            f"following the offered --min-delta {m.group(1)} refused "
            f"again: {capsys.readouterr().err}")

    def test_the_override_does_not_restamp_an_HONEST_promotion(
            self, tmp_path, monkeypatch):
        """⚠ `not d.significant` is load-bearing: with the flag set AND a
        genuinely significant winner, dropping it writes
        `significance_overridden: true` on an honest promotion — a false
        record in the field that exists to be audited."""
        import json as _json
        rc, live, _r, _n = self._h()._run(
            tmp_path, monkeypatch, cand_wins=6,
            extra_argv=("--allow-insignificant-ship",))
        assert rc == 0 and live
        _art = _json.loads(live[0].read_text())
        assert _art["gate"]["significance_overridden"] is False
        assert "OVERRIDDEN" not in _art["gate_arm"]

    def test_the_override_DEFAULTS_off(self):
        """Every call site passes it explicitly, so the default was
        untested — the next caller that omits it would get override-on."""
        mod = _otd()
        inc = [{"score": 1.0}] * 36 + [{"score": 0.0}] * 4
        cand = [{"score": 1.0}] * 40
        d = mod._ship_decision(inc, cand, min_delta=0.02,
                               valid=True, aggregate_ok=True)
        assert d.p_value == pytest.approx(0.0625)
        assert d.ships is False and d.overridden is False


class TestAnUnusableMarginRefusesInsteadOfCrashing:
    """⚠ Ported from `run_gepa.py:566`, whose comment names each of these
    as already-fixed THERE. Measured here before porting: `0` raised an
    uncaught ZeroDivisionError, `1e-320` an uncaught OverflowError, `1.0`
    paid for the whole optimizer and could never ship, and a NEGATIVE
    margin made `delta > min_delta` trivially true — so
    `--allow-insignificant-ship` shipped a candidate measurably WORSE than
    the incumbent."""

    def test_reject_side(self, tmp_path, monkeypatch, capsys):
        h = TestTheDecisionIsActuallyUSED()
        # ⚠ READ capsys INSIDE THE LOOP. Read once after it, a SINGLE
        # iteration emitting the message satisfies the assertion for all
        # five — the other four could refuse for any reason at all, or the
        # message could name the wrong margin.
        for i, bad in enumerate(("0", "1", "1.5", "-0.5", "1e-320")):
            rc, live, _r, n = h._run(tmp_path / f"b{i}", monkeypatch,
                                     cand_wins=6, min_delta=bad)
            err = capsys.readouterr().err
            assert rc == 2, f"--min-delta {bad} was accepted"
            assert n == 0, f"--min-delta {bad} paid for the optimizer"
            assert not live
            assert "not a usable margin" in err, (
                f"--min-delta {bad} was refused for another reason: {err}")
            assert bad in err, f"the refusal does not name {bad}: {err}"

    def test_admit_side(self, tmp_path, monkeypatch, capsys):
        """A guard's admit side is half its contract.

        ⚠ AND IT MUST ACTUALLY ADMIT. The first version used `1e-06`,
        which is inside the BOUNDS and then refuses at the RESOLUTION
        pre-flight (ceil(1/1e-6) = 1,000,000 against a 60-row tier) — so
        the "admit side" never admitted anything and the only thing it
        asserted was the absence of one string. Each margin here is
        checked to reach the decision."""
        h = TestTheDecisionIsActuallyUSED()
        for i, ok in enumerate(("0.02", "0.05")):
            rc, live, _r, n = h._run(tmp_path / f"o{i}", monkeypatch,
                                     cand_wins=6, min_delta=ok)
            err = capsys.readouterr().err
            assert "not a usable margin" not in err, (
                f"--min-delta {ok} is inside the documented bounds and "
                f"was rejected")
            assert n > 0, f"--min-delta {ok} never reached the evaluation"
            assert rc == 0 and live, (
                f"--min-delta {ok} did not reach a ship decision: {err}")


class TestAnUpstreamOutageCannotManufactureAShip:
    """⚠ `_call` swallowed EVERY exception into a bare `None`, making "the
    upstream was down" and "the model called no tool" the same
    observation — scored 0.0 with `err=""`, invisible in the unreplayable
    count and in the artifact. The two arms run hours apart on the same
    shared slot, so a restart during one of them manufactures discordant
    pairs in ONE direction. Measured before the fix: a 6-replay outage
    confined to the incumbent arm gave p=0.0156 and ships=True on
    descriptions that were effectively identical."""

    def test_a_marked_outage_is_excluded_from_the_comparison(self):
        mod = _otd()
        inc = [{"score": 0.0, "err": "transport"}] * 6 + \
              [{"score": 1.0, "err": ""}] * 54
        cand = [{"score": 1.0, "err": ""}] * 60
        d = mod._ship_decision(inc, cand, min_delta=0.02,
                               valid=True, aggregate_ok=True)
        assert d.candidate_wins == 0, "an outage became candidate wins"
        assert d.transport_excluded == 6
        assert d.ships is False

    def test_a_GENUINE_no_tool_answer_still_counts(self):
        """The admit side — excluding everything would pass the test
        above and destroy the gate."""
        mod = _otd()
        inc = [{"score": 0.0, "err": ""}] * 6 + \
              [{"score": 1.0, "err": ""}] * 54
        cand = [{"score": 1.0, "err": ""}] * 60
        d = mod._ship_decision(inc, cand, min_delta=0.02,
                               valid=True, aggregate_ok=True)
        assert d.candidate_wins == 6 and d.transport_excluded == 0
        assert d.ships is True

    def test_the_adapter_MARKS_a_transport_failure(self, monkeypatch):
        """⚠ The exclusion is inert unless the source marks it — the
        `built-but-unwired` half. `_call` returns a sentinel so the
        one-value contract its existing stubs rely on is unchanged."""
        mod = _otd()
        adapter = mod.ToolDescAdapter.__new__(mod.ToolDescAdapter)
        adapter.url = "http://127.0.0.1:9"
        adapter.timeout = 0.05
        assert adapter._call({"messages": []}) is mod.TRANSPORT_FAILED

    def test_evaluate_TURNS_the_sentinel_into_a_marked_row(self,
                                                           monkeypatch):
        """⚠ THE WIRE BETWEEN THE TWO HALVES. `_call` returning the
        sentinel and `_ship_decision` excluding marked rows are both
        pinned — and the line that converts one into the other was not, so
        writing `err: ""` unconditionally left both halves green and the
        whole guard inert. `built-but-unwired-loops`, between two pinned
        components."""
        from pathlib import Path as _P
        mod = _otd()
        adapter = mod.ToolDescAdapter("http://127.0.0.1:9", _P("/tmp"), {})
        monkeypatch.setattr(
            mod, "_load_recorded_payload",
            lambda fx, d: {"messages": [], "tools": [], "max_tokens": 64})
        calls = {"n": 0}

        def _call(payload):
            calls["n"] += 1
            return (mod.TRANSPORT_FAILED if calls["n"] == 1 else None)
        monkeypatch.setattr(adapter, "_call", _call)
        monkeypatch.setattr(adapter, "_swap_descriptions",
                            lambda tools, cand: (tools, True))

        batch = [{"chosen_tools": [{"name": "execute"}]} for _ in range(2)]
        out = adapter.evaluate(batch, {}, capture_traces=True)
        errs = [t.get("err") for t in out.trajectories]
        assert errs[0] == "transport", (
            "a replay that never reached the model was recorded as a "
            "legitimate no-tool answer")
        assert errs[1] == "", "a genuine no-tool answer was marked"
        assert out.trajectories[0]["picked"] is None, (
            "the sentinel leaked into `picked`")

    def test_the_sentinel_is_not_None(self):
        """`None` already means "the model called no tool"; conflating the
        two is the defect."""
        mod = _otd()
        assert mod.TRANSPORT_FAILED is not None
        assert mod._transport_failed({"err": "transport"}) is True
        assert mod._transport_failed({"err": ""}) is False
