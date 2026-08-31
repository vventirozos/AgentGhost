"""§4DZ — the map decision consulted evidence it computed afterwards.

`map_status` was decided at the top of the fit; `brier_cv` and the
cross-validated delta interval were computed ~45 lines BELOW it. So the
decision, and the base-rate warning under it, could only ever see the
IN-SAMPLE point estimate.

Live state that exposed it (2026-08-30): in-sample 0.046703 against a base
rate of 0.047776 reads as a win by 0.0011, so `map_status` lands on
"applied" and the `brier > brier_base` warning stays silent — while the
cross-validated interval is [-0.00207, +0.00021], which STRADDLES ZERO. The
module's own `_cv_delta_ci` docstring says the sign of the INTERVAL licenses
a "beats the base rate" claim; the licence was never asked for.

Two questions were also conflated. `map_status` answers "was the Platt map
applied?" — and applying it is CORRECT here, because it beats the raw
composite. `beats_base_rate` answers "is the underlying model better than a
constant?" — and it is not. Both can be true at once, which is exactly the
live state, so they are separate fields.
"""
import json

import pytest

from ghost_agent.core.calibration import (BEATS_INDISTINGUISHABLE, BEATS_NO,
                                          BEATS_UNKNOWN, BEATS_YES,
                                          NOT_INFORMATIVE, FittedParams,
                                          _base_rate_verdict)


# ── the verdict is read off the INTERVAL ─────────────────────────────────

@pytest.mark.parametrize("lo,hi,expected", [
    (-0.0100, -0.0020, BEATS_YES),               # entirely better
    (+0.0020, +0.0100, BEATS_NO),                # entirely worse
    (-0.0021, +0.0002, BEATS_INDISTINGUISHABLE),  # the LIVE interval
    (-0.0100, +0.0100, BEATS_INDISTINGUISHABLE),  # wide and useless
    (0.0, 0.0, BEATS_INDISTINGUISHABLE),          # degenerate
    (-0.0100, 0.0, BEATS_INDISTINGUISHABLE),      # touches zero
])
def test_the_interval_decides_not_the_point_estimate(lo, hi, expected):
    """⚠ THE WHOLE POINT. A favourable midpoint with an interval spanning
    zero is not a result — "a margin is not a result" (§4CY). The midpoint
    of the live interval is negative (i.e. "better"), and the honest verdict
    is still INDISTINGUISHABLE."""
    assert _base_rate_verdict((None, lo, hi)) == expected


def test_no_interval_means_no_verdict_at_all():
    """⚠ THE POINT-ESTIMATE ARM IS GONE (2026-08-30), AND WITH IT A WHOLE
    CLASS OF UNKILLABLE MUTANT.

    `_base_rate_verdict` used to take `brier_cv` and `brier_base` and fall
    back to comparing them when no interval existed. From `fit()` that arm
    is unreachable — `_cv_delta_ci` returns None only below `2*k` = 10 rows
    and the default fit floor is 40 — so those two arguments were dead, and
    a mutant SWAPPING them survived every test that could be written. The
    pin guarding "that arm is unreachable" read a DEFAULT-constructed
    tracker, while the floor is a per-tracker constructor argument: a
    reviewer set it to 8, made the arm live, and the swap then inverted a
    real verdict with the suite green.

    The arm also answered BEATS_NO — a measured word — from exactly the
    point comparison this change says licenses nothing. Now: no usable
    interval, no verdict. There is no argument left to get wrong.
    """
    assert _base_rate_verdict(None) == BEATS_UNKNOWN
    assert _base_rate_verdict(None) in NOT_INFORMATIVE

    import inspect

    from ghost_agent.core.calibration import _base_rate_verdict as _f
    # ⚠ COUNT, NOT NAME. Asserting the parameter is *called* `delta_ci`
    # reddens on a pure rename that changes no behaviour. What matters is
    # that there is nothing to pass BESIDES the interval.
    n_params = len(inspect.signature(_f).parameters)
    assert n_params == 1, (
        f"the verdict takes {n_params} arguments — anything beyond the "
        "interval is unobservable from `fit()` and cannot be pinned")


def test_unknown_is_not_a_licence():
    """`unknown` must be in NOT_INFORMATIVE — an absent measurement is not
    permission. If it were treated as "fine", every params file written
    before the CI existed would silently read as informative."""
    assert BEATS_UNKNOWN in NOT_INFORMATIVE
    assert BEATS_NO in NOT_INFORMATIVE
    assert BEATS_INDISTINGUISHABLE in NOT_INFORMATIVE
    assert BEATS_YES not in NOT_INFORMATIVE


@pytest.mark.parametrize("junk", [
    (None, "x", 0.1), (None, None, None), (None, float("nan"), 0.0),
    "not-a-tuple", 42, (None,),
])
def test_the_verdict_never_raises_and_fails_closed(junk):
    """It runs inside the fit; it must not break it. And an error must land
    on `unknown` (not informative), never on `yes`."""
    v = _base_rate_verdict(junk)
    assert v in NOT_INFORMATIVE, v


# ── the evidence exists before the decision that reads it ────────────────

def _fit_on(seed, n, sep, base_p, tmp_path):
    """A real corpus through the real tracker. `random.Random` deliberately:
    `random.seed(3)` at module scope leaked global RNG state into every test
    collected after this file — under `-n 8 --dist loadfile` that silently
    reseeds a whole worker."""
    import random
    from ghost_agent.core.calibration import CalibrationTracker

    rng = random.Random(seed)
    tracker = CalibrationTracker(tmp_path)
    for _ in range(n):
        y = 1.0 if rng.random() < base_p else 0.0
        c = min(1.0, max(0.0, rng.gauss(0.5 + (sep if y else -sep), 0.25)))
        tracker.record(composite=c, outcome=y,
                       entropy_component=0.5, competence_component=c)
    return tracker.fit()


def test_an_applied_map_with_a_straddling_interval_is_indistinguishable(tmp_path):
    """⚠ THE CALL SITE, EXECUTED. This is the state the whole change exists
    for and NOTHING reproduced it: the live store fits `map_status=applied`
    with a CV interval of [-0.00187, +0.000354], which straddles zero.

    Every pin on the verdict was a unit test of the pure function
    `_base_rate_verdict` or an AST/text assertion about `fit()`. The USE was
    unpinned, so all of these survived the full suite: hardcoding
    `beats_base_rate = BEATS_YES`; passing `None` for the interval (the
    original defect, reintroduced at the call site); passing the in-sample
    `brier_cal` instead of `brier_cv`; and collapsing the verdict into
    `BEATS_YES if map_status == "applied" else BEATS_UNKNOWN` — the exact
    conflation of the two questions this change was written to end.
    """
    from ghost_agent.core.calibration import BEATS_INDISTINGUISHABLE

    params = _fit_on(5, 120, 0.020, 0.6, tmp_path)
    assert params is not None, "the fixture produced no fit"
    # Integrity: this pin is only meaningful while the fixture reaches an
    # APPLIED map with an interval that actually straddles zero. Read the
    # params' OWN stored interval -- recomputing it from the recorded
    # composites measures a different quantity, because `fit()` scores
    # `_composite_for(...)` under the FITTED weights.
    lo, hi = params.brier_cv_delta_lo, params.brier_cv_delta_hi
    assert params.map_status == "applied", (
        f"fixture drifted: map is {params.map_status!r}, so this no longer "
        "pins the applied path")
    assert lo < 0.0 < hi, (
        f"fixture drifted: interval {lo:+.5f}..{hi:+.5f} no longer straddles "
        "zero, so 'indistinguishable' is not the answer under test")

    assert params.beats_base_rate == BEATS_INDISTINGUISHABLE, (
        f"the interval {lo:+.5f}..{hi:+.5f} straddles zero, so the model is "
        f"indistinguishable from a constant -- but the fit recorded "
        f"{params.beats_base_rate!r}")


def test_an_applied_map_with_a_negative_interval_beats_the_base_rate(tmp_path):
    """The counterweight. Without it, 'always answer INDISTINGUISHABLE'
    passes the test above, and a verdict that can only say one thing is not
    a measurement."""
    import random
    from ghost_agent.core.calibration import BEATS_YES, CalibrationTracker

    rng = random.Random(3)
    tracker = CalibrationTracker(tmp_path)
    for _ in range(200):
        y = 1.0 if rng.random() < 0.5 else 0.0
        c = rng.uniform(0.75, 0.98) if y else rng.uniform(0.02, 0.25)
        tracker.record(composite=c, outcome=y,
                       entropy_component=0.5, competence_component=c)
    params = tracker.fit()
    assert params is not None
    lo, hi = params.brier_cv_delta_lo, params.brier_cv_delta_hi
    assert params.map_status == "applied", f"fixture drifted: {params.map_status}"
    assert hi < 0.0, (
        f"fixture drifted: interval {lo:+.5f}..{hi:+.5f} does not sit "
        "strictly below zero, so YES is not the answer under test")
    assert params.beats_base_rate == BEATS_YES, (
        f"a strictly negative interval is a real win; the fit recorded "
        f"{params.beats_base_rate!r}")


def test_the_warning_fires_for_the_indistinguishable_case(tmp_path, caplog):
    """⚠ THIS PIN WAS A GREP AND IS NOW EXECUTED.

    It used to assert two strings were present in the module source. A
    mutant that made the branch unreachable (`elif False and ...`) and one
    that deleted the warning while leaving the text in a comment BOTH
    survived it, and the whole 635-test calibration set with it. Its second
    assertion, `"INDISTINGUISHABLE" in src`, could not fail in any world at
    all -- the constant `BEATS_INDISTINGUISHABLE` is defined in that module.
    """
    import logging
    from ghost_agent.core.calibration import BEATS_INDISTINGUISHABLE

    with caplog.at_level(logging.WARNING, logger="ghost_agent.core.calibration"):
        params = _fit_on(5, 120, 0.020, 0.6, tmp_path)

    assert params.beats_base_rate == BEATS_INDISTINGUISHABLE, "fixture drifted"
    warned = [r.getMessage() for r in caplog.records
              if r.levelno >= logging.WARNING
              and "indistinguishable" in r.getMessage().lower()]
    assert warned, (
        "the point comparison `brier > brier_base` is SILENT for a model "
        "whose estimate edges the base rate while its interval straddles "
        "zero -- the live state. Nothing warned:\n  "
        + "\n  ".join(r.getMessage()[:110] for r in caplog.records))


# ── the two questions stay separate ─────────────────────────────────────

def test_map_status_and_beats_base_rate_are_different_questions(tmp_path):
    """The live params are `map_status="applied"` AND
    `beats_base_rate="indistinguishable"` at the same time: the Platt map
    genuinely beats the RAW composite, while the model does not beat a
    constant. Collapsing them would either stop applying a map that helps,
    or claim a licence the data does not support.

    ⚠ ASSERTED ON A REAL FIT. The first version checked only that two
    dataclass FIELDS existed and that one had the right default. A mutant
    collapsing the two questions into
    `beats_base_rate = BEATS_YES if map_status == "applied" else UNKNOWN`
    passed it, and the whole suite.
    """
    from ghost_agent.core.calibration import (BEATS_INDISTINGUISHABLE,
                                              BEATS_UNKNOWN, FittedParams)
    fields = FittedParams.__dataclass_fields__
    assert fields["beats_base_rate"].default == BEATS_UNKNOWN, (
        "the default must be the non-informative one, so a params object "
        "that never ran the check cannot read as licensed")

    params = _fit_on(5, 120, 0.020, 0.6, tmp_path)
    assert (params.map_status, params.beats_base_rate) == (
        "applied", BEATS_INDISTINGUISHABLE), (
        f"the two questions must be able to disagree on ONE fit, and this "
        f"is the live combination: got map_status={params.map_status!r}, "
        f"beats_base_rate={params.beats_base_rate!r}. If they can no longer "
        "differ, one of them is derived from the other.")


# ── the consumer asks once ──────────────────────────────────────────────

def test_learning_health_reports_the_stored_verdict(tmp_path):
    """⚠ The renderer used to recompute the comparison from `lo`/`hi` — two
    implementations of one decision, free to drift. It must now report what
    the producer decided."""
    from ghost_agent.core.learning_health import render_learning_health

    # ⚠ `calib_dir = memory_dir.parent / "calibration"` — a SIBLING of the
    # memory dir, not a child. Getting this wrong makes the fixture invisible
    # and the assertions vacuous.
    mem = tmp_path / "memory"
    mem.mkdir()
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "calibration_params.json").write_text(json.dumps({
        "threshold": 0.84, "brier": 0.0468, "n_samples": 1254,
        "brier_base_rate": 0.0478, "brier_raw": 0.0546, "brier_cv_delta_lo": -0.00207,
        "brier_cv_delta_hi": 0.00021, "beats_base_rate": BEATS_INDISTINGUISHABLE,
        "w_entropy": 0.0, "w_competence": 0.4, "w_effort": 0.6,
        "lambda_uncertainty": 0.0, "map_status": "applied",
        # The renderer reads these; without them the whole CALIBRATION
        # section is skipped and the assertions below are vacuous.
        "brier_cv": 0.046881, "samples_on_disk": 1254,
        "samples_this_epoch": 1254, "epoch": "test.graded",
    }))
    # The base-rate comparison line renders only with samples behind it;
    # without these the CALIBRATION section prints a bare header and every
    # assertion below is vacuous.
    (tmp_path / "calibration" / "calibration.jsonl").write_text(
        "\n".join(json.dumps({"composite": 0.9, "outcome": 1,
                              "epoch": "test.graded"})
                  for _ in range(40)) + "\n")
    out = render_learning_health(mem)
    assert "INDISTINGUISHABLE" in out, out[:600]
    assert "straddles zero" in out


def test_the_renderer_honours_a_stored_verdict_that_contradicts_the_interval(
        tmp_path):
    """The distinguishing test: if the renderer still re-derived from lo/hi
    it would say "beats" here, because the interval alone says so. It must
    report the STORED verdict — one decision, one place."""
    from ghost_agent.core.learning_health import render_learning_health

    # ⚠ `calib_dir = memory_dir.parent / "calibration"` — a SIBLING of the
    # memory dir, not a child. Getting this wrong makes the fixture invisible
    # and the assertions vacuous.
    mem = tmp_path / "memory"
    mem.mkdir()
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "calibration_params.json").write_text(json.dumps({
        "threshold": 0.84, "brier": 0.0468, "n_samples": 1254,
        "brier_base_rate": 0.0478, "brier_raw": 0.0546,
        # An interval that on its own reads as "beats"…
        "brier_cv_delta_lo": -0.0100, "brier_cv_delta_hi": -0.0020,
        # …and a stored verdict that says otherwise.
        "beats_base_rate": BEATS_INDISTINGUISHABLE,
        "w_entropy": 0.0, "w_competence": 0.4, "w_effort": 0.6,
        "lambda_uncertainty": 0.0, "map_status": "applied",
        # The renderer reads these; without them the whole CALIBRATION
        # section is skipped and the assertions below are vacuous.
        "brier_cv": 0.046881, "samples_on_disk": 1254,
        "samples_this_epoch": 1254, "epoch": "test.graded",
    }))
    (tmp_path / "calibration" / "calibration.jsonl").write_text(
        "\n".join(json.dumps({"composite": 0.9, "outcome": 1,
                              "epoch": "test.graded"})
                  for _ in range(40)) + "\n")
    out = render_learning_health(mem)
    assert "INDISTINGUISHABLE" in out, (
        "the renderer re-derived the verdict from lo/hi instead of reading "
        "the stored one")
    assert "beats the base-rate" not in out


# ══ round two: what the first version of this change got wrong ═══════════

def test_a_rejected_map_never_carries_the_licence(tmp_path):
    """⚠ THE VERDICT MUST DESCRIBE THE PREDICTOR WE SHIP.

    `_cv_brier` and `_cv_delta_ci` fit a Platt map PER FOLD, so they measure
    the CALIBRATED model. On the three rejection paths the map is discarded
    and the RAW composite is shipped — so the interval describes a predictor
    that was explicitly refused. Measured on an anti-correlated composite:
    `map_status="rejected_inverted"` AND `beats_base_rate="yes"`, about a
    shipped scorer whose Brier was 0.724 against a base rate of 0.250 —
    three times WORSE than a constant, carrying the one answer that
    licenses trusting it.

    ⚠ THIS PIN IS EXECUTED, NOT READ. The first version asserted the source
    text `if map_status != "applied":` was followed by "BEATS_UNKNOWN"
    within 900 chars. A mutant that deleted the assignment and left the
    comment `# beats_base_rate = BEATS_UNKNOWN` behind SURVIVED it, along
    with all 185 tests in the calibration set: the text was still there and
    nothing ran the branch.

    The fixture below separates the two worlds: a composite drawn as the
    TRUE probability of its own outcome. It is already calibrated, so the
    Platt map has little to add and is usually discarded, while the
    composite itself crushes the base rate — so the interval says YES about
    a map that does not ship. Fixed: `unknown`. Downgrade deleted: `yes`.

    ⚠ "usually", not "cannot": across seeds 0-7 this lands on
    `discarded_worse` six times and on `applied` twice, and the margin at
    seed 3 is 0.17% of the Brier. That is a CHOSEN seed, not a property, so
    the integrity asserts below are load-bearing — they fail loudly the day
    the fit stops reaching this branch, rather than passing vacuously.

    ⚠ `random.Random`, not `random.seed`. The first version reseeded the
    GLOBAL RNG and never restored it, so every test collected after this
    file saw a deterministic stream; under `-n 8 --dist loadfile` that
    contaminates a whole worker.
    """
    import random
    from ghost_agent.core.calibration import (BEATS_UNKNOWN,
                                              CalibrationTracker,
                                              NOT_INFORMATIVE)

    rng = random.Random(3)
    tracker = CalibrationTracker(tmp_path)
    for _ in range(400):
        c = rng.random()
        y = 1.0 if rng.random() < c else 0.0
        tracker.record(composite=c, outcome=y,
                       entropy_component=0.5, competence_component=c)
    params = tracker.fit()

    # The fixture is only meaningful if it actually reaches the branch AND
    # the interval actually says yes -- otherwise the two worlds agree and
    # the pin is decoration again.
    #
    # ⚠ READ THE PARAMS' OWN INTERVAL. Recomputing it here with
    # `_cv_delta_ci(rows)` over the RECORDED composites measured a different
    # quantity than the branch it guards: `fit()` scores
    # `_composite_for(...)` under the FITTED weights. The two coincided only
    # because this fit happens to land on w_competence=1.0 with the
    # composite as the competence component -- change the objective or the
    # weight grid and the guard silently stops guarding.
    assert params.map_status != "applied", (
        f"fixture drifted: map was {params.map_status!r}, so the rejection "
        "branch never ran and this pin proves nothing")
    ci_hi = params.brier_cv_delta_hi
    assert ci_hi < 0.0, (
        f"fixture drifted: the fit's own interval upper bound {ci_hi:+.4f} "
        "does not say yes, so the downgrade has nothing to downgrade")

    # ⚠ EXACT VALUE, NOT MEMBERSHIP. `in NOT_INFORMATIVE` also accepts
    # BEATS_NO and BEATS_INDISTINGUISHABLE, and mutants recording either one
    # here survived. Those are MEASURED words -- "we ran the comparison and
    # the model lost" / "...and it was a tie" -- for a comparison that was
    # never run. Recording one is a fabricated neutral, which is exactly what
    # this project forbids elsewhere.
    assert params.beats_base_rate == BEATS_UNKNOWN, (
        f"the shipped predictor is the RAW composite ({params.map_status}), "
        f"so no comparison was run -- but the fit reports "
        f"beats_base_rate={params.beats_base_rate!r}")
    assert BEATS_UNKNOWN in NOT_INFORMATIVE, (
        "and the value it records must fail closed for every consumer")



@pytest.mark.parametrize("ci", [
    (None, 0.01, -0.01),                       # inverted
    (None, float("nan"), -0.001),              # NaN bound
    (None, float("inf"), float("-inf")),       # infinities
    (None, "-0.01", "-0.002"),                 # strings that float()
    # NOTE: bools are deliberately NOT here. `(None, True, False)` is
    # lo=1.0/hi=0.0 -- an INVERTED interval caught by the ordering guard, so
    # it never exercised the bool clause. The real bool cases have their own
    # test below, where they can actually fail.
    (0.0, -0.01, -0.002, 9),                   # producer shape changed
    (-0.01, -0.002),                           # 2-tuple
    (None, float("nan"), float("nan")),        # broken, not a tested tie
])
def test_yes_is_unreachable_from_a_malformed_interval(ci):
    """⚠ YES IS THE ONLY LICENSING ANSWER. Every one of these returned it
    before the guards: `float()` accepts strings, `isinstance(True, int)` is
    True, and nothing checked finiteness, ordering or arity."""
    assert _base_rate_verdict(ci) != BEATS_YES
    assert _base_rate_verdict(ci) in NOT_INFORMATIVE


@pytest.mark.parametrize("ci,would_be_without_the_bool_guard", [
    ((None, False, False), BEATS_INDISTINGUISHABLE),
    ((None, True, True), BEATS_NO),
])
def test_bool_bounds_are_unknown_not_a_measured_word(ci, would_be_without_the_bool_guard):
    """⚠ THE BOOL CASE IN THE LIST ABOVE WAS VACUOUS.

    `(None, True, False)` is lo=1.0, hi=0.0 — an INVERTED interval, so the
    ordering guard rejects it and the bool clause never decides anything. A
    mutant replacing `isinstance(lo_raw, bool) or isinstance(hi_raw, bool)`
    with `False or False` survived the whole calibration set.

    Bools can never reach YES (no bool is < 0.0), so the honest failure is
    not a false licence — it is laundering a caller's type error into a
    MEASURED word. Without the guard `(False, False)` reports
    "indistinguishable", which reads as "we ran the comparison and it was a
    tie". We did not: someone passed booleans where an interval goes.
    """
    assert _base_rate_verdict(ci) == BEATS_UNKNOWN, (
        "a bool CI bound is a caller type error; reporting "
        f"{would_be_without_the_bool_guard!r} claims a comparison we never ran")
    assert would_be_without_the_bool_guard != BEATS_UNKNOWN, (
        "this pin is only meaningful while the two worlds disagree")



def test_the_verdict_survives_a_real_save_and_load(tmp_path):
    """⚠ THE READER SET, AGAIN. `_save_params` wrote the field and the
    renderer read it, while `load_params` did not reconstruct it — so every
    RELOADED fit reported the default `unknown` forever, through startup
    `apply_fitted` and through `stats()`.

    The previous 'round trip' test built a dataclass by hand and asserted
    `asdict` contained what was just passed in — it could not fail, and a
    mutant deleting the producer assignment survived all 21 tests.
    """
    from ghost_agent.core.calibration import CalibrationTracker

    tracker = CalibrationTracker(tmp_path)
    p = FittedParams(w_entropy=0.0, w_competence=0.4, threshold=0.84,
                     lambda_uncertainty=0.0, brier=0.0468, n_samples=1254,
                     fitted_at="2026-08-30T00:00:00Z",
                     beats_base_rate=BEATS_INDISTINGUISHABLE)
    tracker._save_params(p)
    back = tracker.load_params()
    assert back is not None, "the params did not reload at all"
    assert back.beats_base_rate == BEATS_INDISTINGUISHABLE, (
        f"the verdict did not survive the round trip: {back.beats_base_rate}")


def test_stats_surfaces_the_verdict(tmp_path):
    """`stats()` has NO production caller — it is a debug/test surface, and
    its docstring says so since 2026-08-30. This pin exists only so the key
    does not silently disappear from that surface; it cannot catch a live
    defect, and the fail-closed `getattr` default beneath it is unreachable
    on a dataclass. Kept deliberately small and honestly labelled rather
    than dressed up as a consumer pin."""
    from ghost_agent.core.calibration import CalibrationTracker

    tracker = CalibrationTracker(tmp_path)
    tracker._save_params(FittedParams(
        w_entropy=0.0, w_competence=0.4, threshold=0.84,
        lambda_uncertainty=0.0, brier=0.0468, n_samples=1254,
        fitted_at="2026-08-30T00:00:00Z",
        beats_base_rate=BEATS_INDISTINGUISHABLE))
    st = tracker.stats()
    assert st.get("beats_base_rate") == BEATS_INDISTINGUISHABLE, st


def test_an_unknown_verdict_does_not_print_a_point_estimate_claim(tmp_path):
    """⚠ `"unknown"` is TRUTHY, so it skipped the fallback AND failed the
    word lookup, landing on the point-estimate branch — printing 'beats
    (point estimate only — no CI recorded)' while a CI WAS recorded, which
    is the exact claim this change abolishes. Reachable the moment anything
    routes params through `load_params`."""
    from ghost_agent.core.learning_health import render_learning_health

    mem = tmp_path / "memory"
    mem.mkdir()
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "calibration_params.json").write_text(json.dumps({
        "threshold": 0.84, "brier": 0.0468, "n_samples": 1254,
        "brier_base_rate": 0.0478, "brier_raw": 0.0546,
        "brier_cv": 0.046881, "samples_on_disk": 1254,
        "samples_this_epoch": 1254,
        "brier_cv_delta_lo": -0.00207, "brier_cv_delta_hi": 0.00021,
        "beats_base_rate": BEATS_UNKNOWN,          # the load_params default
        "w_entropy": 0.0, "w_competence": 0.4, "w_effort": 0.6,
        "lambda_uncertainty": 0.0, "map_status": "applied",
    }))
    (tmp_path / "calibration" / "calibration.jsonl").write_text(
        "\n".join(json.dumps({"composite": 0.9, "outcome": 1,
                              "epoch": "test.graded"}) for _ in range(40)) + "\n")
    out = render_learning_health(mem)
    assert "point estimate only" not in out, (
        "a CI is recorded; the point-estimate caveat is a lie here")
    # ⚠ THIS ASSERTION USED TO SAY THE OPPOSITE, AND ENSHRINED THE BUG.
    # It demanded that a stored `unknown` "derive the real verdict" from the
    # interval. But `unknown` is not "no verdict recorded" -- it is a verdict:
    # the producer measured, found the interval described a map it then
    # REFUSED to ship, and withheld the licence. Deriving from that interval
    # is the consumer overturning the producer on the one population the
    # downgrade exists for. Absence (`cal.get` -> None) may derive; a
    # recorded refusal may not.
    assert "NO base-rate comparison" in out, (
        "a stored 'unknown' is a recorded REFUSAL, not a missing value; the "
        "renderer must report that no comparison applies to the shipped "
        f"predictor rather than deriving one:\n{out[:400]}")
    for licensing in ("beats the base-rate predictor",
                      "is INDISTINGUISHABLE from the base-rate predictor"):
        assert licensing not in out, (
            f"the renderer printed {licensing!r} for a verdict the producer "
            "withheld")
    # ⚠ AND THE STATED CAUSE MUST MATCH THE FIXTURE. This params file says
    # `map_status="applied"`, so the refusal here is NOT "the map was
    # rejected" -- it is "no usable interval was recorded". The renderer's
    # first version hardcoded the rejection wording, which it had no field to
    # check: `map_status` was not even in the collector. This very assertion
    # block used to lock that in, asserting the false sentence was printed.
    assert "the Platt map was" not in out, (
        "the renderer claimed a map rejection on a fit whose map was "
        f"APPLIED:\n{out[:400]}")
    # ⚠ THE CAUSE MUST MATCH WHAT IS ON FILE. This fixture HAS an interval
    # (lo/hi are set), so "no usable interval was recorded" is false here —
    # and the same line then prints the interval, contradicting itself. The
    # honest cause is narrower: the fit declined to draw a verdict from it.
    assert "recorded no verdict from its interval" in out, (
        f"the refusal must state the cause it can observe:\n{out[:400]}")
    assert "no usable cross-validated interval" not in out, (
        f"the sentence claims there is no interval and then prints one:\n"
        f"{out[:400]}")


def test_the_point_estimate_caveat_survives_when_there_is_no_ci(tmp_path):
    """The counterweight: a stored verdict with NO interval behind it is
    still a point comparison, and the caveat must stay."""
    from ghost_agent.core.learning_health import render_learning_health

    mem = tmp_path / "memory"
    mem.mkdir()
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "calibration_params.json").write_text(json.dumps({
        "threshold": 0.84, "brier": 0.06, "n_samples": 1254,
        "brier_base_rate": 0.05, "brier_raw": 0.07, "brier_cv": 0.06,
        "samples_on_disk": 1254, "samples_this_epoch": 1254,
        "beats_base_rate": BEATS_NO,               # stored, but no interval
        "w_entropy": 0.0, "w_competence": 0.4, "w_effort": 0.6,
        "lambda_uncertainty": 0.0, "map_status": "applied",
    }))
    (tmp_path / "calibration" / "calibration.jsonl").write_text(
        "\n".join(json.dumps({"composite": 0.9, "outcome": 1,
                              "epoch": "test.graded"}) for _ in range(40)) + "\n")
    out = render_learning_health(mem)
    assert "point estimate only" in out, out[:600]


def test_the_warnings_use_the_constants_not_string_literals(tmp_path, caplog):
    """The constants exist so the values are not duplicated; a literal
    comparison silently disarms the warning if a constant changes.

    ⚠ THE GREP VERSION WAS SATISFIED BY A COMMENT and could not see
    reachability -- a mutant that made both warning branches dead while
    leaving the text intact survived it. It also contradicted the other
    grep pin, which ACCEPTED the literal form it forbids.

    The property that actually matters is behavioural: the two verdicts
    that are not a licence must each produce their own warning, and the
    licensing one must produce neither. A literal-vs-constant comparison is
    only a defect insofar as it silently stops the branch firing, so pin the
    firing.
    """
    import logging

    from ghost_agent.core.calibration import (BEATS_INDISTINGUISHABLE,
                                              BEATS_YES)

    def _warnings_for(tag, fit_args):
        # ⚠ A FRESH DIR PER FIT. Sharing one made the second fit load the
        # first fit's 120 rows and land on a different verdict -- the
        # fixture, not the code, decided the outcome.
        d = tmp_path / tag
        d.mkdir()
        with caplog.at_level(logging.WARNING,
                             logger="ghost_agent.core.calibration"):
            caplog.clear()
            params = _fit_on(*fit_args, d)
        return params, [r.getMessage().lower() for r in caplog.records]

    p_ind, w_ind = _warnings_for("ind", (5, 120, 0.020, 0.6))
    assert p_ind.beats_base_rate == BEATS_INDISTINGUISHABLE, "fixture drifted"
    assert any("indistinguishable" in m for m in w_ind), (
        "the indistinguishable verdict must warn; nothing did")

    p_no, w_no = _warnings_for("no", (5, 120, 0.010, 0.6))
    assert p_no.beats_base_rate == BEATS_NO, "fixture drifted"
    assert any("base rate" in m or "base-rate" in m or "loses" in m
               for m in w_no), (
        "a model measurably WORSE than a constant must warn; nothing did")


# ── the fail-closed defaults, and the consumers that now read the verdict ──

def test_an_absent_field_loads_as_unknown_not_as_a_licence(tmp_path):
    """⚠ THE PATH THE LIVE FILE TAKES TODAY. `calibration_params.json` on
    this machine predates the field entirely, so every read of it goes
    through the default — and mutants flipping BOTH defaults (`load_params`
    and `stats`) to `BEATS_YES` survived the whole calibration suite,
    because every pin exercised a value that had been STORED.

    A default is the branch nobody tests and the one production is actually
    on."""
    import json

    from ghost_agent.core.calibration import (BEATS_UNKNOWN, SCHEMA_VERSION,
                                              CalibrationTracker,
                                              NOT_INFORMATIVE)

    blob = {"schema": SCHEMA_VERSION,
            "w_entropy": 0.0, "w_competence": 0.4, "threshold": 0.84,
            "lambda_uncertainty": 0.0, "brier": 0.0468, "n_samples": 1254,
            "fitted_at": "2026-08-30T06:58:59Z", "map_status": "applied"}
    assert "beats_base_rate" not in blob, "the fixture must be a legacy file"
    (tmp_path / "calibration_params.json").write_text(json.dumps(blob))

    loaded = CalibrationTracker(tmp_path).load_params()
    assert loaded is not None, "the legacy params did not load at all"
    assert loaded.beats_base_rate == BEATS_UNKNOWN, (
        f"a params file with no recorded verdict loaded as "
        f"{loaded.beats_base_rate!r} -- a licence nothing ever measured")
    assert loaded.beats_base_rate in NOT_INFORMATIVE


@pytest.mark.asyncio
async def test_the_idle_refit_emits_the_licence_in_the_calib_line(tmp_path,
                                                                 monkeypatch):
    """⚠ EXECUTED. The first version of this pin grepped `agent.py` for the
    string "beats_base_rate" — and the string appears in the comment above
    the emit and in the `getattr` that reads it, so deleting the emit kwarg
    outright left the pin green. That is the same source-text failure this
    very round was convened to fix, committed again inside the fix.

    This drives the real idle calibration phase with a real
    `CalibrationTracker` over a corpus whose CV interval straddles zero (the
    live state) and asserts on the payload the operator's 📐 CALIB line
    actually carries.
    """
    import datetime
    import random
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from ghost_agent.core import metacog_log
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.calibration import (BEATS_INDISTINGUISHABLE,
                                              CalibrationTracker)

    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    calib_dir = tmp_path / "system" / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    tracker = CalibrationTracker(calib_dir)
    rng = random.Random(5)
    for _ in range(120):
        y = 1.0 if rng.random() < 0.6 else 0.0
        c = min(1.0, max(0.0, rng.gauss(0.5 + (0.020 if y else -0.020), 0.25)))
        tracker.record(composite=c, outcome=y,
                       entropy_component=0.5, competence_component=c)

    emitted = []
    monkeypatch.setattr(metacog_log, "emit",
                        lambda sub, **kw: emitted.append((sub, kw)))

    ctx = MagicMock()
    ctx.calibration_tracker = tracker
    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    ctx.journal = None
    ctx.frontier_tracker = None
    ctx.reflector = None
    ctx.prm_scorer = None
    ctx.postmortem_engine = None
    ctx.trajectory_collector = None
    ctx.complexity_dispatcher = None
    ctx.last_activity_time = (datetime.datetime.now()
                              - datetime.timedelta(seconds=1200))
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    for k in ("prm_train_cooldown", "router_train_cooldown",
              "self_narrative_cooldown", "calib_refit_cooldown"):
        setattr(ctx.args, k, None)

    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    # The calibration phase is COOLDOWN-gated, not idle-banded; this puts it
    # well past the cooldown so the phase actually runs.
    agent._last_calib_refit_at = (datetime.datetime.now()
                                  - datetime.timedelta(days=10))
    try:
        await agent._biological_tick()
    except Exception:
        pass

    calib = [kw for sub, kw in emitted
             if getattr(sub, "name", str(sub)).upper().endswith("CALIB")]
    assert calib, (
        f"the calibration phase emitted no CALIB line at all; "
        f"emitted={[getattr(s, 'name', s) for s, _ in emitted]}")
    payload = calib[-1]
    assert payload.get("beats_base_rate") == BEATS_INDISTINGUISHABLE, (
        "the line the operator watches must carry the base-rate verdict, "
        f"not only the map. got {payload!r}")
    _refit = str(payload.get("refit", ""))
    # Property, not token: the qualifier's PREFIX changed 2026-08-31
    # (`no_signal:` → `no_prob:` + `no_rank:`, because one Brier comparison
    # cannot license the words "no signal"). What must hold either way is
    # that the line is not a bare `ok` and names the verdict itself.
    assert _refit != "ok" and BEATS_INDISTINGUISHABLE in _refit, (
        "a refit whose score is indistinguishable from a constant still "
        f"reads as healthy: refit={_refit!r}")
    # §4EB: the SECOND verdict rides the same line. Without it the operator
    # reads one statistic and takes it for "is there signal" — the two have
    # disagreed on identical rows (probability indistinguishable, ordering
    # AUC 0.652). Fails in the world where only the probability verdict is
    # surfaced.
    assert "ranks_outcomes" in payload, (
        f"the CALIB line carries no rank verdict: {payload!r}")
    _ranks = payload.get("ranks_outcomes")
    if _ranks != BEATS_YES:
        assert f"no_rank:{_ranks}" in _refit, (
            "a score that does not rank still reads as ranking: "
            f"refit={_refit!r}")


def test_the_startup_calib_payload_carries_the_verdict():
    """⚠ TWICE A GREP, THEN A HARNESS GRADING ITS OWN COPY.

    v1 asserted `"beats_base_rate" in getsource(main)` — commenting the
    kwarg out left the literal in place, green. v2 rebuilt the `_mc_emit`
    call inside the test and asserted on that, which tests the test.
    `calib_startup_fields` was extracted from `lifespan` (untestable: a
    decorated async context manager) precisely so the payload the process
    actually emits can be executed here.
    """
    from ghost_agent.core.calibration import (BEATS_INDISTINGUISHABLE,
                                              BEATS_UNKNOWN, FittedParams)
    from ghost_agent.main import calib_startup_fields

    params = FittedParams(
        w_entropy=0.0, w_competence=0.4, threshold=0.84,
        lambda_uncertainty=0.0, brier=0.0468, n_samples=1254,
        fitted_at="2026-08-30T00:00:00Z", map_status="applied",
        beats_base_rate=BEATS_INDISTINGUISHABLE)

    payload = calib_startup_fields(params)
    assert payload["beats_base_rate"] == BEATS_INDISTINGUISHABLE, (
        f"the startup line reports {payload.get('beats_base_rate')!r} for "
        f"params storing {BEATS_INDISTINGUISHABLE!r}")
    assert payload["map"] == "applied", (
        "and it must still carry the map — these are different questions")
    assert payload["loaded"] == "startup"

    # And it must READ the params, not hardcode a licence.
    other = calib_startup_fields(FittedParams(
        w_entropy=0.0, w_competence=0.4, threshold=0.84,
        lambda_uncertainty=0.0, brier=0.0468, n_samples=1254,
        fitted_at="2026-08-30T00:00:00Z", beats_base_rate="yes"))
    assert other["beats_base_rate"] == "yes", (
        "the payload does not vary with the stored verdict — it is a "
        "constant dressed as a measurement")

    # ⚠ AND THE MAP MUST VARY TOO. Both fixtures above carry
    # `map_status="applied"`, so `"map": "applied"` hardcoded survived — the
    # startup line could report an APPLIED map for a rejected one, which is
    # the 2026-07-29 defect, on the very surface this helper was extracted to
    # make pinnable. A fixture where the fixed and broken worlds agree is not
    # a pin.
    rejected = calib_startup_fields(FittedParams(
        w_entropy=0.0, w_competence=0.4, threshold=0.61,
        lambda_uncertainty=0.0, brier=0.0468, n_samples=1254,
        fitted_at="2026-08-30T00:00:00Z", map_status="rejected_inverted",
        beats_base_rate=BEATS_UNKNOWN))
    assert rejected["map"] == "rejected_inverted", (
        f"the startup line reports map={rejected['map']!r} for a REJECTED "
        "map — a rejected calibration reading as healthy")
    assert rejected["threshold"] == 0.61, (
        "the payload does not carry the fit's own threshold")
    assert rejected["beats_base_rate"] == BEATS_UNKNOWN


def test_the_loader_itself_rejects_an_unrecognised_verdict(tmp_path):
    """⚠ `_known_verdict` WAS PINNED; ITS USE WAS NOT. A mutant restoring
    the bare `str(d.get(...))` in `load_params` survived, because every test
    called the coercion function directly. Load a real file instead."""
    import json

    from ghost_agent.core.calibration import (BEATS_UNKNOWN, SCHEMA_VERSION,
                                              CalibrationTracker)

    (tmp_path / "calibration_params.json").write_text(json.dumps({
        "schema": SCHEMA_VERSION, "w_entropy": 0.0, "w_competence": 0.4,
        "threshold": 0.84, "lambda_uncertainty": 0.0, "brier": 0.0468,
        "n_samples": 1254, "fitted_at": "2026-08-30T00:00:00Z",
        "map_status": "applied", "beats_base_rate": "probably"}))

    loaded = CalibrationTracker(tmp_path).load_params()
    assert loaded is not None
    assert loaded.beats_base_rate == BEATS_UNKNOWN, (
        f"a params file carrying {'probably'!r} loaded as "
        f"{loaded.beats_base_rate!r}; consumers testing `in NOT_INFORMATIVE` "
        "then read it as LICENSED, because an unknown word is not in that set")


@pytest.mark.asyncio
async def test_an_unrecognised_verdict_does_not_silence_the_refit_warning(
        tmp_path, monkeypatch):
    """⚠ THE CONSUMER SIDE OF THE SAME FAIL-OPEN. The CALIB line tested
    `if _beats in NOT_INFORMATIVE`, so anything outside the vocabulary
    skipped the `no_signal` suffix and the operator saw a clean `refit=ok`.
    It now tests `!= BEATS_YES`, which fails closed."""
    import datetime
    import json
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from ghost_agent.core import metacog_log
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.calibration import CalibrationTracker

    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    calib_dir = tmp_path / "system" / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    tracker = CalibrationTracker(calib_dir)

    class _Junk:
        """A fit that returns a verdict outside the vocabulary."""

        def __getattr__(self, name):
            raise AttributeError(name)

    real_fit = tracker.fit

    def _fit_with_junk(*a, **kw):
        params = real_fit(*a, **kw)
        if params is not None:
            object.__setattr__(params, "beats_base_rate", "probably")
        return params

    import random
    rng = random.Random(5)
    for _ in range(120):
        y = 1.0 if rng.random() < 0.6 else 0.0
        c = min(1.0, max(0.0, rng.gauss(0.5 + (0.020 if y else -0.020), 0.25)))
        tracker.record(composite=c, outcome=y,
                       entropy_component=0.5, competence_component=c)
    monkeypatch.setattr(tracker, "fit", _fit_with_junk)

    emitted = []
    monkeypatch.setattr(metacog_log, "emit",
                        lambda sub, **kw: emitted.append((sub, kw)))

    ctx = MagicMock()
    ctx.calibration_tracker = tracker
    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    for attr in ("journal", "frontier_tracker", "reflector", "prm_scorer",
                 "postmortem_engine", "trajectory_collector",
                 "complexity_dispatcher"):
        setattr(ctx, attr, None)
    ctx.last_activity_time = (datetime.datetime.now()
                              - datetime.timedelta(seconds=1200))
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    for k in ("prm_train_cooldown", "router_train_cooldown",
              "self_narrative_cooldown", "calib_refit_cooldown"):
        setattr(ctx.args, k, None)

    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    agent._last_calib_refit_at = (datetime.datetime.now()
                                  - datetime.timedelta(days=10))
    try:
        await agent._biological_tick()
    except Exception:
        pass

    calib = [kw for sub, kw in emitted
             if getattr(sub, "name", str(sub)).upper().endswith("CALIB")]
    assert calib, "no CALIB line was emitted"
    refit = str(calib[-1].get("refit", ""))
    # Property, not token — see the note on the sibling test above. The
    # fixture sets the junk verdict on the returned object directly, so it
    # reaches the line uncoerced BY DESIGN; what must hold is that the line
    # is not a bare `ok` and shows the operator the word it could not
    # recognise, rather than swallowing it.
    assert refit != "ok" and "probably" in refit, (
        f"an unrecognised verdict read as licensed: refit={refit!r}")


@pytest.mark.asyncio
async def test_the_activity_ledger_records_the_missing_licence(tmp_path,
                                                               monkeypatch):
    """⚠ THE OTHER HALF OF THE OPERATOR'S VIEW, AND IT WAS UNPINNED.

    `introspect activity` is the second place a refit is read. Deleting the
    whole `_map_note` licence block survived 667 tests, silently reverting
    the ledger to an unqualified "confidence recalibrated (τ=…, Brier=…,
    n=…)" — the exact unqualified summary this change exists to end.
    """
    import datetime
    import random
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.calibration import CalibrationTracker

    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    calib_dir = tmp_path / "system" / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    tracker = CalibrationTracker(calib_dir)
    rng = random.Random(5)
    for _ in range(120):
        y = 1.0 if rng.random() < 0.6 else 0.0
        c = min(1.0, max(0.0, rng.gauss(0.5 + (0.020 if y else -0.020), 0.25)))
        tracker.record(composite=c, outcome=y,
                       entropy_component=0.5, competence_component=c)

    recorded = []
    monkeypatch.setattr(GhostAgent, "_record_autonomous_activity",
                        lambda self, phase, summary, **kw:
                        recorded.append((phase, summary)))

    ctx = MagicMock()
    ctx.calibration_tracker = tracker
    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    for attr in ("journal", "frontier_tracker", "reflector", "prm_scorer",
                 "postmortem_engine", "trajectory_collector",
                 "complexity_dispatcher"):
        setattr(ctx, attr, None)
    ctx.last_activity_time = (datetime.datetime.now()
                              - datetime.timedelta(seconds=1200))
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    for k in ("prm_train_cooldown", "router_train_cooldown",
              "self_narrative_cooldown", "calib_refit_cooldown"):
        setattr(ctx.args, k, None)

    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    agent._last_calib_refit_at = (datetime.datetime.now()
                                  - datetime.timedelta(days=10))
    try:
        await agent._biological_tick()
    except Exception:
        pass

    calib = [sm for ph, sm in recorded if ph == "calibration"]
    assert calib, f"no calibration activity was recorded; got {recorded}"
    summary = calib[-1]
    assert "NO base-rate licence" in summary, (
        "a refit whose score is indistinguishable from a constant was "
        f"recorded as an unqualified success: {summary!r}")


def test_a_stored_verdict_reaches_the_startup_log_through_load_params(tmp_path):
    """The startup line reads a `FittedParams` from `load_params`, so the
    loader reconstructing the field is what makes that line true. Pinned by
    executing the same hop, because the justification comment on that loader
    was itself wrong once — it named `apply_fitted` and `stats()` as the
    consumers, and neither reads this field."""
    from ghost_agent.core.calibration import (BEATS_INDISTINGUISHABLE,
                                              CalibrationTracker, FittedParams)

    tracker = CalibrationTracker(tmp_path)
    tracker._save_params(FittedParams(
        w_entropy=0.0, w_competence=0.4, threshold=0.84,
        lambda_uncertainty=0.0, brier=0.0468, n_samples=1254,
        fitted_at="2026-08-30T00:00:00Z",
        beats_base_rate=BEATS_INDISTINGUISHABLE))

    reloaded = CalibrationTracker(tmp_path).load_params()
    assert getattr(reloaded, "beats_base_rate", None) == BEATS_INDISTINGUISHABLE, (
        "the value the startup CALIB line emits is whatever this attribute "
        "holds; if the loader drops it the line reports a default forever")


def test_a_bool_interval_bound_is_not_printed_as_a_measured_interval(tmp_path):
    """⚠ THE VERDICT WAS SAFE; THE NUMBER BESIDE IT WAS NOT.

    `_have_ci` tested `isinstance(_dlo, (int, float))`, and `True` is an
    `int`. A params file carrying `"brier_cv_delta_lo": true` therefore
    rendered `[95% CI of the delta +1.00000..+1.00000]` — a fabricated
    interval printed with five decimal places of false precision, beside a
    verdict that had (correctly) refused those same bounds.
    """
    import json

    from ghost_agent.core.learning_health import render_learning_health

    mem = tmp_path / "memory"
    mem.mkdir()
    calib = tmp_path / "calibration"
    calib.mkdir()
    (calib / "calibration.jsonl").write_text("")
    (calib / "calibration_params.json").write_text(json.dumps({
        "w_entropy": 0.0, "w_competence": 0.4, "threshold": 0.84,
        "lambda_uncertainty": 0.0, "brier": 0.0468, "brier_raw": 0.0468,
        "brier_base_rate": 0.0478, "n_samples": 1254, "samples": 1254,
        "n_negative": 18, "brier_cv": 0.046881,
        "beats_base_rate": "yes", "map_status": "applied",
        "brier_cv_delta_lo": True, "brier_cv_delta_hi": True,
    }))

    out = render_learning_health(mem)
    assert "95% CI of the delta +1.00000" not in out, (
        f"a boolean was rendered as a measured interval:\n{out[:400]}")
    assert "point estimate only" in out, (
        "with no usable interval the honesty caveat must be present")


# ── the causes are distinguished, and every consumer is executed ─────────

def _render_with(tmp_path, **params):
    import json

    from ghost_agent.core.learning_health import render_learning_health

    mem = tmp_path / "memory"
    mem.mkdir(parents=True, exist_ok=True)
    calib = tmp_path / "calibration"
    calib.mkdir(parents=True, exist_ok=True)
    (calib / "calibration.jsonl").write_text("")
    # ⚠ SCHEMA BY DEFAULT. Without it every test through this helper
    # rendered a file `load_params` REFUSES — so assertions like "beats the
    # base-rate predictor" were pinned on params the agent is not running,
    # and the `NOT IN FORCE` notice was silently present in all of them.
    # A fixture that does not match the producer tests the fixture.
    from ghost_agent.core.calibration import SCHEMA_VERSION as _SV

    blob = {"schema": _SV,
            "w_entropy": 0.0, "w_competence": 0.4, "threshold": 0.84,
            "lambda_uncertainty": 0.0, "brier": 0.0468, "brier_raw": 0.0468,
            "brier_base_rate": 0.0478, "n_samples": 1254, "samples": 1254,
            "n_negative": 18, "brier_cv": 0.046881,
            "fitted_at": "2026-08-30T00:00:00Z"}
    blob.update(params)
    (calib / "calibration_params.json").write_text(json.dumps(blob))
    return render_learning_health(mem)


def test_the_refusal_states_the_cause_it_can_observe(tmp_path):
    """⚠ THE RENDERER ASSERTED A CAUSE IT HAD NO FIELD TO CHECK.

    `unknown` has TWO sources — a rejected Platt map, and a fit with no
    usable CV interval (`min_samples_for_fit` is a per-tracker argument, so
    a 9-row fit reaches it with `map_status="applied"`). The refusal branch
    hardcoded the first, and `map_status` was not in the collector's dict at
    all. Measured on a real 9-row fit: applied map, `unknown` verdict, no
    interval — and all three clauses of "the Platt map was rejected, so the
    shipped predictor is the raw composite and the interval below describes
    the map we discarded" false at once.
    """
    rejected = _render_with(tmp_path / "a", beats_base_rate="unknown",
                            map_status="rejected_inverted",
                            brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002)
    assert "the Platt map was rejected_inverted" in rejected, rejected[:400]
    assert "does NOT describe what ships" in rejected, rejected[:400]

    no_ci = _render_with(tmp_path / "b", beats_base_rate="unknown",
                         map_status="applied")
    assert "no usable cross-validated interval" in no_ci, no_ci[:400]
    assert "Platt map was" not in no_ci, (
        f"a map that was APPLIED was reported as rejected:\n{no_ci[:400]}")


def test_a_derived_verdict_is_visibly_derived(tmp_path):
    """⚠ THE COMMENT AND THE DOCS BOTH CALLED THIS A "LABELLED" FALLBACK.

    It was not labelled at all: rendering a legacy params file produced
    output BYTE-IDENTICAL to the same file with a stored verdict added, so
    an operator could not tell a measurement the fit recorded from one the
    renderer invented at display time. That is the live store's path today —
    its params file predates the field entirely."""
    common = dict(map_status="applied", brier_cv_delta_lo=-0.01,
                  brier_cv_delta_hi=-0.002)
    derived = _render_with(tmp_path / "a", **common)
    stored = _render_with(tmp_path / "b", beats_base_rate="yes", **common)

    assert "DERIVED at render time" in derived, (
        f"a verdict computed by the renderer is presented as the fit's "
        f"own:\n{derived[:400]}")
    assert "DERIVED at render time" not in stored, (
        "a stored verdict must not be labelled as derived")
    d_line = [l for l in derived.splitlines() if "base-rate predictor" in l][0]
    s_line = [l for l in stored.splitlines() if "base-rate predictor" in l][0]
    assert d_line != s_line, (
        "derived and stored verdicts render identically, so the label is "
        "invisible where it matters")


def test_an_unrecognised_verdict_fails_closed_everywhere(tmp_path):
    """⚠ `in NOT_INFORMATIVE` FAILS OPEN. A value outside the vocabulary is
    not in that set, so every consumer testing membership read it as
    LICENSED — and `load_params` did a bare `str()` with no validation, so a
    typo or a hand-edited params file was enough."""
    from ghost_agent.core.calibration import BEATS_UNKNOWN, _known_verdict

    for junk in ("probably", "", "YES", 7, None, True, "yes "):
        assert _known_verdict(junk) == BEATS_UNKNOWN, (
            f"{junk!r} loaded as a verdict; only the four words are verdicts")
    assert _known_verdict("yes") == "yes", "the real vocabulary still loads"

    out = _render_with(tmp_path, beats_base_rate="probably",
                       map_status="applied",
                       brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002)
    assert "NO usable base-rate verdict" in out, out[:400]
    assert "beats the base-rate predictor" not in out, (
        f"an unrecognised verdict printed a licence:\n{out[:400]}")
    assert "point estimate only" not in out, (
        "a CI is recorded; the point-estimate caveat is a lie here too")


def test_a_stored_unknown_with_no_interval_still_refuses(tmp_path):
    """⚠ THE REFUSAL BRANCH MUST NOT DEPEND ON HAVING AN INTERVAL.

    A mutant narrowing it to `if _stored == BEATS_UNKNOWN and _have_ci:`
    survived, because no test covered a stored `unknown` with no interval —
    and that is exactly the shape a small fit produces (`_cv_delta_ci`
    returns None below 10 rows, so the params carry the refusal and no
    lo/hi). Without the interval the refusal fell through to the
    point-estimate branch and printed a licence."""
    out = _render_with(tmp_path, beats_base_rate="unknown",
                       map_status="applied")
    assert "NO base-rate comparison" in out, out[:400]
    assert "beats the base-rate predictor" not in out, (
        f"a refusal with no interval printed a licence:\n{out[:400]}")
    assert "point estimate only" not in out, (
        f"and it must not be dressed as a point comparison:\n{out[:400]}")


def test_the_rejection_path_says_the_two_briers_are_one_number(tmp_path):
    """⚠ THE FIX QUALIFIED THE INTERVAL AND LEFT THE POINT ESTIMATES ALONE.

    On all three rejection paths `fit()` sets `brier = brier_raw` and
    `calibrated = composites`, so the follow-on line printed
    `in-sample X · raw composite X` — one value twice, reading as two
    independent measurements agreeing — directly beneath a CV figure that
    describes the DISCARDED map. The shipped predictor had no out-of-sample
    number anywhere on the line that was warning about it."""
    rejected = _render_with(tmp_path / "a", beats_base_rate="unknown",
                            map_status="discarded_worse", brier=0.166,
                            brier_raw=0.166, brier_cv=0.1693,
                            brier_base_rate=0.2497,
                            brier_cv_delta_lo=-0.101, brier_cv_delta_hi=-0.059)
    # ⚠ `"in-sample"` ALSO MATCHES THE SECTION HEADER (`Brier X
    # (in-sample)`). This worked only because the fixture sets `brier_cv`,
    # so the header says "(CV)" — one fixture change from silently testing
    # the wrong line, which is a defect this file has already had twice.
    line = [l for l in rejected.splitlines() if "· raw composite" in l][0]
    assert "discarded_worse" in line, (
        f"the in-sample line does not say the map was rejected:\n{line}")
    assert "SAME number" in line, (
        f"it prints one value twice without saying so:\n{line}")
    assert "DISCARDED map, not what ships" in line, line

    applied = _render_with(tmp_path / "b", beats_base_rate="yes",
                           map_status="applied", brier=0.166,
                           brier_raw=0.180, brier_cv=0.1693,
                           brier_base_rate=0.2497,
                           brier_cv_delta_lo=-0.101, brier_cv_delta_hi=-0.059)
    a_line = [l for l in applied.splitlines() if "· raw composite" in l][0]
    assert "DISCARDED" not in a_line, (
        f"an APPLIED map was described as discarded:\n{a_line}")
    assert "in-sample is NOT performance" in a_line, a_line


@pytest.mark.asyncio
async def test_the_startup_path_actually_calls_the_helper(tmp_path, monkeypatch):
    """⚠ THE EXTRACTION MOVED THE PIN AWAY FROM THE SITE IT COVERS.

    `calib_startup_fields` was pulled out of `lifespan` so a test could
    execute the payload. But nothing pinned the CALL: replacing
    `_mc_emit(_mc_ss.CALIB, **calib_startup_fields(_cp))` with a hand-rolled
    emit carrying only `threshold` and `brier` orphans the helper, strips
    `map` and `beats_base_rate` from the startup line, and leaves every pin
    green — including the new executed one, which tests the helper in
    isolation.

    `lifespan` IS drivable (test_biological_watchdog does it), so drive it.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from ghost_agent.core import metacog_log
    from ghost_agent.core.calibration import (BEATS_INDISTINGUISHABLE,
                                              CalibrationTracker, FittedParams)
    from ghost_agent.main import lifespan

    calib_dir = tmp_path / "calibration"
    calib_dir.mkdir()
    CalibrationTracker(calib_dir)._save_params(FittedParams(
        w_entropy=0.0, w_competence=0.4, threshold=0.84,
        lambda_uncertainty=0.0, brier=0.0468, n_samples=1254,
        fitted_at="2026-08-30T00:00:00Z", map_status="rejected_inverted",
        beats_base_rate=BEATS_INDISTINGUISHABLE))

    emitted = []
    monkeypatch.setattr(metacog_log, "emit",
                        lambda sub, **kw: emitted.append((sub, kw)))

    mock_app = MagicMock()
    mock_app.state.args = MagicMock()
    mock_app.state.args.no_memory = True
    for k in ("swarm_nodes_parsed", "worker_nodes_parsed",
              "visual_nodes_parsed", "coding_nodes_parsed",
              "image_gen_nodes_parsed"):
        setattr(mock_app.state.args, k, [])
    mock_app.state.args.upstream_url = "http://mock"
    ctx = MagicMock()
    ctx.tor_proxy = None
    ctx.memory_dir = str(tmp_path / "memory")
    ctx.calibration_tracker = CalibrationTracker(calib_dir)
    mock_app.state.context = ctx

    fake_agent = MagicMock()
    fake_agent.biological_watchdog = AsyncMock(side_effect=asyncio.sleep)

    with patch("ghost_agent.main.LLMClient") as MockLLM, \
         patch("ghost_agent.main.importlib.util.find_spec", return_value=False), \
         patch("ghost_agent.main.ProfileMemory"), \
         patch("ghost_agent.main.GraphMemory"), \
         patch("ghost_agent.main.GhostAgent", return_value=fake_agent):
        MockLLM.return_value = MagicMock(close=AsyncMock())
        async with lifespan(mock_app):
            pass

    startup = [kw for sub, kw in emitted
               if getattr(sub, "name", str(sub)).upper().endswith("CALIB")
               and kw.get("loaded") == "startup"]
    assert startup, (
        "the startup path emitted no CALIB line carrying loaded='startup'; "
        f"emitted={[(getattr(s, 'name', s), sorted(k)) for s, k in emitted]}")
    payload = startup[-1]
    # ⚠ THIS ASSERTION USED TO PIN THE DEFECT. It saved
    # `map_status="rejected_inverted"` with `beats_base_rate=indistinguishable`
    # and demanded that pair be emitted VERBATIM — so the startup line
    # reported a measured verdict about a map that does not ship, and fixing
    # it would have reddened this test. `load_params` is a reader of the
    # rejection rule like any other; it now asks `downgrade_for_map`, so the
    # licence is withheld before it ever reaches this line.
    from ghost_agent.core.calibration import BEATS_UNKNOWN
    assert payload.get("beats_base_rate") == BEATS_UNKNOWN, (
        f"the startup line reports {payload.get('beats_base_rate')!r} for a "
        f"REJECTED map — a verdict measured on a predictor that does not "
        f"ship: {payload!r}")
    assert payload.get("map") == "rejected_inverted", (
        f"and it must still carry the map status: {payload!r}")


# ── the rejection rule, for EVERY status and both callers ───────────────

@pytest.mark.parametrize("map_status,applied", [
    (None, False),                 # an explicit JSON null is malformed
    ("None", False),               # what `load_params`' str() makes of null
    ("applied", True),
    ("rejected_inverted", False),
    ("rejected_step", False),
    ("discarded_worse", False),
    ("", False),                   # read three different ways before
    ("wibble", False),             # a status from a future version
    (7, False),                    # not even a string
])
def test_every_map_status_gets_the_same_answer_everywhere(map_status, applied):
    """⚠ THE DOWNGRADE WAS PINNED FOR ONE OF ITS THREE STATUSES.

    The only covering test drove a real fit, and its chosen seed lands on
    `discarded_worse`. Its integrity assert (`map_status != "applied"`) is
    satisfied by ANY rejection, so exempting `rejected_inverted` —
    `if map_status not in ("applied", "rejected_inverted")` — survived the
    full suite while reproducing the change's headline defect verbatim:
    `rejected_inverted` + `beats_base_rate="yes"` about a shipped scorer
    with Brier 0.62 against a base rate of 0.18.

    `""` was read three ways at once: APPLIED by the renderer, REJECTED by
    the agent's CALIB line, and defaulted to applied only when the key was
    absent by `load_params`. One question, one answer.
    """
    from ghost_agent.core.calibration import (BEATS_UNKNOWN, BEATS_YES,
                                              downgrade_for_map, map_applied)

    assert map_applied(map_status) is applied
    expected = BEATS_YES if applied else BEATS_UNKNOWN
    assert downgrade_for_map(BEATS_YES, map_status) == expected, (
        f"map_status={map_status!r} must {'keep' if applied else 'withhold'} "
        "the licence")


def test_an_absent_map_status_is_not_the_same_as_an_explicit_null():
    """⚠ AGE IS NOT MALFORMEDNESS. A file with no `map_status` predates the
    field and reads as applied; `"map_status": null` is a broken value and
    must fail closed. Collapsing them let a null read as APPLIED in the
    learning-health collector while `load_params` — which stringifies it to
    `"None"` — failed closed on the same file."""
    from ghost_agent.core.calibration import map_applied_in_params

    assert map_applied_in_params({}) is True, (
        "a params file older than `map_status` must still derive normally")
    assert map_applied_in_params({"map_status": None}) is False, (
        "an explicit null is a malformed value, not an old file")
    assert map_applied_in_params({"map_status": "applied"}) is True


@pytest.mark.parametrize("corpus,expect_status", [
    ("anticorrelated", "rejected_inverted"),
    ("already_calibrated", "discarded_worse"),
])
def test_a_real_fit_withholds_the_licence_on_each_reachable_rejection(
        corpus, expect_status, tmp_path):
    """The executed counterpart: two DIFFERENT rejection statuses, each
    reached by a real fit, each required to withhold. One seed landing on
    one status is not coverage of a rule that spans three."""
    import random

    from ghost_agent.core.calibration import BEATS_UNKNOWN, CalibrationTracker

    rng = random.Random(7)
    tracker = CalibrationTracker(tmp_path)
    for _ in range(400):
        if corpus == "anticorrelated":
            y = rng.random() < 0.8
            c = rng.uniform(0.05, 0.4) if y else rng.uniform(0.6, 0.95)
            outcome = 1.0 if y else 0.0
        else:
            c = rng.random()
            outcome = 1.0 if rng.random() < c else 0.0
        tracker.record(composite=c, outcome=outcome,
                       entropy_component=0.5, competence_component=c)

    params = tracker.fit()
    assert params.map_status == expect_status, (
        f"fixture drifted: expected {expect_status}, got {params.map_status} "
        "— this parametrisation exists so the rule is pinned for MORE THAN "
        "ONE status, so a drift onto another one must fail loudly")
    assert params.beats_base_rate == BEATS_UNKNOWN, (
        f"{expect_status} ships the raw composite, so no comparison applies; "
        f"the fit recorded {params.beats_base_rate!r}")


def test_a_derived_verdict_on_a_rejected_map_is_also_withheld(tmp_path):
    """⚠ THE RULE HAD ONE HOME AND TWO CALLERS.

    `fit()` applied it; the renderer's DERIVE path — taken by every params
    file written before the field existed, the live one included — called
    `_base_rate_verdict`, which sees only an interval. Reproduced on a real
    anti-correlated fit with the field stripped (exactly the pre-change
    shape): the producer refused the licence, and the renderer printed
    "beats the base-rate predictor" for a scorer 3.5x worse than a constant.
    """
    rejected = _render_with(tmp_path / "a", map_status="rejected_inverted",
                            brier_cv_delta_lo=-0.18, brier_cv_delta_hi=-0.14)
    assert "beats the base-rate predictor" not in rejected, (
        f"the derive path granted a licence the producer refuses:\n"
        f"{rejected[:400]}")
    assert "NO base-rate comparison" in rejected, rejected[:400]

    # ...and it must still derive normally when the map IS in force.
    applied = _render_with(tmp_path / "b", map_status="applied",
                           brier_cv_delta_lo=-0.18, brier_cv_delta_hi=-0.14)
    assert "beats the base-rate predictor" in applied, (
        f"the fix broke the legitimate derive path:\n{applied[:400]}")
    assert "DERIVED at render time" in applied


def test_a_malformed_verdict_cannot_blank_the_whole_report(tmp_path):
    """⚠ ONE ADVISORY FIELD DESTROYED EVERY SECTION. `_stored not in _words`
    RAISES on an unhashable value, and `introspect` catches it and returns
    "Learning health unavailable: TypeError" — losing lessons, episodes, PRM
    and router reporting over a malformed calibration key."""
    for junk in ([], {}, [1, 2]):
        out = _render_with(tmp_path / f"d{id(junk)}", beats_base_rate=junk,
                           map_status="applied", brier_cv_delta_lo=-0.01,
                           brier_cv_delta_hi=-0.002)
        assert "LESSONS:" in out, (
            f"the report did not render at all for beats_base_rate={junk!r}")
        assert "NO usable base-rate verdict" in out, (
            f"a malformed verdict must be named, not silently dropped:\n"
            f"{out[:400]}")
        assert "beats the base-rate predictor" not in out


def test_the_indistinguishable_warning_prints_the_interval_in_order(
        tmp_path, caplog):
    """⚠ THE WARNING THAT EXISTS TO SHOW THE INTERVAL DID NOT PIN IT.
    Swapping `_delta_ci[1]` and `_delta_ci[2]` printed an INVERTED interval
    — `[+0.01322, -0.00300]` — in the one message whose whole purpose is to
    show that the interval straddles zero, and survived the suite."""
    import logging
    import re

    with caplog.at_level(logging.WARNING, logger="ghost_agent.core.calibration"):
        params = _fit_on(5, 120, 0.020, 0.6, tmp_path)

    msg = [r.getMessage() for r in caplog.records
           if "INDISTINGUISHABLE" in r.getMessage()]
    assert msg, "the indistinguishable warning did not fire"
    m = re.search(r"\[([+-][\d.]+), ([+-][\d.]+)\]", msg[-1])
    assert m, f"no interval in the warning: {msg[-1]}"
    lo, hi = float(m.group(1)), float(m.group(2))
    assert lo <= hi, (
        f"the warning printed an INVERTED interval [{lo}, {hi}] — an "
        "interval shown low-high is the only reason this message exists")
    assert (lo, hi) == (round(params.brier_cv_delta_lo, 5),
                        round(params.brier_cv_delta_hi, 5)), (
        f"the warning's interval [{lo}, {hi}] is not the fit's own "
        f"[{params.brier_cv_delta_lo}, {params.brier_cv_delta_hi}]")
    assert lo < 0.0 < hi, "fixture drifted: this interval must straddle zero"

    # ⚠ AND THE TWO BRIERS MUST BE THE RIGHT WAY ROUND. The message reads
    # "fitted model (CV Brier X) is INDISTINGUISHABLE from always predicting
    # the base rate P (Brier Y)" -- three numbers from three different
    # quantities. Swapping the model's CV Brier with the base RATE survived:
    # the sentence still parses, and every existing assertion was about the
    # interval. An operator reading `CV Brier 0.583` for a model whose Brier
    # is 0.248 cannot tell the instrument is lying.
    m2 = re.search(r"CV Brier ([\d.]+)\) is INDISTINGUISHABLE from always "
                   r"predicting the base rate ([\d.]+) \(Brier ([\d.]+)\)",
                   msg[-1])
    assert m2, f"the warning's shape changed: {msg[-1]}"
    shown_cv, shown_rate, shown_base_brier = (float(m2.group(i))
                                              for i in (1, 2, 3))
    assert abs(shown_cv - params.brier_cv) < 5e-4, (
        f"the warning reports CV Brier {shown_cv} but the fit's is "
        f"{params.brier_cv}")
    assert abs(shown_base_brier - params.brier_base_rate) < 5e-4, (
        f"the warning reports a base-rate Brier of {shown_base_brier} but "
        f"the fit's is {params.brier_base_rate}")
    assert 0.0 <= shown_rate <= 1.0, (
        f"the 'base rate' slot holds {shown_rate}, which is not a rate — the "
        "three numbers in this message are not the three quantities it names")


def test_the_refusal_wording_carries_its_evidence(tmp_path):
    """Both new branches were unpinned in their DETAIL: deleting the
    interval note, or trimming the rejection sentence to just the status,
    survived. The sentence has to say what ships and why the number beside
    it does not describe it."""
    rejected = _render_with(tmp_path / "a", beats_base_rate="unknown",
                            map_status="rejected_inverted",
                            brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002)
    line = [l for l in rejected.splitlines() if "base-rate" in l][0]
    assert "shipped predictor is the raw composite" in line, line
    assert "does NOT describe what ships" in line, line

    no_ci = _render_with(tmp_path / "b", beats_base_rate="unknown",
                         map_status="rejected_inverted")
    nline = [l for l in no_ci.splitlines() if "base-rate" in l][0]
    assert "interval measures the map we discarded" not in nline, (
        f"claimed an interval that is not on file:\n{nline}")
    assert "shipped predictor is the raw composite" in nline, nline


def test_a_bool_is_not_an_interval_bound_anywhere(tmp_path):
    """`_have_ci`'s isinstance half was unpinned — widening it to
    `_dlo is not None` survived, so a string bound would reach the `:+.5f`
    format and raise inside the renderer."""
    # ⚠ SYMMETRIC FIXTURES HIDE HALF A GUARD. Setting lo AND hi to the same
    # bad value meant dropping the `_dhi` half of the check survived — and a
    # string reaching `f"{_dhi:+.5f}"` raises out of the renderer.
    for bad in ("-0.01", None, [0.1]):
        for which, lo, hi in (("both", bad, bad),
                              ("lo only", bad, -0.002),
                              ("hi only", -0.01, bad)):
            out = _render_with(tmp_path / f"x{id(bad)}_{which.replace(' ','')}",
                               beats_base_rate="yes", map_status="applied",
                               brier_cv_delta_lo=lo, brier_cv_delta_hi=hi)
            assert "LESSONS:" in out, (
                f"the report failed to render for a {type(bad).__name__} "
                f"bound ({which})")
            assert "95% CI of the delta" not in out, (
                f"a {bad!r} bound ({which}) was rendered as a measured "
                f"interval:\n{out[:300]}")


def test_a_fit_the_agent_will_not_load_is_not_reported_as_live(tmp_path):
    """⚠ TWO READERS, TWO POLICIES, ONE FILE. `load_params` refuses a
    params file whose `schema` does not match and the agent runs on
    hardcoded defaults — while `learning_health` read the same file with no
    schema check and rendered a full CALIBRATION section, base-rate licence
    and all, for a fit that is in force nowhere."""
    out = _render_with(tmp_path, schema="0.0-ancient", beats_base_rate="yes",
                       map_status="applied", brier_cv_delta_lo=-0.01,
                       brier_cv_delta_hi=-0.002)
    assert "NOT IN FORCE" in out, (
        f"a fit the agent refuses to load was reported as live:\n{out[:500]}")
    assert "hardcoded defaults" in out, out[:500]

    from ghost_agent.core.calibration import SCHEMA_VERSION
    live = _render_with(tmp_path / "ok", schema=SCHEMA_VERSION,
                        beats_base_rate="yes", map_status="applied",
                        brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002)
    assert "NOT IN FORCE" not in live, (
        f"a CURRENT params file was flagged as stale:\n{live[:500]}")
    assert "beats the base-rate predictor" in live, live[:400]


def test_the_no_cv_branch_also_says_the_two_briers_are_one_number(tmp_path):
    """⚠ THE OTHER HALF OF THE LINE, AGAIN. The CV branch got the
    "these are the SAME number" qualification; the `no CV recorded` branch
    did not, and deleting it there survived the suite. On a rejection path
    `brier == brier_raw`, so that branch printed one value twice with
    nothing saying so — the shape the qualification exists to abolish."""
    out = _render_with(tmp_path, beats_base_rate="unknown",
                       map_status="rejected_inverted", brier=0.7,
                       brier_raw=0.7, brier_base_rate=0.2469, brier_cv=None)
    line = [l for l in out.splitlines() if "IN-SAMPLE" in l][0]
    assert "SAME number" in line, (
        f"one value printed twice with no qualification:\n{line}")
    assert "rejected_inverted" in line, line

    ok = _render_with(tmp_path / "ok", beats_base_rate="yes",
                      map_status="applied", brier=0.166, brier_raw=0.180,
                      brier_base_rate=0.2469, brier_cv=None)
    ok_line = [l for l in ok.splitlines() if "IN-SAMPLE" in l][0]
    assert "SAME number" not in ok_line, (
        f"an APPLIED map's two distinct Briers were called the same:\n{ok_line}")


# ── the RENDERER, driven with every status (the gap that hid three bugs) ──

@pytest.mark.parametrize("map_status,in_force", [
    ("applied", True),
    ("rejected_inverted", False),
    ("rejected_step", False),
    ("discarded_worse", False),
    ("", False),
    ("wibble", False),
    (7, False),
    (None, False),
])
def test_the_renderer_agrees_with_the_helper_for_every_status(
        map_status, in_force, tmp_path):
    """⚠ THE HELPERS WERE PINNED FOR EVERY VALUE; THE RENDERER FOR NONE.

    Three separate defects hid in that gap: a second copy of the rejection
    vocabulary (`not in ("", "applied")`) still deciding the in-sample line
    and contradicting the verdict line one row above it for `map_status=""`;
    the same copy at two more sites; and a status rendered as an empty gap
    in the sentence. Every one survived a suite that tested the helper
    exhaustively and never once drove the thing the helper exists for.

    `rejected_step` is here for a second reason: no real fit reaches it, so
    this is its ONLY end-to-end coverage of a rule its own docstring says
    spans three statuses.
    """
    # ⚠ BOTH BRANCHES. The CV branch and the `no CV recorded` branch print
    # this line from two different expressions, and only the first was
    # driven — so the second kept the pre-fix vocabulary and disagreed with
    # the first for `""` and for any non-string. Parametrising the status
    # and not the branch left half the rule untested.
    for label, cv in (("with_cv", 0.63), ("no_cv", None)):
        out = _render_with(tmp_path / f"{label}", beats_base_rate="yes",
                           map_status=map_status, brier=0.63, brier_raw=0.63,
                           brier_base_rate=0.18, brier_cv=cv,
                           brier_cv_delta_lo=-0.18, brier_cv_delta_hi=-0.14)
        verdict_line = [l for l in out.splitlines()
                        if "base-rate predictor" in l
                        or "base-rate comparison" in l][0]
        # ⚠ NOT THE HEADER, NOT THE VERDICT LINE. `CALIBRATION: … Brier X
        # (in-sample)` contains "in-sample", and the refusal sentence
        # contains "raw composite" ("the shipped predictor is the raw
        # composite"). Both matched before this, so the assertions below
        # were testing whichever line came first rather than the one under
        # test. The detail line is the only one with the "·" separator.
        # The CV branch prints a separate detail line ("· raw composite");
        # the no-CV branch folds everything onto one line ("; raw composite").
        _detail = [l for l in out.splitlines() if "· raw composite" in l]
        in_sample_line = (_detail[0] if _detail else
                          [l for l in out.splitlines() if "IN-SAMPLE" in l][0])

        if in_force:
            assert "beats the base-rate predictor" in verdict_line, verdict_line
            assert "SAME number" not in in_sample_line, in_sample_line
        else:
            assert "NO base-rate comparison" in verdict_line, (
                f"[{label}] map_status={map_status!r} is not in force, but "
                f"the renderer granted the licence:\n{verdict_line}")
            assert "beats the base-rate predictor" not in verdict_line
            # ⚠ AND THE TWO LINES MUST AGREE. They were decided by two
            # different expressions, so for `""` one said rejected and the
            # other applied, in the same block.
            assert "SAME number" in in_sample_line, (
                f"[{label}] the verdict line says the map was rejected and "
                f"the in-sample line does not:\n{verdict_line}\n"
                f"{in_sample_line}")
            # ⚠ AND THE STATUS MUST BE VISIBLE. Rendering it bare put an
            # empty gap in the sentence for `""`: "the Platt map was , so".
            assert "map was  " not in verdict_line, (
                f"[{label}] the status rendered as a blank:\n{verdict_line}")
            assert repr(map_status) in verdict_line or (
                isinstance(map_status, str) and map_status
                and map_status in verdict_line), (
                f"[{label}] the sentence does not name the status it is "
                f"reporting ({map_status!r}):\n{verdict_line}")


def test_a_stored_licence_on_a_rejected_map_is_overruled(tmp_path):
    """⚠ THE AUTHORITY WAS APPLIED TO DERIVED VERDICTS ONLY.

    Files where the field is ABSENT were protected; files where it is WRONG
    were trusted. And "wrong" is not hypothetical — `fit()` wrote exactly
    `map_status="rejected_inverted"` + `beats_base_rate="yes"` before this
    change, so every params file from that build carries it. Rendered: the
    licence on one line, the admission that the map was discarded on the
    next."""
    out = _render_with(tmp_path, beats_base_rate="yes",
                       map_status="rejected_inverted", brier=0.63,
                       brier_raw=0.62, brier_base_rate=0.18, brier_cv=0.63,
                       brier_cv_delta_lo=-0.18, brier_cv_delta_hi=-0.14)
    assert "beats the base-rate predictor" not in out, (
        f"a STORED licence survived a rejected map:\n{out[:500]}")
    assert "NO base-rate comparison" in out, out[:500]


def test_no_malformed_number_can_blank_the_report(tmp_path):
    """⚠ SAME BLAST RADIUS, ONE VARIABLE OVER. The verdict coercion stopped
    an unhashable `beats_base_rate` from raising — but `cal["brier"]` was
    unguarded, so a missing key, a null or a string reached the `_score < _bb`
    comparison and raised TypeError, which `introspect` reports as "Learning
    health unavailable", losing every other section."""
    # ⚠ `brier` IS ONLY READ WHEN THERE IS NO USABLE `brier_cv`. The first
    # version of this test left `brier_cv` at its fixture default, so
    # `_honest` was always set and the `brier` fallback never ran — the
    # unguarded expression survived every case. Drop brier_cv in the cases
    # that are about brier.
    for label, extra in [("missing brier", {"brier": None, "brier_cv": None}),
                         ("string brier", {"brier": "0.166", "brier_cv": None}),
                         ("bool brier", {"brier": True, "brier_cv": None}),
                         ("absent brier", {"brier": None, "brier_cv": -1}),
                         ("bool brier_cv", {"brier_cv": True}),
                         ("string base rate", {"brier_base_rate": "x"})]:
        out = _render_with(tmp_path / label.replace(" ", "_"),
                           beats_base_rate="yes", map_status="applied",
                           brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002,
                           **extra)
        assert "LESSONS:" in out, (
            f"the whole report died on {label}: {out[:200]}")
        assert "Brier True" not in out, (
            f"a bool was rendered as a Brier score ({label})")

    # ⚠ AND THE COMPARISON ITSELF. Every case above stores a verdict, so
    # `_stored in _words` short-circuits and `_score` is never compared —
    # which is why an unguarded `_score = ... else cal.get("brier")` survived
    # all of them. The point-estimate arm is only reached with NO stored
    # verdict and NO interval: the oldest legacy shape. That is the path
    # where a string or a null `brier` reaches `_score < _bb` and raises.
    for label, extra in [("string brier, no verdict", {"brier": "0.166"}),
                         ("null brier, no verdict", {"brier": None}),
                         ("bool brier, no verdict", {"brier": True})]:
        out = _render_with(tmp_path / label.replace(" ", "_").replace(",", ""),
                           map_status="applied", brier_cv=None, **extra)
        assert "LESSONS:" in out, (
            f"the whole report died on {label}: {out[:200]}")
        assert "NO USABLE BRIER" in out, (
            f"{label}: an unusable Brier must be named, not compared:\n"
            f"{out[:300]}")


def test_a_fit_with_unusable_numerics_is_flagged_not_in_force(tmp_path):
    """⚠ THE FLAG MIRRORED ONE OF `load_params`' FOUR REFUSALS. It compared
    `schema` only, while the loader also returns None on bad JSON, a
    non-dict, and any unusable required numeric — so `"threshold": "abc"`
    made the agent fall back to hardcoded defaults while this report still
    printed the fit and its licence."""
    from ghost_agent.core.calibration import SCHEMA_VERSION

    for label, extra in [("bad threshold", {"threshold": "abc"}),
                         ("missing weight", {"w_competence": None})]:
        out = _render_with(tmp_path / label.replace(" ", "_"),
                           schema=SCHEMA_VERSION, beats_base_rate="yes",
                           map_status="applied", brier_cv_delta_lo=-0.01,
                           brier_cv_delta_hi=-0.002, **extra)
        assert "NOT IN FORCE" in out, (
            f"{label}: `load_params` refuses this file, but the report "
            f"presents it as the live scorer:\n{out[:500]}")


def test_a_future_schema_is_not_accepted_by_prefix(tmp_path):
    """The schema check must be equality. A `ghost.calibration.v2` file is
    one `load_params` refuses, and a prefix test would call it live."""
    out = _render_with(tmp_path, schema="ghost.calibration.v2",
                       beats_base_rate="yes", map_status="applied",
                       brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002)
    assert "NOT IN FORCE" in out, (
        f"a future schema was accepted as live:\n{out[:400]}")
    assert "'ghost.calibration.v2'" in out, (
        "the report must name the schema it found")


def test_the_no_verdict_refusal_still_shows_the_interval_on_file(tmp_path):
    """Fix 5's evidence note on the *no usable verdict* branch was unpinned —
    deleting it wholesale survived. Reachable: a bootstrap yielding NaN
    bounds stores `unknown` alongside a CI, and this note is the operator's
    only clue that the interval is what is broken."""
    out = _render_with(tmp_path, beats_base_rate="unknown",
                       map_status="applied", brier_cv_delta_lo=-0.01,
                       brier_cv_delta_hi=-0.002)
    line = [l for l in out.splitlines() if "base-rate comparison" in l][0]
    assert "interval on file" in line, (
        f"the refusal drops the only evidence on file:\n{line}")
    assert "-0.01000..-0.00200" in line, line
    # ⚠ AND IT MUST NOT DENY THE INTERVAL IT IS ABOUT TO PRINT. The earlier
    # version of this pin asserted only the second half of a sentence whose
    # first half read "no usable cross-validated interval was recorded" —
    # pinning one clause of a self-contradicting line.
    assert "no usable cross-validated interval" not in line, (
        f"the line denies the interval it then prints:\n{line}")


def test_the_oldest_legacy_shape_keeps_its_point_estimate_caveat(tmp_path):
    """⚠ THE BRANCH THE OLDEST FILES TAKE HAD NO TEST AT ALL. Verdict
    ABSENT and no interval: two mutants survived — dropping the `_have_ci`
    condition from the derive guard, and collapsing all three point-estimate
    arms to "matches"."""
    # ⚠ ALL THREE CASES BELOW USED map_status="applied", so the arms they
    # claim to cover were never tested against a REJECTED map — and those
    # three arms turned out to be the only ones `_map_in_force` did not
    # gate. The counterweight is at the bottom of this test.
    beats = _render_with(tmp_path / "a", brier=0.0468, brier_cv=0.0468,
                         brier_base_rate=0.0600, map_status="applied")
    line = [l for l in beats.splitlines() if "base-rate predictor" in l][0]
    assert "point estimate only" in line, (
        f"a point comparison with no interval must say so:\n{line}")
    assert "beats" in line, line

    loses = _render_with(tmp_path / "b", brier=0.0900, brier_cv=0.0900,
                         brier_base_rate=0.0500, map_status="applied")
    l2 = [l for l in loses.splitlines() if "base-rate predictor" in l][0]
    assert "LOSES TO" in l2, (
        f"a measurably worse model must not read as a tie:\n{l2}")
    assert "point estimate only" in l2, l2

    tie = _render_with(tmp_path / "c", brier=0.0500, brier_cv=0.0500,
                       brier_base_rate=0.0500, map_status="applied")
    l3 = [l for l in tie.splitlines() if "base-rate predictor" in l][0]
    assert "matches" in l3, l3

    # ⚠ AND THE SAME SHAPE WITH A REJECTED MAP MUST NOT REACH THOSE ARMS AT
    # ALL. This is the population the whole change is about: a params file
    # from any build between 2026-07-29 (map_status) and 2026-08-21 (the CI)
    # has a map status, no interval and no verdict. Before this, the point
    # comparison granted "beats the base-rate predictor" for EVERY status —
    # byte-identical output for applied, rejected_inverted, rejected_step,
    # discarded_worse and "" — directly above the line explaining that the
    # number behind it describes the map that was discarded.
    for status in ("rejected_inverted", "rejected_step", "discarded_worse", ""):
        out = _render_with(tmp_path / f"rej_{status or 'blank'}",
                           brier=0.0468, brier_cv=0.0468,
                           brier_base_rate=0.0600, map_status=status)
        line = [l for l in out.splitlines()
                if "base-rate predictor" in l or "base-rate comparison" in l][0]
        assert "beats" not in line, (
            f"map_status={status!r}: a point comparison granted the licence "
            f"for a map that does not ship:\n{line}")
        assert "NO base-rate comparison" in line, (
            f"map_status={status!r}:\n{line}")


def test_the_not_in_force_line_names_the_schema_it_found(tmp_path):
    """`schema_on_disk` hardcoded to None survived — the operator would be
    told a file is stale without being told what it carries."""
    out = _render_with(tmp_path, schema="ghost.calibration.v0-old",
                       beats_base_rate="yes", map_status="applied",
                       brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002)
    assert "'ghost.calibration.v0-old'" in out, (
        f"the report does not name the schema on disk:\n{out[:400]}")


def test_a_collector_block_without_map_status_reads_as_applied(tmp_path):
    """⚠ THE THIRD PLACE ABSENCE MEANS SOMETHING DIFFERENT FROM NULL.

    `cal` is the COLLECTOR's dict, not the params file. A block with no
    `map_status` key comes from a collector older than the field, and must
    read as applied — same reasoning as an old params file. Reading it with
    `.get()` collapsed absent into null, so every pre-existing caller
    passing a hand-rolled block (the calibration audit's fixture among them)
    suddenly rendered as a rejected map and lost the "in-sample is NOT
    performance" line.

    Pinned directly on the renderer's decision function so it cannot drift
    from the params-side rule again.
    """
    from ghost_agent.core.calibration import map_applied_in_params

    assert map_applied_in_params({"brier": 0.02}) is True, (
        "a collector block predating `map_status` must read as applied")
    assert map_applied_in_params({"map_status": None}) is False, (
        "an explicit null must still fail closed")
    assert map_applied_in_params({"map_status": "rejected_step"}) is False

    # And end-to-end: a params file with no map_status at all.
    out = _render_with(tmp_path, beats_base_rate="yes",
                       brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002)
    assert "beats the base-rate predictor" in out, (
        f"a params file older than `map_status` was treated as rejected:\n"
        f"{out[:400]}")
    assert "in-sample is NOT performance" in out, out[:400]


def test_one_malformed_field_never_costs_another_section(tmp_path):
    """⚠ THE SUCCESS SIGNAL WAS TOO WEAK TO SEE THE DAMAGE.

    The earlier version of this check asserted only `"LESSONS:" in out` — so
    it passed while a bad `brier_base_rate` silently amputated EIGHT lines:
    the verdict, the under-powered notice, the CONSUMER DEAD line and the
    whole feature-ablation table, all nested inside one `isinstance` guard.
    "The report still rendered" is not the property; "the report still says
    what it knows, and says what it could not read" is.
    """
    from ghost_agent.core.calibration import SCHEMA_VERSION as _SCHEMA

    good = _render_with(tmp_path / "good", beats_base_rate="yes",
                        map_status="applied", brier_cv_delta_lo=-0.01,
                        brier_cv_delta_hi=-0.002,
                        feature_contrib={"effort_component": 0.0007})
    assert "feature ABLATION" in good, good[:400]

    for label, extra in [
            ("string base rate", {"brier_base_rate": "x"}),
            ("bool raw brier", {"brier_raw": True}),
            ("list feature_contrib", {"feature_contrib": [1, 2]}),
            ("string in feature_contrib", {"feature_contrib": {"a": "0.1"}}),
            ("null in feature_contrib", {"feature_contrib": {"a": None}}),
            # ⚠ SCHEMA MUST MATCH OR `load_params` BAILS BEFORE `int()`.
            # Without it the file is refused at the schema check and the
            # OverflowError path is never reached — the fixture tested
            # nothing.
            ("infinite n_samples", {"n_samples": float("inf"),
                                    "schema": _SCHEMA}),
            ("huge n_samples", {"n_samples": 1e400, "schema": _SCHEMA}),
    ]:
        out = _render_with(tmp_path / label.replace(" ", "_"),
                           beats_base_rate="yes", map_status="applied",
                           brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002,
                           **extra)
        assert "LESSONS:" in out, f"{label} blanked the report: {out[:200]}"
        # ⚠ AND NO BOOL OR STRING MAY BE RENDERED AS A NUMBER. Asserting
        # only that a "⚠" appears somewhere is too weak a signal — the
        # report carries unrelated warnings, so it was satisfied while
        # `_n` handed a bool straight through to the verdict line.
        for bad_render in ("raw composite True", "predictor (True)",
                           "in-sample True", "Brier True",
                           "raw composite x", "predictor ('x')"):
            assert bad_render not in out, (
                f"{label}: rendered {bad_render!r} — a non-number presented "
                f"as a measurement:\n{out[:500]}")

    # ⚠ AND AN OMISSION MUST BE NAMED. A bad baseline silently removed the
    # verdict, the under-powered notice, the CONSUMER DEAD line and the
    # whole ablation table — eight lines — with the suite green because
    # nothing asserted the report SAYS what it could not read.
    amputated = _render_with(tmp_path / "amputated", beats_base_rate="yes",
                             map_status="applied", brier_base_rate="x",
                             brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002,
                             feature_contrib={"effort_component": 0.0007})
    assert "NO USABLE BASELINE" in amputated, (
        f"the verdict and feature table were dropped with no notice:\n"
        f"{amputated[:500]}")
    assert "feature ABLATION" not in amputated, (
        "a table scored against an unreadable baseline must not be shown")


def test_a_non_dict_params_file_does_not_blank_the_report(tmp_path):
    """The in-force flag's own comment lists "a non-dict" as a refusal it
    reports — while the code six lines later did `params.get("schema")` and
    raised AttributeError on exactly that."""
    import json

    from ghost_agent.core.learning_health import render_learning_health

    for blob in ([1, 2, 3], "hello", 42):
        d = tmp_path / f"nd{abs(hash(str(blob)))}"
        mem = d / "memory"
        mem.mkdir(parents=True)
        calib = d / "calibration"
        calib.mkdir(parents=True)
        (calib / "calibration.jsonl").write_text("")
        (calib / "calibration_params.json").write_text(json.dumps(blob))
        out = render_learning_health(mem)
        assert "LESSONS:" in out, (
            f"a {type(blob).__name__} params file blanked the report")


def test_the_loader_never_hands_out_a_licence_for_a_rejected_map(tmp_path):
    """⚠ THE THIRD READER. `fit()` and the renderer both route the verdict
    through `downgrade_for_map`; `load_params` did not, so one file gave
    opposite answers on two operator surfaces — the startup CALIB line said
    `{map: rejected_inverted, beats_base_rate: yes}` while `introspect
    learning` said no comparison applied.

    Also pins the absent-vs-null form: `.get(k, default)` and
    `.get(k) or default` differ exactly on an explicit null, and the second
    would load `null` and `""` as an APPLIED map."""
    import json

    from ghost_agent.core.calibration import (BEATS_UNKNOWN, SCHEMA_VERSION,
                                              CalibrationTracker, map_applied)

    base = {"schema": SCHEMA_VERSION, "w_entropy": 0.0, "w_competence": 0.4,
            "threshold": 0.84, "lambda_uncertainty": 0.0, "brier": 0.05,
            "n_samples": 100, "fitted_at": "2026-08-30T00:00:00Z"}

    def _load(**extra):
        d = tmp_path / f"l{abs(hash(str(sorted(extra.items()))))}"
        d.mkdir()
        (d / "calibration_params.json").write_text(json.dumps({**base, **extra}))
        return CalibrationTracker(d).load_params()

    for status in ("rejected_inverted", "rejected_step", "discarded_worse"):
        p = _load(map_status=status, beats_base_rate="yes")
        assert p.beats_base_rate == BEATS_UNKNOWN, (
            f"{status}: the loader handed out a stored licence for a map "
            f"that does not ship ({p.beats_base_rate!r})")

    keep = _load(map_status="applied", beats_base_rate="yes")
    assert keep.beats_base_rate == "yes", (
        "the loader must still pass through a legitimate licence")

    null = _load(map_status=None, beats_base_rate="yes")
    assert not map_applied(null.map_status), (
        f"an explicit null map_status loaded as {null.map_status!r}, which "
        "reads as a map in force")
    assert null.beats_base_rate == BEATS_UNKNOWN


def test_every_brier_display_site_is_guarded(tmp_path):
    """`_brier_shown` has three call sites and only one was pinned, so
    reverting either of the others to the raw `cal['brier']` survived —
    printing `in-sample True` on the applied-map branch."""
    for status in ("applied", "rejected_inverted"):
        out = _render_with(tmp_path / status, beats_base_rate="yes",
                           map_status=status, brier=True, brier_cv=0.0468,
                           brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002)
        assert "in-sample True" not in out, (
            f"map_status={status}: a bool rendered as an in-sample Brier:\n"
            f"{out[:400]}")
        assert "UNREADABLE" in out, (
            f"map_status={status}: an unusable Brier was neither shown nor "
            f"named:\n{out[:400]}")


# ── round 7: the rule that went to the neighbour, and the NaN crack ──────

@pytest.mark.parametrize("label,extra,licensed", [
    ("absent", {}, True),
    ("explicit null", {"beats_base_rate": None}, False),
    ("stored unknown", {"beats_base_rate": "unknown"}, False),
    ("typo", {"beats_base_rate": "typo"}, False),
    ("stored yes", {"beats_base_rate": "yes"}, True),
])
def test_null_is_not_more_permissive_than_a_typo(label, extra, licensed,
                                                 tmp_path):
    """⚠ THE ABSENT-VS-NULL RULE WENT TO THE NEIGHBOUR.

    `map_status` got `map_applied_in_params`, a guarded collector forward, a
    comment and a test. `beats_base_rate` — the key this change is ABOUT —
    kept a bare `.get()`, so absent and null collapsed and the renderer read
    a null as "legacy file, please derive". That made `null` the single most
    permissive value the field accepts: a stored `unknown` withholds, an
    outright typo withholds, and `null` gets "beats the base-rate predictor"
    plus a DERIVED tag asserting the file predates a field it contains.
    """
    out = _render_with(tmp_path / label.replace(" ", "_"),
                       map_status="applied", brier_cv_delta_lo=-0.010,
                       brier_cv_delta_hi=-0.002, **extra)
    line = [l for l in out.splitlines()
            if "base-rate predictor" in l or "base-rate comparison" in l][0]
    if licensed:
        assert "beats the base-rate predictor" in line, f"{label}: {line}"
    else:
        assert "beats the base-rate predictor" not in line, (
            f"{label}: a malformed stored verdict was granted the licence "
            f"that an honest 'unknown' is refused:\n{line}")
    # Only a genuinely absent field may claim to predate the field.
    if label != "absent":
        assert "DERIVED at render time" not in out, (
            f"{label}: the report claims this file predates a field it "
            f"contains:\n{line}")

    # ⚠ AND THE LOADER MUST AGREE. It derives for an ABSENT field, so the
    # absent-vs-null distinction has to hold there too — `.get(k) is not
    # None` and `k in d` differ on exactly one value, and it is this one.
    import json

    from ghost_agent.core.calibration import (SCHEMA_VERSION,
                                              CalibrationTracker)
    d = tmp_path / f"load_{label.replace(' ', '_')}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "calibration_params.json").write_text(json.dumps({
        "schema": SCHEMA_VERSION, "w_entropy": 0.0, "w_competence": 0.4,
        "threshold": 0.84, "lambda_uncertainty": 0.0, "brier": 0.0468,
        "n_samples": 1254, "fitted_at": "2026-08-30T00:00:00Z",
        "map_status": "applied", "brier_cv_delta_lo": -0.010,
        "brier_cv_delta_hi": -0.002, **extra}))
    loaded = CalibrationTracker(d).load_params()
    assert (loaded.beats_base_rate == "yes") is licensed, (
        f"{label}: load_params says {loaded.beats_base_rate!r} while the "
        f"report says {'beats' if licensed else 'no licence'} — the two "
        "surfaces must not disagree about one file")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_baseline_is_announced_not_vanished(bad, tmp_path):
    """⚠ TWO PREDICATES THAT WERE NOT COMPLEMENTS. `NaN < 0` and
    `NaN >= 0` are BOTH False, so a NaN baseline fell through the
    announcement AND the block: the file loads (no NOT IN FORCE), nothing
    says NO USABLE BASELINE, and the verdict line, the in-sample line, the
    negative-class line, the CONSUMER DEAD notice and the whole ablation
    table silently disappear — the same eight-line amputation the
    announcement exists to abolish, for the one value class the sibling
    verdict function hardened with `math.isfinite`."""
    for key in ("brier_raw", "brier_base_rate"):
        out = _render_with(tmp_path / f"{key}{bad}", beats_base_rate="yes",
                           map_status="applied", brier_cv_delta_lo=-0.01,
                           brier_cv_delta_hi=-0.002,
                           feature_contrib={"effort_component": 0.0007},
                           **{key: bad})
        assert "NO USABLE BASELINE" in out, (
            f"{key}={bad!r}: the verdict and feature table vanished with no "
            f"notice:\n{out[:500]}")
        assert "feature ABLATION" not in out, (
            f"{key}={bad!r}: a table scored against a non-finite baseline")


def test_a_non_finite_interval_is_not_printed_as_a_measurement(tmp_path):
    """`_base_rate_verdict` rejects non-finite bounds; `_have_ci`, the
    DISPLAY guard for the same two values, did not — so infinite bounds
    printed `[95% CI of the delta -inf..+inf]` beside a real stored
    verdict."""
    out = _render_with(tmp_path, beats_base_rate="yes", map_status="applied",
                       brier_cv_delta_lo=float("-inf"),
                       brier_cv_delta_hi=float("inf"))
    assert "-inf" not in out and "+inf" not in out, (
        f"a non-finite interval was rendered as a measurement:\n{out[:400]}")
    assert "point estimate only" in out, (
        "with no usable interval the honesty caveat must be present")


@pytest.mark.parametrize("key", ["n_negative", "n_samples"])
def test_the_negative_class_counts_reject_bools(key, tmp_path):
    """The comment above this line calls it "THE NUMBER EVERY OTHER VERDICT
    RESTS ON", and `isinstance(True, int)` is True. Every sibling guard in
    this block rejects bools; this one did not, so `n_negative: true`
    rendered `negative class: True/1258 (0.1%) ⚠ UNDER-POWERED` on a file
    certified in force, with no warning anywhere."""
    out = _render_with(tmp_path / key, beats_base_rate="yes",
                       map_status="applied", brier_cv_delta_lo=-0.01,
                       brier_cv_delta_hi=-0.002, **{key: True})
    assert "negative class: True" not in out, out[:400]
    assert "/True" not in out, out[:400]


def test_a_bool_cannot_top_the_ablation_table(tmp_path):
    """The boundary filter rejects non-numerics; the bool clause was
    unpinned because no fixture used one. `{"entropy_component": true}`
    renders `+1.000000 helps` and sorts to the TOP of the table as the
    largest possible held-out delta."""
    out = _render_with(tmp_path, beats_base_rate="yes", map_status="applied",
                       brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002,
                       feature_contrib={"entropy_component": True,
                                        "effort_component": 0.0007})
    assert "+1.000000" not in out, (
        f"a bool was ranked as the strongest feature:\n{out[:500]}")
    assert "1 feature-ablation entry unusable" in out, (
        f"the dropped entry must be counted, not silently omitted:\n"
        f"{out[:500]}")
    assert "effort_component" in out, "the usable entry must survive"


def test_a_refused_file_with_no_baseline_still_says_not_in_force(tmp_path):
    """⚠ THE EXACT WORLD FIX #6 NAMES, AND NO TEST BUILT IT. Every case
    asserting NOT IN FORCE carried valid `brier_raw`/`brier_base_rate`
    defaults, so re-nesting the notice behind those two values survived."""
    out = _render_with(tmp_path, schema="ghost.calibration.v0-STALE",
                       beats_base_rate="yes", map_status="applied",
                       brier_raw=None, brier_base_rate=None)
    assert "NOT IN FORCE" in out, (
        f"a refused file that also lacks its baseline reported nothing:\n"
        f"{out[:500]}")
    assert "NO USABLE BASELINE" in out, out[:500]


def test_the_omission_notices_name_and_count_what_they_dropped(tmp_path):
    """Announcement lines are the fixes' own claimed property, and all four
    were unpinned: the baseline notice could stop naming the values, and the
    ablation notice could stop counting."""
    baseline = _render_with(tmp_path / "b", beats_base_rate="yes",
                            map_status="applied", brier_base_rate="x")
    assert "brier_base_rate='x'" in baseline, (
        f"the notice must name the value it could not read:\n{baseline[:400]}")

    dropped = _render_with(tmp_path / "d", beats_base_rate="yes",
                           map_status="applied", brier_cv_delta_lo=-0.01,
                           brier_cv_delta_hi=-0.002,
                           feature_contrib={"a": "x", "b": None,
                                            "effort_component": 0.0007})
    assert "2 feature-ablation entries unusable" in dropped, (
        f"the notice must COUNT what it dropped:\n{dropped[:500]}")

    listed = _render_with(tmp_path / "l", beats_base_rate="yes",
                          map_status="applied", brier_cv_delta_lo=-0.01,
                          brier_cv_delta_hi=-0.002, feature_contrib=[1, 2])
    assert "not a mapping" in listed, (
        f"a non-mapping feature_contrib must be named, not silently "
        f"omitted:\n{listed[:500]}")


def test_the_verdict_vocabulary_in_messages_is_all_four_words(tmp_path):
    """`sorted(_words)` lists the three PRINTABLE verdicts; quoting it as
    the vocabulary omits `unknown`, which is a verdict."""
    out = _render_with(tmp_path, beats_base_rate="typo", map_status="applied",
                       brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002)
    line = [l for l in out.splitlines() if "not one of" in l][0]
    for word in ("yes", "no", "indistinguishable", "unknown"):
        assert f"'{word}'" in line, (
            f"the vocabulary quoted to the operator omits {word!r}:\n{line}")


def test_every_surface_tells_one_story_about_one_file(tmp_path):
    """⚠ CROSS-SURFACE, NOT PER-SURFACE. Each reader was pinned in
    isolation, so the loader deriving nothing while the renderer derives
    went unseen for six rounds — and the live params file is exactly that
    shape (no stored verdict, a CI on file)."""
    import json

    from ghost_agent.core.calibration import (SCHEMA_VERSION,
                                              CalibrationTracker)
    from ghost_agent.core.learning_health import render_learning_health
    from ghost_agent.main import calib_startup_fields

    for label, status, ci_hi, want_licence in [
            ("clear win, applied", "applied", -0.002, True),
            ("clear win, rejected", "rejected_inverted", -0.002, False),
            ("straddles zero", "applied", 0.004, False),
    ]:
        d = tmp_path / label.replace(" ", "_").replace(",", "")
        mem = d / "memory"
        mem.mkdir(parents=True)
        calib = d / "calibration"
        calib.mkdir(parents=True)
        (calib / "calibration.jsonl").write_text("")
        (calib / "calibration_params.json").write_text(json.dumps({
            "schema": SCHEMA_VERSION, "w_entropy": 0.0, "w_competence": 0.4,
            "threshold": 0.84, "lambda_uncertainty": 0.0, "brier": 0.0468,
            "brier_raw": 0.0468, "brier_base_rate": 0.06, "brier_cv": 0.0468,
            "n_samples": 1254, "samples": 1254, "n_negative": 18,
            "map_status": status, "fitted_at": "2026-08-30T00:00:00Z",
            "brier_cv_delta_lo": -0.010, "brier_cv_delta_hi": ci_hi}))

        params = CalibrationTracker(calib).load_params()
        assert params is not None, f"{label}: the file did not load"
        loaded = params.beats_base_rate
        startup = calib_startup_fields(params)["beats_base_rate"]
        stats = CalibrationTracker(calib).stats().get("beats_base_rate")
        rendered = render_learning_health(mem)
        renders_licence = "beats the base-rate predictor" in rendered

        assert loaded == startup == stats, (
            f"{label}: three loader-side surfaces disagree — load_params="
            f"{loaded!r} startup={startup!r} stats={stats!r}")
        assert (loaded == "yes") is want_licence, (
            f"{label}: load_params says {loaded!r}")
        assert renders_licence is want_licence, (
            f"{label}: the report says {'beats' if renders_licence else 'no'} "
            f"while every other surface says {loaded!r}")


@pytest.mark.parametrize("key", ["threshold", "w_entropy", "w_competence",
                                 "w_effort"])
@pytest.mark.parametrize("bad", [True, "abc", [1, 2], float("nan")])
def test_the_header_numbers_are_guarded_too(key, bad, tmp_path):
    """⚠ THE THIRD VALUE IN A GUARDED F-STRING. `_hdr_num` covers `brier_cv`
    and `brier` in the header line; `threshold` sits in the SAME f-string,
    and `threshold: true` LOADS FINE (`float(True)` is 1.0), so the section
    printed `threshold True` on a file certified in force. The three weights
    on the next line had the same gap. Guarding the values that have bitten
    so far is how the neighbour keeps biting."""
    out = _render_with(tmp_path / f"{key}_{type(bad).__name__}",
                       beats_base_rate="yes", map_status="applied",
                       brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002,
                       **{key: bad})
    for rendered in (f"threshold {bad}", f"entropy {bad}",
                     f"competence {bad}", f"effort {bad}"):
        assert rendered not in out, (
            f"{key}={bad!r} was rendered as a number:\n{out[:400]}")
    assert "UNREADABLE" in out, (
        f"{key}={bad!r} was neither shown nor named:\n{out[:400]}")


# ══ properties, not sites — the three pins that end the neighbour hunt ═══

def test_every_numeric_read_goes_through_the_guard():
    """⚠ THE PIN THAT MAKES SITE-BY-SITE REVIEW UNNECESSARY.

    Six review rounds each hardened the numeric reads the previous round had
    looked at and missed a neighbour: the bool clause reached `_num` and not
    `n_negative`; `math.isfinite` reached three of six guards; the `>= 0`
    sentinel check reached three of four Brier sites. Every round found a
    real defect and every round found the SAME defect, because the
    reviewer's attention defined the boundary.

    This asserts the boundary mechanically: every numeric key read off
    `calibration_params.json` inside the calibration section must pass
    through `usable_number` (directly or via one of the local wrappers that
    delegate to it) before it reaches an f-string or a comparison. A new
    numeric key added without a guard fails here, with no reviewer involved.
    """
    import ast
    import inspect

    from ghost_agent.core import learning_health as LH

    src = inspect.getsource(LH)
    tree = ast.parse(src)

    # every wrapper that delegates to the single guard
    delegating = {"usable_number", "_num", "_n", "_hdr_num", "_show"}
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name in delegating
               and n.name != "usable_number"]:
        body = ast.dump(fn)
        assert "usable_number" in body, (
            f"`{fn.name}` no longer delegates to the single guard — it is a "
            "private copy, which is exactly how the guards drifted apart")

    NUMERIC_KEYS = {
        "brier", "brier_cv", "brier_raw", "brier_base_rate", "threshold",
        "w_entropy", "w_competence", "w_effort", "lambda_uncertainty",
        "brier_cv_delta_lo", "brier_cv_delta_hi", "platt_a",
    }

    # Collect `cal[...]` / `cal.get(...)` reads of a numeric key and check
    # each is an argument to a delegating guard.
    guarded, bare = set(), []

    class V(ast.NodeVisitor):
        def visit_Call(self, node):
            fname = getattr(node.func, "id", None) or getattr(
                node.func, "attr", None)
            if fname in delegating:
                for a in ast.walk(node):
                    if isinstance(a, ast.Constant) and a.value in NUMERIC_KEYS:
                        guarded.add((a.value, a.lineno))
            self.generic_visit(node)

    V().visit(tree)

    for node in ast.walk(tree):
        key = lineno = None
        if (isinstance(node, ast.Subscript)
                and getattr(node.value, "id", None) == "cal"
                and isinstance(node.slice, ast.Constant)
                and node.slice.value in NUMERIC_KEYS):
            key, lineno = node.slice.value, node.lineno
        elif (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "get"
                and getattr(node.func.value, "id", None) == "cal"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in NUMERIC_KEYS):
            key, lineno = node.args[0].value, node.lineno
        if key is not None and (key, lineno) not in guarded:
            bare.append((key, lineno))

    # `!r` reprs inside warning text are deliberate: they report the raw
    # value the guard rejected, so they are allowed.
    src_lines = src.splitlines()
    bare = [(k, ln) for k, ln in bare
            if "!r}" not in src_lines[ln - 1]
            and "schema_on_disk" not in src_lines[ln - 1]]

    assert not bare, (
        "these numeric reads bypass the single guard, which is how every "
        "previous round's fix left a neighbour unguarded:\n  "
        + "\n  ".join(f"{k!r} at learning_health.py:{ln}" for k, ln in bare))


@pytest.mark.parametrize("bad", [
    True, False, "0.5", "abc", None, [1, 2], {"a": 1},
    float("nan"), float("inf"), float("-inf"), -1.0, -0.5, 1e400,
])
def test_no_hostile_value_in_any_key_breaks_or_fabricates(bad, tmp_path):
    """⚠ A PROPERTY OVER THE WHOLE KEY SET, NOT A LIST OF KEYS THAT BIT.

    Three separate rounds fixed `brier`, then `brier_raw`/`brier_base_rate`,
    then `n_negative` — each time by adding the key that had just been
    demonstrated. This asserts the invariant over EVERY key instead, so the
    next key is covered before anyone demonstrates it:

      1. the report never raises (`introspect` turns any raise into
         "Learning health unavailable", losing lessons, episodes, PRM and
         router reporting too);
      2. no unusable value is ever rendered as a number; and
      3. nothing is dropped in silence.
    """
    import json

    from ghost_agent.core.calibration import SCHEMA_VERSION
    from ghost_agent.core.learning_health import render_learning_health

    KEYS = ["brier", "brier_cv", "brier_raw", "brier_base_rate", "threshold",
            "w_entropy", "w_competence", "w_effort", "n_samples",
            "n_negative", "map_status", "beats_base_rate",
            "brier_cv_delta_lo", "brier_cv_delta_hi", "feature_contrib",
            "epoch", "fitted_at", "schema"]

    for key in KEYS:
        d = tmp_path / f"{key}_{abs(hash(str(bad)))}"
        mem = d / "memory"
        mem.mkdir(parents=True)
        calib = d / "calibration"
        calib.mkdir(parents=True)
        (calib / "calibration.jsonl").write_text("")
        blob = {"schema": SCHEMA_VERSION, "w_entropy": 0.0,
                "w_competence": 0.4, "threshold": 0.84,
                "lambda_uncertainty": 0.0, "brier": 0.0468,
                "brier_raw": 0.0468, "brier_base_rate": 0.0478,
                "brier_cv": 0.046881, "n_samples": 1254, "samples": 1254,
                "n_negative": 18, "map_status": "applied",
                "fitted_at": "2026-08-30T00:00:00Z",
                "feature_contrib": {"effort_component": 0.0007}}
        blob[key] = bad
        (calib / "calibration_params.json").write_text(
            json.dumps(blob, allow_nan=True))

        try:
            out = render_learning_health(mem)
        except Exception as exc:            # noqa: BLE001
            raise AssertionError(
                f"{key}={bad!r} raised {type(exc).__name__} out of the "
                f"report — every other section is lost with it") from exc

        assert "LESSONS:" in out, f"{key}={bad!r} blanked the report"

        # (2) no unusable value rendered as a number — ON ANY LINE.
        #
        # ⚠ THIS CHECK ONLY SCANNED THE VERDICT LINE AT FIRST, and R7.2
        # caught it: a mutant admitting bools into the guard left this test
        # GREEN, because `brier: true` renders on the in-sample DETAIL line
        # while the verdict line is computed from `brier_cv`. A property
        # that inspects one line is a site pin wearing a property's clothes.
        #
        # Warning lines are exempt by construction: they quote the raw value
        # with `!r` precisely to say what could not be read.
        NUMERIC = {"brier", "brier_cv", "brier_raw", "brier_base_rate",
                   "threshold", "w_entropy", "w_competence", "w_effort",
                   "n_samples", "n_negative", "brier_cv_delta_lo",
                   "brier_cv_delta_hi"}
        if key in NUMERIC and not isinstance(bad, (int, float)) or (
                key in NUMERIC and isinstance(bad, bool)):
            # ⚠ SCOPED TO THE CALIBRATION SECTION. Scanning the whole
            # report matched "True" inside an unrelated FORESIGHT line —
            # a check so broad it fails on innocent text is as useless as
            # one so narrow it misses the defect, and gets deleted just as
            # fast.
            import re as _re

            _lines = out.splitlines()
            _start = next((i for i, l in enumerate(_lines)
                           if l.startswith("CALIBRATION:")), None)
            assert _start is not None, (
                f"{key}={bad!r}: the calibration section did not render at "
                f"all, so this assertion would pass vacuously")
            _end = next((i for i in range(_start + 1, len(_lines))
                         if _re.match(r"^[A-Z][A-Z /-]+:", _lines[i])),
                        len(_lines))
            body = [l for l in _lines[_start:_end] if "⚠" not in l]
            for line in body:
                assert str(bad) not in line, (
                    f"{key}={bad!r} was rendered as a number outside a "
                    f"warning:\n{line}")


def test_the_shared_fixture_builds_a_file_the_agent_would_load(tmp_path):
    """⚠ ONE ASSERTION THAT PROTECTS ~50 TESTS. `_render_with` omits nothing
    today, but deleting `"schema"` from its blob made every test through it
    render a file `load_params` REFUSES — carrying `⚠ NOT IN FORCE … NOTHING
    below describes the live scorer` — and the whole file stayed green,
    because fixed and broken worlds then agree everywhere."""
    out = _render_with(tmp_path, beats_base_rate="yes", map_status="applied",
                       brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002)
    assert "NOT IN FORCE" not in out, (
        "the shared fixture builds params the agent would refuse, so every "
        f"assertion through it is pinned on a scorer that is not running:\n"
        f"{out[:400]}")


@pytest.mark.parametrize("sentinel", [-1.0, float("nan"), float("-inf")])
def test_the_not_recorded_sentinel_never_becomes_a_score(sentinel, tmp_path):
    """⚠ `-1.0` IS THIS MODULE'S OWN "NOT RECORDED" VALUE, written by
    `fit()` for `brier_raw`, `brier_base_rate` and `brier_cv`. It was
    recognised as such at three of the four Brier sites and not at `brier`,
    so a params file the agent certifies as IN FORCE rendered "beats the
    base-rate predictor" off a number meaning "we never computed this" —
    while `load_params` read `unknown` for the same file.

    The single-key fuzz cannot see this: it varies one key at a time, so
    `brier_cv` stays usable and the sentinel never reaches `_score`. Both
    Brier fields have to be unusable together, which is exactly the shape a
    fit that never computed either one writes.
    """
    out = _render_with(tmp_path, beats_base_rate="yes", map_status="applied",
                       brier=sentinel, brier_cv=sentinel)
    assert "beats the base-rate predictor" not in out, (
        f"brier=brier_cv={sentinel!r} (not recorded) was scored as a win:\n"
        f"{out[:400]}")
    assert "NO USABLE BRIER" in out, (
        f"an unusable Brier must be named:\n{out[:400]}")
    for tok in ("-1.0", "nan", "-inf"):
        assert f"Brier {tok}" not in out, (
            f"{tok} was rendered as a Brier score:\n{out[:400]}")


# ── the five fixes this round made and did not pin ───────────────────────

def _with_samples(tmp_path, raw=None, **extra):
    """A params file WITH a sample corpus, so the section actually renders.

    ⚠ An earlier non-dict test wrote an empty `calibration.jsonl`, so
    `if params or all_samples:` was False and the CALIBRATION section never
    rendered at all — the missing notice it was written to catch was
    invisible, and fixed and broken worlds agreed by construction.
    """
    import json

    from ghost_agent.core.calibration import SCHEMA_VERSION
    from ghost_agent.core.learning_health import render_learning_health

    mem = tmp_path / "memory"
    mem.mkdir(parents=True, exist_ok=True)
    calib = tmp_path / "calibration"
    calib.mkdir(parents=True, exist_ok=True)
    (calib / "calibration.jsonl").write_text("\n".join(
        json.dumps({"composite": 0.9, "outcome": 1, "epoch": "t"})
        for _ in range(40)) + "\n")
    if raw is not None:
        (calib / "calibration_params.json").write_text(raw)
    else:
        blob = {"schema": SCHEMA_VERSION, "w_entropy": 0.0,
                "w_competence": 0.4, "threshold": 0.84,
                "lambda_uncertainty": 0.0, "brier": 0.0468,
                "brier_raw": 0.0468, "brier_base_rate": 0.0478,
                "brier_cv": 0.046881, "n_samples": 1254, "samples": 1254,
                "n_negative": 18, "map_status": "applied",
                "fitted_at": "2026-08-30T00:00:00Z"}
        blob.update(extra)
        (calib / "calibration_params.json").write_text(json.dumps(blob))
    return render_learning_health(mem)


@pytest.mark.parametrize("raw,why", [
    ("[1, 2, 3]", "a non-dict"),
    ('"hello"', "a JSON string"),
    ("{not json at all", "bad JSON"),
])
def test_the_refusals_the_notice_names_can_actually_fire(raw, why, tmp_path):
    """⚠ THE NOTICE NAMED TWO CASES IN WHICH IT COULD NOT FIRE.

    `_load_json` returns `{}` for bad JSON and a non-dict was normalised to
    `{}` — after which `not params` short-circuited the check to "in force".
    So the sentence "it also refuses bad JSON, a non-dict, and unusable
    required numerics" was true of `load_params` and false of the flag
    reporting it, for exactly the first two."""
    out = _with_samples(tmp_path / why.replace(" ", "_"), raw=raw)
    assert "NOT IN FORCE" in out, (
        f"{why}: `load_params` refuses this file and the report presented it "
        f"as the live scorer:\n{out[:500]}")


def test_a_feature_that_hurts_is_not_called_no_contribution(tmp_path):
    """⚠ TWO WORDS FOR THREE OUTCOMES. A negative ablation delta means
    dropping the feature IMPROVES held-out Brier — it is actively hurting —
    and it rendered as "no measurable contribution", the same phrase as a
    delta of zero. The live params file carries
    `competence_component: -4.3e-05` and read exactly that, beside an
    `effort_component` labelled "helps"."""
    out = _render_with(tmp_path, beats_base_rate="yes", map_status="applied",
                       brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002,
                       feature_contrib={"competence_component": -4.3e-05,
                                        "effort_component": 0.000891,
                                        "entropy_component": 0.0})
    hurt = [l for l in out.splitlines() if "competence_component" in l][0]
    assert "HURTS" in hurt, (
        f"a feature whose removal IMPROVES the score is reported as inert:\n"
        f"{hurt}")
    helps = [l for l in out.splitlines() if "effort_component" in l][0]
    assert "helps" in helps and "HURTS" not in helps, helps
    zero = [l for l in out.splitlines() if "entropy_component" in l][0]
    assert "no measurable contribution" in zero, zero


def test_an_unusable_brier_suppresses_the_verdict_it_announces(tmp_path):
    """⚠ ANNOUNCING THAT A VERDICT CANNOT BE COMPUTED, THEN COMPUTING ONE.
    `_score = nan` fell into the comparison chain, where `nan < x` and
    `nan > x` are both False, landing on `else: "matches"` — a MEASURED
    word, printed directly beneath a notice saying the comparison is
    impossible."""
    # ⚠ THE VERDICT KEY MUST BE ABSENT. Passing `beats_base_rate=None`
    # makes it PRESENT-and-null, which routes to the refusal branch and
    # never reaches the point-estimate chain this test is about — so the
    # fixture agreed in both worlds and two mutants survived it.
    out = _render_with(tmp_path, map_status="applied",
                       brier="0.166", brier_cv=None)
    assert "NO USABLE BRIER" in out, out[:400]
    line = [l for l in out.splitlines()
            if "base-rate predictor" in l or "base-rate comparison" in l][0]
    for word in ("matches", "beats", "LOSES TO"):
        assert word not in line, (
            f"a verdict was printed under a notice saying it cannot be "
            f"computed ({word!r}):\n{line}")


def test_the_tie_arm_carries_the_same_caveat_as_its_siblings(tmp_path):
    """`beats` and `LOSES TO` both say "(point estimate only — no CI
    recorded)"; `matches` asserted a bare equivalence — a stronger claim
    than either, and the one this block's own comment forbids implying
    without a test behind it."""
    out = _render_with(tmp_path, brier=0.05, brier_cv=0.05,
                       brier_base_rate=0.05, map_status="applied")
    line = [l for l in out.splitlines() if "base-rate predictor" in l][0]
    assert "matches" in line, line
    assert "point estimate only" in line, (
        f"the tie arm asserts an equivalence with no test behind it:\n{line}")


@pytest.mark.parametrize("field", ["n_samples", "n_entropy_observed",
                                   "n_effort_observed",
                                   "n_excluded_other_epochs", "n_negative"])
def test_load_params_never_raises_whatever_the_file_holds(field, tmp_path):
    """⚠ THE CONTRACT SAID "NEVER A CRASH" AND FIVE FIELDS COULD CRASH IT.
    `except (KeyError, TypeError, ValueError)` does not cover
    `OverflowError`, which is an `ArithmeticError` — and `int(float('inf'))`
    raises it. A caller had already diagnosed this and worked around it with
    a broad `except` on its own side, leaving `stats()` and the startup path
    exposed. The contract is fixed where it is stated."""
    import json

    from ghost_agent.core.calibration import (SCHEMA_VERSION,
                                              CalibrationTracker)

    for bad in (float("inf"), float("-inf"), float("nan"), 1e400):
        d = tmp_path / f"{field}_{bad}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "calibration_params.json").write_text(json.dumps({
            "schema": SCHEMA_VERSION, "w_entropy": 0.0, "w_competence": 0.4,
            "threshold": 0.84, "lambda_uncertainty": 0.0, "brier": 0.05,
            "n_samples": 100, "fitted_at": "2026-08-30T00:00:00Z",
            field: bad}))
        tracker = CalibrationTracker(d)
        try:
            assert tracker.load_params() is None, (
                f"{field}={bad!r} produced a params object from an "
                "unreadable file")
            tracker.stats()
        except Exception as exc:            # noqa: BLE001
            raise AssertionError(
                f"{field}={bad!r} raised {type(exc).__name__} out of "
                "load_params/stats, whose documented contract is to return "
                "None on ANY problem, never to crash") from exc


def test_an_unreadable_negative_class_count_is_announced(tmp_path):
    """The guard added for these counts deleted the line in silence — the
    line this block itself calls "THE NUMBER EVERY OTHER VERDICT RESTS ON".
    Its sibling fix in the same commit counts and names what it drops."""
    out = _render_with(tmp_path, beats_base_rate="yes", map_status="applied",
                       brier_cv_delta_lo=-0.01, brier_cv_delta_hi=-0.002,
                       n_negative=True)
    assert "NEGATIVE-CLASS COUNT UNREADABLE" in out, (
        f"the line every verdict rests on vanished with no notice:\n"
        f"{out[:500]}")
    assert "negative class:" not in out, "and the unreadable line must not print"


def test_the_verdict_survives_the_log_line_truncation():
    """⚠ A FIELD IN A LOG LINE IS NOT A FIELD THE OPERATOR SEES.

    `pretty_log` truncates each rendered line at a fixed width with an
    ellipsis. `map` and `beats_base_rate` were appended at the END of the
    startup payload, so they were cut off the real `ghost-agent.log` every
    single time — the change added the verdict to a line and the operator
    still could not read it. Verified against the live log after a restart:
    `loaded=startup threshold=0.84 w_entropy=0.00 lam=0.00…`.

    The idle-refit line does not need this: it folds the verdict into the
    FIRST field as `refit=ok/no_prob:<verdict>/no_rank:<verdict>`.
    """
    from ghost_agent.core.calibration import BEATS_INDISTINGUISHABLE, FittedParams
    from ghost_agent.main import calib_startup_fields

    keys = list(calib_startup_fields(FittedParams(
        w_entropy=0.0, w_competence=0.4, threshold=0.84,
        lambda_uncertainty=0.0, brier=0.0468, n_samples=1254,
        fitted_at="2026-08-30T00:00:00Z", map_status="applied",
        beats_base_rate=BEATS_INDISTINGUISHABLE)))

    for field in ("map", "beats_base_rate"):
        assert keys.index(field) <= 2, (
            f"{field!r} is at position {keys.index(field)} of "
            f"{len(keys)} in the payload; the line is truncated, so a field "
            "after the first few is not visible to the operator")
    assert keys.index("beats_base_rate") < keys.index("brier"), (
        "the licence must precede the numbers it qualifies")
