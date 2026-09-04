"""A placeholder is not a label: the verdicts are measured on rows that saw
something.

§4EO, 2026-09-04. `grade_turn_outcome` returns `_UNVERIFIED_PRIOR` (0.83) for
"nothing was checked and nothing visibly broke" — a stand-in for a measurement
that never happened, exactly as the neutral 0.5 is for unobserved entropy. This
module already refuses to let the entropy stand-in vote (`entropy_observed`
gates `w_entropy`); the outcome stand-in had no such gate, and on the live store
it is **63% of the fit population** (672 of 1074).

Both verdicts were computed over it:

  * `beats_base_rate` compares the model against always predicting the base
    rate — and the base rate IS the placeholder (0.838 vs a constant 0.83), so
    two thirds of the rows asked the model to reproduce the comparand. The
    fitted map obliged: it mapped the whole corpus onto [0.700, 0.867].
  * `ranks_outcomes` binarises at 0.5, so every placeholder entered the AUC as a
    POSITIVE — 672 of 1017 (66%) of the "successes" the score was credited with
    ordering were turns nobody checked.

The property under review: **a row that carries no verdict may be fitted on, but
may not judge the fit.**
"""

import json
import logging
import random

import pytest

from ghost_agent.core import calibration as C
from ghost_agent.core.calibration import (BEATS_NO, BEATS_UNKNOWN, BEATS_YES,
                                          CalibrationTracker, label_is_verdict,
                                          verdict_pairs)

PRIOR = C._UNVERIFIED_PRIOR


# ── the predicate, tied to its producer ────────────────────────────────────

def test_every_grade_path_is_classified_from_the_GRADER_not_a_constant_list():
    """⚠ ENUMERATE FROM THE PRODUCER. A hand-written list of grade constants
    is a second copy of the ladder, free to drift the moment
    `grade_turn_outcome` grows a path — and this module has paid for exactly
    that twice (the source-rank twins, the cap twins).

    The world it fails in: a new grade lands on the prior by arithmetic (say a
    penalty that happens to cancel) and is silently reclassified as "nobody
    checked", removing real evidence from every verdict.
    """
    g = C.grade_turn_outcome
    checked = {
        "verifier FAILED": g(verifier_verdict="failed"),
        "verifier PASSED": g(verifier_verdict="passed"),
        "shape failed": g(shape_failed=True),
        "budget exhausted": g(budget_exhausted=True),
        "one exec failure": g(execution_failure_count=1),
        "many exec failures": g(execution_failure_count=9),
        # The shape rule withholds a PASS and drops the turn onto the graded
        # execution-failure ladder — which its own caller only ever reaches
        # with at least one counted failure (see the reachability pin below).
        "unacked total failure": g(verifier_verdict="passed",
                                   unacked_total_failure=True,
                                   execution_failure_count=1),
    }
    for name, grade in checked.items():
        assert label_is_verdict(grade), (
            f"{name} produced {grade}, which reads as 'nobody checked'")
    # The path that saw nothing.
    assert not label_is_verdict(g())
    assert not label_is_verdict(g(verifier_verdict=None,
                                  execution_failure_count=0))
    assert g() == PRIOR


def test_a_WITHHELD_pass_with_no_counted_failure_is_also_not_a_verdict():
    """The second route to the prior, and it is classified the same way ON
    PURPOSE.

    `unacked_total_failure` withholds a verifier PASS: the grader's own words
    are "the shape says 'unverified and something broke', not 'checked and
    wrong'". With zero counted failures that ladder lands exactly on
    `_UNVERIFIED_PRIOR`, and "unverified" is precisely what
    `label_is_verdict` refuses. Both routes to 0.83 mean the same thing — no
    trustworthy verdict was obtained — so both are excluded.

    The world it fails in: someone reads this collision as a bug and special-
    cases the withheld PASS back INTO the verdict population, which would
    admit a row whose label is, by the rule's own design, not a verdict.
    """
    assert C.grade_turn_outcome(verifier_verdict="passed",
                                unacked_total_failure=True,
                                execution_failure_count=0) == PRIOR
    assert not label_is_verdict(PRIOR)


def test_the_live_caller_cannot_produce_that_collision():
    """The reachability the pin above rests on — checked, because a
    justification is a claim.

    `core/agent.py` sets `unacked_total_failure` only when
    `execution_failure_count > 0`, so the live grade for a withheld PASS is
    always at least one `_EXEC_FAILURE_PENALTY` below the prior and lands in
    the verdict population. ⚠ If this fails because that caller changed, the
    collision becomes REACHABLE and real "all tools failed" turns start
    leaving the verdict population — go read the pin above before deciding
    whether that is wanted.
    """
    import inspect
    from ghost_agent.core import agent as A
    src = inspect.getsource(A)
    assert "_calib_unacked = int(execution_failure_count or 0) > 0" in src, (
        "the live caller no longer gates `unacked_total_failure` on a "
        "counted failure — the prior collision is now reachable")
    # And the arithmetic that makes the gate sufficient.
    assert C.grade_turn_outcome(verifier_verdict="passed",
                                unacked_total_failure=True,
                                execution_failure_count=1) < PRIOR


def test_the_predicate_is_exact_not_a_tolerance():
    """The world it fails in: a tolerance is used, and a genuine neighbouring
    grade — the degraded ladder passes within 0.15 of the prior — is thrown
    away as filler."""
    assert not label_is_verdict(PRIOR)
    for near in (PRIOR - 0.001, PRIOR + 0.001, 0.8, 0.86, 0.0, 1.0):
        assert label_is_verdict(near), f"{near} was swallowed as a placeholder"


def test_the_predicate_is_total():
    """Labelling must never break a fit: a malformed row carries no verdict,
    it does not raise."""
    for junk in (None, "", "nope", [], {}, object()):
        assert label_is_verdict(junk) is False


def test_verdict_pairs_filters_and_survives_nothing():
    rows = [(0.9, 1.0), (0.4, PRIOR), (0.2, 0.0), (0.7, PRIOR)]
    assert verdict_pairs(rows) == [(0.9, 1.0), (0.2, 0.0)]
    assert verdict_pairs([]) == []
    assert verdict_pairs(None) == []
    assert verdict_pairs([(0.5, PRIOR)]) == []


# ── the fit, from the consumer's side ──────────────────────────────────────

def _feed(t, n, *, seed, ranking=True, outcome=None, lo=0.05, hi=0.95):
    """`competence_component` carries the score: with no observed entropy or
    effort the weight search pins those to 0 and the composite IS competence,
    so these fixtures control the ranking directly."""
    rng = random.Random(seed)
    for _ in range(n):
        good = rng.random() < 0.8
        c = rng.uniform(0.6, hi) if good else rng.uniform(lo, 0.4)
        if not ranking:
            c = rng.uniform(lo, hi)
        y = outcome if outcome is not None else (1.0 if good else 0.0)
        t.record(composite=round(c, 4), outcome=y,
                 entropy_component=0.5, competence_component=round(c, 4))


def test_the_counts_partition_the_fit_population(tmp_path):
    """`n_samples` is what the WEIGHTS were fitted on; `n_verdict_rows` is what
    judged them. Reading one as the other is reading power that does not
    exist — live, 1074 against 402."""
    t = CalibrationTracker(tmp_path)
    _feed(t, 300, seed=1)
    _feed(t, 500, seed=2, outcome=PRIOR)
    p = t.fit()
    assert p.n_verdict_rows == 300
    assert p.n_unverified_prior == 500
    assert p.n_verdict_rows + p.n_unverified_prior == p.n_samples
    # The placeholders were FITTED on — this narrows what measures the
    # scorer, never what the scorer is.
    assert p.n_samples == 800


def test_the_recorded_baseline_IS_the_verdict_rows_baseline(tmp_path):
    """Recomputed, not merely 'plausible'.

    The world it fails in: `brier_base_rate` keeps describing all 800 rows
    while `brier_cv` describes 300 — the log line then compares two numbers
    from different populations, which is how a 0.041-vs-0.113 'crushing win'
    gets printed.
    """
    t = CalibrationTracker(tmp_path)
    _feed(t, 300, seed=3)
    _feed(t, 500, seed=4, outcome=PRIOR)
    p = t.fit()
    rows = [s for s in t._load_epoch() if label_is_verdict(s.outcome)]
    base = sum(s.outcome for s in rows) / len(rows)
    expect = sum((base - s.outcome) ** 2 for s in rows) / len(rows)
    assert p.brier_base_rate == pytest.approx(round(expect, 6), abs=5e-7)
    # …and it is NOT the all-rows baseline, which the placeholders flatten.
    allrows = t._load_epoch()
    abase = sum(s.outcome for s in allrows) / len(allrows)
    aexpect = sum((abase - s.outcome) ** 2 for s in allrows) / len(allrows)
    assert abs(p.brier_base_rate - aexpect) > 0.02, (
        "fixture no longer separates the two populations")


def test_placeholders_cannot_vote_in_the_RANK_verdict(tmp_path):
    """THE DECISIVE PIN. Placeholders binarise to POSITIVE, so a corpus whose
    placeholders score low drags the all-rows AUC below chance while the rows
    that actually saw something order fine.

    The world it fails in: `_auc_ci(composites)` — the pre-§4EO call — reports
    `ranks_outcomes = no` ("orders turns BACKWARDS. No use of this score is
    supported") about a score that ranks at 0.9 on every row anybody checked.
    """
    t = CalibrationTracker(tmp_path)
    # 300 verdict rows that rank cleanly…
    rng = random.Random(11)
    for _ in range(300):
        good = rng.random() < 0.7
        c = rng.uniform(0.75, 0.95) if good else rng.uniform(0.05, 0.25)
        t.record(composite=round(c, 4), outcome=1.0 if good else 0.0,
                 entropy_component=0.5, competence_component=round(c, 4))
    # …and 500 placeholders parked at the BOTTOM of the score range.
    for _ in range(500):
        c = rng.uniform(0.0, 0.15)
        t.record(composite=round(c, 4), outcome=PRIOR,
                 entropy_component=0.5, competence_component=round(c, 4))
    p = t.fit()

    epoch = t._load_epoch()
    scored = [(C._composite_for(s, p.w_entropy, p.lambda_uncertainty,
                                p.w_effort), s.outcome) for s in epoch]
    auc_all = C._auc(scored)
    auc_verdict = C._auc(verdict_pairs(scored))
    assert auc_all < 0.5 < auc_verdict, (
        f"fixture no longer separates the two AUCs: all={auc_all} "
        f"verdict={auc_verdict}")
    assert p.auc == pytest.approx(round(auc_verdict, 4), abs=5e-5)
    assert p.ranks_outcomes == BEATS_YES
    assert C._rank_verdict(C._auc_ci(scored)) == BEATS_NO, (
        "the all-rows measurement would have called this backwards — that is "
        "the verdict this pin exists to keep off the params")


def test_the_resolution_is_recorded_beside_the_verdict(tmp_path):
    """`indistinguishable` covers two opposite states — a measured tie, and a
    comparison with no power. The half-width says which."""
    t = CalibrationTracker(tmp_path)
    _feed(t, 400, seed=5)
    p = t.fit()
    assert p.brier_cv_delta_lo is not None and p.brier_cv_delta_hi is not None
    expect = (p.brier_cv_delta_hi - p.brier_cv_delta_lo) / 2.0
    assert p.delta_halfwidth == pytest.approx(round(expect, 6), abs=5e-7)
    assert p.delta_halfwidth > 0


def test_the_component_rank_table_is_measured_on_the_verdict_rows(tmp_path):
    """The number that made the live finding visible: `competence_component`
    carries the largest weight (0.5) and ranks at chance.

    The world it fails in: the table is measured on all rows, so the same
    placeholder mass that flattens the Brier comparison flattens this one too
    — and the one instrument with power at this base rate reports nothing.
    """
    t = CalibrationTracker(tmp_path)
    _feed(t, 300, seed=6)
    _feed(t, 400, seed=7, outcome=PRIOR)
    p = t.fit()
    assert p.component_auc is not None
    assert set(p.component_auc) == {"entropy_component",
                                    "competence_component",
                                    "effort_component"}
    epoch = t._load_epoch()
    for attr, got in p.component_auc.items():
        want = C._auc(verdict_pairs([(getattr(s, attr), s.outcome)
                                     for s in epoch]))
        assert got == pytest.approx(round(want, 4), abs=5e-5), attr
    # competence carries the score in this fixture, so it must be the one
    # that ranks — a table that cannot tell the components apart is not a
    # measurement.
    assert p.component_auc["competence_component"] > 0.6
    assert p.component_auc["entropy_component"] == pytest.approx(0.5, abs=1e-9)


def test_a_corpus_of_pure_placeholders_never_reaches_a_verdict(tmp_path):
    """Every label a placeholder: the fit BAILS, because a constant outcome
    column has no information — and that is what makes the empty-verdict
    branch inside `fit` unreachable.

    ⚠ This pin and the next are a pair: this one proves the state cannot
    arrive through `fit`, the next proves the arithmetic still answers if it
    ever does. Neither alone is enough — the first would license deleting a
    ZeroDivisionError guard, the second would license a guard nobody can
    reach.
    """
    t = CalibrationTracker(tmp_path)
    _feed(t, 400, seed=8, outcome=PRIOR)
    assert C._outcome_variance(t._load_epoch()) < 1e-9
    assert t.fit() is None, (
        "a corpus with no verdict at all produced a fit, so the empty-verdict "
        "branch in `fit` is now reachable and needs its own pin")
    assert not (tmp_path / "calibration_params.json").exists(), (
        "a bailing fit must not overwrite the params in force")


def test_the_baseline_helper_answers_on_an_empty_population():
    """The world it fails in: `fit` divides by zero the day the variance bail
    is loosened, or someone deletes the guard because no test reaches it."""
    assert C.base_rate_brier([]) == (None, None)
    assert C.base_rate_brier(None) == (None, None)
    base, brier = C.base_rate_brier([(0.9, 1.0), (0.1, 0.0)])
    assert base == pytest.approx(0.5)
    assert brier == pytest.approx(0.25)
    # It is the base-rate predictor, not the score's: the composites are
    # ignored entirely.
    assert C.base_rate_brier([(0.0, 1.0), (1.0, 0.0)]) == (0.5, 0.25)


def test_the_baseline_and_the_cv_brier_are_written_together(tmp_path):
    """⚠ THE PAIR IS ATOMIC. A corpus too thin to cross-validate must not
    leave a usable baseline behind.

    The world it fails in: `brier_cv = -1.0` sits beside a real
    `brier_base_rate`, and the learning-health renderer's documented fallback
    ("no CV Brier, use the in-sample one") compares a full-population
    in-sample number against a verdict-population baseline.
    """
    t = CalibrationTracker(tmp_path, min_samples_for_fit=12)
    _feed(t, 6, seed=9)                       # 6 verdict rows: below 2*k
    _feed(t, 200, seed=10, outcome=PRIOR)
    p = t.fit()
    assert p is not None
    assert p.n_verdict_rows == 6
    assert p.brier_cv == -1.0, "fixture no longer lands below the CV floor"
    assert p.brier_base_rate == -1.0, (
        "a usable baseline was left beside an unusable CV Brier")


def test_the_new_fields_round_trip(tmp_path):
    """⚠ THE READER SET. Three fields in this loader's history were written by
    `_save_params` and never reconstructed, so every reloaded fit carried the
    default forever."""
    t = CalibrationTracker(tmp_path)
    _feed(t, 300, seed=12)
    _feed(t, 200, seed=13, outcome=PRIOR)
    p = t.fit()
    r = CalibrationTracker(tmp_path).load_params()
    for f in ("n_verdict_rows", "n_unverified_prior", "delta_halfwidth",
              "component_auc", "brier_base_rate", "auc", "ranks_outcomes"):
        assert getattr(r, f) == getattr(p, f), f


def test_a_params_file_predating_the_fields_loads_as_not_recorded(tmp_path):
    t = CalibrationTracker(tmp_path)
    _feed(t, 300, seed=14)
    t.fit()
    path = tmp_path / "calibration_params.json"
    d = json.loads(path.read_text())
    for k in ("n_verdict_rows", "n_unverified_prior", "delta_halfwidth",
              "component_auc"):
        d.pop(k, None)
    path.write_text(json.dumps(d))
    r = CalibrationTracker(tmp_path).load_params()
    assert r is not None, "an older params file must still load"
    assert r.n_verdict_rows == 0 and r.n_unverified_prior == 0
    assert r.delta_halfwidth is None and r.component_auc is None


# ── the operator surface ───────────────────────────────────────────────────

def test_the_warning_names_the_population_it_measured(tmp_path, caplog):
    """The world it fails in: the line quotes a base rate and a Brier without
    saying they describe 402 of 1074 rows, and `n=1074` on the neighbouring
    surface reads as the evidence behind the verdict."""
    t = CalibrationTracker(tmp_path)
    rng = random.Random(21)
    for _ in range(900):
        good = rng.random() < 0.86
        c = min(0.99, max(0.01, rng.gauss(0.80 + (0.08 if good else -0.08), 0.26)))
        t.record(composite=round(c, 4), outcome=1.0 if good else 0.0,
                 entropy_component=0.5, competence_component=round(c, 4))
    for _ in range(600):
        c = min(0.99, max(0.01, rng.gauss(0.80, 0.26)))
        t.record(composite=round(c, 4), outcome=PRIOR,
                 entropy_component=0.5, competence_component=round(c, 4))
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        p = t.fit()
    assert p.beats_base_rate in C.NOT_INFORMATIVE, (
        f"fixture no longer reaches the branch: {p.beats_base_rate}")
    text = caplog.text
    assert f"{p.n_verdict_rows} row(s) carrying a VERDICT" in text
    assert f"{p.n_unverified_prior} placeholder row(s)" in text
    assert "resolves differences no smaller than" in text
    assert ("%+.5f" % p.delta_halfwidth) in text


def test_the_report_states_the_verdict_population_and_the_component_ranks(
        tmp_path, monkeypatch):
    """`introspect learning` is the operator's other read of this fit.

    The world it fails in: the report prints `n=1074` beside an
    `indistinguishable` verdict measured on 402 rows, so the number a reader
    takes as the evidence is two and a half times the evidence that exists —
    and the one instrument with power at this base rate (the component ranks)
    is not on the screen at all.
    """
    from ghost_agent.core import learning_health as LH
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    cal_dir = tmp_path / "system" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    t = CalibrationTracker(cal_dir)
    rng = random.Random(21)
    for _ in range(900):
        good = rng.random() < 0.86
        c = min(0.99, max(0.01, rng.gauss(0.80 + (0.08 if good else -0.08), 0.26)))
        t.record(composite=round(c, 4), outcome=1.0 if good else 0.0,
                 entropy_component=0.5, competence_component=round(c, 4))
    for _ in range(600):
        c = min(0.99, max(0.01, rng.gauss(0.80, 0.26)))
        t.record(composite=round(c, 4), outcome=PRIOR,
                 entropy_component=0.5, competence_component=round(c, 4))
    p = t.fit()
    assert p is not None and p.n_unverified_prior == 600

    report = LH.render_learning_health(tmp_path / "system" / "memory")
    assert f"verdict population: {p.n_verdict_rows} row(s)" in report, report
    assert f"{p.n_unverified_prior} placeholder row(s)" in report
    assert "resolution: this corpus cannot resolve" in report
    assert "component RANK" in report
    for _k, _v in p.component_auc.items():
        assert f"{_v:.4f}" in report, f"{_k} missing from the rank table"


def test_the_report_omits_the_verdict_when_the_baseline_is_unusable(
        tmp_path, monkeypatch):
    """The atomic pair, from the renderer's side.

    The world it fails in: a thin verdict population leaves `brier_cv = -1.0`
    beside a usable baseline, the renderer takes its documented in-sample
    fallback, and a full-population number is declared a winner against a
    verdict-population baseline.
    """
    from ghost_agent.core import learning_health as LH
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    cal_dir = tmp_path / "system" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    t = CalibrationTracker(cal_dir, min_samples_for_fit=12)
    _feed(t, 6, seed=9)
    _feed(t, 200, seed=10, outcome=PRIOR)
    p = t.fit()
    assert p is not None and p.brier_base_rate == -1.0

    report = LH.render_learning_health(tmp_path / "system" / "memory")
    assert "NO USABLE BASELINE" in report, report
    assert "the base-rate predictor" not in report, (
        "a verdict was rendered against a baseline that does not exist")


# ── the bench exclusion explains itself ────────────────────────────────────

def _s(comp, outcome, *, origin, entropy=0.5, effort=0.5):
    return C.CalibrationSample(
        composite=comp, entropy_component=entropy, competence_component=comp,
        uncertainty_pressure=0.0, outcome=outcome, domain="",
        ts="2026-08-01T00:00:00Z", entropy_observed=False,
        effort_component=effort, effort_observed=False, source="turn",
        req_id="", epoch=C.CURRENT_EPOCH, origin=origin)


def test_the_bench_exclusion_names_the_component_that_drove_it():
    """An exclusion that cannot explain itself gets re-diagnosed by hand.

    The gate has thrown rows away since 2026-08-31 saying only "they rank
    backwards", so answering WHY took a bespoke script — and the answer
    mattered: `competence_component` is a per-DOMAIN prior carrying the
    largest weight, and on a degenerate domain mix it cannot track per-turn
    outcomes (live: 0.189 against effort at 0.755).

    The world it fails in: the message is a bare verdict, and the next
    operator to ask "why were 309 rows dropped?" writes the script again.
    """
    rng = random.Random(3)
    real, bench = [], []
    for _ in range(200):
        good = rng.random() < 0.8
        c = rng.uniform(0.6, 0.95) if good else rng.uniform(0.05, 0.4)
        real.append(_s(round(c, 4), 1.0 if good else 0.0, origin="user"))
    for _ in range(150):
        good = rng.random() < 0.8
        c = rng.uniform(0.6, 0.95) if good else rng.uniform(0.05, 0.4)
        # competence (== composite here) INVERTED, effort left aligned.
        bench.append(_s(round(1.0 - c, 4), 1.0 if good else 0.0,
                        origin="bench", effort=round(c, 4)))
    params = C.FittedParams(w_entropy=0.0, w_competence=1.0, threshold=0.5,
                            lambda_uncertainty=0.0, brier=0.1, n_samples=1,
                            fitted_at="", w_effort=0.0)
    kept, verdict, ci, diag = C._apply_bench_direction_gate(real + bench, params)
    assert verdict == BEATS_NO
    assert diag is not None
    assert set(diag) == {"entropy_component", "competence_component",
                         "effort_component"}
    worst = min(diag, key=lambda k: diag[k])
    assert worst == "competence_component", (
        f"the diagnosis blamed {worst}: {diag}")
    assert diag["effort_component"] > 0.5 > diag["competence_component"], (
        "the fixture no longer separates an inverted component from an "
        "aligned one, so this pin cannot tell a diagnosis from a constant")


def test_no_exclusion_means_no_diagnosis():
    """A diagnosis is evidence about rows that were dropped. Manufacturing one
    when nothing was dropped is the fabricated-neutral defect, one field
    over."""
    rng = random.Random(4)
    rows = [_s(round(rng.uniform(0.05, 0.95), 4),
               1.0 if rng.random() < 0.8 else 0.0, origin="user")
            for _ in range(200)]
    params = C.FittedParams(w_entropy=0.0, w_competence=1.0, threshold=0.5,
                            lambda_uncertainty=0.0, brier=0.1, n_samples=1,
                            fitted_at="", w_effort=0.0)
    kept, verdict, ci, diag = C._apply_bench_direction_gate(rows, params)
    assert verdict != BEATS_NO and diag is None
    # Cold start: no instrument, no exclusion, no diagnosis.
    assert C._apply_bench_direction_gate(rows, None)[3] is None


def test_the_exclusion_message_carries_the_diagnosis(tmp_path, caplog):
    t = CalibrationTracker(tmp_path)
    rng = random.Random(5)
    for _ in range(300):
        good = rng.random() < 0.8
        c = rng.uniform(0.6, 0.95) if good else rng.uniform(0.05, 0.4)
        t.record(composite=round(c, 4), outcome=1.0 if good else 0.0,
                 entropy_component=0.5, competence_component=round(c, 4))
    for _ in range(150):
        good = rng.random() < 0.8
        c = rng.uniform(0.6, 0.95) if good else rng.uniform(0.05, 0.4)
        t.record(composite=round(1.0 - c, 4), outcome=1.0 if good else 0.0,
                 entropy_component=0.5, competence_component=round(1.0 - c, 4),
                 origin="bench")
    t.fit()                                    # cold start: nothing excluded
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        p = t.fit()
    assert p.bench_verdict == BEATS_NO and p.n_bench_misaligned == 150
    line = [r.getMessage() for r in caplog.records
            if "EXCLUDED from the fit" in r.getMessage()]
    assert line, caplog.text
    assert "Driven by" in line[0]
    assert "competence_component" in line[0]
    assert "an unmeasured component mix" not in line[0]


def test_every_surface_that_prints_n_also_prints_the_verdict_count(tmp_path):
    """⚠ CROSS-SURFACE CONSISTENCY, PINNED AS A TABLE (§R R5).

    Four surfaces report these verdicts. Three of them print a row count
    beside the verdict, and until §4EO that count was `n_samples` — what the
    WEIGHTS were fitted on, 2.7x the evidence the verdict actually rests on.

    The world it fails in: one surface is migrated and the others are not, so
    the operator reads `n=1074` on the startup line and
    `verdict population: 402` in the report and has no way to know they are
    answering different questions.
    """
    from ghost_agent import main as M
    t = CalibrationTracker(tmp_path)
    _feed(t, 300, seed=31)
    _feed(t, 400, seed=32, outcome=PRIOR)
    p = t.fit()
    assert p.n_samples == 700 and p.n_verdict_rows == 300

    fields = M.calib_startup_fields(p)
    assert fields["n"] == p.n_samples
    assert fields["n_verdict"] == p.n_verdict_rows, (
        "the startup line prints the fitted count with no verdict count "
        "beside it")
    # A params file older than the field reads as "not recorded", never as
    # "the verdict rests on no rows at all".
    stale = C.FittedParams(w_entropy=0.1, w_competence=0.9, threshold=0.8,
                           lambda_uncertainty=0.0, brier=0.04, n_samples=900,
                           fitted_at="")
    assert M.calib_startup_fields(stale)["n_verdict"] == 0


def test_a_params_file_predating_the_fields_still_renders(tmp_path, monkeypatch):
    """The live state at deploy time: the params on disk were fitted by the
    PREVIOUS build and carry none of the new fields.

    The world it fails in: the renderer formats `None` into
    `verdict population: None row(s)`, or raises and takes the whole
    `introspect learning` report down with it — the eight-line amputation
    this file already carries three warnings about.
    """
    from ghost_agent.core import learning_health as LH
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    cal_dir = tmp_path / "system" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    t = CalibrationTracker(cal_dir)
    _feed(t, 300, seed=41)
    assert t.fit() is not None
    path = cal_dir / "calibration_params.json"
    d = json.loads(path.read_text())
    for k in ("n_verdict_rows", "n_unverified_prior", "delta_halfwidth",
              "component_auc"):
        d.pop(k, None)
    path.write_text(json.dumps(d))

    report = LH.render_learning_health(tmp_path / "system" / "memory")
    # The three lines the new fields drive — absent, not rendered as `None`.
    # ⚠ A blanket `"None" not in report` FAILED here on an unrelated
    # pre-existing line ("hit-rate: None ⚠ CONTAMINATED"). Guarding a proxy
    # for the property instead of the property is how a pin starts reporting
    # its neighbours' business.
    assert "verdict population" not in report
    assert "component RANK" not in report
    assert "resolution: this corpus" not in report
    # The rest of the section is intact — a missing advisory field must not
    # amputate the report.
    assert "negative class:" in report, report
