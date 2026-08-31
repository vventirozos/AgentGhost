"""Discrimination beside probability, and the auxiliary-population gate.

§4EB, 2026-08-31. Two defects, one root: the module reported ONE statistic and
let it stand for "is there signal".

  (a) `_apply_bench_mass_cap` bounds how MANY bench rows join the fit, on the
      premise that they are exchangeable with real turns. Measured on the live
      store they are not — 252 bench rows ranked outcomes BACKWARDS (AUC 0.273
      [0.116, 0.460]) against 0.662 [0.580, 0.738] for the 1022 real ones, and
      fit alone produced a Platt slope of -12.8. They were under the cap so
      every one was admitted, dragging pooled AUC to 0.577. The cap tests the
      volume of the assumption; nothing tested the assumption.

  (b) The hourly warning ended on "nothing may treat this score as
      informative". At a base rate of 0.855 with 5.6% negatives, a Brier
      comparison cannot support that: the SAME rows were indistinguishable as
      a probability AND ordered turns at AUC 0.652. "No signal" and "no
      probability calibration" license different things.
"""

import json
import math
import random
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import calibration as C
from ghost_agent.core.calibration import (BEATS_INDISTINGUISHABLE, BEATS_NO,
                                          BEATS_UNKNOWN, BEATS_YES,
                                          CURRENT_EPOCH, CalibrationSample,
                                          CalibrationTracker, FittedParams)


def _s(comp, outcome, *, origin="user", ts="2026-08-01T00:00:00Z"):
    return CalibrationSample(
        composite=comp, entropy_component=0.5, competence_component=comp,
        uncertainty_pressure=0.0, outcome=outcome, domain="", ts=ts,
        entropy_observed=False, effort_component=0.5, effort_observed=False,
        source="turn", req_id="", epoch=CURRENT_EPOCH, origin=origin)


# ── the rank statistic ──────────────────────────────────────────────────────

def test_auc_is_a_rank_statistic_with_known_values():
    perfect = [(0.1, 0.0), (0.2, 0.0), (0.8, 1.0), (0.9, 1.0)]
    assert C._auc(perfect) == 1.0
    assert C._auc([(c, 1.0 - y) for c, y in perfect]) == 0.0
    # All-tied scores carry no ordering at all — exactly 0.5, not a nan.
    assert C._auc([(0.5, 1.0), (0.5, 0.0), (0.5, 1.0), (0.5, 0.0)]) == 0.5


def test_auc_is_none_when_a_class_is_missing():
    """Undefined, and it must SAY undefined.

    Returning 0.5 would manufacture a measured tie out of no measurement —
    the fabricated-neutral defect this module already carries a rule against.
    """
    assert C._auc([(0.3, 1.0), (0.7, 1.0)]) is None
    assert C._auc([(0.3, 0.0), (0.7, 0.0)]) is None
    assert C._auc([]) is None


def test_auc_binarises_graded_outcomes_at_the_same_cut_as_n_negative():
    graded = [(0.2, 0.25), (0.3, 0.49), (0.8, 0.51), (0.9, 1.0)]
    assert C._auc(graded) == 1.0


def test_auc_is_invariant_to_the_platt_map_when_it_is_applied():
    """The property that makes raw-vs-calibrated a non-question when a > 0.

    Platt with a positive slope is strictly increasing, so it cannot change
    an ordering. Fails in the world where `_auc` stops being a pure rank
    statistic — e.g. bucketing scores, or comparing values rather than ranks
    — which would silently make the reported AUC depend on the map.
    """
    pairs = [(0.1, 0.0), (0.4, 0.0), (0.6, 1.0), (0.95, 1.0)]
    raw = C._auc(pairs)
    for a, b in ((1.0, 0.0), (2.5, -0.3), (0.4, 1.2)):
        mapped = [(C.apply_platt(c, a, b), y) for c, y in pairs]
        assert C._auc(mapped) == raw
    inverted = [(C.apply_platt(c, -2.0, 0.0), y) for c, y in pairs]
    assert C._auc(inverted) == pytest.approx(1.0 - raw)


def test_auc_ci_is_deterministic_and_brackets_its_point():
    rng = random.Random(3)
    pairs = [(rng.random(), 1.0 if rng.random() < 0.8 else 0.0)
             for _ in range(200)]
    a = C._auc_ci(pairs)
    assert a is not None
    assert a == C._auc_ci(pairs), "an audit number that moves on its own"
    point, lo, hi = a
    assert lo <= point <= hi
    assert point == C._auc(pairs)


def test_auc_ci_refuses_to_speak_below_the_floor():
    rng = random.Random(4)
    pairs = [(rng.random(), 1.0 if rng.random() < 0.7 else 0.0)
             for _ in range(C._RANK_MIN_N - 1)]
    assert C._auc_ci(pairs) is None


# ── the verdict is INTERVAL-driven, like beats_base_rate ────────────────────

def test_rank_verdict_reads_the_interval_not_the_point():
    """The §4DZ discipline, applied to the second statistic.

    A point estimate on the right side of chance is a margin, not a result.
    The world this fails in: return YES on `point > 0.5` — the first row below
    is then licensed while its interval spans chance.
    """
    assert C._rank_verdict((0.62, 0.48, 0.74)) == BEATS_INDISTINGUISHABLE
    assert C._rank_verdict((0.66, 0.58, 0.74)) == BEATS_YES
    assert C._rank_verdict((0.27, 0.12, 0.46)) == BEATS_NO
    assert C._rank_verdict(None) == BEATS_UNKNOWN
    # Exactly-at-chance bounds are not a licence in either direction.
    assert C._rank_verdict((0.6, 0.5, 0.8)) == BEATS_INDISTINGUISHABLE
    assert C._rank_verdict((0.4, 0.2, 0.5)) == BEATS_INDISTINGUISHABLE


def test_rank_verdict_uses_the_frozen_vocabulary():
    for ci in ((0.9, 0.8, 0.95), (0.1, 0.05, 0.2), (0.5, 0.4, 0.6), None):
        assert C._rank_verdict(ci) in C._VERDICTS
    assert BEATS_NO in C.NOT_INFORMATIVE
    assert BEATS_INDISTINGUISHABLE in C.NOT_INFORMATIVE


# ── (a) the auxiliary-population direction gate ─────────────────────────────

def _params(**over):
    d = dict(w_entropy=0.0, w_competence=1.0, threshold=0.8,
             lambda_uncertainty=0.0, brier=0.0, n_samples=0, fitted_at="",
             w_effort=0.0)
    d.update(over)
    return FittedParams(**d)


def _aligned(n, *, origin, seed, invert=False):
    """Rows whose composite ranks the outcome (or, inverted, misranks it)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        good = rng.random() < 0.8
        c = rng.uniform(0.6, 0.95) if good else rng.uniform(0.05, 0.4)
        if invert:
            c = 1.0 - c
        out.append(_s(round(c, 4), 1.0 if good else 0.0, origin=origin))
    return out


def test_backwards_ranking_bench_rows_are_excluded():
    real = _aligned(200, origin="user", seed=1)
    bench = _aligned(120, origin="bench", seed=2, invert=True)
    kept, verdict, ci = C._apply_bench_direction_gate(real + bench, _params())
    assert verdict == BEATS_NO
    assert ci is not None and ci[2] < 0.5
    assert all(s.origin != "bench" for s in kept)
    assert len(kept) == len(real)


def test_aligned_bench_rows_are_kept():
    """NEGATIVE CONTROL. Without it the gate could pass by dropping bench
    unconditionally, which is a different (and worse) change."""
    real = _aligned(200, origin="user", seed=1)
    bench = _aligned(120, origin="bench", seed=7)
    kept, verdict, _ci = C._apply_bench_direction_gate(real + bench, _params())
    assert verdict == BEATS_YES
    assert len(kept) == len(real) + len(bench)


def _noise_bench(n, seed):
    rng = random.Random(seed)
    return [_s(round(rng.random(), 4), 1.0 if rng.random() < 0.8 else 0.0,
               origin="bench") for _ in range(n)]


def test_the_gate_excludes_only_on_a_CONFIDENT_inversion():
    """Conservative by design: noise must not cost real data.

    Bench rows with no ordering at all land `indistinguishable`, and the
    operator decision that admitted bench at all (§4BF 1c) stands until there
    is evidence against its premise. Asserted over SEVERAL draws, not one: at
    a nominal 95% interval a single null draw excludes ~2.5% of the time by
    construction (seed 11 is such a draw — the first one tried), and a test
    that pinned one seed would be pinning the tail, not the rule.
    """
    real = _aligned(200, origin="user", seed=1)
    verdicts = []
    for seed in range(20, 32):
        noise = _noise_bench(120, seed)
        kept, verdict, _ = C._apply_bench_direction_gate(real + noise,
                                                         _params())
        verdicts.append(verdict)
        if verdict != BEATS_NO:
            assert len(kept) == len(real) + len(noise)
    assert verdicts.count(BEATS_INDISTINGUISHABLE) >= 10, (
        f"noise is being read as a direction: {verdicts}")


def test_the_gates_false_positive_rate_stays_near_nominal():
    """MEASURE THE MECHANISM. The gate DROPS REAL DATA when it fires, so its
    error rate under a true null is a property, not a detail.

    Composite independent of outcome; a 95% interval should exclude chance
    downward ~2.5% of the time. Measured 2026-08-31 across four shapes:
    2.25%-3.25%. The bound below is loose enough to survive resampling noise
    and tight enough to fail the world this guards — a CI computed too narrow
    (e.g. a normal approximation on 5% negatives, or dropping the tie
    correction), which fires far more often and quietly starves the fit.
    """
    trials, fired = 200, 0
    for t in range(trials):
        rng = random.Random(5000 + t)
        pairs = [(rng.random(), 0.0 if rng.random() < 0.2 else 1.0)
                 for _ in range(120)]
        if C._rank_verdict(C._auc_ci(pairs)) == BEATS_NO:
            fired += 1
    assert fired / trials <= 0.10, (
        f"the direction gate fires on {fired}/{trials} pure-noise "
        "populations — it would drop real data")


def test_the_gate_has_no_opinion_without_evidence():
    real = _aligned(200, origin="user", seed=1)
    bench = _aligned(120, origin="bench", seed=2, invert=True)
    # No previous fit → no instrument to test the population against.
    kept, verdict, ci = C._apply_bench_direction_gate(real + bench, None)
    assert (verdict, ci) == (BEATS_UNKNOWN, None) and len(kept) == 320
    # Too few auxiliary rows → not evidence either, however they rank.
    few = _aligned(C._RANK_MIN_N - 1, origin="bench", seed=2, invert=True)
    kept2, verdict2, _ = C._apply_bench_direction_gate(real + few, _params())
    assert verdict2 == BEATS_UNKNOWN and len(kept2) == len(real) + len(few)


def test_the_gate_never_breaks_the_fit():
    real = _aligned(200, origin="user", seed=1)
    bench = _aligned(120, origin="bench", seed=2, invert=True)
    broken = MagicMock()
    broken.w_entropy = "not a number"
    kept, verdict, _ = C._apply_bench_direction_gate(real + bench, broken)
    assert verdict == BEATS_UNKNOWN and len(kept) == 320


# ── one builder, every surface ──────────────────────────────────────────────

def test_fit_population_is_the_only_builder():
    """R1 enumeration: the chain must exist in exactly one place.

    `fit`, `_load_epoch` and `stats` each assembled it by hand, and this
    module already carries two bugs from that (§4L R2 NEW-4, §4BF 1c) — a
    metric describing a population the fit never consumed. The world this
    fails in: a fourth filter is added to one caller and not the others.
    """
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(C))
    callers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("_apply_bench_mass_cap",
                                     "_apply_bench_direction_gate",
                                     "_migrate_leaked_pressure"):
            fn = next((a for a in ast.walk(tree)
                       if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and a.lineno <= node.lineno <= (a.end_lineno or 0)
                       and a.name != "_fit_population"), None)
            if fn is not None:
                callers.add(fn.name)
    assert not callers, (
        "the fit population is assembled outside `_fit_population` in "
        f"{sorted(callers)} — every surface must ask the one builder")


def test_the_three_surfaces_describe_the_same_population(tmp_path):
    t = CalibrationTracker(tmp_path)
    for s in _aligned(200, origin="user", seed=1):
        t.record(composite=s.composite, outcome=s.outcome,
                 entropy_component=0.5, competence_component=s.composite,
                 origin="user")
    for s in _aligned(120, origin="bench", seed=2, invert=True):
        t.record(composite=s.composite, outcome=s.outcome,
                 entropy_component=0.5, competence_component=s.composite,
                 origin="bench")
    # COLD START: the gate scores the auxiliary rows with the PREVIOUSLY
    # fitted weights, and on a first-ever fit there are none — so there is no
    # evidence and everything is admitted. Pinned because it is a real
    # property of the design, not an accident: the exclusion costs one refit.
    first = t.fit()
    assert first is not None
    assert first.bench_verdict == BEATS_UNKNOWN
    assert first.n_bench_misaligned == 0
    assert first.n_samples == 320

    # SECOND refit, now with an instrument to test the population against —
    # and note the instrument it uses was itself fit on the polluted pool,
    # which must still be enough to see the inversion.
    params = t.fit()
    assert params.n_bench_misaligned == 120
    assert params.bench_verdict == BEATS_NO
    assert params.n_samples == 200
    # The metric surfaces must describe exactly what the fit consumed, not a
    # superset — the §4L "population the agent never fits" rule.
    assert len(t._load_epoch()) == params.n_samples
    assert t.stats()["samples"] == params.n_samples


# ── (b) the report says WHICH kind of signal is missing ────────────────────

def test_fit_records_the_rank_measurement(tmp_path):
    t = CalibrationTracker(tmp_path)
    for s in _aligned(300, origin="user", seed=9):
        t.record(composite=s.composite, outcome=s.outcome,
                 entropy_component=0.5, competence_component=s.composite)
    p = t.fit()
    assert p is not None
    assert p.ranks_outcomes == BEATS_YES
    assert p.auc > 0.5 and p.auc_ci_lo is not None and p.auc_ci_hi is not None
    assert p.auc_ci_lo > 0.5, "a YES verdict whose interval spans chance"


def test_the_warning_names_the_kind_of_signal(tmp_path, caplog):
    """The line the operator reads must not say "no signal" on the strength
    of a Brier comparison alone.

    The world this fails in: the warning ends on "nothing may treat this
    score as informative" while the same rows rank at AUC 0.65.
    """
    import logging
    t = CalibrationTracker(tmp_path)
    rng = random.Random(21)
    # THE LIVE REGIME, reproduced: composites clustered high (mean ~0.80),
    # a skewed base rate (0.86), and a separation big enough that the score
    # ORDERS turns but far too small to beat a constant as a probability.
    # Grid-searched to sit in that window with margin — AUC ~0.62 with the
    # interval's lower bound ~0.57, delta CI straddling zero.
    for _ in range(900):
        good = rng.random() < 0.86
        c = rng.gauss(0.80 + (0.08 if good else -0.08), 0.26)
        c = min(0.99, max(0.01, c))
        t.record(composite=round(c, 4), outcome=1.0 if good else 0.0,
                 entropy_component=0.5, competence_component=round(c, 4))
    with caplog.at_level(logging.WARNING, logger=C.logger.name):
        p = t.fit()
    assert p is not None
    # ⚠ ASSERT THE PRECONDITION, DO NOT BRANCH ON IT. The first version of
    # this test wrapped its assertions in `if p.beats_base_rate in
    # NOT_INFORMATIVE and ...`, so a fixture that missed the branch asserted
    # NOTHING and a mutation stripping the rank clause survived.
    assert p.beats_base_rate in C.NOT_INFORMATIVE, (
        f"fixture no longer reaches the branch under test: {p.beats_base_rate}")
    assert p.ranks_outcomes == BEATS_YES, (
        f"fixture no longer ranks: {p.ranks_outcomes} auc={p.auc}")
    # `record.message` is ALREADY formatted by caplog — re-applying `% args`
    # to it chokes on the literal `95% CI` in the text ("unsupported format
    # character 'C'"). `caplog.text` is the rendered log.
    text = caplog.text
    assert "AS A PROBABILITY" in text
    assert "DOES still rank" in text
    assert ("AUC %.3f" % p.auc) in text, (
        f"the warning must quote the measurement: {text!r}")
    assert ("%.3f" % (p.auc_ci_lo or 0.0)) in text, (
        "the warning must quote the INTERVAL, not just the point estimate")


# ── the reader set ─────────────────────────────────────────────────────────

def test_new_fields_round_trip_through_the_params_file(tmp_path):
    """⚠ THE READER SET AGAIN. This module already records a field that
    `_save_params` wrote and `load_params` did not reconstruct, so every
    RELOADED fit carried the default forever. Fails in that world."""
    t = CalibrationTracker(tmp_path)
    written = _params(
        w_entropy=0.1, w_competence=0.9, threshold=0.7,
        auc=0.6612, auc_ci_lo=0.5801, auc_ci_hi=0.7382,
        ranks_outcomes=BEATS_YES, bench_verdict=BEATS_NO,
        n_bench_misaligned=252)
    t._save_params(written)
    back = t.load_params()
    assert back is not None
    for f in ("auc", "auc_ci_lo", "auc_ci_hi", "ranks_outcomes",
              "bench_verdict", "n_bench_misaligned"):
        assert getattr(back, f) == getattr(written, f), f


def test_a_params_file_without_them_loads_as_not_recorded(tmp_path):
    t = CalibrationTracker(tmp_path)
    t._save_params(_params())
    d = json.loads((tmp_path / "calibration_params.json").read_text())
    for f in ("auc", "auc_ci_lo", "auc_ci_hi", "ranks_outcomes",
              "bench_verdict", "n_bench_misaligned"):
        d.pop(f, None)
    (tmp_path / "calibration_params.json").write_text(json.dumps(d))
    back = t.load_params()
    assert back is not None
    assert back.auc == -1.0 and back.auc_ci_lo is None
    assert back.ranks_outcomes == BEATS_UNKNOWN
    assert back.bench_verdict == BEATS_UNKNOWN
    assert back.n_bench_misaligned == 0


def test_an_unrecognised_rank_verdict_is_not_a_licence(tmp_path):
    """`in NOT_INFORMATIVE` is FALSE for a typo, so an uncoerced value would
    read as licensed — the exact fail-open `beats_base_rate` was hardened
    against. Fails in the world where the loader keeps the raw string."""
    t = CalibrationTracker(tmp_path)
    t._save_params(_params())
    path = tmp_path / "calibration_params.json"
    d = json.loads(path.read_text())
    d["ranks_outcomes"] = "probably"
    d["bench_verdict"] = ["not", "a", "word"]     # unhashable, must not raise
    path.write_text(json.dumps(d))
    back = t.load_params()
    assert back is not None
    assert back.ranks_outcomes == BEATS_UNKNOWN
    assert back.bench_verdict == BEATS_UNKNOWN


def test_a_rejected_inverted_fit_reports_the_RAW_direction(tmp_path):
    """An anti-correlated corpus must be REPORTED as anti-correlated.

    ⚠ This does NOT distinguish raw from calibrated: `fit` rebinds
    `calibrated = composites` on every rejection branch, so a mutation
    swapping them survives — proven equivalent, not a gap. What it does pin
    is the SIGN reaching the field: a score that orders turns backwards must
    land `auc < 0.5` and `ranks_outcomes = no`, the single most dangerous
    direction for this verdict to be wrong in. Fails wherever a sign is
    dropped between the statistic and the stored params.
    """
    t = CalibrationTracker(tmp_path)
    rng = random.Random(31)
    for _ in range(400):
        good = rng.random() < 0.7
        # ANTI-correlated on purpose: good turns score LOW.
        c = rng.uniform(0.05, 0.4) if good else rng.uniform(0.6, 0.95)
        t.record(composite=round(c, 4), outcome=1.0 if good else 0.0,
                 entropy_component=0.5, competence_component=round(c, 4))
    p = t.fit()
    assert p is not None
    assert p.map_status == "rejected_inverted", (
        f"fixture no longer reaches the branch under test: {p.map_status}")
    assert p.auc < 0.5, (
        f"an anti-correlated score reported AUC {p.auc} — the calibrated "
        "column was measured instead of the shipped raw one")
    assert p.ranks_outcomes == BEATS_NO


# ── R5: cross-surface consistency, pinned as a table ────────────────────────

def test_every_surface_that_reports_the_probability_verdict_reports_the_rank_one():
    """⚠ MIGRATE THE WHOLE READER SET.

    `calibration.py` already carries the note: adding a field to the producer
    and the renderer but not the COLLECTOR left `cal.get(...)` permanently
    None and the fix inert. The first cut of §4EB committed that against the
    note — `fit()` and the agent's CALIB line got `ranks_outcomes`, the
    startup line and the learning-health collector did not, so one params
    file told two operator surfaces different stories.

    Enumerated from the AST, not from a list, so a NEW surface that reports
    one verdict without the other fails here rather than shipping silently.
    """
    import ast
    import inspect
    from ghost_agent import main as M
    from ghost_agent.core import agent as A
    from ghost_agent.core import learning_health as LH

    offenders = []
    for mod in (M, A, LH):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.dump(node)
            # A surface REPORTS the verdict when it puts the name in a dict
            # key or a keyword argument, not merely when it reads one.
            reports_prob = ("'beats_base_rate'" in body
                            or "arg='beats_base_rate'" in body)
            reports_rank = ("'ranks_outcomes'" in body
                            or "arg='ranks_outcomes'" in body)
            if reports_prob and not reports_rank:
                offenders.append(f"{mod.__name__}.{node.name}")
    assert not offenders, (
        "these surfaces report the probability verdict without the rank "
        f"verdict: {offenders}")


def test_the_startup_line_carries_both_verdicts_before_the_numbers():
    """The startup line is TRUNCATED at a fixed width — this module already
    lost `beats_base_rate` off the end of the live log once. Both verdicts
    must precede the numeric detail, which is what may be dropped."""
    from ghost_agent.main import calib_startup_fields
    fields = calib_startup_fields(_params(auc=0.66, ranks_outcomes=BEATS_YES,
                                          beats_base_rate=BEATS_NO))
    assert fields["beats_base_rate"] == BEATS_NO
    assert fields["ranks_outcomes"] == BEATS_YES
    keys = list(fields)
    for verdict in ("beats_base_rate", "ranks_outcomes"):
        assert keys.index(verdict) < keys.index("threshold"), (
            f"{verdict} sits after the numeric detail and will be truncated")


def test_the_learning_health_report_states_the_rank_verdict(tmp_path,
                                                            monkeypatch):
    """`introspect learning` is the operator's other read of this fit. Fails
    in the world where it reports only the Brier comparison — which is the
    sentence that cannot support the words "no signal"."""
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
    p = t.fit()
    assert p is not None and p.ranks_outcomes == BEATS_YES

    report = LH.render_learning_health(tmp_path / "system" / "memory")
    assert "AUC" in report, "the report states only the probability verdict"
    assert "RANKS outcomes" in report
    assert ("%.3f" % p.auc) in report


def test_the_report_headline_counts_the_population_the_fit_CONSUMED(tmp_path,
                                                                    monkeypatch):
    """⚠ THE FIFTH SURFACE, AND IT IS A DIFFERENT KIND.

    `learning_health` re-derives the fit population from raw JSONL through a
    dict-level twin of the chain. §4EB added the direction gate to the fit
    and not to that twin, so the twin applied THREE of four filters and the
    report headline said 1275 samples where the fit consumed 1022 — the
    "population the agent never fits" defect, which is precisely what the
    twin exists to prevent, reintroduced by the change that fixed it one
    layer up.

    The twin ASKS rather than re-derives: it takes the fit's stored
    `bench_verdict`. Fails in the world where the two counts diverge.
    """
    from ghost_agent.core import learning_health as LH
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    cal_dir = tmp_path / "system" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    t = CalibrationTracker(cal_dir)
    for s in _aligned(200, origin="user", seed=1):
        t.record(composite=s.composite, outcome=s.outcome,
                 entropy_component=0.5, competence_component=s.composite,
                 origin="user")
    for s in _aligned(120, origin="bench", seed=2, invert=True):
        t.record(composite=s.composite, outcome=s.outcome,
                 entropy_component=0.5, competence_component=s.composite,
                 origin="bench")
    t.fit()                      # cold start: admits everything
    p = t.fit()                  # now with an instrument: excludes bench
    assert p.n_bench_misaligned == 120 and p.n_samples == 200

    report = LH.render_learning_health(tmp_path / "system" / "memory")
    head = next(l for l in report.splitlines() if "CALIBRATION:" in l)
    assert f"{p.n_samples} samples" in head, (
        f"the headline describes a population the fit never consumed: {head!r}")


def test_the_dict_twin_applies_the_STORED_verdict_not_its_own(tmp_path):
    """One verdict, one decider. A twin that re-measured could disagree with
    the fit on the same rows — the failure this module already paid for on
    `map_status` (three private copies) and `beats_base_rate` (one file,
    four surfaces, two answers)."""
    rows = ([{"origin": "user", "outcome": 1.0} for _ in range(10)]
            + [{"origin": "bench", "outcome": 1.0} for _ in range(5)])
    # Verdict says INVERTED → bench dropped.
    assert len(C.apply_bench_direction_rows(rows, BEATS_NO)) == 10
    # Anything else → untouched, including a params file that predates
    # the field entirely.
    for v in (BEATS_YES, BEATS_INDISTINGUISHABLE, BEATS_UNKNOWN, "", None):
        assert len(C.apply_bench_direction_rows(rows, v)) == 15, v
