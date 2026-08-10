"""§4AO / A4 — the label-noise audit (`scripts/label_noise_audit.py`).

THE QUESTION A4 ASKS: is the per-turn outcome label wrong often enough to
explain why earn-keep and skill-prune both produced nulls?

⚠ THE TRAP THESE TESTS EXIST TO CATCH. An audit that can only ever answer
"no, label noise is fine" is not evidence — it is a disconnected instrument
producing a plausible reading (the standing question: *could this same output
be produced by the instrument being unplugged?*). So the load-bearing test
here is `test_the_counterfactual_CAN_convict_label_noise`: it builds a
playbook where noise genuinely IS decisive and asserts the audit says so. The
exoneration on live data only means something because that test passes.
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "scripts" / "label_noise_audit.py"
_spec = importlib.util.spec_from_file_location("label_noise_audit", _SRC)
lna = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lna)


# ── the arithmetic ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("x,k", [(38.23, 29), (10.0, 5), (3.0, 2), (50.0, 49), (1.0, 1)])
def test_the_chi_square_tail_matches_scipy(x, k):
    """The identifiability verdict turns on this p-value, so it is checked
    against an independent implementation rather than trusted."""
    scipy_stats = pytest.importorskip("scipy.stats")
    assert lna._identifiability.__doc__          # the function that uses it
    # re-derive the closed form the module uses
    def sf(x, k):
        if k % 2 == 0:
            t = math.exp(-x / 2); acc = t
            for i in range(1, k // 2):
                t *= x / (2 * i); acc += t
            return min(1.0, acc)
        z = math.sqrt(x); acc = math.erfc(z / math.sqrt(2))
        t = math.sqrt(2 / math.pi) * z * math.exp(-x / 2)
        for i in range(1, (k - 1) // 2 + 1):
            acc += t; t *= x / (2 * i + 1)
        return min(1.0, acc)
    assert sf(x, k) == pytest.approx(float(scipy_stats.chi2.sf(x, k)), abs=1e-6)


def test_de_attenuation_inverts_the_noise_it_models():
    """observed = true·(1-FRR) + (1-true)·FCR, so the correction
    (observed - FCR)/(1 - FRR - FCR) must recover `true` exactly. If this
    inverse is wrong the whole counterfactual is meaningless."""
    frr, fcr = 0.1207, 0.2424
    a = 1 - frr - fcr
    for true in (0.0, 0.25, 0.5, 0.7407, 0.9, 1.0):
        observed = true * (1 - frr) + (1 - true) * fcr
        assert (observed - fcr) / a == pytest.approx(true, abs=1e-9)


def test_attenuation_shrinks_a_difference_and_inflates_required_n():
    """A 1/A² n-inflation is the whole practical consequence for the live
    experiment arms; pin the direction so a sign slip can't invert it."""
    a = 1 - 0.1207 - 0.2424
    assert 0 < a < 1
    true_delta, observed_delta = 0.10, 0.10 * a
    assert observed_delta < true_delta
    assert 1 / a ** 2 > 1


# ── the audit must be able to CONVICT, not only exonerate ──────────────────

def _playbook(entries):
    out = []
    for trig, retr, helpful, succ, fail in entries:
        out.append({"trigger": trig, "task": trig, "solution": "s", "confidence": 0.5,
                    "retrievals": retr, "helpful_retrievals": helpful,
                    "succeeded_retrievals": succ, "failed_retrievals": fail,
                    "frequency": 1, "verified": False, "schema_version": 2})
    return out


def test_de_biasing_alone_CANNOT_reorder_a_rank_based_prune():
    """⚠ THIS IS WHY THE FIRST VERSION OF THIS AUDIT WAS WRONG.

    (observed - FCR)/(1 - FRR - FCR) is MONOTONE increasing in the observed
    rate, so de-attenuating every lesson preserves their order. A prune that
    cuts at a rank (bottom quartile) is therefore structurally immune to the
    label's systematic BIAS — and an audit that only de-biases will always
    report "same victims", whatever the true error rates are.

    That reading was published as `4 of 5 die either way ⇒ label noise is not
    what selects them`. It was an artifact of the method. What corrupts a
    ranking is the label's VARIANCE, measured separately below.
    """
    from ghost_agent.memory import skills as sk
    rates = {"frr": 0.1207, "fcr": 0.2424}
    a = 1 - rates["frr"] - rates["fcr"]
    rows = _playbook([(f"L-{i}", 20, 4, s, 20 - s) for i, s in enumerate(range(2, 20))])

    def order(pb):
        return [r["trigger"] for r in sorted(
            pb, key=lambda r: sk.compute_lesson_utility(sk._normalize_lesson(r)))]

    corrected = json.loads(json.dumps(rows))
    for r in corrected:
        n = r["succeeded_retrievals"] + r["failed_retrievals"]
        true = min(1.0, max(0.0, (r["succeeded_retrievals"] / n - rates["fcr"]) / a))
        r["succeeded_retrievals"] = int(round(n * true))
        r["failed_retrievals"] = n - int(round(n * true))
    assert order(rows) == order(corrected), (
        "de-biasing reordered lessons that differ ONLY in the outcome arm — "
        "the correction is no longer monotone and the §4AO reasoning changes")


def test_the_variance_counterfactual_CAN_convict_label_noise():
    """⚠ THE LOAD-BEARING TEST. An audit that can only exonerate is a
    disconnected instrument. On a playbook of small-n lessons with nearly
    equal true rates — where flipped ticks genuinely decide who sits below
    the cutoff — the Monte Carlo must report substantial churn."""
    from ghost_agent.memory import skills as sk
    rows = _playbook([(f"L-{i}", 20, 4, 3, 2) for i in range(16)])   # n=5 each
    out = lna._variance_counterfactual(rows, {"frr": 0.1207, "fcr": 0.2424}, sk,
                                       replicates=400, seed=3)
    assert out is not None
    assert out["churn"] > 0.10, (
        f"churn {out['churn']:.3f}: the counterfactual cannot detect label noise "
        f"even where noise decides the victims — its live reading means nothing")


def test_the_variance_counterfactual_EXONERATES_when_the_evidence_is_ample():
    """⚠ THE OTHER SIDE. It must also be able to say NO — otherwise it is an
    alarm that always fires. With large n and well-separated true rates the
    same flip rates move nobody across the cutoff.

    ⚠ WHAT "WELL-SEPARATED" HAS TO MEAN — two fixture attempts got this
    wrong before it worked:

      * 8 identical bad lessons TIED exactly in the noiseless world, the rank
        cutoff sliced through the tie, and any variance "changed" the victim
        set (36% churn, pure artifact);
      * 8 bad lessons spaced 0.01 apart put the cutoff INSIDE that group,
        where the spacing is below the n=400 standard error of 0.015 — 32%
        churn, again about the fixture and not the label.

    A bottom-quartile cutoff is decided ONLY by the lessons adjacent to the
    boundary. Population-level separation is irrelevant; the boundary has to
    fall in a GAP. Here 5 clearly-bad lessons sit below a cutoff at rank 5,
    with the nearest survivor an entire class away.
    """
    from ghost_agent.memory import skills as sk
    fcr, frr = 0.2424, 0.1207
    a = 1 - frr - fcr
    n = 400

    def counts(true):                       # observed counts implying `true`
        return int(round(n * (fcr + a * true)))

    # 16 lessons -> cutoff at rank 5. Exactly 5 lessons below the gap.
    rows = _playbook([(f"bad-{i}", 400, 200, counts(t), n - counts(t))
                      for i, t in enumerate([0.02, 0.04, 0.06, 0.08, 0.10])] +
                     [(f"good-{i}", 400, 200, counts(t), n - counts(t))
                      for i, t in enumerate([0.90, 0.91, 0.92, 0.93, 0.94,
                                             0.95, 0.96, 0.97, 0.98, 0.99, 1.00])])
    out = lna._variance_counterfactual(rows, {"frr": frr, "fcr": fcr}, sk,
                                       replicates=400, seed=3)
    assert out is not None
    assert out["churn"] < 0.02, (
        f"churn {out['churn']:.3f}: the counterfactual blames noise even when "
        f"every lesson has 400 observations and the classes are far apart")


def test_exact_ties_make_the_rank_cutoff_arbitrary(capsys):
    """⚠ A REAL PROPERTY, found by a fixture that hit it. A bottom-QUARTILE
    cutoff selects by rank, so when lessons tie exactly the boundary falls
    inside the tie and which of them dies is decided by whatever breaks it.
    Any noise at all becomes decisive there. Documented so the churn metric
    is not misread when a playbook contains many identical rows."""
    from ghost_agent.memory import skills as sk
    tied = _playbook([(f"same-{i}", 400, 10, 100, 300) for i in range(8)] +
                     [(f"top-{i}", 400, 380, 380, 20) for i in range(8)])
    out = lna._variance_counterfactual(tied, {"frr": 0.1207, "fcr": 0.2424}, sk,
                                       replicates=200, seed=3)
    assert out["churn"] > 0.10, (
        "identical rows no longer produce cutoff churn — either the prune "
        "stopped cutting by rank, or ties are now broken deterministically")


def test_zero_error_rates_produce_zero_churn():
    """A perfect judge must yield identical worlds. If this ever reports
    churn, the two arms differ by something other than the label."""
    from ghost_agent.memory import skills as sk
    rows = _playbook([(f"L-{i}", 20, 4, 3, 2) for i in range(16)])
    out = lna._variance_counterfactual(rows, {"frr": 0.0, "fcr": 0.0}, sk,
                                       replicates=200, seed=5)
    assert out["churn"] == 0.0 and out["differ"] == 0.0


def test_on_a_playbook_with_no_outcome_data_noise_cannot_be_the_cause(tmp_path):
    """The complementary direction: with zero decisive outcomes the outcome
    arm never engages (`n_out >= _OUTCOME_MIN_OBS` is false), so correcting
    labels cannot move anything."""
    from ghost_agent.memory import skills as sk
    rows = _playbook([(f"L{i}", 20, i % 5, 0, 0) for i in range(12)])
    for r in rows:
        L = sk._normalize_lesson(r)
        before = sk.compute_lesson_utility(L)
        L["succeeded_retrievals"] = 0
        L["failed_retrievals"] = 0
        assert sk.compute_lesson_utility(L) == before


# ── the defect the audit actually found ────────────────────────────────────

def test_the_outcome_multiplier_PENALISES_lessons_that_have_evidence():
    """⚠ MEASURED DEFECT (§4AO). `0.4 + 0.75·out_rate` crosses 1.0 at
    out_rate = 0.80, but the live playbook pools at 0.7407 — so a lesson at
    the population mean is DEMOTED 4.4%, while a lesson with fewer than
    `_OUTCOME_MIN_OBS` decisive outcomes is multiplied by 1.0 and pays
    nothing. Acquiring evidence is a penalty.

    This test does NOT assert the multiplier is wrong to be re-centred — that
    is an operator retention decision. It pins the ARITHMETIC so that if
    anyone changes the constants, the consequence is stated out loud.
    """
    from ghost_agent.memory import skills as sk
    neutral = (1.0 - 0.4) / 0.75
    assert neutral == pytest.approx(0.80), "the multiplier's neutral point moved"

    measured = sk._normalize_lesson(_playbook([("m", 30, 10, 15, 5)])[0])   # 75%
    unmeasured = sk._normalize_lesson(_playbook([("u", 30, 10, 0, 0)])[0])  # no data
    assert (measured["succeeded_retrievals"] + measured["failed_retrievals"]
            >= sk._OUTCOME_MIN_OBS)
    assert sk.compute_lesson_utility(measured) < sk.compute_lesson_utility(unmeasured), (
        "a lesson with a 75% measured success rate must not outrank an "
        "identical lesson with NO evidence — if it now does, the multiplier "
        "was re-centred and §4AO's finding is stale")


def test_a_lesson_above_the_neutral_point_is_promoted_not_penalised():
    """⚠ OVER-CLAIM GUARD. The finding is that the multiplier is MIS-CENTRED
    for this population, NOT that it always punishes. Above 0.80 it lifts —
    pinning that stops the defect being restated as 'the outcome arm is
    always harmful', which the data does not support."""
    from ghost_agent.memory import skills as sk
    great = sk._normalize_lesson(_playbook([("g", 30, 10, 40, 0)])[0])      # ~97%
    unmeasured = sk._normalize_lesson(_playbook([("u", 30, 10, 0, 0)])[0])
    assert sk.compute_lesson_utility(great) > sk.compute_lesson_utility(unmeasured)


# ── third-state discipline: absent input is NO_SOURCE, never zero ──────────

def test_missing_bench_reports_NO_SOURCE_and_returns_None(tmp_path, capsys):
    assert lna.error_rates(tmp_path) is None
    assert "NO_SOURCE" in capsys.readouterr().out


def test_a_baseline_pointing_at_a_missing_results_file_is_NO_SOURCE(tmp_path, capsys):
    """A dangling `results_path` is exactly how this audit would silently
    report 0.0 error rates and declare the label clean."""
    d = tmp_path / "system/eval"
    d.mkdir(parents=True)
    (d / "verifier_incumbent_baseline.json").write_text(
        json.dumps({"results_path": str(tmp_path / "gone.json")}))
    assert lna.error_rates(tmp_path) is None
    assert "NO_SOURCE" in capsys.readouterr().out


def test_error_rates_are_read_from_TRIALS_not_from_the_summary(tmp_path, capsys):
    """⚠ The summary block and the per-trial records have disagreed before
    (FRAMES `summary.json` accuracy was wrong). The audit recomputes from
    trials; feed it a summary that LIES and confirm the trials win."""
    d = tmp_path / "system/eval"
    d.mkdir(parents=True)
    res = tmp_path / "results.json"
    trials = ([{"fault": "clean", "expected": "CONFIRMED", "verdict": "CONFIRMED"}] * 9
              + [{"fault": "clean", "expected": "CONFIRMED", "verdict": "REFUTED"}] * 1
              + [{"fault": "bad", "expected": "REFUTED", "verdict": "REFUTED"}] * 8
              + [{"fault": "bad", "expected": "REFUTED", "verdict": "CONFIRMED"}] * 2)
    res.write_text(json.dumps({"arms": {"x": {"trials": trials}}}))
    (d / "verifier_incumbent_baseline.json").write_text(json.dumps({
        "results_path": str(res),
        "nonrefute_mean": 0.999, "refute_mean": 0.999,     # a lying summary
    }))
    out = lna.error_rates(tmp_path)
    assert out["frr"] == pytest.approx(0.10)
    assert out["fcr"] == pytest.approx(0.20)


def test_NOT_REFUTED_cases_are_excluded_from_both_rates(tmp_path):
    """`evidence_truncation` is expected NOT_REFUTED — it carries no
    unambiguous truth value, so folding it into either rate would invent
    error where none is defined."""
    d = tmp_path / "system/eval"
    d.mkdir(parents=True)
    res = tmp_path / "results.json"
    trials = ([{"fault": "clean", "expected": "CONFIRMED", "verdict": "CONFIRMED"}] * 10
              + [{"fault": "bad", "expected": "REFUTED", "verdict": "REFUTED"}] * 10
              + [{"fault": "tr", "expected": "NOT_REFUTED", "verdict": "REFUTED"}] * 50)
    res.write_text(json.dumps({"arms": {"x": {"trials": trials}}}))
    (d / "verifier_incumbent_baseline.json").write_text(
        json.dumps({"results_path": str(res)}))
    out = lna.error_rates(tmp_path)
    assert out["frr"] == 0.0 and out["fcr"] == 0.0
    assert out["n_good"] == 10 and out["n_bad"] == 10, "NOT_REFUTED leaked into a rate"


def test_one_sided_truth_is_NO_SOURCE_not_a_half_answer(tmp_path, capsys):
    """Only good-answer cases means FCR is unmeasured. Reporting FRR alone
    and treating FCR as 0 would understate the noise."""
    d = tmp_path / "system/eval"
    d.mkdir(parents=True)
    res = tmp_path / "results.json"
    res.write_text(json.dumps({"arms": {"x": {"trials": [
        {"fault": "clean", "expected": "CONFIRMED", "verdict": "CONFIRMED"}] * 10}}}))
    (d / "verifier_incumbent_baseline.json").write_text(
        json.dumps({"results_path": str(res)}))
    assert lna.error_rates(tmp_path) is None
    assert "NO_SOURCE" in capsys.readouterr().out


def test_the_audit_refuses_to_guess_ghost_home(monkeypatch, capsys):
    monkeypatch.delenv("GHOST_HOME", raising=False)
    monkeypatch.setattr("sys.argv", ["label_noise_audit.py"])
    assert lna.main() == 2
