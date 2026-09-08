"""§4FD pins for core/ppi.py (prediction-powered inference for the arm report).

Each pin names the world in which it fails (R4): a wrong λ formula, a wrong variance
formula, an unclipped λ, a representativeness check that cannot flag, or a refusal that
hides its reason.
"""
import math
import random

import pytest

from ghost_agent.core.ppi import PPIEstimate, ppi_difference, ppi_mean


def _bern(rng, p, n):
    return [1.0 if rng.random() < p else 0.0 for _ in range(n)]


def _noisy_judge(rng, ys, flip):
    return [1.0 - y if rng.random() < flip else y for y in ys]


def test_lambda_zero_reduces_to_the_gold_only_mean():
    """World where it fails: the residual/rectifier algebra is wrong, so λ=0 does not
    reproduce the classical estimate and interval exactly."""
    rng = random.Random(1)
    y = _bern(rng, 0.3, 40)
    fl = _noisy_judge(rng, y, 0.2)
    fu = _bern(rng, 0.3, 500)
    est = ppi_mean(y, fl, fu, clip=(0.0, 0.0))
    assert est.refusal is None
    assert est.lam == 0.0
    assert est.estimate == pytest.approx(sum(y) / len(y))
    assert est.halfwidth == pytest.approx(est.classical_halfwidth)
    assert est.shrink == pytest.approx(0.0)


def test_perfect_judge_shrinks_the_interval_toward_the_unlabeled_count():
    """World where it fails: λ* is computed with the wrong denominator or clipped to 0,
    or Var(θ̂) omits the /N term, so a perfect judge does not shrink the interval."""
    rng = random.Random(2)
    y = _bern(rng, 0.3, 40)
    fl = list(y)                      # judge == gold on labeled rows
    fu = _bern(rng, 0.3, 4000)
    est = ppi_mean(y, fl, fu)
    assert est.refusal is None
    assert est.lam >= 0.9
    assert est.halfwidth < 0.35 * est.classical_halfwidth
    assert est.shrink > 0.65


def test_uninformative_judge_falls_back_to_the_classical_interval():
    """World where it fails: λ is not driven by Cov(Y, f), so an independent judge still
    changes the estimate, or the clip lets λ̂ go negative and inflate the width."""
    rng = random.Random(3)
    y = _bern(rng, 0.3, 200)
    fl = _bern(rng, 0.5, 200)         # independent of y
    fu = _bern(rng, 0.5, 5000)
    est = ppi_mean(y, fl, fu)
    assert est.refusal is None
    assert 0.0 <= est.lam <= 0.15
    assert est.halfwidth <= 1.10 * est.classical_halfwidth
    assert est.halfwidth >= 0.90 * est.classical_halfwidth


def test_halfwidth_is_the_stated_variance_formula_recomputed_independently():
    """Pin identity, not a property: recompute z·sqrt(λ²Var(f_u)/N + Var(Y−λf_l)/n) from the
    returned λ with this test's own arithmetic. World where it fails: a term is dropped,
    n and N are swapped, or the residual is Y−f instead of Y−λf."""
    rng = random.Random(4)
    y = _bern(rng, 0.3, 40)
    fl = _noisy_judge(rng, y, 0.2)
    fu = _bern(rng, 0.3, 80)          # N small on purpose so the /N term is material
    est = ppi_mean(y, fl, fu)
    lam = est.lam

    def var(v):
        m = sum(v) / len(v)
        return sum((x - m) ** 2 for x in v) / (len(v) - 1)

    resid = [yi - lam * fi for yi, fi in zip(y, fl)]
    expected = 1.959963984540054 * math.sqrt(lam * lam * var(fu) / len(fu) + var(resid) / len(y))
    assert est.halfwidth == pytest.approx(expected, rel=1e-9)
    assert est.estimate == pytest.approx(lam * (sum(fu) / len(fu)) + sum(resid) / len(resid))


def test_coverage_of_the_fixed_sample_interval_tracks_the_classical_interval():
    """World where it fails: Var(θ̂) is mis-specified (e.g. residual variance divided by N
    instead of n) — coverage collapses far below the classical z-interval's coverage at
    the same n. Both are asymptotic z-intervals on Bernoulli data at n=40, so both sit a
    little under nominal; the pin is that PPI does not sit materially under classical."""
    rng = random.Random(4)
    p_true = 0.3
    covered = classical_covered = 0
    reps = 600
    for _ in range(reps):
        y = _bern(rng, p_true, 40)
        fl = _noisy_judge(rng, y, 0.2)
        yu = _bern(rng, p_true, 600)
        fu = _noisy_judge(rng, yu, 0.2)
        est = ppi_mean(y, fl, fu)
        assert est.refusal is None
        if abs(est.estimate - p_true) <= est.halfwidth:
            covered += 1
        if abs(sum(y) / len(y) - p_true) <= est.classical_halfwidth:
            classical_covered += 1
    cov, cov_c = covered / reps, classical_covered / reps
    assert cov >= 0.88, (cov, cov_c)
    assert cov >= cov_c - 0.04, (cov, cov_c)


def test_difference_of_independent_arms_combines_variances_in_quadrature():
    """World where it fails: the difference half-width adds linearly (too wide) or takes
    one arm's width (too narrow), or the sign convention is not a − b."""
    rng = random.Random(5)
    ya, yb = _bern(rng, 0.2, 40), _bern(rng, 0.4, 40)
    a = ppi_mean(ya, _noisy_judge(rng, ya, 0.2), _bern(rng, 0.2, 400))
    b = ppi_mean(yb, _noisy_judge(rng, yb, 0.2), _bern(rng, 0.4, 400))
    diff, hw, refusal = ppi_difference(a, b)
    assert refusal is None
    assert diff == pytest.approx(a.estimate - b.estimate)
    assert hw == pytest.approx(math.sqrt(a.halfwidth ** 2 + b.halfwidth ** 2))


def test_representativeness_flags_a_skewed_gold_set_and_passes_a_random_one():
    """World where it fails: the check compares the wrong quantities (e.g. gold means)
    or its threshold cannot be crossed, so §4ER's bottom-quintile labelling is invisible."""
    rng = random.Random(6)
    yu = _bern(rng, 0.3, 2000)
    fu = _noisy_judge(rng, yu, 0.2)
    # Skewed: the gold rows are exactly the rows the judge scored 1 (a "label the shaky
    # ones" policy).
    y_sk = [1.0 if rng.random() < 0.8 else 0.0 for _ in range(60)]
    f_sk = [1.0] * 60
    est_sk = ppi_mean(y_sk, f_sk, fu)
    assert est_sk.refusal is None
    assert est_sk.representative is False
    assert abs(est_sk.judge_gap_z) > 2.0
    # Random: a uniform slice of the same population.
    idx = rng.sample(range(2000), 60)
    est_ok = ppi_mean([yu[i] for i in idx], [fu[i] for i in idx],
                      [fu[i] for i in range(2000) if i not in set(idx)])
    assert est_ok.refusal is None
    assert est_ok.representative is True


def test_refusals_name_their_reason_instead_of_returning_a_number():
    """World where it fails: a thin input returns an estimate anyway, or the refusal is
    an empty string / None so the report cannot say why the line is missing."""
    r1 = ppi_mean([1.0], [1.0], [0.0, 1.0, 1.0])
    assert r1.estimate is None and r1.halfwidth is None
    assert "gold" in r1.refusal and "1" in r1.refusal
    r2 = ppi_mean([1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.5])
    assert r2.estimate is None
    assert "unlabeled" in r2.refusal
    r3 = ppi_mean([1.0, 0.0], [1.0], [0.0, 1.0])
    assert "paired" in r3.refusal
    d, hw, why = ppi_difference(r1, r2)
    assert d is None and hw is None and why.startswith("arm A:")


def test_anytime_halfwidth_is_the_union_bound_over_both_populations():
    """World where it fails: cs_halfwidth ignores λ, uses α instead of α/2, or drops one
    of the two populations — it must equal r(resid, α/2) + λ·r(f_unlabeled, α/2)."""
    from ghost_agent.core.experiments import asymp_cs_radius
    rng = random.Random(7)
    y = _bern(rng, 0.3, 60)
    fl = _noisy_judge(rng, y, 0.2)
    fu = _bern(rng, 0.3, 800)
    est = ppi_mean(y, fl, fu, radius_fn=asymp_cs_radius)
    assert est.refusal is None and est.cs_halfwidth is not None
    resid = [yi - est.lam * fi for yi, fi in zip(y, fl)]
    expected = asymp_cs_radius(resid, alpha=0.025) + est.lam * asymp_cs_radius(fu, alpha=0.025)
    assert est.cs_halfwidth == pytest.approx(expected)
    # A CS read repeatedly is wider than a one-shot z-interval at the same n.
    assert est.cs_halfwidth > est.halfwidth
    diff, hw, why = ppi_difference(est, est, anytime=True)
    assert why is None and hw == pytest.approx(2 * est.cs_halfwidth)
    est_plain = ppi_mean(y, fl, fu)
    _, hw2, why2 = ppi_difference(est_plain, est_plain, anytime=True)
    assert hw2 is None and "radius_fn" in why2


def test_lambda_star_is_the_stated_formula_in_the_unclipped_regime():
    """§4FH (test-quality lens): the perfect-judge pin let `cov/var` (dropping
    the `1+n/N` term) survive because the clip hid it. Recompute λ* here at
    n=40, N=60 where the estimate is interior."""
    rng = random.Random(11)
    y = _bern(rng, 0.4, 40)
    fl = _noisy_judge(rng, y, 0.25)
    fu = _bern(rng, 0.4, 60)
    est = ppi_mean(y, fl, fu)

    def var(v):
        m = sum(v) / len(v); return sum((x - m) ** 2 for x in v) / (len(v) - 1)

    def cov(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        return sum((x - ma) * (z - mb) for x, z in zip(a, b)) / (len(a) - 1)
    expected = cov(y, fl) / (var(fl) * (1.0 + 40 / 60))
    assert 0.05 < expected < 0.95, expected        # interior, so the clip is inert
    assert est.lam == pytest.approx(expected, rel=1e-9)


def test_normal_quantile_is_exact_at_the_report_alpha():
    """§4FH m1. World where it fails: α snaps to the nearest tabulated level
    and the arm report's α = 0.00625 renders with z(0.01) — every one-look
    width 5.8% too narrow."""
    from ghost_agent.core.ppi import _z
    assert _z(0.05) == pytest.approx(1.959963984540054, rel=1e-9)
    assert _z(0.00625) == pytest.approx(2.7344, abs=2e-4)
    assert _z(0.01) == pytest.approx(2.5758293035489004, rel=1e-9)
    assert _z(0.5) == pytest.approx(0.6744897501960817, rel=1e-9)


def test_constant_series_with_a_gap_are_not_representative():
    """§4FH m4. World where it fails: zero spread makes z=0 and the most
    skewed slice possible ([1,1,1] gold vs [0,0,0] unlabeled) passes."""
    est = ppi_mean([1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0])
    assert est.refusal is None and est.representative is False
    est2 = ppi_mean([1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
    assert est2.representative is True      # constant and EQUAL: no gap


def test_estimate_is_a_dataclass_with_every_diagnostic_populated():
    """World where it fails: a field is silently left None on the success path, so a
    renderer prints 'None' where the report promised a number."""
    rng = random.Random(8)
    y = _bern(rng, 0.3, 30)
    est = ppi_mean(y, _noisy_judge(rng, y, 0.2), _bern(rng, 0.3, 300))
    assert isinstance(est, PPIEstimate)
    for name in ("estimate", "halfwidth", "lam", "classical_halfwidth", "shrink",
                 "representative", "judge_gap", "judge_gap_z"):
        assert getattr(est, name) is not None, name
    assert est.n_labeled == 30 and est.n_unlabeled == 300
