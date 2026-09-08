"""Prediction-powered inference (PPI++) for the live experiment arms (§4FD, 2026-09-07).

Why this exists. §4CE measured that every live A/B arm read "no difference" for a
difference the design could not detect: a halved failure rate needs ~230–260 RESOLVED
turns per arm and the arms hold 67–76, because only rows with a verifier/human/structural
verdict count. Meanwhile almost every row carries a JUDGE score (the verifier verdict, the
calibrated confidence) and a small subset carries a GOLD human label. Prediction-powered
inference (Angelopoulos et al., 2023; PPI++ 2023) uses the judge on ALL rows and corrects
its bias with the gold subset, so the interval shrinks with the unlabeled count instead of
the gold count. arXiv 2601.05420 reports intervals 35–55% narrower than plain PPI at 1–10%
gold; arXiv 2601.21471 shows judge-only arm selection is provably unsafe, which is why
the gold correction is not optional.

The estimator, for a mean θ of the gold label Y with a judge prediction f:

    θ̂(λ) = λ · mean_unlabeled(f) + mean_labeled(Y − λ f)
    Var(θ̂) = λ² Var(f)/N + Var(Y − λ f)/n
    λ* = Cov(Y, f) / (Var(f) · (1 + n/N))          (clipped to [0, 1])

λ = 0 reduces to the classical gold-only mean; λ = 1 is plain PPI. The clip keeps a noisy
λ̂ from inflating variance at small n (the paper's recommendation).

**What it assumes, and the guard for it.** The labeled rows must be exchangeable with the
unlabeled ones — a random subset. This agent's labels are NOT random by design: §4ER asks
for a label on bottom-quintile turns. So every estimate carries a representativeness check
(judge mean on labeled vs unlabeled rows, two-sample z) and the report must print it. A
flagged estimate is still computed — absent is not withheld — but it is labelled as resting
on a skewed gold set.

Anytime validity. The existing arm report reads its intervals daily, so it uses an
asymptotic confidence sequence (`experiments.asymp_cs_radius`). The PPI estimator is the sum
of two independent means, so a CS for it is the sum of a CS on the labeled residuals
(Y − λ f) at α/2 and λ times a CS on the unlabeled judge scores at α/2 (union bound —
conservative, never anti-conservative). Pass that radius function as ``radius_fn`` to get
``cs_halfwidth`` beside the fixed-sample ``halfwidth``.

Pure functions, no I/O, no logging. Consumers: ``core.experiments`` (arm report).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

def _z(alpha: float) -> float:
    """Two-sided normal quantile for confidence 1-alpha — EXACT, via the
    standard library. (The first version snapped to the nearest tabulated
    level: at the arm report's α = 0.05/4/2 = 0.00625 it used z(0.01) = 2.576
    instead of 2.734 and printed every one-look width 5.8% too narrow —
    review §4FH m1.) Degenerate α is clamped into (0, 1)."""
    from statistics import NormalDist
    a = min(max(float(alpha), 1e-12), 1.0 - 1e-12)
    return NormalDist().inv_cdf(1.0 - a / 2.0)


def _mean(v: Sequence[float]) -> float:
    return sum(float(x) for x in v) / len(v)


def _var(v: Sequence[float]) -> float:
    """Sample variance (n-1); 0.0 for n < 2 is never reached (callers refuse)."""
    n = len(v)
    m = _mean(v)
    return sum((float(x) - m) ** 2 for x in v) / (n - 1)


def _cov(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = _mean(a), _mean(b)
    return sum((float(x) - ma) * (float(y) - mb) for x, y in zip(a, b)) / (n - 1)


@dataclass
class PPIEstimate:
    """One PPI++ mean estimate with its diagnostics.

    ``refusal`` is set (and the numeric fields are None) when the inputs cannot support an
    estimate — the reason is the value, so a report can say WHY a number is missing.
    """
    estimate: Optional[float]
    halfwidth: Optional[float]            # fixed-sample z-interval half-width
    cs_halfwidth: Optional[float]         # anytime-valid half-width, if radius_fn was given
    lam: Optional[float]
    n_labeled: int
    n_unlabeled: int
    classical_halfwidth: Optional[float]  # gold-only z-interval, for the shrink figure
    shrink: Optional[float]               # 1 - halfwidth / classical_halfwidth
    representative: Optional[bool]        # labeled judge-mean ≈ unlabeled judge-mean
    judge_gap: Optional[float]            # mean f(labeled) - mean f(unlabeled)
    judge_gap_z: Optional[float]          # that gap in standard errors
    refusal: Optional[str] = None


def ppi_mean(gold: Sequence[float],
             judge_labeled: Sequence[float],
             judge_unlabeled: Sequence[float],
             *,
             alpha: float = 0.05,
             clip: Tuple[float, float] = (0.0, 1.0),
             radius_fn: Optional[Callable[..., Optional[float]]] = None,
             ) -> PPIEstimate:
    """PPI++ estimate of mean(Y).

    ``gold`` and ``judge_labeled`` are paired (same rows, same order); ``judge_unlabeled``
    are the judge scores on rows WITHOUT a gold label. Scores are any real numbers (a
    0/1 verdict, a probability, a calibrated confidence) — λ absorbs the scale.

    ``radius_fn(vals, alpha=...)`` must return a half-width for the mean of ``vals`` or
    None; when given, ``cs_halfwidth`` is the union-bound anytime-valid half-width.
    """
    n = len(gold)
    N = len(judge_unlabeled)
    if n != len(judge_labeled):
        return _refuse(n, N, "gold and judge_labeled must be paired (same length)")
    if n < 2:
        return _refuse(n, N, f"fewer than 2 gold labels (have {n})")
    if N < 2:
        return _refuse(n, N, f"fewer than 2 judged unlabeled rows (have {N})")

    y = [float(v) for v in gold]
    f_l = [float(v) for v in judge_labeled]
    f_u = [float(v) for v in judge_unlabeled]

    var_fl = _var(f_l)
    var_fu = _var(f_u)
    # λ*: uninformative or constant judge → 0 (classical); clipped for small-n stability.
    if var_fl <= 0.0:
        lam = 0.0
    else:
        lam = _cov(y, f_l) / (var_fl * (1.0 + n / N))
    lo, hi = clip
    lam = max(lo, min(hi, lam))

    resid = [yi - lam * fi for yi, fi in zip(y, f_l)]
    estimate = lam * _mean(f_u) + _mean(resid)
    var_est = (lam * lam) * var_fu / N + _var(resid) / n
    z = _z(alpha)
    halfwidth = z * math.sqrt(max(var_est, 0.0))

    classical = z * math.sqrt(_var(y) / n)
    shrink = (1.0 - halfwidth / classical) if classical > 0 else 0.0

    # Representativeness: are the gold rows a random-looking slice of the judged rows?
    gap = _mean(f_l) - _mean(f_u)
    se_gap = math.sqrt(var_fl / n + var_fu / N)
    if se_gap > 0:
        gap_z = gap / se_gap
    else:
        # Both series constant: a non-zero gap with zero spread is the most
        # skewed slice possible ([1,1,1] gold vs [0,0,0] unlabeled), not a
        # representative one (review §4FH m4).
        gap_z = 0.0 if abs(gap) < 1e-12 else math.copysign(float("inf"), gap)
    # 2.5σ (≈1.2% false flags per arm): a deliberate "label the shaky ones"
    # policy sits far beyond this (z ≫ 3), while a random 20-row slice
    # would trip a 2σ bar one time in ten across two arms.
    representative = abs(gap_z) <= 2.5

    cs_hw: Optional[float] = None
    if radius_fn is not None:
        r_l = radius_fn(resid, alpha=alpha / 2.0)
        r_u = radius_fn(f_u, alpha=alpha / 2.0)
        if r_l is not None and r_u is not None:
            cs_hw = float(r_l) + lam * float(r_u)

    return PPIEstimate(estimate=estimate, halfwidth=halfwidth, cs_halfwidth=cs_hw, lam=lam,
                       n_labeled=n, n_unlabeled=N, classical_halfwidth=classical,
                       shrink=shrink, representative=representative, judge_gap=gap,
                       judge_gap_z=gap_z, refusal=None)


def _refuse(n: int, N: int, why: str) -> PPIEstimate:
    return PPIEstimate(estimate=None, halfwidth=None, cs_halfwidth=None, lam=None,
                       n_labeled=n, n_unlabeled=N, classical_halfwidth=None, shrink=None,
                       representative=None, judge_gap=None, judge_gap_z=None, refusal=why)


def ppi_difference(a: PPIEstimate, b: PPIEstimate, *, anytime: bool = False
                   ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """(a − b, half-width, refusal) for two INDEPENDENT arm estimates.

    Variances add; with ``anytime`` the union-bound CS half-widths add (conservative),
    otherwise the fixed-sample half-widths combine in quadrature.
    """
    if a.refusal:
        return None, None, f"arm A: {a.refusal}"
    if b.refusal:
        return None, None, f"arm B: {b.refusal}"
    diff = a.estimate - b.estimate                      # type: ignore[operator]
    if anytime:
        if a.cs_halfwidth is None or b.cs_halfwidth is None:
            return diff, None, "no anytime half-width on one arm (radius_fn absent)"
        return diff, a.cs_halfwidth + b.cs_halfwidth, None
    hw = math.sqrt(a.halfwidth ** 2 + b.halfwidth ** 2)  # type: ignore[operator]
    return diff, hw, None
