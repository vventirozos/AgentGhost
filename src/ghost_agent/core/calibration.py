"""Closed-loop confidence calibration — roadmap phase 2.5.

The composite confidence (:mod:`core.confidence`) is computed every
turn but historically was never checked against what actually
happened. The threshold ``τ = 0.55`` and the entropy/competence
weights ``0.5 / 0.5`` were asserted constants, never fit to data — so
"the agent is 80 % confident" had no demonstrated relationship to the
turn succeeding 80 % of the time.

This module closes that loop. It is the calibration *spine*:

  1. **Measure.** Every turn that produced a confidence reading is
     paired with the realized outcome and appended to a JSONL log.
     From that log we compute a rolling **Brier score**
     ``mean((C − outcome)²)`` and an **Expected Calibration Error**
     over a 10-bin reliability table.
  2. **Self-tune.** Once enough samples accumulate, a small grid
     search re-fits the entropy/competence weights, the
     verbalised-uncertainty penalty ``λ``, and the decision threshold
     ``τ`` to minimise Brier on the logged history. The fitted params
     are persisted and loaded back into :class:`CompositeConfidence`.
  3. **Unify.** The recorded sample carries the verbalised-uncertainty
     *pressure* alongside the objective entropy/competence components,
     so the previously-disjoint "the agent said it was unsure" track
     and "the generation/​domain was uncertain" track are fit together.

Design non-negotiables (same as every other Stage-1 module):

* **Local-only, pure stdlib.** ``math`` + ``json`` only. No numpy, no
  hosted scorer, no outbound traffic.
* **JSONL / JSON on disk.** Human-diffable, append-only history, atomic
  param writes (``.tmp`` + ``os.replace``). Schema-versioned.
* **Fail-safe.** A recording or fit failure is logged at debug and
  never breaks a turn. ``load_params`` returns ``None`` on any problem
  so a corrupt file degrades to the hardcoded defaults, never a crash.
* **Bail-on-thin-data.** Like ``prm.trainer``: below the sample floor,
  or with only one outcome class present, ``fit`` returns ``None`` with
  a logged reason and writes no params — a confidently-miscalibrated
  threshold is worse than the neutral default.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import random
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

SCHEMA_VERSION = "ghost.calibration.v1"

# ── Corpus EPOCH ──────────────────────────────────────────────────────
# The history is append-only and `DEFAULT_MAX_HISTORY` exceeds it, so every
# refit pools the entire corpus. That is only valid while the label scheme
# and the feature set hold still — and they have not.
#
# Measured 2026-08-02 on 1709 live samples, the pooled fit was structurally
# broken in three compounding ways:
#
#   * The LABEL changed on 2026-07-27. Rows before it are binary {0.0, 1.0}
#     with base rate 0.955; rows after are graded with base rate 0.855. The
#     drop is a redefinition, not a decline in turn quality.
#   * The FEATURES changed on the same day. `effort_component` and real
#     logprob entropy are observed only from then on, so 1298 of 1709 rows
#     (76%) collapse to competence alone under `_composite_for` and outvote
#     the informative ones 3:1.
#   * The competence prior WARMS UP. Its cold-start shrinkage toward 0.5
#     means the mean rises 0.757 → 0.898 across the corpus while the label
#     mean falls 0.952 → 0.838.
#
# Pooled, the score rises exactly as the labels fall — a textbook Simpson's
# paradox that forces a NEGATIVE Platt slope. The map was then rejected as
# "anti-correlated" on every refit for ~26 days (141 rejections), Brier
# stayed on the raw scale (0.0542) and lost to the base rate (0.0365), and
# both warnings fired hourly. The signal was fine; the CORPUS was invalid.
# Same code on the current epoch alone: AUC 0.711, slope +2.242, Brier
# 0.0249 against a 0.0255 base rate — map applied, base rate beaten.
#
# So: a label-scheme or feature change MUST start a new epoch, and `fit`
# reads only the current one. Bumping this constant is not bookkeeping — it
# is what keeps the fit valid. Expect the loop to go quiet for a while after
# a bump (the fit bails below the sample floor rather than crossing epochs),
# which is the correct behaviour: no fit beats a fit on incomparable rows.
CURRENT_EPOCH = "2026-07-27.graded"

# Epoch boundaries for rows written BEFORE the field existed, newest first.
# Every row recorded from now on carries an explicit `epoch`, so this table
# is frozen history — it never needs an entry for a future epoch, and
# bumping `CURRENT_EPOCH` does not require touching it.
_UNTAGGED_EPOCHS = (
    ("2026-07-27.graded", "2026-07-27"),  # graded labels + effort/entropy
    ("2026-07-07.binary", ""),            # everything older
)


def epoch_for_ts(ts: object) -> str:
    """Epoch of an UNTAGGED legacy row, derived from its timestamp.

    Total and deterministic: anything that is not a recognisable ISO date
    lands in the OLDEST epoch. The boundaries are compared as strings, which
    only orders ISO dates correctly — a bare `t >= start` also promotes
    garbage, since "not-a-timestamp" > "2026-07-27" lexicographically. The
    failure direction matters: a mis-parsed row promoted into the current
    epoch is precisely the contamination this whole mechanism prevents, so
    the shape is checked before the comparison is trusted.
    """
    t = str(ts or "")
    if not (len(t) >= 10 and t[:4].isdigit() and t[4] == "-"
            and t[5:7].isdigit() and t[7] == "-" and t[8:10].isdigit()):
        return _UNTAGGED_EPOCHS[-1][0]
    for name, start in _UNTAGGED_EPOCHS:
        if not start or t >= start:
            return name
    return _UNTAGGED_EPOCHS[-1][0]

# Minimum number of samples that actually OBSERVED token logprobs before the
# entropy weight is allowed to move off zero. Below this the entropy column
# is mostly the neutral stand-in and any fitted w_entropy would be noise
# dressed as signal.
_MIN_ENTROPY_SAMPLES = 30

# Same evidence floor for the turn-effort weight.
_MIN_EFFORT_SAMPLES = 30

# Divergence backstop for the Platt slope. A separable batch drives the fit
# toward a near-step function that reports ~0 or ~1 for every turn and makes
# `below_threshold` a hair-trigger on one competence value; the L2 penalty in
# `_fit_platt` bounds this in normal use and this is the guard rail. Module
# level because the grid search consults it too — the candidate objective and
# the acceptance test must share one definition of "safe", or the search can
# select a point the stage below then rejects.
_MAX_SLOPE = 50.0

# Brier improvements smaller than this are treated as a TIE by the weight
# search, which then prefers the simpler model. Its ONLY job is to stop a
# strict `<` over floats from handing the decision to the last decimal place
# — the inverted-map selection this module used to make won by 4e-5. It is
# deliberately NOT the noise guard: measured over 1000 pure-noise corpora the
# spurious in-sample gain is heavy-tailed (median 0, but ~1e-3 at the tail,
# and it does NOT shrink as 1/n), so any tolerance large enough to absorb the
# tail also eats most of a real feature's contribution. Noise is handled
# where the evidence lives, by `_MIN_SEPARATION_SIGMAS` below.
_BRIER_TIE_TOL = 1e-4

# A feature's weight may only leave zero when the feature demonstrably
# SEPARATES the outcome classes: |mean(ok) − mean(bad)| must reach this many
# standard errors of that difference. The observation floors above ask "do we
# have enough samples?" and never "is there anything in them" — so the grid
# was free to buy a weight for a column of pure noise, which it does, because
# any extra degree of freedom pays off in-sample. The module already states
# the rule ("a feature is only useful if it VARIES and SEPARATES"); this is
# that rule applied where it bites, in the fit rather than only in the report.
#
# 2.5σ from measurement: over 1000 pure-noise corpora spanning n = 40…800 it
# false-admits 4.9% of the time (2.0σ admits 8.8%), while the live effort
# feature separates at 3.64σ and is never in danger. Tightening to 3.0σ buys
# only 1.5 points of false-admit and halves the real feature's headroom.
_MIN_SEPARATION_SIGMAS = 2.5


def _separation_sigmas(samples, attr: str) -> float:
    """|mean(success) − mean(failure)| for ``attr``, in standard errors.

    0.0 whenever the question is unanswerable (a class missing, fewer than
    two rows in either class, or zero pooled variance) — so an undecidable
    feature is pinned rather than admitted. Pure and total.

    Accepts :class:`CalibrationSample` objects OR raw JSONL dicts, because
    the telemetry (`learning_health`) reports the same verdict this gate
    decides and MUST call this exact function to do it. A second
    implementation over there would drift, and the module has already been
    bitten by that: `entropy_learnable` once printed "LEARNABLE" while the
    fit was pinning the weight to 0.
    """
    def _get(s, name):
        if isinstance(s, dict):
            return s.get(name)
        return getattr(s, name, None)

    try:
        def _out(s) -> float:
            try:
                return float(_get(s, "outcome") or 0.0)
            except (TypeError, ValueError):
                return 0.0
        usable = [s for s in samples if _get(s, attr) is not None]
        ok = [float(_get(s, attr)) for s in usable if _out(s) >= 0.5]
        bad = [float(_get(s, attr)) for s in usable if _out(s) < 0.5]
        if len(ok) < 2 or len(bad) < 2:
            return 0.0
        m_ok = sum(ok) / len(ok)
        m_bad = sum(bad) / len(bad)
        v_ok = sum((v - m_ok) ** 2 for v in ok) / len(ok)
        v_bad = sum((v - m_bad) ** 2 for v in bad) / len(bad)
        se = math.sqrt(v_ok / len(ok) + v_bad / len(bad))
        delta = abs(m_ok - m_bad)
        if se <= 0.0:
            # Zero within-class spread. Two very different cases share this
            # denominator and must NOT share an answer: a CONSTANT column
            # (means equal too) is undecidable → 0.0, but a column that is
            # constant WITHIN each class and different BETWEEN them is a
            # PERFECT separator — the best feature obtainable, not the
            # worst. Returning 0.0 for both would make this gate reject the
            # ideal signal outright.
            return 0.0 if delta <= 0.0 else math.inf
        return delta / se
    except Exception:  # noqa: BLE001 — a gate must never break a fit
        return 0.0

# ── Graded outcome labels ─────────────────────────────────────────────
# The label used to be binary: 0.0 if (any execution failure OR verifier
# REFUTED OR budget exhausted) else 1.0. That produced 1177 positives to 49
# negatives — 96.1% one class — and it measured the wrong thing. "Nothing
# visibly broke" is not "the answer was good", and a turn that hit one tool
# error, recovered, and answered correctly was labelled a FAILURE.
#
# A graded label in [0, 1] fixes both at once: it carries information on
# EVERY turn instead of on the 4% that break, and it stops mislabelling
# recoveries. Brier and log-loss both accept soft targets unchanged.
#
# The constants are measured, not chosen. Across 302 verdict-bearing
# trajectories the agent passed 251 — so a clean turn that no verifier
# could check is scored at the observed P(good | checkable) rather than an
# asserted 1.0. Claiming 1.0 for an unverified turn is precisely the
# verification theatre this project keeps finding.
_UNVERIFIED_PRIOR = 0.83
# Each execution failure the turn had to absorb costs this much. A turn that
# recovered from one error is materially worse than a clean one, but nothing
# like a refuted answer.
_EXEC_FAILURE_PENALTY = 0.15
# Floor for a turn that answered at all — reserve 0.0 for a verdict of WRONG.
_DEGRADED_FLOOR = 0.15
# Budget exhaustion: the reply is explicitly flagged working-state/PARTIAL,
# i.e. the agent itself reports it did not finish.
_BUDGET_EXHAUSTED_GRADE = 0.2
# A user-reported failure ("it still doesn't work", a pasted traceback).
# Strong — the human is telling you the delivered work is broken — but a
# notch above an explicit correction's 0.0, because attribution is slightly
# looser: the report might name a defect the previous turn did not cause.
_FAILURE_REPORT_GRADE = 0.15
# §4E Tier 3 (2026-08-01): a task the turn closed DONE was later REOPENED —
# delayed evidence the closing was premature. Same strength as a failure
# report (attribution equally loose: the reopen can also reflect scope
# growth rather than wrongness), never 0.0.
_TASK_REOPENED_GRADE = 0.15


def grade_turn_outcome(*, verifier_verdict=None, execution_failure_count: int = 0,
                       budget_exhausted: bool = False,
                       unacked_total_failure: bool = False) -> float:
    """Map a finished turn's observable signals onto a quality label in [0, 1].

    IMPORTANT — this is a PROXY, not ground truth. Only the verifier arms
    are checked facts; everything else is a prior over "did this go well",
    and calibrating against it teaches the agent about *process health*, not
    correctness. Keep at least one ground-truth source (user corrections)
    flowing and tracked separately, or the score drifts toward rewarding
    "did not visibly break" — which is what the agent already over-indexes
    on. Sample provenance (`CalibrationSample.source`) exists so the two can
    always be told apart.

    ``unacked_total_failure`` is the 2026-08-04 shape rule
    (``distill.outcome_heuristics.resolve_turn_outcome`` rule 2b) in this
    function's currency: when every tool call this turn failed and the reply
    never said so, the verifier's PASS is not evidence the turn went well, so
    it does not buy the 1.0. This mirror is load-bearing — the corpus label
    and the calibration label are two different consumers of the same ladder,
    and grading req 03b96c28's fabricated ``0`` as a perfect 1.0 would teach
    the confidence model that a confident answer over three broken tools is
    exactly what success looks like. The turn falls through to the graded
    execution-failure path instead of getting a hard 0.0: the shape says
    "unverified and something broke", not "checked and wrong".

    Pure and total: never raises, always returns a value in [0, 1].
    """
    try:
        verdict = str(verifier_verdict or "").strip().lower()
        if verdict == "failed":
            return 0.0            # checked and WRONG — the one hard negative
        if verdict == "passed" and not unacked_total_failure:
            return 1.0            # checked and RIGHT
        if budget_exhausted:
            return _BUDGET_EXHAUSTED_GRADE
        try:
            fails = max(0, int(execution_failure_count or 0))
        except (TypeError, ValueError):
            fails = 0
        grade = _UNVERIFIED_PRIOR - (_EXEC_FAILURE_PENALTY * fails)
        return _clamp01(max(_DEGRADED_FLOOR, grade))
    except Exception:  # noqa: BLE001 — labelling must never break a turn
        return _UNVERIFIED_PRIOR


# Source precedence for per-request label resolution (§4L Lens-A/B,
# 2026-08-07). Higher rank = stronger evidence, mirroring the outcome
# ladder: a user correction outranks a late verifier verdict outranks
# the inline turn grade; §4E's task-reopened retro-negative and the
# failure-report tier sit between. §4BF 1c adds `bench_validator`: the
# bank oracle's mechanical verdict on a bench solve — ground truth for
# its own population, so it outranks every inferred tier; the human
# stays on top (bench req_ids never collide with real ones, the order
# is for principle, not traffic).
_SOURCE_RANK = {"turn": 0, "verifier_late": 1, "task_reopened": 2,
                "failure_report": 3, "bench_validator": 4,
                "user_correction": 5}


def resolve_superseded_rows(rows):
    """Dict-level twin of `_resolve_superseded` for raw JSONL consumers
    (learning-health feature stats — §4L R3 MINOR-1: they counted BOTH
    labels of a corrected request in separation sigmas while the fit saw
    one). Same rank table, same recency tiebreak, same no-req
    passthrough; one implementation of the ladder per data shape, both
    reading `_SOURCE_RANK` so they cannot drift."""
    best: dict = {}
    passthrough = []
    for i, r in enumerate(rows or []):
        rid = str(r.get("req_id") or "") if isinstance(r, dict) else ""
        if not rid:
            passthrough.append((i, r))
            continue
        rank = _SOURCE_RANK.get(str(r.get("source") or "turn"), 0)
        prev = best.get(rid)
        if prev is None or rank >= prev[0]:
            best[rid] = (rank, i, r)
    keep = passthrough + [(i, r) for (_rk, i, r) in best.values()]
    keep.sort(key=lambda t: t[0])
    return [r for _i, r in keep]


def _resolve_superseded(samples):
    """One label per request: the highest-ranked source wins, recency
    breaks ties.

    ⚠ The corpus is append-only and the fit used to consume EVERY row,
    so a §4E retro-negative (0.15) landed NEXT TO the original 1.0 turn
    row and was averaged — the docstring said "re-label", the arithmetic
    said "counter-weight at half strength" (measured live: req 316dc5b3
    carried both). Rows without a req_id cannot be joined and pass
    through untouched; file order is the recency tiebreak (append-only).
    """
    best: dict = {}
    passthrough = []
    for i, s in enumerate(samples):
        rid = getattr(s, "req_id", "") or ""
        if not rid:
            passthrough.append((i, s))
            continue
        rank = _SOURCE_RANK.get(getattr(s, "source", "turn") or "turn", 0)
        prev = best.get(rid)
        if prev is None or rank >= prev[0]:
            best[rid] = (rank, i, s)
    keep = passthrough + [(i, s) for (_r, i, s) in best.values()]
    keep.sort(key=lambda t: t[0])
    return [s for _i, s in keep]


def apply_bench_mass_cap_rows(rows):
    """Dict-level twin of `_apply_bench_mass_cap` for raw-JSONL consumers
    (learning-health label/feature stats — §4BF 1c R1 review: they gated
    "entropy learnable" and the separation sigmas on an UNCAPPED,
    bench-flooded superset while the fit computed the same gates on the
    capped blend — the "instrument printed LEARNABLE while the fit pinned
    to 0" lie, one origin richer). Same one-implementation-per-shape rule
    as `resolve_superseded_rows`: both cap twins keep the newest bench
    rows up to the real-row count, zero when no real rows."""
    real_n = sum(1 for r in rows or []
                 if str((r or {}).get("origin") or "user") != "bench")
    bench = [r for r in rows or []
             if str((r or {}).get("origin") or "user") == "bench"]
    if len(bench) <= real_n:
        return list(rows or [])
    keep = {id(r) for r in bench[-real_n:]} if real_n else set()
    return [r for r in rows or []
            if str((r or {}).get("origin") or "user") != "bench"
            or id(r) in keep]


def apply_bench_direction_rows(rows, bench_verdict):
    """Dict-level twin of `_apply_bench_direction_gate` for raw-JSONL
    consumers — and it ASKS rather than re-derives.

    ⚠ THE VERDICT IS ALREADY COMPUTED. `fit` decides whether the auxiliary
    population ranks the same way and writes the answer to
    `bench_verdict` on the params. A telemetry twin that re-ran the
    measurement would be a second definition of one verdict, free to
    disagree with the fit on the same rows — the failure this module has
    now paid for on `map_status` (three private copies) and on
    `beats_base_rate` (one file, four surfaces, two answers). So this takes
    the STORED verdict as an argument and only applies it.

    Anything other than a confident inversion keeps every row, so a params
    file that predates the field (verdict absent → not `"no"`) behaves
    exactly as before.
    """
    if str(bench_verdict or "") != BEATS_NO:
        return list(rows or [])
    return [r for r in rows or []
            if str((r or {}).get("origin") or "user") != _AUX_ORIGIN]


def _apply_bench_mass_cap(samples):
    """§4BF Track 1c: blend populations under an EQUAL-MASS cap.

    Bench solves accumulate labeled rows ~an order of magnitude faster
    than real turns (~6-11/night vs scarce live turns, most unknown). The
    operator decision admits them to the fit — but the instrument being
    fitted scores REAL turns, so bench may contribute at most as many
    rows as the real population (the newest bench rows win; recency is
    file order, the corpus is append-only). With zero real rows bench
    contributes nothing: a calibration curve fit purely on bench would
    describe a population the composite never scores in production.
    Applied in `fit` AND `_load_epoch` so Brier/ECE/reliability describe
    the exact population the fit consumed (the §4L "population the agent
    never fits" rule, one origin richer).
    """
    real_n = sum(1 for s in samples
                 if getattr(s, "origin", "user") != "bench")
    bench = [s for s in samples if getattr(s, "origin", "user") == "bench"]
    if len(bench) <= real_n:
        return list(samples)
    keep = {id(s) for s in bench[-real_n:]} if real_n else set()
    return [s for s in samples
            if getattr(s, "origin", "user") != "bench" or id(s) in keep]


# The model-text-only hedge scan landed 2026-08-01; before it, the
# finalize-appended SYSTEM disclaimer ("⚠ Unverified: … I cannot
# confirm") was scanned as the agent's own hedge — a deterministic echo
# of the label (§4L Lens-A MAJOR-2: the live λ=0.2 was bought entirely
# by 4 such rows, and zeroing them flips the fit to λ=0.0).
_PRESSURE_SEMANTICS_FIX_TS = "2026-08-02T00:00:00"


def _migrate_leaked_pressure(samples):
    """Zero `uncertainty_pressure` on rows recorded before the
    model-text-only scan fix.

    ⚠ The feature's SEMANTICS changed mid-epoch without an epoch bump —
    a violation of CURRENT_EPOCH's own contract. Bumping the epoch now
    would discard 600+ comparable rows to fix 7; instead this dated
    loader migration makes the semantics uniform: before the fix the
    feature was unmeasurable, so it reads 0.0 (its usual value). Any
    FUTURE feature-semantics change must either bump the epoch or add
    its own dated migration here — this function is the precedent.
    """
    out = []
    for s in samples:
        if (getattr(s, "uncertainty_pressure", 0.0)
                and str(getattr(s, "ts", "")) < _PRESSURE_SEMANTICS_FIX_TS):
            s = replace(s, uncertainty_pressure=0.0)
        out.append(s)
    return out


def _outcome_variance(samples) -> float:
    """Population variance of the outcome column. Replaces "are both binary
    classes present?" as the has-this-data-any-information test, which is
    the same question once labels are continuous (for 0/1 labels the
    variance is 0 exactly when one class is missing, so binary corpora keep
    their previous behaviour bit-for-bit)."""
    if not samples:
        return 0.0
    vals = [s.outcome for s in samples]
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def _sigmoid(z: float) -> float:
    # Overflow-safe: exp(710) overflows a float64.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z)) if z < 700 else 1.0
    ez = math.exp(z) if z > -700 else 0.0
    return ez / (1.0 + ez)


def apply_platt(composite: float, a: float, b: float) -> float:
    """Map a blended composite onto a calibrated probability.

    Identity when ``a == 1 and b == 0`` is NOT true of this transform, so
    callers must only apply it with fitted parameters; `FittedParams`
    defaults exist so an unfitted file is detectable, and
    :meth:`CompositeConfidence.score` skips the mapping in that case.
    """
    return _clamp01(_sigmoid(a * float(composite) + b))


def _fit_platt(pairs, *, iters: int = 50, ridge: float = 1e-2):
    """Fit (a, b) of ``sigmoid(a·c + b)`` on log-loss by Newton/IRLS.

    Log-loss (not Brier) because it is convex here and is the proper scoring
    rule for probability estimates; Brier is then reported on the result.

    Newton rather than gradient descent because the ANSWER MUST NOT DEPEND ON
    THE ITERATION BUDGET. Plain GD on this data was still at a = 1.28 after
    4000 steps and only reached its true optimum of a ≈ −0.08 after ~60000 —
    and those two values sit on opposite sides of the safety guard in
    `fit`, so an under-converged run would have silently adopted a map the
    converged fit rejects. IRLS converges here in well under 20 iterations.

    ``ridge`` is a REAL L2 penalty (it enters the gradient as well as the
    Hessian), making this a MAP estimate rather than an MLE. That is what
    bounds the slope on linearly SEPARABLE data, where the unpenalised
    likelihood has its optimum at infinity: with damping applied only to
    the Hessian — the earlier form — the fixed point was still that
    infinite optimum, and the slope simply grew with the iteration budget
    (a = 51 → 207 → 233 at 5 → 50 → 200 iterations on a separable corpus),
    contradicting the guarantee in the paragraph above. A separable batch
    is entirely reachable here: 40 samples split cleanly by competence is
    all it takes.

    Deliberately UNWEIGHTED. Class-balancing was tried first, on the theory
    that ~96% positives would drown the minority — but balancing optimises a
    reweighted distribution, so the resulting probabilities describe a 50/50
    world that does not exist, and the unweighted Brier got worse. The
    minority class is handled where it belongs: by the threshold search,
    which uses Youden's J and weights sensitivity and specificity equally
    regardless of prevalence.
    """
    n = len(pairs)
    if n < 2:
        return 1.0, 0.0
    # Variance test, not class-presence — see `fit`. Soft targets in [0, 1]
    # are valid for log-loss (cross-entropy with a soft label), so a graded
    # corpus is fittable; a CONSTANT one is not.
    _ymean = sum(y for _, y in pairs) / n
    if sum((y - _ymean) ** 2 for _, y in pairs) / n < 1e-9:
        return 1.0, 0.0
    a, b = 1.0, 0.0
    for _ in range(iters):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for c, y in pairs:
            p = _sigmoid(a * c + b)
            err = p - y
            w = p * (1.0 - p)
            g_a += err * c
            g_b += err
            h_aa += w * c * c
            h_ab += w * c
            h_bb += w
        # L2 penalty on BOTH the gradient and the Hessian. Adding it to the
        # Hessian alone is Levenberg damping — it changes the step size but
        # not the fixed point, so it cannot bound a separable fit.
        g_a += ridge * a
        g_b += ridge * b
        h_aa += ridge
        h_bb += ridge
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a -= da
        b -= db
        if not (math.isfinite(a) and math.isfinite(b)):
            return 1.0, 0.0
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    if not (math.isfinite(a) and math.isfinite(b)):
        return 1.0, 0.0
    return a, b


def _cv_brier(pairs, *, k: int = 5, seed: int = 7) -> Optional[float]:
    """OUT-OF-SAMPLE Brier: Platt fitted on k−1 folds, scored on the held-out
    one, averaged over every row.

    ⚠ WHY THIS EXISTS (audit 2026-08-10). `brier` in the params file is the
    IN-SAMPLE Brier of a 2-parameter Platt map on the rows it was fitted to,
    and `learning_health` printed it as the headline next to
    `brier_base_rate` — a comparison the model cannot lose fairly, because
    only one side of it paid for its parameters. Measured on the live store:
    in-sample claimed a 3.0% gain over the base rate; honest 5-fold CV gives
    **1.0%**. The gain is real but a third the advertised size.

    Deterministic (fixed seed, no shuffle-on-read) so two fits on identical
    data produce the identical number — an audit trail that moves on its own
    is not one.
    """
    rows = [(float(c), float(o)) for c, o in pairs]
    n = len(rows)
    if n < 2 * k:
        return None
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    total = 0.0
    for f in range(k):
        test = set(idx[f::k])
        train = [rows[i] for i in range(n) if i not in test]
        if len(train) < 2:
            return None
        a, b = _fit_platt(train)
        for i in test:
            c, o = rows[i]
            total += (apply_platt(c, a, b) - o) ** 2
    return total / n


#: The four answers to "is this model better than predicting the base rate?"
BEATS_YES = "yes"
BEATS_NO = "no"
BEATS_INDISTINGUISHABLE = "indistinguishable"
BEATS_UNKNOWN = "unknown"

#: Verdicts that do NOT license treating the score as informative.
NOT_INFORMATIVE = frozenset({BEATS_NO, BEATS_INDISTINGUISHABLE, BEATS_UNKNOWN})
# The whole vocabulary — anything outside it is not a verdict.
_VERDICTS = frozenset({BEATS_YES}) | NOT_INFORMATIVE


def map_applied(map_status) -> bool:
    """Is the Platt map in force, so a CV interval describes what ships?

    ⚠ ONE ANSWER FOR ONE QUESTION. `""` was read three different ways: the
    renderer treated it as APPLIED, `agent.py` emitted `refit=map_` (i.e.
    rejected), and `load_params` defaulted only an ABSENT key to "applied".
    ⚠ ABSENT IS NOT THE SAME AS NULL. "Written before `map_status` existed"
    is a fact about a file's AGE and reads as applied; an explicit
    `"map_status": null` is a malformed value and fails closed. This
    function only ever sees a VALUE, so `None` is the null case — use
    `map_applied_in_params` when you hold the dict and can tell the two
    apart. `load_params` already made that distinction by accident:
    `str(d.get("map_status", "applied"))` yields `"applied"` for an absent
    key and the string `"None"` for a null. That left the loader failing
    closed on a null while the learning-health collector passed `None`
    through, which read as APPLIED — two readers, two policies, one file.
    """
    return map_status == "applied"


def map_applied_in_params(params: dict) -> bool:
    """`map_applied` for a raw params dict, where absence is observable."""
    if "map_status" not in params:
        return True          # older than the field itself
    return map_applied(params["map_status"])


def downgrade_for_map(verdict: str, map_status) -> str:
    """Withhold a verdict that does not describe the shipped predictor.

    ⚠ THE RULE HAD ONE HOME AND TWO CALLERS (2026-08-30). `_cv_brier` and
    `_cv_delta_ci` fit a Platt map per fold, so the interval measures the
    CALIBRATED model; on every rejection path `fit()` ships the RAW
    composite instead. `fit()` applied that rule. `learning_health`, which
    DERIVES a verdict for any params file written before the field existed,
    did not — it called `_base_rate_verdict`, which sees only an interval.

    That is not a corner: every params file written before this change has
    the field absent, including the live one, so the derive path is the live
    path. Measured on a real anti-correlated fit with the field stripped —
    exactly what the previous producer wrote — the renderer printed "beats
    the base-rate predictor" for a shipped scorer whose Brier was 3.5x worse
    than a constant. The producer's refusal was correct and the consumer
    re-granted the licence one branch above the branch that had just been
    fixed to stop it.

    So the rule lives here, and both callers ask it.
    """
    return verdict if map_applied(map_status) else BEATS_UNKNOWN


def _verdict_from_params(d: dict) -> str:
    """The stored verdict, or the one its own interval implies.

    ⚠ THE DERIVE RULE HAD TWO OF ITS THREE READERS. `learning_health`
    derives a verdict from `brier_cv_delta_lo/hi` when the field is absent —
    every params file written before 2026-08-30 is on that path, INCLUDING
    THE LIVE ONE — and this loader, holding the same two fields, did not. So
    one file told two operator surfaces different stories:

        load_params / stats / calib_startup_fields : "unknown"
        agent idle CALIB                           : refit=ok/no_prob:unknown
        activity ledger                            : ", NO base-rate licence"
        introspect learning                        : "beats the base-rate predictor"

    Today's live file straddles zero so both land in NOT_INFORMATIVE and the
    disagreement is invisible; one refit whose interval clears zero puts them
    in open contradiction.

    Absence is the only licence to derive. A present-but-null verdict is
    malformed and fails closed, exactly as a typo does — collapsing those two
    made `null` the most permissive value the field accepts.
    """
    status = str(d.get("map_status", "applied"))
    if "beats_base_rate" in d:
        return downgrade_for_map(_known_verdict(d.get("beats_base_rate")),
                                 status)
    lo, hi = d.get("brier_cv_delta_lo"), d.get("brier_cv_delta_hi")
    if lo is None or hi is None:
        return BEATS_UNKNOWN
    return downgrade_for_map(_base_rate_verdict((None, lo, hi)), status)


def _known_verdict(value) -> str:
    """Coerce a persisted verdict to the vocabulary, or to `unknown`.

    ⚠ `str(d.get(...))` accepted anything. A params file carrying a typo, a
    value from a future version, or a hand edit loaded as that raw string and
    then flowed into consumers that test `in NOT_INFORMATIVE` — which is
    false for an unrecognised word, so it read as LICENSED. A verdict we
    cannot recognise is a verdict we do not have.
    """
    # ⚠ TOTAL, INCLUDING UNHASHABLE. `value in _VERDICTS` RAISES on a list
    # or dict — and this helper's own docstring names "a hand edit" as the
    # threat. The raise propagated: `load_params` discarded the entire fit,
    # and the renderer (which does its own membership test) aborted the WHOLE
    # learning-health report with `TypeError: unhashable type: 'list'`, so one
    # malformed advisory field destroyed lessons, episodes, PRM and router
    # reporting too.
    try:
        return value if value in _VERDICTS else BEATS_UNKNOWN
    except TypeError:
        return BEATS_UNKNOWN


def _base_rate_verdict(delta_ci) -> str:
    """Decide ONCE whether the model beats a constant, from the INTERVAL.

    ⚠ ASKED ONCE, STORED, AND READ — never re-derived. `learning_health`
    used to recompute this from the raw `lo`/`hi` fields at its own render
    site, which is how a producer's honest measurement and a consumer's
    rendering of it can drift apart. One function, one answer, one field on
    the params.

    The SIGN OF THE INTERVAL licenses the claim, not the point estimate:
    an interval straddling zero means the two predictors are
    indistinguishable at this sample size, however favourable the midpoint
    looks. `unknown` when no interval could be computed (too few rows, or a
    params file written before the CI existed) — and `unknown` is in
    `NOT_INFORMATIVE`, so an absent measurement never reads as a licence.

    ⚠ INTERVAL-ONLY, DELIBERATELY (2026-08-30). This took `brier_cv` and
    `brier_base` too, and fell back to a point comparison when no interval
    existed. Two problems. First, from `fit()` that arm is unreachable
    (`_cv_delta_ci` returns None only below `2*k` = 10 rows, and the default
    fit floor is 40), so its two arguments were dead there — and a mutant
    swapping them survived every test that could ever be written, because no
    reachable world observes the difference. A pin guarding "that arm is
    unreachable" read a DEFAULT-constructed tracker, while the floor is a
    per-tracker constructor argument: setting it to 8 made the arm live
    again with the suite green, and the swap then inverts a real verdict.
    Second, the arm returned BEATS_NO from a point estimate — a measured
    word from exactly the comparison this whole change says does not license
    one.

    So: no usable interval, no verdict. `unknown` now means precisely "no
    interval I could read", at every caller, and there is no argument left
    to get wrong. The renderer keeps its own clearly-labelled point-estimate
    line for legacy params, which is where that fallback belongs.
    """
    try:
        # ⚠ EXACTLY THREE. `_cv_delta_ci` returns `(delta, lo, hi)`; anything
        # else is a producer whose shape changed under us. Accepting a longer
        # tuple means a future reorder keeps returning YES from whatever now
        # sits at indices 1 and 2.
        if delta_ci is not None and len(delta_ci) != 3:
            return BEATS_UNKNOWN
        if delta_ci:
            lo_raw, hi_raw = delta_ci[1], delta_ci[2]
            # ⚠ REAL, FINITE, ORDERED — or we have no interval at all.
            # `float()` accepts strings and `isinstance(True, int)` is True,
            # so without these `("-0.01", "-0.002")`, `(nan, -0.001)`,
            # `(inf, -inf)` and an INVERTED `(0.01, -0.01)` all returned
            # YES — the single answer that licenses trusting the score.
            # A NaN bound is a broken measurement, not a tested tie, so it
            # lands on `unknown` rather than `indistinguishable`.
            if (isinstance(lo_raw, bool) or isinstance(hi_raw, bool)
                    or not isinstance(lo_raw, (int, float))
                    or not isinstance(hi_raw, (int, float))):
                return BEATS_UNKNOWN
            lo, hi = float(lo_raw), float(hi_raw)
            if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
                return BEATS_UNKNOWN
            if hi < 0.0:
                return BEATS_YES          # entirely better than the base rate
            if lo > 0.0:
                return BEATS_NO           # entirely worse
            return BEATS_INDISTINGUISHABLE
        # No interval, no verdict. See the docstring: the point-estimate
        # fallback that used to live here was unreachable from `fit()`,
        # carried two arguments nothing could observe, and answered with a
        # measured word.
        return BEATS_UNKNOWN
    except Exception:  # noqa: BLE001 — a verdict must never break the fit
        return BEATS_UNKNOWN


def _cv_delta_ci(pairs, *, k: int = 5, seed: int = 7, boots: int = 2000,
                 alpha: float = 0.05):
    """Paired bootstrap CI for (cross-validated model loss − base-rate loss).

    ⚠ WHY (queue #8, 2026-08-21). `learning_health` called the model a WINNER
    whenever `brier_cv` sat below `brier_base_rate` by more than 1e-6 — a
    tolerance that makes any numerical wobble a victory. Measured on the live
    store the same day: brier_cv 0.03889 against a base rate of 0.03998, a
    delta of −0.00108 whose 95% paired-bootstrap CI is
    **[−0.00246, +0.00017]**. It straddles zero, so the honest verdict is
    "indistinguishable" and the rendered one was "beats the base-rate
    predictor". The same defect as the experiment report's "no difference
    detected yet": a verdict stated without the power to support it — on the
    instrument every keep/kill decision in this project reads.

    PAIRED (resamples per-row loss DIFFERENCES rather than the two Briers
    independently): both predictors are scored on the very same rows, and the
    correlation between them is most of the variance — resampling them apart
    would inflate the interval and hide a real win as readily as the point
    estimate manufactured a fake one.

    Deterministic (fixed seeds, no shuffle-on-read) so two fits on identical
    data produce the identical interval: an audit number that moves on its
    own is not one. Returns ``(delta, lo, hi)``, or None when there are too
    few rows to cross-validate.
    """
    rows = [(float(c), float(o)) for c, o in pairs]
    n = len(rows)
    if n < 2 * k:
        return None
    base = sum(o for _, o in rows) / n
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    delta_rows = [0.0] * n
    for f in range(k):
        test = set(idx[f::k])
        train = [rows[i] for i in range(n) if i not in test]
        if len(train) < 2:
            return None
        a, b = _fit_platt(train)
        for i in test:
            c, o = rows[i]
            delta_rows[i] = (apply_platt(c, a, b) - o) ** 2 - (base - o) ** 2
    delta = sum(delta_rows) / n
    rng = random.Random(seed + 1)
    means = []
    for _ in range(boots):
        acc = 0.0
        for _ in range(n):
            acc += delta_rows[rng.randrange(n)]
        means.append(acc / n)
    means.sort()
    lo = means[int((alpha / 2.0) * len(means))]
    hi = means[min(len(means) - 1, int((1.0 - alpha / 2.0) * len(means)))]
    return delta, lo, hi


#: Bootstrap replicates for the rank CI. Fewer than `_cv_delta_ci`'s 2000
#: because AUC needs a re-sort per replicate; 800 is stable to ~0.005 here.
_AUC_BOOTS = 800
#: Below this many rows in a population, its rank direction is not evidence.
_RANK_MIN_N = 40


def _auc(pairs) -> Optional[float]:
    """Tie-aware Mann-Whitney AUC of composite vs BINARISED outcome.

    ⚠ A DIFFERENT QUESTION FROM `_cv_delta_ci`, and the reason both exist.
    Brier-against-the-base-rate asks "is this score a good PROBABILITY"; at
    the live base rate (0.855, 5.6% negatives) the achievable improvement is
    a fraction of a percent and the comparison is nearly powerless. AUC asks
    "does this score ORDER turns correctly", which is the question every
    threshold consumer actually depends on. Measured 2026-08-31 they gave
    opposite readings on the same rows: delta CI [-0.00195, +0.00021]
    (indistinguishable) while user-origin AUC was 0.662 [0.580, 0.738].
    Reporting only the first read as "no signal" when the truth was "no
    probability calibration".

    ``None`` when one class is absent — AUC is undefined then, and returning
    0.5 would manufacture a measured tie out of no measurement at all.
    """
    rows = [(float(c), 1 if float(o) >= 0.5 else 0) for c, o in pairs]
    n_pos = sum(y for _, y in rows)
    n_neg = len(rows) - n_pos
    if not n_pos or not n_neg:
        return None
    rows.sort(key=lambda r: r[0])
    ranks = [0.0] * len(rows)
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1][0] == rows[i][0]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = mid
        i = j + 1
    s_pos = sum(r for r, (_, y) in zip(ranks, rows) if y)
    return (s_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auc_ci(pairs, *, boots: int = _AUC_BOOTS, seed: int = 17,
            alpha: float = 0.05):
    """``(auc, lo, hi)`` by nonparametric bootstrap, or ``None``.

    Deterministic (fixed seed), for the same reason `_cv_delta_ci` is: an
    audit number that moves on its own is not one. Replicates that lose a
    class are dropped rather than scored — with 71 negatives in 1274 rows
    that is rare, but counting them as 0.5 would pull every interval toward
    a tie, which is the direction that manufactures the "no signal" reading
    this function exists to check.
    """
    point = _auc(pairs)
    if point is None:
        return None
    rows = list(pairs)
    n = len(rows)
    if n < _RANK_MIN_N:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(boots):
        a = _auc([rows[rng.randrange(n)] for _ in range(n)])
        if a is not None:
            vals.append(a)
    if len(vals) < boots // 2:
        return None
    vals.sort()
    lo = vals[int((alpha / 2.0) * len(vals))]
    hi = vals[min(len(vals) - 1, int((1.0 - alpha / 2.0) * len(vals)))]
    return point, lo, hi


def _rank_verdict(auc_ci) -> str:
    """Does the score ORDER outcomes? Decided by the INTERVAL vs 0.5.

    Same discipline §4DZ imposed on the base-rate verdict, and for the same
    reason: a point estimate on the right side of chance is a margin, not a
    result. Reuses the four-word verdict vocabulary — for a rank statistic
    `BEATS_NO` means "orders them BACKWARDS", which is worse than chance in
    exactly the sense that word already carries.
    """
    if not auc_ci:
        return BEATS_UNKNOWN
    _point, lo, hi = auc_ci
    if lo > 0.5:
        return BEATS_YES
    if hi < 0.5:
        return BEATS_NO
    return BEATS_INDISTINGUISHABLE


#: Origin of rows admitted to the fit only as auxiliary data.
_AUX_ORIGIN = "bench"


@dataclass
class _Population:
    """The rows a fit consumes, and why the rest were dropped.

    ⚠ ONE BUILDER, THREE SURFACES. `fit`, `_load_epoch` and `stats` each
    assembled this chain by hand. Every metric an operator reads must
    describe the population the fit actually consumed — this module already
    carries two bugs from that rule being broken one filter at a time (§4L
    R2 NEW-4 averaged superseded duplicates; §4BF 1c reported over uncapped
    bench rows). A shared builder makes the next filter reach all three by
    construction instead of by remembering.
    """
    samples: List["CalibrationSample"]
    n_superseded: int = 0
    n_bench_capped: int = 0
    n_bench_misaligned: int = 0
    bench_rank: Optional[Tuple[float, float, float]] = None
    bench_verdict: str = BEATS_UNKNOWN


def _apply_bench_direction_gate(samples, params):
    """Auxiliary rows may join the fit only while they RANK the same way.

    ⚠ THE MASS CAP GUARDS THE WRONG HALF. `_apply_bench_mass_cap` bounds how
    MANY bench rows enter, on the premise that they are otherwise
    exchangeable with real turns. Measured 2026-08-31 on the live store they
    are not: 252 bench rows scored AUC 0.273 [0.116, 0.460] — confidently
    BACKWARDS — against 0.662 [0.580, 0.738] for the 1022 real ones, and fit
    alone they produce a Platt slope of -12.8. They were under the cap
    (252 <= 1022) so every one was admitted, dragging the pooled AUC to
    0.577. This is the same lesson `CURRENT_EPOCH` encodes one axis over:
    no fit beats a fit on incomparable rows. The cap tests the volume of the
    assumption; this tests the assumption.

    CONSERVATIVE — excludes ONLY on a confident inversion (`BEATS_NO`, i.e.
    the whole interval below chance). An indistinguishable or merely weak
    auxiliary population is still admitted: dropping real data on noise is
    the worse error, and the operator decision that admitted bench at all
    (§4BF 1c) stands until there is evidence against its premise.

    Scored with the PREVIOUSLY fitted weights — the instrument currently in
    production, which is exactly what the question is about, and which keeps
    the decision independent of the fit it feeds. With no params yet (cold
    start) there is no evidence, so the rows are admitted and the verdict is
    `unknown`. The decision is recorded in the params, so a corpus that
    oscillates across refits is visible rather than silent.
    """
    bench = [s for s in samples
             if getattr(s, "origin", "user") == _AUX_ORIGIN]
    # ⚠ NO SIZE CHECK HERE. `_auc_ci` owns the `_RANK_MIN_N` floor and
    # returns None below it; a second copy was unobservable (a mutation
    # sweep proved it equivalent) and would be the usual place for the two
    # to drift apart. The `params is None` check IS also equivalent — the
    # guard below catches the AttributeError — but a cold start is a NORMAL
    # state, and routing an expected condition through an exception handler
    # is how a real fault later gets mistaken for one.
    if params is None:
        return list(samples), BEATS_UNKNOWN, None
    try:
        pairs = [(_composite_for(s, params.w_entropy,
                                 params.lambda_uncertainty, params.w_effort),
                  s.outcome) for s in bench]
    except Exception:  # noqa: BLE001 — a gate must never break the fit
        return list(samples), BEATS_UNKNOWN, None
    ci = _auc_ci(pairs)
    verdict = _rank_verdict(ci)
    if verdict != BEATS_NO:
        return list(samples), verdict, ci
    return ([s for s in samples
             if getattr(s, "origin", "user") != _AUX_ORIGIN], verdict, ci)


def _fit_population(epoch_samples, params=None) -> _Population:
    """THE population — the single authority every surface must ask.

    Order matters: supersession resolves duplicate labels for one request,
    the mass cap bounds auxiliary volume, the direction gate tests whether
    that auxiliary volume belongs at all, and the pressure migration makes
    a mid-epoch feature-semantics change uniform.
    """
    resolved = _resolve_superseded(list(epoch_samples))
    capped = _apply_bench_mass_cap(resolved)
    gated, verdict, ci = _apply_bench_direction_gate(capped, params)
    return _Population(
        samples=_migrate_leaked_pressure(gated),
        n_superseded=len(epoch_samples) - len(resolved),
        n_bench_capped=len(resolved) - len(capped),
        n_bench_misaligned=len(capped) - len(gated),
        bench_rank=ci,
        bench_verdict=verdict,
    )


def _feature_contribution(attr: str, composites, *, rebuild) -> Optional[float]:
    """Held-out Brier DELTA from dropping ``attr``: >0 means the feature helps.

    ⚠ WHY THIS EXISTS, and why it is not `_separation_sigmas`. That gate is a
    difference-of-means test between outcome classes, and the live store has
    **18 negatives in 694 rows**. At that balance it can only resolve a gap of
    ~0.60 SD of the feature, so it reports `[DEAD]` for anything subtler —
    including features that demonstrably work.

    Measured 2026-08-10: `competence_component` scores 0.05σ and is printed
    `[DEAD]`, yet ABLATING it costs 0.021 AUC. The mean-difference verdict and
    the ablation disagree, the ablation is the one that answers the question
    actually being asked ("does the composite get worse without it?"), and
    nobody had ever run it. An under-powered test reported as a verdict is
    worse than no test, because it licenses removing something that works.
    """
    try:
        without = rebuild(attr)
        if without is None:
            return None
        b_with = _cv_brier(composites)
        b_without = _cv_brier(without)
        if b_with is None or b_without is None:
            return None
        return round(b_without - b_with, 6)
    except Exception:  # noqa: BLE001 — telemetry must never break a fit
        return None


def _composite_for(sample, w_e: float, lam: float, w_eff: float = 0.0) -> float:
    """Recompute a sample's composite under candidate (w_e, w_eff, λ).

    Mirrors :meth:`CompositeConfidence.score` EXACTLY, including the
    missing-feature renormalisation: only components the sample actually
    OBSERVED contribute, and the blend is divided by their weights. A
    sample with neither entropy nor effort is scored on competence alone
    rather than blended with neutral stand-ins. Fitting and scoring must
    agree — if the fit blended a stand-in while the scorer didn't (or vice
    versa), the fitted weights would be optimal for a formula the agent
    never actually evaluates.
    """
    w_c = max(0.0, 1.0 - w_e - w_eff)
    parts = [(w_c, sample.competence_component)]
    if sample.entropy_observed:
        parts.append((w_e, sample.entropy_component))
    if getattr(sample, "effort_observed", False):
        parts.append((w_eff, sample.effort_component))
    tot = sum(w for w, _ in parts)
    c = (sum(w * v for w, v in parts) / tot) if tot > 0 else sample.competence_component
    return _clamp01(c * (1.0 - lam * sample.uncertainty_pressure))


# ──────────────────────────────────────────────────────────────────────
# dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationSample:
    """One (confidence, outcome) pair.

    The objective components (``entropy_component`` = ``1 − e`` and
    ``competence_component`` = shrunk ``p``) and the verbalised
    ``uncertainty_pressure`` are stored separately from the recorded
    ``composite`` so :meth:`CalibrationTracker.fit` can recompute the
    composite for *candidate* weights/λ without re-running the agent.
    """

    composite: float
    entropy_component: float
    competence_component: float
    uncertainty_pressure: float
    outcome: float  # 1.0 = turn succeeded, 0.0 = turn failed
    domain: str = ""
    ts: str = ""
    # False when no token logprobs were observed for the turn and
    # ``entropy_component`` is the neutral 0.5 stand-in rather than a
    # measurement. Such samples are EXCLUDED from the entropy-weight fit —
    # see :meth:`CalibrationTracker.fit`. Defaults to False so a legacy
    # record (which was always the fabricated neutral) is treated as
    # unobserved, which is exactly what it was.
    entropy_observed: bool = False
    # Turn-shape (effort) component and whether it was measured. This is the
    # first per-TURN input; see `confidence.effort_component`.
    effort_component: float = 0.5
    effort_observed: bool = False
    # PROVENANCE of the label. Without it, mixing signal tiers is
    # irreversible: you can never audit which tier is noisy, nor drop one,
    # without discarding the whole corpus. Every future source (implicit
    # rephrase-negatives, generated probes) must carry its own tag so the
    # mix stays visible and separable.
    #   "turn"            — the graded end-of-turn label (a PROXY)
    #   "user_correction" — the user said the answer was wrong (ground truth)
    #   "failure_report"  — the user pasted breakage next turn (ground truth)
    #   "task_reopened"   — a task this turn closed DONE was later reopened
    #                       (delayed ground truth; §4E Tier 3)
    # Legacy records predate the field and were all end-of-turn labels.
    source: str = "turn"
    # The request id of the turn this sample scores. "" on legacy records
    # and non-turn sources. The §4E Tier-3 join key: a task reopen carries
    # the closing turn's req_id and finds this sample by it.
    req_id: str = ""
    # Which CORPUS EPOCH this row belongs to — see `CURRENT_EPOCH`. The fit
    # reads one epoch only, so a row from a different label scheme or
    # feature set can never be blended with the current one. Untagged legacy
    # rows get theirs derived from `ts` on read (`epoch_for_ts`), never the
    # current value: silently promoting them is the exact contamination the
    # field exists to prevent.
    epoch: str = ""
    # Which POPULATION the turn belongs to (§4BF Track 1c): "user" = a real
    # turn, "bench" = an idle bench-bank solve. Legacy rows predate the
    # field and were all real turns (sim was write-gated), so the default
    # is correct on read. Kept separate from `source` — source ranks LABEL
    # AUTHORITY within one request, origin separates populations across
    # requests (the fit blends them under an equal-mass cap; see `fit`).
    origin: str = "user"


@dataclass
class FittedParams:
    """Result of a fit, persisted to ``calibration_params.json`` and
    loaded back into :class:`CompositeConfidence`."""

    w_entropy: float
    w_competence: float
    threshold: float
    lambda_uncertainty: float
    brier: float
    n_samples: int
    fitted_at: str
    schema: str = SCHEMA_VERSION
    # Platt recalibration of the blended score into a PROBABILITY:
    #   p = 1 / (1 + exp(-(platt_a · composite + platt_b)))
    #
    # The weight search decides how to COMBINE the signals but has no way to
    # set their overall level, so the raw composite ranked turns correctly
    # while being systematically mis-scaled. Measured on 1208 live samples:
    # AUC 0.679 (real discrimination) but Brier 0.060 against 0.037 for
    # simply always predicting the base rate — i.e. the number was worse
    # than useless AS A PROBABILITY. Fitting these two parameters takes it
    # to 0.031, better than the base rate. Identity defaults (a=1, b=0) mean
    # an unfitted/legacy params file behaves exactly as before.
    platt_a: float = 1.0
    platt_b: float = 0.0
    # Weight on the turn-EFFORT component (see `confidence.effort_component`).
    # 0 until enough samples measure turn shape, same evidence rule as
    # w_entropy. w_competence is the remainder: 1 - w_entropy - w_effort.
    w_effort: float = 0.0
    n_effort_observed: int = 0
    # Brier of the RAW composite and of the base-rate predictor, kept so the
    # fit can be audited against the only baseline that matters. A model
    # that cannot beat `brier_base_rate` is not adding information.
    brier_raw: float = -1.0
    brier_base_rate: float = -1.0
    # 95% paired-bootstrap CI for (brier_cv - brier_base_rate). None when not
    # computed (too few rows, or a params file written before this existed).
    # THE SIGN OF THE INTERVAL, not of the point estimate, is what licenses a
    # "beats the base rate" claim — see `_cv_delta_ci`.
    brier_cv_delta_lo: Optional[float] = None
    brier_cv_delta_hi: Optional[float] = None
    # How many of ``n_samples`` carried REAL observed logprob entropy. When
    # this is below `_MIN_ENTROPY_SAMPLES` the fit pins ``w_entropy`` to 0
    # deliberately — the number makes that visible instead of leaving an
    # operator to wonder why the weight never moves.
    n_entropy_observed: int = 0
    # What happened to the Platt probability map in this fit:
    #   "applied"            — map adopted;
    #   "rejected_inverted"  — slope <= 0 (composite anti-correlated),
    #                          identity map kept;
    #   "rejected_step"      — slope > _MAX_SLOPE backstop, identity kept;
    #   "discarded_worse"    — calibrated Brier no better than raw.
    # The refit summary line must surface this — logging `refit=ok` while
    # the map was rejected reads as a healthy calibration when the score
    # is in fact predicting nothing (2026-07-29 log audit).
    map_status: str = "applied"
    #: Does the model beat always-predicting-the-base-rate? Decided ONCE by
    #: `_base_rate_verdict` from the CROSS-VALIDATED delta INTERVAL, not the
    #: point estimate. Distinct from `map_status`, which answers a different
    #: question — whether the Platt MAP was applied (it can be correctly
    #: applied, because it beats the raw composite, while the underlying
    #: model is indistinguishable from a constant; that is the live state).
    #: Anything in `NOT_INFORMATIVE` means no consumer may treat the score
    #: or its threshold as carrying information.
    beats_base_rate: str = BEATS_UNKNOWN
    # Which corpus epoch produced this fit, and how many rows were excluded
    # as belonging to older ones. Without both numbers an operator reading
    # `n_samples=541` against a 1709-row file has no way to tell a healthy
    # epoch filter from a truncated read. Empty default so a params file
    # written before epochs existed loads unchanged.
    epoch: str = ""
    n_excluded_other_epochs: int = 0
    # ── HONEST PERFORMANCE (audit 2026-08-10) ────────────────────────────
    # `brier` above is IN-SAMPLE: a 2-parameter Platt map scored on the rows
    # it was fitted to. It was printed as the headline beside
    # `brier_base_rate`, a comparison only one side of which paid for its
    # parameters. On the live store that overstated the gain 3x (3.0%
    # in-sample vs 1.0% cross-validated). -1.0 = not computed (too few rows,
    # or a params file written before this existed).
    brier_cv: float = -1.0
    # THE NUMBER EVERY OTHER VERDICT RESTS ON. The live store carries 18
    # negatives in 694 rows (2.6%), and the separation gate, the weights and
    # the Brier comparison are all estimated from those 18. Recorded so a
    # reader can see the power behind a verdict instead of inferring it.
    n_negative: int = 0
    # Held-out Brier DELTA per feature (>0 = dropping it makes things worse,
    # i.e. it helps). The mean-difference gate calls `competence_component`
    # DEAD at 0.05sigma while ablation says removing it costs 0.021 AUC;
    # this records the measurement that actually answers the question.
    feature_contrib: Optional[Dict[str, float]] = None
    # ── DISCRIMINATION, the question Brier cannot ask (2026-08-31) ────────
    # `beats_base_rate` answers "is this a good PROBABILITY". At a base rate
    # of 0.855 with 5.6% negatives that comparison is nearly powerless, and
    # reporting only it read as "no signal" when the truth was "no
    # probability calibration": the same rows gave delta CI
    # [-0.00195, +0.00021] (indistinguishable) and user-origin AUC 0.662
    # [0.580, 0.738] (real ordering). A ranking use — flagging the lowest-
    # confidence turns — is licensed by `ranks_outcomes`, never by
    # `beats_base_rate`; a probability claim needs the latter. -1.0 / unknown
    # = not recorded, never a measured value.
    auc: float = -1.0
    auc_ci_lo: Optional[float] = None
    auc_ci_hi: Optional[float] = None
    #: INTERVAL-driven, like `beats_base_rate` (§4DZ): `yes` = whole CI above
    #: chance, `no` = whole CI BELOW it (ordered backwards), otherwise
    #: indistinguishable. In `NOT_INFORMATIVE` ⇒ no ranking use either.
    ranks_outcomes: str = BEATS_UNKNOWN
    #: Did the auxiliary (bench) population rank the same way, and how many
    #: rows the direction gate dropped when it did not.
    bench_verdict: str = BEATS_UNKNOWN
    n_bench_misaligned: int = 0


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    mean_outcome: float


# ──────────────────────────────────────────────────────────────────────
# tracker
# ──────────────────────────────────────────────────────────────────────

class CalibrationTracker:
    """Append-only calibration log + grid-search refit.

    Constructed once per agent (in ``main.lifespan``) and hung on
    ``context.calibration_tracker``. Writes acquire a lock; reads parse
    the tail of the JSONL. Everything is best-effort: a disk error
    leaves the in-flight turn untouched.
    """

    HISTORY_NAME = "calibration.jsonl"
    PARAMS_NAME = "calibration_params.json"

    # Defaults mirror the prm.trainer bail floors — below these a fit is
    # noise. Both classes (success AND failure) must also be present.
    DEFAULT_MIN_SAMPLES = 40
    DEFAULT_MAX_HISTORY = 4000

    def __init__(
        self,
        calib_dir: Path,
        *,
        min_samples_for_fit: int = DEFAULT_MIN_SAMPLES,
        max_history: int = DEFAULT_MAX_HISTORY,
    ):
        self.dir = Path(calib_dir)
        self.history_path = self.dir / self.HISTORY_NAME
        self.params_path = self.dir / self.PARAMS_NAME
        self.min_samples_for_fit = max(1, int(min_samples_for_fit))
        self.max_history = max(1, int(max_history))
        self._lock = threading.RLock()

    # ----------------------------------------------------------- recording

    def record(
        self,
        *,
        composite: float,
        entropy_component: float,
        competence_component: float,
        outcome: float,
        uncertainty_pressure: float = 0.0,
        domain: str = "",
        entropy_observed: bool = False,
        effort_component: float = 0.5,
        effort_observed: bool = False,
        source: str = "turn",
        req_id: str = "",
        epoch: str = "",
        origin: str = "user",
    ) -> bool:
        """Append one (confidence, outcome) pair. Never raises; returns
        True when the sample actually reached disk (callers that report
        success to the operator must not claim an unwritten row).

        ``entropy_observed`` records whether ``entropy_component`` came from
        real token logprobs. Pass it through faithfully — a fabricated
        neutral marked as observed re-poisons the entropy fit.

        ``epoch`` defaults to :data:`CURRENT_EPOCH`. Pass it explicitly ONLY
        to re-label an existing sample (the retro-negative tiers), where the
        stored features belong to whichever epoch produced them.
        """
        try:
            sample = CalibrationSample(
                composite=_clamp01(composite),
                entropy_component=_clamp01(entropy_component),
                competence_component=_clamp01(competence_component),
                uncertainty_pressure=_clamp01(uncertainty_pressure),
                # Clamped, NOT binarised. The old `1.0 if >= 0.5 else 0.0`
                # crushed every graded label back to two values, which is
                # exactly the constant-column problem the grading exists to
                # remove. Binary inputs are unaffected (0.0/1.0 clamp to
                # themselves).
                outcome=_clamp01(outcome),
                domain=str(domain or ""),
                ts=_utcnow_iso(),
                entropy_observed=bool(entropy_observed),
                effort_component=_clamp01(effort_component),
                effort_observed=bool(effort_observed),
                source=str(source or "turn"),
                req_id=str(req_id or ""),
                epoch=str(epoch or CURRENT_EPOCH),
                origin=str(origin or "user"),
            )
            with self._lock:
                self.dir.mkdir(parents=True, exist_ok=True)
                with self.history_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(sample)) + "\n")
            return True
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("CalibrationTracker.record failed: %s", exc)
            return False

    def record_late_verdict_correction(self, req_id: str,
                                       outcome: float) -> bool:
        """Re-label a turn's calibration sample after a LATE verifier
        verdict (§4L Lens-A MAJOR-1).

        The sample is written once at finalize with only the INLINE
        verdict — but with critic nodes live, most verdicts land late
        via `_record_late_verdict`, which backfilled the corpus,
        selfhood, lessons and the operator line while the calibration
        store kept the optimistic inline grade: measured live, 47 late
        REFUTED corrections against 7 stored hard negatives, with 6 of 9
        joinable late-refuted turns still sitting in the fit as
        positives. Negatives are the scarce class (18/678) — this
        starved and mislabelled exactly the class the fit needs most.

        Same no-leakage contract as the retro-negative: the stored
        FEATURES are reused untouched, only the label differs, and
        `_resolve_superseded` (source rank verifier_late > turn) makes
        this a re-label, not a counter-weight. Idempotent per request.
        """
        try:
            req_id = str(req_id or "")
            if not req_id:
                return False
            with self._lock:
                samples = self._load_samples()
                base = None
                for s in samples:
                    if s.req_id != req_id:
                        continue
                    if s.source == "verifier_late":
                        return False      # already corrected
                    if s.source == "turn":
                        base = s          # last write wins
                if base is None:
                    return False          # no join — skip, never default
                if abs(base.outcome - outcome) < 1e-9:
                    return False          # nothing to correct
                return self.record(
                    composite=base.composite,
                    entropy_component=base.entropy_component,
                    competence_component=base.competence_component,
                    uncertainty_pressure=base.uncertainty_pressure,
                    outcome=_clamp01(outcome),
                    domain=base.domain,
                    entropy_observed=base.entropy_observed,
                    effort_component=base.effort_component,
                    effort_observed=base.effort_observed,
                    source="verifier_late",
                    req_id=req_id,
                    # Inherit the turn's epoch — same rationale as the
                    # retro-negative below: reused features must not
                    # smuggle an older feature set into the live fit.
                    epoch=base.epoch,
                    # A re-label never migrates populations.
                    origin=base.origin,
                )
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("record_late_verdict_correction failed: %s", exc)
            return False

    def record_task_reopened_negative(self, closed_req_id: str) -> bool:
        """§4E Tier 3: retroactive negative for the turn that closed a task
        later REOPENED. Finds that turn's ``source="turn"`` sample by
        ``req_id`` and re-records ITS OWN stored components (no-leakage:
        the feature side is untouched, only the label differs) at
        ``_TASK_REOPENED_GRADE`` with ``source="task_reopened"``.

        Returns True when a retro sample was written. Skips (False) when:
        no join (legacy rows / stampless close), the closing turn already
        carries a task_reopened retro row (idempotent — one retro negative
        per closing turn, however often statuses cycle afterwards), or the
        closing turn's own label was already negative (mirrors Tier 2's
        no-double-counting rule). Never raises.
        """
        try:
            closed_req_id = str(closed_req_id or "")
            if not closed_req_id:
                return False
            with self._lock:
                samples = self._load_samples()
                base = None
                for s in samples:
                    if s.req_id != closed_req_id:
                        continue
                    if s.source == "task_reopened":
                        return False  # already retro-labeled
                    if s.source == "turn":
                        base = s  # last write wins
                if base is None:
                    return False
                if base.outcome < 0.5:
                    return False  # the turn is already a negative
                # record() reports whether the row reached disk — pass that
                # through so the operator log never claims an unwritten
                # retro-negative (and a failed write stays retryable).
                return self.record(
                    composite=base.composite,
                    entropy_component=base.entropy_component,
                    competence_component=base.competence_component,
                    uncertainty_pressure=base.uncertainty_pressure,
                    outcome=_TASK_REOPENED_GRADE,
                    domain=base.domain,
                    entropy_observed=base.entropy_observed,
                    effort_component=base.effort_component,
                    effort_observed=base.effort_observed,
                    source="task_reopened",
                    req_id=closed_req_id,
                    # Inherit the CLOSING turn's epoch. The retro row reuses
                    # that turn's stored features, so stamping it with the
                    # current epoch would smuggle an older feature set into
                    # the live fit — with a negative label attached, i.e. the
                    # most damaging row it could possibly contribute.
                    epoch=base.epoch,
                    origin=base.origin,
                )
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("record_task_reopened_negative failed: %s", exc)
            return False

    def record_bench_validator_verdict(self, req_id: str,
                                       passed: bool) -> bool:
        """§4BF Track 1c: re-label a BENCH solve's calibration sample with
        the bank oracle's mechanical verdict.

        A bench attempt records its ``source="turn"`` sample inside
        `handle_chat` like any other turn — but that label is the inline
        PROXY grade, written before the bank validator has run. The
        validator's exit code is ground truth for this population, so it
        re-labels at the ``bench_validator`` rank using the same
        no-leakage contract as every other retro tier: the stored
        FEATURES are reused untouched, only the label differs, and
        `_resolve_superseded` makes it a re-label, not a counter-weight.
        Idempotent per request; joins ``origin="bench"`` turn rows only
        (a real turn's req_id can never be re-labeled by a bench oracle).
        Never raises.
        """
        try:
            req_id = str(req_id or "")
            if not req_id:
                return False
            outcome = 1.0 if passed else 0.0
            with self._lock:
                samples = self._load_samples()
                base = None
                for s in samples:
                    if s.req_id != req_id:
                        continue
                    if s.source == "bench_validator":
                        return False      # already labeled
                    if s.source == "turn" and s.origin == "bench":
                        base = s          # last write wins
                if base is None:
                    return False          # no join — skip, never default
                if abs(base.outcome - outcome) < 1e-9:
                    return False          # inline grade already agreed
                return self.record(
                    composite=base.composite,
                    entropy_component=base.entropy_component,
                    competence_component=base.competence_component,
                    uncertainty_pressure=base.uncertainty_pressure,
                    outcome=outcome,
                    domain=base.domain,
                    entropy_observed=base.entropy_observed,
                    effort_component=base.effort_component,
                    effort_observed=base.effort_observed,
                    source="bench_validator",
                    req_id=req_id,
                    epoch=base.epoch,
                    origin=base.origin,
                )
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("record_bench_validator_verdict failed: %s", exc)
            return False

    # ----------------------------------------------------------- reading

    def _load_epoch(self, limit: Optional[int] = None,
                    epoch: Optional[str] = None) -> List[CalibrationSample]:
        """Samples from ONE epoch (the current one by default).

        Every metric the operator reads — Brier, ECE, the reliability table
        — must be scoped the same way the fit is, or the report describes a
        population the agent never fits. Mixing epochs here is what made the
        stored-composite AUC read 0.676 where the honest recomputed value on
        the current epoch is 0.530.
        """
        want = CURRENT_EPOCH if epoch is None else epoch
        scoped = [s for s in self._load_samples(limit=limit)
                  if s.epoch == want]
        # ⚠ Same RESOLUTION as the fit (§4L R2 NEW-4): metrics used to
        # average superseded duplicate labels (reported Brier 0.410 vs
        # 0.810 on the fit's resolved population) — the exact
        # "population the agent never fits" failure this docstring
        # warns about, one filter deeper. Same ORIGIN CAP as the fit too
        # (§4BF 1c) — the metrics must describe the blended population
        # the fit consumes, not a bench-flooded superset.
        return _fit_population(scoped, self.load_params()).samples

    def _load_samples(self, limit: Optional[int] = None) -> List[CalibrationSample]:
        if not self.history_path.exists():
            return []
        out: List[CalibrationSample] = []
        try:
            with self.history_path.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("CalibrationTracker load failed: %s", exc)
            return []
        if limit is not None:
            lines = lines[-limit:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out.append(
                    CalibrationSample(
                        composite=_clamp01(d.get("composite", 0.5)),
                        entropy_component=_clamp01(d.get("entropy_component", 0.5)),
                        competence_component=_clamp01(
                            d.get("competence_component", 0.5)
                        ),
                        uncertainty_pressure=_clamp01(
                            d.get("uncertainty_pressure", 0.0)
                        ),
                        # Preserve graded values on read too (legacy 0.0/1.0
                        # records clamp to themselves).
                        outcome=_clamp01(d.get("outcome", 0.0)),
                        domain=str(d.get("domain", "")),
                        ts=str(d.get("ts", "")),
                        # Absent on legacy records, which were ALWAYS the
                        # fabricated neutral — default False is correct.
                        entropy_observed=bool(d.get("entropy_observed", False)),
                        effort_component=_clamp01(d.get("effort_component", 0.5)),
                        effort_observed=bool(d.get("effort_observed", False)),
                        source=str(d.get("source") or "turn"),
                        req_id=str(d.get("req_id") or ""),
                        # Untagged legacy rows get their epoch DERIVED, not
                        # defaulted to the current one — see `CURRENT_EPOCH`.
                        epoch=str(d.get("epoch") or epoch_for_ts(d.get("ts"))),
                        # Legacy rows were all real turns (sim write-gated).
                        origin=str(d.get("origin") or "user"),
                    )
                )
            except Exception:
                # Skip malformed lines without poisoning the read.
                continue
        return out

    def sample_count(self) -> int:
        return len(self._load_samples())

    def recent_samples(self, limit: int = 500, *,
                       exclude_origin: str = "") -> List[CalibrationSample]:
        """The most recent ``limit`` samples, ALL epochs.

        Public read for consumers that only need the per-turn scores keyed by
        ``req_id`` (offline triage ranking, `core.risk`) rather than a fit.
        Epoch is deliberately NOT filtered here: ranking compares turns
        against each other within one recent window, and dropping older-epoch
        rows would silently shrink that window — whereas a FIT must never mix
        epochs, which is why `fit` uses `_load_epoch` instead.

        ``exclude_origin`` (§4BF 1c, R4 review): filters BEFORE the window is
        taken — a caller-side filter after the tail is a no-op against
        dilution, because the rows bench pushed out of the newest-N stay
        pushed out. Real-turn consumers (reflection triage) pass "bench" so
        their window holds ``limit`` REAL rows regardless of idle volume.
        """
        try:
            n = max(1, int(limit))
        except (TypeError, ValueError):
            n = 500
        if not exclude_origin:
            return self._load_samples(limit=n)
        rows = [s for s in self._load_samples()
                if getattr(s, "origin", "user") != exclude_origin]
        return rows[-n:]

    # ----------------------------------------------------------- metrics

    def brier_score(self, *, window: Optional[int] = None) -> Optional[float]:
        """Rolling Brier score ``mean((C − outcome)²)`` over the recent
        ``window`` samples of the CURRENT EPOCH (all by default). ``None``
        when no data."""
        samples = self._load_epoch(limit=window)
        if not samples:
            return None
        return sum((s.composite - s.outcome) ** 2 for s in samples) / len(samples)

    def reliability_table(
        self, *, bins: int = 10, window: Optional[int] = None
    ) -> List[ReliabilityBin]:
        """10-bin reliability table: for each confidence band, the mean
        predicted confidence vs the mean realized outcome. A perfectly
        calibrated agent has ``mean_confidence ≈ mean_outcome`` in every
        populated bin. Current epoch only — the stored composite column is
        not comparable across a formula change."""
        bins = max(1, int(bins))
        samples = self._load_epoch(limit=window)
        table: List[ReliabilityBin] = []
        for i in range(bins):
            lo = i / bins
            hi = (i + 1) / bins
            # Last bin is closed on the right so C == 1.0 lands somewhere.
            in_bin = [
                s for s in samples
                if (s.composite >= lo and (s.composite < hi or (i == bins - 1 and s.composite <= hi)))
            ]
            if in_bin:
                mc = sum(s.composite for s in in_bin) / len(in_bin)
                mo = sum(s.outcome for s in in_bin) / len(in_bin)
            else:
                mc = mo = 0.0
            table.append(ReliabilityBin(lo, hi, len(in_bin), mc, mo))
        return table

    def ece(self, *, bins: int = 10, window: Optional[int] = None) -> Optional[float]:
        """Expected Calibration Error: sample-weighted mean absolute gap
        between confidence and outcome across reliability bins. ``None``
        when no data."""
        table = self.reliability_table(bins=bins, window=window)
        total = sum(b.count for b in table)
        if total <= 0:
            return None
        return sum(
            (b.count / total) * abs(b.mean_confidence - b.mean_outcome)
            for b in table if b.count
        )

    # ----------------------------------------------------------- fitting

    def fit(self, *, min_samples: Optional[int] = None) -> Optional[FittedParams]:
        """Grid-search refit of (weights, λ, τ) minimising Brier.

        Reads the CURRENT EPOCH only (see :data:`CURRENT_EPOCH`) — rows from
        an older label scheme or feature set are not comparable and pooling
        them inverts the fit.

        Returns the :class:`FittedParams` on success (and persists them),
        or ``None`` with a logged ``bail_reason`` when the data is too
        thin or single-class. No params file is written on a bail — the
        previous fit (or the hardcoded defaults) stays in force.
        """
        floor = min_samples if min_samples is not None else self.min_samples_for_fit
        # ⚠ Accepted residual (§4BF 1c R3/R4): max_history is a raw LINE
        # tail, origin-blind — sustained bench volume slowly shrinks the
        # real mass inside this window (the equal-mass cap protects the
        # RATIO, not the absolute real count). Latent at today's corpus
        # (~1.8k rows vs 4k window); revisit if bench cadence grows.
        all_samples = self._load_samples(limit=self.max_history)
        epoch_samples = [s for s in all_samples if s.epoch == CURRENT_EPOCH]
        _pop = _fit_population(epoch_samples, self.load_params())
        samples = _pop.samples
        # ⚠ FOUR different exclusions, counted apart (§4L R2 NEW-5, one
        # richer for 1c, one for the direction gate): epoch scope,
        # source-rank supersession, the bench mass cap, and bench rows whose
        # rank direction disqualifies them. Folding cap-drops into
        # "superseded" would make a bench-heavy night read as thousands of
        # duplicate requests.
        n_excluded = len(all_samples) - len(epoch_samples)
        n_superseded = _pop.n_superseded
        n_bench_capped = _pop.n_bench_capped
        if _pop.n_bench_misaligned:
            _br = _pop.bench_rank or (float("nan"),) * 3
            logger.warning(
                "calibration: %d bench row(s) EXCLUDED from the fit — they "
                "rank outcomes backwards (AUC %.3f [%.3f, %.3f], whole "
                "interval below chance), so pooling them with real turns "
                "fits incomparable rows. See `bench_verdict` on the stored "
                "params.", _pop.n_bench_misaligned, _br[0], _br[1], _br[2])
        if n_superseded:
            logger.debug(
                "calibration: %d duplicate-request row(s) resolved by "
                "source-rank supersession before the fit", n_superseded)
        if n_bench_capped:
            logger.debug(
                "calibration: %d bench row(s) dropped by the equal-mass "
                "cap before the fit (§4BF 1c)", n_bench_capped)
        if len(samples) < floor:
            logger.debug(
                "calibration fit bail: %d samples in epoch %s < floor %d "
                "(%d rows excluded as older epochs)",
                len(samples), CURRENT_EPOCH, floor, n_excluded,
            )
            return None
        if n_excluded:
            logger.debug(
                "calibration: fitting on %d rows of epoch %s; %d older-epoch "
                "rows excluded as incomparable (different label scheme or "
                "feature set)", len(samples), CURRENT_EPOCH, n_excluded,
            )
        n_pos = sum(1 for s in samples if s.outcome >= 0.5)
        n_neg = len(samples) - n_pos
        # Bail on "no information in the labels", measured as variance rather
        # than as "both binary classes present". For 0/1 labels the two tests
        # are identical (variance is 0 exactly when one class is missing), so
        # binary corpora behave as before — but a GRADED corpus can carry
        # plenty of signal while every sample sits above 0.5, and the old
        # test would have refused to fit it at all.
        if _outcome_variance(samples) < 1e-9:
            logger.debug(
                "calibration fit bail: outcome column has no variance "
                "(pos=%d neg=%d)", n_pos, n_neg,
            )
            return None

        # The entropy weight may only move when enough samples carried a
        # REAL logprob observation. Previously every sample contributed a
        # flat, fabricated 0.5 to the entropy column, which is not merely
        # noisy but structurally fatal: with zero variance there, any
        # w_e > 0 could only drag composites toward 0.5, so the grid was
        # guaranteed to pick w_e = 0. That is exactly what was observed
        # live — 1200/1201 stored samples at 0.5 and w_entropy stuck at 0
        # (2026-07-27). Unobserved samples are no longer blended with the
        # stand-in at all (see `_composite_for`); they score on competence
        # alone, so a weight fit from the observed minority cannot degrade
        # them. Below the floor we pin w_e at 0 and log WHY rather than
        # fitting a weight on fiction.
        #
        # Enough observations is necessary but NOT sufficient: the column
        # must also separate the outcome classes (`_MIN_SEPARATION_SIGMAS`),
        # or the grid buys a weight for noise.
        observed = [s for s in samples if s.entropy_observed]
        obs_pos = sum(1 for s in observed if s.outcome >= 0.5)
        obs_neg = len(observed) - obs_pos
        ent_sigmas = _separation_sigmas(observed, "entropy_component")
        entropy_fittable = (
            len(observed) >= _MIN_ENTROPY_SAMPLES and obs_pos > 0 and obs_neg > 0
            and ent_sigmas >= _MIN_SEPARATION_SIGMAS
        )
        we_range = range(0, 11) if entropy_fittable else range(0, 1)

        # Same evidence rule for the turn-effort weight: it only moves once
        # enough samples actually MEASURED turn shape, with both outcome
        # classes represented. Legacy samples (recorded before the feature
        # existed) carry effort_observed=False and are excluded, exactly as
        # unobserved entropy is.
        eff_observed = [s for s in samples if getattr(s, "effort_observed", False)]
        eff_pos = sum(1 for s in eff_observed if s.outcome >= 0.5)
        eff_neg = len(eff_observed) - eff_pos
        eff_sigmas = _separation_sigmas(eff_observed, "effort_component")
        effort_fittable = (
            len(eff_observed) >= _MIN_EFFORT_SAMPLES and eff_pos > 0 and eff_neg > 0
            and eff_sigmas >= _MIN_SEPARATION_SIGMAS
        )
        weff_range = range(0, 11) if effort_fittable else range(0, 1)
        if not effort_fittable:
            logger.debug(
                "calibration: w_effort pinned to 0 — %d/%d samples measured "
                "turn shape (pos=%d neg=%d, need >=%d with both classes), "
                "separation %.2fσ (need >=%.1fσ)",
                len(eff_observed), len(samples), eff_pos, eff_neg,
                _MIN_EFFORT_SAMPLES, eff_sigmas, _MIN_SEPARATION_SIGMAS)
        if not entropy_fittable:
            logger.debug(
                "calibration: w_entropy pinned to 0 — %d/%d samples observed "
                "real logprob entropy (pos=%d neg=%d, need >=%d of each "
                "class), separation %.2fσ (need >=%.1fσ). Upstream refuses "
                "logprobs on tools+stream payloads, so most turns carry no "
                "token entropy.",
                len(observed), len(samples), obs_pos, obs_neg,
                _MIN_ENTROPY_SAMPLES, ent_sigmas, _MIN_SEPARATION_SIGMAS,
            )

        # Grid over entropy weight (→ competence = 1−w_e) and the
        # uncertainty penalty λ, over EVERY sample under the same per-sample
        # formula the scorer uses (`_composite_for`).
        #
        # The objective is the Brier the pipeline will ACTUALLY DELIVER —
        # i.e. after the Platt stage below, under the same acceptance rules.
        # Scoring candidates on the RAW Brier instead was an objective
        # mismatch with a real cost: on the live corpus the raw-best point
        # (0.054164, slope −0.077 → map REJECTED) beat the runner-up
        # (0.054204, slope +0.388 → map ACCEPTED) by 4e-5, so a coin flip in
        # the fourth decimal decided whether the agent got a probability map
        # at all. 321 of 396 grid points had a positive slope; optimising the
        # wrong quantity is what steered the search into the other 75.
        #
        # Selecting on the delivered Brier cannot do worse: a point whose map
        # is rejected simply scores its raw Brier, exactly as before.
        #
        # Ties are then broken toward the SIMPLEST model, because a strict
        # `<` over floats hands the decision to the last decimal place. Both
        # known failures of this search were exactly that: the inverted-map
        # selection above won by 4e-5, and on a corpus whose entropy column
        # is pure noise the grid buys w_entropy = 0.2 for a 4.9e-5 in-sample
        # gain. Measured against a real signal the margin is not close — the
        # live effort weight is worth 1.0e-2, some 200× the noise gain — so a
        # tolerance an order of magnitude above the noise floor and far below
        # the standard error of a Brier estimate at these sample sizes
        # (~5e-3) separates them cleanly without suppressing anything real.
        # This is the same principle as the `_MIN_*_SAMPLES` floors: a weight
        # must EARN its way off zero, and noise never earns anything.
        rows = []  # (delivered, we_i, weff_i, lam_i)
        for we_i in we_range:
            w_e = we_i / 10.0
            for weff_i in weff_range:
                w_eff = weff_i / 10.0
                if w_e + w_eff > 1.0:
                    continue          # competence would go negative
                for lam_i in range(0, 6):
                    lam = lam_i / 10.0
                    cand = [(_composite_for(s, w_e, lam, w_eff), s.outcome)
                            for s in samples]
                    b_raw = sum((c - y) ** 2 for c, y in cand) / len(cand)
                    a_c, b_c = _fit_platt(cand)
                    if 0.0 < a_c <= _MAX_SLOPE:
                        b_cal = sum(
                            (apply_platt(c, a_c, b_c) - y) ** 2
                            for c, y in cand) / len(cand)
                        # The stage below keeps the map only when it helps,
                        # so the delivered score is the better of the two.
                        delivered = min(b_cal, b_raw)
                    else:
                        delivered = b_raw   # map will be rejected
                    rows.append((delivered, we_i, weff_i, lam_i))

        assert rows
        _floor = min(r[0] for r in rows)
        # Complexity is compared on the INTEGER grid indices, not the divided
        # floats: 0.1 + 0.2 != 0.3 in binary, so summing the floats would rank
        # two equally simple points by a rounding artefact.
        brier, we_i, weff_i, lam_i = min(
            (r for r in rows if r[0] <= _floor + _BRIER_TIE_TOL),
            key=lambda r: (r[1] + r[2], r[3], r[0]),
        )
        w_e, w_eff, lam = we_i / 10.0, weff_i / 10.0, lam_i / 10.0
        w_c = max(0.0, 1.0 - w_e - w_eff)

        composites = [(_composite_for(s, w_e, lam, w_eff), s.outcome)
                      for s in samples]

        # Recalibration stage. The weight search decides how to COMBINE the
        # signals but cannot set their level — measured live, the composite
        # ranked turns well (AUC 0.679) while scoring Brier 0.060 against
        # 0.037 for a constant base-rate predictor. Fitting a two-parameter
        # Platt map turns that into 0.031. Class-balanced so the ~4% of
        # negatives are not drowned by the majority class.
        platt_a, platt_b = _fit_platt(composites)
        calibrated = [(apply_platt(c, platt_a, platt_b), y) for c, y in composites]

        base = sum(y for _, y in composites) / len(composites)
        brier_raw = sum((c - y) ** 2 for c, y in composites) / len(composites)
        brier_base = sum((base - y) ** 2 for _, y in composites) / len(composites)
        brier_cal = sum((p - y) ** 2 for p, y in calibrated) / len(calibrated)

        # Adopt the mapping only if it is both BETTER and SAFE.
        #
        # A non-positive slope is the important rejection. Platt is monotonic
        # in `a`, so `a <= 0` inverts the ranking — the agent would report
        # LOWER confidence on turns the score rates higher. The optimiser
        # chooses that whenever the composite is anti-correlated with
        # success, which is exactly what the live corpus shows (leak-free
        # AUC 0.473, i.e. no discrimination). In that state the honest fit is
        # "this score predicts nothing", and the right response is to leave
        # the raw behaviour alone and say so — not to ship an inverted or
        # flattened confidence. A near-zero slope is rejected for the same
        # reason: it collapses every turn onto the base rate, which scores a
        # great Brier while making `below_threshold` permanently inert.
        # Reject only what is actually unsafe.
        #
        # INVERSION (slope <= 0): the map would report lower confidence on
        # turns the score rates higher. The optimiser picks this whenever the
        # composite is anti-correlated with success, which is what the live
        # corpus shows (leak-free AUC 0.473). Calibrating noise is not an
        # improvement — leave the raw scale alone and say so.
        #
        # DIVERGENCE (slope enormous): a separable batch drives the fit to a
        # near-step function, which reports ~0 or ~1 for every turn and makes
        # `below_threshold` a hair-trigger on one competence value. The L2
        # penalty above bounds this in normal use; the cap is a backstop.
        #
        # A SMALL POSITIVE slope is NOT rejected. An earlier version demanded
        # slope >= 0.5 on the theory that a flat map would leave the gate
        # inert — that reasoning was wrong. Platt is strictly monotone for
        # a > 0 and the threshold is refit on the SAME scale, so the
        # below-threshold decision set is bit-identical before and after; the
        # map changes only calibration quality. Rejecting a = 0.30 discarded
        # a measured 0.069 Brier improvement and changed no decision at all.
        # Slope also cannot be judged without the composite's spread: a = 3.0
        # over a 0.02-wide range moves probabilities less than a = 0.3 over
        # the full unit interval.
        # ⚠ THE EVIDENCE IS COMPUTED BEFORE THE DECISION THAT USES IT.
        #
        # `brier_cv` and the delta CI used to be computed ~45 lines BELOW
        # this block, so the map decision — and the base-rate warning under
        # it — could only ever consult the IN-SAMPLE point estimate. The
        # decision literally happened before its evidence existed. Measured
        # 2026-08-30 on the live store: in-sample 0.046703 vs base rate
        # 0.047776 reads as a win by 0.0011, `map_status` lands on
        # "applied", and no warning fires — while the cross-validated
        # interval is [-0.00207, +0.00021], which STRADDLES ZERO. This
        # module's own `_cv_delta_ci` docstring says the sign of the
        # INTERVAL, not of the point estimate, is what licenses a "beats the
        # base rate" claim (queue #8); the licence was never asked for here.
        brier_cv = _cv_brier(composites)
        _delta_ci = _cv_delta_ci(composites)
        beats_base_rate = _base_rate_verdict(_delta_ci)

        map_status = "applied"
        if platt_a <= 0.0:
            logger.warning(
                "calibration: REJECTED the probability map (slope %.3f <= 0)."
                " The composite is anti-correlated with outcomes, so the map"
                " would INVERT the confidence ordering. Confidence stays on"
                " the raw scale; the score needs a feature that actually"
                " predicts turn failure, not better calibration.", platt_a)
            platt_a, platt_b = 1.0, 0.0
            calibrated = composites
            brier = brier_raw
            map_status = "rejected_inverted"
        elif platt_a > _MAX_SLOPE:
            logger.warning(
                "calibration: REJECTED the probability map (slope %.1f > %.1f)."
                " That is a near-step function — almost certainly a linearly"
                " separable batch fitted in-sample — and would make the"
                " below-threshold gate a hair-trigger on a single composite"
                " value.", platt_a, _MAX_SLOPE)
            platt_a, platt_b = 1.0, 0.0
            calibrated = composites
            brier = brier_raw
            map_status = "rejected_step"
        elif brier_cal <= brier_raw:
            brier = brier_cal
        else:
            logger.debug(
                "calibration: discarding Platt map (Brier %.4f > raw %.4f)",
                brier_cal, brier_raw)
            platt_a, platt_b = 1.0, 0.0
            calibrated = composites
            brier = brier_raw
            map_status = "discarded_worse"
        # ⚠ THE VERDICT MUST DESCRIBE THE PREDICTOR WE ACTUALLY SHIP.
        #
        # `_cv_brier` and `_cv_delta_ci` fit a Platt map PER FOLD, so they
        # measure the CALIBRATED model. On the three rejection paths above
        # the map is thrown away and the RAW composite is shipped
        # (`platt_a, platt_b = 1.0, 0.0`) — so the interval describes a
        # predictor that was explicitly refused. Measured: an anti-correlated
        # composite (the live "AUC 0.473" state) yields `rejected_inverted`
        # AND `beats_base_rate="yes"`, about a shipped scorer whose Brier is
        # 0.724 against a base rate of 0.250 — three times WORSE than a
        # constant, carrying the one answer that licenses trusting it.
        #
        # A fold-wise refit of the WEIGHTS would give a cross-validated
        # measurement of the raw composite, so this is conservatism, not
        # impossibility — say which it is. We decline to invent one from the
        # calibrated fit, because that is how the wrong licence was granted
        # in the first place. So: not measured, therefore `unknown`.
        #
        # It must be exactly `BEATS_UNKNOWN`, never `BEATS_NO` or
        # `BEATS_INDISTINGUISHABLE`. Those two are MEASURED words — "we ran
        # the comparison and the model lost" / "…and it was a tie" — and we
        # ran no comparison. Recording either would be a fabricated neutral.
        # The OTHER question, on the same rows.
        #
        # Measured on the RAW composite. `_auc_ci(calibrated)` would give the
        # IDENTICAL number today — AUC is invariant to a strictly increasing
        # transform, so the applied branch (a > 0) agrees by construction, and
        # every rejection branch above reassigns `calibrated = composites`.
        # A mutation sweep proved the two equivalent, so this is a clarity and
        # independence choice, not a correctness fix: the raw column is what
        # ships whenever the map is refused, and reading it directly means
        # this line does not silently depend on those three reassignments
        # staying in place. (An earlier version of this comment claimed the
        # inverted branch would report `1 - AUC`. It would not — `calibrated`
        # is rebound there. A justification has to be checked like a claim.)
        _auc_res = _auc_ci(composites)
        ranks_outcomes = _rank_verdict(_auc_res)

        if not map_applied(map_status):
            logger.warning(
                "calibration: map %s, so the shipped predictor is the RAW "
                "composite — the cross-validated interval describes the "
                "CALIBRATED model and does not apply to it. Recording "
                "beats_base_rate=%s rather than a licence we did not measure.",
                map_status, BEATS_UNKNOWN)
            beats_base_rate = downgrade_for_map(beats_base_rate, map_status)

        # ⚠ DRIVEN BY THE INTERVAL. The point comparison `brier > brier_base`
        # is silent for the case that actually matters — a model whose point
        # estimate edges the base rate while its CI says the two are
        # indistinguishable. That is "a margin is not a result" (§4CY), and
        # it is the live state of this store.
        # ⚠ SAY WHICH KIND OF SIGNAL IS MISSING. Both warnings below used to
        # end on "not adding information" / "nothing may treat this score as
        # informative", which reads as "the score is noise". At this corpus's
        # base rate that conclusion does not follow from a Brier comparison:
        # measured 2026-08-31, the SAME rows were indistinguishable as a
        # probability and ordered turns at AUC 0.662 [0.580, 0.738]. So each
        # warning now carries the rank measurement beside the probability one
        # — and it is interval-driven too, so this cannot become a second
        # licence granted on a point estimate (the §4DZ defect).
        if ranks_outcomes == BEATS_YES:
            _rank_note = (
                " It DOES still rank: AUC %.3f [%.3f, %.3f], so a RANKING use "
                "(flagging the lowest-confidence turns) is supported where a "
                "probability claim is not — see `ranks_outcomes`."
                % (_auc_res[0], _auc_res[1], _auc_res[2]))
        elif ranks_outcomes == BEATS_NO:
            _rank_note = (
                " And it orders turns BACKWARDS: AUC %.3f [%.3f, %.3f], whole "
                "interval below chance. No use of this score is supported."
                % (_auc_res[0], _auc_res[1], _auc_res[2]))
        elif _auc_res:
            _rank_note = (
                " Nor does it rank: AUC %.3f [%.3f, %.3f] spans chance. No use "
                "of this score is supported."
                % (_auc_res[0], _auc_res[1], _auc_res[2]))
        else:
            _rank_note = (" Discrimination was NOT measured (too few rows or "
                          "one outcome class), so it is unknown, not absent.")

        if beats_base_rate == BEATS_NO:
            logger.warning(
                "calibration: fitted model (CV Brier %.4f) is WORSE than "
                "always predicting the base rate %.3f (Brier %.4f) — the "
                "confidence score is not adding information AS A PROBABILITY."
                "%s",
                brier_cv, base, brier_base, _rank_note)
        elif beats_base_rate == BEATS_INDISTINGUISHABLE:
            logger.warning(
                "calibration: fitted model (CV Brier %.4f) is INDISTINGUISHABLE "
                "from always predicting the base rate %.3f (Brier %.4f) — the "
                "95%% CI of the delta [%+.5f, %+.5f] straddles zero. The map "
                "is still applied (it beats the RAW composite, a different "
                "question). Nothing may treat this score AS A PROBABILITY: "
                "see `beats_base_rate` on the stored params.%s",
                brier_cv, base, brier_base,
                _delta_ci[1] if _delta_ci else float("nan"),
                _delta_ci[2] if _delta_ci else float("nan"),
                _rank_note)

        # ── HONEST PERFORMANCE (audit 2026-08-10) ────────────────────────
        # `brier` above is IN-SAMPLE and was the headline. Cross-validate so
        # the reported gain is one the model did not buy with its own
        # parameters; both are stored and the renderer labels which is which.
        # `composites` is already a list of (composite, outcome) PAIRS.
        # (`brier_cv` and `_delta_ci` are computed ABOVE, before the map
        # decision that must consult them. They used to be computed here,
        # which is why that decision could not.)
        n_negative = sum(1 for _c, y in composites if y < 0.5)

        # Per-feature ABLATION. `_separation_sigmas` is a difference-of-means
        # test and there are 18 negatives — it cannot resolve what it is
        # asked to judge, and it calls a feature that demonstrably helps
        # DEAD. This measures the question actually being asked.
        def _rebuild(drop: str):
            if drop == "entropy_component":
                w = dict(w_e=0.0, lam=lam, w_eff=w_eff)
            elif drop == "effort_component":
                w = dict(w_e=w_e, lam=lam, w_eff=0.0)
            elif drop == "competence_component":
                # Competence is the RESIDUAL (1 - w_e - w_eff), so dropping it
                # means handing its share to effort rather than zeroing a
                # weight that does not exist as a variable.
                w = dict(w_e=w_e, lam=lam, w_eff=min(1.0, w_eff + w_c))
            else:
                return None
            return [(_composite_for(s, **w), s.outcome) for s in samples]

        feature_contrib = {}
        for _attr in ("entropy_component", "competence_component",
                      "effort_component"):
            _d = _feature_contribution(_attr, composites, rebuild=_rebuild)
            if _d is not None:
                feature_contrib[_attr] = _d

        # Threshold is picked on the CALIBRATED scores (Youden's J on the
        # "predict success when p ≥ τ" decision) — it must live on the same
        # scale `below_threshold` will compare against, or the gate fires at
        # the wrong point.
        threshold = _best_threshold(calibrated)

        params = FittedParams(
            w_entropy=round(w_e, 4),
            w_competence=round(w_c, 4),
            threshold=round(threshold, 4),
            lambda_uncertainty=round(lam, 4),
            brier=round(brier, 6),
            n_samples=len(samples),
            fitted_at=_utcnow_iso(),
            n_entropy_observed=len(observed),
            platt_a=round(platt_a, 6),
            platt_b=round(platt_b, 6),
            brier_raw=round(brier_raw, 6),
            brier_base_rate=round(brier_base, 6),
            w_effort=round(w_eff, 4),
            n_effort_observed=len(eff_observed),
            map_status=map_status,
            beats_base_rate=beats_base_rate,
            epoch=CURRENT_EPOCH,
            n_excluded_other_epochs=n_excluded,
            brier_cv=round(brier_cv, 6) if brier_cv is not None else -1.0,
            brier_cv_delta_lo=(round(_delta_ci[1], 6)
                               if _delta_ci is not None else None),
            brier_cv_delta_hi=(round(_delta_ci[2], 6)
                               if _delta_ci is not None else None),
            n_negative=n_negative,
            feature_contrib=feature_contrib or None,
            auc=round(_auc_res[0], 4) if _auc_res else -1.0,
            auc_ci_lo=round(_auc_res[1], 4) if _auc_res else None,
            auc_ci_hi=round(_auc_res[2], 4) if _auc_res else None,
            ranks_outcomes=ranks_outcomes,
            bench_verdict=_pop.bench_verdict,
            n_bench_misaligned=_pop.n_bench_misaligned,
        )
        self._save_params(params)
        return params

    # ----------------------------------------------------------- params io

    def load_params(self) -> Optional[FittedParams]:
        """Read the persisted fitted params, or ``None`` if absent /
        corrupt / wrong-schema (degrade to hardcoded defaults)."""
        if not self.params_path.exists():
            return None
        try:
            d = json.loads(self.params_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("calibration params load failed: %s", exc)
            return None
        if not isinstance(d, dict) or d.get("schema") != SCHEMA_VERSION:
            return None
        try:
            return FittedParams(
                w_entropy=float(d["w_entropy"]),
                w_competence=float(d["w_competence"]),
                threshold=float(d["threshold"]),
                lambda_uncertainty=float(d.get("lambda_uncertainty", 0.0)),
                brier=float(d.get("brier", 0.0)),
                n_samples=int(d.get("n_samples", 0)),
                fitted_at=str(d.get("fitted_at", "")),
                n_entropy_observed=int(d.get("n_entropy_observed", 0)),
                # Identity defaults: a params file written before the
                # recalibration stage existed replays exactly as before.
                platt_a=float(d.get("platt_a", 1.0)),
                platt_b=float(d.get("platt_b", 0.0)),
                brier_raw=float(d.get("brier_raw", -1.0)),
                brier_base_rate=float(d.get("brier_base_rate", -1.0)),
                w_effort=float(d.get("w_effort", 0.0)),
                n_effort_observed=int(d.get("n_effort_observed", 0)),
                # ⚠ `.get(key, default)` NOT `.get(key) or default`. The
                # second turns an explicit `null` and `""` into "applied" —
                # a malformed value reading as a map in force. With the
                # default form a null stringifies to "None", which
                # `map_applied` rejects, so the file fails closed.
                map_status=str(d.get("map_status", "applied")),
                # ⚠ THE READER SET AGAIN. `_save_params` WRITES this field and
                # for one commit this loader did not reconstruct it, so every
                # RELOADED fit carried the default forever. Identical to the
                # collector omission this same change had already fixed once,
                # one reader along.
                #
                # ⚠ AND THE FIRST VERSION OF THIS COMMENT WAS FALSE. It named
                # two consequences that do not exist: `apply_fitted`
                # (confidence.py) reads only the weights, threshold, lambda
                # and Platt pair and never touches this field, and `stats()`
                # has no production caller at all. The live path from producer
                # to operator is `_save_params` → params.json →
                # `learning_health._load_json` (raw dict, NOT this loader) →
                # collector → renderer. This loader is on the path that feeds
                # the metacog CALIB line, and that is the reason to keep it
                # correct — a justification has to be checked like a claim.
                # ⚠ THE THIRD READER, AND IT DID NOT ASK. `fit()` and the
                # learning-health renderer both route the verdict through
                # `downgrade_for_map`; this loader did not, so ONE file
                # produced opposite answers on two operator surfaces — the
                # startup CALIB line emitted
                # `{map: rejected_inverted, beats_base_rate: yes}` while
                # `introspect learning` said no comparison applied. A params
                # file from the previous build carries exactly that pair.
                beats_base_rate=_verdict_from_params(d),
                epoch=str(d.get("epoch", "")),
                n_excluded_other_epochs=int(d.get("n_excluded_other_epochs", 0)),
                # Defaults keep a pre-audit params file loading unchanged;
                # -1.0 / 0 / None each read as "not recorded", never as a
                # measured value of zero.
                brier_cv=float(d.get("brier_cv", -1.0)),
                # None-preserving: a params file written before this existed
                # must load as "no CI recorded", not as a CI of zero width
                # (which would license a bogus verdict).
                brier_cv_delta_lo=(
                    float(d["brier_cv_delta_lo"])
                    if d.get("brier_cv_delta_lo") is not None else None),
                brier_cv_delta_hi=(
                    float(d["brier_cv_delta_hi"])
                    if d.get("brier_cv_delta_hi") is not None else None),
                n_negative=int(d.get("n_negative", 0)),
                feature_contrib=(d.get("feature_contrib")
                                 if isinstance(d.get("feature_contrib"), dict)
                                 else None),
                # ⚠ THE READER SET, AGAIN. This file already records a field
                # that `_save_params` wrote and this loader did not
                # reconstruct, so every reloaded fit carried the default
                # forever. Same defaults discipline: -1.0 / None / `unknown`
                # each read as "not recorded", never as a measured value.
                auc=float(d.get("auc", -1.0)),
                auc_ci_lo=(float(d["auc_ci_lo"])
                           if d.get("auc_ci_lo") is not None else None),
                auc_ci_hi=(float(d["auc_ci_hi"])
                           if d.get("auc_ci_hi") is not None else None),
                # Coerced through the vocabulary for the same reason
                # `beats_base_rate` is: an unrecognised word is not a verdict,
                # and `in NOT_INFORMATIVE` is FALSE for one, so a typo would
                # read as a licence.
                ranks_outcomes=_known_verdict(d.get("ranks_outcomes")),
                bench_verdict=_known_verdict(d.get("bench_verdict")),
                n_bench_misaligned=int(d.get("n_bench_misaligned", 0)),
            )
        # ⚠ `OverflowError` IS AN `ArithmeticError`, NOT A `ValueError`.
        # `int(float('inf'))` raises out of this function for five fields
        # (n_samples, n_entropy_observed, n_effort_observed,
        # n_excluded_other_epochs, n_negative), against this module's own
        # documented contract: "returns None on ANY problem so a corrupt file
        # degrades to the hardcoded defaults, never a crash." A caller had
        # already diagnosed this gap and worked around it with a broad
        # `except Exception` on its own side, leaving `stats()` and the
        # startup path exposed. Fix the contract where it is stated.
        except Exception as exc:  # noqa: BLE001 — the contract is "never raise"
            logger.debug("calibration params malformed: %s", exc)
            return None

    def _save_params(self, params: FittedParams) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.params_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
            os.replace(tmp, self.params_path)
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("calibration params save failed: %s", exc)

    # ----------------------------------------------------------- summary

    def stats(self) -> Dict[str, object]:
        """Introspection summary.

        ⚠ NO PRODUCTION CALLER TODAY. The docstring used to claim "for
        ``introspect`` / the calib log"; neither is true. `introspect
        learning` renders through `learning_health.render_learning_health`,
        which reads the params JSON directly, and the metacog CALIB lines
        read a `FittedParams` from `fit()` / `load_params()`. This is a
        test-and-debug surface. Said plainly so the next reader does not
        assume, as this change's author did, that adding a key here reaches
        an operator.
        """
        all_samples = self._load_samples(limit=self.max_history)
        # §4BF 1c (R1 review): the SAME resolution + origin cap as the fit.
        # The raw-epoch version reported a brier over superseded duplicates
        # and uncapped bench rows — a population the fit never consumes
        # (measured in review: 0.2614 over 7 raw rows vs 0.29 over the
        # fit's 2). The raw count stays visible as `samples_raw_epoch`.
        raw_epoch = [s for s in all_samples if s.epoch == CURRENT_EPOCH]
        samples = _fit_population(raw_epoch, self.load_params()).samples
        brier = (
            sum((s.composite - s.outcome) ** 2 for s in samples) / len(samples)
            if samples else None
        )
        params = self.load_params()
        return {
            # Current epoch, resolved + capped — the population the fit and
            # every metric below describe. `samples_all_epochs` is reported
            # beside it so a drop after an epoch bump reads as a filter,
            # not as data loss.
            "samples": len(samples),
            "samples_raw_epoch": len(raw_epoch),
            "samples_all_epochs": len(all_samples),
            "epoch": CURRENT_EPOCH,
            "brier": round(brier, 4) if brier is not None else None,
            # Same window as `brier` above. Calling `self.ece()` with no
            # argument scored the ENTIRE file while brier scored only the
            # max_history tail, so the two numbers sat side by side
            # describing different populations (measured: 0.0478 vs 0.1593
            # on the same report).
            "ece": (round(self.ece(window=self.max_history) or 0.0, 4)
                    if samples else None),
            "fitted": params is not None,
            "threshold": params.threshold if params else None,
            "w_entropy": params.w_entropy if params else None,
            "lambda_uncertainty": params.lambda_uncertainty if params else None,
            "map_status": (getattr(params, "map_status", "applied")
                           if params else None),
            # The keep/kill instrument must see the LICENCE, not just whether
            # the Platt map was applied — they answer different questions,
            # and the live store is "applied" AND "indistinguishable".
            "beats_base_rate": (getattr(params, "beats_base_rate",
                                        BEATS_UNKNOWN)
                                if params else None),
        }


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────

def _clamp01(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(v):
        return 0.5
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _best_threshold(pairs: List[Tuple[float, float]]) -> float:
    """Pick τ maximising Youden's J for "predict success when C ≥ τ".

    J = sensitivity + specificity − 1 = TPR − FPR. Candidate thresholds
    are the midpoints between sorted unique composites (plus the rails),
    so the chosen cut is robust to the exact float values. Ties break
    toward the *higher* threshold (more conservative — flips more turns
    into "below threshold → arbitrate"). Falls back to 0.5 when J never
    beats the trivial classifier.
    """
    # Degenerate-input fallbacks return the MEDIAN of the supplied scores
    # rather than a hardcoded constant. The caller may be on the raw
    # composite scale or on the Platt-mapped probability scale, and a fixed
    # 0.55 is only meaningful on the former — returning it into a mapped
    # caller put the gate at an arbitrary point of a different distribution.
    # The median is scale-free and always sits inside the observed range.
    def _median_conf(default=0.55):
        vals = sorted(c for c, _ in pairs)
        if not vals:
            return default
        mid = len(vals) // 2
        return (vals[mid] if len(vals) % 2
                else (vals[mid - 1] + vals[mid]) / 2.0)

    if not pairs:
        return 0.55
    n_pos = sum(1 for _, o in pairs if o >= 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return _median_conf()
    confs = sorted({round(c, 4) for c, _ in pairs})
    candidates = [0.0]
    for i in range(len(confs) - 1):
        candidates.append((confs[i] + confs[i + 1]) / 2.0)
    candidates.append(1.0)
    best_tau = 0.5
    best_j = -2.0
    for tau in candidates:
        tp = sum(1 for c, o in pairs if c >= tau and o >= 0.5)
        fp = sum(1 for c, o in pairs if c >= tau and o < 0.5)
        tpr = tp / n_pos
        fpr = fp / n_neg
        j = tpr - fpr
        if j > best_j or (abs(j - best_j) < 1e-9 and tau > best_tau):
            best_j = j
            best_tau = tau
    # Documented fallback (was missing): if no threshold beats the trivial
    # classifier (Youden J <= 0 — the composite is uncorrelated with outcome,
    # i.e. the miscalibrated case this exists to catch), the loop would pick a
    # DEGENERATE rail (τ=1.0 via the higher-tie-break → "below" ALWAYS True, or
    # τ=0.0 → always False). Return the median of the observed scores — a
    # neutral cut that is valid on WHICHEVER scale the caller passed, unlike
    # the hardcoded 0.5 this used to return.
    if best_j <= 1e-9:
        return _clamp01(_median_conf(0.5))
    return _clamp01(best_tau)


def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


__all__ = [
    "CalibrationTracker",
    "CalibrationSample",
    "FittedParams",
    "ReliabilityBin",
    "SCHEMA_VERSION",
    "CURRENT_EPOCH",
    "epoch_for_ts",
    # The base-rate verdict vocabulary. `NOT_INFORMATIVE` is the guard a
    # consumer is meant to test against ("may I treat this score as
    # carrying information?"), so it belongs on the declared surface.
    "BEATS_YES",
    "BEATS_NO",
    "BEATS_INDISTINGUISHABLE",
    "BEATS_UNKNOWN",
    "NOT_INFORMATIVE",
    "map_applied",
    "map_applied_in_params",
    "downgrade_for_map",
]
