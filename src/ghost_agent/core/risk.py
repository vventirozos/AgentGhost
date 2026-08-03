"""Turn-risk estimate + the throttle that finally consumes it.

The gap this closes
-------------------
Two failure predictors were MEASURED on this agent and then used for nothing:

1. **Depth.** Over 342 labelled trajectories the probability that a turn ends
   badly rises monotonically with how many steps it has already taken:
   17.8% at step 1, 35.6% at 4, 42.3% at 6, 52.0% at 8, 60.6% at 12 (§4H).
   That is a usable policy signal with no model attached to it at all.
2. **The calibrated confidence composite** (AUC 0.727 post-hoc). §4H records
   the measured verdict plainly: it has **no behavioural consumer** —
   `_METACOG_ARBITER_ENABLED = False`, and the score is logged, never acted on.

So the agent could tell you it was probably failing, and did nothing
differently. This module is the missing consumer, in two halves:

* **Live (mid-turn).** `turn_risk()` scores the turn IN FLIGHT from signals
  available before it ends — depth, turn shape (`confidence.effort_component`,
  measured AUC 0.670), and the strike ledger. When a turn is both deep and
  struggling, `risk_steer_message()` injects ONE steer telling the model to
  bank what it knows, shrink the next check, or stop and report honestly.
  The point is the last option: at step 8 the agent is at ~52% and its
  best move is often to raise its hand, which it currently never does.
* **Offline (post-hoc).** `triage_score()` / `rank_for_triage()` — §4H item 2.
  Reflection, post-mortem and self-play all select work by the BINARY "did it
  fail" flag; a graded score is strictly more informative and orders the
  queue by how badly a turn went. The ranking runs on ONE scale: the
  calibration join (on `req_id`, the §4E Tier-3 key) is used only when it
  covers the entire pool, otherwise everything is ranked by the
  trajectory-shape proxy. Mixing the two — the obvious implementation — was
  measured WORSE than either alone (Kendall τ 0.605 vs 0.766 / 0.763) and on
  the live corpus it demoted the very trajectories it was meant to promote.
  See `rank_for_triage`.

Honesty about what these numbers are
------------------------------------
`depth_failure_prior` is empirical — those five points are measured on this
agent's own corpus. The BLEND in `turn_risk` is not: it is a documented
heuristic ordering over three correlated signals (a deep turn has, by
construction, made many tool calls), deliberately NOT called a probability.
It exists to rank turns for a steer, and its only claim is that higher means
worse. Note the consequence spelled out at the weights: the calibrated term
has the SMALLEST decision span, so the gate is in practice driven by the two
uncalibrated ones.

What `core.calibration` stores is likewise not a calibrated probability — the
on-disk `composite` column is the RAW pre-Platt score, kept on one stable
scale by design. It is a monotone transform of a calibrated probability,
which is enough to rank WITHIN itself and not enough to compare against
anything else.

Whether the live steer actually helps is not asserted here — it ships as one
arm of the `risk_steer` experiment (`core.experiments`), so the answer comes
from randomized live traffic instead of from this docstring.

Kill switches: ``GHOST_RISK_STEER=0`` disables the live steer outright (the
experiment framework's own ``GHOST_EXPERIMENTS=0`` does too, by putting every
request on the control path). ``GHOST_TRIAGE_RANKING=0`` reverts reflection to
chronological selection.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("GhostAgent")

# The measured depth→failure curve (§4H, 342 labelled trajectories). Steps are
# 1-based. Values BETWEEN the measured points are linearly interpolated; past
# the last point the curve is held flat rather than extrapolated — 12 steps is
# where the corpus runs out, and inventing 80% at step 30 would be fabrication.
_DEPTH_CURVE: Tuple[Tuple[int, float], ...] = (
    (1, 0.178),
    (4, 0.356),
    (6, 0.423),
    (8, 0.520),
    (12, 0.606),
)

# Blend weights.
#
# ⚠ A WEIGHT IS NOT AN IMPORTANCE when the inputs have different ranges.
# Measured contribution spans over all reachable inputs:
#
#     depth   (w=0.5)   contributes [0.089, 0.303]   span 0.214
#     effort  (w=0.3)   contributes [0.000, 0.300]   span 0.300
#     failure (w=0.2)   contributes [0.000, 0.200]   span 0.200
#
# So depth — the only EMPIRICALLY CALIBRATED term — has the SMALLEST decision
# span, and mostly acts as a 0.089 offset. Depth alone cannot cross the steer
# threshold at any step: a 30-step turn that has run no tools scores 0.303.
# That is deliberate (a long turn making steady progress is not in trouble)
# but it means the gate is driven by the two UNCALIBRATED terms, which is
# exactly why the blend is documented as a heuristic ordering and why the
# steer text quotes the depth prior rather than this score.
_W_DEPTH = 0.5
_W_EFFORT = 0.3
_W_FAILURE = 0.2

# Failure pressure saturates here: 4 strikes reads as "this turn is in
# trouble" and more does not make it worse for ranking purposes.
#
# NOTE what the input actually is: `execution_failure_count` is a DECAYED
# strike ledger (the turn loop decrements it on recovery), not a cumulative
# count. A turn that failed six times and recovered contributes 0 here — by
# design, since a recovered turn is not in trouble, but it means this term
# reads "currently struggling", never "has struggled".
_FAILURE_SATURATE = 4.0

# Steer gate. Both conditions must hold:
#   * step >= _MIN_STEER_STEP — below step 6 the measured failure rate is
#     under 42% and a steer is mostly noise on turns doing fine;
#   * score >= _STEER_THRESHOLD.
# Worked examples with the weights above:
#   step 6, 5 clean calls          → 0.30  (no steer — a normal working turn)
#   step 8, 8 calls, 1 failure     → 0.44  (no steer)
#   step 10, 10 calls, 2 repeats,
#           1 failure              → 0.51  (steer — measured ~56% failure)
#   step 12, 12 calls, 3 repeats,
#           2 failures             → 0.63  (steer)
_MIN_STEER_STEP = 6
_STEER_THRESHOLD = 0.50

ENV_STEER_KILL = "GHOST_RISK_STEER"
ENV_TRIAGE_KILL = "GHOST_TRIAGE_RANKING"

# The experiment whose treatment arm enables the live steer.
EXPERIMENT = "risk_steer"


def _env_off(name: str) -> bool:
    return str(os.getenv(name, "1")).strip().lower() in ("0", "false", "off", "no")


def steer_enabled() -> bool:
    return not _env_off(ENV_STEER_KILL)


def triage_ranking_enabled() -> bool:
    return not _env_off(ENV_TRIAGE_KILL)


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return min(1.0, max(0.0, v))


def depth_failure_prior(step: int) -> float:
    """P(turn ends badly) given it has reached ``step`` — measured, §4H.

    ``step`` is 1-based (the first model turn is step 1). Linear interpolation
    between measured points; flat outside the measured range in BOTH
    directions.
    """
    try:
        s = int(step)
    except (TypeError, ValueError):
        return _DEPTH_CURVE[0][1]
    if s <= _DEPTH_CURVE[0][0]:
        return _DEPTH_CURVE[0][1]
    if s >= _DEPTH_CURVE[-1][0]:
        return _DEPTH_CURVE[-1][1]
    for (x0, y0), (x1, y1) in zip(_DEPTH_CURVE, _DEPTH_CURVE[1:]):
        if x0 <= s <= x1:
            if x1 == x0:
                return y1
            frac = (s - x0) / float(x1 - x0)
            return y0 + frac * (y1 - y0)
    return _DEPTH_CURVE[-1][1]


@dataclass(frozen=True)
class RiskReading:
    """One mid-turn risk estimate.

    ``score`` is the blended heuristic — an ORDERING, not a probability (see
    the module docstring). Its REACHABLE range is [0.089, 0.803], not [0, 1]:
    the depth term contributes a floor of 0.089 at step 1 and the three terms
    cannot simultaneously max out. Band cut-points are read against that real
    range, so do not reason about them as if 0.0 or 1.0 were attainable.

    ``depth_prior`` IS a measured probability and is reported separately so a
    consumer that wants the honest number can use it directly; the steer text
    quotes this one.
    """

    score: float
    depth_prior: float
    effort_struggle: float
    failure_pressure: float
    step: int
    band: str  # "low" | "elevated" | "high"

    @property
    def should_steer(self) -> bool:
        return self.step >= _MIN_STEER_STEP and self.score >= _STEER_THRESHOLD


def turn_risk(*, step: int, tool_names: Optional[Sequence[str]] = None,
              execution_failures: int = 0,
              transient_failures: int = 0) -> RiskReading:
    """Score the turn currently in flight.

    ``step`` is 1-based. ``tool_names`` is every tool this REQUEST has run so
    far (the turn loop's `tools_run_this_turn` accumulates across the whole
    request despite its name). Pure and total — never raises.
    """
    try:
        s = max(1, int(step))
    except (TypeError, ValueError):
        s = 1
    depth = depth_failure_prior(s)

    # Turn shape. `effort_component` returns 1.0 for a clean short turn, so
    # struggle is its complement. No tool calls yet = no evidence of struggle
    # (0.0), matching how calibration treats an unobserved effort component
    # rather than fabricating a neutral.
    struggle = 0.0
    try:
        names = [str(n) for n in (tool_names or []) if n]
        if names:
            from .confidence import effort_component
            struggle = 1.0 - _clamp01(effort_component(names))
    except Exception:  # noqa: BLE001 — risk scoring must never break a turn
        struggle = 0.0

    try:
        fails = max(0, int(execution_failures or 0)) + max(0, int(transient_failures or 0))
    except (TypeError, ValueError):
        fails = 0
    pressure = min(1.0, fails / _FAILURE_SATURATE)

    score = _clamp01(_W_DEPTH * depth + _W_EFFORT * struggle + _W_FAILURE * pressure)
    band = "high" if score >= _STEER_THRESHOLD else (
        "elevated" if score >= 0.35 else "low")
    return RiskReading(score=score, depth_prior=depth, effort_struggle=struggle,
                       failure_pressure=pressure, step=s, band=band)


def risk_steer_message(reading: RiskReading) -> str:
    """The one-shot steer injected on a deep, struggling turn.

    Quotes the MEASURED depth prior, not the blended score: the agent is being
    told a fact about its own history, and the fact should be the one that was
    actually measured. Ends on the option the agent never takes on its own —
    stop and report — because that is the whole point of knowing the number.

    Two things this text must NOT do, both of which the first version did:

    * **Launder the held-flat prior as a measurement at the current depth.**
      The curve stops at step 12; past that the value is HELD, not measured.
      Saying "turns that reach this depth fail 61% of the time" at step 40
      states a measurement that does not exist, inside the model's own
      context, where it becomes a premise it will reason from.
    * **Assert a cause.** The 342-trajectory result is an observational
      conditional failure rate. "The cause is almost always X" is a causal
      claim no part of that analysis supports — deep turns are also simply
      where the hard tasks live.
    """
    pct = int(round(reading.depth_prior * 100))
    beyond = reading.step > _DEPTH_CURVE[-1][0]
    where = (f"at {_DEPTH_CURVE[-1][0]} steps or more"
             if beyond else "at this depth")
    return (
        f"SYSTEM ALERT (risk governor): this request is at step {reading.step}. "
        f"Measured on this agent's own history, turns {where} end badly about "
        f"{pct}% of the time — one common pattern behind that is continuing to "
        "grind an approach that has already stopped producing new information "
        "(the other is simply that hard tasks run long). Before your next tool "
        "call, do these IN ORDER:\n"
        "1. STATE what is now CONFIRMED (exact values, paths, structures you "
        "have actually observed) and what is still ASSUMED. Write confirmed "
        "facts somewhere durable if this is project work.\n"
        "2. Pick the SINGLE smallest check that would distinguish the leading "
        "assumption from its alternative, and run only that.\n"
        "3. If you cannot name such a check — or the last two steps produced no "
        "new information — STOP. Report what is known, what is blocked, and "
        "what you would need. An honest partial answer at this depth beats a "
        "confident wrong one; a failed tool that you REPORT is not a failed turn."
    )


# ──────────────────────────────────────────────────────────────────────
# Offline triage (§4H item 2)
# ──────────────────────────────────────────────────────────────────────

# Shape-proxy weights for a FINISHED trajectory. Same spirit as the live
# blend; depth and effort are what a trajectory actually carries.
_T_W_DEPTH = 0.5
_T_W_EFFORT = 0.3
_T_W_ERROR = 0.2


def _tool_names_of(traj) -> List[str]:
    out: List[str] = []
    for tc in (getattr(traj, "tool_calls", None) or []):
        name = getattr(tc, "name", None)
        if name is None and isinstance(tc, dict):
            name = tc.get("name")
        if name:
            out.append(str(name))
    return out


def _tool_error_fraction(traj) -> float:
    calls = getattr(traj, "tool_calls", None) or []
    if not calls:
        return 0.0
    errs = 0
    for tc in calls:
        err = getattr(tc, "error", None)
        if err is None and isinstance(tc, dict):
            err = tc.get("error")
        if str(err or "").strip():
            errs += 1
    return errs / float(len(calls))


def shape_triage_score(traj) -> float:
    """Post-hoc risk from trajectory shape alone — the fallback when no
    calibration sample exists for the turn. Higher = worse."""
    try:
        steps = int(getattr(traj, "n_steps", 0) or 0)
    except Exception as e:  # noqa: BLE001 — any hostile attribute
        logger.debug("triage: n_steps unreadable (%s) — treated as shallow",
                     type(e).__name__)
        steps = 0
    depth = depth_failure_prior(max(1, steps))
    struggle = 0.0
    names = _tool_names_of(traj)
    if names:
        try:
            from .confidence import effort_component
            struggle = 1.0 - _clamp01(effort_component(names))
        except Exception:  # noqa: BLE001
            struggle = 0.0
    return _clamp01(_T_W_DEPTH * depth + _T_W_EFFORT * struggle
                    + _T_W_ERROR * _tool_error_fraction(traj))


def calibrated_risk_index(samples: Iterable[Any]) -> Dict[str, float]:
    """``{req_id: 1 - stored_composite}`` from calibration samples.

    The stored `composite` column is the RAW pre-Platt score — the calibration
    spine deliberately keeps ONE stable scale on disk, so this is NOT a
    calibrated probability and must not be described as one. It is a monotone
    transform of one, which is all a ranking needs *within this set*; see
    `rank_for_triage` for why that qualifier is load-bearing.

    Rows with a missing or non-finite composite are SKIPPED, not defaulted:
    `_clamp01(None)` returns 0.0, so a null would score 1.0 — maximum risk —
    and the least informative record in the store would jump the queue ahead
    of every real disaster.
    """
    out: Dict[str, float] = {}
    for s in samples or []:
        try:
            req = str(getattr(s, "req_id", "") or "")
            if not req:
                continue
            raw = getattr(s, "composite", None)
            if raw is None:
                continue
            val = float(raw)
            if not math.isfinite(val):
                continue
            out[req] = 1.0 - _clamp01(val)
        except (TypeError, ValueError):
            continue
        except Exception:  # noqa: BLE001
            continue
    return out


def triage_score(traj, *, calibrated: Optional[Dict[str, float]] = None) -> float:
    """Risk score for ONE finished trajectory. Higher = attend first.

    ⚠ Two different scales live here and they must never be mixed inside one
    ranking — `rank_for_triage` is the API to use. This function returns the
    calibrated value when the join hits and the shape proxy otherwise, which
    is only correct once the caller has established the whole pool is on one
    scale.
    """
    try:
        if calibrated:
            key = str(getattr(traj, "session_id", "") or "")
            if key and key in calibrated:
                return _clamp01(calibrated[key])
    except Exception:  # noqa: BLE001
        pass
    return shape_triage_score(traj)


def rank_for_triage(trajectories: Iterable[Any], *,
                    calibrated: Optional[Dict[str, float]] = None,
                    limit: Optional[int] = None) -> List[Any]:
    """Worst-first ordering of trajectories, ON A SINGLE SCALE.

    **Why "single scale" is the whole design.** The obvious implementation —
    calibrated score where it exists, shape proxy elsewhere — sorts one list
    on two incommensurable quantities. Measured, that is worse than either
    input alone:

        shape only        Kendall tau 0.766 vs. true badness
        calibrated only   Kendall tau 0.763
        MIXED             Kendall tau 0.605     <- the obvious implementation

    and on this agent's live corpus 7 of the 8 joinable failures scored LOWER
    under the calibrated path than under their own shape proxy — so HAVING a
    calibration sample DEMOTED a trajectory, and a real 12-step disaster fell
    out of the reflection queue in favour of a milder turn that simply had no
    sample. The raw pre-Platt composite has no meaningful level (the weight
    fit "cannot set their level", per `core.confidence`), so comparing it
    numerically against a proxy that does is meaningless.

    So: the calibrated scale is used only when it covers EVERY item in the
    pool; otherwise everything ranks by shape. At today's ~5% join rate that
    means shape in practice, and the calibrated path arms itself
    automatically if coverage ever becomes complete.

    Ties break on the ORIGINAL order (Python's sort is stable), so an
    all-equal-score corpus degrades exactly to chronological behaviour
    instead of shuffling.
    """
    items = list(trajectories or [])
    if not items:
        return []
    try:
        use_calibrated = False
        if calibrated:
            keys = [str(getattr(t, "session_id", "") or "") for t in items]
            use_calibrated = all(k and k in calibrated for k in keys)
            if not use_calibrated:
                logger.debug(
                    "triage: calibrated coverage %d/%d — ranking by shape "
                    "(mixing scales measures worse than either alone)",
                    sum(1 for k in keys if k and k in calibrated), len(items))
        scored = []
        for i, t in enumerate(items):
            # Per-item guard: a single malformed record used to escape into
            # the outer handler and return the WHOLE pool unsorted, silently
            # reverting triage to chronological order.
            try:
                score = (_clamp01(calibrated[str(getattr(t, "session_id", "") or "")])
                         if use_calibrated else shape_triage_score(t))
            except Exception as e:  # noqa: BLE001
                # Sorts LAST in a worst-first queue, which is the right place
                # for something that could not be assessed — but say so, or an
                # unparseable record silently reads as the safest turn in the
                # corpus.
                logger.debug("triage: record %r unscoreable (%s: %s) — ranked "
                             "last", getattr(t, "id", "?"), type(e).__name__, e)
                score = 0.0
            scored.append((score, i, t))
        # Sort on (-score, original index): explicit, so stability does not
        # depend on the sort algorithm's guarantees or on object identity.
        scored.sort(key=lambda row: (-row[0], row[1]))
        items = [row[2] for row in scored]
    except Exception as e:  # noqa: BLE001 — ranking must never break the loop
        logger.debug("triage ranking skipped: %s: %s", type(e).__name__, e)
        return items[:limit] if limit else items
    return items[:limit] if limit else items


__all__ = [
    "RiskReading", "turn_risk", "depth_failure_prior", "risk_steer_message",
    "steer_enabled", "triage_ranking_enabled", "EXPERIMENT",
    "ENV_STEER_KILL", "ENV_TRIAGE_KILL",
    "shape_triage_score", "triage_score", "rank_for_triage",
    "calibrated_risk_index",
]
