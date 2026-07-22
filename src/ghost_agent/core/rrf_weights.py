"""Learned RRF intent→source weights — load-if-present capability.

The memory bus fuses the retrieval tiers with a per-intent source-weight
matrix (`bus.MemoryBus._INTENT_WEIGHTS`) that is hand-tuned magic. This
module lets an OFFLINE-fitted matrix supersede those defaults at runtime,
using the same posture as the PRM / router / GEPA checkpoints: when a
fitted `weights.json` exists it is loaded and used; when absent, the bus
keeps its hand-tuned defaults — **zero behaviour change**. That fail-safe
default matters here more than anywhere else, because the fusion weights
sit on the hot retrieval path that feeds every turn; an ungrounded auto-
fit must never silently degrade recall.

`fit_intent_weights` derives a matrix from ``(intent, source, success)``
observations: a source that earns MORE than its fair share of a turn's
judged-used items — relative to the corpus base rate, not an assumed
coin flip — is up-weighted. Cells below a sample floor keep the base
weight, evidence-proportional shrinkage keeps a thin sample near the
prior, and a hard deviation clamp plus the global band mean a biased
sample can neither zero out, invert, nor explode a source.

Pure stdlib; JSON; fail-safe load (any problem → ``None`` → defaults).
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

logger = logging.getLogger("GhostAgent")

SCHEMA_VERSION = "ghost.rrf.weights.v1"

# Serialises the two writers of the observations ledger: the post-turn
# hydration judge APPENDS (bus.judge_hydration_usefulness, via to_thread)
# while the dream cycle's refit READS + TRIM-REWRITES (dream._refit_rrf_
# weights). Without it, lines appended between the refit's read and its
# os.replace were silently discarded (found 2026-07-15). Same process,
# different threads → a plain threading.Lock suffices.
LEDGER_LOCK = threading.Lock()

# Mirror of bus.MemoryBus._INTENT_WEIGHTS — the safe fallback AND the
# anchor a fit starts from (cells with too little data keep these).
DEFAULT_INTENT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "factual":    {"graph": 2.0, "vector": 1.0, "skill": 0.5, "episodic": 0.3, "session": 0.8},
    "procedural": {"graph": 0.5, "vector": 1.0, "skill": 2.0, "episodic": 1.5, "session": 0.5},
    "contextual": {"graph": 1.0, "vector": 1.5, "skill": 1.0, "episodic": 1.0, "session": 1.2},
}

WEIGHT_MIN = 0.1
WEIGHT_MAX = 3.0

# ---------------------------------------------------------------- calibration
# The fit is RELATIVE, not absolute (see fit_intent_weights). These constants
# decide how fast — and how far — evidence may pull a cell off its prior.

# Pseudo-observations pulled toward the fit's OWN base rate before the lift is
# taken. Kills the rate==0 blow-up (live 2026-07-22: factual/graph n=24 used=0
# → an infinite down-lift) and damps thin cells generally.
SMOOTHING_OBS = 10.0
# Same idea in the turn-aware estimator's unit (turns, not items).
SMOOTHING_TURNS = 2.0

# Evidence needed to move a cell HALFWAY from its prior toward what the data
# says, in the estimator's own sample unit. Per-ITEM observations arrive in
# correlated per-turn batches (~11.6 items share ONE judge verdict), so a raw
# item count massively overstates the evidence — hence the deliberately large
# obs figure vs the turn figure.
PRIOR_STRENGTH_OBS = 150.0
PRIOR_STRENGTH_TURNS = 25.0

# Compression applied to the lift before it scales the prior: adj = lift**GAMMA.
# 0.5 (sqrt) keeps the DIRECTION of a strong signal while halving its swing in
# log space — a tier that is used 2x its fair share earns ~1.41x, not 2x.
LIFT_GAMMA = 0.5

# Hard ceiling on how far ONE cell may travel from its hand-set prior, as a
# multiplier. Whatever the sample says, a 2.0-prior cell stays in [1.0, 3.0] —
# so a learned matrix can re-order tiers only mildly, never INVERT them.
MAX_DEVIATION = 2.0


def _clamp(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(v):
        return 1.0
    return WEIGHT_MIN if v < WEIGHT_MIN else WEIGHT_MAX if v > WEIGHT_MAX else v


def load_intent_weights(path) -> Optional[Dict[str, Dict[str, float]]]:
    """Read a fitted weight matrix, or ``None`` when absent / corrupt /
    wrong-schema (so the caller falls back to the hand-tuned defaults)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("rrf weights load failed: %s", e)
        return None
    if not isinstance(d, dict) or d.get("schema") != SCHEMA_VERSION:
        return None
    raw = d.get("weights")
    if not isinstance(raw, dict):
        return None
    # Deep-merge over the hand-tuned defaults: a truncated / hand-edited /
    # externally-produced file that is schema-valid but missing a source or
    # intent must NOT silently replace a tuned weight with the bus's 1.0
    # lookup fallback (found 2026-07-15).
    out: Dict[str, Dict[str, float]] = {i: dict(sw)
                                        for i, sw in DEFAULT_INTENT_WEIGHTS.items()}
    loaded_any = False
    for intent, sw in raw.items():
        if not isinstance(sw, dict):
            continue
        for src, val in sw.items():
            try:
                out.setdefault(str(intent), {})[str(src)] = _clamp(float(val))
                loaded_any = True
            except (TypeError, ValueError):
                continue
    return out if loaded_any else None


def _normalise_observation(obs) -> Optional[Tuple[str, str, bool, Optional[str]]]:
    """``(intent, source, success, turn|None)`` from a 3-tuple, a 4-tuple
    carrying the turn id, or a mapping (a raw ledger record). ``None`` for
    anything unusable — a malformed row must never abort a refit."""
    if isinstance(obs, dict):
        try:
            return (str(obs["intent"]), str(obs["source"]), bool(obs["success"]),
                    str(obs["turn"]) if obs.get("turn") else None)
        except (KeyError, TypeError, ValueError):
            return None
    try:
        intent, source, success = obs[0], obs[1], obs[2]
        turn = obs[3] if len(obs) > 3 else None
    except (TypeError, IndexError, KeyError):
        return None
    return (str(intent), str(source), bool(success),
            str(turn) if turn else None)


def _per_turn_lift(rows) -> Optional[Dict[Tuple[str, str], Tuple[float, int]]]:
    """Turn-normalised usefulness per (intent, source) cell.

    Returns ``{(intent, source): (lift, n_turns)}`` where ``lift`` is the
    tier's share of the turn's JUDGED-USED set divided by its share of that
    turn's INJECTED items, averaged over the turns that produced at least
    one used item. ``lift == 1.0`` means "earns exactly its fair share".

    This is the fix for the items-per-turn confound: measured live
    2026-07-22 over 130 turn-batches, hydration injects 11.58 items and the
    judge marks 1.65 used, but the per-source item counts are wildly uneven
    (vector 4.55/turn, graph 1.09, skill 0.92). A raw per-ITEM rate is then
    mechanically capped near 1.65/4.55 ≈ 0.36 for vector while a 0.92-item
    tier can reach 1.0 — so the old fit measured tier VERBOSITY, not
    usefulness. Shares are immune to that.

    ``None`` when any row lacks a turn id (mixed / legacy ledger) → the
    caller falls back to the pooled estimator."""
    turns: Dict[str, list] = {}
    for intent, source, success, turn in rows:
        if not turn:
            return None
        t = turns.setdefault(turn, [intent, {}, {}])
        t[1][source] = t[1].get(source, 0) + 1
        if success:
            t[2][source] = t[2].get(source, 0) + 1
    acc: Dict[Tuple[str, str], list] = {}
    for _turn, (intent, inj, used) in turns.items():
        used_total = sum(used.values())
        if used_total <= 0:
            continue  # uninformative turn: no credit to distribute
        inj_total = sum(inj.values()) or 1
        for source, cnt in inj.items():
            cell = acc.setdefault((intent, source), [0.0, 0.0, 0])
            cell[0] += used.get(source, 0) / used_total
            cell[1] += cnt / inj_total
            cell[2] += 1
    out: Dict[Tuple[str, str], Tuple[float, int]] = {}
    for key, (credit, share, n_turns) in acc.items():
        lift = ((credit + SMOOTHING_TURNS) / (share + SMOOTHING_TURNS)
                if (share + SMOOTHING_TURNS) > 0 else 1.0)
        out[key] = (lift, n_turns)
    return out


def _shrunk_weight(anchor: float, lift: float, evidence: float,
                   prior_strength: float) -> float:
    """Pull ``anchor`` toward what the data says, in proportion to how much
    data there is, and never past ``MAX_DEVIATION``."""
    lift = max(float(lift), 1e-6)
    evidence = max(float(evidence), 0.0)
    shrink = evidence / (evidence + prior_strength) if prior_strength > 0 else 1.0
    adj = lift ** (LIFT_GAMMA * shrink)
    adj = min(MAX_DEVIATION, max(1.0 / MAX_DEVIATION, adj))
    return _clamp(anchor * adj)


def fit_intent_weights(
    observations: Iterable[Tuple[str, str, bool]],
    *,
    base: Optional[Dict[str, Dict[str, float]]] = None,
    min_obs_per_cell: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Fit an intent→source weight matrix from ``(intent, source,
    success)`` observations — optionally ``(intent, source, success,
    turn)`` or raw ledger mappings, which unlock the turn-normalised
    estimator (see ``_per_turn_lift``).

    The fit is RELATIVE and MULTIPLICATIVE on the hand-set prior:

        weight = prior * clamp(lift ** (GAMMA * shrink))

    where ``lift`` is the cell's usefulness relative to the fit's OWN
    empirical base rate (turn-share based when turn ids are present), and
    ``shrink = n / (n + PRIOR_STRENGTH)`` so a thin sample barely moves the
    prior at all. ``lift == 1`` (the cell performs exactly like the average
    injected item) reproduces the prior EXACTLY.

    Calibrating against the observed base rate instead of an assumed 0.5 is
    the 2026-07-22 fix. The old curve read "rate 0.5 → base weight", but the
    real judged-used rate is ~0.14, so EVERY well-sampled cell landed on the
    lower branch and was crushed toward ``WEIGHT_MIN``: live weights.json
    had factual/graph at 0.1 (n=24, used=0) — a 6.6x INVERSION of its 2.0
    prior, leaving the graph at 0.3x vector on factual queries — and
    procedural/skill at 0.311, BELOW session's 0.424. The matrix had
    collapsed into noise. With the new calibration the same live counts give
    factual/graph ≈ 1.84 and procedural/skill ≈ 1.77: still the dominant
    tier for their intent, mildly damped by weak evidence.

    Degenerate samples (zero successes, or all successes) carry no contrast
    between tiers and are refused outright — the priors stand.

    ``min_obs_per_cell`` defaults to 20; cells below the floor keep the
    ``base`` weight untouched. Always returns a full matrix (base merged)."""
    base = base or DEFAULT_INTENT_WEIGHTS
    out: Dict[str, Dict[str, float]] = {i: dict(sw) for i, sw in base.items()}

    rows = [r for r in (_normalise_observation(o) for o in observations) if r]
    if not rows:
        return out
    n_total = len(rows)
    wins_total = sum(1 for r in rows if r[2])
    if wins_total == 0 or wins_total == n_total:
        # No contrast: every tier looks identical, so there is nothing to
        # learn. Refusing beats crushing (or exploding) the whole matrix.
        logger.debug("rrf fit skipped: degenerate sample (%d/%d used)",
                     wins_total, n_total)
        return out
    base_rate = wins_total / n_total

    agg: Dict[Tuple[str, str], list] = {}
    for intent, source, success, _turn in rows:
        cell = agg.setdefault((intent, source), [0, 0])
        cell[0] += 1
        cell[1] += 1 if success else 0

    per_turn = _per_turn_lift(rows)

    for (intent, source), (n, wins) in agg.items():
        if n < max(1, int(min_obs_per_cell)):
            continue
        anchor = _clamp(out.get(intent, {}).get(source, 1.0))
        cell_turns = per_turn.get((intent, source)) if per_turn else None
        if cell_turns:
            lift, evidence = cell_turns
            strength = PRIOR_STRENGTH_TURNS
        else:
            # Pooled fallback (legacy ledger without turn ids): the cell's
            # smoothed used-rate against the corpus base rate. Same "lift
            # 1 → prior" contract, coarser evidence unit.
            rate = ((wins + SMOOTHING_OBS * base_rate)
                    / (n + SMOOTHING_OBS))
            lift = rate / base_rate
            evidence, strength = float(n), PRIOR_STRENGTH_OBS
        out.setdefault(intent, {})[source] = round(
            _shrunk_weight(anchor, lift, evidence, strength), 3)
    return out


def save_intent_weights(path, weights: Dict[str, Dict[str, float]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"schema": SCHEMA_VERSION, "weights": weights}, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, p)


__all__ = [
    "DEFAULT_INTENT_WEIGHTS",
    "LEDGER_LOCK",
    "load_intent_weights",
    "fit_intent_weights",
    "save_intent_weights",
    "SCHEMA_VERSION",
]
