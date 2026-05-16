"""Pure weighting functions for frontier-aware self-play.

Self-play already picks clusters via ``FrontierTracker.pick_seed``, which
favours brittle clusters (recent failures + struggled wins) and skips
saturated ones. That signal works but only sees outcomes. It cannot
distinguish "the agent reliably aces this cluster" from "the PRM has
no opinion about this cluster yet because we've barely seen it" — both
look quiet to the brittle-pool scorer.

This module adds two complementary signals:

  uncertainty  — derived from ``PRMScorer.uncertainty(state, action)``.
                 High when the PRM scores a representative state for
                 the cluster near 0.5 (boundary). Zero when the model
                 is confident either way. 1.0 when the PRM is untrained
                 (we explicitly want to explore clusters the model has
                 no opinion on).

  rarity       — derived from trajectory counts: ``1 / (1 + log1p(n))``.
                 Smooth, bounded in (0, 1], decays slowly so a cluster
                 with 100 trajectories still gets non-negligible weight
                 if its PRM uncertainty is high.

The two are multiplied so a cluster needs BOTH "we don't know much
about it" AND "we don't have many examples of it" to win. Either
signal alone collapses the product toward zero, which is what we want —
mastered clusters (low uncertainty) and well-explored clusters (low
rarity) drop out automatically.

All functions are pure: no I/O, no globals, no logging side effects.
The caller (typically ``FrontierTracker.pick_frontier_seed``) wires
them together with the storage layer.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Tuple

from ..prm import ActionFeatures, PlanState, PRMScorer


def representative_state(cluster_key: str) -> Tuple[PlanState, ActionFeatures]:
    """Synthesise a (state, action) pair that stands in for "a fresh
    attempt at this cluster". The PRM's features are coarse enough that
    a templated request like ``"solve a {cluster} challenge"`` plus a
    blank action lands the prediction in the same neighbourhood as a
    real first-step PRM call would.

    We deliberately do NOT sample real trajectories here — that would
    couple the seed-picker to the trajectory store's read latency and
    re-introduce I/O into what's supposed to be a pure-math layer.
    Aggregated uncertainty over real trajectories is a future
    refinement; the templated state is good enough for the boundary-
    distance signal we actually use.
    """
    cluster = (cluster_key or "general").replace("_", " ")
    state = PlanState(
        user_request=f"solve a {cluster} challenge",
        steps_so_far=0,
        failures_so_far=0,
        pending_count=1,
        plan_depth=1,
        tools_used_this_turn=(),
        tools_failed_this_turn=(),
    )
    action = ActionFeatures(description="", tool_name="", tool_args={})
    return state, action


def compute_cluster_uncertainty(
    prm_scorer: Optional[PRMScorer],
    cluster_keys: Iterable[str],
) -> Dict[str, float]:
    """For each cluster, compute the PRM's uncertainty against a
    representative state. Returns a dict keyed by cluster_key.

    When ``prm_scorer`` is None, every cluster gets uncertainty 1.0 —
    so the combined weight reduces to rarity-only, which is itself a
    reasonable frontier signal. When the scorer has no trained model,
    the wrapped ``uncertainty()`` already returns 1.0 by contract;
    we pass through.
    """
    out: Dict[str, float] = {}
    for key in cluster_keys:
        if prm_scorer is None:
            out[key] = 1.0
            continue
        state, action = representative_state(key)
        try:
            out[key] = float(prm_scorer.uncertainty(state, action))
        except Exception:
            # Fail-safe: a scorer that can't score a cluster is
            # treated as maximum uncertainty so the cluster still gets
            # exploration weight, rather than silently disappearing.
            out[key] = 1.0
    return out


def compute_cluster_rarity(
    trajectory_counts: Dict[str, int],
    cluster_keys: Iterable[str],
) -> Dict[str, float]:
    """Per-cluster rarity in (0, 1]: ``1 / (1 + log1p(count))``.

    Counts come from iterating the trajectory store and grouping by
    ``Trajectory.cluster``. Clusters with no trajectories at all get
    rarity 1.0 (cold-start), one trajectory ≈ 0.59, ten ≈ 0.29, a
    hundred ≈ 0.18. The log decay means even well-explored clusters
    keep some weight — combined with low uncertainty they fall away
    naturally, but a high-uncertainty mastered-looking cluster can
    still get picked.
    """
    out: Dict[str, float] = {}
    for key in cluster_keys:
        n = max(0, int(trajectory_counts.get(key, 0)))
        out[key] = 1.0 / (1.0 + math.log1p(n))
    return out


def combine_weights(
    uncertainty: Dict[str, float],
    rarity: Dict[str, float],
    exclude: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Multiplicative combiner. Both signals must be present for a
    cluster to receive non-zero weight. Excluded clusters (typically
    saturated ones from ``FrontierTracker.list_saturated_clusters()``)
    get weight 0 — they're not deleted from the dict so the caller can
    log what was filtered out.

    The product is bounded in [0, 1] because each factor is. We
    intentionally don't renormalise; ``pick_weighted`` handles the
    sampling math and zero-weight entries naturally never get picked.
    """
    excluded = set(exclude or ())
    out: Dict[str, float] = {}
    keys = set(uncertainty.keys()) | set(rarity.keys())
    for key in keys:
        if key in excluded:
            out[key] = 0.0
            continue
        u = float(uncertainty.get(key, 0.0))
        r = float(rarity.get(key, 0.0))
        # Negative or NaN slips become 0; the combiner is not the place
        # to debug upstream signal generation.
        if not (math.isfinite(u) and math.isfinite(r)) or u < 0.0 or r < 0.0:
            out[key] = 0.0
        else:
            out[key] = u * r
    return out


def pick_weighted(
    weights: Dict[str, float],
    *,
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """Sample a single cluster_key in proportion to its weight.

    Returns None when every weight is zero (nothing to pick — caller
    should fall back to the existing cold-start path). Uses an injected
    ``rng`` so tests can pin behaviour deterministically.
    """
    r = rng or random
    items = [(k, w) for k, w in weights.items() if w > 0.0]
    if not items:
        return None
    total = sum(w for _, w in items)
    if total <= 0.0:
        return None
    pick = r.random() * total
    acc = 0.0
    for k, w in items:
        acc += w
        if pick <= acc:
            return k
    # Floating-point edge: pick == total. Return the last positive key.
    return items[-1][0]


def count_trajectories_by_cluster(
    trajectories: Iterable,
) -> Dict[str, int]:
    """Group an iterable of Trajectory objects by their ``cluster``
    field. Trajectories with a None or empty cluster are skipped (we
    only weight known clusters; an unclassified trajectory contributes
    no signal). Pure function — caller does the I/O of iterating the
    on-disk store.
    """
    counts: Dict[str, int] = {}
    for t in trajectories:
        key = getattr(t, "cluster", None)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts
