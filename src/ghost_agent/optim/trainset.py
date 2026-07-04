"""Training-set construction from trajectory logs.

The optimizer needs (input, expected_output) pairs. Trajectories give us
that for free: the user_request is the input, the validator-approved
final_response (or pass-labeled self-consistency sample) is the expected
output. We filter HARD on outcome — only `Outcome.PASSED` survives —
because feeding GEPA failed trajectories defeats the purpose of an
optimizer.

For self-consistency batches, we additionally dedupe by `batch_id` and
keep the lowest-temperature pass (the most conservative, usually the
most faithful to the instruction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from ..distill.schema import Trajectory, Outcome


@dataclass
class TrainExample:
    """One (input, output) pair for a signature.

    `signature_name` ties it to a specific OptimizableSignature — an
    optimizer run is *always scoped to one signature at a time* so we
    never cross-contaminate planning examples into tool-selection
    training and vice versa.
    """

    signature_name: str
    inputs: Dict[str, str] = field(default_factory=dict)
    expected_output: Dict[str, str] = field(default_factory=dict)
    source_trajectory_id: str = ""
    weight: float = 1.0  # for class imbalance / source weighting


def filter_by_outcome(
    trajectories: Iterable[Trajectory],
    *,
    require_passed: bool = True,
    min_steps: int = 0,
    max_steps: Optional[int] = None,
) -> List[Trajectory]:
    """Drop trajectories that fail the outcome / shape filters.

    Defaults: keep only PASSED. `min_steps` can exclude trivial turns
    (plain "hi" echoed back — no signal for planning); `max_steps`
    excludes abnormally long sessions that may reflect a stuck run.
    """
    kept: List[Trajectory] = []
    for t in trajectories:
        if require_passed and t.outcome != Outcome.PASSED.value:
            continue
        n = int(t.n_steps or 0)
        if n < min_steps:
            continue
        if max_steps is not None and n > max_steps:
            continue
        kept.append(t)
    return kept


def _dedupe_self_consistency(trajectories: List[Trajectory]) -> List[Trajectory]:
    """For each batch_id, keep the lowest-temperature passing sample.
    Non-batch trajectories pass through unchanged."""
    # Group by batch_id (None → unique key)
    groups: Dict[str, List[Trajectory]] = {}
    loose: List[Trajectory] = []
    for t in trajectories:
        if t.batch_id:
            groups.setdefault(t.batch_id, []).append(t)
        else:
            loose.append(t)

    chosen: List[Trajectory] = list(loose)
    for bid, members in groups.items():
        # Sort by (temperature, sample_index) — lowest temp wins.
        members_sorted = sorted(
            members,
            key=lambda x: (float(x.temperature or 0.0),
                           int(x.sample_index or 0)),
        )
        chosen.append(members_sorted[0])
    return chosen


def build_trainset(
    trajectories: Iterable[Trajectory],
    signature_name: str,
    *,
    require_passed: bool = True,
    min_steps: int = 0,
    max_steps: Optional[int] = None,
    max_examples: Optional[int] = None,
) -> List[TrainExample]:
    """Build a TrainExample list for `signature_name`.

    The signature's `inputs`/`outputs` metadata could in principle drive
    a custom extraction, but we keep it simple: every signature gets
    the trajectory's `user_request` as primary input and `final_response`
    as primary output. Signature-specific shaping is the GEPA optimizer's
    job, not ours.
    """
    kept = filter_by_outcome(
        trajectories,
        require_passed=require_passed,
        min_steps=min_steps,
        max_steps=max_steps,
    )
    deduped = _dedupe_self_consistency(kept)

    examples: List[TrainExample] = []
    for t in deduped:
        if not t.user_request or not t.final_response:
            continue
        examples.append(TrainExample(
            signature_name=signature_name,
            inputs={
                "user_request": t.user_request,
                # Optional secondary inputs — not every signature uses them,
                # but carrying them through is harmless.
                "cluster": t.cluster or "",
                "tier": t.tier or "",
            },
            expected_output={
                "final_response": t.final_response,
                "plan": t.planning_output or "",
            },
            source_trajectory_id=t.id,
            weight=1.0,
        ))
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def split_train_eval(
    examples: List[TrainExample],
    *,
    eval_fraction: float = 0.2,
    random_state: int = 0,
) -> Tuple[List[TrainExample], List[TrainExample]]:
    """Deterministic split into (train, eval) for holdout scoring.

    Shuffles with a seeded RNG so the split is reproducible across
    optimizer runs — critical for A/B comparability.
    """
    if eval_fraction <= 0.0 or not examples:
        return list(examples), []
    if eval_fraction >= 1.0:
        return [], list(examples)

    import random
    rng = random.Random(random_state)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_fraction))
    # Never let the holdout consume the ENTIRE corpus. With a single example
    # `max(1, int(1*0.2))` is 1, which put the only example in eval and left
    # `train_set` EMPTY — so GEPA would then "optimize" on nothing. Keep at
    # least one example in train (a 1-example corpus → train=[it], eval=[]).
    n_eval = min(n_eval, len(shuffled) - 1)
    eval_set = shuffled[:n_eval]
    train_set = shuffled[n_eval:]
    return train_set, eval_set
