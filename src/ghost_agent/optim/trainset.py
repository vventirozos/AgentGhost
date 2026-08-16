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
from typing import Dict, Iterable, List, Optional, Tuple, Sequence

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
    # §4BF 1c: which POPULATION the example came from (the trajectory's
    # task_kind — "user_request" or "bench"). Carried as an explicit
    # FIELD, not an input: inputs feed the tuned prompt, and the origin
    # must stay separable for audits/per-origin eval without teaching the
    # model to condition its text on a token real traffic never carries.
    # Deliberately EXCLUDED from `_example_identity` — identity decides
    # holdout membership, which must never reshuffle over a metadata field.
    origin: str = "user_request"


# Only REAL user turns are gold. A `reflection` trajectory's `final_response`
# is a DIAGNOSIS-and-revised-plan block, and its outcome is upgraded to PASSED
# by the plan judge — so without this filter the optimizer was being taught
# that the correct answer to "deep research X" is a critique of a previous
# attempt. Re-measured on the live corpus 2026-08-04: 96 of 506 PASSED
# trajectories (19.0%) are reflection records. `core/experiments.py` and
# `optim/tool_ontology.py` both already filter on task_kind; this module and
# learning_health did not.
#
# This tuple is the DEFAULT of `filter_by_outcome` (506 -> 410 today) and is
# what analysis callers get. `build_trainset` deliberately passes
# `task_kinds=None` and filters per FIELD instead — see its docstring. So the
# shipped behaviour is NOT the "505 -> 409 whole-record filter" the §4J
# write-up describes; that was the first, reverted fix.
#
# §4BF 1c (admissibility: gepa_trainset = bench_feature): "bench" joins the
# gold kinds. HONEST SCOPE (R3 review corrected the first version of this
# comment): the oracle graded the attempt's solution.py exit code, NOT the
# final_response — a bench row's outcome=passed (via the oracle write-back
# overlay) proves the SOLVE was correct, while its final_response is the
# solver's closing reply on that oracle-passed run (typically restating the
# solution). That is weaker than "oracle-graded output" but stronger than
# an unverified turn; the input side is clean (bench rows record the bank
# challenge text via the user_request override), failed attempts are
# excluded by require_passed + the overlay, and the real-only private gate
# means no promotion is ever judged on these rows. The reflection exclusion
# that motivated this tuple is unchanged. Every bench example carries
# `origin="bench"` (see TrainExample) so it stays separable.
_GOLD_TASK_KINDS: Tuple[str, ...] = ("user_request", "bench")


def filter_by_outcome(
    trajectories: Iterable[Trajectory],
    *,
    require_passed: bool = True,
    min_steps: int = 0,
    max_steps: Optional[int] = None,
    task_kinds: Optional[Sequence[str]] = _GOLD_TASK_KINDS,
) -> List[Trajectory]:
    """Drop trajectories that fail the outcome / shape / kind filters.

    Defaults: keep only PASSED **user turns**. `min_steps` can exclude
    trivial turns (plain "hi" echoed back — no signal for planning);
    `max_steps` excludes abnormally long sessions that may reflect a stuck
    run. Pass ``task_kinds=None`` to keep every kind (analysis only — never
    for building gold).
    """
    kept: List[Trajectory] = []
    for t in trajectories:
        if require_passed and t.outcome != Outcome.PASSED.value:
            continue
        if task_kinds is not None:
            if str(getattr(t, "task_kind", "") or "") not in task_kinds:
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
    # PER-FIELD kind filtering, not per-record. The thing that poisons a
    # trainset is a reflection's `final_response` — a DIAGNOSIS block taught
    # as the answer to the original request. Its `planning_output` is a
    # revised PLAN, which is exactly what a planning signature wants.
    #
    # Filtering whole records instead cost 100% of the plan targets:
    # `planning_output` is populated ONLY on reflection trajectories (157 of
    # 157 on the live corpus), so `planning.decompose` went from 96 keyed
    # examples to 0 and silently fell back to grading whole final replies as
    # "the plan" — a worse failure than the one being fixed, and one that
    # printed no message at all because run_gepa's `elif keyed:` branch is
    # also skipped at zero.
    kept = filter_by_outcome(
        trajectories,
        require_passed=require_passed,
        min_steps=min_steps,
        max_steps=max_steps,
        task_kinds=None,
    )
    deduped = _dedupe_self_consistency(kept)

    examples: List[TrainExample] = []
    for t in deduped:
        if not t.user_request or not t.final_response:
            continue
        _is_gold = str(getattr(t, "task_kind", "") or "") in _GOLD_TASK_KINDS
        _plan = t.planning_output or ""
        # A record that contributes NEITHER a trustworthy final_response nor
        # a plan carries no target at all.
        if not _is_gold and not _plan:
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
                "final_response": t.final_response if _is_gold else "",
                "plan": _plan,
            },
            source_trajectory_id=t.id,
            weight=1.0,
            origin=(str(getattr(t, "task_kind", "") or "") or "user_request"),
        ))
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def per_origin_selection(
    examples: List[TrainExample],
    max_examples: Optional[int] = None,
) -> Tuple[List[TrainExample], List[TrainExample]]:
    """§4BF 1c corpus selection: returns ``(real, bench)`` after the
    equal-mass bench cap and the total cap.

    Replaces plain head-truncation (R1 review): ``examples[:max]`` iterated
    real-root examples first, so on a healthy corpus every bench example
    was silently dropped right after "admitting N bench trajectories"
    printed — and on a thin corpus bench flooded unchecked. Rules:
    bench ≤ len(real) (newest bench first); under a total cap real takes
    priority but bench keeps a reserve of up to half the cap, so an
    admission that survives the equal-mass cap cannot be silently zeroed
    by the total cap.
    """
    real = [e for e in examples if getattr(e, "origin", "") != "bench"]
    bench = [e for e in examples if getattr(e, "origin", "") == "bench"]
    bench = bench[-len(real):] if real else []
    if max_examples is not None:
        cap = max(0, int(max_examples))
        real_take = min(len(real), max(cap - len(bench), (cap + 1) // 2))
        real = real[:real_take]
        rem = max(0, cap - len(real))
        bench = bench[-rem:] if rem else []
    return real, bench


def real_only_gate(
    public: List[TrainExample],
    private: List[TrainExample],
) -> Tuple[List[TrainExample], List[TrainExample], int]:
    """§4BF 1c (R1 review CRIT): the PRIVATE ship-gate tier is REAL-ONLY.

    ``holdout_tier`` hashes identity with origin deliberately excluded
    (holdout stability), so bench examples land in the private tier —
    the A/B promotion would then be graded partly against bench-style
    outputs, and bench volume could satisfy the resolution check on a
    real tier too coarse to resolve the delta. Doctrine: bench may TEACH
    (public tier), it may never GRADE. Returns
    ``(public, private, n_moved)`` with bench private examples moved to
    the public side.

    The move then RE-CAPS the public tier at equal mass (R2 review:
    moving 100% of bench into a tier holding only ~70% of real quietly
    inverted the corpus-level equal-mass cap — public went bench-majority
    right after the run printed "equal-mass cap"). Newest bench rows win,
    consistent with every other cap in the 1c stack.
    """
    bench_priv = [e for e in private if getattr(e, "origin", "") == "bench"]
    if not bench_priv:
        return list(public), list(private), 0
    new_public = list(public) + bench_priv
    real_pub = [e for e in new_public if getattr(e, "origin", "") != "bench"]
    bench_pub = [e for e in new_public if getattr(e, "origin", "") == "bench"]
    if len(bench_pub) > len(real_pub):
        bench_pub = bench_pub[-len(real_pub):] if real_pub else []
        new_public = real_pub + bench_pub
    return (
        new_public,
        [e for e in private if getattr(e, "origin", "") != "bench"],
        len(bench_priv),
    )


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


def _example_identity(ex: TrainExample) -> str:
    """Stable identity for holdout assignment. `source_trajectory_id` when
    present (same trajectory → same tier across ALL signatures, so a task the
    optimizer saw for planning can never grade tool-selection); otherwise a
    content key. Deliberately excludes `signature_name` for id-bearing
    examples for exactly that cross-signature reason."""
    if ex.source_trajectory_id:
        return f"traj:{ex.source_trajectory_id}"
    return "content:" + "|".join(
        f"{k}={ex.inputs.get(k, '')}" for k in sorted(ex.inputs)
    )


def holdout_tier(identity: str, *, private_pct: int = 30) -> str:
    """Deterministic PUBLIC/PRIVATE bucket for one example identity.

    sha256-based so membership depends ONLY on the identity — never on
    corpus size, ordering, or an RNG seed. The seeded-shuffle split above
    re-deals membership whenever the corpus grows, which leaks: an example
    the optimizer trained on in run N can land in the "holdout" of run N+1.
    Hash membership is forever, which is the property the ship-gate needs.
    """
    import hashlib
    pct = max(0, min(100, int(private_pct)))
    bucket = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "private" if bucket < pct else "public"


def split_public_private(
    examples: List[TrainExample],
    *,
    private_pct: int = 30,
) -> Tuple[List[TrainExample], List[TrainExample]]:
    """Partition into (public, private) by per-item hash — see `holdout_tier`.

    PUBLIC is everything the optimizer may touch (its train set AND its
    internal validation split). PRIVATE is reserved for the A/B ship-gate
    and must never be passed to the optimizer. Order within each tier is
    preserved. If hashing would leave the public side empty while examples
    exist, the first private example is reassigned — an optimizer with no
    train data "optimizes" nothing, and a starved run is worse than a
    marginally smaller holdout.
    """
    public: List[TrainExample] = []
    private: List[TrainExample] = []
    for ex in examples:
        if holdout_tier(_example_identity(ex), private_pct=private_pct) == "private":
            private.append(ex)
        else:
            public.append(ex)
    if examples and not public:
        public.append(private.pop(0))
    return public, private
