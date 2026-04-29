"""Per-step label derivation from trajectory terminal outcomes.

The PRM doesn't see hand-annotated labels. It mines them from the
trajectories the agent already records, using the trick AlphaZero uses
to credit-assign through a winning rollout: Monte Carlo value backprop.

Algorithm:

  Given a trajectory with N tool calls and terminal outcome ``o``:
    * ``o == PASSED``  → terminal value 1.0
    * ``o == FAILED``  → terminal value 0.0
    * ``o == UNKNOWN`` → trajectory is skipped (no useful gradient)
  Then for each step ``i`` (0-indexed from the start of the
  trajectory), the per-step value is::

        V(step_i) = γ^(N-i-1) * terminal_value

  With γ = 0.9 and N = 4 steps that gives steps in the PASSED case
  values ``[0.729, 0.81, 0.9, 1.0]`` — the step right before the win
  gets full credit; earlier steps get exponentially less. In the
  FAILED case all step values are 0 (every step in a losing rollout
  is a counterexample).

The continuous value is what training code consumes. Binary callers
(plain logistic regression on 0/1 labels) threshold via
``label_step_value_binary`` — by default the threshold is 0.5, which
with γ=0.9 means the last ~6 steps of a PASSED trajectory cross to
positive. That window is well-matched to typical agent turns.

Trajectories with no tool calls (UNKNOWN included) yield zero step
samples — they contribute nothing to training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from ..distill.schema import Outcome, Trajectory, ToolCall
from .features import (
    ActionFeatures,
    PlanState,
    extract_step_features,
    FeatureVector,
)


# ──────────────────────────────────────────────────────────────────────
# Spec
# ──────────────────────────────────────────────────────────────────────

@dataclass
class StepLabelSpec:
    """Tunable knobs for label derivation. Exposed so tests can sweep
    without monkey-patching the module."""

    # γ in V(step_i) = γ^(N-i-1) * terminal_value. Closer to 1.0 →
    # earlier steps get more credit; closer to 0 → only the final
    # step matters. 0.9 is the AlphaZero default and works well at
    # the small step-counts a tool-using agent produces.
    discount_factor: float = 0.9

    # Minimum samples required per step to keep a trajectory. 0 means
    # we accept any non-empty trajectory; a higher floor prunes
    # super-short turns that contribute noise.
    min_steps: int = 1

    # Decision threshold for the binary view. 0.5 means "the step is
    # within ~log(0.5)/log(γ) of the terminal" — see derive_step_labels
    # docstring for the math.
    binary_threshold: float = 0.5

    # When True, include FAILED trajectories. Setting to False trains
    # on PASSED-only data — useful for sanity checks but biases the
    # model toward predicting success everywhere. Default True.
    include_failed: bool = True


# ──────────────────────────────────────────────────────────────────────
# Public dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass
class StepSample:
    """One training sample = (state, action, value) extracted from a
    single ``ToolCall`` inside a trajectory.

    ``trajectory_id`` is preserved so a downstream filter can drop
    samples whose source trajectory was later marked FAILED via the
    corrections sidecar (the iter pipeline already overlays those, but
    a caller might want to apply additional provenance rules)."""

    state: PlanState
    action: ActionFeatures
    value: float                 # continuous, ∈ [0, 1]
    binary: int                  # thresholded, ∈ {0, 1}
    trajectory_id: str = ""
    step_index: int = 0
    terminal_outcome: str = Outcome.UNKNOWN.value


# ──────────────────────────────────────────────────────────────────────
# Label derivation
# ──────────────────────────────────────────────────────────────────────

def derive_step_labels(
    traj: Trajectory,
    spec: Optional[StepLabelSpec] = None,
) -> List[float]:
    """Return one continuous value per ``ToolCall`` in ``traj``.

    Length of the returned list is exactly ``len(traj.tool_calls)``.
    UNKNOWN-outcome trajectories (or empty trajectories) yield ``[]``.
    """
    spec = spec or StepLabelSpec()
    calls = list(traj.tool_calls or ())
    if len(calls) < max(1, spec.min_steps):
        return []

    outcome = (traj.outcome or "").lower()
    if outcome == Outcome.UNKNOWN.value:
        return []
    if outcome == Outcome.FAILED.value and not spec.include_failed:
        return []

    terminal = 1.0 if outcome == Outcome.PASSED.value else 0.0
    n = len(calls)
    # Coerce gamma defensively: NaN / non-finite must not produce NaN
    # labels. Out-of-range values are clamped into [0, 1] (values
    # outside that interval would push V outside the binary threshold).
    try:
        gamma = float(spec.discount_factor)
    except (TypeError, ValueError):
        gamma = 0.9  # spec default
    import math as _math
    if not _math.isfinite(gamma):
        gamma = 0.9
    if gamma < 0.0:
        gamma = 0.0
    if gamma > 1.0:
        gamma = 1.0

    out: List[float] = []
    for i in range(n):
        # i = 0 is the FIRST step in the trajectory. The exponent
        # (N-i-1) goes N-1 → 0 as i increases, so the LAST step gets
        # γ^0 = 1.0 and the first gets γ^(N-1).
        weight = gamma ** (n - i - 1)
        out.append(float(weight * terminal))
    return out


def label_step_value_binary(
    value: float,
    threshold: float = 0.5,
) -> int:
    """Threshold a continuous step value into a binary classification
    label suitable for plain logistic regression.

    Note: thresholding loses information (the discount-encoded notion
    of how-close-to-success the step was). Callers training a
    regression-style model should use the continuous value directly
    and skip this function."""
    return 1 if float(value) >= float(threshold) else 0


# ──────────────────────────────────────────────────────────────────────
# Sample iteration
# ──────────────────────────────────────────────────────────────────────

def _build_state_for_step(
    traj: Trajectory,
    step_index: int,
) -> PlanState:
    """Reconstruct the agent's *prefix-state* immediately before tool
    call ``step_index`` fired.

    "Prefix" matters: when training, we want the PRM to learn "given
    what was known when the agent CHOSE this step, would this step
    succeed?". Building the state from the full trajectory (including
    later steps) would leak information about the future and bias the
    model toward post-hoc plausibility.
    """
    calls = list(traj.tool_calls or ())
    prior = calls[:step_index]
    used = tuple(
        (tc.name or "").strip() for tc in prior if (tc.name or "").strip()
    )
    failed = tuple(
        (tc.name or "").strip()
        for tc in prior
        if (tc.name or "").strip() and (tc.error or "").strip()
    )
    return PlanState(
        user_request=traj.user_request or "",
        steps_so_far=int(step_index),
        failures_so_far=len(failed),
        pending_count=max(0, len(calls) - step_index - 1),
        plan_depth=int(traj.n_steps or 0),
        tools_used_this_turn=used,
        tools_failed_this_turn=failed,
    )


def _build_action_for_step(call: ToolCall) -> ActionFeatures:
    args = call.arguments if isinstance(call.arguments, dict) else {}
    # Use the call's result (if present) as a description proxy: the
    # plan candidate's "description" slot is what the agent intended
    # to do, and a tool-call's args are usually the closest
    # description we have post-hoc. Truncate to a reasonable bound.
    desc = ""
    # Prefer 'description' / 'goal' / 'intent' arg keys when callers
    # emit them; otherwise fall back to a join of arg values.
    for key in ("description", "goal", "intent", "summary"):
        v = args.get(key)
        if isinstance(v, str) and v:
            desc = v[:400]
            break
    if not desc:
        for v in args.values():
            if isinstance(v, str) and v:
                desc = v[:400]
                break
    return ActionFeatures(
        description=desc,
        tool_name=(call.name or "").strip(),
        tool_args=args,
    )


def iter_step_samples(
    trajectories: Iterable[Trajectory],
    spec: Optional[StepLabelSpec] = None,
) -> Iterator[StepSample]:
    """Stream ``StepSample`` objects from a trajectory iterable.

    UNKNOWN trajectories and empty trajectories are silently skipped.
    The iteration is lazy so a corpus that doesn't fit in memory is
    fine — callers can stream straight into ``StepValueModel.fit``.
    """
    spec = spec or StepLabelSpec()
    for traj in trajectories:
        labels = derive_step_labels(traj, spec)
        if not labels:
            continue
        calls = list(traj.tool_calls or ())
        for i, (call, value) in enumerate(zip(calls, labels)):
            state = _build_state_for_step(traj, i)
            action = _build_action_for_step(call)
            binary = label_step_value_binary(value, spec.binary_threshold)
            yield StepSample(
                state=state,
                action=action,
                value=float(value),
                binary=int(binary),
                trajectory_id=str(traj.id or ""),
                step_index=int(i),
                terminal_outcome=str(traj.outcome or Outcome.UNKNOWN.value),
            )


def class_balance(samples: Sequence[StepSample]) -> dict:
    """Summary helper. Useful before training to catch severe imbalance
    (e.g. when ``include_failed=False`` would yield 100% positive)."""
    n_pos = sum(1 for s in samples if s.binary == 1)
    n_neg = sum(1 for s in samples if s.binary == 0)
    total = n_pos + n_neg
    return {
        "positive": n_pos,
        "negative": n_neg,
        "total": total,
        "positive_ratio": (n_pos / total) if total else 0.0,
    }
