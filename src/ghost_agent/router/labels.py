"""Label derivation from trajectories.

The router never sees human-annotated labels — it mines them from
trajectory outcomes the agent already recorded. The policy encoded here
is:

    * Mark a trajectory "easy" when it resolved quickly with minimal
      tooling: few steps, zero-to-one tool calls, no coding / browser
      /visual pool escalation.
    * Mark it "hard" when ANY of the following fired: plan produced
      with >3 steps, ≥4 tool calls, a coding/browser/visual tool was
      used, or the validator only passed after multi-step reflection.
    * Return None for ambiguous middle-ground trajectories — those
      just don't get used as training data. A sparse label set beats
      noisy labels for a fail-safe classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from ..distill.schema import Trajectory


_HEAVYWEIGHT_TOOLS = frozenset({
    "browser",
    "execute",
    "image_gen",
    "delegate_to_swarm",
    "deep_research",
    "vision",
    "knowledge_base",
})


@dataclass
class LabelSpec:
    """Thresholds for easy/hard derivation. Exposed so tests (and the
    baseline calibration step) can sweep them without monkey-patching."""

    # Easy ceiling — above any of these, the trajectory can't be "easy"
    easy_max_steps: int = 2
    easy_max_tool_calls: int = 1
    # Hard floor — meeting any of these flips the label to "hard"
    hard_min_steps: int = 4
    hard_min_tool_calls: int = 4
    heavyweight_tools: frozenset = field(default_factory=lambda: _HEAVYWEIGHT_TOOLS)


def derive_label(traj: Trajectory, spec: Optional[LabelSpec] = None) -> Optional[str]:
    """Return 'easy', 'hard', or None (ambiguous).

    None trajectories are dropped from training rather than guessed at;
    training on ambiguous-but-confidently-labeled data has killed more
    routers than data scarcity ever has.
    """
    s = spec or LabelSpec()

    # Failed outcomes are strong "hard" signals: if the agent couldn't
    # even finish the task under the current config, something harder
    # is going on. (We count runtime error failures the same as
    # validator-fail — both mean the task isn't easy.)
    from ..distill.schema import Outcome
    if traj.outcome == Outcome.FAILED.value:
        return "hard"

    steps = int(traj.n_steps or 0)
    n_calls = len(traj.tool_calls or ())
    heavy_used = any(
        (tc.name or "").strip() in s.heavyweight_tools
        for tc in (traj.tool_calls or ())
    )

    # Hard criteria (any one trips the label).
    if heavy_used:
        return "hard"
    if steps >= s.hard_min_steps:
        return "hard"
    if n_calls >= s.hard_min_tool_calls:
        return "hard"

    # Easy criteria (all must hold).
    if steps <= s.easy_max_steps and n_calls <= s.easy_max_tool_calls:
        return "easy"

    return None


def label_trajectories(
    trajectories: Iterable[Trajectory],
    spec: Optional[LabelSpec] = None,
) -> List[Tuple[Trajectory, str]]:
    """Filter & label: returns (traj, label) for non-ambiguous trajectories
    only."""
    spec = spec or LabelSpec()
    out: List[Tuple[Trajectory, str]] = []
    for t in trajectories:
        label = derive_label(t, spec)
        if label is None:
            continue
        out.append((t, label))
    return out


def class_balance(labels: Sequence[str]) -> dict:
    """Summary of the easy/hard mix; useful before training to catch
    severe class imbalance early."""
    c_easy = sum(1 for l in labels if l == "easy")
    c_hard = sum(1 for l in labels if l == "hard")
    total = c_easy + c_hard
    return {
        "easy": c_easy,
        "hard": c_hard,
        "total": total,
        "hard_ratio": (c_hard / total) if total else 0.0,
    }
