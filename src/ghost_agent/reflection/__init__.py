"""Ghost Agent reflection / self-critique loop.

The fourth biological phase. Sits between dream (Phase 2) and
synthetic self-play (Phase 3): when the agent has been idle long
enough for REM consolidation to complete but not long enough to
warrant fresh self-play, it pulls recent failed trajectories from the
trajectory log and re-runs them through a self-critique prompt.

Mechanics, mirrored on the dream phase:

  * Per-phase cooldown anchor (`_last_reflection_at`).
  * Cooldown set BEFORE the `await` and re-affirmed in `finally` so
    an exception can't leave the anchor un-advanced (same pattern as
    dream and self-play — see CLAUDE.md "Biological watchdog cooldown
    anchors").
  * Activity clock is NOT reset by this phase (user idleness drives
    the watchdog — same rule as Phase 1 and Phase 2).
  * Idempotency: a trajectory is reflected on AT MOST once — a cache
    of reflected trajectory ids lives on the context.

Reflection produces (failed attempt, reflected plan) pairs and writes
them back to the trajectory log. Downstream, these pairs are the
premium training data for Stage 2 rejection-sample SFT: the trajectory
shape where a reflected path succeeds while the original failed.
"""

from .prompts import REFLECTION_PROMPT_TEMPLATE, build_reflection_prompt
from .loop import Reflector, ReflectionOutcome, ReflectionRunReport
from .postmortem import (
    PostMortemEngine,
    PostMortemRunReport,
    DefectReport,
    DefectQueue,
    TranscriptSignature,
    compute_signature,
    select_failed_runs,
)

__all__ = [
    "REFLECTION_PROMPT_TEMPLATE",
    "build_reflection_prompt",
    "Reflector",
    "ReflectionOutcome",
    "ReflectionRunReport",
    "PostMortemEngine",
    "PostMortemRunReport",
    "DefectReport",
    "DefectQueue",
    "TranscriptSignature",
    "compute_signature",
    "select_failed_runs",
]
