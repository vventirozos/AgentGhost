"""Ghost Agent trajectory logging + self-consistency sampling.

Local-only corpus of (prompt, plan, tool_calls, final_response, outcome)
tuples. Two uses:

  1. Passive: every real user turn / self-play round gets logged (after
     redaction) to `$GHOST_HOME/trajectories/YYYY-MM-DD/*.jsonl`.
     This corpus is the input for Stage 2 rejection-sample SFT — the
     pipeline builds a dataset without ever having run a data-collection
     campaign.

  2. (RETIRED 2026-07-27) `self_consistency.sample()` — the offline
     rejection-sampling sampler. Flagged INERT by learning-health
     telemetry (no production OR offline caller anywhere in the tree;
     the arbiter reimplements the dual-sample pattern separately) and
     removed. `optim.trainset._dedupe_self_consistency` stays: old
     trajectory corpora may still carry its batch_ids.

Redaction runs on every write. No trajectory leaves this machine.
"""

from .schema import Trajectory, ToolCall, Outcome
from .redact import redact_text, redact_trajectory, RedactionConfig
from .collector import TrajectoryCollector
from .outcome_heuristics import (
    classify_chat_outcome,
    apply_chat_outcome_heuristics,
    FailureClassification,
)

__all__ = [
    "Trajectory",
    "ToolCall",
    "Outcome",
    "redact_text",
    "redact_trajectory",
    "RedactionConfig",
    "TrajectoryCollector",
    "classify_chat_outcome",
    "apply_chat_outcome_heuristics",
    "FailureClassification",
]
