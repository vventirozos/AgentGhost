"""Ghost Agent trajectory logging + self-consistency sampling.

Local-only corpus of (prompt, plan, tool_calls, final_response, outcome)
tuples. Two uses:

  1. Passive: every real user turn / self-play round gets logged (after
     redaction) to `$GHOST_HOME/trajectories/YYYY-MM-DD/*.jsonl`.
     This corpus is the input for Stage 2 rejection-sample SFT — the
     pipeline builds a dataset without ever having run a data-collection
     campaign.

  2. Offline-only: `self_consistency.sample()` runs the same prompt N
     times at varied temperatures, uses the validator (when available) to
     label each sample pass/fail, and writes the labeled batch to the
     trajectory store. That's the rejection-sampling corpus: (failed
     attempt, successful attempt) pairs from the same model. NOTE: this is
     invoked by the offline GEPA/training tooling, NOT by the live agent
     loop (the arbiter reimplements the dual-sample pattern separately).

Redaction runs on every write. No trajectory leaves this machine.
"""

from .schema import Trajectory, ToolCall, Outcome
from .redact import redact_text, redact_trajectory, RedactionConfig
from .collector import TrajectoryCollector
from .self_consistency import SelfConsistencySampler, Sample
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
    "SelfConsistencySampler",
    "Sample",
    "classify_chat_outcome",
    "apply_chat_outcome_heuristics",
    "FailureClassification",
]
