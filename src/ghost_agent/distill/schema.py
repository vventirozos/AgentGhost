"""Trajectory schema.

The shape captured here is deliberately richer than the eval harness's
TaskResult — eval cares about outcome-only summaries, while distill
needs the full trace (system prompt, plan, every tool call) so a later
SFT pipeline can train on the actual decision sequence.

Kept as plain dataclasses (no pydantic) so (a) it's a zero-cost
dependency in tests and (b) the JSONL format is just
`json.dumps(asdict(traj))` — easy to diff, easy to migrate.
"""

from __future__ import annotations

import datetime
import enum
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


class Outcome(str, enum.Enum):
    """Final label for a trajectory.

    Pass/fail come from the validator when available. `UNKNOWN` is
    legitimate for real user turns where no automated verifier applies
    — those trajectories still count for analysis (tool-selection
    patterns, embedding clustering) even if they can't feed RL.
    """

    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: str = ""
    error: str = ""
    duration_s: float = 0.0


@dataclass
class Trajectory:
    """One turn's worth of agent activity.

    Fields are ordered by write-time: metadata first, then prompt, then
    tool sequence, then outcome. That's the order a reader naturally
    scans and also keeps `final_response` last in the JSONL line so
    `tail` is useful for debugging.
    """

    # --- Metadata ---
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    session_id: str = ""             # groups turns from the same conversation
    task_kind: str = "user_request"  # user_request | challenge_template | self_play | reflection
    cluster: Optional[str] = None
    tier: Optional[str] = None
    model: str = ""
    temperature: float = 0.0
    # Integer sample index when this trajectory is part of an N-sample
    # self-consistency batch; None for one-shot turns.
    sample_index: Optional[int] = None
    # Shared identifier across all samples in a self-consistency batch;
    # lets downstream tooling group them without re-deriving from ids.
    batch_id: Optional[str] = None

    # --- Prompt / request ---
    system_prompt: str = ""
    user_request: str = ""
    planning_output: Optional[str] = None

    # --- Execution trace ---
    tool_calls: List[ToolCall] = field(default_factory=list)
    n_steps: int = 0

    # --- Cost ---
    tokens_in: int = 0
    tokens_out: int = 0
    duration_s: float = 0.0

    # --- Outcome ---
    outcome: str = Outcome.UNKNOWN.value  # stored as string for JSON friendliness
    failure_reason: str = ""
    validator_signal: Dict[str, Any] = field(default_factory=dict)

    # --- Final response (last so `tail` is readable) ---
    final_response: str = ""

    # Free-form; stays out of the core schema.
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=False, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trajectory":
        tc_list = [ToolCall(**t) if not isinstance(t, ToolCall) else t
                   for t in (d.get("tool_calls") or [])]
        # Strip sub-fields we rebuild manually, so **d doesn't double-pass them.
        d = dict(d)
        d.pop("tool_calls", None)
        return cls(tool_calls=tc_list, **d)
