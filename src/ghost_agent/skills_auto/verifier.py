"""Skill verification.

Every acquired skill is only as good as the moment it was learned. The
world changes — challenge templates get tightened, tools get renamed,
libraries upgrade, the LLM itself drifts. The verifier re-tests a
candidate's exemplar against a provided validator function and returns
a pass/fail the caller uses to decide whether the skill stays in the
active registry or is marked deprecated.

Intentionally minimal — the verifier does NOT know how to run
trajectories, challenge templates, or tools. Those are the caller's
concerns. The verifier's job is the bookkeeping around the decision:
accept, deprecate, or retain-and-monitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .extractor import SkillCandidate


VerifyFn = Callable[[SkillCandidate], bool]


@dataclass
class VerificationResult:
    candidate_name: str
    passed: bool
    action: str           # "keep" | "deprecate" | "retain_monitor"
    reason: str = ""
    # New confidence after the verification event. Increases on pass,
    # decreases on fail, capped in [0, 1].
    updated_confidence: float = 0.0


def verify_candidate(
    candidate: SkillCandidate,
    verify_fn: VerifyFn,
    *,
    deprecate_below_confidence: float = 0.25,
    pass_boost: float = 0.1,
    fail_penalty: float = 0.3,
) -> VerificationResult:
    """Run `verify_fn(candidate)` and return a VerificationResult.

    Policy:
        pass       → boost confidence by `pass_boost`, action=keep
        fail       → deduct `fail_penalty`. If new confidence is below
                     `deprecate_below_confidence` → action=deprecate.
                     Otherwise action=retain_monitor (keep the skill
                     active but flag a re-verification soon).
    """
    try:
        passed = bool(verify_fn(candidate))
        reason = "" if passed else "verify_fn returned False"
    except Exception as e:
        passed = False
        reason = f"verify_fn raised {type(e).__name__}: {e}"

    if passed:
        new_conf = min(1.0, candidate.confidence + pass_boost)
        return VerificationResult(
            candidate_name=candidate.name,
            passed=True,
            action="keep",
            reason="",
            updated_confidence=float(new_conf),
        )

    new_conf = max(0.0, candidate.confidence - fail_penalty)
    if new_conf < deprecate_below_confidence:
        action = "deprecate"
    else:
        action = "retain_monitor"
    return VerificationResult(
        candidate_name=candidate.name,
        passed=False,
        action=action,
        reason=reason,
        updated_confidence=float(new_conf),
    )
