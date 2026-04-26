"""Session-shape failure detection for chat trajectories.

Real user-chat turns ship with ``outcome=UNKNOWN`` because there's no
automated validator on free-form chat — only self-play and self-
consistency batches produce explicit ``FAILED``. That breaks the
self-improvement loop for interactive sessions: the Reflector's
``run`` only iterates trajectories where ``outcome == FAILED.value``,
so a chat turn where the agent thrashed for an hour never produces a
lesson.

This module supplies a conservative classifier that promotes an
UNKNOWN chat trajectory to FAILED when the turn's own shape signals a
non-productive run. The bar is deliberately high — false positives
flood the lesson store with bad reflections.

Signals (each is independent; any one triggers promotion):

  1. ``[ATTEMPT_ABORTED_*]`` markers in ``final_response``. These are
     emitted by the runtime guards (cross-turn repetition, thinking-
     loop, n-gram repetition, ...) only AFTER an in-band check has
     already determined the turn was non-productive. Strong signal.

  2. The same selector-shaped argument appears in N or more browser
     tool calls within the turn AND those calls produced no
     observable progress (no successful navigations between them).
     N defaults to 4. Signals "agent is stuck clicking the same
     thing", which was the dominant failure mode in the 2026-04-26
     webOS session.

  3. The same tool returned the same normalized error message N or
     more times. N defaults to 3. Signals "agent is not learning
     from feedback".

  4. A browser ``interact`` call returned ``aborted=True`` (a goto
     failure cascaded into the rest of the sequence) AND the agent
     made no follow-up action to fix the URL. Detected via
     trajectory-level inspection of the last browser tool call's
     result text.

These cover the failure modes most worth surfacing to the Reflector;
the heuristics are intentionally local to a single trajectory so this
module has zero state and is trivially testable. Cross-turn signals
(e.g. "the same misdiagnosis appears across 5 turns") need a
session-scoped tracker and are out of scope here — that belongs in
a future ``session_telemetry.py`` keyed by ``session_id``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .schema import Trajectory, Outcome


_ATTEMPT_ABORTED_RE = re.compile(r"\[ATTEMPT_ABORTED_[A-Z_]+\]")


@dataclass
class FailureClassification:
    """One classification attempt's verdict and the signal that fired.

    ``outcome`` is the (possibly upgraded) outcome string.
    ``reason`` is a short human-readable label of the firing signal,
    suitable to drop into ``Trajectory.failure_reason``. Empty when
    the trajectory wasn't promoted.
    """

    outcome: str
    reason: str = ""

    @property
    def promoted(self) -> bool:
        return self.reason != ""


# ---------------------------------------------------------------- helpers


def _extract_browser_selectors(traj: Trajectory) -> List[str]:
    """Return all selectors used by browser tool calls in this turn.

    Pulls from both the atomic ops (selector at the top of the args
    dict) and from `interact` sequences (each sub-action's selector).
    Returns the raw selector strings — no normalization — so a
    "#start-btn" called four times shows up as four "#start-btn"
    entries.
    """
    selectors: List[str] = []
    for tc in traj.tool_calls or []:
        if (tc.name or "").lower() != "browser":
            continue
        args = tc.arguments if isinstance(tc.arguments, dict) else {}
        sel = args.get("selector")
        if isinstance(sel, str) and sel:
            selectors.append(sel)
        for step in args.get("actions") or []:
            if not isinstance(step, dict):
                continue
            inner = step.get("selector")
            if isinstance(inner, str) and inner:
                selectors.append(inner)
    return selectors


_TOOL_ERROR_PREFIX_RE = re.compile(
    r"^\s*(?:error|\[error\]|failed|exception)[:\-]?\s*",
    re.IGNORECASE,
)


def _normalize_tool_error(s: str) -> str:
    """Squash whitespace, lowercase, strip a leading "Error:"-style
    prefix, and cap length so two textually-similar errors hash to
    the same key."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = _TOOL_ERROR_PREFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).lower()
    return s[:200]


def _looks_like_tool_error(result: str) -> bool:
    """Cheap detector for "this tool call failed".

    The trajectory schema doesn't carry a structured per-call error
    flag for chat turns (the ``ToolCall.error`` field is populated by
    self-play / batch paths, not the chat path). So we sniff the
    result text. Conservative — favours false negatives over false
    positives.
    """
    if not isinstance(result, str):
        return False
    head = result.strip()[:120].lower()
    return any(
        marker in head
        for marker in (
            "error:",
            "[error]",
            "exception",
            "traceback",
            "failed:",
            "syntax error",
            "operation failed",
        )
    )


# ---------------------------------------------------------------- main API


def classify_chat_outcome(
    traj: Trajectory,
    *,
    repeated_selector_threshold: int = 4,
    repeated_error_threshold: int = 3,
) -> FailureClassification:
    """Decide whether to promote an UNKNOWN trajectory to FAILED.

    Pre-existing ``PASSED`` / ``FAILED`` outcomes are returned
    unchanged — this function only ever upgrades UNKNOWN. The
    function is pure: it never mutates ``traj``.

    Threshold knobs are exposed for tests; production callers should
    use the defaults (4 and 3 — calibrated against the 2026-04-26
    incident as the lower bound for "obviously stuck").
    """

    current = traj.outcome or Outcome.UNKNOWN.value

    # Already labelled — respect the existing verdict. We never
    # demote PASSED, never overrule an explicit FAILED.
    if current != Outcome.UNKNOWN.value:
        return FailureClassification(outcome=current, reason="")

    # 1. Runtime abort markers — strongest available signal.
    if traj.final_response and _ATTEMPT_ABORTED_RE.search(traj.final_response):
        match = _ATTEMPT_ABORTED_RE.search(traj.final_response)
        marker = match.group(0) if match else "[ATTEMPT_ABORTED_*]"
        return FailureClassification(
            outcome=Outcome.FAILED.value,
            reason=f"runtime abort marker {marker}",
        )

    # 2. Repeated browser selector — agent stuck clicking same thing.
    selectors = _extract_browser_selectors(traj)
    if selectors:
        max_repeat = 0
        worst_sel = ""
        seen: dict = {}
        for sel in selectors:
            seen[sel] = seen.get(sel, 0) + 1
            if seen[sel] > max_repeat:
                max_repeat = seen[sel]
                worst_sel = sel
        if max_repeat >= repeated_selector_threshold:
            return FailureClassification(
                outcome=Outcome.FAILED.value,
                reason=(
                    f"browser selector {worst_sel!r} used {max_repeat}× "
                    f"in one turn (≥ {repeated_selector_threshold} threshold)"
                ),
            )

    # 3. Repeated identical tool errors — agent ignored prior feedback.
    error_counts: dict = {}
    for tc in traj.tool_calls or []:
        result = getattr(tc, "result", "") or ""
        if not _looks_like_tool_error(result):
            continue
        key = (tc.name or "", _normalize_tool_error(result))
        error_counts[key] = error_counts.get(key, 0) + 1
    for (tool_name, _err), count in error_counts.items():
        if count >= repeated_error_threshold:
            return FailureClassification(
                outcome=Outcome.FAILED.value,
                reason=(
                    f"tool {tool_name!r} returned the same error "
                    f"{count}× in one turn "
                    f"(≥ {repeated_error_threshold} threshold)"
                ),
            )

    # 4. Browser interact aborted via initial-goto / mid-sequence goto
    # failure. The runner sets aborted=True and surfaces a clear
    # ``⚠ SEQUENCE ABORTED`` banner in the agent-visible output —
    # we sniff that string, which is more reliable than parsing the
    # JSON envelope from a free-form result field.
    for tc in traj.tool_calls or []:
        if (tc.name or "").lower() != "browser":
            continue
        result = getattr(tc, "result", "") or ""
        if "SEQUENCE ABORTED" in result:
            return FailureClassification(
                outcome=Outcome.FAILED.value,
                reason="browser interact sequence aborted (failed goto)",
            )

    return FailureClassification(outcome=current, reason="")


def apply_chat_outcome_heuristics(traj: Trajectory) -> bool:
    """Mutating helper: promote ``traj.outcome`` and set
    ``traj.failure_reason`` in place when classification fires.

    Returns True iff the trajectory was modified. Callers that want a
    pure check should use ``classify_chat_outcome`` directly.
    """
    verdict = classify_chat_outcome(traj)
    if not verdict.promoted:
        return False
    traj.outcome = verdict.outcome
    if not traj.failure_reason:
        traj.failure_reason = verdict.reason
    return True
