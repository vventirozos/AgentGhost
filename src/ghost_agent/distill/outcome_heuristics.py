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


def _tool_call_failed(tc) -> bool:
    """True if a ToolCall failed, preferring its STRUCTURED ``error`` flag.

    As of 2026-07-07 the chat recorder populates ``ToolCall.error`` too (it was
    previously self-play/batch only), so a structured failure with atypical
    result text — e.g. the native-tools corruption shapes — is caught even when
    the text sniff below would miss it. The text sniff remains the fallback for
    legacy trajectories written before the flag was populated."""
    if getattr(tc, "error", ""):
        return True
    return _looks_like_tool_error(getattr(tc, "result", "") or "")


def _looks_like_tool_error(result: str) -> bool:
    """Cheap text detector for "this tool call failed" — the FALLBACK when the
    structured ``ToolCall.error`` flag isn't set (legacy trajectories).
    Conservative: favours false negatives over false positives.
    """
    if not isinstance(result, str):
        return False
    # A NON-ZERO exit-code banner is a hard failure signal even without an
    # "error:" prefix (127 = command not found, 130 = SIGINT, 1..9, …). The
    # banner can trail stdout, so search the whole result, not just the head.
    _exit_m = re.search(r"EXIT CODE:\s*(\d+)", result)
    if _exit_m is not None and _exit_m.group(1) != "0":
        return True
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
    # Per the module contract (signal 2), the repeats only count as
    # "stuck" when there was NO observable progress between them: a
    # successful navigation resets the tallies, so paginating through
    # results by clicking `#next-page` four times (each click followed
    # by a new page) is NOT promoted to FAILED.
    if traj.tool_calls:
        max_repeat = 0
        worst_sel = ""
        seen: dict = {}
        for tc in traj.tool_calls:
            if (tc.name or "").lower() != "browser":
                continue
            args = tc.arguments if isinstance(tc.arguments, dict) else {}
            op = str(args.get("operation") or args.get("op") or "").lower()
            result = getattr(tc, "result", "") or ""
            if op in ("navigate", "goto") and not _looks_like_tool_error(result):
                seen.clear()  # observable progress — restart the window
                continue
            sels = []
            sel = args.get("selector")
            if isinstance(sel, str) and sel:
                sels.append(sel)
            for step in args.get("actions") or []:
                if isinstance(step, dict) and isinstance(step.get("selector"), str) and step.get("selector"):
                    sels.append(step["selector"])
            for sel in sels:
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
    # Prefer the structured ToolCall.error flag; fall back to the text sniff.
    error_counts: dict = {}
    for tc in traj.tool_calls or []:
        if not _tool_call_failed(tc):
            continue
        sig = getattr(tc, "error", "") or _normalize_tool_error(getattr(tc, "result", "") or "")
        key = (tc.name or "", sig)
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


def resolve_turn_outcome(
    *,
    current: str,
    verifier: Optional[str] = None,
    execution_failed: bool = False,
) -> str:
    """Combine a turn's quality signals into ONE outcome — the single source
    of truth for the trajectory corpus, calibration, and selfhood.

    Historically these signals diverged: calibration and the selfhood model
    were made verifier-aware, but the trajectory corpus (which feeds the
    Reflector, PRM, and skills-auto) saw only the shape heuristics above. So a
    verifier-caught wrong answer stayed ``UNKNOWN`` in the corpus and never
    became a lesson or a PRM negative. This unifies them.

    Priority, strongest first:
      1. a STRUCTURAL execution failure (non-zero exit / tool error) is ground
         truth  → FAILED;
      2. a REFUTED verifier verdict (already thresholded at conf ≥ 0.7 by the
         caller)                                                     → FAILED;
      3. an existing FAILED (from the shape heuristics or a prior signal) is
         never upgraded away                                         → FAILED;
      4. a SUPPORTED verifier verdict                                → PASSED;
      5. otherwise keep ``current`` (UNKNOWN for a signal-free chat turn).

    ``verifier`` is the ``verifier_backfill`` tag: ``"passed"`` | ``"failed"``
    | ``None``. ``current`` is the outcome already on the trajectory (after the
    shape heuristics ran).
    """
    cur = current or Outcome.UNKNOWN.value
    if execution_failed or verifier == "failed":
        return Outcome.FAILED.value
    if cur == Outcome.FAILED.value:
        return Outcome.FAILED.value
    if verifier == "passed":
        return Outcome.PASSED.value
    return cur
