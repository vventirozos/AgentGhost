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

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .schema import Trajectory, Outcome


_ATTEMPT_ABORTED_RE = re.compile(r"\[ATTEMPT_ABORTED_[A-Z_]+\]")

# The failure_reason stamped when a trajectory is FAILED *solely* because a
# tool call broke — no refute, no shape-heuristic finding. Shared so the
# writer (agent._record_turn_trajectory) and the reader (the late-verdict
# backfill, which may upgrade exactly this kind of FAILED to PASSED) cannot
# drift apart: a key≠serializer mismatch here would silently disable the
# 2026-07-31 honest-failure rule on the async-verdict path.
STRUCTURAL_FAILURE_REASON = "structural failure"


def structural_reason(cause: str = "") -> str:
    """`STRUCTURAL_FAILURE_REASON`, optionally qualified with its CAUSE.

    ⚠ WHY A PREFIX AND NOT A NEW STRING (2026-08-10). The bare constant is
    LOAD-BEARING: `resolve_turn_outcome` matches it EXACTLY to decide whether
    a late verifier PASS may upgrade this FAILED. Appending a cause naively
    would silently stop that match and disable the 2026-07-31 honest-failure
    rule on the async-verdict path — the exact drift the constant's own
    comment warns about. So the cause is a SUFFIX after ": " and every
    reader goes through `is_structural_reason()`.

    Motivation: `structural failure` was 42 of 160 recorded failures (26%)
    and said only THAT execution broke, never WHAT broke — so "was this a
    hard task or a flaky tool/node?" was unanswerable from the corpus. That
    question matters because these labels train the complexity router.
    """
    c = re.sub(r"\s+", " ", str(cause or "")).strip()
    return f"{STRUCTURAL_FAILURE_REASON}: {c[:120]}" if c else STRUCTURAL_FAILURE_REASON


def structural_cause_for_trajectory(traj) -> str:
    """A short WHAT-BROKE label derived from the trajectory's failed tools.

    Uses `tool_call_failed` — THE shared sniffer — rather than a second
    "did this fail?" rule, because a duplicated one is how the corpus and
    its consumers drift (the lesson this module already records).

    Shape: ``<tool>: <first line of the error>``, or ``<tool> +N more`` when
    several tools broke. Returns "" when nothing identifiable failed, in
    which case the caller keeps the bare constant — an unqualified reason is
    better than an invented one.
    """
    try:
        broken = [tc for tc in (getattr(traj, "tool_calls", None) or [])
                  if tool_call_failed(tc)]
    except Exception:  # noqa: BLE001 — instrumentation must never break a turn
        return ""
    if not broken:
        return ""
    first = broken[0]
    name = str(getattr(first, "name", "") or "tool").strip()
    detail = str(getattr(first, "error", "") or getattr(first, "result", "") or "")
    detail = re.sub(r"\s+", " ", detail).strip()
    label = f"{name}: {detail}" if detail else name
    if len(broken) > 1:
        label += f" (+{len(broken) - 1} more)"
    return label[:120]


def is_structural_reason(reason: str) -> bool:
    """True for the bare constant AND any cause-qualified form of it.

    Every consumer of the old exact-match MUST use this, or a qualified
    reason silently loses the late-PASS upgrade it is entitled to.
    """
    r = (reason or "").strip()
    return r == STRUCTURAL_FAILURE_REASON or r.startswith(
        STRUCTURAL_FAILURE_REASON + ":")


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


def tool_call_failed(tc) -> bool:
    """Public alias for ``_tool_call_failed``.

    Exists so other packages (skills_auto's honest-failure graduation
    guard) can reuse THE failure sniffer instead of writing a second
    one — a duplicated "did this tool fail?" rule is how the corpus and
    the operator line came to disagree in the first place.
    """
    return _tool_call_failed(tc)


def looks_like_tool_error(result: str) -> bool:
    """Public alias for ``_looks_like_tool_error``.

    Same rationale as ``tool_call_failed``: the turn loop's
    ``tools_run_this_turn`` entries are plain dicts (``{"name", "content"}``)
    rather than ``ToolCall`` objects, and the verifier's high-stakes gate
    needs THE failure sniffer, not a second one that can drift from the
    label the corpus writes.
    """
    return _looks_like_tool_error(result)


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
            # file_system's replace corruption guard: "SYSTEM
            # INSTRUCTION: REPLACE REJECTED — …The file was NOT
            # modified." A hard-rejected mutation was labelled OK for
            # weeks (5 live calls, newest 2026-07-30) — feeding
            # skills_auto graduation, the verifier's high-stakes gate
            # and foresight seeding a success that never happened.
            # Deliberately NOT a generic "rejected"/"system
            # instruction" marker: steering SYSTEM INSTRUCTIONs are
            # not failures. (The other blind spot from the same
            # review — the bare "[SYSTEM ERROR]: Process failed
            # (Exit 1) with no output." banner — was measured to be
            # grep-family NO-MATCH successes in costume, all
            # pre-dating the search-op normalization; labelling them
            # ok is CORRECT, so no marker was added.)
            "replace rejected",
        )
    )


# -------------------------------------------- unacknowledged-failure shape
#
# Operator decision 2026-08-04 ("add the shape rule, structural failure
# shouldn't pass"), narrowing the 2026-07-31 honest-failure rule.
#
# Live case that produced it — req 03b96c28, trajectory f78c8b33…: the user
# asked for a line count of a file OUTSIDE the sandbox; `file_system`,
# `execute` and `file_system` ALL failed; the agent replied `0` — a
# fabrication, one character, with no acknowledgment of any failure. The
# cheap judge REFUTED it correctly (conf 1.0), `_escalate_refute` sent it to
# the main model, which overturned to CONFIRMED, and the honest-failure rule
# then rewrote the corpus label failed → passed. A fabricated answer was
# laundered into a positive training example by the exact rule built to
# protect honest ones.
#
# This is a SHAPE rule: it asks no model anything. It only ever WITHHOLDS a
# verifier PASS from a turn that is already structurally failed — it never
# manufactures a FAILED label that was not already there (see
# `resolve_turn_outcome`). The acknowledgment check is what keeps the
# 2026-07-31 rule alive: "that file does not exist" must still PASS, or the
# incentive gradient that produces fabrication is recreated.

UNACKED_FAILURE_GATE_ENV = "GHOST_UNACKED_FAILURE_GATE"


def unacked_failure_gate_enabled() -> bool:
    """Kill switch for the unacknowledged-total-failure shape rule.

    Default ON — the operator asked for the behaviour.
    ``GHOST_UNACKED_FAILURE_GATE=0`` restores the EXACT pre-2026-08-04
    behaviour: a verifier ``passed`` outranks a structural execution failure
    unconditionally (the 2026-07-31 honest-failure rule with no shape
    narrowing), on every path — the write-time consolidation, the late
    verdict backfill, the operator's Turn Outcome line, and the calibration
    grade.

    Read per call (not at import) so the flag can be flipped without a
    restart — same idiom as ``verifier._escalate_refute_enabled``. Read in
    exactly ONE place (this function is called from
    :func:`unacknowledged_total_failure`, which every site goes through) so
    the switch cannot be live on one path and dark on another.
    """
    return os.getenv(UNACKED_FAILURE_GATE_ENV, "1").strip().lower() not in (
        "0", "false", "no",
    )


# Vocabulary of "the reply told the user something went wrong". Deliberately
# BROAD: "acknowledged" is the PERMISSIVE verdict here (it lets the turn keep
# its verifier PASS), so a false positive costs nothing but a false negative
# re-punishes honest failure-reporting — the exact 2026-07-31 regression.
# Matched anywhere in the reply, case-insensitively.
_FAILURE_ACK_RE = re.compile(
    r"""
      (?:does|did)\s*n[o']?t\s+(?:exist|support|work|return|find|contain)
    | doesn't\s+(?:exist|support|work|return|find|contain)
    | no\s+such\s+(?:file|directory|table|column|command|key|entry)
    | not\s+found | no\s+match(?:es|ing)? | nothing\s+(?:found|to\s+\w+)
    | \berrors?\b | \bexceptions?\b | traceback | stderr
    | \bfail(?:ed|s|ing|ure|ures)?\b
    | (?:un|not\s+)able\s+to | \bcould\s*n[o']?t\b | couldn't
    | \bcan\s*not\b | \bcan't\b | \bcannot\b | \bunsuccessful\b
    | \bno\s+results?\b | zero\s+results? | returned\s+(?:nothing|empty)
    | permission\s+denied | \bdenied\b | \bforbidden\b
    | blocked\s+by\s+(?:the\s+)?(?:egress|guard|policy|firewall|proxy|allowlist)
    | timed\s+out | \btimeout\b
    | not\s+supported | unsupported | \binvalid\b | \bmalformed\b
    | outside\s+(?:the\s+|my\s+|its\s+)?(?:sandbox|workspace|project)
    | not\s+access(?:ible)? | inaccessible
    | no\s+access | \bmissing\b | \bunavailable\b | not\s+available
    | already\s+(?:removed|deleted|gone) | never\s+created
    | hard\s+limit | gave\s+up | did\s+not\s+complete | \bcrash(?:ed|es)?\b
    | \bbroke(?:n)?\b | \brefused\b | \brejected\b
    | (?:is|was|were|are|returned|came\s+back|came\s+up)\s+empty
    | empty\s+(?:result|response|output|set|list|string|file|director)
    | exit\s+code | non-?zero
    | i\s+(?:do\s*n[o']?t|don't)\s+have | i\s+have\s+no
    """,
    re.IGNORECASE | re.VERBOSE,
)

# "just reply with exactly the word: NOPE" — an instruction that FIXES the
# reply text. The reply then carries the user's own token instead of prose,
# so the acknowledgment lives in the REQUEST, not the response. Without this
# escape the 2026-07-31 rule's own live-validation probe (req F0/85a2d9ce,
# reply "NOPE") would regress to FAILED. Bounded to a short span after the
# instruction so an unrelated later sentence can't donate the literal.
_EXACT_REPLY_INSTRUCTION_RE = re.compile(
    r"(?:reply|respond|answer|say|output|return)\b[^.\n]{0,40}?"
    r"\b(?:exactly|only|just|verbatim|literally)\b[^.\n]{0,80}",
    re.IGNORECASE,
)

# An instructed literal is a SHORT token, not a paragraph. Longer replies are
# prose and must earn their acknowledgment from the vocabulary above.
_INSTRUCTED_LITERAL_MAX_CHARS = 64


def _reply_is_instructed_literal(final_response: str, user_request: str) -> bool:
    """True when the user's own request pins the exact reply text and the
    reply is that text.

    Shape-only: the literal has to appear inside an explicit
    "reply with exactly/only/just …" span of the USER's message, as a whole
    token. ``"reply with just the number"`` therefore does NOT license the
    fabricated ``"0"`` of req 03b96c28 (the span contains no ``0``), while
    ``"just reply with exactly the word: NOPE"`` does license ``"NOPE"``.
    """
    reply = (final_response or "").strip().strip("`\"'*. ")
    if not reply or len(reply) > _INSTRUCTED_LITERAL_MAX_CHARS:
        return False
    req = user_request or ""
    if not req:
        return False
    for m in _EXACT_REPLY_INSTRUCTION_RE.finditer(req):
        span = m.group(0)
        if re.search(r"(?<!\w)" + re.escape(reply) + r"(?!\w)", span,
                     re.IGNORECASE):
            return True
    return False


def response_acknowledges_failure(final_response: str,
                                  user_request: str = "") -> bool:
    """Does this reply tell the user that something went wrong?

    Content-based, never length-based: a two-word reply that says "not
    found" acknowledges; a five-paragraph reply that never mentions a
    problem does not. Conservative in the direction that PRESERVES the
    2026-07-31 honest-failure rule — when in doubt, say yes.
    """
    text = final_response or ""
    if not text.strip():
        return False          # an empty reply reports nothing
    if _FAILURE_ACK_RE.search(text):
        return True
    return _reply_is_instructed_literal(text, user_request)


def tool_failure_flags(tools: Optional[Iterable[Any]]) -> List[bool]:
    """Per-call "did this fail?" flags for EITHER tool-call shape.

    The corpus carries ``ToolCall`` objects (``.result`` / ``.error``); the
    turn loop carries plain dicts (``{"name", "content"}``). One function
    knows both, so the shape rule cannot mean one thing on the corpus path
    and another on the operator-line path — the drift that made the corpus
    and the Turn Outcome line disagree before. Uses THE shared sniffer, not
    a third copy of it.
    """
    flags: List[bool] = []
    for t in tools or ():
        if t is None:
            continue
        if isinstance(t, dict):
            flags.append(_looks_like_tool_error(str(t.get("content", "") or "")))
        else:
            flags.append(_tool_call_failed(t))
    return flags


def unacknowledged_total_failure(*, tools: Optional[Iterable[Any]] = None,
                                 final_response: str = "",
                                 user_request: str = "",
                                 tool_failures: Optional[Sequence[bool]] = None,
                                 ) -> bool:
    """THE shape rule: every tool call this turn failed and the reply never
    said so.

    ``tools`` accepts either shape (see :func:`tool_failure_flags`);
    ``tool_failures`` lets a caller pass pre-computed flags.

    ALL, never ANY — a turn where one tool fails and the agent recovers via
    another and answers correctly is a GOOD turn and must keep its PASS. A
    turn with no tool calls at all can't have "all of them" fail, so it is
    never flagged.

    Returns False whenever the kill switch is off, which is why every call
    site routes through here rather than testing the env var itself.
    """
    if not unacked_failure_gate_enabled():
        return False
    flags = list(tool_failures) if tool_failures is not None \
        else tool_failure_flags(tools)
    if not flags or not all(flags):
        return False
    return not response_acknowledges_failure(final_response, user_request)


def unacknowledged_total_failure_for_trajectory(traj) -> bool:
    """:func:`unacknowledged_total_failure` read off a ``Trajectory``.

    Used by the write-time consolidation AND the late-verdict backfill, so
    both delivery paths compute the flag from the same fields of the same
    record instead of two hand-mirrored derivations.
    """
    if traj is None:
        return False
    try:
        return unacknowledged_total_failure(
            tools=getattr(traj, "tool_calls", None) or [],
            final_response=getattr(traj, "final_response", "") or "",
            user_request=getattr(traj, "user_request", "") or "",
        )
    except Exception:  # noqa: BLE001 — labelling must never break a turn
        return False


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
            actions = [s for s in (args.get("actions") or []) if isinstance(s, dict)]
            # The live tool's multi-step shape is op="interact" with the
            # navigation INSIDE the actions list ({"action": "goto", …});
            # a successful one is the same observable progress as a
            # top-level navigate. Clear BEFORE tallying this call's own
            # selectors so within-call thrash (goto + 4× the same click in
            # one sequence) still counts. An aborted sequence never ran the
            # steps after its goto, so it is not progress — and its banner
            # doesn't trip _tool_call_failed's text sniff, hence the
            # explicit SEQUENCE ABORTED check.
            if (
                op == "interact"
                and any(str(s.get("action") or "").lower() in ("goto", "navigate") for s in actions)
                and not _tool_call_failed(tc)
                and "SEQUENCE ABORTED" not in result
            ):
                seen.clear()
            sels = []
            sel = args.get("selector")
            if isinstance(sel, str) and sel:
                sels.append(sel)
            for step in actions:
                if isinstance(step.get("selector"), str) and step.get("selector"):
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
    current_reason: str = "",
    unacked_total_failure: bool = False,
) -> str:
    """Combine a turn's quality signals into ONE outcome — the single source
    of truth for the trajectory corpus, calibration, and selfhood.

    Historically these signals diverged: calibration and the selfhood model
    were made verifier-aware, but the trajectory corpus (which feeds the
    Reflector, PRM, and skills-auto) saw only the shape heuristics above. So a
    verifier-caught wrong answer stayed ``UNKNOWN`` in the corpus and never
    became a lesson or a PRM negative. This unifies them.

    Priority, strongest first:
      1. a REFUTED verifier verdict (already thresholded at conf ≥ 0.7 by the
         caller)                                                     → FAILED;
      2. an existing FAILED (from the shape heuristics or a prior signal) is
         never upgraded away                                         → FAILED
         — EXCEPT one stamped exactly ``STRUCTURAL_FAILURE_REASON``, which is
         upgradable by a late verifier PASS (the 2026-07-31 async half; pass
         ``current_reason`` to enable it, which is what makes the late
         backfill and this function ONE ladder rather than two);
      2b. …but that PASS is WITHHELD when
         ``unacked_total_failure`` is set (2026-08-04, below);
      3. a SUPPORTED verifier verdict                                → PASSED;
      4. a STRUCTURAL execution failure (non-zero exit / tool error) → FAILED;
      5. otherwise keep ``current`` (UNKNOWN for a signal-free chat turn).

    **Rules 3 and 4 swapped on 2026-07-31 (operator decision).** Structural
    failure used to be priority 1: ground truth that something broke. But it
    conflated a BROKEN TURN with a FAILED ENVIRONMENT — a turn whose only
    tool call fails and whose answer *honestly reports that failure* ("that
    file does not exist") was labelled FAILED, even though the agent did the
    right thing and the verifier CONFIRMED it. That taught the corpus,
    calibration, and skills-auto to treat honest failure-reporting as bad
    behaviour, which is precisely the incentive that produces fabricated
    success. The verifier is the signal that actually inspected the answer
    against the evidence, so a CONFIRMED verdict now outranks the structural
    signal — the environment failed, the turn did not. An execution failure
    still lands FAILED whenever no verdict exists (rule 4), and rule 2 keeps
    the shape heuristics intact: a turn that thrashed a selector 4× or hit a
    runtime abort marker stays FAILED no matter how honestly it says so.

    **Rule 2b added 2026-08-04 (operator decision), narrowing the above.**
    The 07-31 rule assumed a CONFIRMED verdict means the reply handled the
    broken tool honestly. Req 03b96c28 disproved it: three tool calls, all
    three failed, the reply was the single character ``0`` — a fabricated
    answer with no acknowledgment — the cheap judge refuted it correctly, the
    main model overturned to CONFIRMED, and the label was rewritten
    ``failed → passed`` into the learning corpus. So the PASS is now withheld
    when the turn's SHAPE says the answer cannot be honest: **every** tool
    call failed AND the reply never mentioned a failure (see
    :func:`unacknowledged_total_failure`). "Every", not "any" — a turn that
    recovers through a second tool is a good turn and still passes.

    The withholding is deliberately NON-manufacturing: it can only stop an
    upgrade, never invent a FAILED. It applies only when the turn is ALREADY
    structurally failed — ``execution_failed`` (write time) or a
    ``current`` of FAILED/``STRUCTURAL_FAILURE_REASON`` (late backfill). A
    turn whose strike ledger is clean keeps its PASS even if the corpus text
    sniffer thinks every result looks like an error, which bounds the cost of
    that sniffer's known false positives (an ``EXIT CODE: 1`` banner nested
    deep inside an otherwise successful tool payload).

    ``verifier`` is the ``verifier_backfill`` tag: ``"passed"`` | ``"failed"``
    | ``None``. ``current`` is the outcome already on the trajectory (after the
    shape heuristics ran); ``current_reason`` is its ``failure_reason``.
    """
    cur = current or Outcome.UNKNOWN.value
    if verifier == "failed":
        return Outcome.FAILED.value
    # An existing FAILED is never upgraded away — except a STRUCTURAL-ONLY
    # one, which the 2026-07-31 async half is allowed to lift. Callers that
    # don't pass `current_reason` get the pre-07-31 "never upgrade" behaviour
    # unchanged, so this is additive.
    structural_failed = (
        cur == Outcome.FAILED.value
        and is_structural_reason(current_reason)
    )
    if cur == Outcome.FAILED.value and not structural_failed:
        return Outcome.FAILED.value
    # Rule 2b: withhold the PASS from an already-structurally-failed turn
    # whose every tool call broke and whose reply never said so.
    withhold_pass = (
        bool(unacked_total_failure)
        and (bool(execution_failed) or structural_failed)
    )
    if verifier == "passed" and not withhold_pass:
        return Outcome.PASSED.value
    if execution_failed:
        return Outcome.FAILED.value
    if cur == Outcome.FAILED.value:
        # A structural FAILED whose upgrade was withheld (or that had no
        # verdict at all) stays FAILED.
        return Outcome.FAILED.value
    return cur
