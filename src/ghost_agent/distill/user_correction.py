"""Detect when the user's *next* message is a correction of the agent's
prior turn.

Why this exists
---------------
``outcome_heuristics.py`` promotes UNKNOWN→FAILED only when a turn's
*own shape* signals a non-productive run (runtime abort markers,
selector thrashing, repeated identical tool errors). That covers the
mechanically-stuck failure mode but misses the dominant interactive-
chat failure mode: the agent confidently produced an answer that was
*wrong*, the user pushed back, and we want to learn from it.

The cheapest, most reliable supervisor for an interactive agent is the
human already in the loop. If their next message is a correction
("no", "that's wrong", "actually I meant…", or a rephrase of the same
ask), the prior assistant turn was FAILED — by the user's own
verdict, no validator required. This module is the predicate that
spots those messages.

Design
------
Pure-function classifier with **two independent signals**; promotion
requires BOTH to fire. Designed to err on the side of *missed*
corrections rather than *false-positive* corrections — false-positive
lessons poison retrieval, missed lessons just don't help (and the
biological backstop will catch them later if the trajectory sticks
around).

  Signal A — explicit correction phrase.
    Regex match against a curated phrase list anchored at the start
    of the user's message: "no", "wrong", "that's not right",
    "actually I meant", "try again", "you misunderstood", etc.
    Anchored — a phrase appearing mid-sentence after a user's own
    discussion does NOT fire. Cheap, deterministic.

  Signal B — semantic rephrase of the prior user request.
    Token-overlap (Jaccard on lowered, stop-stripped tokens) between
    the previous user message and the current one. A high overlap
    means the user is re-asking the same question, which is strong
    evidence that the assistant's response was inadequate. Threshold
    is conservative (≥ 0.4) — typical follow-up clarifications share
    much of the original wording.

  Promotion verdict.
    ``is_correction = True`` only when BOTH signals fire. The
    rationale is in the audit comment in ``handle_chat``: a single
    signal has too many false positives ("No, I think you're right"
    is not a correction; a rephrase that just adds context isn't
    necessarily a correction). Two independent signals dramatically
    cuts FP rate at the cost of missing some genuine corrections.

The classifier is purely lexical — no LLM call, no embeddings, no
external dependencies. A future enhancement could add a small LLM
judge for ambiguous cases (one signal fires but not both), but that
expands the failure surface and is intentionally deferred until the
heuristic-only version proves it's missing the most important
corrections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# ---------- Signal A: correction phrases ----------------------------

# Anchored at the start of the user's message (allowing leading
# punctuation / whitespace). The intent is to fire on direct
# rebuttals, not on phrases that happen to appear mid-thought.
_CORRECTION_PHRASES = (
    r"no\b",
    r"nope\b",
    r"wrong\b",
    r"that[' ]?s\s+(?:not|wrong|incorrect)",
    r"that[' ]?s\s+not\s+(?:right|correct|what)",
    r"that\s+(?:isn[' ]?t|is\s+not)\s+(?:right|correct|what)",
    r"actually\b",
    r"i\s+(?:meant|said|asked)",
    r"i\s+didn[' ]?t\s+(?:mean|say|ask)",
    r"you\s+(?:misunderstood|misread|missed|got\s+it\s+wrong)",
    r"try\s+again\b",
    r"that[' ]?s\s+(?:still\s+)?(?:wrong|incorrect|not\s+it)",
    r"not\s+(?:what|quite)\s+(?:i|what)",
    r"incorrect\b",
    r"that[' ]?s\s+the\s+wrong\s+",
    r"re[- ]?do\b",
    r"redo\s+(?:it|that|this)",
    r"start\s+over\b",
    r"this\s+is\s+wrong\b",
    r"that\s+failed\b",
    r"didn[' ]?t\s+work\b",
    r"doesn[' ]?t\s+work\b",
    # Polite forms — same intent.
    r"actually,?\s+",
    r"hmm,?\s+(?:no|not)",
)

_CORRECTION_RE = re.compile(
    r"^\s*(?:" + "|".join(_CORRECTION_PHRASES) + r")",
    re.IGNORECASE,
)


# ---------- Signal B: rephrase similarity ---------------------------

# Common stopwords + question scaffolding tokens. Removing them makes
# Jaccard reflect content overlap, not boilerplate overlap (otherwise
# "what is X" vs "what is Y" overlaps on `what is`).
_STOPWORDS = frozenset(
    """
    a an the of to for in on at by from with about into over under
    is are was were be been being am
    do does did done doing
    have has had having
    can could should would may might must will shall
    i you he she it we they me him her us them my your his their our
    this that these those there here
    and or but so if then because as than though although while
    not no nor only just very really quite
    what which who whom whose where when why how
    please could would
    all every each some any many few much more less most least
    let lets gonna wanna gotta
    """.split()
)

# Word-character tokens of length ≥ 2. Drops punctuation, digits-only
# fragments, and single-character noise.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")


def _content_tokens(text: str) -> set:
    """Return the lowercase, stopword-stripped content-token set of
    ``text``. Empty / non-string input returns the empty set."""
    if not isinstance(text, str) or not text:
        return set()
    return {
        tok.lower()
        for tok in _TOKEN_RE.findall(text)
        if tok.lower() not in _STOPWORDS
    }


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


# ---------- Public API ----------------------------------------------


@dataclass
class CorrectionVerdict:
    """One classification's result.

    ``is_correction`` is True iff the user's current message is judged
    a correction of the prior assistant turn. ``confidence`` is in
    [0, 1]. ``signals`` lists which detectors fired, for audit and
    debugging. ``reason`` is a short string suitable for
    ``Trajectory.failure_reason``.
    """

    is_correction: bool
    confidence: float
    signals: List[str] = field(default_factory=list)
    reason: str = ""


# Tunable knobs — kept module-level so tests can monkey-patch and so
# operators can override at runtime if the FP rate proves too high or
# too low in production. The defaults are intentionally conservative.
JACCARD_REPHRASE_THRESHOLD = 0.40
"""Minimum content-token Jaccard overlap (prev_user vs current_user)
for the rephrase signal to fire. Calibrated so a user adding a
clarifying clause to their original question fires, but a wholly new
follow-up question does not."""

MIN_CURRENT_TOKENS_FOR_REPHRASE = 2
"""Rephrase signal requires the current message to contain at least
this many content tokens. A bare 'no' or 'wrong' has zero content
tokens and would trivially overlap with anything otherwise."""


def classify_user_correction(
    *,
    prev_user_request: str,
    prev_assistant_response: str,
    current_user_text: str,
) -> CorrectionVerdict:
    """Decide whether ``current_user_text`` is a correction of the
    assistant turn that produced ``prev_assistant_response``.

    Two-signal predicate:

      A. Anchored correction-phrase regex on ``current_user_text``.
      B. Token-overlap rephrase: Jaccard(prev_user, current_user)
         over content tokens (stopwords stripped) ≥ threshold.

    Promotion requires BOTH. ``prev_assistant_response`` is accepted
    by the API for forward compatibility (a future LLM-judge variant
    will use it) but the heuristic version doesn't read it.

    Pure: never raises, never mutates inputs, no I/O.
    """
    # Defensive normalization. Calling code may pass None when the
    # message history is malformed; treat as empty rather than crash.
    pu = prev_user_request if isinstance(prev_user_request, str) else ""
    cu = current_user_text if isinstance(current_user_text, str) else ""
    # `prev_assistant_response` reserved for future signals.
    _ = prev_assistant_response

    signals: List[str] = []
    confidence = 0.0

    # Signal A — explicit correction phrase.
    # Truncate to the first 240 chars so a long user message that
    # happens to contain "no" mid-paragraph doesn't bleed into Signal
    # A; the regex is also start-anchored as a second-line defence.
    head = (cu or "").lstrip()[:240]
    if head and _CORRECTION_RE.search(head):
        signals.append("phrase")
        confidence += 0.55

    # Signal B — semantic rephrase of the prior user request.
    cu_tokens = _content_tokens(cu)
    pu_tokens = _content_tokens(pu)
    if (
        cu_tokens
        and pu_tokens
        and len(cu_tokens) >= MIN_CURRENT_TOKENS_FOR_REPHRASE
    ):
        overlap = _jaccard(pu_tokens, cu_tokens)
        if overlap >= JACCARD_REPHRASE_THRESHOLD:
            signals.append(f"rephrase(jaccard={overlap:.2f})")
            confidence += 0.45

    # Cap at 1.0 for downstream consumers that treat this as a
    # probability.
    if confidence > 1.0:
        confidence = 1.0

    is_correction = ("phrase" in signals) and any(
        s.startswith("rephrase") for s in signals
    )

    if is_correction:
        reason = (
            "user-correction signal: "
            + " + ".join(signals)
        )
    else:
        reason = ""

    return CorrectionVerdict(
        is_correction=is_correction,
        confidence=confidence,
        signals=signals,
        reason=reason,
    )
