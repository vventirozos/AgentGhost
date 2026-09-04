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
from typing import List, Optional


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


def has_correction_phrase(text) -> bool:
    """Signal A on its own — does this message OPEN like a correction?

    Exposed because it is the CHEAP GATE for the adjudicated second signal:
    the caller must be able to ask "is this even a candidate?" without paying
    for a judge, and it must ask with the SAME regex this module promotes on.
    Measured on 1601 consecutive live turn pairs it fires 14 times (0.9%), so
    an LLM adjudication behind it costs about one call per 114 turns.

    A second private copy of this test in the caller is how the gate and the
    promotion come to disagree about what a candidate is — the one-authority
    rule this module already applies to `_content_tokens`.
    """
    head = (text if isinstance(text, str) else "").lstrip()[:240]
    return bool(head and _CORRECTION_RE.search(head))


# ---------- Affirmation veto (Signal A+B false-positive guard) -------

# An affirming follow-up can trip BOTH signals: "actually the sort you
# wrote works great, thanks" starts with a correction phrase
# ("actually") and clears the rephrase overlap by echoing the request's
# content words — yet it is PRAISE. Promoting it retracts a good lesson
# and stamps a good turn FAILED (the exact self-poisoning class the
# 2026-07-05 chess post-mortem cleaned out of vector memory). When the
# message carries a clear affirmation and NO negative marker, the
# correction verdict is vetoed.
_AFFIRMATION_RE = re.compile(
    r"\b(?:works?\s+(?:great|now|fine|perfectly|well|as\s+expected)|"
    r"it\s+works|working\s+(?:now|great|fine|perfectly)|"
    r"you\s*(?:['’]?re|are|were)\s+right|"
    r"that['’ ]?s\s+(?:right|correct|it|perfect)|"
    r"looks\s+(?:good|great|right|correct)|spot\s+on|"
    r"perfect|excellent|awesome|brilliant|well\s+done|nicely\s+done|"
    r"good\s+job|great\s+job|"
    r"exactly\s+(?:right|what\s+i\s+(?:wanted|meant|asked)))\b",
    re.IGNORECASE,
)

# Strong negative markers BLOCK the veto: "no, it works locally but
# your fix broke prod" praises nothing — the complaint must win.
# Deliberately phrase-based (no bare "error"/"exception" substrings —
# those false-positive on benign content like "no errors now"; see the
# outcome-heuristics deferred finding). A bare leading "no" also
# blocks: "No, you're right" is genuinely ambiguous, and a missed veto
# only costs a lesson — a wrongly vetoed real correction costs the
# label, so ambiguity resolves toward correction.
_NEGATIVE_MARKER_RE = re.compile(
    r"^\s*no\b|"
    r"\b(?:nope|wrong|incorrect|"
    r"not\s+(?:right|correct|working|what)|"
    r"didn['’]?t\s+work|doesn['’]?t\s+work|"
    r"isn['’]?t\s+working|stopped\s+working|"
    r"still\s+(?:broken|failing|fails|wrong|not\s+working)|"
    r"failed|failing|fails|broke|broken|"
    r"misunderstood|misread)\b",
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


#: Bounded budget for the adjudication call. It runs on the ~0.9% of turns
#: that clear Signal A, and it blocks the user's turn while it does — so it
#: gets a small ceiling and a hard failure mode (see `classify_user_correction`
#: on `contradicts=None`: every timeout falls back to today's behaviour).
ADJUDICATION_MAX_TOKENS = 200


def adjudication_prompt(*, prev_user_request: str,
                        prev_assistant_response: str,
                        current_user_text: str) -> str:
    """The Signal-B question, asked of the ASSISTANT'S ANSWER.

    ⚠ THE DISTINCTIONS THIS PROMPT EXISTS TO DRAW were all found in live
    traffic (§4EP, the 12 blocked candidates):

      * "no let's change topic, do you think covid vaccines…" — a topic
        change that opens with "no". NOT a correction.
      * "no go ahead, i give you the permission" — a permission grant.
      * "nope, nothing for now, mark the project as done" — an ANSWER to a
        question the assistant asked. Promoting it stamps a correct turn
        FAILED.
      * "Actually, you remember now — Chess Coach v3 was just a stepping
        stone…" — a message asserting a FALSE PREMISE about the assistant's
        own history (a live continuity-probe). Promoting it would teach the
        agent that its correct memory was the error. Called out explicitly
        because it is correction-SHAPED and the most damaging false positive
        available.

    Against genuine ones: "No, that's wrong. You made a mistake there. The
    correct answer is Canberra"; "no I was asking about the mysql problem";
    "that's not true. you have many mechanisms that work when i'm not
    around".
    """
    return (
        "You judge whether a user's message CORRECTS the assistant's "
        "previous answer.\n\n"
        "It is a CORRECTION only if the user asserts that something the "
        "assistant SAID is wrong, inaccurate, or answered the wrong "
        "question.\n\n"
        "It is NOT a correction if the user is:\n"
        "- changing the subject (even if the message starts with \"no\")\n"
        "- answering a question the assistant asked (\"no, nothing for "
        "now\")\n"
        "- granting permission or agreeing to proceed\n"
        "- adding new information without contradicting anything\n"
        "- asserting a premise about the assistant's own past that the "
        "assistant did not claim\n\n"
        "PREVIOUS USER REQUEST:\n" + (prev_user_request or "")[:800] +
        "\n\nASSISTANT'S ANSWER:\n" + (prev_assistant_response or "")[:2000] +
        "\n\nUSER'S NEXT MESSAGE:\n" + (current_user_text or "")[:800] +
        "\n\nAnswer with JSON only: {\"corrects\": true|false}"
    )


def parse_adjudication(text) -> Optional[bool]:
    """Read the judge's answer, or ``None`` when it did not give one.

    ⚠ FAILS CLOSED, and `None` is not `False`. `False` means "a judge looked
    and said no", which SUPPRESSES a promotion the lexical test might have
    made; `None` means "no judge ruled" and restores that test. Collapsing
    the two would let an unparseable reply silently veto real corrections —
    the absent-is-not-withheld rule, one module over.
    """
    import json as _json
    import re as _re
    raw = text if isinstance(text, str) else ""
    m = _re.search(r"\{[^{}]*\}", raw, _re.DOTALL)
    if not m:
        return None
    try:
        val = _json.loads(m.group(0)).get("corrects")
    except Exception:  # noqa: BLE001 — a judge that malforms has not ruled
        return None
    return bool(val) if isinstance(val, bool) else None


def classify_user_correction(
    *,
    prev_user_request: str,
    prev_assistant_response: str,
    current_user_text: str,
    contradicts: Optional[bool] = None,
) -> CorrectionVerdict:
    """Decide whether ``current_user_text`` is a correction of the
    assistant turn that produced ``prev_assistant_response``.

    Two-signal predicate:

      A. Anchored correction-phrase regex on ``current_user_text``.
      B. Token-overlap rephrase: Jaccard(prev_user, current_user)
         over content tokens (stopwords stripped) ≥ threshold.

    Promotion requires Signal A **and** a corroborating second signal.

    ⚠ SIGNAL B WAS A LEXICAL PROXY FOR A SEMANTIC PROPERTY (§4EP,
    2026-09-04). "Did the user re-ask the same question in similar words?"
    is not the question; "does this message assert that my previous answer
    was wrong?" is. Measured over 1601 consecutive live turn pairs, the
    conjunction promoted **2**, while 12 more fired Signal A and were
    blocked — and hand-labelling those 12 found **6 genuine corrections**
    ("No, that's wrong. The correct answer is Canberra", "no I was asking
    about the mysql problem") against 6 non-corrections that merely open
    with "no" (a topic change, a permission grant, an answer to a yes/no
    question). So Signal A alone is ~50% precise: dropping the conjunction
    would have poisoned the corpus, and keeping it costs ~70% of the
    strongest negative label the system can get.

    Neither cheap replacement separates the two classes. Measured on those
    12: Jaccard(current, prior ASSISTANT REPLY) is 0.000-0.167 for real
    corrections and 0.000-0.108 for the others, fully overlapping; "did the
    reply end with a question?" is 4/6 in BOTH classes — exactly zero
    information. Stop patching a proxy and ask the question.

    ``contradicts`` is that answer, supplied by the caller (this function
    stays pure and does no I/O — see `adjudication_prompt`):

      * ``True``/``False`` — a judge ruled; **it decides**, because a
        semantic read of the actual reply outranks a token-overlap guess.
      * ``None`` — no judge ran (offline, no client, timeout, unparseable).
        Falls back to the lexical rephrase test, i.e. **exactly today's
        behaviour**. Every failure path lands here, so the judge can only
        ever ADD promotions, never silently remove the old ones.

    ``prev_assistant_response`` is still not read here — it is the judge's
    subject, and the judge lives at the call site.

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

    # ⚠ ASK THE VERDICT ONCE. When a judge ruled, the lexical test is not
    # consulted at all — two authorities on one question is how a promotion
    # gets granted by whichever signal happened to fire.
    _rephrased = any(s.startswith("rephrase") for s in signals)
    if contradicts is None:
        _corroborated = _rephrased
    else:
        _corroborated = bool(contradicts)
        signals.append(f"adjudicated({'yes' if contradicts else 'no'})")
    is_correction = ("phrase" in signals) and _corroborated

    # Affirmation veto — only consulted when both signals fired, so the
    # common paths pay nothing. A clear affirmation with no negative
    # marker means the user is praising the result, not correcting it,
    # no matter how correction-shaped the opener ("actually …") is.
    if is_correction and _AFFIRMATION_RE.search(cu) \
            and not _NEGATIVE_MARKER_RE.search(cu):
        signals.append("affirmation-veto")
        is_correction = False
        confidence = 0.0

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


# ══════════════════════════════════════════════════════════════════════
# Tier 2: FAILURE REPORTS — how this operator actually says "that's wrong"
# ══════════════════════════════════════════════════════════════════════
#
# `classify_user_correction` above requires a correction PHRASE plus a
# REPHRASE of the original request. Measured over the stored session
# history (2026-07-27): **0 of 246 eligible triples** fired, while 20 of
# them (8.1%) were unmistakable reports that the delivered work was
# broken. The classifier is not malfunctioning — it is structurally blind
# to how corrections actually arrive here:
#
#     "game.js:45 Failed to load frame definitions: TypeError: …"
#     "Failed to load resource: the server responded with a status of 404"
#     "it still does the same. the game never starts"
#     "the pages all show, i enter data, but i have to refresh … fix that."
#
# None contains a rebuttal phrase, and a pasted stack trace shares almost
# no tokens with "build me a game", so neither signal can fire. Yet each
# is direct evidence that the previous turn's output was wrong — roughly
# one turn in twelve, about 1/day. That is the largest untapped source of
# GROUND-TRUTH negatives in the system.
#
# Unlike a correction this needs only ONE signal: pasting a traceback at
# the agent is unambiguous in a way that a bare "no" is not.

# Signal A — a pasted DIAGNOSTIC. Deliberately specific: bare "error" /
# "errors" is excluded because it false-positives on success reports
# ("zero console errors", "no errors now") — the exact inversion that
# turns a compliment into a recorded failure.
_DIAGNOSTIC_RE = re.compile(
    r"(?:Traceback\s*\(most recent call last\)|"
    r"\bUncaught\b|"
    r"\b(?:Type|Reference|Syntax|Value|Key|Index|Attribute|Runtime)Error\b|"
    r"\bException\b(?!\s*(?:handling|safety))|"
    r"\bFailed to (?:load|fetch|start|compile|build)\b|"
    r"\bis not a function\b|\bis not defined\b|\bundefined is not\b|"
    r"\bstatus (?:code )?(?:of )?[45]\d\d\b|\bHTTP [45]\d\d\b|\b[45]0[0-9] \(\)|"
    r"\b\w+\.(?:js|py|ts|tsx|jsx|go|rs|java):\d+\b|"
    r"\bstack trace\b|\bsegmentation fault\b|\bcore dumped\b)",
    re.IGNORECASE,
)

# Signal B is split by STRENGTH, because the praise veto must treat the
# two differently.
#
# WEAK — a bare "fix it / fix that". Genuinely ambiguous: it reads as a
# complaint in "new game doesn't work, please fix it", but as forward-
# looking instruction in "perfect, fix it exactly like that next time".
# Praise is allowed to veto this one.
_BREAKAGE_WEAK_RE = re.compile(
    r"\bfix\s+(?:that|it|this)\b", re.IGNORECASE,
)

# STRONG — an explicit statement that the delivered thing IS broken.
# "still" carries most of the weight: it means a previous fix did not
# take, which is precisely a negative on the turn that claimed to fix it.
# Praise must NOT veto these. Mixed messages are the common shape —
# "the minesweeper right click doesn't work though, but the rest looks
# good" is a defect report with a polite softener, and an earlier version
# of this guard silently discarded exactly that. A softener does not
# cancel a named breakage.
_BREAKAGE_STRONG_RE = re.compile(
    r"(?:\bstill\s+(?:does(?:n[''"
    r"]?t)?|doesn[''"
    r"]?t|not|the same|broken|fails?|crash)|"
    r"\b(?:does|did|do)(?:n[''"
    r"]?t| not)\s+work\b|"
    r"\bwo(?:n[''"
    r"]?t|uld not)\s+(?:start|work|run|load|open|launch)\b|"
    r"\bnot\s+working\b|\bnothing\s+happens\b|\bnever\s+(?:starts?|loads?|runs?)\b|"
    r"\bis\s+broken\b|\bbroke\s+(?:it|the)\b|"
    r"\bi\s+(?:have|need)\s+to\s+refresh\b|"
    r"\bneeds?\s+(?:a\s+)?manual\s+reload\b)",
    re.IGNORECASE,
)


@dataclass
class FailureReportVerdict:
    """Result of :func:`classify_failure_report`.

    ``is_failure_report`` is True iff the user's message reports that the
    agent's delivered work is broken. ``signals`` lists which detectors
    fired (audit), ``reason`` is a short string for the sample record.
    """

    is_failure_report: bool
    confidence: float
    signals: List[str] = field(default_factory=list)
    reason: str = ""


def classify_failure_report(current_user_text: str) -> FailureReportVerdict:
    """Decide whether ``current_user_text`` reports broken delivered work.

    Single-signal by design (see the module note above): either a pasted
    diagnostic OR an explicit breakage statement is sufficient.

    **Direction guard.** The praise veto is the load-bearing part, not the
    matching. A real session message reads:

        "menu-bounce fix works — START GAME now clears all overlays, the
         loop runs at 60fps with zero console errors."

    That contains "fix" and "errors" and is *praise*. Recording it as a
    negative would invert the label — the same class of bug as a tolerant
    parser that silently produces a wrong answer instead of failing loudly.
    An affirmation therefore vetoes the verdict UNLESS a hard diagnostic is
    present (a pasted traceback alongside "works" still means something
    broke).

    Pure: never raises, no I/O.
    """
    cu = current_user_text if isinstance(current_user_text, str) else ""
    if not cu.strip():
        return FailureReportVerdict(False, 0.0, [], "")

    signals: List[str] = []
    confidence = 0.0
    has_diagnostic = bool(_DIAGNOSTIC_RE.search(cu))
    has_strong = bool(_BREAKAGE_STRONG_RE.search(cu))
    has_weak = bool(_BREAKAGE_WEAK_RE.search(cu))
    if has_diagnostic:
        signals.append("diagnostic")
        confidence += 0.6
    if has_strong:
        signals.append("breakage")
        confidence += 0.5
    elif has_weak:
        signals.append("breakage-weak")
        confidence += 0.3

    is_failure = bool(signals)

    # Praise veto. It applies ONLY when the evidence is weak — a bare
    # "fix it" that praise recasts as forward-looking instruction. A hard
    # diagnostic or a named breakage outranks any amount of politeness:
    # "the right click doesn't work though, but the rest looks good" is a
    # defect report with a softener, and vetoing it discards exactly the
    # ground truth this tier exists to capture.
    if is_failure and not has_diagnostic and not has_strong \
            and _AFFIRMATION_RE.search(cu):
        signals.append("affirmation-veto")
        is_failure = False
        confidence = 0.0

    if confidence > 1.0:
        confidence = 1.0
    return FailureReportVerdict(
        is_failure_report=is_failure,
        confidence=confidence if is_failure else 0.0,
        signals=signals,
        reason=("user-reported failure: " + " + ".join(signals)) if is_failure else "",
    )
