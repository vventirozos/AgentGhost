"""Lesson/heuristic actionability gate — the quality filter for the skill
playbook.

The autonomous loops (dream REM, self-play) are asked for imperative
behavioural RULES, but a small worker model also emits OBSERVATIONS —
"The agent is capable of…", "The user frequently…", "On a regex_parse task
that has a familiar shape…" — and sometimes a raw code snippet or a user
PREFERENCE. Those land in the playbook as ``mistake="none"`` pseudo-lessons
that match no real query yet dominate retrieval (measured 2026-07-16: such
entries were 28% of all playbook retrievals, the single most-retrieved item
being a chess persona note). Prompt instructions alone don't hold against a
small model, so a deterministic gate default-REJECTS anything that doesn't
read as an actionable rule.

Lives here (a leaf module, only ``re``) so BOTH the producer side
(``core.dream``) and the write chokepoint (``memory.skills.learn_lesson``)
can share it without an import cycle.
"""
from __future__ import annotations

import re

_HEURISTIC_MIN_LEN = 12
_HEURISTIC_MAX_LEN = 600

# Observation/profile openers — descriptive statements about an actor,
# never instructions. Checked as a prefix of the normalised text.
_HEURISTIC_SUBJECT_BLOCKLIST = (
    "the agent", "this agent", "the user", "this user", "the system",
    "this system", "the model", "the assistant", "the operator",
    "agents ", "users ", "it is ", "there is ", "there are ",
    "requests ", "the request",
)

# First word of an imperative rule ("Always wrap…", "Use absolute paths…").
_HEURISTIC_IMPERATIVE_STARTERS = frozenset({
    "always", "never", "prefer", "avoid", "use", "ensure", "verify",
    "check", "validate", "wrap", "keep", "run", "add", "set", "treat",
    "confirm", "do", "don't", "dont", "remember", "apply", "include",
    "escape", "quote", "pin", "cap", "limit", "strip", "sanitize",
    "sanitise", "batch", "cache", "log", "default", "force", "require",
    "skip", "favor", "favour", "double-check", "re-read", "reread",
    "test", "read", "write", "call", "pass", "return", "handle",
    "guard", "normalize", "normalise", "convert", "parse", "split",
    "sort", "restart", "close", "flush", "await", "retry", "escalate",
    "ask", "state", "make", "stop", "start", "prefix", "compare",
})

# Conditional openers are only rules if an imperative/modal follows
# ("When coaching chess, always name the threat" — yes;
#  "When asked for news, the naftemporiki skill is used" — no).
_HEURISTIC_CONDITIONAL_STARTERS = frozenset({
    "when", "if", "while", "before", "after", "during", "on", "for",
})

_HEURISTIC_MODAL_RE = re.compile(
    r"\b(?:should|must|always|never|use|avoid|prefer|ensure|verify|"
    r"check|validate|wrap|keep|treat|confirm|do not|don'?t|re-?read|"
    r"remember|require|limit|escalate|ask|state)\b",
    re.IGNORECASE,
)

# No plain hyphen in the class: "double-check" / "re-read" must survive
# as single starter tokens (em/en dashes still split).
_HEURISTIC_FIRST_WORD_RE = re.compile(r"[\s,:;—–]+")


def _is_actionable_heuristic(text) -> bool:
    """True iff ``text`` reads as an imperative behavioural rule.

    Default-reject: the reflector and user-correction pipeline carry the
    real mistake-and-fix signal, so a false reject costs little while a
    false accept pollutes the playbook until utility pruning gets to it.
    """
    if not isinstance(text, str):
        return False
    t = " ".join(text.split())
    if not (_HEURISTIC_MIN_LEN <= len(t) <= _HEURISTIC_MAX_LEN):
        return False
    low = t.lower()
    if any(low.startswith(prefix) for prefix in _HEURISTIC_SUBJECT_BLOCKLIST):
        return False
    first = _HEURISTIC_FIRST_WORD_RE.split(low, 1)[0]
    if first in _HEURISTIC_IMPERATIVE_STARTERS:
        return True
    if first in _HEURISTIC_CONDITIONAL_STARTERS:
        rest = low[len(first):]
        return bool(_HEURISTIC_MODAL_RE.search(rest))
    return False


def _is_mistake_less(mistake) -> bool:
    """A 'lesson' with no real mistake is a RULE or an OBSERVATION, not a
    mistake-and-fix correction."""
    return (str(mistake or "").strip().lower() in ("none", "", "n/a"))


# --- conversational-trigger detection (2026-07-18) -------------------------
#
# The 2026-07-16 gate waves through any lesson that records a real mistake.
# The overnight 2026-07-17/18 REM cycle showed the gap: real mistakes
# attached to TRIGGERS lifted verbatim from user chat — "proceed with the
# next task", "it still does the same. the game never starts, notify me in
# slack when…". A trigger is the lesson's retrieval key; raw conversation
# fragments are keys no future query will ever match, so such entries are
# permanent playbook noise no matter how genuine the mistake was.
#
# Default-ACCEPT here (the inverse of the heuristic gate): a false reject
# throws away a real correction, so only unambiguous user-speech tells
# reject. "you/your" is deliberately allowed — rules addressed to the agent
# ("…before you edit") legitimately use it.

# Direct-address phrases — the agent being asked to contact/inform the
# operator mid-conversation ("notify me in slack when…"). Deliberately
# PHRASE-level, not bare pronouns: user-QUESTION triggers ("How do I
# parse JSON?", "Please parse JSON!") are legitimate recurring retrieval
# keys — the paraphrase-normalised dedup counts on them — so a bare
# `\bi\b` / `please` match would reject real corrections.
_TRIGGER_USER_SPEECH_RE = re.compile(
    r"\b(?:notify|tell|ping|remind|message|text|email|send|slack)\s+(?:me|us)\b"
    r"|\blet me know\b",
    re.IGNORECASE,
)

# Pronoun-initial fragments with no antecedent ("it still does the same…")
# and mid-conversation continuations. As PREFIXES of the normalised text.
_TRIGGER_FRAGMENT_STARTERS = (
    "it ", "its ", "it's ", "that ", "this ", "these ", "those ",
    "same ", "still ", "again ", "and ", "but ", "also ",
    "ok ", "okay ", "yes ", "no ", "now ",
)

# Error-signature exemption (2026-07-20): genuine error-keyed triggers
# legitimately start with fragment-starter words — "No module named
# 'requests' when running sandbox scripts", "No such file or directory:
# /workspace/out.csv", "Still ENOENT after the path fix". An error message
# is a prime retrieval key (exactly what the trigger field is for), so a
# fragment-starter prefix only rejects when no error cue is present.
# Cues: canonical errno phrasings, error/exception vocabulary, errno
# constants and CamelCase exception names (case-sensitive on purpose —
# "eat"/"error-free prose" must not match the constant patterns), and
# quoted identifiers ('requests', "utils.py").
_TRIGGER_ERROR_SIGNATURE_RE = re.compile(
    r"(?i:^no (?:module named|such file|matching|attribute|space left)\b)"
    r"|(?i:\b(?:error|errno|exception|traceback|not found|failed|failing|"
    r"denied|refused|timed? ?out)\b)"
    r"|\bE[A-Z]{2,}\b"
    r"|\b[A-Z][a-zA-Z]+(?:Error|Exception|Warning)\b"
    r"|['\"][\w.\-/]{2,}['\"]"
)

# Bare turn-level commands ("proceed with the next task") — instructions
# about the CONVERSATION, not about any recurring technical situation.
# Only short triggers reject on these: a long trigger starting with
# "continue" may legitimately describe a resume-a-job scenario.
_TRIGGER_TURN_COMMANDS = (
    "proceed", "continue", "go ahead", "try again", "do it", "next",
    "carry on", "keep going", "retry", "resume",
)
_TRIGGER_TURN_COMMAND_MAX_LEN = 60


def _is_conversational_trigger(trigger) -> bool:
    """True iff ``trigger`` reads as a raw chat fragment rather than a
    generalisable situation key."""
    if not isinstance(trigger, str):
        return False
    t = " ".join(trigger.split())
    if not t:
        return False
    low = t.lower()
    if _TRIGGER_USER_SPEECH_RE.search(low):
        return True
    if any(low.startswith(s) for s in _TRIGGER_FRAGMENT_STARTERS):
        # Searched on the case-preserved text: the errno/exception-name
        # cues in the regex are deliberately case-sensitive.
        return not _TRIGGER_ERROR_SIGNATURE_RE.search(t)
    if len(low) <= _TRIGGER_TURN_COMMAND_MAX_LEN and any(
        low.startswith(c) for c in _TRIGGER_TURN_COMMANDS
    ):
        return True
    return False


def is_actionable_lesson(mistake, solution, task) -> bool:
    """The lesson-level gate applied at the write chokepoint (2026-07-16).

    A lesson that records a REAL mistake is a genuine correction — keep it
    unless its trigger is a raw chat fragment (2026-07-18; see
    ``_is_conversational_trigger``), since the solution phrasing is
    secondary to the fact that something went wrong and was fixed but the
    trigger must still be a matchable key. A MISTAKE-LESS entry is a rule/observation,
    so its SOLUTION must read as an actionable heuristic; otherwise it is a
    pseudo-lesson (an observation / profile note / snippet) that matches no
    real query yet dominates retrieval.

    Note: ``task`` is accepted but NOT used to reject on ``solution == task``
    — the dream heuristics loop legitimately stores ``task = solution[:80]``,
    so equality is the normal shape of a valid short rule, not a degeneracy.
    """
    if not _is_mistake_less(mistake):
        # Real correction — keep, UNLESS its retrieval key is raw chat
        # (see _is_conversational_trigger above). An empty task/trigger
        # is not conversational and still passes, as before.
        return not _is_conversational_trigger(task)
    return _is_actionable_heuristic(str(solution or "").strip())


__all__ = [
    "_is_actionable_heuristic",
    "is_actionable_lesson",
    "_is_mistake_less",
    "_is_conversational_trigger",
]
