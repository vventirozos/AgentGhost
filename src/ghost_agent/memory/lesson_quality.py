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


def is_actionable_lesson(mistake, solution, task) -> bool:
    """The lesson-level gate applied at the write chokepoint (2026-07-16).

    A lesson that records a REAL mistake is a genuine correction — always
    keep it (its solution phrasing is secondary to the fact that something
    went wrong and was fixed). A MISTAKE-LESS entry is a rule/observation,
    so its SOLUTION must read as an actionable heuristic; otherwise it is a
    pseudo-lesson (an observation / profile note / snippet) that matches no
    real query yet dominates retrieval.

    Note: ``task`` is accepted but NOT used to reject on ``solution == task``
    — the dream heuristics loop legitimately stores ``task = solution[:80]``,
    so equality is the normal shape of a valid short rule, not a degeneracy.
    """
    if not _is_mistake_less(mistake):
        return True
    return _is_actionable_heuristic(str(solution or "").strip())


__all__ = ["_is_actionable_heuristic", "is_actionable_lesson", "_is_mistake_less"]
