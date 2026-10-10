"""§4MT: the NEXT-MESSAGE REACTION judge — does the owner's next message show
that the previous reply was wrong or did not do what was asked?

Operator decision (2026-10-09): "rule now, judge after calibration". The
lexical correction gate opened on 4 of 170 adjacent pairs (§4MT signals
lens); the owner rarely says "wrong" — they re-ask, narrow, or repeat. This
judge reads the request, the reply and the next message, and answers one
word. It is a SEED signal only (`owner_seeds`) — never an outcome label.

Calibration (2026-10-09, the 26 hand-graded §4MM turns that have an adjacent
owner next message; main model, no thinking, temperature 0; two runs agreed
26/26): precision 5/6, recall 5/17. 4 of the 5 caught failures were missed by
the suspect-failure rule; rule ∪ judge: recall 9/17, precision 9/10. It is
ON because its precision matches the rule's (0.83 vs 0.9) and it adds
failures the rule misses — a bar chosen AFTER this one measurement, on six
positives (the precision interval is wide). A false seed costs one practice
run, and no lesson from it is kept without its proof. Re-measure at the next
hand-graded census; ``GHOST_REACTION_JUDGE=0`` turns it off.

``judge_prompt`` is the single source of the prompt (calibration script and
production use the same text); ``parse_verdict`` reads the reply.
"""
from __future__ import annotations

import os
import re
from typing import Optional

_REPLY_CHARS = 1800
_TEXT_CHARS = 700

SYSTEM = (
    "You review one exchange between a user and an AI assistant, plus the "
    "user's NEXT message. Decide whether the next message shows that the "
    "assistant's reply was WRONG or did NOT do what the user asked: the user "
    "corrects it, disputes a fact, says it failed or is missing something, "
    "repeats or narrows the same request because the answer was not usable, "
    "or asks again for what was already requested. A next message that moves "
    "on to a new topic, asks a natural follow-up building on a usable answer, "
    "thanks the assistant, or gives a new instruction is NOT a failure. "
    "Answer with exactly one word: FAILED or OK."
)


def enabled() -> bool:
    return os.environ.get("GHOST_REACTION_JUDGE", "1").strip() != "0"


def judge_prompt(request: str, reply: str, next_message: str) -> list:
    """Chat messages for one judgement."""
    req = str(request or "")[:_TEXT_CHARS]
    rep = str(reply or "")
    if len(rep) > _REPLY_CHARS:
        rep = rep[: _REPLY_CHARS // 2] + "\n…\n" + rep[-_REPLY_CHARS // 2:]
    nxt = str(next_message or "")[:_TEXT_CHARS]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"USER REQUEST:\n{req}\n\nASSISTANT REPLY:\n{rep}\n\n"
            f"USER'S NEXT MESSAGE:\n{nxt}\n\nOne word — FAILED or OK:")},
    ]


def parse_verdict(text: str) -> Optional[bool]:
    """True = the reply failed, False = OK, None = unreadable."""
    t = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S).strip().upper()
    m = re.search(r"\b(FAILED|OK)\b", t)
    if not m:
        return None
    return m.group(1) == "FAILED"
