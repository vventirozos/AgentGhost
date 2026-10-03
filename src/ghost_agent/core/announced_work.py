"""§4KW — did a tool-free reply only announce work it never did?

The §4IW guard gives the model one continuation when a turn ends on an
announcement (since the §4KW follow-up, also after tools ran — the worker is
told which) ("Let me search…", "Θα κάνω ένα poster…"). Its
detector, `reply_smoothing.narration_only`, is a list of openers and work
verbs; on the recorded corpus it caught 1 of the 4 real announcements in 155
tool-free replies — req slack-3120ad1e ("Έχεις δίκιο … Θα κάνω ένα poster …",
the member had to write "you didn't provide a poster") and slack-541256be
("*Investigating cache hit rate…* … Let me research this specifically.", no
research ever happened) were both missed. Adding "research" and "κάνω" to the
list fixes those two sentences and misses the next phrasing.

So the question is asked instead, of the small worker model, ONLY when the
list says no: one bounded YES/NO about this reply and this request.

Measured before shipping (worker = Gemma 4 E4B on Nova):
  * recorded corpus, 155 tool-free replies with a request id: 4/4 real
    announcements, 0 false alarms on the other 151 (the first wording had 3 —
    clarifying questions — and was replaced);
  * a hand-written held-out set, 3 runs: 9/12 announcements, 0/16 false
    alarms (refusals, "Yes. I'll check it tomorrow.", a clarifying question,
    "Let me be clear: …", a Greek offer); the 3 misses are short shapes the
    list already catches;
  * cost: ~1.0 s per checked reply (median, one at a time, uncached; 1.4 s
    p90; 3.0 s median with 3 in parallel); bounded at CHECK_BUDGET_S.
    A wording with the fixed question first (cacheable) was measured and
    REJECTED: no faster (1.0 s) and 24 vs 11 YES on the tool-turn replies,
    mostly false alarms on system notes.

A false alarm costs one extra generation whose directive allows "answer
directly"; a miss ships the announcement (the pre-§4KW behaviour). The check
is skipped — the old behaviour — when no worker is configured, the worker
fails or times out, or its answer is not YES/NO; and (agent.py,
`_worker_check_applies`) with GHOST_ANNOUNCED_WORK_CHECK=0, on sim and bench
turns, on the §4KV thinking-loop retry, about the same reply twice, more than
twice a request, or once a pending-promise steer was given. It is awaited inside the turn, so its latency is the user's.
"""
from __future__ import annotations

import re
from typing import Any, Optional

#: Replies longer than this are not checked: every announcement in the corpus
#: was under 400 chars, and the check's cost is paid per reply.
MAX_CHECKED_CHARS = 800
#: Seconds the user may wait for the check; past it the check is skipped (the
#: pre-§4KW behaviour). Measured: ~1.0 s median, 1.4 s p90 uncached, 0.7–1.1 s
#: live. ⚠ ENFORCED FROM OUTSIDE, BY CANCELLATION — never as route()'s own
#: timeout: a short budget there is under llm._MIN_HTTP_FLOOR (every call
#: declined) or a slow answer charged as a NODE FAULT on the breaker the
#: worker shares with the critic (review, reproduced). Our cancellation is
#: never charged; route()'s own deadline charges only a node that held our
#: request alone and stayed silent (`_charge_route_deadline_miss`).
CHECK_BUDGET_S = 3.0
#: route()'s own budget. Under the outer CHECK_BUDGET_S it never applies, so
#: this check never charges a node (fresh review: by design — a hung node is
#: caught by route()'s default callers and the critic).
ROUTE_TIMEOUT_S = 12.0

_SYSTEM = "You check one assistant reply. Answer with exactly one word: YES or NO."
_VERDICT_RE = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)


def check_prompt(request: str, reply: str, *, tools_ran: bool = False) -> str:
    """The question, exactly as measured (wording v2 — the first one flagged
    clarifying questions). ``tools_ran``: the request already ran tools —
    the second sentence says so (measured on the 617 recorded tool-turn
    replies of ≤800 chars: 11 YES, 7 real announcements that had shipped,
    1 false alarm — "Shall I proceed …?" — and 3 shapes this guard never
    sees: an aborted turn, a forced final, a leaked call)."""
    situation = ("The assistant used some tools during this request, and the reply below is all the "
                 "user received at the end." if tools_ran else
                 "The assistant replied WITHOUT using any tool, and the reply below is all the "
                 "user received.")
    return (
        "A user sent a message to an AI assistant that can use tools (web search, image generation, "
        f"code execution). {situation}\n\n"
        f"USER MESSAGE:\n{str(request or '')[:600]}\n\nASSISTANT REPLY:\n{str(reply or '')}\n\n"
        "Question: does the reply END by announcing work the assistant is about to do now — a search, an "
        "investigation, an image, a script (\"Let me search for…\", \"I'll create the image…\", \"Θα κάνω…\", "
        "\"Investigating…\") — without having done it?\n"
        "Answer NO if the reply answers, explains, refuses, asks the user something, or offers further help.\n"
        "Answer YES or NO.")


def parse_verdict(text: Any) -> Optional[bool]:
    """True for YES, False for NO, None for anything else."""
    if not isinstance(text, str):
        return None
    m = _VERDICT_RE.match(text.strip())
    if not m:
        return None
    return m.group(1).lower() == "yes"


async def worker_finds_announcement(llm_client: Any, request: str, reply: str, *,
                                    tools_ran: bool = False, model: str = "default") -> bool:
    """True only when the worker model answers YES. Never raises; False when
    the check cannot run (no worker pool, reply empty or too long, a failed
    or unparseable call)."""
    try:
        text = str(reply or "").strip()
        if not text or len(text) > MAX_CHECKED_CHARS:
            return False
        if llm_client is None or not getattr(llm_client, "worker_clients", None):
            return False
        payload = {
            "model": model or "default",
            "messages": [{"role": "system", "content": _SYSTEM},
                         {"role": "user", "content": check_prompt(request, text, tools_ran=tools_ran)}],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        from .llm import RoutingTask     # shown in the stream as "check announcement"
        from ..utils.aio import wait_for
        result = await wait_for(llm_client.route(
            task=RoutingTask.CHECK_ANNOUNCEMENT, payload=payload, max_tokens=4, temperature=0.0,
            fallback=None, timeout=ROUTE_TIMEOUT_S, total_budget=ROUTE_TIMEOUT_S),
            CHECK_BUDGET_S)
        return parse_verdict(result) is True
    except Exception:  # noqa: BLE001 — a check never costs the turn
        return False
