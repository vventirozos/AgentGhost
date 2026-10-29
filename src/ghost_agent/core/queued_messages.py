"""Queued messages: what the user sends while a turn is running (2026-10-28).

Turns are serialized on one model slot, so a second generation cannot run
beside the first. A message that arrives mid-turn is ROUTED by the system
(no model), in this order:

  1. not the turn's own sender (conversation; owner or probe) → QUEUE with
     one reason for every miss. Checked FIRST, so a stop word or a status
     question from anyone else never reaches the turn. Member turns and
     Slack conversations have no inbox (operator 2026-10-28: Slack stays as
     it is) — such a message always waits for its own request.
  2. empty / longer than MAX_TEXT → QUEUE.
  3. a bare stop word → CANCEL, the registry's cooperative cancel (~0.2 s);
     also on a turn already finishing (its streamed final). A turn still
     waiting for the turn lock is not reached (cancelling one ran its
     cleanup beside the running turn — r5); a hold still withdraws its yes
     (tools.memory.note_queued_withdrawal, by time).
  4. a status question and nothing else → STATUS, answered here from the
     registry (with anything more it is an instruction → 7).
  5. a rule command (the system answers those at request start) → QUEUE.
  6. the turn no longer accepts (its loop has exited) → QUEUE.
  7. everything else → INJECT: read at the turn's next model call, the
     model decides now / after / instead. The row starts with
     QUEUED_LABEL, so none of the loop's own detectors (all anchored at the
     start of a row: steers, tool results, the state block) can take it for
     one of its rows, whatever the user typed. A word filter did this for
     three review rounds and leaked each time (§4RZ r3) — it is gone; only
     owner and probe turns have an inbox, and the same text sent as a
     normal message reaches the model the same way.

QUEUE means the client sends it as a normal message, which runs after the
turn — the behaviour before this module. The loop drains the inbox right
before it rebuilds the request, after every exit of the step, so a drained
message is one the model reads. Messages accepted but never drained (the
loop ended first) are LEFTOVERS for that conversation and sender, kept in
memory for LEFTOVER_TTL_S (a restart drops them). Off unless GHOST_QUEUED_MESSAGES=1.

Model-facing (step 2): every line that names the request — the state block,
the alerts' "THE REQUEST YOU ARE ANSWERING", the repair directive, the
REMINDER — says, once a message was delivered, that the user sent N
messages while the model worked and that they are part of the request
(`added_line`); the verifier and the tool-output summariser judge against
`current_ask` (the request plus those messages).
"""

from __future__ import annotations

import contextvars
import os
import re
import time
from typing import Optional, Tuple

PATH_INJECT = "inject"
PATH_CANCEL = "cancel"
PATH_STATUS = "status"
PATH_QUEUE = "queue"

#: turn kinds (utils.logging.request_kind) whose inbox is open — not "member":
#: Slack is left as it is (operator, 2026-10-28)
OPEN_KINDS = ("owner", "probe")

#: the whole message, folded: halt the turn. "cancel that" is NOT here — it
#: may name a background job; the model reads it.
_STOP_RE = re.compile(r"(?:stop|cancel|abort|σταματα|σταματησε|ακυρο|ακυρωσε)[\s.!]*")

#: longest queued message accepted (characters)
MAX_TEXT = 8000
#: messages one turn's inbox holds before refusing
MAX_PENDING = 8
#: leftovers: how long they are kept, and how many conversations hold some.
#: In memory only — a restart drops them.
LEFTOVER_TTL_S = 30 * 60
LEFTOVER_MAX_SESSIONS = 64


def enabled() -> bool:
    return os.environ.get("GHOST_QUEUED_MESSAGES", "0").strip() == "1"


def is_stop_word(text: str) -> bool:
    from ..memory.profile import _fold
    return bool(_STOP_RE.fullmatch(_fold(str(text or "")).strip()))


def is_bare_status_question(text: str) -> bool:
    """A status question and nothing else: anything more ("how's it going?
    btw use metric units") is an instruction and goes to the model."""
    from .agent import is_status_question, _STATUS_QUESTION_RE
    t = str(text or "").strip()
    return is_status_question(t) and not any(map(_is_content, _STATUS_QUESTION_RE.sub("", t)))


def _is_content(ch: str) -> bool:
    """A character that says something: letters, digits, symbols, emoji
    ("are you done? 🛑"), private-use glyphs, and characters newer than this
    Python's Unicode table (Cn: py3.10 knows Unicode 13 — "🫠" is Cn there).
    Spaces, punctuation and invisible characters do not."""
    import unicodedata
    cat = unicodedata.category(ch)
    return cat in ("Co", "Cn") or cat[0] not in "PZC"


#: one reason for every identity miss: "another sender" would tell any key
#: holder that this conversation has someone else's live turn
_NO_TURN = "no running turn for this sender"


def same_sender(turn, *, session_id: str, kind: str) -> bool:
    if turn is None or not turn.running or not turn.session_id:
        return False
    if str(session_id or "") != turn.session_id:
        return False
    return kind in OPEN_KINDS and kind == turn.kind


def route(text: str, turn, *, session_id: str, kind: str) -> Tuple[str, str]:
    """``(path, reason)`` for a message sent to ``turn`` (an ActiveTurn or
    None). The sender's identity is what the POST itself declared. Identity
    is checked FIRST: a stop word or a status question from anyone else
    never reaches the turn."""
    if not same_sender(turn, session_id=session_id, kind=kind):
        return PATH_QUEUE, _NO_TURN
    t = str(text or "").strip()        # the caller posts this same stripped text
    if not t:
        return PATH_QUEUE, "empty"
    if len(t) > MAX_TEXT:
        return PATH_QUEUE, "too long to add to a running turn"
    if is_stop_word(t):
        return PATH_CANCEL, "stop word"
    if is_bare_status_question(t):
        return PATH_STATUS, "status question"
    # A bare yes/no/selection is INJECTED, not queued: injected, a yes
    # confirms nothing (tools.memory._confirm_allowed reads the request's own
    # message, and refuses the request that showed the preview), while
    # queued it would run AFTER the turn — and confirm a preview the user had
    # not seen when they typed it.
    from .failure_replay import is_rule_command
    from ..memory.experience import is_system_command
    if is_rule_command(t) or is_system_command(t):
        return PATH_QUEUE, "a rule command"
    if not turn.accepting:
        return PATH_QUEUE, "the turn is finishing"
    return PATH_INJECT, ""


def _dur(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def status_reply(turn, now: Optional[float] = None) -> str:
    """What the system says to a status question while the turn runs."""
    now = time.time() if now is None else now
    parts = [f"Working on it — {_dur(now - turn.started_at)} so far, step {turn.step + 1}."]
    if turn.tools_now:
        parts.append(f"Running now: {', '.join(turn.tools_now)} "
                     f"(for {_dur(now - turn.tools_since)}).")
    if turn.tools_done:
        parts.append(f"Tool calls finished: {turn.tools_done}.")
    if turn.inbox:
        parts.append(f"{len(turn.inbox)} of your messages wait for the next step.")
    return " ".join(parts)


#: first line of every queued row the model reads (the state block and every
#: alert name it, step 2). The system knows the row by the `_queued` KEY, never by this text
#: (the user can type it); the label is what keeps the loop's start-anchored
#: detectors off the user's text.
QUEUED_LABEL = "[Sent by the user while you were working]"


#: §4RZ step 2: the texts the model was handed during THIS request, oldest
#: first. Request-scoped (a ContextVar: handle_chat starts it empty and
#: restores it in its finally; the drain adds to it). Every model-facing line
#: that names "the request" reads it, so none can say "nothing new arrived"
#: once something did.
_DELIVERED: "contextvars.ContextVar[tuple]" = contextvars.ContextVar("queued_delivered", default=())


def begin_request():
    """A request starts: nothing delivered yet. Returns the reset token."""
    return _DELIVERED.set(())


def end_request(token) -> None:
    try:
        _DELIVERED.reset(token)
    except Exception:  # noqa: BLE001 — another context, a bad token: never raise in a finally
        _DELIVERED.set(())


def set_delivered(texts):
    """Re-enter a request's deliveries in another context (the streamed
    final drains after handle_chat's finally). Returns the reset token."""
    return _DELIVERED.set(tuple(texts or ()))


def note_delivered(text: str) -> None:
    _DELIVERED.set(_DELIVERED.get() + (str(text),))


def delivered() -> tuple:
    return _DELIVERED.get()


def added_line() -> str:
    """The sentence every alert that names the request adds once the user has
    sent something mid-turn ('' before): those messages are PART of the
    request, and how they combine with it."""
    n = len(delivered())
    if not n:
        return ""
    many = n > 1
    return (f" The user also sent {n} message{'s' if many else ''} while you worked "
            f"(the row{'s' if many else ''} starting \"{QUEUED_LABEL}\""
            + ("; an older one may survive only in a summary" if many else "") + "): "
            f"{'they are' if many else 'it is'} part of this request — a correction or change "
            "replaces what it contradicts (a constraint listed from the first message "
            "included), an addition is done as well, and a message that "
            "says to stop or to do something else instead wins. A yes or a confirmation sent "
            "this way confirms no pending preview: ask for it again in your reply.")


ADDED_HEAD = "\n\nThe user added while the work ran:\n"
_EARLIER_NOTE_RE = re.compile(r"^- \(\d+ earlier not shown\)$", re.M)


def without_frame(text: str) -> str:
    """``current_ask``'s text without its own words (the header, the "N earlier
    not shown" note, the "- " bullets): what the user wrote, for a reader that
    takes the request's WORDS as its subject (s2-r6: the header gave the topic
    check the subject words 'added' and 'work')."""
    t = str(text or "")
    # s2-r7: only a frame current_ask wrote (something delivered), and the
    # LAST header — the user's own text may contain the same words
    i = t.rfind(ADDED_HEAD) if delivered() else -1
    if i < 0:
        return t
    rest = _EARLIER_NOTE_RE.sub("", t[i + len(ADDED_HEAD):])
    return t[:i] + "\n" + "\n".join(ln[2:] if ln.startswith("- ") else ln
                                    for ln in rest.split("\n") if ln.strip())


def constraints_note() -> str:
    """The line under a block of constraints read from the request's FIRST
    message, once the user sent more ('' before): a later message may have
    replaced one (s2-r7: the block and the constraint steer kept asserting
    them as MUST HOLD after the user changed them)."""
    if not delivered():
        return ""
    return ("(These come from the user's first message. A message they sent "
            "while you worked replaces any of these it contradicts.)")


def current_ask(first: str, limit: Optional[int] = None, added_max: Optional[int] = None) -> str:
    """What was asked, for readers that judge or summarise against it (the
    verifier, the tool-output summariser): the request, then each message the
    user sent while the turn ran. ``limit``: the whole fits it with the
    added messages kept — the request's text is what gets cut (a reader that
    cuts from the end lost the additions first)."""
    texts = delivered()
    first = str(first or "")
    if not texts:
        return first                 # unchanged: each reader cuts as it always did
    head = ADDED_HEAD
    lines = [f"- {' '.join(t.split())}" for t in texts]
    if limit is None:
        if added_max is not None:    # s2-r3: the request whole, the additions bounded
            return first + head + "\n".join(_newest_lines(lines, added_max))
        return first + head + "\n".join(lines)
    # s2-r2: the request keeps at least its first half of the room (a pasted
    # log as an addition pushed it out whole); the additions get the rest,
    # NEWEST first (it is the one that wins), each line cut at its end
    if limit < len(head) + 80:       # s2-r4/r5: no room for a useful addition: the request alone
        return first[:limit]
    keep = max(limit // 2, limit - len(head) - len("\n".join(lines)))
    if len(first) > keep:
        first = first[:max(0, keep - 1)] + "…"   # s2-r3: a cut request is marked too
    room = limit - len(first) - len(head)
    return (first + head + "\n".join(_newest_lines(lines, room)))[:limit]


def _newest_lines(lines, room: int) -> list:
    """As many lines as fit ``room`` (joined by newlines), NEWEST first kept,
    each cut at its end with "…"."""
    kept = []
    total = room
    if (len("\n".join(lines)) > room and room > 45
            and not room - 30 < len(lines[-1]) <= room):
        # room for the "N earlier not shown" note — never at the cost of
        # cutting a newest line that fits whole (s2-r6)
        room -= 30
    for line in reversed(lines):
        sep = 1 if kept else 0                   # the newline between kept lines
        if room - sep <= 1:
            break
        if len(line) > room - sep and kept and room - sep - 1 < 30:
            # s2-r7/r8: no "- x…" scrap of an older line — under 30 the
            # "N earlier not shown" note (≤ 27) is what that room is for
            break
        line = line if len(line) <= room - sep else line[:room - sep - 1] + "…"
        kept.insert(0, line)
        room -= len(line) + sep
    if len(kept) < len(lines) and kept:
        # s2-r4: older ones left out are said, not silently dropped
        note = f"- ({len(lines) - len(kept)} earlier not shown)"
        if len("\n".join([note] + kept)) <= total:
            kept.insert(0, note)
    return kept


def request_row_index(msgs) -> int:
    """§4RZ step 3: the index of the turn's REQUEST row in ``msgs`` — the row
    handle_chat marks (`_request`); without the mark (it is popped when the
    turn ends, before a streamed final records), the last user row that is
    not one the user sent mid-turn (`_queued`) — which can be a loop steer,
    as "the newest user row" always could: a caller that must not land on a
    steer filters those itself. -1 when there is none. A queued row is never
    the request: readers that took "the newest user row" took the user's
    addition and lost the request."""
    rows = list(msgs or [])
    for i, m in enumerate(rows):
        if isinstance(m, dict) and m.get("_request") and m.get("role") == "user":
            return i
    return max((i for i, m in enumerate(rows) if isinstance(m, dict)
                and m.get("role") == "user" and not m.get("_queued")), default=-1)


def queued_row(item: dict) -> dict:
    """The user row a drained message becomes. `_queued` is a KEY, never a
    text prefix; req_messages drops extra keys. Stored history keeps the
    user's own text (the delivered rows), not the label."""
    return {"role": "user", "content": f"{QUEUED_LABEL}\n{item['text']}", "_queued": True,
            "_queued_at": item["ts"]}



def changed_mid_turn(traj) -> bool:
    """§4RZ s4-r8: did the user send messages while this recorded turn ran
    (`extra.queued_messages`)? Such a turn is RECORDED but never graded,
    judged, replayed or promoted: its request alone is not what it answered,
    and every reader that tried to fit the messages in (s4-r6…r8: the judge's
    next message, head cuts, redaction, small-talk mutes, the next turn's
    walk) read them wrong in its own way. A dict row or a trajectory object."""
    if traj is None:
        return False
    extra = traj.get("extra") if isinstance(traj, dict) else getattr(traj, "extra", None)
    added = extra.get("queued_messages") if isinstance(extra, dict) else None
    return isinstance(added, list) and any(str(a or "").strip() for a in added)
