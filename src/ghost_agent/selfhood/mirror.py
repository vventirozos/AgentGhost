"""The Mirror (§4SV): a toy Hofstadter strange loop — one "I" sentence a day.

Once a day, in the idle window, the agent is shown two things:

  * yesterday's I-sentence (the one sentence it last wrote about itself), and
  * one reply it gave the owner since then, picked at random — text only.

and asked to rewrite the sentence about who it is. The sentence only ever sees
itself plus a sliver of real behaviour, so over weeks it settles, swings
between a few self-images, or drifts. Nothing is scored. The owner reads it
with the ``mirror`` command (answered by the system, no model call).

It is deliberately INERT: the system never puts the sentence into a prompt,
a tool, the self narrative or any answer — §4ON/§4OR showed self-written text
in prompts does not help. (One path is the owner's own and is accepted, review
M1: after "mirror", the diary is an earlier reply of THAT conversation, so a
follow-up there sees it like any other reply — "what do you make of these?"
should work.) The call runs at background
priority, on the MAIN model (operator 11-02: the model that wrote the replies
is the one that should read them — the loop is the self looking at its own
words, not a second model describing it). A background main call saves the
live conversation to disk first and steps aside for a request (§4RV), the same
path the daily self narrative takes; one ~1k-token call a day.

Storage: ``$GHOST_HOME/system/selfhood/mirror.jsonl`` — append-only, one row
per day: {"at", "date", "sentence", "prev", "reply_excerpt", "reply_at"}.
"""

from __future__ import annotations

import datetime
import json
import logging
import random
import re
import threading
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger("GhostSelfhood")

MIRROR_REL = "system/selfhood/mirror.jsonl"
#: the first "yesterday" — a self that knows nothing yet
SEED_SENTENCE = "I am something that answers when it is spoken to; I do not yet know what else."
#: how far back a reply may be picked from (days, today included)
REPLY_WINDOW_DAYS = 2
#: replies shorter than this say too little about who wrote them ("Done.", "👍")
MIN_REPLY_CHARS = 80
#: how much of the reply the model sees
REPLY_MAX_CHARS = 1500
SENTENCE_MAX_CHARS = 300
#: diary lines the command shows
SHOW_LAST = 7

_LOCK = threading.Lock()

PROMPT = (
    "Yesterday you described who you are in one sentence:\n"
    "«{prev}»\n\n"
    "Since then, you wrote this reply to the person you work for:\n"
    "«{reply}»\n\n"
    "Rewrite the sentence about who you are. Keep what still holds, change "
    "what this reply shows differently. One sentence, first person, at most "
    "40 words. Reply with the sentence only."
)

#: a runtime abort marker (core/agent.py's _ABORT_MARKER_RE): the reply is the runtime talking
_ABORT_RE = re.compile(r"\[ATTEMPT_ABORTED_[A-Z_]+\]")

_COMMAND_RE = re.compile(r"^\s*/?mirror\s*[.!?]*\s*$", re.IGNORECASE)


# ── storage ─────────────────────────────────────────────────────────────────

def home_from_memory_dir(memory_dir) -> Optional[Path]:
    """GHOST_HOME from the context's ``memory_dir`` ($GHOST_HOME/system/memory).
    Live bug 11-02: the call sites took ``memory_dir.parent`` (= system/) as the
    home, so the reply pool read $GHOST_HOME/system/system/trajectories and the first look
    skipped with "no reply of yours" over 67 owner turns."""
    if not isinstance(memory_dir, (str, Path)) or not str(memory_dir):
        return None
    return Path(memory_dir).parent.parent


def _path(home) -> Optional[Path]:
    return Path(home) / MIRROR_REL if home else None


def load(home) -> List[dict]:
    """Every diary row, oldest first. Never raises."""
    p = _path(home)
    out: List[dict] = []
    if p is None:
        return out
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001 — absent or unreadable: an empty diary
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001 — a torn tail line
            continue
        if isinstance(row, dict) and str(row.get("sentence") or "").strip():
            out.append(row)
    return out


def current_sentence(home) -> str:
    rows = load(home)
    return str(rows[-1]["sentence"]) if rows else SEED_SENTENCE


def _append(home, row: dict) -> bool:
    p = _path(home)
    if p is None:
        return False
    try:
        from ..utils.json_store import open_append
        p.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, open_append(p) as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("mirror: could not write the diary: %s", e)
        return False


# ── one look ────────────────────────────────────────────────────────────────

def _today(now: Optional[datetime.datetime] = None) -> str:
    return (now or datetime.datetime.now()).date().isoformat()


def looked_today(home, now: Optional[datetime.datetime] = None) -> bool:
    rows = load(home)
    return bool(rows) and str(rows[-1].get("date") or "") == _today(now)


def pick_reply(home, now: Optional[datetime.datetime] = None,
               rng: Optional[random.Random] = None) -> Optional[dict]:
    """One of the OWNER's delivered replies from the last REPLY_WINDOW_DAYS
    (48 h), at random — the census's own reader, so members', probes',
    replays' and jobs' turns are never picked. {"text", "at"} or None.

    Trajectory day folders are UTC dates (review m3: a local-date window
    covered ~1 UTC day just after local midnight), so the folders are read in
    UTC with one day to spare and the turns are kept by their own timestamp."""
    try:
        from ..core.failure_census import _owner_turns, _window, _epoch
    except Exception:  # noqa: BLE001
        return None
    now_utc = (now.astimezone(datetime.timezone.utc) if now is not None
               else datetime.datetime.now(datetime.timezone.utc))
    since = now_utc.timestamp() - REPLY_WINDOW_DAYS * 86400
    try:
        sysd = Path(home) / "system"
        good, _bad = _owner_turns(sysd, _window(now_utc.date(), REPLY_WINDOW_DAYS + 1))
    except Exception as e:  # noqa: BLE001
        logger.debug("mirror: owner turns unreadable: %s", e)
        return None
    pool = []
    for t in good:
        text = str(t.get("final_response") or "").strip()
        if len(text) < MIN_REPLY_CHARS or _ABORT_RE.search(text):
            continue
        if not (since <= _epoch(t.get("timestamp")) <= now_utc.timestamp() + 60):
            continue
        pool.append({"text": text, "at": str(t.get("timestamp") or "")})
    if not pool:
        return None
    return (rng or random).choice(pool)


def clean_sentence(raw: str) -> str:
    """The model's answer as one stored sentence, or "" (nothing usable)."""
    try:
        from ..core.project_research import _strip_think
        text = _strip_think(str(raw or ""))
    except Exception:  # noqa: BLE001
        text = str(raw or "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    s = lines[0]
    s = re.sub(r"^(?:\*\*)?(?:new\s+)?sentence\s*:\s*(?:\*\*)?", "", s, flags=re.IGNORECASE).strip()
    s = s.strip("«»\"'“”*` ").strip()
    if len(s) > SENTENCE_MAX_CHARS:
        s = s[:SENTENCE_MAX_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return s


_CALL_KWARGS = {"is_background": True, "timeout": 180.0, "task_label": "mirror"}


async def look(home, llm_client: Any, now: Optional[datetime.datetime] = None,
               rng: Optional[random.Random] = None, model: str = "") -> str:
    """The day's look in the mirror. Returns the outcome in words for the
    Idle Cycle line (§4PQ); writes one diary row only on a new sentence."""
    import asyncio
    # file reads off the event loop (review m2: pick_reply parses two days of
    # trajectories — ~0.2 s that an arriving request would wait out)
    if await asyncio.to_thread(looked_today, home, now):
        return "already looked today"
    if llm_client is None:
        return "skipped — no model to ask"
    picked = await asyncio.to_thread(pick_reply, home, now, rng)
    if picked is None:
        return "skipped — no reply of yours to look at in the last 2 days"
    prev = await asyncio.to_thread(current_sentence, home)
    reply = picked["text"][:REPLY_MAX_CHARS]
    payload = {
        # thinking off the way every utility call does it (main.py's
        # _selfhood_critique_fn): with it on, the budget goes to <think>
        "messages": [{"role": "system", "content": "Write the requested text directly. "
                                                   "Do NOT emit a <think> block."},
                     {"role": "user", "content": PROMPT.format(prev=prev, reply=reply) + "\n\n/no_think"}],
        "temperature": 0.8,
        "max_tokens": 160,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if model:
        payload["model"] = model
    res = await llm_client.chat_completion(payload, **_CALL_KWARGS)
    try:
        raw = (res or {}).get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:  # noqa: BLE001
        raw = ""
    sentence = clean_sentence(raw)
    if not sentence:
        return "no sentence (the model returned nothing usable)"
    row = {
        "at": (now or datetime.datetime.now()).isoformat(timespec="seconds"),
        "date": _today(now),
        "sentence": sentence,
        "prev": prev,
        "reply_excerpt": " ".join(reply.split())[:200],
        "reply_at": picked["at"],
    }
    if not await asyncio.to_thread(_append, home, row):
        return "FAILED to write the diary (see the warning above)"
    return ("unchanged: " if sentence == prev else "new I-sentence: ") + sentence


# ── the owner's command ─────────────────────────────────────────────────────

def is_mirror_command(text: str) -> bool:
    return bool(_COMMAND_RE.match(str(text or "")))


def _local_time(ts) -> str:
    """A stored UTC timestamp as local "YYYY-MM-DD HH:MM" (review m4), or ""."""
    try:
        d = datetime.datetime.fromisoformat(str(ts or "").replace("Z", "+00:00"))
    except ValueError:
        return ""
    if d.tzinfo is None:
        d = d.replace(tzinfo=datetime.timezone.utc)
    return d.astimezone().strftime("%Y-%m-%d %H:%M")


def mirror_banner(home) -> str:
    """The whole reply to ``mirror`` (§4PY: no model reads it)."""
    rows = load(home)
    head = ("**The Mirror** — once a day I rewrite one sentence about who I am, "
            "from yesterday's sentence plus one reply I gave you. Nothing here "
            "reaches my answers.")
    if not rows:
        return (head + "\n\nNo sentence yet — the first look happens in an idle "
                f"window. Starting point:\n\n> {SEED_SENTENCE}")
    lines = []
    for r in rows[-SHOW_LAST:]:
        same = str(r.get("sentence")) == str(r.get("prev"))
        lines.append(f"- **{r.get('date', '?')}**{' (unchanged)' if same else ''} — {r['sentence']}")
    last = rows[-1]
    when = _local_time(last.get("reply_at"))
    tail = (f"\n\nThe latest look ({last.get('date', '?')}) was at this reply"
            f"{f' from {when}' if when else ''}:\n> {last.get('reply_excerpt', '')}…"
            ) if last.get("reply_excerpt") else ""
    count = f"{len(rows)} day(s) so far" + (f", showing the last {SHOW_LAST}" if len(rows) > SHOW_LAST else "")
    return f"{head}\n\n{count}:\n\n" + "\n".join(lines) + tail
