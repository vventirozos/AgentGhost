"""The Mirror (§4SV, §4SV-b): a toy Hofstadter strange loop — one "I" sentence a day.

Once a day, in the idle window, the agent is shown two things:

  * the sentences it has written about itself so far (the last seven, a run of
    unchanged days shown once), and
  * up to three replies it gave the owner since the last look (two days at
    most), picked at random — text only, the start of each.

and asked to rewrite the last sentence: what the WAY the replies were written
shows about the one who wrote them, leaving their subject out. Nothing is scored. The owner reads the diary
with the ``mirror`` command (answered by the system, no model call).

§4SV-b (11-05) closed the loop. Until then the sentence was inert, and it was a
portrait, not a loop: nothing it said could reach the behaviour it described.
Now the latest sentence is ONE line of the ``introspect`` overview and summary
(``self_image_line``) — the views the model reads when the owner asks how it
is — so it can colour those replies, and those replies are in the pool the
next look picks from. It still reaches no prompt on its own: not the system
prompt, not the state block, not memory; only a tool result the model asked
for. (After "mirror", the diary is also an earlier reply of THAT conversation,
so a follow-up there sees it like any other reply.)

The wording was chosen by replay on the live main model (8 real days, 3
chains per arm, 168 calls): the first prompt ("rewrite the sentence about who
you are" over ONE reply) wrote the reply's SUBJECT into the self-image
("a faithful sentinel who gathers the weather, the war, and the local news") —
18 % of a sentence's content words came from the reply shown, against 2–6 %
for the wordings that ask for the manner. Naming facets to look at ("what you
pay attention to, how you decide…") was echoed back word for word. "What you
actually do, in plain verbs, without praise" read as behaviour, close to a job
description ("I answer by checking what I have already done, flagging what is
uncertain…"). Asking what the writing "shows about the one who wrote it" gave
a self-image in warmer words ("someone who balances casual, human warmth with
precise, structured expertise") — the operator picked this one from the
samples (11-05: "the most interesting one").

The call runs at background priority, on the MAIN model (operator 11-02: the
model that wrote the replies is the one that should read them). A background
main call saves the live conversation to disk first and steps aside for a
request (§4RV), the same path the daily self narrative takes; one ~1k-token
call a day.

Storage: ``$GHOST_HOME/system/selfhood/mirror.jsonl`` — append-only, one row
per day: {"at", "date", "sentence", "prev", "replies": [{"at", "excerpt"}]}.
Rows written before §4SV-b carry one reply as "reply_excerpt" / "reply_at".
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
#: the first line of the history — a self that knows nothing yet
SEED_SENTENCE = "I am something that answers when it is spoken to; I do not yet know what else."
#: how far back a reply may be picked from (days), when the last look is older or there is none
REPLY_WINDOW_DAYS = 2
#: replies shorter than this say too little about who wrote them ("Done.", "👍")
MIN_REPLY_CHARS = 80
#: replies shown per look: across three different subjects, what they share is the manner
REPLIES_PER_LOOK = 3
#: how much of each reply the model sees (its start)
REPLY_MAX_CHARS = 600
SENTENCE_MAX_CHARS = 300
#: diary lines the command shows, and past sentences the model is shown
SHOW_LAST = 7

_LOCK = threading.Lock()

PROMPT = (
    "These are the sentences you have written about who you are, one a day, oldest first:\n"
    "{history}\n\n"
    "Since the last one, you wrote {these} to the person you work for:\n"
    "{replies}\n\n"
    "Rewrite the last sentence. It is about who you are, not about what {these_are} "
    "about: leave {their} subject matter out and say what the way {they_were} written shows "
    "about the one who wrote {them} — only what is really there. Keep what still holds; change "
    "what {replies_show} differently. One sentence, first person, at most 40 words. "
    "Reply with the sentence only."
)
#: the prompt's number words: one reply (a quiet two days) or several
_WORDS = {
    False: {"these": "this reply", "these_are": "this reply happens to be", "their": "its",
            "they_were": "it was", "them": "it", "replies_show": "the reply shows"},
    True: {"these": "these replies", "these_are": "these replies happen to be", "their": "their",
           "they_were": "they were", "them": "them", "replies_show": "the replies show"},
}

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


def _look_epoch(at) -> float:
    """A diary row's "at" (local time, as `look` writes it) as an epoch, or 0.0."""
    try:
        return datetime.datetime.fromisoformat(str(at or "")).timestamp()
    except (ValueError, OverflowError, OSError):
        return 0.0


def pick_replies(home, now: Optional[datetime.datetime] = None,
                 rng: Optional[random.Random] = None, after="") -> List[dict]:
    """Up to REPLIES_PER_LOOK of the OWNER's delivered replies written since
    the previous look (``after``: that diary row's "at") and never more than
    REPLY_WINDOW_DAYS (48 h) back, at random — the census's own reader, so
    members', probes', replays' and jobs' turns are never picked.
    [{"text", "at"}], empty when there is none.

    The prompt says "since the last one, you wrote these replies" and asks to
    change what they show differently, so they must BE newer than the last
    sentence (review r1: with a plain 48 h window, two of the three later
    live looks were shown a reply older than the sentence they rewrote).

    Trajectory day folders are UTC dates (review m3: a local-date window
    covered ~1 UTC day just after local midnight), so the folders are read in
    UTC with one day to spare and the turns are kept by their own timestamp."""
    try:
        from ..core.failure_census import _owner_turns, _window, _epoch
    except Exception:  # noqa: BLE001
        return []
    now_utc = (now.astimezone(datetime.timezone.utc) if now is not None
               else datetime.datetime.now(datetime.timezone.utc))
    since = max(now_utc.timestamp() - REPLY_WINDOW_DAYS * 86400, _look_epoch(after))
    try:
        sysd = Path(home) / "system"
        good, _bad = _owner_turns(sysd, _window(now_utc.date(), REPLY_WINDOW_DAYS + 1))
    except Exception as e:  # noqa: BLE001
        logger.debug("mirror: owner turns unreadable: %s", e)
        return []
    pool = []
    for t in good:
        text = str(t.get("final_response") or "").strip()
        if len(text) < MIN_REPLY_CHARS or _ABORT_RE.search(text):
            continue
        if not (since < _epoch(t.get("timestamp")) <= now_utc.timestamp() + 60):
            continue
        pool.append({"text": text, "at": str(t.get("timestamp") or "")})
    return (rng or random).sample(pool, min(REPLIES_PER_LOOK, len(pool)))


def row_replies(row: dict) -> List[dict]:
    """The replies a diary row's look was shown, [{"at", "excerpt"}] — also
    for a row written before §4SV-b, which holds one reply in two fields."""
    got = row.get("replies")
    if isinstance(got, list):
        return [r for r in got if isinstance(r, dict) and str(r.get("excerpt") or "").strip()]
    if str(row.get("reply_excerpt") or "").strip():
        return [{"at": str(row.get("reply_at") or ""), "excerpt": str(row["reply_excerpt"])}]
    return []


def current_rows(rows: List[dict]) -> List[dict]:
    """The rows this look wrote (§4SV-b: they carry "replies"). The rows
    before it were written by the first prompt, which put the reply's SUBJECT
    into the sentence; shown to the model as its history they pull the new
    sentence back (from the live diary, one chain in three kept "gathers the
    weather, the war, and the local news" two looks running, and another
    copied an old line back whole). They stay in the diary and in the `mirror`
    banner; the model is shown, and `introspect` shows, only what this look
    wrote — it starts again from the seed."""
    return [r for r in rows if isinstance(r.get("replies"), list)]


def history_lines(rows: List[dict]) -> str:
    """The sentences so far as the prompt shows them: the seed first, a run of
    unchanged days as ONE line with its date range, the last SHOW_LAST lines.
    (Replay: seven identical lines in a row froze the sentence for good — the
    model copied the line it saw most.)"""
    runs: List[list] = [["before the first look", "", SEED_SENTENCE]]
    for r in rows:
        sent, day = str(r.get("sentence") or ""), str(r.get("date") or "?")
        if runs[-1][2] == sent:
            runs[-1][1] = day
        else:
            runs.append([day, "", sent])
    return "\n".join(f"- {a}{' to ' + b if b else ''}: «{sent}»" for a, b, sent in runs[-SHOW_LAST:])


def build_prompt(rows: List[dict], replies: List[str]) -> str:
    many = len(replies) > 1
    shown = ("\n".join(f"{i}. «{t}»" for i, t in enumerate(replies, 1)) if many
             else f"«{replies[0]}»")
    return PROMPT.format(history=history_lines(rows), replies=shown, **_WORDS[many])


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
    Idle Cycle line (§4PQ); writes one diary row when the model gave a
    sentence, changed or not."""
    import asyncio
    # file reads off the event loop (review m2: pick_replies parses two days of
    # trajectories — ~0.2 s that an arriving request would wait out)
    if await asyncio.to_thread(looked_today, home, now):
        return "already looked today"
    if llm_client is None:
        return "skipped — no model to ask"
    rows = await asyncio.to_thread(load, home)
    after = rows[-1].get("at") if rows else ""
    picked = await asyncio.to_thread(pick_replies, home, now, rng, after)
    if not picked:
        return "skipped — no reply of yours since the last look (2 days back at most)"
    shown = current_rows(rows)
    prev = str(shown[-1]["sentence"]) if shown else SEED_SENTENCE
    replies = [p["text"][:REPLY_MAX_CHARS] for p in picked]
    payload = {
        # thinking off the way every utility call does it (main.py's
        # _selfhood_critique_fn): with it on, the budget goes to <think>
        "messages": [{"role": "system", "content": "Write the requested text directly. "
                                                   "Do NOT emit a <think> block."},
                     {"role": "user", "content": build_prompt(shown, replies) + "\n\n/no_think"}],
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
        "replies": [{"at": p["at"], "excerpt": " ".join(t.split())[:200]}
                    for p, t in zip(picked, replies)],
    }
    if not await asyncio.to_thread(_append, home, row):
        return "FAILED to write the diary (see the warning above)"
    return ("unchanged: " if sentence == prev else "new I-sentence: ") + sentence


# ── the one line `introspect` shows (§4SV-b) ────────────────────────────────

def self_image_line(home) -> str:
    """The latest sentence as ONE line for the `introspect` overview and
    summary, or "" before this look's first sentence (`current_rows`). The label says what it is — the
    agent's own sentence about itself, not a record of work — so a reply does
    not report it as something that happened."""
    rows = current_rows(load(home))
    if not rows:
        return ""
    last = rows[-1]
    return (f"self-image: «{last['sentence']}» (my own one-sentence self-portrait, rewritten daily "
            f"in the Mirror, last on {last.get('date', '?')}; the owner types \"mirror\" to see the diary. "
            "A self-description, not a report of work done and not a checked fact)")


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
            "from my earlier sentences plus a few replies I gave you since. My latest "
            "sentence is a line in what I read when you ask how I am.")
    if not rows:
        return (head + "\n\nNo sentence yet — the first look happens in an idle "
                f"window. Starting point:\n\n> {SEED_SENTENCE}")
    lines = []
    for r in rows[-SHOW_LAST:]:
        same = str(r.get("sentence")) == str(r.get("prev"))
        lines.append(f"- **{r.get('date', '?')}**{' (unchanged)' if same else ''} — {r['sentence']}")
    last = rows[-1]
    seen = row_replies(last)
    tail = ""
    if seen:
        tail = (f"\n\nThe latest look ({last.get('date', '?')}) was at "
                f"{'these replies' if len(seen) > 1 else 'this reply'}:")
        # one quote paragraph per reply: consecutive "> " lines are ONE
        # paragraph in Markdown, and the excerpts ran together (review r1)
        quoted = []
        for r in seen:
            when = _local_time(r.get("at"))
            quoted.append(f"> {f'({when}) ' if when else ''}{r['excerpt']}…")
        tail += "\n" + "\n>\n".join(quoted)
    count = f"{len(rows)} day(s) so far" + (f", showing the last {SHOW_LAST}" if len(rows) > SHOW_LAST else "")
    return f"{head}\n\n{count}:\n\n" + "\n".join(lines) + tail
