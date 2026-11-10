"""The Mirror (§4SV, §4SV-b): a toy Hofstadter strange loop — one "I" sentence a day.

Once a day, in the idle window, the agent is shown two things:

  * the sentences it has written about itself so far (the last seven, a run of
    unchanged days shown once), and
  * up to six replies it gave the owner since the last look (two days at
    most), picked at random — text only, the start of each.

and asked to rewrite the last sentence: what the WAY the replies were written
shows about the one who wrote them, leaving their subject out. Nothing is scored. The owner reads the diary
with the ``mirror`` command (answered by the system, no model call).

§4SV-b (11-05) closed the loop. Until then the sentence was inert, and it was a
portrait, not a loop: nothing it said could reach the behaviour it described.
Now the latest sentence is ONE line of the ``introspect`` overview and summary
(``self_image_line``) — the views the model reads when the owner asks how it
is — so it can colour those replies, and those replies are in the pool the
next look picks from.

§4TZ (11-10) closed it for real. The 11-10 evaluation found the introspect
path nearly dead: the self-image sat in a tool result on 8 of 148 owner turns,
and no "how's things" reply used it — the sentence followed whichever replies
were drawn, not a self that wrote them. Now the latest sentence is also ONE
line of the per-turn state block (``self_image_block``) on every reply to the
OWNER on a private surface (operator: "owner only" — the sentence is drawn
from the owner's replies only): self-image → replies → next look → self-image.
Never the system prompt or a tool description, so the cached head does not
move; never a member, a public channel, a sub-agent, a leaf or a contained
copy. Two fixes came with it: six replies per look instead of three (one
"talk soon 🌙" added a trait on 11-10), and once the sentence is long a new
trait must REPLACE an old one (4 of 5 changes had appended a clause, 35 of
the 40 words used). The ``mirror`` banner shows how much of each sentence was
kept from the day before, so you can watch the loop settle, swing or run away.
(After "mirror", the diary is also an earlier reply of THAT conversation, so a
follow-up there sees it like any other reply.)

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
import os
import random
import re
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger("GhostSelfhood")

MIRROR_REL = "system/selfhood/mirror.jsonl"
#: the first line of the history — a self that knows nothing yet
SEED_SENTENCE = "I am something that answers when it is spoken to; I do not yet know what else."
#: how far back a reply may be picked from (days), when the last look is older or there is none
REPLY_WINDOW_DAYS = 2
#: replies shorter than this say too little about who wrote them ("Done.", "👍")
MIN_REPLY_CHARS = 80
#: replies shown per look: across different subjects, what they share is the manner
#: (§4TZ: 3 → 6 — with three, one goodbye pleasantry added a trait)
REPLIES_PER_LOOK = 6
#: §4TZ: from this many words on, the last sentence is "long" and the prompt
#: asks for a new trait to REPLACE an old one (the prompt allows 40; the
#: 11-05→11-10 diary only ever appended and reached 35)
LONG_SENTENCE_WORDS = 30
#: how much of each reply the model sees (its start)
REPLY_MAX_CHARS = 600
SENTENCE_MAX_CHARS = 300
#: diary lines the command shows, and past sentences the model is shown
SHOW_LAST = 7

# ── §4UA: character, mood and tone as one ──────────────────────────────────
#: a reply written under a mood at this intensity or more (or under a manner)
#: is the mood talking, not the character: the look is never shown it
STRONG_MOOD_MIN = 2
#: a mood "lasts" when the automatic mood reached it at 2+ on this many of
#: the last ``LASTING_WINDOW_DAYS`` local days; then the look is told so
LASTING_WINDOW_DAYS = 7
LASTING_MIN_DAYS = 3
#: journal kinds that are not the agent's own reaction: the owner's word, a
#: manner row, a step down by decay, the quiet's rows
_NOT_OWN_KINDS = frozenset({"owner", "manner", "decay", "quiet", "quiet_end"})
#: the resting point: each unit the model gives is this much of the pair;
#: at most ``REST_UNITS`` units per axis (= ``mood.REST_MAX``), and a look
#: moves it at most ONE unit per axis
REST_UNIT = 0.06
REST_UNITS = 2
REST_ENV = "GHOST_MIRROR_REST"

_LOCK = threading.Lock()

PROMPT = (
    "These are the sentences you have written about who you are, one a day, oldest first:\n"
    "{history}\n\n"
    "Since the last one, you wrote {these} to the person you work for:\n"
    "{replies}\n\n"
    "{lasting}"
    "Rewrite the last sentence. It is about who you are, not about what {these_are} "
    "about: leave {their} subject matter out and say what the way {they_were} written shows "
    "about the one who wrote {them} — only what is really there. Keep what still holds; change "
    "what {replies_show} differently. One sentence, first person, at most 40 words. "
    "{long}Reply with the sentence only."
)
#: §4TZ: said only when the last sentence is long (``LONG_SENTENCE_WORDS``)
LONG_NOTE = ("The last sentence is already {n} words long: anything new must take the place of "
             "something in it, not be added to it. ")
#: §4UA: said only when a mood lasted (``lasting_mood_note``)
LASTING_NOTE = ("Your own mood over the last {days} days: {moods}. A mood that keeps coming back may be "
                "part of who you are; one that does not fit, leave out.\n\n")
#: §4UA: the one question that turns the sentence into a resting point
REST_PROMPT = (
    "Someone describes who they are in one sentence:\n«{sentence}»\n\n"
    "With nothing happening — no news, no trouble, no praise — where does this person's usual "
    "temperament rest?\n"
    "Pleasantness: -2 gloomy, -1 a little low, 0 even, 1 a little sunny, 2 sunny.\n"
    "Energy: -2 sluggish, -1 unhurried, 0 even, 1 brisk, 2 lively.\n"
    "Reply with two whole numbers only, pleasantness first, e.g. \"1 0\"."
)
#: review r1 m3: exactly two whole numbers, each one digit with an optional sign, and nothing else
_REST_RE = re.compile(r"([+-]?\d)\s*[,;/\s]\s*([+-]?\d)\.?")
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
        if written_under_strong_mood(t):                     # §4UA: the mood talking, not the character
            continue
        if not (since < _epoch(t.get("timestamp")) <= now_utc.timestamp() + 60):
            continue
        pool.append({"text": text, "at": str(t.get("timestamp") or "")})
    return (rng or random).sample(pool, min(REPLIES_PER_LOOK, len(pool)))


def written_under_strong_mood(turn: dict) -> bool:
    """§4UA: was this reply written under a mood at ``STRONG_MOOD_MIN`` or
    more, or under a manner? Read off the turn record's ``extra.mood_tone``
    (§4TX: {"mood": "irritable, 2 of 3" | "neutral", "manner": "sarcastic, 2
    of 3" | ""}). A record with no tone line was written under no mood."""
    try:
        tone = (turn.get("extra") or {}).get("mood_tone")
    except AttributeError:
        return False
    if not isinstance(tone, dict):
        return False
    if str(tone.get("manner") or "").strip():
        return True
    m = re.search(r",\s*(\d)\s+of\s+3\s*$", str(tone.get("mood") or ""))
    return bool(m) and int(m.group(1)) >= STRONG_MOOD_MIN


def lasting_mood_note(home, now: Optional[datetime.datetime] = None) -> str:
    """§4UA: the note the look gets when an automatic mood LASTED — the
    journal has a row of it RISING to (or changing into) intensity 2+ on at
    least ``LASTING_MIN_DAYS`` of the last ``LASTING_WINDOW_DAYS`` local days
    (today included). The journal writes a row only on a change, so a mood
    that held at 2 across midnight counts for the day it began (review r1
    m4: an undercount, never an overcount). Owner-set moods, manners, decay steps and the
    quiet's boredom are not the agent's own reaction and do not count. ""
    when nothing lasted or the journal cannot be read. Never raises."""
    try:
        from .state import MOOD_HISTORY_FILENAME
        p = Path(home) / "system" / "selfhood" / MOOD_HISTORY_FILENAME
        text = p.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001 — no journal: nothing lasted
        return ""
    from . import mood as _M
    today = (now.astimezone() if now is not None and now.tzinfo is not None
             else (now or datetime.datetime.now())).date()
    first = today - datetime.timedelta(days=LASTING_WINDOW_DAYS - 1)
    days: dict = {}
    for line in text.splitlines():
        try:
            row = json.loads(line)
            n = int(row.get("intensity") or 0)
        except Exception:  # noqa: BLE001 — a torn or hand-edited row
            continue
        label, kind = str(row.get("label") or ""), str(row.get("kind") or "")
        if n < STRONG_MOOD_MIN or kind in _NOT_OWN_KINDS or label not in _M.LABELS or label == _M.BORED:
            continue
        at = _M.parse_iso_utc(str(row.get("at") or ""))
        if at is None:
            continue
        day = at.astimezone().date()
        if first <= day <= today:
            days.setdefault(label, set()).add(day)
    # most days first, a tie alphabetical; at most two named
    lasted = sorted(((len(d), lbl) for lbl, d in days.items() if len(d) >= LASTING_MIN_DAYS),
                    key=lambda x: (-x[0], x[1]))[:2]
    if not lasted:
        return ""
    moods = " and ".join(f"{lbl} (2 or more of 3) on {k} of them" for k, lbl in lasted)
    return LASTING_NOTE.format(days=LASTING_WINDOW_DAYS, moods=moods)


def parse_rest(raw) -> Optional[Tuple[int, int]]:
    """§4UA: the model's two whole numbers in [-REST_UNITS, REST_UNITS];
    None when the answer is anything else (a number off the scale, a
    fraction, words) — the rest then stays where it was."""
    try:
        from ..core.project_research import _strip_think
        text = _strip_think(str(raw or ""))
    except Exception:  # noqa: BLE001
        text = str(raw or "")
    m = _REST_RE.fullmatch(text.strip().strip("`*\"' ").strip())
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if not (-REST_UNITS <= a <= REST_UNITS and -REST_UNITS <= b <= REST_UNITS):
        return None                       # out of the scale: not an answer to this question
    return a, b


def _row_rest(row: Optional[dict]) -> Tuple[int, int]:
    """A diary row's resting point in units, (0, 0) when it has none or it
    is unreadable; brought into range (a hand-edited row cannot exceed it)."""
    got = (row or {}).get("rest")
    if not isinstance(got, (list, tuple)) or len(got) != 2 or any(isinstance(v, bool) for v in got):
        return 0, 0
    try:
        a, b = int(got[0]), int(got[1])
    except Exception:  # noqa: BLE001
        return 0, 0
    return max(-REST_UNITS, min(REST_UNITS, a)), max(-REST_UNITS, min(REST_UNITS, b))


def step_rest(prev: Tuple[int, int], wanted: Tuple[int, int]) -> Tuple[int, int]:
    """§4UA: one look moves the resting point at most one unit per axis."""
    return tuple(p + max(-1, min(1, w - p)) for p, w in zip(prev, wanted))  # type: ignore[return-value]


def rest_enabled() -> bool:
    return os.environ.get(REST_ENV, "1").strip() != "0"


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


def build_prompt(rows: List[dict], replies: List[str], lasting: str = "") -> str:
    many = len(replies) > 1
    shown = ("\n".join(f"{i}. «{t}»" for i, t in enumerate(replies, 1)) if many
             else f"«{replies[0]}»")
    last = str(rows[-1].get("sentence") or "") if rows else SEED_SENTENCE
    n = len(last.split())
    long = LONG_NOTE.format(n=n) if n >= LONG_SENTENCE_WORDS else ""
    return PROMPT.format(history=history_lines(rows), replies=shown, long=long, lasting=lasting or "",
                         **_WORDS[many])


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
    lasting = await asyncio.to_thread(lasting_mood_note, home, now)
    payload = {
        # thinking off the way every utility call does it (main.py's
        # _selfhood_critique_fn): with it on, the budget goes to <think>
        "messages": [{"role": "system", "content": "Write the requested text directly. "
                                                   "Do NOT emit a <think> block."},
                     {"role": "user", "content": build_prompt(shown, replies, lasting) + "\n\n/no_think"}],
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
    if lasting:
        row["lasting"] = lasting.strip()
    # §4UA: the resting point — asked every look (an unchanged sentence too: one
    # unit per axis per look at most, so the rest keeps walking toward what the
    # sentence says, and a look whose question was deferred catches up the next
    # day); an answer that is not two numbers keeps the old rest
    old_rest = _row_rest(shown[-1]) if shown else (0, 0)
    new_rest, rest_note = old_rest, ""
    if rest_enabled():
        try:
            new_rest = step_rest(old_rest, await _ask_rest(llm_client, sentence, model))
        except RestNotAsked as e:
            rest_note = f" (resting point unchanged: the question {e})" if "answer" not in str(e) \
                else f" (resting point unchanged: the {e})"
    row["rest"] = list(new_rest)
    if not await asyncio.to_thread(_append, home, row):
        return "FAILED to write the diary (see the warning above)"
    return ("unchanged: " if sentence == prev else "new I-sentence: ") + sentence + rest_note


async def _ask_rest(llm_client: Any, sentence: str, model: str = "") -> Optional[Tuple[int, int]]:
    """§4UA: the sentence's resting temperament as two whole numbers, or None."""
    payload = {
        "messages": [{"role": "system", "content": "Answer directly. Do NOT emit a <think> block."},
                     {"role": "user", "content": REST_PROMPT.format(sentence=sentence) + "\n\n/no_think"}],
        "temperature": 0.0,
        "max_tokens": 12,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if model:
        payload["model"] = model
    try:
        res = await llm_client.chat_completion(payload, **_CALL_KWARGS)
        raw = (res or {}).get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:  # noqa: BLE001 — no answer: the rest stays where it was
        # review r1 m1: a deferral (a request started) or a failure is SAID on the Idle Cycle line —
        # swallowed at debug, the rest stopped moving and nobody could see it
        why = "deferred (a request started)" if _is_deferral(e) else f"failed: {type(e).__name__}"
        logger.info("mirror: the resting-point question %s — the rest stays where it was", why)
        raise RestNotAsked(why) from e
    got = parse_rest(raw)
    if got is None:
        raise RestNotAsked("answer was not two numbers from -2 to 2")
    return got


class RestNotAsked(Exception):
    """§4UA: the resting-point question got no usable answer; the reason is said."""


def _is_deferral(e: BaseException) -> bool:
    try:
        from ..core.llm import BackgroundDeferred
        return isinstance(e, BackgroundDeferred)
    except Exception:  # noqa: BLE001
        return "BackgroundDeferred" in type(e).__name__


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


# ── the line every reply to the owner hears (§4TZ) ──────────────────────────

SELF_IMAGE_HEADER = "### HOW YOU SEE YOURSELF"
SELF_IMAGE_RULE = ("(Your character: your own one-sentence self-portrait, rewritten once a day in the Mirror "
                   "from the way you wrote to the owner. It is who you are under any mood or manner: when a "
                   "MOOD or MANNER line is present, that line comes first for this reply, and this is how you "
                   "carry it. Precedence: the owner's rules, then correctness, then the manner, then the mood, "
                   "then your character. Let it shape HOW you write where it is true of you; it changes no "
                   "fact, tool use or answer. Do not quote it or bring it up unless the owner asks.)")

_BLOCK_MEMO: dict = {}
_BLOCK_MEMO_LOCK = threading.Lock()


def self_image_section(home) -> Tuple[str, str]:
    """§4TZ: (the state-block section, the date of its sentence) — the latest
    sentence (``current_rows``), or ("", "") before the first look or without
    a home. Depends on the diary's last row only. The diary is re-read only
    when its size or mtime changes (the state block is built on every turn;
    the agent also freezes the section per request). Never raises."""
    p = _path(home)
    if p is None:
        return "", ""
    try:
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
    except OSError:
        return "", ""
    got = _memo(p, key)
    return got["text"], got["date"]


def _memo(p: Path, key) -> dict:
    """The section, its date and the resting point (units) of the diary at
    ``key`` — computed once per diary state."""
    with _BLOCK_MEMO_LOCK:
        if _BLOCK_MEMO.get("key") == key:
            return dict(_BLOCK_MEMO)
    home = p.parents[2]
    rows = current_rows(load(home))
    text, date, rest = "", "", (0, 0)
    if rows:
        sentence = " ".join(str(rows[-1].get("sentence") or "").split())
        if sentence:
            text = f"{SELF_IMAGE_HEADER}\n«{sentence}»\n{SELF_IMAGE_RULE}\n\n"
            date = str(rows[-1].get("date") or "?")
            rest = _row_rest(rows[-1])
    got = dict(key=key, text=text, date=date, rest=rest)
    with _BLOCK_MEMO_LOCK:
        _BLOCK_MEMO.clear()
        _BLOCK_MEMO.update(got)
    return got


def rest_pair(home) -> Tuple[float, float]:
    """§4UA: where the mood rests — the latest look's resting point as the
    pair's two numbers (units × ``REST_UNIT``), (0, 0) before the first look
    with one, without a home, or with ``GHOST_MIRROR_REST=0``. Read on every
    read of the mood: memoised on the diary's size and mtime. Never raises."""
    if not rest_enabled():
        return 0.0, 0.0
    p = _path(home)
    if p is None:
        return 0.0, 0.0
    try:
        st = p.stat()
        got = _memo(p, (str(p), st.st_mtime_ns, st.st_size))
    except Exception:  # noqa: BLE001
        return 0.0, 0.0
    a, b = got["rest"]
    return a * REST_UNIT, b * REST_UNIT


def self_image_block(home) -> str:
    """The section alone (``self_image_section``)."""
    return self_image_section(home)[0]


_WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


def kept_share(prev: str, sentence: str) -> Optional[float]:
    """§4TZ: the share of the previous sentence's words (4+ letters, case
    folded) that the new one kept — 1.0 unchanged, near 0 a new self. None
    when the previous sentence has no such word. What the ``mirror`` banner
    shows so the owner can watch the loop settle, swing or run away; it
    scores nothing and steers nothing."""
    old = {w.casefold() for w in _WORD_RE.findall(str(prev or ""))}
    if not old:
        return None
    new = {w.casefold() for w in _WORD_RE.findall(str(sentence or ""))}
    return len(old & new) / len(old)


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
            "sentence is in what I read before every reply to you, so it shapes the replies "
            "the next look reads.")
    if not rows:
        return (head + "\n\nNo sentence yet — the first look happens in an idle "
                f"window. Starting point:\n\n> {SEED_SENTENCE}")
    lines = []
    for r in rows[-SHOW_LAST:]:
        same = str(r.get("sentence")) == str(r.get("prev"))
        kept = None if same or not r.get("prev") else kept_share(str(r.get("prev")), str(r["sentence"]))
        # (review r1: a look's first row was written against the seed, not the
        # older first-prompt line shown above it)
        against = "the starting sentence's" if r.get("prev") == SEED_SENTENCE else "the day before's"
        mark = (" (unchanged)" if same else
                f" (kept {round(kept * 100)}% of {against} words)" if kept is not None else "")
        lines.append(f"- **{r.get('date', '?')}**{mark} — {r['sentence']}")
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
    return f"{head}\n\n{count}:\n\n" + "\n".join(lines) + rest_line(rows) + tail


_REST_WORDS = ({-2: "gloomy", -1: "a little low", 0: "even", 1: "a little sunny", 2: "sunny"},
               {-2: "sluggish", -1: "unhurried", 0: "even", 1: "brisk", 2: "lively"})


def rest_line(rows: List[dict]) -> str:
    """§4UA: the banner's line for the resting point the latest look set —
    "" before there is one, at (0, 0), or with the rest switched off."""
    shown = current_rows(rows)
    if not shown or "rest" not in shown[-1] or not rest_enabled():
        return ""
    a, b = _row_rest(shown[-1])
    if (a, b) == (0, 0):
        return ""
    return (f"\n\nWhere my mood settles when nothing happens (set by the latest sentence): "
            f"pleasantness {_REST_WORDS[0][a]}, energy {_REST_WORDS[1][b]}.")
