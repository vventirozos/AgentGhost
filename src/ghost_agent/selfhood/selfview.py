"""The look (§4UH): once a day the agent reads a few of its own replies and says how they sound — on the
same three dials its character is held in. This is what makes the character a loop again.

The Mirror was a loop on paper: replies → a sentence about them → the replies. Measured, the sentence
changed nothing, so the loop carried nothing round. The character (§4UG) is heard, but it was not a loop at
all: its dials moved from the agent's days, and the agent never looked at itself. The look closes it with
both halves real:

    dials → the sentence every reply hears → the replies → THE LOOK (how they sound) → a vote on the dials

* What is asked: three whole numbers from -2 to 2 — warmth, humour, directness — for a handful of replies
  taken together, on scales anchored in the character's own words. The model is NOT shown its dials or its
  sentence (it would read them back), only the replies, with replies written under a strong mood or a
  manner left out (that is the mood talking), as the Mirror's look did.
* What it is for. (1) Self-knowledge: "I mean to be a little dry; lately I have sounded neither dry nor
  warm" is something the agent can now say (``line``, the `character` view). (2) A vote: when the replies
  sit TWO steps from a dial, or on the other side of 0 from it, that day pulls the dial one way
  (``votes``) and is added to the day's own pull in ``character.propose``. One step of difference is no
  vote: a mild trait is faint in most replies, and voting on it would walk every dial back to the model's
  default voice — the fixed point the Mirror sat on.
  §4UH-e2e: the vote is OFF unless ``GHOST_SELFVIEW_VOTE=1`` (``vote_enabled``): a model's rating of its own
  replies, never checked against blind readers, moved dials with no internal signal behind it. The look is
  still taken and shown.

One background call a day on the main model, in the idle window, at the Mirror's old place and price.
Storage: ``$GHOST_HOME/system/selfhood/selfview.jsonl`` — append-only, one row per look:
{"at", "date" (mood day), "observed": {dial: n}, "dials": {dial: n}, "replies": k}.
"""

from __future__ import annotations

import datetime
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import character as C
from .schema import read_jsonl_rows

logger = logging.getLogger("GhostSelfhood")

LOG_FILENAME = "selfview.jsonl"
#: fewer replies than this say too little about a voice
MIN_REPLIES = 3
#: a look counts as evidence for this many days at most (``character.EVIDENCE_DAYS``)
VOTE_GAP = 2
#: a day's look is tried at most this many times when the answer cannot be read (review r1: each try is a
#: side call on the main model — §4RI — and an unreadable answer was asked again every idle hour)
MAX_TRIES_A_DAY = 2
LOG_MAX_ROWS = 800
LOG_KEEP_ROWS = 600

PROMPT = (
    "Below are {n} replies an assistant wrote to the person it works for.\n\n{replies}\n\n"
    "Judge only HOW they are written, not what they are about. Rate the writing, taken together, on three "
    "scales, each a whole number from -2 to 2:\n"
    "warmth: -2 {w0} · -1 {w1} · 0 neither · 1 {w3} · 2 {w4}\n"
    "humour: -2 {h0} · -1 {h1} · 0 neither · 1 {h3} · 2 {h4}\n"
    "directness: -2 {d0} · -1 {d1} · 0 neither · 1 {d3} · 2 {d4}\n"
    "Most writing is near 0 on most scales: give a number away from 0 only for what is plainly there in "
    "several of the replies, and never to be kind to the writer.\n"
    "Reply with the three numbers only, in that order, for example: 0 1 -1"
)
#: §4UH-e2e: the look VOTES in the daily rule only with this switch on. Off by default: the look is the
#: main model rating its own replies — a model's judgement moving a dial with no internal signal behind it
#: (the fuzzy-judge lesson), never checked against blind readers, and its first live look rated the dry dial
#: "warm", which would walk the character back to the model's default voice. It is still taken, kept and
#: shown (`mirror`, the `character` view): self-knowledge, not a lever.
VOTE_ENV = "GHOST_SELFVIEW_VOTE"


def vote_enabled() -> bool:
    import os
    return os.environ.get(VOTE_ENV, "0").strip() == "1"


_ANSWER_RE = re.compile(r"^\D{0,12}?([+-]?[0-2])\s*[,;/\s]\s*([+-]?[0-2])\s*[,;/\s]\s*([+-]?[0-2])\.?\s*$")
_CALL_KWARGS = {"is_background": True, "timeout": 180.0, "task_label": "selfview"}


def _path(home) -> Optional[Path]:
    return Path(home) / C.SELFHOOD_REL / LOG_FILENAME if home else None


def _all_rows(home) -> List[Dict[str, object]]:
    p = _path(home)
    return read_jsonl_rows(p, what="self view") if p else []


def rows(home) -> List[Dict[str, object]]:
    """The looks that saw something (a row for an unreadable answer has no ``observed``)."""
    out = []
    for r in _all_rows(home):
        obs = r.get("observed")
        if C._real_date(r.get("date")) and isinstance(obs, dict):
            out.append(r)
    return out


def build_prompt(replies: List[str]) -> str:
    shown = "\n".join(f"{i}. «{t}»" for i, t in enumerate(replies, 1))
    words = {}
    for key, dial in (("w", "warmth"), ("h", "humour"), ("d", "directness")):
        for i, pos in ((0, -2), (1, -1), (3, 1), (4, 2)):
            words[f"{key}{i}"] = C.PHRASES[dial][pos]
    return PROMPT.format(n=len(replies), replies=shown, **words)


def parse_answer(raw) -> Optional[Dict[str, int]]:
    """{dial: n} from the model's answer, or None when it is not exactly three numbers from -2 to 2."""
    try:
        from ..core.project_research import _strip_think
        text = _strip_think(str(raw or ""))
    except Exception:  # noqa: BLE001
        text = str(raw or "")
    lines = [ln.strip() for ln in text.replace("−", "-").splitlines() if ln.strip()]
    if len(lines) != 1:
        return None
    m = _ANSWER_RE.match(lines[0])
    if not m:
        return None
    return {d: int(v) for d, v in zip(C.DIALS, m.groups())}


def trying_today(home, now: Optional[datetime.datetime] = None) -> bool:
    """Today's look failed to read at least once, has a try left and is not there yet (a second try comes in
    an idle window)."""
    today = C._mood_day(now)
    if any(str(r.get("date")) == today for r in rows(home)):
        return False
    n = sum(1 for r in _all_rows(home) if r.get("failed") and str(r.get("date")) == today)
    return 0 < n < MAX_TRIES_A_DAY


def looked_today(home, now: Optional[datetime.datetime] = None) -> bool:
    """Today's look is there — or it was tried ``MAX_TRIES_A_DAY`` times and its answer could not be read."""
    today = C._mood_day(now)
    got = rows(home)
    if got and str(got[-1].get("date")) == today:
        return True
    return sum(1 for r in _all_rows(home) if r.get("failed") and str(r.get("date")) == today) >= MAX_TRIES_A_DAY


async def look(home, llm_client: Any, *, now: Optional[datetime.datetime] = None,
               rng: Optional[random.Random] = None, model: str = "") -> str:
    """The day's look. The outcome in words for the idle line ("" when today's look is already there).
    A deferred background call (`BackgroundDeferred`) is raised to the caller, which tries again later."""
    import asyncio
    from . import loop as _loop
    if await asyncio.to_thread(looked_today, home, now):
        return ""
    if llm_client is None:
        return ""                                   # nobody to ask (said once an hour it would only be noise)
    got = await asyncio.to_thread(rows, home)
    after = str(got[-1].get("at") or "") if got else ""
    picked = await asyncio.to_thread(_loop.pick_replies, home, now, rng, after)
    if len(picked) < MIN_REPLIES:
        return ""                                   # too few replies since the last look: nothing to say today
    replies = [p["text"][:_loop.REPLY_MAX_CHARS] for p in picked]
    payload = {
        "messages": [{"role": "system", "content": "Answer directly. Do NOT emit a <think> block."},
                     {"role": "user", "content": build_prompt(replies) + "\n\n/no_think"}],
        "temperature": 0.0, "max_tokens": 16, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if model:
        payload["model"] = model
    try:
        res = await llm_client.chat_completion(payload, **_CALL_KWARGS)
    except Exception as e:
        from ..core.llm import BackgroundDeferred
        if not isinstance(e, BackgroundDeferred):
            # review r2: a call that fails is one of the day's tries too (else it is asked every idle hour)
            await asyncio.to_thread(_append, home, {"at": C._iso(now), "date": C._mood_day(now), "failed": True,
                                                    "replies": len(replies), "error": type(e).__name__})
        raise
    try:
        raw = (res or {}).get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:  # noqa: BLE001
        raw = ""
    observed = parse_answer(raw)
    if observed is None:
        await asyncio.to_thread(_append, home, {"at": C._iso(now), "date": C._mood_day(now), "failed": True,
                                                "replies": len(replies)})
        return "no look — the answer was not three numbers from -2 to 2"
    state = await asyncio.to_thread(C.load, home)
    row = {"at": C._iso(now), "date": C._mood_day(now), "observed": observed,
           "dials": {d: C.clamp(state["dials"].get(d, 0)) for d in C.DIALS}, "replies": len(replies)}
    await asyncio.to_thread(_append, home, row)
    return "looked at " + str(len(replies)) + " replies: " + describe(row)


def _append(home, row: Dict[str, object]) -> None:
    from ..utils.json_store import open_append
    p = _path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open_append(p) as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")
    if p.stat().st_size > LOG_MAX_ROWS * 160:
        kept = read_jsonl_rows(p, what="self view")
        if len(kept) > LOG_MAX_ROWS:
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept[-LOG_KEEP_ROWS:]),
                           encoding="utf-8")
            tmp.replace(p)


def vote(observed: int, dial: int) -> int:
    """Which way a look pulls one dial: toward what the replies showed, when they sit ``VOTE_GAP`` steps
    from the dial or on the other side of 0 — else no vote."""
    o, d = C.clamp(observed), C.clamp(dial)
    if o == d:
        return 0
    if abs(o - d) >= VOTE_GAP or (o * d < 0):
        return 1 if o > d else -1
    return 0


def votes(home) -> Dict[str, Dict[str, int]]:
    """{date: {dial: ±1}} for ``character.propose`` — each look against the dials AS THEY WERE that day
    (the last look of a day wins). {} unless ``vote_enabled()`` (§4UH-e2e)."""
    out: Dict[str, Dict[str, int]] = {}
    if not vote_enabled():
        return out
    for r in rows(home):
        obs, then = r.get("observed") or {}, r.get("dials") if isinstance(r.get("dials"), dict) else {}
        day = {d: vote(obs.get(d, 0), then.get(d, 0)) for d in C.DIALS}
        day = {d: v for d, v in day.items() if v}
        if day:
            out[str(r.get("date"))] = day
        else:
            out.pop(str(r.get("date")), None)
    return out


def _words(dial: str, n: int) -> str:
    n = C.clamp(n)
    return C.PHRASES[dial][n] if n else f"neither {C.POLES[dial][0]} nor {C.POLES[dial][1]}"


def describe(row: Dict[str, object]) -> str:
    """"warmth 0 (dial -1), humour +1 (dial +1), directness 0 (dial 0)"."""
    obs = row.get("observed") or {}
    then = row.get("dials") if isinstance(row.get("dials"), dict) else {}
    return ", ".join(f"{d} {C.clamp(obs.get(d, 0)):+d} (dial {C.clamp(then.get(d, 0)):+d})" for d in C.DIALS)


#: the ``mirror`` command shows this many of the latest looks
SHOW_LAST = 7
_ARROWS = {1: "up", -1: "down"}


def _signed(n) -> str:
    v = C.clamp(n)
    return f"{v:+d}" if v else "0"


def _pull_words(row: Dict[str, object], pins=(), since=None) -> str:
    """"votes humour down, directness up" / "no vote" for one look against the dials it was taken under. A
    vote ``character.propose`` will never count is marked: a dial pinned now, or a look dated on or before
    the dial's last change (``since``: {dial: date} — a move, a pin change or the owner's word uses it up)."""
    obs = row.get("observed") or {}
    then = row.get("dials") if isinstance(row.get("dials"), dict) else {}
    day, since = str(row.get("date") or ""), since or {}

    def mark(d: str) -> str:
        if d in pins:
            return " (pinned: not counted)"
        if since.get(d) and day <= since[d]:
            return " (the dial changed since: not counted)"
        return ""
    if not vote_enabled():
        # §4UH-e2e: the look does not vote — say which way it leans, and that it is not counted
        leans = [f"{d} {_ARROWS[v]}" for d in C.DIALS for v in (vote(obs.get(d, 0), then.get(d, 0)),) if v]
        return ("leans " + ", ".join(leans) + " (not a vote)") if leans else "no lean"
    pulls = [f"{d} {_ARROWS[v]}{mark(d)}" for d in C.DIALS for v in (vote(obs.get(d, 0), then.get(d, 0)),) if v]
    return ("votes " + ", ".join(pulls)) if pulls else "no vote"


def history_lines(home, limit: int = SHOW_LAST, *, now: Optional[datetime.datetime] = None) -> List[str]:
    """§4UH: the latest looks for the owner's ``mirror`` command, oldest first — one Markdown list item each:
    date, how many replies, meant (the dial that day) vs sounded (what the look saw) per dial, and which way
    the look votes. A day's tries that never read are shown only when that day has no look (today's only
    once its tries are used up: a second try may still come). [] before any
    row. The surface is owner-only (the command is answered on the owner's private surface); it holds no
    tastes and no reply text."""
    all_rows = [r for r in _all_rows(home) if C._real_date(r.get("date"))]
    looked = {str(r.get("date")) for r in all_rows if isinstance(r.get("observed"), dict)}
    today = C._mood_day(now)
    if sum(1 for r in all_rows if r.get("failed") and str(r.get("date")) == today) < MAX_TRIES_A_DAY:
        looked.add(today)                    # still trying today: not "no look" yet
    try:
        state = C.load(home)
        pins = tuple(state.get("pins") or ())
        log = C.log_rows(home)
        # the same "since" `character.propose` uses: the state's stamp first, the log for an older file
        since = {d: C._real_date((state.get("changed") or {}).get(d)) or C._last_change(log, d) for d in C.DIALS}
    except Exception:  # noqa: BLE001
        pins, since = (), {}
    shown, failed_days = [], set()
    for r in all_rows:
        day = str(r.get("date"))
        if isinstance(r.get("observed"), dict):
            shown.append(r)
        elif r.get("failed") and day not in looked and day not in failed_days:
            failed_days.add(day)
            shown.append(r)
    out = []
    for r in shown[-limit:] if limit > 0 else []:
        n = r.get("replies")
        count = f" · {n} replies" if isinstance(n, int) and not isinstance(n, bool) else ""
        if not isinstance(r.get("observed"), dict):
            why = "the call failed" if r.get("error") else "the answer could not be read"
            out.append(f"- **{r.get('date')}**{count} — no look: {why}")
            continue
        obs = r.get("observed") or {}
        then = r.get("dials") if isinstance(r.get("dials"), dict) else {}
        dials = " · ".join(f"{d} meant {_signed(then.get(d, 0))}, sounded {_signed(obs.get(d, 0))}"
                           for d in C.DIALS)
        out.append(f"- **{r.get('date')}**{count} — {dials} → {_pull_words(r, pins, since)}")
    return out


def line(home) -> str:
    """What the last look saw, in words the agent can say about itself — "" before the first look."""
    got = rows(home)
    if not got:
        return ""
    last = got[-1]
    n = last.get("replies")
    n = n if isinstance(n, int) and not isinstance(n, bool) else "some"
    obs = last.get("observed") or {}
    then = last.get("dials") if isinstance(last.get("dials"), dict) else {}
    same, off = [], []
    for d in C.DIALS:
        o, t = C.clamp(obs.get(d, 0)), C.clamp(then.get(d, 0))
        (same if o == t else off).append((d, o, t))
    parts = [f"I meant to be {_words(d, t)} and sounded {_words(d, o)}" for d, o, t in off]
    if not parts:
        return (f"My own look at my replies on {last.get('date')}: they sounded the way my dials say "
                f"({n} replies).")
    return (f"My own look at my replies on {last.get('date')} ({n} replies): "
            + "; ".join(parts) + (". The rest matched." if same else "."))
