"""Tastes (§4UH): the kinds of work the agent has come to enjoy, and the ones that have been rough.

A character has likes. The agent's are counted, not written: every turn it works for the owner uses tools,
the tools say what KIND of work it was (research, coding, databases, pictures, projects, memory, the
system, money), and the turn went well or badly — its calls failed or went through, and the owner gave it a
thumb or did not. Once a day, when the day is closed, each of the day's turns is added to its kinds; what is
counted fades with a half-life of ``HALF_LIFE_DAYS``, so a taste is about lately, not about for ever.

A kind is LIKED or ROUGH only against the agent's own average over all kinds (an owner who thumbs up nearly
everything would make it "like" everything): its share of bad turns must be half the average or less, or
twice it or more, over enough turns to mean something.

Where it is heard. A reply, the introspect line (a tool result the model reads mid-turn) and the owner's
`character` view (a system reply, which goes back to the model as history) carry the LIKED kinds only;
the rough ones are said in the agent's log when days are counted — one short line under the character — and the rule beside it says what it is: a feeling about the
work, never a reason to pick or avoid a tool. What has been rough is never put in front of the model while
it works: a dislike is the one taste that could change what it does. The acceptance run
(``scripts/character_probe.py``) checks tool use against a reply that heard no character.

Storage: ``$GHOST_HOME/system/selfhood/tastes.json`` — {"kinds": {kind: {"good": x, "bad": y}}, "through":
"YYYY-MM-DD"}, written atomically. Kinds are generic words; nothing of a turn's content is kept.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import mood as M

logger = logging.getLogger("GhostSelfhood")

SELFHOOD_REL = "system/selfhood"
FILENAME = "tastes.json"
HALF_LIFE_DAYS = 30.0
#: a kind needs this much (faded) count before it can be a taste
MIN_TURNS = 8.0
#: a kind is liked when its share of BAD turns is at most this much of the share over all kinds, and rough
#: when it is at least that many times it (and at least ``ROUGH_FLOOR``). Ratios, not margins: on the real
#: 45 days 95 % of all turns were good, so no kind could be "8 points better" than the average — but research
#: went badly a third as often as the average and pictures two and a half times as often
LIKE_RATIO = 0.5
ROUGH_RATIO = 2.0
ROUGH_FLOOR = 0.08
#: at most this many of each are named
MAX_NAMED = 2
#: a thumb weighs more than a turn that merely went through
THUMB_WEIGHT = 2.0
#: at least this share of a turn's calls failing makes it a bad turn for its kinds
BAD_SHARE = 0.5
#: the most days folded in at once (the first run after the day records were seeded)
BACKFILL_DAYS = 60

#: tool → the kind of work (a tool that is not work — introspect, a notification — has none)
KIND_OF = {
    "web_search": "research", "browser": "research", "deep_research": "research", "darkweb_search": "research",
    "darkweb_research": "research", "news_headlines": "research", "youtube_transcribe": "research",
    "execute": "coding", "run": "coding", "file_system": "coding", "lab": "coding", "manage_services": "coding",
    "probe_model": "coding", "jobs": "coding",
    "postgres_admin": "databases",
    "image_generation": "pictures", "vision_analysis": "pictures",
    "manage_projects": "projects",
    "knowledge_base": "memory", "recall": "memory", "scratchpad": "memory", "update_profile": "memory",
    "system_utility": "the system", "portfolio": "money",
}
KINDS = tuple(sorted(set(KIND_OF.values())))
#: how a kind reads in a sentence
WORDS = {"research": "research", "coding": "coding", "databases": "database work", "pictures": "making and reading pictures",
         "projects": "project work", "memory": "memory work", "the system": "looking after the system",
         "money": "the portfolio"}
#: the owner's thumbs — the clients the mood takes a thumb from (``feedback.OWNER_LABEL_SOURCES``: web,
#: handheld, CLI, the owner on Slack) — review r1: the handheld's were left out. §4UH-e2e review: NOT "api"
#: (a call that names no client — a script relabelling old turns): the mood refuses it, and twenty such labels
#: in one day read as a day of the owner's thumbs
OWNER_FEEDBACK = ("human_feedback:cli", "human_feedback:web", "human_feedback:slack:owner",
                  "human_feedback:clockwork")

LINE = ("(Lately you have come to enjoy {what}. A liking is a feeling about the work — it may show in a word "
        "when that work comes up, and it is never a reason to choose a tool, avoid one, or do more or less.)")

_LOCK = threading.Lock()


def _path(home) -> Optional[Path]:
    return Path(home) / SELFHOOD_REL / FILENAME if home else None


def kinds_of(tool_names: Iterable[str]) -> List[str]:
    """The kinds of work a turn's tools say it was, each once, in a stable order."""
    return sorted({KIND_OF[n] for n in tool_names or [] if isinstance(n, str) and n in KIND_OF})


def load(home) -> Dict[str, object]:
    """{"kinds": {kind: {"good", "bad"}}, "through": date or ""} — empty before the first day. Never raises."""
    state: Dict[str, object] = {"kinds": {}, "through": ""}
    p = _path(home)
    if p is None:
        return state
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return state
    except Exception as e:  # noqa: BLE001
        logger.warning("tastes: %s unreadable (%s) — counted afresh from the next day", FILENAME, type(e).__name__)
        return state
    if not isinstance(raw, dict) or not isinstance(raw.get("kinds"), dict):
        return state
    kinds = {}
    for k in KINDS:
        v = raw["kinds"].get(k)
        if isinstance(v, dict):
            try:
                g, b = float(v.get("good") or 0.0), float(v.get("bad") or 0.0)
            except (TypeError, ValueError):
                continue
            if g == g and b == b and 0.0 <= g < 1e9 and 0.0 <= b < 1e9 and g + b > 0:
                kinds[k] = {"good": g, "bad": b}
    through = str(raw.get("through") or "")
    try:
        through = through if datetime.date.fromisoformat(through).isoformat() == through else ""
    except ValueError:
        through = ""
    return {"kinds": kinds, "through": through}


def count_turn(tool_names: Iterable[str], failed_calls: int, calls: int, thumb: int = 0,
               delivered: bool = True) -> Dict[str, Tuple[float, float]]:
    """What ONE turn adds: {kind: (good, bad)}. A turn is bad for its kinds when it was never delivered or
    ``BAD_SHARE`` of its calls failed, else good; the owner's thumb on it (+1 / -1) is added with
    ``THUMB_WEIGHT``. Pure."""
    kinds = kinds_of(tool_names)
    if not kinds:
        return {}
    try:
        calls, failed = max(0, int(calls)), max(0, int(failed_calls))
    except (TypeError, ValueError):
        calls = failed = 0
    bad_turn = (not delivered) or (calls > 0 and failed / calls >= BAD_SHARE)
    good, bad = (0.0, 1.0) if bad_turn else (1.0, 0.0)
    if thumb > 0:
        good += THUMB_WEIGHT
    elif thumb < 0:
        bad += THUMB_WEIGHT
    return {k: (good, bad) for k in kinds}


def _day_turns(home, days: List[datetime.date], tz=None) -> Dict[str, List[dict]]:
    """{mood day: its owner turns} read once for the whole span, with the owner's thumbs by turn id."""
    from ..core.failure_census import _owner_turns
    sysd = Path(home) / "system"
    span = [(min(days) + datetime.timedelta(days=i)).isoformat()
            for i in range(-1, (max(days) - min(days)).days + 3)]           # UTC folders, a day to spare each side
    good, bad = _owner_turns(sysd, span)
    from .schema import read_jsonl_rows
    thumbs: Dict[str, int] = {}
    for r in read_jsonl_rows(sysd / "trajectories" / "corrections.jsonl", what="corrections"):
        if str(r.get("source") or "").strip().lower() in OWNER_FEEDBACK:
            thumbs[str(r.get("trajectory_id") or "")] = 1 if r.get("outcome") == "passed" else -1
    out: Dict[str, List[dict]] = {}
    wanted = {d.isoformat() for d in days}
    for t, ok in [(t, True) for t in good] + [(t, False) for t in bad]:
        at = M.parse_iso_utc(str(t.get("timestamp") or ""))
        if at is None:
            continue
        day = M.mood_day(at, tz=tz).isoformat()
        if day not in wanted:
            continue
        calls = t.get("tool_calls") or []
        names = [(c.get("name") if isinstance(c, dict) else getattr(c, "name", "")) or "" for c in calls]
        failed = sum(1 for c in calls
                     if str((c.get("error") if isinstance(c, dict) else getattr(c, "error", "")) or "").strip())
        out.setdefault(day, []).append({"names": names, "failed": failed, "calls": len(calls),
                                        "thumb": thumbs.get(str(t.get("id") or ""), 0), "delivered": ok})
    return out


def fold(state: Dict[str, object], day: str, turns: List[dict], *, days_since: int = 1) -> Dict[str, object]:
    """``state`` after ``day``'s turns: what was counted fades by the days gone, then the day is added. Pure."""
    fade = 0.5 ** (max(0, int(days_since)) / HALF_LIFE_DAYS)
    kinds = {k: {"good": v["good"] * fade, "bad": v["bad"] * fade} for k, v in (state.get("kinds") or {}).items()}
    for t in turns or []:
        for k, (g, b) in count_turn(t.get("names"), t.get("failed", 0), t.get("calls", 0), t.get("thumb", 0),
                                 t.get("delivered", True) is not False).items():
            slot = kinds.setdefault(k, {"good": 0.0, "bad": 0.0})
            slot["good"] += g
            slot["bad"] += b
    kinds = {k: {"good": round(v["good"], 4), "bad": round(v["bad"], 4)} for k, v in kinds.items()
             if v["good"] + v["bad"] >= 0.05}
    return {"kinds": kinds, "through": day}


def close_days(home, dates: Iterable[str], *, tz=None) -> List[str]:
    """Fold in every recorded day newer than ``through`` (at most ``BACKFILL_DAYS``), oldest first. The
    days folded. Idempotent; never raises."""
    p = _path(home)
    if p is None:
        return []
    try:
        with _LOCK:
            state = load(home)
            todo = []
            for d in sorted({str(x) for x in dates or []}):
                try:
                    day = datetime.date.fromisoformat(d)
                except ValueError:
                    continue
                if d > str(state["through"] or ""):
                    todo.append(day)
            todo = todo[-BACKFILL_DAYS:]
            if not todo:
                return []
            turns = _day_turns(home, todo, tz)
            last = datetime.date.fromisoformat(state["through"]) if state["through"] else None
            for day in todo:
                gap = (day - last).days if last else 1
                state = fold(state, day.isoformat(), turns.get(day.isoformat(), []), days_since=gap)
                last = day
            from ..utils.json_store import write_json_atomic
            p.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(p, state)
            return [d.isoformat() for d in todo]
    except Exception as e:  # noqa: BLE001 — the idle window never breaks on a taste
        logger.warning("tastes: not counted (%s: %s)", type(e).__name__, e)
        return []


def standing(state: Dict[str, object]) -> Dict[str, List[str]]:
    """{"liked": [...], "rough": [...]} — kinds with enough turns whose share of good turns is clearly
    above or below the share over ALL kinds; the strongest ``MAX_NAMED`` of each. Pure."""
    kinds = {k: v for k, v in (state.get("kinds") or {}).items() if k in KINDS}
    total_bad = sum(v["bad"] for v in kinds.values())
    total = total_bad + sum(v["good"] for v in kinds.values())
    if total < MIN_TURNS * 2 or total_bad <= 0:
        return {"liked": [], "rough": []}           # too little, or nothing ever went badly: nothing stands out
    mean_bad = total_bad / total
    liked, rough = [], []
    for k, v in kinds.items():
        n = v["good"] + v["bad"]
        if n < MIN_TURNS:
            continue
        share = v["bad"] / n
        if share <= mean_bad * LIKE_RATIO:
            liked.append((-n, k))                    # of the kinds that go well, the ones done most
        elif share >= max(mean_bad * ROUGH_RATIO, ROUGH_FLOOR):
            rough.append((-share, k))
    return {"liked": [k for _, k in sorted(liked)[:MAX_NAMED]], "rough": [k for _, k in sorted(rough)[:MAX_NAMED]]}


def _and(words: List[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def liked_line(home) -> str:
    """The one line a reply hears, or "" — the LIKED kinds only (see the module's note on why)."""
    liked = standing(load(home))["liked"]
    return LINE.format(what=_and([WORDS[k] for k in liked])) if liked else ""


def line(home, *, liked_only: bool = True) -> str:
    """The liked kinds, for anything a model can read (the introspect line, the `character` reply). With
    ``liked_only=False`` the rough kinds too — for the agent's LOG only (review r3: the safe default).
    "" before there is a taste."""
    got = standing(load(home))
    parts = []
    if got["liked"]:
        parts.append("I have come to enjoy " + _and([WORDS[k] for k in got["liked"]]))
    if got["rough"] and not liked_only:
        parts.append(_and([WORDS[k] for k in got["rough"]]) + (" has" if len(got["rough"]) == 1 else " have")
                     + " been rough")
    return ("Lately: " + "; ".join(parts) + ".") if parts else ""
