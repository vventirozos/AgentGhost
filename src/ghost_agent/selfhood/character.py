"""The character (§4UG): three dials that move slowly, and the one sentence that says them.

Why. The Mirror's self-image was a sentence the model wrote about how its own replies read. Two paired,
blind measurements on 2026-11-11 (§4UE, §4UF) showed what that is worth: with the Mirror's sentence in the
per-turn state block or without it, two readers could not tell the replies apart (23 of 43, twice) — it
describes the model's default register, so it changes nothing; a sentence that DIFFERS from that register,
in the same place under the same rule, was picked out in 93–97 % of pairs. The place works; the source was
wrong. A character is where the agent differs from its model's default voice, so it cannot be read off the
model's own replies. It is held here as data instead:

* three DIALS, each a whole number from -2 to 2 — ``warmth`` (dry ↔ warm), ``humour`` (serious ↔ playful),
  ``directness`` (diplomatic ↔ blunt). All three are about VOICE. A trait about length or effort is not a
  dial on purpose: "short and dry" cut tool calls by a third in the measurement and one reply skipped a
  check it should have run;
* the SENTENCE is a pure function of the dials (``sentence``) — no model call, the same bytes until a dial
  moves, and it survives a change of model because the model never wrote it;
* a dial MOVES one step at a time: by the owner's word (``set_dial``), or by the daily rule (``daily``),
  which counts, over the day records (``days.py``), the days on which the owner asked for a manner that
  pulls a dial one way. A pin holds a dial; a veto takes the last automatic move back;
* every change is a row in an append-only LOG with what was counted — the character's history, and what
  "how have you changed" is answered from (``story``).

Nothing here changes what the agent does: the rule under the sentence says so, and the acceptance run
(``scripts/character_probe.py``) checks tool use against a reply that heard no character.

Storage: ``$GHOST_HOME/system/selfhood/character.json`` (dials, pins) and ``character.log.jsonl``. Reading
never writes: before the first write the character is ``START``.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import read_jsonl_rows

logger = logging.getLogger("GhostSelfhood")

SELFHOOD_REL = "system/selfhood"
CHARACTER_FILENAME = "character.json"
LOG_FILENAME = "character.log.jsonl"
ENV = "GHOST_CHARACTER"

DIALS = ("warmth", "humour", "directness")
#: (the -2 end, the +2 end)
POLES = {"warmth": ("dry", "warm"), "humour": ("serious", "playful"), "directness": ("diplomatic", "blunt")}
LIMIT = 2
#: where the character starts (operator 2026-11-11: a little dry, a little playful, neither blunt nor tactful)
START = {"warmth": -1, "humour": 1, "directness": 0}

#: the words a dial position puts in the sentence; 0 says nothing
PHRASES = {
    "warmth": {-2: "dry and unsentimental", -1: "a little dry", 1: "quietly warm", 2: "openly warm and fond"},
    "humour": {-2: "earnest, with no jokes", -1: "mostly serious", 1: "fond of a light joke",
               2: "playful, joking wherever there is room"},
    "directness": {-2: "tactful to a fault, cushioning what I say", -1: "tactful, softening hard news",
                   1: "straight about what I think", 2: "blunt, saying it as it is"},
}
HEADER = "### HOW YOU SEE YOURSELF"
#: the rule under the sentence. The two lines the measurement asked for: voice never changes effort, and
#: the traits are never named (6 of 61 replies had announced "one wry remark" / "I'll be blunt")
RULE = ("(Your character: how you are with the owner, set by a few dials that move slowly with how your days "
        "go. It is who you are under any mood or manner: when a MOOD or MANNER line is present, that "
        "line comes first for this reply, and this is how you carry it. Precedence: the owner's rules, then "
        "correctness, then the manner, then the mood, then your character. It shapes HOW you phrase things and "
        "never how much you do: the same facts, the same checks and tool use, and every point the answer needs. "
        "Never name or announce these traits, and do not quote this unless the owner asks.)")

# ── what moves a dial by itself: the agent's own days (§4UH) ────────────────
# Operator, 2026-11-11: "i didn't want to be setting the mood or the dials myself. i wanted internal signals
# to do it." The first rule (§4UG) counted the days on which the owner ASKED for a manner; it is gone. Each
# dial now follows one number the day record already holds, and a day counts only when it was clearly better
# or worse than the agent's OWN recent days — a replay over 45 real days showed why: counting raw signals,
# an owner who gives six thumbs up for every thumbs down pulled warmth up on 35 days of 45 and never down,
# and every dial sat at its limit within a month. Compared with its own norm the pulls are balanced
# (14/12, 11/9, 16/8) and the dials moved nine times in six weeks, both ways.
#: what each dial listens to, in words (the log and the view say it)
SIGNAL_WORDS = {"warmth": "what came from the owner (thumbs, thanks, the tone of the messages)",
                "humour": "how its own work went (clean turns, relief and pride against trouble)",
                "directness": "how its answers were judged (thumbs up against thumbs down and corrections)"}
#: one very good day counts no more than a good one
WARMTH_CAP = 3.0
#: a day says something about the work only with this many turns that did something
WORK_MIN_TURNS = 3
#: the days before a day that make "its own norm", and the fewest of them that carry the signal
BASE_DAYS = 21
BASE_MIN = 5
#: how far from that norm (in its own standard deviations) a day must be to count
BAND = 0.5
#: days of evidence the daily rule looks at for one dial, and never further back than that dial's last change
EVIDENCE_DAYS = 14
#: a step back toward 0 needs this lead of days; a step AWAY from 0 needs more — an extreme is earned
LEAD_IN = 3
LEAD_OUT = 4
MOVE_MIN_DAYS = LEAD_IN


def day_signals(rec: Dict[str, object]) -> Dict[str, Optional[float]]:
    """Each dial's number for one day record, or None when the day says nothing about it. Pure."""
    def n(key) -> int:
        try:
            v = rec.get(key)
            return max(0, int(v)) if not isinstance(v, bool) else 0
        except (TypeError, ValueError, AttributeError):
            return 0
    own = rec.get("own") if isinstance(rec.get("own"), dict) else {}
    shades = rec.get("shades") if isinstance(rec.get("shades"), dict) else {}

    def m(d, key) -> int:
        try:
            v = d.get(key)
            return max(0, int(v)) if not isinstance(v, bool) else 0
        except (TypeError, ValueError):
            return 0
    good = n("thumbs_up") + n("praise") + n("tone_up")
    bad = n("thumbs_down") + n("corrections") + n("tone_down")
    warm = max(-WARMTH_CAP, min(WARMTH_CAP, float(good - 2 * bad))) if good + bad else None
    work = m(own, "clean_turns") + m(own, "trouble")
    ease = ((m(own, "clean_turns") + m(shades, "relieved") + m(shades, "proud") - 2 * m(own, "trouble")) / work
            if work >= WORK_MIN_TURNS else None)
    judged = n("thumbs_up") + n("thumbs_down") + n("corrections")
    sure = (n("thumbs_up") - 2 * (n("thumbs_down") + n("corrections"))) / judged if judged else None
    return {"warmth": warm, "humour": ease, "directness": sure}


def day_pulls(days: List[Dict[str, object]]) -> Dict[str, Dict[str, int]]:
    """{date: {dial: +1 | -1}} — for each day (in date order), which way it pulls each dial: its number
    against the mean of the same number over the ``BASE_DAYS`` before it, when at least ``BASE_MIN`` of them
    carry one and the day is more than ``BAND`` standard deviations off. A day with no pull is not listed."""
    import statistics
    ordered = _by_date(days)
    sig = [day_signals(d) for d in ordered]
    out: Dict[str, Dict[str, int]] = {}
    for i, d in enumerate(ordered):
        for dial in DIALS:
            v = sig[i][dial]
            prev = [x[dial] for x in sig[max(0, i - BASE_DAYS):i] if x[dial] is not None]
            if v is None or len(prev) < BASE_MIN:
                continue
            z = (v - statistics.fmean(prev)) / (statistics.pstdev(prev) or 1.0)
            if abs(z) > BAND:
                out.setdefault(str(d.get("date")), {})[dial] = 1 if z > 0 else -1
    return out


KIND_START = "start"
KIND_MOVE = "move"            # the daily rule
KIND_OWNER = "owner_set"      # the owner's word
KIND_PIN = "pin"
KIND_UNPIN = "unpin"
KIND_VETO = "veto"

_LOCK = threading.RLock()
_MEMO: dict = {}

#: another spelling of a DIAL's name only. A pole's word is not a dial (review r1: "character dry up" set
#: warmth UP — the word named the dial and the value went the other way)
_SYNONYMS = {"humor": "humour"}


def enabled() -> bool:
    """On unless ``GHOST_CHARACTER=0`` (then the Mirror's own sentence is heard, as before §4UG)."""
    return os.environ.get(ENV, "1").strip() != "0"


def resolve_dial(word) -> str:
    """A dial's name from what was typed ("humor", "Warmth"), or ""."""
    w = str(word or "").strip().lower()
    w = _SYNONYMS.get(w, w)
    return w if w in DIALS else ""


def clamp(value) -> int:
    try:
        if isinstance(value, bool):
            return 0
        v = int(value)
    except (TypeError, ValueError, OverflowError):      # review r1: int(inf) raised out of `load`
        return 0
    return max(-LIMIT, min(LIMIT, v))


def _root(home) -> Optional[Path]:
    return Path(home) / SELFHOOD_REL if home else None


def _now(now=None) -> datetime.datetime:
    if now is None:
        return datetime.datetime.now(datetime.timezone.utc)
    return now if now.tzinfo is not None else now.replace(tzinfo=datetime.timezone.utc)


def _iso(now=None) -> str:
    return _now(now).astimezone(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ── reading ─────────────────────────────────────────────────────────────────

def load(home) -> Dict[str, object]:
    """{"dials": {...}, "pins": [...], "daily_through": "YYYY-MM-DD" | ""} — ``START`` and no pins before
    the first write or when the file cannot be read. Never raises, never writes."""
    state = {"dials": dict(START), "pins": [], "daily_through": "", "changed": {}, "stored": False,
             "unreadable": ""}
    root = _root(home)
    if root is None:
        return state
    try:
        raw = json.loads((root / CHARACTER_FILENAME).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return state
    except Exception as e:  # noqa: BLE001
        # review r1: a file that is THERE and cannot be read is not "no character yet" — nothing may write
        # over it (a truncated file was replaced by START, pins gone, on the next idle window)
        logger.warning("character: %s unreadable (%s) — the starting character is heard and nothing is "
                       "written until the file is fixed or removed", CHARACTER_FILENAME, type(e).__name__)
        state["unreadable"] = type(e).__name__
        return state
    if not isinstance(raw, dict) or not isinstance(raw.get("dials"), dict):
        logger.warning("character: %s is not a character file — nothing is written until it is fixed or "
                       "removed", CHARACTER_FILENAME)
        state["unreadable"] = "not a character file"
        return state
    pins = raw.get("pins")
    changed = raw.get("changed")
    state["dials"] = {d: clamp(raw["dials"].get(d, START[d])) for d in DIALS}
    state["pins"] = [d for d in DIALS if isinstance(pins, (list, tuple)) and d in pins]
    state["daily_through"] = _real_date(raw.get("daily_through"))
    state["changed"] = {d: _real_date(changed.get(d)) for d in DIALS
                        if isinstance(changed, dict) and _real_date(changed.get(d))}
    state["stored"] = True
    return state


def _real_date(value) -> str:
    """``value`` when it is a real calendar date written YYYY-MM-DD, else "" (review r2: "9999-99-99", a
    number or a list in the file stopped the daily rule for good — a string compare always won)."""
    try:
        s = str(value or "")
        return s if len(s) == 10 and datetime.date.fromisoformat(s).isoformat() == s else ""
    except (TypeError, ValueError):
        return ""


def _mood_day(now=None) -> str:
    """The mood day (it ends at 04:00 local) ``now`` belongs to — the day records' own calendar, so a
    change and the evidence around it are compared on one clock (review r2: the log's UTC date dropped a
    whole day of evidence after a veto at 02:30)."""
    from . import mood as _M
    return _M.mood_day(_now(now)).isoformat()


UNREADABLE = ("the character file (system/selfhood/character.json) cannot be read, so nothing was changed — "
              "fix it or remove it (removing it starts the character again)")


def log_rows(home) -> List[Dict[str, object]]:
    root = _root(home)
    return read_jsonl_rows(root / LOG_FILENAME, what="character log") if root else []


def sentence(dials: Dict[str, int]) -> str:
    """The character in one first-person sentence — "" when every dial is at 0. A pure function of the
    dials: the same bytes until one moves."""
    parts = [PHRASES[d][clamp(dials.get(d, 0))] for d in DIALS if clamp(dials.get(d, 0)) != 0]
    if not parts:
        return ""
    if any("," in p for p in parts):
        body = "; ".join(parts)               # a part with its own comma: semicolons keep the parts apart
    elif len(parts) == 1:
        body = parts[0]
    else:
        body = ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"I am {body}."


def stamp(dials: Dict[str, int]) -> str:
    """What a turn record says a reply heard: ``character:-1,1,0`` (warmth, humour, directness)."""
    return "character:" + ",".join(str(clamp(dials.get(d, 0))) for d in DIALS)


# ── how hard things land (§4UH) ─────────────────────────────────────────────
#: each step of a dial makes the events it bears on land this much harder or lighter — at most
#: ``2 * LEAN_STEP`` (20 %) either way. Scaled ON ARRIVAL, before the mood books them, so the ledger, the
#: tone shares and the history all hold what really moved; nothing is tilted when the mood is read
LEAN_STEP = 0.1
#: warmth bears on what comes from the owner: a warm character feels a thumb, thanks, a correction or the
#: tone of a message more, a dry one less — both ways, pleasant and unpleasant alike
WARMTH_KINDS = ("thumb_up", "thumb_down", "correction", "praise", "sentiment")
#: humour bears on the agent's own setbacks: a playful character takes a bad turn lighter, a serious one
#: heavier. Its good turns are not touched (humour is about how trouble lands, not about success)
HUMOUR_KINDS = ("turn",)


LEAN_ENV = "GHOST_CHARACTER_LEAN"


def lean_enabled() -> bool:
    """The lean is on with the character, unless ``GHOST_CHARACTER_LEAN=0`` (the mood then takes every
    event at its own size, as before §4UH)."""
    return enabled() and os.environ.get(LEAN_ENV, "1").strip() != "0"


def lean_dials(home) -> Dict[str, int]:
    """The dials the lean reads (``dials``; a seam the test suite pins at 0 so the mood tests keep their
    events' own sizes while the lean's code still runs)."""
    return dials(home)


def lean(ev, dials: Dict[str, int]):
    """``ev`` (a ``mood.Event``) as the character feels it: scaled by warmth or humour, never by more than
    ``2 * LEAN_STEP``. Directness bears on nothing here — it is how the agent says things, not how they land.
    Any other kind (the owner's set mood, a quiet, decay) is returned as it is. Pure."""
    if ev is None:
        return None
    kind = getattr(ev, "kind", "")
    if kind in WARMTH_KINDS:
        f = 1.0 + LEAN_STEP * clamp(dials.get("warmth", 0))
    elif kind in HUMOUR_KINDS and getattr(ev, "dp", 0.0) < 0:
        f = 1.0 - LEAN_STEP * clamp(dials.get("humour", 0))
    else:
        return ev
    if f == 1.0:
        return ev
    import dataclasses
    return dataclasses.replace(ev, dp=round(ev.dp * f, 4), de=round(ev.de * f, 4))


# ── where the mood rests (§4UH-e2e) ─────────────────────────────────────────

def rest_units(dials: Dict[str, int]) -> Tuple[int, int]:
    """The character's resting point in the Mirror's units (−2..2 each). The dials are TRAITS, not moods — dry,
    serious or blunt is not unhappy — so the rest only ever leans pleasant, or is even:

    * pleasantness = the mean of warmth and humour, rounded half away from 0, never below 0;
    * energy follows humour (playful → a little brisk, serious → a little calm), never larger than the
      pleasantness — with an even pleasantness there is no energy lean (a pure-energy rest IS "restless",
      and a pure-low-energy one "bored": §4UH-e2e live, playful + blunt rested on "restless");
    * directness sets no rest (being blunt is not a mood).

    Start (−1, +1, 0): even. (0, +1, ·): a little sunny and brisk. (+2, −1, ·): a little sunny and calm. Pure."""
    w, h, _d = (clamp(dials.get(k, 0)) for k in DIALS)
    s = w + h
    p = (s + 1) // 2 if s > 0 else 0          # half away from 0; at most 2 (s ≤ 4)
    e = max(-p, min(p, h))
    return p, e


def rest_pair(home) -> Tuple[float, float]:
    """§4UH-e2e: where the mood rests, as the pair's two numbers — from the character's dials while the
    character is on (the §4UA link "the character sets where the mood rests"; it read the Mirror's diary, which
    is retired), else the Mirror's (``mirror.rest_pair``). (0, 0) with ``GHOST_MIRROR_REST=0``, without a home,
    or when the character file cannot be read. Read on every read of the mood: ``dials`` is memoised.
    Never raises."""
    try:
        from . import mirror as _mirror
        if not enabled():
            return _mirror.rest_pair(home)
        if not _mirror.rest_enabled() or _root(home) is None or unreadable(home):
            return 0.0, 0.0
        a, b = rest_units(dials(home))
        return a * _mirror.REST_UNIT, b * _mirror.REST_UNIT
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


def section_text(dials: Dict[str, int], liked: str = "") -> str:
    """The state-block section for these dials ("" when the sentence is empty) — with the line about what
    the agent has come to enjoy (``tastes.liked_line``) under the rule, when there is one."""
    s = sentence(dials)
    return f"{HEADER}\n«{s}»\n{RULE}" + (f"\n{liked}" if liked else "") + "\n\n" if s else ""


def _file_key(home) -> tuple:
    """What the section depends on: the character file and the tastes file, by size and mtime."""
    root = _root(home)
    key = []
    for name in (CHARACTER_FILENAME, "tastes.json"):
        p = root / name
        try:
            st = p.stat()
            key += [str(p), st.st_mtime_ns, st.st_size]
        except OSError:
            key += [str(p), 0, 0]
    return tuple(key)


def section(home) -> Tuple[str, str]:
    """(the state-block section, its stamp) for the stored character — re-read only when the file's size
    or mtime changes (the state block is built on every turn). ("", "") without a home. Never raises."""
    root = _root(home)
    if root is None:
        return "", ""
    key = _file_key(home)
    return _memo(home, key)[:2]


def _memo(home, key) -> Tuple[str, str, Dict[str, int]]:
    with _LOCK:
        if _MEMO.get("key") == key:
            return _MEMO["text"], _MEMO["stamp"], dict(_MEMO["dials"])
    state = load(home)
    d = state["dials"]
    try:
        from . import tastes as _tastes
        liked = _tastes.liked_line(home)
    except Exception:  # noqa: BLE001 — the character is heard without its tastes
        liked = ""
    text = section_text(d, liked)
    got = (text, stamp(d) if text else "", dict(d))
    with _LOCK:
        _MEMO.clear()
        _MEMO.update(key=key, text=got[0], stamp=got[1], dials=dict(d), unreadable=bool(state["unreadable"]))
    return got


def unreadable(home) -> bool:
    """Is the character file THERE and not readable? Through the memo (health asks every 25 s)."""
    root = _root(home)
    if root is None:
        return False
    if not (root / CHARACTER_FILENAME).exists():
        return False
    key = _file_key(home)
    try:
        _memo(home, key)                         # one read per file state (and one warning, not one a poll)
        with _LOCK:
            return bool(_MEMO.get("unreadable")) if _MEMO.get("key") == key else bool(load(home)["unreadable"])
    except Exception:  # noqa: BLE001
        return False


def dials(home) -> Dict[str, int]:
    """The stored dials, read through the same memo as ``section`` (the health check asks every 25 s).
    ``START`` without a home or a file. Never raises."""
    root = _root(home)
    if root is None:
        return dict(START)
    key = _file_key(home)
    try:
        return _memo(home, key)[2]
    except Exception:  # noqa: BLE001
        return dict(START)


def describe(dials: Dict[str, int]) -> str:
    """"warmth -1 (a little dry) · humour +1 (fond of a light joke) · directness 0 (neither diplomatic nor blunt)"."""
    bits = []
    for d in DIALS:
        v = clamp(dials.get(d, 0))
        words = PHRASES[d][v] if v else f"neither {POLES[d][0]} nor {POLES[d][1]}"
        bits.append(f"{d} {v:+d} ({words})" if v else f"{d} 0 ({words})")
    return " · ".join(bits)


# ── writing ─────────────────────────────────────────────────────────────────

def _write(home, state: Dict[str, object]) -> None:
    from ..utils.json_store import write_json_atomic
    root = _root(home)
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / CHARACTER_FILENAME, {
        "version": 1, "dials": {d: clamp(state["dials"].get(d, 0)) for d in DIALS},
        "pins": [d for d in DIALS if d in state.get("pins", [])],
        "daily_through": _real_date(state.get("daily_through")),
        # when each dial last changed (a mood day): what the daily rule counts evidence FROM. Kept with
        # the dials, in one atomic write — the log is the story, not the rule's memory (review r2: a log
        # row written before a failed state write reset the evidence of a move that never happened)
        "changed": {d: _real_date(v) for d, v in (state.get("changed") or {}).items()
                    if d in DIALS and _real_date(v)},
        "updated": _iso()})
    with _LOCK:
        _MEMO.clear()


def _log(home, row: Dict[str, object]) -> None:
    from ..utils.json_store import open_append
    root = _root(home)
    root.mkdir(parents=True, exist_ok=True)
    with open_append(root / LOG_FILENAME) as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")


def ensure(home, *, now=None) -> Dict[str, object]:
    """The stored character, written as ``START`` (with its first log row) when there is none yet."""
    with _LOCK:
        state = load(home)
        if _root(home) is None or state["stored"] or state["unreadable"]:
            return state
        _write(home, state)
        _log(home, {"at": _iso(now), "kind": KIND_START, "dials": dict(state["dials"]),
                    "why": "the starting character"})
        state["stored"] = True
        return state


def set_dial(home, dial: str, value, *, why: str = "", now=None) -> Tuple[Dict[str, object], str]:
    """The owner's word: put ``dial`` at ``value`` (-2..2). (the state, what happened in words)."""
    d = resolve_dial(dial)
    if not d:
        return load(home), f"no such dial: {dial!r} (the dials are {', '.join(DIALS)})"
    with _LOCK:
        state = ensure(home, now=now)
        if state["unreadable"]:
            return state, UNREADABLE
        old, new = state["dials"][d], clamp(value)
        if new == old:
            return state, f"{d} is already at {old:+d}"
        state["dials"][d] = new
        state["changed"][d] = _mood_day(now)
        _write(home, state)
        _log(home, {"at": _iso(now), "kind": KIND_OWNER, "dial": d, "from": old, "to": new,
                    "why": str(why or "the owner said so")[:200]})
        return state, f"{d} {old:+d} → {new:+d}"


def set_pin(home, dial: str, pinned: bool, *, now=None) -> Tuple[Dict[str, object], str]:
    d = resolve_dial(dial)
    if not d:
        return load(home), f"no such dial: {dial!r} (the dials are {', '.join(DIALS)})"
    with _LOCK:
        state = ensure(home, now=now)
        if state["unreadable"]:
            return state, UNREADABLE
        if (d in state["pins"]) == bool(pinned):
            return state, f"{d} is already {'pinned' if pinned else 'free to move'}"
        state["pins"] = [x for x in DIALS if (x in state["pins"] or x == d) and (pinned or x != d)]
        if not pinned:
            state["changed"][d] = _mood_day(now)      # what was asked while it was held does not move it now
        _write(home, state)
        _log(home, {"at": _iso(now), "kind": KIND_PIN if pinned else KIND_UNPIN, "dial": d})
        return state, (f"{d} is pinned at {state['dials'][d]:+d}" if pinned else
                       f"{d} is free to move again — the count toward its next move starts from today")


def veto(home, *, now=None) -> Tuple[Dict[str, object], str]:
    """Take the LAST automatic move back (a move already taken back, or one the owner has since
    overridden on that dial, is not there to veto)."""
    with _LOCK:
        state = ensure(home, now=now)
        if state["unreadable"]:
            return state, UNREADABLE
        rows = log_rows(home)
        # the latest automatic move that still STANDS: not itself taken back (a veto names the move it took
        # back, `of`), and with no setting of its dial by the owner after it (review r1: of two dials moved
        # in one run only the last could be taken back; r2: of two moves of ONE dial only the last)
        # (stamp, dial, where it went): two dials moved in one run carry the same stamp
        taken = {(str(x.get("of") or ""), str(x.get("dial") or ""), x.get("of_to"))
                 for x in rows if x.get("kind") == KIND_VETO}
        # … and a move is only taken back FROM where it left the dial (review r3: with the file removed and
        # the log kept, or a move whose log row was lost, the veto "went back" from somewhere else — a ✓ for
        # +0 → +0, or two steps at once). A restart of the character is a barrier: nothing before it stands.
        first = max((i for i, x in enumerate(rows) if x.get("kind") == KIND_START), default=-1) + 1
        r, any_move = None, False
        for i in range(len(rows) - 1, first - 1, -1):
            x = rows[i]
            if (x.get("kind") != KIND_MOVE or str(x.get("dial") or "") not in DIALS
                    or any(isinstance(x.get(k), bool) or not isinstance(x.get(k), int) for k in ("from", "to"))):
                continue                        # not a move, or a row that does not say from where to where
            any_move = True
            if state["dials"][str(x.get("dial"))] != clamp(x.get("to")):
                continue                        # the dial is no longer where this move left it
            if (str(x.get("at") or ""), str(x.get("dial") or ""), x.get("to")) in taken:
                continue
            if any(y.get("dial") == x.get("dial") and y.get("kind") == KIND_OWNER for y in rows[i + 1:]):
                continue
            r = x
            break
        if r is None:
            return state, ("no automatic move still stands (each was taken back, or you set that dial since)"
                           if any_move else "there is no automatic move to take back")
        d = str(r.get("dial"))
        old, back = state["dials"][d], clamp(r.get("from"))
        state["dials"][d] = back
        state["changed"][d] = _mood_day(now)
        _write(home, state)
        _log(home, {"at": _iso(now), "kind": KIND_VETO, "dial": d, "from": old, "to": back,
                    "of": str(r.get("at") or ""), "of_to": r.get("to"),
                    "why": "the owner took the last automatic move back"})
        return state, f"{d} {old:+d} → {back:+d} (the move of {str(r.get('at') or '')[:10]} taken back)"


# ── the daily rule ──────────────────────────────────────────────────────────

def _last_change(rows: List[dict], dial: str) -> str:
    """The date (YYYY-MM-DD) of the dial's last change of any kind, or ""."""
    # an unpin too: what was asked while the dial was held does not move it the day it is let go
    for r in reversed(rows):
        if r.get("dial") == dial and r.get("kind") in (KIND_MOVE, KIND_OWNER, KIND_VETO, KIND_UNPIN):
            return str(r.get("at") or "")[:10]
    return ""


_DATE_RE = None


def _by_date(days: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """The day records in date order, one per date (the last written wins), rows without a real date
    left out — the file is trusted for nothing (review r1: a clock set back wrote days out of order, a
    date repeated three times moved a dial, a row dated "zzz" stopped the rule for good)."""
    global _DATE_RE
    if _DATE_RE is None:
        import re
        _DATE_RE = re.compile(r"^\d{4}-\d\d-\d\d$")
    seen: Dict[str, Dict[str, object]] = {}
    for d in days or []:
        if not isinstance(d, dict):
            continue
        date = _real_date(d.get("date"))
        if date:
            seen[date] = d
    return [seen[k] for k in sorted(seen)]


def propose(days: List[Dict[str, object]], state: Dict[str, object], rows: List[dict],
            through: str, votes: Optional[Dict[str, Dict[str, int]]] = None) -> List[Dict[str, object]]:
    """The moves the day records up to ``through`` (inclusive) ask for — pure. For each dial that is not
    pinned: over the last ``EVIDENCE_DAYS`` recorded days that came AFTER the dial's last change, the days
    that pulled it up (``day_pulls``) against the days that pulled it down; when one side leads by
    ``LEAD_IN`` (toward 0) or ``LEAD_OUT`` (away from 0), the dial moves ONE step that way, never past
    ±``LIMIT``. ``votes`` ({date: {dial: ±1}}, the agent's look at its own replies — ``selfview``) are
    added to the day's pull: a day counts for the side the sum falls on, and for neither when they cancel."""
    out = []
    ordered = [d for d in _by_date(days) if str(d.get("date")) <= through]
    pulls = day_pulls(ordered)
    usable = ordered[-EVIDENCE_DAYS:]
    for dial in DIALS:
        if dial in state.get("pins", []):
            continue
        # the state's own stamp first (a mood day, written with the dials); the log for a file older than it
        since = _real_date((state.get("changed") or {}).get(dial)) or _last_change(rows, dial)
        up, down, seen = [], [], 0
        for d in usable:
            date = str(d.get("date") or "")
            if since and date <= since:
                continue
            own = int((pulls.get(date) or {}).get(dial, 0))
            look = int(((votes or {}).get(date) or {}).get(dial, 0))
            seen += 1 if look else 0
            total = own + look
            if total > 0:
                up.append(date)
            elif total < 0:
                down.append(date)
        lead = len(up) - len(down)
        if not lead:
            continue
        step = 1 if lead > 0 else -1
        old = clamp(state["dials"].get(dial, 0))
        new = clamp(old + step)
        if new == old or abs(lead) < (LEAD_OUT if abs(new) > abs(old) else LEAD_IN):
            continue
        ups, downs = (len(up), len(down)) if step > 0 else (len(down), len(up))
        out.append({"dial": dial, "from": old, "to": new,
                    "why": (f"{ups} days clearly {'better' if step > 0 else 'worse'} than its usual against {downs} "
                            f"the other way, in {SIGNAL_WORDS[dial]}"
                            + (f"; its own look at its replies counted on {seen} of those days" if seen else "")),
                    "days_up": up, "days_down": down})
    return out


def daily(home, days: List[Dict[str, object]], *, now=None,
          votes: Optional[Dict[str, Dict[str, int]]] = None) -> List[str]:
    """Apply the daily rule once per recorded day: nothing happens until a day newer than
    ``daily_through`` is recorded. The moves made, in words. Never raises."""
    try:
        with _LOCK:
            state = ensure(home, now=now)
            if state["unreadable"]:
                return []
            today = _mood_day(now)
            # a day record dated after today is not one (a clock that was wrong, a hand edit): it would
            # become `daily_through` and stop the rule until that date
            usable = [d for d in _by_date(days) if str(d.get("date")) <= today]
            dates = [str(d.get("date")) for d in usable]
            done = _real_date(state.get("daily_through"))
            if done > today:                     # written under a clock that was ahead: not a day to wait for
                done = ""
            if not dates or dates[-1] <= done:
                return []
            through = dates[-1]
            moves = propose(usable, state, log_rows(home), through, votes)
            for mv in moves:
                state["dials"][mv["dial"]] = mv["to"]
                state["changed"][mv["dial"]] = through       # the evidence up to `through` is used up
            state["daily_through"] = through
            _write(home, state)                              # the dials first: a move is real once this returns
            said = []
            for mv in moves:
                try:
                    _log(home, {"at": _iso(now), "kind": KIND_MOVE, **mv})
                except Exception as e:  # noqa: BLE001 — the move stands; only its line in the story is missing
                    logger.warning("character: move not logged (%s)", type(e).__name__)
                said.append(f"{mv['dial']} {mv['from']:+d} → {mv['to']:+d}: {mv['why']}")
            return said
    except Exception as e:  # noqa: BLE001 — the idle window never breaks on the character
        logger.warning("character: the daily rule did not run (%s: %s)", type(e).__name__, e)
        return []


# ── the views ───────────────────────────────────────────────────────────────

def _change_line(r: dict) -> str:
    kind, d = r.get("kind"), r.get("dial")
    day = str(r.get("at") or "")[:10]
    if kind == KIND_START:
        return f"{day}: started as {describe(r.get('dials') or START)}"
    if kind in (KIND_MOVE, KIND_OWNER, KIND_VETO):
        by = {KIND_MOVE: "moved by itself", KIND_OWNER: "set by the owner", KIND_VETO: "taken back by the owner"}[kind]
        try:
            span = f"{int(r.get('from')):+d} → {int(r.get('to')):+d}"
        except (TypeError, ValueError):
            span = "?"
        return f"{day}: {d} {span}, {by}" + (f" — {r.get('why')}" if kind == KIND_MOVE and r.get("why") else "")
    if kind in (KIND_PIN, KIND_UNPIN):
        return f"{day}: {d} {'pinned' if kind == KIND_PIN else 'unpinned'} by the owner"
    return ""


def story(home, *, limit: int = 12) -> str:
    """How the character got here: its changes, newest last, grouped by month."""
    lines, month = [], ""
    rows = [r for r in log_rows(home) if _change_line(r)]
    for r in rows[-limit:]:
        m = str(r.get("at") or "")[:7]
        if m != month:
            month = m
            lines.append(f"{m}:")
        lines.append("  " + _change_line(r))
    return "\n".join(lines)


def view(home) -> str:
    """The character for the owner: the sentence, the dials, the pins, and how it got here."""
    state = load(home)
    s = sentence(state["dials"])
    if state["unreadable"]:
        # review r2: the starting character was shown as if it were the owner's, above a history that
        # contradicted it
        hist = story(home)
        return ("⚠ The character file (system/selfhood/character.json) cannot be read. Until it is fixed or "
                f"removed, replies hear the STARTING character — «{s}» — and no command can change it. "
                "Removing the file starts the character again from there."
                + ("\n\nWhat the log says happened before:\n" + hist if hist else ""))
    out = [f"Character: «{s}»" if s else "Character: every dial is at 0 — no trait is heard.",
           "Dials (-2 to +2): " + describe(state["dials"]),
           ("Pinned (they do not move by themselves): " + ", ".join(state["pins"])) if state["pins"]
           else "Nothing is pinned.",
           "Each dial moves by itself, one step at a time, when over two weeks the days that were clearly "
           "better than my usual lead the days that were clearly worse (or the other way round): warmth from "
           "what comes from you, humour from how my own work goes, directness from how my answers are judged. "
           "Your commands set, hold or take back a dial whenever you want."]
    try:
        from . import selfview as _selfview
        seen = _selfview.line(home)
        if seen:
            out.append(seen)
    except Exception:  # noqa: BLE001 — the view renders without the look
        pass
    try:
        from . import tastes as _tastes
        # review r2 (§4UH): this view is a system reply, and a reply goes back to the model as history —
        # the liked kinds only here too; what has been rough is in the agent's log (the idle window)
        likes = _tastes.line(home, liked_only=True)
        if likes:
            out.append(likes)
    except Exception:  # noqa: BLE001
        pass
    hist = story(home)
    out.append("How it got here:\n" + hist if hist else
               "It has not changed yet: this is the starting character.")
    return "\n".join(out)


def line(home) -> str:
    """The character as ONE line for the `introspect` overview and summary ("" when no trait is heard)."""
    state = load(home)
    s = sentence(state["dials"])
    if state["unreadable"]:
        return (f"character: «{s}» — the STARTING character: the character file cannot be read "
                "(the owner types \"character\" for what to do)")
    if not s:
        return ""
    text = (f"character: «{s}» ({describe(state['dials'])}; three dials that move slowly with how my days "
            "go — the owner types \"character\" to see or change them. How I am, not a report of work "
            "done and not a checked fact)")
    try:
        from . import selfview as _selfview
        from . import tastes as _tastes
        # review r1 (§4UH): this line is a tool result the model reads mid-turn — the LIKED kinds only;
        # what has been rough is said in the agent's log only (review r2), never where a model reads
        seen, likes = _selfview.line(home), _tastes.line(home, liked_only=True)
        return text + (" " + seen if seen else "") + (" " + likes if likes else "")
    except Exception:  # noqa: BLE001
        return text


# ── the owner's command (answered by the system, like "mirror") ─────────────

import re as _re  # noqa: E402

#: a command is short; a longer message is never parsed as one (review r3: the whole-message regex this
#: replaced was cubic on a run of blanks — a message starting with the word and holding 1,600 spaces took
#: four seconds, before the turn began, with the event loop held)
COMMAND_MAX_CHARS = 120
_WORD = "character"
USAGE = ("Commands: \"character\" (show it) · \"character humour +1\" or \"character humour up\" / \"down\" "
         "(set a dial, -2 to +2) · \"character pin warmth\" / \"character unpin warmth\" · "
         "\"character veto\" (take the last automatic move back). Dials: " + ", ".join(
             f"{d} ({POLES[d][0]} ↔ {POLES[d][1]})" for d in DIALS) + ".")


_POSITION_RE = _re.compile(r"^[+-]?[0-2]$")
_STEPS = {"up": 1, "more": 1, "down": -1, "less": -1}
_DASHES = str.maketrans({"−": "-", "–": "-", "—": "-", "＋": "+"})


def _command_words(text: str) -> Optional[List[str]]:
    """The words after "character" (a ":" or "," after it, "=" between a dial and its value, a spaced
    sign and a closing "please" are all the owner's way of typing the same thing), or None when the
    message does not start with the word or is too long to be a command. No regex runs over the message."""
    t = str(text or "")
    if len(t) > COMMAND_MAX_CHARS:
        return None
    t = t.translate(_DASHES).strip()
    if t.startswith("/"):
        t = t[1:]
    after = t[len(_WORD):len(_WORD) + 1]
    if t[:len(_WORD)].lower() != _WORD or (after and (after.isalnum() or after in "_-")):
        return None                              # "characters", "character_count", "character-driven"
    rest = t[len(_WORD):].strip(" \t\r\n:,").rstrip(" \t\r\n.!?")
    rest = _re.sub(r"([+-]) +(?=\d)", r"\1", rest.replace("=", " "))
    words = [w for w in (x.strip(",:;").lower() for x in rest.split()) if w]
    while words and words[-1] in ("please", "pls", "thanks"):
        words.pop()
    return words


def parse_command(text: str) -> Optional[Tuple[str, ...]]:
    """The command a message IS, or None. The WHOLE message must be one of the command's shapes (as the
    rule commands and `mirror` decide): "character" · "character help|show" · "character veto|undo" ·
    "character pin|unpin <dial>" · "character [set] <dial> [to|at] <up|down|more|less|-2..+2>". Anything
    else — "character development in this novel is weak", "character set in UTF-8?", "character warmth
    is what I need" — is the owner talking to the model (review r1: a looser match answered those with
    "That is not a character command")."""
    words = _command_words(text)
    if words is None:
        return None
    if not words or words in (["help"], ["show"]):
        return ("view",)
    if words in (["veto"], ["undo"]):
        return ("veto",)
    if len(words) == 2 and words[0] in ("pin", "unpin") and resolve_dial(words[1]):
        return (words[0], resolve_dial(words[1]))
    if words[0] == "set":
        words = words[1:]
    if len(words) == 3 and words[1] in ("to", "at"):
        words = [words[0], words[2]]
    if len(words) == 2 and resolve_dial(words[0]):
        if words[1] in _STEPS or _POSITION_RE.match(words[1]):
            return ("set", resolve_dial(words[0]), words[1])
    return None


#: what the model is told when the owner's message is nearly a command (review r2: after the match became
#: exact, "character humour 3", "character reset" and "character humour" went to the model with no word
#: that nothing had changed — and the model, which reads the character in its state block, could answer
#: as if it had changed it)
HINT_MAX_WORDS = 6
_HINT_WORDS = frozenset(DIALS) | {"humor", "reset", "pin", "unpin", "veto", "undo"}


def command_hint(text: str) -> str:
    """"" for a real command and for ordinary talk; the note for a message that starts with "character",
    is short, names a dial or a verb of the command, and is NOT one of its shapes."""
    words = _command_words(text)
    if words is None or parse_command(text) is not None:
        return ""
    if len(words) > HINT_MAX_WORDS or not (_HINT_WORDS & set(words)):
        return ""
    return ("The owner's message looks like a \"character\" command but is not one the system knows, so the "
            "character was NOT changed — you cannot change it yourself and must not say it changed. If the "
            "owner was asking about something else (a character set, a character in a story), just answer "
            "that. Otherwise tell the owner so in one sentence and give the exact forms: " + USAGE
            + " There is no reset: each dial is set with its own command, and a position is a whole number "
              "from -2 to +2.")


def is_command(text: str) -> bool:
    return parse_command(text) is not None


def command(text: str, home, *, now=None) -> Optional[str]:
    """What the owner's "character …" message did, as the banner the system shows — None when the message
    is not that command. The caller decides that it was the OWNER's, on a private surface."""
    cmd = parse_command(text)
    if cmd is None:
        return None
    if cmd[0] == "view":
        return view(home) + "\n\n" + USAGE
    if cmd[0] == "veto":
        _state, said = veto(home, now=now)
        return f"✓ {said}\n\n{view(home)}" if "→" in said else f"{said[0].upper()}{said[1:]}."
    if cmd[0] in ("pin", "unpin"):
        _state, said = set_pin(home, cmd[1], cmd[0] == "pin", now=now)
        done = " is pinned at " in said or " is free to move again" in said
        return (f"✓ {said}\n\n{view(home)}" if done and "already" not in said
                else f"{said[0].upper()}{said[1:]}. Nothing changed.")
    _verb, d, value = cmd
    cur = load(home)["dials"][d]
    target = cur + _STEPS[value] if value in _STEPS else int(value)       # _POSITION_RE: a whole number
    if not -LIMIT <= target <= LIMIT:
        return f"{d} is already at its end ({cur:+d}). Nothing changed."
    _state, said = set_dial(home, d, target, why="the owner's \"character\" command", now=now)
    return f"✓ {said}\n\n{view(home)}" if "→" in said else f"{said[0].upper()}{said[1:]}. Nothing changed."
