"""The day record (§4UG): what one day was like, as counts — no model call, no words of a message.

Until now nothing kept a day. The mood halves every two hours, its journal is a list of label changes, and
the only thing that looked back further was the Mirror's note about a mood that lasted three days of seven
(which counted a mood the owner's thumb had made as the agent's own: a journal row carries the kind of the
event that crossed a boundary, not of the events that built the mood). The day record is the layer between
the mood (hours) and the character (weeks): one row per mood day (``mood.mood_day`` — it ends at 04:00),
written once the day is over, from the two files the mood already keeps:

* ``mood.events.jsonl`` — every event that moved or tried to move the pair, with its kind and size. Counted
  BY KIND, so what the agent's own work did and what the owner did stay apart;
* ``mood.history.jsonl`` — the label changes, for how long each mood was held (the arc) and the day's peak.

A row holds numbers, labels, manner names and the events' own cause sentences (the system's words: "2 tool
calls failed (of 5) in this turn"). It never holds a message, a task name or a subject, so it can be shown
to the character's rules and, later, to a model without carrying the owner's work along.

Storage: ``$GHOST_HOME/system/selfhood/days.jsonl``, append-only, one row per day, written by
``close_days`` from the idle window. Reading never writes.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import mood as M
from .schema import read_jsonl_rows

logger = logging.getLogger("GhostSelfhood")

DAYS_FILENAME = "days.jsonl"
SELFHOOD_REL = "system/selfhood"
#: how far back ``close_days`` goes for days nobody recorded (the agent was off, or this is the first run)
BACKFILL_DAYS = 14
#: the biggest events kept per day (the owner's own set_mood is not one: it is the owner's word, not an event)
BIGGEST = 3
DAYS_MAX_ROWS = 800
DAYS_KEEP_ROWS = 600

_LOCK = threading.Lock()

#: the manner an owner's manner event names — ``state.set_manner`` writes "the owner set the manner: X, n of 3"
_MANNER_SET_RE = re.compile(r"^the owner set the manner: ([a-z]+), \d of 3$")
#: the label an owner's set_mood left — the event's ``after`` reads "cheerful, 2 of 3" or "neutral"
_AFTER_RE = re.compile(r"^([a-z]+)(?:, (\d) of 3)?$")
#: the parts of a turn's cause sentence (``mood.turn_events`` joins them with "; ")
#: Order matters: the clean-turn sentence may itself say "… went through, 1 of them failed" (review r1: it
#: was counted as trouble twice). The trouble sentence is the one that ends "in this turn".
_OWN_PARTS = (
    ("clean_turns", re.compile(r"^a turn of \d+ tool calls? went through")),
    ("research", re.compile(r"^read \d+ sources in a research turn$")),
    ("tasks_done", re.compile(r"^finished \d+ project tasks?(?: \(in a background job\))?$")),
    ("long_turns", re.compile(r"^a very long turn \(")),
    ("trouble", re.compile(r"(?:failed\b.*|a loop guard stepped in .*|the context ran out of room .*) in this turn$")),
)


def _root(home) -> Optional[Path]:
    return Path(home) / SELFHOOD_REL if home else None


def _when(row) -> Optional[datetime.datetime]:
    return M.parse_iso_utc(str((row or {}).get("at") or ""))


def _own_counts(cause: str) -> Dict[str, int]:
    """What one turn event's cause says happened: each part of the sentence counted once."""
    out: Dict[str, int] = {}
    for part in str(cause or "").split("; "):
        for name, rx in _OWN_PARTS:
            if rx.search(part.strip()):
                out[name] = out.get(name, 0) + 1
                break
    return out


def owner_thumbs(home, day: datetime.date, *, tz=None) -> Tuple[int, int]:
    """(👍, 👎) the owner gave on mood day ``day``, from the label store (``corrections.jsonl``): each
    turn counts once, by the owner's LAST thumb on it, on the day that thumb was given. §4UH review r2: the
    mood journal cannot say this — a thumb sent again, re-booked after the character's lean changed, or
    given at the mood's limit is journalled with whatever size it moved, so its size cannot tell a new
    thumb from the same one. Two limits (review r3): a thumb taken back (web, handheld, Slack) writes no
    row here and so still counts; and a thumb changed on a later day counts on both days (a day record is
    written once)."""
    from .tastes import OWNER_FEEDBACK
    if not home:
        return 0, 0
    last: Dict[str, Tuple[str, str]] = {}
    for r in read_jsonl_rows(Path(home) / "system" / "trajectories" / "corrections.jsonl", what="labels"):
        # review: matched as the mood matches it (`feedback.is_owner_label_source` lowercases)
        if str(r.get("source") or "").strip().lower() in OWNER_FEEDBACK and r.get("outcome") in ("passed", "failed"):
            last[str(r.get("trajectory_id") or "")] = (str(r.get("outcome")), str(r.get("timestamp") or ""))
    up = down = 0
    for outcome, ts in last.values():
        at = M.parse_iso_utc(ts)
        if at is None or M.mood_day(at, tz=tz) != day:
            continue
        if outcome == "passed":
            up += 1
        else:
            down += 1
    return up, down


#: held reasons that still say something HAPPENED that day (the owner's hold, the mood at its limit)
#: — the owner's hold, and a tone refused only because sentiment's share was full (a cap on NEW messages, not
#: a repeat: review). Not "already counted for this reply" (a repeat). (The same strings as
#: ``SelfStateThread.HELD_OWNER_HOLD`` / ``HELD_OWNER_LATER`` and ``SelfModel.HELD_SENTIMENT_FULL`` — pinned
#: equal in the tests; literals here so this reader imports neither. An event at the mood's limit is journalled
#: as applied, so it is counted already.) Known: under the owner's hold the reaction ledger is not asked, so a
#: tone on a reply that also has a thumb counts on a held day and not on another — WARMTH_CAP bounds it.
EVIDENCE_HELD = frozenset({"the owner set the mood (held until the day ends)",
                           "the owner set the mood after the message",
                           "sentiment's share of the mood is full"})


def _count_evidence(rec: Dict[str, object], kind: str, cause: str, dp: float, ev: dict) -> None:
    """What one event says about the day: corrections, praise, tones, the owner's moods and manners, the
    agent's own work. Thumbs are counted from the label store (``owner_thumbs``)."""
    if kind in (M.KIND_THUMB_UP, M.KIND_THUMB_DOWN, M.KIND_THUMB_OFF):
        pass                                        # counted from the label store below (§4UH review r2)
    elif kind == M.KIND_CORRECTION:
        rec["corrections"] = int(rec["corrections"]) + 1
    elif kind == M.KIND_PRAISE:
        rec["praise"] = int(rec["praise"]) + 1
    elif kind == M.KIND_SENTIMENT:
        if dp:
            rec["tone_up" if dp > 0 else "tone_down"] = int(rec["tone_up" if dp > 0 else "tone_down"]) + 1
    elif kind == M.KIND_OWNER:
        m = _AFTER_RE.match(str(ev.get("after") or ""))
        # "neutral" is the owner taking the mood OFF, not a mood the owner asked for
        if m and m.group(1) in M.LABELS and m.group(1) not in rec["owner_moods"]:
            rec["owner_moods"].append(m.group(1))
    elif kind == M.KIND_MANNER:
        m = _MANNER_SET_RE.match(cause)
        if m and m.group(1) in M.MANNERS and m.group(1) not in rec["manners"]:
            rec["manners"].append(m.group(1))
    elif kind == M.KIND_TURN:
        for name, n in _own_counts(cause).items():
            rec["own"][name] = int(rec["own"].get(name, 0)) + n


def build_day(home, day: datetime.date, *, tz=None, now: Optional[datetime.datetime] = None) -> Dict[str, object]:
    """The record of mood day ``day``, computed from the mood's two journals. Pure reading; never raises
    past an unreadable file (an absent journal is an empty day)."""
    root = _root(home)
    events = read_jsonl_rows(root / "mood.events.jsonl", what="mood events") if root else []
    history = read_jsonl_rows(root / "mood.history.jsonl", what="mood journal") if root else []
    rec: Dict[str, object] = {
        "date": day.isoformat(), "events": 0, "held": 0, "by_kind": {},
        "thumbs_up": 0, "thumbs_down": 0, "corrections": 0, "praise": 0, "tone_up": 0, "tone_down": 0,
        "owner_moods": [], "manners": [],
        "own": {"trouble": 0, "clean_turns": 0, "research": 0, "tasks_done": 0, "long_turns": 0},
        "arc": {}, "peak": None, "biggest": [], "shades": {},
    }
    by_kind: Dict[str, Dict[str, float]] = {}
    candidates = []
    for ev in events:
        at = _when(ev)
        if at is None or M.mood_day(at, tz=tz) != day:
            continue
        kind = str(ev.get("kind") or "")
        held = ev.get("held")
        try:
            dp, de = float(ev.get("dp") or 0.0), float(ev.get("de") or 0.0)
        except (TypeError, ValueError):
            dp = de = 0.0
        cause = str(ev.get("cause") or "")
        if held:
            rec["held"] = int(rec["held"]) + 1          # it happened, it did not move the mood
            # §4UH-e2e review (MAJOR): a day the owner set the mood held every event until 04:00, and the
            # record skipped them — that day's corrections, tones and turns were lost to the character (a day
            # of corrections could pull the dials UP). What the owner's hold or the mood's limit refused still
            # HAPPENED: it is evidence about the day. What was refused as a repeat (already counted) is not
            # counted twice.
            if str(held) not in EVIDENCE_HELD:
                continue
            _count_evidence(rec, kind, cause, dp, ev)
            continue
        rec["events"] = int(rec["events"]) + 1
        slot = by_kind.setdefault(kind, {"n": 0, "dp": 0.0, "de": 0.0})
        slot["n"] += 1
        slot["dp"] = round(slot["dp"] + dp, 4)
        slot["de"] = round(slot["de"] + de, 4)
        _count_evidence(rec, kind, cause, dp, ev)
        if kind not in (M.KIND_OWNER, M.KIND_MANNER):
            candidates.append((abs(dp) + abs(de), {"at": str(ev.get("at") or ""), "kind": kind,
                                                   "dp": round(dp, 4), "de": round(de, 4), "cause": cause[:160]}))
    rec["by_kind"] = by_kind
    rec["thumbs_up"], rec["thumbs_down"] = owner_thumbs(home, day, tz=tz)
    candidates.sort(key=lambda c: -c[0])
    rec["biggest"] = [c[1] for c in candidates[:BIGGEST] if c[0] > 0]
    rec["arc"], rec["peak"] = _arc(history, day, tz, now)
    shades: Dict[str, int] = {}
    for row in (read_jsonl_rows(root / "shades.jsonl", what="shades") if root else []):
        at = _when(row)
        if at is not None and M.mood_day(at, tz=tz) == day and row.get("shade"):
            shades[str(row["shade"])] = shades.get(str(row["shade"]), 0) + 1
    rec["shades"] = shades
    return rec


def _arc(history: List[dict], day: datetime.date, tz, now: Optional[datetime.datetime] = None):
    """(minutes held per label that day, the day's peak) from the journal's label changes. The label held
    when the day began is the last row before it; the last label runs to the day's end (or to now)."""
    start = M.mood_day_end(day - datetime.timedelta(days=1), tz=tz)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    end = min(M.mood_day_end(day, tz=tz), now)
    rows = []
    for r in history:
        at = _when(r)
        if at is None or str(r.get("kind") or "") == M.KIND_MANNER:      # a manner row repeats the mood as it stood
            continue
        try:
            level = int(r.get("intensity") or 0)
        except (TypeError, ValueError):
            level = 0
        rows.append((at, str(r.get("label") or M.NEUTRAL), level, str(r.get("kind") or "")))
    rows.sort(key=lambda r: r[0])
    # a day is [start, end): a row stamped exactly 04:00 is the NEW day's, as its event is (review r1)
    label, level, kind0 = M.NEUTRAL, 0, ""
    for at, lab, lev, kind in rows:
        if at >= start:
            break
        label, level, kind0 = lab, lev, kind
    arc: Dict[str, int] = {}
    peak: Optional[Dict[str, object]] = None
    peak_level = 0
    cursor, kind_now = start, kind0
    for at, lab, lev, kind in [r for r in rows if start <= r[0] < end] + [(end, "", 0, "")]:
        seconds = max(0.0, (at - cursor).total_seconds())
        if seconds > 0 and label != M.NEUTRAL and level > 0:
            if seconds >= 60:
                arc[label] = arc.get(label, 0) + int(seconds // 60)
            # the peak is the strongest mood HELD that day — the one carried in from the day before
            # included, a label replaced the instant it was written not; the boredom of a long quiet is
            # in the arc, never the peak (it is read over a neutral mood)
            if seconds >= 60 and kind_now != M.KIND_QUIET and level > peak_level:
                peak_level = level
                peak = {"label": label, "intensity": level, "at": cursor.isoformat().replace("+00:00", "Z")}
        cursor, label, level, kind_now = at, lab, lev, kind
    return arc, peak


def load_days(home, *, limit: int = 0) -> List[Dict[str, object]]:
    """The recorded days, oldest first by date (the newest ``limit`` when given); a date recorded twice
    counts once, its first row (§4UH: rows seeded at deploy were written before ones already there).
    [] when there are none."""
    root = _root(home)
    if root is None:
        return []
    seen, rows = set(), []
    for r in read_jsonl_rows(root / DAYS_FILENAME, what="day records"):
        d = str(r.get("date") or "")
        if d and d not in seen:
            seen.add(d)
            rows.append(r)
    rows.sort(key=lambda r: str(r.get("date")))
    return rows[-limit:] if limit else rows


def close_days(home, *, now: Optional[datetime.datetime] = None, tz=None) -> List[str]:
    """Write the record of every mood day that is OVER and not yet recorded (at most ``BACKFILL_DAYS``
    back, and never a day before the mood's journals begin). The dates written, oldest first. Idempotent:
    a recorded day is never written again. Never raises."""
    root = _root(home)
    if root is None:
        return []
    try:
        today = M.mood_day(now, tz=tz)
        with _LOCK:
            have = {str(r.get("date")) for r in load_days(home)}
            first = _first_journal_day(root, tz)
            if first is None:
                return []
            written = []
            for back in range(BACKFILL_DAYS, 0, -1):
                day = today - datetime.timedelta(days=back)
                if day < first or day.isoformat() in have:
                    continue
                rec = build_day(home, day, tz=tz, now=now)
                rec["written_at"] = (now or datetime.datetime.now(datetime.timezone.utc)).astimezone(
                    datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
                _append(root / DAYS_FILENAME, rec)
                written.append(day.isoformat())
            return written
    except Exception as e:  # noqa: BLE001 — the idle window never breaks on a diary
        logger.warning("day records: not written (%s: %s)", type(e).__name__, e)
        return []


def _first_journal_day(root: Path, tz) -> Optional[datetime.date]:
    """The mood day of the oldest event the journal still holds — nothing before it can be told."""
    for row in read_jsonl_rows(root / "mood.events.jsonl", what="mood events"):
        at = _when(row)
        if at is not None:
            return M.mood_day(at, tz=tz)
    return None


def _append(path: Path, row: dict) -> None:
    from ..utils.json_store import open_append
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_append(path) as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")
    rows = read_jsonl_rows(path, what="day records")
    if len(rows) > DAYS_MAX_ROWS:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[-DAYS_KEEP_ROWS:]),
                       encoding="utf-8")
        tmp.replace(path)


def day_line(rec: Dict[str, object]) -> str:
    """One line a person can read: "2026-11-10: irritable 40 min, content 3 h · own work: 1 trouble,
    4 clean · you: 2 👍, 1 correction, manner sarcastic"."""
    def _span(minutes: int) -> str:
        return f"{minutes} min" if minutes < 90 else f"{minutes / 60:.1f} h".replace(".0 h", " h")
    arc = rec.get("arc") or {}
    moods = ", ".join(f"{lab} {_span(int(m))}" for lab, m in sorted(arc.items(), key=lambda kv: -int(kv[1]))) \
        or "no mood"
    own = rec.get("own") or {}
    own_bits = [f"{own[k]} {name}" for k, name in (("trouble", "trouble"), ("clean_turns", "clean"),
                                                   ("tasks_done", "task done"), ("research", "research"))
                if own.get(k)]
    you = []
    for key, name in (("thumbs_up", "👍"), ("thumbs_down", "👎"), ("corrections", "correction"),
                      ("praise", "thanks")):
        if rec.get(key):
            you.append(f"{rec[key]} {name}")
    if rec.get("owner_moods"):
        you.append("set mood " + "/".join(rec["owner_moods"]))
    if rec.get("manners"):
        you.append("manner " + "/".join(rec["manners"]))
    if rec.get("shades"):
        moods += " (" + ", ".join(f"{k} ×{v}" if v > 1 else k for k, v in sorted(rec["shades"].items())) + ")"
    return (f"{rec.get('date')}: {moods}"
            + (" · own work: " + ", ".join(own_bits) if own_bits else "")
            + (" · you: " + ", ".join(you) if you else ""))
