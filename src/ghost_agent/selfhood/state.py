"""Self-state thread — proposal item #3.

The cross-session "state vector": the mood and when the agent was last
active.

§4TC (2026-11-06): the mood is a PAIR of numbers that decays and is moved by
events (``selfhood/mood.py`` holds the arithmetic; this file holds the stored
pair, the journal and the read-modify-write). ``mood()`` is a pure read — it
returns what the stored pair is worth NOW; ``nudge`` is the one function
every event goes through; ``set_owner_mood`` / ``reset_mood`` /
``set_manner`` are the owner's word (``introspect(action="set_mood")``).
Every change of label or intensity appends one row to
``mood.history.jsonl``.

§4TB (2026-11-05): the open questions and unfinished threads this file also
held are gone with the `self_state` tool that authored them. A ``state.json``
written before that still carries the two lists: they are ignored on read
and are no longer written (``SelfState.from_dict`` filters unknown keys). A
mood written before §4TC (a label with ``source`` self / derived) has no
pair: it is ignored and the mood starts neutral; rows of that shape in the
history file are skipped by the reader.

Storage: a single JSON file at ``$GHOST_HOME/system/selfhood/
state.json``. Read-modify-write under a process lock; corruption
(JSON decode error, partial write) is treated as "no prior state"
rather than crashed — we always start from a known-good empty state
rather than poison the next session with a half-state.
"""

from __future__ import annotations

from ..utils.json_store import open_append  # torn-tail-safe JSONL appends (§4MF)
import datetime
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import mood as M
from .schema import MOOD_RECENT_CAUSES, Mood, SelfState, _utcnow_iso

logger = logging.getLogger("GhostSelfhood")


STATE_FILENAME = "state.json"
MOOD_HISTORY_FILENAME = "mood.history.jsonl"

#: Bounded growth for the mood journal: once it holds more than
#: ``MOOD_HISTORY_MAX_ROWS`` rows the oldest are dropped, down to
#: ``MOOD_HISTORY_KEEP_ROWS``.
MOOD_HISTORY_MAX_ROWS = 1200
MOOD_HISTORY_KEEP_ROWS = 1000

# Input caps for the texts a mood row carries.
_MOOD_LABEL_MAX_CHARS = 64
_MOOD_EVIDENCE_MAX_CHARS = 300

#: the cause a row carries when the owner set the mood in chat
OWNER_CAUSE = "the owner said so (set_mood)"
#: … and when a mood stepped down or ended with nothing happening
DECAY_CAUSE = "it wore off"


def _cap_text(s: str, cap: int) -> str:
    """Strip + cap with an ellipsis when truncated (total length ≤ cap)."""
    s = (s or "").strip()
    if len(s) <= cap:
        return s
    return s[: cap - 1].rstrip() + "…"


def _aware(now: Optional[datetime.datetime]) -> datetime.datetime:
    """``now`` as an aware UTC datetime (None = the wall clock)."""
    if now is None:
        return datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=datetime.timezone.utc)
    return now.astimezone(datetime.timezone.utc)


def _owner_is_asking() -> bool:
    """Is the request being served the OWNER's? (Then no boredom is read:
    his message ended the quiet.) Unknown → no."""
    try:
        from ..utils.logging import request_kind
        return request_kind() == "owner"
    except Exception:  # noqa: BLE001
        return False


#: two readings of where a quiet began that are this close are the same quiet
_SAME_QUIET_S = 60.0
#: how far back in the journal an open quiet is looked for
_BORED_SCAN_ROWS = 50


def _iso(now: Optional[datetime.datetime]) -> str:
    """The on-disk timestamp for ``now`` (None = the wall clock)."""
    if now is None:
        return _utcnow_iso()
    if now.tzinfo is not None:
        now = now.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return now.isoformat() + "Z"


@dataclass(frozen=True)
class MoodView:
    """The mood as it is NOW — what every reader gets."""
    label: str = M.NEUTRAL
    intensity: int = 0
    p: float = 0.0
    e: float = 0.0
    manner: str = ""
    manner_intensity: int = 0
    source: str = M.SOURCE_AUTO
    set_at: str = ""            # when this label and intensity began
    cause: str = ""             # why it is what it is
    manner_set_at: str = ""
    recent: Tuple[Dict[str, str], ...] = ()   # the last events that moved it

    @property
    def neutral(self) -> bool:
        return self.intensity == 0

    def tone(self) -> str:
        """The one line a reply's prompt carries ("" when there is nothing
        to say)."""
        return M.tone_line(self.label, self.intensity, self.manner, self.manner_intensity)

    def describe(self) -> str:
        """"irritable, 2 of 3" / "neutral"."""
        return M.NEUTRAL if self.neutral else f"{self.label}, {self.intensity} of 3"


class SelfStateThread:
    """Single-file persisted self-state.

    Read-on-construct, write-on-every-mutation. Cheap (file is small)
    and means a crash-restart picks up exactly where the last successful
    write left off."""

    def __init__(self, root: Path, *, enabled: bool = True):
        self.root = Path(root)
        self.path = self.root / STATE_FILENAME
        self.mood_history_path = self.root / MOOD_HISTORY_FILENAME
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        self._state: SelfState = self._read_or_empty()
        #: how long the owner has been quiet, in seconds (None = unknown) —
        #: attached by the process that owns the clock (``main.py``); this
        #: class only ever READS it
        self._quiet_fn: Optional[Callable[[], Optional[float]]] = None
        #: (start of the quiet as a timestamp, highest boredom level the
        #: journal has a row for) — what `_note_bored` last worked out
        self._bored_seen: Optional[Tuple[float, int]] = None

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def _read_or_empty(self) -> SelfState:
        if not self.path.exists():
            return SelfState()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "self-state read failed (%s); starting from empty state", e,
            )
            return SelfState()
        try:
            return SelfState.from_dict(data)
        except Exception as e:
            logger.warning(
                "self-state schema mismatch (%s); starting from empty state", e,
            )
            return SelfState()

    def _flush(self) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # Atomic-ish: write to a sibling temp file then rename.
                # Avoids leaving a half-written state.json that would
                # poison the next session's wake-up.
                tmp = self.path.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps(self._state.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp.replace(self.path)
        except Exception as e:
            logger.warning("self-state flush failed: %s", e)

    # -----------------------------------------------------------------
    # Read API
    # -----------------------------------------------------------------

    @property
    def state(self) -> SelfState:
        return self._state

    @property
    def lock(self) -> threading.RLock:
        """Public handle on the state RLock, for callers that need a
        read-modify-write to be atomic against concurrent mutators.
        Re-entrant. Exposed deliberately: reaching into ``_lock`` from
        another module would turn a future rename into a silent no-op
        inside that caller's blanket exception guard."""
        return self._lock

    def attach_owner_clock(self, quiet_fn: Optional[Callable[[], Optional[float]]]) -> None:
        """Give the thread a way to ask how long the owner has been quiet —
        the FALLBACK, used only while no owner turn has been stamped in the
        state file (``last_owner_turn_at``)."""
        self._quiet_fn = quiet_fn if callable(quiet_fn) else None

    def quiet_seconds(self, *, now: Optional[datetime.datetime] = None) -> Optional[float]:
        """How long the owner has been quiet, or None when nobody knows.
        Measured from the PERSISTED stamp of the last owner turn, so a
        restart does not change it; the in-memory clock of the process is
        asked only when there is no stamp yet."""
        age = M.age_seconds(self._state.last_owner_turn_at, now=now)
        if age is not None:
            return age
        fn = self._quiet_fn
        if fn is None:
            return None
        try:
            q = fn()
            return float(q) if q is not None else None
        except Exception as e:  # noqa: BLE001 — an unreadable clock is an unknown quiet
            logger.debug("owner clock unreadable: %s", e)
            return None

    def _quiet(self, quiet_s, now: Optional[datetime.datetime]) -> Optional[float]:
        """The quiet a READ applies. An explicit value wins; a request being
        answered by the owner is never quiet (his message ended it)."""
        if quiet_s is not None:
            return quiet_s
        if _owner_is_asking():
            return None
        return self.quiet_seconds(now=now)

    @staticmethod
    def _evolve(stored: Optional[Mood], now: Optional[datetime.datetime]):
        """What the stored pair is worth at ``now``: ``(p, e, label,
        intensity, novelty)``. Pure — decay, then the label and intensity
        read off with what was held. (Boredom is not here: it is never a
        value of the pair.)"""
        if stored is None:
            return 0.0, 0.0, M.NEUTRAL, 0, False
        # an age that cannot be established is not "now": the pair is
        # worth nothing (a mood must never be presented as current on a
        # timestamp nobody can read)
        age = M.age_seconds(stored.updated_at, now=now)
        if age is None:
            p = e = 0.0
        else:
            p, e = M.decay(stored.p, age), M.decay(stored.e, age)
        label, n, novelty = M.read_off(p, e, held_label=stored.label, held_intensity=stored.intensity,
                                       novelty=stored.novelty)
        return p, e, label, n, novelty

    @staticmethod
    def _wore_off_at(stored: Optional[Mood], n: int, now: Optional[datetime.datetime]) -> str:
        """WHEN a mood that only decayed came down to intensity ``n`` — the
        stamp its journal row and its "since" carry, though nobody was
        looking at that moment. Never later than ``now``, never before the
        pair was stored; ``now`` when it cannot be worked out."""
        try:
            start = M.parse_iso_utc(stored.updated_at) if stored is not None else None
            if start is None:
                return _iso(now)
            end = _aware(now)
            at = start + datetime.timedelta(
                seconds=M.decay_step_s(max(abs(stored.p), abs(stored.e)), n))
            return _iso(min(max(at, start), end).replace(microsecond=0))
        except Exception:  # noqa: BLE001 — a stamp never costs the mood
            return _iso(now)

    def mood(self, *, now: Optional[datetime.datetime] = None, quiet_s=None) -> MoodView:
        """The mood NOW. A read never MOVES the mood, whoever asks: no event
        is applied and nothing of a boredom is stored. What a read can do,
        and only while it sees a boredom (``_note_bored``): settle a mood
        that wore off unseen — the stored pair is rewritten to its decayed
        value, the same value any reader computes from it, and the "it wore
        off" row is added — and add the journal row of a boredom level it is
        the first to see (at most one per level per quiet)."""
        stored = self._state.mood
        p, e, label, n, _nov = self._evolve(stored, now)
        manner = stored.manner if stored is not None else ""
        base = dict(
            p=round(p, 4), e=round(e, 4),
            manner=manner,
            manner_intensity=(stored.manner_intensity if manner else 0),
            manner_set_at=(stored.manner_set_at if manner else ""),
            recent=tuple(dict(r) for r in (stored.recent if stored is not None else ())),
        )
        if n == 0:
            # boredom: only over a NEUTRAL mood — a real mood that has not
            # worn off (the owner's included) is not replaced by it
            quiet = self._quiet(quiet_s, now)
            level = M.boredom_level(quiet)
            if level:
                start = _aware(now) - datetime.timedelta(seconds=float(quiet))
                self._note_bored(level, start, now)
                return MoodView(label=M.BORED, intensity=level, source=M.SOURCE_AUTO,
                                set_at=_iso(self._bored_since(level, start, now)),
                                cause=M.quiet_cause(level), **base)
        base.update(label=label, intensity=n)
        if stored is None and n == 0:
            return MoodView(**base)                     # nothing on file, nothing to say
        if stored is not None and (label, n) == (stored.label, stored.intensity):
            return MoodView(source=(stored.source if n else M.SOURCE_AUTO),
                            set_at=stored.set_at, cause=(stored.cause if n else ""), **base)
        # the pair has decayed past a boundary since it was stored and nobody
        # has written that down yet
        keeps_owner = (stored is not None and stored.source == M.SOURCE_OWNER
                       and n and label == stored.label)
        return MoodView(source=(M.SOURCE_OWNER if keeps_owner else M.SOURCE_AUTO),
                        set_at=self._wore_off_at(stored, n, now),
                        cause=(DECAY_CAUSE if n else ""), **base)

    # -----------------------------------------------------------------
    # The journal
    # -----------------------------------------------------------------

    def _append_history_locked(self, m: Mood, kind: str, note: str = "",
                               prev: str = "") -> None:
        """One journal row for a change of label or intensity — and one
        operator line (🪞). Failures are swallowed — the mood itself is
        already in state.json."""
        try:
            from ..utils.logging import Icons, pretty_log
            pretty_log("Selfhood", f"mood {prev or M.NEUTRAL} → {self._describe(m)}: {m.cause}",
                       icon=Icons.SELF_STATE)
        except Exception:  # noqa: BLE001 — a log line never costs the row
            pass
        if not self.enabled:
            return
        try:
            self.mood_history_path.parent.mkdir(parents=True, exist_ok=True)
            rec = {"label": m.label, "intensity": m.intensity,
                   "p": round(m.p, 3), "e": round(m.e, 3),
                   "at": m.set_at, "cause": m.cause, "kind": kind}
            if note:
                rec["note"] = note
            with open_append(self.mood_history_path) as f:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
            self._maybe_compact_mood_history_locked()
        except Exception as e:
            logger.warning("mood history append failed: %s", e)

    def _maybe_compact_mood_history_locked(self) -> None:
        """Keep the newest ``MOOD_HISTORY_KEEP_ROWS`` rows once the journal
        holds more than ``MOOD_HISTORY_MAX_ROWS``. Caller holds the lock.
        Works on BYTES: a line that is not UTF-8 (a torn write, a hand edit)
        is counted and carried like any other, so one bad byte cannot switch
        the cap off. Best-effort — a failed compaction never fails the mood
        write."""
        try:
            with self.mood_history_path.open("rb") as f:
                lines = f.readlines()
            if len(lines) <= MOOD_HISTORY_MAX_ROWS:
                return
            tail = lines[-MOOD_HISTORY_KEEP_ROWS:]
            tmp = self.mood_history_path.with_suffix(".jsonl.tmp")
            with tmp.open("wb") as f:
                f.writelines(tail)
            tmp.replace(self.mood_history_path)
            logger.info(
                "mood history compacted to newest %d records", len(tail),
            )
        except Exception as e:
            logger.warning("mood history compaction failed: %s", e)

    def _open_boredom(self, since: Optional[float] = None) -> int:
        """The highest boredom level the journal holds a row for in the quiet
        that is still open — i.e. since the last "the quiet ended" row, and,
        when ``since`` (a timestamp) is given, not before the quiet began.
        0 when there is none. Other rows in between (a thumb that came
        without a turn, a wearing off) do not close a quiet."""
        level = 0
        for row in reversed(self.mood_history(limit=_BORED_SCAN_ROWS)):
            kind = row.get("kind")
            if kind == M.KIND_QUIET_END:
                break
            if kind != M.KIND_QUIET:
                continue
            at = M.parse_iso_utc(str(row.get("at") or ""))
            if since is not None and (at is None or at.timestamp() < since - _SAME_QUIET_S):
                break
            level = max(level, int(row.get("intensity") or 0))
        return level

    def _bored_since(self, level: int, start: datetime.datetime,
                     now: Optional[datetime.datetime]) -> datetime.datetime:
        """Since when a reader sees boredom at ``level`` in the quiet that
        began at ``start``: when the quiet crossed that level's threshold —
        or, if a mood was still alive then (a thumb that came without a
        turn), when that mood wore off. (Both lie behind ``now``: the level
        is only read once its threshold is crossed and the mood is neutral.)
        The same moment dates the journal row, so a bored row never precedes
        the "it wore off" row of the mood it followed."""
        at = start + datetime.timedelta(seconds=M.BOREDOM_THRESHOLDS[level])
        stored = self._state.mood
        if stored is not None:
            neutral_at = M.parse_iso_utc(stored.set_at if stored.intensity == 0
                                         else self._wore_off_at(stored, 0, now))
            if neutral_at is not None and neutral_at > at:
                at = neutral_at
        return at.replace(microsecond=0)

    def _note_bored(self, level: int, start: datetime.datetime,
                    now: Optional[datetime.datetime]) -> None:
        """A read saw boredom at ``level`` in the quiet that began at
        ``start``. FIRST, whenever the stored mood is not neutral, it is
        settled — a mood that wore off unseen gets its "it wore off" row
        before any bored row, whatever is remembered about this quiet. THEN
        the journal row of each level this quiet has not had one for is
        written ("bored 1", "bored 2"), dated by ``_bored_since`` — and
        nothing when it has them (a poller must not add a row per poll).
        What the quiet already has is read off the journal
        (``_open_boredom``); the answer is remembered in memory. Rows are
        appended in time order. Never raises: a read is a read."""
        try:
            with self._lock:
                stored = self._state.mood
                if stored is not None and stored.intensity > 0:
                    self._settle_locked(now)
                seen = self._bored_seen
                if seen is not None and abs(seen[0] - start.timestamp()) < _SAME_QUIET_S:
                    logged = seen[1]
                else:
                    logged = self._open_boredom(start.timestamp())
                for k in range(logged + 1, level + 1):
                    cur = self._state.mood
                    row = Mood(p=(cur.p if cur is not None else 0.0), e=(cur.e if cur is not None else 0.0),
                               label=M.BORED, intensity=k, set_at=_iso(self._bored_since(k, start, now)),
                               cause=M.quiet_cause(k))
                    self._append_history_locked(
                        row, M.KIND_QUIET,
                        prev=(M.NEUTRAL if k == 1 else f"{M.BORED}, {k - 1} of 3"))
                self._bored_seen = (start.timestamp(), max(logged, level))
        except Exception as e:  # noqa: BLE001
            logger.debug("boredom not journalled: %s: %s", type(e).__name__, e)

    # -----------------------------------------------------------------
    # Write API — the mood
    # -----------------------------------------------------------------

    def _store_locked(self, m: Mood) -> Mood:
        m.p, m.e = round(m.p, 4), round(m.e, 4)
        self._state.mood = m          # one reference swap: a reader never sees it torn
        self._flush()
        return m

    def _settle_locked(self, now: Optional[datetime.datetime]) -> Mood:
        """Bring the stored pair to ``now`` (decay) and write it down when
        that crossed a boundary. Returns the mood as it stands (stored or
        not). Caller holds the lock."""
        stored = self._state.mood
        p, e, label, n, novelty = self._evolve(stored, now)
        base = stored if stored is not None else Mood()
        cur = Mood(
            p=p, e=e, updated_at=_iso(now), label=label, intensity=n, novelty=novelty,
            source=base.source, set_at=base.set_at, cause=base.cause,
            recent=list(base.recent), manner=base.manner,
            manner_intensity=base.manner_intensity, manner_set_at=base.manner_set_at,
        )
        if (label, n) == (base.label, base.intensity):
            return cur                       # nothing to write: decay is a function of time
        cur.cause = DECAY_CAUSE
        # a mood that wore off is dated when it did, not when it was noticed
        cur.set_at = self._wore_off_at(stored, n, now)
        # an owner-set mood that only steps down is still the owner's
        if not (base.source == M.SOURCE_OWNER and n and label == base.label):
            cur.source = M.SOURCE_AUTO
        self._store_locked(cur)
        self._append_history_locked(cur, M.KIND_DECAY, prev=self._describe(base))
        return cur

    def settle(self, *, now: Optional[datetime.datetime] = None) -> MoodView:
        """Write down what time did to the mood (a step down, a wearing
        off). Boredom is never settled: it is not in the pair."""
        with self._lock:
            return self._view_of(self._settle_locked(now))

    def owner_arrived(self, *, now: Optional[datetime.datetime] = None) -> None:
        """An owner's request has arrived: the quiet, if there was one, is
        over. Writes down what time did to the mood, adds ONE "the quiet
        ended" row when the journal holds a boredom of this quiet, and stamps
        the owner's presence. The pair is not touched by the quiet."""
        with self._lock:
            cur = self._settle_locked(now)
            was = self._open_boredom()
            if was > 0:
                row = Mood(p=cur.p, e=cur.e, label=cur.label, intensity=cur.intensity,
                           set_at=_iso(now), cause=M.QUIET_ENDED_CAUSE)
                self._append_history_locked(row, M.KIND_QUIET_END, prev=f"{M.BORED}, {was} of 3")
            self.mark_owner_turn(now=now)

    def mark_owner_turn(self, *, now: Optional[datetime.datetime] = None) -> None:
        """Stamp that the owner is here (a request of his arrived, or ended).
        The next quiet is measured from this."""
        with self._lock:
            self._state.last_owner_turn_at = _iso(now)
            self._bored_seen = None
            self._flush()

    def nudge(self, dp: float, de: float, cause: str, kind: str, *, novelty: bool = False,
              now: Optional[datetime.datetime] = None) -> Tuple[MoodView, MoodView]:
        """THE function every event goes through: move the pair by
        ``(dp, de)``, clamp it, record it. Returns ``(before, after)``."""
        with self._lock:
            base = self._settle_locked(now)
            before = self._view_of(base)
            try:
                dp, de = float(dp), float(de)
            except (TypeError, ValueError):
                return before, before
            if dp == 0.0 and de == 0.0:
                return before, before
            at = _iso(now)
            p, e = M.clamp(base.p + dp), M.clamp(base.e + de)
            # the novelty tag belongs to the move that set it: a later event
            # that is not itself a research event clears it (review r2 —
            # research, then three thumbs, read "curious, 3 of 3: the owner
            # gave a thumbs-up")
            label, n, nov = M.read_off(p, e, held_label=base.label, held_intensity=base.intensity,
                                       novelty=bool(novelty))
            changed = (label, n) != (base.label, base.intensity)
            cause = _cap_text(str(cause or ""), _MOOD_EVIDENCE_MAX_CHARS)
            new = Mood(
                p=p, e=e, updated_at=at, label=label, intensity=n, novelty=nov,
                source=(M.SOURCE_AUTO if changed else base.source),
                set_at=(at if changed else base.set_at),
                cause=(cause if changed else base.cause),
                recent=(list(base.recent) + [{"at": at, "cause": cause, "kind": str(kind or "")}]
                        )[-MOOD_RECENT_CAUSES:],
                manner=base.manner, manner_intensity=base.manner_intensity,
                manner_set_at=base.manner_set_at,
            )
            self._store_locked(new)
            if changed:
                self._append_history_locked(new, str(kind or ""), prev=before.describe())
            return before, self._view_of(new)

    @staticmethod
    def _describe(m: Mood) -> str:
        return M.NEUTRAL if not m.intensity else f"{m.label}, {m.intensity} of 3"

    @staticmethod
    def _view_of(m: Mood) -> MoodView:
        return MoodView(
            label=m.label, intensity=m.intensity, p=round(m.p, 4), e=round(m.e, 4),
            manner=m.manner, manner_intensity=(m.manner_intensity if m.manner else 0),
            source=(m.source if m.intensity else M.SOURCE_AUTO), set_at=m.set_at,
            cause=(m.cause if m.intensity else ""),
            manner_set_at=(m.manner_set_at if m.manner else ""),
            recent=tuple(dict(r) for r in m.recent),
        )

    def set_owner_mood(self, label: str, direction: Tuple[float, float], intensity: int, *,
                       note: str = "", now: Optional[datetime.datetime] = None,
                       ) -> Tuple[MoodView, MoodView]:
        """The owner said how to be: put the pair on ``label``'s anchor at
        ``intensity``. From there it decays and reacts like any mood.
        Returns ``(before, after)``."""
        with self._lock:
            base = self._settle_locked(now)
            before = self._view_of(base)
            n = max(1, min(3, int(intensity)))
            p, e = M.anchor_for(direction, n)
            at = _iso(now)
            changed = (label, n) != (base.label, base.intensity)
            new = Mood(
                p=p, e=e, updated_at=at, label=label, intensity=n,
                novelty=(label == M.CURIOUS), source=M.SOURCE_OWNER,
                set_at=(at if changed else (base.set_at or at)), cause=OWNER_CAUSE,
                recent=(list(base.recent) + [{"at": at, "cause": OWNER_CAUSE, "kind": M.KIND_OWNER}]
                        )[-MOOD_RECENT_CAUSES:],
                manner=base.manner, manner_intensity=base.manner_intensity,
                manner_set_at=base.manner_set_at,
            )
            self._store_locked(new)
            if changed:
                self._append_history_locked(new, M.KIND_OWNER,
                                            _cap_text(note, _MOOD_EVIDENCE_MAX_CHARS),
                                            prev=before.describe())
            return before, self._view_of(new)

    def reset_mood(self, *, note: str = "", now: Optional[datetime.datetime] = None,
                   ) -> Tuple[MoodView, MoodView]:
        """The owner said "neutral": the pair goes to 0, 0 AND the manner is
        cleared. Returns ``(before, after)``."""
        with self._lock:
            base = self._settle_locked(now)
            before = self._view_of(base)
            at = _iso(now)
            was_mood = base.intensity > 0
            if not was_mood and not base.manner and base.p == 0.0 and base.e == 0.0:
                return before, before            # nothing on file to reset
            new = Mood(
                p=0.0, e=0.0, updated_at=at, label=M.NEUTRAL, intensity=0, novelty=False,
                source=M.SOURCE_AUTO, set_at=(at if was_mood else base.set_at),
                cause=(OWNER_CAUSE if was_mood else base.cause),
                recent=(list(base.recent) + [{"at": at, "cause": OWNER_CAUSE, "kind": M.KIND_OWNER}]
                        )[-MOOD_RECENT_CAUSES:],
            )
            self._store_locked(new)
            if was_mood:
                self._append_history_locked(new, M.KIND_OWNER,
                                            _cap_text(note, _MOOD_EVIDENCE_MAX_CHARS),
                                            prev=before.describe())
            return before, self._view_of(new)

    def set_manner(self, manner: str, intensity: int, *,
                   now: Optional[datetime.datetime] = None) -> Tuple[MoodView, MoodView]:
        """The owner set a way of talking ("" clears it). It does not decay
        and stays until the owner changes or clears it. The mood itself is
        left as it is. Returns ``(before, after)``."""
        with self._lock:
            base = self._settle_locked(now)
            before = self._view_of(base)
            man = M.resolve_manner(manner)
            if not man and not base.manner:
                return before, before
            new = Mood(
                p=base.p, e=base.e, updated_at=base.updated_at or _iso(now), label=base.label,
                intensity=base.intensity, novelty=base.novelty, source=base.source,
                set_at=base.set_at, cause=base.cause, recent=list(base.recent),
                manner=man,
                manner_intensity=(max(1, min(3, int(intensity))) if man else 0),
                manner_set_at=(_iso(now) if man else ""),
            )
            if man and man == base.manner and new.manner_intensity == base.manner_intensity:
                new.manner_set_at = base.manner_set_at or new.manner_set_at
            self._store_locked(new)
            return before, self._view_of(new)

    def mood_history(self, *, limit: int = 20) -> List[Dict[str, object]]:
        """The newest ``limit`` rows of the mood journal, oldest first. Rows
        written before §4TC (a label with no pair) are skipped. Empty when
        the file is missing or ``limit`` <= 0."""
        if limit <= 0 or not self.mood_history_path.exists():
            return []
        try:
            # errors="replace": one byte that is not UTF-8 costs its own line
            # (it no longer parses, or parses with a replaced character) —
            # not the whole journal
            lines = self.mood_history_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            logger.warning("mood history read failed: %s", e)
            return []
        out: List[Dict[str, object]] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except (ValueError, RecursionError):
                continue                       # not JSON (a bad byte, a torn line, nesting too deep)
            if not isinstance(d, dict) or "at" not in d or "p" not in d:
                continue                       # a pre-§4TC row
            label = str(d.get("label") or "")
            try:
                n = max(0, min(3, int(d.get("intensity") or 0)))
            except (TypeError, ValueError):
                n = 0
            if label not in M.LABELS and label != M.NEUTRAL:
                continue
            out.append({"label": label, "intensity": n,
                        "p": M.clamp(d.get("p")), "e": M.clamp(d.get("e")),
                        "at": str(d.get("at") or ""), "cause": str(d.get("cause") or ""),
                        "kind": str(d.get("kind") or ""), "note": str(d.get("note") or "")})
        return out[-limit:]

    def mood_timeline(self, *, now: Optional[datetime.datetime] = None, tz=None,
                      limit: int = 200) -> List[Dict[str, object]]:
        """Today's rows of the journal (the local day of ``now``), oldest
        first, each with its local time as ``"hhmm"``."""
        if now is None:
            now = datetime.datetime.now(datetime.timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=datetime.timezone.utc)
        today = now.astimezone(tz).date()
        out: List[Dict[str, object]] = []
        for row in self.mood_history(limit=limit):
            at = M.parse_iso_utc(str(row.get("at") or ""))
            if at is None:
                continue
            local = at.astimezone(tz)
            if local.date() != today:
                continue
            out.append({**row, "hhmm": local.strftime("%H:%M")})
        return out

    def touch_session(self) -> None:
        with self._lock:
            self._state.last_session_at = _utcnow_iso()
            self._flush()

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def format_as_prefix(self, *, max_chars: int = 1200,
                         now: Optional[datetime.datetime] = None,
                         include_mood: bool = True) -> str:
        """Render the current state as first-person text (the wake-up text
        the functional harness renders, and the head of the state block of
        the diary's prompt).

        Empty when there's nothing worth surfacing. The mood line is the
        label and its intensity only — no time, no cause. A neutral mood has
        no line. ``include_mood=False`` leaves the mood line out: the diary
        reads the mood from the JOURNAL, not from what a read shows at this
        instant (a boredom level, a step of decay), because its prompt's
        hash is its regenerate skip-guard."""
        view = self.mood(now=now, quiet_s=(None if include_mood else 0.0))
        if ((view.neutral or not include_mood) and not view.manner
                and not self._state.last_session_at):
            return ""

        lines: List[str] = []
        if self._state.last_session_at:
            lines.append(f"I was last active on {self._state.last_session_at}.")
        if include_mood and not view.neutral:
            lines.append(f"My mood: {view.describe()}.")
        if view.manner:
            lines.append(f"My manner (set by the owner): {view.manner}, "
                         f"{view.manner_intensity} of 3.")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        return text
