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
import math
import os
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import mood as M
from .schema import MOOD_RECENT_CAUSES, Mood, SelfState, _utcnow_iso, read_jsonl_rows

logger = logging.getLogger("GhostSelfhood")


STATE_FILENAME = "state.json"
MOOD_HISTORY_FILENAME = "mood.history.jsonl"

#: Bounded growth for the mood journal: once it holds more than
#: ``MOOD_HISTORY_MAX_ROWS`` rows the oldest are dropped, down to
#: ``MOOD_HISTORY_KEEP_ROWS``.
MOOD_HISTORY_MAX_ROWS = 1200
MOOD_HISTORY_KEEP_ROWS = 1000
#: §4TX: every event, applied or held, one row — what moved the pair and what was refused, not only the
#: changes of label the history keeps (review: "why is it irritable" could not be answered from the files)
MOOD_EVENTS_FILENAME = "mood.events.jsonl"
MOOD_EVENTS_MAX_ROWS = 3000
MOOD_EVENTS_KEEP_ROWS = 2500

# Input caps for the texts a mood row carries.
_MOOD_LABEL_MAX_CHARS = 64
_MOOD_EVIDENCE_MAX_CHARS = 300

#: the cause a row carries when the owner set the mood in chat
OWNER_CAUSE = "the owner said so (set_mood)"
#: … and when a mood stepped down or ended with nothing happening
DECAY_CAUSE = "it wore off"
#: §4UD (G2): … and when the owner took the last set mood back
UNDO_CAUSE = "the owner took the last set mood back (set_mood undo)"
#: §4UD (F9): the row that closes a boredom when a job the owner started did work
JOB_QUIET_ENDED_CAUSE = "the quiet ended (a background job of the owner's did work)"

#: §4UD (F12): the one copy kept of a state file that could not be read (overwritten by the next one)
STATE_BAD_SUFFIX = ".bad"

#: the stored stamps and how far ahead of the clock each may lie (``mood.sane_future``)
_STAMP_FIELDS = (("updated_at", M.STAMP_AHEAD_MAX_S), ("set_at", M.STAMP_AHEAD_MAX_S),
                 ("owner_at", M.STAMP_AHEAD_MAX_S), ("manner_set_at", M.STAMP_AHEAD_MAX_S),
                 ("held_until", M.HOLD_AHEAD_MAX_S))

#: the stored numbers keep this many decimals: the pair without the owner's tone is worked out from them
#: (``p − up + down``), and with 4 decimals every write moved it by up to 0.0001 against a run that never read a tone
_PAIR_DP = 12
#: a tone share under this, in a neutral mood, is dropped (with what it held in the pair)
_SHARE_MIN = 5e-5

#: (pleasantness up, energy up, pleasantness down, energy down) — the caps of the four tone shares
_SIDE_CAPS = (M.SENTIMENT_SHARE_MAX, M.SENTIMENT_SHARE_MAX_E, M.SENTIMENT_SHARE_MAX, M.SENTIMENT_SHARE_MAX_E)
_NO_SIDES = (0.0, 0.0, 0.0, 0.0)
Sides = Tuple[float, float, float, float]


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


def _shown(dev: float, r: float) -> float:
    """§4UA: a number as readers see it — rest + deviation, clamped. At rest
    0 exactly the old ``round(dev, 4)`` (review r2: ``clamp(-0.0 + 0.0)``
    wrote ``0.0`` where the journal had ``-0.0``)."""
    return round(dev, 4) if r == 0.0 else round(M.clamp(dev + r), 4)


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
    #: §4TX: an owner-set mood is held (no decay, no event moves it) until this stamp; "" when not held
    held_until: str = ""
    #: the manner's steps still ahead, ``(on-disk stamp, level from then)``,
    #: the last one the end of its day (§4TS); () with no manner
    manner_steps: Tuple[Tuple[str, int], ...] = ()
    #: §4UA review r2: what an event really moved the STORED pair by (the
    #: deviation; ``p``/``e`` above are rest + deviation, clamped) — set on
    #: ``nudge``'s after-view only
    moved: Tuple[float, float] = field(default=(0.0, 0.0), compare=False, repr=False)
    #: §4UD (F1): the part of ``moved`` that is NOT the owner's message tone — what the pair without the tone
    #: moved by. The reaction ledger books this for the reply (a run that read no tone books the same)
    base_moved: Tuple[float, float] = field(default=(0.0, 0.0), compare=False, repr=False)
    #: §4UD: what the event changed the four tone shares by (up p, up e, down p, down e) …
    toned: Sides = field(default=_NO_SIDES, compare=False, repr=False)
    #: … and what the sink kept OUT of them (the cap, the pair's ±1 limit), per share, ≥ 0
    refused: Sides = field(default=_NO_SIDES, compare=False, repr=False)

    @property
    def neutral(self) -> bool:
        return self.intensity == 0

    def tone(self, *, remarked: bool = False, audience: str = "owner") -> str:
        """The one line a reply's prompt carries ("" when there is nothing
        to say)."""
        return M.tone_line(self.label, self.intensity, self.manner, self.manner_intensity,
                           remarked=remarked, audience=audience)

    @property
    def held(self) -> bool:
        return bool(self.held_until)

    def signature(self) -> str:
        """What a listener hears — the label, the level, the manner: the same
        signature twice is the same tone line's mood (§4TX, one remark per
        conversation)."""
        return f"{self.label}:{self.intensity}:{self.manner}:{self.manner_intensity}"

    def describe(self) -> str:
        """"irritable, 2 of 3" / "neutral"."""
        return M.NEUTRAL if self.neutral else f"{self.label}, {self.intensity} of 3"


def _others_axis(v: float, up: float, dn: float) -> float:
    """§4UD (F1): one number of the pair as OTHER people hear it — without the
    tone share that lies on its side, only toward zero, never past it. With
    ``up`` and ``dn`` the two shares of the axis, the pair without any tone is
    ``v − up + dn``: a positive ``v`` gives up ``up`` and keeps the pull of
    ``dn`` (so it reads ``base − dn``, at most the tone-free value), a
    negative one the other way round. (A first cut also took a 1e-6 "epsilon"
    off the result against the stored numbers' rounding; review r2 found no
    test for it, and the case it was for — the pair exactly on a threshold —
    is settled where levels are read, ``mood.LEVEL_EPS``.)"""
    if v > 0:
        return max(0.0, v - max(0.0, up))
    if v < 0:
        return min(0.0, v + max(0.0, dn))
    return v


def _sides_of(m: Optional[Mood]) -> Sides:
    if m is None:
        return _NO_SIDES
    return (m.private_up_p, m.private_up_e, m.private_dn_p, m.private_dn_e)


def _sides_kw(sides: Sides) -> Dict[str, float]:
    return dict(private_up_p=sides[0], private_up_e=sides[1], private_dn_p=sides[2], private_dn_e=sides[3])


def tone_sink(sides: Sides, delta: Optional[Sides], bp: float, be: float) -> Tuple[float, float, Sides, Sides]:
    """§4UD (F1): THE place the owner's tone shares are written — every path
    that books, withdraws or carries a tone goes through here, and nothing
    else writes them. ``sides`` are the four shares as they stand, ``delta``
    what the caller asks to change them by, ``(bp, be)`` the pair WITHOUT
    any tone after the event's own move. Each share ends in [0, its cap]
    (``mood.SENTIMENT_SHARE_MAX``; the energy's in proportion); the pair is
    ``base + up − down`` and stays in [-1, 1] — what does not fit there
    leaves the share, never the base. Returns ``(p, e, shares, refused)``:
    ``refused`` is what was asked for and kept out, per share.

    (§4UB: five review rounds found a ledger path around a cap placed before
    the write, so the cap moved to the write. Review F1 then found the write
    itself booking more than a tone had moved: the caller now passes the
    tone's own contribution and this function only ever narrows it.)"""
    d = delta or _NO_SIDES
    want = [max(0.0, float(sides[i])) + float(d[i]) for i in range(4)]
    got = [max(0.0, min(_SIDE_CAPS[i], want[i])) for i in range(4)]
    for base, up, dn in ((bp, 0, 2), (be, 1, 3)):
        v = base + got[up] - got[dn]
        if v > 1.0:
            got[up] = max(0.0, got[up] - (v - 1.0))
        elif v < -1.0:
            got[dn] = max(0.0, got[dn] - (-1.0 - v))
    refused = tuple(max(0.0, want[i] - got[i]) for i in range(4))
    return (M.clamp(bp + got[0] - got[2]), M.clamp(be + got[1] - got[3]),
            (got[0], got[1], got[2], got[3]), refused)   # type: ignore[return-value]


class SelfStateThread:
    """Single-file persisted self-state.

    Read-on-construct, write-on-every-mutation. Cheap (file is small)
    and means a crash-restart picks up exactly where the last successful
    write left off."""

    def __init__(self, root: Path, *, enabled: bool = True):
        self.root = Path(root)
        self.path = self.root / STATE_FILENAME
        self.mood_history_path = self.root / MOOD_HISTORY_FILENAME
        self.mood_events_path = self.root / MOOD_EVENTS_FILENAME
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        #: §4TX review r3: moved only when the owner REPLACES the pair (a set mood, a "neutral" that
        #: cleared one) — the reaction ledger's amounts belong to one pair (in memory, like the ledger)
        self.pair_epoch = 0
        #: §4UD (F12): "" when state.json was read (or there was none); else why it could not be
        #: ("unreadable: …") — the state then started empty, and a copy of the file is kept beside it
        self.load_problem: str = ""
        #: stamps already reported as lying too far ahead of the clock (logged once each)
        self._insane_logged: set = set()
        self._state: SelfState = self._read_or_empty()
        #: how long the owner has been quiet, in seconds (None = unknown) —
        #: attached by the process that owns the clock (``main.py``); this
        #: class only ever READS it
        self._quiet_fn: Optional[Callable[[], Optional[float]]] = None
        #: (start of the quiet as a timestamp, highest boredom level the
        #: journal has a row for) — what `_note_bored` last worked out
        self._bored_seen: Optional[Tuple[float, int]] = None
        #: §4UA: where the pair rests — attached by the process that owns the
        #: character (``main.py``); None = rest at (0, 0), as before
        self._rest_fn: Optional[Callable[[], Tuple[float, float]]] = None

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def _read_or_empty(self) -> SelfState:
        if not self.path.exists():
            return SelfState()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"not a JSON object ({type(data).__name__})")
            return SelfState.from_dict(data)
        except Exception as e:  # noqa: BLE001 — unreadable bytes, bad JSON, a wrong shape
            return self._unreadable(e)

    def _unreadable(self, e: BaseException) -> SelfState:
        """§4UD (F12): state.json is there and cannot be read. The mood, the
        manner, the owner's hold and the owner stamp start empty — SAID: one
        file itself is kept, renamed ``state.json.bad`` (the next write
        makes a new state.json), ``load_problem`` says why for the health line
        and ``introspect``, and the operator's log gets one loud line."""
        why = f"{type(e).__name__}: {e}"
        self.load_problem = _cap_text("unreadable: " + why, 160)
        kept = ""
        try:
            bad = self.path.with_name(self.path.name + STATE_BAD_SUFFIX)
            os.replace(self.path, bad)   # a rename: never follows a planted link
            kept = f"; the file is kept as {bad.name}"
        except OSError as ce:
            kept = f"; no copy could be kept ({type(ce).__name__})"
        logger.error("self-state unreadable (%s) — the mood, the manner and the owner's hold start EMPTY%s",
                     why, kept)
        return SelfState()

    def _flush(self) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # Write a sibling temp file, force it to disk, then rename: a
                # crash or a power cut leaves the old file or the new one,
                # never a renamed file whose bytes were still in the cache
                # (§4UD F12: that is the truncated state.json a start reads)
                tmp = self.path.with_suffix(".json.tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    f.write(json.dumps(self._state.to_dict(), ensure_ascii=False, indent=2))
                    f.flush()
                    os.fsync(f.fileno())
                tmp.replace(self.path)
                if self.load_problem and self._state.mood is not None:
                    self._good_again()
        except Exception as e:
            logger.warning("self-state flush failed: %s", e)

    def _stored(self, now: Optional[datetime.datetime] = None) -> Optional[Mood]:
        """§4UD (F3): THE read of the stored mood — every reader and writer in
        this class takes it from here. A stamp that lies further ahead of
        ``now`` than it can (``mood.sane_future``: a clock that was wrong
        when it was written, a hand edit) reads as absent, so what hangs on
        it has expired — the owner's hold, the manner, "the owner spoke after
        this message", and a pair nobody can date is worth nothing. Logged
        once per stamp.

        Review r2: decided ONCE. The first read that finds such a stamp
        blanks it in the stored mood and writes the file — expired per read,
        an owner's hold stamped 31 h ahead came BACK three hours later (the
        stamp had drifted under the 28 h), or was gone for good if an event
        happened to be written in between. (Not at load: a load has no clock
        of its own; the first read follows within seconds.)"""
        m = self._state.mood
        sane = self._sane(m, now)
        if sane is not m:
            with self._lock:
                if self._state.mood is m:          # nobody replaced it meanwhile
                    self._state.mood = sane
                    self._flush()
        return sane

    def _sane(self, m: Optional[Mood], now: Optional[datetime.datetime]) -> Optional[Mood]:
        """``m`` with every stamp that lies too far ahead of ``now`` blanked (a copy; ``m`` itself when none does)."""
        if m is None:
            return None
        when = _aware(now)
        gone = {}
        for name, ahead in _STAMP_FIELDS:
            v = getattr(m, name)
            if v and M.parse_iso_utc(v) is not None and M.sane_future(v, now=when, ahead_s=ahead) is None:
                gone[name] = ""
                self._say_insane(name, v)
        return replace(m, **gone) if gone else m

    def _say_insane(self, name: str, value: str) -> None:
        if (name, value) in self._insane_logged:
            return
        if len(self._insane_logged) > 64:
            self._insane_logged.clear()
        self._insane_logged.add((name, value))
        logger.warning("mood: stored %s=%s lies too far ahead of the clock — read as expired", name, value)
        try:
            from ..utils.logging import Icons, pretty_log
            pretty_log("Selfhood", f"stored {name} {value} is ahead of the clock — expired",
                       icon=Icons.SELF_STATE, level="WARNING")
        except Exception:  # noqa: BLE001
            pass

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

    def attach_rest(self, rest_fn: Optional[Callable[[], Tuple[float, float]]]) -> None:
        """§4UA: give the thread its resting point — the character's (from its
        dials, ``character.rest_pair``, §4UH-e2e). The thread only READS it, on
        every read of the pair."""
        self._rest_fn = rest_fn if callable(rest_fn) else None

    def rest(self) -> Tuple[float, float]:
        """§4UA: the resting point now, clamped (``M.rest_clamp``); (0, 0)
        when none is attached or it cannot be read — never raises."""
        fn = self._rest_fn
        if fn is None:
            return 0.0, 0.0
        try:
            return M.rest_clamp(fn())
        except Exception:  # noqa: BLE001 — the rest never costs the mood
            return 0.0, 0.0

    def quiet_seconds(self, *, now: Optional[datetime.datetime] = None) -> Optional[float]:
        """How long the owner has been quiet, or None when nobody knows.
        Measured from the PERSISTED stamp of the last owner turn, so a
        restart does not change it; the in-memory clock of the process is
        asked only when there is no stamp yet."""
        # §4UD (F9): from the LATER of the owner's last turn and the last time a background job the owner
        # started did work (the face read "bored — quiet for 6 hours" while the agent worked for the owner
        # all day). A stamp too far ahead of the clock is not read (F3)
        when = _aware(now)
        ages = [max(0.0, (when - at).total_seconds())
                for at in (M.sane_future(ts, now=when) for ts in (self._state.last_owner_turn_at,
                                                                  self._state.last_owner_job_at))
                if at is not None]
        if ages:
            return min(ages)
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
        value of the pair.) §4UA: ``p, e`` are the DEVIATION from the
        resting point (what is stored); label and intensity are read off it
        alone, as before — the rest only moves the numbers a reader sees."""
        if stored is None:
            return 0.0, 0.0, M.NEUTRAL, 0, False
        # an age that cannot be established is not "now": the pair is
        # worth nothing (a mood must never be presented as current on a
        # timestamp nobody can read)
        age = SelfStateThread._age(stored, now)
        if age is None:
            p = e = 0.0
        else:
            p, e = M.decay(stored.p, age), M.decay(stored.e, age)
        label, n, novelty = M.read_off(p, e, held_label=stored.label, held_intensity=stored.intensity,
                                       novelty=stored.novelty)
        return p, e, label, n, novelty

    @staticmethod
    def _held_until(stored: Optional[Mood], now: Optional[datetime.datetime]) -> Optional[datetime.datetime]:
        """§4TX: the end of the owner's hold when it is still on at ``now``, else None."""
        if stored is None or not stored.held_until:
            return None
        end = M.parse_iso_utc(stored.held_until)
        return end if (end is not None and _aware(now) < end) else None

    @staticmethod
    def _age(stored: Mood, now: Optional[datetime.datetime]) -> Optional[float]:
        """How long the stored pair has been decaying at ``now`` (None = unknown).
        §4TX: no time at all while the owner's hold is on; after it, the time
        since the hold ended (or since the pair was written, if later)."""
        age = M.age_seconds(stored.updated_at, now=now)
        if age is None or not stored.held_until:
            return age
        end = M.parse_iso_utc(stored.held_until)
        if end is None:
            return age
        return min(age, max(0.0, (_aware(now) - end).total_seconds()))

    @staticmethod
    def _sides_now(stored: Optional[Mood], now: Optional[datetime.datetime]) -> Sides:
        """§4UD: the four tone shares at ``now`` (up p, up e, down p, down e) — they decay with the pair."""
        sides = _sides_of(stored)
        if stored is None or not any(sides):
            return _NO_SIDES
        age = SelfStateThread._age(stored, now)
        if age is None:
            return _NO_SIDES
        return tuple(M.decay(v, age) for v in sides)   # type: ignore[return-value]

    @staticmethod
    def _private_now(stored: Optional[Mood], now: Optional[datetime.datetime]) -> Tuple[float, float]:
        """§4TX: the owner's-sentiment part of the pair at ``now``, NET (up − down) per axis."""
        up_p, up_e, dn_p, dn_e = SelfStateThread._sides_now(stored, now)
        return up_p - dn_p, up_e - dn_e

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
            held = M.parse_iso_utc(stored.held_until) if stored.held_until else None
            if held is not None and held > start:
                start = held                 # §4TX: it began to decay when the owner's hold ended
            end = _aware(now)
            at = start + datetime.timedelta(
                seconds=M.decay_step_s(max(abs(stored.p), abs(stored.e)), n))
            return _iso(min(max(at, start), end).replace(microsecond=0))
        except Exception:  # noqa: BLE001 — a stamp never costs the mood
            return _iso(now)

    def mood(self, *, now: Optional[datetime.datetime] = None, quiet_s=None,
             for_reply: bool = False, audience: str = "owner") -> MoodView:
        """The mood NOW. A read never MOVES the mood, whoever asks: no event
        is applied and nothing of a boredom is stored. What a read can do,
        and only while it sees a boredom (``_note_bored``): settle a mood
        that wore off unseen — the stored pair is rewritten to its decayed
        value, the same value any reader computes from it, and the "it wore
        off" row is added — and add the journal row of a boredom level it is
        the first to see (at most one per level per quiet).

        §4TX: ``for_reply`` — the mood a reply's tone line is made of: no
        boredom (a quiet is not something to take out on whoever writes
        next; the face and introspect still show it). ``audience="others"``
        — the mood a member or a public channel hears: the part the owner's
        own sentiment put in the pair is left out, and so is the cause."""
        stored = self._stored(now)
        rest = self.rest()                       # §4UA: shown numbers only
        p, e, label, n, _nov = self._evolve(stored, now)
        held = self._held_until(stored, now)
        base = dict(
            p=_shown(p, rest[0]), e=_shown(e, rest[1]),
            recent=tuple(dict(r) for r in (stored.recent if stored is not None else ())),
            held_until=(_iso(held) if held is not None else ""),
            **self._manner_now(stored, now),
        )
        if audience in ("others", "public") and stored is not None:
            # review r2: the owner's share comes out only TOWARD zero (others never hear a mood the agent
            # does not have, nor a stronger one), and others never get the cause or the recent events
            up_p, up_e, dn_p, dn_e = self._sides_now(stored, now)
            # review r3: with a share taken out, a fresh read — the owner's level, held by the margin, is
            # not carried over to others
            # review r6: ANY share — the owner's held level may exist only because of the owner's share (r4/r5 kept
            # it for a small share, and others heard it rise while the mood decayed). Others may step down a
            # little before the owner does; they never hear more than the owner does.
            # §4UD (F1): … and a share that has since gone — the held level may still be the one a tone
            # helped reach (``tone_touched``: kept while the stored level is one held by the margin)
            shared = bool(stored.tone_touched or up_p or up_e or dn_p or dn_e)
            q_p, q_e = _others_axis(p, up_p, dn_p), _others_axis(e, up_e, dn_e)
            o_label, o_n, _ = M.read_off(q_p, q_e, held_label=("" if shared else stored.label),
                                         held_intensity=(0 if shared else stored.intensity),
                                         novelty=stored.novelty)
            base.update(p=_shown(q_p, rest[0]), e=_shown(q_e, rest[1]), recent=())
            if o_n == 0:
                return MoodView(**base)
            same = (o_label, o_n) == (label, n) and not shared
            # review r5: "since" is the owner's own moment when the owner's share was taken out — not shown then
            return MoodView(label=o_label, intensity=o_n,
                            source=(stored.source if same and o_label == stored.label else M.SOURCE_AUTO),
                            set_at=(stored.set_at if same else ""), cause="", **base)
        if n == 0 and not for_reply:
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
            rp, re_ = self.rest()                # §4UA: the pair as read (rest + deviation)
            rec = {"label": m.label, "intensity": m.intensity,
                   "p": round(m.p if rp == 0.0 else M.clamp(m.p + rp), 3),
                   "e": round(m.e if re_ == 0.0 else M.clamp(m.e + re_), 3),
                   "at": m.set_at, "cause": m.cause, "kind": kind}
            if note:
                rec["note"] = note
            with open_append(self.mood_history_path) as f:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
            self._maybe_compact_mood_history_locked()
        except Exception as e:
            logger.warning("mood history append failed: %s", e)

    def _append_event_locked(self, *, kind: str, cause: str, dp: float, de: float, held: str = "",
                             after: str = "", now: Optional[datetime.datetime] = None) -> None:
        """§4TX: one row of ``mood.events.jsonl`` — every event, applied or
        held (``held`` says why it was not applied). Never the words of a
        message: the cause is the event's own sentence. Best-effort."""
        if not self.enabled:
            return
        try:
            self.mood_events_path.parent.mkdir(parents=True, exist_ok=True)
            rec = {"at": _iso(now), "kind": str(kind or "")[:24],
                   "dp": round(float(dp), 4), "de": round(float(de), 4),
                   "cause": _cap_text(str(cause or ""), _MOOD_EVIDENCE_MAX_CHARS)}
            if held:
                rec["held"] = held
            if after:
                rec["after"] = after
            with open_append(self.mood_events_path) as f:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
            self._compact_locked(self.mood_events_path, MOOD_EVENTS_MAX_ROWS, MOOD_EVENTS_KEEP_ROWS)
        except Exception as e:  # noqa: BLE001
            logger.warning("mood events append failed: %s", e)

    def private_share(self, *, now: Optional[datetime.datetime] = None) -> Tuple[float, float]:
        """§4UB: what the owner's message sentiment has in the pair now (decayed, stored — a restart keeps it;
        the reaction ledger takes it back out when a stronger reaction replaces a tone)."""
        with self._lock:
            return self._private_now(self._stored(now), now)

    def private_sides(self, *, now: Optional[datetime.datetime] = None) -> Sides:
        """§4UD: the four tone shares now — (up p, up e, down p, down e), each ≥ 0, decayed."""
        with self._lock:
            return self._sides_now(self._stored(now), now)

    def held_until_now(self, *, now: Optional[datetime.datetime] = None) -> Optional[datetime.datetime]:
        """The end of the owner's hold while it is on at ``now``, else None."""
        return self._held_until(self._stored(now), now)

    def note_held(self, kind: str, cause: str, dp: float, de: float, why: str, *,
                  now: Optional[datetime.datetime] = None) -> None:
        """§4TX: an event that was not applied, with why — one events-log row."""
        with self._lock:
            self._append_event_locked(kind=kind, cause=cause, dp=dp, de=de, held=why, now=now)

    def mood_events(self, *, limit: int = 50) -> List[Dict[str, object]]:
        """The newest ``limit`` rows of ``mood.events.jsonl``, oldest first."""
        if limit <= 0:
            return []
        return [d for d in read_jsonl_rows(self.mood_events_path, what="mood events log") if "at" in d][-limit:]

    def held_back(self, *, now: Optional[datetime.datetime] = None, tz=None) -> List[Tuple[str, int]]:
        """§4UD (F8): the events of this mood day that were NOT applied, as
        ``[(why, how many)]``, the most frequent reason first — read off
        ``mood.events.jsonl`` (every event, applied or held, is a row there;
        until now nothing read it, so "why didn't my thumbs-up count?" had no
        answer). The reasons are the system's own sentences, never a
        message's words."""
        today = M.mood_day(now, tz=tz)
        counts: Dict[str, int] = {}
        for row in self.mood_events(limit=MOOD_EVENTS_MAX_ROWS):
            why = str(row.get("held") or "").strip()
            at = M.parse_iso_utc(str(row.get("at") or ""))
            if why and at is not None and M.mood_day(at, tz=tz) == today:
                counts[why] = counts.get(why, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    def _maybe_compact_mood_history_locked(self) -> None:
        self._compact_locked(self.mood_history_path, MOOD_HISTORY_MAX_ROWS, MOOD_HISTORY_KEEP_ROWS)

    def _compact_locked(self, path: Path, max_rows: int, keep_rows: int) -> None:
        """Keep the newest ``MOOD_HISTORY_KEEP_ROWS`` rows once the journal
        holds more than ``MOOD_HISTORY_MAX_ROWS``. Caller holds the lock.
        Works on BYTES: a line that is not UTF-8 (a torn write, a hand edit)
        is counted and carried like any other, so one bad byte cannot switch
        the cap off. Best-effort — a failed compaction never fails the mood
        write."""
        try:
            with path.open("rb") as f:
                lines = f.readlines()
            if len(lines) <= max_rows:
                return
            tail = lines[-keep_rows:]
            tmp = path.with_suffix(".jsonl.tmp")
            with tmp.open("wb") as f:
                f.writelines(tail)
            tmp.replace(path)
            logger.info("%s compacted to newest %d records", path.name, len(tail))
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
        stored = self._stored(now)
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
                stored = self._stored(now)
                if stored is not None and stored.intensity > 0:
                    self._settle_locked(now)
                seen = self._bored_seen
                if seen is not None and abs(seen[0] - start.timestamp()) < _SAME_QUIET_S:
                    logged = seen[1]
                else:
                    logged = self._open_boredom(start.timestamp())
                for k in range(logged + 1, level + 1):
                    cur = self._stored(now)
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
        """THE write of the stored mood. §4UD: a neutral pair whose tone
        shares have all but gone drops them, with what they held (the pair is
        then exactly the pair without any tone). ``tone_touched`` is set
        while a share stands; with none left it stays only while the stored
        level is one the pair would not reach read afresh (a level held by
        the margin — it may be the level a tone helped reach), so a pair that
        holds no tone and no such level reads for other people as it does
        for the owner."""
        sides = _sides_of(m)
        if m.intensity == 0 and any(sides) and max(sides) < _SHARE_MIN:
            m.p, m.e = M.clamp(m.p - sides[0] + sides[2]), M.clamp(m.e - sides[1] + sides[3])
            sides = _NO_SIDES
        m.private_up_p, m.private_up_e, m.private_dn_p, m.private_dn_e = (round(v, _PAIR_DP) for v in sides)
        m.tone_touched = True if any(sides) else (
            bool(m.tone_touched) and m.intensity > M.level_for(max(abs(m.p), abs(m.e)), 0))
        m.p, m.e = round(m.p, _PAIR_DP), round(m.e, _PAIR_DP)
        self._state.mood = m          # one reference swap: a reader never sees it torn
        self._flush()
        return m

    def _good_again(self) -> None:
        """Review r2: a mood was WRITTEN — the state on file is a good one again, so ``load_problem`` (why
        the file could not be read at start) stops being said. The ``.bad`` copy and the ERROR line stay."""
        self.load_problem = ""

    def _settle_locked(self, now: Optional[datetime.datetime]) -> Mood:
        """Bring the stored pair to ``now`` (decay) and write it down when
        that crossed a boundary. Returns the mood as it stands (stored or
        not). Caller holds the lock."""
        if now is None:
            now = _aware(None)               # one instant for the pair, its shares and the stamp
        stored = self._stored(now)
        p, e, label, n, novelty = self._evolve(stored, now)
        base = stored if stored is not None else Mood()
        cur = Mood(
            p=p, e=e, updated_at=_iso(now), label=label, intensity=n, novelty=novelty,
            source=base.source, set_at=base.set_at, cause=base.cause,
            recent=list(base.recent), **self._kept_manner(base, now), **self._kept_owner(stored, now),
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
            return self._view_of(self._settle_locked(now), now)

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

    def mark_owner_job(self, *, now: Optional[datetime.datetime] = None) -> None:
        """§4UD (F9): a background job the OWNER started did work just now —
        the agent is not idle, so the quiet is measured from here (when this
        is later than the owner's own last turn). A boredom the journal holds
        is closed with its own row. The pair is not touched. The caller
        decides that the job is the owner's and that it worked."""
        with self._lock:
            cur = self._settle_locked(now)
            was = self._open_boredom()
            if was > 0:
                row = Mood(p=cur.p, e=cur.e, label=cur.label, intensity=cur.intensity,
                           set_at=_iso(now), cause=JOB_QUIET_ENDED_CAUSE)
                self._append_history_locked(row, M.KIND_QUIET_END, prev=f"{M.BORED}, {was} of 3")
            self._state.last_owner_job_at = _iso(now)
            self._bored_seen = None
            self._flush()

    def mark_owner_turn(self, *, now: Optional[datetime.datetime] = None) -> None:
        """Stamp that the owner is here (a request of his arrived, or ended).
        The next quiet is measured from this."""
        with self._lock:
            self._state.last_owner_turn_at = _iso(now)
            self._bored_seen = None
            self._flush()

    #: why an event was not applied (§4TX)
    HELD_OWNER_HOLD = "the owner set the mood (held until the day ends)"
    HELD_OWNER_LATER = "the owner set the mood after the message"

    def hold_reason(self, *, now: Optional[datetime.datetime] = None, arrived: str = "") -> str:
        """§4TX: why an event would not be applied right now — "" when it
        would be. The owner's hold is on; or the owner set, reset or changed
        the manner after ``arrived`` (when the message the event is about came in)."""
        stored = self._stored(now)
        if self._held_until(stored, now) is not None:
            return self.HELD_OWNER_HOLD
        if arrived and stored is not None and stored.owner_at:
            a, o = M.parse_iso_utc(arrived), M.parse_iso_utc(stored.owner_at)
            if a is not None and o is not None and o >= a:
                return self.HELD_OWNER_LATER
        return ""

    def nudge(self, dp: float, de: float, cause: str, kind: str, *, novelty: bool = False,
              now: Optional[datetime.datetime] = None, arrived: str = "",
              tone: Optional[Sides] = None) -> Tuple[MoodView, MoodView]:
        """THE function every event goes through: move the pair by
        ``(dp, de)``, clamp it, record it. Returns ``(before, after)``.
        §4TX: an event is held — recorded in the events log, not applied —
        while the owner's hold is on, or when it is about a message that
        ``arrived`` before the owner last set the mood. Returns the SAME view
        twice when nothing was applied.

        §4UD (F1): the pair is two things added together — the pair WITHOUT
        the owner's message tone, and the four tone shares. ``(dp, de)``
        moves the first (clamped to [-1, 1] on its own, exactly as it would
        in a run that never read a tone); ``tone`` — ``(up p, up e, down p,
        down e)`` — is what the caller asks the shares to change by, and
        ``tone_sink`` decides what of it gets in. An event of kind
        ``sentiment`` with no ``tone`` IS a tone: its ``(dp, de)`` goes to
        the shares of its sign and the tone-free pair does not move."""
        with self._lock:
            base = self._settle_locked(now)
            rest = self.rest()                   # §4UA: one rest for both views of the event
            before = self._view_of(base, now, rest)
            try:
                dp, de = float(dp), float(de)
                delta = tuple(float(v) for v in tone) if tone is not None else None
                if delta is not None and len(delta) != 4:
                    raise ValueError("tone is four numbers")
            except (TypeError, ValueError) as e:
                logger.warning("mood event refused (%s): its size is not a number (%s)", kind, e)
                return before, before
            if not all(math.isfinite(v) for v in (dp, de) + (delta or ())):
                # §4UD (F13): inf / NaN went through ``clamp`` as 0 and ZEROED the pair (0.6 → 0.0),
                # journalled as a normal event
                logger.warning("mood event refused (%s, %s): a size that is not finite", kind, cause)
                return before, before
            if delta is None and kind == M.KIND_SENTIMENT:
                delta = (max(0.0, dp), max(0.0, de), max(0.0, -dp), max(0.0, -de))
                ask_p, ask_e, dp, de = dp, de, 0.0, 0.0
            else:
                ask_p = dp + ((delta[0] - delta[2]) if delta else 0.0)
                ask_e = de + ((delta[1] - delta[3]) if delta else 0.0)
            if dp == 0.0 and de == 0.0 and not (delta and any(delta)):
                return before, before
            why = self.hold_reason(now=now, arrived=arrived)
            if why:
                self._append_event_locked(kind=kind, cause=cause, dp=ask_p, de=ask_e, held=why, now=now)
                return before, before
            at = _iso(now)
            owner = self._carried(base)
            was = _sides_of(base)
            # the pair without the owner's tone, moved on its own — what a run that read no tone holds
            bp0 = M.clamp(base.p - was[0] + was[2])
            be0 = M.clamp(base.e - was[1] + was[3])
            bp, be = M.clamp(bp0 + dp), M.clamp(be0 + de)
            p, e, sides, refused = tone_sink(was, delta, bp, be)
            owner.update(_sides_kw(sides))
            # the novelty tag belongs to the move that set it: a later event
            # that is not itself a research event clears it (review r2 —
            # research, then three thumbs, read "curious, 3 of 3: the owner
            # gave a thumbs-up")
            label, n, nov = M.read_off(p, e, held_label=base.label, held_intensity=base.intensity,
                                       novelty=bool(novelty))
            changed = (label, n) != (base.label, base.intensity)
            cause = _cap_text(str(cause or ""), _MOOD_EVIDENCE_MAX_CHARS)
            # the pair WITHOUT the message tone: a tone-only move leaves it (undo leaves the tone out too, and
            # a tone-free run must hold the same undo point — the twin invariant)
            if (abs(bp - bp0) > 1e-9 or abs(be - be0) > 1e-9) and (owner.get("prev") or {}):
                # §4UD-b M2: what the owner's last set replaced is undoable only while the stored mood is
                # the one that set made — once an event has moved it, undo would put back a mood from
                # before things that happened since (and say the set mood "was" a level it never had)
                owner = dict(owner, prev={})
            new = Mood(
                p=p, e=e, updated_at=at, label=label, intensity=n, novelty=nov,
                source=(M.SOURCE_AUTO if changed else base.source),
                set_at=(at if changed else base.set_at),
                cause=(cause if changed else base.cause),
                recent=(list(base.recent) + [{"at": at, "cause": cause, "kind": str(kind or "")}]
                        )[-MOOD_RECENT_CAUSES:],
                **self._kept_manner(base, now), **owner,
            )
            self._store_locked(new)
            self._append_event_locked(kind=kind, cause=cause, dp=ask_p, de=ask_e, after=self._describe(new),
                                      now=now)
            if changed:
                self._append_history_locked(new, str(kind or ""), prev=before.describe())
            r9 = lambda *vs: tuple(round(v, 9) for v in vs)   # noqa: E731 — float dust is not a move
            return before, self._view_of(
                new, now, rest, moved=r9(p - base.p, e - base.e), base_moved=r9(bp - bp0, be - be0),
                toned=r9(*(sides[i] - was[i] for i in range(4))), refused=r9(*refused))

    @staticmethod
    def _describe(m: Mood) -> str:
        return M.NEUTRAL if not m.intensity else f"{m.label}, {m.intensity} of 3"

    @staticmethod
    def _kept_manner(m: Optional[Mood], now: Optional[datetime.datetime]) -> Dict[str, object]:
        """The manner fields a WRITE carries over (review §4TS): the stored
        ones while the manner's day is on, none once it is over — so an
        expired manner is dropped at the next write instead of staying on
        disk, where a clock set back would bring it back."""
        if m is None or not m.manner:
            return dict(manner="", manner_intensity=0, manner_set_at="")
        if M.manner_level(m.manner_intensity, m.manner_set_at, now=_aware(now)) == 0:
            return dict(manner="", manner_intensity=0, manner_set_at="")
        return dict(manner=m.manner, manner_intensity=m.manner_intensity, manner_set_at=m.manner_set_at)

    @classmethod
    def _kept_owner(cls, m: Optional[Mood], now: Optional[datetime.datetime]) -> Dict[str, object]:
        """§4TX: the owner fields a WRITE carries over — the hold while it is
        on, when the owner last spoke, and the sentiment share brought to
        ``now`` (it decays with the pair, so it is stored decayed with it)."""
        if m is None:
            return dict(held_until="", owner_at="", tone_touched=False, prev={}, **_sides_kw(_NO_SIDES))
        held = cls._held_until(m, now)
        return dict(held_until=(m.held_until if held is not None else ""), owner_at=m.owner_at,
                    tone_touched=bool(m.tone_touched), prev=dict(m.prev or {}),
                    **_sides_kw(cls._sides_now(m, now)))

    @staticmethod
    def _carried(base: Mood) -> Dict[str, object]:
        """The owner fields of a mood ALREADY settled to now (``_settle_locked``'s result), as the next write
        carries them — nothing decays a second time."""
        return dict(held_until=base.held_until, owner_at=base.owner_at, tone_touched=bool(base.tone_touched),
                    prev=dict(base.prev or {}), **_sides_kw(_sides_of(base)))

    @staticmethod
    def _manner_now(m: Optional[Mood], now: Optional[datetime.datetime]) -> Dict[str, object]:
        """The manner fields a reader sees at ``now`` (§4TS): the stored
        manner at the level its day has come to — none once the day is
        over — with the steps still ahead. The stored fields are never
        rewritten by this: the level is a function of when it was set."""
        if m is None or not m.manner:
            return dict(manner="", manner_intensity=0, manner_set_at="", manner_steps=())
        when = _aware(now)
        level = M.manner_level(m.manner_intensity, m.manner_set_at, now=when)
        if level == 0:
            return dict(manner="", manner_intensity=0, manner_set_at="", manner_steps=())
        ahead = tuple((_iso(at), lvl) for at, lvl in M.manner_schedule(m.manner_intensity, m.manner_set_at)
                      if at > when)
        return dict(manner=m.manner, manner_intensity=level, manner_set_at=m.manner_set_at,
                    manner_steps=ahead)

    def _view_of(self, m: Mood, now: Optional[datetime.datetime] = None, rest=None,
                 moved: Tuple[float, float] = (0.0, 0.0), base_moved: Tuple[float, float] = (0.0, 0.0),
                 toned: Sides = _NO_SIDES, refused: Sides = _NO_SIDES) -> MoodView:
        cls = type(self)
        rp, re_ = self.rest() if rest is None else rest   # §4UA: a reader sees rest + deviation
        return MoodView(
            label=m.label, intensity=m.intensity, p=_shown(m.p, rp), e=_shown(m.e, re_), moved=moved,
            base_moved=base_moved, toned=toned, refused=refused,
            source=(m.source if m.intensity else M.SOURCE_AUTO), set_at=m.set_at,
            cause=(m.cause if m.intensity else ""),
            recent=tuple(dict(r) for r in m.recent),
            held_until=(m.held_until if cls._held_until(m, now) is not None else ""),
            **cls._manner_now(m, now),
        )

    def set_owner_mood(self, label: str, direction: Tuple[float, float], intensity: int, *,
                       note: str = "", now: Optional[datetime.datetime] = None, hold: bool = True,
                       hold_until: str = "") -> Tuple[MoodView, MoodView]:
        """The owner said how to be: put the pair on ``label``'s anchor at
        ``intensity``. §4TX: it is HELD there — no decay, no event moves it —
        until the end of the day (the next 04:00, ``mood.manner_ends_at``);
        from then it decays and reacts like any mood. ``hold=False`` (a step
        DOWN on a mood: "be less irritable"): set, not held — it decays and
        reacts at once, as the mood it stepped down did. ``hold_until``: keep
        this hold end (a step down on a held mood: review r4 — a new hold at
        03:30 ran to the NEXT day's 04:00).

        §4UD (F1): the pair the owner places holds NO tone share — a set mood
        and a step down alike. (§4TX r7 carried a part of the share over a
        step down, so that others would not hear a mood only the owner's
        messages had made. Since §4UB the shares are capped at 0.12, under
        the 0.16 a level is given up at: the level a step lands on is one
        other people already heard. And a share carried onto a pair the
        owner placed was a share of nothing — in the fuzz against a tone-free
        twin it was the one way the two runs' ledgers parted.)
        Returns ``(before, after)``."""
        with self._lock:
            base = self._settle_locked(now)
            before = self._view_of(base, now)
            n = max(1, min(3, int(intensity)))
            # §4UA: the anchor IS the deviation — the label and level read off it are the ones the owner named
            p, e = M.anchor_for(direction, n)
            at = _iso(now)
            changed = (label, n) != (base.label, base.intensity)
            new = Mood(
                p=p, e=e, updated_at=at, label=label, intensity=n,
                novelty=(label == M.CURIOUS), source=M.SOURCE_OWNER,
                set_at=(at if changed else (base.set_at or at)), cause=OWNER_CAUSE,
                recent=(list(base.recent) + [{"at": at, "cause": OWNER_CAUSE, "kind": M.KIND_OWNER}]
                        )[-MOOD_RECENT_CAUSES:],
                **self._kept_manner(base, now),
                held_until=((hold_until or self._hold_end(at)) if hold else ""), owner_at=at,
                prev=self._replaced(base, at),
            )
            self._store_locked(new)
            self.pair_epoch += 1
            self._append_event_locked(kind=M.KIND_OWNER, cause=OWNER_CAUSE, dp=p - base.p, de=e - base.e,
                                      after=self._describe(new), now=now)
            if changed:
                self._append_history_locked(new, M.KIND_OWNER,
                                            _cap_text(note, _MOOD_EVIDENCE_MAX_CHARS),
                                            prev=before.describe())
            return before, self._view_of(new, now)

    @staticmethod
    def _replaced(base: Mood, at: str) -> Dict[str, object]:
        """§4UD (G2): what the owner's set mood is about to replace — kept in
        the state so ``undo_owner_mood`` can put it back. The pair WITHOUT
        the owner's message tone, and when. Its label and level are kept only
        when it was itself a mood the owner set and still HELD (then nothing
        but the owner's word made them). Review r2: the label and level of an
        automatic mood are the OWNER's reading, tone included — an undo that
        kept them held "content, 2" for everyone over a pair that is
        "content, 1" without the owner's tone. Nothing here depends on a tone."""
        up_p, up_e, dn_p, dn_e = _sides_of(base)
        prev: Dict[str, object] = {
            "p": round(M.clamp(base.p - up_p + dn_p), _PAIR_DP), "e": round(M.clamp(base.e - up_e + dn_e), _PAIR_DP),
            "at": at}
        if base.held_until and base.source == M.SOURCE_OWNER and base.intensity:
            prev.update(label=base.label, intensity=base.intensity, owner=True, held_until=base.held_until)
        return prev

    def undo_owner_mood(self, *, now: Optional[datetime.datetime] = None) -> Optional[Tuple[MoodView, MoodView]]:
        """§4UD (G2): put back what the owner's LAST set mood replaced — the
        pair as it was then (without any message tone), worn off by the time
        since, and READ AFRESH: its label and level are what that pair gives
        now, for everyone alike. Only a mood the owner had set and whose hold
        is still on comes back as it was, held to the same end. One step:
        there is nothing to undo after it ("neutral" is not undone — it
        clears what could be). None when nothing is on file to put back. It
        is the owner's word: the ledger's amounts are void, events about
        earlier messages are held, and the tone shares stay out."""
        with self._lock:
            base = self._settle_locked(now)
            prev = dict(base.prev or {})
            if not prev.get("at"):
                return None
            before = self._view_of(base, now)
            at = _iso(now)
            # what was replaced is read by the same rule as what is stored: a stamp too far ahead is no time
            then = self._sane(Mood(p=float(prev.get("p") or 0.0), e=float(prev.get("e") or 0.0),
                                   updated_at=str(prev["at"]), held_until=str(prev.get("held_until") or "")), now)
            held = self._held_until(then, now) if prev.get("owner") else None
            if held is not None:
                label, n = str(prev.get("label") or M.NEUTRAL), int(prev.get("intensity") or 0)
                p, e = then.p, then.e
            else:
                p, e, label, n, _nov = self._evolve(then, now)       # fresh: nothing held from the owner's view
            owners = held is not None and n > 0
            new = Mood(p=p, e=e, updated_at=at, label=(label if n else M.NEUTRAL), intensity=n,
                       # a held "curious" is read back as curious (as `set_owner_mood` stores it): without the tag
                       # the next settle read "cheerful", booked a change and dropped the owner's hold
                       novelty=(owners and label == M.CURIOUS),
                       source=(M.SOURCE_OWNER if owners else M.SOURCE_AUTO), set_at=at,
                       cause=(UNDO_CAUSE if n else ""),
                       recent=(list(base.recent) + [{"at": at, "cause": UNDO_CAUSE, "kind": M.KIND_OWNER}]
                               )[-MOOD_RECENT_CAUSES:],
                       **self._kept_manner(base, now),
                       held_until=(then.held_until if owners else ""), owner_at=at)
            self._store_locked(new)
            self.pair_epoch += 1
            self._append_event_locked(kind=M.KIND_OWNER, cause=UNDO_CAUSE, dp=p - base.p, de=e - base.e,
                                      after=self._describe(new), now=now)
            if (new.label, n) != (base.label, base.intensity):
                self._append_history_locked(new if n else replace(new, cause=UNDO_CAUSE), M.KIND_OWNER,
                                            prev=before.describe())
            return before, self._view_of(new, now)

    @staticmethod
    def _hold_end(at: str) -> str:
        """§4TX: when an owner-set mood stops being held — the manner's day end."""
        try:
            end = M.manner_ends_at(at)
        except Exception:  # noqa: BLE001
            end = None
        return _iso(end) if end is not None else ""

    def _manner_row_locked(self, base: Mood, cause: str, now: Optional[datetime.datetime]) -> None:
        """§4TX: a manner change is a history row (kind "manner") — the pair as it stands, the change as cause."""
        row = Mood(p=base.p, e=base.e, label=base.label, intensity=base.intensity,
                   set_at=_iso(now), cause=cause)
        self._append_history_locked(row, M.KIND_MANNER, prev=self._describe(base))
        self._append_event_locked(kind=M.KIND_MANNER, cause=cause, dp=0.0, de=0.0, now=now)

    def reset_mood(self, *, note: str = "", now: Optional[datetime.datetime] = None,
                   keep_manner: bool = False) -> Tuple[MoodView, MoodView]:
        """The owner said "neutral": the pair goes to 0, 0 AND the manner is
        cleared. ``keep_manner`` (§4TX: a mood stepped down to nothing): the
        manner stays as it is. Returns ``(before, after)``."""
        with self._lock:
            base = self._settle_locked(now)
            before = self._view_of(base, now)
            at = _iso(now)
            was_mood = base.intensity > 0
            # what the owner can SEE (review §4TS): a mood, or a manner whose
            # day is not over. Only then is "neutral" his word on record — an
            # expired manner or a pair decayed under the first threshold (it
            # never reaches exactly 0) is zeroed silently, with no "the owner
            # said so" event to show up later as a reason
            visible = was_mood or (bool(before.manner) and not keep_manner)
            # the STORED record, not ``base`` (round-2 review): ``base`` has
            # an expired manner already blanked, so testing it left that
            # manner on disk for a clock set back to revive
            stored = self._state.mood          # raw on purpose: what is ON FILE, whatever its stamps say
            on_file = bool(stored is not None and stored.manner)
            if not visible and (keep_manner or not on_file) and base.p == 0.0 and base.e == 0.0:
                if stored is not None and stored.prev:
                    stored.prev = {}             # "neutral" always clears what an undo could put back (below)
                    self._flush()
                return before, before            # nothing on file to reset
            new = Mood(
                p=0.0, e=0.0, updated_at=at, label=M.NEUTRAL, intensity=0, novelty=False,
                source=M.SOURCE_AUTO, set_at=(at if was_mood else base.set_at),
                cause=(OWNER_CAUSE if was_mood else base.cause),
                recent=((list(base.recent) + [{"at": at, "cause": OWNER_CAUSE, "kind": M.KIND_OWNER}])
                        if visible else list(base.recent))[-MOOD_RECENT_CAUSES:],
                owner_at=(at if visible else base.owner_at),
                # §4UD review r2: a "neutral" is not undone, and it clears what an undo could put back — ALWAYS:
                # whether it is "the owner's word on record" (``visible``) depends on the owner's own tone, and
                # what an undo finds must not
                prev={},
                **(self._kept_manner(base, now) if keep_manner else {}),
            )
            self._store_locked(new)
            self.pair_epoch += 1
            if visible:
                self._append_event_locked(kind=M.KIND_OWNER, cause=OWNER_CAUSE + ": neutral",
                                          dp=-base.p, de=-base.e, after=M.NEUTRAL, now=now)
            if was_mood:
                self._append_history_locked(new, M.KIND_OWNER,
                                            _cap_text(note, _MOOD_EVIDENCE_MAX_CHARS),
                                            prev=before.describe())
            if before.manner and not keep_manner:
                # §4TX: the manner it cleared is said too (review: it vanished without a line)
                self._manner_row_locked(new, f"the owner cleared the manner ({before.manner}) with neutral", now)
            return before, self._view_of(new, now)

    def set_manner(self, manner: str, intensity: int, *,
                   now: Optional[datetime.datetime] = None) -> Tuple[MoodView, MoodView]:
        """The owner set a way of talking ("" clears it). It lasts until the
        end of the day, stepping down on the way (§4TS, ``mood.manner_level``);
        setting it again — the same manner included — starts a new day. The
        mood itself is left as it is. Returns ``(before, after)``."""
        with self._lock:
            base = self._settle_locked(now)
            before = self._view_of(base, now)
            man = M.resolve_manner(manner)
            stored = self._state.mood          # raw on purpose: an expired manner still ON FILE is dropped below
            if not man and not before.manner and not (stored is not None and stored.manner):
                return before, before            # nothing set (an expired one on file is dropped below)
            owner = self._carried(base)
            owner["owner_at"] = _iso(now)
            new = Mood(
                p=base.p, e=base.e, updated_at=base.updated_at or _iso(now), label=base.label,
                intensity=base.intensity, novelty=base.novelty, source=base.source,
                set_at=base.set_at, cause=base.cause, recent=list(base.recent),
                manner=man,
                manner_intensity=(max(1, min(3, int(intensity))) if man else 0),
                manner_set_at=(_iso(now) if man else ""),
                **owner,
            )
            self._store_locked(new)
            if man:
                self._manner_row_locked(new, f"the owner set the manner: {man}, {new.manner_intensity} of 3", now)
            elif before.manner:
                self._manner_row_locked(new, f"the owner cleared the manner ({before.manner})", now)
            return before, self._view_of(new, now)

    def mood_history(self, *, limit: int = 20) -> List[Dict[str, object]]:
        """The newest ``limit`` rows of the mood journal, oldest first. Rows
        written before §4TC (a label with no pair) are skipped. Empty when
        the file is missing or ``limit`` <= 0."""
        if limit <= 0:
            return []
        out: List[Dict[str, object]] = []
        # one byte that is not UTF-8, a torn line, nesting too deep: that line, not the journal (the shared reader)
        for d in read_jsonl_rows(self.mood_history_path, what="mood journal"):
            if "at" not in d or "p" not in d:
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
        """Today's rows of the journal, oldest first, each with its local
        time as ``"hhmm"``. §4UD (F7): "today" is the MOOD day of ``now``
        (``mood.mood_day``: it ends at 04:00, as the owner's hold and the
        manner do), not the calendar day."""
        days = self.mood_days(days=1, now=now, tz=tz, limit=limit)
        return days[-1][1] if days else []

    def mood_days(self, *, days: int = 1, now: Optional[datetime.datetime] = None, tz=None,
                  limit: int = MOOD_HISTORY_MAX_ROWS) -> List[Tuple[datetime.date, List[Dict[str, object]]]]:
        """§4UD (G3): the journal's rows of the last ``days`` mood days
        (today included), as ``[(day, rows)]`` oldest day first — every day
        of the span, with ``[]`` for one that has no row."""
        today = M.mood_day(now, tz=tz)
        span = [today - datetime.timedelta(days=k) for k in range(max(1, int(days)) - 1, -1, -1)]
        by_day: Dict[datetime.date, List[Dict[str, object]]] = {d: [] for d in span}
        for row in self.mood_history(limit=limit):
            at = M.parse_iso_utc(str(row.get("at") or ""))
            if at is None:
                continue
            day = M.mood_day(at, tz=tz)
            if day in by_day:
                local = at.astimezone(tz) if tz is not None else at.astimezone()
                by_day[day].append({**row, "hhmm": local.strftime("%H:%M")})
        return [(d, by_day[d]) for d in span]

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
            # the diary (include_mood=False) gets the name only: the level steps
            # with the clock (§4TS), and its prompt's hash is its skip-guard
            lines.append(f"My manner (set by the owner): {view.manner}"
                         + (f", {view.manner_intensity} of 3." if include_mood else "."))
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        return text
