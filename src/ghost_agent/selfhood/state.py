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
from dataclasses import dataclass, field
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
    #: ``nudge``'s after-view only; the reaction ledger books this
    moved: Tuple[float, float] = field(default=(0.0, 0.0), compare=False, repr=False)
    #: §4UB review r6: what the sentiment-share cap took out of the pair in that same event (included in
    #: ``moved``) — the share it removed belongs to the tones the ledger booked, not to the event's reply
    capped: Tuple[float, float] = field(default=(0.0, 0.0), compare=False, repr=False)

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


def _toward_zero(v: float, share: float) -> float:
    """``v`` without the part ``share`` put in it — only toward zero, never past it."""
    if v > 0 and share > 0:
        return max(0.0, v - share)
    if v < 0 and share < 0:
        return min(0.0, v - share)
    return v


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
        self._state: SelfState = self._read_or_empty()
        #: how long the owner has been quiet, in seconds (None = unknown) —
        #: attached by the process that owns the clock (``main.py``); this
        #: class only ever READS it
        self._quiet_fn: Optional[Callable[[], Optional[float]]] = None
        #: (start of the quiet as a timestamp, highest boredom level the
        #: journal has a row for) — what `_note_bored` last worked out
        self._bored_seen: Optional[Tuple[float, int]] = None
        #: §4UA: where the pair rests — attached by the process that owns the
        #: Mirror (``main.py``); None = rest at (0, 0), as before
        self._rest_fn: Optional[Callable[[], Tuple[float, float]]] = None

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

    def attach_rest(self, rest_fn: Optional[Callable[[], Tuple[float, float]]]) -> None:
        """§4UA: give the thread its resting point — the character's, read
        from the Mirror. The thread only READS it, on every read of the pair."""
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
    def _private_now(stored: Optional[Mood], now: Optional[datetime.datetime]) -> Tuple[float, float]:
        """§4TX: the owner's-sentiment part of the pair at ``now`` — it decays with the pair."""
        if stored is None or (not stored.private_p and not stored.private_e):
            return 0.0, 0.0
        age = SelfStateThread._age(stored, now)
        if age is None:
            return 0.0, 0.0
        return M.decay(stored.private_p, age), M.decay(stored.private_e, age)

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
        stored = self._state.mood
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
            pp, pe = self._private_now(stored, now)
            q_p, q_e = _toward_zero(p, pp), _toward_zero(e, pe)
            # review r3: with a share taken out, a fresh read — the owner's level, held by the margin, is
            # not carried over to others
            # review r6: ANY share — the owner's held level may exist only because of the owner's share (r4/r5 kept
            # it for a small share, and others heard it rise while the mood decayed). Others may step down a
            # little before the owner does; they never hear more than the owner does.
            shared = bool(pp or pe)
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

    @staticmethod
    def _cap_share_locked(owner: Dict[str, object], p: float, e: float) -> Tuple[float, float]:
        """§4UB: keep the stored private share within ±``mood.SENTIMENT_SHARE_MAX`` (energy: the same fraction
        of the tone's own energy), taking the excess out of the pair with it. Returns the pair."""
        cap_p = M.SENTIMENT_SHARE_MAX
        cap_e = M.SENTIMENT_SHARE_MAX * abs(M.EV_SENTIMENT_POSITIVE[1] / M.EV_SENTIMENT_POSITIVE[0])
        x = float(owner.get("private_p") or 0.0)
        if abs(x) > cap_p:
            over = x - (cap_p if x > 0 else -cap_p)
            p, owner["private_p"] = M.clamp(p - over), (cap_p if x > 0 else -cap_p)
        y = float(owner.get("private_e") or 0.0)
        if abs(y) > cap_e:
            over = y - (cap_e if y > 0 else -cap_e)
            e, owner["private_e"] = M.clamp(e - over), (cap_e if y > 0 else -cap_e)
        return p, e

    def private_share(self, *, now: Optional[datetime.datetime] = None) -> Tuple[float, float]:
        """§4UB: what the owner's message sentiment has in the pair now (decayed, stored — a restart keeps it;
        the reaction ledger takes it back out when a stronger reaction replaces a tone)."""
        with self._lock:
            return self._private_now(self._state.mood, now)

    def note_held(self, kind: str, cause: str, dp: float, de: float, why: str, *,
                  now: Optional[datetime.datetime] = None) -> None:
        """§4TX: an event that was not applied, with why — one events-log row."""
        with self._lock:
            self._append_event_locked(kind=kind, cause=cause, dp=dp, de=de, held=why, now=now)

    def mood_events(self, *, limit: int = 50) -> List[Dict[str, object]]:
        """The newest ``limit`` rows of ``mood.events.jsonl``, oldest first."""
        if limit <= 0 or not self.mood_events_path.exists():
            return []
        out: List[Dict[str, object]] = []
        try:
            lines = self.mood_events_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        for line in lines[-limit:]:
            try:
                d = json.loads(line)
            except (ValueError, RecursionError):
                continue
            if isinstance(d, dict) and "at" in d:
                out.append(d)
        return out

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
        stored = self._state.mood
        if self._held_until(stored, now) is not None:
            return self.HELD_OWNER_HOLD
        if arrived and stored is not None and stored.owner_at:
            a, o = M.parse_iso_utc(arrived), M.parse_iso_utc(stored.owner_at)
            if a is not None and o is not None and o >= a:
                return self.HELD_OWNER_LATER
        return ""

    def nudge(self, dp: float, de: float, cause: str, kind: str, *, novelty: bool = False,
              now: Optional[datetime.datetime] = None, arrived: str = "",
              private_adjust: Optional[Tuple[float, float]] = None) -> Tuple[MoodView, MoodView]:
        """THE function every event goes through: move the pair by
        ``(dp, de)``, clamp it, record it. Returns ``(before, after)``.
        §4TX: an event is held — recorded in the events log, not applied —
        while the owner's hold is on, or when it is about a message that
        ``arrived`` before the owner last set the mood. ``private_adjust``: a
        change to the owner's private sentiment share that comes with a
        non-sentiment event (the reaction ledger: a stronger reaction to the
        same reply replaced the message's sentiment). Returns the SAME view
        twice when nothing was applied."""
        with self._lock:
            base = self._settle_locked(now)
            rest = self.rest()                   # §4UA: one rest for both views of the event
            before = self._view_of(base, now, rest)
            try:
                dp, de = float(dp), float(de)
            except (TypeError, ValueError):
                return before, before
            if dp == 0.0 and de == 0.0:
                return before, before
            why = self.hold_reason(now=now, arrived=arrived)
            if why:
                self._append_event_locked(kind=kind, cause=cause, dp=dp, de=de, held=why, now=now)
                return before, before
            at = _iso(now)
            p, e = M.clamp(base.p + dp), M.clamp(base.e + de)
            owner = self._kept_owner(self._state.mood, now)
            if private_adjust is not None:
                ap_, ae_ = float(private_adjust[0]), float(private_adjust[1])
                if kind == M.KIND_SENTIMENT:
                    # review r4: only the part the clamp let through is the owner's share in the pair
                    ap_ *= max(0.0, min(1.0, (p - base.p) / dp)) if dp else 0.0
                    ae_ *= max(0.0, min(1.0, (e - base.e) / de)) if de else 0.0
                owner.update(private_p=M.clamp(owner["private_p"] + ap_),
                             private_e=M.clamp(owner["private_e"] + ae_))
            elif kind == M.KIND_SENTIMENT:
                # §4TX: the share the owner's sentiment put in — what was really added, after the clamp
                owner.update(private_p=M.clamp(owner["private_p"] + (p - base.p)),
                             private_e=M.clamp(owner["private_e"] + (e - base.e)))
            # §4UB review r5: THE cap on what the owner's message sentiment holds in the pair, kept at the one place
            # the share is written. Five review rounds found a ledger path around each cap placed before it
            # (a restart, a second tone on one reply, a changed thumb, a tone that stopped counting); here no path
            # gets around it: whatever exceeds ±SENTIMENT_SHARE_MAX leaves the share AND the pair together
            p0, e0 = p, e
            p, e = self._cap_share_locked(owner, p, e)
            capped = (round(p0 - p, 4), round(e0 - e, 4))
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
                **self._kept_manner(base, now), **owner,
            )
            self._store_locked(new)
            self._append_event_locked(kind=kind, cause=cause, dp=dp, de=de, after=self._describe(new), now=now)
            if changed:
                self._append_history_locked(new, str(kind or ""), prev=before.describe())
            return before, self._view_of(new, now, rest, moved=(round(p - base.p, 4), round(e - base.e, 4)),
                                         capped=capped)

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
            return dict(held_until="", owner_at="", private_p=0.0, private_e=0.0)
        pp, pe = cls._private_now(m, now)
        held = cls._held_until(m, now)
        return dict(held_until=(m.held_until if held is not None else ""), owner_at=m.owner_at,
                    private_p=round(pp, 4), private_e=round(pe, 4))

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
                 moved: Tuple[float, float] = (0.0, 0.0), capped: Tuple[float, float] = (0.0, 0.0)) -> MoodView:
        cls = type(self)
        rp, re_ = self.rest() if rest is None else rest   # §4UA: a reader sees rest + deviation
        return MoodView(
            label=m.label, intensity=m.intensity, p=_shown(m.p, rp), e=_shown(m.e, re_), moved=moved, capped=capped,
            source=(m.source if m.intensity else M.SOURCE_AUTO), set_at=m.set_at,
            cause=(m.cause if m.intensity else ""),
            recent=tuple(dict(r) for r in m.recent),
            held_until=(m.held_until if cls._held_until(m, now) is not None else ""),
            **cls._manner_now(m, now),
        )

    def set_owner_mood(self, label: str, direction: Tuple[float, float], intensity: int, *,
                       note: str = "", now: Optional[datetime.datetime] = None, hold: bool = True,
                       hold_until: str = "", keep_share: bool = False) -> Tuple[MoodView, MoodView]:
        """The owner said how to be: put the pair on ``label``'s anchor at
        ``intensity``. §4TX: it is HELD there — no decay, no event moves it —
        until the end of the day (the next 04:00, ``mood.manner_ends_at``);
        from then it decays and reacts like any mood. ``hold=False`` (a step
        DOWN on a mood: "be less irritable"): set, not held — it decays and
        reacts at once, as the mood it stepped down did. ``keep_share`` (a step
        down — review r7: zeroing the owner's sentiment share let others hear a
        mood only the owner's messages had made): the share is carried over in
        proportion to the step. ``hold_until``: keep
        this hold end (a step down on a held mood: review r4 — a new hold at
        03:30 ran to the NEXT day's 04:00).
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
                **self._share_after_step(base, p, e, now, keep_share),
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

    @classmethod
    def _share_after_step(cls, base: Mood, p: float, e: float, now, keep: bool) -> Dict[str, float]:
        """§4TX r7: the owner's sentiment share after the owner put the pair on (p, e) — none for the owner's own word
        (a set mood), the same PART of each axis for a step down (only where it lies on the pair's side)."""
        if not keep:
            return dict(private_p=0.0, private_e=0.0)
        pp, pe = cls._private_now(base, now)

        def _part(share: float, old: float, new: float, cap: float) -> float:
            if not share or not old or share * old <= 0 or old * new <= 0:
                return 0.0
            # §4UB review r6: never more than the cap (an anchor larger than the pair on an axis grew it past)
            bound = min(abs(new), abs(share), cap)
            return max(-bound, min(bound, share * (new / old)))
        cap_p = M.SENTIMENT_SHARE_MAX
        cap_e = M.SENTIMENT_SHARE_MAX * abs(M.EV_SENTIMENT_POSITIVE[1] / M.EV_SENTIMENT_POSITIVE[0])
        return dict(private_p=round(_part(pp, base.p, p, cap_p), 4), private_e=round(_part(pe, base.e, e, cap_e), 4))

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
            stored = self._state.mood
            on_file = bool(stored is not None and stored.manner)
            if not visible and (keep_manner or not on_file) and base.p == 0.0 and base.e == 0.0:
                return before, before            # nothing on file to reset
            new = Mood(
                p=0.0, e=0.0, updated_at=at, label=M.NEUTRAL, intensity=0, novelty=False,
                source=M.SOURCE_AUTO, set_at=(at if was_mood else base.set_at),
                cause=(OWNER_CAUSE if was_mood else base.cause),
                recent=((list(base.recent) + [{"at": at, "cause": OWNER_CAUSE, "kind": M.KIND_OWNER}])
                        if visible else list(base.recent))[-MOOD_RECENT_CAUSES:],
                owner_at=(at if visible else base.owner_at),
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
            stored = self._state.mood
            if not man and not before.manner and not (stored is not None and stored.manner):
                return before, before            # nothing set (an expired one on file is dropped below)
            owner = self._kept_owner(stored, now)
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
            # the diary (include_mood=False) gets the name only: the level steps
            # with the clock (§4TS), and its prompt's hash is its skip-guard
            lines.append(f"My manner (set by the owner): {view.manner}"
                         + (f", {view.manner_intensity} of 3." if include_mood else "."))
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        return text
