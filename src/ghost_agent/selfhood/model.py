"""SelfModel — the facade the rest of the agent talks to.

Holds the four substrates (autobiographical log, self-state thread,
recognition layer, narrative summariser) behind one attribute so the
agent only ever wires ``context.self_model``. The biological
watchdog phase, the post-turn capture hook, and the prompt
assembly path all read from this one object.

Disabled mode: when ``enabled=False`` (e.g. ``--no-memory``,
``--no-self-model``, or missing memory_dir), every method becomes a
no-op. This lets the lifespan unconditionally attach a SelfModel and
the callers don't have to branch on availability.
"""

from __future__ import annotations

import datetime
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Awaitable, Callable, Dict, Iterable, Optional, Tuple

from .autobiographical import (
    AutobiographicalMemory,
    _derive_cluster,
    answer_gist,
    redact_pii,
    summarise_turn_first_person,
)
from . import mood as M
from .narrative import NarrativeSummariser
from .recognition import build_wakeup_prefix
from .schema import Experience
from .state import MoodView, SelfStateThread

logger = logging.getLogger("GhostSelfhood")


CritiqueFn = Callable[[str], Awaitable[str]]


def _iso_ago(seconds: float) -> str:
    """The on-disk stamp of the moment ``seconds`` ago (UTC, "Z")."""
    try:
        at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=max(0.0, float(seconds)))
    except (TypeError, ValueError, OverflowError):
        return ""
    return at.replace(tzinfo=None, microsecond=0).isoformat() + "Z"


def owner_quiet_seconds(context, now=None) -> Optional[float]:
    """How long the owner has been quiet, from the clock the agent already
    keeps (``context.last_owner_activity_time`` — written by the turn loop's
    heartbeat for an owner request, read by the quiet-hours hold). None when
    the clock was never set (e.g. since a restart): an unknown quiet is no
    boredom. This only READS the clock. It is the FALLBACK: the mood
    measures a quiet from the owner stamp in its own state file
    (``SelfStateThread.quiet_seconds``) and asks this only while there is
    none."""
    import datetime as _dt
    last = getattr(context, "last_owner_activity_time", None)
    if not isinstance(last, _dt.datetime):
        return None
    try:
        if last <= _dt.datetime.min:
            return None
        return max(0.0, ((now or _dt.datetime.now()) - last).total_seconds())
    except Exception:  # noqa: BLE001 — an aware/naive mix is an unknown quiet
        return None


class SelfModel:
    """Top-level selfhood facade.

    Construction is cheap (no LLM call, no heavy I/O). The state file
    is read once at construct time; the autobiographical log is read
    lazily."""

    def __init__(
        self,
        root: Path,
        *,
        enabled: bool = True,
        narrative_critique_fn: Optional[CritiqueFn] = None,
        max_recent_experiences_for_narrative: int = 12,
    ):
        self.root = Path(root)
        self.enabled = bool(enabled)
        if self.enabled:
            self.autobio: Optional[AutobiographicalMemory] = AutobiographicalMemory(
                self.root, enabled=True,
            )
            self.state: Optional[SelfStateThread] = SelfStateThread(
                self.root, enabled=True,
            )
            self.narrative: Optional[NarrativeSummariser] = NarrativeSummariser(
                self.root,
                critique_fn=narrative_critique_fn,
                max_recent_experiences=max_recent_experiences_for_narrative,
                enabled=True,
            )
        else:
            self.autobio = None
            self.state = None
            self.narrative = None
        # §4TX: the reaction ledger — what the owner's reactions to ONE reply have already done to the mood
        self._ledger_lock = threading.Lock()
        #: reply (request id) → {slot: (dp, de, at, pair epoch)} — the reactions standing for it. §4UD (F1):
        #: the thumb, the correction and the thanks only — the message's TONE has a book of its own
        #: (``_tones``), so this one is byte for byte what a run that never read a tone holds (eviction included)
        self._standing: "OrderedDict[str, Dict[str, Tuple[float, float, datetime.datetime, int]]]" = OrderedDict()
        #: reply → what its reactions REALLY moved the pair-without-tone by so far, and when, and under which
        #: word of the owner (``(dp, de, at, owner_at)``) — it decays with the pair (review r2: a thumb changed
        #: after the first one wore off overshot), and an owner's set or reset since makes it void
        self._applied: Dict[str, Tuple[float, float, datetime.datetime, str]] = {}
        #: §4UD: reply → the tone of the owner's message about it (same shape as a slot), newest last
        self._tones: "OrderedDict[str, Tuple[float, float, datetime.datetime, int]]" = OrderedDict()
        #: reply → what that tone holds of the state's tone shares now (signed like the tone, same shape): the
        #: tone's MARGINAL part of the reply's verdict — what the verdict is worth with it minus without it —
        #: no bigger than the room the other replies' tones left. It comes back out when a stronger reaction
        #: replaces the tone. Dropped with its ``_tones`` row
        self._private_booked: Dict[str, Tuple[float, float, datetime.datetime, str]] = {}
        #: §4UD-b M1: replies whose thumb the owner took back (in memory, bounded) — the ONE case where the
        #: same thumb sent again (its label unchanged) is a new thumb. Cleared when a thumb is booked again
        self._taken_back: "OrderedDict[str, bool]" = OrderedDict()
        #: §4UB: when the agent last had trouble the owner saw — a turn that lowered the mood, a thumbs-down, a
        #: correction (in memory). An upbeat tone right after it may be sarcasm (``sentiment.doubtful``)
        self._trouble_at: Optional[datetime.datetime] = None
        #: §4UG: when a turn of the agent's OWN last went badly (not a thumb, not a correction) — a good
        #: turn soon after it is relief (``shade.for_turn``). In memory: a restart forgets it
        self._own_trouble_at: Optional[datetime.datetime] = None
        #: §4UG: the mood's shade — relieved / chastened / proud, for one moment, beside the label
        from .shade import ShadeStore
        self.shade = ShadeStore(self.root if self.enabled else None, enabled=self.enabled)
        #: an owner message (its request id) → the reply it answers (the owner's previous request id)
        self._answers: "OrderedDict[str, str]" = OrderedDict()
        #: conversation → its last owner reply (review r2: one global "last reply" booked a Slack correction
        #: against the last web reply); "" = a request with no conversation key
        self._last_reply: "OrderedDict[str, str]" = OrderedDict()
        #: conversation → the owner request still running in it (review r4: a message sent while the
        #: previous turn was still running was booked against the reply before that one)
        self._inflight: "OrderedDict[str, str]" = OrderedDict()

    # -----------------------------------------------------------------
    # The wake-up text (no longer spliced into any prompt — §4NM)
    # -----------------------------------------------------------------

    def build_wakeup_prefix(
        self, *, recent_experiences_n: int = 3, query: Optional[str] = None,
    ) -> str:
        """Compose the first-person wake-up text. Nothing on the request path
        calls this since the prefix left the prompt (§4NM); the functional
        harness (`scripts/selfhood_functional_test.py`) still renders it to
        inspect what the stores hold. Empty string when there's nothing to
        remember (no prior experiences AND no state).

        (§4TB: its companion `note_referenced_experiences` — which credited
        the experiences a reply echoed from this text — had no caller left
        and is removed, with the id stamp it read.)

        When ``query`` (the current user request) is supplied, the prefix
        also surfaces past experiences *relevant* to it — not just the
        most recent ones — so the agent recalls "the time I did something
        like this" rather than only "the last thing I did"."""
        if not self.enabled:
            return ""
        narrative_text = self.narrative.latest() if self.narrative is not None else ""
        return build_wakeup_prefix(
            autobio=self.autobio,
            state=self.state,
            narrative=narrative_text,
            recent_experiences_n=recent_experiences_n,
            query=query,
        )

    def recall_relevant(self, query: str, *, limit: int = 5):
        """Relevance-ranked search over my own autobiographical past.
        Returns a list of Experience records, best match first. Empty
        list when selfhood is disabled or nothing matches."""
        if not self.enabled or self.autobio is None or not query:
            return []
        try:
            return self.autobio.search_my_past(query, limit=limit)
        except Exception as e:
            logger.debug("recall_relevant skipped: %s", e)
            return []

    def capture_turn(
        self,
        *,
        trajectory_id: str,
        user_request: str,
        tool_names: Iterable[str],
        outcome: str,
        final_response: str,
        failure_reason: str = "",
        user_handle: str = "",
        cluster: Optional[str] = None,
    ) -> Optional[Experience]:
        """Write a first-person experience record for the turn that
        just finished. Never raises — selfhood capture is secondary."""
        if not self.enabled or self.autobio is None:
            return None
        try:
            tool_list = [t for t in tool_names if t]
            # Redact PII at the boundary so both the summary template
            # (which quotes the user_request) AND the stored prefix
            # share the scrubbed text. We do not redact the agent's
            # own final_response — that's the agent's own prose and
            # is far less likely to contain raw user data, and the
            # summary builder doesn't quote it verbatim.
            safe_request = redact_pii(user_request or "")
            summary = summarise_turn_first_person(
                user_request=safe_request,
                tool_names=tool_list,
                outcome=outcome,
                final_response=final_response,
                failure_reason=failure_reason,
            )
            user_short = safe_request.strip().replace("\n", " ")[:80]
            # Wire the cluster field: prefer the caller-supplied label
            # (the trajectory's own cluster), else derive a coarse topic
            # from the request so recall / narrative can generalise.
            resolved_cluster = (cluster or "").strip() or None
            if resolved_cluster is None:
                resolved_cluster = _derive_cluster(
                    f"{user_request} {summary}"
                )
            exp = Experience(
                trajectory_id=str(trajectory_id or ""),
                summary=summary,
                user_handle=str(user_handle or "")[:80],
                user_first_words=user_short,
                tools_used=tool_list[:10],
                outcome=str(outcome or "unknown"),
                cluster=resolved_cluster,
                # The reply itself, clipped and redacted — recall's
                # answer-side vocabulary (schema.Experience.answer_gist).
                answer_gist=answer_gist(final_response),
            )
            self.autobio.append(exp)
            if self.state is not None:
                self.state.touch_session()
            return exp
        except Exception as e:
            logger.debug("capture_turn skipped: %s", e)
            return None

    def mark_session_boot(self) -> None:
        """Record a session-boundary event. Idempotent within the same
        minute so a crash-restart loop doesn't flood the log."""
        if not self.enabled or self.autobio is None:
            return
        prior = ""
        if self.state is not None:
            prior = self.state.state.last_session_at
        try:
            self.autobio.mark_session_boot(prior_session_at=prior)
            if self.state is not None:
                self.state.touch_session()
        except Exception as e:
            logger.debug("mark_session_boot skipped: %s", e)

    def record_outcome(
        self, trajectory_id: str, outcome: str, *, failure_reason: str = "",
    ) -> bool:
        """Backfill a turn's verdict once the verifier / reflection layer
        has decided whether it actually succeeded. The capture path runs
        before that verdict exists, so most records start ``unknown``;
        this closes the loop so the agent's self-memory is verdict-aware.

        Never raises — backfill is secondary to the user turn."""
        if not self.enabled or self.autobio is None or not trajectory_id:
            return False
        try:
            return self.autobio.update_outcome(
                str(trajectory_id), outcome, failure_reason=failure_reason,
            )
        except Exception as e:
            logger.debug("record_outcome skipped: %s", e)
            return False

    # -----------------------------------------------------------------
    # The mood (§4TC) — read, and the ONLY writers besides set_mood
    # -----------------------------------------------------------------

    def mood(self, *, now=None, quiet_s=None, for_reply: bool = False, audience: str = "owner") -> MoodView:
        """The mood NOW (a pure read). A neutral view when selfhood is
        disabled or the read fails — a reader never has to branch.
        ``for_reply`` / ``audience``: the mood a reply hears (§4TX, ``SelfStateThread.mood``)."""
        if not self.enabled or self.state is None:
            return MoodView()
        try:
            return self.state.mood(now=now, quiet_s=quiet_s, for_reply=for_reply, audience=audience)
        except Exception as e:
            logger.debug("mood read skipped: %s: %s", type(e).__name__, e)
            return MoodView()

    def attach_owner_clock(self, quiet_fn) -> None:
        """How long the owner has been quiet (seconds, or None) — the clock
        is the agent's; the mood only reads it, and only while its own
        state file holds no owner stamp yet (boredom)."""
        if self.enabled and self.state is not None:
            self.state.attach_owner_clock(quiet_fn)

    def attach_rest(self, rest_fn) -> None:
        """§4UA: where the mood rests — the character's resting point, from
        its dials (``character.rest_pair``, §4UH-e2e). The mood only reads it."""
        if self.enabled and self.state is not None:
            self.state.attach_rest(rest_fn)

    def _lean(self, ev: Optional["M.Event"]) -> Optional["M.Event"]:
        """§4UH: the event as the character feels it (``character.lean``) — scaled ON ARRIVAL, before it is
        booked anywhere. Unchanged with the character off, its file unreadable, or on any error."""
        if ev is None:
            return None
        try:
            from . import character as _C
            if not _C.lean_enabled():
                return ev
            home = self.root.parent.parent
            if _C.unreadable(home):
                return ev
            return _C.lean(ev, _C.lean_dials(home))
        except Exception as e:  # noqa: BLE001
            logger.debug("mood lean skipped: %s: %s", type(e).__name__, e)
            return ev

    def _apply(self, ev: Optional["M.Event"], *, now=None, leaned: bool = False) -> Optional[MoodView]:
        """Put one event through ``SelfStateThread.nudge`` (leaned by the character unless the caller
        already did). Never raises — the mood is secondary to everything."""
        if ev is None or not self.enabled or self.state is None:
            return None
        if not leaned:
            ev = self._lean(ev)
        try:
            _before, after = self.state.nudge(ev.dp, ev.de, ev.cause, ev.kind,
                                             novelty=ev.novelty, now=now)
        except Exception as e:
            logger.debug("mood event skipped: %s: %s", type(e).__name__, e)
            return None
        return after

    def note_owner_turn(self, signals: "M.TurnSignals", *, now=None) -> Optional[MoodView]:
        """The end of an OWNER's turn: what happened in it moves the mood
        (``mood.turn_events``), as one move. The caller decides that the turn
        was the owner's — this method does not know who asked. A plain chat
        turn changes nothing. Returns the mood after, or None when nothing
        moved."""
        try:
            ev = M.merge_events(M.turn_events(signals))
        except Exception as e:
            logger.debug("mood turn events skipped: %s: %s", type(e).__name__, e)
            return None
        # §4UH: the character leans how hard the turn LANDS (what is booked); whether it was trouble, a
        # relief or a good turn is a fact about the turn and is judged on its own size (review r2: a playful
        # character no longer registered a failed call as trouble, and the upbeat-doubt check went blind)
        booked = self._lean(ev)
        own_age = None
        try:
            if self._own_trouble_at is not None:
                own_age = (self._now_of(now) - self._own_trouble_at).total_seconds()
        except Exception:  # noqa: BLE001
            own_age = None
        if ev is not None and ev.dp <= self.TROUBLE_MIN_DP:
            self._trouble_at = self._now_of(now)
        elif ev is not None and ev.dp > 0:
            self._trouble_at = None          # review r3: a good turn after it — the trouble is over
        after = self._apply(booked, now=now, leaned=True)
        if ev is not None:
            from . import shade as _shade
            self._set_shade(_shade.for_turn(signals, moved_dp=ev.dp, own_trouble_age_s=own_age),
                            ev.cause, after, now)
            # §4UG (review r1): the agent's own trouble, for relief, is a bad turn that MOVED the mood —
            # one the owner's set mood held is not something to be relieved about later, and a held
            # good turn does not end it
            if self._moved(after):
                if ev.dp <= self.TROUBLE_MIN_DP:
                    self._own_trouble_at = self._now_of(now)
                elif ev.dp > 0:
                    self._own_trouble_at = None
        return after

    @staticmethod
    def _moved(after: Optional[MoodView]) -> bool:
        """Did the event behind this after-view move the stored pair at all?"""
        try:
            moved = getattr(after, "moved", (0.0, 0.0))
            return abs(float(moved[0])) > 0.0 or abs(float(moved[1])) > 0.0
        except Exception:  # noqa: BLE001
            return False

    def _set_shade(self, name: str, cause: str, after: Optional[MoodView], now) -> None:
        """§4UG: set the mood's shade — only for an event that MOVED the mood (one the owner's set mood
        held sets none). Never raises."""
        try:
            if not name or after is None or not self.enabled or not self._moved(after):
                return
            self.shade.set(name, cause, now=self._now_of(now))
        except Exception as e:  # noqa: BLE001
            logger.debug("mood shade skipped: %s: %s", type(e).__name__, e)

    #: §4UB: how long after the agent's trouble an upbeat message is doubted
    TROUBLE_RECENT_S = 1800.0
    #: review r2: a turn is trouble the owner saw when it lowered the mood this much (one full failure); 1 failed
    #: call of 6 in a good turn (−0.04 net) is not
    TROUBLE_MIN_DP = -0.10

    def recent_trouble(self, now=None) -> bool:
        """§4UB: did the agent have trouble the owner saw (a turn that lowered the mood, a thumbs-down, a
        correction) in the last ``TROUBLE_RECENT_S``? In memory: a restart forgets it."""
        at = self._trouble_at
        if at is None:
            return False
        try:
            age = (self._now_of(now) - at).total_seconds()
        except Exception:  # noqa: BLE001
            return False
        return -60.0 <= age <= self.TROUBLE_RECENT_S

    def note_job_tasks_done(self, done: int, *, now=None) -> Optional[MoodView]:
        """§4UB (evaluation: project work a background job finished never registered): project tasks a job the
        OWNER started closed DONE — the task-done event only, nothing else of the job's run (its failures are
        not the owner's turn). The caller decides that the job is the owner's."""
        try:
            n = int(done or 0)
        except (TypeError, ValueError):
            return None
        if n <= 0:
            return None
        ev = M.merge_events(M.turn_events(M.TurnSignals(tasks_done=n)))
        if ev is not None:
            ev = M.Event(ev.dp, ev.de, ev.cause + " (in a background job)", ev.kind, novelty=ev.novelty)
        after = self._apply(ev, now=now)
        if ev is not None:
            from . import shade as _shade
            self._set_shade(_shade.PROUD, ev.cause, after, now)
        return after

    #: how many replies the ledger remembers (in memory: a restart forgets it, and a later reaction counts in full)
    LEDGER_REPLIES = 200
    #: why a reaction moved nothing
    HELD_ALREADY_COUNTED = "already counted for this reply"
    HELD_AT_LIMIT = "the mood is already at its limit"
    #: §4UB
    HELD_SENTIMENT_FULL = "sentiment's share of the mood is full"

    @property
    def _last_reply_id(self) -> str:
        """The last owner reply of any conversation (tests, diagnostics)."""
        return next(reversed(self._last_reply.values()), "") if self._last_reply else ""

    def _owner_word(self) -> int:
        """Which pair the ledger's amounts belong to: ``SelfStateThread.pair_epoch``, moved only when the
        owner REPLACES the pair (a set mood, a "neutral" that cleared one) — review r3: the manner moved the
        old stamp too, and a thumb changed after "be terse" counted the old thumb twice."""
        try:
            return int(getattr(self.state, "pair_epoch", 0) or 0)
        except Exception:  # noqa: BLE001
            return 0

    @staticmethod
    def _now_of(now) -> datetime.datetime:
        if now is None:
            return datetime.datetime.now(datetime.timezone.utc)
        return now if now.tzinfo is not None else now.replace(tzinfo=datetime.timezone.utc)

    @staticmethod
    def _booked_now(entry, now: datetime.datetime, owner_word: int) -> Tuple[float, float]:
        """A booked amount at ``now``: decayed with the pair; nothing once the owner set or reset the mood
        after it was booked (the pair it was part of is gone)."""
        if not entry or entry[3] != owner_word:
            return 0.0, 0.0
        elapsed = max(0.0, (now - entry[2]).total_seconds())
        return M.decay(entry[0], elapsed), M.decay(entry[1], elapsed)

    @classmethod
    def _standing_now(cls, standing, now: datetime.datetime, owner_word: int) -> Dict[str, Tuple[float, float]]:
        """The reactions standing for one reply, each decayed to ``now`` (review r3: a weaker reaction brought
        a worn-off stronger one back at full strength); none from before the owner replaced the pair."""
        out: Dict[str, Tuple[float, float]] = {}
        for slot, entry in (standing or {}).items():
            v = cls._booked_now(entry, now, owner_word)
            if v[0] or v[1]:
                out[slot] = v
        return out

    def note_owner_reaction(self, kind: str, *, now=None, request_id: str = "",
                            reply_id: str = "") -> Optional[MoodView]:
        """A thumbs-up / thumbs-down (``reply_id``: the reply it rates), a
        correction of the last reply or thanks (``request_id``: the owner's
        message that said it) — ``mood.KIND_*``. The caller decides that it
        was the OWNER's. Through the reaction ledger (``_react``)."""
        if kind in (M.KIND_THUMB_DOWN, M.KIND_CORRECTION):
            self._trouble_at = self._now_of(now)
        elif kind in (M.KIND_THUMB_UP, M.KIND_PRAISE):
            self._trouble_at = None
        after = self._react(M.reaction_event(str(kind or "")), now=now, request_id=request_id,
                            reply_id=reply_id)[0]
        from . import shade as _shade
        _ev = M.reaction_event(str(kind or ""))
        self._set_shade(_shade.for_reaction(str(kind or "")), _ev.cause if _ev else "", after, now)
        return after

    #: §4UB (evaluation: a thumb given while the owner's set mood held was dropped without a word)
    HOLD_NOTE = "Your set mood holds until {until} — this thumb is saved but doesn't move the mood."

    def reaction_hold_note(self, *, now=None) -> str:
        """§4UB: what to tell the owner when a reaction they give right now would not move the mood because
        their own set mood holds ("" when it would move it). Never raises."""
        try:
            if not self.enabled or self.state is None:
                return ""
            if self.state.hold_reason(now=now) != self.state.HELD_OWNER_HOLD:
                return ""
            end = self.state.held_until_now(now=now)
            until = end.astimezone().strftime("%H:%M") if end is not None else "the day ends"
            return self.HOLD_NOTE.format(until=until)
        except Exception:  # noqa: BLE001
            return ""

    def note_owner_sentiment(self, label: str, score: float, *, now=None, request_id: str = "",
                             arrived: str = "") -> Tuple[Optional[MoodView], float, str]:
        """§4TW: the tone of an owner's message (``mood.sentiment_event``). The caller decides that it was the
        OWNER's message. ``(mood after or None, pleasantness applied, why none)`` — §4TX: through the reaction
        ledger, and held (not applied) when the owner set the mood after the message ``arrived``.

        §4UB: what sentiment has in the pair (the state's tone shares) stays under
        ``mood.SENTIMENT_SHARE_MAX`` on each side — a tone gets the room left, and is held when there is none:
        sentiment alone never makes a heard mood. §4UD: the fit is ``_react``'s (one for every path)."""
        return self._react(M.sentiment_event(label, score), now=now, request_id=request_id, arrived=arrived)

    def thumb_was_taken_back(self, reply_id: str) -> bool:
        """§4UD-b M1: did the owner take back the thumb on this reply (since the agent started)? The feedback
        route lets a thumb whose label did not change move the mood only then: the reaction ledger is in
        memory, so after a restart the same thumb sent again (a client's retry on the restart's 5xx) would
        otherwise count again."""
        with self._ledger_lock:
            return str(reply_id or "") in self._taken_back

    def _target(self, request_id: str, reply_id: str) -> str:
        if reply_id:
            return str(reply_id)
        if request_id:
            return self._answers.get(str(request_id)) or ("msg:" + str(request_id))
        return ""

    @staticmethod
    def _total(standing: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
        """What the reactions to one reply are worth together: the strongest
        pleasant one plus the strongest unpleasant one — a thumbs-down, the
        correction that came with it and the unhappy tone of that message are
        ONE verdict, the strongest of them; a thumb changed is replaced."""
        ups = [v for v in standing.values() if v[0] > 0]
        downs = [v for v in standing.values() if v[0] < 0]
        up = max(ups, key=lambda v: v[0]) if ups else (0.0, 0.0)
        down = min(downs, key=lambda v: v[0]) if downs else (0.0, 0.0)
        return up[0] + down[0], up[1] + down[1]

    #: §4UD (G1): a thumb taken back
    THUMB_OFF_CAUSE = "the owner took a thumb back"
    NOTHING_TO_TAKE_BACK = ("no thumb of that kind stands for this reply (never given, already taken back, given "
                            "before the agent last restarted, or before the owner last set the mood)")

    @classmethod
    def _marginal(cls, standing: Dict[str, Tuple[float, float]], tone: Tuple[float, float]) -> Tuple[float, float]:
        """§4UD (F1): what a reply's TONE adds to its verdict — the verdict with it (``_total``) minus the
        verdict without it, per axis, on the tone's own side only. Nothing when a stronger reaction of the same
        side stands (a thumbs-up over an upbeat message); the difference when the tone is the stronger (an
        upbeat message over a thumbs-up that has worn down to 0.06 adds 0.02, not its whole 0.08 — review F1:
        the whole size was booked as the owner's private share, and that phantom share was then netted against
        real unhappy tone)."""
        if not tone or not tone[0]:
            return 0.0, 0.0
        with_tone = cls._total({**standing, M.KIND_SENTIMENT: tone})
        without = cls._total(standing)
        sign = 1.0 if tone[0] > 0 else -1.0
        return (sign * max(0.0, sign * (with_tone[0] - without[0])),
                sign * max(0.0, sign * (with_tone[1] - without[1])))

    def _fit(self, want: Tuple[float, float], booked: Tuple[float, float], now) -> Tuple[float, float]:
        """§4UB/§4UD: a reply's tone share no bigger than the room the OTHER replies' tones leave under the cap on
        its side — for a new tone, a second tone on the same reply, and a tone a changed or withdrawn thumb
        brings back alike. (The sink caps too; fitting here keeps what is reported and booked honest.)"""
        up_p, up_e, dn_p, dn_e = self.state.private_sides(now=now)
        out = []
        for want_i, booked_i, up, dn, cap in ((want[0], booked[0], up_p, dn_p, M.SENTIMENT_SHARE_MAX),
                                              (want[1], booked[1], up_e, dn_e, M.SENTIMENT_SHARE_MAX_E)):
            if not want_i:
                out.append(0.0)
                continue
            sign = 1.0 if want_i > 0 else -1.0
            mine = max(0.0, booked_i * sign)                    # what this reply already holds on that side
            others = max(0.0, (up if sign > 0 else dn) - mine)
            out.append(sign * max(0.0, min(abs(want_i), cap - others)))
        return out[0], out[1]

    @staticmethod
    def _side_delta(want: Tuple[float, float], booked: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """The change of the four tone shares (up p, up e, down p, down e) that takes a reply from ``booked`` to ``want``."""
        return (max(0.0, want[0]) - max(0.0, booked[0]), max(0.0, want[1]) - max(0.0, booked[1]),
                max(0.0, -want[0]) - max(0.0, -booked[0]), max(0.0, -want[1]) - max(0.0, -booked[1]))

    @staticmethod
    def _booked_after(want: float, booked: float, got_up: float, got_dn: float) -> float:
        """One axis of a reply's tone booking after the sink: where it asked for MORE, what really got in; where
        it gave back, what it asked (a share the sink no longer had is not owed)."""
        up_w, up_b, dn_w, dn_b = max(0.0, want), max(0.0, booked), max(0.0, -want), max(0.0, -booked)
        up = min(up_w, max(0.0, up_b + got_up)) if up_w > up_b else up_w
        dn = min(dn_w, max(0.0, dn_b + got_dn)) if dn_w > dn_b else dn_w
        return up - dn

    def retract_owner_reaction(self, positive: bool, *, now=None, reply_id: str = "") -> Tuple[Optional[MoodView], float, str]:
        """§4UD (G1): the owner took a thumb back (the 👍 when ``positive``, else the 👎) — it leaves the
        reply's verdict, and the mood moves by what that changes: back by what the thumb still held, less what
        the reply's other reactions now count for. A thumb that does not stand for the reply (never given,
        already taken back, of the other kind, from before the owner replaced the pair, forgotten by a restart —
        the ledger is in memory, so a thumb can be taken back only until the agent restarts) is a no-op with
        a log line. The caller decides that it was the OWNER's."""
        kind = M.KIND_THUMB_OFF
        return self._react(M.Event(0.0, 0.0, self.THUMB_OFF_CAUSE, kind), now=now, reply_id=str(reply_id or ""),
                           retract=(1.0 if positive else -1.0))

    def _react(self, ev: Optional["M.Event"], *, now=None, request_id: str = "", reply_id: str = "",
               arrived: str = "", retract: float = 0.0) -> Tuple[Optional[MoodView], float, str]:
        """§4TX (review: a thumbs-down, the correction and the unhappy tone of
        the same message counted three times; toggling a thumb drifted the
        mood down): an owner's reaction is booked against the REPLY it is
        about. The mood moves by the difference its arrival makes to that
        reply's total (``_total``), never by the reaction on its own.

        §4UD (F1): two books per reply. The thumb, the correction and the
        thanks move the pair-without-tone by the change in THEIR total — the
        same arithmetic, on the same numbers, as a run that never read a
        tone. The message's tone holds its marginal part of the verdict
        (``_marginal``) in the state's tone shares, fitted to the room left
        (``_fit``); a tone event never moves the other book. ``retract`` (±1):
        take the thumb of that sign out of the reply's slots."""
        if ev is None or not self.enabled or self.state is None:
            return None, 0.0, ""
        ev = self._lean(ev)
        try:
            why = self.state.hold_reason(now=now, arrived=arrived)
            if why:
                if retract:
                    self.state.note_held(ev.kind, ev.cause, 0.0, 0.0, why, now=now)
                else:
                    self.state.nudge(ev.dp, ev.de, ev.cause, ev.kind, novelty=ev.novelty, now=now, arrived=arrived)
                return None, 0.0, why
            at = self._now_of(now)
            is_tone = ev.kind == M.KIND_SENTIMENT
            with self._ledger_lock:
                target = self._target(request_id, reply_id)
                word = self._owner_word()
                if not target:
                    if retract:
                        logger.info("mood: a thumb taken back names no reply — nothing to take back")
                        return None, 0.0, self.NOTHING_TO_TAKE_BACK
                    # no reply to book it against: it counts as it is (a tone still goes through the sink's cap)
                    before, after = self.state.nudge(ev.dp, ev.de, ev.cause, ev.kind, novelty=ev.novelty, now=now,
                                                     arrived=arrived)
                    if after is before:
                        return None, 0.0, self.state.hold_reason(now=now, arrived=arrived)
                    return after, after.moved[0], ("" if any(after.moved) else self.HELD_AT_LIMIT)
                entries = dict(self._standing.get(target) or {})
                tone = self._tones.get(target)
                if retract:
                    thumb = self._standing_now({"thumb": entries["thumb"]}, at, word).get("thumb") \
                        if "thumb" in entries else None
                    if thumb is None or thumb[0] * retract <= 0:
                        logger.info("mood: thumb taken back on reply %s — %s (nothing changed)",
                                    str(target)[:12], self.NOTHING_TO_TAKE_BACK)
                        return None, 0.0, self.NOTHING_TO_TAKE_BACK
                    del entries["thumb"]
                    self._taken_back[str(target)] = True
                    self._taken_back.move_to_end(str(target))
                    while len(self._taken_back) > self.LEDGER_REPLIES:
                        self._taken_back.popitem(last=False)
                elif is_tone:
                    tone = (ev.dp, ev.de, at, word)
                else:
                    slot = "thumb" if ev.kind in (M.KIND_THUMB_UP, M.KIND_THUMB_DOWN) else ev.kind
                    if slot == "thumb":
                        self._taken_back.pop(str(target), None)
                    # review r2: the ledger alone says whether a thumb counts (the label store cannot: a thumb
                    # taken back leaves its label). The same thumb again REPLACES the one that stands: the
                    # mood moves by what the first has worn down by, never by a second thumb
                    entries[slot] = (ev.dp, ev.de, at, word)
                standing = self._standing_now(entries, at, word)
                dp = de = 0.0
                was = (0.0, 0.0)
                if not is_tone:
                    # the reply's own verdict, without any tone: byte for byte a tone-free run's arithmetic
                    total = self._total(standing)
                    was = self._booked_now(self._applied.get(target), at, word)
                    dp, de = round(total[0] - was[0], 4), round(total[1] - was[1], 4)
                    if abs(dp) < 0.0005 and abs(de) < 0.0005:
                        dp = de = 0.0            # already counted (a tone-free run holds it here, below)
                    self._standing[target] = entries
                    self._standing.move_to_end(target)
                    while len(self._standing) > self.LEDGER_REPLIES:
                        old, _ = self._standing.popitem(last=False)
                        self._applied.pop(old, None)
                booked = (0.0, 0.0)
                want = full = (0.0, 0.0)
                if tone is not None:
                    # the tone's part of the verdict NOW (rebuilt on every reaction — a thumb changed back, or
                    # taken back, brings it back), fitted to the room the other replies' tones leave
                    full = self._marginal(standing, self._booked_now(tone, at, word))
                    booked = self._booked_now(self._private_booked.get(target), at, word)
                    want = self._fit(full, booked, now)
                    self._tones[target] = tone
                    if is_tone:
                        self._tones.move_to_end(target)
                    while len(self._tones) > self.LEDGER_REPLIES:
                        # a forgotten tone keeps its share in the state (it decays there, under the cap); it
                        # can no longer be taken back out
                        old, _ = self._tones.popitem(last=False)
                        self._private_booked.pop(old, None)
                delta = self._side_delta(want, booked)
                if abs(dp) < 0.0005 and abs(de) < 0.0005 and all(abs(v) < 0.0005 for v in delta):
                    # (review r3: a re-booked slot of about the same size moved ±0.0001 the wrong way by rounding)
                    full_up = is_tone and abs(full[0]) - abs(want[0]) >= 0.0005
                    held = self.HELD_SENTIMENT_FULL if full_up else self.HELD_ALREADY_COUNTED
                    self.state.note_held(ev.kind, ev.cause, ev.dp, ev.de, held, now=now)
                    return None, 0.0, held
                before, after = self.state.nudge(dp, de, ev.cause, ev.kind, novelty=ev.novelty, now=now,
                                                 arrived=arrived, tone=(delta if tone is not None else None))
                if after is before:
                    # held after all (the owner set the mood in between): nothing booked
                    return None, 0.0, (self.state.hold_reason(now=now, arrived=arrived)
                                       or self.HELD_ALREADY_COUNTED)
                if dp or de:
                    # review r2: what REALLY moved (the clamp may have absorbed it), not the event's size —
                    # §4UD: of the pair WITHOUT the tone (``base_moved``)
                    self._applied[target] = (was[0] + after.base_moved[0], was[1] + after.base_moved[1], at, word)
                if tone is not None:
                    got = after.toned
                    self._private_booked[target] = (self._booked_after(want[0], booked[0], got[0], got[2]),
                                                    self._booked_after(want[1], booked[1], got[1], got[3]),
                                                    at, word)
                ap, ae = after.moved
                if ap == 0.0 and ae == 0.0:
                    # the pair is at ±1 (a tone with no room under its cap was held above, before the write)
                    return after, 0.0, self.HELD_AT_LIMIT
                return after, ap, ""
        except Exception as e:
            logger.debug("mood reaction skipped: %s: %s", type(e).__name__, e)
            return None, 0.0, ""

    def owner_arrived(self, *, now=None, request_id: str = "", conversation: str = "") -> None:
        """An owner's request has arrived. Writes down what time did to the
        mood, closes a boredom the journal holds ("the quiet ended"), and
        stamps the owner's presence in the state file. The caller decides
        that the request is the OWNER's. Never raises."""
        if not self.enabled or self.state is None:
            return
        if request_id:
            # §4TX: this message answers the owner's previous reply (the reaction ledger's key)
            with self._ledger_lock:
                key, rid = str(conversation or ""), str(request_id)
                running = self._inflight.get(key)
                self._answers[rid] = ((running if running and running != rid else "")
                                      or self._last_reply.get(key) or ("msg:" + rid))
                while len(self._answers) > self.LEDGER_REPLIES:
                    self._answers.popitem(last=False)
                self._inflight[key] = rid
                self._inflight.move_to_end(key)
                while len(self._inflight) > self.LEDGER_REPLIES:
                    self._inflight.popitem(last=False)
        try:
            self.state.owner_arrived(now=now)
        except Exception as e:
            logger.debug("mood owner-arrival skipped: %s: %s", type(e).__name__, e)

    def owner_turn_ended(self, *, now=None, request_id: str = "", conversation: str = "") -> None:
        """An owner's turn has ended: the next quiet is measured from here
        (persisted, so a restart does not change it). ``request_id``: this
        reply — the next owner message's reactions are about it. Never raises."""
        if not self.enabled or self.state is None:
            return
        if request_id:
            with self._ledger_lock:
                key = str(conversation or "")
                if self._inflight.get(key) == str(request_id):
                    del self._inflight[key]
                self._last_reply[key] = str(request_id)
                self._last_reply.move_to_end(key)
                while len(self._last_reply) > self.LEDGER_REPLIES:
                    self._last_reply.popitem(last=False)
        try:
            self.state.mark_owner_turn(now=now)
        except Exception as e:
            logger.debug("mood owner-turn stamp skipped: %s: %s", type(e).__name__, e)

    def owner_job_worked(self, *, now=None) -> None:
        """§4UD (F9): a background job the OWNER started did work in the turn that just ended — the quiet is
        measured from here (``SelfStateThread.mark_owner_job``). The caller decides that the job is the
        owner's and that the turn did work. Never raises."""
        if not self.enabled or self.state is None:
            return
        try:
            self.state.mark_owner_job(now=now)
        except Exception as e:
            logger.debug("mood owner-job stamp skipped: %s: %s", type(e).__name__, e)

    # -----------------------------------------------------------------
    # Idle-path APIs (called by biological watchdog phase 2.8)
    # -----------------------------------------------------------------

    async def consolidate_narrative(self) -> str:
        """Re-generate the running first-person narrative. Called by
        the biological watchdog phase 2.8 during idle windows.

        Returns the new narrative text (or empty string when the
        consolidation was skipped — e.g. no experiences yet)."""
        if not self.enabled or self.autobio is None or self.narrative is None:
            return ""
        try:
            return await self.narrative.regenerate(
                autobio=self.autobio, state=self.state,
            )
        except Exception as e:
            logger.warning("narrative consolidation failed: %s", e)
            return ""

    # -----------------------------------------------------------------
    # Introspection / debug
    # -----------------------------------------------------------------

    def stats(self, *, audience: str = "owner") -> dict:
        """Cheap snapshot of what the selfhood module is holding —
        useful for the introspective_consistency / consciousness_probe
        scripts and for log-line summaries. ``audience="public"`` (§4TX
        review r3: introspect on a public surface): the mood without the
        owner's own sentiment share, as the public reply hears it."""
        if not self.enabled:
            return {"enabled": False}
        _mood = self.mood(audience=audience) if audience != "owner" else self.mood()
        _st = self.state
        try:
            _rest = _st.rest() if _st is not None else (0.0, 0.0)
        except Exception:  # noqa: BLE001
            _rest = (0.0, 0.0)
        _steps = [list(x) for x in _mood.manner_steps]
        try:
            _quiet = _st.quiet_seconds() if _st is not None else None
        except Exception:  # noqa: BLE001
            _quiet = None
        return {
            "enabled": True,
            "root": str(self.root),
            "experience_count": self.autobio.count() if self.autobio else 0,
            # Boot markers, SEPARATED (introspect review M2): they were 22%
            # of "experiences on file" and 84% of the top cluster — restart
            # noise rendered as what the agent thinks about.
            "session_boots": (self.autobio.boot_count()
                              if self.autobio else 0),
            # §4TC: the mood as it is NOW — "" for a neutral one — with its
            # intensity (0–3), who set it ("auto" | "owner"), since when and
            # the latest cause; and the manner the owner set, if any.
            "last_mood": "" if _mood.neutral else _mood.label,
            "last_mood_intensity": _mood.intensity,
            "last_mood_source": _mood.source,
            "last_mood_set_at": _mood.set_at,
            "last_mood_cause": _mood.cause,
            "manner": _mood.manner,
            # the CURRENT, stepped level (0–3) — not the level it was set at
            "manner_intensity": _mood.manner_intensity,
            # §4TS: the manner's steps still ahead, (on-disk stamp, level)
            "manner_steps": _steps,
            # §4UD: when the manner's day is over (the last step), "" with no manner
            "manner_until": (str(_steps[-1][0]) if _mood.manner and _steps else ""),
            # §4UD (F6): the end of the owner's hold while it is on, "" otherwise
            "held_until": _mood.held_until,
            # §4UD (W1): THE label a face shows — the mood's, else the resting region, else "neutral"
            "face": M.face_label(_mood.label, _mood.intensity, _rest),
            # since when the owner (or a job of theirs) has been quiet, ISO; "" when unknown
            "quiet_since": (_iso_ago(_quiet) if _quiet is not None else ""),
            # §4UD (F12): "" when state.json was read; else why it was not (the state started empty)
            "load_problem": str(getattr(_st, "load_problem", "") or ""),
            "narrative_present": bool(self.narrative.latest()) if self.narrative else False,
            "last_session_at": (self.state.state.last_session_at if self.state else ""),
            "clusters": (self.autobio.cluster_counts() if self.autobio else {}),
        }
