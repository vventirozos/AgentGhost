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
        #: reply (request id) → {slot: (dp, de, at, pair epoch)} — the reactions standing for it
        self._standing: "OrderedDict[str, Dict[str, Tuple[float, float, datetime.datetime, int]]]" = OrderedDict()
        #: reply → what its reactions REALLY moved the pair by so far, and when, and under which word of the
        #: owner (``(dp, de, at, owner_at)``) — it decays with the pair (review r2: a thumb changed after the
        #: first one wore off overshot), and an owner's set or reset since makes it void
        self._applied: Dict[str, Tuple[float, float, datetime.datetime, str]] = {}
        #: reply → the part of ``_applied`` the owner's message sentiment put in (same shape): it is the
        #: "private share" others do not hear, and it comes back out when a stronger reaction replaces it
        self._private_booked: Dict[str, Tuple[float, float, datetime.datetime, str]] = {}
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

    def _apply(self, ev: Optional["M.Event"], *, now=None) -> Optional[MoodView]:
        """Put one event through ``SelfStateThread.nudge``. Never raises —
        the mood is secondary to everything."""
        if ev is None or not self.enabled or self.state is None:
            return None
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
        return self._apply(ev, now=now)

    #: how many replies the ledger remembers (in memory: a restart forgets it, and a later reaction counts in full)
    LEDGER_REPLIES = 200
    #: why a reaction moved nothing
    HELD_ALREADY_COUNTED = "already counted for this reply"
    HELD_AT_LIMIT = "the mood is already at its limit"

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

    @staticmethod
    def _sentiment_counts(standing: Dict[str, Tuple[float, float]]) -> bool:
        """Is the message sentiment one of the two reactions ``_total`` keeps for this reply?"""
        v = standing.get(M.KIND_SENTIMENT)
        if not v or not v[0]:
            return False
        rest = [w[0] for k, w in standing.items() if k != M.KIND_SENTIMENT]
        if v[0] > 0:
            return all(v[0] >= w for w in rest if w > 0)
        return all(v[0] <= w for w in rest if w < 0)

    def note_owner_reaction(self, kind: str, *, now=None, request_id: str = "",
                            reply_id: str = "") -> Optional[MoodView]:
        """A thumbs-up / thumbs-down (``reply_id``: the reply it rates), a
        correction of the last reply or thanks (``request_id``: the owner's
        message that said it) — ``mood.KIND_*``. The caller decides that it
        was the OWNER's. Through the reaction ledger (``_react``)."""
        return self._react(M.reaction_event(str(kind or "")), now=now, request_id=request_id,
                           reply_id=reply_id)[0]

    def note_owner_sentiment(self, label: str, score: float, *, now=None, request_id: str = "",
                             arrived: str = "") -> Tuple[Optional[MoodView], float, str]:
        """§4TW: the tone of an owner's message (``mood.sentiment_event``). The caller decides that it was the
        OWNER's message. ``(mood after or None, pleasantness applied, why none)`` — §4TX: through the reaction
        ledger, and held (not applied) when the owner set the mood after the message ``arrived``."""
        return self._react(M.sentiment_event(label, score), now=now, request_id=request_id, arrived=arrived)

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

    def _react(self, ev: Optional["M.Event"], *, now=None, request_id: str = "", reply_id: str = "",
               arrived: str = "") -> Tuple[Optional[MoodView], float, str]:
        """§4TX (review: a thumbs-down, the correction and the unhappy tone of
        the same message counted three times; toggling a thumb drifted the
        mood down): an owner's reaction is booked against the REPLY it is
        about. The mood moves by the difference its arrival makes to that
        reply's total (``_total``), never by the reaction on its own."""
        if ev is None or not self.enabled or self.state is None:
            return None, 0.0, ""
        try:
            why = self.state.hold_reason(now=now, arrived=arrived)
            if why:
                self.state.nudge(ev.dp, ev.de, ev.cause, ev.kind, novelty=ev.novelty, now=now, arrived=arrived)
                return None, 0.0, why
            at = self._now_of(now)
            with self._ledger_lock:
                target = self._target(request_id, reply_id)
                word = self._owner_word()
                was = booked = (0.0, 0.0)
                adjust: Optional[Tuple[float, float]] = None
                private = (0.0, 0.0)
                if not target:
                    dp, de = ev.dp, ev.de
                else:
                    slot = "thumb" if ev.kind in (M.KIND_THUMB_UP, M.KIND_THUMB_DOWN) else ev.kind
                    entries = dict(self._standing.get(target) or {})
                    entries[slot] = (ev.dp, ev.de, at, word)
                    standing = self._standing_now(entries, at, word)
                    total = self._total(standing)
                    was = self._booked_now(self._applied.get(target), at, word)
                    dp, de = round(total[0] - was[0], 4), round(total[1] - was[1], 4)
                    # the owner's own message tone is private while it is one of the two that count
                    # (review r3: rebuilt on every reaction — a thumb changed back brings it back)
                    if self._sentiment_counts(standing):
                        private = standing[M.KIND_SENTIMENT]
                    booked = self._booked_now(self._private_booked.get(target), at, word)
                    adjust = (round(private[0] - booked[0], 4), round(private[1] - booked[1], 4))
                    self._standing[target] = entries
                    self._standing.move_to_end(target)
                    while len(self._standing) > self.LEDGER_REPLIES:
                        old, _ = self._standing.popitem(last=False)
                        self._applied.pop(old, None)
                        self._private_booked.pop(old, None)
                if dp == 0.0 and de == 0.0:
                    self.state.note_held(ev.kind, ev.cause, ev.dp, ev.de, self.HELD_ALREADY_COUNTED, now=now)
                    return None, 0.0, self.HELD_ALREADY_COUNTED
                before, after = self.state.nudge(dp, de, ev.cause, ev.kind, novelty=ev.novelty, now=now,
                                                 arrived=arrived, private_adjust=adjust)
                if after is before:
                    # held after all (the owner set the mood in between): nothing booked
                    return None, 0.0, (self.state.hold_reason(now=now, arrived=arrived)
                                       or self.HELD_ALREADY_COUNTED)
                ap, ae = round(after.p - before.p, 4), round(after.e - before.e, 4)
                if ev.kind == M.KIND_SENTIMENT:
                    # review r5/r6: book what the state really stored — the change scaled by what the clamp
                    # let through, on top of what was booked
                    rp = max(0.0, min(1.0, ap / dp)) if dp else 0.0
                    re_ = max(0.0, min(1.0, ae / de)) if de else 0.0
                    private = (booked[0] + (private[0] - booked[0]) * rp,
                               booked[1] + (private[1] - booked[1]) * re_)
                if target:
                    # review r2: what REALLY moved (the clamp may have absorbed it), not the event's size
                    self._applied[target] = (was[0] + ap, was[1] + ae, at, word)
                    self._private_booked[target] = (private[0], private[1], at, word)
                if ap == 0.0 and ae == 0.0:
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
            "manner_intensity": _mood.manner_intensity,
            # §4TS: the manner's steps still ahead, (on-disk stamp, level)
            "manner_steps": [list(x) for x in _mood.manner_steps],
            "narrative_present": bool(self.narrative.latest()) if self.narrative else False,
            "last_session_at": (self.state.state.last_session_at if self.state else ""),
            "clusters": (self.autobio.cluster_counts() if self.autobio else {}),
        }
