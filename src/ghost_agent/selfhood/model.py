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

import logging
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Optional

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

    def mood(self, *, now=None, quiet_s=None) -> MoodView:
        """The mood NOW (a pure read). A neutral view when selfhood is
        disabled or the read fails — a reader never has to branch."""
        if not self.enabled or self.state is None:
            return MoodView()
        try:
            return self.state.mood(now=now, quiet_s=quiet_s)
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

    def note_owner_reaction(self, kind: str, *, now=None) -> Optional[MoodView]:
        """A thumbs-up / thumbs-down, a correction of the last reply, thanks
        (``mood.KIND_*``). The caller decides that it was the OWNER's."""
        return self._apply(M.reaction_event(str(kind or "")), now=now)

    def owner_arrived(self, *, now=None) -> None:
        """An owner's request has arrived. Writes down what time did to the
        mood, closes a boredom the journal holds ("the quiet ended"), and
        stamps the owner's presence in the state file. The caller decides
        that the request is the OWNER's. Never raises."""
        if not self.enabled or self.state is None:
            return
        try:
            self.state.owner_arrived(now=now)
        except Exception as e:
            logger.debug("mood owner-arrival skipped: %s: %s", type(e).__name__, e)

    def owner_turn_ended(self, *, now=None) -> None:
        """An owner's turn has ended: the next quiet is measured from here
        (persisted, so a restart does not change it). Never raises."""
        if not self.enabled or self.state is None:
            return
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

    def stats(self) -> dict:
        """Cheap snapshot of what the selfhood module is holding —
        useful for the introspective_consistency / consciousness_probe
        scripts and for log-line summaries."""
        if not self.enabled:
            return {"enabled": False}
        _mood = self.mood()
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
            "narrative_present": bool(self.narrative.latest()) if self.narrative else False,
            "last_session_at": (self.state.state.last_session_at if self.state else ""),
            "clusters": (self.autobio.cluster_counts() if self.autobio else {}),
        }
