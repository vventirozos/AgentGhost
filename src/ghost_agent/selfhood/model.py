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
    detect_referenced_experiences,
    redact_pii,
    summarise_turn_first_person,
)
from .mood import (
    SELF_MOOD_GRACE_HOURS,
    STREAK_MAX_AGE_DAYS,
    STREAK_WINDOW,
    MoodSignals,
    age_seconds,
    derive_mood,
    mood_is_stale,
)
from .narrative import NarrativeSummariser
from .recognition import build_wakeup_prefix
from .schema import Experience
from .state import SelfStateThread
from .values import ValuesThread

logger = logging.getLogger("GhostSelfhood")


CritiqueFn = Callable[[str], Awaitable[str]]


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
            # Normative substrate (operating principles) — the values
            # layer that moves selfhood from episodic-only to
            # behaviour-shaping. Surfaced in the wake-up prefix every turn.
            self.values: Optional[ValuesThread] = ValuesThread(
                self.root, enabled=True,
            )
        else:
            self.autobio = None
            self.state = None
            self.narrative = None
            self.values = None
        # Ids of the experiences the most recent wake-up prefix surfaced.
        # note_referenced_experiences unions these into its candidate pool
        # — the IDF-retrieved entries in the prefix are precisely the ones
        # likely older than the recent(50) window it scans.
        self._last_prefix_experience_ids: tuple = ()

    # -----------------------------------------------------------------
    # Hot-path APIs (called by handle_chat per turn)
    # -----------------------------------------------------------------

    def build_wakeup_prefix(
        self, *, recent_experiences_n: int = 3, query: Optional[str] = None,
    ) -> str:
        """Compose the first-person wake-up text the prompt assembly
        path splices into the system prompt. Empty string when there's
        nothing to remember (no prior experiences AND no state).

        When ``query`` (the current user request) is supplied, the prefix
        also surfaces past experiences *relevant* to it — not just the
        most recent ones — so the agent recalls "the time I did something
        like this" rather than only "the last thing I did"."""
        if not self.enabled:
            return ""
        narrative_text = self.narrative.latest() if self.narrative is not None else ""
        prefix = build_wakeup_prefix(
            autobio=self.autobio,
            state=self.state,
            narrative=narrative_text,
            values=self.values,
            recent_experiences_n=recent_experiences_n,
            query=query,
        )
        self._last_prefix_experience_ids = self._prefix_experience_ids(
            recent_experiences_n=recent_experiences_n, query=query,
        )
        return prefix

    def _prefix_experience_ids(
        self,
        *,
        recent_experiences_n: int,
        query: Optional[str],
        relevant_experiences_n: int = 3,
    ) -> tuple:
        """Ids of the experiences the wake-up prefix surfaces — mirrors
        the recent+relevant retrieval in ``recognition.build_wakeup_prefix``
        (which renders text, not ids). Cheap: ``recent`` is a bounded tail
        read and the search rides the (mtime, size)-keyed index cache the
        prefix build just warmed. Never raises."""
        if self.autobio is None:
            return ()
        ids: list = []
        try:
            if recent_experiences_n > 0:
                ids = [e.id for e in self.autobio.recent(limit=recent_experiences_n)]
            if query and query.strip() and relevant_experiences_n > 0:
                seen = set(ids)
                relevant = [
                    e for e in self.autobio.search_my_past(
                        query, limit=relevant_experiences_n + len(seen),
                    )
                    if e.id not in seen
                ][:relevant_experiences_n]
                ids.extend(e.id for e in relevant)
        except Exception as e:
            logger.debug("prefix experience-id stamp skipped: %s", e)
        return tuple(ids)

    # -----------------------------------------------------------------
    # Values / principles (normative substrate)
    # -----------------------------------------------------------------

    def note_principle(self, text: str):
        """Author an operating principle. Returns the Principle (or None
        when disabled / empty). Surfaced in the wake-up prefix every turn."""
        if not self.enabled or self.values is None:
            return None
        try:
            return self.values.note_principle(text)
        except Exception as e:
            logger.debug("note_principle skipped: %s", e)
            return None

    def principles(self):
        """Current operating principles (list of Principle). Empty when
        disabled / none authored."""
        if not self.enabled or self.values is None:
            return []
        try:
            return self.values.principles()
        except Exception:
            return []

    def principles_text(self) -> str:
        """Bulleted principle list for a self-critique gate prompt."""
        if not self.enabled or self.values is None:
            return ""
        try:
            return self.values.as_text()
        except Exception:
            return ""

    async def evaluate_response_alignment(self, response: str, *, critique_fn):
        """Independent check that ``response`` doesn't contradict a stated
        operating principle. Returns ``(aligned, note)``. ``aligned`` is
        True when there are no principles, no response, or no critique_fn
        (fail-open — the gate must never block a turn on its own absence).

        ``critique_fn`` is an async ``str -> str`` (the same shape the
        Reflector / narrative summariser use); it is given the principles
        and the response and asked for an ALIGNED/VIOLATION verdict."""
        principles = self.principles_text()
        if not principles or not (response or "").strip() or critique_fn is None:
            return True, ""
        prompt = (
            "You are auditing whether a response honours the agent's own "
            "stated operating principles.\n\nPRINCIPLES:\n"
            f"{principles}\n\nRESPONSE:\n{str(response)[:2000]}\n\n"
            "Does the response CONTRADICT any principle? Reply on the first "
            "line with exactly 'VERDICT: ALIGNED' or 'VERDICT: VIOLATION', "
            "then one sentence naming the principle if violated."
        )
        try:
            out = await critique_fn(prompt)
            up = (out or "").upper()
            v_pos = up.find("VIOLATION")
            a_pos = up.find("ALIGNED")
            violated = v_pos != -1 and (a_pos == -1 or v_pos < a_pos)
            lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
            note = (lines[0] if lines else "")[:200]
            return (not violated), note
        except Exception as e:
            logger.debug("evaluate_response_alignment skipped: %s", e)
            return True, ""

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

    def note_referenced_experiences(
        self, *, prefix_text: str, response_text: str,
    ) -> int:
        """Detect which experiences from the wake-up prefix were
        actually echoed in the agent's response, and bump their
        reference counters. Returns the number of experiences whose
        counter was incremented (0 when disabled / no match).

        The detector is a pure function — see
        ``autobiographical.detect_referenced_experiences``. Reference
        counts get persisted to ``reference_counts.json`` so the
        signal survives process restarts."""
        if not self.enabled or self.autobio is None:
            return 0
        if not prefix_text or not response_text:
            return 0
        try:
            recent_pool = self.autobio.recent(limit=50)
        except Exception:
            recent_pool = []
        # Union in the experiences the wake-up prefix actually surfaced:
        # the IDF-retrieved ones are exactly the entries likely older than
        # the newest 50, and a recall-surfaced memory the model echoed
        # must be creditable too.
        stamped = getattr(self, "_last_prefix_experience_ids", ())
        if stamped:
            try:
                pool_ids = {e.id for e in recent_pool}
                missing = [i for i in stamped if i not in pool_ids]
                if missing:
                    recent_pool = recent_pool + self.autobio.get_by_ids(missing)
            except Exception as e:
                logger.debug("prefix-id pool union skipped: %s", e)
        try:
            ids = detect_referenced_experiences(
                prefix_text=prefix_text,
                response_text=response_text,
                experiences=recent_pool,
            )
        except Exception as e:
            logger.debug("ref-detection failed: %s", e)
            return 0
        for eid in ids:
            try:
                self.autobio.record_reference(eid)
            except Exception:
                continue
        return len(ids)

    def stale_open_questions(self, *, max_age_days: float = 3.0):
        """Surface open questions that have been carrying for more than
        ``max_age_days``. Used by an idle hook to prompt the agent to
        revisit, refile, or resolve them so the open-questions list
        doesn't become write-only."""
        if not self.enabled or self.state is None:
            return []
        try:
            return self.state.stale_open_questions(max_age_days=max_age_days)
        except Exception as e:
            logger.debug("stale_open_questions skipped: %s", e)
            return []

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

    def update_derived_mood(
        self,
        *,
        pressure_lockdown: bool = False,
        idle_seconds: float = 0.0,
        operator_turn: bool = False,
        now=None,
    ) -> Optional["object"]:
        """Recompute the functional mood from live signals and persist
        it when unambiguous (mood rework 2026-08-20).

        Called from two sites, both origin-gated to REAL traffic
        (``turn_origin == "user"`` — checked inline at the post-turn
        site and at the phase gate for 2.8c):
          - the post-turn capture path (after ``capture_turn``, inside
            ``_record_turn_trajectory`` — so it requires a wired
            trajectory collector, and on STREAMED turns the verifier
            verdict lands late, meaning the streak can lag one turn),
            with this turn's context-pressure lockdown flag;
          - biological-watchdog phase 2.8c, with the idle clock — the
            only path that can derive "idle", and what keeps the
            staleness TTL honest across quiet stretches.

        Contract:
          - ambiguous signals → no write at all (the prior mood stands
            and its staleness clock keeps running — never fabricate a
            neutral);
          - a self-authored mood younger than ``SELF_MOOD_GRACE_HOURS``
            is respected, not clobbered (otherwise the ``set_mood``
            tool action would survive at most one turn);
          - a derived ACUTE mood ("idle"/"overloaded") is only replaced
            or retired by a caller entitled to falsify it (acute-hold +
            retirement rules; ``operator_turn`` distinguishes real
            operator turns from scheduled/job-resume turns via
            ``is_internal_request`` at the hook);
          - re-deriving the same label refreshes ``set_at`` without a
            history append (``SelfStateThread.set_mood`` dedup);
          - transitions are pretty-logged (🪞) so the operator sees
            mood changes in the live stream.

        Returns the persisted Mood on write, None otherwise. Never
        raises — mood is a gauge, secondary to everything."""
        if not self.enabled or self.state is None:
            return None
        try:
            # Normalize ONCE (review R5): derive, retirement, and the
            # acute-hold must all read the same coerced value — a
            # caller passing None would otherwise get post-turn
            # semantics in one check and tick semantics in another.
            idle_seconds = float(idle_seconds or 0.0)
            pressure_lockdown = bool(pressure_lockdown)
            operator_turn = bool(operator_turn)
            # Signal gathering needs no lock — the streak and questions
            # are inputs, not the record being read-modified.
            outcomes: list = []
            if self.autobio is not None:
                # Verdict-bearing back-scan, NOT a raw-tail read: the
                # raw tail is boot markers + unknown-verdict turns
                # almost everywhere (live mix ≈ 19% verdicts, newest
                # verdict observed 78 records back) — a recent(16) read
                # starved the streak to zero and made stuck/satisfied
                # unreachable (review R2). Age-bounded so a streak of
                # weeks-old verdicts can't keep minting a "fresh" mood
                # via the heartbeat set_at refresh (review R3).
                outcomes = self.autobio.recent_verdicts(
                    limit=STREAK_WINDOW,
                    max_age_days=STREAK_MAX_AGE_DAYS,
                    now=now,
                )

            open_qs = self.state.open_questions()
            newest_age_h = None
            ages = [
                a for a in (
                    age_seconds(q.opened_at, now=now) for q in open_qs
                ) if a is not None
            ]
            if ages:
                newest_age_h = min(ages) / 3600.0

            verdict = derive_mood(MoodSignals(
                outcomes=outcomes,
                pressure_lockdown=pressure_lockdown,
                open_question_count=len(open_qs),
                newest_question_age_hours=newest_age_h,
                idle_seconds=idle_seconds,
            ))
            if verdict is None:
                # Acute-state retirement (review R2): "idle" and
                # "overloaded" are one-observation states. When the
                # very signal that minted them is now absent — a user
                # turn just completed (so not idle), a turn ended
                # cleanly (so not locked down) — the reading is
                # FALSIFIED, not merely unrefreshed; letting it stand
                # for the 48h TTL re-creates the stale-label defect at
                # smaller scale. Streak states (stuck/satisfied) and
                # curious are standing assessments and age out
                # normally. Retiring is not fabricating a neutral —
                # it's clearing a gauge whose basis is gone.
                self._maybe_retire_acute_mood(
                    pressure_lockdown=pressure_lockdown,
                    idle_seconds=idle_seconds,
                    operator_turn=operator_turn,
                )
                return None
            label, evidence = verdict

            # Prior-read → grace check → transition detection → write,
            # all under the state's RLock (re-entrant: set_mood takes it
            # again). Today every mood writer runs on the event-loop
            # thread, but that invariant is one asyncio.to_thread
            # refactor away from false (the neighboring selfhood calls
            # already run off-thread) — an unlocked read-modify-write
            # here would let a concurrent tool set_mood land between
            # read and write, clobbering the grace window and
            # mislabelling the logged transition (review R1).
            with self.state.lock:
                prior = self.state.mood()
                if (prior is not None
                        and getattr(prior, "source", "self") == "self"
                        and not mood_is_stale(prior.set_at, now=now)):
                    age = age_seconds(prior.set_at, now=now)
                    if (age is not None
                            and age < SELF_MOOD_GRACE_HOURS * 3600.0):
                        return None
                # Acute-hold (review R4/R5): a FRESH derived ACUTE mood
                # may only be REPLACED by a caller entitled to falsify
                # it — otherwise the lockdown-blind idle tick overwrote
                # "overloaded" with a streak label 40 minutes after the
                # mint, with zero new events. Rules:
                #   - a STALE acute prior no longer deserves the hold
                #     (it is already dropped/flagged everywhere; letting
                #     it veto a live decisive derivation left the gauge
                #     dark indefinitely — R5);
                #   - "overloaded": only a completed turn may act on it,
                #     and only an OPERATOR turn may mint the replacement
                #     label; an internal clean turn instead RETIRES it
                #     (it falsifies "the last turn ended under lockdown"
                #     but must not launder an operator-facing standing
                #     label into place — R5: idle→overloaded→satisfied
                #     via two internal turns displaced "idle" with the
                #     operator away all night);
                #   - "idle": an operator turn may replace it, and a
                #     fresher acute "overloaded" from any completed turn
                #     outranks it.
                # Same-label re-derivation (heartbeat) always allowed.
                retired_over_label = ""
                if (prior is not None
                        and getattr(prior, "source", "self") == "derived"
                        and prior.label in ("idle", "overloaded")
                        and label != prior.label
                        and not mood_is_stale(prior.set_at, now=now)):
                    if prior.label == "overloaded":
                        if idle_seconds != 0.0:
                            return None
                        if not operator_turn:
                            retired_over_label = prior.label
                            self.state.clear_mood()
                    elif (prior.label == "idle"
                            and not (idle_seconds == 0.0
                                     and (operator_turn
                                          or label == "overloaded"))):
                        return None
                if retired_over_label:
                    mood = None
                    transition = False
                else:
                    transition = not (
                        prior is not None and prior.label == label
                        and getattr(prior, "source", "self") == "derived"
                    )
                    mood = self.state.set_mood(
                        label, evidence, source="derived")
            if retired_over_label:
                # Internal clean turn falsified "overloaded": retired,
                # nothing minted (see the acute-hold comment above).
                try:
                    from ..utils.logging import Icons, pretty_log
                    pretty_log(
                        "Selfhood",
                        f"mood {retired_over_label} retired — its basis "
                        "no longer holds",
                        icon=Icons.SELF_STATE,
                    )
                except Exception:
                    pass
                return None
            if mood is not None and transition:
                # Guarded separately: the write above already happened,
                # and a logging hiccup must not turn "wrote" into the
                # caller seeing None.
                try:
                    from ..utils.logging import Icons, pretty_log
                    if prior is not None and prior.label == label:
                        # Provenance flip (self→derived on the same
                        # label): "mood curious → curious" reads as a
                        # bug in the live stream — say what happened.
                        msg = (f"mood {label} re-derived from live "
                               f"signals (was self-noted): {evidence}")
                    else:
                        prior_label = (prior.label if prior is not None
                                       else "(none)")
                        msg = f"mood {prior_label} → {label}: {evidence}"
                    pretty_log("Selfhood", msg, icon=Icons.SELF_STATE)
                except Exception:
                    pass
            return mood
        except Exception as e:
            logger.debug("update_derived_mood skipped: %s: %s",
                         type(e).__name__, e)
            return None

    def _maybe_retire_acute_mood(
        self, *, pressure_lockdown: bool, idle_seconds: float,
        operator_turn: bool = False,
    ) -> bool:
        """Clear a derived acute mood whose basis is falsified (see the
        call site in ``update_derived_mood``). Only ever touches
        ``source="derived"`` moods — a self-authored label is the
        agent's own statement and ages out via grace + TTL instead.

        Retirement happens ONLY on the post-turn path (idle_seconds ==
        0.0, i.e. a turn just completed). "overloaded" is turn-scoped:
        ANY cleanly-completed turn falsifies "the last turn ended under
        lockdown". "idle" is operator-scoped and additionally requires
        ``operator_turn`` — scheduled tasks and job resumes run this
        same hook at 3am while the operator is genuinely away, and
        letting them retire "idle" re-created the retire→re-derive
        flap (review R4; the hook derives operator_turn from
        ``is_internal_request(req_id)``). The idle tick can NEVER
        retire: its clock is saw-toothed by phase 3's
        last_activity_time reset in the self-play finally, so a small
        idle reading there does not prove the operator returned
        (probe-proven flap, review R3).
        Returns True when a mood was cleared. Never raises."""
        try:
            if self.state is None:
                return False
            if idle_seconds != 0.0:
                return False
            with self.state.lock:
                prior = self.state.mood()
                if (prior is None
                        or getattr(prior, "source", "self") != "derived"):
                    return False
                # The lockdown guard is unreachable from the current
                # call site (derive_mood returns "overloaded" whenever
                # the flag is set, so the verdict-None branch implies
                # it's off) but is part of this helper's OWN contract —
                # a caller passing lockdown=True must not retire the
                # reading it just confirmed. Pinned by a direct test.
                falsified = (
                    (prior.label == "idle" and operator_turn)
                    or (prior.label == "overloaded"
                        and not pressure_lockdown)
                )
                if not falsified:
                    return False
                cleared = self.state.clear_mood()
            if cleared:
                try:
                    from ..utils.logging import Icons, pretty_log
                    pretty_log(
                        "Selfhood",
                        f"mood {prior.label} retired — its basis no "
                        "longer holds",
                        icon=Icons.SELF_STATE,
                    )
                except Exception:
                    pass
            return cleared
        except Exception as e:
            logger.debug("acute-mood retirement skipped: %s: %s",
                         type(e).__name__, e)
            return False

    # -----------------------------------------------------------------
    # Idle-path APIs (called by biological watchdog phase 2.8)
    # -----------------------------------------------------------------

    async def consolidate_narrative(self, *, meta_insights: str = "") -> str:
        """Re-generate the running first-person narrative. Called by
        the biological watchdog phase 2.8 during idle windows.

        ``meta_insights`` lets the caller fold in cross-phase learning —
        heuristics the dream phase consolidated, failure patterns the
        reflection phase found — so the diary becomes self-knowledge,
        not just an experience log.

        Returns the new narrative text (or empty string when the
        consolidation was skipped — e.g. no experiences yet)."""
        if not self.enabled or self.autobio is None or self.narrative is None:
            return ""
        try:
            return await self.narrative.regenerate(
                autobio=self.autobio, state=self.state,
                meta_insights=meta_insights,
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
        _mood = self.state.mood() if self.state else None
        return {
            "enabled": True,
            "root": str(self.root),
            "experience_count": self.autobio.count() if self.autobio else 0,
            # Boot markers, SEPARATED (introspect review M2): they were 22%
            # of "experiences on file" and 84% of the top cluster — restart
            # noise rendered as what the agent thinks about.
            "session_boots": (self.autobio.boot_count()
                              if self.autobio else 0),
            "open_questions": len(self.state.open_questions()) if self.state else 0,
            "unfinished_threads": len(self.state.unfinished_threads()) if self.state else 0,
            "last_mood": _mood.label if _mood else "",
            # Provenance + freshness (mood rework 2026-08-20) so the
            # introspection surfaces can render honest age/stale flags
            # instead of presenting an old label as current.
            "last_mood_source": (getattr(_mood, "source", "self")
                                 if _mood else ""),
            "last_mood_set_at": (_mood.set_at if _mood else ""),
            "narrative_present": bool(self.narrative.latest()) if self.narrative else False,
            "last_session_at": (self.state.state.last_session_at if self.state else ""),
            "clusters": (self.autobio.cluster_counts() if self.autobio else {}),
            "principle_count": len(self.values.principles()) if self.values else 0,
        }
