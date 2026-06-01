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
        return build_wakeup_prefix(
            autobio=self.autobio,
            state=self.state,
            narrative=narrative_text,
            values=self.values,
            recent_experiences_n=recent_experiences_n,
            query=query,
        )

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
        return {
            "enabled": True,
            "root": str(self.root),
            "experience_count": self.autobio.count() if self.autobio else 0,
            "open_questions": len(self.state.open_questions()) if self.state else 0,
            "unfinished_threads": len(self.state.unfinished_threads()) if self.state else 0,
            "last_mood": (self.state.mood().label if self.state and self.state.mood() else ""),
            "narrative_present": bool(self.narrative.latest()) if self.narrative else False,
            "last_session_at": (self.state.state.last_session_at if self.state else ""),
            "clusters": (self.autobio.cluster_counts() if self.autobio else {}),
            "principle_count": len(self.values.principles()) if self.values else 0,
        }
