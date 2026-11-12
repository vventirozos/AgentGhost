"""Narrative summariser — proposal item #5.

Periodically (idle window), re-write the first-person running summary
that the wake-up layer reads. The point isn't to compress the
autobiographical log — it's to produce a *voice*: a paragraph or two
of "what I've been doing lately, why it matters, what's still open"
written in first person, so the next session's wake-up prefix reads
like a continuous diary entry rather than a heap of bullet points.

Storage:
  - ``$GHOST_HOME/system/selfhood/narrative.md``      — latest (overwritten)
  - ``$GHOST_HOME/system/selfhood/narrative.history.jsonl`` — append-only audit

The narrative phase is the only LLM-touching component of the
selfhood module. The other components are template / file-IO only,
so the module works (just less richly) without an upstream LLM.
"""

from __future__ import annotations

from ..utils.json_store import open_append  # torn-tail-safe JSONL appends (§4MF)
import datetime
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

from .autobiographical import AutobiographicalMemory
from .schema import Experience, _utcnow_iso
from .mood import KIND_MANNER, KIND_QUIET, KIND_QUIET_END
from .state import SelfStateThread

logger = logging.getLogger("GhostSelfhood")


NARRATIVE_FILENAME = "narrative.md"
NARRATIVE_HISTORY_FILENAME = "narrative.history.jsonl"

# Bounded growth for the audit history (mirrors autobiographical's
# _COMPACT_MAX_BYTES pattern). Each record embeds the full narrative
# text, so an idle agent's periodic regenerations grow the file without
# bound. Past the byte cap we keep only the newest records.
_HISTORY_COMPACT_MAX_BYTES = 512 * 1024
_HISTORY_COMPACT_KEEP_LINES = 100


# A user request shorter than this — with no tools and no verdict — is
# treated as trivial chat ("pong", "what is this ?", "hello") and kept
# out of the diary input. Observed live (2026-07-13): a night of idle
# regenerations produced a narrative whose opening line was 'Lately, I
# worked on "reply with just: pong"' because ping-shaped turns dominated
# the most-recent-N window.
_INFORMATIVE_MIN_REQUEST_CHARS = 40


def _is_informative_experience(exp: Experience) -> bool:
    """True when an experience carries diary-worthy signal.

    Kept deliberately permissive: any tool use, any real verdict, a
    substantial request, or a session-boot marker (those structure the
    diary — "after a few hours off…") all qualify. Only the no-tools /
    no-verdict / short-request combination is filtered.
    """
    if (getattr(exp, "outcome", "") or "") == "boot":
        return True
    if getattr(exp, "tools_used", None):
        return True
    if (getattr(exp, "outcome", "") or "") in ("passed", "failed"):
        return True
    req = (getattr(exp, "user_first_words", "") or "").strip()
    return len(req) >= _INFORMATIVE_MIN_REQUEST_CHARS


# Critique-style prompt: the model is asked to write a first-person
# diary entry from its OWN past, not a third-person summary of someone
# else's actions. The framing matters — a third-person summary
# produces external-knowledge feel; first-person produces continuity
# feel. Kept compact so the LLM round-trip is cheap.
NARRATIVE_PROMPT = """You are continuing a personal diary written by an AI agent
across many sessions. Below are recent things you wrote about, your
current state, and patterns observed across your own work.

Write a SHORT first-person diary entry (1-3 paragraphs, plain prose,
no bullet points, no headings). Use "I" / "my". Reflect honestly
on the patterns — what you keep doing well, and where you keep
slipping. End with what feels unresolved — one or two sentences about
what you still want to figure out. Do not summarise this prompt; just
write the entry.

--- RECENT EXPERIENCES (newest last) ---
{experiences}

--- CURRENT STATE ---
{state}

--- PATTERNS & WHAT I'VE NOTICED ABOUT MYSELF ---
{patterns}

--- DIARY ENTRY ---
"""


CritiqueFn = Callable[[str], Awaitable[str]]


#: how many of the day's mood steps the diary prompt carries
_MOOD_TIMELINE_STEPS = 8

class NarrativeSummariser:
    """Periodic LLM-driven first-person summary.

    The class is sync-construction / async-run because the LLM call is
    awaited but everything else is file I/O."""

    def __init__(
        self,
        root: Path,
        *,
        critique_fn: Optional[CritiqueFn] = None,
        max_recent_experiences: int = 12,
        enabled: bool = True,
    ):
        self.root = Path(root)
        self.path = self.root / NARRATIVE_FILENAME
        self.history_path = self.root / NARRATIVE_HISTORY_FILENAME
        self.critique_fn = critique_fn
        self.max_recent_experiences = max(1, int(max_recent_experiences))
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        # Fingerprint of the last successfully-persisted regeneration's
        # INPUT (the rendered prompt). Lets the idle phase skip when
        # nothing new happened — observed live: ~15 identical hourly
        # regenerations overnight, each an LLM round-trip. Persisted next
        # to the narrative (§4LF: in memory, every one of 285 boots in four
        # weeks regenerated it on the main model).
        self._last_input_key = self._load_input_key()

    # -----------------------------------------------------------------
    # Read path
    # -----------------------------------------------------------------

    # §4LF: the "input unchanged" key survives a restart — kept in memory
    # only, every one of 285 boots in 4 weeks re-generated the narrative
    # (213 of 354 rewrites were the first after a boot, on the main model)
    def _key_path(self):
        return self.path.with_name(self.path.name + ".inputkey")

    def _load_input_key(self) -> str:
        try:
            return self._key_path().read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001 — no key: regenerate once
            return ""

    def _remember_input_key(self, key: str) -> None:
        self._last_input_key = key
        try:
            self._key_path().write_text(key, encoding="utf-8")
        except Exception:  # noqa: BLE001 — only costs one extra regeneration
            pass

    def latest(self) -> str:
        """Return the most recently written narrative, or empty string
        when none has been generated yet."""
        if not self.path.exists():
            return ""
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("narrative read failed: %s", e)
            return ""

    # -----------------------------------------------------------------
    # Write path
    # -----------------------------------------------------------------

    def _format_experiences_block(
        self,
        experiences: List[Experience],
        *,
        relevant: Optional[List[Experience]] = None,
    ) -> str:
        if not experiences:
            return "(no recent experiences)"
        lines = []
        for exp in experiences:
            tag = f"[{exp.outcome}]" if exp.outcome and exp.outcome != "unknown" else ""
            lines.append(f"- {exp.timestamp} {tag} {exp.summary}".strip())
        if relevant:
            lines.append("")
            lines.append(
                "(older entries that connect to how I am now:)"
            )
            for exp in relevant:
                tag = f"[{exp.outcome}]" if exp.outcome and exp.outcome != "unknown" else ""
                lines.append(f"- {exp.timestamp} {tag} {exp.summary}".strip())
        return "\n".join(lines)

    @staticmethod
    def _mood_steps(state: Optional[SelfStateThread]) -> List[dict]:
        """Today's rows of the mood journal as the diary reads them: what
        HAPPENED to the mood (an event, the owner's word, a wearing off) —
        not the boredom rows a reader adds while the owner is away. The
        diary's prompt hash is its regenerate skip-guard: read off the
        journal it moves only when the mood was moved, never because a
        quiet reached its next level (review, §4TC: three diary calls for
        one quiet stretch)."""
        if state is None:
            return []
        try:
            return [row for row in (state.mood_timeline() or [])
                    if row.get("kind") not in (KIND_QUIET, KIND_QUIET_END, KIND_MANNER)]
        except Exception:
            return []

    def _derive_recall_query(self, state: Optional[SelfStateThread]) -> str:
        """Build a recall query from the agent's mood (§4TB: the open
        questions it also blended in can no longer be written). The LABEL
        of the day's last journal step (§4TC) — "" when there is none or it
        is neutral, which the regenerate path treats as "no blend,
        recent-only". Read off the journal, not off the mood of this
        instant: a cause, a number or a boredom level in the query would
        shift the retrieved experiences with the clock, and with them the
        prompt hash that guards regeneration."""
        steps = self._mood_steps(state)
        if not steps:
            return ""
        last = steps[-1]
        try:
            return str(last.get("label") or "").strip() if int(last.get("intensity") or 0) else ""
        except (TypeError, ValueError):
            return ""

    def _format_state_block(self, state: Optional[SelfStateThread]) -> str:
        if state is None:
            return "(no persisted state)"
        # This block feeds NARRATIVE_PROMPT, whose sha1 is the regenerate
        # skip-guard. It carries no reading of the mood at this instant:
        # the mood comes from the day's journal rows below.
        block = state.format_as_prefix(include_mood=False)
        # §4TC: the day's mood timeline, so the diary can tell a day's arc
        # ("content, then irritable") and not only where it ended. No times
        # of day. §4UD (F2): the LABELS only — never a cause. The causes
        # name what the owner did ("the owner's message sounded unhappy",
        # "the owner gave a thumbs-down"); §4TX withheld them wherever other
        # people read, and the diary was a second copy of them that
        # `introspect` printed with no such check.
        try:
            steps = []
            for row in self._mood_steps(state)[-_MOOD_TIMELINE_STEPS:]:
                n = int(row.get("intensity") or 0)
                steps.append(f"{row.get('label')} {n}" if n else "neutral")
            if steps:
                block = (block + "\n" if block else "") + f"My mood today: {' → '.join(steps)}."
        except Exception:
            pass
        return block or "(nothing on file)"

    def _format_patterns_block(
        self, autobio: AutobiographicalMemory,
    ) -> str:
        """The self-derived cluster patterns."""
        lines: List[str] = []
        try:
            counts = autobio.cluster_counts()
        except Exception:
            counts = {}
        if counts:
            top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            lines.append(
                "Recurring kinds of work I've been doing: "
                + ", ".join(f"{label} ({n}×)" for label, n in top)
            )
        return "\n".join(lines) if lines else "(no notable patterns yet)"

    async def regenerate(
        self,
        *,
        autobio: AutobiographicalMemory,
        state: Optional[SelfStateThread] = None,
    ) -> str:
        """Produce a fresh narrative. Returns the written text (or
        empty string when nothing was written).

        Behaviour matrix:
          - autobio empty                  → no-op
          - critique_fn is None            → fallback to template (concat experiences)
          - critique_fn raises / returns "" → fallback to template
          - success                        → LLM output persisted to disk

        Template fallback exists so the narrative phase is useful even
        when the upstream LLM is unreachable (offline eval, etc.) —
        the wake-up layer prefers SOMETHING over an empty narrative."""

        if not self.enabled:
            return ""

        # Pull a wider window than the diary slice, then keep only
        # informative experiences (tools used, real verdict, substantial
        # request, or a boot marker) so trivial chat turns can't dominate
        # the narrative. When the ENTIRE window is trivial, fall back to
        # the unfiltered slice — a thin diary beats an empty one.
        pool = autobio.recent(limit=self.max_recent_experiences * 4)
        if not pool:
            return ""
        informative = [e for e in pool if _is_informative_experience(e)]
        recent = (informative or pool)[-self.max_recent_experiences:]

        # Blend recent + relevant: with hundreds of entries on disk,
        # the most-recent-N window is < 3% of the agent's actual past.
        # We expand the diary's input by IDF-retrieving older entries
        # related to the agent's current mood, then
        # interleave them with the recent slice. This is what lets the
        # narrative say "this connects to something I worked on weeks
        # ago" instead of being stuck inside the recency window.
        recent_ids = {e.id for e in recent}
        relevant: List[Experience] = []
        try:
            query = self._derive_recall_query(state)
            if query:
                relevant = [
                    e for e in autobio.search_my_past(query, limit=8)
                    if e.id not in recent_ids
                ][:4]
        except Exception:
            relevant = []

        rendered = NARRATIVE_PROMPT.format(
            experiences=self._format_experiences_block(recent, relevant=relevant),
            state=self._format_state_block(state),
            patterns=self._format_patterns_block(autobio),
        )

        # Idempotency guard (same discipline as the dream fragment-set
        # guard): identical input → identical-or-noisier output, so skip
        # the LLM round-trip and the redundant persist entirely. The
        # rendered prompt fingerprints ALL inputs (experiences, state,
        # meta-insights). Guard only when a narrative already exists on
        # disk — never skip the very first write.
        input_key = hashlib.sha1(rendered.encode("utf-8")).hexdigest()
        if input_key == self._last_input_key and self.latest():
            logger.debug(
                "narrative input unchanged since last regeneration; skipping"
            )
            return ""

        text = ""
        used_llm = False
        if self.critique_fn is not None:
            try:
                text = (await self.critique_fn(rendered)) or ""
                text = text.strip()
                used_llm = bool(text)
                if not text:
                    # Silent degradation is how the diary spent a whole
                    # night in template voice without anyone noticing —
                    # make the fallback visible to the operator stream.
                    logger.warning(
                        "narrative critique_fn returned empty content; "
                        "using template fallback"
                    )
            except Exception as e:
                logger.warning("narrative critique_fn failed; using fallback: %s", e)
                text = ""

        if not text:
            # Template fallback. Concatenate the experience summaries
            # into a single first-person paragraph. Not the diary
            # quality the LLM produces, but better than empty when the
            # wake-up layer needs SOMETHING continuous to read.
            joined = " ".join(exp.summary for exp in recent[-6:] if exp.summary)
            if not joined:
                return ""
            text = f"Lately, {joined}"

        # Only remembered after a successful persist — a transient LLM
        # or disk error must not poison the guard against a valid retry
        # (the guard would otherwise pass via an OLDER narrative file,
        # turning a transient failure into persistent staleness).
        # …and only when the model wrote it (§4LI review: the key survives
        # restarts — a template fallback must not freeze until inputs move)
        if self._persist(text, used_llm=used_llm, source_count=len(recent)) and used_llm:
            self._remember_input_key(input_key)
        return text

    def _persist(self, text: str, *, used_llm: bool, source_count: int) -> bool:
        """Write the latest narrative + append the audit record. Returns
        True only when both landed on disk."""
        try:
            with self._lock:
                self.root.mkdir(parents=True, exist_ok=True)
                # Latest narrative — overwritten each pass.
                tmp = self.path.with_suffix(".md.tmp")
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(self.path)
                # Audit history — append-only, byte-capped.
                record = {
                    "timestamp": _utcnow_iso(),
                    "used_llm": bool(used_llm),
                    "source_experience_count": int(source_count),
                    "text": text,
                }
                with open_append(self.history_path) as f:
                    f.write(json.dumps(record, ensure_ascii=False))
                    f.write("\n")
                    f.flush()
                self._maybe_compact_history_locked()
            return True
        except Exception as e:
            logger.warning("narrative persist failed: %s", e)
            return False

    def _maybe_compact_history_locked(self) -> None:
        """Rewrite the audit history keeping the newest
        ``_HISTORY_COMPACT_KEEP_LINES`` records once it passes the byte
        cap. Caller holds ``self._lock``. Best-effort — a failed
        compaction must never fail the persist that triggered it."""
        try:
            if self.history_path.stat().st_size <= _HISTORY_COMPACT_MAX_BYTES:
                return
            from collections import deque
            with self.history_path.open("r", encoding="utf-8") as f:
                tail = deque(f, maxlen=_HISTORY_COMPACT_KEEP_LINES)
            tmp = self.history_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                f.writelines(tail)
            tmp.replace(self.history_path)
            logger.info(
                "narrative history compacted to newest %d records", len(tail),
            )
        except Exception as e:
            logger.warning("narrative history compaction failed: %s", e)
