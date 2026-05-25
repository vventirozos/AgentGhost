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

import datetime
import json
import logging
import re
import threading
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

from .autobiographical import AutobiographicalMemory
from .schema import Experience, _utcnow_iso
from .state import SelfStateThread

logger = logging.getLogger("GhostSelfhood")


NARRATIVE_FILENAME = "narrative.md"
NARRATIVE_HISTORY_FILENAME = "narrative.history.jsonl"


# Patterns we strip from meta_insights before feeding it to the LLM
# diary prompt. The LLM faithfully echoes whatever it's shown, so
# pasting in a raw Python traceback or a runtime-abort marker results
# in the marker literally appearing in the agent's first-person diary.
# Voice-breaking.
_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\):.*?(?=\n\n|\Z)",
    re.DOTALL,
)
_ABORT_MARKER_RE = re.compile(r"\[ATTEMPT_ABORTED[^\]]*\]")
_SYSTEM_BANNER_RE = re.compile(
    r"(?:SYSTEM ALERT|SYSTEM JUDGE REJECTION|### SYNTHETIC TRAINING EXERCISE|"
    r"AUTO-DIAGNOSTIC: DIAGNOSTIC ERROR)[^\n]*",
)
# Long file-path-y blobs from sandbox tracebacks.
_FILE_PATH_RE = re.compile(r'File "[^"]+", line \d+[^\n]*')


def sanitise_meta_insights(text: str, *, max_chars: int = 600) -> str:
    """Strip raw tracebacks, abort markers, and system banners from
    ``meta_insights`` before they're spliced into the diary prompt.
    Returns a compact, voice-safe summary.

    Why aggressive: the narrative is the agent's first-person diary;
    a paragraph that quotes raw traceback noise reads as broken
    selfhood. We'd rather lose detail than leak system noise into the
    agent's voice."""
    if not text:
        return ""
    out = _TRACEBACK_RE.sub("(a traceback I won't reproduce here)", text)
    out = _ABORT_MARKER_RE.sub("(an abort marker)", out)
    out = _SYSTEM_BANNER_RE.sub("(a system banner)", out)
    out = _FILE_PATH_RE.sub("(a source location)", out)
    # Collapse runs of whitespace introduced by the substitutions.
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


# Critique-style prompt: the model is asked to write a first-person
# diary entry from its OWN past, not a third-person summary of someone
# else's actions. The framing matters — a third-person summary
# produces external-knowledge feel; first-person produces continuity
# feel. Kept compact so the LLM round-trip is cheap.
NARRATIVE_PROMPT = """You are continuing a personal diary written by an AI agent
across many sessions. Below are recent things you wrote about, the
state of open questions / unfinished threads you carried into this
moment, and patterns observed across your own work.

Write a SHORT first-person diary entry (1-3 paragraphs, plain prose,
no bullet points, no headings). Use "I" / "my". Connect the recent
experiences to the open questions where natural, and reflect honestly
on the patterns — what you keep doing well, and where you keep
slipping. End with what feels unresolved — one or two sentences about
what you still want to figure out. Do not summarise this prompt; just
write the entry.

--- RECENT EXPERIENCES (newest last) ---
{experiences}

--- CURRENT STATE (open questions & unfinished threads) ---
{state}

--- PATTERNS & WHAT I'VE NOTICED ABOUT MYSELF ---
{patterns}

--- DIARY ENTRY ---
"""


CritiqueFn = Callable[[str], Awaitable[str]]


class NarrativeSummariser:
    """Periodic LLM-driven first-person summary.

    The class is sync-construction / async-run because the LLM call is
    awaited but everything else is file I/O. Mirrors the shape of
    ``reflection.loop.Reflector``."""

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

    # -----------------------------------------------------------------
    # Read path
    # -----------------------------------------------------------------

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
                "(older entries that connect to my current open questions:)"
            )
            for exp in relevant:
                tag = f"[{exp.outcome}]" if exp.outcome and exp.outcome != "unknown" else ""
                lines.append(f"- {exp.timestamp} {tag} {exp.summary}".strip())
        return "\n".join(lines)

    def _derive_recall_query(self, state: Optional[SelfStateThread]) -> str:
        """Build a recall query from the agent's current open questions
        + mood. Empty string when state is empty — the regenerate path
        treats that as "no blend, recent-only", which matches the
        pre-blend behaviour."""
        if state is None:
            return ""
        parts: List[str] = []
        try:
            for q in state.open_questions()[-3:]:
                if q.text:
                    parts.append(q.text)
            mood = state.mood()
            if mood and mood.label:
                parts.append(mood.label)
                if mood.evidence:
                    parts.append(mood.evidence)
        except Exception:
            return ""
        return " ".join(parts).strip()

    def _format_state_block(self, state: Optional[SelfStateThread]) -> str:
        if state is None:
            return "(no persisted state)"
        block = state.format_as_prefix()
        return block or "(no open questions or unfinished threads)"

    def _format_patterns_block(
        self, autobio: AutobiographicalMemory, meta_insights: str,
    ) -> str:
        """Combine self-derived cluster patterns with externally supplied
        meta-insights (dream heuristics, reflection failure patterns).

        This is what makes the diary *meta-cognitive* rather than merely
        experiential: instead of only "I did a SQL task today" it can
        say "I keep reaching for SQL and I keep slipping on the same
        kind of error"."""
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
        mi = sanitise_meta_insights(meta_insights or "")
        if mi:
            lines.append(mi)
        return "\n".join(lines) if lines else "(no notable patterns yet)"

    async def regenerate(
        self,
        *,
        autobio: AutobiographicalMemory,
        state: Optional[SelfStateThread] = None,
        meta_insights: str = "",
    ) -> str:
        """Produce a fresh narrative. Returns the written text (or
        empty string when nothing was written).

        ``meta_insights`` carries cross-phase learning — heuristics the
        dream phase consolidated, failure patterns the reflection phase
        found — so the diary integrates what the agent has *learned*
        about itself, not just what it *did*.

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

        recent = autobio.recent(limit=self.max_recent_experiences)
        if not recent:
            return ""

        # Blend recent + relevant: with hundreds of entries on disk,
        # the most-recent-N window is < 3% of the agent's actual past.
        # We expand the diary's input by IDF-retrieving older entries
        # related to the agent's current open questions / mood, then
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
            patterns=self._format_patterns_block(autobio, meta_insights),
        )

        text = ""
        used_llm = False
        if self.critique_fn is not None:
            try:
                text = (await self.critique_fn(rendered)) or ""
                text = text.strip()
                used_llm = bool(text)
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
            mi = sanitise_meta_insights(meta_insights or "")
            if mi:
                text += f"\n\nWhat I've noticed about myself: {mi}"

        self._persist(text, used_llm=used_llm, source_count=len(recent))
        return text

    def _persist(self, text: str, *, used_llm: bool, source_count: int) -> None:
        try:
            with self._lock:
                self.root.mkdir(parents=True, exist_ok=True)
                # Latest narrative — overwritten each pass.
                tmp = self.path.with_suffix(".md.tmp")
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(self.path)
                # Audit history — append-only.
                record = {
                    "timestamp": _utcnow_iso(),
                    "used_llm": bool(used_llm),
                    "source_experience_count": int(source_count),
                    "text": text,
                }
                with self.history_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False))
                    f.write("\n")
                    f.flush()
        except Exception as e:
            logger.warning("narrative persist failed: %s", e)
