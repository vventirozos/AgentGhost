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
import threading
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

from .autobiographical import AutobiographicalMemory
from .schema import Experience, _utcnow_iso
from .state import SelfStateThread

logger = logging.getLogger("GhostSelfhood")


NARRATIVE_FILENAME = "narrative.md"
NARRATIVE_HISTORY_FILENAME = "narrative.history.jsonl"


# Critique-style prompt: the model is asked to write a first-person
# diary entry from its OWN past, not a third-person summary of someone
# else's actions. The framing matters — a third-person summary
# produces external-knowledge feel; first-person produces continuity
# feel. Kept compact so the LLM round-trip is cheap.
NARRATIVE_PROMPT = """You are continuing a personal diary written by an AI agent
across many sessions. Below are recent things you wrote about, and the
state of open questions / unfinished threads you carried into this
moment.

Write a SHORT first-person diary entry (1-3 paragraphs, plain prose,
no bullet points, no headings). Use "I" / "my". Connect the recent
experiences to the open questions where natural. End with what feels
unresolved — one or two sentences about what you still want to figure
out. Do not summarise this prompt; just write the entry.

--- RECENT EXPERIENCES (newest last) ---
{experiences}

--- CURRENT STATE (open questions & unfinished threads) ---
{state}

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

    def _format_experiences_block(self, experiences: List[Experience]) -> str:
        if not experiences:
            return "(no recent experiences)"
        lines = []
        for exp in experiences:
            tag = f"[{exp.outcome}]" if exp.outcome and exp.outcome != "unknown" else ""
            lines.append(f"- {exp.timestamp} {tag} {exp.summary}".strip())
        return "\n".join(lines)

    def _format_state_block(self, state: Optional[SelfStateThread]) -> str:
        if state is None:
            return "(no persisted state)"
        block = state.format_as_prefix()
        return block or "(no open questions or unfinished threads)"

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

        recent = autobio.recent(limit=self.max_recent_experiences)
        if not recent:
            return ""

        rendered = NARRATIVE_PROMPT.format(
            experiences=self._format_experiences_block(recent),
            state=self._format_state_block(state),
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
