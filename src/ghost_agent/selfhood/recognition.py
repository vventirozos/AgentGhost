"""Recognition layer — proposal item #4.

Wake-up retrieval: when a new session starts, pull the agent's own
prior records (autobiographical experiences + self-state + narrative)
and frame them in the first person, so the freshly-instantiated
"I" reads its own past as autobiographical memory rather than as
external diary entries.

Pure read path. Never writes — the autobiographical writer and the
state thread are the canonical owners. This module just formats the
prefix that gets spliced into the system prompt.

The prefix has a hard char cap and a hard structural separator so the
existing system prompt assembly (``core/prompts.SYSTEM_PROMPT``) can
absorb it without budget surprises. Failing silently is a feature:
"empty prefix" simply means "no remembered past" and the session
boots as before.
"""

from __future__ import annotations

from typing import Optional

from .autobiographical import AutobiographicalMemory
from .state import SelfStateThread


# The marker the system prompt inserts before/after the autobiographical
# block. Kept explicit so a downstream consumer (e.g. an evaluator) can
# strip the block and run the agent without selfhood injection.
PREFIX_OPEN = "<!-- SELFHOOD:BEGIN -->"
PREFIX_CLOSE = "<!-- SELFHOOD:END -->"


def build_wakeup_prefix(
    *,
    autobio: Optional[AutobiographicalMemory],
    state: Optional[SelfStateThread],
    narrative: Optional[str],
    recent_experiences_n: int = 3,
    max_chars: int = 2400,
) -> str:
    """Compose the first-person wake-up prefix.

    Order is deliberate:
      1. Narrative ("the running diary I've been keeping") — sets voice
      2. State thread (open questions, unfinished, mood) — load-bearing
         continuity material
      3. Recent experiences (most recent N) — episodic flavour

    Empty when all three sources are empty; the caller is expected to
    skip prefix injection in that case rather than splicing a blank
    block."""

    parts = []

    if narrative and narrative.strip():
        parts.append("Where I last left off (my running first-person diary):")
        parts.append(narrative.strip())

    if state is not None:
        state_block = state.format_as_prefix()
        if state_block:
            parts.append(state_block)

    if autobio is not None and recent_experiences_n > 0:
        recent = autobio.recent(limit=recent_experiences_n)
        if recent:
            parts.append("Recent things I remember doing:")
            for exp in recent:
                line = f"  - {exp.summary}"
                if exp.outcome and exp.outcome != "unknown":
                    line += f" [{exp.outcome}]"
                parts.append(line)

    if not parts:
        return ""

    body = "\n\n".join(parts).strip()
    if len(body) > max_chars:
        body = body[: max_chars - 1].rstrip() + "…"

    return (
        f"{PREFIX_OPEN}\n"
        f"### CONTINUITY FROM MY PAST SESSIONS\n"
        f"What follows is mine — entries I wrote in earlier sessions. "
        f"Read them as autobiographical memory, not external knowledge.\n\n"
        f"{body}\n"
        f"{PREFIX_CLOSE}\n"
    )


def strip_wakeup_prefix(text: str) -> str:
    """Remove the selfhood block from ``text`` if present. Used by
    evaluators / tests that want to inspect the underlying system
    prompt without the continuity scaffold."""
    if not text or PREFIX_OPEN not in text:
        return text
    start = text.index(PREFIX_OPEN)
    end_marker = text.find(PREFIX_CLOSE)
    if end_marker == -1:
        return text  # malformed — leave alone
    end = end_marker + len(PREFIX_CLOSE)
    # Also trim a single trailing newline that the prefix itself
    # introduces, so callers don't see double blank lines after strip.
    if end < len(text) and text[end] == "\n":
        end += 1
    return text[:start] + text[end:]
