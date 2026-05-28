"""Workspace recognition layer — wake-up prefix renderer.

Companion to ``selfhood.recognition``: a pure read function that
composes the workspace state into a string the prompt assembly path
splices into the system prompt. Failure mode is "empty string", so a
fresh install with no tracked files / no activity boots silently.

The block carries explicit BEGIN/END markers so an evaluator can strip
the workspace block independently of the selfhood block.
"""

from __future__ import annotations

from typing import List, Optional

from .activity import WorkspaceActivity
from .state import WorkspaceStateThread


WORKSPACE_PREFIX_OPEN = "<!-- WORKSPACE:BEGIN -->"
WORKSPACE_PREFIX_CLOSE = "<!-- WORKSPACE:END -->"


def build_workspace_prefix(
    *,
    activity: Optional[WorkspaceActivity],
    state: Optional[WorkspaceStateThread],
    narrative: Optional[str] = None,
    recent_events_n: int = 5,
    file_changes: Optional[list] = None,
    max_chars: int = 2400,
) -> str:
    """Compose the workspace wake-up prefix.

    Order:
      1. Narrative ("the running summary of my workspace")
      2. File changes since last scan (most load-bearing — this is the
         "what changed while I was away" the user wants)
      3. State thread (tracked-file list, last session timestamp)
      4. Recent activity events (research, task outcomes, commands)

    Empty when all sources are empty; the caller skips prefix injection
    rather than splicing a blank block."""

    parts: List[str] = []

    if narrative and narrative.strip():
        parts.append("My running summary of the workspace:")
        parts.append(narrative.strip())

    if file_changes:
        parts.append("Files that changed since I last looked:")
        for ch in file_changes[:10]:
            label = f" ({ch.get('label')})" if ch.get("label") else ""
            parts.append(f"  - {ch.get('path')}{label}: {ch.get('change')}")

    if state is not None:
        state_block = state.format_as_prefix()
        if state_block:
            parts.append(state_block)

    if activity is not None and recent_events_n > 0:
        events = activity.recent(limit=recent_events_n)
        if events:
            parts.append("Recent things that happened in my workspace:")
            for ev in events:
                summary = (ev.summary or "").strip()
                if not summary:
                    summary = f"({ev.kind})"
                parts.append(f"  - [{ev.kind}] {summary}")

    if not parts:
        return ""

    body = "\n\n".join(parts).strip()
    if len(body) > max_chars:
        body = body[: max_chars - 1].rstrip() + "…"

    return (
        f"{WORKSPACE_PREFIX_OPEN}\n"
        f"### WORKSPACE STATE — WHAT'S OUTSIDE OF ME\n"
        f"This is the state of the user's workspace, not my own internal "
        f"state. Read it as 'what I'm looking at', distinct from the "
        f"selfhood block above which is 'who I am'.\n\n"
        f"{body}\n"
        f"{WORKSPACE_PREFIX_CLOSE}\n"
    )


def strip_workspace_prefix(text: str) -> str:
    """Remove the workspace block from ``text`` if present."""
    if not text or WORKSPACE_PREFIX_OPEN not in text:
        return text
    start = text.index(WORKSPACE_PREFIX_OPEN)
    end_marker = text.find(WORKSPACE_PREFIX_CLOSE)
    if end_marker == -1:
        return text
    end = end_marker + len(WORKSPACE_PREFIX_CLOSE)
    if end < len(text) and text[end] == "\n":
        end += 1
    return text[:start] + text[end:]
