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
from .schema import _PROJECTS_PATH_RE, filter_events_for_project
from .state import WorkspaceStateThread


WORKSPACE_PREFIX_OPEN = "<!-- WORKSPACE:BEGIN -->"
WORKSPACE_PREFIX_CLOSE = "<!-- WORKSPACE:END -->"


def _narrative_is_cross_project(narrative: str, active_project_id: str) -> bool:
    """True when the (global) narrative paragraph references a project
    OTHER than the active one. The persisted narrative is consolidated
    across all projects, so on a project-scoped turn it can describe a
    DIFFERENT project's files — exactly the bleed that trapped a fresh
    build. When that's the case we drop it rather than mislead."""
    if not active_project_id:
        return False
    active = active_project_id.strip().lower()
    for m in _PROJECTS_PATH_RE.finditer(narrative or ""):
        found = m.group(1).lower()
        # Prefix-tolerant: an LLM-rewritten path may truncate or extend the
        # hex id (projects/991e52be… vs the full 991e52bce912). Only when
        # NEITHER is a prefix of the other is it genuinely another project.
        # Exact-match here dropped correct narratives on every turn.
        if not (found.startswith(active) or active.startswith(found)):
            return True
    return False


def build_workspace_prefix(
    *,
    activity: Optional[WorkspaceActivity],
    state: Optional[WorkspaceStateThread],
    narrative: Optional[str] = None,
    recent_events_n: int = 5,
    file_changes: Optional[list] = None,
    file_warnings: Optional[list] = None,
    max_chars: int = 2400,
    active_project_id: Optional[str] = None,
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

    # Drop the global narrative when a project is active AND the narrative
    # talks about a DIFFERENT project — otherwise a new project inherits a
    # prior one's "I pulled minecraft-clone/index.html" summary and chases
    # files that don't exist in its sandbox.
    if (
        narrative and narrative.strip()
        and not _narrative_is_cross_project(narrative, active_project_id or "")
    ):
        parts.append("My running summary of the workspace:")
        parts.append(narrative.strip())

    if file_changes:
        parts.append("Files that changed since I last looked:")
        for ch in file_changes[:10]:
            label = f" ({ch.get('label')})" if ch.get("label") else ""
            parts.append(f"  - {ch.get('path')}{label}: {ch.get('change')}")

    # Broken-file flags (feature 2A): a tracked .py I just touched no longer
    # parses. Surfaced loudly so I fix it before relying on it mid-task.
    if file_warnings:
        parts.append("⚠ Files I touched that currently DO NOT PARSE — fix these "
                     "before relying on them:")
        for w in file_warnings[:10]:
            parts.append(f"  - {w.get('path')}: {w.get('error')}")

    if state is not None:
        state_block = state.format_as_prefix()
        if state_block:
            parts.append(state_block)

    if activity is not None and recent_events_n > 0:
        # Pull a wider window THEN scope to the active project, so a busy
        # OTHER project can't crowd this project's events out of the tail.
        events = activity.recent(limit=max(recent_events_n * 4, recent_events_n))
        events = filter_events_for_project(events, active_project_id)[-recent_events_n:]
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
    # Search for CLOSE from AFTER the OPEN marker. Searching from 0 could
    # match a stray/earlier CLOSE, yielding end < start and duplicating the
    # block instead of removing it.
    end_marker = text.find(WORKSPACE_PREFIX_CLOSE, start)
    if end_marker == -1:
        return text
    end = end_marker + len(WORKSPACE_PREFIX_CLOSE)
    if end < len(text) and text[end] == "\n":
        end += 1
    return text[:start] + text[end:]
