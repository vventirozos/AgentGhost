"""workspace tool — read-only view of the user's workspace state.

Companion to ``introspect`` (read selfhood) and ``self_state`` (write
selfhood). This tool answers questions like:
  * "what changed since yesterday?"
  * "what did my scheduled tasks do?"
  * "what papers / pages have I already pulled?"
  * "what commands did I just run?"

All five workflows (work research, AI research, coding, scheduling,
automation) ask one of these questions repeatedly. The wake-up prefix
already grounds casual answers; this tool is the explicit handle for
deterministic snapshots and ad-hoc queries.

The writable counterpart is ``workspace_track`` (``tools/workspace_track.py``).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")


_VALID_ACTIONS = frozenset(
    {"summary", "stats", "files", "changes", "tasks",
     "research", "commands", "narrative", "recent"}
)

_DEFAULT_LIMIT = 10
_MAX_LIMIT = 50


def _clamp_limit(value, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    if n <= 0:
        return default
    return min(n, _MAX_LIMIT)


def _render_stats(stats: dict) -> str:
    if not stats:
        return "No workspace state on file."
    if stats.get("enabled") is False:
        return "Workspace continuity is disabled."
    lines: List[str] = []
    lines.append(f"Tracked files: {stats.get('tracked_files', 0)}")
    lines.append(f"Workspace events on file: {stats.get('event_count', 0)}")
    lines.append(f"URLs already pulled: {stats.get('seen_urls', 0)}")
    last = stats.get("last_session_at") or ""
    if last:
        lines.append(f"Last session: {last}")
    lines.append(
        "Running narrative: "
        f"{'present' if stats.get('narrative_present') else 'none yet'}"
    )
    kinds = stats.get("event_kinds") or {}
    if kinds:
        ranked = sorted(kinds.items(), key=lambda kv: kv[1], reverse=True)
        top = ", ".join(f"{k}={v}" for k, v in ranked[:6])
        lines.append(f"Event mix: {top}")
    return "\n".join(lines)


def _render_files(workspace_model) -> str:
    state = workspace_model.state
    if state is None:
        return "No workspace state available."
    tracked = state.tracked_files()
    if not tracked:
        return (
            "I am not watching any files yet. Use the workspace_track "
            "tool with action='track' to add one."
        )
    lines = [f"I am watching {len(tracked)} file(s):"]
    for tf in tracked:
        snap = tf.last_snapshot
        if snap is None:
            seen = "never scanned"
        elif not snap.exists:
            seen = "missing on disk"
        else:
            seen = f"last seen {snap.captured_at}, {snap.size} bytes"
        label = f" ({tf.label})" if tf.label else ""
        lines.append(f"  - {tf.path}{label} — {seen}")
    return "\n".join(lines)


def _render_changes(workspace_model) -> str:
    if workspace_model.state is None:
        return "No workspace state available."
    if not workspace_model.state.tracked_files():
        return "Nothing to diff — no tracked files."
    changes = workspace_model.scan_tracked()
    if changes:
        lines = [f"{len(changes)} change(s) since my last scan:"]
        for ch in changes:
            label = f" ({ch.get('label')})" if ch.get("label") else ""
            lines.append(f"  - {ch.get('path')}{label}: {ch.get('change')}")
        return "\n".join(lines)
    # Fall back to recent file_changed events — the wake-up prefix
    # scan that runs at the start of every turn often consumes the
    # diff before this tool can see it. The activity log keeps a
    # record of every diff it found, so the user can still ask
    # "what's changed lately?" and get an answer.
    if workspace_model.activity is None:
        return "No changes detected since the last scan."
    recent = workspace_model.activity.recent(limit=10, kind="file_changed")
    if not recent:
        return "No changes detected since the last scan."
    lines = ["No new changes right now. Recent file-change events on file:"]
    for ev in recent:
        lines.append(f"  - [{ev.timestamp}] {ev.summary or ev.kind}")
    return "\n".join(lines)


def _render_events(
    workspace_model, kind: str, limit: int, *, label: str,
) -> str:
    if workspace_model.activity is None:
        return "No activity log available."
    events = workspace_model.activity.recent(limit=limit, kind=kind)
    if not events:
        return f"No {label} on file."
    lines = [f"Recent {label}:"]
    for ev in events:
        lines.append(f"  - [{ev.timestamp}] {ev.summary or ev.kind}")
    return "\n".join(lines)


def _render_summary(workspace_model) -> str:
    parts: List[str] = []
    parts.append("My workspace right now:")
    parts.append(_render_stats(workspace_model.stats()))

    if workspace_model.narrative is not None:
        narr = (workspace_model.narrative.latest() or "").strip()
        if narr:
            parts.append("\nMy running summary of the workspace:")
            parts.append(narr)

    if workspace_model.state is not None and workspace_model.state.tracked_files():
        try:
            changes = workspace_model.scan_tracked()
        except Exception as e:  # noqa: BLE001
            logger.debug("summary scan failed: %s", e)
            changes = []
        if changes:
            parts.append("\nChanges since I last looked:")
            for ch in changes[:10]:
                parts.append(f"  - {ch.get('path')}: {ch.get('change')}")

    if workspace_model.activity is not None:
        recent_tasks = workspace_model.activity.recent(limit=3, kind="task_outcome")
        if recent_tasks:
            parts.append("\nRecent scheduled-task outcomes:")
            for ev in recent_tasks:
                parts.append(f"  - {ev.summary or ev.kind}")
        recent_research = workspace_model.activity.recent(limit=3, kind="research")
        if recent_research:
            parts.append("\nRecent research artifacts I pulled:")
            for ev in recent_research:
                parts.append(f"  - {ev.summary or ev.kind}")

    return "\n".join(parts).rstrip()


async def tool_workspace(
    action: str = None,
    limit: int = None,
    workspace_model=None,
    **kwargs,
) -> str:
    """Read-only workspace introspection. Never raises."""
    raw_action = (action or "summary").strip().lower()
    if raw_action not in _VALID_ACTIONS:
        return (
            "SYSTEM ERROR: 'action' must be one of "
            f"{sorted(_VALID_ACTIONS)}."
        )

    if workspace_model is None or not getattr(workspace_model, "enabled", False):
        return (
            "Workspace continuity is unavailable — the workspace module "
            "is disabled (--no-workspace-model / --no-memory)."
        )

    try:
        if raw_action == "stats":
            return _render_stats(workspace_model.stats())

        if raw_action == "narrative":
            if workspace_model.narrative is None:
                return "No workspace narrative on file."
            text = (workspace_model.narrative.latest() or "").strip()
            return text or "No workspace narrative on file yet."

        if raw_action == "files":
            return _render_files(workspace_model)

        if raw_action == "changes":
            return _render_changes(workspace_model)

        if raw_action == "tasks":
            return _render_events(
                workspace_model, "task_outcome",
                _clamp_limit(limit, _DEFAULT_LIMIT),
                label="scheduled-task outcomes",
            )

        if raw_action == "research":
            return _render_events(
                workspace_model, "research",
                _clamp_limit(limit, _DEFAULT_LIMIT),
                label="research artifacts",
            )

        if raw_action == "commands":
            return _render_events(
                workspace_model, "command",
                _clamp_limit(limit, _DEFAULT_LIMIT),
                label="command outcomes",
            )

        if raw_action == "recent":
            if workspace_model.activity is None:
                return "No activity log available."
            events = workspace_model.activity.recent(
                limit=_clamp_limit(limit, _DEFAULT_LIMIT),
            )
            if not events:
                return "No workspace events on file."
            lines = ["Recent workspace events:"]
            for ev in events:
                lines.append(
                    f"  - [{ev.timestamp}] [{ev.kind}] {ev.summary or ev.kind}"
                )
            return "\n".join(lines)

        pretty_log("Workspace", "snapshot requested", icon=Icons.BRAIN_SUM)
        return _render_summary(workspace_model)
    except Exception as e:  # noqa: BLE001
        logger.warning("workspace tool failed: %s: %s", type(e).__name__, e)
        return f"Workspace introspection failed: {type(e).__name__}: {e}"
