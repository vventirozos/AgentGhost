"""workspace tool — read-only ACTIVITY LEDGER of the user's workspace (what already happened: tracked-file changes, task/command outcomes, research pulled). It does no work — see §4FJ for why the advertised text says so.

Companion to ``introspect`` (the agent's own selfhood). This tool answers
questions like:
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


def _TO_failed(text):
    """A tool-level exception, DECLARED failed (§4LZ B8: these plain
    strings were booked OK by the turn loop and the corpus)."""
    from .outcome import ToolOutcome
    return ToolOutcome.failed(text, reason_code="tool_exception")


logger = logging.getLogger("GhostAgent")


_VALID_ACTIONS = frozenset(
    {"summary", "stats", "files", "changes", "tasks",
     "research", "commands", "narrative", "recent",
     # Models repeatedly guessed action='search' (a strike, then a
     # recovery turn) — make the guess correct. 'recall' is an alias.
     "search", "recall"}
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
        _shown = workspace_model.display_path(tf.path) if hasattr(workspace_model, "display_path") else tf.path
        lines.append(f"  - {_shown}{label} — {seen}")
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


def _active_pid(workspace_model) -> str:
    """The task-local project pin when one is bound (an autopilot tick), else
    the shared attribute (§4PF r4)."""
    try:
        return str(workspace_model._stamp_project_id() or "")
    except Exception:  # noqa: BLE001
        return str(getattr(workspace_model, "current_project_id", "") or "")


def _scoped(workspace_model, events):
    """§4PF (W12): the listings are scoped to the active project like the
    wake-up prefix is, and say so."""
    from ..workspace.schema import filter_events_for_project
    pid = _active_pid(workspace_model)
    return (filter_events_for_project(events, pid) if pid else list(events)), pid


def _finished_job_ids(events) -> set:
    return {str((ev.payload or {}).get("job_id")) for ev in events
            if ev.kind == "command" and (ev.payload or {}).get("job_id")
            and "promoted to background job" not in str(ev.summary or "")}


def event_line(ev, *, with_kind: bool = False) -> str:
    from ..workspace.recognition import who_tag
    kind = f" [{ev.kind}]" if with_kind else ""
    return f"  - [{ev.timestamp}]{kind}{who_tag(ev)} {ev.summary or ev.kind}"


def _render_events(
    workspace_model, kind: str, limit: int, *, label: str,
) -> str:
    if workspace_model.activity is None:
        return "No activity log available."
    pid = _active_pid(workspace_model)
    # §4PF r4: the reader filters by project WHILE scanning — a window of the
    # newest N rows went blank behind another project's rows
    events = workspace_model.activity.recent(limit=max(limit * 4, limit), kind=kind, project_id=pid or None)
    events, pid = _scoped(workspace_model, events)
    if kind == "command":
        # §4PF (W7): a promotion line is superseded by the job's real outcome
        done = _finished_job_ids(events)
        events = [ev for ev in events
                  if not ("promoted to background job" in str(ev.summary or "")
                          and str((ev.payload or {}).get("job_id")) in done)]
    events = events[-limit:]
    scope = f" (project {pid})" if pid else ""
    if not events:
        return f"No {label} on file{scope}."
    lines = [f"Recent {label}{scope}:"]
    for ev in events:
        lines.append(event_line(ev))
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
    query: str = None,
    workspace_model=None,
    **kwargs,
) -> str:
    """Read-only workspace introspection. Never raises. §4PE (W1): every
    success is DECLARED — the ledger splices command lines, task errors and
    notes verbatim, and an undeclared result let a recorded `grep "EXIT
    CODE: 1"` read as this tool's own failure banner."""
    from .outcome import ToolOutcome as _TO
    res = await _tool_workspace_inner(action, limit, query, workspace_model, **kwargs)
    if isinstance(res, _TO):
        return res
    if str(res).lstrip().startswith("SYSTEM ERROR"):
        return _TO.rejected(str(res), world_changed=False, reason_code="workspace_bad_args")
    return _TO.ok(str(res), world_changed=False)


async def _tool_workspace_inner(action, limit, query, workspace_model, **kwargs):
    # str() so a non-string action (a malformed tool call) can't raise
    # AttributeError on .strip() and escape the never-raises contract.
    raw_action = str(action or "summary").strip().lower()
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
            # §4PE (W10): stored LLM text was the whole result — headed, so a
            # narrative that begins "Error…" is the narrative, not a failure
            return ("WORKSPACE NARRATIVE (the agent's own running summary):\n" + text) if text \
                else "No workspace narrative on file yet."

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

        if raw_action in ("search", "recall"):
            # Accept the common near-miss arg names too — the point of
            # this action is to absorb guessed calls, not strike on them.
            q = str(query or kwargs.get("q") or kwargs.get("text")
                    or kwargs.get("keywords") or "").strip()
            if not q:
                return (
                    "SYSTEM ERROR: action='search' needs a 'query' string — "
                    "keywords to look for in the workspace activity log."
                )
            hits = workspace_model.search_events(
                q, limit=_clamp_limit(limit, _DEFAULT_LIMIT),
            )
            if not hits:
                return (
                    f"No workspace events match '{q}'. This searches only my "
                    "activity log (file changes, task/command outcomes, "
                    "research pulls, notes). For stored facts or ingested "
                    "documents use the recall tool; for project state use "
                    "manage_projects."
                )
            lines = [f"Workspace events matching '{q}':"]
            for ev in hits:
                proj = f" (project {ev.project_id})" if ev.project_id else ""
                lines.append(
                    f"  - [{ev.timestamp}] [{ev.kind}] {ev.summary or ev.kind}{proj}"
                )
            return "\n".join(lines)

        if raw_action == "recent":
            if workspace_model.activity is None:
                return "No activity log available."
            _lim = _clamp_limit(limit, _DEFAULT_LIMIT)
            _pid0 = _active_pid(workspace_model)
            events, _pid = _scoped(workspace_model,
                                   workspace_model.activity.recent(limit=_lim * 4, project_id=_pid0 or None))
            events = events[-_lim:]
            if not events:
                return "No workspace events on file" + (f" (project {_pid})" if _pid else "") + "."
            lines = ["Recent workspace events" + (f" (project {_pid})" if _pid else "") + ":"]
            for ev in events:
                lines.append(event_line(ev, with_kind=True))
            return "\n".join(lines)

        pretty_log("Workspace", "snapshot requested", icon=Icons.BRAIN_SUM)
        return _render_summary(workspace_model)
    except Exception as e:  # noqa: BLE001
        logger.warning("workspace tool failed: %s: %s", type(e).__name__, e)
        return _TO_failed(f"Workspace introspection failed: {type(e).__name__}: {e}")   # declared (§4LZ B8)
