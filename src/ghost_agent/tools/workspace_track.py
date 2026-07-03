"""workspace_track tool — the agent's write path into workspace state.

Author the workspace's tracked-file list and free-form notes. The
companion read tool is ``workspace``.

Actions:
  * ``track`` — add a file path to the watchlist (optionally with label).
  * ``untrack`` — remove a file from the watchlist.
  * ``note`` — append a free-form workspace event (kind='note').
  * ``mark_seen`` — record a URL as already-pulled (manual dedup).

Everything routes through ``WorkspaceModel`` so bounded / atomic /
dedup guarantees from the state thread are preserved.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")


_VALID_ACTIONS = frozenset({"track", "untrack", "note", "mark_seen"})


async def tool_workspace_track(
    action: str = None,
    path: str = None,
    label: str = None,
    text: str = None,
    url: str = None,
    workspace_model=None,
    **kwargs,
) -> str:
    # str() so a non-string action can't raise AttributeError on .strip().
    raw_action = str(action or "").strip().lower()
    if raw_action not in _VALID_ACTIONS:
        return (
            "SYSTEM ERROR: 'action' is mandatory and must be one of "
            f"{sorted(_VALID_ACTIONS)}."
        )

    if workspace_model is None or not getattr(workspace_model, "enabled", False):
        return (
            "Workspace continuity is unavailable — the workspace module "
            "is disabled. Nothing was recorded."
        )

    try:
        if raw_action == "track":
            path = (path or "").strip()
            if not path:
                return "SYSTEM ERROR: 'path' is required for track."
            tf = workspace_model.track_file(path, label=(label or "").strip())
            if tf is None:
                return "Nothing recorded — the path was empty."
            pretty_log("Workspace", f"now tracking {path}", icon=Icons.IDEA)
            desc = f" labelled '{tf.label}'" if tf.label else ""
            return f"Now tracking '{tf.path}'{desc}."

        if raw_action == "untrack":
            path = (path or "").strip()
            if not path:
                return "SYSTEM ERROR: 'path' is required for untrack."
            ok = workspace_model.untrack_file(path)
            if not ok:
                return f"'{path}' is not on the watchlist."
            pretty_log("Workspace", f"untracked {path}", icon=Icons.OK)
            return f"No longer tracking '{path}'."

        if raw_action == "note":
            text = (text or "").strip()
            if not text:
                return "SYSTEM ERROR: 'text' is required for note."
            ev = workspace_model.note(text)
            if ev is None:
                return "Nothing recorded — the note was empty."
            pretty_log("Workspace", f"note recorded: {text[:60]}", icon=Icons.IDEA)
            return f"Recorded workspace note: {text}"

        if raw_action == "mark_seen":
            url = (url or "").strip()
            if not url:
                return "SYSTEM ERROR: 'url' is required for mark_seen."
            if workspace_model.state is None:
                return "Workspace state unavailable."
            added = workspace_model.state.mark_url_seen(url)
            if not added:
                return f"URL already on the seen list: {url}"
            pretty_log("Workspace", f"marked seen: {url}", icon=Icons.OK)
            return f"Marked '{url}' as already-pulled."
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "workspace_track tool failed: %s: %s", type(e).__name__, e,
        )
        return f"Workspace track operation failed: {type(e).__name__}: {e}"

    return "SYSTEM ERROR: unreachable workspace_track branch."
