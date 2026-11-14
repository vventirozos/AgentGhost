"""After each tool result: what it changed for the rest of the request.

One call from the dispatcher (`after_tool_result`), three consumers
(2026-11-07, req 298833ef):

  * the fetch memo forgets LOCAL pages after a call that may have changed
    them (`fetch_memo.forget_local`) — a re-render of the agent's own
    `index.html` after an edit was answered with the render from before it;
  * a notification held until the request ends names what changed after it
    was written (`tools.notify_tool.note_change`);
  * a trusted data result is kept for a build this request starts
    (`request_data.note_result`).

`may_change_workspace` is deliberately WIDE for the memo: any `execute`
counts (a script can write any file; dropping a memo entry costs one
re-render, a stale one shows the owner old work). `change_label` is
NARROW for the notification note: it names a change only when one is
plausible (a file written, a project task run, a mutating shell command),
so "node --check" is not reported as a change.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("GhostAgent")

#: is_mutating is True for these but they write no workspace file
_STATE_ONLY = frozenset({"manage_tasks", "update_profile", "introspect", "knowledge_base",
                         "vision_analysis"})
#: manage_projects actions that only read
_PROJECT_READS = frozenset({"status", "get", "list", "changes", "task_list", "budget", "ledger",
                            "artifact_list", "show", "info"})
#: browser operations that only read a page (anything else — click, type, an
#: actions list — can change a local app's state, e.g. its localStorage)
_BROWSER_READS = frozenset({"navigate", "goto", "extract_text", "read", "fetch", "screenshot"})


def _act(args) -> str:
    a = args if isinstance(args, dict) else {}
    return str(a.get("action") or a.get("operation") or a.get("op") or "").strip().lower()


#: file_system operations that only READ. Anything else counts as a write
#: (review r3 m1: a write set missed the `write_append` / `append_file`
#: aliases, which are rewritten to `write` only after the batch check, and
#: `quarantine_delete`). An unknown operation costs one extra re-render.
_FS_READS = frozenset({"read", "read_chunked", "read_lines", "read_file", "outline", "symbols",
                       "inspect", "search", "find", "grep", "head", "tail", "list_files", "list",
                       "ls", "dir", "tree", "list_dir", "list_directory", "quarantine_list",
                       "quarantine_inspect"})
#: run / lab / manage_services actions that only read (review r1 m2:
#: `manage_services start` brings a local page up)
_SERVICE_READS = frozenset({"status", "logs", "log", "url", "collect", "list", "get", "tail",
                            "ps", "info", "show"})


def call_may_change_workspace(fname: str, args) -> bool:
    """From the name and arguments alone — before the call runs (review r1
    M1: a write and a re-render in ONE batch; the memo is consulted at
    dispatch, before any call of the batch has run)."""
    name = str(fname or "").strip().lower()
    a = args if isinstance(args, dict) else {}
    if name == "manage_projects":
        return _act(a) not in _PROJECT_READS
    if name == "browser":
        return bool(a.get("actions")) or (_act(a) not in _BROWSER_READS)
    if name == "file_system":
        # `operation` only — the key the tool reads (review r4 m1: a stray
        # `action: "read"` beside `operation: "write"` read as a read)
        return str(a.get("operation") or "").strip().lower() not in _FS_READS
    if name in ("execute", "image_generation"):
        return True
    if name in ("run", "lab", "manage_services"):
        return _act(a) not in _SERVICE_READS
    return False


def may_change_workspace(fname: str, args, is_mutating: bool) -> bool:
    """May this call have changed a file or a page the workspace serves?"""
    name = str(fname or "").strip().lower()
    if call_may_change_workspace(name, args):
        return True
    if name in _STATE_ONLY or name in ("manage_projects", "browser"):
        return False
    return bool(is_mutating)


def batch_may_change_workspace(calls, canon=None, opaque=None) -> bool:
    """Does any call of this batch (OpenAI-shaped tool calls) look like it
    may change the workspace? Unparseable arguments count as a change.
    ``canon(name)`` maps a raw name to the tool it runs (review r2 M1: the
    model emits ``fs`` / ``filesystem``; the raw name matched nothing);
    ``opaque(name)`` is True for an acquired skill — unvetted code that may
    write anything (review r2 m1). Either failing counts as a change."""
    import json
    for c in calls or []:
        try:
            fn = (c or {}).get("function") or {}
            raw = fn.get("arguments")
            args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
            name = str(fn.get("name") or "")
            name = (canon(name) if canon else None) or name
            if opaque and opaque(name):
                return True
        except Exception:  # noqa: BLE001
            return True
        if call_may_change_workspace(name, args):
            return True
    return False


def change_label(fname: str, args, is_mutating: bool) -> Optional[str]:
    """A short name for what the call changed, for the note under a held
    notification — or None when it plausibly changed no file."""
    name = str(fname or "").strip().lower()
    a = args if isinstance(args, dict) else {}
    if name == "file_system" and is_mutating:
        path = str(a.get("path") or a.get("file_path") or a.get("filename") or "").strip()
        return path or f"a file ({_act(a) or 'write'})"
    if name == "manage_projects" and _act(a) not in _PROJECT_READS:
        return f"project work (manage_projects {_act(a) or 'call'})"
    if name == "execute":
        from .triggers import looks_mutating_command
        cmd = str(a.get("command") or a.get("cmd") or "")
        if a.get("content") and not cmd:
            return f"a script run ({str(a.get('filename') or 'inline')})"
        return "a shell command" if looks_mutating_command(cmd) else None
    if name == "image_generation":
        return "an image"
    if name in ("run", "lab") and is_mutating:
        return f"{name} {_act(a) or 'call'}"
    return None


def after_tool_result(*, fname: str, args: Dict[str, Any], text: Any, ok: bool, ran: bool,
                      changed_world: bool, is_mutating: bool, memo=None,
                      stopped: bool = False, req_id=None, opaque: bool = False) -> Dict[str, Any]:
    """The dispatcher's one call per tool result. Never raises. Returns what
    it did (for logs and tests): {"memo_forgot": n, "change": label|None,
    "data_kept": bool}.

    ``ran``: the call was dispatched and not refused (a FAILED script may
    still have written files — the memo forgets on that too).
    ``changed_world``: the outcome says it changed something (the note under
    a held notification names only those)."""
    out: Dict[str, Any] = {"memo_forgot": 0, "change": None, "data_kept": False}
    try:
        if ran and (opaque or may_change_workspace(fname, args, is_mutating)):
            from .fetch_memo import forget_local
            out["memo_forgot"] = forget_local(memo)
        if changed_world:
            label = change_label(fname, args, is_mutating) or (
                f"a custom tool ({fname})" if opaque else None)
            if label:
                from ..tools.notify_tool import note_change
                note_change(label, req_id)
                out["change"] = label
    except Exception:  # noqa: BLE001
        logger.debug("request effects: change bookkeeping failed", exc_info=True)
    try:
        if not stopped:
            from .request_data import note_result
            out["data_kept"] = note_result(fname, args, text, ok=ok, req_id=req_id)
    except Exception:  # noqa: BLE001
        logger.debug("request effects: data bookkeeping failed", exc_info=True)
    return out
