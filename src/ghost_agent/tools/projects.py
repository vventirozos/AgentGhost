"""LLM-facing tool surface for long-term projects.

One consolidated tool (`manage_projects`) with an action dispatch,
matching the pattern used by `manage_tasks`, `knowledge_base`, and
`file_system`. The handler operates against a ``ProjectStore`` that
lives on ``context.project_store`` and mutates ``context.current_project_id``
as a side effect of ``switch`` / ``create`` / ``exit``.

Actions:
    create / list / get / switch / exit / update / delete / resume
    task_add / task_update / task_decompose / task_next / task_list
    artifact_add / event_log / promote_from_context
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from ..memory.projects import ProjectStore, ProjectKind, ProjectStatus
from ..core.planning import ProjectPlan, TaskStatus, DependencyType
from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

# Duplicate-create policy: as long as a non-ARCHIVED project with
# the same title exists, refuse to create a new one and reuse the
# existing id instead. The previous time-windowed policy (5 minutes)
# expired on long-running projects — the agent loops over `create`
# whenever it re-reads the original user prompt, and after the window
# elapsed it actually succeeded in spawning a duplicate, leaving the
# model split between two projects. Existence-based has no such
# expiry. A user who genuinely wants a fresh project with the same
# title must archive the old one first (`update` action,
# status="ARCHIVED"). The constant is kept for tests that simulate
# the "no guard" case by patching it to None.
_DUPLICATE_CREATE_WINDOW_SECONDS = None  # None == always reuse


_ACTIONS = {
    # project-level
    "create", "list", "get", "switch", "exit", "update", "delete",
    "archive", "resume", "status",
    # task-level
    "task_add", "task_update", "task_decompose", "task_next", "task_list",
    # artifacts / events
    "artifact_add", "event_log",
    # inbox promotion (suggestion-accepted path)
    "promote_from_context",
    # self-advancing loop
    "autoadvance",
}


def _err(msg: str) -> str:
    return f"ERROR: {msg}"


def _ok(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, default=str)
    except Exception:
        return str(payload)


_PROJECT_KEY_PREFIX = "proj::"
_CURRENT_SENTINEL = "__current_project__"


def _iter_scratchpad_items(scratchpad) -> List[tuple]:
    """Best-effort snapshot of the scratchpad's current key/value pairs.

    Returns an empty list when the scratchpad is missing or doesn't
    expose its internal ``_data`` mapping — callers must tolerate that
    (e.g. MagicMock-based test contexts).
    """
    if scratchpad is None:
        return []
    data = getattr(scratchpad, "_data", None)
    if data is None:
        return []
    try:
        return [
            (k, v) for k, v in data.items()
            if not str(k).startswith(_PROJECT_KEY_PREFIX)
            and k != _CURRENT_SENTINEL
        ]
    except Exception:
        return []


def _snapshot_scratchpad(context, project_id: Optional[str]):
    """Capture in-flight scratchpad keys into the project's event log."""
    if not project_id:
        return
    store = _resolve_store(context)
    if store is None:
        return
    items = _iter_scratchpad_items(getattr(context, "scratchpad", None))
    if not items:
        return
    try:
        store.log_event(project_id, None, "scratchpad_snapshot",
                        {"keys": dict(items)})
    except Exception:
        logger.debug("scratchpad snapshot failed", exc_info=True)


def _hydrate_scratchpad(context, project_id: Optional[str]):
    """Restore the most recent scratchpad_snapshot for a project into
    the live scratchpad.

    We clear non-namespaced keys first so a switch doesn't leak state
    from the previous project, then restore the target's saved keys.
    Sentinel keys (`__current_project__`, project-namespaced `proj::…`)
    are preserved.
    """
    sp = getattr(context, "scratchpad", None)
    if sp is None:
        return
    data = getattr(sp, "_data", None)
    if data is None:
        return
    try:
        # Clear free-chat keys (not namespaced, not sentinels)
        victims = [
            k for k in list(data.keys())
            if not str(k).startswith(_PROJECT_KEY_PREFIX)
            and k != _CURRENT_SENTINEL
        ]
        for k in victims:
            try:
                sp.delete(k)
            except Exception:
                pass
    except Exception:
        pass
    if not project_id:
        return
    store = _resolve_store(context)
    if store is None:
        return
    try:
        events = store.list_events(project_id, limit=1,
                                   event_type="scratchpad_snapshot")
    except Exception:
        events = []
    if not events:
        return
    keys = (events[0].get("payload") or {}).get("keys") or {}
    for k, v in keys.items():
        try:
            sp.set(k, v)
        except Exception:
            pass


def _set_current(context, project_id: Optional[str]):
    """Update the runtime + persist to scratchpad so the selection
    survives process restarts.

    Snapshots the outgoing project's scratchpad keys and hydrates the
    incoming project's last snapshot — this is what makes "go back and
    forth" feel like resuming work instead of starting over.
    """
    prev = getattr(context, "current_project_id", None)
    if prev and prev != project_id:
        _snapshot_scratchpad(context, prev)
    context.current_project_id = project_id
    # Keep the workspace model's active-project pointer in lock-step so any
    # research/file events recorded after a mid-request create/switch are
    # stamped with the right project (and the wake-up prefix scopes to it).
    _wm = getattr(context, "workspace_model", None)
    if _wm is not None:
        try:
            _wm.current_project_id = project_id or ""
        except Exception:
            pass
    sp = getattr(context, "scratchpad", None)
    if sp is not None:
        try:
            if project_id is None:
                sp.delete(_CURRENT_SENTINEL)
            else:
                sp.set(_CURRENT_SENTINEL, project_id)
        except Exception:
            pass
    if project_id and project_id != prev:
        _hydrate_scratchpad(context, project_id)
    elif project_id is None:
        # Leaving project mode: clear the hydrated keys so free chat
        # starts fresh. Named sentinels survive.
        _hydrate_scratchpad(context, None)


def _workspace_note(project_id: str) -> str:
    """Tell the model that entering a project moves its working directory.

    File ops (file_system) and code execution (execute) are scoped to
    ``sandbox/projects/<id>/`` while the project is active. Without this
    cue the model writes a file, then runs it from a path it assumes is the
    sandbox root, and burns strikes on "can't open file" before recovering
    (observed live). Steering it toward bare relative names keeps writes and
    runs in the same place."""
    return (
        f"Working directory is now projects/{project_id}/. Files you write "
        "with file_system and run with execute live HERE — reference them by "
        "bare name (e.g. 'chart.py', 'notes.md'), not '/workspace/...'. To "
        "clean up this project's files later: remove sandbox/projects/"
        f"{project_id}/."
    )


def _resolve_store(context) -> Optional[ProjectStore]:
    return getattr(context, "project_store", None)


def _coerce_str_list(value) -> Optional[List[str]]:
    """Normalize a parameter that the schema declares as ``array<string>``
    but small LLMs routinely pass as a bare string.

    Behavior:
      - ``None`` → ``None`` (caller treats as "not provided")
      - already a list → strip / drop blanks / drop non-strings
      - a string → split on newlines first, fall back to commas if a
        single line was passed. This rescues the symptom from the trace
        where iterating a string passed character-by-character created
        one task per character.

    Returning ``None`` instead of ``[]`` distinguishes "user didn't pass
    this" from "user explicitly passed an empty list" — the difference
    matters for ``task_update`` where ``alternatives=[]`` should clear
    the list.
    """
    if value is None:
        return None
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
            elif item:
                out.append(str(item).strip())
        return out
    if isinstance(value, str):
        stripped = value.strip()
        # JSON-array string: happens when the model emits a tool call
        # whose "array<string>" parameter is serialized as a single
        # string like `'["design", "implement"]'`. Qwen-family models
        # do this regularly, which was the failure mode in the
        # 2026-04-19 trace where task_decompose collapsed to 1 task.
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return _coerce_str_list(parsed)
            except (json.JSONDecodeError, ValueError):
                pass  # fall through to newline / marker handling
        # Preferred path: the model used newlines (what we told it to do).
        if "\n" in value:
            parts = value.split("\n")
            return [p.strip() for p in parts if p and p.strip()]
        # Fallback for the real-world symptom where the XML tool-call
        # parser flattens the newlines: detect numbered or bulleted
        # list markers embedded in a single line. This ONLY fires when
        # there are at least two such markers, which makes misfiring
        # on "step 1. do the thing" sentences extremely unlikely.
        marker_spans = _list_marker_spans(value)
        if len(marker_spans) >= 2:
            parts: List[str] = []
            for i, (start, end) in enumerate(marker_spans):
                seg_end = marker_spans[i + 1][0] if i + 1 < len(marker_spans) else len(value)
                segment = value[end:seg_end].strip(" \t.,;")
                if segment:
                    parts.append(segment)
            if parts:
                return parts
        # Single string, no list structure → keep it whole.
        return [value.strip()] if value.strip() else []
    # Fall through for unexpected types — log and drop rather than crash
    logger.debug("unexpected type for string-list param: %r", type(value))
    return []


_TASK_LIST_SLIM_FIELDS = (
    "id", "parent_id", "description", "status", "depth", "position",
    "result_summary",
)


# Matches list item markers at string start or after whitespace:
#   "1. "  "2) "  "10. "   (numbered)
#   "- "   "* "   "• "     (bulleted)
# The leading boundary is "start of string" or whitespace, captured so
# we know where the previous segment ended. The marker itself (number
# + punctuation or bullet) is the piece we strip off the next segment.
_LIST_MARKER_RE = re.compile(
    r"(?:^|(?<=\s))(?P<marker>(?:\d{1,3}[.)]\s+|[-*•]\s+))",
)


def _list_marker_spans(value: str) -> List[tuple]:
    """Return (start, end) spans of every list marker found in value.

    ``start`` is the index where the marker begins; ``end`` is the index
    where the task text begins (i.e. just past the marker). At least
    two markers must be present for the caller to treat this as a list
    — a single "Step 1. Plan this" in prose is not a list.
    """
    spans: List[tuple] = []
    for m in _LIST_MARKER_RE.finditer(value):
        spans.append((m.start("marker"), m.end("marker")))
    return spans


def _link_project_in_graph(context, project_id: str, title: str):
    """Register a project as a node in the knowledge graph so the
    existing spreading-activation retrieval can surface its tasks when
    the user's query touches adjacent topics.

    No-op when graph_memory is unset (tests, worker contexts).
    """
    gm = getattr(context, "graph_memory", None)
    if gm is None:
        return
    try:
        gm.add_triplets([{
            "subject": f"project:{project_id}",
            "predicate": "HAS_TITLE",
            "object": title,
        }])
    except Exception:
        logger.debug("graph project link skipped", exc_info=True)


def _link_task_in_graph(context, project_id: str, task_id: str, description: str):
    gm = getattr(context, "graph_memory", None)
    if gm is None:
        return
    try:
        gm.add_triplets([
            {"subject": f"project:{project_id}",
             "predicate": "HAS_TASK", "object": f"task:{task_id}"},
            {"subject": f"task:{task_id}",
             "predicate": "HAS_DESCRIPTION", "object": description[:120]},
        ])
    except Exception:
        logger.debug("graph task link skipped", exc_info=True)


def _briefing(store: ProjectStore, project_id: str) -> Dict[str, Any]:
    """Compact status snapshot suitable for a resume banner."""
    proj = store.get_project(project_id)
    if not proj:
        return {}
    plan = ProjectPlan(store, project_id)
    nxt = plan.next_ready_leaf()
    events = store.list_events(project_id, limit=5)
    return {
        "project": {
            "id": proj["id"],
            "title": proj["title"],
            "kind": proj["kind"],
            "status": proj["status"],
            "goal": proj["goal"],
        },
        "task_tree": plan.render() or "(empty)",
        "next_ready": (
            {"id": nxt.id, "description": nxt.description, "status": nxt.status.value}
            if nxt else None
        ),
        "recent_events": [
            {"type": e["type"], "ts": e["ts"], "payload": e["payload"]}
            for e in events
        ],
    }


# ------------------------------------------------------------------ handler

async def tool_manage_projects(
    context,
    action: str = "",
    # create / update
    title: str = "",
    kind: str = "GENERAL",
    goal: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    # identifiers
    project_id: Optional[str] = None,
    task_id: Optional[str] = None,
    task_ids: Optional[List[str]] = None,
    parent_id: Optional[str] = None,
    # task-level
    description: str = "",
    status: Optional[str] = None,
    dependency_type: str = "ALL",
    alternatives: Optional[List[str]] = None,
    postconditions: Optional[List[str]] = None,
    result: str = "",
    failure_reason: str = "",
    subtasks: Optional[List[str]] = None,
    # artifacts
    artifact_kind: str = "",
    payload: str = "",
    # misc
    status_filter: Optional[str] = None,
    limit: int = 20,
    event_type: Optional[str] = None,
    context_summary: str = "",
    **_unused,
) -> str:
    act = (action or "").strip().lower()
    if act not in _ACTIONS:
        return _err(
            f"unknown action '{action}'. valid: {', '.join(sorted(_ACTIONS))}"
        )

    # Canonicalise model-supplied ids at the boundary. IDs are lowercase
    # hex, but the LLM routinely mangles the case of opaque hex tokens
    # when it echoes one back (e.g. 9b5bd5cd812b -> 9B5Bd5Cd812B), which
    # otherwise surfaced as "project not found". The store also
    # canonicalises (defence in depth); doing it here too keeps
    # context.current_project_id stored in canonical form.
    from ..memory.projects import _canon_id as _canon_pid
    if project_id:
        project_id = _canon_pid(project_id)
    if task_id:
        task_id = _canon_pid(task_id)

    store = _resolve_store(context)
    if store is None:
        return _err("project_store is not configured on this context")

    # Coerce list-typed params at the boundary so every action below
    # sees real lists. The `is None` checks downstream still work
    # because _coerce_str_list returns None when the input is None.
    subtasks = _coerce_str_list(subtasks)
    alternatives = _coerce_str_list(alternatives)
    postconditions = _coerce_str_list(postconditions)
    task_ids = _coerce_str_list(task_ids)

    # Rescue path: the model routinely confuses `task_id` (singular)
    # with `task_ids` (list) and sends `'["real_id"]'` in the singular
    # field. Detect and re-route so the handler doesn't return "task
    # not found" for an obviously-correct id.
    if isinstance(task_id, str) and task_id.strip().startswith("["):
        expanded = _coerce_str_list(task_id)
        if expanded:
            if len(expanded) == 1:
                task_id = expanded[0]
            else:
                task_ids = list(task_ids or []) + expanded
                task_id = None

    pretty_log("Project Tool", f"action={act}", icon=Icons.BRAIN_PLAN)

    # Resolve an implicit project_id from context.current_project_id so
    # the LLM doesn't have to pass it on every call while in project mode.
    if project_id is None and act not in {"create", "list", "promote_from_context"}:
        project_id = getattr(context, "current_project_id", None)

    try:
        # ---- project lifecycle ------------------------------------------

        if act == "create":
            if not title:
                return _err("title is required for action=create")
            normalized = title.strip().lower()
            now = time.time()
            for existing in store.list_projects():
                if existing["status"] == "ARCHIVED":
                    continue
                if (existing["title"] or "").strip().lower() != normalized:
                    continue
                # Existence-based guard: a non-archived project with
                # the same title means the user is asking us to
                # continue, not start over. We reuse it regardless of
                # how long ago it was created. ``_DUPLICATE_CREATE_WINDOW_SECONDS``
                # is honored as a max-age only when explicitly set to
                # an int (tests patch it to 0 to disable the guard).
                window = _DUPLICATE_CREATE_WINDOW_SECONDS
                if window is not None:
                    age = now - float(existing.get("created_at", 0))
                    if age > window:
                        continue
                # Track retry count in project metadata so we can
                # escalate the response when the model is clearly in
                # a loop. The metadata also gives the operator a
                # visible signal (via pretty_log) that something is
                # wrong at the model level.
                existing_meta = dict(existing.get("metadata") or {})
                retry_count = int(existing_meta.get("duplicate_create_retries", 0)) + 1
                existing_meta["duplicate_create_retries"] = retry_count
                try:
                    store.update_project(existing["id"], metadata=existing_meta)
                except Exception:
                    pass
                _set_current(context, existing["id"])

                # Escalate wording on retries. The first refused call
                # gets a firm note; retry 3+ gets an alarmed
                # instruction plus an operator-visible warning.
                is_escalated = retry_count >= 3
                if is_escalated:
                    pretty_log(
                        "Project Create Loop",
                        f"retry #{retry_count} on title '{title[:60]}' "
                        f"(project {existing['id']}) — model not trusting "
                        f"the duplicate guard",
                        level="WARNING", icon=Icons.WARN,
                    )
                    instruction = (
                        f"STOP. This is retry #{retry_count} of a create "
                        f"call that was already refused. You are in a "
                        f"LOOP. The project '{title}' EXISTS at id "
                        f"{existing['id']}. Your next tool call MUST be "
                        f"one of: task_next (find next work), task_list "
                        f"(see progress), task_update (mark a task "
                        f"DONE/FAILED), or a plain user-facing final "
                        f"message. It MUST NOT be manage_projects with "
                        f"action=create, switch, or resume for this "
                        f"title. Re-reading the user's original 'start a "
                        f"new project' message is a bug — that request "
                        f"has been fulfilled."
                    )
                else:
                    instruction = (
                        f"A non-archived project with this title already "
                        f"exists at id {existing['id']} — reusing it. "
                        f"STOP calling create. Use task_add / "
                        f"task_decompose / task_update on this id. To "
                        f"start a genuinely separate project with the "
                        f"same name, archive the existing one first "
                        f"(action=archive)."
                    )
                return _ok({
                    "refused": True,
                    "action_taken": "reused_existing_project",
                    "existing_project_id": existing["id"],
                    "retry_count": retry_count,
                    "agent_instruction": instruction,
                    # Kept for backward compatibility with callers that
                    # read the happy-path shape, but `refused=True` is
                    # the authoritative signal.
                    "created": existing["id"],
                    "duplicate_of": existing["id"],
                    "note": instruction,
                    "briefing": _briefing(store, existing["id"]),
                })
            pid = store.create_project(
                title=title, kind=kind or "GENERAL", goal=goal,
                metadata=metadata,
            )
            _link_project_in_graph(context, pid, title)
            _set_current(context, pid)
            return _ok({"created": pid, "briefing": _briefing(store, pid)})

        if act == "list":
            return _ok({
                "projects": store.list_projects(status_filter=status_filter),
                "current": getattr(context, "current_project_id", None),
            })

        if act == "get":
            if not project_id:
                return _err("project_id is required for action=get")
            proj = store.get_project(project_id)
            if not proj:
                return _err(f"project not found: {project_id}")
            return _ok(proj)

        if act == "switch":
            if not project_id:
                return _err("project_id is required for action=switch")
            proj = store.get_project(project_id)
            if not proj:
                return _err(f"project not found: {project_id}")
            _set_current(context, project_id)
            return _ok({"switched_to": project_id,
                        "workspace": f"projects/{project_id}",
                        "note": _workspace_note(project_id),
                        "briefing": _briefing(store, project_id)})

        if act == "exit":
            prev = getattr(context, "current_project_id", None)
            _set_current(context, None)
            return _ok({"exited": prev})

        if act == "update":
            if not project_id:
                return _err("project_id is required for action=update")
            fields: Dict[str, Any] = {}
            if title:
                fields["title"] = title
            if goal:
                fields["goal"] = goal
            if status:
                fields["status"] = status
            if kind and kind != "GENERAL":
                fields["kind"] = kind
            if metadata is not None:
                fields["metadata"] = metadata
            if not fields:
                return _err("no updatable fields provided")
            ok = store.update_project(project_id, **fields)
            return _ok({"updated": ok, "fields": list(fields.keys())})

        if act == "delete":
            # Hard, irreversible delete: DB row + all tasks/artifacts/events
            # (cascade) + the on-disk workspace (<sandbox>/projects/<id>/).
            # Use action=archive to merely hide a resumable project.
            if not project_id:
                return _err("project_id is required for action=delete")
            ok = store.delete_project(project_id, hard=True)
            if getattr(context, "current_project_id", None) == project_id:
                _set_current(context, None)
            return _ok({"deleted": ok, "hard": True,
                        "note": "Project, its tasks/artifacts/events, and its "
                                "workspace files were permanently removed."})

        if act == "archive":
            # Soft delete: flips status to ARCHIVED; the project (and its
            # files) survive and can be brought back with action=resume.
            if not project_id:
                return _err("project_id is required for action=archive")
            ok = store.delete_project(project_id, hard=False)
            if getattr(context, "current_project_id", None) == project_id:
                _set_current(context, None)
            return _ok({"archived": ok,
                        "note": "Project hidden but kept; use action=resume to "
                                "bring it back, or action=delete to remove it "
                                "permanently."})

        if act == "resume":
            if not project_id:
                return _err("project_id is required for action=resume")
            proj = store.get_project(project_id)
            if not proj:
                return _err(f"project not found: {project_id}")
            _set_current(context, project_id)
            store.log_event(project_id, None, "project_resumed", {})
            _b = _briefing(store, project_id)
            if isinstance(_b, dict):
                _b = {**_b, "workspace": f"projects/{project_id}",
                      "note": _workspace_note(project_id)}
            return _ok(_b)

        if act == "status":
            cur = getattr(context, "current_project_id", None)
            if not cur:
                return _ok({"current": None, "mode": "free_chat"})
            return _ok({"current": cur, "mode": "project",
                        "briefing": _briefing(store, cur)})

        # ---- task lifecycle ---------------------------------------------

        if act == "task_add":
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            if not description:
                return _err("description is required for action=task_add")
            try:
                dep = DependencyType[dependency_type.upper()]
            except KeyError:
                return _err(f"bad dependency_type: {dependency_type}")
            # Sibling duplicate guard: the 2026-04-19 trace 94 showed
            # Qwen calling task_add for the same description on
            # consecutive turns (re-read the user's bullet list and
            # restarted from #1). Refuse when a non-terminal sibling
            # has the same trimmed/lowercased description — same
            # reasoning as the project-level guard.
            normalized = description.strip().lower()
            for existing in store.list_tasks(project_id):
                if existing.get("parent_id") != parent_id:
                    continue
                if existing.get("status") in {"DONE", "FAILED"}:
                    continue
                if (existing.get("description") or "").strip().lower() == normalized:
                    return _ok({
                        "refused": True,
                        "action_taken": "reused_existing_task",
                        "existing_task_id": existing["id"],
                        "task_id": existing["id"],  # back-compat
                        "agent_instruction": (
                            f"A sibling task with this description already "
                            f"exists at id {existing['id']} in status "
                            f"{existing['status']}. You are likely looping "
                            f"over the user's list and restarted from the "
                            f"beginning. STOP. Move on to the NEXT item in "
                            f"the user's list — not this one again."
                        ),
                    })
            plan = ProjectPlan(store, project_id)
            tid = plan.add_task(
                description=description, parent_id=parent_id,
                dependency_type=dep,
                alternatives=alternatives, postconditions=postconditions,
            )
            _link_task_in_graph(context, project_id, tid, description)
            return _ok({
                "task_id": tid,
                "status": "PENDING",
                "agent_instruction": (
                    "Task added in PENDING status. Added != Done. DO NOT "
                    "call task_update with status=DONE on this task until "
                    "the described work is ACTUALLY complete (file "
                    "written, test passed, etc). If the user asked only "
                    "to ADD tasks (not execute), STOP here — do not "
                    "advance the task."
                ),
            })

        if act == "task_update":
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            # Bulk update path: `task_ids=[...]` lets the agent mark
            # several tasks DONE/FAILED/etc. in one tool call instead
            # of looping turn-by-turn (which routinely truncates and
            # blows the budget). Each id is processed independently;
            # missing ids are reported but don't abort the batch.
            target_ids = list(task_ids) if task_ids else []
            if task_id and task_id not in target_ids:
                target_ids.insert(0, task_id)
            if not target_ids:
                return _err("task_id or task_ids is required for action=task_update")

            if status:
                try:
                    st_enum = TaskStatus[status.upper()]
                except KeyError:
                    return _err(f"bad status: {status}")
            else:
                st_enum = None

            plan = ProjectPlan(store, project_id)
            updated: List[Dict[str, Any]] = []
            missing: List[str] = []
            for tid in target_ids:
                if tid not in plan.tree.nodes:
                    missing.append(tid)
                    continue
                if st_enum is not None:
                    plan.update_status(tid, st_enum, result=result,
                                       failure_reason=failure_reason)
                extras: Dict[str, Any] = {}
                # Per-task description rewrites only make sense in the
                # single-id case — broadcasting one description across
                # many tasks is almost certainly a mistake.
                if description and len(target_ids) == 1:
                    extras["description"] = description
                if alternatives is not None:
                    extras["alternatives"] = alternatives
                if postconditions is not None:
                    extras["postconditions"] = postconditions
                if extras:
                    store.update_task(tid, **extras)
                updated.append({k: store.get_task(tid).get(k)
                                for k in ("id", "status", "result_summary")})
            payload: Dict[str, Any] = {"updated": updated,
                                       "count": len(updated)}
            if missing:
                payload["missing"] = missing
            return _ok(payload)

        if act == "task_decompose":
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            if not subtasks:
                return _err("subtasks list is required for action=task_decompose")
            plan = ProjectPlan(store, project_id)
            # Three accepted shapes:
            #   (a) task_id given and exists       → decompose under that task
            #   (b) task_id absent OR equals project_id → top-level fan-out
            #   (c) task_id given but unknown      → error
            # Shape (b) is the natural "fresh project, here are my tasks"
            # case — without it the LLM has to first create a root task,
            # then decompose it, which is awkward enough that Qwen models
            # routinely loop on the error and create duplicate projects.
            target_id = task_id
            if target_id and target_id == project_id:
                target_id = None  # treat project handle as "top level"
            if target_id and target_id not in plan.tree.nodes:
                return _err(f"unknown task: {target_id}")

            ids: List[str] = []
            pairs: List[tuple] = []  # (task_id, description) for graph linking
            try:
                if target_id:
                    ids = plan.decompose(target_id, subtasks)
                    # decompose drops blank descriptions, so the non-blank
                    # (stripped) subtasks align 1:1 with the returned ids.
                    # zip-ing against the RAW subtasks shifted the pairing
                    # whenever a blank was dropped → wrong graph descriptions.
                    non_blank = [d.strip() for d in subtasks if d and d.strip()]
                    pairs = list(zip(ids, non_blank))
                else:
                    for desc in subtasks:
                        if not desc or not desc.strip():
                            continue
                        tid = plan.add_task(desc.strip())
                        ids.append(tid)
                        pairs.append((tid, desc.strip()))
            except ValueError as e:
                return _err(str(e))
            for tid, desc in pairs:
                _link_task_in_graph(context, project_id, tid, desc)
            return _ok({"created": ids,
                        "parent_id": target_id})

        if act == "task_next":
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            plan = ProjectPlan(store, project_id)
            nxt = plan.next_ready_leaf()
            if not nxt:
                return _ok({"next": None, "reason": "no READY/PENDING leaf available"})
            return _ok({"next": {"id": nxt.id, "description": nxt.description,
                                 "status": nxt.status.value}})

        if act == "task_list":
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            tasks = store.list_tasks(project_id, status_filter=status_filter)
            # Slim down by default: a 50-task project with full rows
            # easily clears 100KB and triggers context offloading,
            # which the LLM then reads as "something was wrong".
            # `verbose=true` (passed via metadata) opts back in.
            verbose = bool((metadata or {}).get("verbose"))
            if not verbose:
                tasks = [
                    {k: t.get(k) for k in _TASK_LIST_SLIM_FIELDS if k in t}
                    for t in tasks
                ]
            return _ok({"tasks": tasks, "count": len(tasks)})

        # ---- artifacts / events ------------------------------------------

        if act == "artifact_add":
            if not task_id:
                return _err("task_id is required for action=artifact_add")
            if not artifact_kind:
                return _err("artifact_kind is required (file|url|note|tool_call)")
            aid = store.add_artifact(task_id, artifact_kind, payload)
            return _ok({"artifact_id": aid})

        if act == "event_log":
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            return _ok({
                "events": store.list_events(
                    project_id, limit=limit, event_type=event_type
                ),
            })

        # ---- self-advancing loop ---------------------------------------

        if act == "autoadvance":
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            from ..core.project_advancer import advance_once

            tool_runner = None
            tools_map = None
            try:
                from .registry import get_available_tools
                tools_map = get_available_tools(context)
            except Exception:
                tools_map = None
            if tools_map:
                async def _run(name: str, args: Dict[str, Any]) -> str:
                    handler = tools_map.get(name)
                    if not handler:
                        return f"ERROR: tool {name} unavailable"
                    return await handler(**args)
                tool_runner = _run
            result = await advance_once(context, project_id,
                                        tool_runner=tool_runner)
            return _ok({
                "ok": result.ok,
                "task_id": result.task_id,
                "classification": result.classification,
                "summary": result.summary,
                "artifact_id": result.artifact_id,
            })

        # ---- inbox promotion (suggestion-accepted path) -----------------

        if act == "promote_from_context":
            if not title:
                return _err("title is required for promotion")
            pid = store.create_project(
                title=title, kind=kind or "GENERAL", goal=goal,
                metadata={"promoted_from_context": True},
            )
            _link_project_in_graph(context, pid, title)
            plan = ProjectPlan(store, pid)
            root_desc = title if not goal else goal
            root = plan.add_task(root_desc)
            _link_task_in_graph(context, pid, root, root_desc)
            if subtasks:
                ids = plan.decompose(root, subtasks)
                # Pair against the non-blank (stripped) subtasks — decompose
                # drops blanks, so zip-ing the raw list would misalign.
                non_blank = [d.strip() for d in subtasks if d and d.strip()]
                for tid, desc in zip(ids, non_blank):
                    _link_task_in_graph(context, pid, tid, desc)
            if context_summary:
                store.log_event(pid, root, "context_snapshot",
                                {"summary": context_summary[:2000]})
            _set_current(context, pid)
            return _ok({"promoted": pid, "briefing": _briefing(store, pid)})

    except ValueError as e:
        return _err(str(e))
    except Exception as e:
        logger.exception("manage_projects failed")
        return _err(f"internal error: {e}")

    return _err("unreachable")


# ------------------------------------------------------------------ tool def

MANAGE_PROJECTS_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "manage_projects",
        "description": (
            "Create and operate on long-term projects (coding or general) "
            "with task/subtask trees that survive across sessions. Use this "
            "when the user wants to START a new named multi-step effort, "
            "RESUME prior work, inspect the task tree, advance a task, or "
            "switch projects. Do NOT use this for ad-hoc chat or one-shot "
            "queries — those stay in free-chat mode. Do NOT call `create` "
            "twice for the same effort — if a project already appears in "
            "DYNAMIC SYSTEM STATE under CURRENT PROJECT, work on THAT one. "
            "Workflow: `create` or `switch` to enter project mode; "
            "`task_decompose` (with ONLY `subtasks`, no `task_id`) to "
            "fan out a fresh project's top-level tasks; `task_decompose` "
            "WITH `task_id` to break a specific task into subtasks; "
            "`task_add` for one-off SINGLE additions — if the user "
            "provides a LIST (numbered, bulleted, or comma-separated) "
            "of 2+ tasks to add, use `task_decompose` with "
            "subtasks=[…] instead; looping task_add N times wastes N-1 "
            "turns AND frequently produces duplicates when the model "
            "re-reads the user's list mid-loop. `task_next` to find "
            "the next leaf to execute; `task_update` to mark DONE/"
            "FAILED/PAUSED/NEEDS_USER (pass `task_ids=[…]` to update "
            "many at once — strongly preferred over looping). IMPORTANT: "
            "`task_add` creates tasks in PENDING status; 'Added' does "
            "NOT mean 'Done'. Do NOT mark a task DONE until its "
            "described work is actually complete. `resume` when "
            "the user asks to pick up "
            "an old project; `exit` to leave project mode; "
            "`archive` to HIDE a project (reversible — status→ARCHIVED, "
            "files kept, `resume` brings it back); `delete` to "
            "PERMANENTLY remove a project and ALL its data — tasks, "
            "artifacts, events, AND its workspace files on disk (NOT "
            "reversible). Use `delete` only when the user clearly means "
            "to erase it; otherwise prefer `archive`. "
            "`promote_from_context` only when the user has explicitly "
            "accepted a suggestion to convert the current chat into a "
            "project."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": sorted(_ACTIONS),
                    "description": "Which sub-operation to perform.",
                },
                "title": {"type": "string", "description": "Project title (create/promote_from_context)."},
                "kind": {"type": "string", "enum": ["CODING", "GENERAL"],
                         "description": "Project kind (defaults to GENERAL)."},
                "goal": {"type": "string", "description": "One-sentence project goal."},
                "metadata": {"type": "object", "description": "Optional free-form project metadata."},
                "project_id": {"type": "string", "description": "Target project id (omit for the current project)."},
                "task_id": {"type": "string",
                            "description": "A SINGLE task id (bare string like 'eed65da9a1e4'). For task_* and artifact_add actions. DO NOT wrap it in brackets — `task_id=\"[\"abc\"]\"` is wrong; use `task_ids=[\"abc\"]` instead for the list form."},
                "task_ids": {"type": "array", "items": {"type": "string"},
                             "description": "A LIST of task ids for batch task_update only — e.g. mark several tasks DONE in one call. Use this (not task_id) whenever you have more than one id. Pass as a real JSON array, not a string."},
                "parent_id": {"type": "string", "description": "Parent task id when creating a subtask."},
                "description": {"type": "string", "description": "Task description (task_add/task_update)."},
                "status": {"type": "string",
                           "enum": ["PENDING", "READY", "IN_PROGRESS", "DONE",
                                    "FAILED", "BLOCKED", "PAUSED", "NEEDS_USER"]},
                "dependency_type": {"type": "string", "enum": ["ALL", "ANY", "BEST"]},
                "alternatives": {"type": "array", "items": {"type": "string"}},
                "postconditions": {"type": "array", "items": {"type": "string"}},
                "result": {"type": "string", "description": "Short result summary (task_update with status=DONE)."},
                "failure_reason": {"type": "string"},
                "subtasks": {"type": "array", "items": {"type": "string"},
                             "description": "Ordered subtask descriptions for task_decompose / promote_from_context."},
                "artifact_kind": {"type": "string",
                                  "enum": ["file", "url", "note", "tool_call"]},
                "payload": {"type": "string"},
                "status_filter": {"type": "string"},
                "limit": {"type": "integer"},
                "event_type": {"type": "string"},
                "context_summary": {"type": "string",
                                    "description": "Optional textual snapshot captured at promotion time."},
            },
            "required": ["action"],
        },
    },
}
