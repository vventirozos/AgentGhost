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
from ..utils.constraints import extract_constraints
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
    # artifacts / events / durable working memory
    "artifact_add", "event_log", "ledger", "config",
    # inbox promotion (suggestion-accepted path)
    "promote_from_context",
    # self-advancing loop
    "autoadvance",
    # auto-research
    "research", "research_list",
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


def _no_active_project(store) -> str:
    """Non-error response for a READ action (get / task_list / task_next) invoked
    with no resolvable project.

    Returns the project list + a switch hint instead of an ``ERROR:`` string —
    which the agent loop scores as a failure and burns a STRIKE for. Forgetting
    to say WHICH project is a recoverable usage slip, not a failure: the model
    just needs to pick one and ``switch``. Returning the list here also saves
    the extra ``action=list`` round-trip the model otherwise does to recover
    (observed live: ``action=get`` struck the model at the very start of a
    'resume my project' turn, before it listed + switched)."""
    try:
        projects = store.list_projects()
    except Exception:
        projects = []
    return _ok({
        "no_active_project": True,
        "projects": projects,
        "agent_instruction": (
            "No project is active in this conversation — this is NOT an error. "
            "Pick one from `projects` above and call `manage_projects "
            "action=switch project_id=\"<id>\"`, then retry your read."
        ),
    })


_PROJECT_KEY_PREFIX = "proj::"
_CURRENT_SENTINEL = "__current_project__"
# Conversation that owns the current project activation. Written next to
# `__current_project__` by `_set_current` and compared by
# `reconcile_conversation` at every request start. Without this binding,
# `context.current_project_id` was process-global STICKY state: a project
# activated in one chat silently captured every other conversation's file
# writes into `<sandbox>/projects/<id>/` (observed in production — an SQL
# snippet request landed `migration.sql` inside an unrelated project's
# workspace).
_CURRENT_CONV_SENTINEL = "__current_project_conv__"


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
            and k not in (_CURRENT_SENTINEL, _CURRENT_CONV_SENTINEL)
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
            and k not in (_CURRENT_SENTINEL, _CURRENT_CONV_SENTINEL)
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
                sp.delete(_CURRENT_CONV_SENTINEL)
            else:
                sp.set(_CURRENT_SENTINEL, project_id)
                # Bind the activation to the conversation that asked for it
                # (set per-request by `reconcile_conversation`). Empty/None
                # means "unbound" — reconcile treats that as not owned by
                # ANY conversation, so a stale activation can never leak.
                sp.set(_CURRENT_CONV_SENTINEL,
                       getattr(context, "conversation_key", None) or "")
        except Exception:
            pass
    if project_id and project_id != prev:
        _hydrate_scratchpad(context, project_id)
    elif project_id is None:
        # Leaving project mode: clear the hydrated keys so free chat
        # starts fresh. Named sentinels survive.
        _hydrate_scratchpad(context, None)


def _delete_eligibility_error(context, store, rid: str) -> Optional[str]:
    """Return a refusal string when `rid` is NOT eligible for a hard
    delete in the current request, else None.

    Hard delete is permanent (row + workspace wiped), and the model only
    ever reaches it from a user message. A bare "delete it" can only
    refer to something the user has actually seen, so eligibility is:

      * the harness recorded no request-start snapshot
        (``context.request_start_project_id`` attribute absent — direct
        tool tests / non-chat surfaces): gate inactive;
      * ``rid`` was the active project when the user's message arrived
        (the snapshot), i.e. the thing "it" plausibly refers to;
      * the user's own message names the project, by title or id.

    Everything else is refused — most importantly a project the agent
    created seconds earlier in the SAME request. Observed live: one
    "i don't really like it, delete it an make something else" cascaded
    into six successive hard deletes, five of them of self-created
    projects the user never saw, because each new turn re-read the
    instruction as unfulfilled.
    """
    if not hasattr(context, "request_start_project_id"):
        return None
    if rid == getattr(context, "request_start_project_id", None):
        return None
    user_msg = str(getattr(context, "last_user_content", "") or "").lower()
    if user_msg:
        if rid.lower() in user_msg:
            return None
        try:
            proj = store.get_project(rid)
        except Exception:
            proj = None
        title = (proj or {}).get("title") if isinstance(proj, dict) else ""
        if title and str(title).lower() in user_msg:
            return None
    return (
        f"REFUSED: hard delete of '{rid}' is not allowed in this request. "
        "It was not the active project when the user's message arrived, and "
        "the message does not name it — the user has never seen this project, "
        "so 'delete it' cannot mean this one. If YOU created it earlier in "
        "this request: STOP cycling ideas. Keep this project and BUILD it "
        "now. To remove a different project, the user must name it "
        "explicitly (or use action=archive, which is reversible)."
    )


def conversation_fingerprint(messages) -> str:
    """Stable identity for a conversation: hash of its FIRST user message.

    The chat API ships the full message history on every request and
    carries no conversation id, so the first user turn — constant for the
    lifetime of a conversation, different across conversations — is the
    only stable thing to key on. Returns "" when there is no user message
    (callers treat "" as "owns nothing").
    """
    first = next(
        (m.get("content") for m in (messages or [])
         if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    if isinstance(first, list):  # multimodal content blocks
        first = " ".join(
            b.get("text", "") for b in first
            if isinstance(b, dict) and b.get("type") == "text"
        )
    if not first:
        return ""
    import hashlib
    return hashlib.sha1(str(first).strip().encode("utf-8", "replace")).hexdigest()


def _message_references_project(context, project_id: str) -> bool:
    """True when THIS request's user message names the given project by id or
    title. The chat API ships no stable conversation id, so a project bound to
    one conversation gets parked for every other request whose first-message
    fingerprint differs — including later turns of the SAME human session once a
    per-turn 'Context:' preamble changes the first message. When the user
    explicitly references the project ("proceed with task 3 of the PetAI
    project"), that intent overrides the fingerprint mismatch. Best-effort;
    never raises."""
    msg = str(getattr(context, "last_user_content", "") or "").lower()
    if not msg:
        return False
    pid = str(project_id or "").lower()
    if pid and (pid in msg or pid[:8] in msg):
        return True
    store = getattr(context, "project_store", None)
    if store is None:
        return False
    try:
        proj = store.get_project(project_id) or {}
        title = str(proj.get("title") or "").strip().lower()
        # Require a reasonably distinctive title (≥3 chars) and a whole-word
        # match so a generic one-word title can't latch onto stray prose.
        if len(title) >= 3 and re.search(r"\b" + re.escape(title) + r"\b", msg):
            return True
    except Exception:
        return False
    return False


def reconcile_conversation(context, conv_key: str):
    """Scope the active project to the conversation that activated it.

    Called once at request start, BEFORE any sandbox listing or file op.
    `context.current_project_id` is process-global, so without this a
    project switched-to in one conversation stayed active for every other
    conversation hitting the same process — their file writes landed in
    `<sandbox>/projects/<id>/` and their turns carried that project's
    scratchpad. Policy:

    * binding's conversation == `conv_key` → (re)activate the bound project
      for this request (returning to the owning conversation resumes it);
    * binding belongs to another conversation, is unbound (legacy bare
      sentinel written before the conv sentinel existed), or `conv_key` is
      "" → deactivate for this request, PRESERVING the binding so the
      owning conversation gets it back.
    """
    context.conversation_key = conv_key or ""
    sp = getattr(context, "scratchpad", None)
    if sp is None:
        return
    try:
        bound_pid = sp.get(_CURRENT_SENTINEL)
        bound_conv = sp.get(_CURRENT_CONV_SENTINEL)
    except Exception:
        return
    cur = getattr(context, "current_project_id", None)
    if not bound_pid or not isinstance(bound_pid, str):
        return
    owns = bool(conv_key) and bound_conv == conv_key
    # Escape hatch: even on a fingerprint mismatch, an explicit user reference
    # to the project ("the PetAI project") means they want to work on it now —
    # keep/activate it instead of parking it and forcing a recovery dance that
    # loops the agent (observed live: request D7 called manage_projects get 3x,
    # then hit the no-progress loop breaker, then a scrub fallback — all because
    # the project it was told to advance had just been deactivated).
    referenced = (not owns) and _message_references_project(context, bound_pid)
    if owns or referenced:
        if referenced:
            # Re-bind the conversation to THIS request's key so the binding
            # FOLLOWS the active conversation. Without this, `current_project_id`
            # (process-global) can be cleared mid-request by another reconcile,
            # and the `_conversation_bound_project` fallback in
            # `project_scoped_sandbox` would still fail (bound_conv != conv) — so
            # file_system / report_pdf writes fall back to the sandbox ROOT
            # instead of projects/<id>/ (observed live: a report PDF and stray
            # project directories created at the sandbox root).
            try:
                sp.set(_CURRENT_CONV_SENTINEL, conv_key or "")
            except Exception:
                pass
        if cur != bound_pid:
            context.current_project_id = bound_pid
            _wm = getattr(context, "workspace_model", None)
            if _wm is not None:
                try:
                    _wm.current_project_id = bound_pid
                except Exception:
                    pass
            _hydrate_scratchpad(context, bound_pid)
            pretty_log("Project Scope",
                       (f"Conversation owns project '{bound_pid}' — reactivated"
                        if owns else
                        f"User referenced project '{bound_pid}' — activated for "
                        "this request (conversation fingerprint differs)"),
                       icon=Icons.BRAIN_PLAN)
        return
    if cur:
        # Another conversation's project is live — park it for this request.
        _snapshot_scratchpad(context, cur)
        context.current_project_id = None
        _wm = getattr(context, "workspace_model", None)
        if _wm is not None:
            try:
                _wm.current_project_id = ""
            except Exception:
                pass
        _hydrate_scratchpad(context, None)
        pretty_log("Project Scope",
                   f"Project '{cur}' belongs to another conversation — "
                   "deactivated for this request",
                   icon=Icons.BRAIN_PLAN)


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
        "with file_system and run with execute live HERE — ALWAYS reference "
        "them by BARE name (e.g. 'chart.py', 'notes.md', 'sub/app.js'). Do "
        "NOT prefix paths with '/workspace/', 'sandbox/', or 'projects/"
        f"{project_id}/' — that double-nests the path and 404s (browser file "
        "URLs especially). Scratch files are cleaned up automatically when "
        "the project completes."
    )


# Build-flavoured signals in a project title/goal. A GENERAL project defaults
# its autoadvance leaves to research (which web-searches build tasks — observed
# live), so a clearly-software goal should be created as CODING up front.
_CODING_PROJECT_RE = re.compile(
    r"\b(build|implement|coding|app|apps|application|website|web ?app|web ?page|"
    r"webpage|game|games|operating system|cli|command line|api|script|program|"
    r"dashboard|front ?end|frontend|back ?end|backend|server|library|module|"
    r"component|html|css|javascript|typescript|python|single[ -]file|browser|"
    r"interface|parser|compiler|algorithm|webgl|canvas|simulator|emulator)\b",
    re.I,
)


# A goal asking for ONE cohesive file. Such a deliverable is built best in a
# single turn (one-shot generation) — splitting it into per-feature tasks that
# must merge into one file produces global collisions and a broken page
# (observed live). Detected so create/decompose keep it as one task.
_SINGLE_FILE_RE = re.compile(
    r"\b(single[ -]file|one[ -]file|single (html|page|file)|"
    r"one (html|index\.html|page|html file)|in (a |one )?single (html|file|page)|"
    r"all[ -]in[ -]one (file|html|page)|self[ -]contained (html|file|page))\b",
    re.I,
)


def _is_cohesive_single_file(title: str, goal: str) -> bool:
    return bool(_SINGLE_FILE_RE.search(f"{title or ''} {goal or ''}"))


def _feature_key(desc: str) -> str:
    """A coarse identity for a task — the feature name before the first colon
    (or the first few words). 'Core Shell: HTML structure…' and 'Core Shell:
    index.html skeleton…' share the key 'core shell', so they dedup."""
    d = " ".join((desc or "").strip().lower().split())
    head = d.split(":", 1)[0]
    return head[:40] if head else d[:40]


def _filter_duplicate_subtasks(store, project_id: str, subtasks):
    """Drop subtask descriptions whose feature already exists as a non-terminal
    task, and drop intra-list duplicates. Stops the project piling up duplicate
    tasks when the model calls create-with-subtasks AND task_decompose, or
    decomposes twice (observed live — duplicate Core Shell / File Explorer that
    then failed)."""
    try:
        existing = store.list_tasks(project_id)
    except Exception:
        existing = []
    terminal = {"DONE", "FAILED", "BLOCKED", "ARCHIVED"}
    seen = {_feature_key(t.get("description")) for t in existing
            if str(t.get("status", "")).upper() not in terminal}
    out = []
    for d in (subtasks or []):
        if not d or not d.strip():
            continue
        k = _feature_key(d)
        if k in seen:
            continue
        seen.add(k)
        out.append(d)
    return out


def _infer_kind(title: str, goal: str, explicit_kind: str) -> str:
    """Pick the project kind. An explicit CODING (or any non-GENERAL kind) from
    the caller wins; otherwise infer CODING from a build-flavoured title/goal
    so the autoadvancer treats the project's leaves as coding work rather than
    defaulting them to research."""
    ek = (explicit_kind or "").strip().upper()
    if ek and ek != "GENERAL":
        return ek
    if _CODING_PROJECT_RE.search(f"{title or ''} {goal or ''}"):
        return "CODING"
    return ek or "GENERAL"


def _parse_advance_count(count) -> Optional[int]:
    """Coerce a model-supplied `count` for autoadvance into max_tasks:
    a positive int, or ``None`` for "all" (run to completion). Anything
    blank/unparseable defaults to 1 — a vague nudge never runs away."""
    if count is None:
        return 1
    s = str(count).strip().lower()
    if s in ("all", "*", "rest", "remaining", "everything", "finish"):
        return None
    try:
        return max(1, int(s))
    except (TypeError, ValueError):
        return 1


_ADVANCE_STOP_BLURB = {
    "project_done": "All tasks are complete — the project is done.",
    "completed_with_failures": "Advanced every task it could; some FAILED and "
                               "were skipped (listed below) — retry those.",
    "count_reached": "Advanced the requested number of tasks; more remain.",
    "hard_cap": "Hit the safety cap mid-run — call autoadvance again to continue.",
    "repeated_failures": "Stopped: several tasks failed in a row — likely a "
                         "systemic problem (e.g. a broken foundation). Fix it, "
                         "then re-run autoadvance.",
    "needs_user": "Paused: a task needs your input/decision before continuing.",
    "budget_or_inactive": "Paused: the project's step budget is exhausted "
                          "(raise steps_cap to continue) or it is not ACTIVE.",
    "failed": "Stopped: a task FAILED — review it before continuing.",
    "no_store": "Project store unavailable.",
}


def _advance_batch_instruction(batch, max_tasks) -> str:
    blurb = _ADVANCE_STOP_BLURB.get(batch.stop_reason, batch.stop_reason)
    done = sum(1 for a in batch.advanced if (a.get("status") or "") == "DONE")
    failed = sum(1 for a in batch.advanced if (a.get("status") or "") == "FAILED")
    tail = f", {failed} FAILED" if failed else ""
    failed_ids = [a.get("task_id") for a in batch.advanced
                  if (a.get("status") or "") == "FAILED" and a.get("task_id")]
    fail_list = (" Failed task ids: " + ", ".join(failed_ids) + ".") if failed_ids else ""
    # Don't pull the built artifacts back into THIS turn to inspect/fix them —
    # on a large build that overflowed the model's context (151K>131K observed
    # live, after the agent read a 110KB file back and fought it for 30 turns).
    # Each task already self-verified; report from the summaries and stop.
    no_reload = ("Do NOT read the full built file(s) back into this turn or try "
                 "to fix them now — on a large build that overflows the context "
                 "window. Report from the summaries above.")
    # These mean "something needs your attention before more progress is
    # possible" — report and hand back; do NOT manually rebuild (that long
    # manual build thrashed ~700s and broke project scoping).
    stopped_early = batch.stop_reason in (
        "failed", "repeated_failures", "budget_or_inactive", "needs_user")
    if stopped_early:
        return (
            f"Autonomous batch ran {batch.count} task(s) ({done} DONE{tail}). "
            f"{blurb}{fail_list} STOP now: tell the user EXACTLY which tasks "
            f"completed and which did not, and ask how to proceed (retry the "
            f"failed task(s), refine them, or take over). Do NOT keep building "
            f"the project yourself this turn — re-running autoadvance after a "
            f"fix continues from where it stopped. {no_reload}"
        )
    return (
        f"Autonomous batch ran {batch.count} task(s) ({done} DONE{tail}). "
        f"{blurb}{fail_list} Summarize for the user what was built (and any "
        f"failed tasks they may want to retry) and STOP — wait for their next "
        f"direction. {no_reload}"
    )


def _resolve_store(context) -> Optional[ProjectStore]:
    return getattr(context, "project_store", None)


# A project in one of these states is finished — the duplicate-create guard
# treats it as superseded (a same-title `create` starts a fresh project)
# rather than reusing it.
_TERMINAL_PROJECT_STATUSES = {"DONE", "FAILED", "BLOCKED", "ARCHIVED"}


def _resolve_project_ref(store, project_id: Optional[str],
                         title: str = "") -> tuple:
    """Resolve a project reference to a concrete id.

    Users (and the model on their behalf) refer to projects by NAME —
    e.g. "delete the minecraft clone project" — but the destructive
    actions key on the opaque hex id. When the given ``project_id`` isn't a
    real id, fall back to a case-insensitive title match (also accepting an
    explicit ``title``). This is what stops a delete from silently no-
    op'ing when a name was passed where an id was expected.

    Returns ``(resolved_id, error)``:
      * a real id was found            → (id, None)
      * exactly one title match        → (id, None)
      * several title matches          → (None, "ambiguous … <ids>")
      * nothing matched                → (None, None)
    """
    if project_id:
        try:
            if store.get_project(project_id):
                return project_id, None
        except Exception:
            pass
    needle = (title or project_id or "").strip().lower()
    if not needle:
        return None, None
    try:
        matches = [p for p in store.list_projects()
                   if (p.get("title") or "").strip().lower() == needle]
    except Exception:
        matches = []
    if len(matches) == 1:
        return matches[0]["id"], None
    if len(matches) > 1:
        listing = ", ".join(f"{p['id']} ({p['status']})" for p in matches)
        return None, (
            f"ambiguous: {len(matches)} projects are titled '{needle}'. "
            f"Pass a specific project_id — candidates: {listing}"
        )
    return None, None


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


def _link_concepts_in_graph(context, project_id: str):
    """Extract the project's libraries/techniques and link them to SHARED
    canonical nodes in the knowledge graph (feature 3A) so cross-project
    retrieval can bridge projects that use the same tech. Re-runnable —
    add_triplets is idempotent (re-adds just bump edge weight). Best-effort.
    """
    gm = getattr(context, "graph_memory", None)
    store = getattr(context, "project_store", None)
    if gm is None or store is None or not project_id:
        return
    try:
        from ..core.project_concepts import link_project_concepts
        proj = store.get_project(project_id)
        if proj:
            link_project_concepts(gm, proj)
    except Exception:
        logger.debug("graph concept link skipped", exc_info=True)


def _briefing(store: ProjectStore, project_id: str) -> Dict[str, Any]:
    """Compact status snapshot suitable for a resume banner."""
    proj = store.get_project(project_id)
    if not proj:
        return {}
    plan = ProjectPlan(store, project_id)
    nxt = plan.next_ready_leaf()
    events = store.list_events(project_id, limit=5)
    try:
        from ..core.project_research import get_research_index
        research = [
            {"topic": r.get("topic"), "path": r.get("path")}
            for r in get_research_index(store, project_id)[-5:]
        ]
    except Exception:
        research = []
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
        "research": research,
    }


# Tasks whose description implies a RUNNABLE / VISUAL artifact — a page, a
# game, a rendered scene, a chart. Marking one DONE without any verification
# evidence is the "wrote index.html, declared victory, never looked" anti-
# pattern from the live Minecraft run.
#
# Kept HIGH-PRECISION on purpose. The first cut used broad words (game,
# scene, render, animation, 3d, graphics) and false-positived on a
# *philosophy* project: a text reflection titled "…whether the scene is
# pre-existing or being constructed" matched " scene", and "game theory"
# matched "game" — gating ordinary research tasks and costing ~6 wasted
# turns. So the set is now limited to tokens that are unambiguous in a
# software-task context, matched on word boundaries (so "hud" doesn't fire
# inside "shudder", nor "3d" inside "add").
_VISUAL_ARTIFACT_KEYWORDS = (
    "webgl", "three.js", "threejs", "babylon.js", "pixi.js", "canvas",
    "shader", "hud", "sprite", "playable", "gameplay", "game loop",
    "render loop", "frame loop", "minecraft", "frontend", "front-end",
)

_VISUAL_ARTIFACT_RE = re.compile(
    r"(?<![a-z0-9])(?:"
    + "|".join(re.escape(k) for k in _VISUAL_ARTIFACT_KEYWORDS)
    + r")(?![a-z0-9])"
)


def _is_visual_artifact_task(description: str) -> bool:
    return bool(_VISUAL_ARTIFACT_RE.search((description or "").lower()))


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
    constraints: Optional[List[str]] = None,
    depends_on: Optional[List[str]] = None,
    sequential: bool = False,
    result: str = "",
    failure_reason: str = "",
    subtasks: Optional[List[str]] = None,
    # artifacts
    artifact_kind: str = "",
    payload: str = "",
    deliverables: Optional[List[str]] = None,
    # durable project working memory
    ledger: str = "",
    config_key: str = "",
    config_value: str = "",
    # autonomous batch pacing
    count: Any = None,
    # research
    topic: str = "",
    topics: Optional[List[str]] = None,
    max_topics: int = 5,
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
    constraints = _coerce_str_list(constraints)
    depends_on = _coerce_str_list(depends_on)
    topics = _coerce_str_list(topics)
    task_ids = _coerce_str_list(task_ids)
    # Canonicalize each id in the BULK path the same way the singular task_id
    # is (line ~933) — otherwise an LLM-case-mangled id in task_ids is judged
    # "missing" against the lowercase tree keys and silently skipped.
    if task_ids:
        task_ids = [_canon_pid(t) for t in task_ids]
    deliverables = _coerce_str_list(deliverables)

    # `metadata` is an object param the model often stringifies as JSON — parse
    # it so downstream `dict(metadata)` / `metadata.get(...)` don't raise on a str.
    if isinstance(metadata, str):
        _m = metadata.strip()
        if _m[:1] in ("{", "["):
            try:
                metadata = json.loads(_m)
            except (ValueError, TypeError):
                metadata = None
        else:
            metadata = None

    # `result` is the canonical name for a task's completion summary, but the
    # model very naturally reaches for `result_summary` / `summary` — and when
    # it does, the value lands in **_unused and the DONE-gate never sees the
    # evidence, so a legit close keeps getting refused (observed: 6 wasted
    # turns). Accept the obvious aliases.
    if not (result or "").strip():
        result = (_unused.get("result_summary") or _unused.get("summary")
                  or _unused.get("result_text") or result or "")

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
    # BUT: never let the auto-fill shadow an EXPLICIT title for a title-
    # resolvable action — auto-filling `current` there made
    # `delete title="chess"` (with any project active) resolve to and HARD-
    # DELETE the ACTIVE project instead of "chess", because _resolve_project_ref
    # short-circuits on a valid project_id and ignores the title.
    # NB: "update" is NOT here — its `title` arg is a RENAME value, not a
    # lookup key, so auto-filling `current` for a rename is correct.
    _TITLE_RESOLVABLE = {"delete", "archive", "get", "switch"}
    if project_id is None and act not in {"create", "list", "promote_from_context"}:
        if not (title and act in _TITLE_RESOLVABLE):
            project_id = getattr(context, "current_project_id", None)

    try:
        # ---- project lifecycle ------------------------------------------

        if act == "create":
            if not title:
                return _err("title is required for action=create")
            # Capture the user's explicit constraints at the moment of
            # creation — both the ones the model passed and the ones we can
            # extract deterministically from the triggering message
            # (negations, "YOU will ..." role assertions, CAPS emphasis).
            # They persist in project metadata, are re-rendered into every
            # briefing, and gate the DONE transition. This is the fix for
            # the chess-game failure where "don't come up with some random
            # AI" survived less than 4s of reasoning.
            req_constraints = list(constraints or [])
            for clause in extract_constraints(
                    getattr(context, "last_user_content", "") or ""):
                if clause.lower() not in {c.lower() for c in req_constraints}:
                    req_constraints.append(clause)
            if req_constraints:
                pretty_log("Constraints",
                           f"captured {len(req_constraints)} on create: "
                           f"{req_constraints[0][:70]}…",
                           icon=Icons.CONSTRAINT)
            normalized = title.strip().lower()
            now = time.time()
            for existing in store.list_projects():
                # Only reuse a same-title project that is still IN-FLIGHT.
                # The guard exists to stop the model spawning duplicates of
                # work that is still in progress — NOT to block starting a
                # genuinely new effort once the old one has finished. Reusing
                # a terminal (DONE/FAILED/BLOCKED/ARCHIVED) project here
                # resurrected a completed "Minecraft Clone" — with its stale
                # tasks and on-disk files — when the user explicitly asked to
                # create a NEW one (observed live, after a delete that never
                # took effect). Terminal projects are superseded by a fresh
                # create; only ACTIVE / PAUSED / NEEDS_USER are reused.
                if existing["status"] in _TERMINAL_PROJECT_STATUSES:
                    continue
                if (existing["title"] or "").strip().lower() != normalized:
                    continue
                # Existence-based guard: an in-flight project with the same
                # title means the user is asking us to continue, not start
                # over. We reuse it regardless of how long ago it was created.
                # ``_DUPLICATE_CREATE_WINDOW_SECONDS`` is honored as a max-age
                # only when explicitly set to an int (tests patch it to 0 to
                # disable the guard).
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
                # A re-issued create that carries NEW explicit constraints is
                # a CORRECTION, not a duplicate: the user restated the request
                # because the existing decomposition missed something (chess
                # incident: the pending "AI opponent" task was treated as
                # "matches exactly what the user wants" after the user had
                # explicitly excluded it). Merge the fresh constraints into
                # the project so briefings/DONE-gates see them.
                prior_constraints = [str(c) for c in
                                     (existing_meta.get("constraints") or [])]
                fresh = [c for c in req_constraints
                         if c.lower() not in {p.lower() for p in prior_constraints}]
                if fresh:
                    existing_meta["constraints"] = prior_constraints + fresh
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
                if fresh:
                    instruction = (
                        "CORRECTION DETECTED: the user re-issued this request "
                        "with explicit constraints the existing plan may "
                        "violate: " + " | ".join(fresh) + ". Do NOT treat the "
                        "pending tasks as 'matching what the user wants' — "
                        "they were decomposed BEFORE this correction. FIRST "
                        "re-read each pending task description against these "
                        "constraints and revise any that conflict "
                        "(task_update with description=...), THEN proceed. "
                        + instruction
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
            # Delete-then-recreate is a rejection signal: the user threw the
            # previous attempt away and is re-asking, usually with added
            # emphasis. Surface it so the model re-plans from the CURRENT
            # message instead of replaying the decomposition it remembers.
            tombstone = None
            try:
                tombstone = store.find_deleted_similar(title)
            except Exception:
                pass
            if tombstone:
                pretty_log("Correction Detect",
                           f"recreate of recently-deleted "
                           f"'{tombstone.get('title')}' — re-planning from "
                           f"current message", icon=Icons.CONSTRAINT)
            create_meta = dict(metadata or {})
            if req_constraints:
                create_meta["constraints"] = req_constraints
            if tombstone:
                create_meta["correction_of"] = tombstone.get("id")
            pid = store.create_project(
                title=title, kind=_infer_kind(title, goal, kind), goal=goal,
                metadata=create_meta,
            )
            _link_project_in_graph(context, pid, title)
            _link_concepts_in_graph(context, pid)
            _set_current(context, pid)
            # If the model passed `subtasks` to create (it routinely does,
            # believing create decomposes too), DECOMPOSE them now instead of
            # silently dropping them. Without this the project was created with
            # ZERO tasks while the model believed the tasks existed — so a
            # later "proceed" found "no plan" and the whole flow derailed
            # (observed live, twice). Top-level fan-out, sequential when asked.
            created_task_ids: List[str] = []
            if _is_cohesive_single_file(title, goal):
                # A cohesive SINGLE FILE is built best in ONE turn (one-shot
                # generation), NOT split into per-feature tasks that must merge
                # into one file and collide (duplicate globals → the page threw
                # on load; observed live). Collapse to ONE build task so the
                # "proceed" path is a single full agent turn — the path that
                # worked before the per-turn split.
                plan = ProjectPlan(store, pid)
                _desc = ("Build the complete single-file deliverable: "
                         f"{(goal or title).strip()}")[:400]
                one = plan.add_task(_desc, constraints=req_constraints or None)
                created_task_ids = [one]
                _link_task_in_graph(context, pid, one, _desc)
                instruction = (
                    "SINGLE-FILE build: a cohesive single file is built best in "
                    "ONE focused turn, so this is ONE task — do NOT split it "
                    "into per-feature tasks (they would have to merge into one "
                    "file, which collides and breaks). Present this one-task "
                    "plan and STOP. When the user says proceed, BUILD THE WHOLE "
                    "FILE YOURSELF in that turn (write it directly, growing it "
                    "with file_system as needed) — do NOT call autoadvance for "
                    "it; the autonomous loop builds fragments unfit for a single "
                    "cohesive file. The project workspace is ALREADY set up: "
                    "write to the BARE filename (e.g. 'index.html') — do NOT "
                    "invent or prefix a 'projects/<name>/' directory, and do "
                    "NOT write the file before this create call has run."
                )
            elif subtasks:
                subtasks = _filter_duplicate_subtasks(store, pid, subtasks)
                plan = ProjectPlan(store, pid)
                prev_id = None
                pairs: List[tuple] = []
                for desc in subtasks:
                    if not desc or not desc.strip():
                        continue
                    deps = [prev_id] if (sequential and prev_id) else None
                    tid = plan.add_task(desc.strip(), depends_on=deps,
                                        constraints=req_constraints or None)
                    created_task_ids.append(tid)
                    pairs.append((tid, desc.strip()))
                    prev_id = tid
                for tid, desc in pairs:
                    _link_task_in_graph(context, pid, tid, desc)
                instruction = (
                    f"Project created WITH {len(created_task_ids)} task(s) from "
                    "your subtasks. Present the plan and STOP — do NOT begin "
                    "executing. Advance ONE task at a time, only when the user "
                    "directs you ('proceed to next task'); for a multi-task "
                    "go-ahead use autoadvance count=<N|all>."
                )
            else:
                instruction = (
                    "Project created. Break the goal into tasks with "
                    "task_decompose — make EACH task own a file or a bounded "
                    "function you can build and verify on its own (e.g. one "
                    "file per feature behind a thin shell), NOT N tasks that "
                    "all edit the same file. Then STOP and present the plan. "
                    "DO NOT begin executing tasks now — advance ONE task at a "
                    "time, and only when the user directs you (e.g. 'proceed to "
                    "next task'). As you work, record durable facts (file "
                    "layout, key APIs) with action=ledger so later turns don't "
                    "re-derive them."
                )
            # Tell the model its working directory moved to the project dir —
            # the SAME note switch/resume return. Without it (create used to
            # omit it), the model wrote `index.html` expecting the sandbox
            # ROOT, then burned ~4 turns discovering the file_system write had
            # been project-scoped and "moving" it (observed live). The note
            # steers it to bare relative filenames that land in the right
            # place the first time.
            if tombstone:
                _ago = max(0, int(time.time() - float(
                    tombstone.get("deleted_at") or 0)))
                instruction = (
                    f"CORRECTION CONTEXT: the user DELETED a similar project "
                    f"('{tombstone.get('title')}', {_ago // 60} min ago) before "
                    f"re-asking. Deleting a build is a rejection of the "
                    f"previous approach — plan from the CURRENT message only; "
                    f"do NOT reuse the remembered decomposition or deliverable "
                    f"shape from the deleted attempt. " + instruction
                )
            if req_constraints:
                instruction = (
                    "EXPLICIT USER CONSTRAINTS captured on this project (they "
                    "gate task completion): "
                    + " | ".join(req_constraints) + ". " + instruction
                )
            return _ok({"created": pid,
                        "tasks_created": created_task_ids,
                        "workspace": f"projects/{pid}",
                        "constraints": req_constraints,
                        "note": _workspace_note(pid),
                        "agent_instruction": instruction,
                        "briefing": _briefing(store, pid)})

        if act == "list":
            projs = store.list_projects(status_filter=status_filter)
            current = getattr(context, "current_project_id", None)
            # Pre-rendered id-first lines + an explicit directive. The ids
            # were always present in the JSON, but the model's user-facing
            # summaries dropped them (2026-07-05: the user was left
            # deleting by TITLE — title refs are ambiguity-prone, and a
            # bare title delete has cascaded before). Making the id the
            # first column of a ready-made display block, with a note that
            # it MUST be shown, is the reliable contract for a small model.
            lines = [
                f"{p.get('id')}  {str(p.get('status') or ''):<9}  "
                f"{p.get('title')}"
                + ("   <- ACTIVE" if p.get("id") == current else "")
                for p in projs
            ]
            return _ok({
                "projects": projs,
                "current": current,
                "display": "\n".join(lines) or "(no projects)",
                "note": (
                    "ALWAYS show each project's id (first column of "
                    "'display') when presenting projects to the user — the "
                    "id is the only unambiguous reference for follow-up "
                    "get/switch/update/delete calls."),
            })

        if act == "get":
            if not project_id:
                return _no_active_project(store)
            # Resolve a title/slug to the real id (e.g. `get petai`) via the
            # shared resolver so a title isn't a hard "not found".
            rid, rerr = _resolve_project_ref(store, project_id)
            if rerr:
                return _err(rerr)                # ambiguous title
            if not rid:
                return _err(f"project not found: {project_id}")
            return _ok(store.get_project(rid))

        if act == "switch":
            if not project_id:
                return _err("project_id is required for action=switch")
            # Resolve a title/slug to the real id so `switch petai` works
            # instead of erroring + striking on a recoverable title/id mixup.
            rid, rerr = _resolve_project_ref(store, project_id)
            if rerr:
                return _err(rerr)                # ambiguous title
            if not rid:
                return _err(f"project not found: {project_id}")
            _set_current(context, rid)
            return _ok({"switched_to": rid,
                        "workspace": f"projects/{rid}",
                        "note": _workspace_note(rid),
                        "briefing": _briefing(store, rid)})

        if act == "exit":
            prev = getattr(context, "current_project_id", None)
            _set_current(context, None)
            return _ok({"exited": prev})

        if act == "update":
            if not project_id:
                return _err("project_id is required for action=update")
            # project_id may be a title/stale id — resolve it so an update
            # doesn't silently match 0 rows and report success.
            _uid, _uerr = _resolve_project_ref(store, project_id, "")
            if _uerr:
                return _err(_uerr)
            if not _uid:
                return _err(f"project not found: {project_id!r} — nothing was updated.")
            project_id = _uid
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
            if not ok:
                return _err(f"update matched no rows for {project_id} — nothing was changed.")
            return _ok({"updated": ok, "fields": list(fields.keys())})

        if act == "delete":
            # Hard, irreversible delete: DB row + all tasks/artifacts/events
            # (cascade) + the on-disk workspace (<sandbox>/projects/<id>/).
            # Use action=archive to merely hide a resumable project.
            if not project_id and not title:
                return _err("project_id (or title) is required for action=delete")
            rid, rerr = _resolve_project_ref(store, project_id, title)
            if rerr:
                return _err(rerr)
            if not rid:
                # Fail LOUDLY. A silent no-op here is what let a "delete"
                # be reported as done while the project survived — and then
                # be resurrected by the duplicate-create guard.
                return _err(
                    f"project not found: {project_id or title!r} — NOTHING was "
                    f"deleted. Use action=list to see the real ids, then pass "
                    f"the exact project_id."
                )
            gate = _delete_eligibility_error(context, store, rid)
            if gate:
                pretty_log("Project Guard",
                           f"Hard delete of '{rid}' refused (not user-visible "
                           "in this request)", icon=Icons.STOP)
                return _err(gate)
            ok = store.delete_project(rid, hard=True)
            if not ok:
                return _err(f"delete failed for {rid} — nothing was removed.")
            if getattr(context, "current_project_id", None) == rid:
                _set_current(context, None)
            return _ok({"deleted": True, "project_id": rid, "hard": True,
                        "note": "Project, its tasks/artifacts/events, and its "
                                "workspace files were permanently removed."})

        if act == "archive":
            # Soft delete: flips status to ARCHIVED; the project (and its
            # files) survive and can be brought back with action=resume.
            if not project_id and not title:
                return _err("project_id (or title) is required for action=archive")
            rid, rerr = _resolve_project_ref(store, project_id, title)
            if rerr:
                return _err(rerr)
            if not rid:
                return _err(
                    f"project not found: {project_id or title!r} — NOTHING was "
                    f"archived. Use action=list to see the real ids."
                )
            ok = store.delete_project(rid, hard=False)
            if not ok:
                return _err(f"archive failed for {rid} — nothing was changed.")
            if getattr(context, "current_project_id", None) == rid:
                _set_current(context, None)
            return _ok({"archived": True, "project_id": rid,
                        "note": "Project hidden but kept; use action=resume to "
                                "bring it back, or action=delete to remove it "
                                "permanently."})

        if act == "resume":
            if not project_id:
                return _err("project_id is required for action=resume")
            proj = store.get_project(project_id)
            if not proj:
                return _err(f"project not found: {project_id}")
            # Flip an ARCHIVED project back to ACTIVE — otherwise autoadvance
            # refuses it ("not ACTIVE") and the duplicate-create guard still
            # treats it as terminal, contradicting the "archive is reversible"
            # contract.
            if str(proj.get("status", "")).upper() == "ARCHIVED":
                store.update_project(project_id, status="ACTIVE")
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
                constraints=constraints,
                depends_on=depends_on,
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
            # Explicit user constraints active for this project: the
            # project-level set (captured at create/correction time) plus
            # whatever each task carries. A DONE transition with active
            # constraints demands result evidence that ADDRESSES them —
            # "written" is not "honors what the user forbade".
            proj_constraints: List[str] = []
            try:
                _proj = store.get_project(project_id) or {}
                proj_constraints = [
                    str(c) for c in
                    ((_proj.get("metadata") or {}).get("constraints") or [])
                ]
            except Exception:
                pass
            updated: List[Dict[str, Any]] = []
            missing: List[str] = []
            gated: List[str] = []
            gated_constraints: List[str] = []
            active_constraints: List[str] = []
            judged_violations: List[str] = []
            constraint_audit = None  # (ok, reason) — one audit per call
            for tid in target_ids:
                if tid not in plan.tree.nodes:
                    missing.append(tid)
                    continue
                # DONE-gate: a visual/runnable-artifact task cannot be marked
                # DONE with NO verification evidence. The model must either
                # pass a `result` (what it actually observed rendered) or go
                # verify first — the browser screenshot now carries an
                # objective RENDER_CHECK it can cite. 'Written' is not 'works'.
                if (st_enum == TaskStatus.DONE
                        and not (result or "").strip()
                        and not (plan.tree.nodes[tid].result_summary or "").strip()
                        and _is_visual_artifact_task(plan.tree.nodes[tid].description)):
                    gated.append(tid)
                    continue
                # Constraint DONE-gate: same evidence requirement when the
                # task or project carries explicit user constraints. The
                # gate is deliberately evidence-based, not judgement-based —
                # judging WHETHER the evidence honors the constraints is the
                # verifier's job (it receives the constraints in context).
                task_constraints = [
                    str(c) for c in
                    (plan.tree.nodes[tid].constraints or [])
                ] + proj_constraints
                if (st_enum == TaskStatus.DONE and task_constraints
                        and not (result or "").strip()
                        and not (plan.tree.nodes[tid].result_summary or "").strip()):
                    gated_constraints.append(tid)
                    for c in task_constraints:
                        if c not in active_constraints:
                            active_constraints.append(c)
                    continue
                # Constraint JUDGMENT gate (2026-07-08): the evidence gate
                # above only checks that result text EXISTS — the chess
                # session showed the model happily supplies evidence prose
                # while the artifact violates the constraint (it built a
                # heuristic engine after restating "Ghost plays directly").
                # One background LLM audit of the project's workspace files
                # per task_update call; refusal carries the evidence. Escape
                # hatch for a user-approved exception: result text starting
                # with "CONSTRAINT-OVERRIDE:". Fails open on infra errors.
                if (st_enum == TaskStatus.DONE and task_constraints
                        and not str(result or "").lstrip().upper()
                                .startswith("CONSTRAINT-OVERRIDE:")):
                    if constraint_audit is None:
                        try:
                            from ..core.build_gates import constraint_gate
                            from ..core.project_advancer import _gather_project_files
                            _files = _gather_project_files(store, project_id)
                            constraint_audit = await constraint_gate(
                                context, task_constraints, _files,
                                is_background=False)
                        except Exception:
                            logger.debug("constraint judgment gate skipped",
                                         exc_info=True)
                            constraint_audit = (True, "")
                    if not constraint_audit[0]:
                        judged_violations.append(tid)
                        continue
                # Register deliverables BEFORE flipping the task to DONE.
                # update_status can roll the whole project to DONE on this
                # same call (when it's the last open task), which fires the
                # cleanup sweep synchronously — so anything registered after
                # would already be gone. Accept both the explicit
                # `deliverables=[...]` list and a legacy `artifact_kind=file`
                # + `payload` pair; dedup is handled by the store.
                if st_enum == TaskStatus.DONE:
                    keep_paths = list(deliverables or [])
                    if artifact_kind == "file" and (payload or "").strip():
                        keep_paths.append(payload)
                    for rel in keep_paths:
                        try:
                            store.register_file_artifact(tid, rel)
                        except Exception:
                            logger.debug("deliverable registration skipped: %s",
                                         rel, exc_info=True)
                    # Durable fact recorded at completion time → design ledger,
                    # so the next turn inherits it instead of re-deriving it.
                    if (ledger or "").strip() and project_id:
                        try:
                            store.append_ledger(project_id, ledger.strip())
                        except Exception:
                            logger.debug("ledger append skipped", exc_info=True)
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
                if constraints is not None:
                    extras["constraints"] = constraints
                if extras:
                    store.update_task(tid, **extras)
                updated.append({k: store.get_task(tid).get(k)
                                for k in ("id", "status", "result_summary")})
            # A research task often saves its brief with a bare file_system
            # write (not action=research), which left the file out of the
            # research index — invisible to the briefing and never re-read
            # (observed live). Closing any task is a natural "I produced
            # something" checkpoint: pick up newly-written research/*.md so the
            # next turn's briefing surfaces it. Best-effort.
            try:
                from ..core.project_research import reconcile_research_dir
                _n = reconcile_research_dir(store, project_id)
                if _n:
                    pretty_log("Project Research",
                               f"Indexed {_n} directly-written research brief(s)",
                               icon=Icons.BRAIN_PLAN)
            except Exception:
                logger.debug("research reconcile skipped", exc_info=True)
            payload: Dict[str, Any] = {"updated": updated,
                                       "count": len(updated)}
            if missing:
                payload["missing"] = missing
            if judged_violations:
                payload["constraint_violations"] = judged_violations
                payload["agent_instruction_violation"] = (
                    f"REFUSED to mark {len(judged_violations)} task(s) DONE: "
                    "an audit of the project files found a violation of the "
                    "user's stated constraints. "
                    + (constraint_audit[1] if constraint_audit else "")
                    + " Fix the deliverable so it actually honors the "
                    "constraint, then re-call task_update. Only if the USER "
                    "has explicitly approved this exception, re-call with a "
                    "`result` beginning with 'CONSTRAINT-OVERRIDE:' and "
                    "their reason."
                )
            if gated_constraints:
                payload["gated_constraints"] = gated_constraints
                payload["constraints"] = active_constraints
                payload["agent_instruction_constraints"] = (
                    f"Held {len(gated_constraints)} task(s) at non-DONE: the "
                    f"user attached EXPLICIT CONSTRAINTS and you provided no "
                    f"result evidence. Re-read the deliverable, confirm each "
                    f"constraint holds, then re-call task_update with a "
                    f"`result` that states — per constraint — HOW the work "
                    f"honors it. Constraints: "
                    + " | ".join(active_constraints)
                    + ". If the deliverable violates one (e.g. it contains a "
                    f"coded stand-in for something the user said YOU must do "
                    f"at runtime), FIX THE DELIVERABLE FIRST — do not word "
                    f"the result around the violation."
                )
            if gated:
                payload["gated_unverified"] = gated
                payload["agent_instruction"] = (
                    f"Held {len(gated)} task(s) at non-DONE pending verification "
                    f"evidence: {gated}. Two ways to clear the gate:\n"
                    f"  • If the task BUILDS A VISUAL/RUNNABLE artifact (a game, "
                    f"page, canvas, chart): VERIFY it first — browser "
                    f"operation='screenshot' (click_center=true for a game), "
                    f"check the RENDER_CHECK / PRE_INTERACTION lines — then "
                    f"re-call task_update with a `result` describing what you saw.\n"
                    f"  • If the task is NOT visual (a text / research / markdown "
                    f"/ analysis task — this gate can misfire on wording): just "
                    f"re-call task_update with a one-line `result` describing what "
                    f"you produced. That counts as evidence and clears the gate "
                    f"immediately. (Pass it as `result=...`.)\n"
                    f"  • Also pass `deliverables=[...]` with the files the user "
                    f"should keep — everything else in the project workspace is "
                    f"deleted when the project finishes."
                )
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
            # Top-level fan-out: drop subtasks that duplicate an existing
            # non-terminal task (the model often calls create-with-subtasks
            # AND decompose, or decomposes twice — observed live: duplicate
            # Core Shell / File Explorer that then failed).
            if not target_id:
                # A cohesive single-file deliverable must NOT be fanned out
                # into per-feature tasks (they'd merge into one file and
                # collide). Refuse the split and steer to a one-turn build.
                _proj = store.get_project(project_id) or {}
                if (_is_cohesive_single_file(_proj.get("title"), _proj.get("goal"))
                        and len([d for d in subtasks if d and d.strip()]) > 1):
                    return _ok({
                        "created": [],
                        "refused": True,
                        "agent_instruction": (
                            "This is a SINGLE-FILE deliverable — do NOT split it "
                            "into per-feature tasks (they would have to merge "
                            "into one file and collide, breaking the page). "
                            "Build the WHOLE file yourself in ONE turn instead "
                            "(write it directly with file_system); keep it as a "
                            "single task."),
                    })
                subtasks = _filter_duplicate_subtasks(store, project_id, subtasks)
                if not subtasks:
                    return _ok({"created": [], "parent_id": None,
                                "note": "all subtasks duplicated existing "
                                        "tasks; nothing added"})

            ids: List[str] = []
            pairs: List[tuple] = []  # (task_id, description) for graph linking
            try:
                if target_id:
                    ids = plan.decompose(target_id, subtasks,
                                         sequential=sequential)
                    # decompose drops blank descriptions, so the non-blank
                    # (stripped) subtasks align 1:1 with the returned ids.
                    # zip-ing against the RAW subtasks shifted the pairing
                    # whenever a blank was dropped → wrong graph descriptions.
                    non_blank = [d.strip() for d in subtasks if d and d.strip()]
                    pairs = list(zip(ids, non_blank))
                else:
                    # Top-level fan-out. When `sequential`, chain each new
                    # task to the previous one via depends_on so the
                    # autoadvancer runs them strictly in order.
                    prev_id = None
                    for desc in subtasks:
                        if not desc or not desc.strip():
                            continue
                        deps = [prev_id] if (sequential and prev_id) else None
                        tid = plan.add_task(desc.strip(), depends_on=deps)
                        ids.append(tid)
                        pairs.append((tid, desc.strip()))
                        prev_id = tid
            except ValueError as e:
                return _err(str(e))
            # Stamp explicit constraints onto every subtask created in this
            # call — both the ones passed by the model and the ones the
            # triggering message carries. plan.decompose doesn't thread
            # them, so persist post-hoc (same store row, one UPDATE each).
            decompose_constraints = list(constraints or [])
            for clause in extract_constraints(
                    getattr(context, "last_user_content", "") or ""):
                if clause.lower() not in {c.lower()
                                          for c in decompose_constraints}:
                    decompose_constraints.append(clause)
            if decompose_constraints:
                for tid in ids:
                    try:
                        store.update_task(tid, constraints=decompose_constraints)
                    except Exception:
                        logger.debug("constraint stamp skipped for %s", tid,
                                     exc_info=True)
            for tid, desc in pairs:
                _link_task_in_graph(context, project_id, tid, desc)
            return _ok({"created": ids,
                        "parent_id": target_id,
                        "sequential": bool(sequential),
                        "constraints": decompose_constraints,
                        "agent_instruction": (
                            f"Plan created with {len(ids)} task(s). STOP here: "
                            "present this task list to the user and wait for "
                            "direction. DO NOT start executing — advance ONE "
                            "task per explicit go-ahead (e.g. 'proceed to next "
                            "task'), never the whole tree in one turn."
                        )})

        if act == "task_next":
            if not project_id:
                return _no_active_project(store)
            plan = ProjectPlan(store, project_id)
            nxt = plan.next_ready_leaf()
            if not nxt:
                return _ok({"next": None, "reason": "no READY/PENDING leaf available"})
            return _ok({"next": {"id": nxt.id, "description": nxt.description,
                                 "status": nxt.status.value}})

        if act == "task_list":
            if not project_id:
                return _no_active_project(store)
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

        if act == "ledger":
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            # With `ledger` text → append one durable fact. Without it → read
            # the current ledger back (so the agent can review/refresh it).
            note = (ledger or "").strip()
            if note:
                new = store.append_ledger(project_id, note)
                # The ledger is a prime source of technique/library mentions —
                # re-extract concepts so the cross-project map stays current.
                _link_concepts_in_graph(context, project_id)
                return _ok({"ledger": new, "action_taken": "appended"})
            return _ok({"ledger": store.get_ledger(project_id)})

        if act == "config":
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            # With `config_key` → upsert one setting (empty `config_value`
            # deletes it). Without a key → read the current config map back.
            k = (config_key or "").strip()
            if k:
                cfg = store.set_config_value(project_id, k, config_value or "")
                # Config carries library/version hints — refresh concepts.
                _link_concepts_in_graph(context, project_id)
                return _ok({
                    "config": cfg,
                    "action_taken": "deleted" if not (config_value or "").strip() else "set",
                })
            return _ok({"config": store.get_config(project_id)})

        # ---- self-advancing loop ---------------------------------------

        if act == "autoadvance":
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            from ..core.project_advancer import (
                advance_many, default_llm_classifier, default_code_generator,
                pinned_project_context,
            )
            from ..core.coding_executor import build_coding_task

            tool_runner = None
            tools_map = None
            try:
                from .registry import get_available_tools
                # Pin the tool runner to the TARGET project: the process-
                # global current_project_id can be cleared mid-batch by a
                # concurrent conversation's reconcile (and need not equal
                # the project being advanced), which would silently land
                # this batch's file writes at the sandbox root.
                tools_map = get_available_tools(
                    pinned_project_context(context, project_id))
            except Exception:
                tools_map = None
            if tools_map:
                async def _run(name: str, args: Dict[str, Any]) -> str:
                    handler = tools_map.get(name)
                    if not handler:
                        return f"ERROR: tool {name} unavailable"
                    return await handler(**args)
                tool_runner = _run

            # count → max_tasks: int, or None for "all" (run to completion).
            # A bounded loop of advance_once ticks, checkpointing each task —
            # the autonomous BATCH path. A single "proceed" the agent does
            # itself as a full turn; this is for "do the next N" / "all".
            max_tasks = _parse_advance_count(count)
            batch = await advance_many(
                context, project_id,
                max_tasks=max_tasks,
                tool_runner=tool_runner,
                llm_classifier=default_llm_classifier(context),
                code_generator=default_code_generator(context),
                coding_executor=build_coding_task,
                # Continue past an isolated failed task (apps are usually
                # independent) and report all failures — a single flaky task
                # shouldn't halt the whole batch. A circuit breaker still
                # stops on repeated (systemic) failures.
                stop_on_fail=False,
            )
            return _ok({
                "advanced": batch.advanced,
                "count": batch.count,
                "requested": ("all" if max_tasks is None else max_tasks),
                "stop_reason": batch.stop_reason,
                "agent_instruction": _advance_batch_instruction(batch, max_tasks),
            })

        # ---- auto-research ----------------------------------------------

        if act == "research":
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            from ..core.project_research import (
                research_topic, research_project,
            )
            if topic and topic.strip():
                rr = await research_topic(context, project_id, topic.strip())
                if not rr.ok:
                    return _err(f"research failed: {rr.error}")
                return _ok({
                    "researched": rr.topic, "path": rr.path,
                    "sources": len(rr.sources),
                    "summary_preview": (rr.summary or "")[:280],
                    "agent_instruction": (
                        f"Findings saved to {rr.path} in the project workspace. "
                        "Read that file with file_system when you need the "
                        "detail; it is also listed under RESEARCH NOTES in the "
                        "project briefing from now on."
                    ),
                })
            # No explicit topic → auto-derive several from goal + tasks.
            results = await research_project(
                context, project_id, topics=topics, max_topics=max_topics)
            ok_results = [r for r in results if r.ok]
            return _ok({
                "researched": [
                    {"topic": r.topic, "path": r.path, "sources": len(r.sources)}
                    for r in ok_results
                ],
                "count": len(ok_results),
                "agent_instruction": (
                    "Auto-derived and researched the topics above; each brief is "
                    "saved in research/ in the project workspace and surfaced "
                    "under RESEARCH NOTES in the briefing. Read a file when you "
                    "need its detail."
                ) if ok_results else "No topics could be derived or researched.",
            })

        if act == "research_list":
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            from ..core.project_research import get_research_index
            return _ok({"research": get_research_index(store, project_id)})

        # ---- inbox promotion (suggestion-accepted path) -----------------

        if act == "promote_from_context":
            if not title:
                return _err("title is required for promotion")
            pid = store.create_project(
                title=title, kind=_infer_kind(title, goal, kind), goal=goal,
                metadata={"promoted_from_context": True},
            )
            _link_project_in_graph(context, pid, title)
            _link_concepts_in_graph(context, pid)
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
            "switch projects. GATING for `create`: only create a project when "
            "(1) the user EXPLICITLY asks to start/track a project, OR (2) the "
            "deliverable GENUINELY spans MULTIPLE files/modules AND needs "
            "MULTIPLE turns to build. Do NOT use this for ad-hoc chat or "
            "one-shot queries — those stay in free-chat mode. A self-contained "
            "SINGLE-FILE deliverable (even a large one — a one-file browser OS, "
            "game, or script) is a one-shot: build it directly with file_system "
            "write, do NOT create a project for it. Past similar projects in "
            "memory are NOT a reason to create a new one. Do NOT call `create` "
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
            "described work is actually complete. PACING: after you "
            "create or decompose a plan, present it and STOP — advance "
            "ONE task per explicit user go-ahead ('proceed to next "
            "task'), never the whole tree in a single turn (that floods "
            "the context window on large projects). `resume` when "
            "the user asks to pick up "
            "an old project. GRANULARITY when decomposing: make each task "
            "own a FILE or bounded function you can build+verify alone; do "
            "NOT split one file into N tasks. MEMORY: `ledger` records a "
            "durable fact (file layout, key API/function name, convention) "
            "that is surfaced in the briefing every turn — use it (or pass "
            "`ledger=…` on a DONE task_update) so later turns inherit what "
            "you learned instead of re-reading files. `exit` to leave "
            "project mode; "
            "`archive` to HIDE a project (reversible — status→ARCHIVED, "
            "files kept, `resume` brings it back); `delete` to "
            "PERMANENTLY remove a project and ALL its data — tasks, "
            "artifacts, events, AND its workspace files on disk (NOT "
            "reversible). Use `delete` only when the user clearly means "
            "to erase it; otherwise prefer `archive`. "
            "`promote_from_context` only when the user has explicitly "
            "accepted a suggestion to convert the current chat into a "
            "project. `research` to web-research a topic (pass `topic`) or "
            "auto-derive several from the project goal (omit `topic`) — each "
            "is summarized into research/<slug>.md in the project workspace "
            "and listed under RESEARCH NOTES in the briefing; `research_list` "
            "to see what has already been researched (read a brief with "
            "file_system before re-researching the same thing)."
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
                "constraints": {"type": "array", "items": {"type": "string"},
                                "description": "EXPLICIT user requirements/prohibitions quoted from their message, verbatim or near-verbatim (e.g. \"don't come up with some random AI\", \"YOU will play against me\", \"no external libraries\"). Pass on create/task_add/task_decompose/task_update. They are re-shown every turn in the briefing and GATE task completion: a task with constraints cannot go DONE without result evidence addressing them. Negations and role assignments (what the user said YOU must do) belong here, never paraphrased away."},
                "depends_on": {"type": "array", "items": {"type": "string"},
                               "description": "task_add: ids of sibling tasks that must be DONE before THIS task becomes eligible to run (a prerequisite edge, distinct from parent/child). Use it to order peer tasks; omit for independent tasks."},
                "sequential": {"type": "boolean",
                               "description": "task_decompose: when true, chain the subtasks so each one only runs after the previous is DONE (the autoadvancer executes them in order). Default false = all subtasks independently runnable."},
                "topic": {"type": "string",
                          "description": "action=research: a single topic to research now. The agent web-searches it, summarizes the findings into research/<slug>.md in the project workspace, and surfaces it in the project briefing. Omit topic (and topics) to auto-derive several topics from the project goal + open tasks."},
                "topics": {"type": "array", "items": {"type": "string"},
                           "description": "action=research: an explicit list of topics to research (each persisted as its own brief). If omitted and no single topic is given, topics are auto-derived from the project."},
                "max_topics": {"type": "integer",
                               "description": "action=research: cap on how many auto-derived topics to research (default 5)."},
                "result": {"type": "string", "description": "Short result summary (task_update with status=DONE)."},
                "failure_reason": {"type": "string"},
                "subtasks": {"type": "array", "items": {"type": "string"},
                             "description": "Ordered subtask descriptions for task_decompose / promote_from_context. GRANULARITY: make each task own a FILE or a clearly-bounded function/module you can build AND verify on its own (e.g. 'src/parser.py: parse the CSV', 'apps/terminal.js: terminal app'). AVOID splitting one file into N tasks (e.g. 6 tasks that all edit index.html) — that forces re-reading the whole file every turn and does not scale. Prefer a thin shell/entrypoint + one file per feature."},
                "artifact_kind": {"type": "string",
                                  "enum": ["file", "url", "note", "tool_call"]},
                "payload": {"type": "string"},
                "deliverables": {"type": "array", "items": {"type": "string"},
                                 "description": "task_update with status=DONE: the project-relative paths of files the USER should keep (the actual deliverables — reports, code, generated images). Everything else in the project workspace (screenshots, helper scripts, temp files) is DELETED when the project finishes, so list every file worth keeping here. Paths are relative to the project workspace, e.g. [\"report.pdf\", \"src/solver.py\"]."},
                "status_filter": {"type": "string"},
                "limit": {"type": "integer"},
                "event_type": {"type": "string"},
                "context_summary": {"type": "string",
                                    "description": "Optional textual snapshot captured at promotion time."},
                "count": {"type": "string",
                          "description": "action=autoadvance: how many tasks to advance autonomously in a bounded loop — a number (e.g. \"3\") or \"all\" to run to completion. Use this ONLY for an explicit MULTI-task request: 'do the next 3 tasks' → count=\"3\"; 'proceed with all remaining tasks' / 'finish the project' → count=\"all\". A single 'proceed'/'next' you do YOURSELF as one focused full turn — do NOT route that here (autoadvance runs a lighter single-step-per-task executor). The loop checkpoints each task and stops at the first of: done · a task that needs you · budget · a FAILED task."},
                "ledger": {"type": "string",
                           "description": "action=ledger: ONE durable fact to append to the project's design ledger — file layout, a key function/API name, a convention, where something lives (e.g. 'windows are .window divs, opened via openApp(id), drag via makeDraggable'). The ledger is surfaced in the project briefing every turn, so the next turn inherits these facts instead of re-reading files to rediscover them. Omit `ledger` to read the current ledger back. May also be passed on a task_update status=DONE to record the decision as the task closes."},
                "config_key": {"type": "string",
                               "description": "action=config: the name of ONE durable project setting to record — an env var, key flag, dependency version, the model, a port, a DB URI (e.g. 'GHOST_MODEL', 'port', 'torch'). Surfaced in the project briefing every turn so the next turn runs/builds under the right settings instead of re-discovering them from requirements.txt / env / argv. Omit both config_key and config_value to read the current config map back; pass config_key with an empty config_value to delete that setting."},
                "config_value": {"type": "string",
                                 "description": "action=config: the value for `config_key` (e.g. 'qwen-3.6-35b-a3', '8000', '2.3.1'). Empty value deletes the key."},
            },
            "required": ["action"],
        },
    },
}
