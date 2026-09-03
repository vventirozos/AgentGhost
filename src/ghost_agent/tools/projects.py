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

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..memory.projects import (ProjectStore, ProjectKind, ProjectStatus,
                               _constraint_list)
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
    "artifact_add", "artifact_list", "event_log", "ledger", "config",
    # per-file manifest + per-file action history (2026-07-24)
    "describe_file", "file_history", "unregister_file",
    # human-attested release + versioning (2026-07-25)
    "release", "create_version", "unrelease", "verify_release",
    # task removal (round 3 — store.delete_task existed unexposed)
    "task_delete",
    # constraint lifecycle (2026-08-01 — stale-constraint incident)
    "constraint_retire",
    # cross-project features (2026-07-25 final round)
    "search", "set_dependency", "clone",
    # workspace hygiene
    "cleanup",
    # inbox promotion (suggestion-accepted path)
    "promote_from_context",
    # self-advancing loop
    "autoadvance",
    # auto-research
    "research", "research_list",
}


def _err(msg: str):
    """A manage_projects refusal — and it says so.

    Its sibling `_ok` declares; this returned a bare `f"ERROR: {msg}"` that
    coerced to FAILED, so 15 live argument/lookup refusals (one of which
    literally reads "NOTHING was deleted") armed the pre-flight guard against
    a corrected re-issue that keys identically on tool+target+op.
    """
    from .outcome import ToolOutcome
    return ToolOutcome.rejected(f"ERROR: {msg}",
                                reason_code="manage_projects_refused")


# Cap on what the constraint judgment gate reads per task (it is one blocking
# LLM call on the user's turn). The gate itself re-truncates; this just keeps
# the read cheap.
_TASK_FILE_MAX_CHARS = 20000


def _files_for_task(store, project_id: str, task_id: str,
                    deliverables=None) -> Dict[str, str]:
    """``{rel_path: content}`` for the files THIS task produced — the only
    correct input to the constraint judgment gate.

    Sources, in order: the ``deliverables=[...]`` passed to this very
    ``task_update`` call, then the file artifacts already registered against
    the task. Returns ``{}`` when the task has no attributable files, which
    the caller treats as "skip the judgment gate" — auditing files that some
    OTHER task wrote is precisely the deadlock this replaced (see the call
    site). Never raises.
    """
    rels: List[str] = []
    for rel in (deliverables or []):
        rel = str(rel or "").strip().replace("\\", "/").lstrip("./")
        if rel and rel not in rels:
            rels.append(rel)
    try:
        for art in store.list_artifacts(task_id=task_id) or []:
            if (art.get("kind") or "") != "file":
                continue
            rel = str(art.get("payload") or "").strip().replace("\\", "/")
            rel = rel.lstrip("./")
            if rel and rel not in rels:
                rels.append(rel)
    except Exception:
        logger.debug("artifact lookup failed for task %s", task_id,
                     exc_info=True)

    root = getattr(store, "sandbox_root", None)
    pid = str(project_id or "").strip().lower()
    if not root or not pid or not rels:
        return {}
    base = (Path(root) / "projects" / pid).resolve()
    out: Dict[str, str] = {}
    for rel in rels[:8]:
        try:
            p = (base / rel).resolve()
            # Containment: a registered path is model-supplied text.
            if not str(p).startswith(str(base) + os.sep):
                continue
            if not p.is_file():
                continue
            out[rel] = p.read_text(encoding="utf-8",
                                   errors="replace")[:_TASK_FILE_MAX_CHARS]
        except Exception:
            continue
    return out


# A recorded task failure that a LIVE mechanical check can re-test. When a
# FAILED task's failure_reason carries one of these markers, a DONE attempt
# re-runs the syntax check on the referenced file(s) NOW instead of trusting
# the stored verdict: the 2026-08-01 Mini AI incident wedged on exactly this —
# the agent rewrote core.py to a clean state (py_compile exit 0, ran, learned
# XOR) but the task kept its autoadvance-era "does NOT parse" evidence and
# the model burned the rest of the request arguing with a stale ledger.
_PARSE_FAILURE_MARKERS = ("does not parse", "syntax check failed",
                          "syntaxerror", "py_compile")
# No space in the path class: with one, the leftmost match bridged whole
# phrases ("py_compile failed for core.py" matched as ONE candidate that
# is_file() rejects, consuming the real filename) and the recheck went
# silently inert on realistic failure texts. Space-bearing filenames just
# don't match — fail-safe (recheck inapplicable), never wrongly-closing.
_PARSE_RECHECK_PATH_RE = re.compile(
    r"(?<![\w./-])([\w][\w./-]{0,80}?\.(?:py|js|mjs|cjs|json|html?|htm))\b",
    re.IGNORECASE)


def _parse_failure_candidates(store, project_id: str, task_id: str,
                              failure_reason: str,
                              deliverables=None) -> List[Path]:
    """Absolute, containment-checked paths of the checkable files a
    parse-shaped failure_reason refers to. Sources: paths named in the
    failure text itself, the task's registered file artifacts, and the
    deliverables passed to this call. Only files that exist are returned."""
    rels: List[str] = []
    for m in _PARSE_RECHECK_PATH_RE.finditer(failure_reason or ""):
        rel = m.group(1).strip().replace("\\", "/").lstrip("./")
        if rel and rel not in rels:
            rels.append(rel)
    for rel in (deliverables or []):
        rel = str(rel or "").strip().replace("\\", "/").lstrip("./")
        if rel and rel not in rels:
            rels.append(rel)
    try:
        for art in store.list_artifacts(task_id=task_id) or []:
            if (art.get("kind") or "") != "file":
                continue
            rel = str(art.get("payload") or "").strip().replace("\\", "/")
            rel = rel.lstrip("./")
            if rel and rel not in rels:
                rels.append(rel)
    except Exception:
        logger.debug("artifact lookup failed for %s", task_id, exc_info=True)
    root = getattr(store, "sandbox_root", None)
    pid = str(project_id or "").strip().lower()
    if not root or not pid:
        return []
    base = (Path(root) / "projects" / pid).resolve()
    out: List[Path] = []
    for rel in rels[:8]:
        try:
            p = (base / rel).resolve()
            if not str(p).startswith(str(base) + os.sep):
                continue
            if p.is_file():
                out.append(p)
        except Exception:
            continue
    return out


async def _recheck_stale_parse_failure(store, project_id: str, task_id: str,
                                       failure_reason: str,
                                       deliverables=None):
    """Re-test a parse-shaped task failure against the CURRENT files.

    Returns ``(verdict, detail)``:
      * ``(None, "")``  — not applicable (failure isn't parse-shaped, or no
        referenced file exists to check); callers change nothing.
      * ``(True, evidence)`` — every referenced file NOW passes the same
        syntax check that failed it; ``evidence`` is usable result text and
        the stored failure evidence is stale.
      * ``(False, diagnostic)`` — a file STILL fails; ``diagnostic`` names
        the current error (never the stale one).
    Never raises.
    """
    low = (failure_reason or "").lower()
    if not any(marker in low for marker in _PARSE_FAILURE_MARKERS):
        return None, ""
    try:
        paths = _parse_failure_candidates(store, project_id, task_id,
                                          failure_reason, deliverables)
    except Exception:
        logger.debug("parse-recheck candidate scan failed", exc_info=True)
        return None, ""
    if not paths:
        return None, ""
    from .file_system import _syntax_feedback, _find_node
    # Count a file as "checked" ONLY when the checker can actually verify
    # its type: _syntax_feedback returns "" for clean AND for uncheckable
    # (unknown extension, node binary absent, read error) — treating those
    # as passes would MANUFACTURE pass evidence and force-close a still-
    # broken task (review finding, 2026-08-01). .py/.json verify
    # in-process; .js/.html need a node binary.
    checkable = {".py", ".json"}
    if _find_node():
        checkable |= {".js", ".mjs", ".cjs", ".html", ".htm"}
    checked: List[str] = []
    for p in paths:
        if p.suffix.lower() not in checkable:
            continue
        try:
            diag = await _syntax_feedback(p, p.name)
        except Exception:
            logger.debug("live syntax re-check failed for %s", p,
                         exc_info=True)
            continue
        if diag:
            return False, (f"{p.name} STILL fails the live syntax check: "
                           + " ".join(diag.split())[:300])
        checked.append(p.name)
    if not checked:
        return None, ""
    return True, (f"live syntax re-check: {', '.join(checked)} parse(s) "
                  f"cleanly now — the recorded parse-failure evidence was "
                  f"stale and has been superseded")


def _ok(payload: Any) -> "ToolOutcome":
    """A manage_projects SUCCESS — and it says so.

    The text is unchanged; the status is what is new. These payloads embed
    the project ledger, and a stored `autoadvance_failed` event quotes the
    failing tool's `--- EXECUTION RESULT --- EXIT CODE: 1` verbatim. The
    dispatch loop's banner rule found that quote and booked the READ as a
    failed shell command: 4 live rows, all successes, reported to the model
    under the AUTHORITATIVE multi-step banner and recorded as metacog
    incompetence. Position cannot tell a quoted banner from an envelope
    (JSON is one line), so the producer says it instead.
    """
    from .outcome import ToolOutcome
    # ⚠ A payload that reports its OWN refusal is not a success. `task_update`
    # returns `updated: []` with `gated_constraints` when a constraint holds
    # every task at non-DONE — nothing landed, and a declared ok silences the
    # sniffer that would otherwise have caught it.
    if isinstance(payload, dict):
        _held = payload.get("gated_constraints") or payload.get(
            "agent_instruction_constraints")
        _nothing = (payload.get("updated") == []
                    or payload.get("count") == 0)
        if _held and _nothing:
            import json as _j
            return ToolOutcome.rejected(
                _j.dumps(payload, default=str),
                reason_code="task_update_held_by_constraint")
    if isinstance(payload, str):
        return ToolOutcome.ok(payload)
    try:
        return ToolOutcome.ok(json.dumps(payload, default=str))
    except Exception:
        return ToolOutcome.ok(str(payload))


async def _not_found_with_recall(context, term) -> str:
    """A `get` miss that tries memory before giving up (§4C routing fix).

    Returns a non-striking `_ok` payload when recall finds related facts —
    the model should answer FROM them, not retry project lookups; falls
    back to a plain `_err` with a `recall` redirect when memory is empty
    or unavailable."""
    base = f"No tracked project matches {str(term)!r}."
    vm = getattr(context, "memory_system", None)
    search = getattr(vm, "search_advanced", None) if vm is not None else None
    hits = []
    if callable(search):
        try:
            # record_retrievals=False: this is a lookup-miss probe, not a
            # real retrieval — the default bumped prune-survival stats on
            # 3 arbitrary nearby rows (including hits discarded by the
            # 1.15 gate below) on every project-name miss.
            raw = await asyncio.to_thread(
                search, str(term), 3, record_retrievals=False)
            for h in raw or []:
                if not isinstance(h, dict) or not h.get("text"):
                    continue
                try:
                    score = float(h.get("score", 9.0))
                except (TypeError, ValueError):
                    continue
                if score < 1.15:  # tool_recall's MEDIUM-or-better band
                    hits.append(str(h["text"])[:200])
        except Exception:
            hits = []
    if hits:
        return _ok({
            "project": None,
            "note": base + (" However, MEMORY RECALL found related facts — "
                            "answer from these instead of giving up:"),
            "memory_recall": hits[:3],
        })
    return _err(base + " If this is a question about a topic rather than a "
                       "tracked project, use the `recall` tool.")


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

# Marker key written by tools/swarm.py at dispatch: `_swarm_task_id::<output_key>`
# → task id. Its existence means a detached worker still owns `<output_key>`
# and will write to it minutes later, in whatever conversation/project happens
# to be live at that moment. Both keys therefore belong to the JOB, not to the
# project that was active when it was dispatched — see
# `_background_job_keys`.
_SWARM_TASK_PREFIX = "_swarm_task_id::"


def _scratchpad_keys(scratchpad) -> List[str]:
    """Best-effort list of the scratchpad's live keys.

    Prefers the public `keys()` accessor and falls back to the internal
    `_data` mapping; returns [] for anything that supports neither (e.g.
    MagicMock-based test contexts).
    """
    if scratchpad is None:
        return []
    getter = getattr(scratchpad, "keys", None)
    if callable(getter):
        try:
            keys = list(getter())
            if all(isinstance(k, str) for k in keys):
                return keys
        except Exception:
            pass
    data = getattr(scratchpad, "_data", None)
    try:
        return [k for k in list(data.keys()) if isinstance(k, str)]
    except Exception:
        return []


def _background_job_keys(context, scratchpad) -> set:
    """Scratchpad keys owned by background jobs — NEVER ours to delete.

    A swarm task dispatched from conversation A writes its result long after
    the dispatching turn ended. Until 2026-07-22 the project layer deleted
    every non-sentinel key at any FOREIGN conversation's request start, so
    conversation B saying "hi" durably destroyed A's in-flight results: the
    worker then re-created the key in whatever scope was live, and
    `jobs(action='collect')`'s resolver (a bare `scratchpad.get(output_key)`)
    returned None — an empty job result with no error. The live prod DB was
    down to its 2 sentinel rows.

    Two independent sources, unioned (either alone is enough):
      * the `_swarm_task_id::<output_key>` markers in the scratchpad itself;
      * `meta["output_key"]` of every job the registry still retains
        (running, or finished but not yet collected).
    """
    protected: set = set()
    for k in _scratchpad_keys(scratchpad):
        if k.startswith(_SWARM_TASK_PREFIX):
            protected.add(k)
            target = k[len(_SWARM_TASK_PREFIX):]
            if target:
                protected.add(target)
    # Only consult a registry that already exists — never create one here.
    registry = getattr(context, "job_registry", None)
    lister = getattr(registry, "list", None)
    if callable(lister):
        try:
            for job in lister():
                if getattr(job, "collected", False):
                    continue
                out = (getattr(job, "meta", None) or {}).get("output_key")
                if isinstance(out, str) and out:
                    protected.add(out)
                    protected.add(f"{_SWARM_TASK_PREFIX}{out}")
        except Exception:
            pass
    return protected


def _protected_scratchpad_keys(context, scratchpad) -> set:
    """Every key a project switch must leave alone: the activation sentinels,
    the `proj::`-namespaced convention keys, and background-job keys."""
    protected = {_CURRENT_SENTINEL, _CURRENT_CONV_SENTINEL}
    protected.update(
        k for k in _scratchpad_keys(scratchpad)
        if k.startswith(_PROJECT_KEY_PREFIX)
    )
    protected.update(_background_job_keys(context, scratchpad))
    return protected


def _sp_set(scratchpad, key: str, value, namespace=None):
    """`scratchpad.set` with an explicit scope, tolerating scratchpads that
    predate the namespace argument (mocks, alternative implementations)."""
    try:
        return scratchpad.set(key, value, namespace=namespace)
    except TypeError:
        return scratchpad.set(key, value)


def _set_scratchpad_scope(scratchpad, project_id: Optional[str]):
    """Point the scratchpad's default write scope at the active project (or
    the global scope in free chat), so keys written this turn are tagged with
    the project that owns them and can later be cleared on their own."""
    if scratchpad is None:
        return
    try:
        scratchpad.active_namespace = project_id or None
    except Exception:
        pass


def _iter_scratchpad_items(scratchpad, project_id: Optional[str] = None,
                           protected: Optional[set] = None) -> List[tuple]:
    """Best-effort snapshot of the scratchpad keys owned by ``project_id``.

    Scope-aware scratchpads report their own membership; older/mock ones fall
    back to the historical "everything that isn't a sentinel or `proj::`"
    rule. Background-job keys are excluded either way — they are process-wide,
    so snapshotting them would file another conversation's swarm result into
    THIS project's event log (and re-hydrate a stale copy later).
    """
    if scratchpad is None:
        return []
    data = getattr(scratchpad, "_data", None)
    if data is None:
        return []
    spared = protected or set()
    scoped = None
    lister = getattr(scratchpad, "keys_in_namespace", None)
    if project_id and callable(lister):
        try:
            candidates = list(lister(project_id))
            if all(isinstance(k, str) for k in candidates):
                scoped = candidates
        except Exception:
            scoped = None
    try:
        if scoped is not None:
            return [(k, data[k]) for k in scoped
                    if k in data and k not in spared
                    and not k.startswith(_PROJECT_KEY_PREFIX)
                    and k not in (_CURRENT_SENTINEL, _CURRENT_CONV_SENTINEL)]
        return [
            (k, v) for k, v in data.items()
            if not str(k).startswith(_PROJECT_KEY_PREFIX)
            and k not in (_CURRENT_SENTINEL, _CURRENT_CONV_SENTINEL)
            and k not in spared
        ]
    except Exception:
        return []


def _snapshot_scratchpad(context, project_id: Optional[str]) -> bool:
    """Capture in-flight scratchpad keys into the project's event log.

    Returns False only when there WAS project state to save and saving it
    failed — callers use that to avoid clearing keys they can no longer
    restore.
    """
    if not project_id:
        return True
    sp = getattr(context, "scratchpad", None)
    items = _iter_scratchpad_items(
        sp, project_id, _protected_scratchpad_keys(context, sp))
    if not items:
        return True
    store = _resolve_store(context)
    if store is None:
        return False
    try:
        store.log_event(project_id, None, "scratchpad_snapshot",
                        {"keys": dict(items)})
        return True
    except Exception:
        logger.debug("scratchpad snapshot failed", exc_info=True)
        return False


def _clear_other_project_scopes(context, sp, project_id: Optional[str]):
    """Drop the scratchpad keys owned by OTHER projects, and nothing else.

    Previously this deleted every key that wasn't `proj::`-prefixed or a
    sentinel — including keys belonging to no project at all. Because
    `_park_current_project` runs it at REQUEST START for any conversation
    that doesn't own the bound project, one unrelated chat message wiped the
    whole scratchpad (verified in prod). Now each entry carries the scope
    that wrote it, so we clear scope-by-scope and spare:

      * the incoming project's own scope (about to be re-hydrated),
      * the global/free-chat scope (owned by nobody — includes results
        written by detached workers),
      * sentinels, `proj::` keys and in-flight background-job keys
        (`_protected_scratchpad_keys`), even when a stale scope tag says
        otherwise.
    """
    protected = _protected_scratchpad_keys(context, sp)
    target = project_id or None
    namespaces = getattr(sp, "namespaces", None)
    clear_ns = getattr(sp, "clear_namespace", None)
    if callable(namespaces) and callable(clear_ns):
        try:
            names = [n for n in list(namespaces()) if n and n != target]
        except Exception:
            names = None
        if names is not None:
            for ns in names:
                try:
                    clear_ns(ns, protect=protected)
                except Exception:
                    pass
            return
    # Scratchpad without scope support: keep the historical behaviour but
    # never touch a protected key — an unowned background-job result must
    # survive here too (that is the data-loss bug).
    data = getattr(sp, "_data", None)
    if data is None:
        return
    try:
        victims = [
            k for k in list(data.keys())
            if not str(k).startswith(_PROJECT_KEY_PREFIX)
            and k not in (_CURRENT_SENTINEL, _CURRENT_CONV_SENTINEL)
            and k not in protected
        ]
    except Exception:
        return
    for k in victims:
        try:
            sp.delete(k)
        except Exception:
            pass


def _hydrate_scratchpad(context, project_id: Optional[str]):
    """Restore the most recent scratchpad_snapshot for a project into
    the live scratchpad.

    We clear the OTHER projects' scoped keys first so a switch doesn't leak
    state from the previous project, then restore the target's saved keys
    into the target's scope. Sentinel keys (`__current_project__`,
    project-namespaced `proj::…`) and background-job keys are preserved.
    """
    sp = getattr(context, "scratchpad", None)
    if sp is None:
        return
    _clear_other_project_scopes(context, sp, project_id)
    # From here on, this turn's writes belong to the incoming project.
    _set_scratchpad_scope(sp, project_id)
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
    # A legacy snapshot (taken before job keys were excluded) can carry a
    # stale copy of a background-job key — restoring it would overwrite the
    # live result the worker wrote in the meantime.
    live_jobs = _background_job_keys(context, sp)
    for k, v in keys.items():
        if k in live_jobs:
            continue
        try:
            _sp_set(sp, k, v, namespace=project_id)
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
                # ⚠ record WHAT this conversation is closing, HERE — while
                # the ownership sentinel is still readable, one line before
                # it is deleted. This is the only point every genuine close
                # route (exit, archive, hard delete) passes through; the
                # first version stamped in `_park_current_project`, whose
                # only callers are the request-start HYGIENE branches, where
                # ownership is false by construction — the stamp was
                # unreachable from every production path and the same-turn-
                # close false-refute class it closed was silently back
                # (round-14 convergence lens: "the fix severs what it
                # feeds"). The verifier's scope heal reads
                # (conversation_key, pid); only an owned binding stamps.
                try:
                    if prev and _conversation_bound_project_pid(
                            context) == prev:
                        context.last_closed_project = (
                            getattr(context, "conversation_key", None), prev)
                except Exception:
                    pass
                sp.delete(_CURRENT_SENTINEL)
                sp.delete(_CURRENT_CONV_SENTINEL)
            else:
                # namespace=None: the sentinels belong to no project scope,
                # so a later scope clear can never take them with it (an
                # evicted sentinel is what `reconcile_conversation`'s
                # fail-closed branch treats as "no binding").
                _sp_set(sp, _CURRENT_SENTINEL, project_id, namespace=None)
                # Bind the activation to the conversation that asked for it
                # (set per-request by `reconcile_conversation`). Empty/None
                # means "unbound" — reconcile treats that as not owned by
                # ANY conversation, so a stale activation can never leak.
                _sp_set(sp, _CURRENT_CONV_SENTINEL,
                        getattr(context, "conversation_key", None) or "",
                        namespace=None)
        except Exception:
            pass
        # Keep the default write scope in step even when the project is
        # unchanged (idempotent switch takes neither branch below).
        _set_scratchpad_scope(sp, project_id)
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


def _conversation_bound_project_pid(context) -> str:
    """The pid this conversation's sentinels bind, or "". Thin wrapper so
    the park-stamp ownership gate and the verifier heal share one notion."""
    try:
        from .file_system import _conversation_bound_project
        return _conversation_bound_project(context) or ""
    except Exception:
        return ""


def _park_current_project(context, cur: str, why: str):
    """Deactivate the process-global active project for THIS request,
    snapshotting its scratchpad so the owning conversation can resume it."""
    saved = _snapshot_scratchpad(context, cur)
    # (No close-stamp here, deliberately. This function's only callers are
    # the request-start HYGIENE branches, where the conversation owns the
    # parked project in NEITHER case — a stamp here is either wrong (the
    # round-13 finding: healing to a project this conversation never
    # touched) or unreachable (the round-14 finding: an ownership gate that
    # its own call sites falsify by construction). The live stamp is in
    # `_set_current(context, None)`, the one point every genuine close
    # passes through.)
    context.current_project_id = None
    _wm = getattr(context, "workspace_model", None)
    if _wm is not None:
        try:
            _wm.current_project_id = ""
        except Exception:
            pass
    if saved:
        _hydrate_scratchpad(context, None)
    else:
        # The snapshot that makes this reversible did not land. Clearing the
        # project's keys now would destroy state nothing can restore, and
        # this runs at REQUEST START for conversations that merely happen to
        # share the process — leaking is recoverable, deleting is not. Leave
        # the entries tagged with `cur` (the next successful switch clears
        # them) and only reset the write scope.
        _set_scratchpad_scope(getattr(context, "scratchpad", None), None)
        logger.warning(
            f"Project '{cur}' parked WITHOUT a scratchpad snapshot — its keys "
            f"are kept in memory rather than dropped (nothing could restore them)")
    pretty_log("Project Scope", why, icon=Icons.BRAIN_PLAN)


def reconcile_conversation(context, conv_key: str):
    """Scope the active project to the conversation that activated it.

    Thin wrapper around :func:`_reconcile_conversation` that leaves the
    scratchpad's default write scope matching whatever project this request
    ended up with — every branch below returns from a different place, and a
    stale scope would tag this turn's keys with the PREVIOUS request's
    project (so the next switch clears the wrong set).
    """
    try:
        _reconcile_conversation(context, conv_key)
    finally:
        _set_scratchpad_scope(getattr(context, "scratchpad", None),
                              getattr(context, "current_project_id", None))


def _reconcile_conversation(context, conv_key: str):
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
        # No valid binding — never set, cleared, or LRU-evicted (the
        # scratchpad keeps ~50 entries / 24h). Returning here used to fail
        # OPEN: the process-global `current_project_id` from the PREVIOUS
        # conversation stayed active for this request, so its file writes
        # landed in that project's workspace. Fail CLOSED instead: park the
        # stale project unless this very message names it (same escape
        # hatch as the fingerprint-mismatch path below).
        if cur and not _message_references_project(context, cur):
            _park_current_project(
                context, cur,
                f"Project '{cur}' has no conversation binding — "
                "deactivated for this request")
        elif cur:
            # Explicitly referenced → keep it active and re-bind it to this
            # conversation so sentinel-based scoping fallbacks resolve it.
            _set_current(context, cur)
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
                # namespace=None — sentinels are scope-free (see _set_current).
                _sp_set(sp, _CURRENT_CONV_SENTINEL, conv_key or "",
                        namespace=None)
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
        _park_current_project(
            context, cur,
            f"Project '{cur}' belongs to another conversation — "
            "deactivated for this request")


def _rearm_inherited_constraints(meta: Dict[str, Any]) -> List[str]:
    """Constraints a fork/clone inherits as ACTIVE: the source's active
    list plus its retired list (case-insensitively deduped, order
    preserved). A DONE/RELEASED source keeps its constraint knowledge in
    ``constraints_retired`` (lifecycle, 2026-08-01) — new work on the same
    deliverable line makes that intent live again."""
    merged: List[str] = []
    seen = set()
    for c in (_constraint_list(meta.get("constraints"))
              + _constraint_list(meta.get("constraints_retired"))):
        if c.lower() not in seen:
            merged.append(c)
            seen.add(c.lower())
    return merged


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


# Generic TYPE prefixes are not feature identities. A decompose shaped
# "Implement: core.py…" / "Implement: train.py…" / "Implement: demo.py…"
# collapsed onto the single key "implement", so the dedup filter silently ate
# every implement-task after the first — observed live 2026-08-01 (project
# 85a350ee099c "Mini AI": 5 planned tasks → 3 persisted, and the next request
# burned turns updating phantom train.py/demo.py task ids that never existed).
# When the colon-head is one of these, identity must come from a longer slice
# of the whole description instead.
_GENERIC_FEATURE_HEADS = frozenset({
    "research", "design", "implement", "implementation", "build", "create",
    "add", "write", "code", "fix", "test", "tests", "verify", "update",
    "refactor", "document", "documentation", "docs", "deploy", "setup",
    "set up", "plan", "review", "analyze", "analyse", "investigate",
    "task", "step", "phase", "todo", "feature", "integrate",
})


def _feature_key(desc: str) -> str:
    """A coarse identity for a task — the feature name before the first colon
    (or the first few words). 'Core Shell: HTML structure…' and 'Core Shell:
    index.html skeleton…' share the key 'core shell', so they dedup. A GENERIC
    head ('Implement: …', 'Research: …') identifies a task TYPE, not a
    feature — those key on a longer slice of the full description so distinct
    deliverables never collapse onto one key."""
    d = " ".join((desc or "").strip().lower().split())
    head = d.split(":", 1)[0].strip()
    if head and head not in _GENERIC_FEATURE_HEADS:
        return head[:40]
    return d[:80]


def _filter_duplicate_subtasks(store, project_id: str, subtasks):
    """Split subtask descriptions into ``(kept, dropped)`` — dropped being the
    ones whose feature already exists as a non-terminal task, plus intra-list
    duplicates. Stops the project piling up duplicate tasks when the model
    calls create-with-subtasks AND task_decompose, or decomposes twice
    (observed live — duplicate Core Shell / File Explorer that then failed).

    Returns BOTH lists so the caller can report what was dropped: silently
    eating subtasks made the model's presented plan diverge from the persisted
    ledger with no signal to anyone (2026-08-01 Mini AI incident)."""
    try:
        existing = store.list_tasks(project_id)
    except Exception:
        existing = []
    terminal = {"DONE", "FAILED", "BLOCKED", "ARCHIVED"}
    seen = {_feature_key(t.get("description")) for t in existing
            if str(t.get("status", "")).upper() not in terminal}
    kept = []
    dropped = []
    for d in (subtasks or []):
        if not d or not d.strip():
            continue
        k = _feature_key(d)
        if k in seen:
            dropped.append(d.strip())
            continue
        seen.add(k)
        kept.append(d)
    return kept, dropped


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
    "project_failed": "Stopped: the project is in a FAILED state — one or "
                      "more tasks failed (see failed_tasks). NOTHING is "
                      "runnable until a failed task is fixed. Do NOT report "
                      "the project as complete.",
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
        "failed", "repeated_failures", "budget_or_inactive", "needs_user",
        "project_failed")
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


def _released_guard(store: ProjectStore, project_id) -> Optional[str]:
    """Error string when ``project_id`` is RELEASED (mutations must fork a
    version), else None. Applied to every mutating action — a RELEASED
    project is human-attested working software and the agent's own measured
    failure mode is regressing exactly such artifacts (2026-07-25 audit)."""
    try:
        if not project_id:
            return None
        proj = store.get_project(project_id)
        if proj and str(proj.get("status", "")).upper() == "RELEASED":
            return (
                f"project {project_id} is RELEASED (human-attested, "
                f"immutable). To make changes, fork a new version: "
                f"manage_projects action=create_version project_id="
                f"{project_id} description=\"<the requested change>\" — "
                f"the released version keeps working untouched. To run/use "
                f"it, follow the RELEASE DIRECTIONS in the briefing."
            )
    except Exception:
        return None
    return None


def _project_service_entries(context, project_id: str) -> list:
    """Service-registry entries owned by this project.

    Ownership is the first-class ``project_id`` field stamped at start
    (2026-07-30, §4G) — the old workdir/command string match survives only
    as the fallback for LEGACY entries that predate the field, and an
    entry stamped for a DIFFERENT project is never heuristic-matched (a
    command mentioning two project paths must not be claimed twice)."""
    try:
        from ..sandbox.services import get_service_supervisor
        sup = get_service_supervisor(getattr(context, "sandbox_manager", None))
        if sup is None:
            return []
        needle = f"projects/{project_id}"
        out = []
        for e in sup.list_entries():
            owner = e.get("project_id")
            if owner:
                if str(owner) == str(project_id):
                    out.append(e)
                continue
            # Legacy (unstamped) entry: match workdir OR command — live
            # registries carried workdir="/workspace" with the project path
            # inside the COMMAND (`cd /workspace/projects/<id> && node
            # server.js`), observed 2026-07-25 when the v2 release
            # rehearsal missed a running, registered service.
            if needle in str(e.get("workdir") or "") \
                    or needle in str(e.get("command") or ""):
                out.append(e)
        return out
    except Exception:
        return []


def project_services_summary(context, project_id: str) -> list:
    """Cheap (NO container execs) per-project service facts for the
    every-turn briefing: ``[{name, port, command, started_at, stale}]``.
    ``stale`` is True when the entry's container generation no longer
    matches the live container — the process died with the old container.
    Liveness beyond that is deliberately NOT probed here (it would cost
    docker execs on every turn); the model verifies via
    ``manage_services status`` when it matters."""
    entries = _project_service_entries(context, project_id)
    if not entries:
        return []
    _gen = None
    try:
        sm = getattr(context, "sandbox_manager", None)
        _gen = str(getattr(getattr(sm, "container", None), "id", None) or "")
    except Exception:
        _gen = None
    out = []
    for e in entries:
        stamped = str(e.get("container_id") or "")
        out.append({
            "name": e.get("name") or e.get("key"),
            "port": e.get("port"),
            "command": str(e.get("command") or "")[:80],
            "started_at": e.get("started_at"),
            "stale": bool(_gen and stamped and _gen != stamped),
        })
    return out


def _probe_tcp(port, attempts: int = 4, delay_s: float = 1.0) -> bool:
    """Bounded local TCP reachability probe (release rehearsal)."""
    import socket
    import time as _time
    for _ in range(max(1, attempts)):
        try:
            with socket.socket() as s:
                s.settimeout(1.5)
                if s.connect_ex(("127.0.0.1", int(port))) == 0:
                    return True
        except Exception:
            pass
        _time.sleep(delay_s)
    return False


def _stop_project_services(context, project_id: str,
                           purge_state: bool = False) -> int:
    """Stop (and thereby deregister from liveness) every service whose
    workdir/command belongs to ``project_id``. Lifecycle coupling
    (2026-07-25 review H6): hard delete used to rmtree the workspace out
    from under a RUNNING service, and archiving a released project left its
    rehearsed services up forever. ``purge_state`` (2026-07-30) removes the
    services' writable state dirs too — the HARD-DELETE contract is "no
    files left behind"; archive passes False and keeps state exactly like
    it keeps the workspace. Best-effort; returns services stopped."""
    stopped = 0
    try:
        from ..sandbox.services import get_service_supervisor
        sup = get_service_supervisor(getattr(context, "sandbox_manager", None))
        if sup is None:
            return 0
        for e in _project_service_entries(context, project_id):
            try:
                # The registry KEY addresses the exact entry across project
                # scopes; display name kept as fallback for stub entries.
                _addr = str(e.get("key") or e.get("name") or "")
                sup.stop(_addr)
                if purge_state:
                    sup.purge_state(_addr)
                stopped += 1
            except Exception:
                continue
        if stopped:
            pretty_log("Project Service", f"stopped {stopped} service(s) of project "
                                  f"{project_id} (lifecycle)", icon=Icons.STOP)
    except Exception:
        return stopped
    return stopped


def _release_rehearsal(context, store: ProjectStore, project_id: str) -> Dict[str, Any]:
    """Cold-start the project from its own recorded operational facts and
    verify it comes up — the release gate that keeps the dossier honest.

    Service projects: restart every registered service and TCP-probe its
    port. Static/GENERAL projects (no services): verify every registered
    deliverable exists on disk (rehearsal-after-tidy also catches
    deliverables the cleanup keep-set missed). Deterministic, no LLM.
    Returns {ok, services, checks, detail}.
    """
    from ..sandbox.services import get_service_supervisor
    entries = _project_service_entries(context, project_id)
    services: list = []
    checks: list = []
    ok = True
    if entries:
        sup = get_service_supervisor(getattr(context, "sandbox_manager", None))
        for e in entries:
            name = str(e.get("name") or "")
            _addr = str(e.get("key") or name)
            port = e.get("port")
            cmd_str = str(e.get("command") or "")
            try:
                _r_out = sup.restart(_addr)
            except Exception as ex:
                ok = False
                checks.append(f"{name}: restart raised {type(ex).__name__}")
                continue
            # The supervisor reports failures as "Error: …" strings, not
            # exceptions — a failed restart must fail the rehearsal
            # (review 2026-07-30: a portless service's failed restart
            # passed as `alive=True`).
            if isinstance(_r_out, str) and _r_out.startswith("Error:"):
                ok = False
                checks.append(f"{name}: restart failed — {_r_out[:120]}")
                continue
            # The lease may have MOVED the port on restart (preference
            # semantics), and the substituted command moved with it — read
            # BOTH from the fresh registry entry so the dossier records
            # what actually runs, not the pre-restart snapshot (review
            # 2026-07-30: dossier stored old-port command + new port).
            try:
                _fresh = next((e2 for e2 in sup.list_entries()
                               if e2.get("key") == _addr), None)
                if _fresh is not None:
                    if _fresh.get("port") is not None:
                        if port is not None and _fresh["port"] != port:
                            checks.append(f"{name}: port moved "
                                          f"{port}→{_fresh['port']} (lease)")
                        port = _fresh.get("port")
                    if _fresh.get("command"):
                        cmd_str = str(_fresh["command"])
            except Exception:
                pass
            alive = _probe_tcp(port) if port else True
            services.append({"name": name,
                             "command": cmd_str,
                             "port": port})
            if port and not alive:
                ok = False
                checks.append(f"{name}: port {port} unreachable after restart")
            else:
                checks.append(f"{name}: up"
                              + (f" · port {port} reachable" if port else ""))
    else:
        if not store.list_deliverables(project_id):
            ok = False
            checks.append("no services registered AND no deliverables — "
                          "nothing to release")

    # ── deliverables must EXIST, whether or not the project has services ──
    # Two defects fixed here on operator instruction (2026-08-22):
    #
    # 1. THE CHECK ONLY RAN WHEN THERE WERE NO SERVICES. It lived in the
    #    `else` above, so any project with a registered service was released
    #    without its deliverables ever being looked at — the service half and
    #    the file half were mutually exclusive when both are part of "is this
    #    thing actually shippable".
    #
    # 2. IT STATTED THE RAW PAYLOAD. Some rows carry the redundant
    #    `projects/<id>/` prefix (pre-2026-07-20 registrations), so
    #    `ws / payload` resolved to `<ws>/projects/<id>/…` and read as
    #    MISSING. Measured on the live WebOS project: the raw check called 3
    #    deliverables missing, two of which — `index.html` and `server.js`,
    #    the project's actual output — are right there. That is a permanent
    #    FALSE block whose documented repair (`unregister_file`) would have
    #    told the operator to delete the registration of a file that exists.
    #    `missing_deliverables` normalises through the same path contract
    #    registration uses and reports the one genuinely absent file.
    # ⚠ A RELEASE GATE MUST BE CONSERVATIVE, and `missing_deliverables` is
    # deliberately NOT: it returns nothing when the whole workspace is absent,
    # because "the directory is gone" is not evidence about individual files
    # and flagging all of them would bury a real single-file loss. That is the
    # right call for the briefing and the WRONG one here — measured before
    # this guard: deleting the entire workspace made the rehearsal report
    # "all 1 deliverable(s) present" and PASS. The most complete failure
    # available read as full success.
    _delivs = store.list_deliverables(project_id)
    if _delivs:
        _ws_dir = Path(str((store.get_project(project_id) or {}).get(
            "workspace_dir") or ""))
        if not _ws_dir or not _ws_dir.is_dir():
            ok = False
            checks.append(
                f"{len(_delivs)} deliverable(s) recorded but the project "
                f"workspace does not exist: {_ws_dir or '(unset)'}")
    _missing = sorted(store.missing_deliverables(project_id))
    if _missing:
        ok = False
        checks.append(
            f"deliverable(s) recorded but NOT on disk: {_missing[:5]}"
            + (f" (+{len(_missing) - 5} more)" if len(_missing) > 5 else "")
            + " — restore the file(s), or drop the stale record with "
              "manage_projects action=unregister_file")
    elif store.list_deliverables(project_id):
        checks.append(
            f"all {len(store.list_deliverables(project_id))} "
            f"deliverable(s) present")
    return {"ok": ok, "services": services, "checks": checks,
            "detail": "; ".join(checks)}


def _briefing(store: ProjectStore, project_id: str) -> Dict[str, Any]:
    """Compact status snapshot suitable for a resume banner.

    RELEASED projects get a RUNBOOK shape (2026-07-25 round 2): the old
    dev-style briefing told the model about task trees and "files you
    write live HERE" on an immutable workspace — contradicting the
    system-prompt runbook mode. Operational facts only, plus routing.
    """
    proj = store.get_project(project_id)
    if not proj:
        return {}
    if str(proj.get("status", "")).upper() == "RELEASED":
        rel = store.get_release(project_id) or {}
        children = [
            {"id": c["id"], "title": c["title"], "status": c["status"]}
            for c in store.list_children(project_id)
        ]
        return {
            "project": {
                "id": proj["id"], "title": proj["title"],
                "kind": proj["kind"], "status": "RELEASED",
                "goal": proj["goal"],
                "version": (proj.get("metadata") or {}).get("version", 1),
            },
            "release": rel,
            "versions": children,
            "note": ("RELEASED (immutable). To RUN it: follow release.directions "
                     "(start the listed services via manage_services if down). "
                     "To CHANGE it: action=create_version. Do not write into "
                     "this workspace."),
        }
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
    # Recent interactive work (the automatic finalize-chain record,
    # 2026-07-18) — without it a status/resume answer for a project
    # whose tasks all closed reads "DONE, nothing open" even after an
    # evening of post-completion debugging.
    try:
        work_log = [
            {"ts": w["ts"], **(w.get("payload") or {})}
            for w in store.recent_work_logs(project_id, limit=5)
        ]
    except Exception:
        work_log = []
    # Deliverable manifest + terminal-state retrospective + last dream
    # digest (2026-07-18) — the remaining write-only plumbing, now read.
    try:
        deliverables = store.list_deliverables(project_id)[:20]
    except Exception:
        deliverables = []
    try:
        _missing_deliverables = {
            p for p in store.missing_deliverables(project_id)
            if p in deliverables}
    except Exception:
        _missing_deliverables = set()
    # File manifest (2026-07-24): per-file descriptions, additive key so
    # existing consumers of `deliverables` (bare paths) are untouched.
    try:
        file_map = {
            p: {"desc": e.get("desc", ""), "role": e.get("role", "")}
            for p, e in store.get_file_manifest(project_id).items()
        }
    except Exception:
        file_map = {}
    retrospective = None
    if str(proj.get("status", "")).upper() in ("DONE", "FAILED", "ARCHIVED"):
        try:
            retrospective = plan.tree.generate_retrospective()
        except Exception:
            retrospective = None
    try:
        _dd = store.list_events(project_id, limit=1, event_type="dream_digest")
        last_dream_digest = (_dd[0].get("payload") or {}) if _dd else None
    except Exception:
        last_dream_digest = None
    return {
        "project": {
            "id": proj["id"],
            "title": proj["title"],
            "kind": proj["kind"],
            "status": proj["status"],
            "goal": proj["goal"],
        },
        # Capped (2026-07-25 round 2): a 50-task project's full render blew
        # past 100KB into the tool result — the exact hazard task_list was
        # slimmed for.
        "task_tree": ((lambda t: t[:3500] + "\n… (tree truncated — use "
                       "action=task_list for the rest)" if len(t) > 3500
                       else t)(plan.render() or "(empty)")),
        "next_ready": (
            {"id": nxt.id, "description": nxt.description, "status": nxt.status.value}
            if nxt else None
        ),
        "recent_events": [
            {"type": e["type"], "ts": e["ts"], "payload": e["payload"]}
            for e in events
        ],
        "recent_work_log": work_log,
        "deliverables": deliverables,
        # A registered deliverable is a CLAIM (queue #10): nothing verifies
        # the file exists at registration and nothing reconciles later. The
        # prompt briefing marks the gone ones ⚠ MISSING and points the reader
        # HERE for detail — so this view has to carry the same warning, or
        # the model walks from the caveat straight to an unqualified list.
        "deliverables_missing": sorted(_missing_deliverables),
        "file_map": file_map,
        # RELEASED runbook (2026-07-25): present only on released projects —
        # the operational directions the resume/run flow leads with.
        "release": (store.get_release(project_id) or None),
        "retrospective": retrospective,
        "last_dream_digest": last_dream_digest,
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
    # Empty = not passed. The old default "GENERAL" made an explicit
    # `update kind=GENERAL` indistinguishable from no intent, so a CODING
    # project could never be set back (round-2 M9). _infer_kind treats ""
    # as no-explicit for create/promote.
    kind: str = "",
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
    # per-file manifest / history
    file_path: str = "",
    file_role: str = "",
    # release runbook prose (action=release)
    directions: str = "",
    # cross-project search (action=search)
    query: str = "",
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
    act = str(action or "").strip().lower()
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
    # parent_id too: the store canonicalises it on write, but the sibling-
    # duplicate guard in task_add compares it RAW against stored (canonical)
    # parent ids — a case-mangled parent id slipped past the guard.
    if parent_id:
        parent_id = _canon_pid(parent_id)

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
    if metadata is not None and not isinstance(metadata, dict):
        # A non-dict here (e.g. metadata='["x"]' parsing to a LIST) would be
        # PERSISTED verbatim and poison every later `dict(metadata)` /
        # `metadata.get(...)` reader — reject at the boundary instead.
        return _err(
            f"metadata must be a JSON object, got {type(metadata).__name__} "
            '— e.g. metadata={"key": "value"}'
        )

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

    # Include the target file for the per-file actions — "action=describe_file"
    # alone was undiagnosable in the live log (2026-07-24: three calls, no way
    # to see what they targeted or carried).
    _log_detail = f"action={act}"
    if act in ("describe_file", "file_history"):
        _log_detail += (f" file={file_path or payload or '?'}"
                        + (f" desc={description[:60]!r}" if description else " (no desc)"))
    pretty_log("Project Tool", _log_detail, icon=Icons.BRAIN_PLAN)

    # Resolve an implicit project_id from context.current_project_id so
    # the LLM doesn't have to pass it on every call while in project mode.
    # BUT: never let the auto-fill shadow an EXPLICIT title for a title-
    # resolvable action — auto-filling `current` there made
    # `delete title="chess"` (with any project active) resolve to and HARD-
    # DELETE the ACTIVE project instead of "chess", because _resolve_project_ref
    # short-circuits on a valid project_id and ignores the title.
    # NB: "update" is NOT here — its `title` arg is a RENAME value, not a
    # lookup key, so auto-filling `current` for a rename is correct.
    _TITLE_RESOLVABLE = {"delete", "archive", "get", "switch", "resume",
                         "release", "create_version"}
    if project_id is None and act not in {"create", "list", "promote_from_context"}:
        if not (title and act in _TITLE_RESOLVABLE):
            project_id = getattr(context, "current_project_id", None)

    # RELEASED immutability choke point (2026-07-25): every action that
    # would mutate a released project's record or workspace is refused with
    # a steer to create_version. Reads, switch/resume/status, archive
    # (retiring a release is legitimate), delete (tombstoned, explicit),
    # and release/create_version themselves stay allowed.
    _RELEASED_MUTATING = {
        "task_add", "task_update", "task_decompose", "artifact_add",
        "ledger", "config", "describe_file", "autoadvance", "update",
        "research", "cleanup", "unregister_file", "task_delete",
        "set_dependency",
    }
    if act in _RELEASED_MUTATING:
        # ledger/config READ forms (no payload) stay allowed.
        _is_read_form = (
            (act == "ledger" and not (ledger or "").strip())
            or (act == "config" and not (config_key or "").strip())
            or (act == "describe_file" and not (description or "").strip()
                and not (file_path or payload or "").strip())
        )
        if not _is_read_form:
            _rg = _released_guard(store, project_id)
            if _rg:
                return _err(_rg)

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
            # str(): a dict/list/int `title` arrives straight from
            # json.loads and `.strip()` on it raises AttributeError, which
            # the loop reports as a missing argument (§4DX round 3).
            normalized = str(title).strip().lower()
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
                if (existing["title"] or "").strip().lower() != normalized:
                    continue
                # A same-title RELEASED project must NOT be reused (2026-07-25):
                # the old path bound the conversation to it, wrote retry
                # metadata straight through the store (bypassing the
                # immutability choke), then told the model to task_add — which
                # the released guard refused. Contradictory guard pair; steer
                # to versioning instead.
                if str(existing.get("status", "")).upper() == "RELEASED":
                    return _err(
                        f"a RELEASED project with this title already exists "
                        f"({existing['id']}). It is immutable — to build on "
                        f"it, fork a new version: manage_projects "
                        f"action=create_version project_id={existing['id']} "
                        f"description=\"<what you want to change>\". To start "
                        f"something genuinely unrelated, pick a different "
                        f"title.")
                if existing["status"] in _TERMINAL_PROJECT_STATUSES:
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
                prior_constraints = _constraint_list(
                    existing_meta.get("constraints"))
                fresh = [c for c in req_constraints
                         if c.lower() not in {p.lower() for p in prior_constraints}]
                if fresh:
                    existing_meta["constraints"] = prior_constraints + fresh
                    # Re-arm: a retired constraint the user RESTATES is live
                    # intent again — drop it from the retired list so the
                    # lifecycle can't shadow it (retirement happens on
                    # project DONE; see store.retire_constraints).
                    _retired = _constraint_list(
                        existing_meta.get("constraints_retired"))
                    if _retired:
                        _fresh_keys = {c.lower() for c in fresh}
                        existing_meta["constraints_retired"] = [
                            c for c in _retired
                            if c.lower() not in _fresh_keys]
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
                subtasks, _create_dropped = _filter_duplicate_subtasks(
                    store, pid, subtasks)
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
                if _create_dropped:
                    instruction += (
                        f" NOTE: {len(_create_dropped)} subtask(s) were NOT "
                        "created (duplicates of each other/existing tasks): "
                        + " | ".join(d[:80] for d in _create_dropped[:6])
                        + ". If any is genuinely distinct work, re-add it "
                        "with task_add and a more specific description.")
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
            def _fam(p):
                m = p.get("metadata") or {}
                bits = []
                if m.get("version"):
                    bits.append(f"v{m['version']}")
                if m.get("parent_project_id"):
                    bits.append(f"fork of {m['parent_project_id']}")
                return f"  [{', '.join(bits)}]" if bits else ""
            lines = [
                f"{p.get('id')}  {str(p.get('status') or ''):<9}  "
                f"{p.get('title')}{_fam(p)}"
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
            if not project_id and not title:
                return _no_active_project(store)
            # Resolve a title/slug to the real id (e.g. `get petai`) via the
            # shared resolver so a title isn't a hard "not found". The
            # explicit `title` param is honoured too — the suppression above
            # already keeps `current` from shadowing it.
            rid, rerr = _resolve_project_ref(store, project_id, title)
            if rerr:
                return _err(rerr)                # ambiguous title
            if not rid:
                # Recall-fallback on empty lookups (journal §4C routing
                # variance): "when does project Kestrel ship?" routed here,
                # got a bare not-found, and the model gave up — while the
                # answer sat in vector memory at 0.76 relevance. Attach any
                # memory hits so the model can answer without a second,
                # unprompted tool hop.
                return await _not_found_with_recall(context, project_id or title)
            # Slim view (2026-07-25 round 2): the raw row dumped the FULL
            # metadata blob — 60-entry manifest, 4KB release directions,
            # runtime counters — into the tool result. Compact summary +
            # the briefing; deep metadata stays reachable via the dedicated
            # read actions (describe_file/ledger/config/artifact_list).
            _p = store.get_project(rid) or {}
            _m = _p.get("metadata") or {}
            return _ok({
                "id": _p.get("id"), "title": _p.get("title"),
                "kind": _p.get("kind"), "status": _p.get("status"),
                "goal": _p.get("goal"),
                "version": _m.get("version", 1),
                "parent_project_id": _m.get("parent_project_id"),
                "workspace_dir": _p.get("workspace_dir"),
                "briefing": _briefing(store, rid),
            })

        if act == "switch":
            if not project_id and not title:
                return _err("project_id (or title) is required for action=switch")
            # Resolve a title/slug to the real id so `switch petai` works
            # instead of erroring + striking on a recoverable title/id mixup.
            rid, rerr = _resolve_project_ref(store, project_id, title)
            if rerr:
                return _err(rerr)                # ambiguous title
            if not rid:
                return _err(f"project not found: {project_id or title!r}")
            _set_current(context, rid)
            _b = _briefing(store, rid)
            # A RELEASED runbook briefing carries its own routing note —
            # don't override it with the "files you write live HERE"
            # workspace note on an immutable workspace (round-2 M5).
            _note = (_b.get("note") if isinstance(_b, dict) and _b.get("release")
                     else _workspace_note(rid))
            return _ok({"switched_to": rid,
                        "workspace": f"projects/{rid}",
                        "note": _note,
                        "briefing": _b})

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
                if str(status).upper() == "RELEASED":
                    # RELEASED is earned, never assigned: only action=release
                    # (human command + rehearsal + dossier) may set it.
                    return _err(
                        "status=RELEASED cannot be set via update — use "
                        "action=release (requires the project DONE, usage "
                        "`directions`, and a passing release rehearsal).")
                fields["status"] = status
            if kind:
                fields["kind"] = kind
            if metadata is not None:
                fields["metadata"] = metadata
            if not fields:
                return _err("no updatable fields provided")
            ok = store.update_project(project_id, **fields)
            if not ok:
                return _err(f"update matched no rows for {project_id} — nothing was changed.")
            return _ok({"updated": ok, "fields": list(fields.keys())})

        if act == "constraint_retire":
            # Constraint lifecycle (2026-08-01): stop a stored constraint
            # from replaying into future requests WITHOUT deleting the
            # record — it moves to metadata.constraints_retired (project
            # DONE retires all of them automatically; the user restating
            # one re-arms it). payload = the exact constraint text, its
            # 0-based index in the active list, or "all".
            if not project_id and not title:
                return _err("project_id (or title) is required for "
                            "action=constraint_retire")
            _rid, _rerr = _resolve_project_ref(store, project_id, title)
            if _rerr:
                return _err(_rerr)
            if not _rid:
                return _err(f"project not found: {project_id or title!r}")
            proj = store.get_project(_rid) or {}
            active = _constraint_list(
                (proj.get("metadata") or {}).get("constraints"))
            if not active:
                return _err("project has no active constraints to retire")
            sel = str(payload or "").strip()
            if not sel:
                return _err(
                    "payload is required: the exact constraint text, its "
                    f"index (0-{len(active) - 1}), or 'all'. Active: "
                    + " | ".join(f"[{i}] {c}" for i, c in enumerate(active)))
            # Exact text wins over index — a constraint whose text IS a
            # digit must not be shadowed by positional selection.
            text_matches = [c for c in active if c.lower() == sel.lower()]
            if sel.lower() == "all":
                targets = active
            elif text_matches:
                targets = text_matches
            elif sel.isdigit() and int(sel) < len(active):
                targets = [active[int(sel)]]
            else:
                return _err(
                    f"no active constraint matches {sel!r}. Active: "
                    + " | ".join(f"[{i}] {c}"
                                 for i, c in enumerate(active)))
            retired = store.retire_constraints(
                _rid, reason="explicit constraint_retire", only=targets)
            if not retired:
                return _err("nothing was retired (concurrent change?) — "
                            "re-run action=get and retry")
            pretty_log("Project Constraints",
                       f"retired {len(retired)} constraint(s) on {_rid}: "
                       + " | ".join(c[:60] for c in retired),
                       icon=Icons.CONSTRAINT)
            remaining = [c for c in active if c not in retired]
            return _ok({"retired": retired, "active_constraints": remaining,
                        "note": ("Retired constraints stop replaying into "
                                 "requests/verifier immediately. They are "
                                 "kept in metadata.constraints_retired and "
                                 "re-arm only if the user restates them.")})

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
            # Stop the project's services BEFORE the workspace vanishes —
            # rmtree under a running service left orphaned processes +
            # registry entries (review H6). Hard delete also purges their
            # writable state dirs ("no files left behind").
            _stop_project_services(context, rid, purge_state=True)
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
            # Retiring a project retires its services too — archiving a
            # RELEASED project used to leave its rehearsed services running
            # forever (review H6). Resume can restart them per the dossier.
            _stop_project_services(context, rid)
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
            if not project_id and not title:
                return _err("project_id (or title) is required for action=resume")
            # Same title/slug resolution as get/switch — "resume petai" is
            # the natural way users refer back to an old project.
            rid, rerr = _resolve_project_ref(store, project_id, title)
            if rerr:
                return _err(rerr)                # ambiguous title
            if not rid:
                return _err(f"project not found: {project_id or title!r}")
            project_id = rid
            proj = store.get_project(project_id)
            if not proj:
                return _err(f"project not found: {project_id}")
            # Un-archive by restoring the status it was archived FROM
            # (metadata.archived_from, stashed by delete_project since
            # 2026-07-25). The old unconditional →ACTIVE silently STRIPPED
            # a RELEASED project's attestation on resume — the guards all
            # key on status and vanished while RELEASE.md still claimed
            # immutability. Legacy archives without the stash restore to
            # ACTIVE as before.
            if str(proj.get("status", "")).upper() == "ARCHIVED":
                _from = str(((proj.get("metadata") or {})
                             .get("archived_from")) or "ACTIVE").upper()
                store.update_project(project_id, status=_from)

                def _clear_af(meta):
                    meta.pop("archived_from", None)
                    return meta
                try:
                    store._atomic_metadata_update(project_id, _clear_af)
                except Exception:
                    pass
                if _from == "RELEASED":
                    store.log_event(project_id, None, "project_resumed",
                                    {"restored_status": "RELEASED"})
                    _b = _briefing(store, project_id)
                    if isinstance(_b, dict):
                        _b = {**_b,
                              "note": ("Restored to RELEASED (immutable). "
                                       "To run it, follow the release "
                                       "directions; to change it, "
                                       "action=create_version.")}
                    _set_current(context, project_id)
                    return _ok(_b)
            _set_current(context, project_id)
            store.log_event(project_id, None, "project_resumed", {})
            _b = _briefing(store, project_id)
            if isinstance(_b, dict):
                # Keep the runbook note on RELEASED resumes (round-2 M5).
                _note = (_b.get("note") if _b.get("release")
                         else _workspace_note(project_id))
                _b = {**_b, "workspace": f"projects/{project_id}",
                      "note": _note}
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
                dep = DependencyType[str(dependency_type).upper()]
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
                    st_enum = TaskStatus[str(status).upper()]
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
            result_only: List[str] = []
            gated: List[str] = []
            gated_constraints: List[str] = []
            active_constraints: List[str] = []
            judged_violations: List[str] = []
            judged_violation_reasons: List[str] = []
            still_failing: List[str] = []
            still_failing_reasons: List[str] = []
            # (ok, reason) audit cache keyed by the task's FILE SET. The
            # audit is per-file-set (constraint_gate judges specific files);
            # a single per-CALL cache reused the FIRST task's verdict for
            # every sibling in a bulk close, so a constraint-VIOLATING
            # sibling with different files inherited the first task's clean
            # verdict and closed DONE unaudited (the exact bypass this gate
            # exists to stop). Keyed by frozenset(files) so identical file
            # sets still cost one audit, but distinct files are each judged.
            _constraint_audit_cache: Dict[frozenset, Any] = {}
            for tid in target_ids:
                if tid not in plan.tree.nodes:
                    missing.append(tid)
                    continue
                # Stale-evidence invalidation (2026-08-01): closing a FAILED
                # task whose recorded failure is a MECHANICAL check (a parse/
                # syntax verdict) re-runs that check on the file as it is NOW.
                # The world changes between the failure and the close attempt
                # — the Mini AI incident had a rewritten, verified core.py on
                # disk while the ledger's "does NOT parse" from an earlier
                # build blocked every close. A passing live check becomes the
                # result evidence; a failing one refuses the DONE with the
                # CURRENT diagnostic instead of the stale text.
                live_evidence = ""
                if (st_enum == TaskStatus.DONE
                        and plan.tree.nodes[tid].status == TaskStatus.FAILED
                        and (plan.tree.nodes[tid].failure_reason or "").strip()):
                    _verdict, _detail = await _recheck_stale_parse_failure(
                        store, project_id, tid,
                        plan.tree.nodes[tid].failure_reason, deliverables)
                    if _verdict is False:
                        still_failing.append(tid)
                        still_failing_reasons.append(_detail)
                        continue
                    if _verdict is True:
                        live_evidence = _detail
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
                # A passing live re-check counts as evidence: it is exactly
                # the "go verify first" this gate demands, already done.
                task_constraints = [
                    str(c) for c in
                    (plan.tree.nodes[tid].constraints or [])
                ] + proj_constraints
                if (st_enum == TaskStatus.DONE and task_constraints
                        and not (result or "").strip()
                        and not live_evidence
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
                    # Audit ONLY the files THIS task produced.
                    #
                    # This used to pass the project-wide file collector from
                    # core.project_advancer — every text file in the project
                    # workspace. That collector exists for the coding executor's
                    # non-regression guard (which must see all files); using it
                    # here made the gate judge the WHOLE PROJECT on every task
                    # close, which is a permanent deadlock: one violating file
                    # written by an early task blocks the DONE transition of
                    # every OTHER task forever, and no amount of fixing the
                    # current task's own artifact can clear it. Observed live
                    # 2026-07-11 (project 6051abfb21b8): closing task #12 was
                    # blocked by a verbatim quote in context_boundary.md — task
                    # #1's artifact, already DONE — so the model re-read that
                    # file, correctly concluded "the audit may be stale",
                    # retried, tripped the no-progress loop breaker, and the
                    # request died after 189s with zero progress. Every later
                    # task_update would have hit the identical wall.
                    #
                    # Scoping to this task's own files makes the gate
                    # actionable ("fix what you just wrote"), cheaper, and
                    # non-deadlocking. When a task has NO attributable files
                    # we skip the judgment gate — the evidence gate above
                    # still applies, and auditing someone else's artifact is
                    # exactly the bug.
                    _task_files = _files_for_task(store, project_id, tid,
                                                  deliverables)
                    # Audit THIS task's own files. Cache by the file set so a
                    # bulk close with identical file sets still costs one
                    # audit, but a sibling with DIFFERENT files is judged on
                    # its OWN artifact — never the first task's verdict.
                    _task_audit = None
                    if _task_files:
                        _fkey = frozenset(_task_files)
                        _task_audit = _constraint_audit_cache.get(_fkey)
                        if _task_audit is None:
                            try:
                                from ..core.build_gates import constraint_gate
                                _task_audit = await constraint_gate(
                                    context, task_constraints, _task_files,
                                    is_background=False)
                            except Exception:
                                logger.debug("constraint judgment gate skipped",
                                             exc_info=True)
                                _task_audit = (True, "")
                            _constraint_audit_cache[_fkey] = _task_audit
                    # Refuse only tasks that HAD files to audit. Fileless
                    # tasks skip the judgment gate (the evidence gate above
                    # still applies), per the _files_for_task contract.
                    if (_task_files and _task_audit is not None
                            and not _task_audit[0]):
                        judged_violations.append(tid)
                        _reason = _task_audit[1] if len(_task_audit) > 1 else ""
                        if _reason:
                            judged_violation_reasons.append(_reason)
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
                    # Manifest seed (2026-07-24): the model's own result
                    # summary for this DONE describes what it built — reuse
                    # it as each registered file's description head (refined
                    # later via describe_file).
                    _fdesc = " ".join((result or "").split())[:180]
                    for rel in keep_paths:
                        try:
                            store.register_file_artifact(
                                tid, rel, description=_fdesc)
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
                    # A passing live re-check doubles as the result when the
                    # model supplied none — it is real, current evidence.
                    plan.update_status(tid, st_enum,
                                       result=(result or live_evidence),
                                       failure_reason=failure_reason)
                extras: Dict[str, Any] = {}
                # `result` without `status`: persist it as the task's
                # result_summary. update_status is guarded on st_enum, so
                # this call used to write NOTHING while still returning
                # {"updated": [...], "count": 1} — a success-shaped no-op
                # that the gate-recovery instruction steered the model
                # into repeating forever. The stored summary also clears
                # the evidence gates on the next status=DONE call.
                if st_enum is None and (result or "").strip():
                    extras["result_summary"] = str(result).strip()
                    result_only.append(tid)
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
                # Name the ids that DO exist (2026-08-01): a bare
                # `missing: [...]` next to `count: 0` read as an empty
                # success, and the model kept retrying invented ids for
                # tasks that were never persisted (decompose dedup) —
                # phantom train.py/demo.py ids burned 4 turns in the Mini
                # AI incident. Showing the real ledger breaks the illusion
                # in one round-trip.
                _valid = [
                    {"id": t.id, "status": t.status.value,
                     "description": (t.description or "")[:80]}
                    for t in plan.tree.nodes.values()
                ][:20]
                payload["agent_instruction_missing"] = (
                    f"{len(missing)} task id(s) do NOT exist in this "
                    f"project: {missing}. Nothing was updated for them — "
                    "do not retry these ids. The project's ACTUAL tasks "
                    "are listed in `valid_tasks`; if the work you want to "
                    "record has no task there, it was never persisted — "
                    "create it first with task_add, then task_update THAT "
                    "id to done with a `result` describing the evidence."
                )
                payload["valid_tasks"] = _valid
            if result_only:
                payload["result_recorded_status_unchanged"] = result_only
                payload["agent_instruction_result_only"] = (
                    f"Recorded `result` as evidence on {len(result_only)} "
                    "task(s), but you passed no `status`, so the task "
                    "status is UNCHANGED — a result alone does not close a "
                    "task. To mark it complete, re-call task_update with "
                    "BOTH status=done AND the task id(s)."
                )
            if still_failing:
                payload["still_failing"] = still_failing
                payload["agent_instruction_still_failing"] = (
                    f"REFUSED to mark {len(still_failing)} task(s) DONE: the "
                    "recorded failure was re-checked LIVE against the current "
                    "file(s) and it STILL fails. Current diagnostic(s): "
                    + " | ".join(still_failing_reasons)
                    + " Fix the file first, then re-call task_update with "
                    "status=done."
                )
            if judged_violations:
                payload["constraint_violations"] = judged_violations
                payload["agent_instruction_violation"] = (
                    f"REFUSED to mark {len(judged_violations)} task(s) DONE: "
                    "an audit of the project files found a violation of the "
                    "user's stated constraints. "
                    + ("; ".join(dict.fromkeys(judged_violation_reasons))
                       if judged_violation_reasons else "")
                    + " Fix the deliverable so it actually honors the "
                    "constraint, then re-call task_update with status=done "
                    "and a `result`. Only if the USER has explicitly "
                    "approved this exception, re-call with status=done and "
                    "a `result` beginning with 'CONSTRAINT-OVERRIDE:' and "
                    "their reason."
                )
            if gated_constraints:
                payload["gated_constraints"] = gated_constraints
                payload["constraints"] = active_constraints
                payload["agent_instruction_constraints"] = (
                    f"Held {len(gated_constraints)} task(s) at non-DONE: the "
                    f"user attached EXPLICIT CONSTRAINTS and you provided no "
                    f"result evidence. Re-read the deliverable, confirm each "
                    f"constraint holds, then re-call task_update with "
                    f"status=done AND a `result` that states — per constraint "
                    f"— HOW the work honors it (a result without status=done "
                    f"records evidence but does NOT close the task). "
                    f"Constraints: "
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
                    f"re-call task_update with status=done AND a `result` "
                    f"describing what you saw.\n"
                    f"  • If the task is NOT visual (a text / research / markdown "
                    f"/ analysis task — this gate can misfire on wording): just "
                    f"re-call task_update with status=done and a one-line `result` "
                    f"describing what you produced. That counts as evidence and "
                    f"clears the gate immediately. (Pass BOTH `status=done` and "
                    f"`result=...` — a result without status does not close the "
                    f"task.)\n"
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
                subtasks, dropped_dupes = _filter_duplicate_subtasks(
                    store, project_id, subtasks)
                if not subtasks:
                    return _ok({"created": [], "parent_id": None,
                                "dropped_duplicates": dropped_dupes,
                                "note": "all subtasks duplicated existing "
                                        "tasks; nothing added"})
            else:
                dropped_dupes = []

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
            # Persisted-vs-planned transparency (2026-08-01): the model plans
            # N tasks, presents N to the user, but only len(ids) exist. A
            # silent drop here left the agent operating on PHANTOM tasks for
            # a whole request (Mini AI incident) — every drop must be echoed
            # back so the model can re-add genuinely-distinct work.
            _dropped_note = {}
            if dropped_dupes:
                _dropped_note = {
                    "dropped_duplicates": dropped_dupes,
                    "agent_instruction_dropped": (
                        f"{len(dropped_dupes)} subtask(s) were NOT created — "
                        "they deduplicated against existing/sibling tasks: "
                        + " | ".join(d[:80] for d in dropped_dupes[:6])
                        + ". Only the ids in `created` exist. If any dropped "
                        "subtask is genuinely distinct work, re-add it with "
                        "task_add and a more specific description. Present "
                        "ONLY the tasks in `tasks` to the user."
                    ),
                }
            return _ok({"created": ids,
                        **_dropped_note,
                        # id+description pairs (2026-07-25, req f59a793d):
                        # bare ids made this output unverifiable — the
                        # verifier's "task tree" evidence slot held nothing
                        # but hex ids + the STOP boilerplate, so it refuted
                        # a plan reply's contents as "not confirmed in the
                        # task tree". Echoing what was created also lets
                        # later turns reference tasks without a task_list
                        # round-trip. `created` keeps its shape (bare ids)
                        # for existing consumers.
                        "tasks": [{"id": tid, "description": desc}
                                  for tid, desc in pairs],
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
                # "Nothing ready" without saying WHY reads as "all done".
                # Name the tasks parked in FAILED/NEEDS_USER and how to
                # revive them — the 2026-08-01 wedge had exactly one FAILED
                # leaf behind this message and the model concluded the
                # state was unreachable.
                _parked = [
                    {"id": t.id, "status": t.status.value,
                     "description": (t.description or "")[:100],
                     "failure_reason": (t.failure_reason or "")[:160]}
                    for t in plan.tree.nodes.values()
                    if t.status in (TaskStatus.FAILED, TaskStatus.NEEDS_USER)
                ][:10]
                _payload: Dict[str, Any] = {
                    "next": None,
                    "reason": "no READY/PENDING leaf available"}
                if _parked:
                    _payload["parked_tasks"] = _parked
                    _payload["agent_instruction"] = (
                        f"No runnable task, but {len(_parked)} task(s) are "
                        "parked (FAILED/NEEDS_USER — listed in "
                        "parked_tasks). A FAILED task is NOT done: either "
                        "fix its deliverable and task_update it with "
                        "status=done plus a `result` (stale recorded "
                        "failures are re-checked live), or set it "
                        "status=pending to retry. A NEEDS_USER task wants "
                        "the user's answer — ask them.")
                return _ok(_payload)
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

        if act == "artifact_list":
            # Read side of the artifact store (2026-07-18). Until now
            # artifacts were write-only through the tool surface: the
            # deliverable manifest, autoadvance's tool_call output
            # payloads (up to 8k chars each) and note artifacts were
            # unreachable — the model literally could not read back
            # what it had recorded. Scope with task_id, filter with
            # artifact_kind, page with limit. Payload heads are bounded
            # except `file` (a path IS the payload).
            if not task_id and not project_id:
                return _err("no active project (pass project_id, task_id, "
                            "or switch first)")
            arts = (store.list_artifacts(task_id=task_id) if task_id
                    else store.list_artifacts(project_id=project_id))
            if artifact_kind:
                arts = [a for a in arts
                        if str(a.get("kind") or "") == artifact_kind]
            total = len(arts)
            arts = arts[-max(1, int(limit)):]
            out = []
            for a in arts:
                p = str(a.get("payload") or "")
                if str(a.get("kind")) != "file" and len(p) > 400:
                    p = p[:400] + f"… [truncated; {len(p)} chars total]"
                out.append({"task_id": a.get("task_id"),
                            "kind": a.get("kind"), "payload": p})
            return _ok({"artifacts": out, "total": total,
                        "shown": len(out)})

        if act == "event_log":
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            return _ok({
                "events": store.list_events(
                    project_id, limit=limit, event_type=event_type
                ),
            })

        if act == "cleanup":
            # Explicit, user-triggered tidy (2026-07-18): same narrow
            # deletion set as the recurring idle pass (debris +
            # unregistered, UNREFERENCED media; never source files,
            # never the keep-set) but with NO age gate — the user asked
            # NOW. The DONE transition still runs the full sweep; this
            # is for "clean up the project" mid-flight or post-DONE.
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            # Destructive → validate the model-supplied id against the store
            # like delete/archive do. cleanup used to pass the raw string
            # straight to tidy_project_workspace, so a traversal id like
            # "../.." resolved to the sandbox PARENT and the tidy walked
            # (and deleted from) directories outside the project sandbox.
            rid, rerr = _resolve_project_ref(store, project_id)
            if rerr:
                return _err(rerr)                # ambiguous title
            if not rid:
                return _err(
                    f"project not found: {project_id!r} — nothing was "
                    f"cleaned. Use action=list to see the real ids."
                )
            from ..core.workspace_cleanup import tidy_project_workspace
            ts = tidy_project_workspace(store, rid, min_age_hours=0.0)
            return _ok({
                "status": ts.get("status"),
                "deleted": ts.get("deleted"),
                "freed_bytes": ts.get("freed_bytes"),
                "kept_referenced_assets": ts.get("kept_referenced"),
                "note": ("Deliverables, source files and referenced media "
                         "are never touched by cleanup."),
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

        if act == "describe_file":
            # Per-file manifest write (2026-07-24): record WHAT a file
            # is/does the moment the insight is earned, so the next session
            # (or a smaller model) is directed to the right files instead of
            # re-deriving the layout. Without `description` → read the
            # manifest back.
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            fp = (file_path or payload or "").strip()
            desc = (description or "").strip()
            if not desc and not fp:
                # Bare describe_file → read the manifest back.
                mf = store.get_file_manifest(project_id)
                return _ok({"file_manifest": mf, "files": len(mf)})
            if not desc:
                # file_path given but no text: the caller INTENDED to write —
                # steer instead of silently returning the (possibly empty)
                # manifest, which read as "the tool returned empty" live.
                return _err(
                    f"describe_file needs `description` — the one-line 'what "
                    f"{fp} is/does' text to record. Retry with BOTH "
                    f"file_path and description.")
            if not fp:
                return _err("file_path is required (project-relative, e.g. "
                            "'server.js')")
            if not store.describe_file(project_id, fp, desc, role=file_role):
                return _err(f"could not describe {fp!r} (blank/traversal "
                            "path, or project not found)")
            return _ok({
                "described": fp,
                "desc": desc[:200],
                "action_taken": "manifest_updated",
                "note": "PROJECT_MAP.md refreshed in the workspace.",
            })

        if act == "unregister_file":
            # Repair path for renamed/removed deliverables (2026-07-25): a
            # stale file-artifact row otherwise blocks release FOREVER
            # ("deliverable missing on disk", no tool could fix the record).
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            fp = (file_path or payload or "").strip()
            if not fp:
                return _err("file_path is required (the project-relative "
                            "path to remove from the deliverable record + "
                            "file map)")
            counts = store.unregister_file(project_id, fp)
            if not (counts["artifacts_removed"] or counts["manifest_removed"]):
                return _err(f"{fp!r} was not registered (check "
                            f"action=artifact_list for the recorded paths)")
            return _ok({
                "unregistered": fp, **counts,
                "note": ("Removed from deliverables + file map; if the file "
                         "still exists on disk it is no longer in the "
                         "cleanup keep-set. Re-register the renamed file "
                         "with task_update deliverables=[...] or "
                         "describe_file."),
            })

        if act == "file_history":
            # Per-file action journal read (2026-07-24): every work_log /
            # event row that touched this file — "what happened to X?"
            # answered from the journal instead of re-reading the file.
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            fp = (file_path or payload or "").strip()
            if not fp:
                return _err("file_path is required")
            hist = store.file_history(project_id, fp, limit=max(1, int(limit)))
            mf_entry = store.get_file_manifest(project_id).get(
                store._normalize_rel_path(project_id, fp) or fp)
            return _ok({
                "file": fp,
                "description": (mf_entry or {}).get("desc", ""),
                "history": hist,
                "shown": len(hist),
            })

        # ---- human-attested release + versioning (2026-07-25) ----------

        if act == "release":
            # DONE → RELEASED, gated on: (1) the human asked (this call IS
            # the confirmation), (2) model-supplied usage directions, (3) a
            # deterministic rehearsal — tidy, then cold-start the services /
            # verify deliverables — so the dossier is written from facts
            # that worked seconds before RELEASED was stamped.
            if not project_id and not title:
                return _err("no active project (pass project_id, a title, "
                            "or switch first)")
            # Title-resolve like every other destructive action (round-2 M9):
            # "release the journal" must work without the hex id.
            _rid, _rerr = _resolve_project_ref(store, project_id, title)
            if _rerr:
                return _err(_rerr)
            if _rid:
                project_id = _rid
            proj = store.get_project(project_id)
            if not proj:
                return _err(f"project not found: {project_id}")
            _status = str(proj.get("status", "")).upper()
            if _status == "RELEASED":
                return _ok({"status": "RELEASED",
                            "note": "already released",
                            "release": store.get_release(project_id)})
            if _status != "DONE":
                return _err(
                    f"project is {_status}, not DONE — finish it first "
                    f"(all tasks closed, status rolled up to DONE), then "
                    f"release.")
            _dirs = " ".join((directions or "").split())
            if len(_dirs) < 30:
                return _err(
                    "release needs `directions`: a usage runbook for the "
                    "USER (what this project is, how to use it, what the "
                    "features are, any known limitations). Write it from "
                    "what you know of the project and retry — the tool "
                    "adds the verified service commands/ports itself.")
            # (1) tidy so the rehearsal validates the KEPT file set.
            from ..core.workspace_cleanup import tidy_project_workspace
            try:
                tidy_project_workspace(store, project_id, min_age_hours=0.0)
            except Exception as e:
                logger.debug("release tidy skipped: %s", e)
            # (2) rehearse.
            reh = _release_rehearsal(context, store, project_id)
            if not reh.get("ok"):
                store.log_event(project_id, None, "release_rehearsal_failed",
                                {"detail": reh.get("detail", "")[:400]})
                # WARNING pretty: a failed release attempt is exactly the
                # actionable class per the chat-noise policy — and stop
                # whatever the rehearsal restarted (don't leave half-started
                # services of a NOT-released project running).
                pretty_log("Release Failed",
                           f"{proj.get('title')}: rehearsal failed — "
                           f"{reh.get('detail', '')[:120]}",
                           icon=Icons.FAIL, level="WARNING")
                _stop_project_services(context, project_id)
                return _err(
                    f"release rehearsal FAILED — project stays DONE. "
                    f"{reh.get('detail', '')}. Fix the issue (or register "
                    f"the missing deliverables) and retry.")
            # (3) compose the dossier from REHEARSED facts.
            mf = store.get_file_manifest(project_id)
            meta = (proj.get("metadata") or {})
            # Re-release after an unrelease bumps the dossier REVISION
            # (fork numbering owns "version"; a same-project re-test is a
            # revision of the same version).
            _prior = meta.get("release") or {}
            _revision = int(_prior.get("revision") or 0) + 1 \
                if _prior.get("unreleased_at") else int(_prior.get("revision") or 1)
            release = {
                "version": int(meta.get("version") or 1),
                "revision": _revision,
                "released_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "directions": _dirs[:4000],
                "services": reh.get("services") or [],
                "urls": [f"http://127.0.0.1:{s['port']}/"
                         for s in (reh.get("services") or [])
                         if s.get("port")],
                "deliverables": [
                    {"path": p, "desc": (mf.get(p) or {}).get("desc", "")}
                    for p in store.list_deliverables(project_id)[:30]
                ],
                "config": (meta.get("config") or {}),
                "rehearsal": reh.get("detail", "")[:600],
            }
            if not store.set_release(project_id, release):
                return _err("failed to persist the release dossier")
            # Seed config.port from the REHEARSED service (2026-07-25): the
            # first live fork served on the released version's port because
            # the parent never recorded one, so create_version's bump had
            # nothing to bump. The rehearsal's discovered port is the
            # authoritative value — record it so every future fork can
            # allocate around it.
            try:
                _svc_ports = [s["port"] for s in release["services"]
                              if s.get("port")]
                if _svc_ports and not str(
                        (meta.get("config") or {}).get("port", "")).strip():
                    store.set_config_value(project_id, "port",
                                           str(_svc_ports[0]))
            except Exception as e:
                logger.debug("release port seed skipped: %s", e)
            store.update_project(project_id, status="RELEASED")
            # OS-level immutability (round 2): mode bits catch what the
            # tool guards can't see (shell writes). AFTER RELEASE.md — the
            # dossier render needs the last write.
            try:
                store.set_workspace_readonly(project_id, True)
            except Exception:
                logger.debug("release chmod skipped", exc_info=True)
            store.log_event(project_id, None, "project_released",
                            {"version": release["version"],
                             "services": len(release["services"]),
                             "rehearsal": release["rehearsal"][:200]})
            _rel_title = re.sub(r"\s+v\d+$", "", str(proj.get("title") or ""))
            pretty_log("Project Released",
                       f"{_rel_title} v{release['version']} — "
                       f"rehearsal passed ({release['rehearsal'][:80]})",
                       icon=Icons.OK)
            return _ok({
                "status": "RELEASED",
                "version": release["version"],
                "rehearsal": release["rehearsal"],
                "release_md": "RELEASE.md written to the workspace",
                "note": ("This project is now immutable. Changes fork a new "
                         "version via action=create_version."),
            })

        if act == "create_version":
            # Fork a RELEASED project into a fresh development line. The
            # released version keeps working untouched (its services, its
            # dossier); the copy inherits the development knowledge (files,
            # ledger, config, manifest, constraints) but NOT the release
            # dossier, task history, or journal — and its service port is
            # bumped so v(n) and v(n+1) never collide.
            if not project_id and not title:
                return _err("no active project (pass project_id, a title, "
                            "or switch first)")
            _rid, _rerr = _resolve_project_ref(store, project_id, title)
            if _rerr:
                return _err(_rerr)
            if _rid:
                project_id = _rid
            parent = store.get_project(project_id)
            if not parent:
                return _err(f"project not found: {project_id}")
            if str(parent.get("status", "")).upper() != "RELEASED":
                return _err(
                    f"project is {parent.get('status')}, not RELEASED — "
                    f"non-released projects are edited IN PLACE (just make "
                    f"the changes); create_version exists to protect a "
                    f"released artifact.")
            pmeta = (parent.get("metadata") or {})
            new_version = int(pmeta.get("version") or 1) + 1
            base_title = re.sub(r"\s+v\d+$", "", str(parent.get("title") or ""))
            # Fork-exists guard (2026-07-25): a model retry loop calling
            # create_version twice minted TWO "X v2" projects, both on the
            # same bumped port. An existing non-archived fork is returned
            # idempotently instead.
            for _child in store.list_children(project_id):
                if str(_child.get("status", "")).upper() == "ARCHIVED":
                    continue
                _set_current(context, _child["id"])
                _b = _briefing(store, _child["id"])
                if isinstance(_b, dict):
                    _b = {**_b, "workspace": f"projects/{_child['id']}",
                          "note": (f"A development fork already exists — "
                                   f"switched to it ({_child['title']}, "
                                   f"{_child['status']}). Not creating a "
                                   f"duplicate."),
                          "existing_fork": True}
                return _ok(_b)
            # Inherited development knowledge; release dossier deliberately
            # NOT carried; port bumped to avoid colliding with the running
            # released version.
            new_cfg = dict(pmeta.get("config") or {})
            _old_port = None
            for _pk in ("port", "PORT"):
                if str(new_cfg.get(_pk, "")).strip().isdigit():
                    _old_port = int(new_cfg[_pk])
                    new_cfg[_pk] = str(_old_port + 1)
                    break
            new_meta = {
                "version": new_version,
                "parent_project_id": project_id,
                "design_ledger": pmeta.get("design_ledger") or "",
                "config": new_cfg,
                "file_manifest": pmeta.get("file_manifest") or {},
                # A fork is new in-flight work on the same deliverable line,
                # so the parent's constraints are live intent again — active
                # AND retired (a RELEASED parent necessarily passed DONE,
                # which retires; copying only the active list would inherit
                # [] and silently drop the constraint knowledge). They
                # retire again when THIS version goes DONE.
                "constraints": _rearm_inherited_constraints(pmeta),
                "research_index": pmeta.get("research_index") or [],
            }
            new_pid = store.create_project(
                f"{base_title} v{new_version}",
                kind=str(parent.get("kind") or "GENERAL"),
                goal=str(parent.get("goal") or ""),
                metadata=new_meta,
            )
            # Copy workspace files (not RELEASE.md — the dossier belongs to
            # the released version; not service scaffolding or caches).
            import shutil
            src_ws = Path(str(parent.get("workspace_dir") or ""))
            dst_ws = Path(str((store.get_project(new_pid) or {})
                              .get("workspace_dir") or ""))
            copied = 0
            if src_ws.is_dir() and str(dst_ws):
                _ignore = shutil.ignore_patterns(
                    "RELEASE.md", ".services", "__pycache__", "*.pyc",
                    "node_modules")
                shutil.copytree(src_ws, dst_ws, ignore=_ignore,
                                dirs_exist_ok=True)
                copied = sum(1 for p in dst_ws.rglob("*") if p.is_file())
                # copytree copies MODE BITS — a fork of a chmod'd-read-only
                # released workspace would itself be unwritable. Restore.
                try:
                    store.set_workspace_readonly(new_pid, False)
                except Exception:
                    pass
            # Seed task + re-register the parent's deliverables on it so the
            # copied files are in THIS project's cleanup keep-set (an
            # unregistered copy would be deleted by the end-of-life sweep).
            seed_desc = " ".join((description or "").split()) or (
                f"Apply requested changes (fork of {base_title} "
                f"v{new_version - 1})")
            seed_tid = store.add_task(new_pid, seed_desc)
            # Skip claims whose file is not in the SOURCE (queue #10): a fork
            # copies files, so a registration for a path that was never there
            # protects nothing — and re-registering it propagates the phantom
            # into the child, where the briefing asserts it all over again.
            # Phantom claims compound across forks otherwise.
            try:
                _src_missing = store.missing_deliverables(project_id)
            except Exception:
                _src_missing = set()
            for rel_path in store.list_deliverables(project_id):
                if rel_path in _src_missing:
                    continue
                try:
                    store.register_file_artifact(seed_tid, rel_path)
                except Exception:
                    pass
            if _old_port is not None:
                store.append_ledger(
                    new_pid,
                    f"v{new_version} runs on port {_old_port + 1} — "
                    f"v{new_version - 1} (RELEASED) keeps {_old_port}; "
                    f"register services under a -v{new_version} name")
            store.log_event(project_id, None, "version_forked",
                            {"child": new_pid, "version": new_version})
            store.log_event(new_pid, seed_tid, "version_created",
                            {"parent": project_id,
                             "version": new_version,
                             "files_copied": copied})
            _set_current(context, new_pid)
            store.render_project_map(new_pid)
            pretty_log("Version Forked",
                       f"{base_title} v{new_version} ({new_pid}) — "
                       f"{copied} file(s) copied; v{new_version - 1} stays "
                       f"released and running",
                       icon=Icons.BRAIN_PLAN)
            _b = _briefing(store, new_pid)
            if isinstance(_b, dict):
                # Action result FIRST (2026-07-25, req f59a793d): the fork
                # provenance used to live only in the tail keys + a
                # pretty_log line, so any head-truncation of this payload
                # (the verifier's evidence packer caps items at a few
                # hundred chars) dropped exactly the facts that prove what
                # this call DID — the late verifier then refuted a correct
                # "created v3 from v2" claim as unsubstantiated.
                _b = {"result": (
                          f"Forked {base_title} v{new_version - 1} → "
                          f"{base_title} v{new_version} ({new_pid}): "
                          f"{copied} file(s) copied from parent "
                          f"{project_id}; v{new_version - 1} stays "
                          f"RELEASED and running."),
                      **_b,
                      "workspace": f"projects/{new_pid}",
                      "note": _workspace_note(new_pid),
                      "version": new_version,
                      "parent_project_id": project_id,
                      "seed_task": seed_tid}
            return _ok(_b)

        if act == "unrelease":
            # Human-commanded demotion RELEASED → DONE (round 3): the only
            # sanctioned demote. The dossier is RETAINED (stamped
            # unreleased_at) for history and future re-release; the
            # workspace becomes writable again. Re-releasing bumps the
            # dossier's revision.
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            proj = store.get_project(project_id)
            if not proj:
                return _err(f"project not found: {project_id}")
            if str(proj.get("status", "")).upper() != "RELEASED":
                return _err(f"project is {proj.get('status')}, not RELEASED "
                            f"— nothing to unrelease.")
            rel = store.get_release(project_id)
            if rel:
                rel["unreleased_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                store.set_release(project_id, rel)
            store.update_project(project_id, status="DONE")
            try:
                store.set_workspace_readonly(project_id, False)
            except Exception:
                pass
            store.log_event(project_id, None, "project_unreleased", {})
            pretty_log("Project Unreleased",
                       f"{proj.get('title')} → DONE (workspace writable; "
                       f"dossier retained)", icon=Icons.STOP, level="WARNING")
            return _ok({"status": "DONE", "action_taken": "unreleased",
                        "note": ("Editable again. Re-release with "
                                 "action=release when re-tested — the "
                                 "dossier revision will bump.")})

        if act == "verify_release":
            # Release health-check (round 3): re-run the rehearsal against
            # today's reality and report drift vs the dossier. Read-only on
            # the record (no status change); restarting a dead service is
            # sanctioned operational healing per the runbook.
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            proj = store.get_project(project_id)
            if not proj:
                return _err(f"project not found: {project_id}")
            if str(proj.get("status", "")).upper() != "RELEASED":
                return _err(f"project is {proj.get('status')}, not RELEASED "
                            f"— verify_release only checks released projects.")
            rel = store.get_release(project_id)
            reh = _release_rehearsal(context, store, project_id)
            drift = []
            _now_names = {s.get("name") for s in (reh.get("services") or [])}
            for s in (rel.get("services") or []):
                if s.get("name") not in _now_names:
                    drift.append(f"service '{s.get('name')}' from the dossier "
                                 f"is no longer registered")
            healthy = bool(reh.get("ok")) and not drift
            store.log_event(project_id, None, "release_health",
                            {"healthy": healthy,
                             "detail": reh.get("detail", "")[:300],
                             "drift": drift[:5]})
            if not healthy:
                pretty_log("Release Health",
                           f"{proj.get('title')}: DEGRADED — "
                           f"{'; '.join(drift) or reh.get('detail', '')[:100]}",
                           icon=Icons.WARN, level="WARNING")
            return _ok({
                "healthy": healthy,
                "rehearsal": reh.get("detail", ""),
                "drift": drift,
                "note": ("Healthy — the dossier still matches reality."
                         if healthy else
                         "Degraded — fix the environment, or unrelease / "
                         "create_version if the artifact itself must change."),
            })

        if act == "task_delete":
            # store.delete_task existed since the beginning but was never
            # exposed — a mistaken duplicate task was permanent (round 3).
            if not task_id:
                return _err("task_id is required for action=task_delete")
            _t = store.get_task(task_id)
            if not _t:
                return _err(f"task not found: {task_id}")
            _rg = _released_guard(store, _t.get("project_id"))
            if _rg:
                return _err(_rg)
            if not store.delete_task(task_id):
                return _err(f"delete failed for task {task_id}")
            store.log_event(_t.get("project_id"), task_id, "task_deleted",
                            {"description": str(_t.get("description"))[:120]})
            return _ok({"deleted": task_id,
                        "description": str(_t.get("description"))[:120]})

        # ---- cross-project features (2026-07-25 final round) ------------

        if act == "search":
            # "Which project touched X / used technique Y" — keyword search
            # over titles/goals/deliverables/manifests/ledgers/work_logs/
            # research across ALL projects. Read-only.
            q = (query or "").strip()
            if not q:
                return _err("query is required for action=search")
            hits = store.search_projects(q, limit=max(1, int(limit or 5)))
            if not hits:
                return _ok({"hits": [], "note": (
                    "No project matched. Try action=list for titles, or "
                    "recall for general memory — facts about untracked work "
                    "often live there.")})
            return _ok({"hits": hits, "query": q})

        if act == "set_dependency":
            # Inter-project dependency edges: this project cannot autoadvance
            # until every project in `depends_on` is DONE or RELEASED.
            # Replaces the whole list; empty list clears. Cycles rejected.
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            deps_in = depends_on or []
            resolved = []
            for ref in deps_in:
                rid2, rerr2 = _resolve_project_ref(store, str(ref), "")
                if rerr2:
                    return _err(rerr2)
                if not rid2:
                    return _err(f"dependency not found: {ref!r}")
                if rid2 == project_id:
                    return _err("a project cannot depend on itself")
                resolved.append(rid2)
            # Cycle guard: walking each candidate's own dependency chain
            # must never reach back to THIS project (depth-capped).
            def _reaches(start: str, target: str, depth: int = 0) -> bool:
                if depth > 10:
                    return False
                meta2 = (store.get_project(start) or {}).get("metadata") or {}
                for d in meta2.get("depends_on_projects") or []:
                    if d == target or _reaches(d, target, depth + 1):
                        return True
                return False
            for rid2 in resolved:
                if _reaches(rid2, project_id):
                    return _err(
                        f"cycle: {rid2} already depends (directly or "
                        f"transitively) on this project")
            def _mut(meta):
                if resolved:
                    meta["depends_on_projects"] = resolved
                else:
                    meta.pop("depends_on_projects", None)
                return meta
            if store._atomic_metadata_update(project_id, _mut) is None:
                return _err(f"project not found: {project_id}")
            store.log_event(project_id, None, "dependencies_set",
                            {"depends_on": resolved})
            return _ok({"depends_on": resolved,
                        "note": ("Autoadvance will wait until every "
                                 "dependency is DONE or RELEASED."
                                 if resolved else "Dependencies cleared.")})

        if act == "clone":
            # Template/clone-without-lineage: start a NEW project from an
            # existing one's shape — any source status (a RELEASED source is
            # read-only-copied). Unlike create_version: no parent/version
            # lineage, fresh identity, title REQUIRED.
            if not project_id:
                return _err("no active project (pass project_id — the "
                            "SOURCE — or switch first)")
            if not title:
                return _err("title is required for action=clone (the NEW "
                            "project's name)")
            src = store.get_project(project_id)
            if not src:
                return _err(f"project not found: {project_id}")
            smeta = (src.get("metadata") or {})
            new_cfg = dict(smeta.get("config") or {})
            for _pk in ("port", "PORT"):
                if str(new_cfg.get(_pk, "")).strip().isdigit():
                    new_cfg[_pk] = str(int(new_cfg[_pk]) + 1)
                    break
            new_pid = store.create_project(
                title, kind=str(src.get("kind") or "GENERAL"),
                goal=(goal or str(src.get("goal") or "")),
                metadata={
                    "design_ledger": smeta.get("design_ledger") or "",
                    "config": new_cfg,
                    "file_manifest": smeta.get("file_manifest") or {},
                    # Active + retired, same rationale as create_version:
                    # a DONE/RELEASED source carries its constraint
                    # knowledge in the retired list.
                    "constraints": _rearm_inherited_constraints(smeta),
                    "cloned_from": project_id,
                })
            import shutil
            src_ws = Path(str(src.get("workspace_dir") or ""))
            dst_ws = Path(str((store.get_project(new_pid) or {})
                              .get("workspace_dir") or ""))
            copied = 0
            if src_ws.is_dir() and str(dst_ws):
                shutil.copytree(
                    src_ws, dst_ws, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(
                        "RELEASE.md", ".services", "__pycache__", "*.pyc",
                        "node_modules"))
                copied = sum(1 for p in dst_ws.rglob("*") if p.is_file())
                # copytree copies mode bits — a RELEASED source is a-w.
                try:
                    store.set_workspace_readonly(new_pid, False)
                except Exception:
                    pass
            seed_tid = store.add_task(
                new_pid, " ".join((description or "").split())
                or f"Adapt the clone of '{src.get('title')}' to its new purpose")
            # Same as the fork path (queue #10): do not copy a claim whose
            # file is not in the source — it protects nothing and propagates
            # a phantom deliverable into the clone.
            try:
                _src_missing = store.missing_deliverables(project_id)
            except Exception:
                _src_missing = set()
            for rel_path in store.list_deliverables(project_id):
                if rel_path in _src_missing:
                    continue
                try:
                    store.register_file_artifact(seed_tid, rel_path)
                except Exception:
                    pass
            store.log_event(new_pid, seed_tid, "project_cloned",
                            {"source": project_id, "files_copied": copied})
            _set_current(context, new_pid)
            store.render_project_map(new_pid)
            pretty_log("Project Cloned",
                       f"'{title}' ({new_pid}) from {src.get('title')} — "
                       f"{copied} file(s); no lineage",
                       icon=Icons.BRAIN_PLAN)
            _b = _briefing(store, new_pid)
            if isinstance(_b, dict):
                _b = {**_b, "workspace": f"projects/{new_pid}",
                      "note": _workspace_note(new_pid),
                      "cloned_from": project_id, "seed_task": seed_tid}
            return _ok(_b)

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
            # Pin the EVENT stamp alongside the sandbox pin above: when the
            # LLM passes an explicit project_id different from the currently
            # switched project, record_* would otherwise stamp the batch's
            # command/research outcomes with the CHAT turn's project. Scoped,
            # so the turn's own stamp is restored after the batch.
            from ..workspace import pinned_event_project as _pin_evt
            with _pin_evt(project_id):
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
            _adv_payload: Dict[str, Any] = {
                "advanced": batch.advanced,
                "count": batch.count,
                "requested": ("all" if max_tasks is None else max_tasks),
                "stop_reason": batch.stop_reason,
                "agent_instruction": _advance_batch_instruction(batch, max_tasks),
            }
            # A failure-shaped stop names the LEDGER's failed tasks, not just
            # this batch's: on an already-FAILED project the batch advanced
            # nothing, so the model otherwise has no ids/reasons to act on
            # (2026-08-01: "project done" was relayed over a FAILED ledger).
            if batch.stop_reason in ("project_failed",
                                     "completed_with_failures", "failed",
                                     "repeated_failures"):
                try:
                    _adv_payload["failed_tasks"] = [
                        {"id": t["id"],
                         "description": (t.get("description") or "")[:100],
                         "failure_reason": (t.get("failure_reason") or "")[:200]}
                        for t in store.list_tasks(project_id)
                        if str(t.get("status", "")).upper() == "FAILED"
                    ][:10]
                    if _adv_payload["failed_tasks"]:
                        _adv_payload["agent_instruction"] += (
                            " To fix a failed task: repair its deliverable, "
                            "then task_update it with status=done and a "
                            "`result` naming the evidence (a stale recorded "
                            "failure is re-checked live), or revise it with "
                            "status=pending to let autoadvance retry it.")
                except Exception:
                    logger.debug("failed-task enrichment skipped",
                                 exc_info=True)
                # ⚠ NOT `_ok`. A failure-shaped stop DECLARES a success
                # otherwise, and a declared ok now settles the corpus label —
                # so 5 live rows reporting FAILED tasks were exempted from
                # the sniffer while `_turn_had_tool_failure` still flagged
                # them, a fresh three-reader split. PARTIAL is the honest
                # status: the batch really did advance `count` tasks, and it
                # really did stop on a failure.
                from .outcome import ToolOutcome as _TO
                import json as _json
                # PARTIAL only when something ACTUALLY advanced — a batch
                # that stopped on a failed project with `advanced == []`
                # landed nothing, and PARTIAL would tell the model "PART OF
                # THIS LANDED" over an empty list.
                _mk = _TO.partial if batch.count else _TO.failed
                return _mk(
                    _json.dumps(_adv_payload, default=str),
                    world_changed=bool(batch.count),
                    reason_code=f"autoadvance_{batch.stop_reason}")
            return _ok(_adv_payload)

        # ---- auto-research ----------------------------------------------

        if act == "research":
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            from ..core.project_research import (
                research_topic, research_project,
            )
            if topic and str(topic).strip():
                rr = await research_topic(context, project_id, str(topic).strip())
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
            "file_system before re-researching the same thing). "
            "`artifact_list` to read back recorded artifacts — the "
            "deliverable file manifest, notes, urls, and stored tool_call "
            "outputs (optionally scope with task_id and/or artifact_kind; "
            "long payloads are truncated to 400 chars). "
            "`cleanup` to remove debris from the project workspace NOW "
            "(stray screenshots, caches, helper scaffolding) — "
            "deliverables, source files and media referenced by the code "
            "are never touched; use when the user asks to tidy up. "
            "`describe_file` to record WHAT a file is/does the moment you "
            "learn it (file_path + description, optional file_role) — it "
            "feeds the per-file map in the briefing and PROJECT_MAP.md, so "
            "future sessions read the right file instead of everything; "
            "call it whenever you create a file or work out an undescribed "
            "one. `file_history` (file_path) to see every recorded action "
            "that touched a file — check it BEFORE re-reading a large file "
            "to answer 'what changed / what was tried'. "
            "`release` (ONLY when the USER explicitly confirms the project "
            "works): promotes a DONE project to RELEASED — pass `directions` "
            "(a usage runbook for the user); the tool tidies the workspace, "
            "COLD-STARTS the services / verifies deliverables (release "
            "rehearsal), and freezes a release dossier + RELEASE.md. A "
            "RELEASED project is IMMUTABLE: to change it use "
            "`create_version` (optional description = the requested "
            "change), which forks a v(n+1) copy — files, ledger, config "
            "(port bumped), file map carried over; task history and the "
            "release dossier stay with the released version, which keeps "
            "running untouched. `unrelease` (user command only) demotes "
            "RELEASED→DONE for in-place fixes (dossier retained; re-release "
            "bumps its revision). `verify_release` health-checks a released "
            "project against its dossier (restarts dead services, probes "
            "ports, reports drift) — use when the user asks whether a "
            "released app still runs. `task_delete` (task_id) removes a "
            "mistaken/duplicate task permanently. "
            "`search` (query) finds WHICH project touched a file / used a "
            "technique — searches every project's deliverables, file map, "
            "ledger, journal, and research; use it before answering 'which "
            "project did X'. `set_dependency` (depends_on=[ids/titles]) "
            "makes this project wait for others: autoadvance holds until "
            "every dependency is DONE/RELEASED (empty list clears; cycles "
            "rejected). `clone` (project_id=SOURCE, title=NEW name, "
            "optional description/goal) starts a FRESH project from an "
            "existing one's files+knowledge WITHOUT lineage — use for "
            "'make another one like X'; for changing a RELEASED project "
            "itself use create_version instead."
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
                "description": {"type": "string", "description": "Task description (task_add/task_update) — OR, for action=describe_file, the one-line 'what this file is/does' text to record (REQUIRED there, e.g. 'Node/Express static host + API on port 8100')."},
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
                "payload": {"type": "string",
                            "description": "artifact_add: the artifact body/path. constraint_retire: WHICH active constraint to retire — its exact text, its 0-based index, or 'all'. Retiring stops the constraint from replaying into future requests/verifier checks (it is kept in metadata.constraints_retired; the user restating it re-arms it). Project DONE retires all constraints automatically. Use when a stored constraint no longer reflects user intent — e.g. a phrasing rule for deliverables that keeps firing on ordinary replies."},
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
                "file_path": {"type": "string",
                              "description": "actions describe_file/file_history/unregister_file: the project-relative file (e.g. 'server.js', 'src/app.py'). unregister_file removes a RENAMED/DELETED file from the deliverable record + file map (then re-register the new name)."},
                "file_role": {"type": "string",
                              "description": "action=describe_file: optional short role tag, e.g. 'entrypoint', 'styles', 'data-layer'."},
                "query": {"type": "string",
                          "description": "action=search: what to find across ALL projects (a filename, technique, library, topic)."},
                "directions": {"type": "string",
                               "description": "action=release (REQUIRED there): the usage runbook for the USER — what the project is, how to use it, features, known limitations. Written in second person, ≥2 sentences. The tool appends the VERIFIED service commands/ports itself; do not guess them here."},
            },
            "required": ["action"],
        },
    },
}
