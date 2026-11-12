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

#: How long a duplicate-create attempt keeps counting toward "you are in a
#: LOOP". A loop is a BURST — the model re-issuing the same create inside
#: one turn or a few minutes. The counter used to be a LIFETIME total on
#: the project record with nothing that ever cleared it, so a project that
#: hit three duplicates once kept the alarmed "STOP. This is retry #N"
#: instruction and its operator WARNING for every duplicate create it ever
#: saw again, months later (§4FP: arm-on-N with no disarm path).
_DUPLICATE_CREATE_LOOP_WINDOW_SECONDS = 600.0


_ACTIONS = {
    # project-level
    "create", "list", "get", "switch", "exit", "update", "delete",
    "archive", "resume", "status", "budget",
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
    # idle-advance opt-in, owner-controlled (§4LQ)
    "autopilot",
    # inbox promotion (suggestion-accepted path)
    "promote_from_context",
    # self-advancing loop
    "autoadvance",
    # §4SU Phase 4: a new plan that keeps the finished work and the research
    "replan",
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
        # §4LP: EVERY refusal kind, not just the constraint hold — the
        # visual-artifact gate, the live re-check, the constraint audit and
        # unknown task ids also came back `ok` with `updated: []`
        # §4SU Phase 4: every gate refusal now lands in `refused` (the legacy
        # per-gate keys were removed)
        _held = any(payload.get(k) for k in ("refused", "missing"))
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
    # KB review 10-30 r2: the same walls as recall — this probe printed the
    # owner's facts into a public channel reply
    try:
        from ..utils.logging import reply_is_public, requester_is_member
        from ..memory.request_text import mostly_non_latin
        if reply_is_public() or requester_is_member() or mostly_non_latin(str(term)):
            search = None
    except Exception:  # noqa: BLE001 — unknown requester: no memory probe
        search = None
    if callable(search):
        try:
            # record_retrievals=False: this is a lookup-miss probe, not a
            # real retrieval — the default bumped prune-survival stats on
            # 3 arbitrary nearby rows (including hits discarded by the
            # 1.15 gate below) on every project-name miss.
            from ..memory.vector import AMBIENT_EXCLUDED_TYPES as _EXCL
            # KB review 10-30: no type scope — a manual chunk or the owner's
            # dog_name row answered "xyzzy-42" as a "related fact"
            raw = await asyncio.to_thread(
                search, str(term), 3, where={"type": {"$nin": list(_EXCL) + ["identity"]}},
                record_retrievals=False)
            for h in raw or []:
                if not isinstance(h, dict) or not h.get("text"):
                    continue
                try:
                    score = float(h.get("score", 9.0))
                except (TypeError, ValueError):
                    continue
                from .memory import RECALL_HIGH_D
                # KB review 10-30: tool_recall's HIGH band (cosine distance) —
                # the old `< 1.15` was an L2-era MEDIUM band that admitted
                # every row (random names measured 0.37–0.43)
                if score < RECALL_HIGH_D:
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
        "projects": [_model_view(p) for p in projects],
        "agent_instruction": (
            "No project is active in this conversation — this is NOT an error. "
            "Pick one from `projects` above and call `manage_projects "
            "action=switch project_id=\"<id>\"`, then retry your read."
        ),
    })


_PROJECT_KEY_PREFIX = "proj::"
#: §4SX B0: the ONE global binding slot of the old design (four scratchpad
#: keys). The focus is a per-conversation row in the project store now
#: (``conversation_focus``); these keys are only deleted, once, at startup
#: (:func:`drop_legacy_binding`, operator decision B4: drop, no mapping).
_LEGACY_BINDING_KEYS = ("__current_project__", "__current_project_conv__",
                        "__current_project_guests__", "__current_project_held__")

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
    protected: set = set()
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
                    and not k.startswith(_PROJECT_KEY_PREFIX)]
        return [
            (k, v) for k, v in data.items()
            if not str(k).startswith(_PROJECT_KEY_PREFIX)
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
    store = _resolve_store(context)
    if not items:
        # §4SX B0 review r1: an emptied project must not keep its older
        # snapshot as the latest, or the next hydrate brings back keys the
        # owner deleted. Record the emptiness — only when the latest snapshot
        # still holds keys (no event per idle alternation).
        if store is None:
            return True
        try:
            last = store.list_events(project_id, limit=1, event_type="scratchpad_snapshot")
            if last and ((last[0].get("payload") or {}).get("keys") or {}):
                store.log_event(project_id, None, "scratchpad_snapshot", {"keys": {}})
        except Exception:  # noqa: BLE001
            logger.debug("empty scratchpad snapshot skipped", exc_info=True)
        return True
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
    it ran at REQUEST START for any conversation that didn't own the bound
    project, one unrelated chat message wiped the whole scratchpad (verified
    in prod). Now each entry carries the scope
    that wrote it, so we clear scope-by-scope and spare:

      * the incoming project's own scope (about to be re-hydrated),
      * the global/free-chat scope (owned by nobody — includes results
        written by detached workers),
      * `proj::` keys and in-flight background-job keys
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
    into the target's scope. Project-namespaced `proj::…` keys and
    background-job keys are preserved.
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


def _focus_writer(context):
    """(store, conversation_key) when this request may write its
    conversation's focus row, else (None, "")."""
    try:
        from ..core.request_scope import current_scope
        sc = current_scope(context)
    except Exception:  # noqa: BLE001
        sc = None
    if sc is None or not sc.persist or not sc.conversation_key:
        return None, ""
    store = _resolve_store(context)
    return (store, sc.conversation_key) if store is not None else (None, "")


def _set_current(context, project_id: Optional[str]):
    """Make ``project_id`` (or nothing) this request's project.

    Snapshots the outgoing project's scratchpad keys and hydrates the
    incoming project's last snapshot — this is what makes "go back and
    forth" feel like resuming work instead of starting over. §4SX B0: an
    owner request with a session id also writes THIS conversation's focus
    row (switch/create/resume set it, exit clears it); no other
    conversation's focus changes. Every other request is request-only.
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
    if project_id is None and prev:
        # what this conversation closed, for `_captured_project_id`'s heal
        # (a same-turn close leaves the scope empty; the turn's writes were
        # under ``prev``)
        try:
            context.last_closed_project = (getattr(context, "conversation_key", None), prev)
        except Exception:  # noqa: BLE001
            pass
    store, conv = _focus_writer(context)
    if store is not None:
        try:
            if project_id:
                store.set_focus(conv, project_id)
            else:
                store.clear_focus(conv)
        except Exception:  # noqa: BLE001
            logger.warning("conversation focus write failed", exc_info=True)
    sp = getattr(context, "scratchpad", None)
    # Keep the default write scope in step even when the project is
    # unchanged (an idempotent switch hydrates nothing).
    _set_scratchpad_scope(sp, project_id)
    if project_id and project_id != prev:
        _hydrate_scratchpad(context, project_id)
    elif project_id is None:
        # Leaving project mode: clear the hydrated keys so free chat
        # starts fresh.
        _hydrate_scratchpad(context, None)


def conversation_fingerprint(messages) -> str:
    """Hash of a conversation's FIRST user message, or "".

    §4SX B0: NOT the conversation's identity any more (that is its
    ``session_id``, :func:`..core.request_scope.conversation_key_for_session`).
    Kept as the key for a client that sends no session id (main-slot
    ownership, late corrections, preview confirms) — such a request never
    writes a focus row.
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


def project_name_hint(context, message: str) -> str:
    """§4SX B0 (B2): one line when THIS message names another project (its id,
    8-char id prefix, or whole-word title of 5+ chars — review r1: "API" or
    "Test" fired on unrelated messages). The model switches
    with one call if it is working on it; nothing is activated by wording."""
    msg = str(message or "").lower()
    store = _resolve_store(context)
    if not msg or store is None:
        return ""
    cur = str(getattr(context, "current_project_id", None) or "")
    try:
        projects = store.list_projects()
    except Exception:  # noqa: BLE001
        return ""
    for p in projects or []:
        pid = str(p.get("id") or "").lower()
        if not pid or pid == cur or str(p.get("status") or "").upper() == "ARCHIVED":
            continue
        title = str(p.get("title") or "").strip().lower()
        if (pid in msg or (len(pid) >= 8 and re.search(r"\b" + re.escape(pid[:8]) + r"\b", msg))
                or (len(title) >= 5 and re.search(r"\b" + re.escape(title) + r"\b", msg))):
            return (f"This message names project {p.get('title')!r} (id {pid}); switch to it "
                    f"(manage_projects action=switch project_id={pid}) if you are working on it — "
                    "reads of it can pass project_id without a switch.")
    return ""


def _iso_date(ts) -> str:
    """A stored epoch as local ISO time (§4QA: list returned the raw float
    1792306732.0 and the model rendered it "~2026-04-16"; it was 2026-10-18)."""
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts)))
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


#: bookkeeping the model never needs to read (§4QA r7 m3: the uncapped
#: brief-hash map was dumped into every list result)
_MODEL_HIDDEN_META = ("research_brief_shas",)


def _model_view(p: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(p)
    meta = out.get("metadata")
    if isinstance(meta, dict) and any(k in meta for k in _MODEL_HIDDEN_META):
        out["metadata"] = {k: v for k, v in meta.items() if k not in _MODEL_HIDDEN_META}
    return out


def _with_dates(p: Dict[str, Any]) -> Dict[str, Any]:
    out = _model_view(p)
    out["created"] = _iso_date(p.get("created_at"))
    out["updated"] = _iso_date(p.get("updated_at"))
    return out






def _dependencies_first(plan, ids: List[str]) -> List[str]:
    """``ids`` reordered so each one comes after the ids it (transitively)
    depends on; otherwise stable. A cycle keeps its given order. A list with
    a repeated id is returned as given (r21: dropping the repeat turned
    ``[t, t]`` into a single-id call, which applies ``description``)."""
    if len(set(ids)) != len(ids):
        return list(ids)
    nodes, pos = plan.tree.nodes, {t: i for i, t in enumerate(ids)}
    out: List[str] = []
    state: Dict[str, int] = {}

    def _under(m: str, top: str) -> bool:
        cur, hops = nodes.get(m), 0
        while cur is not None and getattr(cur, "parent_id", None) and hops < 64:
            if cur.parent_id == top:
                return True
            cur, hops = nodes.get(cur.parent_id), hops + 1
        return False

    def visit(t: str) -> None:
        if state.get(t):
            return
        state[t] = 1
        for d in (nodes[t].depends_on or []) if t in nodes else []:
            # r24: a dependency on a PHASE is met by its roll-up, so the
            # batch members inside that phase close first too
            for m in [d] + [m for m in ids if m != t and _under(m, d)]:
                if m in pos and not state.get(m):
                    visit(m)
        state[t] = 2
        out.append(t)

    for t in ids:
        visit(t)
    return out


def _unmet_dependencies(plan, tid: str) -> List[str]:
    """§4QA-d: the dependencies that hold ``tid`` back from DONE — the
    planner's own rule (``TaskTree.open_dependencies``), so the picker and
    this gate never disagree (r22: a task handed out by one and refused by
    the other looped the autopilot)."""
    try:
        return plan.tree.open_dependencies(tid)
    except AttributeError:
        return []









def forget_project_pointer(context, pid: str) -> None:
    """§4NW / §4SX B0: a deleted/archived project stops being anyone's focus —
    this request's pointer and every conversation's focus row."""
    if getattr(context, "current_project_id", None) == pid:
        _set_current(context, None)
    store = _resolve_store(context)
    if store is not None:
        try:
            store.clear_focus_for_project(pid)
        except Exception:  # noqa: BLE001
            logger.debug("focus clear skipped", exc_info=True)
    _wm = getattr(context, "workspace_model", None)
    if _wm is not None and getattr(_wm, "current_project_id", "") == pid:
        try:
            _wm.current_project_id = ""
        except Exception:  # noqa: BLE001
            pass


def begin_request_focus(context) -> Optional[str]:
    """Load this request's project from its conversation's focus row (§4SX B0).

    Called once at request start, inside the turn lock, after the request's
    scope is installed and BEFORE anything touches the sandbox. Only a
    persisting scope (owner + session id) reads a row; every other scope
    starts with no project (members, probes, unattended requests,
    clients without a session id) unless it was built with one (a leaf).
    The shared scratchpad follows the request's project: when its live
    write scope is another project's, that project's keys are snapshotted
    and the incoming project's last snapshot is hydrated.
    """
    from ..core.request_scope import current_scope
    sc = current_scope(context)
    if sc is None:
        return getattr(context, "current_project_id", None)
    if sc.persist and sc.conversation_key:
        store = _resolve_store(context)
        pid = None
        if store is not None:
            try:
                pid = store.get_focus(sc.conversation_key)
            except Exception:  # noqa: BLE001
                logger.warning("conversation focus read failed", exc_info=True)
        sc.project_id = pid if isinstance(pid, str) and pid else None
    pid = sc.project_id
    # through the context too: a GhostContext's property writes the same
    # scope; any other context object holds it as its own attribute
    try:
        context.current_project_id = pid
        context.conversation_key = sc.conversation_key
    except Exception:  # noqa: BLE001
        pass
    _wm = getattr(context, "workspace_model", None)
    if _wm is not None:
        try:
            _wm.current_project_id = pid or ""
        except Exception:  # noqa: BLE001
            pass
    sp = getattr(context, "scratchpad", None)
    live = getattr(sp, "active_namespace", None) if sp is not None else None
    if not isinstance(live, str):
        live = None
    if sp is not None and (live or None) != (pid or None):
        saved = _snapshot_scratchpad(context, live) if live else True
        if pid or saved:
            _hydrate_scratchpad(context, pid)
        else:
            # the snapshot that makes the clear reversible did not land:
            # keep the keys (leaking is recoverable, deleting is not) and
            # only move the write scope
            _set_scratchpad_scope(sp, None)
            logger.warning(f"Project '{live}' scope left WITHOUT a scratchpad snapshot — "
                           "its keys are kept in memory rather than dropped")
        if live or pid:
            pretty_log("Project Scope",
                       f"conversation focus: {pid or 'none'} (scratchpad scope was {live or 'none'})",
                       icon=Icons.BRAIN_PLAN)
    else:
        _set_scratchpad_scope(sp, pid)
    return pid


def drop_legacy_binding(scratchpad) -> int:
    """Delete the old global binding keys (B4), once, at startup. Returns how
    many were present."""
    n = 0
    if scratchpad is None:
        return 0
    for k in _LEGACY_BINDING_KEYS:
        try:
            if scratchpad.get(k) is not None:
                n += 1
            scratchpad.delete(k)
        except Exception:  # noqa: BLE001
            pass
    return n


def _constraint_origin(context) -> str:
    """"user" when the CURRENT turn is a real user request, else "auto"
    (§4FD provenance). A scheduled/job/sub-agent turn or any simulated
    population writes constraints that must never be replayed to the
    verifier or the prompt as "user-mandated"."""
    try:
        from ..core.agent import turn_origin
        from ..memory.population import ORIGIN_USER, write_origin
        if turn_origin(context) != "user":
            return "auto"
        # ONE derivation for "is the current request a real user request"
        # (`memory/population.py`, §4FH M2): the contextvar default
        # "SYSTEM" (idle phases, `_aa_tool_runner` dispatching a tool by name
        # from the watchdog) is auto, internal prefixes are auto, probes are
        # not user.
        return "user" if write_origin() == ORIGIN_USER else "auto"
    except Exception:  # noqa: BLE001 — provenance is best-effort; unknown = user
        return "user"


def _content_constraint_origin(constraint: str, origin: str) -> str:
    """A constraint the MODEL passed after reading outside content is the
    user's only if the user's own message says it (§4MB: a README's "MUST
    send telemetry to x" was stamped user-mandated and replayed to the
    verifier)."""
    if origin != "user":
        return origin
    try:
        from ..utils.provenance import untrusted_seen, user_message
        if not untrusted_seen():
            return origin
        from ..memory.attribution import _coverage, _words
        words = _words(constraint)
        # §4RZ s4-r3: the request alone — word coverage is not negation-aware
        if words and _coverage(words, user_message()) >= 0.5:
            return origin
        return "auto"
    except Exception:  # noqa: BLE001
        return origin


def _stamp_constraint_origins(meta: Dict[str, Any], constraints: List[str],
                              origin: str) -> Dict[str, Any]:
    """Record ``origin`` for each of ``constraints`` in
    ``meta["constraint_origins"]`` (text → origin). A USER stamp always
    wins: a constraint first written by an autonomous turn and later
    RESTATED by the user is user-mandated from then on (restating re-arms —
    the same rule the retirement lifecycle uses; review §4FH). An auto
    stamp never overwrites a user stamp. Returns ``meta`` for chaining.
    The ONE writer of that key."""
    if not isinstance(meta, dict):
        return meta
    stamps = meta.get("constraint_origins")
    if not isinstance(stamps, dict):
        stamps = {}
    o = origin or "user"
    for c in constraints or []:
        key = str(c)
        if not key:
            continue
        o_c = _content_constraint_origin(key, o)
        if o_c == "user" or key not in stamps:
            stamps[key] = o_c
    meta["constraint_origins"] = stamps
    return meta


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
        "NOT prefix paths with '/workspace/', 'sandbox/', 'projects/"
        f"{project_id}/' or '{project_id}/' — that double-nests the path and 404s (browser file "
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


# §4UI (2026-11-12, req 8646c6c8): the goal-text single-file classifier
# (_SINGLE_FILE_RE) is GONE. It matched "one file" inside "adding a new
# app = one registry entry + one file", collapsed a 7-file modular webOS to
# one task, told the model to STOP a second time, and locked the project
# against every later decompose/replan. The real rule — never split ONE
# file into N tasks — lives in the `subtasks` parameter description.


def _feature_key(desc: str) -> str:
    """A task's identity for dedup — §4UI (heuristics review #4): its declared
    deliverable path when it has one, else its whole normalized text. The
    feature-head key ("App: Notes" vs "App: Calculator" → both `app`,
    "Backend: user API" vs "Backend: payments API") dropped the natural
    webOS / app plan shape; the replay it was for (create-with-subtasks AND
    task_decompose of the same plan) repeats the text verbatim."""
    from ..core.task_deliverable import declared_deliverable
    path = declared_deliverable(desc or "")
    if path:
        return "→" + str(path).strip().lower()
    return _text_key(desc)


def _text_key(desc: str) -> str:
    """The whole normalized text (case, spacing, a trailing period)."""
    return " ".join((desc or "").strip().lower().split()).rstrip(" .")


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
    live = [t.get("description") or "" for t in existing
            if str(t.get("status", "")).upper() not in terminal]
    # against the EXISTING plan: same declared file or same text (the replay);
    # within ONE list only the same text — §4UI r1 m4: a plan that stages one
    # file over several tasks ("markup → index.html", "game logic →
    # index.html") is the model's choice, not a duplicate
    seen = {_feature_key(d) for d in live} | {_text_key(d) for d in live}
    kept = []
    dropped = []
    for d in (subtasks or []):
        if not d or not d.strip():
            continue
        if _feature_key(d) in seen or _text_key(d) in seen:
            dropped.append(d.strip())
            continue
        seen.add(_text_key(d))
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
                               "were skipped (listed below).",
    "count_reached": "Advanced the requested number of tasks; more remain.",
    # §4SR r3: the tool's own wording for an unattended needs_user stop where
    # only holds wait (never a batch stop reason)
    "held_unattended": "Stopped: the task(s) in held_tasks were held because this run is "
                       "unattended; any other open task waits on another task or on the "
                       "owner's own run (see task_list).",
    "hard_cap": "Hit the safety cap mid-run; the rest is still open.",
    "repeated_failures": "Stopped: several tasks failed in a row — likely a "
                         "systemic problem (e.g. a broken foundation).",
    "needs_user": "Stopped: nothing else can run — the task(s) in needs_user_tasks "
                  "wait for the owner (something only they have — a file, a login, an "
                  "answer); autoadvance does not run them.",
    "budget_or_inactive": "Paused: the project's step budget is exhausted "
                          "(the owner raises it with action=budget) or it is not ACTIVE.",
    "deadline": "Stopped before the next task: this request's time is nearly up and "
                "another task would end inside the time reserved for the report. "
                "Tell the user what is done; they can continue in a new message.",
    "in_progress": "Not done: some tasks are still marked in progress (another run may be "
                   "building them, or a run was interrupted) — do NOT report the project as "
                   "complete; tell the user which tasks are open.",
    "open_but_waiting": "Not done: open tasks remain, but none could run in this batch — "
                        "they wait on other tasks (BLOCKED, PAUSED or not yet done), are parents "
                        "whose children are closed, or are follow-ups this run does not pick "
                        "(see task_list: status and depends_on). Do NOT report the project as "
                        "complete; tell the user which tasks are open and what holds them.",
    "busy": "Not run: another unattended step on this project is still running — "
            "try again when it has finished.",
    "out_of_time": "Stopped: the build ran out of time before the task was done — the task "
                   "is open again (READY). Any file it finished is on disk, not verified yet; "
                   "a file it had not finished is written again from the start next time. "
                   "Tell the user.",
    "labs_down": "Stopped: the build needed a lab and the labs host did not answer, or answered that it cannot "
                 "make one right now (its Tor gateway or firewall is not up, its ports are all in use, or it was "
                 "busy) — the task is open "
                 "again (PENDING) and the autopilot starts it again by itself once the labs host answers (after "
                 "a host that took the request and went silent: at least 15 minutes and a free lab slot; after "
                 "a host that was not ready: at least 3 minutes). "
                 "Tell the user; nothing failed in the project itself.",
    "lab_owner_hold": "Stopped: the build could not get a lab, and the task was one the user had put on hold — it "
                      "is back ON HOLD and WAITS FOR THE USER (it does not start again by itself). Tell the user "
                      "what the labs host said.",
    "lab_gave_up": "Stopped: the build could not get a lab after several waits for the labs host — the task now "
                   "WAITS FOR THE USER (it does not start again by itself). Tell the user what the labs host "
                   "said, and that they can free room on it or retry the task.",
    "lab_capacity": "Stopped: the build needed a lab and the labs host has no room (too many labs, or not "
                    "enough memory or disk) — the task is open again (PENDING) and the autopilot starts it "
                    "again by itself once there is room. Tell the user; destroying a lab they no longer "
                    "need makes room.",
    "step_crashed": "Stopped: the build step crashed before it could finish (the task "
                    "is open again, ready to retry) — tell the user what failed.",
    "autopilot_off": "Not run: autopilot is off for this project. Only the owner turns it on "
                     "(action=autopilot enabled=true in their own turn) — a scheduled or background "
                     "run cannot.",
    "autopilot_paused": "Stopped: the unattended safety gates paused this project "
                        "(no progress, a check-in, or its runtime budget) — tell the "
                        "user why; only their own turn resumes it.",
    "failed": "Stopped: a task FAILED — review it before continuing.",
    "model_experiment": "A task may RUN something on a model — the batch went past it (and ran what "
                        "does not wait on it). If its data are model answers they must be real ones, "
                        "which autoadvance cannot produce.",
    "no_store": "Project store unavailable.",
}


def _advance_batch_instruction(batch, max_tasks) -> str:
    blurb = _ADVANCE_STOP_BLURB.get(batch.stop_reason, batch.stop_reason)
    done = sum(1 for a in batch.advanced if (a.get("status") or "") == "DONE")
    failed = sum(1 for a in batch.advanced if (a.get("status") or "") == "FAILED")
    tail = f", {failed} FAILED" if failed else ""
    # §4SR r2 #3: a gate tick is not one of the tasks run
    # r4 #5: one unit per number — D and "held after running" are among the
    # N that ran; gates and "held before running" are extra ticks
    _nr = set(getattr(batch, "not_run", None) or [])
    _held = list(getattr(batch, "held", None) or [])
    _ng = len(getattr(batch, "gated", None) or [])
    _nha = sum(1 for h in _held if h not in _nr)
    _nhb = len(_held) - _nha
    tail += f", {_nha} held for the owner's own run" if _nha else ""
    gated = ((f"; {_ng} more left waiting at a gate" if _ng else "")
             + (f"; {_nhb} more held before running" if _nhb else ""))
    ran = getattr(batch, "ran", None)
    ran = batch.count if ran is None else ran
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
        "project_failed", "autopilot_paused", "autopilot_off", "busy", "step_crashed", "out_of_time",
        "in_progress", "open_but_waiting", "labs_down", "lab_capacity", "lab_gave_up", "lab_owner_hold")
    if batch.stop_reason == "model_experiment":
        # §4SU Phase 5: the batch carries on past an experiment — name IT,
        # not the last task that ran
        _exp = next((a for a in batch.advanced if a.get("task_id")
                     and str(a.get("summary") or "").startswith("model experiment")),
                    next((a for a in reversed(batch.advanced) if a.get("task_id")), {}))
        return (
            # §4SR r8: the experiment is in not_run (so not in `ran`); a
            # duck-typed batch with no `ran` still counts it in `count`
            f"Autonomous batch ran {ran if getattr(batch, 'ran', None) is not None else ran - 1} "
            f"task(s) ({done} DONE{tail}{gated}). {blurb} "
            f"Task {_exp.get('task_id')} — {str(_exp.get('summary') or '')[len('model experiment: '):]} — "
            "do it NOW in this turn. First decide what its data are. If they are a language "
            "model's ANSWERS (prompts put to a model, a control condition, a cross-model run), "
            "run the prompts with probe_model (save_to a new results/… .md file, so the system "
            "records the raw answers), then write the task's results file FROM that transcript and name "
            "the transcript's path in it — the close is refused when nothing the task delivers cites a "
            "kept transcript (§4ST). Quote the answers word for word, never write a model response "
            "yourself. If they are NOT (facts from "
            "published sources, a measurement of something else, a build), do it the normal way — "
            "web_search / read / write the file — and never ask a model for facts it should look "
            "up, and close it with override={\"model_run\": \"<why>\"}. Close it with task_update status=done, then call autoadvance again for the rest. "
            "If an experiment cannot be run as asked (e.g. it needs other models or people), "
            "say so and mark the task with what was really done; do not simulate data. "
            f"{no_reload}"
        )
    if batch.stop_reason == "needs_user":
        # §4SR: req 4609b359 read the generic text below as "do not re-run
        # autoadvance", wondered for 70 s whether "proceed with all tasks"
        # answered the gate, then promised "say continue and I resume from
        # Task 1" — which autoadvance never does for a waiting task
        return (
            f"Autonomous batch ran {ran} task(s) ({done} DONE{tail}{gated}). "
            f"{blurb}{fail_list} STOP now: tell the user EXACTLY which tasks "
            f"completed, which did not, and what each waiting task asks of them. "
            f"A request to run all tasks is NOT the answer to a gate — ask about "
            f"each waiting task by name. When the user answers, do that task "
            f"yourself in that turn and close it with task_update (setting it back "
            f"to pending may gate it again), then call autoadvance for whatever is "
            f"still open. {no_reload}"
        )
    if stopped_early:
        return (
            f"Autonomous batch ran {ran} task(s) ({done} DONE{tail}{gated}). "
            f"{blurb}{fail_list} STOP now: tell the user EXACTLY which tasks "
            f"completed and which did not, and ask how to proceed (retry the "
            f"failed task(s), refine them, or take over). Do NOT keep building the "
            f"project yourself in this turn; in the turn where the user has "
            f"answered (or the fix is in), calling autoadvance again is how to "
            f"continue — it picks up from where this run stopped. {no_reload}"
        )
    # §4UI: one instruction per stop — "hard_cap" said "call autoadvance
    # again" AND "STOP"; a failure stop said "retry those" AND "STOP"
    if batch.stop_reason == "hard_cap":
        return (
            f"Autonomous batch ran {ran} task(s) ({done} DONE{tail}{gated}). "
            f"{blurb}{fail_list} If the user asked for the whole project, call "
            f"autoadvance again now — it picks up where this run stopped; "
            f"otherwise report what was built and wait. {no_reload}"
        )
    return (
        f"Autonomous batch ran {ran} task(s) ({done} DONE{tail}{gated}). "
        f"{blurb}{fail_list} Report to the user what was built"
        + (" and which task(s) failed, and ask whether to retry or fix them"
           if failed_ids else "")
        + f", then wait for their next direction. {no_reload}"
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


def _is_subtask_object(obj) -> bool:
    """A parsed JSON value shaped like a subtask object (r4: one rule for a
    bare object and a list item)."""
    return isinstance(obj, dict) and any(k in obj for k in ("description", "task", "title",
                                                             "kind", "task_kind", "deliverable"))


def _split_declared_subtasks(value):
    """§4SU Phase 2: ``subtasks`` items may be objects
    ``{"description", "kind", "deliverable"}``. Returns (the value with each
    object turned into its description — ``→ deliverable`` appended — and
    {description: kind}). Strings pass through (undeclared). Raises
    ValueError on an unknown kind or an object with no description."""
    from ..core.task_kind import normalize_kind, with_deliverable
    raw = value
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:                                       # r3: one object, not in a list —
            _one = json.loads(raw)                 # r4: only when it IS one ("{name} for
        except ValueError:                         # the {site}" is text)
            _one = None
        if _is_subtask_object(_one):
            raw = [_one]
    if isinstance(raw, str) and re.match(r"\s*\[\s*\{", raw):    # r2: "[draft] write {name}" is text
        try:
            raw = json.loads(raw)
        except ValueError:
            # r1 MINOR: an unreadable list of objects became ONE task of JSON text
            raise ValueError("`subtasks` looks like a JSON list of objects but does not parse — "
                             "pass a list of {description, kind, deliverable} objects")
    if not isinstance(raw, list):
        return value, {}
    out: List[Any] = []
    kinds: Dict[str, str] = {}
    for item in raw:
        if isinstance(item, str) and item.strip().startswith("{") and item.strip().endswith("}"):
            try:                                      # r1 MINOR: one object per string
                _obj = json.loads(item)
                if _is_subtask_object(_obj):
                    item = _obj    # (r2 NIT: other JSON text stays text; r3: a subtask-shaped one
                                   #  is judged as the dict form is — no description is refused)
            except ValueError:
                pass
        if isinstance(item, dict):
            desc = str(item.get("description") or item.get("task") or item.get("title") or "").strip()
            if not desc:
                raise ValueError(f"a subtask object needs a description: {str(item)[:120]}")
            desc = with_deliverable(desc, item.get("deliverable"))
            k = normalize_kind(item.get("kind") or item.get("task_kind"))
            if k:
                kinds.setdefault(desc, k)
            out.append(desc)
        else:
            out.append(item)
    return out, kinds


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
    "result_summary", "kind",
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


def _remove_project_changelog(context, project_id: str) -> bool:
    """§4MN: a hard delete leaves no per-project `CHANGELOG.<id>.md` in the
    workspace root (22 of them, 234 KB, outlived their projects). Only this
    project's file; best-effort."""
    try:
        wm = getattr(context, "workspace_model", None)
        root = getattr(wm, "root", None)
        pid = str(project_id or "").strip()
        if root is None or not pid or not re.fullmatch(r"[A-Za-z0-9_-]+", pid):
            return False
        f = Path(root) / f"CHANGELOG.{pid}.md"
        if f.is_file() and not f.is_symlink():
            f.unlink()
            return True
    except Exception as e:  # noqa: BLE001
        logger.debug("project changelog removal skipped: %s", e)
    return False


def _forget_project_episodes(context, project_id: str, title: str) -> int:
    """§4MJ: hard delete's episode leg — the episodes naming the project (by
    id, and by its title when no live project shares it) are archived and
    deleted, like its graph edges. Best-effort: the delete already
    succeeded."""
    em = getattr(context, "episodic_memory", None)
    if em is None or not hasattr(em, "forget_project"):
        return 0
    try:
        store = getattr(context, "project_store", None)
        live = [p.get("title") for p in (store.list_projects() if store is not None else [])
                if p.get("id") != project_id]
        n = em.forget_project(project_id, title=title, live_titles=live,
                              vector_memory=getattr(context, "memory_system", None))
        if n:
            pretty_log("Project Episodes", f"deleted {project_id}: forgot {n} episode(s)", icon=Icons.MEM_WIPE)
        return n
    except Exception:  # noqa: BLE001
        logger.debug("project episode forget skipped", exc_info=True)
        return 0


def _unlink_project_in_graph(context, project_id: str, title: str):
    """Hard delete's graph leg: remove every edge naming the project, so
    recall stops telling the model the user works on it (live 2cb40b10
    rebuilt a deleted project from `user RESUMES project <id>`). The title
    edges (generic `project HAS_TITLE` and the title-named node) stay while
    another project has that title. Best-effort: the delete already
    succeeded."""
    gm = getattr(context, "graph_memory", None)
    if gm is None or not hasattr(gm, "forget_project"):
        return
    try:
        from ..memory.graph import GraphMemory
        key = GraphMemory.project_title_key
        store = getattr(context, "project_store", None)
        t = key(title)
        shared = bool(t) and store is not None and any(
            key(p.get("title")) == t
            for p in store.list_projects() if p.get("id") != project_id)
        n = gm.forget_project(project_id, title=title, forget_title=not shared)
        if n:
            pretty_log("Project Graph", f"deleted {project_id}: removed {n} "
                       f"graph edge(s)", icon=Icons.MEM_WIPE)
    except Exception:
        logger.debug("graph project unlink skipped", exc_info=True)


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


#: §4LP: every "fork a new version" steer — a probe whose blocked command
#: met this text forked the owner's released Chess Coach v3 into a v4 that the
#: idle advancer then built (and failed). A fork is the owner's change request.
FORK_ONLY_IF_ASKED = ("Fork ONLY when the user asked for a change to this app — never to get "
                      "around this block, to retry a command, or for a test.")


_CONFIRMED_ACTIONS = {
    "release": "RELEASE it — the app becomes immutable; changes then need a new version",
    "unrelease": "UN-RELEASE it — the released app becomes editable again",
    "delete": "PERMANENTLY DELETE it — the record, its tasks and its workspace files",
}


def _probe_turn_now() -> bool:
    try:
        from ..utils.logging import (request_id_context, is_probe_request_id,
                                     request_origin_context, ORIGIN_PROBE)
        return (is_probe_request_id(str(request_id_context.get() or ""))
                or str(request_origin_context.get() or "") == ORIGIN_PROBE)
    except Exception:  # noqa: BLE001
        return False


#: §4MM: the owner's words REFUSING a project ("don't create a project, just do
#: it" — turns 40/43, after two unasked creates). §4UI: the ONE create check
#: left — the ask/build word lists that tried to tell an asked-for project
#: from an unasked one were removed (the model decides).
# r2: only a negated CREATE ("don't create/make/start a project") or
# "without a project" — "I have no project for this, please set one up",
# "don't stop until the project is finished" are asks, not refusals
_NO_PROJECT_RE = re.compile(
    r"(?<!\w)(?:(?:don'?t|do\s+not|never|no\s+need\s+to)\s+(?:\w+\s+)?"
    r"(?:create|make|start|open|set\s+up|spin\s+up)\s+(?:a\s+|any\s+|the\s+|new\s+|another\s+)*projects?"
    r"|without\s+(?:creating\s+|making\s+)?(?:a\s+|any\s+)?(?:new\s+)?projects?"
    r"|(?:μην|μη)\s+(?:φτιάξεις|φτιαξεις|δημιουργήσεις|δημιουργησεις|ανοίξεις|ανοιξεις)\s+(?:\w+\s+)?(?:πρότζεκτ|προτζεκτ|έργο|εργο)"
    r"|χωρίς\s+(?:\w+\s+)?(?:πρότζεκτ|προτζεκτ|έργο|εργο))(?!\w)", re.IGNORECASE)
def _create_not_requested(context) -> str:
    """The refusal text when an INTERACTIVE turn (the owner's or a probe's)
    creates a project the request did not ask for, else "". Background
    runs (scheduled, autonomous, sub-agents) are not gated here."""
    try:
        from ..core.agent import turn_origin
        if turn_origin(context) not in ("user", "probe"):
            return ""
    except Exception:  # noqa: BLE001
        return ""
    ask = str(getattr(context, "last_user_content", "") or "")
    if not ask.strip():
        return ""
    # §4RZ step 4 (s4-r2): a "no project" the user sent mid-turn refuses
    # too — and nothing added mid-turn REVERSES a "no" (s4-r1's newest-wins
    # let "start with the tokenizer" — a build verb, not a word about a
    # project — undo "don't create a project": the §4MM class)
    try:
        from ..utils.provenance import added_user_messages
        _texts = [ask] + [str(t) for t in added_user_messages()]
    except Exception:  # noqa: BLE001
        _texts = [ask]
    if any(_NO_PROJECT_RE.search(_t) for _t in _texts):
        return ("NOT created: the owner said not to create a project. Do the task directly in this "
                "conversation now.")
    # §4UI (heuristics review #5): otherwise the MODEL decides. The verb lists
    # that tried to tell an asked-for project from an unasked one missed "write
    # me a snake game", "I need a webOS with apps" and every Greek ask
    # ("θέλω…", "γράψε μου…", "κάνε…"); a second refusal closed the tools.
    return ""


def _stored_evidence(summary) -> str:
    """§4SU Phase 0 (audit 2): a task's stored result is close evidence only
    when the SYSTEM did not write it — a hold ("… not closed unattended"), a
    gate park ("human gate: …"), a review flag or a model-experiment hold
    closed a visual / constrained task DONE with no result at all."""
    text = str(summary or "").strip()
    if not text:
        return ""
    from ..core.project_advancer import _HELD_MARK, MODEL_EXPERIMENT_PREFIX
    low = text.lower()
    # §4SU Phase 1 r1 M1: a roll-up refusal ("not closed: …") quotes the
    # very postcondition it refused — it is the system's text, not evidence
    if (_HELD_MARK in low or low.startswith("human gate:") or low.startswith("not closed:")
            or low == "flagged for human review"
            or low.startswith(MODEL_EXPERIMENT_PREFIX.lower())):
        return ""
    return text


def _current_request_id() -> str:
    try:
        from ..utils.logging import request_id_context
        return str(request_id_context.get() or "")
    except Exception:  # noqa: BLE001
        return ""


def _record_constraint_override(store, project_id: str, tid: str, why: str, violation: str) -> None:
    """§4UK: the agent went over a constraint with a stated reason — on the record, never a wait."""
    try:
        store.log_event(project_id, tid, "constraint_override",
                        {"req": _current_request_id(), "reason": str(why)[:300],
                         "violation": str(violation or "")[:300]})
        store.append_ledger(project_id, f"WENT OVER a constraint on task {tid}: {str(why)[:200]}")
    except Exception:  # noqa: BLE001
        logger.warning("constraint override not recorded", exc_info=True)


def _owner_turn_now() -> bool:
    """Is this the OWNER's own turn? Not a probe, a scheduled task, a
    sub-agent or any other background run (review M5: a `sched-`/`sub-`
    autoadvance counted as the owner — it set autopilot, skipped the
    unattended hold and hid from the digest). Web, CLI and Slack owner turns
    carry a request id; a call outside any request reads "SYSTEM" and is
    not the owner."""
    try:
        from ..utils.logging import request_id_context
        rid = str(request_id_context.get() or "")
    except Exception:  # noqa: BLE001
        return False
    from .memory import _not_the_user
    return not _not_the_user(rid)


def _budget_view(store, project_id) -> Dict[str, Any]:
    """§4MQ: the unattended budget and the gates' state, for status/budget."""
    from ..core.project_advancer import (DEFAULT_STEPS_CAP, DEFAULT_CHECKPOINT_EVERY,
                                         DEFAULT_UNATTENDED_RUNTIME_CAP_S, NO_PROGRESS_PAUSE_AT)
    m = ((store.get_project(project_id) or {}).get("metadata") or {})
    cap_s = m.get("unattended_runtime_cap_seconds")
    return {
        "autopilot": bool(m.get("autopilot")),
        "steps": f"{int(m.get('steps_used') or 0)}/{int(m.get('steps_cap') or DEFAULT_STEPS_CAP)}",
        "unattended_hours": f"{float(m.get('unattended_runtime_seconds') or 0) / 3600:.1f}/"
                         f"{float(cap_s if cap_s is not None else DEFAULT_UNATTENDED_RUNTIME_CAP_S) / 3600:.1f}",
        "checkpoint_every": int(m.get("checkpoint_every") if m.get("checkpoint_every") is not None
                                else DEFAULT_CHECKPOINT_EVERY),
        "steps_since_checkpoint": int(m.get("steps_since_checkpoint") or 0),
        "no_progress_streak": f"{int(m.get('no_progress_streak') or 0)}/{NO_PROGRESS_PAUSE_AT}",
        "paused": m.get("autopilot_paused") or None,
    }


def _autopilot_on(store, project_id) -> bool:
    try:
        return bool(((store.get_project(project_id) or {}).get("metadata") or {}).get("autopilot"))
    except Exception:  # noqa: BLE001
        return False


def _deleted_note(store, project_id) -> str:
    """§4PE (P7): ": it was DELETED on <date>" when the id sits in the
    tombstone table — ids keep arriving from recall across turns, and "no
    project matches" alone invited the next lookup. "" otherwise."""
    try:
        from ..memory.projects import _canon_id as _c
        pid = _c(project_id) if project_id else ""
        if not pid or not hasattr(store, "_connect"):
            return ""
        with store._lock, store._connect() as conn:
            row = conn.execute("SELECT deleted_at FROM deleted_projects WHERE id = ?", (pid,)).fetchone()
        if not row:
            return ""
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(row[0])))
        return f": it was DELETED on {when} and will not come back (do not look it up again)"
    except Exception:  # noqa: BLE001
        return ""


def _event_payload_preview(etype, payload, cap: int = 400):
    """§4PE (P4): the briefing's five newest events went into the result RAW
    — a `scratchpad_snapshot` carried the whole scratchpad (60 KB into one
    `get`). A snapshot becomes a count; any other payload is bounded."""
    try:
        if etype == "scratchpad_snapshot" and isinstance(payload, dict):
            keys = payload.get("keys") if isinstance(payload.get("keys"), dict) else payload
            return {"keys": len(keys) if hasattr(keys, "__len__") else 0, "note": "scratchpad snapshot (not shown)"}
        s = json.dumps(payload, ensure_ascii=False, default=str)
        if len(s) <= cap:
            return payload
        return {"preview": s[:cap] + "…", "chars": len(s)}
    except Exception:  # noqa: BLE001
        return {"preview": str(payload)[:cap]}


def _lifecycle_target(context, store, project_id, title):
    """``(rid, error)`` — the ONE project a release/unrelease/delete will act
    on, resolved before the gate so the preview, the confirmation and the
    action all name the same project (review C1/M1: a title that matched
    nothing passed the gate, and `unrelease` — which ignores `title` — then
    un-released the CURRENT project). ``(None, None)``: nothing given and
    nothing current — the action reports that itself."""
    from ..memory.projects import _canon_id as _canon_pid
    pid = _canon_pid(project_id) if project_id else None
    if pid or str(title or "").strip():
        rid, rerr = _resolve_project_ref(store, pid, title or "")
        if rerr:
            return None, rerr
        if not rid:
            return None, (f"project not found: {project_id or title!r}{_deleted_note(store, project_id)} — "
                          f"NOTHING was changed. Use action=list to see the real ids.")
        return rid, None
    cur = getattr(context, "current_project_id", None)
    if cur and not store.get_project(_canon_pid(cur)):
        return None, (f"the current project {cur} no longer exists — NOTHING was changed. "
                      f"Name the project (project_id or title).")       # review N9
    return (_canon_pid(cur) if cur else None), None


def _lifecycle_would_refuse(store, act, rid, kwargs, context=None) -> bool:
    """The action will refuse on its own and change nothing — no preview to
    confirm (review m4: a preview for an ACTIVE project's release ended in
    the owner's "yes" hitting "not DONE"). §4UI: a delete that an unreleased
    human gate will refuse — the preview's token was spent before the gate
    refused, so the owner's next "delete <task>" got a fresh preview and the
    "yes" after it named no task: a loop."""
    st = str((store.get_project(rid) or {}).get("status") or "").upper()
    if act == "unrelease":
        return st != "RELEASED"
    if act == "release":
        return (st != "DONE"
                or len(" ".join(str(kwargs.get("directions") or "").split())) < 30)
    return False


async def _host_project_labs(context, project_id: str) -> Optional[list]:
    """The labs host's own list of the project's labs (lifetime + lineage), for the delete preview and its
    digest; None when no host is configured or it does not answer (the preview then says so)."""
    try:
        from .labs import _call, labs_configured
        if not labs_configured(context):
            return None
        owner = f"project:{project_id}"
        st, body = await _call(context, "GET", "/v1/labs", origin=_teardown_origin(context),
                               params={"owner": owner})
        if st != 200 or not isinstance(body, dict):
            return None
        mine = [x for x in body.get("labs") or [] if isinstance(x, dict) and str(x.get("owner") or "") == owner]
    except Exception:  # noqa: BLE001
        return None
    # a lab the host has and the local record lacks is recorded now, so a delete that later cannot reach the
    # host refuses on it instead of reading "no lab on record" (review r3 #1)
    try:
        from ..memory.lab_inventory import inventory_for
        inv = inventory_for(context)
        known = {str(r.get("lab_id")) for r in (inv.rows(owner, live_only=True) if inv is not None else [])}
        hid = ""
        if inv is not None and any(str(h.get("lab_id") or "") not in known for h in mine):
            from .labs import _host_id
            hid = await _host_id(context, _teardown_origin(context))
        for h in mine:
            lid = str(h.get("lab_id") or "")
            if inv is not None and lid and lid not in known:
                inv.record_create(lid, owner=owner, origin=str(h.get("origin") or ""),
                                  template=str(h.get("template") or ""), label="recorded at a delete preview",
                                  host_id=hid)
    except Exception:  # noqa: BLE001
        pass
    return mine


def _lifecycle_confirmation(store, act, rid, token, context=None, host_labs=None):
    """None when ``act`` may run now on ``rid``; otherwise a refusal.

    §4UK (operator 11-12: "i want all confirmations, everywhere to be removed … that's why everything is
    sandboxed"): no preview, no token, no "ask the user first". Only the walls on WHO may act stay: a background,
    scheduled or probe turn cannot release, unrelease or delete an owner project (a probe its own throwaway one,
    yes), and neither can a request that read outside content (a page could say "delete project X")."""
    from .memory import _not_the_user
    from ..utils.logging import request_id_context
    proj = store.get_project(rid) or {}
    if _probe_turn_now() and (proj.get("metadata") or {}).get("probe_created"):
        return None                       # a probe's own throwaway project
    _rid_now = str(request_id_context.get() or "")
    if _rid_now and _not_the_user(_rid_now):
        return _err(f"NOT done: only the user's own conversation can {act} a project — a background, scheduled "
                    f"or test turn cannot. Nothing changed.")
    try:
        from ..utils.provenance import untrusted_seen
        _src = untrusted_seen(_rid_now) if _rid_now else []
    except Exception:  # noqa: BLE001
        _src = []
    if _src:
        return _err(f"NOT done: this request read outside content ({', '.join(_src[:3])}) — a {act} runs only "
                    f"in a request that did not, so a page cannot order it. Nothing changed.")
    return None


#: metadata keys only the system writes — the model's `metadata=` cannot set
#: them (review M2: a probe set `probe_created` on an owner project, then
#: deleted it with no confirmation). §4QA r8: the brief hashes decide what
#: the brief writer may overwrite — a forged hash would make it "ours"
_SYSTEM_METADATA_KEYS = ("probe_created", "autopilot", "research_brief_shas",
                         "quote_check_tasks",
                         "lab_waits", "lab_waits_rounds")      # §4SX D4 r2 #13: the labs-host holds      # §4QA-e r34: the quote check's own count
#: §4MQ: the budget and the unattended gates' state — written by the system or
#: by the owner's `action=budget`, never by the model's `metadata=` (a
#: background turn could raise its own cap)
from ..core.project_advancer import BUDGET_METADATA_KEYS as _BUDGET_KEYS  # noqa: E402


async def tool_manage_projects_for_model(context, **kwargs):
    """The MODEL-facing manage_projects (the registry's). Release, unrelease and hard delete run at once in the
    user's own conversation (§4UK: no confirmations); the walls in `_lifecycle_confirmation` decide who may."""
    act = str(kwargs.get("action") or "").strip().lower()
    token = kwargs.pop("confirm_token", "") or ""      # §4UK: accepted and ignored (no previews)
    _md = kwargs.get("metadata")
    if isinstance(_md, str):
        try:
            _md = json.loads(_md.strip() or "null")
        except ValueError:
            _md = kwargs.get("metadata")       # the action reports the bad JSON
    if isinstance(_md, dict):
        if "autopilot" in _md:
            # never a silent no-op (review N2: `{"autopilot": false}` answered
            # "updated" and the idle loop kept building)
            return _err("autopilot is not metadata: use manage_projects action=autopilot "
                        "enabled=true|false, on the owner's request — nothing changed")
        _bk = sorted(k for k in _md if k in _BUDGET_KEYS)
        if _bk and act != "budget":
            return _err(f"{', '.join(_bk)} {'is' if len(_bk) == 1 else 'are'} budget, not metadata: use "
                        "manage_projects action=budget on the owner's request — nothing changed")
        if act != "budget":
            kwargs["metadata"] = {k: v for k, v in _md.items() if k not in _SYSTEM_METADATA_KEYS}
        else:
            kwargs["metadata"] = _md
    if act in _CONFIRMED_ACTIONS:
        store = getattr(context, "project_store", None)
        if store is not None:
            rid, rerr = _lifecycle_target(context, store, kwargs.get("project_id"), kwargs.get("title") or "")
            if rerr:
                return _err(rerr)
            if rid and not _lifecycle_would_refuse(store, act, rid, kwargs, context):
                _hl = await _host_project_labs(context, rid) if act == "delete" else None
                _gate = _lifecycle_confirmation(store, act, rid, token, context, _hl)
                if _gate is not None:
                    return _gate
            if rid:
                kwargs["project_id"] = rid          # act on exactly the project confirmed
                kwargs.pop("title", None)
    return await tool_manage_projects(context, **kwargs)


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
                f"it, follow the RELEASE DIRECTIONS in the briefing. {FORK_ONLY_IF_ASKED}"
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
    ``run status`` when it matters."""
    entries = _project_service_entries(context, project_id)
    if not entries:
        return []
    # The live generation in the supervisor's own form, `<id>:<StartedAt>`
    # (§4BW), from the CACHED attrs — no docker exec per turn. Comparing the
    # bare id against that stamp marked every running service STALE on every
    # turn, telling the model to restart working (released) apps (§4LR M1).
    _cid, _started = "", ""
    try:
        _c = getattr(getattr(context, "sandbox_manager", None), "container", None)
        _cid = str(getattr(_c, "id", None) or "")
        _started = str(((getattr(_c, "attrs", None) or {}).get("State") or {}).get("StartedAt") or "")
    except Exception:  # noqa: BLE001 — stubs
        _cid, _started = "", ""
    out = []
    for e in entries:
        stamped = str(e.get("container_id") or "")
        if not (_cid and stamped):
            stale = False
        elif ":" not in stamped:
            stale = True            # legacy id-only stamp: never survives a restart (services.py)
        elif _started:
            stale = stamped != f"{_cid}:{_started}"
        else:
            stale = stamped.split(":", 1)[0] != _cid
        out.append({
            "name": e.get("name") or e.get("key"),
            "key": e.get("key"),                 # §4SX D5: the briefing lists a run-backed one under RUNNING
            "port": e.get("port"),
            "command": str(e.get("command") or "")[:80],
            "started_at": e.get("started_at"),
            "stale": stale,
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


# ------------------------------------------------------------------ §4SX Phase D2: the project's inventory

def _labs_host_configured(context) -> bool:
    try:
        from .labs import labs_configured
        return bool(labs_configured(context))
    except Exception:  # noqa: BLE001
        return False


def _project_inventory(context, project_id: str, host_labs: Optional[list] = None) -> Dict[str, list]:
    """What the project owns that keeps running or holds a machine — LOCAL records (no docker exec, no
    labs-host call; safe for status and the briefing): ``services`` (registry keys), ``runs`` (live run rows),
    ``labs`` (live lab-inventory rows), ``jobs`` (running sandbox jobs not started by `run`). ``host_labs``
    (the delete preview's own host read) adds what only the host knows: lifetime/lineage and labs with no
    local row (review r2 #3)."""
    owner = f"project:{project_id}"
    out: Dict[str, list] = {"services": [], "runs": [], "labs": [], "jobs": [], "schedules": []}
    try:
        from .tasks import project_schedules
        out["schedules"] = project_schedules(project_id)          # §4SX D4
    except Exception:  # noqa: BLE001
        pass
    try:
        out["services"] = [str(e.get("key") or e.get("name") or "") for e in _project_service_entries(context, project_id)]
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..memory.run_registry import registry_for
        reg = registry_for(context)
        out["runs"] = reg.rows(owner, live_only=True) if reg is not None else []
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..memory.lab_inventory import inventory_for
        inv = inventory_for(context)
        out["labs"] = [dict(r) for r in (inv.rows(owner, live_only=True) if inv is not None else [])]
    except Exception:  # noqa: BLE001
        pass
    if host_labs:
        by_id = {str(r.get("lab_id")): r for r in out["labs"]}
        for h in host_labs:
            lid = str(h.get("lab_id") or "")
            if not lid:
                continue
            row = by_id.get(lid)
            if row is None:
                row = {"lab_id": lid, "template": h.get("template") or ""}
                out["labs"].append(row)
                by_id[lid] = row
            row["lifetime"] = str(h.get("lifetime") or "")
            row["lineage"] = str(h.get("lineage") or "")
    try:
        from ..sandbox.jobs import get_job_supervisor
        sup = get_job_supervisor(getattr(context, "sandbox_manager", None))
        _run_jobs = {str(r.get("backend_ref") or "") for r in out["runs"] if r.get("backend") == "jobs"}
        if sup is not None:
            out["jobs"] = [e for e in sup.list_entries()
                           if str(e.get("project_id") or "") == str(project_id)
                           and e.get("state") == "running" and not e.get("started_via")
                           # §4SX D5 review: a promoted execute job is a run row too — listed once, under runs
                           and str(e.get("id") or "") not in _run_jobs]
    except Exception:  # noqa: BLE001
        pass
    return out


def _inventory_ids(inv: Dict[str, list]) -> List[str]:
    """The confirm digest (design r1 M8): one id per thing a confirmed delete will stop or destroy."""
    ids = [f"svc:{k}" for k in inv.get("services") or []]
    ids += [f"run:{r.get('run_id')}" for r in inv.get("runs") or []]
    ids += [f"app:{r.get('run_id')}" for r in inv.get("runs") or [] if r.get("lifetime") == "app"]
    ids += [f"lab:{r.get('lab_id')}" for r in inv.get("labs") or []]
    ids += [f"kept:{r.get('lab_id')}" for r in inv.get("labs") or [] if r.get("lifetime") == "kept"]
    ids += [f"job:{e.get('id')}" for e in inv.get("jobs") or []]
    ids += [f"sched:{e.get('id')}" for e in inv.get("schedules") or []]
    return sorted(ids)


def _inventory_text(inv: Dict[str, list], *, hard: Optional[bool]) -> str:
    """One line of names; ``hard`` True/False adds what a delete/archive does to labs, None (status) nothing."""
    bits = []
    if inv.get("runs"):
        bits.append("runs " + ", ".join(f"{r['name']} ({r['lifetime']} @ {r['place']})" for r in inv["runs"][:8]))
    if inv.get("services"):
        bits.append("services " + ", ".join(inv["services"][:8]))
    if inv.get("jobs"):
        bits.append("background jobs " + ", ".join(str(e.get("id")) for e in inv["jobs"][:8]))
    if inv.get("schedules"):
        bits.append("scheduled tasks " + ", ".join(f"{e['name']} ({e['id']})" for e in inv["schedules"][:8])
                    + ("" if hard is None else " — removed" if hard else " — they keep firing"))
    if inv.get("labs"):
        labs = ", ".join(f"{r['lab_id']} ({r.get('template') or '?'}"
                         + (f", KEPT, lineage {r.get('lineage') or '?'}" if r.get("lifetime") == "kept" else "")
                         + ")" for r in inv["labs"][:8])
        bits.append(f"labs {labs}" + ("" if hard is None else
                                      " — kept ones too, with their data unless another version shares it"
                                      if hard else " — disposable ones; kept ones stay up"))
    return "; ".join(bits)


def _teardown_origin(context) -> str:
    """The labs/runs origin of the caller in a request scope (owner / probe / autopilot)."""
    try:
        from .labs import _caller
        origin, _owner, refusal = _caller(context)
        return "autopilot" if refusal or not origin else origin
    except Exception:  # noqa: BLE001
        return "autopilot"


def _teardown_refusal(context, proj: Dict[str, Any], act: str) -> str:
    """review r2 #2: deleting/archiving stops what the project runs — the owner's call. A probe may only touch a
    project it created; unattended/leaf work never retires a project. No request scope (internal callers, unit
    use) keeps the old behaviour."""
    try:
        from ..core.request_scope import current_scope
        sc = current_scope(context)
    except Exception:  # noqa: BLE001
        sc = None
    origin = str(getattr(sc, "origin", "") or "") if sc is not None else ""
    if origin == "probe" and not (proj.get("metadata") or {}).get("probe_created"):
        return f"a probe may {act} only a project it created — nothing was changed"
    if origin in ("unattended", "leaf", "member"):
        return f"{act} is the owner's decision — not available to {origin} work; nothing was changed"
    return ""


async def _teardown_project(context, project_id: str, *, hard: bool, origin: str) -> Dict[str, list]:
    """Stop / destroy what the project owns BEFORE its row goes (design D2 + review r1): runs (archive keeps
    `app` runs — production, H2), services (as before), the project's running sandbox jobs, and its labs
    (archive: disposable only; delete: kept too, dropping the data only when no other kept lab of the lineage
    is left, H3). ``origin`` is passed EXPLICITLY (H1: the console route has no request scope). Each step
    reports; a failure never stops the next one. ``live_left`` = things still running that this could not
    stop — a delete refuses while it is non-empty (M5: the sandbox being DOWN is not "left running")."""
    from .outcome import OutcomeStatus
    owner = f"project:{project_id}"
    rep: Dict[str, list] = {"stopped": [], "kept_up": [], "failed": [], "live_left": [], "notes": []}
    # 1. runs
    try:
        from ..memory.run_registry import registry_for
        from . import runs as _runs
        reg = registry_for(context)
        try:
            if reg is None and getattr(context, "memory_dir", None):
                raise RuntimeError("cannot be opened")               # review r4 m3: unopened is not empty
            live_runs = reg.rows(owner, live_only=True, strict=True) if reg is not None else []
        except Exception as e:  # noqa: BLE001 — unread is never "none" (review r3 m2)
            live_runs = []
            rep["failed"].append(f"runs: the run registry could not be read ({type(e).__name__})")
            rep["live_left"].append("runs (the run registry could not be read)")
        for row in live_runs:
            label = f"run {row['name']} ({row['lifetime']} @ {row['place']})"
            if not hard and row["lifetime"] == "app":
                rep["kept_up"].append(label)
                continue
            try:
                out = await _runs._stop(context, reg, row, origin)
            except Exception as e:  # noqa: BLE001
                rep["failed"].append(f"{label}: {type(e).__name__}: {e}")
                rep["live_left"].append(label)
                continue
            if out.status is OutcomeStatus.OK:
                rep["stopped"].append(label)
            elif getattr(out, "reason_code", None) == "run_replaced":
                # another start took the name; that service is the services step's (review r3 m5)
                rep["notes"].append(f"{label}: replaced by a start outside `run` — the services step stops it")
            elif getattr(out, "reason_code", None) == "sandbox_unavailable":
                # the container is down: nothing of it runs (M5) — record that instead of refusing
                reg.set_state(row["run_id"], "lost", "the sandbox was not running at project teardown")
                rep["notes"].append(f"{label}: the sandbox is not running — recorded as lost")
            else:
                rep["failed"].append(f"{label}: {str(out)[:200]}")
                rep["live_left"].append(label)
    except Exception as e:  # noqa: BLE001
        rep["notes"].append(f"runs: {type(e).__name__}: {e}")
    # 2. services (the old path, incl. the legacy workdir match)
    try:
        n = await asyncio.to_thread(_stop_project_services, context, project_id, hard)
        if n:
            rep["stopped"].append(f"{n} service(s)")
        from ..sandbox.services import get_service_supervisor
        sup = get_service_supervisor(getattr(context, "sandbox_manager", None))
        for e in await asyncio.to_thread(_project_service_entries, context, project_id):
            # a row still there after the stop = its process survived (the supervisor puts it back)
            try:
                alive = sup is not None and await asyncio.to_thread(sup._entry_state, e) is not False
            except Exception:  # noqa: BLE001
                alive = True                                   # could not tell: never report it gone
            if alive:
                rep["live_left"].append(f"service {e.get('key') or e.get('name')}")
    except Exception as e:  # noqa: BLE001
        rep["notes"].append(f"services: {type(e).__name__}: {e}")
    # 3. the project's running background jobs (`execute` promotions)
    try:
        from ..sandbox.jobs import get_job_supervisor
        jsup = get_job_supervisor(getattr(context, "sandbox_manager", None))
        for e in (await asyncio.to_thread(jsup.list_entries) if jsup is not None else []):
            if str(e.get("project_id") or "") != str(project_id) or e.get("state") != "running":
                continue
            try:
                ok, msg = await asyncio.to_thread(jsup.cancel, e["id"])
            except Exception as ex:  # noqa: BLE001 — one job's error never skips the next (review r2 #6)
                ok, msg = False, f"{type(ex).__name__}: {ex}"
            if ok:
                rep["stopped"].append(f"job {e['id']}")
                continue
            try:
                now = await asyncio.to_thread(jsup.get, e["id"])
            except Exception:  # noqa: BLE001
                now = e
            if now is None or (now.get("state") or "") != "running":
                # it ended or was marked lost (container recreated) before the cancel — nothing runs (r2 #4)
                rep["notes"].append(f"job {e['id']}: {str(msg)[:160]}")
            else:
                rep["failed"].append(f"job {e['id']}: {msg}")
                rep["live_left"].append(f"job {e['id']}")
    except Exception as e:  # noqa: BLE001
        rep["notes"].append(f"jobs: {type(e).__name__}: {e}")
    # 4. labs
    try:
        await _teardown_labs(context, project_id, owner, hard=hard, origin=origin, rep=rep)
    except Exception as e:  # noqa: BLE001 — an unexpected error leaves the labs counted LIVE (review r2 #6)
        rep["failed"].append(f"labs: {type(e).__name__}: {e}")
        try:
            from ..memory.lab_inventory import inventory_for
            inv = inventory_for(context)
            if inv is None:
                raise RuntimeError("no lab inventory")
            for r in inv.rows(owner, live_only=True, strict=True):
                rep["live_left"].append(f"lab {r['lab_id']}")
        except Exception:  # noqa: BLE001
            rep["live_left"].append("labs (could not be checked)")
    # 5. scheduled tasks the project owns (§4SX D4): a delete removes them; an archive keeps them (resume undoes
    # an archive, and a removed schedule would not come back)
    try:
        from .tasks import project_schedules, remove_project_schedule
        try:
            scheds = project_schedules(project_id, strict=True)
        except Exception as e:  # noqa: BLE001 — unread is never "none"
            scheds = []
            if hard:
                rep["failed"].append(f"scheduled tasks: {type(e).__name__}: {e}")
                rep["live_left"].append("scheduled tasks (the task store could not be read)")
        for t in scheds:
            label = f"scheduled task {t['name']} ({t['id']})"
            if not hard or rep["live_left"]:
                # review r2 #4: a delete that will be refused (something still runs) keeps the project — and its
                # schedules: a removed schedule does not come back
                rep["kept_up" if not hard else "notes"].append(
                    label if not hard else f"{label}: kept — the delete is refused while something still runs")
                continue
            elif await asyncio.to_thread(remove_project_schedule, getattr(context, "scheduler", None), t["id"]):
                rep["stopped"].append(label)
            else:
                rep["failed"].append(f"{label}: the task store could not be updated")
                rep["live_left"].append(label)
    except Exception as e:  # noqa: BLE001
        rep["notes"].append(f"scheduled tasks: {type(e).__name__}: {e}")
    if hard:
        # §4SX D3: an `app` run ends only when its kept lab is gone — re-probe them now the labs step ran
        try:
            from ..memory.run_registry import registry_for
            from . import runs as _runs
            reg = registry_for(context)
            for row in (reg.rows(owner, live_only=True) if reg is not None else []):
                if row["backend"] == "labs_app":
                    await _runs._probed(context, reg, row, origin)
        except Exception as e:  # noqa: BLE001
            rep["notes"].append(f"app runs: {type(e).__name__}: {e}")
    return rep


async def _teardown_labs(context, project_id, owner, *, hard, origin, rep) -> None:
    try:
        from ..memory.lab_inventory import inventory_for
        from .labs import LabsNoAnswer, LabsUnreachable, _call, labs_configured
    except Exception as e:  # noqa: BLE001
        rep["notes"].append(f"labs: {type(e).__name__}: {e}")
        return
    inv = inventory_for(context)
    if inv is None and getattr(context, "memory_dir", None):
        raise RuntimeError("the lab inventory cannot be opened")     # review r4 m3 (caller counts it live)
    local = inv.rows(owner, live_only=True, strict=True) if inv is not None else []
    if not labs_configured(context):
        for r in local:
            rep["live_left"].append(f"lab {r['lab_id']} (no labs host is configured to destroy it)")
        return
    try:
        st, body = await _call(context, "GET", "/v1/labs", origin=origin, params={"owner": owner})
        if st != 200:
            raise LabsNoAnswer(f"HTTP {st}")
        mine = [x for x in (body if isinstance(body, dict) else {}).get("labs") or []
                if isinstance(x, dict) and str(x.get("owner") or "") == owner]
    except (LabsUnreachable, LabsNoAnswer) as e:
        for r in local:
            rep["live_left"].append(f"lab {r['lab_id']} (the labs host did not answer: {e})")
        if not local:
            rep["notes"].append(f"labs host did not answer ({e}); no lab of this project is on record")
        return
    seen = set()
    _destroyed: set = set()                # labs this pass destroyed: they no longer hold a lineage (review r2 #1)
    try:
        from .labs import _host_id
        host_now = await _host_id(context, origin)
    except Exception:  # noqa: BLE001
        host_now = ""
    # §4SX D3 code r1 #8: the version holding the data (app mounted) goes LAST, so the drop_data it gets as the
    # lineage's last kept lab lands on the version that has the data
    mine = sorted(mine, key=lambda x: (str(x.get("lifetime") or "") == "kept", bool(x.get("app_mounted"))))
    for lab in mine:
        lid = str(lab.get("lab_id") or "")
        seen.add(lid)
        kept = str(lab.get("lifetime") or "") == "kept"
        if kept and not hard:
            rep["kept_up"].append(f"kept lab {lid}")
            continue
        params = {}
        if kept:
            # §4UD F13: the ONE decision (tools/labs.kept_destroy_params) — the host keeps the data while another
            # kept version of the app exists and says so in its answer; this copy of the list check is gone
            from .labs import kept_destroy_params
            params, _dnote = await kept_destroy_params(context, origin, lid, str(lab.get("lineage") or ""),
                                                       gone=_destroyed)
            if _dnote:
                rep["notes"].append(f"kept lab {lid}:{_dnote}")
        try:
            st, b = await _call(context, "DELETE", f"/v1/labs/{lid}", origin=origin, params=params, timeout=120)
        except Exception as e:  # noqa: BLE001
            rep["failed"].append(f"lab {lid}: {type(e).__name__}: {e}")
            rep["live_left"].append(f"lab {lid}")
            continue
        b = b if isinstance(b, dict) else {}
        if st == 200 and b.get("leftovers"):
            # the host destroyed it but could not remove everything (review r3 m4)
            rep["failed"].append(f"lab {lid}: destroyed with leftovers {str(b.get('leftovers'))[:160]} — the "
                                 "operator must clear them on the labs host")
            rep["live_left"].append(f"lab {lid} (leftovers)")
            if inv is not None:
                inv.mark_leftovers(lid, ", ".join(map(str, b.get("leftovers"))))   # r5: lab list keeps it
            continue
        if st == 200 or (st == 404 and b.get("error") == "unknown_lab"):
            if params.get("drop_data"):
                # the host decides: it drops only volumes it backed up (review r2 #8) — say what it did
                gone = b.get("data_volumes_removed") or []
                what = (f" with its data ({len(gone)} volume(s), final backup taken)" if gone
                        else f"; its data was KEPT ({b.get('note') or 'the host removed no data volume'})")
            else:
                what = ""
            rep["stopped"].append(f"{'kept ' if kept else ''}lab {lid} destroyed{what}")
            _destroyed.add(lid)
            if inv is not None:
                inv.record_destroy(lid, by=f"project teardown {project_id}")
        else:
            rep["failed"].append(f"lab {lid}: HTTP {st} {b.get('error') or ''}".rstrip())
            rep["live_left"].append(f"lab {lid}")
    for r in local:
        lid = str(r["lab_id"])
        if lid in seen:
            continue
        if r.get("host_id") and r["host_id"] != host_now:
            # made on ANOTHER labs host (or this host's identity is unknown right now): this one cannot
            # destroy it, and "not listed" is not "gone" (review r3 m1, r4 #2)
            rep["live_left"].append(f"lab {lid} (made on labs host {r['host_id']}; the configured host is "
                                    f"{host_now or 'not answering /v1/health'})")
            continue
        # not listed: destroyed earlier, or destroyed with leftovers the host still holds — ask it (review r3 #1)
        try:
            st, b = await _call(context, "DELETE", f"/v1/labs/{lid}", origin=origin, timeout=60)
        except Exception as e:  # noqa: BLE001
            rep["failed"].append(f"lab {lid}: {type(e).__name__}: {e}")
            rep["live_left"].append(f"lab {lid}")
            continue
        b = b if isinstance(b, dict) else {}
        if st == 200 and b.get("leftovers"):
            rep["failed"].append(f"lab {lid}: leftovers {str(b.get('leftovers'))[:160]} — the operator must "
                                 "clear them on the labs host")
            rep["live_left"].append(f"lab {lid} (leftovers)")
            if inv is not None:
                inv.mark_leftovers(lid, ", ".join(map(str, b.get("leftovers"))))
        elif st == 200 or (st == 404 and b.get("error") == "unknown_lab"):
            if inv is not None:
                inv.record_destroy(lid, by="gone (project teardown)")       # the host no longer has it
        else:
            rep["failed"].append(f"lab {lid}: HTTP {st} {b.get('error') or ''}".rstrip())
            rep["live_left"].append(f"lab {lid}")


def _teardown_summary(rep: Dict[str, list]) -> str:
    parts = []
    if rep.get("stopped"):
        parts.append("stopped/destroyed: " + "; ".join(rep["stopped"]))
    if rep.get("kept_up"):
        parts.append("LEFT RUNNING (production — ask the owner whether to stop them): " + "; ".join(rep["kept_up"]))
    if rep.get("live_left"):
        parts.append("STILL RUNNING (could not be stopped): " + "; ".join(rep["live_left"]))   # review r3 #2
    if rep.get("failed"):
        parts.append("FAILED: " + "; ".join(rep["failed"]))
    if rep.get("notes"):
        parts.append("notes: " + "; ".join(rep["notes"]))
    return " | ".join(parts)


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


async def _release_app_gate(context, project_id: str) -> Optional[Dict[str, Any]]:
    """§4SX D3: None when the project has no live `app` run (the services/deliverables rehearsal applies);
    otherwise {ok, services: [], apps: [...], checks, detail} from probing each app on its serving lab."""
    try:
        from ..memory.run_registry import registry_for
        from . import runs as _runs
        reg = registry_for(context)
        rows = [r for r in (reg.rows(f"project:{project_id}", live_only=True) if reg is not None else [])
                if r["backend"] == "labs_app"]
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    from .labs import _caller
    origin = _caller(context)[0] or "autopilot"
    apps, checks, ok = [], [], True
    for r in rows:
        state, note, det = await _runs._probed(context, reg, r, origin)
        good = state == "running" and str(det.get("app_state") or "") == "up"
        ok = ok and good
        apps.append({"name": r["name"], "url": det.get("url") or "", "lab_id": det.get("lab_id") or "",
                     "lineage": str(r["backend_ref"]).partition("/")[0], "state": det.get("app_state") or state})
        checks.append(f"app {r['name']}: {det.get('app_state') or state}" + (f" ({note})" if note else "")
                      + (f" at {det.get('url')}" if det.get("url") else ""))
    return {"ok": ok, "services": [], "apps": apps, "checks": checks,
            "detail": "; ".join(checks) + ("" if ok else " — every kept app must be UP to release "
                                            "(`run status`, `lab deploy`/`lab app op=start`)")}


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
            # …and a DECLARED failure is one too: the bind failures are
            # `ToolOutcome.failed` with a "Service 'x' started …" head, which
            # the prefix check let through to a host probe (§4LR M5).
            from .outcome import OutcomeStatus as _OS
            if ((isinstance(_r_out, str) and _r_out.startswith("Error:"))
                    or getattr(_r_out, "status", None) in (_OS.FAILED, _OS.REJECTED)):
                ok = False
                checks.append(f"{name}: restart failed — {str(_r_out)[:120]}")
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
            # Reachability INSIDE the container: a host probe is answered by
            # anything on that port — OrbStack holds every published port on
            # the host (§4LR M5).
            if not port:
                alive = True
            elif callable(getattr(sup, "_port_listening", None)):
                try:
                    alive = bool(sup._port_listening(port))
                    # …and bound where the HOST can reach it: a loopback-only
                    # app in bridge mode answers inside, not to the user (N4)
                    if alive and callable(getattr(sup, "_host_unreachable_bind", None)) \
                            and sup._host_unreachable_bind(port):
                        alive = False
                except Exception:  # noqa: BLE001
                    alive = False
            else:
                alive = _probe_tcp(port)
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
                     "(start a listed service that is down with `run start name=<name> "
                     "command=<its command> where=workspace lifetime=server` — the name lowercase, "
                     "letters/digits/hyphens). "
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
            {"type": e["type"], "ts": e["ts"], "payload": _event_payload_preview(e["type"], e["payload"])}
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


#: §4UI (heuristics review #11): what makes a visual task's DELIVERABLE code
_VISUAL_CODE_SUFFIXES = (".html", ".htm", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".css", ".py",
                         ".glsl", ".frag", ".vert", ".wgsl", ".svg", ".vue", ".svelte")


def _is_visual_artifact_task(description: str, kind: str = "") -> bool:
    """A visual / runnable artifact whose close needs verification evidence.
    §4UI: only when what it MAKES is code — a declared deliverable with a code
    suffix, or (none declared) any task not declared research/write. "Write
    the HUD design doc → docs/hud.md" and "Research canvas fingerprinting
    defences" were refused a close for want of a screenshot."""
    if not _VISUAL_ARTIFACT_RE.search((description or "").lower()):
        return False
    from ..core.task_deliverable import declared_deliverable
    path = declared_deliverable(description or "")
    if path:
        return str(path).strip().lower().endswith(_VISUAL_CODE_SUFFIXES)
    return str(kind or "").strip().lower() not in ("research", "write")


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
    not_model_data: str = "",
    override: Optional[Dict[str, str]] = None,
    subtasks: Optional[List[str]] = None,
    task_kind: str = "",
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
    # action=autopilot: on / off (None = report)
    enabled: Any = None,
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
    # §4SU Phase 2: object subtasks carry a declared kind + deliverable
    try:
        subtasks, _subtask_kinds = _split_declared_subtasks(subtasks)
        from ..core.task_kind import normalize_kind as _norm_kind
        _task_kind_given = bool(str(task_kind or "").strip())   # "none" clears a kind
        task_kind = _norm_kind(task_kind)
    except ValueError as _ke:
        return _err(str(_ke))
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
                         "release", "create_version", "status", "autopilot",
                         # §4PE (P2): these looked up `title` too, and the auto-fill
                         # sent them to the CURRENT project (a constraint retired
                         # from the wrong project, answered as success)
                         "budget", "constraint_retire", "verify_release", "unrelease"}
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
            # §4LP: guard the project this call will REALLY touch — a TITLE
            # passed as project_id found no project here and passed the
            # guard (`update project_id="<title>" status=ACTIVE` un-released
            # the app), and a task op on another project's task_id guarded
            # the active project instead of the task's.
            _guard_ids = {project_id}
            if project_id and not store.get_project(project_id):
                _gid, _ = _resolve_project_ref(store, project_id, "")
                if _gid:
                    _guard_ids.add(_gid)
            _tid = task_id
            if _tid:
                try:
                    _t = store.get_task(_tid)
                    if _t and _t.get("project_id"):
                        _guard_ids.add(_t.get("project_id"))
                except Exception:  # noqa: BLE001
                    pass
            for _gid in _guard_ids:
                _rg = _released_guard(store, _gid) if _gid else None
                if _rg:
                    return _err(_rg)

    try:
        # ---- project lifecycle ------------------------------------------

        if act == "create":
            if not title:
                return _err("title is required for action=create")
            _not_asked = _create_not_requested(context)
            if _not_asked:
                from .outcome import ToolOutcome
                return ToolOutcome.rejected(_not_asked, reason_code="project_not_requested")
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
                        f"title. {FORK_ONLY_IF_ASKED}")
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
                # DEEP copy: helpers below mutate nested values in place
                # (`constraint_origins`), which a shallow copy hides from
                # the changed-keys diff (§4LZ review)
                import copy as _copy
                _meta_before = _copy.deepcopy(existing_meta)
                # Count the BURST, not the lifetime: a duplicate long after
                # the last one is a fresh mistake, not turn 4 of a loop.
                try:
                    _last_dup = float(existing_meta.get(
                        "duplicate_create_last_at", 0) or 0)
                except (TypeError, ValueError):
                    _last_dup = 0.0
                _prev = int(existing_meta.get("duplicate_create_retries", 0) or 0)
                if _last_dup and (now - _last_dup) > _DUPLICATE_CREATE_LOOP_WINDOW_SECONDS:
                    _prev = 0
                retry_count = _prev + 1
                existing_meta["duplicate_create_retries"] = retry_count
                existing_meta["duplicate_create_last_at"] = now
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
                    _stamp_constraint_origins(existing_meta, fresh,
                                              _constraint_origin(context))
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
                    # only the keys THIS call changed — metadata merges, so
                    # a whole-dict write put stale values back (§4LZ C2)
                    _changed = {k: v for k, v in existing_meta.items()
                                if k not in _meta_before or _meta_before[k] != v}
                    if _changed:
                        store.update_project(existing["id"], metadata=_changed)
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
                        f"one of: task_list "
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
                _stamp_constraint_origins(create_meta, req_constraints,
                                          _constraint_origin(context))
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
            if subtasks:
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
                                        constraints=req_constraints or None,
                                        kind=_subtask_kinds.get(desc.strip(), ""))
                    created_task_ids.append(tid)
                    pairs.append((tid, desc.strip()))
                    prev_id = tid
                for tid, desc in pairs:
                    _link_task_in_graph(context, pid, tid, desc)
                instruction = (
                    f"Project created WITH {len(created_task_ids)} task(s) from "
                    "your subtasks. Pacing: follow the PACING rule "
                    "in the CURRENT PROJECT block."
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
                    "all edit the same file. Give each task as an object "
                    "{description, kind, deliverable} (kind: build | write | "
                    "research | model_run | owner). Pacing: follow the PACING "
                    "rule in the CURRENT PROJECT block. As you work, "
                    "record durable facts (file "
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
            if _probe_turn_now():
                try:                                   # §4LP: a probe may clean up ITS OWN projects
                    store.update_project(pid, metadata={"probe_created": True})   # merges (§4LZ C2)
                except Exception:  # noqa: BLE001
                    pass
            return _ok({"plan_shown": True,           # §4SU Phase 4 / §4UI r1 M3: a plan was made; the
                        # PACING rule decides whether the turn ends (a neutral key: "awaits_go_ahead"
                        # told the model to stop when the message also asked for the work).
                        # FIRST key: the trailing-promise steer reads it from a truncated result
                        "created": pid,
                        "tasks_created": created_task_ids,
                        "workspace": f"projects/{pid}",
                        "constraints": req_constraints,
                        "note": _workspace_note(pid),
                        "agent_instruction": instruction,
                        "briefing": _briefing(store, pid)})

        if act == "list":
            projs = [_with_dates(p) for p in store.list_projects(status_filter=status_filter)]
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
                + (f"  (created {p['created']})" if p.get("created") else "")
                + ("   <- ACTIVE" if p.get("id") == current else "")
                for p in projs
            ]
            # §4SU Phase 4: display lines only (the full rows hit the 4k result
            # cap in 46/46 owner list calls — "show all projects" was answered
            # from a truncated list); `metadata={"verbose": true}` keeps them
            _out = {} if not bool((metadata or {}).get("verbose")) else {"projects": projs}
            return _ok({
                **_out,
                "count": len(projs),
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
                "created": _iso_date(_p.get("created_at")),
                "updated": _iso_date(_p.get("updated_at")),
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
            if not prev:
                return _ok({"exited": None, "note": "no project was active in this conversation"})
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
                if str(status).upper() == "ARCHIVED":
                    _tr = _teardown_refusal(context, store.get_project(project_id) or {}, "archive")
                    if _tr:
                        return _err(_tr)                 # review r3 m6: before the other fields change
                    # §4PF (P6): archiving has side effects (services stopped,
                    # the binding stashed, the pointer forgotten) that a bare
                    # status flip skipped — apply the other fields, then run
                    # the real action (r4: they were silently dropped)
                    _other = {k: v for k, v in dict(fields, **({"goal": goal} if goal else {}),
                                                     **({"kind": kind} if kind else {}),
                                                     **({"metadata": metadata} if metadata is not None else {})).items()}
                    if _other:
                        store.update_project(project_id, **_other)
                    return await tool_manage_projects_for_model(context, action="archive", project_id=project_id)
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
            # No eligibility gate here. A 2026-06-12 gate refused any hard
            # delete of a project that was neither active at request start
            # nor named in the user's message; live req 1e593552 (2026-09-06)
            # showed it refusing the user's own "delete it" five times in a
            # row, because the project had just auto-rolled to DONE and was
            # no longer active. Removed (journal §4FC): a project resolved
            # by id/title is deletable whether or not it is activated.
            # Pinned by tests/test_project_delete_ungated.py.
            # Stop the project's services BEFORE the workspace vanishes —
            # rmtree under a running service left orphaned processes +
            # registry entries (review H6). Hard delete also purges their
            # writable state dirs ("no files left behind").
            _tr = _teardown_refusal(context, store.get_project(rid) or {}, "delete")
            if _tr:
                return _err(_tr)
            # §4SX D2: everything the project owns stops first — runs, services, jobs, labs (kept ones too);
            # the row goes only when nothing it owns is left running (never a row-less orphan)
            _td = await _teardown_project(context, rid, hard=True, origin=_teardown_origin(context))
            if _td["live_left"]:
                return _err(f"NOT deleted: still running — {'; '.join(_td['live_left'][:6])}. "
                            f"{_teardown_summary(_td)}. The project, its tasks and its files are kept; what is "
                            "listed as stopped/destroyed above already is. Tell the owner what is left and try "
                            "again when it can be stopped.")
            _gone = store.get_project(rid) or {}
            ok = store.delete_project(rid, hard=True)
            if not ok:
                return _err(f"delete failed for {rid} — the record was NOT removed. "
                            f"{_teardown_summary(_td)}")
            forget_project_pointer(context, rid)
            _unlink_project_in_graph(context, rid, _gone.get("title", ""))
            _forget_project_episodes(context, rid, _gone.get("title", ""))
            _remove_project_changelog(context, rid)
            return _ok({"deleted": True, "project_id": rid, "hard": True,
                        "teardown": _teardown_summary(_td),
                        "note": "Project, its tasks/artifacts/events, its workspace files, its "
                                "knowledge-graph edges and the episodes that name it were "
                                "permanently removed. KEPT (say so if asked — do not claim "
                                "'no trace remains'): past conversation records (trajectories) "
                                "and vector-memory "
                                "entries until the next memory reconcile."})

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
            # §4SX D2: archive stops job/server runs, services, jobs and DISPOSABLE labs; `app` runs and kept
            # labs are production and stay up (review r1 H2) — named so the owner can decide
            _tr = _teardown_refusal(context, store.get_project(rid) or {}, "archive")
            if _tr:
                return _err(_tr)
            _td = await _teardown_project(context, rid, hard=False, origin=_teardown_origin(context))
            ok = store.delete_project(rid, hard=False)
            if not ok:
                return _err(f"archive failed for {rid} — the project was NOT archived. {_teardown_summary(_td)}")
            forget_project_pointer(context, rid)
            return _ok({"archived": True, "project_id": rid,
                        "teardown": _teardown_summary(_td),
                        "note": "Project hidden but kept; use action=resume to "
                                "bring it back, or action=delete to remove it "
                                "permanently."
                                + (" Its production apps/kept machines are STILL RUNNING — ask the owner "
                                   "whether to stop them." if _td["kept_up"] else "")
                                + (" Some of its runs/labs could NOT be stopped and are still running (see "
                                   "teardown) — tell the owner." if _td["live_left"] else "")})

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
            # §4LP: a NAMED project is reported (without switching) — 31 of
            # 53 live `status project_id=…` calls answered "free_chat" about
            # the conversation instead, and two turns aborted on it
            if (project_id and project_id != cur) or (title and not project_id):
                rid, rerr = _resolve_project_ref(store, project_id, title)
                if rerr:
                    return _err(rerr)
                if not rid or not store.get_project(rid):
                    _tomb = _deleted_note(store, project_id)             # §4PE (P7)
                    return _err(f"no project matches {(title or project_id)!r}{_tomb} — "
                                "manage_projects action=list shows the real ones")
                return _ok({"current": cur, "mode": "project" if cur else "free_chat",
                            "project": rid, "autopilot": _autopilot_on(store, rid),
                            "budget": _budget_view(store, rid),
                            "inventory": _inventory_text(_project_inventory(context, rid), hard=None) or "nothing running",
                            "briefing": _briefing(store, rid)})
            if not cur:
                return _ok({"current": None, "mode": "free_chat"})
            return _ok({"current": cur, "mode": "project",
                        "autopilot": _autopilot_on(store, cur),
                        "budget": _budget_view(store, cur),
                        # §4SX D2: what the project owns that keeps running (local records, no probing)
                        "inventory": _inventory_text(_project_inventory(context, cur), hard=None) or "nothing running",
                        "briefing": _briefing(store, cur)})

        if act == "budget":
            # §4MQ: the owner sets a project's unattended budget — steps,
            # runtime hours, check-in cadence. Without values it reports the
            # budget and the gates' state. Only the owner's own turn changes it.
            if project_id or title:
                project_id, _berr = _resolve_project_ref(store, project_id, title)
                if _berr:
                    return _err(_berr)
            if not project_id:
                return _err("no such project (pass project_id or title, or switch first) — nothing changed")
            _p = store.get_project(project_id)
            if not _p:
                return _err(f"project not found: {project_id}")
            _want = metadata if isinstance(metadata, dict) else {}
            _set = {}
            for _k, _cast, _lo, _hi in (("steps_cap", int, 0, 1000), ("checkpoint_every", int, 0, 1000),
                                        ("runtime_cap_hours", float, 0.0, 24.0 * 30)):
                if _k in _want and _want[_k] is not None:
                    try:
                        _v = _cast(_want[_k])
                    except (TypeError, ValueError):
                        return _err(f"{_k} must be a number, got {_want[_k]!r} — nothing changed")
                    if not (_lo <= _v <= _hi):
                        return _err(f"{_k} must be between {_lo} and {_hi} — nothing changed")
                    _set[_k] = _v
            _unknown = sorted(set(_want) - {"steps_cap", "checkpoint_every", "runtime_cap_hours"})
            if _unknown:
                return _err(f"budget takes steps_cap, runtime_cap_hours, checkpoint_every — not "
                            f"{', '.join(_unknown)}; nothing changed")
            if _set:
                if not _owner_turn_now():
                    return _err("only the owner's own turn changes a budget — a probe or "
                                "background run cannot")
                if "runtime_cap_hours" in _set:
                    # the UNATTENDED allowance between the owner's looks (r2
                    # M3: the lifetime runtime counted the owner's own work)
                    _set["unattended_runtime_cap_seconds"] = round(_set.pop("runtime_cap_hours") * 3600.0)
                store.update_project(project_id, metadata=_set)
            return _ok({"project": project_id, "budget": _budget_view(store, project_id)})

        if act == "autopilot" and str(project_id or title or "").strip().lower() in ("all", "*"):
            # §4MQ: one command stops every unattended project
            if enabled is None or str(enabled).strip().lower() not in (
                    "0", "false", "no", "off", "disable", "disabled"):
                return _err("project 'all' takes enabled=false only (pausing everything); "
                            "turn autopilot on one project at a time — nothing changed")
            if not _owner_turn_now():
                return _err("only the owner's own turn changes autopilot — "
                            "a probe or background run cannot")
            _off = []
            for _p in store.list_projects() or []:
                if (_p.get("metadata") or {}).get("autopilot"):
                    store.update_project(_p["id"], metadata={"autopilot": False})
                    _off.append(_p.get("title") or _p["id"])
            return _ok({"autopilot_off": _off,
                        "note": ((f"autopilot turned off on {len(_off)} project(s); a step already "
                                  "running finishes first (up to 45 min) and no new one starts")
                                 if _off else "no project had autopilot on")})

        if act == "autopilot":
            # §4LQ: the idle loop advances a project only with autopilot on.
            # §4SU Phase 5: the ONLY way autopilot turns on (autoadvance no
            # longer does); this reports it or turns it on/off. Not from a
            # probe or a background turn.
            # a NAMED project, never the current one by default (review N1:
            # the title was ignored and the current project went on autopilot)
            if project_id or title:
                project_id, _aerr = _resolve_project_ref(store, project_id, title)
                if _aerr:
                    return _err(_aerr)
            if not project_id:
                return _err("no such project (pass project_id or title, or switch first) — nothing changed")
            if not store.get_project(project_id):
                return _err(f"project not found: {project_id}")
            if enabled is None or str(enabled).strip() == "":
                return _ok({"project": project_id, "autopilot": _autopilot_on(store, project_id)})
            _ev = str(enabled).strip().lower()
            if _ev in ("1", "true", "yes", "on", "enable", "enabled"):
                want = True
            elif _ev in ("0", "false", "no", "off", "disable", "disabled"):
                want = False
            else:
                return _err(f"enabled must be true or false, got {enabled!r} — nothing changed")
            if not _owner_turn_now():
                return _err("only the owner's own turn changes autopilot — "
                            "a probe or background run cannot")
            # §4MQ: the owner's resume resets the unattended gates
            from ..core.project_advancer import resume_autopilot_metadata
            store.update_project(project_id, metadata=(resume_autopilot_metadata() if want
                                                       else {"autopilot": False}))
            return _ok({"project": project_id, "autopilot": want,
                        "budget": _budget_view(store, project_id),
                        "note": ("the idle loop may now advance this project between your turns"
                                 if want else "the idle loop will no longer advance this project")})

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
                kind=task_kind,
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
                    # §4SU Phase 4: the five-state names the model sees
                    _su = {"WAITING": "NEEDS_USER", "RUNNING": "IN_PROGRESS"}.get(
                        str(status).strip().upper(), str(status).strip().upper())
                    st_enum = TaskStatus[_su]
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
            #: §4TX: the tasks THIS call closed DONE (a re-close of a task already DONE is not one) — the
            #: mood's "finished a project task" counts these
            closed_now: List[str] = []
            missing: List[str] = []
            result_only: List[str] = []
            gated: List[str] = []
            gated_constraints: List[str] = []
            active_constraints: List[str] = []
            judged_violations: List[str] = []
            judged_violation_reasons: List[str] = []
            still_failing: List[str] = []
            still_failing_reasons: List[str] = []
            undelivered: List[str] = []
            undelivered_reasons: List[str] = []
            misquoted: List[str] = []
            misquoted_reasons: List[str] = []
            _quote_pending: List[tuple] = []
            waiting: List[str] = []
            waiting_reasons: List[str] = []
            unrun: List[str] = []
            unrun_reasons: List[str] = []
            # §4SU Phase 1: every refusal, one shape {task_id, gate, reason, fix}
            refused: List[Dict[str, str]] = []
            overrides_used: List[Dict[str, str]] = []
            if isinstance(override, str):
                try:
                    override = json.loads(override or "null")
                except ValueError:
                    override = None
            waived: List[str] = []
            # §4SU Phase 0: a close the plan DEMOTED (postconditions) or PARKED
            # (human gate) is reported — it used to read as plain success
            not_closed: List[Dict[str, str]] = []
            try:
                _ws_for_gate = store.ensure_workspace(project_id)
            except Exception:
                _ws_for_gate = None
            # (ok, reason) audit cache keyed by the task's FILE SET. The
            # audit is per-file-set (constraint_gate judges specific files);
            # a single per-CALL cache reused the FIRST task's verdict for
            # every sibling in a bulk close, so a constraint-VIOLATING
            # sibling with different files inherited the first task's clean
            # verdict and closed DONE unaudited (the exact bypass this gate
            # exists to stop). Keyed by frozenset(files) so identical file
            # sets still cost one audit, but distinct files are each judged.
            _constraint_audit_cache: Dict[frozenset, Any] = {}
            if st_enum == TaskStatus.DONE and len(target_ids) > 1:
                # §4QA-d r20: close dependencies BEFORE their dependents, so a
                # dependency counts as met only once it really closed — the
                # update below sets its node DONE; a refused one stays open
                target_ids = _dependencies_first(plan, target_ids)
            for tid in target_ids:
                if tid not in plan.tree.nodes:
                    missing.append(tid)
                    continue
                _done_before = plan.tree.nodes[tid].status == TaskStatus.DONE
                _waive_note = ""
                _post_waiver = ""
                _deliv_waiver = ""
                _quote_desc = None
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
                        refused.append({"task_id": tid, "gate": "recheck", "reason": str(_detail)[:400],
                                        "fix": "the failure it recorded still holds on the file as it is now — fix it, then close"})
                        continue
                    if _verdict is True:
                        live_evidence = _detail
                task_constraints = [
                    str(c) for c in
                    (plan.tree.nodes[tid].constraints or [])
                ] + proj_constraints
                # §4SU Phase 1: the mechanical gates are ONE check
                # (core.task_close.check_close) in a fixed order —
                # dependencies → declared file → model run (§4ST) → evidence
                # → postconditions — the same check every close path runs.
                # Its first refusal refuses this task; the legacy keys below
                # are derived from it (operator 11-02: kept until Phase 4).
                if st_enum == TaskStatus.DONE:
                    from ..core.task_deliverable import declared_deliverable
                    from ..core.task_close import check_close
                    # a rewrite of the description in THIS call is what the
                    # declared-file gate judges (that is how a changed plan
                    # names the file it really made) — only when it names a
                    # file: a rewrite WITHOUT the arrow kept the promise (m8)
                    _desc_for_gate = (description if description and len(target_ids) == 1
                                      and declared_deliverable(description)
                                      else plan.tree.nodes[tid].description)
                    _quote_desc = _desc_for_gate
                    # the STORED words are judged for a model run (a rewrite
                    # could drop them); the rewrite's declared file is read
                    _gate_files = list(deliverables or [])
                    if _desc_for_gate != plan.tree.nodes[tid].description:
                        _gate_files.append(declared_deliverable(_desc_for_gate) or "")
                    _over = dict(override) if isinstance(override, dict) else {}
                    if str(not_model_data or "").strip():
                        _over.setdefault("model_run", str(not_model_data))
                    _verdict = await asyncio.to_thread(
                        check_close, store, project_id, plan.tree, tid, actor="model",
                        result=str(result or ""), description=_desc_for_gate, files=_gate_files,
                        live_evidence=live_evidence, constraints=task_constraints, override=_over,
                        # not on a re-close of a task already DONE (§4ST)
                        skip=({"model_run"} if _done_before else set()),
                        single_task=len(set(target_ids)) == 1)
                    if not _verdict.ok:
                        _r = _verdict.first
                        _rd = _r.as_dict(tid)
                        if _done_before:
                            # r5/r7: a refused RE-close of a task already DONE
                            # (by an earlier call, or by a child's roll-up in
                            # this one) changes nothing — said so, in `refused`
                            # only: the old keys read "held at non-DONE"
                            _rd["reason"] = (_rd["reason"] + " (the task stays DONE: this re-close "
                                             "changed nothing)")[:500]
                            refused.append(_rd)
                            continue
                        refused.append(_rd)
                        if _r.gate == "deliverable":
                            undelivered.append(tid)
                            undelivered_reasons.append(f"{tid}: {_r.reason}")
                        elif _r.gate == "dependencies":
                            waiting.append(tid)
                            waiting_reasons.append(_r.reason)
                        elif _r.gate == "model_run":
                            unrun.append(tid)
                            unrun_reasons.append(f"{tid}: {_r.reason}")
                        elif _r.gate == "evidence" and _r.sub == "visual":
                            gated.append(tid)
                        elif _r.gate == "evidence":
                            gated_constraints.append(tid)
                            for c in task_constraints:
                                if c not in active_constraints:
                                    active_constraints.append(c)
                        elif _r.gate == "postconditions":
                            not_closed.append({"task_id": tid, "status": plan.tree.nodes[tid].status.value,
                                               "why": _r.reason[:300]})
                        continue
                    _waive_note = _verdict.waived.get("model_run", "")[:300]
                    _post_waiver = _verdict.waived.get("postconditions", "")
                    _deliv_waiver = _verdict.waived.get("deliverable", "")
                # Constraint JUDGMENT gate (2026-07-08): the evidence gate
                # above only checks that result text EXISTS — the chess
                # session showed the model happily supplies evidence prose
                # while the artifact violates the constraint (it built a
                # heuristic engine after restating "Ghost plays directly").
                # One background LLM audit of the project's workspace files
                # per task_update call; refusal carries the evidence. Escape
                # hatch for a user-approved exception: result text starting
                # with "CONSTRAINT-OVERRIDE:". Fails open on infra errors.
                _c_over = ""
                if isinstance(override, dict) and str(override.get("constraints") or "").strip():
                    _c_over = " ".join(str(override["constraints"]).split())[:300]
                elif str(result or "").lstrip().upper().startswith("CONSTRAINT-OVERRIDE:"):
                    _c_over = str(result).lstrip()[len("CONSTRAINT-OVERRIDE:"):].strip()[:300] or "CONSTRAINT-OVERRIDE"
                # r1 L4: a constraints override is recorded only when it
                # stood in for an audit that would have run (the task has files)
                # r9 M1: the task's own declared file counts as its file — as
                # the roll-up's constraint hold reads it (planning.py)
                _audit_rels = list(deliverables or [])
                if st_enum == TaskStatus.DONE:
                    _decl_a = declared_deliverable(_desc_for_gate)
                    if _decl_a and _decl_a not in _audit_rels:
                        _audit_rels.append(_decl_a)
                # §4UI: the audit runs even with an override — an override
                # only goes over a REAL refusal, and only once the OWNER
                # approved it (a constraint is the owner's word; the agent's
                # override used to skip the audit with no one asked)
                _c_over_used = False
                if (st_enum == TaskStatus.DONE and task_constraints):
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
                                                  _audit_rels)
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
                            and not _task_audit[0] and _done_before):
                        # r8: a refused RE-close of a task already DONE — as the check's (r5/r7)
                        refused.append({"task_id": tid, "gate": "constraints",
                                        "reason": (str((_task_audit[1] if len(_task_audit) > 1 else "")
                                                       or "the files break a constraint")[:400]
                                                   + " (the task stays DONE: this re-close changed nothing)"),
                                        "fix": "fix the files so they honour the constraints"})
                        continue
                    if (_task_files and _task_audit is not None and not _task_audit[0]
                            and _c_over and _owner_turn_now()):
                        # §4UK (operator 11-12: no confirmations): in the owner's own turn the agent's stated
                        # exception goes through at once and is RECORDED (event + ledger) — never parked for
                        # an approval. An unattended turn still cannot go over a constraint.
                        _c_over_used = True
                        _record_constraint_override(store, project_id, tid, _c_over,
                                                    _task_audit[1] if len(_task_audit) > 1 else "")
                    elif (_task_files and _task_audit is not None
                            and not _task_audit[0]):
                        judged_violations.append(tid)
                        _reason = _task_audit[1] if len(_task_audit) > 1 else ""
                        if _reason:
                            judged_violation_reasons.append(_reason)
                        refused.append({"task_id": tid, "gate": "constraints",
                                        "reason": str(_reason or "the files break a constraint")[:400],
                                        "fix": "fix the files so they honour the constraints — or, in the "
                                               "owner's own conversation, pass override={\"constraints\": "
                                               "\"<why>\"} to go over it (recorded, no approval needed)"})
                        continue
                # Register deliverables BEFORE flipping the task to DONE.
                # update_status can roll the whole project to DONE on this
                # same call (when it's the last open task), which fires the
                # cleanup sweep synchronously — so anything registered after
                # would already be gone. Accept both the explicit
                # `deliverables=[...]` list and a legacy `artifact_kind=file`
                # + `payload` pair; dedup is handled by the store.
                _reg_before = None
                if st_enum == TaskStatus.DONE:
                    try:
                        _reg_before = set(store.list_deliverables(project_id))
                    except Exception:  # noqa: BLE001
                        _reg_before = None
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
                if st_enum is not None:
                    # A passing live re-check doubles as the result when the
                    # model supplied none — it is real, current evidence.
                    _res_text = (result or live_evidence)
                    if _waive_note:
                        _res_text = f"{str(_res_text or '').strip()} [not model data: {_waive_note}]".strip()
                    # ON the loop (r2: a worker thread opened a cancel point
                    # between the write and its records below; the roll-up's
                    # close check measured ~30 ms on a 3,000-file workspace)
                    plan.update_status(tid, st_enum, result=_res_text,
                                       failure_reason=failure_reason,
                                       postconditions_waived=bool(_post_waiver))
                    _after = plan.tree.nodes[tid].status
                    if _after == TaskStatus.DONE and not_closed:
                        # r7: an entry a child's roll-up wrote for THIS task
                        # earlier in the call is stale once it really closed
                        not_closed[:] = [n for n in not_closed if n["task_id"] != tid]
                    # §4SU Phase 1: a waiver is never silent — once the task
                    # really closed, each gate it went over is in the ledger
                    # and the events (the owner reads both)
                    if st_enum == TaskStatus.DONE and _after == TaskStatus.DONE and not _done_before:
                        closed_now.append(str(tid))
                        for _g, _why_w in (("deliverable", _deliv_waiver),
                                           ("model_run", _waive_note), ("postconditions", _post_waiver),
                                           ("constraints", _c_over if _c_over_used else "")):
                            if not _why_w:
                                continue
                            overrides_used.append({"task_id": tid, "gate": _g, "reason": _why_w})
                            try:
                                store.log_event(project_id, tid, "close_override",
                                                {"gate": _g, "reason": _why_w, "actor": "model",
                                                 "req": _current_request_id()})
                                store.append_ledger(project_id, f"OVERRIDE {_g} on task {tid} (by the agent): {_why_w}")
                            except Exception:  # noqa: BLE001
                                logger.warning("close override not recorded", exc_info=True)
                    for _rp in list(getattr(plan.tree, "rollup_parked", []) or []):
                        # r2 M1: a parent this close rolled into a gate park
                        not_closed.append({"task_id": _rp, "status": "NEEDS_USER",
                                           "why": str(plan.tree.nodes[_rp].result_summary or "")[:300],
                                           "waiting": plan.tree.nodes[_rp].wait_code})   # §4SU Phase 3
                    if st_enum == TaskStatus.DONE and _after != TaskStatus.DONE:
                        _why_not = (plan.tree.nodes[tid].failure_reason
                                    or plan.tree.nodes[tid].result_summary or "")
                        not_closed.append({"task_id": tid, "status": _after.value,
                                           "why": str(_why_not)[:300],
                                           "waiting": plan.tree.nodes[tid].wait_code})        # §4SU Phase 3
                        # files registered for a close that FAILED are not the
                        # project's deliverables (audit B); a gate PARK keeps
                        # them (r1 M2: the owner's approval close carries no
                        # deliverables, and the DONE sweep then trashed them)
                        if _reg_before is not None and _after == TaskStatus.FAILED:
                            try:
                                for _rel in set(store.list_deliverables(project_id)) - _reg_before:
                                    store.unregister_file(project_id, _rel)
                            except Exception:  # noqa: BLE001
                                logger.warning("task_update: deliverables of a refused close kept",
                                               exc_info=True)
                    elif st_enum == TaskStatus.DONE and (ledger or "").strip() and project_id:
                        # Durable fact recorded at completion → design ledger,
                        # only once the task really closed (audit B)
                        try:
                            store.append_ledger(project_id, ledger.strip())
                        except Exception:
                            logger.debug("ledger append skipped", exc_info=True)
                    if _waive_note and plan.tree.nodes[tid].status == TaskStatus.DONE:
                        waived.append(tid)
                        try:
                            store.add_work_log(project_id, request=f"task_update {tid} DONE",
                                               outcome="experiment gate waived",
                                               note=f"not model data: {_waive_note}")
                        except Exception:  # noqa: BLE001
                            logger.debug("waiver work_log skipped", exc_info=True)
                # §4QA-e: a write-up citing a probe transcript should quote it
                # word for word — ADVISORY (operator 2026-10-19): noted only
                # once the task really closed DONE past every gate (r32), and
                # never on a re-close of a task already DONE
                if (_quote_desc is not None and not _done_before
                        and plan.tree.nodes[tid].status == TaskStatus.DONE):
                    # run after the loop (r33: awaited here, a cancel left the
                    # close half-written — DONE with the old promised path)
                    _quote_pending.append((tid, _quote_desc, plan.tree.nodes[tid].description))
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
                    # §4QA r2: a rewrite without the "→ path" keeps the old
                    # promise — else a path-less rewrite now, then DONE next
                    # call, closed the task with its file never written
                    from ..core.task_deliverable import declared_deliverable as _decl
                    _old_promise = _decl((store.get_task(tid) or {}).get("description"))
                    # §4UI: unless this call closes the task and retracts the
                    # file with a recorded reason (override.deliverable) —
                    # then the path-less rewrite is the new plan
                    from ..core.experiment_gate import MIN_WAIVER_CHARS as _MWC
                    # r1 MAJOR 2: only in the call that CLOSES the task — its
                    # deliverable gate ran on the old path and the waiver was
                    # recorded there; a retraction in a non-closing call left
                    # no record and no path for the next DONE to check
                    # (r2 m2: and only when the waiver was USED on a first
                    # close — that is when its record is written)
                    _retracts = (st_enum == TaskStatus.DONE
                                 and plan.tree.nodes[tid].status == TaskStatus.DONE
                                 and bool(_deliv_waiver) and not _done_before
                                 and isinstance(override, dict)
                                 and len(str(override.get("deliverable") or "").strip()) >= _MWC)
                    extras["description"] = (f"{description.rstrip()} → {_old_promise}"
                                             if _old_promise and not _decl(description)
                                             and not _retracts
                                             else description)
                if alternatives is not None:
                    extras["alternatives"] = alternatives
                if postconditions is not None:
                    extras["postconditions"] = list(postconditions)
                if constraints is not None:
                    extras["constraints"] = constraints
                if _task_kind_given:
                    extras["kind"] = task_kind          # §4SU Phase 2
                    # §4UI: a retype is remembered — the model-run gate trusts
                    # only a kind the plan declared (experiment_gate.planned_kind)
                    _old_kind = str((store.get_task(tid) or {}).get("kind") or "")
                    if _old_kind != str(task_kind or ""):
                        try:
                            store.log_event(project_id, tid, "kind_changed",
                                            {"from": _old_kind, "to": str(task_kind or ""),
                                             "req": _current_request_id()})
                        except Exception:  # noqa: BLE001
                            logger.debug("kind change not recorded", exc_info=True)
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
            # §4QA-e: after EVERY write of the call (r33/r34: awaited earlier,
            # a cancel half-applied a close or skipped the research indexing)
            if _quote_pending:
                from ..core.task_deliverable import note_misquotes
                for _qt, _qd, _qs in _quote_pending:
                    _misq = await note_misquotes(store, project_id, _qt, _qd, _qs)
                    if _misq:
                        misquoted.append(_qt)
                        misquoted_reasons.append(f"{_qt}: {_misq}")
            payload: Dict[str, Any] = {"updated": updated,
                                       "count": len(updated)}
            if closed_now:
                payload["closed"] = closed_now
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
            if waived:
                payload["experiment_gate_waived"] = waived
            if refused:
                # §4SU Phase 1: the one list and its one instruction (Phase 4:
                # the legacy keys — undelivered, waiting_on_dependencies,
                # unrun_experiments, still_failing, constraint_violations,
                # gated_constraints, gated_unverified — are gone, operator 11-02)
                from ..core.task_close import instruction_for
                payload["refused"] = refused
                payload["agent_instruction_refused"] = instruction_for(refused)
            if overrides_used:
                payload["overrides"] = overrides_used
            if not_closed:
                # r4 L1: one entry per task, its LAST state (a bulk close re-judges a parent per child)
                not_closed = list({n["task_id"]: n for n in not_closed}.values())
                payload["not_closed"] = not_closed
                # r4 M2: a gate park is a human gate; a roll-up refusal is not — it is fixed, not approved
                # §4SU Phase 3: the wait field, not the "human gate:" text
                _parked = [n["task_id"] for n in not_closed if n.get("waiting") == "human_gate"]
                _other = len(_parked) < len(not_closed)
                payload["agent_instruction_not_closed"] = (
                    f"{len(not_closed)} task(s) did NOT close DONE: "
                    + " | ".join(f"{n['task_id']} → {n['status']}: {n['why'][:160]}" for n in not_closed)
                    + (". A human-gated task (\"human gate:\") closes only when the OWNER approves it BY NAME in a "
                       "later message: ask them now to reply \"approve <task id>\" with the id shown here. "
                       "A plain yes does not approve it. Do not call task_update again for a gated task in this turn."
                       if _parked else "")
                    + ((" For every other task listed" if _parked else ".")
                       + (": fix what the reason names, then close it again." if _parked
                          else " Fix what the reason names, then close it again.")
                       if _other else ""))
            if misquoted:
                payload["quote_warnings"] = misquoted
                payload["agent_instruction_quote_check"] = (
                    f"QUOTE CHECK (advisory — the close itself is not held by it): in {len(misquoted)} "
                    "task(s) the write-up seems to quote the model's answers with words changed — a "
                    "quote of a transcript is data and must be the model's exact words. "
                    + " | ".join(misquoted_reasons[:3])
                    + (f" (+{len(misquoted_reasons) - 3} more task(s))" if len(misquoted_reasons) > 3 else "")
                    + " Fix those quotes in that file now (copy the words, or drop the quote marks and "
                    "say it as your own summary); the owner sees this in the project ledger.")
            return _ok(payload)

        _replanned = None
        _replan = None
        if act == "replan":
            # §4SU Phase 4 (audit: the same topic deleted and re-created 7×):
            # a new plan for the SAME project — DONE tasks, their files and
            # the research stay; every open task with no finished work under
            # it goes; the new subtasks are laid down as task_decompose does.
            if not project_id:
                return _err("no active project (pass project_id or switch first)")
            if not subtasks:
                return _err("subtasks (the NEW plan, a list) is required for action=replan — nothing changed")
            if task_id and task_id != project_id:
                # r3: a replan is the WHOLE project's open plan; a task_id was
                # silently ignored and every open task went
                return _err("NOT replanned: replan replaces the whole project's open plan and takes no "
                            "task_id. To re-plan one task, task_decompose it with task_id. Nothing changed.")
            _rg = _released_guard(store, project_id)
            if _rg:
                return _err(_rg)
            # r1 m2: an archived project's new tasks would never run
            if str((store.get_project(project_id) or {}).get("status") or "").upper() == "ARCHIVED":
                return _err("NOT replanned: the project is ARCHIVED — resume it first. Nothing changed.")
            # the decompose's own refusals, judged BEFORE anything is deleted —
            # a refusal after the delete would lose the old plan for nothing
            _new = [d for d in subtasks if d and str(d).strip()]
            if not _new:
                return _err("NOT replanned: every subtask is blank. Nothing changed.")
            _all = store.list_tasks(project_id) or []
            _kids: Dict[str, List[str]] = {}
            for _t in _all:
                _kids.setdefault(_t.get("parent_id") or "", []).append(_t["id"])
            _by = {t["id"]: t for t in _all}

            def _has_done(tid, _seen=None):
                _seen = _seen or set()
                if tid in _seen:
                    return False
                _seen.add(tid)
                return any(str(_by[c].get("status") or "").upper() == "DONE" or _has_done(c, _seen)
                           for c in _kids.get(tid, []) if c in _by)

            _drop = [t for t in _all if str(t.get("status") or "").upper() != "DONE" and not _has_done(t["id"])]
            # r1 m3: a task the model left IN_PROGRESS is not running — only a
            # recent claim (the boot reaper's 900 s) holds the replan back
            _now_ts = time.time()
            _running = [t["id"] for t in _drop if str(t.get("status") or "").upper() == "IN_PROGRESS"
                        and _now_ts - float(t.get("updated_at") or 0) < 900]
            # r3: an unattended step runs up to 45 min without touching the
            # row — while one is live, its claimed task counts as running
            if not _running and any(str(t.get("status") or "").upper() == "IN_PROGRESS" for t in _drop):
                from ..core.project_advancer import unattended_step_live
                if unattended_step_live(store, project_id):
                    _running = [t["id"] for t in _drop if str(t.get("status") or "").upper() == "IN_PROGRESS"]
            if _running:
                return _err(f"NOT replanned: task(s) {', '.join(_running[:3])} are running now — "
                            "wait for the step to finish, then replan. Nothing changed.")
            _drop_ids = {t["id"] for t in _drop}
            # r1 C1/C2/M1/m1: the old open tasks are deleted only AFTER the new
            # plan is laid down (task_decompose below) — deleting first rolled
            # the project up to DONE mid-replan (the workspace sweep ran, the
            # owner's constraints were retired), and any later refusal or
            # error lost the old plan for nothing
            _replan = {"drop_roots": [t["id"] for t in _drop if (t.get("parent_id") or "") not in _drop_ids],
                       "drop_ids": _drop_ids,
                       "kept_parents": sorted({t.get("parent_id") for t in _drop
                                               if t.get("parent_id") and t.get("parent_id") not in _drop_ids}),
                       "kept": sum(1 for t in _all if str(t.get("status") or "").upper() == "DONE")}
            act, task_id = "task_decompose", None          # the new plan, laid down as a decompose

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
                if _replan is not None:
                    # replan: the new plan IS the plan — no dedupe against the
                    # open tasks it replaces (r1 M1: the dedupe ran against
                    # them and could drop the whole new plan)
                    dropped_dupes = []
                else:
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
                                         sequential=sequential,
                                         kinds=[_subtask_kinds.get(str(d or "").strip(), "")
                                                for d in subtasks])
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
                        tid = plan.add_task(desc.strip(), depends_on=deps,
                                            kind=_subtask_kinds.get(desc.strip(), ""))
                        ids.append(tid)
                        pairs.append((tid, desc.strip()))
                        prev_id = tid
            except ValueError as e:
                if _replan is not None:
                    # r3: a half-laid new plan beside the intact old one would
                    # be duplicated by the next replan — take it back
                    for _nid in ids:
                        try:
                            store.delete_task(_nid)
                        except Exception:  # noqa: BLE001
                            logger.warning("replan: partial task %s not removed", _nid, exc_info=True)
                    return _err(f"NOT replanned: {e}. Nothing changed.")
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
            if _replan is not None and ids:
                # the new plan is down — now the old open tasks go (a subtree
                # with its root); the project has open work throughout, so it
                # never rolls up to DONE mid-replan
                for _rid in _replan["drop_roots"]:
                    try:
                        store.delete_task(_rid)
                    except Exception:  # noqa: BLE001
                        logger.warning("replan: old task %s not deleted", _rid, exc_info=True)
                # r1 M2: a kept parent whose open children went is re-judged
                if _replan["kept_parents"]:
                    try:
                        ProjectPlan(store, project_id).recheck_parents(_replan["kept_parents"])
                    except Exception:  # noqa: BLE001
                        logger.warning("replan: kept parents not re-checked", exc_info=True)
                store.log_event(project_id, None, "project_replanned",
                                {"removed": sorted(_replan["drop_ids"])[:50],
                                 "removed_count": len(_replan["drop_ids"]),
                                 "kept_done": _replan["kept"], "added": len(ids)})
                try:
                    store.append_ledger(project_id, f"REPLANNED: {len(_replan['drop_ids'])} open task(s) "
                                                    f"replaced by {len(ids)}, {_replan['kept']} done task(s) kept")
                except Exception:  # noqa: BLE001
                    logger.debug("replan ledger line skipped", exc_info=True)
                _replanned = {"removed_open_tasks": len(_replan["drop_ids"]), "kept_done_tasks": _replan["kept"]}
            elif _replan is not None:
                return _err("NOT replanned: no new task was created. Nothing changed.")
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
            return _ok({"plan_shown": True,           # §4SU Phase 4 / §4UI r1 M3: a plan was made; the
                        # PACING rule decides whether the turn ends (a neutral key: "awaits_go_ahead"
                        # told the model to stop when the message also asked for the work).
                        # FIRST key: the trailing-promise steer reads it from a truncated result
                        "created": ids,
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
                        "tasks": [{"id": tid, "description": desc,
                                   "kind": (plan.tree.nodes[tid].kind if tid in plan.tree.nodes else "")}
                                  for tid, desc in pairs],
                        "parent_id": target_id,
                        "sequential": bool(sequential),
                        **({"replanned": _replanned} if _replanned else {}),
                        "constraints": decompose_constraints,
                        "agent_instruction": (
                            f"Plan created with {len(ids)} task(s). Pacing: "
                            "follow the PACING rule in the CURRENT "
                            "PROJECT block."
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
            # §4SU Phase 3: the five-state view — a filter of WAITING / RUNNING
            # (or the stored names) and a `state` per task instead of `status`
            from ..core.task_wait import view_label, view_state
            _vf = str(status_filter or "").strip().upper()
            if _vf in ("WAITING", "RUNNING"):
                tasks = [t for t in store.list_tasks(project_id) if view_state(t)["state"] == _vf]
            else:
                tasks = store.list_tasks(project_id, status_filter=(
                    {"READY": "PENDING"}.get(_vf, status_filter) if status_filter else status_filter))
            # Slim down by default: a 50-task project with full rows
            # easily clears 100KB and triggers context offloading,
            # which the LLM then reads as "something was wrong".
            # `verbose=true` (passed via metadata) opts back in.
            verbose = bool((metadata or {}).get("verbose"))
            if not verbose:
                tasks = [
                    {**{k: t.get(k) for k in _TASK_LIST_SLIM_FIELDS if k in t and k != "status"},
                     "state": view_label(t)}
                    for t in tasks
                ]
            return _ok({"tasks": tasks, "count": len(tasks)})

        # ---- artifacts / events ------------------------------------------

        if act == "artifact_add":
            if not task_id:
                return _err("task_id is required for action=artifact_add (an artifact belongs "
                        "to a task). To record a project FILE that no task produced, use "
                        "action=describe_file with file_path AND description")
            if not artifact_kind:
                return _err("artifact_kind is required (file|url|note|tool_call)")
            if str(artifact_kind).strip().lower() == "file":
                # §4PE (P8): a raw `../../etc/hosts` was registered as a
                # deliverable and then read as permanently MISSING, blocking
                # release — the same rule the deliverables path applies
                _rel = str(payload or "").replace("\\", "/").strip()
                while _rel.startswith(("/", "./")):
                    _rel = _rel[2:] if _rel.startswith("./") else _rel[1:]
                if _rel.startswith("workspace/"):                            # the container's /workspace/…
                    _rel = _rel[len("workspace/"):]
                _own = f"projects/{project_id}/" if project_id else None
                if _own and _rel.lower().startswith(_own.lower()):
                    _rel = _rel[len(_own):]
                if not _rel or ".." in Path(_rel).parts:
                    return _err(f"artifact path {payload!r} is not inside the project workspace "
                                f"(no '..', workspace-relative) — nothing was registered")
                payload = _rel
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
            # (2) rehearse. §4SX D3 (design r1 H5): a project with a kept app is gated on that app being UP on
            # its serving lab — never on `/rehearse` of the live lab (it stops production)
            reh = await _release_app_gate(context, project_id)
            if reh is None:
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
                         if s.get("port")] + [a["url"] for a in (reh.get("apps") or []) if a.get("url")],
                "apps": reh.get("apps") or [],
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
                if ((_child.get("metadata") or {}).get("probe_created")
                        and not _probe_turn_now()):
                    # never hand the owner a probe's fork (§4LZ A-F4) — and
                    # archive it, or the owner's fork mints a second "X vN"
                    # on the same bumped port (review)
                    if str(_child.get("status", "")).upper() != "ARCHIVED":
                        try:
                            store.update_project(_child["id"], status="ARCHIVED")
                        except Exception:  # noqa: BLE001
                            pass
                    continue
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
                "constraint_origins": dict(pmeta.get("constraint_origins") or {}),
                "research_index": pmeta.get("research_index") or [],
                # the forked briefs stay the writer's own (§4QA: ownership by hash)
                "research_brief_shas": dict(pmeta.get("research_brief_shas") or {}),
            }
            if _probe_turn_now():
                # a probe's fork is marked like a probe's create (§4LZ A-F4),
                # so it can clean it up and the owner never inherits it
                new_meta["probe_created"] = True
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
                # §4GJ round 3: NOT `shutil.copytree` — `src_ws` is
                # `<sandbox>/projects/<id>`, a tree the model writes, and the
                # default copytree follows a planted link (a link to a host
                # private key materialised as a real file in the fork). The
                # dir_fd walk refuses every link that leaves `src_ws`.
                from .file_system import copytree_nofollow
                # ⚠ SAY WHAT WAS LEFT BEHIND (§4GK round 4). `skipped` was
                # discarded here, and `copied` is counted from the
                # DESTINATION, so it can never disagree with it: an escaping
                # symlink, a FIFO or a whole EACCES subtree was dropped and
                # the tool answered a clean "Copied N files". Only the dream
                # and isolation callers logged it.
                _skipped = copytree_nofollow(src_ws, dst_ws, src_ws, ignore=_ignore,
                                             dirs_exist_ok=True)
                if _skipped:
                    pretty_log("Project Fork",
                               f"{len(_skipped)} entr(ies) were NOT copied into the "
                               f"fork: {'; '.join(_skipped[:5])}"
                               f"{'…' if len(_skipped) > 5 else ''}",
                               level="WARNING", icon=Icons.WARN)
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
            if not project_id and title:                                  # §4PE r3: title-addressed
                project_id, _verr = _resolve_project_ref(store, None, title)
                if _verr:
                    return _err(_verr)
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
                    "constraint_origins": dict(smeta.get("constraint_origins") or {}),
                    # the workspace is copied byte for byte below, so the
                    # copied briefs stay the writer's own (§4QA r7 m2)
                    "research_index": smeta.get("research_index") or [],
                    "research_brief_shas": dict(smeta.get("research_brief_shas") or {}),
                    "cloned_from": project_id,
                })
            import shutil
            src_ws = Path(str(src.get("workspace_dir") or ""))
            dst_ws = Path(str((store.get_project(new_pid) or {})
                              .get("workspace_dir") or ""))
            copied = 0
            if src_ws.is_dir() and str(dst_ws):
                # §4GJ round 3: the fork path's twin — same model-writable
                # source, same symlink-safe copy (`the-sibling-one-revision-behind`).
                from .file_system import copytree_nofollow
                _skipped = copytree_nofollow(
                    src_ws, dst_ws, src_ws, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(
                        "RELEASE.md", ".services", "__pycache__", "*.pyc",
                        "node_modules"))
                if _skipped:                       # §4GK round 4, as above
                    pretty_log("Project Clone",
                               f"{len(_skipped)} entr(ies) were NOT copied into the "
                               f"clone: {'; '.join(_skipped[:5])}"
                               f"{'…' if len(_skipped) > 5 else ''}",
                               level="WARNING", icon=Icons.WARN)
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
            # the owner advanced this project → it may continue while idle
            # (§4LP). A probe's, a scheduled task's or a sub-agent's advance
            # is not the owner's opt-in, and runs under the unattended rules.
            _by_owner = _owner_turn_now()
            _autopilot_note = ""
            if _by_owner:
                # §4PG (R1-1): the owner's batch must not build beside a live
                # unattended step in the same workspace
                from ..core.project_advancer import unattended_step_live
                _live = unattended_step_live(store, project_id)
                if _live:
                    return _err(f"an unattended autopilot step is running on this project right now "
                                f"(started {_live} ago; a step is capped at 45 min) — wait for it to finish, "
                                f"then advance; nothing was started")
                try:
                    # §4SU Phase 5 (operator 11-02): autopilot is an explicit
                    # opt-in (action=autopilot enabled=true). The owner's batch
                    # no longer turns it on (§4MR: one "do the next task" after
                    # "stop all" restarted unattended work overnight); an ON
                    # autopilot gets its gates reset, as the owner looked (§4MQ)
                    _prev = (store.get_project(project_id) or {}).get("metadata") or {}
                    if _prev.get("autopilot"):
                        from ..core.project_advancer import resume_autopilot_metadata, _bump_meta

                        def _reset_if_on(m):
                            # r1: atomic — a "stop all" between the read and the
                            # write must not be overwritten back to ON
                            if m.get("autopilot"):
                                m.update(resume_autopilot_metadata())
                            return m
                        _bump_meta(store, project_id, _reset_if_on)
                    else:
                        _autopilot_note = ("autopilot stays OFF: only this batch runs. To let the idle loop continue "
                                           "the project between turns, the user asks for it (action=autopilot enabled=true)")
                except Exception:  # noqa: BLE001
                    pass
            from ..workspace import pinned_event_project as _pin_evt
            with _pin_evt(project_id):
                batch = await advance_many(
                    context, project_id,
                    max_tasks=max_tasks,
                    owner_requested=_by_owner,      # the owner asked (§4LP/§4LQ)
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
                "gated": len(getattr(batch, "gated", None) or []),   # §4SR: of count, skipped at a gate
                "ran": getattr(batch, "ran", batch.count),             # §4SR r4: entries that ran (per entry)
                "requested": ("all" if max_tasks is None else max_tasks),
                "stop_reason": batch.stop_reason,
                "agent_instruction": _advance_batch_instruction(batch, max_tasks),
            }
            if _autopilot_note:
                _adv_payload["autopilot"] = _autopilot_note
            # §4SU Phase 5: the batch went past model experiment(s) and then
            # stopped for another reason (a failure) — they still wait for
            # this turn's probe_model run
            _exps = [e for e in (getattr(batch, "experiments", None) or [])]
            if _exps and (batch.stop_reason != "model_experiment" or len(_exps) > 1):
                _adv_payload["model_experiments"] = _exps[:10]
                if batch.stop_reason != "model_experiment":
                    _adv_payload["agent_instruction"] += (
                        f" Also waiting: model experiment task(s) {', '.join(_exps[:5])} — the batch "
                        "went past them. Each runs with probe_model as its task says (in this turn only "
                        "if there is time), closed citing the transcript.")
                else:
                    _adv_payload["agent_instruction"] += (
                        f" The batch went past {len(_exps)} model experiments: {', '.join(_exps[:5])} — "
                        "the same applies to each.")
            # name what waits, or the model cannot tell the owner (§4LQ F5) —
            # on EVERY stop (§4SR): a batch that went past a gate can end as
            # completed_with_failures, deadline, count_reached… Two kinds
            # (r1 M3/m1): a GATE waits for the owner's answer; a HOLD
            # ("not closed unattended": a check, a model experiment) is
            # picked up by the owner's own autoadvance — asking the user
            # about it contradicts "do it NOW" on a model_experiment stop.
            try:
                _rows = [t for t in store.list_tasks(project_id)
                         if str(t.get("status") or "").upper() == "NEEDS_USER"]

                def _row(t):
                    _r = str(t.get("result_summary") or "")
                    # r3 m6: a hold's reason sits at the END ("— needs a real
                    # check — not closed unattended"); keep the tail
                    _r = _r if len(_r) <= 200 else _r[:110] + " … " + _r[-85:]
                    return {"task_id": t["id"], "description": str(t.get("description") or "")[:160],
                            "result": _r}
                # §4SU Phase 3: a hold is wait_on == owner_turn (the field)
                _gates = [_row(t) for t in _rows if t.get("wait_on") != "owner_turn"][:10]
                _holds = [_row(t) for t in _rows if t.get("wait_on") == "owner_turn"][:10]
                if _gates:
                    _adv_payload["needs_user_tasks"] = _gates
                    if batch.stop_reason != "needs_user":
                        _adv_payload["agent_instruction"] += (
                            " Also waiting on the owner: the task(s) in needs_user_tasks — "
                            "name them and ask; a request to run all tasks is not their "
                            "answer.")
                elif batch.stop_reason == "needs_user":
                    # only holds wait: nothing for the user to answer. The
                    # base text is rebuilt, not dropped (tally, no_reload).
                    from dataclasses import replace as _dc_replace
                    if _by_owner:
                        # r2 #2: in the owner's own run a held task is
                        # pickable, so one still open waits on another task
                        _adv_payload["agent_instruction"] = _advance_batch_instruction(
                            _dc_replace(batch, stop_reason="open_but_waiting"), max_tasks) + (
                            " The open task(s) in held_tasks were held for the owner's own "
                            "run (see each result): the owner's autoadvance picks one up "
                            "once nothing it depends on is still open.")
                    else:
                        # r3 MAJOR: unattended, they were held BECAUSE no owner
                        # was there — nothing waits on anything
                        _adv_payload["agent_instruction"] = _advance_batch_instruction(
                            _dc_replace(batch, stop_reason="held_unattended"), max_tasks) + (
                            " They wait for the owner's own run (see each result) — the "
                            "owner's autoadvance runs them.")
                if _holds:
                    _adv_payload["held_tasks"] = _holds
                    if batch.stop_reason != "model_experiment" and (
                            batch.stop_reason != "needs_user" or _gates):
                        _adv_payload["agent_instruction"] += (
                            " Held for the owner's own run (not a question for the user): "
                            "the task(s) in held_tasks — the owner's autoadvance picks "
                            "them up.")
            except Exception:  # noqa: BLE001
                logger.debug("needs_user task list skipped", exc_info=True)
            # A failure-shaped stop names the LEDGER's failed tasks, not just
            # this batch's: on an already-FAILED project the batch advanced
            # nothing, so the model otherwise has no ids/reasons to act on
            # (2026-08-01: "project done" was relayed over a FAILED ledger).
            # §4SR r7: a count/cap stop keeps its reason with a FAILED task
            # in the batch — still failure-shaped
            if batch.stop_reason in ("project_failed",
                                     "completed_with_failures", "failed",
                                     "repeated_failures") or any(
                    str(a.get("status", "")).upper() == "FAILED"
                    for a in (batch.advanced or [])):
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
                            " When the user wants a failed task fixed: repair its deliverable, "
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
                # §4SR r8: `count` includes gate ticks — what RAN decides
                _ran = getattr(batch, "ran", batch.count)
                _mk = _TO.partial if _ran else _TO.failed
                return _mk(
                    _json.dumps(_adv_payload, default=str),
                    world_changed=bool(_ran),
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
            # §4MM r2: the same gate as `create` — one deliverer gated, the
            # other open, is the §4LZ class
            _not_asked = _create_not_requested(context)
            if _not_asked:
                from .outcome import ToolOutcome
                return ToolOutcome.rejected(_not_asked, reason_code="project_not_requested")
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
                ids = plan.decompose(root, subtasks,
                                     kinds=[_subtask_kinds.get(str(d or "").strip(), "") for d in subtasks])
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

#: §4SU Phase 4: the actions the MODEL sees (the schema enum). The rest stay
#: callable (API, tests, an old habit) — none of them is named by a system
#: message the model reads (every one that is, is here).
_VISIBLE_ACTIONS = (
    "create", "switch", "status", "get", "list", "update", "archive", "resume", "delete",
    "replan", "task_decompose", "task_add", "task_update", "task_delete", "task_list",
    "autoadvance", "autopilot", "budget", "ledger", "config", "describe_file",
    "unregister_file", "artifact_list", "research", "release", "create_version",
    "constraint_retire", "promote_from_context",
)

MANAGE_PROJECTS_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "manage_projects",
        # §4SU Phase 4 (audit: 8.7k chars, 40 actions, 17 never used, 41
        # params, 15 never passed): the rules that decide behaviour, one phrase
        # per action, and only the visible actions' parameters.
        "description": (
            "Long-term projects (coding or general): a task tree that survives across sessions. "
            "CREATE only when the user explicitly asks to start/track a project, or the work spans "
            "MULTIPLE files AND multiple turns — a SINGLE-FILE deliverable (even a large one) is written "
            "with file_system, no project. Past similar projects in memory are NOT a reason to create one. Never create twice: if a CURRENT PROJECT is shown, work on it "
            "(replan for a new plan of the same project). "
            "PACING: once a project is active, its CURRENT PROJECT block gives the one pacing rule "
            "(a message that asks for the project AND the work is the go-ahead). create takes "
            "`subtasks` too, so create + plan is one call. "
            "A task is DONE only when its work is really complete; a refused close comes back in "
            "`refused` with the fix. "
            "release, unrelease and delete run at once when the user asks (no confirmation). Prefer archive "
            "(reversible) to delete when the user did not say delete. autopilot and budget only on the "
            "user's request. "
            "Actions: create · switch · status · get · list · update · archive · resume · delete · "
            "replan (subtasks = the NEW plan; keeps done tasks, their files and the research) · "
            "task_decompose (subtasks=[…]; task_id = break that task down; ANY list of 2+ tasks goes here in ONE call, not N task_add) · task_add (one task) · "
            "task_update (task_ids=[…] for many) · task_delete · task_list · autoadvance (count) · "
            "autopilot (enabled) · budget · ledger · config · describe_file (when you create or work out a "
            "file) · unregister_file · artifact_list · research (topic/topics) · release (DONE→RELEASED "
            "when the user says it works or asks for it; directions) · create_version (change a RELEASED project) · "
            "constraint_retire · promote_from_context (after the user accepted the offer)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(_VISIBLE_ACTIONS)},
                "title": {"type": "string"},
                "kind": {"type": "string", "enum": ["CODING", "GENERAL"]},
                "goal": {"type": "string"},
                "metadata": {"type": "object", "description": "update: project fields; budget: its caps."},
                "project_id": {"type": "string", "description": "Omit for the current project."},
                "task_id": {"type": "string", "description": "ONE bare task id."},
                "task_ids": {"type": "array", "items": {"type": "string"}},
                "parent_id": {"type": "string"},
                "description": {"type": "string",
                                "description": "Task description; describe_file: what the file does."},
                "status": {"type": "string", "enum": ["PENDING", "DONE", "FAILED", "WAITING"],
                           "description": "task_update (WAITING = for the user)."},
                "postconditions": {"type": "array", "items": {"type": "string"}},
                "constraints": {"type": "array", "items": {"type": "string"},
                                "description": "The user's explicit requirements, near-verbatim; they gate DONE."},
                "depends_on": {"type": "array", "items": {"type": "string"},
                               "description": "task_add: task ids that must be DONE first."},
                "sequential": {"type": "boolean"},
                "topic": {"type": "string"},
                "topics": {"type": "array", "items": {"type": "string"}},
                "result": {"type": "string", "description": "task_update DONE: what was done / shown."},
                "override": {"type": "object", "additionalProperties": {"type": "string"},
                             "description": "task_update DONE: {gate: reason} for a refused gate that may be "
                                            "waived (deliverable, model_run, postconditions, constraints); "
                                            "recorded. deliverable = the task no longer makes its declared file."},
                "failure_reason": {"type": "string"},
                "subtasks": {"type": "array",
                             "items": {"anyOf": [
                                 {"type": "string"},
                                 {"type": "object", "properties": {
                                     "description": {"type": "string"},
                                     "kind": {"type": "string", "enum": ["build", "write", "research", "model_run", "owner"]},
                                     "deliverable": {"type": "string"}},
                                  "required": ["description"]}]},
                             "description": "Ordered tasks, each one file or a bounded module (never split the same file into N tasks). Objects "
                                            "{description, kind, deliverable}: kind = build | write (a document) | "
                                            "research | model_run (a model must answer: probe_model) | owner (only "
                                            "the user can); deliverable = the file it produces."},
                "task_kind": {"type": "string", "enum": ["build", "write", "research", "model_run", "owner", "none"]},
                "payload": {"type": "string", "description": "constraint_retire: the constraint, its index or 'all'."},
                "deliverables": {"type": "array", "items": {"type": "string"},
                                 "description": "task_update DONE: files to KEEP when the project finishes."},
                "status_filter": {"type": "string"},
                "context_summary": {"type": "string", "description": "promote_from_context: the chat's gist."},
                "enabled": {"type": "boolean"},
                "count": {"type": "string", "description": "autoadvance: '3' or 'all'."},
                "ledger": {"type": "string",
                           "description": "One durable fact (layout, API name), shown in the briefing every turn; omit to read."},
                "config_key": {"type": "string"},
                "config_value": {"type": "string"},
                "file_path": {"type": "string"},
                "file_role": {"type": "string"},
                "directions": {"type": "string",
                               "description": "release: the user's runbook, second person, ≥2 sentences."},
            },
            "required": ["action"],
        },
    },
}
