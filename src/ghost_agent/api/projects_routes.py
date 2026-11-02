"""HTTP routes for long-term project management.

Exposes the ``ProjectStore`` plus a subset of ``ProjectPlan`` /
``project_advancer`` behaviour to web and Slack clients. Routes are
stateless; auth reuses the ``X-Ghost-Key`` header via the same
``verify_api_key`` dependency used by the core chat routes.

Endpoints:
  GET    /api/projects
  POST   /api/projects
  GET    /api/projects/{pid}
  PATCH  /api/projects/{pid}
  DELETE /api/projects/{pid}
  POST   /api/projects/{pid}/switch
  POST   /api/projects/{pid}/resume
  POST   /api/projects/{pid}/advance
  GET    /api/projects/{pid}/events
  GET    /api/projects/{pid}/tasks
  POST   /api/projects/{pid}/tasks
  PATCH  /api/projects/{pid}/tasks/{tid}
  DELETE /api/projects/{pid}/tasks/{tid}
"""

import asyncio
import re
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .routes import verify_api_key, get_agent, _mark_foreground
from ..core.planning import ProjectPlan, TaskStatus
from ..core.project_advancer import (
    advance_once,
    default_code_generator,
    default_llm_classifier,
    pinned_project_context,
)
from ..core.prompts import build_project_briefing
from ..memory.projects import _canon_id
from ..workspace import pinned_event_project

logger = logging.getLogger("GhostAgent")

async def _origin_dep(request: Request) -> None:
    """§4PG probe: every write on this surface (create, task add/update,
    delete) is an event with an origin — the advance route derived the
    caller's identity, the others wrote `system`, so an owner's own UI
    action (or a probe's) could read as autonomous work in the digest.
    FastAPI runs an async dependency in the request's task, so the
    ContextVars set here are the endpoint's for the rest of the request."""
    try:
        from ..utils.logging import request_origin_context, request_id_context
        import uuid as _uuid
        origin = _request_origin(request)
        rid = str(request.headers.get("X-Request-Id") or "").strip() or _uuid.uuid4().hex[:12]
        if origin == "probe" and not rid.startswith("probe-"):
            rid = f"probe-{rid}"
        request_origin_context.set(origin)
        request_id_context.set(rid)
    except Exception:  # noqa: BLE001 — identity is best-effort here; the advance route sets its own
        pass


projects_router = APIRouter(prefix="/api/projects",
                            dependencies=[Security(verify_api_key), Depends(_origin_dep)])


# --------------------------------------------------------------------- models

class ProjectCreate(BaseModel):
    title: str = Field(..., min_length=1)
    kind: str = "GENERAL"
    goal: str = ""
    metadata: Optional[Dict[str, Any]] = None


class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    goal: Optional[str] = None
    kind: Optional[str] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskCreate(BaseModel):
    description: str = Field(..., min_length=1)
    parent_id: Optional[str] = None
    dependency_type: str = "ALL"
    alternatives: Optional[List[str]] = None
    postconditions: Optional[List[str]] = None
    kind: str = ""                     # §4SU Phase 2: declared task kind (core/task_kind.py)


class TaskUpdate(BaseModel):
    description: Optional[str] = None
    status: Optional[str] = None
    result_summary: Optional[str] = None
    failure_reason: Optional[str] = None
    alternatives: Optional[List[str]] = None
    postconditions: Optional[List[str]] = None
    kind: Optional[str] = None         # §4SU Phase 2 ("" clears it)


def _store(request: Request):
    agent = get_agent(request)
    store = getattr(agent.context, "project_store", None)
    if store is None:
        raise HTTPException(503, "project_store not configured")
    return store


def _context(request: Request):
    return get_agent(request).context


def _set_current_or_held(ctx, pid: str) -> None:
    """``_set_current`` for the console routes. The shared context still
    carries the last chat's conversation key; when that chat is held off the
    owner's binding (§4QA), the switch would last no request at all — refuse
    it (409) BEFORE anything changes (r15 MAJOR-2: refusing after the switch
    had already re-scoped the context and evicted that chat's guest seat)."""
    from ..tools.projects import _CURRENT_SENTINEL, _guest_leaving_owner, _set_current
    sp = getattr(ctx, "scratchpad", None)
    conv = getattr(ctx, "conversation_key", None)
    if sp is not None and _guest_leaving_owner(sp, pid, conv):
        raise HTTPException(409, f"not switched: another conversation owns the active project "
                                 f"binding ({sp.get(_CURRENT_SENTINEL)}); switch from that chat")
    _set_current(ctx, pid)


# --------------------------------------------------------------------- projects

@projects_router.get("")
async def list_projects(request: Request, status: Optional[str] = None):
    store = _store(request)
    return {
        "projects": store.list_projects(status_filter=status),
        "current": getattr(_context(request), "current_project_id", None),
    }


def _guarded_metadata(md, origin: str = "owner"):
    """§4PG (R3-2): the HTTP surface honours the same rule as the tool —
    `autopilot` and the budget/gate keys are never metadata a caller writes
    (any key holder could opt a project into the idle loop and reset its
    gates); the probe stamp is the system's (a probe's own is dropped, an
    owner's refused)."""
    if not isinstance(md, dict):
        return md
    from ..tools.projects import _SYSTEM_METADATA_KEYS
    from ..core.project_advancer import BUDGET_METADATA_KEYS
    if origin == "probe" and "probe_created" in md:
        md = {k: v for k, v in md.items() if k != "probe_created"}
    bad = sorted(k for k in md if k in _SYSTEM_METADATA_KEYS or k in BUDGET_METADATA_KEYS)
    if bad:
        raise HTTPException(400, f"{', '.join(bad)}: not metadata — autopilot is turned on by the owner's own "
                                 f"turn (manage_projects action=autopilot), budgets by action=budget")
    return md


def _request_origin(request: Request) -> str:
    """§4PG (R3-1): who is calling — the chat path derives this from the
    X-Ghost-Origin header and the request id; this surface never did, so a
    probe's advance ran as the owner's and every row said `owner`."""
    try:
        from ..utils.logging import ORIGIN_PROBE, is_probe_request_id
        hdr = str(request.headers.get("X-Ghost-Origin") or "").strip().lower()
        rid = str(request.headers.get("X-Request-Id") or "").strip()
        if hdr == ORIGIN_PROBE or is_probe_request_id(rid):
            return "probe"
    except Exception:  # noqa: BLE001
        pass
    return "owner"


@projects_router.post("", status_code=201)
async def create_project(body: ProjectCreate, request: Request):
    store = _store(request)
    _md = _guarded_metadata(body.metadata, _request_origin(request))
    if _request_origin(request) == "probe":
        _md = dict(_md or {}, probe_created=True)             # §4PG: a probe's project is a test fixture
    try:
        pid = store.create_project(
            title=body.title, kind=body.kind, goal=body.goal,
            metadata=_md,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": pid, "project": store.get_project(pid)}


@projects_router.get("/{pid}")
async def get_project(pid: str, request: Request):
    # Canonicalize once at entry (store ids are strip+lowercase): the store
    # canonicalizes its own lookups, but raw path pids were compared against
    # canonical ids and stored into `current_project_id`, so a case-mangled
    # pid 404'd on ownership checks and diverged the workspace path.
    pid = _canon_id(pid)
    proj = _store(request).get_project(pid)
    if not proj:
        raise HTTPException(404, "project not found")
    return proj


@projects_router.patch("/{pid}")
async def update_project(pid: str, body: ProjectUpdate, request: Request):
    store = _store(request)
    pid = _canon_id(pid)
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(400, "no fields to update")
    if "metadata" in fields:
        fields["metadata"] = _guarded_metadata(fields["metadata"])      # §4PG (R3-2)
    try:
        # Off-loop: a status=DONE transition fires the workspace cleanup
        # sweep (sync filesystem walk) inline from the store hook.
        ok = await asyncio.to_thread(store.update_project, pid, **fields)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not ok:
        raise HTTPException(404, "project not found")
    return store.get_project(pid)


@projects_router.delete("/{pid}", status_code=204)
async def delete_project(pid: str, request: Request, hard: bool = False):
    store = _store(request)
    pid = _canon_id(pid)
    if not store.get_project(pid):
        raise HTTPException(404, "project not found")
    # Stop the project's services FIRST, as the tool path does (§4LZ C3:
    # the API deleted/archived the project and left its services running,
    # registry rows pointing at a project that no longer existed). A hard
    # delete also purges their state dirs.
    try:
        from ..tools.projects import _stop_project_services
        await asyncio.to_thread(_stop_project_services, _context(request), pid, hard)
    except Exception:  # noqa: BLE001 — the delete itself must still run
        import logging as _lg
        _lg.getLogger("GhostAgent").warning("service stop before project delete failed", exc_info=True)
    # Off-loop: a hard delete rmtree's the project workspace (sync
    # filesystem work that must not block the process-wide event loop).
    _title = (store.get_project(pid) or {}).get("title", "")
    ok = await asyncio.to_thread(store.delete_project, pid, hard=hard)
    if not ok:
        raise HTTPException(404, "project not found")
    ctx = _context(request)
    if hard:
        # The tool path's graph leg (2cb40b10): edges naming a deleted
        # project kept steering the model to rebuild it.
        from ..tools.projects import _unlink_project_in_graph
        await asyncio.to_thread(_unlink_project_in_graph, ctx, pid, _title)
    # Route through the tool-side helper: a raw attribute write left the
    # conversation sentinel + workspace_model pointer naming the deleted id,
    # so reconcile reactivated the deleted project and the sandbox recreated
    # projects/<deleted-id>/. §4NW: it also clears a PERSISTED binding when
    # the live pointer is elsewhere (after a restart it was None).
    from ..tools.projects import forget_project_pointer
    forget_project_pointer(ctx, pid)
    # 204 No Content MUST have an empty body (RFC 9110). JSONResponse(204,
    # content=None) serialized a 4-byte `null`, which strict HTTP/2 clients
    # and proxies reject.
    return Response(status_code=204)


@projects_router.post("/{pid}/switch")
async def switch_project(pid: str, request: Request):
    store = _store(request)
    pid = _canon_id(pid)
    proj = store.get_project(pid)
    if not proj:
        raise HTTPException(404, "project not found")
    # Route through the tool-side setter: a raw attribute write skipped the
    # conversation sentinel + workspace_model pointer + scratchpad
    # snapshot/hydrate, so the next chat turn's reconcile silently undid
    # the switch (or, with no binding, the activation leaked globally).
    _set_current_or_held(_context(request), pid)
    return {"switched_to": pid,
            "briefing": build_project_briefing(store, pid)}


@projects_router.post("/{pid}/resume")
async def resume_project(pid: str, request: Request):
    store = _store(request)
    pid = _canon_id(pid)
    proj = store.get_project(pid)
    if not proj:
        raise HTTPException(404, "project not found")
    _set_current_or_held(_context(request), pid)
    store.log_event(pid, None, "project_resumed", {})
    return {"project": proj,
            "briefing": build_project_briefing(store, pid)}


@projects_router.post("/{pid}/advance")
async def advance_project(pid: str, request: Request):
    ctx = _context(request)
    store = _store(request)
    pid = _canon_id(pid)
    if not store.get_project(pid):
        raise HTTPException(404, "project not found")
    # §4SU Phase 0 (audit): the same rule as the tool's autoadvance (§4PG
    # R1-1) — never build beside a live unattended step in this workspace
    from ..core.project_advancer import unattended_step_live
    _live = unattended_step_live(store, pid)
    if _live:
        raise HTTPException(409, f"an unattended autopilot step is running on this project "
                                 f"(started {_live} ago) — wait for it to finish; nothing was started")
    # Build the SAME project-pinned tool runner the manage_projects
    # autoadvance action uses. advance_once without one runs the degraded
    # classify-only path, which marks a research leaf DONE having done
    # nothing — theatrical completion on a network-reachable authed route —
    # so when a runner can't be built we refuse rather than fail open.
    tools_map = None
    try:
        from ..tools.registry import get_available_tools
        tools_map = get_available_tools(pinned_project_context(ctx, pid))
    except Exception:
        tools_map = None
    if not tools_map:
        raise HTTPException(
            503, "tool runner unavailable — refusing to advance on the "
                 "degraded classify-only path")

    async def _run(name: str, args: Dict[str, Any]) -> str:
        handler = tools_map.get(name)
        if not handler:
            return f"ERROR: tool {name} unavailable"
        return await handler(**args)

    from ..core.coding_executor import build_coding_task
    # §4CB R1 A-F1: mark the WHOLE advance as a foreground request (same
    # bracket as game_routes). This route runs an LLM classifier plus a
    # 4096-token code generation on the main slot; unmarked, the biological
    # tick treated the process as idle and ran journal consolidation
    # MID-advance, and the RSS watchdog's opt-in execv restart could kill
    # the process mid-advance. try/finally so a raise never leaks the count.
    _mark_foreground(get_agent(request), +1)
    try:
        # Pin the event stamp to the project being advanced (as the idle tick
        # and tool path both do) — otherwise workspace events stamp whatever
        # project the chat side currently has active.
        from ..utils.logging import request_origin_context, request_id_context
        _origin = _request_origin(request)
        _o_tok = request_origin_context.set(_origin)
        _rid = str(request.headers.get("X-Request-Id") or "").strip() or f"adv-{pid}"
        if _origin == "probe" and not _rid.startswith("probe-"):
            _rid = f"probe-{_rid}"            # r3: the mark travels as the id PREFIX (every reader agrees)
        _r_tok = request_id_context.set(_rid)
        try:
          with pinned_event_project(pid):
            result = await advance_once(
                ctx, pid,
                owner_requested=(_origin != "probe"),   # §4PG (R3-1): a probe's advance is not the owner's
                tool_runner=_run,
                llm_classifier=default_llm_classifier(ctx),
                code_generator=default_code_generator(ctx),
                coding_executor=build_coding_task,
            )
        finally:
            request_origin_context.reset(_o_tok)
            request_id_context.reset(_r_tok)
    finally:
        _mark_foreground(get_agent(request), -1)
    return {
        "ok": bool(result.ok) and not re.search(r"\b(?:failed|could not be built)\b", str(result.summary or "")),   # §4PG (R3-6)
        "task_id": result.task_id,
        "classification": result.classification,
        "summary": result.summary,
        "artifact_id": result.artifact_id,
    }


@projects_router.get("/{pid}/events")
async def list_events(pid: str, request: Request,
                      limit: int = 50, type: Optional[str] = None):
    store = _store(request)
    pid = _canon_id(pid)
    if not store.get_project(pid):
        raise HTTPException(404, "project not found")
    # Clamp: a negative limit means "no limit" in SQLite (unbounded response)
    # and a huge value floods memory/context.
    limit = max(1, min(int(limit), 1000))
    return {"events": store.list_events(pid, limit=limit, event_type=type)}


# --------------------------------------------------------------------- tasks

@projects_router.get("/{pid}/tasks")
async def list_tasks(pid: str, request: Request, status: Optional[str] = None):
    store = _store(request)
    pid = _canon_id(pid)
    if not store.get_project(pid):
        raise HTTPException(404, "project not found")
    return {"tasks": store.list_tasks(pid, status_filter=status)}


@projects_router.post("/{pid}/tasks", status_code=201)
async def add_task(pid: str, body: TaskCreate, request: Request):
    store = _store(request)
    pid = _canon_id(pid)
    if not store.get_project(pid):
        raise HTTPException(404, "project not found")
    try:
        tid = store.add_task(
            pid, description=body.description, parent_id=body.parent_id,
            dependency_type=body.dependency_type,
            alternatives=body.alternatives,
            postconditions=body.postconditions,
            kind=body.kind,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": tid, "task": store.get_task(tid)}


@projects_router.patch("/{pid}/tasks/{tid}")
async def update_task(pid: str, tid: str, body: TaskUpdate, request: Request):
    store = _store(request)
    pid = _canon_id(pid)
    tid = _canon_id(tid)          # §4SU Phase 1: a mixed-case id was a silent 200 no-op
    existing = store.get_task(tid)
    if not existing or existing["project_id"] != pid:
        raise HTTPException(404, "task not found in this project")
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(400, "no fields to update")
    # Route status updates through the plan so cascades happen
    if "status" in fields:
        try:
            st = TaskStatus[str(fields["status"]).upper()]
        except KeyError:
            raise HTTPException(400, f"bad status: {fields['status']}")
        # §4SU Phase 1: the other fields land FIRST — the close is judged on
        # the task as the owner just edited it (it was judged on the old
        # postconditions, then they were overwritten)
        leftover = {k: v for k, v in fields.items()
                    if k not in {"status", "result_summary", "failure_reason"}}
        if leftover:
            try:
                store.update_task(tid, **leftover)
            except ValueError as e:
                raise HTTPException(400, str(e))
        plan = ProjectPlan(store, pid)
        _result = fields.get("result_summary", "") or ""
        _over: list = []
        _was_done = existing.get("status") == "DONE"       # r1 L1: a re-click records nothing again
        if st == TaskStatus.DONE and not _was_done:
            # §4SU Phase 1 (operator 11-02: "close, record why"): the owner's
            # click runs the same close check as every path and always
            # closes; each gate it went over is recorded as the owner's override
            from ..core.task_close import check_close
            # r1 L2: the task's and the project's constraints, as the model's path
            _cons = [str(c) for c in (getattr(plan.tree.nodes.get(tid), "constraints", None) or [])]
            try:
                _cons += [str(c) for c in (((store.get_project(pid) or {}).get("metadata") or {})
                                           .get("constraints") or [])]
            except Exception:  # noqa: BLE001
                pass
            _v = await asyncio.to_thread(check_close, store, pid, plan.tree, tid, actor="owner",
                                         result=_result, constraints=_cons)
            _over = list(_v.overridden)
            if existing.get("wait_code") == "rollup_constraints":
                # r9 L1: the roll-up held it for the constraint audit — the
                # click goes over that audit, recorded like the other gates
                # (§4SU Phase 3: the wait field, not the summary text)
                from ..core.task_close import Refusal
                _cw = str(existing.get("wait_why") or "")
                _over.append(Refusal("constraints", _cw.split(":", 1)[1].strip() if ":" in _cw else _cw))
        def _close_and_record():
            plan.update_status(
                tid, st,
                result=_result,
                failure_reason=fields.get("failure_reason", "") or "",
                # §4SU Phase 0: the owner's own click in the web console IS the
                # explicit human-gate approval (the agent's closes are not)
                human_approved=True,
                # r2 M1: a re-click (or a result edit) of a task already DONE is
                # the owner's close too — never demoted to FAILED
                postconditions_waived=_was_done or any(r.gate == "postconditions" for r in _over),
            )
            if _over and (store.get_task(tid) or {}).get("status") == "DONE":
                for r in _over:
                    try:
                        store.log_event(pid, tid, "close_override",
                                        {"gate": r.gate, "reason": r.reason[:300], "actor": "owner"})
                        store.append_ledger(pid, f"OVERRIDE {r.gate} on task {tid} (by the owner, web console): "
                                                 f"{r.reason[:200]}")
                    except Exception:  # noqa: BLE001
                        logger.warning("owner close override not recorded", exc_info=True)
        # Off-loop: the DONE cascade can complete the project and fire the
        # workspace cleanup sweep (sync filesystem walk) from the store hook.
        # r3 L2: the close AND its override records in ONE thread — a cancel
        # at this await cannot separate them (the thread runs to its end)
        await asyncio.to_thread(_close_and_record)
        _parked = [p for p in (getattr(plan.tree, "rollup_parked", []) or []) if p != tid]
    else:
        try:
            store.update_task(tid, **fields)
        except ValueError as e:
            raise HTTPException(400, str(e))
        _parked = []
    out = dict(store.get_task(tid) or {})
    if _parked:
        # r2 L1: a parent this click parked (gate) or refused (close check)
        out["not_closed"] = [{"task_id": p, "status": "NEEDS_USER",
                              "why": str((store.get_task(p) or {}).get("result_summary") or "")[:300]}
                             for p in _parked]
    return out


@projects_router.delete("/{pid}/tasks/{tid}", status_code=204)
async def delete_task(pid: str, tid: str, request: Request):
    store = _store(request)
    pid = _canon_id(pid)
    existing = store.get_task(tid)
    if not existing or existing["project_id"] != pid:
        raise HTTPException(404, "task not found in this project")
    store.delete_task(tid)
    # 204 No Content MUST have an empty body (RFC 9110). JSONResponse(204,
    # content=None) serialized a 4-byte `null`, which strict HTTP/2 clients
    # and proxies reject.
    return Response(status_code=204)
