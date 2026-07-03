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

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .routes import verify_api_key, get_agent
from ..core.planning import ProjectPlan, TaskStatus
from ..core.project_advancer import advance_once
from ..core.prompts import build_project_briefing

logger = logging.getLogger("GhostAgent")

projects_router = APIRouter(prefix="/api/projects",
                            dependencies=[Security(verify_api_key)])


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


class TaskUpdate(BaseModel):
    description: Optional[str] = None
    status: Optional[str] = None
    result_summary: Optional[str] = None
    failure_reason: Optional[str] = None
    alternatives: Optional[List[str]] = None
    postconditions: Optional[List[str]] = None


def _store(request: Request):
    agent = get_agent(request)
    store = getattr(agent.context, "project_store", None)
    if store is None:
        raise HTTPException(503, "project_store not configured")
    return store


def _context(request: Request):
    return get_agent(request).context


# --------------------------------------------------------------------- projects

@projects_router.get("")
async def list_projects(request: Request, status: Optional[str] = None):
    store = _store(request)
    return {
        "projects": store.list_projects(status_filter=status),
        "current": getattr(_context(request), "current_project_id", None),
    }


@projects_router.post("", status_code=201)
async def create_project(body: ProjectCreate, request: Request):
    store = _store(request)
    try:
        pid = store.create_project(
            title=body.title, kind=body.kind, goal=body.goal,
            metadata=body.metadata,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": pid, "project": store.get_project(pid)}


@projects_router.get("/{pid}")
async def get_project(pid: str, request: Request):
    proj = _store(request).get_project(pid)
    if not proj:
        raise HTTPException(404, "project not found")
    return proj


@projects_router.patch("/{pid}")
async def update_project(pid: str, body: ProjectUpdate, request: Request):
    store = _store(request)
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(400, "no fields to update")
    try:
        ok = store.update_project(pid, **fields)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not ok:
        raise HTTPException(404, "project not found")
    return store.get_project(pid)


@projects_router.delete("/{pid}", status_code=204)
async def delete_project(pid: str, request: Request, hard: bool = False):
    store = _store(request)
    ok = store.delete_project(pid, hard=hard)
    if not ok:
        raise HTTPException(404, "project not found")
    ctx = _context(request)
    if getattr(ctx, "current_project_id", None) == pid:
        ctx.current_project_id = None
    # 204 No Content MUST have an empty body (RFC 9110). JSONResponse(204,
    # content=None) serialized a 4-byte `null`, which strict HTTP/2 clients
    # and proxies reject.
    return Response(status_code=204)


@projects_router.post("/{pid}/switch")
async def switch_project(pid: str, request: Request):
    store = _store(request)
    proj = store.get_project(pid)
    if not proj:
        raise HTTPException(404, "project not found")
    _context(request).current_project_id = pid
    return {"switched_to": pid,
            "briefing": build_project_briefing(store, pid)}


@projects_router.post("/{pid}/resume")
async def resume_project(pid: str, request: Request):
    store = _store(request)
    proj = store.get_project(pid)
    if not proj:
        raise HTTPException(404, "project not found")
    _context(request).current_project_id = pid
    store.log_event(pid, None, "project_resumed", {})
    return {"project": proj,
            "briefing": build_project_briefing(store, pid)}


@projects_router.post("/{pid}/advance")
async def advance_project(pid: str, request: Request):
    ctx = _context(request)
    store = _store(request)
    if not store.get_project(pid):
        raise HTTPException(404, "project not found")
    result = await advance_once(ctx, pid)
    return {
        "ok": result.ok,
        "task_id": result.task_id,
        "classification": result.classification,
        "summary": result.summary,
        "artifact_id": result.artifact_id,
    }


@projects_router.get("/{pid}/events")
async def list_events(pid: str, request: Request,
                      limit: int = 50, type: Optional[str] = None):
    store = _store(request)
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
    if not store.get_project(pid):
        raise HTTPException(404, "project not found")
    return {"tasks": store.list_tasks(pid, status_filter=status)}


@projects_router.post("/{pid}/tasks", status_code=201)
async def add_task(pid: str, body: TaskCreate, request: Request):
    store = _store(request)
    if not store.get_project(pid):
        raise HTTPException(404, "project not found")
    try:
        tid = store.add_task(
            pid, description=body.description, parent_id=body.parent_id,
            dependency_type=body.dependency_type,
            alternatives=body.alternatives,
            postconditions=body.postconditions,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": tid, "task": store.get_task(tid)}


@projects_router.patch("/{pid}/tasks/{tid}")
async def update_task(pid: str, tid: str, body: TaskUpdate, request: Request):
    store = _store(request)
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
        plan = ProjectPlan(store, pid)
        plan.update_status(
            tid, st,
            result=fields.get("result_summary", "") or "",
            failure_reason=fields.get("failure_reason", "") or "",
        )
        # Non-status fields still need direct persistence
        leftover = {k: v for k, v in fields.items()
                    if k not in {"status", "result_summary", "failure_reason"}}
        if leftover:
            store.update_task(tid, **leftover)
    else:
        try:
            store.update_task(tid, **fields)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return store.get_task(tid)


@projects_router.delete("/{pid}/tasks/{tid}", status_code=204)
async def delete_task(pid: str, tid: str, request: Request):
    store = _store(request)
    existing = store.get_task(tid)
    if not existing or existing["project_id"] != pid:
        raise HTTPException(404, "task not found in this project")
    store.delete_task(tid)
    # 204 No Content MUST have an empty body (RFC 9110). JSONResponse(204,
    # content=None) serialized a 4-byte `null`, which strict HTTP/2 clients
    # and proxies reject.
    return Response(status_code=204)
