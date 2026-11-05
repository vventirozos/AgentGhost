"""Integration tests for the /api/projects HTTP endpoints."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.api.projects_routes import projects_router


@pytest.fixture
def client(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    scratchpad = Scratchpad(persist_path=tmp_path / "sp.db")

    # Minimal agent/context that satisfies verify_api_key + the routes.
    args = SimpleNamespace(api_key="")  # empty → auth disabled
    context = SimpleNamespace(
        args=args,
        project_store=store,
        scratchpad=scratchpad,
        graph_memory=None,
        current_project_id=None,
    )
    agent = SimpleNamespace(context=context)

    app = FastAPI()
    app.state.agent = agent
    app.include_router(projects_router)
    return TestClient(app), store, context


def test_list_projects_empty(client):
    tc, *_ = client
    r = tc.get("/api/projects")
    assert r.status_code == 200
    assert r.json()["projects"] == []
    assert r.json()["current"] is None


def test_create_and_get_project(client):
    tc, *_ = client
    r = tc.post("/api/projects", json={"title": "Build CLI", "kind": "CODING",
                                       "goal": "Ship v1"})
    assert r.status_code == 201
    pid = r.json()["id"]
    g = tc.get(f"/api/projects/{pid}")
    assert g.status_code == 200
    assert g.json()["title"] == "Build CLI"


def test_create_rejects_empty_title(client):
    tc, *_ = client
    r = tc.post("/api/projects", json={"title": ""})
    # Pydantic rejects min_length=1 with 422
    assert r.status_code in (400, 422)


def test_get_missing_returns_404(client):
    tc, *_ = client
    r = tc.get("/api/projects/not-a-real-id")
    assert r.status_code == 404


def test_list_filters_by_status(client):
    tc, *_ = client
    a = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    b = tc.post("/api/projects", json={"title": "B"}).json()["id"]
    # Archive A
    tc.delete(f"/api/projects/{a}")
    lst = tc.get("/api/projects?status=ACTIVE").json()["projects"]
    assert all(p["id"] != a for p in lst)


def test_update_project(client):
    tc, *_ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.patch(f"/api/projects/{pid}", json={"goal": "new goal",
                                               "status": "PAUSED"})
    assert r.status_code == 200
    assert r.json()["goal"] == "new goal"
    assert r.json()["status"] == "PAUSED"


def test_update_returns_404_for_missing(client):
    tc, *_ = client
    r = tc.patch("/api/projects/none", json={"goal": "x"})
    assert r.status_code == 404


def test_update_rejects_empty_body(client):
    tc, *_ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.patch(f"/api/projects/{pid}", json={})
    assert r.status_code == 400


def test_delete_archives(client):
    tc, store, _ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.delete(f"/api/projects/{pid}")
    assert r.status_code == 204
    assert store.get_project(pid)["status"] == "ARCHIVED"


def test_delete_clears_current_if_matching(client):
    tc, _, ctx = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    ctx.current_project_id = pid
    tc.delete(f"/api/projects/{pid}")
    assert ctx.current_project_id is None


def test_delete_missing_returns_404(client):
    tc, *_ = client
    r = tc.delete("/api/projects/none")
    assert r.status_code == 404


def test_switch_sets_that_sessions_focus(client):
    """§4SX B0 (r1 #1): the console switch acts on the conversation named by
    session_id — never the shared context (which carries the last chat)."""
    tc, store, ctx = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.post(f"/api/projects/{pid}/switch", params={"session_id": "s1"})
    assert r.status_code == 200
    assert "briefing" in r.json()
    assert store.get_focus("session:s1") == pid
    assert ctx.current_project_id is None
    assert tc.get("/api/projects", params={"session_id": "s1"}).json()["current"] == pid
    assert tc.get("/api/projects", params={"session_id": "s2"}).json()["current"] is None
    assert tc.get("/api/projects").json()["current"] is None


def test_switch_takes_the_session_in_the_body_too(client):
    tc, store, _ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    assert tc.post(f"/api/projects/{pid}/switch", json={"session_id": "slack:C1:T1"}).status_code == 200
    assert store.get_focus("slack:C1:T1") == pid


def test_switch_without_a_session_is_refused_and_changes_nothing(client):
    tc, store, _ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.post(f"/api/projects/{pid}/switch")
    assert r.status_code == 400 and "session_id" in r.json()["detail"]
    assert tc.post(f"/api/projects/{pid}/resume").status_code == 400
    assert store.focus_rows() == []


def test_two_sessions_switch_independently_from_the_console(client):
    tc, store, _ = client
    a = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    b = tc.post("/api/projects", json={"title": "B"}).json()["id"]
    tc.post(f"/api/projects/{a}/switch", params={"session_id": "s1"})
    tc.post(f"/api/projects/{a}/switch", params={"session_id": "s2"})
    tc.post(f"/api/projects/{b}/switch", params={"session_id": "s2"})
    assert store.get_focus("session:s1") == a and store.get_focus("session:s2") == b


def test_switch_missing_404(client):
    tc, *_ = client
    r = tc.post("/api/projects/none/switch", params={"session_id": "s1"})
    assert r.status_code == 404


def test_resume_logs_event(client):
    tc, store, _ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.post(f"/api/projects/{pid}/resume", params={"session_id": "s1"})
    assert r.status_code == 200
    evs = store.list_events(pid, event_type="project_resumed")
    assert evs
    assert store.get_focus("session:s1") == pid


def _fake_tools_map(monkeypatch, tools):
    """Point the route's lazy `get_available_tools` import at a canned map."""
    import ghost_agent.tools.registry as registry
    monkeypatch.setattr(registry, "get_available_tools", lambda _ctx: tools)


def test_advance_noop_when_no_tasks(client, monkeypatch):
    tc, *_ = client
    async def _search(**kwargs):
        return "results"
    _fake_tools_map(monkeypatch, {"web_search": _search})
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.post(f"/api/projects/{pid}/advance")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["classification"] == "idle"


def test_advance_runs_task_with_runner(client, monkeypatch):
    tc, store, _ = client
    calls = []
    async def _search(**kwargs):
        calls.append(kwargs)
        return ("Foo is a placeholder term used in programming examples. "
                "It commonly pairs with bar and baz in tutorials.")
    _fake_tools_map(monkeypatch, {"web_search": _search})
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    tid = tc.post(f"/api/projects/{pid}/tasks",
                  json={"description": "Research foo"}).json()["id"]
    r = tc.post(f"/api/projects/{pid}/advance")
    assert r.status_code == 200
    assert r.json()["classification"] == "research"
    # The runner actually executed — no "(no tool runner)" theatrical DONE.
    assert calls
    task = store.get_task(tid)
    assert task["status"] == "DONE"
    assert "(no tool runner)" not in (task.get("result_summary") or "")


def test_advance_refuses_without_tool_runner(client, monkeypatch):
    """No buildable runner → refuse (503), never the classify-only path
    that marks a research leaf DONE having done nothing."""
    tc, store, _ = client
    _fake_tools_map(monkeypatch, {})
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    tid = tc.post(f"/api/projects/{pid}/tasks",
                  json={"description": "Research foo"}).json()["id"]
    r = tc.post(f"/api/projects/{pid}/advance")
    assert r.status_code == 503
    assert store.get_task(tid)["status"] != "DONE"


def test_advance_refuses_when_registry_raises(client, monkeypatch):
    tc, store, _ = client
    import ghost_agent.tools.registry as registry
    def _boom(_ctx):
        raise RuntimeError("no registry here")
    monkeypatch.setattr(registry, "get_available_tools", _boom)
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.post(f"/api/projects/{pid}/advance")
    assert r.status_code == 503


def test_advance_passes_runner_and_pins_events(client, monkeypatch):
    """The route hands advance_once a working tool_runner + executors and
    runs it under pinned_event_project(pid)."""
    tc, _, _ = client
    async def _search(**kwargs):
        return "hit"
    _fake_tools_map(monkeypatch, {"web_search": _search})

    captured = {}
    async def _fake_advance(context, project_id, tool_runner=None, **kw):
        from ghost_agent.workspace.model import _EVENT_PROJECT_OVERRIDE
        captured["project_id"] = project_id
        captured["tool_runner"] = tool_runner
        captured["kw"] = kw
        captured["event_pin"] = _EVENT_PROJECT_OVERRIDE.get()
        if tool_runner is not None:
            captured["runner_output"] = await tool_runner("web_search", {})
        return SimpleNamespace(ok=True, task_id="t1", classification="research",
                               summary="ok", artifact_id=None)
    import ghost_agent.api.projects_routes as pr
    monkeypatch.setattr(pr, "advance_once", _fake_advance)

    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.post(f"/api/projects/{pid.upper()}/advance")
    assert r.status_code == 200
    assert captured["project_id"] == pid          # canonicalized path pid
    assert captured["tool_runner"] is not None
    assert captured["runner_output"] == "hit"
    assert captured["event_pin"] == pid           # events stamp THIS project
    assert captured["kw"].get("coding_executor") is not None


def test_events_endpoint(client):
    tc, *_ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.get(f"/api/projects/{pid}/events")
    assert r.status_code == 200
    types = [e["type"] for e in r.json()["events"]]
    assert "project_created" in types


# --------------------------------------------------------------------- tasks

def test_list_tasks_empty(client):
    tc, *_ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.get(f"/api/projects/{pid}/tasks")
    assert r.status_code == 200
    assert r.json()["tasks"] == []


def test_add_and_list_tasks(client):
    tc, *_ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    t = tc.post(f"/api/projects/{pid}/tasks",
                json={"description": "root"})
    assert t.status_code == 201
    tid = t.json()["id"]
    lst = tc.get(f"/api/projects/{pid}/tasks").json()
    assert lst["tasks"][0]["id"] == tid


def test_add_task_to_missing_project_404(client):
    tc, *_ = client
    r = tc.post("/api/projects/none/tasks",
                json={"description": "x"})
    assert r.status_code == 404


def test_add_task_rejects_empty_description(client):
    tc, *_ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.post(f"/api/projects/{pid}/tasks", json={"description": ""})
    assert r.status_code in (400, 422)


def test_update_task_status_cascades(client):
    tc, store, _ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    tid = tc.post(f"/api/projects/{pid}/tasks",
                  json={"description": "root"}).json()["id"]
    r = tc.patch(f"/api/projects/{pid}/tasks/{tid}",
                 json={"status": "DONE", "result_summary": "ok"})
    assert r.status_code == 200
    assert r.json()["status"] == "DONE"
    assert r.json()["result_summary"] == "ok"


def test_update_task_cross_project_404(client):
    tc, *_ = client
    pid1 = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    pid2 = tc.post("/api/projects", json={"title": "B"}).json()["id"]
    tid = tc.post(f"/api/projects/{pid1}/tasks",
                  json={"description": "x"}).json()["id"]
    r = tc.patch(f"/api/projects/{pid2}/tasks/{tid}",
                 json={"status": "DONE"})
    assert r.status_code == 404


def test_update_task_bad_status(client):
    tc, *_ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    tid = tc.post(f"/api/projects/{pid}/tasks",
                  json={"description": "x"}).json()["id"]
    r = tc.patch(f"/api/projects/{pid}/tasks/{tid}",
                 json={"status": "BOGUS"})
    assert r.status_code == 400


def test_delete_task(client):
    tc, store, _ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    tid = tc.post(f"/api/projects/{pid}/tasks",
                  json={"description": "x"}).json()["id"]
    r = tc.delete(f"/api/projects/{pid}/tasks/{tid}")
    assert r.status_code == 204
    assert store.get_task(tid) is None


def test_delete_task_cross_project_404(client):
    tc, *_ = client
    pid1 = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    pid2 = tc.post("/api/projects", json={"title": "B"}).json()["id"]
    tid = tc.post(f"/api/projects/{pid1}/tasks",
                  json={"description": "x"}).json()["id"]
    r = tc.delete(f"/api/projects/{pid2}/tasks/{tid}")
    assert r.status_code == 404


# ------------------------------------------------- _set_current + canon pid

def test_delete_of_a_focused_project_clears_every_row(client):
    """Deleting a project clears every conversation's focus on it — a stale
    pointer made the sandbox recreate projects/<deleted-id>/."""
    tc, store, ctx = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    tc.post(f"/api/projects/{pid}/switch", params={"session_id": "s1"})
    tc.post(f"/api/projects/{pid}/switch", params={"session_id": "s2"})
    r = tc.delete(f"/api/projects/{pid}", params={"hard": "true"})
    assert r.status_code == 204
    assert store.get_project(pid) is None
    assert store.focus_rows() == []


def test_case_mangled_pid_resolves(client):
    """Store ids are canonical (strip+lowercase); a case-mangled path pid
    must resolve on every route instead of 404ing on ownership checks."""
    tc, _, ctx = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    mangled = pid.upper()
    assert tc.get(f"/api/projects/{mangled}").status_code == 200
    tid = tc.post(f"/api/projects/{mangled}/tasks",
                  json={"description": "x"}).json()["id"]
    r = tc.patch(f"/api/projects/{mangled}/tasks/{tid}",
                 json={"status": "DONE"})
    assert r.status_code == 200
    r = tc.delete(f"/api/projects/{mangled}/tasks/{tid}")
    assert r.status_code == 204
    # switch stores the CANONICAL id, never the raw path casing (a raw-cased
    # current_project_id diverges the workspace path on a case-sensitive FS).
    r = tc.post(f"/api/projects/{mangled}/switch", params={"session_id": "s1"})
    assert r.status_code == 200
    assert tc.app.state.agent.context.project_store.get_focus("session:s1") == pid


def test_case_mangled_pid_delete(client):
    tc, store, _ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.delete(f"/api/projects/{pid.upper()}", params={"hard": "true"})
    assert r.status_code == 204
    assert store.get_project(pid) is None
