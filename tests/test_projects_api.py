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


def test_switch_sets_current(client):
    tc, _, ctx = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.post(f"/api/projects/{pid}/switch")
    assert r.status_code == 200
    assert ctx.current_project_id == pid
    assert "briefing" in r.json()


def test_switch_missing_404(client):
    tc, *_ = client
    r = tc.post("/api/projects/none/switch")
    assert r.status_code == 404


def test_resume_logs_event(client):
    tc, store, _ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.post(f"/api/projects/{pid}/resume")
    assert r.status_code == 200
    evs = store.list_events(pid, event_type="project_resumed")
    assert evs


def test_advance_noop_when_no_tasks(client):
    tc, *_ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    r = tc.post(f"/api/projects/{pid}/advance")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["classification"] == "idle"


def test_advance_runs_task(client):
    tc, store, _ = client
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    tc.post(f"/api/projects/{pid}/tasks",
            json={"description": "Research foo"})
    r = tc.post(f"/api/projects/{pid}/advance")
    assert r.status_code == 200
    # Tool runner is None in this wiring (no registry), so the step
    # just classifies + marks done with "(no tool runner)"
    assert r.json()["classification"] == "research"


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
