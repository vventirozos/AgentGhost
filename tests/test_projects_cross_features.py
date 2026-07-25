"""Final-round cross-project features (2026-07-25): search, dependencies,
clone."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.projects import tool_manage_projects
from ghost_agent.core.project_advancer import advance_once
from ghost_agent.core.prompts import build_project_briefing
from ghost_agent.memory.scratchpad import Scratchpad


@pytest.fixture
def store(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return ProjectStore(tmp_path / "memory", sandbox_root=sandbox)


@pytest.fixture
def context(store, tmp_path):
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None, contradiction_log=None,
        current_project_id=None, llm_client=None, sandbox_manager=None,
    )


# ------------------------------------------------------------------ search

def test_search_finds_by_deliverable_manifest_and_worklog(store):
    p1 = store.create_project("Journal", kind="CODING", goal="training log")
    t1 = store.add_task(p1, "build")
    store.register_file_artifact(t1, "weight-tracker.html",
                                 description="weight chart with canvas")
    p2 = store.create_project("WebOS", kind="CODING", goal="desktop shell")
    store.add_work_log(p2, request="fix the terminal app drag handler",
                       files=["terminal.js"], outcome="completed", note="ok")

    hits = store.search_projects("canvas weight chart")
    assert hits and hits[0]["project_id"] == p1
    assert any(m["kind"] == "manifest" for m in hits[0]["matches"])

    hits2 = store.search_projects("terminal drag")
    assert hits2 and hits2[0]["project_id"] == p2
    assert any(m["kind"] == "work_log" for m in hits2[0]["matches"])

    assert store.search_projects("quantum blockchain nonsense") == []


@pytest.mark.asyncio
async def test_search_action(context, store):
    p1 = store.create_project("Journal", kind="CODING", goal="jiu jitsu training")
    res = await tool_manage_projects(context, action="search",
                                     query="jiu jitsu training")
    data = json.loads(res)
    assert data["hits"][0]["project_id"] == p1
    res2 = await tool_manage_projects(context, action="search", query="")
    assert "query is required" in res2


# ------------------------------------------------------------ dependencies

@pytest.mark.asyncio
async def test_dependency_gates_autoadvance_until_done(context, store):
    lib = store.create_project("SharedLib", kind="CODING", goal="lib")
    store.add_task(lib, "build lib")
    app = store.create_project("App", kind="CODING", goal="app")
    store.add_task(app, "build app")
    res = await tool_manage_projects(context, action="set_dependency",
                                     project_id=app, depends_on=[lib])
    assert json.loads(res)["depends_on"] == [lib]

    r = await advance_once(context, app)
    assert r.classification == "blocked"
    assert "SharedLib" in r.summary

    # Dependency finishes → gate opens (advance proceeds past the dep check;
    # it may fail later for other reasons, but not on the dependency).
    for t in store.list_tasks(lib):
        store.update_task(t["id"], status="DONE")
    assert store.get_project(lib)["status"] == "DONE"
    r2 = await advance_once(context, app)
    assert "dependency" not in (r2.summary or "")


@pytest.mark.asyncio
async def test_dependency_cycle_rejected(context, store):
    a = store.create_project("A", goal="a")
    b = store.create_project("B", goal="b")
    await tool_manage_projects(context, action="set_dependency",
                               project_id=a, depends_on=[b])
    res = await tool_manage_projects(context, action="set_dependency",
                                     project_id=b, depends_on=[a])
    assert "cycle" in res
    res2 = await tool_manage_projects(context, action="set_dependency",
                                      project_id=a, depends_on=[a])
    assert "cannot depend on itself" in res2


def test_dependency_shown_in_briefing(store):
    lib = store.create_project("SharedLib", goal="lib")
    app = store.create_project("App", goal="app")
    def _m(meta):
        meta["depends_on_projects"] = [lib]
        return meta
    store._atomic_metadata_update(app, _m)
    text = build_project_briefing(store, app)
    assert "DEPENDS ON" in text and "SharedLib" in text


# ------------------------------------------------------------------- clone

@pytest.mark.asyncio
async def test_clone_copies_without_lineage(context, store):
    src = store.create_project("Journal", kind="CODING", goal="training log")
    tid = store.add_task(src, "build")
    ws = Path(store.get_project(src)["workspace_dir"])
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "index.html").write_text("<h1>app</h1>", encoding="utf-8")
    store.register_file_artifact(tid, "index.html", description="the shell")
    store.append_ledger(src, "single-file SPA shell")

    res = await tool_manage_projects(context, action="clone",
                                     project_id=src, title="Climbing Log",
                                     description="adapt for climbing")
    data = json.loads(res)
    new_pid = data["project"]["id"]
    child = store.get_project(new_pid)
    cmeta = child["metadata"]
    assert child["title"] == "Climbing Log"
    # Knowledge carried; NO version/parent lineage; provenance via cloned_from.
    assert "single-file SPA shell" in cmeta["design_ledger"]
    assert "index.html" in cmeta["file_manifest"]
    assert "parent_project_id" not in cmeta and "version" not in cmeta
    assert cmeta["cloned_from"] == src
    # Files copied + re-registered; seed task carries the purpose.
    assert (Path(child["workspace_dir"]) / "index.html").is_file()
    assert "index.html" in store.list_deliverables(new_pid)
    tasks = store.list_tasks(new_pid)
    assert any("climbing" in t["description"].lower() for t in tasks)
    # list_children of the source is EMPTY — clone is not a version fork.
    assert store.list_children(src) == []


@pytest.mark.asyncio
async def test_clone_requires_title(context, store):
    src = store.create_project("S", goal="g")
    res = await tool_manage_projects(context, action="clone", project_id=src)
    assert "title is required" in res


# -------------------------- smoother narration-only revert (2026-07-25)

def test_narration_only_trim_detected():
    from ghost_agent.core.agent import _is_narration_only_trim
    orig = ("Canvas charts: the Jiu Jitsu Journal (weight-tracker.html). "
            "Terminal app: WebOS (terminal.js). "
            "Let me search more specifically for terminal-related content.")
    assert _is_narration_only_trim(
        "Let me search more specifically for terminal-related content.", orig)
    # A real (substantive) trim result is NOT reverted.
    assert not _is_narration_only_trim(
        "Canvas charts were used in the Jiu Jitsu Journal's weight tracker.",
        orig)
    # No-op trims never trigger.
    assert not _is_narration_only_trim(orig, orig)
