"""Tool + briefing surfaces for the file manifest / file history (2026-07-24).

Covers: the `describe_file` and `file_history` manage_projects actions, the
`ProjectStore.file_history` journal slice (mixed path forms, as observed
live), the annotated DELIVERABLES briefing section, and `_briefing`'s
additive `file_map` key.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.core.prompts import build_project_briefing
from ghost_agent.tools.projects import tool_manage_projects, _briefing
from ghost_agent.memory.scratchpad import Scratchpad


@pytest.fixture
def store(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return ProjectStore(tmp_path / "memory", sandbox_root=sandbox)


@pytest.fixture
def pid(store):
    return store.create_project("App", kind="CODING", goal="Ship the app")


@pytest.fixture
def context(store, pid, tmp_path):
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None, contradiction_log=None,
        current_project_id=pid, llm_client=None,
    )


# ------------------------------------------------------ describe_file action

@pytest.mark.asyncio
async def test_describe_file_action_writes_and_reads(context, store, pid):
    res = await tool_manage_projects(
        context, action="describe_file", project_id=pid,
        file_path="server.js", description="Node service on :8100",
        file_role="entrypoint")
    data = json.loads(res)
    assert data.get("described") == "server.js"
    mf = store.get_file_manifest(pid)
    assert mf["server.js"]["desc"] == "Node service on :8100"
    # No description → read the manifest back.
    res2 = await tool_manage_projects(
        context, action="describe_file", project_id=pid)
    data2 = json.loads(res2)
    assert "server.js" in data2.get("file_manifest", {})


@pytest.mark.asyncio
async def test_describe_file_action_requires_path(context):
    res = await tool_manage_projects(
        context, action="describe_file", description="orphan desc")
    assert "file_path is required" in res


@pytest.mark.asyncio
async def test_describe_file_with_path_but_no_desc_steers(context):
    """file_path without description = an INTENDED write — must return an
    instructive error, not silently fall back to the manifest read-back
    (live 2026-07-24: three calls 'returned empty' and nothing was
    recorded while the model believed it had succeeded)."""
    res = await tool_manage_projects(
        context, action="describe_file", file_path="server.js")
    assert "needs `description`" in res
    assert "server.js" in res


# ---------------------------------------------------- file_history (store)

def test_file_history_matches_mixed_path_forms(store, pid):
    """Live payloads mix bare names and absolute /workspace paths — both
    must match after normalization (observed on 6a471d630e81)."""
    store.add_work_log(pid, request="fix the page",
                       files=[f"/workspace/projects/{pid}/index.html"],
                       outcome="verifier:failed", note="AppShell demo bug")
    store.add_work_log(pid, request="polish styles",
                       files=["index.html"], outcome="completed", note="ok")
    store.add_work_log(pid, request="unrelated",
                       files=["server.js"], outcome="completed", note="n/a")
    hist = store.file_history(pid, "index.html")
    assert len(hist) == 2
    assert {h["outcome"] for h in hist} == {"verifier:failed", "completed"}
    assert store.file_history(pid, "nope.js") == []


@pytest.mark.asyncio
async def test_file_history_action_returns_history_and_desc(context, store, pid):
    store.describe_file(pid, "index.html", "single-page UI shell")
    store.add_work_log(pid, request="fix render",
                       files=["index.html"], outcome="completed", note="done")
    res = await tool_manage_projects(
        context, action="file_history", project_id=pid,
        file_path="index.html")
    data = json.loads(res)
    assert data["description"] == "single-page UI shell"
    assert data["shown"] == 1
    assert data["history"][0]["request"].startswith("fix render")


# ------------------------------------------------------- briefing surfaces

def test_briefing_annotates_described_deliverables(store, pid, tmp_path):
    tid = store.add_task(pid, "build")
    # The files must EXIST: since queue #10 the briefing lifts a registered
    # deliverable that is not on disk out of the packed "undescribed" line
    # and marks it ⚠ MISSING, because that packed line reads as a list of
    # things that exist. This fixture registered two files it never wrote —
    # the very shape that defect is about — so it now writes them.
    ws = tmp_path / "sandbox" / "projects" / pid
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "server.js").write_text("//")
    (ws / "notes.txt").write_text("n")
    store.register_file_artifact(tid, "server.js", description="Node service")
    store.register_file_artifact(tid, "notes.txt")  # undescribed
    text = build_project_briefing(store, pid)
    assert "server.js — Node service" in text
    assert "notes.txt" in text
    assert "undescribed" in text  # nudge toward describe_file


def test_tool_briefing_carries_file_map(store, pid):
    tid = store.add_task(pid, "build")
    store.register_file_artifact(tid, "app.css", description="theme styles")
    b = _briefing(store, pid)
    assert b["file_map"]["app.css"]["desc"] == "theme styles"
    assert "app.css" in b["deliverables"]  # bare-path list unchanged
