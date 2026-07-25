"""RELEASED lifecycle (2026-07-25): human-attested terminal state.

DONE → RELEASED only via action=release (human command + usage directions +
a deterministic rehearsal); RELEASED is immutable (tool guard + file-write
guard); changes fork a v(n+1) via create_version with precise inheritance
rules; the briefing switches to a runbook-first mode.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore, ProjectStatus
from ghost_agent.core.prompts import build_project_briefing
from ghost_agent.tools.projects import tool_manage_projects, _briefing
from ghost_agent.tools.file_system import _released_write_block
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


def _done_project_with_deliverable(store, title="App"):
    """A DONE GENERAL project with one registered deliverable ON DISK."""
    pid = store.create_project(title, kind="GENERAL", goal="ship a report")
    tid = store.add_task(pid, "write the report")
    ws = Path(store.get_project(pid)["workspace_dir"])
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "report.html").write_text("<h1>report</h1>", encoding="utf-8")
    store.register_file_artifact(tid, "report.html", description="the report")
    store.update_task(tid, status="DONE")
    store.update_project(pid, status="DONE")
    return pid


_DIRECTIONS = ("Open report.html in a browser to view the final report. "
               "It is self-contained; no services are required.")


# ------------------------------------------------------------ release gate

@pytest.mark.asyncio
async def test_release_requires_done(context, store):
    pid = store.create_project("Active", kind="GENERAL", goal="g")
    res = await tool_manage_projects(context, action="release",
                                     project_id=pid, directions=_DIRECTIONS)
    assert "not DONE" in res


@pytest.mark.asyncio
async def test_release_requires_directions(context, store):
    pid = _done_project_with_deliverable(store)
    res = await tool_manage_projects(context, action="release", project_id=pid)
    assert "needs `directions`" in res
    assert store.get_project(pid)["status"] == "DONE"  # unchanged


@pytest.mark.asyncio
async def test_release_happy_path_general_project(context, store):
    pid = _done_project_with_deliverable(store)
    res = await tool_manage_projects(context, action="release",
                                     project_id=pid, directions=_DIRECTIONS)
    data = json.loads(res)
    assert data["status"] == "RELEASED"
    assert store.get_project(pid)["status"] == "RELEASED"
    rel = store.get_release(pid)
    assert rel["directions"].startswith("Open report.html")
    assert rel["deliverables"][0]["path"] == "report.html"
    assert rel["deliverables"][0]["desc"] == "the report"
    # RELEASE.md rendered in the workspace.
    ws = Path(store.get_project(pid)["workspace_dir"])
    text = (ws / "RELEASE.md").read_text(encoding="utf-8")
    assert "How to use" in text and "report.html" in text


@pytest.mark.asyncio
async def test_release_rehearsal_fails_on_missing_deliverable(context, store):
    pid = _done_project_with_deliverable(store)
    ws = Path(store.get_project(pid)["workspace_dir"])
    (ws / "report.html").unlink()  # deliverable vanished
    res = await tool_manage_projects(context, action="release",
                                     project_id=pid, directions=_DIRECTIONS)
    assert "rehearsal FAILED" in res
    assert store.get_project(pid)["status"] == "DONE"  # stays DONE


@pytest.mark.asyncio
async def test_release_service_rehearsal_probes_port(context, store, monkeypatch):
    pid = _done_project_with_deliverable(store, title="Svc")

    class _FakeSup:
        def __init__(self):
            self.restarted = []
        def list_entries(self):
            # workdir deliberately "/workspace" with the project path only
            # in the COMMAND — the live registry shape that the workdir-only
            # matcher missed (2026-07-25 v2 release).
            return [{"name": "svc",
                     "command": f"cd /workspace/projects/{pid} && node server.js",
                     "port": 8123, "workdir": "/workspace"}]
        def restart(self, name):
            self.restarted.append(name)
            return "ok"

    import ghost_agent.tools.projects as tp
    monkeypatch.setattr("ghost_agent.sandbox.services.get_service_supervisor",
                        lambda sm: _FakeSup())
    monkeypatch.setattr(tp, "_probe_tcp", lambda port, **kw: True)
    res = await tool_manage_projects(context, action="release",
                                     project_id=pid, directions=_DIRECTIONS)
    data = json.loads(res)
    assert data["status"] == "RELEASED"
    rel = store.get_release(pid)
    assert rel["services"][0]["name"] == "svc"
    assert rel["urls"] == ["http://127.0.0.1:8123/"]


# ------------------------------------------------------ immutability guards

@pytest.mark.asyncio
async def test_update_cannot_set_released(context, store):
    pid = _done_project_with_deliverable(store)
    res = await tool_manage_projects(context, action="update",
                                     project_id=pid, status="RELEASED")
    assert "action=release" in res
    assert store.get_project(pid)["status"] == "DONE"


@pytest.fixture
def released(context, store):
    pid = _done_project_with_deliverable(store)
    import asyncio
    # asyncio.run (fresh loop) — get_event_loop() broke under the full
    # suite when a prior test left no current loop on this thread.
    res = asyncio.run(
        tool_manage_projects(context, action="release", project_id=pid,
                             directions=_DIRECTIONS))
    assert json.loads(res)["status"] == "RELEASED"
    return pid


@pytest.mark.asyncio
async def test_mutating_actions_refused_on_released(context, store, released):
    for kwargs in (
        {"action": "task_add", "description": "new work"},
        {"action": "ledger", "ledger": "a new fact"},
        {"action": "config", "config_key": "port", "config_value": "9"},
        {"action": "describe_file", "file_path": "report.html",
         "description": "changed"},
        {"action": "autoadvance"},
    ):
        res = await tool_manage_projects(context, project_id=released, **kwargs)
        assert "create_version" in res, kwargs["action"]
    # READ forms still allowed.
    res = await tool_manage_projects(context, action="ledger",
                                     project_id=released)
    assert "create_version" not in res


@pytest.mark.asyncio
async def test_add_task_does_not_reopen_released(store, released):
    store.add_task(released, "sneaky store-level task")
    assert store.get_project(released)["status"] == "RELEASED"


def test_file_write_guard_blocks_released_workspace(store, released):
    blocked = _released_write_block(store, f"/sb/projects/{released}", "x.html")
    assert blocked and "create_version" in blocked
    # Non-released project id in path → no block.
    assert _released_write_block(store, "/sb/projects/aaaabbbbcccc", "x.html") is None
    # No store → never blocks.
    assert _released_write_block(None, f"/sb/projects/{released}", "x") is None


# ----------------------------------------------------------- create_version

@pytest.mark.asyncio
async def test_create_version_only_from_released(context, store):
    pid = _done_project_with_deliverable(store)
    res = await tool_manage_projects(context, action="create_version",
                                     project_id=pid)
    assert "edited IN PLACE" in res


@pytest.mark.asyncio
async def test_create_version_inheritance_and_lineage(context, store, released):
    store2 = store
    # Give the parent some development knowledge to inherit.
    # (ledger/config were writable before release; write directly via store.)
    store2.append_ledger(released, "report is a single self-contained html")
    parent_meta_cfg = {"port": "8100"}
    def _m(meta):
        meta["config"] = parent_meta_cfg
        return meta
    store2._atomic_metadata_update(released, _m)

    res = await tool_manage_projects(context, action="create_version",
                                     project_id=released,
                                     description="add a dark theme")
    data = json.loads(res)
    new_pid = data["project"]["id"]
    assert new_pid != released
    assert data["version"] == 2
    assert data["parent_project_id"] == released

    child = store2.get_project(new_pid)
    assert child["title"].endswith("v2")
    assert child["status"] == "ACTIVE"
    cmeta = child["metadata"]
    # Inherited: ledger, manifest, config (port BUMPED); lineage recorded.
    assert "self-contained html" in cmeta["design_ledger"]
    assert cmeta["config"]["port"] == "8101"
    assert "report.html" in cmeta["file_manifest"]
    assert cmeta["parent_project_id"] == released
    # NOT inherited: release dossier; fresh work_log.
    assert "release" not in cmeta
    assert store2.recent_work_logs(new_pid) == []
    # Files copied, RELEASE.md excluded.
    cws = Path(child["workspace_dir"])
    assert (cws / "report.html").is_file()
    assert not (cws / "RELEASE.md").exists()
    # Copied deliverables re-registered (cleanup keep-set safety).
    assert "report.html" in store2.list_deliverables(new_pid)
    # Seed task carries the change request.
    tasks = store2.list_tasks(new_pid)
    assert any("dark theme" in t["description"] for t in tasks)
    # Parent untouched and still RELEASED.
    assert store2.get_project(released)["status"] == "RELEASED"


# ------------------------------------------------------------ briefing mode

def test_released_briefing_is_runbook_mode(store, released):
    text = build_project_briefing(store, released)
    assert "PROJECT (RELEASED)" in text
    assert "RELEASE DIRECTIONS" in text
    assert "create_version" in text
    # Development scaffolding suppressed.
    assert "ONE TASK AT A TIME" not in text
    assert "NEXT TASK" not in text


def test_tool_briefing_carries_release_key(store, released):
    b = _briefing(store, released)
    assert b["release"]["directions"].startswith("Open report.html")
    assert b["project"]["status"] == "RELEASED"
