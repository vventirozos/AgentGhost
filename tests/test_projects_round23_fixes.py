"""Round-2/3 fixes from the 2026-07-25 projects re-evaluation.

R2: service-lifecycle coupling; released-workspace hardening (chmod +
execute-shell guard); runbook-shaped _briefing; task_tree cap; slim get;
kind update fix. R3: unrelease (dossier retained, revision bump on
re-release); verify_release health-check; task_delete.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.projects import tool_manage_projects
from ghost_agent.tools.execute import _released_shell_block
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


async def _released(store, title="App"):
    pid = store.create_project(title, kind="GENERAL", goal="ship")
    tid = store.add_task(pid, "build")
    ws = Path(store.get_project(pid)["workspace_dir"])
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "out.html").write_text("<h1>x</h1>", encoding="utf-8")
    store.register_file_artifact(tid, "out.html")
    store.update_task(tid, status="DONE")
    store.update_project(pid, status="DONE")
    ctx = SimpleNamespace(project_store=store, scratchpad=None,
                          graph_memory=None, contradiction_log=None,
                          current_project_id=pid, llm_client=None,
                          sandbox_manager=None)
    res = await tool_manage_projects(
        ctx, action="release", project_id=pid,
        directions="Open out.html in a browser to view the deliverable.")
    assert json.loads(res)["status"] == "RELEASED"
    return pid


# ---------------------------------------------------- R2b hardening

@pytest.mark.asyncio
async def test_release_makes_workspace_readonly_and_fork_writable(context, store):
    pid = await _released(store)
    ws = Path(store.get_project(pid)["workspace_dir"])
    mode = (ws / "out.html").stat().st_mode
    assert not (mode & stat.S_IWUSR)  # released file is read-only
    # Fork restores writability on the COPY.
    res = await tool_manage_projects(context, action="create_version",
                                     project_id=pid, description="change")
    v2 = json.loads(res)["project"]["id"]
    cws = Path(store.get_project(v2)["workspace_dir"])
    assert (cws / "out.html").stat().st_mode & stat.S_IWUSR


@pytest.mark.asyncio
async def test_hard_delete_removes_readonly_workspace(context, store):
    pid = await _released(store)
    # unrelease is not needed — delete must handle read-only trees itself.
    # (Deletion gate requires user-visibility; bypass via store directly.)
    ws = Path(store.get_project(pid)["workspace_dir"])
    assert ws.exists()
    store.delete_project(pid, hard=True)
    assert not ws.exists()  # rmtree succeeded despite chmod a-w


@pytest.mark.asyncio
async def test_shell_guard_blocks_mutation_allows_read(store):
    pid = await _released(store)
    blocked = _released_shell_block(
        store, f"echo hacked > projects/{pid}/out.html")
    assert blocked and "create_version" in blocked
    assert _released_shell_block(store, f"cat projects/{pid}/out.html") is None
    assert _released_shell_block(store, "rm -rf projects/aaaabbbbcccc/x") is None


# ---------------------------------------------------- R2c views

@pytest.mark.asyncio
async def test_switch_to_released_returns_runbook_briefing(context, store):
    pid = await _released(store)
    res = await tool_manage_projects(context, action="switch", project_id=pid)
    data = json.loads(res)
    b = data["briefing"]
    assert b["release"]["directions"].startswith("Open out.html")
    assert "task_tree" not in b            # dev scaffolding gone
    assert "immutable" in data["note"]     # runbook note, not workspace note
    assert "files you write" not in data["note"].lower()


@pytest.mark.asyncio
async def test_get_returns_slim_summary(context, store):
    pid = await _released(store)
    res = await tool_manage_projects(context, action="get", project_id=pid)
    data = json.loads(res)
    assert data["status"] == "RELEASED"
    assert "metadata" not in data          # no raw blob dump
    assert data["briefing"]["release"]


# ---------------------------------------------------- R2d kind fix

@pytest.mark.asyncio
async def test_update_can_set_kind_back_to_general(context, store):
    pid = store.create_project("K", kind="CODING", goal="g")
    res = await tool_manage_projects(context, action="update",
                                     project_id=pid, kind="GENERAL")
    assert json.loads(res)["updated"]
    assert store.get_project(pid)["kind"] == "GENERAL"


# ---------------------------------------------------- R3 unrelease/verify/task_delete

@pytest.mark.asyncio
async def test_unrelease_then_rerelease_bumps_revision(context, store):
    pid = await _released(store)
    res = await tool_manage_projects(context, action="unrelease",
                                     project_id=pid)
    assert json.loads(res)["status"] == "DONE"
    ws = Path(store.get_project(pid)["workspace_dir"])
    assert (ws / "out.html").stat().st_mode & stat.S_IWUSR  # writable again
    assert store.get_release(pid).get("unreleased_at")      # dossier retained
    # Re-release → revision bumps to 2.
    res2 = await tool_manage_projects(
        context, action="release", project_id=pid,
        directions="Open out.html in a browser to view the deliverable.")
    assert json.loads(res2)["status"] == "RELEASED"
    assert store.get_release(pid)["revision"] == 2


@pytest.mark.asyncio
async def test_verify_release_healthy_and_drift(context, store):
    pid = await _released(store)
    res = await tool_manage_projects(context, action="verify_release",
                                     project_id=pid)
    assert json.loads(res)["healthy"] is True
    # Remove the deliverable → degraded.
    ws = Path(store.get_project(pid)["workspace_dir"])
    import os as _os
    _os.chmod(ws, 0o755)               # dir writable — unlink needs it
    _os.chmod(ws / "out.html", 0o644)
    (ws / "out.html").unlink()
    res2 = await tool_manage_projects(context, action="verify_release",
                                      project_id=pid)
    d2 = json.loads(res2)
    assert d2["healthy"] is False
    assert "Degraded" in d2["note"]
    # Status unchanged — health check never demotes.
    assert store.get_project(pid)["status"] == "RELEASED"


@pytest.mark.asyncio
async def test_task_delete_exposed_and_guarded(context, store):
    pid = store.create_project("T", goal="g")
    tid = store.add_task(pid, "dupe task")
    res = await tool_manage_projects(context, action="task_delete",
                                     task_id=tid)
    assert json.loads(res)["deleted"] == tid
    assert store.get_task(tid) is None
    # Guarded on RELEASED projects.
    rpid = await _released(store)
    rtid = store.add_task(rpid, "x")  # store-level; RELEASED doesn't reopen
    res2 = await tool_manage_projects(context, action="task_delete",
                                      task_id=rtid)
    assert "create_version" in res2
