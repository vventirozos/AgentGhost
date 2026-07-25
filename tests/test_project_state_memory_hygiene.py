"""Fixes from the operator's 2026-07-25 log-entry review.

1. Project-state backstop: smart-memory facts / profile notes about a
   MANAGED project route to the project store, not user memory (observed:
   the profile's 3-slot notes.info churned to 100% project chatter and
   three project-status "facts" stored at 0.90 as timeless user truths).
2. Evidence packer: short bookkeeping confirmations CARRYING a 12-hex id
   are linkage evidence (the task-id-vs-project-id category-error refute).
3. Release seeds config.port from the rehearsed service so create_version's
   port bump always has a source.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ghost_agent.core.agent as agent_mod
from ghost_agent.core.agent import _collect_verifier_evidence
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.projects import tool_manage_projects
from ghost_agent.memory.scratchpad import Scratchpad


@pytest.fixture
def store(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return ProjectStore(tmp_path / "memory", sandbox_root=sandbox)


def _agent_with(store, current=None):
    a = agent_mod.GhostAgent.__new__(agent_mod.GhostAgent)
    a.context = SimpleNamespace(project_store=store,
                                current_project_id=current)
    return a


# ------------------------------------------- project-state backstop

def test_fact_naming_managed_project_id_is_project_state(store):
    pid = store.create_project("Journal", kind="CODING", goal="g")
    a = _agent_with(store)
    assert a._is_tracked_project_state(
        f"The user is working on project {pid} and requires a dark theme.")
    # A 12-hex token that is NOT a managed project → not project state.
    assert not a._is_tracked_project_state(
        "The commit abcdefabcdef fixed the parser.")


def test_bound_project_title_match_is_project_state(store):
    pid = store.create_project("Jiu jitsu Journal", kind="CODING", goal="g")
    a = _agent_with(store, current=pid)
    assert a._is_tracked_project_state(
        "The user has decided to release the Jiu jitsu Journal project.")
    # Same sentence with NO bound project → no title scan → allowed.
    a2 = _agent_with(store, current=None)
    assert not a2._is_tracked_project_state(
        "The user has decided to release the Jiu jitsu Journal project.")


def test_preferences_survive_even_inside_project_work(store):
    pid = store.create_project("Journal", kind="CODING", goal="g")
    a = _agent_with(store, current=pid)
    assert not a._is_tracked_project_state(
        f"The user prefers dark UIs (noted while working on {pid}).")


def test_no_store_never_blocks(store):
    a = agent_mod.GhostAgent.__new__(agent_mod.GhostAgent)
    a.context = SimpleNamespace(project_store=None, current_project_id=None)
    assert not a._is_tracked_project_state("anything at all abcdefabcdef")


# ------------------------------------------- id-bearing evidence

def test_short_id_bearing_confirmation_is_packed():
    tools = [
        {"name": "manage_projects",
         "content": '{"updated": true, "task": "d8a307dd196f"}'},
        {"name": "execute", "content": "OUTPUT: ok\nEXIT CODE: 0"},
    ]
    ev = _collect_verifier_evidence(tools)
    assert "[manage_projects]" in ev
    assert "d8a307dd196f" in ev


def test_short_idless_confirmation_still_excluded():
    tools = [
        {"name": "manage_tasks", "content": '{"exited": "ok"}'},
        {"name": "execute", "content": "OUTPUT: ok\nEXIT CODE: 0"},
    ]
    ev = _collect_verifier_evidence(tools)
    assert "manage_tasks" not in ev


# ------------------------------------------- release seeds config.port

@pytest.mark.asyncio
async def test_release_seeds_port_from_rehearsed_service(store, tmp_path, monkeypatch):
    pid = store.create_project("Svc", kind="CODING", goal="serve")
    tid = store.add_task(pid, "build server")
    ws = Path(store.get_project(pid)["workspace_dir"])
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "server.js").write_text("// srv", encoding="utf-8")
    store.register_file_artifact(tid, "server.js")
    store.update_task(tid, status="DONE")
    store.update_project(pid, status="DONE")

    class _FakeSup:
        def list_entries(self):
            return [{"name": "svc", "command": "node server.js",
                     "port": 8144, "workdir": f"/workspace/projects/{pid}"}]
        def restart(self, name):
            return "ok"

    import ghost_agent.tools.projects as tp
    monkeypatch.setattr("ghost_agent.sandbox.services.get_service_supervisor",
                        lambda sm: _FakeSup())
    monkeypatch.setattr(tp, "_probe_tcp", lambda port, **kw: True)
    context = SimpleNamespace(project_store=store,
                              scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                              graph_memory=None, contradiction_log=None,
                              current_project_id=pid, llm_client=None,
                              sandbox_manager=None)
    res = await tool_manage_projects(
        context, action="release", project_id=pid,
        directions="Run the service and open the page in your browser to use it.")
    assert json.loads(res)["status"] == "RELEASED"
    # The rehearsed port is now durable config — the fork's bump source.
    assert store.get_config(pid)["port"] == "8144"
