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


# ------------------- paraphrased work-state (round 2, same-day refill)

# The three values that refilled the freshly-scrubbed notes.info ring on
# 2026-07-25: paraphrased project state carrying NO 12-hex id and NO
# project title, drained from buffered journal items with no project
# bound — every store-based check missed them.
LIVE_LEAKED_VALUES = [
    "The user requires a robust dark/light theme toggle implementation "
    "that persists state and correctly cycles between modes, as previous "
    "attempts failed verification.",
    "The user requires the dark/light theme toggle to be fully functional, "
    "including persistence and correct state transitions, as previous "
    "attempts were refuted by a verifier.",
    "The user requires a dark/light theme toggle implementation with "
    "persistence, and the system needs to verify the functionality against "
    "specific project IDs and visual evidence.",
]


def test_verification_workflow_phrasing_is_project_state(store):
    a = _agent_with(store)  # nothing bound, no id in the text
    for value in LIVE_LEAKED_VALUES:
        assert a._is_tracked_project_state(value), value


def test_workflow_cue_screens_even_without_a_store():
    a = agent_mod.GhostAgent.__new__(agent_mod.GhostAgent)
    a.context = SimpleNamespace(project_store=None, current_project_id=None)
    assert a._is_tracked_project_state(
        "The user requires X, as previous attempts were refuted by a verifier.")


def test_durable_facts_unaffected_by_workflow_cue(store):
    a = _agent_with(store)
    assert not a._is_tracked_project_state(
        "The user works at EvolMonkey and lives in Athens.")
    # Preference phrasing is exempted BEFORE the workflow cue runs.
    assert not a._is_tracked_project_state(
        "The user prefers packages from verifiers they trust.")


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


# ------------------------------- claim-conditioned evidence (2026-07-25)

def _mk(name, content):
    return {"name": name, "content": content}


def test_claim_pulls_displaced_mid_turn_evidence():
    """Dark-theme reproduction: the 🌙→☀️ observation lives in a mid-turn
    click result, displaced beyond newest-3 by a failed retry tail. The
    claim's tokens must pull it back into evidence."""
    tools = [
        _mk("browser", "clicked #theme-toggle: button now shows sun icon, "
                       "dark theme toggle switched page to light mode css"),
        _mk("browser", "Error: click: runner exit 1 — TargetClosedError"),
        _mk("browser", "navigated: page loaded, nav visible"),
        _mk("browser", "interact: scrolled to footer, no errors"),
    ]
    claim = ("The dark theme toggle works: clicking it switched the page "
             "to light mode and the button shows the sun icon.")
    ev = _collect_verifier_evidence(tools, claim_text=claim)
    assert "sun icon" in ev            # displaced item pulled in
    assert "TargetClosedError" in ev   # newest window intact
    assert len(ev) <= 4000


def test_no_claim_keeps_legacy_newest_window():
    tools = [
        _mk("browser", "old evidence about the sun icon toggle theme"),
        _mk("execute", "OUTPUT: a\nEXIT CODE: 0"),
        _mk("execute", "OUTPUT: b\nEXIT CODE: 0"),
        _mk("execute", "OUTPUT: c\nEXIT CODE: 0"),
    ]
    ev = _collect_verifier_evidence(tools)
    assert "sun icon" not in ev  # positional newest-3 only, as before


def test_irrelevant_older_output_not_pulled():
    tools = [
        _mk("web_search", "weather in paris is sunny and warm today"),
        _mk("execute", "OUTPUT: a\nEXIT CODE: 0"),
        _mk("execute", "OUTPUT: b\nEXIT CODE: 0"),
        _mk("execute", "OUTPUT: c\nEXIT CODE: 0"),
    ]
    claim = "The dark theme toggle switches the journal to light mode."
    ev = _collect_verifier_evidence(tools, claim_text=claim)
    assert "paris" not in ev  # overlap below threshold — not pulled
