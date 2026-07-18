"""Read-side plumbing for previously write-only project data (2026-07-18).

Five records were persisted but never reached the LLM:
  * deliverable `file` artifacts (the project's own manifest — cleanup-only)
  * `tool_call` / `note` / `url` artifact payloads (no read action existed)
  * TaskTree.generate_retrospective() (no caller)
  * `dream_digest` events (reachable only via a manual event_log call)
  * tasks.actual_cost (no writer, no reader)

Covers: ProjectStore.list_deliverables, the `artifact_list` tool action,
the DELIVERABLES / RETROSPECTIVE / LAST DREAM DIGEST briefing sections,
retrospective cost totals, advancer cost-stamp wiring pins, and the
status-snapshot keys.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.core.planning import ProjectPlan
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
    return store.create_project("Game", kind="CODING", goal="Build the game")


@pytest.fixture
def context(store, pid, tmp_path):
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None, contradiction_log=None,
        current_project_id=pid, llm_client=None,
    )


# ---------------------------------------------------------- deliverables

def test_list_deliverables_dedupes_and_preserves_order(store, pid):
    t1 = store.add_task(pid, "Create parser")
    t2 = store.add_task(pid, "Create renderer")
    store.add_artifact(t1, "file", "parser.py")
    store.add_artifact(t1, "file", "output/levels.json")
    store.add_artifact(t2, "file", "parser.py")        # dup across tasks
    store.add_artifact(t2, "note", "not a file")       # other kinds excluded
    assert store.list_deliverables(pid) == ["parser.py", "output/levels.json"]


def test_briefing_renders_deliverables_capped(store, pid):
    tid = store.add_task(pid, "Create files")
    for i in range(15):
        store.add_artifact(tid, "file", f"src/mod{i:02d}.js")
    briefing = build_project_briefing(store, pid, max_deliverables=12)
    assert "DELIVERABLES (15 file(s)" in briefing
    assert "src/mod00.js" in briefing
    assert "(+3 more)" in briefing
    assert "artifact_list" in briefing  # pointer to the detail action


def test_briefing_no_deliverables_no_section(store, pid):
    assert "DELIVERABLES" not in build_project_briefing(store, pid)


# ---------------------------------------------------------- artifact_list

@pytest.mark.asyncio
async def test_artifact_list_project_scope_and_kinds(context, store, pid):
    tid = store.add_task(pid, "Build")
    store.add_artifact(tid, "file", "a.py")
    store.add_artifact(tid, "tool_call", "OUTPUT: " + "z" * 1000)
    out = json.loads(await tool_manage_projects(context, action="artifact_list"))
    arts = out["artifacts"]
    assert out["total"] == 2 and len(arts) == 2
    kinds = {a["kind"] for a in arts}
    assert kinds == {"file", "tool_call"}
    tool_payload = next(a for a in arts if a["kind"] == "tool_call")["payload"]
    assert "truncated" in tool_payload and len(tool_payload) < 500
    file_payload = next(a for a in arts if a["kind"] == "file")["payload"]
    assert file_payload == "a.py"  # paths never truncated


@pytest.mark.asyncio
async def test_artifact_list_task_scope_and_kind_filter(context, store, pid):
    t1 = store.add_task(pid, "One")
    t2 = store.add_task(pid, "Two")
    store.add_artifact(t1, "file", "one.py")
    store.add_artifact(t2, "file", "two.py")
    store.add_artifact(t2, "note", "a decision")
    out = json.loads(await tool_manage_projects(
        context, action="artifact_list", task_id=t2))
    assert out["total"] == 2
    out = json.loads(await tool_manage_projects(
        context, action="artifact_list", task_id=t2, artifact_kind="note"))
    assert out["total"] == 1
    assert out["artifacts"][0]["payload"] == "a decision"


@pytest.mark.asyncio
async def test_artifact_list_respects_limit_keeps_newest(context, store, pid):
    tid = store.add_task(pid, "Many")
    for i in range(6):
        store.add_artifact(tid, "file", f"f{i}.py")
    out = json.loads(await tool_manage_projects(
        context, action="artifact_list", limit=2))
    assert out["total"] == 6 and out["shown"] == 2
    assert [a["payload"] for a in out["artifacts"]] == ["f4.py", "f5.py"]


# --------------------------------------------------------- retrospective

def test_retrospective_includes_cost_total(store, pid):
    t1 = store.add_task(pid, "Fast task")
    t2 = store.add_task(pid, "Slow task")
    store.update_task(t1, status="DONE", result_summary="ok", actual_cost=30.0)
    store.update_task(t2, status="DONE", result_summary="ok", actual_cost=90.5)
    plan = ProjectPlan(store, pid)
    retro = plan.tree.generate_retrospective()
    assert retro["total_actual_cost_s"] == pytest.approx(120.5)


def test_briefing_retrospective_only_for_terminal_status(store, pid):
    t1 = store.add_task(pid, "Build the parser")
    store.update_task(t1, status="DONE", result_summary="ok", actual_cost=120.0)
    t2 = store.add_task(pid, "Wire animations")
    t3 = store.add_task(pid, "Polish the HUD")  # stays PENDING → no rollup
    store.update_task(t2, status="FAILED",
                      failure_reason="Animation 'undefined' not found")
    assert store.get_project(pid)["status"] == "ACTIVE"
    briefing = build_project_briefing(store, pid)   # ACTIVE
    assert "RETROSPECTIVE" not in briefing
    store.update_task(t3, status="DONE", result_summary="ok")
    store.update_project(pid, status="DONE")
    briefing = build_project_briefing(store, pid)
    assert "RETROSPECTIVE" in briefing
    assert "Wire animations" in briefing
    assert "Animation 'undefined' not found" in briefing
    assert "measured effort 2 min" in briefing


# ---------------------------------------------------------- dream digest

def test_briefing_shows_latest_dream_digest(store, pid):
    store.log_event(pid, None, "dream_digest",
                    {"event_count": 9, "summary": "Built 4 modules; renderer wired."})
    store.log_event(pid, None, "dream_digest", {"event_count": 2})
    briefing = build_project_briefing(store, pid)
    assert "LAST DREAM DIGEST: 2 events consolidated" in briefing


def test_briefing_no_dream_digest_no_line(store, pid):
    assert "LAST DREAM DIGEST" not in build_project_briefing(store, pid)


# ------------------------------------------------------- status snapshot

def test_status_snapshot_carries_new_keys(store, pid):
    tid = store.add_task(pid, "Build")
    store.add_artifact(tid, "file", "a.py")
    store.update_task(tid, status="DONE", result_summary="ok")
    store.update_project(pid, status="DONE")
    store.log_event(pid, None, "dream_digest", {"event_count": 3})
    snap = _briefing(store, pid)
    assert snap["deliverables"] == ["a.py"]
    assert snap["retrospective"] and snap["retrospective"]["completed"] == 1
    assert snap["last_dream_digest"] == {"event_count": 3}


def test_status_snapshot_retrospective_none_while_active(store, pid):
    snap = _briefing(store, pid)
    assert snap["retrospective"] is None


# ------------------------------------------------------------ wiring pins

def test_advancer_stamps_actual_cost_on_both_finalize_paths():
    import inspect
    import ghost_agent.core.project_advancer as adv
    src = inspect.getsource(adv)
    assert src.count("actual_cost=") >= 3  # coding DONE + coding FAILED + tool DONE
    fin = inspect.getsource(adv._finalize_coding)
    assert "actual_cost" in fin
