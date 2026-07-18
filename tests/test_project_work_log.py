"""Turn→project work log + its surfacing (2026-07-18).

Before this feature, agent.py never wrote to the project store: everything
depended on the model voluntarily calling task_update, so work done outside
an open task — all post-completion debugging, notably — left zero trace.
The 2026-07-17 overnight session ran 7 debugging requests against the game
project; the store's last event was 21:41 and the briefing kept saying
"DONE, no open tasks" while the game was broken.

Covers:
  * ProjectStore.add_work_log / recent_work_logs (bounds, order, payload)
  * build_project_briefing: RECENT WORK LOG + STUCK TASKS sections
  * tools/projects._briefing: recent_work_log in the status snapshot
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.core.prompts import build_project_briefing


@pytest.fixture
def store(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return ProjectStore(tmp_path / "memory", sandbox_root=sandbox)


@pytest.fixture
def pid(store):
    return store.create_project("Game", kind="CODING", goal="Build the game")


# ------------------------------------------------------------- store API

def test_add_work_log_persists_bounded_payload(store, pid):
    store.add_work_log(
        pid,
        request="fix the game so it actually starts " + "x" * 500,
        files=["index.html", "game.js", "index.html"],  # dup collapses
        tools={"execute": 3, "browser": 2},
        outcome="verifier:passed",
        note="Root cause: startGame wrapper re-showed levelSelect. " + "y" * 500,
    )
    logs = store.recent_work_logs(pid)
    assert len(logs) == 1
    p = logs[0]["payload"]
    assert len(p["request"]) <= ProjectStore.WORK_LOG_REQUEST_CHARS
    assert len(p["note"]) <= ProjectStore.WORK_LOG_NOTE_CHARS
    assert p["files"] == ["game.js", "index.html"]
    assert p["files_truncated"] == 0
    assert p["tools"] == {"execute": 3, "browser": 2}
    assert p["outcome"] == "verifier:passed"


def test_work_log_files_capped_with_truncation_count(store, pid):
    files = [f"f{i:02d}.js" for i in range(20)]
    store.add_work_log(pid, request="r", files=files)
    p = store.recent_work_logs(pid)[0]["payload"]
    assert len(p["files"]) == ProjectStore.WORK_LOG_MAX_FILES
    assert p["files_truncated"] == 20 - ProjectStore.WORK_LOG_MAX_FILES


def test_recent_work_logs_newest_first_and_type_filtered(store, pid):
    store.add_work_log(pid, request="first")
    store.log_event(pid, None, "task_added", {"description": "noise"})
    store.add_work_log(pid, request="second")
    logs = store.recent_work_logs(pid, limit=10)
    assert [w["payload"]["request"] for w in logs] == ["second", "first"]
    assert all(w["type"] == "work_log" for w in logs)


# ------------------------------------------------------- briefing surfacing

def test_briefing_renders_recent_work_log_even_when_project_done(store, pid):
    tid = store.add_task(pid, "Create index.html")
    store.update_task(tid, status="DONE", result_summary="wrote index.html")
    proj = store.get_project(pid)
    # roll-up may or may not have fired; force DONE to model the real case
    store.update_project(pid, status="DONE")
    store.add_work_log(
        pid, request="the game never starts, fix it",
        files=["index.html"], outcome="completed",
        note="Removed the double showScreen(levelSelect) call.",
    )
    briefing = build_project_briefing(store, pid)
    assert "RECENT WORK LOG" in briefing
    assert "the game never starts" in briefing
    assert "index.html" in briefing
    assert "double showScreen" in briefing


def test_briefing_shows_stuck_tasks_with_failure_reason(store, pid):
    tid = store.add_task(pid, "Wire the animation engine")
    store.update_task(tid, status="FAILED",
                      failure_reason="Animation 'undefined' not found every frame")
    briefing = build_project_briefing(store, pid)
    assert "STUCK TASKS" in briefing
    assert "Wire the animation engine" in briefing
    assert "Animation 'undefined' not found" in briefing


def test_briefing_without_work_log_or_stuck_has_no_empty_sections(store, pid):
    briefing = build_project_briefing(store, pid)
    assert "RECENT WORK LOG" not in briefing
    assert "STUCK TASKS" not in briefing


# ------------------------------------------------------- status snapshot

def test_tool_briefing_includes_recent_work_log(store, pid):
    from ghost_agent.tools.projects import _briefing
    store.add_work_log(pid, request="probe", files=["a.js"], outcome="completed",
                       note="found it")
    snap = _briefing(store, pid)
    assert snap["recent_work_log"]
    assert snap["recent_work_log"][0]["request"] == "probe"
    assert snap["recent_work_log"][0]["files"] == ["a.js"]


# ------------------------------------------------- defect-reopen behavior

class _Ctx:
    """Minimal context stub for _note_defect_on_done_project."""
    def __init__(self, store, pid):
        self.project_store = store
        self.current_project_id = pid


def _agent_with(store, pid):
    from unittest.mock import MagicMock, AsyncMock
    from ghost_agent.core.agent import GhostAgent
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    agent = GhostAgent(ctx)
    agent.context = _Ctx(store, pid)
    return agent


def test_defect_report_on_done_project_reopens_and_adds_task(store, pid):
    tid = store.add_task(pid, "Create game.js")
    store.update_task(tid, status="DONE", result_summary="wrote it")
    store.update_project(pid, status="DONE")
    agent = _agent_with(store, pid)

    recorded = agent._note_defect_on_done_project(
        "it still does the same. the game never starts")
    assert recorded is True
    proj = store.get_project(pid)
    assert proj["status"] == "ACTIVE"  # add_task's DONE→ACTIVE semantic
    open_tasks = [t for t in store.list_tasks(pid)
                  if t["status"] in ("PENDING", "READY")]
    assert any(t["description"].startswith("FIX (defect):") for t in open_tasks)
    assert any("game never starts" in t["description"] for t in open_tasks)


def test_second_defect_report_does_not_stack_duplicates(store, pid):
    store.update_project(pid, status="DONE")
    agent = _agent_with(store, pid)
    assert agent._note_defect_on_done_project("the page is blank") is True
    assert agent._note_defect_on_done_project("still blank!!") is False
    defects = [t for t in store.list_tasks(pid)
               if t["description"].startswith("FIX (defect):")]
    assert len(defects) == 1


def test_defect_hook_ignores_active_projects(store, pid):
    agent = _agent_with(store, pid)  # project is ACTIVE
    assert agent._note_defect_on_done_project("the button broke") is False
    assert not [t for t in store.list_tasks(pid)
                if t["description"].startswith("FIX (defect):")]


def test_defect_hook_never_raises_without_store():
    from unittest.mock import MagicMock, AsyncMock
    from ghost_agent.core.agent import GhostAgent
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    agent = GhostAgent(ctx)
    agent.context = type("C", (), {"project_store": None,
                                   "current_project_id": None})()
    assert agent._note_defect_on_done_project("it broke") is False


# ------------------------------------------------- wiring pins (source)

def test_finalize_chain_writes_work_log():
    import inspect
    import ghost_agent.core.agent as agent_mod
    src = inspect.getsource(agent_mod.GhostAgent._finalize_and_return)
    assert "add_work_log" in src
    # verifier-aware outcome + consumed accumulators
    assert "verifier_backfill" in src[src.index("add_work_log") - 2000:
                                      src.index("add_work_log")]
    assert "_project_work_files" in src


def test_dispatch_accumulates_project_work():
    import inspect
    import ghost_agent.core.agent as agent_mod
    src = inspect.getsource(
        agent_mod.GhostAgent._dispatch_and_process_tool_batch)
    assert "_project_work_files" in src
    assert "_project_work_tools" in src


def test_handle_chat_resets_accumulators_and_calls_defect_hook():
    import inspect
    import ghost_agent.core.agent as agent_mod
    src = inspect.getsource(agent_mod.GhostAgent.handle_chat)
    assert "_project_work_files = set()" in src
    assert "_note_defect_on_done_project(lc)" in src
    # defect hook rides the bug-report gate
    assert (src.index("_is_bug_report_intent(lc)")
            < src.index("_note_defect_on_done_project(lc)"))
