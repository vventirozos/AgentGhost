"""Intent-based pacing (rec 3): classify a "proceed" directive into a count
and run a BOUNDED advance_many loop with correct stop conditions.

A single "proceed" stays a full agent turn; "do the next N" / "proceed with
all" route to the autonomous loop. These tests cover the intent mapping, the
count coercion, the loop's stop logic (isolated from advance_once via a
scripted fake), and the tool path end-to-end on a real store.
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import project_advancer as PA
from ghost_agent.core.project_advancer import (
    classify_advance_intent, advance_many, AdvanceResult,
)
from ghost_agent.memory.projects import ProjectStore, ProjectStatus
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import (
    tool_manage_projects, _parse_advance_count, _advance_batch_instruction,
)
from ghost_agent.core.project_advancer import AdvanceManyResult


# --------------------------------------------------------------- intent classify

@pytest.mark.parametrize("text,mode,count", [
    ("proceed", "one", 1),
    ("next task", "one", 1),
    ("continue", "one", 1),
    ("", "one", 1),
    ("looks good, go ahead", "one", 1),
    ("do the next 3 tasks", "n", 3),
    ("two more tasks please", "n", 2),
    ("advance 5 tasks", "n", 5),
    ("proceed with all remaining tasks", "all", None),
    ("finish the project", "all", None),
    ("just do everything", "all", None),
    ("keep going until done", "all", None),
    ("do the rest", "all", None),
])
def test_classify_advance_intent(text, mode, count):
    got = classify_advance_intent(text)
    assert got["mode"] == mode
    assert got["count"] == count


def test_single_next_task_is_not_all():
    # "next task" must be ONE, not swept up by the 'tasks?' number regex.
    assert classify_advance_intent("the next task")["count"] == 1


def test_batch_instruction_on_failure_says_stop_not_rebuild():
    # A failed/paused batch must tell the agent to report + hand back to the
    # user — NOT to keep building manually (that thrashed ~700s live and broke
    # project scoping).
    b = AdvanceManyResult([{"status": "DONE"}, {"status": "FAILED"}], "failed", None)
    instr = _advance_batch_instruction(b, None)
    assert "STOP" in instr
    assert "Do NOT keep building" in instr or "do not keep building" in instr.lower()
    assert "yourself as a focused turn" not in instr  # the old, removed advice


def test_batch_instruction_on_completion_is_plain_summary():
    b = AdvanceManyResult([{"status": "DONE"}, {"status": "DONE"}], "project_done", None)
    instr = _advance_batch_instruction(b, None)
    assert "Summarize" in instr
    assert "Do NOT keep building" not in instr


@pytest.mark.parametrize("raw,expected", [
    (None, 1), ("", 1), ("1", 1), ("3", 3), (3, 3),
    ("all", None), ("ALL", None), ("*", None), ("rest", None),
    ("garbage", 1), ("0", 1), ("-2", 1),
])
def test_parse_advance_count(raw, expected):
    assert _parse_advance_count(raw) == expected


# --------------------------------------------------------------- advance_many loop

class _FakeStore:
    def __init__(self, task_status=None, project_status="ACTIVE"):
        self.task_status = task_status or {}
        self.project_status = project_status

    def get_task(self, tid):
        return {"status": self.task_status.get(tid)}

    def get_project(self, pid):
        return {"status": self.project_status}


def _script(monkeypatch, results):
    seq = list(results)

    async def fake_advance_once(context, project_id, **kw):
        return seq.pop(0)

    monkeypatch.setattr(PA, "advance_once", fake_advance_once)


@pytest.mark.asyncio
async def test_advance_many_count_reached(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore(
        {"t1": "DONE", "t2": "DONE", "t3": "DONE"}))
    _script(monkeypatch, [
        AdvanceResult(True, "t1", "coding", "ok"),
        AdvanceResult(True, "t2", "coding", "ok"),
        AdvanceResult(True, "t3", "coding", "ok"),
        AdvanceResult(True, "t4", "coding", "ok"),  # extra, should not run
    ])
    r = await advance_many(ctx, "p", max_tasks=3)
    assert r.count == 3
    assert r.stop_reason == "count_reached"


@pytest.mark.asyncio
async def test_advance_many_all_until_done(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore({"t1": "DONE", "t2": "DONE"}))
    _script(monkeypatch, [
        AdvanceResult(True, "t1", "research", "ok"),
        AdvanceResult(True, "t2", "research", "ok"),
        AdvanceResult(True, None, "idle", "no ready leaf"),
    ])
    r = await advance_many(ctx, "p", max_tasks=None)
    assert r.count == 2
    assert r.stop_reason == "project_done"


@pytest.mark.asyncio
async def test_advance_many_stops_on_needs_user(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore({"t1": "DONE", "t2": "NEEDS_USER"}))
    _script(monkeypatch, [
        AdvanceResult(True, "t1", "coding", "ok"),
        AdvanceResult(True, "t2", "needs_user", "needs human"),
        AdvanceResult(True, "t3", "coding", "should not run"),
    ])
    r = await advance_many(ctx, "p", max_tasks=None)
    assert r.count == 2
    assert r.stop_reason == "needs_user"


@pytest.mark.asyncio
async def test_advance_many_stops_on_failure(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore({"t1": "DONE", "t2": "FAILED"}))
    _script(monkeypatch, [
        AdvanceResult(True, "t1", "coding", "ok"),
        AdvanceResult(True, "t2", "coding", "tool produced no usable result"),
        AdvanceResult(True, "t3", "coding", "should not run"),
    ])
    r = await advance_many(ctx, "p", max_tasks=None, stop_on_fail=True)
    assert r.count == 2
    assert r.stop_reason == "failed"


@pytest.mark.asyncio
async def test_advance_many_continues_past_failure_when_allowed(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore({"t1": "FAILED", "t2": "DONE"}))
    _script(monkeypatch, [
        AdvanceResult(True, "t1", "coding", "failed"),
        AdvanceResult(True, "t2", "coding", "ok"),
        AdvanceResult(True, None, "idle", "done"),
    ])
    r = await advance_many(ctx, "p", max_tasks=None, stop_on_fail=False)
    assert r.count == 2
    # advanced both, but one FAILED → reported distinctly from a clean finish
    assert r.stop_reason == "completed_with_failures"


@pytest.mark.asyncio
async def test_advance_many_circuit_breaker_on_repeated_failures(monkeypatch):
    # stop_on_fail=False continues past failures, but 3 in a row trips the
    # circuit breaker (systemic problem) — don't grind through everything.
    ctx = SimpleNamespace(project_store=_FakeStore(
        {"t1": "FAILED", "t2": "FAILED", "t3": "FAILED", "t4": "DONE"}))
    _script(monkeypatch, [
        AdvanceResult(True, "t1", "coding", "x"),
        AdvanceResult(True, "t2", "coding", "x"),
        AdvanceResult(True, "t3", "coding", "x"),
        AdvanceResult(True, "t4", "coding", "should not run"),
    ])
    r = await advance_many(ctx, "p", max_tasks=None, stop_on_fail=False,
                           max_consecutive_fails=3)
    assert r.count == 3
    assert r.stop_reason == "repeated_failures"


@pytest.mark.asyncio
async def test_advance_many_consecutive_counter_resets_on_success(monkeypatch):
    # fail, succeed, fail, succeed, done → 2 isolated failures, never 3 in a
    # row → completes with failures (circuit breaker NOT tripped).
    ctx = SimpleNamespace(project_store=_FakeStore(
        {"t1": "FAILED", "t2": "DONE", "t3": "FAILED", "t4": "DONE"}))
    _script(monkeypatch, [
        AdvanceResult(True, "t1", "coding", "x"),
        AdvanceResult(True, "t2", "coding", "x"),
        AdvanceResult(True, "t3", "coding", "x"),
        AdvanceResult(True, "t4", "coding", "x"),
        AdvanceResult(True, None, "idle", "done"),
    ])
    r = await advance_many(ctx, "p", max_tasks=None, stop_on_fail=False,
                           max_consecutive_fails=3)
    assert r.count == 4
    assert r.stop_reason == "completed_with_failures"


@pytest.mark.asyncio
async def test_advance_many_blocked_budget_vs_done(monkeypatch):
    # blocked + project still ACTIVE → budget/inactive
    ctx = SimpleNamespace(project_store=_FakeStore(project_status="ACTIVE"))
    _script(monkeypatch, [AdvanceResult(True, None, "blocked", "budget")])
    r = await advance_many(ctx, "p", max_tasks=None)
    assert r.stop_reason == "budget_or_inactive"

    # blocked because the last task rolled the project to DONE → project_done
    ctx2 = SimpleNamespace(project_store=_FakeStore(project_status="DONE"))
    _script(monkeypatch, [AdvanceResult(True, None, "blocked", "project is DONE")])
    r2 = await advance_many(ctx2, "p", max_tasks=None)
    assert r2.stop_reason == "project_done"


@pytest.mark.asyncio
async def test_advance_many_hard_cap(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore({"t": "DONE"}))
    # never finishes — every tick returns a DONE task
    _script(monkeypatch, [AdvanceResult(True, "t", "coding", "ok")] * 10)
    r = await advance_many(ctx, "p", max_tasks=None, hard_cap=3)
    assert r.count == 3
    assert r.stop_reason == "hard_cap"


# --------------------------------------------------------------- tool path (real store)

@pytest.fixture
def context(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None, contradiction_log=None,
        current_project_id=None, llm_client=None,
    )


@pytest.fixture
def no_subtools(monkeypatch):
    """Force tool_runner=None inside the autoadvance handler so research tasks
    complete cleanly without invoking real sub-tools (which need a full
    context). Isolates the wiring (tool → advance_many → loop → store)."""
    import ghost_agent.tools.registry as reg
    monkeypatch.setattr(reg, "get_available_tools", lambda ctx: None)


@pytest.mark.asyncio
async def test_tool_autoadvance_all_completes_research_project(context, no_subtools):
    store = context.project_store
    await tool_manage_projects(context, action="create", title="Lit Review")
    pid = context.current_project_id
    await tool_manage_projects(context, action="task_decompose",
                               subtasks=["research topic A", "research topic B"])
    out = json.loads(await tool_manage_projects(
        context, action="autoadvance", count="all"))
    # both research tasks advance (no tool runner → marked DONE) and the loop
    # reports completion
    assert out["requested"] == "all"
    assert out["count"] == 2
    assert out["stop_reason"] == "project_done"
    assert store.get_project(pid)["status"] == ProjectStatus.DONE.value


@pytest.mark.asyncio
async def test_tool_autoadvance_count_one_default(context, no_subtools):
    store = context.project_store
    await tool_manage_projects(context, action="create", title="P")
    pid = context.current_project_id
    await tool_manage_projects(context, action="task_decompose",
                               subtasks=["research a", "research b", "research c"])
    out = json.loads(await tool_manage_projects(context, action="autoadvance"))
    assert out["requested"] == 1
    assert out["count"] == 1
    # two tasks still open
    open_tasks = [t for t in store.list_tasks(pid) if t["status"] != "DONE"]
    assert len(open_tasks) == 2
