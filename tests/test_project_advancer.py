"""Tests for the self-advancing project loop (Phase 5)."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import json
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.core.project_advancer import (
    advance_once, classify_task, project_dream_pass,
    AdvanceResult, DEFAULT_STEPS_CAP,
)
from ghost_agent.core.planning import ProjectPlan, TaskStatus
from ghost_agent.tools.projects import tool_manage_projects


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def context(tmp_path, store):
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None,
        current_project_id=None,
    )


# --------------------------------------------------------------------- classifier

@pytest.mark.parametrize("desc,expected", [
    ("Research recent BGE-M3 benchmarks", "research"),
    ("Investigate why the cron job skipped Thursday", "research"),
    ("Compare Postgres 16 vs 17 replication", "research"),
    ("Summarize the design doc", "research"),
    ("Implement the new login flow", "coding"),
    ("Write unit tests for the parser", "coding"),
    ("Refactor the job queue", "coding"),
    ("Fix the off-by-one in pagination", "coding"),
    ("Approve the final migration plan", "needs_user"),
    ("Publish the release notes", "needs_user"),
    ("Xyzzy foo bar baz", "research"),  # default fallback
])
def test_classify_task_buckets(desc, expected):
    assert classify_task(desc) == expected


def test_classify_needs_user_wins_over_coding():
    assert classify_task("Implement and approve the change") == "needs_user"


# --------------------------------------------------------------------- advance_once basics

async def test_advance_once_no_project(context):
    r = await advance_once(context, "nope")
    assert r.ok is False
    assert r.classification == "idle"


async def test_advance_once_idle_when_no_leaf(context, store):
    pid = store.create_project("P")
    r = await advance_once(context, pid)
    assert r.ok is True
    assert r.classification == "idle"


async def test_advance_once_archived_project_blocked(context, store):
    pid = store.create_project("P")
    store.update_project(pid, status="ARCHIVED")
    r = await advance_once(context, pid)
    assert r.classification == "blocked"


# --------------------------------------------------------------------- research / coding paths

async def test_advance_research_task_runs_web_search(context, store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Research BGE-M3 embeddings")
    calls = []

    async def runner(name, args):
        calls.append((name, args))
        return "search results about BGE"

    r = await advance_once(context, pid, tool_runner=runner)
    assert r.ok is True
    assert r.classification == "research"
    assert r.task_id == tid
    assert calls and calls[0][0] == "web_search"
    assert "BGE-M3" in calls[0][1]["query"]
    # Task marked DONE with summary
    assert store.get_task(tid)["status"] == "DONE"
    assert "search results" in store.get_task(tid)["result_summary"]
    # Artifact recorded
    arts = store.list_artifacts(task_id=tid)
    assert arts and arts[0]["kind"] == "tool_call"


async def test_advance_coding_task_with_generator_runs_execute(context, store):
    """With a code_generator, a coding task generates real code and runs
    it via execute — NOT the old no-op comment stub."""
    pid = store.create_project("P", kind="CODING")
    tid = store.add_task(pid, "Implement the login flow")

    seen = {}

    async def runner(name, args):
        seen["name"] = name
        seen["args"] = args
        return "ok"

    async def code_gen(desc):
        return "python3 -c \"print('login flow')\""

    r = await advance_once(context, pid, tool_runner=runner, code_generator=code_gen)
    assert r.classification == "coding"
    assert store.get_task(tid)["actual_tool_used"] == "execute"
    assert seen["name"] == "execute"
    # The generated command is run; the old "# Autoadvance stub" comment
    # is gone for good.
    assert "Autoadvance stub" not in seen["args"].get("command", "")
    assert "print('login flow')" in seen["args"]["command"]


async def test_advance_coding_task_without_generator_degrades_to_research(context, store):
    """Without a code_generator there is no code to run, so a coding task
    now RESEARCHES the task (web_search) instead of executing an inert
    comment that marked it DONE having done nothing."""
    pid = store.create_project("P", kind="CODING")
    tid = store.add_task(pid, "Implement the login flow")

    seen = {}

    async def runner(name, args):
        seen["name"] = name
        seen["args"] = args
        return "ok"

    r = await advance_once(context, pid, tool_runner=runner)
    assert r.classification == "coding"
    assert seen["name"] == "web_search"
    assert store.get_task(tid)["actual_tool_used"] == "web_search"


async def test_advance_needs_user_marks_task(context, store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Approve the production deploy")
    called = []

    async def runner(name, args):
        called.append(name)
        return "should not be called"

    r = await advance_once(context, pid, tool_runner=runner)
    assert r.classification == "needs_user"
    assert store.get_task(tid)["status"] == "NEEDS_USER"
    assert called == []  # never invoked a tool


async def test_advance_tool_failure_marks_task_failed(context, store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Research something")

    async def runner(name, args):
        raise RuntimeError("network down")

    r = await advance_once(context, pid, tool_runner=runner)
    assert r.classification == "research"
    t = store.get_task(tid)
    assert t["status"] == "FAILED"
    assert "network down" in t["failure_reason"]


async def test_advance_without_tool_runner_marks_done_with_no_output(context, store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Research foo")
    r = await advance_once(context, pid, tool_runner=None)
    assert r.ok is True
    assert store.get_task(tid)["status"] == "DONE"
    assert store.get_task(tid)["result_summary"] == "(no tool runner)"


async def test_advance_uses_llm_classifier_when_provided(context, store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "ambiguous task title")

    async def classifier(desc):
        return "coding"

    async def runner(name, args):
        return "done"

    r = await advance_once(context, pid, tool_runner=runner,
                           llm_classifier=classifier)
    assert r.classification == "coding"


async def test_advance_falls_back_when_llm_classifier_crashes(context, store):
    pid = store.create_project("P")
    store.add_task(pid, "Research foo bar")

    async def classifier(desc):
        raise ValueError("offline")

    async def runner(name, args):
        return "ok"

    r = await advance_once(context, pid, tool_runner=runner,
                           llm_classifier=classifier)
    # Heuristic kicked in → research
    assert r.classification == "research"


# --------------------------------------------------------------------- budgets

async def test_budget_increments_on_each_tick(context, store):
    pid = store.create_project("P", metadata={"steps_cap": 3})
    for i in range(3):
        store.add_task(pid, f"Research item {i}")

    async def runner(name, args):
        return "x"

    await advance_once(context, pid, tool_runner=runner)
    await advance_once(context, pid, tool_runner=runner)
    proj = store.get_project(pid)
    assert proj["metadata"]["steps_used"] == 2


async def test_budget_exhaustion_blocks_further_advance(context, store):
    pid = store.create_project("P", metadata={"steps_cap": 1})
    store.add_task(pid, "Research A")
    store.add_task(pid, "Research B")

    async def runner(name, args):
        return "x"

    r1 = await advance_once(context, pid, tool_runner=runner)
    assert r1.classification == "research"
    r2 = await advance_once(context, pid, tool_runner=runner)
    assert r2.classification == "blocked"
    assert "budget" in r2.summary.lower()
    evs = store.list_events(pid, event_type="budget_exhausted")
    assert evs


async def test_default_budget_applied(context, store):
    pid = store.create_project("P")  # no metadata
    store.add_task(pid, "Research X")

    async def runner(name, args):
        return "x"

    await advance_once(context, pid, tool_runner=runner)
    proj = store.get_project(pid)
    assert proj["metadata"]["steps_used"] == 1
    assert proj["metadata"]["steps_cap"] == DEFAULT_STEPS_CAP


# --------------------------------------------------------------------- in-progress guard

async def test_advance_marks_in_progress_before_running(context, store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Research topic")
    seen_status = []

    async def runner(name, args):
        # Snapshot the task state while the tool runs
        seen_status.append(store.get_task(tid)["status"])
        return "ok"

    await advance_once(context, pid, tool_runner=runner)
    assert seen_status == ["IN_PROGRESS"]


# --------------------------------------------------------------------- tool surface

async def test_autoadvance_action_on_tool(context, store):
    pid = store.create_project("P", metadata={"steps_cap": 2})
    store.add_task(pid, "Research something")
    # Switch via tool so context.current_project_id is set
    from ghost_agent.tools.projects import tool_manage_projects as t
    await t(context, action="switch", project_id=pid)

    # Without a real registry wiring, the tool will attempt
    # get_available_tools — with our minimal SimpleNamespace context
    # it may succeed or fail; both are OK as long as the advancer
    # runs and the action doesn't crash.
    res_json = await t(context, action="autoadvance")
    data = json.loads(res_json)
    assert "ok" in data
    assert "summary" in data


# --------------------------------------------------------------------- dream pass

def test_project_dream_pass_writes_digest(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "x")
    store.log_event(pid, tid, "autoadvance_step", {"tool": "web_search"})
    n = project_dream_pass(store)
    assert n == 1
    digests = store.list_events(pid, event_type="dream_digest")
    assert digests and digests[0]["payload"]["event_count"] >= 1


def test_project_dream_pass_skips_projects_without_events(store):
    store.create_project("quiet")
    assert project_dream_pass(store) == 0


def test_project_dream_pass_skips_archived_projects(store):
    pid = store.create_project("P")
    store.log_event(pid, None, "autoadvance_step", {"tool": "x"})
    store.update_project(pid, status="ARCHIVED")
    assert project_dream_pass(store) == 0


def test_project_dream_pass_accepts_llm_summarizer(store):
    pid = store.create_project("P")
    store.log_event(pid, None, "autoadvance_step", {"tool": "x"})
    summarized = project_dream_pass(
        store, llm_summarize=lambda evs: f"processed {len(evs)} events"
    )
    assert summarized == 1
    ev = store.list_events(pid, event_type="dream_digest")[0]
    assert "processed" in ev["payload"]["summary"]


def test_project_dream_pass_handles_nil_store():
    assert project_dream_pass(None) == 0
