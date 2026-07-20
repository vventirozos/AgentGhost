"""Fail-closed fixes for the project advancer (2026-07-20 three-stack review).

Covers:
  H5  — verify fail-open: `classify_verify_result` + the fail-closed runner
        wrapper. A verify PASSES only on an explicit `EXIT CODE: 0` that is
        neither the grep-no-match rewrite nor egress-guard prose; anything
        ambiguous leaves the task NOT DONE (end-to-end through the REAL
        coding executor's `_run_verify`).
  H10 — a runner-less tick must not close a task DONE with
        "(no tool runner)" (theatrical completion).
  MED — blocked/idle ticks stamp `last_autoadvance_ts` so the idle
        round-robin rotates instead of starving on a permanently-blocked
        project; `budget_exhausted` event spam is bounded.
  MED — `record_runtime` fires on FAILED ticks too, so the runtime rail
        sees the wall-clock failed builds burn.
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.coding_executor import build_coding_task
from ghost_agent.core.project_advancer import (
    advance_once, classify_verify_result, _verify_fail_closed_runner,
)
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad


GREP_NO_MATCH = (
    "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\n"
    "(no matches — `grep` exited 1, which for this command means the "
    "pattern/target was NOT FOUND, not that the command failed.)"
)
EGRESS_BLOCKED = (
    "SANDBOX EGRESS BLOCKED (known blind spot — command NOT executed): "
    "this references 127.0.0.1:8000 / :8088, but inside your sandbox that "
    "loopback is the CONTAINER'S OWN, not the host's."
)
CLEAN_PASS = "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nmarker"


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


class _FakeLLM:
    """Returns the same completion content for every chat_completion call."""

    def __init__(self, content: str):
        self.content = content

    async def chat_completion(self, payload, **_kw):
        return {"choices": [{"message": {"content": self.content}}]}


def _spec_json(verify: str) -> str:
    return json.dumps({
        "files": [{"path": "notes.txt", "content": "hello world\n"}],
        "verify": verify,
        "summary": "wrote notes",
        "ledger": "",
    })


def _coding_context(store, verify_output: str):
    """Context + runner for an end-to-end coding tick through the REAL
    build_coding_task: the fake LLM emits a one-file spec with a shell
    verify, the fake runner answers the verify with ``verify_output``."""
    ctx = SimpleNamespace(
        project_store=store,
        llm_client=_FakeLLM(_spec_json("grep -q marker notes.txt")),
        args=SimpleNamespace(model="m"),
        graph_memory=None,
        current_project_id=None,
    )
    calls = []

    async def runner(name, args):
        calls.append((name, args))
        if name == "file_system":
            return f"Successfully wrote {args.get('path')}"
        if name == "execute":
            return verify_output
        return "OK"

    return ctx, runner, calls


# ----------------------------------------------------- verify classifier unit

@pytest.mark.parametrize("out,expected", [
    (CLEAN_PASS, "pass"),
    ("--- EXECUTION RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nok", "pass"),
    ("--- EXECUTION RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\nboom", "fail"),
    ("--- COMMAND RESULT ---\nEXIT CODE: 124\nSTDOUT/STDERR:\n", "fail"),
    # the three fail-open shapes: all must be inconclusive, never pass
    (GREP_NO_MATCH, "inconclusive"),
    (EGRESS_BLOCKED, "inconclusive"),
    ("Session spilled to /workspace/.spill/x.log — output too large",
     "inconclusive"),                       # success-shaped prose, no exit code
    ("", "inconclusive"),
    (None, "inconclusive"),
])
def test_classify_verify_result(out, expected):
    assert classify_verify_result(out) == expected


async def test_fail_closed_runner_rewrites_only_inconclusive_execute():
    async def runner(name, args):
        return {"execute": GREP_NO_MATCH, "file_system": "Successfully wrote x"}[name]

    wrapped = _verify_fail_closed_runner(runner)
    out = await wrapped("execute", {"command": "grep -q marker x"})
    assert out.startswith("ERROR: verify inconclusive")
    # original preserved for retry feedback, minus its `EXIT CODE:` banner —
    # `_looks_like_failure` trusts one found ANYWHERE, so the quoted
    # success-shaped rewrite must not smuggle a pass past the ERROR prefix
    assert "NOT FOUND, not that the command failed" in out
    assert "EXIT CODE:" not in out
    # non-execute tools and conclusive results pass through untouched
    assert await wrapped("file_system", {}) == "Successfully wrote x"

    async def passing(name, args):
        return CLEAN_PASS

    assert await _verify_fail_closed_runner(passing)("execute", {}) == CLEAN_PASS


# ------------------------------------------------- H5 end-to-end via advancer

async def _run_coding_tick(store, verify_output):
    pid = store.create_project("P", kind="CODING")
    tid = store.add_task(pid, "Build the notes file")
    ctx, runner, calls = _coding_context(store, verify_output)
    res = await advance_once(ctx, pid, tool_runner=runner,
                             coding_executor=build_coding_task)
    return pid, tid, res, calls


async def test_grep_no_match_verify_does_not_mark_done(store):
    # `grep -q marker file` with the marker ABSENT comes back as execute.py's
    # friendly `EXIT CODE: 0 … NOT FOUND` rewrite — the verify proved the
    # required text is missing, so the task must NOT close DONE.
    pid, tid, res, calls = await _run_coding_tick(store, GREP_NO_MATCH)
    t = store.get_task(tid)
    assert t["status"] == "FAILED"
    assert "inconclusive" in t["failure_reason"]
    assert any(name == "execute" for name, _ in calls)   # verify actually ran


async def test_egress_prose_verify_does_not_mark_done(store):
    # The egress guard returns success-shaped prose for a command it did NOT
    # execute — nothing was verified.
    pid, tid, res, calls = await _run_coding_tick(store, EGRESS_BLOCKED)
    t = store.get_task(tid)
    assert t["status"] == "FAILED"
    assert "inconclusive" in t["failure_reason"]


async def test_explicit_exit_zero_verify_still_passes(store):
    pid, tid, res, calls = await _run_coding_tick(store, CLEAN_PASS)
    assert store.get_task(tid)["status"] == "DONE"
    assert res.classification == "coding"


async def test_failed_coding_tick_records_runtime(store):
    # The runtime rail was blind to failed builds — the most expensive ticks.
    pid, tid, res, calls = await _run_coding_tick(store, GREP_NO_MATCH)
    meta = store.get_project(pid)["metadata"] or {}
    assert "runtime_used_seconds" in meta
    assert meta.get("tool_call_used") == 1


# --------------------------------------------------------- H10 no tool runner

async def test_no_runner_generated_command_not_done(context, store):
    pid = store.create_project("P", kind="CODING")
    tid = store.add_task(pid, "Build the widget")

    async def gen(desc):
        return "echo hi"

    res = await advance_once(context, pid, tool_runner=None,
                             code_generator=gen)
    assert res.classification == "blocked"
    t = store.get_task(tid)
    assert t["status"] == "PENDING"          # claim released, not DONE
    assert "(no tool runner)" not in (t.get("result_summary") or "")


async def test_no_runner_self_analysis_still_completes(store, tmp_path):
    # Control: an introspective research task needs NO tool — the agent is
    # the source. A runner-less tick may still close it on real output.
    ctx = SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None,
        current_project_id=None,
        llm_client=_FakeLLM("## Memory\nContext window only."),
        args=SimpleNamespace(model="m"),
    )
    pid = store.create_project("Meta")
    tid = store.add_task(pid, "Analyze your own memory architecture")
    res = await advance_once(ctx, pid, tool_runner=None)
    assert store.get_task(tid)["status"] == "DONE"
    assert res.classification == "research"


# ------------------------------------------------- round-robin rotation stamp

async def test_budget_exhausted_tick_stamps_and_bounds_events(context, store):
    pid = store.create_project("P", metadata={"steps_cap": 0})
    store.add_task(pid, "Research x")

    for _ in range(3):
        res = await advance_once(context, pid)
        assert res.classification == "blocked"

    meta = store.get_project(pid)["metadata"] or {}
    assert float(meta.get("last_autoadvance_ts", 0)) > 0
    # one exhaustion episode → ONE event, not one per pick
    assert len(store.list_events(pid, event_type="budget_exhausted")) == 1


async def test_secondary_rail_block_stamps_and_bounds_events(context, store):
    pid = store.create_project("P", metadata={"runtime_cap_seconds": 0})
    store.add_task(pid, "Research x")

    for _ in range(3):
        res = await advance_once(context, pid)
        assert res.classification == "blocked"
        assert "runtime" in res.summary

    meta = store.get_project(pid)["metadata"] or {}
    assert float(meta.get("last_autoadvance_ts", 0)) > 0
    assert len(store.list_events(pid, event_type="budget_exhausted")) == 1


async def test_idle_tick_stamps_last_autoadvance_ts(context, store):
    pid = store.create_project("P")          # no tasks → idle
    res = await advance_once(context, pid)
    assert res.classification == "idle"
    meta = store.get_project(pid)["metadata"] or {}
    assert float(meta.get("last_autoadvance_ts", 0)) > 0


async def test_blocked_project_rotates_out_of_round_robin(context, store):
    # The scheduler picks min(last_autoadvance_ts) over ACTIVE projects. A
    # permanently-blocked project that never re-stamps stays the minimum
    # forever and starves everything else — a blocked tick must rotate it.
    blocked_pid = store.create_project("A", metadata={"steps_cap": 0})
    store.add_task(blocked_pid, "Research a")
    other_pid = store.create_project("B")
    store.add_task(other_pid, "Research b")

    def pick():
        actives = store.list_projects(status_filter="ACTIVE")
        return min(
            actives,
            key=lambda p: float(
                (p.get("metadata") or {}).get("last_autoadvance_ts", 0) or 0
            ),
        )["id"]

    res = await advance_once(context, blocked_pid)
    assert res.classification == "blocked"
    assert pick() == other_pid               # rotation, not starvation


# ------------------------------------------------ runtime rail on tool ticks

async def test_tool_failure_output_records_runtime(context, store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Research x")

    async def runner(name, args):
        return "ERROR: network down"

    await advance_once(context, pid, tool_runner=runner)
    assert store.get_task(tid)["status"] == "FAILED"
    meta = store.get_project(pid)["metadata"] or {}
    assert "runtime_used_seconds" in meta
    assert meta.get("tool_call_used") == 1


async def test_tool_runner_exception_records_runtime(context, store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Research x")

    async def runner(name, args):
        raise RuntimeError("boom")

    await advance_once(context, pid, tool_runner=runner)
    assert store.get_task(tid)["status"] == "FAILED"
    meta = store.get_project(pid)["metadata"] or {}
    assert "runtime_used_seconds" in meta
    assert meta.get("tool_call_used") == 1
