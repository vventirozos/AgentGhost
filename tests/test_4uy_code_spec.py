"""§4UY C18 — the spec executor (and the agentic leaf routed from it) is gone.

Before: when a project task's agent turn could not run (task turns switched
off, no model / store, or the turn's prompt could not be built), the advancer
fell back to `coding_executor.build_coding_task` — a 3,272-line one-call spec
builder last seen in the live log on 2026-11-07, whose pins drove a path
production no longer took. Now the production executor is a MARKER: a coding
task wired with it runs as an agent turn or goes back to the queue, and the
marker itself builds nothing. Each test names the world it fails in.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ghost_agent.core import project_turns as PT
from ghost_agent.core.project_advancer import TASK_TURN_UNAVAILABLE_PREFIX, advance_many, advance_once
from ghost_agent.memory.projects import ProjectStore


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def _ctx(store):
    return SimpleNamespace(project_store=store, llm_client=SimpleNamespace(foreground_requests=0),
                           current_project_id=None, scratchpad=None, graph_memory=None,
                           workspace_model=None, last_user_content="")


def _recording_marked_executor(calls, idle=False):
    """A production-MARKED executor that records any call — the world where the
    advancer still falls back to a builder makes `calls` non-empty."""
    async def _build(*a, **kw):
        calls.append(kw)
        from ghost_agent.core.coding_executor import CodingResult
        return CodingResult(True, "built by a fallback", files=["parser.py"], verified=True)
    return PT.mark_real_executor(_build, idle=idle)


async def _coding(_d):
    return "coding"


def _project(store):
    pid = store.create_project("Parser", kind="CODING", goal="a parser")
    tid = store.add_task(pid, "Build the parser module → parser.py")
    return pid, tid


@pytest.mark.parametrize("idle", [False, True])
async def test_turns_off_no_builder_runs_and_the_task_is_reopened(store, idle):
    # world: the advancer falls back to the executor (calls == [1 build]), or
    # leaves the claim IN_PROGRESS, or marks it DONE / FAILED on nothing
    pid, tid = _project(store)
    calls = []
    res = await advance_once(_ctx(store), pid, owner_requested=True, tool_runner=lambda *a, **k: None,
                             llm_classifier=_coding, coding_executor=_recording_marked_executor(calls, idle))
    assert calls == []
    assert res.classification == "blocked" and res.summary.startswith(TASK_TURN_UNAVAILABLE_PREFIX)
    assert res.worked is False                       # nothing ran: never charged as work
    assert store.get_task(tid)["status"] == "PENDING"


async def test_an_unbuildable_turn_does_not_fall_back_either(store, monkeypatch):
    # world: a turn whose prompt could not be built hands the task to a builder
    monkeypatch.setenv("GHOST_PROJECT_TASK_TURNS", "1")

    def _bad(*a, **k):
        raise ValueError("no prompt")
    monkeypatch.setattr(PT, "task_turn_prompt", _bad)
    pid, tid = _project(store)
    calls = []
    res = await advance_once(_ctx(store), pid, owner_requested=True, tool_runner=lambda *a, **k: None,
                             llm_classifier=_coding, coding_executor=_recording_marked_executor(calls))
    assert calls == [] and res.summary.startswith(TASK_TURN_UNAVAILABLE_PREFIX)
    assert store.get_task(tid)["status"] == "PENDING"


async def test_an_unmarked_executor_is_still_called(store):
    # the guard is the MARK, not "never call an executor": a caller-supplied
    # build step (the advancer's own tests) still builds. World: the release
    # branch swallowed every coding task.
    pid, tid = _project(store)
    calls = []

    async def _stub(*a, **kw):
        calls.append(kw)
        from ghost_agent.core.coding_executor import CodingResult
        return CodingResult(False, "stub build failed", files=[])
    res = await advance_once(_ctx(store), pid, owner_requested=True, tool_runner=lambda *a, **k: None,
                             llm_classifier=_coding, coding_executor=_stub)
    assert len(calls) == 1 and not res.summary.startswith(TASK_TURN_UNAVAILABLE_PREFIX)


def test_an_owner_batch_stops_at_the_first_unrunnable_task(store):
    # world: the batch read the reopen as a budget stop, or re-picked the same
    # task in a loop (every tick would reopen it again)
    pid, tid = _project(store)
    store.add_task(pid, "Build the printer → printer.py")
    calls = []

    async def _runner(*_a, **_k):
        return "OK"
    res = asyncio.run(advance_many(_ctx(store), pid, max_tasks=None, owner_requested=True, tool_runner=_runner,
                                   llm_classifier=_coding, coding_executor=_recording_marked_executor(calls),
                                   stop_on_fail=False))
    assert calls == []
    assert res.stop_reason == "step_crashed" and len(res.advanced) == 1
    assert res.advanced[0]["status"] == "PENDING"


async def test_the_production_marker_builds_nothing():
    # world: `build_coding_task` still runs a spec build (calls the model or
    # the tool runner) when someone calls it directly
    from ghost_agent.core.coding_executor import SPEC_EXECUTOR_REMOVED, build_coding_task
    used = []

    async def runner(name, args):
        used.append(name)
        return "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nok"

    class _LLM:
        async def chat_completion(self, *a, **k):
            used.append("llm")
            return {"choices": [{"message": {"content": "{}"}}]}
    ctx = SimpleNamespace(llm_client=_LLM(), args=SimpleNamespace(model="m"))
    for env in ("spec", "agentic"):
        res = await build_coding_task(ctx, "Build parser.py", tool_runner=runner, executor=env)
        assert res.ok is False and res.summary == SPEC_EXECUTOR_REMOVED and res.files == []
    assert used == []
    assert build_coding_task._ghost_real_executor is True        # the advancer's marker still holds


def test_the_removed_builders_are_gone():
    # world: the deleted executors are still importable (dead code kept alive)
    from ghost_agent.core import coding_executor as CE
    from ghost_agent.core import coding_loop as CL
    for name in ("_generate_build_spec", "_apply_file", "_run_verify"):
        assert not hasattr(CE, name)
    for name in ("build_coding_task_agentic", "executor_kind", "leaf_shape", "snapshot_workspace"):
        assert not hasattr(CL, name)
    # what other modules still read survives
    assert callable(CL.run_leaf_turn) and callable(CL.build_leaf_context)
    assert CE.leaf_deadline_context.get() == 0.0 and CE._TASK_NAMED_FILE_RE.search("write a.py")
