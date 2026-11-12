"""§4SU Phase 5 (operator 11-02, "do 1 and 5"): a held task is re-CHECKED by
the owner's run, not rebuilt (the autopilot opt-in and the model-experiment
batch are pinned in test_4lp_projects / test_4mq_long_horizon /
test_4sr_gate_words)."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import ghost_agent.core.project_advancer as PA
from ghost_agent.core.coding_executor import CodingResult
from ghost_agent.core.planning import ProjectPlan
from ghost_agent.memory.projects import ProjectStore

pytestmark = pytest.mark.asyncio


def _ctx(store):
    return SimpleNamespace(project_store=store, llm_client=None, scratchpad=None,
                           workspace_model=None, current_project_id=None)


def _held_build(tmp_path, write=True):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Build App", kind="CODING")
    store.update_project(pid, metadata={"autopilot": True})
    tid = ProjectPlan(store, pid).add_task("Implement hello.py")
    calls = []

    async def exe(ctx, desc, **kw):
        calls.append(desc)
        if write:
            (Path(store.ensure_workspace(pid)) / "hello.py").write_text("print('hi')\n")
        return CodingResult(True, "built hello.py", files=["hello.py"])
    return store, pid, tid, exe, calls


async def test_an_unverified_build_hold_is_rechecked_not_rebuilt(tmp_path):
    store, pid, tid, exe, calls = _held_build(tmp_path)
    await PA.advance_once(_ctx(store), pid, tool_runner=AsyncMock(return_value=""), coding_executor=exe,
                          owner_requested=False)
    assert store.get_task(tid)["status"] == "NEEDS_USER" and len(calls) == 1
    ev = store.latest_task_event(pid, tid, ("autoadvance_needs_user",))
    assert ev["payload"]["hold"] == "unverified_build"
    res = await PA.advance_once(_ctx(store), pid, tool_runner=AsyncMock(return_value=""), coding_executor=exe,
                                owner_requested=True)
    t = store.get_task(tid)
    assert t["status"] == "DONE" and len(calls) == 1, "rebuilt instead of re-checked"
    assert res.classification == "recheck" and "re-checked" in t["result_summary"]
    assert "not closed unattended" not in t["result_summary"]


async def test_a_held_build_whose_file_is_gone_is_rebuilt(tmp_path):
    store, pid, tid, exe, calls = _held_build(tmp_path)
    await PA.advance_once(_ctx(store), pid, tool_runner=AsyncMock(return_value=""), coding_executor=exe,
                          owner_requested=False)
    (Path(store.ensure_workspace(pid)) / "hello.py").unlink()
    await PA.advance_once(_ctx(store), pid, tool_runner=AsyncMock(return_value=""), coding_executor=exe,
                          owner_requested=True)
    assert len(calls) == 2 and store.get_task(tid)["status"] == "DONE"


async def test_an_empty_held_file_is_rebuilt(tmp_path):
    store, pid, tid, exe, calls = _held_build(tmp_path)
    await PA.advance_once(_ctx(store), pid, tool_runner=AsyncMock(return_value=""), coding_executor=exe,
                          owner_requested=False)
    (Path(store.ensure_workspace(pid)) / "hello.py").write_text("")
    await PA.advance_once(_ctx(store), pid, tool_runner=AsyncMock(return_value=""), coding_executor=exe,
                          owner_requested=True)
    assert len(calls) == 2


async def test_a_verify_task_hold_is_run_again(tmp_path):
    # "needs a real check": the check itself is what was missing — it runs
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("App")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Verify the full pipeline end to end")
    runs = []

    async def runner(name, args):
        runs.append(name)
        return "all 12 checks passed"
    await PA.advance_once(_ctx(store), pid, tool_runner=runner)
    assert store.get_task(tid)["status"] == "NEEDS_USER"
    assert store.latest_task_event(pid, tid, ("autoadvance_needs_user",))["payload"]["hold"] == "verify_task"
    await PA.advance_once(_ctx(store), pid, tool_runner=runner, owner_requested=True)
    assert len(runs) == 2 and store.get_task(tid)["status"] == "DONE"


def _manual_hold(tmp_path, kind, desc="Write the brief → research/brief.md", files=()):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Notes")
    tid = store.add_task(pid, desc)
    why = f"the file is missing — {PA._HELD_MARK}"
    store.update_task(tid, status="NEEDS_USER", result_summary=f"wrote notes — {why}",
                      wait_on="owner_turn", wait_code=kind)
    store.update_project(pid, status="NEEDS_USER")
    store.log_event(pid, tid, "autoadvance_needs_user",
                    {"description": desc, "reason": why, "hold": kind, "files": list(files)})
    return store, pid, tid


async def test_an_undelivered_hold_closes_once_the_file_is_there(tmp_path):
    store, pid, tid = _manual_hold(tmp_path, "undelivered", files=["research/brief.md"])
    f = Path(store.ensure_workspace(pid)) / "research" / "brief.md"
    f.parent.mkdir(parents=True)
    f.write_text("# Brief\n\n" + "A real paragraph about the topic. " * 20)
    runs = []

    async def runner(name, args):
        runs.append(name)
        return "x"
    res = await PA.advance_once(_ctx(store), pid, tool_runner=runner, owner_requested=True)
    t = store.get_task(tid)
    assert t["status"] == "DONE" and not runs and res.classification == "recheck"
    assert t["result_summary"].startswith("wrote notes — re-checked")


async def test_an_undelivered_hold_still_missing_runs_the_task(tmp_path):
    store, pid, tid = _manual_hold(tmp_path, "undelivered", files=[])
    runs = []

    async def runner(name, args):
        runs.append(name)
        return "x"
    res = await PA.advance_once(_ctx(store), pid, tool_runner=runner, owner_requested=True)
    assert res.classification != "recheck" and store.get_task(tid)["status"] != "DONE", \
        "the re-check closed a task whose declared file is missing"


async def test_a_research_hold_is_run_again(tmp_path):
    store, pid, tid = _manual_hold(tmp_path, "research_missed", desc="Research the market", files=[])
    runs = []

    async def runner(name, args):
        runs.append(name)
        return "x"
    res = await PA.advance_once(_ctx(store), pid, tool_runner=runner, owner_requested=True)
    assert res.classification != "recheck"


async def test_an_unrun_experiment_hold_still_unrun_is_not_closed(tmp_path, monkeypatch):
    store, pid, tid = _manual_hold(tmp_path, "unrun_experiment", desc="Write up the run → results/run.md",
                                   files=["results/run.md"])
    f = Path(store.ensure_workspace(pid)) / "results" / "run.md"
    f.parent.mkdir(parents=True)
    f.write_text("# Run\n\n" + "Model response: I am aware. " * 20)
    import ghost_agent.core.experiment_gate as EG
    monkeypatch.setattr(EG, "unrun_experiment_problem", lambda *a, **k: "no probe transcript")
    res = await PA._recheck_held(_ctx(store), store, ProjectPlan(store, pid), pid,
                                 ProjectPlan(store, pid).tree.nodes[tid], store.get_task(tid)["result_summary"])
    assert res is None and store.get_task(tid)["status"] == "NEEDS_USER"


async def test_an_unattended_run_never_rechecks(tmp_path):
    store, pid, tid = _manual_hold(tmp_path, "undelivered", files=["research/brief.md"])
    f = Path(store.ensure_workspace(pid)) / "research" / "brief.md"
    f.parent.mkdir(parents=True)
    f.write_text("# Brief\n\n" + "text " * 50)
    store.update_project(pid, status="ACTIVE")
    await PA.advance_once(_ctx(store), pid, tool_runner=AsyncMock(return_value="x"), owner_requested=False)
    assert store.get_task(tid)["status"] == "NEEDS_USER"


# ─────────────── review r1 + battery survivors ───────────────

async def _held(tmp_path):
    store, pid, tid, exe, calls = _held_build(tmp_path)
    await PA.advance_once(_ctx(store), pid, tool_runner=AsyncMock(return_value=""), coding_executor=exe,
                          owner_requested=False)
    assert store.get_task(tid)["status"] == "NEEDS_USER"
    return store, pid, tid, exe, calls


async def test_a_rewritten_task_is_rebuilt_not_rechecked(tmp_path):
    # r1 MAJOR: the new work never happened
    store, pid, tid, exe, calls = await _held(tmp_path)
    store.update_task(tid, description="Implement hello.py with a --name flag and unit tests")
    await PA.advance_once(_ctx(store), pid, tool_runner=AsyncMock(return_value=""), coding_executor=exe,
                          owner_requested=True)
    assert len(calls) == 2


async def test_an_unmet_postcondition_reruns_instead_of_failing(tmp_path):
    store, pid, tid, exe, calls = await _held(tmp_path)
    store.update_task(tid, postconditions=["benchmark report written"])
    res = await PA.advance_once(_ctx(store), pid, tool_runner=AsyncMock(return_value=""), coding_executor=exe,
                                owner_requested=True)
    assert res.classification != "recheck" and len(calls) == 2


async def test_a_file_outside_the_workspace_never_passes(tmp_path):
    import os
    outside = tmp_path / "outside.md"
    outside.write_text("# real content\n" * 20)
    store, pid, tid = _manual_hold(tmp_path, "undelivered", desc="Write notes")
    rel = os.path.relpath(outside, Path(store.ensure_workspace(pid)))
    store.log_event(pid, tid, "autoadvance_needs_user",
                    {"description": "Write notes", "reason": f"the file is missing — {PA._HELD_MARK}",
                     "hold": "undelivered", "files": [rel]})
    res = await PA._recheck_held(_ctx(store), store, ProjectPlan(store, pid), pid,
                                 ProjectPlan(store, pid).tree.nodes[tid], store.get_task(tid)["result_summary"])
    assert res is None


async def test_a_rechecked_file_is_registered(tmp_path):
    store, pid, tid = _manual_hold(tmp_path, "undelivered", files=["research/brief.md"])
    f = Path(store.ensure_workspace(pid)) / "research" / "brief.md"
    f.parent.mkdir(parents=True)
    f.write_text("# Brief\n\n" + "A real paragraph about the topic. " * 20)
    await PA.advance_once(_ctx(store), pid, tool_runner=AsyncMock(return_value="x"), owner_requested=True)
    assert "research/brief.md" in [str(d) for d in store.list_deliverables(pid)]


async def test_an_unverified_hold_with_no_files_is_rebuilt(tmp_path):
    store, pid, tid = _manual_hold(tmp_path, "unverified_build", desc="Implement hello.py", files=[])
    assert not (store.list_artifacts(task_id=tid) or [])
    res = await PA._recheck_held(_ctx(store), store, ProjectPlan(store, pid), pid,
                                 ProjectPlan(store, pid).tree.nodes[tid], store.get_task(tid)["result_summary"])
    assert res is None


async def test_a_research_undelivered_hold_records_its_kind(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Notes")
    store.update_project(pid, status="ACTIVE", metadata={"autopilot": True})
    tid = store.add_task(pid, "Research the market → research/market.md")
    import ghost_agent.core.task_deliverable as TD
    monkeypatch.setattr(TD, "deliverable_problem", lambda ws, d: "the declared deliverable research/market.md does not exist")

    async def runner(name, args):
        return "Market sizes: " + "a real finding with numbers 12% and 40M. " * 10
    monkeypatch.setattr(PA, "_research_output_usable", lambda o: True)
    await PA.advance_once(_ctx(store), pid, tool_runner=runner, owner_requested=False)
    ev = store.latest_task_event(pid, tid, ("autoadvance_needs_user",))
    assert store.get_task(tid)["status"] == "NEEDS_USER", store.get_task(tid)
    assert ev["payload"]["hold"] == "undelivered"


async def test_the_real_batch_moves_past_an_experiment_to_independent_work(tmp_path):
    # C3: the exclusion reaches the real pick — without it the experiment is
    # re-picked every tick and the independent task never runs
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Study", goal="Compare how Gemma answers three prompts")
    store.update_project(pid, status="ACTIVE")
    first = store.add_task(pid, "Set up the bibliography folder")
    exp = store.add_task(pid, "Run the three prompts on Gemma with probe_model")
    # §4UI: a later task runs past the experiment when it DECLARES what it
    # waits on (an undeclared later peer may be the experiment's analysis)
    other = store.add_task(pid, "Collect the citation list of related papers", depends_on=[first])
    runs = []

    async def runner(name, args):
        runs.append(name)
        return "Found papers: " + "Smith 2021, Jones 2022. " * 10
    r = await PA.advance_many(_ctx(store), pid, max_tasks=None, owner_requested=True,
                              tool_runner=runner, stop_on_fail=False)
    assert exp in r.experiments
    assert store.get_task(other)["status"] != "PENDING", "the independent task never ran"
    assert sum(1 for a in r.advanced if a.get("task_id") == exp) == 1


async def test_a_deadline_stop_keeps_its_reason_and_lists_the_experiment(monkeypatch):
    # r1: "do it NOW" at the deadline is the worst time to start an experiment
    from ghost_agent.core.project_advancer import AdvanceResult

    class _FakeStore:
        def __init__(self, status):
            self.status = status

        def get_task(self, tid):
            return {"status": self.status.get(tid)}

        def get_project(self, pid):
            return {"status": "ACTIVE"}

        def list_tasks(self, pid):
            return [{"id": k, "status": v} for k, v in self.status.items()]
    seq = [AdvanceResult(True, "e", "blocked", PA.MODEL_EXPERIMENT_PREFIX + ": x"),
           AdvanceResult(True, "t2", "research", "ok")]
    ticks = []

    async def fake(context, project_id, **kw):
        ticks.append(1)
        return seq.pop(0)
    monkeypatch.setattr(PA, "advance_once", fake)
    monkeypatch.setattr(PA, "_paused_since", lambda *a, **k: False)
    monkeypatch.setattr(PA, "_batch_would_cross_deadline", lambda est: len(ticks) >= 2)
    ctx = SimpleNamespace(project_store=_FakeStore({"e": "PENDING", "t2": "DONE"}))
    r = await PA.advance_many(ctx, "p", owner_requested=True, max_tasks=None, stop_on_fail=False)
    assert r.stop_reason == "deadline" and r.experiments == ["e"]


async def test_a_stop_landing_between_read_and_reset_stays_off(tmp_path, monkeypatch):
    # r1: the read said ON, a "stop all" turned it OFF, the reset must not turn it back on
    import json
    from ghost_agent.tools.projects import tool_manage_projects
    from ghost_agent.utils.logging import request_id_context
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    ctx = SimpleNamespace(project_store=store, llm_client=None, scratchpad=None, workspace_model=None,
                          current_project_id=None, args=None, sandbox_dir=str(tmp_path / "sb"),
                          memory_dir=str(tmp_path / "m"), last_user_content="")
    pid = store.create_project("App")
    store.update_project(pid, status="ACTIVE", metadata={"autopilot": False})
    real = store.get_project

    def stale(p):
        d = dict(real(p) or {})
        d["metadata"] = dict(d.get("metadata") or {}, autopilot=True)   # the read before the stop
        return d
    monkeypatch.setattr(store, "get_project", stale)

    async def none(*a, **k):
        return PA.AdvanceManyResult([], "project_done", 1)
    monkeypatch.setattr(PA, "advance_many", none)
    tok = request_id_context.set("req-owner")
    try:
        await tool_manage_projects(ctx, action="autoadvance", project_id=pid, count="1")
    finally:
        request_id_context.reset(tok)
    monkeypatch.setattr(store, "get_project", real)
    assert not (store.get_project(pid)["metadata"]).get("autopilot")
