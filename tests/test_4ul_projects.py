"""§4UL (operator 11-12: "1. A 2. C 3. B. i don't want to be asked. 4. A. proceed with all batches") —
the projects side: leftover confirmation wording gone, the verify-word hold only for a build that did
not verify, a DAILY unattended allowance (owner runs never capped, no check-in by default), plan order
as a soft dependency (depends_on in subtask objects / task_update, stale dependents), a write into a
project folder is project work (note_project_write), retired constraints re-armed on any reopen, and
the DONE clean-up sweep deferred to the request's end (flush_deferred_sweeps)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ghost_agent.core.project_advancer as PA
from ghost_agent.core.coding_executor import CodingResult
from ghost_agent.core.planning import (ProjectPlan, TaskStatus, later_done_dependents,
                                       soft_order_blocked)
from ghost_agent.memory import projects as MP
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools import projects as TP
from ghost_agent.tools.projects import note_project_write, tool_manage_projects
from ghost_agent.utils.logging import request_id_context

SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def tctx(tmp_path, monkeypatch, store):
    import ghost_agent.tools.registry as reg

    async def _web_search(query="", **_kw):
        return f"stub results for {query}"
    monkeypatch.setattr(reg, "get_available_tools", lambda ctx: {"web_search": _web_search})
    return SimpleNamespace(project_store=store, scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                           graph_memory=None, contradiction_log=None, current_project_id=None,
                           llm_client=None, workspace_model=None)


def _ctx(store):
    return SimpleNamespace(project_store=store, llm_client=None, scratchpad=None,
                           workspace_model=None, current_project_id=None)


# ───────────────────────────── 1a: no confirmation wording ─────────────────────────────

def _strings(rel):
    """Every string literal (f-string parts joined) in a module, from its parsed tree — what the model can read."""
    import ast
    out = []
    for n in ast.walk(ast.parse((SRC / rel).read_text())):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append(n.value)
        elif isinstance(n, ast.JoinedStr):
            out.append("".join(v.value for v in n.values if isinstance(v, ast.Constant) and isinstance(v.value, str)))
    return "\n".join(out)


def test_no_leftover_ask_or_approve_wording_reaches_the_model():
    prompts = _strings("core/prompts.py")
    assert "does not answer a gate" not in prompts and "name each waiting task and ask" not in prompts
    tools = _strings("tools/projects.py")
    for gone in ("name them and ask", "approve <task id>", "tell the owner what it needs and stop",
                 "CORRECTION CONTEXT", "owner must raise it"):
        assert gone not in tools, gone
    close = _strings("core/task_close.py")
    assert "do it now" in close and "say what only the owner can give" in close


def test_exit_and_unrelease_are_visible_actions():
    assert "exit" in TP._VISIBLE_ACTIONS and "unrelease" in TP._VISIBLE_ACTIONS


# ───────────────────────────── 1e: verify-word hold ─────────────────────────────

def test_a_verified_build_is_not_held_for_its_verify_words():
    desc = "Build and test the login page"
    assert PA._unattended_close(False, desc, unverified_build=False)[0] == TaskStatus.DONE
    assert PA._unattended_close(False, desc, unverified_build=True)[0] == TaskStatus.NEEDS_USER
    assert PA._unattended_close(False, desc)[0] == TaskStatus.NEEDS_USER       # no build: still held
    assert PA._unattended_close(True, desc, unverified_build=True)[0] == TaskStatus.DONE


async def test_a_build_s_verify_hold_is_rechecked_by_the_owner_s_run(tmp_path, store):
    assert "verify_task" in PA.HOLD_KINDS_RECHECKED
    pid = store.create_project("App", kind="CODING")
    desc = "Build and test hello.py"
    tid = store.add_task(pid, desc)
    why = f"needs a real check — {PA._HELD_MARK}"
    store.update_task(tid, status="NEEDS_USER", result_summary=f"built hello.py — {why}",
                      wait_on="owner_turn", wait_code="verify_task")
    store.update_project(pid, status="NEEDS_USER")
    store.log_event(pid, tid, "autoadvance_needs_user",
                    {"description": desc, "reason": why, "hold": "verify_task", "files": ["hello.py"]})
    (Path(store.ensure_workspace(pid)) / "hello.py").write_text("print('hi')\n")
    built = []

    async def exe(ctx, d, **kw):
        built.append(d)
        return CodingResult(True, "built", files=["hello.py"])
    res = await PA.advance_once(_ctx(store), pid, tool_runner=None, coding_executor=exe, owner_requested=True)
    assert res.classification == "recheck" and not built and store.get_task(tid)["status"] == "DONE"


# ───────────────────────────── 1f + 3B: the daily allowance ─────────────────────────────

def _autopilot(store, n=3, **meta):
    pid = store.create_project("Job", metadata={"autopilot": True, "unattended_day": PA._today(), **meta})
    store.update_project(pid, status="ACTIVE")
    return pid, [store.add_task(pid, f"Write chapter {i}") for i in range(n)]


def _closing_step(store, ran):
    async def step(context, project_id, owner_requested=False, claim_sink=None, **kw):
        ran.append(owner_requested)
        open_ = [t for t in store.list_tasks(project_id) if t["status"] == "PENDING"]
        if not open_:
            return PA.AdvanceResult(True, None, "idle", "no READY/PENDING leaf")
        if claim_sink is not None:
            claim_sink.append(open_[0]["id"])
        store.update_task(open_[0]["id"], status="DONE", result_summary="written")
        return PA.AdvanceResult(True, open_[0]["id"], "research", "worked")
    return step


async def test_the_daily_allowance_rests_the_project_quietly_and_a_new_day_continues(store, monkeypatch):
    pid, _ = _autopilot(store, n=4, unattended_per_day=2)
    ran = []
    monkeypatch.setattr(PA, "advance_once", _closing_step(store, ran))
    for _ in range(3):
        res = await PA.advance_unattended(_ctx(store), pid)
    m = store.get_project(pid)["metadata"]
    assert len(ran) == 2 and res.summary.startswith("resting until tomorrow")
    assert m["autopilot"] is True and not m.get("autopilot_paused")
    assert len(store.list_events(pid, event_type="autopilot_resting")) == 1
    monkeypatch.setattr(PA, "_today", lambda: "2999-01-01")
    await PA.advance_unattended(_ctx(store), pid)
    assert len(ran) == 3 and store.get_project(pid)["metadata"]["unattended_steps_today"] == 1


async def test_the_default_allowance_is_thirty_tasks_a_day():
    assert PA.DAILY_UNATTENDED_TASKS == 30 and PA.daily_cap({}) == 30 and PA.daily_cap({"unattended_per_day": 5}) == 5


async def test_the_owner_s_run_is_never_capped(store):
    pid = store.create_project("Job", metadata={"steps_cap": 1, "steps_used": 99,
                                                "unattended_day": PA._today(), "unattended_steps_today": 999})
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Research the abacus")

    async def runner(name, args):
        return "[Source: https://example.org]\nThe abacus is an old counting frame used for arithmetic."
    res = await PA.advance_once(_ctx(store), pid, tool_runner=runner, owner_requested=True)
    assert res.task_id == tid and "budget" not in res.summary.lower()


async def test_no_check_in_pause_unless_the_owner_set_one(store, monkeypatch):
    pid, _ = _autopilot(store, n=12)
    ran = []
    monkeypatch.setattr(PA, "advance_once", _closing_step(store, ran))
    for _ in range(11):
        await PA.advance_unattended(_ctx(store), pid)
    m = store.get_project(pid)["metadata"]
    assert len(ran) == 11 and m["autopilot"] is True and not m.get("autopilot_paused")
    assert PA.DEFAULT_CHECKPOINT_EVERY == 0


async def test_the_budget_action_takes_the_daily_allowance_and_maps_steps_cap(store):
    pid = store.create_project("Job")
    ctx = SimpleNamespace(project_store=store, workspace_model=None, current_project_id=None)
    tok = request_id_context.set("req-owner")
    try:
        res = json.loads(await TP.tool_manage_projects_for_model(ctx, action="budget", project_id=pid,
                                                                 metadata={"steps_cap": 12}))
        bad = await TP.tool_manage_projects_for_model(ctx, action="budget", project_id=pid,
                                                      metadata={"tool_call_cap": 3})
    finally:
        request_id_context.reset(tok)
    assert store.get_project(pid)["metadata"]["unattended_per_day"] == 12
    assert res["budget"]["unattended_today"] == "0/12 tasks"
    assert "unattended_per_day" in bad and "owner must raise" not in bad


# ───────────────────────────── 2a: plan order ─────────────────────────────

def test_soft_order_holds_later_siblings_behind_a_waiting_task_not_a_failed_one(store):
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    a, b, c, d = (plan.add_task(x) for x in ("a", "b", "c", "d"))
    plan.update_status(a, TaskStatus.FAILED, failure_reason="x")
    assert soft_order_blocked(ProjectPlan(store, pid)) == set()   # §4UL review M1: a failure holds nothing
    plan.update_status(a, TaskStatus.NEEDS_USER, wait=("owner_turn", "held", "look"))
    store.update_task(d, depends_on=[])                  # no own deps: still in plan order
    plan = ProjectPlan(store, pid)
    assert soft_order_blocked(plan) == {b, c, d}
    assert soft_order_blocked(plan, independent={c}) == {b, d}
    plan2 = ProjectPlan(store, store.create_project("Q"))
    x, y, z = (plan2.add_task(t) for t in ("x", "y", "z"))
    store.update_task(z, depends_on=[x])                 # declared deps govern z, not order
    plan2 = ProjectPlan(store, plan2.project_id)
    assert soft_order_blocked(plan2, excluded={x}) == {y}


def test_a_failure_inside_an_earlier_phase_holds_the_next_phase(store):
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    ph1 = plan.add_task("phase 1")
    k = plan.add_task("experiment", parent_id=ph1)
    ph2 = plan.add_task("phase 2")
    w = plan.add_task("write findings", parent_id=ph2)
    plan.update_status(k, TaskStatus.NEEDS_USER, wait=("owner_turn", "model_experiment", "run it"))
    assert {ph2, w} <= soft_order_blocked(ProjectPlan(store, pid))


async def test_decompose_takes_depends_on_positions_ids_and_independence(tctx):
    await tool_manage_projects(tctx, action="create", title="Order")
    pid = tctx.current_project_id
    out = json.loads(await tool_manage_projects(tctx, action="task_decompose", subtasks=[
        {"description": "Run experiment one", "kind": "research"},
        {"description": "Run experiment two", "kind": "research", "depends_on": []},
        {"description": "Write the findings", "kind": "write", "depends_on": [1, "2"]},
        {"description": "Write the glossary", "kind": "write", "depends_on": ["zzzz"]},
    ]))
    ids = out["created"]
    assert len(ids) == 4
    s = tctx.project_store
    assert s.get_task(ids[2])["depends_on"] == [ids[0], ids[1]]
    assert s.get_task(ids[3])["depends_on"] == []                     # nothing resolved: order stands
    assert out["depends_on_notes"] and "zzzz" in out["depends_on_notes"][0]
    assert s.get_project(pid)["metadata"]["independent_tasks"] == [ids[1]]


async def test_the_batch_does_not_build_past_an_earlier_waiting_task(tctx, monkeypatch):
    await tool_manage_projects(tctx, action="create", title="Experiments")
    pid = tctx.current_project_id
    s = tctx.project_store
    e = s.add_task(pid, "Run the probe experiment")
    f = s.add_task(pid, "Write findings.md from the experiments")
    s.update_task(e, status="NEEDS_USER", wait_on="owner", wait_code="asked", wait_why="which model?")
    res = await PA.advance_once(_ctx(s), pid, tool_runner=None)
    assert res.task_id is None and "plan order" in res.summary
    assert s.get_task(f)["status"] == "PENDING"


async def test_task_update_depends_on_sets_and_clears_independence(tctx):
    await tool_manage_projects(tctx, action="create", title="Upd")
    pid = tctx.current_project_id
    s = tctx.project_store
    a, b = s.add_task(pid, "first"), s.add_task(pid, "second")
    await tool_manage_projects(tctx, action="task_update", task_id=b, depends_on=[])
    assert s.get_project(pid)["metadata"]["independent_tasks"] == [b]
    await tool_manage_projects(tctx, action="task_update", task_id=b, depends_on=[a])
    assert s.get_task(b)["depends_on"] == [a]
    assert not s.get_project(pid)["metadata"].get("independent_tasks")
    await tool_manage_projects(tctx, action="task_update", task_id=b, depends_on=["nope"])
    assert s.get_task(b)["depends_on"] == [a]                         # an unknown id changes nothing


async def test_a_late_close_marks_the_done_work_built_on_it_stale(tctx):
    await tool_manage_projects(tctx, action="create", title="Stale")
    pid = tctx.current_project_id
    s = tctx.project_store
    a, b, c = s.add_task(pid, "data"), s.add_task(pid, "chart"), s.add_task(pid, "report")
    s.update_task(c, depends_on=[a])
    for t in (b, c):
        s.update_task(t, status="DONE", result_summary="done")
    assert set(later_done_dependents(ProjectPlan(s, pid), a)) == {b, c}
    out = json.loads(await tool_manage_projects(tctx, action="task_update", task_id=a, status="DONE",
                                                result="collected the data and checked it"))
    assert out.get("closed") == [a], out
    assert set(out["stale_dependents"]) == {b, c} and "re-check" in out["agent_instruction_stale"]
    assert s.list_events(pid, event_type="stale_dependent")


# ───────────────────────────── 2d: a write into a project folder ─────────────────────────────

def _owner(ctx, text="make the header blue please"):
    ctx.last_user_content = text
    return request_id_context.set("req-owner-1")


async def test_a_write_focuses_and_reopens_a_done_project_with_a_change_task(tctx):
    s = tctx.project_store
    pid = s.create_project("Site", metadata={"constraints": ["keep it one page"]})
    t = s.add_task(pid, "Build index.html")
    tok = request_id_context.set("req-earlier")
    try:
        s.update_task(t, status="DONE", result_summary="built")
    finally:
        request_id_context.reset(tok)
    assert s.get_project(pid)["status"] == "DONE"
    assert s.get_project(pid)["metadata"]["constraints"] == []
    ws = Path(s.ensure_workspace(pid))
    tok = _owner(tctx)
    try:
        note = note_project_write(tctx, ws / "index.html")
    finally:
        request_id_context.reset(tok)
    p = s.get_project(pid)
    assert tctx.current_project_id == pid and p["status"] == "ACTIVE" and "reopened" in note
    change = [x for x in s.list_tasks(pid) if x["description"].startswith("Change: ")]
    assert len(change) == 1 and change[0]["status"] == "IN_PROGRESS"
    assert change[0]["description"] == "Change: make the header blue please"
    assert "REOPENED" in (p["metadata"].get("design_ledger") or json.dumps(p["metadata"]))
    assert p["metadata"]["constraints"] == ["keep it one page"]          # re-armed with the reopen


async def test_a_write_does_nothing_for_a_background_turn_or_outside_projects(tctx):
    s = tctx.project_store
    pid = s.create_project("Site")
    t = s.add_task(pid, "Build it")
    s.update_task(t, status="DONE", result_summary="built")
    ws = Path(s.ensure_workspace(pid))
    tok = request_id_context.set("sched-1")
    try:
        assert note_project_write(tctx, ws / "a.txt") == ""
    finally:
        request_id_context.reset(tok)
    tok = _owner(tctx)
    try:
        assert note_project_write(tctx, Path(s.sandbox_root) / "notes.txt") == ""
        assert note_project_write(tctx, ws) == ""                      # the folder itself
    finally:
        request_id_context.reset(tok)
    assert s.get_project(pid)["status"] == "DONE" and tctx.current_project_id is None


async def test_a_write_in_the_request_that_closed_the_project_is_not_a_change(tctx):
    s = tctx.project_store
    pid = s.create_project("Site")
    t = s.add_task(pid, "Build it")
    tok = _owner(tctx)
    try:
        s.update_task(t, status="DONE", result_summary="built")
        assert note_project_write(tctx, Path(s.ensure_workspace(pid)) / "NOTES.md") == ""
    finally:
        request_id_context.reset(tok)
    assert s.get_project(pid)["status"] == "DONE"


# ───────────────────────────── 2e: the deferred DONE sweep ─────────────────────────────

def test_the_done_sweep_waits_for_the_request_end(store):
    swept = []
    store.on_project_done = swept.append
    pid = store.create_project("P")
    t = store.add_task(pid, "only")
    tok = request_id_context.set("req-sweep")
    try:
        store.update_task(t, status="DONE", result_summary="done")
        assert store.get_project(pid)["status"] == "DONE" and swept == []
        assert MP.flush_deferred_sweeps("req-sweep") == 1
    finally:
        request_id_context.reset(tok)
    assert swept == [pid]
    assert MP.flush_deferred_sweeps("req-sweep") == 0


def test_a_project_reopened_before_the_end_is_not_swept(store):
    swept = []
    store.on_project_done = swept.append
    pid = store.create_project("P")
    t = store.add_task(pid, "only")
    tok = request_id_context.set("req-sweep-2")
    try:
        store.update_task(t, status="DONE", result_summary="done")
        store.add_task(pid, "one more thing")
        MP.flush_deferred_sweeps("req-sweep-2")
    finally:
        request_id_context.reset(tok)
    assert swept == []


def test_outside_a_request_the_sweep_runs_now_and_a_stale_deferral_is_flushed(store, monkeypatch):
    swept = []
    store.on_project_done = swept.append
    a = store.create_project("A")
    store.update_task(store.add_task(a, "x"), status="DONE", result_summary="d")
    assert swept == [a]                                                  # SYSTEM: immediate
    b = store.create_project("B")
    tok = request_id_context.set("req-never-flushed")
    try:
        store.update_task(store.add_task(b, "x"), status="DONE", result_summary="d")
    finally:
        request_id_context.reset(tok)
    assert swept == [a]
    monkeypatch.setattr(MP, "DEFERRED_SWEEP_MAX_AGE_S", -1.0)
    MP.flush_deferred_sweeps("some-other-request")
    assert swept == [a, b]
