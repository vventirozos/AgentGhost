"""§4SU Phase 3 — why a task waits is FIELDS (wait_on / wait_code / wait_why,
core/task_wait.py), never the summary text; READY folds into PENDING and a
paused task waits on the owner; the model sees five states."""
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import ghost_agent.core.project_advancer as PA
from ghost_agent.core.coding_executor import CodingResult
from ghost_agent.core.planning import ProjectPlan, TaskStatus
from ghost_agent.core.task_wait import (view_state, view_label, wait_from_marks, fold_status,
                                        is_hold, default_wait)
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.projects import tool_manage_projects
from tests.test_4qa_project_task_integrity import _ctx as _tool_ctx
from tests.test_4su_phase0 import owner_turn  # noqa: F401  (fixture)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")


def _actx(store):
    return SimpleNamespace(project_store=store, llm_client=None, scratchpad=None,
                           workspace_model=None, current_project_id=None)


def _wait(store, tid):
    t = store.get_task(tid)
    return t["status"], t["wait_on"], t["wait_code"]


# ───────────────────────────── the vocabulary ─────────────────────────────

@pytest.mark.parametrize("status,summary,expect", [
    ("NEEDS_USER", "wrote x — needs a real check — not closed unattended", ("owner_turn", "held")),
    ("NEEDS_USER", "human gate: cto approval", ("owner", "human_gate")),
    ("NEEDS_USER", "not closed: deliverable: report.md does not exist", ("owner", "rollup_deliverable")),
    ("NEEDS_USER", "not closed: constraints: its files have not been audited", ("owner", "rollup_constraints")),
    ("NEEDS_USER", "flagged for human review", ("owner", "review")),
    ("NEEDS_USER", "which people?", ("owner", "asked")),
    ("BLOCKED", "", ("dependency", "child_failed")),
    ("PAUSED", "", ("owner", "paused")),
    ("PENDING", "not closed unattended", ("", "")),
])
def test_the_old_marks_are_read_once(status, summary, expect):
    on, code, _ = wait_from_marks(status, summary)
    assert (on, code) == expect


def test_the_five_states():
    assert view_state({"status": "IN_PROGRESS"}) == {"state": "RUNNING"}
    assert view_state({"status": "READY"}) == {"state": "PENDING"}
    assert view_label({"status": "NEEDS_USER", "wait_on": "owner_turn", "wait_why": "x"}) == "WAITING(owner_turn: x)"
    assert view_state({"status": "BLOCKED"})["on"] == "dependency"
    assert view_state(SimpleNamespace(status=TaskStatus.DONE)) == {"state": "DONE"}
    assert fold_status(TaskStatus.READY) == "PENDING" and fold_status("paused") == "NEEDS_USER"
    assert default_wait("NEEDS_USER") == ("owner", "asked")


# ───────────────────────────── storage ─────────────────────────────

def test_an_old_database_is_migrated_once(tmp_path):
    s = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = s.create_project("P")
    ids = {k: s.add_task(pid, k) for k in ("held", "gate", "ready", "paused", "blocked", "done")}
    db = next(Path(tmp_path / "m").rglob("*.db"))
    con = sqlite3.connect(db)
    for col in ("wait_on", "wait_code", "wait_why"):
        con.execute(f"ALTER TABLE tasks DROP COLUMN {col}")
    rows = {"held": ("NEEDS_USER", "built — needs a real check — not closed unattended"),
            "gate": ("NEEDS_USER", "human gate: cto"), "ready": ("READY", ""),
            "paused": ("PAUSED", ""), "blocked": ("BLOCKED", ""), "done": ("DONE", "ok")}
    for k, (st, summ) in rows.items():
        con.execute("UPDATE tasks SET status = ?, result_summary = ? WHERE id = ?", (st, summ, ids[k]))
    con.commit()
    con.close()
    s2 = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    assert _wait(s2, ids["held"]) == ("NEEDS_USER", "owner_turn", "held")
    assert _wait(s2, ids["gate"]) == ("NEEDS_USER", "owner", "human_gate")
    assert _wait(s2, ids["ready"]) == ("PENDING", "", "")
    assert _wait(s2, ids["paused"]) == ("NEEDS_USER", "owner", "paused")
    assert _wait(s2, ids["blocked"]) == ("BLOCKED", "dependency", "blocked")    # r6: no failed child → its own block
    assert _wait(s2, ids["done"]) == ("DONE", "", "")
    # idempotent: a later open changes nothing (a hold stays a hold)
    ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    assert _wait(s2, ids["held"]) == ("NEEDS_USER", "owner_turn", "held")


def test_leaving_a_wait_clears_it_and_entering_one_has_a_safe_default(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "x")
    store.update_task(tid, status="NEEDS_USER", result_summary="which colour?")
    assert _wait(store, tid) == ("NEEDS_USER", "owner", "asked")      # never a hold by default
    assert store.get_task(tid)["wait_why"] == "which colour?"
    store.update_task(tid, status="NEEDS_USER", wait_on="owner_turn", wait_code="held")
    store.update_task(tid, status="NEEDS_USER", result_summary="the owner must approve the wording")
    # r1 MAJOR: a status write with no wait is the safe default — never the old hold
    assert _wait(store, tid) == ("NEEDS_USER", "owner", "asked")
    store.update_task(tid, result_summary="a summary-only edit")          # no status: the wait stands
    assert _wait(store, tid) == ("NEEDS_USER", "owner", "asked")
    store.update_task(tid, status="PENDING")
    assert _wait(store, tid) == ("PENDING", "", "")
    store.update_task(tid, status="READY")
    assert _wait(store, tid) == ("PENDING", "", "")
    store.update_task(tid, status="PAUSED")
    assert _wait(store, tid) == ("NEEDS_USER", "owner", "paused")
    with pytest.raises(ValueError):
        store.update_task(tid, status="NEEDS_USER", wait_on="someone")
    t2 = store.add_task(pid, "y", status="BLOCKED")
    assert _wait(store, t2) == ("BLOCKED", "dependency", "blocked")       # r5: not the cascade's code


def test_the_event_does_not_list_the_wait_columns(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "x")
    store.update_task(tid, status="NEEDS_USER")
    ev = store.latest_task_event(pid, tid, ("task_updated",))
    assert ev["payload"]["fields"] == ["status"]


# ───────────────────────────── the tree ─────────────────────────────

def test_the_tree_carries_and_persists_the_wait(store):
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    tid = plan.add_task("Build it")
    plan.update_status(tid, TaskStatus.NEEDS_USER, result="x", wait=("owner_turn", "gate_form", "why"))
    assert _wait(store, tid) == ("NEEDS_USER", "owner_turn", "gate_form")
    assert ProjectPlan(store, pid).tree.nodes[tid].wait_on == "owner_turn"
    plan.update_status(tid, TaskStatus.PENDING)
    assert _wait(store, tid) == ("PENDING", "", "")
    plan.update_status(tid, TaskStatus.NEEDS_USER, result="flagged")
    assert _wait(store, tid) == ("NEEDS_USER", "owner", "asked")


def test_a_rollup_refusal_is_a_rollup_wait_and_clears_on_done(store):
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    parent = plan.add_task("Write it up → report.md")
    child = plan.add_task("draft", parent_id=parent)
    plan.update_status(child, TaskStatus.DONE, result="drafted")
    assert _wait(store, parent) == ("NEEDS_USER", "owner", "rollup_deliverable")
    assert store.get_task(parent)["result_summary"].startswith("not closed:")
    plan = ProjectPlan(store, pid)
    plan.update_status(parent, TaskStatus.DONE)          # the owner's way, no result
    t = store.get_task(parent)
    assert (t["status"], t["wait_on"], t["result_summary"]) == ("DONE", "", "")


def test_a_rollup_restate_is_judged_by_the_field_not_the_text(store):
    # a park of ANOTHER kind whose text happens to start "not closed:" is left alone
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    parent = plan.add_task("Write it up → report.md")
    child = plan.add_task("draft", parent_id=parent)
    store.update_task(parent, status="NEEDS_USER", result_summary="not closed: I need your logo first",
                      wait_on="owner", wait_code="asked")
    plan = ProjectPlan(store, pid)
    plan.update_status(child, TaskStatus.DONE, result="drafted")
    t = store.get_task(parent)
    assert t["wait_code"] == "asked" and t["result_summary"] == "not closed: I need your logo first"


# ───────────────────────────── readers ─────────────────────────────

def _exe(calls):
    async def exe(ctx, desc, **kw):
        calls.append(desc)
        return CodingResult(True, "built", files=[])
    return exe


async def _runner(name, args):
    return "Result 1: a long relevant page about the topic. " * 20


async def test_the_owner_s_run_takes_back_a_hold_by_its_field(store):
    pid = store.create_project("P", kind="CODING")
    store.update_project(pid, status="NEEDS_USER")
    tid = store.add_task(pid, "Build the parser")
    store.update_task(tid, status="NEEDS_USER", result_summary="the owner's run decides",
                      wait_on="owner_turn", wait_code="gate_form")
    calls = []
    await PA.advance_once(_actx(store), pid, tool_runner=_runner, coding_executor=_exe(calls), owner_requested=True)
    assert calls and store.get_task(tid)["status"] == "DONE"


async def test_a_text_mark_without_the_field_is_not_a_hold(store):
    # the mark is text now: a model's own summary ending in the words is not a hold
    pid = store.create_project("P", kind="CODING")
    store.update_project(pid, status="NEEDS_USER")
    tid = store.add_task(pid, "Build the parser")
    store.update_task(tid, status="NEEDS_USER", result_summary=f"why? — {PA._HELD_MARK}",
                      wait_on="owner", wait_code="asked")
    calls = []
    res = await PA.advance_once(_actx(store), pid, tool_runner=_runner, coding_executor=_exe(calls), owner_requested=True)
    assert not calls and res.classification == "blocked"


async def test_an_unattended_hold_writes_its_field(store):
    pid = store.create_project("P", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Verify the export works")
    calls = []
    await PA.advance_once(_actx(store), pid, tool_runner=_runner, coding_executor=_exe(calls), owner_requested=False)
    assert _wait(store, tid) == ("NEEDS_USER", "owner_turn", "verify_task")


async def test_the_owner_click_over_a_constraint_hold_reads_the_field(store, monkeypatch):
    from ghost_agent.api import projects_routes as R
    pid = store.create_project("P")
    tid = store.add_task(pid, "Build it")
    store.update_task(tid, status="NEEDS_USER", result_summary="not closed: constraints: unaudited",
                      wait_on="owner", wait_code="rollup_constraints", wait_why="constraints: unaudited")
    monkeypatch.setattr(R, "_store", lambda r: store)
    await R.update_task(pid, tid, R.TaskUpdate(status="DONE"), SimpleNamespace(headers={}))
    ev = store.latest_task_event(pid, tid, ("close_override",))
    assert ev and ev["payload"]["gate"] == "constraints" and "unaudited" in ev["payload"]["reason"]


# ───────────────────────────── what the model sees ─────────────────────────────

def test_the_briefing_shows_five_states(store):
    from ghost_agent.core.prompts import build_project_briefing
    pid = store.create_project("P")
    a = store.add_task(pid, "held one")
    store.update_task(a, status="NEEDS_USER", wait_on="owner_turn", wait_code="held")
    b = store.add_task(pid, "asked one")
    store.update_task(b, status="NEEDS_USER", result_summary="which?")
    store.add_task(pid, "open one")
    c = store.add_task(pid, "running one")
    store.update_task(c, status="IN_PROGRESS")
    out = build_project_briefing(store, pid)
    assert "(WAITING(owner_turn))" in out and "(WAITING(owner))" in out
    assert "(PENDING)" in out and "(RUNNING)" in out
    _open = [l for l in out.split("OPEN TASKS", 1)[1].splitlines()[1:] if l.startswith("  - [")]
    assert len(_open) == 4 and not any("NEEDS_USER" in l or "IN_PROGRESS" in l for l in _open)


async def test_task_list_shows_and_filters_the_five_states(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _tool_ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="P", goal="g")
    pid = ctx.current_project_id
    a = store.add_task(pid, "held")
    store.update_task(a, status="NEEDS_USER", wait_on="owner_turn", wait_code="held", wait_why="check it")
    store.add_task(pid, "open")
    out = json.loads(str(await tool_manage_projects(ctx, action="task_list")))
    states = {t["description"]: t["state"] for t in out["tasks"]}
    assert states == {"held": "WAITING(owner_turn: check it)", "open": "PENDING"}
    out = json.loads(str(await tool_manage_projects(ctx, action="task_list", status_filter="waiting")))
    assert [t["description"] for t in out["tasks"]] == ["held"]


def test_is_hold_needs_the_status_and_the_field():
    assert is_hold({"status": "NEEDS_USER", "wait_on": "owner_turn"})
    assert not is_hold({"status": "PENDING", "wait_on": "owner_turn"})
    assert not is_hold({"status": "NEEDS_USER", "wait_on": "owner", "result_summary": "not closed unattended"})


# ───────────────────────────── review r1 ─────────────────────────────

async def test_the_model_re_parking_a_hold_makes_it_an_owner_wait(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _tool_ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="P", goal="g")
    pid = ctx.current_project_id
    tid = store.add_task(pid, "Build it")
    ProjectPlan(store, pid).update_status(tid, TaskStatus.NEEDS_USER, result="built — unverified",
                                          wait=("owner_turn", "unverified_build", "x"))
    await tool_manage_projects(ctx, action="task_update", task_id=tid, status="needs_user",
                               result="the owner must approve the wording first")
    assert _wait(store, tid) == ("NEEDS_USER", "owner", "asked")


def test_a_hold_that_is_blocked_then_asked_is_not_a_hold_again(store):
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    tid = plan.add_task("x")
    plan.update_status(tid, TaskStatus.NEEDS_USER, wait=("owner_turn", "held", "x"))
    plan.update_status(tid, TaskStatus.BLOCKED)
    assert _wait(store, tid) == ("BLOCKED", "dependency", "blocked")
    plan.update_status(tid, TaskStatus.NEEDS_USER, result="which DB?")
    assert _wait(store, tid) == ("NEEDS_USER", "owner", "asked")


def test_the_boot_reaper_writes_pending_not_ready(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "x")
    store.update_task(tid, status="IN_PROGRESS")
    store.reset_orphaned_in_progress(older_than_seconds=-1)
    assert store.get_task(tid)["status"] == "PENDING"


def test_a_store_without_the_wait_columns_still_writes(tmp_path, monkeypatch):
    s = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = s.create_project("P")
    db = next(Path(tmp_path / "m").rglob("*.db"))
    con = sqlite3.connect(db)
    for col in ("wait_on", "wait_code", "wait_why"):
        con.execute(f"ALTER TABLE tasks DROP COLUMN {col}")
    con.commit()
    con.close()
    s._wait_cols_ok = False
    tid = s.add_task(pid, "x")
    s.update_task(tid, status="NEEDS_USER", wait_on="owner_turn")
    assert s.get_task(tid)["status"] == "NEEDS_USER"


# ───────────────────────────── review r2 ─────────────────────────────

def _exe_raises(exc):
    async def exe(ctx, desc, **kw):
        raise exc
    return exe


async def test_a_claimed_hold_comes_back_when_the_owner_s_run_crashes(store):
    pid = store.create_project("P", kind="CODING")
    store.update_project(pid, status="NEEDS_USER")
    tid = store.add_task(pid, "Build the parser")
    store.update_task(tid, status="NEEDS_USER", result_summary="built — needs a check",
                      wait_on="owner_turn", wait_code="unverified_build", wait_why="needs a check")
    await PA.advance_once(_actx(store), pid, tool_runner=_runner,
                          coding_executor=_exe_raises(RuntimeError("boom")), owner_requested=True)
    assert _wait(store, tid) == ("NEEDS_USER", "owner_turn", "unverified_build")
    assert store.get_task(tid)["result_summary"] == "built — needs a check"
    assert store.claimed_hold(pid, tid) is None


async def test_a_claimed_hold_comes_back_when_the_owner_s_run_is_stopped(store):
    import asyncio
    pid = store.create_project("P", kind="CODING")
    store.update_project(pid, status="NEEDS_USER")
    tid = store.add_task(pid, "Build the parser")
    store.update_task(tid, status="NEEDS_USER", wait_on="owner_turn", wait_code="gate_form")
    with pytest.raises(asyncio.CancelledError):
        await PA.advance_once(_actx(store), pid, tool_runner=_runner,
                              coding_executor=_exe_raises(asyncio.CancelledError()), owner_requested=True)
    assert _wait(store, tid) == ("NEEDS_USER", "owner_turn", "gate_form")


async def test_a_finished_claim_is_not_given_back_later(store):
    pid = store.create_project("P", kind="CODING")
    store.update_project(pid, status="NEEDS_USER")
    tid = store.add_task(pid, "Build the parser")
    store.update_task(tid, status="NEEDS_USER", wait_on="owner_turn", wait_code="gate_form")
    calls = []
    await PA.advance_once(_actx(store), pid, tool_runner=_runner, coding_executor=_exe(calls), owner_requested=True)
    assert store.get_task(tid)["status"] == "DONE"
    store.update_task(tid, status="IN_PROGRESS")            # a later ordinary claim
    assert store.release_claim(tid) == "PENDING"


def test_the_boot_reaper_gives_a_claimed_hold_back(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Build it")
    store.update_task(tid, status="IN_PROGRESS")
    store.log_event(pid, tid, "hold_claimed", {"wait_on": "owner_turn", "wait_code": "research_missed",
                                               "wait_why": "no source", "summary": "held: no source"})
    other = store.add_task(pid, "Other")
    store.update_task(other, status="IN_PROGRESS")
    assert store.reset_orphaned_in_progress(older_than_seconds=0) == 2
    assert _wait(store, tid) == ("NEEDS_USER", "owner_turn", "research_missed")
    assert _wait(store, other) == ("PENDING", "", "")


def test_a_failed_child_does_not_overwrite_an_owner_wait_on_its_parent(store):
    plan = ProjectPlan(store, store.create_project("P"))
    ph = plan.add_task("Phase")
    a = plan.add_task("step one", parent_id=ph)
    b = plan.add_task("step two", parent_id=ph)
    plan.update_status(ph, TaskStatus.NEEDS_USER, wait=("owner", "paused", "paused by the owner"))
    plan.update_status(a, TaskStatus.FAILED, failure_reason="x")
    assert _wait(store, ph) == ("NEEDS_USER", "owner", "paused")
    plan.update_status(a, TaskStatus.PENDING)
    assert _wait(store, ph) == ("NEEDS_USER", "owner", "paused")
    assert ProjectPlan(store, plan.project_id).next_ready_leaf() is None
    # its children closing does not close it either
    plan.update_status(a, TaskStatus.DONE, result="done a")
    plan.update_status(b, TaskStatus.DONE, result="done b")
    assert store.get_task(ph)["status"] == "NEEDS_USER"


# ───────────────────────────── battery pins ─────────────────────────────

def test_a_task_added_ready_or_paused_is_stored_folded(store):
    pid = store.create_project("P")
    r = store.add_task(pid, "ready one", status="READY")
    p = store.add_task(pid, "paused one", status="PAUSED")
    assert _wait(store, r) == ("PENDING", "", "")
    assert _wait(store, p) == ("NEEDS_USER", "owner", "paused")


def test_a_paused_write_says_the_owner_paused_it(store):
    pid = store.create_project("P")
    t = store.add_task(pid, "phase")
    store.update_task(t, status="PAUSED", result_summary="half done")
    assert store.get_task(t)["wait_why"] == "paused by the owner"


def test_the_tree_clears_a_wait_a_cascade_lifted(store):
    plan = ProjectPlan(store, store.create_project("P"))
    ph = plan.add_task("Phase")
    a = plan.add_task("step", parent_id=ph)
    plan.update_status(a, TaskStatus.FAILED, failure_reason="x")
    assert plan.tree.nodes[ph].wait_code == "child_failed"
    plan.update_status(a, TaskStatus.PENDING)
    assert plan.tree.nodes[ph].status == TaskStatus.PENDING
    assert (plan.tree.nodes[ph].wait_on, plan.tree.nodes[ph].wait_code) == ("", "")


# ───────────────────────────── review r3 ─────────────────────────────

def test_a_released_parent_whose_children_closed_is_closed(store):
    plan = ProjectPlan(store, store.create_project("P"))
    ph = plan.add_task("Phase")
    a = plan.add_task("step", parent_id=ph)
    plan.update_status(ph, TaskStatus.NEEDS_USER, wait=("owner", "asked", "which?"))
    plan.update_status(a, TaskStatus.DONE, result="done a, checked")
    assert store.get_task(ph)["status"] == "NEEDS_USER"
    plan.update_status(ph, TaskStatus.PENDING)
    assert store.get_task(ph)["status"] == "DONE"


def test_a_released_parent_with_a_failed_child_is_blocked(store):
    plan = ProjectPlan(store, store.create_project("P"))
    ph = plan.add_task("Phase")
    a = plan.add_task("step", parent_id=ph)
    plan.add_task("step two", parent_id=ph)
    plan.update_status(ph, TaskStatus.NEEDS_USER, wait=("owner", "paused", "paused"))
    plan.update_status(a, TaskStatus.FAILED, failure_reason="x")
    plan.update_status(ph, TaskStatus.PENDING)
    assert _wait(store, ph) == ("BLOCKED", "dependency", "child_failed")


def test_a_claim_ends_when_anyone_moves_the_task_on(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Build")
    store.log_event(pid, tid, "hold_claimed", {"wait_on": "owner_turn", "wait_code": "held", "summary": "old"})
    store.update_task(tid, status="IN_PROGRESS")
    store.update_task(tid, status="DONE", result_summary="the model closed it")
    assert store.claimed_hold(pid, tid) is None
    store.update_task(tid, status="IN_PROGRESS")        # a later ordinary claim
    assert store.release_claim(tid) == "PENDING"


def test_a_hold_that_cannot_be_restored_is_never_left_runnable(store, monkeypatch):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Build")
    store.log_event(pid, tid, "hold_claimed", {"wait_on": "owner_turn", "wait_code": "held"})
    store.update_task(tid, status="IN_PROGRESS")
    real = store.update_task

    def flaky(task_id, **kw):
        if kw.get("status") == "NEEDS_USER":
            raise sqlite3.OperationalError("database is locked")
        return real(task_id, **kw)
    monkeypatch.setattr(store, "update_task", flaky)
    store.reset_orphaned_in_progress(older_than_seconds=0)
    assert store.get_task(tid)["status"] == "IN_PROGRESS"


async def test_the_claim_is_logged_before_the_task_is_claimed(store, monkeypatch):
    pid = store.create_project("P", kind="CODING")
    store.update_project(pid, status="NEEDS_USER")
    tid = store.add_task(pid, "Build the parser")
    store.update_task(tid, status="NEEDS_USER", wait_on="owner_turn", wait_code="gate_form")
    seen = []
    real = store.update_task

    def spy(task_id, **kw):
        if kw.get("status") == "IN_PROGRESS":
            seen.append(store.claimed_hold(pid, task_id))
        return real(task_id, **kw)
    monkeypatch.setattr(store, "update_task", spy)
    await PA.advance_once(_actx(store), pid, tool_runner=_runner, coding_executor=_exe([]), owner_requested=True)
    assert seen and seen[0] and seen[0]["wait_code"] == "gate_form"


def test_a_dependency_waiting_for_the_owner_is_not_the_model_s_to_close(store):
    from ghost_agent.core.task_close import gate_dependencies
    plan = ProjectPlan(store, store.create_project("P"))
    a = plan.add_task("ask the owner")
    b = plan.add_task("report", depends_on=[a])
    plan.update_status(a, TaskStatus.NEEDS_USER, wait=("owner", "asked", "which?"))
    r = gate_dependencies(plan.tree, b)
    assert r and "stop" in r.fix and "close the task(s) it depends on first" not in r.fix


# ───────────────────────────── review r4 ─────────────────────────────

def test_a_reopened_done_parent_stays_open(store):
    plan = ProjectPlan(store, store.create_project("P"))
    ph = plan.add_task("Phase")
    a = plan.add_task("step", parent_id=ph)
    plan.add_task("later", depends_on=[ph])
    plan.update_status(a, TaskStatus.DONE, result="done, checked")
    assert store.get_task(ph)["status"] == "DONE"
    plan.update_status(ph, TaskStatus.PENDING)
    assert store.get_task(ph)["status"] == "PENDING"


def test_a_claim_whose_write_failed_ends_with_the_next_status_write(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Build")
    store.update_task(tid, status="NEEDS_USER", wait_on="owner_turn", wait_code="held")
    store.log_event(pid, tid, "hold_claimed", {"wait_on": "owner_turn", "wait_code": "held", "summary": "old"})
    store.update_task(tid, status="DONE", result_summary="closed by the model")   # NEEDS_USER → DONE
    store.update_task(tid, status="IN_PROGRESS")
    assert store.release_claim(tid) == "PENDING"


@pytest.mark.parametrize("dep_status,word", [("IN_PROGRESS", "wait for it"), ("FAILED", "redo it")])
def test_a_running_or_failed_dependency_is_not_closed_by_the_model(store, dep_status, word):
    from ghost_agent.core.task_close import gate_dependencies
    plan = ProjectPlan(store, store.create_project("P"))
    a = plan.add_task("experiment")
    b = plan.add_task("report", depends_on=[a])
    plan.update_status(a, TaskStatus[dep_status], failure_reason="x" if dep_status == "FAILED" else "")
    r = gate_dependencies(plan.tree, b)
    assert r and word in r.fix


# ───────────────────────────── review r4 (2) ─────────────────────────────


def test_a_failed_child_closed_done_lifts_its_block(store):
    plan = ProjectPlan(store, store.create_project("P"))
    ph = plan.add_task("Phase")
    a = plan.add_task("one", parent_id=ph)
    b = plan.add_task("two", parent_id=ph)
    plan.update_status(a, TaskStatus.FAILED, failure_reason="x")
    assert store.get_task(ph)["status"] == "BLOCKED"
    plan.update_status(a, TaskStatus.DONE, result="fixed and checked")
    assert store.get_task(ph)["status"] == "PENDING"
    assert ProjectPlan(store, plan.project_id).next_ready_leaf().id == b


def test_the_dependency_fix_names_every_kind(store):
    from ghost_agent.core.task_close import gate_dependencies
    plan = ProjectPlan(store, store.create_project("P"))
    ph = plan.add_task("experiments")
    k = plan.add_task("exp 1", parent_id=ph)
    held = plan.add_task("model run")
    asked = plan.add_task("pick people")
    rep = plan.add_task("report", depends_on=[ph, held, asked])
    plan.update_status(k, TaskStatus.FAILED, failure_reason="x")
    plan.update_status(held, TaskStatus.NEEDS_USER, wait=("owner_turn", "model_experiment", "run it"))
    plan.update_status(asked, TaskStatus.NEEDS_USER, wait=("owner", "asked", "who?"))
    fix = gate_dependencies(plan.tree, rep).fix
    assert f"{ph} is held up by failed task(s) {k}" in fix
    assert f"{held} is held for the owner's own run" in fix
    assert f"{asked} waits for the owner" in fix


# ───────────────────────────── review r5 ─────────────────────────────

def test_a_parent_the_model_blocked_stays_blocked_when_a_child_closes(store):
    plan = ProjectPlan(store, store.create_project("P"))
    ph = plan.add_task("Integrate the API")
    a = plan.add_task("client", parent_id=ph)
    plan.add_task("tests", parent_id=ph)
    plan.update_status(ph, TaskStatus.BLOCKED, result="waiting on the API key")
    assert _wait(store, ph) == ("BLOCKED", "dependency", "blocked")
    plan.update_status(a, TaskStatus.DONE, result="client written, checked")
    assert store.get_task(ph)["status"] == "BLOCKED"


def test_a_nested_failure_names_the_failed_leaf(store):
    from ghost_agent.core.task_close import gate_dependencies
    plan = ProjectPlan(store, store.create_project("P"))
    p = plan.add_task("experiments")
    c = plan.add_task("batch 1", parent_id=p)
    leaf = plan.add_task("exp 1", parent_id=c)
    rep = plan.add_task("report", depends_on=[p])
    plan.update_status(leaf, TaskStatus.FAILED, failure_reason="x")
    assert store.get_task(c)["status"] == "BLOCKED" and store.get_task(p)["status"] == "BLOCKED"
    fix = gate_dependencies(plan.tree, rep).fix
    assert f"held up by failed task(s) {leaf}" in fix


def test_a_model_blocked_dependency_is_not_called_failed(store):
    from ghost_agent.core.task_close import gate_dependencies
    plan = ProjectPlan(store, store.create_project("P"))
    a = plan.add_task("get the vendor key")
    rep = plan.add_task("report", depends_on=[a])
    plan.update_status(a, TaskStatus.BLOCKED, result="waiting on the vendor")
    fix = gate_dependencies(plan.tree, rep).fix
    assert "is blocked" in fix and "redo" not in fix


def test_a_refusal_text_does_not_survive_as_the_done_record(store):
    plan = ProjectPlan(store, store.create_project("P"))
    ph = plan.add_task("Phase")
    a = plan.add_task("step", parent_id=ph)
    plan.update_status(ph, TaskStatus.NEEDS_USER, result="not closed: deliverable: x.md does not exist",
                       wait=("owner", "rollup_deliverable", "x.md does not exist"))
    plan.update_status(a, TaskStatus.FAILED, failure_reason="broke")
    assert store.get_task(ph)["status"] == "BLOCKED"
    plan.update_status(a, TaskStatus.DONE, result="fixed, checked")
    row = store.get_task(ph)
    assert row["status"] == "DONE" and not row["result_summary"].startswith("not closed:")


# ───────────────────────────── review r6 ─────────────────────────────

def test_the_cascade_never_recodes_a_model_block(store):
    plan = ProjectPlan(store, store.create_project("P"))
    ph = plan.add_task("Integrate")
    a = plan.add_task("client", parent_id=ph)
    plan.update_status(ph, TaskStatus.BLOCKED, result="waiting on the vendor key")
    plan.update_status(a, TaskStatus.FAILED, failure_reason="x")
    plan.update_status(a, TaskStatus.PENDING)
    assert _wait(store, ph)[:2] == ("BLOCKED", "dependency") and store.get_task(ph)["wait_code"] == "blocked"


def test_a_cascade_block_with_an_old_reason_is_lifted(store):
    plan = ProjectPlan(store, store.create_project("P"))
    ph = plan.add_task("Phase")
    a = plan.add_task("step", parent_id=ph)
    plan.tree.nodes[ph].failure_reason = "Postcondition(s) not satisfied: old"
    plan.update_status(a, TaskStatus.FAILED, failure_reason="x")
    assert store.get_task(ph)["wait_code"] == "child_failed"
    plan.update_status(a, TaskStatus.PENDING)
    assert store.get_task(ph)["status"] == "PENDING"


def test_a_failed_parent_whose_subtasks_stand_is_not_sent_to_pending(store):
    from ghost_agent.core.task_close import gate_dependencies
    plan = ProjectPlan(store, store.create_project("P"))
    p = plan.add_task("phase")
    k = plan.add_task("part", parent_id=p)
    rep = plan.add_task("report", depends_on=[p])
    plan.update_status(k, TaskStatus.DONE, result="done, checked")
    plan.update_status(p, TaskStatus.FAILED, failure_reason="owner rejected")
    fix = gate_dependencies(plan.tree, rep).fix
    assert "replan it" in fix and "status=PENDING" not in fix


def test_a_parent_blocked_by_a_blocked_leaf_says_so(store):
    from ghost_agent.core.task_close import gate_dependencies
    plan = ProjectPlan(store, store.create_project("P"))
    p = plan.add_task("phase")
    leaf = plan.add_task("get key", parent_id=p)
    rep = plan.add_task("report", depends_on=[p])
    plan.update_status(leaf, TaskStatus.BLOCKED, result="vendor")
    assert store.get_task(p)["wait_code"] == "child_failed"
    fix = gate_dependencies(plan.tree, rep).fix
    assert f"blocked task(s) {leaf}" in fix and "redo" not in fix


def test_an_old_cascade_block_migrates_as_the_cascade(tmp_path):
    s = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = s.create_project("P")
    ph = s.add_task(pid, "phase")
    kid = s.add_task(pid, "step", parent_id=ph)
    db = next(Path(tmp_path / "m").rglob("*.db"))
    con = sqlite3.connect(db)
    for col in ("wait_on", "wait_code", "wait_why"):
        con.execute(f"ALTER TABLE tasks DROP COLUMN {col}")
    con.execute("UPDATE tasks SET status = 'BLOCKED' WHERE id = ?", (ph,))
    con.execute("UPDATE tasks SET status = 'FAILED' WHERE id = ?", (kid,))
    con.commit()
    con.close()
    s2 = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    assert _wait(s2, ph) == ("BLOCKED", "dependency", "child_failed")


# ───────────────────────────── review r7 ─────────────────────────────

def test_a_rollup_close_lifts_the_block_above(store):
    from ghost_agent.core.planning import DependencyType
    plan = ProjectPlan(store, store.create_project("P"))
    root = plan.add_task("root")
    d = plan.add_task("phase d", parent_id=root)
    k1 = plan.add_task("k1", parent_id=d)
    x = plan.add_task("x", parent_id=root)
    plan.update_status(k1, TaskStatus.FAILED, failure_reason="broke")
    assert store.get_task(root)["status"] == "BLOCKED"
    plan.update_status(k1, TaskStatus.PENDING)
    plan.update_status(k1, TaskStatus.DONE, result="fixed, checked")
    assert store.get_task(d)["status"] == "DONE"
    assert store.get_task(root)["status"] == "PENDING"
    assert ProjectPlan(store, plan.project_id).next_ready_leaf().id == x


def test_a_done_subtree_is_not_named_for_redo(store):
    from ghost_agent.core.task_close import _failed_leaves
    plan = ProjectPlan(store, store.create_project("P"))
    d = plan.add_task("d")
    a = plan.add_task("a", parent_id=d)
    t1 = plan.add_task("try1", parent_id=a)
    plan.add_task("try2", parent_id=a)
    c2 = plan.add_task("c2", parent_id=d)
    plan.tree.nodes[a].status = TaskStatus.DONE
    plan.tree.nodes[t1].status = TaskStatus.FAILED
    plan.tree.nodes[c2].status = TaskStatus.FAILED
    assert _failed_leaves(plan.tree, d) == [c2]


def test_a_failed_parent_with_open_subtasks_is_replanned_not_revived(store):
    from ghost_agent.core.task_close import gate_dependencies
    plan = ProjectPlan(store, store.create_project("P"))
    p = plan.add_task("phase")
    plan.add_task("part", parent_id=p)
    rep = plan.add_task("report", depends_on=[p])
    plan.tree.nodes[p].status = TaskStatus.FAILED
    fix = gate_dependencies(plan.tree, rep).fix
    assert "replan it" in fix and "status=PENDING" not in fix


def test_a_failed_middle_parent_whose_subtasks_stand_is_not_revived(store):
    # r8: X blocked by a FAILED non-leaf C whose children are DONE
    from ghost_agent.core.task_close import gate_dependencies
    plan = ProjectPlan(store, store.create_project("P"))
    x = plan.add_task("x")
    c = plan.add_task("c", parent_id=x)
    plan.add_task("open", parent_id=x)
    k = plan.add_task("k", parent_id=c)
    rep = plan.add_task("report", depends_on=[x])
    plan.tree.nodes[k].status = TaskStatus.DONE
    plan.tree.nodes[c].status = TaskStatus.FAILED
    plan.tree.nodes[x].status = TaskStatus.BLOCKED
    plan.tree.nodes[x].wait_code = "child_failed"
    fix = gate_dependencies(plan.tree, rep).fix
    assert "status=PENDING" not in fix and "replan" in fix


# ───────────────────────────── battery pins (final) ─────────────────────────────

def test_a_best_phase_recovered_lifts_the_block_above(store):
    from ghost_agent.core.planning import DependencyType
    plan = ProjectPlan(store, store.create_project("P"))
    root = plan.add_task("root")
    d = plan.add_task("phase d", parent_id=root, dependency_type=DependencyType.BEST)
    k1 = plan.add_task("k1", parent_id=d)
    k2 = plan.add_task("k2", parent_id=d)
    x = plan.add_task("x", parent_id=root)
    plan.update_status(k1, TaskStatus.FAILED, failure_reason="a")
    plan.update_status(k2, TaskStatus.FAILED, failure_reason="b")
    assert store.get_task(d)["status"] == "FAILED" and store.get_task(root)["status"] == "BLOCKED"
    plan.update_status(k1, TaskStatus.PENDING)
    plan.update_status(k1, TaskStatus.DONE, result="fixed, checked")
    assert store.get_task(d)["status"] == "DONE"
    assert store.get_task(root)["status"] == "PENDING"
    assert ProjectPlan(store, plan.project_id).next_ready_leaf().id == x


def test_a_direct_done_drops_a_refusal_text_after_a_cascade_block(store):
    plan = ProjectPlan(store, store.create_project("P"))
    ph = plan.add_task("Phase")
    a = plan.add_task("step", parent_id=ph)
    plan.update_status(ph, TaskStatus.NEEDS_USER, result="not closed: deliverable: x.md does not exist",
                       wait=("owner", "rollup_deliverable", "x.md does not exist"))
    plan.update_status(a, TaskStatus.FAILED, failure_reason="broke")
    assert store.get_task(ph)["wait_code"] == "child_failed"
    plan.update_status(ph, TaskStatus.DONE)
    assert not store.get_task(ph)["result_summary"].startswith("not closed:")


async def test_the_owner_click_is_not_judged_by_a_lookalike_summary(store, monkeypatch):
    from ghost_agent.api import projects_routes as R
    pid = store.create_project("P")
    tid = store.add_task(pid, "Build it")
    store.update_task(tid, status="NEEDS_USER", result_summary="not closed: constraints: my own note",
                      wait_on="owner", wait_code="asked", wait_why="which units?")
    monkeypatch.setattr(R, "_store", lambda r: store)
    await R.update_task(pid, tid, R.TaskUpdate(status="DONE"), SimpleNamespace(headers={}))
    ev = store.latest_task_event(pid, tid, ("close_override",))
    assert not (ev and ev["payload"].get("gate") == "constraints")
