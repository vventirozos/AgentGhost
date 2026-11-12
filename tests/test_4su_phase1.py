"""§4SU Phase 1 — one close check (core/task_close.check_close) on every path.

T = the model's task_update, H = the web console PATCH, A = the advancer
(pinned in test_4pg_autopilot_review / test_4su_phase5), R = the parent roll-up.
"""
from tests._refused import refused_ids, refused_reasons  # §4SU Phase 4
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import task_close as TC
from ghost_agent.core.planning import ProjectPlan, TaskStatus, TaskTree
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.projects import tool_manage_projects
from tests.test_4qa_project_task_integrity import _ctx
from tests.test_4ss_b45ec99f_fixes import EXP1
from tests.test_4su_phase0 import owner_turn  # noqa: F401  (fixture)


async def _project(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="Site", goal="Ship the site")
    return ctx, store, ctx.current_project_id


async def _done(ctx, tid, **kw):
    return json.loads(str(await tool_manage_projects(ctx, action="task_update", task_id=tid,
                                                     status="done", **kw)))


# ───────────────────────────── the verdict itself ─────────────────────────────

def test_an_unknown_actor_is_a_programming_error(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "x")
    with pytest.raises(ValueError):
        TC.check_close(store, pid, ProjectPlan(store, pid).tree, tid, actor="anyone")


def test_the_first_refusal_in_the_fixed_order_decides(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    dep = store.add_task(pid, "Build")
    tid = store.add_task(pid, "Write the report → report.md", depends_on=[dep],
                         postconditions=["tests pass with exit code zero"])
    tree = ProjectPlan(store, pid).tree
    v = TC.check_close(store, pid, tree, tid, actor="model", result="done")
    assert not v.ok and [r.gate for r in v.refused] == ["dependencies"]
    v = TC.check_close(store, pid, tree, tid, actor="model", result="done", skip={"dependencies"})
    assert v.first.gate == "deliverable"
    (Path(store.ensure_workspace(pid)) / "report.md").write_text("# Report\n\nThe findings: every number was gathered and checked twice.\n")
    v = TC.check_close(store, pid, tree, tid, actor="model", result="done", skip={"dependencies"})
    assert v.first.gate == "postconditions"


def test_the_owner_goes_over_every_gate_and_each_is_listed(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    dep = store.add_task(pid, "Build")
    tid = store.add_task(pid, "Write the report → report.md", depends_on=[dep],
                         postconditions=["tests pass with exit code zero"])
    v = TC.check_close(store, pid, ProjectPlan(store, pid).tree, tid, actor="owner", result="ok")
    assert v.ok and [r.gate for r in v.overridden] == ["dependencies", "deliverable", "postconditions"]


def test_a_model_run_waiver_needs_a_reason_and_one_task(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, EXP1)
    tree = ProjectPlan(store, pid).tree
    assert TC.check_close(store, pid, tree, tid, actor="model", result="ran it").first.gate == "model_run"
    short = TC.check_close(store, pid, tree, tid, actor="model", result="ran it", override={"model_run": "no"})
    assert short.first.gate == "model_run" and "at least" in short.first.fix
    why = "the scores are hand-computed rubric numbers, not model answers"
    many = TC.check_close(store, pid, tree, tid, actor="model", result="ran it",
                          override={"model_run": why}, single_task=False)
    assert not many.ok
    ok = TC.check_close(store, pid, tree, tid, actor="model", result="ran it", override={"model_run": why})
    assert ok.ok and ok.waived == {"model_run": why}


def test_the_advancer_and_roll_up_cannot_waive(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Summarise", postconditions=["tests pass with exit code zero"])
    tree = ProjectPlan(store, pid).tree
    for actor in ("advancer_owner", "advancer_unattended", "rollup"):
        v = TC.check_close(store, pid, tree, tid, actor=actor, result="ok",
                           override={"postconditions": "it no longer applies to this task"})
        assert v.first.gate == "postconditions", actor


def test_hold_kinds_match_the_phase5_recheck():
    from ghost_agent.core.project_advancer import HOLD_KINDS_RECHECKED
    assert TC.hold_kind("deliverable") in HOLD_KINDS_RECHECKED
    assert TC.hold_kind("model_run") in HOLD_KINDS_RECHECKED
    assert TC.hold_kind("postconditions") == "refused_postconditions"
    assert TC.hold_kind("postconditions") not in HOLD_KINDS_RECHECKED


# ───────────────────────────── T: the model's close ─────────────────────────────

async def test_a_refused_close_lists_the_gate_and_keeps_the_legacy_key(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    tid = store.add_task(pid, "Write the report → report.md")
    out = await _done(ctx, tid, result="wrote it")
    assert store.get_task(tid)["status"] == "PENDING"
    assert out["refused"][0]["gate"] == "deliverable" and out["refused"][0]["task_id"] == tid
    assert refused_ids(out, "undelivered"), "legacy key dropped before Phase 4"
    assert "agent_instruction_refused" in out


async def test_a_postcondition_waiver_closes_and_is_recorded(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    tid = store.add_task(pid, "Summarise", postconditions=["tests pass with exit code zero"])
    why = "there are no tests in a prose summary"
    out = await _done(ctx, tid, result="summary written", override={"postconditions": why})
    assert store.get_task(tid)["status"] == "DONE"
    assert out["overrides"] == [{"task_id": tid, "gate": "postconditions", "reason": why}]
    ev = store.latest_task_event(pid, tid, ("close_override",))
    assert ev and ev["payload"]["gate"] == "postconditions"
    assert "OVERRIDE postconditions" in str(store.get_ledger(pid))


async def test_an_override_on_a_gate_that_cannot_be_waived_is_ignored(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    # §4UI: "deliverable" became waivable — a dependency still is not
    dep = store.add_task(pid, "Gather the data")
    tid = store.add_task(pid, "Write the report", depends_on=[dep])
    out = await _done(ctx, tid, result="wrote it", override={"dependencies": "the data is not needed"})
    assert store.get_task(tid)["status"] == "PENDING" and out["refused"][0]["gate"] == "dependencies"


async def test_an_override_given_as_json_text_is_read(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    tid = store.add_task(pid, "Summarise", postconditions=["tests pass with exit code zero"])
    await _done(ctx, tid, result="summary", override=json.dumps({"postconditions": "no tests in prose"}))
    assert store.get_task(tid)["status"] == "DONE"


# ───────────────────────────── H: the owner's click ─────────────────────────────

async def test_the_owner_s_click_closes_and_records_each_gate(tmp_path, monkeypatch):
    from ghost_agent.api import projects_routes as R
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Write the report → report.md", postconditions=["tests pass with exit code zero"])
    monkeypatch.setattr(R, "_store", lambda r: store)
    await R.update_task(pid, tid.upper(), R.TaskUpdate(status="DONE"), SimpleNamespace(headers={}))
    assert store.get_task(tid)["status"] == "DONE", "the owner's click always closes"
    evs = store.list_events(pid, event_type="close_override")
    assert sorted(e["payload"]["gate"] for e in evs) == ["deliverable", "postconditions"]
    assert all(e["payload"]["actor"] == "owner" for e in evs)


async def test_the_click_is_judged_on_the_edited_postconditions(tmp_path, monkeypatch):
    from ghost_agent.api import projects_routes as R
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Summarise", postconditions=["tests pass with exit code zero"])
    monkeypatch.setattr(R, "_store", lambda r: store)
    await R.update_task(pid, tid, R.TaskUpdate(status="DONE", postconditions=[]), SimpleNamespace(headers={}))
    assert store.get_task(tid)["status"] == "DONE"
    assert not store.list_events(pid, event_type="close_override")


# ───────────────────────────── R: the roll-up ─────────────────────────────

def test_a_roll_up_does_not_close_a_parent_whose_file_is_missing(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    top = store.add_task(pid, "Write the final report → report.md")
    kid = store.add_task(pid, "Gather the numbers", parent_id=top)
    ProjectPlan(store, pid).update_status(kid, TaskStatus.DONE, result="gathered")
    t = store.get_task(top)
    assert t["status"] == "NEEDS_USER" and t["result_summary"].startswith("not closed: deliverable")
    assert store.latest_task_event(pid, top, ("rollup_refused",))
    assert not store.latest_task_event(pid, top, ("human_gate_triggered",))


def test_a_roll_up_does_not_close_an_unrun_experiment_parent(tmp_path):
    # the §4ST bypass: an experiment parent closed by its children's roll-up
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    top = store.add_task(pid, EXP1)
    kid = store.add_task(pid, "Draft the prompts", parent_id=top)
    ProjectPlan(store, pid).update_status(kid, TaskStatus.DONE, result="drafted")
    assert store.get_task(top)["status"] == "NEEDS_USER"
    assert "model_run" in store.get_task(top)["result_summary"]


def test_a_roll_up_with_nothing_missing_still_closes(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    top = store.add_task(pid, "Write the final report → report.md")
    kid = store.add_task(pid, "Write it", parent_id=top)
    (Path(store.ensure_workspace(pid)) / "report.md").write_text("# Report\n\nThe findings: every number was gathered and checked twice.\n")
    ProjectPlan(store, pid).update_status(kid, TaskStatus.DONE, result="wrote report.md")
    assert store.get_task(top)["status"] == "DONE"


def test_a_bare_tree_has_no_checker_and_rolls_up_as_before():
    tree = TaskTree()
    top = tree.add_task("Write the final report → report.md")
    a = tree.add_task("Write it", parent_id=top)
    tree.update_status(a, TaskStatus.DONE, result="wrote")
    assert tree.nodes[top].status == TaskStatus.DONE


async def test_a_roll_up_refusal_shows_in_the_model_s_reply(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    top = store.add_task(pid, "Write the final report → report.md")
    kid = store.add_task(pid, "Gather the numbers", parent_id=top)
    out = await _done(ctx, kid, result="gathered")
    assert store.get_task(kid)["status"] == "DONE"
    assert any(n["task_id"] == top and "deliverable" in n["why"] for n in out.get("not_closed", []))


# ───────────────────────────── review r1 ─────────────────────────────

async def test_a_roll_up_refusal_text_is_not_postcondition_evidence(tmp_path, owner_turn):
    # r1 M1: "not closed: postconditions: not shown by the result: <pc>" quoted
    # the postcondition, so a close with no result passed on the echo
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    top = store.add_task(pid, "Integrate the modules", postconditions=["integration tests pass green"])
    kid = store.add_task(pid, "Wire module A", parent_id=top)
    await _done(ctx, kid, result="wired A")
    assert store.get_task(top)["result_summary"].startswith("not closed:")
    out = await _done(ctx, top)
    assert store.get_task(top)["status"] == "NEEDS_USER"
    assert out["refused"][0]["gate"] == "postconditions"


async def test_an_unattended_hold_text_is_not_postcondition_evidence(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    tid = store.add_task(pid, "Implement the parser", postconditions=["integration tests pass green"])
    plan = ProjectPlan(store, pid)
    v = TC.check_close(store, pid, plan.tree, tid, actor="advancer_unattended", result="wrote parser.py",
                       skip={"dependencies", "model_run", "evidence"})
    from ghost_agent.core.project_advancer import _HELD_MARK
    plan.update_status(tid, TaskStatus.NEEDS_USER, result=f"wrote parser.py — {v.first.reason} — {_HELD_MARK}",
                       wait=("owner_turn", "refused_deliverable", v.first.reason))
    out = await _done(ctx, tid)
    assert store.get_task(tid)["status"] == "NEEDS_USER" and out["refused"][0]["gate"] == "postconditions"


def test_a_real_stored_result_still_shows_a_postcondition(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Implement", postconditions=["integration tests pass green"])
    store.update_task(tid, result_summary="ran the suite: integration tests pass green")
    assert TC.check_close(store, pid, ProjectPlan(store, pid).tree, tid, actor="model").ok


def test_a_done_parent_is_not_re_judged_by_a_child_re_close(tmp_path):
    # r1 M2: the owner's click over the missing file was undone by a child re-close
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    par = store.add_task(pid, "Write the summary → summary.md")
    c1 = store.add_task(pid, "step a", parent_id=par)
    ProjectPlan(store, pid).update_status(c1, TaskStatus.DONE, result="did a")
    assert store.get_task(par)["status"] == "NEEDS_USER"
    ProjectPlan(store, pid).update_status(par, TaskStatus.DONE, result="owner says done", human_approved=True)
    ProjectPlan(store, pid).update_status(c1, TaskStatus.DONE, result="did a, re-verified")
    t = store.get_task(par)
    assert t["status"] == "DONE" and t["result_summary"] == "owner says done"


def test_a_roll_up_refusal_is_in_the_digest(tmp_path):
    # r1 M3: the project stopped NEEDS_USER and "While you were away" was empty
    from ghost_agent.core.project_digest import summarize_since
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P", goal="g")
    par = store.add_task(pid, "Write the final report → report.md")
    kid = store.add_task(pid, "Gather the numbers", parent_id=par)
    base = max([int(e["id"]) for e in store.list_events(pid, limit=100)] or [0])
    ProjectPlan(store, pid).update_status(kid, TaskStatus.DONE, result="gathered")
    r = summarize_since(store, base)
    assert any("report.md" in d for _, d in r.needs_user)


def test_a_parent_closing_later_drops_the_refusal_text(tmp_path):
    # r1 D2: the ALL parent closed DONE with "not closed: …" as its record
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    par = store.add_task(pid, "Write the final report → report.md")
    kid = store.add_task(pid, "Gather the numbers", parent_id=par)
    ProjectPlan(store, pid).update_status(kid, TaskStatus.DONE, result="gathered")
    (Path(store.ensure_workspace(pid)) / "report.md").write_text(
        "# Report\n\nThe findings: every number was gathered and checked twice.\n")
    k2 = store.add_task(pid, "Check the numbers", parent_id=par)
    ProjectPlan(store, pid).update_status(k2, TaskStatus.DONE, result="checked")
    t = store.get_task(par)
    assert t["status"] == "DONE" and not t["result_summary"].startswith("not closed:")
    assert "gathered" in t["result_summary"]


async def test_a_re_click_on_a_done_task_records_nothing_again(tmp_path, monkeypatch):
    # r1 L1
    from ghost_agent.api import projects_routes as R
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Write the report → report.md")
    monkeypatch.setattr(R, "_store", lambda r: store)
    for _ in range(2):
        await R.update_task(pid, tid, R.TaskUpdate(status="DONE"), SimpleNamespace(headers={}))
    assert len(store.list_events(pid, event_type="close_override")) == 1


async def test_the_owner_s_click_records_a_constraints_evidence_gate(tmp_path, monkeypatch):
    # r1 L2: the owner's check had no constraints, so this gate was never recorded
    from ghost_agent.api import projects_routes as R
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    store.update_project(pid, metadata={"constraints": ["no external libraries"]})
    tid = store.add_task(pid, "Write the parser")
    monkeypatch.setattr(R, "_store", lambda r: store)
    await R.update_task(pid, tid, R.TaskUpdate(status="DONE"), SimpleNamespace(headers={}))
    assert [e["payload"]["gate"] for e in store.list_events(pid, event_type="close_override")] == ["evidence"]


async def test_a_constraints_override_with_no_files_is_not_recorded(tmp_path, owner_turn):
    # r1 L4: no audit would have run, so nothing was overridden
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    store.update_project(pid, metadata={"constraints": ["no external libraries"]})
    tid = store.add_task(pid, "Write the parser")
    out = await _done(ctx, tid, result="wrote it with the stdlib only",
                      override={"constraints": "the owner approved one library"})
    assert store.get_task(tid)["status"] == "DONE"
    assert not out.get("overrides") and not store.list_events(pid, event_type="close_override")


# ───────────────────────────── review r2 ─────────────────────────────

async def test_a_re_click_never_demotes_the_owner_s_close(tmp_path, monkeypatch):
    # r2 M1: the second click skipped the check, so the postcondition demoted it
    from ghost_agent.api import projects_routes as R
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Tune", postconditions=["benchmark shows latency under 5 ms"])
    monkeypatch.setattr(R, "_store", lambda r: store)
    await R.update_task(pid, tid, R.TaskUpdate(status="DONE"), SimpleNamespace(headers={}))
    await R.update_task(pid, tid, R.TaskUpdate(status="DONE", result_summary="measured, see notes"),
                        SimpleNamespace(headers={}))
    assert store.get_task(tid)["status"] == "DONE"


async def test_the_click_reports_a_parent_it_parked(tmp_path, monkeypatch):
    # r2 L1
    from ghost_agent.api import projects_routes as R
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    par = store.add_task(pid, "Write the final report → report.md")
    kid = store.add_task(pid, "Gather", parent_id=par)
    monkeypatch.setattr(R, "_store", lambda r: store)
    out = await R.update_task(pid, kid, R.TaskUpdate(status="DONE"), SimpleNamespace(headers={}))
    assert out["status"] == "DONE" and out["not_closed"][0]["task_id"] == par


def test_an_owner_close_drops_the_refusal_text_and_the_grandparent_is_judged_on_real_results(tmp_path):
    # r2 M2: the parent's DONE kept "not closed: … <pc>" and the grandparent passed on it
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    pc = ["benchmark shows latency under 5 ms"]
    g = store.add_task(pid, "Ship", postconditions=pc)
    p = store.add_task(pid, "Tune", parent_id=g, postconditions=pc)
    k = store.add_task(pid, "Profile", parent_id=p)
    ProjectPlan(store, pid).update_status(k, TaskStatus.DONE, result="profiled")
    assert store.get_task(p)["result_summary"].startswith("not closed:")
    ProjectPlan(store, pid).update_status(p, TaskStatus.DONE, human_approved=True, postconditions_waived=True)
    assert store.get_task(p)["status"] == "DONE" and not store.get_task(p)["result_summary"].startswith("not closed:")
    assert store.get_task(g)["status"] == "NEEDS_USER"


def test_held_children_are_not_the_roll_up_s_evidence(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tree = ProjectPlan(store, pid).tree
    a = tree.add_task("x")
    tree.nodes[a].status = TaskStatus.DONE
    tree.nodes[a].result_summary = "not closed: postconditions: not shown by the result: y"
    par = tree.add_task("p")
    tree.nodes[par].children.append(a)
    assert tree._done_children_text(tree.nodes[par]) == ""


# ───────────────────────────── review r3 ─────────────────────────────

def _held_text(pc):
    from ghost_agent.core.project_advancer import _HELD_MARK
    return f"built x — not shown by the result: {pc} — {_HELD_MARK}"


@pytest.mark.parametrize("dep", ["BEST", "ALL", "ANY"])
def test_no_roll_up_branch_passes_on_a_hold_text(tmp_path, dep):
    # r3 M1: the BEST branch read the hold's quote of the postcondition as evidence
    from ghost_agent.core.planning import DependencyType
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    pc = "benchmark shows latency under 5 ms"
    par = store.add_task(pid, "Tune", postconditions=[pc], dependency_type=DependencyType[dep].value)
    kid = store.add_task(pid, "Build", parent_id=par)
    plan = ProjectPlan(store, pid)
    plan.tree.nodes[kid].result_summary = _held_text(pc)
    store.update_task(kid, result_summary=_held_text(pc))
    ProjectPlan(store, pid).update_status(kid, TaskStatus.DONE, human_approved=True)
    assert store.get_task(par)["status"] == "NEEDS_USER", dep


def test_a_new_refusal_reason_is_restated_and_logged(tmp_path):
    # r3 D2: the owner was still told the file was missing
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    par = store.add_task(pid, "Write the final report → report.md",
                         postconditions=["benchmark shows latency under 5 ms"])
    k1 = store.add_task(pid, "Gather", parent_id=par)
    ProjectPlan(store, pid).update_status(k1, TaskStatus.DONE, result="gathered")
    assert "deliverable" in store.get_task(par)["result_summary"]
    (Path(store.ensure_workspace(pid)) / "report.md").write_text(
        "# Report\n\nThe findings: every number was gathered and checked twice.\n")
    k2 = store.add_task(pid, "Check", parent_id=par)
    ProjectPlan(store, pid).update_status(k2, TaskStatus.DONE, result="checked")
    assert "postconditions" in store.get_task(par)["result_summary"]
    assert len(store.list_events(pid, event_type="rollup_refused")) == 2


def test_the_same_refusal_is_not_logged_twice(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    par = store.add_task(pid, "Write the final report → report.md")
    k1 = store.add_task(pid, "Gather", parent_id=par)
    k2 = store.add_task(pid, "Check", parent_id=par)
    ProjectPlan(store, pid).update_status(k1, TaskStatus.DONE, result="gathered")
    ProjectPlan(store, pid).update_status(k2, TaskStatus.DONE, result="checked")
    ProjectPlan(store, pid).update_status(k2, TaskStatus.DONE, result="checked again")
    assert len(store.list_events(pid, event_type="rollup_refused")) == 1


async def test_the_owner_s_override_records_are_written_with_the_close(tmp_path, monkeypatch):
    # r3 L2: one worker call does the close AND its records — when it returns
    # (a cancel at that await cannot stop the thread), both are in the store
    import asyncio as _aio
    from ghost_agent.api import projects_routes as R
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Write the report → report.md")
    monkeypatch.setattr(R, "_store", lambda r: store)
    seen = []
    orig = _aio.to_thread

    async def rec(fn, *a, **k):
        out = await orig(fn, *a, **k)
        seen.append(((store.get_task(tid) or {}).get("status"),
                     len(store.list_events(pid, event_type="close_override"))))
        return out
    monkeypatch.setattr(R.asyncio, "to_thread", rec)
    await R.update_task(pid, tid, R.TaskUpdate(status="DONE"), SimpleNamespace(headers={}))
    assert ("DONE", 1) in seen and not any(st == "DONE" and n == 0 for st, n in seen)


def test_a_crashing_evidence_filter_judges_on_nothing(monkeypatch):
    import ghost_agent.tools.projects as TP
    monkeypatch.setattr(TP, "_stored_evidence", lambda s: 1 / 0)
    tree = TaskTree()
    top = tree.add_task("p")
    a = tree.add_task("a", parent_id=top)
    tree.nodes[a].status = TaskStatus.DONE
    tree.nodes[a].result_summary = "did it"
    assert tree._done_children_text(tree.nodes[top]) == ""


# ───────────────────────────── review r4 ─────────────────────────────

def test_best_is_chosen_by_real_text_not_a_longer_gate_text(tmp_path):
    # r4 M1
    from ghost_agent.core.planning import DependencyType
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    pc = "benchmark shows latency under 5 ms"
    par = store.add_task(pid, "Tune", postconditions=[pc], dependency_type=DependencyType.BEST.value)
    k1 = store.add_task(pid, "A", parent_id=par)
    k2 = store.add_task(pid, "B", parent_id=par)
    store.update_task(k1, status="DONE", result_summary="human gate: owner approves the long deploy of everything")
    ProjectPlan(store, pid).update_status(k2, TaskStatus.DONE, result="benchmark shows latency under 5 ms")
    assert store.get_task(par)["status"] == "DONE"


async def test_a_roll_up_refusal_is_not_given_the_gate_instruction(tmp_path, owner_turn):
    # r4 M2: the model asked the owner to "approve" a task that is fixed, not approved
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    top = store.add_task(pid, "Write the final report → report.md")
    kid = store.add_task(pid, "Gather", parent_id=top)
    out = await _done(ctx, kid, result="gathered")
    ins = out["agent_instruction_not_closed"]
    assert "approve <task id>" not in ins and "fix what the reason names" in ins.lower() and ". Fix what" in ins


async def test_a_parent_is_listed_once_in_a_bulk_close(tmp_path, owner_turn):
    # r4 L1
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    top = store.add_task(pid, "Integrate", postconditions=["alpha calibrated", "bravo tested", "charlie shipped"])
    k1 = store.add_task(pid, "K1", parent_id=top)
    k2 = store.add_task(pid, "K2", parent_id=top)
    out = json.loads(str(await tool_manage_projects(ctx, action="task_update", task_ids=[k1, k2],
                                                    status="done", result="alpha calibrated; bravo tested")))
    assert [n["task_id"] for n in out["not_closed"]].count(top) == 1


def test_the_digest_lists_a_waiting_task_once(tmp_path):
    # r4 L2
    from ghost_agent.core.project_digest import summarize_since
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P", goal="g")
    par = store.add_task(pid, "Write the final report → report.md", postconditions=["benchmark shows latency under 5 ms"])
    base = max([int(e["id"]) for e in store.list_events(pid, limit=100)] or [0])
    k1 = store.add_task(pid, "Gather", parent_id=par)
    ProjectPlan(store, pid).update_status(k1, TaskStatus.DONE, result="gathered")
    (Path(store.ensure_workspace(pid)) / "report.md").write_text(
        "# Report\n\nThe findings: every number was gathered and checked twice.\n")
    k2 = store.add_task(pid, "Check", parent_id=par)
    ProjectPlan(store, pid).update_status(k2, TaskStatus.DONE, result="checked")
    assert len(store.list_events(pid, event_type="rollup_refused")) == 2
    assert len(summarize_since(store, base).needs_user) == 1


# ──────────────────────── review r5–r7 (bulk closes) ────────────────────────

async def test_a_refused_re_close_of_a_done_task_says_it_stays_done(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    tid = store.add_task(pid, "Calibrate", postconditions=["alpha calibrated"])
    await _done(ctx, tid, result="alpha calibrated")
    out = await _done(ctx, tid, result="also added docs")
    assert store.get_task(tid)["status"] == "DONE"
    assert not out.get("not_closed")
    assert "stays DONE" in out["refused"][0]["reason"]


async def test_a_parent_rolled_up_by_its_child_earlier_in_the_call_is_not_reported_held(tmp_path, owner_turn):
    # r6 repro C: [k, top] — the old keys said "held at non-DONE" for a DONE task
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    top = store.add_task(pid, "Build the snake game canvas page")
    k = store.add_task(pid, "Write the notes", parent_id=top)
    out = json.loads(str(await tool_manage_projects(ctx, action="task_update", task_ids=[k, top], status="done")))
    assert store.get_task(top)["status"] == "DONE"
    assert not refused_ids(out, "gated_unverified") and not out.get("not_closed")
    assert all("stays DONE" in r["reason"] for r in out.get("refused", []))


async def test_a_parent_that_closes_after_a_roll_up_refusal_is_not_listed(tmp_path, owner_turn):
    # r7 M2: [c1, ph] — c1's roll-up refused ph, then ph's own close (with a waiver) passed
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    ph = store.add_task(pid, "Benchmark", postconditions=["benchmark table with p-values"])
    c1 = store.add_task(pid, "Run it", parent_id=ph)
    out = json.loads(str(await tool_manage_projects(
        ctx, action="task_update", task_ids=[c1, ph], status="done", result="ran it",
        override={"postconditions": "the owner dropped the p-values"})))
    assert store.get_task(ph)["status"] == "DONE"
    assert ph not in [n["task_id"] for n in out.get("not_closed", [])]


async def test_a_grandparent_closes_in_one_bulk_call(tmp_path, owner_turn):
    # r7 M1: the r6 skip left GP waiting; the same work in two calls closed it
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    gp = store.add_task(pid, "Ship", postconditions=["latency under 5 ms"])
    p = store.add_task(pid, "Tune", parent_id=gp)
    c = store.add_task(pid, "Measure", parent_id=p)
    await tool_manage_projects(ctx, action="task_update", task_ids=[c, p], status="done",
                               result="measured: latency under 5 ms")
    assert store.get_task(gp)["status"] == "DONE"


def test_the_digest_drops_a_task_that_has_since_closed(tmp_path):
    from ghost_agent.core.project_digest import summarize_since
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P", goal="g")
    par = store.add_task(pid, "Write the final report → report.md")
    base = max([int(e["id"]) for e in store.list_events(pid, limit=100)] or [0])
    k1 = store.add_task(pid, "Gather", parent_id=par)
    ProjectPlan(store, pid).update_status(k1, TaskStatus.DONE, result="gathered")
    ProjectPlan(store, pid).update_status(par, TaskStatus.DONE, result="owner wrote it", human_approved=True)
    assert summarize_since(store, base).needs_user == []


# ───────────────────────────── review r8 ─────────────────────────────

def test_a_refused_re_close_alone_reads_as_nothing_changed():
    ins = TC.instruction_for([{"task_id": "t", "gate": "postconditions",
                               "reason": "x (the task stays DONE: this re-close changed nothing)", "fix": "f"}])
    assert "changed nothing" in ins and "NOT closed" not in ins
    mixed = TC.instruction_for([{"task_id": "t", "gate": "deliverable", "reason": "missing", "fix": "f"},
                                {"task_id": "u", "gate": "postconditions",
                                 "reason": "x (the task stays DONE: this re-close changed nothing)", "fix": "f"}])
    assert "NOT closed" in mixed


async def test_a_constraint_audit_refusal_of_a_re_close_says_it_stays_done(tmp_path, owner_turn, monkeypatch):
    # r8 M: the audit path still said "REFUSED to mark … DONE" for a DONE task
    import ghost_agent.core.build_gates as BG
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    tid = store.add_task(pid, "Write the engine", constraints=["Ghost plays directly"])
    (Path(store.ensure_workspace(pid)) / "game.py").write_text("print('engine')\n")
    await _done(ctx, tid, result="wrote game.py; Ghost plays directly", deliverables=["game.py"])
    assert store.get_task(tid)["status"] == "DONE"

    async def _bad(*a, **k):
        return (False, "game.py uses a heuristic engine")
    monkeypatch.setattr(BG, "constraint_gate", _bad)
    out = await _done(ctx, tid, result="tweaked it; Ghost plays directly", deliverables=["game.py"])
    assert store.get_task(tid)["status"] == "DONE"
    assert not refused_ids(out, "constraint_violations")
    assert "stays DONE" in out["refused"][0]["reason"] and out["refused"][0]["gate"] == "constraints"


def test_a_hidden_digest_row_does_not_count_its_project(tmp_path):
    from ghost_agent.core.project_digest import summarize_since
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P", goal="g")
    par = store.add_task(pid, "Write the final report → report.md")
    store.add_task(pid, "Still open")
    base = max([int(e["id"]) for e in store.list_events(pid, limit=100)] or [0])
    k1 = store.add_task(pid, "Gather", parent_id=par)
    ProjectPlan(store, pid).update_status(k1, TaskStatus.DONE, result="gathered")
    ProjectPlan(store, pid).update_status(par, TaskStatus.DONE, result="owner wrote it", human_approved=True)
    assert summarize_since(store, base).projects_touched == 0


def test_a_parent_waiting_on_the_owner_for_another_reason_keeps_its_note(tmp_path):
    # r3 D2 guard: only a roll-up refusal is re-stated, never another NEEDS_USER note
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    par = store.add_task(pid, "Write the final report → report.md")
    kid = store.add_task(pid, "Gather", parent_id=par)
    store.update_task(par, status="NEEDS_USER", result_summary="waiting for the owner's colour choice")
    ProjectPlan(store, pid).update_status(kid, TaskStatus.DONE, result="gathered")
    assert store.get_task(par)["result_summary"] == "waiting for the owner's colour choice"


async def test_a_re_stated_parent_is_listed_once_in_a_bulk_re_close(tmp_path, owner_turn):
    # r4 L1: each child re-close changed the reason, so the parent was listed per child
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _project(tmp_path)
    top = store.add_task(pid, "Integrate", postconditions=["alpha calibrated", "bravo tested", "charlie shipped"])
    k1 = store.add_task(pid, "K1", parent_id=top)
    k2 = store.add_task(pid, "K2", parent_id=top)
    await _done(ctx, k1, result="alpha calibrated")
    await _done(ctx, k2, result="bravo tested")
    out = json.loads(str(await tool_manage_projects(ctx, action="task_update", task_ids=[k1, k2],
                                                    status="done", result="charlie shipped")))
    assert [n["task_id"] for n in out.get("not_closed", [])].count(top) == 1


# ──────────── operator 11-02: a constrained parent waits for the owner ────────────

def _constrained_parent(tmp_path, with_file=True):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("Chess")
    top = store.add_task(pid, "Build the chess engine", constraints=["Ghost plays directly"])
    kid = store.add_task(pid, "Write the board", parent_id=top)
    if with_file:
        (Path(store.ensure_workspace(pid)) / "engine.py").write_text("def best_move(): return heuristic()\n")
        store.register_file_artifact(top, "engine.py", description="the engine")
    return store, pid, top, kid


def test_a_constrained_parent_with_files_is_not_rolled_up_unaudited(tmp_path):
    store, pid, top, kid = _constrained_parent(tmp_path)
    ProjectPlan(store, pid).update_status(kid, TaskStatus.DONE, result="board written")
    t = store.get_task(top)
    assert t["status"] == "NEEDS_USER" and t["result_summary"].startswith("not closed: constraints:")
    assert store.latest_task_event(pid, top, ("rollup_refused",))


def test_a_fileless_constrained_parent_still_rolls_up(tmp_path):
    store, pid, top, kid = _constrained_parent(tmp_path, with_file=False)
    ProjectPlan(store, pid).update_status(kid, TaskStatus.DONE, result="board written")
    assert store.get_task(top)["status"] == "DONE"


def test_project_constraints_hold_a_parent_with_files_too(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("Chess")
    store.update_project(pid, metadata={"constraints": ["no external engines"]})
    top = store.add_task(pid, "Build the engine")
    kid = store.add_task(pid, "Write the board", parent_id=top)
    (Path(store.ensure_workspace(pid)) / "engine.py").write_text("import stockfish\n")
    store.register_file_artifact(top, "engine.py", description="the engine")
    ProjectPlan(store, pid).update_status(kid, TaskStatus.DONE, result="board written")
    assert store.get_task(top)["status"] == "NEEDS_USER"


async def test_the_model_s_own_close_audits_and_closes_the_held_parent(tmp_path, owner_turn, monkeypatch):
    import ghost_agent.core.build_gates as BG
    owner_turn("a1b2c3d4")
    store, pid, top, kid = _constrained_parent(tmp_path)
    ctx = _ctx(store, tmp_path)
    ctx.current_project_id = pid
    ProjectPlan(store, pid).update_status(kid, TaskStatus.DONE, result="board written")
    audited = []

    async def _ok(ctx_, cons, files, **k):
        audited.append(sorted(files))
        return (True, "")
    monkeypatch.setattr(BG, "constraint_gate", _ok)
    await _done(ctx, top, result="engine searches itself; Ghost plays directly")
    assert audited == [["engine.py"]] and store.get_task(top)["status"] == "DONE"


async def test_the_audit_reads_the_parent_s_declared_file(tmp_path, owner_turn, monkeypatch):
    # r9 M1: a child wrote notes.md; the held parent's own close skipped the audit
    import ghost_agent.core.build_gates as BG
    owner_turn("a1b2c3d4")
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("Chess")
    top = store.add_task(pid, "Write the engine notes → notes.md", constraints=["Ghost plays directly"])
    kid = store.add_task(pid, "Draft them", parent_id=top)
    (Path(store.ensure_workspace(pid)) / "notes.md").write_text(
        "# Notes\n\nThe engine uses a fixed heuristic table for every move it plays.\n")
    store.register_file_artifact(kid, "notes.md", description="notes")
    ProjectPlan(store, pid).update_status(kid, TaskStatus.DONE, result="drafted")
    assert store.get_task(top)["status"] == "NEEDS_USER"
    ctx = _ctx(store, tmp_path)
    ctx.current_project_id = pid

    async def _bad(ctx_, cons, files, **k):
        return (False, "notes.md: a heuristic table, not Ghost playing")
    monkeypatch.setattr(BG, "constraint_gate", _bad)
    out = await _done(ctx, top, result="notes written; Ghost plays directly")
    assert store.get_task(top)["status"] == "NEEDS_USER"
    assert out["refused"][0]["gate"] == "constraints"


async def test_the_owner_s_click_records_the_audit_it_went_over(tmp_path, monkeypatch):
    # r9 L1
    from ghost_agent.api import projects_routes as R
    store, pid, top, kid = _constrained_parent(tmp_path)
    ProjectPlan(store, pid).update_status(kid, TaskStatus.DONE, result="board written")
    monkeypatch.setattr(R, "_store", lambda r: store)
    await R.update_task(pid, top, R.TaskUpdate(status="DONE", result_summary="I checked it"),
                        SimpleNamespace(headers={}))
    assert store.get_task(top)["status"] == "DONE"
    assert "constraints" in [e["payload"]["gate"] for e in store.list_events(pid, event_type="close_override")]
