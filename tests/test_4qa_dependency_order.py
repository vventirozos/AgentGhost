"""§4QA-d review r19: task `depends_on` must hold everywhere, not only in the
autopilot's picker. M2: an owner turn could close the report while every
experiment it rests on was open. m3: a chain held behind a BLOCKED/PAUSED
task was reported "project done". m6: the briefing counted parents twice and
showed tasks held for the turn as questions for the user."""
from tests._refused import refused_ids, refused_reasons  # §4SU Phase 4
import json
from types import SimpleNamespace

import pytest

from ghost_agent.core import project_advancer as pa
from ghost_agent.core.planning import TaskStatus
from ghost_agent.core.prompts import build_project_briefing
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import tool_manage_projects


def _ctx(store, tmp_path):
    return SimpleNamespace(
        project_store=store, scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None, contradiction_log=None, current_project_id=None,
        llm_client=None, args=SimpleNamespace(model="m"))


async def _project(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    ctx = _ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="Recursion")
    pid = ctx.current_project_id
    exp = store.add_task(pid, "Run the control condition")
    report = store.add_task(pid, "Write the final report", depends_on=[exp])
    return ctx, store, pid, exp, report


async def _done(ctx, *tids):
    out = await tool_manage_projects(ctx, action="task_update", task_ids=list(tids),
                                     status="done", result="did it")
    return json.loads(str(out))


async def test_a_task_cannot_close_while_its_dependency_is_open(tmp_path):
    ctx, store, pid, exp, report = await _project(tmp_path)
    store.update_task(exp, status="NEEDS_USER")
    out = await _done(ctx, report)
    assert refused_ids(out, "waiting_on_dependencies") == [report]
    assert f"{exp} (NEEDS_USER)" in refused_reasons(out, "waiting_on_dependencies")
    assert store.get_task(report)["status"] == "PENDING"


@pytest.mark.parametrize("dependent_first", [True, False])
async def test_closing_the_dependency_in_the_same_call_is_allowed(tmp_path, dependent_first):
    ctx, store, pid, exp, report = await _project(tmp_path)
    out = await (_done(ctx, report, exp) if dependent_first else _done(ctx, exp, report))
    assert not refused_ids(out, "waiting_on_dependencies")
    assert {store.get_task(t)["status"] for t in (exp, report)} == {"DONE"}


async def test_a_dependency_cycle_never_holds_a_close(tmp_path):
    ctx, store, pid, exp, report = await _project(tmp_path)
    store.update_task(exp, depends_on=[report])          # exp ⇄ report
    out = await _done(ctx, report)
    assert not refused_ids(out, "waiting_on_dependencies")
    assert store.get_task(report)["status"] == "DONE"


def test_the_held_close_is_a_rejection_not_a_success():
    # §4LP: a payload that reports its own refusal with nothing updated is REJECTED
    from ghost_agent.tools.projects import _ok
    out = _ok({"updated": [], "count": 0, "refused": [{"task_id": "abc", "gate": "dependencies"}]})
    assert out.status.value == "rejected" if hasattr(out.status, "value") else out.status == "rejected"
    assert _ok({"updated": [], "count": 0, "refused": [{"task_id": "abc", "gate": "deliverable"}]}).reason_code == \
        "task_update_held_by_constraint"


def test_the_briefing_marks_waiting_and_held_tasks_and_counts_leaves(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Recursion", goal="g")
    exp = store.add_task(pid, "Task 3.1: Run Layer 0")
    store.update_task(exp, status="NEEDS_USER",
                      result_summary="model experiment? — in your turn, probe_model — not closed unattended",
                      wait_on="owner_turn")
    ask = store.add_task(pid, "Pick the human subjects")
    store.update_task(ask, status="NEEDS_USER", result_summary="which people?")
    parent = store.add_task(pid, "Phase 5: analysis")
    store.add_task(pid, "Task 5.1: compare layers", parent_id=parent, depends_on=[exp])
    b = build_project_briefing(store, pid)
    # §4SU Phase 3: the five-state view
    assert "OPEN TASKS (3: " in b and "1 WAITING(owner_turn)" in b and "1 WAITING(owner)" in b and "1 PENDING" in b
    assert "(WAITING(owner_turn))" in b and "not waiting for the user's input" in b
    assert "(PENDING · waits on 1 unfinished task(s))" in b
    assert f"[{parent}]" not in b.split("OPEN TASKS")[1].split("STUCK")[0]


async def test_a_chain_behind_a_blocked_task_is_not_project_done(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Recursion", goal="write a survey")
    a = store.add_task(pid, "Collect sources")
    store.add_task(pid, "Write the survey", depends_on=[a])
    store.update_task(a, status="BLOCKED", failure_reason="paywalled")

    async def runner(*_a, **_k):
        return "OK"

    batch = await pa.advance_many(_ctx(store, tmp_path), pid, max_tasks=None, tool_runner=runner,
                                  owner_requested=True)
    assert batch.stop_reason == "open_but_waiting"
    from ghost_agent.tools.projects import _advance_batch_instruction
    assert "Do NOT report the project as complete" in _advance_batch_instruction(batch, None)


async def test_a_refused_dependency_in_the_same_call_still_holds_its_dependent(tmp_path):
    """r20 M1: `report` waits on `exp`; `exp` promises a file that does not
    exist, so its own close is refused — the report closed DONE anyway."""
    ctx, store, pid, exp, report = await _project(tmp_path)
    store.update_task(exp, description="Run the control condition → results/control.md")
    out = await _done(ctx, report, exp)
    assert refused_ids(out, "undelivered") == [exp] and refused_ids(out, "waiting_on_dependencies") == [report]
    assert store.get_task(report)["status"] == "PENDING"


def test_dependencies_first_orders_a_chain_and_keeps_a_cycle():
    from ghost_agent.tools.projects import _dependencies_first
    n = lambda deps: SimpleNamespace(depends_on=deps)  # noqa: E731
    plan = SimpleNamespace(tree=SimpleNamespace(nodes={
        "c": n(["b"]), "b": n(["a"]), "a": n([]), "x": n(["y"]), "y": n(["x"])}))
    assert _dependencies_first(plan, ["c", "b", "a"]) == ["a", "b", "c"]
    assert sorted(_dependencies_first(plan, ["x", "y", "z"])) == ["x", "y", "z"]


def test_an_open_parent_whose_children_are_closed_is_listed(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    pid = store.create_project("P", goal="g")
    parent = store.add_task(pid, "Phase 1")
    child = store.add_task(pid, "Step 1", parent_id=parent)
    store.update_task(child, status="DONE", result_summary="ok")
    b = build_project_briefing(store, pid)
    assert f"[{parent}] Phase 1" in b


# ── review r21: a parent's dependency holds its children; structure is not order ──

def _phase(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    pid = store.create_project("P", goal="g")
    exp = store.add_task(pid, "Run the experiment")
    report = store.add_task(pid, "Write the report", depends_on=[exp])
    s1 = store.add_task(pid, "Draft section 1", parent_id=report)
    s2 = store.add_task(pid, "Draft section 2", parent_id=report)
    return store, pid, exp, report, s1, s2


def test_a_cycle_through_a_phase_still_drains(tmp_path):
    """r23 M3: A ⇄ Bc, and A waits on phase D whose step d1 waits on A. The
    removed rule waived every dependency of a task in a cycle and this
    drained; following depends_on alone deadlocked it. D's step waits on A,
    so D waits on A: a cycle, not held."""
    from ghost_agent.core.planning import ProjectPlan
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    pid = store.create_project("P", goal="g")
    a = store.add_task(pid, "A")
    bc = store.add_task(pid, "Bc", depends_on=[a])
    d = store.add_task(pid, "D")
    d1 = store.add_task(pid, "d1", parent_id=d, depends_on=[a])
    store.update_task(a, depends_on=[bc, d])
    for _ in range(4):
        plan = ProjectPlan(store, pid)
        nxt = plan.next_ready_leaf()
        if nxt is None:
            break
        plan.update_status(nxt.id, TaskStatus.DONE, result="ok")
    assert {store.get_task(t)["status"] for t in (a, bc, d1, d)} == {"DONE"}


def test_a_plan_with_cross_phase_edges_never_deadlocks(tmp_path):
    """r22 M2: inheriting a phase's edges made P1.report <-> P2.t1 a hidden
    cycle and nothing could run; edges are not inherited now."""
    from ghost_agent.core.planning import ProjectPlan
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    pid = store.create_project("P", goal="g")
    p1 = store.add_task(pid, "Phase 1")
    p2 = store.add_task(pid, "Phase 2", depends_on=[p1])
    t1 = store.add_task(pid, "Collect data", parent_id=p2)
    store.add_task(pid, "Report", parent_id=p1, depends_on=[t1])
    assert ProjectPlan(store, pid).next_ready_leaf().id == t1


def test_the_picker_and_the_gate_agree_on_a_task_in_a_cycle(tmp_path):
    """r22 M1: a task in a cycle had ALL its open dependencies waived by the
    picker but only the cyclic one by the gate — picked, then refused."""
    from ghost_agent.core.planning import ProjectPlan
    from ghost_agent.tools.projects import _unmet_dependencies
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    pid = store.create_project("P", goal="g")
    y = store.add_task(pid, "y")
    a = store.add_task(pid, "a")
    store.add_task(pid, "b", depends_on=[a])
    b = [t["id"] for t in store.list_tasks(pid) if t["description"] == "b"][0]
    store.update_task(a, depends_on=[b, y])
    plan = ProjectPlan(store, pid)
    assert _unmet_dependencies(plan, a) == [y]
    nxt = plan.next_ready_leaf(skip=lambda n: n.id == y)
    assert nxt is None or nxt.id != a
    store.update_task(y, status="DONE", result_summary="ok")
    plan = ProjectPlan(store, pid)
    assert _unmet_dependencies(plan, a) == [] and plan.next_ready_leaf().id == a


@pytest.mark.parametrize("shape", ["phase_on_step", "step_on_phase"])
def test_a_phase_waiting_on_its_own_step_and_a_step_on_its_phase_are_not_held(tmp_path, shape):
    """Separate projects: both edges together form a cycle, which is exempt
    anyway and hid each rule."""
    from ghost_agent.core.planning import ProjectPlan
    from ghost_agent.tools.projects import _unmet_dependencies
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    pid = store.create_project("P", goal="g")
    phase = store.add_task(pid, "Phase 1")
    step = store.add_task(pid, "Step 1", parent_id=phase)
    if shape == "phase_on_step":
        store.update_task(phase, depends_on=[step])
    else:
        store.update_task(step, depends_on=[phase])
    plan = ProjectPlan(store, pid)
    assert plan.next_ready_leaf().id == step
    assert _unmet_dependencies(plan, step) == []
    plan.update_status(step, TaskStatus.DONE, result="ok")       # the phase still rolls up
    assert ProjectPlan(store, pid).tree.nodes[phase].status == TaskStatus.DONE


async def test_a_repeated_id_keeps_its_old_meaning(tmp_path):
    from ghost_agent.tools.projects import _dependencies_first
    plan = SimpleNamespace(tree=SimpleNamespace(nodes={"t": SimpleNamespace(depends_on=[])}))
    assert _dependencies_first(plan, ["t", "t"]) == ["t", "t"]


# ── review r24: a parent's own depends_on is not a "waits on" edge ─────────

def test_a_phase_edge_never_makes_a_false_cycle_that_runs_the_report_early(tmp_path):
    """P2 depends on P1 (phase order); the report depends on P2; P1's
    wrap-up step depends on the report. Following P2's own depends_on found
    a "cycle" P2→P1→a2→report and waived the report's edge to P2."""
    from ghost_agent.core.planning import ProjectPlan
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    pid = store.create_project("P", goal="g")
    p1 = store.add_task(pid, "Phase 1")
    a1 = store.add_task(pid, "Run experiment", parent_id=p1)
    p2 = store.add_task(pid, "Phase 2", depends_on=[p1])
    b1 = store.add_task(pid, "Analyse results", parent_id=p2)
    rep = store.add_task(pid, "Write report", depends_on=[p2])
    a2 = store.add_task(pid, "Wrap-up citing the report", parent_id=p1, depends_on=[rep])
    order = []
    for _ in range(6):
        plan = ProjectPlan(store, pid)
        nxt = plan.next_ready_leaf()
        if nxt is None:
            break
        order.append(nxt.id)
        plan.update_status(nxt.id, TaskStatus.DONE, result="ok")
    assert order.index(rep) > order.index(b1)
    assert order[:2] == [a1, b1] and a2 in order


async def test_a_phase_closed_by_hand_is_not_held_by_its_own_step(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    ctx = _ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="P")
    pid = ctx.current_project_id
    phase = store.add_task(pid, "Phase 1")
    step = store.add_task(pid, "Step 1", parent_id=phase)
    store.update_task(phase, depends_on=[step])
    from ghost_agent.core.planning import ProjectPlan
    from ghost_agent.tools.projects import _unmet_dependencies
    assert _unmet_dependencies(ProjectPlan(store, pid), phase) == []


def test_a_step_of_an_ANY_phase_depending_on_that_phase_is_not_held(tmp_path):
    from ghost_agent.core.planning import ProjectPlan
    from ghost_agent.tools.projects import _unmet_dependencies
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    pid = store.create_project("P", goal="g")
    phase = store.add_task(pid, "Pick one approach")
    store.update_task(phase, dependency_type="ANY")
    step = store.add_task(pid, "Approach A", parent_id=phase)
    store.update_task(step, depends_on=[phase])
    plan = ProjectPlan(store, pid)
    assert _unmet_dependencies(plan, step) == [] and plan.next_ready_leaf().id == step


async def test_a_bulk_close_closes_a_dependency_phases_steps_first(tmp_path):
    """r24 m1: [rep, c1] where rep depends on phase PH and c1 is PH's only
    step — rep was refused although PH rolled up in the same call."""
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    ctx = _ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="P")
    pid = ctx.current_project_id
    ph = store.add_task(pid, "Phase")
    c1 = store.add_task(pid, "Step", parent_id=ph)
    rep = store.add_task(pid, "Report", depends_on=[ph])
    out = await _done(ctx, rep, c1)
    assert not refused_ids(out, "waiting_on_dependencies")
    assert {store.get_task(t)["status"] for t in (ph, c1, rep)} == {"DONE"}
