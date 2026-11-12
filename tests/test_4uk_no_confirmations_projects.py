"""§4UK (operator 2026-11-12: "i want all confirmations, everywhere to be removed. i don't wanna be asked
for anything. that's why everything is sandboxed" — "keep clarifying questions"): the project system never
parks a task for the owner's approval. Gate words run, HUMAN_GATE postconditions close (and are never graded
as text), a gated task never holds a delete / replan, and the agent's constraint exception in the owner's own
turn goes through on the record. Unattended turns still cannot go over a constraint."""
import json

import pytest

from ghost_agent.core import project_advancer as pa
from ghost_agent.core import project_safety as ps
from ghost_agent.core.planning import TaskStatus, TaskTree
from ghost_agent.tools.projects import tool_manage_projects
from tests.test_4su_phase0 import _project, _done, owner_turn  # noqa: F401
from tests.test_4ui_project_gaps import _constrained

GATE = ["HUMAN_GATE: owner approves the deploy"]


# ── no word gate ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("desc", ["Approve the final migration plan", "Publish the release notes",
                                  "Send the newsletter", "Confirm the schema with the team",
                                  "Decide between SQLite and Postgres"])
def test_gate_words_are_never_needs_user(desc):
    assert pa.classify_task(desc) != "needs_user"
    assert pa.classify_task(desc, default="coding") != "needs_user"
    assert pa.settle_gate(desc, "needs_user") != "needs_user"


def test_the_gate_word_machinery_is_gone():
    for name in ("needs_user_gate", "gate_form", "_NEEDS_USER_RE", "_GATE_FORMS", "_GATE_LOOSE_RE"):
        assert not hasattr(pa, name), name
    assert not hasattr(ps, "enforce_human_gate")
    assert "needs_user" not in pa.CLASSIFY_PROMPT
    import ghost_agent.tools.projects as P
    for name in ("_APPROVE_WORD_RE", "_DELETE_WORD_RE", "_HOLD_BACK_RE", "_owner_released",
                 "_gate_released_for_delete", "_unreleased_delete_gates", "_human_gate_approved",
                 "_GATE_RELEASE_HOW", "_park_constraint_override", "_constraint_override_approved"):
        assert not hasattr(P, name), name
    # the batch path (a gate-word task running with the rest) is pinned in
    # tests/test_4sr_gate_words.py::test_a_gate_word_task_runs_with_the_rest


# ── HUMAN_GATE postconditions are notes ──────────────────────────────────────

def test_is_human_gate_text():
    assert ps.is_human_gate_text("HUMAN_GATE: cto approves")
    assert ps.is_human_gate_text("  human_gate: x")
    assert not ps.is_human_gate_text("tests pass")
    assert not ps.is_human_gate_text(None)


def test_a_gated_task_closes_done_and_the_gate_is_never_graded():
    tree = TaskTree()
    tid = tree.add_task("Deploy the site")
    tree.nodes[tid].postconditions = list(GATE)
    tree.update_status(tid, TaskStatus.DONE, result="deployed")    # the result never echoes the gate
    assert tree.nodes[tid].status == TaskStatus.DONE


@pytest.mark.asyncio
async def test_a_gated_task_closes_done_through_the_tool(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "DONE"
    assert not store.list_events(pid, event_type="human_gate_parked")


@pytest.mark.asyncio
async def test_open_gated_tasks_hold_no_delete_or_replan(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, t1 = await _project(tmp_path, postconditions=GATE)
    t2 = store.add_task(pid, "Deploy the API", postconditions=GATE)
    out = json.loads(str(await tool_manage_projects(ctx, action="task_delete", task_id=t1)))
    assert out.get("deleted") == t1 and store.get_task(t1) is None
    out = str(await tool_manage_projects(ctx, action="replan", project_id=pid,
                                         subtasks=["Write the docs → docs/index.md"]))
    assert "NOT replanned" not in out and store.get_task(t2) is None
    t3 = store.add_task(pid, "Ship it", postconditions=GATE)
    out = str(await tool_manage_projects(ctx, action="update", project_id=pid, status="DONE"))
    assert "NOT updated" not in out, out
    assert store.get_task(t3) is not None


# ── a constraint exception: recorded in the owner's turn, refused unattended ─

@pytest.mark.asyncio
async def test_the_owner_turn_override_goes_through_on_the_record(tmp_path, owner_turn, monkeypatch):
    ctx, store, pid, tid, _ = await _constrained(tmp_path, owner_turn, monkeypatch)
    out = await _done(ctx, tid, result="wrote it",
                      override={"constraints": "lark is the only parser that fits the grammar"})
    assert store.get_task(tid)["status"] == "DONE", out
    ev = store.list_events(pid, event_type="constraint_override")
    assert ev and "lark" in ev[0]["payload"]["reason"]
    assert ev[0]["payload"]["violation"] == "uses an external library"
    assert any(o["gate"] == "constraints" for o in out.get("overrides") or [])
    assert not store.list_events(pid, event_type="human_gate_parked")


@pytest.mark.asyncio
async def test_an_unattended_override_is_still_refused(tmp_path, owner_turn, monkeypatch):
    ctx, store, pid, tid, _ = await _constrained(tmp_path, owner_turn, monkeypatch)
    owner_turn("sched-1234")
    out = await _done(ctx, tid, result="wrote it",
                      override={"constraints": "lark is the only parser that fits the grammar"})
    assert store.get_task(tid)["status"] not in ("DONE", "NEEDS_USER")
    assert any(r["gate"] == "constraints" for r in out["refused"])
    assert not store.list_events(pid, event_type="constraint_override")


@pytest.mark.asyncio
async def test_an_owner_turn_violation_without_an_override_is_refused(tmp_path, owner_turn, monkeypatch):
    ctx, store, pid, tid, _ = await _constrained(tmp_path, owner_turn, monkeypatch)
    out = await _done(ctx, tid, result="wrote it")
    assert store.get_task(tid)["status"] != "DONE"
    assert any(r["gate"] == "constraints" and "no approval needed" in r["fix"] for r in out["refused"])
