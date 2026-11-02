"""§4SU Phase 0 — the confirmed project-system bugs from the 2026-11-02 audit.

Each test names the audit finding it pins (PROJECT_JOURNAL §4SU)."""

from tests._refused import refused_ids, refused_reasons  # §4SU Phase 4
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.planning import TaskStatus
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.projects import tool_manage_projects
from ghost_agent.utils.logging import request_id_context
from tests.test_4qa_project_task_integrity import _ctx

GATE = ["HUMAN_GATE: owner approves the deploy"]


@pytest.fixture
def owner_turn():
    """Run a block as the owner's request ``rid`` (a web/CLI request id)."""
    def _set(rid):
        request_id_context.set(rid)
    yield _set
    request_id_context.set("SYSTEM")            # async tests run in their own context


async def _project(tmp_path, desc="Deploy the site", **task_kw):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="Site", goal="Ship the site")
    pid = ctx.current_project_id
    tid = store.add_task(pid, desc, **task_kw)
    return ctx, store, pid, tid


def _say(ctx, text):
    """The owner's message of the current request (provenance records it; r5)."""
    from ghost_agent.utils.provenance import note_user_message
    ctx.last_user_content = text
    note_user_message(text)


async def _done(ctx, tid, **kw):
    return json.loads(str(await tool_manage_projects(ctx, action="task_update", task_id=tid,
                                                     status="done", **kw)))


# ─────────────── A: a human gate is approved only explicitly ───────────────

async def test_a_second_close_in_the_same_turn_is_not_the_approval(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    out = await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"
    assert out["not_closed"][0]["task_id"] == tid
    assert "OWNER approves" in out["agent_instruction_not_closed"]
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"


async def test_the_owner_s_next_request_approves(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    await _done(ctx, tid, result="deployed")
    owner_turn("e5f6a7b8")                       # the owner answered: a NEW request
    _say(ctx, f"approve {tid}")
    out = await _done(ctx, tid, result="approved by the owner")
    assert store.get_task(tid)["status"] == "DONE" and "not_closed" not in out


async def test_a_background_turn_never_approves(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    await _done(ctx, tid, result="deployed")
    for rid in ("SYSTEM", "sched-1234", "probe-ab12"):
        owner_turn(rid)
        await _done(ctx, tid, result="deployed")
        assert store.get_task(tid)["status"] == "NEEDS_USER", rid


async def test_an_advancer_park_is_approved_by_the_owner_s_turn(tmp_path, owner_turn):
    owner_turn("SYSTEM")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    store.update_task(tid, status="NEEDS_USER", result_summary="human gate: owner approves the deploy")
    store.log_event(pid, tid, "human_gate_triggered", {"reason": "owner approves the deploy"})
    owner_turn("a1b2c3d4")
    _say(ctx, f"approved: {tid}")
    await _done(ctx, tid, result="the owner said go")
    assert store.get_task(tid)["status"] == "DONE"


async def test_no_park_record_is_no_approval(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    store.update_task(tid, status="NEEDS_USER")          # parked by nobody we know
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"


async def test_the_park_records_its_request(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    await _done(ctx, tid, result="deployed")
    ev = store.list_events(pid, event_type="human_gate_parked")
    assert ev and ev[0]["payload"]["req"] == "a1b2c3d4"


# ─────────── B: a close the plan refuses is reported, writes nothing ───────────

async def test_a_postcondition_demotion_is_reported_and_writes_nothing(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, desc="Write the summary",
                                          postconditions=["tests pass with exit code zero"])
    ws = Path(store.ensure_workspace(pid))
    (ws / "notes.txt").write_text("notes\n")
    out = await _done(ctx, tid, result="wrote it", deliverables=["notes.txt"],
                      ledger="FACT: summary complete")
    # §4SU Phase 1: the close is REFUSED by the one check (the task is left
    # as it was), no longer demoted to FAILED after the fact
    assert store.get_task(tid)["status"] == "PENDING"
    assert out["refused"][0]["gate"] == "postconditions" and out["refused"][0]["task_id"] == tid
    assert "tests pass" in out["not_closed"][0]["why"]
    assert "notes.txt" not in [str(d) for d in store.list_deliverables(pid)]
    assert "summary complete" not in str(store.get_ledger(pid))


async def test_a_real_close_still_registers_and_records(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, desc="Write the summary")
    ws = Path(store.ensure_workspace(pid))
    (ws / "notes.txt").write_text("notes\n")
    out = await _done(ctx, tid, result="wrote it", deliverables=["notes.txt"], ledger="FACT: summary complete")
    assert store.get_task(tid)["status"] == "DONE" and "not_closed" not in out
    assert "summary complete" in str(store.get_ledger(pid))


async def test_an_earlier_deliverable_survives_a_refused_close(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, desc="Write the summary",
                                          postconditions=["tests pass with exit code zero"])
    ws = Path(store.ensure_workspace(pid))
    (ws / "notes.txt").write_text("notes\n")
    store.register_file_artifact(tid, "notes.txt")
    await _done(ctx, tid, result="wrote it", deliverables=["notes.txt"])
    assert "notes.txt" in [str(d) for d in store.list_deliverables(pid)]


# ─────────────── 2: system-written text is never evidence ───────────────

@pytest.mark.parametrize("held", [
    "built the canvas — needs a real check — not closed unattended",
    "human gate: owner approves",
    "flagged for human review",
    "model experiment? — its data may be a model's answers",
])
async def test_a_system_hold_is_not_the_visual_evidence(tmp_path, owner_turn, held):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, desc="Build the snake game canvas in index.html")
    store.update_task(tid, status="NEEDS_USER", result_summary=held)
    out = await _done(ctx, tid)
    assert store.get_task(tid)["status"] != "DONE"
    assert tid in refused_ids(out, "gated_unverified")


async def test_the_owner_s_own_stored_result_is_still_evidence(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, desc="Build the snake game canvas in index.html")
    store.update_task(tid, result_summary="screenshot: the snake moves, RENDER_CHECK ok")
    await _done(ctx, tid)
    assert store.get_task(tid)["status"] == "DONE"


async def test_a_system_hold_is_not_the_constraint_evidence(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, desc="Write the parser",
                                          constraints=["no external libraries"])
    store.update_task(tid, status="NEEDS_USER", result_summary="parser built — not closed unattended",
                      wait_on="owner_turn")
    out = await _done(ctx, tid)
    assert store.get_task(tid)["status"] != "DONE"
    assert tid in refused_ids(out, "gated_constraints")


# ─────────────── D: a paused project stays paused ───────────────

def test_a_released_claim_does_not_unpause_the_project(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Build it")
    store.update_task(tid, status="IN_PROGRESS")
    store.update_project(pid, status="PAUSED")           # the owner pauses mid-step
    store.update_task(tid, status="READY")              # the step's claim is released
    assert store.get_project(pid)["status"] == "PAUSED"


def test_the_owner_s_redo_still_reopens_a_paused_project(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Build it")
    store.update_task(tid, status="DONE")
    store.update_project(pid, status="PAUSED")
    store.update_task(tid, status="PENDING")             # "redo task X"
    assert store.get_project(pid)["status"] == "ACTIVE"


# ─────────────── C: a revived child unblocks its parent ───────────────

def test_a_revived_child_unblocks_the_parent_and_is_handed_out(tmp_path):
    from ghost_agent.core.planning import ProjectPlan
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    top = store.add_task(pid, "Phase 1")
    a = store.add_task(pid, "Step A", parent_id=top)
    b = store.add_task(pid, "Step B", parent_id=top)
    plan = ProjectPlan(store, pid)
    plan.update_status(a, TaskStatus.FAILED, failure_reason="broke")
    assert store.get_task(top)["status"] == "BLOCKED"
    plan = ProjectPlan(store, pid)
    plan.update_status(a, TaskStatus.PENDING)
    assert store.get_task(top)["status"] == "PENDING"
    leaf = ProjectPlan(store, pid).next_ready_leaf()
    assert leaf is not None and leaf.id in (a, b)


def test_a_parent_still_blocked_by_another_failed_child_stays_blocked(tmp_path):
    from ghost_agent.core.planning import ProjectPlan
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    top = store.add_task(pid, "Phase 1")
    a = store.add_task(pid, "Step A", parent_id=top)
    b = store.add_task(pid, "Step B", parent_id=top)
    plan = ProjectPlan(store, pid)
    plan.update_status(a, TaskStatus.FAILED)
    plan.update_status(b, TaskStatus.FAILED)
    ProjectPlan(store, pid).update_status(a, TaskStatus.PENDING)
    assert store.get_task(top)["status"] == "BLOCKED"


def test_unblocking_climbs_the_tree(tmp_path):
    from ghost_agent.core.planning import ProjectPlan
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    root = store.add_task(pid, "Release")
    mid = store.add_task(pid, "Phase 1", parent_id=root)
    leaf = store.add_task(pid, "Step A", parent_id=mid)
    ProjectPlan(store, pid).update_status(leaf, TaskStatus.FAILED)
    assert store.get_task(mid)["status"] == "BLOCKED" and store.get_task(root)["status"] == "BLOCKED"
    ProjectPlan(store, pid).update_status(leaf, TaskStatus.READY)
    assert store.get_task(mid)["status"] == "PENDING" and store.get_task(root)["status"] == "PENDING"


# ─────────── a failed dependency pauses autopilot and tells the owner ───────────

from tests.test_4mq_long_horizon import ctx as _mq_ctx, notes, _meta  # noqa: E402,F401  (fixtures)


@pytest.fixture
def ap_store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def ap_ctx(ap_store):
    from types import SimpleNamespace
    return SimpleNamespace(project_store=ap_store, workspace_model=None, current_project_id=None)


async def test_a_failed_dependency_pauses_autopilot_once_and_says_why(ap_store, ap_ctx, notes):
    from ghost_agent.core.project_advancer import advance_unattended
    pid = ap_store.create_project("Long job", metadata={"autopilot": True})
    ap_store.update_project(pid, status="ACTIVE")
    a = ap_store.add_task(pid, "Fetch the data")
    ap_store.add_task(pid, "Chart the data", depends_on=[a])
    ap_store.update_task(a, status="FAILED", failure_reason="source down")
    ap_store.update_project(pid, status="ACTIVE")
    for _ in range(3):
        await advance_unattended(ap_ctx, pid)
    m = _meta(ap_store, pid)
    assert m["autopilot"] is False and m["autopilot_paused"]["reason"] == "stuck"
    assert a in m["autopilot_paused"]["detail"]
    assert len(notes) == 1


async def test_an_empty_or_finished_project_is_not_stuck(ap_store, ap_ctx, notes):
    from ghost_agent.core.project_advancer import advance_unattended
    pid = ap_store.create_project("Long job", metadata={"autopilot": True})
    ap_store.update_project(pid, status="ACTIVE")
    a = ap_store.add_task(pid, "Fetch the data")
    ap_store.update_task(a, status="FAILED")             # failed, but nothing waits on it
    ap_store.update_project(pid, status="ACTIVE")
    await advance_unattended(ap_ctx, pid)
    assert (_meta(ap_store, pid).get("autopilot_paused") or {}).get("reason") != "stuck"


# ─────────── the HTTP advance never builds beside a live unattended step ───────────

async def test_the_http_advance_refuses_during_a_live_unattended_step(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace
    from fastapi import HTTPException
    from ghost_agent.api import projects_routes as R
    from ghost_agent.core import project_advancer as PA
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    called = []

    async def _adv(*a, **k):
        called.append(1)
        return PA.AdvanceResult(True, None, "idle", "nothing")
    monkeypatch.setattr(R, "advance_once", _adv)
    monkeypatch.setattr(R, "_mark_foreground", lambda *a, **k: None)
    monkeypatch.setattr(R, "_context", lambda r: _ctx(store, tmp_path))
    monkeypatch.setattr(R, "_store", lambda r: store)
    monkeypatch.setattr(R, "get_agent", lambda r: SimpleNamespace())
    from unittest.mock import AsyncMock
    from ghost_agent.tools import registry as RG
    monkeypatch.setattr(RG, "get_available_tools", lambda c: {"web_search": AsyncMock(return_value="")})
    store.update_project(pid, metadata={"step_in_flight": {"boot": PA._BOOT_ID, "ts": time.time(), "id": "x"}})
    with pytest.raises(HTTPException) as ei:
        await R.advance_project(pid, SimpleNamespace(headers={}))
    assert ei.value.status_code == 409 and called == []
    store.update_project(pid, metadata={"step_in_flight": None})
    await R.advance_project(pid, SimpleNamespace(headers={}))
    assert called == [1]


async def test_the_web_console_click_approves_a_gated_task(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ghost_agent.api import projects_routes as R
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Deploy", postconditions=GATE)
    monkeypatch.setattr(R, "_store", lambda r: store)
    await R.update_task(pid, tid, R.TaskUpdate(status="DONE"), SimpleNamespace(headers={}))
    assert store.get_task(tid)["status"] == "DONE"


# ─────────── the DONE sweep moves files to the trash and records it ───────────

def test_the_sweep_moves_to_the_host_trash_and_logs_an_event(tmp_path):
    from ghost_agent.core import workspace_cleanup as WC
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Build the site")
    ws = Path(store.ensure_workspace(pid))
    (ws / "index.html").write_text("<script src='app.js'></script>")
    (ws / "dist").mkdir()
    (ws / "dist/site.zip").write_bytes(b"PK\x03\x04zip")
    store.register_file_artifact(tid, "index.html")
    out = WC.sweep_project_workspace(store, pid)
    assert "dist/site.zip" in out["deleted"] and not (ws / "dist/site.zip").exists()
    moved = list((tmp_path / "project_trash").rglob("site.zip"))
    assert len(moved) == 1 and moved[0].read_bytes() == b"PK\x03\x04zip"
    assert str(tmp_path / "s") not in str(moved[0])                # never inside the sandbox
    ev = store.list_events(pid, event_type="workspace_sweep")
    assert ev and "dist/site.zip" in ev[0]["payload"]["moved"]


def test_old_trash_days_are_purged(tmp_path):
    import time
    from ghost_agent.core import workspace_cleanup as WC
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    trash = tmp_path / "project_trash"
    old = time.strftime("%Y%m%d", time.localtime(time.time() - 40 * 86400))
    new = time.strftime("%Y%m%d")
    (trash / old / "p").mkdir(parents=True)
    (trash / new / "p").mkdir(parents=True)
    assert WC.purge_project_trash(store) == 1
    assert not (trash / old).exists() and (trash / new).exists()


def test_a_log_the_project_names_is_kept(tmp_path):
    from ghost_agent.core import workspace_cleanup as WC
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Train the model")
    store.update_task(tid, status="DONE", result_summary="trained; wrote results/train.log")
    ws = Path(store.ensure_workspace(pid))
    (ws / "results").mkdir()
    (ws / "results/train.log").write_text("epoch 1\n")
    (ws / "results/debug.log").write_text("noise\n")
    (ws / "train.py").write_text("print(1)\n")
    store.register_file_artifact(tid, "train.py")
    WC.sweep_project_workspace(store, pid)
    assert (ws / "results/train.log").exists() and not (ws / "results/debug.log").exists()



# ─────────────── review r1 ───────────────

@pytest.mark.parametrize("said", ["status?", "keep going with the other tasks", "no, wait", "yes but not yet", ""])
async def test_a_later_owner_message_that_does_not_say_yes_is_no_approval(tmp_path, owner_turn, said):
    # r1 M4: any later owner turn approved the deploy gate
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    await _done(ctx, tid, result="deployed")
    owner_turn("e5f6a7b8")
    _say(ctx, said)
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER", said


async def test_a_rewrite_cannot_drop_the_gate(tmp_path, owner_turn):
    # r1 C1: postconditions=[] then done closed the gate in one turn
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    await tool_manage_projects(ctx, action="task_update", task_id=tid, postconditions=[])
    assert GATE[0] in store.get_task(tid)["postconditions"]
    await tool_manage_projects(ctx, action="task_update", task_id=tid, postconditions=["deployed"])
    assert GATE[0] in store.get_task(tid)["postconditions"] and "deployed" in store.get_task(tid)["postconditions"]
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"


async def test_deleting_a_gated_task_needs_the_owner_s_yes(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    _say(ctx, "clean up the plan")
    out = str(await tool_manage_projects(ctx, action="task_delete", task_id=tid))
    assert "NOT deleted" in out and store.get_task(tid)
    _say(ctx, f"delete {tid}")
    out = str(await tool_manage_projects(ctx, action="task_delete", task_id=tid))
    # r2 C2: a yes in the SAME request that asked is not the release
    assert "NOT deleted" in out and store.get_task(tid)
    owner_turn("e5f6a7b8")
    _say(ctx, f"approve {tid}")          # r4: an approval is not a delete
    out = str(await tool_manage_projects(ctx, action="task_delete", task_id=tid))
    assert "NOT deleted" in out and store.get_task(tid)
    owner_turn("c9d0e1f2")
    _say(ctx, f"delete {tid}")
    await tool_manage_projects(ctx, action="task_delete", task_id=tid)
    assert not store.get_task(tid)


async def test_deleting_the_parent_of_a_gated_task_needs_the_owner_s_yes(tmp_path, owner_turn):
    # r2 C1: delete cascades to the subtree — the child's gate went with it
    owner_turn("a1b2c3d4")
    ctx, store, pid, _ = await _project(tmp_path)
    top = store.add_task(pid, "Release")
    gated = store.add_task(pid, "Deploy", parent_id=top, postconditions=GATE)
    _say(ctx, f"delete {gated}")
    out = str(await tool_manage_projects(ctx, action="task_delete", task_id=top))
    assert "NOT deleted" in out and gated in out and store.get_task(gated)
    owner_turn("e5f6a7b8")
    _say(ctx, f"delete {gated}")
    await tool_manage_projects(ctx, action="task_delete", task_id=top)
    assert not store.get_task(top) and not store.get_task(gated)


@pytest.mark.parametrize("said", ["8080", "3", "1, 2", "all"])
async def test_a_number_or_all_is_not_the_approval(tmp_path, owner_turn, said):
    # r2 M4: user_confirms reads these as an item pick
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    await _done(ctx, tid, result="deployed")
    owner_turn("e5f6a7b8")
    _say(ctx, said)
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"


@pytest.mark.parametrize("said", ["lgtm {tid}", "εγκρίνω το {tid}", "approved: {tid}", "APPROVE {TID}", "approve {short}", "approve “{tid}”"])
async def test_approval_words_approve(tmp_path, owner_turn, said):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    await _done(ctx, tid, result="deployed")
    owner_turn("e5f6a7b8")
    _say(ctx, said.format(tid=tid, TID=tid.upper(), short=tid[:6]))
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "DONE"


async def test_a_queued_wait_withdraws_the_yes(tmp_path, owner_turn, monkeypatch):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    await _done(ctx, tid, result="deployed")
    owner_turn("e5f6a7b8")
    _say(ctx, f"approve {tid}")
    import ghost_agent.utils.provenance as prov
    monkeypatch.setattr(prov, "added_user_messages", lambda: ["wait, don't deploy yet"])
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"
    owner_turn("c9d0e1f2")              # the withheld close re-put the gate in e5f6a7b8
    _say(ctx, f"approve {tid}")
    monkeypatch.setattr(prov, "added_user_messages", lambda: ["yes"])
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "DONE"


async def test_a_roll_up_park_is_recorded_and_reported(tmp_path, owner_turn):
    # r2 M1: the roll-up park wrote no event, so the owner's yes found nothing
    owner_turn("a1b2c3d4")
    ctx, store, pid, _ = await _project(tmp_path)
    top = store.add_task(pid, "Release", postconditions=GATE)
    kid = store.add_task(pid, "Build", parent_id=top)
    out = await _done(ctx, kid, result="built dist/app.zip")
    assert store.get_task(top)["status"] == "NEEDS_USER"
    assert any(n["task_id"] == top and n["why"].startswith("human gate:") for n in out["not_closed"])
    ev = store.latest_task_event(pid, top, ("human_gate_parked", "human_gate_triggered"))
    assert ev and ev["payload"]["req"] == "a1b2c3d4"
    assert store.get_task(top)["result_summary"].startswith("human gate:")
    owner_turn("e5f6a7b8")
    _say(ctx, f"approve {top}")
    await _done(ctx, top, result="released")
    assert store.get_task(top)["status"] == "DONE"


def test_a_roll_up_parks_a_gated_parent():
    # r1 M1: the children closed and the gated parent (and the project) went DONE
    from ghost_agent.core.planning import TaskTree
    tree = TaskTree()
    top = tree.add_task("Release", postconditions=GATE)
    a = tree.add_task("Build", parent_id=top)
    tree.update_status(a, TaskStatus.DONE, result="built")
    assert tree.nodes[top].status == TaskStatus.NEEDS_USER
    tree.update_status(top, TaskStatus.DONE, result="released", human_approved=True)
    assert tree.nodes[top].status == TaskStatus.DONE


async def test_a_park_keeps_the_close_s_deliverables(tmp_path, owner_turn):
    # r1 M2: unregistered at the park, the owner's approval close then swept them
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    ws = Path(store.ensure_workspace(pid))
    (ws / "out").mkdir()
    (ws / "out/chart.png").write_bytes(b"png")
    out = await _done(ctx, tid, result="charted", deliverables=["out/chart.png"])
    assert store.get_task(tid)["status"] == "NEEDS_USER"
    assert "out/chart.png" in [str(d) for d in store.list_deliverables(pid)]
    assert out["not_closed"][0]["why"].startswith("human gate: owner approves")
    owner_turn("e5f6a7b8")
    _say(ctx, f"approve {tid}")
    await _done(ctx, tid, result="approved")
    assert (ws / "out/chart.png").exists()


async def test_the_park_is_found_behind_many_events(tmp_path, owner_turn):
    # r1 M3: past the newest 200 events, every owner yes re-parked the task
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    await _done(ctx, tid, result="deployed")
    for i in range(210):
        store.log_event(pid, None, "note", {"i": i})
    owner_turn("e5f6a7b8")
    _say(ctx, f"approve {tid}")
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "DONE"


async def test_an_unrelated_failure_does_not_pause_a_project_waiting_on_the_owner(ap_store, ap_ctx, notes):
    # r1: B waits on the owner's question Q, A failed on its own → not "stuck"
    from ghost_agent.core.project_advancer import advance_unattended
    pid = ap_store.create_project("Long job", metadata={"autopilot": True})
    ap_store.update_project(pid, status="ACTIVE")
    a = ap_store.add_task(pid, "Optional extra chart")
    q = ap_store.add_task(pid, "Which colours?")
    ap_store.add_task(pid, "Draw it", depends_on=[q])
    ap_store.update_task(a, status="FAILED")
    ap_store.update_task(q, status="NEEDS_USER")
    ap_store.update_project(pid, status="ACTIVE")
    await advance_unattended(ap_ctx, pid)
    assert (_meta(ap_store, pid).get("autopilot_paused") or {}).get("reason") != "stuck"


async def test_tasks_under_a_blocked_parent_are_stuck_on_its_failed_child(ap_store, ap_ctx, notes):
    from ghost_agent.core.planning import ProjectPlan
    from ghost_agent.core.project_advancer import advance_unattended
    pid = ap_store.create_project("Long job", metadata={"autopilot": True})
    ap_store.update_project(pid, status="ACTIVE")
    top = ap_store.add_task(pid, "Phase 1")
    a = ap_store.add_task(pid, "Step A", parent_id=top)
    ap_store.add_task(pid, "Step B", parent_id=top)
    ProjectPlan(ap_store, pid).update_status(a, TaskStatus.FAILED)
    ap_store.update_project(pid, status="ACTIVE")
    await advance_unattended(ap_ctx, pid)
    m = _meta(ap_store, pid)
    assert m["autopilot_paused"]["reason"] == "stuck" and a in m["autopilot_paused"]["detail"]


def test_a_parent_the_model_blocked_stays_blocked():
    from ghost_agent.core.planning import TaskTree
    tree = TaskTree()
    top = tree.add_task("Phase 1")
    a = tree.add_task("Step A", parent_id=top)
    tree.nodes[top].status = TaskStatus.BLOCKED
    tree.nodes[top].failure_reason = "waiting on the API key"
    tree.update_status(a, TaskStatus.PENDING)
    assert tree.nodes[top].status == TaskStatus.BLOCKED


def test_the_sweep_event_tells_moved_from_deleted(tmp_path, monkeypatch):
    from ghost_agent.core import workspace_cleanup as WC
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Build")
    ws = Path(store.ensure_workspace(pid))
    (ws / "index.html").write_text("x")
    (ws / "a.zip").write_bytes(b"z")
    (ws / "b.zip").write_bytes(b"z")
    store.register_file_artifact(tid, "index.html")
    real = WC._to_project_trash
    monkeypatch.setattr(WC, "_to_project_trash", lambda s, p, r, f, rel, st: rel != "b.zip" and real(s, p, r, f, rel, st))
    WC.sweep_project_workspace(store, pid)
    ev = store.list_events(pid, event_type="workspace_sweep")[0]["payload"]
    assert ev["moved"] == ["a.zip"] and ev["deleted"] == ["b.zip"]


# ─────────────── r2 battery survivors ───────────────

async def _parked(tmp_path, owner_turn, rid="a1b2c3d4"):
    owner_turn(rid)
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"
    return ctx, store, pid, tid


async def test_a_yes_in_the_request_that_parked_is_not_the_approval(tmp_path, owner_turn):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    _say(ctx, f"approve {tid}")                      # same request a1b2c3d4
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"


async def test_a_yes_outside_an_owner_turn_is_not_the_approval(tmp_path, owner_turn):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("SYSTEM")                                # a scheduled / autopilot turn
    _say(ctx, f"approve {tid}")
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"


async def test_a_task_that_was_never_parked_is_not_approved_by_a_yes(tmp_path, owner_turn):
    # PENDING (not NEEDS_USER), a stale park event from an earlier request
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    store.log_event(pid, tid, "human_gate_parked", {"reason": "x", "req": "00000000"})
    _say(ctx, f"approve {tid}")
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"


async def test_needs_user_with_no_park_record_is_not_approved(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    store.update_task(tid, status="NEEDS_USER")
    owner_turn("e5f6a7b8")
    _say(ctx, f"approve {tid}")
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"


async def test_the_newest_park_decides(tmp_path, owner_turn):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)      # parked in a1b2c3d4
    owner_turn("e5f6a7b8")
    store.log_event(pid, tid, "human_gate_parked", {"reason": "re-asked", "req": "e5f6a7b8"})
    _say(ctx, f"approve {tid}")
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"


async def test_deleting_a_gated_task_outside_an_owner_turn_is_refused(tmp_path, owner_turn):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("SYSTEM")
    _say(ctx, f"approve {tid}")
    out = str(await tool_manage_projects(ctx, action="task_delete", task_id=tid))
    assert "NOT deleted" in out and store.get_task(tid)


def test_an_any_roll_up_parks_a_gated_parent():
    from ghost_agent.core.planning import TaskTree, TaskStatus
    tree = TaskTree()
    top = tree.add_task("Pick a host", postconditions=GATE)
    from ghost_agent.core.planning import DependencyType
    tree.nodes[top].dependency_type = DependencyType.ANY
    a = tree.add_task("Try host A", parent_id=top)
    tree.add_task("Try host B", parent_id=top)
    tree.update_status(a, TaskStatus.DONE, result="A works")
    assert tree.nodes[top].status == TaskStatus.NEEDS_USER


async def test_a_task_running_is_not_stuck(ap_store, ap_ctx, notes):
    from ghost_agent.core.project_advancer import _pause_if_stuck
    pid = ap_store.create_project("Long job", metadata={"autopilot": True})
    a = ap_store.add_task(pid, "Fetch the data")
    ap_store.add_task(pid, "Chart the data", depends_on=[a])
    ap_store.add_task(pid, "Write it up")
    ap_store.update_task(a, status="FAILED")
    b = [t for t in ap_store.list_tasks(pid) if t["description"] == "Write it up"][0]["id"]
    ap_store.update_task(b, status="IN_PROGRESS")
    _pause_if_stuck(ap_ctx, ap_store, pid)
    assert not _meta(ap_store, pid).get("autopilot_paused")


async def test_a_runnable_leaf_is_not_stuck(ap_store, ap_ctx, notes):
    from ghost_agent.core.project_advancer import _pause_if_stuck
    pid = ap_store.create_project("Long job", metadata={"autopilot": True})
    a = ap_store.add_task(pid, "Fetch the data")
    ap_store.add_task(pid, "Chart the data", depends_on=[a])
    ap_store.add_task(pid, "Write it up")                # free to run
    ap_store.update_task(a, status="FAILED")
    _pause_if_stuck(ap_ctx, ap_store, pid)
    assert not _meta(ap_store, pid).get("autopilot_paused")
    # the positive twin: without the free leaf it pauses
    pid2 = ap_store.create_project("Long job 2", metadata={"autopilot": True})
    a2 = ap_store.add_task(pid2, "Fetch")
    ap_store.add_task(pid2, "Chart", depends_on=[a2])
    ap_store.update_task(a2, status="FAILED")
    _pause_if_stuck(ap_ctx, ap_store, pid2)
    assert _meta(ap_store, pid2)["autopilot_paused"]["reason"] == "stuck"


def test_the_sweep_purges_old_trash(tmp_path):
    from ghost_agent.core import workspace_cleanup as WC
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Build")
    old = WC.project_trash_dir(store) / "20200101"
    old.mkdir(parents=True)
    (old / "f").write_text("x")
    ws = Path(store.ensure_workspace(pid))
    (ws / "index.html").write_text("x")
    store.register_file_artifact(tid, "index.html")
    (ws / "a.zip").write_bytes(b"z")
    WC.sweep_project_workspace(store, pid)
    assert not old.exists()


def test_a_second_sweep_of_the_same_name_keeps_both(tmp_path):
    from ghost_agent.core import workspace_cleanup as WC
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    ws = Path(store.ensure_workspace(pid))
    for body in (b"first", b"second"):
        (ws / "a.zip").write_bytes(body)
        assert WC._to_project_trash(store, pid, ws, ws / "a.zip", "a.zip", "20261102")
    got = sorted(p.read_bytes() for p in (WC.project_trash_dir(store) / "20261102").rglob("*.zip"))
    assert got == [b"first", b"second"]


async def test_a_released_gate_does_not_hold_a_delete(tmp_path, owner_turn):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("e5f6a7b8")
    _say(ctx, f"approve {tid}")
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "DONE"
    owner_turn("SYSTEM")                                # tidy-up later, not the owner's turn
    _say(ctx, "")
    await tool_manage_projects(ctx, action="task_delete", task_id=tid)
    assert not store.get_task(tid)


@pytest.mark.parametrize("said", ["approved, but wait for the backup", "lgtm — don't deploy yet"])
async def test_an_approval_that_holds_back_is_not_the_approval(tmp_path, owner_turn, said):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("e5f6a7b8")
    _say(ctx, said)
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"


async def test_a_dependency_chain_through_a_failure_is_stuck(ap_store, ap_ctx, notes):
    from ghost_agent.core.project_advancer import _pause_if_stuck
    pid = ap_store.create_project("Long job", metadata={"autopilot": True})
    a = ap_store.add_task(pid, "Fetch")
    b = ap_store.add_task(pid, "Clean", depends_on=[a])
    ap_store.add_task(pid, "Chart", depends_on=[b])
    ap_store.update_task(a, status="FAILED")
    _pause_if_stuck(ap_ctx, ap_store, pid)
    assert _meta(ap_store, pid)["autopilot_paused"]["reason"] == "stuck"


async def test_the_project_cannot_be_set_done_past_an_open_gate(tmp_path, owner_turn):
    # r3 MAJOR: action=update status=DONE swept the project past the gate
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    out = str(await tool_manage_projects(ctx, action="update", project_id=pid, status="DONE"))
    assert "NOT updated" in out and tid in out
    assert str(store.get_project(pid)["status"]).upper() != "DONE"


def test_a_best_re_roll_keeps_the_gate_text():
    # r3 minor: BEST copied the child's summary before parking
    from ghost_agent.core.planning import TaskTree, TaskStatus, DependencyType
    tree = TaskTree()
    top = tree.add_task("Pick the best draft", postconditions=GATE)
    tree.nodes[top].dependency_type = DependencyType.BEST
    a = tree.add_task("Draft A", parent_id=top)
    b = tree.add_task("Draft B", parent_id=top)
    tree.update_status(a, TaskStatus.DONE, result="draft A text")
    tree.update_status(b, TaskStatus.DONE, result="a much longer draft B text")
    assert tree.nodes[top].status == TaskStatus.NEEDS_USER
    tree.update_status(b, TaskStatus.DONE, result="an even longer re-done draft B text")
    assert tree.nodes[top].result_summary.startswith("human gate:")



# ─────────────── r3 (operator: "name it") ───────────────

@pytest.mark.parametrize("said", [
    "yes", "ok thanks", "go ahead and tell me the weather", "yes, deploy it",
    "Approved. Wait, not prod!", "LGTM! but hold off on the site deploy",
    "approve the site deploy tomorrow", "all.", "όλα", "(8080", "sure, 3", "approve",
])
async def test_a_plain_or_held_back_yes_does_not_release_the_gate(tmp_path, owner_turn, said):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("e5f6a7b8")
    _say(ctx, said)
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER", said


async def test_approving_one_gate_releases_only_that_one(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    docs = store.add_task(pid, "Publish the release notes", postconditions=["HUMAN_GATE: owner reads the notes"])
    await tool_manage_projects(ctx, action="task_update", task_ids=[tid, docs], status="done", result="done")
    owner_turn("e5f6a7b8")
    _say(ctx, f"approve {docs}")
    await tool_manage_projects(ctx, action="task_update", task_ids=[tid, docs], status="done", result="done")
    assert store.get_task(docs)["status"] == "DONE"
    assert store.get_task(tid)["status"] == "NEEDS_USER"


async def test_approval_by_task_id(tmp_path, owner_turn):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("e5f6a7b8")
    _say(ctx, f"approve {tid[:8]}")
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "DONE"


@pytest.mark.parametrize("said", ["approve the site deploy", "approve Deploy the site", "approve it"])
async def test_title_words_do_not_release_the_gate(tmp_path, owner_turn, said):
    # r4 (operator: task id only): the model could rename the gate to match the owner's words
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("e5f6a7b8")
    _say(ctx, said)
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER", said


@pytest.mark.parametrize("said", [
    "Don\u2019t approve {tid} yet", "I haven't approved {tid}", "I can't approve {tid}",
    "Δεν εγκρίνω το {tid}", "should I approve {tid}?", "approve {tid} only if the tests pass",
    "approve {tid}; actually no",
])
async def test_an_id_approval_that_holds_back_or_asks_is_not_the_approval(tmp_path, owner_turn, said):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("e5f6a7b8")
    _say(ctx, said.format(tid=tid))
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER", said


async def test_the_id_must_be_a_whole_word(tmp_path, owner_turn):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("e5f6a7b8")
    _say(ctx, f"approve commit {tid}9f and x{tid[:8]}")
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"


async def test_a_project_delete_keeps_an_open_gate(tmp_path, owner_turn):
    # r4 MAJOR: a hard project delete wiped the gated task with no owner word
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    _say(ctx, "delete the project")
    out = str(await tool_manage_projects(ctx, action="delete", project_id=pid))
    assert "NOT deleted" in out and tid in out and store.get_project(pid)
    owner_turn("e5f6a7b8")
    _say(ctx, f"delete {tid}")
    await tool_manage_projects(ctx, action="delete", project_id=pid)
    assert not store.get_project(pid)


async def test_the_park_reply_tells_the_model_to_ask_by_name(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    out = await _done(ctx, tid, result="deployed")
    assert "approve <task id>" in out["agent_instruction_not_closed"]
    assert "plain yes does not" in out["agent_instruction_not_closed"]


@pytest.mark.parametrize("said", ["{tid}", "ok {tid}", "yes {tid}", "go ahead with {tid}"])
async def test_the_id_without_an_approval_word_is_not_the_approval(tmp_path, owner_turn, said):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("e5f6a7b8")
    _say(ctx, said.format(tid=tid))
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER", said


async def test_an_approved_gate_does_not_block_the_project_done(tmp_path, owner_turn):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    store.add_task(pid, "Write the notes")
    owner_turn("e5f6a7b8")
    _say(ctx, f"approve {tid}")
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "DONE"
    out = str(await tool_manage_projects(ctx, action="update", project_id=pid, status="DONE"))
    assert "NOT updated" not in out


async def test_another_id_does_not_release_the_gate(tmp_path, owner_turn):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("e5f6a7b8")
    other = "0" * 12 if not tid.startswith("0") else "f" * 12
    _say(ctx, f"approve {other}")
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"



@pytest.mark.parametrize("said", [
    "approve {tid}, leave {other} parked", "approve {other} and delete {tid}", "approve {tid} {other}",
    "I dont approve {tid}", "approve {tid} when the tests pass", "approve {tid} after the backup",
    "εγκρίνω το {tid} όταν περάσουν τα τεστ", "approve {tid}？",
])
async def test_r5_the_id_must_be_the_only_one_and_follow_its_verb(tmp_path, owner_turn, said):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, postconditions=GATE)
    other = store.add_task(pid, "Publish the notes", postconditions=["HUMAN_GATE: owner reads"])
    await tool_manage_projects(ctx, action="task_update", task_ids=[tid, other], status="done", result="done")
    owner_turn("e5f6a7b8")
    _say(ctx, said.format(tid=tid, other=other))
    await tool_manage_projects(ctx, action="task_update", task_ids=[tid, other], status="done", result="done")
    assert store.get_task(other)["status"] == "NEEDS_USER", said
    if "{tid}," in said or "{tid} {other}" in said or "dont" in said or "when" in said or "after" in said \
            or "όταν" in said or "？" in said:
        assert store.get_task(tid)["status"] == "NEEDS_USER", said


async def test_r5_a_stale_context_message_is_not_the_owner_s(tmp_path, owner_turn):
    # the context keeps the LAST owner text; only this request's own message counts
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("e5f6a7b8")
    ctx.last_user_content = f"approve {tid}"           # not recorded for this request
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"


@pytest.mark.parametrize("said", ["approve {tid}, the client is ready", "approve {tid} — the event went fine"])
async def test_r5_words_ending_in_nt_do_not_hold_back(tmp_path, owner_turn, said):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("e5f6a7b8")
    _say(ctx, said.format(tid=tid))
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "DONE", said


@pytest.mark.parametrize("said", ["I don’t approve {tid}", "approve {short3}", "approve {short5}"])
async def test_r5_curly_negation_and_short_prefixes_do_not_release(tmp_path, owner_turn, said):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("e5f6a7b8")
    _say(ctx, said.format(tid=tid, short3=tid[:3], short5=tid[:5]))
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER", said


async def test_r5_an_approval_does_not_delete_the_project(tmp_path, owner_turn):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    _say(ctx, "delete the project")
    await tool_manage_projects(ctx, action="delete", project_id=pid)
    owner_turn("e5f6a7b8")
    _say(ctx, f"approve {tid}")
    out = str(await tool_manage_projects(ctx, action="delete", project_id=pid))
    assert "NOT deleted" in out and store.get_project(pid)


@pytest.mark.parametrize("said", ["approve {tid} before deploying", "approve {tid} provided it builds"])
async def test_r5_a_condition_holds_back(tmp_path, owner_turn, said):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("e5f6a7b8")
    _say(ctx, said.format(tid=tid))
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER", said


async def test_r5_the_verb_must_carry_the_id(tmp_path, owner_turn):
    ctx, store, pid, tid = await _parked(tmp_path, owner_turn)
    owner_turn("e5f6a7b8")
    _say(ctx, f"approve the deploy, it is task {tid}")
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"


async def test_r5_an_ambiguous_prefix_names_no_task(tmp_path, owner_turn, monkeypatch):
    import itertools
    import ghost_agent.memory.projects as MP
    ids = itertools.chain(["abcdef111111", "abcdef222222"], (f"{i:012x}" for i in itertools.count(1)))
    owner_turn("a1b2c3d4")
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="Site", goal="Ship the site")
    pid = ctx.current_project_id
    monkeypatch.setattr(MP, "_new_id", lambda: next(ids))
    tid = store.add_task(pid, "Deploy", postconditions=GATE)
    store.add_task(pid, "Deploy mirror")
    assert tid == "abcdef111111"
    await _done(ctx, tid, result="deployed")
    owner_turn("e5f6a7b8")
    _say(ctx, "approve abcdef")
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "NEEDS_USER"
    owner_turn("c9d0e1f2")
    _say(ctx, "approve abcdef1")
    await _done(ctx, tid, result="deployed")
    assert store.get_task(tid)["status"] == "DONE"


def test_reviving_one_failed_child_reopens_an_any_parent():
    from ghost_agent.core.planning import TaskTree, TaskStatus, DependencyType
    tree = TaskTree()
    top = tree.add_task("Pick a host")
    tree.nodes[top].dependency_type = DependencyType.ANY
    a = tree.add_task("Host A", parent_id=top)
    b = tree.add_task("Host B", parent_id=top)
    tree.update_status(a, TaskStatus.FAILED, result="down")
    tree.update_status(b, TaskStatus.FAILED, result="down")
    assert tree.nodes[top].status == TaskStatus.BLOCKED
    tree.update_status(a, TaskStatus.PENDING)
    assert tree.nodes[top].status == TaskStatus.PENDING
