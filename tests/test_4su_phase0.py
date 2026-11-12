"""§4SU Phase 0 — the confirmed project-system bugs from the 2026-11-02 audit.

Each test names the audit finding it pins (PROJECT_JOURNAL §4SU)."""

from tests._refused import refused_ids  # §4SU Phase 4
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


async def _done(ctx, tid, **kw):
    return json.loads(str(await tool_manage_projects(ctx, action="task_update", task_id=tid,
                                                     status="done", **kw)))


# ─────────────── A (§4UK: the human-gate approval tests are gone — gates no longer park) ───────────────


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


async def test_a_dependency_chain_through_a_failure_is_stuck(ap_store, ap_ctx, notes):
    from ghost_agent.core.project_advancer import _pause_if_stuck
    pid = ap_store.create_project("Long job", metadata={"autopilot": True})
    a = ap_store.add_task(pid, "Fetch")
    b = ap_store.add_task(pid, "Clean", depends_on=[a])
    ap_store.add_task(pid, "Chart", depends_on=[b])
    ap_store.update_task(a, status="FAILED")
    _pause_if_stuck(ap_ctx, ap_store, pid)
    assert _meta(ap_store, pid)["autopilot_paused"]["reason"] == "stuck"


# ─────────────── r3 (operator: "name it") ───────────────


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
