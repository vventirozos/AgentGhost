"""§4MQ (2026-10-08): multi-day work as many bounded UNATTENDED steps.

Each test drives the real gated entry (`advance_unattended`) or the real
project tool against a real `ProjectStore`; only the step body is stubbed
where the test is about the gate around it."""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from ghost_agent.core import project_advancer as PA
from ghost_agent.core.project_advancer import AdvanceResult, advance_unattended
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.projects import tool_manage_projects, tool_manage_projects_for_model
from ghost_agent.utils.logging import request_id_context


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def ctx(store):
    return SimpleNamespace(project_store=store, workspace_model=None, current_project_id=None)


@pytest.fixture
def notes(monkeypatch):
    """Every owner notification the gates send."""
    sent = []

    class _Log:
        def record(self, phase, text, **kw):
            sent.append((phase, text, kw))
    import ghost_agent.core.autonomous_activity as AA
    monkeypatch.setattr(AA, "get_activity_log", lambda c: _Log())
    return sent


def _autopilot_project(store, n_tasks=3, **meta):
    pid = store.create_project("Long job", metadata={"autopilot": True, **meta})
    store.update_project(pid, status="ACTIVE")
    tids = [store.add_task(pid, f"Write chapter {i}") for i in range(n_tasks)]
    return pid, tids


def _meta(store, pid):
    return (store.get_project(pid) or {}).get("metadata") or {}


def _step(store, pid, *, done: bool, delay: float = 0.0):
    """A stand-in step body: claims the first open task; closes it DONE or
    leaves it READY (a step that moved nothing)."""
    async def fake(context, project_id, owner_requested=False, **kw):
        assert owner_requested is False
        if delay:
            await asyncio.sleep(delay)
        open_ = [t for t in store.list_tasks(project_id)
                 if str(t["status"]).upper() in ("PENDING", "READY")]
        if not open_:
            return AdvanceResult(True, None, "idle", "no READY/PENDING leaf")
        t = open_[0]
        if kw.get("claim_sink") is not None:
            kw["claim_sink"].append(t["id"])         # the real claim reports its leaf
        if done:
            store.update_task(t["id"], status="DONE", result_summary="chapter written")
        PA._increment_budget(store, project_id)
        return AdvanceResult(True, t["id"], "research", "worked on it")
    return fake


async def test_three_steps_without_progress_pause_autopilot_and_tell_the_owner_once(store, ctx, notes, monkeypatch):
    pid, _ = _autopilot_project(store)
    monkeypatch.setattr(PA, "advance_once", _step(store, pid, done=False))
    for _ in range(3):
        await advance_unattended(ctx, pid)
    m = _meta(store, pid)
    assert m["autopilot"] is False and m["autopilot_paused"]["reason"] == "no progress"
    assert PA.idle_candidates(store) == []                       # the idle loop no longer picks it
    assert len(notes) == 1 and "autopilot paused" in notes[0][1] and notes[0][2]["severity"] == "notify"


async def test_a_step_that_closes_its_task_resets_the_streak(store, ctx, notes, monkeypatch):
    pid, _ = _autopilot_project(store, n_tasks=4)
    monkeypatch.setattr(PA, "advance_once", _step(store, pid, done=False))
    await advance_unattended(ctx, pid)
    await advance_unattended(ctx, pid)
    monkeypatch.setattr(PA, "advance_once", _step(store, pid, done=True))
    await advance_unattended(ctx, pid)
    m = _meta(store, pid)
    assert m["no_progress_streak"] == 0 and m["autopilot"] is True and notes == []


async def test_a_tick_that_claimed_nothing_is_not_scored(store, ctx, notes, monkeypatch):
    pid, _ = _autopilot_project(store, n_tasks=0)
    for _ in range(4):
        await advance_unattended(ctx, pid)                       # the real step: "no READY/PENDING leaf"
    assert int(_meta(store, pid).get("no_progress_streak") or 0) == 0 and notes == []


async def test_the_owner_is_asked_to_look_every_checkpoint_steps(store, ctx, notes, monkeypatch):
    pid, _ = _autopilot_project(store, n_tasks=5, checkpoint_every=2)
    monkeypatch.setattr(PA, "advance_once", _step(store, pid, done=True))
    await advance_unattended(ctx, pid)
    assert _meta(store, pid)["autopilot"] is True
    await advance_unattended(ctx, pid)
    m = _meta(store, pid)
    assert m["autopilot"] is False and m["autopilot_paused"]["reason"] == "check-in"
    assert len(notes) == 1


async def test_a_step_past_its_wall_cap_is_stopped_charged_and_its_task_reopened(store, ctx, notes, monkeypatch):
    pid, tids = _autopilot_project(store, n_tasks=1)

    async def hang(context, project_id, owner_requested=False, claim_sink=None, **kw):
        store.update_task(tids[0], status="IN_PROGRESS")
        claim_sink.append(tids[0])
        await asyncio.sleep(30)
    monkeypatch.setattr(PA, "advance_once", hang)
    t0 = time.monotonic()
    res = await advance_unattended(ctx, pid, step_timeout_s=0.2)
    assert time.monotonic() - t0 < 5
    assert res.classification == "timeout"
    assert store.get_task(tids[0])["status"] == "READY"
    m = _meta(store, pid)
    assert m["steps_used"] == 1 and m["no_progress_streak"] == 1 and not m.get("step_in_flight")


async def test_a_step_lost_to_a_restart_is_charged_on_the_next_tick(store, ctx, notes, monkeypatch):
    pid, _ = _autopilot_project(store, steps_used=4, no_progress_streak=2)
    store.update_project(pid, metadata={"step_in_flight": {"boot": "41-1", "ts": time.time() - 600, "id": "x"}})   # a dead process's step
    ran = []

    async def body(*a, **k):
        ran.append(1)
        return AdvanceResult(True, None, "idle", "x")
    monkeypatch.setattr(PA, "advance_once", body)
    res = await advance_unattended(ctx, pid)
    m = _meta(store, pid)
    assert m["steps_used"] == 5                                   # the lost step is charged
    assert m["autopilot"] is False and res.classification == "blocked" and not ran
    assert not m.get("step_in_flight")


async def test_the_default_runtime_cap_pauses_unattended_work(store, ctx, notes, monkeypatch):
    pid, _ = _autopilot_project(store, unattended_runtime_seconds=PA.DEFAULT_UNATTENDED_RUNTIME_CAP_S + 1)
    ran = []
    monkeypatch.setattr(PA, "advance_once", lambda *a, **k: ran.append(1))
    res = await advance_unattended(ctx, pid)
    assert not ran and res.classification == "blocked"
    assert _meta(store, pid)["autopilot_paused"]["reason"] == "runtime budget used"


async def test_an_unattended_batch_stops_when_the_gates_pause(store, ctx, notes, monkeypatch):
    pid, _ = _autopilot_project(store, n_tasks=8)
    monkeypatch.setattr(PA, "advance_once", _step(store, pid, done=False))
    batch = await PA.advance_many(ctx, pid, max_tasks=None, owner_requested=False, stop_on_fail=False)
    assert batch.stop_reason == "autopilot_paused" and batch.count == 3


async def test_an_owner_batch_stops_before_a_step_that_would_cross_the_deadline(store, ctx, monkeypatch):
    pid, _ = _autopilot_project(store, n_tasks=6)

    async def slow_owner_step(context, project_id, owner_requested=False, **kw):
        await asyncio.sleep(0.05)
        open_ = [t for t in store.list_tasks(project_id) if str(t["status"]).upper() in ("PENDING", "READY")]
        if not open_:
            return AdvanceResult(True, None, "idle", "no READY/PENDING leaf")
        store.update_task(open_[0]["id"], status="DONE")
        return AdvanceResult(True, open_[0]["id"], "coding", "built")
    monkeypatch.setattr(PA, "advance_once", slow_owner_step)
    import ghost_agent.utils.logging as G
    monkeypatch.setattr(G, "request_remaining_s", lambda rid: 30.03)
    monkeypatch.setattr(G, "request_deadline_s", lambda rid="": 1800.0)
    import ghost_agent.core.agent as A
    monkeypatch.setattr(A, "effective_report_floor", lambda d: 30.0)
    batch = await PA.advance_many(ctx, pid, max_tasks=None, owner_requested=True)
    # §4MR: the FIRST step is checked too (an estimate stands in for a duration)
    assert batch.stop_reason == "deadline" and batch.count == 0
    monkeypatch.setattr(G, "request_remaining_s", lambda rid: 30.0 + PA.FIRST_STEP_ESTIMATE_S + 0.03)
    batch = await PA.advance_many(ctx, pid, max_tasks=None, owner_requested=True)
    # room for the estimated first step: it runs, and the short steps that
    # follow (measured) all fit
    assert batch.count == 6 and batch.stop_reason == "project_done"


# ── the project tool ──────────────────────────────────────────────────

async def test_the_model_cannot_raise_a_budget_through_metadata(ctx, store):
    pid = store.create_project("Job")
    res = await tool_manage_projects_for_model(ctx, action="update", project_id=pid,
                                               metadata={"steps_cap": 999, "note": "x"})
    assert "action=budget" in res
    assert int(_meta(store, pid).get("steps_cap") or PA.DEFAULT_STEPS_CAP) == PA.DEFAULT_STEPS_CAP


async def test_only_the_owners_turn_sets_a_budget(ctx, store):
    pid = store.create_project("Job")
    tok = request_id_context.set("probe-9")
    try:
        res = await tool_manage_projects_for_model(ctx, action="budget", project_id=pid,
                                                   metadata={"steps_cap": 200})
        assert "only the owner" in res
        request_id_context.set("req-owner")
        res = json.loads(await tool_manage_projects_for_model(
            ctx, action="budget", project_id=pid,
            metadata={"steps_cap": 200, "runtime_cap_hours": 2, "checkpoint_every": 5}))
    finally:
        request_id_context.reset(tok)
    m = _meta(store, pid)
    assert m["steps_cap"] == 200 and m["unattended_runtime_cap_seconds"] == 7200 and m["checkpoint_every"] == 5
    assert "runtime_cap_seconds" not in m          # the lifetime cap would block the owner's own runs (r2 M3)
    assert res["budget"]["steps"] == "0/200" and res["budget"]["unattended_hours"].endswith("/2.0")


async def test_the_owners_resume_resets_the_gates(ctx, store):
    pid, _ = _autopilot_project(store, no_progress_streak=3, steps_since_checkpoint=10)
    store.update_project(pid, metadata={"autopilot": False, "autopilot_paused": {"reason": "no progress"}})
    tok = request_id_context.set("req-owner")
    try:
        await tool_manage_projects(ctx, action="autopilot", project_id=pid, enabled=True)
    finally:
        request_id_context.reset(tok)
    m = _meta(store, pid)
    assert m["autopilot"] is True and not m.get("autopilot_paused")
    assert m["no_progress_streak"] == 0 and m["steps_since_checkpoint"] == 0


async def test_one_command_pauses_every_project_and_never_turns_them_all_on(ctx, store):
    a, _ = _autopilot_project(store)
    b, _ = _autopilot_project(store)
    tok = request_id_context.set("req-owner")
    try:
        res = await tool_manage_projects(ctx, action="autopilot", project_id="all", enabled=True)
        assert "enabled=false only" in res
        res = json.loads(await tool_manage_projects(ctx, action="autopilot", project_id="all", enabled=False))
    finally:
        request_id_context.reset(tok)
    assert len(res["autopilot_off"]) == 2
    assert not _meta(store, a)["autopilot"] and not _meta(store, b)["autopilot"]


async def test_status_shows_the_budget_and_the_gates(ctx, store):
    pid, _ = _autopilot_project(store, steps_used=7, no_progress_streak=1)
    res = json.loads(await tool_manage_projects(ctx, action="status", project_id=pid, title="Long job"))
    assert res["budget"]["steps"] == "7/50" and res["budget"]["no_progress_streak"] == "1/3"


def test_the_idle_loop_uses_the_gated_step():
    """The idle advancer is the only unattended entry; it must go through the
    gates (AST: the call site names advance_unattended, never advance_once)."""
    import ast
    import inspect
    import ghost_agent.core.agent as A
    tree = ast.parse(inspect.getsource(A))
    names = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
             and (n.module or "").endswith("project_advancer") for a in n.names
             if a.name in ("advance_once", "advance_unattended")}
    assert "_advance_unattended" in names and "_advance_once" not in names
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_advance_unattended" in called
    assert not ({"advance_once", "_advance_once"} & called)       # nothing in agent.py steps ungated


# ── r2 (fresh reader) ─────────────────────────────────────────────────

async def test_a_paused_project_is_not_advanced_by_a_background_run_and_notifies_once(store, ctx, notes, monkeypatch):
    """r2 C1: after the pause, each scheduled run still built one step and
    sent another "paused" message."""
    pid, _ = _autopilot_project(store, n_tasks=8)
    ran = []
    inner = _step(store, pid, done=False)

    async def counting(*a, **k):
        ran.append(1)
        return await inner(*a, **k)
    monkeypatch.setattr(PA, "advance_once", counting)
    b = await PA.advance_many(ctx, pid, max_tasks=None, owner_requested=False, stop_on_fail=False)
    assert b.stop_reason == "autopilot_paused" and len(ran) == 3
    for _ in range(3):
        b = await PA.advance_many(ctx, pid, max_tasks=None, owner_requested=False, stop_on_fail=False)
        assert b.stop_reason == "autopilot_off"
    assert len(ran) == 3 and len(notes) == 1


async def test_the_owners_off_switch_stops_a_running_background_batch(store, ctx, notes, monkeypatch):
    """r2 C2: the batch only watched for a GATE pause; "stop all" let it build
    every remaining task."""
    pid, _ = _autopilot_project(store, n_tasks=8)
    inner = _step(store, pid, done=True)
    ran = []

    async def then_owner_stops_all(*a, **k):
        r = await inner(*a, **k)
        ran.append(1)
        if len(ran) == 1:
            tok = request_id_context.set("req-owner")
            try:
                await tool_manage_projects(ctx, action="autopilot", project_id="all", enabled=False)
            finally:
                request_id_context.reset(tok)
        return r
    monkeypatch.setattr(PA, "advance_once", then_owner_stops_all)
    b = await PA.advance_many(ctx, pid, max_tasks=None, owner_requested=False, stop_on_fail=False)
    assert len(ran) == 1 and b.stop_reason == "autopilot_off"


async def test_a_concurrent_step_is_refused_not_charged_as_lost(store, ctx, notes, monkeypatch):
    """r2 M1: a second step saw the first one's marker and charged a step that
    never died."""
    pid, _ = _autopilot_project(store)
    monkeypatch.setattr(PA, "advance_once", _step(store, pid, done=True, delay=0.2))
    first = asyncio.ensure_future(advance_unattended(ctx, pid))
    await asyncio.sleep(0.05)
    second = await advance_unattended(ctx, pid)
    await first
    m = _meta(store, pid)
    assert second.classification == "blocked" and "running" in second.summary
    assert m["steps_used"] == 1 and not m.get("step_in_flight")
    assert not [e for e in store.list_events(pid) if "autopilot_step_lost" in json.dumps(e)]


async def test_a_timeout_reopens_only_its_own_leaf(store, ctx, notes, monkeypatch):
    """r2 M2: every IN_PROGRESS task went back to READY — a leaf another
    step was building got claimed and built twice."""
    pid, tids = _autopilot_project(store, n_tasks=2)
    store.update_task(tids[1], status="IN_PROGRESS")          # someone else's live claim

    async def hang(context, project_id, owner_requested=False, claim_sink=None, **kw):
        store.update_task(tids[0], status="IN_PROGRESS")
        claim_sink.append(tids[0])
        await asyncio.sleep(30)
    monkeypatch.setattr(PA, "advance_once", hang)
    await advance_unattended(ctx, pid, step_timeout_s=0.1)
    assert store.get_task(tids[0])["status"] == "READY"
    assert store.get_task(tids[1])["status"] == "IN_PROGRESS"


async def test_the_owners_own_runtime_does_not_spend_the_unattended_allowance(store, ctx, notes, monkeypatch):
    """r2 M3: lifetime runtime (the owner's builds too) tripped the "6 h of
    unattended work" pause on every resume, forever."""
    pid, _ = _autopilot_project(store, runtime_used_seconds=PA.DEFAULT_UNATTENDED_RUNTIME_CAP_S + 5,
                                unattended_runtime_seconds=PA.DEFAULT_UNATTENDED_RUNTIME_CAP_S + 5)
    store.update_project(pid, metadata={"autopilot": False})
    tok = request_id_context.set("req-owner")
    try:
        await tool_manage_projects(ctx, action="autopilot", project_id=pid, enabled=True)
    finally:
        request_id_context.reset(tok)
    ran = []
    inner = _step(store, pid, done=True)

    async def counting(*a, **k):
        ran.append(1)
        return await inner(*a, **k)
    monkeypatch.setattr(PA, "advance_once", counting)
    await advance_unattended(ctx, pid)
    assert ran and _meta(store, pid)["autopilot"] is True and notes == []


async def test_a_task_held_for_the_owner_is_neither_progress_nor_a_failure(store, ctx, notes, monkeypatch):
    """r2 m1: three test-shaped tasks held for the owner paused the project as
    "no progress" after three hold notices."""
    pid, tids = _autopilot_project(store, n_tasks=4, no_progress_streak=2)

    async def hold(context, project_id, owner_requested=False, claim_sink=None, **kw):
        store.update_task(tids[0], status="NEEDS_USER")
        return AdvanceResult(True, tids[0], "coding", "held")
    monkeypatch.setattr(PA, "advance_once", hold)
    await advance_unattended(ctx, pid)
    m = _meta(store, pid)
    assert m["no_progress_streak"] == 2 and m["autopilot"] is True and notes == []


async def test_a_step_that_raises_is_charged_and_its_leaf_reopened(store, ctx, notes, monkeypatch):
    """r2 m3: a raising step left no charge, no streak, and its leaf claimed
    until the next boot."""
    pid, tids = _autopilot_project(store, n_tasks=1)

    async def boom(context, project_id, owner_requested=False, claim_sink=None, **kw):
        store.update_task(tids[0], status="IN_PROGRESS")
        claim_sink.append(tids[0])
        raise RuntimeError("sandbox went away")
    monkeypatch.setattr(PA, "advance_once", boom)
    res = await advance_unattended(ctx, pid)
    m = _meta(store, pid)
    assert res.classification == "error" and store.get_task(tids[0])["status"] == "READY"
    assert m["steps_used"] == 1 and m["no_progress_streak"] == 1 and not m.get("step_in_flight")


async def test_an_unattended_step_needs_autopilot_on(store, ctx, notes, monkeypatch):
    pid, _ = _autopilot_project(store)
    store.update_project(pid, metadata={"autopilot": False})
    ran = []
    monkeypatch.setattr(PA, "advance_once", lambda *a, **k: ran.append(1))
    res = await advance_unattended(ctx, pid)
    assert not ran and res.classification == "blocked" and notes == []


def test_a_second_pause_while_paused_sends_nothing(store, ctx, notes):
    """r2 C1: only the on → off transition tells the owner."""
    pid, _ = _autopilot_project(store)
    PA.pause_autopilot(ctx, pid, "no progress", "x")
    PA.pause_autopilot(ctx, pid, "runtime budget used", "y")
    assert len(notes) == 1 and _meta(store, pid)["autopilot_paused"]["reason"] == "no progress"


async def test_a_step_clears_only_its_own_in_flight_marker(store, ctx, notes, monkeypatch):
    """r2 M1: a step's `finally` cleared whatever marker was there — another
    process's live step lost its crash charge."""
    pid, _ = _autopilot_project(store)
    other = {"boot": "other-boot", "ts": time.time(), "id": "theirs"}
    inner = _step(store, pid, done=True)

    async def body(*a, **k):
        r = await inner(*a, **k)
        store.update_project(pid, metadata={"step_in_flight": other})   # another process stamped it
        return r
    monkeypatch.setattr(PA, "advance_once", body)
    await advance_unattended(ctx, pid)
    assert _meta(store, pid)["step_in_flight"] == other


async def test_a_step_that_finished_as_its_cap_hit_is_not_charged_twice(store, ctx, notes, monkeypatch):
    """r2 m2: `wait_for` can raise the timeout after the step finished; it
    was charged again and scored as no progress."""
    pid, tids = _autopilot_project(store, n_tasks=2)

    async def done_then_slow(context, project_id, owner_requested=False, claim_sink=None, **kw):
        claim_sink.append(tids[0])
        store.update_task(tids[0], status="DONE")
        PA._increment_budget(store, project_id)
        await asyncio.sleep(30)
    monkeypatch.setattr(PA, "advance_once", done_then_slow)
    await advance_unattended(ctx, pid, step_timeout_s=0.1)
    m = _meta(store, pid)
    assert m["steps_used"] == 1 and m["no_progress_streak"] == 0


# ── §4MR (fresh-eye verification): the REAL step, end to end ──────────

from ghost_agent.core.coding_executor import CodingResult  # noqa: E402

_SEARCH_OUT = ("1. The abacus — Wikipedia\nThe abacus is a calculating tool used since ancient times "
               "in Sumer, Babylon, China and Rome. Beads slide on rods ...\n"
               "2. History of the abacus\nThe Chinese suanpan dates to the 2nd century BC ...\n") * 4


def _coding_project(store, descs, **meta):
    pid = store.create_project("Build job", kind="CODING", metadata={"autopilot": True, **meta})
    store.update_project(pid, status="ACTIVE")
    return pid, [store.add_task(pid, d) for d in descs]


def _executor(raise_=False):
    async def ex(context, description, **kw):
        if raise_:
            raise RuntimeError("spec LLM down")
        return CodingResult(ok=True, summary="built the page", files=["page.html"])
    return ex


async def _runner(name, args):
    return _SEARCH_OUT


async def test_a_real_coding_step_that_closes_its_task_is_progress(store, ctx, notes):
    pid, tids = _coding_project(store, ["Build the login page", "Build the signup page"])
    await advance_unattended(ctx, pid, tool_runner=_runner, coding_executor=_executor())
    m = _meta(store, pid)
    assert store.get_task(tids[0])["status"] == "DONE"
    assert m["steps_since_checkpoint"] == 1 and m["no_progress_streak"] == 0


async def test_a_real_timeout_reopens_the_leaf_the_real_claim_reported(store, ctx, notes):
    """The real `advance_once(claim_sink=…)` — every other gate test stubs it."""
    pid = store.create_project("Research job", metadata={"autopilot": True})
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Research the history of the abacus")

    async def hang(name, args):
        await asyncio.sleep(5)
    res = await advance_unattended(ctx, pid, tool_runner=hang, step_timeout_s=0.2)
    assert res.task_id == tid and store.get_task(tid)["status"] == "READY"


async def test_a_crashed_build_reopens_its_task_and_three_crashes_pause(store, ctx, notes):
    """§4MR M1: the crash path "left the leaf open" as IN_PROGRESS — no tick
    could retry it, the gate scored nothing, the owner's batch said done."""
    pid, tids = _coding_project(store, ["Build the login page", "Build the signup page"])
    await advance_unattended(ctx, pid, tool_runner=_runner, coding_executor=_executor(raise_=True))
    assert store.get_task(tids[0])["status"] == "READY"
    for _ in range(2):
        await advance_unattended(ctx, pid, tool_runner=_runner, coding_executor=_executor(raise_=True))
    m = _meta(store, pid)
    assert m["autopilot"] is False and m["autopilot_paused"]["reason"] == "no progress" and len(notes) == 1


async def test_an_owner_batch_over_a_crashing_build_says_it_crashed(store, ctx, notes):
    pid, tids = _coding_project(store, ["Build the login page"])
    b = await PA.advance_many(ctx, pid, max_tasks=None, owner_requested=True, tool_runner=_runner,
                              coding_executor=_executor(raise_=True), stop_on_fail=False)
    assert b.stop_reason == "step_crashed"


async def test_an_owner_batch_never_reports_done_over_a_task_still_in_progress(store, ctx, notes):
    pid, tids = _coding_project(store, ["Build the login page"])
    store.update_task(tids[0], status="IN_PROGRESS")         # another run's claim, or one cut by a restart
    b = await PA.advance_many(ctx, pid, max_tasks=None, owner_requested=True, tool_runner=_runner,
                              coding_executor=_executor(), stop_on_fail=False)
    assert b.stop_reason == "in_progress"


async def test_the_owners_resume_keeps_a_running_steps_marker(store, ctx, notes, monkeypatch):
    """§4MR M2: the resume cleared a LIVE step's marker and a second step ran
    beside it on the same project."""
    pid, _ = _autopilot_project(store)
    started, live, overlap = asyncio.Event(), [], []

    async def slow(context, project_id, owner_requested=False, **kw):
        live.append(1)
        if len(live) > 1:
            overlap.append(1)
        started.set()
        await asyncio.sleep(0.2)
        live.pop()
        return AdvanceResult(True, None, "idle", "nothing")
    monkeypatch.setattr(PA, "advance_once", slow)
    first = asyncio.ensure_future(advance_unattended(ctx, pid))
    await started.wait()
    tok = request_id_context.set("req-owner")
    try:
        await tool_manage_projects(ctx, action="autopilot", project_id=pid, enabled=True)
    finally:
        request_id_context.reset(tok)
    second = await advance_unattended(ctx, pid)
    await first
    assert not overlap and second.classification == "blocked"


async def test_a_batch_refused_by_a_running_step_does_not_blame_the_budget(store, ctx, notes):
    pid, _ = _autopilot_project(store)
    store.update_project(pid, metadata={"step_in_flight": {"boot": PA._BOOT_ID, "ts": 0, "id": "x"}})
    b = await PA.advance_many(ctx, pid, max_tasks=None, owner_requested=False, stop_on_fail=False)
    assert b.stop_reason == "busy"


async def test_the_owners_autoadvance_says_when_it_turns_autopilot_back_on(store, ctx, notes, monkeypatch):
    """§4MR m1: after "stop all", one "do the next task" restarted overnight
    work without a word (the turn-on itself is the §4LQ decision)."""
    pid, _ = _autopilot_project(store)
    store.update_project(pid, metadata={"autopilot": False, "autopilot_paused": {"reason": "check-in"}})

    async def one(context, project_id, owner_requested=False, **kw):
        return AdvanceResult(True, None, "idle", "no READY/PENDING leaf")
    monkeypatch.setattr(PA, "advance_once", one)
    tok = request_id_context.set("req-owner")
    try:
        out = json.loads(await tool_manage_projects(ctx, action="autoadvance", project_id=pid, count="1"))
    finally:
        request_id_context.reset(tok)
    assert "autopilot is now ON" in out["autopilot"] and "check-in" in out["autopilot"]


def test_a_pause_appears_in_the_while_you_were_away_digest(store, ctx, notes):
    from ghost_agent.core.project_digest import render_digest, summarize_since
    pid, _ = _autopilot_project(store)
    PA.pause_autopilot(ctx, pid, "no progress", "x")
    text = render_digest(summarize_since(store, 0))
    assert "PAUSED its autopilot" in text and "no progress" in text



async def test_an_exhausted_step_budget_pauses_autopilot_and_says_so(store, ctx, notes, monkeypatch):
    """§4MR: the budget stop blocked every tick silently with autopilot ON."""
    pid, _ = _autopilot_project(store, steps_used=50, steps_cap=50)
    ran = []
    monkeypatch.setattr(PA, "advance_once", lambda *a, **k: ran.append(1))
    for _ in range(3):
        await advance_unattended(ctx, pid)
    m = _meta(store, pid)
    assert not ran and m["autopilot"] is False and m["autopilot_paused"]["reason"] == "step budget used"
    assert len(notes) == 1



async def test_the_crash_path_itself_reopens_its_task(store, ctx, notes):
    """The owner's run has no unattended gate behind it: the crash path must
    leave the task claimable on its own."""
    pid, tids = _coding_project(store, ["Build the login page"])
    res = await PA.advance_once(ctx, pid, owner_requested=True, tool_runner=_runner,
                                coding_executor=_executor(raise_=True))
    assert res.summary.startswith("coding executor crashed") and store.get_task(tids[0])["status"] == "READY"


async def test_a_step_that_returns_with_its_leaf_still_claimed_gets_it_reopened(store, ctx, notes, monkeypatch):
    """Any path that stops without closing its leaf — the gate reopens it."""
    pid, tids = _autopilot_project(store, n_tasks=1)

    async def stops_open(context, project_id, owner_requested=False, claim_sink=None, **kw):
        store.update_task(tids[0], status="IN_PROGRESS")
        claim_sink.append(tids[0])
        return AdvanceResult(True, tids[0], "blocked", "stopped without closing it")
    monkeypatch.setattr(PA, "advance_once", stops_open)
    await advance_unattended(ctx, pid)
    assert store.get_task(tids[0])["status"] == "READY" and _meta(store, pid)["no_progress_streak"] == 1
