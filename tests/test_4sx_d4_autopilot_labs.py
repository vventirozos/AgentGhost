"""§4SX Phase D4: autopilot on labs — the leaf gets `lab` (disposable only), a build that could not get a lab waits
for the labs host and starts again by itself (core/lab_waits.py), the leaf's labs end with it, and a scheduled task
may belong to a project (listed in its inventory, removed by its delete)."""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core import lab_waits as W
from ghost_agent.core import project_advancer as PA
from ghost_agent.core.coding_executor import CodingResult
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools import labs as L
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome

HEALTH_ROOM = {"ok": True, "ram_budget_mb": 12000, "kept_mb": 4000, "disposable_mb_used": 1000, "labs": 1,
               "max_labs": 6, "mem_available_mb": 6000, "disk_free_mb": 50000}


@pytest.fixture(autouse=True)
def _clean():
    W._SEEN.clear()
    yield
    W._SEEN.clear()


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")


def _actx(store, **kw):
    return SimpleNamespace(project_store=store, llm_client=None, scratchpad=None, workspace_model=None,
                           current_project_id=None, args=SimpleNamespace(labs_host="", labs_key_file=""), **kw)


# ------------------------------------------------------------------ the leaf's tools

def test_the_leaf_has_lab_and_the_prompt_says_how():
    from ghost_agent.core import coding_loop as CL
    assert CL.LEAF_ALLOWED_TOOLS == ("file_system", "execute", "lab", "run")
    p = CL._leaf_prompt("build x", "", {}, None, False, "")
    assert "`lab` gives a disposable one" in p and "destroy it when you are done" in p


# ------------------------------------------------------------------ lab_waits

def test_a_no_lab_answer_is_remembered_per_build_and_only_a_create_clears_it():
    W.note_outcome("b1", "lab_labs_unreachable", False, action="create")
    W.note_outcome("b1", "", True, action="list")                   # review r2 #3: a read never clears it
    W.note_outcome("b2", "lab_lab_limit", False, action="create")
    assert W.take_seen("b1") == ("labs_down", 0) and W.take_seen("b1") == ("", 0)     # taken once
    W.note_outcome("b2", "", True, action="create")                 # the build got its lab
    assert W.take_seen("b2") == ("", 0)
    W.note_outcome("b3", "lab_budget_exceeded", False, action="create",
                   text="LAB ERROR (budget_exceeded): x\nneeded_mb: 3072\navailable_mb: 100")
    assert W.take_seen("b3") == ("lab_capacity", 3072)
    for rc in ("lab_lab_limit", "lab_disk_low"):
        W.note_outcome("b4", rc, False, action="create")
        assert W.take_seen("b4")[0] == "lab_capacity"
    W.note_outcome("b5", "lab_no_answer", False, action="create")
    assert W.take_seen("b5")[0] == "labs_no_answer"                 # review r2 #14 / r3 #4: a hanging host
    W.note_outcome("b6", "lab_labs_unreachable", False, action="templates")
    assert W.take_seen("b6") == ("", 0)                             # review r3 #3: only a create is noted
    W.note_outcome("", "lab_labs_unreachable", False, action="create")
    assert W.take_seen("") == ("", 0)
    for k in range(300):
        W.note_outcome(f"x{k}", "lab_lab_limit", False, action="create")
    assert len(W._SEEN) <= 256


def test_a_disk_hold_needs_the_hosts_own_floor():
    """review r3 #1: the host refuses a create below disk_floor_mb (10 GB by default) — 8 GB free is no room."""
    assert W.has_room({**HEALTH_ROOM, "disk_free_mb": 8000}, "lab_capacity") is False
    assert W.has_room({**HEALTH_ROOM, "disk_free_mb": 12000}, "lab_capacity") is True
    assert W.has_room({**HEALTH_ROOM, "disk_free_mb": 12000, "disk_floor_mb": 20000}, "lab_capacity") is False
    assert W.has_room({**HEALTH_ROOM, "disk_free_mb": 21100, "disk_floor_mb": 20000}, "lab_capacity") is True


def test_a_no_answer_hold_waits_its_cooldown_and_a_free_slot():
    """review r3 #4: the create may have landed — never retried the moment health answers."""
    import time as _t
    now = _t.time()
    assert W.has_room(HEALTH_ROOM, "labs_no_answer", held_at=now) is False
    assert W.has_room(HEALTH_ROOM, "labs_no_answer", held_at=now - W.NO_ANSWER_COOLDOWN_S - 1) is True
    assert W.has_room({**HEALTH_ROOM, "labs": 6}, "labs_no_answer", held_at=now - 10**5) is False
    assert W.has_room({**HEALTH_ROOM, "ok": False}, "labs_no_answer", held_at=now - 10**5) is False


async def test_a_no_answer_hold_is_kept_by_the_pick_until_the_cooldown(store, monkeypatch):
    pid = store.create_project("P")
    t1 = store.add_task(pid, "build a")
    W.set_hold(store, pid, t1, "labs_no_answer", "x")
    monkeypatch.setattr(L, "labs_configured", lambda c: True)

    async def _call(ctx, m, path, **kw):
        return 200, HEALTH_ROOM
    monkeypatch.setattr(L, "_call", _call)
    assert await W.held_ids(SimpleNamespace(), store, pid) == {t1}
    W._mutate(store, pid, lambda h: {k: {**v, "at": 1.0} for k, v in h.items()})
    assert await W.held_ids(SimpleNamespace(), store, pid) == set()


async def test_unattended_work_with_no_project_lists_and_creates_no_lab(store, monkeypatch):
    """review r3 #2: an empty owner is no host filter — it would list every owner's labs."""
    calls = []

    async def _call(c, m, path, **kw):
        calls.append(path)
        return 200, {"labs": []}
    monkeypatch.setattr(L, "_call", _call)
    monkeypatch.setattr(L, "labs_configured", lambda c: True)
    monkeypatch.setattr(L, "_caller", lambda c: ("autopilot", "", ""))
    for action in ("list", "create"):
        out = await L.tool_lab(_actx(store), action=action, template="base")
        assert "no project uses no lab" in str(out), str(out)
    assert calls == []


def test_a_capacity_hold_needs_the_size_the_create_needed():
    assert W.has_room(HEALTH_ROOM, "lab_capacity", 3000) is True
    assert W.has_room(HEALTH_ROOM, "lab_capacity", 7500) is False


@pytest.mark.parametrize("health,code,want", [
    (HEALTH_ROOM, "labs_down", True),
    (HEALTH_ROOM, "lab_capacity", True),
    ({**HEALTH_ROOM, "ok": False}, "labs_down", False),
    (None, "labs_down", False),
    ({**HEALTH_ROOM, "labs": 6}, "lab_capacity", False),
    ({**HEALTH_ROOM, "disposable_mb_used": 7500}, "lab_capacity", False),
    ({**HEALTH_ROOM, "mem_available_mb": 500}, "lab_capacity", False),
    ({**HEALTH_ROOM, "disk_free_mb": 100}, "lab_capacity", False),
    ({"ok": True}, "lab_capacity", False),
    ({"ok": True}, "labs_down", True),
])
def test_has_room(health, code, want):
    assert W.has_room(health, code) is want


async def test_held_ids_lifts_on_room_drops_stale_and_keeps_on_silence(store, monkeypatch):
    pid = store.create_project("P")
    t1 = store.add_task(pid, "build a")
    t2 = store.add_task(pid, "build b")
    W.set_hold(store, pid, t1, "labs_down", "x")
    W.set_hold(store, pid, t2, "lab_capacity", "y")
    W.set_hold(store, pid, "gone-task", "labs_down", "z")
    ctx = SimpleNamespace()
    answers = {"v": (0, None)}

    async def _call(c, m, path, **kw):
        if isinstance(answers["v"], Exception):
            raise answers["v"]
        return answers["v"]
    monkeypatch.setattr(L, "_call", _call)
    monkeypatch.setattr(L, "labs_configured", lambda c: True)
    answers["v"] = L.LabsUnreachable("down")
    assert await W.held_ids(ctx, store, pid) == {t1, t2}
    assert set(W.holds_of(store.get_project(pid))) == {t1, t2}               # the stale one went
    answers["v"] = (200, {**HEALTH_ROOM, "labs": 6})                         # answers, but full
    assert await W.held_ids(ctx, store, pid) == {t2}
    assert set(W.holds_of(store.get_project(pid))) == {t2}
    assert store.latest_task_event(pid, t1, ("lab_wait_lifted",))["payload"]["code"] == "labs_down"
    answers["v"] = (200, HEALTH_ROOM)
    assert await W.held_ids(ctx, store, pid) == set() and W.holds_of(store.get_project(pid)) == {}


async def test_held_ids_lifts_everything_when_labs_are_no_longer_configured(store, monkeypatch):
    pid = store.create_project("P")
    t1 = store.add_task(pid, "build a")
    W.set_hold(store, pid, t1, "labs_down", "x")
    monkeypatch.setattr(L, "labs_configured", lambda c: False)
    assert await W.held_ids(SimpleNamespace(), store, pid) == set()


async def test_the_lab_tool_tells_lab_waits_under_the_running_build(monkeypatch):
    seen = []
    monkeypatch.setattr(W, "note_outcome", lambda key, rc, ok, action="", text="": seen.append((key, rc, ok, action)))
    tok = W.BUILD_KEY.set("bld1")
    try:
        L._note_lab_wait(ToolOutcome("x", status=OutcomeStatus.FAILED, reason_code="lab_lab_limit"), "create")
        L._note_lab_wait(ToolOutcome("ok", status=OutcomeStatus.OK), "list")
    finally:
        W.BUILD_KEY.reset(tok)
    assert seen == [("bld1", "lab_lab_limit", False, "create"), ("bld1", "", True, "list")]


# ------------------------------------------------------------------ the advancer

def _fake_build(lab_wait="labs_down", ok=False):
    calls = []

    async def exe(ctx, desc, **kw):
        calls.append(desc)
        return CodingResult(ok, "agentic leaf failed: the lab could not be created", files=[], lab_wait=lab_wait)
    return exe, calls


async def _tick(store, pid, *, owner=False, exe=None, ctx=None):
    async def runner(name, args):
        return "ok"
    return await PA.advance_once(ctx or _actx(store), pid, tool_runner=runner, coding_executor=exe,
                                 llm_classifier=AsyncMock(return_value="coding"), owner_requested=owner)


async def test_a_build_that_could_not_get_a_lab_goes_back_to_pending_with_a_hold(store, monkeypatch):
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Build the parser module", kind="build")
    exe, calls = _fake_build("lab_capacity")
    res = await _tick(store, pid, exe=exe)
    assert calls and res.summary.startswith(PA.LAB_WAIT_PREFIX) and "[lab_capacity]" in res.summary
    t = store.get_task(tid)
    assert t["status"] == "PENDING"
    assert W.holds_of(store.get_project(pid))[tid]["code"] == "lab_capacity"
    assert store.latest_task_event(pid, tid, ("autoadvance_lab_wait",))["payload"]["code"] == "lab_capacity"
    assert store.get_project(pid)["status"] == "ACTIVE"                  # the autopilot is not stopped


async def test_the_unattended_pick_skips_a_held_task_until_the_labs_host_has_room(store, monkeypatch):
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Build the parser module", kind="build")
    W.set_hold(store, pid, tid, "labs_down", "x")
    health = {"v": None}

    async def held(ctx, st, p, proj=None):
        if health["v"] is None:
            return {tid}
        W.clear_holds(st, p, [tid])
        return set()
    monkeypatch.setattr(W, "held_ids", held)
    exe, calls = _fake_build(lab_wait="", ok=True)
    res = await _tick(store, pid, exe=exe)
    assert not calls and res.classification == "idle" and "waiting for the labs host" in res.summary
    health["v"] = HEALTH_ROOM
    res = await _tick(store, pid, exe=exe)
    assert calls and W.holds_of(store.get_project(pid)) == {}


async def test_the_owners_run_takes_a_held_task_and_the_claim_clears_the_hold(store, monkeypatch):
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Build the parser module", kind="build")
    W.set_hold(store, pid, tid, "labs_down", "x")
    monkeypatch.setattr(W, "held_ids", AsyncMock(side_effect=AssertionError("the owner's run never asks")))
    exe, calls = _fake_build(lab_wait="", ok=True)
    await _tick(store, pid, owner=True, exe=exe)
    assert calls and W.holds_of(store.get_project(pid)) == {}


async def test_a_failed_build_without_a_lab_wait_is_still_failed(store):
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Build the parser module", kind="build")
    exe, _ = _fake_build(lab_wait="")
    await _tick(store, pid, exe=exe)
    assert store.get_task(tid)["status"] == "FAILED" and W.holds_of(store.get_project(pid)) == {}


async def test_a_batch_stops_on_a_lab_wait_with_its_own_reason(store, monkeypatch):
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    store.add_task(pid, "Build the parser module", kind="build")
    store.add_task(pid, "Build the lexer module", kind="build")
    exe, calls = _fake_build("labs_down")

    async def runner(name, args):
        return "ok"
    batch = await PA.advance_many(_actx(store), pid, max_tasks=5, tool_runner=runner, coding_executor=exe,
                                  llm_classifier=AsyncMock(return_value="coding"), owner_requested=True)
    assert batch.stop_reason == "labs_down" and len(calls) == 1
    from ghost_agent.tools.projects import _ADVANCE_STOP_BLURB, _advance_batch_instruction
    assert "labs host did not answer" in _ADVANCE_STOP_BLURB["labs_down"]
    assert "no room" in _ADVANCE_STOP_BLURB["lab_capacity"]
    assert "labs host did not answer" in _advance_batch_instruction(batch, 5)


# ------------------------------------------------------------------ the leaf's labs end with it

async def test_destroy_leaf_labs_destroys_only_the_leafs_disposable_labs(tmp_path, monkeypatch):
    from ghost_agent.memory.lab_inventory import inventory_for
    ctx = SimpleNamespace(memory_dir=tmp_path, args=SimpleNamespace(labs_host="h", labs_key_file="k"))
    monkeypatch.setattr(L, "labs_configured", lambda c: True)
    inv = inventory_for(ctx)
    inv.record_create("lab_00000old", owner="project:p", origin="autopilot", template="t", leaf="other-a1")
    inv.record_create("lab_0000new1", owner="project:p", origin="autopilot", template="t", leaf="me-a1")
    inv.record_create("lab_0000new2", owner="project:p", origin="autopilot", template="t", leaf="me-a1")
    inv.record_create("lab_000owner", owner="project:p", origin="owner", template="t", leaf="me-a1")
    inv.record_create("lab_000other", owner="project:q", origin="autopilot", template="t", leaf="me-a1")
    sent = []

    async def _call(c, method, path, **kw):
        sent.append((method, path, kw.get("origin")))
        if path.endswith("new2"):
            return 200, {"destroyed": True, "leftovers": ["vol x"]}
        return 200, {"destroyed": True}
    monkeypatch.setattr(L, "_call", _call)
    gone = await L.destroy_leaf_labs(ctx, "project:p", "me-a1")
    assert gone == ["lab_0000new1"]
    assert sent == [("GET", "/v1/labs", "autopilot"), ("DELETE", "/v1/labs/lab_0000new1", "autopilot"),
                    ("DELETE", "/v1/labs/lab_0000new2", "autopilot")]
    assert inv.get("lab_0000new2")["leftovers"] == "vol x" and inv.get("lab_0000new2")["destroyed_at"] is None
    assert inv.get("lab_00000old")["destroyed_at"] is None


async def test_the_leaf_turn_cleans_its_labs_even_when_it_crashes(monkeypatch):
    from ghost_agent.core import coding_loop as CL
    done = []

    async def fake_destroy(c, owner, leaf):
        done.append((owner, leaf))
        return []
    monkeypatch.setattr(L, "destroy_leaf_labs", fake_destroy)

    class _Boom:
        def __init__(self, iso):
            self.available_tools, self.disabled_tools = {}, set()

        async def handle_chat(self, *a, **k):
            raise RuntimeError("boom")
    import ghost_agent.core.agent as A
    monkeypatch.setattr(A, "GhostAgent", _Boom)
    monkeypatch.setattr(CL, "build_leaf_context", lambda c, **k: SimpleNamespace(args=SimpleNamespace(model="m")))
    with pytest.raises(RuntimeError):
        await CL.run_leaf_turn(SimpleNamespace(), leaf_id="x", prompt="p", is_background=False, project_id="p7")
    assert done == [("project:p7", "x")]


# ------------------------------------------------------------------ scheduled tasks a project owns

class _Sched:
    def __init__(self):
        self.jobs = {}

    def add_job(self, fn, *a, id=None, name=None, **k):
        self.jobs[id] = SimpleNamespace(id=id, name=name, next_run_time=None)

    def get_jobs(self):
        return list(self.jobs.values())

    def get_job(self, jid):
        return self.jobs.get(jid)

    def remove_job(self, jid):
        self.jobs.pop(jid)


@pytest.fixture
def tasks(tmp_path, monkeypatch):
    from ghost_agent.tools import tasks as T
    monkeypatch.setattr(T, "task_store_path", str(tmp_path / "tasks.json"))
    monkeypatch.setattr(T, "run_proactive_task_fn", lambda *a: None)
    monkeypatch.setattr(T, "_schedule_refusal", lambda *a: None)
    T._STORE_STATE["unreadable"] = False
    return T


async def test_a_project_owned_task_is_recorded_listed_and_removed(tasks):
    T = tasks
    s = _Sched()
    out = await T.tool_manage_tasks(action="create", scheduler=s, task_name="nightly", cron_expression="interval:3600",
                                    prompt="check", project_owned=True, _ctx_project_id="p1")
    assert "belongs to project:p1" in str(out)
    await T.tool_manage_tasks(action="create", scheduler=s, task_name="mine", cron_expression="interval:3600",
                              prompt="check")
    got = T.project_schedules("p1")
    assert [g["name"] for g in got] == ["nightly"]
    assert "project:p1" in await T.tool_list_tasks(s)
    assert T.remove_project_schedule(s, got[0]["id"]) and T.project_schedules("p1") == []
    assert got[0]["id"] not in s.jobs and len(s.jobs) == 1


async def test_project_owned_needs_a_project_in_focus(tasks):
    out = await tasks.tool_manage_tasks(action="create", scheduler=_Sched(), task_name="n",
                                        cron_expression="interval:3600", prompt="x", project_owned="true",
                                        _ctx_project_id=None)
    assert "needs a project in focus" in str(out)


def test_the_registry_passes_the_systems_project_id_not_the_models(monkeypatch):
    from ghost_agent.tools import registry as RG
    seen = {}

    async def fake(**kw):
        seen.update(kw)
        return "ok"
    monkeypatch.setattr(RG, "tool_manage_tasks", fake)
    ctx = SimpleNamespace(scheduler=None, memory_system=None, current_project_id="real", llm_client=None,
                          sandbox_manager=None, args=SimpleNamespace())
    import asyncio
    fn = RG.get_available_tools(ctx)["manage_tasks"]
    asyncio.run(fn(action="list", _ctx_project_id="forged"))
    assert seen["_ctx_project_id"] == "real"


def test_the_schema_offers_project_owned():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    d = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "manage_tasks")
    assert d["function"]["parameters"]["properties"]["project_owned"]["type"] == "boolean"


async def test_the_inventory_lists_schedules_and_delete_removes_them_archive_keeps_them(tasks, tmp_path):
    T = tasks
    from ghost_agent.tools import projects as P
    s = _Sched()
    await T.tool_manage_tasks(action="create", scheduler=s, task_name="nightly", cron_expression="interval:3600",
                              prompt="check", project_owned=True, _ctx_project_id="p1")
    ctx = SimpleNamespace(scheduler=s, memory_dir=None, sandbox_manager=None,
                          args=SimpleNamespace(labs_host="", labs_key_file=""))
    inv = P._project_inventory(ctx, "p1")
    assert [e["name"] for e in inv["schedules"]] == ["nightly"]
    assert any(i.startswith("sched:") for i in P._inventory_ids(inv))
    assert "scheduled tasks nightly" in P._inventory_text(inv, hard=True)
    rep = await P._teardown_project(ctx, "p1", hard=False, origin="owner")
    assert any("scheduled task nightly" in k for k in rep["kept_up"]) and T.project_schedules("p1")
    rep = await P._teardown_project(ctx, "p1", hard=True, origin="owner")
    assert any("scheduled task nightly" in k for k in rep["stopped"]) and T.project_schedules("p1") == []
    assert s.jobs == {}


async def test_an_unreadable_task_store_blocks_a_delete(tasks, monkeypatch):
    T = tasks
    from ghost_agent.tools import projects as P

    def boom(pid, strict=False):
        raise RuntimeError("the scheduled-task store could not be read")
    monkeypatch.setattr(T, "project_schedules", boom)
    ctx = SimpleNamespace(scheduler=None, memory_dir=None, sandbox_manager=None,
                          args=SimpleNamespace(labs_host="", labs_key_file=""))
    rep = await P._teardown_project(ctx, "p1", hard=True, origin="owner")
    assert any("scheduled tasks" in x for x in rep["live_left"])
    rep = await P._teardown_project(ctx, "p1", hard=False, origin="owner")
    assert not any("scheduled tasks" in x for x in rep["live_left"])


# ------------------------------------------------------------------ review r2

def _lctx(tmp_path, pid="p1"):
    kf = tmp_path / "k"
    kf.write_text("k" * 64)
    (tmp_path / "sb").mkdir(exist_ok=True)
    return SimpleNamespace(args=SimpleNamespace(labs_host="127.0.0.1:8700", labs_key_file=str(kf)),
                           sandbox_dir=tmp_path / "sb", memory_dir=tmp_path, tor_proxy=None,
                           current_project_id=pid, conversation_key="", sandbox_manager=None)


@pytest.fixture
def lfake(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_4sx_b_lab_tool import FakeLabs
    f = FakeLabs()
    monkeypatch.setattr(L, "_call", f)
    monkeypatch.setattr(L, "_time_left_s", lambda: None)
    return f


def _scoped(ctx, origin):
    from ghost_agent.core import request_scope as RS
    return RS.install(RS.RequestScope(conversation_key="", project_id=ctx.current_project_id, origin=origin,
                                      context=ctx))


async def test_unattended_work_touches_only_its_own_projects_labs(tmp_path, lfake):
    """r2 #1."""
    from ghost_agent.core import request_scope as RS
    ctx = _lctx(tmp_path)
    tok = _scoped(ctx, "leaf")
    try:
        lfake.on("GET", "/v1/labs/lab_0000aa01", (200, {"lab_id": "lab_0000aa01", "owner": "conversation:c"}))
        out = await L.tool_lab(ctx, action="exec", lab_id="lab_0000aa01", member="m", cmd="cat /etc/x")
        assert "belongs to conversation:c" in str(out)
        assert not [c for c in lfake.calls if c.method == "POST"]
        out = await L.tool_lab(ctx, action="list", all_owners=True)
        assert "only its own labs" in str(out)
        from ghost_agent.memory.lab_inventory import inventory_for
        inventory_for(ctx).record_create("lab_0000aa02", owner="project:q", origin="owner", template="t")
        lfake.on("GET", "/v1/labs/lab_0000aa02", (404, {"error": "unknown_lab"}))
        out = await L.tool_lab(ctx, action="destroy", lab_id="lab_0000aa02")
        assert "belongs to project:q" in str(out) and not [c for c in lfake.calls if c.method == "DELETE"]
        lfake.on("GET", "/v1/labs/lab_0000aa03", (200, {"lab_id": "lab_0000aa03", "owner": "project:p1",
                                                        "state": "ready", "members": []}))
        out = await L.tool_lab(ctx, action="status", lab_id="lab_0000aa03")
        assert "belongs to" not in str(out)
        out = await L.tool_lab(ctx, action="keep", lab_id="lab_0000aa03", params={})
        assert "not available to unattended work" in str(out)
    finally:
        RS.reset(tok)


async def test_too_little_turn_time_is_an_outcome_never_a_crash(tmp_path, lfake, monkeypatch):
    """r2 #9 (backup was not in the up-front list; the escape path itself too)."""
    from ghost_agent.core import request_scope as RS
    ctx = _lctx(tmp_path)
    tok = _scoped(ctx, "owner")
    try:
        monkeypatch.setattr(L, "_time_left_s", lambda: 40.0)
        out = await L.tool_lab(ctx, action="backup", lab_id="lab_0000aa03")
        assert "no_time" in str(out) and lfake.calls == []

        async def boom(*a, **k):
            raise L._NoTurnTime()
        monkeypatch.setattr(L, "_time_left_s", lambda: None)
        monkeypatch.setattr(L, "_long_call", boom)
        out = await L.tool_lab(ctx, action="rehearse", lab_id="lab_0000aa03")
        assert "no_time" in str(out)
    finally:
        RS.reset(tok)


async def test_a_probe_keeps_only_a_lab_of_its_own_probe_project(tmp_path, lfake):
    """r2 #10."""
    from ghost_agent.core import request_scope as RS
    ctx = _lctx(tmp_path, pid="pprobe")
    ctx.project_store = SimpleNamespace(get_project=lambda pid: {"metadata": {"probe_created": True}})
    tok = _scoped(ctx, "probe")
    try:
        lfake.on("GET", "/v1/labs/lab_0000aa04", (200, {"lab_id": "lab_0000aa04", "owner": "project:other",
                                                        "lifetime": "kept", "app": {"name": "a"}}))
        out = await L.tool_lab(ctx, action="keep", lab_id="lab_0000aa04",
                               params={"app": {"name": "a", "member": "m", "port": 80}})
        assert "probe project in focus" in str(out) and not [c for c in lfake.calls if c.method == "POST"]
    finally:
        RS.reset(tok)


async def test_a_cancel_during_the_leaf_cleanup_is_raised(monkeypatch):
    """r2 #11."""
    import asyncio
    from ghost_agent.core import coding_loop as CL

    async def slow(c, owner, leaf):
        await asyncio.sleep(10)
    monkeypatch.setattr(L, "destroy_leaf_labs", slow)

    class _Ok:
        def __init__(self, iso):
            self.available_tools, self.disabled_tools = {}, set()

        async def handle_chat(self, *a, **k):
            return "done", None, None
    import ghost_agent.core.agent as A
    monkeypatch.setattr(A, "GhostAgent", _Ok)
    monkeypatch.setattr(CL, "build_leaf_context", lambda c, **k: SimpleNamespace(args=SimpleNamespace(model="m")))
    t = asyncio.ensure_future(CL.run_leaf_turn(SimpleNamespace(), leaf_id="x", prompt="p", is_background=False,
                                               project_id="p7"))
    await asyncio.sleep(0.05)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t


async def test_the_build_key_and_leaf_id_are_set_for_the_leaf_turn(monkeypatch):
    from ghost_agent.core import coding_loop as CL
    seen = {}

    class _Ok:
        def __init__(self, iso):
            self.available_tools, self.disabled_tools = {}, set()

        async def handle_chat(self, *a, **k):
            seen.update(build=W.BUILD_KEY.get(), leaf=W.LEAF_ID.get())
            return "done", None, None
    import ghost_agent.core.agent as A
    monkeypatch.setattr(A, "GhostAgent", _Ok)
    monkeypatch.setattr(CL, "build_leaf_context", lambda c, **k: SimpleNamespace(args=SimpleNamespace(model="m")))
    await CL.run_leaf_turn(SimpleNamespace(), leaf_id="abcd1234-a2", prompt="p", is_background=False)
    assert seen == {"build": "abcd1234", "leaf": "abcd1234-a2"} and W.BUILD_KEY.get() == ""


async def test_a_task_that_waited_max_rounds_goes_to_the_owner(store):
    """r2 #6."""
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Build the parser module", kind="build")
    for _ in range(W.MAX_ROUNDS):
        W.set_hold(store, pid, tid, "lab_capacity", "x")
    W.clear_holds(store, pid, [tid])
    exe, _ = _fake_build("lab_capacity")
    res = await _tick(store, pid, exe=exe)
    t = store.get_task(tid)
    assert t["status"] == "NEEDS_USER" and t["wait_on"] == "owner" and t["wait_code"] == "lab_capacity"
    assert res.summary.startswith(PA.LAB_GAVE_UP_PREFIX) and W.holds_of(store.get_project(pid)) == {}
    assert not res.summary.startswith(PA.LAB_WAIT_PREFIX)
    W.clear_rounds(store, pid, tid)
    assert W.rounds_of(store, pid, tid) == 0


async def test_the_hold_is_written_before_the_claim_is_released(store, monkeypatch):
    """r2 #8."""
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Build the parser module", kind="build")
    order = []
    real_rel = store.release_claim
    monkeypatch.setattr(store, "release_claim", lambda t, **k: order.append("release") or real_rel(t, **k))
    real_set = W.set_hold
    monkeypatch.setattr(W, "set_hold", lambda *a, **k: order.append("hold") or real_set(*a, **k))
    exe, _ = _fake_build("labs_down")
    await _tick(store, pid, exe=exe)
    assert order == ["hold", "release"]


async def test_a_lab_wait_is_not_scored_as_no_progress(store, monkeypatch):
    """r2 #6: waiting for the labs host does not pause the autopilot."""
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    store.add_task(pid, "Build the parser module", kind="build")
    res = PA.AdvanceResult(True, "t", "coding", f"{PA.LAB_WAIT_PREFIX} [labs_down]: x", None)
    monkeypatch.setattr(PA, "advance_once", AsyncMock(return_value=res))
    monkeypatch.setattr(PA, "_step_score", lambda *a: (_ for _ in ()).throw(AssertionError("scored")))
    store.update_project(pid, metadata={"autopilot": True})
    out = await PA.advance_unattended(_actx(store), pid)
    assert out is res
    assert int((store.get_project(pid)["metadata"] or {}).get("no_progress_streak") or 0) == 0


def test_the_holds_are_system_metadata():
    """r2 #13."""
    from ghost_agent.tools.projects import _SYSTEM_METADATA_KEYS
    assert "lab_waits" in _SYSTEM_METADATA_KEYS and "lab_waits_rounds" in _SYSTEM_METADATA_KEYS


async def test_a_refused_delete_keeps_the_projects_schedules(tasks, monkeypatch):
    """r2 #4."""
    T = tasks
    from ghost_agent.tools import projects as P
    s = _Sched()
    await T.tool_manage_tasks(action="create", scheduler=s, task_name="nightly", cron_expression="interval:3600",
                              prompt="check", project_owned=True, _ctx_project_id="p1")

    async def labs_left(context, project_id, owner, *, hard, origin, rep):
        rep["live_left"].append("lab lab_x (the labs host did not answer)")
    monkeypatch.setattr(P, "_teardown_labs", labs_left)
    ctx = SimpleNamespace(scheduler=s, memory_dir=None, sandbox_manager=None,
                          args=SimpleNamespace(labs_host="", labs_key_file=""))
    rep = await P._teardown_project(ctx, "p1", hard=True, origin="owner")
    assert T.project_schedules("p1") and any("kept — the delete is refused" in n for n in rep["notes"])


def test_a_new_schedule_since_the_preview_asks_again(monkeypatch):
    """r2 #12."""
    from ghost_agent.tools import projects as P
    monkeypatch.setattr(P, "_labs_host_configured", lambda c: False)
    monkeypatch.setattr(P, "_project_inventory", lambda c, rid, hl=None: {
        "services": [], "runs": [], "labs": [], "jobs": [], "schedules": [{"id": "task_1", "name": "n"}]})
    assert P._grew_since_preview(SimpleNamespace(), "delete", "p1", {"inventory": []}) == ["sched:task_1"]


# ------------------------------------------------------------------ review r4 + battery survivors

async def test_a_give_up_stops_the_batch_as_waiting_for_the_owner(store):
    """r4 #1: a give-up never says the task starts again by itself."""
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Build the parser module", kind="build")
    for _ in range(W.MAX_ROUNDS):
        W.set_hold(store, pid, tid, "labs_down", "x")
    W.clear_holds(store, pid, [tid])
    exe, _ = _fake_build("labs_down")

    async def runner(name, args):
        return "ok"
    batch = await PA.advance_many(_actx(store), pid, max_tasks=5, tool_runner=runner, coding_executor=exe,
                                  llm_classifier=AsyncMock(return_value="coding"), owner_requested=True)
    assert batch.stop_reason == "lab_gave_up"
    from ghost_agent.tools.projects import _ADVANCE_STOP_BLURB, _advance_batch_instruction
    assert "WAITS FOR THE USER" in _ADVANCE_STOP_BLURB["lab_gave_up"]
    txt = _advance_batch_instruction(batch, 5)
    assert "WAITS FOR THE USER" in txt and "starts it again by itself" not in txt


async def test_a_create_the_host_never_answered_is_noted_for_the_build(tmp_path, lfake):
    """r4 test gap + battery L4: through the real tool, under the build's key."""
    ctx = _lctx(tmp_path)
    tok = _scoped(ctx, "leaf")
    k = W.BUILD_KEY.set("bld-na")
    try:
        lfake.on("POST", "/v1/labs", L.LabsNoAnswer("ReadTimeout"))
        out = await L.tool_lab(ctx, action="create", template="base")
        assert out.reason_code == "lab_no_answer"
        assert W.take_seen("bld-na") == ("labs_no_answer", 0)
        lfake.on("POST", "/v1/labs", (409, {"error": "lab_limit", "message": "4 labs"}))
        await L.tool_lab(ctx, action="create", template="base")
        assert W.take_seen("bld-na")[0] == "lab_capacity"
    finally:
        W.BUILD_KEY.reset(k)
        from ghost_agent.core import request_scope as RS
        RS.reset(tok)


async def test_a_leafs_create_is_tagged_and_idles_out_and_its_untracked_lab_is_destroyed(tmp_path, lfake):
    """r4 #3: a create the host made but never answered has no inventory row — the host tag finds it."""
    ctx = _lctx(tmp_path)
    tok = _scoped(ctx, "leaf")
    k = W.LEAF_ID.set("t1-a1")
    try:
        lfake.on("POST", "/v1/labs", L.LabsNoAnswer("ReadTimeout"))
        await L.tool_lab(ctx, action="create", template="base", purpose="db for tests")
        body = [c for c in lfake.calls if c.method == "POST"][0].json
        tag = L.leaf_tag("t1-a1")
        assert body["purpose"] == f"db for tests {tag}" and body["ttl_idle_s"] == L.LEAF_LAB_IDLE_S
    finally:
        W.LEAF_ID.reset(k)
        from ghost_agent.core import request_scope as RS
        RS.reset(tok)
    lfake.on("GET", "/v1/labs", (200, {"labs": [
        {"lab_id": "lab_0000un01", "owner": "project:p1", "origin": "autopilot", "purpose": f"db for tests {tag}"},
        {"lab_id": "lab_0000un02", "owner": "project:p1", "origin": "autopilot", "purpose": f"x {L.leaf_tag('t2-a1')}"},
        {"lab_id": "lab_0000un03", "owner": "project:p1", "origin": "autopilot", "purpose": f"{tag}",
         "lifetime": "kept"},
        {"lab_id": "lab_0000un04", "owner": "project:p1", "origin": "autopilot", "purpose": "no tag"}]}))
    lfake.on("DELETE", "/v1/labs/lab_0000un01", (200, {"destroyed": True}))
    assert await L.destroy_leaf_labs(ctx, "project:p1", "t1-a1") == ["lab_0000un01"]
    assert [c.path for c in lfake.calls if c.method == "DELETE"] == ["/v1/labs/lab_0000un01"]
    k2 = W.LEAF_ID.set("")
    try:                                                             # the owner's create carries no tag
        tok = _scoped(ctx, "owner")
        lfake.calls.clear()
        lfake.on("POST", "/v1/labs", L.LabsNoAnswer("ReadTimeout"))
        await L.tool_lab(ctx, action="create", template="base", purpose="mine")
        assert [c for c in lfake.calls if c.method == "POST"][0].json["purpose"] == "mine"
    finally:
        W.LEAF_ID.reset(k2)
        from ghost_agent.core import request_scope as RS
        RS.reset(tok)


async def test_a_failed_agentic_build_carries_the_lab_wait_its_leaves_hit(tmp_path, monkeypatch):
    """battery C2: the build's failure names the wait one of ITS leaves got."""
    from ghost_agent.core import coding_loop as CL

    async def fake_turn(context, *, leaf_id, prompt, is_background, **kw):
        W.note_outcome(str(leaf_id).rsplit("-a", 1)[0], "lab_lab_limit", False, action="create",
                       text="LAB ERROR\nneeded_mb: 2048")
        return "SUMMARY: could not get a lab"
    monkeypatch.setattr(CL, "run_leaf_turn", fake_turn)

    class Store:
        def get_project(self, pid):
            return {"id": pid, "workspace_dir": str(tmp_path), "metadata": {}}
    ctx = SimpleNamespace(current_project_id="p1", project_store=Store(), llm_client=object(),
                          args=SimpleNamespace(model="m"))

    async def runner(name, args):
        return "EXIT CODE: 1"
    res = await CL.build_coding_task_agentic(ctx, "build x", tool_runner=runner, max_attempts=1)
    assert res.ok is False and res.lab_wait == "lab_capacity" and res.lab_needed_mb == 2048


async def test_a_build_that_got_its_lab_starts_the_wait_count_over(store):
    """battery A8."""
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Build the parser module", kind="build")
    W.set_hold(store, pid, tid, "labs_down", "x")
    W.set_hold(store, pid, tid, "labs_down", "x")
    W.clear_holds(store, pid, [tid])
    assert W.rounds_of(store, pid, tid) == 2
    exe, _ = _fake_build("", ok=True)
    await _tick(store, pid, exe=exe)
    assert W.rounds_of(store, pid, tid) == 0


# ------------------------------------------------------------------ review r5

async def test_a_long_purpose_keeps_its_leaf_tag_within_the_hosts_cut(tmp_path, lfake):
    """r5: the host cuts purpose at 120 chars — the tag must survive it."""
    ctx = _lctx(tmp_path)
    tok = _scoped(ctx, "leaf")
    k = W.LEAF_ID.set("t9-a1")
    try:
        lfake.on("POST", "/v1/labs", L.LabsNoAnswer("ReadTimeout"))
        await L.tool_lab(ctx, action="create", template="base", purpose="p" * 300)
        sent = [c for c in lfake.calls if c.method == "POST"][0].json["purpose"]
        assert sent[:120].endswith(L.leaf_tag("t9-a1"))
    finally:
        W.LEAF_ID.reset(k)
        from ghost_agent.core import request_scope as RS
        RS.reset(tok)


async def test_the_leaf_sweep_never_takes_an_owner_made_lab_with_a_copied_tag(tmp_path, lfake):
    """r5 #3."""
    ctx = _lctx(tmp_path)
    tag = L.leaf_tag("t3-a1")
    lfake.on("GET", "/v1/labs", (200, {"labs": [
        {"lab_id": "lab_0000ow01", "owner": "project:p1", "origin": "owner", "purpose": f"copy {tag}"},
        {"lab_id": "lab_0000ap01", "owner": "project:p1", "origin": "autopilot", "purpose": f"mine {tag}"}]}))
    lfake.on("DELETE", "/v1/labs/lab_0000ap01", (200, {"destroyed": True}))
    assert await L.destroy_leaf_labs(ctx, "project:p1", "t3-a1") == ["lab_0000ap01"]
    assert [c.path for c in lfake.calls if c.method == "DELETE"] == ["/v1/labs/lab_0000ap01"]


async def test_a_leaf_labs_recipe_is_the_owners_without_the_tag_and_short_ttl(tmp_path, lfake):
    """r5 #4."""
    ctx = _lctx(tmp_path)
    tok = _scoped(ctx, "owner")
    try:
        tag = L.leaf_tag("t4-a1")
        lfake.on("GET", "/v1/labs/lab_00005b01/spec", (200, {"lab_id": "lab_00005b01", "create": {
            "template": "base", "purpose": f"db {tag}", "ttl_idle_s": L.LEAF_LAB_IDLE_S}, "keep": None}))
        out = str(await L.tool_lab(ctx, action="spec", lab_id="lab_00005b01"))
        assert "#leaf-" not in out and "ttl_idle_s" not in out and '"purpose": "db"' in out
    finally:
        from ghost_agent.core import request_scope as RS
        RS.reset(tok)


async def test_a_give_up_starts_the_wait_count_over_for_the_owners_retry(store):
    """r5 #1."""
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Build the parser module", kind="build")
    for _ in range(W.MAX_ROUNDS):
        W.set_hold(store, pid, tid, "labs_down", "x")
    W.clear_holds(store, pid, [tid])
    exe, _ = _fake_build("labs_down")
    res = await _tick(store, pid, exe=exe)
    assert res.summary.startswith(PA.LAB_GAVE_UP_PREFIX)
    assert W.rounds_of(store, pid, tid) == 0


# ------------------------------------------------------------------ review r6

async def test_a_host_ttl_class_shorter_than_the_leaf_default_drops_the_systems_ttl(tmp_path, lfake):
    ctx = _lctx(tmp_path)
    tok = _scoped(ctx, "leaf")
    k = W.LEAF_ID.set("t6-a1")
    try:
        lfake.on("POST", "/v1/labs", (400, {"error": "bad_request", "field": "ttl_idle_s",
                                             "message": "ttl_idle_s must be between 1 and 3600"}),
                 L.LabsNoAnswer("ReadTimeout"))
        await L.tool_lab(ctx, action="create", template="base")
        posts = [c.json for c in lfake.calls if c.method == "POST"]
        assert len(posts) == 2 and "ttl_idle_s" not in posts[1]
        lfake.calls.clear()                                          # the MODEL's own TTL is never dropped
        lfake.on("POST", "/v1/labs", (400, {"error": "bad_request", "field": "ttl_idle_s", "message": "x"}))
        out = await L.tool_lab(ctx, action="create", template="base", ttl_idle_s=99999)
        assert len([c for c in lfake.calls if c.method == "POST"]) == 1 and "LAB ERROR" in str(out)
    finally:
        W.LEAF_ID.reset(k)
        from ghost_agent.core import request_scope as RS
        RS.reset(tok)


async def test_an_untracked_lab_already_gone_counts_and_a_refused_one_is_logged(tmp_path, lfake, caplog):
    import logging
    ctx = _lctx(tmp_path)
    tag = L.leaf_tag("t7-a1")
    lfake.on("GET", "/v1/labs", (200, {"labs": [
        {"lab_id": "lab_0000aa71", "owner": "project:p1", "origin": "autopilot", "purpose": f"a {tag}"},
        {"lab_id": "lab_0000aa72", "owner": "project:p1", "origin": "autopilot", "purpose": f"b {tag}"}]}))
    lfake.on("DELETE", "/v1/labs/lab_0000aa71", (404, {"error": "unknown_lab"}))
    lfake.on("DELETE", "/v1/labs/lab_0000aa72", (409, {"error": "busy"}))
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        assert await L.destroy_leaf_labs(ctx, "project:p1", "t7-a1") == ["lab_0000aa71"]
    assert any("lab_0000aa72" in r.getMessage() and "untracked" in r.getMessage() for r in caplog.records)


async def test_the_owners_retry_after_a_give_up_gets_a_hold_again(store):
    """r5 #1 / r6: the retry waits again — it does not give up at once."""
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Build the parser module", kind="build")
    for _ in range(W.MAX_ROUNDS):
        W.set_hold(store, pid, tid, "labs_down", "x")
    W.clear_holds(store, pid, [tid])
    exe, _ = _fake_build("labs_down")
    assert (await _tick(store, pid, exe=exe)).summary.startswith(PA.LAB_GAVE_UP_PREFIX)
    store.update_task(tid, status="PENDING", wait_on=None, wait_code=None)
    res = await _tick(store, pid, exe=exe, owner=True)
    assert res.summary.startswith(PA.LAB_WAIT_PREFIX) and W.rounds_of(store, pid, tid) == 1


async def test_a_build_that_failed_for_another_reason_starts_the_wait_count_over(store):
    """r6: rounds counted before a build that got past the labs host do not carry into a retry."""
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Build the parser module", kind="build")
    W.set_hold(store, pid, tid, "labs_down", "x")
    W.set_hold(store, pid, tid, "labs_down", "x")
    W.clear_holds(store, pid, [tid])
    exe, _ = _fake_build("", ok=False)
    await _tick(store, pid, exe=exe)
    assert W.rounds_of(store, pid, tid) == 0
