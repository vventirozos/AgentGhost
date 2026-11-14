"""§4UL review (projects / task turns) — the fixes, each pinned by behaviour:

M1  a FAILED task does not hold the plan; autopilot pauses (once) only when every open task waits on
    failed work — through depends_on, a BLOCKED ancestor, or plan order behind a sibling blocked by it.
M2/M7  unattended, a task turn's DONE with no check after its last write is refused INSIDE the turn
    (nothing rolls up) and the task is held for the owner on every exit path.
M3  idle autopilot's task turn delivers the project-done promise from the hook.
M4  a task turn cannot edit the project, its constraints or other tasks; the owner's run may state a
    constraint exception like the owner's own turn.
M5  a task turn's DONE sweep waits for the request it runs in, and that request's later write is not a change.
M6  an execute command is a project write only for the paths it WRITES.
m1  only idle autopilot's turns take the background lane.
m2  a deferral does not spend the daily allowance.
m4  `ls` / `cat` is not a check.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import project_turns as PT
from ghost_agent.core.planning import ProjectPlan, TaskStatus, soft_order_blocked, soft_order_stoppers
from ghost_agent.memory import projects as MP
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome
from ghost_agent.utils.logging import request_id_context
from tests.test_4ul_autopilot_turns import _advance, _ctx, _executor, _fake_turn, _project


@pytest.fixture
def turns_on(monkeypatch):
    monkeypatch.setenv("GHOST_PROJECT_TASK_TURNS", "1")


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def _autopilot(store, pid):
    m = dict(store.get_project(pid).get("metadata") or {})
    m["autopilot"] = True
    store.update_project(pid, metadata=m)


# ── M1: plan order and the stuck pause ──────────────────────────────────────

def test_a_failed_task_does_not_hold_the_tasks_after_it(store):
    pid = store.create_project("P", kind="CODING", goal="g")
    a = store.add_task(pid, "Build a → a.py")
    b = store.add_task(pid, "Build b → b.py")
    store.update_task(a, status="FAILED", failure_reason="boom")
    assert b not in soft_order_blocked(ProjectPlan(store, pid))
    store.update_task(a, status="NEEDS_USER")                     # a wait still holds it
    assert soft_order_stoppers(ProjectPlan(store, pid)) == {b: {a}}


def _stuck_project(store, held_status="FAILED"):
    """A BLOCKED parent (its child failed) before a plain later task: plan order holds the later one."""
    pid = store.create_project("P", kind="CODING", goal="g")
    parent = store.add_task(pid, "Data layer")
    child = store.add_task(pid, "Schema → schema.sql", parent_id=parent)
    later = store.add_task(pid, "API → api.py")
    store.update_task(child, status=held_status, failure_reason="boom")
    store.update_task(parent, status="BLOCKED")
    return pid, child, later


def test_every_open_task_behind_failed_work_pauses_autopilot_once(store):
    pid, child, later = _stuck_project(store)
    assert later in soft_order_blocked(ProjectPlan(store, pid))
    _autopilot(store, pid)
    PA_ = __import__("ghost_agent.core.project_advancer", fromlist=["_pause_if_stuck"])
    ctx = _ctx(store)
    PA_._pause_if_stuck(ctx, store, pid)
    meta = store.get_project(pid)["metadata"]
    assert meta["autopilot"] is False and meta["autopilot_paused"]["reason"] == "stuck"
    assert child in meta["autopilot_paused"]["detail"]
    PA_._pause_if_stuck(ctx, store, pid)                           # once: no second event
    assert len(store.list_events(pid, event_type="autopilot_paused")) == 1


def test_a_task_behind_the_owner_is_not_stuck(store):
    pid = store.create_project("P", kind="CODING", goal="g")
    a = store.add_task(pid, "Build a → a.py")
    b = store.add_task(pid, "Pick the colours")
    c = store.add_task(pid, "Build c → c.py")
    d = store.add_task(pid, "Build d → d.py")
    store.update_task(c, depends_on=[a])                           # c waits on the failed task …
    store.update_task(a, status="FAILED", failure_reason="boom")
    store.update_task(b, status="NEEDS_USER")                      # … but d waits (plan order) on the owner
    _autopilot(store, pid)
    from ghost_agent.core.project_advancer import _pause_if_stuck
    _pause_if_stuck(_ctx(store), store, pid)
    assert store.get_project(pid)["metadata"]["autopilot"] is True
    store.update_task(d, status="DONE", result_summary="built")    # now only c is open: stuck on failed work
    _pause_if_stuck(_ctx(store), store, pid)
    assert store.get_project(pid)["metadata"]["autopilot"] is False


async def test_the_unattended_tick_runs_past_a_failed_task(turns_on, monkeypatch, store):
    pid = store.create_project("P", kind="CODING", goal="g")
    a = store.add_task(pid, "Build a → a.py")
    b = store.add_task(pid, "Build b → b.py")
    store.update_task(a, status="FAILED", failure_reason="boom")
    seen = []
    _fake_turn(monkeypatch, lambda ag, kw: "nothing", seen)
    res = await _advance(_ctx(store), pid, owner=False)
    assert res.task_id == b and seen


# ── M2 / M7: the unattended DONE gate ───────────────────────────────────────

def _closing_script(store, tid, *, check=False, then=None):
    """Plays the model: write, maybe a check, then DONE through the WRAPPED tool (committed only when the
    wrapper lets it through), then ``then`` (an exception to raise, or a reply)."""
    async def go(agent, kw):
        await agent.available_tools["file_system"](operation="write", path="a.py", content="x")
        if check:
            await agent.available_tools["execute"](command="python a.py")
        out = await agent.available_tools["manage_projects"](action="task_update", task_id=tid, status="DONE",
                                                             result="built a.py")
        if getattr(out, "status", None) is OutcomeStatus.OK:
            store.update_task(tid, status="DONE", result_summary="built a.py")
        kw["_out"] = out
        if isinstance(then, BaseException):
            raise then
        return then or "done"
    return go


async def test_an_unverified_unattended_done_is_refused_in_the_turn_and_never_rolls_up(turns_on, monkeypatch,
                                                                                       store):
    pid, tid = _project(store, "Build a → a.py")
    seen = []
    _fake_turn(monkeypatch, _closing_script(store, tid), seen)
    res = await _advance(_ctx(store), pid, owner=False)
    out = seen[0]["_out"]
    assert out.status is OutcomeStatus.REJECTED and out.reason_code == "task_turn_unverified_done"
    row = store.get_task(tid)
    assert row["status"] == "NEEDS_USER" and row["wait_code"] == "unverified_build"
    assert "built a.py" in (row.get("result_summary") or "")
    assert res.classification == "needs_user"
    # the only task — had the DONE committed, the project would have rolled up DONE (sweep, retired constraints)
    assert store.get_project(pid)["status"] != "DONE"


async def test_a_checked_unattended_done_closes(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build a → a.py")
    seen = []
    _fake_turn(monkeypatch, _closing_script(store, tid, check=True), seen)
    await _advance(_ctx(store), pid, owner=False)
    assert seen[0]["_out"].status is OutcomeStatus.OK
    # §4UM: the finished plan leaves the project open ("ready") — only the owner closes it
    assert store.get_task(tid)["status"] == "DONE" and store.project_state(pid) == "ready"


@pytest.mark.parametrize("then", ["timeout", "deferred", "crash", "[ATTEMPT_ABORTED_UPSTREAM] owner"])
async def test_an_unverified_done_is_held_on_every_exit_path(turns_on, monkeypatch, store, then):
    from ghost_agent.core.llm import BackgroundDeferred
    boom = {"timeout": asyncio.TimeoutError(), "deferred": BackgroundDeferred("owner"),
            "crash": RuntimeError("bad")}.get(then, then)
    pid, tid = _project(store, "Build a → a.py")
    _fake_turn(monkeypatch, _closing_script(store, tid, then=boom), [])
    res = await _advance(_ctx(store), pid, owner=False)
    row = store.get_task(tid)
    assert row["status"] == "NEEDS_USER" and row["wait_code"] == "unverified_build", then
    assert res.classification == "needs_user"


async def test_the_owners_run_closes_without_the_gate(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build a → a.py")
    seen = []
    _fake_turn(monkeypatch, _closing_script(store, tid), seen)
    await _advance(_ctx(store), pid, owner=True)
    assert seen[0]["_out"].status is OutcomeStatus.OK and store.get_task(tid)["status"] == "DONE"


def test_a_look_only_command_is_not_a_check():
    w = PT._TurnWatch()
    w.note("file_system", {"operation": "write"}, ToolOutcome.ok("w"))
    for cmd in ("ls -la", "cat a.py | head -5", "grep -n x a.py | wc -l", "cd projects/x && ls"):
        w.note("execute", {"command": cmd}, ToolOutcome.ok("EXIT CODE: 0"))
        assert not w.verified, cmd
    w.note("lab", {"action": "exec", "command": "ls /srv"}, ToolOutcome.ok("EXIT CODE: 0"))
    assert not w.verified
    w.note("execute", {"command": "cd app && python -m pytest -q"}, ToolOutcome.ok("EXIT CODE: 0"))
    assert w.verified
    w.note("file_system", {"operation": "edit"}, ToolOutcome.ok("e"))
    w.note("execute", {"filename": "check.py", "content": "print(open('a.py').read())"},
           ToolOutcome.ok("EXIT CODE: 0"))
    assert w.verified                                              # a script it wrote and ran
    w.note("file_system", {"operation": "edit"}, ToolOutcome.ok("e"))
    w.note("browser", {"action": "navigate", "url": "http://localhost:8080"}, ToolOutcome.ok("page"))
    assert w.verified


# ── M4: the turn's scope ────────────────────────────────────────────────────

async def test_a_task_turn_touches_only_its_task_and_reads_other_projects(store):
    from tests.test_4ul_autopilot_turns import _FakeTools
    pid, tid = _project(store)
    other = store.add_task(pid, "Another task")
    sub = store.add_task(pid, "A subtask it made", parent_id=tid)
    calls = []
    agent = _FakeTools(calls)
    PT._prepare(PT._TurnWatch(pid, tid, store=store))(agent)
    mp = agent.available_tools["manage_projects"]
    for act in ("update", "constraint_retire", "clone", "cleanup", "set_dependency"):
        assert (await mp(action=act)).reason_code == "task_turn_scope", act
    assert (await mp(action="task_update", task_id=other, status="DONE")).reason_code == "task_turn_scope"
    assert (await mp(action="task_update", task_ids=[tid, other], status="DONE")).reason_code == "task_turn_scope"
    assert (await mp(action="task_add", project_id="someone-else", description="x")).reason_code == \
        "task_turn_scope"
    assert calls == []
    for kw in ({"action": "task_update", "task_id": tid, "result": "notes"},
               {"action": "task_update", "task_id": sub, "status": "FAILED"},
               {"action": "get", "project_id": "someone-else"},
               {"action": "task_add", "description": "follow-up"}):
        assert (await mp(**kw)).status is OutcomeStatus.OK, kw
    assert len(calls) == 4


def test_the_owners_task_turn_carries_the_owners_authority(store):
    from ghost_agent.core.request_scope import ORIGIN_OWNER, ORIGIN_UNATTENDED, RequestScope, install, reset
    from ghost_agent.tools.projects import _owner_task_turn, _owner_turn_now

    def check(rid, parent, origin):
        if parent:
            MP.set_request_parent(rid, parent)
        tok, stok = request_id_context.set(rid), install(RequestScope(origin=origin))
        try:
            return _owner_task_turn(None), _owner_turn_now()
        finally:
            reset(stok)
            request_id_context.reset(tok)
    assert check("sub-ptask-aaa-1", "a1b2c3d4", ORIGIN_OWNER) == (True, False)
    assert check("sub-ptask-aaa-2", "a1b2c3d4", ORIGIN_UNATTENDED) == (False, False)
    assert check("sub-ptask-aaa-3", "sched-nightly-1", ORIGIN_OWNER) == (False, False)
    assert check("sub-ptask-aaa-4", "", ORIGIN_OWNER) == (False, False)          # idle: no owner around it
    assert check("sub-leaf-aaa-5", "a1b2c3d4", ORIGIN_OWNER) == (False, False)


# ── M3 / M5: the request a task turn runs in ────────────────────────────────

def test_the_project_done_promise_is_delivered_by_the_hook_only_with_no_request_around():
    assert PT.project_done_hook_delivers("SYSTEM") and PT.project_done_hook_delivers("")
    assert PT.project_done_hook_delivers("sub-ptask-idle-1")                     # idle autopilot's turn
    MP.set_request_parent("sub-ptask-owner-1", "req-owner-9")
    assert not PT.project_done_hook_delivers("sub-ptask-owner-1")                # the owner's request delivers
    assert not PT.project_done_hook_delivers("req-owner-9")


def test_a_task_turns_done_sweep_waits_for_the_request_it_runs_in(store):
    swept = []
    store.on_project_done = swept.append
    pid = store.create_project("P")
    t = store.add_task(pid, "only")
    MP.set_request_parent("sub-ptask-t-1", "req-outer-1")
    tok = request_id_context.set("sub-ptask-t-1")
    try:
        store.update_task(t, status="DONE", result_summary="done")
        assert MP.flush_deferred_sweeps("sub-ptask-t-1") == 0 and swept == []    # the turn's own end
    finally:
        request_id_context.reset(tok)
    assert MP.flush_deferred_sweeps("req-outer-1") == 1 and swept == [pid]


def test_idle_autopilots_turn_sweeps_at_its_own_end(store):
    swept = []
    store.on_project_done = swept.append
    pid = store.create_project("P")
    t = store.add_task(pid, "only")
    tok = request_id_context.set("sub-ptask-idle-2")
    try:
        store.update_task(t, status="DONE", result_summary="done")
        assert MP.flush_deferred_sweeps("sub-ptask-idle-2") == 1
    finally:
        request_id_context.reset(tok)
    assert swept == [pid]


def test_a_live_request_is_never_swept_as_stale():
    assert MP.DEFERRED_SWEEP_MAX_AGE_S >= 3 * 3600                  # past the 180-min request deadline


async def test_a_write_after_the_owners_task_turn_closed_the_project_is_the_same_work(tmp_path, monkeypatch,
                                                                                       store):
    from ghost_agent.tools.projects import note_project_write
    ctx = SimpleNamespace(project_store=store, current_project_id=None, last_user_content="build it",
                          sandbox_dir=tmp_path / "sb")
    pid = store.create_project("Site")
    t = store.add_task(pid, "Build it")
    MP.set_request_parent("sub-ptask-site-1", "req-owner-site")
    tok = request_id_context.set("sub-ptask-site-1")
    try:
        store.update_task(t, status="DONE", result_summary="built")
    finally:
        request_id_context.reset(tok)
    tok = request_id_context.set("req-owner-site")
    try:
        (Path(store.ensure_workspace(pid)) / "NOTES.md").write_text("notes")
        assert note_project_write(ctx, Path(store.ensure_workspace(pid)) / "NOTES.md") == ""
    finally:
        request_id_context.reset(tok)
    # §4UM: the plan's own request — no change-log entry; the project stays open ("ready")
    assert MP.flush_change_log("req-owner-site", reply="Built.") == 0 and store.list_changes(pid) == []
    assert store.project_state(pid) == "ready"


async def test_the_turn_records_the_request_it_runs_in(turns_on, monkeypatch, store):
    pid, tid = _project(store)
    seen = []
    _fake_turn(monkeypatch, lambda ag, kw: "nothing", seen)
    tok = request_id_context.set("req-owner-rec")
    try:
        await _advance(_ctx(store), pid, owner=True)
    finally:
        request_id_context.reset(tok)
    assert MP.request_root(seen[0]["request_id"]) == "req-owner-rec"


# ── m1: the lane ────────────────────────────────────────────────────────────

async def test_only_idle_autopilots_turn_takes_the_background_lane(turns_on, monkeypatch, store):
    pid, tid = _project(store)
    seen = []
    _fake_turn(monkeypatch, lambda ag, kw: "nothing", seen)
    tok = request_id_context.set("sched-nightly-7")
    try:
        await _advance(_ctx(store), pid, owner=False)
    finally:
        request_id_context.reset(tok)
    assert seen[0]["origin"] == "unattended" and seen[0]["is_background"] is False
    store.update_task(tid, status="PENDING")
    await _advance(_ctx(store), pid, owner=False)                  # idle: no request around it
    assert seen[1]["is_background"] is True


# ── m2: the allowance ───────────────────────────────────────────────────────

async def test_a_deferral_does_not_spend_the_daily_allowance(turns_on, monkeypatch, store):
    from ghost_agent.core.llm import BackgroundDeferred
    from ghost_agent.core.project_advancer import advance_unattended
    pid, tid = _project(store)
    _autopilot(store, pid)

    def script(ag, kw):
        raise BackgroundDeferred("owner")
    _fake_turn(monkeypatch, script, [])
    res = await advance_unattended(_ctx(store), pid, tool_runner=lambda *a, **k: None,
                                   coding_executor=_executor(idle=True))
    assert res.classification == "deferred"
    assert int(store.get_project(pid)["metadata"].get("unattended_steps_today") or 0) == 0


# ── M6: execute writes, by target ───────────────────────────────────────────

@pytest.mark.parametrize("cmd", [
    "python projects/abc/app.py 2>&1 | tail",
    "cat projects/abc/app.py > /tmp/x",
    "pytest projects/abc/tests 2>/dev/null",
    "cp projects/abc/app.py /tmp/x.py",
    "grep 'x >= 1' projects/abc/app.py",
    "cat > /tmp/run.sh <<'EOF'\necho hi > projects/abc/zz\nEOF\nbash /tmp/run.sh",
    "echo \"unterminated > projects/abc/q",
])
def test_a_command_that_only_reads_a_project_file_is_not_a_write(cmd):
    from ghost_agent.tools.execute import _PROJECT_PATH_RE, _shell_write_targets
    assert not [t for t in _shell_write_targets(cmd) if _PROJECT_PATH_RE.fullmatch(t)], cmd


@pytest.mark.parametrize("cmd, want", [
    ("echo hi > projects/abc/a.txt", ["projects/abc/a.txt"]),
    ("python gen.py >> ./projects/abc/log.txt", ["projects/abc/log.txt"]),
    ("cp /tmp/x.py projects/abc/", ["projects/abc/x.py"]),
    ("mv a.py b.py projects/abc/src/", ["projects/abc/src/a.py", "projects/abc/src/b.py"]),
    ("sed -i 's/a/b/' projects/abc/app.py", ["projects/abc/app.py"]),
    ("mkdir -p projects/abc/static && touch projects/abc/static/x", ["projects/abc/static",
                                                                     "projects/abc/static/x"]),
    ("python gen.py | tee -a projects/abc/out.txt", ["projects/abc/out.txt"]),
    ("FOO=1 cp x projects/abc/y 2>/dev/null", ["projects/abc/y"]),
    ("tar -czf projects/abc/out.tgz src", ["projects/abc/out.tgz"]),
    ("cat > projects/abc/a.py <<'EOF'\nprint(1)\nEOF", ["projects/abc/a.py"]),
])
def test_a_command_that_writes_into_a_project_names_what_it_wrote(cmd, want, monkeypatch, tmp_path):
    import ghost_agent.tools.projects as P
    from ghost_agent.core.request_scope import RequestScope, install, reset
    from ghost_agent.tools.execute import _note_execute_project_writes
    noted = []
    monkeypatch.setattr(P, "note_project_write", lambda ctx, rel: noted.append(str(rel)), raising=False)
    tok = install(RequestScope(context=SimpleNamespace(sandbox_dir=tmp_path)))
    try:
        _note_execute_project_writes({"command": cmd, "sandbox_dir": tmp_path}, "EXIT CODE: 0\nSTDOUT/STDERR:\n")
    finally:
        reset(tok)
    assert [n.split(str(tmp_path) + "/")[-1] for n in noted] == want, cmd
