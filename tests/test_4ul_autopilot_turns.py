"""§4UL (decision 1A) — autopilot and owner-requested batches run each project
task as ONE full main-loop agent turn (`core/project_turns.py`); the build
executor stays only as the fallback. Plus (1d): a `lab exec` / `run collect`
program failure is a fix-and-retry step like `execute`'s, and the fork-P
hooks (project writes noted, DONE sweeps flushed at the request's end)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ghost_agent.core import project_turns as PT
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome


@pytest.fixture
def turns_on(monkeypatch):
    monkeypatch.setenv("GHOST_PROJECT_TASK_TURNS", "1")


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def _ctx(store, fg=0):
    return SimpleNamespace(project_store=store, llm_client=SimpleNamespace(foreground_requests=fg),
                           current_project_id=None, scratchpad=None, graph_memory=None,
                           workspace_model=None, last_user_content="")


def _executor(idle=False, calls=None):
    async def _build(*a, **kw):
        if calls is not None:
            calls.append(kw)
        from ghost_agent.core.coding_executor import CodingResult
        return CodingResult(False, "stub executor ran", files=[], detail="")
    return PT.mark_real_executor(_build, idle=idle)


class _FakeTools:
    """A fake agent for `prepare_agent`: its tools record calls."""

    def __init__(self, calls):
        async def mp(**kw):
            calls.append(("manage_projects", kw))
            return ToolOutcome.ok("updated")

        async def ex(**kw):
            calls.append(("execute", kw))
            return kw.get("_res", ToolOutcome.ok("EXIT CODE: 0\nSTDOUT/STDERR:\nok"))

        async def fs(**kw):
            calls.append(("file_system", kw))
            return ToolOutcome.ok("written")

        async def lab(**kw):
            calls.append(("lab", kw))
            return kw.get("_res", ToolOutcome.ok("lab ok"))
        self.base = {"manage_projects": mp, "execute": ex, "file_system": fs, "lab": lab}
        self.available_tools = dict(self.base)

    def _rebuild_available_tools(self):
        self.available_tools = dict(self.base)
        return self.available_tools


def _fake_turn(monkeypatch, script, seen):
    """Patch `coding_loop.run_leaf_turn`: ``script(agent, kwargs)`` plays the
    model (calls tools through the WRAPPED dict); its return is the reply."""
    from ghost_agent.core import coding_loop as cl

    async def fake(context, **kw):
        seen.append(kw)
        calls = []
        agent = _FakeTools(calls)
        kw["prepare_agent"](agent)
        kw["_calls"] = calls
        r = script(agent, kw)
        return (await r) if asyncio.iscoroutine(r) else (r or "done")
    monkeypatch.setattr(cl, "run_leaf_turn", fake)


async def _advance(ctx, pid, owner=True, executor=None):
    from ghost_agent.core.project_advancer import advance_once
    return await advance_once(ctx, pid, owner_requested=owner, tool_runner=lambda *a, **k: None,
                              coding_executor=executor or _executor())


def _project(store, desc="Build the parser module → parser.py"):
    pid = store.create_project("Parser", kind="CODING", goal="a parser")
    tid = store.add_task(pid, desc)
    return pid, tid


def _close(store, tid, status="DONE"):
    """The model closes its task through manage_projects (as a real turn does); the fake handler's write is
    played by the store update when the call went through."""
    async def go(agent, kw):
        out = await agent.available_tools["manage_projects"](action="task_update", task_id=tid, status=status,
                                                             result="wrote parser.py and ran it")
        if getattr(getattr(out, "status", None), "value", "ok") == "ok":
            store.update_task(tid, status=status, result_summary="wrote parser.py and ran it")
        return "closed"
    return go


# ── gating ──────────────────────────────────────────────────────────────────

def test_production_default_is_on_and_the_suite_pins_it_off(monkeypatch):
    assert not PT.enabled()                          # tests/conftest.py
    monkeypatch.delenv("GHOST_PROJECT_TASK_TURNS", raising=False)
    assert PT.enabled()


def test_only_a_marked_executor_runs_tasks_as_turns(turns_on, store):
    async def stub(*a, **k):
        return None
    ctx = _ctx(store)
    assert not PT.wants_task_turn(ctx, stub)
    assert not PT.wants_task_turn(ctx, None)
    assert PT.wants_task_turn(ctx, _executor())
    assert not PT.wants_task_turn(SimpleNamespace(project_store=store, llm_client=None), _executor())


def test_the_production_executors_are_marked():
    from ghost_agent.core.coding_executor import build_coding_task
    assert build_coding_task._ghost_real_executor is True
    assert build_coding_task._ghost_idle_executor is False
    # the idle loop marks ITS executor idle: the call, read from the parsed tree
    import ast
    import inspect
    import ghost_agent.core.agent as A
    calls = [n for n in ast.walk(ast.parse(inspect.getsource(A)))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_mre"]
    assert any(isinstance(c.args[0], ast.Name) and c.args[0].id == "_aa_build"
               and any(k.arg == "idle" and getattr(k.value, "value", None) is True for k in c.keywords)
               for c in calls)


# ── the prompt ──────────────────────────────────────────────────────────────

def test_the_prompt_names_the_task_and_how_to_close_it():
    nxt = SimpleNamespace(id="t123", description="Build the parser → parser.py", kind="")
    p = PT.task_turn_prompt({"id": "p1", "title": "Parser", "goal": "a parser"}, nxt, "coding", "owner")
    assert "t123" in p and "task_update" in p and "status=DONE" in p and "FAILED" in p
    assert "do not ask questions" in p and "kept (production)" not in p
    u = PT.task_turn_prompt({"id": "p1"}, nxt, "coding", "unattended")
    assert "kept (production) machines are the owner's" in u and "WITHOUT closing" in u


# ── the turn's own walls ────────────────────────────────────────────────────

async def test_project_steering_is_refused_inside_a_task_turn_and_survives_a_rebuild():
    calls = []
    agent = _FakeTools(calls)
    watch = PT._TurnWatch()
    PT._prepare(watch)(agent)
    for act in ("autoadvance", "autopilot", "delete", "switch", "create", "replan", " AutoAdvance "):
        out = await agent.available_tools["manage_projects"](action=act)
        assert out.status is OutcomeStatus.REJECTED and out.reason_code == "task_turn_scope"
    assert calls == []
    out = await agent.available_tools["manage_projects"](action="task_update", task_id="t", status="DONE")
    assert out.status is OutcomeStatus.OK and calls[-1][0] == "manage_projects"
    agent._rebuild_available_tools()                 # a dispatch miss heals the dict from the registry
    out = await agent.available_tools["manage_projects"](action="autoadvance")
    assert out.reason_code == "task_turn_scope"


def test_the_watch_reads_a_check_after_the_last_write():
    w = PT._TurnWatch()
    w.note("file_system", {"operation": "write"}, ToolOutcome.ok("w"))
    assert not w.verified
    w.note("execute", {"command": "pytest"}, ToolOutcome.ok("EXIT CODE: 0"))
    assert w.verified
    w.note("file_system", {"operation": "edit"}, ToolOutcome.ok("e"))
    assert not w.verified                            # edited after its check
    w.note("execute", {}, "EXIT CODE: 1\nSTDOUT/STDERR:\nboom")
    assert not w.verified
    w.note("lab", {"action": "exec", "cmd": "pytest -q"}, ToolOutcome.ok("[l/m exit=0 1s]\nEXIT CODE: 0\n"))
    assert w.verified
    w.note("lab", {"action": "keep"}, ToolOutcome.rejected(
        "LAB ERROR (not_allowed): kept machines (production) are the owner's — not available to unattended work."))
    assert w.kept_refused


# ── run_leaf_turn: the task turn's surface ──────────────────────────────────

async def test_a_task_turn_keeps_every_tool_and_its_own_origin(monkeypatch):
    from ghost_agent.core import coding_loop as cl
    from ghost_agent.core import agent as agent_mod
    from ghost_agent.core.lab_waits import LEAF_ID
    seen = {}

    class FakeAgent:
        def __init__(self, iso):
            self.iso = iso
            self.available_tools = {"file_system": 1, "browser": 1, "manage_projects": 1, "web_search": 1}
            self.disabled_tools = set(); self.max_turns_override = None

        def _rebuild_available_tools(self):
            return self.available_tools

        async def handle_chat(self, body, background_tasks=None, request_id=None, scope=None):
            seen.update(avail=set(self.available_tools), disabled=set(self.disabled_tools), rid=request_id,
                        origin=scope.origin, pid=scope.project_id, label=self.iso.turn_origin_label,
                        leaf=LEAF_ID.get(), allow=self.iso._subagent_allowed_tools, turns=self.max_turns_override)
            return "ok", None, None
    monkeypatch.setattr(agent_mod, "GhostAgent", FakeAgent)
    base = SimpleNamespace(sandbox_dir="/tmp/sb", current_project_id=None, memory_system=object(),
                           graph_memory=None, scratchpad=None,
                           args=SimpleNamespace(perfect_it=True, smart_memory=0.5, native_tools=False),
                           workspace_model=object(), journal=object(), llm_client=object())
    prepared = []
    await cl.run_leaf_turn(base, leaf_id="ptask-x", prompt="p", is_background=False, project_id="p1",
                           allowed_tools=None, origin="owner", request_id="sub-ptask-x-1", label="ptask",
                           leaf_end=False, max_turns=40, prepare_agent=prepared.append)
    assert seen["avail"] == {"file_system", "browser", "manage_projects", "web_search"} and not seen["disabled"]
    assert seen["rid"] == "sub-ptask-x-1" and seen["origin"] == "owner" and seen["pid"] == "p1"
    assert seen["label"] == "ptask" and seen["allow"] is None and seen["turns"] == 40
    assert not seen["leaf"] and len(prepared) == 1   # an owner's run keeps what it starts: no leaf tag
    # the leaf's own defaults are unchanged
    await cl.run_leaf_turn(base, leaf_id="L", prompt="p", is_background=False, project_id="p1")
    assert seen["origin"] == "leaf" and seen["rid"] == "sub-leaf-L" and seen["avail"] == {"file_system"}


# ── the advancer: outcomes ──────────────────────────────────────────────────

async def test_the_owner_run_builds_a_task_as_a_turn_and_reads_its_close(turns_on, monkeypatch, store):
    pid, tid = _project(store)
    seen = []
    _fake_turn(monkeypatch, _close(store, tid), seen)
    built = []
    res = await _advance(_ctx(store), pid, owner=True, executor=_executor(calls=built))
    assert res.task_id == tid and res.classification == "coding" and "wrote parser.py" in res.summary
    assert store.get_task(tid)["status"] == "DONE" and built == []      # the executor never ran
    kw = seen[0]
    assert kw["origin"] == "owner" and kw["is_background"] is False and kw["leaf_end"] is False
    assert kw["allowed_tools"] is None and kw["request_id"].startswith("sub-ptask-") and kw["label"] == "ptask"
    assert kw["project_id"] == pid and tid in kw["prompt"]
    ev = store.list_events(pid, event_type="task_turn")
    assert ev and ev[0]["payload"]["status"] == "done" and ev[0]["payload"]["origin"] == "owner"


async def test_an_unattended_turn_is_background_and_unattended(turns_on, monkeypatch, store):
    pid, tid = _project(store)
    seen = []

    async def script(agent, kw):
        await agent.available_tools["file_system"](operation="write", path="parser.py", content="x")
        await agent.available_tools["execute"](command="python parser.py")
        store.update_task(tid, status="DONE", result_summary="built and ran it")
    _fake_turn(monkeypatch, script, seen)
    res = await _advance(_ctx(store), pid, owner=False)
    assert seen[0]["origin"] == "unattended" and seen[0]["is_background"] is True and seen[0]["leaf_end"] is True
    assert store.get_task(tid)["status"] == "DONE" and res.classification == "coding"


async def test_an_unattended_done_nothing_checked_waits_for_the_owner(turns_on, monkeypatch, store):
    pid, tid = _project(store)

    async def script(agent, kw):
        await agent.available_tools["file_system"](operation="write", path="parser.py", content="x")
        await agent.available_tools["manage_projects"](action="task_update", task_id=tid, status="DONE",
                                                       result="wrote it")
    _fake_turn(monkeypatch, script, [])
    res = await _advance(_ctx(store), pid, owner=False)
    row = store.get_task(tid)
    assert res.classification == "needs_user" and row["status"] == "NEEDS_USER"
    assert row["wait_on"] == "owner_turn" and row["wait_code"] == "unverified_build"


async def test_a_kept_lab_refusal_holds_the_task_for_the_owners_run(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Deploy the parser site to the kept lab")

    async def script(agent, kw):
        await agent.available_tools["lab"](action="keep", _res=ToolOutcome.rejected(
            "LAB ERROR (not_allowed): kept machines (production) are the owner's — not available to unattended "
            "work."))
        return "everything else is ready; the deploy waits for the owner"
    _fake_turn(monkeypatch, script, [])
    res = await _advance(_ctx(store), pid, owner=False)
    row = store.get_task(tid)
    assert res.classification == "needs_user" and row["status"] == "NEEDS_USER"
    assert row["wait_code"] == "kept_lab"
    ev = store.list_events(pid, event_type="autoadvance_needs_user")
    assert ev[0]["payload"]["hold"] == "kept_lab"
    # the owner's run takes the hold back and runs it with owner authority
    seen = []
    _fake_turn(monkeypatch, _close(store, tid), seen)
    res2 = await _advance(_ctx(store), pid, owner=True)
    assert res2.task_id == tid and seen[0]["origin"] == "owner" and store.get_task(tid)["status"] == "DONE"


async def test_a_turn_that_never_closes_its_task_fails_it_with_the_reply(turns_on, monkeypatch, store):
    pid, tid = _project(store)
    _fake_turn(monkeypatch, lambda a, kw: "I looked at the files.", [])
    res = await _advance(_ctx(store), pid, owner=True)
    row = store.get_task(tid)
    assert row["status"] == "FAILED" and "without closing it" in (row.get("failure_reason") or "")
    assert "I looked at the files." in res.summary


async def test_a_turn_that_splits_its_task_leaves_the_subtasks(turns_on, monkeypatch, store):
    pid, tid = _project(store)

    def script(agent, kw):
        store.add_task(pid, "Lexer → lexer.py", parent_id=tid)
        store.add_task(pid, "Grammar → grammar.py", parent_id=tid)
    _fake_turn(monkeypatch, script, [])
    res = await _advance(_ctx(store), pid, owner=True)
    assert "split into 2" in res.summary and store.get_task(tid)["status"] != "FAILED"


async def test_idle_autopilot_waits_while_a_user_request_is_active(turns_on, monkeypatch, store):
    pid, tid = _project(store)
    seen = []
    _fake_turn(monkeypatch, _close(store, tid), seen)
    res = await _advance(_ctx(store, fg=1), pid, owner=False, executor=_executor(idle=True))
    assert res.classification == "deferred" and seen == []
    assert store.get_task(tid)["status"] == "PENDING"
    # a non-idle unattended batch is not gated on it (its calls yield one by one)
    await _advance(_ctx(store, fg=1), pid, owner=False, executor=_executor(idle=False))
    assert len(seen) == 1


@pytest.mark.parametrize("boom, cls, head", [
    (asyncio.TimeoutError(), "coding", "code build ran out of time"),
    ("deferred", "deferred", "deferred"),
    (RuntimeError("bad"), "blocked", "coding executor crashed (task turn)"),
])
async def test_an_aborted_turn_gives_the_task_back(turns_on, monkeypatch, store, boom, cls, head):
    from ghost_agent.core.llm import BackgroundDeferred
    pid, tid = _project(store)

    def script(agent, kw):
        raise BackgroundDeferred("owner") if boom == "deferred" else boom
    _fake_turn(monkeypatch, script, [])
    res = await _advance(_ctx(store), pid, owner=False)
    assert res.classification == cls and res.summary.startswith(head) or head in res.summary
    assert store.get_task(tid)["status"] == "PENDING"


async def test_an_upstream_abort_in_an_unattended_turn_is_a_deferral(turns_on, monkeypatch, store):
    pid, tid = _project(store)
    _fake_turn(monkeypatch, lambda a, kw: "[ATTEMPT_ABORTED_UPSTREAM] owner request", [])
    res = await _advance(_ctx(store, fg=1), pid, owner=False)      # §4UL round 6: the owner really took the slot
    assert res.classification == "deferred" and store.get_task(tid)["status"] == "PENDING"


async def test_too_little_time_left_starts_no_turn(turns_on, monkeypatch, store):
    import time
    from ghost_agent.core.coding_executor import leaf_deadline_context
    pid, tid = _project(store)
    seen = []
    _fake_turn(monkeypatch, _close(store, tid), seen)
    tok = leaf_deadline_context.set(time.monotonic() + 30)
    try:
        res = await _advance(_ctx(store), pid, owner=False)
    finally:
        leaf_deadline_context.reset(tok)
    assert seen == [] and res.summary.startswith("code build ran out of time")
    assert store.get_task(tid)["status"] == "PENDING"


async def test_switched_off_or_unbuildable_the_executor_runs_it(monkeypatch, store):
    pid, tid = _project(store)
    seen, built = [], []
    _fake_turn(monkeypatch, _close(store, tid), seen)
    await _advance(_ctx(store), pid, owner=True, executor=_executor(calls=built))
    assert seen == [] and len(built) == 1            # GHOST_PROJECT_TASK_TURNS=0 (suite default)
    monkeypatch.setenv("GHOST_PROJECT_TASK_TURNS", "1")
    store.update_task(tid, status="PENDING")

    def bad(*a, **k):
        raise ValueError("no prompt")
    monkeypatch.setattr(PT, "task_turn_prompt", bad)
    await _advance(_ctx(store), pid, owner=True, executor=_executor(calls=built))
    assert seen == [] and len(built) == 2            # the fallback, logged


async def test_a_research_task_is_a_turn_too(turns_on, monkeypatch, store):
    pid = store.create_project("Trip", kind="GENERAL", goal="plan a trip")
    tid = store.add_task(pid, "Research ferry schedules to Naxos → research/ferries.md")
    seen = []
    _fake_turn(monkeypatch, _close(store, tid), seen)
    from ghost_agent.core.project_advancer import advance_once
    res = await advance_once(_ctx(store), pid, owner_requested=True, tool_runner=lambda *a, **k: None,
                             llm_classifier=None, coding_executor=_executor())
    assert seen and res.task_id == tid and store.get_task(tid)["status"] == "DONE"


# ── (1d) lab exec / run collect program failures ────────────────────────────

def test_a_lab_exec_that_ran_and_failed_is_a_program_failure():
    from ghost_agent.core.agent import _lab_program_failed as f
    ran = ToolOutcome("[lab_1/db exit=1 2.0s]\nEXIT CODE: 1\nSTDERR:\nFAILED test_x\n",
                      status=OutcomeStatus.FAILED, reason_code="lab_exit_nonzero")
    assert f("lab", ran, str(ran))
    killed = ToolOutcome("[lab_1/db exit=137 2.0s]\nEXIT CODE: 137\nSTDERR:\nKilled\n",
                         status=OutcomeStatus.FAILED, reason_code="lab_exit_nonzero")
    assert not f("lab", killed, str(killed))
    timed = ToolOutcome("[lab_1/db TIMED OUT after 60 s]\nTIMED OUT: …\nSTDOUT:\nx\n",
                        status=OutcomeStatus.FAILED, reason_code="lab_timed_out")
    assert not f("lab", timed, str(timed))
    refused = ToolOutcome.rejected("LAB ERROR (not_allowed): x", reason_code="lab_not_allowed")
    assert not f("lab", refused, str(refused))
    assert not f("execute", ran, str(ran))


def test_a_run_collect_of_a_job_that_exited_non_zero_is_a_program_failure():
    from ghost_agent.core.agent import _lab_program_failed as f
    ex = ToolOutcome("[run tests exited exit=2]\nEXIT CODE: 2\n2 failed\n",
                     status=OutcomeStatus.FAILED, reason_code="run_job_failed")
    assert f("run", ex, str(ex))
    lost = ToolOutcome("[run tests lost]\nEXIT CODE: none\n",
                       status=OutcomeStatus.FAILED, reason_code="run_job_failed")
    assert not f("run", lost, str(lost))


async def test_a_red_lab_exec_with_new_output_is_not_a_strike():
    from ghost_agent.core.strikes import StrikeLedger
    import tests.test_4jj_search_yield_steer as H
    agent = H._agent()
    n = {"i": 0}

    async def lab(**kw):
        n["i"] += 1
        return ToolOutcome(f"[lab_1/db exit=1 1.0s]\nEXIT CODE: 1\nSTDERR:\nFAILED test_{n['i']}\n",
                           status=OutcomeStatus.FAILED, reason_code="lab_exit_nonzero")
    agent.available_tools = {"lab": lab}
    strikes = StrikeLedger()
    ts = H._ts([("lab", {"action": "exec", "lab_id": "lab_1", "member": "db", "cmd": "pytest"})], strikes, set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.execution_failure_count == 0


# ── the removed go-ahead machinery stays removed ────────────────────────────

def test_the_confirm_wait_takeover_is_gone():
    import ghost_agent.core.agent as A
    import ghost_agent.core.receipt_marks as RM
    for name in ("_confirm_wait_reply", "_CONFIRM_WAIT_CODES", "_CONFIRM_WAIT_RE", "_PREVIEW_HEAD_RE",
                 "_quiet_kb_result"):
        assert not hasattr(A, name), name
    assert not hasattr(RM, "NOT_ANSWERED_YET") and not hasattr(RM, "CONFIRM_WAIT_REPLY_HEAD")
    assert "confirm_dead_end" not in A._DEAD_END_REFUSALS


# ── fork-P hooks ────────────────────────────────────────────────────────────

async def test_a_write_into_a_projects_folder_is_noted(monkeypatch, tmp_path):
    import ghost_agent.tools.projects as P
    from ghost_agent.core.request_scope import RequestScope, install, reset
    from ghost_agent.tools.file_system import tool_file_system
    noted = []
    monkeypatch.setattr(P, "note_project_write", lambda ctx, rel: noted.append(rel), raising=False)
    (tmp_path / "projects" / "abc123").mkdir(parents=True)
    ctx = SimpleNamespace(sandbox_dir=tmp_path)
    tok = install(RequestScope(context=ctx))
    try:
        out = await tool_file_system(operation="write", sandbox_dir=tmp_path,
                                     path="projects/abc123/app.py", content="print(1)\n")
        await tool_file_system(operation="write", sandbox_dir=tmp_path, path="notes.txt", content="x\n")
    finally:
        reset(tok)
    assert getattr(out, "status", None) is not OutcomeStatus.REJECTED
    assert noted == ["projects/abc123/app.py"]
    # the hook's note (a DONE project reopened) reaches the model, status kept
    monkeypatch.setattr(P, "note_project_write", lambda ctx, rel: "Project abc123 was DONE; reopened.",
                        raising=False)
    tok = install(RequestScope(context=ctx))
    try:
        out = await tool_file_system(operation="write", sandbox_dir=tmp_path,
                                     path="projects/abc123/b.py", content="print(2)\n")
    finally:
        reset(tok)
    assert str(out).startswith("SUCCESS") and str(out).endswith("was DONE; reopened.")


def test_an_execute_that_writes_into_a_project_is_noted(monkeypatch, tmp_path):
    import ghost_agent.tools.projects as P
    from ghost_agent.core.request_scope import RequestScope, install, reset
    from ghost_agent.tools.execute import _note_execute_project_writes
    noted = []
    monkeypatch.setattr(P, "note_project_write", lambda ctx, rel: noted.append(rel), raising=False)
    tok = install(RequestScope(context=SimpleNamespace(sandbox_dir=tmp_path)))
    try:
        ok = "EXIT CODE: 0\nSTDOUT/STDERR:\n"
        _note_execute_project_writes({"command": "cp a.py projects/abc/a.py", "sandbox_dir": tmp_path}, ok)
        _note_execute_project_writes({"command": "cat projects/abc/a.py", "sandbox_dir": tmp_path}, ok)
        _note_execute_project_writes({"command": "cp a projects/abc/b", "sandbox_dir": tmp_path},
                                     "EXIT CODE: 1\nSTDOUT/STDERR:\nno")
        _note_execute_project_writes({"command": "cp a projects/abc/c", "sandbox_dir": tmp_path,
                                      "container_workdir": "/workspace/projects/abc"}, ok)
    finally:
        reset(tok)
    assert noted == ["projects/abc/a.py"]


def test_the_execute_wrapper_keeps_the_real_function_readable():
    import inspect
    from ghost_agent.tools import execute as ex
    assert ex.tool_execute.__name__ == "tool_execute"
    assert inspect.unwrap(ex.tool_execute) is ex._tool_execute_impl


async def test_the_request_end_flushes_deferred_sweeps(monkeypatch):
    import ghost_agent.memory.projects as P
    from tests.test_forced_final_no_answer import _agent as _ff_agent, _resp
    from tests.helpers import FakeBgTasks
    flushed = []
    monkeypatch.setattr(P, "flush_deferred_sweeps", lambda rid: flushed.append(rid), raising=False)
    agent, _ = _ff_agent(monkeypatch, [_resp("Hello there."), _resp("(unreachable)")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "hi"}]}, FakeBgTasks(),
                            request_id="req-4ul-sweep")
    assert "req-4ul-sweep" in flushed
