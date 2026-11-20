"""§4UY — the projects subsystem's fixes (findings P1–P23 of §4UX-3).

Every test here fails in the world the finding describes (the fix stubbed or the old code restored). Owner and
people are fictional (Elena, Markos, Kosmas)."""

import ast
import asyncio
import json
import os
import pathlib
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.core import project_turns as PT
from ghost_agent.core import request_scope as RS
from ghost_agent.memory import projects as MP
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools import projects as P
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome
from ghost_agent.utils.logging import request_id_context

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "ghost_agent"
OK = ToolOutcome.ok("EXIT CODE: 0\nSTDOUT/STDERR:\nok")


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def _ctx(store, pid=None):
    return SimpleNamespace(project_store=store, scratchpad=None, graph_memory=None, workspace_model=None,
                           current_project_id=pid, last_user_content="", sandbox_manager=None,
                           conversation_key="conv-elena")


class _rid:
    def __init__(self, rid):
        self.rid = rid

    def __enter__(self):
        self.tok = request_id_context.set(self.rid)

    def __exit__(self, *a):
        request_id_context.reset(self.tok)


def _code(out):
    return getattr(out, "reason_code", None)


# ── P1 / P9: ONE who-may-act gate over the whole action table ────────────────

def test_every_action_is_in_exactly_one_class_and_every_owner_action_is_refused_in_a_task_turn():
    assert (P._READ_ACTIONS | P._WORK_ACTIONS | P._OWNER_ACTIONS) == set(P._ACTIONS)
    assert not (P._READ_ACTIONS & P._WORK_ACTIONS) and not (P._READ_ACTIONS & P._OWNER_ACTIONS)
    assert not (P._WORK_ACTIONS & P._OWNER_ACTIONS)
    assert P._OWNER_ACTIONS <= PT.REFUSED_PROJECT_ACTIONS


def test_only_the_model_entry_calls_the_action_table():
    """The gate sits at `tool_manage_projects_for_model`; no other production code may reach the table around it."""
    callers = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) \
                        == "tool_manage_projects":
                    callers.append(fn.name)
    assert set(callers) <= {"tool_manage_projects_for_model", "_closes_in_dependency_order"}, callers


def _owner_kwargs(act, pid, tid):
    kw = {"action": act, "project_id": pid}
    if act == "update":
        kw.update(status="PAUSED", title="Hijacked")
    elif act == "budget":
        kw["metadata"] = {"unattended_per_day": 999}
    elif act == "autopilot":
        kw["enabled"] = True
    elif act in ("task_delete",):
        kw = {"action": act, "task_id": tid}
    elif act == "replan":
        kw["subtasks"] = ["New plan → n.py", "Other → o.py"]
    elif act == "set_dependency":
        kw["payload"] = "x"
    elif act == "constraint_retire":
        kw["payload"] = "all"
    return kw


@pytest.mark.parametrize("act", sorted(P._OWNER_ACTIONS))
@pytest.mark.parametrize("who", ["sched-autopilot-7", "probe-3f34a566", "sub-ptask-1a2b3c4d", "SYSTEM",
                                 "member", "untrusted"])
async def test_no_owner_action_runs_outside_the_owners_clean_turn(store, act, who):
    pid = store.create_project("Elena's shop", kind="CODING", goal="g")
    tid = store.add_task(pid, "Build it → a.py")
    before = (store.get_project(pid), store.list_tasks(pid))
    ctx = _ctx(store, pid)
    rid = {"member": "req-markos-1", "untrusted": "req-elena-web"}.get(who, who)
    tok = RS.install(RS.RequestScope(conversation_key="conv", project_id=pid,
                                     origin="member" if who == "member" else "owner", context=ctx))
    try:
        if who == "untrusted":
            from ghost_agent.utils.provenance import note_untrusted_content
            note_untrusted_content("web_search", req_id=rid)
        with _rid(rid):
            out = await P.tool_manage_projects_for_model(ctx, **_owner_kwargs(act, pid, tid))
    finally:
        RS.reset(tok)
    assert _code(out) == ("untrusted_content" if who == "untrusted" else "who_may_act"), (act, who, str(out))
    after = (store.get_project(pid), store.list_tasks(pid))
    assert after[0]["status"] == before[0]["status"] and after[0]["title"] == before[0]["title"]
    assert len(after[1]) == len(before[1]) and after[0].get("metadata") == before[0].get("metadata")


@pytest.mark.parametrize("act", sorted(P._READ_ACTIONS | P._WORK_ACTIONS))
def test_read_and_work_actions_are_not_walled(store, act):
    assert P._who_may_act(_ctx(store), store, act, {"action": act}) is None


async def test_the_owner_runs_an_owner_action_and_a_probe_its_own_project(store):
    pid = store.create_project("Kosmas notes", kind="GENERAL", goal="g")
    with _rid("req-elena-2"):
        out = await P.tool_manage_projects_for_model(_ctx(store, pid), action="update", project_id=pid,
                                                     title="Kosmas notes v2")
    assert _code(out) is None and store.get_project(pid)["title"] == "Kosmas notes v2"
    own = store.create_project("probe thing", kind="GENERAL", goal="g", metadata={"probe_created": True})
    with _rid("probe-1234abcd"):
        out = await P.tool_manage_projects_for_model(_ctx(store, own), action="update", project_id=own,
                                                     title="probe thing 2")
    assert _code(out) is None and store.get_project(own)["title"] == "probe thing 2"


@pytest.mark.parametrize("key", sorted(P._SYSTEM_METADATA_KEYS))
async def test_the_model_never_writes_a_system_metadata_key(store, key):
    pid = store.create_project("Elena's app", kind="CODING", goal="g")
    with _rid("req-elena-3"):
        await P.tool_manage_projects_for_model(_ctx(store, pid), action="update", project_id=pid,
                                               metadata={key: "forged", "colour": "teal"})
        await P.tool_manage_projects_for_model(_ctx(store), action="create", title="Markos site",
                                               metadata={key: "forged"})
    m = store.get_project(pid)["metadata"] or {}
    assert m.get(key) != "forged"
    assert m.get("notes", {}).get("colour") == "teal" or key in ("autopilot",) or key in P._BUDGET_KEYS
    for p in store.list_projects():
        assert (p.get("metadata") or {}).get(key) != "forged"


async def test_a_forged_archived_from_cannot_resume_to_released(store):
    pid = store.create_project("Old shop", kind="CODING", goal="g")
    with _rid("req-elena-4"):
        await P.tool_manage_projects_for_model(_ctx(store, pid), action="update", project_id=pid,
                                               metadata={"archived_from": "RELEASED"})
    assert "archived_from" not in (store.get_project(pid)["metadata"] or {})


# ── P2: a plan finished before §4UM ──────────────────────────────────────────

def _legacy_project(store, ts):
    pid = store.create_project("Elena dashboard", kind="CODING", goal="g")
    t1 = store.add_task(pid, "Build → index.html")
    store.update_task(t1, status="DONE", result_summary="built")
    with store._lock, store._connect() as conn:          # the old roll-up close, before §4UM
        conn.execute("DELETE FROM project_events WHERE project_id = ? AND type = 'project_auto_rollup'", (pid,))
        conn.execute("UPDATE projects SET metadata_json = '{}' WHERE id = ?", (pid,))
        conn.commit()
    store.log_event(pid, None, "project_auto_rollup", {"new_status": "DONE"})
    with store._lock, store._connect() as conn:
        conn.execute("UPDATE project_events SET ts = ? WHERE project_id = ? AND type = 'project_auto_rollup'",
                     (ts, pid))
        conn.commit()
    return pid


def test_a_legacy_finish_is_the_first_plan_and_the_next_request_is_a_change(store):
    legacy = time.time() - 86400 * 5
    pid = _legacy_project(store, legacy)
    t2 = store.add_task(pid, "Portfolio panel → js/apps/portfolio.js")
    with _rid("req-elena-5"):
        store.update_task(t2, status="DONE", result_summary="added")
    m = store.get_project(pid)["metadata"]
    assert m["first_ready_req"] == "" and abs(float(m["first_ready_at"]) - legacy) < 1
    assert MP._was_ready_before(store, pid, "req-elena-5", time.time())


def test_a_failed_rollup_is_not_a_finished_plan(store):
    pid = store.create_project("Markos tool", kind="CODING", goal="g")
    store.log_event(pid, None, "project_auto_rollup", {"new_status": "FAILED"})
    assert MP.legacy_ready_at(store, pid) is None
    assert not MP._was_ready_before(store, pid, "req-x", time.time() + 5)


def test_the_backfill_script_repairs_a_swallowed_request_on_a_fixture(tmp_path):
    import importlib.util
    from ghost_agent.memory import store_lock
    spec = importlib.util.spec_from_file_location(
        "bf", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "backfill_4uy_first_ready.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    mem, sb = tmp_path / "mem", tmp_path / "sb"
    store = ProjectStore(mem, sandbox_root=sb)
    legacy = time.time() - 86400 * 3
    pid = _legacy_project(store, legacy)
    ws = pathlib.Path(store.ensure_workspace(pid))
    t2 = store.add_task(pid, "Panel → js/panel.js")
    (ws / "js").mkdir(exist_ok=True)
    (ws / "js" / "panel.js").write_text("export const panel = 1;\n")
    store.add_work_log(pid, request="add the panel for Kosmas", files=["js/panel.js"], outcome="completed")
    store.update_task(t2, status="DONE", result_summary="panel")
    store._atomic_metadata_update(pid, lambda m: {**m, "first_ready_at": time.time(),
                                                  "first_ready_req": "req-swallowed"})
    (t1,) = [t["id"] for t in store.list_tasks(pid) if t["description"].startswith("Build")]
    with store._lock, store._connect() as conn:            # review 2: t1 finished the FIRST plan (legacy)
        conn.execute("UPDATE tasks SET created_at = ?, updated_at = ? WHERE id = ?", (legacy - 60, legacy - 30, t1))
        conn.commit()
    plan = bf.plan_repair(store, pid)
    assert "(1 task(s))" in plan["entry"]["summary"], plan["entry"]["summary"]      # t2 only: t1 predates the cut
    try:
        assert bf.main(["--memory-dir", str(mem), "--sandbox-root", str(sb)]) == 0      # dry run
        assert store.get_project(pid)["metadata"]["first_ready_req"] == "req-swallowed"
        assert bf.main(["--memory-dir", str(mem), "--sandbox-root", str(sb), "--apply"]) == 0
    finally:
        store_lock.release_writer_lock(mem)
    m = store.get_project(pid)["metadata"]
    assert m["first_ready_req"] == "" and abs(float(m["first_ready_at"]) - legacy) < 1
    (e,) = store.list_changes(pid, kind="change")
    assert e["req_id"] == "req-swallowed" and "js/panel.js" in (e["added"] + e["changed"])
    assert bf.plan_repair(store, pid) is None                                            # idempotent


# ── P3: a task turn's progress counts for the request it runs in ────────────

def test_progress_of_an_inner_turn_is_credited_to_the_outer_request():
    from ghost_agent.utils import logging as L
    with L._REQ_STATE_LOCK:
        L._REQ_STATE["req-elena-outer"] = {"started": time.monotonic()}
    try:
        L.set_progress_parent("sub-ptask-9f8e7d6c", "req-elena-outer")
        assert L.note_code_progress("sub-ptask-9f8e7d6c", "wrote js/app.js", key="k1")
        assert L.request_recent_progress("req-elena-outer")
        assert L.request_progress_what("req-elena-outer") == "wrote js/app.js"
        with L._REQ_STATE_LOCK:                               # a tracked inner turn: counted on both
            L._REQ_STATE["sub-ptask-tracked1"] = {"started": time.monotonic()}
            L._REQ_STATE["req-elena-outer"].pop("progress_t", None)
        L.set_progress_parent("sub-ptask-tracked1", "req-elena-outer")
        assert L.note_code_progress("sub-ptask-tracked1", "wrote js/b.js", key="k2")
        assert L.request_recent_progress("req-elena-outer")
        assert L.request_progress_what("req-elena-outer") == "wrote js/b.js"
    finally:
        with L._REQ_STATE_LOCK:
            L._REQ_STATE.pop("req-elena-outer", None)
            L._REQ_STATE.pop("sub-ptask-tracked1", None)


def test_a_progressing_batch_is_judged_on_the_time_it_can_still_earn(monkeypatch):
    from ghost_agent.core import coding_executor as CE
    from ghost_agent.core import project_advancer as PA
    from ghost_agent.utils import logging as L
    monkeypatch.setattr(L, "request_remaining_s", lambda rid: 300.0)
    monkeypatch.setattr(CE, "_request_left_s", lambda: 5000.0)
    monkeypatch.setattr(L, "request_recent_progress", lambda rid: True)
    with _rid("req-elena-batch"):
        assert not PA._batch_would_cross_deadline(1200.0)
        monkeypatch.setattr(L, "request_recent_progress", lambda rid: False)
        assert PA._batch_would_cross_deadline(1200.0)


# ── P4: a plan-only request cannot run the plan ─────────────────────────────

async def test_autoadvance_after_a_plan_only_decompose_in_the_same_request_is_refused(store):
    ctx = _ctx(store)
    with _rid("req-elena-plan"):
        await P.tool_manage_projects_for_model(ctx, action="create", title="Elena blog")
        await P.tool_manage_projects_for_model(ctx, action="task_decompose",
                                               subtasks=["Layout → index.html", "Posts → posts.js"])
        out = await P.tool_manage_projects_for_model(ctx, action="autoadvance", count="all")
    assert _code(out) == "plan_only_request"
    assert all(t["status"] == "PENDING" for t in store.list_tasks(ctx.current_project_id))


async def test_go_ahead_on_the_plan_call_is_kept_and_a_new_request_is_free(store, monkeypatch):
    seen = []
    from ghost_agent.core import project_advancer as PA

    async def fake_many(*a, **k):
        seen.append(1)
        return PA.AdvanceManyResult([], "count_reached", None)
    monkeypatch.setattr(PA, "advance_many", fake_many)
    ctx = _ctx(store)
    with _rid("req-elena-go"):
        await P.tool_manage_projects_for_model(ctx, action="create", title="Markos game",
                                               subtasks=["Board → board.js", "Rules → rules.js"], go_ahead=True)
        out = await P.tool_manage_projects_for_model(ctx, action="autoadvance")
    assert _code(out) != "plan_only_request" and seen
    with _rid("req-elena-plan2"):
        await P.tool_manage_projects_for_model(ctx, action="task_decompose", subtasks=["A → a.js", "B → b.js"])
    with _rid("req-elena-next"):
        out = await P.tool_manage_projects_for_model(ctx, action="autoadvance")
    assert _code(out) != "plan_only_request"


# ── P6: a check is a successful run of the project's own code ───────────────

def _watch(tmp_path, files=("app.py",)):
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    for f in files:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text("x = 1\n")
    w = PT._TurnWatch(root=root)
    w.note("file_system", {"operation": "write", "path": "index.html"}, ToolOutcome.ok("w"))
    return w


@pytest.mark.parametrize("name,kw", [
    ("execute", {"command": "python3 -c 'print(1)'"}),
    ("execute", {"content": "print(1 + 1)"}),
    ("execute", {"command": "node --check app.js"}),
    ("execute", {"command": "python -m py_compile app.py"}),
    ("execute", {"command": "tsc --noEmit"}),
    ("execute", {"command": "ruff check ."}),
    ("browser", {"operation": "screenshot", "url": "https://example.org/"}),
])
def test_what_runs_nothing_of_the_project_is_not_a_check(tmp_path, name, kw):
    assert not _watch(tmp_path).verified or name == "never"
    w = _watch(tmp_path)
    w.note(name, kw, OK)
    assert not w.verified, (name, kw)


def test_a_run_that_failed_is_not_a_check(tmp_path):
    w = _watch(tmp_path)
    w.note("execute", {"command": "python app.py"}, ToolOutcome("EXIT CODE: 1\nSTDOUT/STDERR:\nboom",
                                                               status=OutcomeStatus.FAILED))
    assert not w.verified


@pytest.mark.parametrize("name,kw", [
    ("execute", {"command": "python app.py"}),
    ("execute", {"command": "pytest -q"}),
    ("execute", {"command": "npm test"}),
    ("execute", {"content": "import app\nprint(app.x)"}),
    ("execute", {"command": "curl -s http://localhost:8080/"}),
    ("execute", {"command": "node index.html"}),                     # a file this turn wrote
    ("browser", {"operation": "screenshot", "url": "http://localhost:8080/"}),
    ("lab", {"action": "http", "path": "/"}),
])
def test_a_successful_run_of_the_project_is_a_check(tmp_path, name, kw):
    w = _watch(tmp_path)
    w.note(name, kw, OK)
    assert w.verified, (name, kw)


# ── P7: the declared file must be this task's work ──────────────────────────

def test_a_declared_file_that_already_existed_does_not_close_the_task_alone(tmp_path):
    from ghost_agent.core.task_close import gate_deliverable
    f = tmp_path / "notes.md"
    f.write_text("# Notes for Elena\n\nA real document with enough words in it to not be hollow at all.\n" * 3)
    old = time.time() - 3600
    os.utime(f, (old, old))
    since = time.time() - 60
    r = gate_deliverable(tmp_path, "Write the notes → notes.md", since=since)
    assert r is not None and "already existed" in r.reason
    # review: the result naming the file is prose, not evidence — this task's own calls must have touched it
    assert gate_deliverable(tmp_path, "Write the notes → notes.md", since=since,
                            touched={str(f.resolve())}) is None
    os.utime(f, None)                                       # written now: this task's
    assert gate_deliverable(tmp_path, "Write the notes → notes.md", since=since) is None


async def test_a_read_of_the_existing_file_in_this_request_is_the_tasks_evidence(store):
    """Review MAJOR (P7): `result` naming the file passed (14 of 16 live results do). The evidence is this
    request's own successful call on the file, recorded by the registry's wrapper."""
    from ghost_agent.core.planning import ProjectPlan
    from ghost_agent.core.task_close import check_close
    pid = store.create_project("Old file 2", kind="GENERAL", goal="g")
    ws = pathlib.Path(store.ensure_workspace(pid))
    f = ws / "guide.md"
    f.write_text("# Guide for Kosmas\n\nA real document with enough words in it to not be hollow.\n" * 3)
    os.utime(f, (time.time() - 7200, time.time() - 7200))
    t = store.add_task(pid, "Write the guide → guide.md")

    async def fake_fs(**kw):
        return ToolOutcome.ok("# Guide for Kosmas …")

    async def failing_fs(**kw):
        return ToolOutcome.failed("Error: no such file", reason_code="x")
    with _rid("sched-nightly-ev"):                            # a background close: the refusal stands
        v = check_close(store, pid, ProjectPlan(store, pid).tree, t, actor="model",
                        result="guide.md already covers it")
        assert not v.ok and v.refused[0].gate == "deliverable"
        await P.with_task_evidence("file_system", failing_fs, _ctx(store, pid), lambda: ws)(
            operation="read", path="guide.md")
        assert not check_close(store, pid, ProjectPlan(store, pid).tree, t, actor="model", result="x").ok
        await P.with_task_evidence("file_system", fake_fs, _ctx(store, pid), lambda: ws)(
            operation="read", path="guide.md")
        assert check_close(store, pid, ProjectPlan(store, pid).tree, t, actor="model", result="x").ok
    with _rid("sched-nightly-ev-other"):                      # another request's read is not this one's
        assert not check_close(store, pid, ProjectPlan(store, pid).tree, t, actor="model", result="x").ok


def test_the_registry_tools_record_evidence_and_watch_writers(mock_context, monkeypatch):
    """Behaviour, not source: the registry's file_system / execute record this request's evidence, and a watched
    writer (report_pdf) watches the project folder before it runs."""
    from ghost_agent.core import task_deliverable as TD
    from ghost_agent.tools import registry as R
    from ghost_agent.tools import report_pdf as RP
    noted, watched = [], []
    monkeypatch.setattr(TD, "note_tool_evidence", lambda rid, name, kw, res, folder: noted.append(name))
    monkeypatch.setattr(P, "watch_tool_folder", lambda ctx, folder: watched.append(folder))

    async def fake(**kw):
        return "ok"
    monkeypatch.setattr(R, "tool_file_system", fake)
    monkeypatch.setattr(RP, "tool_generate_pdf", fake)
    tools = R.get_available_tools(mock_context)
    asyncio.run(tools["file_system"](operation="read", path="a.md"))
    asyncio.run(tools["report_pdf"](title="Kosmas"))
    assert noted == ["file_system"] and len(watched) == 1


# ── P8: every writer into a project folder is classified, and the watched ones are watched ──

_HOOKS = {"prepare_project_write", "note_project_write", "note_project_writes", "note_system_path",
          "mark_task_write", "prepare_project_write_hook", "note_project_write_hook", "note_project_writes_hook"}
_WRITE_ATTRS = {"write_text", "write_bytes", "savefig", "save", "touch", "rename", "replace", "unlink", "rmtree",
                "copy", "copy2", "copyfile", "move", "copytree"}


def _writes_files(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name == "open":
            modes = [a for a in node.args[1:2]] + [k.value for k in node.keywords if k.arg == "mode"]
            if any(isinstance(m, ast.Constant) and any(c in str(m.value) for c in "wax") for m in modes):
                return True
        elif name in _WRITE_ATTRS and isinstance(node.func, ast.Attribute) and not (
                name == "replace" and node.args and isinstance(node.args[0], ast.Constant)):
            return True
    return False


def test_every_tools_module_that_writes_files_is_hooked_or_classified():
    unclassified = []
    for path in sorted((SRC / "tools").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not _writes_files(tree):
            continue
        hooked = any(isinstance(n, ast.Call) and (getattr(n.func, "attr", None) or getattr(n.func, "id", None))
                     in _HOOKS for n in ast.walk(tree)) or any(
            isinstance(n, (ast.FunctionDef,)) and n.name in _HOOKS for n in ast.walk(tree))
        if not hooked and path.stem not in P.TOOL_WRITER_MODULES:
            unclassified.append(path.stem)
    assert not unclassified, f"writers with no change-log hook and no classification: {unclassified}"


def test_the_registry_watches_the_project_folder_before_a_watched_writer_runs(store, monkeypatch, tmp_path):
    assert set(P.PROJECT_FOLDER_WRITERS) >= {"report_pdf", "probe_model", "run"}
    pid = store.create_project("Kosmas report", kind="GENERAL", goal="g")
    ws = store.ensure_workspace(pid)
    seen = []
    monkeypatch.setattr(P, "prepare_project_write", lambda ctx, path=None, project_id="", evidence=False:
                        seen.append(project_id))
    wrapped = P.with_project_write_watch(lambda **k: "ran", _ctx(store, pid), lambda: ws)
    assert wrapped(x=1) == "ran" and seen == [pid]


def test_a_background_jobs_files_are_logged_by_its_wake_turn(store, monkeypatch):
    pid = store.create_project("Elena data", kind="CODING", goal="g")
    ws = pathlib.Path(store.ensure_workspace(pid))
    (ws / "a.py").write_text("print(1)\n")
    MP.watch_project(store, pid, rid="req-elena-job", ask="run the import", write=True, evidence=True)
    (ws / "run.sh").write_text("python a.py > out.csv\n")
    MP.note_change_write(store, pid, "run.sh", ask="run the import", rid="req-elena-job")
    MP.flush_change_log("req-elena-job", reply="Started the import in the background.")
    for st in MP._END_STATES["req-elena-job"].values():      # the request ended ten minutes ago …
        st["t"] -= 600
    (ws / "out.csv").write_text("a,b\n1,2\n")                 # … and the job wrote five minutes ago
    os.utime(ws / "out.csv", (time.time() - 300, time.time() - 300))
    monkeypatch.setattr(P, "_job_wake_starter", lambda ctx: "req-elena-job")
    _real = time.time
    late = _real() + 600                                      # the wake runs ten minutes after the job wrote
    with monkeypatch.context() as mp, _rid("job-1a2b3c4d"):   # the job's wake turn starts
        mp.setattr(MP.time, "time", lambda: late)
        P.begin_request_watch(_ctx(store), "job-1a2b3c4d", "background job finished")
    MP.flush_change_log("job-1a2b3c4d", reply="The import finished.")
    rows = store.list_changes(pid, kind="change")
    assert any("out.csv" in (r["added"] + r["changed"]) and r["req_id"] == "job-1a2b3c4d" for r in rows), rows
    assert "out.csv" in store.get_file_manifest(pid)          # P11: an added file enters the manifest


# ── P10 / P11 / P18 / P19: the change log and the work log ──────────────────

def test_a_change_log_overwritten_without_its_header_is_kept_and_rendered_again(store):
    pid = store.create_project("Markos app", kind="CODING", goal="g")
    ws = pathlib.Path(store.ensure_workspace(pid))
    store.add_change(pid, ask="add a page", added=["page.html"], summary="Page added.")
    (ws / "PROJECT_CHANGES.md").write_text("my own notes\n")
    store.add_change(pid, ask="add another", added=["two.html"], summary="Two added.")
    text = (ws / "PROJECT_CHANGES.md").read_text()
    assert text.startswith(MP.CHANGES_FILE_HEADER.splitlines()[0]) and "two.html" in text
    assert "my own notes" in (ws / MP.CHANGES_FOREIGN_NAME).read_text()


def test_the_briefing_lists_changes_not_deploys_and_added_files_enter_the_manifest(store):
    pid = store.create_project("Elena site", kind="CODING", goal="g")
    store.add_change(pid, ask="add tetris", added=["js/tetris.js"], summary="Tetris added.")
    for i in range(6):
        store.add_change(pid, kind="deploy", ask=f"push {i}", changed=["/"], summary="pushed")
    lines = MP.recent_change_lines(store, pid, 5)
    assert lines and all("deploy" not in ln.lower().split(":")[0] for ln in lines) and any("tetris" in ln
                                                                                         for ln in lines)
    MP._manifest_added(store, pid, {"num": 1, "added": ["js/tetris.js"], "summary": "Tetris added."})
    assert "js/tetris.js" in store.get_file_manifest(pid)


def test_a_long_summary_keeps_its_ellipsis_within_the_limit():
    reply = ("I added the panel for Kosmas. " * 5) + ("and then a very long sentence without an end " * 10)
    s = MP._reply_summary(reply)
    assert len(s) <= MP.CHANGE_SUMMARY_MAX and s.endswith("…")
    s = MP._reply_summary("panel " * 120)                     # no sentence end: cut at a word, ellipsis kept
    assert len(s) <= MP.CHANGE_SUMMARY_MAX and s.endswith("panel…")


def test_work_log_paths_are_project_relative(store):
    pid = store.create_project("Paths", kind="CODING", goal="g")
    store.add_work_log(pid, request="x", files=[f"projects/{pid}/js/a.js", f"/workspace/projects/{pid}/b.js",
                                                "js/c.js"])
    files = store.list_events(pid, event_type="work_log")[0]["payload"]["files"]
    assert sorted(files) == ["b.js", "js/a.js", "js/c.js"]


def test_get_says_ready_for_a_ready_project(store):
    pid = store.create_project("Ready one", kind="CODING", goal="g")
    t = store.add_task(pid, "Build → a.py")
    store.update_task(t, status="DONE", result_summary="ok")
    b = P._briefing(store, pid)["project"]
    assert (b["status"], b["state"]) == ("READY", "ready")


# ── P12: the briefing of a project with nothing open ────────────────────────

def test_a_ready_projects_briefing_has_no_pacing_and_no_already_active(store):
    from ghost_agent.core.prompts import build_project_briefing
    pid = store.create_project("Elena shop", kind="CODING", goal="a shop")
    t = store.add_task(pid, "Build → index.html")
    ws = pathlib.Path(store.ensure_workspace(pid))
    (ws / "index.html").write_text("<p>shop</p>")
    store.register_file_artifact(t, "index.html", description="the shop page")
    store.update_task(t, status="DONE", result_summary="ok")
    b = build_project_briefing(store, pid)
    assert "PACING" not in b and "ALREADY ACTIVE" not in b
    files = [ln for ln in b.splitlines() if ln.startswith("FILES (on disk now)")]
    assert not files or "index.html" not in files[0]
    store.add_task(pid, "More → more.js")                     # open work again: the pacing rule is back
    assert "PACING" in build_project_briefing(store, pid)


# ── P13 / P14 / P15 / P16 / P17 / P20 / P22 ─────────────────────────────────

async def test_task_add_kind_is_the_tasks_kind(store):
    pid = store.create_project("Kinds", kind="GENERAL", goal="g")
    with _rid("req-elena-k"):
        await P.tool_manage_projects_for_model(_ctx(store, pid), action="task_add", project_id=pid,
                                               description="Look it up", kind="research")
    (t,) = store.list_tasks(pid)
    assert str(t.get("kind") or "").lower() == "research"


async def test_two_closes_in_one_batch_land_in_dependency_order(store):
    pid = store.create_project("Order", kind="GENERAL", goal="g")
    a = store.add_task(pid, "First")
    b = store.add_task(pid, "Second")
    store.update_task(b, depends_on=[a])
    ctx = _ctx(store, pid)
    with _rid("req-elena-order"):
        outs = await asyncio.gather(
            P.tool_manage_projects_for_model(ctx, action="task_update", task_id=b, status="DONE",
                                             result="second done"),
            P.tool_manage_projects_for_model(ctx, action="task_update", task_id=a, status="DONE",
                                             result="first done"))
    assert store.get_task(a)["status"] == "DONE" and store.get_task(b)["status"] == "DONE", [str(o) for o in outs]


def test_a_task_set_running_and_left_goes_back_at_the_requests_end(store):
    pid = store.create_project("Claims", kind="GENERAL", goal="g")
    t = store.add_task(pid, "Do it")
    store.update_task(t, status="IN_PROGRESS")
    MP.note_request_claim(store, "req-elena-claim", t)
    MP.flush_deferred_sweeps("req-elena-claim")
    assert store.get_task(t)["status"] != "IN_PROGRESS"


def test_a_stopped_batch_leaves_what_it_finished_on_the_record(store):
    pid = store.create_project("Stop", kind="GENERAL", goal="g")
    with pytest.raises(asyncio.CancelledError):
        with P._record_batch_stop(store, pid, time.time() - 5):
            raise asyncio.CancelledError()
    assert store.list_events(pid, event_type="batch_stopped")
    assert any("stopped" in str((e.get("payload") or {}).get("outcome") or "")
               for e in store.list_events(pid, event_type="work_log"))


def test_a_deleted_projects_folder_goes_to_the_trash(store):
    pid = store.create_project("Delete me", kind="CODING", goal="g")
    ws = pathlib.Path(store.ensure_workspace(pid))
    (ws / "a.py").write_text("print('Elena')\n")
    store.delete_project(pid, hard=True)
    assert not ws.exists()
    from ghost_agent.core.workspace_cleanup import project_trash_dir
    kept = list(project_trash_dir(store).rglob("a.py"))
    assert kept and kept[0].read_text() == "print('Elena')\n"


def test_an_unclosed_owner_task_turn_is_not_failed():
    assert PT.TASK_TURN_UNCLOSED_PREFIX
    from ghost_agent.core import project_advancer as PA
    assert PA._task_turn_unclosed_prefix() == PT.TASK_TURN_UNCLOSED_PREFIX
    assert "task_unclosed" in P._ADVANCE_STOP_BLURB


async def test_a_batch_whose_tasks_closed_done_is_not_a_failure(store, monkeypatch):
    from ghost_agent.core import project_advancer as PA
    pid = store.create_project("Elena tracker", kind="GENERAL", goal="g")
    old = store.add_task(pid, "Old part")
    store.update_task(old, status="FAILED", failure_reason="earlier run")
    new = store.add_task(pid, "New part")

    async def fake_many(*a, **k):
        store.update_task(new, status="DONE", result_summary="ok")
        return PA.AdvanceManyResult([{"task_id": new, "status": "DONE", "summary": "ok"}], "project_failed", None)
    monkeypatch.setattr(PA, "advance_many", fake_many)
    with _rid("req-elena-batch2"):
        out = await P.tool_manage_projects_for_model(_ctx(store, pid), action="autoadvance", go_ahead=True)
    assert getattr(out, "status", OutcomeStatus.OK) is OutcomeStatus.OK, out
    assert json.loads(str(getattr(out, "text", out)))["failed_tasks"]


# ── pins added after the mutation sweep ────────────────────────────────────

async def test_a_new_message_from_the_user_reopens_plan_only(store, monkeypatch):
    """Review 3: the new message re-opens the decision; the model's go_ahead=true on its next call makes it."""
    from ghost_agent.core import project_advancer as PA
    from ghost_agent.utils.provenance import note_added_user_message

    async def fake_many(*a, **k):
        return PA.AdvanceManyResult([], "count_reached", None)
    monkeypatch.setattr(PA, "advance_many", fake_many)
    ctx = _ctx(store)
    with _rid("req-elena-mid"):
        await P.tool_manage_projects_for_model(ctx, action="create", title="Elena list",
                                               subtasks=["A → a.js", "B → b.js"])
        assert _code(await P.tool_manage_projects_for_model(ctx, action="autoadvance")) == "plan_only_request"
        note_added_user_message("ok, build it now", req_id="req-elena-mid")
        assert _code(await P.tool_manage_projects_for_model(ctx, action="autoadvance",
                                                            go_ahead=True)) != "plan_only_request"


def test_a_task_turns_plan_call_does_not_decide_for_the_owner_request():
    from ghost_agent.memory.projects import set_request_parent
    set_request_parent("sub-ptask-77aa88bb", "req-elena-root")
    with _rid("sub-ptask-77aa88bb"):
        P._note_plan_call(False)                              # a task's turn splits its task: plan only
    with _rid("req-elena-root"):
        assert P._plan_only_request() == ""


async def test_a_task_a_background_request_set_running_goes_back_at_its_end(store):
    # §4UY P16 (review): only a background / scheduled / sub-agent request's own mark is taken back
    pid = store.create_project("Claims2", kind="GENERAL", goal="g")
    t = store.add_task(pid, "Do it")
    with _rid("sched-nightly-41"):
        await P.tool_manage_projects_for_model(_ctx(store, pid), action="task_update", task_id=t,
                                               status="IN_PROGRESS")
    assert store.get_task(t)["status"] == "IN_PROGRESS"
    MP.flush_deferred_sweeps("sched-nightly-41")
    assert store.get_task(t)["status"] != "IN_PROGRESS"


async def test_owner_marks_task_in_progress_is_kept(store):
    pid = store.create_project("Elena shop", kind="GENERAL", goal="g")
    tid = store.add_task(pid, "Photo shoot (owner does it)")
    with _rid("req-elena-ip"):
        out = await P.tool_manage_projects_for_model(_ctx(store, pid), action="task_update", task_id=tid,
                                                     status="IN_PROGRESS")
    assert store.get_task(tid)["status"] == "IN_PROGRESS", out
    MP.flush_deferred_sweeps("req-elena-ip")
    assert store.get_task(tid)["status"] == "IN_PROGRESS", store.get_task(tid)


async def test_task_turn_self_mark_releases_the_advancers_claim(store):
    pid = store.create_project("Markos app", kind="CODING", goal="g")
    tid = store.add_task(pid, "Build → a.py")
    store.update_task(tid, status="IN_PROGRESS")          # the advancer's claim
    rid = "sub-ptask-1234abcd"
    with _rid(rid):
        await P.tool_manage_projects(_ctx(store, pid), action="task_update", task_id=tid, status="IN_PROGRESS")
    MP.flush_deferred_sweeps(rid)                          # handle_chat's finally for the task turn
    assert store.get_task(tid)["status"] == "IN_PROGRESS", store.get_task(tid)


async def test_a_task_turn_never_registers_a_request_claim(store):
    """A task turn's RUNNING mark on a task that was still PENDING (a sibling it starts on) is not taken back at
    the turn's end either — the batch, not the turn's request, owns what is running."""
    pid = store.create_project("Markos app2", kind="CODING", goal="g")
    sib = store.add_task(pid, "Write tests → test_a.py")
    rid = "sub-ptask-5678ef01"
    with _rid(rid):
        await P.tool_manage_projects(_ctx(store, pid), action="task_update", task_id=sib, status="IN_PROGRESS")
    MP.flush_deferred_sweeps(rid)
    assert store.get_task(sib)["status"] == "IN_PROGRESS", store.get_task(sib)


async def test_a_background_mark_of_an_owner_kind_or_already_running_task_stays(store):
    pid = store.create_project("Kinds3", kind="GENERAL", goal="g")
    own = store.add_task(pid, "Sign the contract", kind="owner")
    run = store.add_task(pid, "Already running")
    store.update_task(run, status="IN_PROGRESS")
    with _rid("sched-nightly-42"):
        for t in [x for x in (own, run) if x]:
            await P.tool_manage_projects_for_model(_ctx(store, pid), action="task_update", task_id=t,
                                                   status="IN_PROGRESS")
    MP.flush_deferred_sweeps("sched-nightly-42")
    for t in [x for x in (own, run) if x]:
        assert store.get_task(t)["status"] == "IN_PROGRESS", t


async def test_a_task_turn_that_marks_its_task_running_is_not_rerun_by_the_batch(turns_on, monkeypatch, store):
    """Review MAJOR: the turn's own RUNNING mark was a request claim; its handle_chat end put the task back to
    PENDING before `_finish` read it — no unclosed stop, and `count=all` ran the same task again and again."""
    from ghost_agent.core import coding_loop as cl
    from ghost_agent.core.project_advancer import advance_many
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_4ul_autopilot_turns as H
    pid = store.create_project("Loop", kind="CODING", goal="g")
    tid = store.add_task(pid, "Build → a.py")
    runs = []

    async def fake(context, **kw):
        runs.append(kw.get("request_id"))
        kw["prepare_agent"](H._FakeTools([]))
        if len(runs) > 3:
            raise RuntimeError("the batch re-ran the same task")
        with _rid(kw["request_id"]):
            await P.tool_manage_projects(_ctx(store, pid), action="task_update", task_id=tid, status="IN_PROGRESS")
        MP.flush_deferred_sweeps(kw["request_id"])          # the turn's handle_chat finally
        return "working on it"
    monkeypatch.setattr(cl, "run_leaf_turn", fake)
    with _rid("req-elena-loop"):
        res = await advance_many(H._ctx(store), pid, max_tasks=None, owner_requested=True,
                                 tool_runner=lambda *a, **k: None, coding_executor=H._executor())
    assert len(runs) == 1 and res.stop_reason == "task_unclosed", (runs, res.stop_reason)


async def test_a_stopped_autoadvance_records_the_stop(store, monkeypatch):
    from ghost_agent.core import project_advancer as PA
    pid = store.create_project("Stop2", kind="GENERAL", goal="g")
    store.add_task(pid, "One")

    async def fake_many(*a, **k):
        raise asyncio.CancelledError()
    monkeypatch.setattr(PA, "advance_many", fake_many)
    with _rid("req-elena-stop"), pytest.raises(asyncio.CancelledError):
        await P.tool_manage_projects_for_model(_ctx(store, pid), action="autoadvance", go_ahead=True)
    assert store.list_events(pid, event_type="batch_stopped")


def test_a_stamp_after_an_older_finish_is_not_the_first_plan(store):
    legacy = time.time() - 86400 * 4
    pid = _legacy_project(store, legacy)
    store._atomic_metadata_update(pid, lambda m: {**m, "first_ready_at": time.time() - 60,
                                                  "first_ready_req": "req-elena-later"})
    assert MP._was_ready_before(store, pid, "req-elena-later", time.time())


def test_the_close_check_passes_the_tasks_creation_time(store):
    from ghost_agent.core.planning import ProjectPlan
    from ghost_agent.core.task_close import check_close
    pid = store.create_project("Old file", kind="GENERAL", goal="g")
    ws = pathlib.Path(store.ensure_workspace(pid))
    f = ws / "guide.md"
    f.write_text("# Guide for Markos\n\nA real document with enough words in it to not be hollow.\n" * 3)
    os.utime(f, (time.time() - 7200, time.time() - 7200))
    t = store.add_task(pid, "Write the guide → guide.md")
    v = check_close(store, pid, ProjectPlan(store, pid).tree, t, actor="model", result="done")
    assert not v.ok and v.refused[0].gate == "deliverable"


@pytest.fixture
def turns_on(monkeypatch):
    monkeypatch.setenv("GHOST_PROJECT_TASK_TURNS", "1")


async def test_a_task_turn_runs_under_its_outer_request_and_an_unclosed_one_stops_the_batch(turns_on, monkeypatch,
                                                                                           store):
    from ghost_agent.core import coding_loop as cl
    from ghost_agent.core.project_advancer import advance_many
    from ghost_agent.utils import logging as L
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_4ul_autopilot_turns as H
    pid = store.create_project("Kosmas parser", kind="CODING", goal="a parser")
    a = store.add_task(pid, "Lexer → lexer.py")
    store.add_task(pid, "Grammar → grammar.py")
    seen = []

    async def fake(context, **kw):
        seen.append(L._PROGRESS_PARENTS.get(str(kw.get("request_id") or "")))
        kw["prepare_agent"](H._FakeTools([]))
        return "I looked at the files."
    monkeypatch.setattr(cl, "run_leaf_turn", fake)
    with _rid("req-elena-outer2"):
        res = await advance_many(H._ctx(store), pid, max_tasks=None, owner_requested=True,
                                 tool_runner=lambda *a, **k: None, coding_executor=H._executor())
    assert seen and seen[0] == "req-elena-outer2"
    assert res.stop_reason == "task_unclosed" and len(seen) == 1
    assert store.get_task(a)["status"] not in ("FAILED", "IN_PROGRESS")


# ── review round: P9 reduce-only, P4 split / later go_ahead, P6 real runs ──

@pytest.mark.parametrize("kw,walled", [
    ({"action": "autopilot", "enabled": False}, False),
    ({"action": "budget", "metadata": {"unattended_per_day": 1}}, False),
    ({"action": "update", "status": "PAUSED"}, False),
    ({"action": "archive"}, True),                               # reversible, but not reduce-only: stays walled
    ({"action": "autopilot", "enabled": True}, True),
    ({"action": "budget", "metadata": {"unattended_per_day": 900}}, True),
    ({"action": "budget", "metadata": {"checkpoint_every": 0}}, True),
    ({"action": "update", "status": "PAUSED", "title": "Renamed"}, True),
    ({"action": "update", "status": "ACTIVE"}, True),
])
async def test_reduce_only_owner_actions_run_after_a_web_read(store, kw, walled):
    pid = store.create_project("Kosmas site", kind="CODING", goal="g")
    ctx = _ctx(store, pid)
    tok = RS.install(RS.RequestScope(conversation_key="conv", project_id=pid, origin="owner", context=ctx))
    try:
        from ghost_agent.utils.provenance import note_untrusted_content
        note_untrusted_content("web_search", req_id="req-elena-web2")
        with _rid("req-elena-web2"):
            out = await P.tool_manage_projects_for_model(ctx, project_id=pid, **kw)
    finally:
        RS.reset(tok)
    assert (_code(out) == "untrusted_content") is walled, (kw, str(out)[:300])


async def test_proceed_request_that_decomposes_a_task_first(store, monkeypatch):
    from ghost_agent.core import project_advancer as PA

    async def fake_many(*a, **k):
        return PA.AdvanceManyResult([], "count_reached", None)
    monkeypatch.setattr(PA, "advance_many", fake_many)
    pid = store.create_project("Elena blog", kind="CODING", goal="g")
    t = store.add_task(pid, "Build the site → index.html")
    ctx = _ctx(store, pid)
    with _rid("req-elena-proceed"):
        await P.tool_manage_projects_for_model(ctx, action="task_decompose", task_id=t,
                                               subtasks=["Layout → index.html", "Styles → style.css"])
        out = await P.tool_manage_projects_for_model(ctx, action="autoadvance", count="all")
    assert _code(out) != "plan_only_request", str(out)[:300]


async def test_a_later_go_ahead_with_no_new_message_does_not_lift_plan_only(store, monkeypatch):
    """Review 4 (decided strict, the fb249da8 shape): the model's own later go_ahead=true is no new decision."""
    from ghost_agent.core import project_advancer as PA

    async def fake_many(*a, **k):
        return PA.AdvanceManyResult([], "count_reached", None)
    monkeypatch.setattr(PA, "advance_many", fake_many)
    ctx = _ctx(store)
    with _rid("req-elena-plan-and-build"):
        await P.tool_manage_projects_for_model(ctx, action="create", title="Kosmas game",
                                               subtasks=["Board → board.js", "Rules → rules.js"])
        await P.tool_manage_projects_for_model(ctx, action="task_decompose",
                                               subtasks=["Sound → s.js", "Menu → m.js"], go_ahead=True)
        out = await P.tool_manage_projects_for_model(ctx, action="autoadvance", count="all")
    assert _code(out) == "plan_only_request", str(out)[:300]


def _p6_watch(tmp_path):
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    (root / "app.py").write_text("x=1\n")
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests/test_app.py").write_text("import app\n")
    (root / "run.sh").write_text("echo hi\n")
    w = PT._TurnWatch(root=root)
    w.note("file_system", {"operation": "write", "path": "app.py"}, ToolOutcome.ok("w"))
    return w


@pytest.mark.parametrize("cmd", [
    "python3 -m pytest -q", "python -m pytest", "python3 -m unittest discover", "bash run.sh -n 5",
    "cd /workspace/projects/proj && python3 -m pytest -q",
    "uvicorn app:app --port 8000 & sleep 2; curl -s localhost:8000",
])
def test_a_real_run_of_the_project_is_a_check(tmp_path, cmd):
    w = _p6_watch(tmp_path)
    w.note("execute", {"command": cmd}, OK)
    assert w.verified, cmd


@pytest.mark.parametrize("cmd", ["bash -n run.sh", "python3 -m py_compile app.py", "python3 -m json.tool x.json"])
def test_a_syntax_check_before_the_script_is_still_not_a_run(tmp_path, cmd):
    w = _p6_watch(tmp_path)
    w.note("execute", {"command": cmd}, OK)
    assert not w.verified, cmd



# ── review 2 ─────────────────────────────────────────────────────────────────

async def _no_batch(monkeypatch):
    from ghost_agent.core import project_advancer as PA
    ran = []

    async def fake_many(*a, **k):
        ran.append(1)
        return PA.AdvanceManyResult([], "count_reached", None)
    monkeypatch.setattr(PA, "advance_many", fake_many)
    return ran


async def test_break_it_down_split_with_go_ahead_false_is_plan_only(store, monkeypatch):
    """Review 2 (P4a): an explicit go_ahead=false on a split ("break task 1 down") is the plan-only decision."""
    ran = await _no_batch(monkeypatch)
    pid = store.create_project("Elena blog", kind="CODING", goal="g")
    t = store.add_task(pid, "Build the site → index.html")
    ctx = _ctx(store, pid)
    with _rid("req-elena-breakdown"):
        await P.tool_manage_projects_for_model(ctx, action="task_decompose", task_id=t, go_ahead=False,
                                               subtasks=["Layout → index.html", "Styles → style.css"])
        out = await P.tool_manage_projects_for_model(ctx, action="autoadvance", count="all")
    assert _code(out) == "plan_only_request" and not ran, str(out)[:200]


async def test_after_a_refused_autoadvance_a_go_ahead_true_does_not_lift_it(store, monkeypatch):
    """Review 2 (P4b): the model re-judged its own refusal with a go_ahead=true plan call — only a NEW user
    message lifts a refusal."""
    ran = await _no_batch(monkeypatch)
    ctx = _ctx(store)
    with _rid("req-elena-planonly"):
        await P.tool_manage_projects_for_model(ctx, action="create", title="Kosmas game",
                                               subtasks=["Board → board.js", "Rules → rules.js"])
        first = await P.tool_manage_projects_for_model(ctx, action="autoadvance", count="all")
        tid = store.list_tasks(ctx.current_project_id)[0]["id"]
        await P.tool_manage_projects_for_model(ctx, action="task_decompose", task_id=tid, go_ahead=True,
                                               subtasks=["Grid → grid.js", "Pieces → pieces.js"])
        await P.tool_manage_projects_for_model(ctx, action="task_decompose", go_ahead=True,
                                               subtasks=["Sound → s.js", "Menu → m.js"])
        out = await P.tool_manage_projects_for_model(ctx, action="autoadvance", count="all")
    assert _code(first) == "plan_only_request"
    assert _code(out) == "plan_only_request" and not ran, str(out)[:200]


def _pacing_briefing(store):
    """The PACING rule as the model reads it: the briefing of an active project with a task."""
    from ghost_agent.core.planning import ProjectPlan
    from ghost_agent.core.prompts import build_project_briefing
    pid = store.create_project("Elena pacing", goal="g")
    ProjectPlan(store, pid).add_task("task a")
    return " ".join(build_project_briefing(store, pid).split())


def test_the_pacing_prose_says_what_the_latch_does(store):
    b = _pacing_briefing(store)
    assert "refused for the rest of this request" not in b
    assert "plan-only decision is lifted only by go_ahead=true AFTER a NEW message from the user" in b
    desc = json.dumps(P.MANAGE_PROJECTS_TOOL_DEF["function"]["parameters"]["properties"]["go_ahead"])
    assert "true lifts plan only just after a new user message" in desc and "refused this request" not in desc


def _mod_root(tmp_path):
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    (root / "app.py").write_text("print(1)\n")
    (root / "mypkg").mkdir(exist_ok=True)
    (root / "mypkg" / "__init__.py").write_text("")
    (root / "mypkg" / "cli.py").write_text("print(2)\n")
    (root / "mypkg" / "srv").mkdir(exist_ok=True)
    (root / "mypkg" / "srv" / "__main__.py").write_text("print(4)\n")
    (root / "src").mkdir(exist_ok=True)
    (root / "src" / "main.py").write_text("print(3)\n")
    return root


@pytest.mark.parametrize("cmd,runs", [
    ("python -m app", True), ("python3 -m mypkg.cli", True), ("python -m mypkg", True),
    ("cd src && python -m main", True), ("python3 -m mypkg.cli --verbose", True),
    ("python -m http.server 8000", False), ("python3 -m mypkg.nothere", False),    # no mypkg/nothere.py, mypkg/nothere/ or mypkg.py
    ("python3 -m json.tool x.json", False), ("python -m main", False),
    ("python3 -m mypkg.srv", True),                                       # the package mypkg/srv/
    ("python3 -m app.sub", True),                                         # app.py is the project's
])
def test_python_dash_m_of_a_project_module_is_a_run(tmp_path, cmd, runs):
    """Review 2 (P6): `python -m mod.sub` runs the project's code when mod/sub.py, the package mod/sub/ or
    mod.py is the project's."""
    root = _mod_root(tmp_path)
    assert PT.run_touches_project("execute", {"command": cmd}, "", root) is runs, cmd


def test_python_dash_m_of_a_module_written_this_turn_is_a_run(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    assert PT.run_touches_project("execute", {"command": "python -m tool.cli"}, "", root,
                                  known=("tool/cli.py",))
    assert not PT.run_touches_project("execute", {"command": "python -m tool.cli"}, "", root)


def _stale_project(store, kind=""):
    pid = store.create_project("Docs", kind="GENERAL", goal="g")
    ws = pathlib.Path(store.ensure_workspace(pid))
    f = ws / "guide.md"
    f.write_text("# Guide\n" + "A real document for Elena with enough words in it to not be hollow.\n" * 3)
    old = time.time() - 7200
    os.utime(f, (old, old))
    t = store.add_task(pid, "Review the guide → guide.md", **({"kind": kind} if kind else {}))
    return pid, ws, t


def _close(store, pid, t):
    from ghost_agent.core.planning import ProjectPlan
    from ghost_agent.core.task_close import check_close
    return check_close(store, pid, ProjectPlan(store, pid).tree, t, actor="model", result="checked")


@pytest.mark.parametrize("rid", ["sched-nightly-7", "sub-ptask-77aa88bb", "job-autopilot-3"])
def test_a_background_close_of_a_stale_declared_file_is_refused(store, rid):
    """Review 2 (P7 iii): the refusal stays for unattended / background closes."""
    pid, ws, t = _stale_project(store)
    with _rid(rid):
        v = _close(store, pid, t)
    assert not v.ok and v.refused[0].gate == "deliverable" and v.refused[0].sub == "stale", rid


def test_the_owners_close_of_a_stale_declared_file_stands_with_a_note(store):
    """Review 2 (P7 ii): in the owner's own turn the stale check is advisory (a fuzzy judge must not gate)."""
    from ghost_agent.core.task_close import STALE_NOTE
    pid, ws, t = _stale_project(store)
    with _rid("req-elena-next"):
        v = _close(store, pid, t)
    assert v.ok and not v.refused and v.notes and STALE_NOTE in v.notes[0].reason


async def test_the_owners_task_update_close_carries_the_note(store):
    pid, ws, t = _stale_project(store)
    with _rid("req-elena-mark-done"):
        out = await P.tool_manage_projects(_ctx(store, pid), action="task_update", task_id=t, status="DONE",
                                           result="checked the guide")
    assert store.get_task(t)["status"] == "DONE", str(out)[:300]
    assert "predates this task" in str(out) and "close_note" in str(out), str(out)[:500]


async def test_a_refused_close_carries_no_note(store):
    pid, ws, t = _stale_project(store)
    (ws / "guide.md").write_text("")                       # hollow now: refused by the file check itself
    with _rid("req-elena-mark-done2"):
        out = await P.tool_manage_projects(_ctx(store, pid), action="task_update", task_id=t, status="DONE",
                                           result="checked")
    assert store.get_task(t)["status"] != "DONE" and "close_note" not in str(out)


def test_an_owner_kind_task_has_no_stale_check(store):
    """Review 2 (P7 i): an owner-kind task is the owner's own doing — even a background close of it."""
    pid, ws, t = _stale_project(store, kind="owner")
    with _rid("sched-nightly-8"):
        v = _close(store, pid, t)
    assert v.ok and not v.notes, v


@pytest.mark.parametrize("name,kw,res", [
    ("execute", {"command": "cd src && python main.py"}, "EXIT CODE: 0\nSTDOUT/STDERR:\nhello"),
    ("execute", {"command": "cd /workspace/projects/{pid}/src && cat main.py"}, "EXIT CODE: 0\nSTDOUT/STDERR:\n"),
    ("file_system", {"operation": "read", "path": "projects/{pid}/src/main.py"}, ToolOutcome.ok("print")),
    ("file_system", {"operation": "read", "path": "/workspace/projects/{pid}/src/main.py"}, ToolOutcome.ok("p")),
    ("file_system", {"operation": "read", "path": "src/main.py"},
     "--- src/main.py CONTENTS ---\nError: codes the service returns\n"),
    ("file_system", {"operation": "read", "path": "src/main.py"},
     "--- src/main.py CONTENTS ---\nEXIT CODE: 1\nprevious run output\n"),
    ("execute", {"command": "python3 -c 'import base64; base64.b64decode(\"" + "A" * 300 + "\")' && cat src/main.py"},
     "EXIT CODE: 0\nSTDOUT/STDERR:\nok"),
])
def test_the_evidence_readers_see_the_call_on_the_file(store, name, kw, res):
    """Review 2 (P7 iv): a leading cd, `projects/<id>/…` paths, a read whose CONTENT looks like an error, an
    over-long word in the command — each is this request's evidence for the file."""
    from ghost_agent.core.task_deliverable import note_tool_evidence, touched_in
    pid = store.create_project("Ev", kind="CODING", goal="g")
    ws = pathlib.Path(store.ensure_workspace(pid))
    (ws / "src").mkdir()
    (ws / "src" / "main.py").write_text("print('hello Elena')\n")
    kw = {k: (v.replace("{pid}", ws.name) if isinstance(v, str) else v) for k, v in kw.items()}
    rid = f"req-ev-{abs(hash((name, str(kw)))) % 10**8}"
    note_tool_evidence(rid, name, kw, res, ws)
    assert str((ws / "src" / "main.py").resolve()) in touched_in(rid), (name, kw)


@pytest.mark.parametrize("name,res", [
    ("file_system", "Error: failed to read 'src/main.py': permission denied"),
    ("file_system", ToolOutcome.failed("Error: no such file", reason_code="x")),
    ("execute", "EXIT CODE: 1\nSTDOUT/STDERR:\nboom"),
])
def test_a_failed_call_is_no_evidence(store, name, res):
    from ghost_agent.core.task_deliverable import note_tool_evidence, touched_in
    pid = store.create_project("Ev2", kind="CODING", goal="g")
    ws = pathlib.Path(store.ensure_workspace(pid))
    (ws / "src").mkdir()
    (ws / "src" / "main.py").write_text("print(1)\n")
    kw = {"operation": "read", "path": "src/main.py"} if name == "file_system" else {"command": "python src/main.py"}
    rid = f"req-ev-fail-{abs(hash((name, str(res)))) % 10**8}"
    note_tool_evidence(rid, name, kw, res, ws)
    assert not touched_in(rid)


# ── review 3 ─────────────────────────────────────────────────────────────────
# P4: the `_Pacing` table, one pin per row.

async def _adv(ctx, **kw):
    return await P.tool_manage_projects_for_model(ctx, action="autoadvance", count="all", **kw)


async def _plan(ctx, **kw):
    return await P.tool_manage_projects_for_model(ctx, action="create", title="Elena blog",
                                                  subtasks=["A → a.js", "B → b.js"], **kw)


async def test_pacing_row_request_start_drops_the_record(store, monkeypatch):
    """Row 'request starts' (reviewer B): a NEW request under a reused id inherits nothing."""
    from ghost_agent.utils.provenance import note_user_message
    ran = await _no_batch(monkeypatch)
    ctx = _ctx(store)
    with _rid("probe-fixed-id"):
        note_user_message("plan a blog", req_id="probe-fixed-id")
        await _plan(ctx)
        assert _code(await _adv(ctx)) == "plan_only_request"
    with _rid("probe-fixed-id"):
        note_user_message("do all", req_id="probe-fixed-id")
        out = await _adv(ctx)
    assert _code(out) != "plan_only_request" and ran, str(out)[:200]


async def test_pacing_row_plan_call_go_ahead_true_is_go(store, monkeypatch):
    ran = await _no_batch(monkeypatch)
    ctx = _ctx(store)
    with _rid("req-r3-go"):
        await _plan(ctx, go_ahead=True)
        out = await _adv(ctx)
    assert _code(out) != "plan_only_request" and ran


async def test_pacing_row_plan_call_go_ahead_false_after_a_new_message_is_plan_only(store, monkeypatch):
    """Row 'go_ahead=false' (reviewer D): an explicit plan-only decision after a new message stands."""
    from ghost_agent.utils.provenance import note_added_user_message
    ran = await _no_batch(monkeypatch)
    ctx = _ctx(store)
    with _rid("req-r3-d"):
        await _plan(ctx)
        note_added_user_message("also plan a settings page", req_id="req-r3-d")
        await P.tool_manage_projects_for_model(ctx, action="task_decompose", go_ahead=False,
                                               subtasks=["Settings → settings.js", "Prefs → prefs.js"])
        out = await _adv(ctx)
    assert _code(out) == "plan_only_request" and not ran


async def test_pacing_row_go_ahead_false_overrides_an_earlier_go(store, monkeypatch):
    ran = await _no_batch(monkeypatch)
    ctx = _ctx(store)
    with _rid("req-r3-go-then-false"):
        await _plan(ctx, go_ahead=True)
        await P.tool_manage_projects_for_model(ctx, action="task_decompose", go_ahead=False,
                                               subtasks=["Settings → settings.js", "Prefs → prefs.js"])
        out = await _adv(ctx)
    assert _code(out) == "plan_only_request" and not ran


async def test_pacing_row_omitted_keeps_go_after_a_batch_ran(store, monkeypatch):
    """Rows 'omitted keeps go' + 'an autoadvance that ran sets go' (reviewer C): "do all" → a mid-batch replan
    with go_ahead omitted → the next autoadvance runs."""
    ran = await _no_batch(monkeypatch)
    pid = store.create_project("Kosmas game", kind="CODING", goal="g")
    store.add_task(pid, "Board → board.js")
    store.add_task(pid, "Rules → rules.js")
    ctx = _ctx(store, pid)
    with _rid("req-r3-c"):
        first = await _adv(ctx)
        await P.tool_manage_projects_for_model(ctx, action="replan",
                                               subtasks=["Board v2 → board2.js", "Rules v2 → rules2.js"])
        out = await _adv(ctx)
    assert _code(first) != "plan_only_request" and _code(out) != "plan_only_request" and len(ran) == 2


async def test_pacing_row_autoadvance_go_ahead_true_with_no_new_message_is_refused(store, monkeypatch):
    """Row 'autoadvance' (review 4, reviewer B — fb249da8): a plan-only decision re-judged by the model with
    go_ahead=true on autoadvance, no word from the user since → refused."""
    ran = await _no_batch(monkeypatch)
    ctx = _ctx(store)
    with _rid("req-r4-b"):
        await _plan(ctx, go_ahead=False)
        out = await _adv(ctx, go_ahead=True)
    assert _code(out) == "plan_only_request" and not ran


async def test_pacing_row_go_ahead_true_after_a_new_message_lifts_plan_only(store, monkeypatch):
    from ghost_agent.utils.provenance import note_added_user_message
    ran = await _no_batch(monkeypatch)
    ctx = _ctx(store)
    with _rid("req-r4-lift"):
        await _plan(ctx)
        note_added_user_message("ok, build it", req_id="req-r4-lift")
        await P.tool_manage_projects_for_model(ctx, action="task_decompose", go_ahead=True,
                                               subtasks=["Sound → s.js", "Menu → m.js"])
        out = await _adv(ctx)
    assert _code(out) != "plan_only_request" and ran


async def test_a_message_before_an_explicit_false_does_not_lift_it(store, monkeypatch):
    """go_ahead=false after a message is a new decision: the message before it does not lift it."""
    from ghost_agent.utils.provenance import note_added_user_message
    ran = await _no_batch(monkeypatch)
    ctx = _ctx(store)
    with _rid("req-r4-msg-false"):
        note_added_user_message("plan it only", req_id="req-r4-msg-false")
        await _plan(ctx, go_ahead=False)
        out = await _adv(ctx, go_ahead=True)
    assert _code(out) == "plan_only_request" and not ran


async def test_an_omitted_plan_call_keeps_the_message_count_of_plan_only(store, monkeypatch):
    """Row 'omitted, already plan_only': n is kept — a message before a second omitted plan call still counts."""
    from ghost_agent.utils.provenance import note_added_user_message
    ran = await _no_batch(monkeypatch)
    ctx = _ctx(store)
    with _rid("req-r4-keep-n"):
        await _plan(ctx)
        note_added_user_message("ok go", req_id="req-r4-keep-n")
        await P.tool_manage_projects_for_model(ctx, action="task_decompose",
                                               subtasks=["Sound → s.js", "Menu → m.js"])
        out = await _adv(ctx, go_ahead=True)
    assert _code(out) != "plan_only_request" and ran


async def test_create_go_ahead_false_then_task_add_is_plan_only(store, monkeypatch):
    """Review 4 (reviewer A): create records an explicit go_ahead without subtasks too; task_add is no plan
    call."""
    from ghost_agent.utils.provenance import note_user_message
    ran = await _no_batch(monkeypatch)
    ctx = _ctx(store)
    with _rid("req-rev4-a"):
        note_user_message("make a project for Elena's blog and add the tasks; don't build yet", req_id="req-rev4-a")
        await P.tool_manage_projects_for_model(ctx, action="create", title="Elena blog", go_ahead=False)
        pid = ctx.current_project_id
        await P.tool_manage_projects_for_model(ctx, action="task_add", project_id=pid,
                                               description="Layout → index.html")
        await P.tool_manage_projects_for_model(ctx, action="task_add", project_id=pid,
                                               description="Styles → style.css")
        out = await _adv(ctx)
    assert _code(out) == "plan_only_request" and not ran, str(out)[:300]


async def test_create_with_no_subtasks_and_no_go_ahead_records_nothing(store, monkeypatch):
    ran = await _no_batch(monkeypatch)
    ctx = _ctx(store)
    with _rid("req-rev4-none"):
        await P.tool_manage_projects_for_model(ctx, action="create", title="Markos notes")
        await P.tool_manage_projects_for_model(ctx, action="task_add", project_id=ctx.current_project_id,
                                               description="Notes → notes.md")
        out = await _adv(ctx)
    assert _code(out) != "plan_only_request" and ran


async def test_pacing_row_autoadvance_go_ahead_true_after_a_refusal_is_refused(store, monkeypatch):
    ran = await _no_batch(monkeypatch)
    ctx = _ctx(store)
    with _rid("req-r3-adv-refused"):
        await _plan(ctx)
        assert _code(await _adv(ctx)) == "plan_only_request"
        out = await _adv(ctx, go_ahead=True)
    assert _code(out) == "plan_only_request" and not ran


async def test_pacing_row_a_new_message_reopens_but_does_not_decide(store, monkeypatch):
    """Row 'user message added mid-turn' (reviewer A): a message alone decides nothing ("don't start yet" keeps
    plan_only); after "ok go" the model's go_ahead=true is heard."""
    from ghost_agent.utils.provenance import note_added_user_message
    ran = await _no_batch(monkeypatch)
    ctx = _ctx(store)
    with _rid("req-r3-a"):
        await _plan(ctx)
        assert _code(await _adv(ctx)) == "plan_only_request"
        note_added_user_message("don't start anything yet, just show me the plan", req_id="req-r3-a")
        assert _code(await _adv(ctx)) == "plan_only_request" and not ran
        note_added_user_message("ok go", req_id="req-r3-a")
        out = await _adv(ctx, go_ahead=True)
    assert _code(out) != "plan_only_request" and ran


def test_the_pacing_table_is_in_the_docstring_and_the_prose(store):
    doc = P._Pacing.__doc__
    for row in ("request starts", "go_ahead=true", "go_ahead=false", "go_ahead omitted", "split of a task",
                "autoadvance", "user message added mid-turn"):
        assert row in doc, row
    b = _pacing_briefing(store)
    assert "re-opens the decision without making it" in b and "autoadvance takes go_ahead too" in b
    assert "an omitted one keeps a go already given" in b
    desc = P.MANAGE_PROJECTS_TOOL_DEF["function"]["parameters"]["properties"]["go_ahead"]["description"]
    assert "autoadvance" in desc and "omitted keeps a go" in desc and "just after a new user message" in desc


# P7: the owner's own batch is the owner's turn.

def test_a_task_turn_of_the_owners_batch_closes_with_a_note(store):
    from ghost_agent.core.request_scope import ORIGIN_OWNER, RequestScope, install, reset
    from ghost_agent.memory.projects import set_request_parent
    pid, ws, t = _stale_project(store)
    set_request_parent("sub-ptask-r3owner-1", "a1b2c3d4")
    with _rid("sub-ptask-r3owner-1"):
        tok = install(RequestScope(origin=ORIGIN_OWNER))
        try:
            assert P._owner_task_turn(None)
            v = _close(store, pid, t)
        finally:
            reset(tok)
    assert v.ok and v.notes, v


def test_the_advancer_close_of_the_owners_batch_notes_an_unattended_one_refuses(store):
    from ghost_agent.core.planning import ProjectPlan
    from ghost_agent.core.task_close import check_close
    pid, ws, t = _stale_project(store)
    with _rid("sched-idle-3"):
        v_owner = check_close(store, pid, ProjectPlan(store, pid).tree, t, actor="advancer_owner", result="x")
        v_un = check_close(store, pid, ProjectPlan(store, pid).tree, t, actor="advancer_unattended", result="x")
    assert v_owner.ok and v_owner.notes
    assert not v_un.ok and v_un.refused[0].sub == "stale"


# P6: inline imports; a path only inside open(…) is read, not run.

def _r3_watch(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src/app.py").write_text("print(1)\n")
    (root / "mypkg").mkdir()
    (root / "mypkg/__init__.py").write_text("")
    (root / "mypkg/cli.py").write_text("")
    (root / "main.py").write_text("")
    (root / "app.py").write_text("x=1\n")
    w = PT._TurnWatch(root=root)
    w.note("file_system", {"operation": "write", "path": "src/app.py"}, ToolOutcome.ok("w"))
    return w


@pytest.mark.parametrize("cmd", ["python3 -c 'import mypkg.cli'", "python3 -c 'import main'",
                                 "python3 -c 'from mypkg.cli import x'", "python3 -c \"(import main)\"",
                                 "cd src && python -m app", "PYTHONPATH=src python -m app"])
def test_inline_code_that_imports_the_project_is_a_run(tmp_path, cmd):
    w = _r3_watch(tmp_path)
    w.note("execute", {"command": cmd}, OK)
    assert w.verified, cmd


@pytest.mark.parametrize("cmd", ["python3 -c 'import ast; ast.parse(open(\"app.py\").read())'",
                                 "python3 -c \"print(len(open('main.py').read()))\"",
                                 "python3 -c 'import json; print(1)'"])
def test_inline_code_that_only_opens_a_project_file_is_not_a_run(tmp_path, cmd):
    w = _r3_watch(tmp_path)
    w.note("execute", {"command": cmd}, OK)
    assert not w.verified, cmd


def test_inline_code_content_that_only_opens_a_file_is_not_a_run(tmp_path):
    root = _r3_watch(tmp_path).root
    assert not PT.run_touches_project("execute", {"content": "print(open('app.py').read())"}, "", root)
    assert PT.run_touches_project("execute", {"content": "import app\nprint(app.x)"}, "", root)


# P20: the delete's note says where the folder really went.

async def test_the_delete_note_says_removed_when_the_trash_was_unusable(store, tmp_path):
    from ghost_agent.core.workspace_cleanup import project_trash_dir
    pid = store.create_project("Elena tool", kind="CODING", goal="g")
    ws = pathlib.Path(store.ensure_workspace(pid))
    (ws / "a.py").write_text("x=1\n")
    tr = project_trash_dir(store)
    tr.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "elsewhere").mkdir()
    tr.symlink_to(tmp_path / "elsewhere")                      # the trash is unusable → removed
    with _rid("req-elena-del"):
        out = await P.tool_manage_projects_for_model(_ctx(store, pid), action="delete", project_id=pid)
    assert store.deleted_folder(pid) == "removed"
    assert "project trash (kept" not in str(out) and "could NOT be moved" in str(out), str(out)[:400]


async def test_the_delete_note_says_trash_when_it_went_there(store):
    pid = store.create_project("Markos tool", kind="CODING", goal="g")
    ws = pathlib.Path(store.ensure_workspace(pid))
    (ws / "a.py").write_text("x=1\n")
    with _rid("req-markos-del"):
        out = await P.tool_manage_projects_for_model(_ctx(store, pid), action="delete", project_id=pid)
    assert store.deleted_folder(pid) == "trash" and "went to the host's project trash" in str(out)


# P2: the reviewer's fixture (old code stamped the swallowing request).

def test_the_backfill_repairs_the_old_codes_stamp(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bf3", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "backfill_4uy_first_ready.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    legacy = time.time() - 86400 * 3
    pid = _legacy_project(store, legacy)
    ws = pathlib.Path(store.ensure_workspace(pid))
    t2 = store.add_task(pid, "Panel → js/panel.js")
    (ws / "js").mkdir(exist_ok=True)
    (ws / "js" / "panel.js").write_text("export const panel = 1;\n")
    store.add_work_log(pid, request="add the panel for Kosmas", files=["js/panel.js"], outcome="completed")
    with _rid("req-kosmas-7"):
        store.update_task(t2, status="DONE", result_summary="added")
    store._atomic_metadata_update(pid, lambda m: {**m, "first_ready_at": time.time(),
                                                  "first_ready_req": "req-kosmas-7"})
    with store._lock, store._connect() as conn:
        conn.execute("DELETE FROM project_changes WHERE project_id=?", (pid,))
        conn.commit()
    plan = bf.plan_repair(store, pid)
    assert plan["req"] == "req-kosmas-7" and plan["stamp"]["first_ready_req"] == ""
    assert "js/panel.js" in plan["entry"]["added"] + plan["entry"]["changed"]
    bf.apply_repair(store, plan)
    m = store.get_project(pid)["metadata"]
    assert m["first_ready_req"] == "" and abs(float(m["first_ready_at"]) - legacy) < 1
    (e,) = store.list_changes(pid, kind="change")
    assert e["req_id"] == "req-kosmas-7" and bf.plan_repair(store, pid) is None



# ── review 4: P6 — code that exec()s / compile()s what it opened runs it ─────

def _r4_watch(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "app.py").write_text("def main(f): print(f.read())\n")
    (root / "in.txt").write_text("x\n")
    (root / "app.js").write_text("console.log(1)\n")
    w = PT._TurnWatch(root=root)
    w.note("file_system", {"operation": "write", "path": "app.py"}, ToolOutcome.ok("w"))
    return w


@pytest.mark.parametrize("cmd", [
    """python3 -c 'import app; app.main(open("in.txt"))'""",
    """cd /workspace/projects/proj && python3 -c 'import app; app.main(open("in.txt"))'""",
    """python3 -c "exec(open('app.py').read())" """,
    """python3 -c "exec(compile(open('app.py').read(), 'app.py', 'exec'))" """,
    """python3 - <<'EOF'\nexec(open("app.py").read())\nEOF""",
    """python3 - <<'EOF'\nwith open("app.py") as f: exec(f.read())\nEOF""",
    """python3 -c "import runpy; runpy.run_path('app.py')" """,
    """python3 -c "from app import main; main(open('in.txt'))" """,
    """python3 app.py < in.txt""",
    """node -e "require('./app.js')" """,
])
def test_exec_of_an_opened_project_file_is_a_run(tmp_path, cmd):
    w = _r4_watch(tmp_path)
    w.note("execute", {"command": cmd}, OK)
    assert w.verified, cmd


def test_exec_in_inline_content_keeps_its_opened_path(tmp_path):
    root = _r4_watch(tmp_path).root
    assert PT.run_touches_project("execute", {"content": "with open('app.py') as f:\n    exec(f.read())"}, "",
                                  root)
    assert not PT.run_touches_project("execute", {"content": "print(open('app.py').read())"}, "", root)
