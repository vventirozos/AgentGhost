"""§4SX D5 review r1: the defects a fresh reader found in the `run` switch, each pinned through the real path
(journal §4SX-r).

M1 a leaf's promoted execute job is never stopped at leaf end · M2 the leaf tag is written IN the insert (a
cancelled leaf kept no untagged server) and survives a restart · M3 `run collect` reads the exit the way the wake
does and marks the job delivered · M4 an owner-less turn (a job's wake, a scheduled task) still reads its own job ·
M5 a landed job ends its run row, the RUNNING block shows the newest rows and keeps the STALE mark · minors: a leaf
never stops another's run, the workspace start runs execute's guards, the aliases' old words, the owner filter.
"""
from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_4sx_c_run_tool import (  # noqa: E402,F401
    JobSandbox, SvcSandbox, _ctx, _fast, _kill_jobs, happy_handler, scope)

from ghost_agent.core import lab_waits as W  # noqa: E402
from ghost_agent.core import request_scope as RS  # noqa: E402
from ghost_agent.memory.run_registry import RunRegistry, registry_for  # noqa: E402
from ghost_agent.tools import runs as R  # noqa: E402
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome  # noqa: E402


def _leaf(leaf_id):
    return W.LEAF_ID.set(leaf_id)


async def _land(c, name, tries=80):
    for _ in range(tries):
        out = await R.tool_run(c, action="collect", name=name)
        if out.status is not OutcomeStatus.UNRESOLVED:
            return out
        await asyncio.sleep(0.25)
    return out


# ------------------------------------------------------------------ M1

def test_a_promoted_job_inside_a_leaf_is_not_tagged_for_the_leafs_end(tmp_path, scope):
    c = _ctx(tmp_path)
    scope(c, origin="autopilot")
    tok = _leaf("t9-a1")
    try:
        assert R.record_promoted_job(c, {"id": "job-1a2b3c4d", "command": "pip install -r r.txt"}) == "job-1a2b3c4d"
    finally:
        W.LEAF_ID.reset(tok)
    (row,) = registry_for(c).rows(live_only=True)
    assert "leaf" not in RunRegistry.meta_of(row)


async def test_the_leafs_end_leaves_its_promoted_job_running(tmp_path, scope, monkeypatch):
    c = _ctx(tmp_path)
    scope(c, origin="autopilot")
    tok = _leaf("t9-a1")
    try:
        R.record_promoted_job(c, {"id": "job-1a2b3c4d", "command": "pytest -q"})
    finally:
        W.LEAF_ID.reset(tok)
    stopped = []

    async def fake_stop(context, reg, row, origin):
        stopped.append(row["name"])
        return ToolOutcome("ok", status=OutcomeStatus.OK)
    monkeypatch.setattr(R, "_stop", fake_stop)
    assert await R.stop_leaf_runs(c, "conversation:conv1", "t9-a1") == [] and stopped == []


# ------------------------------------------------------------------ M2

def test_the_registry_writes_meta_in_the_insert(tmp_path):
    reg = RunRegistry(tmp_path)
    rid = reg.add(name="a", place="workspace", lifetime="job", command="x", port=None, owner="o", origin="owner",
                  backend="jobs", backend_ref="j", meta={"leaf": "t1-a1", "gone": None})
    assert RunRegistry.meta_of(reg.get(rid)) == {"leaf": "t1-a1"}
    rid2 = reg.add(name="b", place="workspace", lifetime="job", command="x", port=None, owner="o",
                   origin="owner", backend="jobs", backend_ref="k")
    assert RunRegistry.meta_of(reg.get(rid2)) == {}


async def test_a_job_a_leaf_starts_carries_the_leaf_even_when_the_leaf_is_cancelled_mid_start(tmp_path, scope):
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb)
    scope(c, origin="autopilot")
    tok = _leaf("t3-a2")
    try:
        t = asyncio.ensure_future(R.tool_run(c, action="start", name="bench", command="sleep 20",
                                             where="workspace", lifetime="job"))
        await asyncio.sleep(0)
        t.cancel()                                   # the leaf's wait_for ends the turn mid-start
        with pytest.raises(asyncio.CancelledError):
            await t
        for _ in range(80):                          # the shielded start still records its row
            row = registry_for(c).live_by_name("conversation:conv1", "bench")
            if row is not None:
                break
            await asyncio.sleep(0.1)
        assert row is not None and RunRegistry.meta_of(row).get("leaf") == "t3-a2"
    finally:
        W.LEAF_ID.reset(tok)
        _kill_jobs(sb)


async def test_a_server_a_leaf_restarts_keeps_the_leaf_tag(tmp_path, scope):
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c, origin="autopilot")
    tok = _leaf("t4-a1")
    try:
        assert (await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace",
                                 lifetime="server")).status is OutcomeStatus.OK
        r = await R.tool_run(c, action="restart", name="web")
        assert r.status is OutcomeStatus.OK, str(r)
    finally:
        W.LEAF_ID.reset(tok)
    row = registry_for(c).live_by_name("conversation:conv1", "web")
    assert "restarted as" not in (row["state_note"] or "") and RunRegistry.meta_of(row).get("leaf") == "t4-a1"


async def test_a_start_outside_a_leaf_has_no_leaf_tag(tmp_path, scope):
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace", lifetime="server")
    assert "leaf" not in RunRegistry.meta_of(registry_for(c).live_by_name("conversation:conv1", "web"))


# ------------------------------------------------------------------ M3

async def test_collect_normalises_the_exit_like_the_wake_and_strips_the_marker(tmp_path, scope, monkeypatch):
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    import ghost_agent.tools.execute as EX
    seen = []

    def norm(cmd, code, out=""):
        seen.append((cmd, code))
        return 0                                     # e.g. a `| head` that closed the pipe early
    monkeypatch.setattr(EX, "_normalise_exit", norm)
    try:
        await R.tool_run(c, action="start", name="g", command="echo found-it; (exit 1)", where="workspace",
                         lifetime="job")
        out = await _land(c, "g")
        assert out.status is OutcomeStatus.OK and "EXIT CODE: 0" in out and "found-it" in out, str(out)
        assert seen and seen[0][1] == 1 and "PIPESTATUS" not in str(out)
    finally:
        _kill_jobs(sb)


async def test_collect_of_a_real_failure_stays_failed(tmp_path, scope):
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    try:
        await R.tool_run(c, action="start", name="f", command="echo nope; (exit 3)", where="workspace", lifetime="job")
        out = await _land(c, "f")
        assert out.status is OutcomeStatus.FAILED and "EXIT CODE: 3" in out, str(out)
    finally:
        _kill_jobs(sb)


async def test_a_collected_job_is_marked_delivered_so_its_wake_does_not_repeat_it(tmp_path, scope, monkeypatch):
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    woke = []
    real = SandboxJobSupervisor.mark_woken
    monkeypatch.setattr(SandboxJobSupervisor, "mark_woken", lambda self, jid: woke.append(jid) or real(self, jid))
    try:
        await R.tool_run(c, action="start", name="w", command="echo hi", where="workspace", lifetime="job")
        out = await _land(c, "w")
        assert out.status is OutcomeStatus.OK
        ref = registry_for(c).latest_by_name("conversation:conv1", "w")["backend_ref"]
        assert woke and woke[-1] == ref
    finally:
        _kill_jobs(sb)


# ------------------------------------------------------------------ M4

def _ownerless(tmp_path, sandbox=None):
    c = _ctx(tmp_path, sandbox=sandbox, conv="")
    tok = RS.install(RS.RequestScope(conversation_key="", project_id=None, origin="unattended", context=c))
    return c, tok


async def test_an_ownerless_turn_reads_and_cancels_its_job_through_the_jobs_rule(tmp_path, monkeypatch):
    import ghost_agent.tools.delegate as D
    calls = []

    async def fake_jobs(action=None, job_id=None, context=None, **kw):
        calls.append((action, job_id))
        return "jobs-said"
    monkeypatch.setattr(D, "tool_jobs", fake_jobs)
    c, tok = _ownerless(tmp_path)
    try:
        assert await R.tool_run(c, action="collect", name="job-1a2b3c4d") == "jobs-said"
        assert await R.tool_run(c, action="stop", name="job-1a2b3c4d") == "jobs-said"
        assert await R.tool_run(c, action="status") == "jobs-said"
        assert calls == [("collect", "job-1a2b3c4d"), ("cancel", "job-1a2b3c4d"), ("status", None)]
        r = await R.tool_run(c, action="start", name="web", command="x", where="workspace", lifetime="server")
        assert r.reason_code == "run_no_owner" and "switch" in r
        r = await R.tool_run(c, action="restart", name="job-1a2b3c4d")
        assert r.reason_code == "run_no_owner"
    finally:
        RS.reset(tok)


async def test_the_jobs_alias_in_an_ownerless_turn_falls_back_to_the_jobs_tool(tmp_path, monkeypatch, scope):
    import ghost_agent.tools.delegate as D
    owner_ctx = _ctx(tmp_path)
    scope(owner_ctx)
    R.record_promoted_job(owner_ctx, {"id": "job-1a2b3c4d"})            # the OWNER's row
    calls = []

    async def fake_jobs(action=None, job_id=None, context=None, **kw):
        calls.append((action, job_id))
        return "jobs-said"
    monkeypatch.setattr(D, "tool_jobs", fake_jobs)
    c, tok = _ownerless(tmp_path)
    try:
        assert await R.jobs_alias(context=c, action="collect", job_id="job-1a2b3c4d") == "jobs-said"
        assert calls == [("collect", "job-1a2b3c4d")]
    finally:
        RS.reset(tok)


def test_run_name_for_job_reads_only_the_callers_rows(tmp_path, scope):
    c = _ctx(tmp_path)
    scope(c)
    R.record_promoted_job(c, {"id": "job-1a2b3c4d"})
    assert R.run_name_for_job(c, "job-1a2b3c4d", "conversation:conv1") == "job-1a2b3c4d"
    assert R.run_name_for_job(c, "job-1a2b3c4d", "conversation:other") == ""


# ------------------------------------------------------------------ M5

def test_a_landed_job_ends_its_run_row(tmp_path, scope):
    c = _ctx(tmp_path, pid="41f745517fe4")
    scope(c)
    R.record_promoted_job(c, {"id": "job-1a2b3c4d"})
    R.record_promoted_job(c, {"id": "job-2a2b3c4d"})
    n = R.end_landed_job_runs(c, [{"id": "job-1a2b3c4d", "state": "done", "exit_code": 2},
                                  {"id": "job-2a2b3c4d", "state": "running"}, "junk"])
    assert n == 1
    reg = registry_for(c)
    ended = reg.latest_by_name("project:41f745517fe4", "job-1a2b3c4d")
    assert ended["state"] == "exited" and ended["state_note"] == "exit 2" and ended["ended_at"] is not None
    assert reg.live_by_name("project:41f745517fe4", "job-2a2b3c4d") is not None
    assert R.end_landed_job_runs(c, [{"id": "job-2a2b3c4d", "state": "cancelled"}]) == 1
    assert reg.latest_by_name("project:41f745517fe4", "job-2a2b3c4d")["state"] == "stopped"


def _calls_in(fn_node):
    import ast
    out = []
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Call):
            f = n.func
            out.append(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
            out += [a.id if isinstance(a, ast.Name) else a.attr for a in n.args
                    if isinstance(a, (ast.Name, ast.Attribute))]
    return out


def test_the_reaper_writes_landed_jobs_back():
    """R1 enumeration: the function that drains `take_unreported` also ends the landed jobs' run rows."""
    import ast
    tree = ast.parse(Path(R.__file__).parent.parent.joinpath("main.py").read_text())
    fns = [f for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
           and "take_unreported" in _calls_in(f)]
    assert fns and all("end_landed_job_runs" in _calls_in(f) for f in fns)


def test_the_running_summary_ends_a_landed_job_and_lists_the_newest(tmp_path, scope, monkeypatch):
    c = _ctx(tmp_path, pid="41f745517fe4")
    scope(c)
    reg = registry_for(c)
    own = "project:41f745517fe4"
    for i in range(14):
        rid = reg.add(name=f"s{i:02d}", place="workspace", lifetime="server", command="x", port=None, owner=own,
                      origin="owner", backend="services", backend_ref=f"k{i}")
        with reg._lock:
            with contextlib.closing(reg._connect()) as cn, cn:
                cn.execute("UPDATE runs SET started_at=? WHERE run_id=?", (1000.0 + i, rid))
    R.record_promoted_job(c, {"id": "job-1a2b3c4d"})

    class _Jobs:
        def list_entries(self):
            return [{"id": "job-1a2b3c4d", "state": "done", "exit_code": 0}]
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (None, _Jobs()))
    out = R.project_running_summary(c, "41f745517fe4")
    names = [r["name"] for r in out if r["kind"] == "run"]
    assert "job-1a2b3c4d" not in names                                 # it landed: not "running" any more
    assert names[0] == "s13" and len(names) == 12 and "s00" not in names
    assert reg.latest_by_name(own, "job-1a2b3c4d")["state"] == "exited"


def test_a_run_backed_server_keeps_the_stale_mark(tmp_path):
    from ghost_agent.core.prompts import build_project_briefing
    from ghost_agent.memory.projects import ProjectStore
    store = ProjectStore(tmp_path / "ps", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Shop")
    running = [{"kind": "run", "name": "web", "lifetime": "server", "place": "workspace", "state": "running",
                "at": None, "port": 8101, "service_key": "h:web"},
               {"kind": "run", "name": "api", "lifetime": "server", "place": "workspace", "state": "running",
                "at": None, "port": 8102, "service_key": "h:api"}]
    services = [{"name": "web", "key": "h:web", "port": 8101, "stale": True},
                {"name": "api", "key": "h:api", "port": 8102, "stale": False}]
    text = build_project_briefing(store, pid, services=services, running=running)
    web = next(ln for ln in text.splitlines() if "run web" in ln)
    api = next(ln for ln in text.splitlines() if "run api" in ln)
    assert "STALE" in web and "STALE" not in api and "SERVICES" not in text


# ------------------------------------------------------------------ minors

async def test_a_leaf_never_stops_or_restarts_a_run_it_did_not_start(tmp_path, scope, monkeypatch):
    c = _ctx(tmp_path)
    scope(c, origin="autopilot")
    reg = registry_for(c)
    own = "conversation:conv1"
    reg.add(name="owners", place="workspace", lifetime="server", command="x", port=None, owner=own, origin="owner",
            backend="services", backend_ref="k1")
    reg.add(name="other", place="workspace", lifetime="server", command="x", port=None, owner=own,
            origin="autopilot", backend="services", backend_ref="k2", meta={"leaf": "t2-a1"})
    reg.add(name="mine", place="workspace", lifetime="server", command="x", port=None, owner=own,
            origin="autopilot", backend="services", backend_ref="k3", meta={"leaf": "t1-a1"})
    hit = []

    async def fake_stop(context, reg_, row, origin):
        hit.append(row["name"])
        return ToolOutcome("stopped", status=OutcomeStatus.OK)
    monkeypatch.setattr(R, "_stop", fake_stop)
    tok = _leaf("t1-a1")
    try:
        for n in ("owners", "other"):
            for a in ("stop", "restart"):
                r = await R.tool_run(c, action=a, name=n)
                assert r.reason_code == "run_not_allowed", (n, a, str(r))
        assert (await R.tool_run(c, action="stop", name="mine")).status is OutcomeStatus.OK
    finally:
        W.LEAF_ID.reset(tok)
    assert hit == ["mine"]
    # outside a leaf, unattended work (a scheduled task the owner set up) still may
    assert (await R.tool_run(c, action="stop", name="owners")).status is OutcomeStatus.OK


async def test_a_workspace_start_runs_executes_guards(tmp_path, scope):
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb, host="10.20.30.40:8700")
    scope(c)
    r = await R.tool_run(c, action="start", name="x", command="curl http://10.20.30.40:8700/v1/labs",
                         where="workspace", lifetime="job")
    assert r.status is OutcomeStatus.REJECTED and r.reason_code == "lab_host_from_workspace"
    r = await R.tool_run(c, action="start", name="y", command="curl -s http://127.0.0.1:8000/api/health",
                         where="workspace", lifetime="server")
    assert r.status is OutcomeStatus.REJECTED and r.reason_code == "run_agent_port"
    assert registry_for(c).rows(live_only=True) == []


async def test_the_aliases_take_their_old_words(tmp_path, scope, monkeypatch):
    seen = []

    async def fake_run(context=None, **kw):
        seen.append(kw)
        return ToolOutcome("ok", status=OutcomeStatus.OK)
    monkeypatch.setattr(R, "tool_run", fake_run)
    c = _ctx(tmp_path)
    scope(c)
    await R.manage_services_alias(context=c, action="log", name="web")
    assert seen[-1] == {"action": "logs", "name": "web"}
    reg = registry_for(c)
    reg.add(name="job-1a2b3c4d", place="workspace", lifetime="job", command="x", port=None,
            owner="conversation:conv1", origin="owner", backend="jobs", backend_ref="job-1a2b3c4d")
    await R.jobs_alias(context=c, action="list", job_id="job-1a2b3c4d")
    assert seen[-1] == {"action": "status", "name": "job-1a2b3c4d"}


def test_a_promoted_job_is_listed_once_in_the_project_inventory(tmp_path, scope, monkeypatch):
    from ghost_agent.tools import projects as P
    c = _ctx(tmp_path, pid="41f745517fe4")
    scope(c)
    R.record_promoted_job(c, {"id": "job-1a2b3c4d"})

    class _Sup:
        def list_entries(self):
            return [{"id": "job-1a2b3c4d", "project_id": "41f745517fe4", "state": "running"},
                    {"id": "job-9a2b3c4d", "project_id": "41f745517fe4", "state": "running"}]
    import ghost_agent.sandbox.jobs as SJ
    monkeypatch.setattr(SJ, "get_job_supervisor", lambda sm: _Sup())
    inv = P._project_inventory(c, "41f745517fe4")
    assert [e["id"] for e in inv["jobs"]] == ["job-9a2b3c4d"]
    assert [r["name"] for r in inv["runs"]] == ["job-1a2b3c4d"]


def test_the_run_result_classifier_keeps_none_unresolved():
    from ghost_agent.distill.outcome_heuristics import _run_result_failed
    assert not _run_result_failed("[run b exited]\nEXIT CODE: none\n")
    assert _run_result_failed("[run b exited exit=3]\nEXIT CODE: 3\nboom\n")
    assert not _run_result_failed("[run b exited exit=0]\nEXIT CODE: 0\n")


def test_nothing_stale_names_the_old_tools():
    import ast
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    ex = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "execute")["function"]["description"]
    assert "`jobs`" not in ex and "`run collect`" in ex
    tree = ast.parse(Path(R.__file__).parent.joinpath("projects.py").read_text())
    notes = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and n.value.startswith("RELEASED (immutable)")]
    assert notes and all("where=workspace lifetime=server" in v for v in notes)


async def test_a_reused_run_name_never_points_the_alias_at_the_newer_job(tmp_path, scope, monkeypatch):
    """review r3 M1: `jobs cancel job_id=<old>` stopped the NEW job that reused the run name."""
    import ghost_agent.tools.delegate as D
    from ghost_agent.main import _collect_hint
    c = _ctx(tmp_path)
    scope(c)
    reg = registry_for(c)
    own = "conversation:conv1"
    old = reg.add(name="bench", place="workspace", lifetime="job", command="x", port=None, owner=own,
                  origin="owner", backend="jobs", backend_ref="job-aaaa1111")
    reg.set_state(old, "exited", "exit 0")
    reg.add(name="bench", place="workspace", lifetime="job", command="x", port=None, owner=own,
            origin="owner", backend="jobs", backend_ref="job-bbbb2222")
    assert R.run_name_for_job(c, "job-aaaa1111", own) == "" and R.run_name_for_job(c, "job-bbbb2222", own) == "bench"
    seen, ran = [], []

    async def fake_jobs(action=None, job_id=None, context=None, **kw):
        seen.append((action, job_id))
        return "jobs-said"

    async def fake_run(context=None, **kw):
        ran.append(kw)
        return ToolOutcome("ok", status=OutcomeStatus.OK)
    monkeypatch.setattr(D, "tool_jobs", fake_jobs)
    monkeypatch.setattr(R, "tool_run", fake_run)
    # r4: acts on the OLD job's own row — never `run … name=bench` (the newer job), never the jobs tool
    assert (await R.jobs_alias(context=c, action="cancel", job_id="job-aaaa1111")).reason_code == "run_ended"
    await R.jobs_alias(context=c, action="collect", job_id="job-aaaa1111")
    assert ran == [] and seen == []
    assert await R.jobs_alias(context=c, action="collect", job_id="job-cccc3333") == "jobs-said"   # no row at all
    assert seen == [("collect", "job-cccc3333")]
    assert _collect_hint({"id": "job-aaaa1111"}, c) == "jobs(action='collect')"


async def test_an_exited_job_with_no_known_exit_is_unresolved_live_and_stored(tmp_path, scope, monkeypatch):
    """review r3 minor 1: the live verdict and the stored one agree."""
    from ghost_agent.distill.outcome_heuristics import _run_result_failed

    class _Jobs:
        def reap(self):
            return []

        def list_entries(self):
            return [{"id": "job-1a2b3c4d", "state": "done", "exit_code": None}]

        def log_tail(self, jid, n):
            return "out"

        def mark_woken(self, jid):
            return True

        def _load_launches(self):
            return {}
    c = _ctx(tmp_path)
    scope(c)
    R.record_promoted_job(c, {"id": "job-1a2b3c4d"})
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (None, _Jobs()))
    out = await R.tool_run(c, action="collect", name="job-1a2b3c4d")
    assert out.status is OutcomeStatus.UNRESOLVED and out.reason_code == "run_exit_unknown", str(out)
    assert not _run_result_failed(str(out))


def test_an_alias_stored_run_row_is_read_by_the_run_rule():
    """review r3 minor 2."""
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error as fn
    for tool in ("jobs", "manage_services"):
        assert fn("[run b stopped]\nEXIT CODE: none\n(`jobs` is now `run`)", tool)
        assert not fn("[run b exited exit=0]\nEXIT CODE: 0\nok", tool)
        assert not fn("[run api logs]\nERROR: relation users does not exist\n", tool)   # a log is content


def test_the_wake_turn_of_a_job_marks_it_delivered():
    """review r3 minor 3: the §4NS wake arm."""
    from ghost_agent.utils.logging import request_id_context

    class _J:
        def _load_launches(self):
            return {}
    for rid in ("job-1a2b3c4d", "sub-job-1a2b3c4d"):
        tok = request_id_context.set(rid)
        try:
            assert R._delivers(_J(), "job-1a2b3c4d") and not R._delivers(_J(), "job-9a2b3c4d")
        finally:
            request_id_context.reset(tok)


async def test_a_start_that_lands_after_the_leafs_end_stops_itself(tmp_path, scope, monkeypatch):
    """review r3 minor 4: past the end's bounded wait, the late start stops its own row when it lands."""
    import threading
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb)
    scope(c, origin="autopilot")
    real = SandboxJobSupervisor.start_detached
    gate = threading.Event()

    def slow(self, *a, **k):
        gate.wait(10)
        return real(self, *a, **k)
    monkeypatch.setattr(SandboxJobSupervisor, "start_detached", slow)
    monkeypatch.setattr(R, "LEAF_INFLIGHT_WAIT_S", 0.2)
    tok = _leaf("t7-a1")
    try:
        t = asyncio.ensure_future(R.tool_run(c, action="start", name="late", command="sleep 30",
                                             where="workspace", lifetime="job"))
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
    finally:
        W.LEAF_ID.reset(tok)
    try:
        assert await R.stop_leaf_runs(c, "conversation:conv1", "t7-a1") == []        # the wait timed out
        gate.set()
        for _ in range(100):
            row = registry_for(c).latest_by_name("conversation:conv1", "late")
            if row is not None and row["ended_at"] is not None:
                break
            await asyncio.sleep(0.1)
        assert row is not None and row["state"] == "stopped", row
    finally:
        _kill_jobs(sb)


# ------------------------------------------------------------------ review r2

from test_4sx_c_run_tool import LAB, FakeLabs, _lab, fake  # noqa: E402,F401


async def test_a_leaf_cancelled_mid_start_still_stops_that_run_at_its_end(tmp_path, scope, monkeypatch):
    """r2 M1: the leaf's end waits for its in-flight start, then stops the row it recorded."""
    import threading
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb)
    scope(c, origin="autopilot")
    real = SandboxJobSupervisor.start_detached

    def slow(self, *a, **k):
        threading.Event().wait(1.0)
        return real(self, *a, **k)
    monkeypatch.setattr(SandboxJobSupervisor, "start_detached", slow)
    tok = _leaf("t3-a2")
    try:
        t = asyncio.ensure_future(R.tool_run(c, action="start", name="bench", command="sleep 30",
                                             where="workspace", lifetime="job"))
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
    finally:
        W.LEAF_ID.reset(tok)
    try:
        assert registry_for(c).live_by_name("conversation:conv1", "bench") is None     # not recorded yet
        assert await R.stop_leaf_runs(c, "conversation:conv1", "t3-a2") == ["bench"]
        row = registry_for(c).latest_by_name("conversation:conv1", "bench")
        assert row["ended_at"] is not None and row["state"] == "stopped"
        assert R._LEAF_INFLIGHT.get("t3-a2") is None
    finally:
        _kill_jobs(sb)


async def test_a_restart_outside_the_leaf_keeps_the_rows_leaf(tmp_path, scope):
    """r2 minor 3: the restart carries the OLD row's leaf (the current context's would be none here)."""
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c, origin="autopilot")
    tok = _leaf("t4-a1")
    try:
        await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace", lifetime="server")
    finally:
        W.LEAF_ID.reset(tok)
    r = await R.tool_run(c, action="restart", name="web")
    assert r.status is OutcomeStatus.OK, str(r)
    assert RunRegistry.meta_of(registry_for(c).live_by_name("conversation:conv1", "web")).get("leaf") == "t4-a1"


async def test_a_lab_server_restart_keeps_the_rows_leaf(tmp_path, scope, fake):
    c = _ctx(tmp_path)
    scope(c, origin="autopilot")
    _lab(fake, origin="autopilot")
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_ee00000001"}))
    tok = _leaf("t6-a1")
    try:
        r = await R.tool_run(c, action="start", name="api", command="x", where=f"{LAB}/db", lifetime="server",
                             port=80)
        assert r.status is OutcomeStatus.OK, str(r)
    finally:
        W.LEAF_ID.reset(tok)
    fake.on("GET", f"/v1/labs/{LAB}/jobs/job_ee00000001", (200, {"state": "running"}))
    fake.on("DELETE", f"/v1/labs/{LAB}/jobs/job_ee00000001", (200, {"state": "exited", "killed": True}))
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_ee00000002"}))
    r = await R.tool_run(c, action="restart", name="api")
    row = registry_for(c).live_by_name("conversation:conv1", "api")
    assert row is not None and row["backend_ref"].endswith("job_ee00000002"), str(r)
    assert RunRegistry.meta_of(row).get("leaf") == "t6-a1"


async def test_a_leaf_may_stop_its_own_promoted_job_but_its_end_leaves_it(tmp_path, scope, monkeypatch):
    """r2 minor 1."""
    c = _ctx(tmp_path)
    scope(c, origin="autopilot")
    hit = []

    async def fake_stop(context, reg_, row, origin):
        hit.append(row["name"])
        return ToolOutcome("stopped", status=OutcomeStatus.OK)
    monkeypatch.setattr(R, "_stop", fake_stop)
    tok = _leaf("t8-a1")
    try:
        R.record_promoted_job(c, {"id": "job-1a2b3c4d", "command": "npm run dev"})
        assert (await R.tool_run(c, action="stop", name="job-1a2b3c4d")).status is OutcomeStatus.OK
    finally:
        W.LEAF_ID.reset(tok)
    tok = _leaf("t8-a2")                                 # another build: refused
    try:
        R.record_promoted_job(c, {"id": "job-2a2b3c4d", "command": "npm run dev"})
    finally:
        W.LEAF_ID.reset(tok)
    tok = _leaf("t9-a1")
    try:
        assert (await R.tool_run(c, action="stop", name="job-2a2b3c4d")).reason_code == "run_not_allowed"
    finally:
        W.LEAF_ID.reset(tok)
    assert hit == ["job-1a2b3c4d"]
    assert await R.stop_leaf_runs(c, "conversation:conv1", "t8-a2") == []


async def test_only_the_owner_or_the_starter_marks_a_collected_job_delivered(tmp_path, scope, monkeypatch):
    """r2 M2 — and a collect of a job still running never marks it."""
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    from ghost_agent.utils.logging import request_id_context
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    woke = []
    monkeypatch.setattr(SandboxJobSupervisor, "mark_woken", lambda self, jid: woke.append(jid) or True)
    try:
        await R.tool_run(c, action="start", name="slow", command="sleep 3", where="workspace", lifetime="job")
        assert (await R.tool_run(c, action="collect", name="slow")).status is OutcomeStatus.UNRESOLVED
        assert woke == []                                                    # still running
        await R.tool_run(c, action="start", name="w", command="echo hi", where="workspace", lifetime="job")
        ref = registry_for(c).live_by_name("conversation:conv1", "w")["backend_ref"]
        tok = request_id_context.set("sched-abc123")                         # a scheduled task reading it
        try:
            out = await _land(c, "w")
        finally:
            request_id_context.reset(tok)
        assert out.status is OutcomeStatus.OK and woke == []
        out = await R.tool_run(c, action="collect", name="w")                # the owner's turn
        assert out.status is OutcomeStatus.OK and woke == [ref]
    finally:
        _kill_jobs(sb)


def test_delivers_reads_the_starter():
    from ghost_agent.utils.logging import request_id_context

    class _J:
        def _load_launches(self):
            return {"job-1": {"started_by": "sub-x1"}}
    tok = request_id_context.set("sub-x1")
    try:
        assert R._delivers(_J(), "job-1") and not R._delivers(_J(), "job-2")
    finally:
        request_id_context.reset(tok)
    tok = request_id_context.set("sched-zz")
    try:
        assert not R._delivers(_J(), "job-1")
    finally:
        request_id_context.reset(tok)
    assert R._delivers(_J(), "job-9")                                         # internal code / the owner


def test_an_ended_row_is_never_rewritten_by_a_landed_job(tmp_path, scope):
    """r2 minor 4 + the test gap: an already-ended row with the same job id stays as it ended."""
    c = _ctx(tmp_path, pid="41f745517fe4")
    scope(c)
    R.record_promoted_job(c, {"id": "job-1a2b3c4d"})
    reg = registry_for(c)
    row = reg.live_by_name("project:41f745517fe4", "job-1a2b3c4d")
    reg.set_state(row["run_id"], "stopped", "stopped by the owner")
    assert R.end_landed_job_runs(c, [{"id": "job-1a2b3c4d", "state": "done", "exit_code": 0}]) == 0
    assert reg.latest_by_name("project:41f745517fe4", "job-1a2b3c4d")["state"] == "stopped"
    reg.set_state(row["run_id"], "exited", "late", only_live=True)
    assert reg.latest_by_name("project:41f745517fe4", "job-1a2b3c4d")["state_note"] == "stopped by the owner"


def test_the_hint_names_run_collect_after_the_row_ended(tmp_path, scope):
    """r2 minor 2: the reaper ends the row before it writes the hint."""
    from ghost_agent.main import _collect_hint
    c = _ctx(tmp_path)
    scope(c)
    R.record_promoted_job(c, {"id": "job-1a2b3c4d"})
    R.end_landed_job_runs(c, [{"id": "job-1a2b3c4d", "state": "done", "exit_code": 0}])
    assert _collect_hint({"id": "job-1a2b3c4d"}, c) == "run(action='collect', name='job-1a2b3c4d')"


async def test_the_alias_hint_follows_the_result(tmp_path, scope, monkeypatch):
    """r2 minor 5: the result's own head stays first (every head reader anchors on it)."""
    async def fake_run(context=None, **kw):
        return ToolOutcome("RUN ERROR: boom", status=OutcomeStatus.FAILED)
    monkeypatch.setattr(R, "tool_run", fake_run)
    c = _ctx(tmp_path)
    scope(c)
    out = await R.manage_services_alias(context=c, action="status")
    assert str(out).startswith("RUN ERROR: boom") and "is now `run`" in str(out).splitlines()[-1]
    assert out.status is OutcomeStatus.FAILED


def test_the_stored_collect_verdict_matches_the_live_one():
    from ghost_agent.distill.outcome_heuristics import _run_result_failed
    assert _run_result_failed("[run b expired]\nEXIT CODE: none\nkilled at ttl")
    assert _run_result_failed("[run b stopped]\nEXIT CODE: none\n")
    assert _run_result_failed("[run b lost]\nEXIT CODE: none\n")
    assert not _run_result_failed("[run b exited]\nEXIT CODE: none\nlog says\nEXIT CODE: 3\n")
    assert not _run_result_failed("[run b exited exit=0]\nEXIT CODE: 0\nEXIT CODE: 3\n")
    ok = ToolOutcome("[run b exited exit=3]\nEXIT CODE: 3\n", status=OutcomeStatus.OK)
    assert not _run_result_failed(ok)                                       # declared verdict first


async def test_an_ownerless_stop_ends_the_jobs_run_row(tmp_path, scope, monkeypatch):
    """r2 minor 6."""
    import ghost_agent.tools.delegate as D
    owner_ctx = _ctx(tmp_path)
    scope(owner_ctx)
    R.record_promoted_job(owner_ctx, {"id": "job-1a2b3c4d"})

    async def fake_jobs(action=None, job_id=None, context=None, **kw):
        return "cancelled"

    class _Jobs:
        def get(self, jid):
            return {"id": jid, "state": "cancelled"}
    monkeypatch.setattr(D, "tool_jobs", fake_jobs)
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (None, _Jobs()))
    c, tok = _ownerless(tmp_path)
    try:
        assert await R.tool_run(c, action="stop", name="job-1a2b3c4d") == "cancelled"
    finally:
        RS.reset(tok)
    assert registry_for(owner_ctx).latest_by_name("conversation:conv1", "job-1a2b3c4d")["state"] == "stopped"


# ------------------------------------------------------------------ §4SX E (postgres_admin removed)

def test_a_model_reaching_for_postgres_is_sent_to_a_lab():
    from ghost_agent.core.agent import GhostAgent
    avail = ["execute", "lab", "run", "file_system"]
    for n in ("postgres", "postgres_admin", "postgres-admin", "postgresadmin", "Postgres_Admin", "postgresql", "psql"):
        assert GhostAgent._canonicalise_tool_name(n, avail) == "lab", n


async def test_an_sql_shaped_lab_call_says_where_sql_runs_now(tmp_path):
    from ghost_agent.tools.labs import tool_lab
    out = await tool_lab(SimpleNamespace(args=SimpleNamespace(labs_host="", labs_key_file="")), action="query",
                         query="SELECT 1")
    assert out.status is OutcomeStatus.REJECTED and "no local database" in out
    import ghost_agent.tools.labs as L
    _orig = L.labs_configured
    L.labs_configured = lambda ctx: True            # configured: the hint would name pg-single — only for SQL
    try:
        plain = await tool_lab(SimpleNamespace(args=SimpleNamespace(labs_host="h", labs_key_file="")), action="nope")
    finally:
        L.labs_configured = _orig
    assert "pg-single" not in plain and "no local database" not in plain


def test_the_postgres_tool_is_gone_everywhere():
    import importlib.util
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    from ghost_agent.tools import validators as V
    assert importlib.util.find_spec("ghost_agent.tools.database") is None
    assert not [d for d in TOOL_DEFINITIONS if "postgres" in d["function"]["name"]]
    assert not hasattr(V, "validate_sql") and V.__all__ == ["validate_shell"]



def test_the_boot_args_have_no_default_db(monkeypatch):
    import sys as _sys
    from ghost_agent.main import parse_args
    monkeypatch.setattr(_sys, "argv", ["ghost-agent"])
    assert not hasattr(parse_args(), "default_db")
    monkeypatch.setattr(_sys, "argv", ["ghost-agent", "--default-db", "postgresql://x"])
    with pytest.raises(SystemExit):
        parse_args()


# ------------------------------------------------------------------ battery survivors (D5 battery run 3)

async def test_collect_strips_the_stage_status_marker(tmp_path, scope, monkeypatch):
    """R8: the marker line is ours, never the job's output (and its codes are not the job's)."""
    class _Jobs:
        def reap(self):
            return []

        def list_entries(self):
            return [{"id": "job-1a2b3c4d", "state": "done", "exit_code": 0}]

        def log_tail(self, jid, n):
            return "real output\n__GHOST_PIPESTATUS=0 0\n"

        def mark_woken(self, jid):
            return True

        def _load_launches(self):
            return {}
    c = _ctx(tmp_path)
    scope(c)
    R.record_promoted_job(c, {"id": "job-1a2b3c4d"})
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (None, _Jobs()))
    out = await R.tool_run(c, action="collect", name="job-1a2b3c4d")
    assert out.status is OutcomeStatus.OK and "real output" in out and "__GHOST_PIPESTATUS" not in out


def test_only_a_landed_job_ends_a_row_even_when_a_running_one_comes_first(tmp_path, scope):
    """R23: a running entry is skipped, not a KeyError that ends nothing after it."""
    c = _ctx(tmp_path, pid="41f745517fe4")
    scope(c)
    R.record_promoted_job(c, {"id": "job-2a2b3c4d"})                    # the running one, first in row order
    R.record_promoted_job(c, {"id": "job-1a2b3c4d"})
    n = R.end_landed_job_runs(c, [{"id": "job-2a2b3c4d", "state": "running"},
                                  {"id": "job-1a2b3c4d", "state": "done", "exit_code": 0}])
    assert n == 1
    reg = registry_for(c)
    assert reg.live_by_name("project:41f745517fe4", "job-2a2b3c4d") is not None
    assert reg.latest_by_name("project:41f745517fe4", "job-1a2b3c4d")["state"] == "exited"


def test_a_landed_job_racing_a_stop_never_rewrites_the_stop(tmp_path, scope, monkeypatch):
    """R51: the rows were read live, a `run stop` ended one in between — the write-back keeps "stopped"."""
    c = _ctx(tmp_path, pid="41f745517fe4")
    scope(c)
    R.record_promoted_job(c, {"id": "job-1a2b3c4d"})
    reg = registry_for(c)
    stale = reg.rows(live_only=True)
    reg.set_state(stale[0]["run_id"], "stopped", "stopped by the owner")       # the concurrent stop
    monkeypatch.setattr(RunRegistry, "rows", lambda self, owner=None, **k: [dict(r) for r in stale])
    R.end_landed_job_runs(c, [{"id": "job-1a2b3c4d", "state": "done", "exit_code": 0}])
    monkeypatch.undo()
    row = registry_for(c).latest_by_name("project:41f745517fe4", "job-1a2b3c4d")
    assert row["state"] == "stopped" and row["state_note"] == "stopped by the owner"


def test_a_stored_collect_under_the_failure_banner_is_still_read():
    """R30b: the banner line above the head is skipped, as strikes.is_program_output does."""
    from ghost_agent.distill.outcome_heuristics import _run_result_failed
    assert _run_result_failed("[FAILURE BANNER] EXIT CODE: 3\n[run b exited exit=3]\nEXIT CODE: 3\nboom")
    assert not _run_result_failed("[FAILURE BANNER] x\n[run api logs]\nERROR: relation users does not exist")
    assert not _run_result_failed("[run b exited exit=0]\nEXIT CODE: 0\nok")


def test_a_lab_call_waits_for_a_changing_run_on_that_lab_only():
    """R33: `run start where=lab_x/…` then `lab exec lab_id=lab_x` — ordered; another lab is not."""
    from ghost_agent.core.agent import _run_call_depends
    run = {"action": "start", "name": "api", "where": "lab_0123abcd/db", "command": "x", "lifetime": "server"}
    assert _run_call_depends("lab", {"action": "exec", "lab_id": "lab_0123abcd"}, "exec", "run", run, "start")
    assert not _run_call_depends("lab", {"action": "exec", "lab_id": "lab_99999999"}, "exec", "run", run, "start")
    assert _run_call_depends("run", run, "start", "lab", {"action": "exec", "lab_id": "lab_0123abcd"}, "exec")
    assert not _run_call_depends("run", run, "start", "lab", {"action": "exec", "lab_id": "lab_99999999"}, "exec")


# ------------------------------------------------------------------ review r4

async def test_the_old_job_of_a_reused_run_name_is_read_by_its_id(tmp_path, scope):
    """r4 M1 (the reviewer's repro, real supervisors): `jobs collect job_id=<A>` after `bench` was reused gets A."""
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    try:
        await R.tool_run(c, action="start", name="bench", command="echo FIRST-RESULT", where="workspace",
                         lifetime="job")
        own = "conversation:conv1"
        old_ref = registry_for(c).latest_by_name(own, "bench")["backend_ref"]
        await _land(c, "bench")
        b = await R.tool_run(c, action="start", name="bench", command="sleep 20", where="workspace", lifetime="job")
        assert b.status is OutcomeStatus.OK, str(b)
        got = await R.jobs_alias(context=c, action="collect", job_id=old_ref)
        assert got.status is OutcomeStatus.OK and "FIRST-RESULT" in str(got) and "newer run" in str(got), str(got)
        st = await R.jobs_alias(context=c, action="status", job_id=old_ref)
        assert st.status is OutcomeStatus.OK and "exited" in str(st).splitlines()[1]
        assert old_ref in str(st).splitlines()[-1]                                  # the hint names the job
        stop = await R.jobs_alias(context=c, action="cancel", job_id=old_ref)
        assert stop.reason_code == "run_ended"
        assert registry_for(c).live_by_name(own, "bench") is not None                # the newer run still runs
    finally:
        _kill_jobs(sb)


async def test_two_late_starts_stop_each_row_once(tmp_path, scope, monkeypatch):
    """r4 minor 1 (the reviewer's repro): the late stops of one leaf run one at a time."""
    c = _ctx(tmp_path)
    scope(c, origin="autopilot")
    owner = "conversation:conv1"
    reg = registry_for(c)
    calls = []

    async def slow_stop(context, reg_, row, origin):
        calls.append(row["name"])
        await asyncio.sleep(0.3)
        reg_.set_state(row["run_id"], "stopped", "stopped", only_live=True)
        return ToolOutcome("ok", status=OutcomeStatus.OK)
    monkeypatch.setattr(R, "_stop", slow_stop)
    monkeypatch.setattr(R, "LEAF_INFLIGHT_WAIT_S", 0.05)
    gates = {"a": asyncio.Event(), "b": asyncio.Event()}

    async def start(name):
        await gates[name].wait()
        reg.add(name=name, place="workspace", lifetime="job", command="sleep 30", port=None, owner=owner,
                origin="autopilot", backend="jobs", backend_ref=f"job-{name}0000000", meta=R._leaf_meta())
        return ToolOutcome("started", status=OutcomeStatus.OK)
    tok = W.LEAF_ID.set("deadbeef-a1")
    try:
        ta = asyncio.ensure_future(R._locked(owner, "a", start("a")))
        tb = asyncio.ensure_future(R._locked(owner, "b", start("b")))
        await asyncio.sleep(0.01)
    finally:
        W.LEAF_ID.reset(tok)
    assert await R.stop_leaf_runs(c, owner, "deadbeef-a1") == []
    gates["a"].set()
    await ta
    await asyncio.sleep(0.05)
    gates["b"].set()
    await tb
    for _ in range(40):
        if not R._LATE_STOPS:
            break
        await asyncio.sleep(0.1)
    assert sorted(calls) == ["a", "b"], calls
    assert not reg.rows(owner, live_only=True)


async def test_an_exited_job_with_no_code_is_not_reported_as_still_running(tmp_path, scope, monkeypatch):
    """r4 minor 3: the head says so, and the turn loop's op outcome is not "unresolved"."""
    class _Jobs:
        def reap(self):
            return []

        def list_entries(self):
            return [{"id": "job-1a2b3c4d", "state": "done", "exit_code": None}]

        def log_tail(self, jid, n):
            return "out"

        def mark_woken(self, jid):
            return True

        def _load_launches(self):
            return {}
    c = _ctx(tmp_path)
    scope(c)
    R.record_promoted_job(c, {"id": "job-1a2b3c4d"})
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (None, _Jobs()))
    out = await R.tool_run(c, action="collect", name="job-1a2b3c4d")
    assert "exit code was not recorded" in str(out).splitlines()[0]
    import ast
    src = Path(R.__file__).parent.parent.joinpath("core", "agent.py").read_text()
    tree = ast.parse(src)
    keys = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)
            and any(isinstance(k, ast.Constant) and k.value == "unresolved" for k in n.keys)]
    vals = [ast.unparse(v) for d in keys for k, v in zip(d.keys, d.values)
            if isinstance(k, ast.Constant) and k.value == "unresolved"]
    assert vals and any("run_exit_unknown" in v for v in vals)


def test_the_landed_hint_for_a_reused_run_name_names_the_job_id(tmp_path, scope):
    """r4 minor 4."""
    from ghost_agent.main import _collect_hint
    c = _ctx(tmp_path)
    scope(c)
    reg = registry_for(c)
    own = "conversation:conv1"
    a = reg.add(name="bench", place="workspace", lifetime="job", command="x", port=None, owner=own, origin="owner",
                backend="jobs", backend_ref="job-aaaa1111")
    reg.set_state(a, "exited", "exit 0")
    reg.add(name="bench", place="workspace", lifetime="job", command="x", port=None, owner=own, origin="owner",
            backend="jobs", backend_ref="job-bbbb2222")
    e = {"id": "job-aaaa1111", "started_via": "run", "label": "bench"}
    assert _collect_hint(e, c) == "jobs(action='collect', job_id='job-aaaa1111')"
    assert _collect_hint({"id": "job-bbbb2222", "started_via": "run", "label": "bench"}, c) == \
        "run(action='collect', name='bench')"


def test_a_stored_empty_run_status_through_the_alias_is_not_a_failure():
    """r4 minor 5."""
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error as fn
    assert not fn("No runs here. (`run start name=… command=…`)\nConnectionError: probe note", "jobs")


async def test_the_sql_hint_is_executable():
    """r4 minor 6: names the template's member and the postgres role (lab exec runs as root)."""
    from ghost_agent.tools.labs import tool_lab
    out = await tool_lab(SimpleNamespace(args=SimpleNamespace(labs_host="", labs_key_file="")), action="query",
                         sql="SELECT 1")
    assert "no labs host is configured" in out and "lab create" not in out      # no lab to create here
    import ghost_agent.tools.labs as L
    _orig = L.labs_configured
    L.labs_configured = lambda ctx: True
    try:
        out = await tool_lab(SimpleNamespace(args=SimpleNamespace(labs_host="h", labs_key_file="")), action="query",
                             sql="SELECT 1 WHERE a='x'")
    finally:
        L.labs_configured = _orig
    assert "member=pg1" in out and "psql -U postgres" in out and "stdin=" in out and "§4SX" not in out



# ------------------------------------------------------------------ review r5

def test_an_exit_unknown_call_is_neither_succeeded_nor_failed_in_the_summary():
    """r5 M1: a mixed batch never lists it under SUCCEEDED ("do NOT report it as failed")."""
    from ghost_agent.tools.tool_failure import summarize_multi_op_outcomes as S
    out = S([{"tool": "run", "ok": True, "unresolved": False, "exit_unknown": True},
             {"tool": "execute", "ok": False, "preview": "EXIT CODE: 1 boom"},
             {"tool": "file_system", "ok": True}])
    assert "SUCCEEDED: file_system\n" in out and "EXIT CODE NOT RECORDED" in out and "run" in out.split("RECORDED")[1]
    assert "SUCCEEDED: run" not in out and "file_system; run" not in out


def test_the_turn_loop_books_an_exit_unknown_collect_in_its_own_bucket():
    """r5 M1: the op-outcome dict the loop builds — its exit_unknown / unresolved expressions EVALUATED on an
    exit-unknown collect and on a plain unresolved one."""
    import ast
    tree = ast.parse(Path(R.__file__).parent.parent.joinpath("core", "agent.py").read_text())
    dicts = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)
             and {"ok", "unresolved", "exit_unknown"} <= {k.value for k in n.keys if isinstance(k, ast.Constant)}]
    assert len(dicts) == 1
    d = dicts[0]
    vals = {k.value: ast.unparse(v) for k, v in zip(d.keys, d.values) if isinstance(k, ast.Constant)}
    def ev(expr, code):
        _o = SimpleNamespace(reason_code=code, status=OutcomeStatus.UNRESOLVED)
        return eval(expr, {"getattr": getattr, "_outcome": _o, "_OutcomeStatus": OutcomeStatus, "_intr": False})
    assert ev(vals["unresolved"], "run_no_answer") is True          # the env reaches a real verdict
    assert ev(vals["exit_unknown"], "run_exit_unknown") is True
    assert ev(vals["exit_unknown"], "run_no_answer") is False
    try:
        assert not ev(vals["unresolved"], "run_exit_unknown")
    except NameError as e:                         # the expression reads a loop local this env lacks
        raise AssertionError(f"extend the env for {vals['unresolved']!r}: {e}")


def test_a_stored_run_error_under_the_banner_is_a_failure():
    """r5 M2."""
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error as fn, _run_result_failed
    row = "[FAILURE BANNER] RUN ERROR: 'x' is not running\nRUN ERROR: 'x' is not running"
    assert _run_result_failed(row) and fn(row, "run") and fn(row, "jobs")
    assert not fn("[FAILURE BANNER] x\n[run api logs]\nERROR: relation", "jobs")
    # r6: the alias route itself (the generic head scan reads "error:" as a failure) — status under the banner
    listing = "[FAILURE BANNER] x\nRUNS (probed now):\n  api server workspace failed (error: port in use)"
    assert not fn(listing, "run") and not fn(listing, "jobs") and not fn(listing.split("\n", 1)[1], "manage_services")


async def test_the_leaf_stop_lock_survives_a_new_event_loop():
    """r5 minor 1: the same leaf id on a second loop never raises 'bound to a different event loop'."""
    import threading
    c = SimpleNamespace(memory_dir=None)

    async def _slow_rows(context, owner, leaf_id, gone):      # a real row pass awaits: the lock is contended
        await asyncio.sleep(0.01)
        return gone
    _orig = R._stop_leaf_rows
    R._stop_leaf_rows = _slow_rows

    def _run():
        async def two():
            await asyncio.gather(R.stop_leaf_runs(c, "o", "loopy-a1"), R.stop_leaf_runs(c, "o", "loopy-a1"))
        asyncio.run(two())
    errs = []

    def _guarded():
        try:
            _run()
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    try:
        for _ in range(2):
            t = threading.Thread(target=_guarded)
            t.start()
            t.join()
        await asyncio.gather(R.stop_leaf_runs(c, "o", "loopy-a1"), R.stop_leaf_runs(c, "o", "loopy-a1"))
    finally:
        R._stop_leaf_rows = _orig
    assert not errs, errs



# ------------------------------------------------------------------ review r6

def test_an_exit_unknown_collect_counts_as_finished_beside_a_running_call():
    from ghost_agent.tools.tool_failure import summarize_multi_op_outcomes as S
    out = S([{"tool": "run", "ok": True, "unresolved": False, "exit_unknown": True},
             {"tool": "execute", "ok": False, "unresolved": True}])
    assert "1 call(s) STILL RUNNING, 1 finished" in out and "FINISHED: run" in out and "RECORDED" in out


async def test_a_passed_through_start_refusal_is_a_stored_failure(tmp_path, scope):
    """r7: the REAL results of the workspace guards, called as `run` and as `manage_services`, grade failed."""
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error as fn
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb, host="10.20.30.40:8700")
    scope(c)
    for cmd in ("curl http://10.20.30.40:8700/v1/labs", "curl -s http://127.0.0.1:8000/api/health"):
        r = await R.tool_run(c, action="start", name="y", command=cmd, where="workspace", lifetime="server")
        a = await R.manage_services_alias(c, action="start", name="y", command=cmd)
        for out, tn in ((r, "run"), (a, "manage_services")):
            assert out.status is OutcomeStatus.REJECTED and str(out).startswith("RUN ERROR"), str(out)[:60]
            assert fn(str(out), tn) and fn("[FAILURE BANNER] x\n" + str(out), tn), (tn, str(out)[:60])


# ------------------------------------------------------------------ review r7

def test_one_head_rule_where_a_run_result_leaves_the_tool():
    from ghost_agent.distill.outcome_heuristics import _RUN_COLLECT_RE, _looks_like_tool_error as fn
    from ghost_agent.tools.outcome import ToolOutcome as T
    H = R._headed
    lab = H(T("LAB ERROR (labs_unreachable): the labs host h did not answer", status=OutcomeStatus.FAILED))
    restart = H(T("The old server WAS stopped; starting it again failed:\nLAB ERROR (x): y",
                  status=OutcomeStatus.REJECTED, reason_code="lab_x"))
    for o in (lab, restart):
        assert str(o).startswith("RUN ERROR — ") and fn(str(o), "run") and fn(str(o), "manage_services")
    assert restart.reason_code == "lab_x" and restart.status is OutcomeStatus.REJECTED
    nans = H(T("RUN ERROR (no_answer): the labs host did not answer — it may have started.",
               status=OutcomeStatus.UNRESOLVED))
    assert str(nans).startswith("RUN UNRESOLVED (no_answer): the labs host") and nans.status is OutcomeStatus.UNRESOLVED
    assert not fn(str(nans), "run") and not fn(str(nans), "jobs")
    coll = T("[run b exited]\nEXIT CODE: 2\nboom", status=OutcomeStatus.FAILED)
    assert H(coll) is coll and fn(str(coll), "run")                       # a collect keeps its own head
    ok = T("[run b logs]\nRUN ERROR printed by the program", status=OutcomeStatus.OK)
    assert H(ok) is ok and not fn(str(ok), "run")
    bad = T("[run b logs]\nsomething", status=OutcomeStatus.FAILED)       # not a collect head: gets the head
    assert str(H(bad)).startswith("RUN ERROR") and fn(str(H(bad)), "run")
    assert R._COLLECT_HEAD_RE.pattern == _RUN_COLLECT_RE.pattern
    # the alias path applies it too
    assert str(R._with_hint(T("LAB ERROR (z): q", status=OutcomeStatus.FAILED), "(hint)")).startswith("RUN ERROR")


async def test_an_unanswered_lab_start_is_stored_unresolved_not_failed(tmp_path, scope, monkeypatch):
    """r7 M2 end to end: the lab start's no-answer through `tool_run` reads `RUN UNRESOLVED`."""
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error as fn
    import ghost_agent.tools.labs as L
    c = _ctx(tmp_path, host="10.20.30.40:8700")
    scope(c)

    async def _no_answer(ctx, method, path, **k):
        if method == "GET":
            return 200, {"owner": "x", "origin": "owner"}
        raise L.LabsNoAnswer("timed out")                       # the exec POST: it may have started
    monkeypatch.setattr(L, "_call", _no_answer)
    monkeypatch.setattr(L, "labs_configured", lambda ctx: True)
    r = await R.tool_run(c, action="start", name="db", command="sleep 9", where="lab_0123abcd/pg1",
                         lifetime="job")
    assert r.status is OutcomeStatus.UNRESOLVED, (r.status, str(r)[:120])
    assert str(r).startswith("RUN UNRESOLVED") and not fn(str(r), "run")


async def test_a_live_job_under_a_reused_name_is_stopped_only_under_runs_rules(tmp_path, scope, monkeypatch):
    """battery R76/R76b: the reused-name cancel's probe and leaf guards (and the allowed path stops THAT row)."""
    c = _ctx(tmp_path)
    scope(c)
    owner = "conversation:conv1"
    reg = registry_for(c)
    stopped = []

    async def fake_stop(context, reg_, row, origin):
        stopped.append(row["backend_ref"])
        return ToolOutcome("stopped", status=OutcomeStatus.OK)
    monkeypatch.setattr(R, "_stop", fake_stop)
    tok = W.LEAF_ID.set("leafa000-a1")
    try:
        reg.add(name="bench", place="workspace", lifetime="job", command="sleep 30", port=None, owner=owner,
                origin="owner", backend="jobs", backend_ref="job-old0000000", meta=R._leaf_meta())
    finally:
        W.LEAF_ID.reset(tok)
    reg.add(name="bench", place="workspace", lifetime="job", command="sleep 30", port=None, owner=owner,
            origin="owner", backend="jobs", backend_ref="job-new0000000")
    old = [r for r in reg.rows(owner, live_only=True) if r["backend_ref"] == "job-old0000000"][0]
    r = await R._old_job_row_op(c, old, "probe", "stop")
    assert r.reason_code == "run_not_allowed" and "probe" in str(r) and stopped == []
    tok = W.LEAF_ID.set("leafb000-a1")                       # another build's leaf
    try:
        r = await R._old_job_row_op(c, old, "autopilot", "stop")
        assert r.reason_code == "run_not_allowed" and stopped == []
    finally:
        W.LEAF_ID.reset(tok)
    tok = W.LEAF_ID.set("leafa000-a1")                       # its own leaf may
    try:
        r = await R._old_job_row_op(c, old, "autopilot", "stop")
    finally:
        W.LEAF_ID.reset(tok)
    assert r.status is OutcomeStatus.OK and stopped == ["job-old0000000"]
    r = await R._old_job_row_op(c, old, "owner", "stop")    # the owner, outside a leaf
    assert r.status is OutcomeStatus.OK and stopped == ["job-old0000000"] * 2
    # r9: a probe stops its OWN probe row; a leaf stops a job it detached (meta.started_leaf)
    reg.add(name="pb", place="workspace", lifetime="job", command="sleep 30", port=None, owner=owner,
            origin="probe", backend="jobs", backend_ref="job-prb0000000")
    pr = [x for x in reg.rows(owner, live_only=True) if x["backend_ref"] == "job-prb0000000"][0]
    assert (await R._old_job_row_op(c, pr, "probe", "stop")).status is OutcomeStatus.OK
    reg.add(name="det", place="workspace", lifetime="job", command="sleep 30", port=None, owner=owner,
            origin="autopilot", backend="jobs", backend_ref="job-det0000000", meta={"started_leaf": "leafc000-a1"})
    dt = [x for x in reg.rows(owner, live_only=True) if x["backend_ref"] == "job-det0000000"][0]
    tok = W.LEAF_ID.set("leafc000-a1")
    try:
        assert (await R._old_job_row_op(c, dt, "autopilot", "stop")).status is OutcomeStatus.OK
    finally:
        W.LEAF_ID.reset(tok)
    assert stopped[-2:] == ["job-prb0000000", "job-det0000000"]
    tok = W.LEAF_ID.set("leafb000-a1")                       # r10: another leaf never stops a detached job
    try:
        assert (await R._old_job_row_op(c, dt, "autopilot", "stop")).reason_code == "run_not_allowed"
    finally:
        W.LEAF_ID.reset(tok)
    assert len(stopped) == 4


# ------------------------------------------------------------------ review r8

def test_each_headed_failure_keeps_its_own_error_line():
    """r8 M1: the digest/task-state error line names each failure, not one constant head."""
    from ghost_agent.core.strikes import error_line
    from ghost_agent.tools.execute import _AGENT_PORT_PROBE_MSG
    from ghost_agent.tools.outcome import ToolOutcome as T
    outs = [R._headed(T(t, status=OutcomeStatus.REJECTED)) for t in
            ("LAB ERROR (lab_not_found): no such lab", _AGENT_PORT_PROBE_MSG, "Not run: this run call names h")]
    lines = [error_line(o, tool="run") for o in outs]
    assert len(set(lines)) == 3, lines
    assert "lab_not_found" in lines[0] and "SANDBOX EGRESS" in lines[1] and "Not run" in lines[2]
    assert all(str(o).startswith("RUN ERROR — ") for o in outs)


def test_an_unanswered_run_is_not_evidence():
    """r8 M2: `RUN UNRESOLVED` (also under an alias, and under the banner) is no answer, never substantive."""
    from ghost_agent.core.evidence_gate import _empty_reason
    from ghost_agent.tools.outcome import ToolOutcome as T
    u = str(R._headed(T("RUN ERROR (no_answer): the labs host did not answer — it may have started",
                        status=OutcomeStatus.UNRESOLVED)))
    assert _empty_reason("run", {}, u, False) == "run: no answer"
    from ghost_agent.core.evidence_gate import assess_turn_evidence
    a = assess_turn_evidence([{"name": "run", "content": u}])
    assert a.consulted == 1 and a.substantive == 0, a
    assert _empty_reason("run", {}, "[FAILURE BANNER] x\n" + u, False) == "run: no answer"
    assert _empty_reason("run", {}, "[run b exited exit=0]\nEXIT CODE: 0\nfine", False) is None
    assert _empty_reason("run", {}, "RUN ERROR — LAB ERROR (x): y", False) == "run: error"


def test_a_log_that_prints_exit_code_is_content():
    """r8 minor 2: `[run x logs]` whose first log line reads EXIT CODE: 1 is a log, not a collect."""
    from ghost_agent.core.strikes import is_program_output
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error as fn
    for t in ("[run api logs]\nEXIT CODE: 1 from a child", "[run api logs — stopped]\nEXIT CODE: 1",
              "[run web app logs] up\nEXIT CODE: 2"):
        assert not fn(t, "run") and not is_program_output("run", t) and not R._COLLECT_HEAD_RE.match(t), t
    assert fn("[run b exited exit=1]\nEXIT CODE: 1\nboom", "run")
    assert is_program_output("run", "[run b exited exit=1]\nEXIT CODE: 1\nboom")
    for n in ("logs", "app"):                       # r9: a run NAMED logs/app still collects
        t = f"[run {n} exited exit=1]\nEXIT CODE: 1\nboom"
        assert fn(t, "run") and is_program_output("run", t) and R._COLLECT_HEAD_RE.match(t), n
        assert not fn(f"[run {n} exited exit=0]\nEXIT CODE: 0\nok", "run"), n


def test_the_head_rule_and_the_alias_hint_keep_every_field():
    """r8 minor 3/4: world_changed (the repeat guard reads an in-flux lab), declared, call_args, duration_s."""
    from ghost_agent.tools.outcome import ToolOutcome as T
    src = T("LAB ERROR (in_flux): busy", status=OutcomeStatus.FAILED, world_changed=True, reason_code="lab_in_flux",
            declared=True, call_args={"action": "stop"}, duration_s=1.5)
    for o in (R._headed(src), R._with_hint(src, "(hint)")):
        assert (o.world_changed, o.declared, o.call_args, o.duration_s, o.reason_code) == \
               (True, True, {"action": "stop"}, 1.5, "lab_in_flux"), o
    und = T("RUN ERROR (no_answer): x", status=OutcomeStatus.UNRESOLVED, world_changed=None, declared=False)
    assert R._headed(und).declared is False and R._headed(und).world_changed is None



# ------------------------------------------------------------------ review r9

def test_every_unresolved_run_result_is_stored_unresolved():
    """r9: the UNRESOLVED shapes with no RUN ERROR head get `RUN UNRESOLVED — ` too; the stored predicate, the
    corpus flags and the evidence gate all read them as no verdict (never a success)."""
    from ghost_agent.core.evidence_gate import _empty_reason
    from ghost_agent.distill.outcome_heuristics import (_looks_like_tool_error as fn, is_unresolved_tool_result,
                                                        tool_failure_flags)
    from ghost_agent.tools.outcome import ToolOutcome as T
    shapes = ["[run x] stop sent, but the backend did not answer the check (t) — `run status name=x` shows it.",
              "[run x] restart sent, but the sandbox did not answer the check — `run status name=x` shows it.",
              "[run x — running] no result yet; collect it later",
              "RUN ERROR (no_answer): the labs host did not answer — it may have stopped"]
    for t in shapes:
        o = str(R._headed(T(t, status=OutcomeStatus.UNRESOLVED)))
        assert o.startswith("RUN UNRESOLVED") and "RUN UNRESOLVED — RUN" not in o, o
        assert is_unresolved_tool_result(o) and is_unresolved_tool_result("[FAILURE BANNER] b\n" + o), o
        assert not fn(o, "run") and _empty_reason("run", {}, o, False) == "run: no answer", o
        assert tool_failure_flags([{"name": "run", "content": o}]) == [], o
    unk = "[run b exited — the exit code was not recorded: judge it by the output]\nEXIT CODE: none\nx"
    assert R._headed(T(unk, status=OutcomeStatus.UNRESOLVED)) == unk and is_unresolved_tool_result(unk)
    for ok in ("[run b exited exit=0]\nEXIT CODE: 0\nfine", "RUNS (probed now):\n  api server running"):
        assert not is_unresolved_tool_result(ok) and str(R._headed(T(ok))) == ok
    # r10: a declared OK outcome is never unresolved by its text (a file holding saved run output)
    for txt in ("RUN UNRESOLVED — [run api — running] no result yet", unk):
        assert not is_unresolved_tool_result(T(txt, status=OutcomeStatus.OK))
        assert is_unresolved_tool_result(T(txt, status=OutcomeStatus.OK, declared=False))   # coerced: by text
    # the gate gets TEXT in production (row_call_facts str()s the content): a headed result is enough
    from ghost_agent.core.evidence_gate import assess_turn_evidence
    a = assess_turn_evidence([{"name": "run", "content": R._headed(T("[run x] odd", status=OutcomeStatus.UNRESOLVED))}])
    assert a.consulted == 1 and a.substantive == 0 and a.empty == ["run: no answer"], a


def test_the_alias_path_heads_a_result_once():
    """r10 minor 4: tool_run heads, then _with_hint heads again — idempotent for every state."""
    from ghost_agent.tools.outcome import ToolOutcome as T
    for t, st in (("[run x] stop sent, but the backend did not answer the check", OutcomeStatus.UNRESOLVED),
                  ("RUN ERROR (no_answer): x", OutcomeStatus.UNRESOLVED),
                  ("LAB ERROR (z): q", OutcomeStatus.FAILED), ("[run x stopped]", OutcomeStatus.OK)):
        once = R._headed(T(t, status=st))
        assert str(R._headed(once)) == str(once), str(once)
        hinted = str(R._with_hint(once, "(hint)"))
        assert hinted == f"{once}\n(hint)" and hinted.count("RUN UNRESOLVED") + hinted.count("RUN ERROR") <= 1
