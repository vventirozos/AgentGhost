"""§4SX Phase C: the `run` tool (src/ghost_agent/tools/runs.py), its host-side registry
(memory/run_registry.py) and the one new jobs-supervisor entry (`start_detached`).

Workspace runs drive the REAL ServiceSupervisor / SandboxJobSupervisor over the fake sandboxes their own
test files use; lab runs go through the lab tool's `_call` seam (FakeLabs, as in test_4sx_b_lab_tool.py).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import request_scope as RS
from ghost_agent.tools import labs as L
from ghost_agent.tools import runs as R
from ghost_agent.tools.outcome import OutcomeStatus

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_4sx_b_lab_tool import FakeLabs, LAB  # noqa: E402
from test_sandbox_services import FakeSandbox as SvcSandbox, happy_handler  # noqa: E402
from test_sandbox_job_promotion import FakeSandbox as JobSandbox  # noqa: E402


def _ctx(tmp_path, *, sandbox=None, pid=None, conv="conv1", host="127.0.0.1:8700"):
    kf = tmp_path / "labs.key"
    kf.write_text("k" * 64)
    mem = tmp_path / "memory"
    mem.mkdir(exist_ok=True)
    sb = tmp_path / "sbx"
    sb.mkdir(exist_ok=True)
    return SimpleNamespace(args=SimpleNamespace(labs_host=host, labs_key_file=str(kf)), memory_dir=mem,
                           sandbox_dir=sb, sandbox_manager=sandbox, tor_proxy="socks5://127.0.0.1:9050",
                           current_project_id=pid, conversation_key=conv)


@pytest.fixture
def scope():
    toks = []

    def _set(ctx, origin="owner"):
        toks.append(RS.install(RS.RequestScope(conversation_key=ctx.conversation_key,
                                               project_id=ctx.current_project_id, origin=origin, context=ctx)))
    yield _set
    for t in reversed(toks):
        RS.reset(t)


@pytest.fixture
def fake(monkeypatch):
    f = FakeLabs()
    monkeypatch.setattr(L, "_call", f)
    return f


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    import ghost_agent.sandbox.services as svc
    monkeypatch.setattr(svc.time, "sleep", lambda s: None)


def _ok(r):
    return r.status is OutcomeStatus.OK


# ------------------------------------------------------------------ owners and refusals

def test_services_key_half_is_safe_and_per_conversation():
    assert R.services_key_half("project:41f745517fe4") == "41f745517fe4"
    a = R.services_key_half("conversation:slack:dm:U1")
    b = R.services_key_half("conversation:slack:dm:U2")
    assert a == "c-" + hashlib.sha1(b"slack:dm:U1").hexdigest()[:12] and a != b
    from ghost_agent.sandbox.services import _SAFE_PROJECT_ID_RE
    assert _SAFE_PROJECT_ID_RE.match(a)
    assert R.services_key_half("") is None and R.services_key_half("project:../x") is None


async def test_a_member_is_refused(tmp_path, scope):
    c = _ctx(tmp_path)
    scope(c, "member")
    r = await R.tool_run(c, action="status")
    assert r.status is OutcomeStatus.REJECTED and r.reason_code == "run_not_allowed"


async def test_no_owner_is_refused(tmp_path, scope):
    c = _ctx(tmp_path, conv="")
    scope(c)
    r = await R.tool_run(c, action="start", name="a", command="x", where="workspace", lifetime="job")
    assert r.reason_code == "run_no_owner"
    # §4SX D5 review M4: a bare status in an owner-less turn is the jobs tool's own (its jobs only)
    assert getattr(await R.tool_run(c, action="status"), "reason_code", None) != "run_no_owner"


@pytest.mark.parametrize("args,code", [
    ({"action": "nope"}, "run_bad_request"),
    ({"action": "start", "name": "Bad Name", "command": "x", "where": "workspace", "lifetime": "job"}, "run_bad_request"),
    ({"action": "start", "name": "a", "command": "x", "lifetime": "job"}, "run_bad_request"),          # no where
    ({"action": "start", "name": "a", "command": "", "where": "workspace", "lifetime": "job"}, "run_bad_request"),
    ({"action": "start", "name": "a", "command": "x", "where": "workspace", "lifetime": "app"}, "run_app_via_keep"),
    ({"action": "start", "name": "a", "command": "x", "where": f"{LAB}/db", "lifetime": "server"}, "run_bad_request"),
    ({"action": "start", "name": "a", "command": "x", "where": "lab_xyz/db", "lifetime": "job"}, "run_bad_request"),
    ({"action": "start", "name": "a", "command": "x", "where": "workspace", "lifetime": "job", "port": "x"},
     "run_bad_request"),
    ({"action": "stop", "name": "ghost"}, "run_unknown"),
])
async def test_bad_calls_are_refused_before_anything_runs(tmp_path, scope, fake, args, code):
    c = _ctx(tmp_path)
    scope(c)
    r = await R.tool_run(c, **args)
    assert r.status is OutcomeStatus.REJECTED and r.reason_code == code, str(r)
    assert fake.calls == [] and r.world_changed is False


# ------------------------------------------------------------------ workspace server (real ServiceSupervisor)

async def test_a_workspace_server_is_a_services_row_under_the_conversations_key(tmp_path, scope):
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    r = await R.tool_run(c, action="start", name="web", command="python3 -m http.server 8100",
                         where="workspace", lifetime="server", port=8100)
    assert _ok(r) and r.world_changed is True and "not a result" in r
    half = R.services_key_half("conversation:conv1")
    reg = json.loads((tmp_path / "ws" / ".services" / "registry.json").read_text())
    assert f"{half}:web" in reg and "web" not in reg                        # design r1 H3
    from ghost_agent.memory.run_registry import registry_for
    row = registry_for(c).live_by_name("conversation:conv1", "web")
    assert row["backend"] == "services" and row["backend_ref"] == f"{half}:web"
    assert row["backend_token"] == reg[f"{half}:web"]["token"]
    s = await R.tool_run(c, action="status")
    assert "web  server  workspace  running" in s
    again = await R.tool_run(c, action="start", name="web", command="x", where="workspace", lifetime="server")
    assert again.reason_code == "run_exists"
    u = await R.tool_run(c, action="url", name="web")
    assert "127.0.0.1:8100" in u
    st = await R.tool_run(c, action="stop", name="web")
    assert _ok(st) and st.world_changed is True
    assert registry_for(c).latest_by_name("conversation:conv1", "web")["state"] == "stopped"


async def test_a_service_run_did_not_start_is_never_replaced(tmp_path, scope):
    """design r1 H3: a live service under the same key, started outside `run`, is refused."""
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    from ghost_agent.sandbox.services import get_service_supervisor
    get_service_supervisor(sb).start("web", "python3 app.py", port=8100,
                                     project_id=R.services_key_half("conversation:conv1"))
    n = len(sb.calls)
    r = await R.tool_run(c, action="start", name="web", command="python3 other.py", where="workspace",
                         lifetime="server")
    assert r.reason_code == "run_exists" and not any("nohup" in x for x in sb.calls[n:])


async def test_a_start_outside_run_that_reuses_the_name_reads_as_replaced(tmp_path, scope):
    """design r1 M3: the token decides, not prose."""
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace", lifetime="server")
    from ghost_agent.sandbox.services import get_service_supervisor
    get_service_supervisor(sb).restart("web", project_id=R.services_key_half("conversation:conv1"))
    s = await R.tool_run(c, action="status", name="web")
    assert "replaced" in s
    st = await R.tool_run(c, action="stop", name="web")
    assert st.reason_code == "run_replaced"


async def test_a_server_that_did_not_come_up_is_failed_and_not_recorded(tmp_path, scope):
    sb = SvcSandbox(tmp_path / "ws", lambda cmd: ("", 1) if "nohup" in cmd else ("", 0))
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    r = await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace", lifetime="server")
    assert r.status is OutcomeStatus.FAILED and r.reason_code == "run_start_failed"
    from ghost_agent.memory.run_registry import registry_for
    assert registry_for(c).rows("conversation:conv1") == []


async def test_two_conversations_get_two_services(tmp_path, scope):
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    for conv in ("a", "b"):
        c = _ctx(tmp_path, sandbox=sb, conv=conv)
        scope(c)
        r = await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace",
                             lifetime="server")
        assert _ok(r), str(r)
    reg = json.loads((tmp_path / "ws" / ".services" / "registry.json").read_text())
    assert {k for k in reg if k.endswith(":web")} == {f"{R.services_key_half('conversation:' + x)}:web" for x in "ab"}


# ------------------------------------------------------------------ workspace job (real SandboxJobSupervisor)

def seed_job(ctx, sandbox_id, status="running", label="x"):
    """A jobs-registry row for a promoted sandbox job (review 11-05 r14: `run … name=<job id>` reads THIS record)."""
    from ghost_agent.core.jobs import get_job_registry
    reg = get_job_registry(ctx)
    j = reg.register("sandbox", label, sandbox_job_id=sandbox_id)
    if status != "running":
        reg.finish(j.id, status=status, result="r")
    return j


def _kill_jobs(sb):
    from ghost_agent.sandbox.jobs import get_job_supervisor
    for e in get_job_supervisor(sb).list_entries():
        try:
            pid = int(e.get("pid"))
            os.killpg(pid, signal.SIGKILL) if os.getpgid(pid) == pid else os.kill(pid, signal.SIGKILL)
        except (OSError, TypeError, ValueError):
            pass


async def test_a_workspace_job_starts_detached_at_once_and_collects_when_done(tmp_path, scope):
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    try:
        t0 = time.monotonic()
        r = await R.tool_run(c, action="start", name="hello", command="echo hello-run; sleep 1", where="workspace",
                             lifetime="job")
        assert _ok(r) and time.monotonic() - t0 < 10, str(r)          # design r1 H1: no budget wait
        from ghost_agent.sandbox.jobs import get_job_supervisor
        (e,) = get_job_supervisor(sb).list_entries()
        assert e["started_via"] == "run" and e["label"] == "hello"
        early = await R.tool_run(c, action="collect", name="hello")
        assert early.status in (OutcomeStatus.UNRESOLVED, OutcomeStatus.OK)
        for _ in range(60):
            out = await R.tool_run(c, action="collect", name="hello")
            if out.status is not OutcomeStatus.UNRESOLVED:
                break
            await asyncio.sleep(0.25)
        assert _ok(out) and "hello-run" in out and "EXIT CODE: 0" in out, str(out)
    finally:
        _kill_jobs(sb)


async def test_a_quiet_job_is_not_killed_and_stop_cancels_it(tmp_path, scope):
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    try:
        r = await R.tool_run(c, action="start", name="quiet", command="sleep 30", where="workspace", lifetime="job")
        assert _ok(r)
        s = await R.tool_run(c, action="status", name="quiet")
        assert "running" in s
        st = await R.tool_run(c, action="stop", name="quiet")
        assert _ok(st), str(st)
        from ghost_agent.sandbox.jobs import get_job_supervisor
        (e,) = get_job_supervisor(sb).list_entries()
        assert e["state"] == "cancelled"
    finally:
        _kill_jobs(sb)


async def test_the_jobs_tool_does_not_adopt_a_run_job(tmp_path, scope):
    """design r1 H2: one run, reported once."""
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    try:
        await R.tool_run(c, action="start", name="quiet", command="sleep 30", where="workspace", lifetime="job")
        from ghost_agent.core.jobs import JobRegistry
        from ghost_agent.tools.delegate import SANDBOX_KIND, _sync_sandbox_jobs
        reg = JobRegistry()
        _sync_sandbox_jobs(reg, c)
        assert reg.list(kind=SANDBOX_KIND) == []
    finally:
        _kill_jobs(sb)


def test_start_detached_respects_the_cap_and_the_rerun_guard(tmp_path, monkeypatch):
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    sb = JobSandbox(tmp_path / "ws")
    sup = SandboxJobSupervisor(sb)
    monkeypatch.setenv("GHOST_SANDBOX_JOB_MAX", "1")
    try:
        e, why = sup.start_detached("sleep 20", label="a")
        assert e and not why
        dup, _ = sup.start_detached("sleep 20", label="a")
        assert dup["duplicate"] is True and dup["id"] == e["id"]
        none, why = sup.start_detached("sleep 21", label="b")
        assert none is None and "already running" in why
        assert len(sup.list_entries()) == 1
    finally:
        _kill_jobs(sb)


def test_a_run_jobs_wake_names_run_collect():
    from ghost_agent.main import _collect_hint
    assert _collect_hint({"started_via": "run", "label": "bench"}) == "run(action='collect', name='bench')"
    assert _collect_hint({"label": "x"}) == "run(action='collect')"     # review 11-05 W-M2


# ------------------------------------------------------------------ lab runs (FakeLabs)

def _lab(fake, owner="conversation:conv1", origin="owner"):
    fake.on("GET", f"/v1/labs/{LAB}", (200, {"lab_id": LAB, "owner": owner, "origin": origin, "state": "ready"}))

async def test_a_lab_job_is_a_detached_exec_wrapped_in_a_timeout(tmp_path, scope, fake):
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"lab_id": LAB, "member": "db", "job_id": "job_aaaa000001"}))
    r = await R.tool_run(c, action="start", name="bench", command="pgbench -T 60", where=f"{LAB}/db", lifetime="job")
    assert _ok(r) and r.world_changed is True
    call = fake.calls[-1]
    assert call.method == "POST" and call.json["detach"] is True and call.json["member"] == "db"
    # design r1 M4; --foreground keeps it in the job's process group, which `run stop` kills (live 11-04)
    assert call.json["cmd"] == f"timeout --foreground -k 30 {R.LAB_JOB_TTL_S} bash -c 'pgbench -T 60'"
    assert call.origin == "owner"
    path = f"/v1/labs/{LAB}/jobs/job_aaaa000001"
    fake.on("GET", path, (200, {"state": "running", "exit_code": None, "output_tail": "tps 10"}))
    s = await R.tool_run(c, action="status")
    assert f"bench  job  {LAB}/db  running" in s
    co = await R.tool_run(c, action="collect", name="bench")
    assert co.status is OutcomeStatus.UNRESOLVED
    fake.on("GET", path, (200, {"state": "exited", "exit_code": 0, "output_tail": "tps 99\n"}))
    co = await R.tool_run(c, action="collect", name="bench")
    assert _ok(co) and "tps 99" in co and "EXIT CODE: 0" in co
    fake.on("GET", path, (200, {"state": "exited", "exit_code": 2, "output_tail": "boom"}))
    from ghost_agent.memory.run_registry import registry_for
    rows = registry_for(c).rows("conversation:conv1")
    assert rows[-1]["state"] == "exited"


async def test_a_lab_run_whose_lab_was_destroyed_is_lost_and_a_kill_is_stopped(tmp_path, scope, fake):
    """design r1 L1."""
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_bbbb000001"}))
    await R.tool_run(c, action="start", name="api", command="python3 -m http.server 80", where=f"{LAB}/db",
                     lifetime="server", port=80)
    path = f"/v1/labs/{LAB}/jobs/job_bbbb000001"
    # the real host: GET never says `killed`; after a group kill there is no .rc → `lost` (review r1 minor 2)
    fake.on("GET", path, (200, {"state": "running", "exit_code": None, "output_tail": ""}),
            (200, {"state": "lost", "exit_code": None, "output_tail": ""}))
    fake.on("DELETE", path, (200, {"state": "exited", "killed": True, "exit_code": None}))
    st = await R.tool_run(c, action="stop", name="api")
    assert _ok(st) and st.world_changed is True
    from ghost_agent.memory.run_registry import registry_for
    assert registry_for(c).latest_by_name("conversation:conv1", "api")["state_note"] == "killed"
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_bbbb000002"}))
    fake.on("GET", f"/v1/labs/{LAB}/jobs/job_bbbb000001", (200, {"state": "lost"}))
    await R.tool_run(c, action="start", name="api", command="x", where=f"{LAB}/db", lifetime="server", port=80)
    fake.on("GET", f"/v1/labs/{LAB}/jobs/job_bbbb000002", (404, {"error": "unknown_lab", "message": "gone"}))
    s = await R.tool_run(c, action="status", name="api")
    assert "lost (its lab was destroyed)" in s


async def test_a_lab_run_needs_a_configured_host(tmp_path, scope, fake):
    c = _ctx(tmp_path, host="")
    scope(c)
    r = await R.tool_run(c, action="start", name="a", command="x", where=f"{LAB}/db", lifetime="job")
    assert r.reason_code == "labs_unconfigured" and fake.calls == []


async def test_a_lab_start_with_no_answer_is_unresolved(tmp_path, scope, fake):
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    fake.on("POST", f"/v1/labs/{LAB}/exec", L.LabsNoAnswer("ReadTimeout"))
    r = await R.tool_run(c, action="start", name="a", command="x", where=f"{LAB}/db", lifetime="job")
    assert r.status is OutcomeStatus.UNRESOLVED and "may have started" in r


async def test_a_probe_changes_only_its_own_runs(tmp_path, scope, fake):
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_cccc000001"}))
    await R.tool_run(c, action="start", name="api", command="x", where=f"{LAB}/db", lifetime="job")
    scope(c, "probe")
    r = await R.tool_run(c, action="stop", name="api")
    assert r.reason_code == "run_not_allowed"


async def test_restarting_a_job_is_refused(tmp_path, scope, fake):
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_dddd000001"}))
    await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    fake.on("GET", f"/v1/labs/{LAB}/jobs/job_dddd000001", (200, {"state": "running"}))
    r = await R.tool_run(c, action="restart", name="j")
    assert r.status is OutcomeStatus.REJECTED and "not restarted" in r


# ------------------------------------------------------------------ not advertised in C

def test_run_is_advertised_and_the_old_names_are_aliases_after_the_d5_switch():
    """§4SX D5: `run` is in the schema; `manage_services` / `jobs` are not, but stay dispatchable."""
    import ast
    from ghost_agent.tools import registry as REG
    names = {(d.get("function") or {}).get("name") for d in REG.TOOL_DEFINITIONS}
    assert "run" in names and not {"manage_services", "jobs"} & names
    tree = ast.parse(Path(REG.__file__).read_text())
    keys = [k.value for n in ast.walk(tree) if isinstance(n, ast.Dict) for k in n.keys
            if isinstance(k, ast.Constant)]
    assert {"run", "manage_services", "jobs"} <= set(keys)


def test_the_phase_c_operator_probe_route_is_gone():
    """§4SX D5: `run` is advertised; live checks use probe-header chat turns."""
    from ghost_agent.api import routes as RT
    assert not hasattr(RT, "operator_run_probe")
    assert not [r for r in RT.router.routes if getattr(r, "path", "") == "/api/operator/run-probe"]


async def test_a_lost_service_run_never_reaches_another_projects_service_by_name(tmp_path, scope):
    """r1 M1: with its own key gone, stop/restart/logs refuse — the name would resolve to B's `web`."""
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace", lifetime="server")
    from ghost_agent.sandbox.services import get_service_supervisor
    sup = get_service_supervisor(sb)
    sup.start("web", "python3 b.py", project_id="bbb222")                  # project B's web
    reg = sup._load()
    reg.pop(f"{R.services_key_half('conversation:conv1')}:web")
    sup._save(reg)
    n = len(sb.calls)
    for act in ("logs", "restart"):
        r = await R.tool_run(c, action=act, name="web")
        assert r.status is OutcomeStatus.REJECTED, (act, str(r))
    r = await R.tool_run(c, action="stop", name="web")
    assert ("already lost" in r or r.reason_code == "run_ended") and r.world_changed is False
    assert not any("kill" in x or "nohup" in x for x in sb.calls[n:])
    assert "bbb222:web" in sup._load()


async def test_a_restart_that_did_not_restart_is_failed(tmp_path, scope, monkeypatch):
    """r1 M2: the same token after `restart` = nothing restarted."""
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace", lifetime="server")
    from ghost_agent.sandbox.services import get_service_supervisor
    monkeypatch.setattr(get_service_supervisor(sb), "restart", lambda *a, **k: "cannot restart — survived TERM+KILL")
    r = await R.tool_run(c, action="restart", name="web")
    # r2 M1: the old process survived the kill — FAILED, nothing changed, the run is still running
    assert r.status is OutcomeStatus.FAILED and r.reason_code == "run_restart_failed" and r.world_changed is False
    from ghost_agent.memory.run_registry import registry_for
    assert registry_for(c).live_by_name("conversation:conv1", "web")["state"] == "running"
    assert "running" in await R.tool_run(c, action="status", name="web")


async def test_an_unrecorded_start_is_undone(tmp_path, scope, monkeypatch):
    """r1 M3: no row → the process is stopped again, never left untracked."""
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    from ghost_agent.memory.run_registry import RunRegistry
    monkeypatch.setattr(RunRegistry, "add", lambda self, **k: None)
    try:
        r = await R.tool_run(c, action="start", name="j", command="sleep 20", where="workspace", lifetime="job")
        assert r.status is OutcomeStatus.FAILED and r.reason_code == "run_registry_unavailable"
        from ghost_agent.sandbox.jobs import get_job_supervisor
        (e,) = get_job_supervisor(sb).list_entries()
        assert e["state"] == "cancelled"
    finally:
        _kill_jobs(sb)


async def test_two_starts_of_one_name_at_once_start_one(tmp_path, scope):
    """r1 M4."""
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    a, b = await asyncio.gather(*(R.tool_run(c, action="start", name="web", command="python3 a.py",
                                             where="workspace", lifetime="server") for _ in range(2)))
    assert sorted([a.status.value, b.status.value]) == ["ok", "rejected"]
    from ghost_agent.memory.run_registry import registry_for
    assert len(registry_for(c).rows("conversation:conv1")) == 1


async def test_stopping_a_finished_job_keeps_its_result(tmp_path, scope, fake):
    """r1 minor 1."""
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_eeee000001"}))
    await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    fake.on("GET", f"/v1/labs/{LAB}/jobs/job_eeee000001", (200, {"state": "exited", "exit_code": 0, "output_tail": "ok"}))
    r = await R.tool_run(c, action="stop", name="j")
    assert "already exited" in r and r.world_changed is False
    assert not any(x.method == "DELETE" for x in fake.calls)


async def test_a_lab_job_the_host_forgot_is_lost_not_stuck(tmp_path, scope, fake):
    """r1 minor 3: unknown_job → lost, so the name can be started again."""
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_ffff000001"}), (202, {"job_id": "job_ffff000002"}))
    await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    fake.on("GET", f"/v1/labs/{LAB}/jobs/job_ffff000001", (404, {"error": "unknown_job"}))
    r = await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    assert _ok(r), str(r)


async def test_a_legacy_unscoped_service_is_never_taken_over(tmp_path, scope):
    """r1 minor 4."""
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    from ghost_agent.sandbox.services import get_service_supervisor
    get_service_supervisor(sb).start("web", "python3 legacy.py")
    r = await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace", lifetime="server")
    assert r.reason_code == "run_exists" and "web" in get_service_supervisor(sb)._load()


async def test_a_probe_starts_lab_runs_only_on_a_probes_lab_and_other_owners_are_named(tmp_path, scope, fake):
    """r1 minor 5."""
    c = _ctx(tmp_path)
    scope(c, "probe")
    _lab(fake, owner="conversation:other", origin="owner")
    r = await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    assert r.reason_code == "run_not_allowed" and not any(x.method == "POST" for x in fake.calls)
    scope(c)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_abab000001"}))
    r = await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    assert _ok(r) and "belongs to conversation:other" in r


def test_the_sweeper_line_names_run_collect_for_a_run_job():
    """r1 minor 6: the sweeper's landing line (not only the capped one)."""
    import ast
    from ghost_agent import main as M
    src = Path(M.__file__).read_text()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.AsyncFunctionDef)
              and n.name == "_reap_sandbox_jobs")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_collect_hint"]
    assert calls


async def test_a_missing_project_folder_is_a_failed_start_not_a_crash(tmp_path, scope, monkeypatch):
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb, pid="p404")
    scope(c)
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor

    def boom(self, *a, **k):
        raise RuntimeError("job launcher failed (exit 2): cd: no such dir")
    monkeypatch.setattr(SandboxJobSupervisor, "start_detached", boom)
    r = await R.tool_run(c, action="start", name="j", command="x", where="workspace", lifetime="job")
    assert r.status is OutcomeStatus.FAILED and "no such dir" in r


async def test_a_portless_server_url(tmp_path, scope):
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    from ghost_agent.memory.run_registry import registry_for
    registry_for(c).add(name="w", place="workspace", lifetime="server", command="x", port=None,
                        owner="conversation:conv1", origin="owner", backend="services", backend_ref="k")
    r = await R.tool_run(c, action="url", name="w")
    assert "portless" in r and "None" not in r



# ------------------------------------------------------------------ review r2

async def test_a_cancelled_lab_start_is_still_recorded(tmp_path, scope, fake):
    """r2 M2: the POST → record pair is shielded."""
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    gate = asyncio.Event()
    orig = fake.__call__

    async def slow(context, method, path, **kw):
        if method == "POST":
            await gate.wait()
        return await orig(context, method, path, **kw)
    L._call = slow
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_1234567890"}))
    t = asyncio.ensure_future(R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job"))
    await asyncio.sleep(0.05)
    t.cancel()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await t
    await asyncio.sleep(0.05)
    from ghost_agent.memory.run_registry import registry_for
    (row,) = registry_for(c).rows("conversation:conv1")
    assert row["backend_ref"] == f"{LAB}/job_1234567890"


async def test_an_unrecorded_lab_start_says_whether_the_kill_was_confirmed(tmp_path, scope, fake, monkeypatch):
    """r2 minor 2."""
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    from ghost_agent.memory.run_registry import RunRegistry
    monkeypatch.setattr(RunRegistry, "add", lambda self, **k: None)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_aa00000001"}))
    fake.on("DELETE", f"/v1/labs/{LAB}/jobs/job_aa00000001", L.LabsNoAnswer("ReadTimeout"))
    r = await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    assert "NOT confirmed" in r and "may still run" in r
    fake.on("DELETE", f"/v1/labs/{LAB}/jobs/job_aa00000001", (200, {"state": "exited", "killed": True}))
    r = await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    assert "was killed again" in r


async def test_collect_and_logs_of_a_stopped_run_keep_its_recorded_end(tmp_path, scope, fake):
    """r2 minor 4: a stopped lab job re-probes as `lost`; collect keeps `stopped`."""
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_bb00000001"}))
    await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    path = f"/v1/labs/{LAB}/jobs/job_bb00000001"
    fake.on("GET", path, (200, {"state": "running", "output_tail": "partial"}))
    fake.on("DELETE", path, (200, {"state": "exited", "killed": True}))
    await R.tool_run(c, action="stop", name="j")
    fake.on("GET", path, (200, {"state": "lost", "output_tail": "partial"}))
    co = await R.tool_run(c, action="collect", name="j")
    assert "[run j stopped" in co and "lost" not in co
    lg = await R.tool_run(c, action="logs", name="j")
    assert "stopped" in lg and "partial" in lg
    from ghost_agent.memory.run_registry import registry_for
    assert registry_for(c).latest_by_name("conversation:conv1", "j")["state"] == "stopped"


async def test_logs_after_a_run_stop_say_stopped_not_stopped_outside_run(tmp_path, scope):
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace", lifetime="server")
    await R.tool_run(c, action="stop", name="web")
    lg = await R.tool_run(c, action="logs", name="web")
    assert "outside" not in lg and lg.status is OutcomeStatus.OK


async def test_a_job_that_ends_before_the_cancel_keeps_its_end(tmp_path, scope, monkeypatch):
    """r2 minor 3."""
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    try:
        await R.tool_run(c, action="start", name="j", command="sleep 30", where="workspace", lifetime="job")
        from ghost_agent.sandbox.jobs import SandboxJobSupervisor
        monkeypatch.setattr(SandboxJobSupervisor, "cancel", lambda self, jid: (False, "already done — nothing to cancel"))
        calls = {"n": 0}
        real = R._probe

        async def probe(context, row, origin):
            calls["n"] += 1
            if calls["n"] == 1:
                return "running", "", {}
            return "exited", "exit 0", {"exit_code": 0}
        monkeypatch.setattr(R, "_probe", probe)
        r = await R.tool_run(c, action="stop", name="j")
        monkeypatch.setattr(R, "_probe", real)
        assert "already exited" in r and r.world_changed is False
        from ghost_agent.memory.run_registry import registry_for
        assert registry_for(c).latest_by_name("conversation:conv1", "j")["state"] == "exited"
    finally:
        _kill_jobs(sb)


async def test_a_crashed_server_restarts_whether_or_not_status_ran_first(tmp_path, scope):
    """r2 minor 5."""
    alive = {"v": True}

    def handler(cmd):
        if "kill -0" in cmd and "sig TERM" not in cmd:
            return ("", 0 if alive["v"] else 1)
        return happy_handler()(cmd)
    sb = SvcSandbox(tmp_path / "ws", handler)
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace", lifetime="server")
    alive["v"] = False
    assert "exited" in await R.tool_run(c, action="status", name="web")
    alive["v"] = True
    r = await R.tool_run(c, action="restart", name="web")
    assert r.status is OutcomeStatus.OK, str(r)



# ------------------------------------------------------------------ review r3

async def test_a_refused_job_kill_is_a_failed_stop(tmp_path, scope, monkeypatch):
    """r3 MAJOR 1: cancel → (False, "could not kill … left RUNNING") is not "already running" OK."""
    sb = JobSandbox(tmp_path / "ws")
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    try:
        await R.tool_run(c, action="start", name="j", command="sleep 30", where="workspace", lifetime="job")
        from ghost_agent.sandbox.jobs import SandboxJobSupervisor
        monkeypatch.setattr(SandboxJobSupervisor, "cancel",
                            lambda self, jid: (False, "Error: could not kill job x. Left RUNNING."))
        r = await R.tool_run(c, action="stop", name="j")
        assert r.status is OutcomeStatus.FAILED and r.reason_code == "run_stop_failed"
    finally:
        _kill_jobs(sb)


async def test_collect_of_a_passed_job_keeps_its_exit_code_when_the_backend_forgot_it(tmp_path, scope, fake):
    """r3 minor 1."""
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_cc00000001"}))
    await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    path = f"/v1/labs/{LAB}/jobs/job_cc00000001"
    fake.on("GET", path, (200, {"state": "exited", "exit_code": 0, "output_tail": "done"}))
    assert _ok(await R.tool_run(c, action="collect", name="j"))
    fake.on("GET", path, (404, {"error": "unknown_lab"}))                       # the lab is gone now
    r = await R.tool_run(c, action="collect", name="j")
    assert _ok(r) and "EXIT CODE: 0" in r
    fake.on("GET", path, L.LabsNoAnswer("ReadTimeout"))
    r = await R.tool_run(c, action="collect", name="j")
    assert r.status is OutcomeStatus.UNRESOLVED


async def test_an_unrecorded_restart_is_stopped_again(tmp_path, scope, monkeypatch):
    """r3 minor 2."""
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace", lifetime="server")
    from ghost_agent.memory.run_registry import RunRegistry
    monkeypatch.setattr(RunRegistry, "add", lambda self, **k: None)
    r = await R.tool_run(c, action="restart", name="web")
    assert r.reason_code == "run_registry_unavailable"
    from ghost_agent.sandbox.services import get_service_supervisor
    row = get_service_supervisor(sb)._load().get(f"{R.services_key_half('conversation:conv1')}:web")
    assert row is None or not row.get("pid")


async def test_a_cancelled_start_keeps_its_lock_until_its_work_is_done(tmp_path, scope, fake):
    """r3 minor 3: a second start of the name waits for the first, even after the first's turn is cancelled."""
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    gate = asyncio.Event()
    orig = fake.__call__

    async def slow(context, method, path, **kw):
        if method == "POST":
            await gate.wait()
        return await orig(context, method, path, **kw)
    L._call = slow
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_dd00000001"}), (202, {"job_id": "job_dd00000002"}))
    fake.on("GET", f"/v1/labs/{LAB}/jobs/job_dd00000001", (200, {"state": "running"}))
    t1 = asyncio.ensure_future(R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job"))
    await asyncio.sleep(0.05)
    t1.cancel()
    t2 = asyncio.ensure_future(R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job"))
    await asyncio.sleep(0.05)
    gate.set()
    r2 = await t2
    assert r2.reason_code == "run_exists", str(r2)
    from ghost_agent.memory.run_registry import registry_for
    assert len(registry_for(c).rows("conversation:conv1")) == 1



# ------------------------------------------------------------------ review r4

async def test_a_run_stop_is_recorded_as_its_own_stop(tmp_path, scope):
    """r4 minor 1: not "stopped outside run"."""
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace", lifetime="server")
    await R.tool_run(c, action="stop", name="web")
    s = await R.tool_run(c, action="status", name="web")
    assert "outside" not in s and "stopped (stopped by run)" in s


async def test_nothing_starts_after_the_user_said_stop(tmp_path, scope, fake, monkeypatch):
    """r4 minor 2: the shield keeps a started step whole, but no backend step starts after a stop."""
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    monkeypatch.setattr(R, "_stop_asked", lambda: True)
    r = await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    assert r.reason_code == "run_stopped_by_user" and r.world_changed is False
    assert not any(x.method == "POST" for x in fake.calls)


async def test_a_restart_of_a_crashed_run_keeps_its_recorded_end_and_no_answer_is_unresolved(tmp_path, scope,
                                                                                            monkeypatch):
    """r4 minor 3."""
    alive = {"v": True}

    def handler(cmd):
        if "kill -0" in cmd and "sig TERM" not in cmd:
            return ("", 0 if alive["v"] else 1)
        return happy_handler()(cmd)
    sb = SvcSandbox(tmp_path / "ws", handler)
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace", lifetime="server")
    alive["v"] = False
    await R.tool_run(c, action="status", name="web")                 # records `exited`
    from ghost_agent.memory.run_registry import registry_for
    first = registry_for(c).latest_by_name("conversation:conv1", "web")
    alive["v"] = True
    assert _ok(await R.tool_run(c, action="restart", name="web"))
    assert registry_for(c).get(first["run_id"])["state"] == "exited"          # its crash, kept
    from ghost_agent.sandbox.services import get_service_supervisor
    sup = get_service_supervisor(sb)
    monkeypatch.setattr(sup, "_entry_state", lambda e: None)
    monkeypatch.setattr(sup, "restart", lambda *a, **k: "?")
    monkeypatch.setattr(R, "_probe", lambda *a, **k: asyncio.sleep(0, ("running", "", {})))
    r = await R.tool_run(c, action="restart", name="web")
    assert r.status is OutcomeStatus.UNRESOLVED and r.reason_code == "run_unverified"


async def test_a_lab_restart_of_an_ended_server_does_not_claim_a_stop(tmp_path, scope, fake):
    """r4 minor 4."""
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_ee00000001"}))
    await R.tool_run(c, action="start", name="api", command="x", where=f"{LAB}/db", lifetime="server", port=80)
    fake.on("GET", f"/v1/labs/{LAB}/jobs/job_ee00000001", (200, {"state": "exited", "exit_code": 1}))
    fake.on("GET", f"/v1/labs/{LAB}", (404, {"error": "unknown_lab", "message": "gone"}))
    r = await R.tool_run(c, action="restart", name="api")
    assert "had already ended" in str(r) and "WAS stopped" not in str(r) and r.world_changed is False, str(r)


async def test_an_unrecorded_start_whose_stop_fails_says_so(tmp_path, scope, monkeypatch):
    sb = SvcSandbox(tmp_path / "ws", happy_handler())
    c = _ctx(tmp_path, sandbox=sb)
    scope(c)
    from ghost_agent.memory.run_registry import RunRegistry
    monkeypatch.setattr(RunRegistry, "add", lambda self, **k: None)

    async def never(svc, key):
        return False
    monkeypatch.setattr(R, "_svc_undo", never)
    r = await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace", lifetime="server")
    assert "stopping it again FAILED" in r
