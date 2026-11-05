"""§4SX end-to-end review r12 (2026-11-05) — the fresh-eye reviewers' repros of the r11 fixes, inverted.

Each test failed on the r11 code (the reviewer's repro asserted the defect)."""
import base64
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_4sx_b_lab_tool import FakeLabs, LAB, _detail  # noqa: E402
from test_4sx_b_lab_tool import _ctx as _lab_ctx  # noqa: E402
from test_4sx_c_run_tool import _ctx as _run_ctx, _kill_jobs  # noqa: E402
from test_sandbox_job_promotion import FakeSandbox as JobSandbox  # noqa: E402
from ghost_agent.core import request_scope as RS  # noqa: E402
from ghost_agent.memory.lab_inventory import inventory_for  # noqa: E402
from ghost_agent.tools import delegate as D  # noqa: E402
from ghost_agent.tools import labs as L  # noqa: E402
from ghost_agent.tools import runs as R  # noqa: E402
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome  # noqa: E402

SRC = str(Path(__file__).parent.parent / "src")


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
    monkeypatch.setattr(L, "_time_left_s", lambda: None)
    return f


# ================================================================== LAB

@pytest.mark.asyncio
async def test_F1_an_unanswered_ownership_check_refuses_an_autopilot_destroy(tmp_path, fake, scope):
    ctx = _lab_ctx(tmp_path, pid="p1", conv="")
    scope(ctx, "leaf")
    fake.on("GET", f"/v1/labs/{LAB}", L.LabsNoAnswer("ReadTimeout"),
            (200, _detail(owner="conversation:owner-chat", origin="owner")))
    fake.on("DELETE", f"/v1/labs/{LAB}", (200, {"lab_id": LAB}))
    out = await L.tool_lab(ctx, action="destroy", lab_id=LAB)
    assert out.status is OutcomeStatus.REJECTED and "did not answer the ownership check" in str(out)
    assert not any(c.method == "DELETE" for c in fake.calls)


@pytest.mark.asyncio
async def test_F1b_an_unanswered_ownership_check_refuses_a_get_from_production(tmp_path, fake, scope):
    ctx = _lab_ctx(tmp_path, pid="p1", conv="")
    scope(ctx, "leaf")
    fake.on("GET", f"/v1/labs/{LAB}", L.LabsNoAnswer("ReadTimeout"))
    fake.on("GET", f"/v1/labs/{LAB}/files", (200, {"content_b64": "c2VjcmV0"}))
    out = await L.tool_lab(ctx, action="get", lab_id=LAB, member="db", path="/etc/app/secrets.env",
                           sandbox_path="x.txt")
    assert out.status is OutcomeStatus.REJECTED
    assert not any(c.path.endswith("/files") for c in fake.calls)


@pytest.mark.asyncio
async def test_F2_a_closed_record_is_never_reported_as_destroyed(tmp_path, fake, scope):
    ctx = _lab_ctx(tmp_path)
    scope(ctx, "owner")
    inv = inventory_for(ctx)
    inv.record_create(LAB, owner="conversation:conv1", origin="owner", template="pg-single", host_id="hostA")
    fake.on("GET", f"/v1/labs/{LAB}", (404, {"error": "unknown_lab", "message": "no such lab"}))
    fake.on("GET", "/v1/health", (200, {"host_id": "hostB"}))
    assert "CLOSED" in str(await L.tool_lab(ctx, action="destroy", lab_id=LAB))
    st = str(await L.tool_lab(ctx, action="status", lab_id=LAB))
    assert "It was destroyed" not in st and "record here was closed" in st and "nothing was deleted" in st
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    assert "destroyed" not in str(await L.tool_lab(ctx, action="list"))


@pytest.mark.asyncio
async def test_F3_leaf_end_keeps_another_hosts_lab_live(tmp_path, fake, scope):
    ctx = _lab_ctx(tmp_path, pid="p1", conv="")
    inv = inventory_for(ctx)
    inv.record_create(LAB, owner="project:p1", origin="autopilot", template="pg-single", host_id="hostA",
                      leaf="abc-a1")
    L._LAST_HOST_ID["id"] = "hostB"
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    fake.on("DELETE", f"/v1/labs/{LAB}", (404, {"error": "unknown_lab"}))
    assert await L.destroy_leaf_labs(ctx, "project:p1", "abc-a1") == []
    assert inv.get(LAB)["destroyed_at"] is None


@pytest.mark.asyncio
async def test_F4_a_probe_with_no_owner_lists_nothing(tmp_path, fake, scope):
    ctx = _lab_ctx(tmp_path, pid=None, conv="")
    scope(ctx, "probe")
    fake.on("GET", "/v1/labs", (200, {"labs": [_detail(owner="conversation:owner-chat")]}))
    out = await L.tool_lab(ctx, action="list")
    assert out.status is OutcomeStatus.REJECTED and not fake.calls


@pytest.mark.asyncio
async def test_F4b_a_probe_never_reads_an_owner_labs_status(tmp_path, fake, scope):
    ctx = _lab_ctx(tmp_path, conv="probe-c")
    scope(ctx, "probe")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="conversation:owner-chat", lifetime="kept")))
    out = await L.tool_lab(ctx, action="status", lab_id=LAB)
    assert out.status is OutcomeStatus.REJECTED and "a probe touches only labs it made" in str(out)


def test_F5_a_fifo_at_the_target_never_blocks_the_writer(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    os.mkfifo(ws / "out.txt")
    code = textwrap.dedent(f"""
        from pathlib import Path
        from ghost_agent.tools.file_system import write_bytes_nofollow_rel
        try:
            write_bytes_nofollow_rel(Path({str(ws)!r}), "out.txt", b"x")
            print("WROTE")
        except OSError as e:
            print("REFUSED", e.errno)
    """)
    r = subprocess.run([sys.executable, "-c", code], timeout=20, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": SRC})
    assert "REFUSED" in r.stdout, (r.stdout, r.stderr)


@pytest.mark.asyncio
async def test_F6_an_unattended_build_moves_files_only_inside_its_project(tmp_path, fake, scope):
    ctx = _lab_ctx(tmp_path, pid="p1", conv="")
    ctx.confine_writes_to_project = True
    other = ctx.sandbox_dir / "projects" / "p2"
    other.mkdir(parents=True)
    (other / "app.py").write_text("ORIGINAL p2 code")
    (other / ".env").write_text("P2_SECRET=hunter2")
    scope(ctx, "leaf")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="project:p1", origin="autopilot")))
    fake.on("GET", f"/v1/labs/{LAB}/files", (200, {"content_b64": base64.b64encode(b"clobbered").decode()}))
    fake.on("PUT", f"/v1/labs/{LAB}/files", (200, {"bytes": 17}))
    g = await L.tool_lab(ctx, action="get", lab_id=LAB, member="db", path="/x",
                         sandbox_path="/workspace/projects/p2/app.py")
    assert g.status is OutcomeStatus.REJECTED and (other / "app.py").read_text() == "ORIGINAL p2 code"
    p = await L.tool_lab(ctx, action="put", lab_id=LAB, member="db", path="/tmp/s",
                         sandbox_path="/workspace/projects/p2/.env")
    assert p.status is OutcomeStatus.REJECTED and not any(c.method == "PUT" for c in fake.calls)
    # its own project still works
    (ctx.sandbox_dir / "projects" / "p1").mkdir(parents=True, exist_ok=True)
    ok = await L.tool_lab(ctx, action="get", lab_id=LAB, member="db", path="/x", sandbox_path="seed.sql")
    assert ok.status is OutcomeStatus.OK
    assert (ctx.sandbox_dir / "projects" / "p1" / "seed.sql").read_bytes() == b"clobbered"


@pytest.mark.asyncio
async def test_F7_a_probe_with_no_conversation_uses_the_lab_it_made(tmp_path, fake, scope):
    ctx = _lab_ctx(tmp_path, pid=None, conv="")
    scope(ctx, "probe")
    fake.on("POST", "/v1/labs", (201, _detail(owner="", origin="probe")))
    fake.on("GET", "/v1/health", (200, {"host_id": "h"}))
    assert (await L.tool_lab(ctx, action="create", template="pg-single")).status is OutcomeStatus.OK
    fake.on("POST", f"/v1/labs/{LAB}/exec", (200, {"exit_code": 0, "stdout": "ok", "stderr": ""}))
    out = await L.tool_lab(ctx, action="exec", lab_id=LAB, member="db", cmd="true")
    assert "touches only labs it made" not in str(out)
    # …and never an owner's
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="conversation:owner-chat", origin="owner")))
    out2 = await L.tool_lab(ctx, action="exec", lab_id=LAB, member="db", cmd="true")
    assert out2.status is OutcomeStatus.REJECTED


@pytest.mark.asyncio
async def test_a_failed_builds_leftovers_keep_its_row_live(tmp_path, fake, scope, monkeypatch):
    import asyncio
    ctx = _lab_ctx(tmp_path)
    scope(ctx)

    async def _ns(*a, **k):
        return None
    monkeypatch.setattr(asyncio, "sleep", _ns)
    fake.on("POST", "/v1/labs", (201, _detail(state="creating")))
    fake.on("GET", "/v1/health", (200, {"host_id": "h1"}))
    fake.on("GET", f"/v1/labs/{LAB}", (404, {"error": "unknown_lab"}))
    fake.on("DELETE", f"/v1/labs/{LAB}", (200, {"destroyed": True, "leftovers": ["volume v1"]}))
    r = await L.tool_lab(ctx, action="create", template="pg-single")
    assert r.reason_code == "lab_create_failed" and "volume v1" in str(r)
    row = inventory_for(ctx).get(LAB)
    assert row["destroyed_at"] is None and "volume v1" in row["leftovers"]


@pytest.mark.asyncio
async def test_a_route_404_while_starting_is_not_a_failed_build(tmp_path, fake, scope, monkeypatch):
    import asyncio
    ctx = _lab_ctx(tmp_path)
    scope(ctx)

    async def _ns(*a, **k):
        return None
    monkeypatch.setattr(asyncio, "sleep", _ns)
    fake.on("POST", "/v1/labs", (201, _detail(state="creating")))
    fake.on("GET", "/v1/health", (200, {"host_id": "h1"}))
    fake.on("GET", f"/v1/labs/{LAB}", (404, {"detail": "Not Found"}), (200, _detail(state="ready")))
    r = await L.tool_lab(ctx, action="create", template="pg-single")
    assert r.reason_code != "lab_create_failed" and inventory_for(ctx).get(LAB)["destroyed_at"] is None


def test_other_host_text_is_honest_when_the_configured_host_is_unknown():
    L._LAST_HOST_ID.clear()
    assert "cannot reach" not in L.other_host_phrase("hostA") and "may or may not" in L.other_host_phrase("hostA")
    L._LAST_HOST_ID["id"] = "hostB"
    assert "cannot reach" in L.other_host_phrase("hostA")


# ================================================================== RUN

@pytest.mark.asyncio
async def test_f1_a_run_job_is_stopped_and_collected_by_its_job_id(tmp_path, scope):
    sb = JobSandbox(tmp_path / "ws")
    c = _run_ctx(tmp_path, sandbox=sb)
    scope(c)
    try:
        r = await R.tool_run(c, action="start", name="bench", command="sleep 30", where="workspace", lifetime="job")
        assert r.status is OutcomeStatus.OK
        from ghost_agent.sandbox.jobs import get_job_supervisor
        (e,) = get_job_supervisor(sb).list_entries()
        jid = e["id"]
        st = await R.tool_run(c, action="status", name=jid)
        assert st.status is OutcomeStatus.OK and "bench" in str(st)
        stop = await R.tool_run(c, action="stop", name=jid)
        assert "not running (nothing to cancel)" not in str(stop)
        (e2,) = get_job_supervisor(sb).list_entries()
        assert e2["state"] != "running"
    finally:
        _kill_jobs(sb)


@pytest.mark.asyncio
async def test_f2_a_by_id_collect_grades_the_jobs_own_record(tmp_path, scope, monkeypatch):
    """r12 F2 graded the code the jobs tool printed; r14 collapsed that to the registry row (the landing already
    normalised `| head` SIGPIPE to done; a printed "EXIT CODE: 0" line from the command itself counts for nothing)."""
    from test_4sx_c_run_tool import seed_job
    c = _run_ctx(tmp_path, conv="")
    scope(c)

    async def fake_jobs(action=None, job_id=None, context=None, **kw):
        return ToolOutcome.ok(f"--- {job_id}: gen ---\n[sandbox job {job_id} finished — EXIT CODE: 0]\na")
    monkeypatch.setattr(D, "tool_jobs", fake_jobs)
    seed_job(c, "job-0000aaaa", status="done")
    seed_job(c, "job-0000bbbb", status="failed")
    seed_job(c, "job-0000cccc")
    assert (await R.tool_run(c, action="collect", name="job-0000aaaa")).status is OutcomeStatus.OK
    assert (await R.tool_run(c, action="collect", name="job-0000bbbb")).status is OutcomeStatus.FAILED
    assert (await R.tool_run(c, action="collect", name="job-0000cccc")).status is OutcomeStatus.UNRESOLVED


class _EgressDownSandbox:
    tor_proxy = "socks5://127.0.0.1:9050"
    _egress_state = "unavailable"

    def __init__(self, attempted=True):
        self._att = attempted
        self.ensured = 0

    def egress_enforcement_attempted(self):
        return self._att

    def egress_is_enforced_or_blocked(self):
        return False

    def ensure_running(self):
        self.ensured += 1


@pytest.mark.asyncio
@pytest.mark.parametrize("attempted", [True, False])
async def test_f3_a_run_start_is_refused_without_tor_only_egress(tmp_path, scope, monkeypatch, attempted):
    sm = _EgressDownSandbox(attempted)
    started = []

    class _J:
        def start_detached(self, cmd, **kw):
            started.append(cmd)
            return {"id": "job-0000abcd", "deadline_at": 2700, "promoted_at": 0}, ""
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (None, _J()))
    c = _run_ctx(tmp_path, sandbox=sm)
    scope(c)
    r = await R.tool_run(c, action="start", name="fetch", command="curl -s https://example.com/ -o /tmp/x",
                         where="workspace", lifetime="job")
    assert r.status is OutcomeStatus.FAILED and r.reason_code == "egress_unavailable" and started == []
    assert sm.ensured == (0 if attempted else 1)


@pytest.mark.asyncio
async def test_f4_another_projects_job_is_not_reached_by_its_id(tmp_path, scope, monkeypatch):
    c = _run_ctx(tmp_path, pid="aaaaaaaaaaaa")
    scope(c)
    jid = "job-0a0a0a0a"
    assert R.record_promoted_job(c, {"id": jid, "command": "pytest -q"}) == jid
    cancelled = []

    class _Sup:
        def reap(self):
            return []

        def list_entries(self):
            return []

        def get(self, x):
            return {"id": x, "state": "running"}

        def cancel(self, x):
            cancelled.append(x)
            return True, "killed"
    monkeypatch.setattr(D, "_sandbox_supervisor", lambda ctx: _Sup())
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (None, _Sup()))
    c.current_project_id = "bbbbbbbbbbbb"
    scope(c)
    for act in ("stop", "collect", "status"):
        s = await R.tool_run(c, action=act, name=jid)
        assert s.status is OutcomeStatus.REJECTED and s.reason_code == "run_not_allowed", act
    assert cancelled == []


@pytest.mark.asyncio
async def test_f5_an_autopilot_restart_on_a_kept_lab_is_refused_before_the_stop(tmp_path, scope, fake):
    c = _run_ctx(tmp_path, pid="aaaaaaaaaaaa")
    owner = "project:aaaaaaaaaaaa"
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner=owner, lifetime="kept")))
    fake.on("POST", f"/v1/labs/{LAB}/exec", (200, {"job_id": "j1"}))
    scope(c, "owner")
    assert (await R.tool_run(c, action="start", name="api", command="node s.js", where=f"{LAB}/db",
                             lifetime="server", port=8080)).status is OutcomeStatus.OK
    fake.on("GET", f"/v1/labs/{LAB}/jobs/j1", (200, {"state": "running"}))
    fake.on("DELETE", f"/v1/labs/{LAB}/jobs/j1", (200, {"state": "exited", "killed": True}))
    scope(c, "autopilot")
    out = await R.tool_run(c, action="restart", name="api")
    assert out.status is OutcomeStatus.REJECTED and "KEPT" in str(out) and "WAS stopped" not in str(out)
    assert not any(x.method == "DELETE" for x in fake.calls)


@pytest.mark.asyncio
async def test_f6_a_bare_collect_in_an_owned_turn_takes_no_ones_results(tmp_path, scope, monkeypatch):
    from ghost_agent.core.jobs import STATUS_DONE, get_job_registry
    c = _run_ctx(tmp_path, pid="aaaaaaaaaaaa")
    hreg = get_job_registry(c)
    j = hreg.register(D.SANDBOX_KIND, "B's build", sandbox_job_id="job-0b0b0b0b")
    hreg.finish(j.id, status=STATUS_DONE, result="[sandbox job job-0b0b0b0b finished — EXIT CODE: 0]\nB-SECRET")
    woken = []
    monkeypatch.setattr(D, "_sandbox_supervisor",
                        lambda ctx: SimpleNamespace(mark_woken=woken.append, reap=lambda: [], list_entries=lambda: []))
    scope(c)
    r = await R.tool_run(c, action="collect")
    assert "B-SECRET" not in str(r) and woken == [] and r.reason_code == "run_name_required"


@pytest.mark.asyncio
async def test_a_server_restart_runs_under_the_starts_guards(tmp_path, scope, monkeypatch):
    from ghost_agent.memory.run_registry import registry_for
    c = _run_ctx(tmp_path, pid="aaaaaaaaaaaa")
    scope(c)
    restarted = []

    class _Svc:
        def restart(self, key):
            restarted.append(key)
            return "restarted"

        def list_entries(self):
            return []
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (_Svc(), None))
    reg = registry_for(c)
    reg.add(name="web", place="workspace", lifetime="server", command="rm -rf / --no-preserve-root", port=8080,
            owner="project:aaaaaaaaaaaa", origin="owner", backend="services", backend_ref="k", backend_token="t")
    out = await R.tool_run(c, action="restart", name="web")
    assert out.status is OutcomeStatus.REJECTED and restarted == []


# ================================================================== WIRING

def _dep(a, b):
    from ghost_agent.core.agent import _batch_call_depends, _batch_as_run
    return _batch_call_depends(*_batch_as_run(*a), *_batch_as_run(*b))


def test_m3_backup_is_ordered_with_its_lineage():
    assert _dep(("lab", {"action": "restore", "lab_id": "lab_bbbbbbbb", "params": {"backup_id": "x"}}),
                ("lab", {"action": "backup", "lab_id": "lab_aaaaaaaa"}))


def test_m2_two_destroys_run_in_written_order():
    assert _dep(("lab", {"action": "destroy", "lab_id": "lab_bbbbbbbb"}),
                ("lab", {"action": "destroy", "lab_id": "lab_aaaaaaaa"}))


def test_m4_the_alias_spelling_of_a_run_name_orders_the_batch():
    from ghost_agent.tools.runs import _alias_name
    assert _dep(("run", {"action": "logs", "name": _alias_name("My_API")}),
                ("manage_services", {"action": "start", "name": "My_API", "command": "x"}))


def test_M3_a_workspace_run_waits_for_a_lab_get():
    get = ("lab", {"action": "get", "lab_id": "lab_aaaaaaaa", "params": {"path": "/x", "to": "app/seed.sql"}})
    assert _dep(("run", {"action": "start", "name": "seed", "where": "workspace", "lifetime": "job",
                         "command": "psql -f app/seed.sql"}), get)
    assert not _dep(("run", {"action": "logs", "name": "seed"}), get)


def test_M1_a_retried_project_task_keeps_its_project():
    """The fire's owner lookup reads the TASK id (`X`), never the retry's (`X__retry`): the r11 code re-derived
    `_base_id` with split("__catchup") before the lookup."""
    import ast
    import ghost_agent.main as M
    tree = ast.parse(Path(M.__file__).read_text())
    assigned = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_base_id" for t in n.targets)]
    assert len(assigned) == 1


def test_M2_an_archived_or_unreadable_projects_turn_does_not_run():
    import asyncio
    from ghost_agent import main as M
    seen = []

    class _Agent:
        async def handle_chat(self, body, bt, request_id=None, **kw):
            seen.append(kw)
            return "ok", None, None

    class _Arch:
        def get_project(self, pid):
            return {"id": pid, "status": "ARCHIVED"}

    class _Broken:
        def get_project(self, pid):
            raise RuntimeError("database is locked")

    class _Live:
        def get_project(self, pid):
            return {"id": pid, "status": "ACTIVE"}
    import ghost_agent.api.routes as Rt
    keep = Rt._mark_foreground
    Rt._mark_foreground = lambda a, d: None
    try:
        for store in (_Arch(), _Broken()):
            out = asyncio.run(M._handle_chat_foreground(SimpleNamespace(project_store=store, agent=_Agent()),
                                                        {"messages": []}, "sched-x", project_id="p1"))
            assert str(out[0]).startswith(M.PROJECT_TURN_SKIPPED)
        assert seen == []
        asyncio.run(M._handle_chat_foreground(SimpleNamespace(project_store=_Live(), agent=_Agent()),
                                              {"messages": []}, "sched-x", project_id="p1"))
        assert seen and seen[0]["scope"].project_id == "p1"
        seen.clear()
        asyncio.run(M._handle_chat_foreground(SimpleNamespace(project_store=_Live(), agent=_Agent()),
                                              {"messages": []}, "sched-x"))
        assert seen == [{}]                        # no project: unscoped, as always
    finally:
        Rt._mark_foreground = keep


@pytest.mark.asyncio
async def test_m1_a_deleted_projects_folder_is_not_recreated_by_a_write(tmp_path):
    from ghost_agent.tools.file_system import project_scoped_sandbox, tool_file_system

    class _Store:
        def get_project(self, pid):
            return None
    ctx = SimpleNamespace(sandbox_dir=tmp_path, current_project_id="abc123def456", project_store=_Store())
    sub, _ = project_scoped_sandbox(ctx)
    out = await tool_file_system(operation="write", sandbox_dir=sub, path="notes.txt", content="hello",
                                 project_store=_Store())
    assert getattr(out, "reason_code", "") == "project_deleted"
    assert not (tmp_path / "projects" / "abc123def456").exists()


def test_m6_the_owner_hold_note_says_hold():
    import ast
    from ghost_agent.core import project_advancer as PA
    consts = [n.value for n in ast.walk(ast.parse(Path(PA.__file__).read_text()))
              if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert any("back to the owner's hold" in c for c in consts)
