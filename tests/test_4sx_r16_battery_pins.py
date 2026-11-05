"""§4SX battery r16 — pins for the mutants of scripts/mutate_4sx_r11.py that survived the r16 run.

Each test drives the real code path the mutant breaks and fails on the mutant (L8 L16 R3 R12 P2 P3 X4 X11 Y9 Z7 V3
N7 N10). Z7's code lives in a closure inside `main.lifespan`; that one function is lifted out of the parsed module
and RUN with fakes (behaviour, not source text)."""
import ast
import base64
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_4sx_b_lab_tool import FakeLabs, LAB, _detail  # noqa: E402
from test_4sx_b_lab_tool import _ctx as _lab_ctx  # noqa: E402
from test_4sx_c_run_tool import _ctx as _run_ctx  # noqa: E402
from test_4sx_r15_review import _setup as _r15_setup  # noqa: E402
from ghost_agent.core import request_scope as RS  # noqa: E402
from ghost_agent.tools import labs as L  # noqa: E402
from ghost_agent.tools import runs as R  # noqa: E402
from ghost_agent.tools.outcome import OutcomeStatus  # noqa: E402


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
    monkeypatch.setattr(L, "_LAST_HOST_ID", {})          # module memo: every test starts with no known host
    return f


# ------------------------------------------------------------------ L8: only the OWNER's destroy closes the record

@pytest.mark.asyncio
async def test_L8_an_unattended_destroy_of_another_hosts_lab_leaves_the_record_open(tmp_path, fake, scope):
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _lab_ctx(tmp_path, pid="p1")
    scope(c, origin="autopilot")
    inv = inventory_for(c)
    inv.record_create(LAB, owner="project:p1", origin="owner", template="pg-single", host_id="old-host")
    fake.on("GET", f"/v1/labs/{LAB}", (404, {"error": "unknown_lab", "message": "no such lab"}))
    fake.on("GET", "/v1/health", (200, {"host_id": "new-host"}))
    r = await L.tool_lab(c, action="destroy", lab_id=LAB)
    assert r.status is not OutcomeStatus.OK and "CLOSED" not in str(r)
    assert inv.get(LAB)["destroyed_at"] is None                  # still live for the owner / project teardown
    assert not any(x.method == "DELETE" for x in fake.calls)


# ------------------------------------------------------------------ L16: the host id asked once is remembered

@pytest.mark.asyncio
async def test_L16_the_host_id_learned_by_a_destroy_names_the_configured_host_later(tmp_path, fake, scope):
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _lab_ctx(tmp_path)
    scope(c)
    inv = inventory_for(c)
    inv.record_create(LAB, owner="conversation:conv1", origin="owner", template="pg-single", host_id="old-host")
    inv.record_create("lab_cccccccc", owner="conversation:conv1", origin="owner", template="pg-single",
                      host_id="old-host")
    fake.on("GET", f"/v1/labs/{LAB}", (404, {"error": "unknown_lab", "message": "no such lab"}))
    fake.on("GET", "/v1/health", (200, {"host_id": "new-host"}))
    await L.tool_lab(c, action="destroy", lab_id=LAB)            # asks /v1/health once
    out = L._service_reply(404, {"error": "unknown_lab", "message": "no"}, c, "lab_cccccccc")
    assert "the configured host is new-host" in str(out) and "expired" not in str(out)
    n = sum(1 for x in fake.calls if x.path == "/v1/health")
    assert await L.lab_on_other_host(c, "lab_cccccccc") == "old-host"
    assert sum(1 for x in fake.calls if x.path == "/v1/health") == n   # answered from the memo, not asked again


# ------------------------------------------------------------------ X4: an owner-less probe reaches its OWN lab

@pytest.mark.asyncio
async def test_X4_an_ownerless_probe_reaches_the_ownerless_lab_a_probe_made(tmp_path, fake, scope):
    c = _lab_ctx(tmp_path, conv="")
    scope(c, origin="probe")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="", origin="probe")))
    r = await L.tool_lab(c, action="status", lab_id=LAB)
    assert r.status is OutcomeStatus.OK, str(r)
    # control: an owner's lab stays walled for the same probe
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="conversation:conv1", origin="owner")))
    r = await L.tool_lab(c, action="status", lab_id=LAB)
    assert r.status is OutcomeStatus.REJECTED


# ------------------------------------------------------------------ V3: the destroy's own check reads the ORIGIN

@pytest.mark.asyncio
async def test_V3_a_probe_never_destroys_a_lab_the_host_says_a_probe_did_not_make(tmp_path, fake, scope):
    """The wall's ownership check got a 404 (nothing to hold it to); the destroy's own status read then shows the
    lab under the probe's owner key but made by the OWNER — it is refused, never deleted."""
    c = _lab_ctx(tmp_path, conv="probe-c")
    scope(c, origin="probe")
    fake.on("GET", f"/v1/labs/{LAB}", (404, {"error": "unknown_lab"}),
            (200, _detail(owner="conversation:probe-c", origin="owner")))
    fake.on("DELETE", f"/v1/labs/{LAB}", (200, {"destroyed": True}))
    r = await L.tool_lab(c, action="destroy", lab_id=LAB)
    assert r.status is OutcomeStatus.REJECTED and "a probe destroys only its own labs" in str(r)
    assert not any(x.method == "DELETE" for x in fake.calls)


# ------------------------------------------------------------------ N7: `get` shows where the file landed

@pytest.mark.asyncio
async def test_N7_get_shows_the_workspace_path_it_wrote_not_the_given_spelling(tmp_path, fake, scope):
    c = _lab_ctx(tmp_path)
    scope(c)
    fake.on("GET", f"/v1/labs/{LAB}/files", (200, {"content_b64": base64.b64encode(b"dump").decode()}))
    r = await L.tool_lab(c, action="get", lab_id=LAB, member="db", path="/x.sql", sandbox_path="/etc/dump.sql")
    assert r.status is OutcomeStatus.OK, str(r)
    assert (c.sandbox_dir / "etc" / "dump.sql").read_bytes() == b"dump"
    assert "→ /workspace/etc/dump.sql" in str(r) and "→ /etc/dump.sql" not in str(r)


# ------------------------------------------------------------------ R3 / Y9 / R12: a workspace server restart

class _Svc:
    """A services supervisor whose one entry ``k`` (token ``t``) is alive."""

    def __init__(self, on_restart=None):
        self.restarted, self.on_restart = [], on_restart

    def restart(self, key):
        self.restarted.append(key)
        if self.on_restart:
            self.on_restart()
        return "cannot restart — survived TERM+KILL"

    def list_entries(self):
        return [{"key": "k", "token": "t", "port": 8080}]

    def _entry_state(self, e):
        return True


def _server_row(c, command):
    from ghost_agent.memory.run_registry import registry_for
    reg = registry_for(c)
    rid = reg.add(name="web", place="workspace", lifetime="server", command=command, port=8080,
                  owner="project:aaaaaaaaaaaa", origin="owner", backend="services", backend_ref="k",
                  backend_token="t")
    return reg, rid


@pytest.mark.asyncio
async def test_Y9_a_restart_of_a_live_server_runs_under_the_shell_guards(tmp_path, scope, monkeypatch):
    c = _run_ctx(tmp_path, pid="aaaaaaaaaaaa")
    scope(c)
    svc = _Svc()
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (svc, None))
    _server_row(c, "rm -rf / --no-preserve-root")
    out = await R.tool_run(c, action="restart", name="web")
    assert out.status is OutcomeStatus.REJECTED and out.reason_code == "run_shell_blocked", str(out)
    assert svc.restarted == []


@pytest.mark.asyncio
async def test_R3_a_crashing_shell_validator_refuses_the_restart(tmp_path, scope, monkeypatch):
    import ghost_agent.tools.validators as V
    c = _run_ctx(tmp_path, pid="aaaaaaaaaaaa")
    scope(c)
    svc = _Svc()
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (svc, None))

    def boom(cmd):
        raise RuntimeError("validator bug")
    monkeypatch.setattr(V, "validate_shell", boom)
    _server_row(c, "python3 app.py")
    out = await R.tool_run(c, action="restart", name="web")
    assert out.status is OutcomeStatus.REJECTED and "shell validator failed" in str(out), str(out)
    assert svc.restarted == []


@pytest.mark.asyncio
async def test_R12_a_restart_that_could_not_stop_the_server_revives_a_row_ended_meanwhile(tmp_path, scope,
                                                                                          monkeypatch):
    """While the restart ran, a concurrent probe saw the process down and ENDED the row; the process survived the
    kill — the row must read running again (ended_at cleared), not stay ended while the server runs."""
    c = _run_ctx(tmp_path, pid="aaaaaaaaaaaa")
    scope(c)
    holder = {}
    svc = _Svc(on_restart=lambda: holder["reg"].set_state(holder["rid"], "exited", "probe saw it down"))
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (svc, None))
    holder["reg"], holder["rid"] = _server_row(c, "python3 app.py")
    out = await R.tool_run(c, action="restart", name="web")
    assert out.reason_code == "run_restart_failed", str(out)
    row = [r for r in holder["reg"].rows("project:aaaaaaaaaaaa") if r["run_id"] == holder["rid"]][0]
    assert row["state"] == "running" and row["ended_at"] is None


# ------------------------------------------------------------------ P2 / P3: the boot reaper's orphan set

def test_P2_P3_only_autopilot_labs_and_non_app_runs_are_leaf_orphans(tmp_path):
    from ghost_agent.core import leaf_reaper as LR
    from ghost_agent.memory.lab_inventory import inventory_for
    from ghost_agent.memory.run_registry import registry_for
    c = _lab_ctx(tmp_path)
    inv = inventory_for(c)
    # P2: an OWNER's lab that carries a leaf tag is never the reaper's
    inv.record_create("lab_aaaaaaaa", owner="project:p1", origin="owner", template="pg-single", leaf="leaf-owner")
    inv.record_create("lab_bbbbbbbb", owner="project:p1", origin="autopilot", template="pg-single", leaf="leaf-a")
    reg = registry_for(c)
    # P3: an app run (production, kept) tagged with a leaf is never the reaper's
    reg.add(name="shop", place=f"{LAB}/web", lifetime="app", command="x", port=None, owner="project:p1",
            origin="autopilot", backend="labs_app", backend_ref=f"{LAB}/web", meta={"leaf": "leaf-app"})
    reg.add(name="srv", place="workspace", lifetime="server", command="x", port=None, owner="project:p1",
            origin="autopilot", backend="services", backend_ref="k", meta={"leaf": "leaf-srv"})
    assert LR.orphan_leaves(c) == {"leaf-a": {"project:p1"}, "leaf-srv": {"project:p1"}}


# ------------------------------------------------------------------ X11: a FIFO target is never written

def test_X11_a_fifo_with_a_reader_is_refused_not_written(tmp_path):
    from ghost_agent.tools.file_system import write_bytes_nofollow_rel
    fifo = tmp_path / "out.bin"
    os.mkfifo(str(fifo))
    rfd = os.open(str(fifo), os.O_RDONLY | os.O_NONBLOCK)      # a reader: the O_NONBLOCK open no longer fails
    try:
        with pytest.raises(OSError):
            write_bytes_nofollow_rel(tmp_path, "out.bin", b"PAYLOAD")
        try:
            got = os.read(rfd, 64)
        except BlockingIOError:
            got = b""
        assert got == b""
    finally:
        os.close(rfd)


# ------------------------------------------------------------------ N10: the jobs alias acts on the run row

@pytest.mark.asyncio
async def test_N10_jobs_alias_by_registry_id_acts_on_the_owners_run_row(tmp_path, monkeypatch, scope):
    c, sid, rid, sup = _r15_setup(tmp_path, monkeypatch, scope)
    c.current_project_id = "aaaaaaaaaaaa"                            # back in project A, the job's owner
    scope(c)
    out = await R.jobs_alias(context=c, action="cancel", job_id=rid)
    assert sup.cancelled == [sid]
    assert "this ran as `run stop name=" in str(out), str(out)     # the run row's own route, not the by-id fallback


# ------------------------------------------------------------------ Z7: a retried project task runs in its project

def _lifted_run_proactive_task(ns):
    """`main.lifespan`'s `_run_proactive_task`, compiled on its own: its closure names become ``ns`` globals."""
    import ghost_agent.main as M
    tree = ast.parse(Path(M.__file__).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run_proactive_task")
    mod = ast.Module(body=[fn], type_ignores=[])
    g = dict(vars(M))
    g.update(ns)
    exec(compile(mod, M.__file__, "exec"), g)
    return g["_run_proactive_task"]


@pytest.mark.asyncio
async def test_Z7_a_retried_project_task_runs_in_its_project():
    import ghost_agent.main as M
    seen = []

    async def hcf(context, body, request_id, project_id=None):
        seen.append(project_id)
        return M.PROJECT_TURN_SKIPPED + "]", None, None         # stop right after the dispatch

    tasks = SimpleNamespace(should_defer_scheduled_task=lambda llm: False,
                            _load_task_store=lambda: {"t1": {"owner": "project:p1"}},
                            note_task_fired=lambda *a, **k: None)
    fn = _lifted_run_proactive_task({"_sched": None, "_tools_tasks": tasks, "context": SimpleNamespace(),
                                     "args": SimpleNamespace(model="m"), "_handle_chat_foreground": hcf})
    assert await fn("t1", "do it") is False
    assert await fn("t1__retry", "do it") is False
    assert seen == ["p1", "p1"]
