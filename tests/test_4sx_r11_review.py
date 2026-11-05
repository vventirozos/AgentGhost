"""§4SX end-to-end review r11 (2026-11-05) — every agent-side finding, pinned.

Each test fails on the code before the fix (the reviewers' repros, inverted)."""
import asyncio
import base64
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_4sx_b_lab_tool import FakeLabs, LAB, _detail  # noqa: E402
from test_4sx_b_lab_tool import _ctx as _lab_ctx  # noqa: E402
from ghost_agent.core import request_scope as RS  # noqa: E402
from ghost_agent.tools import labs as L  # noqa: E402
from ghost_agent.tools import runs as R  # noqa: E402
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome  # noqa: E402

pytestmark = pytest.mark.asyncio


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


# ------------------------------------------------------------------ L-C1: put/get never follow a link

async def test_get_never_writes_through_a_link_planted_during_the_host_call(tmp_path, fake, scope, monkeypatch):
    c = _lab_ctx(tmp_path)
    scope(c)
    outside = tmp_path / "OUTSIDE.txt"
    outside.write_text("original")
    target = c.sandbox_dir / "dump.sql"
    real = fake.__call__

    async def racing(context, method, path, **kw):
        if method == "GET" and path.endswith("/files"):
            os.symlink(str(outside), str(target))      # sandboxed code wins the race
            return 200, {"content_b64": base64.b64encode(b"PWNED").decode()}
        return await real(context, method, path, **kw)
    monkeypatch.setattr(L, "_call", racing)
    r = await L.tool_lab(c, action="get", lab_id=LAB, member="db", path="/x", sandbox_path="dump.sql")
    assert outside.read_text() == "original"
    assert not target.is_symlink() and target.read_bytes() == b"PWNED"      # the link was replaced, not followed
    assert r.status is OutcomeStatus.OK


async def test_get_never_writes_through_a_planted_parent_folder_link(tmp_path, fake, scope, monkeypatch):
    c = _lab_ctx(tmp_path)
    scope(c)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    real = fake.__call__

    async def racing(context, method, path, **kw):
        if method == "GET" and path.endswith("/files"):
            os.symlink(str(elsewhere), str(c.sandbox_dir / "out"))
            return 200, {"content_b64": base64.b64encode(b"PWNED").decode()}
        return await real(context, method, path, **kw)
    monkeypatch.setattr(L, "_call", racing)
    await L.tool_lab(c, action="get", lab_id=LAB, member="db", path="/x", sandbox_path="out/x.sql")
    assert not (elsewhere / "x.sql").exists()


async def test_put_refuses_a_file_swapped_for_a_link_after_the_check(tmp_path, fake, scope, monkeypatch):
    c = _lab_ctx(tmp_path)
    scope(c)
    secret = tmp_path / "KEY"
    secret.write_text("the labs key")
    f = c.sandbox_dir / "seed.sql"
    f.write_text("select 1;")
    real_path = L._sandbox_path

    def swapping(context, rel):
        p, err = real_path(context, rel)
        f.unlink()
        os.symlink(str(secret), str(f))                # swapped right after the check
        return p, err
    monkeypatch.setattr(L, "_sandbox_path", swapping)
    r = await L.tool_lab(c, action="put", lab_id=LAB, member="db", path="/tmp/seed.sql", sandbox_path="seed.sql")
    assert r.status is OutcomeStatus.REJECTED
    assert not any(x.method == "PUT" for x in fake.calls)


# ------------------------------------------------------------------ L-M2 / L-M3: rows the host does not list

async def test_list_never_retires_another_hosts_lab_when_health_fails(tmp_path, fake, scope):
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _lab_ctx(tmp_path)
    scope(c)
    inv = inventory_for(c)
    inv.record_create("lab_cccccccc", owner="conversation:conv1", origin="owner", template="pg-single",
                      host_id="old-host")
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    fake.on("GET", "/v1/health", (503, {"error": "x"}))
    r = await L.tool_lab(c, action="list")
    assert "expired" not in str(r) and "not answering /v1/health" in str(r)
    assert inv.get("lab_cccccccc")["destroyed_at"] is None


async def test_list_keeps_another_hosts_lab_for_project_teardown(tmp_path, fake, scope):
    from ghost_agent.memory.lab_inventory import inventory_for
    from ghost_agent.tools import projects as P
    c = _lab_ctx(tmp_path, pid="p1")
    scope(c)
    inv = inventory_for(c)
    inv.record_create("lab_cccccccc", owner="project:p1", origin="owner", template="pg-single", host_id="old-host")
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    fake.on("GET", "/v1/health", (200, {"host_id": "new-host"}))
    await L.tool_lab(c, action="list")
    rep = {"stopped": [], "kept_up": [], "failed": [], "live_left": [], "notes": []}
    await P._teardown_labs(c, "p1", "project:p1", hard=True, origin="owner", rep=rep)
    assert rep["live_left"], rep


async def test_list_asks_the_host_about_leftovers_before_retiring(tmp_path, fake, scope):
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _lab_ctx(tmp_path)
    scope(c)
    inv = inventory_for(c)
    inv.record_create("lab_dddddddd", owner="conversation:conv1", origin="owner", template="pg-single")
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    fake.on("GET", "/v1/labs/lab_dddddddd", (404, {"error": "unknown_lab"}))
    fake.on("DELETE", "/v1/labs/lab_dddddddd", (200, {"destroyed": True, "leftovers": ["volume x"]}))
    r = await L.tool_lab(c, action="list")
    assert "still holds volume x" in str(r)
    row = inv.get("lab_dddddddd")
    assert row["destroyed_at"] is None and row["leftovers"]


async def test_list_does_not_retire_a_lab_the_host_still_has(tmp_path, fake, scope):
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _lab_ctx(tmp_path)
    scope(c)
    inv = inventory_for(c)
    inv.record_create("lab_eeeeeeee", owner="conversation:conv1", origin="owner", template="pg-single")
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    fake.on("GET", "/v1/labs/lab_eeeeeeee", (200, _detail(lab_id="lab_eeeeeeee")))
    r = await L.tool_lab(c, action="list")
    assert inv.get("lab_eeeeeeee")["destroyed_at"] is None
    assert "the labs host still has it" in str(r)          # said as what it is, not as an HTTP answer
    assert not any(x.method == "DELETE" for x in fake.calls)


async def test_destroy_of_another_hosts_lab_closes_the_record_only_for_the_owner(tmp_path, fake, scope):
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _lab_ctx(tmp_path)
    scope(c)
    inv = inventory_for(c)
    inv.record_create(LAB, owner="conversation:conv1", origin="owner", template="pg-single", host_id="old-host")
    fake.on("GET", f"/v1/labs/{LAB}", (404, {"error": "unknown_lab", "message": "no such lab"}))
    fake.on("GET", "/v1/health", (200, {"host_id": "new-host"}))
    r = await L.tool_lab(c, action="destroy", lab_id=LAB)
    assert "gone from the labs host" not in str(r)
    assert "made on labs host old-host" in str(r) and "nothing was deleted" in str(r).lower()
    assert inv.get(LAB)["destroyed_at"] is not None
    assert not any(x.method == "DELETE" for x in fake.calls)


async def test_a_404_for_another_hosts_lab_never_says_it_expired(tmp_path, fake, scope):
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _lab_ctx(tmp_path, pid="p1")
    scope(c, origin="autopilot")
    inv = inventory_for(c)
    inv.record_create(LAB, owner="project:p1", origin="owner", template="pg-single", host_id="old-host")
    L._LAST_HOST_ID["id"] = "new-host"
    out = L._service_reply(404, {"error": "unknown_lab", "message": "no"}, c, LAB)
    assert "expire" not in str(out) and "may still be running there" in str(out)


async def test_a_run_on_another_hosts_lab_is_not_lost(tmp_path, fake, scope):
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _lab_ctx(tmp_path)
    inv = inventory_for(c)
    inv.record_create(LAB, owner="conversation:conv1", origin="owner", template="pg-single", host_id="old-host")
    fake.on("GET", f"/v1/labs/{LAB}/jobs/job_1", (404, {"error": "unknown_lab"}))
    fake.on("GET", "/v1/health", (200, {"host_id": "new-host"}))
    st, note, _ = await R._probe(c, {"backend": "labs_job", "backend_ref": f"{LAB}/job_1"}, "owner")
    assert st == "unknown" and "old-host" in note


async def test_destroy_whose_status_check_got_no_answer_says_nothing_was_deleted(tmp_path, fake, scope):
    c = _lab_ctx(tmp_path)
    scope(c)
    fake.on("GET", f"/v1/labs/{LAB}", L.LabsNoAnswer("ReadTimeout"))
    r = await L.tool_lab(c, action="destroy", lab_id=LAB)
    assert "NOTHING was deleted" in str(r) and "may or may not" not in str(r)
    assert not any(x.method == "DELETE" for x in fake.calls)


# ------------------------------------------------------------------ L-m5 / L-m6: walls

async def test_a_probe_cannot_touch_an_owner_lab(tmp_path, fake, scope):
    c = _lab_ctx(tmp_path, conv="probe-conv")
    scope(c, origin="probe")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="conversation:conv1")))
    for kw in (dict(action="member", member="db", op="kill"), dict(action="exec", member="db", cmd="id"),
               dict(action="extend", ttl_idle_s=600)):
        r = await L.tool_lab(c, lab_id=LAB, **kw)
        assert r.status is OutcomeStatus.REJECTED and "a probe touches only labs it made" in str(r), kw
    assert not any(x.method in ("POST", "PUT", "PATCH") for x in fake.calls)


async def test_autopilot_never_runs_or_reads_on_a_kept_lab(tmp_path, fake, scope):
    c = _lab_ctx(tmp_path, pid="p1")
    scope(c, origin="leaf")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="project:p1", lifetime="kept")))
    for kw in (dict(action="exec", member="db", cmd="psql -c 'drop table x'"),
               dict(action="get", member="db", path="/data/pg/x", sandbox_path="x"),
               dict(action="logs", member="db")):
        r = await L.tool_lab(c, lab_id=LAB, **kw)
        assert r.status is OutcomeStatus.REJECTED and "KEPT" in str(r), kw
    assert not any(x.path.endswith(("/exec", "/files", "/logs")) for x in fake.calls)
    r = await L.tool_lab(c, action="status", lab_id=LAB)       # a status read stays open
    assert r.status is OutcomeStatus.OK


async def test_an_ownership_check_error_is_not_someone_elses(tmp_path, fake, scope):
    c = _lab_ctx(tmp_path, pid="p1")
    scope(c, origin="leaf")
    fake.on("GET", f"/v1/labs/{LAB}", (500, {"error": "internal"}))
    r = await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="id")
    assert "someone else" not in str(r) and "HTTP 500" in str(r)


# ------------------------------------------------------------------ L-m9: truthful create texts

async def test_a_lab_gone_while_starting_is_a_failed_build(tmp_path, fake, scope, monkeypatch):
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _lab_ctx(tmp_path)
    scope(c)
    monkeypatch.setattr(asyncio, "sleep", _nosleep)
    fake.on("POST", "/v1/labs", (201, _detail(state="creating")))
    fake.on("GET", "/v1/health", (200, {"host_id": "h1"}))
    fake.on("GET", f"/v1/labs/{LAB}", (404, {"error": "unknown_lab"}))
    fake.on("DELETE", f"/v1/labs/{LAB}", (404, {"error": "unknown_lab"}))     # r13 R6: the host's answer closes it
    r = await L.tool_lab(c, action="create", template="pg-single")
    assert "create_failed" in str(r) and "TTL" not in str(r)
    assert inventory_for(c).get(LAB)["destroyed_at"] is not None


async def test_evicted_labs_are_recorded_destroyed(tmp_path, fake, scope):
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _lab_ctx(tmp_path)
    scope(c)
    inv = inventory_for(c)
    inv.record_create("lab_aaaaaaaa", owner="project:p9", origin="autopilot", template="pg-single")
    fake.on("POST", "/v1/labs", (201, _detail(evicted=[{"lab_id": "lab_aaaaaaaa", "owner": "project:p9"}])))
    fake.on("GET", "/v1/health", (200, {"host_id": "h1"}))
    await L.tool_lab(c, action="create", template="pg-single")
    row = inv.get("lab_aaaaaaaa")
    assert row["destroyed_at"] is not None and "evicted" in row["destroyed_by"]


async def _nosleep(*a, **k):
    return None


# ------------------------------------------------------------------ L-M4: batch order

def test_batch_orders_workspace_files_and_lineage_across_labs():
    from ghost_agent.core.agent import _batch_call_depends as dep
    get_a = {"action": "get", "lab_id": "lab_aaaaaaaa", "sandbox_path": "app/seed.sql"}
    deploy_b = {"action": "deploy", "lab_id": "lab_bbbbbbbb", "params": {"folder": "app"}}
    assert dep("lab", deploy_b, "lab", get_a) and dep("lab", get_a, "lab", deploy_b)
    keep_b = {"action": "keep", "lab_id": "lab_bbbbbbbb"}
    destroy_a = {"action": "destroy", "lab_id": "lab_aaaaaaaa"}
    assert dep("lab", destroy_a, "lab", keep_b)
    switch_b = {"action": "switch", "lab_id": "lab_bbbbbbbb", "params": '{"from_lab": "lab_aaaaaaaa"}'}
    exec_a = {"action": "exec", "lab_id": "lab_aaaaaaaa"}
    assert dep("lab", switch_b, "lab", exec_a) and dep("lab", exec_a, "lab", switch_b)
    # r12 m2: two destroys run in written order (a shared data volume); unrelated execs still run side by side
    assert dep("lab", destroy_a, "lab", {"action": "destroy", "lab_id": "lab_bbbbbbbb"})
    assert dep("lab", {"action": "backup", "lab_id": "lab_bbbbbbbb"}, "lab", {"action": "restore", "lab_id": "lab_aaaaaaaa"})
    assert not dep("lab", exec_a, "lab", {"action": "exec", "lab_id": "lab_bbbbbbbb"})
    # r12 M3: a workspace run after a lab get waits for it; a read does not
    assert dep("run", {"action": "start", "where": "workspace", "name": "seed"}, "lab", get_a)
    assert not dep("run", {"action": "status"}, "lab", get_a)


def test_batch_orders_the_old_names_like_run():
    from ghost_agent.core.agent import _batch_as_run, _batch_call_depends as dep
    n, a = _batch_as_run("manage_services", {"action": "start", "name": "api", "command": "python3 app.py"})
    assert n == "run" and dep("execute", {"command": "curl $URL"}, n, a)
    n2, a2 = _batch_as_run("jobs", {"action": "cancel", "job_id": "job-1a2b3c4d"})
    assert n2 == "run" and a2["action"] == "stop" and a2["name"] == "job-1a2b3c4d"


# ------------------------------------------------------------------ RUN

class _Jobs:
    def __init__(self, entry=None):
        self.started, self.entry = [], entry

    def start_detached(self, cmd, **kw):
        self.started.append(cmd)
        return {"id": "job-0000abcd", "deadline_at": 2700, "promoted_at": 0}, ""

    def cancel(self, jid):
        return True, "ok"

    def get(self, jid):
        return self.entry

    def reap(self):
        return None

    def list_entries(self):
        return [self.entry] if self.entry else []


async def test_a_workspace_job_runs_through_bash_with_pipefail(tmp_path, scope, monkeypatch):
    fj = _Jobs()
    monkeypatch.setattr(R, "_supervisors", lambda c: (None, fj))
    c = _lab_ctx(tmp_path)
    scope(c)
    await R.tool_run(c, action="start", name="t", command="cd app && FOO=1 pytest", where="workspace", lifetime="job")
    assert fj.started and fj.started[0].startswith("bash -c ") and "set -o pipefail; cd app && FOO=1 pytest" in fj.started[0]


@pytest.mark.parametrize("cmd,code", [("rm -rf / --no-preserve-root", "run_shell_blocked"),
                                      ("python3 server.py --port 8101 &", "run_daemon_blocked"),
                                      ("yt-dlp https://youtu.be/x", "run_youtube_blocked")])
async def test_a_workspace_run_gets_executes_shell_guards(tmp_path, scope, monkeypatch, cmd, code):
    fj = _Jobs()
    monkeypatch.setattr(R, "_supervisors", lambda c: (None, fj))
    c = _lab_ctx(tmp_path)
    scope(c)
    r = await R.tool_run(c, action="start", name="a", command=cmd, where="workspace", lifetime="job")
    assert r.status is OutcomeStatus.REJECTED and r.reason_code == code and fj.started == []


async def test_autopilot_runs_only_on_its_own_disposable_labs(tmp_path, scope, fake):
    c = _lab_ctx(tmp_path, pid="aaaaaaaaaaaa")
    scope(c, "autopilot")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="project:bbbbbbbbbbbb")))
    r = await R.tool_run(c, action="start", name="x", command="psql -c 'drop table t'", where=f"{LAB}/db",
                         lifetime="job")
    assert r.status is OutcomeStatus.REJECTED and "belongs to project:bbbbbbbbbbbb" in str(r)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="project:aaaaaaaaaaaa", lifetime="kept")))
    r2 = await R.tool_run(c, action="start", name="y", command="id", where=f"{LAB}/db", lifetime="job")
    assert r2.status is OutcomeStatus.REJECTED and "KEPT" in str(r2)
    assert not any(x.method == "POST" for x in fake.calls)


def test_a_live_state_never_lands_on_an_ended_row(tmp_path):
    from ghost_agent.memory.run_registry import RunRegistry
    reg = RunRegistry(tmp_path)
    rid = reg.add(name="w", place="workspace", lifetime="server", command="x", port=None, owner="o",
                  origin="owner", backend="services", backend_ref="k")
    reg.set_state(rid, "stopped", "stopped by run")
    reg.set_state(rid, "running", "")                  # a racing status probe that read before the stop
    assert reg.get(rid)["state"] == "stopped" and reg.live_by_name("o", "w") is None
    reg.set_state(rid, "running", "a restart could not stop it", revive=True)
    assert reg.get(rid)["ended_at"] is None and reg.live_by_name("o", "w") is not None


async def test_a_collected_failed_job_by_id_is_failed(tmp_path, scope, monkeypatch):
    c = _lab_ctx(tmp_path, conv="")
    scope(c)
    from test_4sx_c_run_tool import seed_job
    seed_job(c, "job-0000abcd", status="failed")                 # r14: the registry row decides

    async def fake_jobs(action=None, job_id=None, context=None, **kw):
        return ToolOutcome.ok(f"--- {job_id} [FAILED]: build ---\nTraceback ...")
    import ghost_agent.tools.delegate as D
    monkeypatch.setattr(D, "tool_jobs", fake_jobs)
    r = await R.tool_run(c, action="collect", name="job-0000abcd")
    assert r.status is OutcomeStatus.FAILED and str(r).startswith("RUN ERROR — ")


async def test_an_owned_turn_reaches_a_job_with_no_run_row_by_its_id(tmp_path, scope, monkeypatch):
    c = _lab_ctx(tmp_path)
    scope(c)
    seen = []

    async def fake_jobs(action=None, job_id=None, context=None, **kw):
        seen.append((action, job_id))
        return ToolOutcome.ok(f"Job {job_id} is still RUNNING (5s so far): x. Check back later.")
    import ghost_agent.tools.delegate as D
    monkeypatch.setattr(D, "tool_jobs", fake_jobs)
    st = await R.tool_run(c, action="status", name="job-1a2b3c4d")     # no such job anywhere
    assert st.reason_code == "run_unknown" and seen == []
    from test_4sx_c_run_tool import seed_job
    seed_job(c, "job-1a2b3c4d")
    st = await R.tool_run(c, action="status", name="job-1a2b3c4d")     # r14: the registry row's own state
    assert st.status is OutcomeStatus.OK and "running" in str(st) and seen == []
    col = await R.tool_run(c, action="collect", name="job-1a2b3c4d")
    assert col.reason_code == "run_still_running"
    r = await R.tool_run(c, action="collect")          # r12 F6: a bare collect in an owned turn names nothing
    assert seen == [("collect", "job-1a2b3c4d")] and r.reason_code == "run_name_required"


def _str_consts(path):
    import ast
    tree = ast.parse(Path(path).read_text())
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def test_no_result_names_the_jobs_tool():
    import re
    root = Path(__file__).parent.parent / "src" / "ghost_agent"
    for f in ("tools/delegate.py", "tools/execute.py", "main.py"):
        live = [c for c in _str_consts(root / f) if re.search(r"jobs\(action=", c)]
        assert live == [], (f, live)


def test_a_run_mutation_is_not_evidence():
    from ghost_agent.core.evidence_gate import assess_turn_evidence as audit_evidence
    row = {"role": "tool", "name": "run", "content": ToolOutcome("Still running: this is not a result",
                                                                   status=OutcomeStatus.OK,
                                                                   call_args={"action": "start", "name": "a"})}
    a = audit_evidence([row])
    assert a.consulted == 0 and a.substantive == 0


def test_an_unanswered_call_is_not_still_running():
    from ghost_agent.tools.tool_failure import summarize_multi_op_outcomes as multi_step_outcome_summary
    ops = [{"tool": "run", "ok": False, "unanswered": True}, {"tool": "file_system", "ok": True}]
    out = multi_step_outcome_summary(ops)
    assert "NO ANSWER / NOT VERIFIED" in out and "STILL RUNNING" not in out


def test_the_old_names_carry_runs_egress_policy():
    from ghost_agent.memory.egress import OUTBOUND_TOOLS
    assert OUTBOUND_TOOLS["manage_services"] == OUTBOUND_TOOLS["jobs"] == OUTBOUND_TOOLS["run"] == "url_literals"


async def test_manage_services_port_zero_means_no_port(monkeypatch):
    seen = {}

    async def fake_run(context=None, **kw):
        seen.update(kw)
        return ToolOutcome.ok("ok")
    monkeypatch.setattr(R, "tool_run", fake_run)
    await R.manage_services_alias(context=None, action="start", name="api", command="python3 app.py", port=0)
    assert seen.get("port") in (None, "")


# ------------------------------------------------------------------ leaf work after a restart

async def test_the_boot_reaper_ends_the_previous_processes_leaf_work(tmp_path, monkeypatch):
    from ghost_agent.core import leaf_reaper as LR
    from ghost_agent.memory.lab_inventory import inventory_for
    from ghost_agent.memory.run_registry import registry_for
    c = _lab_ctx(tmp_path)
    inv = inventory_for(c)
    inv.record_create("lab_aaaaaaaa", owner="project:p1", origin="autopilot", template="pg-single", leaf="leaf-1")
    inv.record_create("lab_bbbbbbbb", owner="project:p1", origin="owner", template="pg-single")      # never
    reg = registry_for(c)
    reg.add(name="srv", place="workspace", lifetime="server", command="x", port=None, owner="project:p1",
            origin="autopilot", backend="services", backend_ref="k", meta={"leaf": "leaf-2"})
    reg.add(name="own", place="workspace", lifetime="server", command="x", port=None, owner="project:p1",
            origin="owner", backend="services", backend_ref="k2")                                   # never
    leaves = LR.orphan_leaves(c)
    assert leaves == {"leaf-1": {"project:p1"}, "leaf-2": {"project:p1"}}
    calls = []

    async def stop(ctx, owner, leaf):
        calls.append(("stop", owner, leaf))
        return []

    async def destroy(ctx, owner, leaf):
        calls.append(("destroy", owner, leaf))
        return []
    monkeypatch.setattr(R, "stop_leaf_runs", stop)
    monkeypatch.setattr(L, "destroy_leaf_labs", destroy)
    await LR.reap_orphan_leaf_work(c, leaves)
    assert sorted(calls) == sorted([("stop", "project:p1", "leaf-1"), ("destroy", "project:p1", "leaf-1"),
                                    ("stop", "project:p1", "leaf-2"), ("destroy", "project:p1", "leaf-2")])
    # runs before labs, per leaf
    assert calls.index(("stop", "project:p1", "leaf-1")) < calls.index(("destroy", "project:p1", "leaf-1"))


def _call_lines(path, name):
    import ast
    tree = ast.parse(Path(path).read_text())
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if (isinstance(f, ast.Name) and f.id == name) or (isinstance(f, ast.Attribute) and f.attr == name):
                out.append(n.lineno)
    return out


def test_the_boot_reaper_is_wired_before_the_watchdog():
    main = Path(__file__).parent.parent / "src" / "ghost_agent" / "main.py"
    reap, dog = _call_lines(main, "start_leaf_reaper"), _call_lines(main, "biological_watchdog")
    assert reap and dog and min(reap) < min(dog)


def test_leaf_end_bounds_run_stops_and_lab_destroys_separately():
    import ast
    from ghost_agent.core import coding_loop as CL
    assert CL.LEAF_RUN_STOP_BOUND_S == 90.0 and CL.LEAF_LAB_DESTROY_BOUND_S == 150.0
    tree = ast.parse(Path(CL.__file__).read_text())
    bounds = {n.args[1].id for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "wait_for"
              and len(n.args) == 2 and isinstance(n.args[1], ast.Name)}
    assert {"LEAF_RUN_STOP_BOUND_S", "LEAF_LAB_DESTROY_BOUND_S"} <= bounds


def test_each_attempt_drops_an_earlier_attempts_lab_note():
    from ghost_agent.core import coding_loop as CL
    drop, snap = _call_lines(CL.__file__, "_drop_seen"), _call_lines(CL.__file__, "snapshot_workspace")
    assert drop and snap and min(drop) < max(snap)


def test_a_no_lab_answer_on_an_owner_hold_has_its_own_state():
    from ghost_agent.core import project_advancer as PA
    from ghost_agent.tools import projects as P
    assert not PA.LAB_NO_LAB_HELD_PREFIX.startswith(PA.LAB_WAIT_PREFIX[:20])
    assert "lab_owner_hold" in _str_consts(PA.__file__)
    assert P._ADVANCE_STOP_BLURB["lab_owner_hold"].startswith("Stopped: the build could not get a lab")


# ------------------------------------------------------------------ projects

def test_an_unattended_turn_gets_its_projects_scope():
    from ghost_agent import main as M

    class _S:
        def __init__(self, rows):
            self.rows = rows

        def get_project(self, pid):
            return self.rows.get(pid)
    ctx = SimpleNamespace(project_store=_S({"p1": {"id": "p1", "status": "ACTIVE"},
                                             "p2": {"id": "p2", "status": "ARCHIVED"}}))
    s = M._unattended_project_scope(ctx, "p1")
    assert s is not None and s.project_id == "p1" and s.origin == RS.ORIGIN_UNATTENDED
    assert M._unattended_project_scope(ctx, "p2") is None
    assert M._unattended_project_scope(ctx, "gone") is None
    assert M._unattended_project_scope(ctx, None) is None


def test_a_deleted_projects_folder_is_not_recreated(tmp_path):
    from ghost_agent.tools.file_system import project_scoped_sandbox

    class _Store:
        def get_project(self, pid):
            return None
    ctx = SimpleNamespace(sandbox_dir=tmp_path, current_project_id="abc123def456", project_store=_Store())
    project_scoped_sandbox(ctx)
    assert not (tmp_path / "projects" / "abc123def456").exists()

    class _Store2:
        def get_project(self, pid):
            return {"id": pid}
    ctx.project_store = _Store2()
    project_scoped_sandbox(ctx)
    assert (tmp_path / "projects" / "abc123def456").is_dir()
