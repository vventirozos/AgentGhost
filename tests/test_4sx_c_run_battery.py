"""§4SX Phase C battery round 1: the cases scripts/mutate_4sx_c.py found unpinned.

Scripted supervisors (FakeSvc / FakeJobs, injected through ``runs._supervisors``) give exact control over
tokens, liveness and answers the real supervisors only produce in rare races.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.tools import labs as L
from ghost_agent.tools import runs as R
from ghost_agent.tools.outcome import OutcomeStatus

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_4sx_b_lab_tool import FakeLabs, LAB  # noqa: E402
from test_4sx_c_run_tool import _ctx, _lab, scope  # noqa: E402,F401  (fixture re-export)
from test_sandbox_job_promotion import FakeSandbox as JobSandbox  # noqa: E402

OWNER = "conversation:conv1"
HALF = R.services_key_half(OWNER)
KEY = f"{HALF}:web"


@pytest.fixture
def fake(monkeypatch):
    f = FakeLabs()
    monkeypatch.setattr(L, "_call", f)
    monkeypatch.setattr(L, "_time_left_s", lambda: None)
    return f


class FakeSvc:
    """The ServiceSupervisor surface `run` uses. ``alive`` is a list of answers for `_entry_state`
    (the last one repeats)."""

    def __init__(self):
        self.ents = {}
        self.alive = [True]
        self.starts, self.stops, self.restarts = [], [], []
        self.new_token = True
        self.stop_removes = True
        self.n = 0

    def _tok(self):
        self.n += 1
        return f"t{self.n}"

    def list_entries(self):
        return [dict(e) for e in self.ents.values()]

    def _entry_state(self, e):
        return self.alive.pop(0) if len(self.alive) > 1 else self.alive[0]

    def start(self, name, command, port=None, project_id=None):
        self.starts.append((name, command, port, project_id))
        key = f"{project_id}:{name}"
        old = self.ents.get(key)
        tok = self._tok() if (self.new_token or old is None) else old["token"]
        self.ents[key] = {"key": key, "pid": 100 + self.n, "token": tok, "port": port or 8100}
        return f"started {name}"

    def stop(self, key):
        self.stops.append(key)
        if self.stop_removes:
            self.ents.pop(key, None)
        return f"stopped {key}"

    def restart(self, key):
        self.restarts.append(key)
        if self.new_token and key in self.ents:
            self.ents[key] = {**self.ents[key], "token": self._tok()}
        return f"restarted {key}"

    def logs(self, key, n):
        return f"log of {key}"


class FakeJobs:
    def __init__(self):
        self.ents = {}
        self.calls = []
        self.dup = False
        self.raise_on_start = None

    def reap(self):
        pass

    def list_entries(self):
        return [dict(e) for e in self.ents.values()]

    def start_detached(self, command, **kw):
        self.calls.append((command, kw))
        if self.raise_on_start:
            raise self.raise_on_start
        e = {"id": f"job{len(self.ents) + 1}", "state": "running", "exit_code": None, "label": kw.get("label"),
             "deadline_at": 2700.0, "promoted_at": 0.0, "started_via": kw.get("started_via", "run")}
        if self.dup:
            return {**e, "duplicate": True}, ""
        self.ents[e["id"]] = e
        return e, ""

    def cancel(self, jid):
        self.ents[jid]["state"] = "cancelled"
        return True, "cancelled"

    def log_tail(self, jid, n):
        return "tail"


@pytest.fixture
def sups(monkeypatch):
    s = SimpleNamespace(svc=FakeSvc(), jobs=FakeJobs())
    monkeypatch.setattr(R, "_supervisors", lambda context: (s.svc, s.jobs))
    return s


def _reg(c):
    from ghost_agent.memory.run_registry import registry_for
    return registry_for(c)


async def _web(c):
    r = await R.tool_run(c, action="start", name="web", command="python3 a.py", where="workspace",
                         lifetime="server", port=8100)
    assert r.status is OutcomeStatus.OK, str(r)
    return r


# ------------------------------------------------------------------ dispatch

async def test_an_unknown_action_is_named_as_unknown(tmp_path, scope):
    c = _ctx(tmp_path)
    scope(c)
    r = await R.tool_run(c, action="nope", name="web")
    assert r.status is OutcomeStatus.REJECTED and "unknown action 'nope'" in r


async def test_no_registry_means_nothing_runs(tmp_path, scope, sups, monkeypatch):
    import ghost_agent.memory.run_registry as RRm
    monkeypatch.setattr(RRm, "registry_for", lambda context: None)
    c = _ctx(tmp_path)
    scope(c)
    r = await R.tool_run(c, action="start", name="web", command="x", where="workspace", lifetime="server")
    assert r.reason_code == "run_registry_unavailable" and r.world_changed is False and sups.svc.starts == []


async def test_a_port_out_of_range_is_refused(tmp_path, scope, sups):
    c = _ctx(tmp_path)
    scope(c)
    r = await R.tool_run(c, action="start", name="web", command="x", where="workspace", lifetime="server",
                         port=70000)
    assert r.status is OutcomeStatus.REJECTED and "65535" in r and sups.svc.starts == []


async def test_a_lab_member_must_be_a_safe_name(tmp_path, scope, fake):
    c = _ctx(tmp_path)
    scope(c)
    for bad in (f"{LAB}/DB", f"{LAB}/db;rm", f"{LAB}/"):
        r = await R.tool_run(c, action="start", name="j", command="x", where=bad, lifetime="job")
        assert r.status is OutcomeStatus.REJECTED, bad
    assert fake.calls == []


async def test_an_unanswered_live_run_still_blocks_its_name(tmp_path, scope, fake):
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_1000000001"}))
    await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    fake.on("GET", f"/v1/labs/{LAB}/jobs/job_1000000001", L.LabsNoAnswer("ReadTimeout"))
    r = await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    assert r.reason_code == "run_exists" and "already unknown" in r
    assert sum(1 for x in fake.calls if x.method == "POST") == 1


# ------------------------------------------------------------------ workspace server

async def test_a_start_that_left_the_old_token_is_failed(tmp_path, scope, sups):
    """A dead row of ours under the key, and the start did not write a new token: nothing started."""
    sups.svc.ents[KEY] = {"key": KEY, "pid": 9, "token": "old", "port": 8100}
    sups.svc.alive = [False, True]                    # dead before; "alive" after — only the token tells
    sups.svc.new_token = False
    c = _ctx(tmp_path)
    scope(c)
    r = await R.tool_run(c, action="start", name="web", command="x", where="workspace", lifetime="server")
    assert r.reason_code == "run_start_failed" and _reg(c).rows(OWNER) == []


async def test_a_start_whose_process_is_dead_is_failed(tmp_path, scope, sups):
    sups.svc.alive = [False]
    c = _ctx(tmp_path)
    scope(c)
    r = await R.tool_run(c, action="start", name="web", command="x", where="workspace", lifetime="server")
    assert r.reason_code == "run_start_failed" and _reg(c).rows(OWNER) == []


async def test_an_unrecorded_server_start_is_stopped_or_says_it_may_still_run(tmp_path, scope, sups, monkeypatch):
    from ghost_agent.memory.run_registry import RunRegistry
    monkeypatch.setattr(RunRegistry, "add", lambda self, **k: None)
    c = _ctx(tmp_path)
    scope(c)
    r = await R.tool_run(c, action="start", name="web", command="x", where="workspace", lifetime="server")
    assert r.reason_code == "run_registry_unavailable" and "stopped again" in r and sups.svc.stops == [KEY]
    sups.svc.stop_removes = False                     # the stop did not take: the row is still alive
    r = await R.tool_run(c, action="start", name="web", command="x", where="workspace", lifetime="server")
    assert "may still run" in r and KEY in r


async def test_a_stop_asked_inside_the_start_starts_nothing(tmp_path, scope, sups, monkeypatch):
    answers = iter([False])
    monkeypatch.setattr(R, "_stop_asked", lambda: next(answers, True))
    c = _ctx(tmp_path)
    scope(c)
    r = await R.tool_run(c, action="start", name="web", command="x", where="workspace", lifetime="server")
    assert r.status is not OutcomeStatus.OK and "not started" in r and sups.svc.starts == []


# ------------------------------------------------------------------ workspace job

async def test_a_duplicate_job_is_refused_and_not_recorded(tmp_path, scope, sups):
    sups.jobs.dup = True
    c = _ctx(tmp_path)
    scope(c)
    r = await R.tool_run(c, action="start", name="j", command="sleep 9", where="workspace", lifetime="job")
    assert r.reason_code == "run_exists" and _reg(c).rows(OWNER) == []


async def test_a_job_start_error_is_a_failed_start(tmp_path, scope, sups):
    sups.jobs.raise_on_start = RuntimeError("no folder")
    c = _ctx(tmp_path)
    scope(c)
    r = await R.tool_run(c, action="start", name="j", command="x", where="workspace", lifetime="job")
    assert r.reason_code == "run_start_failed" and "no folder" in r


async def test_a_project_job_runs_in_its_project_folder(tmp_path, scope, sups):
    c = _ctx(tmp_path, pid="41f745517fe4")
    scope(c)
    r = await R.tool_run(c, action="start", name="j", command="x", where="workspace", lifetime="job")
    assert r.status is OutcomeStatus.OK
    (_, kw), = sups.jobs.calls
    assert kw["workdir"] == "/workspace/projects/41f745517fe4" and kw["project_id"] == "41f745517fe4"


async def test_a_cancelled_job_reads_as_stopped(tmp_path, scope, sups):
    c = _ctx(tmp_path)
    scope(c)
    await R.tool_run(c, action="start", name="j", command="x", where="workspace", lifetime="job")
    sups.jobs.ents["job1"]["state"] = "cancelled"
    s = await R.tool_run(c, action="status", name="j")
    assert "j  job  workspace  stopped" in s


# ------------------------------------------------------------------ lab

async def test_an_unrecorded_lab_start_whose_kill_left_it_running_says_not_confirmed(tmp_path, scope, fake,
                                                                                    monkeypatch):
    from ghost_agent.memory.run_registry import RunRegistry
    monkeypatch.setattr(RunRegistry, "add", lambda self, **k: None)
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": "job_2000000001"}))
    fake.on("DELETE", f"/v1/labs/{LAB}/jobs/job_2000000001", (200, {"state": "running"}))
    r = await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    assert "NOT confirmed" in r


async def test_a_lab_start_without_a_job_id_is_failed(tmp_path, scope, fake):
    c = _ctx(tmp_path)
    scope(c)
    _lab(fake)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {}))
    r = await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime="job")
    assert r.reason_code == "run_start_failed" and "no job id" in r and _reg(c).rows(OWNER) == []


async def _lab_job(c, fake, jid="job_3000000001", lifetime="job"):
    _lab(fake)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (202, {"job_id": jid}))
    r = await R.tool_run(c, action="start", name="j", command="x", where=f"{LAB}/db", lifetime=lifetime,
                         port=80 if lifetime == "server" else None)
    assert r.status is OutcomeStatus.OK, str(r)
    return f"/v1/labs/{LAB}/jobs/{jid}"


async def test_a_host_kill_reads_as_stopped(tmp_path, scope, fake):
    c = _ctx(tmp_path)
    scope(c)
    path = await _lab_job(c, fake)
    fake.on("GET", path, (200, {"state": "exited", "killed": True, "exit_code": None}))
    s = await R.tool_run(c, action="status", name="j")
    assert "stopped (killed)" in s


async def test_an_unanswered_probe_is_not_written_back(tmp_path, scope, fake):
    c = _ctx(tmp_path)
    scope(c)
    path = await _lab_job(c, fake)
    fake.on("GET", path, L.LabsNoAnswer("ReadTimeout"))
    s = await R.tool_run(c, action="status", name="j")
    assert "unknown" in s
    row = _reg(c).latest_by_name(OWNER, "j")
    assert row["state"] == "running" and row["state_note"] == ""


async def test_a_failed_job_collects_as_failed(tmp_path, scope, fake):
    c = _ctx(tmp_path)
    scope(c)
    path = await _lab_job(c, fake)
    fake.on("GET", path, (200, {"state": "exited", "exit_code": 2, "output_tail": "boom"}))
    co = await R.tool_run(c, action="collect", name="j")
    assert co.status is OutcomeStatus.FAILED and co.reason_code == "run_job_failed" and "EXIT CODE: 2" in co


async def test_collect_of_a_server_is_refused(tmp_path, scope, fake):
    c = _ctx(tmp_path)
    scope(c)
    await _lab_job(c, fake, lifetime="server")
    r = await R.tool_run(c, action="collect", name="j")
    assert r.status is OutcomeStatus.REJECTED and "reads a finished job" in r


async def test_a_lab_stop_answered_still_running_is_failed(tmp_path, scope, fake):
    c = _ctx(tmp_path)
    scope(c)
    path = await _lab_job(c, fake)
    fake.on("GET", path, (200, {"state": "running"}))
    fake.on("DELETE", path, (200, {"state": "running"}))
    r = await R.tool_run(c, action="stop", name="j")
    assert r.reason_code == "run_stop_failed"
    assert _reg(c).latest_by_name(OWNER, "j")["state"] == "running"


async def test_a_lab_stop_on_a_destroyed_lab_is_already_gone(tmp_path, scope, fake):
    c = _ctx(tmp_path)
    scope(c)
    path = await _lab_job(c, fake)
    fake.on("GET", path, (200, {"state": "running"}))
    fake.on("DELETE", path, (404, {"error": "unknown_lab"}))
    r = await R.tool_run(c, action="stop", name="j")
    assert r.status is OutcomeStatus.OK and "already gone" in r and r.world_changed is False
    assert _reg(c).latest_by_name(OWNER, "j")["state"] == "lost"


# ------------------------------------------------------------------ workspace stop / restart

async def test_a_stop_of_a_run_replaced_since_its_last_probe_is_refused(tmp_path, scope, sups):
    c = _ctx(tmp_path)
    scope(c)
    await _web(c)
    sups.svc.ents[KEY]["token"] = "someone-else"
    r = await R.tool_run(c, action="stop", name="web")
    assert r.reason_code == "run_replaced" and sups.svc.stops == []


async def test_a_server_alive_after_the_stop_is_failed(tmp_path, scope, sups):
    c = _ctx(tmp_path)
    scope(c)
    await _web(c)
    sups.svc.stop_removes = False
    sups.svc.alive = [True]
    r = await R.tool_run(c, action="stop", name="web")
    assert r.reason_code == "run_stop_failed" and _reg(c).latest_by_name(OWNER, "web")["state"] == "running"


async def test_a_stop_whose_check_got_no_answer_is_unresolved(tmp_path, scope, sups):
    c = _ctx(tmp_path)
    scope(c)
    await _web(c)
    sups.svc.stop_removes = False
    sups.svc.alive = [True, None]                     # the first probe sees it; the check after gets nothing
    r = await R.tool_run(c, action="stop", name="web")
    assert r.status is OutcomeStatus.UNRESOLVED and r.reason_code == "run_unverified"


async def test_an_unanswered_service_probe_is_unknown_not_exited(tmp_path, scope, sups):
    c = _ctx(tmp_path)
    scope(c)
    await _web(c)
    sups.svc.alive = [None]
    s = await R.tool_run(c, action="status", name="web")
    assert "unknown (the sandbox did not answer" in s


async def test_a_restart_closes_the_old_row_and_opens_a_new_one(tmp_path, scope, sups):
    c = _ctx(tmp_path)
    scope(c)
    await _web(c)
    old = _reg(c).live_by_name(OWNER, "web")
    r = await R.tool_run(c, action="restart", name="web")
    assert r.status is OutcomeStatus.OK and r.world_changed is True, str(r)
    o = _reg(c).get(old["run_id"])
    new = _reg(c).live_by_name(OWNER, "web")
    assert o["state"] == "stopped" and o["state_note"] == f"restarted as {new['run_id']}"
    assert new["run_id"] != old["run_id"] and new["backend_token"] != old["backend_token"]


# ------------------------------------------------------------------ the registry

def test_registry_end_is_stamped_once_and_live_rows_exclude_ended(tmp_path, monkeypatch):
    from ghost_agent.memory import run_registry as RRm
    reg = RRm.RunRegistry(tmp_path)
    a = reg.add(name="a", place="workspace", lifetime="job", command="x", port=None, owner="o", origin="owner",
                backend="jobs", backend_ref="j1")
    reg.set_state(a, "exited", "exit 0")
    first = reg.get(a)["ended_at"]
    assert first is not None
    real = time.time
    monkeypatch.setattr(RRm.time, "time", lambda: real() + 100)
    reg.set_state(a, "stopped", "")
    assert reg.get(a)["ended_at"] == first
    assert reg.live_by_name("o", "a") is None
    assert [r["run_id"] for r in reg.rows("o", live_only=True)] == []


def test_registry_latest_is_the_newest_and_rows_are_per_owner(tmp_path, monkeypatch):
    from ghost_agent.memory import run_registry as RRm
    reg = RRm.RunRegistry(tmp_path)
    t = [1000.0]
    monkeypatch.setattr(RRm.time, "time", lambda: t[0])
    a = reg.add(name="a", place="w", lifetime="job", command="", port=None, owner="o", origin="", backend="jobs",
                backend_ref="1")
    t[0] += 5
    b = reg.add(name="a", place="w", lifetime="job", command="", port=None, owner="o", origin="", backend="jobs",
                backend_ref="2")
    reg.add(name="a", place="w", lifetime="job", command="", port=None, owner="other", origin="", backend="jobs",
            backend_ref="3")
    assert reg.latest_by_name("o", "a")["run_id"] == b
    assert [r["run_id"] for r in reg.rows("o")] == [a, b]


def test_registry_prunes_only_old_ended_rows(tmp_path, monkeypatch):
    from ghost_agent.memory import run_registry as RRm
    reg = RRm.RunRegistry(tmp_path)
    t = [1_000_000.0]
    monkeypatch.setattr(RRm.time, "time", lambda: t[0])
    kw = dict(place="w", lifetime="job", command="", port=None, owner="o", origin="", backend="jobs", backend_ref="")
    live = reg.add(name="live", **kw)
    old = reg.add(name="old", **kw)
    reg.set_state(old, "exited")
    t[0] += 10
    recent = reg.add(name="recent", **kw)
    reg.set_state(recent, "exited")
    t[0] += RRm.PRUNE_AFTER_S
    reg.add(name="new", **kw)
    names = {r["name"] for r in reg.rows("o")}
    assert "old" not in names and {"live", "recent", "new"} <= names
    assert reg.get(live)["state"] == "running"


# ------------------------------------------------------------------ jobs start_detached guards

def test_start_detached_kills_what_it_could_not_record(tmp_path, monkeypatch):
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    sup = SandboxJobSupervisor(JobSandbox(tmp_path / "ws"))
    killed = []
    monkeypatch.setattr(sup, "_launch", lambda jid, opts: 4242)
    monkeypatch.setattr(sup, "_read_pid", lambda jid: 4242)
    monkeypatch.setattr(sup, "_kill_pgroup", lambda pid: killed.append(pid))
    monkeypatch.setattr(sup, "_promote", lambda *a, **k: None)
    e, why = sup.start_detached("sleep 5", label="a")
    assert e is None and "cap" in why and killed == [4242]

    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(sup, "_promote", boom)
    with pytest.raises(RuntimeError):
        sup.start_detached("sleep 6", label="b")
    assert killed == [4242, 4242]


# ------------------------------------------------------------------ the operator route


async def test_reconcile_agrees_with_the_supervisors_and_reports_old_tool_rows(tmp_path, scope, sups, fake):
    """Acceptance §14 C: both registries agree. A vanished service → lost; a lab row is counted, never read;
    an old-tool service is reported, not adopted; a run job with no run row is named."""
    c = _ctx(tmp_path)
    scope(c)
    await _web(c)
    await R.tool_run(c, action="start", name="wj", command="x", where="workspace", lifetime="job")
    await _lab_job(c, fake, jid="job_4000000001")
    sups.svc.ents["legacy:api"] = {"key": "legacy:api", "pid": 7, "token": "z", "port": 8101}
    sups.jobs.ents["job9"] = {"id": "job9", "state": "running", "started_via": "run"}
    n_lab = len(fake.calls)
    rep = await R.reconcile(c)
    assert rep["live"] == 3 and rep["probed"] == 2 and rep["labs"] == 1 and rep["ended"] == {}
    assert rep["untracked_services"] == ["legacy:api"] and rep["orphan_run_jobs"] == ["job9"]
    assert len(fake.calls) == n_lab                                   # design r1 M4: labs never read
    assert "legacy:api" in sups.svc.ents                              # reported, not adopted/stopped
    sups.svc.ents.pop(KEY)                                            # stopped outside `run`
    rep = await R.reconcile(c)
    web = _reg(c).latest_by_name(OWNER, "web")
    assert web["state"] == "lost" and web["run_id"] in rep["ended"] and rep["live"] == 3
    line = R.reconcile_line(rep)
    assert "1 ended" in line and "lab 1 not probed" in line and "job9" in line
    rep = await R.reconcile(c)
    assert rep["live"] == 2 and rep["ended"] == {}                    # the lost row is history now


async def test_reconcile_leaves_an_unanswered_row_running(tmp_path, scope, sups):
    c = _ctx(tmp_path)
    scope(c)
    await _web(c)
    sups.svc.alive = [None]
    rep = await R.reconcile(c)
    assert rep["unknown"] == 1 and rep["ended"] == {}
    assert _reg(c).live_by_name(OWNER, "web")["state"] == "running"
    assert "1 unanswered" in R.reconcile_line(rep)


async def test_the_boot_line_never_raises_and_warns_on_an_orphan(tmp_path, scope, sups, monkeypatch):
    lines = []
    monkeypatch.setattr(R, "pretty_log", lambda title, msg, **k: lines.append((title, msg, k.get("level"))))
    c = _ctx(tmp_path)
    scope(c)
    await R.runs_boot_line(c)
    assert lines[-1][0] == "Runs" and lines[-1][2] == "INFO" and "0 live run(s)" in lines[-1][1]
    sups.jobs.ents["job9"] = {"id": "job9", "state": "running", "started_via": "run"}
    await R.runs_boot_line(c)
    assert lines[-1][2] == "WARNING" and "job9" in lines[-1][1]

    async def boom(context):
        raise RuntimeError("db locked")
    monkeypatch.setattr(R, "reconcile", boom)
    await R.runs_boot_line(c)
    assert "reconcile failed: RuntimeError" in lines[-1][1]


def test_the_boot_line_is_wired_after_the_labs_line():
    import ast
    import ghost_agent.main as M
    src = Path(M.__file__).read_text()
    names = [n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(ast.parse(src))
             if isinstance(n, (ast.Name, ast.Attribute)) and getattr(n, "id", getattr(n, "attr", "")) in
             ("start_labs_boot_line", "start_runs_boot_line")]
    assert "start_runs_boot_line" in names and "start_labs_boot_line" in names
    calls = {n.func.id: n.lineno for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in ("start_labs_boot_line", "start_runs_boot_line")}
    assert calls["start_labs_boot_line"] < calls["start_runs_boot_line"]
    # inside its own try (a failing import/schedule never breaks the boot)
    tries = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Try)
             and any(isinstance(x, ast.Call) and getattr(x.func, "id", "") == "start_runs_boot_line"
                     for b in n.body for x in ast.walk(b))]
    assert len(tries) == 1 and len(tries[0].body) <= 2


async def test_reconcile_reports_only_true_strays(tmp_path, scope, sups):
    """review: a run's own crashed server is not "outside run"; a tracked run job is no orphan; an ended or
    non-run job is not one either."""
    c = _ctx(tmp_path)
    scope(c)
    await _web(c)
    await R.tool_run(c, action="start", name="wj", command="x", where="workspace", lifetime="job")
    sups.jobs.ents["job7"] = {"id": "job7", "state": "done", "started_via": "run"}
    sups.jobs.ents["job8"] = {"id": "job8", "state": "running"}                  # the jobs tool's own
    sups.svc.alive = [False]                                                      # web crashed
    rep = await R.reconcile(c)
    assert list(rep["ended"].values()) == ["exited (its process is not running)"]
    assert rep["untracked_services"] == [] and rep["orphan_run_jobs"] == []


async def test_reconcile_with_the_sandbox_down_or_a_raising_probe_still_reports(tmp_path, scope, sups, monkeypatch):
    c = _ctx(tmp_path)
    scope(c)
    await _web(c)
    await R.tool_run(c, action="start", name="wj", command="x", where="workspace", lifetime="job")
    real = R._probe

    async def flaky(context, row, origin):
        if row["backend"] == "services":
            raise RuntimeError("registry dir is a symlink")
        return await real(context, row, origin)
    monkeypatch.setattr(R, "_probe", flaky)
    rep = await R.reconcile(c)
    assert rep["probed"] == 2 and rep["unknown"] == 1 and rep["ended"] == {}
    assert _reg(c).live_by_name(OWNER, "web")["state"] == "running"
    monkeypatch.setattr(R, "_probe", real)
    monkeypatch.setattr(R, "_supervisors", lambda context: (None, None))
    rep = await R.reconcile(c)
    assert rep["unknown"] == 2 and rep["ended"] == {} and "2 unanswered" in R.reconcile_line(rep)


async def test_start_runs_boot_line_schedules_one_held_task(tmp_path, monkeypatch):
    seen = []

    async def fake(context):
        seen.append(context)
    monkeypatch.setattr(R, "runs_boot_line", fake)
    R.start_runs_boot_line("ctx")
    assert len(R._BOOT_TASKS) == 1
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert seen == ["ctx"] and not R._BOOT_TASKS


async def test_a_service_left_by_a_run_that_ended_earlier_is_outside_run(tmp_path, scope, sups):
    """r2: only runs live in THIS pass own their services; history does not."""
    c = _ctx(tmp_path)
    scope(c)
    await _web(c)
    sups.svc.alive = [False]
    await R.reconcile(c)                                           # pass 1: web reads exited (its row stays)
    rep = await R.reconcile(c)                                     # pass 2: web is history now
    assert rep["live"] == 0 and rep["untracked_services"] == [KEY]
