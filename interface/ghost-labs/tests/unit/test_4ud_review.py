"""Pins for the §4UD review (host side): F1 lock × action table, F2 probe deadline, F6 job state, F7 pump,
F12 kept reads, F13 drop_data, F17 logs bytes, F18 PGUSER, G3 TTL clamp + loopback hint.

Docker is the in-memory fake (tests/unit/fakes.py). The lab lock is the REAL one; only the time a caller waits for
it (LOCK_WAIT_S) is shortened. Each test fails on the code before its fix."""
import http.server
import socket
import socketserver
import threading
import time
from types import SimpleNamespace

import pytest

from ghost_labs import labs as L
from ghost_labs.errors import LabError
from ghost_labs.store import dumps

from .fakes import FakeContainer
from .test_core import ROOT, add_lab, env  # noqa: F401
from .test_kept_fixes import app_of, kept, make_kept  # noqa: F401


def _member(m, lab_id, member="m1"):
    c = FakeContainer(m.docker, m.container_name(lab_id, member), {"ghost.labs": "1", "ghost.lab.id": lab_id,
                                                                  "ghost.lab.member": member})
    m.docker._containers[c.name] = c
    return c


class _RunningExec:
    """A foreground exec that runs until `release` — through the real Manager.exec."""

    def __init__(self, m, lab_id, origin="autopilot"):
        self.m, self.lab_id = m, lab_id
        self.release, self.started = threading.Event(), threading.Event()
        self.result, self.error = None, None
        m.docker.api.exec_create = lambda cid, argv, **kw: {"Id": "x1"}

        def exec_start(i, stream=True, demux=True):
            self.started.set()
            self.release.wait(20)
            yield (b"done\n", None)
        m.docker.api.exec_start = exec_start
        m.docker.api.exec_inspect = lambda i: {"ExitCode": 0}

        def go():
            try:
                self.result = m.exec(lab_id, {"member": "m1", "cmd": "sleep 900", "timeout_s": 900}, origin)
            except LabError as e:
                self.error = e
        self.t = threading.Thread(target=go, daemon=True)
        self.t.start()
        assert self.started.wait(5), "the exec never started"

    def end(self):
        self.release.set()
        self.t.join(10)
        assert not self.t.is_alive()


@pytest.fixture
def running(env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(L, "LOCK_WAIT_S", 0.4)
    add_lab(env, "lab_x", origin="autopilot", owner="project:p1", owner_use=0)
    _member(env, "lab_x")
    ex = _RunningExec(env, "lab_x")
    yield env, ex
    ex.release.set()


# ================================================================== F1: the lock × action table
def test_f1_destroy_does_not_wait_behind_a_running_exec_and_the_exec_says_destroyed(running):
    m, ex = running
    t0 = time.monotonic()
    out = m.destroy("lab_x", by="autopilot", origin="autopilot")          # a leaf's end
    assert out["destroyed"] is True and time.monotonic() - t0 < 0.3       # (the old code: busy after the wait)
    assert m.store.lab("lab_x") is None
    ex.end()
    assert ex.result is None and ex.error.code == "unknown_lab" and "destroyed" in ex.error.message


@pytest.mark.parametrize("op", ["kill", "stop", "restart", "pause", "isolate"])
def test_f1_member_ops_go_ahead_while_an_exec_runs(running, op):
    m, ex = running
    m.docker.networks.create("ghost-lab-lab_x", labels={})
    out = m.member_action("lab_x", "m1", op, "owner")
    assert out["action"] == op
    ex.end()
    assert ex.error is None and ex.result["exit_code"] == 0


def test_f1_a_second_exec_and_a_screenshot_still_wait_then_answer_busy(running):
    m, ex = running
    with pytest.raises(LabError) as e:
        m.exec("lab_x", {"member": "m1", "cmd": "true"}, "owner")
    assert e.value.code == "busy" and "command is still running" in e.value.message
    with pytest.raises(LabError) as e:
        with m.locked("lab_x"):                 # what screenshot and every kept action take
            pass
    assert e.value.code == "busy"
    assert m.lock("lab_x").acquire(blocking=False) is False      # the kept tick / an eviction: skipped
    ex.end()
    assert m.exec("lab_x", {"member": "m1", "cmd": "true"}, "owner")["exit_code"] == 0    # free again afterwards
    assert m.lock("lab_x").acquire(blocking=False) is True
    m.lock("lab_x").release()


def test_f1_reads_and_put_never_wait_for_an_exec(running):
    m, ex = running
    assert m.detail("lab_x")["lab_id"] == "lab_x"                                   # status
    assert "log line" in m.logs("lab_x", "m1", 10, "autopilot")["logs"]             # logs
    import base64
    assert m.put_file("lab_x", {"member": "m1", "path": "/tmp/a", "content_b64": base64.b64encode(b"x").decode()},
                      "autopilot")["bytes"] == 1                                    # put
    assert m.extend("lab_x", {"ttl_idle_s": 600}, "autopilot")["lab_id"] == "lab_x"


def test_f1_two_state_changes_still_exclude_each_other(running):
    m, ex = running
    lk = m.lock("lab_x")
    assert lk.acquire(timeout=0.1, over_exec=True)         # a member op in progress
    try:
        with pytest.raises(LabError) as e:
            m.destroy("lab_x", by="owner", origin="owner")
        assert e.value.code == "busy" and m.store.lab("lab_x")
    finally:
        lk.release()


def test_f1_the_idle_reaper_skips_a_lab_with_a_running_command_and_the_hard_limit_does_not(running):
    m, ex = running
    m.store.update_lab("lab_x", last_used_at=time.time() - 10 ** 6, ttl_idle_s=60)
    assert m.reap_once() == [] and m.store.lab("lab_x")              # in use is not idle
    m.store.update_lab("lab_x", created_at=time.time() - 10 ** 7)     # past its max lifetime
    assert m.reap_once() == [{"lab_id": "lab_x", "reason": "max_lifetime"}]
    ex.end()
    assert ex.error.code == "unknown_lab"


def test_f1_the_exec_touches_the_lab_when_it_ends(running):
    m, ex = running
    m.store.update_lab("lab_x", last_used_at=1.0)
    ex.end()
    assert m.store.lab("lab_x")["last_used_at"] > time.time() - 5
    assert m.exec_running("lab_x") is False


def test_f1_an_owner_create_does_not_evict_a_lab_with_a_running_command(running, monkeypatch):
    m, ex = running
    for i in range(3):
        add_lab(m, f"lab_o{i}", origin="owner")
    monkeypatch.setattr(m, "wait_ready", lambda *a, **k: None)
    monkeypatch.setattr(m, "wait_tor", lambda *a, **k: None)
    with pytest.raises(LabError) as e:                       # 4 labs live, the autopilot one is not a victim
        m.create({"template": "pg-single", "owner": "conversation:c"}, "owner")
    assert e.value.code == "lab_limit" and m.store.lab("lab_x")
    ex.end()
    out = m.create({"template": "pg-single", "owner": "conversation:c"}, "owner")
    assert [v["lab_id"] for v in out["evicted"]] == ["lab_x"]


def test_f1_a_failed_exec_start_leaves_the_lock_free(env):  # noqa: F811
    add_lab(env, "lab_y")
    _member(env, "lab_y")
    with pytest.raises(LabError):
        env.exec("lab_y", {"member": "nope", "cmd": "true"}, "owner")
    assert env.lock("lab_y").locked() is False


# ================================================================== F2: one total deadline for the http probe
class _Drip(socketserver.BaseRequestHandler):
    head = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n"
    drip_head = False

    def handle(self):
        try:
            self.request.recv(4096)
            if self.drip_head:
                for b in b"HTTP/1.1 200 OK\r\nX-A: 1\r\nX-B: 2\r\nX-C: 3\r\n":
                    self.request.sendall(bytes([b]))
                    time.sleep(0.05)
            else:
                self.request.sendall(self.head)
            for _ in range(200):
                self.request.sendall(b"d")
                time.sleep(0.05)                 # far under any per-recv timeout
        except OSError:
            pass


class _Srv(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _serve(handler):
    srv = _Srv(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_f2_a_dripping_body_is_cut_at_the_total_deadline_and_marked_truncated():
    srv = _serve(type("D", (_Drip,), {}))
    try:
        t0 = time.monotonic()
        out = L.http_probe("127.0.0.1", srv.server_address[1], "GET", "/", "m1:80", 16384, total_s=0.6)
        took = time.monotonic() - t0
    finally:
        srv.shutdown()
        srv.server_close()
    assert took < 1.5, took                       # the old read ran for as long as bytes kept coming
    assert out["status"] == 200 and out["truncated"] is True and 1 <= len(out["raw"]) < 40


def test_f2_dripping_headers_cannot_pass_the_deadline_either():
    srv = _serve(type("D", (_Drip,), {"drip_head": True}))
    try:
        t0 = time.monotonic()
        out = L.http_probe("127.0.0.1", srv.server_address[1], "GET", "/", "m1:80", 16384, total_s=0.5)
        took = time.monotonic() - t0
    finally:
        srv.shutdown()
        srv.server_close()
    assert took < 1.5 and out["status"] is None and "0.5" in out["error"]


class _Silent(socketserver.BaseRequestHandler):
    """Headers, then nothing — and no Content-Length, so the body ends only with the connection."""

    def handle(self):
        try:
            self.request.recv(4096)
            self.request.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\nfirst")
            time.sleep(5)
        except OSError:
            pass


def test_f2_a_silent_stream_is_cut_at_the_deadline_not_at_twice_it():
    srv = _serve(_Silent)
    try:
        t0 = time.monotonic()
        out = L.http_probe("127.0.0.1", srv.server_address[1], "GET", "/", "m1:80", 16384, total_s=0.6)
        took = time.monotonic() - t0
    finally:
        srv.shutdown()
        srv.server_close()
    assert 0.5 < took < 1.0, took
    assert out["status"] == 200 and out["raw"] == b"first" and out["truncated"] is True


class _Garbage(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.recv(4096)
        self.request.sendall(b"this is not http\r\n\r\n")


def test_f2_http_client_errors_are_a_clean_status_null_answer(env):  # noqa: F811
    m = env
    add_lab(m, "lab_h")
    m.store.x("UPDATE members SET ip='127.0.0.1' WHERE lab_id='lab_h'")
    srv = _serve(_Garbage)
    try:
        port = srv.server_address[1]
        bad_line = m.http("lab_h", {"member": "m1", "port": port, "path": "/"}, "owner")
        bad_url = m.http("lab_h", {"member": "m1", "port": port, "path": "/a b"}, "owner")     # InvalidURL
    finally:
        srv.shutdown()
        srv.server_close()
    assert bad_line["status"] is None and "BadStatusLine" in bad_line["error"]
    assert bad_url["status"] is None and "InvalidURL" in bad_url["error"]


def test_f2_a_whole_body_is_not_marked_truncated(env):  # noqa: F811
    from .test_review_r11 import _Body
    m = env
    add_lab(m, "lab_h")
    m.store.x("UPDATE members SET ip='127.0.0.1' WHERE lab_id='lab_h'")
    _Body.body = b"hello"
    srv = socketserver.TCPServer(("127.0.0.1", 0), _Body)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = m.http("lab_h", {"member": "m1", "port": srv.server_address[1], "path": "/"}, "owner")
    finally:
        srv.shutdown()
        srv.server_close()
    assert out["status"] == 200 and out["body_head"] == "hello" and "truncated" not in out


def test_f2_the_screenshot_gives_its_browser_only_what_the_deadline_has_left(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_s")
    waits = []
    clock = {"t": 1000.0}
    monkeypatch.setattr(L.time, "monotonic", lambda: clock["t"])

    def create_container(image, **kw):
        c = FakeContainer(m.docker, kw.get("name"), kw.get("labels") or {}, image)

        def wait(timeout=None):
            waits.append(timeout)
            raise TimeoutError("read timed out")
        c.wait = wait
        m.docker._containers[c.name] = c
        return {"Id": c.id}
    monkeypatch.setattr(m.docker.api, "create_container", create_container)
    with pytest.raises(LabError):
        m._shoot("lab_s", {"name": "m1", "ip": "10.89.9.10"}, "10.89.9.127", 80, "/", 800, 600, {}, deadline=1020.0)
    assert waits == [20]                          # not a fresh 60 s
    clock["t"] = 1019.5
    with pytest.raises(LabError) as e:
        m._shoot("lab_s", {"name": "m1", "ip": "10.89.9.10"}, "10.89.9.127", 80, "/", 800, 600, {}, deadline=1020.0)
    assert e.value.code == "not_ready" and waits == [20]          # no time left: the browser is not waited for


# ================================================================== F6: a job's state
def _job(m, lab_id="lab_j", started="2026-11-10T10:00:00Z", pid_alive=b"yes\n", rc=b""):
    add_lab(m, lab_id)
    c = _member(m, lab_id)
    c.attrs["State"]["StartedAt"] = started
    log = []

    def exec_run(cmd, environment=None, **kw):
        s = cmd[-1] if isinstance(cmd, list) else str(cmd)
        log.append(s)
        if "kill -0" in s and "echo yes" in s:
            return SimpleNamespace(exit_code=0, output=pid_alive)
        if ".rc" in s and "cat" in s and "kill" not in s:
            return SimpleNamespace(exit_code=0, output=rc)
        return SimpleNamespace(exit_code=0, output=b"")
    c.exec_run = exec_run
    jid = m.start_job(lab_id, "m1", "sleep 1000", {})["job_id"]
    del log[:]
    return c, jid, log


def test_f6_an_rc_file_means_exited_whatever_the_pid_says(env):  # noqa: F811
    c, jid, _ = _job(env, rc=b"0\n", pid_alive=b"yes\n")           # the pid number is alive: someone else's
    j = env.job("lab_j", jid)
    assert j["state"] == "exited" and j["exit_code"] == 0


def test_f6_a_job_of_an_earlier_container_start_is_lost_and_its_pid_is_never_signalled(env):  # noqa: F811
    c, jid, log = _job(env, pid_alive=b"yes\n")
    assert env.job("lab_j", jid)["state"] == "running"             # same start: the pid is the job's
    c.attrs["State"]["StartedAt"] = "2026-11-10T11:30:00Z"         # the member was restarted
    del log[:]
    j = env.job("lab_j", jid, kill=True, origin="owner")
    assert j["state"] == "lost" and "killed" not in j
    assert not any("kill -TERM" in s or "kill -KILL" in s for s in log), log
    assert env.store.q("SELECT state FROM jobs WHERE job_id=?", (jid,))[0]["state"] == "lost"


def test_f6_a_kill_in_the_same_start_signals_the_group_and_says_killed(env):  # noqa: F811
    c, jid, log = _job(env, pid_alive=b"no\n")
    j = env.job("lab_j", jid, kill=True, origin="owner")
    assert j["state"] == "exited" and j["killed"] is True and any("kill -TERM -- -$p" in s for s in log)


def test_f6_a_kill_of_a_job_that_already_ended_signals_nothing(env):  # noqa: F811
    c, jid, log = _job(env, rc=b"3\n", pid_alive=b"yes\n")
    j = env.job("lab_j", jid, kill=True, origin="owner")
    assert j["state"] == "exited" and j["exit_code"] == 3 and "killed" not in j
    assert not any("kill -TERM" in s for s in log)


def test_f6_a_job_row_from_before_the_start_record_keeps_working(env):  # noqa: F811
    add_lab(env, "lab_j")
    c = _member(env, "lab_j")
    c.exec_run = lambda cmd, **kw: SimpleNamespace(exit_code=0, output=b"yes\n" if "kill -0" in cmd[-1] else b"")
    env.store.x("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)", ("job_old", "lab_j", "m1", "x", time.time(), None, "running", None))
    assert env.job("lab_j", "job_old")["state"] == "running"


def test_f6_the_entrypoint_clears_job_pid_files_and_keeps_logs_and_exit_codes(tmp_path):
    import subprocess
    jobs = tmp_path / "var/log/lab/jobs"
    jobs.mkdir(parents=True)
    for n in ("job_a.pid", "job_a.log", "job_a.rc"):
        (jobs / n).write_text("1")
    line = next(l for l in (ROOT / "images/base/lab-entrypoint").read_text().splitlines() if l.startswith("rm -f "))
    # the entrypoint's own clean-up line, run against a copy of the tree
    subprocess.run(["bash", "-c", line.replace("/var/log/", f"{tmp_path}/var/log/").replace("/run/", f"{tmp_path}/run/")],
                   check=True)
    assert sorted(p.name for p in jobs.iterdir()) == ["job_a.log", "job_a.rc"]


def test_f11_a_job_answer_names_its_log_file_and_says_when_the_tail_was_cut(env):  # noqa: F811
    add_lab(env, "lab_j")
    c = _member(env, "lab_j")
    big = b"z" * 5000

    def exec_run(cmd, **kw):
        s = cmd[-1]
        if "tail -c" in s:
            return SimpleNamespace(exit_code=0, output=big[-int(s.split("tail -c ")[1].split()[0]):])
        return SimpleNamespace(exit_code=0, output=b"0\n" if ".rc" in s else b"no\n")
    c.exec_run = exec_run
    env.store.x("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)", ("job_9", "lab_j", "m1", "x", time.time(), None, "running", None))
    j = env.job("lab_j", "job_9", tail_bytes=100)
    assert j["output_truncated"] is True and j["log_path"] == "/var/log/lab/jobs/job_9.log" and len(j["output_tail"]) == 100
    assert env.job("lab_j", "job_9", tail_bytes=60000)["output_truncated"] is False


# ================================================================== F7: the pump keeps a running total
def test_f7_many_tiny_chunks_are_pumped_in_linear_time(env):  # noqa: F811
    m = env
    add_lab(m, "lab_p")
    _member(m, "lab_p")
    m.cfg.output_cap_bytes = 1000
    n = 300000
    m.docker.api.exec_create = lambda cid, argv, **kw: {"Id": "x"}
    m.docker.api.exec_start = lambda i, stream=True, demux=True: ((b"x\n", None) for _ in range(n))
    m.docker.api.exec_inspect = lambda i: {"ExitCode": 0}
    t0 = time.monotonic()
    out = m._exec("lab_p", "m1", "yes", timeout_s=120)
    took = time.monotonic() - t0
    assert took < 5, took                                      # re-summing the buffer per chunk: minutes at this size
    assert out["stdout"] == "x\n" * 500 and out["stdout_truncated"] is True and out["exit_code"] == 0


def test_f7_the_pump_keeps_the_tail_of_a_long_stream_and_all_of_a_short_one(env):  # noqa: F811
    m = env
    add_lab(m, "lab_p")
    _member(m, "lab_p")
    m.cfg.output_cap_bytes = 10
    m.docker.api.exec_create = lambda cid, argv, **kw: {"Id": "x"}
    m.docker.api.exec_inspect = lambda i: {"ExitCode": 0}
    m.docker.api.exec_start = lambda i, stream=True, demux=True: iter([(b"abcdefgh", b"e1"), (b"ijklmnop", None),
                                                                      (b"qrstuvwx", None), (b"yz", b"e2")])
    out = m._exec("lab_p", "m1", "x", timeout_s=5)
    assert out["stdout"] == "qrstuvwxyz" and out["stdout_truncated"] is True
    assert out["stderr"] == "e1e2" and out["stderr_truncated"] is False


# ================================================================== F12: a kept lab's content is the owner's
@pytest.mark.parametrize("call", ["logs", "get", "http", "screenshot"])
def test_f12_unattended_work_reads_a_kept_lab_s_status_only(kept, env, monkeypatch, call):  # noqa: F811
    m = env
    make_kept(m, "lab_k")
    monkeypatch.setattr(m, "read_raw", lambda *a, **k: b"file")
    calls = {"logs": lambda o: m.logs("lab_k", "m1", 10, o),
             "get": lambda o: m.get_file("lab_k", "m1", "/app/x", o),
             "http": lambda o: m.http("lab_k", {"member": "m1", "port": 1, "path": "/"}, o, total_s=0.2),
             "screenshot": lambda o: m.screenshot("lab_k", {"member": "m1", "port": 1, "path": "/ bad"}, o)}
    for origin in ("autopilot", "probe"):                   # (a probe: only a lab a probe made)
        with pytest.raises(LabError) as e:
            calls[call](origin)
        assert e.value.code == "owner_only" and "status only" in e.value.message
    assert m.detail("lab_k")["lifetime"] == "kept"          # the status stays readable
    try:
        calls[call]("owner")                                # the owner gets past the guard
    except LabError as e:
        assert e.code != "owner_only"


def test_f12_a_disposable_lab_s_logs_stay_open_to_unattended_work(env):  # noqa: F811
    add_lab(env, "lab_d", origin="autopilot")
    _member(env, "lab_d")
    assert "log line" in env.logs("lab_d", "m1", 10, "autopilot")["logs"]


# ================================================================== F13: the host decides about the data
def _vol(m, name="ghost-labdata-p-chess-m1"):
    m.docker.volumes.create(name, labels={"ghost.labs": "1"})
    return name


def test_f13_drop_data_keeps_the_shared_volume_while_another_version_is_kept(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_v1")                                         # serves, holds the data
    make_kept(env, "lab_v2", mounted=False, state="staged")          # the next version, kept meanwhile
    vol = _vol(env)
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: pytest.fail("no final backup: the data stays"))
    out = kept.destroy_with_data("lab_v1", "owner")
    assert out["destroyed"] is True and not env.store.lab("lab_v1")
    assert out["data_volumes_removed"] == [] and vol in env.docker._volumes
    assert out["data_shared_with"] == ["lab_v2"] and "lab_v2" in out["note"]


def test_f13_the_last_version_takes_its_data_with_it(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_v1")
    make_kept(env, "lab_other", lineage="q")                         # another lineage: not a sharer
    make_kept(env, "lab_other2", name="blog")                        # another app of the lineage: not a sharer
    vol = _vol(env)
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: {"backup_id": "bk_f", "members": []})
    out = kept.destroy_with_data("lab_v1", "owner")
    assert out["data_volumes_removed"] == [vol] and "data_shared_with" not in out and out["final_backup"] == "bk_f"


def test_f13_a_fresh_data_version_owns_its_volume_and_drops_it(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_v1", data="fresh")
    make_kept(env, "lab_v2", mounted=False, state="staged")
    from ghost_labs.kept import data_volume_name
    own = _vol(env, data_volume_name("p", "chess", "m1", "lab_v1"))
    shared = _vol(env)
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: {"backup_id": "bk_f", "members": []})
    out = kept.destroy_with_data("lab_v1", "owner")
    assert out["data_volumes_removed"] == [own] and shared in env.docker._volumes


def test_f13_the_health_answer_names_the_guard():
    from ghost_labs.service import FEATURES
    assert {"drop_data_guard", "exec_free_lock"} <= set(FEATURES)


# ================================================================== F17: logs are bounded in bytes
def test_f17_one_huge_log_line_is_cut_to_the_output_cap(env):  # noqa: F811
    m = env
    add_lab(m, "lab_l")
    c = _member(m, "lab_l")
    m.cfg.output_cap_bytes = 2000
    c.logs = lambda tail=200, stream=False, follow=None: iter([b"A" * 700] * 5000) if stream else b"A" * 3_500_000
    cmds = []

    def exec_run(cmd, **kw):
        cmds.append(cmd[-1])
        return SimpleNamespace(exit_code=0, output=b"== /var/log/lab/app.log ==\n" + b"B" * 3_000_000)
    c.exec_run = exec_run
    out = m.logs("lab_l", "m1", 1, "owner")
    assert len(out["logs"]) < 2 * 2000 + 200 and out["truncated"] is True
    assert out["logs"].count("A") <= 2000 and out["logs"].count("B") <= 2000 and out["logs"].endswith("B")
    assert "tail -n 1" in cmds[0] and "| tail -c " in cmds[0]         # the member cuts each file too


def test_f17_short_logs_are_whole_and_secrets_are_redacted_across_the_cut(env):  # noqa: F811
    m = env
    add_lab(m, "lab_l")
    c = _member(m, "lab_l")
    out = m.logs("lab_l", "m1", 5, "owner")
    assert out["logs"].startswith("== container stdout/stderr ==\nlog line") and "truncated" not in out
    secret = "s3cr3t-token-abcdef"
    m.store.set_secret("p", "app", "TOK", secret)
    m.cfg.output_cap_bytes = 60
    c.logs = lambda tail=200, stream=False, follow=None: iter([("x" * 5000 + secret + "y" * 50).encode()])
    got = m.logs("lab_l", "m1", 5, "owner")["logs"]
    assert "<secret:TOK>" in got and secret[-10:] not in got


def test_f7_the_tail_buffer_keeps_a_running_total_and_stays_bounded():
    buf = L.TailBuf(10)
    for i in range(5000):
        buf.add(b"%04d" % i)
        assert buf.held == sum(len(c) for c in buf.chunks)          # the running total is the real one
        assert 0 < buf.held < 10 + 4 and len(buf.chunks) <= 4       # never more than keep + one chunk
    assert buf.total == 20000 and buf.data().endswith(b"49984999") and len(buf.data()) >= 10
    one = L.TailBuf(10)
    one.add(b"x" * 100)                                             # a single chunk is never dropped
    one.add(b"")
    assert one.data() == b"x" * 100 and one.held == 100


def test_bounded_tail_keeps_exactly_the_last_bytes():
    assert L._bounded_tail(iter([b"ab", b"cd", b"ef", b"gh"]), 3) == (b"fgh", True)
    assert L._bounded_tail(iter([b"ab", b"", b"c"]), 3) == (b"abc", False)
    assert L._bounded_tail(b"abcdef", 4) == (b"cdef", True)
    assert L._bounded_tail(iter([]), 4) == (b"", False)


# ================================================================== F18: a bare psql works
@pytest.mark.parametrize("tpl", ["pg-single", "patroni-3"])
def test_f18_pg_templates_make_psql_connect_as_postgres(env, monkeypatch, tpl):  # noqa: F811
    m = env
    monkeypatch.setattr(m, "wait_ready", lambda *a, **k: None)
    monkeypatch.setattr(m, "wait_tor", lambda *a, **k: None)
    out = m.create({"template": tpl, "owner": "conversation:c"}, "owner")
    for mem in out["members"]:
        c = m.docker._containers[m.container_name(out["lab_id"], mem["name"])]
        assert c.create_kwargs["environment"]["PGUSER"] == "postgres"


# ================================================================== G3: TTL rule + templates + loopback
def test_g3_create_clamps_the_idle_ttl_as_extend_does_and_says_so(env, monkeypatch):  # noqa: F811
    m = env
    monkeypatch.setattr(m, "wait_ready", lambda *a, **k: None)
    out = m.create({"template": "pg-single", "owner": "conversation:c", "ttl_idle_s": 90000}, "owner")
    assert out["clamped_to_s"] == 86400 and m.store.lab(out["lab_id"])["ttl_idle_s"] == 86400
    ok = m.create({"template": "pg-single", "owner": "conversation:c", "ttl_idle_s": 600}, "owner")
    assert "clamped_to_s" not in ok and m.store.lab(ok["lab_id"])["ttl_idle_s"] == 600
    for bad in (0, -5, "two hours"):
        with pytest.raises(LabError) as e:
            m.create({"template": "pg-single", "owner": "conversation:c", "ttl_idle_s": bad}, "owner")
        assert e.value.code == "bad_request" and e.value.extra["field"] == "ttl_idle_s"
    ext = m.extend(out["lab_id"], {"ttl_idle_s": 10 ** 8}, "owner")
    assert ext["clamped_to_s"] == 86400


def test_g3_the_python_app_template_says_where_the_app_must_listen(env):  # noqa: F811
    assert "0.0.0.0" in env.templates["python-app"]["description"]


def _closed_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.parametrize("inside_ok,want", [(0, True), (1, False)])
def test_g3_a_refused_probe_says_when_the_app_listens_on_loopback_only(env, inside_ok, want):  # noqa: F811
    m = env
    add_lab(m, "lab_h")
    m.store.x("UPDATE members SET ip='127.0.0.1' WHERE lab_id='lab_h'")
    c = _member(m, "lab_h")
    asked = []

    def exec_run(cmd, **kw):
        asked.append(cmd[-1])
        return SimpleNamespace(exit_code=inside_ok, output=b"")
    c.exec_run = exec_run
    port = _closed_port()
    out = m.http("lab_h", {"member": "m1", "port": port, "path": "/"}, "owner")
    assert out["status"] is None and "ConnectionRefused" in out["error"]
    assert L.http_probe("127.0.0.1", port, "GET", "/", "m1:80", 10)["refused"] is True
    assert bool(out.get("loopback_only")) is want and f"/dev/tcp/127.0.0.1/{port}" in asked[0]


# ================================================================== review of the §4UD fixes
def test_a_path_that_cannot_be_sent_is_a_clean_status_null_answer(env):  # noqa: F811
    add_lab(env, "lab_h")
    env.store.x("UPDATE members SET ip='127.0.0.1' WHERE lab_id='lab_h'")
    srv = _serve(_Garbage)
    try:
        out = env.http("lab_h", {"member": "m1", "port": srv.server_address[1], "path": "/caf\u00e9"}, "owner")
    finally:
        srv.shutdown()
        srv.server_close()
    assert out["status"] is None and "UnicodeEncodeError" in out["error"]


def test_a_cut_file_log_is_marked_truncated(env):  # noqa: F811
    m = env
    add_lab(m, "lab_l")
    c = _member(m, "lab_l")
    m.cfg.output_cap_bytes = 2000

    def exec_run(cmd, **kw):                       # the member's own `tail -c` cut it to what was asked
        n = int(cmd[-1].rsplit("tail -c ", 1)[1].split()[0])
        return SimpleNamespace(exit_code=0, output=(b"B" * 10000)[-n:])
    c.exec_run = exec_run
    assert m.logs("lab_l", "m1", 5, "owner")["truncated"] is True
    c.exec_run = lambda cmd, **kw: SimpleNamespace(exit_code=0, output=b"short")
    assert "truncated" not in m.logs("lab_l", "m1", 5, "owner")


def test_the_reaper_does_not_retire_a_kept_version_under_a_running_command(running):
    m, ex = running
    m.store.update_lab("lab_x", lifetime="kept", retire_at=time.time() - 5)
    assert m.reap_once() == [] and m.store.lab("lab_x")
    ex.end()
    assert m.reap_once() == [{"lab_id": "lab_x", "reason": "retired"}]


def _api(m, kept=None):
    from fastapi.testclient import TestClient
    from ghost_labs.api import create_app
    from .test_api import Svc
    svc = Svc(m)
    svc.kept = kept
    return TestClient(create_app(svc, "k" * 64))


def _h(origin=None):
    return {"X-Labs-Key": "k" * 64, **({"X-Labs-Origin": origin} if origin else {})}


@pytest.mark.parametrize("lab_origin,caller,status", [
    ("owner", "owner", 200), ("owner", "autopilot", 403), ("owner", "probe", 403), ("owner", None, 403),
    ("probe", "probe", 200), ("probe", "autopilot", 403), ("probe", "owner", 200)])
def test_r2_a_job_s_answer_on_a_kept_lab_is_refused_to_a_caller_that_may_not_touch_it(kept, env, lab_origin, caller, status):  # noqa: F811
    """Through the real app: the route must pass WHO reads (a job answer carries the output)."""
    make_kept(env, "lab_k")
    env.store.update_lab("lab_k", origin=lab_origin)
    c = env.docker._containers[env.container_name("lab_k", "m1")]
    c.exec_run = lambda cmd, **kw: SimpleNamespace(exit_code=0, output=b"0\n" if ".rc" in cmd[-1] else b"SECRET OUT")
    env.store.x("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)", ("job_k", "lab_k", "m1", "x", time.time(), None, "running", None))
    r = _api(env).get("/v1/labs/lab_k/jobs/job_k", headers=_h(caller))
    assert r.status_code == status
    if status == 200:
        assert "SECRET OUT" in r.json()["output_tail"] and "output_withheld" not in r.json()
    else:
        assert r.json()["error"] == "owner_only" and "SECRET" not in r.text
    assert _api(env).get("/v1/labs/lab_k", headers=_h(caller)).status_code == 200      # the status stays readable


def test_r2_a_disposable_lab_s_job_is_read_by_unattended_work(env):  # noqa: F811
    add_lab(env, "lab_j", origin="autopilot")
    c = _member(env, "lab_j")
    c.exec_run = lambda cmd, **kw: SimpleNamespace(exit_code=0, output=b"0\n" if ".rc" in cmd[-1] else b"out")
    env.store.x("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)", ("job_j", "lab_j", "m1", "x", time.time(), None, "running", None))
    r = _api(env).get("/v1/labs/lab_j/jobs/job_j", headers=_h("autopilot"))
    assert r.status_code == 200 and r.json()["output_tail"] == "out"
    assert r.json()["output_truncated"] is False and r.json()["log_path"] == "/var/log/lab/jobs/job_j.log"


def test_r2_the_kept_guard_is_read_again_after_the_exec_waited_for_the_lock(env, monkeypatch):  # noqa: F811
    """Real threads: the lab is KEPT while an autopilot exec waits behind a state change."""
    m = env
    monkeypatch.setattr(L, "LOCK_WAIT_S", 5.0)
    add_lab(m, "lab_w", origin="autopilot", owner="project:p")
    _member(m, "lab_w")
    ran = []
    m.docker.api.exec_create = lambda cid, argv, **kw: ran.append(argv) or {"Id": "x"}
    m.docker.api.exec_start = lambda i, stream=True, demux=True: iter([(b"ok", None)])
    m.docker.api.exec_inspect = lambda i: {"ExitCode": 0}
    lk = m.lock("lab_w")
    assert lk.acquire(timeout=1)                        # a /keep in progress
    res = {}

    def go():
        try:
            res["out"] = m.exec("lab_w", {"member": "m1", "cmd": "rm -rf /data"}, "autopilot")
        except LabError as e:
            res["err"] = e
    t = threading.Thread(target=go, daemon=True)
    t.start()
    time.sleep(0.2)                                     # the exec passed the first guard and waits for the lock
    m.store.update_lab("lab_w", lifetime="kept")
    lk.release()
    t.join(5)
    assert res.get("err") is not None and res["err"].code == "owner_only" and ran == []
    assert lk.locked() is False
    # … and a lab destroyed while the exec waited is unknown_lab, not "no member"
    add_lab(m, "lab_v")
    _member(m, "lab_v")
    lk2 = m.lock("lab_v")
    assert lk2.acquire(timeout=1)
    res.clear()
    t = threading.Thread(target=lambda: res.update(e=_err(lambda: m.exec("lab_v", {"member": "m1", "cmd": "x"}, "owner"))),
                         daemon=True)
    t.start()
    time.sleep(0.2)
    m.store.delete_lab("lab_v")
    m.store.remember_destroyed("lab_v", "deleted", "owner", "")
    lk2.release()
    t.join(5)
    assert res["e"].code == "unknown_lab" and "destroyed" in res["e"].message


def _err(fn):
    try:
        fn()
    except LabError as e:
        return e
    return None


def test_r2_a_detached_job_started_on_a_lab_that_is_being_destroyed_is_not_started(env):  # noqa: F811
    m = env
    add_lab(m, "lab_d")
    c = _member(m, "lab_d")

    def exec_run(cmd, environment=None, **kw):          # the destroy lands while the job script runs
        m.store.delete_lab("lab_d")
        m.store.remember_destroyed("lab_d", "deleted", "owner", "")
        return SimpleNamespace(exit_code=0, output=b"")
    c.exec_run = exec_run
    with pytest.raises(LabError) as e:
        m.start_job("lab_d", "m1", "sleep 1", {})
    assert e.value.code == "unknown_lab"
    assert m.store.q("SELECT * FROM jobs") == [] and m.store.q("SELECT * FROM job_starts") == []


def test_r2_a_destroy_between_the_check_and_the_inserts_leaves_no_job_row(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_d")
    _member(m, "lab_d")
    real = m.store.x

    def x(sql, args=()):
        real(sql, args)
        if sql.startswith("INSERT OR REPLACE INTO job_starts"):
            m.store.delete_lab("lab_d2")                # (nothing) — then the real destroy of THIS lab's row only
            real("DELETE FROM labs WHERE lab_id=?", ("lab_d",))
    monkeypatch.setattr(m.store, "x", x)
    with pytest.raises(LabError) as e:
        m.start_job("lab_d", "m1", "sleep 1", {})
    assert e.value.code == "unknown_lab"
    assert m.store.q("SELECT * FROM jobs") == [] and m.store.q("SELECT * FROM job_starts") == []


def test_r2_deleting_a_lab_deletes_its_job_start_records(env):  # noqa: F811
    m = env
    add_lab(m, "lab_d")
    add_lab(m, "lab_e")
    for lab in ("lab_d", "lab_e"):
        _member(m, lab)
        m.start_job(lab, "m1", "sleep 1", {})
    assert len(m.store.q("SELECT * FROM job_starts")) == 2
    m.destroy("lab_d", origin="owner")
    assert [r["lab_id"] for r in m.store.q("SELECT lab_id FROM job_starts")] == ["lab_e"]
    assert [r["lab_id"] for r in m.store.q("SELECT lab_id FROM jobs")] == ["lab_e"]


def test_r2_clamp_idle_at_the_boundary():
    assert L.clamp_idle({"ttl_idle_s": 100}, 5, 100) == (100, False)          # AT the maximum: not clamped
    assert L.clamp_idle({"ttl_idle_s": 101}, 5, 100) == (100, True)
    assert L.clamp_idle({"ttl_idle_s": 99}, 5, 100) == (99, False)
    assert L.clamp_idle({}, 5, 100) == (5, False)


def test_r2_only_a_refused_connection_asks_the_member_about_loopback(env):  # noqa: F811
    m = env
    add_lab(m, "lab_h")
    m.store.x("UPDATE members SET ip='127.0.0.1' WHERE lab_id='lab_h'")
    c = _member(m, "lab_h")
    asked = []
    c.exec_run = lambda cmd, **kw: asked.append(cmd) or SimpleNamespace(exit_code=0, output=b"")
    srv = _serve(_Garbage)                              # something DOES listen on the lab address: a bad answer
    try:
        out = m.http("lab_h", {"member": "m1", "port": srv.server_address[1], "path": "/"}, "owner")
    finally:
        srv.shutdown()
        srv.server_close()
    assert out["status"] is None and "loopback_only" not in out and asked == []


def test_r2_an_exec_whose_exit_code_cannot_be_read_under_a_destroy_still_says_destroyed(running):
    m, ex = running

    def gone(i):
        raise RuntimeError("No such exec instance")
    m.docker.api.exec_inspect = gone
    m.destroy("lab_x", by="autopilot", origin="autopilot")
    ex.end()
    assert ex.error is not None and ex.error.code == "unknown_lab"


def test_r2_the_real_health_answer_names_the_features_and_the_summary_the_latest_end(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from ghost_labs import config as C
    from ghost_labs.api import create_app
    from ghost_labs.service import Service
    from .fakes import FakeDocker, runner_ok
    monkeypatch.setattr(C, "meminfo", lambda: {"MemTotal": 8000, "MemAvailable": 7000})
    monkeypatch.setattr(C, "CONFIG_DIR", tmp_path / "etc")
    svc = Service(C.Config(ram_budget_mb=4000, kept_mb=1500), FakeDocker(), ROOT / "templates", tmp_path / "state",
                  runner=runner_ok(), start_threads=False)
    cl = TestClient(create_app(svc, "k" * 64))
    h = cl.get("/v1/health", headers=_h("owner")).json()
    assert {"drop_data_guard", "exec_free_lock", "job_env"} <= set(h["features"])
    m = svc.mgr
    add_lab(m, "lab_d", ttl_max=86400)
    add_lab(m, "lab_k", lifetime="kept", ttl_idle=0, ttl_max=0)
    d = cl.get("/v1/labs/lab_d", headers=_h("owner")).json()
    assert d["max_expires_at"] == pytest.approx(m.store.lab("lab_d")["created_at"] + 86400)
    assert cl.get("/v1/labs/lab_k", headers=_h("owner")).json()["max_expires_at"] is None      # kept: no end
    listed = {x["lab_id"]: x for x in cl.get("/v1/labs", headers=_h("owner")).json()["labs"]}
    assert listed["lab_k"]["max_expires_at"] is None and listed["lab_d"]["max_expires_at"] > time.time()


# ================================================================== round 3
@pytest.mark.parametrize("op", ["kill", "stop", "restart", "isolate"])
def test_r3_a_member_op_reads_the_kept_guard_again_after_it_waited_for_the_lock(env, monkeypatch, op):  # noqa: F811
    """Real threads: the owner's /keep holds the lock; an autopilot member op passed the guard on the still
    disposable row and waits; the lab is KEPT; the op must be refused, the member untouched."""
    m = env
    monkeypatch.setattr(L, "LOCK_WAIT_S", 5.0)
    add_lab(m, "lab_w", origin="autopilot", owner="project:p")
    c = _member(m, "lab_w")
    m.docker.networks.create("ghost-lab-lab_w", labels={})
    lk = m.lock("lab_w")
    assert lk.acquire(timeout=1)
    res = {}
    t = threading.Thread(target=lambda: res.update(e=_err(lambda: m.member_action("lab_w", "m1", op, "autopilot"))),
                         daemon=True)
    t.start()
    time.sleep(0.2)
    m.store.update_lab("lab_w", lifetime="kept")
    lk.release()
    t.join(5)
    assert res["e"] is not None and res["e"].code == "owner_only"
    assert c.status == "running" and not m.member("lab_w", "m1")["isolated"] and lk.locked() is False
    assert m.member_action("lab_w", "m1", op, "owner")["action"] == op          # the owner's own op goes ahead
    assert lk.locked() is False


def test_r3_a_screenshot_reads_the_kept_guard_again_after_it_waited_for_the_lock(env, monkeypatch):  # noqa: F811
    m = env
    monkeypatch.setattr(L, "LOCK_WAIT_S", 5.0)
    add_lab(m, "lab_w", origin="autopilot", owner="project:p")
    _member(m, "lab_w")
    monkeypatch.setattr(m, "http", lambda *a, **k: {"status": 200})
    shot = []
    monkeypatch.setattr(m, "_shoot", lambda *a, **k: shot.append(1) or {"png_b64": ""})
    lk = m.lock("lab_w")
    assert lk.acquire(timeout=1)
    res = {}
    t = threading.Thread(target=lambda: res.update(e=_err(
        lambda: m.screenshot("lab_w", {"member": "m1", "port": 80, "path": "/"}, "autopilot"))), daemon=True)
    t.start()
    time.sleep(0.2)
    m.store.update_lab("lab_w", lifetime="kept")
    lk.release()
    t.join(5)
    assert res["e"] is not None and res["e"].code == "owner_only" and shot == [] and lk.locked() is False
    assert m.screenshot("lab_w", {"member": "m1", "port": 80, "path": "/"}, "owner") == {"png_b64": ""}
    assert lk.locked() is False


# ================================================================== round 3 leftovers
def test_r3_an_exit_code_that_cannot_be_read_while_a_destroy_is_under_way_answers_unknown_lab(env, monkeypatch):  # noqa: F811
    """The destroy has removed the container and still HOLDS the lock (its row goes last): the exec's failing
    exit-code read must wait for it and answer unknown_lab — it answered 500 internal while the row was there."""
    m = env
    monkeypatch.setattr(L, "LOCK_WAIT_S", 5.0)
    add_lab(m, "lab_x", origin="autopilot", owner_use=0)
    _member(m, "lab_x")
    ex = _RunningExec(m, "lab_x")
    m.docker.api.exec_inspect = lambda i: (_ for _ in ()).throw(RuntimeError("No such exec instance"))
    lk = m.lock("lab_x")
    assert lk.acquire(timeout=1, over_exec=True)             # the destroy's state change, in progress
    ex.release.set()                                         # the command's stream ends (its container went)
    time.sleep(0.3)
    assert ex.t.is_alive()                                   # it waits for the state change, it does not answer 500
    m.store.delete_lab("lab_x")
    m.store.remember_destroyed("lab_x", "deleted", "owner", "")
    lk.release()
    ex.t.join(5)
    assert ex.error is not None and ex.error.code == "unknown_lab" and m.exec_running("lab_x") is False


def test_r3_an_exec_that_fails_on_a_lab_that_stays_is_still_its_own_error(env):  # noqa: F811
    m = env
    add_lab(m, "lab_y")
    _member(m, "lab_y")
    m.docker.api.exec_create = lambda cid, argv, **kw: {"Id": "x"}
    m.docker.api.exec_start = lambda i, stream=True, demux=True: iter([(b"ok", None)])
    m.docker.api.exec_inspect = lambda i: (_ for _ in ()).throw(RuntimeError("docker is down"))
    with pytest.raises(RuntimeError):
        m.exec("lab_y", {"member": "m1", "cmd": "true"}, "owner")
    assert m.lock("lab_y").locked() is False and m.store.lab("lab_y")


@pytest.mark.parametrize("origin,status", [("autopilot", 403), (None, 403), ("probe", 403), ("owner", 202)])
def test_r3_a_detached_exec_on_a_kept_lab_is_guarded_through_the_app(kept, env, origin, status):  # noqa: F811
    make_kept(env, "lab_k")
    c = env.docker._containers[env.container_name("lab_k", "m1")]
    r = _api(env).post("/v1/labs/lab_k/exec", json={"member": "m1", "cmd": "rm -rf /data", "detach": True},
                       headers=_h(origin))
    assert r.status_code == status
    started = [x for x in c.exec_log if "setsid" in " ".join(map(str, x))]
    assert (len(started) == 1) is (status == 202) and (len(env.store.q("SELECT * FROM jobs")) == 1) is (status == 202)
    assert env.lock("lab_k").locked() is False               # never held while the job runs


def test_r3_a_detached_start_reads_the_kept_guard_again_after_it_waited_for_the_lock(env, monkeypatch):  # noqa: F811
    m = env
    monkeypatch.setattr(L, "LOCK_WAIT_S", 5.0)
    add_lab(m, "lab_w", origin="autopilot", owner="project:p")
    c = _member(m, "lab_w")
    lk = m.lock("lab_w")
    assert lk.acquire(timeout=1)                             # the owner's /keep
    res = {}
    t = threading.Thread(target=lambda: res.update(e=_err(
        lambda: m.exec("lab_w", {"member": "m1", "cmd": "x", "detach": True}, "autopilot"))), daemon=True)
    t.start()
    time.sleep(0.2)
    m.store.update_lab("lab_w", lifetime="kept")
    lk.release()
    t.join(5)
    assert res["e"] is not None and res["e"].code == "owner_only" and c.exec_log == []
    assert m.store.q("SELECT * FROM jobs") == [] and lk.locked() is False


def test_r3_a_detached_start_does_not_wait_for_a_running_command(running):
    m, ex = running
    out = m.exec("lab_x", {"member": "m1", "cmd": "sleep 5", "detach": True}, "autopilot")
    assert out["job_id"].startswith("job_")
    ex.end()


# ================================================================== F-A: per-member readiness
def _ready_lab(m, monkeypatch, script, timeout_s=9, members=("pg1", "pg2")):
    """A lab whose spec declares a ready check on every member; `script(member, cmd, n)` -> exit code."""
    add_lab(m, "lab_r", state="creating")
    m.store.x("DELETE FROM members WHERE lab_id='lab_r'")
    spec = {"members": [], "egress": "none", "readiness": {"member": members[0], "cmd": "tpl-check", "timeout_s": timeout_s}}
    for i, name in enumerate(members):
        m.store.put_member({"lab_id": "lab_r", "name": name, "ip": f"10.89.9.{10 + i}", "image": "ghost-lab/pg:18",
                            "mem_mb": 256, "cpus": 1.0, "cap_add": "[]", "ports_json": "[]", "isolated": 0})
        spec["members"].append({"name": name, "image": "ghost-lab/pg:18", "mem_mb": 256, "cpus": 1.0, "ports": [],
                                "cap_add": [], "ready": "pg_isready -q"})
    m.store.update_lab("lab_r", spec_json=dumps(spec))
    calls, clock = [], {"t": 1000.0}
    monkeypatch.setattr(L.time, "time", lambda: clock["t"])
    monkeypatch.setattr(L.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))

    def _exec(lab_id, member, cmd, timeout_s=0, **kw):
        calls.append((member, cmd, timeout_s))
        n = sum(1 for c in calls if c[:2] == (member, cmd))
        return {"exit_code": script(member, cmd, n), "stdout": "", "stderr": ""}
    monkeypatch.setattr(m, "_exec", _exec)
    return spec, calls, clock


def test_fa_every_member_is_polled_under_one_deadline_and_the_template_check_decides_the_state(env, monkeypatch):  # noqa: F811
    m = env
    # the template check passes at once; pg1's socket on its 2nd look; pg2 never
    spec, calls, clock = _ready_lab(m, monkeypatch, lambda mem, cmd, n: 0 if cmd == "tpl-check" else
                                    (0 if (mem == "pg1" and n >= 2) else 1))
    m.wait_ready("lab_r", spec)
    row = m.store.lab("lab_r")
    assert row["state"] == "ready"                                   # a member not ready never fails the create
    d = m.detail("lab_r")
    assert d["readiness"]["passed"] is True and d["readiness"]["members"] == {"pg1": True, "pg2": False}
    by = {x["name"]: x for x in d["members"]}
    assert by["pg1"]["ready"] is True and by["pg2"]["ready"] is False and by["pg2"]["ready_cmd"] == "pg_isready -q"
    assert clock["t"] - 1000.0 <= 9 + 3                              # ONE total wait: the template's timeout
    assert [c for c in calls if c[1] == "tpl-check"] == [("pg1", "tpl-check", 30)]       # not re-run once passed
    assert len([c for c in calls if c[0] == "pg1" and c[1] != "tpl-check"]) == 2         # a ready member is left alone
    assert all(1 <= c[2] <= 9 for c in calls if c[1] != "tpl-check")                     # each check inside the rest


def test_fa_all_ready_ends_the_wait_at_once_and_a_failing_template_check_is_still_unready(env, monkeypatch):  # noqa: F811
    m = env
    spec, calls, clock = _ready_lab(m, monkeypatch, lambda mem, cmd, n: 0)
    m.wait_ready("lab_r", spec)
    assert clock["t"] == 1000.0 and m.store.lab("lab_r")["state"] == "ready"
    assert m.detail("lab_r")["readiness"]["members"] == {"pg1": True, "pg2": True}
    m.store.delete_lab("lab_r")
    spec, calls, clock = _ready_lab(m, monkeypatch, lambda mem, cmd, n: 1 if cmd == "tpl-check" else 0)
    m.wait_ready("lab_r", spec)
    # the template check never passed: no member check is RUN, and none is reported as "not ready" (null)
    assert m.store.lab("lab_r")["state"] == "unready" and m.detail("lab_r")["readiness"]["members"] == {"pg1": None, "pg2": None}
    assert {c[1] for c in calls} == {"tpl-check"}
    assert [x["ready"] for x in m.detail("lab_r")["members"]] == [None, None]


def test_fa_a_member_check_that_raises_is_not_ready_and_a_lab_without_checks_is_as_before(env, monkeypatch):  # noqa: F811
    m = env

    def script(mem, cmd, n):
        if cmd != "tpl-check":
            raise LabError("not_ready", "member is restarting")
        return 0
    spec, calls, clock = _ready_lab(m, monkeypatch, script, members=("pg1",))
    m.wait_ready("lab_r", spec)
    assert m.store.lab("lab_r")["state"] == "ready" and m.detail("lab_r")["members"][0]["ready"] is False
    m.store.delete_lab("lab_r")
    spec, calls, clock = _ready_lab(m, monkeypatch, lambda mem, cmd, n: 0, members=("pg1",))
    for mm in spec["members"]:
        mm.pop("ready")
    m.store.update_lab("lab_r", spec_json=dumps(spec))
    m.wait_ready("lab_r", spec)
    d = m.detail("lab_r")
    assert "members" not in d["readiness"] and "ready" not in d["members"][0] and len(calls) == 1


def test_fa_the_shipped_templates_declare_their_member_checks_and_the_recipe_keeps_them(env):  # noqa: F811
    t = env.templates
    assert [x.get("ready") for x in t["pg-single"]["members"]] == ["pg_isready -q"]
    assert all(x.get("ready") == "pg_isready -q" for x in t["patroni-3"]["members"])
    assert "mysqladmin ping" in t["mysql-single"]["members"][0]["ready"]
    for name in ("web", "python-app", "blank-dev"):
        assert all("ready" not in x for x in t[name]["members"])
    from ghost_labs import templates as T
    with pytest.raises(LabError):
        T.parse({"name": "x", "members": [{"name": "a", "image": "i", "mem_mb": 64, "ready": "a\nb"}]})
    add_lab(env, "lab_a")
    env.store.update_lab("lab_a", spec_json=dumps({"members": [{"name": "a", "image": "i", "mem_mb": 64, "cpus": 1.0,
                                                                "ports": [], "cap_add": [], "ready": "true"}],
                                                   "egress": "none", "readiness": {"member": "a", "cmd": "true"}}))
    assert env.spec("lab_a")["create"]["members"][0]["ready"] == "true"


# ================================================================== F-B: cancel a running foreground exec
class _Cmd:
    """A foreground exec through the real Manager.exec whose stream ends when its group is killed (or `end`)."""

    def __init__(self, m, lab_id, origin="owner", tag="", member="m1", timeout_s=900):
        self.m, self.done, self.started = m, threading.Event(), threading.Event()
        self.result = self.error = None
        self.c = m.docker._containers[m.container_name(lab_id, member)]
        self.kills = []
        real = self.c.exec_run

        def exec_run(cmd, **kw):
            line = cmd[-1] if isinstance(cmd, list) else str(cmd)
            if "kill -KILL" in line:
                self.kills.append(line)
                self.done.set()
            return real(cmd, **kw)
        self.c.exec_run = exec_run
        me = self

        def exec_start(i, stream=True, demux=True):
            me.started.set()
            me.done.wait(20)
            yield (b"partial output\n", None)
        m.docker.api.exec_create = lambda cid, argv, **kw: {"Id": "x1"}
        m.docker.api.exec_start = exec_start
        m.docker.api.exec_inspect = lambda i: {"ExitCode": 0}

        def go():
            try:
                self.result = m.exec(lab_id, {"member": member, "cmd": "sleep 900", "timeout_s": timeout_s,
                                              **({"tag": tag} if tag else {})}, origin)
            except LabError as e:
                self.error = e
        self.t = threading.Thread(target=go, daemon=True)
        self.t.start()
        assert self.started.wait(5)

    def end(self):
        self.done.set()
        self.t.join(10)
        assert not self.t.is_alive()


@pytest.mark.parametrize("started_by,tag,caller,caller_tag,ends", [
    ("owner", "", "owner", "", True),
    ("autopilot", "#leaf-aaaaaaaaaa", "owner", "", True),                    # the owner ends anything on the lab
    ("autopilot", "#leaf-aaaaaaaaaa", "autopilot", "#leaf-aaaaaaaaaa", True),   # the build that started it
    ("autopilot", "#leaf-aaaaaaaaaa", "autopilot", "#leaf-bbbbbbbbbb", False),  # another build
    ("autopilot", "#leaf-aaaaaaaaaa", "autopilot", "", False),               # unattended work with no build
    ("autopilot", "", "autopilot", "", False),                               # an untagged command is nobody's
    ("owner", "", "autopilot", "#leaf-aaaaaaaaaa", False),                   # never the owner's command
    ("owner", "#leaf-aaaaaaaaaa", "autopilot", "#leaf-aaaaaaaaaa", False),   # … whatever tag it was given
    ("probe", "", "probe", "", True),
    ("owner", "", "probe", "", False),
    ("probe", "", "autopilot", "#leaf-aaaaaaaaaa", False),
])
def test_fb_who_may_cancel_a_running_command(env, started_by, tag, caller, caller_tag, ends):  # noqa: F811
    m = env
    add_lab(m, "lab_c", origin="autopilot", owner="project:p")
    _member(m, "lab_c")
    cmd = _Cmd(m, "lab_c", origin=started_by, tag=tag)
    cl = _api(m)
    r = cl.post("/v1/labs/lab_c/exec/cancel", json=({"tag": caller_tag} if caller_tag else {}), headers=_h(caller))
    if ends:
        assert r.status_code == 200 and r.json()["running"] == 1
        assert [x["member"] for x in r.json()["cancelled"]] == ["m1"] and "sleep 900" in r.json()["cancelled"][0]["cmd"]
        cmd.t.join(10)
        assert not cmd.t.is_alive() and cmd.error is None
        assert cmd.result["cancelled"] is True and cmd.result["exit_code"] is None and cmd.result["timed_out"] is False
        assert "partial output" in cmd.result["stdout"]
        assert len(cmd.kills) == 1 and "kill -KILL -- -$p" in cmd.kills[0]                         # the GROUP
        assert "f=/run/ghost-exec/" in cmd.kills[0] and "p=$(cat $f.pid" in cmd.kills[0]           # by its pid file
        assert "kill -0 -- -$p" in cmd.kills[0]                                                     # … and a look
        assert cmd.result["killed"] is True and r.json()["cancelled"][0]["outcome"] == "killed"
    else:
        assert r.status_code == 403 and r.json()["error"] == "owner_only"
        time.sleep(0.3)
        assert cmd.t.is_alive() and cmd.kills == []
        cmd.end()
        assert cmd.result["exit_code"] == 0 and "cancelled" not in cmd.result
    assert m.exec_running("lab_c") is False and m._fg == {}


def test_fb_nothing_running_and_a_member_filter(env):  # noqa: F811
    m = env
    add_lab(m, "lab_c")
    _member(m, "lab_c")
    m.store.put_member({"lab_id": "lab_c", "name": "m2", "ip": "10.89.9.11", "image": "i", "mem_mb": 64, "cpus": 1.0,
                        "cap_add": "[]", "ports_json": "[]", "isolated": 0})
    cl = _api(m)
    r = cl.post("/v1/labs/lab_c/exec/cancel", json={}, headers=_h("owner"))
    assert r.status_code == 200 and r.json() == {"lab_id": "lab_c", "running": 0, "cancelled": []}
    cmd = _Cmd(m, "lab_c")
    other = cl.post("/v1/labs/lab_c/exec/cancel", json={"member": "m2"}, headers=_h("owner"))
    assert other.json()["cancelled"] == [] and other.json()["running"] == 0 and cmd.t.is_alive()
    assert cl.post("/v1/labs/lab_c/exec/cancel", json={"member": "nope"}, headers=_h("owner")).status_code == 404
    assert cl.post("/v1/labs/lab_nope/exec/cancel", json={}, headers=_h("owner")).json()["error"] == "unknown_lab"
    assert cl.post("/v1/labs/lab_c/exec/cancel", json={"member": "m1"}, headers=_h("owner")).json()["running"] == 1
    cmd.t.join(10)
    assert cmd.result["cancelled"] is True


def test_fb_autopilot_never_cancels_on_a_kept_lab_and_the_owner_does(kept, env):  # noqa: F811
    make_kept(env, "lab_k")
    cmd = _Cmd(env, "lab_k", origin="owner")
    cl = _api(env)
    r = cl.post("/v1/labs/lab_k/exec/cancel", json={"tag": "#leaf-aaaaaaaaaa"}, headers=_h("autopilot"))
    assert r.status_code == 403 and r.json()["error"] == "owner_only" and cmd.kills == []
    assert cl.post("/v1/labs/lab_k/exec/cancel", json={}, headers=_h("owner")).json()["running"] == 1
    cmd.t.join(10)
    assert cmd.result["cancelled"] is True
    # a build's own command on a lab that was KEPT while it ran: the build no longer ends anything there
    add_lab(env, "lab_b", origin="autopilot", owner="project:p")
    _member(env, "lab_b")
    mine = _Cmd(env, "lab_b", origin="autopilot", tag="#leaf-aaaaaaaaaa")
    env.store.update_lab("lab_b", lifetime="kept")
    r = cl.post("/v1/labs/lab_b/exec/cancel", json={"tag": "#leaf-aaaaaaaaaa"}, headers=_h("autopilot"))
    assert r.status_code == 403 and mine.kills == [] and mine.t.is_alive()
    mine.end()


def test_fb_a_cancel_after_the_member_restarted_signals_no_pid(env):  # noqa: F811
    m = env
    add_lab(m, "lab_c")
    c = _member(m, "lab_c")
    c.attrs["State"]["StartedAt"] = "2026-11-11T10:00:00Z"
    cmd = _Cmd(m, "lab_c")
    c.attrs["State"]["StartedAt"] = "2026-11-11T10:05:00Z"          # restarted under the command
    assert m.cancel_exec("lab_c", {}, "owner")["running"] == 1
    time.sleep(0.5)
    assert cmd.kills == []                                          # the pid file's number is nobody's now
    cmd.t.join(10)
    # … so nothing was killed, the stream is still open, and that is what the answer says
    assert cmd.result["cancelled"] is True and cmd.result["exit_code"] is None and cmd.result["killed"] is False
    assert m.exec_running("lab_c") is True and len(m._fg["lab_c"]) == 1      # the mark stays while it runs
    cmd.done.set()
    for _ in range(50):
        if not m.exec_running("lab_c"):
            break
        time.sleep(0.1)
    assert m.exec_running("lab_c") is False and m._fg == {}


def test_fb_a_timeout_still_kills_the_group_and_is_not_a_cancel(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_c")
    _member(m, "lab_c")
    cmd = _Cmd(m, "lab_c", timeout_s=1)
    cmd.t.join(10)
    assert cmd.result["timed_out"] is True and "cancelled" not in cmd.result and len(cmd.kills) == 1
    assert cmd.result["killed"] is True


def test_fb_the_health_answer_names_cancel():
    from ghost_labs.service import FEATURES
    assert {"exec_cancel", "member_ready"} <= set(FEATURES)


# ================================================================== F-C: folders
import base64 as _b64  # noqa: E402
import io as _io  # noqa: E402
import tarfile as _tarfile  # noqa: E402


def _tar(entries, mode="w:gz"):
    """entries: [(name, bytes | None for a dir | ("link", target) | ("hard", target) | ("fifo",) | ("dev",))]"""
    buf = _io.BytesIO()
    with _tarfile.open(fileobj=buf, mode=mode) as t:
        for name, v in entries:
            ti = _tarfile.TarInfo(name)
            if v is None:
                ti.type, ti.mode = _tarfile.DIRTYPE, 0o755
                t.addfile(ti)
            elif isinstance(v, tuple):
                ti.type = {"link": _tarfile.SYMTYPE, "hard": _tarfile.LNKTYPE, "fifo": _tarfile.FIFOTYPE,
                           "dev": _tarfile.CHRTYPE}[v[0]]
                ti.linkname = v[1] if len(v) > 1 else ""
                t.addfile(ti)
            else:
                ti.size, ti.mode = len(v), 0o4755
                t.addfile(ti, _io.BytesIO(v))
    return buf.getvalue()


def _names(tar_bytes):
    with _tarfile.open(fileobj=_io.BytesIO(tar_bytes)) as t:
        return {m.name: (m.type, m.mode, m.uid, (t.extractfile(m).read() if m.isfile() else None)) for m in t}


def _put(m, entries, lab="lab_t", origin="owner", path="/srv/app"):
    return m.put_tree(lab, {"member": "m1", "path": path, "archive_b64": _b64.b64encode(_tar(entries)).decode()}, origin)


def test_fc_a_folder_put_is_rebuilt_from_checked_entries(env):  # noqa: F811
    m = env
    add_lab(m, "lab_t")
    c = _member(m, "lab_t")
    got = {}
    c.put_archive = lambda path, data: got.update(path=path, data=data) or True
    out = _put(m, [("./a.txt", b"A"), ("sub", None), ("sub/b.bin", b"BB"), ("sub/./c", b"")])
    assert out == {"lab_id": "lab_t", "member": "m1", "path": "/srv/app", "entries": 4, "bytes": 3}
    assert got["path"] == "/srv/app" and ["mkdir", "-p", "/srv/app"] in c.exec_log
    names = _names(got["data"])
    assert set(names) == {"a.txt", "sub", "sub/b.bin", "sub/c"}
    assert names["a.txt"][1] == 0o755 and names["a.txt"][2] == 0 and names["a.txt"][3] == b"A"      # no setuid bit, root
    assert names["sub"][0] == _tarfile.DIRTYPE and names["sub/b.bin"][3] == b"BB"


@pytest.mark.parametrize("bad", [
    [("/etc/passwd", b"x")], [("../up", b"x")], [("a/../../up", b"x")], [("ok", b"x"), ("sub/../../x", b"y")],
    [("l", ("link", "/etc/shadow"))], [("h", ("hard", "a.txt"))], [("f", ("fifo",))], [("d", ("dev",))],
    [("", b"x")], [(".", b"x")],
])
def test_fc_one_bad_entry_refuses_the_whole_put_and_nothing_is_written(env, bad):  # noqa: F811
    m = env
    add_lab(m, "lab_t")
    c = _member(m, "lab_t")
    c.put_archive = lambda *a: pytest.fail("nothing may reach the member")
    with pytest.raises(LabError) as e:
        _put(m, [("good.txt", b"g")] + bad)
    assert e.value.code == "bad_request" and e.value.extra["field"] == "archive_b64"
    assert c.exec_log == []


def test_fc_put_bounds_paths_and_garbage(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_t")
    c = _member(m, "lab_t")
    c.put_archive = lambda *a: True
    monkeypatch.setattr(L, "TREE_MAX_ENTRIES", 3)
    assert _put(m, [(f"f{i}", b"x") for i in range(3)])["entries"] == 3
    with pytest.raises(LabError) as e:
        _put(m, [(f"f{i}", b"x") for i in range(4)])
    assert "more than 3 entries" in e.value.message
    m.cfg.file_max_bytes = 4000
    with pytest.raises(LabError) as e:                      # 20 kB of zeros: a tiny archive, a big folder
        _put(m, [("big", b"\0" * 20000)])
    assert "unpacks to more than 16000" in e.value.message
    import os
    with pytest.raises(LabError) as e:                      # the archive itself over the limit
        _put(m, [("rnd", os.urandom(6000))])
    assert "max 4000" in e.value.message
    for path in ("relative/dir", "/srv/../etc", ""):
        with pytest.raises(LabError) as e:
            _put(m, [("a", b"x")], path=path)
        assert e.value.extra["field"] == "path"
    for junk in ("not base64 !!", _b64.b64encode(b"not a tar").decode()):
        with pytest.raises(LabError) as e:
            m.put_tree("lab_t", {"member": "m1", "path": "/x", "archive_b64": junk}, "owner")
        assert e.value.code == "bad_request"


def _docker_archive(folder, entries):
    """What Docker's get_archive returns for a folder: an uncompressed tar, every name under <folder>/."""
    raw = _tar([(folder, None)] + [(f"{folder}/{n}", v) for n, v in entries], mode="w")
    return iter([raw[:700], raw[700:]]), {"mode": (1 << 31) | 0o755, "size": 0}


def test_fc_a_folder_get_keeps_files_and_dirs_strips_the_top_and_redacts(env):  # noqa: F811
    m = env
    add_lab(m, "lab_t")
    c = _member(m, "lab_t")
    m.store.set_secret("p", "app", "TOK", "s3cr3t-token-abcdef")
    c.get_archive = lambda path: _docker_archive("out", [("r.txt", b"result s3cr3t-token-abcdef"), ("d", None),
                                                         ("d/x.bin", b"\x00\x01"), ("lnk", ("link", "/etc/shadow")),
                                                         ("d/pipe", ("fifo",))])
    out = m.get_tree("lab_t", "m1", "/root/out", "owner")
    assert (out["entries"], out["skipped"]) == (3, ["lnk", "d/pipe"])
    names = _names(_b64.b64decode(out["archive_b64"]))
    assert set(names) == {"r.txt", "d", "d/x.bin"} and names["r.txt"][3] == b"result <secret:TOK>"
    assert out["bytes"] == len(b"result <secret:TOK>") + 2


def test_fc_get_refusals_and_the_file_route_says_when_a_path_is_a_folder(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_t")
    c = _member(m, "lab_t")
    c.get_archive = lambda path: (iter([_tar([("f", b"x")], mode="w")]), {"mode": 0o644, "size": 1})
    with pytest.raises(LabError) as e:
        m.get_tree("lab_t", "m1", "/root/f", "owner")
    assert e.value.extra.get("kind") == "file"
    c.get_archive = lambda path: _docker_archive("out", [("a", b"x")])
    with pytest.raises(LabError) as e:                       # the single-file route, asked for a folder
        m.get_file("lab_t", "m1", "/root/out", "owner")
    assert e.value.code == "bad_request" and e.value.extra["kind"] == "directory"
    c.get_archive = lambda path: (iter([b""]), {"mode": (1 << 31) | (1 << 27), "size": 0})      # a symlink to a dir
    with pytest.raises(LabError) as e:
        m.get_file("lab_t", "m1", "/root/l", "owner")
    assert "kind" not in e.value.extra
    with pytest.raises(LabError):
        m.get_tree("lab_t", "m1", "/root/l", "owner")
    monkeypatch.setattr(L, "TREE_MAX_ENTRIES", 2)
    c.get_archive = lambda path: _docker_archive("out", [("a", b"x"), ("b", b"y"), ("c", b"z")])
    with pytest.raises(LabError) as e:
        m.get_tree("lab_t", "m1", "/root/out", "owner")
    assert "more than 2 entries" in e.value.message
    m.cfg.file_max_bytes = 1000
    c.get_archive = lambda path: _docker_archive("out", [("a", b"q" * 9000)])
    with pytest.raises(LabError) as e:
        m.get_tree("lab_t", "m1", "/root/out", "owner")
    assert "is over 4000 bytes" in e.value.message
    with pytest.raises(LabError):
        m.get_tree("lab_t", "m1", "relative", "owner")


@pytest.mark.parametrize("origin,put_status,get_status", [("owner", 200, 200), ("autopilot", 403, 403), (None, 403, 403)])
def test_fc_a_kept_lab_s_folders_are_the_owner_s_through_the_app(kept, env, origin, put_status, get_status):  # noqa: F811
    make_kept(env, "lab_k")
    c = env.docker._containers[env.container_name("lab_k", "m1")]
    puts = []
    c.put_archive = lambda path, data: puts.append(path) or True
    c.get_archive = lambda path: _docker_archive("out", [("a", b"x")])
    cl = _api(env)
    r = cl.put("/v1/labs/lab_k/tree", json={"member": "m1", "path": "/srv/x",
                                            "archive_b64": _b64.b64encode(_tar([("a", b"x")])).decode()}, headers=_h(origin))
    assert r.status_code == put_status and (puts == ["/srv/x"]) is (put_status == 200)
    g = cl.get("/v1/labs/lab_k/tree", params={"member": "m1", "path": "/root/out"}, headers=_h(origin))
    assert g.status_code == get_status
    if get_status == 200:
        assert g.json()["entries"] == 1 and set(_names(_b64.b64decode(g.json()["archive_b64"]))) == {"a"}


def test_fc_the_health_answer_names_tree():
    from ghost_labs.service import FEATURES
    assert "tree" in FEATURES


# ================================================================== battery survivors (mutate_4sx_labs.py)
def test_bat_the_lineage_join_guard_itself_refuses_a_probe_in_both_cases(kept, env):  # noqa: F811
    """`D34 lineage join any kept` / `other project ok` survived: the two older pins assert only the error CODE,
    which a later probe-tier check also gives. The guard's own message is asserted here."""
    from .test_review_r11 import _spec
    m = env
    make_kept(m, "lab_live", lineage="shop")
    m.store.update_lab("lab_live", origin="owner", owner="project:shop")       # the owner's production, same project
    add_lab(m, "lab_pr", owner="project:shop", origin="probe")
    _spec(m, "lab_pr")
    with pytest.raises(LabError) as e:
        kept.keep("lab_pr", {"app": {"name": "chess", "member": "m1", "port": 80}}, "probe")
    assert e.value.code == "owner_only" and "holds a kept lab this caller may not touch" in e.value.message
    m.store.update_lab("lab_live", origin="probe", owner="conversation:other")   # a probe's lab of ANOTHER owner
    with pytest.raises(LabError) as e:
        kept.keep("lab_pr", {"app": {"name": "chess", "member": "m1", "port": 80}}, "probe")
    assert e.value.code == "owner_only" and "holds a kept lab this caller may not touch" in e.value.message


# ================================================================== part-2 review: the fix list
@pytest.mark.parametrize("kill_exit,kill_raises,stream_ends,killed,outcome,mark_stays", [
    (0, False, True, True, "killed", False),         # signalled and seen gone
    (1, False, False, False, "not_killed", True),    # signalled, the group is still there
    (0, True, False, False, "not_killed", True),     # the kill could not be delivered (a paused member)
    (1, False, True, True, "killed", False),         # the look failed but the command's stream ended: it is gone
])
def test_m1_a_cancel_and_a_timeout_say_killed_only_when_the_command_is_gone(env, monkeypatch, kill_exit, kill_raises,  # noqa: F811
                                                                           stream_ends, killed, outcome, mark_stays):
    m = env
    monkeypatch.setattr(L, "CANCEL_CONFIRM_S", 4.0)
    for how in ("cancel", "timeout"):
        lab = f"lab_{how[:2]}"
        add_lab(m, lab)
        _member(m, lab)
        cmd = _Cmd(m, lab, timeout_s=1 if how == "timeout" else 900)
        real = cmd.c.exec_run

        def exec_run(line, _real=real, _cmd=cmd, **kw):
            text = line[-1] if isinstance(line, list) else str(line)
            if "kill -KILL" not in text:
                return _real(line, **kw)
            _cmd.kills.append(text)
            if stream_ends:
                _cmd.done.set()
            if kill_raises:
                raise RuntimeError("cannot exec in a paused container")
            return type("R", (), {"exit_code": kill_exit, "output": b""})()
        cmd.c.exec_run = exec_run
        if how == "cancel":
            ans = m.cancel_exec(lab, {}, "owner")
            assert [x["outcome"] for x in ans["cancelled"]] == [outcome]
        cmd.t.join(10)
        assert not cmd.t.is_alive() and cmd.result["killed"] is killed and len(cmd.kills) == 1
        assert m.exec_running(lab) is mark_stays and (lab in m._fg) is mark_stays
        if mark_stays:
            # a later cancel finds the command still listed and sends the SAME kill again; said as it went
            assert m.cancel_exec(lab, {}, "owner")["cancelled"][0]["outcome"] == "not_killed" and len(cmd.kills) == 2
            cmd.c.exec_run = lambda line, **kw: (cmd.kills.append("again"), type("R", (), {"exit_code": 0, "output": b""})())[1]
            assert m.cancel_exec(lab, {}, "owner")["cancelled"][0]["outcome"] == "killed" and len(cmd.kills) == 3
            assert m.exec_running(lab) is True                    # the mark goes when the stream ends, not before
        cmd.done.set()
        for _ in range(50):
            if not m.exec_running(lab):
                break
            time.sleep(0.1)
        assert m.exec_running(lab) is False and lab not in m._fg


def test_m1_a_command_that_ends_by_itself_carries_no_killed_and_a_slow_end_is_not_called_killed(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_c")
    _member(m, "lab_c")
    cmd = _Cmd(m, "lab_c")
    cmd.end()
    assert "killed" not in cmd.result and cmd.result["exit_code"] == 0
    # a cancel whose command has not answered inside the wait says nothing about it ("" — never "killed")
    monkeypatch.setattr(L, "CANCEL_CONFIRM_S", 0.2)
    slow = _Cmd(m, "lab_c")
    gate, entered = threading.Event(), threading.Event()
    monkeypatch.setattr(m, "_kill_group", lambda c, xid, started: (entered.set(), gate.wait(10), True)[2])
    ans = m.cancel_exec("lab_c", {}, "owner")
    assert ans["cancelled"][0]["outcome"] == "" and slow.t.is_alive() and entered.wait(5)
    gate.set()
    slow.end()
    assert slow.result["killed"] is True


def test_m7_a_member_check_never_takes_a_turn_from_the_template_check(env, monkeypatch):  # noqa: F811
    m = env
    # the template check passes on its 3rd look; every member check would burn 30 s of the 9 s there are
    spec, calls, clock = _ready_lab(m, monkeypatch, lambda mem, cmd, n: (0 if n >= 3 else 1) if cmd == "tpl-check" else 1)
    real = m._exec

    def slow(lab_id, member, cmd, timeout_s=0, **kw):
        if cmd != "tpl-check":
            clock["t"] += timeout_s
        return real(lab_id, member, cmd, timeout_s=timeout_s, **kw)
    monkeypatch.setattr(m, "_exec", slow)
    m.wait_ready("lab_r", spec)
    assert m.store.lab("lab_r")["state"] == "ready"
    assert len([c for c in calls if c[1] == "tpl-check"]) == 3
    assert [c[1] for c in calls[:3]] == ["tpl-check"] * 3                 # no member check before it passed
    d = m.detail("lab_r")["readiness"]["members"]
    assert d["pg1"] is False and d["pg2"] is None                         # ran and failed / never ran


def _tar_of(entries, gz=False):
    """A tar the way a container engine answers get_archive: [(name, bytes | None for a dir | ("sym"|"hard", target))]."""
    out = _io.BytesIO()
    with _tarfile.open(fileobj=out, mode="w:gz" if gz else "w") as t:
        for name, body in entries:
            ti = _tarfile.TarInfo(name)
            if body is None:
                ti.type, ti.mode = _tarfile.DIRTYPE, 0o755
                t.addfile(ti)
            elif isinstance(body, tuple):
                ti.type = _tarfile.SYMTYPE if body[0] == "sym" else _tarfile.LNKTYPE
                ti.linkname = body[1]
                t.addfile(ti)
            else:
                ti.size = len(body)
                t.addfile(ti, _io.BytesIO(body))
    return out.getvalue()


def _get_tree_of(m, entries, path="/srv/app", lab="lab_g"):
    if m.store.lab(lab) is None:
        add_lab(m, lab)
        _member(m, lab)
    c = m.docker._containers[m.container_name(lab, "m1")]
    asked = []

    def get_archive(p):
        asked.append(p)
        return iter([_tar_of(entries)]), {"mode": (1 << 31) | 0o755}
    c.get_archive = get_archive
    r = m.get_tree(lab, "m1", path, "owner")
    with _tarfile.open(fileobj=_io.BytesIO(_b64.b64decode(r["archive_b64"])), mode="r:gz") as t:
        got = [(ti.name, None if ti.isdir() else t.extractfile(ti).read()) for ti in t]
    return r, got, asked


@pytest.mark.parametrize("path,asked,names", [
    ("/srv/app", "/srv/app", ["app", "app/top.txt", "app/sub", "app/sub/b.txt", "app/sub/deep", "app/sub/deep/c.txt"]),
    ("/srv/app/", "/srv/app", ["app/", "app/top.txt", "app/sub/", "app/sub/b.txt", "app/sub/deep/", "app/sub/deep/c.txt"]),
    ("/srv/app/.", "/srv/app", ["./app", "./app/top.txt", "./app/sub", "./app/sub/b.txt", "./app/sub/deep",
                                "./app/sub/deep/c.txt"]),
    ("/srv/x/../app", "/srv/app", ["app", "app/top.txt", "app/sub", "app/sub/b.txt", "app/sub/deep", "app/sub/deep/c.txt"]),
])
def test_m4_a_fetched_folder_keeps_its_layout_under_either_archive_naming(env, path, asked, names):  # noqa: F811
    body = {n.strip("./"): b for n, b in zip(names, [None, b"T", None, b"B", None, b"C"])}
    r, got, was = _get_tree_of(env, [(n, body[n.strip("./")]) for n in names], path=path)
    assert was == [asked] and r["path"] == asked
    assert got == [("top.txt", b"T"), ("sub", None), ("sub/b.txt", b"B"), ("sub/deep", None), ("sub/deep/c.txt", b"C")]
    assert r["entries"] == 5 and r["bytes"] == 3 and r["skipped"] == [] and r["skipped_total"] == 0


@pytest.mark.parametrize("names", [
    ["./", "./top.txt", "./sub/b.txt"],          # the archive is named by "." — not this folder's name
    ["app", "app/a.txt", "other/b.txt"],         # one entry under another name
    ["app", "app/../../etc/passwd"],
    ["apple/a.txt"],
])
def test_m4_an_archive_not_named_by_the_folder_is_an_error_for_the_whole_call(env, names):  # noqa: F811
    with pytest.raises(LabError) as e:
        _get_tree_of(env, [(n, None if n.endswith("/") or "." not in n.rsplit("/", 1)[-1] else b"x") for n in names])
    assert e.value.code == "internal" and "nothing is returned" in e.value.message


@pytest.mark.parametrize("path", ["/", "//", "/.", "/srv/..", "//srv"])
def test_m4_the_root_folder_is_never_fetched(env, path):  # noqa: F811
    add_lab(env, "lab_g")
    c = _member(env, "lab_g")
    c.get_archive = lambda p: pytest.fail("the member is not asked")
    with pytest.raises(LabError) as e:
        env.get_tree("lab_g", "m1", path, "owner")
    assert e.value.code == "bad_request" and e.value.extra["field"] == "path"


def test_m4_links_devices_and_backslash_names_are_listed_as_skipped_never_sent(env):  # noqa: F811
    many = [(f"app/l{i}", ("sym", "/etc/passwd")) for i in range(60)]
    r, got, _ = _get_tree_of(env, [("app", None), ("app/ok.txt", b"ok"), ("app/hard", ("hard", "etc/passwd")),
                                   ("app/a\\b.txt", b"x"), ("app/sub\\d", None)] + many)
    assert got == [("ok.txt", b"ok")] and r["entries"] == 1
    assert r["skipped"][:3] == ["hard", "a\\b.txt", "sub\\d"] and len(r["skipped"]) == 50 and r["skipped_total"] == 63


@pytest.mark.parametrize("prefix", ["app", "./app"])
def test_4ud_c_a_hard_link_inside_the_folder_is_sent_as_a_copy(env, prefix):  # noqa: F811
    """Real Docker (§4UD-c, seen on the labs host): `ln top.txt hard.txt` → the archive holds `hard.txt` as the
    file and `top.txt` as a LINK to it; the folder came back without top.txt. Both names now arrive, same bytes.
    A link to a name outside the folder, to one not sent (a link to a link that was skipped), or one that comes
    BEFORE its target, is skipped like a symlink."""
    p = prefix
    r, got, _ = _get_tree_of(env, [(p, None), (f"{p}/hard.txt", b"same"), (f"{p}/top.txt", ("hard", f"{p}/hard.txt")),
                                   (f"{p}/sub", None), (f"{p}/sub/x", ("hard", f"{p}/hard.txt")),
                                   (f"{p}/out", ("hard", "etc/shadow")), (f"{p}/up", ("hard", f"{p}/../app/hard.txt")),
                                   (f"{p}/early", ("hard", f"{p}/late.txt")), (f"{p}/late.txt", b"L")])
    assert got == [("hard.txt", b"same"), ("top.txt", b"same"), ("sub", None), ("sub/x", b"same"), ("late.txt", b"L")]
    assert r["entries"] == 5 and r["bytes"] == 13 and r["skipped"] == ["out", "up", "early"]


def test_m8_a_small_archive_that_unpacks_huge_is_refused_by_the_bytes_read(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_t")
    c = _member(m, "lab_t")
    c.put_archive = lambda *a: pytest.fail("nothing may reach the member")
    m.cfg.file_max_bytes = 400_000
    bomb = _tar_of([("big", b"\0" * 300_000_000)], gz=True)
    assert len(bomb) < 400_000
    made = []
    real = L.tar_bytes_bounded

    def counted(data, limit, slack=0):
        try:
            out = real(data, limit, slack)
        except ValueError:
            made.append("refused")
            raise
        made.append(len(out))
        return out
    monkeypatch.setattr(L, "tar_bytes_bounded", counted)
    opened = []
    real_open = _tarfile.open
    monkeypatch.setattr(L.tarfile, "open", lambda *a, **k: (opened.append(k.get("mode")), real_open(*a, **k))[1])
    with pytest.raises(LabError) as e:
        m.put_tree("lab_t", {"member": "m1", "path": "/srv/app", "archive_b64": _b64.b64encode(bomb).decode()}, "owner")
    assert "unpacks to more than" in e.value.message and made == ["refused"] and opened == []


@pytest.mark.parametrize("n,ok", [(100, True), (101, True), (102, False)])
def test_m8_the_bounded_reader_never_produces_more_than_its_bound(n, ok):
    import gzip
    data = gzip.compress(b"a" * n)
    import zlib
    seen = []
    real = zlib.decompressobj

    class _D:
        def __init__(self, *a):
            self._d = real(*a)

        def decompress(self, data, max_length=0):
            seen.append(max_length)
            return self._d.decompress(data, max_length)

        def __getattr__(self, name):                 # unconsumed_tail, eof, unused_data: the real object's
            return getattr(self._d, name)
    zlib.decompressobj = _D
    try:
        if ok:
            assert L.tar_bytes_bounded(data, 100, slack=1) == b"a" * n
        else:
            with pytest.raises(ValueError):
                L.tar_bytes_bounded(data, 100, slack=1)
    finally:
        zlib.decompressobj = real
    assert seen == [102]                                   # asked for bound + 1 bytes, never "all of it"
    with pytest.raises(ValueError):
        L.tar_bytes_bounded(b"a" * 102, 100, slack=1)       # a plain tar over the bound
    with pytest.raises(ValueError):
        L.tar_bytes_bounded(b"\x1f\x8b" + b"junk" * 5, 100)


@pytest.mark.parametrize("name", ["a\\b.txt", "sub\\..\\..\\etc", "\\abs"])
def test_fix_a_backslash_name_is_refused_on_put(env, name):  # noqa: F811
    m = env
    add_lab(m, "lab_t")
    c = _member(m, "lab_t")
    c.put_archive = lambda *a: pytest.fail("nothing may reach the member")
    with pytest.raises(LabError) as e:
        _put(m, [("ok.txt", b"x"), (name, b"x")])
    assert e.value.code == "bad_request" and e.value.extra["field"] == "archive_b64"


@pytest.mark.parametrize("n,ok", [(500, True), (501, False)])
def test_fix_a_member_ready_command_is_bounded(n, ok):
    from ghost_labs import templates as T
    spec = {"name": "x", "members": [{"name": "a", "image": "i", "mem_mb": 64, "ready": "t" * n}]}
    if ok:
        assert T.parse(spec)["members"][0]["ready"] == "t" * n
    else:
        with pytest.raises(LabError) as e:
            T.parse(spec)
        assert e.value.extra["field"] == "ready"


def test_fix_a_cancel_counts_as_use_of_the_lab_only_when_it_is_allowed(env):  # noqa: F811
    m = env
    add_lab(m, "lab_c")
    _member(m, "lab_c")
    m.store.update_lab("lab_c", last_used_at=5.0, last_owner_use_at=5.0)
    with pytest.raises(LabError):
        m.cancel_exec("lab_c", {"member": "nope"}, "owner")
    assert m.store.lab("lab_c")["last_used_at"] == 5.0
    m.cancel_exec("lab_c", {}, "owner")                       # nothing running: still the owner using the lab
    row = m.store.lab("lab_c")
    assert row["last_used_at"] > 5.0 and row["last_owner_use_at"] > 5.0


@pytest.mark.parametrize("mode,sent", [(0o4777, 0o755), (0o2755, 0o755), (0o600, 0o600 & 0o755), (0o1666, 0o644),
                                       (0, 0o644)])
def test_fix_a_fetched_file_carries_plain_permission_bits_only(env, mode, sent):  # noqa: F811
    m = env
    add_lab(m, "lab_g")
    c = _member(m, "lab_g")
    out = _io.BytesIO()
    with _tarfile.open(fileobj=out, mode="w") as t:
        ti = _tarfile.TarInfo("app/run.sh")
        ti.size, ti.mode, ti.uid, ti.uname = 2, mode, 0, "root"
        t.addfile(ti, _io.BytesIO(b"hi"))
    c.get_archive = lambda p: (iter([out.getvalue()]), {"mode": (1 << 31) | 0o755})
    r = m.get_tree("lab_g", "m1", "/srv/app", "owner")
    with _tarfile.open(fileobj=_io.BytesIO(_b64.b64decode(r["archive_b64"])), mode="r:gz") as t:
        (got,) = list(t)
    assert got.mode == sent and got.uid == 0 and got.uname == ""


def test_fix_files_just_over_the_unpacked_bound_are_refused_by_their_sizes(env, monkeypatch):  # noqa: F811
    """Inside the header room the byte bound leaves, the per-entry sizes still decide (16 000 allowed, 17 000 sent)."""
    m = env
    add_lab(m, "lab_t")
    c = _member(m, "lab_t")
    c.put_archive = lambda *a: pytest.fail("nothing may reach the member")
    m.cfg.file_max_bytes = 4000
    with pytest.raises(LabError) as e:
        _put(m, [("a", b"\0" * 9000), ("b", b"\0" * 8000)])
    assert e.value.code == "bad_request" and "16000" in e.value.message
    c.put_archive = lambda *a: True
    assert _put(m, [("a", b"\0" * 9000), ("b", b"\0" * 7000)])["bytes"] == 16000


# ---------------------------------------------------------------- part-2 r2 N1: the cancel retry and the pid file
class _BashMember:
    """A member whose exec_run runs the REAL kill script in local bash, /run/ghost-exec mapped to a temp folder."""

    def __init__(self, root):
        self.root, self.attrs = str(root), {"State": {"StartedAt": "t0"}}

    def reload(self):
        pass

    def exec_run(self, argv, **kw):
        import subprocess
        argv = [a.replace("/run/ghost-exec", self.root) for a in argv]
        r = subprocess.run(argv, capture_output=True, timeout=30)
        return SimpleNamespace(exit_code=r.returncode, output=r.stdout)


def test_r2_n1_the_real_kill_script_says_not_killed_without_a_pid_file(env, tmp_path):  # noqa: F811
    """A cancel landing before the command's shell wrote its pid file signalled nobody: never "killed"."""
    assert env._kill_group(_BashMember(tmp_path), "abc123", "t0") is False


def test_r2_n1_the_real_kill_script_kills_the_group_and_only_then_drops_the_pid_file(env, tmp_path):  # noqa: F811
    import os
    import subprocess
    p = subprocess.Popen(["sleep", "60"], start_new_session=True)
    # reaped as soon as it dies, as the command's own parent does in a member: an unreaped child is a zombie, and
    # `kill -0` on Linux still finds a zombie's group (the script would wait and answer "not gone")
    threading.Thread(target=p.wait, daemon=True).start()
    try:
        (tmp_path / "abc123.pid").write_text(str(os.getpgid(p.pid)))
        assert env._kill_group(_BashMember(tmp_path), "abc123", "t0") is True
        assert p.wait(5) is not None
        assert not (tmp_path / "abc123.pid").exists()
    finally:
        p.kill()


def test_r2_n1_a_not_killed_command_keeps_its_pid_file_and_a_retry_is_not_called_killed(env, monkeypatch):  # noqa: F811
    """The group survives the kill: exec answers not killed and must NOT delete the pid file; the next cancel's
    retry reads it, the group is still there, and the outcome stays not killed (it read "killed" before)."""
    import re
    m = env
    monkeypatch.setattr(L, "CANCEL_CONFIRM_S", 3.0)
    add_lab(m, "lab_rr")
    _member(m, "lab_rr")
    cmd = _Cmd(m, "lab_rr")
    pidfiles, removed = {"x"}, []
    real = cmd.c.exec_run

    def exec_run(line, **kw):          # models the script: no file → 2, a group that will not die → 1
        text = line[-1] if isinstance(line, list) else str(line)
        if "kill -KILL" in text:
            cmd.kills.append(text)
            return SimpleNamespace(exit_code=1 if pidfiles else 2, output=b"")
        if "rm -f /run/ghost-exec" in text:
            removed.append(text)
            if ".pid" in text:
                pidfiles.clear()
        return real(line, **kw)
    cmd.c.exec_run = exec_run
    first = m.cancel_exec("lab_rr", {}, "owner")
    cmd.t.join(1)                                   # the stream is still open: exec keeps the mark
    assert first["cancelled"][0]["outcome"] in ("not_killed", "")
    for _ in range(50):
        if cmd.result is not None:
            break
        time.sleep(0.1)
    assert cmd.result is not None and cmd.result["killed"] is False
    assert all(".pid" not in x for x in removed) and pidfiles     # the pid file is kept for the retry
    again = m.cancel_exec("lab_rr", {}, "owner")
    assert again["cancelled"][0]["outcome"] != "killed"
    assert m.exec_running("lab_rr")
    assert len(cmd.kills) >= 2 and all(re.search(r"ghost-exec/\w+;", k) or "ghost-exec" in k for k in cmd.kills)
    cmd.done.set()


def test_r2_n3_host_refuses_a_gzip_cut_short_or_with_a_second_stream():
    whole = _tar_of([("a", b"a" * 4000), ("b", b"b" * 4000), ("c", b"c" * 4000)], gz=True)
    assert L.tar_bytes_bounded(whole, 10**6)
    for cut in (len(whole) - 8, len(whole) // 2):
        with pytest.raises(ValueError, match="cut short"):
            L.tar_bytes_bounded(whole[:cut], 10**6)
    with pytest.raises(ValueError, match="more than one gzip"):
        L.tar_bytes_bounded(whole + _tar_of([("d", b"d")], gz=True), 10**6)
    assert L.tar_bytes_bounded(whole + b"\0" * 512, 10**6) == L.tar_bytes_bounded(whole, 10**6)   # zero padding
    for tail in (b"\0" * 8 + whole, b"\0" * 8 + b"junk", b"junk" + b"\0" * 8):      # … and nothing after it
        with pytest.raises(ValueError, match="more than one gzip"):
            L.tar_bytes_bounded(whole + tail, 10**6)



def test_r3_hard_links_are_bounded_by_what_is_built_not_what_was_read(env):  # noqa: F811
    """r3 MAJOR 1: each hard link is a whole copy in the answer; 150 links to 1.5 MB built 226 MB in memory before
    the packed-size check. r4: the bound is what the agent accepts (file_max_bytes), exactly, not the 4× read bound."""
    env.cfg.file_max_bytes = 10_000
    base = [("app", None), ("app/big", b"x" * 2_500)]
    r, got, _ = _get_tree_of(env, base + [(f"app/l{i}", ("hard", "app/big")) for i in range(3)], lab="lab_h")
    assert r["entries"] == 4 and r["bytes"] == 10_000                    # exactly at the bound: returned
    with pytest.raises(LabError) as e:
        _get_tree_of(env, base + [(f"app/l{i}", ("hard", "app/big")) for i in range(3)] + [("app/one", b"y")],
                     lab="lab_h2")
    assert e.value.code == "bad_request" and "more than 10000 bytes (hard links count as copies)" in e.value.message
    with pytest.raises(LabError) as e:                                  # no links: the text does not blame them
        _get_tree_of(env, [("app", None), ("app/a", b"a" * 6_000), ("app/b", b"b" * 4_001)], lab="lab_h3")
    assert "more than 10000 bytes" in e.value.message and "hard link" not in e.value.message


def test_r3_a_hard_link_is_resolved_only_inside_this_folder(env):  # noqa: F811
    """The target must sit under the fetched folder's own name: `other/big` is not `app/big` even though the
    rest of the path is the same."""
    r, got, _ = _get_tree_of(env, [("app", None), ("app/big", b"B"), ("app/l", ("hard", "other/big"))])
    assert got == [("big", b"B")] and r["skipped"] == ["l"]


def test_r4_a_not_killed_command_keeps_only_its_pid_file(env, monkeypatch):  # noqa: F811
    """A command not confirmed killed keeps its pid file for the retry; its stdin file is removed (r5: the shell
    that deletes it may never have run)."""
    m = env
    monkeypatch.setattr(L, "CANCEL_CONFIRM_S", 1.0)
    add_lab(m, "lab_si")
    _member(m, "lab_si")
    cmd = _Cmd(m, "lab_si")
    rms = []
    real = cmd.c.exec_run

    def exec_run(line, **kw):
        text = line[-1] if isinstance(line, list) else str(line)
        if "kill -KILL" in text:
            cmd.kills.append(text)
            return SimpleNamespace(exit_code=1, output=b"")          # the group will not die
        if text.startswith("rm -f /run/ghost-exec"):
            rms.append(text)
        return real(line, **kw)
    cmd.c.exec_run = exec_run
    m.cancel_exec("lab_si", {}, "owner")
    for _ in range(50):
        if cmd.result is not None:
            break
        time.sleep(0.1)
    assert cmd.result is not None and cmd.result["killed"] is False
    assert len(rms) == 1 and rms[0].endswith(".stdin") and ".pid" not in rms[0]
    cmd.done.set()


def test_r4_the_commands_own_shell_takes_its_stdin_and_deletes_the_file_first(env, tmp_path):  # noqa: F811
    """The REAL command line, run in a local shell (/run/ghost-exec mapped to a temp folder): the command reads
    what was piped, and the file is gone before the command runs — nothing later has to reach the member (r4: a
    paused member refused the cleanup exec and the file, maybe a password, stayed)."""
    import shlex
    import subprocess
    m = env
    add_lab(m, "lab_in")
    c = _member(m, "lab_in")
    seen = {}

    def put(cont, path, data, mode):
        p = tmp_path / path.rsplit("/", 1)[-1]
        p.write_bytes(data)
        seen["file"] = p

    def create(cid, argv, **kw):
        inner = shlex.split(argv[-1].split("exec setsid -w sh -c ", 1)[1])[0]
        seen["inner"] = inner.replace("/run/ghost-exec", str(tmp_path))
        raise RuntimeError("stop after create")          # the command is run below, in a real shell
    m._put_bytes = put
    m.docker.api.exec_create = create
    c.exec_run = lambda *a, **k: SimpleNamespace(exit_code=0, output=b"")
    c.status = "running"
    c.reload = lambda: None
    with pytest.raises(RuntimeError):
        m._exec("lab_in", "m1", "x", 30, stdin="s3cret\n")
    r = subprocess.run(["sh", "-c", seen["inner"]], capture_output=True, text=True, timeout=30,
                       env={"PATH": "/usr/bin:/bin", "GHOST_CMD": f"ls {tmp_path}; cat"})
    assert "s3cret" in r.stdout, r.stderr
    assert not seen["file"].exists() and ".stdin" not in r.stdout


def test_r4_an_exec_that_cannot_start_removes_its_stdin_file(env):  # noqa: F811
    m = env
    add_lab(m, "lab_cs")
    c = _member(m, "lab_cs")
    lines = []
    m._put_bytes = lambda *a: None
    c.exec_run = lambda line, **k: (lines.append(line[-1]), SimpleNamespace(exit_code=0, output=b""))[1]
    c.status = "running"
    c.reload = lambda: None

    def boom(*a, **k):
        raise RuntimeError("engine said no")
    m.docker.api.exec_create = boom
    with pytest.raises(RuntimeError):
        m._exec("lab_cs", "m1", "x", 30, stdin="pw")
    assert any("rm -f /run/ghost-exec/" in x and x.endswith(".stdin") for x in lines)


def test_r4_a_command_that_ends_by_itself_has_its_pid_file_removed(env):  # noqa: F811
    m = env
    add_lab(m, "lab_pe")
    _member(m, "lab_pe")
    cmd = _Cmd(m, "lab_pe")
    rms = []
    real = cmd.c.exec_run

    def exec_run(line, **kw):
        text = line[-1] if isinstance(line, list) else str(line)
        if text.startswith("rm -f /run/ghost-exec"):
            rms.append(text)
        return real(line, **kw)
    cmd.c.exec_run = exec_run
    cmd.end()
    assert cmd.result is not None and "killed" not in cmd.result
    assert len(rms) == 1 and ".stdin" in rms[0] and rms[0].endswith(".pid")


def test_r5_the_stdin_file_is_removed_when_the_commands_shell_never_ran(env):  # noqa: F811
    """r5 MAJOR: a workdir that does not exist (or a member paused between create and start) fails the START —
    the shell that deletes the file never ran. The host's own removal after the run is what takes it then."""
    m = env
    add_lab(m, "lab_wd")
    c = _member(m, "lab_wd")
    c.status = "running"
    c.reload = lambda: None
    lines, files = [], set()

    def put(cont, path, data, mode):
        files.add(path)

    def exec_run(line, **k):
        text = line[-1]
        lines.append(text)
        if text.startswith("rm -f"):
            for f in text.split()[2:]:
                files.discard(f)
        return SimpleNamespace(exit_code=0, output=b"")
    m._put_bytes = put
    c.exec_run = exec_run
    m.docker.api.exec_create = lambda cid, argv, **kw: {"Id": "x1"}

    def start(i, stream=True, demux=True):
        raise RuntimeError('OCI runtime exec failed: chdir to cwd ("/nope") set in config.json failed')
        yield  # noqa
    m.docker.api.exec_start = start
    m.docker.api.exec_inspect = lambda i: {"ExitCode": 126}
    m._exec("lab_wd", "m1", "psql", 30, workdir="/nope", stdin="pw\n")
    assert files == set(), lines


def test_r5_only_a_resolved_hard_link_is_blamed(env):  # noqa: F811
    env.cfg.file_max_bytes = 10_000
    with pytest.raises(LabError) as e:              # a link that resolves to nothing, plus too much content
        _get_tree_of(env, [("app", None), ("app/l", ("hard", "app/missing")), ("app/a", b"a" * 10_001)], lab="lab_r5")
    assert "more than 10000 bytes" in e.value.message and "hard link" not in e.value.message


def test_r6_the_shell_takes_its_stdin_before_it_writes_its_pid_file(env):  # noqa: F811
    """Taken first: a kill that lands once the pid file exists finds the stdin file already gone."""
    import shlex
    m = env
    add_lab(m, "lab_or")
    c = _member(m, "lab_or")
    c.status, c.reload = "running", (lambda: None)
    seen = {}
    m._put_bytes = lambda *a: None
    c.exec_run = lambda *a, **k: SimpleNamespace(exit_code=0, output=b"")

    def create(cid, argv, **kw):
        seen["inner"] = shlex.split(argv[-1].split("exec setsid -w sh -c ", 1)[1])[0]
        raise RuntimeError("stop")
    m.docker.api.exec_create = create
    with pytest.raises(RuntimeError):
        m._exec("lab_or", "m1", "x", 30, stdin="pw")
    assert seen["inner"].index("exec 0<") < seen["inner"].index("rm -f") < seen["inner"].index("echo $$")


def test_r6_the_kill_script_removes_both_files_once_the_group_is_gone(env, tmp_path):  # noqa: F811
    import os
    import subprocess
    p = subprocess.Popen(["sleep", "60"], start_new_session=True)
    # reaped as soon as it dies, as the command's own parent does in a member: an unreaped child is a zombie, and
    # `kill -0` on Linux still finds a zombie's group (the script would wait and answer "not gone")
    threading.Thread(target=p.wait, daemon=True).start()
    try:
        (tmp_path / "abc123.pid").write_text(str(os.getpgid(p.pid)))
        (tmp_path / "abc123.stdin").write_text("pw")
        assert env._kill_group(_BashMember(tmp_path), "abc123", "t0") is True
        assert not (tmp_path / "abc123.stdin").exists() and not (tmp_path / "abc123.pid").exists()
    finally:
        p.kill()
