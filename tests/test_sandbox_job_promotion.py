"""Promotion of long-running sandbox commands to supervised jobs.

The unit under test is :mod:`ghost_agent.sandbox.jobs`. It is exercised
END-TO-END against real processes rather than mocks: the runner script it
writes is plain ``sh``, and the bind mount means "container path" and "host
path" differ only by a prefix — so a fake sandbox manager that rewrites
``/workspace`` to a tmp dir and shells out locally runs the SAME script,
records the SAME pid, and writes the SAME exit sentinel the container would.
That is what makes these tests able to catch a broken pidfile, a stale exit
sentinel, or a kill that misses the process group.

Two macOS shims are needed and are called out where they are applied: there
is no ``setsid`` binary (replaced by a three-line Python equivalent, so the
job really does become its own session leader and the group-kill under test
is the real one), and there is no ``/proc``, which makes the zombie half of
``_pid_alive`` a no-op locally — the launcher re-parents its grandchild
exactly as the container does, so no zombie is left to be misread as alive.
"""

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ghost_agent.sandbox import jobs as sbx_jobs
from ghost_agent.sandbox.jobs import SandboxJobSupervisor

# Stand-in for the `setsid` binary macOS does not ship. Faithful: the job
# becomes a session/process-group leader, so its recorded `$$` IS the group
# leader and `kill -- -<pid>` reaps the whole tree — the property
# _kill_pgroup, cancel(), and the TTL reaper all depend on.
_SETSID_SHIM = "import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])"


class _FakeExecResult:
    def __init__(self, output: bytes, exit_code: int):
        self.output = output
        self.exit_code = exit_code


class FakeSandbox:
    """Runs container commands on the host with ``/workspace`` rewritten to a
    tmp dir. Not a mock: the shell really runs."""

    def __init__(self, root: Path):
        self.host_workspace = Path(root)
        self.container = None
        self.calls = []
        self.host_workspace.mkdir(parents=True, exist_ok=True)
        # OUTSIDE the workspace on purpose: a fresh file inside it would be
        # read by the progress probe as the job making progress, and every
        # wedge test would promote instead of dying.
        self._setsid = self.host_workspace.parent / "_setsid_shim.py"
        self._setsid.write_text(_SETSID_SHIM)

    def _translate(self, cmd: str) -> str:
        return (str(cmd)
                .replace("setsid ", f"{sys.executable} {self._setsid} ")
                .replace("/workspace", str(self.host_workspace)))

    def _translate_scripts(self):
        """The runner script is written from the host but names CONTAINER
        paths — inside the container the bind mount makes those the same
        file. Locally there is no mount, so rewrite the script's own paths
        the same way the command string is rewritten."""
        for script in (self.host_workspace / ".jobs").glob("*.cmd.sh"):
            body = script.read_text()
            if "/workspace" in body:
                script.write_text(body.replace("/workspace",
                                               str(self.host_workspace)))

    def _run(self, cmd: str, workdir=None):
        self._translate_scripts()
        real = self._translate(cmd)
        cwd = self._translate(str(workdir)) if workdir else str(self.host_workspace)
        Path(cwd).mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(real, shell=True, cwd=cwd,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return proc.stdout, proc.returncode

    def execute(self, cmd, timeout=600, workdir=None, **kwargs):
        self.calls.append(cmd)
        out, code = self._run(cmd, workdir)
        return out.decode("utf-8", "replace"), code

    def _exec_run(self, cmd, deadline_s=None, workdir=None, **kwargs):
        self.calls.append(cmd)
        out, code = self._run(cmd, workdir)
        return _FakeExecResult(out, code)


@pytest.fixture
def sup(tmp_path):
    return SandboxJobSupervisor(FakeSandbox(tmp_path))


@pytest.fixture(autouse=True)
def _fast_rails(monkeypatch):
    """Short progress window so tests don't have to burn two real minutes."""
    monkeypatch.setenv("GHOST_SANDBOX_JOB_PROGRESS_WINDOW_S", "10")
    yield


def _kwargs(sup):
    return {"workdir": "/workspace", "demux": False}


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _cleanup(sup):
    """Kill leftovers by PROCESS GROUP — and only when the job really is its
    own group leader. Killing `getpgid(pid)` unconditionally would take out
    pytest itself if the setsid shim ever stopped working."""
    for entry in sup.list_entries():
        if entry.get("state") != sbx_jobs.STATE_RUNNING:
            continue
        pid = int(entry["pid"])
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


# ---------------------------------------------------------------- fast path

def test_fast_command_completes_normally_and_leaves_nothing_behind(sup):
    out, code, job = sup.run("echo hello", timeout=30, exec_kwargs=_kwargs(sup))
    assert job is None
    assert code == 0
    assert b"hello" in out
    # No bookkeeping survives a completed run: script, pid, exit AND log.
    leftovers = sorted(p.name for p in sup.host_dir.glob("job-*"))
    assert leftovers == [], f"completed run left {leftovers}"


def test_failing_command_reports_its_own_exit_code(sup):
    out, code, job = sup.run("sh -c 'echo boom >&2; exit 42'", timeout=30,
                             exec_kwargs=_kwargs(sup))
    assert job is None
    assert code == 42
    assert b"boom" in out


def test_stderr_is_captured_with_stdout(sup):
    out, code, _ = sup.run("sh -c 'echo out; echo err >&2'", timeout=30,
                           exec_kwargs=_kwargs(sup))
    assert code == 0
    assert b"out" in out and b"err" in out


# ---------------------------------------------------------------- promotion

def test_command_still_writing_at_the_budget_is_promoted_not_killed(sup):
    try:
        out, code, job = sup.run(
            "sh -c 'for i in 1 2 3 4 5 6 7 8 9 10; do echo tick $i; "
            "sleep 0.3; done'",
            timeout=1, exec_kwargs=_kwargs(sup))
        assert job is not None, "a command still producing output must promote"
        # A promotion is NOT a failure — exit 0 is what keeps the turn loop
        # from scoring a transient strike.
        assert code == 0
        assert b"tick" in out
        assert job["state"] == sbx_jobs.STATE_RUNNING
        assert _alive(job["pid"]), "the promoted process must still be running"
        # Registry is on disk (survives an agent restart) and the log keeps
        # growing where the model can read it.
        reg = json.loads((sup.host_dir / "registry.json").read_text())
        assert job["id"] in reg
        assert job["log"] == f".jobs/{job['id']}.log"
        assert (sup.host_dir / f"{job['id']}.log").exists()
    finally:
        _cleanup(sup)


def test_promoted_job_deadline_is_a_separate_lifetime_cap(sup, monkeypatch):
    monkeypatch.setenv("GHOST_SANDBOX_JOB_TTL_S", "600")
    try:
        _out, _code, job = sup.run(
            "sh -c 'while :; do echo x; sleep 0.2; done'",
            timeout=1, exec_kwargs=_kwargs(sup))
        assert job is not None
        # The clock starts at PROMOTION, not at launch.
        assert job["deadline_at"] - job["promoted_at"] == pytest.approx(600, abs=2)
        assert job["promoted_at"] > job["started_at"]
    finally:
        _cleanup(sup)


def test_promotion_credits_a_silent_job_whose_own_io_moved(sup, monkeypatch):
    """A silent downloader must promote too — that is the entire point of the
    second signal. This command writes a FILE and prints nothing to its log,
    so only the per-process I/O counters can earn it a promotion.

    The counters are stubbed because they are read from ``/proc``, which the
    container has and this host does not; the SHELL that reads them is
    covered by ``test_probe_*`` (and by the live docker run). What is pinned
    here is the RULE that consumes them.
    """
    counter = {"n": 1000}

    def _moving_probe(pid):
        counter["n"] += 4096
        return True, counter["n"]

    monkeypatch.setattr(sup, "_probe", _moving_probe)
    try:
        _out, code, job = sup.run(
            "sh -c 'while :; do echo data >> grow.bin; sleep 0.2; done'",
            timeout=4, exec_kwargs=_kwargs(sup))
        assert job is not None, "the job's own I/O must earn promotion"
        assert code == 0
    finally:
        _cleanup(sup)


def test_another_jobs_writes_do_not_count_as_this_jobs_progress(sup, monkeypatch):
    """The defect a LIVE docker run caught, that no unit test had: the first
    implementation asked "did a file under the workdir change recently", so a
    genuinely wedged `sleep 300` was PROMOTED because a DIFFERENT job was
    writing in the same shared /workspace. Progress must attribute to the job
    that earned it — a job whose OWN counters never move is wedged no matter
    how busy its directory is."""
    try:
        # A real, busy neighbour hammering the shared workdir throughout. It
        # also prints, so it earns its own promotion via log growth on any
        # platform — the point is the FILE it keeps writing next door.
        _o, _c, busy = sup.run(
            "sh -c 'while :; do echo tick; echo data >> noisy.bin; "
            "sleep 0.2; done'",
            timeout=1, exec_kwargs=_kwargs(sup))
        assert busy is not None, "the noisy neighbour promotes on log growth"

        monkeypatch.setattr(sup, "_probe", lambda pid: (True, 500))  # frozen
        _out, code, wedged = sup.run("sleep 30", timeout=4,
                                     exec_kwargs=_kwargs(sup))
        assert wedged is None, (
            "a silent idle process must not inherit another job's progress")
        assert code == 124
        assert (sup.host_dir.parent / "noisy.bin").exists(), (
            "the neighbour really was writing into the shared workdir")
    finally:
        _cleanup(sup)


def test_io_that_stopped_before_the_window_is_not_progress(sup, monkeypatch):
    """Counters that moved early and then froze describe a job that has since
    wedged. The comparison is against a sample from BEFORE the window."""
    monkeypatch.setenv("GHOST_SANDBOX_JOB_PROGRESS_WINDOW_S", "10")
    frozen = {"n": 1000}
    calls = []

    def _probe(pid):
        calls.append(1)
        # Moves once at the first probe, then never again.
        if len(calls) == 1:
            frozen["n"] += 8192
        return True, frozen["n"]

    monkeypatch.setattr(sup, "_probe", _probe)
    _out, code, job = sup.run("sleep 60", timeout=13, exec_kwargs=_kwargs(sup))
    assert job is None
    assert code == 124


def test_a_working_command_is_promoted_LONG_before_the_budget(sup, monkeypatch):
    """The whole point of the 2026-08-12 change: a command that is
    demonstrably working should stop making the turn wait. Blocking for the
    full budget to reach a decision that was obvious at 90 s is pure
    latency — the operator waits, and so does the model."""
    monkeypatch.setenv("GHOST_SANDBOX_JOB_PROMOTE_AFTER_S", "5")
    try:
        t0 = time.monotonic()
        _out, code, job = sup.run(
            "sh -c 'while :; do echo tick; sleep 0.2; done'",
            timeout=60, exec_kwargs=_kwargs(sup))
        elapsed = time.monotonic() - t0
        assert job is not None and code == 0
        assert elapsed < 20, (
            f"promoted after {elapsed:.0f}s — it must not wait out the "
            f"60s budget once the command has proven itself")
        assert job["ran_before_promotion_s"] < 20
    finally:
        _cleanup(sup)


def test_a_silent_command_still_gets_the_FULL_budget_before_dying(sup,
                                                                  monkeypatch):
    """The threshold and the budget answer different questions. Lowering the
    KILL time along with the promotion time would make a silent
    pure-computation run die 6.7x sooner — a regression dressed up as a
    speed-up. A quiet command must still get every second of its budget."""
    monkeypatch.setenv("GHOST_SANDBOX_JOB_PROMOTE_AFTER_S", "2")
    monkeypatch.setenv("GHOST_SANDBOX_JOB_PROGRESS_WINDOW_S", "10")
    t0 = time.monotonic()
    _out, code, job = sup.run("sleep 60", timeout=8, exec_kwargs=_kwargs(sup))
    elapsed = time.monotonic() - t0
    assert job is None, "a silent command must not be promoted early"
    assert code == 124
    assert elapsed >= 7.5, (
        f"killed after {elapsed:.1f}s — the silent command was denied its "
        f"full 8s budget; the promotion threshold must not shorten the kill")


def test_a_command_that_goes_quiet_early_but_works_later_still_promotes(
        sup, monkeypatch):
    """Promotion is re-evaluated at every probe past the threshold, not once.
    A command that is slow to start must not be judged on its first
    seconds."""
    monkeypatch.setenv("GHOST_SANDBOX_JOB_PROMOTE_AFTER_S", "2")
    monkeypatch.setenv("GHOST_SANDBOX_JOB_PROGRESS_WINDOW_S", "10")
    monkeypatch.setattr(sbx_jobs, "_PROBE_EVERY_S", 1.0)
    try:
        _out, code, job = sup.run(
            "sh -c 'sleep 4; while :; do echo late; sleep 0.2; done'",
            timeout=30, exec_kwargs=_kwargs(sup))
        assert job is not None, (
            "a command that starts producing later must still be promoted")
        assert code == 0
    finally:
        _cleanup(sup)


# ---------------------------------------------------------------- the wedge

def test_wedged_command_is_killed_at_the_budget(sup):
    """No output, no file change → still a timeout. Without this the feature
    is just a slower timeout."""
    out, code, job = sup.run("sleep 30", timeout=1, exec_kwargs=_kwargs(sup))
    assert job is None, "a silent, idle process must NOT be promoted"
    assert code == 124
    assert sup.list_entries() == []
    # And the process is really gone, not merely un-promoted.
    time.sleep(0.5)
    assert not (sup.host_dir / "registry.json").exists() \
        or json.loads((sup.host_dir / "registry.json").read_text()) == {}


def test_output_older_than_the_window_does_not_count_as_progress(sup, monkeypatch):
    """A command that printed a banner at t=0 and then went quiet is wedged;
    the window is what makes 'has written output' mean 'recently'."""
    monkeypatch.setenv("GHOST_SANDBOX_JOB_PROGRESS_WINDOW_S", "10")
    # Budget 12s > window 10s, so the initial burst falls outside the window.
    out, code, job = sup.run("sh -c 'echo banner; sleep 60'", timeout=12,
                             exec_kwargs=_kwargs(sup))
    assert job is None
    assert code == 124
    assert b"banner" in out


def test_probe_parses_the_container_answer(sup, monkeypatch):
    """`<matched> <alive> <io>` from the container, and every malformed shape
    degrades to the plain pid check rather than inventing evidence."""
    seen = {}

    def _exec(cmd, timeout=30):
        seen["cmd"] = cmd
        return seen["out"], seen["code"]

    monkeypatch.setattr(sup, "_exec", _exec)
    monkeypatch.setattr(sup, "_pid_state", lambda pid: "fallback")

    seen.update(out="3 1 40960\n", code=0)
    assert sup._probe(1234) == (True, 40960)
    assert "S=1234" in seen["cmd"], "the probe must scope to the job's session"

    # Matched processes, none of them alive (all zombies).
    seen.update(out="2 0 40960\n", code=0)
    assert sup._probe(1234) == (False, 40960)

    # No process matched: either the session is gone or /proc is unreadable —
    # indistinguishable here, so defer, and offer NO io evidence.
    seen.update(out="0 0 0\n", code=0)
    assert sup._probe(1234) == ("fallback", None)

    # Probe failed outright / garbage.
    seen.update(out="", code=1)
    assert sup._probe(1234) == ("fallback", None)
    seen.update(out="wat\n", code=0)
    assert sup._probe(1234) == ("fallback", None)
    seen.update(out="3 1 not-a-number\n", code=0)
    assert sup._probe(1234) == (True, None)


def test_probe_rejects_a_malformed_pid(sup):
    assert sup._probe(None) == (False, None)
    assert sup._probe("garbage") == (False, None)


@pytest.mark.skipif(not Path("/proc/self/io").exists(),
                    reason="no /proc — the shell half is covered by the live "
                           "docker run instead")
def test_probe_reads_real_proc_counters(sup):
    """When /proc IS available, the shell really returns moving counters for
    a live session."""
    try:
        _o, _c, job = sup.run(
            "sh -c 'while :; do echo data >> grow.bin; sleep 0.2; done'",
            timeout=4, exec_kwargs=_kwargs(sup))
        assert job is not None
        alive, io_a = sup._probe(job["pid"])
        assert alive is True and io_a is not None
        time.sleep(1.0)
        _alive_b, io_b = sup._probe(job["pid"])
        assert io_b != io_a, "a working job's I/O counters must move"
    finally:
        _cleanup(sup)


def test_probe_on_a_dead_pid_reports_not_alive(sup):
    dead = subprocess.Popen(["true"])
    dead.wait()
    alive, _io = sup._probe(dead.pid)
    assert alive is False


# ---------------------------------------------------------------- lifecycle

def test_reap_lands_a_finished_job_with_its_exit_code(sup):
    _out, _code, job = sup.run(
        "sh -c 'echo working; sleep 2; echo done; exit 7'",
        timeout=1, exec_kwargs=_kwargs(sup))
    assert job is not None
    for _ in range(60):
        changed = sup.reap()
        if changed:
            break
        time.sleep(0.2)
    entry = sup.get(job["id"])
    assert entry["state"] == sbx_jobs.STATE_DONE
    assert entry["exit_code"] == 7
    assert "done" in sup.log_tail(job["id"])


def test_a_job_finishing_mid_reap_is_landed_not_declared_lost(sup):
    """reap() reads the exit sentinel, then probes liveness. A job that exits
    BETWEEN those two steps was marked LOST — destroying its exit code and its
    output. This reproduces the interleaving exactly: the first sentinel read
    answers "not finished", and by the liveness probe the process is gone."""
    jid = "job-0000cafe"
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / f"{jid}.exit").write_text("3\n")
    (sup.host_dir / f"{jid}.log").write_text("all done\n")
    dead = subprocess.Popen(["true"])
    dead.wait()                       # reaped → its pid is not alive
    (sup.host_dir / "registry.json").write_text(json.dumps({jid: {
        "id": jid, "state": sbx_jobs.STATE_RUNNING, "pid": dead.pid,
        "command": "x", "deadline_at": time.time() + 999,
        "promoted_at": time.time(), "container_id": None,
    }}))

    real_read = sup._read_exit
    fired = []

    def _absent_on_the_first_look(job_id):
        if not fired:
            fired.append(1)
            return None
        return real_read(job_id)

    sup._read_exit = _absent_on_the_first_look
    changed = sup.reap()
    sup._read_exit = real_read

    assert [e["state"] for e in changed] == [sbx_jobs.STATE_DONE], (
        "a job that finished during the reap must land, not read as LOST")
    assert sup.get(jid)["exit_code"] == 3


def test_a_pid_from_a_dead_generation_is_lost_without_waiting(sup):
    """The re-read grace above must NOT be paid when the container itself is
    gone: no new sentinel can appear, and the wait would stall every reap."""
    jid = "job-0000dead"
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps({jid: {
        "id": jid, "state": sbx_jobs.STATE_RUNNING, "pid": os.getpid(),
        "command": "x", "deadline_at": time.time() + 999,
        "promoted_at": time.time(), "container_id": "OLD",
    }}))
    sup.sandbox.container = type("C", (), {"id": "NEW"})()
    waited = []
    sup._wait_for_sentinel = lambda *a, **k: waited.append(a) or None
    t0 = time.monotonic()
    changed = sup.reap()
    assert [e["state"] for e in changed] == [sbx_jobs.STATE_LOST]
    assert waited == [], "no sentinel can arrive from a dead container"
    assert time.monotonic() - t0 < 2.0


def test_reap_kills_a_job_past_its_lifetime_cap(sup):
    try:
        _out, _code, job = sup.run(
            "sh -c 'while :; do echo x; sleep 0.2; done'",
            timeout=1, exec_kwargs=_kwargs(sup))
        assert job is not None
        pid = job["pid"]
        # Backdate the deadline: the cap is the rail, the clock is incidental.
        reg = json.loads((sup.host_dir / "registry.json").read_text())
        reg[job["id"]]["deadline_at"] = time.time() - 1
        (sup.host_dir / "registry.json").write_text(json.dumps(reg))

        changed = sup.reap()
        assert [e["state"] for e in changed] == [sbx_jobs.STATE_EXPIRED]
        time.sleep(0.5)
        assert not _alive(pid), "an expired job must actually be killed"
    finally:
        _cleanup(sup)


def test_cancel_kills_the_process_group(sup):
    try:
        _out, _code, job = sup.run(
            "sh -c 'while :; do echo x; sleep 0.2; done'",
            timeout=1, exec_kwargs=_kwargs(sup))
        assert job is not None
        killed, msg = sup.cancel(job["id"])
        assert killed is True
        assert "cancelled" in msg
        time.sleep(0.5)
        assert not _alive(job["pid"])
        assert sup.get(job["id"])["state"] == sbx_jobs.STATE_CANCELLED
        # Idempotent — and a second cancel reports FALSE, not a phantom kill.
        killed2, msg2 = sup.cancel(job["id"])
        assert killed2 is False
        assert "already" in msg2
    finally:
        _cleanup(sup)


def test_concurrency_cap_kills_rather_than_detaching_untracked(sup, monkeypatch):
    monkeypatch.setenv("GHOST_SANDBOX_JOB_MAX", "1")
    try:
        _o, _c, first = sup.run("sh -c 'while :; do echo x; sleep 0.2; done'",
                                timeout=1, exec_kwargs=_kwargs(sup))
        assert first is not None
        _o2, code2, second = sup.run(
            "sh -c 'while :; do echo y; sleep 0.2; done'",
            timeout=1, exec_kwargs=_kwargs(sup))
        assert second is None, "the cap must refuse a second promotion"
        assert code2 == 124
        assert len(sup.list_entries()) == 1
    finally:
        _cleanup(sup)


def test_forget_drops_the_row_and_the_log(sup):
    _out, _code, job = sup.run("sh -c 'echo hi; sleep 3'", timeout=1,
                               exec_kwargs=_kwargs(sup))
    assert job is not None
    for _ in range(60):
        if sup.reap():
            break
        time.sleep(0.2)
    assert (sup.host_dir / f"{job['id']}.log").exists()
    sup.forget(job["id"])
    assert sup.get(job["id"]) is None
    assert not (sup.host_dir / f"{job['id']}.log").exists()


def test_forget_refuses_to_drop_a_running_job(sup):
    try:
        _o, _c, job = sup.run("sh -c 'while :; do echo x; sleep 0.2; done'",
                              timeout=1, exec_kwargs=_kwargs(sup))
        assert job is not None
        sup.forget(job["id"])
        assert sup.get(job["id"]) is not None, "a running job is not forgettable"
    finally:
        _cleanup(sup)


def test_terminal_rows_are_bounded(sup):
    reg = {}
    for i in range(30):
        jid = f"job-{i:08x}"
        reg[jid] = {"id": jid, "state": sbx_jobs.STATE_DONE, "exit_code": 0,
                    "finished_at": float(i), "promoted_at": float(i),
                    "pid": 4242, "deadline_at": float(i) + 60,
                    "command": "x"}
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps(reg))
    sup.reap()
    kept = sup.list_entries()
    assert len(kept) == 20
    # The OLDEST are the ones dropped.
    assert all(float(e["finished_at"]) >= 10 for e in kept)


# ---------------------------------------------------------------- hardening

def test_partial_exit_sentinel_is_not_read_as_success(sup):
    jid = "job-abcdef01"
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / f"{jid}.exit").write_text("")
    assert sup._read_exit(jid) is None
    (sup.host_dir / f"{jid}.exit").write_text("not-a-number")
    assert sup._read_exit(jid) is None
    (sup.host_dir / f"{jid}.exit").write_text("0\n")
    assert sup._read_exit(jid) == 0


def test_malformed_job_ids_are_rejected_before_touching_paths(sup):
    for bad in ("../../etc/passwd", "job-../x", "", None, "job-XYZ"):
        assert sbx_jobs.valid_job_id(bad) is False
        with pytest.raises(ValueError):
            sup._paths(bad)
    killed, msg = sup.cancel("../../etc/passwd")
    assert killed is False and "malformed" in msg


def test_registry_read_drops_malformed_rows(sup):
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps({
        "job-00000001": {"id": "job-00000001", "state": "running",
                         "pid": 4242, "deadline_at": time.time() + 60},
        "../evil": {"id": "../evil", "state": "running", "pid": 4242,
                    "deadline_at": time.time() + 60},
        "job-00000002": "not-a-dict",
        "job-00000003": {"id": "job-00000003", "state": "running",
                         "pid": 4242},                      # no deadline
        "job-00000004": {"id": "job-00000004", "state": "running",
                         "deadline_at": time.time() + 60},  # no pid
    }))
    ids = [e.get("id") for e in sup.list_entries()]
    assert ids == ["job-00000001"]


def test_corrupt_registry_is_survivable(sup):
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text("{not json")
    assert sup.list_entries() == []
    assert sup.reap() == []


def test_entry_from_a_previous_container_generation_is_lost_not_killed(sup):
    """A recreated container invalidates every pid; a same-numbered pid in
    the new container is an unrelated process and must never be signalled."""
    jid = "job-0000beef"
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps({jid: {
        "id": jid, "state": sbx_jobs.STATE_RUNNING, "pid": os.getpid(),
        "command": "x", "deadline_at": time.time() + 999,
        "promoted_at": time.time(), "container_id": "OLD-GENERATION",
    }}))
    sup.sandbox.container = type("C", (), {"id": "NEW-GENERATION"})()
    killed = []
    sup._kill_pgroup = lambda pid: killed.append(pid)
    changed = sup.reap()
    assert [e["state"] for e in changed] == [sbx_jobs.STATE_LOST]
    assert killed == [], "must not signal a pid from a dead generation"


def test_kill_switch_env_disables_promotion(monkeypatch):
    monkeypatch.setenv("GHOST_SANDBOX_JOBS", "0")
    assert sbx_jobs.jobs_enabled() is False
    monkeypatch.setenv("GHOST_SANDBOX_JOBS", "1")
    assert sbx_jobs.jobs_enabled() is True


@pytest.mark.parametrize("value,expected", [
    ("", 45 * 60.0),            # unset → default
    ("nonsense", 45 * 60.0),    # unparseable → default, not 0
    ("0", 45 * 60.0),           # below the floor → default, not "reap always"
    ("99999999", 45 * 60.0),    # above the ceiling → default
    ("120", 120.0),
])
def test_ttl_env_is_bounded(monkeypatch, value, expected):
    monkeypatch.setenv("GHOST_SANDBOX_JOB_TTL_S", value)
    assert sbx_jobs.job_ttl_s() == expected


def test_get_job_supervisor_is_memoised_and_degrades(tmp_path):
    assert sbx_jobs.get_job_supervisor(None) is None
    assert sbx_jobs.get_job_supervisor(object()) is None
    fake = FakeSandbox(tmp_path)
    first = sbx_jobs.get_job_supervisor(fake)
    assert first is not None
    assert sbx_jobs.get_job_supervisor(fake) is first


def test_log_tail_and_read_are_safe_on_a_missing_log(sup):
    assert sup.log_tail("job-00000009") == "(no output)"
    assert sup._read_log("job-00000009") == b""


# ------------------------------------------------- kill safety (§ review)

def test_kill_pgroup_refuses_pid_1_and_0(sup, monkeypatch):
    """`kill -- -1` means EVERY process the caller may signal, not "group 1" —
    verified in a live container: the shell issuing it did not survive to
    print its next line. A pid of 1 in a registry row would therefore destroy
    the whole container: every service, every other job, tor."""
    sent = []
    monkeypatch.setattr(sup, "_exec", lambda cmd, timeout=30: sent.append(cmd) or ("", 0))
    for bad in (1, 0, -1, "1", None, "garbage"):
        assert sup._kill_pgroup(bad) is False
    assert sent == [], f"a signal was sent for a forbidden pid: {sent}"


def _fake_proc(tmp_path, procs):
    """Build a synthetic /proc. `procs` maps pid -> (state, ppid, pgrp, sid).

    The layout mirrors the real one exactly, INCLUDING a comm containing
    spaces and parentheses — the reason both scripts parse after the LAST
    ')' rather than splitting fields.
    """
    root = tmp_path / "proc"
    root.mkdir(parents=True, exist_ok=True)
    for pid, (state, ppid, pgrp, sid) in procs.items():
        d = root / str(pid)
        d.mkdir(exist_ok=True)
        (d / "stat").write_text(
            f"{pid} (my (weird) proc) {state} {ppid} {pgrp} {sid} 0 0\n")
        (d / "io").write_text(
            "rchar: 100\nwchar: 200\nsyscr: 1\nread_bytes: 4096\n")
    return root


def _run_shell(script, proc_root):
    """Run a supervisor shell script with /proc redirected at the fixture."""
    body = script.replace("/proc/", f"{proc_root}/")
    return subprocess.run(["/bin/sh", "-c", body], capture_output=True,
                          text=True)


def test_the_kill_script_selects_exactly_the_jobs_processes(sup, tmp_path,
                                                            monkeypatch):
    """EXECUTES the generated shell against a synthetic /proc instead of
    grepping it for spelling. The two shells that carry this feature's core
    mechanism had ZERO executed coverage on this host (no /proc, and
    /bin/sh is bash rather than the container's dash), so a regression in
    either would have been pinned only by the words used to write it.

    `kill` is a shell BUILTIN, so it cannot be intercepted with a PATH shim —
    the selection is what this asserts, by swapping the signal for an echo.
    Which pids the script chooses IS the logic; `kill` itself is not."""
    procs = {
        7: ("S", 1, 7, 7),      # the job's session leader
        8: ("S", 7, 8, 7),      # child that changed pgroup (the `timeout` case)
        9: ("S", 8, 8, 7),      # grandchild
        10: ("S", 9, 10, 99),   # RE-SESSIONED but still ours by ancestry
        11: ("S", 1, 11, 11),   # UNRELATED session — must not be touched
        1: ("S", 0, 1, 1),      # init — must never be touched
    }
    proc_root = _fake_proc(tmp_path, procs)
    sent = {}
    monkeypatch.setattr(
        sup, "_exec",
        lambda cmd, timeout=30: (sent.setdefault("cmd", cmd), ("", 0))[1])
    monkeypatch.setattr(sup, "_probe", lambda pid: (False, 0))
    sup._kill_pgroup(7)

    inner = shlex.split(sent["cmd"])[2].replace("/proc/", f"{proc_root}/")
    inner = inner.replace('mine "$p" && kill -"$1" "$p" 2>/dev/null;',
                          'mine "$p" && echo "$p";')
    inner = inner.split("sig TERM;")[0] + "sig TERM"
    out = subprocess.run(["/bin/sh", "-c", inner], capture_output=True,
                         text=True)
    hit = set(out.stdout.split())

    assert "9" in hit, "a grandchild in the job's session must be signalled"
    assert "8" in hit, "a child that changed process group must be signalled"
    assert "10" in hit, (
        "a re-sessioned descendant still reachable by ppid must be signalled "
        "— session membership alone is not ancestor-closed")
    assert "11" not in hit, "an UNRELATED session must never be signalled"
    assert "1" not in hit, "init must never be signalled"
    assert hit == {"7", "8", "9", "10"}, f"selected the wrong set: {hit}"


def test_the_probe_script_sums_only_the_jobs_session(sup, tmp_path):
    """Same treatment for the I/O probe: run it against a synthetic /proc and
    check the arithmetic, not the spelling. Without this the entire I/O half
    of the promotion rule is unexecuted on this host."""
    procs = {
        7: ("S", 1, 7, 7),
        8: ("S", 7, 8, 7),
        9: ("Z", 8, 8, 7),      # a zombie member — counted, but not "alive"
        11: ("S", 1, 11, 11),   # another session — must NOT be summed
    }
    proc_root = _fake_proc(tmp_path, procs)
    captured = {}

    def _exec(cmd, timeout=30):
        captured["cmd"] = cmd
        inner = shlex.split(cmd)[2].replace("/proc/", f"{proc_root}/")
        r = subprocess.run(["/bin/sh", "-c", inner], capture_output=True,
                           text=True)
        return r.stdout, r.returncode

    sup._exec = _exec
    alive, io_chars = sup._probe(7)
    # 3 session members × (rchar 100 + wchar 200); pid 11 excluded.
    assert io_chars == 900, f"summed the wrong set: {io_chars}"
    assert alive is True, "a non-zombie member means the job is alive"

    # …and an all-zombie session is NOT alive.
    procs2 = {7: ("Z", 1, 7, 7), 8: ("Z", 7, 8, 7)}
    proc_root2 = _fake_proc(tmp_path / "z", procs2)

    def _exec2(cmd, timeout=30):
        inner = shlex.split(cmd)[2].replace("/proc/", f"{proc_root2}/")
        r = subprocess.run(["/bin/sh", "-c", inner], capture_output=True,
                           text=True)
        return r.stdout, r.returncode

    sup._exec = _exec2
    alive2, _io = sup._probe(7)
    assert alive2 is False, (
        "a session of nothing but zombies is finished, not running — the "
        "half of _pid_state that never executes on a host without /proc")


def test_a_kill_that_leaves_a_survivor_reports_false(sup, monkeypatch):
    """Verification is against the SESSION: a surviving grandchild must not
    read as a clean kill just because the leader is gone."""
    monkeypatch.setattr(sup, "_exec", lambda cmd, timeout=30: ("", 0))
    monkeypatch.setattr(sup, "_probe", lambda pid: (True, 10))   # still alive
    assert sup._kill_pgroup(4242) is False


def test_registry_rows_that_would_signal_the_container_are_dropped(sup):
    """The registry lives on the bind mount — INSIDE the sandbox, writable by
    the very commands this module supervises. A row is untrusted input."""
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps({
        "job-0000dea1": {"id": "job-0000dea1", "state": "running", "pid": 1,
                         "deadline_at": time.time() + 60, "command": "x"},
        "job-0000dea2": {"id": "job-0000dea2", "state": "running", "pid": 0,
                         "deadline_at": time.time() + 60, "command": "x"},
    }))
    assert sup.list_entries() == []
    killed = []
    sup._kill_pgroup = lambda pid: killed.append(pid) or True
    assert sup.reap() == []
    assert killed == []


def test_a_row_with_no_deadline_is_dropped_not_instantly_expired(sup):
    """`float(None or 0)` is 0 — "expired in 1970". A missing field must not
    mean "expire AND kill on the next sweep"."""
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps({
        "job-0000dea3": {"id": "job-0000dea3", "state": "running",
                         "pid": 4242, "command": "x"},
    }))
    killed = []
    sup._kill_pgroup = lambda pid: killed.append(pid) or True
    sup.reap()
    assert killed == [], "a malformed row must never trigger a kill"


def test_expiry_checks_the_container_generation_before_killing(sup):
    """A recreated container invalidates every pid, and fresh containers hand
    out LOW pids — so an expired row from a dead generation points at an
    unrelated process. services.py had to learn this in 2026-07."""
    jid = "job-0000ca11"
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps({jid: {
        "id": jid, "state": sbx_jobs.STATE_RUNNING, "pid": 4242,
        "command": "x", "deadline_at": time.time() - 1,      # expired
        "promoted_at": time.time(), "container_id": "OLD",
    }}))
    sup.sandbox.container = type("C", (), {"id": "NEW"})()
    killed = []
    sup._kill_pgroup = lambda pid: killed.append(pid) or True
    changed = sup.reap()
    assert [e["state"] for e in changed] == [sbx_jobs.STATE_LOST]
    assert killed == [], "an expired row from a dead generation must not kill"


def test_cancel_refuses_to_signal_a_dead_generation(sup):
    jid = "job-0000ca12"
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps({jid: {
        "id": jid, "state": sbx_jobs.STATE_RUNNING, "pid": 4242,
        "command": "x", "deadline_at": time.time() + 999,
        "promoted_at": time.time(), "container_id": "OLD",
    }}))
    sup.sandbox.container = type("C", (), {"id": "NEW"})()
    killed = []
    sup._kill_pgroup = lambda pid: killed.append(pid) or True
    ok, msg = sup.cancel(jid)
    assert ok is False and "recreated" in msg
    assert killed == []


def test_cancel_reports_false_when_the_process_survives(sup):
    """Never report a kill that did not happen — the caller marks its own row
    on this boolean, and a surviving process would go untracked."""
    jid = "job-0000ca13"
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps({jid: {
        "id": jid, "state": sbx_jobs.STATE_RUNNING, "pid": 4242,
        "command": "x", "deadline_at": time.time() + 999,
        "promoted_at": time.time(), "container_id": None,
    }}))
    sup._exec = lambda cmd, timeout=30: ("", 0)     # kill "works", pid alive
    sup._pid_state = lambda pid: True
    ok, msg = sup.cancel(jid)
    assert ok is False
    assert "still alive" in msg
    assert sup.get(jid)["state"] == sbx_jobs.STATE_RUNNING, (
        "a failed kill must leave the row RUNNING, not lie about it")


# ------------------------------------------- probe failure is not death

def test_an_infra_error_is_unknown_not_death(sup, monkeypatch):
    """`sandbox.execute` reports a wedged daemon as exit 1 with an
    [SANDBOX INFRA ERROR] body — indistinguishable from "that pid is gone" on
    the exit code alone. Reading a docker hiccup as DEATH deleted a live
    job's log and left its process running untracked."""
    monkeypatch.setattr(
        sup, "_exec",
        lambda cmd, timeout=30: ("[SANDBOX INFRA ERROR — not your code] x", 1))
    assert sup._pid_state(4242) is None
    assert sup._pid_alive(4242) is False        # strict view for kill checks
    assert sup._probe(4242) == (None, None)


def test_reap_leaves_a_row_running_when_the_probe_fails(sup):
    jid = "job-0000f00d"
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps({jid: {
        "id": jid, "state": sbx_jobs.STATE_RUNNING, "pid": 4242,
        "command": "x", "deadline_at": time.time() + 999,
        "promoted_at": time.time(), "container_id": None,
    }}))
    sup._pid_state = lambda pid: None            # probe inconclusive
    assert sup.reap() == []
    assert sup.get(jid)["state"] == sbx_jobs.STATE_RUNNING, (
        "a docker hiccup must not stop tracking a live process for good")


def test_promotion_survives_an_inconclusive_probe_at_the_budget(sup, monkeypatch):
    monkeypatch.setattr(sup, "_probe", lambda pid: (None, None))
    try:
        _out, code, job = sup.run(
            "sh -c 'while :; do echo x; sleep 0.2; done'",
            timeout=4, exec_kwargs=_kwargs(sup))
        assert job is not None, (
            "an unreadable probe must not be treated as a dead process")
        assert code == 0
    finally:
        _cleanup(sup)


# ------------------------------------------------------ the re-run guard

def test_an_identical_command_returns_the_running_job_instead_of_a_second_copy(sup):
    """The banner tells the model not to re-run; this makes it TRUE. Without
    it, a model that ignores the banner detaches a second copy of a heavy
    command competing for the same container CPU — and the strike counter
    that used to bound that loop is deliberately gone."""
    cmd = "sh -c 'while :; do echo x; sleep 0.2; done'"
    try:
        _o1, _c1, first = sup.run(cmd, timeout=1, exec_kwargs=_kwargs(sup))
        assert first is not None
        _o2, code2, second = sup.run(cmd, timeout=1, exec_kwargs=_kwargs(sup))
        assert second is not None
        assert second["id"] == first["id"], "the SAME job must come back"
        assert second["duplicate"] is True
        assert code2 == 0
        assert len(sup.list_entries()) == 1, "no second job was created"
    finally:
        _cleanup(sup)


def test_a_rewritten_script_is_not_the_job_still_running_the_old_one(sup):
    """The command for a script run (`python3 -u build.py`) is stable across
    every rewrite of build.py. Comparing on it alone meant a model that read
    the promotion banner, FIXED the script and re-ran it was handed the OLD
    job back — told "already running", given the old job's log as its output,
    and blocked from running the fix for up to the whole TTL."""
    cmd = "sh -c 'while :; do echo x; sleep 0.2; done'"
    try:
        _o, _c, first = sup.run(cmd, timeout=1, exec_kwargs=_kwargs(sup),
                                identity=f"{cmd}#hash-of-version-1")
        assert first is not None
        # Same command, DIFFERENT content → a different job.
        _o2, _c2, second = sup.run(cmd, timeout=1, exec_kwargs=_kwargs(sup),
                                   identity=f"{cmd}#hash-of-version-2")
        assert second is not None
        assert second["id"] != first["id"], (
            "a rewritten script must be allowed to run")
        assert not second.get("duplicate")
        # …and the SAME content still matches.
        _o3, _c3, third = sup.run(cmd, timeout=1, exec_kwargs=_kwargs(sup),
                                  identity=f"{cmd}#hash-of-version-2")
        assert third["id"] == second["id"] and third["duplicate"] is True
    finally:
        _cleanup(sup)


def test_the_execute_tool_folds_the_script_content_into_the_identity():
    """The supervisor cannot see the script; the tool has to supply this."""
    import inspect
    from ghost_agent.tools import execute as execute_mod
    src = inspect.getsource(execute_mod.tool_execute)
    assert "identity=f\"{cmd}#" in src
    assert "hashlib.sha1(clean_content" in src


def test_the_guard_only_matches_the_same_command_and_workdir(sup):
    try:
        _o, _c, first = sup.run("sh -c 'while :; do echo a; sleep 0.2; done'",
                                timeout=1, exec_kwargs=_kwargs(sup))
        assert first is not None
        _o2, _c2, other = sup.run("sh -c 'while :; do echo b; sleep 0.2; done'",
                                  timeout=1, exec_kwargs=_kwargs(sup))
        assert other is not None
        assert other["id"] != first["id"]
    finally:
        _cleanup(sup)


def test_a_finished_job_does_not_block_a_re_run(sup):
    """Only a RUNNING job matches — otherwise every completed long command
    would be un-runnable for as long as its row survived."""
    cmd = "sh -c 'echo once'"
    out1, code1, job1 = sup.run(cmd, timeout=30, exec_kwargs=_kwargs(sup))
    assert job1 is None and code1 == 0
    out2, code2, job2 = sup.run(cmd, timeout=30, exec_kwargs=_kwargs(sup))
    assert job2 is None and code2 == 0
    assert b"once" in out2


# ------------------------------------------------ hard ceiling + window

def test_the_runner_script_carries_a_last_resort_timeout(sup):
    """The classic path's in-container `timeout` survived the agent dying; the
    job path must keep an equivalent, or a deploy-by-kill mid-run leaves an
    immortal process with no row, no TTL and no reaper."""
    jid = "job-0000abcd"
    sup._write_script(jid, "sleep 1", 3300)
    body = (sup.host_dir / f"{jid}.cmd.sh").read_text()
    # budget+TTL PLUS the margin that keeps the TTL reaper ahead of it.
    assert f"timeout -k 5s {3300 + int(sbx_jobs._CEILING_MARGIN_S)}s" in body
    # Probed, never assumed: a host/image without coreutils still runs.
    assert "command -v timeout" in body


def test_the_progress_window_is_clamped_to_the_budget(monkeypatch):
    """A window >= the budget silently degrades the rail to "did it ever do
    anything", because no sample is ever older than the window.

    Calls production's `effective_window_s` — the previous two versions of
    this test each passed for an unrelated reason: the first because the
    wedge it ran did no I/O either way, the second because it reimplemented
    the arithmetic in its own body and never touched the code."""
    monkeypatch.setenv("GHOST_SANDBOX_JOB_PROGRESS_WINDOW_S", "3600")
    assert sbx_jobs.progress_window_s() == 3600.0
    assert sbx_jobs.effective_window_s(600) == 300.0, "half the budget"
    assert sbx_jobs.effective_window_s(60) == 30.0
    monkeypatch.setenv("GHOST_SANDBOX_JOB_PROGRESS_WINDOW_S", "120")
    assert sbx_jobs.effective_window_s(600) == 120.0, (
        "the configured window wins when it is the smaller of the two")
    assert sbx_jobs.effective_window_s(4) == 10.0, "the floor still applies"


def test_the_ceiling_leaves_room_for_the_ttl_reaper_to_win(sup):
    """The runner's last-resort `timeout` fires at budget+TTL; `deadline_at`
    is also budget+TTL — but the sweeper only polls every 60 s, so without a
    margin the ceiling usually wins and the job lands DONE/exit 124 instead of
    EXPIRED, telling the model its command FAILED."""
    from ghost_agent.main import _SANDBOX_JOB_REAP_EVERY_S
    assert sbx_jobs._CEILING_MARGIN_S > _SANDBOX_JOB_REAP_EVERY_S
    jid = "job-0000ce11"
    sup._write_script(jid, "sleep 1", 3300)
    body = (sup.host_dir / f"{jid}.cmd.sh").read_text()
    assert f"timeout -k 5s {3300 + int(sbx_jobs._CEILING_MARGIN_S)}s" in body


def test_the_kill_uses_the_form_dash_actually_accepts(sup, monkeypatch):
    """The container's /bin/sh is dash, whose builtin `kill` parses `--` as a
    pid: `kill -TERM -- -123` fails with "Illegal number: -" and sends
    NOTHING (measured in python:3.11-slim-bookworm). The group-kill half was
    therefore dead in production, and there was no plain-pid fallback."""
    sent = {}
    monkeypatch.setattr(sup, "_exec",
                        lambda cmd, timeout=30: (sent.setdefault("cmd", cmd),
                                                 ("", 0))[1])
    monkeypatch.setattr(sup, "_probe", lambda pid: (False, 0))
    sup._kill_pgroup(4242)
    cmd = sent["cmd"]
    assert "-- -$S" not in cmd, "dash rejects `--` before a negative pid"
    assert 'kill -"$1" -$S' in cmd, "the group kill must still be attempted"
    assert 'kill -"$1" $S' in cmd, (
        "and a plain-pid fallback, or a job whose /proc scan comes up empty "
        "is marked reaped while its process keeps running")


def test_expiry_probes_before_signalling(sup):
    """A generation match only says "same container" — not that this pid is
    still the job's process. An expired row whose process died without a
    sentinel would otherwise have a recycled pid's whole SESSION killed."""
    jid = "job-0000ee11"
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps({jid: {
        "id": jid, "state": sbx_jobs.STATE_RUNNING, "pid": 4242,
        "command": "x", "deadline_at": time.time() - 1,
        "promoted_at": time.time(), "container_id": None,
    }}))
    killed = []
    sup._kill_pgroup = lambda pid: killed.append(pid) or True
    sup._pid_state = lambda pid: False          # the process is already gone
    changed = sup.reap()
    assert [e["state"] for e in changed] == [sbx_jobs.STATE_EXPIRED]
    assert killed == [], "a dead pid must not be signalled at expiry"


def test_cleanup_paths_outside_the_sandbox_are_refused(sup, tmp_path):
    """⚠ SECURITY BOUNDARY. The registry is bind-mounted read-write INTO the
    container, so a sandboxed command can write `cleanup_paths` — and
    unlinking them unvalidated is an arbitrary HOST file-deletion primitive
    against anything the agent's uid can remove."""
    outside = tmp_path.parent / "precious_do_not_delete.txt"
    outside.write_text("secret")
    inside = sup.host_dir.parent / "scratch.txt"
    inside.write_text("scratch")
    escape = sup.host_dir.parent / ".." / outside.name

    safe = sup._safe_cleanup_paths([str(outside), str(inside), str(escape),
                                    "/etc/passwd", None, 42])
    assert safe == [inside], f"only in-sandbox files may be unlinked: {safe}"

    # …and the deletion path honours it.
    sup._cleanup_files("job-0000aa11",
                       entry={"cleanup_paths": [str(outside), str(inside)]})
    assert outside.exists(), "a file outside the sandbox must survive"
    assert not inside.exists()


def test_cleanup_paths_never_delete_a_directory(sup):
    victim = sup.host_dir.parent / "a_directory"
    victim.mkdir(exist_ok=True)
    assert sup._safe_cleanup_paths([str(victim)]) == []
    sup._cleanup_files("job-0000aa12", entry={"cleanup_paths": [str(victim)]})
    assert victim.is_dir()


def test_the_re_run_guard_declines_to_match_a_truncated_command(sup):
    """The stored command is capped, so comparing at that length is
    prefix-only: two different long commands sharing a prefix would collide
    and the second would never run while being told it was already running."""
    long_cmd = "sh -c 'echo " + ("x" * 2100) + "'"
    assert sup._find_running(long_cmd, "/workspace") is None
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps({"job-0000bb11": {
        "id": "job-0000bb11", "state": sbx_jobs.STATE_RUNNING, "pid": 4242,
        "command": long_cmd[:2000], "workdir": "/workspace",
        "deadline_at": time.time() + 999, "promoted_at": time.time(),
    }}))
    assert sup._find_running(long_cmd, "/workspace") is None, (
        "a prefix match must not be treated as the same command")


# ------------------------------------------------- caller-owned cleanup

def test_a_promoted_job_owns_the_callers_temp_file(sup, tmp_path):
    """The ephemeral script an `execute(content=…)` run is executing cannot be
    deleted while the job runs — so the supervisor takes ownership and
    unlinks it when the job ends. The path is persisted on the row, so a
    restart still cleans up."""
    victim = tmp_path / ".ephemeral_deadbeef.py"
    victim.write_text("print('x')")
    try:
        _o, _c, job = sup.run("sh -c 'while :; do echo x; sleep 0.2; done'",
                              timeout=1, exec_kwargs=_kwargs(sup),
                              cleanup_paths=[str(victim)])
        assert job is not None
        assert victim.exists(), "still running — the file must survive"
        assert sup.get(job["id"])["cleanup_paths"] == [str(victim)]
        sup.cancel(job["id"])
        assert not victim.exists(), "the job ended — the file must be gone"
    finally:
        _cleanup(sup)


# ------------------------------------ round-3 security boundary pins

def test_a_forged_exit_sentinel_is_rejected(sup):
    """The sentinel lives in a directory the supervised command can write.
    Without a nonce, one line — `echo 0 > /workspace/.jobs/<jid>.exit` —
    made `execute` return EXIT CODE 0 in 0.0s for a command that kept
    running, with no registry row, no TTL and no reaper. Any command that
    merely WIPES .jobs/ produced the same result by accident."""
    jid = "job-0000f0f0"
    sup._nonces[jid] = "deadbeefdeadbeef"
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / f"{jid}.exit").write_text("0\n")          # forged
    assert sup._read_exit(jid) is None, "an unauthenticated sentinel is not an exit"
    (sup.host_dir / f"{jid}.exit").write_text("deadbeefdeadbeef 7\n")
    assert sup._read_exit(jid) == 7, "the runner's own sentinel still lands"


def test_the_runner_script_quotes_the_nonce_back(sup):
    jid = sup._new_id()
    sup._write_script(jid, "true", 3300)
    body = (sup.host_dir / f"{jid}.cmd.sh").read_text()
    nonce = sup._nonces[jid]
    assert len(nonce) >= 16
    assert f'echo "{nonce} $_rc"' in body


def test_a_row_with_no_container_stamp_is_never_signalled(sup):
    """The registry is on the bind mount, so a supervised command can write
    a row naming ANY live pid. An unstamped row that read as "same
    generation" turned the TTL reaper into an arbitrary in-container
    session-kill primitive — measured live: a planted row killed an
    unrelated setsid'd service tree."""
    jid = "job-0000dd11"
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps({jid: {
        "id": jid, "state": sbx_jobs.STATE_RUNNING, "pid": 4242,
        "command": "planted", "deadline_at": 1,      # already expired
        "promoted_at": time.time(),                   # NO container_id
    }}))
    sup.sandbox.container = type("C", (), {"id": "GEN-1"})()
    killed = []
    sup._kill_pgroup = lambda pid: killed.append(pid) or True
    changed = sup.reap()
    assert [e["state"] for e in changed] == [sbx_jobs.STATE_LOST]
    assert killed == [], "an unstamped row must never be signalled"


def test_a_symlinked_jobs_dir_is_refused(sup, tmp_path):
    """Every file this module writes and deletes derives from host_dir, which
    is INSIDE the bind mount — so the sandbox can replace it with a symlink.
    Measured: that made the host write the runner script into, and delete
    files from, an arbitrary directory at the agent's uid."""
    victim = tmp_path.parent / "victim_dir"
    victim.mkdir(exist_ok=True)
    jobs = Path(sup.sandbox.host_workspace) / ".jobs"
    if jobs.exists():
        shutil.rmtree(jobs)
    jobs.symlink_to(victim, target_is_directory=True)
    with pytest.raises(RuntimeError, match="not a real directory"):
        _ = sup.host_dir


def test_a_non_finite_deadline_is_dropped(sup):
    """NaN/Infinity parse as floats and pass a `<= 0` test, so
    `time.time() >= deadline` is permanently False — an infinite TTL
    smuggled in as a number."""
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    for bad in ("nan", "Infinity", "1e400"):
        (sup.host_dir / "registry.json").write_text(json.dumps({
            "job-0000ff01": {"id": "job-0000ff01", "state": "running",
                             "pid": 4242, "deadline_at": bad,
                             "command": "x"}}))
        assert sup.list_entries() == [], f"deadline_at={bad!r} must be rejected"


def test_an_unknown_state_cannot_opt_a_row_out_of_reaping(sup):
    """A single trailing space made the sweep skip the row: its pid was
    never probed and its lifetime cap never fired."""
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps({
        "job-0000ff02": {"id": "job-0000ff02", "state": "done ",
                         "pid": 4242, "deadline_at": time.time() + 99,
                         "command": "x", "container_id": None}}))
    rows = sup.list_entries()
    assert rows and rows[0]["state"] == sbx_jobs.STATE_RUNNING, (
        "an unknown state must normalise to running so the cap still applies")


def test_trim_drops_the_row_it_names_not_the_one_a_field_points_at(sup):
    """`reg.pop(entry['id'])` aimed the delete at a DIFFERENT row: it
    destroyed a RUNNING job and its log, left the intended row in place — so
    the count never dropped and the next sweep ate another live job."""
    reg = {"job-11111111": {"id": "job-11111111", "state": "running",
                            "pid": 4242, "deadline_at": time.time() + 999,
                            "command": "live"}}
    for i in range(21):
        key = f"job-2000{i:04x}"
        reg[key] = {"id": "job-11111111",      # ← poisoned / desynced id
                    "state": sbx_jobs.STATE_DONE, "pid": 4242,
                    "deadline_at": time.time(), "finished_at": float(i),
                    "command": "old"}
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps(reg))
    sup._trim_terminal(reg)
    assert "job-11111111" in reg, "the LIVE row must survive the trim"
    assert len([e for e in reg.values()
                if e["state"] != sbx_jobs.STATE_RUNNING]) == 20


def test_a_probe_killed_by_its_own_timeout_is_unknown_not_dead(sup, monkeypatch):
    """Every probe is wrapped in `timeout -k 5s 15s` by the sandbox layer;
    its expiry surfaces as exit 124 with the generic "[SYSTEM ERROR]" line
    and NO infra marker. Reading that as death gave a live job exit 137 with
    its log deleted."""
    monkeypatch.setattr(
        sup, "_exec",
        lambda cmd, timeout=30: (
            "[SYSTEM ERROR]: Process failed (Exit 124) with no output.", 124))
    assert sup._pid_state(4242) is None
    assert sup._probe(4242) == (None, None)


# ------------------------------------------------- the docker.py seam

class _StubDocker:
    """Just enough DockerSandbox to exercise _execute_impl's branch: which
    run mechanism was chosen, and did the 2-tuple contract survive."""

    from ghost_agent.sandbox.docker import DockerSandbox as _real

    execute = _real.execute
    execute_promotable = _real.execute_promotable
    _execute_impl = _real._execute_impl
    supports_job_promotion = True

    def __init__(self, root):
        self.host_workspace = Path(root)
        self.container = None
        self.ran = []
        self.ready = []

    def ensure_running(self):
        pass

    def mark_ready(self):
        self.ready.append("ok")

    def invalidate_ready(self):
        self.ready.append("invalid")

    def _exec_run(self, cmd, deadline_s=None, **kwargs):
        self.ran.append(cmd)
        return _FakeExecResult(b"out", 0)

    def _spill_run_output(self, text):
        return None


def test_classic_execute_never_takes_the_job_path(tmp_path):
    sbx = _StubDocker(tmp_path)
    out, code = sbx.execute("echo hi", timeout=5)
    assert (out, code) == ("out", 0), "execute must stay a 2-tuple"
    assert sbx.ran and sbx.ran[0].startswith("timeout -k 5s 5s "), (
        "internal callers keep the kill-at-the-budget contract")


def test_promotable_execute_uses_the_job_supervisor(tmp_path, monkeypatch):
    sbx = _StubDocker(tmp_path)
    seen = {}

    def _fake_run(cmd, *, timeout, workdir=None, label=None, project_id=None,
                  exec_kwargs=None, cleanup_paths=None, identity=None):
        seen.update(cmd=cmd, timeout=timeout, workdir=workdir, label=label,
                    exec_kwargs=exec_kwargs, cleanup_paths=cleanup_paths)
        return b"partial", 0, {"id": "job-00000001"}

    monkeypatch.setattr(sbx_jobs, "get_job_supervisor",
                        lambda mgr: type("S", (), {"run": staticmethod(_fake_run)})())
    out, code, job = sbx.execute_promotable("yt-dlp x", timeout=42,
                                            label="dl", project_id="p")
    assert job == {"id": "job-00000001"}
    assert code == 0 and out == "partial"
    # The raw command goes to the supervisor — NOT wrapped in `timeout`,
    # which would kill the process before it could ever be promoted.
    assert seen["cmd"] == "yt-dlp x"
    assert "timeout -k" not in seen["cmd"]
    assert seen["timeout"] == 42
    assert seen["exec_kwargs"]["workdir"] == "/workspace"
    assert sbx.ran == [], "no classic exec should have run"


def test_kill_switch_sends_promotable_runs_back_down_the_classic_path(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_SANDBOX_JOBS", "0")
    sbx = _StubDocker(tmp_path)
    out, code, job = sbx.execute_promotable("yt-dlp x", timeout=7)
    assert job is None
    assert code == 0
    assert sbx.ran and sbx.ran[0].startswith("timeout -k 5s 7s ")


def test_promoted_run_with_no_output_is_not_reported_as_an_error(
        tmp_path, monkeypatch):
    """A silent downloader promoted on file growth has printed nothing; the
    empty-output error sentinel must not brand that a failure."""
    sbx = _StubDocker(tmp_path)
    monkeypatch.setattr(
        sbx_jobs, "get_job_supervisor",
        lambda mgr: type("S", (), {"run": staticmethod(
            lambda *a, **k: (b"", 0, {"id": "job-00000002"}))})())
    out, code, job = sbx.execute_promotable("yt-dlp -q x", timeout=5)
    assert job is not None
    assert "SYSTEM ERROR" not in out
    assert out == "(no output yet)"


def test_an_interrupt_during_supervise_still_kills_the_launched_process(
        sup, monkeypatch):
    """§4BW CRITICAL-2. `run()`'s launch→registration guard was
    `except Exception`, but the window it protects is dominated by
    `_supervise`'s poll-loop sleep — so an interrupt (KeyboardInterrupt from
    a Ctrl-C, a pytest timeout, a killed mutation batch) lands there far more
    often than an ordinary Exception, and BaseException slipped straight past
    the kill: an immortal detached process with no row and no reaper. That is
    the source of the week-old busy-loop orphans on this box.

    Mutation witness: with `except Exception` the KeyboardInterrupt
    propagates and `_kill_pgroup` is NEVER called — this test fails."""
    killed = []
    monkeypatch.setattr(sup, "_launch", lambda jid, kw: 4242)
    monkeypatch.setattr(sup, "_read_pid", lambda jid: 4242)
    monkeypatch.setattr(sup, "_kill_pgroup",
                        lambda pid: killed.append(pid) or True)
    monkeypatch.setattr(sup, "_cleanup_files",
                        lambda *a, **k: None)

    def _interrupt(*a, **k):
        raise KeyboardInterrupt("Ctrl-C during the quiet poll phase")

    monkeypatch.setattr(sup, "_supervise", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        sup.run("while :; do sleep 1; done", timeout=5.0,
                workdir="/workspace")

    assert killed == [4242], (
        "an interrupted run left its detached process group alive — this is "
        "how the immortal orphans are minted")
