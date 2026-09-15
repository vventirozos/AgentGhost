"""§4GV (2026-09-14): the jobs that outlived their tests by sixteen days.

Four `sh -c 'while :; do echo …; sleep 0.2; done'` loops were found alive on
this machine, reparented to init, the oldest started 2026-08-29 — each
burning a tenth of a core since. They are sandbox-job fixtures: the suite
spawns real host processes through a `setsid` shim so the job is its own
process-group leader, and a test that dies before its `_cleanup` — a crash, a
timeout, an interrupted run — leaves the group orphaned with nothing left
that knows it exists. §4GQ taught `_cleanup` to sweep every registry row
rather than only the RUNNING ones; that stops new leaks from tests that
finish, and reaches nothing that escaped before it existed.

⚠ The obvious sweep — kill anything whose command line looks like a busy
loop — is "guard the thing, not a proxy" with the process table for a blast
radius. The reaper touches ONLY pids the shim recorded, and only while the
live process still carries the argv recorded with it.

The world each pin fails in: a tree where the shim stops recording, where the
reaper kills on the pid alone (so a reused pid is a stranger's death
sentence), or where it fires on a process nothing recorded.
"""
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from tests.conftest import (JOB_REGISTRY, _owner_is_alive, _owner_pid,
                            _same_process, reap_abandoned_registries,
                            reap_detached_jobs)


def _session_sweep_row(pid: int, argv: list) -> None:
    """Put a pid we spawned into the registry the session fixture reaps.

    ⚠ CALLED BEFORE ANY ASSERTION, from a pid WE captured — not from the
    shim's row. The shim's row is the thing under test; a pin that exercises
    a tree where the shim records nothing gets an `AssertionError` out of the
    helper, and the process it already spawned is then orphaned with no id
    anywhere. That is not hypothetical either: running the §4GV battery left
    a `session-registry-probe` loop on this machine, spawned by the mutant
    "the shim records nothing", found by the next sweep.
    """
    try:
        with open(JOB_REGISTRY, "a") as fh:
            fh.write(json.dumps({"pid": int(pid), "pgid": int(pid),
                                 "argv": argv, "at": time.time()}) + "\n")
    except OSError:
        pass


def _spawn_recorded(tmp_path, registry, marker, nap="0.2"):
    """Spawn a detached loop through the real shim, with the registry env
    var pointed at `registry` — i.e. exactly how the suite spawns jobs.

    The wrapper shell echoes `$!`, so this helper knows the pid it created
    even when the shim under test records nothing: a test that spawns a real
    detached process must assume it will not reach its own cleanup, and a
    HELPER that spawns one must assume its caller never will.
    """
    from tests.test_sandbox_job_promotion import _SETSID_SHIM
    shim = tmp_path / "_setsid_shim.py"
    shim.write_text(_SETSID_SHIM)
    env = dict(os.environ, GHOST_TEST_JOB_REGISTRY=str(registry))
    inner = f"while :; do echo {marker}; sleep {nap}; done"
    # Backgrounded through a throwaway shell that exits immediately, so the
    # loop is orphaned onto init with no parent left holding it — the exact
    # shape of the four strays. A `run(...)` without `&` would block forever:
    # the shim EXECs into the loop. `echo $!` is how we keep a handle on it.
    proc = subprocess.run(
        f"{sys.executable} {shim} sh -c '{inner}' >/dev/null 2>&1 & echo $!",
        shell=True, env=env, cwd=str(tmp_path), timeout=30,
        capture_output=True, text=True)
    spawned = int((proc.stdout or "").strip() or 0)
    assert spawned, "the wrapper shell did not report a pid"
    _session_sweep_row(spawned, ["sh", "-c", inner])
    for _ in range(50):                      # the shim writes before exec
        if registry.exists() and registry.read_text().strip():
            break
        time.sleep(0.1)
    # Read defensively: a shim that records nothing leaves NO FILE, and a
    # `FileNotFoundError` here would hide the condition the caller is
    # actually testing behind a plumbing error.
    text = registry.read_text() if registry.exists() else ""
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    assert rows, "the shim recorded nothing — the reaper has no provenance"
    return rows[-1]


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _group_members(pgid) -> list:
    out = subprocess.run(["pgrep", "-g", str(pgid)],
                         capture_output=True, text=True)
    return [int(x) for x in out.stdout.split()]


def test_THIS_FILES_OWN_SPAWNS_ARE_IN_THE_SESSION_REGISTRY(tmp_path):
    """The file that documents the leak must not be the one that leaks.

    Each pin spawns into its own `tmp_path` registry, which the session
    fixture never reads — so a pin that dies before its `finally` orphans a
    real loop forever. It happened: the first, failing run of the pin below
    left a `reaper-probe` loop with ppid 1, found seven hours later by a
    routine sweep. The helper now records into the session registry as well,
    and this pin fails in a tree where someone removes that.
    """
    registry = tmp_path / "jobs.jsonl"
    before = (JOB_REGISTRY.read_text() if JOB_REGISTRY.exists() else "")
    row = _spawn_recorded(tmp_path, registry, "session-registry-probe")
    try:
        after = JOB_REGISTRY.read_text()
        assert str(row["pid"]) in after and str(row["pid"]) not in before, (
            "a spawn this file made is invisible to the session reaper — a "
            "test that dies before its cleanup leaks it permanently")
    finally:
        try:
            os.killpg(int(row["pgid"]), signal.SIGKILL)
        except OSError:
            pass


def test_a_spawn_whose_HELPER_fails_is_still_reachable(tmp_path, monkeypatch):
    """The leak that the §4GV battery itself produced.

    Under the mutant "the shim records nothing", `_spawn_recorded` raises at
    its own assertion — after the process exists and before the caller has a
    row to kill. The pid is captured from the wrapper shell now, and written
    to the session registry BEFORE that assertion, so the sweep can still
    reach it. This pin fails in the tree where that registration happens
    after (or instead of) nothing.
    """
    import tests.test_sandbox_job_promotion as sjp
    blind = sjp._SETSID_SHIM.replace(
        "_reg = os.environ.get('GHOST_TEST_JOB_REGISTRY')\n", "_reg = None\n")
    assert blind != sjp._SETSID_SHIM
    monkeypatch.setattr(sjp, "_SETSID_SHIM", blind)
    before = JOB_REGISTRY.read_text() if JOB_REGISTRY.exists() else ""
    with pytest.raises(AssertionError, match="recorded nothing"):
        _spawn_recorded(tmp_path, tmp_path / "jobs.jsonl", "blind-shim-probe")
    rows = [json.loads(l) for l in JOB_REGISTRY.read_text().splitlines()
            if l.strip() and l not in before.splitlines()]
    mine = [r for r in rows if "blind-shim-probe" in " ".join(r.get("argv") or [])]
    assert mine, ("the helper raised and left a live process with its pid "
                  "written nowhere — an orphan by construction")
    pid = mine[-1]["pid"]
    try:
        assert _alive(pid)
        assert reap_detached_jobs(JOB_REGISTRY) >= 1
        for _ in range(50):
            if not _alive(pid):
                break
            time.sleep(0.1)
        assert not _alive(pid), "the session sweep could not reach it"
    finally:
        try:
            os.killpg(int(pid), signal.SIGKILL)
        except OSError:
            pass


def test_a_job_that_outlives_its_test_is_reaped(tmp_path):
    """End to end: spawn a detached loop the way the suite does, walk away
    from it the way a crashed test does, and let the session reaper find it
    from the record alone."""
    registry = tmp_path / "jobs.jsonl"
    row = _spawn_recorded(tmp_path, registry, "reaper-probe")
    assert _alive(row["pid"]), "the probe died on its own — it proves nothing"
    try:
        assert reap_detached_jobs(registry) == 1
        for _ in range(50):
            if not _alive(row["pid"]):
                break
            time.sleep(0.1)
        assert not _alive(row["pid"]), "the recorded job survived the reaper"
        assert not registry.exists(), "the registry was not cleared"
    finally:
        try:
            os.killpg(int(row["pgid"]), signal.SIGKILL)
        except OSError:
            pass


def test_the_whole_group_dies_not_just_the_leader(tmp_path):
    """The job is a shell loop; its `sleep` is a separate process in the same
    group. Signalling the leader alone leaves that child orphaned and
    running — which is most of what "sixteen days" was made of."""
    registry = tmp_path / "jobs.jsonl"
    row = _spawn_recorded(tmp_path, registry, "group-probe", nap="120")
    try:
        for _ in range(50):                  # wait for the `sleep` to exist
            if len(_group_members(row["pgid"])) > 1:
                break
            time.sleep(0.1)
        assert len(_group_members(row["pgid"])) > 1, (
            "the job never had a child — this pin cannot distinguish")
        assert reap_detached_jobs(registry) == 1
        for _ in range(50):
            if not _group_members(row["pgid"]):
                break
            time.sleep(0.1)
        assert not _group_members(row["pgid"]), (
            "the leader died but its group did not")
    finally:
        try:
            os.killpg(int(row["pgid"]), signal.SIGKILL)
        except OSError:
            pass


def test_a_reused_pid_is_not_a_death_sentence(tmp_path):
    """The identity check is the whole safety story. Pids are recycled, and a
    row whose pid now belongs to something else must be dropped, not acted
    on. The stand-in for "something else" is a real live process this suite
    did not spawn as a job."""
    registry = tmp_path / "jobs.jsonl"
    stranger = subprocess.Popen(["sleep", "120"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    try:
        registry.write_text(json.dumps({
            "pid": stranger.pid, "pgid": stranger.pid,
            "argv": ["sh", "-c", "while :; do echo gone; sleep 0.2; done"],
            "at": time.time()}) + "\n")
        assert reap_detached_jobs(registry) == 0
        time.sleep(0.3)
        assert stranger.poll() is None, (
            "a process that merely inherited a recorded pid was killed")
    finally:
        stranger.kill()
        stranger.wait(timeout=10)


def test_nothing_unrecorded_is_ever_touched(tmp_path):
    """A busy loop this suite did not record is somebody else's process."""
    registry = tmp_path / "jobs.jsonl"
    proc = subprocess.Popen(
        ["sh", "-c", "while :; do echo stranger; sleep 0.2; done"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        registry.write_text("")             # recorded: nothing
        assert reap_detached_jobs(registry) == 0
        assert proc.poll() is None, "an unrecorded process was killed"
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_a_missing_or_corrupt_registry_is_not_an_error(tmp_path):
    assert reap_detached_jobs(tmp_path / "nope.jsonl") == 0
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    assert reap_detached_jobs(bad) == 0


@pytest.mark.parametrize("argv,live,expect", [
    (["sh", "-c", "while :; do echo x; sleep 0.2; done"],
     "sh -c while :; do echo x; sleep 0.2; done", True),
    # `ps` drops the quotes the shell kept — the comparison must too
    (["sh", "-c", "echo 'a b'"], 'sh -c echo "a b"', True),
    (["sh", "-c", "echo x"], "sh -c echo y", False),
    ([], "sh -c echo x", False),
    (["sh", "-c", "echo x"], "", False),
])
def test_the_identity_check_needs_both_factors(argv, live, expect):
    assert _same_process(argv, live) is expect


def test_the_registry_is_named_by_its_OWNER_and_is_discoverable():
    """Leftovers from a PREVIOUS run are the case that motivated this, so the
    name has to be findable by a later run — and it has to identify its owner,
    so a sibling worker's live registry can be told apart from an abandoned
    one."""
    import fnmatch
    from tests.conftest import JOB_REGISTRY_GLOB
    assert fnmatch.fnmatch(JOB_REGISTRY.name, JOB_REGISTRY_GLOB), (
        "this run's registry is not found by the sweep's own glob")
    assert _owner_pid(JOB_REGISTRY) == os.getpid()
    assert _owner_is_alive(JOB_REGISTRY)
    assert os.environ.get("GHOST_TEST_JOB_REGISTRY") == str(JOB_REGISTRY), (
        "the session fixture did not point the shim at the registry")


class TestTheSweepsAreScopedDifferently:
    """§4GZ. `tests/test_sandbox_job_promotion.py` writes rows for jobs that
    are STILL RUNNING — measured, four of them within four seconds of that
    file starting. With one shared registry and a teardown that swept it, any
    xdist worker finishing first would SIGKILL another worker's live jobs and
    then unlink the rows for the ones it could not reach. The world these
    pins fail in is that tree."""

    def test_a_live_siblings_registry_is_never_touched(self, tmp_path,
                                                       monkeypatch):
        import tests.conftest as cf
        sibling = tmp_path / f"ghost-test-detached-jobs-{os.getpid()}.jsonl"
        proc = subprocess.Popen(["sleep", "120"], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            sibling.write_text(json.dumps({
                "pid": proc.pid, "pgid": proc.pid,
                "argv": ["sleep", "120"], "at": time.time()}) + "\n")
            monkeypatch.setattr(cf.tempfile, "gettempdir", lambda: str(tmp_path))
            monkeypatch.setattr(cf, "JOB_REGISTRY", tmp_path / "not-mine.jsonl")
            assert cf.reap_abandoned_registries() == 0, (
                "a registry owned by a LIVE session was reaped")
            assert proc.poll() is None
            assert sibling.exists(), "a live sibling's rows were destroyed"
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_an_abandoned_registry_IS_swept(self, tmp_path, monkeypatch):
        import tests.conftest as cf
        dead = 999_999                       # no such pid
        assert not _owner_is_alive(tmp_path / f"x-{dead}.jsonl")
        abandoned = tmp_path / f"ghost-test-detached-jobs-{dead}.jsonl"
        proc = subprocess.Popen(["sleep", "120"], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            abandoned.write_text(json.dumps({
                "pid": proc.pid, "pgid": proc.pid,
                "argv": ["sleep", "120"], "at": time.time()}) + "\n")
            monkeypatch.setattr(cf.tempfile, "gettempdir", lambda: str(tmp_path))
            monkeypatch.setattr(cf, "JOB_REGISTRY", tmp_path / "not-mine.jsonl")
            assert cf.reap_abandoned_registries() == 1
            for _ in range(50):
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
            assert proc.poll() is not None, "an inherited stray survived"
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)

    def test_the_owner_question_is_asked_of_the_pid_not_the_clock(self):
        """The obvious substitute — "is this file older than N minutes" — is
        wrong in both directions: a live worker that has not spawned a job in
        N minutes looks abandoned, and two back-to-back runs look live."""
        assert _owner_pid("ghost-test-detached-jobs-4242.jsonl") == 4242
        assert _owner_pid("ghost-test-detached-jobs.jsonl") == 0
        assert _owner_is_alive(f"x-{os.getpid()}.jsonl")
        assert not _owner_is_alive("x-999999.jsonl")
        assert not _owner_is_alive("x-notanumber.jsonl")


def test_the_fixture_sweeps_the_right_SET_at_each_end(monkeypatch, tmp_path):
    """§4GZ. The two ends reap different sets, and both halves were pinned
    only through the helpers — so a mutant that made teardown sweep
    everything, and one that dropped the abandoned-registry scan from the
    start, both SURVIVED the first battery pass. The wiring is the thing that
    can regress ([[fix-is-the-least-reviewed-code]]), so this drives the real
    fixture function and watches three registries at once:

      mine      — reaped at the start AND at the end
      abandoned — owner pid dead: reaped at the START
      sibling   — owner pid alive: NEVER touched, at either end
    """
    import tests.conftest as cf
    mine = tmp_path / f"ghost-test-detached-jobs-{os.getpid()}.jsonl"
    abandoned = tmp_path / "ghost-test-detached-jobs-999999.jsonl"
    sibling = tmp_path / f"ghost-test-detached-jobs-{os.getppid()}.jsonl"
    procs = {}
    for name, path in (("mine", mine), ("abandoned", abandoned),
                       ("sibling", sibling)):
        pr = subprocess.Popen(["sleep", "120"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        procs[name] = pr
        path.write_text(json.dumps({
            "pid": pr.pid, "pgid": pr.pid, "argv": ["sleep", "120"],
            "at": time.time()}) + "\n")
    monkeypatch.setenv("GHOST_TEST_JOB_REGISTRY", str(mine))
    monkeypatch.setattr(cf.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(cf, "JOB_REGISTRY", mine)
    try:
        gen = cf._reap_detached_sandbox_jobs.__wrapped__()
        next(gen)                                    # everything up to yield
        for _ in range(50):
            if procs["mine"].poll() is not None and \
                    procs["abandoned"].poll() is not None:
                break
            time.sleep(0.1)
        assert procs["mine"].poll() is not None, "this run's own rows survived"
        assert procs["abandoned"].poll() is not None, (
            "a registry whose owning session is DEAD was not swept at the "
            "start — the inherited stray is the case that motivated all this")
        assert procs["sibling"].poll() is None, (
            "a live sibling worker's job was killed at session start")

        # teardown: never a LIVE sibling's registry (the abandoned sweep
        # runs at both ends now — safe because of the owner guard, not
        # because of the timing)
        pr = subprocess.Popen(["sleep", "120"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        procs["sibling2"] = pr
        sibling.write_text(json.dumps({
            "pid": pr.pid, "pgid": pr.pid, "argv": ["sleep", "120"],
            "at": time.time()}) + "\n")
        list(gen)
        time.sleep(0.5)
        assert pr.poll() is None, (
            "teardown killed a LIVE sibling's job — with `-n 6 --dist "
            "loadfile` that is another worker mid-test")
        assert sibling.exists(), "a live sibling's rows were destroyed"
    finally:
        for pr in procs.values():
            if pr.poll() is None:
                pr.kill()
            pr.wait(timeout=10)


def test_the_session_reaps_before_it_runs_anything(monkeypatch, tmp_path):
    """The four strays predate the reaper: nothing this run does could have
    recorded them, and every later run inherits them. So the sweep has to
    happen at session START, not only at teardown — driven here through the
    real fixture function, against a registry of our own."""
    from tests import conftest as cf
    registry = tmp_path / "session.jsonl"
    monkeypatch.setenv("GHOST_TEST_JOB_REGISTRY", str(registry))
    monkeypatch.setattr(cf, "JOB_REGISTRY", registry)
    row = _spawn_recorded(tmp_path, registry, "previous-run")
    assert _alive(row["pid"])
    try:
        gen = cf._reap_detached_sandbox_jobs.__wrapped__()
        next(gen)                            # everything up to the yield
        for _ in range(50):
            if not _alive(row["pid"]):
                break
            time.sleep(0.1)
        assert not _alive(row["pid"]), (
            "a job left by an earlier run survived into this one")
        assert os.environ["GHOST_TEST_JOB_REGISTRY"] == str(registry), (
            "the fixture did not point the shim at the registry it reaps")
        list(gen)                            # and the teardown half
    finally:
        try:
            os.killpg(int(row["pgid"]), signal.SIGKILL)
        except OSError:
            pass
