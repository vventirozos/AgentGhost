"""Boot-dependency guards for the host service topology (2026-08-21).

After the 2026-08-21 reboot the agent was down for ~20 minutes in a launchd
respawn loop. Two ops scripts now carry the logic that prevents a repeat, and
this module pins BOTH by executing them — not by grepping their source.

Why executed, not text-asserted: a `assert "nc -z" in src` pin passes for a
script whose loop logic is inverted, whose bound never fires, or whose stale-
lock branch deletes a LIVE postmaster's lock file. Every invariant below is a
behaviour that can silently invert while every plausible token still appears
in the file, so each test runs the real code path and asserts on what happened.

The two failures being pinned:

  1. tor had NO autostart, so `--mandatory-tor` (fail-closed) aborted boot on
     every respawn. `bin/start-ghost-agent.sh` now WAITS for :9050 rather than
     racing it. The regression to fear is the wait becoming a hang (no bound)
     or a no-op (bound of 0 / inverted test), both of which look fine in diff.

  2. postgres wedged forever on a stale `postmaster.pid` whose PID had been
     reused across the reboot by an unrelated process (AirPlayUIAgent). The
     regression to fear is the opposite error: a wrapper that deletes the lock
     unconditionally, which would let a SECOND postmaster start against a live
     data directory. The "live postgres is left alone" case is the important
     half and is tested with a real running process, not a mock.

Both scripts live at the ops-script location (/Users/vasilis/Data/AI/bin,
outside this repo), so every test SKIPS when they are not deployed — same
convention as tests/test_remote_serve_scripts.py.
"""

import os
import re
import shutil
import socket
import stat
import subprocess
import tempfile

import pytest

OPS_BIN = os.environ.get("GHOST_OPS_BIN", "/Users/vasilis/Data/AI/bin")
AGENT_LAUNCHER = os.path.join(OPS_BIN, "start-ghost-agent.sh")
POSTGRES_LAUNCHER = os.path.join(OPS_BIN, "start-postgres.sh")
ORBSTACK_LAUNCHER = os.path.join(OPS_BIN, "start-orbstack-engine.sh")

BASH = "/bin/bash"


def _read_or_skip(path):
    if not os.path.exists(path):
        pytest.skip(f"ops script not deployed at {path}")
    with open(path, "r") as f:
        return f.read()


def _free_port():
    """A port nothing is listening on — for the 'dependency is down' case."""
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


# ---------------------------------------------------------------------------
#  1. The Tor boot gate in start-ghost-agent.sh
# ---------------------------------------------------------------------------

def _extract_tor_gate():
    """Pull the tor-wait block out of the launcher so it can be RUN alone.

    The launcher's tail is `exec python -m src.ghost_agent.main ...`, so the
    file cannot be executed wholesale in a test. The block is delimited by its
    first assignment (TOR_HOST=) and the `fi` that closes the up/down report.
    """
    src = _read_or_skip(AGENT_LAUNCHER)
    m = re.search(r"^TOR_HOST=.*?^fi$", src, re.DOTALL | re.MULTILINE)
    if m is None:
        pytest.fail(
            "could not locate the tor-wait block in start-ghost-agent.sh — "
            "if it was intentionally restructured, update this extractor; do "
            "NOT delete the test (the gate is what stops the respawn loop)."
        )
    block = m.group(0)
    # Anti-vacuity: a truncated or empty extraction must fail the test rather
    # than let every assertion below pass against a near-empty script.
    assert "nc" in block and "TOR_WAIT_MAX" in block, (
        f"extracted block looks wrong ({len(block)} bytes):\n{block[:400]}"
    )
    return block


def _run_gate(block, *, port, wait_max, timeout=60):
    script = block.replace('TOR_PORT="9050"', f'TOR_PORT="{port}"')
    assert f'TOR_PORT="{port}"' in script, "port substitution failed"
    return subprocess.run(
        [BASH, "-c", script],
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "GHOST_TOR_WAIT_MAX": str(wait_max)},
    )


class TestTorBootGate:
    def test_proceeds_immediately_when_port_is_open(self):
        """The gate must not add latency when tor is already up."""
        block = _extract_tor_gate()
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            r = _run_gate(block, port=port, wait_max=60, timeout=30)
        finally:
            listener.close()

        assert r.returncode == 0, r.stderr
        assert "Tor SOCKS is up." in r.stdout, r.stdout
        # The distinguishing assertion: it must NOT have entered the wait loop.
        assert "Waiting for Tor" not in r.stdout, (
            "gate slept even though the port was open — the liveness test is "
            f"inverted:\n{r.stdout}"
        )
        assert "still unreachable" not in r.stderr

    def test_waits_instead_of_falling_straight_through(self):
        """The whole point: a down dependency must WAIT, not return instantly.

        Before this gate existed the launcher exec'd immediately and main.py
        raised `mandatory-tor: Tor proxy unreachable at boot` seconds later,
        which KeepAlive turned into a respawn loop.
        """
        block = _extract_tor_gate()
        r = _run_gate(block, port=_free_port(), wait_max=6, timeout=45)

        assert r.returncode == 0, r.stderr
        assert "Waiting for Tor" in r.stdout, (
            f"gate did not wait for a DOWN port:\n{r.stdout}\n{r.stderr}"
        )
        # It reports progress against the bound rather than looping silently.
        assert "/6s" in r.stdout, r.stdout

    def test_wait_is_bounded_and_falls_through_loudly(self):
        """Bounded, so a permanently-dead tor stays VISIBLE as a failure.

        An unbounded wait would convert a loud fail-closed abort into a daemon
        that sits in `state = running` forever with no listener — strictly
        worse than the crash loop it replaced.
        """
        block = _extract_tor_gate()
        r = _run_gate(block, port=_free_port(), wait_max=4, timeout=45)

        assert r.returncode == 0, "the gate must fall through, not abort"
        assert "still unreachable" in r.stderr, r.stderr
        # It must NOT claim success when the port never answered.
        assert "Tor SOCKS is up." not in r.stdout, (
            f"gate reported tor up while the port was dead:\n{r.stdout}"
        )
        # The operator gets the next command to run, not just a complaint.
        assert "launchctl print system/com.local.tor" in r.stderr

    def test_gate_runs_before_the_agent_is_exec_d(self):
        """Ordering pin: waiting AFTER exec would be waiting after the crash."""
        src = _read_or_skip(AGENT_LAUNCHER)
        gate_at = src.index("TOR_HOST=")
        # rindex, not index: the launcher's `--help` fast path execs the same
        # module near the top and legitimately precedes the gate (it must —
        # `--help` should not wait on tor). The REAL launch is the last one.
        exec_at = src.rindex('exec "$PY" -m src.ghost_agent.main')
        assert gate_at < exec_at, "tor gate must precede the exec"
        # Pin the two-exec structure so this stops being a silent assumption.
        assert src.count('exec "$PY" -m src.ghost_agent.main') == 2, (
            "launcher exec count changed — re-check which one this pins"
        )
        assert "--help" in src[src.index('exec "$PY" -m src.ghost_agent.main'):][:80]

    def test_uses_an_absolute_nc_path(self):
        """launchd hands a daemon PATH=/usr/bin:/bin:/usr/sbin:/sbin.

        A bare `nc` happens to resolve there, but this pins the absolute path
        so a future edit cannot reach for a /opt/homebrew tool that exists in
        an interactive shell and is missing under launchd — a class of bug
        that only ever shows up at boot.
        """
        block = _extract_tor_gate()
        assert "/usr/bin/nc" in block
        assert os.path.exists("/usr/bin/nc")


# ---------------------------------------------------------------------------
#  2. The stale-lock recovery in start-postgres.sh
# ---------------------------------------------------------------------------

def _stage_postgres_wrapper(tmp_path):
    """Copy the wrapper with PGBIN/PGDATA pointed at throwaway test doubles.

    PGBIN becomes a stub that records the fact it was reached, so "did the
    wrapper actually go on to start postgres?" is observable without running a
    real postmaster.
    """
    src = _read_or_skip(POSTGRES_LAUNCHER)

    pgdata = tmp_path / "pgdata"
    pgdata.mkdir()
    marker = tmp_path / "started"
    stub = tmp_path / "postgres-stub"
    stub.write_text(f'#!/bin/bash\necho "$@" > "{marker}"\nexit 0\n')
    stub.chmod(0o755)

    staged = src
    staged = re.sub(r'^PGBIN=.*$', f'PGBIN="{stub}"', staged, count=1, flags=re.MULTILINE)
    staged = re.sub(r'^PGDATA=.*$', f'PGDATA="{pgdata}"', staged, count=1, flags=re.MULTILINE)
    assert str(stub) in staged and str(pgdata) in staged, "staging substitution failed"

    path = tmp_path / "start-postgres-staged.sh"
    path.write_text(staged)
    path.chmod(0o755)
    return path, pgdata, marker


def _run_wrapper(path):
    return subprocess.run([BASH, str(path)], capture_output=True, text=True, timeout=30)


class TestPostgresStaleLock:
    def test_starts_clean_when_no_lock_present(self, tmp_path):
        script, pgdata, marker = _stage_postgres_wrapper(tmp_path)
        r = _run_wrapper(script)
        assert r.returncode == 0, r.stderr
        assert marker.exists(), "postgres was never reached"
        assert f"-D {pgdata}" in marker.read_text()

    def test_removes_lock_whose_pid_is_gone(self, tmp_path):
        """The ordinary stale lock: the process simply no longer exists."""
        script, pgdata, marker = _stage_postgres_wrapper(tmp_path)
        lock = pgdata / "postmaster.pid"
        # A PID that cannot be running: claim one, then let it exit.
        dead = subprocess.Popen(["/usr/bin/true"])
        dead.wait()
        lock.write_text(f"{dead.pid}\n{pgdata}\n")

        r = _run_wrapper(script)
        assert r.returncode == 0, r.stderr
        assert not lock.exists(), "stale lock was not removed"
        assert marker.exists(), "postgres was not started after clearing the lock"

    def test_removes_lock_whose_pid_was_REUSED_by_another_process(self, tmp_path):
        """The exact 2026-08-21 failure, and the one postgres cannot self-heal.

        The lock named PID 784; after the reboot 784 was AirPlayUIAgent. The
        PID is alive, so postgres' own check says "another postmaster is
        running" and refuses — permanently, every 10s under KeepAlive.
        """
        script, pgdata, marker = _stage_postgres_wrapper(tmp_path)
        lock = pgdata / "postmaster.pid"

        # A real, live process that is definitively NOT postgres.
        impostor = subprocess.Popen(["/bin/sleep", "30"])
        try:
            lock.write_text(f"{impostor.pid}\n{pgdata}\n")
            assert os.path.exists(f"/proc/{impostor.pid}") or True  # macOS: no procfs
            r = _run_wrapper(script)
        finally:
            impostor.kill()
            impostor.wait()

        assert r.returncode == 0, r.stderr
        assert not lock.exists(), (
            "lock held by a REUSED pid was not cleared — this is the bug that "
            "wedged postgres through an entire uptime"
        )
        assert marker.exists(), "postgres was not started after clearing the lock"
        assert "reboot PID reuse" in r.stderr, r.stderr

    def test_leaves_a_LIVE_postmaster_alone(self, tmp_path):
        """The safety half — and the one a careless 'just rm the lock' breaks.

        If a real postmaster owns the data directory, the wrapper must exit
        without deleting its lock and without starting a second postmaster
        against the same PGDATA.
        """
        script, pgdata, marker = _stage_postgres_wrapper(tmp_path)
        lock = pgdata / "postmaster.pid"

        # A live process whose `ps -o comm=` really does contain "postgres".
        #
        # Measured on this host: COPYING a signed system binary breaks its
        # signature and the copy dies instantly (comm == ''), and a shebang
        # script reports the INTERPRETER (/bin/sleep). A symlink invoked by
        # its own path is the one form that carries the name through.
        #
        # The directory is deliberately NOT pytest's tmp_path: tmp_path is
        # named after the test, and "test_leaves_a_LIVE_postmaster_alone"
        # truncates to a string that itself contains "postmaster" — which
        # would satisfy the wrapper's match no matter what the binary was
        # called, making this test pass for the wrong reason.
        bindir = tempfile.mkdtemp(prefix="svc-")
        assert "postgres" not in bindir and "postmaster" not in bindir, (
            f"test scaffold path {bindir!r} would match on its own — the "
            f"assertion below would be vacuous"
        )
        fake_pg = os.path.join(bindir, "postgres")
        os.symlink("/bin/sleep", fake_pg)
        live = subprocess.Popen([fake_pg, "30"])
        try:
            comm = subprocess.run(
                ["ps", "-p", str(live.pid), "-o", "comm="],
                capture_output=True, text=True,
            ).stdout
            assert "postgres" in comm, f"test double did not present as postgres: {comm!r}"

            lock.write_text(f"{live.pid}\n{pgdata}\n")
            r = _run_wrapper(script)
        finally:
            live.kill()
            live.wait()
            shutil.rmtree(bindir, ignore_errors=True)

        assert r.returncode == 0, r.stderr
        assert lock.exists(), "wrapper deleted a LIVE postmaster's lock file"
        assert not marker.exists(), (
            "wrapper started a SECOND postmaster against a data directory that "
            "was already in use"
        )
        assert "not starting a second one" in r.stderr, r.stderr

    def test_unreadable_pid_line_is_treated_as_stale(self, tmp_path):
        """A truncated lock (interrupted write) must not wedge the boot."""
        script, pgdata, marker = _stage_postgres_wrapper(tmp_path)
        lock = pgdata / "postmaster.pid"
        lock.write_text("\n")

        r = _run_wrapper(script)
        assert r.returncode == 0, r.stderr
        assert not lock.exists()
        assert marker.exists()

    def test_missing_binary_exits_EX_CONFIG_not_a_hot_loop(self, tmp_path):
        """A brew upgrade that moves the binary is a CONFIG fault.

        Exiting 78 keeps the KeepAlive retry from spinning on something that
        cannot succeed until a human intervenes.
        """
        src = _read_or_skip(POSTGRES_LAUNCHER)
        staged = re.sub(
            r'^PGBIN=.*$', 'PGBIN="/nonexistent/postgres"', src, count=1, flags=re.MULTILINE)
        path = tmp_path / "missing-bin.sh"
        path.write_text(staged)
        path.chmod(0o755)

        r = _run_wrapper(path)
        assert r.returncode == 78, f"expected EX_CONFIG(78), got {r.returncode}"
        assert "FATAL" in r.stderr


# ---------------------------------------------------------------------------
#  3. launchd deployment invariants
# ---------------------------------------------------------------------------

BOOT_DAEMONS = [
    "com.local.tor",
    "com.local.postgres",
    "com.local.llama-server",
    "com.local.ghost-agent",
    "com.local.ghost-client",
    "com.local.ghost-slackbot",
]


def _plist_or_skip(label):
    path = f"/Library/LaunchDaemons/{label}.plist"
    if not os.path.exists(path):
        pytest.skip(f"{label} not deployed on this host")
    return path


def _plist_get(path, key):
    r = subprocess.run(
        ["/usr/libexec/PlistBuddy", "-c", f"Print :{key}", path],
        capture_output=True, text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else None


class TestLaunchdDeployment:
    @pytest.mark.parametrize("label", BOOT_DAEMONS)
    def test_is_a_system_daemon_that_runs_at_boot(self, label):
        """A LaunchAgent would only start at LOGIN.

        tor as a login-scoped service is precisely what left the (system-
        daemon) agent respawn-looping from boot until someone logged in.
        """
        path = _plist_or_skip(label)
        assert _plist_get(path, "RunAtLoad") == "true", f"{label} lacks RunAtLoad"
        assert not os.path.exists(
            os.path.expanduser(f"~/Library/LaunchAgents/{label}.plist")
        ), f"{label} also present as a LaunchAgent — the two will race"

    @pytest.mark.parametrize("label", BOOT_DAEMONS)
    def test_log_paths_are_writable_by_the_service_user(self, label):
        """Guards the EX_CONFIG(78) trap.

        A daemon with `UserName` set whose log files it cannot open dies
        before the program runs, with an EMPTY log and no traceback.
        """
        path = _plist_or_skip(label)
        user = _plist_get(path, "UserName")
        if not user or user == "root":
            pytest.skip(f"{label} runs as root")

        import pwd
        uid = pwd.getpwnam(user).pw_uid
        for key in ("StandardOutPath", "StandardErrorPath"):
            log = _plist_get(path, key)
            if not log or not os.path.exists(log):
                continue
            st = os.stat(log)
            writable = st.st_uid == uid and (st.st_mode & stat.S_IWUSR)
            assert writable, (
                f"{label}: {key}={log} is owned by uid {st.st_uid}, not {user} "
                f"(uid {uid}) — launchd will fail this job EX_CONFIG(78) with "
                f"an empty log"
            )

    def test_no_rival_homebrew_tor_job_exists(self):
        """Regression pin, hit for real on 2026-08-21.

        Tor was first installed under Homebrew's own label
        (homebrew.mxcl.tor) in /Library/LaunchDaemons. Within the hour
        `brew services` re-created its OWN copy at
        ~/Library/LaunchAgents/homebrew.mxcl.tor.plist, which then sat in
        `error` state fighting the system daemon for :9050. Homebrew owns
        that label; the daemon now uses com.local.tor, which brew cannot
        claim. `brew services list` reporting tor as "none" is CORRECT here.
        """
        for path in (
            os.path.expanduser("~/Library/LaunchAgents/homebrew.mxcl.tor.plist"),
            "/Library/LaunchDaemons/homebrew.mxcl.tor.plist",
        ):
            assert not os.path.exists(path), (
                f"a Homebrew-labelled tor job reappeared at {path} — it will "
                f"fight com.local.tor for :9050. Remove it; do not run "
                f"`brew services start tor`."
            )

    def test_tor_config_binds_the_socks_port_the_agent_probes(self):
        """The agent's boot probe targets 127.0.0.1:9050 specifically."""
        torrc = "/opt/homebrew/etc/tor/torrc"
        if not os.path.exists(torrc):
            pytest.skip("torrc not deployed on this host")
        body = open(torrc).read()
        socks = [l for l in body.splitlines() if l.strip().startswith("SocksPort")]
        assert len(socks) == 1, f"expected exactly one SocksPort line, got {socks}"
        assert "127.0.0.1:9050" in socks[0], socks[0]
        # Per-engine circuit isolation rides SOCKS username auth
        # (utils/helpers.py builds "<tag>:isolate@host:port").
        assert "IsolateSOCKSAuth" in socks[0], socks[0]
        # DataDirectory must be explicit: $HOME is undefined for a daemon.
        assert re.search(r"^DataDirectory\s+\S+", body, re.MULTILINE), (
            "torrc must pin DataDirectory — a launchd daemon has no $HOME"
        )


# ---------------------------------------------------------------------------
#  4. The OrbStack engine starter
# ---------------------------------------------------------------------------

class TestOrbstackEngineStarter:
    """OrbStack is the one dependency that cannot be a boot daemon.

    It is a GUI VM manager, so a headless reboot leaves the sandbox tools
    unavailable until someone logs in — that part is a platform limit. What IS
    fixable is the gap this script closes: on 2026-08-21 the OrbStack *app*
    launched 9s after boot while the VM engine did not spawn until 16:05:59,
    six minutes later, because the app starts it on demand.
    """

    def test_verifies_the_ENGINE_not_just_the_app_process(self):
        """The distinguishing behaviour, and the whole reason this exists.

        A script that checked only for the OrbStack process would have
        reported success throughout the six-minute window in which docker did
        not work.
        """
        src = _read_or_skip(ORBSTACK_LAUNCHER)
        assert "status" in src and "Running" in src, (
            "must confirm the engine reached Running, not merely that `orb "
            "start` returned"
        )

    def test_uses_absolute_tool_paths_for_the_launchd_PATH(self, tmp_path):
        """Executed pin: run it with the PATH launchd actually provides.

        A bare `docker` resolves in an interactive shell and is 'command not
        found' under launchd (it lives in /usr/local/bin). This exact bug was
        present in the first version of this script and only surfaced by
        running it.
        """
        src = _read_or_skip(ORBSTACK_LAUNCHER)
        # Stub out orb so the test never touches the real VM.
        stub_dir = tempfile.mkdtemp(prefix="orbstub-")
        try:
            orb = os.path.join(stub_dir, "orb")
            with open(orb, "w") as f:
                f.write('#!/bin/bash\n[ "$1" = "status" ] && echo Running\nexit 0\n')
            os.chmod(orb, 0o755)

            staged = re.sub(r'^ORB=.*$', f'ORB="{orb}"', src, count=1, flags=re.MULTILINE)
            # Skip the wait-for-GUI loop: no OrbStack app in the test env.
            staged = staged.replace('for _ in $(seq 1 30); do', 'for _ in $(seq 1 0); do')
            path = tmp_path / "orb-staged.sh"
            path.write_text(staged)
            path.chmod(0o755)

            r = subprocess.run(
                [BASH, str(path)], capture_output=True, text=True, timeout=60,
                # The bare PATH launchd hands a job — no /usr/local/bin.
                env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(tmp_path)},
            )
        finally:
            shutil.rmtree(stub_dir, ignore_errors=True)

        assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
        combined = r.stdout + r.stderr
        assert "command not found" not in combined, (
            f"script reached for a tool that is not on launchd's PATH:\n{combined}"
        )
        assert "engine Running" in combined, combined

    def test_missing_orb_exits_EX_CONFIG(self, tmp_path):
        """A one-shot agent must not report success when orb is gone."""
        src = _read_or_skip(ORBSTACK_LAUNCHER)
        staged = re.sub(
            r'^ORB=.*$', 'ORB="/nonexistent/orb"', src, count=1, flags=re.MULTILINE)
        path = tmp_path / "orb-missing.sh"
        path.write_text(staged)
        path.chmod(0o755)

        r = subprocess.run([BASH, str(path)], capture_output=True, text=True, timeout=60)
        assert r.returncode == 78, f"expected EX_CONFIG(78), got {r.returncode}"

    def test_is_a_login_agent_not_a_boot_daemon(self):
        """Pins the platform reality so nobody 'promotes' it and expects boot."""
        agent = os.path.expanduser("~/Library/LaunchAgents/com.local.orbstack-engine.plist")
        if not os.path.exists(agent):
            pytest.skip("orbstack-engine agent not deployed on this host")
        assert not os.path.exists("/Library/LaunchDaemons/com.local.orbstack-engine.plist"), (
            "OrbStack cannot run pre-login — a system daemon would fail at boot"
        )
        assert _plist_get(agent, "RunAtLoad") == "true"
        # One-shot: KeepAlive would relaunch it forever after it exits.
        assert _plist_get(agent, "KeepAlive") is None
