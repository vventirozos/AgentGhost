"""Web-client TLS certificate auto-renewal (2026-09-29).

`bin/renew-ghost-cert.sh` (run daily by the LaunchAgent
com.local.ghost-cert-renew) renews the Tailscale certificate the web client
serves on :8080 once it has under 30 days left, then restarts the web client
and verifies the NEW certificate is on the wire. Before this it was manual and
was found with 10 days left.

Every test EXECUTES the real script against a stub `tailscale` that mints
self-signed certificates with openssl — nothing is text-asserted. The
behaviours pinned are the ones that could silently invert:

  * a not-due certificate is left alone (no daily reissue / rate-limit burn);
  * a bad fetch (unparseable, wrong name, key mismatch, not newer, CLI error)
    leaves the LIVE files byte-identical — a broken cert installed and served
    is worse than an expiring one;
  * the restart kills ONLY a listener whose argv proves it is our uvicorn,
    and success is claimed only once the respawned listener serves the new
    certificate's fingerprint.

The restart cases run a fake web client (a python TLS listener whose argv
matches uvicorn's) under a bash respawn loop standing in for launchd
KeepAlive.

The script lives at the ops-script location (/Users/vasilis/Data/AI/bin,
outside this repo), so every test SKIPS when it is not deployed — same
convention as tests/test_service_autostart.py.
"""

import os
import signal
import socket
import subprocess
import sys
import time

import pytest

OPS_BIN = os.environ.get("GHOST_OPS_BIN", "/Users/vasilis/Data/AI/bin")
RENEWER = os.path.join(OPS_BIN, "renew-ghost-cert.sh")
AGENT_PLIST = os.path.expanduser("~/Library/LaunchAgents/com.local.ghost-cert-renew.plist")

OPENSSL = "/usr/bin/openssl"
DOMAIN = "eva.test.ts.net"
BARE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"   # what launchd hands a job

STUB_TAILSCALE = r"""#!/bin/bash
# Stub `tailscale cert`: records its argv, mints a self-signed cert.
echo "$@" >> "$STUB_LOG"
[ "${STUB_FAIL:-0}" = "1" ] && { echo "stub: acme error" >&2; exit 1; }
cert=""; key=""; dom=""
while [ $# -gt 0 ]; do
    case "$1" in
        cert) ;;
        --cert-file) cert="$2"; shift ;;
        --key-file) key="$2"; shift ;;
        --min-validity) shift ;;
        *) dom="$1" ;;
    esac
    shift
done
san="${STUB_SAN:-$dom}"
/usr/bin/openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes \
    -keyout "$key" -out "$cert" -days "${STUB_DAYS:-90}" \
    -subj "/CN=$san" -addext "subjectAltName=DNS:$san" >/dev/null 2>&1 || exit 1
if [ "${STUB_MISMATCH:-0}" = "1" ]; then
    /usr/bin/openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:prime256v1 \
        -out "$key" >/dev/null 2>&1
fi
exit 0
"""

# argv must read like the real one:
#   uvicorn server:app --host … --port P --ssl-keyfile D.key --ssl-certfile D.crt
# FAKE_STALE_DIR: serve the pair from there instead (a respawn that picked up
# the wrong files). FAKE_IGNORE_TERM: a hung server that survives TERM.
FAKE_UVICORN = r"""
import os, signal, socket, ssl, sys
if os.environ.get("FAKE_IGNORE_TERM"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
a = sys.argv[1:]
port = int(a[a.index("--port") + 1])
crt, key = a[a.index("--ssl-certfile") + 1], a[a.index("--ssl-keyfile") + 1]
stale = os.environ.get("FAKE_STALE_DIR")
if stale:
    crt, key = os.path.join(stale, crt), os.path.join(stale, key)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(crt, key)
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", port))
s.listen(8)
while True:
    c, _ = s.accept()
    try:
        ctx.wrap_socket(c, server_side=True).close()
    except Exception:
        c.close()
"""


def _need_script():
    if not os.path.exists(RENEWER):
        pytest.skip(f"ops script not deployed at {RENEWER}")
    if not os.path.exists(OPENSSL):
        pytest.skip("no /usr/bin/openssl")


def _free_port():
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _mint(cert_dir, days):
    """Write DOMAIN.crt/.key into cert_dir, valid for `days`."""
    subprocess.run(
        [OPENSSL, "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
         "-nodes", "-keyout", str(cert_dir / f"{DOMAIN}.key"),
         "-out", str(cert_dir / f"{DOMAIN}.crt"), "-days", str(days),
         "-subj", f"/CN={DOMAIN}", "-addext", f"subjectAltName=DNS:{DOMAIN}"],
        check=True, capture_output=True)


def _fp_file(path):
    return subprocess.run([OPENSSL, "x509", "-in", str(path), "-noout", "-fingerprint", "-sha256"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _fp_served(port):
    s = subprocess.run([OPENSSL, "s_client", "-connect", f"127.0.0.1:{port}",
                        "-servername", DOMAIN], input="", capture_output=True, text=True, timeout=10)
    return subprocess.run([OPENSSL, "x509", "-noout", "-fingerprint", "-sha256"],
                          input=s.stdout, capture_output=True, text=True).stdout.strip()


def _listener_pid(port):
    r = subprocess.run(["/usr/sbin/lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                       capture_output=True, text=True)
    return r.stdout.split()[0] if r.stdout.split() else None


def _wait_listening(port, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        pid = _listener_pid(port)
        if pid:
            return pid
        time.sleep(0.1)
    raise AssertionError(f"nothing listening on :{port}")


@pytest.fixture
def env(tmp_path):
    """A cert dir holding a DUE certificate (10 days left) + the stub CLI."""
    _need_script()
    cert_dir = tmp_path / "interface"
    cert_dir.mkdir()
    _mint(cert_dir, 10)
    stub = tmp_path / "tailscale"
    stub.write_text(STUB_TAILSCALE)
    stub.chmod(0o755)
    e = {
        "PATH": BARE_PATH,
        "HOME": str(tmp_path),
        "GHOST_CERT_DOMAIN": DOMAIN,
        "GHOST_CERT_DIR": str(cert_dir),
        "GHOST_TAILSCALE_BIN": str(stub),
        "GHOST_CERT_RESTART": "0",
        "STUB_LOG": str(tmp_path / "stub.log"),
    }
    return cert_dir, e


def _run(e, **extra):
    return subprocess.run(["/bin/bash", RENEWER], capture_output=True, text=True,
                          timeout=90, env={**e, **extra})


def _live_bytes(cert_dir):
    return ((cert_dir / f"{DOMAIN}.crt").read_bytes(), (cert_dir / f"{DOMAIN}.key").read_bytes())


class TestWhenToRenew:
    def test_not_due_is_left_alone_and_tailscale_is_not_called(self, env):
        cert_dir, e = env
        _mint(cert_dir, 60)
        before = _live_bytes(cert_dir)
        r = _run(e)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "not due" in r.stdout
        assert not os.path.exists(e["STUB_LOG"]), "a not-due run must not hit the CA"
        assert _live_bytes(cert_dir) == before

    def test_due_cert_is_renewed_and_previous_pair_kept(self, env):
        cert_dir, e = env
        old = _live_bytes(cert_dir)
        r = _run(e)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "installed" in r.stdout
        assert _live_bytes(cert_dir) != old
        assert (cert_dir / f"{DOMAIN}.crt.prev").read_bytes() == old[0]
        assert (cert_dir / f"{DOMAIN}.key.prev").read_bytes() == old[1]
        assert (os.stat(cert_dir / f"{DOMAIN}.key").st_mode & 0o777) == 0o600
        # The renewed cert is no longer due: a second run is a no-op.
        assert "not due" in _run(e).stdout

    def test_missing_cert_is_treated_as_due(self, env):
        cert_dir, e = env
        (cert_dir / f"{DOMAIN}.crt").unlink()
        r = _run(e)
        assert r.returncode == 0, r.stdout + r.stderr
        assert (cert_dir / f"{DOMAIN}.crt").exists()

    def test_asks_tailscale_for_min_validity_beyond_the_threshold(self, env):
        """Without --min-validity Tailscale may hand back its cached, still-due
        cert, and the job would 'renew' to the same expiry every day."""
        _, e = env
        assert _run(e).returncode == 0
        argv = open(e["STUB_LOG"]).read().split()
        hours = int(argv[argv.index("--min-validity") + 1].rstrip("h"))
        assert hours > 30 * 24

    def test_no_temp_dirs_left_behind(self, env):
        cert_dir, e = env
        _run(e)
        _run(e, STUB_FAIL="1")
        assert not [p for p in os.listdir(cert_dir) if p.startswith(".renew.")]


class TestBadFetchLeavesLiveFilesUntouched:
    @pytest.mark.parametrize("extra, why", [
        ({"STUB_FAIL": "1"}, "tailscale cert failed"),
        ({"STUB_MISMATCH": "1"}, "key does not match"),
        ({"STUB_SAN": "other.test.ts.net"}, "does not name"),
        ({"STUB_DAYS": "5"}, "does not outlive"),
    ])
    def test_rejected(self, env, extra, why):
        cert_dir, e = env
        before = _live_bytes(cert_dir)
        r = _run(e, **extra)
        assert r.returncode == 1, r.stdout + r.stderr
        assert why in r.stdout
        assert _live_bytes(cert_dir) == before
        assert not (cert_dir / f"{DOMAIN}.crt.prev").exists()

    def test_missing_tailscale_is_EX_CONFIG(self, env):
        _, e = env
        r = _run(e, GHOST_TAILSCALE_BIN="/nonexistent/tailscale")
        assert r.returncode == 78


class _FakeWebClient:
    """A TLS listener with uvicorn's argv, optionally under a respawn loop."""

    def __init__(self, tmp_path, cert_dir, port, respawn, argv0="uvicorn", env=None):
        server = tmp_path / argv0
        server.write_text(FAKE_UVICORN)
        inner = (f"{sys.executable} {server} server:app --host 127.0.0.1 --port {port} "
                 f"--ssl-keyfile {DOMAIN}.key --ssl-certfile {DOMAIN}.crt")
        cmd = f"while true; do {inner}; sleep 0.3; done" if respawn else f"exec {inner}"
        self.proc = subprocess.Popen(["/bin/bash", "-c", cmd], cwd=cert_dir,
                                     start_new_session=True, env={**os.environ, **(env or {})},
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.pid = _wait_listening(port)

    def stop(self):
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # macOS: killpg on a group whose leader is an unreaped zombie
            # (the script already TERMed it) is EPERM, not ESRCH.
            pass
        self.proc.wait(timeout=5)


class TestRestart:
    def test_restarts_and_verifies_the_new_cert_on_the_wire(self, env, tmp_path):
        cert_dir, e = env
        port = _free_port()
        web = _FakeWebClient(tmp_path, cert_dir, port, respawn=True)
        try:
            old_served = _fp_served(port)
            r = _run(e, GHOST_CERT_RESTART="1", GHOST_CERT_PORT=str(port))
            assert r.returncode == 0, r.stdout + r.stderr
            assert "OK: web client" in r.stdout
            new_pid = _listener_pid(port)
            assert new_pid and new_pid != web.pid
            served = _fp_served(port)
            assert served == _fp_file(cert_dir / f"{DOMAIN}.crt")
            assert served != old_served
        finally:
            web.stop()

    def test_refuses_to_kill_a_listener_that_is_not_our_uvicorn(self, env, tmp_path):
        cert_dir, e = env
        port = _free_port()
        web = _FakeWebClient(tmp_path, cert_dir, port, respawn=False, argv0="someone_else")
        try:
            r = _run(e, GHOST_CERT_RESTART="1", GHOST_CERT_PORT=str(port))
            assert r.returncode == 1, r.stdout + r.stderr
            assert "refusing to kill" in r.stdout
            assert web.proc.poll() is None, "an unrelated listener was killed"
            assert _listener_pid(port) == web.pid
        finally:
            web.stop()

    def test_listener_that_never_comes_back_is_a_failure_not_success(self, env, tmp_path):
        cert_dir, e = env
        port = _free_port()
        web = _FakeWebClient(tmp_path, cert_dir, port, respawn=False)
        try:
            r = _run(e, GHOST_CERT_RESTART="1", GHOST_CERT_PORT=str(port),
                     GHOST_CERT_RESTART_WAIT="4")
            assert r.returncode == 1, r.stdout + r.stderr
            assert "did not come back" in r.stdout
            assert "OK:" not in r.stdout
        finally:
            web.stop()

    def test_respawn_serving_the_wrong_cert_is_a_failure(self, env, tmp_path):
        """A listener came back, but on the old pair: the pid changed, so
        only the on-the-wire fingerprint check can tell."""
        cert_dir, e = env
        stale = tmp_path / "stale"
        stale.mkdir()
        for suffix in ("crt", "key"):
            (stale / f"{DOMAIN}.{suffix}").write_bytes((cert_dir / f"{DOMAIN}.{suffix}").read_bytes())
        port = _free_port()
        web = _FakeWebClient(tmp_path, cert_dir, port, respawn=True,
                             env={"FAKE_STALE_DIR": str(stale)})
        try:
            r = _run(e, GHOST_CERT_RESTART="1", GHOST_CERT_PORT=str(port))
            assert r.returncode == 1, r.stdout + r.stderr
            assert "not the new cert" in r.stdout
            assert "OK:" not in r.stdout
        finally:
            web.stop()

    def test_hung_server_that_survives_TERM_is_not_mistaken_for_a_restart(self, env, tmp_path):
        """The same pid still listening is NOT a restarted client — it must be
        diagnosed as 'did not come back', not probed as if it were new."""
        cert_dir, e = env
        port = _free_port()
        web = _FakeWebClient(tmp_path, cert_dir, port, respawn=False,
                             env={"FAKE_IGNORE_TERM": "1"})
        try:
            r = _run(e, GHOST_CERT_RESTART="1", GHOST_CERT_PORT=str(port),
                     GHOST_CERT_RESTART_WAIT="4")
            assert r.returncode == 1, r.stdout + r.stderr
            assert "did not come back" in r.stdout
        finally:
            web.stop()

    def test_no_listener_installs_and_exits_clean(self, env):
        cert_dir, e = env
        r = _run(e, GHOST_CERT_RESTART="1", GHOST_CERT_PORT=str(_free_port()))
        assert r.returncode == 0, r.stdout + r.stderr
        assert "no listener" in r.stdout


def _plist_get(path, key):
    r = subprocess.run(["/usr/libexec/PlistBuddy", "-c", f"Print :{key}", path],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


class TestLaunchAgent:
    def _plist(self):
        if not os.path.exists(AGENT_PLIST):
            pytest.skip("cert-renew agent not deployed on this host")
        return AGENT_PLIST

    def test_runs_the_deployed_script(self):
        assert _plist_get(self._plist(), "ProgramArguments:0") == RENEWER

    def test_is_scheduled_and_catches_up_at_login(self):
        p = self._plist()
        assert _plist_get(p, "StartCalendarInterval:Hour") is not None
        assert _plist_get(p, "RunAtLoad") == "true"
        # One-shot per run: KeepAlive would re-run it in a loop.
        assert _plist_get(p, "KeepAlive") is None

    def test_log_is_writable_by_the_user(self):
        log = _plist_get(self._plist(), "StandardOutPath")
        assert log and os.access(os.path.dirname(log), os.W_OK)
        if os.path.exists(log):
            assert os.access(log, os.W_OK)
