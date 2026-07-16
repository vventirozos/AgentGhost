"""Tests for supervised long-lived sandbox services
(sandbox/services.py + tools/sandbox_services.py + the browser SSRF
service-port allowlist + docker.py port publishing — 2026-07-11)."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import json
import re
import types
from pathlib import Path

import pytest

from ghost_agent.sandbox.services import (
    BLOCKED_PORTS, MAX_SERVICES,
    ServiceSupervisor, default_service_ports, publishable_service_ports,
    get_service_supervisor, active_service_ports,
    is_published_port, remote_access_hint, REMOTE_SERVE_SCRIPT,
    REMOTE_UNSERVE_SCRIPT,
)
from ghost_agent.tools.sandbox_services import (
    tool_manage_services, MANAGE_SERVICES_TOOL_DEFINITION,
)


# ──────────────────────────────────────────────────────────────────────
# Fake sandbox manager
# ──────────────────────────────────────────────────────────────────────

class FakeSandbox:
    """Records executed commands; a handler decides (out, code) per call."""

    def __init__(self, tmp_path, handler=None):
        self.host_workspace = Path(tmp_path)
        self.calls = []
        self.handler = handler or (lambda cmd: ("", 0))

    def execute(self, cmd, timeout=600, **kw):
        self.calls.append(cmd)
        out, code = self.handler(cmd)
        # Simulate the container-side cmd.sh writing its pidfile as its first
        # action (the real launch does `echo $$ > <name>.pid`). start() reads
        # THAT for the real pid instead of the launcher's transient $!.
        if "nohup" in cmd and code == 0:
            m = re.search(r'\.services/([A-Za-z0-9_-]+)\.cmd\.sh', cmd)
            tok = out.strip().split()[-1] if out.strip() else ""
            if m and tok.isdigit():
                svc = self.host_workspace / ".services"
                svc.mkdir(parents=True, exist_ok=True)
                pf = svc / f"{m.group(1)}.pid"
                if not pf.exists():   # a handler may have written a distinct one
                    pf.write_text(tok)
        return out, code


def happy_handler(pid="12345", port_listens=True):
    def handler(cmd):
        if "nohup" in cmd:
            return (f"{pid}\n", 0)
        if "kill -0" in cmd:
            return ("", 0)  # alive
        if "python3 -c" in cmd:  # port probe
            return ("", 0 if port_listens else 1)
        return ("", 0)
    return handler


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    import ghost_agent.sandbox.services as svc
    monkeypatch.setattr(svc.time, "sleep", lambda s: None)


# ──────────────────────────────────────────────────────────────────────
# default_service_ports
# ──────────────────────────────────────────────────────────────────────

class TestDefaultServicePorts:
    def test_range_spec(self):
        assert default_service_ports("8100-8102") == [8100, 8101, 8102]

    def test_comma_spec_and_blocked_filtered(self):
        assert default_service_ports("8100,8000,8088,9050,8080") == [8100]

    def test_empty_disables(self):
        assert default_service_ports("") == []

    def test_garbage_is_empty(self):
        assert default_service_ports("not-ports") == []

    def test_oversized_range_dropped(self):
        assert default_service_ports("10000-99999") == []

    def test_env_default(self, monkeypatch):
        monkeypatch.delenv("GHOST_SANDBOX_SERVICE_PORTS", raising=False)
        assert default_service_ports() == [8100, 8101, 8102, 8103, 8104]


class TestPublishablePorts:
    """A second agent (throwaway / test suite) must NOT try to publish host
    ports the running instance already holds — that made `containers.run`
    fail outright and (before the fix) bricked the sandbox."""

    def test_taken_port_is_filtered_out(self, monkeypatch):
        import socket
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        taken = s.getsockname()[1]
        try:
            monkeypatch.setattr(
                "ghost_agent.sandbox.services.default_service_ports",
                lambda spec=None: [taken],
            )
            assert publishable_service_ports() == []
        finally:
            s.close()

    def test_free_port_is_kept(self, monkeypatch):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
        s.close()  # released → now bindable
        monkeypatch.setattr(
            "ghost_agent.sandbox.services.default_service_ports",
            lambda spec=None: [free],
        )
        assert publishable_service_ports() == [free]


# ──────────────────────────────────────────────────────────────────────
# ServiceSupervisor
# ──────────────────────────────────────────────────────────────────────

class TestSupervisorValidation:
    def _sup(self, tmp_path):
        return ServiceSupervisor(FakeSandbox(tmp_path))

    def test_bad_name(self, tmp_path):
        assert "invalid service name" in self._sup(tmp_path).start(
            "1bad name!", "python3 -m http.server 8100")

    def test_missing_command(self, tmp_path):
        assert "'command' is required" in self._sup(tmp_path).start(
            "web", "")

    def test_mock_server_command_refused(self, tmp_path):
        out = self._sup(tmp_path).start(
            "fake", "python3 -m http.server 8000 --bind 127.0.0.1:8000")
        assert "forbidden" in out and "8000" in out

    @pytest.mark.parametrize("port", sorted(BLOCKED_PORTS))
    def test_blocked_ports_refused(self, tmp_path, port):
        out = self._sup(tmp_path).start(
            "web", "python3 -m http.server", port=port)
        assert "reserved" in out

    def test_port_out_of_range(self, tmp_path):
        assert "out of range" in self._sup(tmp_path).start(
            "web", "cmd", port=80)


class TestSupervisorLifecycle:
    def test_start_happy_path(self, tmp_path):
        sb = FakeSandbox(tmp_path, happy_handler())
        sup = ServiceSupervisor(sb)
        out = sup.start("dash", "python3 -m http.server 8100", port=8100)
        assert "RUNNING" in out and "pid 12345" in out
        assert "http://127.0.0.1:8100" in out and "listening ✓" in out
        # Command shipped as a script via the bind mount.
        script = (tmp_path / ".services" / "dash.cmd.sh").read_text()
        assert "python3 -m http.server 8100" in script
        # Registry persisted host-side.
        reg = json.loads((tmp_path / ".services" / "registry.json").read_text())
        assert reg["dash"]["pid"] == 12345 and reg["dash"]["port"] == 8100
        # Launch used setsid+nohup redirected to the service log.
        launch = next(c for c in sb.calls if "nohup" in c)
        assert "setsid" in launch and ".services/dash.log" in launch

    def test_start_exports_port_env_var(self, tmp_path):
        # The assigned port is exported so an app can BIND it instead of
        # hardcoding one (2026-07-12) — PORT (the convention) + the explicit
        # GHOST_SERVICE_PORT.
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        sup.start("dash", "python3 app.py", port=8100)
        script = (tmp_path / ".services" / "dash.cmd.sh").read_text()
        assert "export PORT=8100" in script
        assert "export GHOST_SERVICE_PORT=8100" in script
        # The env exports precede the command.
        assert script.index("export PORT=8100") < script.index("python3 app.py")

    def test_start_exports_host_0000(self, tmp_path):
        # HOST=0.0.0.0 so the app binds the container's forwarding interface,
        # not loopback (2026-07-12) — docker's bridge-publish maps host
        # 127.0.0.1:<port> -> container <port>, and a loopback-bound app never
        # receives the forwarded traffic (unreachable from host/remote).
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        sup.start("dash", "python3 app.py", port=8100)
        script = (tmp_path / ".services" / "dash.cmd.sh").read_text()
        assert "export HOST=0.0.0.0" in script
        assert "export GHOST_SERVICE_HOST=0.0.0.0" in script
        assert script.index("export HOST=0.0.0.0") < script.index("python3 app.py")

    def test_start_no_port_no_env_export(self, tmp_path):
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        sup.start("daemon", "python3 worker.py")   # no port
        script = (tmp_path / ".services" / "daemon.cmd.sh").read_text()
        assert "PORT=" not in script
        # No port -> nothing published -> no host-bind export either.
        assert "HOST=" not in script


# ──────────────────────────────────────────────────────────────────────
# Remote access: put a published sandbox port on the tailnet (2026-07-12)
# ──────────────────────────────────────────────────────────────────────

class TestIsPublishedPort:
    def test_in_default_range_is_published(self):
        assert is_published_port(8100) is True
        assert is_published_port(8104) is True

    def test_out_of_range_is_not_published(self):
        # A port the container never publishes to the host never leaves the
        # sandbox, so it isn't a remote-exposure candidate.
        assert is_published_port(9099) is False

    def test_respects_configured_range(self, monkeypatch):
        monkeypatch.setenv("GHOST_SANDBOX_SERVICE_PORTS", "9000-9002")
        assert is_published_port(9001) is True
        assert is_published_port(8100) is False

    def test_publishing_disabled_publishes_nothing(self, monkeypatch):
        monkeypatch.setenv("GHOST_SANDBOX_SERVICE_PORTS", "")
        assert is_published_port(8100) is False

    def test_actual_published_set_is_authoritative(self):
        # The 2026-07-15 fix: a 2nd instance publishes NOTHING (all fixed ports
        # taken), so the actual set is empty even though the port is in the
        # configured range. The runtime set must win over the range fallback.
        assert is_published_port(8100, published_ports=set()) is False
        assert is_published_port(8100, published_ports={8100}) is True
        # A port NOT in the actual set (published by another instance) is False
        # even though it's a configured-range port.
        assert is_published_port(8104, published_ports={8100}) is False

    def test_none_published_set_falls_back_to_configured_range(self):
        # Before any container is created the manager reports None → the old
        # configured-range behaviour (single-instance common case) is preserved.
        assert is_published_port(8100, published_ports=None) is True

    def test_supervisor_consults_the_managers_published_set(self):
        # A supervisor whose manager published NOTHING must not offer a remote
        # hint for a configured-range port (the operator-misdirection bug).
        from unittest.mock import MagicMock
        from ghost_agent.sandbox.services import ServiceSupervisor
        mgr = MagicMock()
        mgr.published_service_ports.return_value = set()  # 2nd instance
        sup = ServiceSupervisor(mgr)
        assert sup._published_ports() == set()
        assert is_published_port(8100, published_ports=sup._published_ports()) is False


class TestRemoteAccessHint:
    def test_hint_names_the_serve_script_and_url_and_bind(self):
        h = remote_access_hint(8100)
        assert REMOTE_SERVE_SCRIPT in h and "8100" in h
        assert ".ts.net" in h                 # the tailnet URL shape
        assert "0.0.0.0" in h                  # the bind requirement
        assert REMOTE_UNSERVE_SCRIPT in h      # teardown pointer

    def test_running_report_includes_hint_for_published_port(self, tmp_path):
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        out = sup.start("dash", "python3 app.py", port=8100)
        assert "RUNNING" in out
        assert "Remote access" in out
        assert REMOTE_SERVE_SCRIPT in out

    def test_running_report_omits_hint_for_unpublished_port(self, tmp_path):
        # 9099 is a valid service port (1024-65535, not blocked) but outside
        # the published range, so it can't be reached from the host — no hint.
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        out = sup.start("dash", "python3 app.py", port=9099)
        assert "RUNNING" in out
        assert "Remote access" not in out

    def test_status_list_shows_remote_pointer_for_published_service(self, tmp_path):
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        sup.start("dash", "python3 app.py", port=8100)
        listing = sup.status()
        assert "remote:" in listing and REMOTE_SERVE_SCRIPT in listing

    def test_alive_but_port_never_binds_surfaces_the_log(self, tmp_path):
        # THE fix: process stays alive but the port never comes up (crash on
        # import / missing dep / wrong bind port). The cause is in the log —
        # surface it instead of a vague "NOT listening yet", so the agent
        # doesn't browse→fail→install→restart.
        def handler(cmd):
            if "nohup" in cmd:
                (tmp_path / ".services").mkdir(parents=True, exist_ok=True)
                (tmp_path / ".services" / "web.log").write_text(
                    "Traceback ...\nModuleNotFoundError: No module named 'chess'")
                return ("555\n", 0)
            if "kill -0" in cmd:
                return ("", 0)          # process ALIVE
            if "python3 -c" in cmd:
                return ("", 1)          # port NEVER listening
            return ("", 0)
        sup = ServiceSupervisor(FakeSandbox(tmp_path, handler))
        out = sup.start("web", "python3 app.py", port=8100)
        assert "nothing is listening on port 8100" in out
        assert "No module named 'chess'" in out      # the actual cause, shown NOW
        assert "restart" in out                       # tells it how to recover
        assert "RUNNING" not in out                   # not falsely "up"

    def test_start_immediate_exit_reports_log(self, tmp_path):
        log_path = tmp_path / ".services" / "web.log"

        def handler(cmd):
            if "nohup" in cmd:
                # Simulate the service logging its crash before dying —
                # start() resets the log BEFORE launch, so the fixture must
                # be written by the launch itself, as in reality.
                log_path.write_text("python3: No module named flask\n")
                return ("777\n", 0)
            if "kill -0" in cmd:
                return ("", 1)  # dead
            return ("", 0)
        sup = ServiceSupervisor(FakeSandbox(tmp_path, handler))
        out = sup.start("web", "python3 app.py", port=8101)
        assert "exited immediately" in out
        assert "No module named flask" in out
        assert sup._load() == {}  # not registered

    def test_start_twice_requires_restart(self, tmp_path):
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        sup.start("dash", "cmd1", port=8100)
        out = sup.start("dash", "cmd2", port=8101)
        assert "already running" in out

    def test_max_services_cap(self, tmp_path):
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        for i in range(MAX_SERVICES):
            assert "RUNNING" in sup.start(f"s{i}", "cmd")
        out = sup.start("one-too-many", "cmd")
        assert "already running — stop one first" in out.replace("services", "services")
        assert f"{MAX_SERVICES} services" in out

    def test_stop_kills_process_group(self, tmp_path):
        sb = FakeSandbox(tmp_path, happy_handler())
        sup = ServiceSupervisor(sb)
        sup.start("dash", "cmd", port=8100)

        killed = []

        def stop_handler(cmd):
            if "kill -TERM" in cmd or "kill -KILL" in cmd:
                killed.append(cmd)
                return ("", 0)
            if "kill -0" in cmd:
                # Alive before TERM, dead after.
                return ("", 0 if not killed else 1)
            return ("", 0)
        sb.handler = stop_handler
        out = sup.stop("dash")
        assert "stopped" in out
        assert any("-- -12345" in c for c in killed)  # group kill
        assert sup._load() == {}

    def test_stop_unknown(self, tmp_path):
        sup = ServiceSupervisor(FakeSandbox(tmp_path))
        assert "no service named" in sup.stop("ghosty")

    def test_restart_reuses_registration(self, tmp_path):
        sb = FakeSandbox(tmp_path, happy_handler())
        sup = ServiceSupervisor(sb)
        sup.start("dash", "python3 -m http.server 8100", port=8100)

        # After stop, everything is dead until the next nohup launch.
        relaunched = []

        def handler(cmd):
            if "nohup" in cmd:
                relaunched.append(cmd)
                return ("888\n", 0)
            if "kill -0" in cmd:
                return ("", 0 if relaunched else 1)
            if "python3 -c" in cmd:
                return ("", 0)
            return ("", 0)
        sb.handler = handler
        out = sup.restart("dash")
        assert "RUNNING" in out and "pid 888" in out
        assert sup._load()["dash"]["command"] == "python3 -m http.server 8100"

    def test_status_lists_running_and_dead(self, tmp_path):
        sb = FakeSandbox(tmp_path, happy_handler())
        sup = ServiceSupervisor(sb)
        sup.start("alive", "cmd", port=8100)
        sup.start("gone", "cmd2")
        reg = sup._load()

        def handler(cmd):
            if "kill -0" in cmd:
                return ("", 0 if str(reg["alive"]["pid"]) in cmd else 1)
            if "python3 -c" in cmd:
                return ("", 0)
            return ("", 0)
        # Make the pids distinct so the handler can tell them apart.
        reg["gone"]["pid"] = 99999
        sup._save(reg)
        sb.handler = handler
        out = sup.status()
        assert "alive: RUNNING" in out
        assert "gone: DEAD" in out
        # A dead entry present -> status points at the one-shot cleanup.
        assert "stop-all" in out


class TestReliablePidAndCleanup:
    """Real-pid tracking + stop-all + orphaned-port reclaim (2026-07-12).
    The launcher's `$!` tracked a transient wrapper, so stop()/restart() killed
    the wrong pid and left the REAL service orphaned — hung processes piled up.
    Now the script records its own pid and stop reclaims the port as a fallback."""

    def test_cmd_script_records_pid_and_tags_name(self, tmp_path):
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        sup.start("dash", "python3 app.py", port=8100)
        script = (tmp_path / ".services" / "dash.cmd.sh").read_text()
        assert "echo $$ >" in script and "dash.pid" in script
        assert "GHOST_SERVICE_NAME" in script

    def test_simple_command_is_execd_so_pid_is_the_real_process(self, tmp_path):
        # `exec` makes the setsid shell BECOME the command (same pid), so the
        # recorded $$ is the real process — not a wrapper the shell forks then
        # exits from (which showed status=DEAD while the service ran).
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        sup.start("dash", "python3 app.py", port=8100)
        script = (tmp_path / ".services" / "dash.cmd.sh").read_text()
        assert "exec python3 app.py" in script

    def test_compound_command_is_not_execd(self, tmp_path):
        # A compound command can't be exec'd; it runs normally (port-reclaim
        # covers it). Prefer workdir= over `cd x && …` anyway.
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        sup.start("web", "node a.js && node b.js", port=8101)
        script = (tmp_path / ".services" / "web.cmd.sh").read_text()
        assert "exec node" not in script
        assert "node a.js && node b.js" in script


class TestZombiesAndWorkdirValidation:
    """The 137s partial-failure postmortem (2026-07-12). Root causes: (a) the
    container's non-reaping PID 1 turned dead launchers into ZOMBIES, which
    pass `kill -0` — so a never-started service looked 'already running' and
    stop() was a no-op; (b) a bad workdir ('/projects/<id>', missing the
    /workspace prefix) made the launch cd fail INVISIBLY inside the async
    subshell — no log, no error, 3 identical failed rounds."""

    def test_pid_alive_check_is_zombie_proof(self, tmp_path):
        sb = FakeSandbox(tmp_path, happy_handler())
        sup = ServiceSupervisor(sb)
        sup._pid_alive(307)
        cmd = sb.calls[-1]
        assert "kill -0 307" in cmd
        # And it must ALSO reject state Z from /proc/<pid>/stat.
        assert "/proc/307/stat" in cmd and "!= Z" in cmd

    def test_zombie_pid_reads_as_dead(self, tmp_path):
        # A zombie passes bare kill -0 but the composite check exits 1.
        def handler(cmd):
            if "kill -0" in cmd:
                return ("", 1)      # composite: zombie -> exit 1
            return ("", 0)
        sup = ServiceSupervisor(FakeSandbox(tmp_path, handler))
        assert sup._pid_alive(307) is False

    def test_bad_absolute_workdir_is_healed_to_workspace(self, tmp_path):
        # The model's exact live mistake: '/projects/<id>'. test -d fails for
        # it, succeeds for '/workspace/projects/<id>' -> healed, launch works.
        def handler(cmd):
            if cmd.startswith("test -d "):
                return ("", 0 if "/workspace/projects/x" in cmd else 1)
            return happy_handler()(cmd)
        sb = FakeSandbox(tmp_path, handler)
        sup = ServiceSupervisor(sb)
        out = sup.start("web", "python3 app.py", port=8100,
                        workdir="/projects/x")
        assert "RUNNING" in out
        launch = next(c for c in sb.calls if "nohup" in c)
        assert "cd /workspace/projects/x" in launch
        assert sup._load()["web"]["workdir"] == "/workspace/projects/x"

    def test_nonexistent_workdir_refuses_to_launch(self, tmp_path):
        def handler(cmd):
            if cmd.startswith("test -d "):
                return ("", 1)                  # nothing exists
            return happy_handler()(cmd)
        sb = FakeSandbox(tmp_path, handler)
        out = ServiceSupervisor(sb).start("web", "python3 app.py",
                                          port=8100, workdir="/nope/nowhere")
        assert "does not exist" in out and "Nothing was launched" in out
        assert not any("nohup" in c for c in sb.calls)   # never launched

    def test_redundant_cd_is_stripped_when_workdir_covers_it(self, tmp_path):
        # Model habitually passes BOTH workdir=X and command='cd X && …';
        # from inside X the inner relative cd would fail (X/X). Strip it.
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        out = sup.start("web", "cd projects/x && python3 app.py",
                        port=8100, workdir="projects/x")
        assert "RUNNING" in out
        script = (tmp_path / ".services" / "web.cmd.sh").read_text()
        assert "cd projects/x" not in script
        assert "exec python3 app.py" in script     # simple again -> exec'd
        assert sup._load()["web"]["command"] == "python3 app.py"

    def test_docker_container_runs_with_init(self):
        # tini as PID 1 reaps zombies — the systemic fix. Wiring-pinned.
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "sandbox" / "docker.py").read_text()
        assert 'run_kwargs["init"] = True' in src

    def test_start_records_pidfile_pid_not_launcher_pid(self, tmp_path):
        # Script wrote 777; launcher $! was a transient 999. Registry must use
        # 777 — the whole point of the fix.
        svc = tmp_path / ".services"

        def handler(cmd):
            if "nohup" in cmd:
                m = re.search(r'\.services/([A-Za-z0-9_-]+)\.cmd\.sh', cmd)
                svc.mkdir(parents=True, exist_ok=True)
                (svc / f"{m.group(1)}.pid").write_text("777")
                return ("999\n", 0)      # $! wrapper — must be ignored
            if "kill -0" in cmd:
                return ("", 0)
            if "python3 -c" in cmd:
                return ("", 0)
            return ("", 0)
        sup = ServiceSupervisor(FakeSandbox(tmp_path, handler))
        sup.start("web", "python3 app.py", port=8100)
        assert sup._load()["web"]["pid"] == 777

    def test_stop_all_stops_everything_and_clears_registry(self, tmp_path):
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        sup.start("a", "cmd", port=8100)
        sup.start("b", "cmd", port=8101)
        out = sup.stop_all()
        assert "Stopped 2" in out
        assert "No services registered" in sup.status()

    def test_stop_all_empty_is_a_noop(self, tmp_path):
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        assert "nothing to stop" in sup.stop_all()

    def test_stop_reclaims_orphaned_port_via_ss(self, tmp_path):
        # The exact bug: registry pid looks DEAD but the port is still held by
        # the mis-tracked real process. stop() must find it via `ss` and kill
        # it — otherwise the orphan hangs on forever.
        killed, state = [], {"alive": True}

        def handler(cmd):
            if "nohup" in cmd:
                return ("100\n", 0)
            if "kill -0" in cmd:
                return ("", 0 if state["alive"] else 1)
            if "python3 -c" in cmd:
                return ("", 0)                      # port STILL listening
            if "ss -" in cmd:
                # The full `ss | grep | cut` pipeline yields just the pid.
                return ("625\n", 0)                 # ss finds the orphan
            if "kill -TERM" in cmd or "kill -KILL" in cmd:
                killed.append(cmd)
                return ("", 0)
            return ("", 0)
        sup = ServiceSupervisor(FakeSandbox(tmp_path, handler))
        sup.start("web", "python3 app.py", port=8102)   # registers (alive)
        state["alive"] = False                          # pid now mis-tracked/dead
        sup.stop("web")
        assert any("625" in c for c in killed), killed

    def test_status_empty(self, tmp_path):
        assert "No services registered" in \
            ServiceSupervisor(FakeSandbox(tmp_path)).status()

    def test_logs_tail(self, tmp_path):
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        sup.start("dash", "cmd")
        sup.host_dir.joinpath("dash.log").write_text(
            "\n".join(f"line{i}" for i in range(100)))
        out = sup.logs("dash", lines=3)
        assert "line99" in out and "line96" not in out

    def test_active_ports_registry_driven(self, tmp_path):
        sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
        sup.start("a", "cmd", port=8100)
        sup.start("b", "cmd")          # no port
        assert sup.active_ports() == frozenset({8100})
        # Helper is fail-safe without a sandbox.
        assert active_service_ports(None) == frozenset()
        assert active_service_ports(sup.sandbox) == frozenset({8100})

    def test_supervisor_cached_on_sandbox(self, tmp_path):
        sb = FakeSandbox(tmp_path)
        s1 = get_service_supervisor(sb)
        s2 = get_service_supervisor(sb)
        assert s1 is s2
        assert get_service_supervisor(None) is None


# ──────────────────────────────────────────────────────────────────────
# tool_manage_services
# ──────────────────────────────────────────────────────────────────────

class TestManageServicesTool:
    def test_no_sandbox(self):
        out = asyncio.run(tool_manage_services(action="status"))
        assert "Sandbox manager not initialized" in out

    def test_unknown_action(self, tmp_path):
        out = asyncio.run(tool_manage_services(
            action="explode", sandbox_manager=FakeSandbox(tmp_path)))
        assert "unknown action" in out

    def test_action_healing_list_to_status(self, tmp_path):
        out = asyncio.run(tool_manage_services(
            action="list", sandbox_manager=FakeSandbox(tmp_path)))
        assert "No services registered" in out

    def test_start_via_tool_with_near_miss_args(self, tmp_path):
        out = asyncio.run(tool_manage_services(
            action="start", service="dash", cmd="python3 -m http.server 8100",
            service_port=8100,
            sandbox_manager=FakeSandbox(tmp_path, happy_handler())))
        assert "RUNNING" in out and "8100" in out

    def test_schema_exposes_workdir(self):
        # Regression (2026-07-12): the handler + supervisor accepted workdir,
        # but it was MISSING from the tool schema, so the model couldn't see it
        # and wasted ~50s baking `cd` into the command (and tripped the loop
        # breaker) when an app lived in a subdirectory.
        props = (MANAGE_SERVICES_TOOL_DEFINITION["function"]["parameters"]
                 ["properties"])
        assert "workdir" in props
        assert "relative to /workspace" in props["workdir"]["description"]

    def test_start_via_tool_honors_workdir(self, tmp_path):
        # workdir threads through the tool -> supervisor -> the launch runs the
        # command FROM that dir (cd <workdir> && …), and the registry records
        # it. Relative paths anchor to /workspace explicitly (2026-07-12).
        sb = FakeSandbox(tmp_path, happy_handler())
        out = asyncio.run(tool_manage_services(
            action="start", name="dash", command="python3 app.py", port=8100,
            workdir="projects/30d5d5b65c38", sandbox_manager=sb))
        assert "RUNNING" in out
        launch = next(c for c in sb.calls if "nohup" in c)
        assert "cd /workspace/projects/30d5d5b65c38" in launch
        reg = json.loads((tmp_path / ".services" / "registry.json").read_text())
        assert reg["dash"]["workdir"] == "/workspace/projects/30d5d5b65c38"


# ──────────────────────────────────────────────────────────────────────
# Browser SSRF allowlist (host guard + runner guard)
# ──────────────────────────────────────────────────────────────────────

class TestBrowserServicePortAllowlist:
    def test_host_guard_admits_registered_loopback_port(self):
        from ghost_agent.tools.browser import _browser_blocked_url
        assert _browser_blocked_url(
            "http://127.0.0.1:8100/", allowed_local_ports={8100}) is None
        assert _browser_blocked_url(
            "http://localhost:8100/x", allowed_local_ports={8100}) is None

    def test_host_guard_still_blocks_everything_else_local(self):
        from ghost_agent.tools.browser import _browser_blocked_url
        # Unlisted loopback port stays blocked.
        assert _browser_blocked_url(
            "http://127.0.0.1:9051/", allowed_local_ports={8100}) is not None
        # No allowlist at all → loopback blocked (pre-existing behaviour).
        assert _browser_blocked_url("http://127.0.0.1:8100/") is not None
        # Allowlist must NOT open non-loopback private hosts on that port.
        assert _browser_blocked_url(
            "http://192.168.0.24:8100/",
            allowed_local_ports={8100}) is not None

    def test_payload_carries_allowed_ports(self):
        from ghost_agent.tools.browser import _build_op_payload
        p = _build_op_payload(
            op="navigate", url="http://127.0.0.1:8100", selector=None,
            out_path=None, wait_until=None, full_page=None, max_chars=None,
            timeout_ms=1000, tor_proxy=None,
            allowed_local_ports=frozenset({8100}))
        assert p["allowed_local_ports"] == [8100]
        p2 = _build_op_payload(
            op="navigate", url="https://x.org", selector=None, out_path=None,
            wait_until=None, full_page=None, max_chars=None,
            timeout_ms=1000, tor_proxy=None)
        assert "allowed_local_ports" not in p2

    def _runner_namespace(self):
        """Exec the in-sandbox runner source with playwright stubbed, so the
        runner-side guard is tested FUNCTIONALLY, not by regex."""
        from ghost_agent.tools.browser import _runner_script
        import unittest.mock as um
        fake_pw = types.ModuleType("playwright")
        fake_api = types.ModuleType("playwright.async_api")
        fake_api.async_playwright = None
        fake_pw.async_api = fake_api
        ns = {}
        with um.patch.dict(sys.modules, {"playwright": fake_pw,
                                         "playwright.async_api": fake_api}):
            src = _runner_script()
            # Don't run the runner's __main__ block.
            src = src.replace('if __name__ == "__main__":', 'if False:')
            exec(compile(src, "browser_runner", "exec"), ns)
        return ns

    def test_runner_guard_admits_allowed_local_port(self):
        ns = self._runner_namespace()
        ns["ALLOWED_LOCAL_PORTS"].add(8100)
        assert ns["_ssrf_should_block"]("http://127.0.0.1:8100/") is False
        assert ns["_ssrf_should_block"]("http://localhost:8100/api") is False
        # Unlisted port / other internal targets remain blocked.
        assert ns["_ssrf_should_block"]("http://127.0.0.1:9051/") is True
        assert ns["_ssrf_should_block"]("http://169.254.169.254/") is True

    def test_runner_guard_blocks_loopback_without_allowlist(self):
        ns = self._runner_namespace()
        assert ns["_ssrf_should_block"]("http://127.0.0.1:8100/") is True


# ──────────────────────────────────────────────────────────────────────
# Wiring
# ──────────────────────────────────────────────────────────────────────

_SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"


class TestWiring:
    def test_tool_advertised_and_dispatchable(self):
        from ghost_agent.tools.registry import TOOL_DEFINITIONS
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        assert "manage_services" in names
        src = (_SRC / "tools" / "registry.py").read_text()
        assert '"manage_services": lambda' in src
        assert "active_service_ports" in src  # browser allowlist wired

    def test_docker_publishes_service_ports_with_conflict_retry(self):
        src = (_SRC / "sandbox" / "docker.py").read_text()
        assert "publishable_service_ports" in src
        assert "port is already allocated" in src


class TestPortConflictRetry:
    """Regression (caught live by the suite): a port-bind failure leaves the
    container CREATED-but-not-started. The retry-without-ports must REMOVE it
    first, or it dies with a 409 name-in-use — which propagated and bricked
    the sandbox entirely."""

    def test_source_removes_before_retry(self):
        """Pin the real source: the remove() must sit between the port-pop
        and the second containers.run."""
        src = (_SRC / "sandbox" / "docker.py").read_text()
        block = src.split("port is already allocated", 1)[1].split(
            "elif getattr(run_err", 1)[0]
        assert "remove(force=True)" in block
        assert "containers.run(**run_kwargs)" in block
        assert block.index("remove(force=True)") < \
            block.index("containers.run(**run_kwargs)")
