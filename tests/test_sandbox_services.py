"""Tests for supervised long-lived sandbox services
(sandbox/services.py + tools/sandbox_services.py + the browser SSRF
service-port allowlist + docker.py port publishing — 2026-07-11)."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import json
import types
from pathlib import Path

import pytest

from ghost_agent.sandbox.services import (
    BLOCKED_PORTS, MAX_SERVICES,
    ServiceSupervisor, default_service_ports, publishable_service_ports,
    get_service_supervisor, active_service_ports,
)
from ghost_agent.tools.sandbox_services import tool_manage_services


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
        return self.handler(cmd)


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
