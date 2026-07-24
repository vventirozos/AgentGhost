"""2026-07-22 sandbox-infra review — docker.py fixes.

- Client-side exec deadline: a wedged daemon no longer hangs the worker
  thread forever (and thus every other turn behind self._lock).
- Adopted-container published-port derivation (from the live PortBindings).
- Stopped-container RESUME instead of destroy+reprovision.
- Spill-log counter seeded past existing files (no clobber after a restart).
- Infra failures marked distinctly (not presented as the model's own code).
"""
import time

import pytest
from unittest.mock import MagicMock, patch

from ghost_agent.sandbox.docker import DockerSandbox, SandboxDaemonTimeout


def _bare_sandbox():
    """A DockerSandbox with __init__ bypassed (no docker daemon needed)."""
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.container = MagicMock()
    return sb


class TestExecDeadline:
    def test_raises_on_daemon_wedge(self):
        sb = _bare_sandbox()

        def _hang(*a, **k):
            time.sleep(5)  # simulate a wedged daemon's blocking socket read
            return (0, b"")

        sb.container.exec_run.side_effect = _hang
        t0 = time.monotonic()
        with pytest.raises(SandboxDaemonTimeout):
            sb._exec_run("sleep 100", deadline_s=0.3)
        # Returned promptly (did not block for the full 5s hang).
        assert time.monotonic() - t0 < 2.0

    def test_returns_on_fast_exec(self):
        sb = _bare_sandbox()
        sb.container.exec_run.return_value = (0, b"OK")
        assert sb._exec_run("echo ok", deadline_s=5) == (0, b"OK")

    def test_propagates_real_exception(self):
        sb = _bare_sandbox()
        sb.container.exec_run.side_effect = ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            sb._exec_run("x", deadline_s=5)


class TestDerivePublishedPorts:
    def test_reads_port_bindings(self):
        c = MagicMock()
        c.attrs = {"HostConfig": {"PortBindings": {
            "8100/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8100"}],
            "8101/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8101"}],
        }}}
        assert DockerSandbox._derive_published_ports(c) == {8100, 8101}

    def test_empty_when_none(self):
        c = MagicMock()
        c.attrs = {"HostConfig": {"PortBindings": None}}
        assert DockerSandbox._derive_published_ports(c) == set()

    def test_empty_on_error(self):
        c = MagicMock()
        c.reload.side_effect = RuntimeError("gone")
        assert DockerSandbox._derive_published_ports(c) == set()


class TestResumeStopped:
    def test_stopped_container_is_resumed_not_recreated(self):
        sb = _bare_sandbox()
        sb.container.status = "exited"
        sb._last_ready_ok = 0.0
        with patch.object(DockerSandbox, "_is_container_ready", return_value=True):
            assert sb._try_resume_stopped() is True
        sb.container.start.assert_called_once()

    def test_running_container_not_resumed(self):
        sb = _bare_sandbox()
        sb.container.status = "running"
        assert sb._try_resume_stopped() is False
        sb.container.start.assert_not_called()

    def test_resume_that_isnt_ready_falls_through(self):
        sb = _bare_sandbox()
        sb.container.status = "exited"
        with patch.object(DockerSandbox, "_is_container_ready", return_value=False):
            assert sb._try_resume_stopped() is False  # → caller recreates


class TestSpillCounterSeed:
    def test_seeds_past_existing_logs(self, tmp_path):
        # A prior process left run_5.log; a fresh process (counter reset to 0)
        # must not clobber it with run_1.log.
        spill = tmp_path / ".ghost_runs"
        spill.mkdir()
        (spill / "run_5.log").write_text("old")
        sb = DockerSandbox.__new__(DockerSandbox)
        sb.host_workspace = tmp_path
        # Reset the class counter as a fresh process would have it.
        DockerSandbox._spill_counter = 0
        DockerSandbox._spill_counter_seeded = False
        try:
            rel = sb._spill_run_output("x" * 10)
            assert rel == ".ghost_runs/run_6.log"
            assert (spill / "run_5.log").read_text() == "old"  # not clobbered
        finally:
            DockerSandbox._spill_counter = 0
            DockerSandbox._spill_counter_seeded = False
