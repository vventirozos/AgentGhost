"""Sandbox readiness probe: single-exec + success TTL (IMPROVEMENTS.md #8).

`execute()` calls `ensure_running()` before EVERY command. The full readiness
probe was 3 docker round-trips (reload + host touch/stat + echo ≈ 100-400ms),
serialized under the lock — a fixed tax on every rg / find / execute / browser
op. Two fixes:
  - `_probe_container_ready` collapses to ONE exec (`stat <sync> && echo OK`;
    the reload was redundant — a dead container fails the exec anyway).
  - A success TTL: after a command exits without an infra error, a probe within
    `_READY_TTL_S` is skipped entirely. Any exec failure / exit 126-128 clears
    the stamp so a broken container/mount still triggers the recreate path.
"""
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.sandbox.docker import DockerSandbox


def _sandbox():
    sb = DockerSandbox(Path("/tmp/workspace"))
    sb.container = MagicMock()
    sb.container.status = "running"
    return sb


def test_probe_is_single_exec_no_reload():
    sb = _sandbox()
    sb.container.exec_run.return_value = (0, b"OK\n")
    with patch("pathlib.Path.touch"), patch("pathlib.Path.exists", return_value=False):
        assert sb._probe_container_ready() is True
    # Exactly one exec, and reload() must NOT be part of the probe anymore.
    assert sb.container.exec_run.call_count == 1
    sb.container.reload.assert_not_called()
    cmd = sb.container.exec_run.call_args.args[0]
    assert "stat" in cmd and "echo OK" in cmd


def test_probe_fails_on_nonzero_exit():
    sb = _sandbox()
    sb.container.exec_run.return_value = (128, b"OCI runtime exec failed")
    with patch("pathlib.Path.touch"), patch("pathlib.Path.exists", return_value=False):
        assert sb._probe_container_ready() is False


def test_fresh_ready_skips_probe():
    sb = _sandbox()
    sb.mark_ready()
    with patch.object(sb, "_is_container_ready") as probe:
        sb._ensure_running_impl()
        probe.assert_not_called()


def test_stale_ready_runs_probe():
    sb = _sandbox()
    # Environment already provisioned this generation → skip the install path
    # and isolate the readiness-probe decision.
    sb._env_verified = True
    sb._tor_attempted = True
    sb.mark_ready()
    # Force the stamp older than the TTL.
    sb._last_ready_ok = time.monotonic() - (sb._READY_TTL_S + 1)
    with patch.object(sb, "_is_container_ready", return_value=True) as probe:
        sb._ensure_running_impl()
        probe.assert_called_once()


def test_invalidate_forces_next_probe():
    sb = _sandbox()
    sb.mark_ready()
    assert sb._ready_is_fresh() is True
    sb.invalidate_ready()
    assert sb._ready_is_fresh() is False


def test_successful_exec_stamps_ready():
    sb = _sandbox()
    sb.invalidate_ready()
    sb.container.exec_run.return_value = MagicMock(output=b"done", exit_code=0)
    with patch.object(sb, "ensure_running"):
        out, code = sb.execute("echo hi")
    assert code == 0
    assert sb._ready_is_fresh() is True


def test_failing_user_command_still_stamps_ready():
    """A non-zero USER exit (a failing script) is NOT an infra fault — it must
    still confirm readiness so the TTL keeps working."""
    sb = _sandbox()
    sb.invalidate_ready()
    sb.container.exec_run.return_value = MagicMock(output=b"boom", exit_code=1)
    with patch.object(sb, "ensure_running"):
        _out, code = sb.execute("false")
    assert code == 1
    assert sb._ready_is_fresh() is True


@pytest.mark.parametrize("oci_code", [126, 127, 128])
def test_oci_exit_codes_invalidate_ready(oci_code):
    sb = _sandbox()
    sb.mark_ready()
    sb.container.exec_run.return_value = MagicMock(output=b"", exit_code=oci_code)
    with patch.object(sb, "ensure_running"):
        sb.execute("weird")
    assert sb._ready_is_fresh() is False


def test_exec_exception_invalidates_ready():
    sb = _sandbox()
    sb.mark_ready()
    sb.container.exec_run.side_effect = RuntimeError("daemon gone")
    with patch.object(sb, "ensure_running"):
        out, code = sb.execute("cmd")
    assert code == 1
    assert sb._ready_is_fresh() is False
