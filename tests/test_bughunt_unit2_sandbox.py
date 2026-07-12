"""Regression tests for bug-hunt unit 2 (sandbox/docker.py) — see BUGHUNT.md.

Fixed bugs pinned here:
1. image-cache commit now happens after EVERY successful provision (a
   container booted from a stale cached image previously never refreshed
   the cache, so every recreation re-paid the full multi-minute install)
2. self.image is never mutated to the cached tag (deleting the cache no
   longer bricks the fallback pull with a Docker Hub 404)
3. provisioning failure sets a backoff — no reinstall storm on every command
4. provisioning installs are wrapped in the in-container `timeout` binary
   (they block a worker thread while holding the provision lock)
5. GHOST_SANDBOX_CPU_QUOTA<=0 means "no cap" instead of a daemon error
6. a container that never becomes ready after creation raises loudly
7. readiness probe retries once on transient errors (a false negative
   force-removes a healthy container); NotFound stays definitive
8. marker/chromium probes run once per container generation, not per command
9. in-container tor spawn is attempted once per generation and verified
10. execute() no longer re-probes readiness after ensure_running()
11. the host workspace dir is recreated before container creation (the
    daemon would otherwise auto-create the bind source as root)
12. get_stats() snapshots the container handle
"""

import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.sandbox.docker import DockerSandbox


def _stub(workspace, tor_proxy=None, image="python:3.11-slim-bookworm"):
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.host_workspace = Path(workspace)
    sb.image = image
    sb.container_name = "ghost-test-bughunt2"
    sb.tor_proxy = tor_proxy
    sb.client = MagicMock()
    sb.docker_lib = MagicMock()

    class _ImageNotFound(Exception):
        pass

    sb.docker_lib.errors.ImageNotFound = _ImageNotFound
    sb.ImageNotFound = _ImageNotFound
    sb.NotFound = type("NotFound", (Exception,), {})
    sb.APIError = type("APIError", (Exception,), {})
    sb._lock = threading.Lock()
    return sb


def _ready_exec_map(overrides=None):
    """exec_run stub: marker present, chromium binary present, everything
    else succeeds. `overrides` maps a substring to a return value or a
    callable(cmd) -> return value."""
    overrides = overrides or {}

    def _exec_run(cmd, *args, **kwargs):
        for key, val in overrides.items():
            if key in cmd:
                return val(cmd) if callable(val) else val
        if "find /root/.cache/ms-playwright" in cmd:
            return (0, b"/root/.cache/ms-playwright/chromium-1/x/headless_shell\n")
        return (0, b"")

    return _exec_run


# ──────────────────────────────────────────────────────────────────────
# 1+4: commit after every successful provision; installs timeout-wrapped
# ──────────────────────────────────────────────────────────────────────

class TestProvisionCommitAndTimeouts:
    def test_commit_runs_even_when_booted_from_cached_image(self, tmp_path):
        # Pre-fix: `if self.image != "ghost-agent-base:latest"` skipped the
        # commit exactly when the container had booted from a STALE cached
        # image — the freshened environment was never written back.
        sb = _stub(tmp_path, image="ghost-agent-base:latest")
        sb.container = MagicMock()
        sb._is_container_ready = MagicMock(return_value=True)

        state = {"installed": False}

        def _find(cmd):
            if state["installed"]:
                return (0, b"/root/.cache/ms-playwright/chromium-1/x/headless_shell\n")
            return (0, b"")

        def _pw(cmd):
            state["installed"] = True
            return (0, b"")

        sb.container.exec_run.side_effect = _ready_exec_map({
            "test -f /root/.supercharged.v4": (1, b""),   # marker missing
            "find /root/.cache/ms-playwright": _find,
            "playwright install": _pw,
        })
        with patch("ghost_agent.sandbox.docker.pretty_log"):
            sb.ensure_running()

        sb.container.commit.assert_called_once_with(
            repository="ghost-agent-base", tag="latest"
        )

    def test_installs_are_wrapped_in_timeout(self, tmp_path):
        # The install exec_runs hold the provision lock; unbounded mirror
        # stalls must be bounded by the in-container `timeout` binary.
        sb = _stub(tmp_path)
        sb.container = MagicMock()
        sb._is_container_ready = MagicMock(return_value=True)

        seen = []

        def _record(cmd, *args, **kwargs):
            seen.append(cmd)
            if "find /root/.cache/ms-playwright" in cmd:
                return (0, b"/root/.cache/ms-playwright/chromium-1/x/headless_shell\n")
            if "test -f /root/.supercharged.v4" in cmd:
                return (1, b"")
            return (0, b"")

        sb.container.exec_run.side_effect = _record
        with patch("ghost_agent.sandbox.docker.pretty_log"):
            sb.ensure_running()

        installs = [c for c in seen if "apt-get install" in c or "pip install" in c
                    or "playwright install" in c]
        assert installs, "provision path did not run"
        for cmd in installs:
            assert cmd.startswith("timeout "), f"unbounded install: {cmd}"


# ──────────────────────────────────────────────────────────────────────
# 2: self.image never mutated; cache deletion can't brick the pull
# ──────────────────────────────────────────────────────────────────────

class TestImageNotMutated:
    def _fresh_container_sandbox(self, tmp_path):
        sb = _stub(tmp_path)
        sb.container = None
        sb.client.containers.get.side_effect = sb.NotFound("none")
        new_container = MagicMock()
        new_container.exec_run.side_effect = _ready_exec_map()
        sb.client.containers.run.return_value = new_container
        return sb

    def test_cached_image_used_for_boot_but_self_image_untouched(self, tmp_path):
        sb = self._fresh_container_sandbox(tmp_path)
        sb.client.images.get.return_value = MagicMock()  # cache exists
        with patch.object(type(sb), "_is_container_ready", return_value=True), \
             patch("ghost_agent.sandbox.docker.pretty_log"), \
             patch("ghost_agent.sandbox.docker.time.sleep"):
            sb.ensure_running()
        assert sb.image == "python:3.11-slim-bookworm"
        assert sb.client.containers.run.call_args.kwargs["image"] == "ghost-agent-base:latest"

    def test_pull_targets_base_image_when_cache_missing(self, tmp_path):
        # Pre-fix: after one cached boot, self.image was permanently
        # "ghost-agent-base:latest"; if the cache was rmi'd, the fallback
        # pulled that tag from Docker Hub (404) instead of the base image.
        sb = self._fresh_container_sandbox(tmp_path)
        sb.client.images.get.side_effect = [
            sb.ImageNotFound("no cache"), sb.ImageNotFound("no base"),
        ]
        with patch.object(type(sb), "_is_container_ready", return_value=True), \
             patch("ghost_agent.sandbox.docker.pretty_log"), \
             patch("ghost_agent.sandbox.docker.time.sleep"):
            sb.ensure_running()
        sb.client.images.pull.assert_called_once_with("python:3.11-slim-bookworm")


# ──────────────────────────────────────────────────────────────────────
# 3: provisioning failure backoff
# ──────────────────────────────────────────────────────────────────────

class TestProvisionBackoff:
    def test_failed_install_backs_off_then_recovers(self, tmp_path):
        sb = _stub(tmp_path)
        sb.container = MagicMock()
        sb._is_container_ready = MagicMock(return_value=True)

        apt_calls = []

        def _apt(cmd):
            apt_calls.append(cmd)
            return (1, b"mirror down")

        sb.container.exec_run.side_effect = _ready_exec_map({
            "test -f /root/.supercharged.v4": (1, b""),
            "apt-get install": _apt,
        })

        with patch("ghost_agent.sandbox.docker.pretty_log"):
            with pytest.raises(Exception, match="System package installation failed"):
                sb.ensure_running()
            assert len(apt_calls) == 1

            # Pre-fix: every subsequent command re-ran the full failing
            # install under the provision lock. Now: fast, explicit error.
            with pytest.raises(Exception, match="failed recently"):
                sb.ensure_running()
            assert len(apt_calls) == 1

            # Backoff expiry → a real retry is allowed again.
            sb._provision_backoff_until = 0.0
            with pytest.raises(Exception, match="System package installation failed"):
                sb.ensure_running()
            assert len(apt_calls) == 2


# ──────────────────────────────────────────────────────────────────────
# 5: CPU quota <= 0 disables the cap
# ──────────────────────────────────────────────────────────────────────

class TestCpuQuotaZero:
    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_nonpositive_quota_omits_cpu_kwargs(self, tmp_path, monkeypatch, value):
        monkeypatch.setenv("GHOST_SANDBOX_CPU_QUOTA", value)
        sb = _stub(tmp_path)
        sb.container = None
        sb.client.containers.get.side_effect = sb.NotFound("none")
        sb.client.images.get.return_value = MagicMock()
        new_container = MagicMock()
        new_container.exec_run.side_effect = _ready_exec_map()
        sb.client.containers.run.return_value = new_container

        with patch.object(type(sb), "_is_container_ready", return_value=True), \
             patch("ghost_agent.sandbox.docker.pretty_log"), \
             patch("ghost_agent.sandbox.docker.time.sleep"):
            sb.ensure_running()

        kwargs = sb.client.containers.run.call_args.kwargs
        # Pre-fix: cpu_quota=0 reached the daemon → "CPU cfs quota cannot
        # be less than 1ms" → container creation bricked.
        assert "cpu_quota" not in kwargs
        assert "cpu_period" not in kwargs


# ──────────────────────────────────────────────────────────────────────
# 6: never-ready container raises instead of proceeding
# ──────────────────────────────────────────────────────────────────────

class TestReadinessTimeoutRaises:
    def test_raises_when_container_never_ready(self, tmp_path):
        sb = _stub(tmp_path)
        sb.container = None
        sb.client.containers.get.side_effect = sb.NotFound("none")
        sb.client.images.get.return_value = MagicMock()
        sb.client.containers.run.return_value = MagicMock()

        with patch.object(type(sb), "_is_container_ready", return_value=False), \
             patch("ghost_agent.sandbox.docker.pretty_log"), \
             patch("ghost_agent.sandbox.docker.time.sleep"):
            with pytest.raises(Exception, match="did not become ready"):
                sb.ensure_running()


# ──────────────────────────────────────────────────────────────────────
# 7: readiness probe retry semantics
# ──────────────────────────────────────────────────────────────────────

class TestProbeRetry:
    def test_transient_error_gets_one_retry(self, tmp_path):
        sb = _stub(tmp_path)
        sb._probe_container_ready = MagicMock(side_effect=[RuntimeError("hiccup"), True])
        with patch("ghost_agent.sandbox.docker.time.sleep"):
            assert sb._is_container_ready() is True
        assert sb._probe_container_ready.call_count == 2

    def test_two_failures_conclude_not_ready(self, tmp_path):
        sb = _stub(tmp_path)
        sb._probe_container_ready = MagicMock(side_effect=RuntimeError("down"))
        with patch("ghost_agent.sandbox.docker.time.sleep"):
            assert sb._is_container_ready() is False
        assert sb._probe_container_ready.call_count == 2

    def test_notfound_is_definitive_no_retry(self, tmp_path):
        sb = _stub(tmp_path)
        sb._probe_container_ready = MagicMock(side_effect=sb.NotFound("gone"))
        assert sb._is_container_ready() is False
        assert sb._probe_container_ready.call_count == 1


# ──────────────────────────────────────────────────────────────────────
# 8+9: per-generation caching of env checks and the tor attempt
# ──────────────────────────────────────────────────────────────────────

class TestPerGenerationCaching:
    def test_marker_probes_run_once_per_generation(self, tmp_path):
        sb = _stub(tmp_path)
        sb.container = MagicMock()
        sb.container.exec_run.side_effect = _ready_exec_map()
        sb._is_container_ready = MagicMock(return_value=True)

        with patch("ghost_agent.sandbox.docker.pretty_log"):
            sb.ensure_running()
            first = sb.container.exec_run.call_count
            assert first > 0  # marker + chromium probes ran
            sb.ensure_running()
            sb.ensure_running()
        assert sb.container.exec_run.call_count == first
        assert sb._env_verified is True

    def test_tor_spawn_attempted_once_per_generation(self, tmp_path):
        sb = _stub(tmp_path, tor_proxy="socks5://127.0.0.1:9050")
        sb.container = MagicMock()
        spawns = []

        def _spawn(cmd):
            spawns.append(cmd)
            return (0, b"")

        sb.container.exec_run.side_effect = _ready_exec_map({
            "pgrep -x tor": (1, b""),          # never running (host owns :9050)
            "RunAsDaemon": _spawn,
        })
        sb._is_container_ready = MagicMock(return_value=True)

        with patch("ghost_agent.sandbox.docker.pretty_log"), \
             patch("ghost_agent.sandbox.docker.time.sleep"):
            sb.ensure_running()
            sb.ensure_running()
            sb.ensure_running()
        # Pre-fix: the doomed spawn re-ran on EVERY command, flooding the
        # log with per-command "Environment Ready" lines.
        assert len(spawns) == 1
        assert sb._tor_attempted is True


# ──────────────────────────────────────────────────────────────────────
# 10: execute() trusts ensure_running's probe
# ──────────────────────────────────────────────────────────────────────

class TestExecuteSingleProbe:
    def test_execute_probes_readiness_exactly_once(self, tmp_path):
        sb = _stub(tmp_path)
        sb.container = MagicMock()
        sb._is_container_ready = MagicMock(return_value=True)

        def _exec_run(cmd, *args, **kwargs):
            if "test -f /root/.supercharged.v4" in cmd:
                return (0, b"")
            if "find /root/.cache/ms-playwright" in cmd:
                return (0, b"/root/.cache/ms-playwright/chromium-1/x/headless_shell\n")
            return SimpleNamespace(output=b"hello", exit_code=0)

        sb.container.exec_run.side_effect = _exec_run
        with patch("ghost_agent.sandbox.docker.pretty_log"):
            out, code = sb.execute("echo hello")

        assert out == "hello"
        assert code == 0
        # Pre-fix this was 2 (ensure_running + a duplicate in execute).
        assert sb._is_container_ready.call_count == 1


# ──────────────────────────────────────────────────────────────────────
# 11: host workspace recreated before container creation
# ──────────────────────────────────────────────────────────────────────

class TestWorkspaceMkdir:
    def test_missing_workspace_dir_is_recreated(self, tmp_path):
        gone = tmp_path / "vanished" / "workspace"
        sb = _stub(gone)
        sb.container = None
        sb.client.containers.get.side_effect = sb.NotFound("none")
        sb.client.images.get.return_value = MagicMock()
        new_container = MagicMock()
        new_container.exec_run.side_effect = _ready_exec_map()
        sb.client.containers.run.return_value = new_container

        assert not gone.exists()
        with patch.object(type(sb), "_is_container_ready", return_value=True), \
             patch("ghost_agent.sandbox.docker.pretty_log"), \
             patch("ghost_agent.sandbox.docker.time.sleep"):
            sb.ensure_running()
        # Pre-fix the docker daemon auto-created this as root-owned,
        # making the host-side readiness touch fail forever after.
        assert gone.is_dir()


# ──────────────────────────────────────────────────────────────────────
# 12: get_stats handle snapshot
# ──────────────────────────────────────────────────────────────────────

class TestGetStats:
    def test_none_container(self, tmp_path):
        sb = _stub(tmp_path)
        sb.container = None
        assert sb.get_stats() is None

    def test_stats_error_returns_none(self, tmp_path):
        sb = _stub(tmp_path)
        sb.container = MagicMock()
        sb.container.stats.side_effect = RuntimeError("removed")
        assert sb.get_stats() is None
