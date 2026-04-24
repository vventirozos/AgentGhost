"""Unit tests for sandbox/docker.py audit fixes (#14, #15)."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture
def mock_docker_env(monkeypatch):
    """Build a fake `docker` library so ``DockerSandbox.__init__`` succeeds
    without touching the host docker daemon."""
    import sys
    import types

    class _ImageNotFound(Exception):
        pass

    class _APIError(Exception):
        pass

    class _NotFound(Exception):
        pass

    class _DockerException(Exception):
        pass

    # Build the `docker.errors` submodule
    fake_errors = types.SimpleNamespace(
        ImageNotFound=_ImageNotFound,
        APIError=_APIError,
        NotFound=_NotFound,
        DockerException=_DockerException,
    )

    fake_docker = types.ModuleType("docker")
    fake_docker.errors = fake_errors
    fake_docker.from_env = lambda: MagicMock(name="docker_client_from_env")
    fake_docker.DockerClient = MagicMock(name="DockerClient")

    fake_docker_errors = types.ModuleType("docker.errors")
    fake_docker_errors.ImageNotFound = _ImageNotFound
    fake_docker_errors.APIError = _APIError
    fake_docker_errors.NotFound = _NotFound
    fake_docker_errors.DockerException = _DockerException

    monkeypatch.setitem(sys.modules, "docker", fake_docker)
    monkeypatch.setitem(sys.modules, "docker.errors", fake_docker_errors)

    return {
        "ImageNotFound": _ImageNotFound,
        "APIError": _APIError,
        "NotFound": _NotFound,
        "DockerException": _DockerException,
    }


def _build_sandbox(tmp_path, errors):
    """Construct a DockerSandbox with the in-test docker fakes wired in."""
    if "ghost_agent.sandbox.docker" in sys.modules:
        del sys.modules["ghost_agent.sandbox.docker"]
    from ghost_agent.sandbox import docker as docker_mod

    sb = docker_mod.DockerSandbox(host_workspace=tmp_path, tor_proxy=None)
    # Replace the live client with a fully scriptable mock
    sb.client = MagicMock()
    sb.NotFound = errors["NotFound"]
    sb.APIError = errors["APIError"]
    return sb, docker_mod


# ---------------------------------------------------------------------------
# Fix #14: ensure_running uses ImageNotFound branch only when image missing
# ---------------------------------------------------------------------------
class TestImagePullPath:
    def test_skips_pull_when_image_present(self, tmp_path, mock_docker_env):
        sb, _ = _build_sandbox(tmp_path, mock_docker_env)

        # No existing container, no existing supercharged marker
        sb.client.containers.get.side_effect = sb.NotFound("no container")

        # `images.get` succeeds for both lookups (cached + final), so pull
        # should NEVER be called.
        sb.client.images.get.return_value = MagicMock()

        # `containers.run` returns a mock that pretends to be ready
        new_container = MagicMock()
        new_container.status = "running"
        new_container.exec_run.return_value = (0, b"OK")
        sb.client.containers.run.return_value = new_container

        # Stub out the readiness probe to short-circuit
        with patch.object(type(sb), "_is_container_ready", return_value=True):
            sb.ensure_running()

        sb.client.images.pull.assert_not_called()

    def test_pulls_only_when_image_not_found(self, tmp_path, mock_docker_env):
        sb, _ = _build_sandbox(tmp_path, mock_docker_env)
        sb.client.containers.get.side_effect = sb.NotFound("no container")

        # First images.get = the cached "ghost-agent-base:latest" lookup,
        # which we want to FAIL so the code falls through to the base image.
        # Second images.get = the base image lookup, which we want to FAIL
        # so the pull branch fires.
        cached_err = mock_docker_env["ImageNotFound"]("no cached image")
        base_err = mock_docker_env["ImageNotFound"]("no base image")
        sb.client.images.get.side_effect = [cached_err, base_err]

        new_container = MagicMock()
        new_container.status = "running"
        new_container.exec_run.return_value = (0, b"OK")
        sb.client.containers.run.return_value = new_container

        with patch.object(type(sb), "_is_container_ready", return_value=True):
            sb.ensure_running()

        # Pull MUST have been called for the base image
        sb.client.images.pull.assert_called_once()

    def test_pull_failure_is_tolerated(self, tmp_path, mock_docker_env):
        sb, _ = _build_sandbox(tmp_path, mock_docker_env)
        sb.client.containers.get.side_effect = sb.NotFound("no container")

        # Both cached + base lookup miss; pull then errors
        sb.client.images.get.side_effect = [
            mock_docker_env["ImageNotFound"]("no cached"),
            mock_docker_env["ImageNotFound"]("no base"),
        ]
        sb.client.images.pull.side_effect = RuntimeError("registry timeout")

        new_container = MagicMock()
        new_container.status = "running"
        new_container.exec_run.return_value = (0, b"OK")
        sb.client.containers.run.return_value = new_container

        # Should NOT raise — ensure_running tolerates pull failures and
        # lets the subsequent `containers.run` surface the real problem.
        with patch.object(type(sb), "_is_container_ready", return_value=True):
            sb.ensure_running()


# ---------------------------------------------------------------------------
# Fix #15: CPU limit injection
# ---------------------------------------------------------------------------
class TestCpuLimit:
    def test_default_cpu_quota_is_200000(self, tmp_path, mock_docker_env, monkeypatch):
        monkeypatch.delenv("GHOST_SANDBOX_CPU_QUOTA", raising=False)
        sb, _ = _build_sandbox(tmp_path, mock_docker_env)
        sb.client.containers.get.side_effect = sb.NotFound("no container")
        sb.client.images.get.return_value = MagicMock()
        new_container = MagicMock()
        new_container.status = "running"
        new_container.exec_run.return_value = (0, b"OK")
        sb.client.containers.run.return_value = new_container

        with patch.object(type(sb), "_is_container_ready", return_value=True):
            sb.ensure_running()

        kwargs = sb.client.containers.run.call_args.kwargs
        assert kwargs.get("cpu_period") == 100000, kwargs
        assert kwargs.get("cpu_quota") == 200000, kwargs

    def test_env_override_is_honoured(self, tmp_path, mock_docker_env, monkeypatch):
        monkeypatch.setenv("GHOST_SANDBOX_CPU_QUOTA", "350000")
        sb, _ = _build_sandbox(tmp_path, mock_docker_env)
        sb.client.containers.get.side_effect = sb.NotFound("no container")
        sb.client.images.get.return_value = MagicMock()
        new_container = MagicMock()
        new_container.status = "running"
        new_container.exec_run.return_value = (0, b"OK")
        sb.client.containers.run.return_value = new_container

        with patch.object(type(sb), "_is_container_ready", return_value=True):
            sb.ensure_running()

        kwargs = sb.client.containers.run.call_args.kwargs
        assert kwargs.get("cpu_quota") == 350000, kwargs
