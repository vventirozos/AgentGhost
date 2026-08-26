"""Lazy sandbox re-init (sandbox/docker.py, 2026-08-26).

Pins the recovery path added after the OrbStack boot race: the agent
booted seconds before the docker socket answered, the DockerSandbox
constructor raised, ``context.sandbox_manager`` stayed None for 7 hours,
and every execute/browser call failed with "Sandbox manager not
initialized" while docker itself had recovered minutes after boot.

Each test names the world where it fails:
- eligibility tests fail if the identity guard is dropped (a copied
  isolated-replay context would resurrect a deliberately detached sandbox);
- backoff tests fail if every tool call starts paying for a docker ping
  while the daemon is down;
- recovery tests fail if a successful construction is not assigned back
  onto the context (the whole point of the feature).
"""

import copy
import threading
import types
from pathlib import Path

import pytest

from ghost_agent.sandbox import docker as docker_mod


class _Ctx:
    """Minimal stand-in for GhostContext (plain class → weakref-able,
    matching the real one)."""

    def __init__(self, tmp_path):
        self.sandbox_dir = Path(tmp_path)
        self.tor_proxy = None
        self.sandbox_manager = None


@pytest.fixture(autouse=True)
def _reset_module_state():
    """The helpers keep module-level state (registration weakref, backoff
    stamp). Restore it around every test so order can't leak."""
    saved = (docker_mod._lazy_ctx_ref, docker_mod._lazy_next_attempt)
    docker_mod._lazy_ctx_ref = None
    docker_mod._lazy_next_attempt = 0.0
    yield
    docker_mod._lazy_ctx_ref, docker_mod._lazy_next_attempt = saved


class _ForbiddenConstruction(BaseException):
    """BaseException on purpose: ensure_sandbox_manager wraps the
    constructor in `except Exception`, so a plain AssertionError sentinel
    would be SWALLOWED and the guard tests would pass vacuously."""


@pytest.fixture
def forbid_construction(monkeypatch):
    """Fail the test if DockerSandbox is constructed at all."""

    def _boom(*a, **k):
        raise _ForbiddenConstruction(
            "DockerSandbox constructed when it must not be")

    monkeypatch.setattr(docker_mod, "DockerSandbox", _boom)


def _allow_construction(monkeypatch, result=None, exc=None):
    """Replace DockerSandbox with a counter; returns the call-count box."""
    calls = {"n": 0}

    def _ctor(sandbox_dir, tor_proxy=None):
        calls["n"] += 1
        if exc is not None:
            raise exc
        sentinel = types.SimpleNamespace(sandbox_dir=sandbox_dir,
                                         tor_proxy=tor_proxy)
        return result if result is not None else sentinel

    monkeypatch.setattr(docker_mod, "DockerSandbox", _ctor)
    return calls


def test_existing_manager_returned_untouched(tmp_path, forbid_construction):
    ctx = _Ctx(tmp_path)
    ctx.sandbox_manager = object()
    # No registration needed — the fast path must not construct anything.
    assert docker_mod.ensure_sandbox_manager(ctx) is ctx.sandbox_manager


def test_unregistered_context_never_attempts(tmp_path, forbid_construction):
    ctx = _Ctx(tmp_path)
    assert docker_mod.ensure_sandbox_manager(ctx) is None


def test_copied_context_not_eligible(tmp_path, forbid_construction):
    """A copy.copy'd context (isolated_replay_context's fork) must NEVER
    lazily rebuild a sandbox — it may have detached its manager on
    purpose for a network=none replay (§4CL)."""
    ctx = _Ctx(tmp_path)
    docker_mod.register_lazy_sandbox(ctx)
    fork = copy.copy(ctx)
    fork.sandbox_manager = None
    assert docker_mod.ensure_sandbox_manager(fork) is None


def test_recovery_assigns_manager_onto_context(tmp_path, monkeypatch):
    ctx = _Ctx(tmp_path)
    docker_mod.register_lazy_sandbox(ctx)
    monkeypatch.setattr(docker_mod, "_docker_endpoint_plausible", lambda: True)
    calls = _allow_construction(monkeypatch)
    got = docker_mod.ensure_sandbox_manager(ctx)
    assert got is not None
    assert ctx.sandbox_manager is got  # assigned back — other readers recover
    assert calls["n"] == 1
    # Second call takes the fast path, no new construction.
    assert docker_mod.ensure_sandbox_manager(ctx) is got
    assert calls["n"] == 1


def test_failure_arms_backoff_and_retries_after_it(tmp_path, monkeypatch):
    ctx = _Ctx(tmp_path)
    docker_mod.register_lazy_sandbox(ctx)
    monkeypatch.setattr(docker_mod, "_docker_endpoint_plausible", lambda: True)
    calls = _allow_construction(monkeypatch, exc=RuntimeError("daemon down"))

    assert docker_mod.ensure_sandbox_manager(ctx) is None
    assert calls["n"] == 1
    assert ctx.sandbox_manager is None

    # Within the backoff window: no second construction attempt.
    assert docker_mod.ensure_sandbox_manager(ctx) is None
    assert calls["n"] == 1

    # Backoff elapsed: it tries again (and can now succeed).
    docker_mod._lazy_next_attempt = 0.0
    _allow_construction(monkeypatch)  # replace the raising ctor
    got = docker_mod.ensure_sandbox_manager(ctx)
    assert got is not None and ctx.sandbox_manager is got


def test_endpoint_precheck_blocks_attempt(tmp_path, monkeypatch):
    """No plausible daemon endpoint → no construction (the wedged-daemon
    ping can block for the client timeout; a stat() must gate it)."""
    ctx = _Ctx(tmp_path)
    docker_mod.register_lazy_sandbox(ctx)
    monkeypatch.setattr(docker_mod, "_docker_endpoint_plausible",
                        lambda: False)
    calls = _allow_construction(monkeypatch)
    assert docker_mod.ensure_sandbox_manager(ctx) is None
    assert calls["n"] == 0
    # The failed pre-check still arms the backoff — the stat() is cheap
    # but the gate keeps the None path O(1) between windows.
    assert docker_mod._lazy_next_attempt > 0.0


def test_registration_resets_backoff(tmp_path, monkeypatch):
    """A fresh boot registration must not inherit a stale backoff stamp."""
    ctx = _Ctx(tmp_path)
    docker_mod._lazy_next_attempt = float("inf")
    docker_mod.register_lazy_sandbox(ctx)
    monkeypatch.setattr(docker_mod, "_docker_endpoint_plausible", lambda: True)
    calls = _allow_construction(monkeypatch)
    assert docker_mod.ensure_sandbox_manager(ctx) is not None
    assert calls["n"] == 1


def test_concurrent_attempt_never_double_constructs(tmp_path, monkeypatch):
    """While one thread holds the attempt lock, another caller returns
    None immediately instead of queueing a second construction."""
    ctx = _Ctx(tmp_path)
    docker_mod.register_lazy_sandbox(ctx)
    monkeypatch.setattr(docker_mod, "_docker_endpoint_plausible", lambda: True)

    entered = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def _slow_ctor(sandbox_dir, tor_proxy=None):
        calls["n"] += 1
        entered.set()
        release.wait(timeout=5)
        return types.SimpleNamespace()

    monkeypatch.setattr(docker_mod, "DockerSandbox", _slow_ctor)

    results = {}
    t = threading.Thread(
        target=lambda: results.__setitem__(
            "bg", docker_mod.ensure_sandbox_manager(ctx)))
    t.start()
    assert entered.wait(timeout=5)
    # Constructor in flight on the other thread: this call must bail out.
    assert docker_mod.ensure_sandbox_manager(ctx) is None
    release.set()
    t.join(timeout=5)
    assert calls["n"] == 1
    assert results["bg"] is not None
    assert ctx.sandbox_manager is results["bg"]


def test_never_raises_on_weird_context(forbid_construction):
    """A context missing attributes entirely still returns None."""
    assert docker_mod.ensure_sandbox_manager(
        types.SimpleNamespace()) is None


def test_failed_construction_closes_carried_client(tmp_path, monkeypatch):
    """§4BO: the constructor attaches its half-built docker client to the
    exception (~11 unix sockets). Boot leaked it at most once; the lazy
    path retries every backoff window, so it MUST close it or a long
    docker outage walks the process to EMFILE."""
    from unittest.mock import MagicMock

    ctx = _Ctx(tmp_path)
    docker_mod.register_lazy_sandbox(ctx)
    monkeypatch.setattr(docker_mod, "_docker_endpoint_plausible", lambda: True)

    carried = MagicMock()

    def _ctor(*a, **k):
        e = RuntimeError("daemon ping failed")
        e.client = carried
        raise e

    monkeypatch.setattr(docker_mod, "DockerSandbox", _ctor)
    assert docker_mod.ensure_sandbox_manager(ctx) is None
    carried.close.assert_called_once()


def test_close_carried_client_variants():
    """The shared §4BO closer (used by the lazy path AND dream.py's
    self-play construction): closes whichever attr carries a client,
    tolerates absence, and swallows a close() that raises."""
    from unittest.mock import MagicMock

    e = RuntimeError("x")
    e.client = MagicMock()
    docker_mod.close_carried_client(e)
    e.client.close.assert_called_once()

    e2 = RuntimeError("x")
    e2.docker_client = MagicMock()
    docker_mod.close_carried_client(e2)
    e2.docker_client.close.assert_called_once()

    # No carried client, and a close() that raises: both must be silent.
    docker_mod.close_carried_client(RuntimeError("bare"))
    e3 = RuntimeError("x")
    e3.client = MagicMock()
    e3.client.close.side_effect = OSError("already dead")
    docker_mod.close_carried_client(e3)  # must not raise


def test_ctor_fallback_client_closed_on_ping_failure(tmp_path, monkeypatch):
    """The darwin fallback client is built AFTER handle_err.client was
    attached, so the caller can never close it — the constructor must
    close it itself when its ping fails, else every failed construction
    (now recurring, per backoff window) leaks its sockets."""
    import sys as real_sys
    from unittest.mock import MagicMock, patch

    class FakeDockerException(Exception):
        pass

    fallback_client = MagicMock()
    fallback_client.ping.side_effect = FakeDockerException("wedged")

    fake_docker = MagicMock()
    fake_docker.errors.DockerException = FakeDockerException
    fake_docker.from_env.side_effect = FakeDockerException("no socket")
    fake_docker.DockerClient.return_value = fallback_client

    fake_errors = MagicMock()
    fake_errors.NotFound = type("NF", (Exception,), {})
    fake_errors.APIError = type("AE", (Exception,), {})

    monkeypatch.setattr(real_sys, "platform", "darwin")
    real_exists = docker_mod.os.path.exists
    sock_suffixes = (".orbstack/run/docker.sock", ".docker/run/docker.sock")
    monkeypatch.setattr(
        docker_mod.os.path, "exists",
        lambda p: True if str(p).endswith(sock_suffixes) else real_exists(p))

    with patch.dict(real_sys.modules, {"docker": fake_docker,
                                       "docker.errors": fake_errors}):
        with pytest.raises(FakeDockerException):
            docker_mod.DockerSandbox(host_workspace=Path(tmp_path))

    fallback_client.close.assert_called_once()
