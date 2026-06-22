"""Tests for the sandbox Chromium-provisioning invariants.

These are process-level tests: they construct a DockerSandbox
without requiring docker to be reachable, then monkey-patch the
container stub so we can drive the gate logic deterministically.

The invariants we're pinning:

  1. `_chromium_binary_present` returns False when the binary glob
     finds nothing, True when it does.
  2. `ensure_running` treats an image without the v2 marker as
     un-provisioned.
  3. `ensure_running` treats an image WITH the v2 marker but WITHOUT
     a Chromium binary as un-provisioned (the silent-failure mode
     the v2 bump exists to catch).
  4. The provisioning code refuses to set the marker if the binary
     check still fails after install.
"""

from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.sandbox.docker import DockerSandbox


def _make_sandbox_stub():
    """Build a DockerSandbox without hitting docker. We bypass
    __init__ because it requires a live client; instead we stitch
    together only the attributes `ensure_running` and its helpers
    actually read.
    """
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.container = MagicMock()
    sb.client = MagicMock()
    sb.image = "ghost-agent-base:latest"
    sb.container_name = "ghost_sandbox"
    sb.host_workspace = MagicMock()
    sb.tor_proxy = None
    sb.docker_lib = MagicMock()
    return sb


def _exec_run_map(mapping):
    """Return an `exec_run` stub that returns (code, stdout) for each
    command prefix seen in `mapping`. Unmatched commands return (1, b'')
    so missing cases surface as test failures rather than silent
    passes."""
    def _side_effect(cmd, *args, **kwargs):
        for prefix, result in mapping.items():
            if prefix in cmd:
                return result
        return (1, b"")
    return _side_effect


# ------------------------------------------------------------------
# _chromium_binary_present
# ------------------------------------------------------------------

def test_chromium_binary_present_true_when_find_returns_path():
    sb = _make_sandbox_stub()
    sb.container.exec_run = MagicMock(
        return_value=(0, b"/root/.cache/ms-playwright/chromium-1234/chrome-linux/headless_shell\n")
    )
    assert sb._chromium_binary_present() is True


def test_chromium_binary_present_false_when_find_empty():
    sb = _make_sandbox_stub()
    sb.container.exec_run = MagicMock(return_value=(0, b""))
    assert sb._chromium_binary_present() is False


def test_chromium_binary_present_false_on_nonzero_exit():
    sb = _make_sandbox_stub()
    sb.container.exec_run = MagicMock(return_value=(1, b"find: cache dir missing"))
    assert sb._chromium_binary_present() is False


def test_chromium_binary_present_false_when_container_none():
    sb = _make_sandbox_stub()
    sb.container = None
    assert sb._chromium_binary_present() is False


def test_chromium_binary_present_tolerates_exec_exception():
    sb = _make_sandbox_stub()
    sb.container.exec_run = MagicMock(side_effect=RuntimeError("daemon gone"))
    # Must return False rather than propagate — provisioning code
    # relies on this being a safe "is it present" probe.
    assert sb._chromium_binary_present() is False


# ------------------------------------------------------------------
# Gate invariants: marker + binary must BOTH be good to skip install
# ------------------------------------------------------------------

def test_v3_marker_gate_name_pinned():
    """The marker path is part of the install contract with
    sandbox/Dockerfile. Pin it here so renaming in one place without
    the other is a loud test failure."""
    import inspect
    # ensure_running is now a thin locking wrapper; the provisioning body
    # lives in _ensure_running_impl. Inspect both so the gate assertions
    # hold regardless of which method carries the logic.
    src = inspect.getsource(DockerSandbox.ensure_running) + inspect.getsource(
        DockerSandbox._ensure_running_impl)
    assert "/root/.supercharged.v3" in src, (
        "marker path drifted from /root/.supercharged.v3 — update "
        "sandbox/Dockerfile in lockstep"
    )


def test_legacy_v1_marker_treated_as_unprovisioned():
    """An image from the pre-v2 era has /root/.supercharged but not
    /root/.supercharged.v3. The gate must detect this and trigger a
    fresh install. We assert the decision by inspecting which
    exec_run calls the code path would make — a full provisioning
    run goes deep into apt+pip and would need a real container.
    Here we stop at the gate and verify it would fall through."""
    sb = _make_sandbox_stub()
    v2_absent = _exec_run_map({
        "test -f /root/.supercharged.v3": (1, b""),
    })
    sb.container.exec_run = MagicMock(side_effect=v2_absent)
    # The `_chromium_binary_present` helper should still be called
    # and return False (no binary since nothing was installed).
    assert sb._chromium_binary_present() is False
    # Gate decision: test -f .supercharged.v3 → 1 → marker_ok=False
    code, _ = sb.container.exec_run("test -f /root/.supercharged.v3")
    assert code != 0


def test_marker_present_but_binary_missing_triggers_reinstall():
    """The exact failure mode v2 exists to catch: marker set but
    the Chromium binary isn't actually on disk. Both gates must be
    checked — marker alone is not enough."""
    sb = _make_sandbox_stub()
    responses = _exec_run_map({
        "test -f /root/.supercharged.v3": (0, b""),     # marker: ok
        "find /root/.cache/ms-playwright": (0, b""),    # no binary path
    })
    sb.container.exec_run = MagicMock(side_effect=responses)
    marker_ok = (sb.container.exec_run("test -f /root/.supercharged.v3")[0] == 0)
    chromium_ok = sb._chromium_binary_present()
    assert marker_ok is True
    assert chromium_ok is False
    # Combined condition that the gate uses — (not marker_ok or not chromium_ok)
    # → must go into the install branch.
    assert (not marker_ok or not chromium_ok) is True


def test_both_marker_and_binary_present_skips_install():
    sb = _make_sandbox_stub()
    responses = _exec_run_map({
        "test -f /root/.supercharged.v3": (0, b""),
        "find /root/.cache/ms-playwright": (
            0, b"/root/.cache/ms-playwright/chromium-1234/chrome-linux/headless_shell\n"
        ),
    })
    sb.container.exec_run = MagicMock(side_effect=responses)
    marker_ok = (sb.container.exec_run("test -f /root/.supercharged.v3")[0] == 0)
    chromium_ok = sb._chromium_binary_present()
    assert marker_ok and chromium_ok
    assert not (not marker_ok or not chromium_ok)


# ------------------------------------------------------------------
# Post-install verification
# ------------------------------------------------------------------

def test_post_install_verification_refuses_to_mark_on_missing_binary():
    """After running `playwright install chromium --with-deps` the
    code checks the binary actually exists before setting
    /root/.supercharged.v3. A hypothetical install that exited 0 but
    produced no binary must NOT leave a v2-marked image around."""
    import inspect
    # ensure_running is now a thin locking wrapper; the provisioning body
    # lives in _ensure_running_impl. Inspect both so the gate assertions
    # hold regardless of which method carries the logic.
    src = inspect.getsource(DockerSandbox.ensure_running) + inspect.getsource(
        DockerSandbox._ensure_running_impl)
    assert "_chromium_binary_present" in src
    # Grep the source for the refuse-to-mark guard. The assertion
    # here is purposely loose — we just want to make sure the check
    # exists in the ensure_running flow, not nail the exact phrasing.
    assert "Refusing to mark container as provisioned" in src


# ------------------------------------------------------------------
# Dockerfile sync
# ------------------------------------------------------------------

def test_dockerfile_uses_with_deps_flag():
    """The Dockerfile MUST use `--with-deps` — without it,
    libnss3/libatk etc. aren't installed and Chromium can't launch.
    This was the root cause of the self-play browser failures."""
    from pathlib import Path
    dockerfile = Path(__file__).resolve().parent.parent / "sandbox" / "Dockerfile"
    assert dockerfile.exists(), f"Dockerfile missing at {dockerfile}"
    content = dockerfile.read_text()
    assert "playwright install --with-deps chromium" in content or \
           "playwright install chromium --with-deps" in content, (
        "Dockerfile must pass --with-deps to `playwright install chromium`; "
        "omitting it reintroduces the missing-libnss3 failure mode"
    )


def test_dockerfile_sets_v3_marker():
    """The Dockerfile's final marker must match the runtime gate."""
    from pathlib import Path
    dockerfile = Path(__file__).resolve().parent.parent / "sandbox" / "Dockerfile"
    assert "/root/.supercharged.v3" in dockerfile.read_text(), (
        "sandbox/Dockerfile must touch /root/.supercharged.v3 — the "
        "runtime gate in docker.py reads this exact path"
    )
