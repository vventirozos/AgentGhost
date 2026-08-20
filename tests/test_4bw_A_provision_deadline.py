"""§4BW-A — provision deadline inversion.

The in-container install caps (`timeout 1800 …`) EXCEEDED the client-side wedge
deadline default (1200s) and the provision execs passed NO per-exec deadline,
so a healthy-but-slow install tripped the client deadline first → mis-diagnosed
as a wedged daemon → provision abort + 300s backoff, while the abandoned worker
kept installing and the retry double-installed.

These pins assert the thing EQUALS a recomputed value (§4BN): the client
deadline every install exec receives is GREATER THAN that install's own
in-container `timeout N` cap.
"""
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../src')))

import pytest
from unittest.mock import MagicMock

import ghost_agent.sandbox.docker as d
from ghost_agent.sandbox.docker import DockerSandbox


# The install commands and their in-container caps, recomputed here so the
# assertions never read the module's own answer.
INSTALL_CMDS = {
    "apt": ("timeout 900 sh -c 'apt-get update && apt-get install -y …'", 900),
    "pysocks": ("timeout 600 pip install --no-cache-dir pysocks requests", 600),
    "pip": ("timeout 1800 pip install --no-cache-dir numpy pandas …", 1800),
    "torch": ("timeout 1800 pip install --no-cache-dir torch "
              "--index-url https://download.pytorch.org/whl/cpu", 1800),
    "playwright": ("timeout 1800 python3 -m playwright install chromium "
                   "--with-deps", 1800),
}


class TestTheDerivationRemovesTheInversion:

    def test_every_install_deadline_strictly_exceeds_its_own_cap(self):
        for name, (cmd, cap) in INSTALL_CMDS.items():
            got = d._provision_deadline_s(cmd)
            assert got > cap, f"{name}: deadline {got} !> in-container cap {cap}"

    def test_the_deadline_equals_cap_plus_grace(self):
        # EQUALS a recomputed value, not merely "> cap".
        for name, (cmd, cap) in INSTALL_CMDS.items():
            assert d._provision_deadline_s(cmd) == cap + d._PROVISION_EXEC_GRACE_S

    def test_the_max_install_cap_no_longer_outlives_the_client_deadline(self):
        # The exact inversion proof-5 demonstrated: max cap 1800 > wedge 1200.
        max_cap = max(cap for _, cap in INSTALL_CMDS.values())
        assert max_cap > d._EXEC_DAEMON_DEADLINE_S  # the bug still exists at the DEFAULT…
        # …but the per-install deadline clears it.
        worst = max(d._provision_deadline_s(cmd) for cmd, _ in INSTALL_CMDS.values())
        assert worst > max_cap

    def test_a_command_without_a_timeout_prefix_uses_the_wedge_default(self):
        assert d._provision_deadline_s("sh -c 'echo hi >> /etc/sudoers'") \
            == d._EXEC_DAEMON_DEADLINE_S
        assert d._provision_deadline_s("test -f /root/.supercharged.v5") \
            == d._EXEC_DAEMON_DEADLINE_S

    def test_grace_can_never_produce_a_non_clearing_deadline(self, monkeypatch):
        # Even a pathological zero grace must still clear the cap.
        # ⚠ monkeypatch the module CONSTANT, never importlib.reload(d): a
        # reload of a production module rebinds its classes for the rest of
        # the session (here SandboxDaemonTimeout), so a later test's
        # `pytest.raises(SandboxDaemonTimeout)` holds a stale class and the
        # freshly-raised one propagates — it broke
        # test_docker_review_fixes::test_raises_on_daemon_wedge run after
        # this one. setattr restores cleanly and touches nothing else.
        monkeypatch.setattr(d, "_PROVISION_EXEC_GRACE_S", 0.0)
        assert d._provision_deadline_s("timeout 1800 pip install x") > 1800


class TestProvisionExecForwardsTheDeadline:

    def test_provision_exec_passes_the_derived_deadline_to_exec_run(self):
        sb = DockerSandbox.__new__(DockerSandbox)
        sb._exec_run = MagicMock(return_value=(0, b""))
        sb._provision_exec("timeout 1800 pip install torch")
        _, kw = sb._exec_run.call_args
        assert kw["deadline_s"] > 1800


class TestTheLiveProvisionCallSitesAllPassADeadline:
    """Drives the REAL provision block with mocks and captures the deadline
    every exec receives — a call site that reverts to bare `_exec_run`
    (deadline_s=None) is caught here, not by reading the source."""

    def _driven_calls(self, monkeypatch):
        calls = []

        def fake_exec_run(cmd, deadline_s=None, **kw):
            calls.append((cmd, deadline_s))
            if str(cmd).startswith("test -f"):
                return (1, b"")           # marker missing → enter provision
            return (0, b"Successfully installed")

        sb = DockerSandbox.__new__(DockerSandbox)
        sb.container = MagicMock()
        sb.container.commit = MagicMock()
        sb.tor_proxy = None
        sb._env_verified = False
        sb._tor_attempted = False
        sb._provision_backoff_until = 0.0
        sb._exec_run = fake_exec_run
        sb._ready_is_fresh = lambda: False
        sb._is_container_ready = lambda: True
        sb._chromium_binary_present = lambda: True
        sb.mark_ready = lambda: None
        sb._ensure_running_impl()
        return calls

    def test_every_timeout_capped_exec_got_a_deadline_above_its_cap(self, monkeypatch):
        calls = self._driven_calls(monkeypatch)
        capped = [(c, dl) for (c, dl) in calls
                  if re.match(r"^\s*timeout\s+(\d+)", str(c))]
        assert capped, "provision drove no timeout-capped installs"
        for cmd, deadline in capped:
            cap = int(re.match(r"^\s*timeout\s+(\d+)", str(cmd)).group(1))
            assert deadline is not None, f"install ran with NO client deadline: {cmd[:50]}"
            assert deadline > cap, f"client deadline {deadline} !> cap {cap}: {cmd[:50]}"

    def test_the_apt_and_pip_and_torch_and_playwright_installs_all_ran(self, monkeypatch):
        calls = self._driven_calls(monkeypatch)
        joined = "\n".join(str(c) for c, _ in calls)
        for needle in ("apt-get install", "pip install --no-cache-dir numpy",
                       "pip install --no-cache-dir torch",
                       "playwright install chromium"):
            assert needle in joined, f"provision did not run: {needle}"
