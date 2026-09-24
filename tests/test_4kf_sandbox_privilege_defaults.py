"""§4KF (2026-09-24): the sandbox's privilege DEFAULTS.

Measured on the live container first: the exec user is root, Docker's
default 14 capabilities were in force, `/etc/sudoers` carried a NOPASSWD ALL
grant twice, and `host` networking was the Linux default from before the
in-container Tor (§4FU) made bridge the only mode the egress rules work in.
Then measured on a throwaway container from the live image: everything the
sandbox does as root works with `cap_drop=ALL` plus seven kept capabilities
and `no-new-privileges`, while raw sockets, mknod, chroot and plain-root
iptables are refused. These pins hold the create-time kwargs to that.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _docker_env(mock_client, mock_container):
    class MockNotFound(Exception):
        pass
    mock_client.containers.get.side_effect = MockNotFound()
    mock_client.containers.run.return_value = mock_container
    mock_container.status = "running"
    mock_container.exec_run.return_value = (0, b"ok")
    return patch.dict("sys.modules", {
        "docker": MagicMock(from_env=MagicMock(return_value=mock_client)),
        "docker.errors": MagicMock(NotFound=MockNotFound),
    })


def _create_kwargs(tmp_path, monkeypatch, *, env=None, platform=None, network=None):
    """The kwargs `ensure_running` hands to `containers.run` for a fresh
    container, under a mocked docker client."""
    from ghost_agent.sandbox import docker as dmod
    for k in ("GHOST_SANDBOX_DROP_CAPS", "GHOST_SANDBOX_NETWORK"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    if platform:
        monkeypatch.setattr(sys, "platform", platform)
    mock_client, mock_container = MagicMock(), MagicMock()
    with _docker_env(mock_client, mock_container):
        sb = dmod.DockerSandbox(host_workspace=tmp_path, network=network)
        sb._is_container_ready = MagicMock(return_value=True)
        sb._verify_environment = MagicMock()
        sb.ensure_running()
    return mock_client.containers.run.call_args.kwargs, sb


def test_by_default_every_capability_is_dropped_but_the_seven_root_needs(tmp_path, monkeypatch):
    """The old default was Docker's full set because "sudo needs setuid" —
    stale: exec is root. Kept caps are exactly the ones the image's jobs
    need to DROP privilege, own package files and signal children."""
    from ghost_agent.sandbox.docker import SANDBOX_KEPT_CAPS
    kw, _ = _create_kwargs(tmp_path, monkeypatch)
    assert kw["cap_drop"] == ["ALL"]
    assert kw["cap_add"] == list(SANDBOX_KEPT_CAPS)
    assert set(SANDBOX_KEPT_CAPS) == {"CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID", "SETGID", "SETUID", "KILL"}
    for gone in ("NET_RAW", "NET_ADMIN", "SYS_ADMIN", "MKNOD", "SYS_CHROOT", "SETPCAP", "SETFCAP", "AUDIT_WRITE",
                 "NET_BIND_SERVICE", "SYS_PTRACE", "DAC_READ_SEARCH"):
        assert gone not in kw["cap_add"], gone
    assert kw["security_opt"] == ["no-new-privileges"]


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF "])
def test_the_operator_can_opt_out_of_the_drop(tmp_path, monkeypatch, value):
    kw, _ = _create_kwargs(tmp_path, monkeypatch, env={"GHOST_SANDBOX_DROP_CAPS": value})
    assert "cap_drop" not in kw and "cap_add" not in kw and "security_opt" not in kw


@pytest.mark.parametrize("value", ["1", "true", "yes", ""])
def test_the_old_opt_in_spelling_and_the_empty_string_keep_the_default(tmp_path, monkeypatch, value):
    """`GHOST_SANDBOX_DROP_CAPS=1` was the pre-§4KF opt-in; it must not
    become an opt-OUT by accident, and an empty string is not a choice."""
    kw, _ = _create_kwargs(tmp_path, monkeypatch, env={"GHOST_SANDBOX_DROP_CAPS": value})
    assert kw["cap_drop"] == ["ALL"] and kw["security_opt"] == ["no-new-privileges"]


def test_bridge_is_the_default_on_linux_too(tmp_path, monkeypatch):
    """The Linux `host` default predates §4FU: under host networking the
    in-container Tor cannot bind and the egress rules do not apply."""
    kw, sb = _create_kwargs(tmp_path, monkeypatch, platform="linux")
    assert kw["network_mode"] == "bridge"
    assert kw["extra_hosts"] == {"host.docker.internal": "host-gateway"}
    assert sb.binds_host_netns() is False, "services must bind loopback, not 0.0.0.0"


def test_bridge_stays_the_default_on_mac_without_the_extra_host(tmp_path, monkeypatch):
    kw, sb = _create_kwargs(tmp_path, monkeypatch, platform="darwin")
    assert kw["network_mode"] == "bridge" and "extra_hosts" not in kw
    assert sb.binds_host_netns() is False


def test_host_networking_is_still_an_explicit_choice(tmp_path, monkeypatch):
    kw, sb = _create_kwargs(tmp_path, monkeypatch, platform="linux", env={"GHOST_SANDBOX_NETWORK": "host"})
    assert kw["network_mode"] == "host" and sb.binds_host_netns() is True
    kw2, sb2 = _create_kwargs(tmp_path, monkeypatch, platform="linux", network="none")
    assert kw2["network_mode"] == "none" and sb2.binds_host_netns() is False


def test_provisioning_never_writes_a_nopasswd_grant(tmp_path, monkeypatch):
    """Drive the real provisioning branch (marker missing) and read every
    exec the sandbox issued: none may touch /etc/sudoers."""
    from ghost_agent.sandbox import docker as dmod
    mock_client, mock_container = MagicMock(), MagicMock()

    def exec_side_effect(cmd, **kwargs):
        c = str(cmd)
        if "test -f /root/.supercharged" in c:
            return (1, b"")
        if "find /root/.cache/ms-playwright" in c:
            return (0, b"/root/.cache/ms-playwright/chromium-1/chrome-linux/headless_shell\n")
        return (0, b"")
    with _docker_env(mock_client, mock_container):
        mock_container.exec_run.side_effect = exec_side_effect
        sb = dmod.DockerSandbox(host_workspace=tmp_path)
        with patch.object(sb, "_is_container_ready", return_value=True):
            sb.container = mock_container
            sb.ensure_running()
    cmds = [str(c[0][0]) for c in mock_container.exec_run.call_args_list]
    assert any("apt-get install -y sudo" in c for c in cmds), "the sudo PACKAGE still ships (model scripts say sudo apt-get)"
    # The one exec allowed to mention sudoers is the in-place STRIP of the
    # grant (a `sed -i '/…/d'`); nothing may append to or echo into it.
    writes = [c for c in cmds if "sudoers" in c and ("sed -i" not in c or ">>" in c or "echo \"ALL" in c or "echo 'ALL" in c)]
    assert not writes, writes
    assert not any("ALL ALL=(ALL) NOPASSWD: ALL" in c and ">>" in c for c in cmds)


def test_the_image_recipe_carries_no_nopasswd_grant():
    """The Dockerfile is the other place the grant lived (both appended it,
    so the live container had the line twice)."""
    text = (_ROOT / "sandbox" / "Dockerfile").read_text(encoding="utf-8")
    active = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert not any("NOPASSWD" in ln for ln in active), "a NOPASSWD grant is back in the image recipe"
    assert any("sudo" in ln for ln in active), "the sudo package itself must still be installed"


def _adopted(tmp_path, monkeypatch, host_config, *, env=None):
    """A sandbox that ADOPTED an existing container with the given HostConfig
    (the post-deploy case: the container outlives the agent process)."""
    from ghost_agent.sandbox import docker as dmod
    for k in ("GHOST_SANDBOX_DROP_CAPS", "GHOST_SANDBOX_NETWORK"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    mock_client, mock_container = MagicMock(), MagicMock()
    mock_container.attrs = {"HostConfig": host_config, "NetworkSettings": {"Networks": {"bridge": {}}}}
    mock_container.exec_run.return_value = (0, b"clean")
    with _docker_env(mock_client, mock_container):
        sb = dmod.DockerSandbox(host_workspace=tmp_path)
    sb.container = mock_container
    sb.tor_proxy = None
    return sb, mock_container


def test_an_adopted_container_with_the_old_flags_is_reported_once_with_the_remedy(tmp_path, monkeypatch):
    """Flags apply at creation, so the live container that outlived the
    deploy kept Docker's default caps silently. Now the agent says so, ONCE
    per generation, and names the remedy — it never recreates on its own."""
    from ghost_agent.sandbox import docker as dmod
    sb, c = _adopted(tmp_path, monkeypatch, {"CapDrop": [], "SecurityOpt": [], "NetworkMode": "bridge"})
    with patch.object(dmod, "pretty_log") as plog:
        sb._enforce_egress_once()
        sb._enforce_egress_once()
    warns = [c_ for c_ in plog.call_args_list if c_.kwargs.get("level") == "WARNING"]
    assert len(warns) == 1, "one warning per generation"
    text = " ".join(str(a) for a in warns[0].args)
    assert "cap_drop none (intended ['ALL'])" in text and "no-new-privileges" in text
    assert "docker rm -f" in text and sb.container_name in text
    assert "cap_drop" in sb._privilege_drift
    assert not c.remove.called and not c.stop.called, "reporting, not recreating"
    # a container created with today's defaults is silent
    sb2, _ = _adopted(tmp_path, monkeypatch, {"CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"], "NetworkMode": "bridge"})
    with patch.object(dmod, "pretty_log") as plog2:
        sb2._enforce_egress_once()
    assert not [c_ for c_ in plog2.call_args_list if c_.kwargs.get("level") == "WARNING"]
    assert sb2._privilege_drift == ""


def test_the_cached_images_sudoers_grant_is_stripped_in_place_once_per_generation(tmp_path, monkeypatch):
    """The recreate boots `ghost-agent-base:latest`, a runtime commit that
    still carries `NOPASSWD: ALL` twice, and provisioning is skipped on the
    marker — so the recipe change alone never reaches a recreated container
    (review MAJOR-1). The grant is removed in place, as root, once."""
    sb, c = _adopted(tmp_path, monkeypatch, {"CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"], "NetworkMode": "bridge"})
    c.exec_run.return_value = (0, b"stripped")
    sb._enforce_egress_once()
    sb._enforce_egress_once()
    strips = [k for k in c.exec_run.call_args_list if "NOPASSWD" in str(k[0][0])]
    assert len(strips) == 1, strips
    cmd = str(strips[0][0][0])
    assert "sed -i" in cmd and "/etc/sudoers" in cmd and strips[0].kwargs.get("user") == "root"
    # a new generation (resume/recreate resets the flag) strips again
    sb._privilege_checked = False
    sb._enforce_egress_once()
    assert len([k for k in c.exec_run.call_args_list if "NOPASSWD" in str(k[0][0])]) == 2


def test_binds_host_netns_reads_the_adopted_container_not_the_default(tmp_path, monkeypatch):
    """A Linux agent upgraded to §4KF adopts its old `host`-mode container;
    services must still bind loopback there, whatever today's default is
    (review MAJOR-3). An unknown/stubbed mode falls back to the decision."""
    sb, c = _adopted(tmp_path, monkeypatch, {"NetworkMode": "host"})
    assert sb.binds_host_netns() is True
    sb_b, _ = _adopted(tmp_path, monkeypatch, {"NetworkMode": "bridge"}, env={"GHOST_SANDBOX_NETWORK": "host"})
    assert sb_b.binds_host_netns() is False, "the container is the truth, not the env"
    sb_u, cu = _adopted(tmp_path, monkeypatch, {}, env={"GHOST_SANDBOX_NETWORK": "host"})
    cu.attrs = {"HostConfig": {}}
    assert sb_u.binds_host_netns() is True, "no mode on the container → the configured decision"


def test_the_egress_rules_do_not_rely_on_the_container_capability_set():
    """The rules are applied through a PRIVILEGED exec, which is what makes
    `cap_drop=ALL` safe for enforcement — and what keeps NET_ADMIN out of
    the model's reach. PARSED pin: the exec that applies the rules passes
    privileged=True."""
    import ast, inspect
    from ghost_agent.sandbox import docker as dmod
    tree = ast.parse(inspect.getsource(dmod))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_enforce_tor_egress")
    privileged_calls = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "_exec_run"
        and any(k.arg == "privileged" and getattr(k.value, "value", None) is True for k in n.keywords)
    ]
    applies_rules = [
        n for n in privileged_calls
        if any(isinstance(a, ast.Call) and getattr(a.func, "attr", "") == "apply_rules_cmd" for a in n.args)
    ]
    assert applies_rules, "the exec that APPLIES the rules is not the privileged one — under cap_drop=ALL it would fail"
    from ghost_agent.sandbox.docker import SANDBOX_KEPT_CAPS
    assert "NET_ADMIN" not in SANDBOX_KEPT_CAPS
