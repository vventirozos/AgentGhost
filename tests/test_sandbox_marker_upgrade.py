"""Sandbox marker ladder: the delta tools are installed, and the previous image
upgrades IN PLACE (§4FR/§4FS, 2026-09-09).

Request d50a34bd: the model verified a 16 MB download the standard way,

    curl -L -o x.pdf … && ls -la x.pdf && file x.pdf && du -h x.pdf

and `file` was not in the sandbox image. The chain exited 127, the tool
result carried a FAILURE BANNER, the turn took Strike 2/6, and the
trajectory recorded a failed step — for a download that had succeeded.
The sandbox even carried a canned hint for this ("that utility isn't
installed"): the gap was known and worked around rather than fixed.

v6 added `file`. Its live check found the same shape one tool over — the
model's chain is `file x.pdf && head -c 200 x.pdf | xxd | head -5`, and the
missing `xxd` scored the download a failure again — so v7 added `xxd`; v8 added
`lsof` and `dnsutils`, the only other commands 415 trajectory files show the
model reaching for; v9 added `iptables` for the Tor-only egress rules (§4FU).
The pins below read the ladder from the source; ONE pin states the current
delta from outside it.

Two things are pinned here:

  * both tools are in BOTH install surfaces (the runtime provisioner and
    the build-time Dockerfile), under one marker version, v7;
  * the previous image (v6) is lifted to v7 by installing ONLY the delta — a full
    provision is ~5 minutes of apt + pip + torch + Chromium on the
    operator's next request, the price every earlier bump paid — and the
    new marker is written only after the binary is PROBED (the v2 lesson:
    a marker must never assert what nobody checked).
"""
import inspect
import os
import re
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ghost_agent.sandbox.docker import DockerSandbox

# Derived from the SOURCE, not hard-coded: the ladder (current marker, the
# marker it supersedes, the delta and its probe) is whatever docker.py says
# today, and these pins check the SHAPE of an upgrade, not a version number.
_SRC = inspect.getsource(DockerSandbox._ensure_running_impl)
_CUR_MARKER = re.search(r'(?<!prev_)marker_path = "([^"]+)"', _SRC).group(1)
_PREV_MARKER = re.search(r'prev_marker_path = "([^"]+)"', _SRC).group(1)
_DELTA = re.search(r"upgrade_delta_cmd = \"(.+?)\"\n", _SRC).group(1)
_DELTA_PKGS = _DELTA.split("apt-get install -y ", 1)[1].rstrip("'")
_VERIFY = re.search(r"upgrade_delta_verify = \"(.+?)\"\n", _SRC).group(1)
_VERIFY_TOOL = re.search(r"command -v (\w+)", _VERIFY).group(1)
CUR = f"test -f {_CUR_MARKER}"
PREV = f"test -f {_PREV_MARKER}"
DELTA_KEY = f"apt-get install -y {_DELTA_PKGS}'"
VERIFY_KEY = f"command -v {_VERIFY_TOOL}"
TOUCH_CUR = f"touch {_CUR_MARKER}"
RM_PREV = f"rm -f {_PREV_MARKER}"
CHROMIUM = "find /root/.cache/ms-playwright"
CHROMIUM_PRESENT = (0, b"/root/.cache/ms-playwright/chromium-1/x/headless_shell\n")


def _stub(workspace):
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.host_workspace = Path(workspace)
    sb.image = "ghost-agent-base:latest"
    sb.container_name = "ghost-test-v6"
    sb.tor_proxy = None
    sb.client = MagicMock()
    sb.docker_lib = MagicMock()
    _NF = type("ImageNotFound", (Exception,), {})
    sb.docker_lib.errors.ImageNotFound = _NF
    sb.ImageNotFound = _NF
    sb.NotFound = type("NotFound", (Exception,), {})
    sb.APIError = type("APIError", (Exception,), {})
    sb._lock = threading.Lock()
    sb.container = MagicMock()
    sb._is_container_ready = MagicMock(return_value=True)
    return sb


def _run(sb, overrides):
    """Drive ensure_running with an exec stub; returns every command seen.
    Unknown commands succeed; the Chromium probe reports a binary."""
    seen = []

    def _exec(cmd, *a, **k):
        seen.append(cmd)
        for key, val in overrides.items():
            if key in cmd:
                return val(cmd) if callable(val) else val
        if CHROMIUM in cmd:
            return CHROMIUM_PRESENT
        return (0, b"")

    sb.container.exec_run.side_effect = _exec
    with patch("ghost_agent.sandbox.docker.pretty_log"):
        sb.ensure_running()
    return seen


def _installs(seen):
    return [c for c in seen if "pip install" in c or "playwright install" in c]


# --- the in-place upgrade -------------------------------------------------

def test_the_previous_image_is_upgraded_in_place_with_only_the_delta(tmp_path):
    """THE POINT. World where it fails: the marker missing means "re-provision from
    scratch" and the operator's next request waits five minutes."""
    sb = _stub(tmp_path)
    seen = _run(sb, {CUR: (1, b""), PREV: (0, b"")})

    delta = [c for c in seen if DELTA_KEY in c]
    assert len(delta) == 1, seen
    assert "timeout 600" in delta[0], "the delta must carry its own in-container cap"
    assert _installs(seen) == [], f"a full provision ran: {_installs(seen)}"
    assert any(VERIFY_KEY in c for c in seen), "the binary was never probed"
    assert TOUCH_CUR in seen
    assert RM_PREV in seen
    sb.container.commit.assert_called_once_with(repository="ghost-agent-base", tag="latest")


def test_the_marker_is_written_only_after_the_binary_is_probed(tmp_path):
    """apt exits 0 but `command -v file` finds nothing (a mirror served a
    stub, a partial extract). The v2 lesson: never stamp what nobody
    checked. World where it fails: the marker is touched on apt's exit
    code alone, and every future boot trusts a v6 image without `file`."""
    sb = _stub(tmp_path)
    seen = _run(sb, {CUR: (1, b""), PREV: (0, b""), VERIFY_KEY: (1, b"")})

    delta_idx = next(i for i, c in enumerate(seen) if DELTA_KEY in c)
    touch_idx = [i for i, c in enumerate(seen) if c == TOUCH_CUR]
    # the only marker write comes from the FULL provision that followed
    assert len(touch_idx) == 1 and touch_idx[0] > delta_idx, (delta_idx, touch_idx)
    assert _installs(seen), "no full provision ran after the failed upgrade"
    assert RM_PREV in seen[touch_idx[0]:], \
        "the old marker outlived the successful full provision"


def test_a_failed_delta_falls_back_to_a_full_provision(tmp_path):
    sb = _stub(tmp_path)
    seen = _run(sb, {CUR: (1, b""), PREV: (0, b""),
                     DELTA_KEY: (1, b"mirror down")})
    assert _installs(seen), "no full provision after the delta failed"
    assert TOUCH_CUR in seen      # from the full provision


def test_no_in_place_upgrade_when_chromium_is_missing(tmp_path):
    """The delta lifts a COMPLETE v5 image. One that also lost its browser
    needs the full flow, which installs Chromium."""
    sb = _stub(tmp_path)
    state = {"installed": False}

    def _find(cmd):
        return CHROMIUM_PRESENT if state["installed"] else (0, b"")

    def _pw(cmd):
        state["installed"] = True
        return (0, b"")

    seen = _run(sb, {CUR: (1, b""), PREV: (0, b""), CHROMIUM: _find, "playwright install": _pw})
    assert not any(DELTA_KEY in c for c in seen), "the delta ran on a broken image"
    assert any("playwright install" in c for c in seen)


def test_a_fresh_image_gets_file_from_the_full_provision(tmp_path):
    """No marker at all → full provision, whose apt list now carries `file`."""
    sb = _stub(tmp_path)
    seen = _run(sb, {"test -f /root/.supercharged": (1, b"")})
    full_apt = [c for c in seen if "apt-get install -y sudo" in c]
    assert len(full_apt) == 1, seen
    for pkg in _DELTA_PKGS.split():
        assert f" {pkg}" in full_apt[0], (pkg, full_apt[0])
    for pkg in ("file", "xxd", "lsof", "dnsutils"):  # earlier deltas stay in the full list
        assert f" {pkg} " in full_apt[0] or f" {pkg}'" in full_apt[0], (pkg, full_apt[0])
    assert not any(DELTA_KEY in c for c in seen), "the delta ran with no previous marker"


def test_a_current_image_does_nothing(tmp_path):
    sb = _stub(tmp_path)
    seen = _run(sb, {CUR: (0, b"")})
    assert not any("apt-get" in c for c in seen)
    assert not any("touch /root/.supercharged" in c for c in seen)
    sb.container.commit.assert_not_called()


# --- the two install surfaces agree ------------------------------------

def test_runtime_and_dockerfile_agree_on_the_ladder_and_the_packages():
    """A package in one surface but not the other means a from-scratch
    build differs from a runtime provision (the existing lockstep pin,
    extended to v6 and `file`)."""
    runtime = inspect.getsource(DockerSandbox._ensure_running_impl)
    dockerfile = (Path(__file__).resolve().parents[1] / "sandbox" / "Dockerfile").read_text()
    assert _CUR_MARKER in runtime and _CUR_MARKER in dockerfile
    cur_n = int(_CUR_MARKER.rsplit(".v", 1)[1]); prev_n = int(_PREV_MARKER.rsplit(".v", 1)[1])
    assert cur_n == prev_n + 1, "the in-place upgrade must target the version it supersedes"
    for pkg in _DELTA_PKGS.split():
        assert f" {pkg}" in runtime.split("apt-get install -y sudo", 1)[1][:400], f"{pkg} missing from the runtime apt list"
        assert pkg in dockerfile, f"{pkg} missing from the Dockerfile apt list"
    # anchored to the apt LINES: "file" also occurs in the word "Dockerfile"
    # and in the changelog comment inside the provisioner (review, 2026-09-09)
    df_apt = "\n".join(l for l in dockerfile.split("\n") if "apt-get install" in l or l.strip().startswith(("sudo", "postgresql-client")))
    rt_apt = runtime.split("apt-get install -y sudo", 1)[1][:400]
    for pkg in ("file", "xxd", "lsof", "dnsutils", "iptables"):
        assert f" {pkg} " in df_apt or f" {pkg} \\" in df_apt, f"{pkg} missing from the Dockerfile apt list"
        assert f" {pkg} " in rt_apt or f" {pkg}'" in rt_apt, f"{pkg} missing from the runtime apt list"
    assert "apt-get update &&" in _DELTA, "the delta must refresh the index before installing"


def test_the_fallback_hint_no_longer_routes_around_file():
    """The canned "that utility isn't installed" hint taught an `od` stand-in
    for `file`. With `file` installed the hint would send the model AWAY
    from a tool it has."""
    from ghost_agent.tools import tool_failure as TF
    hints = dict(TF._FALLBACK_HINTS["execute"])
    text = hints["command not found"]
    assert "`file <f>` →" not in text, text
    assert "`xxd` →" not in text, text
    assert "command -v" in text, "the availability probe advice must stay"


def test_this_versions_delta_is_what_the_evidence_asked_for():
    """The rest of this file reads the ladder from the source so a bump does
    not churn it — which also means a mutant that changes the delta is
    FOLLOWED by those pins (the first battery run proved it). This is the
    one line that states the expectation from outside: v9 exists to add
    iptables for the Tor-only egress rules. Update it with the next bump,
    deliberately."""
    assert _CUR_MARKER.endswith(".v9")
    assert set(_DELTA_PKGS.split()) == {"iptables"}, _DELTA_PKGS      # §4FU: the Tor-only egress rules
    assert _VERIFY_TOOL == "iptables", _VERIFY


# --- the delta obeys the provisioning backoff (review, 2026-09-09) ---------

def test_a_failed_delta_arms_the_backoff_and_the_next_command_does_not_reinstall(tmp_path):
    """A broken mirror must not be hit with a 600 s apt on EVERY command while
    the provision lock is held. World where it fails: the in-place path sits
    above the backoff gate and ignores it."""
    sb = _stub(tmp_path)
    # first call: the delta fails, then the full provision's apt fails → raises, backoff armed
    calls = []
    def _exec(cmd, *a, **k):
        calls.append(cmd)
        if CUR in cmd: return (1, b"")
        if PREV in cmd: return (0, b"")
        if DELTA_KEY in cmd or "apt-get install -y sudo" in cmd: return (1, b"mirror down")
        if CHROMIUM in cmd: return CHROMIUM_PRESENT
        return (0, b"")
    sb.container.exec_run.side_effect = _exec
    with patch("ghost_agent.sandbox.docker.pretty_log"):
        with pytest.raises(Exception):
            sb.ensure_running()
    assert sb._provision_backoff_until > 0, "no backoff armed after the failure"
    n_delta_first = sum(1 for c in calls if DELTA_KEY in c)
    assert n_delta_first == 1
    # second call, inside the backoff: no apt at all, a clear error
    calls.clear()
    with patch("ghost_agent.sandbox.docker.pretty_log"):
        with pytest.raises(Exception, match="retrying in"):
            sb.ensure_running()
    assert not any("apt-get" in c for c in calls), calls


def test_inside_a_backoff_the_delta_is_skipped_not_attempted(tmp_path):
    """Armed by an earlier failure, the backoff must stop the delta too."""
    import time
    sb = _stub(tmp_path)
    sb._provision_backoff_until = time.time() + 300.0
    calls = []
    def _exec(cmd, *a, **k):
        calls.append(cmd)
        if CUR in cmd: return (1, b"")
        if PREV in cmd: return (0, b"")
        if CHROMIUM in cmd: return CHROMIUM_PRESENT
        return (0, b"")
    sb.container.exec_run.side_effect = _exec
    with patch("ghost_agent.sandbox.docker.pretty_log"):
        with pytest.raises(Exception, match="retrying in"):
            sb.ensure_running()
    assert not any(DELTA_KEY in c for c in calls), "the delta ran inside the backoff"
