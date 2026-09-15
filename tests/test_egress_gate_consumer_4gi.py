"""§4GI: the tool-side consumer of the sandbox's egress state.

`_egress_state` had no reader: when Tor-only enforcement could not be
established the sandbox kept serving `execute`/`browser` with DIRECT egress.
The enforcement side now cuts the container off by construction; this is
the belt for the two cases it cannot cover (host networking, a failed
cut-off). ONE predicate, two callers. Pre-fix world: both tools ran.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.sandbox.egress_gate import EGRESS_UNAVAILABLE_MSG, network_refusal
from ghost_agent.tools import browser as browser_mod
from ghost_agent.tools import execute as execute_mod
from ghost_agent.tools.outcome import OutcomeStatus


def _mgr(tor, enforced, probe_raises=False):
    def _probe():
        if probe_raises:
            raise RuntimeError("no container")
        return enforced

    def _never(*a, **k):
        raise AssertionError("the sandbox must not be asked to run anything")
    return SimpleNamespace(tor_proxy=tor, egress_is_enforced_or_blocked=_probe,
                           execute=_never, container=None)


@pytest.mark.parametrize("tor,enforced,raises,refused", [
    ("socks5://127.0.0.1:9050", True, False, False),
    ("socks5://127.0.0.1:9050", False, False, True),
    ("socks5://127.0.0.1:9050", False, True, True),      # a broken probe is unavailable
    (None, False, False, False),                          # no Tor configured
])
def test_network_refusal_table(tor, enforced, raises, refused):
    out = network_refusal(_mgr(tor, enforced, raises))
    if refused:
        assert out is not None and out.status is OutcomeStatus.FAILED
        assert out.reason_code == "egress_unavailable" and EGRESS_UNAVAILABLE_MSG in str(out)
    else:
        assert out is None
    assert network_refusal(None) is None
    assert network_refusal(SimpleNamespace(tor_proxy="socks5://x")) is None   # no predicate: a fake


async def test_execute_refuses_before_running_anything(tmp_path):
    out = await execute_mod.tool_execute(command="curl https://example.org",
                                         sandbox_dir=tmp_path,
                                         sandbox_manager=_mgr("socks5://127.0.0.1:9050", False))
    assert out.status is OutcomeStatus.FAILED and out.reason_code == "egress_unavailable"


async def test_browser_refuses_before_running_anything(tmp_path):
    out = await browser_mod.tool_browser(operation="navigate", url="https://example.org",
                                         sandbox_dir=tmp_path,
                                         sandbox_manager=_mgr("socks5://127.0.0.1:9050", False),
                                         tor_proxy="socks5://127.0.0.1:9050")
    assert out.status is OutcomeStatus.FAILED and out.reason_code == "egress_unavailable"


# ── §4GJ: the refusal names the cause, because the remedies differ ──────────

def test_the_refusal_text_is_per_cause():
    """A remedy the operator cannot apply is worse than none: host
    networking is a CONFIGURATION choice and recreating the sandbox — the
    other branch's advice — changes nothing. Pre-§4GJ both said the same."""
    from ghost_agent.sandbox.egress_gate import (EGRESS_UNAVAILABLE_MSG,
                                                 _refusal_text)
    host = SimpleNamespace(_egress_unavailable_reason="host_networking")
    assert "HOST networking" in _refusal_text(host)
    assert "GHOST_SANDBOX_NETWORK=bridge" in _refusal_text(host)
    # every other cause keeps the generic text, whose remedy ("recreate the
    # sandbox") is the right one for a cut-off that failed
    assert _refusal_text(SimpleNamespace(_egress_unavailable_reason="cut_off_failed")) == EGRESS_UNAVAILABLE_MSG
    assert _refusal_text(SimpleNamespace()) == EGRESS_UNAVAILABLE_MSG   # unknown → generic


def test_the_refusal_carries_the_cause_through_the_tool_outcome():
    from ghost_agent.sandbox.egress_gate import network_refusal
    mgr = _mgr("socks5://127.0.0.1:9050", False)
    mgr._egress_unavailable_reason = "host_networking"
    out = network_refusal(mgr)
    assert out is not None and "GHOST_SANDBOX_NETWORK=bridge" in str(out)


def test_host_mode_reports_the_cause_at_boot_at_critical(monkeypatch):
    """The operator learns it ONCE, at boot, at the level they read — not
    per refused call (pre-§4GJ: a WARNING that named no applicable remedy)."""
    from unittest.mock import MagicMock, patch
    from ghost_agent.sandbox.docker import DockerSandbox
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.tor_proxy = "socks5://127.0.0.1:9050"
    sb._egress_state = ""
    sb._tor_attempted = False
    sb._container_network_mode = lambda: "host"
    with patch("ghost_agent.sandbox.docker.pretty_log") as plog:
        sb._enforce_egress_once()
    assert sb._egress_state == "unavailable"
    assert sb._egress_unavailable_reason == "host_networking"
    levels = [c.kwargs.get("level") for c in plog.call_args_list]
    said = " ".join(str(c.args) for c in plog.call_args_list)
    assert "CRITICAL" in levels, levels
    assert "GHOST_SANDBOX_NETWORK=bridge" in said


# ── §4GJ round 3: the reason travels WITH the state ─────────────────────────

def test_a_later_unavailable_does_not_inherit_the_host_networking_remedy():
    """Round 3: `_egress_unavailable_reason` was written at the host-mode
    branch and never cleared, so a LATER disconnect-failure served the
    host-networking remedy ("set GHOST_SANDBOX_NETWORK=bridge") — advice
    that does nothing for a container whose disconnect failed. Fails in any
    tree where the reason is not part of the transition."""
    from ghost_agent.sandbox.docker import DockerSandbox
    from ghost_agent.sandbox.egress_gate import EGRESS_UNAVAILABLE_MSG, _refusal_text
    sb = DockerSandbox.__new__(DockerSandbox)
    sb._set_egress_state("unavailable", "host_networking")
    assert "GHOST_SANDBOX_NETWORK=bridge" in _refusal_text(sb)
    sb._set_egress_state("unavailable", "cut_off_failed")
    assert _refusal_text(sb) == EGRESS_UNAVAILABLE_MSG
    assert "GHOST_SANDBOX_NETWORK=bridge" not in _refusal_text(sb)


def test_leaving_unavailable_clears_the_reason_entirely():
    """A recovered sandbox must not keep a remedy on file: the reason is
    meaningless unless the state is `unavailable`."""
    from ghost_agent.sandbox.docker import DockerSandbox
    sb = DockerSandbox.__new__(DockerSandbox)
    sb._set_egress_state("unavailable", "host_networking")
    for state in ("blocked", "enforced", ""):
        sb._set_egress_state(state)
        assert sb._egress_unavailable_reason == "", state


def test_each_unavailable_BRANCH_records_its_own_cause(tmp_path):
    """The reason must be the one that BRANCH means, not merely some reason.
    A spot-check mutant that made the disconnect-failure branch record
    `host_networking` survived every other pin here — the mapping from
    branch to cause was unpinned, so the wrong remedy could ship again.
    Drives the real `_block_egress_hard` with a disconnect that fails."""
    from unittest.mock import MagicMock, patch
    from ghost_agent.sandbox.docker import DockerSandbox
    from ghost_agent.sandbox.egress_gate import EGRESS_UNAVAILABLE_MSG, _refusal_text
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.container = MagicMock()
    sb.container.attrs = {"NetworkSettings": {"Networks": {"bridge": {}}}}
    sb.client = MagicMock()
    sb.client.networks.get.return_value.disconnect.side_effect = RuntimeError("nope")
    with patch("ghost_agent.sandbox.docker.pretty_log"):
        sb._block_egress_hard("rules failed")
    assert sb._egress_state == "unavailable"
    assert sb._egress_unavailable_reason == "cut_off_failed"
    assert _refusal_text(sb) == EGRESS_UNAVAILABLE_MSG
    assert "GHOST_SANDBOX_NETWORK=bridge" not in _refusal_text(sb)


def test_the_recreate_makes_the_readiness_stamp_stale(tmp_path):
    """`_recreate_if_cut_off`'s postcondition, pinned directly: dropping the
    container is what `_ready_is_fresh` happens to check today, and the
    explicit `invalidate_ready()` is what keeps the ordering correct if that
    ever changes. Fails in a tree where the recreate leaves the stamp."""
    from unittest.mock import MagicMock, patch
    from ghost_agent.sandbox.docker import DockerSandbox
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.container = MagicMock()
    sb.container.attrs = {"NetworkSettings": {"Networks": {}},
                          "HostConfig": {"NetworkMode": "bridge"}}
    sb._cut_off_at = 0.0
    sb.mark_ready()
    assert sb._last_ready_ok != 0.0
    with patch("ghost_agent.sandbox.docker.pretty_log"):
        assert sb._recreate_if_cut_off() is True
    assert sb._last_ready_ok == 0.0
    assert sb.container is None
