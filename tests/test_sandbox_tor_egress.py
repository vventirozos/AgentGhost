"""Tor-only egress for the sandbox (§4FU, 2026-09-09).

Measured from inside the live container: a plain request left from the
operator's real IP; the code's own comments believed an "internal Tor
daemon" enforced otherwise, and it had never run. The design (operator's
option 1, hybrid): an in-container Tor as the unprivileged user with a
TransPort and a DNSPort; iptables in the container's own network
namespace redirecting all TCP and DNS to them, exempting only Tor's own
traffic and loopback, rejecting everything else; loaded through a
PRIVILEGED exec the container cannot undo; provisioning stays direct;
fail-closed from the moment the rules go in.

Two halves: the CONFIGURATION (tor_egress.py — pinned as data) and the
SEQUENCING (docker.py `_enforce_tor_egress` — pinned through the same exec
stub the other sandbox tests use). The live spike that settled the design
is quoted in the module docstring; its leak probes are what the operator's
post-deploy check runs.
"""
import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from ghost_agent.sandbox import tor_egress as T
from ghost_agent.sandbox.docker import DockerSandbox


# --- the configuration, as data ------------------------------------------

def test_tor_runs_as_the_unprivileged_user_with_trans_and_dns_ports():
    assert f"User {T.TOR_USER}" in T.TORRC
    assert f"TransPort 127.0.0.1:{T.TRANS_PORT}" in T.TORRC
    assert f"DNSPort 127.0.0.1:{T.DNS_PORT}" in T.TORRC
    assert "AutomapHostsOnResolve 1" in T.TORRC and f"VirtualAddrNetworkIPv4 {T.VIRTUAL_NET}" in T.TORRC
    assert "SocksPort 127.0.0.1:" in T.TORRC, "the browser and any proxy-aware tool keep a SOCKS port"


def _nat_rules():
    return [l for l in T.RULES_SCRIPT.splitlines() if l.startswith(f"iptables -t nat -A {T.NAT_CHAIN}")]


def _filter_rules():
    return [l for l in T.RULES_SCRIPT.splitlines() if l.startswith(f"iptables -A {T.FILTER_CHAIN}")]


def test_tors_own_traffic_and_loopback_are_exempted_BEFORE_any_redirect():
    """The contract of the nat chain: exempt first, redirect after. World
    where it fails: Tor's guard connections are redirected into Tor."""
    rules = _nat_rules()
    exempt = [i for i, r in enumerate(rules) if "-j RETURN" in r]
    redirect = [i for i, r in enumerate(rules) if "-j REDIRECT" in r]
    assert exempt and redirect and max(exempt) < min(redirect), rules
    assert any(f"--uid-owner {T.TOR_USER}" in rules[i] for i in exempt)
    assert any("-o lo" in rules[i] for i in exempt)


def test_all_tcp_goes_to_the_transport_and_dns_to_the_dnsport():
    rules = _nat_rules()
    assert f"-p tcp -j REDIRECT --to-ports {T.TRANS_PORT}" in rules[-1], "the catch-all TCP redirect must be LAST"
    assert any(f"-p udp --dport 53 -j REDIRECT --to-ports {T.DNS_PORT}" in r for r in rules)
    assert any(f"-p tcp --dport 53 -j REDIRECT --to-ports {T.DNS_PORT}" in r for r in rules)


def test_everything_else_is_rejected_and_ipv6_is_off():
    """UDP other than DNS, ICMP, and every IPv6 packet: no path out."""
    f = _filter_rules()
    assert f[-1].endswith("-j REJECT --reject-with icmp-port-unreachable"), f
    assert any("--state ESTABLISHED,RELATED -j ACCEPT" in r for r in f), "replies to redirected connections must pass"
    assert any(f"--uid-owner {T.TOR_USER} -j ACCEPT" in r for r in f)
    v6 = [l for l in T.RULES_SCRIPT.splitlines() if l.startswith(f"ip6tables -A {T.V6_CHAIN}")]
    assert v6[-1].endswith("-j REJECT") and any("-o lo -j ACCEPT" in r for r in v6)


def test_the_script_is_idempotent_and_hooks_output_once():
    s = T.RULES_SCRIPT
    for chain, table in ((T.NAT_CHAIN, "-t nat "), (T.FILTER_CHAIN, ""), (T.V6_CHAIN, "")):
        tool = "ip6tables" if chain == T.V6_CHAIN else "iptables"
        assert f"{tool} {table}-N {chain} 2>/dev/null || {tool} {table}-F {chain}" in s, chain
        assert f"{tool} {table}-C OUTPUT -j {chain} 2>/dev/null || {tool} {table}-A OUTPUT -j {chain}" in s, chain
    assert s.startswith("set -e")


def test_the_verification_request_uses_no_proxy_flags():
    """The point is that a PLAIN request is transparently Tor."""
    cmd = T.verify_cmd()
    assert T.CHECK_URL in cmd and "socks" not in cmd and "proxy" not in cmd


def test_parse_tor_check():
    assert T.parse_tor_check('{"IsTor":true,"IP":"37.221.208.71"}') == (True, "37.221.208.71")
    assert T.parse_tor_check('{"IsTor":false,"IP":"1.2.3.4"}') == (False, "1.2.3.4")
    assert T.parse_tor_check("<html>challenge</html>") == (None, "")
    assert T.parse_tor_check("") == (None, "")


def test_start_and_running_commands_name_the_ghost_torrc_and_the_user():
    assert T.TORRC_PATH in T.start_tor_cmd() and "--RunAsDaemon 1" in T.start_tor_cmd()
    assert T.TOR_USER in T.tor_running_as_expected_cmd()
    assert "iptables" in T.apply_rules_cmd()


# --- the sequencing, through the exec stub --------------------------------

CHROMIUM = "find /root/.cache/ms-playwright"
CHROMIUM_PRESENT = (0, b"/root/.cache/ms-playwright/chromium-1/x/headless_shell\n")
TOR_OK = '{"IsTor":true,"IP":"37.221.208.71"}'


def _stub(workspace, network="bridge"):
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.host_workspace = Path(workspace)
    sb.image = "ghost-agent-base:latest"
    sb.container_name = "ghost-test-egress"
    sb.tor_proxy = "socks5://127.0.0.1:9050"
    sb.client = MagicMock()
    sb.docker_lib = MagicMock()
    _NF = type("ImageNotFound", (Exception,), {})
    sb.docker_lib.errors.ImageNotFound = _NF
    sb.ImageNotFound = _NF
    sb.NotFound = type("NotFound", (Exception,), {})
    sb.APIError = type("APIError", (Exception,), {})
    sb._lock = threading.Lock()
    sb.container = MagicMock()
    sb.container.attrs = {"HostConfig": {"NetworkMode": network}}
    sb._is_container_ready = MagicMock(return_value=True)
    return sb


def _drive(sb, overrides=None, marker_present=True):
    """Every exec is recorded as (cmd, kwargs). Marker and Chromium present
    by default (a provisioned image), Tor bootstrapped, verify → IsTor."""
    overrides = overrides or {}
    seen = []

    def _exec(cmd, *a, **k):
        seen.append((cmd, k))
        for key, val in overrides.items():
            if key in cmd:
                return val(cmd) if callable(val) else val
        if "test -f /root/.supercharged" in cmd:
            return (0, b"") if marker_present else (1, b"")
        if CHROMIUM in cmd:
            return CHROMIUM_PRESENT
        if "Bootstrapped 100" in cmd:
            return (0, b"")
        if T.CHECK_URL in cmd:
            return (0, TOR_OK.encode())
        return (0, b"")

    sb.container.exec_run.side_effect = _exec
    with patch("ghost_agent.sandbox.docker.pretty_log") as plog, \
         patch("ghost_agent.sandbox.docker.time.sleep", lambda *_a: None):
        sb.ensure_running()
    return seen, plog


def _cmds(seen):
    return [c for c, _ in seen]


def test_enforcement_runs_after_provisioning_not_before(tmp_path):
    """The hybrid: apt, pip and Chromium go DIRECT (the marker is missing
    here, so the full provision runs), the rules go in afterwards."""
    sb = _stub(tmp_path)
    seen, _ = _drive(sb, marker_present=False)
    cmds = _cmds(seen)
    last_install = max(i for i, c in enumerate(cmds) if "pip install" in c or "apt-get install" in c or "playwright install" in c)
    rules_at = next(i for i, c in enumerate(cmds) if T.NAT_CHAIN in c and "iptables -t nat -N" in c)
    assert rules_at > last_install, (last_install, rules_at)
    assert sb._egress_state == "enforced" and sb._egress_exit_ip == "37.221.208.71"


def test_the_rules_are_loaded_through_a_privileged_exec(tmp_path):
    """The container has no NET_ADMIN of its own — that is what stops the
    model from flushing the rules. World where it fails: the exec is plain
    and iptables fails with 'Permission denied' (or, worse, the container
    was given NET_ADMIN)."""
    sb = _stub(tmp_path)
    seen, _ = _drive(sb)
    rules = [(c, k) for c, k in seen if T.NAT_CHAIN in c and "iptables -t nat -N" in c]
    assert len(rules) == 1
    assert rules[0][1].get("privileged") is True, rules[0][1]
    others = [(c, k) for c, k in seen if k.get("privileged") and T.NAT_CHAIN not in c]
    assert not others, f"only the rule load may be privileged: {others}"


def test_rules_go_in_before_bootstrap_is_awaited_fail_closed(tmp_path):
    sb = _stub(tmp_path)
    seen, _ = _drive(sb)
    cmds = _cmds(seen)
    rules_at = next(i for i, c in enumerate(cmds) if "iptables -t nat -N" in c)
    boot_at = next(i for i, c in enumerate(cmds) if "Bootstrapped 100" in c)
    assert rules_at < boot_at, "the rules must be in BEFORE Tor is known to be up"


def test_tor_is_started_from_the_ghost_torrc_before_the_rules(tmp_path):
    sb = _stub(tmp_path)
    cmds = _cmds(_drive(sb)[0])
    torrc_at = next(i for i, c in enumerate(cmds) if T.TORRC_PATH in c and "printf" in c)
    start_at = next(i for i, c in enumerate(cmds) if "--RunAsDaemon 1" in c)
    rules_at = next(i for i, c in enumerate(cmds) if "iptables -t nat -N" in c)
    assert torrc_at < start_at < rules_at


def test_host_networking_is_never_enforced_and_says_so(tmp_path):
    """iptables in a shared namespace would rewrite the HOST's traffic."""
    sb = _stub(tmp_path, network="host")
    seen, plog = _drive(sb)
    assert not any("iptables" in c for c in _cmds(seen))
    assert sb._egress_state == "unavailable"
    assert any(c.args and c.args[0] == "Sandbox Egress" and "host networking" in str(c.args[1]) for c in plog.call_args_list)


def test_missing_iptables_is_an_error_not_a_silent_direct_egress(tmp_path):
    sb = _stub(tmp_path)
    seen, plog = _drive(sb, {"command -v iptables && command -v tor": (1, b"")})
    assert not any("iptables -t nat -N" in c for c in _cmds(seen))
    assert sb._egress_state == "unavailable"
    assert any(c.kwargs.get("level") == "ERROR" and "NOT enforced" in str(c.args[1]) for c in plog.call_args_list)


def test_a_failed_rule_load_is_an_error_and_state_unavailable(tmp_path):
    sb = _stub(tmp_path)
    seen, plog = _drive(sb, {"iptables -t nat -N": (2, b"iptables: Permission denied (you must be root).")})
    assert sb._egress_state == "unavailable"
    assert any("could not be loaded" in str(c.args[1]) for c in plog.call_args_list)


def test_no_bootstrap_stays_blocked_and_never_flushes(tmp_path):
    """Fail-closed: Tor that never bootstraps leaves the sandbox OFFLINE,
    not exposed. World where it fails: the timeout path removes the rules
    'so the user is not stuck'."""
    sb = _stub(tmp_path)
    with patch.object(T, "BOOTSTRAP_TIMEOUT_S", 0.0):
        seen, plog = _drive(sb, {"Bootstrapped 100": (1, b"")})
    assert sb._egress_state == "blocked"
    assert not any(("-F " in c or "-X " in c or "-D " in c) and "iptables" in c and "-N " not in c for c in _cmds(seen)), "rules were removed on failure"
    assert any("BLOCKED" in str(c.args[1]) for c in plog.call_args_list)


def test_tor_running_as_root_is_blocked_not_trusted(tmp_path):
    """A root-owned Tor is exempted from nothing and would be redirected into
    itself; the uid exemption only works for the unprivileged user."""
    sb = _stub(tmp_path)
    seen, plog = _drive(sb, {"ps -o user=": (1, b"")})
    assert sb._egress_state == "blocked"


def test_a_direct_answer_is_reported_as_a_leak(tmp_path):
    sb = _stub(tmp_path)
    seen, plog = _drive(sb, {T.CHECK_URL: (0, b'{"IsTor":false,"IP":"9.9.9.9"}')})
    assert sb._egress_state == "blocked"
    assert any(c.kwargs.get("level") == "ERROR" and "LEAK" in str(c.args[1]) for c in plog.call_args_list)


def test_an_unusable_verification_answer_is_enforced_unverified(tmp_path):
    sb = _stub(tmp_path)
    seen, plog = _drive(sb, {T.CHECK_URL: (0, b"<html>challenge</html>")})
    assert sb._egress_state == "enforced"
    assert any("no usable answer" in str(c.args[1]) for c in plog.call_args_list)


def test_enforcement_runs_once_per_generation_and_again_after_recreation(tmp_path):
    sb = _stub(tmp_path)
    seen1, _ = _drive(sb)
    seen2, _ = _drive(sb)
    assert sum(1 for c in _cmds(seen1) if "iptables -t nat -N" in c) == 1
    assert sum(1 for c in _cmds(seen2) if "iptables -t nat -N" in c) == 0, "re-applied on every command"
    # what container recreation does: the generation flags AND the
    # readiness stamp go back to their class defaults
    sb._tor_attempted = False
    sb._env_verified = False
    sb._last_ready_ok = 0.0 if isinstance(getattr(type(sb), '_last_ready_ok', 0.0), float) else False
    seen3, _ = _drive(sb)
    assert sum(1 for c in _cmds(seen3) if "iptables -t nat -N" in c) == 1


def test_without_a_tor_proxy_policy_nothing_is_enforced(tmp_path):
    sb = _stub(tmp_path)
    sb.tor_proxy = None
    seen, _ = _drive(sb)
    assert not any("iptables" in c for c in _cmds(seen))


def test_the_image_carries_iptables_at_v9():
    """The rules need the binary; v9 adds it in place from v8."""
    import inspect
    src = inspect.getsource(DockerSandbox._ensure_running_impl)
    assert 'marker_path = "/root/.supercharged.v9"' in src
    assert "apt-get install -y iptables'" in src
    assert "iptables" in (Path(__file__).resolve().parents[1] / "sandbox" / "Dockerfile").read_text()


def test_the_process_patterns_cannot_match_their_own_wrapper():
    """Live, 2026-09-09: an unanchored `pgrep -f 'tor -f …'` matched the
    `sh -c` wrapper running it (root), so Tor was never started and the
    running-check said "root" — the sandbox sat fail-closed. The pattern
    must anchor on Tor's own command line."""
    import re, shlex
    assert T.TOR_PROC_PATTERN.startswith("^tor -f ")
    for cmd in (T.start_tor_cmd(), T.tor_running_as_expected_cmd()):
        # what the wrapper's /proc/<pid>/cmdline holds: "sh -c <inner script>"
        argv = shlex.split(cmd)
        assert argv[:2] == ["sh", "-c"]
        wrapper_cmdline = " ".join(argv)
        pats = re.findall(r"pgrep -f '([^']+)'", argv[2])
        assert pats and all(p.startswith("^tor -f") for p in pats), argv[2]
        # the wrapper's own command line does not match the pattern it carries
        assert not re.search(pats[0], wrapper_cmdline), "the wrapper matches itself"
        assert re.search(pats[0], f"tor -f {T.TORRC_PATH} --RunAsDaemon 1"), "a real Tor does not match"


def test_a_fresh_tor_start_truncates_the_previous_bootstrap_log():
    """The container's filesystem outlives a restart; its processes do not.
    Without truncation the previous Tor's "Bootstrapped 100%" line makes
    the bootstrap poll pass before this Tor has a circuit. The truncation
    must sit in the START branch (never when Tor is already running, or a
    live Tor's log would be cut mid-bootstrap and the poll would never
    pass)."""
    import shlex
    inner = shlex.split(T.start_tor_cmd())[2]
    guard, _, start = inner.partition("||")
    assert "pgrep" in guard and T.TOR_LOG not in guard
    assert f": > {T.TOR_LOG}" in start
    assert start.index(f": > {T.TOR_LOG}") < start.index("tor -f"), "truncate BEFORE the start"
    assert f"chown {T.TOR_USER}:{T.TOR_USER} {T.TOR_LOG}" in start, "root-owned log; Tor drops to debian-tor"
