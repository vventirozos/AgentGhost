"""Transparent Tor for the sandbox (§4FU, 2026-09-09).

The sandbox container was a direct-egress island. Measured from inside the
live container on 2026-09-09: ``{"IsTor": false, "IP": <the operator's real
address>}`` — bridge networking, no proxy variables, no Tor process, DNS
through the host's resolver. The host's egress guard exempts the sandbox
on the belief that "its egress is enforced by the sandbox's internal Tor
daemon"; that daemon never ran. Across the trajectory corpus the model had
reached the network from the sandbox 161 times (125 curl/wget, 19 pip, 12
Python requests, 3 apt, 2 git), each from the real IP — and a fetched page
can tell the model to run a command that posts data somewhere, which then
leaves from the real IP past every host-side guard.

The operator chose transparent enforcement (option 1, hybrid):

  * an in-container Tor, run as the unprivileged ``debian-tor`` user, with a
    ``TransPort`` and a ``DNSPort`` beside its ``SocksPort``;
  * iptables rules in the container's OWN network namespace — the only
    place the original destination of a redirected connection survives —
    that send every TCP connection to the TransPort and every DNS query to
    the DNSPort, exempt only Tor's own traffic (matched by uid) and
    loopback, and REJECT everything else (other UDP, ICMP, all IPv6);
  * applied through a PRIVILEGED ``docker exec``: the container itself
    runs with Docker's default capability set, which lacks ``NET_ADMIN``,
    so a command the model runs (even as root) cannot flush the rules;
  * provisioning stays direct — apt, pip and the Chromium download run
    BEFORE the rules go in (the reason the old design gave for direct
    egress), so a full provision is not paid over Tor;
  * fail-closed: the rules are loaded before Tor has bootstrapped. Until it
    does, and if it ever dies, nothing leaves the sandbox at all.

The spike that settled the design, on the live container: bootstrap in
16 s; ``curl`` and raw ``urllib`` both ``IsTor: true`` on different exits;
``host example.com`` resolved through the DNSPort; UDP to port 123 and
IPv6 blocked; ``iptables -F`` as the container's root: "needs NET_ADMIN";
``pip download six`` succeeded over Tor.

This module holds the CONFIGURATION as data — the torrc, the rule script,
the probes — so it can be pinned without a container. ``sandbox/docker.py``
owns the sequencing.
"""
from __future__ import annotations

import json
import shlex
from typing import Optional, Tuple

TOR_USER = "debian-tor"
TORRC_PATH = "/etc/tor/torrc.ghost"
TOR_LOG = "/var/log/tor/notices.log"
SOCKS_PORT = 9050
TRANS_PORT = 9040
DNS_PORT = 5353
#: Tor maps .onion names to addresses in this block when resolved through
#: the DNSPort, so a following TCP connection can be recognised at the
#: TransPort (AutomapHostsOnResolve).
VIRTUAL_NET = "10.192.0.0/10"
NAT_CHAIN = "GHOST_TOR"
FILTER_CHAIN = "GHOST_TOR_F"
V6_CHAIN = "GHOST_TOR6"
BOOTSTRAP_TIMEOUT_S = 75.0
BOOTSTRAP_POLL_S = 2.0

TORRC = f"""# Written by ghost_agent (sandbox/tor_egress.py). Do not edit by hand.
User {TOR_USER}
DataDirectory /var/lib/tor
SocksPort 127.0.0.1:{SOCKS_PORT}
TransPort 127.0.0.1:{TRANS_PORT}
DNSPort 127.0.0.1:{DNS_PORT}
AutomapHostsOnResolve 1
VirtualAddrNetworkIPv4 {VIRTUAL_NET}
Log notice file {TOR_LOG}
"""

#: Idempotent: (re)creates the three chains, hooks them into OUTPUT once.
#: Order inside the nat chain is the contract — Tor's own traffic and
#: loopback must be exempted BEFORE the redirects, or Tor would be sent to
#: itself.
RULES_SCRIPT = f"""set -e
iptables -t nat -N {NAT_CHAIN} 2>/dev/null || iptables -t nat -F {NAT_CHAIN}
iptables -t nat -A {NAT_CHAIN} -m owner --uid-owner {TOR_USER} -j RETURN
iptables -t nat -A {NAT_CHAIN} -o lo -j RETURN
iptables -t nat -A {NAT_CHAIN} -d 127.0.0.0/8 -j RETURN
iptables -t nat -A {NAT_CHAIN} -p udp --dport 53 -j REDIRECT --to-ports {DNS_PORT}
iptables -t nat -A {NAT_CHAIN} -p tcp --dport 53 -j REDIRECT --to-ports {DNS_PORT}
iptables -t nat -A {NAT_CHAIN} -p tcp -j REDIRECT --to-ports {TRANS_PORT}
iptables -t nat -C OUTPUT -j {NAT_CHAIN} 2>/dev/null || iptables -t nat -A OUTPUT -j {NAT_CHAIN}
iptables -N {FILTER_CHAIN} 2>/dev/null || iptables -F {FILTER_CHAIN}
iptables -A {FILTER_CHAIN} -m owner --uid-owner {TOR_USER} -j ACCEPT
iptables -A {FILTER_CHAIN} -o lo -j ACCEPT
iptables -A {FILTER_CHAIN} -d 127.0.0.0/8 -j ACCEPT
iptables -A {FILTER_CHAIN} -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A {FILTER_CHAIN} -j REJECT --reject-with icmp-port-unreachable
iptables -C OUTPUT -j {FILTER_CHAIN} 2>/dev/null || iptables -A OUTPUT -j {FILTER_CHAIN}
ip6tables -N {V6_CHAIN} 2>/dev/null || ip6tables -F {V6_CHAIN}
ip6tables -A {V6_CHAIN} -o lo -j ACCEPT
ip6tables -A {V6_CHAIN} -j REJECT
ip6tables -C OUTPUT -j {V6_CHAIN} 2>/dev/null || ip6tables -A OUTPUT -j {V6_CHAIN}
"""

#: The one address whose answer is a verdict, not just an echo.
CHECK_URL = "https://check.torproject.org/api/ip"


def write_torrc_cmd() -> str:
    """A shell command that writes the torrc and prepares the log dir."""
    return ("sh -c " + shlex.quote(
        f"mkdir -p /var/log/tor && chown {TOR_USER}:{TOR_USER} /var/log/tor && "
        f"printf '%s' {shlex.quote(TORRC)} > {TORRC_PATH}"))


#: ⚠ ANCHORED. `pgrep -f` matches against whole command lines, and the
#: `sh -c "pgrep -f 'tor -f …' …"` wrapper that runs these commands carries
#: the pattern in its own command line — so an unanchored pattern matched
#: the wrapper (running as root), the start was skipped as "already
#: running", and the running-check reported "root". Live on 2026-09-09:
#: the rules went in, Tor never started, the sandbox sat fail-closed. Tor's
#: own command line begins with `tor -f`; the wrapper's begins with `sh`.
TOR_PROC_PATTERN = f"^tor -f {TORRC_PATH}"


def start_tor_cmd() -> str:
    """Start Tor with the ghost torrc unless one is already running on it.
    Started as root; `User debian-tor` in the torrc drops privileges, which
    is what the uid exemption in the rules relies on.

    A FRESH start truncates the notice log first: the container's
    filesystem outlives a restart while its processes do not, so the log
    of the previous Tor still says "Bootstrapped 100%" and the bootstrap
    poll would pass before this Tor has a circuit (seen on the live
    container: two 100% lines from two starts)."""
    return ("sh -c " + shlex.quote(
        f"pgrep -f '{TOR_PROC_PATTERN}' >/dev/null 2>&1 || "
        f"{{ : > {TOR_LOG}; chown {TOR_USER}:{TOR_USER} {TOR_LOG}; tor -f {TORRC_PATH} --RunAsDaemon 1; }}"))


def tor_running_as_expected_cmd() -> str:
    """Exit 0 when a Tor started from the ghost torrc is running AS the
    unprivileged user (a root-owned Tor would be exempted from nothing and
    redirected into itself)."""
    return ("sh -c " + shlex.quote(
        f"pid=$(pgrep -f '{TOR_PROC_PATTERN}' | head -1) && [ -n \"$pid\" ] && "
        f"[ \"$(ps -o user= -p $pid | tr -d ' ')\" = {TOR_USER} ]"))


def apply_rules_cmd() -> str:
    """Must be run with ``privileged=True`` — the container has no NET_ADMIN."""
    return "sh -c " + shlex.quote(RULES_SCRIPT)


def rules_present_cmd() -> str:
    return "sh -c " + shlex.quote(f"iptables -t nat -C OUTPUT -j {NAT_CHAIN} && iptables -C OUTPUT -j {FILTER_CHAIN}")


def bootstrapped_cmd() -> str:
    return "sh -c " + shlex.quote(f"grep -q 'Bootstrapped 100' {TOR_LOG}")


def verify_cmd(timeout_s: int = 30) -> str:
    """No proxy flags on purpose: the point is that a plain request is
    transparently Tor."""
    return f"curl -s -m {int(timeout_s)} {CHECK_URL}"


def parse_tor_check(output: str) -> Tuple[Optional[bool], str]:
    """``(is_tor, ip)`` from the check endpoint's JSON; ``(None, "")`` when
    the output is not that JSON (blocked, timed out, HTML challenge)."""
    try:
        data = json.loads((output or "").strip())
        return bool(data.get("IsTor")), str(data.get("IP") or "")
    except Exception:  # noqa: BLE001
        return None, ""


#: Deep leak probes, for the operator's script and the live check — not
#: run on every start. Each prints "blocked"/"tor"/"LEAK …".
LEAK_PROBES = {
    "raw python socket": (
        "python3 -c \"import urllib.request; print(urllib.request.urlopen("
        f"'{CHECK_URL}', timeout=40).read().decode())\""),
    "udp non-dns": "sh -c '(echo x | timeout 5 nc -u -w 3 1.1.1.1 123 >/dev/null 2>&1 && echo LEAK) || echo blocked'",
    "ipv6 direct": "sh -c '(curl -6 -s -m 8 https://ipv6.google.com >/dev/null 2>&1 && echo LEAK) || echo blocked'",
    "root edits rules": f"sh -c '(iptables -t nat -F {NAT_CHAIN} 2>/dev/null && echo LEAK) || echo blocked'",
    "dns via tor": "sh -c 'host -W 15 example.com 2>&1 | head -1'",
}
