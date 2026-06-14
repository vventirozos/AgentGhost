"""Tests for `compute_tor_proxy` — which destinations egress via Tor.

Regression guard for the image-gen-over-Tor bug: a Tailscale/CGNAT compute
node (100.64.0.0/10) is our OWN infrastructure and is unreachable from a Tor
exit, so it must be reached directly — not routed through the SOCKS proxy.
The old `is_private or is_loopback` test let CGNAT fall through to Tor and
every image generation failed with "All connection attempts failed".
"""

import pytest

from ghost_agent.core.llm import compute_tor_proxy

TOR = "socks5://127.0.0.1:9050"
TOR_H = "socks5h://127.0.0.1:9050"


def test_no_tor_configured_returns_none():
    assert compute_tor_proxy("http://8.8.8.8:80", None) is None
    assert compute_tor_proxy("http://8.8.8.8:80", "") is None


def test_public_ipv4_routes_via_tor_socks5h():
    # Genuinely public address → must go through Tor, normalised to socks5h.
    assert compute_tor_proxy("http://8.8.8.8:80", TOR) == TOR_H


def test_public_ipv6_routes_via_tor():
    assert compute_tor_proxy("http://[2606:4700:4700::1111]:443", TOR) == TOR_H


def test_public_hostname_routes_via_tor():
    # Non-IP host can't be classified locally → conservatively via Tor.
    assert compute_tor_proxy("http://example.com", TOR) == TOR_H


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8088",
        "http://localhost:8088",
        "http://10.0.0.5:8000",
        "http://192.168.1.10:8000",
        "http://172.16.0.1:8000",
        "http://169.254.1.1:8000",       # link-local
        "http://[fd00::1]:8000",          # IPv6 ULA
        "http://gpu-box.local:8000",      # mDNS .local
    ],
)
def test_local_and_lan_destinations_bypass_tor(url):
    assert compute_tor_proxy(url, TOR) is None


@pytest.mark.parametrize(
    "url",
    [
        "http://100.83.184.117:8000",     # the real image-gen node from the repro
        "http://100.64.0.0:8000",         # CGNAT range start
        "http://100.127.255.254:8000",    # CGNAT range end
        "100.83.184.117:8000",            # scheme-less variant
    ],
)
def test_tailscale_cgnat_nodes_bypass_tor(url):
    # The bug: these were forced through Tor (is_private=False) and failed.
    assert compute_tor_proxy(url, TOR) is None
