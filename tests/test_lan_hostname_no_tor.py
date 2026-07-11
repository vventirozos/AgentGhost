"""Regression: LAN nodes given as a BARE HOSTNAME must not be forced through
Tor (found live 2026-07-11).

`--worker-nodes http://nova:8088|Nova` was routed through the SOCKS proxy
because `nova` is neither `localhost`, nor `*.local`, nor an IP literal — so
every offloaded call died with `ProxyError` and silently fell back to the main
model. The log showed "Routing background task to Worker Node (Nova)" followed
by "All worker nodes failed", i.e. offloading LOOKED configured while doing
nothing at all. The same hole hit an image-gen node at `http://ghost:8000`.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.llm import compute_tor_proxy

TOR = "socks5://127.0.0.1:9050"
TOR_H = "socks5h://127.0.0.1:9050"


class TestLanHostnamesBypassTor:
    @pytest.mark.parametrize("url", [
        "http://nova:8088",          # the worker node that broke
        "http://ghost:8000",         # the image-gen node, same hole
        "http://raspberrypi:8000",
        "http://nova:8088/v1/chat/completions",
        "nova:8088",                 # scheme-less
    ])
    def test_bare_hostname_goes_direct(self, url):
        assert compute_tor_proxy(url, TOR) is None

    @pytest.mark.parametrize("url", [
        "http://localhost:8000",
        "http://nova.local:8088",
        "http://ghost.lan:8080",
        "http://box.home:9000",
        "http://svc.internal:80",
        "http://x.home.arpa:80",
    ])
    def test_lan_names_go_direct(self, url):
        assert compute_tor_proxy(url, TOR) is None

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:8088",
        "http://192.168.0.24:8000",
        "http://10.1.2.3:80",
        "http://100.122.46.101:8000",   # Tailscale CGNAT
    ])
    def test_private_ips_go_direct(self, url):
        assert compute_tor_proxy(url, TOR) is None


class TestPublicStillTorRouted:
    """The security property must survive the fix: anything genuinely public
    still egresses through Tor, and no cleartext DNS lookup is performed to
    decide it."""

    @pytest.mark.parametrize("url", [
        "http://api.openai.com",
        "https://example.com:8443",
        "http://8.8.8.8:80",
        # The classic bypass attempt the hostname parse exists to defeat.
        "http://localhost.attacker.example/",
        "http://nova.attacker.example/",   # dotted → public, despite "nova"
        # A public IPv6 literal has COLONS AND NO DOTS. The dotless-hostname
        # rule must not mistake it for a LAN name — that would leak real-IP
        # traffic outside Tor. (Caught by the suite while fixing the LAN
        # hostname hole; pinned here from both sides.)
        "http://[2606:4700:4700::1111]:443",
    ])
    def test_public_targets_use_tor(self, url):
        assert compute_tor_proxy(url, TOR) == TOR_H

    @pytest.mark.parametrize("url", [
        "http://[::1]:8088",                      # IPv6 loopback
        "http://[fd00::1]:8088",                  # IPv6 ULA
    ])
    def test_private_ipv6_goes_direct(self, url):
        assert compute_tor_proxy(url, TOR) is None

    def test_no_tor_configured_means_direct(self):
        assert compute_tor_proxy("http://api.openai.com", None) is None

    def test_socks5h_normalisation(self):
        # DNS must resolve inside Tor, never locally.
        assert compute_tor_proxy("http://example.com", TOR).startswith("socks5h://")


class TestConsistentWithNotifier:
    """utils.notify makes the same LAN/public call for push targets — the two
    must not disagree, or a node reachable by one path is unreachable by the
    other."""

    @pytest.mark.parametrize("url", [
        "http://nova:8088", "http://ghost.lan:8090", "http://192.168.0.5:80",
        "https://ntfy.sh/topic", "http://8.8.8.8/x",
    ])
    def test_same_verdict(self, url):
        from ghost_agent.utils.notify import url_needs_tor
        via_tor_llm = compute_tor_proxy(url, TOR) is not None
        assert via_tor_llm == url_needs_tor(url), url
