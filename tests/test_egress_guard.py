"""Tests for the runtime fail-closed Tor egress guard.

CRITICAL: every test that calls install() MUST uninstall in a finally /
fixture teardown — a leaked socket monkeypatch would break the rest of
the suite. The `guard_cleanup` autouse fixture is the backstop.
"""

import socket

import pytest

from ghost_agent.utils import egress_guard
from ghost_agent.utils.egress_guard import (
    MandatoryTorError,
    install,
    is_allowed_host,
    is_installed,
    parse_socks_endpoint,
    tor_liveness_ok,
)


@pytest.fixture(autouse=True)
def guard_cleanup():
    """Backstop: ensure the guard is never left installed after a test —
    a leaked socket monkeypatch would break every later test."""
    yield
    if is_installed():
        o = egress_guard._ORIGINALS
        if o:
            socket.socket.connect = o["connect"]
            socket.socket.connect_ex = o["connect_ex"]
            socket.socket.sendto = o["sendto"]
            socket.socket.sendmsg = o["sendmsg"]
        egress_guard._INSTALLED = False


# ──────────────────────────────────────────────────────────────────────
# host classification
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("host", [
    "127.0.0.1", "127.0.0.5", "::1", "0.0.0.0",
    "192.168.0.20", "10.0.0.1", "172.16.5.5",   # RFC1918 private (LAN nodes)
    "169.254.1.1", "fe80::1",                     # link-local
    "localhost", "myhost.local", "db.internal",   # local names
    "",                                           # AF_UNIX
])
def test_allowed_hosts(host):
    assert is_allowed_host(host) is True


@pytest.mark.parametrize("host", [
    "8.8.8.8", "1.1.1.1", "142.250.80.46",        # public IPv4
    "2606:4700:4700::1111",                        # public IPv6
    "example.com", "duckduckgo.com",               # bare public hostnames
])
def test_blocked_hosts(host):
    assert is_allowed_host(host) is False


def test_extra_allow_passes():
    assert is_allowed_host("8.8.8.8") is False
    assert is_allowed_host("8.8.8.8", allow={"8.8.8.8"}) is True


# ──────────────────────────────────────────────────────────────────────
# socks endpoint parsing
# ──────────────────────────────────────────────────────────────────────

def test_parse_socks_endpoint():
    assert parse_socks_endpoint("socks5://127.0.0.1:9050") == ("127.0.0.1", 9050)
    assert parse_socks_endpoint("socks5h://10.0.0.2:9150") == ("10.0.0.2", 9150)
    assert parse_socks_endpoint(None) is None
    assert parse_socks_endpoint("") is None


# ──────────────────────────────────────────────────────────────────────
# install / block / uninstall
# ──────────────────────────────────────────────────────────────────────

def test_install_blocks_public_connect():
    uninstall = install("socks5://127.0.0.1:9050")
    try:
        assert is_installed()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(MandatoryTorError):
                # Guard raises BEFORE any packet — no real network needed.
                s.connect(("8.8.8.8", 53))
        finally:
            s.close()
    finally:
        uninstall()
        assert not is_installed()


def test_install_allows_loopback_connect():
    uninstall = install("socks5://127.0.0.1:9050")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        try:
            # Loopback is allowed: the guard does NOT raise. The actual
            # connect to a closed port raises ConnectionRefused/timeout —
            # anything EXCEPT our guard error means "allowed through".
            with pytest.raises(Exception) as ei:
                s.connect(("127.0.0.1", 1))
            assert not isinstance(ei.value, MandatoryTorError)
        finally:
            s.close()
    finally:
        uninstall()


def test_uninstall_restores():
    original = socket.socket.connect
    uninstall = install("socks5://127.0.0.1:9050")
    assert socket.socket.connect is not original
    uninstall()
    assert socket.socket.connect is original
    assert not is_installed()


def test_double_install_idempotent():
    u1 = install("socks5://127.0.0.1:9050")
    patched = socket.socket.connect
    u2 = install("socks5://127.0.0.1:9050")  # no-op
    assert socket.socket.connect is patched
    u2()
    assert not is_installed()
    # u1 after u2 is harmless.
    u1()
    assert not is_installed()


def test_sendto_blocked_for_public():
    uninstall = install("socks5://127.0.0.1:9050")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            with pytest.raises(MandatoryTorError):
                s.sendto(b"x", ("8.8.8.8", 53))
        finally:
            s.close()
    finally:
        uninstall()


# ──────────────────────────────────────────────────────────────────────
# liveness probe
# ──────────────────────────────────────────────────────────────────────

def test_tor_liveness_ok_true_with_listener():
    lst = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lst.bind(("127.0.0.1", 0))
    lst.listen(1)
    port = lst.getsockname()[1]
    try:
        assert tor_liveness_ok(f"socks5://127.0.0.1:{port}") is True
    finally:
        lst.close()


def test_tor_liveness_false_when_down():
    # Grab an ephemeral port then close it → connect refused.
    tmp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tmp.bind(("127.0.0.1", 0))
    port = tmp.getsockname()[1]
    tmp.close()
    assert tor_liveness_ok(f"socks5://127.0.0.1:{port}", timeout=0.3) is False


def test_tor_liveness_false_when_no_proxy():
    assert tor_liveness_ok(None) is False
    assert tor_liveness_ok("") is False
