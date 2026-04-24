"""Tests for the network egress guard.

The guard is opt-in (context manager); these tests cover that when it's
active, external destinations raise, and that the monkey-patch is
reverted on exit.
"""

import socket

import pytest

from ghost_agent.eval.network_guard import no_external_network, NetworkEgressError


def test_guard_blocks_external_address():
    with no_external_network():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(NetworkEgressError):
            s.connect(("8.8.8.8", 53))
        s.close()


def test_guard_allows_loopback():
    # Spin up a local listener so we can attempt a real connect without
    # hitting the network. If the connect itself fails (e.g. ephemeral
    # port), that's fine — we only care that the guard doesn't block it.
    lst = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lst.bind(("127.0.0.1", 0))
    lst.listen(1)
    port = lst.getsockname()[1]
    try:
        with no_external_network():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.connect(("127.0.0.1", port))
            except NetworkEgressError:
                pytest.fail("guard incorrectly blocked loopback connect")
            except OSError:
                # Any other OSError is benign — we only care about the guard
                pass
            finally:
                s.close()
    finally:
        lst.close()


def test_guard_reverts_on_exit():
    original = socket.socket.connect
    with no_external_network():
        assert socket.socket.connect is not original
    assert socket.socket.connect is original


def test_guard_reverts_on_exception_inside_block():
    original = socket.socket.connect
    with pytest.raises(RuntimeError):
        with no_external_network():
            raise RuntimeError("inner failure")
    assert socket.socket.connect is original


def test_guard_extra_allow_passes_named_host():
    with no_external_network(extra_allow=("10.0.0.5",)):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # The connection itself will fail (10.0.0.5 likely unreachable);
        # we only care that the guard doesn't raise first.
        try:
            s.connect(("10.0.0.5", 1))
        except NetworkEgressError:
            pytest.fail("extra_allow didn't allow the listed host")
        except OSError:
            pass
        finally:
            s.close()


def test_guard_is_reentrant_safe():
    original = socket.socket.connect
    with no_external_network():
        with no_external_network():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with pytest.raises(NetworkEgressError):
                s.connect(("8.8.8.8", 53))
            s.close()
        # Outer block should still have a guard installed
        assert socket.socket.connect is not original
    assert socket.socket.connect is original
