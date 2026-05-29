"""Network-egress assertion.

Ghost is privacy-by-design. The eval harness is the last line of defense
against a new dependency or prompt change silently phoning home: before
and during an eval run we can wrap the code under test with
`no_external_network()`, which turns any non-loopback outbound socket
into a loud exception instead of a silent packet.

Not autouse. Opt-in via context manager so existing tests and code
paths are never perturbed.
"""

from __future__ import annotations

import contextlib
import socket
from typing import Iterable, Set, Tuple


class NetworkEgressError(RuntimeError):
    """Raised when a connection to a non-allowlisted host is attempted
    inside `no_external_network()`."""


_DEFAULT_ALLOW_HOSTS: Tuple[str, ...] = (
    "127.0.0.1",
    "localhost",
    "::1",
    "0.0.0.0",
)


def _host_from_addr(addr) -> str:
    """Extract the host string from a getaddrinfo-shaped address.

    socket.connect accepts:
      - (host, port) for AF_INET
      - (host, port, flowinfo, scopeid) for AF_INET6
      - path string for AF_UNIX
    """
    if isinstance(addr, (tuple, list)) and addr:
        return str(addr[0])
    if isinstance(addr, (str, bytes)):
        # AF_UNIX socket path — always local
        return ""
    return ""


def _is_loopback_host(host: str, allow: Iterable[str]) -> bool:
    if not host:
        return True  # empty = AF_UNIX path, not external
    if host in set(allow):
        return True
    # Catch 127.x.y.z range (loopback) without importing ipaddress for
    # every call in the hot path.
    if host.startswith("127."):
        return True
    return False


@contextlib.contextmanager
def no_external_network(extra_allow: Iterable[str] = ()):
    """Context manager that makes `socket.socket.connect` raise on any
    non-loopback destination.

    Use for eval runs where we want a hard guarantee no bytes leave the
    machine. Skip for tests that genuinely need the network (we don't
    have any — Ghost is fully local — but the knob exists).
    """
    allow: Set[str] = set(_DEFAULT_ALLOW_HOSTS) | set(extra_allow)
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto
    original_sendmsg = socket.socket.sendmsg

    def guarded_connect(self, addr):
        host = _host_from_addr(addr)
        if not _is_loopback_host(host, allow):
            raise NetworkEgressError(
                f"eval guard blocked outbound connect to {addr!r}; "
                "Ghost eval is strictly offline"
            )
        return original_connect(self, addr)

    def guarded_connect_ex(self, addr):
        host = _host_from_addr(addr)
        if not _is_loopback_host(host, allow):
            raise NetworkEgressError(
                f"eval guard blocked outbound connect_ex to {addr!r}; "
                "Ghost eval is strictly offline"
            )
        return original_connect_ex(self, addr)

    def guarded_sendto(self, data, *args):
        # sendto(data, address) or sendto(data, flags, address): the
        # destination is always the LAST positional arg. Connectionless
        # egress (UDP, DNS-tunnel, QUIC) never calls connect(), so without
        # guarding sendto/sendmsg the "no bytes leave the machine"
        # guarantee had a hole.
        if args:
            host = _host_from_addr(args[-1])
            if not _is_loopback_host(host, allow):
                raise NetworkEgressError(
                    f"eval guard blocked outbound sendto to {args[-1]!r}; "
                    "Ghost eval is strictly offline"
                )
        return original_sendto(self, data, *args)

    def guarded_sendmsg(self, buffers, ancdata=(), flags=0, address=None):
        if address is not None:
            host = _host_from_addr(address)
            if not _is_loopback_host(host, allow):
                raise NetworkEgressError(
                    f"eval guard blocked outbound sendmsg to {address!r}; "
                    "Ghost eval is strictly offline"
                )
            return original_sendmsg(self, buffers, ancdata, flags, address)
        return original_sendmsg(self, buffers, ancdata, flags)

    socket.socket.connect = guarded_connect         # type: ignore[assignment]
    socket.socket.connect_ex = guarded_connect_ex   # type: ignore[assignment]
    socket.socket.sendto = guarded_sendto           # type: ignore[assignment]
    socket.socket.sendmsg = guarded_sendmsg         # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect           # type: ignore[assignment]
        socket.socket.connect_ex = original_connect_ex     # type: ignore[assignment]
        socket.socket.sendto = original_sendto             # type: ignore[assignment]
        socket.socket.sendmsg = original_sendmsg           # type: ignore[assignment]
