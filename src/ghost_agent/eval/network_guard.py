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
    """Block non-loopback egress from THIS PROCESS for the duration.

    ⚠ SCOPE — read before relying on this. It is a best-effort guard over the
    Python socket layer, **not** "a hard guarantee no bytes leave the
    machine" (which is what this docstring used to claim, wrongly):

    * It patches `socket.socket.{connect,connect_ex,sendto,sendmsg}` plus
      `socket.create_connection` and `getaddrinfo`. That covers stdlib
      sockets, `httpx`, and anything built on them.
    * It CANNOT stop a subprocess (`curl`, `git`, a sandboxed command) — a
      child process has its own address space and its own libc.
    * It CANNOT stop a library that opens sockets outside CPython's socket
      module. **`curl_cffi` is exactly that**, and `curl_cffi` is what
      `tools/search.py`, `tools/darkweb_search.py`, `tools/file_system.py`,
      `tools/system.py` and `utils/helpers.py` use — i.e. every real egress
      path in this agent is exempt. Measured: raw socket / create_connection
      / httpx blocked; curl_cffi, subprocess curl and uvloop all LEAKED.
    * Under `--runner http` the code under test is the LIVE AGENT in another
      process; the only socket in this one is the allowlisted driver
      connection, so the guard is structurally moot there.

    Use it as a tripwire for accidental stdlib egress in a driver process,
    and rely on `--mandatory-tor` / the sandbox egress guard for anything
    that must actually hold.
    """
    allow: Set[str] = set(_DEFAULT_ALLOW_HOSTS) | set(extra_allow)
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto
    original_sendmsg = socket.socket.sendmsg
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo

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

    def guarded_create_connection(address, *a, **kw):
        host = _host_from_addr(address)
        if not _is_loopback_host(host, allow):
            raise NetworkEgressError(
                f"eval guard blocked create_connection to {address!r}; "
                "Ghost eval is strictly offline")
        return original_create_connection(address, *a, **kw)

    def guarded_getaddrinfo(host, *a, **kw):
        # Resolution itself leaves the machine on most resolvers.
        if not _is_loopback_host(str(host or ""), allow):
            raise NetworkEgressError(
                f"eval guard blocked DNS resolution of {host!r}; "
                "Ghost eval is strictly offline")
        return original_getaddrinfo(host, *a, **kw)

    socket.create_connection = guarded_create_connection
    socket.getaddrinfo = guarded_getaddrinfo
    socket.socket.connect = guarded_connect         # type: ignore[assignment]
    socket.socket.connect_ex = guarded_connect_ex   # type: ignore[assignment]
    socket.socket.sendto = guarded_sendto           # type: ignore[assignment]
    socket.socket.sendmsg = guarded_sendmsg         # type: ignore[assignment]
    try:
        yield
    finally:
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo
        socket.socket.connect = original_connect           # type: ignore[assignment]
        socket.socket.connect_ex = original_connect_ex     # type: ignore[assignment]
        socket.socket.sendto = original_sendto             # type: ignore[assignment]
        socket.socket.sendmsg = original_sendmsg           # type: ignore[assignment]
