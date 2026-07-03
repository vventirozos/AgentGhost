"""Runtime fail-closed Tor egress guard.

The README promises Ghost "fails closed if ``TOR_PROXY`` is unset or the
Tor daemon is unreachable — a silently-cleartext agent is worse than a
stalled one." Historically that promise was *aspirational*: every
outbound helper degraded to a direct cleartext connection when the
proxy was absent or down (``core.llm.get_proxy`` returns ``None``,
``tools.search`` adds ``proxy=`` only ``if tor_proxy``, …). This module
makes the promise real.

The mechanism is the production-grade sibling of
``eval.network_guard.no_external_network``: it monkeypatches the
outbound socket primitives (``connect`` / ``connect_ex`` / ``sendto`` /
``sendmsg``) process-wide and raises :class:`MandatoryTorError` on any
connection to a **globally-routable (public) address**.

Why that policy is exactly right for anonymity, not over-broad:

  * All *anonymised* traffic egresses through the Tor SOCKS proxy at
    ``127.0.0.1:9050`` — i.e. the socket layer sees a **loopback**
    connect, which is allowed. With ``socks5h`` the hostname is resolved
    by Tor's exit, so even DNS doesn't produce a public connect here.
  * Local infrastructure — the llama.cpp upstream, ChromaDB, a LAN
    Postgres, LAN swarm / image-gen nodes — lives on loopback or
    RFC1918 / link-local addresses, which are allowed (LAN traffic does
    not deanonymise via a Tor exit).
  * The ONLY thing blocked is a **direct connection to a public IP** —
    which, by construction, is a connection that bypassed Tor. That is
    precisely the leak the README claims is impossible.

Plus a boot-time liveness probe (:func:`tor_liveness_ok`) so the agent
can *fail to start* rather than start cleartext when ``--mandatory-tor``
is set and Tor is unreachable.

Opt-in via ``--mandatory-tor`` so existing deployments are never
perturbed; when set, the guard is installed for the whole process
lifetime and the liveness probe gates boot.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Callable, Iterable, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("GhostAgent")


class MandatoryTorError(RuntimeError):
    """Raised when a direct connection to a public address is attempted
    while the mandatory-Tor guard is installed."""


# Hostnames that are inherently local and never need a DNS round-trip to
# classify. Anything else non-IP is treated conservatively (blocked) —
# in the Tor path the socket connects to the loopback proxy, so a raw
# public *hostname* reaching connect() means something bypassed the
# proxy.
_LOCAL_HOST_NAMES: Tuple[str, ...] = ("localhost", "ip6-localhost", "ip6-loopback")
_LOCAL_HOST_SUFFIXES: Tuple[str, ...] = (".local", ".lan", ".internal", ".localhost")


def parse_socks_endpoint(tor_proxy: Optional[str]) -> Optional[Tuple[str, int]]:
    """Return ``(host, port)`` for a ``socks5[h]://host:port`` URL, or
    ``None`` when ``tor_proxy`` is falsy / unparseable."""
    if not tor_proxy:
        return None
    try:
        # urlparse needs a scheme it recognises for netloc parsing; the
        # socks5h scheme parses fine on modern Python.
        parsed = urlparse(tor_proxy)
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port or 9050)
        return host, port
    except Exception:
        return None


def _host_from_addr(addr) -> str:
    """Extract the host string from a getaddrinfo-shaped address.

    ``connect`` accepts ``(host, port)`` (AF_INET), ``(host, port,
    flowinfo, scopeid)`` (AF_INET6), or a path/bytes for AF_UNIX (local).
    """
    if isinstance(addr, (tuple, list)) and addr:
        return str(addr[0])
    # AF_UNIX path (str/bytes) or anything else → treat as local.
    return ""


def is_allowed_host(host: str, allow: Iterable[str] = ()) -> bool:
    """Allow loopback / private / link-local / multicast / reserved
    destinations and an explicit allowlist; block globally-routable
    (public) addresses.

    A direct connect to a public IP is, by construction, traffic that
    bypassed the Tor SOCKS proxy (Tor egress is a loopback connect), so
    blocking it is exactly the fail-closed guarantee.
    """
    if not host:
        return True  # AF_UNIX / empty → local
    if host in set(allow):
        return True
    # Non-IP hostname: only the inherently-local ones are allowed; any
    # other bare hostname reaching connect() bypassed socks5h remote
    # resolution, so it is blocked.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        h = host.lower().rstrip(".")
        if h in _LOCAL_HOST_NAMES:
            return True
        return any(h.endswith(sfx) for sfx in _LOCAL_HOST_SUFFIXES)
    # Multicast is link/LAN discovery traffic (mDNS, SSDP), never a unicast
    # public egress — and it cannot route via Tor anyway (Tor is TCP-only).
    # Must be checked BEFORE is_global: CPython reports 224.0.0.0/4 and
    # ff00::/8 as is_global=True because they sit outside the IANA
    # private-use registries, which would wrongly fail-close LAN discovery.
    if ip.is_multicast:
        return True
    # Globally-routable == public Internet == must have gone via Tor.
    if ip.is_global:
        return False
    # loopback / private (RFC1918) / link-local / reserved /
    # unspecified (0.0.0.0) are all local-or-LAN → allowed.
    return True


# Module-level handle so install/uninstall is idempotent and testable.
_INSTALLED: bool = False
_ORIGINALS: dict = {}


def is_installed() -> bool:
    return _INSTALLED


def install(tor_proxy: Optional[str], extra_allow: Iterable[str] = ()) -> Callable[[], None]:
    """Install the process-wide guard. Returns an idempotent uninstall
    callable. Safe to call twice (second call is a no-op that returns the
    same uninstaller)."""
    global _INSTALLED, _ORIGINALS

    allow: Set[str] = set(extra_allow)
    ep = parse_socks_endpoint(tor_proxy)
    if ep is not None:
        allow.add(ep[0])  # loopback already allowed, but explicit is clearer

    def _uninstall() -> None:
        global _INSTALLED
        if not _INSTALLED:
            return
        socket.socket.connect = _ORIGINALS["connect"]          # type: ignore[assignment]
        socket.socket.connect_ex = _ORIGINALS["connect_ex"]    # type: ignore[assignment]
        socket.socket.sendto = _ORIGINALS["sendto"]            # type: ignore[assignment]
        socket.socket.sendmsg = _ORIGINALS["sendmsg"]          # type: ignore[assignment]
        _INSTALLED = False

    if _INSTALLED:
        return _uninstall

    _ORIGINALS = {
        "connect": socket.socket.connect,
        "connect_ex": socket.socket.connect_ex,
        "sendto": socket.socket.sendto,
        "sendmsg": socket.socket.sendmsg,
    }

    def guarded_connect(self, addr):
        host = _host_from_addr(addr)
        if not is_allowed_host(host, allow):
            raise MandatoryTorError(
                f"mandatory-tor guard blocked direct connect to {addr!r}: "
                "public egress must route through Tor (socks5h loopback). "
                "A silently-cleartext connection is worse than a stalled one."
            )
        return _ORIGINALS["connect"](self, addr)

    def guarded_connect_ex(self, addr):
        host = _host_from_addr(addr)
        if not is_allowed_host(host, allow):
            raise MandatoryTorError(
                f"mandatory-tor guard blocked direct connect_ex to {addr!r}"
            )
        return _ORIGINALS["connect_ex"](self, addr)

    def guarded_sendto(self, data, *args):
        # destination is the last positional arg; guards connectionless
        # egress (UDP / DNS-tunnel / QUIC) that never calls connect().
        if args:
            host = _host_from_addr(args[-1])
            if not is_allowed_host(host, allow):
                raise MandatoryTorError(
                    f"mandatory-tor guard blocked direct sendto to {args[-1]!r}"
                )
        return _ORIGINALS["sendto"](self, data, *args)

    def guarded_sendmsg(self, buffers, ancdata=(), flags=0, address=None):
        if address is not None:
            host = _host_from_addr(address)
            if not is_allowed_host(host, allow):
                raise MandatoryTorError(
                    f"mandatory-tor guard blocked direct sendmsg to {address!r}"
                )
            return _ORIGINALS["sendmsg"](self, buffers, ancdata, flags, address)
        return _ORIGINALS["sendmsg"](self, buffers, ancdata, flags)

    socket.socket.connect = guarded_connect          # type: ignore[assignment]
    socket.socket.connect_ex = guarded_connect_ex    # type: ignore[assignment]
    socket.socket.sendto = guarded_sendto            # type: ignore[assignment]
    socket.socket.sendmsg = guarded_sendmsg          # type: ignore[assignment]
    _INSTALLED = True
    return _uninstall


def tor_liveness_ok(tor_proxy: Optional[str], *, timeout: float = 5.0) -> bool:
    """Boot-time probe: can we reach the Tor SOCKS port?

    A plain TCP connect to the SOCKS host:port is a sufficient liveness
    signal — if the daemon is down the connect is refused. Uses the
    *original* socket so it works whether or not the guard is installed
    (the SOCKS endpoint is loopback and allowed regardless). Returns
    ``False`` on any failure rather than raising.
    """
    ep = parse_socks_endpoint(tor_proxy)
    if ep is None:
        return False
    host, port = ep
    connect = _ORIGINALS.get("connect", socket.socket.connect)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(timeout)
        connect(s, (host, port))
        return True
    except Exception:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


__all__ = [
    "MandatoryTorError",
    "install",
    "is_installed",
    "is_allowed_host",
    "tor_liveness_ok",
    "parse_socks_endpoint",
]
