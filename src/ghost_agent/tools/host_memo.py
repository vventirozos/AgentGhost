"""§4LG: memory of CLEARNET sites that keep failing the same way.

Measured over 3 weeks: 65 fetches of a host that had already failed in the
last 24 h, 60 of them failed again (one host: 19 attempts across 12 requests,
all HTTP/2 errors). Two strikes of a repeatable kind (HTTP/2 protocol error,
timeout, 403 / bot challenge) inside the TTL and the site is skipped with its
cause; a clean load clears it. ``GHOST_DEAD_HOST_MEMO=0`` disables it. The
onion twin (a Tor-layer failure means the service is offline) stays in
``browser``.
"""
import os
import re
import time
from typing import Dict
from urllib.parse import urlparse as _urlparse


def _cap(d: Dict[str, list], key_time, limit: int = 512) -> None:
    """Bound the map: drop the oldest entries past ``limit``."""
    if len(d) > limit:
        for k in sorted(d, key=key_time)[: len(d) - limit]:
            d.pop(k, None)


_HOST_FAILS: Dict[str, list] = {}
_HOST_FAIL_TTL = 6 * 3600.0
#: a timeout is often OUR side (a Tor stall, a slow circuit) — remembered
#: for an hour, not six
_HOST_TIMEOUT_TTL = 3600.0
_HOST_FAIL_STRIKES = 2
_HOST_FAIL_MARKERS = ("http2", "timeout", "timed out", "timed_out")


def _clearnet_host(url: str) -> str:
    """``host[:port]`` of a public clearnet URL, or "" — never an onion, and
    never a loopback / private / .local host (§4LH final review: two slow
    starts of a local dev server banned every local port for 6 h)."""
    try:
        p = _urlparse(str(url or ""))
        host = (p.hostname or "").lower()
        port = p.port
    except Exception:  # noqa: BLE001
        return ""
    if not host or host.endswith(".onion") or host.endswith(".local") or host == "localhost":
        return ""
    try:
        import ipaddress
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return ""
    except ValueError:
        pass
    return f"{host}:{port}" if port else host


def _host_memo_on() -> bool:
    return os.environ.get("GHOST_DEAD_HOST_MEMO", "1") == "1"


def host_level_block(reason: str) -> bool:
    """A BLOCKED page that speaks for the whole SITE (a refusal or a bot
    challenge) — not a 404 or a 5xx of one page (§4LH final review: two
    mistyped Wikipedia paths banned en.wikipedia.org)."""
    r = str(reason or "")
    return r.startswith(("HTTP 403", "HTTP 429")) or "bot challenge" in r


def _mark_host_failed(url: str, cause: str) -> None:
    """Record a NAVIGATION failure of a repeatable kind (HTTP/2 protocol
    error, navigation timeout) or a site-level block. A selector timeout
    (`Locator.click: Timeout`) is the page's shape, not the site's health,
    and the URL inside a cause is not read as the cause."""
    host = _clearnet_host(url)
    c = str(cause or "")
    if not (host and _host_memo_on()):
        return
    lc = re.sub(r"https?://\S+", " ", c).lower()
    if c.startswith("blocked:"):
        if not host_level_block(c[len("blocked:"):].strip()):
            return
        kind = "block"
    elif ("page.goto" in lc or "net::err_" in lc) and any(m in lc for m in _HOST_FAIL_MARKERS):
        kind = "timeout" if "http2" not in lc else "http2"
    else:
        return
    ttl = _HOST_TIMEOUT_TTL if kind == "timeout" else _HOST_FAIL_TTL
    now = time.monotonic()
    rec = _HOST_FAILS.get(host)
    if rec is None or (now - rec[1]) >= rec[3]:
        _HOST_FAILS[host] = [1, now, c[:120], ttl]
    else:
        rec[0], rec[1], rec[2], rec[3] = rec[0] + 1, now, c[:120], min(rec[3], ttl)
    _cap(_HOST_FAILS, lambda h: _HOST_FAILS[h][1])


def _mark_host_ok(url: str) -> None:
    host = _clearnet_host(url)
    if host:
        _HOST_FAILS.pop(host, None)


def _dead_host_notice(url: str):
    """A directive message when `url`'s clearnet host failed twice recently."""
    host = _clearnet_host(url)
    if not (host and _host_memo_on()):
        return None
    rec = _HOST_FAILS.get(host)
    if rec is None or rec[0] < _HOST_FAIL_STRIKES or (time.monotonic() - rec[1]) >= rec[3]:
        return None
    return (f"{host} failed {rec[0]} times in the last few hours ({rec[2][:80]}) — fetching it again "
            f"will fail the same way. Use a DIFFERENT source from your results; if none loads, "
            f"answer from what you have and say which pages could not be read.")




# ── onion addresses that cannot load (§4LG, live probe D2) ─────────────────
# Asked for DuckDuckGo's onion, the agent navigated to the 16-character v2
# address (retired by Tor in Oct 2021), got ERR_SOCKS_CONNECTION_FAILED on it
# and on two other one-off onions, and told the owner "Tor's SOCKS proxy is
# failing for all .onion addresses" while presenting the v2 address as
# "official v3". A v2 address is refused before it is dialled, with the
# reason; a first Tor-layer failure says what it means.
_V2_ONION_LABEL_LEN = 16


def _retired_onion_notice(url: str):
    """A directive message when ``url`` is a v2 (16-character) onion address."""
    try:
        host = (_urlparse(str(url or "")).hostname or "").lower()
    except (ValueError, TypeError):
        return None
    if not host.endswith(".onion"):
        return None
    label = host[: -len(".onion")].rsplit(".", 1)[-1]
    if len(label) != _V2_ONION_LABEL_LEN:
        return None
    return (f"{host} is a v2 onion address (16 characters). Tor removed v2 onion services in October "
            f"2021, so it cannot load anywhere and it is NOT a current address — this says nothing about "
            f"Tor working. Current (v3) addresses are 56 characters; an organisation's current one is "
            f"listed on its own clearnet site. Do not report this address as the official one.")


ONION_FIRST_FAILURE_HINT = (
    "This hidden service did not answer over Tor. For an onion address that means the service is offline "
    "or the address does not exist (onion search results and remembered addresses are often dead or "
    "wrong) — it does NOT mean Tor is broken, so do not diagnose or restart Tor. Use a different "
    "address, ideally one taken from the organisation's own clearnet site.")
