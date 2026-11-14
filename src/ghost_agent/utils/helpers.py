import logging
import datetime
import os
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import httpx
import subprocess
import platform
import ipaddress
from typing import Any, Dict, List, Optional

import socket
import re

# Hostnames that always resolve to the local machine, regardless of DNS.
_BLOCKED_HOSTNAMES = {"localhost", "ip6-localhost", "ip6-loopback"}


def _ip_is_internal(ip_str: str) -> bool:
    """True if `ip_str` is a loopback / private / link-local / reserved /
    multicast / unspecified address — i.e. an SSRF-relevant internal target
    (covers 127.0.0.0/8, 10/8, 172.16/12, 192.168/16, 169.254/16 incl. the
    169.254.169.254 cloud-metadata endpoint, ::1, fc00::/7, etc.)."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_reserved or ip.is_multicast or ip.is_unspecified
    )


_NUMERIC_HOST_RE = re.compile(r"^(?:0x[0-9a-f]+|\d+)(?:\.(?:0x[0-9a-f]+|\d+)){0,3}$", re.I)


def canonical_host(url: str) -> str:
    """The host libcurl/Tor will actually use (§4NL r2/r3): percent-DECODED,
    IDNA-mapped (fullwidth letters and U+3002/U+FF0E/U+FF61 dots fold to
    ASCII — libidn2 does this before the SOCKS handshake), lowercased, no
    trailing dot. "" when there is none. Every host judgement in this guard
    and in the dark-web routing reads THIS, never the raw text."""
    from urllib.parse import unquote, urlsplit
    try:
        host = urlsplit(str(url or "").strip()).hostname or ""
    except Exception:  # noqa: BLE001
        return ""
    host = unquote(host)
    if not host.isascii():
        # r4: libcurl folds with libidn2 UTS46 (non-transitional, then
        # transitional); Python's `encode("idna")` is IDNA2003 on Unicode 3.2
        # tables and disagrees on ~430 code points (`ₒₙᵢₒₙ`, U+2064, ß).
        # The `idna` package IS UTS46 (it is what httpx uses). Not encodable
        # either way → "" : the guard refuses "no host" and the router says
        # not-onion (fail-closed on our side; libcurl would connect for many
        # UTS46-valid / IDNA2008-disallowed characters — r5 measured ~17k —
        # which is why the guard must run before every libcurl fetch).
        try:
            import idna as _idna
            try:
                host = _idna.encode(host, uts46=True, transitional=False, std3_rules=False).decode("ascii")
            except _idna.IDNAError:
                host = _idna.encode(host, uts46=True, transitional=True, std3_rules=False).decode("ascii")
        except Exception:  # noqa: BLE001
            return ""
    return host.lower().rstrip(".")


def host_is_ambiguous_numeric(host: str) -> bool:
    """A numeric host form libcurl normalises but `ipaddress` rejects —
    `0177.0.0.1`, `0x7f000001`, `2130706433`, `127.1` — all of which reach
    127.0.0.1 (§4NL r3). A valid dotted quad is judged by `_ip_is_internal`."""
    if not _NUMERIC_HOST_RE.match(host or ""):
        return False
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        return True


def url_ssrf_reason(url: str, *, resolve: bool = True) -> Optional[str]:
    """Return None if `url` is a safe EXTERNAL http(s) target, else a short
    human-readable refusal reason.

    The single SSRF guard shared by every tool that fetches an
    LLM-supplied URL (web fetch, file download, vision, browser). It blocks:
      * non-http(s) schemes — file://, gopher://, dict://, … (local-file read)
      * hosts that ARE, or (best-effort) DNS-RESOLVE TO, a loopback /
        private / link-local / reserved address — the SSRF + cloud-metadata
        (169.254.169.254) + internal-service class.
    DNS resolution failures are treated as "allow" so a transient resolver
    problem can't block a legitimate public fetch.
    """
    from urllib.parse import urlparse
    try:
        parsed = urlparse(str(url))
    except Exception:
        return f"invalid URL: {url!r}"
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return (
            f"refused non-http(s) URL (scheme={scheme!r}); only http/https are "
            f"allowed (blocks file://, gopher://, dict://, …)."
        )
    host = canonical_host(url)
    if not host:
        return f"refused URL with no host: {url!r}"
    if host_is_ambiguous_numeric(host):
        return f"refused ambiguous numeric host {host!r} (SSRF guard)."
    if "%" in (parsed.netloc or ""):
        # §4NL r2: libcurl percent-DECODES the host before connecting, so
        # `127%2e0%2e0%2e1`, `localhos%74`, `…%2eonion` passed every text
        # check here and reached the real host. No legitimate host needs it.
        return "refused percent-encoded host (SSRF guard)."
    if host in _BLOCKED_HOSTNAMES:
        return f"refused local host {host!r} (SSRF guard)."
    if _ip_is_internal(host):
        return f"refused internal/loopback/link-local host {host!r} (SSRF guard)."
    if resolve:
        try:
            port = parsed.port or (443 if scheme == "https" else 80)
            infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        except Exception:
            infos = []
        for info in infos:
            ip = info[4][0]
            if _ip_is_internal(ip):
                return (
                    f"refused host {host!r} resolving to internal address {ip} "
                    f"(SSRF guard)."
                )
    return None

def request_new_tor_identity(control_port=9051, password=""):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(("127.0.0.1", control_port))
            if password:
                s.sendall(f'AUTHENTICATE "{password}"\r\n'.encode())
            else:
                s.sendall(b'AUTHENTICATE\r\n')
            
            resp = s.recv(1024).decode()
            if not resp.startswith("250"):
                return False, f"Tor Auth failed: {resp.strip()}"
                
            s.sendall(b'SIGNAL NEWNYM\r\n')
            resp = s.recv(1024).decode()
            if not resp.startswith("250"):
                return False, f"Tor NEWNYM failed: {resp.strip()}"
                
            return True, "Identity renewed successfully"
    except Exception as e:
        # Fallback: bounce the tor SERVICE, since the control port is unusable.
        #
        # ⚠ macOS deliberately does NOT use `brew services restart tor` any
        # more (2026-08-21). Tor here is supervised by a system LaunchDaemon
        # labelled `com.local.tor`, NOT by Homebrew — that label was chosen
        # precisely because Homebrew reinstalls a RIVAL job under its own
        # label. Shelling out to brew therefore did three bad things at once:
        #
        #   1. it created ~/Library/LaunchAgents/homebrew.mxcl.tor.plist, a
        #      second tor job that cannot bind :9050 and sits in `error`
        #      state fighting the real daemon (observed live);
        #   2. it did NOT restart the tor that is actually running, so no
        #      identity was renewed;
        #   3. it returned True anyway — a lying success. Callers cannot tell
        #      a real rotation from a no-op, which is worse than failing.
        #
        # Restarting a SYSTEM daemon needs root, and the agent runs as an
        # unprivileged user, so this will usually fail — and it should then
        # say so rather than fabricate a renewal. That is fine: the live
        # rotation path is per-circuit SOCKS isolation
        # (`socks_url_with_identity`), which needs neither the control port
        # nor a daemon bounce and has none of the blast radius. See the 503
        # handling in `fetch_via_tor` below.
        label = os.environ.get("GHOST_TOR_SERVICE_LABEL", "com.local.tor")
        try:
            if platform.system() == "Darwin":
                subprocess.run(
                    ["launchctl", "kickstart", "-k", f"system/{label}"],
                    check=True, capture_output=True, timeout=30,
                )
                return True, f"Identity renewed via launchctl kickstart system/{label}"
            else:
                # Linux systemd (needs sudo NOPASSWD or root; worth trying).
                subprocess.run(["sudo", "-n", "systemctl", "restart", "tor"],
                               check=True, capture_output=True, timeout=30)
                return True, "Identity renewed successfully via systemctl restart"
        except Exception as fallback_e:
            return False, (
                f"Tor control port error: {e}. Service restart also failed: "
                f"{fallback_e}. (Rotation still works: per-circuit SOCKS "
                f"isolation does not need either.)"
            )


def socks_url_with_identity(tor_proxy: Optional[str], identity: str) -> Optional[str]:
    """Inject a SOCKS ``username:password`` into a SOCKS URL so Tor's
    ``IsolateSOCKSAuth`` (on by default) assigns a SEPARATE CIRCUIT per
    ``identity`` tag — circuit-per-identity isolation without the global
    NEWNYM thrash, and without needing the control port.

    Distinct identities → distinct circuits → a colluding set of exit
    nodes / sites can't trivially link a sequence of differently-tagged
    requests into one session. Returns ``tor_proxy`` unchanged when it is
    falsy, has no parseable host, or already carries credentials.
    """
    if not tor_proxy or not identity:
        return tor_proxy
    import re as _re
    from urllib.parse import urlparse, urlunparse
    try:
        p = urlparse(tor_proxy)
        if not p.hostname or p.username:
            return tor_proxy  # unparseable or already has creds
        tag = _re.sub(r"[^A-Za-z0-9]", "", str(identity))[:32] or "ghost"
        _host = f"[{p.hostname}]" if ":" in p.hostname else p.hostname     # §4PI: IPv6 keeps its brackets
        netloc = f"{tag}:isolate@{_host}:{p.port or 9050}"
        return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))
    except Exception:
        return tor_proxy

def circuit_for_attempt(proxy: Optional[str], attempt: int) -> Optional[str]:
    """The SOCKS proxy for retry ``attempt`` of one fetch, on a FRESH circuit. §4UN-read r2: a proxy that
    already carries an identity (web_search's page reads pass one) came back unchanged from
    `socks_url_with_identity`, so its "retrying on a fresh circuit" rode the same exit; here the attempt
    is folded into that identity instead. Attempt 0 is the proxy as given."""
    if not proxy or attempt <= 0:
        return proxy
    try:
        from urllib.parse import urlparse, urlunparse
        p = urlparse(proxy)
        if not p.hostname:
            return proxy
        if not p.username:
            return socks_url_with_identity(proxy, f"fetch{attempt}")
        _host = f"[{p.hostname}]" if ":" in p.hostname else p.hostname
        netloc = f"{p.username}r{attempt}:{p.password or 'isolate'}@{_host}:{p.port or 9050}"
        return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))
    except Exception:  # noqa: BLE001
        return proxy


async def aclose_curl_response(resp) -> None:
    """Abort + reap a curl_cffi ASYNC streaming response.

    The sync ``close()`` is a silent no-op on these: it reaps ``stream_task``,
    which AsyncSession responses never set (they set ``astream_task``).
    Setting ``quit_now`` first makes the curl write callback abort the
    transfer, so ``aclose()`` isn't left awaiting a body we decided to drop.
    Safe after a completed stream (awaiting a finished task is a no-op).
    Shared by ``helper_fetch_url_content`` and the file_system download tool.
    """
    try:
        if getattr(resp, "quit_now", None) is not None:
            resp.quit_now.set()
        await resp.aclose()
    except Exception:
        pass


class FetchError(str):
    """§4PD: a fetch FAILURE from `helper_fetch_url_content` — still the
    "Error: …" string every reader shows, but a TYPE the research path can
    test, so a page whose first words are "Error handling best practices"
    is never booked as a failed fetch."""


class PageText(str):
    """§4UN-deep: the text of a fetched page, with what the page says about itself in ``meta``
    ({"title", "published", "modified", "date"}, ISO dates, "" when not stated; a PDF has ``"pdf": "1"``).
    Returned only to a caller that asks (``with_meta=True``); it is still the plain page text to every reader."""
    meta: dict = {}


#: §4UN-deep r2: PDFs parse one at a time on their OWN thread — never in the default executor, which every
#: `asyncio.to_thread` in the agent shares: a hung MuPDF parse, with reads queued behind a lock there, held
#: those shared workers. Queued here, a read that is cancelled before it starts is dropped.
#: r3: the running parse's start time is kept with its executor. A parse busy longer than PDF_HUNG_AFTER_S
#: when the next PDF arrives has hung (MuPDF cannot be interrupted): that executor is abandoned without
#: waiting — its thread is leaked — and a fresh one takes the read. After PDF_MAX_REPLACEMENTS such
#: replacements in the process, a hung reader fails the read at once instead (threads are not free).
_PDF_STATE: Dict[str, Any] = {"cur": {"ex": ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf"),
                                      "since": None},
                              "replaced": 0}
PDF_HUNG_AFTER_S = 60.0
PDF_MAX_REPLACEMENTS = 3
#: the longest a fetch waits for its PDF to be parsed (queue + parse) — inside deep_research's 22 s fetch
#: attempt, so the error names the parse and does not read as a network timeout
PDF_PARSE_TIMEOUT_S = 15.0
_pdf_clock = time.monotonic


#: the parser the reader calls — None is `content_check.pdf_text` (injectable: one name to replace, not a
#: module attribute looked up through an import that may resolve to another module object)
_PDF_PARSER = None


def _pdf_job(cur: Dict[str, Any], data: bytes) -> str:
    from . import content_check as _cc
    parse = _PDF_PARSER or _cc.pdf_text
    cur["since"] = _pdf_clock()
    try:
        return parse(data)
    finally:
        cur["since"] = None


async def _read_pdf(data: bytes, url: str):
    """The PDF's text (``content_check.pdf_text``: its first pages, capped in chars), or a FetchError when
    the parse does not finish within PDF_PARSE_TIMEOUT_S or the reader is unavailable."""
    from .aio import wait_for as _wait_for
    cur = _PDF_STATE["cur"]
    since = cur.get("since")
    if since is not None and _pdf_clock() - since > PDF_HUNG_AFTER_S:
        if _PDF_STATE["replaced"] >= PDF_MAX_REPLACEMENTS:
            return FetchError(f"Error: PDF reader unavailable (a previous parse hung) — {url} was not read.")
        cur["ex"].shutdown(wait=False, cancel_futures=True)
        cur = {"ex": ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf"), "since": None}
        _PDF_STATE["cur"] = cur
        _PDF_STATE["replaced"] += 1
        logger.warning("PDF parse hung for over %.0fs — reader replaced (%d of %d)",
                       PDF_HUNG_AFTER_S, _PDF_STATE["replaced"], PDF_MAX_REPLACEMENTS)
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    job = cur["ex"].submit(_pdf_job, cur, bytes(data))

    def _settle(f) -> None:
        if fut.done():
            return
        if f.cancelled():          # dropped by a replaced reader: a failed read, never the caller's cancel
            fut.set_result(FetchError(f"Error: PDF reader was replaced while {url} waited; not read."))
        elif f.exception() is not None:
            fut.set_exception(f.exception())
        else:
            fut.set_result(f.result())
    def _back(f) -> None:
        try:
            loop.call_soon_threadsafe(_settle, f)
        except RuntimeError:       # the loop is gone (a hung parse finished after its caller's loop closed)
            pass
    job.add_done_callback(_back)
    try:
        return await _wait_for(fut, PDF_PARSE_TIMEOUT_S)
    except asyncio.TimeoutError:
        return FetchError(f"Error: PDF parse timed out after {PDF_PARSE_TIMEOUT_S:.0f}s for {url}.")
    finally:
        job.cancel()               # a read that never started is dropped (a running parse cannot be stopped)


def _page_text(text: str, meta: Optional[dict]) -> "PageText":
    out = PageText(text)
    out.meta = dict(meta or {})
    return out


async def helper_fetch_url_content(
    url: str, *, proxy_override: Optional[str] = None, renew_identity: bool = True,
    with_meta: bool = False, read_pdf: bool = False,
) -> str:
    """Fetch and text-extract a URL over Tor.

    ``with_meta=True`` (§4UN-deep) — return a `PageText` carrying the page's title and dates (its meta
    tags, JSON-LD, <time>, else the Last-Modified header). Nothing is fetched twice for it.

    ``read_pdf=True`` (§4UN-deep) — a PDF (by content type, or a ``.pdf`` path served as one) of at most
    5 MB is read over the same connection and its text returned; other binary files are still refused.
    Without it a PDF is refused as before (the default contract is unchanged).

    ``proxy_override`` — use this SOCKS proxy instead of the env default. The
    deep-research fetch passes its per-query identity-tagged proxy so the
    fetch rides the same isolated circuit as the search, instead of the bare
    env proxy (which dropped the per-query circuit isolation).

    ``renew_identity=False`` — do NOT fire a global Tor NEWNYM on 503 / error.
    A global identity renewal from one fetch re-circuits Tor for every
    concurrently-running sibling fetch (deep_research runs several at once),
    sabotaging them; callers that fan out should pass False.
    """
    # 1. Setup Tor Proxy (computed BEFORE the SSRF guard so the guard knows
    # whether to do a host-side DNS lookup — see below).
    proxy_url = proxy_override or os.getenv("TOR_PROXY", "socks5://127.0.0.1:9050")
    if proxy_url and proxy_url.startswith("socks5://"):
        proxy_url = proxy_url.replace("socks5://", "socks5h://")

    # --- URL VALIDATION (shared SSRF guard) ---
    # Reject non-http(s) schemes and any host that is/resolves to an
    # internal address, so an LLM tool call can't fetch `file:///etc/passwd`,
    # hit cloud-metadata (169.254.169.254), or reach internal services.
    #
    # resolve=False when fetching over Tor: the default resolve=True does a
    # HOST-SIDE getaddrinfo of the target hostname, which LEAKS the DNS query
    # for the very site we're about to visit anonymously — defeating the
    # DNS-over-SOCKS anonymity this fetch exists to provide (and, for a
    # .onion, leaking which hidden service is being visited). Tor routes/
    # resolves at the exit node anyway, so a host lookup buys no protection
    # here; literal-IP internal targets are still blocked without resolving.
    # Mirrors the browser/download tools' `resolve=not anonymous` handling.
    _ssrf = url_ssrf_reason(url, resolve=not bool(proxy_url))
    if _ssrf:
        return FetchError(f"Error: {_ssrf}")
    try:
        import curl_cffi.requests
    except ImportError:
        curl_cffi = None

    # Hard body-size cap. A malicious / misconfigured server can stream
    # multi-GB content under any timeout; we count bytes ourselves. The
    # comment used to say "we count bytes ourselves" while actually reading
    # resp.text (the WHOLE body into RAM) first — now we truly stream and
    # stop at the cap, like the darkweb sibling (_fetch_raw_html).
    MAX_BODY_BYTES = 5 * 1024 * 1024  # 5 MB ceiling
    _STREAM_LIMIT = MAX_BODY_BYTES + 4096
    _base_proxy = proxy_url

    # §4UN-fetch: a 403/429 gets ONE retry with a plain curl identity on a fresh circuit (Cloudflare and
    # Wikipedia refused the browser fingerprint from Tor exits and served a plain curl — §4FR, §4QF); a 429
    # or 503 rotates the circuit. The browser profile is curl_cffi's current Chrome target with ITS OWN
    # headers (a Chrome/120 UA on a chrome110 TLS fingerprint disagreed with itself).
    _plain = False
    _refused = 0
    for attempt in range(3):
        try:
            headers = {"User-Agent": _PLAIN_FETCH_UA} if _plain else None

            # Rotate the CIRCUIT per attempt via SOCKS-auth isolation (no
            # control port, no daemon restart) so a 503-blocked exit is
            # swapped on retry — see the 503 branch below.
            _attempt_proxy = circuit_for_attempt(_base_proxy, attempt) or _base_proxy

            # Read the response HEADERS first (available on a streaming
            # response before the body), decide whether we even want the
            # body, and only then drain it with a byte cap — a rejected
            # PDF/oversized/non-200 page is never downloaded.
            from urllib.parse import urlparse
            url_path = urlparse(url).path.lower()

            async def _drain_curl(resp):
                # AsyncSession responses stream through an asyncio.Queue and
                # MUST be drained with aiter_content(); the sync
                # iter_content() on the same object returns unawaited
                # Queue.get() coroutines instead of bytes (every fetch died
                # with "can't extend bytearray with coroutine").
                buf = bytearray()
                try:
                    async for chunk in resp.aiter_content():
                        if chunk:
                            buf.extend(chunk)
                            if len(buf) >= _STREAM_LIMIT:
                                break
                finally:
                    await aclose_curl_response(resp)
                return bytes(buf)

            async def _drain_async(resp):
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) >= _STREAM_LIMIT:
                        break
                return bytes(buf)

            def _classify(status_code, content_type, _cl):
                # §4UN-deep: a reader that asked for PDFs drains a PDF-looking body and decides by its bytes
                _pdf_ok = bool(read_pdf) and status_code == 200 and (
                    "application/pdf" in content_type
                    or (url_path.endswith(".pdf") and ("octet-stream" in content_type
                                                       or "html" in content_type or not content_type)))
                is_binary = ("application/pdf" in content_type
                             or "application/octet-stream" in content_type
                             or (isinstance(content_type, str)
                                 and content_type.strip().startswith(("image/", "video/", "audio/", "font/")))  # §4PD
                             or url_path.endswith((".pdf", ".zip")))
                try:
                    clen = int(_cl or "0")
                except (TypeError, ValueError):
                    clen = 0
                oversize = clen and clen > MAX_BODY_BYTES
                # Only drain the body for a 200 we intend to parse.
                want = (status_code == 200 and (not is_binary or _pdf_ok) and not oversize)
                if _pdf_ok and not oversize:
                    is_binary = False            # decided by the bytes after the drain
                return is_binary, oversize, want

            _raw = b""
            _final_url = url
            _last_mod = None
            if curl_cffi:
                proxies = {"http": _attempt_proxy, "https": _attempt_proxy} if _attempt_proxy else None
                async with curl_cffi.requests.AsyncSession(impersonate=(None if _plain else _FETCH_IMPERSONATE),
                                                           proxies=proxies, timeout=20.0) as client:
                    resp = await (client.get(url, headers=headers, stream=True) if headers
                                  else client.get(url, stream=True))
                    content_type = resp.headers.get("content-type", "").lower()
                    status_code = resp.status_code
                    _cl = resp.headers.get("content-length")
                    _last_mod = resp.headers.get("last-modified")
                    _fu = getattr(resp, "url", None)
                    _final_url = _fu if isinstance(_fu, str) and _fu else url
                    _is_binary, _oversize, _want = _classify(status_code, content_type, _cl)
                    if _want:
                        _raw = await _drain_curl(resp)
                    else:
                        await aclose_curl_response(resp)
            else:
                # Fallback to httpx if curl_cffi is missing for some reason
                async with httpx.AsyncClient(proxy=_attempt_proxy, timeout=20.0, follow_redirects=True) as client:
                    async with client.stream("GET", url, headers=headers or {"User-Agent": "Mozilla/5.0"}) as resp:
                        content_type = resp.headers.get("content-type", "").lower()
                        status_code = resp.status_code
                        _cl = resp.headers.get("content-length")
                        _last_mod = resp.headers.get("last-modified")
                        _is_binary, _oversize, _want = _classify(status_code, content_type, _cl)
                        if _want:
                            _raw = await _drain_async(resp)

            # Header-only short-circuits (no body was read for these).
            if _is_binary:
                return FetchError("Error: URL points to a binary file. To read PDFs, download them using file_system and ingest them using knowledge_base.")
            if _oversize:
                _clen = int(_cl or "0")
                return FetchError(f"Error: response from {url} is {_clen // (1024*1024)} MB; refusing to read more than {MAX_BODY_BYTES // (1024*1024)} MB.")
            if status_code != 200:
                # §4UN-fetch: 429 (rate limit per exit) and 503 (exit filtered) rotate the circuit; a 403
                # or 429 also gets the plain-curl identity once (the site refused the FINGERPRINT). 401
                # is the application's own answer — no retry.
                if status_code in (403, 429) and not _plain and attempt < 2 and proxy_url and renew_identity:
                    _plain, _refused = True, status_code
                    _log_fetch_kind(url, f"HTTP {status_code}", "retrying once with a plain curl identity "
                                    "on a fresh circuit")
                    await asyncio.sleep(0.5)
                    continue
                if status_code in (429, 503) and proxy_url and attempt < 2 and renew_identity:
                    # Rotate via a fresh SOCKS-isolated CIRCUIT on the next
                    # attempt (handled at the top of the loop), NOT via
                    # request_new_tor_identity: on this box the control
                    # port (9051) is closed, so that path falls back to
                    # bouncing the WHOLE Tor daemon, which an unprivileged
                    # agent cannot do. Per-circuit isolation gets a fresh exit.
                    _log_fetch_kind(url, f"HTTP {status_code}", "retrying on a fresh circuit")
                    await asyncio.sleep(2)
                    continue
                _log_fetch_kind(url, f"HTTP {status_code}", "gave up")
                if status_code == 503 and proxy_url:
                    return FetchError(f"Error: Access Denied (503) via Tor. The site {url} likely blocks Tor exit nodes. Try a different source.")
                if status_code == 429:
                    return FetchError(f"Error: Rate limited (429) by {url} on every Tor circuit tried"
                                      f"{' and with a plain curl identity' if _plain else ''}. Try a different source.")
                if status_code in (401, 403):
                    _how = (" (a browser and a plain curl identity, on two circuits)" if _plain else "")
                    return FetchError(f"Error: Access Denied ({status_code}) from {url}{_how}. Application-level forbidden — Tor rotation will not help.")
                return FetchError(f"Error: Received status {status_code} from {url}")
            _hdrs = {"last-modified": _last_mod} if isinstance(_last_mod, str) and _last_mod else None
            if read_pdf and bytes(_raw[:5]).startswith(b"%PDF-"):
                # §4UN-deep: a PDF is read whole or not at all (a cut PDF does not parse)
                if len(_raw) > MAX_BODY_BYTES:
                    return FetchError(f"Error: the PDF at {url} is larger than "
                                      f"{MAX_BODY_BYTES // (1024*1024)} MB; refusing to read it.")
                from .content_check import page_meta
                _pdf = await _read_pdf(_raw, url)
                if isinstance(_pdf, FetchError):
                    return _pdf
                if not _pdf.strip():
                    return FetchError(f"Error: the PDF at {url} has no extractable text "
                                      f"(a scanned or password-protected PDF).")
                if not with_meta:
                    return _pdf
                _m = page_meta("", _hdrs)
                _m["pdf"] = "1"
                return _page_text(_pdf, _m)
            if read_pdf and ("application/pdf" in content_type or "octet-stream" in content_type):
                # asked for as a PDF and it is not one: some other binary file
                return FetchError("Error: URL points to a binary file. To read PDFs, download them using "
                                  "file_system and ingest them using knowledge_base.")
            # 200 — decode the drained body and cap. §4UN-fetch: with no charset in the header the BYTES go to
            # the parser, which honours the page's own <meta charset> (windows-1253 / iso-8859-7 Greek pages
            # came out as mojibake when decoded as utf-8).
            _charset = page_charset(content_type, _raw)
            try:
                _parse_input = _raw.decode(_charset, errors="replace")
            except (LookupError, TypeError):
                _parse_input = _raw.decode("utf-8", errors="replace")
            if len(_parse_input) > MAX_BODY_BYTES:
                _parse_input = _parse_input[:MAX_BODY_BYTES] + "\n[... TRUNCATED at 5 MB ceiling ...]"

            def _parse_html(html_content):
                from bs4 import BeautifulSoup
                from .content_check import challenge_reason, extract_main_text
                text_content = extract_main_text(html_content)           # §4UN-fetch: menus/banners/forms out
                try:
                    _t = BeautifulSoup(html_content, "html.parser").title
                    _title = _t.get_text(" ", strip=True) if _t is not None else ""
                except Exception:  # noqa: BLE001
                    _title = ""
                _wall = challenge_reason(200, _title, text_content or "", _final_url)
                if _wall:
                    return FetchError(f"Error: BLOCKED — {url} answered with {_wall}, not the page's content. "
                                      f"Nothing was read; do not cite it. Use a different source.")
                if not text_content:
                    return FetchError("Error: No text content extracted from page.")   # §4PE r3: a failure is a TYPE
                if with_meta:
                    from .content_check import page_meta
                    try:
                        _m = page_meta(html_content, _hdrs)
                    except Exception:  # noqa: BLE001 — the meta never costs the page
                        _m = {}
                    if _title and not (_m or {}).get("title"):
                        _m["title"] = " ".join(_title.split())[:200]
                    return _page_text(text_content, _m)
                return text_content

            _parsed = await asyncio.to_thread(_parse_html, _parse_input)
            if isinstance(_parsed, FetchError) and _parsed.startswith("Error: BLOCKED"):
                _log_fetch_kind(url, "challenge page", "not read")
            return _parsed

        except Exception as e:
            if attempt < 2 and proxy_url and renew_identity:
                # Retry on a fresh circuit (rotated at the top of the loop
                # via socks_url_with_identity), NOT a global NEWNYM/daemon
                # restart — see the 503 branch above.
                _log_fetch_kind(url, type(e).__name__, "retrying on a fresh circuit")
                await asyncio.sleep(2)
                continue
            _log_fetch_kind(url, type(e).__name__, "gave up")
            return FetchError(f"Error reading {url}: {str(e)}")
    return FetchError(f"Error fetching {url} after 3 retries.")


_META_CHARSET_RE = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.I)


def page_charset(content_type: Optional[str], raw: bytes) -> str:
    """§4UN-fetch: the charset to decode a page with — a byte-order mark first, then the header's charset, then
    the page's own <meta charset> in its first 4 KB, else utf-8 (never a statistical guess: a guessed charset
    mangles a utf-8 page). r1: a quoted header charset is unquoted, and a page whose bytes decode cleanly as
    UTF-8 is UTF-8 whatever its meta tag claims (CMS templates keep a stale `iso-8859-1` meta)."""
    import codecs
    b = bytes(raw or b"")
    if b.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
        return "utf-16"

    def _valid(cs: str) -> str:
        try:
            codecs.lookup(cs)
            return cs
        except LookupError:
            return ""
    m = re.search(r"""charset\s*=\s*["']?\s*([\w.:-]+)""", str(content_type or ""), re.I)
    if m and _valid(m.group(1)):
        return m.group(1)
    mm = _META_CHARSET_RE.search(b[:4096])
    if mm:
        cs = _valid(mm.group(1).decode("ascii", "ignore"))
        if cs and codecs.lookup(cs).name != "utf-8":
            try:
                if not b.isascii():
                    codecs.getincrementaldecoder("utf-8")().decode(b, final=False)
                    return "utf-8"              # the bytes ARE utf-8 (a cut final character is allowed)
            except UnicodeDecodeError:
                pass
        if cs:
            return cs
    return "utf-8"


#: §4UN-fetch: the browser-profile target — curl_cffi's current Chrome (its own UA and client hints)
_FETCH_IMPERSONATE = "chrome"
#: the plain identity a refused fetch retries with (§4FR / §4QF: served where the browser profile was refused)
_PLAIN_FETCH_UA = "curl/8.7.1"


def _log_fetch_kind(url: str, kind: str, what: str) -> None:
    """§4UN-fetch: one INFO line per fetch failure, by kind — the logs could not say which sites refuse."""
    try:
        from urllib.parse import urlparse
        from .logging import Icons, pretty_log
        host = (urlparse(str(url)).hostname or "?")[:60]
        pretty_log("Fetch Failure", f"{host}: {kind} — {what}", level="INFO", icon=Icons.WARN)
    except Exception:  # noqa: BLE001
        pass
# --------------------------------------------------------------------------
# Removal / negation detection
# --------------------------------------------------------------------------
# Phrases that mark a fact as a REMOVAL or NON-OWNERSHIP statement rather than
# a positive thing to remember. Consolidation must NOT store these: doing so
# manufactures self-perpetuating tombstones — e.g. asking "what pets do I
# own?" makes the model answer "you previously had an iguana that was
# removed", which then gets consolidated back into a fresh "user previously
# had an iguana" fact, keeping the deleted entity alive forever. A removal
# should DELETE knowledge, never insert a new memory about the deletion.
_REMOVAL_NEGATION_PHRASES = (
    "no longer", "previously had", "previously owned", "used to have",
    "used to own", "was removed", "were removed", "has been removed",
    "have been removed", "had been removed", "removed from", "deleted from",
    "does not own", "doesn't own", "do not own", "don't own",
    "does not have", "doesn't have", "do not have", "don't have",
    "did not have", "didn't have", "never had", "never owned",
    "no longer has", "no longer have", "no longer owns", "no longer own",
    "got rid of", "is gone", "are gone", "is no more",
    "not own", "not have an", "not have a",
    # §4KY: "The user is not a doctor" was stored as an identity fact
    "is not a ", "is not an ", "isn't a ", "isn't an ", "am not a ", "am not an ", "is no ", "not a doctor",
    "is not the ", "isn't the ",
    # §4KZ: selling / giving away retracts ownership ("we sold the BMW" left
    # OWNS, DRIVES and HAS_CAR live and added `user SOLD bmw`)
    "sold the ", "sold my ", "sold our ", "sold his ", "sold her ", "gave away", "πούλησα", "πουλήσαμε",
    "πουλησα", "πουλησαμε",
    # Parenthetical tombstone markers an earlier soft-delete left behind,
    # e.g. "Mortimer the iguana (removed)".
    "(removed)", "[removed]", "(deleted)", "[deleted]", "(former)",
)

#: Graph-predicate tokens that encode removal/past-tense semantics.
#: Predicates are uppercase verbs (``PREVIOUSLY_OWNED``, ``NO_LONGER_HAS``,
#: ``REMOVED``). §4M R2 MINOR-3: these were matched as raw SUBSTRINGS, so
#: legitimate predicates false-positived through embedded fragments —
#: ``PERFORMER``/``USES_TRANSFORMERS`` (FORMER), ``WHENEVER_RUNS`` (NEVER).
#: Now matched on ``_``-token boundaries; the multi-token phrases below
#: are checked as substrings still (no embedding risk measured for them).
_REMOVAL_PREDICATE_TOKENS = frozenset({
    "PREVIOUSLY", "FORMER", "REMOVED", "NEVER", "DELETED", "NOT", "PAST",
    "SOLD", "SELLS", "GAVE", "LOST",          # §4KZ: a sale retracts ownership
})
_REMOVAL_PREDICATE_PHRASES = ("NO_LONGER", "NOLONGER", "USED_TO")


def is_removal_or_negation_text(text) -> bool:
    """True iff ``text`` reads as a removal / non-ownership statement.

    Used to stop the memory consolidator from re-storing "X was removed"
    tombstones (see `_REMOVAL_NEGATION_PHRASES`)."""
    if not text:
        return False
    t = " ".join(str(text).lower().split())
    return any(p in t for p in _REMOVAL_NEGATION_PHRASES)


def is_removal_triplet(triplet) -> bool:
    """True iff a graph triplet encodes removal / past-ownership and so must
    not be ingested. Checks the predicate for tombstone verbs and the
    subject/object text for removal phrasing."""
    if not isinstance(triplet, dict):
        return False
    pred = str(triplet.get("predicate", "")).upper()
    if _REMOVAL_PREDICATE_TOKENS & set(pred.split("_")):
        return True
    if any(phrase in pred for phrase in _REMOVAL_PREDICATE_PHRASES):
        return True
    for field in ("subject", "object"):
        if is_removal_or_negation_text(triplet.get(field, "")):
            return True
    return False


def get_utc_timestamp():
    """Returns strict ISO8601 UTC timestamp for Go/iOS clients.

    The `Z` suffix is intentional — Go and iOS Decoder strict-mode requires
    it. Python consumers that want to round-trip the value should use
    `parse_utc_timestamp()` below rather than doing their own `strptime`,
    which avoids the trap of forgetting to strip the `Z`.
    """
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_utc_timestamp(s: str) -> datetime.datetime:
    """Robust parser for the strings produced by `get_utc_timestamp()`.

    Accepts the canonical ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` form, its Z-less
    variant, and ``+00:00``-suffixed variants. Raises `ValueError` on
    genuinely malformed input. Use this helper wherever you parse a
    Ghost-emitted timestamp so a future format change lands in one place.
    """
    if not isinstance(s, str):
        raise ValueError(f"timestamp must be a str, got {type(s).__name__}")
    stripped = s.strip()
    if not stripped:
        raise ValueError("empty timestamp")
    # Normalise the common Z-suffix to the +00:00 form that
    # `datetime.fromisoformat` understands on every supported Python.
    if stripped.endswith("Z"):
        stripped = stripped[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(stripped)
    except ValueError:
        # Fallback for the legacy `%f` strftime shape (some callers produce
        # exactly 6-digit microseconds, which fromisoformat handles, but we
        # keep this branch in case of edge cases like `.%f` without tz).
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.datetime.strptime(stripped, fmt)
                break
            except ValueError:
                continue
        else:
            raise
    # Always return a naive UTC datetime so arithmetic like
    # `(now - mem_time).total_seconds()` works without tz mixing.
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return dt

def recursive_split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 70) -> List[str]:
    if not text: return []
    # The range() loops below step by `chunk_size - chunk_overlap`. When
    # chunk_overlap >= chunk_size that step is <= 0, so range() yields
    # NOTHING and the text is silently DROPPED (a 0 step raises
    # ValueError). Clamp so the step is always >= 1. Reachable from
    # semantic_split_text when a long header shrinks the effective
    # chunk_size below the overlap.
    chunk_size = max(1, int(chunk_size))
    # KB review 10-30: capped at a quarter of the chunk, not chunk_size-1 — a
    # long header shrinks the effective size below the overlap, and the
    # hard-split step collapsed to ONE character: 20 KB of base64 under a
    # 510-char header became 20,001 chunks / 12 M chars (≈ 110 s of embedding
    # under the store lock; ≈ 1.5 h for 1 MB). Every step now advances ≥ 3/4.
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size // 4))
    if len(text) <= chunk_size: return [text]

    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    final_chunks = []
    # (text, first separator index to try). An oversize piece goes back on
    # the stack with the NEXT, finer separator — never re-detected from the
    # top (§4LT CRIT: "Sentence. " + 600 chars without ". " split into the
    # same two pieces forever, freezing the event loop on one URL ingest).
    # The index only grows, so this always terminates.
    stack = [(text, 0)]

    while stack:
        current_text, _sep_from = stack.pop()

        if len(current_text) <= chunk_size:
            final_chunks.append(current_text)
            continue

        found_sep = ""
        _found_at = len(separators) - 1
        for _k in range(_sep_from, len(separators)):
            sep = separators[_k]
            if sep in current_text:
                found_sep = sep
                _found_at = _k
                break

        if not found_sep:
            for i in range(0, len(current_text), chunk_size - chunk_overlap):
                final_chunks.append(current_text[i:i+chunk_size])
            continue

        parts = current_text.split(found_sep)
        buffer = ""
        temp_chunks = []

        for p in parts:
            # Re-attach the separator to EVERY fragment. It used to be
            # re-attached only for punctuation separators ('. ', '? ', …);
            # for whitespace ones ('\n\n', '\n', ' ') `found_sep.strip()`
            # is falsy, so the separator was dropped on rejoin — fusing the
            # last word of each line to the first word of the next in every
            # chunk (live: all 7,131 postgresql-manual chunks), corrupting
            # embeddings, BM25 token matching, and the passage text itself.
            # The final fragment gains one trailing separator that wasn't in
            # the source; the edge .strip() below removes it for whitespace
            # and it is harmless for punctuation.
            fragment = p + found_sep
            if len(buffer) + len(fragment) <= chunk_size:
                buffer += fragment
            else:
                if buffer:
                    temp_chunks.append(buffer.strip())
                    # Honor chunk_overlap on the separator path too — it was
                    # only ever applied on the no-separator hard split, so
                    # separator-path chunks had zero overlap despite the
                    # documented contract. Carry the flushed buffer's tail
                    # into the next chunk so cross-boundary context survives
                    # retrieval. The carry is capped at chunk_size//4:
                    # callers CAN arrive with overlap clamped to
                    # chunk_size-1 (semantic_split_text shrinks the
                    # effective chunk_size below the overlap for long
                    # headers), and an uncapped carry then advances ONE
                    # fragment per chunk — quadratic output blowup that
                    # spun the test suite for 36 CPU-minutes. The cap
                    # guarantees ≥ 3/4 forward progress per chunk.
                    _carry = min(chunk_overlap, chunk_size // 4)
                    buffer = buffer[-_carry:] if _carry > 0 else ""
                buffer += fragment

        if buffer:
            temp_chunks.append(buffer.strip())

        # Push everything back through the stack (reversed, so LIFO pops in
        # document order) — appending fitting chunks to final_chunks here
        # while iterating reversed would emit them backwards.
        for chunk in reversed(temp_chunks):
            if len(chunk) > chunk_size and found_sep == "":
                # Can't be reduced by separators — hard character split.
                pieces = [chunk[i:i+chunk_size]
                          for i in range(0, len(chunk), chunk_size - chunk_overlap)]
                stack.extend((pc, len(separators) - 1) for pc in reversed(pieces))
            elif len(chunk) > chunk_size:
                stack.append((chunk, _found_at + 1))   # finer separators only
            else:
                stack.append((chunk, 0))

    return final_chunks


def semantic_split_text(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> List[str]:
    """Structure-aware chunking that respects document semantics.

    Unlike ``recursive_split_text`` which splits purely by character count,
    this function:
    1. Detects document structure (markdown headers, code blocks, list items)
    2. Groups content by section when possible
    3. Never splits mid-sentence or mid-code-block
    4. Prepends section headers to each chunk for retrieval context

    Falls back to ``recursive_split_text`` for unstructured content.
    """
    import re

    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # Detect if content has markdown structure
    has_headers = bool(re.search(r'^#{1,6}\s', text, re.MULTILINE))
    has_code_blocks = '```' in text

    if not has_headers and not has_code_blocks:
        return recursive_split_text(text, chunk_size, chunk_overlap)

    # Split by sections (markdown headers)
    sections = []
    if has_headers:
        # Split on headers, keeping the header with its content
        parts = re.split(r'(^#{1,6}\s[^\n]*\n)', text, flags=re.MULTILINE)
        current_section = ""
        current_header = ""
        for part in parts:
            if re.match(r'^#{1,6}\s', part):
                if current_section.strip():
                    sections.append((current_header, current_section.strip()))
                    current_section = ""
                elif current_header:
                    # KB review 10-30 r3: a header with no body of its own
                    # (stacked "# " comments, an H1 straight above an H2) was
                    # in no chunk — its text rides the next section's body
                    current_section = current_header + "\n"
                current_header = part.strip()
            else:
                current_section += part
        if current_section.strip():
            sections.append((current_header, current_section.strip()))
        elif current_header:
            sections.append((current_header, current_header))
    else:
        sections = [("", text)]

    # Process each section
    chunks = []
    for header, section_text in sections:
        # Prepend header context to each chunk from this section
        # KB review 10-30: a header line can be any length (a whole paragraph
        # after "# "); its prefix rides EVERY chunk of the section
        header_prefix = f"[Section: {header[:120]}] " if header else ""
        if len(header) > 120:
            # r2: the line exists only as this prefix — cut, its tail (a
            # long `#` comment in a script or config) was in no chunk
            section_text = header + "\n" + section_text
        prefix_len = len(header_prefix)
        effective_size = chunk_size - prefix_len

        if effective_size <= 50:
            effective_size = chunk_size

        if len(section_text) <= effective_size:
            chunks.append(f"{header_prefix}{section_text}" if header_prefix else section_text)
            continue

        # For code blocks, try to keep them intact
        if has_code_blocks and '```' in section_text:
            code_parts = re.split(r'(```[^\n]*\n.*?```)', section_text, flags=re.DOTALL)
            for part in code_parts:
                if part.startswith('```'):
                    # Code block — keep intact up to 2x chunk_size
                    if len(part) <= chunk_size * 2:
                        chunks.append(f"{header_prefix}{part}" if header_prefix else part)
                    else:
                        # Very long code block — split by lines
                        sub_chunks = recursive_split_text(part, effective_size, chunk_overlap)
                        for sc in sub_chunks:
                            chunks.append(f"{header_prefix}{sc}" if header_prefix else sc)
                else:
                    # Prose between code blocks
                    if part.strip():
                        sub_chunks = recursive_split_text(part.strip(), effective_size, chunk_overlap)
                        for sc in sub_chunks:
                            chunks.append(f"{header_prefix}{sc}" if header_prefix else sc)
        else:
            # No code blocks in this section — use recursive split
            sub_chunks = recursive_split_text(section_text, effective_size, chunk_overlap)
            for sc in sub_chunks:
                chunks.append(f"{header_prefix}{sc}" if header_prefix else sc)

    return [c for c in chunks if c.strip()]


logger = logging.getLogger("GhostAgent")   # §4NV: __name__ reached neither the stream nor the durable log


def env_positive(name: str, default: float) -> float:
    """A positive float from the environment, or the default. Never raises.

    ⚠ Three separate traps, all live before this existed:
      * `float(os.getenv(X, "90") or 90)` — `"0"` is TRUTHY, so `or 90` never
        fires. `GHOST_NODE_SLOT_WAIT_S=0` gave every node a 0-second permit
        wait: no off-main node was ever contacted, every pool call reported
        SATURATED, and all traffic landed on the single main slot.
      * a typo'd value (`"90s"`) raised ValueError at the TOP of
        `_do_chat_completion` — killing every call, including main-only ones
        with no slot involved.
      * "<=0 disables" is this codebase's convention elsewhere, so the
        obvious operator reading produced the worst outcome.
    (LLM review R3 lens A, MAJOR-2 — the same trap `_main_fallback_timeout_s`
    fixed one round earlier, at a different site.)"""
    raw = os.environ.get(name)
    try:
        val = float(raw) if raw not in (None, "") else float(default)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not a number — using %s", name, raw, default)
        return float(default)
    if val <= 0:
        logger.warning("%s=%r is not positive — using %s", name, raw, default)
        return float(default)
    return val


def env_flag(name: str, default: bool = False) -> bool:
    """§4ND: an operator on/off switch, read per call. Unset → `default`."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")
