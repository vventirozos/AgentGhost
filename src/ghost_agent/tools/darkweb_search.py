"""Dark-web (.onion) search over Tor.

The clearnet search tool (`search.py`) queries scraper backends — DuckDuckGo,
Brave, Mojeek — that DO NOT index Tor hidden services. To discover `.onion`
content you have to query the *dedicated onion search engines* (Ahmia, Torch,
Haystak, …) and parse their result pages. That is what this module does.

Almost all of the hard Tor plumbing already exists and is reused verbatim
from `search.py`:

  * `_proxy_for_attempt` — folds the attempt index into the SOCKS
    ``username:password`` so each retry rides a DISTINCT Tor circuit. Onion
    *search engine* reachability is exit-node dependent in exactly the same
    way clearnet search is (see the long note in `search.py`), so the same
    per-attempt circuit rotation is what beats a transiently blocked engine.
  * `_sanitize_query` — strips Google-style operators the engines choke on.
  * `_clean_for_cpp` — Llama.cpp grammar-safe output cleaning.
  * `_cache_get`/`_cache_put` — the same bounded TTL cache, namespaced with
    an ``onion::`` key prefix so it never collides with clearnet results.

What is NEW here:

  * `_ONION_ENGINES` — the search-engine registry. Onion endpoints rotate
    and die far more often than clearnet ones, so this is config-driven
    (overridable via the ``GHOST_ONION_ENGINES`` env var, JSON) and we keep
    a WIDE set rather than betting on one engine — breadth + circuit
    rotation, the same philosophy as `_TOR_BACKENDS`. Each engine carries an
    ``index`` tag: endpoints that serve the SAME underlying search index
    (e.g. Ahmia's clearnet mirror and its onion endpoint) share one index so
    cross-engine corroboration ranking counts INDEPENDENT sources, not the
    same index reached over two transports.
  * `_fetch_raw_html` — onion search needs the RAW result HTML so we can
    parse out the result links. `helper_fetch_url_content` strips all tags
    to plain text (destroying the links), so it is unusable for the search
    phase; we fetch raw HTML ourselves, honouring the caller's proxy, under a
    hard body-size cap and on a DEDICATED bounded thread pool. The *research*
    phase reuses the same capped/proxied fetch via `_fetch_onion_text` (NOT
    the shared `helper_fetch_url_content`, which ignores the passed proxy and
    can trigger a global Tor NEWNYM / service restart on every failed fetch).
  * `_ONION_TIMEOUT` — onion round-trips are materially slower than
    clearnet-over-Tor, so the per-request ceiling is higher than search.py's
    `_DDGS_TOR_TIMEOUT` (18s). Measured-empirically tunable, like that one.
"""
import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus, urlparse, parse_qs, unquote

from ..utils.logging import Icons, pretty_log
from ..utils.helpers import url_ssrf_reason
from .search import (
    _sanitize_query,
    _proxy_for_attempt,
    _clean_for_cpp,
    _cache_get,
    _cache_put,
)

# --------------------------------------------------------------------------
# Onion address recognition
# --------------------------------------------------------------------------
# Tor hidden-service addresses are base32 (chars a-z, 2-7): v3 is 56 chars,
# the legacy v2 was 16. Match either, optionally followed by a path.
_ONION_RE = re.compile(
    r"https?://(?:[a-z2-7]{56}|[a-z2-7]{16})\.onion(?:/[^\s\"'<>]*)?",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# Onion search-engine registry
# --------------------------------------------------------------------------
# Each entry: {name, url, index}. ``url`` is a template with a single ``{q}``
# placeholder that receives the URL-encoded query. ``index`` names the
# underlying search index; engines that share an index are NOT independent
# corroboration sources (see `_darkweb_search_raw`). We keep a WIDE set:
# onion engines are individually flaky and per-exit-node reachable, so
# breadth + circuit rotation wins. Override the whole set with the
# GHOST_ONION_ENGINES env var (a JSON list of {"name","url"[,"index"]}
# objects) when endpoints rotate.
#
# Ahmia is listed twice on purpose: its clearnet mirror (ahmia.fi, still
# fetched over Tor) and its onion endpoint have independent REACHABILITY, so
# one is often up when the other is blocked. But they serve the SAME index,
# so they share ``index="ahmia"`` — an Ahmia-only hit must not masquerade as
# cross-engine corroboration just because it was reached over both transports.
# Ahmia also filters known-abuse material at the index level — useful as a
# sane primary even in the personal/experimental posture this tool is built
# for.
_DEFAULT_ONION_ENGINES: List[Dict[str, str]] = [
    {"name": "ahmia", "url": "https://ahmia.fi/search/?q={q}", "index": "ahmia"},
    {
        "name": "ahmia-onion",
        "url": "http://juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd.onion/search/?q={q}",
        "index": "ahmia",
    },
    {
        "name": "torch",
        "url": "http://torchdeedp3i2jigzjdmfpn5ttjhthh5wbmda2rr3jvqjg5p77c54dqd.onion/search?query={q}",
        "index": "torch",
    },
    # NOTE: haystak was dropped from the DEFAULT set after live functional
    # testing (2026-06/07): its v3 endpoint failed 0/8 across many rotated
    # circuits (SOCKS5 connect / timeout), i.e. the hidden service itself is
    # down, not an exit-node block — and its retry could burn a full
    # `_ONION_TIMEOUT` on the 2nd attempt, inflating the tail latency of the
    # whole (concurrent) call. Re-add it, or any replacement onion engine,
    # via the GHOST_ONION_ENGINES env override once a live endpoint is known:
    #   http://haystak5njsmn2hqkewecpaxetahtwhsbsa64jom2k22z5afxhnpxfid.onion/?q={q}
]

# Per-request timeout. Onion search engines are slow — a healthy round trip
# routinely takes 15-25s through a Tor circuit, so an aggressive ceiling
# kills a request that would otherwise succeed (the exact failure mode the
# `_DDGS_TOR_TIMEOUT=18` comment in search.py documents for mojeek). 30s
# clears a healthy onion engine; dead ones still fail fast on connect error.
_ONION_TIMEOUT = 30

# Per-onion-page fetch ceiling during the research (deep-read) phase. Onion
# content pages are slower still than the search engines; give them room.
_ONION_PAGE_TIMEOUT = 35.0

# Overall wall-clock deadline for a SINGLE engine (both attempts combined).
# Because engines are queried concurrently, the whole search can only finish
# once the SLOWEST engine returns — so a dead or hung engine that burns its
# per-attempt timeout twice (~2x _ONION_TIMEOUT) would dominate the gather and
# inflate every call's latency (the failure mode that motivated dropping the
# dead `haystak` default). This hard-caps any one engine's contribution to the
# gather regardless of the engine set, so curation is a tuning aid, not the
# only thing standing between a newly-dead engine and a slow tool. Sized to
# allow ONE full attempt (a slow-but-alive onion engine legitimately needs up
# to _ONION_TIMEOUT) plus a short window for a fast second circuit; a slow
# first attempt simply forfeits most of the retry.
_ONION_ENGINE_DEADLINE = _ONION_TIMEOUT + 8

# Hard body-size ceiling for a single raw fetch. Onion engines are UNTRUSTED
# and adversarial by this tool's own posture; without a cap a hostile or
# misconfigured engine could stream a multi-GB body straight into memory and
# then into BeautifulSoup + a full-document regex sweep. Mirrors the 5 MB cap
# `helper_fetch_url_content` enforces on clearnet fetches.
_MAX_ONION_BODY_BYTES = 5 * 1024 * 1024

# Onion fetches run in worker threads (curl_cffi/httpx are sync). When the
# per-engine deadline fires, `asyncio.wait_for` cancels the AWAIT but cannot
# kill the thread — it keeps running until curl's own timeout. Isolating those
# threads in a dedicated bounded pool means lingering post-deadline fetches
# can never exhaust the process-wide default executor that the rest of the
# agent relies on; excess fetches simply queue here instead.
_ONION_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="onion-fetch")


def _load_engines() -> List[Dict[str, str]]:
    """Return the active onion-engine set, honouring the GHOST_ONION_ENGINES
    override. Falls back to the built-in default on any parse problem so a
    malformed override can never silently disable dark-web search.

    An override entry may carry an ``index`` to mark it as sharing a search
    index with another endpoint; if omitted it defaults to the entry's own
    name (i.e. treated as an independent index)."""
    raw = os.getenv("GHOST_ONION_ENGINES")
    if not raw:
        return [dict(e) for e in _DEFAULT_ONION_ENGINES]
    try:
        parsed = json.loads(raw)
        engines = []
        for e in parsed:
            if not (isinstance(e, dict) and e.get("name")):
                continue
            url = str(e.get("url", ""))
            if "{q}" not in url:
                continue
            # Reject stray format placeholders: url.format(q=...) would raise
            # KeyError at query time and the engine would be silently skipped
            # on every search. Surface the config error at load instead.
            residue = url.replace("{q}", "")
            if "{" in residue or "}" in residue:
                pretty_log(
                    "Darkweb Config",
                    f"engine {e.get('name')!r} URL has an invalid placeholder "
                    "(only {q} is allowed); skipping it.",
                    level="WARNING", icon=Icons.WARN,
                )
                continue
            name = str(e["name"])
            engines.append({"name": name, "url": url, "index": str(e.get("index") or name)})
        return engines or [dict(e) for e in _DEFAULT_ONION_ENGINES]
    except Exception:
        pretty_log(
            "Darkweb Config",
            "GHOST_ONION_ENGINES is malformed; using built-in engine set.",
            level="WARNING",
            icon=Icons.WARN,
        )
        return [dict(e) for e in _DEFAULT_ONION_ENGINES]


def _engine_onion_hosts(engines: List[Dict[str, str]]) -> set:
    """The set of onion hosts belonging to the engines THEMSELVES.

    Onion result pages carry nav/footer/sidebar links — including the
    engine's own onion address and those of sibling engines — which the
    tolerant parser would otherwise harvest as bogus "results" (and which,
    via corroboration ranking, could even sort to the top). We exclude every
    engine's own onion host from the parsed results."""
    hosts = set()
    for e in engines:
        host = _onion_host(e.get("url", ""))
        if host.endswith(".onion"):
            hosts.add(host)
    return hosts


def _normalize_tor_proxy(tor_proxy: Optional[str]) -> str:
    """Resolve and normalise the SOCKS proxy to ``socks5h://`` form.

    ``socks5h`` (note the *h*) routes DNS resolution THROUGH the proxy —
    mandatory for ``.onion``, which has no clearnet DNS. Falls back to the
    ``TOR_PROXY`` env var, then the conventional local Tor port, mirroring
    `helper_fetch_url_content`."""
    proxy = tor_proxy or os.getenv("TOR_PROXY", "socks5://127.0.0.1:9050")
    if proxy and proxy.startswith("socks5://"):
        proxy = proxy.replace("socks5://", "socks5h://")
    return proxy


def _extract_onion(s: str) -> Optional[str]:
    """Pull a single ``.onion`` URL out of an href / redirect wrapper / text.

    Ahmia (and some others) wrap the real target in a redirect URL like
    ``/search/redirect?...&redirect_url=http%3A%2F%2F...onion``; we unwrap
    that first, then fall back to a direct regex scan."""
    if not s:
        return None
    try:
        pr = urlparse(s)
        if pr.query:
            qs = parse_qs(pr.query)
            for key in ("redirect_url", "url", "u", "d"):
                if key in qs and qs[key]:
                    cand = unquote(qs[key][0])
                    m = _ONION_RE.search(cand)
                    if m:
                        return m.group(0)
    except Exception:
        pass
    m = _ONION_RE.search(s)
    return m.group(0) if m else None


def _onion_host(url: str) -> str:
    try:
        return (urlparse(url).hostname or url).lower()
    except Exception:
        return url.lower()


def _parse_onion_results(html: str, exclude_hosts: Optional[set] = None) -> List[Dict[str, str]]:
    """Parse an onion search-engine result page into {url,title,snippet}.

    Each engine renders differently, so the parse is deliberately generic
    and tolerant: walk anchors to harvest (onion-url, title) pairs with a
    nearby snippet, then a regex sweep of the whole document catches any
    onion URL that appeared as plain text rather than a link. De-duped by
    onion host within a single page. Hosts in ``exclude_hosts`` (typically
    the engines' own onion addresses) are dropped so engine nav/footer
    self-links never surface as results."""
    results: List[Dict[str, str]] = []
    seen: set = set(exclude_hosts or ())
    if not html:
        return results

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            onion = _extract_onion(a["href"])
            if not onion:
                continue
            host = _onion_host(onion)
            if host in seen:
                continue
            seen.add(host)
            title = a.get_text(" ", strip=True)
            # Snippet: text of the nearest block-level ancestor, minus the
            # title. Best-effort — a missing snippet is fine.
            snippet = ""
            block = a.find_parent(["li", "div", "article", "section", "p"])
            if block is not None:
                btext = block.get_text(" ", strip=True)
                if title and title in btext:
                    btext = btext.replace(title, "", 1)
                snippet = " ".join(btext.split())[:400]
            results.append(
                {"url": onion, "title": title or host, "snippet": snippet}
            )
    except Exception:
        # BeautifulSoup unavailable or blew up — fall through to regex sweep.
        pass

    # Regex sweep for onion URLs not captured as anchors (plain text, JSON
    # blobs, etc.). These get the host as the title.
    for m in _ONION_RE.finditer(html):
        onion = m.group(0)
        host = _onion_host(onion)
        if host in seen:
            continue
        seen.add(host)
        results.append({"url": onion, "title": host, "snippet": ""})

    return results


def _cap_body(status: Optional[int], content_type: Optional[str],
              content_length: Any, text: Optional[str]) -> Tuple[Optional[int], str]:
    """Apply the untrusted-body guards to a fetched response.

    Refuses binary payloads and bodies whose declared Content-Length exceeds
    the ceiling, and truncates the decoded text as a backstop against a server
    that lied about its size or streamed chunked. Pure/synchronous so it is
    unit-testable without a live fetch."""
    ct = (content_type or "").lower()
    if "application/pdf" in ct or "application/octet-stream" in ct:
        return status, ""
    try:
        clen = int(content_length or 0)
    except (TypeError, ValueError):
        clen = 0
    if clen and clen > _MAX_ONION_BODY_BYTES:
        return status, ""
    text = text or ""
    if len(text) > _MAX_ONION_BODY_BYTES:
        text = text[:_MAX_ONION_BODY_BYTES]
    return status, text


async def _fetch_raw_html(url: str, proxy: Optional[str], timeout: float) -> Tuple[Optional[int], str]:
    """Fetch RAW HTML (tags intact) through the Tor SOCKS proxy.

    Unlike `helper_fetch_url_content`, this does NOT strip tags — the search
    phase needs the markup to parse out result links — and it HONOURS the
    passed proxy rather than reading it from the environment. Uses curl_cffi
    when present (TLS-impersonating, the project default) and falls back to
    httpx. The body is size-capped (`_cap_body`) and the blocking request runs
    on a dedicated bounded pool so a post-deadline lingering fetch can't
    exhaust the shared executor. Returns (status_code, body); (None, "") on
    transport failure."""

    # Read at most this many BYTES off the wire regardless of Content-Length.
    # An untrusted onion engine can send a chunked body with no (or a lying)
    # Content-Length; reading `r.text` (the whole body into RAM) before
    # `_cap_body` could truncate it OOMs the host. Stream and stop at the cap.
    _STREAM_LIMIT = _MAX_ONION_BODY_BYTES + 4096

    def _decode(buf: bytes) -> str:
        return buf.decode("utf-8", errors="replace")

    def run() -> Tuple[Optional[int], str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            import curl_cffi.requests as creq

            proxies = {"http": proxy, "https": proxy} if proxy else None
            with creq.Session(impersonate="chrome110", proxies=proxies, timeout=timeout) as c:
                r = c.get(url, headers=headers, stream=True)
                buf = bytearray()
                try:
                    for chunk in r.iter_content():
                        if chunk:
                            buf.extend(chunk)
                            if len(buf) >= _STREAM_LIMIT:
                                break
                finally:
                    try: r.close()
                    except Exception: pass
                return _cap_body(r.status_code, r.headers.get("content-type"),
                                 r.headers.get("content-length"), _decode(bytes(buf)))
        except ImportError:
            import httpx

            with httpx.Client(proxy=proxy, timeout=timeout, follow_redirects=True) as c:
                with c.stream("GET", url, headers=headers) as r:
                    buf = bytearray()
                    for chunk in r.iter_bytes():
                        buf.extend(chunk)
                        if len(buf) >= _STREAM_LIMIT:
                            break
                    return _cap_body(r.status_code, r.headers.get("content-type"),
                                     r.headers.get("content-length"), _decode(bytes(buf)))

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_ONION_EXECUTOR, run)


def _strip_html(html: str) -> str:
    """Reduce raw HTML to readable text (research phase), dropping chrome."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "iframe", "svg"]):
            tag.decompose()
        txt = soup.get_text(separator=" ", strip=True)
        return " ".join(txt.split()) if txt else ""
    except Exception:
        return html


async def _fetch_onion_text(url: str, tor_proxy: str) -> str:
    """Fetch an onion PAGE and return its readable text.

    The research analogue of `helper_fetch_url_content`, but it (a) honours
    the passed ``tor_proxy`` (and thus any per-query anonymity identity tag),
    and (b) does NOT trigger `request_new_tor_identity` on failure — a global
    NEWNYM / `tor` service restart mid-run would sabotage the sibling onion
    fetches running concurrently. Keeps the shared SSRF guard and the raw
    fetch's body-size cap."""
    reason = url_ssrf_reason(url)
    if reason:
        return f"Error: {reason}"
    try:
        status, html = await _fetch_raw_html(url, tor_proxy, _ONION_PAGE_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        return f"Error reading {url}: {e}"
    if not status:
        return f"Error: no response from {url}"
    if status != 200 or not html:
        return f"Error: received status {status} from {url}"
    return await asyncio.to_thread(_strip_html, html)


async def _query_engine(
    engine: Dict[str, str],
    query: str,
    tor_proxy: str,
    exclude_hosts: Optional[set] = None,
) -> List[Dict[str, str]]:
    """Query one onion engine with per-attempt circuit rotation, under an
    overall wall-clock deadline (`_ONION_ENGINE_DEADLINE`) so one slow or dead
    engine can't dominate the concurrent gather. Returns its parsed results
    (possibly empty); never raises."""

    async def _attempts() -> List[Dict[str, str]]:
        url = engine["url"].format(q=quote_plus(query))
        for attempt in range(2):
            proxy = _proxy_for_attempt(tor_proxy, f"{engine['name']}:{query}", attempt)
            try:
                status, body = await _fetch_raw_html(url, proxy, _ONION_TIMEOUT)
                if status == 200 and body:
                    parsed = _parse_onion_results(body, exclude_hosts)
                    if parsed:
                        pretty_log(
                            "Darkweb Engine",
                            f"{engine['name']}: {len(parsed)} onion result(s)",
                            icon=Icons.TOOL_DARKWEB,
                        )
                        return parsed
            except Exception as e:  # noqa: BLE001
                pretty_log(
                    "Darkweb Engine Error",
                    f"{engine['name']}: {e}",
                    level="WARNING",
                    icon=Icons.WARN,
                )
            # Only pause before an actual RETRY — no point sleeping after the
            # final attempt when we're about to return [].
            if attempt == 0:
                await asyncio.sleep(0.5)
        return []

    try:
        return await asyncio.wait_for(_attempts(), timeout=_ONION_ENGINE_DEADLINE)
    except asyncio.TimeoutError:
        # The underlying fetch runs in a worker thread (curl_cffi/httpx has
        # its own timeout), so it isn't force-killed here — but cancelling the
        # await lets the gather proceed without waiting on this engine. The
        # dedicated `_ONION_EXECUTOR` keeps that lingering thread off the
        # shared pool.
        pretty_log(
            "Darkweb Engine Error",
            f"{engine['name']}: exceeded {_ONION_ENGINE_DEADLINE:.0f}s deadline — skipped",
            level="WARNING",
            icon=Icons.WARN,
        )
        return []


def _apply_anonymous_scrub(query: str, tor_proxy: Optional[str]) -> Tuple[str, Optional[str]]:
    """Deterministic stylometry scrub + per-query SOCKS identity tag, mirroring
    `tool_search`'s anonymous branch so the agent's prose fingerprint and
    cross-query linkability don't leak alongside the Tor-anonymised packets."""
    try:
        from ..utils.stylometry import scrub_query

        query = scrub_query(query) or query
    except Exception:
        pass
    if tor_proxy:
        try:
            import hashlib
            from ..utils.helpers import socks_url_with_identity

            tag = hashlib.md5((query or "").encode("utf-8", "ignore")).hexdigest()[:12]
            tor_proxy = socks_url_with_identity(tor_proxy, tag) or tor_proxy
        except Exception:
            pass
    return query, tor_proxy


_NO_RESULTS_ERROR = (
    "ERROR: dark-web search returned ZERO results across all onion search "
    "engines and circuits. Likely causes: (a) every configured onion engine "
    "is transiently down or blocking this exit node — onion engines are far "
    "flakier than clearnet ones; (b) the query was too specific. DO NOT retry "
    "the same search immediately. Instead: drop to 2-4 PLAIN keywords, or if "
    "Tor itself may be down, fall back to web_search. Note: hidden services "
    "appear and vanish constantly, so a topic having no onion index is normal."
)


async def _darkweb_search_raw(query: str, tor_proxy: str, max_results: int = 12) -> List[Dict[str, Any]]:
    """Core fan-out: query every engine concurrently, merge + rank results.

    Ranking favours onions surfaced by MORE THAN ONE INDEPENDENT INDEX
    (corroboration is the only cheap relevance signal we have for unindexed
    hidden services), preserving discovery order within a tier. Endpoints that
    share an ``index`` (e.g. Ahmia's clearnet + onion mirrors) count once, so
    reaching one index over two transports is NOT mistaken for independent
    agreement. Returns ranked result dicts, each carrying the engine names and
    indexes that surfaced it."""
    engines = _load_engines()
    exclude = _engine_onion_hosts(engines)
    per_engine = await asyncio.gather(
        *[_query_engine(e, query, tor_proxy, exclude) for e in engines],
        return_exceptions=True,
    )

    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for engine, res in zip(engines, per_engine):
        if not isinstance(res, list):
            continue
        idx = engine.get("index") or engine["name"]
        for r in res:
            host = _onion_host(r["url"])
            if host not in merged:
                merged[host] = {**r, "engines": {engine["name"]}, "indexes": {idx}}
                order.append(host)
            else:
                merged[host]["engines"].add(engine["name"])
                merged[host]["indexes"].add(idx)
                # Keep the richest snippet/title seen.
                if not merged[host].get("snippet") and r.get("snippet"):
                    merged[host]["snippet"] = r["snippet"]
                if merged[host].get("title") == host and r.get("title") and r["title"] != host:
                    merged[host]["title"] = r["title"]

    ranked = sorted(
        (merged[h] for h in order),
        key=lambda r: -len(r["indexes"]),
    )
    return ranked[:max_results]


def _format_results(results: List[Dict[str, Any]]) -> str:
    formatted = []
    for i, r in enumerate(results, 1):
        title = _clean_for_cpp(r.get("title") or r["url"])
        snippet = _clean_for_cpp(r.get("snippet") or "")
        engs = ", ".join(sorted(r.get("engines", [])))
        body = (snippet + "\n") if snippet else ""
        formatted.append(f"### {i}. {title}\n{body}[Onion: {r['url']}] (via {engs})")
    return "\n\n".join(formatted)


async def tool_darkweb_search(
    query: Optional[str] = None,
    anonymous: bool = False,
    tor_proxy: Optional[str] = None,
    max_results: int = 12,
    **kwargs: Any,
) -> str:
    """List ranked .onion services matching a query, via onion search engines."""
    if not query:
        return "SYSTEM ERROR: The 'query' parameter is MANDATORY. You must specify it."

    if anonymous and query:
        query, tor_proxy = _apply_anonymous_scrub(query, tor_proxy)

    # NOTE: the operator/quote/boolean stripping in `_sanitize_query` is
    # clearnet-derived but applies cleanly to Ahmia/Torch too — both are plain
    # keyword indexes that do not honour Google-style operators, so removing
    # them can only help, never lose a supported qualifier.
    query = _sanitize_query(query)
    tor_proxy = _normalize_tor_proxy(tor_proxy)
    pretty_log("Darkweb Search", query, icon=Icons.TOOL_DARKWEB)

    cache_key = "onion::" + (query or "").strip().lower()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ranked = await _darkweb_search_raw(query, tor_proxy, max_results=max_results)
    if not ranked:
        return _NO_RESULTS_ERROR

    reached = sorted({e for r in ranked for e in r.get("engines", [])})
    header = f"[Dark-web search — onion results, engines reached: {', '.join(reached)}]"
    output = header + "\n\n" + _format_results(ranked)
    _cache_put(cache_key, output)
    return output


async def tool_darkweb_research(
    query: Optional[str] = None,
    anonymous: bool = False,
    tor_proxy: Optional[str] = None,
    llm_client=None,
    model_name: str = "default",
    max_context: int = 8192,
    workspace_model=None,
    max_sources: int = 6,
    **kwargs: Any,
) -> str:
    """Search .onion services, then fetch + distill the top results into a
    synthesised report. The dark-web analogue of `tool_deep_research`."""
    if not query:
        return "SYSTEM ERROR: The 'query' parameter is MANDATORY. You must specify it."

    # Stronger anonymous tier: re-author the query with the local model (the
    # same posture deep_research takes), falling back to the lexical scrub.
    if anonymous and query:
        try:
            from ..utils.stylometry import neutralize_query

            query = await neutralize_query(query, llm_client=llm_client, model=model_name) or query
        except Exception:
            pass
        _, tor_proxy = _apply_anonymous_scrub(query, tor_proxy)

    query = _sanitize_query(query)
    tor_proxy = _normalize_tor_proxy(tor_proxy)
    pretty_log("Darkweb Research", query, icon=Icons.TOOL_DARKWEB)

    # The deep-read path is expensive (an onion fetch + LLM distill per source);
    # cache the synthesised report so a repeated identical research request
    # doesn't re-fetch every onion. Namespaced distinctly from the list search.
    cache_key = "onion-research::" + (query or "").strip().lower()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ranked = await _darkweb_search_raw(query, tor_proxy, max_results=max_sources)
    if not ranked:
        return _NO_RESULTS_ERROR

    urls = [r["url"] for r in ranked][:max_sources]

    # Size the per-source extract to the worker's context window rather than a
    # fixed 40k chars, so a small-context worker model can't overflow on the
    # distill call (~4 chars/token, reserving room for the prompt + max_tokens).
    reserve_tokens = 2048 + 512
    usable_tokens = max(1024, int(max_context) - reserve_tokens)
    url_char_limit = max(4000, min(40000, usable_tokens * 4))
    fallback_limit = min(10000, url_char_limit)

    sem = asyncio.Semaphore(2)

    async def _fetch_with_timeout(url: str) -> str:
        try:
            return await asyncio.wait_for(
                _fetch_onion_text(url, tor_proxy), timeout=_ONION_PAGE_TIMEOUT + 5
            )
        except asyncio.TimeoutError:
            return f"Error: Fetch of {url} timed out after {_ONION_PAGE_TIMEOUT}s"
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}"

    async def process_url(url: str) -> str:
        async with sem:
            short_url = (url[:35] + "..") if len(url) > 35 else url
            pretty_log("Parsing Onion", url, icon=Icons.TOOL_DARKWEB)
            text = await _fetch_with_timeout(url)

            safe_text = _clean_for_cpp(text[:url_char_limit])

            if llm_client:
                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"Extract ONLY the hard facts explicitly relevant to this "
                                f"query: '{query}'. Ignore all other boilerplate. If no "
                                f"relevant info is found, state that.\n\nSource text:\n{safe_text}"
                            ),
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 2048,
                }
                try:
                    summary_data = await llm_client.chat_completion(payload, use_worker=True)
                    pretty_log("Worker Compute", f"Distilling facts from {short_url}", icon=Icons.TOOL_DEEP)
                    preview = "[EDGE EXTRACTED FACTS]:\n" + (summary_data["choices"][0]["message"].get("content") or "").strip()
                except Exception:
                    # Clean the raw-text fallback: unscrubbed surrogates/control
                    # chars from an onion page can crash the downstream C++ JSON
                    # parser (what _clean_for_cpp prevents). safe_text is already
                    # cleaned; the fallback used raw `text`.
                    preview = _clean_for_cpp(text[:fallback_limit]) + "\n[...truncated...]\n"
            else:
                preview = _clean_for_cpp(text[:fallback_limit]) + "\n[...truncated...]\n"
            return f"### SOURCE: {url}\n{preview}\n"

    async def _bounded(url: str) -> str:
        try:
            return await asyncio.wait_for(process_url(url), timeout=_ONION_PAGE_TIMEOUT + 10)
        except asyncio.TimeoutError:
            return f"### SOURCE: {url}\nError: per-URL timeout exceeded\n"

    tasks = [_bounded(u) for u in urls]
    page_contents = await asyncio.gather(*tasks, return_exceptions=True)
    valid_contents = [c for c in page_contents if isinstance(c, str)]
    full_report = "\n\n".join(valid_contents)

    # Workspace research dedup — record every onion we pulled. Non-fatal.
    if workspace_model is not None and getattr(workspace_model, "enabled", False):
        try:
            for u in urls:
                workspace_model.record_research_artifact(
                    url=u, source="darkweb_research", note=(query or "")[:120],
                )
        except Exception:  # noqa: BLE001
            pass

    result = (
        f"--- DARK-WEB RESEARCH RESULT ---\n{full_report}\n\n"
        "SYSTEM INSTRUCTION: Analyze the text above. These are UNVERIFIED hidden "
        "services — treat claims with suspicion and corroborate before relying on them."
    )
    # Only cache when at least one source actually produced content — otherwise
    # an all-errors run (every onion down/timed out this attempt) would be
    # served back for 300s instead of re-attempting the fetches next time.
    def _source_succeeded(block: str) -> bool:
        # block is "### SOURCE: <url>\n<preview>\n"; the preview begins with
        # "Error:" only when the fetch/timeout failed.
        parts = block.split("\n", 1)
        preview = parts[1].strip() if len(parts) > 1 else ""
        return bool(preview) and not preview.startswith("Error:")

    if any(_source_succeeded(c) for c in valid_contents):
        _cache_put(cache_key, result)
    return result
