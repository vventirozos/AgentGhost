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
    rotation, the same philosophy as `_TOR_BACKENDS`.
  * `_fetch_raw_html` — onion search needs the RAW result HTML so we can
    parse out the result links. `helper_fetch_url_content` strips all tags
    to plain text (destroying the links), so it is unusable for the search
    phase; we fetch raw HTML ourselves. The *research* phase, which only
    wants page text, DOES reuse `helper_fetch_url_content`.
  * `_ONION_TIMEOUT` — onion round-trips are materially slower than
    clearnet-over-Tor, so the per-request ceiling is higher than search.py's
    `_DDGS_TOR_TIMEOUT` (18s). Measured-empirically tunable, like that one.
"""
import asyncio
import json
import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus, urlparse, parse_qs, unquote

from ..utils.logging import Icons, pretty_log
from ..utils.helpers import helper_fetch_url_content
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
# Each entry: {name, url}. ``url`` is a template with a single ``{q}``
# placeholder that receives the URL-encoded query. We keep a WIDE set: onion
# engines are individually flaky and per-exit-node reachable, so breadth +
# circuit rotation wins. Override the whole set with the GHOST_ONION_ENGINES
# env var (a JSON list of {"name","url"} objects) when endpoints rotate.
#
# Ahmia is listed twice on purpose: its clearnet mirror (ahmia.fi, still
# fetched over Tor) and its onion endpoint have independent reachability, so
# one is often up when the other is blocked. Ahmia also filters known-abuse
# material at the index level — useful as a sane primary even in the
# personal/experimental posture this tool is built for.
_DEFAULT_ONION_ENGINES: List[Dict[str, str]] = [
    {"name": "ahmia", "url": "https://ahmia.fi/search/?q={q}"},
    {
        "name": "ahmia-onion",
        "url": "http://juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd.onion/search/?q={q}",
    },
    {
        "name": "torch",
        "url": "http://torchdeedp3i2jigzjdmfpn5ttjhthh5wbmda2rr3jvqjg5p77c54dqd.onion/search?query={q}",
    },
    {
        "name": "haystak",
        "url": "http://haystak5njsmn2hqkewecpaxetahtwhsbsa64jom2k22z5afxhnpxfid.onion/?q={q}",
    },
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


def _load_engines() -> List[Dict[str, str]]:
    """Return the active onion-engine set, honouring the GHOST_ONION_ENGINES
    override. Falls back to the built-in default on any parse problem so a
    malformed override can never silently disable dark-web search."""
    raw = os.getenv("GHOST_ONION_ENGINES")
    if not raw:
        return list(_DEFAULT_ONION_ENGINES)
    try:
        parsed = json.loads(raw)
        engines = [
            {"name": str(e["name"]), "url": str(e["url"])}
            for e in parsed
            if isinstance(e, dict) and e.get("name") and "{q}" in str(e.get("url", ""))
        ]
        return engines or list(_DEFAULT_ONION_ENGINES)
    except Exception:
        pretty_log(
            "Darkweb Config",
            "GHOST_ONION_ENGINES is malformed; using built-in engine set.",
            level="WARNING",
            icon=Icons.WARN,
        )
        return list(_DEFAULT_ONION_ENGINES)


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


def _parse_onion_results(html: str) -> List[Dict[str, str]]:
    """Parse an onion search-engine result page into {url,title,snippet}.

    Each engine renders differently, so the parse is deliberately generic
    and tolerant: walk anchors to harvest (onion-url, title) pairs with a
    nearby snippet, then a regex sweep of the whole document catches any
    onion URL that appeared as plain text rather than a link. De-duped by
    onion host within a single page."""
    results: List[Dict[str, str]] = []
    seen: set = set()
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


async def _fetch_raw_html(url: str, proxy: Optional[str], timeout: float) -> Tuple[Optional[int], str]:
    """Fetch RAW HTML (tags intact) through the Tor SOCKS proxy.

    Unlike `helper_fetch_url_content`, this does NOT strip tags — the search
    phase needs the markup to parse out result links. Uses curl_cffi when
    present (TLS-impersonating, the project default) and falls back to httpx.
    Returns (status_code, body); (None, "") on transport failure."""

    def run() -> Tuple[Optional[int], str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            import curl_cffi.requests as creq

            proxies = {"http": proxy, "https": proxy} if proxy else None
            with creq.Session(impersonate="chrome110", proxies=proxies, timeout=timeout) as c:
                r = c.get(url, headers=headers)
                return r.status_code, (r.text or "")
        except ImportError:
            import httpx

            with httpx.Client(proxy=proxy, timeout=timeout, follow_redirects=True) as c:
                r = c.get(url, headers=headers)
                return r.status_code, (r.text or "")

    return await asyncio.to_thread(run)


async def _query_engine(engine: Dict[str, str], query: str, tor_proxy: str) -> List[Dict[str, str]]:
    """Query one onion engine with per-attempt circuit rotation. Returns its
    parsed results (possibly empty); never raises."""
    url = engine["url"].format(q=quote_plus(query))
    for attempt in range(2):
        proxy = _proxy_for_attempt(tor_proxy, f"{engine['name']}:{query}", attempt)
        try:
            status, body = await _fetch_raw_html(url, proxy, _ONION_TIMEOUT)
            if status == 200 and body:
                parsed = _parse_onion_results(body)
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
        await asyncio.sleep(0.5)
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

    Ranking favours onions surfaced by MORE THAN ONE engine (corroboration is
    the only cheap relevance signal we have for unindexed hidden services),
    preserving discovery order within a tier. Returns ranked result dicts,
    each carrying the set of engines that surfaced it."""
    engines = _load_engines()
    per_engine = await asyncio.gather(
        *[_query_engine(e, query, tor_proxy) for e in engines],
        return_exceptions=True,
    )

    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for engine, res in zip(engines, per_engine):
        if not isinstance(res, list):
            continue
        for r in res:
            host = _onion_host(r["url"])
            if host not in merged:
                merged[host] = {**r, "engines": {engine["name"]}}
                order.append(host)
            else:
                merged[host]["engines"].add(engine["name"])
                # Keep the richest snippet/title seen.
                if not merged[host].get("snippet") and r.get("snippet"):
                    merged[host]["snippet"] = r["snippet"]

    ranked = sorted(
        (merged[h] for h in order),
        key=lambda r: -len(r["engines"]),
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
    pretty_log("Darkweb Research", query, icon=Icons.TOOL_DEEP)

    ranked = await _darkweb_search_raw(query, tor_proxy, max_results=max_sources)
    if not ranked:
        return _NO_RESULTS_ERROR

    urls = [r["url"] for r in ranked][:max_sources]

    sem = asyncio.Semaphore(2)

    async def _fetch_with_timeout(url: str) -> str:
        try:
            return await asyncio.wait_for(
                helper_fetch_url_content(url), timeout=_ONION_PAGE_TIMEOUT
            )
        except asyncio.TimeoutError:
            return f"Error: Fetch of {url} timed out after {_ONION_PAGE_TIMEOUT}s"
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}"

    async def process_url(url: str) -> str:
        async with sem:
            short_url = (url[:35] + "..") if len(url) > 35 else url
            pretty_log("Parsing Onion", url, icon=Icons.TOOL_FILE_R)
            text = await _fetch_with_timeout(url)

            url_char_limit = 40000
            fallback_limit = 10000
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
                    preview = "[EDGE EXTRACTED FACTS]:\n" + summary_data["choices"][0]["message"].get("content", "").strip()
                except Exception:
                    preview = text[:fallback_limit] + "\n[...truncated...]\n"
            else:
                preview = text[:fallback_limit] + "\n[...truncated...]\n"
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

    return (
        f"--- DARK-WEB RESEARCH RESULT ---\n{full_report}\n\n"
        "SYSTEM INSTRUCTION: Analyze the text above. These are UNVERIFIED hidden "
        "services — treat claims with suspicion and corroborate before relying on them."
    )
