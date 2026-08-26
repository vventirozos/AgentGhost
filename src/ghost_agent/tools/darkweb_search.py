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
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus, urlparse, parse_qs, unquote

from ..utils.logging import Icons, pretty_log
from ..core.node_throughput import (
    CHARS_PER_TOKEN, MIN_CHARS as _MIN_DISTILL_CHARS,
    log_plan as log_distill_plan)

# ⚠ This module referenced `logger` without ever defining it — a
# NameError on every circuit-breaker skip, invisible because
# `asyncio.gather(..., return_exceptions=True)` discarded it and the
# engine returning nothing was the intended outcome anyway. The
# breaker 'worked' by accident, with its only observability dead.
logger = logging.getLogger("GhostAgent")
from ..utils.helpers import url_ssrf_reason
from .search import (
    _sanitize_query,
    _proxy_for_attempt,
    _clean_for_cpp,
    _cache_get,
    _cache_put,
    _norm_cache_key,
    # ⚠ IMPORTED, NOT RE-DECLARED. This module had its own 45.0 copy under
    # the same name — two constants one edit apart, so tuning the web
    # summary leash in search.py silently left the darkweb path on the old
    # value (R4 lens A). Onion fetches are the SLOWER of the two; if they
    # ever need a different budget it must be a differently-NAMED constant.
    _WEB_SUMMARY_TIMEOUT_S,
    _RAW_FALLBACK_CHARS,
)
# ⚠ THE CLOCK TUNABLES ARE READ AT CALL TIME, NOT BOUND AT IMPORT. Importing
# them by value silently un-linked the two paths: patching or re-tuning
# `search._RESEARCH_PHASE_TIMEOUT_S` moved the clearnet phase clock and left
# the onion one on the value captured at import, so a budget test that patched
# the sibling tested nothing here. Worse, a tunable added to `search.py` and
# not to this import list is a NameError inside `process_url` — which is
# exactly what happened while writing this change, and was only survivable
# because the sizing call is guarded (it degraded to raw text and said so).
from . import search as _budgets

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
#
# ``form_token_from`` (optional) names a page whose search FORM carries a
# hidden input that must ride the query string — see `_form_token`. Ahmia
# added one (measured 2026-07-28): without it `/search/?q=…` 302-redirects to
# `/` and serves the homepage, which parses to zero results.
_DEFAULT_ONION_ENGINES: List[Dict[str, str]] = [
    {
        "name": "ahmia",
        "url": "https://ahmia.fi/search/?q={q}",
        "index": "ahmia",
        "form_token_from": "https://ahmia.fi/",
    },
    {
        "name": "ahmia-onion",
        "url": "http://juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd.onion/search/?q={q}",
        "index": "ahmia",
        "form_token_from": "http://juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd.onion/",
    },
    {
        # Torch, re-pointed 2026-07-29. The previous `torchdeed…` address was
        # dead (0/10 across fresh circuits, every one a 30s timeout — the same
        # signature that got haystak dropped), which cost the tool a full
        # deadline on EVERY search. The service is alive at this address under
        # its Xapian Omega CGI path; `/search?query=` there returns 404, so
        # the old path would have looked dead even at the right address.
        # HITSPERPAGE lifts one page from 7 unique onions to 28 (the parser
        # de-dupes by host, so this is 100 hits collapsing to 28 hosts).
        # Measured 4/4 reachable at 1.6-6.6s — the fastest engine in the set.
        "name": "torch",
        "url": "http://xmh57jrknzkhv6y3ls3ubitzfqnkrwxhopf5aygthi7d6rplyvk3noyd.onion"
               "/cgi-bin/omega/omega?P={q}&HITSPERPAGE=100",
        "index": "torch",
    },
    {
        # Torgle — added 2026-07-29 as a third INDEPENDENT index, so
        # corroboration ranking has something to corroborate with when torch
        # or ahmia is having a bad day. Measured 4/4 reachable, 19-20 results,
        # but slow (9-25s), which is why it is worth having and not worth
        # relying on alone.
        "name": "torgle",
        "url": "http://no6m4wzdexe3auiupv2zwif7rm6qwxcyhslkcnzisxgeiw6pvjsgafad.onion"
               "/search.php?term={q}",
        "index": "torgle",
    },
    # Measured and REJECTED 2026-07-29, all over live Tor with this module's
    # own fetch+parse (re-measure before believing any of these again —
    # onion endpoints rotate constantly):
    #   haystak   0/2  timeout   (still down; dropped 2026-06/07 for the same)
    #   onionland 0/2  timeout
    #   tor66     0/2  timeout
    #   phobos    0/2  SOCKS "cannot complete" = descriptor not found
    #   tordex    1/6  one 86-result hit, otherwise timeout — too flaky to pay
    #             a deadline for; re-add via GHOST_ONION_ENGINES if it settles
    #             http://tordexu73joywapk2txdr54jed4imqledpcvcuf75qsas2gwdgksvnyd.onion/search?query={q}
]

# Per-request timeout. Onion search engines are slow — a healthy round trip
# routinely takes 15-25s through a Tor circuit, so an aggressive ceiling
# kills a request that would otherwise succeed (the exact failure mode the
# `_DDGS_TOR_TIMEOUT=18` comment in search.py documents for mojeek). 30s
# clears a healthy onion engine; dead ones still fail fast on connect error.
_ONION_TIMEOUT = 30

# Marker text an engine serves when its results are JS-rendered (no
# server-side HTML to parse). A raw-HTML fetch can never extract results from
# such a page, so it is worth naming — but ONLY when the page actually says
# so. This was a bare ``"javascript"`` substring test, which is present in
# almost every modern page (a <script> tag, an analytics blurb, a hidden
# "no-JS" notice meant for someone else) and produced a confidently WRONG
# diagnosis: Ahmia's *homepage* carries a display:none non-JS warning, so
# after a 302 bounced us there we told the operator to swap engines when the
# real fix was one query parameter. Anchor on the sentence a page shows when
# it genuinely needs JS, and treat a redirect-away as the better explanation
# when both are true (see `_diagnose_empty_body`).
# NOTE: no bare ``<noscript>`` alternative here, tempting as it is — plenty of
# working pages wrap an analytics pixel in one, and mislabelling those would
# just re-create this bug with a different trigger. Only an explicit sentence
# counts, and only with the right POLARITY: privacy-focused engines — exactly
# the population this tool scans — advertise "works without JavaScript" and
# "No JavaScript required" in their footers, so a polarity-blind pattern turns
# the engines most likely to WORK into the ones we declare broken. Every
# alternative below is a demand for JS, not a mention of it.
_JS_ONLY_RE = re.compile(
    r"(?:please\s+)?(?:enable|activate|turn\s+on)\s+javascript"
    r"|javascript\s+(?:must\s+be|has\s+to\s+be)\s+enabled"
    r"|(?<!no\s)(?<!not\s)javascript\s+is\s+(?:required|disabled)"
    r"|requires\s+javascript\s+to"
    r"|(?:does\s+not|doesn't)\s+support\s+javascript",
    re.IGNORECASE,
)

# A page that genuinely cannot render without JS is a SHELL: a little markup
# and a notice. Ahmia's real results page is >1 MB and carries the same
# JS-toggled banner in its site-wide template, so the sentence alone can never
# carry the claim — the page also has to be small enough to plausibly contain
# nothing but that notice. This structural check is the reliable half; the
# wording check is the fragile half.
_JS_SHELL_MAX_BYTES = 32 * 1024

# Explicit "your query matched nothing" phrasing. This is by far the most
# common reason a healthy engine returns an empty result set, and the old code
# had no branch for it at all — so a normal no-hits query was reported as
# engine breakage.
_NO_HITS_RE = re.compile(
    r"(?:could\s*n[o']?t|did\s+not|didn't|unable\s+to)\s+find"
    r"|no\s+(?:results?|matches|hits)\b"
    r"|nothing\s+(?:was\s+)?found"
    r"|0\s+results?\s+found",
    re.IGNORECASE,
)

# Some engines gate their search endpoint behind a hidden form field (an
# anti-scraping token) and redirect the request away when it is missing. The
# pair is scraped from the engine's own form and cached briefly: it is stable
# in practice but must be re-read rather than hard-coded, since the whole
# point of such a token is that the operator can rotate it.
_FORM_TOKEN_TTL = 1800.0
_FORM_TOKEN_CACHE: Dict[str, Tuple[float, Optional[Tuple[str, str]]]] = {}
# Deliberately tighter than `_ONION_TIMEOUT`: this fetch is a small homepage
# and it spends the SAME per-engine deadline the actual search needs, so a
# hanging token fetch must not be able to starve the query it exists to
# enable.
_FORM_TOKEN_TIMEOUT = 12.0

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


# ── Per-engine circuit breaker (2026-08-15) ────────────────────────────
# MEASURED, not guessed. Across one day's live log: torch 5 wins / 0
# failures, torgle 5/0, ahmia 0/6, ahmia-onion 0/9 — every ahmia failure a
# full deadline. Probing the endpoints directly over the agent's own Tor:
#
#   ahmia.fi/                      200 in 0.7s     ← the SITE is up
#   ahmia.fi/search/?q=news        302 in 0.7s     ← needs the form token
#   ahmia.fi/search/?q=…&<token>   504 in 31.4s    ← its SEARCH is broken
#   <ahmia onion>/                 200 in 1.6s
#   <ahmia onion>/search/?q=news   504 in 30.7s
#
# So the token logic is CORRECT and the engine is simply down at the
# backend. Deleting it from the table would be the wrong fix — these
# endpoints rotate constantly and ahmia has been the sanest index when it
# works. What is wrong is paying its full deadline on EVERY search: two
# ahmia entries × ~50s of an ~86s budget, spent to learn nothing, on every
# single dark-web query.
#
# So: skip an engine that has failed `_ENGINE_BREAKER_FAILS` times in a
# row, for `_ENGINE_BREAKER_COOLDOWN`, then let ONE probe through (half
# open). A win resets it immediately. State is per-process and in-memory
# by design — a restart re-probes everything, which is the right default
# for endpoints this volatile.
_ENGINE_BREAKER_FAILS = 3
_ENGINE_BREAKER_COOLDOWN = 900.0     # 15 min
#: {engine_name: (consecutive_failures, opened_at_monotonic)}
_ENGINE_BREAKER: Dict[str, Tuple[int, float]] = {}
# ⚠ NO MODULE-LEVEL "skipped this search" LIST. The first version used
# one, and two searches overlap routinely — `core/agent.py` dispatches a
# tool batch through `asyncio.gather`, and `tool_darkweb_research` runs
# its own searches. Measured: search A skipped two engines and reported
# nothing, while B (which skipped none) was the one that cleared the
# list; and a search that contacted EVERY engine reported "ran NO
# engines… do not reword and retry". Fabricating a confident
# infrastructure diagnosis is worse than the silence it replaced, so the
# skip list travels with the RESULT instead.


def _no_results_error(skipped: List[str], total: int,
                      all_skipped: bool = False) -> str:
    """The zero-results message, honest about what was actually asked.

    R4 CRITICAL: only the ALL-skipped case was handled, and the PARTIAL
    case is the one production lives in — this module's own measurements
    put both ahmia endpoints at 0 wins, so they sit in cooldown while
    torch and torgle carry the search. When those two also come back
    empty, the old text claimed "ZERO results across all onion search
    engines and circuits" (false: half were never contacted) and
    prescribed "drop to 2-4 PLAIN keywords" — blaming the query for an
    infrastructure fact, which is the exact class of lie this change set
    exists to remove.
    """
    # `all_skipped` comes from the fan-out, which compares the engines it
    # ACTUALLY dispatched against the ones it skipped — entry for entry.
    # Re-deriving it here from deduped names vs an entry count is what J3
    # was installed to stop: two engine entries sharing a name made "all
    # skipped" unreachable and the tool said it "asked 1 of 3" when it had
    # asked none.
    if all_skipped or (skipped and total and len(skipped) >= total):
        return (
            "ERROR: dark-web search ran NO engines — every configured "
            f"onion engine ({', '.join(skipped)}) is in a failure "
            "cooldown after repeated errors, so this search contacted "
            "nothing and took no time. This is an INFRASTRUCTURE state, "
            "NOT a statement about your query: do not reword and retry. "
            "Either Tor is down here or the engines are, and they will "
            "be re-probed automatically within "
            f"{_ENGINE_BREAKER_COOLDOWN / 60:.0f} minutes. Fall back to "
            "web_search, or proceed and say dark-web search was "
            "unavailable."
        )
    if skipped:
        return (
            "ERROR: dark-web search returned zero results, but it only "
            f"asked {total - len(skipped)} of {total} engines — "
            f"{', '.join(skipped)} are in a failure cooldown after "
            "repeated errors and were NOT contacted. So this is a WEAK "
            "negative: it does not mean no onion index has your query. "
            "Rewording is unlikely to help while the index coverage is "
            "reduced; prefer web_search, or proceed and say dark-web "
            "coverage was partial. The skipped engines are re-probed "
            f"automatically within {_ENGINE_BREAKER_COOLDOWN / 60:.0f} "
            "minutes."
        )
    return _NO_RESULTS_ERROR


# `_all_engines_skipped_error` was removed here (R5): once
# `_no_results_error` took the all/partial/full decision it had
# zero call sites, and a second copy of the same message is how
# the two callers drifted apart in the first place.


def _narrowed_header(skipped: List[str]) -> str:
    return (
        f"\n[⚠ NARROWED: {', '.join(skipped)} skipped — in a failure "
        f"cooldown, not consulted. Cross-engine corroboration is weaker "
        f"than usual, so treat the ordering as discovery order.]"
    )


class _BreakerSkipped(list):
    """An empty result that also says WHICH engine was never contacted.

    A list subclass on purpose: every existing consumer keeps treating it
    as the empty list it is (`if not res`, `res.extend(...)`, the
    `isinstance(res, list)` guard in the gather), so nothing downstream
    needs to change — but the fan-out can distinguish "asked and found
    nothing" from "never asked", which is the whole point.
    """

    def __init__(self, engine_name: str):
        super().__init__()
        self.engine_name = engine_name


def _breaker_should_skip(name: str) -> bool:
    """True when this engine is in an OPEN breaker window."""
    if os.environ.get("GHOST_ONION_BREAKER", "1") != "1":
        return False
    fails, opened = _ENGINE_BREAKER.get(name, (0, 0.0))
    if fails < _ENGINE_BREAKER_FAILS:
        return False
    if (time.monotonic() - opened) >= _ENGINE_BREAKER_COOLDOWN:
        # Half-open: let exactly one probe through. Re-arm the clock so a
        # failing probe does not leak a second one immediately after.
        _ENGINE_BREAKER[name] = (fails, time.monotonic())
        return False
    return True


def _breaker_record(name: str, won: bool) -> None:
    """A win clears the breaker; a failure advances it toward open."""
    if os.environ.get("GHOST_ONION_BREAKER", "1") != "1":
        # Read here too. Otherwise a disabled breaker still counts and
        # still announces "skipping it for 15 min" — while
        # `_breaker_should_skip` returns False. An instrument that lies.
        return
    if won:
        if name in _ENGINE_BREAKER:
            _ENGINE_BREAKER.pop(name, None)
            pretty_log("Darkweb Engine", f"{name}: recovered — breaker cleared",
                       icon=Icons.TOOL_SEARCH)
        return
    fails, opened = _ENGINE_BREAKER.get(name, (0, 0.0))
    fails += 1
    # Stamp `opened` on the transition so the cooldown measures from the
    # moment it opened, not from the first failure of the streak.
    if fails == _ENGINE_BREAKER_FAILS:
        opened = time.monotonic()
        pretty_log(
            "Darkweb Engine",
            f"{name}: {fails} consecutive failures — skipping it for "
            f"{_ENGINE_BREAKER_COOLDOWN / 60:.0f} min (one probe after "
            f"that). Set GHOST_ONION_BREAKER=0 to disable.",
            level="WARNING", icon=Icons.WARN,
        )
    _ENGINE_BREAKER[name] = (fails, opened)

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
    name (i.e. treated as an independent index). It may also carry a
    ``form_token_from`` URL if the engine gates its search endpoint behind a
    hidden form field (see `_form_token`)."""
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
            entry = {"name": name, "url": url, "index": str(e.get("index") or name)}
            token_from = e.get("form_token_from")
            if token_from:
                entry["form_token_from"] = str(token_from)
            engines.append(entry)
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


def _parse_form_token(html: str) -> Optional[Tuple[str, str]]:
    """Extract the (name, value) of a search form's hidden anti-scraping field.

    Prefers the hidden input that lives inside a form whose action looks like
    a search endpoint, so we can't pick up an unrelated hidden field (a CSRF
    token on a newsletter box, say). Falls back to the first hidden input on
    the page when no form context is available. Returns ``None`` when the page
    carries no hidden field — which is the normal case for every engine that
    doesn't gate its endpoint, so callers must treat ``None`` as "no token
    needed", not as an error."""
    if not html:
        return None

    def _from_input(tag) -> Optional[Tuple[str, str]]:
        name, value = tag.get("name"), tag.get("value")
        # A nameless or valueless hidden input carries nothing we can send.
        if not name or value is None:
            return None
        return str(name), str(value)

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        forms = [f for f in soup.find_all("form")
                 if "search" in (f.get("action", "") + " " + " ".join(
                     f.get("class") or []) + " " + (f.get("id") or "")).lower()]
        for form in forms or soup.find_all("form"):
            for tag in form.find_all("input", attrs={"type": "hidden"}):
                found = _from_input(tag)
                if found:
                    return found
        for tag in soup.find_all("input", attrs={"type": "hidden"}):
            found = _from_input(tag)
            if found:
                return found
        return None
    except Exception:
        # BeautifulSoup unavailable or blew up — regex fallback, attribute
        # order agnostic (name may precede or follow value).
        for m in re.finditer(r"<input[^>]*type=[\"']hidden[\"'][^>]*>", html, re.I):
            tag = m.group(0)
            name = re.search(r"name=[\"']([^\"']+)[\"']", tag, re.I)
            value = re.search(r"value=[\"']([^\"']*)[\"']", tag, re.I)
            if name and value:
                return name.group(1), value.group(1)
        return None


async def _form_token(token_url: str, proxy: Optional[str],
                      label: str = "engine") -> Optional[Tuple[str, str]]:
    """Fetch and cache the hidden form token an engine requires, or ``None``.

    Ahmia (measured live 2026-07-28) 302-redirects ``/search/?q=…`` to ``/``
    unless the query string also carries the hidden field from its search
    form; with it, the same endpoint returns a full results page. The token is
    stateless — no cookie or session is involved — so a scraped pair can be
    reused across circuits and requests, which is what makes caching it sound.

    Fails OPEN: on any error the engine is queried without a token, i.e.
    exactly the pre-token behaviour.

    ONLY a successful read is cached. A ``None`` from a page we actually
    fetched means "this engine needs no token" and is worth remembering; a
    ``None`` because Tor timed out means nothing, and caching it would take a
    transient blip and turn it into 30 minutes of an engine silently querying
    without the token it requires — i.e. 30 minutes of zero results, which is
    the very failure this function exists to end. Same discipline as the rest
    of the project: never record a failed measurement as a neutral value."""
    now = time.monotonic()
    cached = _FORM_TOKEN_CACHE.get(token_url)
    if cached and (now - cached[0]) < _FORM_TOKEN_TTL:
        return cached[1]
    try:
        status, body = await _fetch_raw_html(token_url, proxy, _FORM_TOKEN_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        status, body = None, ""
        pretty_log(
            "Darkweb Engine Token",
            f"{label}: could not read the search-form token "
            f"({type(exc).__name__}) — querying without it",
            level="WARNING", icon=Icons.WARN,
        )
    if status != 200 or not body:
        if status is not None:
            pretty_log(
                "Darkweb Engine Token",
                f"{label}: search-form page returned {status} — querying "
                "without a token (not cached; will retry)",
                level="WARNING", icon=Icons.WARN,
            )
        return None
    token = _parse_form_token(body)
    _FORM_TOKEN_CACHE[token_url] = (now, token)
    return token


def _invalidate_form_token(token_url: str) -> None:
    """Drop a cached token so the next attempt re-scrapes it.

    Called when a query still gets redirected away: the most likely cause is
    that the engine rotated the token under us, and a fresh scrape on the next
    (already circuit-rotated) attempt is the self-heal."""
    _FORM_TOKEN_CACHE.pop(token_url, None)


def _apply_form_token(url: str, token: Optional[Tuple[str, str]]) -> str:
    """Append a scraped hidden form field to an already-built query URL.

    Two guards, both learned from how this could fail silently:

    * The token goes before any ``#fragment`` — appended after one it would
      never be sent, so the engine would keep redirecting and we would keep
      re-scraping a token that could not possibly work.
    * A name already present in the query is NOT appended. A hidden field
      named like the engine's own query parameter (``q``) would otherwise be
      appended as a duplicate, and frameworks that take the LAST value would
      see an empty search — which looks like "no hits", never like a bad
      token, so nothing would ever invalidate it."""
    if not token:
        return url
    name, value = token
    head, sep_frag, frag = url.partition("#")
    try:
        if name in parse_qs(urlparse(head).query, keep_blank_values=True):
            return url
    except Exception:
        pass
    sep = "&" if "?" in head else "?"
    head = f"{head}{sep}{quote_plus(name)}={quote_plus(value)}"
    return head + sep_frag + frag


def _diagnose_empty_body(body: str, requested_url: str,
                         final_url: Optional[str]) -> Tuple[str, str]:
    """Explain a 200-with-zero-parseable-results. Returns (kind, message).

    Ordered by how much the evidence actually supports, NOT by how interesting
    the explanation is — these signals are not mutually exclusive, and the
    wrong one is worse than none, because each message ends in different
    operator action. In order:

    1. ``redirected`` — we were served a different location than we asked for.
       Whatever that page says is evidence about a page we never requested.
    2. ``no-hits`` — the engine explicitly says the query matched nothing.
       This is the ordinary case and it is NOT breakage; the old code had no
       branch for it, so a normal empty query was reported as a broken engine.
    3. ``js-only`` — a small shell that demands JavaScript. Requires BOTH the
       wording and the size, because Ahmia's full results page carries a
       JS-toggled banner in its site-wide template: the sentence alone once
       told operators to replace a working engine on a zero-hit query.
    4. ``parser`` — none of the above; the format may have drifted.
    """
    body = body or ""
    if final_url and _redirected_away(requested_url, final_url):
        return "redirected", (
            f"redirected to {_display_target(final_url)} and served that page "
            "instead of results — the search endpoint changed or is rejecting "
            "the query (a missing/rotated form token does exactly this)"
        )
    if _NO_HITS_RE.search(body):
        return "no-hits", (
            "the engine reports no matches for this query — it answered "
            "normally, so this is a query result, not a broken engine"
        )
    if _JS_ONLY_RE.search(body) and len(body) <= _JS_SHELL_MAX_BYTES:
        # When the redirect chain is UNKNOWN we cannot rule out that this
        # notice belongs to some other page we were bounced to — which is
        # precisely how the old detector lied. Say what we don't know rather
        # than repeat a confident claim we can't support.
        if not final_url:
            return "js-only", (
                "page says it needs JavaScript, but the redirect chain is "
                "unknown, so this may be a page we were bounced to rather "
                "than the engine itself — check where the search URL leads "
                "before changing GHOST_ONION_ENGINES"
            )
        return "js-only", (
            f"{len(body)} bytes with no results and a notice demanding "
            "JavaScript — a raw-HTML fetch cannot read this engine (set "
            "GHOST_ONION_ENGINES to a working engine)"
        )
    return "parser", (
        f"{len(body)} bytes served, but the parser found no onion "
        "links — the engine's result format may have drifted"
    )


def _norm_host(netloc: str) -> str:
    """Host in a comparable form: lowercased, credentials and default port
    stripped, ``www.`` folded away. Without this, three redirects that change
    nothing — ``www.`` stripping, ``:80``/``:443`` normalisation, a case
    difference — read as "the engine sent us somewhere else" and cost a token
    eviction plus a wasted retry on every query."""
    host = (netloc or "").lower()
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    for suffix in (":80", ":443"):
        if host.endswith(suffix):
            host = host[: -len(suffix)]
            break
    if host.startswith("www."):
        host = host[4:]
    return host


def _redirected_away(requested_url: str, final_url: str) -> bool:
    """True when the response came from a materially different location.

    Compares normalised host + path only: engines legitimately rewrite the
    query string (reordering params, adding defaults), and neither a scheme
    upgrade nor a cosmetic host rewrite is a redirect *away*. A changed PATH
    means we were served something other than what we asked for.

    Deliberately conservative — a false positive here evicts a good token and
    buys a wasted retry, so when in doubt this says "not a redirect"."""
    try:
        a, b = urlparse(requested_url), urlparse(final_url)
    except Exception:
        return False
    if not b.netloc:
        return False
    return ((_norm_host(a.netloc), a.path.rstrip("/").lower())
            != (_norm_host(b.netloc), b.path.rstrip("/").lower()))


def _display_target(url: str) -> str:
    """Host + path of a redirect target, for the operator's log.

    The HOST is the load-bearing part — "we ended up on a different site" and
    "we ended up on this site's homepage" call for completely different
    responses, and a path alone can't tell them apart. Onion hostnames are
    scrubbed downstream by the log redactor, so including it leaks nothing;
    the query string is dropped because it carries the search terms."""
    try:
        pr = urlparse(url)
        return f"{pr.netloc}{pr.path}" if pr.netloc else (pr.path or "/")
    except Exception:
        return "(unparseable URL)"


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


def _record_final_url(meta: Optional[Dict[str, Any]], response: Any) -> None:
    """Stash the URL a response was actually served from into ``meta``.

    Both HTTP clients expose ``.url``, but as their own URL objects rather
    than str. Never raises — losing this datum must degrade the diagnosis,
    not the fetch."""
    if meta is None:
        return
    try:
        url = getattr(response, "url", None)
        if url:
            meta["final_url"] = str(url)
    except Exception:
        pass


async def _fetch_raw_html(url: str, proxy: Optional[str], timeout: float,
                          *, meta: Optional[Dict[str, Any]] = None) -> Tuple[Optional[int], str]:
    """Fetch RAW HTML (tags intact) through the Tor SOCKS proxy.

    Unlike `helper_fetch_url_content`, this does NOT strip tags — the search
    phase needs the markup to parse out result links — and it HONOURS the
    passed proxy rather than reading it from the environment. Uses curl_cffi
    when present (TLS-impersonating, the project default) and falls back to
    httpx. The body is size-capped (`_cap_body`) and the blocking request runs
    on a dedicated bounded pool so a post-deadline lingering fetch can't
    exhaust the shared executor. Returns (status_code, body); (None, "") on
    transport failure.

    Redirects are followed, so the status alone cannot tell you whether the
    body came from the URL you asked for. Pass a ``meta`` dict to receive the
    ``final_url`` the body was actually served from — without it, a dead
    endpoint that 302s to a working homepage is indistinguishable from a live
    endpoint serving an unparseable page, which is exactly how the Ahmia
    breakage stayed misdiagnosed for weeks."""

    # Read at most this many BYTES off the wire regardless of Content-Length.
    # An untrusted onion engine can send a chunked body with no (or a lying)
    # Content-Length; reading `r.text` (the whole body into RAM) before
    # `_cap_body` could truncate it OOMs the host. Stream and stop at the cap.
    _STREAM_LIMIT = _MAX_ONION_BODY_BYTES + 4096

    def _decode(buf: bytes, content_type: Optional[str] = None) -> str:
        # Honour the declared charset before falling back to utf-8. Onion
        # sites carry a large fraction of Cyrillic/CJK content; force-utf-8
        # decoding turned those bodies into mojibake (and BeautifulSoup,
        # handed an already-decoded str, could no longer sniff the real
        # charset from the meta/Content-Type). Try declared → utf-8 →
        # latin-1 (never raises).
        _charset = None
        if content_type and "charset=" in content_type.lower():
            _charset = content_type.lower().split("charset=", 1)[1].split(";")[0].strip() or None
        for _enc in (_charset, "utf-8"):
            if not _enc:
                continue
            try:
                return buf.decode(_enc)
            except (LookupError, UnicodeDecodeError):
                continue
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
                _record_final_url(meta, r)
                return _cap_body(r.status_code, r.headers.get("content-type"),
                                 r.headers.get("content-length"), _decode(bytes(buf), r.headers.get("content-type")))
        except ImportError:
            import httpx

            with httpx.Client(proxy=proxy, timeout=timeout, follow_redirects=True) as c:
                with c.stream("GET", url, headers=headers) as r:
                    buf = bytearray()
                    for chunk in r.iter_bytes():
                        buf.extend(chunk)
                        if len(buf) >= _STREAM_LIMIT:
                            break
                    _record_final_url(meta, r)
                    return _cap_body(r.status_code, r.headers.get("content-type"),
                                     r.headers.get("content-length"), _decode(bytes(buf), r.headers.get("content-type")))

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
    # resolve=False: this ALWAYS fetches over Tor, so a host-side getaddrinfo
    # would (a) leak the target — and for a .onion, leak WHICH hidden service
    # is being visited, the exact anonymity break dark-web research must
    # avoid — and (b) fail anyway (.onion doesn't resolve via the host
    # resolver; onion addressing lives at the Tor SOCKS layer). The literal-
    # internal-IP string check still runs.
    reason = url_ssrf_reason(url, resolve=False)
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
        base_url = engine["url"].format(q=quote_plus(query))
        token_from = engine.get("form_token_from")
        for attempt in range(2):
            proxy = _proxy_for_attempt(tor_proxy, f"{engine['name']}:{query}", attempt)
            # Re-read per attempt: a retry after a redirect-away has dropped
            # the cached token, so this is where a rotated one gets picked up.
            url = base_url
            token_applied = False
            if token_from:
                url = _apply_form_token(
                    base_url,
                    await _form_token(token_from, proxy, engine["name"]))
                token_applied = url != base_url
            try:
                meta: Dict[str, Any] = {}
                status, body = await _fetch_raw_html(
                    url, proxy, _ONION_TIMEOUT, meta=meta)
                if status == 200 and body:
                    parsed = _parse_onion_results(body, exclude_hosts)
                    if parsed:
                        pretty_log(
                            "Darkweb Engine",
                            f"{engine['name']}: {len(parsed)} onion result(s)",
                            icon=Icons.TOOL_DARKWEB,
                        )
                        return parsed
                    # 200 but ZERO parseable results. This was silent — a
                    # broken engine looked identical to "no onion index has
                    # this query," so the tool blamed the query. Name the
                    # actual cause instead; see `_diagnose_empty_body` for
                    # why the ordering of those causes matters.
                    kind, detail = _diagnose_empty_body(
                        body, url, meta.get("final_url"))
                    if (kind == "redirected" and token_applied
                            and attempt == 0):
                        # A token we DID send still got bounced: most likely
                        # rotated, so re-scrape on the retry. Gated on
                        # attempt 0 (a re-scrape after the last attempt only
                        # bills the next search) and on having actually sent
                        # one — otherwise any redirect, including a captcha
                        # or exit-ban page, throws away a good token.
                        _invalidate_form_token(token_from)
                    pretty_log(
                        "Darkweb Engine Empty",
                        f"{engine['name']}: HTTP 200 but 0 parseable results "
                        f"— {detail}",
                        level="WARNING", icon=Icons.WARN,
                    )
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

    # The deadline was sized for ONE full search attempt plus a short window
    # for a fast second circuit. A token engine spends a fetch of its own
    # inside that budget, so without this extension a cold-cache attempt
    # (12s token + 30s search = 42s) would be guillotined at 38s and return
    # nothing — the deadline killing searches that were about to succeed, the
    # same shape as the mojeek-timeout bug in search.py. Extend by the token
    # budget rather than shrinking the search timeout: a shortened search
    # timeout guillotines slow-but-alive engines, which is the failure we
    # already learned not to re-introduce. Scaled by min() so a test that
    # shrinks the deadline still gets a proportionally small total.
    # Breaker check BEFORE any work — the whole point is to not pay the
    # deadline for an engine measured to be failing.
    if _breaker_should_skip(engine["name"]):
        logger.debug("onion engine %s skipped (breaker open)", engine["name"])
        return _BreakerSkipped(engine["name"])
    deadline = _ONION_ENGINE_DEADLINE
    if engine.get("form_token_from"):
        deadline += min(_FORM_TOKEN_TIMEOUT, _ONION_ENGINE_DEADLINE)
    try:
        _res = await asyncio.wait_for(_attempts(), timeout=deadline)
        # "Won" means RESULTS, not "did not raise": an engine that returns
        # an empty list every time is failing at the only thing it is for,
        # and 302-to-homepage (ahmia's pre-token shape) is exactly that.
        _breaker_record(engine["name"], bool(_res))
        return _res
    except asyncio.TimeoutError:
        _breaker_record(engine["name"], False)
        # The underlying fetch runs in a worker thread (curl_cffi/httpx has
        # its own timeout), so it isn't force-killed here — but cancelling the
        # await lets the gather proceed without waiting on this engine. The
        # dedicated `_ONION_EXECUTOR` keeps that lingering thread off the
        # shared pool.
        pretty_log(
            "Darkweb Engine Error",
            f"{engine['name']}: exceeded {deadline:.0f}s deadline — skipped",
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


async def _darkweb_search_raw(
    query: str, tor_proxy: str, max_results: int = 12
) -> Tuple[List[Dict[str, Any]], List[str], bool, int]:
    """Core fan-out: query every engine concurrently, merge + rank results.

    Ranking favours onions surfaced by MORE THAN ONE INDEPENDENT INDEX
    (corroboration is the only cheap relevance signal we have for unindexed
    hidden services), preserving discovery order within a tier. Endpoints that
    share an ``index`` (e.g. Ahmia's clearnet + onion mirrors) count once, so
    reaching one index over two transports is NOT mistaken for independent
    agreement. Returns ranked result dicts, each carrying the engine names and
    indexes that surfaced it, PLUS the names of engines the circuit
    breaker skipped and whether that was ALL of them — returned rather
    than stashed in a module global, because searches overlap."""
    engines = _load_engines()
    exclude = _engine_onion_hosts(engines)
    per_engine = await asyncio.gather(
        *[_query_engine(e, query, tor_proxy, exclude) for e in engines],
        return_exceptions=True,
    )

    # J3: the denominator is the list THIS call searched, not a second
    # `_load_engines()` read — that re-reads GHOST_ONION_ENGINES (so a
    # mid-process change makes "all skipped" arithmetic wrong) and counts
    # entries while the skip list counts names (two entries sharing a name
    # made "all skipped" unreachable).
    skipped = [r.engine_name for r in per_engine
               if isinstance(r, _BreakerSkipped)]
    all_skipped = bool(engines) and len(skipped) >= len(engines)

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
    return (ranked[:max_results], sorted(set(skipped)), all_skipped,
            len(engines))


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

    cache_key = "onion::" + _norm_cache_key(query)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ranked, _skipped, _all_skipped, _total = await _darkweb_search_raw(
        query, tor_proxy, max_results=max_results)
    if not ranked:
        return _no_results_error(_skipped, _total, _all_skipped)

    reached = sorted({e for r in ranked for e in r.get("engines", [])})
    header = f"[Dark-web search — onion results, engines reached: {', '.join(reached)}]"
    cacheable = header + "\n\n" + _format_results(ranked)
    # R2 M6: cache WITHOUT the NARROWED banner. It describes a transient
    # breaker state, and baking it into a 5-minute cache entry kept
    # telling the operator an engine was "not consulted" long after it had
    # recovered — a stale claim about infrastructure, which is the class
    # of lie this whole change set exists to remove.
    _cache_put(cache_key, cacheable)
    if _skipped:
        return header + _narrowed_header(_skipped) + "\n\n" + \
            _format_results(ranked)
    return cacheable


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

    ranked, _skipped, _all_skipped, _total = await _darkweb_search_raw(
        query, tor_proxy, max_results=max_sources)
    if not ranked:
        # J1: this caller kept blaming the query after the sibling was
        # fixed — and it is the follow-up `darkweb_search`'s own tool
        # description recommends, so it is the one the model reaches for
        # after a thin result set.
        return _no_results_error(_skipped, _total, _all_skipped)

    urls = [r["url"] for r in ranked][:max_sources]

    # ⚠ WHAT `max_context` ACTUALLY BOUNDS. It is the MAIN model's window, and
    # the assembled report is what gets read back into it — so this is a
    # ceiling on each source's SHARE of that report, nothing more. It is NOT
    # the worker's limit: the comment that stood here claimed to size "to the
    # worker's context window" while reading the 35B's 240,000, so it pinned
    # to its own 40k ceiling on every call and never constrained anything. The
    # worker's limit now comes from `plan_distill`, which reads that node's
    # measured throughput. See core/node_throughput.py.
    _report_share_chars = max(
        _MIN_DISTILL_CHARS,
        int(int(max_context) * CHARS_PER_TOKEN * 0.4) // max(1, len(urls)))

    sem = asyncio.Semaphore(2)
    _DISTILL_FANOUT = 2      # == the semaphore width above
    # One snapshot per call, from the single definition in search.py.
    _RESEARCH_PHASE_TIMEOUT_S = _budgets._RESEARCH_PHASE_TIMEOUT_S
    # ⚠ NOT THE CLEARNET FLOOR. That one is derived from the 22s clearnet fetch
    # attempt, but this path's own fetch `wait_for` is `_ONION_PAGE_TIMEOUT + 5`
    # = 40s — so an onion admitted with a budget in [24, 40) is killed
    # mid-fetch by the outer wait_for and reported as "per-URL timeout
    # exceeded". The floor's stated contract ("it can at least come back with a
    # page") was false here by up to 16s.
    _MIN_URL_BUDGET_S = _ONION_PAGE_TIMEOUT + 7
    _QUEUE_ALLOWANCE_S = _budgets._QUEUE_ALLOWANCE_S
    _WEB_SUMMARY_TIMEOUT_S = _budgets._WEB_SUMMARY_TIMEOUT_S
    # ⚠ ONION FETCHES ARE THE SLOW HALF, so the ceiling must leave room for a
    # distill after one. At `_ONION_PAGE_TIMEOUT + 10` (45s) the fetch alone
    # may take 40s and the distiller was left ~3s — it could essentially never
    # buy a distillation, and every onion silently returned raw HTML (review,
    # CONFIRMED). +25 leaves a working margin at measured worker rates.
    # ⚠ +25 DID NOT LEAVE A WORKING MARGIN — measured: worst fetch 40 + the 13s
    # of fixed per-url overhead (2s reserve + queue allowance + safety margin)
    # + a floor distill of ~15-20s is 68-73s against a 60s ceiling, so a SLOW
    # onion could never be distilled — the exact defect the +10 -> +25 change
    # was written to fix. Sized from the arithmetic now, not from a guess.
    _ONION_URL_CEILING_S = _ONION_PAGE_TIMEOUT + 45
    # The whole fetch+distill phase gets ONE clock; each onion is issued its
    # budget when it actually starts (see `_bounded`).
    _phase_deadline = time.monotonic() + _RESEARCH_PHASE_TIMEOUT_S

    async def _fetch_with_timeout(url: str) -> str:
        try:
            # ⚠ COERCE, as the clearnet sibling does. This returned the awaited
            # value verbatim while `process_url` guards it with
            # `isinstance(text, str)` and then indexes it seven lines later —
            # one `return None` from dropping the onion. The `except Exception`
            # net in `_bounded` now REPORTS such a source instead of losing it,
            # but reporting an internal error is not the same as keeping the
            # page: coercing here is what preserves the raw-text fallback.
            _res = await asyncio.wait_for(
                _fetch_onion_text(url, tor_proxy), timeout=_ONION_PAGE_TIMEOUT + 5
            )
            return _res if isinstance(_res, str) else str(_res)
        except asyncio.TimeoutError:
            return f"Error: Fetch of {url} timed out after {_ONION_PAGE_TIMEOUT}s"
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}"

    async def process_url(url: str, budget_s: float) -> str:
        # ⚠ The distill budget is what is LEFT, not a constant. This module
        # imported search.py's 45s — but search runs under a 55s outer
        # deadline and this one under `_ONION_PAGE_TIMEOUT + 10` = 45s, of
        # which the Tor fetch alone may consume 40s. So the summary was
        # handed the entire outer deadline with the fetch already spent, and
        # a 5s fetch was enough to LOSE THE URL (R6 lens A measured exactly
        # that). An onion fetch is the slow half here; size the distill
        # against the clock, not against the clearnet sibling's number.
        #
        # `budget_s` is issued by `_bounded` AFTER the semaphore is acquired,
        # so it is time this URL actually HAS rather than time already spent
        # queueing — the clearnet sibling's req-08766aa1 defect, which this
        # function shared verbatim.
        _url_deadline = time.monotonic() + budget_s
        url = str(url)
        short_url = (url[:35] + "..") if len(url) > 35 else url
        pretty_log("Parsing Onion", url, icon=Icons.TOOL_DARKWEB)
        text = await _fetch_with_timeout(url)

        # ⚠ Same boundary as the clearnet sibling: a failed fetch is an
        # error STRING, and feeding it to a worker prompted "extract
        # the hard facts, and if none are found say so" turns "this
        # onion did not answer" into "this onion contains nothing
        # relevant" — positive negative evidence, manufactured from a
        # timeout. It also made the cache gate below inoperative,
        # since the distiller's output never starts with "Error:".
        if isinstance(text, str) and text.startswith("Error:"):
            pretty_log("Onion Source Failed",
                       f"{short_url} — {text[:90]}",
                       level="WARNING", icon=Icons.WARN)
            return f"### SOURCE: {url}\n{text}\n"

        # Clean the raw-text fallback: unscrubbed surrogates/control chars
        # from an onion page can crash the downstream C++ JSON parser (what
        # _clean_for_cpp prevents). Built once — both degradation routes
        # below return it.
        raw_preview = (_clean_for_cpp(text[:_RAW_FALLBACK_CHARS])
                       + "\n[...truncated...]\n")

        # ⚠ SIZE THE REQUEST TO WHAT THE NODE CAN FINISH. The arithmetic that
        # stood here derived the char limit from `max_context` — the MAIN
        # model's window (240,000 live), not the worker's — so it pinned to
        # its own 40,000-char ceiling on every call and never constrained
        # anything. On Nova that is ~12,500 prompt tokens at ~300 tok/s: 41s
        # of prefill before the first output token, against a budget of 45s
        # at most and often far less once an onion fetch has been paid for.
        # See core/node_throughput.py for the measurements.
        _summary_budget = 0.0
        plan = None
        if llm_client is not None:
            _summary_budget = min(
                _WEB_SUMMARY_TIMEOUT_S,
                _url_deadline - time.monotonic() - 2.0)
            # ⚠ PLANNING MUST NOT BE ABLE TO LOSE THE URL. A client without
            # `plan_distill` (an older or hand-rolled one) would raise here,
            # escape `process_url`, and be swallowed by
            # `gather(return_exceptions=True)` — dropping the source, which is
            # the precise loss this rewrite exists to remove. Degrade to raw
            # text instead, and say so.
            try:
                plan = llm_client.plan_distill(
                    _summary_budget - _QUEUE_ALLOWANCE_S,
                    max_chars=_report_share_chars,
                    # ⚠ DECLARE THE FAN-OUT. Every URL in a wave is planned
                    # before any of it is in flight, so the node looks idle to
                    # all of them and each plan is sized as if it had the box
                    # to itself. That is how four ~19,936-char plans built at
                    # ~306/32 tok/s ended up running at 115/11 (req b86cdd59).
                    concurrency=_DISTILL_FANOUT)
            except Exception as _plan_exc:                      # noqa: BLE001
                pretty_log("Summary Degraded",
                           f"{short_url}: could not size the distill "
                           f"({type(_plan_exc).__name__}) — falling back to "
                           f"raw page text", level="WARNING", icon=Icons.WARN)
                plan = None

        # ⚠ DECLINE, DO NOT POST. The predecessor floored the budget at
        # `max(5.0, ...)`, which does not create time — it just guaranteed
        # that an already-spent clock still posted a request, burning a Tor
        # circuit's worth of worker slot on work that could not finish.
        try:
            _feasible = bool(plan is not None and plan.feasible)
        except Exception:                                       # noqa: BLE001
            _feasible = False          # `getattr(..., False)` swallows only
                                       # AttributeError; a property that
                                       # raises anything else got through.
        if not _feasible:
            if plan is not None:
                log_distill_plan(short_url, plan)
            return f"### SOURCE: {url}\n{raw_preview}\n"

        # ⚠ CONSUMING THE PLAN MUST NOT BE ABLE TO LOSE THE URL EITHER.
        # `text[:plan.char_limit]` and the `max_tokens` field are the two
        # places a malformed plan reaches real work; both raise OUTSIDE the
        # try below, so they escape `process_url` and the source is dropped by
        # `gather(return_exceptions=True)`. Coerce once, here, and degrade
        # like any other failure. Downstream code uses these ints, never the
        # attributes — which also keeps the log's format specs total.
        try:
            _chars = int(plan.char_limit)
            _tokens = int(plan.max_tokens)
            if _chars < 1 or _tokens < 1:
                raise ValueError(f"non-positive plan {_chars}/{_tokens}")
        except Exception as _bad_plan:                          # noqa: BLE001
            pretty_log("Summary Degraded",
                       f"{short_url}: unusable distill plan "
                       f"({type(_bad_plan).__name__}) — falling back to raw "
                       f"page text", level="WARNING", icon=Icons.WARN)
            return f"### SOURCE: {url}\n{raw_preview}\n"

        safe_text = _clean_for_cpp(text[:_chars])
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
            "max_tokens": _tokens,
        }
        try:
            summary_data = await llm_client.chat_completion(
                payload, use_worker=True,
                # ⚠ The fallback is FREE — the `except`
                # two lines below keeps the raw page text.
                # Without this, a worker outage sent EVERY
                # url's distillation to the main 35B,
                # foreground, serialised behind the main
                # lock, each racing a 55s per-url deadline —
                # one research call becoming N unbounded 35B
                # generations (LLM review R3 lens B, B2).
                off_main_only=True,
                # ⚠ BOUND THE TOTAL, NOT JUST THE POST. This whole
                # coroutine runs under an outer per-url deadline; if
                # the pool budget can exceed it, a saturated node
                # makes the outer `wait_for` cancel fetch AND distill
                # and the URL is LOST — strictly worse than the
                # raw-text degradation sitting in the except block
                # below. R2 capped this implicitly via
                # `min(_slot_wait, timeout)`; R4 removed that cap
                # without replacing it, which took the budget to the
                # 90s operator ceiling, 1.6x the outer deadline
                # (R5 lens A). Now stated explicitly, as R4's own
                # doctrine requires.
                timeout=_summary_budget,
                # Bounded, and equal to what the plan reserved for it.
                slot_wait=_QUEUE_ALLOWANCE_S,
                total_budget=_summary_budget,
                task_label="web summary")
            try:
                _sized = plan.describe()
            except Exception:                                   # noqa: BLE001
                _sized = "sizing unavailable"
            pretty_log("Worker Compute",
                       f"Distilled facts from {short_url} — {_sized}",
                       icon=Icons.TOOL_DEEP)
            try:
                # The last hand-tuned constant in the sizing path, measured
                # instead: we know the chars we sent, llama.cpp reports the
                # tokens they became.
                llm_client.note_distill_density(len(safe_text), summary_data)
            except Exception:                                   # noqa: BLE001
                pass
            preview = "[EDGE EXTRACTED FACTS]:\n" + (summary_data["choices"][0]["message"].get("content") or "").strip()
        except Exception as _sum_exc:
            # ⚠ SAY SO. This was a bare `except Exception:` with no
            # log at all, so every degradation here was invisible:
            # the source silently reverts to raw truncated page text,
            # which then feeds `fact_check` as if it were distilled
            # evidence (R4 lens B).
            # ⚠ THE DEGRADATION LOG MUST NOT DESTROY THE DEGRADATION. This
            # block exists to SAVE the source as raw text; anything raised
            # while describing the failure escapes `process_url`, is swallowed
            # by `gather(return_exceptions=True)`, and drops the URL — exactly
            # the loss this rewrite removed. Caught for real by the suite: a
            # `{plan.char_limit:,}` format spec against a non-numeric plan
            # took out two previously-passing tests. Preview is assigned
            # FIRST, and the telemetry cannot reach it.
            preview = raw_preview
            try:
                # ⚠ AFTER the assignment, and inside the guard. This import is
                # telemetry-only, yet it was the FIRST statement in the
                # handler — so a renamed helper, a partially-initialised
                # `core.llm` under reload, or a first-time import raised here
                # and the source was dropped by the very block written to save
                # it (review, CONFIRMED by injection).
                from ..core.llm import _err_text
                pretty_log(
                    "Summary Degraded",
                    f"{short_url}: {_err_text(_sum_exc)} — planned "
                    f"{_chars:,} chars/{_tokens} tok in "
                    f"{_summary_budget:.0f}s — falling back to raw page text",
                    level="WARNING", icon=Icons.WARN)
            except Exception:                                   # noqa: BLE001
                pretty_log("Summary Degraded",
                           f"{short_url}: falling back to raw page text",
                           level="WARNING", icon=Icons.WARN)
        return f"### SOURCE: {url}\n{preview}\n"

    async def _bounded(url: str) -> str:
        # ⚠ The per-URL clock starts when the URL starts, not at `gather` —
        # see the clearnet sibling in search.py for the measurement. With
        # `Semaphore(2)` here, the 3rd and later onions were the starved ones.
        async with sem:
            budget = min(_ONION_URL_CEILING_S,
                         _phase_deadline - time.monotonic())
            if budget < _MIN_URL_BUDGET_S:
                return (f"### SOURCE: {url}\nError: research phase deadline "
                        f"({_RESEARCH_PHASE_TIMEOUT_S:.0f}s) reached before "
                        f"this source could be fetched\n")
            try:
                return await asyncio.wait_for(process_url(url, budget),
                                              timeout=budget)
            except asyncio.TimeoutError:
                # Say the BUDGET. Without it an onion killed on a starved
                # window reads exactly like one that used its full ceiling —
                # the confusion req 08766aa1 turned on.
                return (f"### SOURCE: {url}\nError: per-URL timeout exceeded "
                        f"({budget:.0f}s)\n")
            except Exception as _url_exc:                        # noqa: BLE001
                # ⚠ THE NET. Anything that escapes reaches
                # `gather(return_exceptions=True)` and is filtered out by
                # `isinstance(c, str)` — a SOURCE SILENTLY DROPPED, invisible
                # to operator and model alike and indistinguishable from "the
                # topic has nothing". Guarding each hazard individually has
                # now failed twice (an escaping `raise`, then a format spec
                # inside the degradation handler), so the CLASS is closed here
                # rather than instance by instance: a failed log sink, a
                # renamed helper, a hostile plan object, a non-str url — every
                # one becomes a REPORTED failure.
                #
                # ⚠ `Exception`, never `BaseException` and never a bare
                # `except`: on 3.10 `asyncio.CancelledError` is a
                # BaseException, and swallowing it here would break shutdown
                # and the enclosing `wait_for`.
                try:
                    pretty_log(
                        "Research Source Failed",
                        f"{str(url)[:60]} — internal error "
                        f"{type(_url_exc).__name__}: reported as a failed "
                        f"source rather than silently dropped",
                        level="WARNING", icon=Icons.WARN)
                except Exception:                               # noqa: BLE001
                    pass
                return (f"### SOURCE: {url}\nError: internal error while "
                        f"processing this source "
                        f"({type(_url_exc).__name__})\n")

    tasks = [_bounded(u) for u in urls]
    page_contents = await asyncio.gather(*tasks, return_exceptions=True)
    valid_contents = [c for c in page_contents if isinstance(c, str)]
    # ⚠ PORTED FROM THE CLEARNET SIBLING. Without a banner a partly-failed
    # onion run renders as a confident SHORT report and an all-failed one as
    # an EMPTY report under a confident header — positive negative evidence,
    # the precise failure the `Onion Source Failed` early return exists to
    # prevent. `_lost` also catches anything `gather` returned as an
    # exception. `_degraded` counts sources that fell back to raw truncated
    # HTML: neither failed nor distilled, and previously reported as neither.
    _failed = [c for c in valid_contents if "\nError:" in c]
    _lost = len(urls) - len(valid_contents)
    _degraded = [c for c in valid_contents
                 if "\nError:" not in c
                 and "[EDGE EXTRACTED FACTS]" not in c] if llm_client else []
    _banner = ""
    if _failed or _lost:
        _banner = (
            f"[⚠ SOURCE FAILURES: {len(_failed) + _lost} of {len(urls)} "
            f"hidden services could not be fetched (timeout, down, or "
            f"unreachable). The sections below cover only the "
            f"{len(valid_contents) - len(_failed)} that loaded. This is a "
            f"COVERAGE limit, not evidence of absence — do not conclude a "
            f"claim is unsupported because a failed source did not support "
            f"it.]\n\n")
    if _degraded:
        _banner += (
            f"[⚠ {len(_degraded)} of {len(urls)} hidden services could not be "
            f"distilled in the time available and appear below as RAW, "
            f"TRUNCATED page text rather than extracted facts. Treat those "
            f"sections as unfiltered page dumps.]\n\n")
    full_report = _banner + "\n\n".join(valid_contents)

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
    # ⚠ A PHASE-TRUNCATED RUN IS TRANSIENT — DO NOT CACHE IT. One surviving
    # block is enough to satisfy `_source_succeeded`, so a run where the phase
    # clock refused most onions would be served back for 300s as if it were
    # the topic's real coverage. Same class as the `_narrowed_header` rule
    # this function already follows.
    _phase_truncated = any("research phase deadline" in c
                           for c in valid_contents)

    def _source_succeeded(block: str) -> bool:
        # block is "### SOURCE: <url>\n<preview>\n". The preview begins
        # with "Error:" only when the fetch failed — which was TRUE ONLY
        # WITHOUT an llm_client, i.e. in the configuration that never runs
        # live: with one wired (registry.py always wires one) the distiller
        # rewrote the error into "[EDGE EXTRACTED FACTS]: no relevant
        # information", so this gate returned True for an all-errors run
        # and cached it for 300s. The early return in `process_url` above
        # is what makes this test mean what it says.
        parts = block.split("\n", 1)
        preview = parts[1].strip() if len(parts) > 1 else ""
        return bool(preview) and not preview.startswith("Error:")

    if not _phase_truncated and any(_source_succeeded(c)
                                    for c in valid_contents):
        _cache_put(cache_key, result)
    # R3 MAJOR: research never emitted the NARROWED banner — and it is the
    # tool that depends MOST on cross-engine corroboration, since its
    # ranking picks which onions get deep-read and synthesised into a
    # report. Skipped engines meant that ranking silently degenerated to
    # discovery order and the report was built on it. Appended AFTER the
    # cache write, like the sibling: the banner describes a transient
    # breaker state and must not be served back for 5 minutes.
    if _skipped:
        return result + "\n\n" + _narrowed_header(_skipped).lstrip("\n")
    return result
