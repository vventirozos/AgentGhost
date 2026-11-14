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
    hard body-size cap and on the EVENT LOOP behind `_onion_gate`'s bound —
    never on a thread: curl_cffi's SYNC streaming path double-frees inside
    libcurl and aborts the process (§4GM). The *research*
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
from ..utils.helpers import url_ssrf_reason, aclose_curl_response
from ..utils.helpers import FetchError as _FetchErrorBase   # §4UN-darkres r1: `_FetchTimeout` subclasses it
from . import search as _search      # §4NK: the shared slot/backfill/no-facts helpers
from .search import (
    source_block_failed as _source_block_failed,
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
_ONION_RE = re.compile(                                   # §4PD: v3 only — v2 (16 chars) died in 2021
    # §4UN-dark r1: a subdomain (www.<v3>.onion — the v3 label is what's validated) and a :port are kept
    r"https?://(?:[a-z0-9-]+\.)*[a-z2-7]{56}\.onion(?::\d{1,5})?(?:/[^\s\"'<>]*)?",
    re.IGNORECASE,
)


# §4UN-dark (review r1): the checksum lives in core (core must not import tools); re-exported here.
from ..core.link_grounding import onion_v3_valid  # noqa: E402,F401


#: §4UN-dark: any onion-looking address, garbled ones included (a non-base32 character, a wrong length), with
#: an optional scheme and path — what a summary is scanned for
_ANY_ONION_IN_TEXT_RE = re.compile(r"(?:https?://)?([a-z0-9]{10,}\.onion)(?:/[^\s\"'<>)\]]*)?", re.I)


def keep_page_onions(summary: str, page_text: str) -> str:
    """§4UN-dark: a worker's summary of an onion page keeps an onion address only when that address is
    VERBATIM in the page it summarised and its v3 checksum holds; any other one is replaced by the link
    guard's own mark. The summary is tool output — and an address in tool output passes every later guard."""
    if not summary or ".onion" not in summary.lower():
        return summary
    from ..core.link_grounding import _ONION_MARK
    page = str(page_text or "").lower()

    def _sub(m: "re.Match") -> str:
        host = m.group(1).lower()
        if host in page and onion_v3_valid(host):
            return m.group(0)
        return _ONION_MARK
    return _ANY_ONION_IN_TEXT_RE.sub(_sub, summary)

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
        "result_selector": "li.result",                 # §4UN-dark: its results; the rest is page furniture
    },
    {
        "name": "ahmia-onion",
        "url": "http://juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd.onion/search/?q={q}",
        "index": "ahmia",
        "form_token_from": "http://juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd.onion/",
        "result_selector": "li.result",
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
        # §4LH: over 3 weeks of live use it was useful in 22% of searches (78
        # "no matches", 34 timeouts at the 38 s deadline), so it is a FALLBACK:
        # asked only when the other engines come back thin.
        "name": "torch",
        "url": "http://xmh57jrknzkhv6y3ls3ubitzfqnkrwxhopf5aygthi7d6rplyvk3noyd.onion"
               "/cgi-bin/omega/omega?P={q}&HITSPERPAGE=100",
        "index": "torch",
        "tier": "fallback",
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
    # §4HB: Xapian Omega's zero-hit line — torch's CGI. Fetched over Tor
    # 2026-09-15: every torch "empty" was a 1.7–1.9 KB page reading
    # "No documents match your query", reported as "format may have
    # drifted" at WARNING on every miss because this phrasing was absent.
    r"|no\s+documents?\s+match"
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


def _env_seconds(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name) or default)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


# §4UN-darkres (req 212e6a8c): a hidden service that sends NOTHING back gives
# up here instead of at `_ONION_PAGE_TIMEOUT` — six dead onions, two at a
# time, each holding its slot for 35 s, were most of a 140 s run. A page that
# answers (status + headers) keeps the full read time for its body.
_ONION_FIRST_BYTE_S = _env_seconds("GHOST_ONION_FIRST_BYTE_S", 15.0)
# darkweb_research: onion pages fetched at once. Each URL's budget is issued
# when it STARTS (`_bounded`), so a wider gate no longer starves the later ones.
_RESEARCH_FETCH_WIDTH = 4
# The distiller's time kept back when a URL's fetch is capped: the fetch may
# use what is left of the URL's budget minus this (never less than the first
# byte + 5 s). 212e6a8c: "distill declined (only -5.6s of budget left)".
# ≥ the 16 s queue allowance + a working distill.
_DISTILL_RESERVE_S = 30.0
# r1: the reserve never shortens the first-byte wait below this
_FIRST_BYTE_FLOOR_S = 5.0
# r2: the longest wait for a streamed response's close (abort + reap)
_ACLOSE_WAIT_S = 2.0
# dr2: the read a first-byte retry must still have time for (beside the wait + the distill reserve)
_RETRY_READ_S = 10.0
# The most pages a caller may hand darkweb_research in `urls`.
_MAX_GIVEN_URLS = 8

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
                      all_skipped: bool = False,
                      outcomes: Optional[Dict[str, str]] = None) -> str:
    """The zero-results message plus, §4UN-dark, what each engine's query came to."""
    msg = _no_results_error_text(skipped, total, all_skipped)
    if outcomes and not all_skipped:
        line = outcomes_line(outcomes)
        # r1: an answer is results or "no match" — a page the parser could not read ("nothing parseable",
        # layout drift) says nothing about the query
        answered = [e for e, o in outcomes.items() if o in _ANSWERED]
        if answered and len(answered) == len(outcomes) and not skipped:
            # r1: the base text blames flaky engines or the query — wrong when every engine answered
            return ("ERROR: dark-web search found no onion results: every engine asked "
                    f"({', '.join(sorted(answered))}) ANSWERED with no match. That is a real negative for "
                    "these indexes — not proof the topic is absent from the dark web (hidden services "
                    "appear and vanish, and these indexes are partial). Do not repeat the same search; "
                    "other plain keywords may still match, or say the indexes had nothing."
                    + (f"\nEngines — {line}." if line else ""))
        if line:
            msg += f"\nEngines — {line}."
        if not answered:
            msg += " No engine answered: this says nothing about your query."
    return msg


def _no_results_error_text(skipped: List[str], total: int,
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


#: §4UN-dark: what one engine query came to — said in the result and the empty-search error, because "torch
#: and torgle answered: no match" is evidence and "everything timed out" is not
OUTCOME_RESULTS = "results"
OUTCOME_NO_MATCH = "answered, no match"
OUTCOME_PARSER = "answered, nothing parseable"
OUTCOME_REDIRECTED = "redirected away"
OUTCOME_JS = "needs JavaScript"
OUTCOME_UNREACHABLE = "unreachable"
OUTCOME_TIMED_OUT = "timed out"
OUTCOME_SKIPPED = "skipped (failure cooldown)"
#: best first: an engine's outcome over several phrasings is its best one
_OUTCOME_ORDER = (OUTCOME_RESULTS, OUTCOME_NO_MATCH, OUTCOME_PARSER, OUTCOME_REDIRECTED, OUTCOME_JS,
                  OUTCOME_TIMED_OUT, OUTCOME_UNREACHABLE, OUTCOME_SKIPPED)
#: an ANSWER about the query (r2: "nothing parseable" is layout drift — not an answer, not "reached")
_ANSWERED = (OUTCOME_RESULTS, OUTCOME_NO_MATCH)
#: r1: a page came back on some attempt — kept over a later attempt's timeout or error
_PAGE_OUTCOMES = (OUTCOME_PARSER, OUTCOME_REDIRECTED, OUTCOME_JS)


class _EngineRows(list):
    """One engine's parsed rows, carrying what the query came to (``outcome``)."""

    def __init__(self, rows=(), outcome: str = OUTCOME_RESULTS, engine_name: str = ""):
        super().__init__(rows)
        self.outcome = outcome
        self.engine_name = engine_name


class _Ranked(list):
    """The fan-out's ranked rows plus ``outcomes`` {engine: outcome} — an empty one is still falsy."""

    def __init__(self, rows=(), outcomes: Optional[Dict[str, str]] = None):
        super().__init__(rows)
        self.outcomes = dict(outcomes or {})


def merge_outcomes(*maps: Optional[Dict[str, str]]) -> Dict[str, str]:
    """{engine: best outcome} over several phrasings."""
    out: Dict[str, str] = {}
    rank = {o: i for i, o in enumerate(_OUTCOME_ORDER)}
    for m in maps:
        for eng, o in (m or {}).items():
            if eng not in out or rank.get(o, 99) < rank.get(out[eng], 99):
                out[eng] = o
    return out


def outcomes_line(outcomes: Dict[str, str]) -> str:
    """"results: ahmia, torgle; answered, no match: torch; timed out: ahmia-onion" (outcomes in order)."""
    groups: Dict[str, List[str]] = {}
    for eng, o in outcomes.items():
        groups.setdefault(o, []).append(eng)
    return "; ".join(f"{o}: {', '.join(sorted(groups[o]))}" for o in _OUTCOME_ORDER if o in groups)


#: §4UN-dark r1: there is no learned "page furniture" list. It was read off no-hits pages and then dropped
#: those URLs from every later query for a day, on-topic or not (a real hit vanished; Ahmia's sidebar link
#: was lost as a real result). Off-topic rows (ads, "popular" links) rank last instead (`_rank_key`).


class _BreakerSkipped(_EngineRows):
    """An empty result that also says WHICH engine was never contacted.

    A list subclass on purpose: every existing consumer keeps treating it
    as the empty list it is (`if not res`, `res.extend(...)`, the
    `isinstance(res, list)` guard in the gather), so nothing downstream
    needs to change — but the fan-out can distinguish "asked and found
    nothing" from "never asked", which is the whole point.
    """

    def __init__(self, engine_name: str):
        super().__init__((), OUTCOME_SKIPPED, engine_name)


import contextvars as _contextvars                                    # noqa: E402

#: §4UN-dark: the engines this tool call already charged a failure to (set by the search/research cores)
_BREAKER_CALL: "_contextvars.ContextVar[Optional[set]]" = _contextvars.ContextVar("onion_breaker_call",
                                                                                    default=None)


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
                       icon=Icons.TOOL_DARKWEB)
        return
    _call = _BREAKER_CALL.get()
    if _call is not None:
        # §4UN-dark: ONE failure per engine per tool call — a 4-phrasing call queries each engine 4 times
        # at once, and one bad Tor patch then benched it for 15 min in a single call
        if name in _call:
            return
        _call.add(name)
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

# Onion fetches ride the EVENT LOOP, not a thread pool (§4GM, 2026-09-14).
#
# ⚠ THEY USED TO RUN ON A BOUNDED ThreadPoolExecutor, AND THAT KILLED THE
# AGENT TWICE IN TWELVE HOURS. curl_cffi's SYNCHRONOUS `Session.request`
# with `stream=True` duplicates the easy handle (`curl_easy_duphandle`),
# resets the parent immediately, performs on a thread pool of its own and
# resets the DUPLICATE from a done-callback on yet another thread.
# libcurl-impersonate does not deep-copy that option set, so the two resets
# free the same pointer: `curl_easy_reset` → `Curl_freeset` → "pointer being
# freed was not allocated" → abort(). Measured, not theorised: SIGABRT on
# 2026-09-13 22:19:51 (pid 38307) and 2026-09-14 09:07:40 (pid 65198), same
# stack both times, the second one mid-request one second after an onion
# engine's post-deadline fetch was still lingering — and reproduced locally
# in ~40s with this call shape on 8 threads. The process dies; launchd
# restarts it; the request is lost.
#
# The ASYNC session has no `duphandle` anywhere: it takes a handle from a
# pool and resets it on the loop thread. That is the only shape that cannot
# corrupt libcurl's heap, and it is what every OTHER fetch in this tree
# already uses. Cancellation improves too — `wait_for` now aborts the
# transfer instead of orphaning a thread that runs on to curl's own timeout.
#
# The 16-way bound the pool gave us is KEPT, as a semaphore: excess fetches
# queue instead of piling concurrent circuits onto Tor. It is rebuilt when
# the running loop changes, because a semaphore that has waited on one loop
# cannot be awaited on another and the suite builds a fresh loop per test.
_MAX_CONCURRENT_ONION_FETCHES = 16
_ONION_GATE: "Optional[asyncio.Semaphore]" = None
_ONION_GATE_LOOP: Any = None


def _onion_gate() -> asyncio.Semaphore:
    """The concurrency bound for onion fetches, bound to the RUNNING loop."""
    global _ONION_GATE, _ONION_GATE_LOOP
    loop = asyncio.get_running_loop()
    if _ONION_GATE is None or _ONION_GATE_LOOP is not loop:
        _ONION_GATE = asyncio.Semaphore(_MAX_CONCURRENT_ONION_FETCHES)
        _ONION_GATE_LOOP = loop
    return _ONION_GATE


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
            if str(e.get("tier") or "") == "fallback":
                entry["tier"] = "fallback"
            if e.get("result_selector"):
                entry["result_selector"] = str(e["result_selector"])
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


def _extract_onion(s: str, engine_hosts: Optional[set] = None) -> Optional[str]:
    """Pull a single ``.onion`` URL out of an href / redirect wrapper.

    Ahmia (and some others) wrap the real target in a redirect URL like
    ``/search/redirect?...&redirect_url=http%3A%2F%2F...onion``; we unwrap
    that first, then fall back to a direct regex scan. §4UN-dark r1: a wrapper on the ENGINE's own onion
    (``<torgle>.onion/redirect.php?url=<target>``, host in ``engine_hosts``) is unwrapped too. The href is
    returned as BeautifulSoup decoded it — never entity-decoded again (``&section=2`` became ``§ion=2``)."""
    if not s:
        return None
    _eng = {_onion_host(str(h)) for h in (engine_hosts or ())}
    try:
        pr = urlparse(s)
        _h = (pr.hostname or "").lower()
        if pr.query and (not _h.endswith(".onion") or _onion_host(s) in _eng):
            qs = parse_qs(pr.query)
            # §4UN-dark: the known wrapper keys first, then ANY value that decodes to an onion URL
            # (``?link=http%3A%2F%2F….onion`` lost the result)
            keys = [k for k in ("redirect_url", "url", "u", "d") if k in qs] + \
                [k for k in qs if k not in ("redirect_url", "url", "u", "d")]
            for key in keys:
                for v in qs.get(key) or ():
                    m = _ONION_RE.search(unquote(v))
                    if m and onion_v3_valid(m.group(0)):
                        return m.group(0)
    except Exception:
        pass
    for m in _ONION_RE.finditer(s):
        if onion_v3_valid(m.group(0)):
            return m.group(0)
    return None


def _clean_onion_url(url: str) -> str:
    """§4UN-dark: a URL scanned out of page TEXT (the regex sweep — never an href): ``&amp;`` collapsed
    (``&amp;amp;`` → ``&``) and trailing sentence punctuation dropped — one ended ``…&amp;amp;raw=1.`` and the
    fetch got a 404. r1: only ``&amp;`` — a full unescape turned ``&section=2`` into ``§ion=2`` (legacy
    entities without a semicolon) — and a closing ")" stays when the URL opened one (``/wiki/Foo_(bar)``)."""
    u = str(url or "")
    while "&amp;" in u.lower():
        u = re.sub(r"(?i)&amp;", "&", u)
    while u:
        last = u[-1]
        if last in ".,;:!?]}'\"":
            u = u[:-1]
        elif last == ")" and u.count(")") > u.count("("):
            u = u[:-1]
        else:
            break
    return u


def _url_key(url: str) -> str:
    """A result's identity: the site (`_onion_host`: www.<v3> and <v3> are one), the port (r2: :8080 is
    another service), path and query; scheme and fragment dropped, trailing slash folded."""
    try:
        pr = urlparse(url)
        path = (pr.path or "").rstrip("/")
        try:
            port = pr.port
        except ValueError:
            port = None
        return (_onion_host(url) + (f":{port}" if port and port not in (80, 443) else "") + path
                + (("?" + pr.query) if pr.query else ""))
    except Exception:  # noqa: BLE001
        return str(url or "").lower().rstrip("/")


#: §4UN-dark: one site may hold several results (a forum's home and its threads) — at most this many paths
_MAX_PATHS_PER_HOST = 3


def _onion_host(url: str) -> str:
    """The site: for an onion, its v3 label (r2: ``www.<v3>.onion`` and ``<v3>.onion`` are one site — for the
    per-host cap, agreement and engine self-exclusion); any other host as is. A bare host works too."""
    try:
        raw = str(url or "")
        h = (urlparse(raw if "://" in raw else "http://" + raw).hostname or raw).lower()
    except Exception:
        h = str(url or "").lower()
    if h.endswith(".onion"):
        return h[: -len(".onion")].rsplit(".", 1)[-1] + ".onion"
    return h


#: §4PD: anchors read per engine page (a result page has tens; a page built
#: to stall the parser has thousands)
_PARSE_MAX_ANCHORS = 1500


def _parse_onion_results(html: str, exclude_hosts: Optional[set] = None,
                         *, result_selector: str = "",
                         rest: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Parse an onion search-engine result page into {url,title,snippet}.

    Each engine renders differently, so the parse is tolerant: an engine with a known result container
    (``result_selector``, e.g. Ahmia's ``li.result``) is read inside those containers only; otherwise every
    anchor is walked, then a regex sweep catches onion URLs written as plain text. Hosts in
    ``exclude_hosts`` (the engines' own) are dropped — a wrapper on one is unwrapped first.

    §4UN-dark: results are per URL, not per host (a forum's ``/thread/123`` is the hit; the host's homepage
    link before it kept only the homepage) — at most ``_MAX_PATHS_PER_HOST`` per host. A snippet is the text
    of the LARGEST ancestor that holds this one result URL and no other: results sharing one container got
    the whole list's text, so an unrelated row carried another row's "Acme breach…" and ranked on-topic.
    Every address must pass the v3 checksum (`onion_v3_valid`). r1: only anchors that will be RESULTS count
    toward "another result joins here" (the engine's own cached/report links made every snippet empty).

    ``rest`` (a list, filled when given): the page's visible text with every result's OWN text cut out —
    where an engine's "no documents match" answer is read (r1: a snippet saying "Unable to find your order?"
    sank a 3-result page). A result's own text is its NEAREST block that holds this link and no other (else
    just the link's text) — never a wrapper: r2, deleting the widest block took a banner that sat in the same
    wrapper as an ad, and the ad was booked as a result. r3: built from the text nodes outside those
    elements (nothing deleted, no string subtraction — an ad's text "No" cut the banner). Only
    filled for a page of at most ``_NO_HITS_MAX_LINKS`` results (the only pages it is read for)."""
    results: List[Dict[str, str]] = []
    if not html:
        if rest is not None:
            rest.append("")
        return results
    skip_hosts = {_onion_host(str(h)) for h in (exclude_hosts or ())}
    seen: set = set()
    per_host: Dict[str, int] = {}

    def _admit(onion: str) -> bool:
        host = _onion_host(onion)
        key = _url_key(onion)
        if host in skip_hosts or key in seen:
            return False
        if per_host.get(host, 0) >= _MAX_PATHS_PER_HOST:
            return False
        seen.add(key)
        per_host[host] = per_host.get(host, 0) + 1
        return True

    soup = None
    own_nodes: List[Any] = []                           # each result's own block/link (``rest`` is read outside)
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        containers = []
        if result_selector:
            try:
                containers = soup.select(result_selector)
            except Exception:  # noqa: BLE001 — a bad selector falls back to the generic walk
                containers = []
        scopes = containers or [soup]
        # §4PD: a bound on the anchors read; the caller runs this in a thread
        anchors: List[Tuple[Any, str, Any]] = []
        for scope in scopes:
            for a in scope.find_all("a", href=True):
                if len(anchors) >= _PARSE_MAX_ANCHORS:
                    break
                onion = _extract_onion(a["href"], skip_hosts)
                if onion and _onion_host(onion) not in skip_hosts:
                    anchors.append((a, onion, scope if containers else None))
        # which result URLs each block holds (block-level ancestors, a few levels up)
        _BLOCKS = ("li", "div", "article", "section", "p", "dd", "dt", "td", "tr", "dl", "ul", "ol", "table")
        holds: Dict[int, set] = {}
        chains: Dict[int, List[Any]] = {}
        for a, onion, _own in anchors:
            key = _url_key(onion)
            chain = []
            for anc in a.parents:
                if getattr(anc, "name", None) in ("body", "html", "[document]") or len(chain) >= 8:
                    break
                if anc.name in _BLOCKS:
                    chain.append(anc)
                    holds.setdefault(id(anc), set()).add(key)
            chains[id(a)] = chain
        _text: Dict[int, str] = {}
        for a, onion, _own in anchors:
            if not _admit(onion):
                continue
            host = _onion_host(onion)
            title = " ".join(a.get_text(" ", strip=True).split())
            snippet = ""
            chain = chains.get(id(a), [])
            if _own is not None:
                block = _own                            # the engine's own result container
            else:
                block = None
                for anc in chain:
                    if len(holds.get(id(anc), ())) == 1:
                        block = anc                     # still only this result: widen
                    else:
                        break                           # another result joins here: stop
            if rest is not None and len(own_nodes) <= _NO_HITS_MAX_LINKS:
                # r2: the NEAREST block holding only this link (an ad's own div, not the wrapper that also
                # holds the engine's "no documents match"); a link in a shared list: its own text only
                _near = chain[0] if chain and len(holds.get(id(chain[0]), ())) == 1 else a
                own_nodes.append(_near)
            if block is not None:
                if id(block) not in _text:
                    _text[id(block)] = " ".join(block.get_text(" ", strip=True).split())[:4000]
                btext = _text[id(block)]
                if title and title in btext:
                    btext = btext.replace(title, "", 1)
                snippet = " ".join(btext.split())[:400]
            results.append({"url": onion, "title": title or host, "snippet": snippet})
    except Exception:
        # BeautifulSoup unavailable or blew up — fall through to regex sweep.
        soup = None

    # Regex sweep for onion URLs not captured as anchors (plain text, JSON blobs, etc.); the host is the
    # title. Not with a known container: text outside the result list is page furniture.
    if not result_selector or not results:
        for m in _ONION_RE.finditer(html):
            onion = _clean_onion_url(m.group(0))
            if not onion_v3_valid(onion) or not _admit(onion):
                continue
            results.append({"url": onion, "title": _onion_host(onion), "snippet": ""})

    if rest is not None and len(results) > _NO_HITS_MAX_LINKS:
        rest.append("")                                 # only a page of ≤ _NO_HITS_MAX_LINKS links is read for it
    elif rest is not None:
        try:
            if soup is None:
                raise ValueError("no soup")
            text = _tag_visible_text(soup, skip={id(n) for n in own_nodes})
        except Exception:  # noqa: BLE001
            text = " ".join(_visible_text(html).split())
        rest.append(text)
    return results


def _tag_visible_text(tag: Any, skip: Optional[set] = None) -> str:
    """A tag's visible text, whitespace-folded: its strings outside script/style/template, comments,
    doctypes and processing instructions — and (r3) outside any element whose ``id()`` is in ``skip`` (each
    result's own block or link). Built from the text nodes, never by cutting strings out: r2's substring
    removal cut a banner "No results found." wherever an ad's text ("results", "No") occurred in it."""
    skip = skip or set()
    from bs4.element import NavigableString, CData
    parts: List[str] = []
    for t in tag.find_all(string=True) if hasattr(tag, "find_all") else ():
        if type(t) not in (NavigableString, CData):
            continue                                    # Comment, Script, Stylesheet, TemplateString, Doctype…
        if any(getattr(p, "name", None) in ("script", "style", "template") or id(p) in skip
               for p in t.parents):
            continue
        v = t.strip()
        if v:
            parts.append(v)
    return " ".join(" ".join(parts).split())


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
        status, body = await asyncio.wait_for(                 # §4PD: curl's timeout is a low-speed limit, not a clock
            _fetch_raw_html(token_url, proxy, _FORM_TOKEN_TIMEOUT), timeout=_FORM_TOKEN_TIMEOUT + 2.0)
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


#: §4UN-dark: a page that says "no match" and holds at most this many onion links is a no-hits answer
_NO_HITS_MAX_LINKS = 5
_HIDDEN_BLOCK_RE = re.compile(r"(?is)<(script|style|template)\b.*?</\1\s*>")
_COMMENT_RE = re.compile(r"(?s)<!--.*?-->")
_TAG_RE = re.compile(r"<[^>]{0,2000}>")


def _visible_text(body: str) -> str:
    """§4UN-dark: the page's VISIBLE text — scripts, styles, templates and comments out. "No results" in a
    script's message table or a hidden template hid a drifted layout as an ordinary no-hits answer, which
    never trips the breaker nor the drift warning."""
    t = _HIDDEN_BLOCK_RE.sub(" ", body or "")
    t = _COMMENT_RE.sub(" ", t)
    return _TAG_RE.sub(" ", t)


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
    if _NO_HITS_RE.search(_visible_text(body)):
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


def _empty_body_reason(meta: Dict[str, Any]) -> str:
    """§4UN-dark: why a 200 had no readable body — a binary file, one over the size cap, or an empty
    page ("received status 200" read as a server error)."""
    ct = str(meta.get("content_type") or "").split(";")[0].strip().lower()
    if ct and ("application/pdf" in ct or "application/octet-stream" in ct):
        return f"it is a file ({ct}), not a page this tool reads"
    try:
        if int(meta.get("content_length") or 0) > _MAX_ONION_BODY_BYTES:
            return f"the page is larger than the {_MAX_ONION_BODY_BYTES // (1024 * 1024)} MB cap"
    except (TypeError, ValueError):
        pass
    return "the server answered 200 with an empty page"


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
                          *, meta: Optional[Dict[str, Any]] = None,
                          first_byte: Optional[float] = None,
                          body_deadline: Optional[float] = None) -> Tuple[Optional[int], str]:
    """Fetch RAW HTML (tags intact) through the Tor SOCKS proxy.

    Unlike `helper_fetch_url_content`, this does NOT strip tags — the search
    phase needs the markup to parse out result links — and it HONOURS the
    passed proxy rather than reading it from the environment. Uses curl_cffi
    when present (TLS-impersonating, the project default) and falls back to
    httpx. The body is size-capped (`_cap_body`) and the request runs on the
    EVENT LOOP behind `_onion_gate`'s 16-way bound — never on a worker
    thread: curl_cffi's sync streaming path corrupts libcurl's heap and
    aborts the process (see `_onion_gate`, §4GM). Returns (status_code,
    body); (None, "") on transport failure.

    Redirects are followed, so the status alone cannot tell you whether the
    body came from the URL you asked for. Pass a ``meta`` dict to receive the
    ``final_url`` the body was actually served from — without it, a dead
    endpoint that 302s to a working homepage is indistinguishable from a live
    endpoint serving an unparseable page, which is exactly how the Ahmia
    breakage stayed misdiagnosed for weeks.

    ``first_byte`` (§4UN-darkres): seconds to wait for the status line and
    headers; raises ``asyncio.TimeoutError`` past it. The body keeps
    ``timeout``. (The httpx fallback can only bound the connect.)

    ``body_deadline`` (§4UN-darkres r1, a ``time.monotonic()`` instant): the
    body is read until then; a page still sending at the deadline — or one
    whose transfer errors after bytes arrived — is returned PARTIAL with
    ``meta["cut"] = True`` instead of being lost (curl_cffi path)."""

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

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    proxies = {"http": proxy, "https": proxy} if proxy else None

    async with _onion_gate():
        try:
            import curl_cffi.requests as creq
        except ImportError:
            creq = None

        if creq is not None:
            async with creq.AsyncSession(impersonate="chrome110",
                                         proxies=proxies, timeout=timeout) as c:
                if first_byte:
                    from ..utils.aio import wait_for as _wait_for
                    r = await _wait_for(c.get(url, headers=headers, stream=True), first_byte)
                else:
                    r = await c.get(url, headers=headers, stream=True)
                # Status and headers are parsed BEFORE the body streams, so
                # read them now: the reaper below may abort the transfer, and
                # a caller that reads them off a closed response is reading
                # whatever state the abort left behind.
                _status = r.status_code
                _ctype = r.headers.get("content-type")
                _clen = r.headers.get("content-length")
                _record_final_url(meta, r)
                if meta is not None:                  # §4UN-dark: why a 200 can carry no readable body
                    meta["content_type"], meta["content_length"] = _ctype or "", _clen
                buf = bytearray()
                try:
                    # aiter_content, NOT iter_content: on an AsyncSession
                    # response the sync iterator yields un-awaited
                    # Queue.get() COROUTINES, not bytes.
                    if body_deadline is None:
                        async for chunk in r.aiter_content():
                            if chunk:
                                buf.extend(chunk)
                                if len(buf) >= _STREAM_LIMIT:
                                    break
                    else:
                        await _read_until(r.aiter_content(), buf, _STREAM_LIMIT, body_deadline, meta)
                finally:
                    # Abort + reap. Without this the cap break leaves the
                    # transfer running until the server or curl gives up —
                    # and on a cancellation (the per-engine deadline) the
                    # handle would go back to the pool mid-transfer.
                    # §4UN-darkres r2: bounded — a close that hangs must not turn a page
                    # already read into a timeout; the session's own close follows
                    try:
                        from ..utils.aio import wait_for as _wait_for
                        await _wait_for(aclose_curl_response(r), _ACLOSE_WAIT_S)
                    except asyncio.TimeoutError:
                        pretty_log("Onion Close Slow", f"{str(url)[:60]} — response close took over "
                                   f"{_ACLOSE_WAIT_S:.0f}s; left to the session", level="WARNING", icon=Icons.WARN)
                return _cap_body(_status, _ctype, _clen,
                                 _decode(bytes(buf), _ctype))

        import httpx

        _tmo = httpx.Timeout(timeout, connect=min(timeout, first_byte)) if first_byte else timeout
        async with httpx.AsyncClient(proxy=proxy, timeout=_tmo,
                                     follow_redirects=True) as c:
            async with c.stream("GET", url, headers=headers) as r:
                buf = bytearray()
                async for chunk in r.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) >= _STREAM_LIMIT:
                        break
                _record_final_url(meta, r)
                if meta is not None:
                    meta["content_type"] = r.headers.get("content-type") or ""
                    meta["content_length"] = r.headers.get("content-length")
                return _cap_body(r.status_code, r.headers.get("content-type"),
                                 r.headers.get("content-length"),
                                 _decode(bytes(buf), r.headers.get("content-type")))


async def _read_until(chunks: Any, buf: bytearray, limit: int, deadline: float,
                      meta: Optional[Dict[str, Any]]) -> None:
    """§4UN-darkres r1: drain ``chunks`` into ``buf`` until the stream ends, the
    cap, or ``deadline`` (monotonic). Stopped by the deadline, or by a transfer
    error after some bytes arrived: ``meta["cut"] = True`` and the bytes stay."""
    from ..utils.aio import wait_for as _wait_for
    it = chunks.__aiter__()
    while len(buf) < limit:
        left = deadline - time.monotonic()
        if left <= 0:
            if meta is not None:
                meta["cut"] = True
            return
        try:
            chunk = await _wait_for(it.__anext__(), left)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            if meta is not None:
                meta["cut"] = True
            return
        except Exception:  # noqa: BLE001 — a transfer error mid-body: keep what arrived
            if not buf:
                raise
            if meta is not None:
                meta["cut"] = True
            return
        if chunk:
            buf.extend(chunk)


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


class _FetchTimeout(_FetchErrorBase):
    """§4UN-darkres r1: a fetch failure that IS a timeout — typed, so a URL path
    holding the word "timeout" never reads as one."""


class _FirstByteTimeout(_FetchTimeout):
    """§4UN-darkres dr2: the page sent nothing within the first-byte wait — the one
    failure darkweb_research retries (once, on a fresh circuit)."""


class _CutPage(str):
    """§4UN-darkres r1: page text read until the read time ran out (partial)."""


class _RawPage(str):
    """§4UN-darkres r2: the HTML of a research page fetch, returned UNSTRIPPED so the
    caller strips it outside the fetch's timeout (a 2 MB page took 1.35 s);
    ``cut`` says the body was cut at its read time."""
    cut = False


#: (first-byte seconds, body deadline as a monotonic instant) for the onion page
#: fetch running in THIS task — set by darkweb_research per URL (a context var,
#: so `_fetch_onion_text` keeps its two-argument shape)
_PAGE_LIMITS: "_contextvars.ContextVar[Optional[Tuple[float, float]]]" = _contextvars.ContextVar(
    "_onion_page_limits", default=None)


def _is_timeout_exc(e: BaseException) -> bool:
    """A transport timeout, by TYPE: asyncio/builtin, curl error 28, or a client's *Timeout* class."""
    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
        return True
    if getattr(e, "code", None) == 28:                        # CURLE_OPERATION_TIMEDOUT
        return True
    return any("Timeout" in k.__name__ for k in type(e).__mro__)


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
        return _FetchError(f"Error: {reason}")
    _meta: Dict[str, Any] = {}
    _limits = _PAGE_LIMITS.get()
    _fb = _limits[0] if _limits else _ONION_FIRST_BYTE_S
    _kw: Dict[str, Any] = {"first_byte": _fb}
    if _limits:
        _kw["body_deadline"] = _limits[1]
    try:
        status, html = await _fetch_raw_html(url, tor_proxy, _ONION_PAGE_TIMEOUT, meta=_meta, **_kw)
    except asyncio.TimeoutError:
        return _FirstByteTimeout(f"Error: no answer from {url} within {_fb:.0f}s (timed out)")
    except Exception as e:  # noqa: BLE001
        # "Error:" — the head every failure check reads (§4LV: "Error
        # reading" was summarised as page text, booked OK and cached)
        if _is_timeout_exc(e):
            return _FetchTimeout(f"Error: reading {url} failed: {e}")
        return _FetchError(f"Error: reading {url} failed: {e}")
    if not status:
        return _FetchError(f"Error: no response from {url}")
    if status == 200 and not html:
        return _FetchError(f"Error: {url} — {_empty_body_reason(_meta)}; not read")
    if status != 200 or not html:
        return _FetchError(f"Error: received status {status} from {url}")
    _final = str(_meta.get("final_url") or "")
    if _final and _redirected_away(url, _final):
        # §4PD: the page came from ANOTHER host (an onion bouncing to the
        # clearnet, or to a different onion) — not this source; say so
        return _FetchError(f"Error: {url} redirected to {_display_target(_final)} — a page on another "
                           "host is not this onion; not read")
    if _limits:                 # research path: the caller strips it, outside its timeout (r2)
        raw = _RawPage(html)
        raw.cut = bool(_meta.get("cut"))
        return raw
    text = await asyncio.to_thread(_strip_html, html)
    return _CutPage(text) if _meta.get("cut") else text


async def _query_engine(
    engine: Dict[str, str],
    query: str,
    tor_proxy: str,
    exclude_hosts: Optional[set] = None,
) -> List[Dict[str, str]]:
    """Query one onion engine with per-attempt circuit rotation, under an
    overall wall-clock deadline (`_ONION_ENGINE_DEADLINE`) so one slow or dead
    engine can't dominate the concurrent gather. Returns its parsed results
    (possibly empty) as `_EngineRows`, whose ``outcome`` says what the query
    came to (§4UN-dark); never raises."""
    from ..utils.aio import wait_for as _wait_for

    _diag = {"nohits": False, "outcome": OUTCOME_UNREACHABLE}   # §4PD: an engine that SAID 'no documents match' is not failing
    name = engine["name"]

    async def _attempts() -> List[Dict[str, str]]:
        base_url = engine["url"].format(q=quote_plus(query))
        token_from = engine.get("form_token_from")
        for attempt in range(2):
            proxy = _proxy_for_attempt(tor_proxy, f"{name}:{query}", attempt)
            # Re-read per attempt: a retry after a redirect-away has dropped
            # the cached token, so this is where a rotated one gets picked up.
            url = base_url
            token_applied = False
            if token_from:
                url = _apply_form_token(
                    base_url,
                    await _form_token(token_from, proxy, name))
                token_applied = url != base_url
            try:
                meta: Dict[str, Any] = {}
                # §4UN-dark: a per-attempt CLOCK — curl's timeout is a low-speed limit, so a stalled first
                # attempt could spend the whole engine deadline and the second circuit never ran
                status, body = await _wait_for(
                    _fetch_raw_html(url, proxy, _ONION_TIMEOUT, meta=meta), _ONION_TIMEOUT + 2.0)
                _final = str(meta.get("final_url") or "")
                _away = bool(_final) and _redirected_away(url, _final)
                if status == 200 and body and not _away:
                    _rest: List[str] = []
                    parsed = await asyncio.to_thread(                 # §4PD: off the loop
                        _parse_onion_results, body, exclude_hosts,
                        result_selector=str(engine.get("result_selector") or ""), rest=_rest)
                    if parsed and len(parsed) <= _NO_HITS_MAX_LINKS and \
                            _NO_HITS_RE.search(_rest[0] if _rest else ""):
                        # §4UN-dark: the engine SAYS nothing matched; the few links are its page furniture
                        # (an ad, a sidebar) — the old parse listed them as the query's results. r1: the
                        # phrase is read OUTSIDE every result block — a snippet's "Unable to find your
                        # order?" is a result's text, not the engine's answer
                        parsed = []
                    if parsed:
                        pretty_log(
                            "Darkweb Engine",
                            f"{name}: {len(parsed)} onion result(s)",
                            icon=Icons.TOOL_DARKWEB,
                        )
                        _diag["outcome"] = OUTCOME_RESULTS
                        return parsed
                if status == 200 and body:
                    # 200 but ZERO parseable results (or a page from somewhere else). This was silent — a
                    # broken engine looked identical to "no onion index has this query," so the tool blamed
                    # the query. Name the actual cause instead; see `_diagnose_empty_body` for why the
                    # ordering of those causes matters. §4UN-dark: a page served from ANOTHER location is
                    # never parsed as results (a homepage's "featured" onions became query matches).
                    kind, detail = _diagnose_empty_body(body, url, _final or None)
                    if kind == "redirected" and token_applied and attempt == 0:
                        # A token we DID send still got bounced: most likely
                        # rotated, so re-scrape on the retry. Gated on
                        # attempt 0 (a re-scrape after the last attempt only
                        # bills the next search) and on having actually sent
                        # one — otherwise any redirect, including a captcha
                        # or exit-ban page, throws away a good token.
                        _invalidate_form_token(token_from)
                    if kind == "no-hits":
                        # §4UN-dark: the engine ANSWERED — asking again on a new circuit only spends the
                        # deadline (and a slow retry then charged the engine a failure).
                        _diag["nohits"] = True
                        _diag["outcome"] = OUTCOME_NO_MATCH
                        pretty_log("Darkweb Engine", f"{name}: answered, no match", level="INFO",
                                   icon=Icons.SKIP)
                        return []
                    _diag["outcome"] = {"redirected": OUTCOME_REDIRECTED, "js-only": OUTCOME_JS}.get(
                        kind, OUTCOME_PARSER)
                    pretty_log(
                        "Darkweb Engine Empty",
                        f"{name}: HTTP 200 but 0 parseable results "
                        f"— {detail}",
                        level="WARNING", icon=Icons.WARN,
                    )
                elif status is not None:
                    if _diag["outcome"] not in _PAGE_OUTCOMES:      # r1: a page read earlier says more
                        _diag["outcome"] = OUTCOME_UNREACHABLE
                    pretty_log("Darkweb Engine Skipped",
                               f"{name}: attempt {attempt + 1} of 2 — HTTP {status}"
                               + ("" if body else " (empty body)"),
                               level="INFO", icon=Icons.SKIP)
            except Exception as e:  # noqa: BLE001
                # §4NV: a network failure over Tor is routine (INFO); anything
                # else is a defect in the parse path and must stand out
                _net = _is_network_error(e)
                if _diag["outcome"] in _PAGE_OUTCOMES:
                    pass                                # r1: attempt 1's page ("nothing parseable") says more
                elif isinstance(e, (asyncio.TimeoutError, TimeoutError)):
                    _diag["outcome"] = OUTCOME_TIMED_OUT
                else:
                    _diag["outcome"] = OUTCOME_UNREACHABLE
                pretty_log(
                    "Darkweb Engine Skipped" if _net else "Darkweb Engine Error",
                    f"{name}: attempt {attempt + 1} of 2 — "
                    f"{type(e).__name__}: {str(e)[:160] or 'no detail'}",
                    level="INFO" if _net else "WARNING",
                    icon=Icons.SKIP if _net else Icons.WARN,
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
    if _breaker_should_skip(name):
        logger.debug("onion engine %s skipped (breaker open)", name)
        return _BreakerSkipped(name)
    deadline = _ONION_ENGINE_DEADLINE
    if engine.get("form_token_from"):
        deadline += min(_FORM_TOKEN_TIMEOUT, _ONION_ENGINE_DEADLINE)
    try:
        _res = await asyncio.wait_for(_attempts(), timeout=deadline)
        # "Won" means RESULTS, not "did not raise": an engine that returns
        # an empty list every time is failing at the only thing it is for,
        # and 302-to-homepage (ahmia's pre-token shape) is exactly that.
        if _res or not _diag["nohits"]:
            _breaker_record(name, bool(_res))
        return _EngineRows(_res, _diag["outcome"] if not _res else OUTCOME_RESULTS, name)
    except asyncio.TimeoutError:
        if not _diag["nohits"]:           # §4UN-dark: an engine that answered "no match" is not failing
            _breaker_record(name, False)
        # The fetch is now an awaitable on THIS loop (§4GM), so cancelling
        # the await really does abort the transfer — it no longer orphans a
        # worker thread that runs on until curl's own timeout. `_onion_gate`
        # still bounds how many fetches run at once.
        pretty_log(
            "Darkweb Engine Skipped",
            f"{name}: exceeded {deadline:.0f}s deadline — skipped",
            level="INFO",
            icon=Icons.SKIP,
        )
        return _EngineRows((), OUTCOME_NO_MATCH if _diag["nohits"] else (
            _diag["outcome"] if _diag["outcome"] in _PAGE_OUTCOMES else OUTCOME_TIMED_OUT), name)


def _is_network_error(e: BaseException) -> bool:
    """A transport failure (timeout, refused, reset, a curl error) — not a
    defect in this module's own parsing."""
    if isinstance(e, (OSError, TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True
    return any("curl" in (c.__module__ or "") for c in type(e).__mro__)


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


#: Distinct onions the first wave must find for the fallback tier to stay idle.
_FALLBACK_MIN_HOSTS = 5
#: …and the fallback tier's own deadline, after the first wave.
_FALLBACK_DEADLINE_S = 12.0


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
    all_engines = _load_engines()
    exclude = _engine_onion_hosts(all_engines)
    # §4LH: fallback-tier engines (torch) are asked only when the first wave
    # found fewer than _FALLBACK_MIN_HOSTS distinct onions. `engines` is the
    # list this call actually SEARCHED, so the skip/denominator arithmetic
    # below stays over what was asked.
    if os.getenv("GHOST_ONION_FALLBACK_TIER", "1") == "0":
        first, later = list(all_engines), []
    else:
        first = [e for e in all_engines if e.get("tier") != "fallback"]
        later = [e for e in all_engines if e.get("tier") == "fallback"]
    engines = list(first)
    per_engine = list(await asyncio.gather(
        *[_query_engine(e, query, tor_proxy, exclude) for e in first],
        return_exceptions=True,
    )) if first else []
    _hosts = {_onion_host(r["url"]) for res in per_engine if isinstance(res, list) for r in res}
    if later and len(_hosts) < _FALLBACK_MIN_HOSTS:
        engines += later
        # A SHORT deadline: torch answers in 2-6 s when it answers at all
        # (review §4LH: a full second 38 s deadline after the first wave
        # took a thin search from ~50 s to ~88 s worst case).
        async def _fallback(e):
            # the outer deadline cancels _query_engine before its own timeout
            # records anything — record the miss here, or a dead torch is
            # paid for on every thin search (§4LH final review)
            from ..utils.aio import wait_for as _wait_for
            try:
                return await _wait_for(_query_engine(e, query, tor_proxy, exclude), _FALLBACK_DEADLINE_S)
            except asyncio.TimeoutError:
                _breaker_record(e["name"], False)
                logging.getLogger(__name__).info("fallback engine %s cut at %.0fs", e["name"],
                                                 _FALLBACK_DEADLINE_S)
                raise
        per_engine += list(await asyncio.gather(*[_fallback(e) for e in later], return_exceptions=True))

    # J3: the denominator is the list THIS call searched, not a second
    # `_load_engines()` read — that re-reads GHOST_ONION_ENGINES (so a
    # mid-process change makes "all skipped" arithmetic wrong) and counts
    # entries while the skip list counts names (two entries sharing a name
    # made "all skipped" unreachable).
    skipped = [r.engine_name for r in per_engine
               if isinstance(r, _BreakerSkipped)]
    all_skipped = bool(engines) and len(skipped) >= len(engines)
    # §4UN-dark: what each engine's query came to (a fallback cut by its own deadline: timed out)
    outcomes: Dict[str, str] = {}
    for engine, res in zip(engines, per_engine):
        o = getattr(res, "outcome", None) if isinstance(res, list) else (
            OUTCOME_TIMED_OUT if isinstance(res, asyncio.TimeoutError) else OUTCOME_UNREACHABLE)
        if isinstance(res, list) and res:
            o = OUTCOME_RESULTS
        outcomes = merge_outcomes(outcomes, {engine["name"]: o or OUTCOME_UNREACHABLE})

    # §4UN-dark: merged per URL (a forum's thread is its own result), corroboration counted per URL and,
    # as a weaker signal, per host
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    host_idx: Dict[str, set] = {}
    for engine, res in zip(engines, per_engine):
        if not isinstance(res, list):
            continue
        idx = engine.get("index") or engine["name"]
        for r in res:
            key = _url_key(r["url"])
            host_idx.setdefault(_onion_host(r["url"]), set()).add(idx)
            if key not in merged:
                merged[key] = {**r, "engines": {engine["name"]}, "indexes": {idx}}
                order.append(key)
            else:
                merged[key]["engines"].add(engine["name"])
                merged[key]["indexes"].add(idx)
                # Keep the richest snippet/title seen.
                if not merged[key].get("snippet") and r.get("snippet"):
                    merged[key]["snippet"] = r["snippet"]
                host = _onion_host(r["url"])
                if merged[key].get("title") == host and r.get("title") and r["title"] != host:
                    merged[key]["title"] = r["title"]
    rows = [merged[k] for k in order]
    _mark_on_topic(rows, query)
    for r in rows:
        r["host_indexes"] = len(host_idx.get(_onion_host(r["url"]), ()))
    ranked = _per_host_cap(sorted(enumerate(rows), key=lambda ir: _rank_key(ir[1], ir[0])))
    return (_Ranked(ranked[:max_results], outcomes), sorted(set(skipped)), all_skipped,
            len(engines))


def _mark_on_topic(rows: List[Dict[str, Any]], query: str) -> None:
    """§4UN-dark: ``on_topic`` on each row (a distinctive query token in its title, snippet or URL — the
    web search's own check), set BEFORE the list is cut: research cut to max_sources*3 by engine agreement
    first, so on-topic hits further down a 28-host page were never considered. No topic check: all on."""
    try:
        from .search import distinctive_tokens, result_on_topic
        dist = distinctive_tokens(query)
        for r in rows:
            r["on_topic"] = bool(result_on_topic(
                {"title": r.get("title"), "body": r.get("snippet"), "url": r.get("url")}, dist))
    except Exception:  # noqa: BLE001
        for r in rows:
            r.setdefault("on_topic", True)


def _rank_key(r: Dict[str, Any], i: int, n_queries: int = 0) -> tuple:
    """On-topic first (an ad or a sidebar link two engines both show is off-topic), then independent
    indexes for this URL, then for its host, then (multi-phrasing) how many phrasings found it, then
    discovery order. r2: among OFF-topic rows a bare homepage ranks below a page with a path before agreement
    counts — an ad (a homepage) two engines both show outranked an off-topic real hit."""
    off = not r.get("on_topic", True)
    bare = off and not urlparse(str(r.get("url") or "")).path.strip("/")
    return (off, bare, -len(r.get("indexes") or ()), -int(r.get("host_indexes") or 0), -n_queries, i)


def _per_host_cap(ranked_pairs) -> List[Dict[str, Any]]:
    """At most `_MAX_PATHS_PER_HOST` rows per host: on-topic rows first, then a page with a path before the
    bare homepage."""
    rows = [r for _, r in ranked_pairs]
    by_host: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_host.setdefault(_onion_host(r["url"]), []).append(r)
    keep = set()
    for rs in by_host.values():
        # r1: on-topic first, then a page with a path before the bare homepage
        pref = sorted(rs, key=lambda r: (not r.get("on_topic", True),
                                         not urlparse(str(r.get("url") or "")).path.strip("/")))
        keep.update(id(r) for r in pref[:_MAX_PATHS_PER_HOST])
    return [r for r in rows if id(r) in keep]


#: How many ADDITIONAL phrasings one `darkweb_search` call may carry (§4GN).
#: Four queries × four engines = 16 onion fetches, exactly the bound
#: `_onion_gate` already enforces, in ONE deadline instead of four.
_MAX_EXTRA_QUERIES = 3

#: Never return more than this many merged rows however many phrasings ran —
#: the result block is read by a model, and breadth is worth nothing if it
#: costs the context the investigation needs.
_MULTI_QUERY_RESULT_CEILING = 24


def _merge_across_queries(per_query: List[Tuple[str, List[Dict[str, Any]]]],
                          cap: int) -> List[Dict[str, Any]]:
    """Merge the ranked blocks of several phrasings into one ranked list.

    Same signal the single-query merge prizes — a hidden service surfaced by
    MORE INDEPENDENT INDEXES ranks higher — extended with the phrasing that
    surfaced it. A host found under three different wordings is corroborated
    in a way one wording cannot show, and the row says which wordings, so a
    reader can tell a broad hit from a lucky one. §4UN-dark: per URL, on-topic
    (under any phrasing) first.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    host_idx: Dict[str, set] = {}
    host_qs: Dict[str, set] = {}
    for q, rows in per_query:
        for r in rows:
            key = _url_key(r["url"])
            host = _onion_host(r["url"])
            host_idx.setdefault(host, set()).update(r.get("indexes") or ())
            host_qs.setdefault(host, set()).add(q)
            if key not in merged:
                merged[key] = {**r,
                               "engines": set(r.get("engines") or ()),
                               "indexes": set(r.get("indexes") or ()),
                               "queries": [q]}
                order.append(key)
                continue
            cur = merged[key]
            cur["engines"] |= set(r.get("engines") or ())
            cur["indexes"] |= set(r.get("indexes") or ())
            cur["on_topic"] = bool(cur.get("on_topic", True) or r.get("on_topic", True))
            if q not in cur["queries"]:
                cur["queries"].append(q)
            if not cur.get("snippet") and r.get("snippet"):
                cur["snippet"] = r["snippet"]
            if cur.get("title") == host and r.get("title") and r["title"] != host:
                cur["title"] = r["title"]
    for k in order:                       # a site found under several phrasings / indexes is corroborated
        h = _onion_host(merged[k]["url"])
        merged[k]["host_indexes"] = len(host_idx.get(h, ()))
        merged[k]["site_queries"] = [q for q, _rows in per_query if q in host_qs.get(h, ())]
    ranked = _per_host_cap(sorted(
        enumerate(merged[k] for k in order),
        key=lambda ir: _rank_key(ir[1], ir[0], len(host_qs.get(_onion_host(ir[1]["url"]), ())))))
    return ranked[:cap]


def _one_row_text(text: str) -> str:
    """§4PD: one line, no `[`/`]`, no `###` — a title cannot start a row,
    an `[Onion: …]` line or a `[⚠ …]` banner of its own."""
    t = " ".join(str(text or "").split()).replace("[", "(").replace("]", ")")
    return re.sub(r"#{2,}", "#", t)


def _format_results(results: List[Dict[str, Any]]) -> str:
    formatted = []
    for i, r in enumerate(results, 1):
        # §4PD: one line each, and no `[`: an engine's HTML put `### 2. …
        # [Onion: …] (via ahmia, torch)` and `[⚠ SYSTEM: …]` INSIDE a title
        title = _one_row_text(_clean_for_cpp(r.get("title") or r["url"]))
        snippet = _one_row_text(_clean_for_cpp(r.get("snippet") or ""))
        engs = ", ".join(sorted(r.get("engines", [])))
        body = (snippet + "\n") if snippet else ""
        # The phrasing tag appears ONLY on a multi-query call, so a single
        # query's output is byte-identical to what it always was (the cache
        # and every caller downstream read this text).
        qs = r.get("site_queries") or r.get("queries") or []     # §4UN-dark: the phrasings that found the SITE
        via = f"(via {engs})" if len(qs) < 2 else \
            f"(via {engs}; matched: {'; '.join(qs)})"
        formatted.append(f"### {i}. {title}\n{body}[Onion: {r['url']}] {via}")
    return "\n\n".join(formatted)


def _query_set(query: str, extra_queries: Any) -> List[str]:
    """The phrasings this call will run: the main one plus up to
    ``_MAX_EXTRA_QUERIES`` extras, sanitized, order-preserving, deduped.

    A brief that enumerates ten search strings used to cost ten sequential
    tool calls, so in practice it got two (§4GL/§4GN: an eight-minute
    investigation ran exactly two dark-web queries against a brief naming
    BreachForums, Exploit, RaidForums mirrors, paste sites and leak
    indexes). Accepting the phrasings in ONE call makes breadth cost one
    deadline instead of N.
    """
    # ⚠ DEDUPE BEFORE THE CAP. The cap exists to bound FETCH COST (engines ×
    # phrasings); a repeated or whitespace-variant phrasing costs nothing, so
    # letting it eat a slot would silently narrow a brief that listed five
    # wordings, two of which happened to normalise the same way.
    out, seen = [], set()
    if isinstance(extra_queries, str):
        extra_queries = [extra_queries]
    cand = [query] + [q for q in list(extra_queries or []) if isinstance(q, str)]
    for q in cand:
        q = _sanitize_query(q or "")
        k = _norm_cache_key(q)
        if not q or k in seen:
            continue
        seen.add(k)
        out.append(q)
        if len(out) >= 1 + _MAX_EXTRA_QUERIES:
            break
    return out


from ..utils.helpers import FetchError as _FetchError  # noqa: E402  (§4PD)
from .search import neutralise_report_grammar as _neutralise  # noqa: E402  (§4PD: page text is content)


async def tool_darkweb_search(query: Optional[str] = None, extra_queries: Any = None, *args: Any, **kwargs: Any) -> str:
    """darkweb_search, with a past year the model added to a question about
    now dropped from every query first (§4PU, as `web_search`)."""
    from .search import _drop_stale_year_logged, stale_year_note
    from .outcome import append_note
    _stale = []
    if isinstance(query, str) and query:
        query, _stale = _drop_stale_year_logged(query, "darkweb_search")
    if isinstance(extra_queries, str) and extra_queries:
        extra_queries, _x = _drop_stale_year_logged(extra_queries, "darkweb_search")
        _stale += _x
    elif isinstance(extra_queries, list):
        _eq = []
        for q in extra_queries:
            if isinstance(q, str) and q:
                q, _x = _drop_stale_year_logged(q, "darkweb_search")
                _stale += _x
            _eq.append(q)
        extra_queries = _eq
    out = await _darkweb_search_core(query, extra_queries, *args, **kwargs)
    if _stale and isinstance(out, str):
        out = append_note(out, "\n\n" + stale_year_note(sorted(set(_stale))))
    return out


async def _darkweb_search_core(
    query: Optional[str] = None,
    extra_queries: Any = None,
    anonymous: bool = False,
    tor_proxy: Optional[str] = None,
    max_results: int = 12,
    **kwargs: Any,
) -> str:
    """List ranked .onion services matching one or more queries."""
    try:                                               # §4PD: a model-passed 'max_results' as a string or a negative
        max_results = max(1, min(50, int(max_results)))          #       raised inside gather() and read as 'zero results'
    except (TypeError, ValueError):
        max_results = 12
    if not query:
        return "SYSTEM ERROR: The 'query' parameter is MANDATORY. You must specify it."

    # NOTE: the operator/quote/boolean stripping in `_sanitize_query` is
    # clearnet-derived but applies cleanly to Ahmia/Torch too — both are plain
    # keyword indexes that do not honour Google-style operators, so removing
    # them can only help, never lose a supported qualifier.
    queries = _query_set(query, extra_queries)
    if not queries:
        return "SYSTEM ERROR: The 'query' parameter is MANDATORY. You must specify it."
    base_proxy = _normalize_tor_proxy(tor_proxy)
    pretty_log("Darkweb Search", " | ".join(queries), icon=Icons.TOOL_DARKWEB)

    cache_key = f"onion::{max_results}::" + "||".join(_norm_cache_key(q) for q in queries)   # §4UN-dark: size too
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    _BREAKER_CALL.set(set())                     # §4UN-dark: one breaker failure per engine for this call

    # Each phrasing gets its OWN identity-tagged circuit in anonymous mode —
    # the scrub is per query by design, and running the set on one circuit
    # would relink exactly what the tagging exists to separate.
    plans = []
    for q in queries:
        _q, _p = (_apply_anonymous_scrub(q, base_proxy) if anonymous
                  else (q, base_proxy))
        plans.append((q, _sanitize_query(_q), _normalize_tor_proxy(_p)))

    raw = await asyncio.gather(
        *[_darkweb_search_raw(sq, pp, max_results=max_results)
          for _q, sq, pp in plans],
        return_exceptions=True,
    )
    per_query, _skipped_all, _all_skipped_all, _total = [], [], [], 0
    _outcomes: Dict[str, str] = {}
    _transient = False          # r2: any PHRASING with an engine timed out / unreachable (merged = best only)
    for (label, _sq, _pp), res in zip(plans, raw):
        if isinstance(res, BaseException):
            logger.warning("darkweb_search: phrasing %r failed: %s", label, res)
            _transient = True
            continue
        _ranked, _skipped, _all_skipped, _tot = res
        _po = getattr(_ranked, "outcomes", None) or {}
        _transient = _transient or any(o in (OUTCOME_TIMED_OUT, OUTCOME_UNREACHABLE) for o in _po.values())
        _outcomes = merge_outcomes(_outcomes, _po)
        per_query.append((label, _ranked))
        _skipped_all.extend(_skipped)
        _all_skipped_all.append(_all_skipped)
        _total = max(_total, _tot)
    _skipped = sorted(set(_skipped_all))
    # "All engines skipped" only if that was true for EVERY phrasing — one
    # phrasing that reached the engines disproves the breaker-wide claim.
    _all_skipped = bool(_all_skipped_all) and all(_all_skipped_all)

    if len(queries) == 1:
        ranked = per_query[0][1] if per_query else []
    else:
        ranked = _merge_across_queries(
            per_query,
            min(max_results * len(queries), _MULTI_QUERY_RESULT_CEILING))
    if _outcomes:
        pretty_log("Darkweb Engines", outcomes_line(_outcomes), icon=Icons.TOOL_DARKWEB)   # §4UN-dark
    if not ranked:
        return _no_results_error(_skipped, _total, _all_skipped, _outcomes)

    # §4UN-dark: "reached" = the engines that ANSWERED (it was counted from the rows left after the cut)
    reached = sorted(e for e, o in _outcomes.items() if o in _ANSWERED) or \
        sorted({e for r in ranked for e in r.get("engines", [])})
    header = f"[Dark-web search — onion results, engines reached: {', '.join(reached)}]"
    if len(queries) > 1:
        header = (f"[Dark-web search — {len(queries)} phrasings "
                  f"({'; '.join(queries)}), engines reached: "
                  f"{', '.join(reached)}]")
    # §4LG: say what this IS — the reply described "active MH370 threads on
    # TForum" from directory titles, no onion page ever opened
    header += ("\n[These are search-engine TITLES and snippets only — no page was opened. "
               "Do not describe a page's content, posts or discussion from them; open a page "
               "with darkweb_research or browser first, and say so if it fails to load.]")
    cacheable = header + "\n\n" + _format_results(ranked)
    # R2 M6: cache WITHOUT the NARROWED banner. It describes a transient
    # breaker state, and baking it into a 5-minute cache entry kept
    # telling the operator an engine was "not consulted" long after it had
    # recovered — a stale claim about infrastructure, which is the class
    # of lie this whole change set exists to remove.
    _not_answered = {e: o for e, o in _outcomes.items() if o not in _ANSWERED and o != OUTCOME_SKIPPED}
    _live = (f"\n[Engines — {outcomes_line(_not_answered)}: no answer was read from them this time.]"
             if _not_answered else "")
    if _skipped:
        # §4UN-dark: a run with engines in cooldown is not cached — the next call may reach them
        return header + _narrowed_header(_skipped) + _live + "\n\n" + \
            _format_results(ranked)
    if _transient:
        # r1/r2: nor one where an engine timed out or was unreachable on ANY phrasing — the next call may
        # get its index (decided per phrasing: the merged outcome is each engine's best)
        return header + _live + "\n\n" + _format_results(ranked)
    _cache_put(cache_key, cacheable)
    if _live:
        return header + _live + "\n\n" + _format_results(ranked)
    return cacheable


def _fold(text: Any) -> str:
    """Lower-case, Greek/Latin accents removed (νέα → νεα, ειδήσεις → ειδησεις)."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", str(text or "").lower())
                   if unicodedata.category(c) != "Mn")


#: §4UN-darkres: a query word that asks for news/articles/reports (folded: no accents)
_NEWS_INTENT_WORDS = frozenset({
    "news", "article", "articles", "report", "reports", "reporting", "blog", "blogs", "analysis",
    "analyses", "headline", "headlines", "story", "stories", "journalism", "press",
    "νεα", "ειδηση", "ειδησεις", "αρθρο", "αρθρα", "αναλυση", "αναλυσεις", "αναφορα", "αναφορες",
    "ρεπορταζ", "ενημερωση"})
#: ... and one that asks for markets or products — then shop pages ARE the answer
_MARKET_INTENT_WORDS = frozenset({
    "market", "markets", "marketplace", "marketplaces", "shop", "shops", "store", "stores", "vendor",
    "vendors", "buy", "buying", "sell", "selling", "sale", "sales", "price", "prices", "pricing",
    "product", "products", "listing", "listings", "carding",
    "αγορα", "αγορες", "αγοραζω", "πωληση", "πωλησεις", "πουλαω", "τιμη", "τιμες", "προιον",
    "προιοντα", "καταστημα"})
#: a shop/product page: by its path, or by sale words / a price in its title or snippet
# r1: a price or an amount alone is news ("Bitcoin falls below $60,000", "500 BTC stolen"), and
# `/item?id=` is a forum thread as often as a shop row — neither counts
_SHOP_PATH_RE = re.compile(r"/(?:products?|shop|store|cart|checkout|vendors?|listings?)(?:[/?#._-]|$)", re.I)
_SHOP_TEXT_RE = re.compile(r"\bfor sale\b|\badd to cart\b|\bbuy now\b", re.I)


def _news_not_market(query: Any) -> bool:
    words = set(re.findall(r"\w+", _fold(query)))
    return bool(words & _NEWS_INTENT_WORDS) and not (words & _MARKET_INTENT_WORDS)


def _looks_like_shop(r: Dict[str, Any]) -> bool:
    from urllib.parse import urlparse
    try:
        path = urlparse(str(r.get("url") or "")).path
    except Exception:  # noqa: BLE001
        path = ""
    if _SHOP_PATH_RE.search(path):
        return True
    return any(_SHOP_TEXT_RE.search(str(r.get(k) or "")) for k in ("title", "snippet"))


def rank_for_reading(ranked: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """The onions worth OPENING first: on-topic (a distinctive query token in
    the title/snippet/url — the web search's own check) before off-topic, a
    page with a path before a bare homepage, engine agreement after that.
    §4LG: ranked by engine agreement alone, 32 of 36 pages read yielded "No
    relevant information" — 25 of them site homepages.

    §4UN-darkres: when the query asks for news/articles/reports (and NOT for
    markets or products), a shop/product page goes after the others of its
    topic group — demoted, never dropped (req 212e6a8c read "residence
    permits for sale" for "the latest news on the dark web")."""
    from urllib.parse import urlparse
    try:
        from .search import distinctive_tokens, result_on_topic
        dist = distinctive_tokens(query)
    except Exception:  # noqa: BLE001 — no topic check: keep the engines' order
        return list(ranked)
    demote = _news_not_market(query)

    def key(item):
        i, r = item
        on = result_on_topic({"title": r.get("title"), "body": r.get("snippet"), "url": r.get("url")}, dist)
        path = urlparse(str(r.get("url") or "")).path.strip("/")
        return (not on, demote and _looks_like_shop(r), not path, i)
    return [r for _, r in sorted(enumerate(ranked), key=key)]


# r1: no .txt/.json/.xml — on onion sites a paste, dump or canary IS the content
_ASSET_PATH_RE = re.compile(r"\.(?:js|css|png|jpe?g|gif|svg|ico|webp|woff2?|ttf|map)$", re.I)


def is_onion_reading_candidate(url: str, on_topic: bool = False) -> bool:
    """§4NK: an onion URL worth OPENING for research. Measured on the stored
    darkweb_research sources: a bare homepage yielded a fact 0 times in 25
    reads (pages with a path: 11 of 29 loaded); a script/style/image file and a
    site's own search-results page are not documents. An ON-TOPIC homepage
    stays a candidate (§4LG: topic beats path — "MH370 archive" at a bare
    host is read before an off-topic deep page)."""
    from urllib.parse import urlsplit
    try:
        raw = str(url or "")
        parts = urlsplit(raw if "://" in raw else "http://" + raw)
    except Exception:  # noqa: BLE001
        return False
    path = (parts.path or "").strip("/")
    if not path and not parts.query:
        return bool(on_topic)
    if _ASSET_PATH_RE.search(path):
        return False
    return not _search.is_search_results_url(url)


def _filter_block_onions(block: Any, url: str, page_texts: Dict[str, Any]) -> Any:
    """`keep_page_onions` on the extracted-facts part of one source block (its SOURCE line stays)."""
    mark = "[EDGE EXTRACTED FACTS]:"
    if not isinstance(block, str) or mark not in block:
        return block
    head, _, facts = block.partition(mark)
    page = (page_texts.get(url) or ("",))[0]
    return head + mark + keep_page_onions(facts, f"{page} {url}")


def _one_line(text: Any, limit: int = 120) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= limit else t[: limit - 1] + "…"


#: §4UN-darkres r1: a given URL holding any of these is refused whole — a newline in it
#: forged a `### SOURCE:` block, a second count line and fake facts (review M1)
_URL_JUNK_RE = re.compile(r"[\s\x00-\x1f\x7f\u0085\u00a0\u1680\u2000-\u200f\u2028-\u202f\u205f-\u2064\u3000\ufeff]")


def onion_v3_problem(host: str) -> Optional[str]:
    """§4UN-darkres dr2: which check a v3 onion host FAILS, in words, or None when it
    is valid. The bare "not a valid onion address" made the model distrust it, write
    its own (wrong) decoder and repeat its wrong findings (live dr2)."""
    import base64
    import hashlib
    h = str(host or "").strip().lower().rstrip(".")
    if not h.endswith(".onion"):
        return "not an .onion address"
    label = h[: -len(".onion")].rsplit(".", 1)[-1]
    if len(label) != 56:
        return f"wrong length: got {len(label)} characters, a v3 address has 56"
    if not re.fullmatch(r"[a-z2-7]{56}", label):
        return "not base32: a v3 address uses only the letters a–z and the digits 2–7"
    raw = base64.b32decode(label.upper())
    pub, chk, ver = raw[:32], raw[32:34], raw[34]
    if ver != 3:
        return f"wrong version byte: {ver}, a v3 address ends in version 3"
    if hashlib.sha3_256(b".onion checksum" + pub + b"\x03").digest()[:2] != chk:
        return ("the checksum does not match — at least one character is wrong "
                "(a typo or a deliberately altered address)")
    return None


def _short_target(given: str) -> str:
    """How a REFUSED given URL is named in the report: never whole. An onion-like
    host shows the first 8 characters of its label then "….onion" (r1 M2: a
    garbled address echoed in tool output made the verifier treat a reply citing
    it as "seen in tool output" and only note it); another host shows its name."""
    from urllib.parse import unquote
    low = str(given or "")
    for _ in range(3):                  # r3: "%2eonion" / "%252eonion" is an onion host too
        low = unquote(low)
    low = low.lower()
    m = re.search(r"([a-z0-9]+)\.onion\b", low)
    if m:
        return m.group(1)[:8] + "….onion"
    host = re.sub(r"^[a-z][a-z0-9+.-]*:(//)?", "", low).split("/", 1)[0].split("@")[-1]
    host = re.sub(r"[^a-z0-9.:-]", "", host)[:60]
    return host or "(no host)"


def _fresh_circuit(tor_proxy: Optional[str]) -> Optional[str]:
    """The same SOCKS proxy under a NEW identity: Tor isolates it on its own circuit.
    r3 MAJOR: a live proxy already carries an identity (`--anonymous` is the default:
    ``socks5h://<tag>:isolate@…``) and `socks_url_with_identity` returns such a proxy
    UNCHANGED — the "fresh circuit" was the same one. `circuit_for_attempt` (§4UN-read)
    folds the attempt into the existing identity; a proxy without one gets a random tag."""
    if not tor_proxy:
        return tor_proxy
    try:
        from urllib.parse import urlparse
        from ..utils.helpers import circuit_for_attempt, socks_url_with_identity
        if urlparse(tor_proxy).username:
            return circuit_for_attempt(tor_proxy, 1) or tor_proxy
        import secrets
        return socks_url_with_identity(tor_proxy, "retry-" + secrets.token_hex(6)) or tor_proxy
    except Exception:  # noqa: BLE001
        return tor_proxy


def _given_onion_urls(raw: Any) -> Tuple[List[str], List[Tuple[str, str]]]:
    """§4UN-darkres: the pages a caller handed darkweb_research in `urls` →
    (urls to read, in order, deduplicated; [(how to name it, why skipped)]). Only
    http(s) on a v3 onion whose checksum holds; a bare ``<56 chars>.onion/…``
    gets ``http://``; at most `_MAX_GIVEN_URLS`. An invalid one is never fetched.
    r1: a URL with whitespace or a control character is refused; a kept one is
    REBUILT from its parts (scheme, lower-case host without a trailing dot,
    port, path, query — no userinfo, no fragment), so two spellings of one page
    are one URL and nothing but those parts reaches the report."""
    from urllib.parse import urlsplit
    if raw is None or raw == "":
        return [], []
    items = [raw] if isinstance(raw, str) else (list(raw) if isinstance(raw, (list, tuple)) else [raw])
    keep: List[str] = []
    skipped: List[Tuple[str, str]] = []
    for item in items:
        given = str(item if item is not None else "")
        if not given.strip():
            continue
        given = given.strip()
        if _URL_JUNK_RE.search(given):
            skipped.append((_short_target(given), "contains whitespace or control characters"))
            continue
        url = given
        if "://" not in url and re.match(r"^[a-z2-7]{56}\.onion\.?(?:[/:?#]|$)", url, re.I):
            url = "http://" + url
        try:
            parts = urlsplit(url)
            scheme = (parts.scheme or "").lower()
            host = (parts.hostname or "").rstrip(".")
            port = parts.port
        except Exception:  # noqa: BLE001 — a bad port, an unparseable netloc
            scheme, host, port, parts = "", "", None, None
        if "://" in url and scheme and scheme not in ("http", "https"):
            skipped.append((_short_target(given), "only http(s) onion pages are read"))
            continue
        if not (parts is not None and scheme in ("http", "https") and onion_v3_valid(host)):
            _why = onion_v3_problem(host) if parts is not None and host else "no host"
            skipped.append((_short_target(given), f"not a valid onion address: {_why or 'unreadable URL'}"))
            continue
        rebuilt = f"{scheme}://{host}" + (f":{port}" if port else "") + (parts.path or "/") \
            + (f"?{parts.query}" if parts.query else "")
        if rebuilt in keep:
            continue
        if len(keep) >= _MAX_GIVEN_URLS:
            skipped.append((_short_target(rebuilt), f"over the {_MAX_GIVEN_URLS}-URL limit"))   # r2: never whole
            continue
        keep.append(rebuilt)
    return keep, skipped


def _skipped_line(skipped: List[Tuple[str, str]]) -> str:
    """One report line naming each skipped given URL (already in its safe form) and why."""
    if not skipped:
        return ""
    return "[SKIPPED URLS: " + "; ".join(f"{_one_line(u)} — {why}" for u, why in skipped) + "]"


async def tool_darkweb_research(query: Optional[str] = None, *args: Any, **kwargs: Any) -> str:
    """darkweb_research, with the §4PU stale-year drop first."""
    from .search import _drop_stale_year_logged, stale_year_note
    from .outcome import append_note
    _stale = []
    if isinstance(query, str) and query:
        query, _stale = _drop_stale_year_logged(query, "darkweb_research")
    out = await _darkweb_research_core(query, *args, **kwargs)
    if _stale and isinstance(out, str):
        out = append_note(out, "\n\n" + stale_year_note(_stale))
    return out


async def _darkweb_research_core(
    query: Optional[str] = None,
    anonymous: bool = False,
    tor_proxy: Optional[str] = None,
    llm_client=None,
    model_name: str = "default",
    max_context: int = 8192,
    workspace_model=None,
    max_sources: int = 6,
    urls: Any = None,
    **kwargs: Any,
) -> str:
    """Search .onion services, then fetch + distill the top results into a
    synthesised report. The dark-web analogue of `tool_deep_research`.

    §4UN-darkres: ``urls`` (optional, ≤ 8) are onion pages the caller already
    chose (e.g. darkweb_search's results) — read FIRST, in order; a query's
    results fill the remaining slots, and without a query only they are read."""
    try:                                               # §4PD: a model-passed 'max_sources' as a string or a negative
        max_sources = max(1, min(12, int(max_sources)))          #       raised inside gather() and read as 'zero results'
    except (TypeError, ValueError):
        max_sources = 6
    _given, _given_skipped = _given_onion_urls(urls)
    if not query and not _given:
        if _given_skipped:
            return ("SYSTEM ERROR: none of the given 'urls' can be read, and there is no 'query' to "
                    "search instead.\n" + _skipped_line(_given_skipped))
        return ("SYSTEM ERROR: darkweb_research needs a 'query' (keywords to search .onion services) "
                "or 'urls' (onion pages to read). You must specify one.")

    # Stronger anonymous tier: re-author the query with the local model (the
    # same posture deep_research takes), falling back to the lexical scrub.
    if anonymous and query:
        try:
            from ..utils.stylometry import neutralize_query

            query = await neutralize_query(query, llm_client=llm_client, model=model_name) or query
        except Exception:
            pass
        try:            # §4PU r2: the LLM re-author can put the dropped year back
            from .search import _drop_stale_year_logged
            query = _drop_stale_year_logged(query, "darkweb_research")[0]
        except Exception:  # noqa: BLE001
            pass
        _, tor_proxy = _apply_anonymous_scrub(query, tor_proxy)
    elif anonymous:         # urls only: still a per-call circuit identity
        _, tor_proxy = _apply_anonymous_scrub("|".join(_given), tor_proxy)

    query = _sanitize_query(query) if query else ""
    # what the extractor is asked about when the caller gave pages but no query
    _topic = query or "the main facts and news this page reports"
    tor_proxy = _normalize_tor_proxy(tor_proxy)
    pretty_log("Darkweb Research", (query + (f" + {len(_given)} given page(s)" if _given else ""))
               if query else f"{len(_given)} given page(s)", icon=Icons.TOOL_DARKWEB)

    # The deep-read path is expensive (an onion fetch + LLM distill per source);
    # cache the synthesised report so a repeated identical research request
    # doesn't re-fetch every onion. Namespaced distinctly from the list search.
    cache_key = f"onion-research::{max_sources}::" + (query or "").strip().lower()   # r1: the size too
    if _given or _given_skipped:
        cache_key += "::urls::" + "|".join(_given) + "::skip::" + "|".join(u for u, _ in _given_skipped)
    cached = _cache_get(cache_key)
    if cached is not None:
        # ⚠ A CACHE HIT MUST CARRY THE SAME STATUS AS THE RUN IT CACHED
        # (§4GK round 4). This returned the raw report string, so the
        # `darkweb_research_sources_partial` reason_code was dropped and
        # `ToolOutcome.coerce` booked a plain OK — the same query answered OK
        # from cache five minutes after answering partial. The coverage
        # banner is IN the cached text, so the status is re-derivable without
        # storing it separately. An all-failed run is never cached (the
        # `_source_succeeded` gate below), so FAILED cannot occur here.
        from .outcome import ToolOutcome
        if "[⚠ SOURCE FAILURES:" in cached:
            return ToolOutcome.ok(cached, world_changed=False,
                                  reason_code="darkweb_research_sources_partial")
        if _search.NO_FACTS_BANNER_HEAD in cached:
            return ToolOutcome.ok(cached, world_changed=False,
                                  reason_code="darkweb_research_no_relevant_facts")
        return ToolOutcome.ok(cached, world_changed=False)

    _BREAKER_CALL.set(set())                     # §4UN-dark: one breaker failure per engine for this call
    if query:
        ranked, _skipped, _all_skipped, _total = await _darkweb_search_raw(
            query, tor_proxy, max_results=max_sources * 3)
    else:
        ranked, _skipped, _all_skipped, _total = [], [], False, 0
    if not ranked and not _given:
        # J1: this caller kept blaming the query after the sibling was
        # fixed — and it is the follow-up `darkweb_search`'s own tool
        # description recommends, so it is the one the model reaches for
        # after a thin result set.
        return _no_results_error(_skipped, _total, _all_skipped, getattr(ranked, "outcomes", None))

    _ranked_rows = [r for r in rank_for_reading(ranked, query) if r.get("url")]
    _ordered = [r["url"] for r in _ranked_rows]
    try:
        from .search import distinctive_tokens, result_on_topic
        _dist = distinctive_tokens(query)
        _on = {r["url"]: result_on_topic({"title": r.get("title"), "body": r.get("snippet"),
                                          "url": r.get("url")}, _dist) for r in _ranked_rows}
    except Exception:  # noqa: BLE001 — no topic check: every homepage counts as on-topic
        _on = {u: True for u in _ordered}
    # §4NK: off-topic homepages, asset files and search pages are not read
    # while a real page is available; the rest after the first `max_sources`
    # is the backfill queue for sources that fail to load
    _eligible = [u for u in _ordered if is_onion_reading_candidate(u, _on.get(u, False))] or _ordered
    # §4UN-darkres: the caller's pages first, in order (never demoted or filtered); the
    # search fills the remaining slots and the backfill queue
    _eligible = [u for u in _eligible if u not in _given]
    _fill = max(0, max(max_sources, len(_given)) - len(_given))
    urls = _given + _eligible[:_fill]
    _reserve_urls = _eligible[_fill:_fill + _search.RESEARCH_MAX_RESERVES]

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

    # §4UN-darkres: 4 at a time (was 2 — six dead onions held the two slots for
    # most of 212e6a8c's 140 s). Each URL's budget is issued when it starts, so
    # width no longer starves the later ones; the distill plans declare it.
    sem = asyncio.Semaphore(_RESEARCH_FETCH_WIDTH)
    _DISTILL_FANOUT = _RESEARCH_FETCH_WIDTH      # == the semaphore width above
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

    _page_texts: dict = {}     # url -> (text sent to the extractor, max_tokens) — §4NK recheck
    # §4UN-darkres: what became of each url — loaded / timeout / failed / skipped —
    # set by this code (never parsed out of page text), for the report's count line
    _outcome: Dict[str, str] = {}
    _retried: set = set()
    _given_set = set(_given)

    def _retry_need() -> float:
        return _ONION_FIRST_BYTE_S + _RETRY_READ_S + (_DISTILL_RESERVE_S if llm_client is not None else 0.0)

    def _may_retry(url: str, url_deadline: float) -> bool:
        """dr2: one retry after a first-byte timeout when the URL's time (and the
        phase's) still covers a first-byte wait + a read + the distill reserve. A
        given page qualifies on that alone; a search-filled one only while the phase
        also covers a retry for every given page not yet settled (given first)."""
        if url in _retried:
            return False
        now = time.monotonic()
        need = _retry_need()
        if min(url_deadline, _phase_deadline) - now - 2.0 < need:
            return False
        if url in _given_set:
            return True
        unsettled = sum(1 for g in _given if g not in _outcome)
        return _phase_deadline - now >= need * (1 + unsettled)

    async def _fetch_with_timeout(url: str, cap: float, proxy: Optional[str] = None) -> str:
        try:
            # ⚠ COERCE, as the clearnet sibling does. This returned the awaited
            # value verbatim while `process_url` guards it with
            # `isinstance(text, str)` and then indexes it seven lines later —
            # one `return None` from dropping the onion. The `except Exception`
            # net in `_bounded` now REPORTS such a source instead of losing it,
            # but reporting an internal error is not the same as keeping the
            # page: coercing here is what preserves the raw-text fallback.
            _res = await asyncio.wait_for(
                _fetch_onion_text(url, proxy if proxy is not None else tor_proxy), timeout=cap
            )
            if isinstance(_res, _RawPage):
                # r2: stripped here, outside the fetch's timeout
                _cut = bool(getattr(_res, "cut", False))
                _res = await asyncio.to_thread(_strip_html, str(_res))
                _res = _CutPage(_res) if _cut else _res
            if isinstance(_res, _FetchError):
                # the fetcher's own error text (never page text): a timeout or another failure
                _outcome[url] = "timeout" if isinstance(_res, _FetchTimeout) else "failed"   # r1: by type
                return _res
            _res = _res if isinstance(_res, str) else str(_res)
            if _search.is_bot_check_page(_res):
                # §4NK: a DDoS-guard interstitial is not the hidden service's page
                _outcome[url] = "failed"
                return _FetchError(f"Error: bot-check page from {url} (the service asked to verify a browser)")
            return _res
        except asyncio.TimeoutError:
            _outcome[url] = "timeout"
            return _FetchTimeout(f"Error: Fetch of {url} timed out after {cap:.0f}s")
        except Exception as e:  # noqa: BLE001
            _outcome[url] = "failed"
            return _FetchError(f"Error: {e}")

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
        # §4UN-darkres r1: the distiller's reserve shortens only the wait for the FIRST
        # byte (never below `_FIRST_BYTE_FLOOR_S`). A page that answered reads its body
        # until the full read time or the URL's budget (- 2 s), whichever is first; one
        # still loading then is KEPT, partial and marked (never-lose: review M3 — a page
        # answering at 8 s and done at 30 s was cut at 28 s and booked a timeout).
        async def _attempt(proxy: Optional[str] = None):
            _left = _url_deadline - time.monotonic() - 2.0
            _body_s = max(0.1, min(_ONION_PAGE_TIMEOUT + 5, _left))
            _fb = _ONION_FIRST_BYTE_S
            if llm_client is not None:
                _fb = min(_fb, max(_FIRST_BYTE_FLOOR_S, _left - _DISTILL_RESERVE_S))
            # r2: the body stops 4 s before the fetch's own timeout — room for the bounded close
            # (`_ACLOSE_WAIT_S`) so a cut page comes back as a page, not as a timeout
            _PAGE_LIMITS.set((min(_fb, _body_s), time.monotonic() + max(_body_s * 0.5, _body_s - 4.0)))
            return await _fetch_with_timeout(url, _body_s, proxy)

        text = await _attempt()
        if isinstance(text, _FirstByteTimeout) and _may_retry(url, _url_deadline):
            # dr2: the Tor Project's onion gave no first byte in 15 s and loaded in 2 s
            # a minute later — retry ONCE, now, on a fresh circuit (new SOCKS identity)
            _retried.add(url)
            pretty_log("Onion Retry", f"{short_url} — no first byte; once more on a fresh circuit",
                       icon=Icons.TOOL_DARKWEB)
            _outcome.pop(url, None)                     # counted once: the retry's outcome stands
            text = await _attempt(_fresh_circuit(tor_proxy))

        # ⚠ Same boundary as the clearnet sibling: a failed fetch is an
        # error STRING, and feeding it to a worker prompted "extract
        # the hard facts, and if none are found say so" turns "this
        # onion did not answer" into "this onion contains nothing
        # relevant" — positive negative evidence, manufactured from a
        # timeout. It also made the cache gate below inoperative,
        # since the distiller's output never starts with "Error:".
        if isinstance(text, _FetchError):                      # §4PD: the fetcher's TYPE, not the page's first word
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
        raw_preview = _neutralise(raw_preview)        # §4PD: page text is content (every return site)
        if isinstance(text, _CutPage):
            # r1 M3: what arrived before the read time ran out — kept as raw text, said so
            pretty_log("Onion Page Cut", f"{short_url} — still loading at its read time; partial text kept",
                       level="WARNING", icon=Icons.WARN)
            return (f"### SOURCE: {url}\n[PAGE CUT OFF: the read time ran out while this page was still "
                    f"loading — partial, raw page text below]\n{raw_preview}\n")

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
                if _summary_budget - _QUEUE_ALLOWANCE_S <= 0:
                    # r1: a page that used its whole read time leaves no distill — keep it raw,
                    # never plan on a negative clock
                    pretty_log("Summary Degraded", f"{short_url}: no time left to distil after the "
                               f"page loaded — raw page text kept", level="WARNING", icon=Icons.WARN)
                    return f"### SOURCE: {url}\n{raw_preview}\n"
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
        _page_texts[url] = (safe_text, _tokens, getattr(plan, "predicted_s", 0.0))   # §4NK recheck
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": _search.extract_prompt(_topic, safe_text)}],
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
            pretty_log("Research Compute",
                       f"Distilled facts from {short_url} — {_sized}",
                       icon=Icons.TOOL_DEEP)
            try:
                # The last hand-tuned constant in the sizing path, measured
                # instead: we know the chars we sent, llama.cpp reports the
                # tokens they became.
                llm_client.note_distill_density(len(safe_text), summary_data)
            except Exception:                                   # noqa: BLE001
                pass
            # §4UN-dark: an onion address the summary holds must be VERBATIM in the page (and pass the
            # checksum) — a worker copying an address and changing a letter put a fake one in tool output,
            # which every later link guard then accepted as "seen"
            preview = "[EDGE EXTRACTED FACTS]:\n" + _neutralise(keep_page_onions(
                (summary_data["choices"][0]["message"].get("content") or "").strip(), f"{text} {url}"))
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
            preview = _neutralise(raw_preview)
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
                _outcome[str(url)] = "skipped"
                return (f"### SOURCE: {url}\nError: research phase deadline "
                        f"({_RESEARCH_PHASE_TIMEOUT_S:.0f}s) reached before "
                        f"this source could be fetched\n")
            try:
                block = await asyncio.wait_for(process_url(url, budget),
                                               timeout=budget)
                _outcome.setdefault(str(url), "failed" if _source_block_failed(block) else "loaded")
                return block
            except asyncio.TimeoutError:
                # Say the BUDGET. Without it an onion killed on a starved
                # window reads exactly like one that used its full ceiling —
                # the confusion req 08766aa1 turned on.
                _outcome[str(url)] = "timeout"
                return (f"### SOURCE: {url}\nError: per-URL timeout exceeded "
                        f"({budget:.0f}s)\n")
            except Exception as _url_exc:                        # noqa: BLE001
                _outcome[str(url)] = "failed"
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

    _slot_stats: dict = {}
    _slots = await _search.run_research_slots(
        urls, _reserve_urls, _bounded, phase_deadline=_phase_deadline,
        min_budget_s=_MIN_URL_BUDGET_S, tag="Onion Backfill", stats=_slot_stats)
    urls, page_contents, _lost, _n_slots, _failed_slots, _replaced = _search.slot_coverage(_slots)
    _recheck = None
    _pre_loaded = [c for c in page_contents if isinstance(c, str) and not _source_block_failed(c)]
    if _search.research_no_facts(bool(llm_client), _pre_loaded,
                                 [c for c in _pre_loaded if "[EDGE EXTRACTED FACTS]" not in c] if llm_client else [],
                                 _failed_slots, _lost):
        _recheck = await _search.recheck_all_none(
            page_contents, urls, _page_texts, _topic, llm_client, model_name,
            deadline=_phase_deadline, slot_wait=_QUEUE_ALLOWANCE_S,
            per_call_s=_WEB_SUMMARY_TIMEOUT_S, fanout=_DISTILL_FANOUT, label="onion summary recheck")
        page_contents = _recheck.contents
        # §4UN-dark: the recheck's summaries pass the same verbatim-address filter as the first ones
        page_contents = [_filter_block_onions(c, u, _page_texts) for c, u in zip(page_contents, urls)] + \
            list(page_contents[len(urls):])
    valid_contents = [c for c in page_contents if isinstance(c, str)]
    # ⚠ PORTED FROM THE CLEARNET SIBLING. Without a banner a partly-failed
    # onion run renders as a confident SHORT report and an all-failed one as
    # an EMPTY report under a confident header. §4NK: coverage is counted in
    # SLOTS (a failed onion whose slot a later result filled is a replacement)
    _failed = [c for c in valid_contents if _source_block_failed(c)]
    _degraded = [c for c in valid_contents
                 if not _source_block_failed(c)
                 and "[EDGE EXTRACTED FACTS]" not in c] if llm_client else []
    # §4UN-darkres r1: the per-URL counts the [SOURCES:] line shows — one set of totals
    _oc = list(_outcome.values())
    _n_loaded, _n_to = _oc.count("loaded"), _oc.count("timeout")
    _n_fail = _oc.count("failed") + _lost
    _n_dskip = _oc.count("skipped")                      # the phase deadline never started it
    _n_skip = _n_dskip + len(_given_skipped)
    _n_tried = _n_loaded + _n_to + _n_fail
    _banner = ""
    if _failed_slots or _lost:
        _banner = (
            f"[⚠ SOURCE FAILURES: {_n_to + _n_fail + _n_dskip} of {_n_tried + _n_dskip} "
            f"hidden services could not be fetched (timed out {_n_to}, failed {_n_fail}"
            + (f", not started before the deadline {_n_dskip}" if _n_dskip else "") +
            f"). The sections below cover only the "
            f"{_n_loaded} that loaded. This is a "
            f"COVERAGE limit, not evidence of absence — do not conclude a "
            f"claim is unsupported because a failed source did not support "
            f"it.]\n\n")
    if _replaced:
        _banner += (f"[{_replaced} hidden service(s) failed to load and were replaced by the "
                    f"next search results.]\n\n")
    if _degraded:
        _banner += (
            f"[⚠ {len(_degraded)} of {_n_slots} hidden services could not be "
            f"distilled in the time available and appear below as RAW, "
            f"TRUNCATED page text rather than extracted facts. Treat those "
            f"sections as unfiltered page dumps.]\n\n")
    _loaded = [c for c in valid_contents if not _source_block_failed(c)]
    _no_facts = _search.research_no_facts(bool(llm_client), _loaded, _degraded, _failed_slots, _lost)
    if _no_facts:
        _banner += (
            f"{_search.NO_FACTS_BANNER_HEAD} none of the {len(_loaded)} hidden services that loaded "
            f"had facts relevant to the query. Do not present an answer as researched: try a "
            f"narrower or differently worded query, or say what could not be found.]\n\n")
    # §4UN-darkres: exact counts at the head — tried = every fetch started (backfills
    # included); skipped = a given URL refused + a source the phase deadline never started.
    # From `_outcome` (set by this code), never from page text; a page line starting
    # "[SOURCES:" is quoted by `_neutralise`.
    _counts = (f"[SOURCES: tried {_n_tried}, loaded {_n_loaded}, timed out {_n_to}, "
               f"failed {_n_fail}, skipped {_n_skip}]\n")
    if _given_skipped:
        _counts += _skipped_line(_given_skipped) + "\n"
    full_report = (_counts + "\n" + _banner + (_search.recheck_note(_recheck) if _recheck else "")
                   + "\n\n".join(valid_contents))

    # Workspace research dedup — record every onion we pulled. Non-fatal.
    if workspace_model is not None and getattr(workspace_model, "enabled", False):
        try:
            from .search import record_loaded_sources       # §4LP: loaded sources only
            record_loaded_sources(workspace_model, urls, page_contents, _source_block_failed,
                                  source="darkweb_research", note=(query or "")[:120])
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
    _phase_truncated = (any("research phase deadline" in c for c in valid_contents)
                        or bool(_slot_stats.get("deadline_cut"))    # §4NK r1: a reserve the deadline stopped
                        or bool(_recheck and _recheck.unchecked)    # r2: sources left with a faulty "none"
                        or _no_facts)                               # r2: a "no facts" may be a worker fault — never cached

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
        return bool(preview) and not _source_block_failed(block)

    if not _phase_truncated and any(_source_succeeded(c)
                                    for c in valid_contents):
        _cache_put(cache_key, result)
    # §4GI round 3 — THE SIBLING ONE REVISION BEHIND. `deep_research` got
    # its status in §4GI; this function builds the very same banner and
    # still returned a success-shaped string on EVERY path, so an onion
    # blackout (every hidden service down — the common case for this tool)
    # was booked as a competent research call by `ToolOutcome.coerce`.
    # Same rule as the clearnet sibling, for the same reason: nothing
    # fetched is a FAILED action; partial coverage is a coverage limit the
    # banner already reports, NOT a failed action (a non-OK status there
    # fires the strike ledger and books `success: False` in the corpus).
    from .outcome import ToolOutcome
    _n_ok = len(valid_contents) - len(_failed)
    _status_kw = {"world_changed": False}
    if urls and _n_ok == 0:
        _finish = lambda t: ToolOutcome.failed(                # noqa: E731
            t, reason_code="darkweb_research_all_sources_failed", **_status_kw)
    elif _failed_slots or _lost:
        _finish = lambda t: ToolOutcome.ok(                    # noqa: E731
            t, reason_code="darkweb_research_sources_partial", **_status_kw)
    elif _no_facts:
        _finish = lambda t: ToolOutcome.ok(                    # noqa: E731
            t, reason_code="darkweb_research_no_relevant_facts", **_status_kw)
    else:
        _finish = lambda t: ToolOutcome.ok(t, **_status_kw)    # noqa: E731
    # R3 MAJOR: research never emitted the NARROWED banner — and it is the
    # tool that depends MOST on cross-engine corroboration, since its
    # ranking picks which onions get deep-read and synthesised into a
    # report. Skipped engines meant that ranking silently degenerated to
    # discovery order and the report was built on it. Appended AFTER the
    # cache write, like the sibling: the banner describes a transient
    # breaker state and must not be served back for 5 minutes.
    if _skipped:
        return _finish(result + "\n\n" + _narrowed_header(_skipped).lstrip("\n"))
    return _finish(result)
