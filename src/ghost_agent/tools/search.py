import asyncio
import importlib.util
import json
import logging
import os
import copy
import re
import time
from typing import List, Dict, Any, Callable, Optional, Tuple
from ..utils.logging import Icons, pretty_log
from ..utils.helpers import helper_fetch_url_content

logger = logging.getLogger("GhostAgent")
from ..core.node_throughput import (
    CHARS_PER_TOKEN, MIN_CHARS as _MIN_DISTILL_CHARS, env_float,
    log_plan as log_distill_plan)

# Low-value / bot-walled domains filtered out of search results. Module-level
# (NOT function-local) so web_search's query-reformulation fallback can
# reference it even when the primary attempt raised before any local
# assignment — previously a function-local `junk` left that fallback raising
# UnboundLocalError on exactly the failure (Tor/DDGS down) it exists to handle.
_JUNK_DOMAINS = [
    "duckduckgo.com", "whatsapp.com", "twitter.com", "facebook.com",
    "tiktok.com", "instagram.com", "zhihu.com", "baike.baidu.com",
    "dict.cn", "pinterest.com", "aliexpress.com", "zhidao.baidu.com",
    "yahoo.com", "forbes.com", "bloomberg.com", "scmp.com", "quora.com",
    "medium.com", "msn.com", "cnn.com", "foxnews.com", "wsj.com",
    "csdn.net", "sohu.com", "sina.com", "forums.att.com",
]

# Search engines raced over Tor. Engine reachability over Tor is
# EXIT-NODE-dependent and BAD: measured 2026-07-08 (42 probes, 7 engines ×
# 2 queries × 3 fresh circuits each) the per-(engine,circuit) success rate
# was ~10% — brave 2/6, yahoo 1/6, mojeek 1/6, everything else 0/6 — and
# WHICH engine wins flips from circuit to circuit. Two structural
# consequences drive the design here:
#   1. RACE, don't fan out on one circuit. ddgs's own multi-backend mode
#      runs every engine through the ONE proxy set on the DDGS instance,
#      so a blocked exit IP fails all engines TOGETHER (correlated
#      failure — exactly what we can't afford at 10% per ticket). Instead
#      _race_search_wave fires one single-engine ddgs call PER ENGINE,
#      each tagged onto its OWN Tor circuit, first non-empty wins:
#      6 independent ~10% tickets ≈ 47% per wave vs ~10% correlated.
#   2. Keep the set WIDE; fast failures are cheap lottery tickets.
#      yahoo is back (re-measured 2026-07-08: fails FAST, ~1.4-2.2s
#      RequestError — the old "hangs until timeout" behaviour is gone —
#      and it actually won a probe). Still excluded:
#        * wikipedia  — treats region="wt-wt" as language "wt" and builds
#          `https://wt.wikipedia.org/...`, which doesn't exist: always a
#          ConnectError.
#        * grokipedia — typeahead API, 0/6 on real queries.
_RACE_ENGINES: Tuple[str, ...] = ("mojeek", "duckduckgo", "yandex", "brave", "google", "yahoo")
# Legacy comma-joined form (kept for callers/docs that referenced the old
# single-call multi-backend constant).
_TOR_BACKENDS = ",".join(_RACE_ENGINES)

# Per-request ddgs timeout, in seconds. CRITICAL for Tor reliability and
# measured directly: the engine that actually returns results over Tor
# (usually mojeek) responds in ~10-18s through a Tor circuit, while the
# others fail fast (~1-6s). The previous 8s ceiling KILLED mojeek mid-
# request — producing "error sending request for url (mojeek...)" — so
# EVERY engine came back empty and the search failed even though results
# were reachable; the agent then burned minutes on retries that could
# never win. 18s comfortably clears mojeek's Tor latency; the fast-failing
# engines still fail fast, so a successful search costs ~12-18s and only an
# all-circuits-blocked search pays the full timeout. Do not drop below ~15.
_DDGS_TOR_TIMEOUT = 18

# Per-engine timeout override (2026-07-15). The 2026-07-08 measurement (above)
# attributes the 12-18s latency to MOJEEK — the reliable slow winner; the other
# engines either win FAST or fail fast (~1-6s). A race loser's thread is
# uncancellable (it runs to its ddgs timeout on the dedicated _RACE_POOL), so
# giving the fast engines a shorter ceiling frees their thread ~6s sooner on a
# blocked wave without costing wins (a non-mojeek win arrives well under 12s).
# mojeek keeps the full budget so it can still win late, and the wave deadline
# stays sized off mojeek (_DDGS_TOR_TIMEOUT + grace). Measure per-exit before
# tightening the fast engines further; do NOT shorten mojeek below ~15.
_DDGS_ENGINE_TIMEOUT = {"mojeek": 18}
_DDGS_FAST_ENGINE_TIMEOUT = 12


def _engine_timeout(engine: str) -> int:
    return _DDGS_ENGINE_TIMEOUT.get(engine, _DDGS_FAST_ENGINE_TIMEOUT)

# Small in-process TTL cache so the model's habit of firing many
# near-identical queries in one turn doesn't re-pay the full Tor round
# trip each time. Keyed on the normalized (sanitized, lower-cased) query.
# Only SUCCESSFUL results are cached — never error strings.
_SEARCH_CACHE_TTL = 300.0  # seconds
# 128 (was 64): a research-heavy task fans out many queries; a small cap let
# it evict its OWN recent results and re-pay the Tor lottery mid-task.
_SEARCH_CACHE_MAX = 128
_SEARCH_CACHE: Dict[str, Tuple[float, str]] = {}


def _norm_cache_key(query: str) -> str:
    """Normalize a query into a cache key that collapses TRIVIAL variation
    so near-duplicate queries within a task hit the cache instead of
    re-paying the ~10%-per-exit Tor search lottery. Lower-cases, collapses
    internal whitespace, strips surrounding punctuation, and drops a
    trailing '?' — so 'python asyncio', 'Python  asyncio' and 'python
    asyncio?' share one entry. Meaning-bearing tokens are untouched."""
    q = re.sub(r"\s+", " ", (query or "").strip().lower())
    return q.strip(" \t\n?.!,;:\"'")


def _sanitize_query(query: str) -> str:
    """Strip search operators the ddgs scraper backends choke on.

    The LLM is prone to emitting Google-style operators — `site:`,
    quoted phrases, boolean `OR`/`AND` (e.g.
    ``foo "bar" site:x.com or site:y.com``). The DuckDuckGo / Brave /
    Mojeek HTML scrapers don't honour these the way a real search API
    does: at best they're ignored, at worst the whole query returns ZERO
    results. We reduce to plain keywords so the backends can match. If
    stripping empties the query, the original is returned unchanged.
    """
    if not query:
        return query
    q = query
    # Drop site:/inurl:/intitle:/filetype: operators along with their argument
    _operands = re.findall(
        r'\b(?:site|inurl|intitle|filetype|ext)\s*:\s*(\S+)', q,
        flags=re.IGNORECASE)
    q = re.sub(r'\b(?:site|inurl|intitle|filetype|ext)\s*:\s*\S+', ' ', q, flags=re.IGNORECASE)
    # Drop standalone boolean operators. Case-insensitive but boundary-gated,
    # so only the free-standing token `or`/`and`/`OR`/`AND` goes — never an
    # `or` buried inside a word, and the loss of a stopword in natural prose
    # (e.g. "law and order") is invisible to the DDG/Brave/Mojeek scrapers,
    # which treat or/and as stopwords regardless.
    q = re.sub(r'(?<!\w)(?:or|and)(?!\w)', ' ', q, flags=re.IGNORECASE)
    # Drop quotes but keep the words inside them
    q = q.replace('"', ' ').replace("“", ' ').replace("”", ' ')
    # Collapse whitespace
    q = re.sub(r'\s+', ' ', q).strip()
    if q:
        return q
    # ⚠ THE OPERATOR WAS THE WHOLE QUERY. Returning `query` unchanged here
    # (what this did until 2026-08-15) hands the scrapers the one shape
    # they cannot honour, so the wave is a guaranteed zero. Observed live:
    # `site:reddit.com/r/lgbtgreece/comments/1voyjgf/is_nudism_safe_...`
    # went out verbatim, returned nothing across four waves and two
    # reformulations (one of which was "how to site:reddit.com/..."), and
    # burned ~80s before the turn failed.
    #
    # The operand is not noise — it is the query, spelled as a URL. Mine
    # it: the path slug carries the actual search terms. Only reached when
    # stripping empties the query, so the common `foo site:x.com` case is
    # untouched.
    mined = _keywords_from_operand(" ".join(_operands))
    return mined or query


#: URL furniture that carries no search signal once the path is split.
_OPERAND_STOPWORDS = frozenset({
    "www", "com", "org", "net", "edu", "gov", "io", "co", "uk", "html",
    "htm", "php", "aspx", "index", "http", "https", "amp", "r", "wiki",
    "comments", "comment", "post", "posts", "article", "articles", "blog",
    "page", "pages", "en", "index", "watch", "video", "topic", "thread",
})


def _keywords_from_operand(operand: str) -> str:
    """Recover search terms from a `site:`/`inurl:` argument.

    `reddit.com/r/lgbtgreece/comments/1voyjgf/is_nudism_safe_in_greece`
    → `reddit lgbtgreece nudism safe greece`. Keeps the domain label
    (it biases results toward the right place) and the slug words; drops
    URL furniture and opaque ids, which are what make the query
    unmatchable in the first place.
    """
    if not operand:
        return ""
    text = re.sub(r'^[a-z]+://', ' ', operand, flags=re.IGNORECASE)
    parts = [p for p in re.split(r'[/._\-+?=&#,:~%\s]+', text) if p]
    out, seen = [], set()
    for p in parts:
        low = p.lower()
        if low in _OPERAND_STOPWORDS or len(low) < 3:
            continue
        # Opaque identifiers: reddit post ids, hashes, revision numbers.
        #
        # ⚠ `and not low.isalpha()` was DEAD — a token containing a digit
        # is never isalpha() — so this dropped ANY token with a digit, not
        # just opaque ones: log4shell, gpt4, ipv6, sha256, 2024 all went.
        # `site:en.wikipedia.org/wiki/Log4Shell` mined to `wikipedia`,
        # turning an honest zero (which made the model reformulate) into
        # eight confident results about Wikipedia. A quiet wrong answer is
        # worse than a loud failure.
        #
        # An opaque id is mostly-digits or long-and-mixed; a real term
        # that happens to carry a digit is not.
        _digits = sum(ch.isdigit() for ch in low)
        if _digits and (_digits * 2 >= len(low) or len(low) >= 12):
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(low)
    # A URL slug can be long; the engines already do badly past ~6 terms.
    mined = out[:6]
    # If everything distinctive was filtered away, the mined query is a
    # worse failure than none: `site:en.wikipedia.org/wiki/X` -> "wikipedia"
    # searches for the wrong thing CONFIDENTLY. Only the domain label
    # surviving means exactly that.
    if len(mined) <= 1:
        return ""
    return " ".join(mined)


def _cache_get(key: str) -> Optional[str]:
    entry = _SEARCH_CACHE.get(key)
    if not entry:
        return None
    ts, value = entry
    if (time.monotonic() - ts) > _SEARCH_CACHE_TTL:
        _SEARCH_CACHE.pop(key, None)
        return None
    return value


def _cache_put(key: str, value: str) -> None:
    # Bounded FIFO eviction — drop the oldest entry when full.
    if len(_SEARCH_CACHE) >= _SEARCH_CACHE_MAX:
        oldest = min(_SEARCH_CACHE, key=lambda k: _SEARCH_CACHE[k][0])
        _SEARCH_CACHE.pop(oldest, None)
    _SEARCH_CACHE[key] = (time.monotonic(), value)


def _proxy_for_attempt(base_proxy: Optional[str], query: str, attempt: int,
                       salt: str = "") -> Optional[str]:
    """Return the SOCKS proxy URL for a given retry attempt, tagged so each
    attempt rides a DISTINCT Tor circuit (a fresh exit node).

    Search-engine reachability over Tor is exit-node-dependent: a query
    that fails on one exit (block / CAPTCHA / connect error) routinely
    succeeds on the next. Retrying on the SAME circuit is therefore
    near-useless — yet that's exactly what happened before, because the
    per-query SOCKS tag was identical across attempts. Here we fold the
    attempt index into the SOCKS ``username:password`` so Tor's
    ``IsolateSOCKSAuth`` (on by default) maps each attempt to its own
    circuit. Cheap, control-port-free, and verified to yield different
    exit IPs per tag — the alternative to a slow global NEWNYM.

    ``salt`` extends the tag for callers that need MORE isolation than
    per-(query, attempt): the engine race folds the engine name in, so
    every engine in a wave rides its own circuit (uncorrelated failures)
    instead of all sharing one exit IP.

    Any credentials already on the incoming proxy are stripped and
    replaced: the ``tool_search`` wrapper may have applied a per-query
    tag, but we fold the query hash into our own tag so per-query
    isolation is preserved while still rotating per attempt.
    """
    if not base_proxy:
        return base_proxy
    try:
        import hashlib
        from urllib.parse import urlparse, urlunparse
        from ..utils.helpers import socks_url_with_identity
        p = urlparse(base_proxy)
        if not p.hostname:
            return base_proxy
        bare = urlunparse((p.scheme, f"{p.hostname}:{p.port or 9050}", "", "", "", ""))
        qh = hashlib.md5((query or "").encode("utf-8", "ignore")).hexdigest()[:8]
        # 100ms time bucket (was per-MINUTE). Per-minute meant two same-query
        # searches fired seconds apart inside one minute (a common immediate
        # model retry) got identical SOCKS-auth tags → the same Tor circuits
        # → the same blocked exits, partially defeating the "fresh exits beat
        # a block" design. 100ms is coarse enough that a synchronous wave's
        # back-to-back calls stay on one circuit (the determinism the
        # per-URL fetch relies on) but fine enough that a retry seconds later
        # rides a fresh circuit.
        tbucket = int(time.monotonic() * 10)
        return socks_url_with_identity(
            bare, f"{qh}{salt}a{attempt}n{_PROC_NONCE}t{tbucket}") or base_proxy
    except Exception:
        return base_proxy


def _filter_junk(raw_results) -> List[Dict]:
    """Drop results with missing/relative URLs or junk-domain hosts."""
    valid = []
    for r in raw_results or []:
        # `or` chain, not `.get('href', default)`: a result dict with an
        # explicit href=None (some backends emit that for a malformed hit)
        # made `.get('href', ...).lower()` raise AttributeError — which, at
        # the try-guarded call site, sank the WHOLE engine's result batch for
        # that wave, not just the one bad row.
        url = (r.get('href') or r.get('url') or '').lower()
        if not url or url.startswith("/") or any(j in url for j in _JUNK_DOMAINS):
            continue
        valid.append(r)
    return valid


def _brief_engine_error(e: BaseException) -> str:
    """One readable line per losing engine: URLs stripped (a Tor search URL
    is long enough to swallow the whole log-line budget — the field symptom
    was yahoo errors truncated to `url (h`), whitespace collapsed, capped."""
    s = str(e) or e.__class__.__name__
    # Closing quotes/brackets stay OUT of the match so a repr like
    # "url (https://x.com/y)')" reads "url (<url>)')", not "url (<url>".
    s = re.sub(r"""https?://[^\s'")\]]+""", "<url>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:80]


def _failure_category(msg: str) -> str:
    """Collapse a losing engine's error into a terse category for the
    operator stream. Over Tor the failure reprs are long but boring — the
    operator only needs to know HOW an engine lost, not the exception
    plumbing. Unknown errors stay category "error" and keep a snippet."""
    m = (msg or "").lower()
    if not m or m == "empty" or "no results found" in m:
        return "empty"
    if "timed out" in m or "timeout" in m:
        return "timeout"
    if ("connect" in m or "requesterror" in m or "ssl" in m
            or "error sending request" in m):
        return "conn-error"
    return "error"


# A failed wave is bounded by the slowest engine (ddgs per-request timeout)
# plus a small grace for thread scheduling; a wedged thread must never make
# the caller wait forever.
_RACE_WAVE_GRACE = 4

# Dedicated pool for race threads. Cancelling a loser only cancels the
# asyncio wrapper — the thread runs its ddgs call to completion (up to the
# full 18s timeout) — so a few concurrent waves of 6 engines saturated the
# loop's shared to_thread pool (min(32, cpu+4)) and stalled every OTHER
# to_thread user in the process (found 2026-07-15). This pool is ISOLATED
# (only search-race threads), so sizing it generously can't starve
# unrelated work. Sized for 8 concurrent waves (was 4): deep_research fans
# out AND swarm/delegation workers each fire searches, so >4 concurrent
# waves is realistic, and an over-subscribed wave sits QUEUED here while
# its own 22s deadline ticks down — returning [] having made zero network
# requests. (The deeper fix is a cancellable racer so losers don't linger
# up to 18s each; until then, headroom is the mitigation.)
from concurrent.futures import ThreadPoolExecutor as _TPE
_RACE_POOL = _TPE(max_workers=len(_RACE_ENGINES) * 8,
                  thread_name_prefix="search-race")

# Fresh per process AND per ~minute: an identical SOCKS-auth tag maps to
# the SAME Tor circuit while it lives (~10 min dirtiness), so a retried
# failed query rode the exact same dead exits (found 2026-07-15).
# Uniqueness is all Tor needs; per-query isolation is kept via the query
# hash in the tag.
_PROC_NONCE = os.urandom(2).hex()

# deep_research page-fetch resilience (2026-07-08). Fetch reachability over
# Tor is exit-node-dependent just like search: a URL that times out / 503s
# on one exit often serves fine on the next. Each URL gets its own circuit
# and a circuit-retryable failure is retried on a fresh exit.
_FETCH_ATTEMPTS = 2
# Above the curl_cffi client's own 20s timeout so the client's cleaner
# error surfaces first; the old outer 15s < 20s killed slow-but-live Tor
# fetches before the client could complete (the mojeek-timeout bug's twin).
_FETCH_ATTEMPT_TIMEOUT = 22.0


def _fetch_error_is_retryable(err: str) -> bool:
    """Should a failed page fetch be retried on a FRESH Tor circuit?

    True for exit-node-dependent failures (timeout, 503, connection error,
    5xx) that a different exit can fix; False for definitive application /
    content errors (binary file, 401/403, SSRF refusal, 4xx) where a new
    circuit is wasted effort. Unknown errors default to retryable — the
    cost is at most one extra circuit attempt."""
    e = (err or "").lower()
    _definitive = (
        "binary file", "refusing to read", "(403)", "(401)",
        "will not help", "not allowed", "internal address",
        "loopback", "private address", "invalid url", "ssrf",
    )
    if any(m in e for m in _definitive):
        return False
    if "received status 4" in e:  # 4xx — client/application error
        return False
    return True


# A per-url distillation is one of many inside a research call and its
# fallback is free, so it gets a short leash.
#
# ⚠ 45s WAS SIZED FOR A WORLD WITHOUT CONCURRENCY. Measured on Nova, a useful
# distill (8,000 chars + 250 tokens) costs 18.9s alone, 31.3s at 2-way, 42.7s
# at 3-way and **53.6s at 4-way** — because decode is bandwidth-bound at a
# roughly fixed aggregate rate, so N concurrent requests each get about 1/N.
# A 45s leash therefore cannot fund a 4-way distill at all, and this
# deployment runs at 4-way routinely: `--worker-nodes` and `--critic-nodes`
# point at the SAME box, so a turn-gate verify shares these slots with the
# research wave. The result was not a timeout any more (the sizing prevents
# that) but a REFUSAL — 5 of 8 sources declining and falling back to raw
# HTML, which is the outcome this whole change exists to eliminate.
#
# This is a deliberate latency trade, and it was measured: at a 14s fetch the
# old constants produced 5,131 chars/143 tok per source in 93.8s and the new
# ones 11,576/233 in 130.9s — +40% wall for +126% page coverage and +63%
# output. In the RETRY regime the old set produced ZERO distillations at all.
# Typical measured completion is 105-131s. Tune with GHOST_WEB_SUMMARY_S.
_WEB_SUMMARY_TIMEOUT_S = env_float("GHOST_WEB_SUMMARY_S", 65.0)

# ⚠ THE WHOLE PHASE NEEDS A CLOCK, not just each URL. Every `_bounded()`
# coroutine starts from one `asyncio.gather`, so per-URL timers that begin
# there all run simultaneously and URLs 4-8 spend their budget queued on
# `Semaphore(3)` (see `_bounded`). The phase holds the deadline; each URL is
# issued its budget when it actually starts, clipped by what is left here.
# ⚠ GROW THE CONTAINER WITH ITS CONSUMER. This said "3 waves x
# PER_URL_TIMEOUT = 165s" — 165 is 3 x 55, the OLD per-URL value. The rebase
# grew the phase 25% (120->150) while growing the per-URL leash 45% (55->80),
# so RELATIVE coverage got worse: 8 urls through Semaphore(3) is 3 waves, and
# 3 x 80 = 240 against a 150s phase. Measured end-to-end in the retry regime
# (attempt 1 times out at 22s, the retry succeeds): urls 7 and 8 were declined
# with "14.0s available" and "0.1s available" and fell back to raw HTML, while
# 180 recovered 8/8 for +18.5s. 190 is what the structural pin demands: the
# last wave must still be STARTABLE, i.e. `(waves-1) * per_url +
# _MIN_URL_BUDGET_S`. This still does not cover the true worst case (240s) —
# it covers the measured one plus a startable tail. The pin fails if the ratio
# drifts again, which is what let a container grow 25% while its consumer grew
# 45%.
# ⚠ THIS IS A LATENCY TRADE, STATED. The OLD structure was bounded at ~55s
# only because all eight per-URL clocks ran from `gather` and expired together
# — the same fact that starved URLs 4-8 into posting doomed work. Giving each
# URL a real window means 3 waves, so the phase needs its own ceiling. 120s
# keeps the worst case near 2x the old bound rather than 3x (measured: the cap
# is genuinely enforced), while typical completion is ~60-90s now that a
# distill is sized to finish. Nothing above this bounds a tool call, and the
# model can batch several research calls at once, so it must stay modest.
_RESEARCH_PHASE_TIMEOUT_S = env_float("GHOST_RESEARCH_PHASE_S", 190.0)

# A URL that cannot be given at least this much is not started at all —
# reporting the source as uncovered beats burning a Tor circuit on a fetch
# that provably cannot finish.
#
# ⚠ DERIVED, NOT ASSERTED. This was a flat 12.0, which sits BELOW a single
# fetch attempt (22s): a URL admitted with 12s could not complete even the
# fetch, so the guard was authorising exactly what its comment says it
# prevents. The floor is now one full fetch attempt plus the margin
# `process_url` reserves — admit a URL only if it can at least come back with
# a page. Note this funds ONE of `_FETCH_ATTEMPTS`: a URL admitted at exactly
# this floor whose first attempt times out has no room for the retry. That is
# deliberate — covering both attempts (46.5s) would reject URLs whose first
# attempt would have succeeded, and a source that fetches but only yields raw
# text is still a source. (Distillation on top is a further question, answered per-URL by
# `plan_distill`; a source that fetches but only yields raw text is still a
# source, so it is not required here.)
_MIN_URL_BUDGET_S = _FETCH_ATTEMPT_TIMEOUT + 2.0

# ⚠ THE PLAN ASSUMES IT STARTS SOON — SO ENFORCE THAT. `chat_completion` was
# passed timeout == slot_wait == total_budget, so a call could sit in the
# per-node permit queue for nearly the whole budget and then POST a request
# sized for ALL of it: the plan's arithmetic was measured against time that
# had already been spent. (llama.cpp's `prompt_ms` starts at slot assignment,
# so the learned prefill rate structurally cannot see this cost.) The distill
# is now sized for `budget - _QUEUE_ALLOWANCE_S` and refuses to wait longer
# than that for a permit — `NodeSaturated` is explicitly NOT a node fault, and
# the caller degrades to raw text, which beats posting a doomed request.
#
# ⚠ AND IT MUST GROW WITH THE SERVICE TIME IT QUEUES BEHIND. At 8.0s, against
# a distill that now plans ~46s of work, this tolerates a queue depth of less
# than one caller — and the model batches research calls (three in one tool
# batch is on record, giving 9 concurrent worker requests at a 4-slot node).
# Measured with 3 concurrent deep_research calls: 8/0, 3/5, 0/8 distilled vs
# raw, i.e. 13 of 24 sources reaching `fact_check` as raw HTML on NodeSaturated
# alone. Raising `_WEB_SUMMARY_TIMEOUT_S` made this WORSE, not better: longer
# distills mean a longer queue behind an unchanged reserve.
#
# ⚠ 8.0 is also the exact value of `llm.py`'s own hidden floor
# (`_MIN_SLOT_WAIT + _MIN_HTTP_FLOOR` = 5.0 + 3.0), so any value BELOW 8 is
# silently ignored there while still being subtracted from the planner's
# budget — the plan would over-reserve. Keep this at or above 8.
_QUEUE_ALLOWANCE_S = 16.0

# The per-call distill fan-out, mirrored at module scope so the budget pins
# can check `waves x per-URL <= phase` without executing a research call.
_DISTILL_FANOUT_FOR_TESTS = 3

# Ceiling on ONE url: fetch (≤2 circuits) + LLM distillation.
_PER_URL_TIMEOUT_S = env_float("GHOST_PER_URL_S", 80.0)

# The raw page text kept when distillation is declined or fails. Deliberately
# NOT the distill limit: this text goes to the MAIN model unfiltered, so it is
# capped by what is worth reading, not by what the worker could have read.
_RAW_FALLBACK_CHARS = 10_000

async def _race_search_wave(query: str, tor_proxy: Optional[str], wave: int,
                            max_results: int = 20) -> List[Dict]:
    """Race ALL engines in parallel, each on its OWN Tor circuit; the first
    engine to return non-empty (junk-filtered) results wins the wave.

    This is the core Tor-reliability mechanism (measured 2026-07-08, see
    _RACE_ENGINES): per-(engine, circuit) success is ~10% and failures are
    driven by the exit IP, so the engines must NOT share a circuit. One
    single-engine ddgs call per engine, each with its own SOCKS-auth tag,
    turns one correlated ~10% attempt into len(_RACE_ENGINES) independent
    tickets (~47% per wave). Losers are cancelled as soon as a winner
    lands; a fully-blocked wave costs at most the ddgs timeout + grace.
    """
    from ddgs import DDGS

    def _run_engine(engine: str, proxy: Optional[str]) -> List[Dict]:
        _eng_timeout = _engine_timeout(engine)
        kwargs: Dict[str, Any] = {"timeout": _eng_timeout}
        if proxy:
            kwargs["proxy"] = proxy
        t_start = time.monotonic()
        try:
            with DDGS(**kwargs) as ddgs:
                return list(ddgs.text(query, max_results=max_results, region="wt-wt",
                                      safesearch="moderate", backend=engine))
        except StopIteration as e:
            # StopIteration cannot legally cross an asyncio Future boundary
            # (PEP 479 — it corrupts the event loop's future chaining), and
            # a generator-backed engine can surface one. Convert it.
            raise RuntimeError(f"engine {engine} produced no result stream") from e
        except Exception as e:
            # ddgs's internal future-wait starts its clock marginally BEFORE
            # the primp request, so a hung circuit expires the wait first and
            # surfaces as 'No results found.' — misbucketed as "empty" and
            # corrupting the timeout-vs-empty distinction the Tor runbook
            # diagnoses with (found 2026-07-15). Re-shape by elapsed time.
            elapsed = time.monotonic() - t_start
            if ("no results found" in str(e).lower()
                    and elapsed >= _eng_timeout - 0.5):
                raise RuntimeError(
                    f"engine {engine} timed out after {elapsed:.0f}s") from e
            raise

    t0 = time.monotonic()
    loop = asyncio.get_running_loop()
    tasks: Dict[Any, str] = {}
    for engine in _RACE_ENGINES:
        proxy = _proxy_for_attempt(tor_proxy, query, wave, salt=engine[:4])
        # Dedicated _RACE_POOL, NOT to_thread: uncancellable loser threads
        # must queue against other WAVES, not against the process-wide
        # default executor every other to_thread caller shares.
        task = asyncio.ensure_future(
            loop.run_in_executor(_RACE_POOL, _run_engine, engine, proxy))
        tasks[task] = engine

    # Several searches can race concurrently in one agent turn; the query
    # tag on every wave log line keeps their interleaved output readable.
    qtag = truncate_query(query, 28)
    deadline = _DDGS_TOR_TIMEOUT + _RACE_WAVE_GRACE
    pending = set(tasks)
    failures: List[Tuple[str, str]] = []
    timed_out = False
    try:
        while pending:
            remaining = deadline - (time.monotonic() - t0)
            if remaining <= 0:
                timed_out = True
                break
            done, pending = await asyncio.wait(
                pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                timed_out = True
                break
            for task in done:
                engine = tasks[task]
                try:
                    valid = _filter_junk(task.result())
                except Exception as e:  # noqa: BLE001 — a losing engine must never sink the wave
                    failures.append((engine, _brief_engine_error(e)))
                    continue
                if valid:
                    pretty_log("DDGS Search",
                               f"{engine} won wave {wave} in {time.monotonic() - t0:.1f}s "
                               f"({len(valid)} results) ‹{qtag}›", icon=Icons.TOOL_SEARCH)
                    return valid
                failures.append((engine, "empty"))
    finally:
        for task in tasks:
            if task.done():
                if not task.cancelled():
                    # A loser that co-completed in the winner's batch never
                    # had .result() called; sweep its exception so GC doesn't
                    # log 'Task exception was never retrieved' at ERROR onto
                    # the operator stream (found 2026-07-15).
                    try:
                        task.exception()
                    except Exception:  # noqa: BLE001
                        pass
            else:
                task.cancel()
    if failures or timed_out:
        # Operator stream gets ONE terse line — categories, not reprs:
        #   wave 0 ‹postgresql 20 features…›: no winner — 5 empty; mojeek conn-error
        # "empty" is the boring default so it's a bare count; engines are
        # named only where that carries signal. Unknown errors keep a short
        # snippet (never hide a failure shape we haven't seen before). Full
        # sanitized per-engine detail goes to logger.debug for forensics.
        cats: Dict[str, List[Tuple[str, str]]] = {}
        for engine, msg in failures:
            cats.setdefault(_failure_category(msg), []).append((engine, msg))
        parts: List[str] = []
        if "empty" in cats:
            parts.append(f"{len(cats['empty'])} empty")
        for cat in ("conn-error", "timeout"):
            if cat in cats:
                parts.append(f"{'+'.join(e for e, _ in cats[cat])} {cat}")
        if "error" in cats:
            engines = "+".join(e for e, _ in cats["error"])
            parts.append(f"{engines} error: {cats['error'][0][1][:48]}")
        if timed_out:
            parts.append(f"wave deadline {deadline}s")
        pretty_log("Search Error",
                   f"wave {wave} ‹{qtag}›: no winner — " + "; ".join(parts),
                   level="WARNING", icon=Icons.WARN)
        logging.getLogger(__name__).debug(
            "search wave %s ‹%s› detail: %s", wave, qtag,
            "; ".join(f"{e}: {m}" for e, m in failures))
    return []


def truncate_query(query: str, limit: int = 35) -> str:
    return (query[:limit] + "..") if len(query) > limit else query  # type: ignore


def _reformulate_query(query: str) -> List[str]:
    """Generate 2 reformulated search queries when the original fails.

    Strategy 1: Broaden by removing specific terms (numbers, versions, dates).
    Strategy 2: Hard-trim long queries; convert short ones to question form.
    """
    import re as _re
    reformulations = []

    # Strategy 1: Remove overly specific terms (versions, dates, numbers)
    broader = _re.sub(r'\b\d{4}\b', '', query)       # Remove years
    broader = _re.sub(r'\bv?\d+\.\d+\b', '', broader)  # Remove version numbers
    broader = _re.sub(r'\b\d+\b', '', broader)         # Remove other numbers
    broader = _re.sub(r'\s+', ' ', broader).strip()
    if broader and broader != query and len(broader) > 5:
        reformulations.append(broader)

    # Strategy 2: shorten, or convert to question form. A keyword-stuffed
    # query (>6 words) has near-zero organic hits ANYWHERE, so no circuit
    # can save it — and prepending "how to" keeps all the specificity and
    # fails identically (observed live 2026-07-08: the only total strike-out
    # of the session was an 11-word query whose reformulations both kept
    # every rare term). Hard-trim the broadened form to its first 5 words
    # instead; the question form only helps short queries.
    words = str(query).strip().split()
    if len(words) > 6:
        trimmed = " ".join((broader or query).split()[:5])
        if trimmed and len(trimmed) > 5 and trimmed not in reformulations:
            reformulations.append(trimmed)
    elif words and words[0].lower() not in {"how", "what", "why", "when", "where", "who", "which", "is", "can", "does"}:
        question = f"how to {query}"
        reformulations.append(question)
    elif len(words) > 3:
        # Already a question — try simplifying. For a 4-5 word question the
        # first-5-words "simplification" IS the original query; re-running it
        # labeled "[Reformulated]" burned a full wave on a lie (found
        # 2026-07-15) — fall through to the tutorial/guide fallback instead.
        simplified = " ".join(words[:5])
        if simplified != str(query).strip():
            reformulations.append(simplified)

    # Ensure we have exactly 2 reformulations
    if len(reformulations) == 0:
        reformulations = [f"{query} tutorial", f"{query} guide"]
    elif len(reformulations) == 1:
        reformulations.append(f"{query} example")

    return reformulations[:2]

def _clean_for_cpp(text: str) -> str:
    """Aggressively strip lone surrogates and raw control characters to prevent C++ JSON parser crashes.

    If the chunk is itself parseable JSON we leave it ALONE — the previous
    blanket `{`/`}`→`[`/`]` substitution was actively corrupting JSON
    payloads emitted by API responses, search snippets that happened to
    contain JSON examples, etc. Only non-JSON text gets the aggressive
    structural-char rewrite.
    """
    if not isinstance(text, str): return str(text)
    text = text.encode('utf-8', 'replace').decode('utf-8')
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\t\r")
    stripped = text.strip()
    if stripped and stripped[0] in "{[" and stripped[-1] in "}]":
        try:
            import json as _json
            _json.loads(stripped)
            # Valid JSON — return the (control-stripped) text unchanged so
            # downstream consumers can still parse it.
            return text
        except Exception:
            pass
    # Strip structural braces that confuse the Llama.cpp peg-native grammar parser
    return text.replace("{", "[").replace("}", "]").replace("<", "(").replace(">", ")")

async def tool_search_ddgs(query: str, tor_proxy: str):
    # Fail-closed (§4P): under --mandatory-tor a falsy proxy is replaced with
    # the loopback Tor default so the (always-public) search engines are never
    # queried cleartext — the socket guard can't backstop these curl_cffi
    # engine fetches. Outside mandatory-tor the proxy is unchanged.
    from ..utils.egress_guard import resolve_egress_proxy as _resolve_egress_proxy
    tor_proxy = _resolve_egress_proxy(tor_proxy)
    # Ensure proxy is in correct format for ddgs/httpx
    if tor_proxy and "socks5://" in tor_proxy and "socks5h://" not in tor_proxy:
        tor_proxy = tor_proxy.replace("socks5://", "socks5h://")

    # Strip Google-style operators the ddgs scraper backends choke on
    # (site:/quotes/boolean OR) BEFORE we spend a Tor round trip on a
    # query that would return nothing. This is the in-code backstop for
    # the LLM occasionally ignoring the "plain keywords only" guidance.
    query = _sanitize_query(query)

    # Log with TOR status and truncated query
    pretty_log("DDGS Search", query, icon=Icons.TOOL_SEARCH)

    # Cache hit: the model fires many near-identical queries per turn.
    _cache_key = _norm_cache_key(query)
    _cached = _cache_get(_cache_key)
    if _cached is not None:
        return _cached

    def format_search_results(results: List[Dict]) -> str:
        formatted = []
        for i, res in enumerate(results, 1):
            # `or` not `.get(k, default)`: a row can carry an explicit
            # key=None (the href=None shape _filter_junk guards against),
            # and `.get('href', …)` returns that None instead of the
            # fallback — dropping the source link / rendering the string
            # "None" as the body.
            title = _clean_for_cpp(res.get('title') or 'No Title')
            body = _clean_for_cpp(res.get('body') or res.get('content') or 'No content')
            link = res.get('href') or res.get('url') or '#'
            formatted.append(f"### {i}. {title}\n{body}\n[Source: {link}]")
        return "\n\n".join(formatted)

    if not importlib.util.find_spec("ddgs"):
        return "CRITICAL ERROR: 'ddgs' library is missing. Search is impossible."

    # NOTE: we deliberately do NOT call request_new_tor_identity() between
    # waves. A global NEWNYM re-circuits all of Tor (slow). Instead every
    # engine in a wave rides its OWN circuit and every wave rotates ALL of
    # them (_race_search_wave / _proxy_for_attempt) — search reachability
    # over Tor is exit-node-dependent, so fresh independent exits are what
    # actually beat a block, and it's far cheaper than NEWNYM.
    for wave in range(2):
        valid_results = await _race_search_wave(query, tor_proxy, wave)
        if valid_results:
            clean_output = format_search_results(valid_results[:8])
            _cache_put(_cache_key, clean_output)
            return clean_output
        if wave == 0:
            await asyncio.sleep(1)

    # --- QUERY REFORMULATION ---
    # Both waves with the original query failed (≈12 engine-circuit
    # tickets). Before giving up, try 2 reformulated queries: one broader,
    # one as a question. Each gets one wave of its own; the offset wave
    # index keeps its circuit tags from colliding with the primary waves'.
    reformulations = _reformulate_query(query)
    for ridx, reformulated in enumerate(reformulations):
        pretty_log("Search Retry", f"Reformulated: {truncate_query(reformulated)}", icon=Icons.TOOL_SEARCH)
        valid_results = await _race_search_wave(reformulated, tor_proxy, 10 + ridx)
        if valid_results:
            clean_output = format_search_results(valid_results[:8])
            result = f"[Reformulated query: '{reformulated}']\n\n{clean_output}"
            _cache_put(_cache_key, result)
            return result

    return (
        "ERROR: web search returned ZERO results across all engines and "
        "circuits, even after reformulation. Likely the query was too "
        "specific/long or every Tor exit was transiently blocked. DO NOT "
        "retry the same search. Instead: (a) drop to 2-4 PLAIN keywords (no "
        "quotes/operators/years), or (b) if you already have enough context, "
        "proceed with your own knowledge and state that web search was "
        "unavailable, rather than looping on more searches."
    )

def _record_project_findings(context, query: str, output: str) -> Optional[str]:
    """Main-loop research write-back (2026-09-03, §4EK): a live conversation
    searching about its active project leaves the results in the project's
    research/ dir, where coding leaves read them. Autonomous leaves (pinned
    project contexts) are excluded — their research is already saved as a
    brief. Never raises; returns the project-relative path when written."""
    try:
        if context is None or getattr(context, "is_pinned_project_context", False):
            return None
        pid = getattr(context, "current_project_id", None)
        store = getattr(context, "project_store", None)
        if not pid or store is None:
            return None
        from ..core.project_research import record_main_loop_findings
        rel = record_main_loop_findings(store, pid, query, output)
        if rel:
            logger.info("web_search: results written back to project %s (%s)", pid, rel)
        return rel
    except Exception:  # noqa: BLE001 — never let a record fail a search
        logger.debug("web_search: project findings write-back skipped", exc_info=True)
        return None


async def tool_search(query: Optional[str] = None, anonymous: bool = False, tor_proxy: str = None, context=None, **kwargs):
    if not query:
        return "SYSTEM ERROR: The 'query' parameter is MANDATORY. You must specify it."
    # Stylometric egress scrubbing: under anonymous mode, normalise the
    # outbound query into a neutral keyword form so the agent's prose
    # style (politeness, first-person framing, punctuation habits) — a
    # stable author fingerprint — doesn't leave the box alongside the
    # Tor-anonymised packets. Deterministic + keyword-preserving.
    if anonymous and query:
        try:
            from ..utils.stylometry import scrub_query
            query = scrub_query(query) or query
        except Exception:
            pass
        # Per-identity Tor circuit isolation: tag the SOCKS auth by a hash
        # of the (scrubbed) query so distinct searches ride distinct
        # circuits — a colluding set of exits can't link a sequence of
        # different searches into one session. Best-effort; falls back to
        # the shared proxy on any issue.
        if tor_proxy:
            try:
                import hashlib
                from ..utils.helpers import socks_url_with_identity
                _tag = hashlib.md5((query or "").encode("utf-8", "ignore")).hexdigest()[:12]
                tor_proxy = socks_url_with_identity(tor_proxy, _tag) or tor_proxy
            except Exception:
                pass
    # Tavily support removed. Always using DDGS.
    out = await tool_search_ddgs(query, tor_proxy)
    rel = _record_project_findings(context, query, out)
    if rel and isinstance(out, str):
        # Tell the model where the results now live, so it can point a
        # build at them instead of re-searching.
        out = out.rstrip() + f"\n\n(saved to {rel} in the active project — coding leaves read it)"
    return out

async def tool_deep_research(query: Optional[str] = None, anonymous: bool = False, tor_proxy: str = None, llm_client=None, model_name="default", max_context: int = 8192, workspace_model=None, **kwargs):
    if not query:
        return "SYSTEM ERROR: The 'query' parameter is MANDATORY. You must specify it."
    # Stylometric egress scrubbing (stronger tier): deep-research is
    # already LLM-heavy and latency-tolerant, so under anonymous mode the
    # query is re-authored into a neutral keyword form by the local model
    # (falls back to the deterministic lexical scrub on any failure).
    if anonymous and query:
        try:
            from ..utils.stylometry import neutralize_query
            query = await neutralize_query(query, llm_client=llm_client, model=model_name) or query
        except Exception:
            pass
    # Fail-closed (§4P): deep_research fans out to _race_search_wave DIRECTLY
    # (it does not go through tool_search_ddgs), so it needs the same backstop —
    # under --mandatory-tor a falsy proxy is replaced with the loopback Tor
    # default so the engine fetches + page fetches never leak cleartext.
    from ..utils.egress_guard import resolve_egress_proxy as _resolve_egress_proxy
    tor_proxy = _resolve_egress_proxy(tor_proxy)
    # Ensure proxy is in correct format for ddgs/httpx
    if tor_proxy and "socks5://" in tor_proxy and "socks5h://" not in tor_proxy:
        tor_proxy = tor_proxy.replace("socks5://", "socks5h://")

    # Strip Google-style operators the scraper backends choke on (runs
    # AFTER the optional anonymous re-authoring above so we sanitize
    # whatever query actually goes to the wire).
    query = _sanitize_query(query)

    pretty_log("Deep Research", query, icon=Icons.TOOL_DEEP)

    urls = []

    if not importlib.util.find_spec("ddgs"):
        return "CRITICAL ERROR: 'ddgs' library is missing. Search is impossible."

    # NEWNYM thrash removed; the engine race gives every engine in a wave
    # its own Tor circuit and rotates all of them between waves (see
    # _race_search_wave / _RACE_ENGINES for the measured why).
    for wave in range(2):
        valid_results = await _race_search_wave(query, tor_proxy, wave, max_results=15)
        if valid_results:
            # `.get(k, default)` returns None when the key EXISTS and is
            # None — and `_filter_junk` deliberately keeps such a row. The
            # None then reached process_url, raised, was swallowed by
            # `gather(return_exceptions=True)`, and filtered out: the
            # source vanished and the model was told nothing. Same class
            # this file already fixes at :267, :599 and :916.
            urls = [(r.get('href') or r.get('url') or '')
                    for r in valid_results[:8]]
            urls = [u for u in urls if u]
            break
        if wave == 0:
            await asyncio.sleep(1)
    else:
        return "CRITICAL ERROR: Deep Research search phase failed."

    if not urls: return "ERROR: No search results found. The internet might be blocking your request. Try a different query."

    # ⚠ WHAT `max_context` ACTUALLY BOUNDS. registry.py fills it with
    # `args.max_context` — the MAIN model's window (240,000 live) — and the
    # assembled report is what gets read back into that window, so it is a
    # ceiling on each source's SHARE of the report. It is NOT the worker's
    # limit: sizing the distill prompt by it is the defect this rewrite
    # removes (it never once bound below the 40k ceiling). The worker's own
    # limit comes from `plan_distill`, which reads that node's measured
    # throughput and its advertised context window.
    _report_share_chars = max(
        _MIN_DISTILL_CHARS,
        int(int(max_context) * CHARS_PER_TOKEN * 0.4) // max(1, len(urls)))

    # Page-fetch concurrency. Raised 2→3 (2026-07-08): with a distinct Tor
    # circuit per URL (below) the fetches no longer share one exit, so more
    # of them can run at once without correlated blocking. Kept modest so a
    # research turn doesn't open a dozen Tor circuits + worker LLM calls at
    # once on the RAM-tight box.
    #
    # ⚠ SCOPE, 2026-08-11: this bounds ONE call and nothing more. The clause
    # above about "worker LLM calls" was never true across calls — this object
    # is built per invocation, so three deep_research calls in one tool batch
    # meant 3 × 3 = NINE concurrent worker requests at a node with 4 slots
    # (req 0fb69c5f: the excess queued past the route timeout, every timeout
    # counted as a node fault, and the breaker ejected a healthy Nova for 60s).
    # The worker budget now lives in `LLMClient._node_slot`, keyed by node URL
    # so every caller and every role shares it. What remains here is what this
    # semaphore can actually govern: Tor circuits and page fetches for THIS
    # call.
    sem = asyncio.Semaphore(3)
    _DISTILL_FANOUT = 3      # == the semaphore width above
    # Read from the module global at CALL time, so the operator (and the
    # budget tests) can move it without editing this function.
    PER_URL_TIMEOUT = _PER_URL_TIMEOUT_S
    _phase_deadline = time.monotonic() + _RESEARCH_PHASE_TIMEOUT_S

    async def _fetch_with_timeout(url):
        # Resilient per-URL fetch: each URL rides its OWN Tor circuit, and a
        # circuit-retryable failure (timeout / 503 / connection error) is
        # retried on a FRESH exit — the same exit-node-dependence that made
        # search unreliable applies to fetches, and the same fix (a distinct
        # circuit per attempt, no global NEWNYM) recovers the lost sources.
        # Definitive errors (binary, 401/403, SSRF, 4xx) are NOT retried.
        last = f"Error: Fetch of {url} failed"
        # Fold the incoming per-QUERY identity (the SOCKS username the
        # anonymous path tagged onto tor_proxy) into the salt, so the final
        # circuit is scoped per-(query, url, attempt): distinct URLs and
        # retries get distinct exits (reliability), and the same URL across
        # different research sessions still can't be linked to one exit
        # (anonymity — the property the old verbatim-forward provided).
        try:
            from urllib.parse import urlparse as _urlparse
            _qid = (_urlparse(tor_proxy).username or "") if tor_proxy else ""
        except Exception:
            _qid = ""
        _fetch_salt = ("f" + _qid)[:16] if _qid else "fetch"
        for attempt in range(_FETCH_ATTEMPTS):
            proxy = _proxy_for_attempt(tor_proxy, url, attempt, salt=_fetch_salt)
            try:
                res = await asyncio.wait_for(
                    helper_fetch_url_content(url, proxy_override=proxy,
                                             renew_identity=False),
                    # Outer budget ABOVE the client's own 20s timeout so the
                    # client returns its (cleaner) error first — previously
                    # 15s < 20s guillotined slow-but-live Tor fetches.
                    timeout=_FETCH_ATTEMPT_TIMEOUT,
                )
                if isinstance(res, str) and not res.lstrip().startswith("Error"):
                    return res
                last = res if isinstance(res, str) else str(res)
                if not _fetch_error_is_retryable(last):
                    return last
            except asyncio.TimeoutError:
                last = f"Error: Fetch of {url} timed out after {_FETCH_ATTEMPT_TIMEOUT:.0f}s"
            except Exception as e:
                last = f"Error: {e}"
            if attempt + 1 < _FETCH_ATTEMPTS:
                await asyncio.sleep(0.5)
        return last

    async def process_url(url, budget_s):
        # ⚠ THE DISTILL BUDGET IS WHAT THE CLOCK LEAVES, NOT A CONSTANT.
        # R6 gave `darkweb_search.py` this treatment and left THIS, the
        # original, on a flat `_WEB_SUMMARY_TIMEOUT_S`. 45 is not the
        # remainder of `PER_URL_TIMEOUT`; it is merely smaller than it. The
        # outer `wait_for` covers up to two 22s fetch attempts AND the
        # distill, so a normal slow fetch plus a normal distill blows it and
        # the URL is LOST — measured at 55.00s with the worker node
        # COMPLETELY FREE (R7 lens A). Losing the URL is strictly worse than
        # the raw-text degradation below, and `fact_check` consumes the
        # result either way.
        #
        # `budget_s` is issued by `_bounded` AFTER the semaphore is acquired,
        # so it is time this URL actually HAS rather than time that was
        # already spent queueing behind three other URLs.
        _url_deadline = time.monotonic() + budget_s
        # Shorten URL for log
        url = str(url)
        short_url = (url[:35] + "..") if len(url) > 35 else url
        pretty_log("Parsing Data", url, icon=Icons.TOOL_FILE_R)
        text = await _fetch_with_timeout(url)

        # ⚠ A FAILED FETCH MUST NOT REACH THE DISTILLER. `text` is the
        # error STRING on failure ("Error: Fetch of <url> timed out
        # after 22s"), and handing that to a worker prompted with
        # "Extract ONLY the hard facts... If no relevant info is
        # found, state that" produces "The provided source text does
        # not contain any information relevant to the query."
        #
        # Measured with 8/8 fetches failing: the report contained the
        # word "Error" ZERO times and read as POSITIVE NEGATIVE
        # EVIDENCE — "I checked 8 named sources, none support this".
        # `fact_check` pipes exactly this into a TRUE/FALSE verifier,
        # so a Tor blackout became a confident, citation-backed
        # refutation of a possibly-true claim. The `llm_client=None`
        # fallback was the only honest path, and registry.py always
        # wires a client.
        if isinstance(text, str) and text.startswith("Error:"):
            pretty_log("Research Source Failed", f"{short_url} — {text[:90]}",
                       level="WARNING", icon=Icons.WARN)
            return f"### SOURCE: {url}\n{text}\n"

        # ⚠ CLEAN THE RAW-TEXT FALLBACK TOO. Unscrubbed surrogates / control
        # chars in a fetched page can crash the downstream C++ JSON/grammar
        # parser — the exact thing `_clean_for_cpp` exists to prevent — and
        # this path used raw `text`. Built once here because BOTH degradation
        # routes below (declined plan, failed call) return it.
        raw_preview = (_clean_for_cpp(text[:_RAW_FALLBACK_CHARS])
                       + "\n[...truncated...]\n")

        # ⚠ SIZE THE REQUEST TO WHAT THE NODE CAN FINISH — not to a constant,
        # and not to the MAIN model's context window. The arithmetic that
        # stood here derived `url_char_limit` from `max_context`, which
        # registry.py fills with `args.max_context` = 240,000 (the 35B's
        # window, NOT the worker's), so it pinned to its own 40,000-char
        # ceiling on every call and never constrained anything. On Nova that
        # is ~12,500 prompt tokens at ~300 tok/s prefill: **41s before the
        # first output token, against a 45s budget** — a request that cannot
        # finish even with the node completely idle and a slot free. Measured
        # live 2026-08-25 (req 08766aa1): 42.7s solo, 135/258/258s at this
        # function's own concurrency, and 1 success against 7 degradations in
        # the log window. `plan_distill` derives BOTH knobs (how much we send,
        # how much it may write) from throughput measured on the node itself.
        # See core/node_throughput.py.
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

        # ⚠ DECLINE, DO NOT POST. The predecessor computed the same shortfall
        # and then `raise`d a TimeoutError whose message said it was "taking
        # the raw-text path" — but the raise sat OUTSIDE the try below, so it
        # escaped `process_url`, was swallowed by `gather(return_exceptions=
        # True)`, and the source was dropped from the report entirely. The
        # message described the safe behaviour; the code did the lossy one.
        # A declined plan now degrades exactly like a failed call.
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

        # Sanitize text to remove surrogate unicode characters and raw control
        # characters that crash C++ JSON parsers.
        safe_text = _clean_for_cpp(text[:_chars])
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": f"Extract ONLY the hard facts explicitly relevant to this query: '{query}'. Ignore all other boilerplate. If no relevant info is found, state that.\n\nSource text:\n{safe_text}"}],
            "temperature": 0.0,
            "max_tokens": _tokens
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
                # ⚠ THE TOTAL IS WHAT MATTERS HERE. `slot_wait` alone
                # bounds only the queueing, and queue + POST then
                # exceeded the outer `wait_for` anyway — measured URL
                # LOSS at 55.00s on the shipped R5 code (R6 lens A).
                total_budget=_summary_budget,
                task_label="web summary")
            # Say the SIZING, not just "done". The 40k-on-45s mismatch
            # survived for as long as it did because the stream showed a
            # timeout and never the arithmetic that guaranteed one.
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
            preview = "[EDGE EXTRACTED FACTS]:\n" + (summary_data["choices"][0]["message"].get("content") or "").strip()
        except Exception as _sum_exc:
            # ⚠ SAY SO. This was a bare `except Exception:` with no
            # log at all, so every degradation here was invisible:
            # the source silently reverts to raw truncated page text,
            # which then feeds `fact_check` as if it were distilled
            # evidence. A whole research call could run on raw HTML
            # and read identically in the stream (R4 lens B).
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

    async def _bounded(url):
        # ⚠ THE PER-URL CLOCK STARTS WHEN THE URL STARTS, NOT AT `gather`.
        # This used to be `wait_for(process_url(url), PER_URL_TIMEOUT)` with
        # `process_url` setting its own deadline on entry — but every
        # `_bounded` coroutine is created by ONE `asyncio.gather`, so all
        # eight clocks started simultaneously while `Semaphore(3)` let only
        # three run. URLs 4-8 therefore arrived at the distiller with almost
        # nothing left: measured live (req 08766aa1) the 4th URL was handed a
        # 6s budget for a job needing 27s, posted it anyway, and ReadTimeout'd
        # in 6s — indistinguishable in the log from the three genuine 45s
        # timeouts ahead of it, and it counted as a node fault just the same.
        # The semaphore is acquired HERE, so the budget below is time this URL
        # actually has, clipped by what the whole phase has left.
        async with sem:
            budget = min(PER_URL_TIMEOUT, _phase_deadline - time.monotonic())
            if budget < _MIN_URL_BUDGET_S:
                # Report it as uncovered rather than burning a Tor circuit
                # and a worker slot on work that provably cannot finish. The
                # `_banner` below counts this, so the model is told.
                return (f"### SOURCE: {url}\nError: research phase deadline "
                        f"({_RESEARCH_PHASE_TIMEOUT_S:.0f}s) reached before "
                        f"this source could be fetched\n")
            try:
                return await asyncio.wait_for(process_url(url, budget),
                                              timeout=budget)
            except asyncio.TimeoutError:
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
    # A per-source failure is invisible inside a well-formed report, so
    # count them and say so up front. Without this, 7-of-8-failed and
    # 8-of-8-failed and "the topic genuinely has nothing" all render
    # identically — and the model has no way to weigh what follows.
    _failed = [c for c in valid_contents if "\nError:" in c]
    _lost = len(urls) - len(valid_contents)
    # ⚠ A DEGRADED SOURCE IS NEITHER FAILED NOR DISTILLED, and the banner
    # counted only the first two — so a run where every source fell back to
    # raw truncated HTML (the original defect's steady state: 1 distilled, 7
    # degraded) reported ZERO problems. The model was reading raw page dumps
    # presented as extracted evidence, with nothing saying so.
    _degraded = [c for c in valid_contents
                 if "\nError:" not in c
                 and "[EDGE EXTRACTED FACTS]" not in c] if llm_client else []
    _banner = ""
    if _failed or _lost or _degraded:
        _banner = (
            f"[⚠ SOURCE FAILURES: {len(_failed) + _lost} of {len(urls)} "
            f"sources could not be fetched (timeout, block, or unreachable). "
            f"The sections below cover only the "
            f"{len(valid_contents) - len(_failed)} that loaded. This is a "
            f"COVERAGE limit, not evidence of absence — do not conclude a "
            f"claim is unsupported because a failed source did not support "
            f"it.]\n\n") if (_failed or _lost) else ""
        if _degraded:
            _banner += (
                f"[⚠ {len(_degraded)} of {len(urls)} sources could not be "
                f"distilled in the time available and appear below as RAW, "
                f"TRUNCATED page text rather than extracted facts. Treat "
                f"those sections as unfiltered page dumps.]\n\n")
    full_report = _banner + "\n\n".join(valid_contents)
    # Workspace research dedup: record every URL we pulled so a later
    # research turn can ask "did I already see this?" via the workspace
    # tool. Non-fatal — must never break a successful research turn.
    if workspace_model is not None and getattr(workspace_model, "enabled", False):
        try:
            for u in urls:
                workspace_model.record_research_artifact(
                    url=u, source="deep_research", note=(query or "")[:120],
                )
        except Exception:  # noqa: BLE001
            pass
    return f"--- DEEP RESEARCH RESULT ---\n{full_report}\n\nSYSTEM INSTRUCTION: Analyze the text above."

async def tool_fact_check(query: Optional[str] = None, statement: Optional[str] = None, llm_client=None, tool_definitions=None, deep_research_callable: Optional[Callable] = None, model_name: str = "qwen-3.6-35b-a3", max_context: int = 8192, **kwargs: Any):
    """Verify a claim: run deep_research on it, then have the model judge the
    claim strictly against that evidence.

    2026-07-14 rewrite. The old flow spent a whole LLM round asking the model
    to emit a FORCED deep_research tool call (``tool_choice`` pinned it) just
    to rephrase the claim into a query — and when the model answered in plain
    content instead (the documented native-tools transport corruption family),
    the function fell off the end and returned ``None`` to the dispatcher. It
    also broke whenever ``tool_definitions`` didn't contain deep_research
    (subagent allowlists): forcing a function that isn't in ``tools`` is
    undefined server behaviour. The research call is now made DIRECTLY with
    the claim as the query: one less LLM round, no forced-tool fragility, no
    ``None`` path. ``tool_definitions`` is accepted for back-compat but no
    longer used.
    """
    query_text = str(query or statement or kwargs.get("query")
                     or kwargs.get("statement") or "").strip()
    if not query_text:
        return ("Error: fact_check needs the claim to verify — call it as "
                "fact_check(query='<the exact claim>').")
    pretty_log("Fact Check", query_text[:50] + "..", icon=Icons.TOOL_DEEP)  # type: ignore

    if deep_research_callable is None or llm_client is None:
        return ("Error: fact_check is unavailable in this context (research/"
                "LLM clients not wired). Use deep_research or web_search directly.")

    try:
        dr_result = str(await deep_research_callable(query_text))
    except Exception as exc:
        return (f"Error: fact_check research phase failed: {exc}. "
                f"Try web_search or deep_research directly.")

    # Bound the evidence spliced into the verify prompt the same way raw file
    # reads are bounded (chars ≈ tokens · 3.5) — deep_research can return up
    # to 8 sources × 10 KB, and `max_context` was previously accepted here but
    # never used, so nothing stopped the verify call from overflowing.
    evidence_cap = max(20_000, int(max_context * 3.5 * 0.30))
    if len(dr_result) > evidence_cap:
        dr_result = (dr_result[:evidence_cap]
                     + "\n…[evidence truncated to fit the verification context]")

    messages = [
        {"role": "system", "content": (
            "### ROLE: DEEP FORENSIC VERIFIER\n"
            "Judge the user's claim STRICTLY against the research evidence "
            "provided in the message. Open with a one-word verdict — TRUE, "
            "FALSE, PARTIALLY TRUE, or UNVERIFIABLE — then cite the key "
            "evidence for it and note any disagreement between sources.")},
        {"role": "user", "content": (
            f"CLAIM TO VERIFY:\n{query_text}\n\n[RESEARCH RESULTS]:\n{dr_result}\n\n"
            f"Verify the claim precisely using these results.")},
    ]
    try:
        final_res = await llm_client.chat_completion(
            {"model": model_name, "messages": messages, "temperature": 0.1})
        # `or ""`: some OpenAI-compatible servers send content: null — .get's
        # default doesn't cover an EXISTING null key (same coercion bug class
        # fixed in vision.py), which rendered "FACT CHECK COMPLETE:\nNone".
        verdict = (final_res["choices"][0]["message"].get("content") or "").strip()
    except Exception as exc:
        # The research itself succeeded — hand the evidence back instead of
        # dropping the whole result on a verify-call hiccup.
        # The research landed, the verification did not. `FACT CHECK
        # PARTIAL:` is not a head any predicate in this tree recognises —
        # `result_is_failure`, `result_is_rejection` and `coerce` all said
        # OK — so a half-answer was booked as a clean success. The status
        # says what the prose could not.
        from .outcome import ToolOutcome
        return ToolOutcome.partial(
            f"FACT CHECK PARTIAL: the verification call failed ({exc}); "
            f"judge the claim from the raw research results below.\n"
            f"[RESEARCH RESULTS]:\n{dr_result}",
            world_changed=False, reason_code="factcheck_verify_call_failed")
    if not verdict:
        from .outcome import ToolOutcome
        return ToolOutcome.partial(
            f"FACT CHECK PARTIAL: the verifier returned no text; judge the "
            f"claim from the raw research results below.\n"
            f"[RESEARCH RESULTS]:\n{dr_result}",
            world_changed=False, reason_code="factcheck_verifier_empty")
    return f"FACT CHECK COMPLETE:\n{verdict}"