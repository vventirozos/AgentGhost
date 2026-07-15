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

# Small in-process TTL cache so the model's habit of firing many
# near-identical queries in one turn doesn't re-pay the full Tor round
# trip each time. Keyed on the normalized (sanitized, lower-cased) query.
# Only SUCCESSFUL results are cached — never error strings.
_SEARCH_CACHE_TTL = 300.0  # seconds
_SEARCH_CACHE_MAX = 64
_SEARCH_CACHE: Dict[str, Tuple[float, str]] = {}


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
    return q or query


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
        tbucket = int(time.monotonic() // 60)
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
# to_thread user in the process (found 2026-07-15). Sized for 4 concurrent
# waves; excess waves queue here instead of starving unrelated work.
from concurrent.futures import ThreadPoolExecutor as _TPE
_RACE_POOL = _TPE(max_workers=len(_RACE_ENGINES) * 4,
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
        kwargs: Dict[str, Any] = {"timeout": _DDGS_TOR_TIMEOUT}
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
                    and elapsed >= _DDGS_TOR_TIMEOUT - 0.5):
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
    words = query.strip().split()
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
        if simplified != query.strip():
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
    _cache_key = (query or "").strip().lower()
    _cached = _cache_get(_cache_key)
    if _cached is not None:
        return _cached

    def format_search_results(results: List[Dict]) -> str:
        formatted = []
        for i, res in enumerate(results, 1):
            title = _clean_for_cpp(res.get('title', 'No Title'))
            body = _clean_for_cpp(res.get('body', res.get('content', 'No content')))
            link = res.get('href', res.get('url', '#'))
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

async def tool_search(query: Optional[str] = None, anonymous: bool = False, tor_proxy: str = None, **kwargs):
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
    return await tool_search_ddgs(query, tor_proxy)

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
            urls = [r.get('href', r.get('url', '')) for r in valid_results[:8]]
            break
        if wave == 0:
            await asyncio.sleep(1)
    else:
        return "CRITICAL ERROR: Deep Research search phase failed."

    if not urls: return "ERROR: No search results found. The internet might be blocking your request. Try a different query."

    # Page-fetch concurrency. Raised 2→3 (2026-07-08): with a distinct Tor
    # circuit per URL (below) the fetches no longer share one exit, so more
    # of them can run at once without correlated blocking. Kept modest so a
    # research turn doesn't open a dozen Tor circuits + worker LLM calls at
    # once on the RAM-tight box.
    sem = asyncio.Semaphore(3)
    PER_URL_TIMEOUT = 55.0  # ceiling on fetch (≤2 circuits) + LLM distillation

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

    async def process_url(url):
        async with sem:
            # Shorten URL for log
            short_url = (url[:35] + "..") if len(url) > 35 else url
            pretty_log("Parsing Data", url, icon=Icons.TOOL_FILE_R)
            text = await _fetch_with_timeout(url)

            # 1. Size the per-source extract to the worker's context window
            # (~4 chars/token, reserving room for the prompt + max_tokens) so a
            # small-context worker can't overflow on the distill call — was a
            # hardcoded 40k that ignored max_context. Mirrors darkweb_research.
            reserve_tokens = 2048 + 512
            usable_tokens = max(1024, int(max_context) - reserve_tokens)
            url_char_limit = max(4000, min(40000, usable_tokens * 4))
            fallback_limit = min(10000, url_char_limit)

            # 2. Sanitize text to remove surrogate unicode characters and raw control characters that crash C++ JSON parsers
            safe_text = _clean_for_cpp(text[:url_char_limit])

            if llm_client:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": f"Extract ONLY the hard facts explicitly relevant to this query: '{query}'. Ignore all other boilerplate. If no relevant info is found, state that.\n\nSource text:\n{safe_text}"}],
                    "temperature": 0.0,
                    "max_tokens": 2048
                }
                try:
                    summary_data = await llm_client.chat_completion(payload, use_worker=True, task_label="web summary")
                    pretty_log("Worker Compute", f"Distilling facts from {short_url}", icon=Icons.TOOL_DEEP)
                    preview = "[EDGE EXTRACTED FACTS]:\n" + (summary_data["choices"][0]["message"].get("content") or "").strip()
                except Exception:
                    # Clean the raw-text fallback too: unscrubbed surrogates /
                    # control chars in a fetched page can crash the downstream
                    # C++ JSON/grammar parser (the exact thing _clean_for_cpp
                    # exists to prevent). safe_text is already cleaned; the
                    # fallback used raw `text`.
                    preview = _clean_for_cpp(text[:fallback_limit]) + "\n[...truncated...]\n"
            else:
                preview = _clean_for_cpp(text[:fallback_limit]) + "\n[...truncated...]\n"
            return f"### SOURCE: {url}\n{preview}\n"

    async def _bounded(url):
        try:
            return await asyncio.wait_for(process_url(url), timeout=PER_URL_TIMEOUT)
        except asyncio.TimeoutError:
            return f"### SOURCE: {url}\nError: per-URL timeout exceeded ({PER_URL_TIMEOUT}s)\n"

    tasks = [_bounded(u) for u in urls]
    page_contents = await asyncio.gather(*tasks, return_exceptions=True)
    valid_contents = [c for c in page_contents if isinstance(c, str)]
    full_report = "\n\n".join(valid_contents)
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
        return (f"FACT CHECK PARTIAL: the verification call failed ({exc}); "
                f"judge the claim from the raw research results below.\n"
                f"[RESEARCH RESULTS]:\n{dr_result}")
    if not verdict:
        return (f"FACT CHECK PARTIAL: the verifier returned no text; judge the "
                f"claim from the raw research results below.\n"
                f"[RESEARCH RESULTS]:\n{dr_result}")
    return f"FACT CHECK COMPLETE:\n{verdict}"

    return "SYSTEM ERROR: You failed to use the required `deep_research` tool. You must retry your action AND USE THE TOOL to fact check this claim."