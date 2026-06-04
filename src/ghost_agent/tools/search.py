import asyncio
import importlib.util
import json
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

# Search backends queried over Tor. ddgs>=9 defaults to backend="auto",
# which fans EVERY query across all text engines. Two are actively
# harmful here and are dropped:
#   * wikipedia — its engine treats region="wt-wt" as language "wt" and
#     builds `https://wt.wikipedia.org/...`, a host that doesn't exist, so
#     it ALWAYS raises a ConnectError over Tor (observed in traces).
#   * yahoo     — never returns useful results in testing and is a known
#     hang-until-timeout offender over Tor; pure latency risk, no upside.
# The rest are kept. Empirically (measured directly over Tor) NO single
# engine is reliable — reachability is EXIT-NODE-dependent: `mojeek` and
# `yandex` return results on one circuit while `duckduckgo`/`brave`/
# `google` are CAPTCHA/empty, and on the next circuit it flips. Breadth +
# per-attempt circuit rotation (see _proxy_for_attempt) is what actually
# wins, so we keep a WIDE set rather than betting on one engine. The
# fast-failing engines ("No results found" in ~1.5s) are cheap lottery
# tickets; only genuine hangers are worth excluding. Order is advisory;
# ddgs re-sorts by engine.priority internally.
_TOR_BACKENDS = "mojeek,duckduckgo,yandex,brave,google"

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


def _proxy_for_attempt(base_proxy: Optional[str], query: str, attempt: int) -> Optional[str]:
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
        return socks_url_with_identity(bare, f"{qh}a{attempt}") or base_proxy
    except Exception:
        return base_proxy


def truncate_query(query: str, limit: int = 35) -> str:
    return (query[:limit] + "..") if len(query) > limit else query  # type: ignore


def _reformulate_query(query: str) -> List[str]:
    """Generate 2 reformulated search queries when the original fails.

    Strategy 1: Broaden by removing specific terms (numbers, versions, dates).
    Strategy 2: Convert to a question form.
    """
    import re as _re
    reformulations = []

    # Strategy 1: Remove overly specific terms (versions, dates, numbers)
    broader = _re.sub(r'\b\d{4}\b', '', query)       # Remove years
    broader = _re.sub(r'\bv?\d+\.\d+\b', '', broader)  # Remove version numbers
    broader = _re.sub(r'\b\d+\b', '', broader)         # Remove other numbers
    broader = broader.strip()
    if broader and broader != query and len(broader) > 5:
        reformulations.append(broader)

    # Strategy 2: Convert to question form
    words = query.strip().split()
    if words and words[0].lower() not in {"how", "what", "why", "when", "where", "who", "which", "is", "can", "does"}:
        question = f"how to {query}"
        reformulations.append(question)
    elif len(words) > 3:
        # Already a question — try simplifying
        simplified = " ".join(words[:5])
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

    from ddgs import DDGS
    # NOTE: we deliberately do NOT call request_new_tor_identity() between
    # attempts. A global NEWNYM re-circuits all of Tor (slow). Instead each
    # attempt rotates onto its OWN circuit via _proxy_for_attempt — search
    # reachability over Tor is exit-node-dependent, so a fresh exit per try
    # is what actually beats a block, and it's far cheaper than NEWNYM.
    for attempt in range(3):
        try:
            attempt_proxy = _proxy_for_attempt(tor_proxy, query, attempt)
            def run():
                kwargs: Dict[str, Any] = {"timeout": 8}
                if attempt_proxy:
                    kwargs["proxy"] = attempt_proxy
                with DDGS(**kwargs) as ddgs:
                    res = list(ddgs.text(query, max_results=20, region="wt-wt", safesearch="moderate", backend=_TOR_BACKENDS))
                    if res: return res
                return []
            raw_results = await asyncio.to_thread(run)

            valid_results = []
            for r in raw_results:
                url = r.get('href', r.get('url', '')).lower()
                if not url or url.startswith("/") or any(j in url for j in _JUNK_DOMAINS):
                    continue
                valid_results.append(r)

            if not valid_results:
                raise ValueError("DuckDuckGo returned empty or CAPTCHA garbage.")

            valid_results = valid_results[:8]  # type: ignore # Keep the top 8 relevant valid results

            clean_output = format_search_results(valid_results)
            _cache_put(_cache_key, clean_output)
            return clean_output
        except Exception as e:
            pretty_log("Search Error", str(e), level="WARNING", icon=Icons.WARN)
            if attempt < 2:
                await asyncio.sleep(1)

    # --- QUERY REFORMULATION ---
    # All 3 attempts with the original query failed. Before giving up,
    # try 2 reformulated queries: one broader, one as a question. Each
    # also rides its own fresh circuit (offset index so its tags don't
    # collide with the primary attempts').
    reformulations = _reformulate_query(query)
    for ridx, reformulated in enumerate(reformulations):
        pretty_log("Search Retry", f"Reformulated: {truncate_query(reformulated)}", icon=Icons.TOOL_SEARCH)
        try:
            attempt_proxy = _proxy_for_attempt(tor_proxy, reformulated, 10 + ridx)
            def run_reformulated():
                kwargs_r: Dict[str, Any] = {"timeout": 8}
                if attempt_proxy:
                    kwargs_r["proxy"] = attempt_proxy
                with DDGS(**kwargs_r) as ddgs:
                    res = list(ddgs.text(reformulated, max_results=20, region="wt-wt", safesearch="moderate", backend=_TOR_BACKENDS))
                    return res or []
            raw_results = await asyncio.to_thread(run_reformulated)
            valid_results = []
            for r in raw_results:
                url = r.get('href', r.get('url', '')).lower()
                if not url or url.startswith("/") or any(j in url for j in _JUNK_DOMAINS):
                    continue
                valid_results.append(r)
            if valid_results:
                valid_results = valid_results[:8]
                clean_output = format_search_results(valid_results)
                result = f"[Reformulated query: '{reformulated}']\n\n{clean_output}"
                _cache_put(_cache_key, result)
                return result
        except Exception:
            continue

    return "ERROR: DuckDuckGo returned ZERO results after query reformulation. This usually means the query was too specific or the search engine is blocking the request (CAPTCHA/Tor). TRY A COMPLETELY DIFFERENT APPROACH."

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

    from ddgs import DDGS
    # NEWNYM thrash removed; each attempt rotates onto its own Tor circuit
    # via _proxy_for_attempt instead (see tool_search_ddgs for the why).
    # Pinned backend set drops the broken `wikipedia` (wt-wt → bad host)
    # and useless `yahoo` engines.
    for attempt in range(3):
        try:
            attempt_proxy = _proxy_for_attempt(tor_proxy, query, attempt)
            def run():
                kwargs: Dict[str, Any] = {"timeout": 8}
                if attempt_proxy:
                    kwargs["proxy"] = attempt_proxy
                with DDGS(**kwargs) as ddgs:
                    res = list(ddgs.text(query, max_results=15, region="wt-wt", safesearch="moderate", backend=_TOR_BACKENDS))
                    if res: return res
                return []
            results = await asyncio.to_thread(run)
            if not results:
                raise ValueError("DuckDuckGo returned empty list (Likely Tor Block).")

            valid_urls = []
            for r in results:
                url = r.get('href', r.get('url', '')).lower()
                if not url or url.startswith("/") or any(j in url for j in _JUNK_DOMAINS):
                    continue
                valid_urls.append(r.get('href', r.get('url', '')))
            if not valid_urls:
                raise ValueError("DuckDuckGo returned empty or CAPTCHA garbage.")

            urls = valid_urls[:8]  # type: ignore
            break # Success, we have our URLs
        except Exception as e:
            pretty_log("Search Error", str(e), level="WARNING", icon=Icons.WARN)
            if attempt < 2:
                await asyncio.sleep(1)
            else:
                return f"CRITICAL ERROR: Deep Research search phase failed."

    if not urls: return "ERROR: No search results found. The internet might be blocking your request. Try a different query."

    sem = asyncio.Semaphore(2)
    PER_URL_TIMEOUT = 25.0  # hard ceiling on a single page fetch + summarisation

    async def _fetch_with_timeout(url):
        try:
            return await asyncio.wait_for(helper_fetch_url_content(url), timeout=15.0)
        except asyncio.TimeoutError:
            return f"Error: Fetch of {url} timed out after 15s"
        except Exception as e:
            return f"Error: {e}"

    async def process_url(url):
        async with sem:
            # Shorten URL for log
            short_url = (url[:35] + "..") if len(url) > 35 else url
            pretty_log("Parsing Data", url, icon=Icons.TOOL_FILE_R)
            text = await _fetch_with_timeout(url)
            
            # 1. Hard cap the reading limit to ~15k chars (~4k tokens) to prevent context overflow on the worker node
            url_char_limit = 40000 
            fallback_limit = 10000
            
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
                    summary_data = await llm_client.chat_completion(payload, use_worker=True)
                    pretty_log("Worker Compute", f"Distilling facts from {short_url}", icon=Icons.TOOL_DEEP)
                    preview = "[EDGE EXTRACTED FACTS]:\n" + summary_data["choices"][0]["message"].get("content", "").strip()
                except Exception:
                    preview = text[:fallback_limit] + "\n[...truncated...]\n"
            else:
                preview = text[:fallback_limit] + "\n[...truncated...]\n"
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
    query_text = query or statement or kwargs.get("query") or kwargs.get("statement", "")
    from ..core.agent import extract_json_from_text
    pretty_log("Fact Check", query_text[:50] + "..", icon=Icons.TOOL_DEEP)  # type: ignore
    
    allowed_names = ["deep_research"]
    restricted_tools = [t for t in tool_definitions if t["function"]["name"] in allowed_names]
    
    messages = [
        {"role": "system", "content": "### ROLE: DEEP FORENSIC VERIFIER\nVerify this claim with deep_research."},
        {"role": "user", "content": query_text}
    ]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,
        "tools": restricted_tools,
        "tool_choice": {"type": "function", "function": {"name": "deep_research"}}
    }
    
    plan_response = await llm_client.chat_completion(payload)
    ai_msg = plan_response["choices"][0]["message"]
    
    if ai_msg.get("tool_calls"):
        t_call = ai_msg["tool_calls"][0]["function"]
        t_args_raw = t_call.get("arguments", {})
        
        t_args_dict = {}
        if isinstance(t_args_raw, str):
            try: t_args_dict = json.loads(t_args_raw)
            except: pass
        elif isinstance(t_args_raw, dict):
            t_args_dict = t_args_raw
            
        q = t_args_dict.get("query", query_text)
        dr_result = await deep_research_callable(q)
        
        # Bypass inserting formal API `tool_calls` back into context.
        # This prevents Llama-Server's Chat Template vs API Schema paradox from crashing
        # when it requires JSON strings but Jinja requires Python Dictionaries.
        messages.append({
            "role": "user", 
            "content": f"The `deep_research` tool successfully executed for '{q}'.\n\n[RESEARCH RESULTS]:\n{dr_result}\n\nVerify the original claim precisely using these results."
        })
        
        verify_payload = {"model": model_name, "messages": messages, "temperature": 0.1}
        final_res = await llm_client.chat_completion(verify_payload)
        return f"FACT CHECK COMPLETE:\n{final_res['choices'][0]['message'].get('content', '')}"
    
    return "SYSTEM ERROR: You failed to use the required `deep_research` tool. You must retry your action AND USE THE TOOL to fact check this claim."