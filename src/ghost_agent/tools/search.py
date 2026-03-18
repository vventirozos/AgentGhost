import asyncio
import importlib.util
import json
import os
import copy
from typing import List, Dict, Any, Callable, Optional
from ..utils.logging import Icons, pretty_log
from ..utils.helpers import helper_fetch_url_content

def truncate_query(query: str, limit: int = 35) -> str:
    return (query[:limit] + "..") if len(query) > limit else query  # type: ignore

def _clean_for_cpp(text: str) -> str:
    """Aggressively strip lone surrogates and raw control characters to prevent C++ JSON parser crashes."""
    if not isinstance(text, str): return str(text)
    text = text.encode('utf-8', 'replace').decode('utf-8')
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\t\r")
    # Strip structural braces that confuse the Llama.cpp peg-native grammar parser
    return text.replace("{", "[").replace("}", "]").replace("<", "(").replace(">", ")")

async def tool_search_ddgs(query: str, tor_proxy: str):
    # Ensure proxy is in correct format for ddgs/httpx
    if tor_proxy and "socks5://" in tor_proxy and "socks5h://" not in tor_proxy:
        tor_proxy = tor_proxy.replace("socks5://", "socks5h://")

    # Log with TOR status and truncated query
    pretty_log("DDGS Search", query, icon=Icons.TOOL_SEARCH)
    
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
    from ..utils.helpers import request_new_tor_identity
    for attempt in range(3):
        try:
            def run():
                kwargs: Dict[str, Any] = {"timeout": 20}
                if tor_proxy:
                    kwargs["proxy"] = tor_proxy
                with DDGS(**kwargs) as ddgs:
                    res = list(ddgs.text(query, max_results=10, region="wt-wt", safesearch="moderate"))
                    if res: return res
                return []
            raw_results = await asyncio.to_thread(run)
            
            junk = ["duckduckgo.com", "whatsapp.com", "twitter.com", "facebook.com", "tiktok.com", "instagram.com", "zhihu.com", "baike.baidu.com", "dict.cn", "pinterest.com", "aliexpress.com", "zhidao.baidu.com", "yahoo.com", "forbes.com", "bloomberg.com", "scmp.com", "quora.com", "medium.com", "msn.com", "cnn.com", "foxnews.com", "wsj.com", "csdn.net", "sohu.com", "sina.com"]
            valid_results = []
            for r in raw_results:
                url = r.get('href', r.get('url', '')).lower()
                if not url or url.startswith("/") or any(j in url for j in junk):
                    continue
                valid_results.append(r)

            if not valid_results:
                raise ValueError("DuckDuckGo returned empty or CAPTCHA garbage. Force IP cycling.")
                
            valid_results = valid_results[:5]  # type: ignore # Keep the top 5 relevant valid results
                
            clean_output = format_search_results(valid_results)
            return clean_output
        except Exception as e:
            pretty_log("Search Error", str(e), level="WARNING", icon=Icons.WARN)
            if attempt < 2:
                if tor_proxy:
                    await asyncio.to_thread(request_new_tor_identity)
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(1)

    return "ERROR: DuckDuckGo returned ZERO results. This usually means the query was too specific or the search engine is blocking the request (CAPTCHA/Tor). TRY A BROADER QUERY."

async def tool_search(query: str, anonymous: bool, tor_proxy: str, **kwargs):
    # Tavily support removed. Always using DDGS.
    return await tool_search_ddgs(query, tor_proxy)

async def tool_deep_research(query: str, anonymous: bool, tor_proxy: str, llm_client=None, model_name="default", max_context: int = 8192, **kwargs):
    # Ensure proxy is in correct format for ddgs/httpx
    if tor_proxy and "socks5://" in tor_proxy and "socks5h://" not in tor_proxy:
        tor_proxy = tor_proxy.replace("socks5://", "socks5h://")

    pretty_log("Deep Research", query, icon=Icons.TOOL_DEEP)
    
    urls = []
    
    if not importlib.util.find_spec("ddgs"):
        return "CRITICAL ERROR: 'ddgs' library is missing. Search is impossible."
        
    from ddgs import DDGS
    from ..utils.helpers import request_new_tor_identity
    
    for attempt in range(3):
        try:
            def run():
                kwargs: Dict[str, Any] = {"timeout": 20}
                if tor_proxy:
                    kwargs["proxy"] = tor_proxy
                with DDGS(**kwargs) as ddgs:
                    res = list(ddgs.text(query, max_results=15, region="wt-wt", safesearch="moderate"))
                    if res: return res
                return []
            results = await asyncio.to_thread(run)
            if not results:
                raise ValueError("DuckDuckGo returned empty list (Likely Tor Block). Force IP cycling.")
            
            junk = ["forums.att.com", "quora.com", "facebook.com", "twitter.com", "whatsapp.com", "duckduckgo.com", "zhihu.com", "scmp.com", "pinterest.com", "tiktok.com", "instagram.com", "yahoo.com", "msn.com", "forbes.com", "bloomberg.com", "csdn.net", "sohu.com", "sina.com", "baike.baidu.com", "dict.cn", "aliexpress.com", "zhidao.baidu.com", "medium.com", "cnn.com", "foxnews.com", "wsj.com"]
            
            valid_urls = []
            for r in results:
                url = r.get('href', r.get('url', '')).lower()
                if not url or url.startswith("/") or any(j in url for j in junk):
                    continue
                valid_urls.append(r.get('href', r.get('url', '')))
            if not valid_urls:
                raise ValueError("DuckDuckGo returned empty or CAPTCHA garbage. Force IP cycling.")
            
            urls = valid_urls[:4]  # type: ignore
            break # Success, we have our URLs
        except Exception as e:
            pretty_log("Search Error", str(e), level="WARNING", icon=Icons.WARN)
            if attempt < 2:
                if tor_proxy:
                    await asyncio.to_thread(request_new_tor_identity)
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(1)
            else:
                return f"CRITICAL ERROR: Deep Research search phase failed."

    if not urls: return "ERROR: No search results found. The internet might be blocking your request. Try a different query."

    sem = asyncio.Semaphore(2) 
    async def process_url(url):
        async with sem:
            # Shorten URL for log
            short_url = (url[:35] + "..") if len(url) > 35 else url
            pretty_log("Parsing Data", url, icon=Icons.TOOL_FILE_R)
            text = await helper_fetch_url_content(url)
            
            # 1. Hard cap the reading limit to ~15k chars (~4k tokens) to prevent context overflow on the worker node
            url_char_limit = 15000 
            fallback_limit = 3000
            
            # 2. Sanitize text to remove surrogate unicode characters and raw control characters that crash C++ JSON parsers
            safe_text = _clean_for_cpp(text[:url_char_limit])
            
            if llm_client:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": f"Extract ONLY the hard facts explicitly relevant to this query: '{query}'. Ignore all other boilerplate. If no relevant info is found, state that.\n\nSource text:\n{safe_text}"}],
                    "temperature": 0.0,
                    "max_tokens": 500
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

    tasks = [process_url(u) for u in urls]
    page_contents = await asyncio.gather(*tasks, return_exceptions=True)
    valid_contents = [c for c in page_contents if isinstance(c, str)]
    full_report = "\n\n".join(valid_contents)
    return f"--- DEEP RESEARCH RESULT ---\n{full_report}\n\nSYSTEM INSTRUCTION: Analyze the text above."

async def tool_fact_check(query: Optional[str] = None, statement: Optional[str] = None, llm_client=None, tool_definitions=None, deep_research_callable: Optional[Callable] = None, model_name: str = "qwen-3.5-9b", max_context: int = 8192, **kwargs: Any):
    query_text = query or statement or kwargs.get("query") or kwargs.get("statement", "")
    from ..core.agent import extract_json_from_text
    pretty_log("Fact Check", query_text[:50] + "..", icon=Icons.STOP)  # type: ignore
    
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