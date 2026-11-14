"""§4UN-dark (internet-tools review, 2026-11-13): the dark-web tools' accuracy and honesty fixes.

1. a worker summary keeps an onion address only when it is verbatim in the page, and every address must pass
   the v3 checksum (a one-letter garble is rejected everywhere, the reply guard included);
2. an engine that answered "no match" is not asked again, and is never charged a failure for it;
3. results are per URL (≤3 per host) with a snippet from the block that holds that one result;
4. a no-hits page's links (an ad, a sidebar) are not results; off-topic rows (ads) never lead — r1: nothing is
   remembered as "furniture" (it hid real hits for a day);
5. the empty-search error and the header say what each engine's query came to;
6. a page served from another location is not results;
7. the topic sort runs before the list is cut;
8. one breaker failure per engine per tool call;
9. URL clean-up, any wrapper key, visible-text no-hits, a per-attempt clock, the cache rules, the binary
   page message.
All network access is mocked."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from ghost_agent.core import link_grounding as LG
from ghost_agent.tools import darkweb_search as dw
from ghost_agent.tools.darkweb_search import (_parse_onion_results, keep_page_onions, onion_v3_valid,
                                              outcomes_line)
from tests.onion_v3 import v3_label

A, B, C, D_, E = (v3_label(x) for x in "abcde")
pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _clean():
    from ghost_agent.tools import search as _s
    _s._SEARCH_CACHE.clear()
    dw._FORM_TOKEN_CACHE.clear()
    dw._ENGINE_BREAKER.clear()
    yield
    _s._SEARCH_CACHE.clear()
    dw._ENGINE_BREAKER.clear()


def _garble(label: str) -> str:
    """One character changed — still 56 base32 characters."""
    return ("b" if label[10] != "b" else "c").join((label[:10], label[11:]))


# ───────────────────────────── 1. checksum + verbatim ─────────────────────────────

async def test_the_v3_checksum_accepts_real_addresses_and_rejects_a_one_letter_garble():
    real = "duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion"
    assert onion_v3_valid(real) and onion_v3_valid("http://" + real + "/x")
    assert onion_v3_valid(A + ".onion")
    assert not onion_v3_valid(_garble(A) + ".onion")
    assert not onion_v3_valid("a" * 56 + ".onion")
    assert not onion_v3_valid("abcdefghijklmnop.onion")                      # v2


async def test_a_summary_keeps_only_addresses_verbatim_in_the_page():
    page = f"Mirror list: http://{A}.onion/forum and the paste at {B}.onion"
    summary = (f"- Forum at http://{A}.onion/forum\n- Paste {B}.onion\n- Also http://{_garble(A)}.onion/x\n"
               f"- Market {C}.onion\n- Old {'q' * 10}x1.onion")
    out = keep_page_onions(summary, page)
    assert f"http://{A}.onion/forum" in out and f"{B}.onion" in out
    assert _garble(A) not in out and C not in out and "x1.onion" not in out
    assert out.count(LG._ONION_MARK) == 3


async def test_the_research_summary_cannot_put_an_invented_address_into_tool_output():
    """End to end: the worker 'summarises' a page and invents a near-copy of an address on it."""
    page_html = f"<html><body><p>Leak index: http://{A}.onion/leaks — updated weekly.</p></body></html>"
    rows = dw._Ranked([{"url": f"http://{B}.onion/thread/1", "title": "Acme breach thread",
                        "snippet": "acme breach", "engines": {"torch"}, "indexes": {"torch"}}], {})

    class _Plan:
        feasible, char_limit, max_tokens, predicted_s = True, 4000, 200, 1.0

        def describe(self):
            return "plan"

    class _Client:
        def plan_distill(self, *a, **k):
            return _Plan()

        async def chat_completion(self, payload, **kw):
            return {"choices": [{"message": {"content":
                    f"Acme breach leak index at http://{_garble(A)}.onion/leaks (also http://{A}.onion/leaks)"}}]}

        def note_distill_density(self, *a, **k):
            pass

    async def _raw(*a, **k):
        return rows, [], False, 3

    async def _page(url, proxy):
        return dw._strip_html(page_html)
    with patch.object(dw, "_darkweb_search_raw", _raw), patch.object(dw, "_fetch_onion_text", _page):
        out = await dw.tool_darkweb_research("acme breach", tor_proxy="socks5://127.0.0.1:9050",
                                             llm_client=_Client(), max_context=32000)
    text = str(out)
    assert _garble(A) not in text and f"http://{A}.onion/leaks" in text
    assert LG._ONION_MARK in text


async def test_the_reply_guard_names_a_checksum_failure():
    bad = _garble(A) + ".onion"
    issues = LG.onion_claim_issues(f"The leak site is {bad}.", f"tool said {A}.onion")   # garbled in the reply
    assert issues and "checksum" in issues[0]
    assert LG.onion_claim_issues(f"The leak site is {A}.onion.", f"tool said {A}.onion") == []


# ───────────────────────────── 2. no-hits ─────────────────────────────

async def test_an_engine_that_answered_no_match_is_asked_once_and_never_charged():
    calls = []

    async def _fetch(url, proxy, timeout, **kw):
        calls.append(url)
        return 200, "<html><body><p>No documents match your query.</p></body></html>"
    eng = {"name": "torch", "url": "http://x/search?q={q}", "index": "torch"}
    with patch.object(dw, "_fetch_raw_html", _fetch):
        for _ in range(4):
            res = await dw._query_engine(eng, "acme breach", "socks5h://127.0.0.1:9050")
            assert res == [] and res.outcome == dw.OUTCOME_NO_MATCH
    assert len(calls) == 4                                           # one fetch per search, no second circuit
    assert "torch" not in dw._ENGINE_BREAKER                         # never benched for answering


# ───────────────────────────── 3/4. parser ─────────────────────────────

async def test_a_thread_url_is_kept_beside_its_site_s_homepage():
    html = (f'<ul><li><a href="http://{A}.onion/">Acme forum</a></li>'
            f'<li><a href="http://{A}.onion/thread/123">Acme breach 2026 database leaked</a></li></ul>')
    urls = [r["url"] for r in _parse_onion_results(html)]
    assert urls == [f"http://{A}.onion/", f"http://{A}.onion/thread/123"]


async def test_a_snippet_never_carries_another_result_s_text():
    html = ("<div class='results'>"
            f"<a href='http://{A}.onion/x'>Unrelated shop</a> <a href='http://{B}.onion/y'>Acme breach dump</a>"
            " Acme breach 2026 database leaked</div>"
            f"<div><h4><a href='http://{C}.onion/z'>Cooking</a></h4><p>Recipes and soups</p></div>")
    rows = {r["url"]: r for r in _parse_onion_results(html)}
    assert rows[f"http://{A}.onion/x"]["snippet"] == ""              # its block holds two results
    assert rows[f"http://{C}.onion/z"]["snippet"] == "Recipes and soups"


async def test_ahmia_s_result_list_is_read_and_its_sidebar_is_not():
    html = (f"<aside><a href='http://{E}.onion/'>BUY CC DUMPS</a></aside><ol>"
            f"<li class='result'><h4><a href='/search/redirect?redirect_url=http%3A%2F%2F{A}.onion%2Fp'>Acme"
            f"</a></h4><p>acme breach paste</p></li></ol>")
    rows = _parse_onion_results(html, result_selector="li.result")
    assert [r["url"] for r in rows] == [f"http://{A}.onion/p"] and rows[0]["snippet"] == "acme breach paste"


async def test_a_no_hits_page_s_ad_is_not_a_result_and_ads_do_not_lead():
    ad = f"<div class='ad'><a href='http://{E}.onion/'>BUY CC DUMPS</a></div>"
    nohit = f"<html><body>{ad}<p>No documents match your query.</p></body></html>"
    hit = (f"<html><body>{ad}<div><a href='http://{D_}.onion/'>Daily news</a></div>"
           f"<div><a href='http://{A}.onion/t/9'>Acme breach thread</a></div></body></html>")
    pages = iter([nohit, hit])
    eng = {"name": "torgle", "url": "http://t/search.php?term={q}", "index": "torgle"}
    with patch.object(dw, "_load_engines", lambda: [eng]), \
            patch.object(dw, "_fetch_raw_html", lambda url, proxy, timeout, **kw: _ret(next(pages))):
        first, *_ = await dw._darkweb_search_raw("unrelated words", "socks5h://127.0.0.1:9050")
        ranked, *_ = await dw._darkweb_search_raw("acme breach", "socks5h://127.0.0.1:9050")
    assert list(first) == [] and first.outcomes == {"torgle": dw.OUTCOME_NO_MATCH}
    urls = [r["url"] for r in ranked]
    assert urls[0] == f"http://{A}.onion/t/9"                         # on-topic leads
    assert urls[-1] in (f"http://{E}.onion/", f"http://{D_}.onion/")  # the ad ranks with the off-topic rows


async def _ret(html):
    return 200, html


# ───────────────────────────── 5/6/7. outcomes, redirects, topic before the cut ─────────────────────────────

async def test_the_empty_error_says_what_each_engine_s_query_came_to():
    async def _fetch(url, proxy, timeout, **kw):
        if "torch" in url:
            return 200, "<p>No documents match your query.</p>"
        raise TimeoutError("slow")
    engs = [{"name": "torch", "url": "http://torch/q={q}", "index": "torch"},
            {"name": "torgle", "url": "http://torgle/q={q}", "index": "torgle"}]
    with patch.object(dw, "_load_engines", lambda: engs), patch.object(dw, "_fetch_raw_html", _fetch), \
            patch.object(dw.asyncio, "sleep", _no_sleep):
        out = await dw.tool_darkweb_search("acme breach", tor_proxy="socks5://127.0.0.1:9050")
    assert "answered, no match: torch" in out and "timed out: torgle" in out


async def _no_sleep(*a, **k):
    return None


async def test_a_page_served_from_another_location_is_not_results():
    async def _fetch(url, proxy, timeout, meta=None, **kw):
        if meta is not None:
            meta["final_url"] = "http://elsewhere.example/home"
        return 200, f"<a href='http://{A}.onion/'>Featured</a>"
    eng = {"name": "torgle", "url": "http://t/search.php?term={q}", "index": "torgle"}
    with patch.object(dw, "_fetch_raw_html", _fetch), patch.object(dw.asyncio, "sleep", _no_sleep):
        res = await dw._query_engine(eng, "acme", "socks5h://127.0.0.1:9050")
    assert res == [] and res.outcome == dw.OUTCOME_REDIRECTED
    assert dw._ENGINE_BREAKER.get("torgle", (0, 0))[0] == 1         # not a win


async def test_on_topic_rows_further_down_survive_the_cut():
    links = "".join(f"<div><a href='http://{v3_label('x' + str(i))}.onion/p'>Shop {i}</a></div>" for i in range(30))
    links += f"<div><a href='http://{A}.onion/t'>Acme breach thread</a></div>"
    eng = {"name": "torgle", "url": "http://t/q={q}", "index": "torgle"}
    with patch.object(dw, "_load_engines", lambda: [eng]), \
            patch.object(dw, "_fetch_raw_html", lambda url, proxy, timeout, **kw: _ret(links)):
        ranked, *_ = await dw._darkweb_search_raw("acme breach", "socks5h://127.0.0.1:9050", max_results=5)
    assert ranked[0]["url"] == f"http://{A}.onion/t"
    assert ranked.outcomes == {"torgle": dw.OUTCOME_RESULTS}


# ───────────────────────────── 8. breaker per call ─────────────────────────────

async def test_one_multi_phrasing_call_charges_an_engine_one_failure():
    async def _fetch(url, proxy, timeout, **kw):
        raise ConnectionError("tor patch down")
    eng = {"name": "torgle", "url": "http://t/q={q}", "index": "torgle"}
    with patch.object(dw, "_load_engines", lambda: [eng]), patch.object(dw, "_fetch_raw_html", _fetch), \
            patch.object(dw.asyncio, "sleep", _no_sleep):
        await dw.tool_darkweb_search("acme breach", extra_queries=["acme leak", "acme dump", "acme paste"],
                                     tor_proxy="socks5://127.0.0.1:9050")
    assert dw._ENGINE_BREAKER["torgle"][0] == 1
    assert not dw._breaker_should_skip("torgle")


# ───────────────────────────── 9. minors ─────────────────────────────

async def test_scanned_urls_lose_entities_and_trailing_punctuation_and_any_wrapper_is_unwrapped():
    html = f"see http://{A}.onion/paste?id=5&amp;amp;raw=1. and more"
    assert [r["url"] for r in _parse_onion_results(html)] == [f"http://{A}.onion/paste?id=5&raw=1"]
    wrapped = f"https://engine.example/out?ref=1&link=http%3A%2F%2F{B}.onion%2Fthread"
    assert dw._extract_onion(wrapped) == f"http://{B}.onion/thread"


async def test_no_results_text_in_a_script_is_not_a_no_hits_answer():
    body = "<html><script>var t={empty:'No results found'}</script><div id=app></div>" + "x" * 40000 + "</html>"
    kind, _ = dw._diagnose_empty_body(body, "http://t/q", None)
    assert kind == "parser"


async def test_a_stalled_first_attempt_leaves_time_for_the_second_circuit(monkeypatch):
    monkeypatch.setattr(dw, "_ONION_TIMEOUT", 0.2)
    monkeypatch.setattr(dw, "_ONION_ENGINE_DEADLINE", 3.0)
    n = {"i": 0}

    async def _fetch(url, proxy, timeout, **kw):
        n["i"] += 1
        if n["i"] == 1:
            await asyncio.sleep(10)                                  # a stalled transfer
        return 200, f"<a href='http://{A}.onion/x'>Acme</a>"
    eng = {"name": "torgle", "url": "http://t/q={q}", "index": "torgle"}
    with patch.object(dw, "_fetch_raw_html", _fetch):
        res = await dw._query_engine(eng, "acme", "socks5h://127.0.0.1:9050")
    assert [r["url"] for r in res] == [f"http://{A}.onion/x"] and n["i"] == 2


async def test_a_run_with_an_engine_in_cooldown_is_not_cached_and_the_key_holds_the_size():
    for _ in range(dw._ENGINE_BREAKER_FAILS):
        dw._breaker_record("torch", False)
    hits = {"n": 0}

    async def _fetch(url, proxy, timeout, **kw):
        hits["n"] += 1
        return 200, f"<div><a href='http://{A}.onion/x'>Acme breach</a></div>"
    engs = [{"name": "torch", "url": "http://torch/q={q}", "index": "torch"},
            {"name": "torgle", "url": "http://torgle/q={q}", "index": "torgle"}]
    with patch.object(dw, "_load_engines", lambda: engs), patch.object(dw, "_fetch_raw_html", _fetch):
        await dw.tool_darkweb_search("acme breach", tor_proxy="socks5://127.0.0.1:9050")
        await dw.tool_darkweb_search("acme breach", tor_proxy="socks5://127.0.0.1:9050")
        assert hits["n"] == 2                                         # not served from cache
        dw._ENGINE_BREAKER.clear()
        await dw.tool_darkweb_search("acme breach", tor_proxy="socks5://127.0.0.1:9050")
        n = hits["n"]
        await dw.tool_darkweb_search("acme breach", tor_proxy="socks5://127.0.0.1:9050")
        assert hits["n"] == n                                         # a full run is cached
        await dw.tool_darkweb_search("acme breach", tor_proxy="socks5://127.0.0.1:9050", max_results=3)
        assert hits["n"] > n                                          # another size is another key


async def test_a_binary_onion_page_says_what_it_is():
    async def _fetch(url, proxy, timeout, meta=None, **kw):
        if meta is not None:
            meta.update(content_type="application/pdf", content_length="1000")
        return 200, ""
    with patch.object(dw, "_fetch_raw_html", _fetch):
        out = await dw._fetch_onion_text(f"http://{A}.onion/doc.pdf", "socks5h://127.0.0.1:9050")
    assert "application/pdf" in out and "status 200" not in out


async def test_outcomes_line_orders_best_first():
    line = outcomes_line({"a": dw.OUTCOME_TIMED_OUT, "b": dw.OUTCOME_RESULTS, "c": dw.OUTCOME_NO_MATCH})
    assert line == "results: b; answered, no match: c; timed out: a"


# ───────────────────────────── review round 1 (§4UN-dark r1) ─────────────────────────────

async def test_r1_major1_an_href_query_string_is_not_entity_decoded_twice():
    """`&section=2` in an href BeautifulSoup already decoded became `§ion=2` (legacy no-semicolon
    entities: &sect &para &copy &not …) — the fetch then asked for another page."""
    html = f'<div><a href="http://{A}.onion/v?id=1&amp;section=2&amp;para=3&amp;copy=4">Acme</a></div>'
    assert [r["url"] for r in _parse_onion_results(html)] == [f"http://{A}.onion/v?id=1&section=2&para=3&copy=4"]
    swept = f"<pre>see http://{A}.onion/v?id=1&amp;amp;section=2</pre>"
    assert [r["url"] for r in _parse_onion_results(swept)] == [f"http://{A}.onion/v?id=1&section=2"]
    wrapped = f"https://e.example/out?u=http%3A%2F%2F{B}.onion%2Fv%3Fid%3D1%26not%3D2"
    assert dw._extract_onion(wrapped) == f"http://{B}.onion/v?id=1&not=2"


async def test_r1_major2_a_snippet_saying_unable_to_find_is_not_a_no_hits_answer():
    """A 3-result page whose one snippet reads "Unable to find your order?" was booked "answered, no
    match" — its real results thrown away."""
    rows = [(A, "Acme order tracker", "Unable to find your order? Track acme parcels here."),
            (B, "Acme breach paste", "acme order database"), (C, "Acme forum", "acme orders thread")]
    page = "<html><body>" + "".join(f"<div class='r'><h4><a href='http://{h}.onion/x'>{t}</a></h4><p>{s}</p></div>"
                                    for h, t, s in rows) + "</body></html>"
    eng = {"name": "torgle", "url": "http://t/q={q}", "index": "torgle"}
    with patch.object(dw, "_fetch_raw_html", lambda url, proxy, timeout, **kw: _ret(page)):
        res = await dw._query_engine(eng, "acme order", "socks5h://127.0.0.1:9050")
    assert res.outcome == dw.OUTCOME_RESULTS and len(res) == 3
    # the phrase OUTSIDE every result block still is the engine's answer
    nohit = f"<html><body><div class='ad'><a href='http://{E}.onion/'>BUY</a></div><p>No documents match.</p></body></html>"
    with patch.object(dw, "_fetch_raw_html", lambda url, proxy, timeout, **kw: _ret(nohit)):
        res = await dw._query_engine(eng, "acme order", "socks5h://127.0.0.1:9050")
    assert res == [] and res.outcome == dw.OUTCOME_NO_MATCH


async def test_r1_major3_a_link_once_seen_on_a_no_hits_page_is_still_a_result_later():
    """Links on a no-hits page were remembered as furniture and dropped from every later query, on-topic
    or not — and Ahmia's sidebar was learned without its result selector, then its real li.result lost."""
    nohit = (f"<html><body><aside><a href='http://{E}.onion/'>Acme breach archive</a></aside>"
             f"<p>No documents match your query.</p></body></html>")
    hit = (f"<html><body><div><a href='http://{E}.onion/'>Acme breach archive</a><p>acme breach dumps</p></div>"
           f"<div><a href='http://{D_}.onion/'>Daily news</a></div><div><a href='http://{A}.onion/t'>Acme</a></div>"
           f"<div><a href='http://{B}.onion/t'>B</a></div><div><a href='http://{C}.onion/t'>C</a></div>"
           f"<div><a href='http://{v3_label('f')}.onion/t'>F</a></div></body></html>")
    pages = iter([nohit, hit])
    eng = {"name": "torgle", "url": "http://t/search.php?term={q}", "index": "torgle"}
    with patch.object(dw, "_load_engines", lambda: [eng]), \
            patch.object(dw, "_fetch_raw_html", lambda url, proxy, timeout, **kw: _ret(next(pages))):
        await dw._darkweb_search_raw("unrelated words", "socks5h://127.0.0.1:9050")
        ranked, *_ = await dw._darkweb_search_raw("acme breach", "socks5h://127.0.0.1:9050")
    assert ranked[0]["url"] == f"http://{E}.onion/"
    ahmia_nohit = (f"<html><body><aside><a href='http://{A}.onion/'>Acme</a></aside>"
                   f"<p>No results found.</p></body></html>")
    ahmia_hit = (f"<html><body><aside><a href='http://{E}.onion/'>x</a></aside><ol><li class='result'>"
                 f"<h4><a href='http://{A}.onion/'>Acme breach</a></h4><p>acme breach</p></li></ol></body></html>")
    pages = iter([ahmia_nohit, ahmia_hit])
    ahmia = {"name": "ahmia", "url": "http://a/search/?q={q}", "index": "ahmia", "result_selector": "li.result"}
    with patch.object(dw, "_load_engines", lambda: [ahmia]), \
            patch.object(dw, "_fetch_raw_html", lambda url, proxy, timeout, **kw: _ret(next(pages))):
        await dw._darkweb_search_raw("unrelated words", "socks5h://127.0.0.1:9050")
        ranked, *_ = await dw._darkweb_search_raw("acme breach", "socks5h://127.0.0.1:9050")
    assert [r["url"] for r in ranked] == [f"http://{A}.onion/"]


async def test_r1_minor4_a_parser_drift_is_not_an_answer_and_all_no_match_replaces_the_blame():
    msg = dw._no_results_error([], 2, False, {"torch": dw.OUTCOME_NO_MATCH, "torgle": dw.OUTCOME_PARSER})
    assert "real negative" not in msg
    msg = dw._no_results_error([], 2, False, {"torch": dw.OUTCOME_NO_MATCH, "torgle": dw.OUTCOME_NO_MATCH})
    assert "real negative" in msg and "too specific" not in msg and "transiently down" not in msg


async def test_r1_minor5_an_on_topic_homepage_beats_off_topic_paths_in_the_host_cap():
    rows = [{"url": f"http://{A}.onion/p{i}", "on_topic": False} for i in range(3)] + \
        [{"url": f"http://{A}.onion/", "on_topic": True}]
    kept = dw._per_host_cap(list(enumerate(rows)))
    assert f"http://{A}.onion/" in [r["url"] for r in kept] and len(kept) == 3


async def test_r1_minor6_7_engine_links_neither_blank_snippets_nor_hide_a_wrapped_result():
    eng_host = v3_label("engine") + ".onion"
    html = (f"<div><h4><a href='http://{A}.onion/x'>Acme</a></h4><p>acme breach paste</p>"
            f"<a href='http://{eng_host}/cache?id=1'>cached</a></div>"
            f"<div><a href='http://{eng_host}/redirect.php?url=http%3A%2F%2F{B}.onion%2Fy'>B</a></div>")
    rows = {r["url"]: r for r in _parse_onion_results(html, {eng_host})}
    assert "acme breach paste" in rows[f"http://{A}.onion/x"]["snippet"]
    assert f"http://{B}.onion/y" in rows


async def test_r1_minor8_a_bad_checksum_address_the_user_typed_is_not_an_issue():
    bad = _garble(A) + ".onion"
    claim = f"{bad} fails its checksum, so it is mistyped."
    assert LG.onion_claim_issues(claim, f"user: is {bad} real?", user_text=f"is {bad} real?") == []
    assert "onion_v3_valid" in LG.__dict__ and dw.onion_v3_valid is LG.onion_v3_valid


async def test_r1_minor9_paren_port_and_subdomain_urls_survive_the_parse():
    html = (f"<pre>http://{A}.onion/wiki/Foo_(bar). http://{B}.onion:8080/p?x=1, "
            f"http://www.{C}.onion/forum)</pre>")
    urls = [r["url"] for r in _parse_onion_results(html)]
    assert urls == [f"http://{A}.onion/wiki/Foo_(bar)", f"http://{B}.onion:8080/p?x=1",
                    f"http://www.{C}.onion/forum"]


async def test_r1_minor9_a_run_where_an_engine_timed_out_is_not_cached():
    hits = {"n": 0}

    async def _fetch(url, proxy, timeout, **kw):
        hits["n"] += 1
        if "torch" in url:
            raise TimeoutError("slow")
        return 200, f"<div><a href='http://{A}.onion/x'>Acme breach</a></div>"
    engs = [{"name": "torch", "url": "http://torch/q={q}", "index": "torch"},
            {"name": "torgle", "url": "http://torgle/q={q}", "index": "torgle"}]
    with patch.object(dw, "_load_engines", lambda: engs), patch.object(dw, "_fetch_raw_html", _fetch), \
            patch.object(dw.asyncio, "sleep", _no_sleep):
        await dw.tool_darkweb_search("acme breach", tor_proxy="socks5://127.0.0.1:9050")
        n = hits["n"]
        await dw.tool_darkweb_search("acme breach", tor_proxy="socks5://127.0.0.1:9050")
    assert hits["n"] > n


async def test_r1_minor9_a_parser_page_on_attempt_one_outlives_a_timeout_on_attempt_two():
    n = {"i": 0}

    async def _fetch(url, proxy, timeout, **kw):
        n["i"] += 1
        if n["i"] == 1:
            return 200, "<html><body><div id=app>" + "x" * 5000 + "</div></body></html>"
        raise TimeoutError("slow")
    eng = {"name": "torgle", "url": "http://t/q={q}", "index": "torgle"}
    with patch.object(dw, "_fetch_raw_html", _fetch), patch.object(dw.asyncio, "sleep", _no_sleep):
        res = await dw._query_engine(eng, "acme", "socks5h://127.0.0.1:9050")
    assert res == [] and res.outcome == dw.OUTCOME_PARSER


# ───────────────────────────── review round 2 (§4UN-dark r2) ─────────────────────────────

_AD = f"<a href='http://{E}.onion/'>BUY CC DUMPS</a>"
_R2_NO_HIT_SHAPES = {
    "siblings under body": f"<html><body><p>No documents match your query.</p><div>{_AD}</div></body></html>",
    "wrap div": f"<html><body><div id=wrap><p>No documents match your query.</p><div class=ad>{_AD}</div></div></body></html>",
    "header ad": f"<html><body><header>{_AD}</header><div><p>No documents match your query.</p></div></body></html>",
    "div.main sponsored": f"<html><body><div class=main><p>Sponsored: {_AD}</p><p>No results.</p></div></body></html>",
}


@pytest.mark.parametrize("shape", sorted(_R2_NO_HIT_SHAPES))
async def test_r2_major_a_wrapper_around_banner_and_ad_is_still_a_no_match_page(shape):
    """r1 deleted each result's widest block before reading the page — a block that wrapped the banner
    too took "No documents match" with it, and the ad was booked as a result (a breaker win, cached)."""
    page = _R2_NO_HIT_SHAPES[shape]
    eng = {"name": "torgle", "url": "http://t/q={q}", "index": "torgle"}
    with patch.object(dw, "_fetch_raw_html", lambda url, proxy, timeout, **kw: _ret(page)):
        res = await dw._query_engine(eng, "acme breach", "socks5h://127.0.0.1:9050")
    assert res == [] and res.outcome == dw.OUTCOME_NO_MATCH, shape


async def test_r2_major_an_ahmia_result_item_holding_the_banner_is_a_no_match_page():
    page = (f"<html><body><ol><li class='result'><p>No results found.</p><div>{_AD} cheap</div></li></ol>"
            "</body></html>")
    eng = {"name": "ahmia", "url": "http://a/?q={q}", "index": "ahmia", "result_selector": "li.result"}
    with patch.object(dw, "_fetch_raw_html", lambda url, proxy, timeout, **kw: _ret(page)):
        res = await dw._query_engine(eng, "acme breach", "socks5h://127.0.0.1:9050")
    assert res == [] and res.outcome == dw.OUTCOME_NO_MATCH


async def test_r2_minor2_an_off_topic_ad_seen_twice_ranks_below_an_off_topic_page():
    ad = {"url": f"http://{E}.onion/", "on_topic": False, "indexes": {"ahmia", "torch"}, "host_indexes": 2}
    page = {"url": f"http://{A}.onion/thread/9", "on_topic": False, "indexes": {"torch"}, "host_indexes": 1}
    hit = {"url": f"http://{B}.onion/", "on_topic": True, "indexes": {"torch"}, "host_indexes": 1}
    order = sorted([(0, ad), (1, page), (2, hit)], key=lambda ir: dw._rank_key(ir[1], ir[0]))
    assert [r["url"] for _, r in order] == [hit["url"], page["url"], ad["url"]]


async def test_r2_minor3_only_the_user_s_words_exempt_a_bad_checksum_address():
    bad = _garble(A) + ".onion"
    claim = f"The mirror is {bad}."
    notes: list = []
    assert LG.onion_claim_issues(claim, f"page text: {bad}", notes=notes) == []
    assert notes and "checksum fails" in notes[0]                     # tool-seen: a note, not silence
    notes.clear()
    assert LG.onion_claim_issues(claim, f"user: {bad}", user_text=f"is {bad} real?", notes=notes) == []
    assert notes == []                                                # the user's own words: quoted
    assert "checksum" in LG.onion_claim_issues(claim, "nothing")[0]   # unseen: an issue


async def test_r2_minor4_www_and_bare_onion_are_one_site_and_a_port_is_another_url():
    assert dw._onion_host(f"http://www.{A}.onion/x") == dw._onion_host(f"http://{A}.onion/") == f"{A}.onion"
    assert dw._url_key(f"http://www.{A}.onion/x/") == dw._url_key(f"https://{A}.onion/x")
    assert dw._url_key(f"http://{A}.onion:8080/x") != dw._url_key(f"http://{A}.onion/x")
    html = "".join(f"<div><a href='http://{p}{A}.onion/{i}'>r</a></div>" for i, p in enumerate(["", "www."] * 3))
    assert len(_parse_onion_results(html)) == dw._MAX_PATHS_PER_HOST          # one site, one cap
    assert _parse_onion_results(f"<div><a href='http://www.{A}.onion/x'>self</a></div>", {f"{A}.onion"}) == []


async def test_r2_minor5_an_unreadable_engine_is_not_reached_in_the_header():
    async def _fetch(url, proxy, timeout, **kw):
        if "torch" in url:
            return 200, "<html><body><div id=app>" + "x" * 5000 + "</div></body></html>"
        return 200, f"<div><a href='http://{A}.onion/x'>Acme breach</a></div>"
    engs = [{"name": "torch", "url": "http://torch/q={q}", "index": "torch"},
            {"name": "torgle", "url": "http://torgle/q={q}", "index": "torgle"}]
    with patch.object(dw, "_load_engines", lambda: engs), patch.object(dw, "_fetch_raw_html", _fetch), \
            patch.object(dw.asyncio, "sleep", _no_sleep):
        out = await dw.tool_darkweb_search("acme breach", tor_proxy="socks5://127.0.0.1:9050")
    assert "engines reached: torgle]" in out and "answered, nothing parseable: torch" in out


async def test_r2_minor6_a_phrasing_with_an_unreachable_engine_is_not_cached():
    hits = {"n": 0}

    async def _fetch(url, proxy, timeout, **kw):
        hits["n"] += 1
        if "torch" in url and "leak" in url:
            raise ConnectionError("tor patch down")                  # only on one phrasing
        return 200, f"<div><a href='http://{A}.onion/x'>Acme breach</a></div>"
    engs = [{"name": "torch", "url": "http://torch/q={q}", "index": "torch"},
            {"name": "torgle", "url": "http://torgle/q={q}", "index": "torgle"}]
    with patch.object(dw, "_load_engines", lambda: engs), patch.object(dw, "_fetch_raw_html", _fetch), \
            patch.object(dw.asyncio, "sleep", _no_sleep):
        await dw.tool_darkweb_search("acme breach", extra_queries=["acme leak"], tor_proxy="socks5://127.0.0.1:9050")
        n = hits["n"]
        await dw.tool_darkweb_search("acme breach", extra_queries=["acme leak"], tor_proxy="socks5://127.0.0.1:9050")
    assert hits["n"] > n


@pytest.mark.parametrize("ad_text", ["results", "No", "o"])
async def test_r3_an_ad_whose_text_is_inside_the_banner_does_not_cut_the_banner(ad_text):
    """r2 cut each result's text out of the page as a substring, case-sensitively, everywhere: an ad reading
    "results", "No" or "o" cut "No results found." apart and the ad was booked as a result."""
    page = (f"<html><body><p>No results found.</p><div><a href='http://{E}.onion/'>{ad_text}</a></div>"
            "</body></html>")
    eng = {"name": "torgle", "url": "http://t/q={q}", "index": "torgle"}
    with patch.object(dw, "_fetch_raw_html", lambda url, proxy, timeout, **kw: _ret(page)):
        res = await dw._query_engine(eng, "acme breach", "socks5h://127.0.0.1:9050")
    assert res == [] and res.outcome == dw.OUTCOME_NO_MATCH, ad_text
