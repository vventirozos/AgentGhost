"""§4UN-web (2026-11-13): web_search says when the engines REFUSED, keeps the
query's meaning on retry, and ranks and labels its results better.

Review of the internet tools against 30 days of logs: a refusal (HTTP 403/429,
a captcha page) read as "No results found"; the failure was then reworded into
a different question ("Python 3.13 release date" → "Python release date") and
the reworded results were judged against the reworded query; the fetch memo
refused the one retry the error asked for; Brave won no hedge; the Wikipedia
lookup was two fingerprinted calls (429 over Tor) and lost lower-cased Greek
names; a batch about something else won the race; numbers, `site:` and
"latest" did nothing; fact_check searched the claim's whole sentence, never
looked for a refutation and did not know the date.

All network is mocked. All owner data is fictional (Elena)."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core import fetch_memo as FM
from ghost_agent.tools import search as S
from ghost_agent.tools.outcome import ToolOutcome


# ── 1. blocked ≠ empty ───────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status, text=""):
        self.status_code = status
        self.text = text


class _Engine:
    """The shape `BaseSearchEngine.search` uses — nothing else."""
    name = "stubengine"
    search_method = "GET"
    search_url = "https://engine.example/search"

    def __init__(self, resp, rows=None):
        self.http_client = SimpleNamespace(request=lambda *a, **k: resp)
        self._rows = rows or []
        self.seen_payload = None

    def build_payload(self, **kw):
        self.seen_payload = kw
        return {"q": kw["query"]}

    def extract_results(self, html):
        return list(self._rows)

    def post_extract_results(self, rows):
        return rows


def test_a_non_200_answer_is_a_refusal_not_an_empty_page():
    eng = _Engine(_Resp(403, "<html>forbidden</html>"))
    with pytest.raises(RuntimeError, match=r"^blocked: HTTP 403 from stubengine"):
        S._guarded_engine_search(eng, "elena garden")
    assert S._failure_category("blocked: HTTP 429 from yahoo") == "blocked"


def test_a_challenge_page_with_no_results_is_a_refusal():
    eng = _Engine(_Resp(200, "<div class='anomaly-modal'>bots use DuckDuckGo too</div>"))
    with pytest.raises(RuntimeError, match="blocked: challenge page"):
        S._guarded_engine_search(eng, "elena garden")


def test_a_results_page_and_a_real_empty_page_are_unchanged():
    rows = [{"title": "Garden", "href": "https://g.example"}]
    assert S._guarded_engine_search(_Engine(_Resp(200, "<html>ok</html>"), rows), "garden", timelimit="w") == rows
    # a results page that MENTIONS a captcha is still results
    assert S._guarded_engine_search(_Engine(_Resp(200, "how to solve a captcha"), rows), "captcha") == rows
    # an empty page with no challenge marker is a real empty
    assert S._guarded_engine_search(_Engine(_Resp(200, "<html>no results</html>")), "zzz") == []
    assert S._guarded_engine_search(_Engine(_Resp(200, "")), "zzz") is None


def test_the_patch_installs_on_the_shape_it_was_written_for_and_only_then(monkeypatch):
    from ddgs.base import BaseSearchEngine
    original = BaseSearchEngine.search
    monkeypatch.setitem(S._ddgs_block_patch_state, "applied", None)
    monkeypatch.setattr(BaseSearchEngine, "search", original)
    import inspect
    real_getsource = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda obj: "def search(): return something_else()")
    with patch.object(S, "pretty_log") as log:
        assert S._patch_ddgs_block_detection() is False
    assert BaseSearchEngine.search is original
    assert any("NOT applied" in str(c.args[1]) for c in log.call_args_list)
    monkeypatch.setattr(inspect, "getsource", real_getsource)
    monkeypatch.setitem(S._ddgs_block_patch_state, "applied", None)
    if all(i in real_getsource(original) for i in S._DDGS_SEARCH_IDIOM):
        assert S._patch_ddgs_block_detection() is True
        assert BaseSearchEngine.search is S._guarded_engine_search


# ── driving tool_search_ddgs with scripted waves ────────────────────────────

def _drive(monkeypatch, script, query="postgresql 17 release notes"):
    """script(query, wave, stats, kw) -> rows. Returns (out, calls)."""
    calls = []

    async def _wave(q, proxy, wave, max_results=20, stats=None, **kw):
        calls.append((q, wave, dict(kw)))
        return list(script(q, wave, stats, kw) or [])
    monkeypatch.setattr(S, "_race_search_wave", _wave)
    monkeypatch.setattr(S, "_proxy_reachable", lambda *a, **k: True)
    monkeypatch.setattr(S, "wiki_supplement_wanted", lambda q: False)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    S._SEARCH_CACHE.clear()
    out = asyncio.run(S.tool_search_ddgs(query, "socks5://127.0.0.1:9050"))
    return out, calls


def _tally(stats, **cats):
    t = stats.setdefault("cats", {})
    for c, n in cats.items():
        t[c] = t.get(c, 0) + n


def test_a_blocked_search_reruns_the_same_query_and_is_not_reworded(monkeypatch):
    def script(q, wave, stats, kw):
        _tally(stats, blocked=3, empty=1)
        return []
    out, calls = _drive(monkeypatch, script, "what is the capital of australia")
    assert [(q, w) for q, w, _ in calls] == [("what is the capital of australia", 0),
                                              ("what is the capital of australia", 1),
                                              ("what is the capital of australia", 2)]
    assert "failed on TRANSPORT" in out and "9 engine answer(s) were refusals" in out
    assert "Retry the SAME query once" in out and "ZERO results" not in out


def test_the_blocked_rerun_can_win(monkeypatch):
    good = [{"title": "PostgreSQL 17 release notes", "body": "postgresql 17 release notes",
             "href": "https://www.postgresql.org/docs/release/17.0/"}]
    def script(q, wave, stats, kw):
        if wave < 2:
            _tally(stats, blocked=4)
            return []
        return good
    out, calls = _drive(monkeypatch, script)
    assert "postgresql.org/docs/release/17.0" in out and len(calls) == 3


def test_a_real_empty_is_reworded_once_and_judged_against_the_original(monkeypatch):
    judged = []
    real = S._prune_report
    monkeypatch.setattr(S, "_prune_report", lambda q, rows: (judged.append(q), real(q, rows))[1])
    hit = [{"title": "Canberra is the capital of Australia", "body": "capital australia canberra",
            "href": "https://atlas.example/au"}]
    def script(q, wave, stats, kw):
        if wave < 10:
            _tally(stats, empty=4)
            return []
        return hit
    out, calls = _drive(monkeypatch, script, "what is the capital of australia")
    assert [q for q, _w, _k in calls] == ["what is the capital of australia"] * 2 + ["capital australia"]
    assert out.startswith("[Reformulated query: 'capital australia']")
    assert judged == ["what is the capital of australia"]


def test_nothing_to_drop_means_no_reworded_search_and_an_honest_zero(monkeypatch):
    def script(q, wave, stats, kw):
        _tally(stats, empty=4)
        return []
    out, calls = _drive(monkeypatch, script, "Python 3.13 release date")
    assert len(calls) == 2
    assert "ZERO results" in out and "the engines answered" in out and "reworded" not in out


@pytest.mark.parametrize("query,expected", [
    ("what is the capital of australia", ["capital australia"]),
    ("Python 3.13 release date", []),
    ("how to fix docker compose networking issues on macos", ["fix docker compose networking issues macos"]),
    ("c++ vs .net performance?", []),
    ("when was PostgreSQL 16 released?", ["PostgreSQL 16 released"]),
])
def test_rewording_keeps_every_name_number_and_version(query, expected):
    assert S.reformulate_keeping_meaning(query) == expected


def test_deep_research_keeps_its_own_reformulations():
    # (the deep-research lane's function is untouched by this change)
    assert len(S._reformulate_query("PostgreSQL 16 release notes")) == 2


# ── 5. a low-relevance batch is a soft loss ─────────────────────────────────

def _ddgs_module(by_engine):
    mod = MagicMock(); cls = MagicMock(); mod.DDGS = cls
    inst = MagicMock(); cls.return_value.__enter__.return_value = inst
    inst.text.side_effect = by_engine
    return mod, inst


def test_a_low_relevance_batch_does_not_win_while_a_better_one_is_coming():
    def by_engine(q, **kw):
        if kw["backend"] == "duckduckgo":
            return [{"title": "Kubernetes pricing", "body": "cloud", "href": "https://k.example/a"}]
        if kw["backend"] == "yandex":
            time.sleep(0.3)
            return [{"title": "Kubernetes ingress nginx timeout settings", "body": "ingress nginx timeout",
                     "href": "https://docs.example/ingress"}]
        raise Exception("No results found.")
    mod, _ = _ddgs_module(by_engine)
    stats = {}
    with patch.dict("sys.modules", {"ddgs": mod}):
        rows = asyncio.run(S._race_search_wave("kubernetes ingress nginx timeout", None, 0, stats=stats,
                                               soft_low=True))
    assert rows[0]["href"] == "https://docs.example/ingress"


def test_a_low_relevance_batch_is_the_fallback_when_nothing_beats_it():
    def by_engine(q, **kw):
        if kw["backend"] == "duckduckgo":
            return [{"title": "Kubernetes pricing", "body": "cloud", "href": "https://k.example/a"}]
        raise Exception("No results found.")
    mod, _ = _ddgs_module(by_engine)
    stats = {}
    with patch.dict("sys.modules", {"ddgs": mod}), patch.object(S, "pretty_log") as log:
        rows = asyncio.run(S._race_search_wave("kubernetes ingress nginx timeout", None, 0, stats=stats,
                                               soft_low=True))
    assert [r["href"] for r in rows] == ["https://k.example/a"]
    assert stats["cats"].get("low-relevance") == 1
    lines = [str(c.args[1]) for c in log.call_args_list if c.args and c.args[0] == "Search Wave Empty"]
    # 12. every category names its engines
    assert lines and "duckduckgo low-relevance" in lines[0] and "yandex" in lines[0] and "empty" in lines[0]
    assert "using duckduckgo's low-relevance batch" in lines[0]


# ── 6. the hedge ─────────────────────────────────────────────────────────────

def test_the_hedge_is_yandex_and_yahoo():
    assert S._RACE_HEDGE_ENGINES == ("yandex", "yahoo")
    assert set(S._RACE_HEDGE_ENGINES) <= set(S._RACE_ENGINES)


# ── 7. numbers rank ──────────────────────────────────────────────────────────

def test_the_query_numbers_rank_first_and_match_whole():
    assert S.query_numbers("postgresql 17 release notes 2024") == ["17"]
    r = lambda t: {"title": t, "body": "postgresql release notes", "href": "https://x.example/" + t}
    rows = S.rank_on_topic("postgresql 17 release notes", [r("16 notes"), r("117 notes"), r("17.2 notes")])
    assert rows[0]["title"] == "17.2 notes"
    assert S.result_has_numbers({"title": "Python 3.13.1"}, ["3.13"]) is True
    assert S.result_has_numbers({"title": "Python 3.131"}, ["3.13"]) is False
    assert S.result_has_numbers({"title": "v1.19"}, ["19"]) is False


# ── 8. site: lists that host first ───────────────────────────────────────────

def test_a_site_host_is_listed_first_and_the_note_says_so(monkeypatch):
    rows = [{"title": "asyncio gather explained", "body": "asyncio gather", "href": "https://blog.example/a"},
            {"title": "asyncio gather docs", "body": "asyncio gather", "href": "https://docs.python.org/3/library/asyncio-task.html"}]
    out, calls = _drive(monkeypatch, lambda *a: rows, "site:docs.python.org asyncio gather")
    assert calls[0][0] == "asyncio gather"
    assert out.index("docs.python.org") < out.index("blog.example")
    note = S.site_operator_note("site:docs.python.org asyncio gather")
    assert "listed first" in note and "NOT limited to" in note and "plain keyword" in note
    assert S.site_operator_hosts("-site:pinterest.com cats") == []
    assert [r["href"] for r in S.site_first(rows, ["python.org"])][0].startswith("https://docs.python.org")


def test_the_advice_agrees_with_the_behaviour():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    desc = next(t["function"]["description"] for t in TOOL_DEFINITIONS if t["function"]["name"] == "web_search")
    assert "returns ZERO results" not in desc and "listed first" in desc


# ── 9. snippet-only rows ─────────────────────────────────────────────────────

def test_snippet_only_rows_are_listed_last_and_tagged(monkeypatch):
    good = [{"title": "Elena's garden club tomato prices", "body": "tomato prices garden", "href": "https://garden.example/p"}]
    snip = [{"title": "Tomato prices surge", "body": "tomato prices garden markets", "href": "https://www.forbes.com/x"},
            {"title": "Unrelated celebrity", "body": "gossip", "href": "https://www.msn.com/y"}]
    def script(q, w, stats, kw):
        stats["snippet_rows"] = list(snip)
        return good
    out, _ = _drive(monkeypatch, script, "tomato prices garden")
    assert out.index("garden.example") < out.index("forbes.com")
    assert S.SNIPPET_ONLY_TAG in out and "msn.com" not in out


def test_an_answer_found_only_on_snippet_sites_is_those_snippets(monkeypatch):
    snip = [{"title": "Tomato prices surge", "body": "tomato prices garden", "href": "https://www.bloomberg.com/x"}]
    def script(q, w, stats, kw):
        stats.setdefault("snippet_pool", []).extend(snip)
        _tally(stats, empty=4)
        return []
    out, _ = _drive(monkeypatch, script, "tomato prices garden")
    assert out.startswith("NOTE: the engines found this only on sites") and "bloomberg.com" in out
    assert not S._SEARCH_CACHE


def test_filter_junk_hands_back_the_snippet_rows():
    rows = [{"title": "a", "body": "b", "href": "https://www.forbes.com/a"},
            {"title": "c", "body": "d", "href": "https://ok.example/c"}]
    snippet_rows = []
    kept = S._filter_junk(rows, [], snippet_rows)
    assert [r["href"] for r in kept] == ["https://ok.example/c"]
    assert [r["href"] for r in snippet_rows] == ["https://www.forbes.com/a"]


# ── 10. freshness ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query,limit", [
    ("weather athens today", "w"), ("καιρός αθήνα σήμερα", "w"), ("news this month", "m"),
    ("latest python release", "y"), ("τελευταία νέα", "y"), ("python 3.13 release notes", None),
])
def test_freshness_words_set_a_time_limit(query, limit):
    assert S.freshness_timelimit(query) == limit


def test_only_the_first_wave_is_time_limited(monkeypatch):
    def script(q, w, st, kw):
        _tally(st, empty=4)
        return []
    _out, calls = _drive(monkeypatch, script, "latest python release")
    assert calls[0][2].get("timelimit") == "y" and all("timelimit" not in c[2] for c in calls[1:])


def test_the_engine_call_carries_the_time_limit():
    seen = []
    def by_engine(q, **kw):
        seen.append(kw.get("timelimit"))
        raise Exception("No results found.")
    mod, _ = _ddgs_module(by_engine)
    with patch.dict("sys.modules", {"ddgs": mod}):
        asyncio.run(S._race_search_wave("latest python release", None, 0, timelimit="y"))
        asyncio.run(S._race_search_wave("latest python release", None, 1))
    assert seen[:len(S._RACE_ENGINES)] == ["y"] * len(S._RACE_ENGINES)
    assert all(t is None for t in seen[len(S._RACE_ENGINES):])


# ── 4. the Wikipedia lookup ─────────────────────────────────────────────────

def _wiki_client(monkeypatch, status=200, pages=None):
    calls = []
    def get(url, params=None, **kw):
        calls.append((url, params, kw))
        return SimpleNamespace(status_code=status, json=lambda: {"query": {"pages": pages or {}}})
    fake = SimpleNamespace(get=get)
    monkeypatch.setitem(__import__("sys").modules, "curl_cffi.requests", fake)
    monkeypatch.setitem(__import__("sys").modules, "curl_cffi", SimpleNamespace(requests=fake))
    return calls


def test_the_lookup_is_one_plain_client_call(monkeypatch):
    pages = {"5": {"index": 1, "title": "Νίκος Καζαντζάκης", "extract": "Ο Νίκος Καζαντζάκης ήταν Έλληνας συγγραφέας."}}
    calls = _wiki_client(monkeypatch, pages=pages)
    out = S._wiki_lookup("πότε πέθανε ο καζαντζάκης", None, original="Πότε πέθανε ο Καζαντζάκης")
    assert [a["title"] for a in out] == ["Νίκος Καζαντζάκης — Βικιπαίδεια"]
    assert len(calls) == 1
    _url, params, kw = calls[0]
    assert params["generator"] == "search" and params["prop"] == "extracts"
    assert kw["headers"]["User-Agent"].startswith("curl/") and "impersonate" not in kw


def test_a_refused_lookup_is_one_info_line_and_nothing(monkeypatch):
    _wiki_client(monkeypatch, status=429)
    with patch.object(S, "pretty_log") as log:
        assert S._wiki_lookup("Νίκος Καζαντζάκης", None) == []
    assert any("HTTP 429" in str(c.args[1]) and c.kwargs.get("level") == "INFO" for c in log.call_args_list)


def test_the_greek_subject_comes_from_the_query_as_typed():
    art = {"title": "Νίκος Καζαντζάκης", "body": "Ο Νίκος Καζαντζάκης πέθανε το 1957."}
    assert S.wiki_article_on_topic("πότε πέθανε ο καζαντζάκης", art, "Πότε πέθανε ο Καζαντζάκης") is True
    # an article that has the verb but not the name: lower-cased, the first
    # distinctive word (the verb) was the "subject"; as typed, the name is
    other = {"title": "Ο Σεφέρης πέθανε", "body": "Ο Γιώργος Σεφέρης πέθανε το 1971."}
    assert S.wiki_article_on_topic("πότε πέθανε ο καζαντζάκης", other) is True
    assert S.wiki_article_on_topic("πότε πέθανε ο καζαντζάκης", other, "Πότε πέθανε ο Καζαντζάκης") is False


def test_tool_search_hands_the_typed_text_past_the_scrub(monkeypatch):
    seen = {}
    async def _ddgs(query, proxy):
        seen["original"] = S._typed_original(query)
        seen["sent"] = query
        return "### 1. x\nb\n[Source: https://x.example]"
    monkeypatch.setattr(S, "tool_search_ddgs", _ddgs)
    monkeypatch.setattr("ghost_agent.utils.stylometry.scrub_query", lambda q: q.lower())
    asyncio.run(S.tool_search("Πότε πέθανε ο Καζαντζάκης", anonymous=True, tor_proxy=None))
    assert seen == {"original": "Πότε πέθανε ο Καζαντζάκης", "sent": "πότε πέθανε ο καζαντζάκης"}
    assert S._QUERY_AS_TYPED.get() is None                              # reset after the call


# ── 3. the fetch memo allows the one retry ──────────────────────────────────

TRANSPORT = ("ERROR: web search failed on TRANSPORT, not on the query — 3 engine answer(s) were refusals. "
             "Retry the SAME query once after other work.")


def test_a_transport_failure_may_be_retried_once():
    memo = {}
    key = FM.fetch_key("web_search", {"query": "Elena garden club"})
    assert FM.remember_failure(memo, key, TRANSPORT) is True
    assert FM.repeat_result(memo, key) is None                          # the retry runs
    FM.remember_failure(memo, key, TRANSPORT)
    assert "ALREADY FAILED" in FM.repeat_result(memo, key)              # the second does not


def test_a_success_clears_the_retry_and_other_failures_are_unchanged():
    memo = {}
    key = FM.fetch_key("web_search", {"query": "Elena garden club"})
    FM.remember_failure(memo, key, TRANSPORT)
    FM.remember(memo, key, "### 1. Garden club\n…")
    assert FM.retry_key(key) not in memo and FM.failed_key(key) not in memo
    zero = "ERROR: web search returned ZERO results — the engines answered."
    k2 = FM.fetch_key("web_search", {"query": "zzzz"})
    FM.remember_failure(memo, k2, zero)
    assert "ALREADY FAILED" in FM.repeat_result(memo, k2)
    kb = FM.fetch_key("browser", {"operation": "navigate", "url": "https://x.example"})
    FM.remember_failure(memo, kb, "STATUS: BLOCKED failed on TRANSPORT")
    assert "ALREADY FAILED" in FM.repeat_result(memo, kb)               # a page is not a search


# ── 11. fact_check ───────────────────────────────────────────────────────────

REPORT = ("--- DEEP RESEARCH RESULT ---\n### SOURCE: https://garden.example/club\n[EDGE EXTRACTED FACTS]:\n"
          "- The Elena garden club was founded in 1998.\n\nSYSTEM INSTRUCTION: Analyze the text above.")


def _llm():
    c = MagicMock()
    c.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "TRUE — founded 1998."}}]})
    return c


def test_claim_keywords_keep_names_and_numbers():
    assert S.claim_search_keywords("Is it true that Tim Cook is still the CEO of Apple in 2026?") == \
        "Tim Cook still CEO Apple 2026"
    long = "The Eiffel Tower was built in 1889 for the World Fair held in Paris, France by Gustave Eiffel company"
    kw = S.claim_search_keywords(long).split()
    assert len(kw) == S.FACTCHECK_KEYWORDS_MAX and "1889" in kw and "Paris" in kw


def test_the_research_gets_keywords_and_the_question_when_it_takes_one(monkeypatch):
    got = {}
    async def research(q, source_titles=None, question=None):
        got.update(q=q, question=question)
        return ToolOutcome.ok(REPORT)
    research.accepts_source_titles = True
    research.accepts_question = True
    claim = "Is it true that the Elena garden club was founded in 1998?"
    asyncio.run(S.tool_fact_check(query=claim, llm_client=_llm(), deep_research_callable=research))
    assert got == {"q": "Elena garden club founded 1998", "question": claim}


def test_without_a_question_parameter_the_claim_is_the_query(monkeypatch):
    got = {}
    async def research(q, source_titles=None):
        got["q"] = q
        return ToolOutcome.ok(REPORT)
    research.accepts_source_titles = True
    claim = "The Elena garden club was founded in 1998"
    asyncio.run(S.tool_fact_check(query=claim, llm_client=_llm(), deep_research_callable=research))
    assert got["q"] == claim


def test_the_verifier_gets_the_date_and_the_counter_search(monkeypatch):
    waves = []
    async def _wave(q, proxy, wave, max_results=20, stats=None, **kw):
        waves.append((q, proxy))
        return [{"title": "Elena garden club 1998 founding claim debunked", "body": "garden club founded 2001, not 1998",
                 "href": "https://check.example/garden"}]
    monkeypatch.setattr(S, "_race_search_wave", _wave)
    async def research(q):
        return ToolOutcome.ok(REPORT)
    llm = _llm()
    out = asyncio.run(S.tool_fact_check(query="The Elena garden club was founded in 1998", llm_client=llm,
                                        deep_research_callable=research, tor_proxy="socks5://127.0.0.1:9050"))
    assert "FACT CHECK COMPLETE" in str(out)
    assert waves == [("Elena garden club founded 1998 false debunked", "socks5h://127.0.0.1:9050")]
    msgs = llm.chat_completion.call_args.args[0]["messages"]
    import datetime
    assert f"Today is {datetime.date.today().isoformat()}" in msgs[0]["content"]
    assert "[COUNTER-SEARCH" in msgs[1]["content"] and "check.example/garden" in msgs[1]["content"]


def test_no_proxy_no_counter_search(monkeypatch):
    called = []
    async def _wave(*a, **k):
        called.append(a)
        return []
    monkeypatch.setattr(S, "_race_search_wave", _wave)
    async def research(q):
        return ToolOutcome.ok(REPORT)
    llm = _llm()
    asyncio.run(S.tool_fact_check(query="The Elena garden club was founded in 1998", llm_client=llm,
                                  deep_research_callable=research))
    assert called == []
    assert "[COUNTER-SEARCH" not in llm.chat_completion.call_args.args[0]["messages"][1]["content"]


def test_a_failing_counter_search_never_sinks_the_check(monkeypatch):
    async def _wave(*a, **k):
        raise RuntimeError("tor down")
    monkeypatch.setattr(S, "_race_search_wave", _wave)
    async def research(q):
        return ToolOutcome.ok(REPORT)
    out = asyncio.run(S.tool_fact_check(query="The Elena garden club was founded in 1998", llm_client=_llm(),
                                        deep_research_callable=research, tor_proxy="socks5://127.0.0.1:9050"))
    assert "FACT CHECK COMPLETE" in str(out)


# ── review round 1 ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("query,expected", [
    ("python 3.13 no gil", []),                                   # nothing droppable but…
    ("Γιατί δεν λειτουργεί το wifi", ["δεν λειτουργεί wifi"]),
    ("why is postgres not using my index", ["postgres not using my index"]),
    ("best laptops under 1000 euros", []),
    ("what changed after the 2024 update", ["changed after 2024 update"]),
])
def test_r1_rewording_keeps_negation_and_comparison(query, expected):
    """MAJOR 2: "no gil" → "gil", "δεν λειτουργεί" → "λειτουργεί" — the opposite question."""
    got = S.reformulate_keeping_meaning(query)
    assert got == expected
    for w in ("no", "δεν", "not", "under", "after"):
        if w in query.split():
            assert all(w in g.split() for g in got)


def test_r1_claim_keywords_keep_negation_and_the_counter_label_is_neutral():
    """MAJOR 1: "do not cause" → "cause", and the counter-search then debunked the OPPOSITE."""
    assert S.claim_search_keywords("Vaccines do not cause autism") == "Vaccines not cause autism"
    assert "δεν" in S.claim_search_keywords("Ο Καζαντζάκης δεν πήρε ποτέ το Νόμπελ").split()
    many = "The new law does not apply to small shops in Athens Thessaloniki Patras Heraklion Larissa Volos"
    assert "not" in S.claim_search_keywords(many).split()
    sec = S.factcheck_counter_section("Vaccines not cause autism", [{"title": "t", "body": "b", "href": "https://x.example"}])
    assert "OR debunk its opposite" in sec and "may contradict the claim" not in sec


def test_r1_a_query_echoed_in_the_search_box_is_not_a_challenge():
    """MINOR 3: an empty results page that echoes "recaptcha v3 threshold" is a real empty."""
    page = ('<html><title>recaptcha v3 threshold at DuckDuckGo</title><input name="q" value="recaptcha v3 threshold">'
            '<a href="/html/?q=recaptcha+v3+threshold">x</a><a href="/?q=recaptcha%20v3%20threshold">y</a>'
            '<div>No results.</div></html>')
    assert S.engine_page_is_challenge(page, "recaptcha v3 threshold") is False
    assert S.engine_page_is_challenge('<form id="challenge-form">x</form>', "recaptcha") is True
    assert S.engine_page_is_challenge("<p>Please complete the captcha to continue</p>", "elena garden") is True
    long_empty = "<p>" + ("lorem ipsum " * 2000) + " access denied in the story</p>"
    assert S.engine_page_is_challenge(long_empty, "elena garden") is False
    eng = _Engine(_Resp(200, page))
    assert S._guarded_engine_search(eng, "recaptcha v3 threshold") == []


@pytest.mark.parametrize("query", [
    "current sensor arduino", "measure current multimeter", "current through a resistor", "now tv subscription",
    "τρέχω κάθε μέρα", "how to know now what", "ocean current map",
])
def test_r1_freshness_is_conservative(query):
    """MINOR 4: a noun "current", a brand "now", the verb τρέχω — no time limit."""
    assert S.freshness_timelimit(query) is None


@pytest.mark.parametrize("query,limit", [
    ("current prime minister of greece", "y"), ("τρέχουσα τιμή βενζίνης", "y"), ("bitcoin price right now", "w"),
    ("is the bridge open now?", "w"), ("weather as of now", "w"), ("currently best gpu", "y"),
])
def test_r1_freshness_still_fires(query, limit):
    assert S.freshness_timelimit(query) == limit


def test_r1_a_low_batch_waits_only_a_short_grace_and_no_hedge(monkeypatch):
    """MINOR 5: a low-relevance query went ~2 s → ~30 s (wave deadline + hedge + a second wave)."""
    monkeypatch.setattr(S, "_race_hedge_at", lambda: 0.6)
    monkeypatch.setattr(S, "LOW_RELEVANCE_GRACE_S", 0.4)
    launched = []
    def by_engine(q, **kw):
        launched.append(kw["backend"])
        if kw["backend"] == "duckduckgo":
            return [{"title": "Kubernetes pricing", "body": "cloud", "href": "https://k.example/a"}]
        time.sleep(3.0)
        raise Exception("No results found.")
    mod, _ = _ddgs_module(by_engine)
    t0 = time.monotonic()
    with patch.dict("sys.modules", {"ddgs": mod}):
        rows = asyncio.run(S._race_search_wave("kubernetes ingress nginx timeout", None, 0, soft_low=True))
    assert rows and rows[0]["href"] == "https://k.example/a"
    assert time.monotonic() - t0 < 1.5
    assert len(launched) == len(S._RACE_ENGINES)                    # no hedge for a low batch


def test_r1_without_soft_low_a_low_batch_wins_as_before():
    """MINOR 5: deep_research and the counter-search keep the pre-§4UN-web race."""
    def by_engine(q, **kw):
        if kw["backend"] == "duckduckgo":
            return [{"title": "Kubernetes pricing", "body": "cloud", "href": "https://k.example/a"}]
        time.sleep(1.5)
        return [{"title": "Kubernetes ingress nginx timeout", "body": "ingress nginx timeout", "href": "https://d.example"}]
    mod, _ = _ddgs_module(by_engine)
    t0 = time.monotonic()
    with patch.dict("sys.modules", {"ddgs": mod}):
        rows = asyncio.run(S._race_search_wave("kubernetes ingress nginx timeout", None, 0))
    assert rows[0]["href"] == "https://k.example/a" and time.monotonic() - t0 < 1.2


def test_r1_a_low_first_wave_is_the_answer_no_second_wave(monkeypatch):
    low = [{"title": "Kubernetes pricing", "body": "cloud", "href": "https://k.example/a"}]
    soft = []
    def script(q, w, st, kw):
        soft.append(S._SOFT_LOW_RACE.get())
        return low
    out, calls = _drive(monkeypatch, script, "kubernetes ingress nginx timeout")
    assert len(calls) == 1 and soft == [True] and S.LOW_RELEVANCE_NOTE in out
    assert S._SOFT_LOW_RACE.get() is False


def test_r1_the_wiki_subject_is_the_name_not_a_generic_word():
    """MINOR 6: "Νίκος Καζαντζάκης Βιογραφία" required "βιογραφια"."""
    art = {"title": "Νίκος Καζαντζάκης", "body": "Ο Νίκος Καζαντζάκης ήταν Έλληνας συγγραφέας."}
    assert S.wiki_article_on_topic("νίκος καζαντζάκης βιογραφία", art, "Νίκος Καζαντζάκης Βιογραφία") is True
    assert S.wiki_article_on_topic("Νίκος Καζαντζάκης Βιογραφία", art) is True


def test_r1_site_wildcards_and_the_cache_key(monkeypatch):
    """MINOR 7: "site:*.python.org", "site:.gov"; the site's results are a different answer."""
    assert S.site_operator_hosts("site:*.python.org asyncio gather") == ["python.org"]
    assert S.site_operator_hosts("site:.gov passport renewal") == ["gov"]
    rows = [{"title": "asyncio gather a", "body": "asyncio gather", "href": "https://blog.example/a"},
            {"title": "asyncio gather b", "body": "asyncio gather", "href": "https://docs.python.org/b"}]
    out1, _ = _drive(monkeypatch, lambda *a: rows, "asyncio gather")          # cached, blog first
    assert out1.index("blog.example") < out1.index("docs.python.org")
    out2 = asyncio.run(S.tool_search_ddgs("site:docs.python.org asyncio gather", "socks5://127.0.0.1:9050"))
    assert "[cached:" not in out2 and out2.index("docs.python.org") < out2.index("blog.example")


def test_r1_the_dispatch_hands_fact_check_the_proxy(monkeypatch):
    """MINOR 8: no request scope with a context → the counter-search was silently off."""
    from ghost_agent.tools import registry as R
    got = {}
    async def _fc(**kw):
        got.update(kw)
        return "FACT CHECK COMPLETE:\nTRUE"
    monkeypatch.setattr(R, "tool_fact_check", _fc)
    ctx = MagicMock()
    ctx.tor_proxy = "socks5://127.0.0.1:9050"
    ctx.args.anonymous = True
    asyncio.run(R.get_available_tools(ctx)["fact_check"](query="The Elena garden club was founded in 1998"))
    assert got["tor_proxy"] == "socks5://127.0.0.1:9050" and got["anonymous"] is True


def test_r1_numbers_match_their_spellings():
    """MINOR 9."""
    assert S.query_numbers("laptops under 1,000 euros") == ["1000"]
    assert S.result_has_numbers({"title": "Best laptops under 1,000"}, ["1000"]) is True
    assert S.result_has_numbers({"title": "best laptops under 1000"}, ["1000"]) is True
    assert S.result_has_numbers({"href": "https://x.example/python313-changes"}, ["3.13"], "python 3.13") is True
    assert S.result_has_numbers({"title": "1313 changes"}, ["3.13"], "python 3.13") is False
    assert S.query_numbers("ps5 sales 2019 vs 2022") == ["5", "2019", "2022"] or \
        set(S.query_numbers("ps5 sales 2019 vs 2022")) >= {"2019", "2022"}
    assert S.query_numbers("postgresql 17 release 2024") == ["17"]


# ── review round 2 ───────────────────────────────────────────────────────────

def test_r2_comparison_words_do_not_push_out_content():
    """MAJOR: every meaning word ranked 0 and counted toward the cap — "taller", "Elena", "mayor" lost."""
    kw = S.claim_search_keywords(
        "The Eiffel Tower in Paris is not taller than the Empire State Building in New York").split()
    assert "taller" in kw and "not" in kw and "than" in kw and "Eiffel" in kw
    kw2 = S.claim_search_keywords(
        "Elena Markou has never been the mayor of Athens since 2019 and lived there more than 10 years before").split()
    assert {"Elena", "Markou", "never", "mayor", "Athens", "2019"} <= set(kw2)
    content = [w for w in kw2 if not S.is_meaning_word(w)]
    # (r3: the word a free negation negates — "mayor" — rides outside the cap with it)
    assert len(content) <= S.FACTCHECK_KEYWORDS_MAX + S.FACTCHECK_FREE_NEGATIONS


def test_r2_a_comparison_word_goes_with_the_word_it_compares():
    many = "Papadakis Elena Markou Nikos Andreou Maria Lambrou Giorgos Petrou Kosmas Dimou sold more apples"
    kw = S.claim_search_keywords(many).split()
    assert "apples" not in kw and "more" not in kw                  # its partner was cut: so is it


def test_r2_the_query_echo_is_removed_as_whole_words_only():
    """MINOR 1: query "a" deleted every "a" ("captcha" hidden); per-word removal cut the marker phrase."""
    page = "<p>Please complete the captcha to continue</p>"
    assert S.engine_page_is_challenge(page, "a") is True
    js = "<p>Please enable javascript and cookies to continue</p>"
    assert S.engine_page_is_challenge(js, "enable javascript chrome") is True


def test_r2_a_glued_version_needs_three_digits_and_the_product_word():
    """MINOR 2: "1.3" → "13" matched "Model X13"."""
    assert S.result_has_numbers({"title": "Model X13 review"}, ["1.3"], "model 1.3") is False
    assert S.result_has_numbers({"href": "https://x.example/python313-changes"}, ["3.13"], "python 3.13") is True
    assert S.result_has_numbers({"href": "https://x.example/cp313-wheels"}, ["3.13"], "python 3.13") is False


def test_r2_a_low_time_limited_wave_lets_the_unlimited_wave_run(monkeypatch):
    """MINOR 3: "latest X" whose week/year-limited wave found only a low batch."""
    low = [{"title": "Kubernetes pricing", "body": "cloud", "href": "https://k.example/a"}]
    good = [{"title": "latest kubernetes ingress nginx release", "body": "latest kubernetes ingress nginx",
             "href": "https://docs.example/ingress"}]
    out, calls = _drive(monkeypatch, lambda q, w, st, kw: low if w == 0 else good, "latest kubernetes ingress nginx")
    assert len(calls) == 2 and calls[0][2].get("timelimit") == "y" and "timelimit" not in calls[1][2]
    assert "docs.example/ingress" in out
    # without a time limit a low first wave is still the answer (r1)
    out, calls = _drive(monkeypatch, lambda q, w, st, kw: low, "kubernetes ingress nginx timeout")
    assert len(calls) == 1


def test_r2_currently_is_a_year():
    """MINOR 4."""
    assert S.freshness_timelimit("currently best gpu") == "y"


def test_r2_a_yahoo_consent_footer_link_is_not_a_challenge():
    """MINOR 5."""
    footer = '<div>No results</div><footer><a href="https://consent.yahoo.com/v2/settings">Privacy</a></footer>'
    assert S.engine_page_is_challenge(footer, "elena garden") is False
    form = '<form method="post" action="https://consent.yahoo.com/v2/collectConsent?sessionId=x"></form>'
    assert S.engine_page_is_challenge(form, "elena garden") is True
    refresh = '<meta http-equiv="refresh" content="0;url=https://consent.yahoo.com/v2/collectConsent">'
    assert S.engine_page_is_challenge(refresh, "elena garden") is True


# ── review round 3 ───────────────────────────────────────────────────────────

def test_r3_pote_the_question_word_is_not_pote_the_negation():
    """MINOR 1: accents stripped made "Πότε" (when) the negation "ποτέ" (never)."""
    assert S.is_negation_word("ποτέ") is True and S.is_negation_word("Πότε") is False
    assert S.is_meaning_word("Πότε") is False and S.is_meaning_word("χωρις") is True   # unaccented typing still works
    assert "Πότε" not in S.reformulate_keeping_meaning("Πότε πέθανε ο Καζαντζάκης").__str__().split()


def test_r3_at_most_two_negations_ride_outside_the_cap():
    """MINOR 2: uncapped negations crowded out the subject."""
    claim = ("Elena did not win, did not run, never voted, no party, not a member, "
             "without funding in the 2019 elections")
    kw = S.claim_search_keywords(claim).split()
    assert {"Elena", "2019", "elections"} <= set(kw)
    negs = [w for w in kw if S.is_negation_word(w)]
    assert len(kw) - 2 * min(2, len(negs)) <= S.FACTCHECK_KEYWORDS_MAX
    for i, w in enumerate(kw):                                        # a kept negation keeps its target
        if S.is_negation_word(w):
            assert i + 1 < len(kw) and not S.is_negation_word(kw[i + 1])


def test_r3_a_comparison_partner_ranks_like_a_name():
    """MINOR 3: "taller" and "ψηλότερος" were lost in name-heavy claims."""
    kw = S.claim_search_keywords(
        "The Eiffel Tower in Paris is taller than the Empire State Building in New York City").split()
    assert "taller" in kw and "than" in kw
    kw = S.claim_search_keywords(
        "Ο Όλυμπος στη Θεσσαλία είναι ψηλότερος από τον Κίσσαβο τη Γκιώνα τον Παρνασσό τον Ταΰγετο και τη Βόρα").split()
    assert "ψηλότερος" in kw
