"""§4JO — the encyclopedic supplement for short Greek-script queries.

Survey over Tor (17 recorded Greek queries × 2 circuits, region gr-el):
yandex 30/34 (2.3 s median) — the race's only real Greek ticket, and it
treats "23.125.000" as a model number; bing 25/34 with the same pages at
17.9 s median (too slow for the race — §4HR); google 0/34, mojeek 0/34.
el.wikipedia's full-text search over Tor answered the proper-noun asks
("Δημήτρης Κουφοντίνας", "Ελεγκτικό Συνέδριο", "Οδός Μιχαήλ Βόδα", the
Tempi paraphrase) in ~1 s with the article intro and, correctly, nothing
on the long figure-laden queries.

The supplement: for a short Greek-script query one lookup runs in
parallel with the first web wave on its own circuit; an article whose
title/intro carries the query's SUBJECT (its leading content word — the
§4IL "any distinctive token" rule reads every Greek word as distinctive,
so "Πλωτό νοσοκομείο" would pass for "Τζάνειο Νοσοκομείο") is placed
first in the batch; when every web ticket fails it is the batch; a late
or failed lookup never delays or sinks the search.

World where each pin fails: the gate opens for Latin or long queries,
the subject rule reverts to any-token, the lookup drops the extract or
the URL, the merge stops putting the article first or duplicates it, a
failed web wave stops falling back to the article, or a lookup
exception reaches the caller.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.ghost_agent.tools import search as S
from src.ghost_agent.tools.search import (_SEARCH_CACHE, merge_wiki_first, tool_search_ddgs,
                                          wiki_article_on_topic, wiki_supplement_wanted)


# ── the gate ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("query, wanted", [
    ("Δημήτρης Κουφοντίνας", True),
    ("Τέμπη σιδηροδρομικό δυστύχημα 2023", True),
    ("Οδός Μιχαήλ Βόδα", True),
    ("Ελεγκτικό Συνέδριο ζημιά Δημοσίου 23100000 δραχμές πλαστογραφία κακούργημα", False),   # 7 content words
    ("who is Dimitris Koufontinas", False),                                                  # Latin script
    ("", False),
])
def test_only_short_greek_queries_get_a_lookup(query, wanted):
    assert wiki_supplement_wanted(query) is wanted


# ── the subject rule ─────────────────────────────────────────────────

def test_the_article_must_carry_the_subject_not_any_word():
    tz = "Τζάνειο Νοσοκομείο"
    assert wiki_article_on_topic(tz, {"title": "Πλωτό νοσοκομείο", "body": "Ένα πλωτό νοσοκομείο είναι…"}) is False
    assert wiki_article_on_topic(tz, {"title": "Τζάνειο Γενικό Νοσοκομείο Πειραιά", "body": "…"}) is True
    assert wiki_article_on_topic("Τέμπη σιδηροδρομικό δυστύχημα 2023",
                                 {"title": "Σιδηροδρομικό δυστύχημα στα Τέμπη", "body": "Το δυστύχημα…"}) is True
    assert wiki_article_on_topic("Αλκιβιάδου 154 Πειραιάς ιστορία",
                                 {"title": "Γερμανική εισβολή στην Ελλάδα", "body": "Η εισβολή…"}) is False
    # the §4IL any-token rule WOULD have passed the junk (every Greek word is distinctive)
    assert S.result_on_topic({"title": "Πλωτό νοσοκομείο", "body": "…", "href": ""}, S.distinctive_tokens(tz)) is True


# ── the lookup, with the HTTP client stubbed ─────────────────────────

class _Resp:
    def __init__(self, payload): self._p = payload
    def json(self): return self._p


def _fake_get(calls):
    def get(url, params=None, **kw):
        calls.append((params.get("list") or params.get("prop"), params.get("srsearch") or params.get("titles"), kw.get("proxies")))
        if params.get("list") == "search":
            return _Resp({"query": {"search": [{"title": "Δημήτρης Κουφοντίνας"}, {"title": "Κατάλογος της δράσης της 17Ν"}]}})
        title = params["titles"]
        body = ("Ο Δημήτρης Κουφοντίνας είναι Έλληνας… " * 40) if "Κουφοντίνας" in title else "Κατάλογος επιθέσεων της οργάνωσης…"
        return _Resp({"query": {"pages": {"1": {"extract": body}}}})
    return get


def test_lookup_returns_on_topic_articles_with_capped_intro_and_url(monkeypatch):
    calls = []
    fake = SimpleNamespace(get=_fake_get(calls))
    monkeypatch.setitem(__import__("sys").modules, "curl_cffi.requests", fake)
    monkeypatch.setitem(__import__("sys").modules, "curl_cffi", SimpleNamespace(requests=fake))
    out = S._wiki_lookup("Δημήτρης Κουφοντίνας", "socks5h://127.0.0.1:9050")
    assert [a["title"] for a in out] == ["Δημήτρης Κουφοντίνας — Βικιπαίδεια"]     # the list article lacks the subject
    assert len(out[0]["body"]) == S.WIKI_EXTRACT_CHARS
    assert out[0]["href"] == "https://el.wikipedia.org/wiki/%CE%94%CE%B7%CE%BC%CE%AE%CF%84%CF%81%CE%B7%CF%82_%CE%9A%CE%BF%CF%85%CF%86%CE%BF%CE%BD%CF%84%CE%AF%CE%BD%CE%B1%CF%82"
    assert calls[0][0] == "search" and all(c[2] == {"https": "socks5h://127.0.0.1:9050", "http": "socks5h://127.0.0.1:9050"} for c in calls)


# ── the merge ────────────────────────────────────────────────────────

def test_merge_puts_articles_first_and_drops_duplicates():
    wiki = [{"title": "A — Βικιπαίδεια", "body": "b", "href": "https://el.wikipedia.org/wiki/A"}]
    web = [{"title": "x", "body": "b", "href": "https://x.gr/1"}, {"title": "dup", "body": "b", "href": "https://el.wikipedia.org/wiki/A"}]
    assert [r["href"] for r in merge_wiki_first(wiki, web)] == ["https://el.wikipedia.org/wiki/A", "https://x.gr/1"]
    assert merge_wiki_first([], web) == web


# ── through the real search entry ────────────────────────────────────

def _mock_ddgs_module(text_side_effect):
    mod = MagicMock(); cls = MagicMock(); mod.DDGS = cls
    inst = MagicMock(); cls.return_value.__enter__.return_value = inst
    inst.text.side_effect = text_side_effect
    return mod, inst


ARTICLE = {"title": "Δημήτρης Κουφοντίνας — Βικιπαίδεια", "body": "Ο Δημήτρης Κουφοντίνας…", "href": "https://el.wikipedia.org/wiki/Δ"}


@pytest.mark.asyncio
async def test_article_is_placed_first_when_the_web_wave_wins(monkeypatch):
    _SEARCH_CACHE.clear()
    looked = []
    monkeypatch.setattr(S, "_wiki_lookup", lambda q, proxy: (looked.append(q), [dict(ARTICLE)])[1])
    def by_engine(q, **kw):
        return [{"title": "Κουφοντίνας: νέα δίκη", "body": "Ο Κουφοντίνας…", "href": "https://news.gr/a"}]
    mod, _ = _mock_ddgs_module(by_engine)
    with patch.dict("sys.modules", {"ddgs": mod}), patch("importlib.util.find_spec", return_value=True), \
         patch("src.ghost_agent.tools.search._DDGS_FAST_ENGINE_TIMEOUT", 0.2):
        out = await tool_search_ddgs("Δημήτρης Κουφοντίνας", None)
    assert looked == ["Δημήτρης Κουφοντίνας"]
    assert out.startswith("### 1. Δημήτρης Κουφοντίνας — Βικιπαίδεια"), out[:120]
    assert "### 2. Κουφοντίνας: νέα δίκη" in out


@pytest.mark.asyncio
async def test_article_alone_answers_when_every_web_ticket_fails(monkeypatch):
    _SEARCH_CACHE.clear()
    monkeypatch.setattr(S, "_wiki_lookup", lambda q, proxy: [dict(ARTICLE)])
    mod, _ = _mock_ddgs_module(lambda q, **kw: [])
    with patch.dict("sys.modules", {"ddgs": mod}), patch("importlib.util.find_spec", return_value=True), \
         patch("src.ghost_agent.tools.search._DDGS_FAST_ENGINE_TIMEOUT", 0.2), \
         patch("src.ghost_agent.tools.search._reformulate_query", lambda q: []):
        out = await tool_search_ddgs("Δημήτρης Κουφοντίνας", None)
    assert out.startswith("### 1. Δημήτρης Κουφοντίνας — Βικιπαίδεια") and "ZERO results" not in out


@pytest.mark.asyncio
async def test_latin_queries_never_look_up_and_a_lookup_error_never_sinks_the_search(monkeypatch):
    _SEARCH_CACHE.clear()
    looked = []
    def boom(q, proxy):
        looked.append(q); raise RuntimeError("wiki down")
    monkeypatch.setattr(S, "_wiki_lookup", boom)
    mod, _ = _mock_ddgs_module(lambda q, **kw: [{"title": "T", "body": "b", "href": "https://x.com/1"}])
    with patch.dict("sys.modules", {"ddgs": mod}), patch("importlib.util.find_spec", return_value=True), \
         patch("src.ghost_agent.tools.search._DDGS_FAST_ENGINE_TIMEOUT", 0.2):
        out_en = await tool_search_ddgs("who is dimitris koufontinas", None)
        _SEARCH_CACHE.clear()
        out_gr = await tool_search_ddgs("Δημήτρης Κουφοντίνας", None)
    assert looked == ["Δημήτρης Κουφοντίνας"]                 # the Latin query never asked
    assert "### 1. T" in out_en and "### 1. T" in out_gr        # the error was swallowed, the web batch shipped
