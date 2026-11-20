"""§4VA (2026-11-19): which search engines a wave races over Tor, and how.

Measured on the live system that day (a week of logs, 256 waves): 53 % had a winner; duckduckgo and the
clearnet brave never won; yandex won 30 % (blocked 48 %) — but 5 of 6 queries when asked on THREE fresh
circuits at once; Brave's ONION service returned a results page for 21 of 22 queries; Wikipedia's search
API answered every query, with encyclopedia articles only. And the bench that should have rested the two
dead engines lived in memory, so six restarts in one day reset it six times.

Pinned here (no network: ddgs is faked, and the transport of the engines search.py fetches itself is a
script):
  a. an engine raced on several circuits wins on any of them and has ONE outcome per wave;
  b. a fast win returns without waiting for slower engines or for Wikipedia;
  c. the Brave onion's results page → rows; a page without the markup is empty, a challenge is blocked;
  d. the Brave onion wins late, inside the wave deadline, when the others refused;
  e. the bench's outcomes are kept across restarts, written off the wave's thread;
  f. Wikipedia rides beside the wave: language, two rows, relevance, after the winner, never a win;
  g. web_search, deep_research and the fact-check counter-search all race the same wave;
  h. the privacy guard runs before every engine, the new ones included;
  i. cancelling a wave cancels every attempt and leaves no task.

Every page, title and name below is made up (owner Elena; the bakery of Markos).
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import sys
import threading
import time
import types
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ghost_agent.tools import search as S

#: the real transport (conftest replaces `S._direct_fetch` with a refusal for every test)
_REAL_DIRECT_FETCH = S._direct_fetch

PROXY = "socks5h://127.0.0.1:9050"
QUERY = "postgresql 18 release date"
ROWS = [{"title": "PostgreSQL 18 release date", "href": "https://db.example/news/pg18",
         "body": "PostgreSQL 18 release date: the release notes list asynchronous I/O."},
        {"title": "PostgreSQL 18 released", "href": "https://blog.example/postgresql-18",
         "body": "The PostgreSQL 18 release date and what changed in this release."}]
DDGS_ENGINES = ("duckduckgo", "yandex", "brave", "yahoo")


# ── scripted engines ─────────────────────────────────────────────────────────────────────────

def _blocked(engine="engine"):
    def f(proxy):
        raise RuntimeError(f"blocked: HTTP 403 from {engine}")
    return f


def _answer(rows=None, after=0.0):
    def f(proxy):
        if after:
            time.sleep(after)
        return [dict(r) for r in (ROWS if rows is None else rows)]
    return f


def _fake_ddgs(by_engine, calls):
    """ddgs with one scripted answer per backend: ``by_engine[backend](proxy)``."""
    class _DDGS:
        def __init__(self, **kw):
            self.kw = kw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, backend=None, **kw):
            calls.append((backend, str(self.kw.get("proxy") or ""), query))
            return by_engine[backend](str(self.kw.get("proxy") or ""))
    import importlib.machinery
    return types.SimpleNamespace(DDGS=_DDGS, __spec__=importlib.machinery.ModuleSpec("ddgs", None))


@contextlib.contextmanager
def _use_ddgs(fake):
    """``ddgs`` is the fake for the block — and ONLY that key is put back. (`patch.dict(sys.modules, …)`
    restores the whole dict, dropping every module first imported inside the block; a later
    `from package import module` then hands out the old object while the import system makes a new one.)"""
    had = sys.modules.get("ddgs")
    sys.modules["ddgs"] = fake
    try:
        yield fake
    finally:
        if had is None:
            sys.modules.pop("ddgs", None)
        else:
            sys.modules["ddgs"] = had


def brave_page(rows) -> str:
    """A small page in the shape of the Brave onion's result list (synthetic; class names as probed)."""
    out = ['<html><head><title>q - Brave Search</title></head><body><main><div id="results">']
    for i, r in enumerate(rows):
        snip = (f'<div class="generic-snippet svelte-1"><div class="content t-primary line-clamp-dynamic">'
                f'{r["body"]}</div></div>') if r.get("body") else ""
        out.append(
            f'<div class="snippet svelte-0" data-pos="{i}" data-type="web"><div class="result-body">'
            f'<div class="result-wrapper"><div class="result-content"><a href="{r["href"]}" class="l1">'
            f'<div class="site-name-wrapper"><cite class="snippet-url">site.example <span>› a › b</span></cite></div>'
            f'<div class="title search-snippet-title line-clamp-1" title="{r["title"]}">{r["title"]}</div></a>'
            f'{snip}</div></div></div></div>')
    out.append('</div><aside><a href="https://search.brave.com/help">Help</a>'
               '<a href="https://side.example/widget">A side widget link</a></aside></main></body></html>')
    return "".join(out)


BRAVE_ROWS = [{"title": "PostgreSQL 18 release date — Markos' notes", "href": "https://markos.example/pg18",
               "body": "PostgreSQL 18 <strong>release date</strong> and the changes in the release."},
              {"title": "PostgreSQL 18: release date", "href": "https://docs.example/postgresql/18",
               "body": "The <strong>PostgreSQL</strong> 18 release date, announced by the project."}]


class Net:
    """The scripted transport for the engines search.py fetches itself. Each handler may be a value, an
    exception to raise, or a callable; ``*_delay`` seconds are awaited first (cancellable)."""

    def __init__(self):
        self.calls = []                 # (kind, url, proxy, headers, impersonate, follow)
        self.cancelled = []             # kinds whose request was cancelled in flight
        #: the onion's results page; the default answer is a refusal, so a test's ddgs winner is never
        #: raced by it — a test that wants the page sets ``net.brave = net.page``
        self.page = (200, brave_page(BRAVE_ROWS))
        self.brave = RuntimeError("blocked: HTTP 429 from brave_onion")
        self.brave_delay = 0.0
        self.go = {}                    # path → (status, location, body)
        self.go_delay = 0.0
        self.go_now = self.go_peak = 0  # wrapped-link requests in flight / the most at once
        self.wiki = {"query": {"search": []}}
        self.wiki_status = 200
        self.wiki_delay = 0.0

    async def fetch(self, url, proxy, timeout, *, headers=None, impersonate=None, follow=True, max_bytes=0):
        u = urlparse(url)
        host = u.hostname or ""
        kind = ("wikipedia" if host.endswith("wikipedia.org") else
                "go" if u.path.startswith("/go/") else "brave" if host == S.BRAVE_ONION_HOST else "other")
        self.calls.append((kind, url, proxy, dict(headers or {}), impersonate, follow))
        if kind == "go":
            self.go_now += 1
            self.go_peak = max(self.go_peak, self.go_now)
        try:
            delay = {"wikipedia": self.wiki_delay, "go": self.go_delay, "brave": self.brave_delay}.get(kind, 0.0)
            if delay:
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self.cancelled.append(kind)
            raise
        finally:
            if kind == "go":
                self.go_now -= 1
        if kind == "wikipedia":
            v = self.wiki(url) if callable(self.wiki) else self.wiki
            if isinstance(v, BaseException):
                raise v
            return self.wiki_status, json.dumps(v), ""
        if kind == "go":
            v = self.go.get(u.path, (404, "", ""))
            if isinstance(v, BaseException):
                raise v
            return v[0], v[2], v[1]
        if kind == "brave":
            v = self.brave(url) if callable(self.brave) else self.brave
            if isinstance(v, BaseException):
                raise v
            return v[0], v[1], ""
        raise AssertionError(f"unexpected fetch: {url}")

    def kinds(self):
        return [c[0] for c in self.calls]

    def urls(self, kind):
        return [c[1] for c in self.calls if c[0] == kind]


@pytest.fixture
def net(monkeypatch):
    """The whole table racing, on a scripted transport, with a fresh in-memory ledger and bench."""
    n = Net()
    monkeypatch.setenv("GHOST_SEARCH_ENGINES_OFF", "")
    monkeypatch.setattr(S, "_direct_fetch", n.fetch)
    monkeypatch.setattr(S, "_ENGINE_WAVES", deque(maxlen=S._ENGINE_LEDGER_MAX))
    monkeypatch.setattr(S, "_bench_state", {"waves": 0})
    monkeypatch.setattr(S, "_EMPTY_SAMPLED", set())
    monkeypatch.setattr(S, "WIKI_WAIT_S", 0.3)
    S._SEARCH_CACHE.clear()
    yield n
    S.bench_flush()


def _wave(by_engine, query=QUERY, proxy=PROXY, stats=None, wave=0, **kw):
    """(rows, ddgs calls, seconds, engines-line tokens) of one wave."""
    calls, lines = [], []
    by = {e: _blocked(e) for e in DDGS_ENGINES}
    by.update(by_engine)
    real_log = S._log_wave_engines

    def _log(w, token):
        lines.append(token)
        real_log(w, token)
    took = [0.0]

    async def _main():
        t0 = time.monotonic()
        rows = await S._race_search_wave(query, proxy, wave, stats=stats, **kw)
        took[0] = time.monotonic() - t0
        if S._PROBE_TASKS:                                         # probes finish off the caller's clock
            await asyncio.wait(list(S._PROBE_TASKS), timeout=20)
        return rows
    with _use_ddgs(_fake_ddgs(by, calls)), patch.object(S, "_log_wave_engines", _log):
        rows = asyncio.run(_main())
    took = took[0]
    tokens = [dict(t.split("=") for t in line.split("engines: ", 1)[1].split()) for line in lines]
    return rows, calls, took, tokens


def _circuit(n):
    """A yandex answer on circuit ``n`` (0-based) only; the other circuits of it are refused."""
    salt = S._circuit_salt("yandex", n)

    def f(proxy):
        # the salt sits between the query hash and the attempt index of the SOCKS identity
        if f"{salt}a" in proxy:
            return [dict(r) for r in ROWS]
        raise RuntimeError("blocked: HTTP 403 from yandex")
    return f


# ── the one table ────────────────────────────────────────────────────────────────────────────

def test_one_table_names_every_engine_its_kind_circuits_and_timeout():
    t = S.ENGINE_TABLE
    assert {n: (s.kind, s.circuits) for n, s in t.items()} == {
        "duckduckgo": ("ddgs", 1), "yandex": ("ddgs", 3), "brave": ("ddgs", 1), "yahoo": ("ddgs", 1),
        "brave_onion": ("direct", 1), "wikipedia": ("supplement", 1)}
    assert t["brave_onion"].timeout == 16 and t["yandex"].timeout == 12
    assert S._RACE_ENGINES == ("duckduckgo", "yandex", "brave", "yahoo", "brave_onion")
    assert all(s.name == n for n, s in t.items())
    # the names every other reader uses come from it
    for n, s in t.items():
        assert S._engine_timeout(n) == s.timeout and S._engine_salt(n) == s.salt and S._known_engine(n)
    assert not S._known_engine("duckduckgo_onion") and not S._known_engine("yandex-h")
    # one circuit tag per (engine, circuit) — "brave" and "brave_onion" must not share one
    tags = [S._circuit_salt(n, c) for n, s in t.items() for c in range(s.circuits)]
    assert len(set(tags)) == len(tags) == 8
    # the race pool holds a thread per ddgs circuit of 8 concurrent waves (direct engines take none)
    assert S._RACE_POOL._max_workers == 6 * 8


def test_the_wave_deadline_covers_the_slowest_raced_engines_whole_timeout(monkeypatch):
    assert S._engine_timeout("brave_onion") == 16                 # the page (median 6.1 s, to ~9 s) + its links (~2.5 s)
    assert S._race_wave_deadline() == max(S._engine_timeout(e) for e in S._RACE_ENGINES) + S._RACE_WAVE_GRACE
    slow = dataclasses.replace(S.ENGINE_TABLE["brave_onion"], timeout=20)
    monkeypatch.setitem(S.ENGINE_TABLE, "brave_onion", slow)
    assert S._race_wave_deadline() == 20 + S._RACE_WAVE_GRACE     # sized from the table, never a constant
    monkeypatch.setitem(S.ENGINE_TABLE, "wikipedia", dataclasses.replace(S.ENGINE_TABLE["wikipedia"], timeout=99))
    assert S._race_wave_deadline() == 20 + S._RACE_WAVE_GRACE     # the supplement is not raced: not in it


def test_the_operator_switch_and_the_proxy_decide_what_races(monkeypatch, net):
    assert S.raced_engines(PROXY) == S._RACE_ENGINES
    assert S.raced_engines(None) == DDGS_ENGINES                  # an onion service needs Tor
    monkeypatch.setenv("GHOST_SEARCH_ENGINES_OFF", "Brave_Onion, yahoo ,no_such_engine")
    assert S.engines_switched_off() == {"brave_onion", "yahoo"}
    assert S.raced_engines(PROXY) == ("duckduckgo", "yandex", "brave")


# ── a. three circuits, one outcome ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("winning", [0, 1, 2])
def test_yandex_refused_on_two_circuits_wins_on_the_third_as_one_outcome(net, winning):
    net.brave = RuntimeError("blocked: HTTP 429 from brave_onion")
    stats = {}
    rows, calls, _took, tokens = _wave({"yandex": _circuit(winning)}, stats=stats)
    assert [r["href"] for r in rows] == [r["href"] for r in ROWS]
    yandex = [p for e, p, _q in calls if e == "yandex"]
    assert len(yandex) == 3 and len(set(yandex)) == 3             # three attempts, three SOCKS identities
    assert all(len([1 for e, _p, _q in calls if e == other]) == 1 for other in ("duckduckgo", "brave", "yahoo"))
    # ONE outcome for the engine: on the line, in the ledger the bench reads, in the search's tally
    assert len(tokens) == 1 and tokens[0]["yandex"] == "win"
    assert [o for e, o in S._ENGINE_WAVES[-1][1] if e == "yandex"] == ["win"]
    assert stats["engines"]["yandex"] == {"win": 1}


def test_three_refusals_of_one_engine_count_once(net):
    """FAILS IF the bench (or the reword-or-retry tally) sees one refusal per circuit."""
    net.brave = RuntimeError("blocked: HTTP 429 from brave_onion")
    net.wiki = RuntimeError("connect failed")
    stats = {}
    rows, calls, _took, tokens = _wave({}, stats=stats)
    assert rows == [] and len(calls) == 6
    assert tokens == [{"duckduckgo": "blocked", "yandex": "blocked", "brave": "blocked", "yahoo": "blocked",
                       "brave_onion": "blocked", "wikipedia": "conn-error"}]
    assert sorted(S._ENGINE_WAVES[-1][1]) == sorted(tokens[0].items())
    assert stats["cats"] == {"blocked": 5}                        # five engines, not seven attempts
    assert stats["engines"]["yandex"] == {"blocked": 1}
    assert S.search_failure_counts(stats) == (5, 0, 0)


def test_an_engines_outcome_is_its_most_informative_attempt(net):
    net.brave = RuntimeError("blocked: HTTP 429 from brave_onion")

    def yandex(proxy):
        if f"{S._circuit_salt('yandex', 1)}a" in proxy:
            return []                                             # this circuit reached the index: empty
        if f"{S._circuit_salt('yandex', 2)}a" in proxy:
            raise ConnectionError("error sending request")
        raise RuntimeError("blocked: HTTP 403 from yandex")
    _rows, _calls, _took, tokens = _wave({"yandex": yandex})
    assert tokens[0]["yandex"] == "empty"                         # an answer before a refusal
    assert S._best_attempt(["error sending request", "blocked: HTTP 403"]) == "blocked: HTTP 403"
    assert S._best_attempt(["engine x timed out after 12s", "error sending request"]).endswith("after 12s")
    assert S._best_attempt(["blocked: x", "low-relevance", "No results found."]) == "low-relevance"


def test_an_engine_with_a_circuit_still_out_is_not_refused_yet(net):
    """yandex: one circuit refused at once, two still running when another engine wins — the bench must
    not read that as a refusal."""
    net.brave_delay = 30.0

    def yandex(proxy):
        if f"{S._circuit_salt('yandex', 0)}a" in proxy:
            raise RuntimeError("blocked: HTTP 403 from yandex")
        time.sleep(1.0)
        raise RuntimeError("blocked: HTTP 403 from yandex")
    _rows, _calls, _took, tokens = _wave({"yandex": yandex, "yahoo": _answer(after=0.3)})
    assert tokens[0]["yahoo"] == "win" and tokens[0]["yandex"] == "unfinished"
    assert tokens[0]["duckduckgo"] == "blocked"                   # a one-circuit engine that finished: refused


# ── b. a fast win returns at once ────────────────────────────────────────────────────────────

def test_a_fast_win_does_not_wait_for_slower_engines_or_wikipedia(net):
    net.brave_delay = 60.0                                        # the Brave onion would take a minute
    net.wiki_delay = 60.0                                         # and so would Wikipedia
    rows, _calls, took, tokens = _wave({"yandex": _answer(), "yahoo": _answer(after=3.0)})
    assert [r["href"] for r in rows] == [r["href"] for r in ROWS]
    assert took < 10.0, took
    assert tokens[0]["yandex"] == "win" and tokens[0]["brave_onion"] == "unfinished"
    assert tokens[0]["wikipedia"] == "unfinished"
    assert "brave" in net.cancelled and "wikipedia" in net.cancelled   # their requests were aborted


# ── c. the Brave onion's page ────────────────────────────────────────────────────────────────

def test_the_parser_reads_title_address_and_snippet_of_every_result():
    rows = S.parse_brave_onion(brave_page(BRAVE_ROWS + [
        {"title": "Elena &amp; Markos bakery — PostgreSQL 18", "href": "https://bakery.example/pg", "body": ""},
        {"title": "A wrapped result", "href": "/go/v1/AbCdEf012", "body": "PostgreSQL 18 release date, wrapped."},
        {"title": "Brave's own page", "href": "https://search.brave.com/settings", "body": "not a result"},
        {"title": "A script link", "href": "javascript:void(0)", "body": "not a result"},
    ]))
    assert [(r["title"], r["href"]) for r in rows] == [
        ("PostgreSQL 18 release date — Markos' notes", "https://markos.example/pg18"),
        ("PostgreSQL 18: release date", "https://docs.example/postgresql/18"),
        ("Elena & Markos bakery — PostgreSQL 18", "https://bakery.example/pg"),
        ("A wrapped result", None)]
    assert rows[0]["body"] == "PostgreSQL 18 release date and the changes in the release."   # markup gone
    assert rows[2]["body"] == ""                                                        # no snippet: still a row
    assert rows[3]["go"] == "/go/v1/AbCdEf012" and "go" not in rows[0]
    # the side widget's link and Brave's own are not results
    assert not any("side.example" in str(r["href"]) or "brave" in str(r["href"]) for r in rows)


@pytest.mark.parametrize("page", [
    "", "<html><body><p>Nothing here.</p></body></html>", "<<<not html at all>>>", None,
    # the layout changed: the same links, none of the result markup
    '<html><body><div class="serp-item"><a href="https://markos.example/pg18">PostgreSQL 18</a>'
    '<p class="desc">release date</p></div></body></html>'])
def test_a_page_without_the_result_markup_is_no_rows_never_an_exception(page):
    assert S.parse_brave_onion(page) == []


def test_a_parser_fault_is_no_rows(monkeypatch):
    import bs4

    def _boom(*a, **k):
        raise RecursionError("maximum recursion depth exceeded")
    monkeypatch.setattr(bs4, "BeautifulSoup", _boom)
    assert S.parse_brave_onion(brave_page(BRAVE_ROWS)) == []


def test_a_changed_layout_still_parses_while_the_result_wrapper_stands():
    page = brave_page(BRAVE_ROWS).replace(' data-type="web"', "")
    assert [r["href"] for r in S.parse_brave_onion(page)] == [r["href"] for r in BRAVE_ROWS]


def _brave(net, query=QUERY, timeout=5.0, max_results=20):
    return asyncio.run(S._brave_onion_search(query, PROXY, timeout, max_results))


def test_the_request_asks_the_onion_through_tor_with_the_browser_profile(net):
    net.brave = net.page
    rows = _brave(net, "PostgreSQL 18 & more")
    assert [r["href"] for r in rows] == [r["href"] for r in BRAVE_ROWS]
    assert all(set(r) == {"title", "href", "body"} for r in rows)             # the shape every engine returns
    (kind, url, proxy, headers, impersonate, follow), = net.calls
    u = urlparse(url)
    assert (u.scheme, u.hostname, u.path) == ("https", S.BRAVE_ONION_HOST, "/search")
    assert parse_qs(u.query) == {"q": ["PostgreSQL 18 & more"]}                # only the query; encoded
    assert proxy == PROXY
    from ghost_agent.utils.helpers import _FETCH_IMPERSONATE
    assert impersonate == _FETCH_IMPERSONATE                                  # what every page fetch sends
    assert "User-Agent" not in headers and "Cookie" not in headers            # the profile's own; no cookie


@pytest.mark.parametrize("page,outcome", [
    ((429, "<html>Too many requests</html>"), "blocked"),
    ((200, '<html><body><form id="challenge-form" action="/captcha"><div class="g-recaptcha"></div></form></body></html>'),
     "blocked"),
    ((200, "<html><body><p>Please verify you are human to continue.</p></body></html>"), "blocked"),
    ((200, '<html><body><div class="serp-item"><a href="https://markos.example/pg18">PostgreSQL 18</a></div>'
           + "<p>No results found for your search, try other words.</p>" * 80 + "</body></html>"), "empty"),
])
def test_a_refusal_is_blocked_and_a_page_with_no_result_is_empty(net, page, outcome):
    net.brave = page
    try:
        rows = _brave(net)
        got = "empty" if rows == [] else "rows"
    except Exception as e:  # noqa: BLE001
        got = S._failure_category(S._brief_engine_error(e))
    assert got == outcome
    # …and the wave gives the engine that outcome under its own name
    _rows, _calls, _took, tokens = _wave({})
    assert tokens[0]["brave_onion"] == outcome


def test_an_empty_page_keeps_its_structure_once_for_the_operator(net, tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    net.brave = (200, "<html><body><div class='x'>Elena's secret text</div></body></html>")
    assert _brave(net) == []
    kept = (tmp_path / "system" / "search_empty_samples" / "brave_onion-empty.txt").read_text()
    assert "<div class:x>" in kept and "Elena" not in kept


def go_page(url: str) -> str:
    """A small interstitial in the shape of the one a wrapped link answers with (synthetic)."""
    return ('<html><head><title>Check this link - Brave Search</title><script>(()=>{let t=r=>{'
            'location.replace(d(s,r))};setTimeout(()=>t("2"),h)})();</script></head><body>'
            '<svg xmlns="http://www.w3.org/2000/svg"></svg><div id="main"><main class="svelte-1">'
            '<h1>Check this link before continuing</h1><p>The link you followed is sending you to '
            f'<strong class="host svelte-1">site.example</strong>: <a class="destination svelte-1" href="{url}">{url}</a></p>'
            '<p>If you do not want to visit that site, <a href="/" class="svelte-1">return to Brave Search</a>.</p>'
            f'<div class="actions svelte-1"><a href="{url}">Continue</a></div></main></div></body></html>')


def _wrapped(n, start=0):
    return [{"title": f"PostgreSQL 18 release date {i}", "href": f"/go/v1/token{i}",
             "body": "PostgreSQL 18 release date"} for i in range(start, start + n)]


def _direct(n, start=0):
    return [{"title": f"PostgreSQL 18 direct {i}", "href": f"https://direct{i}.example/pg18",
             "body": "PostgreSQL 18 release date"} for i in range(start, start + n)]


def _targets(n, start=0):
    return {f"/go/v1/token{i}": (200, "", go_page(f"https://target{i}.example/pg?x=1&amp;y={i}"))
            for i in range(start, start + n)}


def _run_brave(net, timeout=5.0):
    """(rows, unretrieved-exception messages) of one attempt."""
    unretrieved = []

    async def _main():
        import gc
        asyncio.get_running_loop().set_exception_handler(lambda l, ctx: unretrieved.append(ctx.get("message")))
        rows = await S._brave_onion_search(QUERY, PROXY, timeout, 20)
        gc.collect()
        await asyncio.sleep(0.05)
        return rows
    return asyncio.run(_main()), unretrieved


def test_the_target_is_read_from_the_interstitial():
    assert S._brave_link_target(200, "", go_page("https://target.example/a?x=1&amp;y=2")) == "https://target.example/a?x=1&y=2"
    # a changed layout: the ONE external address the page names (namespace hosts and Brave's own are not it)
    drift = go_page("https://target.example/a").replace('class="destination', 'class="dest')
    assert S._brave_link_target(200, "", drift) == "https://target.example/a"
    two = drift.replace("</main>", '<a href="https://other.example/b">other</a></main>')
    assert S._brave_link_target(200, "", two) is None                     # two candidates: no answer, no guess
    # …while the destination anchor, when it stands, decides whatever else the page links
    also = go_page("https://target.example/a").replace("</main>", '<a href="https://other.example/b">other</a></main>')
    assert S._brave_link_target(200, "", also) == "https://target.example/a"
    assert S._brave_link_target(200, "", "x" * 30_000 + go_page("https://target.example/far")) == "https://target.example/far"
    assert S._brave_link_target(200, "", '<html><svg xmlns="http://www.w3.org/2000/svg"></svg>'
                                f'<a href="https://{S.BRAVE_ONION_HOST}/">home</a></html>') is None
    # the destination must itself be a page address off Brave's hosts
    assert S._brave_link_target(200, "", go_page(f"https://{S.BRAVE_ONION_HOST}/search?q=x")) is None
    assert S._brave_link_target(200, "", go_page("javascript:alert(1)")) is None
    # secondary paths: a redirect's Location, a refresh page
    assert S._brave_link_target(302, "https://target.example/l", "") == "https://target.example/l"
    assert S._brave_link_target(302, "/go/v1/again", go_page("https://target.example/a")) is None
    assert S._brave_link_target(200, "", '<meta http-equiv="refresh" content="0;url=https://target.example/r">') == \
        "https://target.example/r"
    assert S._brave_link_target(404, "", go_page("https://target.example/a")) is None
    assert S._brave_link_target(200, "", "") is None


def test_a_wrapped_page_resolves_the_rows_that_can_be_shown_on_the_pages_circuit(net):
    net.brave = (200, brave_page(_wrapped(12)))
    net.go = _targets(12)
    net.go_delay = 0.05
    rows, unretrieved = _run_brave(net)
    # the first 8 rows, in page order; the links after them are never asked
    assert [r["href"] for r in rows] == [f"https://target{i}.example/pg?x=1&y={i}" for i in range(S.BRAVE_ONION_SHOWN)]
    go = [c for c in net.calls if c[0] == "go"]
    assert sorted(urlparse(c[1]).path for c in go) == sorted(f"/go/v1/token{i}" for i in range(8))   # once each
    assert all(c[5] is False for c in go)                                     # not followed
    assert {c[2] for c in net.calls} == {PROXY}                               # the page's circuit
    assert all(urlparse(c[1]).hostname == S.BRAVE_ONION_HOST for c in go)
    assert 1 < net.go_peak <= S.BRAVE_ONION_RESOLVE_WIDTH <= 6                # concurrent, with a small bound
    assert unretrieved == []


def test_links_that_fail_are_dropped_rows_never_invented_addresses(net):
    net.brave = (200, brave_page(_wrapped(8)))
    net.go = _targets(8)
    net.go["/go/v1/token1"] = ConnectionError("error sending request")
    net.go["/go/v1/token3"] = (200, "", "<html>nothing</html>")
    net.go["/go/v1/token4"] = (200, "", go_page(f"https://{S.BRAVE_ONION_HOST}/search?q=x"))
    net.go["/go/v1/token6"] = (503, "", "")
    rows, unretrieved = _run_brave(net)
    assert [r["href"] for r in rows] == [f"https://target{i}.example/pg?x=1&y={i}" for i in (0, 2, 5, 7)]
    assert all(set(r) == {"title", "href", "body"} for r in rows)
    assert unretrieved == []                                                  # a link that fails is quiet


@pytest.mark.parametrize("resolving,outcome", [(0, "error"), (2, "error"), (3, "rows")])
def test_a_wrapped_page_with_fewer_than_three_addresses_is_an_error(net, resolving, outcome):
    """The engine reached the index — "empty" would say the query found nothing."""
    net.brave = (200, brave_page(_wrapped(8)))
    net.go = _targets(resolving)
    try:
        rows = _brave(net)
        got = "rows" if len(rows) == resolving else "?"
    except RuntimeError as e:
        got = S._failure_category(S._brief_engine_error(e))
    assert got == outcome
    _rows, _calls, _took, tokens = _wave({})
    assert tokens[0]["brave_onion"] == ("win" if outcome == "rows" else "error")


def test_a_direct_page_asks_no_link_and_returns_every_row(net):
    net.brave = (200, brave_page(_direct(12)))
    rows = _brave(net)
    assert [r["href"] for r in rows] == [f"https://direct{i}.example/pg18" for i in range(12)]
    assert net.kinds() == ["brave"]
    # a page that simply has few results is not an error
    net.brave = (200, brave_page(_direct(1)))
    assert len(_brave(net)) == 1


def test_a_mixed_page_keeps_page_order_and_asks_only_the_wrapped_rows_shown(net):
    page = _direct(2) + _wrapped(3) + _direct(2, start=2) + _wrapped(3, start=3) + _direct(1, start=4)
    net.brave = (200, brave_page(page))                # rows 0-7 shown: d0 d1 w0 w1 w2 d2 d3 w3 | w4 w5 d4
    net.go = _targets(6)
    net.go["/go/v1/token1"] = (200, "", "<html>nothing</html>")
    rows = _brave(net)
    t = lambda i: f"https://target{i}.example/pg?x=1&y={i}"        # noqa: E731
    dd = lambda i: f"https://direct{i}.example/pg18"               # noqa: E731
    assert [r["href"] for r in rows] == [dd(0), dd(1), t(0), t(2), dd(2), dd(3), t(3), dd(4)]
    assert sorted(urlparse(u).path for u in net.urls("go")) == [f"/go/v1/token{i}" for i in range(4)]


def test_a_page_whose_links_all_stay_wrapped_is_an_error_not_an_empty_answer(net):
    """The engine reached the index — "empty" would say the query found nothing."""
    net.brave = (200, brave_page(_wrapped(1)))
    with pytest.raises(RuntimeError) as e:
        _brave(net)
    assert S._failure_category(S._brief_engine_error(e.value)) == "error"
    _rows, _calls, _took, tokens = _wave({})
    assert tokens[0]["brave_onion"] == "error"


def test_the_links_get_what_is_left_of_the_attempts_timeout(net):
    net.brave = (200, brave_page(_wrapped(1) + _direct(3)))
    net.go = _targets(1)
    net.go_delay = 20.0
    t0 = time.monotonic()
    rows = _brave(net, timeout=0.5)
    assert time.monotonic() - t0 < 10.0
    assert [r["href"] for r in rows] == [f"https://direct{i}.example/pg18" for i in range(3)]   # the unresolved row is dropped
    assert "go" in net.cancelled


# ── d. the Brave onion wins late ─────────────────────────────────────────────────────────────

def test_the_brave_onion_wins_late_when_every_other_engine_refused(net):
    net.brave = net.page
    net.brave_delay = 1.0                                          # long after the ddgs engines refused
    stats = {}
    rows, calls, took, tokens = _wave({}, stats=stats)
    assert [r["href"] for r in rows] == [r["href"] for r in BRAVE_ROWS]
    assert took >= 0.9 and took < 10.0                             # waited for it; not for the deadline
    assert tokens == [{"duckduckgo": "blocked", "yandex": "blocked", "brave": "blocked", "yahoo": "blocked",
                       "brave_onion": "win", "wikipedia": "empty"}]
    assert stats["engines"]["brave_onion"] == {"win": 1}
    assert "socks5h://" in net.calls[0][2] and net.calls[0][2] != PROXY       # its own circuit, Tor resolves


def test_the_wave_is_not_called_empty_while_the_onion_is_inside_its_timeout(net, monkeypatch):
    """The deadline is sized from the table: an onion answer at 0.8 s of a 1 s timeout wins; with the
    onion's timeout cut to 0.3 s the same answer is a timeout and the wave is lost."""
    net.brave = net.page
    net.brave_delay = 0.8
    net.wiki = RuntimeError("connect failed")
    monkeypatch.setattr(S, "_RACE_WAVE_GRACE", 0.5)
    for name in S._RACE_ENGINES:
        monkeypatch.setitem(S.ENGINE_TABLE, name, dataclasses.replace(S.ENGINE_TABLE[name], timeout=1.0))
    rows, _calls, _took, tokens = _wave({})
    assert rows and tokens[0]["brave_onion"] == "win"

    async def _slow_onion(url, proxy, timeout, **kw):
        await asyncio.sleep(0.8)
        if timeout < 0.8:
            raise asyncio.TimeoutError()
        return 200, brave_page(BRAVE_ROWS), ""
    monkeypatch.setattr(S, "_direct_fetch", _slow_onion)
    monkeypatch.setitem(S.ENGINE_TABLE, "brave_onion", dataclasses.replace(S.ENGINE_TABLE["brave_onion"], timeout=0.3))
    rows, _calls, _took, tokens = _wave({})
    assert rows == [] and tokens[0]["brave_onion"] == "timeout"


# ── e. the bench survives a restart ──────────────────────────────────────────────────────────

def _token(**outcomes):
    return " | engines: " + " ".join(f"{e}={o}" for e, o in outcomes.items())


def _dead_week(n=30, step=60.0, end_ago=30.0):
    """n waves, the last ``end_ago`` seconds ago, in which duckduckgo and the clearnet brave only refuse.
    Returns the time of the last one."""
    t0 = time.time() - end_ago - (n - 1) * step
    for i in range(n):
        S._record_wave_engines(_token(duckduckgo="blocked", yandex="win", brave="timeout", yahoo="empty",
                                      brave_onion="unfinished", wikipedia="added"), now=t0 + i * step)
    return t0 + (n - 1) * step


def _restart(monkeypatch):
    """What a new process starts with: an empty ledger, an unread bench file."""
    S.bench_flush()
    monkeypatch.setattr(S, "_ENGINE_WAVES", deque(maxlen=S._ENGINE_LEDGER_MAX))
    monkeypatch.setattr(S, "_bench_state", {"waves": 0})


def _write_bench(engines):
    os.makedirs(os.path.dirname(S.bench_file()), exist_ok=True)
    with open(S.bench_file(), "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "engines": engines}, fh)


def _refusals(n=40, newest_ago=60.0, step=60.0, outcome="blocked"):
    now = time.time()
    return [[now - newest_ago - (n - 1 - i) * step, outcome] for i in range(n)]


def test_the_outcomes_are_written_bounded_and_without_a_query(net):
    last = _dead_week(n=S.ENGINE_BENCH_KEEP + 15)
    assert S.bench_flush() is True
    assert S.bench_file() == os.path.join(os.environ["GHOST_HOME"], "system", "search_engine_bench.json")
    data = json.load(open(S.bench_file(), encoding="utf-8"))
    assert data["version"] == 1
    # the raced engines' DECISIVE outcomes only: no supplement, no "unfinished"
    assert set(data["engines"]) == {"duckduckgo", "yandex", "brave", "yahoo"}
    assert all(len(v) == S.ENGINE_BENCH_KEEP for v in data["engines"].values())
    assert {o for _t, o in data["engines"]["brave"]} == {"timeout"}
    ts = [t for t, _o in data["engines"]["duckduckgo"]]
    assert ts == sorted(ts) and ts[-1] == last                                    # newest last
    assert not [f for f in os.listdir(os.path.dirname(S.bench_file())) if f.endswith(".tmp")]


def test_a_fresh_process_benches_the_dead_engines_on_its_first_wave(net, monkeypatch):
    _dead_week()
    _restart(monkeypatch)
    assert len(S._ENGINE_WAVES) == 0
    assert S.raced_engines(PROXY) == ("yandex", "yahoo", "brave_onion")          # the FIRST wave already
    # …and the first real wave of the new process asks neither of them
    _restart(monkeypatch)
    rows, calls, _took, tokens = _wave({"yandex": _answer()})
    assert rows and {e for e, _p, _q in calls} == {"yandex", "yahoo"}
    assert set(tokens[0]) == {"yandex", "yahoo", "brave_onion", "wikipedia"}


def test_what_the_last_process_kept_is_never_a_wave_of_this_one(net, monkeypatch):
    _dead_week()
    _restart(monkeypatch)
    S.raced_engines(PROXY)
    n = len(S._bench_state["extra"])
    assert n > 0 and len(S._ENGINE_WAVES) == 0                                    # read — as bench rows only
    S.raced_engines(PROXY)
    S._record_wave_engines(_token(wikipedia="added"), now=time.time())
    assert len(S._bench_state["extra"]) == n                                      # …once per process
    monkeypatch.setattr(S, "_PROCESS_T0", 0.0)                                    # even with the window wide open
    h = S.engine_health()
    assert h["waves"] == 1 and set(h["engines"]) == {"wikipedia"}


def test_a_record_before_the_first_bench_question_keeps_the_history(net, monkeypatch):
    _dead_week()
    _restart(monkeypatch)
    # the first thing this process does is RECORD (no bench question yet): the kept rows are read first,
    # so the write that follows does not replace the file with one wave
    S._record_wave_engines(_token(duckduckgo="empty", yandex="win"), now=time.time())
    raced = S.raced_engines(PROXY)
    assert "duckduckgo" in raced and "brave" not in raced                         # one answer: back in the race
    _restart(monkeypatch)
    raced = S.raced_engines(PROXY)
    assert "duckduckgo" in raced and "brave" not in raced                         # and that was written too


@pytest.mark.parametrize("content", [
    None, "", "{not json", "[]", '{"version": 1}', '{"version": 1, "engines": []}',
    '{"version": 1, "engines": {"duckduckgo": "blocked"}}',
    '{"version": 1, "engines": {"duckduckgo": [["x", "blocked"], [1, 2, 3], null, [true, "blocked"], [5.0, "evil"]]}}',
    "\x00\xff\xfe binary"])
def test_a_missing_or_damaged_bench_file_is_an_empty_bench(net, monkeypatch, content):
    os.makedirs(os.path.dirname(S.bench_file()), exist_ok=True)
    if content is not None:
        with open(S.bench_file(), "w", encoding="utf-8", errors="replace") as fh:
            fh.write(content)
    assert S.raced_engines(PROXY) == S._RACE_ENGINES
    assert len(S._ENGINE_WAVES) == 0 and not S._bench_state.get("extra")
    # and the next wave writes a good file over it
    now = time.time()
    S._record_wave_engines(_token(yandex="win"), now=now)
    S.bench_flush()
    assert json.load(open(S.bench_file(), encoding="utf-8"))["engines"] == {"yandex": [[now, "win"]]}


def test_only_rows_of_the_tables_shape_and_of_the_last_day_are_taken(net):
    _write_bench({
        "duckduckgo": _refusals(),
        "wikipedia": _refusals(),                                 # the supplement is never benched
        "mojeek": _refusals(),                                    # not an engine of the table
        "brave": [[time.time() + 86400 * 365 + i * 60, "blocked"] for i in range(40)],   # from the future
        "yahoo": [[t, "Blocked!"] for t, _o in _refusals()],
        "yandex": _refusals(newest_ago=S.ENGINE_BENCH_MAX_AGE_S + 3600),    # r1: older than a day — expired
    })
    assert S.raced_engines(PROXY) == ("yandex", "brave", "yahoo", "brave_onion")
    assert {e for _t, e, _o in S._bench_state["extra"]} == {"duckduckgo"}


def test_an_oversized_bench_file_is_not_read(net, monkeypatch):
    _write_bench({"duckduckgo": _refusals()})
    monkeypatch.setattr(S, "BENCH_FILE_MAX_BYTES", 100)
    assert S.raced_engines(PROXY) == S._RACE_ENGINES and not S._bench_state.get("extra")


def test_the_probe_wave_and_the_minimum_still_hold_after_a_restart(net, monkeypatch):
    _dead_week()
    _restart(monkeypatch)
    seen = [S.raced_engines(PROXY) for _ in range(S.ENGINE_BENCH_PROBE_EVERY)]
    assert [("duckduckgo" in r) for r in seen].count(True) == 1                  # one wave in K races everyone
    assert seen[-1] == S._RACE_ENGINES and S.probed_engines() == {"duckduckgo", "brave"}
    # r2: while those probes are in flight the next K-th wave does not launch them again
    seen = [S.raced_engines(PROXY) for _ in range(S.ENGINE_BENCH_PROBE_EVERY)]
    assert [("duckduckgo" in r) for r in seen].count(True) == 0
    S._bench_state["in_flight"].clear()                                           # (their time passed)
    seen = [S.raced_engines(PROXY) for _ in range(S.ENGINE_BENCH_PROBE_EVERY)]
    assert [("duckduckgo" in r) for r in seen].count(True) == 1
    # every engine refused for the whole kept history: nobody sits out (never below the minimum)
    _restart(monkeypatch)
    _write_bench({e: _refusals() for e in S._RACE_ENGINES})
    assert S.raced_engines(PROXY) == S._RACE_ENGINES
    # …and a burst of refusals inside ten minutes is not a dead engine, kept or not
    _restart(monkeypatch)
    _write_bench({"yandex": _refusals(step=1.0), "yahoo": _refusals(outcome="win")})
    assert "yandex" in S.raced_engines(PROXY)


def test_the_two_engine_floor_counts_the_engines_that_can_race(net, monkeypatch):
    """r1: with two engines switched off (or the onion without a proxy), benching two more left ONE."""
    _write_bench({"duckduckgo": _refusals(), "brave": _refusals()})
    assert S.raced_engines(PROXY) == ("yandex", "yahoo", "brave_onion")          # 5 can race: 3 left
    monkeypatch.setenv("GHOST_SEARCH_ENGINES_OFF", "yahoo,brave_onion")
    assert S.raced_engines(PROXY) == ("duckduckgo", "yandex", "brave")           # 3 can race: benching 2 leaves 1 — nobody sits out
    monkeypatch.setenv("GHOST_SEARCH_ENGINES_OFF", "yahoo")
    assert S.raced_engines(None) == ("duckduckgo", "yandex", "brave")            # the onion cannot race without Tor
    assert S.raced_engines(PROXY) == ("yandex", "brave_onion")


def test_a_benched_engine_is_raced_again_by_the_clock_whatever_the_wave_count(net, monkeypatch):
    """r1 (the reviewer's file): 30 yandex refusals, three hours old — 8 restarts of 6 waves each raced
    yandex in 0 of 48 waves (the probe was wave 10 of a counter that restarts at 0)."""
    raced_per_process = []
    for _process in range(8):
        _restart(monkeypatch)
        _write_bench({"yandex": _refusals(newest_ago=3 * 3600.0)})
        assert "yandex" in S.raced_engines(PROXY) and S.probed_engines() == {"yandex"}   # the FIRST wave: a probe
        raced_per_process.append(1 + sum("yandex" in S.raced_engines(PROXY) for _ in range(5)))
        assert S.probed_engines() == frozenset()
    # r2: one probe in flight at a time — the five waves behind it leave the engine benched
    assert all(n == 1 for n in raced_per_process), raced_per_process
    # inside the half hour it rests; past it, it is due
    _restart(monkeypatch)
    _write_bench({"yandex": _refusals(newest_ago=60.0)})
    newest = max(t for t, _o in json.load(open(S.bench_file()))["engines"]["yandex"])
    assert "yandex" in S.benched_engines(now=newest + S.ENGINE_BENCH_PROBE_AFTER_S - 5)
    assert S.probed_engines() == frozenset()
    assert "yandex" not in S.benched_engines(now=newest + S.ENGINE_BENCH_PROBE_AFTER_S + 5)
    assert S.probed_engines() == {"yandex"}


@pytest.mark.parametrize("late,then_raced", [("blocked", False), ("answer", True)])
def test_a_probe_another_engine_beats_still_books_its_outcome(net, monkeypatch, late, then_raced):
    """r1: on the probe wave yahoo wins at once; yandex's circuits answer 0.3 s later. The caller has its
    rows without waiting — and yandex's outcome is booked: a refusal renews its rest, an answer ends it."""
    _write_bench({"yandex": _refusals(newest_ago=3 * 3600.0)})

    def yandex(proxy):
        time.sleep(0.3)
        if late == "blocked":
            raise RuntimeError("blocked: HTTP 403 from yandex")
        return [dict(r) for r in ROWS]
    rows, calls, took, tokens = _wave({"yandex": yandex, "yahoo": _answer()})
    assert rows and tokens[0]["yahoo"] == "win" and tokens[0]["yandex"] == "probe-unfinished"   # r2: said to be a probe
    assert len([1 for e, _p, _q in calls if e == "yandex"]) == 3 and took < 10.0
    assert "yandex" not in S.engine_health()["engines"]
    assert ("yandex" in S.raced_engines(PROXY)) is then_raced
    assert S.probed_engines() == frozenset()                                     # either way: no longer due
    assert S.engine_health()["waves"] == 1                                       # the late outcome is not a wave
    _restart(monkeypatch)
    assert ("yandex" in S.raced_engines(PROXY)) is then_raced                    # and it was written


def test_the_file_is_written_off_the_waves_thread(net, monkeypatch):
    gate, started, threads = threading.Event(), threading.Event(), []
    real = S._bench_write

    def _slow_disk(path, snapshot):
        threads.append(threading.current_thread().name)
        started.set()
        gate.wait(30)
        real(path, snapshot)
    monkeypatch.setattr(S, "_bench_write", _slow_disk)
    try:
        t0 = time.monotonic()
        S._record_wave_engines(_token(yandex="win", duckduckgo="blocked"), now=time.time())
        took = time.monotonic() - t0
        assert started.wait(10) and took < 10.0                   # returned while the disk is stuck
        assert not os.path.exists(S.bench_file())
        assert S.bench_flush(timeout=0.05) is False
    finally:
        gate.set()
    assert S.bench_flush() is True and os.path.exists(S.bench_file())
    assert threads and all(n.startswith("search-bench") for n in threads)
    assert threading.current_thread().name not in threads


def test_a_disk_that_refuses_never_reaches_the_wave(net, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", "/dev/null/not-a-directory")
    rows, _calls, _took, _tokens = _wave({"yandex": _answer()})
    assert rows and S.bench_flush() is True                       # the wave won; the write failed quietly
    assert S.raced_engines(PROXY) == S._RACE_ENGINES


def test_a_real_wave_feeds_the_bench_file(net):
    _wave({"yandex": _answer(after=0.4)})                          # the one-circuit engines have refused by then
    S.bench_flush()
    engines = json.load(open(S.bench_file(), encoding="utf-8"))["engines"]
    assert [o for _t, o in engines["yandex"]] == ["win"]
    assert [o for _t, o in engines["duckduckgo"]] == ["blocked"] and "wikipedia" not in engines


def test_the_bench_rule(net):
    assert (S.ENGINE_BENCH_AFTER, S.ENGINE_BENCH_PROBE_EVERY, S.ENGINE_BENCH_MIN_SPAN_S, S.ENGINE_BENCH_MIN_RACED) == (
        30, 10, 600.0, 2)
    assert S.ENGINE_BENCH_KEEP >= S.ENGINE_BENCH_AFTER
    assert S.ENGINE_BENCH_PROBE_AFTER_S == 7200.0 and S.ENGINE_BENCH_MAX_AGE_S == 86400.0
    # 29 refusals are not 30
    _write_bench({"duckduckgo": _refusals(n=29)})
    assert "duckduckgo" in S.raced_engines(PROXY)


# ── f. Wikipedia beside the wave ─────────────────────────────────────────────────────────────

def _hits(*pairs):
    return {"batchcomplete": "", "query": {"searchinfo": {"totalhits": len(pairs)}, "search": [
        {"ns": 0, "title": t, "pageid": 100 + i, "snippet": s, "titlesnippet": ""} for i, (t, s) in enumerate(pairs)]}}


WIKI_HITS = _hits(
    ("PostgreSQL", 'The <span class="searchmatch">PostgreSQL</span> 18 <span class="searchmatch">release</span> '
                   '<span class="searchmatch">date</span> was in the autumn; a -&gt; b &amp; c'),
    ("Elena's bakery", "A bakery run by Markos; nothing about databases at all, only bread and cake."),
    ("PostgreSQL release history (18)", '<span class="searchmatch">PostgreSQL</span> 18 release date and earlier releases'),
    ("Release date of PostgreSQL 18", "PostgreSQL 18 release date, a third matching article"))


@pytest.mark.parametrize("query,host", [("postgresql 18 release date", "en.wikipedia.org"),
                                        ("Ποιος έγραψε την Οδύσσεια", "el.wikipedia.org"),
                                        ("καιρός Athens αύριο", "el.wikipedia.org"),
                                        ("python asyncio gather α", "en.wikipedia.org"),      # r1: MOST letters decide
                                        ("athens weather tomorrow", "en.wikipedia.org")])
def test_the_language_follows_the_querys_script(net, query, host):
    assert asyncio.run(S._wikipedia_search(query, PROXY, 5.0)) == []
    (kind, url, proxy, headers, impersonate, _follow), = net.calls
    u = urlparse(url)
    assert kind == "wikipedia" and (u.scheme, u.hostname, u.path) == ("https", host, "/w/api.php")
    q = parse_qs(u.query)
    assert q["srsearch"] == [query] and q["list"] == ["search"] and q["format"] == ["json"]
    assert set(q) == {"action", "list", "format", "srlimit", "srprop", "srsearch"}        # no key, no identity
    # §4UN-web / §4QF: a browser-impersonating client is answered 429 over Tor — sent plain
    assert impersonate is None and headers["User-Agent"].startswith("curl/")


def test_hits_become_rows_with_the_markup_stripped_and_the_articles_address():
    rows = S.parse_wikipedia_hits(_hits(("AC/DC (band) & co", 'x <span class="searchmatch">y</span> &lt;script&gt;'),
                                        ("Οδύσσεια", "Η <span class=\"searchmatch\">Οδύσσεια</span> είναι έπος"),
                                        ("", "no title")), "el")
    assert rows == [
        {"title": "AC/DC (band) & co", "body": "x y <script>", "engine": "wikipedia",
         "href": "https://el.wikipedia.org/wiki/AC/DC_(band)_%26_co"},
        {"title": "Οδύσσεια", "body": "Η Οδύσσεια είναι έπος", "engine": "wikipedia",
         "href": "https://el.wikipedia.org/wiki/%CE%9F%CE%B4%CF%8D%CF%83%CF%83%CE%B5%CE%B9%CE%B1"}]
    for bad in (None, [], "x", {"query": None}, {"query": {"search": "x"}}, {"query": {"search": [None, 3]}}):
        assert S.parse_wikipedia_hits(bad, "en") == []


def test_two_on_topic_rows_are_appended_after_the_winners(net):
    net.wiki = WIKI_HITS
    rows, _calls, _took, tokens = _wave({"yandex": _answer()})
    assert [r["href"] for r in rows] == [ROWS[0]["href"], ROWS[1]["href"],
                                         "https://en.wikipedia.org/wiki/PostgreSQL",
                                         "https://en.wikipedia.org/wiki/PostgreSQL_release_history_(18)"]
    assert [r.get("engine") for r in rows] == [None, None, "wikipedia", "wikipedia"]      # marked; at most two
    assert rows[2]["body"] == "The PostgreSQL 18 release date was in the autumn; a -> b & c"
    assert S.wave_wikipedia(rows) == rows[2:]
    assert tokens[0]["wikipedia"] == "added" and tokens[0]["yandex"] == "win"


GREEK_QUERY = "Οδύσσεια περίληψη ραψωδιών"
GREEK_ROWS = [{"title": "Οδύσσεια Ομήρου — περίληψη", "href": "https://sxoleio.example/odysseia",
               "body": "Η Οδύσσεια του Ομήρου: περίληψη των ραψωδιών."},
              {"title": "Περίληψη της Οδύσσειας του Ομήρου", "href": "https://vivlia.example/odysseia",
               "body": "Οδύσσεια Ομήρου, περίληψη και ανάλυση κάθε ραψωδίας."}]
GREEK_HITS = _hits(("Οδύσσεια", 'Η <span class="searchmatch">Οδύσσεια</span>: περίληψη των ραψωδιών του έπους'),
                   ("Όμηρος", "Ο Όμηρος, η Οδύσσεια και μια περίληψη των ραψωδιών"),      # not ABOUT the subject
                   ("Ραψωδίες της Οδύσσειας", "Περίληψη κάθε ραψωδίας της Οδύσσειας"),
                   ("Οδύσσεια (ταινία)", "Η Οδύσσεια, περίληψη των ραψωδιών στην ταινία"))


def test_for_a_greek_query_the_best_wikipedia_row_leads_the_batch(net):
    """§4JO's property (the owner writes Greek): el.wikipedia's article first — ONE (r1: the second
    stays last), still two rows at most, still about the subject, still never a win alone. Any other
    query: after the winner's rows."""
    net.wiki = GREEK_HITS
    rows, _calls, _took, tokens = _wave({"yandex": _answer(GREEK_ROWS)}, query=GREEK_QUERY)
    assert [r["title"] for r in rows] == ["Οδύσσεια", GREEK_ROWS[0]["title"], GREEK_ROWS[1]["title"],
                                          "Ραψωδίες της Οδύσσειας"]
    assert rows[0]["href"] == "https://el.wikipedia.org/wiki/%CE%9F%CE%B4%CF%8D%CF%83%CF%83%CE%B5%CE%B9%CE%B1"
    assert [r.get("engine") for r in rows] == ["wikipedia", None, None, "wikipedia"]
    assert tokens[0]["yandex"] == "win" and tokens[0]["wikipedia"] == "added"
    assert S.wikipedia_first(GREEK_QUERY) and not S.wikipedia_first(QUERY)
    web, wiki = [{"href": "w1"}, {"href": "w2"}], [{"href": "k1"}, {"href": "k2"}]
    assert S.order_with_wikipedia(GREEK_QUERY, web, wiki) == [wiki[0]] + web + [wiki[1]]
    assert S.order_with_wikipedia(GREEK_QUERY, web, wiki[:1]) == wiki[:1] + web
    assert S.order_with_wikipedia(QUERY, web, wiki) == web + wiki
    assert S.order_with_wikipedia(GREEK_QUERY, web, []) == web
    # alone it is still not a win
    rows, _calls, _took, tokens = _wave({}, query=GREEK_QUERY)
    assert rows == [] and len(S.wave_wikipedia(rows)) == 2 and "win" not in tokens[0].values()
    # and articles that are not about the subject are not placed anywhere
    net.wiki = _hits(("Όμηρος", "Ο Όμηρος, η Οδύσσεια και μια περίληψη των ραψωδιών"),
                     ("Φούρνος της Έλενας", "Ο φούρνος του Μάρκου στην Αθήνα φτιάχνει ψωμί και γλυκά κάθε πρωί εδώ και χρόνια."))
    rows, _calls, _took, tokens = _wave({"yandex": _answer(GREEK_ROWS)}, query=GREEK_QUERY)
    assert [r["href"] for r in rows] == [r["href"] for r in GREEK_ROWS] and tokens[0]["wikipedia"] == "off-topic"


def test_web_search_lists_one_wikipedia_row_first_for_a_greek_query(net):
    net.wiki = GREEK_HITS
    with _use_ddgs(_fake_ddgs({e: _answer(GREEK_ROWS) for e in DDGS_ENGINES}, [])):
        out = asyncio.run(S.tool_search_ddgs(GREEK_QUERY, PROXY))
    heads = [l for l in out.splitlines() if l.startswith("### ")]
    assert heads == ["### 1. Οδύσσεια", "### 2. Οδύσσεια Ομήρου — περίληψη", "### 3. Περίληψη της Οδύσσειας του Ομήρου",
                     "### 4. Ραψωδίες της Οδύσσειας"]
    assert out.count(f"— {S.WIKIPEDIA_TAG}]") == 2
    assert out.index("el.wikipedia.org") < out.index("sxoleio.example") < out.rindex("el.wikipedia.org")


def test_the_rows_pass_the_relevance_rule_every_row_passes(net):
    off = _hits(("Elena's bakery", "A bakery run by Markos; nothing about databases at all, only bread and cake."),
                ("Release (music)", "A release date in music is the day an album is first sold to the public."))
    net.wiki = off
    rows, _calls, _took, tokens = _wave({"yandex": _answer()})
    assert [r["href"] for r in rows] == [r["href"] for r in ROWS] and tokens[0]["wikipedia"] == "off-topic"
    # each row is judged alone — no best-of-the-rest backfill
    hits = S.parse_wikipedia_hits(off, "en")
    assert S.prune_off_topic(QUERY, hits)[0] == hits              # the BATCH rule keeps its best three…
    assert S.wikipedia_rows(QUERY, hits) == []                    # …a row from outside the race gets none
    good = S.parse_wikipedia_hits(WIKI_HITS, "en")
    assert [r["title"] for r in good if S.wiki_article_on_topic(QUERY, r)] == [
        "PostgreSQL", "PostgreSQL release history (18)", "Release date of PostgreSQL 18"]
    assert S.wikipedia_rows(QUERY, [{"href": "https://en.wikipedia.org/wiki/X"}]) == []   # no text: not on trust
    # a row that names the subject but matches under 30 % of the query: kept by the batch rule's
    # allowance, dropped here
    weak = [{"title": "PostgreSQL pricing", "body": "cloud plans", "href": "https://en.wikipedia.org/wiki/P"}]
    assert S.prune_off_topic("postgresql 18 release date notes", weak)[0] == weak
    assert S.wikipedia_rows("postgresql 18 release date notes", weak) == []


def test_a_page_the_winner_already_lists_is_not_added_twice(net):
    net.wiki = WIKI_HITS
    winner = ROWS + [{"title": "PostgreSQL - Wikipedia", "href": "https://en.wikipedia.org/wiki/PostgreSQL#History",
                      "body": "PostgreSQL 18 release date and history."}]
    rows, _calls, _took, _tokens = _wave({"yandex": _answer(winner)})
    assert [r["href"] for r in rows].count("https://en.wikipedia.org/wiki/PostgreSQL") == 0
    assert [r["href"] for r in rows if r.get("engine") == "wikipedia"] == [
        "https://en.wikipedia.org/wiki/PostgreSQL_release_history_(18)",
        "https://en.wikipedia.org/wiki/Release_date_of_PostgreSQL_18"]


def test_wikipedia_alone_never_wins_a_wave(net):
    net.wiki = WIKI_HITS
    net.brave = RuntimeError("blocked: HTTP 429 from brave_onion")
    stats = {}
    rows, _calls, _took, tokens = _wave({}, stats=stats)
    assert rows == [] and not rows                                 # the wave is LOST: the caller retries
    assert [r["title"] for r in S.wave_wikipedia(rows)] == ["PostgreSQL", "PostgreSQL release history (18)"]
    assert tokens[0]["wikipedia"] == "added" and "win" not in tokens[0].values()
    assert "wikipedia" not in stats.get("engines", {}) and stats["cats"] == {"blocked": 5}
    assert S.engine_health()["won"] == 0


LOW_QUERY = "postgresql 18 release date notes"
LOW = [{"title": "PostgreSQL pricing", "body": "cloud plans", "href": "https://cloud.example/pricing"}]


def test_a_low_relevance_batch_stays_low_with_wikipedia_beside_it(net):
    net.wiki = WIKI_HITS
    with patch.object(S, "LOW_RELEVANCE_GRACE_S", 0.2):
        rows, _calls, _took, tokens = _wave({"yandex": _answer(LOW)}, query=LOW_QUERY)
    assert [r["href"] for r in rows] == ["https://cloud.example/pricing"]      # not turned into a win
    assert S.prune_off_topic(LOW_QUERY, list(rows))[1] is True
    assert len(S.wave_wikipedia(rows)) == 2 and tokens[0]["yandex"] == "low-relevance"


@pytest.mark.parametrize("failure", [RuntimeError("connect failed"), asyncio.TimeoutError(),
                                     ValueError("not json"), "http-429", "garbage-json", "late"])
def test_a_wikipedia_failure_is_silent(net, monkeypatch, failure):
    """No operator line, no outcome in the search's counts, no unretrieved exception — only its token on
    the durable engines line."""
    if failure == "http-429":
        net.wiki_status = 429
    elif failure == "garbage-json":
        async def _garbage(url, proxy, timeout, **kw):
            if "wikipedia" in url:
                return 200, "<html>not json</html>", ""
            return await net.fetch(url, proxy, timeout, **kw)
        monkeypatch.setattr(S, "_direct_fetch", _garbage)
    elif failure == "late":
        net.wiki_delay = 60.0
    else:
        net.wiki = failure
    unretrieved, logged, stats = [], [], {}

    async def _main():
        import gc
        asyncio.get_running_loop().set_exception_handler(lambda l, ctx: unretrieved.append(ctx.get("message")))
        rows = await S._race_search_wave(QUERY, PROXY, 0, stats=stats)
        await asyncio.sleep(0.05)
        gc.collect()
        return rows
    calls = []
    by = {e: _blocked(e) for e in DDGS_ENGINES}
    by["yandex"] = _answer()
    with _use_ddgs(_fake_ddgs(by, calls)), \
            patch.object(S, "pretty_log", lambda title, *a, **k: logged.append(title)):
        rows = asyncio.run(_main())
    import gc
    gc.collect()
    assert [r["href"] for r in rows] == [r["href"] for r in ROWS]
    assert not [t for t in logged if "wiki" in t.lower()]
    assert "wikipedia" not in stats.get("engines", {}) and not stats.get("cats")
    assert not [m for m in unretrieved if "never retrieved" in str(m)], unretrieved
    token = dict(S._ENGINE_WAVES[-1][1])["wikipedia"]
    assert token == {"http-429": "blocked", "late": "unfinished"}.get(
        failure if isinstance(failure, str) else "", token) and token not in ("added", "win")
    assert token in S._KNOWN_ENGINE_OUTCOMES


def test_web_search_lists_wikipedia_after_the_web_rows_and_tags_it(net, monkeypatch):
    net.wiki = WIKI_HITS
    many = [{"title": f"PostgreSQL 18 release date part {i}", "href": f"https://db{i}.example/pg18",
             "body": "PostgreSQL 18 release date notes."} for i in range(12)]
    with _use_ddgs(_fake_ddgs({e: _answer(many) for e in DDGS_ENGINES}, [])):
        out = asyncio.run(S.tool_search_ddgs(QUERY, PROXY))
    heads = [l for l in out.splitlines() if l.startswith("### ")]
    assert len(heads) == 10                                        # the 8 web rows are all still there
    assert heads[8] == "### 9. PostgreSQL" and heads[9].startswith("### 10. PostgreSQL release history")
    assert f"[Source: https://en.wikipedia.org/wiki/PostgreSQL — {S.WIKIPEDIA_TAG}]" in out
    assert out.count(f"— {S.WIKIPEDIA_TAG}]") == 2 and S.WIKIPEDIA_ONLY_NOTE not in out


def test_a_search_no_engine_wins_retries_as_before_then_shows_wikipedia_and_says_so(net, monkeypatch):
    net.wiki = WIKI_HITS
    net.brave = RuntimeError("blocked: HTTP 429 from brave_onion")
    waves = []
    real = S._race_search_wave

    async def _counted(query, proxy, wave, **kw):
        waves.append(wave)
        return await real(query, proxy, wave, **kw)
    monkeypatch.setattr(S, "_race_search_wave", _counted)

    async def _no_sleep(_s):
        return None
    with _use_ddgs(_fake_ddgs({e: _blocked(e) for e in DDGS_ENGINES}, [])), \
            patch.object(S.asyncio, "sleep", _no_sleep):
        out = asyncio.run(S.tool_search_ddgs(QUERY, PROXY))
    assert waves == [0, 1, 2]                                      # both waves and the fresh-circuit retry ran
    assert isinstance(out, str) and out.startswith(S.WIKIPEDIA_ONLY_NOTE)
    assert out.count("### ") == 2 and "https://en.wikipedia.org/wiki/PostgreSQL" in out
    assert not S._SEARCH_CACHE                                     # never cached: the engines may recover
    # the kept rows are judged against the query AS ASKED
    net.wiki = _hits(("Elena's bakery", "A bakery run by Markos; nothing about databases, only bread and cake."))
    with _use_ddgs(_fake_ddgs({e: _blocked(e) for e in DDGS_ENGINES}, [])), \
            patch.object(S.asyncio, "sleep", _no_sleep), patch.object(S, "_proxy_reachable", lambda *a, **k: True):
        out = asyncio.run(S.tool_search_ddgs(QUERY, PROXY))
    assert "TRANSPORT" in str(getattr(out, "text", out)) and "wikipedia" not in str(getattr(out, "text", out)).lower()


def test_a_low_batch_lists_wikipedia_after_it_outside_the_low_note(net, monkeypatch):
    net.wiki = WIKI_HITS
    by = {e: (lambda proxy: []) for e in DDGS_ENGINES}             # the rivals ANSWERED: the low batch stands
    by["yandex"] = _answer(LOW)
    with _use_ddgs(_fake_ddgs(by, [])), patch.object(S, "LOW_RELEVANCE_GRACE_S", 0.2):
        out = asyncio.run(S.tool_search_ddgs(LOW_QUERY, PROXY))
    assert out.startswith(S.LOW_RELEVANCE_NOTE)
    assert out.index("cloud.example/pricing") < out.index("en.wikipedia.org/wiki/PostgreSQL")
    assert f"— {S.WIKIPEDIA_LOW_TAG}]" in out and not S._SEARCH_CACHE


def test_the_supplement_can_be_switched_off(net, monkeypatch):
    monkeypatch.setenv("GHOST_SEARCH_ENGINES_OFF", "wikipedia")
    net.wiki = WIKI_HITS
    rows, _calls, _took, tokens = _wave({"yandex": _answer()})
    assert [r["href"] for r in rows] == [r["href"] for r in ROWS]
    assert "wikipedia" not in tokens[0] and "wikipedia" not in net.kinds()


# ── g. every caller races the same wave ──────────────────────────────────────────────────────

def _callers():
    async def web_search(by, calls):
        return await S.tool_search("PostgreSQL 18 release date", anonymous=False, tor_proxy=PROXY)

    async def deep_research(by, calls):
        async def fetch(url, **kw):
            calls.append(("fetch", url, ""))
            return "PostgreSQL 18 release date: the page text. " * 20
        with patch.object(S, "helper_fetch_url_content", fetch):
            return await S.tool_deep_research("PostgreSQL 18 release date", False, PROXY)

    async def fact_check(by, calls):
        rows = await S.factcheck_counter_search("postgresql 18 release date", PROXY)
        return "\n".join(r["href"] for r in rows)
    return [("web_search", web_search), ("deep_research", deep_research), ("fact_check", fact_check)]


@pytest.mark.parametrize("name,run", _callers(), ids=[n for n, _ in _callers()])
def test_every_caller_gets_three_circuits_the_onion_and_wikipedia(net, name, run):
    net.brave = RuntimeError("blocked: HTTP 429 from brave_onion")
    net.wiki = WIKI_HITS
    calls = []
    by = {e: _blocked(e) for e in DDGS_ENGINES}
    by["yandex"] = _circuit(2)                                     # refused on two circuits, answers on the third
    with _use_ddgs(_fake_ddgs(by, calls)):
        out = asyncio.run(run(by, calls))
    text = str(getattr(out, "text", out))
    yandex = [p for e, p, _q in calls if e == "yandex"]
    assert len(yandex) == 3 and len(set(yandex)) == 3, name
    assert "brave" in net.kinds() and "wikipedia" in net.kinds(), name
    assert dict(S._ENGINE_WAVES[-1][1])["yandex"] == "win" and len(S._ENGINE_WAVES) == 1, name
    if name == "deep_research":
        fetched = [u for k, u, _ in calls if k == "fetch"]
        assert ROWS[0]["href"] in fetched and "https://en.wikipedia.org/wiki/PostgreSQL" in fetched
    else:
        assert ROWS[0]["href"] in text and "https://en.wikipedia.org/wiki/PostgreSQL" in text, name


@pytest.mark.parametrize("name,run", _callers(), ids=[n for n, _ in _callers()])
def test_every_caller_takes_a_late_onion_win(net, name, run):
    net.brave = net.page
    net.brave_delay = 0.4
    calls = []
    with _use_ddgs(_fake_ddgs({e: _blocked(e) for e in DDGS_ENGINES}, calls)):
        out = asyncio.run(run({}, calls))
    text = str(getattr(out, "text", out)) + " ".join(u for k, u, _ in calls if k == "fetch")
    assert BRAVE_ROWS[0]["href"] in text, name
    assert dict(S._ENGINE_WAVES[-1][1])["brave_onion"] == "win" and len(S._ENGINE_WAVES) == 1, name


def test_deep_research_reads_wikipedia_when_no_engine_wins_and_says_so(net, monkeypatch):
    net.brave = RuntimeError("blocked: HTTP 429 from brave_onion")
    net.wiki = WIKI_HITS
    fetched = []

    async def fetch(url, **kw):
        fetched.append(url)
        return "PostgreSQL 18 release date: the encyclopedia's text. " * 20

    async def _no_sleep(_s):
        return None
    with _use_ddgs(_fake_ddgs({e: _blocked(e) for e in DDGS_ENGINES}, [])), \
            patch.object(S, "helper_fetch_url_content", fetch), patch.object(S.asyncio, "sleep", _no_sleep):
        out = asyncio.run(S.tool_deep_research("postgresql 18 release date", False, PROXY))
    text = str(getattr(out, "text", out))
    assert sorted(fetched) == ["https://en.wikipedia.org/wiki/PostgreSQL",
                               "https://en.wikipedia.org/wiki/PostgreSQL_release_history_(18)"]
    assert "only Wikipedia articles" in text and "CRITICAL ERROR" not in text
    assert len(S._ENGINE_WAVES) >= 2                               # the waves were retried first, as before


# ── h. the privacy guard runs before every engine ────────────────────────────────────────────

@pytest.mark.parametrize("tool", ["web_search", "deep_research"])
def test_the_owners_address_reaches_no_engine_of_the_table(net, tmp_path, tool):
    """The guard sits at the tool boundary (`registry._egress_scrubbed`); every engine — ddgs, the Brave
    onion, Wikipedia — gets the query from the wave behind it."""
    from ghost_agent.memory.profile import ProfileMemory
    from ghost_agent.tools.registry import _egress_scrubbed
    pm = ProfileMemory(tmp_path)
    pm.update("root", "name", "Elena")
    pm.update("root", "location", "Athens, Greece")
    pm.update("root", "address", "Kaledonias 83 Kalopetres 10101, Athens, Greece")
    ctx = SimpleNamespace(profile_memory=pm)
    calls = []

    async def fetch(url, **kw):
        return "bakery page text " * 30
    fn = {"web_search": lambda **kw: S.tool_search(anonymous=False, tor_proxy=PROXY, **kw),
          "deep_research": lambda **kw: S.tool_deep_research(kw.pop("query"), False, PROXY, **kw)}[tool]
    by = {e: _answer([{"title": "Bakery near Kalopetres", "href": "https://bakery.example/a",
                       "body": "bakery near Kalopetres, open daily"}]) for e in DDGS_ENGINES}
    with _use_ddgs(_fake_ddgs(by, calls)), patch.object(S, "helper_fetch_url_content", fetch):
        asyncio.run(_egress_scrubbed(tool, fn, ctx)(query="bakery near Kaledonias 83"))
    sent = [q for _e, _p, q in calls] + [u for _k, u, *_ in net.calls]
    kinds = {e for e, _p, _q in calls} | set(net.kinds())
    # every engine of the table was asked, and none of them saw the street
    assert kinds == set(S.ENGINE_TABLE) - {"brave_onion"} | {"brave"}, kinds
    assert set(net.kinds()) == {"brave", "wikipedia"}
    assert sent and not [s for s in sent if "kaledonias" in s.lower() or "83" in s], sent
    assert all("kalopetres" in s.lower() for s in sent)            # the area took its place, in every request


# ── i. cancelling a wave ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("stage", ["page", "links"])
def test_cancelling_a_wave_cancels_every_attempt_and_leaves_no_task(net, stage):
    net.wiki_delay = 60.0
    if stage == "page":
        net.brave, net.brave_delay = net.page, 60.0
    else:
        net.brave = (200, brave_page(_wrapped(4)))
        net.go, net.go_delay = _targets(4), 60.0
    started = []

    def hang(proxy):
        started.append(proxy)
        time.sleep(0.6)                                            # a ddgs thread cannot be stopped; it ends itself
        return []

    async def _main():
        task = asyncio.ensure_future(S._race_search_wave(QUERY, PROXY, 0))
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10 and not (
                len(started) == 6 and "wikipedia" in net.kinds()
                and ("go" if stage == "links" else "brave") in net.kinds()):
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        t0 = time.monotonic()
        me = asyncio.current_task()
        while time.monotonic() - t0 < 10 and [t for t in asyncio.all_tasks() if t is not me and not t.done()]:
            await asyncio.sleep(0.02)
        return [t for t in asyncio.all_tasks() if t is not me and not t.done()]
    with _use_ddgs(_fake_ddgs({e: hang for e in DDGS_ENGINES}, [])):
        left = asyncio.run(_main())
    assert left == []
    assert len(started) == 6                                       # every circuit of every ddgs engine was out
    assert "wikipedia" in net.cancelled
    assert ("go" if stage == "links" else "brave") in net.cancelled
    assert len(S._ENGINE_WAVES) == 0                               # a cancelled wave records no outcome


# ── the transport itself ─────────────────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, status, chunks, headers=None):
        self.status_code, self._chunks, self.headers = status, chunks, dict(headers or {})
        self.quit_now = threading.Event()
        self.closed = False

    async def aiter_content(self):
        for c in self._chunks:
            self.consumed = getattr(self, "consumed", 0) + 1
            yield c

    def iter_content(self, *a, **k):                               # the sync drain is the trap (§4GM)
        raise AssertionError("iter_content on an async response")

    async def aclose(self):
        self.closed = True


def _fake_curl(monkeypatch, resp, seen):
    monkeypatch.delenv("GHOST_SEARCH_NO_NETWORK", raising=False)      # the client below is a fake
    class _Session:
        def __init__(self, **kw):
            seen.append(("session", kw))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            seen.append(("closed", None))
            return False

        async def get(self, url, **kw):
            seen.append(("get", url, kw))
            return resp
    fake = SimpleNamespace(AsyncSession=_Session)
    monkeypatch.setitem(sys.modules, "curl_cffi.requests", fake)
    monkeypatch.setitem(sys.modules, "curl_cffi", SimpleNamespace(requests=fake))


def test_the_transport_is_one_async_session_per_request_through_socks5h(monkeypatch):
    seen = []
    resp = _FakeResp(302, [b"abc", b"", "δεf".encode()], {"location": "https://target.example/x"})
    _fake_curl(monkeypatch, resp, seen)
    out = asyncio.run(_REAL_DIRECT_FETCH("https://x.example/a", "socks5://tag:isolate@127.0.0.1:9050", 7.0,
                                         headers={"Accept-Language": "en"}, impersonate="chrome", follow=False))
    assert out == (302, "abcδεf", "https://target.example/x")
    (_s, skw), (_g, url, gkw), _closed = seen
    assert skw["proxies"] == {"http": "socks5h://tag:isolate@127.0.0.1:9050",
                              "https": "socks5h://tag:isolate@127.0.0.1:9050"}       # Tor resolves the name
    assert skw["impersonate"] == "chrome" and skw["timeout"] == 7.0 and "cookies" not in skw
    assert gkw["stream"] is True and gkw["allow_redirects"] is False and gkw["headers"] == {"Accept-Language": "en"}
    assert resp.closed and resp.quit_now.is_set()                  # aborted + reaped (aclose_curl_response)
    assert [k for k, *_ in seen] == ["session", "get", "closed"]   # a session per request: nothing is kept


def test_the_transport_caps_the_body_and_refuses_an_onion_without_tor(monkeypatch):
    seen = []
    resp = _FakeResp(200, [b"x" * 600] * 10)
    _fake_curl(monkeypatch, resp, seen)
    status, body, _loc = asyncio.run(_REAL_DIRECT_FETCH("https://x.example/a", PROXY, 5.0, max_bytes=1000))
    assert status == 200 and len(body) == 1000
    assert resp.consumed == 2 and resp.closed                      # reading STOPS at the cap, and the transfer is aborted
    seen.clear()
    with pytest.raises(ConnectionError):
        asyncio.run(_REAL_DIRECT_FETCH(f"https://{S.BRAVE_ONION_HOST}/search?q=x", None, 5.0))
    assert seen == []                                              # nothing was dialled


def test_under_mandatory_tor_a_missing_proxy_is_replaced_never_dialled_direct(monkeypatch):
    seen = []
    _fake_curl(monkeypatch, _FakeResp(200, [b"{}"]), seen)
    import ghost_agent.utils.egress_guard as G
    monkeypatch.setattr(G, "resolve_egress_proxy", lambda proxy, url=None: proxy or "socks5://127.0.0.1:9050")
    asyncio.run(_REAL_DIRECT_FETCH("https://en.wikipedia.org/w/api.php?x=1", None, 5.0))
    assert seen[0][1]["proxies"]["https"] == "socks5h://127.0.0.1:9050"


def test_the_suite_cannot_reach_the_transport_by_accident(monkeypatch):
    """conftest: the direct engines are switched off and the transport itself refuses — by a switch the
    function reads, so it holds for both import paths and whatever was imported when (r1)."""
    import src.ghost_agent.tools.search as S2
    seen = []
    assert S.engines_switched_off() == {"brave_onion", "wikipedia"} and os.environ["GHOST_SEARCH_NO_NETWORK"] == "1"
    fake = SimpleNamespace(AsyncSession=lambda **kw: seen.append(kw))
    monkeypatch.setitem(sys.modules, "curl_cffi.requests", fake)
    for mod in (S, S2):
        assert mod._direct_fetch is not None and mod is not (S2 if mod is S else S)
        with pytest.raises(RuntimeError, match="NO_NETWORK"):
            asyncio.run(mod._direct_fetch("https://en.wikipedia.org/w/api.php", PROXY, 1.0))
    assert seen == []
    # …and a whole wave with the engines switched ON still dials nothing
    monkeypatch.setenv("GHOST_SEARCH_ENGINES_OFF", "")
    monkeypatch.setattr(S, "_bench_state", {"waves": 0})
    with _use_ddgs(_fake_ddgs({e: _answer() for e in DDGS_ENGINES}, [])):
        assert asyncio.run(S._race_search_wave(QUERY, PROXY, 0))
    assert seen == []


# ── the durable line and its reader ──────────────────────────────────────────────────────────

def test_the_engines_line_names_every_engine_once_and_carries_no_query(net, caplog):
    import logging
    net.wiki = WIKI_HITS
    with caplog.at_level(logging.INFO):
        _wave({"yandex": _circuit(1)})
    lines = [r.getMessage() for r in caplog.records if "search wave engines" in r.getMessage()]
    assert len(lines) == 1 and "postgresql" not in lines[0].lower()
    names = [t.split("=")[0] for t in lines[0].split("engines: ", 1)[1].split()]
    assert names == ["duckduckgo", "yandex", "brave", "yahoo", "brave_onion", "wikipedia"]   # launch order, once each
    brave_line = [r.getMessage() for r in caplog.records if r.getMessage().startswith("brave_onion page:")]
    assert all("postgresql" not in m.lower() for m in brave_line)


def test_the_win_line_says_which_circuit_won(net):
    logged = []
    with patch.object(S, "pretty_log", lambda title, msg="", **k: logged.append(str(msg))):
        _wave({"yandex": _circuit(2)})
    assert [m for m in logged if "yandex won wave 0" in m and m.endswith("[circuit 3/3]")]
    logged.clear()
    with patch.object(S, "pretty_log", lambda title, msg="", **k: logged.append(str(msg))):
        _wave({"yahoo": _answer()})
    assert [m for m in logged if "yahoo won wave 0" in m and "circuit" not in m]


def test_the_stats_script_tallies_the_new_engine_names(tmp_path):
    import datetime as dt
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "search_engine_stats", Path(__file__).resolve().parents[1] / "scripts" / "search_engine_stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log = tmp_path / "ghost-agent.log"
    log.write_text(
        "2026-11-01 10:00:00 - GhostAgent - INFO - search wave engines — wave 0 | engines: duckduckgo=blocked yandex=win\n"
        "2026-11-20 10:00:00 - GhostAgent - INFO - search wave engines — wave 0 | engines: duckduckgo=blocked "
        "yandex=win yahoo=unfinished brave_onion=unfinished wikipedia=added\n"
        "2026-11-20 10:00:05 - GhostAgent - INFO - search wave engines — wave 1 | engines: duckduckgo=empty "
        "yandex=timeout brave_onion=win wikipedia=off-topic\n"
        "2026-11-20 10:00:06 - GhostStream - INFO - [abc +7.0s] search page read — read db.example in 900 ms — 2500 of 9000 chars\n"
        "2026-11-20 10:00:07 - GhostStream - INFO - unrelated line\n", encoding="utf-8")
    stats = mod.tally([str(log)], since=dt.date(2026, 11, 10))
    assert stats["engines"]["duckduckgo"] == {"blocked": 1, "empty": 1}
    assert stats["engines"]["yandex"] == {"win": 1, "timeout": 1}
    assert stats["engines"]["brave_onion"] == {"unfinished": 1, "win": 1}
    assert stats["engines"]["wikipedia"] == {"added": 1, "off-topic": 1}
    assert stats["page_reads"] == {"read": 1} and stats["waves"] == 2
    assert "brave_onion" in mod.render(stats) and "added" in mod.render(stats)


# ── review r1 ────────────────────────────────────────────────────────────────────────────────
# M1 — a supplement row must be ABOUT the query's subject (§4JO's rule, restored)

def _art(title, body="…"):
    return {"title": title, "body": body, "href": "https://el.wikipedia.org/wiki/X", "engine": "wikipedia"}


@pytest.mark.parametrize("query,title,body,kept", [
    # the reviewer's table: each of these rows was kept, and for Greek it LED the batch
    ("Τζάνειο Νοσοκομείο", "Πλωτό νοσοκομείο", "Ένα πλωτό νοσοκομείο είναι πλοίο που λειτουργεί ως νοσοκομείο", False),
    ("Δημήτρης Κουφοντίνας", "Δημήτρης Μητροπάνος", "Ο Δημήτρης Μητροπάνος ήταν Έλληνας τραγουδιστής", False),
    ("Νίκος Καζαντζάκης Βιογραφία", "Νίκος Γκάλης", "Ο Νίκος Γκάλης, βιογραφία του καλαθοσφαιριστή", False),
    ("Καζαντζάκης", "Γιώργος Παπαδόπουλος", "Ο Γιώργος Παπαδόπουλος και ο Καζαντζάκης", False),
    ("Αλκιβιάδου 154 Πειραιάς ιστορία", "Ιστορία του Πειραιά", "Η ιστορία του Πειραιά από την αρχαιότητα", False),
    ("python asyncio gather", "Python (genus)", "Python is a genus of constricting snakes", False),
    ("canberra capital australia", "Capital (economics)", "In economics, capital goods; Australia is an example", False),
    # …and the articles those searches are about are kept
    ("Τζάνειο Νοσοκομείο", "Τζάνειο Γενικό Νοσοκομείο Πειραιά", "Το Τζάνειο είναι νοσοκομείο του Πειραιά", True),
    ("Δημήτρης Κουφοντίνας", "Δημήτρης Κουφοντίνας", "Ο Δημήτρης Κουφοντίνας είναι Έλληνας", True),
    ("Νίκος Καζαντζάκης Βιογραφία", "Νίκος Καζαντζάκης", "Ο Νίκος Καζαντζάκης ήταν Έλληνας συγγραφέας.", True),
    ("Καζαντζάκης", "Νίκος Καζαντζάκης", "Έλληνας συγγραφέας", True),
    ("canberra capital australia", "Canberra", "Canberra is the capital city of Australia", True),
    ("python asyncio gather", "Asyncio (Python)", "asyncio is a Python library; gather runs awaitables", True),
    # the cases of the deleted tests/test_search_wiki_supplement_2026_09_21.py
    ("Τέμπη σιδηροδρομικό δυστύχημα 2023", "Σιδηροδρομικό δυστύχημα στα Τέμπη", "Το δυστύχημα των Τεμπών ήταν σιδηροδρομικό", True),
    ("Αλκιβιάδου 154 Πειραιάς ιστορία", "Γερμανική εισβολή στην Ελλάδα", "Η εισβολή…", False),
    ("Δημήτρης Κουφοντίνας", "Κατάλογος της δράσης της 17Ν", "Κατάλογος επιθέσεων της οργάνωσης…", False),
    # a Greek case ending matches on the stem; a one-word query must be IN the title
    ("Οδύσσεια", "Ραψωδίες της Οδύσσειας", "οι ραψωδίες", True),
    ("Οδύσσεια", "Όμηρος", "Ο Όμηρος έγραψε την Οδύσσεια", False),
    ("weather", "Weather", "Weather is the state of the atmosphere", True),
    ("weather", "Climate", "Climate is the long-term weather pattern", False),
])
def test_a_supplement_row_must_be_about_the_querys_subject(query, title, body, kept):
    assert S.wiki_article_on_topic(query, _art(title, body)) is kept
    hits = [_art(title, body)]
    assert bool(S.wikipedia_rows(query, hits)) is kept


def test_the_subject_rule_beats_the_any_token_rule():
    tz = "Τζάνειο Νοσοκομείο"
    # the §4IL any-token rule WOULD have passed the junk (every Greek word is distinctive)
    assert S.result_on_topic({"title": "Πλωτό νοσοκομείο", "body": "…", "href": ""}, S.distinctive_tokens(tz)) is True
    assert S.wiki_article_on_topic(tz, _art("Πλωτό νοσοκομείο")) is False
    # …and so would the batch's relevance rule alone, which is all the supplement had before r1
    junk = [_art("Πλωτό νοσοκομείο", "Ένα πλωτό νοσοκομείο είναι πλοίο που λειτουργεί ως νοσοκομείο")]
    assert S.prune_off_topic(tz, junk) == (junk, False) and S.wikipedia_rows(tz, junk) == []
    weak = [_art("Python (genus)", "Python is a genus of constricting snakes")]
    assert S.prune_off_topic("python asyncio gather", weak) == (weak, False) and S.wikipedia_rows("python asyncio gather", weak) == []


def test_the_greek_subject_comes_from_the_query_as_typed():
    art = _art("Νίκος Καζαντζάκης", "Ο Νίκος Καζαντζάκης πέθανε το 1957.")
    assert S.wiki_article_on_topic("πότε πέθανε ο καζαντζάκης", art, "Πότε πέθανε ο Καζαντζάκης") is True
    # an article that has the verb but not the name: as typed, the name is the subject — dropped
    other = _art("Ο Σεφέρης πέθανε", "Ο Γιώργος Σεφέρης πέθανε το 1971.")
    assert S.wiki_article_on_topic("πότε πέθανε ο καζαντζάκης", other, "Πότε πέθανε ο Καζαντζάκης") is False
    # lower-cased with nothing typed to read (deep research in anonymous mode), the first distinctive
    # word — the verb — is the "subject": §4JO then KEPT the wrong article on it; with the title-and-
    # coverage rule neither article is kept (nothing wrong is placed first; the right one is missed)
    assert S.wiki_article_on_topic("πότε πέθανε ο καζαντζάκης", other) is False
    assert S.wiki_article_on_topic("πότε πέθανε ο καζαντζάκης", art) is False
    assert S._wiki_subject_tokens("x", "Πότε πέθανε ο Καζαντζάκης") == ["καζαντζακησ"]     # not the question word
    assert S._wiki_subject_tokens("Αλκιβιάδου 154 Πειραιάς ιστορία") == ["αλκιβιαδου", "πειραιασ"]


def test_the_wiki_subject_is_the_name_not_a_generic_word():
    """§4UN-web r1 MINOR 6: "Νίκος Καζαντζάκης Βιογραφία" required "βιογραφια"."""
    art = _art("Νίκος Καζαντζάκης", "Ο Νίκος Καζαντζάκης ήταν Έλληνας συγγραφέας.")
    assert S.wiki_article_on_topic("νίκος καζαντζάκης βιογραφία", art, "Νίκος Καζαντζάκης Βιογραφία") is True
    assert S.wiki_article_on_topic("Νίκος Καζαντζάκης Βιογραφία", art) is True
    assert S._wiki_subject_tokens("history") == []                                # nothing but a generic word: no subject
    assert S.wiki_article_on_topic("history", _art("History")) is False


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


def test_web_search_reads_the_name_as_typed_in_the_wave_and_in_the_listing(net, monkeypatch):
    """Anonymous mode lower-cases the query; the sanitiser then drops the quotes. The NAME's capitals
    still decide which article is about the question — in the wave and when the batch is listed."""
    net.wiki = _hits(("Ο Σεφέρης πέθανε", "Πότε πέθανε ο Σεφέρης και όχι ο Καζαντζάκης"),
                     ("Νίκος Καζαντζάκης", "Ο Νίκος Καζαντζάκης πέθανε το 1957."))
    web = [{"title": "Πότε πέθανε ο Καζαντζάκης", "href": "https://nea.example/k", "body": "Ο Καζαντζάκης πέθανε το 1957"}]
    monkeypatch.setattr("ghost_agent.utils.stylometry.scrub_query", lambda q: q.lower())
    with _use_ddgs(_fake_ddgs({e: _answer(web) for e in DDGS_ENGINES}, [])):
        out = asyncio.run(S.tool_search('Πότε πέθανε ο "Καζαντζάκης"', anonymous=True, tor_proxy=PROXY))
    assert parse_qs(urlparse(net.urls("wikipedia")[0]).query)["srsearch"] == ["πότε πέθανε ο καζαντζάκης"]
    assert out.index("### 1. Νίκος Καζαντζάκης\n") < out.index("nea.example") and "Σεφέρης" not in out
    assert S._QUERY_AS_TYPED.get() is None


@pytest.mark.parametrize("caller", ["web_search", "deep_research", "fact_check"])
def test_every_caller_drops_an_article_that_is_not_about_the_subject(net, monkeypatch, caller):
    """The scrubbed (lower-cased) query reaches the engines; web_search still reads the NAME as typed."""
    net.wiki = _hits(("Δημήτρης Μητροπάνος", "Ο Δημήτρης Μητροπάνος ήταν Έλληνας τραγουδιστής, όχι ο Κουφοντίνας"),
                     ("Δημήτρης Κουφοντίνας", "Ο Δημήτρης Κουφοντίνας είναι Έλληνας"))
    web = [{"title": "Δημήτρης Κουφοντίνας — ειδήσεις", "href": "https://nea.example/k",
            "body": "Ο Δημήτρης Κουφοντίνας στις ειδήσεις"}]
    monkeypatch.setattr("ghost_agent.utils.stylometry.scrub_query", lambda q: q.lower())
    fetched = []

    async def fetch(url, **kw):
        fetched.append(url)
        return "Ο Δημήτρης Κουφοντίνας: κείμενο σελίδας. " * 20
    with _use_ddgs(_fake_ddgs({e: _answer(web) for e in DDGS_ENGINES}, [])), patch.object(S, "helper_fetch_url_content", fetch):
        if caller == "web_search":
            out = asyncio.run(S.tool_search("Δημήτρης Κουφοντίνας", anonymous=True, tor_proxy=PROXY))
        elif caller == "deep_research":
            out = asyncio.run(S.tool_deep_research("Δημήτρης Κουφοντίνας", False, PROXY))
        else:
            out = "\n".join(r["title"] for r in asyncio.run(S.factcheck_counter_search("Δημήτρης Κουφοντίνας", PROXY)))
    text = str(getattr(out, "text", out)) + " ".join(fetched)
    assert "%CE%9C%CE%B7%CF%84%CF%81%CE%BF%CF%80%CE%AC%CE%BD%CE%BF%CF%82" not in text and "Μητροπάνος" not in text
    if caller == "web_search":
        assert text.index("### 1. Δημήτρης Κουφοντίνας\n") < text.index("nea.example")     # the right article leads


# M2 — fetched bytes are hostile: linear patterns, and never parsed on the event loop

_HOSTILE = [
    ("refresh tags", lambda: S._brave_link_target(200, "", "<meta http-equiv=refresh " * 4200)),
    ("anchors", lambda: S._brave_link_target(200, "", "<a " * 33_000)),
    ("location", lambda: S._brave_link_target(200, "", 'location="https://' * 5500)),
    ("many urls", lambda: S._brave_link_target(200, "", "https://a.example/x " * 5000)),
    ("challenge anchors", lambda: S.engine_page_is_challenge("<a " * 660_000, "q")),
    ("challenge brackets", lambda: S.engine_page_is_challenge("<" * 2_000_000, "q")),
    ("challenge scripts", lambda: S.engine_page_is_challenge("<script " * 250_000, "q")),
    ("challenge forms", lambda: S.engine_page_is_challenge("<form " * 330_000 + "<meta http-equiv=refresh " * 80_000, "q")),
    ("empty sample", lambda: S.page_structure("<a " * 85_000, "q")),
    ("empty sample attrs", lambda: S.page_structure("<a " + 'class="' * 36_000 + ">", "q")),
    ("snippet", lambda: S._strip_markup("<" * 190_000)),
    ("wikipedia hits", lambda: S.parse_wikipedia_hits({"query": {"search": [
        {"title": "<" * 20_000, "snippet": "<" * 150_000}] * 8}}, "en")),
    ("results page", lambda: S.parse_brave_onion("<a " * 660_000)),
    ("results rows", lambda: S._read_brave_page(brave_page(BRAVE_ROWS * 300), "q")),
]


@pytest.mark.parametrize("name,fn", _HOSTILE, ids=[n for n, _ in _HOSTILE])
def test_a_hostile_page_at_the_body_cap_is_read_in_linear_time(name, fn):
    S._EMPTY_SAMPLED.discard("brave_onion")
    t0 = time.monotonic()
    fn()
    assert time.monotonic() - t0 < 10.0, name              # each took 9 s to over 30 s (r1); now well under 1 s


def test_the_linear_classifiers_still_classify():
    assert S.engine_page_is_challenge('<html><form method="post" action="https://consent.yahoo.com/v2/collect"></form></html>')
    assert S.engine_page_is_challenge('<meta content="0;url=https://consent.yahoo.com/x" http-equiv="refresh">')
    assert S.engine_page_is_challenge("<script>window.location.href = 'https://consent.yahoo.com/x'</script>")
    assert not S.engine_page_is_challenge('<html><a href="https://consent.yahoo.com/privacy">Privacy</a>' + "<p>results text</p>" * 400)
    assert S.engine_page_is_challenge("<html><script>var a = 'x';</script><p>Please verify you are human</p></html>")
    assert not S.engine_page_is_challenge("<html><script>var captcha = 1;</script><style>.captcha{}</style><p>No results.</p></html>")
    assert S.page_structure('<div class="a b" id="x"><p>hello</p><script>if (a<b) {}</script></div>') == \
        '<div class:a class:b id:x>\n<p>\n#text 5\n</p>\n<script>\n</script>\n</div>'
    assert S._strip_markup('a <span class="searchmatch">b</span> &lt;c&gt; <<d') == "a b <c> <<d"


@pytest.mark.parametrize("step", ["_read_brave_page", "_brave_link_target", "_read_wikipedia_body"])
def test_the_event_loop_keeps_ticking_while_fetched_bytes_are_parsed(net, monkeypatch, step):
    """A parse that takes half a second (a hostile page, a slow machine) must not stop the loop: the
    ticker task keeps running while it does."""
    real = getattr(S, step)

    def _slow(*a, **k):
        time.sleep(0.6)
        return real(*a, **k)
    monkeypatch.setattr(S, step, _slow)
    net.brave = (200, brave_page(_wrapped(3)))
    net.go = _targets(3)
    net.wiki = WIKI_HITS
    ticks = []

    async def _main():
        async def _ticker():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.02)
        tk = asyncio.ensure_future(_ticker())
        try:
            if step == "_read_wikipedia_body":
                return await S._wikipedia_search(QUERY, PROXY, 5.0)
            return await S._brave_onion_search(QUERY, PROXY, 5.0, 20)
        finally:
            tk.cancel()
    assert asyncio.run(_main())
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert len(ticks) >= 10 and max(gaps) < 0.4, (len(ticks), max(gaps))


# the minors

@pytest.mark.parametrize("href", ["/go/v1/../../settings", "/go/v2/abc", "/go/v1/abc?x=1", "/go/v1/", "/go/v1/a/b",
                                  "/go/v1/abc#f", "//evil.example/go/v1/abc", "/search?q=x"])
def test_only_an_exact_wrapped_link_path_is_ever_asked(net, href):
    page = brave_page([{"title": "PostgreSQL 18", "href": href, "body": "release date"}] + _direct(3))
    assert [r["href"] for r in S.parse_brave_onion(page)] == [f"https://direct{i}.example/pg18" for i in range(3)]
    net.brave = (200, page)
    _brave(net)
    assert net.kinds() == ["brave"]


def test_the_results_page_is_not_followed_through_a_redirect(net):
    net.brave = (302, "")
    with pytest.raises(RuntimeError, match="blocked: HTTP 302"):
        _brave(net)
    assert net.calls[0][5] is False                                # asked without following


@pytest.mark.parametrize("proxy,ok", [("socks5://127.0.0.1:9050", True), ("socks5h://u:p@127.0.0.1:9050", True),
                                      ("SOCKS5://127.0.0.1:9050", True), ("Socks5H://127.0.0.1:9050", True),
                                      ("http://127.0.0.1:8080", False), ("https://proxy.example:443", False),
                                      ("socks4://127.0.0.1:9050", False), ("127.0.0.1:9050", False),
                                      ("socks5h//127.0.0.1", False), ("xsocks5://127.0.0.1:9050", False)])
def test_only_a_socks5_proxy_is_ever_used(monkeypatch, proxy, ok):
    seen = []
    _fake_curl(monkeypatch, _FakeResp(200, [b"{}"]), seen)
    if ok:
        asyncio.run(_REAL_DIRECT_FETCH("https://en.wikipedia.org/w/api.php", proxy, 5.0))
        assert seen[0][1]["proxies"]["https"] == "socks5h://" + proxy.split("://", 1)[1]
    else:
        with pytest.raises(ConnectionError):
            asyncio.run(_REAL_DIRECT_FETCH("https://en.wikipedia.org/w/api.php", proxy, 5.0))
        assert seen == []                                          # refused before anything is dialled


@pytest.mark.parametrize("which", ["brave", "wikipedia"])
def test_an_attempt_that_cancels_itself_does_not_sink_the_wave(net, which):
    """r1: a CancelledError raised INSIDE one attempt is that attempt's failure; only the wave's own
    cancellation propagates (`test_cancelling_a_wave_…`)."""
    if which == "brave":
        net.brave = asyncio.CancelledError()
    else:
        net.wiki = asyncio.CancelledError()
    rows, _calls, _took, tokens = _wave({"yandex": _answer(after=0.3)})
    assert [r["href"] for r in rows] == [r["href"] for r in ROWS]
    assert tokens[0]["yandex"] == "win"
    assert tokens[0]["brave_onion" if which == "brave" else "wikipedia"] in ("error", "blocked")
    if which == "brave":
        assert tokens[0]["brave_onion"] == "error"
    # and a wave nobody wins ends normally too
    rows, _calls, _took, tokens = _wave({})
    assert rows == []


@pytest.mark.parametrize("url,ok", [
    ("https://target.example/a?x=1", True), ("http://sub.target.example:8080/a", True),
    ("https://brave.com@evil.example/", False), ("https://user:pw@target.example/", False),
    ("https://target.example/a b", False), ("https://target.example/a\nb", False), ("https://target.example/\x00", False),
    ("https://127.0.0.1/admin", False), ("https://10.0.0.5/", False), ("https://192.168.1.1/", False),
    ("https://localhost/x", False), ("https://nas.local/x", False), ("https://box.tail1234.ts.net/x", False),
    ("https://100.101.102.103/", False), ("http://abcdefghijklmnop.onion/", False), ("https://intranet/x", False),
    ("https://[::1]/x", False), ("https://search.brave.com/x", False), ("ftp://target.example/a", False),
    ("javascript:alert(1)", False), ("https://target.example/" + "a" * 3000, False), ("", False),
    # r2: one spelling of a host — numeric forms a resolver reads as loopback, compatibility characters
    ("https://127.1/x", False), ("https://0177.0.0.1/x", False), ("https://0x7f.0.0.1/x", False),
    ("https://2130706433/x", False), ("https://①②⑦.0.0.1/x", False), ("https://１２７.０.０.１/x", False),
    ("https://ｅｖｉｌ.example/x", False), ("https://1.2.3/x", False), ("https://256.1.1.1/x", False),
    ("https://93.184.216.34/x", True), ("https://münchen.example/x", True), ("https://3com.example/x", True),
])
def test_an_address_is_taken_only_when_a_reader_could_open_it(url, ok):
    """The predicate the Sources list uses (`is_public_source_url`), plus: one line, no userinfo."""
    assert (S._brave_page_address(url) == url) is ok
    # …whichever way the interstitial names it
    assert (S._brave_link_target(200, "", go_page(url.replace("&", "&amp;"))) == url) is (ok and '"' not in url)
    assert (S._brave_link_target(302, url, "") == url) is ok
    if url and "\n" not in url and " " not in url and "\x00" not in url:
        only = f'<html><svg xmlns="http://www.w3.org/2000/svg"></svg><p>{url}</p></html>'
        assert (S._brave_link_target(200, "", only) == url) is ok


def test_a_page_that_arrived_but_ran_out_of_time_is_unfinished_not_an_error(net):
    net.brave = (200, brave_page(_wrapped(8)))
    net.go, net.go_delay = _targets(8), 20.0
    with pytest.raises(S.EngineUnfinished):
        _brave(net, timeout=0.3)
    # no time left at all when the page comes in: no link is asked
    net.calls.clear()
    net.brave_delay = 0.4
    with pytest.raises(S.EngineUnfinished):
        _brave(net, timeout=0.3)
    assert "go" not in net.kinds()
    # the wave books it "unfinished": neither a refusal (the bench) nor an answer
    net.brave_delay = 0.0
    stats = {}
    with patch.dict(S.ENGINE_TABLE, {"brave_onion": dataclasses.replace(S.ENGINE_TABLE["brave_onion"], timeout=0.3)}):
        _rows, _calls, _took, tokens = _wave({}, stats=stats)
    assert tokens[0]["brave_onion"] == "unfinished" and stats["cats"] == {"blocked": 4}
    assert "brave_onion" not in S._bench_recent()
    # links that all ANSWERED and named nothing: that is an error
    net.go, net.go_delay = {}, 0.0
    with pytest.raises(RuntimeError) as e:
        _brave(net)
    assert not isinstance(e.value, S.EngineUnfinished)


def test_two_links_to_one_page_are_one_row(net):
    net.brave = (200, brave_page(_wrapped(4) + _direct(1)))
    net.go = _targets(4)
    net.go["/go/v1/token1"] = (200, "", go_page("https://target0.example/pg?x=1&amp;y=0#section"))
    net.go["/go/v1/token3"] = (200, "", go_page("https://direct0.example/pg18/"))
    rows = _brave(net)
    assert [r["href"] for r in rows] == ["https://target0.example/pg?x=1&y=0", "https://target2.example/pg?x=1&y=2",
                                         "https://direct0.example/pg18/"]


def test_one_article_is_one_row_whatever_its_host_or_query_string(net):
    k = S._row_key
    assert k({"href": "https://en.m.wikipedia.org/wiki/PostgreSQL?wprov=sfla1"}) == k({"href": "https://en.wikipedia.org/wiki/PostgreSQL"})
    assert k({"href": "http://www.EN.wikipedia.org/wiki/PostgreSQL/#History"}) == k({"href": "https://en.wikipedia.org/wiki/PostgreSQL"})
    assert k({"href": "https://el.wikipedia.org/wiki/%CE%9F%CE%B4%CF%8D%CF%83%CF%83%CE%B5%CE%B9%CE%B1"}) == k({"href": "https://el.wikipedia.org/wiki/Οδύσσεια"})
    assert k({"href": "https://shop.example/item?id=1"}) != k({"href": "https://shop.example/item?id=2"})   # elsewhere a query string IS the page
    assert k({"href": "https://en.wikipedia.org/wiki/A"}) != k({"href": "https://el.wikipedia.org/wiki/A"})
    net.wiki = WIKI_HITS
    winner = ROWS + [{"title": "PostgreSQL - Wikipedia", "href": "https://en.m.wikipedia.org/wiki/PostgreSQL?useskin=x",
                      "body": "PostgreSQL 18 release date and history."}]
    rows, _calls, _took, _tokens = _wave({"yandex": _answer(winner)})
    assert [r["title"] for r in rows if r.get("engine") == "wikipedia"] == [
        "PostgreSQL release history (18)", "Release date of PostgreSQL 18"]


def test_the_language_is_the_majority_of_the_querys_letters():
    assert S.wikipedia_lang("Ποιος έγραψε την Οδύσσεια") == "el" and S.wikipedia_first("Ποιος έγραψε την Οδύσσεια")
    assert S.wikipedia_lang("docker compose healthcheck α") == "en" and not S.wikipedia_first("docker compose healthcheck α")
    assert S.wikipedia_lang("Οδύσσεια Homer") == "el" and S.wikipedia_lang("Odyssey Όμηρος") == "en"
    assert S.wikipedia_lang("12345 !!") == "en" and S.wikipedia_lang("") == "en"


def test_the_wikipedia_wait_never_takes_a_wave_past_its_deadline(net, monkeypatch):
    monkeypatch.setattr(S, "WIKI_WAIT_S", 60.0)
    monkeypatch.setattr(S, "_race_wave_deadline", lambda: 1.0)
    net.wiki_delay = 120.0

    def hang(proxy):
        time.sleep(1.5)
        return []
    rows, _calls, took, tokens = _wave({e: hang for e in DDGS_ENGINES})
    assert rows == [] and took < 10.0, took                        # the deadline, not deadline + the wait
    assert tokens[0]["wikipedia"] == "unfinished"
    # a win just before the deadline waits only what is left of it
    rows, _calls, took, _tokens = _wave({e: hang for e in DDGS_ENGINES if e != "yandex"} | {"yandex": _answer(after=0.7)})
    assert rows and took < 10.0, took


# a probe is evidence for the BENCH only (operator, after r1)

def _search_with_duckduckgo(net, monkeypatch, newest_ago, duck):
    """One whole web_search with duckduckgo benched (its last refusal ``newest_ago`` seconds old):
    (text the model gets, waves run, first wave's tokens, first wave's stats)."""
    _restart(monkeypatch)
    S._SEARCH_CACHE.clear()
    monkeypatch.setenv("GHOST_SEARCH_ENGINES_OFF", "wikipedia")
    _write_bench({"duckduckgo": _refusals(newest_ago=newest_ago)})
    waves, lines, stats0 = [], [], {}
    real, real_log = S._race_search_wave, S._log_wave_engines

    async def _counted(query, proxy, wave, **kw):
        waves.append((wave, query))
        rows = await real(query, proxy, wave, **kw)
        if wave == 0:
            stats0.update({k: (dict(v) if isinstance(v, dict) else v) for k, v in (kw.get("stats") or {}).items()})
        return rows

    def _log(w, token):
        lines.append(dict(t.split("=") for t in token.split("engines: ", 1)[1].split()))
        real_log(w, token)

    async def _no_sleep(_s):
        return None
    by = {"duckduckgo": duck, "yandex": lambda p: [], "yahoo": lambda p: [], "brave": _blocked("brave")}
    monkeypatch.setattr(S, "_race_search_wave", _counted)
    with _use_ddgs(_fake_ddgs(by, [])), patch.object(S, "_log_wave_engines", _log), \
            patch.object(S.asyncio, "sleep", _no_sleep), patch.object(S, "_proxy_reachable", lambda *a, **k: True):
        out = asyncio.run(S.tool_search_ddgs("zxqvjk plorfx", PROXY))
    monkeypatch.setattr(S, "_race_search_wave", real)
    return str(getattr(out, "text", out)), waves, lines[0], stats0


def test_a_probes_refusal_changes_no_retry_decision_and_no_text_the_model_reads(net, monkeypatch):
    """yandex and yahoo answer empty; brave and the Brave onion refuse: 2 refused, 2 empty — the query is
    reworded, not retried. A probed duckduckgo refusing too would make it 3 against 2 (a transport
    retry, and "3 engine answer(s) were refusals" in the text): the wave must read as if it sat out."""
    resting = _search_with_duckduckgo(net, monkeypatch, 60.0, _blocked("duckduckgo"))
    probing = _search_with_duckduckgo(net, monkeypatch, 3 * 3600.0, _blocked("duckduckgo"))
    assert "duckduckgo" not in resting[2] and probing[2]["duckduckgo"] == "probe-blocked"   # raced, and said so
    assert probing[0] == resting[0] and "4 engine answer(s) were refusals" in probing[0]   # the same text, word for word (not 5)
    assert probing[1] == resting[1]                                               # the same waves
    assert probing[3]["cats"] == resting[3]["cats"] == {"empty": 2, "blocked": 2}
    assert "duckduckgo" not in probing[3].get("engines", {})
    # …booked for the bench (its rest is renewed) and for nothing else
    assert "duckduckgo" not in S.raced_engines(PROXY) and S.probed_engines() == frozenset()
    assert "duckduckgo" not in S.engine_health()["engines"]
    # an EMPTY answer of the probe is not the wave's either (it would tip the count the other way)
    answered = _search_with_duckduckgo(net, monkeypatch, 3 * 3600.0, lambda p: [])
    assert answered[2]["duckduckgo"] == "probe-empty" and answered[3]["cats"] == {"empty": 2, "blocked": 2}
    assert "duckduckgo" not in answered[3].get("engines", {})
    # …but it ANSWERED: from the next wave on it is an engine like the others again (and is counted)
    assert "duckduckgo" in S.raced_engines(PROXY) and S.probed_engines() == frozenset()


def test_a_probe_that_wins_is_a_win_like_any(net, monkeypatch):
    out, waves, tokens, stats = _search_with_duckduckgo(
        net, monkeypatch, 3 * 3600.0, _answer([{"title": "zxqvjk plorfx explained", "href": "https://duck.example/a",
                                                "body": "zxqvjk plorfx, explained"}]))
    assert "https://duck.example/a" in out and [w for w, _q in waves] == [0]
    assert tokens["duckduckgo"] == "win" and stats["engines"]["duckduckgo"] == {"win": 1}
    assert S.engine_health()["engines"]["duckduckgo"] == {"win": 1} and S.engine_health()["won"] == 1
    assert "duckduckgo" in S.raced_engines(PROXY) and S.probed_engines() == frozenset()


def test_a_probes_low_batch_is_still_the_batch_when_it_is_all_there_is(net, monkeypatch):
    _restart(monkeypatch)
    _write_bench({"duckduckgo": _refusals(newest_ago=3 * 3600.0)})
    stats = {}
    with patch.object(S, "LOW_RELEVANCE_GRACE_S", 0.2):
        rows, _calls, _took, tokens = _wave({"duckduckgo": _answer(LOW)}, query=LOW_QUERY, stats=stats)
    assert [r["href"] for r in rows] == ["https://cloud.example/pricing"]
    assert tokens[0]["duckduckgo"] == "low-relevance" and stats["cats"].get("low-relevance") == 1


def test_the_shutdown_hook_gets_the_last_bench_write_to_disk_within_its_bound(net, monkeypatch):
    """Minor 13: the hook the agent's shutdown calls (`main.flush_search_bench`), called directly."""
    from ghost_agent import main as M
    real = S._bench_write

    def _slow_disk(path, snapshot):
        time.sleep(0.4)
        real(path, snapshot)
    monkeypatch.setattr(S, "_bench_write", _slow_disk)
    S._record_wave_engines(_token(yandex="win", duckduckgo="blocked"), now=time.time())
    assert not os.path.exists(S.bench_file())                      # still queued when the shutdown starts
    t0 = time.monotonic()
    assert asyncio.run(M.flush_search_bench()) is True
    assert time.monotonic() - t0 < 10.0 and os.path.exists(S.bench_file())
    assert set(json.load(open(S.bench_file(), encoding="utf-8"))["engines"]) == {"yandex", "duckduckgo"}
    # a disk that never answers holds the shutdown for the bound, no longer — and never raises into it
    gate = threading.Event()
    monkeypatch.setattr(S, "_bench_write", lambda path, snapshot: gate.wait(30))
    S._record_wave_engines(_token(yandex="win"), now=time.time())
    try:
        t0 = time.monotonic()
        assert asyncio.run(M.flush_search_bench(0.3)) is False
        assert time.monotonic() - t0 < 10.0
        assert M.SEARCH_BENCH_FLUSH_S == 3.0
    finally:
        gate.set()
    with patch.object(S, "bench_flush", lambda *a: (_ for _ in ()).throw(OSError("disk gone"))):
        assert asyncio.run(M.flush_search_bench()) is False


# ── review r2 ────────────────────────────────────────────────────────────────────────────────

_KZ = "Ο Νίκος Καζαντζάκης (1883–1957) ήταν Έλληνας μυθιστοριογράφος, ποιητής και φιλόσοφος"
#: the reviewer's table: (query, article title, snippet, is it the RIGHT article?, is it kept?)
_SUBJECT_TABLE = [
    ('Νίκος Καζαντζάκης', 'Νίκος Καζαντζάκης', _KZ, True, True),
    ('νίκος καζαντζάκης', 'Νίκος Καζαντζάκης', _KZ, True, True),
    ('Νικος Καζαντζακης', 'Νίκος Καζαντζάκης', _KZ, True, True),
    ('ΝΙΚΟΣ ΚΑΖΑΝΤΖΑΚΗΣ', 'Νίκος Καζαντζάκης', _KZ, True, True),
    ('Καζαντζάκης', 'Νίκος Καζαντζάκης', _KZ, True, True),
    ('έργα του Καζαντζάκη', 'Νίκος Καζαντζάκης', 'Ο Νίκος Καζαντζάκης (1883–1957) ήταν Έλληνας μυθιστοριογράφος, ποιητής και φιλόσοφος έργα', True, True),
    ('βιβλία του Νίκου Καζαντζάκη', 'Νίκος Καζαντζάκης', _KZ, True, True),
    ('Πότε πέθανε ο Καζαντζάκης', 'Νίκος Καζαντζάκης', 'Ο Νίκος Καζαντζάκης (1883–1957) ήταν Έλληνας μυθιστοριογράφος, ποιητής και φιλόσοφος πέθανε το 1957', True, True),
    ('πότε πέθανε ο καζαντζάκης', 'Νίκος Καζαντζάκης', 'Ο Νίκος Καζαντζάκης (1883–1957) ήταν Έλληνας μυθιστοριογράφος, ποιητής και φιλόσοφος πέθανε το 1957', True, False),
    ('Ποιος ήταν ο Όμηρος', 'Όμηρος', 'Ο Όμηρος ήταν αρχαίος Έλληνας επικός ποιητής', True, True),
    ('ποιος ήταν ο όμηρος', 'Όμηρος', 'Ο Όμηρος ήταν αρχαίος Έλληνας επικός ποιητής', True, True),
    ('πληροφορίες για τον Όμηρο', 'Όμηρος', 'Ο Όμηρος ήταν αρχαίος Έλληνας επικός ποιητής', True, True),
    ('Ποιος έγραψε την Οδύσσεια;', 'Οδύσσεια', 'Η Οδύσσεια είναι επικό ποίημα που αποδίδεται στον Όμηρο, ο οποίος έγραψε', True, True),
    ('ποιος έγραψε την οδύσσεια', 'Οδύσσεια', 'Η Οδύσσεια είναι επικό ποίημα που αποδίδεται στον Όμηρο', True, False),
    ('Οδύσσεια Όμηρος', 'Οδύσσεια (Όμηρος)', 'Η Οδύσσεια είναι επικό ποίημα του Ομήρου', True, True),
    ('Οδύσσεια', 'Οδύσσεια (Όμηρος)', 'Η Οδύσσεια είναι επικό ποίημα', True, True),
    ('Οδύσσεια του Ομήρου', 'Οδύσσεια', 'Η Οδύσσεια είναι επικό ποίημα του Ομήρου', True, False),
    ('ΕΛΣΤΑΤ', 'Ελληνική Στατιστική Αρχή', 'Η Ελληνική Στατιστική Αρχή (ΕΛΣΤΑΤ) είναι ανεξάρτητη αρχή', True, False),
    ('ΕΛΣΤΑΤ πληθωρισμός', 'Ελληνική Στατιστική Αρχή', 'Η Ελληνική Στατιστική Αρχή (ΕΛΣΤΑΤ) είναι ανεξάρτητη αρχή', True, False),
    ('Τζάνειο Νοσοκομείο', 'Τζάνειο Γενικό Νοσοκομείο Πειραιά', 'Το Τζάνειο είναι νοσοκομείο', True, True),
    ('πρωθυπουργός της Ελλάδας', 'Πρωθυπουργός της Ελλάδας', 'Ο πρωθυπουργός της Ελλάδας είναι ο αρχηγός της κυβέρνησης', True, True),
    ('Πρωτεύουσα της Αυστραλίας', 'Καμπέρα', 'Η Καμπέρα είναι η πρωτεύουσα της Αυστραλίας', True, False),
    ('Μάχη του Μαραθώνα', 'Μάχη του Μαραθώνα', 'Η μάχη του Μαραθώνα (490 π.Χ.)', True, True),
    ('Μάχη Μαραθώνα 490 π.Χ.', 'Μάχη του Μαραθώνα', 'Η μάχη του Μαραθώνα (490 π.Χ.)', True, True),
    ('Ταΐγετος ύψος', 'Ταΰγετος', 'Ο Ταΰγετος είναι οροσειρά, ύψος 2.407 μ.', True, False),
    ('Kazantzakis', 'Nikos Kazantzakis', 'Nikos Kazantzakis was a Greek writer', True, True),
    ('Nikos Kazantzakis books', 'Nikos Kazantzakis', 'Nikos Kazantzakis was a Greek writer; his books include Zorba', True, True),
    ('J. R. R. Tolkien', 'J. R. R. Tolkien', 'John Ronald Reuel Tolkien was an English writer', True, True),
    ('JRR Tolkien', 'J. R. R. Tolkien', 'John Ronald Reuel Tolkien was an English writer', True, True),
    ('tolkien', 'J. R. R. Tolkien', 'John Ronald Reuel Tolkien was an English writer', True, True),
    ('Ludwig van Beethoven', 'Ludwig van Beethoven', 'Ludwig van Beethoven was a German composer', True, True),
    ('Beethoven symphonies', 'Ludwig van Beethoven', 'Ludwig van Beethoven was a German composer of nine symphonies', True, True),
    ('NASA', 'NASA', 'The National Aeronautics and Space Administration (NASA)', True, True),
    ('NASA budget 2025', 'NASA', 'NASA is an independent agency; its budget', True, True),
    ('NASA budget 2025', 'Budget of NASA', 'the budget of NASA', True, True),
    ('who wrote the odyssey', 'Odyssey', 'The Odyssey is one of two major epics attributed to Homer, who wrote', True, False),
    ('Who wrote the Odyssey', 'Odyssey', 'The Odyssey is one of two major epics attributed to Homer', True, False),
    ('who wrote the odyssey', 'Homer', 'Homer was a Greek poet who is credited as the author of the Iliad and the Odyssey', True, False),
    ('capital of Australia', 'Canberra', 'Canberra is the capital city of Australia', True, False),
    ('What is the capital of Australia', 'Canberra', 'Canberra is the capital city of Australia', True, False),
    ("McDonald's revenue", "McDonald's", "McDonald's Corporation is an American fast food company; revenue", True, True),
    ('Beyoncé albums', 'Beyoncé', 'Beyoncé is an American singer; albums', True, True),
    ('Jean-Paul Sartre', 'Jean-Paul Sartre', 'Jean-Paul Sartre was a French philosopher', True, True),
    ('United States president', 'President of the United States', 'The president of the United States is the head of state', True, True),
    ('python asyncio gather', 'Asyncio', 'asyncio is a Python library; gather', True, False),
    ('PostgreSQL 18 release date', 'PostgreSQL', 'PostgreSQL release history: version 18 release date', True, True),
    ('iPhone 17 price', 'iPhone 17', 'The iPhone 17 is a smartphone; price', True, True),
    ('how tall is mount everest', 'Mount Everest', "Mount Everest is Earth's highest mountain, how tall 8,849 m", True, False),
    ('Athens weather', 'Athens', 'Athens is the capital of Greece', False, False),
    ('Νίκος Καζαντζάκης', 'Νίκος Γκάλης', 'Ο Νίκος Γκάλης είναι μπασκετμπολίστας', False, False),
    ('Νίκος Καζαντζάκης', 'Διεθνής Αερολιμένας Ηρακλείου «Νίκος Καζαντζάκης»', 'Ο αερολιμένας Νίκος Καζαντζάκης', False, True),
    ('Καζαντζάκης', 'Μουσείο Νίκου Καζαντζάκη', 'Το μουσείο Καζαντζάκη', False, True),
    ('Όμηρος', 'Όμηρος (αποσαφήνιση)', 'Όμηρος μπορεί να αναφέρεται', False, True),
    ('Οδύσσεια', '2001: Η Οδύσσεια του Διαστήματος', 'ταινία του Κιούμπρικ', False, True),
    ('mercury', 'Freddie Mercury', 'Freddie Mercury was a singer', False, True),
    ('python asyncio gather', 'Python (genus)', 'Python is a genus of snakes', False, False),
    ('java', 'Java (programming language)', 'Java is a programming language', True, True),
    ('apple stock price', 'Apple', 'An apple is a fruit; stock; price', False, True),
    ('who wrote the odyssey', '2001: A Space Odyssey', 'film who wrote Kubrick and Clarke the screenplay odyssey', False, False),
    ('Paris Hilton', 'Paris', 'Paris is the capital of France; Hilton hotel', False, False),
    ('Georgia country', 'Georgia (U.S. state)', 'Georgia is a state; country music', False, True),
    ('rust borrow checker', 'Rust (video game)', 'Rust is a survival game; checker', False, True),
]


@pytest.mark.parametrize("query,title,body,right,kept", _SUBJECT_TABLE)
def test_the_subject_table(query, title, body, right, kept):
    row = _art(title, body)
    assert bool(S.wikipedia_rows(query, [row])) is kept


def test_the_subject_tables_score():
    """Where the rule stands on the 62 cases: 35 of the 49 right articles kept, 5 of the 13 wrong ones
    dropped. Known misses (documented): question-form and subject-not-first queries ("who wrote the
    odyssey", "how tall is mount everest", "Οδύσσεια του Ομήρου") and an article whose title does not
    name the subject the way the query does (ΕΛΣΤΑΤ, Καμπέρα, Canberra); known wrong keeps: a title
    that CONTAINS the name (an airport, a museum, a disambiguation page, "Freddie Mercury" for "mercury")."""
    right_kept = sum(1 for *_x, right, kept in _SUBJECT_TABLE if right and kept)
    wrong_kept = [(q, t) for q, t, _b, right, kept in _SUBJECT_TABLE if kept and not right]
    assert (len(_SUBJECT_TABLE), sum(1 for r in _SUBJECT_TABLE if r[3]), right_kept, len(wrong_kept)) == (62, 49, 35, 8)
    kept = {(q, t) for q, t, _b, _r, k in _SUBJECT_TABLE if k}
    for pair in [("who wrote the odyssey", "Odyssey"), ("Who wrote the Odyssey", "Odyssey"),
                 ("how tall is mount everest", "Mount Everest"), ("python asyncio gather", "Asyncio"),
                 ("ποιος έγραψε την οδύσσεια", "Οδύσσεια"), ("Οδύσσεια του Ομήρου", "Οδύσσεια"),
                 ("Athens weather", "Athens"), ("Paris Hilton", "Paris"), ("python asyncio gather", "Python (genus)"),
                 ("who wrote the odyssey", "2001: A Space Odyssey")]:
        assert pair not in kept, pair


#: r2 confirmation: a one-word title that is merely a WORD of the query. A rule that took any title whose
#: words all stand in the query kept 14 of these 22 — it is gone; every one is rejected.
_WORD_OF_THE_QUERY = [
    ('who wrote the odyssey', 'Who', 'Who may refer to the World Health Organization or the band The Who'),
    ('the who tour dates 2026', 'The Who', 'The Who are an English rock band'),
    ('python 3.14 release date', 'Release', 'Release may refer to a software release'),
    ('PostgreSQL 18 release date', 'Date', 'A date is a fruit; calendar date'),
    ('history of the roman empire', 'History', 'History is the study of the past'),
    ('greece inflation rate 2026', 'Greece', 'Greece is a country in Southeast Europe'),
    ('python asyncio gather', 'Python', 'Python may refer to snakes or a programming language'),
    ('capital of Australia', 'Capital', 'Capital may refer to a capital city or economics'),
    ('Athens weather tomorrow', 'Weather', 'Weather is the state of the atmosphere'),
    ('best pizza in new york', 'New York', 'New York is a state in the United States'),
    ('rust borrow checker error', 'Error', 'An error is a mistake'),
    ('apple stock price today', 'Apple', 'An apple is a fruit'),
    ('ιστορία του Πειραιά', 'Ιστορία', 'Ιστορία είναι η μελέτη του παρελθόντος'),
    ('πληθωρισμός στην Ελλάδα 2026', 'Ελλάδα', 'Η Ελλάδα είναι χώρα της νοτιοανατολικής Ευρώπης'),
    ('καιρός Αθήνα αύριο', 'Καιρός', 'Καιρός είναι η κατάσταση της ατμόσφαιρας'),
    ('καιρός Αθήνα αύριο', 'Αθήνα', 'Η Αθήνα είναι η πρωτεύουσα της Ελλάδας'),
    ('Πότε πέθανε ο Καζαντζάκης', 'Πότε', 'Πότε είναι τραγούδι'),
    ('iphone 17 price in greece', 'Price', 'Price is the amount paid'),
    ('how to install docker on ubuntu', 'Docker', 'Docker may refer to a dock worker or software'),
    ('who is the president of france', 'France', 'France is a country'),
    ('New York Times paywall', 'New York', 'New York is a state'),
    ('Tesla model 3 range', 'Tesla', 'Nikola Tesla was an inventor'),
]


@pytest.mark.parametrize("query,title,body", _WORD_OF_THE_QUERY)
def test_a_title_that_is_only_a_word_of_the_query_is_not_its_subject(query, title, body):
    assert S.wikipedia_rows(query, [_art(title, body)]) == []
    assert not hasattr(S, "title_named_by_query")


# probe bookkeeping

def _probe_bench(engine="yandex"):
    _write_bench({engine: _refusals(newest_ago=3 * 3600.0)})


def test_one_probe_of_an_engine_at_a_time(net, monkeypatch):
    _probe_bench()
    t0 = time.time()
    assert "yandex" not in S.benched_engines(now=t0) and S.probed_engines() == {"yandex"}
    for i in range(1, 30):                                         # 27 watchers and a full pool, reproduced (r2)
        assert "yandex" in S.benched_engines(now=t0 + i * 0.1) and S.probed_engines() == frozenset()
    # the probe's own time passes with nothing booked: the next may fly
    late = t0 + S._engine_timeout("yandex") + S._RACE_WAVE_GRACE + 1
    assert "yandex" not in S.benched_engines(now=late) and S.probed_engines() == {"yandex"}
    # booked (an answer): no longer benched at all; booked (a refusal): rests, without waiting out the flight
    S._bench_note("yandex", "blocked", now=late + 1)
    assert S._bench_state["in_flight"] == {}
    assert "yandex" in S.benched_engines(now=late + 2) and S.probed_engines() == frozenset()


def test_a_wave_nobody_wins_does_not_wait_for_its_probe(net, monkeypatch):
    _probe_bench()
    monkeypatch.setenv("GHOST_SEARCH_ENGINES_OFF", "wikipedia")
    done = []

    def yandex(proxy):
        time.sleep(0.8)
        done.append(1)
        raise RuntimeError("blocked: HTTP 403 from yandex")
    seen_at_end = []
    real_log = S._log_wave_engines
    monkeypatch.setattr(S, "_log_wave_engines", lambda w, tok: (seen_at_end.append(len(done)), real_log(w, tok)))
    rows, _calls, took, tokens = _wave({"yandex": yandex})
    assert rows == [] and tokens[0]["yandex"] == "probe-unfinished" and took < 10.0
    assert seen_at_end == [0]                                      # the wave ended with yandex's circuits still out
    assert tokens[0]["duckduckgo"] == "blocked"
    # …and the probe went on in the background and was booked: yandex rests again
    assert len(done) == 3 and S._bench_recent()["yandex"][0][1] == "blocked"
    assert time.time() - S._bench_recent()["yandex"][0][0] < 60 and S._bench_state["in_flight"] == {}


def test_a_wave_the_caller_cancels_books_nothing_for_its_probe(net, monkeypatch):
    """r2: the caller cancelled during the short wait after the win — yandex's stopped attempts were
    booked as an `error`, and it rested two more hours for it."""
    _probe_bench()
    monkeypatch.setattr(S, "WIKI_WAIT_S", 30.0)
    net.wiki_delay = 60.0
    by = {e: _blocked(e) for e in DDGS_ENGINES}
    by["yahoo"] = _answer()
    by["yandex"] = lambda proxy: (time.sleep(0.6), [])[1]
    before = None

    async def _main():
        nonlocal before
        task = asyncio.ensure_future(S._race_search_wave(QUERY, PROXY, 0))
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10 and not S._PROBE_TASKS:   # yahoo has won; the wave waits for Wikipedia
            await asyncio.sleep(0.02)
        assert S._PROBE_TASKS
        before = list(S._bench_recent()["yandex"])
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(1.0)                                   # yandex's threads end; nothing may be booked
        return [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    with _use_ddgs(_fake_ddgs(by, [])):
        left = asyncio.run(_main())
    assert left == [] and not [w for w in S._PROBE_TASKS if not w.done()]
    assert S._bench_recent()["yandex"] == before                   # no outcome booked
    assert len(S._ENGINE_WAVES) == 0


def test_the_shutdown_hook_stops_probes_before_it_flushes(net, monkeypatch):
    from ghost_agent import main as M
    _probe_bench()
    monkeypatch.setenv("GHOST_SEARCH_ENGINES_OFF", "wikipedia")
    by = {e: _blocked(e) for e in DDGS_ENGINES}
    by["yahoo"] = _answer()
    by["yandex"] = lambda proxy: (time.sleep(0.6), [])[1]

    async def _main():
        rows = await S._race_search_wave(QUERY, PROXY, 0)
        assert rows and [w for w in S._PROBE_TASKS if not w.done()]
        before = list(S._bench_recent()["yandex"])
        assert await M.flush_search_bench() is True
        await asyncio.sleep(1.0)
        return before
    with _use_ddgs(_fake_ddgs(by, [])):
        before = asyncio.run(_main())
    assert not [w for w in S._PROBE_TASKS if not w.done()]
    assert S._bench_recent()["yandex"] == before                   # nothing booked after the shutdown began
    assert S.cancel_probes() == 0


def test_eight_links_to_one_page_are_one_row_not_an_error(net):
    """r2: the three-address minimum counts ADDRESSES; the rows are de-duplicated after it."""
    net.brave = (200, brave_page(_wrapped(8)))
    net.go = {f"/go/v1/token{i}": (200, "", go_page("https://target.example/same")) for i in range(8)}
    assert [r["href"] for r in _brave(net)] == ["https://target.example/same"]
    net.go = {f"/go/v1/token{i}": (200, "", go_page("https://target.example/same")) for i in range(2)}
    with pytest.raises(RuntimeError):                              # two addresses of eight links: still an error
        _brave(net)


@pytest.mark.parametrize("value,warns", [("1", True), ("", False), ("0", False), (None, False)])
def test_the_agent_warns_once_at_startup_when_the_test_switch_is_set(monkeypatch, value, warns):
    from ghost_agent import main as M
    logged = []
    monkeypatch.setattr(M, "pretty_log", lambda title, msg="", **k: logged.append((title, str(msg), k.get("level"))))
    if value is None:
        monkeypatch.delenv("GHOST_SEARCH_NO_NETWORK", raising=False)
    else:
        monkeypatch.setenv("GHOST_SEARCH_NO_NETWORK", value)
    assert M.warn_search_test_switch() is warns
    assert len(logged) == (1 if warns else 0)
    if warns:
        assert logged[0][2] == "WARNING" and "GHOST_SEARCH_NO_NETWORK" in logged[0][1]
