"""§4VB: an onion engine's page that is not its RESULT LISTING is never booked as results.

Live 11-20: for some queries ("onion search engines", "tor onion services purpose") Ahmia answered with its
"Help Us to Help You" notice — a help programme's links in the same ``li.result`` rows its listing uses. The
tool logged "ahmia: 1 onion result(s)", handed the model the programme's onion as the search's answer, and
darkweb_research distilled that site three times (one turn 672 s). The listing carries ``#totalResults``; the
notice does not. Fixtures are synthetic (the repository is public): structure as measured, test labels only.
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from ghost_agent.tools import darkweb_search as dw
from ghost_agent.tools.darkweb_search import _parse_onion_results
from tests.onion_v3 import v3_label

A, B, C = (v3_label(x) for x in "xyz")
AHMIA = "juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd.onion"
pytestmark = pytest.mark.asyncio


def _row(label: str, title: str) -> str:
    return (f'<li class="result"><h4><a href="/search/redirect?search_term=q&redirect_url=http://{label}.onion/">'
            f'{title}</a></h4><p>{title} text</p><cite>{label}.onion</cite></li>')


#: the notice page, as served 11-20: title, the help rows (two clearnet, one onion), NO #totalResults
NOTICE = ('<html><head><title>Ahmia &mdash; Help Us to Help You</title></head><body>'
          '<div id="ahmiaResultsPage"><ol>'
          '<li class="result"><a href="/search/redirect?redirect_url=https://help.example/">Help programme</a></li>'
          f'{_row(A, "Help programme onion")}'
          '</ol></div></body></html>')
#: a real listing: the same rows, WITH the "N results" line
LISTING = ('<html><body><div id="ahmiaResultsPage"><p id="totalResults">Showing 3 results</p><ol>'
           f'{_row(A, "Forum A")}{_row(B, "Market B")}{_row(C, "Wiki C")}</ol></div></body></html>')


@pytest.fixture(autouse=True)
def _clean():
    dw._FORM_TOKEN_CACHE.clear()
    dw._ENGINE_BREAKER.clear()
    yield
    dw._ENGINE_BREAKER.clear()


def _ahmia() -> dict:
    return dict(next(e for e in dw._DEFAULT_ONION_ENGINES if e["name"] == "ahmia"))


# ───────────────────────────── the parser ─────────────────────────────

async def test_the_notice_fixture_is_the_defect_world_without_a_marker():
    """The instrument can fail (R6): read as before (no marker), the notice IS booked as one result."""
    got = _parse_onion_results(NOTICE, {AHMIA}, result_selector="li.result")
    assert [r["url"] for r in got] == [f"http://{A}.onion/"]


async def test_a_page_without_the_listing_marker_books_nothing_and_says_why():
    diag: dict = {}
    rest: list = []
    got = _parse_onion_results(NOTICE, {AHMIA}, result_selector="li.result", results_marker="#totalResults",
                               diag=diag, rest=rest)
    assert got == [] and diag == {"no_listing": True} and rest == [""]


async def test_a_listing_with_the_marker_is_read_whole():
    diag: dict = {}
    got = _parse_onion_results(LISTING, {AHMIA}, result_selector="li.result", results_marker="#totalResults",
                               diag=diag)
    assert [r["url"] for r in got] == [f"http://{L}.onion/" for L in (A, B, C)] and diag == {}


async def test_a_selector_the_parser_rejects_reads_the_page_as_before():
    diag: dict = {}
    got = _parse_onion_results(NOTICE, {AHMIA}, result_selector="li.result", results_marker="#[[bad",
                               diag=diag)
    assert len(got) == 1 and diag == {}


#: review r1: an error / maintenance page — no result rows, no marker. Must stay on the OLD path.
ERROR_PAGE = '<html><body><h1>Search is temporarily unavailable</h1></body></html>'
#: review r1: a real listing whose "N results" line was renamed — more rows than a notice holds
DRIFTED = ('<html><body><p class="total-results">Showing 6 results</p><ol>'
           + "".join(_row(v3_label(f"d{i}"), f"Site {i}") for i in range(6)) + '</ol></body></html>')


async def test_a_listing_whose_marker_drifted_keeps_its_results_and_says_so():
    diag: dict = {}
    got = _parse_onion_results(DRIFTED, {AHMIA}, result_selector="li.result", results_marker="#totalResults",
                               diag=diag)
    assert len(got) == 6 and diag == {"marker_missing": True}


@pytest.mark.parametrize("n, kept", [(dw._NO_HITS_MAX_LINKS, 0), (dw._NO_HITS_MAX_LINKS + 1,
                                                                   dw._NO_HITS_MAX_LINKS + 1)])
async def test_the_line_between_a_notice_and_a_drifted_listing(n, kept):
    """Review r2: the boundary itself — at most `_NO_HITS_MAX_LINKS` rows is a notice, one more is a listing."""
    page = ('<html><body><ol>' + "".join(_row(v3_label(f"b{i}"), f"Site {i}") for i in range(n))
            + '</ol></body></html>')
    diag: dict = {}
    got = _parse_onion_results(page, {AHMIA}, result_selector="li.result", results_marker="#totalResults",
                               diag=diag)
    assert len(got) == kept and diag == ({"no_listing": True} if kept == 0 else {"marker_missing": True})


async def test_a_notice_with_only_clearnet_links_is_still_a_notice():
    page = NOTICE.replace(_row(A, "Help programme onion"), "")
    diag: dict = {}
    assert _parse_onion_results(page, {AHMIA}, result_selector="li.result", results_marker="#totalResults",
                                diag=diag) == [] and diag == {"no_listing": True}


async def test_a_page_with_no_result_rows_is_not_judged_by_the_marker():
    diag: dict = {}
    assert _parse_onion_results(ERROR_PAGE, {AHMIA}, result_selector="li.result",
                                results_marker="#totalResults", diag=diag) == [] and diag == {}


# ───────────────────────────── the engine table ─────────────────────────────

async def test_both_ahmia_endpoints_carry_the_marker_and_an_override_keeps_it(monkeypatch):
    for e in dw._DEFAULT_ONION_ENGINES:
        if e["index"] == "ahmia":
            assert e["results_marker"] == "#totalResults", e["name"]
    monkeypatch.setenv("GHOST_ONION_ENGINES",
                       '[{"name": "x", "url": "http://x/?q={q}", "results_marker": "#n"}, '
                       '{"name": "y", "url": "http://y/?q={q}"}]')
    got = {e["name"]: e for e in dw._load_engines()}
    assert got["x"]["results_marker"] == "#n" and "results_marker" not in got["y"]


# ───────────────────────────── one engine query, end to end ─────────────────────────────

async def test_the_notice_is_a_no_match_answer_asked_once_and_never_charged():
    calls = []

    async def _fetch(url, proxy, timeout, **kw):
        calls.append(url)
        return 200, NOTICE
    eng = _ahmia()
    eng.pop("form_token_from")                                  # no token round trip in this test
    with patch.object(dw, "_fetch_raw_html", _fetch):
        for _ in range(3):
            res = await dw._query_engine(eng, "onion search engines", "socks5h://127.0.0.1:9050")
            assert res == [] and res.outcome == dw.OUTCOME_NO_MATCH
    assert len(calls) == 3                                       # one fetch per search: no second circuit
    assert "ahmia" not in dw._ENGINE_BREAKER                     # answering is not failing


async def test_the_listing_still_comes_back_as_results():
    async def _fetch(url, proxy, timeout, **kw):
        return 200, LISTING
    eng = _ahmia()
    eng.pop("form_token_from")
    with patch.object(dw, "_fetch_raw_html", _fetch):
        res = await dw._query_engine(eng, "news", "socks5h://127.0.0.1:9050")
    assert res.outcome == dw.OUTCOME_RESULTS and len(res) == 3


async def test_an_error_page_still_goes_to_the_diagnosis_its_retry_and_the_breaker():
    """Review r1 MAJOR: the marker must not turn a maintenance page into a confident "no match"."""
    calls = []

    async def _fetch(url, proxy, timeout, **kw):
        calls.append(url)
        return 200, ERROR_PAGE
    eng = _ahmia()
    eng.pop("form_token_from")
    with patch.object(dw, "_fetch_raw_html", _fetch):
        res = await dw._query_engine(eng, "news", "socks5h://127.0.0.1:9050")
    assert res == [] and res.outcome == dw.OUTCOME_PARSER and len(calls) == 2


async def test_a_drifted_listing_comes_back_as_results_with_a_warning(capsys):
    async def _fetch(url, proxy, timeout, **kw):
        return 200, DRIFTED
    eng = _ahmia()
    eng.pop("form_token_from")
    with patch.object(dw, "pretty_log") as plog, patch.object(dw, "_fetch_raw_html", _fetch):
        res = await dw._query_engine(eng, "news", "socks5h://127.0.0.1:9050")
    assert res.outcome == dw.OUTCOME_RESULTS and len(res) == 6
    warned = [c for c in plog.call_args_list if c.kwargs.get("level") == "WARNING"]
    assert warned and "#totalResults" in warned[0].args[1]


async def test_the_whole_search_reports_no_notice_link(monkeypatch):
    """The surface the model reads: the tool's text names the notice's onion nowhere."""
    async def _fetch(url, proxy, timeout, **kw):
        return 200, NOTICE
    eng = _ahmia()
    eng.pop("form_token_from")
    with patch.object(dw, "_load_engines", lambda: [eng]), patch.object(dw, "_fetch_raw_html", _fetch):
        out = await dw.tool_darkweb_search("onion search engines", tor_proxy="socks5://127.0.0.1:9050")
    assert A not in str(out)


# ───────────────────────────── R1 enumeration ─────────────────────────────

async def test_every_parse_of_an_engine_page_passes_the_engine_s_marker():
    """Every call of `_parse_onion_results` in src/ passes ``results_marker`` and ``diag`` — a new fetch path
    that parses an engine page without them would book a notice page as results again."""
    src = Path(dw.__file__).resolve().parents[1]
    sites = []
    for py in src.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")
            args = [a.id for a in node.args if isinstance(a, ast.Name)]
            if name == "_parse_onion_results" or "_parse_onion_results" in args:
                sites.append((py.name, node.lineno, {k.arg for k in node.keywords}))
    assert sites, "the enumeration found no call — the instrument is broken"
    for where, line, kws in sites:
        assert {"results_marker", "diag"} <= kws, (where, line)
