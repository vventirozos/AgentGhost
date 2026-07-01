"""Tests for the dark-web (.onion) search tool.

All network access is mocked — `_fetch_raw_html` for the search phase (which
needs raw result HTML) and `helper_fetch_url_content` for the research phase
(which reads onion page text). No real Tor / onion traffic occurs.

Coverage:
  * onion-URL parsing (anchors, ahmia-style redirect wrappers, plain text)
  * cross-engine merge + corroboration ranking + de-dup by onion host
  * graceful zero-results error (no exception)
  * anonymous-mode stylometry scrub + SOCKS identity tagging
  * GHOST_ONION_ENGINES override + malformed-override fallback
  * socks5:// -> socks5h:// normalisation (mandatory for .onion DNS)
  * the research (deep-read) path: search -> fetch -> distil
  * result caching
"""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.tools import darkweb_search as dw
from ghost_agent.tools.darkweb_search import (
    tool_darkweb_search,
    tool_darkweb_research,
    _parse_onion_results,
    _extract_onion,
    _normalize_tor_proxy,
    _load_engines,
)

V3_A = "juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd"
V3_B = "torchdeedp3i2jigzjdmfpn5ttjhthh5wbmda2rr3jvqjg5p77c54dqd"
V3_C = "haystak5njsmn2hqkewecpaxetahtwhsbsa64jom2k22z5afxhnpxfid"


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts with an empty shared search cache."""
    from ghost_agent.tools import search as _s

    _s._SEARCH_CACHE.clear()
    yield
    _s._SEARCH_CACHE.clear()


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def test_parse_anchor_redirect_and_plaintext():
    html = f"""
      <li class="result"><a href="http://{V3_A}.onion/foo">Alpha Market</a>
        <p>Some description text</p></li>
      <a href="/search/redirect?redirect_url=http%3A%2F%2F{V3_B}.onion%2Fx">Beta Forum</a>
      <span>bare url http://{V3_C}.onion/page in text</span>
    """
    res = _parse_onion_results(html)
    hosts = {r["url"].split("//")[1].split(".onion")[0] for r in res}
    assert hosts == {V3_A, V3_B, V3_C}
    alpha = next(r for r in res if V3_A in r["url"])
    assert alpha["title"] == "Alpha Market"
    assert "description" in alpha["snippet"]


def test_extract_onion_unwraps_redirect():
    wrapped = f"/r?redirect_url=http%3A%2F%2F{V3_B}.onion%2Fx"
    assert _extract_onion(wrapped) == f"http://{V3_B}.onion/x"


def test_parse_dedups_within_page():
    html = f'<a href="http://{V3_A}.onion/a">one</a><a href="http://{V3_A}.onion/b">two</a>'
    res = _parse_onion_results(html)
    assert len(res) == 1  # same onion host collapses


def test_clearnet_urls_are_ignored():
    html = '<a href="https://example.com/not-onion">clearnet</a>'
    assert _parse_onion_results(html) == []


# --------------------------------------------------------------------------
# Proxy normalisation
# --------------------------------------------------------------------------
def test_normalize_proxy_forces_socks5h():
    assert _normalize_tor_proxy("socks5://127.0.0.1:9050") == "socks5h://127.0.0.1:9050"
    assert _normalize_tor_proxy("socks5h://127.0.0.1:9050") == "socks5h://127.0.0.1:9050"


# --------------------------------------------------------------------------
# Engine registry override
# --------------------------------------------------------------------------
def test_engine_override_and_malformed_fallback(monkeypatch):
    monkeypatch.setenv(
        "GHOST_ONION_ENGINES",
        json.dumps([{"name": "custom", "url": "http://x.onion/?q={q}"}]),
    )
    engines = _load_engines()
    assert [e["name"] for e in engines] == ["custom"]

    monkeypatch.setenv("GHOST_ONION_ENGINES", "{not json")
    engines = _load_engines()
    # Falls back to the built-in default set rather than disabling search.
    assert any(e["name"] == "ahmia" for e in engines)


# --------------------------------------------------------------------------
# Search fan-out
# --------------------------------------------------------------------------
def _fetch_stub(engine_html: dict):
    """Build an async _fetch_raw_html stub keyed on which engine the URL is."""

    async def _stub(url, proxy, timeout):
        for token, html in engine_html.items():
            if token in url:
                return 200, html
        return 200, ""  # reached but empty

    return _stub


@pytest.mark.asyncio
async def test_search_merges_and_ranks_by_corroboration():
    # V3_A appears in BOTH ahmia and torch -> must rank first.
    ahmia_html = f'<a href="http://{V3_A}.onion/">Shared</a><a href="http://{V3_C}.onion/">OnlyAhmia</a>'
    torch_html = f'<a href="http://{V3_A}.onion/">Shared</a><a href="http://{V3_B}.onion/">OnlyTorch</a>'
    stub = _fetch_stub({"ahmia.fi": ahmia_html, "juhanurmi": ahmia_html, "torchdeed": torch_html, "haystak": ""})

    with patch.object(dw, "_fetch_raw_html", side_effect=stub):
        out = await tool_darkweb_search("drugs market", tor_proxy="socks5://127.0.0.1:9050")

    assert "Onion:" in out
    # The corroborated onion is ranked #1.
    assert out.index(V3_A) < out.index(V3_B)
    assert out.index(V3_A) < out.index(V3_C)
    assert "engines reached" in out


@pytest.mark.asyncio
async def test_slow_engine_bounded_by_deadline(monkeypatch):
    """A slow/hung engine can't dominate the concurrent gather: it is skipped
    at the per-engine deadline while fast engines' results still come through."""
    import asyncio as _asyncio

    monkeypatch.setattr(dw, "_ONION_ENGINE_DEADLINE", 0.3)
    slow_html = f'<a href="http://{V3_A}.onion/">Slow</a>'
    fast_html = f'<a href="http://{V3_B}.onion/">Fast</a>'

    async def _stub(url, proxy, timeout):
        if "ahmia.fi" in url:          # make the ahmia clearnet engine hang
            await _asyncio.sleep(1.0)
            return 200, slow_html
        return 200, fast_html          # torch / ahmia-onion respond instantly

    with patch.object(dw, "_fetch_raw_html", side_effect=_stub):
        out = await tool_darkweb_search("x", tor_proxy="socks5://127.0.0.1:9050")

    assert V3_B in out          # fast engines' results returned
    assert V3_A not in out      # slow engine was deadline-skipped, not awaited


@pytest.mark.asyncio
async def test_search_zero_results_returns_actionable_error():
    stub = _fetch_stub({})  # every engine reached but empty
    with patch.object(dw, "_fetch_raw_html", side_effect=stub):
        out = await tool_darkweb_search("nonexistent topic", tor_proxy="socks5://127.0.0.1:9050")
    assert "ERROR" in out and "ZERO results" in out


@pytest.mark.asyncio
async def test_search_requires_query():
    out = await tool_darkweb_search(query=None)
    assert "MANDATORY" in out


@pytest.mark.asyncio
async def test_search_caches_success():
    html = f'<a href="http://{V3_A}.onion/">Cached</a>'
    stub = _fetch_stub({"ahmia.fi": html})
    with patch.object(dw, "_fetch_raw_html", side_effect=stub) as m:
        await tool_darkweb_search("cache me", tor_proxy="socks5://127.0.0.1:9050")
        calls_after_first = m.call_count
        # Second identical query must hit cache (no new fetches).
        out2 = await tool_darkweb_search("cache me", tor_proxy="socks5://127.0.0.1:9050")
    assert m.call_count == calls_after_first
    assert V3_A in out2


@pytest.mark.asyncio
async def test_anonymous_mode_scrubs_query():
    html = f'<a href="http://{V3_A}.onion/">x</a>'
    stub = _fetch_stub({"ahmia.fi": html})
    with patch.object(dw, "_fetch_raw_html", side_effect=stub):
        with patch("ghost_agent.utils.stylometry.scrub_query", return_value="scrubbed") as msc:
            await tool_darkweb_search("please find me X", anonymous=True, tor_proxy="socks5://127.0.0.1:9050")
    msc.assert_called_once()


# --------------------------------------------------------------------------
# Research (deep-read) path
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_research_searches_then_fetches_and_distils():
    html = f'<a href="http://{V3_A}.onion/">Target</a>'
    stub = _fetch_stub({"ahmia.fi": html, "juhanurmi": html, "torchdeed": html, "haystak": html})

    llm = MagicMock()
    llm.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "Distilled fact about target."}}]}
    )

    with patch.object(dw, "_fetch_raw_html", side_effect=stub):
        with patch.object(dw, "helper_fetch_url_content", new_callable=AsyncMock) as mfetch:
            mfetch.return_value = "Raw onion page body text."
            out = await tool_darkweb_research(
                "target topic", tor_proxy="socks5://127.0.0.1:9050", llm_client=llm
            )

    assert "DARK-WEB RESEARCH RESULT" in out
    assert f"http://{V3_A}.onion/" in out
    assert "Distilled fact about target." in out
    # The onion page was actually fetched.
    mfetch.assert_awaited()


@pytest.mark.asyncio
async def test_research_zero_results_error():
    stub = _fetch_stub({})
    with patch.object(dw, "_fetch_raw_html", side_effect=stub):
        out = await tool_darkweb_research("nothing here", tor_proxy="socks5://127.0.0.1:9050")
    assert "ERROR" in out and "ZERO results" in out


# --------------------------------------------------------------------------
# Registration in the tool registry
# --------------------------------------------------------------------------
def test_tools_advertised_in_definitions():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS

    names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert "darkweb_search" in names
    assert "darkweb_research" in names
    # Schema sanity: query is a required string param.
    for t in TOOL_DEFINITIONS:
        if t["function"]["name"] in ("darkweb_search", "darkweb_research"):
            params = t["function"]["parameters"]
            assert params["properties"]["query"]["type"] == "string"
            assert params["required"] == ["query"]


def test_runner_dict_wires_darkweb_tools():
    from ghost_agent.tools.registry import get_available_tools

    ctx = MagicMock()
    tools = get_available_tools(ctx)
    assert "darkweb_search" in tools
    assert "darkweb_research" in tools
    assert callable(tools["darkweb_search"])
    assert callable(tools["darkweb_research"])
