"""Tests for the Tor/DDGS search-hardening changes.

Covers three independent fixes that together kill the "lots of errors and
mis-guidance" failure mode seen in production traces (yahoo/yandex engines
hanging over Tor, the model emitting `site:`/boolean queries the scraper
backends ignore, and the same near-identical query being re-fired many
times in one turn):

  1. backend pinning  — yahoo/yandex are NOT queried anymore
  2. query sanitizer  — Google-style operators are stripped before the wire
  3. result cache      — a repeated successful query is served from memory
"""
import pytest
from unittest.mock import patch, MagicMock

from src.ghost_agent.tools.search import (
    _sanitize_query,
    _TOR_BACKENDS,
    _cache_get,
    _cache_put,
    _proxy_for_attempt,
    tool_search_ddgs,
)


# --------------------------------------------------------------------------
# 1. Backend set
# --------------------------------------------------------------------------
def test_broken_and_useless_engines_excluded():
    """`wikipedia` is excluded because region='wt-wt' makes its engine build
    the non-existent host `wt.wikipedia.org` (always a ConnectError over
    Tor); `yahoo` is excluded as a pure hang-risk with no useful results.
    The engines that DO return results over Tor are kept — note `yandex`
    is back IN: it was measured returning results, and per-attempt circuit
    rotation (not exclusion) is how we handle its occasional bad exit."""
    backends = [b.strip() for b in _TOR_BACKENDS.split(",")]
    assert "wikipedia" not in backends
    assert "yahoo" not in backends
    # The engines empirically observed returning results over Tor.
    assert "mojeek" in backends
    assert "yandex" in backends
    assert "duckduckgo" in backends


# --------------------------------------------------------------------------
# 1c. Per-attempt circuit rotation
# --------------------------------------------------------------------------
def test_proxy_for_attempt_rotates_circuit_per_attempt():
    base = "socks5h://127.0.0.1:9050"
    p0 = _proxy_for_attempt(base, "some query", 0)
    p1 = _proxy_for_attempt(base, "some query", 1)
    p2 = _proxy_for_attempt(base, "some query", 2)
    # Distinct SOCKS identities → Tor's IsolateSOCKSAuth gives distinct
    # circuits, so the three attempt URLs must differ from one another.
    assert p0 != p1 != p2 and p0 != p2
    # All keep the same host:port and carry an injected credential.
    for p in (p0, p1, p2):
        assert "127.0.0.1:9050" in p
        assert "@" in p  # username:password injected


def test_proxy_for_attempt_same_query_attempt_is_stable():
    base = "socks5h://127.0.0.1:9050"
    assert _proxy_for_attempt(base, "q", 0) == _proxy_for_attempt(base, "q", 0)


def test_proxy_for_attempt_distinct_per_query():
    base = "socks5h://127.0.0.1:9050"
    assert _proxy_for_attempt(base, "query A", 0) != _proxy_for_attempt(base, "query B", 0)


def test_proxy_for_attempt_none_passthrough():
    # No proxy configured → nothing to rotate.
    assert _proxy_for_attempt(None, "q", 0) is None


# --------------------------------------------------------------------------
# 2. Query sanitizer
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    # The exact pathological query shape from the production trace.
    ('elite dangerous federal corvette combat build "gimbal" or "gimbaled" '
     'site:edshipbuilds.com or site:coriolis.io',
     'elite dangerous federal corvette combat build gimbal gimbaled'),
    # site: operator with its argument is removed entirely.
    ('site:wikipedia.org python asyncio', 'python asyncio'),
    # Quoted phrase: quotes go, words stay.
    ('"exact phrase" something', 'exact phrase something'),
    # Uppercase boolean operator removed; lowercase stopword too.
    ('best gpu for machine learning AND inference',
     'best gpu for machine learning inference'),
    # inurl:/intitle:/filetype: are all stripped.
    ('intitle:report filetype:pdf budget', 'budget'),
    # A clean keyword query is untouched.
    ('PostgreSQL 16 release notes', 'PostgreSQL 16 release notes'),
])
def test_sanitize_query_strips_operators(raw, expected):
    assert _sanitize_query(raw) == expected


def test_sanitize_query_preserves_or_inside_words():
    # "for"/"information" contain o-r/a-n-d substrings but are NOT operators.
    assert _sanitize_query("information for sale") == "information for sale"


def test_sanitize_query_falls_back_when_emptied():
    # A query that is ENTIRELY operators would sanitize to "" — we must not
    # send an empty query, so the original is returned unchanged.
    pure_ops = "site:.org OR site:.gov"
    assert _sanitize_query(pure_ops) == pure_ops


def test_sanitize_query_handles_empty():
    assert _sanitize_query("") == ""
    assert _sanitize_query(None) is None


# --------------------------------------------------------------------------
# 2b. Sanitizer is wired into the live search path (operators never reach DDGS)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
@patch("ddgs.DDGS")
@patch("src.ghost_agent.tools.search.importlib.util.find_spec")
async def test_operators_stripped_before_ddgs(mock_find_spec, mock_ddgs):
    mock_find_spec.return_value = True
    inst = MagicMock()
    mock_ddgs.return_value.__enter__.return_value = inst
    inst.text.return_value = [{"title": "T", "body": "B", "href": "http://ok.com"}]

    await tool_search_ddgs('foo "bar" site:x.com or site:y.com', None)

    # The query that actually hit DDGS must be the sanitized keyword form,
    # and the backend must be the pinned Tor-reliable set.
    inst.text.assert_called_once_with(
        "foo bar", max_results=20, region="wt-wt",
        safesearch="moderate", backend=_TOR_BACKENDS,
    )


# --------------------------------------------------------------------------
# 3. Result cache
# --------------------------------------------------------------------------
@pytest.mark.asyncio
@patch("ddgs.DDGS")
@patch("src.ghost_agent.tools.search.importlib.util.find_spec")
async def test_repeated_query_served_from_cache(mock_find_spec, mock_ddgs):
    mock_find_spec.return_value = True
    inst = MagicMock()
    mock_ddgs.return_value.__enter__.return_value = inst
    inst.text.return_value = [{"title": "T", "body": "B", "href": "http://ok.com"}]

    r1 = await tool_search_ddgs("repeated query", None)
    r2 = await tool_search_ddgs("repeated query", None)

    assert r1 == r2
    # DDGS was hit exactly once — the second call came from the cache.
    assert inst.text.call_count == 1


@pytest.mark.asyncio
@patch("ddgs.DDGS")
@patch("src.ghost_agent.tools.search.importlib.util.find_spec")
async def test_failed_search_is_not_cached(mock_find_spec, mock_ddgs):
    mock_find_spec.return_value = True
    inst = MagicMock()
    mock_ddgs.return_value.__enter__.return_value = inst
    inst.text.return_value = []  # empty → ZERO results, must not be cached

    await tool_search_ddgs("doomed query", None)
    # An error result must never be stored.
    assert _cache_get("doomed query") is None


def test_cache_roundtrip_and_ttl_expiry():
    _cache_put("k1", "v1")
    assert _cache_get("k1") == "v1"

    # Simulate an expired entry by back-dating its timestamp past the TTL.
    from src.ghost_agent.tools import search as _search
    ts, val = _search._SEARCH_CACHE["k1"]
    _search._SEARCH_CACHE["k1"] = (ts - (_search._SEARCH_CACHE_TTL + 1), val)
    assert _cache_get("k1") is None  # expired entries are dropped on read
