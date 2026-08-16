"""Tests for the Tor/DDGS search-hardening changes.

Covers three independent fixes that together kill the "lots of errors and
mis-guidance" failure mode seen in production traces (yahoo/yandex engines
hanging over Tor, the model emitting `site:`/boolean queries the scraper
backends ignore, and the same near-identical query being re-fired many
times in one turn):

  1. engine race set  — structurally-broken engines excluded; every engine
     in a wave rides its OWN Tor circuit (per-engine SOCKS salt)
  2. query sanitizer  — Google-style operators are stripped before the wire
  3. result cache      — a repeated successful query is served from memory
"""
import pytest
from unittest.mock import patch, MagicMock

from src.ghost_agent.tools.search import (
    _sanitize_query,
    _RACE_ENGINES,
    _TOR_BACKENDS,
    _cache_get,
    _cache_put,
    _proxy_for_attempt,
    tool_search_ddgs,
)


# --------------------------------------------------------------------------
# 1. Engine race set
# --------------------------------------------------------------------------
def test_broken_engines_excluded_and_race_set_wide():
    """`wikipedia` stays excluded: region='wt-wt' makes its engine build the
    non-existent host `wt.wikipedia.org` (always a ConnectError over Tor).
    `grokipedia` is a typeahead API — 0/6 on real queries. Everything else
    is IN, including `yahoo` (re-measured 2026-07-08: it now fails FAST,
    ~1.4-2.2s, instead of hanging, and actually won a probe). With ~10%
    per-(engine, circuit) success measured over Tor, every fast-failing
    engine is a cheap independent lottery ticket — exclusion is only for
    engines that are structurally broken or hang."""
    assert "wikipedia" not in _RACE_ENGINES
    assert "grokipedia" not in _RACE_ENGINES
    # The wide race set — each of these was observed winning over Tor on
    # at least some circuit.
    for engine in ("mojeek", "duckduckgo", "yandex", "brave", "google", "yahoo"):
        assert engine in _RACE_ENGINES
    # Legacy comma-joined constant stays derived from the race set.
    assert _TOR_BACKENDS == ",".join(_RACE_ENGINES)


def test_race_gives_each_engine_its_own_circuit():
    """The core 2026-07-08 fix: within one wave, engines must NOT share a
    circuit (a blocked exit fails them all together). The per-engine salt
    must yield a distinct SOCKS identity per engine for the same
    (query, wave) — and distinct across waves for the same engine."""
    base = "socks5h://127.0.0.1:9050"
    per_engine = {_proxy_for_attempt(base, "q", 0, salt=e[:4]) for e in _RACE_ENGINES}
    assert len(per_engine) == len(_RACE_ENGINES)
    assert _proxy_for_attempt(base, "q", 0, salt="moje") != _proxy_for_attempt(base, "q", 1, salt="moje")


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
    # send an empty query. When the operands carry no minable words either
    # (bare TLDs), the original is returned unchanged.
    pure_ops = "site:.org OR site:.gov"
    assert _sanitize_query(pure_ops) == pure_ops


# ──────────────────────────────────────────────────────────────────────
# A URL-shaped query is the query, spelled wrong (live failure 2026-08-15)
# ──────────────────────────────────────────────────────────────────────

def test_an_operator_only_query_is_MINED_not_passed_through_raw():
    """THE live failure. The model asked for
    `site:reddit.com/r/lgbtgreece/comments/1voyjgf/is_nudism_safe_in_greece`.
    Stripping the operator emptied the query, so the old fallback handed
    the scrapers the original — the one shape they cannot honour. Result:
    zero across four waves and two reformulations (one of them literally
    "how to site:reddit.com/..."), ~80s burned, and the turn was refuted
    by the late verifier for not retrieving the content.

    The operand is not noise; it is the query as a URL. Mine it."""
    got = _sanitize_query(
        "site:reddit.com/r/lgbtgreece/comments/1voyjgf/is_nudism_safe_in_greece")
    assert "site:" not in got, f"the unmatchable operator survived: {got!r}"
    # The slug's words are what the user actually wanted searched…
    for term in ("reddit", "lgbtgreece", "nudism", "safe", "greece"):
        assert term in got, f"{term!r} lost from {got!r}"
    # The post id is KEPT (corrected 2026-08-16). The first rule dropped
    # any token containing a digit, which also killed `log4shell`, `gpt4`,
    # `ipv6` and `sha256` — turning `site:en.wikipedia.org/wiki/Log4Shell`
    # into a confident search for "wikipedia". And the id is harmless
    # here: the query that actually WON on live Tor in the failing turn
    # was `reddit lgbtgreece nudism safe greece trans man 1voyjgf`, id
    # included. Only MOSTLY-digit or long mixed tokens are dropped now.


def test_url_furniture_and_OPAQUE_ids_are_dropped():
    from ghost_agent.tools.search import _keywords_from_operand
    mined = _keywords_from_operand(
        "example.com/en/wiki/index.html/watch/page/2/abc123/routing")
    assert "routing" in mined                      # a real term survives
    for junk in ("index", "html", "watch", "page", "com", "wiki"):
        assert junk not in mined.split(), f"{junk} is URL furniture"
    assert "abc123" not in mined                   # opaque id, mostly digits


def test_a_path_of_PURE_furniture_refuses_rather_than_guessing():
    """Only the domain label surviving means the mined query would search
    for the wrong thing confidently, so the original is returned and the
    failure stays loud."""
    raw = "inurl:example.com/en/index.html/watch/page/2/abc123"
    assert _sanitize_query(raw) == raw


def test_a_MEANINGFUL_term_carrying_a_digit_SURVIVES():
    """The rule was `any(ch.isdigit()) and not low.isalpha()` — whose
    second clause is DEAD, since a token with a digit is never isalpha().
    So it dropped every digit-bearing term: `site:en.wikipedia.org/wiki/
    Log4Shell` mined to `wikipedia`, and an honest zero (which made the
    model reformulate) became eight confident results about Wikipedia. A
    quiet wrong answer is worse than a loud failure."""
    assert "log4shell" in _sanitize_query(
        "site:en.wikipedia.org/wiki/Log4Shell")
    assert "ipv6" in _sanitize_query("inurl:docs.example.org/ipv6/routing")


def test_a_query_that_mines_to_NOTHING_DISTINCTIVE_is_refused():
    """If only the domain label survives, the mined query searches for
    the wrong thing confidently. Returning the original keeps the failure
    honest and loud."""
    got = _sanitize_query("site:en.wikipedia.org/wiki/")
    assert got == "site:en.wikipedia.org/wiki/"


def test_mining_is_capped_so_it_cannot_produce_a_keyword_stuffed_query():
    """>6 terms has near-zero organic hits anywhere — the module says so
    in `_reformulate_query`, and a long URL slug is exactly how you would
    get there by accident."""
    got = _sanitize_query(
        "site:example.com/alpha/bravo/charlie/delta/echo/foxtrot/golf/hotel")
    assert len(got.split()) <= 6, got


def test_a_NORMAL_query_with_a_site_operator_is_UNCHANGED():
    """Mining only happens when stripping would empty the query, so the
    common `keywords + site:` shape keeps its previous behaviour exactly."""
    assert _sanitize_query("python asyncio tutorial site:docs.python.org") \
        == "python asyncio tutorial"


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

    # The race fires one single-engine call per engine, so DDGS may be hit
    # up to len(_RACE_ENGINES) times — but EVERY call must carry the
    # sanitized keyword form and a backend drawn from the race set.
    assert inst.text.call_count >= 1
    for call in inst.text.call_args_list:
        assert call.args[0] == "foo bar"
        assert call.kwargs["max_results"] == 20
        assert call.kwargs["region"] == "wt-wt"
        assert call.kwargs["safesearch"] == "moderate"
        assert call.kwargs["backend"] in _RACE_ENGINES


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
    calls_after_first = inst.text.call_count
    r2 = await tool_search_ddgs("repeated query", None)

    assert r1 == r2
    # The race may hit several engines on the first search, but the second
    # search must add ZERO ddgs calls — it came from the cache.
    assert calls_after_first >= 1
    assert inst.text.call_count == calls_after_first


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
