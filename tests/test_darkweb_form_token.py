"""Tests for the onion-engine form token and the empty-body diagnosis.

Both cover the same live breakage (measured 2026-07-28): Ahmia began
302-redirecting ``/search/?q=…`` to ``/`` unless the query string also carries
the hidden field from its search form. Because `_fetch_raw_html` follows
redirects, that arrived as "HTTP 200, zero results", and the empty-body
diagnostic — a bare ``"javascript"`` substring test — matched the *homepage's*
hidden no-JS notice and told the operator to swap engines. The engine was
fine; the query was missing one parameter.

So the regression these tests guard is not just "send the token" but "when the
page is empty, say the thing that actually happened."

No network access: every fetch is stubbed.
"""
import pytest
from unittest.mock import patch

from ghost_agent.tools import darkweb_search as dw
from ghost_agent.tools.darkweb_search import (
    _apply_form_token,
    _diagnose_empty_body,
    _form_token,
    _parse_form_token,
    _query_engine,
    _redirected_away,
)

V3_A = "a" * 56

# Trimmed from the REAL ahmia.fi homepage (fetched over Tor, 2026-07-28):
# the search form with its hidden token, plus the display:none warning whose
# text defeated the old substring detector. Both parts matter — a fixture
# without the warning would let the buggy ordering pass.
AHMIA_HOME = """
<html><body>
  <div id="notTorBrowserWarning" style="display:none">
    <p>Unfortunately we have not deployd non-JavaScript version of Ahmia yet.</p>
  </div>
  <form id="searchForm" class="autocomplete" action="/search/" method="get">
    <input id="id_q" type="search" name="q" title="q">
    <input type="hidden" name="2e636d" value="e36dd8">
    <input type="submit" value="Search">
  </form>
  <script src="/static/js/jquery.min.js"></script>
</body></html>
"""


@pytest.fixture(autouse=True)
def _clear_token_cache():
    dw._FORM_TOKEN_CACHE.clear()
    yield
    dw._FORM_TOKEN_CACHE.clear()


def _engine(**over):
    e = {
        "name": "ahmia",
        "url": "https://ahmia.fi/search/?q={q}",
        "index": "ahmia",
        "form_token_from": "https://ahmia.fi/",
    }
    e.update(over)
    return e


# --------------------------------------------------------------------------
# Token parsing
# --------------------------------------------------------------------------
def test_parses_hidden_token_from_real_ahmia_form():
    assert _parse_form_token(AHMIA_HOME) == ("2e636d", "e36dd8")


def test_prefers_the_search_form_over_an_unrelated_one():
    """A hidden field on some other form is not the search token."""
    html = """
      <form action="/newsletter/"><input type="hidden" name="csrf" value="nope"></form>
      <form action="/search/"><input type="hidden" name="tok" value="yes"></form>
    """
    assert _parse_form_token(html) == ("tok", "yes")


def test_no_hidden_field_means_no_token_not_an_error():
    """Every engine that doesn't gate its endpoint lands here — `None` must
    mean "send nothing extra", never "something went wrong"."""
    assert _parse_form_token("<form action='/search/'><input name='q'></form>") is None
    assert _parse_form_token("") is None


def test_regex_fallback_is_attribute_order_agnostic(monkeypatch):
    """With bs4 unavailable the fallback still reads the pair, whichever
    order the attributes appear in."""
    import builtins

    real_import = builtins.__import__

    def _no_bs4(name, *a, **kw):
        if name == "bs4":
            raise ImportError("no bs4")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_bs4)
    assert _parse_form_token(
        '<input value="v1" type="hidden" name="n1">') == ("n1", "v1")


# --------------------------------------------------------------------------
# URL assembly
# --------------------------------------------------------------------------
def test_apply_token_appends_and_encodes():
    out = _apply_form_token("https://ahmia.fi/search/?q=x", ("a b", "c&d"))
    assert out == "https://ahmia.fi/search/?q=x&a+b=c%26d"


def test_apply_token_uses_question_mark_when_no_query():
    assert _apply_form_token("http://e.onion/s", ("t", "1")) == "http://e.onion/s?t=1"


def test_apply_token_is_a_noop_without_a_token():
    assert _apply_form_token("http://e.onion/s?q=x", None) == "http://e.onion/s?q=x"


def test_token_goes_before_a_fragment_not_after_it():
    """Appended after a '#' the token is never sent, so the engine keeps
    redirecting and we keep re-scraping a token that cannot work."""
    assert (_apply_form_token("http://e.onion/s?q=foo#top", ("d", "1"))
            == "http://e.onion/s?q=foo&d=1#top")


def test_a_token_colliding_with_the_query_param_is_not_appended():
    """A hidden field named like the engine's own query param would append a
    duplicate; frameworks taking the LAST value would then search for nothing
    — which looks like "no hits", never like a bad token, so it would never
    be invalidated."""
    assert (_apply_form_token("https://ahmia.fi/search/?q=cats", ("q", ""))
            == "https://ahmia.fi/search/?q=cats")


# --------------------------------------------------------------------------
# Redirect detection
# --------------------------------------------------------------------------
def test_redirect_away_detects_a_changed_path():
    assert _redirected_away("https://ahmia.fi/search/?q=x", "https://ahmia.fi/")


def test_redirect_away_detects_a_changed_host():
    assert _redirected_away("https://ahmia.fi/search/?q=x",
                            "https://someone-else.tld/search/?q=x")


@pytest.mark.parametrize("requested,final,why", [
    ("https://www.ahmia.fi/search/?q=x", "https://ahmia.fi/search/?q=x", "www stripped"),
    ("http://ahmia.fi:80/search/?q=x", "http://ahmia.fi/search/?q=x", "default port"),
    ("https://ahmia.fi:443/search/?q=x", "https://ahmia.fi/search/?q=x", "default port"),
    ("https://AHMIA.fi/Search/?q=x", "https://ahmia.fi/search/?q=x", "case"),
])
def test_cosmetic_rewrites_are_not_a_redirect_away(requested, final, why):
    """A false positive here is not just a bad message: it evicts a good
    token and buys a wasted retry on EVERY query, forever."""
    assert not _redirected_away(requested, final), why


def test_query_string_rewrites_are_not_a_redirect_away():
    """Engines legitimately normalise/reorder query params."""
    assert not _redirected_away(
        "https://ahmia.fi/search/?q=x&t=1", "https://ahmia.fi/search/?t=1&q=x")


def test_scheme_upgrade_is_not_a_redirect_away():
    assert not _redirected_away("http://ahmia.fi/search/?q=x",
                                "https://ahmia.fi/search/?q=x")


def test_missing_final_url_is_not_treated_as_a_redirect():
    assert not _redirected_away("https://ahmia.fi/search/?q=x", "")


# --------------------------------------------------------------------------
# Empty-body diagnosis — the ordering is the whole point
# --------------------------------------------------------------------------
def test_redirect_beats_a_no_js_notice_on_the_page_we_landed_on():
    """THE regression. Ahmia's homepage really does carry a no-JS notice, so
    both signals are present at once; reporting the JS one sent the operator
    to change engines when the endpoint had simply moved."""
    kind, msg = _diagnose_empty_body(
        AHMIA_HOME, "https://ahmia.fi/search/?q=x", "https://ahmia.fi/")
    assert kind == "redirected"
    assert "ahmia.fi/" in msg
    assert "javascript" not in msg.lower()


def test_js_only_reported_when_we_were_not_redirected():
    kind, msg = _diagnose_empty_body(
        "<p>Please enable JavaScript to use this search.</p>",
        "http://e.onion/s?q=x", "http://e.onion/s?q=x")
    assert kind == "js-only"
    assert "JavaScript" in msg


def test_a_real_zero_hit_results_page_is_not_engine_breakage():
    """THE second regression, and the one that survived the first fix. Ahmia
    carries its no-JS banner in the SITE-WIDE template, so a perfectly healthy
    zero-hit search page trips the JavaScript wording with no redirect to
    suppress it — and the operator was told to replace a working engine."""
    page = AHMIA_HOME.replace(
        "</body>",
        "<p>Sorry, but Ahmia couldn't find results for 'zzz'.</p></body>")
    kind, msg = _diagnose_empty_body(
        page, "https://ahmia.fi/search/?q=zzz", "https://ahmia.fi/search/?q=zzz")
    assert kind == "no-hits"
    assert "not a broken engine" in msg
    assert "GHOST_ONION_ENGINES" not in msg


def test_a_large_page_is_never_diagnosed_as_a_js_shell():
    """A page that demands JavaScript AND is a megabyte of markup is not a
    JS shell — the wording alone must not carry the claim."""
    big = "<p>Please enable JavaScript</p>" + ("<div>filler</div>" * 4000)
    kind, _ = _diagnose_empty_body(big, "http://e.onion/s?q=x", "http://e.onion/s?q=x")
    assert kind == "parser"


@pytest.mark.parametrize("boast", [
    "This search engine works without JavaScript.",
    "No JavaScript required. No cookies. No logs.",
    "A non-JavaScript version is available at /lite",
])
def test_privacy_boasts_are_not_a_js_only_page(boast):
    """Polarity matters: 'works without JavaScript' is exactly what the onion
    engines most likely to WORK put in their footers. A polarity-blind
    pattern turns those into the ones we declare broken."""
    kind, _ = _diagnose_empty_body(
        f"<footer>{boast}</footer>", "http://e.onion/s?q=x", "http://e.onion/s?q=x")
    assert kind != "js-only"


def test_an_unknown_redirect_chain_downgrades_the_js_claim():
    """If `final_url` never got recorded we cannot rule out that the no-JS
    notice belongs to a page we were bounced to — which is exactly how the
    old detector lied. The message must admit that rather than re-assert it."""
    kind, msg = _diagnose_empty_body(
        "<p>Please enable JavaScript.</p>", "https://ahmia.fi/search/?q=x", None)
    assert kind == "js-only"
    assert "redirect chain is unknown" in msg
    assert "set GHOST_ONION_ENGINES" not in msg


def test_a_mere_script_tag_is_not_a_js_only_page():
    """The old detector matched the bare word "javascript" anywhere in the
    body, which nearly every page contains."""
    kind, msg = _diagnose_empty_body(
        '<script src="/static/js/jquery.min.js"></script><p>welcome</p>',
        "http://e.onion/s?q=x", "http://e.onion/s?q=x")
    assert kind == "parser"
    assert "format may have drifted" in msg


def test_redirect_message_never_leaks_the_search_terms():
    """The onion HOST is kept — "we ended up on a different site" and "we
    ended up on this site's homepage" need completely different responses,
    and the log redactor scrubs onion hostnames downstream anyway. The QUERY
    STRING is dropped: it carries what the operator searched for."""
    _, msg = _diagnose_empty_body(
        "", f"http://{V3_A}.onion/search/?q=secret+terms",
        f"http://{V3_A}.onion/blocked?q=secret+terms")
    assert "secret" not in msg
    assert "/blocked" in msg


def test_a_cross_host_redirect_names_the_other_host():
    """A bounce to a foreign host is a different event from a homepage
    bounce — a path-only message made them indistinguishable."""
    _, msg = _diagnose_empty_body(
        "", "https://ahmia.fi/search/?q=x", "https://captive-portal.tld/")
    assert "captive-portal.tld" in msg


# --------------------------------------------------------------------------
# Token fetching + caching
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_token_is_fetched_once_and_cached():
    calls = []

    async def _stub(url, proxy, timeout, **kw):
        calls.append(url)
        return 200, AHMIA_HOME

    with patch.object(dw, "_fetch_raw_html", side_effect=_stub):
        assert await _form_token("https://ahmia.fi/", None) == ("2e636d", "e36dd8")
        assert await _form_token("https://ahmia.fi/", None) == ("2e636d", "e36dd8")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_absence_of_a_token_is_cached_too():
    """Otherwise every search re-fetches the homepage of an engine that has
    no token, paying a Tor round-trip to learn nothing."""
    calls = []

    async def _stub(url, proxy, timeout, **kw):
        calls.append(url)
        return 200, "<html><form action='/search/'></form></html>"

    with patch.object(dw, "_fetch_raw_html", side_effect=_stub):
        assert await _form_token("http://e.onion/", None) is None
        assert await _form_token("http://e.onion/", None) is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_token_fetch_failure_fails_open():
    async def _boom(url, proxy, timeout, **kw):
        raise RuntimeError("tor down")

    with patch.object(dw, "_fetch_raw_html", side_effect=_boom):
        assert await _form_token("https://ahmia.fi/", None) is None


@pytest.mark.asyncio
async def test_a_failed_token_fetch_is_not_cached_as_no_token():
    """The dangerous conflation: "Tor blipped" must not be remembered as
    "this engine needs no token", or one transient failure silently strands
    the engine on the un-tokened path — zero results — for the whole TTL."""
    attempts = []

    async def _flaky(url, proxy, timeout, **kw):
        attempts.append(url)
        if len(attempts) == 1:
            raise RuntimeError("tor down")
        return 200, AHMIA_HOME

    with patch.object(dw, "_fetch_raw_html", side_effect=_flaky):
        assert await _form_token("https://ahmia.fi/", None) is None
        assert await _form_token("https://ahmia.fi/", None) == ("2e636d", "e36dd8")
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_a_non_200_token_page_is_not_cached_either():
    calls = []

    async def _blocked(url, proxy, timeout, **kw):
        calls.append(url)
        return 503, ""

    with patch.object(dw, "_fetch_raw_html", side_effect=_blocked):
        assert await _form_token("https://ahmia.fi/", None) is None
        assert await _form_token("https://ahmia.fi/", None) is None
    assert len(calls) == 2  # retried, not cached
    assert "https://ahmia.fi/" not in dw._FORM_TOKEN_CACHE


# --------------------------------------------------------------------------
# End-to-end through _query_engine
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_query_engine_sends_the_token_and_gets_results():
    """The whole fix in one test: without the token the engine 302s to the
    homepage and yields nothing; with it, results come back."""
    seen = []

    async def _stub(url, proxy, timeout, **kw):
        seen.append(url)
        if url == "https://ahmia.fi/":
            return 200, AHMIA_HOME
        if "2e636d=e36dd8" in url:
            return 200, f'<a href="http://{V3_A}.onion/">Hit</a>'
        # Token missing -> redirected to the homepage, exactly like live.
        kw.get("meta", {})["final_url"] = "https://ahmia.fi/"
        return 200, AHMIA_HOME

    with patch.object(dw, "_fetch_raw_html", side_effect=_stub):
        out = await _query_engine(_engine(), "anonymous", "socks5h://127.0.0.1:9050")

    assert [r["url"] for r in out] == [f"http://{V3_A}.onion/"]
    assert any("2e636d=e36dd8" in u for u in seen)


@pytest.mark.asyncio
async def test_engine_without_a_token_url_is_queried_unchanged():
    """No `form_token_from` means no extra fetch and no extra parameter."""
    seen = []

    async def _stub(url, proxy, timeout, **kw):
        seen.append(url)
        return 200, f'<a href="http://{V3_A}.onion/">Hit</a>'

    engine = _engine(name="torch", url="http://torch.onion/s?query={q}")
    engine.pop("form_token_from")
    with patch.object(dw, "_fetch_raw_html", side_effect=_stub):
        out = await _query_engine(engine, "anonymous", "socks5h://127.0.0.1:9050")

    assert seen == ["http://torch.onion/s?query=anonymous"]
    assert len(out) == 1


@pytest.mark.asyncio
async def test_a_token_engine_gets_deadline_headroom_for_its_extra_fetch():
    """The deadline was sized for one search attempt. A token engine spends a
    fetch of its own inside it, so without headroom a cold-cache attempt is
    guillotined just before the search that was about to succeed."""
    import ghost_agent.tools.darkweb_search as m

    seen = {}

    async def _stub(url, proxy, timeout, **kw):
        if url == "https://ahmia.fi/":
            return 200, AHMIA_HOME
        # Take longer than the bare deadline but less than deadline+token.
        await __import__("asyncio").sleep(0.35)
        return 200, f'<a href="http://{V3_A}.onion/">Hit</a>'

    with patch.object(m, "_ONION_ENGINE_DEADLINE", 0.3), \
         patch.object(m, "_FORM_TOKEN_TIMEOUT", 0.3), \
         patch.object(dw, "_fetch_raw_html", side_effect=_stub):
        out = await _query_engine(_engine(), "q", "socks5h://127.0.0.1:9050")
    seen["n"] = len(out)
    assert seen["n"] == 1  # would be 0 without the extension


@pytest.mark.asyncio
async def test_a_redirect_we_did_not_send_a_token_for_keeps_the_cache():
    """Any redirect used to evict the token — including a captcha or exit-ban
    page that has nothing to do with it, costing a re-scrape every query."""
    homepage_hits = []

    async def _stub(url, proxy, timeout, **kw):
        if url == "https://ahmia.fi/":
            homepage_hits.append(url)
            return 200, "<form action='/search/'></form>"   # no token offered
        kw.get("meta", {})["final_url"] = "https://ahmia.fi/captcha"
        return 200, "<p>prove you are human</p>"

    with patch.object(dw, "_fetch_raw_html", side_effect=_stub):
        await _query_engine(_engine(), "q", "socks5h://127.0.0.1:9050")

    # One scrape, cached as "no token"; the captcha redirect must not evict it.
    assert len(homepage_hits) == 1
    assert "https://ahmia.fi/" in dw._FORM_TOKEN_CACHE


@pytest.mark.asyncio
async def test_a_redirect_away_drops_the_cached_token_so_the_retry_rescrapes():
    """Self-heal for a rotated token: the retry must not reuse the stale one
    it was just redirected for."""
    tokens = iter([("tok", "stale"), ("tok", "fresh")])
    homepage_hits = []

    async def _stub(url, proxy, timeout, **kw):
        if url == "https://ahmia.fi/":
            name, value = next(tokens)
            homepage_hits.append(value)
            return 200, (f"<form action='/search/'><input type='hidden' "
                         f"name='{name}' value='{value}'></form>")
        if "tok=fresh" in url:
            return 200, f'<a href="http://{V3_A}.onion/">Hit</a>'
        kw.get("meta", {})["final_url"] = "https://ahmia.fi/"
        return 200, AHMIA_HOME

    with patch.object(dw, "_fetch_raw_html", side_effect=_stub):
        out = await _query_engine(_engine(), "anonymous", "socks5h://127.0.0.1:9050")

    assert homepage_hits == ["stale", "fresh"]
    assert [r["url"] for r in out] == [f"http://{V3_A}.onion/"]
