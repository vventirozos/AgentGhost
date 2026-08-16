"""Two fixes from one live failure (2026-08-15).

THE TURN. "can you search the dark web for underground news?" — 135s, and
the operator got a list of links instead of any news. The trace:

    darkweb search   underground news
    darkweb engine   torch: 7 onion result(s)
    darkweb engine   torgle: 7 onion result(s)
    engine error     ahmia-onion: curl (28) ... less than 1 byte/sec
    engine error     ahmia: exceeded 50s deadline — skipped
    browser          navigate http://<keybase>.onion/undrgrndnews
    browser failed   net::ERR_SOCKS_CONNECTION_FAILED
    browser          navigate http://<keybase>.onion/undrgrndnews   ← AGAIN
    browser failed   net::ERR_SOCKS_CONNECTION_FAILED
    loop breaker     No-progress: repeated 2x — forcing a grounded conclusion

Measured afterwards over the agent's own Tor, which is what turns this
from a guess into two fixes:

    a live onion (duckduckgo)      HTTP 200 in 3.1s   ← Tor is HEALTHY
    the keybase onion it chose     curl exit 97       ← the SERVICE is dead
    a fabricated .onion address    curl exit 97       ← indistinguishable
    ahmia.fi/                      HTTP 200 in 0.7s   ← the SITE is up
    ahmia.fi/search?q=..&<token>   HTTP 504 in 31.4s  ← its SEARCH is down

So: (1) ERR_SOCKS_CONNECTION_FAILED on an onion means the hidden service
is offline, and the next action is a DIFFERENT result — never the same
URL; (2) ahmia is not misconfigured, it is failing, and paying its full
deadline twice on every search buys nothing.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../src')))

import ast
import time
from pathlib import Path

import pytest

from ghost_agent.tools import browser as B
from ghost_agent.tools import darkweb_search as D

_KEYBASE = ("http://keybase5wmilwokqirssclfnsqrjdsi7jdir5wy7y7iu3tanwmtp6oid"
            ".onion/undrgrndnews")
_LIVE = ("http://duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad"
         ".onion/")


def _strike(url, cause="net::ERR_SOCKS_CONNECTION_FAILED"):
    """Fail `url` enough times to be declared dead."""
    for _ in range(B._DEAD_ONION_STRIKES):
        B._mark_onion_dead(url, cause)


# NOTE: no file-local reset fixture. `tests/conftest.py`'s autouse
# `clear_onion_process_state` owns this, and duplicating it here MASKED
# that fixture — R2 added `_ONION_STRIKES` to conftest and a mutation
# removing it stayed green, because this file was resetting the state
# itself. The suite must exercise the thing that protects every OTHER
# test file.


def test_the_thresholds_are_PINNED_not_read_from_the_constant():
    """R1 MINOR: the first tests wrote `range(_ENGINE_BREAKER_FAILS - 1)`
    and `_DEAD_ONION_TTL + 1`, so they adapted to ANY value — a mutation
    setting the breaker to 1 failure, or the memo TTL to 24h, stayed
    green. Pin the numbers, with the reason.

    3 strikes: a transient blip must not sideline an engine.
    2 strikes: an onion rendezvous failure is one circuit's opinion.
    600s: Tor's default MaxCircuitDirtiness — a longer memo outlives the
    circuit whose failure created it."""
    assert D._ENGINE_BREAKER_FAILS == 3
    assert D._ENGINE_BREAKER_COOLDOWN == 900.0
    assert B._DEAD_ONION_STRIKES == 2
    assert B._DEAD_ONION_TTL == 600.0


def test_ONE_failure_is_not_enough_to_condemn_a_service():
    """R1 J3: banning on the first strike is one bad circuit's opinion,
    and the memo outlived the circuit that caused it."""
    B._mark_onion_dead(_KEYBASE, "net::ERR_SOCKS_CONNECTION_FAILED")
    assert B._dead_onion_notice(_KEYBASE) is None


def test_OUR_tor_being_broken_never_condemns_a_LIVE_service():
    """R1 J1 — proven live: with the wider marker list, a synthetic proxy
    error memoised the known-good DuckDuckGo onion and handed the model a
    confidently false 'this service is offline'. ERR_PROXY_CONNECTION_
    FAILED means the PROXY was unreachable; ERR_NAME_NOT_RESOLVED means
    Chromium got no proxy and tried real DNS on a .onion. Both are ours."""
    for cause in ("net::ERR_PROXY_CONNECTION_FAILED",
                  "net::ERR_TUNNEL_CONNECTION_FAILED",
                  "net::ERR_NAME_NOT_RESOLVED"):
        B._DEAD_ONIONS.clear()
        B._ONION_STRIKES.clear()
        _strike(_LIVE, cause)
        assert B._dead_onion_notice(_LIVE) is None, cause


def test_the_memo_has_a_kill_switch(monkeypatch):
    monkeypatch.setenv("GHOST_DEAD_ONION_MEMO", "0")
    _strike(_KEYBASE)
    assert B._dead_onion_notice(_KEYBASE) is None
    assert not B._DEAD_ONIONS


# ──────────────────────────────────────────────────────────────────────
# The dead-onion memo
# ──────────────────────────────────────────────────────────────────────

class TestADeadOnionIsNotRetried:

    def test_the_second_attempt_is_answered_without_touching_tor(self):
        """THE fix. Attempt one fails at the Tor layer; attempt two must
        come back instantly with 'pick a different result' instead of
        paying another round trip to relearn it."""
        assert B._dead_onion_notice(_KEYBASE) is None
        _strike(_KEYBASE)
        note = B._dead_onion_notice(_KEYBASE)
        assert note and "UNREACHABLE" in note
        # …and it must point at the NEXT action, since generic browser
        # advice ("raise the timeout", "use interact") is what invited the
        # identical retry.
        assert "DIFFERENT result" in note

    def test_it_is_keyed_on_HOST_not_url(self):
        """The live failure retried the same host. A path-keyed memo
        would let `/undrgrndnews` and `/other` each pay their own Tor
        timeout on a service that is dead at every path."""
        _strike(_KEYBASE)
        other = _KEYBASE.rsplit("/", 1)[0] + "/somewhere-else"
        assert B._dead_onion_notice(other) is not None

    def test_an_unrelated_onion_is_untouched(self):
        _strike(_KEYBASE)
        assert B._dead_onion_notice(_LIVE) is None

    def test_a_CLEARNET_host_is_never_memoised(self):
        """A SOCKS error on a clearnet URL means something else entirely
        (proxy down, blocked exit) and must not silently blacklist a site
        the agent can reach fine on the next circuit."""
        _strike("https://example.com/x")
        assert not B._DEAD_ONIONS

    def test_only_TOR_LAYER_failures_count(self):
        """A page that timed out, 404'd, or crashed the renderer says
        nothing about whether the service is reachable."""
        for cause in ("Timeout 30000ms exceeded",
                      "net::ERR_ABORTED",
                      "Target page, context or browser has been closed"):
            B._DEAD_ONIONS.clear()
            B._ONION_STRIKES.clear()
            _strike(_KEYBASE, cause)
            assert B._dead_onion_notice(_KEYBASE) is None, cause

    def test_the_memo_EXPIRES(self):
        """Hidden services come back. A permanent blacklist would turn a
        transient outage into an invisible, process-lifetime ban."""
        _strike(_KEYBASE)
        host = B._onion_host(_KEYBASE)
        B._DEAD_ONIONS[host] = time.monotonic() - (B._DEAD_ONION_TTL + 1)
        assert B._dead_onion_notice(_KEYBASE) is None
        assert host not in B._DEAD_ONIONS      # and it is swept, not left

    def test_onion_host_parsing_is_not_silently_broken(self):
        """⚠ THIS CAUGHT A REAL BUG. `_onion_host` first caught bare
        `Exception` and returned "" — which turned a NameError (module
        `urlparse` was never imported, because the `from urllib.parse
        import` near the top of browser.py lives INSIDE the runner-script
        STRING) into "not an onion", disabling the whole feature while
        the compiler and every structural check stayed green."""
        assert B._onion_host(_KEYBASE).endswith(".onion")
        assert B._onion_host("https://example.com/") == ""
        assert B._onion_host("") == ""
        assert B._onion_host(None) == ""


def test_the_helpers_are_MODULE_level_not_inside_the_runner_string():
    """browser.py's `_runner_script` returns the Playwright runner as one
    long string that contains its own `async def` lines. Anchoring an
    insert on text that also appears in there puts the code in the SCRIPT
    — it still compiles, still looks top-level to grep, and is simply
    absent from the module. That happened while writing this fix."""
    src = Path(B.__file__).read_text()
    tree = ast.parse(src)
    top = {n.name for n in tree.body
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert {"_onion_host", "_mark_onion_dead", "_dead_onion_notice"} <= top
    mod_imports = set()
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                mod_imports.add(a.asname or a.name)
    assert "_urlparse" in mod_imports, (
        "the URL parser is not imported at MODULE level — the import "
        "inside the runner string does not count, and the silent-except "
        "made that look like 'no onions here'")


# ──────────────────────────────────────────────────────────────────────
# The engine circuit breaker
# ──────────────────────────────────────────────────────────────────────

class TestAFailingEngineStopsCostingItsDeadline:
    """Measured over one day: torch 5 wins/0 fails, torgle 5/0, ahmia
    0/6, ahmia-onion 0/9 — every ahmia failure a full deadline. Two ahmia
    entries × ~50s, on every dark-web search, to learn nothing."""

    def test_it_takes_repeated_failures_to_open(self):
        for i in range(D._ENGINE_BREAKER_FAILS - 1):
            D._breaker_record("ahmia", False)
            assert not D._breaker_should_skip("ahmia"), (
                f"opened after only {i + 1} failure(s) — a transient blip "
                f"must not sideline an engine")
        D._breaker_record("ahmia", False)
        assert D._breaker_should_skip("ahmia")

    def test_a_healthy_engine_is_never_skipped(self):
        for _ in range(10):
            D._breaker_record("ahmia", False)
        assert not D._breaker_should_skip("torch")

    def test_a_WIN_clears_it_immediately(self):
        for _ in range(D._ENGINE_BREAKER_FAILS):
            D._breaker_record("ahmia", False)
        assert D._breaker_should_skip("ahmia")
        D._breaker_record("ahmia", True)
        assert not D._breaker_should_skip("ahmia")

    # `test_EMPTY_RESULTS_count_as_a_failure` lived here and was removed
    # (R2 M7): it asserted a substring of `ast.unparse(...)`, so a
    # behaviour-preserving `_won = bool(_res)` hoist failed it while the
    # real mutation ("did not raise" = win) was already killed by
    # `test_an_engine_returning_NOTHING_advances_the_breaker`. A proxy
    # that false-alarms and adds no coverage is worse than no test.

    def test_it_HALF_OPENS_after_the_cooldown(self):
        """A permanently-open breaker would silently retire an engine for
        the life of the process — these endpoints come back."""
        for _ in range(D._ENGINE_BREAKER_FAILS):
            D._breaker_record("ahmia", False)
        fails, _ = D._ENGINE_BREAKER["ahmia"]
        D._ENGINE_BREAKER["ahmia"] = (
            fails, time.monotonic() - (D._ENGINE_BREAKER_COOLDOWN + 1))
        assert not D._breaker_should_skip("ahmia"), "never re-probes"
        # …and exactly ONE probe: the clock re-arms so a failing probe
        # does not leak a second one on the very next search.
        assert D._breaker_should_skip("ahmia")

    def test_the_kill_switch_disables_it(self, monkeypatch):
        monkeypatch.setenv("GHOST_ONION_BREAKER", "0")
        for _ in range(D._ENGINE_BREAKER_FAILS * 3):
            D._breaker_record("ahmia", False)
        assert not D._breaker_should_skip("ahmia")

    # `test_the_breaker_is_CHECKED_before_the_deadline_is_paid` lived
    # here and was removed (R2 M7): `i_check < i_wait` is satisfied by
    # TEXT POSITION, so it survived `if False and _breaker_should_skip(…)`
    # — an inert guard — while false-alarming on an extracted local. The
    # behavioural twin `test_the_breaker_stops_a_failing_engine_being_
    # QUERIED` asserts the fetch never happens, which is the property.


class TestTheProductionCallSitesActuallyRun:
    """Same harness the existing browser tests use: `sandbox_dir` +
    `sandbox_manager` passed directly (see
    tests/test_browser_formatter_and_render_fixes.py)."""

    def _sandbox(self):
        from unittest.mock import MagicMock
        stub = MagicMock()
        stub.calls = []

        def _execute(cmd, timeout=300, **kwargs):
            stub.calls.append(cmd)
            return ('[BROWSER_OK] {"status": 200, "url": "http://x/", '
                    '"title": "T", "text": "body", "length": 4, '
                    '"truncated": false}\n', 0)

        stub.execute = _execute
        return stub

    @pytest.mark.asyncio
    async def test_navigate_SHORT_CIRCUITS_without_touching_the_sandbox(
            self, tmp_path):
        """THE behaviour the whole fix exists for: a second navigation to
        a dead onion must not reach Tor at all."""
        from ghost_agent.tools.browser import tool_browser
        sb = self._sandbox()
        _strike(_KEYBASE)
        out = await tool_browser(operation="navigate", url=_KEYBASE,
                                 sandbox_dir=tmp_path, sandbox_manager=sb)
        assert not sb.calls, (
            "the dead onion was re-dialled — the short-circuit is not wired")
        assert "UNREACHABLE" in out and "DIFFERENT result" in out

    @pytest.mark.asyncio
    async def test_a_LIVE_onion_is_NOT_short_circuited(self, tmp_path):
        """The guard must not block everything — that would satisfy the
        test above while breaking the browser."""
        from ghost_agent.tools.browser import tool_browser
        sb = self._sandbox()
        await tool_browser(operation="navigate", url=_LIVE,
                           sandbox_dir=tmp_path, sandbox_manager=sb)
        assert len(sb.calls) == 1

    @pytest.mark.asyncio
    async def test_an_INTERACT_goto_to_a_dead_onion_is_refused(self,
                                                               tmp_path):
        """R1 J2: the memo was navigate-only, so the identical-retry loop
        survived through `interact` — the very operation the OTHER failure
        hint tells the model to switch to."""
        from ghost_agent.tools.browser import tool_browser
        sb = self._sandbox()
        _strike(_KEYBASE)
        out = await tool_browser(
            operation="interact",
            actions=[{"action": "goto", "url": _KEYBASE},
                     {"action": "extract_text"}],
            sandbox_dir=tmp_path, sandbox_manager=sb)
        assert not sb.calls
        assert "UNREACHABLE" in out

    @pytest.mark.asyncio
    async def test_an_interact_TOR_FAILURE_is_recorded(self, tmp_path):
        """The other half of J2: a goto that fails at the Tor layer is
        reported INSIDE a [BROWSER_OK] payload under the key "actions"
        (not "results" — that is only the runner's local variable), so
        `ok` is True and the failure branch never sees it."""
        from unittest.mock import MagicMock
        from ghost_agent.tools.browser import tool_browser
        import json as _json
        payload = {"actions": [{"index": 0, "action": "goto", "ok": False,
                                "error": "Error: Page.goto: "
                                         "net::ERR_SOCKS_CONNECTION_FAILED",
                                "url": _KEYBASE, "aborted_sequence": True}],
                   "aborted": True, "final_url": _KEYBASE,
                   "final_title": ""}
        sb = MagicMock()
        sb.execute = lambda cmd, timeout=300, **kw: (
            "[BROWSER_OK] " + _json.dumps(payload) + "\n", 0)
        for _ in range(B._DEAD_ONION_STRIKES):
            await tool_browser(operation="interact",
                               actions=[{"action": "goto", "url": _KEYBASE}],
                               sandbox_dir=tmp_path, sandbox_manager=sb)
        assert B._dead_onion_notice(_KEYBASE) is not None, (
            "an interact-reported Tor failure never reached the memo")

    @pytest.mark.asyncio
    async def test_the_breaker_stops_a_failing_engine_being_QUERIED(self):
        """Pin the guard's EFFECT, not its source text. The earlier
        `i_check < i_wait` index assertion was satisfied by text position
        and survived `if False and _breaker_should_skip(...)`."""
        from unittest.mock import patch
        eng = {"name": "ahmia", "url": "https://ahmia.fi/search/?q={q}",
               "index": "ahmia"}
        for _ in range(D._ENGINE_BREAKER_FAILS):
            D._breaker_record("ahmia", False)
        with patch.object(D, "_fetch_raw_html") as fetch:
            out = await D._query_engine(eng, "q", "socks5://127.0.0.1:9050")
        assert out == []
        assert fetch.call_count == 0, (
            "the engine was queried despite an open breaker — the deadline "
            "is still being paid, which is the entire cost this saves")

    @pytest.mark.asyncio
    async def test_an_engine_returning_NOTHING_advances_the_breaker(self):
        """R1 M11/J5: `_breaker_record(..., False)` on the timeout branch
        was deletable with the suite green — and every one of ahmia's 15
        measured failures was a timeout. Drive the real function."""
        from unittest.mock import patch
        eng = {"name": "torgle", "url": "http://x.onion/s?q={q}",
               "index": "torgle"}
        with patch.object(D, "_fetch_raw_html", return_value=(200, "<html/>")):
            for _ in range(D._ENGINE_BREAKER_FAILS):
                await D._query_engine(eng, "q", "socks5://127.0.0.1:9050")
        assert D._breaker_should_skip("torgle"), (
            "an engine that returned nothing every time never advanced "
            "toward open")

    @pytest.mark.asyncio
    async def test_a_REAL_navigate_failure_feeds_the_memo(self, tmp_path):
        """R1 M2: `_mark_onion_dead` on the failure path was deletable
        with the suite green, because every other test strikes the memo
        by calling the helper directly. Drive a real runner failure."""
        from unittest.mock import MagicMock
        from ghost_agent.tools.browser import tool_browser
        sb = MagicMock()
        sb.execute = lambda cmd, timeout=300, **kw: (
            "[BROWSER_ERR] Error: Page.goto: "
            "net::ERR_SOCKS_CONNECTION_FAILED at " + _KEYBASE + "\n", 1)
        for _ in range(B._DEAD_ONION_STRIKES):
            out = await tool_browser(operation="navigate", url=_KEYBASE,
                                     sandbox_dir=tmp_path,
                                     sandbox_manager=sb)
        assert B._dead_onion_notice(_KEYBASE) is not None, (
            "a real Tor-layer navigate failure never reached the memo")
        # …and the LAST reply must carry the directive, not the generic
        # browser advice that invited the identical retry.
        assert "DIFFERENT result" in out
        assert "timeout_ms" not in out

    @pytest.mark.asyncio
    async def test_a_TIMED_OUT_engine_advances_the_breaker(self,
                                                           monkeypatch):
        """R1 M11: deleting `_breaker_record(..., False)` from the
        TimeoutError branch left the suite green — and every one of
        ahmia's 15 measured failures was a timeout, i.e. the fix's
        headline scenario was the unpinned one."""
        import asyncio as _aio
        eng = {"name": "ahmia", "url": "https://ahmia.fi/search/?q={q}",
               "index": "ahmia"}
        monkeypatch.setattr(D, "_ONION_ENGINE_DEADLINE", 0.05)
        monkeypatch.setattr(D, "_FORM_TOKEN_TIMEOUT", 0.01)

        async def _hang(*a, **k):
            await _aio.sleep(5)
            return (200, "")

        monkeypatch.setattr(D, "_fetch_raw_html", _hang)
        for _ in range(D._ENGINE_BREAKER_FAILS):
            assert await D._query_engine(
                eng, "q", "socks5://127.0.0.1:9050") == []
        assert D._breaker_should_skip("ahmia"), (
            "an engine that timed out every time never advanced toward "
            "open — which is exactly how ahmia failed")


class TestABreakerSkippedSearchDoesNotBlameTheQuery:
    """R1 C2/J4. With every engine in cooldown the tool returned, in
    0.000s having contacted nobody: "(b) the query was too specific". That
    is verbatim the misattribution this module documents fixing once
    already — "a broken engine looked identical to 'no onion index has
    this query,' so the tool blamed the query"."""

    @pytest.mark.asyncio
    async def test_all_engines_skipped_reports_INFRASTRUCTURE_not_query(
            self, monkeypatch):
        from ghost_agent.tools.darkweb_search import tool_darkweb_search
        for e in D._load_engines():
            for _ in range(D._ENGINE_BREAKER_FAILS):
                D._breaker_record(e["name"], False)
        out = await tool_darkweb_search("q", tor_proxy="socks5://127.0.0.1:9050")
        assert "NO engines" in out, out[:200]
        assert "cooldown" in out
        # the crucial negative: it must NOT blame the query
        assert "too specific" not in out
        assert "do not reword" in out.lower()

    @pytest.mark.asyncio
    async def test_a_PARTIAL_skip_is_declared_in_the_header(self, tmp_path,
                                                            monkeypatch):
        """A skipped engine used to be indistinguishable from an
        attempted-and-fruitless one, and when the skipped engine was the
        corroborating one the ranking silently degenerated to discovery
        order while still LOOKING ranked."""
        from ghost_agent.tools.darkweb_search import tool_darkweb_search
        real = "a" * 56
        html = f'<a href="http://{real}.onion/">Result</a>'

        async def _fetch(url, proxy, timeout, **kw):
            return (200, html) if "xmh57jrk" in url else (200, "")

        monkeypatch.setattr(D, "_fetch_raw_html", _fetch)
        for _ in range(D._ENGINE_BREAKER_FAILS):
            D._breaker_record("ahmia", False)
        out = await tool_darkweb_search("q", tor_proxy="socks5://127.0.0.1:9050")
        assert "NARROWED" in out, out[:300]
        assert "ahmia" in out


# ──────────────────────────────────────────────────────────────────────
# R2 fixes: the ones the R1 fixes broke or missed
# ──────────────────────────────────────────────────────────────────────

class TestConcurrentSearchesDoNotContaminateEachOther:
    """R2 C1. The skip list was a module global cleared per search, and
    searches overlap routinely — `core/agent.py` dispatches a tool batch
    through `asyncio.gather`, and `tool_darkweb_research` runs its own.
    Measured before the fix: search A skipped two engines and reported
    nothing while B cleared the list, and a search that contacted EVERY
    engine reported "ran NO engines… do not reword and retry" — a
    fabricated, action-foreclosing diagnosis, worse than the silence."""

    @pytest.mark.asyncio
    async def test_two_overlapping_searches_report_their_OWN_skips(
            self, monkeypatch):
        import asyncio as _aio
        from ghost_agent.tools.darkweb_search import tool_darkweb_search
        real = "b" * 56
        html = f'<a href="http://{real}.onion/">R</a>'

        async def _fetch(url, proxy, timeout, **kw):
            await _aio.sleep(0.01)
            return (200, html) if "xmh57jrk" in url else (200, "")

        monkeypatch.setattr(D, "_fetch_raw_html", _fetch)
        for _ in range(D._ENGINE_BREAKER_FAILS):
            D._breaker_record("ahmia", False)

        a, b = await _aio.gather(
            tool_darkweb_search("alpha", tor_proxy="socks5://127.0.0.1:9050"),
            tool_darkweb_search("beta", tor_proxy="socks5://127.0.0.1:9050"))
        # BOTH ran with ahmia skipped, so BOTH must say so.
        assert "NARROWED" in a, a[:200]
        assert "NARROWED" in b, b[:200]

    @pytest.mark.asyncio
    async def test_a_search_that_CONTACTED_engines_never_claims_it_did_not(
            self, monkeypatch):
        from ghost_agent.tools.darkweb_search import tool_darkweb_search
        real = "c" * 56

        async def _fetch(url, proxy, timeout, **kw):
            return (200, f'<a href="http://{real}.onion/">R</a>')

        monkeypatch.setattr(D, "_fetch_raw_html", _fetch)
        out = await tool_darkweb_search("q", tor_proxy="socks5://127.0.0.1:9050")
        assert "NO engines" not in out
        assert "do not reword" not in out.lower()


class TestBothCallersBehaveTheSame:
    """R2 J1: the fix landed on `tool_darkweb_search` only, and
    `tool_darkweb_research` — the follow-up the other tool's own
    description recommends — kept blaming the query."""

    @pytest.mark.asyncio
    async def test_research_also_reports_infrastructure_not_query(self):
        from ghost_agent.tools.darkweb_search import tool_darkweb_research
        for e in D._load_engines():
            for _ in range(D._ENGINE_BREAKER_FAILS):
                D._breaker_record(e["name"], False)
        out = await tool_darkweb_research(
            "q", tor_proxy="socks5://127.0.0.1:9050")
        assert "NO engines" in out, out[:200]
        assert "too specific" not in out


class TestStrikesAreCONSECUTIVE:
    """R2 J2: with no success path, "2 strikes" meant "any 2 failures
    inside the TTL". Measured: fail, load ten pages fine, fail once more
    → declared offline, and the model told "other onion sites load fine…
    retrying will fail the same way" about a host it just read."""

    @pytest.mark.asyncio
    async def test_a_SUCCESS_between_failures_resets_the_streak(
            self, tmp_path):
        from unittest.mock import MagicMock
        from ghost_agent.tools.browser import tool_browser
        import json as _json
        ok_sb = MagicMock()
        ok_sb.execute = lambda cmd, timeout=300, **kw: (
            "[BROWSER_OK] " + _json.dumps(
                {"status": 200, "url": _KEYBASE, "title": "T",
                 "text": "body", "length": 4, "truncated": False}) + "\n", 0)
        bad_sb = MagicMock()
        bad_sb.execute = lambda cmd, timeout=300, **kw: (
            "[BROWSER_ERR] net::ERR_SOCKS_CONNECTION_FAILED\n", 1)

        await tool_browser(operation="navigate", url=_KEYBASE,
                           sandbox_dir=tmp_path, sandbox_manager=bad_sb)
        for _ in range(10):
            await tool_browser(operation="navigate", url=_KEYBASE,
                               sandbox_dir=tmp_path, sandbox_manager=ok_sb)
        await tool_browser(operation="navigate", url=_KEYBASE,
                           sandbox_dir=tmp_path, sandbox_manager=bad_sb)
        assert B._dead_onion_notice(_KEYBASE) is None, (
            "declared offline despite ten proven loads in between — the "
            "count is cumulative, not consecutive")


class TestTheMemosAreBounded:
    """R2 M4: swept only for the host being queried, so a long-lived
    process meeting thousands of one-off dead onions grew without bound
    (5000 entries ≈ 1 MB). This codebase caps its other in-process
    caches."""

    def test_strikes_and_bans_are_capped(self):
        for i in range(B._ONION_MEMO_MAX + 200):
            host = f"http://{'a' * 50}{i:06d}.onion/"
            B._mark_onion_dead(host, "net::ERR_SOCKS_CONNECTION_FAILED")
        assert len(B._ONION_STRIKES) <= B._ONION_MEMO_MAX
        for i in range(B._ONION_MEMO_MAX + 200):
            host = f"http://{'b' * 50}{i:06d}.onion/"
            for _ in range(B._DEAD_ONION_STRIKES):
                B._mark_onion_dead(host, "net::ERR_SOCKS_CONNECTION_FAILED")
        assert len(B._DEAD_ONIONS) <= B._ONION_MEMO_MAX


class TestTheNarrowedBannerIsNotCACHED:
    """R2 M6: the banner describes a transient breaker state; baking it
    into a 5-minute cache entry kept announcing an engine as "not
    consulted" after it had recovered."""

    @pytest.mark.asyncio
    async def test_a_recovered_engine_stops_being_announced(self,
                                                            monkeypatch):
        from ghost_agent.tools.darkweb_search import tool_darkweb_search
        real = "d" * 56

        async def _fetch(url, proxy, timeout, **kw):
            return ((200, f'<a href="http://{real}.onion/">R</a>')
                    if "xmh57jrk" in url else (200, ""))

        monkeypatch.setattr(D, "_fetch_raw_html", _fetch)
        for _ in range(D._ENGINE_BREAKER_FAILS):
            D._breaker_record("ahmia", False)
        first = await tool_darkweb_search("same-q",
                                          tor_proxy="socks5://127.0.0.1:9050")
        assert "NARROWED" in first
        D._ENGINE_BREAKER.clear()                 # engine recovers
        second = await tool_darkweb_search("same-q",
                                           tor_proxy="socks5://127.0.0.1:9050")
        assert "NARROWED" not in second, (
            "a cached banner keeps reporting a cooldown that has ended")


class TestOnlyShapesTheRUNNERCanActuallyEmit:
    """⚠ THREE TESTS WERE DELETED HERE (R4), and the reason is the most
    useful thing in this file.

    They asserted that a non-goto action failing at the Tor layer would
    be attributed to `final_url`. Driven against REAL Tor and the shipped
    `_runner_script`, that payload does not exist:

        click a link to a dead onion, then extract_text
          → [BROWSER_OK], step ok=True, NO error at all,
            final_url = 'chrome-error://chromewebdata/'
        explicit goto to a dead onion at index 1
          → step ok=False, error=ERR_SOCKS_CONNECTION_FAILED,
            url = the dead onion (it carries its OWN url),
            final_url = the PREVIOUS, WORKING page

    So a SOCKS failure never commits a document: `final_url` is either an
    error page that parses to no host, or a live page that would be
    blamed for another host's death. The fallback had no correct outcome
    and is gone. The tests that "proved" it worked were hand-writing a
    payload Chromium cannot produce — verifying the code against a
    fiction, which is worse than not testing it.

    What remains true and IS tested below: the only action that can fail
    a navigation is `goto`, and it always carries its own `url`.
    """

    def _sb(self, payload):
        from unittest.mock import MagicMock
        import json as _json
        sb = MagicMock()
        sb.execute = lambda cmd, timeout=300, **kw: (
            "[BROWSER_OK] " + _json.dumps(payload) + "\n", 0)
        return sb

    @pytest.mark.asyncio
    async def test_a_failing_goto_STEP_is_recorded_from_its_own_url(
            self, tmp_path):
        """The measured mid-sequence shape: ok=False, its own `url`."""
        from ghost_agent.tools.browser import tool_browser
        payload = {"actions": [
            {"index": 0, "action": "goto", "ok": True, "url": _LIVE,
             "title": "T"},
            {"index": 1, "action": "goto", "ok": False,
             "error": "Error: Page.goto: net::ERR_SOCKS_CONNECTION_FAILED",
             "url": _KEYBASE, "aborted_sequence": True}],
            "aborted": True, "final_url": _LIVE, "final_title": ""}
        sb = self._sb(payload)
        for _ in range(B._DEAD_ONION_STRIKES):
            await tool_browser(
                operation="interact",
                actions=[{"action": "goto", "url": _LIVE},
                         {"action": "goto", "url": _KEYBASE}],
                sandbox_dir=tmp_path, sandbox_manager=sb)
        assert B._dead_onion_notice(_KEYBASE) is not None
        # …and the live page the sequence was ON must not be blamed.
        assert B._dead_onion_notice(_LIVE) is None

    @pytest.mark.asyncio
    async def test_a_chrome_error_final_url_blames_NOBODY(self, tmp_path):
        """The measured click shape. Nothing is recorded — which is
        correct: the payload does not identify a host."""
        from ghost_agent.tools.browser import tool_browser
        payload = {"actions": [
            {"index": 0, "action": "click", "ok": True},
            {"index": 1, "action": "extract_text", "ok": True,
             "text": "err"}],
            "aborted": False,
            "final_url": "chrome-error://chromewebdata/",
            "final_title": ""}
        sb = self._sb(payload)
        for _ in range(B._DEAD_ONION_STRIKES + 2):
            await tool_browser(
                operation="interact", url=_LIVE,
                actions=[{"action": "click", "selector": "#a"},
                         {"action": "extract_text"}],
                sandbox_dir=tmp_path, sandbox_manager=sb)
        assert B._dead_onion_notice(_LIVE) is None, (
            "a chrome-error page was attributed to the live host the "
            "sequence started from")


class TestTheGuardsThatSurvivedMutation:
    """R3 survivors. Each of these was a real fix with no test — the
    mutation stayed green, which is the only way to find them."""

    def test_the_cap_evicts_OLDEST_not_newest(self):
        """R3 M29: reversing `_cap`'s sort discards every fresh ban on
        arrival — the memo goes silently dead while the suite passes.
        Measured on the mutant: "new host kept? False"."""
        B._DEAD_ONIONS.clear()
        for i in range(B._ONION_MEMO_MAX + 50):
            B._DEAD_ONIONS[f"h{i:05d}.onion"] = float(i)
        B._cap(B._DEAD_ONIONS, lambda h: B._DEAD_ONIONS[h])
        assert len(B._DEAD_ONIONS) == B._ONION_MEMO_MAX
        newest = f"h{B._ONION_MEMO_MAX + 49:05d}.onion"
        assert newest in B._DEAD_ONIONS, "the newest entry was evicted"
        assert "h00000.onion" not in B._DEAD_ONIONS, "the oldest survived"

    def test_the_cap_size_is_PINNED(self):
        """R3 M26: `assert len(...) <= _ONION_MEMO_MAX` adapts to any
        value — a cap of 4 passed it."""
        assert B._ONION_MEMO_MAX == 512

    def test_the_memo_kill_switch_works_on_the_READ_path(self,
                                                         monkeypatch):
        """R3 M27: dropping the guard from `_dead_onion_notice` left the
        suite green, but the read path is what makes a false positive
        RECOVERABLE without a restart — the switch's entire purpose."""
        _strike(_KEYBASE)
        assert B._dead_onion_notice(_KEYBASE) is not None
        monkeypatch.setenv("GHOST_DEAD_ONION_MEMO", "0")
        assert B._dead_onion_notice(_KEYBASE) is None, (
            "an already-banned host stays banned with the memo disabled")

    def test_the_breaker_kill_switch_stops_it_COUNTING(self, monkeypatch):
        """R3 M22: the existing test asserted through
        `_breaker_should_skip`, whose own guard forces the answer — the
        evidence was identical under both hypotheses. Assert the counter
        itself never moves."""
        monkeypatch.setenv("GHOST_ONION_BREAKER", "0")
        for _ in range(D._ENGINE_BREAKER_FAILS * 2):
            D._breaker_record("ahmia", False)
        assert "ahmia" not in D._ENGINE_BREAKER, (
            "the breaker kept counting while disabled — and would then "
            "announce a skip that never happens")

    def test_conftest_ALONE_resets_the_strike_state(self):
        """R3 M17: this file used to carry its own reset fixture, which
        masked conftest's. Removing it means THIS test is the one that
        fails if conftest stops clearing `_ONION_STRIKES` — protecting
        every other browser test that drives a SOCKS failure."""
        assert not B._ONION_STRIKES, (
            "strike state leaked in from a previous test — conftest is "
            "not clearing it")
        B._mark_onion_dead(_KEYBASE, "net::ERR_SOCKS_CONNECTION_FAILED")


class TestTheR3Majors:
    """All four were the same shape: the fix landed on the path its test
    drove and stopped at the edge of it."""

    def _sb(self, payload, exit_code=0, err=None):
        from unittest.mock import MagicMock
        import json as _json
        sb = MagicMock()
        out = (err if err is not None
               else "[BROWSER_OK] " + _json.dumps(payload) + "\n")
        sb.execute = lambda cmd, timeout=300, **kw: (out, exit_code)
        return sb

    @pytest.mark.asyncio
    async def test_a_failed_interact_never_bans_the_UNDIALLED_top_url(
            self, tmp_path):
        """R3 MAJOR, the inversion: with actions[0] a goto, the runner
        never dials the top-level url — but the failure branch banned it,
        so the HEALTHY host was blocked for 600s and the dead one was
        left alone. The only defect that made things worse than before."""
        from ghost_agent.tools.browser import tool_browser
        sb = self._sb(None, exit_code=1,
                      err="[BROWSER_ERR] net::ERR_SOCKS_CONNECTION_FAILED\n")
        for _ in range(B._DEAD_ONION_STRIKES + 1):
            await tool_browser(
                operation="interact", url=_LIVE,
                actions=[{"action": "goto", "url": _KEYBASE}],
                sandbox_dir=tmp_path, sandbox_manager=sb)
        assert B._dead_onion_notice(_LIVE) is None, (
            "banned the host the runner never dialled")

    @pytest.mark.asyncio
    async def test_a_TRAILING_action_does_not_erase_the_strike(self,
                                                               tmp_path):
        """R3 MAJOR: any successful urlless step marked `final_url`
        alive, so a trailing extract_text wiped the failure recorded
        moments earlier. Rebuilt on the shape the runner really emits: a
        failing goto (own url) followed by a successful trailing step."""
        from ghost_agent.tools.browser import tool_browser
        payload = {"actions": [
            {"index": 0, "action": "goto", "ok": False,
             "error": "Error: net::ERR_SOCKS_CONNECTION_FAILED",
             "url": _KEYBASE, "aborted_sequence": True},
            {"index": 1, "action": "extract_text", "ok": True,
             "text": "x"}],
            "aborted": True, "final_url": _LIVE, "final_title": ""}
        sb = self._sb(payload)
        for _ in range(B._DEAD_ONION_STRIKES):
            await tool_browser(
                operation="interact",
                actions=[{"action": "goto", "url": _KEYBASE},
                         {"action": "extract_text"}],
                sandbox_dir=tmp_path, sandbox_manager=sb)
        assert B._dead_onion_notice(_KEYBASE) is not None, (
            "a trailing no-network action erased the strike")

    @pytest.mark.asyncio
    async def test_a_dead_hop_at_index_N_does_not_discard_steps_0_to_N(
            self, tmp_path):
        """R3 MAJOR: the pre-check refused the whole sequence for a dead
        goto at ANY index, so a ten-step sequence whose step 8 was dead
        lost steps 0-7 as well. Only the FIRST navigating hop is refused;
        a later one is left to the runner, which aborts and reports."""
        from ghost_agent.tools.browser import tool_browser
        from unittest.mock import MagicMock
        import json as _json
        payload = {"actions": [], "aborted": False, "final_url": _LIVE}
        calls = []
        sb = MagicMock()

        def _exec(cmd, timeout=300, **kw):
            calls.append(cmd)
            return "[BROWSER_OK] " + _json.dumps(payload) + "\n", 0

        sb.execute = _exec
        _strike(_KEYBASE)
        await tool_browser(
            operation="interact",
            actions=[{"action": "goto", "url": _LIVE},
                     {"action": "extract_text"},
                     {"action": "goto", "url": _KEYBASE}],
            sandbox_dir=tmp_path, sandbox_manager=sb)
        assert calls, (
            "the whole sequence was refused because step 2 was dead — "
            "steps 0 and 1 were fine and never ran")

    @pytest.mark.asyncio
    async def test_a_dead_FIRST_hop_is_still_refused(self, tmp_path):
        """The narrowing must not disable the guard it was narrowing."""
        from unittest.mock import MagicMock
        from ghost_agent.tools.browser import tool_browser
        calls = []
        sb = MagicMock()
        sb.execute = lambda cmd, timeout=300, **kw: (
            calls.append(cmd), ("[BROWSER_OK] {}\n", 0))[1]
        _strike(_KEYBASE)
        out = await tool_browser(
            operation="interact",
            actions=[{"action": "goto", "url": _KEYBASE},
                     {"action": "extract_text"}],
            sandbox_dir=tmp_path, sandbox_manager=sb)
        assert not calls
        assert "UNREACHABLE" in out

    @pytest.mark.asyncio
    async def test_RESEARCH_also_declares_a_narrowed_search(self,
                                                            monkeypatch):
        """R3 MAJOR: research is the tool whose ranking decides which
        onions get deep-read and synthesised — and it was the one that
        never said corroboration had been weakened."""
        from ghost_agent.tools.darkweb_search import tool_darkweb_research
        real = "g" * 56

        async def _fetch(url, proxy, timeout, **kw):
            return ((200, f'<a href="http://{real}.onion/">R</a>')
                    if "xmh57jrk" in url else (200, ""))

        monkeypatch.setattr(D, "_fetch_raw_html", _fetch)
        monkeypatch.setattr(D, "helper_fetch_url_content",
                            lambda *a, **k: "content", raising=False)
        for _ in range(D._ENGINE_BREAKER_FAILS):
            D._breaker_record("ahmia", False)
        out = await tool_darkweb_research(
            "q", tor_proxy="socks5://127.0.0.1:9050")
        assert "NARROWED" in out, out[:300]


class TestAPartialSkipIsAWeakNegative:
    """R4 CRITICAL, and the state production actually lives in: this
    module's own measurements put both ahmia endpoints at 0 wins, so they
    sit in cooldown while torch and torgle carry the search. When those
    two also return nothing, the old text claimed "ZERO results across
    all onion search engines and circuits" — false, half were never
    contacted — and prescribed "drop to 2-4 PLAIN keywords"."""

    async def _empty_search(self, monkeypatch, fn, skip_names):
        async def _fetch(url, proxy, timeout, **kw):
            return (200, "")          # every contacted engine finds nothing
        monkeypatch.setattr(D, "_fetch_raw_html", _fetch)
        for name in skip_names:
            for _ in range(D._ENGINE_BREAKER_FAILS):
                D._breaker_record(name, False)
        return await fn("q", tor_proxy="socks5://127.0.0.1:9050")

    @pytest.mark.asyncio
    async def test_search_declares_partial_coverage(self, monkeypatch):
        from ghost_agent.tools.darkweb_search import tool_darkweb_search
        out = await self._empty_search(monkeypatch, tool_darkweb_search,
                                       ["ahmia", "ahmia-onion"])
        assert "WEAK negative" in out, out[:250]
        assert "were NOT contacted" in out
        assert "ahmia" in out
        # the crucial negative: it must not prescribe rewording
        assert "too specific" not in out
        assert "drop to 2-4 PLAIN keywords" not in out

    @pytest.mark.asyncio
    async def test_research_declares_partial_coverage_too(self,
                                                          monkeypatch):
        from ghost_agent.tools.darkweb_search import tool_darkweb_research
        out = await self._empty_search(monkeypatch, tool_darkweb_research,
                                       ["ahmia", "ahmia-onion"])
        assert "WEAK negative" in out, out[:250]
        assert "too specific" not in out

    @pytest.mark.asyncio
    async def test_a_FULL_coverage_miss_still_reads_as_a_real_negative(
            self, monkeypatch):
        """The honesty cuts both ways: with every engine contacted, zero
        results IS a statement about the query, and the message must not
        be watered down into a coverage excuse."""
        from ghost_agent.tools.darkweb_search import tool_darkweb_search
        out = await self._empty_search(monkeypatch, tool_darkweb_search, [])
        assert "WEAK negative" not in out
        assert "NO engines" not in out


class TestTheRunnerRuleIsMirroredExactly:
    """R4 MAJOR: the host-side "will the runner dial the top-level url?"
    test disagreed with the runner in BOTH directions, and both mutants
    survived the whole suite."""

    def test_navigate_is_NOT_a_goto_alias(self, tmp_path):
        """The runner's interact dispatch is goto-only; `navigate` as an
        action means the runner still dials the TOP-LEVEL url. The host
        treated it as a goto and skipped the check entirely."""
        got = B._runner_first_url(
            "interact", _KEYBASE,
            [{"action": "navigate", "url": _LIVE}], tmp_path)
        assert got == _KEYBASE

    def test_a_goto_with_NO_url_does_not_shield_the_top_level_url(self,
                                                                  tmp_path):
        """The runner keys on the action alone. A urlless goto means it
        dials nothing at the top level — so blaming the top-level url is
        a ban on a host that was never contacted."""
        got = B._runner_first_url(
            "interact", _KEYBASE, [{"action": "goto"}], tmp_path)
        assert got == ""

    def test_a_first_goto_WITH_a_url_is_what_gets_dialled(self, tmp_path):
        got = B._runner_first_url(
            "interact", _LIVE,
            [{"action": "goto", "url": _KEYBASE}], tmp_path)
        assert got == _KEYBASE

    def test_the_SIDECAR_url_is_seen(self, tmp_path):
        """R4 MAJOR: urlless atomic ops resolve their target from the
        `.last_url` sidecar, and the memo was blind to that entire path —
        `navigate(A)` then `click`/`extract_text` with no url is the flow
        this tool's own docstring teaches."""
        prof = tmp_path / B._BROWSER_PROFILE_DIR
        prof.mkdir(parents=True, exist_ok=True)
        (prof / ".last_url").write_text(_KEYBASE, encoding="utf-8")
        assert B._runner_first_url("extract_text", None, None,
                                   tmp_path) == _KEYBASE
        assert B._runner_first_url("click", None, None,
                                   tmp_path) == _KEYBASE

    @pytest.mark.asyncio
    async def test_a_urlless_op_on_a_dead_sidecar_host_is_REFUSED(
            self, tmp_path):
        from unittest.mock import MagicMock
        from ghost_agent.tools.browser import tool_browser
        prof = tmp_path / B._BROWSER_PROFILE_DIR
        prof.mkdir(parents=True, exist_ok=True)
        (prof / ".last_url").write_text(_KEYBASE, encoding="utf-8")
        _strike(_KEYBASE)
        calls = []
        sb = MagicMock()
        sb.execute = lambda cmd, timeout=300, **kw: (
            calls.append(cmd), ("[BROWSER_OK] {}\n", 0))[1]
        out = await tool_browser(operation="extract_text",
                                 sandbox_dir=tmp_path, sandbox_manager=sb)
        assert not calls, "re-dialled a banned host through the sidecar"
        assert "UNREACHABLE" in out

    @pytest.mark.asyncio
    async def test_the_MEASURED_click_failure_blames_nobody(self,
                                                             tmp_path):
        """⚠ THE FOURTH FICTION, and the honest note about it.

        This test used to hand-write `{"action":"click","ok":False,
        "error":"…ERR_SOCKS_CONNECTION_FAILED"}` — and the class docstring
        above records that Chromium does not emit it: a click to a dead
        onion returns ok=True, NO error, `final_url =
        chrome-error://chromewebdata/`. It was the sole guard on removing
        the `final_url` fallback, so it could only fail on a payload that
        cannot exist.

        Rewritten to the MEASURED shape. That means restoring the
        fallback is no longer caught by any test — and that is the honest
        state, not a gap to paper over: under real payloads the fallback
        is unreachable (the only action that can fail a navigation,
        `goto`, always carries its own `url`, and non-goto errors never
        contain a SOCKS marker, so `_ONION_UNREACHABLE_MARKERS` filters
        them). A test that can only fail on an impossible input is worse
        than no test."""
        from unittest.mock import MagicMock
        from ghost_agent.tools.browser import tool_browser
        import json as _json
        payload = {"actions": [
            {"index": 0, "action": "click", "ok": True},
            {"index": 1, "action": "extract_text", "ok": True,
             "text": "error page"}],
            "aborted": False,
            "final_url": "chrome-error://chromewebdata/",
            "final_title": ""}
        sb = MagicMock()
        sb.execute = lambda cmd, timeout=300, **kw: (
            "[BROWSER_OK] " + _json.dumps(payload) + "\n", 0)
        for _ in range(B._DEAD_ONION_STRIKES + 2):
            await tool_browser(
                operation="interact", url=_LIVE,
                actions=[{"action": "click", "selector": "#a"},
                         {"action": "extract_text"}],
                sandbox_dir=tmp_path, sandbox_manager=sb)
        assert B._dead_onion_notice(_LIVE) is None
        assert not B._DEAD_ONIONS, "a chrome-error page condemned a host"

    @pytest.mark.asyncio
    async def test_an_IMPLICIT_first_nav_does_not_refuse_a_later_goto(
            self, tmp_path):
        """R4 MAJOR: when actions[0] is NOT a goto the runner navigates
        implicitly first, so a later explicit goto is not the first hop.
        Refusing it discarded every step before it — and `[click, goto]`
        is an ordinary shape."""
        from unittest.mock import MagicMock
        from ghost_agent.tools.browser import tool_browser
        import json as _json
        calls = []
        sb = MagicMock()

        def _exec(cmd, timeout=300, **kw):
            calls.append(cmd)
            return ("[BROWSER_OK] " + _json.dumps(
                {"actions": [], "aborted": False, "final_url": _LIVE}) +
                "\n", 0)

        sb.execute = _exec
        _strike(_KEYBASE)
        await tool_browser(
            operation="interact", url=_LIVE,
            actions=[{"action": "click", "selector": "#a"},
                     {"action": "goto", "url": _KEYBASE}],
            sandbox_dir=tmp_path, sandbox_manager=sb)
        assert calls, (
            "the sequence was refused for a dead goto at index 1, "
            "discarding the click that came before it")


class TestTheR5Edges:
    """R5: both findings were edges of `_runner_first_url` — the one
    function R4 rewrote. Same pattern as every round: the fix landed on
    the path its test drove (unscoped sandbox, navigating ops) and
    stopped at that path's edge (scoped sandbox, non-navigating ops)."""

    def _sidecar(self, root, url):
        prof = root / B._BROWSER_PROFILE_DIR
        prof.mkdir(parents=True, exist_ok=True)
        (prof / ".last_url").write_text(url, encoding="utf-8")

    def test_the_sidecar_is_found_from_a_PROJECT_SCOPED_dir(self,
                                                            tmp_path):
        """R5 MAJOR: the runner's profile is hardcoded to
        `/workspace/.browser_profile` = the sandbox ROOT, but registry.py
        passes the PROJECT-scoped dir whenever a project is active. So in
        a project session — the majority of real work — R4's sidecar fix
        did nothing: the memo neither refused a banned host nor learned
        from the failure."""
        root = tmp_path
        scoped = root / "projects" / "p1"
        scoped.mkdir(parents=True, exist_ok=True)
        self._sidecar(root, _KEYBASE)
        assert B._runner_first_url("extract_text", None, None, scoped) \
            == _KEYBASE, "project-scoped session cannot see the sidecar"
        # …and the unscoped case must keep working.
        assert B._runner_first_url("extract_text", None, None, root) \
            == _KEYBASE

    @pytest.mark.asyncio
    async def test_a_scoped_urlless_op_on_a_banned_host_is_REFUSED(
            self, tmp_path):
        from unittest.mock import MagicMock
        from ghost_agent.tools.browser import tool_browser
        root = tmp_path
        scoped = root / "projects" / "p1"
        scoped.mkdir(parents=True, exist_ok=True)
        self._sidecar(root, _KEYBASE)
        _strike(_KEYBASE)
        calls = []
        sb = MagicMock()
        sb.execute = lambda cmd, timeout=300, **kw: (
            calls.append(cmd), ("[BROWSER_OK] {}\n", 0))[1]
        out = await tool_browser(operation="extract_text",
                                 sandbox_dir=scoped, sandbox_manager=sb)
        assert not calls, "re-dialled a banned host in a project session"
        assert "UNREACHABLE" in out

    @pytest.mark.asyncio
    async def test_CLOSE_is_never_refused(self, tmp_path):
        """R5 MAJOR: `close` only rmtree's the profile — it dials
        nothing. Refusing it meant the memo's own state blocked the one
        operation that clears the sidecar, for the full 600s TTL, with a
        message about a service `close` never contacts."""
        from unittest.mock import MagicMock
        from ghost_agent.tools.browser import tool_browser
        self._sidecar(tmp_path, _KEYBASE)
        _strike(_KEYBASE)
        calls = []
        sb = MagicMock()
        sb.execute = lambda cmd, timeout=300, **kw: (
            calls.append(cmd), ('[BROWSER_OK] {"closed": true}\n', 0))[1]
        out = await tool_browser(operation="close", sandbox_dir=tmp_path,
                                 sandbox_manager=sb)
        assert calls, "close was refused by the memo it would have cleared"
        assert "UNREACHABLE" not in out

    @pytest.mark.asyncio
    async def test_a_urlless_NAVIGATE_gets_a_PARAMETER_error(self,
                                                             tmp_path):
        """`navigate` never consults the sidecar — the runner raises
        'navigate requires url'. Answering a plain parameter mistake with
        'pick a DIFFERENT result from your search' sends the model to
        solve the wrong problem."""
        from unittest.mock import MagicMock
        from ghost_agent.tools.browser import tool_browser
        self._sidecar(tmp_path, _KEYBASE)
        _strike(_KEYBASE)
        sb = MagicMock()
        sb.execute = lambda cmd, timeout=300, **kw: (
            '[BROWSER_ERR] navigate requires url\n', 1)
        out = await tool_browser(operation="navigate", sandbox_dir=tmp_path,
                                 sandbox_manager=sb)
        assert "DIFFERENT result" not in out

    @pytest.mark.asyncio
    async def test_the_refusal_line_NAMES_the_host(self, tmp_path):
        """It interpolated `url`, which is empty on exactly the paths
        this refusal fires for — so the operator saw
        "  is a known-dead hidden service". The live stream is the only
        place a silent skip surfaces at all."""
        from unittest.mock import MagicMock, patch
        from ghost_agent.tools.browser import tool_browser
        self._sidecar(tmp_path, _KEYBASE)
        _strike(_KEYBASE)
        seen = []
        sb = MagicMock()
        sb.execute = lambda cmd, timeout=300, **kw: ("[BROWSER_OK] {}\n", 0)
        with patch("ghost_agent.tools.browser.pretty_log",
                   side_effect=lambda *a, **k: seen.append(a)):
            await tool_browser(operation="extract_text",
                               sandbox_dir=tmp_path, sandbox_manager=sb)
        skipped = [a for a in seen if a and a[0] == "Browser Skipped"]
        assert skipped, "no refusal line at all"
        assert B._onion_host(_KEYBASE)[:16] in skipped[0][1], (
            f"the refusal line does not name the host: {skipped[0][1]!r}")

    @pytest.mark.asyncio
    async def test_the_all_skipped_DECISION_comes_from_the_fan_out(self):
        """R5: the fan-out computes `all_skipped` entry-for-entry and both
        callers discarded it, letting the message re-derive it from
        deduped NAMES vs an entry COUNT — J3's bug, reintroduced. With
        two entries sharing a name and nothing contacted, the old text
        said it 'asked 1 of 3 engines'. It asked none."""
        assert D._no_results_error(["ahmia", "torch"], 3,
                                   all_skipped=True).count("NO engines")
        # and the partial branch still fires when it genuinely is partial
        assert "WEAK negative" in D._no_results_error(["ahmia"], 4,
                                                      all_skipped=False)


class TestTheRunnerIsARealFileNotAString:
    """The structural fix behind §4BP's whole defect class.

    `_runner_script` used to return 1,205 lines of Python as a STRING —
    48% of browser.py. Four defects in one feature came from it, and the
    worst was that a syntax error anywhere in those lines was
    undetectable: measured, breaking `async def op_navigate(op):` left
    py_compile green, import green, and the browser tests green. It would
    have failed only inside the container, on the operator's next browser
    call."""

    def test_the_runner_is_valid_PYTHON(self):
        """The check that could not exist while it was a string."""
        import ast
        ast.parse(B._runner_script())

    def test_the_runner_lives_in_its_own_MODULE_FILE(self):
        """A real file is compiled and linted by every ordinary tool, and
        its names can never be confused with browser.py's — which is what
        made `grep -n "^def …"` lie four times."""
        from pathlib import Path
        f = Path(B.__file__).parent / "browser_runner.py"
        assert f.is_file(), "the runner is not a real module file"
        assert f.read_text(encoding="utf-8") == B._runner_script()

    def test_browser_py_no_longer_CONTAINS_the_runner(self):
        """Guards the regression that matters: pasting the runner back
        inline would restore every trap at once."""
        from pathlib import Path
        src = Path(B.__file__).read_text(encoding="utf-8")
        # Column 0 — this file's own docstring QUOTES that line as the
        # example of what used to be undetectable, and a substring test
        # matched its own explanation.
        assert "\nasync def op_navigate(op):" not in src, (
            "the runner source is inline in browser.py again")
        assert len(src.splitlines()) < 1600, (
            "browser.py grew back toward its pre-extraction size")

    def test_the_module_scope_is_now_UNAMBIGUOUS(self):
        """The concrete symptom: `import os` / `from urllib.parse import`
        used to appear at column 0 inside the string, so module-level
        reasoning about browser.py was unsound."""
        import ast
        from pathlib import Path
        tree = ast.parse(Path(B.__file__).read_text(encoding="utf-8"))
        mod_imports = {a.asname or a.name
                       for n in tree.body
                       if isinstance(n, (ast.Import, ast.ImportFrom))
                       for a in n.names}
        # everything the memo needs, resolvable by reading tree.body alone
        assert {"os", "time", "Dict", "_urlparse", "Path"} <= mod_imports
