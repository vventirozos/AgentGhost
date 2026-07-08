"""Tests for the per-engine Tor circuit race in web search (2026-07-08).

Measured over Tor: per-(engine, circuit) success is ~10% and failures are
driven by the EXIT IP, not the engine — so ddgs's multi-backend mode (all
engines through the one proxy on the DDGS instance) fails all engines
together whenever the exit is blocked. `_race_search_wave` fixes the
correlation: one single-engine ddgs call per engine, each on its own
circuit (per-engine SOCKS salt), first non-empty junk-filtered result wins.

These tests pin the race semantics; the circuit-tag uniqueness itself is
pinned in test_search_tor_hardening.py::test_race_gives_each_engine_its_own_circuit.
"""
import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from src.ghost_agent.tools.search import (
    _RACE_ENGINES,
    _SEARCH_CACHE,
    _race_search_wave,
    tool_search_ddgs,
)


def _mock_ddgs_module(text_side_effect):
    mod = MagicMock()
    cls = MagicMock()
    mod.DDGS = cls
    inst = MagicMock()
    cls.return_value.__enter__.return_value = inst
    inst.text.side_effect = text_side_effect
    return mod, inst


@pytest.mark.asyncio
async def test_slow_valid_engine_beats_fast_junk_and_errors():
    """One engine is slow-but-valid; the others fail fast or return only
    junk-domain results. The wave must wait past the fast losers and hand
    back the slow engine's results — junk/empty/error must never win."""
    def by_engine(q, **kw):
        engine = kw["backend"]
        if engine == "mojeek":
            time.sleep(0.3)  # the measured shape: mojeek slow but real
            return [{"title": "Real", "body": "b", "href": "http://real-site.com/x"}]
        if engine == "brave":
            return [{"title": "Junk", "body": "b", "href": "http://facebook.com/y"}]
        raise Exception("blocked exit")

    mod, inst = _mock_ddgs_module(by_engine)
    with patch.dict("sys.modules", {"ddgs": mod}):
        results = await _race_search_wave("some query", "socks5h://127.0.0.1:9050", 0)

    assert [r["href"] for r in results] == ["http://real-site.com/x"]
    # Every engine was actually raced (no winner existed until mojeek).
    assert inst.text.call_count == len(_RACE_ENGINES)


@pytest.mark.asyncio
async def test_first_valid_winner_short_circuits_wave():
    """A fast valid engine must end the wave without waiting for slow
    siblings: the caller gets the winner's results immediately."""
    def by_engine(q, **kw):
        if kw["backend"] == "brave":
            return [{"title": "Fast", "body": "b", "href": "http://fast-win.com/a"}]
        time.sleep(1.5)
        return [{"title": "Slow", "body": "b", "href": "http://slow-win.com/b"}]

    mod, _ = _mock_ddgs_module(by_engine)
    t0 = time.monotonic()
    with patch.dict("sys.modules", {"ddgs": mod}):
        results = await _race_search_wave("another query", None, 0)
    elapsed = time.monotonic() - t0

    assert [r["href"] for r in results] == ["http://fast-win.com/a"]
    # Won well before the slow engines' 1.5s — the wave did not barrier.
    assert elapsed < 1.2


@pytest.mark.asyncio
async def test_wave_deadline_bounds_a_wedged_engine():
    """An engine thread that outlives the ddgs timeout must not wedge the
    wave: the deadline (timeout + grace) gives up and returns empty."""
    def hang(q, **kw):
        time.sleep(2.0)
        return [{"title": "Too late", "body": "b", "href": "http://late.com/z"}]

    mod, _ = _mock_ddgs_module(hang)
    with patch.dict("sys.modules", {"ddgs": mod}), \
         patch("src.ghost_agent.tools.search._DDGS_TOR_TIMEOUT", 0.2), \
         patch("src.ghost_agent.tools.search._RACE_WAVE_GRACE", 0.2):
        t0 = time.monotonic()
        results = await _race_search_wave("wedged query", None, 0)
        elapsed = time.monotonic() - t0

    assert results == []
    assert elapsed < 1.5  # gave up at ~0.4s, never waited the full 2s


@pytest.mark.asyncio
async def test_failure_log_is_one_terse_categorized_line():
    """The operator stream gets ONE compact line per failed wave — terse
    categories, not exception reprs (operator request 2026-07-08: the
    grouped-repr format was still a wall of text). Boring "No results
    found" collapses to a bare count; engines are named only for
    non-default categories; no URL or repr plumbing appears."""
    def by_engine(q, **kw):
        if kw["backend"] == "yahoo":
            raise Exception(
                "RequestError('error sending request for url "
                "(https://search.yahoo.com/search;_ylt=VERYLONGTOKEN)')")
        raise Exception("No results found.")

    mod, _ = _mock_ddgs_module(by_engine)
    with patch.dict("sys.modules", {"ddgs": mod}), \
         patch("src.ghost_agent.tools.search.pretty_log") as mock_log:
        results = await _race_search_wave("some concurrent query", None, 0)

    assert results == []
    warn_lines = [c.args[1] for c in mock_log.call_args_list
                  if c.kwargs.get("level") == "WARNING"]
    assert len(warn_lines) == 1
    line = warn_lines[0]
    # Terse: the 5 boring empties are a count, yahoo's transport failure a
    # category — and the whole line stays short.
    assert "5 empty" in line
    assert "yahoo conn-error" in line
    assert "‹some concurrent query›" in line
    assert "https://" not in line
    assert "RequestError" not in line
    assert len(line) < 120


@pytest.mark.asyncio
async def test_failure_log_keeps_snippet_for_unknown_errors():
    """A failure shape we haven't categorized must NOT be hidden behind a
    bare category — the line keeps a short snippet of the message."""
    mod, _ = _mock_ddgs_module(Exception("Weird new failure mode xyz"))
    with patch.dict("sys.modules", {"ddgs": mod}), \
         patch("src.ghost_agent.tools.search.pretty_log") as mock_log:
        results = await _race_search_wave("odd query", None, 0)

    assert results == []
    warn_lines = [c.args[1] for c in mock_log.call_args_list
                  if c.kwargs.get("level") == "WARNING"]
    assert len(warn_lines) == 1
    assert "Weird new failure mode xyz" in warn_lines[0]


@pytest.mark.asyncio
async def test_winner_log_carries_query_tag():
    """The win line must carry the query tag too — several searches race
    concurrently in one turn and their log lines interleave."""
    def by_engine(q, **kw):
        return [{"title": "Hit", "body": "b", "href": "http://hit-site.com/x"}]

    mod, _ = _mock_ddgs_module(by_engine)
    with patch.dict("sys.modules", {"ddgs": mod}), \
         patch("src.ghost_agent.tools.search.pretty_log") as mock_log:
        results = await _race_search_wave("tag me query", None, 0)

    assert results
    win_lines = [c.args[1] for c in mock_log.call_args_list if "won wave" in str(c.args[1])]
    assert win_lines and "‹tag me query›" in win_lines[0]


@pytest.mark.asyncio
async def test_stopiteration_from_engine_is_contained():
    """StopIteration must never cross the thread→Future boundary (PEP 479
    poisons the event loop's future chaining). All engines raising it must
    degrade to the normal ZERO-results error, loop intact."""
    mod, inst = _mock_ddgs_module(StopIteration("exhausted"))
    _SEARCH_CACHE.clear()
    with patch.dict("sys.modules", {"ddgs": mod}), \
         patch("importlib.util.find_spec", return_value=True):
        result = await tool_search_ddgs("doomed race query", None)

    assert "ZERO results" in result
    # The loop is still healthy: schedule and await something trivial.
    await asyncio.sleep(0)
    # Waves + reformulation waves all raced the full set.
    assert inst.text.call_count == 4 * len(_RACE_ENGINES)
