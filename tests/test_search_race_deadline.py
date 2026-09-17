"""§4HR — the wave deadline is sized off the engines actually raced.

Seven days of live waves (2026-09-09→16): mojeek 0 wins of 509, and a
no-winner wave cost a median 18 s — mojeek's budget — because a wave ends
only when its LAST engine gives up. Pins: the raced set carries no dead
engine; the deadline derives from the raced set (not the slow-tier
constant); a wave whose engines all fail fast returns well inside it.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.ghost_agent.tools import search as S


def _mock_ddgs_module(text_side_effect):
    mod = MagicMock()
    cls = MagicMock()
    mod.DDGS = cls
    inst = MagicMock()
    cls.return_value.__enter__.return_value = inst
    inst.text.side_effect = text_side_effect
    return mod, inst


def test_the_deadline_is_the_slowest_raced_engine_plus_grace():
    with patch.object(S, "_RACE_ENGINES", ("a", "b")), \
         patch.object(S, "_DDGS_ENGINE_TIMEOUT", {"a": 7}), \
         patch.object(S, "_DDGS_FAST_ENGINE_TIMEOUT", 3), \
         patch.object(S, "_RACE_WAVE_GRACE", 1), \
         patch.object(S, "_DDGS_TOR_TIMEOUT", 99):
        assert S._race_wave_deadline() == 8          # 7 + 1, not 99 + 1
    with patch.object(S, "_RACE_ENGINES", ("b",)), \
         patch.object(S, "_DDGS_ENGINE_TIMEOUT", {"a": 7}), \
         patch.object(S, "_DDGS_FAST_ENGINE_TIMEOUT", 3), \
         patch.object(S, "_RACE_WAVE_GRACE", 1):
        assert S._race_wave_deadline() == 4          # the fast tier bounds it


def test_the_live_deadline_is_the_fast_tier_now():
    """FAILS IF: a slow-tier engine is raced again without re-measuring, or
    the deadline goes back to the slow-tier constant."""
    assert S._race_wave_deadline() == S._DDGS_FAST_ENGINE_TIMEOUT + S._RACE_WAVE_GRACE
    assert S._race_wave_deadline() < S._DDGS_TOR_TIMEOUT + S._RACE_WAVE_GRACE


def test_no_raced_engine_carries_the_slow_budget():
    for e in S._RACE_ENGINES:
        assert S._engine_timeout(e) == S._DDGS_FAST_ENGINE_TIMEOUT, e


@pytest.mark.asyncio
async def test_a_wave_of_fast_failures_ends_when_its_last_engine_does():
    """FAILS IF: something in the wave waits for the deadline after every
    engine has already failed (the mojeek shape: a dead engine holding the
    wave to its budget)."""
    def fail_fast(q, **kw):
        raise ConnectionError("error sending request")

    mod, inst = _mock_ddgs_module(fail_fast)
    with patch.dict("sys.modules", {"ddgs": mod}), \
         patch("src.ghost_agent.tools.search._DDGS_FAST_ENGINE_TIMEOUT", 5), \
         patch("src.ghost_agent.tools.search._RACE_WAVE_GRACE", 5), \
         patch("src.ghost_agent.tools.search.pretty_log"):
        t0 = time.monotonic()
        results = await S._race_search_wave("q", None, 0)
        elapsed = time.monotonic() - t0
    assert results == []
    assert inst.text.call_count == len(S._RACE_ENGINES)
    assert elapsed < 2.0, elapsed


@pytest.mark.asyncio
async def test_the_wave_uses_the_derived_deadline_not_the_slow_constant():
    """FAILS IF: the wave sizes its deadline off `_DDGS_TOR_TIMEOUT + grace`
    again. A wedged engine answers after 1.2 s; the derived deadline (0.3 s)
    must give up first — the constant (18 + 4) would wait and take its late
    results."""
    def late(q, **kw):
        time.sleep(1.2)
        return [{"title": "Late", "body": "b", "href": "http://late-site.com/z"}]

    mod, _ = _mock_ddgs_module(late)
    with patch.dict("sys.modules", {"ddgs": mod}), \
         patch("src.ghost_agent.tools.search._race_wave_deadline", return_value=0.3), \
         patch("src.ghost_agent.tools.search.pretty_log"):
        t0 = time.monotonic()
        results = await S._race_search_wave("wedged", None, 0)
        elapsed = time.monotonic() - t0
    assert results == []
    assert elapsed < 1.0, elapsed


# ── §4HU — a body the client could not collect is a circuit failure ────
@pytest.mark.parametrize("msg", [
    "DecodeError('Body collection error: error decoding response body')",
    "Body collection error: bad chunk length",   # str(e) without the class name
    "primp.DecodeError: bad gzip member",
])
def test_a_body_collection_failure_is_a_conn_error(msg):
    assert S._failure_category(msg) == "conn-error"


def test_unknown_errors_still_keep_their_snippet_category():
    assert S._failure_category("ValueError: something else entirely") == "error"


@pytest.mark.asyncio
async def test_the_terse_line_names_the_engine_as_conn_error_not_a_repr():
    def by_engine(q, **kw):
        if kw["backend"] == "yahoo":
            raise Exception("DecodeError('Body collection error: error decoding response body: "
                            "https://search.yahoo.com/search?p=verylong')")
        raise Exception("No results found.")

    mod, _ = _mock_ddgs_module(by_engine)
    with patch.dict("sys.modules", {"ddgs": mod}), \
         patch("src.ghost_agent.tools.search.pretty_log") as plog:
        await S._race_search_wave("q", None, 0)
    line = [c.args[1] for c in plog.call_args_list if c.kwargs.get("level") == "WARNING"][0]
    assert "yahoo conn-error" in line
    assert "DecodeError" not in line and "https://" not in line
