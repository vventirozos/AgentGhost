"""Deep research must not launder fetch failures into "nothing found".

LIVE-REACHABLE CRITICAL (2026-08-16). `process_url` passed the fetch
result straight to the distiller — and on failure that result is an error
STRING ("Error: Fetch of <url> timed out after 22s"). The worker prompt is
"Extract ONLY the hard facts... If no relevant info is found, state that",
so the worker faithfully reported that the source contained nothing
relevant. Measured with 8 of 8 fetches failing: the report contained the
word "Error" ZERO times and read as POSITIVE NEGATIVE EVIDENCE — "I
checked 8 named sources and none support this".

`fact_check` pipes that report into a TRUE/FALSE/UNVERIFIABLE verifier, so
a Tor blackout produced a confident, citation-backed refutation of a
possibly-true claim.

The perverse part: the `llm_client=None` path was the only HONEST one (it
printed the error verbatim), and registry.py always wires a client — so
the failure only existed in the configuration that actually runs.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../src')))

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.tools import search as S
from ghost_agent.tools import darkweb_search as D


def _worker_that_says_nothing_found():
    """A faithful distiller: handed an error string, it reports no
    relevant information. That behaviour is CORRECT — the bug was ever
    showing it the error."""
    c = MagicMock()
    c.chat_completion = AsyncMock(return_value={"choices": [{"message": {
        "content": "The provided source text does not contain any "
                   "information relevant to the query."}}]})
    return c


class TestClearnetDeepResearch:

    @pytest.mark.asyncio
    async def test_a_total_fetch_failure_is_VISIBLE_in_the_report(self,
                                                                  monkeypatch):
        async def _all_fail(url, *a, **k):
            return f"Error: Fetch of {url} timed out after 22s"
        monkeypatch.setattr(S, "helper_fetch_url_content", _all_fail,
                            raising=False)
        with patch.object(S, "_race_search_wave", new=AsyncMock(
                return_value=[{"href": f"https://x.example/{i}",
                               "title": "t", "body": "b"} for i in range(4)])):
            out = await S.tool_deep_research(
                "does X cause Y", llm_client=_worker_that_says_nothing_found(),
                tor_proxy=None)
        assert "Error" in out, (
            "every fetch failed and the report never says so — it reads as "
            "evidence that the sources contained nothing")
        assert "SOURCE FAILURES" in out
        # …and it must warn against the exact misreading fact_check makes.
        assert "not evidence of absence" in out

    @pytest.mark.asyncio
    async def test_the_distiller_is_NEVER_shown_an_error_string(self,
                                                                monkeypatch):
        """The root cause, pinned directly: a failed fetch must not reach
        the worker at all — otherwise it burns a worker call to have the
        error rewritten into prose."""
        async def _all_fail(url, *a, **k):
            return f"Error: Fetch of {url} timed out after 22s"
        monkeypatch.setattr(S, "helper_fetch_url_content", _all_fail,
                            raising=False)
        worker = _worker_that_says_nothing_found()
        with patch.object(S, "_race_search_wave", new=AsyncMock(
                return_value=[{"href": "https://x.example/1", "title": "t",
                               "body": "b"}])):
            await S.tool_deep_research("q", llm_client=worker, tor_proxy=None)
        assert worker.chat_completion.await_count == 0, (
            "a worker call was burned distilling an error message")

    @pytest.mark.asyncio
    async def test_a_PARTIAL_failure_is_distinguishable_from_a_total_one(
            self, monkeypatch):
        """7-of-8-failed, 8-of-8-failed and "the topic has nothing" all
        rendered identically before."""
        async def _half(url, *a, **k):
            return ("real page content about the topic"
                    if url.endswith("0") else
                    f"Error: Fetch of {url} timed out after 22s")
        monkeypatch.setattr(S, "helper_fetch_url_content", _half,
                            raising=False)
        with patch.object(S, "_race_search_wave", new=AsyncMock(
                return_value=[{"href": f"https://x.example/{i}",
                               "title": "t", "body": "b"} for i in range(4)])):
            out = await S.tool_deep_research(
                "q", llm_client=_worker_that_says_nothing_found(),
                tor_proxy=None)
        assert "3 of 4" in out, out[:200]


class TestOnionDeepResearch:
    """Same distiller, same bug — and it additionally made the cache gate
    inoperative: `_source_succeeded` tested for a leading "Error:", which
    the distiller had already rewritten, so an all-errors run was cached
    for 300 s."""

    @pytest.mark.asyncio
    async def test_the_onion_distiller_is_never_shown_an_error(self,
                                                               monkeypatch):
        async def _fail(url, proxy):
            return f"Error: Fetch of {url} timed out"
        monkeypatch.setattr(D, "_fetch_onion_text", _fail, raising=False)
        worker = _worker_that_says_nothing_found()
        monkeypatch.setattr(D, "_darkweb_search_raw", AsyncMock(
            return_value=([{"url": "http://" + "a" * 56 + ".onion/",
                            "title": "t", "engines": {"torch"},
                            "indexes": {"torch"}}], [], False, 4)))
        out = await D.tool_darkweb_research(
            "q", llm_client=worker, tor_proxy="socks5://127.0.0.1:9050")
        assert worker.chat_completion.await_count == 0
        assert "Error" in out

    def test_the_cache_gate_now_MEANS_what_it_says(self):
        """It only ever worked without an llm_client — the config that
        never runs live."""
        import inspect
        src = inspect.getsource(D.tool_darkweb_research)
        assert 'startswith("Error:")' in src
        # the early return that makes the test true
        assert 'text.startswith("Error:")' in inspect.getsource(
            D.tool_darkweb_research)
