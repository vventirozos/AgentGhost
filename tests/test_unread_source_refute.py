"""§4HY — a source "actually read" that no tool ever opened.

Live (req 1cc63597): asked for "the official source URL you actually read",
the reply cited postgresql.org "— the homepage announcement states: …" after
one web_search and no page load; the judge CONFIRMED 0.95 from the snippet.
"""
from __future__ import annotations

import pytest

from ghost_agent.core import reply_shape_check as rsc

_ASK = ("What is the latest stable release of PostgreSQL as of today? Cite the official "
        "source URL you actually read.")
_REPLY = ("The latest stable PostgreSQL release is **18.6**, released on **2026-08-13**.\n\n"
          "**Official source:** https://www.postgresql.org/about/release/ — the announcement states it.")


def test_the_live_shape_is_refuted_with_the_url_named():
    issues = rsc.refute_unread_source(_REPLY, _ASK, ["web_search"])
    assert len(issues) == 1
    assert "https://www.postgresql.org/about/release/" in issues[0]
    assert "no page was opened" in issues[0]


@pytest.mark.parametrize("tools", [["web_search", "browser"], ["deep_research"], ["darkweb_research", "web_search"]])
def test_a_page_load_this_turn_clears_it(tools):
    assert rsc.refute_unread_source(_REPLY, _ASK, tools) == []


def test_without_the_ask_a_search_sourced_url_is_fine():
    assert rsc.refute_unread_source(_REPLY, "What is the latest PostgreSQL release? Cite a source.", ["web_search"]) == []


def test_without_a_url_there_is_nothing_to_refute():
    assert rsc.refute_unread_source("PostgreSQL 18.6 (2026-08-13), per the official announcement.", _ASK, ["web_search"]) == []


@pytest.mark.parametrize("ask", [
    "Open the official page and tell me the price.",
    "Which sources did you read for this?",
    "Give me the URL you really visited.",
])
def test_the_ask_vocabulary(ask):
    assert rsc.refute_unread_source(_REPLY, ask, ["web_search"])


def test_page_loading_tools_are_the_fetchers_not_the_searchers():
    assert {"browser", "deep_research", "darkweb_research"} <= rsc.PAGE_LOADING_TOOLS
    assert "web_search" not in rsc.PAGE_LOADING_TOOLS and "darkweb_search" not in rsc.PAGE_LOADING_TOOLS


def _agent():
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.agent import GhostAgent
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    a = GhostAgent(ctx)
    a.available_tools = {}
    a.disabled_tools = set()
    return a


def test_the_mechanical_site_refutes_the_live_shape_before_any_judge():
    """FAILS IF: the site does not consult `refute_unread_source`, or passes
    it the wrong tool names."""
    from ghost_agent.core.verifier import VerifyVerdict
    a = _agent()
    tools = [{"name": "web_search", "content": "### 1. PostgreSQL 18.6 released — postgresql.org/about/release/"}]
    v = a._reply_shape_refutation(_REPLY, _ASK, tools_run=tools)
    assert v is not None and v.verdict == VerifyVerdict.REFUTED
    assert any("no page was opened" in i for i in v.issues)
    assert "never opened" in v.reasoning
    # with a page load this turn the site has nothing to say
    tools2 = tools + [{"name": "browser", "content": "--- BROWSER RESULT --- STATUS: OK ..."}]
    assert a._reply_shape_refutation(_REPLY, _ASK, tools_run=tools2) is None
