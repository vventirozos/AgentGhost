"""A page the site refused is not the agent's failure (§4HN, 2026-09-16).

§4HH made a 4xx/bot-challenge page a DECLARED failure (`STATUS: BLOCKED`) so
the corpus label and the no-progress window see it. Live (req a91c3e16) the
same declaration then cost 4 of 6 strikes toward the forced final and
tripped outcome-heuristic signal 3 — "tool 'browser' returned the same error
4× in one turn" — labelling a verifier-confirmed run FAILED. Four paywalls
in one research turn are four sites' decisions, not repeated agent error.
Each pin names the world it fails in.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.core.agent import GhostAgent
from ghost_agent.distill import outcome_heuristics as OH
from ghost_agent.distill.outcome_heuristics import classify_chat_outcome
from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
from ghost_agent.tools import tool_failure as TF
from ghost_agent.tools.outcome import ToolOutcome

_BLOCKED = ("--- BROWSER RESULT ---\nSTATUS: BLOCKED (HTTP 403 — bot challenge)\nOP: extract_text\n"
            "HINT: This site refused the fetch — the text below (if any) is the challenge or error page, "
            "NOT the article.\nURL: https://www.kelacyber.com/blog/x\nHTTP_STATUS: 403\nTITLE: Just a moment...\n"
            "LENGTH: 0\n--- TEXT ---\n")
_TIMEOUT = ("--- BROWSER RESULT ---\nSTATUS: ERROR\nRunner failed (exit 1): Timeout Page.goto: "
            "Timeout 30000ms exceeded.\n")


# ── the predicate, one authority ──────────────────────────────────────────

def test_the_predicate_reads_the_head_only():
    """FAILS IF: the head test is lost or widened to the whole text — a
    page that merely QUOTES 'STATUS: BLOCKED' deep in its body was read."""
    assert TF.is_blocked_page_result(_BLOCKED) is True
    assert TF.is_blocked_page_result("[FAILURE BANNER] --- BROWSER RESULT ---\n" + _BLOCKED) is True
    assert TF.is_blocked_page_result(_TIMEOUT) is False
    assert TF.is_blocked_page_result("--- BROWSER RESULT ---\nSTATUS: OK\n" + "x" * 300 + "STATUS: BLOCKED") is False


def test_the_corpus_reader_uses_the_same_pattern():
    """FAILS IF: the two modules drift — the corpus reader must not import
    the tools package, so the pattern is duplicated and pinned equal."""
    assert OH._BLOCKED_HEAD_RE.pattern == TF._BLOCKED_HEAD_RE.pattern


# ── signal 3 ──────────────────────────────────────────────────────────────

def _traj(results):
    # The chat path stamps `ToolCall.error` on every declared failure (the
    # text sniff is the legacy fallback), so the fixtures carry it too —
    # a blocked page is exempt BECAUSE of its head, not because it lacks
    # the flag.
    return Trajectory(outcome=Outcome.UNKNOWN.value, final_response="Here's the synthesis.",
                      tool_calls=[ToolCall(name="browser", arguments={"operation": "extract_text",
                                                                        "url": f"https://s{i}.example/"},
                                           result=r, error="browser :: " + r.splitlines()[1][:40])
                                  for i, r in enumerate(results)])


def test_four_blocked_sites_are_not_the_same_error_repeated():
    """FAILS IF: signal 3 still counts refused fetches — the live world."""
    v = classify_chat_outcome(_traj([_BLOCKED] * 4))
    assert v.outcome == Outcome.UNKNOWN.value, v.reason


def test_four_identical_real_errors_still_fire():
    """FAILS IF: the exemption widened past the BLOCKED head."""
    v = classify_chat_outcome(_traj([_TIMEOUT] * 4))
    assert v.outcome == Outcome.FAILED.value
    assert "same error 4×" in v.reason


# ── the strike ledger ─────────────────────────────────────────────────────

_REQ = {"messages": [{"role": "user", "content":
                      "Read these four articles and summarise what they say about the breach."}],
        "model": "Qwen-Test"}


@pytest.fixture
def agent(mock_context):
    return GhostAgent(mock_context)


def _drive(agent, browser_result, n=4):
    state = {"n": 0}

    async def fake(payload, *a, **kw):
        state["n"] += 1
        if state["n"] <= n:
            return {"choices": [{"message": {"content": None, "tool_calls": [{
                "id": f"c{state['n']}",
                "function": {"name": "browser",
                             "arguments": json.dumps({"operation": "extract_text",
                                                      "url": f"https://s{state['n']}.example/"})}}]}}]}
        return {"choices": [{"message": {"content": "The sources agree on the mechanism.", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=fake)
    agent.available_tools["browser"] = AsyncMock(return_value=browser_result)


def _strike_lines(pl):
    return [c.args[0] for c in pl.call_args_list
            if c.args and c.args[0] in ("Execution Fail", "Transient Fail", "Blocked Page", "Failure Cap")]


@pytest.mark.asyncio
async def test_blocked_pages_cost_no_strikes(agent):
    """FAILS IF: a refused fetch still lands on a strike ledger — four of
    them were 4/6 of the way to a forced final on a91c3e16."""
    _drive(agent, ToolOutcome.failed(_BLOCKED, world_changed=False, reason_code="browser_blocked"))
    with patch("ghost_agent.core.agent.pretty_log") as pl:
        await agent.handle_chat(_REQ, background_tasks=MagicMock())
    lines = _strike_lines(pl)
    assert "Blocked Page" in lines
    assert "Execution Fail" not in lines and "Transient Fail" not in lines and "Failure Cap" not in lines


@pytest.mark.asyncio
async def test_a_real_browser_failure_still_strikes(agent):
    """FAILS IF: the exemption swallows every browser failure."""
    _drive(agent, ToolOutcome.failed(_TIMEOUT, world_changed=False, reason_code="browser_error"))
    with patch("ghost_agent.core.agent.pretty_log") as pl:
        await agent.handle_chat(_REQ, background_tasks=MagicMock())
    lines = _strike_lines(pl)
    assert "Transient Fail" in lines or "Execution Fail" in lines
    assert "Blocked Page" not in lines


@pytest.mark.asyncio
async def test_a_blocked_page_is_still_a_declared_failure_for_the_record(agent):
    """FAILS IF: the exemption reaches the outcome — the record and the
    no-progress window must still see a failed fetch (§4HH's point)."""
    _drive(agent, ToolOutcome.failed(_BLOCKED, world_changed=False, reason_code="browser_blocked"), n=1)
    with patch("ghost_agent.core.agent.pretty_log") as pl:
        await agent.handle_chat(_REQ, background_tasks=MagicMock())
    assert "Blocked Page" in _strike_lines(pl)
    assert OH.looks_like_tool_error(_BLOCKED) is True
