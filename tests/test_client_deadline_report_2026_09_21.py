"""§4JP — the client's deadline is a budget: the report floor and the wait refusal.

THE LIVE FAILURE (req fd89fd6d, 2026-09-21, "Redo this" — a Sponza path
tracer): 35 turns, 34 tool calls, cut at exactly 1800 s — the web
interface's GHOST_CHAT_TIMEOUT. The loop never knew the client's deadline:
with 61 s left it started a browser `sleep 60000`, the interface closed the
connection, and the reply that landed was the stitched working narration
("Let me build… Let me rebuild… Let me switch…"). Three user turns hit
that wall in Aug–Sep.

Now the interface sends `X-Ghost-Client-Timeout`, the route stores it in
`client_deadline_context`, `request_remaining_s` derives the time left, and
the loop (1) forces the state report — tools off, the §4IG flag armed —
once fewer than DEADLINE_REPORT_FLOOR_S seconds remain, and (2) refuses,
before running it, any browser wait that would end inside that floor.

World where each pin fails: the header stops reaching the contextvar, the
remaining-time helper drifts, the floor never forces the report (or forces
it without the flag), a long wait runs into the floor, the refusal counts
as a strike, or a client without the header gets a deadline it never had.
"""
import json
import time
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import (DEADLINE_REPORT_FLOOR_S, GhostAgent, deadline_needs_report,
                                    declared_wait_s, wait_crosses_deadline)
from ghost_agent.utils import logging as glog
from tests.helpers import FakeBgTasks, make_context


# ── the pure helpers ─────────────────────────────────────────────────

def test_report_is_forced_only_inside_the_floor_on_a_running_request():
    assert deadline_needs_report(120.0, DEADLINE_REPORT_FLOOR_S, False, False) is True
    assert deadline_needs_report(DEADLINE_REPORT_FLOOR_S, DEADLINE_REPORT_FLOOR_S, False, False) is True
    assert deadline_needs_report(151.0, DEADLINE_REPORT_FLOOR_S, False, False) is False
    assert deadline_needs_report(None, DEADLINE_REPORT_FLOOR_S, False, False) is False       # no deadline known
    assert deadline_needs_report(10.0, DEADLINE_REPORT_FLOOR_S, True, False) is False        # already forced
    assert deadline_needs_report(10.0, DEADLINE_REPORT_FLOOR_S, False, True) is False        # stopping anyway
    assert deadline_needs_report("x", DEADLINE_REPORT_FLOOR_S, False, False) is False


def test_declared_wait_sums_browser_sleeps_and_settle():
    args = {"operation": "interact", "actions": [{"action": "sleep", "ms": 60000}, {"action": "screenshot"},
                                                 {"action": "sleep", "ms": "5000"}], "settle_ms": 2000}
    assert declared_wait_s("browser", args) == 67.0
    assert declared_wait_s("browser", {"operation": "navigate", "url": "http://x"}) == 0.0
    assert declared_wait_s("execute", {"command": "sleep 60"}) == 0.0                  # not declared: unknown
    assert declared_wait_s("browser", "not a dict") == 0.0


def test_a_wait_that_ends_inside_the_floor_is_refused():
    assert wait_crosses_deadline(60.0, 61.0, DEADLINE_REPORT_FLOOR_S) is True     # the fd89fd6d sleep
    assert wait_crosses_deadline(60.0, 400.0, DEADLINE_REPORT_FLOOR_S) is False
    assert wait_crosses_deadline(60.0, 209.0, DEADLINE_REPORT_FLOOR_S) is True     # would end 1 s inside the floor
    assert wait_crosses_deadline(60.0, None, DEADLINE_REPORT_FLOOR_S) is False     # no deadline known
    assert wait_crosses_deadline(0.0, 10.0, DEADLINE_REPORT_FLOOR_S) is False      # nothing declared


def test_remaining_time_is_deadline_minus_elapsed_and_none_without_a_deadline():
    rid = "dl-test-" + str(int(time.time() * 1000))
    tok = glog.request_id_context.set(rid)
    try:
        glog.pretty_log("request started", "x", special_marker="BEGIN", origin="user")
        glog.client_deadline_context.set(0.0)
        assert glog.request_remaining_s(rid) is None
        glog.client_deadline_context.set(1800.0)
        r = glog.request_remaining_s(rid)
        assert r is not None and 1795.0 < r <= 1800.0
        assert glog.request_remaining_s("never-opened") is None
    finally:
        glog.client_deadline_context.set(0.0)
        glog.request_id_context.reset(tok)


# ── through the loop ─────────────────────────────────────────────────

SEARCH_RESULT = "### 1. Reuters\nRevolut confirmed the breach.\n[Source: https://reuters.example/a]\n"


def _resp(content, tool_calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls or []}}]}


def _tc(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _agent(monkeypatch, scripted, tools=None):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    agent.available_tools = tools or {"web_search": AsyncMock(return_value=SEARCH_RESULT)}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=scripted)
    return agent, ctx


def _payloads(ctx):
    out = []
    for c in ctx.llm_client.chat_completion.call_args_list:
        p = c.kwargs.get("messages")
        if p is None and c.args:
            p = c.args[0].get("messages") if isinstance(c.args[0], dict) else c.args[0]
        out.append(p)
    return out


async def test_inside_the_floor_the_next_turn_is_the_report_with_tools_off(monkeypatch):
    """Deadline 100 s (already inside the 150 s floor when the loop starts):
    turn 1 runs a search; turn 2 must be the forced report — the alert names
    the client deadline, the tool call on it is dropped, the flag is armed."""
    search = AsyncMock(return_value=SEARCH_RESULT)
    # 400 s left while the first search runs, 100 s (inside the floor) after it
    monkeypatch.setattr(glog, "request_remaining_s", lambda rid: 100.0 if search.await_count else 400.0)
    agent, ctx = _agent(monkeypatch, [
        _resp("Let me search.", [_tc("c0", "web_search", {"query": "revolut breach"})]),
        _resp("Report: the search found Reuters confirming the breach; nothing else was built.",
              [_tc("c1", "web_search", {"query": "more"})]),                      # dropped: tools are off
        _resp("Report: the search found Reuters confirming the breach; nothing else was built."),
        _resp("(unreachable)"),
    ], tools={"web_search": search})
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "investigate the revolut breach"}]}, FakeBgTasks())
    text = "\n".join(str(m.get("content")) for p in _payloads(ctx) for m in p)
    assert "SYSTEM ALERT (client deadline)" in text, "the deadline report alert never reached the model"
    assert "close its connection in about 100 seconds" in text
    assert search.await_count == 1                                            # the second search never ran
    assert "Reuters confirming the breach" in out


async def test_without_a_deadline_the_loop_is_untouched(monkeypatch):
    monkeypatch.setattr(glog, "request_remaining_s", lambda rid: None)
    agent, ctx = _agent(monkeypatch, [
        _resp("Let me search.", [_tc("c0", "web_search", {"query": "revolut breach"})]),
        _resp("Let me search again.", [_tc("c1", "web_search", {"query": "revolut breach details"})]),
        _resp("The breach was confirmed by Reuters."),
        _resp("(unreachable)"),
    ])
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "investigate the revolut breach"}]}, FakeBgTasks())
    payloads = _payloads(ctx)
    assert not any("client deadline" in str(m.get("content")) for p in payloads for m in p if m.get("role") == "user")
    assert agent.available_tools["web_search"].await_count == 2
    assert "confirmed by Reuters" in out


async def test_a_long_wait_is_refused_before_it_runs_and_is_not_a_strike(monkeypatch):
    """Remaining 200 s, floor 150: a 60 s browser sleep would end inside
    the floor → refused with the deadline note, the browser never runs,
    the model answers on the next turn; no failure strike is recorded."""
    monkeypatch.setattr(glog, "request_remaining_s", lambda rid: 200.0)
    browser = AsyncMock(return_value="--- BROWSER RESULT ---\nSTATUS: OK")
    agent, ctx = _agent(monkeypatch, [
        _resp("Let me wait for the render.", [_tc("c0", "browser", {"operation": "interact",
                                                                    "actions": [{"action": "sleep", "ms": 60000}, {"action": "screenshot"}]})]),
        _resp("The renderer is served at http://127.0.0.1:8104/x.html; the render was not captured."),
        _resp("(unreachable)"),
    ], tools={"browser": browser, "web_search": AsyncMock(return_value=SEARCH_RESULT)})
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "render the sponza scene"}]}, FakeBgTasks())
    assert browser.await_count == 0
    # the synthetic tool message rides inside the volatile state block the
    # loop wraps around the last messages — search every message's text
    text = "\n".join(str(m.get("content")) for p in _payloads(ctx) for m in p)
    assert "SYSTEM PREFLIGHT — deadline" in text and "60 s wait" in text and "about 200 s" in text
    assert "served at" in out


async def test_a_short_wait_runs_when_the_deadline_is_far(monkeypatch):
    monkeypatch.setattr(glog, "request_remaining_s", lambda rid: 1500.0)
    browser = AsyncMock(return_value="--- BROWSER RESULT ---\nSTATUS: OK\nSAVED: x.png")
    agent, ctx = _agent(monkeypatch, [
        _resp("Let me wait for the render.", [_tc("c0", "browser", {"operation": "interact",
                                                                    "actions": [{"action": "sleep", "ms": 60000}, {"action": "screenshot"}]})]),
        _resp("Captured x.png after a minute of accumulation."),
        _resp("(unreachable)"),
    ], tools={"browser": browser, "web_search": AsyncMock(return_value=SEARCH_RESULT)})
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "render the sponza scene"}]}, FakeBgTasks())
    assert browser.await_count == 1
    assert "Captured x.png" in out


# ── the route stores the header ──────────────────────────────────────

class _Req:
    def __init__(self, body, headers):
        self._body = body; self.headers = headers; self.method = "POST"; self.query_params = {}
    async def json(self): return self._body
    async def body(self): return b""


@pytest.mark.parametrize("header, want", [("1800", 1800.0), ("abc", 0.0), ("", 0.0), (None, 0.0), ("-5", 0.0)])
async def test_the_route_stores_the_client_timeout_in_the_contextvar(header, want):
    """Executed through the REAL chat route (`chat_proxy`) with the agent
    stubbed: the deadline the stub sees during `handle_chat` is the header's
    value; absent or malformed → 0.0 (no deadline)."""
    from unittest.mock import MagicMock, patch
    from ghost_agent.api import routes as R
    seen = {}
    agent = MagicMock()
    agent.context.args.model = "ghost-model"; agent.context.args.api_key = ""
    agent.context.llm_client.foreground_requests = 0

    async def _hc(body, background_tasks, request_id=None):
        seen["deadline"] = glog.client_deadline_context.get()
        return "ok", 123, request_id or "r"
    agent.handle_chat = _hc
    headers = {} if header is None else {"X-Ghost-Client-Timeout": header}
    req = _Req({"messages": [{"role": "user", "content": "hi"}], "stream": False}, headers)
    with patch.object(R, "get_agent", return_value=agent):
        await R.chat_proxy(req, MagicMock())
    assert seen["deadline"] == want
