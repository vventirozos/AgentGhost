"""§4SD — no turn count for a request with a deadline; Slack gets 30 minutes.

Operator 10-29: "remove the tool turn limit, and set a limit to slack for 30
mins". A request with a time budget runs until its deadline (extended only by
coding progress) or a breaker ends it; one with no deadline (scheduled, job,
bench, internal) keeps 40 turns, its only bound. The Slack bot sends a 1800 s
X-Ghost-Client-Timeout, which the agent never extends past. Each test names the
world it fails in.
"""
import math
import sys
import time

import pytest

import ghost_agent.core.agent as A
from ghost_agent.utils import logging as L

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_4sd_tool_turn_room import WIN, _props, agent  # noqa: E402,F401  (fixture)


@pytest.mark.parametrize("override,deadline,elapsed,want", [
    (None, 3600.0, 0.0, sys.maxsize),                # an owner / Slack / probe request: no count
    (None, 3600.0, None, A.TURN_CAP_WITHOUT_DEADLINE),   # no running clock: no deadline report can fire
    (None, 0.0, 5.0, A.TURN_CAP_WITHOUT_DEADLINE),   # background: no deadline, so 40 turns
    (None, None, 5.0, A.TURN_CAP_WITHOUT_DEADLINE),
    (None, "x", 5.0, A.TURN_CAP_WITHOUT_DEADLINE),
    (14, 3600.0, 0.0, 14),                           # a leaf / sub-agent keeps its own count
    (2, 0.0, None, 2),
])
def test_the_turn_count(override, deadline, elapsed, want):
    # world: the 40-turn cap still stops a timed build, or a request with no
    # deadline (or no clock) loses its only bound, or a sub-agent's count is ignored
    assert A.main_turn_cap(override, deadline, elapsed) == want


def test_no_deadline_keeps_the_old_forty():
    # world: the background bound drifts
    assert A.TURN_CAP_WITHOUT_DEADLINE == 40


@pytest.mark.parametrize("rid,timed", [("4sdcap-owner", True), ("sched-4sdcap", False)])
async def test_the_loop_asks_with_the_requests_deadline_and_obeys(agent, monkeypatch, rid, timed):
    # world: the loop keeps a literal 40, or reads no deadline, or ignores the answer
    import json
    from unittest.mock import AsyncMock, MagicMock, patch
    monkeypatch.setattr(A._head_saves, "_http", _props(WIN, []))
    seen, sent = [], []
    monkeypatch.setattr(A, "main_turn_cap", lambda o, d, e: seen.append((o, d, e)) or 1)
    monkeypatch.setattr(L, "request_elapsed_s", lambda r: 7.0)      # pretty_log is patched: no BEGIN

    async def _stream(payload, *a, **kw):
        sent.append(1)
        for c in ({"choices": [{"delta": {"reasoning_content": "drafting"}}]},
                  {"choices": [{"delta": {}, "finish_reason": "length"}]}):
            yield f"data: {json.dumps(c)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    agent.context.llm_client.stream_chat_completion = _stream
    agent.context.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]})
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]), \
            patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat({"messages": [{"role": "user", "content": "build it"}], "stream": False},
                                background_tasks=MagicMock(), request_id=rid)
    assert len(seen) == 1 and seen[0][0] is None and seen[0][2] == 7.0
    assert (seen[0][1] > 0) is timed
    assert len(sent) == 1                       # a count of 1 = one turn (cuts would retry otherwise)


# ── Slack: 30 minutes, never extended ────────────────────────────────────────

RID = "4sdslack"


@pytest.fixture
def slack(monkeypatch):
    monkeypatch.setenv("GHOST_MAX_REQUEST_S", "3600")
    monkeypatch.setenv("GHOST_MAX_REQUEST_EXTENDED_S", "inf")
    tok = L.client_deadline_context.set(1800.0)            # what the bot sends
    now = time.monotonic()
    with L._REQ_STATE_LOCK:
        L._REQ_STATE[RID] = {"started": now - 1700, "origin": None, "longest_turn_s": 0.0,
                             "progress_t": now - 5, "progress_what": "wrote app.js"}
    yield
    with L._REQ_STATE_LOCK:
        L._REQ_STATE.pop(RID, None)
    L.client_deadline_context.reset(tok)


def test_a_slack_request_is_thirty_minutes_even_while_it_makes_progress(slack):
    # world: the server's 60 min wins, or coding progress extends a Slack request
    assert L.base_deadline_s(RID) == 1800
    assert L.extension_ceiling_s(RID) == 1800
    assert L.request_deadline_s(RID) == 1800
    assert not math.isinf(L.extension_ceiling_s(RID))


async def test_a_timed_request_whose_model_always_calls_tools_ends_at_its_deadline(monkeypatch):
    # world (review): with no turn count, nothing ends a timed request once its
    # deadline is reached — the report is never forced and the loop spins
    import json
    from unittest.mock import AsyncMock
    from tests.helpers import FakeBgTasks, make_context
    for k, v in (("GHOST_CRITIC_ASYNC", "1"), ("GHOST_CRITIC_REPAIR_BUDGET", "0"), ("GHOST_EVIDENCE_GATE", "0")):
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(L, "request_deadline_s", lambda r="": 3600.0)
    monkeypatch.setattr(L, "request_elapsed_s", lambda r: 3580.0)
    monkeypatch.setattr(L, "request_remaining_s", lambda r="": 20.0)
    ctx = make_context()
    agent = A.GhostAgent(ctx)
    n = {"i": 0}
    agent.available_tools = {"web_search": AsyncMock(side_effect=lambda **k: f"### result {n['i']}\nfact {k}\n")}

    async def _always_a_tool(*a, **k):
        n["i"] += 1
        if n["i"] > 60:
            return {"choices": [{"message": {"role": "assistant", "content": "unreachable"}}]}
        call = {"id": f"c{n['i']}", "type": "function",
                "function": {"name": "web_search", "arguments": json.dumps({"query": f"topic {n['i']}"})}}
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [call]}}]}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_always_a_tool)
    await agent.handle_chat({"messages": [{"role": "user", "content": "research everything"}]}, FakeBgTasks())
    assert n["i"] <= 4, n["i"]                  # the report turn and its retry, not dozens of turns
