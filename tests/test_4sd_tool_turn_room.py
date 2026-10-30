"""§4SD — a tool turn may write up to 64k tokens when the window has room.

Operator 10-29: "16k tokens seems too low when it's writing big files" →
"does this cover all possible scenarios given the context window … 196k?".
The cap is per turn: min(64k, window − this turn's prompt − margin), never
under the old 16k (the prompt is pruned to --max-context, which leaves ~22k of
the 196,608 window for output). Each test names the world it fails in.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ghost_agent.core.agent as A
from ghost_agent.core.agent import (DEFAULT_TOOL_TURN_MAX_TOKENS, TOOL_TURN_MAX_TOKENS_CEIL,
                                    TURN_ROOM_MARGIN_TOKENS, GhostAgent, GhostContext,
                                    output_cap_for_tool_turn)

WIN = 196608


@pytest.mark.parametrize("prompt,window,want", [
    (30000, WIN, TOOL_TURN_MAX_TOKENS_CEIL),                       # early in a build: the full 64k
    (150000, WIN, WIN - 150000 - TURN_ROOM_MARGIN_TOKENS),        # late: what is left
    (174000, WIN, 18512),                                          # at --max-context: 196608-174000-4096
    (190000, WIN, DEFAULT_TOOL_TURN_MAX_TOKENS),                   # over the pruning budget: never under 16k
    (30000, 0, DEFAULT_TOOL_TURN_MAX_TOKENS),                      # window unknown: the pre-§4SD cap
    (30000, None, DEFAULT_TOOL_TURN_MAX_TOKENS),
    ("x", WIN, DEFAULT_TOOL_TURN_MAX_TOKENS),
])
def test_the_cap_is_the_room_the_window_leaves(prompt, window, want):
    # world: a flat 64k promises room a late prompt does not leave, or a full
    # window drops the cap under the old 16k
    assert output_cap_for_tool_turn(prompt, window) == want


def test_prompt_plus_cap_never_passes_the_window_where_the_room_exceeds_the_floor():
    # world: the margin is dropped and a 64k turn runs into the window's end
    for p in range(0, WIN, 997):
        cap = output_cap_for_tool_turn(p, WIN)
        assert DEFAULT_TOOL_TURN_MAX_TOKENS <= cap <= TOOL_TURN_MAX_TOKENS_CEIL
        if cap > DEFAULT_TOOL_TURN_MAX_TOKENS:
            assert p + cap + TURN_ROOM_MARGIN_TOKENS <= WIN


@pytest.fixture
def agent(monkeypatch):
    # the production web/CLI world: a 60-min server budget, no extension
    # ceiling, a client that waits as long as the request runs
    from ghost_agent.utils.logging import client_deadline_context
    monkeypatch.setenv("GHOST_MAX_REQUEST_S", "3600")
    monkeypatch.setenv("GHOST_MAX_REQUEST_EXTENDED_S", "inf")
    tok = client_deadline_context.set(float("inf"))
    A._LENGTH_STOPS.clear()
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock(smart_memory=0.0, max_context=4000, temperature=0.7,
                         native_tools=False, tool_turn_max_tokens=0)
    for k in ("verifier", "self_model", "sandbox_dir", "memory_system", "profile_memory",
              "semantic_memory", "skill_memory", "journal"):
        setattr(ctx, k, None)
    ctx.llm_client = MagicMock()
    ctx.llm_client.upstream_url = "http://main:8088/v1"
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all.return_value = ""
    yield GhostAgent(context=ctx)
    A._LENGTH_STOPS.clear()
    client_deadline_context.reset(tok)


def _props(n_ctx, calls):
    def _http(url, body=None, timeout=5.0):
        calls.append(url)
        if n_ctx is None:
            raise OSError("connection refused")
        return {"default_generation_settings": {"n_ctx": n_ctx}}
    return _http


async def _turns(agent, cuts=1):
    """`cuts` replies cut inside the thinking, then an answer; returns the
    max_tokens each reply was sent."""
    limits = []

    async def _stream(payload, *a, **kw):
        limits.append(payload.get("max_tokens"))
        if len(limits) <= cuts:
            chunks = [{"choices": [{"delta": {"reasoning_content": "drafting const A=1;"}}]},
                      {"choices": [{"delta": {}, "finish_reason": "length"}]}]
        else:
            chunks = [{"choices": [{"delta": {"content": "Written."}}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
        for c in chunks:
            yield f"data: {json.dumps(c)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    agent.context.llm_client.stream_chat_completion = _stream
    agent.context.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]})
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]), \
            patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat({"messages": [{"role": "user", "content": "build a web OS"}],
                                 "stream": False}, background_tasks=MagicMock(), request_id="req-4sd-room")
    return limits


async def test_a_tool_turn_gets_64k_and_so_does_the_retry_after_a_cut(agent, monkeypatch):
    # world: the room is computed but never put on the payload, or the §4RF
    # retry after a cut drops back to 16k
    calls = []
    monkeypatch.setattr(A._head_saves, "_http", _props(WIN, calls))
    limits = await _turns(agent)
    assert limits[:2] == [TOOL_TURN_MAX_TOKENS_CEIL, TOOL_TURN_MAX_TOKENS_CEIL], limits
    assert calls and calls[0] == "http://main:8088/props"
    assert len(calls) == 1                                    # read once, not per turn


async def test_an_unreachable_props_keeps_16k(agent, monkeypatch):
    # world: a /props failure raises into the turn, or caps at 0
    monkeypatch.setattr(A._head_saves, "_http", _props(None, []))
    limits = await _turns(agent)
    assert limits[:2] == [DEFAULT_TOOL_TURN_MAX_TOKENS] * 2


async def test_a_long_prompt_gets_only_the_room_it_leaves(agent, monkeypatch):
    # world: the prompt is not counted — a 170k-token prompt is sent 64k
    monkeypatch.setattr(A._head_saves, "_http", _props(WIN, []))
    monkeypatch.setattr(A, "_estimate_messages_tokens", lambda msgs: 170000)
    payload = {"messages": [{"role": "user", "content": "x"}], "max_tokens": DEFAULT_TOOL_TURN_MAX_TOKENS}
    assert await agent._tool_turn_room(payload) == max(DEFAULT_TOOL_TURN_MAX_TOKENS,
                                                       WIN - 170000 - TURN_ROOM_MARGIN_TOKENS)


async def test_tool_schemas_count_against_the_room(agent, monkeypatch):
    # world: native tool schemas (~7-8k tokens) are left out of the prompt count
    monkeypatch.setattr(A._head_saves, "_http", _props(WIN, []))
    monkeypatch.setattr(A, "_estimate_messages_tokens", lambda msgs: 150000)
    monkeypatch.setattr(A, "estimate_tokens", lambda text: 8000)
    with_tools = await agent._tool_turn_room({"messages": [], "tools": [{"name": "t"}]})
    without = await agent._tool_turn_room({"messages": []})
    assert without - with_tools == 8000


async def test_the_window_is_reread_after_its_ttl(agent, monkeypatch):
    # world: a llama restart with a smaller window is never noticed
    calls = []
    monkeypatch.setattr(A._head_saves, "_http", _props(WIN, calls))
    t = [1000.0]
    monkeypatch.setattr(A.time, "monotonic", lambda: t[0])
    assert await agent._main_window_tokens() == WIN
    t[0] += 599
    await agent._main_window_tokens()
    assert len(calls) == 1
    t[0] += 2
    await agent._main_window_tokens()
    assert len(calls) == 2


async def test_an_explicit_operator_cap_is_left_alone(agent, monkeypatch):
    # world: --tool-turn-max-tokens set by the operator is overridden by the room
    monkeypatch.setattr(A._head_saves, "_http", _props(WIN, []))
    agent.context.args.tool_turn_max_tokens = 12000
    limits = await _turns(agent, cuts=0)
    assert limits[0] == 12000


async def test_a_failed_window_read_is_retried_after_a_minute(agent, monkeypatch):
    # world: a /props failure at first touch pins the 16k floor for 10 min
    calls, state = [], {"up": False}

    def _http(url, body=None, timeout=5.0):
        calls.append(url)
        if not state["up"]:
            raise OSError("down")
        return {"default_generation_settings": {"n_ctx": WIN}}
    monkeypatch.setattr(A._head_saves, "_http", _http)
    t = [1000.0]
    monkeypatch.setattr(A.time, "monotonic", lambda: t[0])
    assert await agent._main_window_tokens() == 0
    state["up"] = True
    t[0] += 59
    assert await agent._main_window_tokens() == 0
    t[0] += 2
    assert await agent._main_window_tokens() == WIN


@pytest.mark.parametrize("max_tokens,final,want", [
    (DEFAULT_TOOL_TURN_MAX_TOKENS, False, True),    # a tool turn with the default cap
    (DEFAULT_TOOL_TURN_MAX_TOKENS, True, False),    # a final answer / report keeps 16k
    (8192, False, False),                           # the forced report's 8k
    (12000, False, False),                          # an operator's own cap
])
def test_only_a_default_tool_turn_takes_the_room(max_tokens, final, want):
    # world: final answers and report turns also get up to 64k (review)
    assert A.turn_takes_room({"max_tokens": max_tokens}, final) is want


async def test_the_non_streamed_overflow_retry_gets_16k_not_the_room(monkeypatch):
    # world (review): the overflow recovery is a NON-streamed call under the
    # client's 1,200 s read timeout, and it kept the turn's 64k cap — a reply
    # of ~70 min at ~15 tok/s that can never land
    import asyncio
    from types import SimpleNamespace
    import httpx
    args = SimpleNamespace(max_context=8000, temperature=0.5, perfect_it=False, smart_memory=0.0)
    context = GhostContext(args=args, sandbox_dir="/tmp", memory_dir="/tmp", tor_proxy=None)
    sent = []

    class _Overflow:
        status_code = 400
        text = "context length exceeded"

    async def _chat(payload, *a, **kw):
        sent.append(json.loads(json.dumps(payload)))
        if len(sent) == 1:
            raise httpx.HTTPStatusError("Overflow", request=MagicMock(), response=_Overflow())
        return {"choices": [{"message": {"content": "recovered ok"}}]}
    context.llm_client = MagicMock()
    context.llm_client.chat_completion = _chat
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    agent = GhostAgent(context)

    async def _room(payload):
        return TOOL_TURN_MAX_TOKENS_CEIL
    monkeypatch.setattr(agent, "_tool_turn_room", _room)
    body = {"messages": [{"role": "user", "content": "please run the tool"}]}
    await asyncio.wait_for(agent.handle_chat(body, background_tasks=None), timeout=10.0)
    assert len(sent) == 2, len(sent)
    assert sent[0].get("max_tokens") == TOOL_TURN_MAX_TOKENS_CEIL           # the turn took the room
    assert sent[1]["stream"] is False
    assert sent[1]["max_tokens"] == DEFAULT_TOOL_TURN_MAX_TOKENS             # the retry did not


# ── §4SD (operator 10-29: "set a limit to slack for 30 mins") ────────────────

@pytest.mark.parametrize("ceiling,elapsed,want", [
    (0, 0, True),                                   # no deadline (background): the room
    (float("inf"), 99999, True),                    # web / CLI: no ceiling
    (1800, 0, False),                               # Slack: 30 min cannot hold a 64k turn
    (3600, 0, False),                               # a probe's flat 60 min cannot either
    (14400, 0, True),                               # a 4 h ceiling, early
    (14400, 10000, False),                          # …and late
    (14400, None, True),                            # elapsed unknown = from the start
    ("x", 0, False),
])
def test_a_turn_gets_the_room_only_when_its_time_could_hold_it(ceiling, elapsed, want):
    # world: a Slack request's first turn is sent 64k (~73 min) under a 30-min limit
    assert A.turn_room_fits_time(ceiling, elapsed) is want


async def test_a_slack_request_keeps_16k(agent, monkeypatch):
    # world: the 30-min Slack budget is never consulted when the turn is sized
    from ghost_agent.utils.logging import client_deadline_context
    monkeypatch.setattr(A._head_saves, "_http", _props(WIN, []))
    tok = client_deadline_context.set(1800.0)
    try:
        limits = await _turns(agent, cuts=0)
    finally:
        client_deadline_context.reset(tok)
    assert limits[0] == DEFAULT_TOOL_TURN_MAX_TOKENS


async def test_a_turn_late_in_a_capped_request_keeps_16k(agent, monkeypatch):
    # world: the time already spent is not subtracted from the ceiling
    from ghost_agent.utils import logging as L
    monkeypatch.setattr(A._head_saves, "_http", _props(WIN, []))
    monkeypatch.setattr(A, "_estimate_messages_tokens", lambda msgs: 30000)
    monkeypatch.setattr(L, "extension_ceiling_s", lambda rid="": 14400.0)
    payload = {"messages": [], "max_tokens": DEFAULT_TOOL_TURN_MAX_TOKENS}
    monkeypatch.setattr(L, "request_elapsed_s", lambda rid: 0.0)
    assert await agent._tool_turn_room(payload) == TOOL_TURN_MAX_TOKENS_CEIL
    monkeypatch.setattr(L, "request_elapsed_s", lambda rid: 10000.0)
    assert await agent._tool_turn_room(payload) == DEFAULT_TOOL_TURN_MAX_TOKENS
