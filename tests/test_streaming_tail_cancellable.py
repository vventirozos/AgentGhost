"""Streaming tail stays cancellable + visible for the whole drain (2026-07-15).

Before the fix, `handle_chat` unregistered the turn in its `finally` the
instant it returned the stream generator — i.e. before a single token
streamed — so the streaming final-generation tail was invisible to
`/api/turns` and uncancellable via `/api/turn/cancel`. The fix defers the
unregister into the stream wrapper's own finally (which runs when the drain
completes) and adds a cooperative cancel boundary inside the stream loop.

These drive the real streaming `handle_chat` path (same harness shape as
test_streaming_scrub_behavioral) and observe the turn registry.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.turns import get_turn_registry


def _ctx():
    ctx = MagicMock()
    ctx.args = MagicMock()
    ctx.args.verbose = False
    ctx.args.max_context = 4000
    ctx.args.smart_memory = 0.0
    ctx.args.temperature = 0.5
    ctx.args.use_planning = True
    ctx.args.native_tools = False
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all.return_value = ""
    ctx.memory_system = None
    ctx.skill_memory = None
    ctx.journal = None
    return ctx


async def _planner_forces_final(payload, *a, **kw):
    # Planner (non-streaming) returns JSON that flips force_final_response so
    # the final-generation streaming path runs.
    return {"choices": [{"message": {"content":
        '{"thought":"answer","next_action_id":"none","required_tool":"all"}'}}]}


def _make_agent(stream_fn):
    ctx = _ctx()
    ctx.llm_client = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_planner_forces_final)
    ctx.llm_client.stream_chat_completion = stream_fn
    return GhostAgent(ctx)


_BODY = {"messages": [{"role": "user", "content": "summarize my recent activity"}],
         "model": "test", "stream": True}


@pytest.mark.asyncio
async def test_turn_stays_registered_until_stream_drains():
    async def fake_stream(*a, **kw):
        for t in ["Hello ", "there ", "world"]:
            yield f'data: {json.dumps({"choices": [{"delta": {"content": t}}]})}\n\n'.encode()
        yield b"data: [DONE]\n\n"

    agent = _make_agent(fake_stream)
    reg = get_turn_registry(agent)

    gen, _, req_id = await agent.handle_chat(dict(_BODY), MagicMock(),
                                             request_id="streamreg1")
    # BEFORE draining: the turn must still be registered (cancellable/visible).
    assert reg.get(req_id) is not None, "streaming turn unregistered before drain"

    chunks = [c async for c in gen]
    assert any(b"world" in c for c in chunks)  # the stream actually ran

    # AFTER draining: unregistered.
    assert reg.get(req_id) is None, "streaming turn not unregistered after drain"


@pytest.mark.asyncio
async def test_cancel_mid_stream_stops_emitting():
    async def fake_stream(*a, **kw):
        for i in range(20):
            yield f'data: {json.dumps({"choices": [{"delta": {"content": f"tok{i} "}}]})}\n\n'.encode()
        yield b"data: [DONE]\n\n"

    agent = _make_agent(fake_stream)
    reg = get_turn_registry(agent)
    gen, _, req_id = await agent.handle_chat(dict(_BODY), MagicMock(),
                                             request_id="streamcancel1")

    # Cancel after the first couple of chunks; the loop's is_cancelled boundary
    # must stop emitting the remaining tokens.
    received = []
    async for c in gen:
        received.append(c)
        if len(received) == 2:
            reg.cancel(req_id)
    text = b"".join(received).decode()
    # It stopped early — not all 20 tokens made it through.
    assert "tok19" not in text, "cancel did not stop the stream"
    # And the turn is unregistered once the (truncated) drain ends.
    assert reg.get(req_id) is None
