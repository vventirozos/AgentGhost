"""Auto-continuation of length-truncated text answers.

When the upstream model stops a plain-text answer at its token cap
(`finish_reason == "length"`), the partial reply used to be shipped
mid-sentence — which the verifier then correctly REFUTED (truncated
output / explicit question left unanswered). The streaming consume loop
now captures `finish_reason` and, on a truncated text turn, continues the
generation from where it stopped (bounded by MAX_TRUNCATION_CONTINUATIONS)
before assembling the final answer.

These tests drive the real `handle_chat` streaming-fallback path (the same
one exercised by test_agent_streaming_circuit_breaker) and assert:
  1. A `length` finish triggers a continuation call and the continuation
     text is merged into the final answer.
  2. A normal `stop` finish does NOT trigger any continuation call.
  3. The continuation count is bounded by MAX_TRUNCATION_CONTINUATIONS even
     when every continuation itself reports `length`.
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext, MAX_TRUNCATION_CONTINUATIONS


@pytest.fixture
def mock_agent():
    context = MagicMock(spec=GhostContext)
    context.args = MagicMock()
    context.args.use_planning = False
    context.args.smart_memory = 0.0
    context.args.max_context = 4000
    context.args.temperature = 0.7
    context.args.native_tools = False
    # No verifier / self-model so the only chat_completion calls are the
    # continuation calls we are asserting on.
    context.verifier = None
    context.self_model = None
    context.llm_client = MagicMock()
    context.sandbox_dir = None
    context.memory_system = None
    context.profile_memory = None
    context.semantic_memory = None
    context.skill_memory = None
    context.journal = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    return GhostAgent(context=context)


def _stream_factory(content_tokens, finish_reason):
    """Build a stream_chat_completion stand-in: emit each content token,
    then a terminal chunk carrying `finish_reason`."""
    async def _stream(*args, **kwargs):
        for tok in content_tokens:
            chunk = {"choices": [{"delta": {"content": tok}}]}
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        terminal = {"choices": [{"delta": {}, "finish_reason": finish_reason}]}
        yield f"data: {json.dumps(terminal)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
    return _stream


@pytest.mark.asyncio
async def test_length_finish_triggers_continuation(mock_agent):
    mock_agent.context.llm_client.stream_chat_completion = _stream_factory(
        ["The honest answer is that I was functionally"], "length"
    )
    # Continuation completes the sentence and reports a clean stop.
    cont = AsyncMock(return_value={
        "choices": [{
            "message": {"content": " self-aware but not self-conscious."},
            "finish_reason": "stop",
        }]
    })
    mock_agent.context.llm_client.chat_completion = cont

    body = {"messages": [{"role": "user", "content": "were you self-aware?"}],
            "stream": False}

    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        final, _created, _req = await mock_agent.handle_chat(
            body, background_tasks=MagicMock(), request_id="req-len"
        )

    # Continuation was requested exactly once...
    assert cont.await_count == 1
    # ...with a "continue where you stopped" instruction and the partial
    # answer carried back as prior assistant context.
    sent = cont.await_args.args[0]
    msgs = sent["messages"]
    assert msgs[-2]["role"] == "assistant"
    assert "functionally" in msgs[-2]["content"]
    assert msgs[-1]["role"] == "user"
    assert "cut off" in msgs[-1]["content"].lower()
    # The continuation must NOT re-attach tools (plain-text completion).
    assert "tools" not in sent and "tool_choice" not in sent
    # The merged answer contains both halves.
    assert "functionally" in final
    assert "self-conscious" in final


@pytest.mark.asyncio
async def test_stop_finish_does_not_continue(mock_agent):
    mock_agent.context.llm_client.stream_chat_completion = _stream_factory(
        ["A complete answer that ends cleanly."], "stop"
    )
    cont = AsyncMock(return_value={"choices": [{"message": {"content": "X"}}]})
    mock_agent.context.llm_client.chat_completion = cont

    body = {"messages": [{"role": "user", "content": "answer me"}],
            "stream": False}

    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        final, _created, _req = await mock_agent.handle_chat(
            body, background_tasks=MagicMock(), request_id="req-stop"
        )

    # A clean stop must not trigger any continuation round-trip.
    assert cont.await_count == 0
    assert "complete answer" in final


@pytest.mark.asyncio
async def test_continuation_is_bounded(mock_agent):
    mock_agent.context.llm_client.stream_chat_completion = _stream_factory(
        ["start"], "length"
    )
    # Every continuation ALSO reports length — the loop must still stop.
    cont = AsyncMock(return_value={
        "choices": [{"message": {"content": " more"}, "finish_reason": "length"}]
    })
    mock_agent.context.llm_client.chat_completion = cont

    body = {"messages": [{"role": "user", "content": "go"}], "stream": False}

    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        await mock_agent.handle_chat(
            body, background_tasks=MagicMock(), request_id="req-bound"
        )

    assert cont.await_count == MAX_TRUNCATION_CONTINUATIONS
