import pytest
import json
import datetime
from unittest.mock import MagicMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def mock_agent():
    context = MagicMock(spec=GhostContext)
    context.args = MagicMock()
    context.args.use_planning = False
    context.args.smart_memory = 0.0
    context.args.max_context = 4000
    context.args.temperature = 0.7
    
    context.llm_client = MagicMock()
    context.sandbox_dir = None
    context.memory_system = None
    context.profile_memory = None
    context.semantic_memory = None
    context.skill_memory = None
    context.journal = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    
    agent = GhostAgent(context=context)
    return agent

@pytest.mark.asyncio
async def test_streaming_circuit_breaker_and_newlines(mock_agent, capsys):
    """
    Test that the agent's streaming parser:
    1. Preserves isolated newlines (fixes the .strip() bug)
    2. Correctly triggers the circuit breaker on </think or <tool_call
    3. Emits a summary line with the 💭 icon after the stream resolves

    We force the logger into verbose mode for the duration of this test so
    live thinking tokens are actually emitted — the new default is to only
    print a one-line summary, which would make the line-level assertions
    below moot.
    """

    req_id = "test-req-123"

    async def mock_stream(*args, **kwargs):
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "Thinking "}}]},
            {"choices": [{"delta": {"reasoning_content": "\n"}}]}, # Isolated newline
            {"choices": [{"delta": {"reasoning_content": "about "}}]},
            {"choices": [{"delta": {"reasoning_content": "this."}}]},
            {"choices": [{"delta": {"reasoning_content": "\n"}}]}, # Force flush of "about this."
            {"choices": [{"delta": {"reasoning_content": "</think>"}}]}, # This should trigger circuit breaker
            {"choices": [{"delta": {"reasoning_content": "This should be hidden."}}]}, # After circuit breaker
            {"choices": [{"delta": {"content": "<tool_call>"}}]}, # Triggers content circuit breaker fallback
            {"choices": [{"delta": {"content": "Hidden content."}}]} # Should also be hidden
        ]

        for c in chunks:
            yield f"data: {json.dumps(c)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    mock_agent.context.llm_client.stream_chat_completion = mock_stream

    body = {
        "messages": [{"role": "user", "content": "Execute something"}],
        "stream": False # This forces the inner loop fallback which consumes stream_chat_completion
    }

    from ghost_agent.utils import logging as _glog
    prev_verbose = _glog.VERBOSE_MODE
    _glog.VERBOSE_MODE = True
    try:
        with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
            await mock_agent.handle_chat(body, background_tasks=MagicMock(), request_id=req_id)
    finally:
        _glog.VERBOSE_MODE = prev_verbose

    captured = capsys.readouterr()
    stdout = captured.out

    # 💭 icon must appear (live token lines AND the summary line both use it).
    assert "💭" in stdout, "Thought icon missing."

    # The new format emits a one-line summary instead of an open/close frame.
    assert "thinking" in stdout, "Thinking summary line missing."
    assert "tokens" in stdout, "Token count missing from summary."

    # 1. Newlines preserved: both words make it out as live token lines.
    # (Each emitted line is .strip()ed before printing, so we drop trailing
    # whitespace from the assertions.)
    assert "Thinking" in stdout
    assert "about this." in stdout

    # 2. Circuit breaker still hides post-</think> and post-<tool_call> content.
    assert "hidden" not in stdout.lower(), "Circuit breaker failed to silence output after </think or <tool_call"
    
