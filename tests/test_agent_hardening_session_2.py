import pytest
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.core.dream import Dreamer

@pytest.fixture
def mock_agent():
    context = MagicMock(spec=GhostContext)
    context.args = MagicMock()
    context.args.smart_memory = 0.0
    context.args.max_context = 4000
    context.args.temperature = 0.7
    
    context.llm_client = MagicMock()
    context.sandbox_dir = None
    context.memory_system = None
    context.profile_memory = None
    context.semantic_memory = MagicMock()
    context.journal = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    
    agent = GhostAgent(context=context)
    return agent

@pytest.mark.asyncio
async def test_strip_think_tags_from_permanent_history(mock_agent):
    """
    Test that the deep cognitive loops are prevented by stripping <think> blocks
    from the permanent message history when finalizing an AI turn.
    """
    req_id = "test-req-think"
    
    async def mock_stream(*args, **kwargs):
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "This is my internal thought loop."}}]},
            {"choices": [{"delta": {"reasoning_content": "</think>"}}]},
            {"choices": [{"delta": {"content": "This is my actual final response."}}]}
        ]
        
        for c in chunks:
            yield f"data: {json.dumps(c)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
        
    mock_agent.context.llm_client.stream_chat_completion = mock_stream
    
    body = {
        "messages": [{"role": "user", "content": "Execute something"}],
        "stream": False
    }
    
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        result = await mock_agent.handle_chat(body, background_tasks=MagicMock(), request_id=req_id)
        
    assert "This is my actual final response." in str(result)
    assert "<think>" not in str(result)
    assert "This is my internal thought loop." not in str(result)
    if isinstance(result, dict) and "messages" in result and len(result["messages"]) > 1:
        ai_turn = result["messages"][-1]
        assert "<think>" not in ai_turn.get("content", "")
