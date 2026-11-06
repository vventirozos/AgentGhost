import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.dream import Dreamer
from ghost_agent.core.prompts import QWEN_TOOL_PROMPT

@pytest.mark.asyncio
async def test_agent_think_tag_stripping():
    # Test that handle_chat strips <think> tags from the final message content
    mock_context = MagicMock()
    mock_context.args.perfect_it = False
    mock_context.args.max_context = 4000
    mock_context.args.temperature = 0.5
    mock_context.profile_memory = None
    mock_context.graph_memory = None
    mock_context.memory_system = None
    mock_context.args.smart_memory = 0.0
    
    mock_client = AsyncMock()
    # Mock stream_chat_completion to return an async generator yielding <think> content
    async def mock_stream(*args, **kwargs):
        chunks = [
            b'data: {"choices": [{"delta": {"content": "<think>\\nI am thinking\\n</think>\\nHello User!"}}]}',
            b'data: [DONE]'
        ]
        for c in chunks:
            yield c
            
    mock_client.stream_chat_completion = mock_stream
    mock_context.llm_client = mock_client
    
    agent = GhostAgent(mock_context)
    body = {
        "model": "test-model",
        "stream": True,
        "messages": [{"role": "user", "content": "Run a calculation for me"}]
    }
    
    final_ai, ct, rid = await agent.handle_chat(body, None)
    
    # Verify the context was appended with the stripped content
    assert "<think>" not in final_ai
    assert "Hello User!" in final_ai

@pytest.mark.asyncio
async def test_agent_payload_penalties():
    # Test that the LLM request payload gets the presence and frequency penalties
    mock_context = MagicMock()
    mock_context.args.perfect_it = False
    mock_context.args.max_context = 4000
    mock_context.args.temperature = 0.5
    mock_context.profile_memory = None
    mock_context.graph_memory = None
    mock_context.memory_system = None
    mock_context.args.smart_memory = 0.0
    
    mock_client = AsyncMock()
    async def mock_stream(*args, **kwargs):
        assert "payload" not in kwargs, "Just checking stream arguments"
        payload = args[0]
        assert payload.get("presence_penalty") == 0.20
        assert payload.get("frequency_penalty") == 0.20
        yield b'data: [DONE]'
        
    mock_client.stream_chat_completion = mock_stream
    mock_context.llm_client = mock_client
    
    agent = GhostAgent(mock_context)
    body = {
        "model": "test-model",
        "stream": True,
        "messages": [{"role": "user", "content": "Run a calculation for me"}]
    }
    
    await agent.handle_chat(body, None)


def test_prompt_updates():
    # Check that prompt files have the requested constraints
    assert "ANTI-PARALYSIS" in QWEN_TOOL_PROMPT
    assert "strictly forbidden to brainstorm multiple alternatives" in QWEN_TOOL_PROMPT
