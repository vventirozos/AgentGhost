import pytest
import copy
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from src.ghost_agent.core.agent import GhostAgent
from src.ghost_agent.tools.registry import get_active_tool_definitions
from src.ghost_agent.core.prompts import SYSTEM_PROMPT

@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.llm_client = AsyncMock()
    ctx.llm_client.image_gen_clients = ["http://local-gpu:5000"]
    ctx.llm_client.vision_clients = ["http://local-gpu:5000"]
    ctx.sandbox_dir = Path("/tmp/sandbox")
    ctx.sandbox_dir.mkdir(parents=True, exist_ok=True)
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.args = MagicMock()
    ctx.args.max_context = 8000
    ctx.args.temperature = 0.7
    ctx.args.smart_memory = 0.0
    return ctx

@pytest.mark.asyncio
async def test_agent_image_gen_limits_and_mutations(mock_context):
    agent = GhostAgent(mock_context)
    
    async def mock_fail_gen(**kwargs):
        return "Error: Image generation failed due to timeout."
    agent.available_tools["image_generation"] = mock_fail_gen
    
    mock_messages = [{"role": "user", "content": "Draw me a cat."}]
    
    # We must mock stream_chat_completion because handle_chat uses it internally
    async def mock_stream(*args, **kwargs):
        payload = args[0]
        # Check if the loop breaker message is present in the payload
        loop_msgs = [m for m in payload.get("messages", []) if "Tool 'image_generation' used too many times" in str(m.get("content", ""))]
        if loop_msgs:
            # If the breaker message is there, we yield text
            yield b'data: {"choices": [{"delta": {"content": "I cannot generate another image right now."}}]}\n\n'
        else:
            # Otherwise we yield a tool call attempt (which will trigger the loop breaker on the next turn)
            yield b'data: {"choices": [{"delta": {"tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "image_generation", "arguments": "{\\"prompt\\": \\"A cat\\"}"}}]}}]}\n\n'
        yield b'data: [DONE]\n\n'

    mock_context.llm_client.stream_chat_completion = mock_stream
    
    body = {"messages": mock_messages, "model": "Qwen-Test"}
    
    with patch("src.ghost_agent.core.agent.pretty_log"):
        # `handle_chat` consumes the stream entirely if stream_response is False (default)
        result = await agent.handle_chat(body, background_tasks=MagicMock())
        response = result[0] if isinstance(result, tuple) else result
        
        # We can check the internal loop history injected in the payload:
        assert isinstance(response, str)
        assert "cannot generate another image" in response

@pytest.mark.asyncio
async def test_agent_image_gen_forces_final_response(mock_context):
    agent = GhostAgent(mock_context)

    mock_messages = [{"role": "user", "content": "Draw me a dog."}]

    # We must mock stream_chat_completion
    call_tracker = {"count": 0}
    async def mock_stream(*args, **kwargs):
        call_tracker["count"] += 1
        if call_tracker["count"] == 1:
            yield b'data: {"choices": [{"delta": {"tool_calls": [{"id": "call_456", "type": "function", "function": {"name": "image_generation", "arguments": "{\\"prompt\\": \\"A dog\\"}"}}]}}]}\n\n'
        else:
            yield b'data: {"choices": [{"delta": {"content": "SUCCESS: Here is your dog: ![Image](/api/download/test.png)"}}]}\n\n'
        yield b'data: [DONE]\n\n'

    mock_context.llm_client.stream_chat_completion = mock_stream
    
    # Mock the tool registry to actually return a success string when called
    async def mock_image_gen(**kwargs):
        return "SUCCESS: Image generated... ![Image](/api/download/test.png)"
    
    agent.available_tools = {"image_generation": mock_image_gen}
    
    body = {"messages": mock_messages, "model": "Qwen-Test"}
    
    with patch("src.ghost_agent.core.agent.pretty_log"):
        # `handle_chat` will invoke the tool, then since image_generation forces a final response,
        # it should loop back to call the LLM ONE MORE TIME to process the tool result.
        result = await agent.handle_chat(body, background_tasks=MagicMock())
        response = result[0] if isinstance(result, tuple) else result
        
        # It should have requested the LLM TWICE
        assert call_tracker["count"] == 2
        
        # Checking that the forced response string is what `handle_chat` returned or logged
        last_msg = body["messages"][-1]
        last_content = last_msg.get("content", "") if isinstance(last_msg, dict) else ""
        assert "![Image]" in last_content

def test_prompts_and_registry_image_gen_instructions():
    # Verify SYSTEM_PROMPT contains the short prompt and vision instructions
    assert "EXACT MODE" in SYSTEM_PROMPT
    assert "ENHANCED MODE" in SYSTEM_PROMPT
    assert "IMAGINATION MODE" in SYSTEM_PROMPT
    assert "short (CLIP limit)" in SYSTEM_PROMPT
    assert "DO NOT rewrite or alter their core idea" in SYSTEM_PROMPT
    assert "first use `vision_analysis` (`describe_picture`)" in SYSTEM_PROMPT
    
    # Verify tool registry descriptions
    mock_ctx = MagicMock()
    mock_ctx.llm_client = MagicMock()
    mock_ctx.llm_client.image_gen_clients = ["http://gpu"]
    mock_ctx.llm_client.vision_clients = ["http://gpu"]
    
    tools = get_active_tool_definitions(mock_ctx)
    
    img_gen_def = next((t for t in tools if t.get("function", {}).get("name") == "image_generation"), None)
    assert img_gen_def is not None
    
    desc = img_gen_def["function"]["description"]
    assert "EXACT" in desc
    assert "ENHANCED" in desc
    assert "IMAGINATION" in desc
    assert "Preserve the user's exact subject" in desc
    assert "under 70 words" in desc
    
    param_desc = img_gen_def["function"]["parameters"]["properties"]["prompt"]["description"]
    assert "EXACTLY as they described it" in param_desc
    assert "SD 1.5 quality boosters" in param_desc
    assert "high-entropy prompt" in param_desc
    
    vision_def = next((t for t in tools if t.get("function", {}).get("name") == "vision_analysis"), None)
    assert vision_def is not None
    assert "analyze generated images in your sandbox if the user complains" in vision_def["function"]["description"]
