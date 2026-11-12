import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from src.ghost_agent.core.prompts import SYSTEM_PROMPT
from src.ghost_agent.tools.vision import tool_vision_analysis
from src.ghost_agent.core.agent import GhostAgent, GhostContext


def test_code_system_prompt_hardened():
    """Verify that the prompt requires exact Markdown formatting and raw filenames."""
    # §4UI: the main prompt carries it (the coding persona is gone)
    assert "![Image](/api/download/" in SYSTEM_PROMPT
    assert "Use the raw filename" in SYSTEM_PROMPT

@pytest.mark.asyncio
async def test_vision_analysis_path_hallucination_guard():
    """Verify that tool_vision_analysis dynamically strips the /api/download/ prefix."""
    mock_llm = MagicMock()
    mock_llm.vision_clients = [{"client": "dummy"}]
    mock_llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "vision data"}}]})
    
    dummy_sandbox = Path("/tmp/dummy_sandbox")
    
    with patch("src.ghost_agent.tools.vision._get_safe_path") as mock_safe_path, \
         patch("src.ghost_agent.tools.vision.asyncio.to_thread") as mock_to_thread:
        
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.read_bytes = lambda: b"fake_image_data"
        mock_safe_path.return_value = mock_path_obj
        mock_to_thread.return_value = b"fake_image_data"
        
        # 1. Test with the hallucinated path
        res1 = await tool_vision_analysis(
            action="describe_picture",
            target="/api/download/test_image.jpg",
            llm_client=mock_llm,
            sandbox_dir=dummy_sandbox
        )
        
        mock_safe_path.assert_called_with(dummy_sandbox, "test_image.jpg")
        
        # 2. Test without the prefix
        res2 = await tool_vision_analysis(
            action="describe_picture",
            target="normal_image.png",
            llm_client=mock_llm,
            sandbox_dir=dummy_sandbox
        )
        mock_safe_path.assert_called_with(dummy_sandbox, "normal_image.png")

@pytest.fixture
def agent():
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.7
    ctx.args.max_context = 8000
    ctx.args.smart_memory = 0.0
    ctx.args.perfect_it = False
    
    ctx.llm_client = MagicMock()
    
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.memory_system = MagicMock()
    ctx.memory_system.search = MagicMock(return_value="")
    ctx.cached_sandbox_state = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all.return_value = "No thoughts."
    
    agent_inst = GhostAgent(ctx)
    return agent_inst

@pytest.mark.asyncio
async def test_agent_save_trigger_fix(agent):
    """
    If user says 'save the environment', it should NOT force a meta intent 
    and should NOT inject the 'CRITICAL: You have not fulfilled...' message.
    """
    agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "I have saved the environment.", "tool_calls": []}}]
    })
    
    body = {"messages": [{"role": "user", "content": "Please save the environment variables to a file."}], "model": "Qwen-Test"}
    
    final_content, _, _ = await agent.handle_chat(body, background_tasks=MagicMock())
    
    # The universal adapter means stream_chat_completion is natively invoked in run_turn now.
    last_call_args = None
    if isinstance(agent.context.llm_client.chat_completion, (AsyncMock, MagicMock)) and agent.context.llm_client.chat_completion.call_count > 0:
        last_call_args = agent.context.llm_client.chat_completion.call_args
    elif isinstance(agent.context.llm_client.stream_chat_completion, (AsyncMock, MagicMock)) and agent.context.llm_client.stream_chat_completion.call_count > 0:
        last_call_args = agent.context.llm_client.stream_chat_completion.call_args
    
    # If using the wrapper, we might need to look at how we patch
    if not last_call_args:
        # Let's just patch handle_chat's argument list? No, the payload is in the mock.
        # But wait, MagicMock closure might have captured the original chat_completion if not careful.
        messages = body["messages"] # The body is mutated locally during chat processing!
    else:
        messages = last_call_args.args[0]["messages"]
        
    critical_injected = False
    for m in messages:
        if "CRITICAL: You have not fulfilled the learning/profile instructions" in str(m.get("content", "")):
            critical_injected = True
            
    assert not critical_injected, "The 'save' keyword incorrectly triggered the meta-intent blocker."
