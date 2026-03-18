import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from src.ghost_agent.core.prompts import CODE_SYSTEM_PROMPT
from src.ghost_agent.tools.vision import tool_vision_analysis
from src.ghost_agent.core.agent import GhostAgent, GhostContext

def test_code_system_prompt_hardened():
    """Verify that the prompt requires exact Markdown formatting and raw filenames."""
    assert "![Image](/api/download/filename.ext)" in CODE_SYSTEM_PROMPT
    assert "Use the raw filename" in CODE_SYSTEM_PROMPT

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
    ctx.args.use_planning = False
    ctx.args.perfect_it = False
    
    ctx.llm_client = MagicMock()
    
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_context_string.return_value = ""
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

@pytest.mark.asyncio
async def test_agent_streaming_flush(agent):
    """
    Ensure `stream_wrapper` captures and yields accumulated intermediate text 
    as the first stream chunk for the UI.
    """
    
    # We simulate the first turn calling a tool normally
    tool_call_msg = {"choices": [{"message": {"content": "Here is an image: ![Image](/api/download/foo.jpg)", "tool_calls": [{"id": "t1", "function": {"name": "system_utility", "arguments": "{}"}}]}}]}
    final_msg = {"choices": [{"message": {"content": "All done.", "tool_calls": []}}]}
    
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[tool_call_msg, final_msg])
    
    # We will force final response loop trigger by setting an overuse of the tool
    # so that force_final_response becomes True and stream_wrapper gets hit.
    # We will pre-populate tool_usage to trigger max uses (20 for most).
    agent.available_tools["system_utility"] = AsyncMock(return_value="System Ok")
    
    agent.context.scratchpad.list_all.return_value = "No thoughts."
    
    body = {"messages": [{"role": "user", "content": "execute a test task"}], "model": "Qwen-Test", "stream": True}
    
    with patch.object(GhostAgent, "_prune_context", side_effect=lambda x, **kwargs: x):
        pass

    agent.context.args.use_planning = True
    p_data_first = {
        "choices": [{"message": {"content": '{"thought": "do it", "next_action_id": "task1", "required_tool": "system_utility"}'}}]
    }
    p_data_second = {
        "choices": [{"message": {"content": '{"thought": "done", "next_action_id": "none", "required_tool": "none"}'}}]
    }
    
    # Stateful generator to yield tool calls first, then text
    call_count = {"stream": 0}
    async def async_generator(*args, **kwargs):
        call_count["stream"] += 1
        if call_count["stream"] == 1:
            yield b'data: {"choices": [{"delta": {"content": "Here is an image: ![Image](/api/download/foo.jpg)", "tool_calls": [{"index": 0, "id": "t1", "function": {"name": "system_utility", "arguments": "{}"}}]}}]}\n\n'
        else:
            yield b'data: {"choices": [{"delta": {"content": "All done."}}]}\n\n'
    agent.context.llm_client.stream_chat_completion = async_generator
    
    # To cleanly test the streaming flush, we patch the agent's turn evaluation directly.
    # We simulate that the LLM has accumulated intermediate text, and now is_final_generation triggers the stream.
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        p_data_first,  # Planner turn 1: decides to use tool
        p_data_second  # Planner turn 2: decides to stop
    ]) 
    
    body = {"messages": [{"role": "user", "content": "execute a python script to run a test"}], "model": "Qwen-Test", "stream": True}
    
    with patch("src.ghost_agent.core.agent.GhostAgent._get_recent_transcript", return_value=""):
        res = await agent.handle_chat(body, background_tasks=MagicMock())
    
    assert isinstance(res, tuple), "Expected tuple (stream_generator, time, id) from handle_chat"
    stream_gen = res[0]
    
    if isinstance(stream_gen, str):
        pytest.fail(f"handle_chat returned a string instead of generator, probably crashed: {stream_gen}")
    
    chunks = []
    async for chunk in stream_gen:
        chunks.append(chunk)

    first_chunk = chunks[0].decode('utf-8')
    assert "Here is an image: ![Image](/api/download/foo.jpg)\\n\\n" in first_chunk
    
    second_chunk = chunks[1].decode('utf-8')
    assert "All done." in second_chunk
