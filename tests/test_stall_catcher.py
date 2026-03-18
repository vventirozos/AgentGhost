import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.core.llm import LLMClient

class MockArgs:
    provider = "mlx"
    model = "qwen2.5-coder-7B-MLX"
    system_prompt = "You are a helpful assistant."
    max_context = 8000
    temperature = 0.5
    max_temperature = 0.8
    smart_memory = 0.0

@pytest.fixture
def mock_context():
    context = MagicMock(spec=GhostContext)
    context.llm_client = AsyncMock(spec=LLMClient)
    context.args = MockArgs()
    context.os_type = "mac"
    context.current_dir = "/tmp"
    context.sandbox_dir = "/tmp"
    context.sandbox_mapped_dir = "/tmp"
    context.tool_system = MagicMock()
    context.memory_system = MagicMock()
    context.profile_memory = None
    context.skill_memory = None
    return context

@pytest.mark.asyncio
async def test_stall_catcher_conversational_filler(mock_context):
    agent = GhostAgent(context=mock_context)
    agent.temperature = 0.5
    agent.max_temperature = 0.8
    
    # Mock LLM to return conversational filler with no tool calls
    mock_context.llm_client.chat_completion.side_effect = [
        # First call: Conversational filler instead of XML
        {"choices": [{"message": {"content": "I understand. I will now create the file for you as requested."}}]},
        # Second call: The actual expected XML response (to prevent infinite loop)
        {"choices": [{"message": {"content": "<tool_call>\n{\"name\": \"execute\", \"parameters\": {\"command\": \"ls\"}}\n</tool_call>"}}]},
        # Third call: Thinking block (agent.py loop might do another step here)
        {"choices": [{"message": {"content": "<think>Wait</think>"}}]},
        # Fourth call: Final conversational response after tool execution
        {"choices": [{"message": {"content": "The command executed successfully. " * 10}}]},
        {"choices": [{"message": {"content": "Done"}}]} # fallback
    ]
    agent.context.tool_system.get_tool_schemas.return_value = {"tools": []}
    agent.context.tool_system.format_tools_for_prompt.return_value = "Tools available: execute"
    agent.available_tools = {"execute": AsyncMock(return_value="mock output")}
    agent._prune_context = AsyncMock(return_value=[{"role": "user", "content": "please run ls"}])
    agent._run_critic_check = AsyncMock(return_value=[])
    
    await agent.handle_chat({"messages": [{"role": "user", "content": "please run ls"}]}, MagicMock())
    
    # Verify the LLM was called 4 times
    # 1. Stall, 2. XML call, 3. Empty thinking stall (because of no tools maybe?), 4. Final
    assert mock_context.llm_client.chat_completion.call_count >= 3
    
    # Check the messages sent on the second call
    second_call_messages = mock_context.llm_client.chat_completion.call_args_list[1][0][0]["messages"]
    
    # The prompt should contain the SYSTEM ALERT about conversational filler
    system_alert = next((m["content"] for m in reversed(second_call_messages) if m["role"] == "user" and "SYSTEM ALERT" in m["content"]), None)
    
    assert system_alert is not None
    assert "Caught conversational filler instead of tool call" not in system_alert  # verify the actual message used
    assert "FAILED to output the actual XML `<tool_call>` block" in system_alert

@pytest.mark.asyncio
async def test_stall_catcher_empty_stall(mock_context):
    agent = GhostAgent(context=mock_context)
    agent.temperature = 0.5
    agent.max_temperature = 0.8
    
    # Mock LLM to return nothing at all
    mock_context.llm_client.chat_completion.side_effect = [
        # First call: Empty response
        {"choices": [{"message": {"content": "   \n  "}}]},
        # Second call: The actual expected XML response (to prevent infinite loop)
        {"choices": [{"message": {"content": "<tool_call>\n{\"name\": \"execute\", \"parameters\": {\"command\": \"ls\"}}\n</tool_call>"}}]},
        # Third call: Final empty response
        {"choices": [{"message": {"content": "Command executed successfully. " * 10}}]}
    ]
    agent.context.tool_system.get_tool_schemas.return_value = {"tools": []}
    agent.context.tool_system.format_tools_for_prompt.return_value = "Tools available: execute"
    agent.available_tools = {"execute": AsyncMock(return_value="mock output")}
    agent._prune_context = AsyncMock(return_value=[{"role": "user", "content": "please run ls"}])
    agent._run_critic_check = AsyncMock(return_value=[])
    
    await agent.handle_chat({"messages": [{"role": "user", "content": "please run ls"}]}, MagicMock())
    
    # Check the messages sent on the second call
    second_call_messages = mock_context.llm_client.chat_completion.call_args_list[1][0][0]["messages"]
    
    # The prompt should contain the SYSTEM ALERT about empty stall
    system_alert = next((m["content"] for m in reversed(second_call_messages) if m["role"] == "user" and "SYSTEM ALERT" in m["content"]), None)
    
    assert system_alert is not None
    assert "stopped abruptly without outputting a valid XML" in system_alert
