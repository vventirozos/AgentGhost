
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def mock_context():
    context = MagicMock(spec=GhostContext)
    context.args = MagicMock()
    context.args.temperature = 0.5
    context.args.max_context = 8000
    context.args.smart_memory = 0.0
    context.args.use_planning = False
    context.sandbox_dir = "/tmp/sandbox"
    context.memory_system = None
    context.profile_memory = None
    context.skill_memory = None
    # Mock scratchpad to avoid AttributeError
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = "Mock Scratchpad Content"
    # Mock LLM client
    context.llm_client = MagicMock()
    context.llm_client.chat_completion = AsyncMock()
    return context

@pytest.mark.asyncio
async def test_handle_chat_json_parsing_failure(mock_context):
    """Test that invalid JSON arguments in XML tool calls emit system_parse_error."""
    mock_context.args.model = "qwen2.5" # Trigger XML processing
    agent = GhostAgent(mock_context)
    
    # Mock LLM response with broken XML tool block
    mock_message = {
        "role": "assistant",
        "content": "Using tool...\n<tool_call>\n{\"name\": \"write_file\", \"arguments\": '{\"filename\": \"test.txt\", \"content\": \"missing_brace\"'\n</tool_call>"
    }
    
    mock_context.llm_client.chat_completion.side_effect = [
        {"choices": [{"message": mock_message}]}, # Turn 1
        {"choices": [{"message": {"role": "assistant", "content": "Fixed"}}]} # Turn 2
    ]
    
    messages = [{"role": "user", "content": "Write a file"}]
    background_tasks = MagicMock()
    
    final_content, _, _ = await agent.handle_chat({"messages": messages, "model": "qwen2.5"}, background_tasks)
    
    assert mock_context.llm_client.chat_completion.call_count >= 2
    
    # Get the arguments of the second call
    second_call_args = mock_context.llm_client.chat_completion.call_args_list[1]
    payload = second_call_args[0][0] 
    sent_messages = payload["messages"]
    
    # We expect a tool message with the system_parse_error response (which might be converted to role=user for qwen)
    error_tool_msg = next((m for m in sent_messages if "SYSTEM ERROR" in str(m.get("content"))), None)
    
    assert error_tool_msg is not None, f"Agent did not inject system_parse_error feedback. Sent messages were: {sent_messages}"
    assert "The JSON inside your <tool_call> was invalid" in error_tool_msg["content"]

