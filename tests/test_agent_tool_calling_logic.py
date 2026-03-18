import pytest
import json
import uuid
import asyncio
from unittest.mock import AsyncMock, MagicMock
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def mock_context():
    args = MagicMock()
    args.model = "test-model"
    args.use_planning = False
    args.temperature = 0.0
    args.max_context = 4000
    args.smart_memory = 0.0
    
    context = GhostContext(args=args, sandbox_dir=None, memory_dir=None, tor_proxy=None)
    context.llm_client = MagicMock()
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = "Empty"
    return context

@pytest.mark.asyncio
async def test_agent_standard_tool_calling(mock_context):
    """Test standard OpenAI compatible tool_calls array processing."""
    agent = GhostAgent(mock_context)
    
    # Mock a sync tool
    mock_tool = AsyncMock(return_value="SUCCESS: Mock tool worked")
    agent.available_tools = {"mock_tool": mock_tool}
    
    # Mock LLM response format for standard tool_calls
    mock_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Running the tool now.",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "mock_tool",
                        "arguments": json.dumps({"param": "value"})
                    }
                }]
            }
        }]
    }
    
    # Second response to finish the trace
    mock_response_finish = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "SUCCESS: Process finished. " * 20,
                "tool_calls": []
            }
        }]
    }
    
    mock_context.llm_client.chat_completion = AsyncMock(side_effect=[mock_response, mock_response_finish])
    
    # Call the handle_chat
    background_tasks = MagicMock()
    body = {"messages": [{"role": "user", "content": "Run mock tool"}]}
    
    final_output, _, _ = await agent.handle_chat(body, background_tasks)
    
    # The tool should have been called
    mock_tool.assert_called_once_with(param="value")
    
    assert "Process finished." in final_output

@pytest.mark.asyncio
async def test_agent_qwen_syntax_healer(mock_context):
    """Test backend syntax healer extracting <tool_call> tags when native tool_calls are empty."""
    agent = GhostAgent(mock_context)
    
    mock_tool = AsyncMock(return_value="SUCCESS: Healer worked")
    agent.available_tools = {"mock_tool": mock_tool}
    
    # Mock LLM response format for Qwen missing native tool_calls but having tags in content
    mock_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "<tool_call>\n{\"name\": \"mock_tool\", \"arguments\": {\"action\": \"test\"}}\n</tool_call>",
                "tool_calls": None
            }
        }]
    }
    
    mock_response_finish = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "All done via syntax healer.",
                "tool_calls": None
            }
        }]
    }
    
    mock_context.llm_client.chat_completion = AsyncMock(side_effect=[mock_response, mock_response_finish])
    background_tasks = MagicMock()
    body = {"messages": [{"role": "user", "content": "Healer tool trigger"}]}
    
    final_output, _, _ = await agent.handle_chat(body, background_tasks)
    
    # Tool should have been parsed and called
    mock_tool.assert_called_once_with(action="test")
    
    # Confirm the raw <tool_call> tag is stripped from final output
    assert "<tool_call>" not in final_output
    assert "All done via syntax healer." in final_output

@pytest.mark.asyncio
async def test_agent_executes_multiple_tools(mock_context):
    """Test that multiple tools in a single model turn run perfectly."""
    agent = GhostAgent(mock_context)
    
    mock_tool_1 = AsyncMock(return_value="T1 OK")
    mock_tool_2 = AsyncMock(return_value="T2 OK")
    agent.available_tools = {"mock_tool_1": mock_tool_1, "mock_tool_2": mock_tool_2}
    
    mock_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Running two tools.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "mock_tool_1",
                            "arguments": "{}"
                        }
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "mock_tool_2",
                            "arguments": "{}"
                        }
                    }
                ]
            }
        }]
    }
    
    mock_response_finish = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "SUCCESS: Batch completed. " * 20,
                "tool_calls": []
            }
        }]
    }
    
    mock_context.llm_client.chat_completion = AsyncMock(side_effect=[mock_response, mock_response_finish])
    
    background_tasks = MagicMock()
    body = {"messages": [{"role": "user", "content": "Parallel tools test"}]}
    
    final_output, _, _ = await agent.handle_chat(body, background_tasks)
    
    mock_tool_1.assert_called_once_with()
    mock_tool_2.assert_called_once_with()
    assert "Batch completed." in final_output
