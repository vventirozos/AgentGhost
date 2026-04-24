import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def mock_context():
    context = MagicMock(spec=GhostContext)
    
    mock_llm = AsyncMock()
    context.llm_client = mock_llm
    
    mock_args = MagicMock()
    mock_args.max_context = 10000
    mock_args.smart_memory = 0.0
    mock_args.use_planning = False
    mock_args.temperature = 0.7
    mock_args.perfect_it = False
    context.args = mock_args
    
    context.journal = MagicMock()
    context.memory_system = MagicMock()
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string = lambda: ""
    context.skill_memory = MagicMock()
    context.sandbox_dir = "/tmp/sandbox"
    context.cached_sandbox_state = None
    
    return context

@pytest.mark.asyncio
async def test_agent_raw_json_fallback(mock_context):
    mock_llm = mock_context.llm_client
    mock_llm.chat_completion.side_effect = [
        {
            "choices": [{"message": {"content": '{"name": "execute", "arguments": {"filename": "test.py", "content": "print(1)"}}'}}]
        },
        {
            "choices": [{"message": {"content": "SUCCESS: I have finished."}}]
        }
    ]
    
    agent = GhostAgent(context=mock_context)
    agent.available_tools = {"execute": AsyncMock(return_value="executed_successfully")}
    
    messages = [{"role": "user", "content": "Run script"}]
    body = {"messages": messages, "model": "test-model"}
    await agent.handle_chat(body, background_tasks=MagicMock(), request_id="req123")
    
    # Verify the tool was extracted and called successfully
    tools_called = [m["name"] for m in body["messages"] if m.get("role") == "tool"]
    assert "execute" in tools_called
    # Check that tool response says executed_successfully
    tool_responses = [m["content"] for m in body["messages"] if m.get("role") == "tool" and m.get("name") == "execute"]
    assert len(tool_responses) > 0
    assert tool_responses[0] == "executed_successfully"

@pytest.mark.asyncio
async def test_agent_conversational_filler_fallback(mock_context):
    mock_llm = mock_context.llm_client
    # Simulate conversational filler mentioning execute but not doing it
    mock_llm.chat_completion.side_effect = [
        {
            "choices": [{"message": {"content": 'I will use the execute tool.'}}]
        },
        {
            "choices": [{"message": {"content": "SUCCESS"}}]
        }
    ]
    
    agent = GhostAgent(context=mock_context)
    agent.available_tools = {"execute": AsyncMock(return_value="executed_successfully")}
    
    messages = [{"role": "user", "content": "Run script"}]
    body = {"messages": messages, "model": "test-model"}
    await agent.handle_chat(body, background_tasks=MagicMock(), request_id="req123")
    
    # We expect our fallback to have appended the SYSTEM ALERT message warning against conversational filler
    alert_messages = [m for m in body["messages"] if m.get("role") == "user" and "SYSTEM ALERT: You provided conversational text mentioning the tool" in m.get("content", "")]
    assert len(alert_messages) == 1
    assert "`execute`" in alert_messages[0]["content"]
    assert "but you DID NOT output the actual XML `<tool_call>` block" in alert_messages[0]["content"]
