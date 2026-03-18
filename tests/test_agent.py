
import pytest
from unittest.mock import MagicMock
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def mock_agent():
    ctx = MagicMock(spec=GhostContext)
    ctx.llm_client = MagicMock()
    ctx.llm_client.vision_clients = None
    ctx.args = MagicMock()
    ctx.args.max_context = 4000
    # Mocking attributes accessed in specific methods if needed
    agent = GhostAgent(context=ctx)
    return agent

def test_prepare_planning_context_truncation(mock_agent):
    # Case 1: Short output
    tools_run = [{"content": "Short output"}]
    result = mock_agent._prepare_planning_context(tools_run)
    assert result == "Tool [unknown]: Short output"

    # Case 2: Long output (Over 5000 chars)
    # create a string of 6000 chars
    long_content = "A" * 30000
    tools_run = [{"content": long_content}]
    result = mock_agent._prepare_planning_context(tools_run)
    
    # Expect truncation
    expected_marker = "\n\n... [TRUNCATED: Tool output too long. Showing top results only.]"
    
    # Result = "Tool [unknown]: " + 4800-chars + marker
    # The length depends on the "Tool [unknown]: " prefix length which is 16 chars
    # So 16 + 4800 + len(marker) = 4816 + 65 = 4881 approximately
    
    assert len(result) < 10000
    assert expected_marker in result
    assert "A" * 4000 in result
    assert result.endswith(expected_marker)

def test_process_rolling_window_sliding(mock_agent):
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "tool", "name": "exec", "content": "Result A"},
        {"role": "tool", "name": "exec", "content": "Result A"}, # Duplicate
        {"role": "assistant", "content": "Memory updated..."},  # Meta-chatter
        {"role": "assistant", "content": "Real response"}
    ]
    
    # The new logic is a pure sliding window. No deduplication, no filtering.
    clean = mock_agent.process_rolling_window(messages, max_tokens=1000)
    
    # Check that all messages are preserved (they all fit in 1000 tokens)
    assert len(clean) == 5
    
    tool_msgs = [m for m in clean if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["content"] == "Result A"
    assert tool_msgs[1]["content"] == "Result A"
    
    assist_msgs = [m for m in clean if m["role"] == "assistant"]
    assert len(assist_msgs) == 2
    assert assist_msgs[1]["content"] == "Real response"

def test_agent_semaphore_initialization(mock_agent):
    """
    Verify that the agent's semaphore is initialized to 10 to allow
    concurrent background tasks and user chats.
    """
    # In asyncio.Semaphore, the initial value is stored in _value (internal) 
    # but we can also just check if we can acquire it multiple times in a loop 
    # or check the internal attribute for a unit test.
    # checking internal _value is implementation specific but simplest for unit test.
    assert mock_agent.agent_semaphore._value == 10

@pytest.mark.asyncio
async def test_agent_streaming(mock_agent):
    from unittest.mock import AsyncMock

    # With the planner disabled, the agent goes directly to final generation on the first turn.
    # Returning empty content + no tool_calls forces is_final_generation=True which triggers
    # stream_chat_completion when stream=True.
    async def mock_final(*args, **kwargs):
        return {
            "choices": [{"message": {"content": '{"thought": "done", "next_action_id": "none", "required_tool": "none"}', "tool_calls": []}}]
        }
    mock_agent.context.llm_client.chat_completion = MagicMock(side_effect=mock_final)

    # Mock the stream path
    async def fake_stream(*args, **kwargs):
        yield b"data: {\"choices\": [{\"delta\": {\"content\": \"hello\"}}]}\n\n"
        yield b"data: [DONE]\n\n"

    mock_agent.context.llm_client.stream_chat_completion = MagicMock(return_value=fake_stream())
    
    # Mock context config
    mock_agent.context.args = MagicMock()
    mock_agent.context.args.use_planning = True
    mock_agent.context.args.smart_memory = 0.0
    mock_agent.context.args.max_context = 4000
    mock_agent.context.args.temperature = 0.5
    mock_agent.context.scratchpad = MagicMock()
    mock_agent.context.scratchpad.list_all.return_value = ""
    mock_agent.context.profile_memory = MagicMock()
    mock_agent.context.memory_system = None
    mock_agent.context.skill_memory = None

    def sync_return(): return ""
    mock_agent.context.profile_memory.get_context_string.return_value = sync_return()

    body = {
        "messages": [{"role": "user", "content": "calculate math"}],
        "model": "test-model",
        "stream": True
    }
    
    bg_tasks = MagicMock()
    
    result_content, created_time, req_id = await mock_agent.handle_chat(body, bg_tasks)
    
    # Verify we get an async generator wrapper back
    assert hasattr(result_content, '__aiter__')
    
    # Verify the yielded content
    chunks = []
    async for chunk in result_content:
        chunks.append(chunk)
        
    assert len(chunks) == 2
    assert chunks[0] == b"data: {\"choices\": [{\"delta\": {\"content\": \"hello\"}}]}\n\n"
    assert chunks[1] == b"data: [DONE]\n\n"
