
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def mock_context():
    ctx = MagicMock(spec=GhostContext)
    ctx.memory_system = MagicMock()
    ctx.skill_memory = MagicMock()
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all.return_value = "None."
    ctx.args = MagicMock()
    ctx.args.max_context = 4000
    ctx.args.temperature = 0.5
    ctx.args.use_planning = False
    ctx.args.smart_memory = 0.0
    ctx.llm_client = AsyncMock()
    ctx.llm_client.chat_completion.return_value = {"choices": [{"message": {"content": "Test response"}}]}
    return ctx

@pytest.mark.asyncio
async def test_memory_search_is_async(mock_context):
    """Verify memory.search is called via to_thread"""
    agent = GhostAgent(mock_context)
    
    # Setup mocks
    mock_context.memory_system.search.return_value = "Memory Context"
    
    # We need to mock asyncio.to_thread to verify it's used
    with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
        # Use side_effect to return different values based on input, or just return a string always
        # The issue might be that it's called multiple times (e.g. for playbook too)
        # So we ensure it returns a string that doesn't break other things
        mock_to_thread.return_value = "Memory Context"
        
        # Trigger handle_chat
        # "Hello" is a trivial trigger, preventing memory fetch. Use something else.
        body = {"messages": [{"role": "user", "content": "Please remember this important fact"}]}
        await agent.handle_chat(body, background_tasks=MagicMock())
        
        # Verify to_thread was called with memory_system.search
        found_call = False
        for call in mock_to_thread.call_args_list:
            # Check if the FIRST argument is the function we expect
            if call.args and call.args[0] == mock_context.memory_system.search:
                assert call.args[1] == "Please remember this important fact"
                found_call = True
                break
        
        assert found_call, f"memory_system.search was not offloaded to thread. Calls: {mock_to_thread.call_args_list}"

@pytest.mark.asyncio
async def test_playbook_context_is_async(mock_context):
    """Verify skill_memory.get_playbook_context is called via to_thread"""
    agent = GhostAgent(mock_context)
    
    # Setup mocks
    mock_context.memory_system.search.return_value = None # Skip memory search to isolate check or just ensure flow continues
    mock_context.skill_memory.get_playbook_context.return_value = "Playbook Context"

    with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.return_value = "Playbook Context"
        
        body = {"messages": [{"role": "user", "content": "Help me code"}]}
        await agent.handle_chat(body, background_tasks=MagicMock())
        
        # Verify to_thread was called with skill_memory.get_playbook_context
        found_call = False
        for call in mock_to_thread.call_args_list:
            if call.args and call.args[0] == mock_context.skill_memory.get_playbook_context:
                assert call.kwargs['query'] == "Help me code"
                found_call = True
                break
        
        assert found_call, "skill_memory.get_playbook_context was not offloaded to thread"
