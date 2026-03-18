
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.agent import GhostAgent as Agent
from ghost_agent.tools.memory import tool_update_profile

@pytest.fixture
def mock_context():
    context = MagicMock()
    context.profile_memory = MagicMock()
    context.memory_system = MagicMock()
    context.llm_client = MagicMock()
    context.args = MagicMock()
    context.args.max_context = 8000
    context.args.temperature = 0.5
    return context

@pytest.mark.asyncio
async def test_agent_handle_chat_async_profile(mock_context):
    # Test that get_context_string is called via to_thread
    
    # Setup agent
    agent = Agent(mock_context)
    
    # Mock inputs
    user_message = "Hello"
    
    # Mock to_thread
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # We need to simulate return values.
        # Agent calls to_thread for:
        # 1. memory.search
        # 2. profile.get_context_string <-- This is what we check
        # 3. skills.get_playbook
        # ... and others
        
        async def side_effect(func, *args, **kwargs):
            if func == mock_context.profile_memory.get_context_string:
                return "Profile Context"
            return None
            
        mock_to_thread.side_effect = side_effect
        
        # We need to mock the rest of handle_chat to avoid crashes
        # Agent.handle_chat calls:
        # - memory/skills (async)
        # - profile (async -> TARGET)
        # - plan
        # - execute
        
        # It's better to just check the specific line if possible, or run a partial flow.
        # But handle_chat is complex.
        
        # Let's try to run handle_chat and assert interactions.
        # We need to mock _plan_and_execute or similar to stop execution early?
        # Or just let it run with mocks.
        
        # Mock LLM response to avoid network
        mock_context.llm_client.chat_completion.return_value = {"choices": [{"message": {"content": "Response"}}]}
        
        # Mock other async calls to avoid side effects
        agent._plan_and_execute = AsyncMock(return_value="Final Response")
        
        body = {"messages": [{"role": "user", "content": user_message}]}
        background_tasks = MagicMock()
        await agent.handle_chat(body, background_tasks)
        
        # Verify
        calls = mock_to_thread.call_args_list
        func_calls = [call.args[0] for call in calls]
        
        assert mock_context.profile_memory.get_context_string in func_calls

@pytest.mark.asyncio
async def test_tool_update_profile_async():
    # Test tool_update_profile
    profile_memory = MagicMock()
    memory_system = MagicMock()
    
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.return_value = "Updated"
        
        await tool_update_profile("Category", "Key", "Value", profile_memory, memory_system)
        
        # Verify
        assert mock_to_thread.call_count >= 1 # Could be 2 if smart_update is called
        # Check first call is profile update
        assert mock_to_thread.call_args_list[0][0][0] == profile_memory.update
        assert mock_to_thread.call_args_list[0][0][1] == "Category"
