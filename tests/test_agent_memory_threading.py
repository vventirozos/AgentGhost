import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.mark.asyncio
async def test_agent_smart_updates_thread_offload():
    """Verify that profile_memory updates are offloaded to background thread"""
    profile_memory = MagicMock()
    profile_memory.update = MagicMock()
    
    context = GhostContext(
        sandbox_dir="/tmp",
        memory_dir="/tmp",
        tor_proxy=None,
        args=MagicMock(model="test")
    )
    context.memory_system = MagicMock()
    context.profile_memory = profile_memory
    
    agent = GhostAgent(context)
    agent.task_tree = MagicMock()
    
    # To trigger smart memory's threading wrap for profile_update, we need an LLM response with score >= 0.9 and profile_update
    fake_llm_response = {
        "choices": [
            {
                "message": {
                    "content": '{"score": 0.95, "fact": "User likes red", "profile_update": {"category": "preferences", "key": "color", "value": "red"}}'
                }
            }
        ]
    }
    
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value=fake_llm_response)
    context.llm_client = mock_llm
    
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        await agent.run_smart_memory_task("i like the color red", "test_model", 0.7)
        
        # It's called twice! Once for `memory_system.search_advanced`, once for `memory_system.add`, and once for `profile_memory.update`
        # We check that ONE of the calls was to `profile_memory.update`
        calls = mock_to_thread.call_args_list
        profile_call = None
        for call in calls:
            args, _ = call
            if args[0] == profile_memory.update:
                profile_call = args
                break
                
        assert profile_call is not None, "to_thread was not called with profile_memory.update"
        assert profile_call[0] == profile_memory.update
        assert profile_call[1] == "preferences"
        assert profile_call[2] == "color"
        assert profile_call[3] == "red"
