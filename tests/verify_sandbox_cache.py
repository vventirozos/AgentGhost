
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.agent import GhostAgent, GhostContext
import json

@pytest.fixture
def agent():
    context = MagicMock(spec=GhostContext)
    context.args = MagicMock()
    context.args.max_context = 8000
    context.args.smart_memory = 0.0
    context.args.temperature = 0.0
    context.args.use_planning = False
    context.llm_client = AsyncMock()
    context.memory_system = MagicMock()
    context.profile_memory = MagicMock()
    context.skill_memory = MagicMock()
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = "Mock Scratchpad"
    context.sandbox_dir = "/tmp/sandbox"
    context.memory_dir = "/tmp/memory"
    context.tor_proxy = None
    
    # Initialize cache with some value
    context.cached_sandbox_state = "File A, File B"
    
    # Mock available tools
    with patch("ghost_agent.core.agent.get_available_tools") as mock_get_tools:
        mock_tools = {
            "file_system": AsyncMock(return_value="Success"),
            "execute": AsyncMock(return_value="Exit Code: 0"),
        }
        mock_get_tools.return_value = mock_tools
        agent = GhostAgent(context)
        return agent

@pytest.mark.asyncio
async def test_cache_invalidation_on_write(agent):
    """Verify cache is cleared on file write."""
    # Setup: Cache is "File A, File B"
    agent.context.cached_sandbox_state = "File A, File B"
    
    # Call file_system with write
    call_write = {
        "id": "1", 
        "function": {
            "name": "file_system", 
            "arguments": json.dumps({"operation": "write", "path": "file.txt", "content": "hi"})
        }
    }
    
    agent.context.llm_client.chat_completion.side_effect = [
        {"choices": [{"message": {"role": "assistant", "tool_calls": [call_write]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}
    ]
    
    await agent.handle_chat({"messages": [{"role": "user", "content": "write file"}]}, MagicMock())
    
    # Assert cache is cleared
    assert agent.context.cached_sandbox_state is None

@pytest.mark.asyncio
async def test_cache_preservation_on_read(agent):
    """Verify cache is NOT cleared on file read."""
    agent.context.cached_sandbox_state = "File A, File B"
    
    call_read = {
        "id": "2", 
        "function": {
            "name": "file_system", 
            "arguments": json.dumps({"operation": "read", "path": "file.txt"})
        }
    }
    
    agent.context.llm_client.chat_completion.side_effect = [
        {"choices": [{"message": {"role": "assistant", "tool_calls": [call_read]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}
    ]
    
    await agent.handle_chat({"messages": [{"role": "user", "content": "read file"}]}, MagicMock())
    
    # Assert cache is PRESERVED
    assert agent.context.cached_sandbox_state == "File A, File B"
