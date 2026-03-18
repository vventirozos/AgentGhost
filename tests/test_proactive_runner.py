
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.main import proactive_runner

@pytest.mark.asyncio
async def test_proactive_runner_integration():
    """Test that proactive_runner calls GLOBAL_AGENT.handle_chat with correct payload."""
    
    # Mock GLOBAL_AGENT and GLOBAL_CONTEXT in ghost_agent.main
    # Since they are global variables, we can patch them.
    
    mock_agent = MagicMock()
    mock_agent.handle_chat = AsyncMock()
    
    mock_context = MagicMock()
    
    with patch("ghost_agent.main.GLOBAL_AGENT", mock_agent), \
         patch("ghost_agent.main.GLOBAL_CONTEXT", mock_context):
        
        task_id = "test_task_123"
        prompt = "Check usage stats"
        
        await proactive_runner(task_id, prompt)
        
        # Verify handle_chat was called
        assert mock_agent.handle_chat.called
        
        # Verify payload
        call_args = mock_agent.handle_chat.call_args
        payload = call_args[0][0] # First arg
        
        assert payload["model"] == mock_context.args.model
        messages = payload["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == f"BACKGROUND TASK: {prompt}"
        
        # Verify background_tasks=None passed
        assert call_args[1]["background_tasks"] is None
