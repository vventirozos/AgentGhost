import pytest
from unittest.mock import AsyncMock, patch
from ghost_agent.tools.memory import tool_self_play


@pytest.mark.asyncio
@patch("ghost_agent.core.dream.Dreamer")
async def test_tool_self_play_bypasses_background_lock(mock_dreamer_class):
    mock_dreamer_instance = mock_dreamer_class.return_value
    mock_dreamer_instance.synthetic_self_play = AsyncMock(return_value="Success report")

    context = "mock_context"
    result = await tool_self_play(context)

    # Ensure it was called with is_background=False to bypass the lock
    mock_dreamer_instance.synthetic_self_play.assert_called_once_with(is_background=False)
    assert "Success report" in result
    assert "SYSTEM: SELF PLAY DONE." in result
