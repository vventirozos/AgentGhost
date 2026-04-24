import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from ghost_agent.tools.memory import tool_self_play


@pytest.mark.asyncio
@patch("ghost_agent.core.dream.Dreamer")
async def test_tool_self_play_bypasses_background_lock(mock_dreamer_class):
    mock_dreamer_instance = mock_dreamer_class.return_value
    mock_dreamer_instance.synthetic_self_play = AsyncMock(return_value="Success report")

    # The tool now requires an explicit user-intent string (see
    # `_user_asked_for_self_play` in tools/memory.py). Seed it so this
    # test exercises the happy path, not the refusal branch.
    context = SimpleNamespace(last_user_content="run self-play")
    result = await tool_self_play(context)

    # Ensure it was called with is_background=False to bypass the lock
    mock_dreamer_instance.synthetic_self_play.assert_called_once_with(is_background=False)
    assert "Success report" in result
    assert "SYSTEM: SELF PLAY DONE." in result
