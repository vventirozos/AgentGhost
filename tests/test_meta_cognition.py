import pytest
from unittest.mock import MagicMock, AsyncMock
from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.dream import Dreamer

@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.memory_system = MagicMock()
    ctx.profile_memory = MagicMock()
    # Mock args for agent init
    ctx.args = MagicMock()
    return ctx

@pytest.mark.asyncio
async def test_smart_memory_thresholds(mock_context):
    """
    Test that a high score (0.95) acts as a trigger for profile memory update.
    """
    agent = GhostAgent(mock_context)

    # Mock LLM to return high score and profile update
    mock_context.llm_client.chat_completion.return_value = {
        "choices": [{"message": {"content": '{"score": 0.95, "fact": "Remember that I love Python", "profile_update": {"category": "coding", "key": "lang", "value": "Python"}}'}}]
    }

    await agent.run_smart_memory_task("Remember that I love Python", "test-model", 0.5)

    # Assert profile memory was updated
    # §4QM: the extractor hands the ledger the owner's own clause behind the value
    _args, _kw = mock_context.profile_memory.update.call_args
    assert _args == ("coding", "lang", "Python")
    assert _kw["source"]["via"] == "extractor" and "python" in _kw["source"]["said"].lower()
    # Assert memory system was also updated (add)
    mock_context.memory_system.add.assert_called()

@pytest.mark.asyncio
async def test_smart_memory_discard(mock_context):
    """
    Test that a low score (0.1) prevents profile memory update.
    """
    agent = GhostAgent(mock_context)

    # Mock LLM returns low score
    mock_context.llm_client.chat_completion.return_value = {
        "choices": [{"message": {"content": '{"score": 0.1, "fact": "User said hi"}'}}]
    }

    await agent.run_smart_memory_task("User said hi", "test-model", 0.5)

    # Assert profile memory was NOT updated
    mock_context.profile_memory.update.assert_not_called()
    # Assert memory system add was NOT called (impl details: if score < selectivity)
    mock_context.memory_system.add.assert_not_called()

@pytest.mark.asyncio
async def test_dream_is_housekeeping_only(mock_context):
    """§4NM: the dream writes no lesson and asks the model for no REM
    synthesis — it is the housekeeping cycle, stamped "skipped" on its
    outcome surface (never "ran")."""
    dreamer = Dreamer(mock_context)
    mock_context.project_store = None
    mock_context.trajectory_collector = None
    mock_context.graph_memory = None
    mock_context.memory_dir = None
    mock_context._last_store_maintenance = None

    result = await dreamer.dream("test-model")

    mock_context.llm_client.chat_completion.assert_not_called()
    assert "Housekeeping cycle done" in result
    assert dreamer.last_dream_outcome["phase"] == "skipped"
