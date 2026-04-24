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
    ctx.skill_memory = MagicMock()
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
    mock_context.profile_memory.update.assert_called_with("coding", "lang", "Python")
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
async def test_dream_heuristics(mock_context):
    """
    Test that Dreamer extracts heuristics when enough memories exist.
    """
    dreamer = Dreamer(mock_context)
    
    # Mock database returning 5 memories
    mock_context.memory_system.collection.get.return_value = {
        "ids": ["1", "2", "3", "4", "5"],
        "documents": ["mem1", "mem2", "mem3", "mem4", "mem5"],
        "metadatas": [{}, {}, {}, {}, {}],
        "embeddings": []
    }
    
    # Mock LLM returning a heuristic
    mock_context.llm_client.chat_completion.return_value = {
        "choices": [{"message": {"content": '{"consolidations": [], "heuristics": ["Always use absolute paths in Docker"]}'}}]
    }
    
    result = await dreamer.dream("test-model")
    
    # Assert learn_lesson was called
    mock_context.skill_memory.learn_lesson.assert_called()
    call_args = mock_context.skill_memory.learn_lesson.call_args
    assert "Always use absolute paths in Docker" in str(call_args)
    assert "Dream Complete" in result

@pytest.mark.asyncio
async def test_dream_low_entropy_guard(mock_context):
    """
    Test that Dreamer aborts if there are not enough memories (< 3).
    """
    dreamer = Dreamer(mock_context)
    
    # Mock database returning only 2 memories
    mock_context.memory_system.collection.get.return_value = {
        "ids": ["1", "2"],
        "documents": ["mem1", "mem2"],
        "metadatas": [{}, {}],
        "embeddings": []
    }
    
    result = await dreamer.dream("test-model")
    
    # Assert LLM was NOT called
    mock_context.llm_client.chat_completion.assert_not_called()
    assert "Not enough entropy" in result
