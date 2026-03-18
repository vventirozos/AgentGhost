
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.dream import Dreamer

@pytest.fixture
def mock_context():
    context = MagicMock()
    context.memory_system = MagicMock()
    context.memory_system.collection = MagicMock()
    context.llm_client = MagicMock()
    context.skill_memory = MagicMock()
    return context

@pytest.mark.asyncio
async def test_dream_async_db_calls(mock_context):
    # Setup
    dreamer = Dreamer(mock_context)
    
    # Mock DB get return
    mock_context.memory_system.collection.get.return_value = {
        "ids": ["1", "2", "3"],
        "documents": ["doc1", "doc2", "doc3"]
    }
    
    # Mock LLM response with valid JSON in Markdown
    llm_response_content = """
    ```json
    {
        "consolidations": [
            {
                "synthesis": "Synthesis 1",
                "merged_ids": ["ID:1", "ID:2"]
            }
        ],
        "heuristics": []
    }
    ```
    """
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": llm_response_content}}]
    })
    
    # Mock asyncio.to_thread
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # We need to simulate the return values of the threaded calls
        # 1. collection.get -> returns dict
        # 2. memory.add -> returns None
        # 3. collection.delete -> returns None
        
        async def side_effect(func, *args, **kwargs):
            if func == mock_context.memory_system.collection.get:
                return {
                    "ids": ["1", "2", "3", "4"],
                    "documents": ["doc1", "doc2", "doc3", "doc4"]
                }
            return None
            
        mock_to_thread.side_effect = side_effect
        
        await dreamer.dream()
        
        # Verify to_thread usage
        
        # Check get() was wrapped
        # args[0] is the function
        calls = mock_to_thread.call_args_list
        
        # We expect at least:
        # 1. collection.get
        # 2. memory.add (if synthesis happened)
        # 3. collection.delete (if synthesis happened)
        
        func_calls = [call.args[0] for call in calls]
        
        assert mock_context.memory_system.collection.get in func_calls
        assert mock_context.memory_system.add in func_calls
        assert mock_context.memory_system.collection.delete in func_calls

@pytest.mark.asyncio
async def test_dream_robust_json_parsing(mock_context):
    dreamer = Dreamer(mock_context)
    
    # Mock DB get to return enough docs
    mock_context.memory_system.collection.get.return_value = {
        "ids": ["1", "2", "3", "4"],
        "documents": ["d1", "d2", "d3", "d4"]
    }
    
    # Mock LLM response with Markdown code blocks (which json.loads would fail on)
    llm_response_content = "Here is the JSON:\n```json\n{\"consolidations\": []}\n```"
    
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": llm_response_content}}]
    })
    
    # We don't care about async db calls here, just that it doesn't crash on JSON
    # But since we haven't implemented async yet, the code calls collection.get synchronously.
    # That's fine for this test if we Mock correctly.
    
    result = await dreamer.dream()
    
    # If it failed to parse, it would return "Dream failed: ..."
    assert "Dream failed" not in result
    assert "Dream Complete" in result or "Dream cycle complete" in result
