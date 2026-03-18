
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
from ghost_agent.tools.memory import tool_unified_forget

@pytest.fixture
def mock_memory_system():
    mem_sys = MagicMock()
    mem_sys.collection = MagicMock()
    # Mock query result structure
    mem_sys.collection.query.return_value = {
        'ids': [['mem_1']],
        'distances': [[0.1]],
        'documents': [['Target content']],
        'metadatas': [[{'type': 'auto'}]]
    }
    return mem_sys

@pytest.mark.asyncio
async def test_unified_forget_async_calls(mock_memory_system):
    """Verify to_thread usage in tool_unified_forget"""
    
    sandbox = Path("/tmp/sandbox")
    target = "forget me"
    
    with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
        # Step 1: Mock the return values for the sequence of to_thread calls
        # We have:
        # 1. memory_system.get_library (fuzzy sweep) -> return list
        # 2. memory_system.collection.query (semantic sweep) -> return candidates
        # 3. memory_system.collection.delete (if candidate found)
        
        # Setup side effects for to_thread based on the callable passed
        async def side_effect(func, *args, **kwargs):
            if func == mock_memory_system.get_library:
                return []
            if func == mock_memory_system.collection.query:
                return {
                    'ids': [['mem_1']],
                    'distances': [[0.1]],
                    'documents': [['Target content']],
                    'metadatas': [[{'type': 'auto'}]]
                }
            if func == mock_memory_system.collection.delete:
                return None
            if func == mock_memory_system.delete_document_by_name:
                return None
            return None
            
        mock_to_thread.side_effect = side_effect

        await tool_unified_forget(target, sandbox, mock_memory_system)
        
        # Verify calls
        # Check query call
        query_called = False
        delete_called = False

        
        for call in mock_to_thread.call_args_list:
            if call.args[0] == mock_memory_system.collection.query:
                query_called = True
            if call.args[0] == mock_memory_system.collection.delete:
                delete_called = True
                
        assert query_called, "collection.query was not offloaded to thread"
        assert delete_called, "collection.delete was not offloaded to thread"
