
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
    """Verify the unified-forget semantic sweep still hits collection.query
    and collection.delete on the vector store. The sweep is now bundled
    inside a single locked helper that runs under asyncio.to_thread, so we
    assert directly on the mock collection's call history rather than on
    the to_thread dispatch shape."""
    sandbox = Path("/tmp/sandbox")
    target = "forget me"

    with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
        async def passthrough(func, *args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        mock_to_thread.side_effect = passthrough

        # The semantic sweep checks `_get_lock` via hasattr; provide a
        # null context so the lock branch is exercised without crashing.
        mock_memory_system._get_lock = MagicMock()
        mock_memory_system._get_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_memory_system._get_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_memory_system.get_library.return_value = []

        await tool_unified_forget(target, sandbox, mock_memory_system)

    assert mock_memory_system.collection.query.called, (
        "Vector semantic sweep must call collection.query"
    )
    assert mock_memory_system.collection.delete.called, (
        "Vector semantic sweep must call collection.delete on the matched id"
    )
