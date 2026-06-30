import pytest
import os
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from ghost_agent.tools.memory import tool_unified_forget, tool_knowledge_base

@pytest.fixture
def mock_memory_system():
    mem_sys = MagicMock()
    mem_sys.collection = MagicMock()
    mem_sys.collection.query.return_value = {
        'ids': [['mem1', 'mem2']],
        'distances': [[0.1, 0.9]],
        'documents': [['Target document', 'Irrelevant document']],
        'metadatas': [[{'type': 'auto'}, {'type': 'auto'}]]
    }
    mem_sys.get_library.return_value = ["test_file.txt", "target_doc.pdf"]
    return mem_sys

@pytest.fixture
def mock_profile_memory():
    prof = MagicMock()
    # Profile sweep now matches on KEYS, not values (the value-match path
    # was destructive: target='python' would wipe any unrelated entry whose
    # value mentioned python). The fixture has 'target_color' as the key
    # so the test still exercises the profile-deletion path.
    prof.load.return_value = {
        "preferences": {"music": "jazz", "target_color": "red"}
    }
    return prof

@pytest.fixture
def mock_graph_memory():
    graph = MagicMock()
    # Mocking the delete_by_target method to return the count of deleted edges
    graph.delete_by_target.return_value = 3
    return graph

@pytest.mark.asyncio
async def test_tool_unified_forget_integration(tmp_path, mock_memory_system, mock_profile_memory, mock_graph_memory):
    """Test that all Memory subsystems (Vector, Profile, Disk, Graph) are targeted securely by unified_forget."""
    
    # 1. Setup mock disk
    (tmp_path / "target_file.txt").write_text("dummy")
    
    # Needs a patch for asyncio.to_thread to execute synchronously for AsyncMock testing
    with patch("ghost_agent.tools.memory.asyncio.to_thread") as mock_to_thread:
        # Simple passthrough mocker for asyncio.to_thread
        async def passthrough(func, *args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        mock_to_thread.side_effect = passthrough

        report = await tool_unified_forget(
            target="target", 
            sandbox_dir=tmp_path, 
            memory_system=mock_memory_system, 
            profile_memory=mock_profile_memory, 
            graph_memory=mock_graph_memory
        )
        
        # Verify 1: Disk swept
        assert "Disk: Deleted" in report
        assert not (tmp_path / "target_file.txt").exists()
        
        # Verify 2: Vector swept
        assert mock_memory_system.delete_document_by_name.call_count == 1
        assert mock_memory_system.collection.delete.call_count == 1 # Found semantic chunk
        # The fixture doc literally contains "target", so it now trips the
        # literal-mention override (more accurate than the old distance-only
        # "derived" label). Accept either phrasing.
        assert "Sweep: Forgot" in report
        
        # Verify 3: Profile swept on key match (not value match)
        mock_profile_memory.delete.assert_called_once_with("preferences", "target_color")
        assert "Profile: Removed preferences.target_color" in report
        
        # Verify 4: Graph Memory swept
        mock_graph_memory.delete_by_target.assert_called_once_with("target")
        assert "Severed 3 topological edges" in report

@pytest.mark.asyncio
async def test_reset_all_triggers_wipe_all(mock_memory_system, mock_graph_memory):
    """Test that tool_knowledge_base('reset_all') triggers Graph wipe."""
    
    mock_memory_system.collection.get.return_value = {'ids': ['1', '2']}
    
    with patch("ghost_agent.tools.memory.asyncio.to_thread") as mock_to_thread:
        async def passthrough(func, *args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        mock_to_thread.side_effect = passthrough
        
        result = await tool_knowledge_base(
            action="reset_all",
            sandbox_dir=Path("/tmp"),
            memory_system=mock_memory_system,
            graph_memory=mock_graph_memory
        )
        
        assert "Wiped clean" in result
        mock_memory_system.collection.delete.assert_called_with(ids=['1', '2'])
        mock_graph_memory.wipe_all.assert_called_once()
