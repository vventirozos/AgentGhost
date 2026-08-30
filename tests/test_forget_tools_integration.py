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
    # ⚠ `target.pdf` ADDED 2026-08-28. The library was
    # ["test_file.txt", "target_doc.pdf"] and forget("target") deleted
    # `target_doc.pdf` on a SUBSTRING match — the same over-deletion the disk
    # sweep stopped doing, and the reason `forget('pdf')` destroyed the
    # PostgreSQL manual. Documents now go on an exact name or stem match and
    # are otherwise reported. `target.pdf` is that exact-stem match, so the
    # "vector swept" assertion below still checks what it was written to
    # check; `target_doc.pdf` stays to prove the partial match is kept.
    mem_sys.get_library.return_value = [
        "test_file.txt", "target.pdf", "target_doc.pdf"]
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
    #
    # ⚠ FIXTURE CHANGED 2026-08-28, deliberately. This was
    # `target_file.txt`, which forget('target') reached through the
    # SUBSTRING tier — and that tier no longer deletes: measured,
    # forget('atlas') unlinked `atlas_migration_plan.py`,
    # `notes_about_atlas.md` and `sub/deep_atlas_notes.txt`, none of which
    # the caller named. Substring hits are reported now, not removed.
    #
    # This test's subject is "all four stores are swept", not "partial name
    # matches are deleted", so the file is named to match exactly and the
    # assertion below still checks what it was written to check. The new
    # contract has its own coverage in
    # tests/test_forget_disk_sweep_scope.py.
    (tmp_path / "target.txt").write_text("dummy")
    
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
        assert not (tmp_path / "target.txt").exists()
        
        # Verify 2: Vector swept
        assert mock_memory_system.delete_document_by_name.call_count == 1
        assert mock_memory_system.delete_document_by_name.call_args[0][0] == "target.pdf"
        assert "Vector: kept 1 ingested document(s)" in report
        assert "target_doc.pdf" in report
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
