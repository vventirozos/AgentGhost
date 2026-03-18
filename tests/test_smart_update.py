
import pytest
from unittest.mock import MagicMock, patch
from ghost_agent.memory.vector import VectorMemory

@pytest.fixture
def mock_vector_memory():
    # Mock dependencies
    memory_dir = MagicMock()
    memory_dir.exists.return_value = True
    upstream_url = "http://localhost:11434"
    
    # Patch __init__
    with pytest.MonkeyPatch.context() as m:
        m.setattr("ghost_agent.memory.vector.VectorMemory.__init__", lambda self, a, b, c=None: None)
        vm = VectorMemory(memory_dir, upstream_url)
        vm.collection = MagicMock()
        vm.add = MagicMock() # Mock add so we can check if it's called
        return vm

def test_smart_update_deduplication_threshold(mock_vector_memory):
    # Scenario: We are adding a memory that is a paraphrase of an existing one.
    # The distance is 0.15.
    # Current Threshold: 0.05 -> Should NOT delete existing (Duplication)
    # Target Threshold: 0.20 -> Should delete existing (Deduplication)
    
    existing_id = "existing_id_123"
    new_text = "The sky is blue today."
    
    # Mock query result
    mock_vector_memory.collection.query.return_value = {
        'ids': [[existing_id]],
        'distances': [[0.15]], # Distance is 0.15
        'documents': [['The sky is azure today.']],
        'metadatas': [[{'timestamp': 'old'}]]
    }
    
    # Run smart_update
    mock_vector_memory.smart_update(new_text)
    
    # Check if delete was called
    # With strict threshold (0.05), this should be FALSE.
    # With relaxed threshold (0.20), this should be TRUE.
    
    # We assert that it *should* match the NEW behavior we want (0.20).
    # So initially this test might fail if the code is still 0.05? 
    # Or strict TDD: write test expecting TRUE, see it fail.
    
    mock_vector_memory.collection.delete.assert_called_with(ids=[existing_id])
    
    # Ensure add is always called (to add the new version)
    mock_vector_memory.add.assert_called()

