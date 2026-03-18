
import pytest
from unittest.mock import MagicMock
from ghost_agent.memory.vector import VectorMemory

@pytest.fixture
def mock_vector_memory():
    # Mock dependencies
    memory_dir = MagicMock()
    memory_dir.exists.return_value = True
    upstream_url = "http://localhost:11434"
    
    # Patch the __init__ to avoid ChromaDB setup
    with pytest.MonkeyPatch.context() as m:
        m.setattr("ghost_agent.memory.vector.VectorMemory.__init__", lambda self, a, b, c=None: None)
        vm = VectorMemory(memory_dir, upstream_url)
        vm.collection = MagicMock()
        vm.library_file = MagicMock()
        return vm

def test_search_sorting_order(mock_vector_memory):
    # Setup mock return values for collection.query
    # We want two documents with same p_score but different timestamps
    
    # Note: search logic calculates p_score based on metadata type and match distance.
    # We need to craft metadata and distances to yield the same p_score.
    # Manual type results in p_score 0.
    
    mock_vector_memory.collection.query.return_value = {
        'ids': [['id1', 'id2']],
        'documents': [['Old Memory', 'New Memory']],
        'metadatas': [[
            {'type': 'manual', 'timestamp': '2023-01-01 12:00:00'},
            {'type': 'manual', 'timestamp': '2024-01-01 12:00:00'}
        ]],
        'distances': [[0.1, 0.1]] # Low distance to pass threshold
    }
    
    # Mock logging used in search
    mock_vector_memory.logger = MagicMock()
    
    # Helper to clean up formatting for easy assertion
    def clean_output(output):
        return [line.split('] ')[1] for line in output.split('\n---\n') if '] ' in line]

    # Run search
    result = mock_vector_memory.search("query", inject_identity=False)
    
    # We expect the NEWER memory (2024) to appear BEFORE the OLDER memory (2023)
    # logic: sort by timestamp DESCENDING.
    
    # Parse result to check order
    # Result format: "[timestamp] (TYPE) **[PREFIX]** content"
    
    assert "New Memory" in result
    assert "Old Memory" in result
    
    lines = result.split('\n---\n')
    first_doc = lines[0]
    
    # Verify New Memory is first
    assert "New Memory" in first_doc, f"Expected New Memory first, got: {first_doc}"
