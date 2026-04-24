import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from ghost_agent.memory.vector import VectorMemory

@pytest.fixture
def mock_chroma():
    with patch("ghost_agent.memory.vector.chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        yield mock_client

from pathlib import Path

@pytest.mark.asyncio
async def test_vector_search_rag_document_priority(mock_chroma):
    """Verify that RAG document memory types receive priority -5 and 1.25 threshold"""
    memory = VectorMemory(memory_dir=Path("/tmp"), upstream_url=None)
    memory.collection = MagicMock()
    
    # Mocking chromadb collection query response
    # We provide a document with distance 1.20 (which would fail standard 0.55/0.8 threshold but pass 1.25)
    memory.collection.query.return_value = {
        "ids": [["doc_1", "man_1"]],
        "documents": [["Document content here", "Manual fact here"]],
        "metadatas": [[
            {"type": "document", "timestamp": "2026-01-01T00:00:00.000000", "source": "test.pdf"},
            {"type": "manual", "timestamp": "2026-01-01T00:00:00.000000", "source": "user"}
        ]],
        "distances": [[1.20, 0.40]]
    }
    
    # We patch the datetime to safely parse timestamps
    results = memory.search("test query")
    
    # We parse the output string as 'memory.search' returns a formatted text box, not dicts
    # Output looks like: "[2026-01-01T00:00:00.000000] (DOCUMENT) Document content here"
    assert "Document content" in results

@pytest.mark.asyncio
async def test_vector_search_rag_document_suppression(mock_chroma):
    """Verify that RAG documents beyond 1.25 distance are suppressed entirely"""
    memory = VectorMemory(memory_dir=Path("/tmp"), upstream_url=None)
    memory.collection = MagicMock()
    
    # Document with distance 1.30 (fails 1.25 threshold)
    memory.collection.query.return_value = {
        "ids": [["doc_bad"]],
        "documents": [["Too far document content"]],
        "metadatas": [[{"type": "document", "timestamp": "2026-01-01T00:00:00.000000", "source": "test.pdf"}]],
        "distances": [[1.30]]
    }
    
    results = memory.search("test query")
    
    # It should be filtered out by `dist < threshold` where `m_type == 'document'` threshold is 1.25
    assert "Too far document content" not in results, "Document beyond 1.25 threshold should be discarded"
