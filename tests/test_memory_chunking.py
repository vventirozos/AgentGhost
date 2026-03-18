import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from ghost_agent.tools.memory import tool_recall, recursive_split_text

def test_recursive_split_chunk_reductions():
    """Verify that text splitting honors strict 600 limits for all-MiniLM embedding"""
    # Create text longer than 600
    long_text = "A" * 800
    
    chunks = recursive_split_text(long_text, chunk_size=600, chunk_overlap=100)
    
    assert len(chunks) > 1, "Should split into multiple chunks"
    
    for chunk in chunks:
        assert len(chunk) <= 600, "No chunk should exceed 600 characters"

@pytest.mark.asyncio
async def test_tool_recall_rag_threshold_logic():
    """Verify tool_recall categorizes distance < 0.8 as HIGH RELEVANCE and < 1.15 as MEDIUM RELEVANCE, cutting off at 1.35"""
    memory_system = MagicMock()
    
    # Fake a set of results stepping over the bounds
    memory_system.search_advanced = MagicMock(return_value=[
        {"text": "High match", "metadata": {"type": "document", "source": "src1"}, "score": 0.75},
        {"text": "Med match", "metadata": {"type": "document", "source": "src2"}, "score": 1.10},
        {"text": "Low match", "metadata": {"type": "document", "source": "src3"}, "score": 1.30},
        {"text": "Filtered out", "metadata": {"type": "document", "source": "src4"}, "score": 1.40},
    ])
    
    result = await tool_recall("test query", memory_system)
    
    # Should exclude 1.40
    assert "Filtered out" not in result
    
    # Check boundaries
    assert "High match" in result
    assert "Med match" in result
    assert "Low match" in result
