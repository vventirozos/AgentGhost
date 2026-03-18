import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from ghost_agent.tools.memory import tool_gain_knowledge
from ghost_agent.tools.file_system import tool_read_file

@pytest.mark.asyncio
async def test_knowledge_base_rejects_pdf_url_directly():
    """Verify that the knowledge base rejects direct ingestion of PDF URLs."""
    memory_system = MagicMock()
    memory_system.get_library.return_value = []
    
    # Passing a PDF URL
    filename = "https://example.com/document.pdf"
    sandbox_dir = Path("/tmp/fake_sandbox")
    
    result = await tool_gain_knowledge(filename, sandbox_dir, memory_system)
    
    assert "Error:" in result
    assert "use file_system(operation='download') first" in result

@pytest.mark.asyncio
async def test_knowledge_base_file_resolution_optimization(tmp_path):
    """Verify that file resolution properly searches and identifies files without bounded rglob."""
    memory_system = MagicMock()
    memory_system.get_library.return_value = []
    
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    
    # Create nested structure with a file
    nested_dir = sandbox_dir / "deep" / "nested"
    nested_dir.mkdir(parents=True)
    
    target_file = nested_dir / "The-Bitcoin-Paper.txt"
    target_file.write_text("fake logic content")
    
    # Test Priority 2: Stem match (passing 'bitcoin' should find 'The-Bitcoin-Paper.txt' through substring)
    result = await tool_gain_knowledge("bitcoin", sandbox_dir, memory_system)
    
    # If it reached chunking/embedding, it resolved the file
    assert "SUCCESS: Ingested" in result

@pytest.mark.asyncio
async def test_file_system_pdf_reading_guard():
    """Verify that read_file explicitly explains how to read or ingest PDFs."""
    sandbox_dir = Path("/tmp/fake_sandbox")
    
    result = await tool_read_file("document.pdf", sandbox_dir)
    
    assert "Error:" in result
    assert "is a PDF" in result
    assert "permanently index it into your vector memory" in result
    assert "read_chunked" in result
