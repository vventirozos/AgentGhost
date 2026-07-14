
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock, mock_open
from pathlib import Path
from ghost_agent.tools.memory import tool_gain_knowledge
from ghost_agent.memory.vector import VectorMemory

@pytest.mark.asyncio
async def test_tool_gain_knowledge_pdf_uses_streaming_ingest(tmp_path):
    """PDFs take the STREAMING path (2026-07-13), not the old whole-document
    extract → split → ingest. The old path materialised the full text, the
    full chunk list AND an enriched copy in RAM, refused >1000 pages and
    silently truncated at 5M chars — none of which survives a real reference
    manual (PostgreSQL: ~3k pages, ~10M chars)."""
    sandbox_dir = tmp_path
    filename = "test.pdf"
    (sandbox_dir / filename).touch()

    mock_memory = MagicMock(spec=VectorMemory)
    mock_memory.get_library.return_value = []

    from ghost_agent.memory.pdf_ingest import IngestStats
    stats = IngestStats(pages=3, chars=1200, chunks=2, sections=1)

    with patch("ghost_agent.memory.pdf_ingest.ingest_pdf_streaming",
               return_value=stats) as mock_stream, \
         patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:

        async def side_effect(func, *args, **kwargs):
            if callable(func):
                return func(*args, **kwargs)
            return None
        mock_to_thread.side_effect = side_effect

        result = await tool_gain_knowledge(filename, sandbox_dir, mock_memory)

    # The streaming ingester ran, and it owns the store writes (the tool must
    # NOT also call ingest_document itself — that would double-embed).
    mock_stream.assert_called_once()
    assert mock_stream.call_args.args[1] == filename
    mock_memory.ingest_document.assert_not_called()
    # Reports real structure, and points the model at the doc-QA action.
    assert "3 pages" in result and "2 chunks" in result
    assert "action='query'" in result

@pytest.mark.asyncio
async def test_tool_gain_knowledge_async_extraction_text(tmp_path):
    # Setup
    sandbox_dir = tmp_path
    filename = "test.txt"
    file_path = sandbox_dir / filename
    file_path.write_text("Text Content")
    
    mock_memory = MagicMock(spec=VectorMemory)
    
    with patch("builtins.open", mock_open(read_data="Text Content")) as mock_file_open, \
         patch("ghost_agent.tools.memory.recursive_split_text", return_value=["chunk1"]), \
         patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
         
        async def side_effect(func, *args, **kwargs):
            if callable(func):
                return func(*args, **kwargs)
            return None
        mock_to_thread.side_effect = side_effect

        # Run tool
        await tool_gain_knowledge(filename, sandbox_dir, mock_memory)
        
        # Verify open was used (utf-8-sig strips a BOM; errors="replace" keeps
        # non-UTF-8 mangling visible rather than silently dropped).
        mock_file_open.assert_called_with(file_path, "r", encoding="utf-8-sig", errors="replace")
        
        # Verify to_thread was called
        assert mock_to_thread.call_count >= 2
