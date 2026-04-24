
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock, mock_open
from pathlib import Path
from ghost_agent.tools.memory import tool_gain_knowledge
from ghost_agent.memory.vector import VectorMemory

@pytest.mark.asyncio
async def test_tool_gain_knowledge_async_extraction_pdf(tmp_path):
    # Setup
    sandbox_dir = tmp_path
    filename = "test.pdf"
    file_path = sandbox_dir / filename
    file_path.touch()
    
    mock_memory = MagicMock(spec=VectorMemory)
    
    # Mock fitz (PyMuPDF)
    with patch("fitz.open") as mock_fitz_open, \
         patch("ghost_agent.tools.memory.recursive_split_text", return_value=["chunk1"]), \
         patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        
        # Setup fitz mock
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "PDF Content"
        mock_doc.__iter__.return_value = [mock_page]
        mock_fitz_open.return_value = mock_doc
        
        # Configure to_thread to execute the callable if it's the extraction function
        # We can't easily check identity of inner function, but we can return data
        
        # Side effect to run the function passed to to_thread?
        # The code will call: await asyncio.to_thread(_extract_text)
        # And later: await asyncio.to_thread(memory_system.ingest_document, ...)
        
        # Let's mock to_thread to return "PDF Content" for the first call (extraction)
        # and None for the second call (ingestion)
        # But wait, if we mock it, the inner function won't run unless we run it.
        
        async def side_effect(func, *args, **kwargs):
            if callable(func):
                return func(*args, **kwargs)
            return None
            
        mock_to_thread.side_effect = side_effect

        # Run tool
        await tool_gain_knowledge(filename, sandbox_dir, mock_memory)
        
        # Verify fitz was used (inside the thread)
        mock_fitz_open.assert_called_with(file_path)
        
        # Verify to_thread was called at least twice (extraction + ingestion)
        assert mock_to_thread.call_count >= 2

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
        
        # Verify open was used
        mock_file_open.assert_called_with(file_path, "r", encoding="utf-8", errors="ignore")
        
        # Verify to_thread was called
        assert mock_to_thread.call_count >= 2
