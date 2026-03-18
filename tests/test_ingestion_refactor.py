
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.memory.vector import VectorMemory
from ghost_agent.tools.memory import tool_gain_knowledge

@pytest.fixture
def mock_vector_memory():
    memory_dir = MagicMock()
    memory_dir.exists.return_value = True
    upstream_url = "http://localhost:11434"
    
    with pytest.MonkeyPatch.context() as m:
        m.setattr("ghost_agent.memory.vector.VectorMemory.__init__", lambda self, a, b, c=None: None)
        vm = VectorMemory(memory_dir, upstream_url)
        vm.collection = MagicMock()
        vm.library_file = MagicMock()
        # Mock ingest_document to return success
        vm.ingest_document = MagicMock(return_value=(True, "Success"))
        return vm

def test_ingest_document_enrichment(mock_vector_memory):
    # Test that chunks are enriched with source filename
    filename = "test_doc.txt"
    chunks = ["chunk1", "chunk2"]
    
    # Restore the real ingest_document method for this test, but mock collection
    with pytest.MonkeyPatch.context() as m:
        # We need to bind the real method to the mock object
        # Using a trick: define a simple wrapper or just import the class and use the unbound method
        from ghost_agent.memory.vector import VectorMemory as RealVectorMemory
        
        # Bind the real method to our mock instance
        mock_vector_memory.ingest_document = RealVectorMemory.ingest_document.__get__(mock_vector_memory, RealVectorMemory)
        
        # Initialize mock_vector_memory.collection.upsert
        mock_vector_memory.collection.upsert = MagicMock()
        mock_vector_memory._update_library_index = MagicMock()
        mock_vector_memory.get_library = MagicMock(return_value=[]) 
        
        # Run ingest
        mock_vector_memory.ingest_document(filename, chunks)
        
        # Check upsert call
        # upsert(documents=..., metadatas=..., ids=...)
        call_args = mock_vector_memory.collection.upsert.call_args
        assert call_args, "Upsert was not called"
        
        documents = call_args.kwargs['documents']
        
        # Verify enrichment
        assert documents[0] == f"[Source: {filename}]\nchunk1"
        assert documents[1] == f"[Source: {filename}]\nchunk2"

@pytest.mark.asyncio
async def test_tool_gain_knowledge_calls_ingest_async(mock_vector_memory):
    # Test that tool_gain_knowledge uses asyncio.to_thread(memory_system.ingest_document, ...)
    
    # Mock inputs
    filename = "test_doc.txt"
    sandbox_dir = MagicMock()
    sandbox_dir.rglob.return_value = []
    
    # Mock file existence and content
    # We need to mock how tool_gain_knowledge reads the file.
    # It constructs a path: sandbox_dir / filename
    mock_file = MagicMock()
    mock_file.exists.return_value = True
    mock_file.read_text.return_value = "Content" # Not used directly if we mock open
    
    # Mock fitz (PyMuPDF) if needed, or just standard open
    # The tool checks for .pdf extension. Let's use .txt
    
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file_handle = MagicMock()
        mock_file_handle.read.return_value = "File Content that will be chunked"
        mock_open.return_value.__enter__.return_value = mock_file_handle
        
        # Mock path join
        sandbox_dir.__truediv__.return_value = mock_file
        
        # Mock recursive_split_text
        with patch("ghost_agent.tools.memory.recursive_split_text", return_value=["chunk1", "chunk2"]):
            
            # Mock asyncio.to_thread
            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                # to_thread is called twice:
                # 1. _extract_text -> returns str
                # 2. ingest_document -> returns (True, "Success")
                
                async def side_effect(func, *args, **kwargs):
                    if func.__name__ == "_extract_text":
                        return "Mocked File Content"
                    elif func == mock_vector_memory.ingest_document:
                         return (True, "Success")
                    return None
                    
                mock_to_thread.side_effect = side_effect
                
                await tool_gain_knowledge(filename, sandbox_dir, mock_vector_memory)
                
                # Check if to_thread was called with ingest_document
                found_call = False
                for call in mock_to_thread.call_args_list:
                    if call.args and call.args[0] == mock_vector_memory.ingest_document:
                        assert call.args[1] == filename
                        assert call.args[2] == ["chunk1", "chunk2"]
                        found_call = True
                        break
                
                assert found_call, "Did not call ingest_document via to_thread"
