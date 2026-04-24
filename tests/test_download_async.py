
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock, mock_open
from pathlib import Path
from ghost_agent.tools.file_system import tool_download_file

@pytest.mark.asyncio
async def test_tool_download_file_writes_asynchronously():
    # Setup mocks
    mock_url = "http://example.com/file.txt"
    mock_sandbox = Path("/tmp/sandbox")
    mock_content = b"chunk1chunk2"
    
    # Mock httpx client
    with patch("ghost_agent.tools.file_system.curl_requests", None), \
         patch("ghost_agent.tools.file_system.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        
        # Mock stream response
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Length": "100"}
        
        # Mock aiter_bytes to yield chunks
        async def mock_aiter_bytes():
            yield b"chunk1"
            yield b"chunk2"
        mock_resp.aiter_bytes = mock_aiter_bytes
        
        # ensure stream is NOT an AsyncMock (which is awaited on call), but returns an async context manager
        mock_context = MagicMock()
        mock_context.__aenter__.return_value = mock_resp
        mock_context.__aexit__.return_value = None
        
        mock_client.stream = MagicMock(return_value=mock_context)

        # Mock open()
        m_open = mock_open()
        
        # Mock asyncio.to_thread
        # We need to spy on it to ensure it's called with f.write
        with patch("ghost_agent.tools.file_system.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            with patch("builtins.open", m_open):
                # Also need to mock os.path.basename and urlparse or ensure safe path works
                # Since we pass mock sandbox, _get_safe_path logic might fail if not careful.
                # However, for unit test, we can mock _get_safe_path too if needed, 
                # or just use a concrete path structure if the file system allows checking.
                # But to avoid FS access, mocking _get_safe_path is safer.
                with patch("ghost_agent.tools.file_system._get_safe_path") as mock_get_safe_path:
                    mock_target = MagicMock()
                    mock_target.parent.mkdir = MagicMock()
                    mock_get_safe_path.return_value = mock_target
                    
                    # Execute
                    await tool_download_file(mock_url, mock_sandbox, None, "file.txt")
                    
                    # Verify
                    mock_file_handle = m_open()
                    
                    # Verify to_thread was called for each chunk with write
                    assert mock_to_thread.call_count == 1
                    
                    # Inspect calls to ensure the first argument was the write method
                    # (mock_file_handle.write) and second was the chunk
                    calls = mock_to_thread.call_args_list
                    assert calls[0][0][0] == mock_file_handle.write
                    assert calls[0][0][1] == bytearray(b"chunk1chunk2")
