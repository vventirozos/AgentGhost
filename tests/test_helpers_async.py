
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from ghost_agent.utils.helpers import helper_fetch_url_content

@pytest.mark.asyncio
async def test_helper_fetch_url_content_offloads_parsing():
    # Mock httpx.AsyncClient and block curl_cffi so it falls back to httpx
    with patch("ghost_agent.utils.helpers.httpx.AsyncClient") as mock_client_cls, \
         patch.dict("sys.modules", {"curl_cffi": None, "curl_cffi.requests": None}):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        
        # Mock response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # A simple HTML that needs parsing
        mock_resp.text = "<!doctype html><html><body><script>bad</script><p>  Good  Text  </p></body></html>"
        mock_client.get.return_value = mock_resp
        
        # We need to spy on asyncio.to_thread
        # Since asyncio.to_thread is a function, we patch it
        with patch("ghost_agent.utils.helpers.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            # We want the mock to actually run the function if possible, or at least simulate the return
            # But for this test, we just want to verify it WAS called with the parsing logic.
            # So we'll have it return the expected parsed string.
            mock_to_thread.return_value = "Good Text"
            
            result = await helper_fetch_url_content("http://example.com")
            
            # Verify result
            assert result == "Good Text"
            
            # Verify asyncio.to_thread was called
            assert mock_to_thread.called
            
            # Optionally verify what it was called with
            # The first arg should be the parser function, second arg the html text
            args, _ = mock_to_thread.call_args
            assert callable(args[0])
            assert args[1].strip() == mock_resp.text.strip()

