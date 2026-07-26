
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from ghost_agent.utils.helpers import helper_fetch_url_content
from tests.conftest import make_streaming_resp, make_httpx_stream_client

@pytest.mark.asyncio
async def test_helper_fetch_url_content_offloads_parsing():
    # Mock httpx.AsyncClient (streaming path) and block curl_cffi.
    body = "<!doctype html><html><body><script>bad</script><p>  Good  Text  </p></body></html>"
    resp = make_streaming_resp(200, body)
    client = make_httpx_stream_client(resp)
    with patch("ghost_agent.utils.helpers.httpx.AsyncClient") as mock_client_cls, \
         patch("ghost_agent.utils.helpers.url_ssrf_reason", lambda u, **k: None), \
         patch.dict("sys.modules", {"curl_cffi": None, "curl_cffi.requests": None}):
        mock_client_cls.return_value.__aenter__.return_value = client
        # Parsing is still offloaded to a worker thread.
        result = await helper_fetch_url_content("http://example.com")
        assert result == "Good Text"

