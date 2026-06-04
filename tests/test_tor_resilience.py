import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from src.ghost_agent.utils.helpers import request_new_tor_identity, helper_fetch_url_content
from src.ghost_agent.tools.search import tool_search_ddgs, tool_deep_research

def test_request_new_tor_identity_success():
    with patch("socket.socket") as mock_socket:
        mock_instance = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_instance
        # Simulate successful auth and newnym
        mock_instance.recv.side_effect = [b"250 OK\r\n", b"250 OK\r\n"]
        
        success, msg = request_new_tor_identity(password="testpass")
        
        assert success is True
        assert "renewed successfully" in msg
        mock_instance.sendall.assert_any_call(b'AUTHENTICATE "testpass"\r\n')
        mock_instance.sendall.assert_any_call(b'SIGNAL NEWNYM\r\n')

def test_request_new_tor_identity_auth_fail():
    with patch("socket.socket") as mock_socket:
        mock_instance = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_instance
        mock_instance.recv.return_value = b"515 Authentication failed\r\n"
        
        success, msg = request_new_tor_identity()
        
        assert success is False
        assert "Tor Auth failed" in msg

@pytest.mark.asyncio
async def test_helper_fetch_url_content_403_no_tor_renewal():
    """Post-audit: 401/403 are application-level forbidden errors.

    Rotating the Tor exit node won't help (the origin is rejecting the
    request itself), and burning identities on every 401/403 just slows
    down the next legitimate request. Helper now returns immediately on
    401/403 with a clear error and DOES NOT call request_new_tor_identity.
    Only 503 (gateway/exit-node level) triggers Tor rotation.
    """
    mock_curl = MagicMock()
    mock_requests = MagicMock()
    mock_curl.requests = mock_requests

    with patch.dict("sys.modules", {"curl_cffi": mock_curl, "curl_cffi.requests": mock_requests}), \
         patch("src.ghost_agent.utils.helpers.request_new_tor_identity") as mock_renew, \
         patch("asyncio.sleep", new_callable=AsyncMock), \
         patch("os.getenv") as mock_getenv:

        mock_getenv.return_value = "socks5://127.0.0.1:9050"

        mock_session_instance = AsyncMock()
        mock_requests.AsyncSession.return_value.__aenter__.return_value = mock_session_instance

        resp_403 = MagicMock()
        resp_403.status_code = 403
        resp_403.text = "Forbidden"
        mock_session_instance.get.return_value = resp_403

        result = await helper_fetch_url_content("http://example.com")

        # Returns the application-level forbidden error, no Tor rotation,
        # no second GET attempt.
        assert "Access Denied (403)" in result
        assert "Application-level forbidden" in result
        assert mock_renew.call_count == 0
        assert mock_session_instance.get.call_count == 1


@pytest.mark.asyncio
async def test_helper_fetch_url_content_retry_503_renews_tor():
    """503 is a network/gateway-level signal — the Tor exit node may be
    being rate-limited or filtered. Rotation IS appropriate here.
    """
    mock_curl = MagicMock()
    mock_requests = MagicMock()
    mock_curl.requests = mock_requests

    with patch.dict("sys.modules", {"curl_cffi": mock_curl, "curl_cffi.requests": mock_requests}), \
         patch("src.ghost_agent.utils.helpers.request_new_tor_identity") as mock_renew, \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
         patch("os.getenv") as mock_getenv:

        mock_getenv.return_value = "socks5://127.0.0.1:9050"

        mock_session_instance = AsyncMock()
        mock_requests.AsyncSession.return_value.__aenter__.return_value = mock_session_instance

        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_503.text = "Service Unavailable"
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.text = "<html><body>Some content</body></html>"
        mock_session_instance.get.side_effect = [resp_503, resp_200]

        result = await helper_fetch_url_content("http://example.com")

        assert "Some content" in result
        assert mock_renew.call_count == 1
        mock_sleep.assert_called_with(5)

@pytest.mark.asyncio
async def test_tool_search_ddgs_retry():
    """Tor-resilience redesign: the search path no longer calls a global
    NEWNYM between attempts. A global re-circuit is slow; instead each
    attempt rotates onto its OWN Tor circuit via _proxy_for_attempt, so a
    retry escapes a bad exit node without the NEWNYM cost. On a transient
    failure we just pause briefly (1s) and try the next circuit — NO
    request_new_tor_identity call.
    """
    mock_ddgs_module = MagicMock()
    mock_ddgs_class = MagicMock()
    mock_ddgs_module.DDGS = mock_ddgs_class

    with patch.dict("sys.modules", {"ddgs": mock_ddgs_module}), \
         patch("importlib.util.find_spec", return_value=True), \
         patch("src.ghost_agent.utils.helpers.request_new_tor_identity") as mock_renew, \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        # Fail once, then succeed on the second attempt (a fresh circuit).
        mock_ddgs_instance = MagicMock()
        mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs_instance

        mock_ddgs_instance.text.side_effect = [Exception("Tor blocked")] + [[{"title": "t", "body": "b", "href": "h"}]]

        result = await tool_search_ddgs("test query", "socks5://127.0.0.1:9050")

        assert "1. t" in result
        # NEWNYM thrash removed — rotation no longer happens on search retry.
        assert mock_renew.call_count == 0
        # Backoff between the two attempts is a short 1s pause, not 5s.
        mock_sleep.assert_called_with(1)

@pytest.mark.asyncio
async def test_tool_deep_research_retry():
    """Same NEWNYM-free retry contract as web_search (see above)."""
    mock_ddgs_module = MagicMock()
    mock_ddgs_class = MagicMock()
    mock_ddgs_module.DDGS = mock_ddgs_class

    with patch.dict("sys.modules", {"ddgs": mock_ddgs_module}), \
         patch("importlib.util.find_spec", return_value=True), \
         patch("src.ghost_agent.utils.helpers.request_new_tor_identity") as mock_renew, \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        mock_ddgs_instance = MagicMock()
        mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs_instance

        # Fail once, then valid results on the second attempt (fresh circuit).
        mock_ddgs_instance.text.side_effect = [Exception("Tor blocked deep")] + [[{"title": "t1", "body": "b1", "href": "http://example.com/good"}]]

        # mock semaphore and requests for deep research parsing
        with patch("src.ghost_agent.tools.search.helper_fetch_url_content", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = "Mocked content of site"
            result = await tool_deep_research("test query deep", anonymous=True, tor_proxy="socks5://127.0.0.1:9050")

            assert "http://example.com/good" in result
            assert mock_renew.call_count == 0
            mock_sleep.assert_called_with(1)


from src.ghost_agent.tools.file_system import tool_download_file
from src.ghost_agent.tools.system import tool_get_weather, tool_check_health

@pytest.mark.asyncio
async def test_tool_download_file_retry():
    mock_requests = MagicMock()
    
    with patch("src.ghost_agent.tools.file_system.curl_requests", mock_requests), \
         patch("src.ghost_agent.tools.file_system.request_new_tor_identity") as mock_renew, \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
         
        mock_session_instance = AsyncMock()
        mock_requests.AsyncSession.return_value.__aenter__.return_value = mock_session_instance
        
        # Responses: 503, Exception, then 200
        resp_503 = MagicMock()
        resp_503.status_code = 503
        
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.headers = {}
        
        async def dummy_stream():
            yield b"data chunk"
        resp_200.aiter_content.return_value = dummy_stream()
        
        mock_session_instance.get.side_effect = [resp_503, Exception("Timeout"), resp_200]
        
        sandbox_mock = MagicMock()
        from unittest.mock import mock_open
        with patch("src.ghost_agent.tools.file_system._get_safe_path", return_value=MagicMock()), \
             patch("builtins.open", mock_open()):
            result = await tool_download_file("http://example.com/file.txt", sandbox_mock, "socks5://127.0.0.1:9050", "file.txt")
        
        assert "SUCCESS" in result
        assert mock_renew.call_count == 2
        assert mock_sleep.call_count == 2

@pytest.mark.asyncio
async def test_tool_get_weather_retry():
    mock_requests = MagicMock()
    
    with patch("src.ghost_agent.tools.system.curl_requests", mock_requests), \
         patch("src.ghost_agent.tools.system.request_new_tor_identity") as mock_renew, \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
         
        mock_session = AsyncMock()
        mock_requests.AsyncSession.return_value.__aenter__.return_value = mock_session
        
        resp_403 = MagicMock()
        resp_403.status_code = 403
        
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"results": [{"latitude": 0, "longitude": 0, "name": "TestCity"}], "current": {"temperature_2m": 20}}
        
        mock_session.get.side_effect = [resp_403, resp_200, resp_200] # geo search 403, geo search 200, forecast 200
        
        res = await tool_get_weather("socks5://127.0.0.1:9050", location="TestCity")
        assert "REPORT (Source: Open-Meteo)" in res
        assert mock_renew.call_count == 1
        assert mock_sleep.call_count == 1

@pytest.mark.asyncio
async def test_tool_check_health_retry():
    mock_requests = MagicMock()
    
    with patch("src.ghost_agent.tools.system.curl_requests", mock_requests), \
         patch("src.ghost_agent.tools.system.request_new_tor_identity") as mock_renew, \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
         
        mock_session = AsyncMock()
        mock_requests.AsyncSession.return_value.__aenter__.return_value = mock_session
        
        resp_fail = MagicMock()
        resp_fail.status_code = 503
        
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        
        resp_tor_ok = MagicMock()
        resp_tor_ok.status_code = 200
        resp_tor_ok.json.return_value = {"IsTor": True}
        
        # 1.1.1.1 call: 503, 200
        # torproject call: 503, 200_tor
        mock_session.get.side_effect = [resp_fail, resp_ok, resp_fail, resp_tor_ok]
        
        ctx = MagicMock()
        ctx.tor_proxy = "socks5://127.0.0.1:9050"
        ctx.llm_client = None
        ctx.memory_system = None
        ctx.sandbox_dir = None
        ctx.scheduler = None

        res = await tool_check_health(context=ctx)
        
        assert "Internet: Connected (200)" in res
        assert "Tor: Connected (Anonymous)" in res
        assert mock_renew.call_count == 2
        assert mock_sleep.call_count == 2
