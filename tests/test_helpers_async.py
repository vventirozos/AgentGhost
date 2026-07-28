
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


# --------------------------------------------------------------------------
# curl_cffi AsyncSession path (2026-07-28 regression). AsyncSession responses
# stream through an asyncio.Queue: the SYNC iter_content() on them returns
# unawaited Queue.get() coroutines instead of bytes, so draining with it broke
# every live fetch ("can't extend bytearray with coroutine") while the
# list-backed mock kept tests green. These tests pin the async drain/close.
# --------------------------------------------------------------------------

def _curl_session_for(resp_factory):
    """A fake curl_cffi module whose AsyncSession .get() returns resp_factory()."""
    import types

    class _Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return resp_factory()

    return types.SimpleNamespace(requests=types.SimpleNamespace(AsyncSession=_Session))


@pytest.mark.asyncio
async def test_curl_async_path_never_uses_sync_iter_content():
    body = "<html><body><p>Async Drained</p></body></html>"
    resp = make_streaming_resp(200, body)
    resp.iter_content = MagicMock(
        side_effect=AssertionError("sync iter_content() on an AsyncSession response"))
    fake_curl = _curl_session_for(lambda: resp)
    with patch("ghost_agent.utils.helpers.url_ssrf_reason", lambda u, **k: None), \
         patch.dict("sys.modules", {"curl_cffi": fake_curl, "curl_cffi.requests": fake_curl.requests}):
        result = await helper_fetch_url_content("http://example.com", renew_identity=False)
    assert result == "Async Drained"
    resp.aiter_content.assert_called_once()
    assert resp.aclose.await_count >= 1


@pytest.mark.asyncio
async def test_curl_async_drain_stops_at_cap_and_aborts_transfer():
    # An "endless" body: the drain must break at the 5 MB cap after ~6 chunks
    # and abort the transfer (quit_now + aclose), not read all 32.
    consumed = {"n": 0}

    async def _endless():
        for _ in range(32):
            consumed["n"] += 1
            yield b"a" * (1024 * 1024)

    resp = make_streaming_resp(200, "")
    resp.aiter_content = MagicMock(side_effect=_endless)
    fake_curl = _curl_session_for(lambda: resp)
    with patch("ghost_agent.utils.helpers.url_ssrf_reason", lambda u, **k: None), \
         patch.dict("sys.modules", {"curl_cffi": fake_curl, "curl_cffi.requests": fake_curl.requests}):
        result = await helper_fetch_url_content("http://example.com", renew_identity=False)
    assert consumed["n"] <= 7
    assert "TRUNCATED at 5 MB ceiling" in result
    resp.quit_now.set.assert_called()
    assert resp.aclose.await_count >= 1


@pytest.mark.asyncio
async def test_curl_async_reject_branch_aborts_without_draining():
    # Header-only short-circuit (binary content-type): the body must never be
    # drained, and the streaming transfer must be aborted via the ASYNC close
    # (the sync close() only reaps stream_task, which AsyncSession responses
    # never set).
    resp = make_streaming_resp(200, "%PDF-1.7", content_type="application/pdf")
    # Record the abort ORDER: quit_now must be set BEFORE aclose, else aclose
    # awaits the full (potentially multi-GB) transfer we decided to drop.
    order = []
    resp.quit_now.set = MagicMock(side_effect=lambda: order.append("quit_now"))
    resp.aclose = AsyncMock(side_effect=lambda: order.append("aclose"))
    fake_curl = _curl_session_for(lambda: resp)
    with patch("ghost_agent.utils.helpers.url_ssrf_reason", lambda u, **k: None), \
         patch.dict("sys.modules", {"curl_cffi": fake_curl, "curl_cffi.requests": fake_curl.requests}):
        result = await helper_fetch_url_content("http://example.com", renew_identity=False)
    assert "binary file" in result
    resp.aiter_content.assert_not_called()
    assert order == ["quit_now", "aclose"]

