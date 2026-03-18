import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import Request, UploadFile
from interface.server import chat_proxy, upload_proxy, download_proxy, GHOST_API_KEY

@pytest.mark.asyncio
@patch("interface.server.httpx.AsyncClient")
async def test_chat_proxy_adds_auth_header_streaming(mock_client_class):
    mock_request = MagicMock(spec=Request)
    mock_request.json = AsyncMock(return_value={"stream": True})
    
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()
    mock_client_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    
    async def fake_aiter(*args, **kwargs):
        yield b"data1"
    mock_response.aiter_bytes = MagicMock(side_effect=fake_aiter)
    
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_response
    mock_context.__aexit__.return_value = None
    mock_client.stream.return_value = mock_context
    
    response = await chat_proxy(mock_request)
    
    # consume stream
    if hasattr(response, "body_iterator"):
        async for _ in response.body_iterator:
            pass
            
    # Check that stream was called with the correct headers
    mock_client.stream.assert_called_once()
    args, kwargs = mock_client.stream.call_args
    assert "headers" in kwargs
    assert kwargs["headers"] == {"X-Ghost-Key": GHOST_API_KEY}

@pytest.mark.asyncio
@patch("interface.server.httpx.AsyncClient")
async def test_chat_proxy_adds_auth_header_non_streaming(mock_client_class):
    mock_request = MagicMock(spec=Request)
    mock_request.json = AsyncMock(return_value={"stream": False})
    
    mock_client = MagicMock()
    
    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"response": "ok"})
    mock_client.post = AsyncMock(return_value=mock_response)
    
    # AsyncContextManager mock for AsyncClient
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None
    mock_client_class.return_value = mock_context
    
    await chat_proxy(mock_request)
    
    mock_client.post.assert_called_once()
    args, kwargs = mock_client.post.call_args
    assert "headers" in kwargs
    assert kwargs["headers"] == {"X-Ghost-Key": GHOST_API_KEY}

@pytest.mark.asyncio
@patch("interface.server.httpx.AsyncClient")
async def test_upload_proxy_adds_auth_header(mock_client_class):
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.txt"
    mock_file.file = b"content"
    mock_file.content_type = "text/plain"
    
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"response": "ok"})
    mock_client.post = AsyncMock(return_value=mock_response)
    
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None
    mock_client_class.return_value = mock_context
    
    await upload_proxy(mock_file)
    
    mock_client.post.assert_called_once()
    args, kwargs = mock_client.post.call_args
    assert "headers" in kwargs
    assert kwargs["headers"] == {"X-Ghost-Key": GHOST_API_KEY}

@pytest.mark.asyncio
@patch("interface.server.httpx.AsyncClient")
async def test_download_proxy_adds_auth_header(mock_client_class):
    mock_client_instances = [MagicMock(), MagicMock()]
    mock_client_class.side_effect = mock_client_instances
    
    # 1st instance (the stream client)
    mock_response_stream = AsyncMock()
    mock_response_stream.raise_for_status = MagicMock()
    async def fake_aiter(*args, **kwargs):
        yield b"data"
    mock_response_stream.aiter_bytes = MagicMock(side_effect=fake_aiter)
    
    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__.return_value = mock_response_stream
    mock_stream_ctx.__aexit__.return_value = None
    mock_client_instances[0].send.return_value = mock_response_stream
    mock_client_instances[0].aclose = AsyncMock()
    
@pytest.mark.asyncio
@patch("interface.server.httpx.AsyncClient")
async def test_download_proxy_adds_auth_header(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {"content-type": "image/jpeg", "content-disposition": "attachment"}
    
    async def fake_aiter(*args, **kwargs):
        yield b"data"
    mock_response.aiter_bytes = MagicMock(side_effect=fake_aiter)
    mock_response.aclose = AsyncMock()
    
    # Mock build_request and send
    mock_req = MagicMock()
    mock_client.build_request.return_value = mock_req
    mock_client.send = AsyncMock(return_value=mock_response)
    mock_client.aclose = AsyncMock()

    response = await download_proxy("test.jpg")
    
    if hasattr(response, "body_iterator"):
        async for _ in response.body_iterator:
            pass
            
    mock_client.build_request.assert_called_once()
    _, kwargs_build = mock_client.build_request.call_args
    assert "headers" in kwargs_build
    assert kwargs_build["headers"] == {"X-Ghost-Key": GHOST_API_KEY}
    
    mock_client.send.assert_called_once()
    args_send, kwargs_send = mock_client.send.call_args
    assert args_send[0] == mock_req
    assert "stream" in kwargs_send and kwargs_send["stream"] is True
