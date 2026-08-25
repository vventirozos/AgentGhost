import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import Request, UploadFile
import interface.server as server
from interface.server import chat_proxy, upload_proxy, download_proxy
# ⚠ THE KEY IS READ FROM THE MODULE AT ASSERT TIME, never bound by value at
# import. `interface.server` computes `GHOST_API_KEY` from the environment
# at import, and `test_interface_chat_timeout.py` calls
# `importlib.reload(server)` — so a module-level `from ... import
# GHOST_API_KEY` here freezes whatever the env held when THIS file was
# imported, while the proxies under test read the module's CURRENT value.
# When a neighbour left `GHOST_API_KEY=test-key` in the environment, the two
# diverged and six interface tests failed under xdist:
#
#   assert {'X-Ghost-Key': 'test-key'} == {'X-Ghost-Key': '0dc28f40...'}
#
# The leak is fixed at its source (the Slack suites now restore the env),
# but reading it dynamically is what makes this file's assertion state what
# it means: the proxy forwards THE SERVER'S key, whatever it is.


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
    assert kwargs["headers"] == {"X-Ghost-Key": server.GHOST_API_KEY}

@pytest.mark.asyncio
async def test_chat_proxy_adds_auth_header_non_streaming():
    # The non-streaming path now reuses the shared pooled client returned
    # by `_get_http_client()` instead of constructing `httpx.AsyncClient()`
    # per request, so we patch the helper directly.
    import interface.server as server_mod

    mock_request = MagicMock(spec=Request)
    mock_request.json = AsyncMock(return_value={"stream": False})

    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"response": "ok"})

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch.object(server_mod, "_get_http_client", return_value=mock_client):
        await chat_proxy(mock_request)

    mock_client.post.assert_called_once()
    args, kwargs = mock_client.post.call_args
    assert "headers" in kwargs
    assert kwargs["headers"] == {"X-Ghost-Key": server.GHOST_API_KEY}

@pytest.mark.asyncio
async def test_upload_proxy_adds_auth_header():
    # The proxy now reads the upload body via _read_capped_upload, which
    # iterates `await file.read(65536)` and requires real bytes/bytearray
    # chunks. The fixture returns the body on the first read, then empty
    # bytes (EOF) on the second. Upload now rides the shared pooled client
    # (_get_http_client), so patch the helper instead of the class.
    import interface.server as server_mod

    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.txt"
    mock_file.content_type = "text/plain"
    mock_file.read = AsyncMock(side_effect=[b"content", b""])

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"response": "ok"})
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch.object(server_mod, "_get_http_client", return_value=mock_client):
        await upload_proxy(mock_file)

    mock_client.post.assert_called_once()
    args, kwargs = mock_client.post.call_args
    assert "headers" in kwargs
    assert kwargs["headers"] == {"X-Ghost-Key": server.GHOST_API_KEY}

@pytest.mark.asyncio
async def test_download_proxy_adds_auth_header():
    # Download also rides the shared pooled client now.
    import interface.server as server_mod

    mock_client = MagicMock()

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

    with patch.object(server_mod, "_get_http_client", return_value=mock_client):
        response = await download_proxy("test.jpg")

        if hasattr(response, "body_iterator"):
            async for _ in response.body_iterator:
                pass

    mock_client.build_request.assert_called_once()
    _, kwargs_build = mock_client.build_request.call_args
    assert "headers" in kwargs_build
    assert kwargs_build["headers"] == {"X-Ghost-Key": server.GHOST_API_KEY}

    mock_client.send.assert_called_once()
    args_send, kwargs_send = mock_client.send.call_args
    assert args_send[0] == mock_req
    assert "stream" in kwargs_send and kwargs_send["stream"] is True


@pytest.mark.asyncio
async def test_the_proxy_forwards_the_SERVERS_current_key(monkeypatch):
    """⚠ PINS THE DYNAMIC READ. Every assertion above compares against
    `server.GHOST_API_KEY`; written as a by-value `from interface.server
    import GHOST_API_KEY` they compare against whatever the environment
    held when THIS file was imported, and pass anyway in a clean process —
    which is why the defect lived here unseen until xdist co-located a
    neighbour that changed the env and a test that reloads the server.

    Rotating the server's key makes the two readings differ WITHOUT a
    reload, so this pin needs no `importlib.reload` and cannot contaminate
    the session the way the original failure did. Reverting the four
    assertions to the by-value import kills it.
    """
    import interface.server as server_mod

    monkeypatch.setattr(server_mod, "GHOST_API_KEY", "rotated-key-xyz")

    # Same fixture shape as `test_upload_proxy_adds_auth_header` above:
    # `_read_capped_upload` iterates `await file.read(65536)` and needs a
    # real EOF, or it trips the pathological-producer cap (413).
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.txt"
    mock_file.content_type = "text/plain"
    mock_file.read = AsyncMock(side_effect=[b"content", b""])

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"response": "ok"})
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch.object(server_mod, "_get_http_client", return_value=mock_client):
        await upload_proxy(mock_file)

    _, kwargs = mock_client.post.call_args
    assert kwargs["headers"] == {"X-Ghost-Key": "rotated-key-xyz"}, (
        "the proxy did not forward the server's CURRENT key")
    assert kwargs["headers"] == {"X-Ghost-Key": server_mod.GHOST_API_KEY}


def test_this_file_does_not_FREEZE_the_key_at_import():
    """⚠ THE PIN ABOVE DOES NOT CATCH THE BINDING, and this one does.

    Mutation-checked: reverting all four assertions to a by-value
    `from interface.server import GHOST_API_KEY` leaves the behavioural
    pin GREEN in a clean process — it compares the forwarded header
    against a literal and against `server_mod.GHOST_API_KEY`, both of
    which follow the rotation. It can only fail once a NEIGHBOUR has
    already diverged the two, which is precisely the condition that made
    the original defect invisible for so long.

    The defect is a property of this module's namespace, so check the
    namespace: a frozen copy shows up as a module-level global here.
    """
    import sys
    mod = sys.modules[__name__]
    assert not hasattr(mod, "GHOST_API_KEY"), (
        "this module binds GHOST_API_KEY by value at import; it will "
        "silently disagree with `interface.server` after any reload or "
        "environment change. Read `server.GHOST_API_KEY` at assert time.")
