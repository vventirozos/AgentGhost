"""Tests for the configurable chat-proxy timeout in interface/server.py.

A long agent turn (deep browser automation, many sequential tool calls)
can run for minutes without flushing any SSE bytes. The old flat
`timeout=600.0` made httpx's *read* timeout fire at 600s, abort the
stream, and the UI render a bare "No response". These tests pin:

- `_chat_timeout()` builds an httpx.Timeout with a long read/write/pool
  window and a short, separate connect timeout;
- the default ceiling is well above the old 600s;
- both the streaming and non-streaming chat paths hand that Timeout to
  httpx (NOT a flat 600.0);
- the ceiling is overridable via the GHOST_CHAT_TIMEOUT env var.
"""

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import Request

# interface.server raises at import unless GHOST_API_KEY is set. Existing
# interface tests rely on the env providing it; set a default so this file
# is runnable standalone too (harmless when the real key is already set).
os.environ.setdefault("GHOST_API_KEY", "test-ghost-key")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import interface.server as server  # noqa: E402
from interface.server import chat_proxy  # noqa: E402


def test_chat_timeout_builds_split_httpx_timeout():
    """read/write/pool track CHAT_TIMEOUT_S; connect is the short override."""
    t = server._chat_timeout()
    assert isinstance(t, httpx.Timeout)
    assert t.read == server.CHAT_TIMEOUT_S
    assert t.write == server.CHAT_TIMEOUT_S
    assert t.pool == server.CHAT_TIMEOUT_S
    assert t.connect == server.CHAT_CONNECT_TIMEOUT_S
    # Connect must stay short so a dead backend fails fast.
    assert server.CHAT_CONNECT_TIMEOUT_S <= 30


def test_default_ceiling_exceeds_old_600s():
    """Regression guard: the default must beat the old flat 600s read cap."""
    assert server.CHAT_TIMEOUT_S > 600


@pytest.mark.asyncio
async def test_streaming_chat_passes_chat_timeout():
    """The streaming path hands client.stream the long httpx.Timeout."""
    mock_request = MagicMock(spec=Request)
    mock_request.json = AsyncMock(return_value={"stream": True, "messages": []})

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    async def fake_aiter(*args, **kwargs):
        yield b"data: hi\n\n"

    mock_response.aiter_bytes = MagicMock(side_effect=fake_aiter)

    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_response
    mock_context.__aexit__.return_value = None

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context

    with patch.object(server, "_get_http_client", return_value=mock_client):
        response = await chat_proxy(mock_request)
        # Drain the StreamingResponse so the background worker runs and the
        # upstream stream() call is actually issued.
        if hasattr(response, "body_iterator"):
            async for _ in response.body_iterator:
                pass

    mock_client.stream.assert_called_once()
    _, kwargs = mock_client.stream.call_args
    timeout = kwargs["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == server.CHAT_TIMEOUT_S
    assert timeout.read != 600.0


@pytest.mark.asyncio
async def test_non_streaming_chat_passes_chat_timeout():
    """The non-streaming path hands client.post the long httpx.Timeout."""
    mock_request = MagicMock(spec=Request)
    mock_request.json = AsyncMock(return_value={"stream": False, "messages": []})

    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"response": "ok"})

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch.object(server, "_get_http_client", return_value=mock_client):
        await chat_proxy(mock_request)

    mock_client.post.assert_called_once()
    _, kwargs = mock_client.post.call_args
    timeout = kwargs["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == server.CHAT_TIMEOUT_S
    assert timeout.read != 600.0


def test_env_override_changes_ceiling():
    """GHOST_CHAT_TIMEOUT overrides the default ceiling on (re)import."""
    with patch.dict(os.environ, {"GHOST_CHAT_TIMEOUT": "4242"}):
        reloaded = importlib.reload(server)
        try:
            assert reloaded.CHAT_TIMEOUT_S == 4242.0
            assert reloaded._chat_timeout().read == 4242.0
        finally:
            # Restore the module to its env-default state for other tests.
            importlib.reload(reloaded)
