import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json
import httpx
from types import SimpleNamespace

# 1. Test Agent handling 400 Context Overflow
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.mark.asyncio
async def test_agent_context_overflow_recovery():
    # Setup
    args = SimpleNamespace(max_context=8000, temperature=0.5, perfect_it=False, smart_memory=0.0)
    context = GhostContext(args=args, sandbox_dir="/tmp", memory_dir="/tmp", tor_proxy=None)
    
    mock_llm_client = MagicMock()
    # First call raises context overflow, second call succeeds
    
    class MockResponse:
        status_code = 400
        text = "context length exceeded"
        
    mock_llm_client.chat_completion = AsyncMock(side_effect=[
        httpx.HTTPStatusError("Overflow", request=MagicMock(), response=MockResponse()),
        {"choices": [{"message": {"content": "recovered ok"}}]}
    ])
    context.llm_client = mock_llm_client
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    
    agent = GhostAgent(context)
    
    body = {
        "messages": [
            {"role": "system", "content": "You are AI"},
            {"role": "user", "content": "hello"}
        ]
    }
    
    # We expect handle_chat to intercept the HTTPStatusError, prune, and retry ONCE.
    # Instead of an infinite loop, it should return gracefully hitting the 2nd mock.
    
    # To prevent infinite looping in the test if the bug is still present, we add a timeout.
    try:
        await asyncio.wait_for(agent.handle_chat(body, background_tasks=None), timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail("handle_chat hit an infinite loop trying to recover from 400 Context Overflow")
        
    assert mock_llm_client.chat_completion.call_count == 2
    
    # Verify the second payload has pruned messages
    payload_args, payload_kwargs = mock_llm_client.chat_completion.call_args_list[1]
    second_payload_msgs = payload_args[0]["messages"]
    
    assert any("SYSTEM ALERT: The conversation history was truncated" in str(m) for m in second_payload_msgs)


# 2. Test PERFECT IT Context truncation
@patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[])
@pytest.mark.asyncio
async def test_agent_perfect_it_truncation(mock_get_tools):
    args = SimpleNamespace(max_context=8000, temperature=0.5, perfect_it=True, smart_memory=0.0)
    context = GhostContext(args=args, sandbox_dir="/tmp", memory_dir="/tmp", tor_proxy=None)
    mock_llm_client = MagicMock()
    mock_llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})
    context.llm_client = mock_llm_client
    context.scratchpad = MagicMock()
    agent = GhostAgent(context)

    body = {
        "messages": [
            {"role": "user", "content": "hello"}
        ]
    }
    
    # We inject monkey-patch for tools_run_this_turn temporarily
    original_handle_chat = agent.handle_chat
    
    async def wrapped_handle_chat(*args, **kwargs):
        # We simulate that a huge tool output happened
        with patch.object(agent, '_prune_context', new_callable=AsyncMock) as mock_prune:
            mock_prune.return_value = [{"role": "system", "content": "mocked system Last Tool Output: None"}]
            return await original_handle_chat(*args, **kwargs)
             
    await wrapped_handle_chat(body, background_tasks=None)
    
    # We check the final payload for Perfect It instruction limits
    payload_args, _ = mock_llm_client.chat_completion.call_args_list[0]
    sys_msg = payload_args[0]["messages"][0]["content"] if payload_args[0]["messages"] else ""
    # "Last Tool Output: " should exist but not be massive (since tools_run_this_turn was empty, it defaults to None)
    assert 'Last Tool Output: None' in sys_msg or "Last Tool Output: " in sys_msg

# 3. Test Deep Research Gather Error Isolation
from ghost_agent.tools.search import tool_deep_research

@pytest.mark.asyncio
@patch("ghost_agent.tools.search.helper_fetch_url_content")
@patch("ghost_agent.tools.search.importlib.util.find_spec")
async def test_deep_research_gather_resilience(mock_find, mock_fetch):
    mock_find.return_value = True
    
    class MockDDGS:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def text(self, *args, **kwargs):
            return [{"url": "http://good.com"}, {"url": "http://bad.com"}]
            
    with patch("ghost_agent.tools.search.DDGS", MockDDGS, create=True):
        # Mock helper_fetch to fail for one, succeed for another
        async def mock_fetch_side_effect(url, *args, **kwargs):
            if "bad.com" in url:
                raise ValueError("Simulated network failure")
            return "Good Content"
            
        mock_fetch.side_effect = mock_fetch_side_effect
        
        result = await tool_deep_research("test query", anonymous=False, tor_proxy=None)
        
        assert "Good Content" in result
        assert "Simulated network failure" not in result # Unhandled exceptions bubble up if not return_exceptions=True
        assert "DEEP RESEARCH RESULT" in result

# 4. Test Interface Server chunk_size
from interface.server import chat_proxy
from fastapi import Request

@pytest.mark.asyncio
@patch("interface.server.httpx.AsyncClient")
async def test_chat_proxy_no_chunk_size_streaming(mock_client_class):
    mock_request = MagicMock(spec=Request)
    mock_request.json = AsyncMock(return_value={"stream": True})
    
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()
    mock_client_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    
    # We want to inspect the kwargs given to aiter_bytes
    async def fake_aiter(*args, **kwargs):
        yield b"data1"
        yield b"data2"
    mock_response.aiter_bytes = MagicMock(side_effect=fake_aiter)
    
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_response
    mock_context.__aexit__.return_value = None
    mock_client.stream.return_value = mock_context
    
    response = await chat_proxy(mock_request)
    
    # It returns a StreamingResponse. To consume it:
    import builtins
    if hasattr(response, "body_iterator"):
        async for _ in response.body_iterator:
            pass
            
    # Check that aiter_bytes was called with chunk_size=None
    mock_response.aiter_bytes.assert_called_once_with(chunk_size=None)

# 5. Test Docker execute demux=False output property
from ghost_agent.sandbox.docker import DockerSandbox
from pathlib import Path

@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.os.path.exists")
def test_docker_demux_false(mock_exists):
    # Bypass init checks
    mock_exists.return_value = True
    
    sandbox = DockerSandbox(Path("/tmp"))
    sandbox.container = MagicMock()
    # Stub ensure_running and _is_container_ready
    sandbox.ensure_running = MagicMock()
    sandbox._is_container_ready = MagicMock(return_value=True)
    
    # Mock exec_run to return a single output (bytes), not a tuple, because demux=False
    mock_exec_result = MagicMock()
    mock_exec_result.output = b"hello combined stdout and stderr"
    mock_exec_result.exit_code = 0
    sandbox.container.exec_run.return_value = mock_exec_result
    
    out, code = sandbox.execute("echo test")
    
    assert code == 0
    assert out == "hello combined stdout and stderr"
    
    # Check demux=False was passed
    exec_args, exec_kwargs = sandbox.container.exec_run.call_args
    assert exec_kwargs.get("demux") is False
