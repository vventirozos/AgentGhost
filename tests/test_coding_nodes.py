import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.llm import LLMClient

@pytest.fixture
def mock_coding_nodes():
    return [
        {"url": "http://coding-node-1:8000", "model": "qwen2.5-coder:7b"},
        {"url": "http://coding-node-2:8000", "model": "deepseek-coder:6.7b"}
    ]

@pytest.mark.asyncio
async def test_llm_client_initializes_coding_nodes(mock_coding_nodes):
    client = LLMClient(upstream_url="http://main-node:8000", coding_nodes=mock_coding_nodes)
    assert len(client.coding_clients) == 2
    assert client.coding_clients[0]["model"] == "qwen2.5-coder:7b"
    assert client.coding_clients[1]["model"] == "deepseek-coder:6.7b"
    assert hasattr(client, "_coding_index")
    await client.close()

@pytest.mark.asyncio
async def test_get_coding_node_round_robin(mock_coding_nodes):
    client = LLMClient(upstream_url="http://main-node:8000", coding_nodes=mock_coding_nodes)
    
    node1 = client.get_coding_node()
    node2 = client.get_coding_node()
    node3 = client.get_coding_node()
    
    assert node1["model"] == "qwen2.5-coder:7b"
    assert node2["model"] == "deepseek-coder:6.7b"
    assert node3["model"] == "qwen2.5-coder:7b"  # Wraps around
    await client.close()

@pytest.mark.asyncio
async def test_get_coding_node_by_model(mock_coding_nodes):
    client = LLMClient(upstream_url="http://main-node:8000", coding_nodes=mock_coding_nodes)
    
    # Matching exact model
    node = client.get_coding_node("deepseek-coder:6.7b")
    assert node["model"] == "deepseek-coder:6.7b"
    
    # Matching partial model string
    node = client.get_coding_node("qwen")
    assert node["model"] == "qwen2.5-coder:7b"
    
    await client.close()

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_chat_completion_uses_coding_node(mock_post, mock_coding_nodes):
    client = LLMClient(upstream_url="http://main-node:8000", coding_nodes=mock_coding_nodes)
    
    # Mock the coding node response
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "Coding node response"}}]}
    
    # Inject mock into the specifically created httpx clients for coding nodes
    client.coding_clients[0]["client"].post = AsyncMock(return_value=mock_response)
    
    payload = {"messages": [{"role": "user", "content": "Write unit tests"}], "model": "any"}
    response = await client.chat_completion(payload, use_coding=True)
    
    assert response["choices"][0]["message"]["content"] == "Coding node response"
    client.coding_clients[0]["client"].post.assert_called_once()
    
    # Ensure standard post wasn't called
    assert not mock_post.called
    await client.close()

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_chat_completion_coding_node_fallback(mock_post, mock_coding_nodes):
    client = LLMClient(upstream_url="http://main-node:8000", coding_nodes=mock_coding_nodes)
    
    # Make all coding nodes fail
    for coding_client in client.coding_clients:
        coding_client["client"].post = AsyncMock(side_effect=Exception("Node Offline"))
        
    # Mock the main node fallback response
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "Main node fallback"}}]}
    
    # Have the main HTTP client return success
    client.http_client.post = AsyncMock(return_value=mock_response)
    
    payload = {"messages": [{"role": "user", "content": "Write unit tests"}], "model": "any"}
    response = await client.chat_completion(payload, use_coding=True)
    
    # We should have fallen back to main upstream node
    assert response["choices"][0]["message"]["content"] == "Main node fallback"
    
    # Ensure coding nodes were actually attempted
    assert client.coding_clients[0]["client"].post.called
    assert client.coding_clients[1]["client"].post.called
    
    assert client.http_client.post.called
    await client.close()

@pytest.mark.asyncio
async def test_stream_chat_completion_uses_coding_node(mock_coding_nodes):
    client = LLMClient(upstream_url="http://main-node:8000", coding_nodes=mock_coding_nodes)
    
    # Mock the coding node response for streaming
    async def mock_aiter_lines():
        yield b'data: {"choices": [{"delta": {"content": "Streamed "}}]}'
        yield b'data: {"choices": [{"delta": {"content": "code"}}]}'
        yield b'data: [DONE]'
        
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.aclose = AsyncMock()
    
    client.coding_clients[0]["client"].build_request = MagicMock()
    client.coding_clients[0]["client"].send = AsyncMock(return_value=mock_response)
    
    payload = {"messages": [{"role": "user", "content": "Write unit tests"}], "model": "any"}
    
    chunks = []
    async for chunk in client.stream_chat_completion(payload, use_coding=True):
        chunks.append(chunk.decode('utf-8'))
        
    assert len(chunks) == 3
    assert 'content": "Streamed "' in chunks[0]
    
    client.coding_clients[0]["client"].send.assert_called_once()
    await client.close()
