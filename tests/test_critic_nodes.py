"""Dedicated CRITIC node pool on LLMClient.

The critic pool (`--critic-nodes`) serves the self-evaluation verifier
from a separate, typically slower off-host model (e.g. a spare Mac Mini
running a small judge model). It is kept distinct from the worker pool so
a slow verdict never queues ahead of the fast routing/validation chores
the worker pool serves, and off the foreground slot so it never competes
with the main model for the KV-cache. These tests mirror
`test_coding_nodes.py`: init, round-robin, by-model selection, routed
chat_completion, and graceful fallback to the main upstream.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.llm import LLMClient


@pytest.fixture
def mock_critic_nodes():
    return [
        {"url": "http://critic-node-1:8001", "model": "qwen3:9b"},
        {"url": "http://critic-node-2:8001", "model": "gemma2:9b"},
    ]


@pytest.mark.asyncio
async def test_llm_client_initializes_critic_nodes(mock_critic_nodes):
    client = LLMClient(upstream_url="http://main-node:8000", critic_nodes=mock_critic_nodes)
    assert len(client.critic_clients) == 2
    assert client.critic_clients[0]["model"] == "qwen3:9b"
    assert client.critic_clients[1]["model"] == "gemma2:9b"
    assert hasattr(client, "_critic_index")
    await client.close()


@pytest.mark.asyncio
async def test_critic_pool_absent_by_default():
    client = LLMClient(upstream_url="http://main-node:8000")
    assert client.critic_clients == []
    assert client.get_critic_node() is None
    await client.close()


@pytest.mark.asyncio
async def test_get_critic_node_round_robin(mock_critic_nodes):
    client = LLMClient(upstream_url="http://main-node:8000", critic_nodes=mock_critic_nodes)

    node1 = client.get_critic_node()
    node2 = client.get_critic_node()
    node3 = client.get_critic_node()

    assert node1["model"] == "qwen3:9b"
    assert node2["model"] == "gemma2:9b"
    assert node3["model"] == "qwen3:9b"  # wraps around
    await client.close()


@pytest.mark.asyncio
async def test_get_critic_node_by_model(mock_critic_nodes):
    client = LLMClient(upstream_url="http://main-node:8000", critic_nodes=mock_critic_nodes)

    node = client.get_critic_node("gemma2:9b")
    assert node["model"] == "gemma2:9b"

    node = client.get_critic_node("qwen")  # partial match
    assert node["model"] == "qwen3:9b"

    await client.close()


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_chat_completion_uses_critic_node(mock_post, mock_critic_nodes):
    client = LLMClient(upstream_url="http://main-node:8000", critic_nodes=mock_critic_nodes)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "Critic verdict"}}]}

    client.critic_clients[0]["client"].post = AsyncMock(return_value=mock_response)

    payload = {"messages": [{"role": "user", "content": "verify this"}], "model": "any"}
    response = await client.chat_completion(payload, use_critic=True)

    assert response["choices"][0]["message"]["content"] == "Critic verdict"
    client.critic_clients[0]["client"].post.assert_called_once()
    # The dedicated pool was used, not the main upstream.
    assert not mock_post.called
    await client.close()


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_chat_completion_critic_node_fallback(mock_post, mock_critic_nodes):
    client = LLMClient(upstream_url="http://main-node:8000", critic_nodes=mock_critic_nodes)

    # All critic nodes offline → fall back to the main upstream.
    for critic_client in client.critic_clients:
        critic_client["client"].post = AsyncMock(side_effect=Exception("Node Offline"))

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "Main node fallback"}}]}
    client.http_client.post = AsyncMock(return_value=mock_response)

    payload = {"messages": [{"role": "user", "content": "verify this"}], "model": "any"}
    response = await client.chat_completion(payload, use_critic=True)

    assert response["choices"][0]["message"]["content"] == "Main node fallback"
    assert client.critic_clients[0]["client"].post.called
    assert client.critic_clients[1]["client"].post.called
    assert client.http_client.post.called
    await client.close()


@pytest.mark.asyncio
async def test_critic_pool_isolated_from_worker_pool(mock_critic_nodes):
    """A critic node must NOT leak into the worker pool (the whole point of
    the separation): use_worker with no worker nodes still falls through to
    main upstream, never to the critic clients."""
    client = LLMClient(upstream_url="http://main-node:8000", critic_nodes=mock_critic_nodes)
    assert client.worker_clients == []
    assert len(client.critic_clients) == 2
    await client.close()
