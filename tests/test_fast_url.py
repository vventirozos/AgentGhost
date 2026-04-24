import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import httpx

from src.ghost_agent.core.llm import LLMClient
from src.ghost_agent.main import parse_args
from src.ghost_agent.utils.logging import Icons

@pytest.mark.asyncio
async def test_llm_client_swarm_success():
    client = LLMClient(
        upstream_url="http://ghost:8088", 
        swarm_nodes=[{"url": "http://node1:8088", "model": "qwen"}]
    )
    
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "swarm reply"}}]}
    mock_response.raise_for_status = MagicMock()
    mock_swarm_post = AsyncMock(return_value=mock_response)
    
    # We have one swarm client
    client.swarm_clients[0]["client"].post = mock_swarm_post
    
    mock_main_post = AsyncMock()
    client.http_client.post = mock_main_post
    
    with patch("src.ghost_agent.core.llm.pretty_log") as mock_log:
        resp = await client.chat_completion({"model": "qwen", "prompt": "hello"}, use_swarm=True)
        
        assert resp == {"choices": [{"message": {"content": "swarm reply"}}]}
        mock_swarm_post.assert_called_once()
        mock_main_post.assert_not_called()
        
    await client.close()

@pytest.mark.asyncio
async def test_llm_client_swarm_fallback():
    # Test that it falls back to the main client if swarm nodes fail
    client = LLMClient(
        upstream_url="http://ghost:8088", 
        swarm_nodes=[
            {"url": "http://node1:8088", "model": "qwen"},
            {"url": "http://node2:8088", "model": "llama"}
        ]
    )
    
    mock_swarm_post_1 = AsyncMock(side_effect=httpx.ConnectError("Connection Error"))
    mock_swarm_post_2 = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
    
    client.swarm_clients[0]["client"].post = mock_swarm_post_1
    client.swarm_clients[1]["client"].post = mock_swarm_post_2
    
    mock_main_response = MagicMock()
    mock_main_response.json.return_value = {"choices": [{"message": {"content": "main reply"}}]}
    mock_main_response.raise_for_status = MagicMock()
    mock_main_post = AsyncMock(return_value=mock_main_response)
    client.http_client.post = mock_main_post
    
    with patch("src.ghost_agent.core.llm.pretty_log") as mock_log:
        resp = await client.chat_completion({"model": "qwen", "prompt": "hello"}, use_swarm=True)
        assert resp == {"choices": [{"message": {"content": "main reply"}}]}
        
        # It should try both swarm nodes before falling back to main
        assert mock_swarm_post_1.call_count == 1
        assert mock_swarm_post_2.call_count == 1
        mock_main_post.assert_called_once()    
        
    await client.close()

@pytest.mark.asyncio
async def test_llm_client_swarm_partial_match():
    # Test that get_swarm_node returns correct node based on target_model
    client = LLMClient(
        upstream_url="http://ghost:8088", 
        swarm_nodes=[
            {"url": "http://node1:8088", "model": "llama-3-8b"},
            {"url": "http://node2:8088", "model": "qwen-coder-32b"}
        ]
    )
    
    node = client.get_swarm_node("qwen")
    assert node["model"] == "qwen-coder-32b"
    assert node["url"] == "http://node2:8088"
    
    node = client.get_swarm_node("llama")
    assert node["model"] == "llama-3-8b"
    assert node["url"] == "http://node1:8088"
    
    # If not found, it should do round robin
    node_a = client.get_swarm_node("unknown")
    node_b = client.get_swarm_node("unknown")
    
    assert node_a["url"] == "http://node1:8088"
    assert node_b["url"] == "http://node2:8088"
    
    await client.close()

@pytest.mark.asyncio
async def test_llm_client_swarm_empty_fallback():
    # Test that it falls back to the main client seamlessly if no swarm nodes are provided
    client = LLMClient(
        upstream_url="http://ghost:8088", 
        swarm_nodes=[]
    )
    
    mock_main_response = MagicMock()
    mock_main_response.json.return_value = {"choices": [{"message": {"content": "main reply"}}]}
    mock_main_response.raise_for_status = MagicMock()
    mock_main_post = AsyncMock(return_value=mock_main_response)
    client.http_client.post = mock_main_post
    
    with patch("src.ghost_agent.core.llm.pretty_log") as mock_log:
        resp = await client.chat_completion({"model": "qwen", "prompt": "hello"}, use_swarm=True)
        assert resp == {"choices": [{"message": {"content": "main reply"}}]}
        
        mock_main_post.assert_called_once()    
        
    await client.close()

def test_parse_args_swarm_nodes():
    test_args = ["main.py", "--swarm-nodes", "http://node1:8088|qwen,http://node2:8080|llama,http://node3:8088"]
    with patch.object(sys, 'argv', test_args):
        args = parse_args()
        assert len(args.swarm_nodes_parsed) == 3
        assert args.swarm_nodes_parsed[0]["url"] == "http://node1:8088"
        assert args.swarm_nodes_parsed[0]["model"] == "qwen"
        assert args.swarm_nodes_parsed[1]["url"] == "http://node2:8080"
        assert args.swarm_nodes_parsed[1]["model"] == "llama"
        assert args.swarm_nodes_parsed[2]["url"] == "http://node3:8088"
        assert args.swarm_nodes_parsed[2]["model"] == "default"

def test_parse_args_swarm_nodes_empty():
    test_args = ["main.py"]
    with patch.object(sys, 'argv', test_args):
        args = parse_args()
        assert args.swarm_nodes_parsed == []
