import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from ghost_agent.core.llm import LLMClient

@pytest.mark.asyncio
async def test_llm_client_image_gen_initialization():
    image_nodes = [{"url": "http://node1:8000", "model": "lcm"}]
    client = LLMClient("http://upstream:8080", image_gen_nodes=image_nodes)
    
    assert len(client.image_gen_clients) == 1
    assert client.image_gen_clients[0]["model"] == "lcm"
    assert client.image_gen_clients[0]["url"] == "http://node1:8000"
    
    await client.close()

@pytest.mark.asyncio
async def test_llm_client_get_image_gen_node():
    image_nodes = [
        {"url": "http://node1", "model": "lcm-a"},
        {"url": "http://node2", "model": "lcm-b"}
    ]
    client = LLMClient("http://upstream:8080", image_gen_nodes=image_nodes)
    
    # Test specific model routing
    node_b = client.get_image_gen_node("lcm-b")
    assert node_b["model"] == "lcm-b"
    
    # Test round robin
    node_1 = client.get_image_gen_node()
    node_2 = client.get_image_gen_node()
    node_3 = client.get_image_gen_node()
    
    assert node_1["model"] == "lcm-a"
    assert node_2["model"] == "lcm-b"
    assert node_3["model"] == "lcm-a"
    
    await client.close()

@pytest.mark.asyncio
async def test_llm_client_generate_image():
    image_nodes = [{"url": "http://node1:8000", "model": "lcm"}]
    client = LLMClient("http://upstream:8080", image_gen_nodes=image_nodes)
    
    # Mock the internal httpx client
    mock_post = AsyncMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"b64_json": "test"}]}
    mock_post.return_value = mock_response
    client.image_gen_clients[0]["client"].post = mock_post
    
    res = await client.generate_image({"prompt": "cat", "steps": 5})
    assert res["data"][0]["b64_json"] == "test"
    mock_post.assert_called_once_with("/v1/images/generations", json={"prompt": "cat", "steps": 5})
    
    await client.close()
