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


class TestWarmupBackoffMatchesTheNode:
    """§4KA: the 503 backoff was 8 s x 3, sized for a '~5-10 s model load' that
    no longer exists. Readiness is now a REAL 1-step preflight generation
    (13-20 s measured) and a failed one retries 5 times with a 20 s delay, so
    three 8 s sleeps could not outlast even the happy path — a request arriving
    during a node restart burned all three attempts and reported 'generation
    failed' for a node that was merely starting."""

    def test_the_total_wait_covers_a_cold_node(self):
        from ghost_agent.core.llm import LLMClient
        total = sum(LLMClient._IMAGE_WARMUP_BACKOFF)
        assert total >= 40, f"only {total}s of retry wait for a node that takes ~20s to warm"

    def test_the_backoff_escalates(self):
        from ghost_agent.core.llm import LLMClient
        b = LLMClient._IMAGE_WARMUP_BACKOFF
        assert list(b) == sorted(b) and b[0] < b[-1], "a flat backoff wastes the first attempt"

    def test_there_is_one_delay_per_retry(self):
        # attempts 0 and 1 sleep; attempt 2 raises. An index error here would
        # turn a warming node into an unhandled exception.
        from ghost_agent.core.llm import LLMClient
        assert len(LLMClient._IMAGE_WARMUP_BACKOFF) == 2

    def test_it_fits_inside_the_pools_own_timeout(self):
        from ghost_agent.core.llm import LLMClient
        # 45 s of backoff plus a ~200 s generation must still answer inside the
        # 1200 s the pool allows.
        assert sum(LLMClient._IMAGE_WARMUP_BACKOFF) + 200 < 1200
