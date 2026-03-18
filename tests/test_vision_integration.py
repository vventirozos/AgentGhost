import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from src.ghost_agent.tools.registry import get_active_tool_definitions, get_available_tools
from src.ghost_agent.core.llm import LLMClient
from src.ghost_agent.tools.vision import tool_vision_analysis
from src.ghost_agent.core.agent import GhostAgent, GhostContext

class DummyArgs:
    model = "default-model"
    temperature = 0.5
    max_context = 8192
    smart_memory = 0.0
    use_planning = False
    anonymous = False

@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    client.vision_clients = []
    return client

@pytest.fixture
def mock_context(mock_llm_client, tmp_path):
    ctx = MagicMock(spec=GhostContext)
    ctx.args = DummyArgs()
    ctx.llm_client = mock_llm_client
    ctx.sandbox_dir = tmp_path
    ctx.tor_proxy = None
    ctx.scratchpad = MagicMock()
    return ctx

def test_get_active_tool_definitions_no_vision(mock_context):
    mock_context.llm_client.vision_clients = None
    tools = get_active_tool_definitions(mock_context)
    names = [t["function"]["name"] for t in tools]
    assert "vision_analysis" in names

def test_get_active_tool_definitions_with_vision(mock_context):
    mock_context.llm_client.vision_clients = [{"client": AsyncMock()}]
    tools = get_active_tool_definitions(mock_context)
    names = [t["function"]["name"] for t in tools]
    assert "vision_analysis" in names

def test_get_available_tools_vision_injected(mock_context):
    mock_context.llm_client.vision_clients = [{"client": AsyncMock()}]
    tools = get_available_tools(mock_context)
    assert "vision_analysis" in tools
    assert callable(tools["vision_analysis"])

@pytest.mark.asyncio
async def test_tool_vision_analysis_no_configured_nodes(mock_context):
    mock_context.llm_client.vision_clients = None
    
    # Mock LLM response
    mock_resp = {"choices": [{"message": {"content": "native vision response"}}]}
    mock_context.llm_client.chat_completion = AsyncMock(return_value=mock_resp)
    
    with patch("src.ghost_agent.tools.vision._get_safe_path") as mock_safe_path, \
         patch("asyncio.to_thread") as mock_thread:
        
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_safe_path.return_value = mock_path_obj
        mock_thread.return_value = b"fakebytes"
        
        res = await tool_vision_analysis(
            action="describe_picture",
            target="file.jpg",
            llm_client=mock_context.llm_client,
            sandbox_dir=mock_context.sandbox_dir
        )
        assert "native vision response" in res

@pytest.mark.asyncio
async def test_llm_client_vision_routing():
    # Test llm_client get_vision_node logic
    client = LLMClient(upstream_url="http://fake", visual_nodes=[{"url": "http://vision", "model": "vision-model"}])
    assert client.vision_clients is not None
    assert len(client.vision_clients) == 1
    
    node = client.get_vision_node()
    assert node["model"] == "vision-model"

    # Mock post response
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "vision response"}}]}
    mock_resp.raise_for_status = MagicMock()
    client.vision_clients[0]["client"].post = AsyncMock(return_value=mock_resp)
    
    # Test chat_completion with use_vision=True
    res = await client.chat_completion({"model": "test"}, use_vision=True)
    assert res["choices"][0]["message"]["content"] == "vision response"

    await client.close()

@pytest.mark.asyncio
async def test_tool_vision_analysis_path_healing(mock_context):
    client = MagicMock(spec=LLMClient)
    client.vision_clients = [{"client": AsyncMock()}]
    mock_resp = {"choices": [{"message": {"content": "vision response"}}]}
    client.chat_completion = AsyncMock(return_value=mock_resp)

    # Mock file reading so it doesn't crash on absolute path reads
    with patch("src.ghost_agent.tools.vision._get_safe_path") as mock_safe_path, \
         patch("asyncio.to_thread") as mock_thread:
        
        # Test case 1: /sandbox/ prefix stripped
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_safe_path.return_value = mock_path_obj
        mock_thread.return_value = b"fakebytes"

        target_hallucinated = "/sandbox/gen_abcd.png"
        res = await tool_vision_analysis(
            action="describe_picture",
            target=target_hallucinated,
            llm_client=client,
            sandbox_dir=mock_context.sandbox_dir
        )
        assert mock_safe_path.call_args[0][1] == "/gen_abcd.png", "The /sandbox/ prefix should have been replaced with /"
        assert "vision response" in res

        # Test case 2: /api/download/ prefix stripped
        target_api = "/api/download/gen_xyz.png"
        res2 = await tool_vision_analysis(
            action="describe_picture",
            target=target_api,
            llm_client=client,
            sandbox_dir=mock_context.sandbox_dir
        )
        assert mock_safe_path.call_args[0][1] == "gen_xyz.png", "The /api/download/ prefix should have been removed entirely"
        assert "vision response" in res2
