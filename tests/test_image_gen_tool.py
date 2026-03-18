import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
import json

from ghost_agent.tools.image_gen import tool_generate_image
from ghost_agent.core.llm import LLMClient
from ghost_agent.tools.registry import get_available_tools, get_active_tool_definitions

@pytest.mark.asyncio
async def test_tool_generate_image_offline(tmp_path):
    mock_llm_client = AsyncMock()
    mock_llm_client.image_gen_clients = []
    
    result = await tool_generate_image(prompt="test", steps=5, llm_client=mock_llm_client, sandbox_dir=tmp_path)
    assert "ERROR" in result
    assert "offline" in result.lower()

@pytest.mark.asyncio
async def test_tool_generate_image_success(tmp_path):
    mock_llm_client = AsyncMock()
    mock_llm_client.image_gen_clients = [{"model": "lcm-model"}]
    
    # Mock base64 image data (just "test" string base64 encoded to avoid massive payload)
    mock_llm_client.generate_image.return_value = {"data": [{"b64_json": "dGVzdA=="}]}
    
    result = await tool_generate_image(prompt="draw a cat", steps=4, llm_client=mock_llm_client, sandbox_dir=tmp_path)
    
    assert "SUCCESS" in result
    assert "![Image](/api/download/gen_" in result
    
    # Check if file was written
    files = list(tmp_path.glob("gen_*.png"))
    assert len(files) == 1
    assert files[0].read_bytes() == b"test"
    
def test_image_gen_tool_registration():
    mock_context = MagicMock()
    
    # Without clients
    mock_context.llm_client.image_gen_clients = []
    defs = get_active_tool_definitions(mock_context)
    assert not any(t["function"]["name"] == "image_generation" for t in defs)
    
    tools = get_available_tools(mock_context)
    assert "image_generation" not in tools
    
    # With clients
    mock_context.llm_client.image_gen_clients = [{"model": "lcm"}]
    defs = get_active_tool_definitions(mock_context)
    assert any(t["function"]["name"] == "image_generation" for t in defs)
    
    tools = get_available_tools(mock_context)
    assert "image_generation" in tools
