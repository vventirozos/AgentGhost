import pytest
import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from src.ghost_agent.tools.image_gen import tool_generate_image

@pytest.fixture
def mock_sandbox(tmp_path):
    return tmp_path

@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    # By default, assume it has image_gen_clients
    client.image_gen_clients = ["mock_client"]
    client.generate_image = AsyncMock()
    return client

@pytest.mark.asyncio
async def test_tool_generate_image_success(mock_sandbox, mock_llm_client):
    # Setup mock response
    fake_img_data = b"fake_image_data_bytes"
    fake_b64 = base64.b64encode(fake_img_data).decode('utf-8')
    
    mock_llm_client.generate_image.return_value = {
        "data": [{"b64_json": fake_b64}]
    }

    prompt = "A detailed test image"
    
    # Run the tool
    result = await tool_generate_image(prompt, mock_llm_client, mock_sandbox)

    # Verify generate_image was called correctly with default 50 steps (clipped to 50)
    mock_llm_client.generate_image.assert_awaited_once_with({
        "prompt": prompt,
        "steps": 50
    })

    # Verify result string format
    assert result.startswith("SUCCESS: Image generated and saved to sandbox.")
    assert "![Image](/api/download/gen_" in result
    assert result.endswith(".png)")

    # Extract filename and verify file was written exactly as fake_img_data
    # Extract filename from: ![Image](/api/download/gen_1b9d6bcd.png)
    filename = result.split("/api/download/")[-1].replace(")", "")
    file_path = mock_sandbox / filename
    
    assert file_path.exists()
    assert file_path.read_bytes() == fake_img_data

@pytest.mark.asyncio
async def test_tool_generate_image_offline_client(mock_sandbox, mock_llm_client):
    # Make image_gen_clients None or empty
    mock_llm_client.image_gen_clients = []
    
    result = await tool_generate_image("Test prompt", mock_llm_client, mock_sandbox)
    
    assert result == "ERROR: Image generation node is offline or not configured."
    mock_llm_client.generate_image.assert_not_called()

@pytest.mark.asyncio
@pytest.mark.parametrize("input_steps, expected_steps", [
    (10, 40),      # less than min
    (60, 50),      # greater than max
    (45, 45),      # within range
    ("42", 42),    # string castable to int
])
async def test_tool_generate_image_steps_clipping(mock_sandbox, mock_llm_client, input_steps, expected_steps):
    fake_img_data = b"fake"
    fake_b64 = base64.b64encode(fake_img_data).decode('utf-8')
    mock_llm_client.generate_image.return_value = {
        "data": [{"b64_json": fake_b64}]
    }
    
    await tool_generate_image("prompt", mock_llm_client, mock_sandbox, steps=input_steps)
    
    mock_llm_client.generate_image.assert_awaited_once_with({
        "prompt": "prompt",
        "steps": expected_steps
    })

@pytest.mark.asyncio
async def test_tool_generate_image_invalid_steps_type(mock_sandbox, mock_llm_client):
    # int("invalid") raises ValueError before the try block
    with pytest.raises(ValueError, match="invalid literal for int"):
        await tool_generate_image("prompt", mock_llm_client, mock_sandbox, steps="invalid")
    
    mock_llm_client.generate_image.assert_not_called()

@pytest.mark.asyncio
async def test_tool_generate_image_api_exception(mock_sandbox, mock_llm_client):
    # Simulate API error
    mock_llm_client.generate_image.side_effect = Exception("API timeout")
    
    result = await tool_generate_image("prompt", mock_llm_client, mock_sandbox)
    
    assert result == "ERROR generating image: API timeout"
