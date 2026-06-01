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

    # The tool now snaps requested size to the nearest SDXL bucket and
    # forwards it on the payload. With no width/height passed, the
    # default 1024x1024 (which IS a bucket) is used unchanged.
    mock_llm_client.generate_image.assert_awaited_once_with({
        "prompt": prompt,
        "steps": 6,
        "width": 1024,
        "height": 1024,
    })

    # Verify result string format. The tool was redesigned to embed a
    # SHORT alt text ("generated image") and instruct the model to
    # describe the image in its own words — deliberately NOT pasting the
    # raw prompt back (so it isn't leaked verbatim into the reply).
    assert result.startswith("SUCCESS: Image generated and saved to sandbox.")
    assert "![generated image](/api/download/gen_" in result
    assert "in your own words" in result
    assert prompt not in result  # raw prompt must not be echoed

    # Extract filename and verify file was written exactly as fake_img_data
    import re
    match = re.search(r"(gen_.+?\.png)", result)
    assert match is not None
    filename = match.group(1)
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
    (2, 4),      # less than min
    (10, 8),      # greater than max
    (6, 6),      # within range
    ("5", 5),    # string castable to int
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
        "steps": expected_steps,
        "width": 1024,
        "height": 1024,
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
