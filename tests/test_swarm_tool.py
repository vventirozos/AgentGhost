import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.ghost_agent.tools.swarm import tool_delegate_to_swarm

@pytest.fixture
def mock_llm_client():
    mock_llm = MagicMock()
    mock_llm.swarm_clients = [{"client": AsyncMock(), "model": "test-model"}]
    
    mock_node = {"client": AsyncMock(), "model": "test-model"}
    mock_llm.get_swarm_node.return_value = mock_node
    
    return mock_llm, mock_node

@pytest.mark.asyncio
async def test_tool_delegate_to_swarm_success(mock_llm_client):
    mock_llm, mock_node = mock_llm_client
    
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "Swarm Result"}}]}
    mock_response.raise_for_status = MagicMock()
    mock_node["client"].post.return_value = mock_response
    
    mock_scratchpad = MagicMock()
    
    # Run the outer function with tasks list
    tasks = [{"instruction": "Summarize", "input_data": "Some data", "output_key": "my_key"}]
    result = await tool_delegate_to_swarm(mock_llm, "test-model", mock_scratchpad, tasks=tasks)
    
    assert "SUCCESS" in result
    assert "1 task(s)" in result
    
    # Wait for background tasks to finish
    await asyncio.sleep(0.1)
    
    mock_node["client"].post.assert_awaited_once()
    mock_scratchpad.set.assert_called_once_with("my_key", "Swarm Result")

@pytest.mark.asyncio
async def test_tool_delegate_to_swarm_backward_compatibility(mock_llm_client):
    mock_llm, mock_node = mock_llm_client
    
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "Swarm Result Legacy"}}]}
    mock_response.raise_for_status = MagicMock()
    mock_node["client"].post.return_value = mock_response
    
    mock_scratchpad = MagicMock()
    
    # Run using the old kwarg-style invocation
    result = await tool_delegate_to_swarm(mock_llm, "test-model", mock_scratchpad, instruction="Summarize", input_data="data", output_key="my_key_legacy")
    
    assert "SUCCESS" in result
    
    # Wait for background tasks to finish
    await asyncio.sleep(0.1)
    
    mock_node["client"].post.assert_awaited_once()
    mock_scratchpad.set.assert_called_once_with("my_key_legacy", "Swarm Result Legacy")

@pytest.mark.asyncio
async def test_tool_delegate_to_swarm_safeguard_missing_cluster():
    mock_llm = MagicMock()
    mock_llm.swarm_clients = [] # No swarm clients!
    
    result = await tool_delegate_to_swarm(mock_llm, "test-model", MagicMock(), tasks=[{"instruction": "x", "input_data": "y", "output_key": "z"}])
    assert "SYSTEM WARNING: The Swarm Cluster is not configured" in result


@pytest.mark.asyncio
async def test_tool_delegate_to_swarm_missing_scratchpad():
    result = await tool_delegate_to_swarm(MagicMock(), "test-model", None, tasks=[{"instruction": "x", "input_data": "y", "output_key": "z"}])
    assert "Error: Scratchpad memory is not initialized" in result

@pytest.mark.asyncio
async def test_tool_delegate_to_swarm_offline_fallback(mock_llm_client):
    mock_llm, mock_node = mock_llm_client
    mock_node["client"].post.side_effect = Exception("Connection Refused")
    
    mock_scratchpad = MagicMock()
    
    # Run the outer function
    with patch("src.ghost_agent.tools.swarm.pretty_log"):
        result = await tool_delegate_to_swarm(mock_llm, "test-model", mock_scratchpad, instruction="Summarize", input_data="Some data", output_key="my_key")
    
    # Wait for background tasks to finish
    await asyncio.sleep(0.1)
    
    # Scratchpad should be updated with a system alert
    mock_scratchpad.set.assert_called_once()
    # verify the fallback string logic
    args, _ = mock_scratchpad.set.call_args
    assert args[0] == "my_key"
    assert "SYSTEM ALERT: Swarm execution failed" in args[1]
