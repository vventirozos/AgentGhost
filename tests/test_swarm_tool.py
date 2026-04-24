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
    # Audit fix #4: in addition to the worker writing the actual result,
    # the dispatcher also stamps a `_swarm_task_id::<output_key>` entry
    # so the agent can poll status from a later turn. Assert the result
    # write happened with the right value, not the once-only count.
    mock_scratchpad.set.assert_any_call("my_key", "Swarm Result")

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
    # See note in test_tool_delegate_to_swarm_success — assert_any_call
    # because the dispatcher also writes a task-id metadata entry.
    mock_scratchpad.set.assert_any_call("my_key_legacy", "Swarm Result Legacy")

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

    # Wait for background tasks to finish — swarm worker now retries
    # (2 retries with exponential backoff: 1s + 2s ≈ 3s total). The
    # dispatcher writes a `_swarm_task_id::my_key` handle immediately,
    # but the alert under "my_key" only lands after the worker exhausts
    # its retries. Poll until that specific call appears.
    for _ in range(80):
        await asyncio.sleep(0.1)
        if any(
            c.args and c.args[0] == "my_key"
            for c in mock_scratchpad.set.call_args_list
        ):
            break

    alert_calls = [
        c for c in mock_scratchpad.set.call_args_list
        if c.args and c.args[0] == "my_key"
    ]
    assert len(alert_calls) == 1, f"expected one alert write to my_key, got {alert_calls}"
    assert "SYSTEM ALERT: Swarm execution failed" in alert_calls[0].args[1]

@pytest.mark.asyncio
async def test_tool_delegate_to_swarm_worker_persona(mock_llm_client):
    mock_llm, mock_node = mock_llm_client
    
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "Persona Result"}}]}
    mock_response.raise_for_status = MagicMock()
    mock_node["client"].post.return_value = mock_response
    
    mock_scratchpad = MagicMock()
    custom_persona = "You are a specialized Custom Persona."
    
    # Run the outer function with custom worker persona
    result = await tool_delegate_to_swarm(
        mock_llm, "test-model", mock_scratchpad, 
        instruction="Analyze", input_data="data", output_key="my_key", 
        worker_persona=custom_persona
    )
    
    assert "SUCCESS" in result
    
    # Wait for background tasks to finish
    await asyncio.sleep(0.1)
    
    mock_node["client"].post.assert_awaited_once()
    
    # Verify persona was injected into the system prompt messages payload
    call_args = mock_node["client"].post.call_args[1]["json"]
    messages = call_args["messages"]
    
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == custom_persona
    # Audit fix #4: dispatcher now stamps a `_swarm_task_id::<key>` poll
    # handle in addition to the worker writing the actual result. Use
    # assert_any_call so the test doesn't break on that extra write.
    mock_scratchpad.set.assert_any_call("my_key", "Persona Result")
