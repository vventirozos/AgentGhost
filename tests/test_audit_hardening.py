import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from ghost_agent.core.llm import LLMClient
from ghost_agent.core.agent import extract_json_from_text
from ghost_agent.tools.search import tool_fact_check

@pytest.mark.asyncio
async def test_llm_get_embeddings_max_retries():
    client = LLMClient("http://mock-url")
    # Mock post to consistently fail with ConnectError
    client.http_client.post = AsyncMock(side_effect=httpx.ConnectError("Test Connect Error"))
    
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(httpx.ConnectError, match="Test Connect Error"):
            await client.get_embeddings(["test string for embeddings"])
    
    # Assert it retried 2 times
    assert client.http_client.post.call_count == 2

@pytest.mark.asyncio
async def test_llm_stream_chat_completion_max_retries():
    client = LLMClient("http://mock-url")
    # Mock send to consistently fail with ConnectError
    client.http_client.send = AsyncMock(side_effect=httpx.ConnectError("Test Stream Connect Error"))
    
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(httpx.ConnectError, match="Test Stream Connect Error"):
            async for chunk in client.stream_chat_completion({"model": "test_model"}, use_coding=False):
                pass
                
    # Assert it retried 2 times
    assert client.http_client.send.call_count == 2

def test_agent_extract_json_fallback_logging():
    # Malformed text inside codeblocks that fails JSON parsing AND literal_eval AST fallback
    bad_json_payload = "```json\n{ invalid :: syntax }\n```"
    
    with patch("ghost_agent.core.agent.logger.debug") as mock_logger:
        result = extract_json_from_text(bad_json_payload)
        
        # Must gracefully return empty dict
        assert result == {}
        
        # Verify the custom catch logic correctly logged the parsed exception name
        assert mock_logger.call_count > 0
        logged_calls = [call[0][0] for call in mock_logger.call_args_list]
        
        # It should trigger the inner AST fallback block debug emission
        assert any("JSON AST fallback failed" in log for log in logged_calls)

@pytest.mark.asyncio
async def test_llm_client_node_optional_none_getters():
    client = LLMClient("http://mock-url")
    # Since we changed type hints to Optional[str], calling with None should resolve dynamically
    swarm_res = client.get_swarm_node(target_model=None)
    vision_res = client.get_vision_node(target_model=None)
    worker_res = client.get_worker_node(target_model=None)
    coding_res = client.get_coding_node(target_model=None)
    
    assert swarm_res is None or isinstance(swarm_res, dict)
    assert vision_res is None or isinstance(vision_res, dict)
    assert worker_res is None or isinstance(worker_res, dict)
    assert coding_res is None or isinstance(coding_res, dict)

@pytest.mark.asyncio
async def test_search_fact_check_optional_arguments_kwargs():
    # Assert the search tool gracefully processes kwargs via Dict[str, Any] and Optional boundaries
    with patch("ghost_agent.tools.search.tool_deep_research", new_callable=AsyncMock) as mock_dr:
        mock_dr.return_value = "Simulated valid facts."
        
        mock_llm = AsyncMock()
        mock_llm.chat_completion.side_effect = [
            {"choices": [{"message": {"tool_calls": [{"id": "1", "function": {"name": "deep_research", "arguments": "{}"}}]}}]},
            {"choices": [{"message": {"content": "Simulated valid facts inside the final response."}}]}
        ]
        result = await tool_fact_check(query=None, statement="The sky is green", extraneous_dict={"key": "val"}, tool_definitions=[{"function": {"name": "deep_research"}}], llm_client=mock_llm, deep_research_callable=mock_dr)
        
        assert "Simulated valid facts inside the final response." in result
        mock_dr.assert_called_once()
