import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.tools.search import tool_deep_research

# We need to mock importlib.util.find_spec("ddgs") and asyncio.to_thread
@pytest.fixture
def mock_ddgs():
    with patch("importlib.util.find_spec") as mock_find:
        mock_find.return_value = True
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = [{"href": "http://example.com/1"}]
            yield mock_thread

@pytest.fixture
def mock_fetch():
    with patch("ghost_agent.tools.search.helper_fetch_url_content", new_callable=AsyncMock) as mock_fetch_content:
        # Return a large text block
        mock_fetch_content.return_value = "A" * 20000
        yield mock_fetch_content

@pytest.mark.asyncio
async def test_deep_research_map_reduce_online(mock_ddgs, mock_fetch):
    # Setup Edge Node LLM Client
    llm_client = MagicMock()
    llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Extracted facts."}}]
    })
    
    result = await tool_deep_research(
        query="test", 
        anonymous=False, 
        tor_proxy="", 
        llm_client=llm_client, 
        model_name="Test-Model"
    )
    
    # Check that LLM was called to summarize the 15k chars
    assert llm_client.chat_completion.call_count == 1
    call_args = llm_client.chat_completion.call_args[0][0]
    
    assert call_args["model"] == "Test-Model"
    assert call_args["max_tokens"] == 500
    
    # The source text should be truncated to 15000 in the prompt
    assert len(call_args["messages"][0]["content"]) > 15000
    
    # The result should contain the edge extracted facts label
    assert "[EDGE EXTRACTED FACTS]:" in result
    assert "Extracted facts." in result

@pytest.mark.asyncio
async def test_deep_research_map_reduce_offline(mock_ddgs, mock_fetch):
    # Setup Edge Node LLM Client to fail (offline)
    llm_client = MagicMock()
    llm_client.chat_completion = AsyncMock(side_effect=Exception("Offline"))
    
    result = await tool_deep_research(
        query="test", 
        anonymous=False, 
        tor_proxy="", 
        llm_client=llm_client, 
        model_name="Test-Model"
    )
    
    # Should fallback to 3000 chars of source text
    assert llm_client.chat_completion.call_count == 1
    assert "A" * 3000 in result
    # It shouldn't contain more than that since preview is limited
    # len(result) is ~3000 chars + boilerplate
    assert "[EDGE EXTRACTED FACTS]:" not in result

@pytest.mark.asyncio
async def test_deep_research_map_reduce_none(mock_ddgs, mock_fetch):
    
    result = await tool_deep_research(
        query="test", 
        anonymous=False, 
        tor_proxy="", 
        llm_client=None, 
    )
    
    # Should fallback to 3000 chars of source text directly without calling lmm
    assert "A" * 3000 in result
    assert "[EDGE EXTRACTED FACTS]:" not in result
