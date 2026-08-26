import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.tools.search import tool_deep_research
from ghost_agent.core.node_throughput import DistillPlan

# Stub the search wave itself (the URL source): the wave runs each engine on
# a dedicated executor now, not asyncio.to_thread, so patch at that seam — it
# is also more robust than reaching into the wave's threading internals.
@pytest.fixture
def mock_ddgs():
    with patch("importlib.util.find_spec") as mock_find:
        mock_find.return_value = True
        with patch("ghost_agent.tools.search._race_search_wave",
                   new_callable=AsyncMock) as mock_wave:
            mock_wave.return_value = [{"href": "http://example.com/1"}]
            yield mock_wave

@pytest.fixture
def mock_fetch():
    with patch("ghost_agent.tools.search.helper_fetch_url_content", new_callable=AsyncMock) as mock_fetch_content:
        # Return a large text block
        mock_fetch_content.return_value = "A" * 50000
        yield mock_fetch_content

@pytest.mark.asyncio
async def test_deep_research_map_reduce_online(mock_ddgs, mock_fetch):
    # Setup Edge Node LLM Client
    llm_client = MagicMock()
    llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Extracted facts."}}]
    })
    
    # ⚠ SIZED BY THE WORKER'S PLAN, 2026-08-25. This test used to assert
    # `max_tokens == 2048` and a ~40k-char prompt, i.e. it pinned the defect:
    # on Nova that request is 41s of prefill against a 45s budget and could
    # not finish even with the node idle (req 08766aa1). Both numbers now come
    # from `plan_distill`, which reads the node's measured throughput.
    llm_client.plan_distill = lambda budget_s, **kw: DistillPlan(
        12_000, 320, True, "", 220.0, 12.0, 5, 40.0)

    result = await tool_deep_research(
        query="test",
        anonymous=False,
        tor_proxy="",
        llm_client=llm_client,
        model_name="Test-Model",
        max_context=32768,
    )

    # Check that LLM was called to summarize the source text
    assert llm_client.chat_completion.call_count == 1
    call_args = llm_client.chat_completion.call_args[0][0]

    assert call_args["model"] == "Test-Model"
    assert call_args["max_tokens"] == 320          # the plan's, not a constant
    assert call_args["max_tokens"] != 2048         # the number that shipped

    # The source is truncated to the plan's char limit, plus the instruction
    # boilerplate — NOT to a 40k ceiling derived from the main model.
    body = call_args["messages"][0]["content"].split("Source text:\n", 1)[1]
    assert len(body) == 12_000
    assert len(call_args["messages"][0]["content"]) < 40_000
    
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
