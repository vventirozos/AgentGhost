import pytest

@pytest.mark.asyncio
async def test_fact_check_payload_isolation():
    """Ensure fact_check correctly re-serializes payload dictionaries back to strings during sequential evaluation"""
    from src.ghost_agent.tools.search import tool_fact_check
    from unittest.mock import AsyncMock
    
    # Mock LLM Client
    mock_llm = AsyncMock()
    
    # Mock first response: Planning to use deep_research
    plan_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "deep_research",
                    # The bug: upstream has already evaluated this into a dict, so fact_check sees a natively parsed dict in its historical context array
                    "arguments": {"query": "PostgreSQL 18 latest version release date"}
                }
            }
        ]
    }
    
    # Mock second response: Verification completion
    verify_msg = {
        "role": "assistant",
        "content": "Verified: PostgreSQL 18 will be released soon."
    }
    
    # Setup LLM responses sequentially
    mock_llm.chat_completion.side_effect = [
        {"choices": [{"message": plan_msg}]},
        {"choices": [{"message": verify_msg}]}
    ]
    
    # Mock the internal callable
    mock_deep_research = AsyncMock(return_value="DEEP RESEARCH: Mocked Results")
    
    res = await tool_fact_check(
        query="PostgreSQL 18 release date",
        llm_client=mock_llm,
        tool_definitions=[{"type": "function", "function": {"name": "deep_research"}}],
        deep_research_callable=mock_deep_research
    )
    
    assert "Verification" in res or "Verified:" in res
    
    # The CRITICAL check: The second payload (verify_payload) must NOT contain native tool_calls
    # to bypass the Llama-Server C++ API string constraint vs Jinja template dictionary constraint paradox.
    calls = mock_llm.chat_completion.call_args_list
    assert len(calls) == 2, "Expected 2 chat_completion calls"
    
    verify_payload = calls[1][0][0] # First argument (payload dict) of the second call
    historical_ai_msg = verify_payload["messages"][2] # [system, user, system_bypass]
    
    assert historical_ai_msg["role"] == "user", "Expected the bypassed user message role"
    assert "[RESEARCH RESULTS]" in historical_ai_msg["content"], "Expected dr_result content"
    assert "DEEP RESEARCH: Mocked Results" in historical_ai_msg["content"]
    assert "tool_calls" not in historical_ai_msg, "CRITICAL: Tool calls must be stripped to prevent 500 API errors"

