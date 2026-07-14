import pytest

@pytest.mark.asyncio
async def test_fact_check_payload_isolation():
    """The 2026-07-14 fact_check rewrite eliminates the planning tool-call
    round entirely, so the Llama-Server "JSON string vs Jinja dict" paradox
    this test used to guard against is now structurally impossible: fact_check
    makes exactly ONE chat_completion call (the verify call), none of its
    messages carry a native `tool_calls` array, and the payload has no
    `tools`/`tool_choice` keys at all."""
    from src.ghost_agent.tools.search import tool_fact_check
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock()
    mock_llm.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {
            "content": "Verified: PostgreSQL 18 will be released soon."}}]})
    mock_deep_research = AsyncMock(return_value="DEEP RESEARCH: Mocked Results")

    res = await tool_fact_check(
        query="PostgreSQL 18 release date",
        llm_client=mock_llm,
        tool_definitions=[{"type": "function", "function": {"name": "deep_research"}}],
        deep_research_callable=mock_deep_research,
    )

    assert "Verified:" in res

    calls = mock_llm.chat_completion.call_args_list
    assert len(calls) == 1, "fact_check must make exactly ONE LLM call (verify)"

    verify_payload = calls[0][0][0]
    assert "tools" not in verify_payload
    assert "tool_choice" not in verify_payload
    for msg in verify_payload["messages"]:
        assert "tool_calls" not in msg, (
            "CRITICAL: no message may carry native tool_calls — that shape "
            "crashed Llama-Server's chat-template/API-schema mismatch")
    # The research result reached the verify prompt as plain user content.
    assert "[RESEARCH RESULTS]" in verify_payload["messages"][-1]["content"]
    assert "DEEP RESEARCH: Mocked Results" in verify_payload["messages"][-1]["content"]
