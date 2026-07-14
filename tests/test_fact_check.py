"""tool_fact_check — 2026-07-14 rewrite regressions.

The old flow burned an LLM round forcing a deep_research tool call via
tool_choice just to rephrase the claim; when the model answered in content
instead (native-tools transport corruption family) the function fell off the
end and returned None. The rewrite calls deep_research directly, then runs a
single verify call — these tests pin the new contract.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import AsyncMock

from ghost_agent.tools.search import tool_fact_check


def _llm(content="The claim is TRUE."):
    client = AsyncMock()
    client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": content}}]})
    return client


@pytest.mark.asyncio
async def test_happy_path_single_llm_round():
    llm = _llm("TRUE — confirmed by three sources.")
    research = AsyncMock(return_value="### SOURCE: x\nEvidence body.")
    out = await tool_fact_check(query="the sky is blue", llm_client=llm,
                                deep_research_callable=research)
    assert out.startswith("FACT CHECK COMPLETE:")
    assert "TRUE" in out
    # Research got the CLAIM directly — no planning round.
    research.assert_awaited_once_with("the sky is blue")
    assert llm.chat_completion.await_count == 1
    # The verify call carries the claim AND the evidence.
    payload = llm.chat_completion.await_args[0][0]
    user_msg = payload["messages"][-1]["content"]
    assert "the sky is blue" in user_msg
    assert "Evidence body." in user_msg


@pytest.mark.asyncio
async def test_statement_alias():
    llm = _llm()
    research = AsyncMock(return_value="evidence")
    out = await tool_fact_check(statement="claim via alias", llm_client=llm,
                                deep_research_callable=research)
    assert out.startswith("FACT CHECK COMPLETE:")
    research.assert_awaited_once_with("claim via alias")


@pytest.mark.asyncio
async def test_empty_query_is_an_error_not_a_research_run():
    llm = _llm()
    research = AsyncMock()
    out = await tool_fact_check(query="   ", llm_client=llm,
                                deep_research_callable=research)
    assert out.startswith("Error")
    research.assert_not_awaited()
    llm.chat_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_never_returns_none_on_null_content():
    # content: null from the server — the old code rendered "...\nNone".
    llm = _llm(content=None)
    research = AsyncMock(return_value="the evidence body")
    out = await tool_fact_check(query="claim", llm_client=llm,
                                deep_research_callable=research)
    assert out is not None
    assert "None" not in out
    # Falls back to handing the evidence over rather than dropping it.
    assert out.startswith("FACT CHECK PARTIAL")
    assert "the evidence body" in out


@pytest.mark.asyncio
async def test_research_failure_returns_error_string():
    llm = _llm()
    research = AsyncMock(side_effect=RuntimeError("tor circuit died"))
    out = await tool_fact_check(query="claim", llm_client=llm,
                                deep_research_callable=research)
    assert out.startswith("Error")
    assert "tor circuit died" in out
    llm.chat_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_verify_failure_still_returns_the_evidence():
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(side_effect=RuntimeError("500 upstream"))
    research = AsyncMock(return_value="hard-won research text")
    out = await tool_fact_check(query="claim", llm_client=llm,
                                deep_research_callable=research)
    assert out.startswith("FACT CHECK PARTIAL")
    assert "hard-won research text" in out
    assert "500 upstream" in out


@pytest.mark.asyncio
async def test_missing_clients_is_a_clear_error():
    out = await tool_fact_check(query="claim")
    assert out.startswith("Error")
    assert "unavailable" in out


@pytest.mark.asyncio
async def test_evidence_capped_to_context():
    # max_context was previously accepted but NEVER used — a giant research
    # report went uncapped into the verify payload.
    llm = _llm()
    research = AsyncMock(return_value="x" * 200_000)
    await tool_fact_check(query="claim", llm_client=llm,
                          deep_research_callable=research, max_context=8192)
    payload = llm.chat_completion.await_args[0][0]
    user_msg = payload["messages"][-1]["content"]
    # cap = max(20_000, 8192*3.5*0.30 ≈ 8_601) = 20_000 (+ envelope text)
    assert len(user_msg) < 25_000
    assert "evidence truncated" in user_msg
