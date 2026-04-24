import pytest
import json
from unittest.mock import MagicMock, AsyncMock
from ghost_agent.core.agent import GhostAgent

@pytest.fixture
def mock_agent():
    mock_context = MagicMock()
    mock_context.sandbox_dir = "/tmp/sandbox"
    mock_context.memory_system = MagicMock()
    mock_context.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "{}"}}]})
    
    # We enforce 'use_planning' as true to invoke the prompt formatting block
    mock_context.args.use_planning = True 
    mock_context.args.max_context = 32000
    mock_context.args.smart_memory = 0.0
    mock_context.scratchpad.list_all.return_value = "A" * 2000
    mock_context.cached_sandbox_state = "B" * 2000
    
    mock_context.profile_memory.get_context_string.return_value = "Some profile text"
    mock_context.short_term_memory.get_recent_summary.return_value = "Some recent summary"
    
    agent = GhostAgent(mock_context)
    return agent

@pytest.mark.asyncio
@pytest.mark.skip(reason="System 2 Planner disabled — Qwen reasons natively")
async def test_planner_kv_truncation_enforcement(mock_agent):
    messages = [
        {"role": "user", "content": "Update the python script."}
    ]
    tools_run_this_turn = []
    
    body = {
        "messages": messages,
        "model": "Qwen2.5-Coder-7B-Instruct",
        "turn": 1,
        "max_turns": 3
    }
    
    # Run the chat block
    await mock_agent.handle_chat(
        body=body,
        background_tasks=MagicMock(),
        request_id="test_req"
    )
    
    # Grab the payload that the agent fired at the API for the *planning* call
    planner_payload = mock_agent.context.llm_client.chat_completion.call_args_list[0].args[0]
    
    # Extract the system prompt context string that received the `planner_transient` interpolation
    planner_messages = planner_payload.get("messages", [])
    # In the new structure, planner_transient is in the LAST system message (index 2)
    user_planning_block = next((m["content"] for m in reversed(planner_messages) if m["role"] == "user"), "")
    
    # 1. Assert the raw padding was successfully truncated by our logic to prevent out-of-memory errors
    # 2. We assert it's less than 1600 (1500 + length of '\n...[TRUNCATED]')
    assert "A" * 2000 not in user_planning_block, "The scratch data was not truncated."
    assert "A" * 1500 in user_planning_block, "The scratch data truncater fired too early."
    assert "...[TRUNCATED]" in user_planning_block, "Truncation indicator missing from scrapbook."

    assert "B" * 2000 not in user_planning_block, "The sandbox state data was not truncated."
    assert "B" * 1500 in user_planning_block, "The sandbox data truncater fired too early."
    assert "...[TRUNCATED]" in user_planning_block, "Truncation indicator missing from sandbox state."
