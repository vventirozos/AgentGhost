import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def agent(mock_context):
    return GhostAgent(mock_context)

@pytest.mark.asyncio
async def test_agent_initialization(agent):
    assert agent.context is not None
    assert agent.available_tools is not None

@pytest.mark.asyncio
async def test_handle_chat_basic_flow(agent):
    # Mock LLM response
    agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Hello User", "tool_calls": []}}]
    })
    
    body = {"messages": [{"role": "user", "content": "Hi"}], "model": "Qwen-Test"}
    content, _, _ = await agent.handle_chat(body, background_tasks=MagicMock())
    
    assert content == "Hello User"
    # Verify System Prompt Injection
    call_args = agent.context.llm_client.chat_completion.call_args[0][0]
    messages = call_args["messages"]
    assert messages[0]["role"] == "system"
    assert "Ghost" in messages[0]["content"]

@pytest.mark.asyncio
async def test_a_coding_request_gets_no_persona_but_the_sandbox(agent):
    # §4UI (2026-11-12): the coding persona ("Specialist") was removed — the
    # base prompt carries the coding rules; a coding request gets the sandbox
    agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Code", "tool_calls": []}}]
    })
    body = {"messages": [{"role": "user", "content": "Write a python script to count numbers"}], "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    messages = agent.context.llm_client.chat_completion.call_args[0][0]["messages"]
    assert "Ghost" in messages[0]["content"]
    blob = "\n".join(str(m.get("content")) for m in messages)
    assert "Ghost Advanced Engineering and Database Subsystem" not in blob
    assert "CURRENT SANDBOX STATE" in blob

@pytest.mark.asyncio
async def test_history_truncation(agent):
    # Create long history
    msgs = [{"role": "user", "content": str(i)} for i in range(600)]
    body = {"messages": msgs, "model": "Qwen-Test"}
    
    agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Done", "tool_calls": []}}]
    })
    
    await agent.handle_chat(body, background_tasks=MagicMock())
    
    call_args = agent.context.llm_client.chat_completion.call_args[0][0]
    sent_messages = call_args["messages"]
    
    # Should be truncated to approx 500 + system prompt + new msgs
    # The agent code: if len > 500: keep system + last 500
    # sent_messages includes the history sent to LLM
    assert len(sent_messages) <= 505 # Allow some buffer for injected system/memory prompts

@pytest.mark.asyncio
async def test_tool_execution_loop(agent):
    # Mock LLM to return a tool call then a final answer
    # Turn 1: Call tool
    msg1 = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "file_system", "arguments": '{"operation": "list"}'}
                }]
            }
        }]
    }
    msg2 = {
        "choices": [{"message": {"content": "Here are files", "tool_calls": []}}]
    }
    
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[msg1, msg2])
    
    # Mock Tool execution
    # IMPORTANT: The tool MUST return a string, otherwise pretty_log crashes when it tries to log it.
    agent.available_tools["file_system"] = AsyncMock(return_value="file1.txt")
    
    body = {"messages": [{"role": "user", "content": "List files"}], "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    
    # Verify tool was called
    agent.available_tools["file_system"].assert_called_once()

@pytest.mark.asyncio
async def test_context_shield_edge_summary(agent):
    # Setup long tool result
    
    long_res = "A" * 20000
    msg1 = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "t1", "function": {"name": "system_utility", "arguments": "{}"}}]}}]}
    # Shortened from `"... " * 20` — tightened loop detector now
    # flags exact-string repetition at ~300 chars.
    msg2 = {"choices": [{"message": {"content": "SUCCESS: Final Answer.", "tool_calls": []}}]}
    
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        msg1,
        # Edge Node summary response
        {"choices": [{"message": {"content": "Summarized successfully."}}]},
        msg2
    ])
    
    agent.available_tools["system_utility"] = AsyncMock(return_value=long_res)
    
    body = {"messages": [{"role": "user", "content": "Run tool"}], "model": "Qwen-Test"}
    
    with patch("ghost_agent.core.agent.pretty_log") as mock_log:
        await agent.handle_chat(body, background_tasks=MagicMock())
        
        # Verify edge node was called for summarization
        assert agent.context.llm_client.chat_completion.call_count == 3
        
        # Check that the 2nd call was the Context Shield payload
        summary_call = agent.context.llm_client.chat_completion.call_args_list[1]
        assert summary_call.kwargs.get("use_worker") is True
        assert "Summarize this tool output" in summary_call.args[0]["messages"][0]["content"]
        
        # Check that the 3rd call to the main LLM received the condensed edge summary
        final_call = agent.context.llm_client.chat_completion.call_args_list[2]
        # The tool response is at messages[-1] because the transient system injection is now appended string-wise
        tool_response_msg = final_call.args[0]["messages"][-1]
        assert tool_response_msg.get("role") == "user"
        assert "[EDGE CONDENSED]: Summarized successfully." in tool_response_msg["content"]
        
        # Check logs
        log_msgs = [str(call) for call in mock_log.call_args_list]
        assert any("Offloading 20000 chars from system_utility to Edge" in msg for msg in log_msgs)

@pytest.mark.asyncio
async def test_parallel_tool_execution(agent):
    # Mock LLM to return 2 tool calls in a single turn, simulating parallel tool execution
    
    msg1 = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "file_system", "arguments": '{"operation": "list", "path": "/"}'}},
                    {"id": "call_2", "function": {"name": "system_utility", "arguments": '{"action": "check_health"}'}}
                ]
            }
        }]
    }
    msg2 = {
        "choices": [{"message": {"content": "Parallel tools completed", "tool_calls": []}}]
    }
    
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[msg1, msg2])
    
    # Mock Tool execution
    agent.available_tools["file_system"] = AsyncMock(return_value="file list")
    agent.available_tools["system_utility"] = AsyncMock(return_value="System OK")
    
    body = {"messages": [{"role": "user", "content": "Run two tasks concurrently"}], "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    
    # Verify both simulated parallel tools were called
    agent.available_tools["file_system"].assert_called_once()
    agent.available_tools["system_utility"].assert_called_once()
