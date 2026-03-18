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
async def test_mode_switching_python_specialist(agent):
    # Mock LLM response
    agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Code", "tool_calls": []}}]
    })
    
    # User asks for python code -> Should trigger specialist mode
    body = {"messages": [{"role": "user", "content": "Write a python script to count numbers"}], "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    
    call_args = agent.context.llm_client.chat_completion.call_args[0][0]
    messages = call_args["messages"]
    
    # Base system prompt at messages[0]
    assert "Ghost" in messages[0]["content"]
    
    # Transient injection with specialized persona at messages[-1]
    transient_injection = messages[-1]["content"]
    assert "Ghost Advanced Engineering Subsystem" in transient_injection
    assert "RAW, EXECUTABLE CODE" in transient_injection

@pytest.mark.asyncio
async def test_system_prompt_additive_logic(agent):
    # Verify that the base SYSTEM_PROMPT is still present when a specialized prompt is added
    
    agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Code", "tool_calls": []}}]
    })
    
    # Trigger Python Specialist
    body = {"messages": [{"role": "user", "content": "Write a python script to count numbers"}], "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    
    call_args = agent.context.llm_client.chat_completion.call_args[0][0]
    messages = call_args["messages"]
    
    # Core prompt is at messages[0]
    assert "You are Ghost" in messages[0]["content"] or "Ghost" in messages[0]["content"]
    
    # Specialized prompt is in the transient injection at messages[-1]
    transient_injection = messages[-1]["content"]
    assert "Ghost Advanced Engineering Subsystem" in transient_injection
    
    # Combined they provide the full instruction set
    full_prompt = messages[0]["content"] + transient_injection
    assert len(full_prompt) > 1000 # Should be substantial

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
    agent.context.args.use_planning = False
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
@pytest.mark.skip(reason="System 2 Planner disabled — Qwen reasons natively")
async def test_planning_logic_trigger(agent):
    # Test triggering planning
    
    # 1. Simple task -> No planning
    # Explicitly disable planning for this part to avoid default Mock(True) behavior
    agent.context.args.use_planning = False
    
    agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Simple Answer", "tool_calls": []}}]
    })
    
    with patch.object(agent, '_prepare_planning_context', return_value="Plan context") as mock_prep:
        body = {"messages": [{"role": "user", "content": "hi"}], "model": "Qwen-Test"}
        await agent.handle_chat(body, background_tasks=MagicMock())
        mock_prep.assert_not_called()

    # 2. Complex task -> Trigger planning
    # Enable planning
    agent.context.args.use_planning = True
    
    # Mock LLM response for the complex task: Planner first, then final answer
    p_msg = {"choices": [{"message": {"content": '{"thought": "t", "tree_update": {}, "next_action_id": "none"}'}}]}
    f_msg = {"choices": [{"message": {"content": "Complex Answer", "tool_calls": []}}]}
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[p_msg, f_msg])

    with patch.object(agent, '_prepare_planning_context', return_value="Plan context") as mock_prep:
        body = {"messages": [{"role": "user", "content": "Write a complex python script to analyze stock data"}], "model": "Qwen-Test"}
        await agent.handle_chat(body, background_tasks=MagicMock())
        mock_prep.assert_called()
        
        # Verify that the planner call used the fast node
        planner_call = agent.context.llm_client.chat_completion.call_args_list[0]
        assert planner_call.kwargs.get("use_swarm") is True

@pytest.mark.asyncio
@pytest.mark.skip(reason="System 2 Planner disabled — Qwen reasons natively")
async def test_planner_context_slicing(agent):
    # Verify that the Planner is only fed the last 2 tool outputs
    agent.context.args.use_planning = True

    # Planner call: {"thought": "t", "tree_update": {}, "next_action_id": "none"}
    p_msg = {"choices": [{"message": {"content": '{"thought": "t", "tree_update": {}, "next_action_id": "n"}'}}]}
    
    # 3 Tool execution calls sequentially (must be different to avoid redundancy blocker and special triggers like execute)
    msg1 = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "t1", "function": {"name": "file_system", "arguments": "{}"}}]}}]}
    msg2 = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "t2", "function": {"name": "system_utility", "arguments": "{}"}}]}}]}
    msg3 = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "t3", "function": {"name": "knowledge_base", "arguments": "{}"}}]}}]}
    
    # Final response
    msg4 = {"choices": [{"message": {"content": "Final Answer", "tool_calls": []}}]}
    
    # Sequence of responses:
    # 1. Planner -> 2. Tool 1 -> 3. Planner -> 4. Tool 2 -> 5. Planner -> 6. Tool 3 -> 7. Planner -> 8. Final Answer
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        p_msg, msg1, p_msg, msg2, p_msg, msg3, p_msg, msg4
    ])
    
    agent.available_tools["file_system"] = AsyncMock(return_value="res1")
    agent.available_tools["system_utility"] = AsyncMock(return_value="res2")
    agent.available_tools["knowledge_base"] = AsyncMock(return_value="res3")
    
    with patch.object(agent, '_prepare_planning_context', return_value="Plan context") as mock_prep:
        body = {"messages": [{"role": "user", "content": "Run 3 complex tasks"}], "model": "Qwen-Test"}
        await agent.handle_chat(body, background_tasks=MagicMock())
        
        # We expect 4 planning calls. The arguments passed should never exceed length 2.
        assert mock_prep.call_count == 4
        
        for call in mock_prep.call_args_list:
            tools_list = call[0][0]
            assert len(tools_list) <= 2
            
        # specifically check the last call had length 2 and contained res2 and res3
        last_call_args = mock_prep.call_args_list[-1][0][0]
        assert len(last_call_args) == 2
        assert last_call_args[0]["content"] == "res2"
        assert last_call_args[1]["content"] == "res3"

@pytest.mark.asyncio
async def test_context_shield_edge_summary(agent):
    # Setup long tool result
    agent.context.args.use_planning = False
    
    long_res = "A" * 12000
    msg1 = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "t1", "function": {"name": "system_utility", "arguments": "{}"}}]}}]}
    msg2 = {"choices": [{"message": {"content": "SUCCESS: Final Answer. " * 20, "tool_calls": []}}]}
    
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
        assert any("Offloading 12000 chars from system_utility to Edge" in msg for msg in log_msgs)
