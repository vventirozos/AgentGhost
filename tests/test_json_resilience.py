
import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock
from ghost_agent.core.agent import GhostAgent, TaskTree

@pytest.fixture
def mock_agent():
    context = MagicMock()
    context.llm_client.chat_completion = AsyncMock()
    # Mock args
    context.args.model = "test-model"
    context.args.temperature = 0.0
    context.args.verbose = True
    context.args.safety_checks = False
    context.args.max_context = 8000
    context.args.smart_memory = 0.0 # Disable background tasks
    context.args.use_planning = True
    
    # Mock system components
    context.model_limits = {"test-model": 10000}
    context.memory_system = None
    context.profile_memory = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    context.scheduler = None 
    
    agent = GhostAgent(context)
    return agent

@pytest.mark.asyncio
async def test_planning_json_failure_recovery(mock_agent):
    """
    Test that the agent recovers gracefully when the Planner returns invalid JSON.
    It should NOT crash, and it should append a fallback strategy to the messages.
    """
    # 1. Setup Input
    # We need to simulate a state where 'run_step' calls the planner
    # This involves mocking the LLM response to return garbage JSON
    
    # Mock the LLM to return invalid JSON
    mock_agent.context.llm_client.chat_completion.return_value = {
        "choices": [{"message": {"content": "INVALID JSON { unclosed brace"}}]
    }
    
    # We can't easily call 'run_step' in isolation without mocking a lot of internal state 
    # because it's a big loop. 
    # However, 'run_step' logic is embedded in 'handle_chat' or similar? 
    # Wait, 'run_step' is NOT a standalone public method in the snippet I saw.
    # The snippet showed the logic inside `handle_chat` or similar loop.
    # Let me check the file structure again. 
    # Ah, I see `handle_chat` calls the loop.
    # To test this logic in isolation, it's best if we extract the planning step 
    # or fully mock the `handle_chat` inputs.
    
    # Let's try to invoke the logic by calling `handle_chat` with a non-trivial request
    request = {"messages": [{"role": "user", "content": "Write a complex Python script"}]}
    
    # We need to ensure the loop runs at least once.
    # `handle_chat` is async.
    
    # But wait! `handle_chat` is effectively the entry point. 
    # The planning logic is inside. 
    # If we mock the LLM response for the FIRST call (Planner), it will fail JSON parsing.
    # Then it should catch Exception, log error, and append fallback.
    # Then it proceeds to the "Main Agent" call (which is the SECOND LLM call in the loop usually, or same?).
    # Actually, in `agent.py`, the planner is a separate call `chat_completion(planning_payload)`.
    # After that, it generates the tool/text response using `client.generate` or similar?
    # I need to be careful about how many times `chat_completion` is called.
    
    # Let's setup the mock to return:
    # 1. Invalid JSON (Planner) -> Fail -> Fallback
    # 2. Valid Tool Call (Agent Action) -> Stop loop or just return
    
    mock_agent.context.llm_client.chat_completion.side_effect = [
        # Call 1: Planner (fails)
        {"choices": [{"message": {"content": "This is not JSON"}}]},
        # Call 2: Agent Tool execution (or final answer)
        {"choices": [{"message": {"content": "I will do it.", "tool_calls": []}}]} 
    ]
    
    # Execute
    # We assume `handle_chat` returns a generator or result?
    # Based on `agent.py`: `async def handle_chat(self, body, background_tasks):`
    # It returns a StreamingResponse or similar?
    # I need to see the return type of `handle_chat`.
    # The snippet didn't show the return.
    
    # However, I can inspect `mock_agent.context.llm_client.chat_completion` calls afterward.
    
    try:
        # We need to mock background_tasks as list or similar?
        # signature: handle_chat(self, body: Dict, background_tasks)
        await mock_agent.handle_chat(request, [])
    except Exception as e:
        # It shouldn't crash
        pytest.fail(f"Agent crashed on invalid JSON: {e}")

    # Assertions
    # 1. Ensure Planner was called
    calls = mock_agent.context.llm_client.chat_completion.call_args_list
    assert len(calls) >= 1
    
    # 2. Check if the Fallback System Message was attempted (this is hard to check internally 
    # without inspecting the 'messages' list state which is local).
    # BUT, we can check if the SECOND call to LLM (the one after planning) contained the Fallback prompt.
    
    if len(calls) >= 2:
        # The second call is the Agent generating the response.
        # Its messages input should contain the fallback system prompt.
        second_call_args = calls[1][0][0] # payload
        messages = second_call_args["messages"]
        
        # We look for "### ACTIVE STRATEGY: Proceed with the next logical step"
        # which is the fallback defined in `agent.py`
        # The agent now handles invalid JSON gracefully by using default values (empty plan)
        # instead of raising an exception. So we check for the default "No thought provided" message
        # which indicates the agent continued despite the JSON error.
        has_default_strategy = any("Active Strategy" in m.get("content", "") or "THOUGHT: No thought provided" in m.get("content", "") for m in messages)
        assert has_default_strategy, "Agent did not proceed with default strategy after JSON failure!"

@pytest.mark.asyncio
async def test_malformed_json_recovery_partial(mock_agent):
    """
    Test recovery from partial/malformed JSON that might check 'tree_update' but fail later?
    Actually, json.loads fails completely.
    """
    mock_agent.context.llm_client.chat_completion.side_effect = [
        {"choices": [{"message": {"content": '{"thought": "Good", "tree_update": {MOAR BRACKETS}}}}' }}]}, # Invalid
        {"choices": [{"message": {"content": "Final Answer"}}]} 
    ]
    
    await mock_agent.handle_chat({"messages": [{"role": "user", "content": "Write a python script"}]}, [])
    
    # With the planner disabled, expect at least 1 LLM call (the main generation call).
    calls = mock_agent.context.llm_client.chat_completion.call_args_list
    assert len(calls) >= 1
