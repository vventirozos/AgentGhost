import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.agent import GhostAgent

@pytest.fixture
def agent():
    context = MagicMock()
    context.llm_client.chat_completion = AsyncMock()
    # Ensure args are set defaults
    context.args.use_planning = False
    context.args.smart_memory = 0.0
    context.args.temperature = 0.1
    context.args.max_context = 8000
    context.args.perfect_it = True
    context.sandbox_dir = "/tmp"
    context.memory_system = MagicMock()
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    
    agent = GhostAgent(context)
    # Mock available tools
    agent.available_tools = {
        "execute": AsyncMock(return_value="Exit Code: 0"),
        "test_tool": AsyncMock(return_value="Result")
    }
    return agent

@pytest.mark.asyncio
async def test_perfect_it_security_strip(agent):
    """
    Verify that when 'Perfect It' protocol is triggered, 
    the 'tools' and 'tool_choice' are removed from the LLM payload.
    """
    # 1. Setup: Agent mocks
    # We need a flow: 
    # Turn 1: AI calls 'execute' (heavy tool)
    # Turn 2: AI returns final content. 
    # Logic then triggers "Perfect It".
    
    agent.context.llm_client.chat_completion.side_effect = [
        # Call 1: Agent decides to use execute
        {"choices": [{"message": {"content": None, "tool_calls": [{
            "id": "call_1",
            "function": {"name": "execute", "arguments": '{"code": "print(1)"}'}
        }]}}]},
        # Call 2: Main loop returns final response
        {"choices": [{"message": {"content": "SUCCESS: Done."}}]},
        # Call 3: Perfect It
        {"choices": [{"message": {"content": "Optimization: Do X"}}]}
    ]
    
    agent.available_tools["execute"].return_value = "EXIT CODE: 0"
    
    await agent.handle_chat({"messages": [{"role": "user", "content": "Run code"}]}, [])
    
    # Verify calls
    # Call 1: Agent generates tool call
    # Tool executes (no LLM call)
    # Loop breaks because tool executed
    # Perfect It triggers (Final Call)
    calls = agent.context.llm_client.chat_completion.call_args_list
    assert len(calls) >= 2
    
    # Find the specific call that was the Perfect It injection
    perfect_it_call_args = None
    for call in calls:
        messages_arg = str(call[0][0].get("messages", []))
        if "Perfection Protocol" in messages_arg or "<system_directive>" in messages_arg:
            perfect_it_call_args = call[0][0]
            break
            
    assert perfect_it_call_args is not None, "Could not find Perfect It prompt in any LLM call"

    # Assert tools were stripped
    assert "tools" not in perfect_it_call_args
    assert "tool_choice" not in perfect_it_call_args
    payload_str = str(perfect_it_call_args["messages"])
    assert "Perfection Protocol" in payload_str or "system_directive" in payload_str

@pytest.mark.asyncio
async def test_tool_hallucination(agent):
    """
    Verify agent handles unknown tools gracefully.
    """
    