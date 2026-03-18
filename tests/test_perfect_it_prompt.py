
import pytest
from unittest.mock import MagicMock, AsyncMock
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
    }
    return agent

@pytest.mark.asyncio
async def test_perfect_it_prompt_has_result_instruction(agent):
    """
    Verify that the 'Perfect It' prompt explicitly instructs the LLM 
    to present the tool output first.
    """
    # Setup the interaction flow:
    # 1. User says "Run code"
    # 2. Agent decides to call 'execute' (Turn 1 start)
    # 3. Tool execution finishes (Turn 1 mid)
    # 4. Agent prepares next request (Turn 1 end) -> "Perfect It" injection triggers here
    
    agent.context.llm_client.chat_completion.side_effect = [
        # Call 1: Agent decides to use execute
        {"choices": [{"message": {"content": None, "tool_calls": [{
            "id": "call_1",
            "function": {"name": "execute", "arguments": '{"code": "print(1)"}'}
        }]}}]},
        # Call 2: Main loop returns final response (must be < 50 chars to trigger perfect it)
        {"choices": [{"message": {"content": "SUCCESS: Execution complete!"}}]},
        # Call 3: Perfect It background worker
        {"choices": [{"message": {"content": "Optimization: Do X"}}]}
    ]
    
    agent.available_tools["execute"].return_value = "EXIT CODE: 0\nSTDOUT: 1"
    
    await agent.handle_chat({"messages": [{"role": "user", "content": "Execute python code"}]}, [])
    
    # Capture the calls made to the LLM
    calls = agent.context.llm_client.chat_completion.call_args_list
    assert len(calls) >= 2, "Expected 2 LLM calls: one for tool choice, one for Perfect It"
    
    # Find the specific call that was the Perfect It injection
    perfect_it_call_args = None
    for call in calls:
        messages_arg = str(call[0][0].get("messages", []))
        if "Perfection Protocol" in messages_arg or "<system_directive>" in messages_arg:
            perfect_it_call_args = call[0][0]
            break
            
    assert perfect_it_call_args is not None, "Could not find Perfect It prompt in any LLM call"
    
    # Verify the specific formatting logic:
    # It must contain the final tool output and the system instruction
    prompt_content = str(perfect_it_call_args["messages"])
    
    assert "EXIT CODE: 0" in prompt_content
    assert "STDOUT: 1" in prompt_content
    # Depending on exact string the new prompt uses, check for system_directive
    assert "<system_directive>" in prompt_content or "Perfection Protocol" in prompt_content
    assert "First, succinctly present the tool output/result to the user" in prompt_content
    assert "Perfection Protocol" in prompt_content
