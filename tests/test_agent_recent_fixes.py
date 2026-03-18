import pytest
import copy
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def agent():
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.7
    ctx.args.max_context = 8000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    
    ctx.llm_client = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Hello", "tool_calls": []}}]
    })
    
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_context_string.return_value = ""
    ctx.memory_system = MagicMock()
    ctx.memory_system.search = MagicMock(return_value="")
    ctx.cached_sandbox_state = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.sandbox = MagicMock()
    ctx.sandbox.run_code = MagicMock(return_value="EXIT CODE: 0")
    
    agent_inst = GhostAgent(ctx)
    return agent_inst

@pytest.mark.asyncio
async def test_prune_context_exact_ordering(agent):
    """
    Verifies that `_prune_context` maintains strict chronological ordering
    and does not append `last_user` to the very end out of order.
    """
    # Create a history with: System -> User 1 -> Tool Call -> Tool Result -> User 2
    # We will make max_tokens small enough to cut out 'User 1' but keep 'User 2' and 'Tool' stuff.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Very long user question 1 " * 100}, # Should be pruned
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "function": {"name": "test_tool"}}]},
        {"role": "tool", "content": "Tool result here"},
        {"role": "user", "content": "Short question 2"} # This is last_user
    ]
    
    # Let's prune with a budget that allows System (approx 5 tokens) + Buffer (500) + short question 2 (approx 3 tokens)
    # + tool elements, but NOT the long user question which is ~500 tokens alone.
    # We need to simulate the async LLM call if it triggers summarization, but here it shouldn't because max_tokens 600 > current load.
    pruned = await agent._prune_context(messages, max_tokens=600, model="test")
    
    assert len(pruned) < 5
    
    # The order must be exactly chronological based on the remaining items
    assert pruned[0]["role"] == "system"
    # Ensure it ends with user, tool, assistant in the correct relative original order
    roles = [m["role"] for m in pruned]
    
    # We definitely want to keep the system and the last user
    assert roles[0] == "system"
    assert roles[-1] == "user"
    assert pruned[-1]["content"] == "Short question 2"
    
    # If it kept the tool messages, they must be BEFORE the final user message
    if "tool" in roles:
        assert roles.index("tool") < roles.index("user", 1)
        
@pytest.mark.asyncio
async def test_frequency_penalty_removed(agent):
    """
    Verifies that frequency_penalty is NOT injected into the llm_client payload.
    """
    body = {"messages": [{"role": "user", "content": "Execute a complex task"}], "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    
    call_args = agent.context.llm_client.chat_completion.call_args
    print("DEBUG CALL_ARGS:", call_args)
    payload = call_args.args[0]
    assert "frequency_penalty" not in payload
    assert "temperature" in payload

@pytest.mark.asyncio
async def test_dynamic_state_trailing_system(agent):
    """
    Verifies that when dynamic_state is present, it is appended as a trailing system message
    rather than being merged into the first system prompt, to preserve KV cache hits.
    """
    body = {"messages": [{"role": "system", "content": "BASE SYSTEM PROMPT"}, {"role": "user", "content": "Execute a complex task"}], "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    
    call_args = agent.context.llm_client.chat_completion.call_args
    messages = call_args.args[0]["messages"]
    
    # Base system prompt is at index 0 (overwritten by standard SYSTEM_PROMPT)
    assert messages[0]["role"] == "system"
    assert "You are Ghost" in messages[0]["content"]
    
    # Dynamic state should be in the LAST message (trailing user prompt)
    assert messages[-1]["role"] == "user"
    assert "DYNAMIC SYSTEM STATE" in messages[-1]["content"]
    
    # Ensure there is exactly ONE system prompt in the request (the base one)
    system_prompts = [m for m in messages if m["role"] == "system"]
    assert len(system_prompts) == 1

@pytest.mark.asyncio
async def test_loop_alerts_use_user_role(agent):
    """
    Verifies that system alerts injected during tool execution loops (like loop breakers)
    use the 'user' role instead of 'system' to comply with ChatML constraints.
    """
    tool_call_msg = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "t1", "function": {"name": "deep_research", "arguments": "{}"}}]}}]}
    final_msg = {"choices": [{"message": {"content": "Final Answer", "tool_calls": []}}]}
    
    # The limit for deep_research is 10. Call it 12 times.
    side_effects = [copy.deepcopy(tool_call_msg) for _ in range(12)] + [final_msg]
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=side_effects)
    agent.available_tools["deep_research"] = AsyncMock(return_value="results")
    
    body = {"messages": [{"role": "user", "content": "Search completely"}], "model": "Qwen-Test"}
    
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat(body, background_tasks=MagicMock())
        
        # We need to look at the arguments passed to the LLM on the final call 
        # (or any call after the limit is breached)
        last_call_args = agent.context.llm_client.chat_completion.call_args
        messages = last_call_args.args[0]["messages"]
        
        # Identify the loop breaker message
        loop_breaker_found = False
        loop_breaker_role = None
        for m in messages:
            if "SYSTEM ALERT: Tool 'deep_research' used too many times in a row" in str(m.get("content", "")):
                loop_breaker_found = True
                loop_breaker_role = m.get("role")
                break
                
        assert loop_breaker_found, "Loop breaker message was not injected"
        assert loop_breaker_role == "user", "Loop breaker alert must use 'user' role, not 'system'"
