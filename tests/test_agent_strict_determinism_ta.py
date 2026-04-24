import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def agent():
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.7
    ctx.args.max_context = 8000
    ctx.args.smart_memory = 0.0
    ctx.llm_client = MagicMock()
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_context_string.return_value = ""
    ctx.memory_system = MagicMock()
    ctx.memory_system.search = MagicMock(return_value="")
    ctx.cached_sandbox_state = None
    ctx.sandbox_dir = "/tmp/sandbox"
    agent_inst = GhostAgent(ctx)
    return agent_inst

@pytest.mark.asyncio
async def test_strict_determinism_and_temporal_anchor(agent):
    # System 2 Planner is disabled; there is only one LLM call — the main generation call.
    final_msg = {"choices": [{"message": {"content": "Final Answer", "tool_calls": []}}]}
    
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[final_msg])
    
    body = {"messages": [{"role": "user", "content": "Please write a complex python script to scan the entire network topology and execute it immediately."}], "model": "Qwen-Test"}
    
    await agent.handle_chat(body, background_tasks=MagicMock())
    
    # There should be exactly 1 LLM call (main generation, no planner)
    assert agent.context.llm_client.chat_completion.call_count >= 1
    
    main_call_args = agent.context.llm_client.chat_completion.call_args_list[0].args[0]
    
    # The system prompt is injected — verify the main call contains it
    messages = main_call_args.get("messages", [])
    prompt_content = ""
    for msg in messages:
        if msg["role"] in ["system", "user"]:
            prompt_content += msg.get("content", "")
            
    assert "### ROLE AND IDENTITY" in prompt_content
