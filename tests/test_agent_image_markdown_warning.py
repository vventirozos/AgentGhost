import pytest
from unittest.mock import MagicMock, AsyncMock
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def mock_context():
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.max_context = 8000
    ctx.args.smart_memory = 0.0
    ctx.args.temperature = 0.0
    ctx.tor_proxy = None
    ctx.scheduler = None
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string = lambda: ""
    ctx.memory_system = MagicMock()
    ctx.skill_memory = MagicMock()
    ctx.sandbox_dir = "/tmp"
    
    # Mock LLM Client
    ctx.llm_client = MagicMock()
    
    return ctx

@pytest.mark.asyncio
async def test_hallucinated_image_markdown_stall(mock_context):
    agent = GhostAgent(mock_context)
    
    # The LLM generates text with an image markdown tag, but NO tool call
    llm_output = """<think>
    The user is complaining. I should use vision_analysis first.
    </think>
    I see you are complaining about ![Image](/api/download/gen_123.png). I will fix it.
    """
    
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": llm_output}}]
    })
    
    # Run the agent
    body = {"messages": [{"role": "user", "content": "This image is too artistic"}], "model": "test-model"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    
    # Verify the warning was appended to the messages
    last_msg = body["messages"][-1]
    assert "SYSTEM ALERT: You generated an image markdown tag" in last_msg.get("content", "")

