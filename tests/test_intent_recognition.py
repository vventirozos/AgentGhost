
import pytest
import re
from unittest.mock import MagicMock, patch, AsyncMock
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def mock_agent():
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.5
    ctx.args.max_context = 8000
    ctx.args.smart_memory = 0.0  # Explicitly set float to avoid TypeError
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all.return_value = "None."
    ctx.memory_system = MagicMock()
    ctx.memory_system.search.return_value = ""
    ctx.llm_client = MagicMock()
    ctx.cached_sandbox_state = None  # Explicitly set to avoid AttributeError
    ctx.sandbox_dir = "/tmp/sandbox"
    
    agent = GhostAgent(context=ctx)
    return agent

# §4UI: the "Ghost Specialist Activated" persona is gone; coding intent
# still picks the coding node / sampling / sandbox listing, so the
# classifier's word-boundary behaviour is pinned directly.
from ghost_agent.core.agent import detect_coding_intent


@pytest.mark.parametrize("msg,expected", [
    ("this is a short message.", False),          # "sh" inside "short"
    ("please write a python script to calculate pi.", True),
    ("i am working on mysql optimization.", False),
    ("please explain analyze this postgres query.", True),
    ("can you execute that plan?", False),        # "execute" alone
    ("can you write a bash script for this?", True),
    ("look at main.py", True),
])
def test_coding_intent_classifier(msg, expected):
    assert detect_coding_intent(msg)[0] is expected


@pytest.mark.asyncio
async def test_intent_sets_use_coding_flag(mock_agent):
    """Verify that handle_chat sets use_coding=True when coding intent is detected."""
    # Positive Test: "script" SHOULD trigger coding intent
    message = {"role": "user", "content": "Can you write a bash script for this?"}
    
    mock_agent.run_smart_memory_task = MagicMock()
    mock_agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Ok", "tool_calls": []}}]
    })
    
    await mock_agent.handle_chat({"messages": [message]}, MagicMock())
    
    # Assert that chat_completion was called with use_coding=True
    call_kwargs = mock_agent.context.llm_client.chat_completion.call_args.kwargs
    assert call_kwargs.get("use_coding") is True

@pytest.mark.asyncio
async def test_intent_sets_use_coding_flag_negative(mock_agent):
    """Verify that handle_chat sets use_coding=False when NO coding intent is detected."""
    # Negative Test: a conversational message that is NOT a trivial greeting
    # (otherwise the fast path intercepts it before use_coding is touched).
    message = {"role": "user", "content": "Tell me about the history of jazz music."}

    mock_agent.run_smart_memory_task = MagicMock()
    mock_agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Ok", "tool_calls": []}}]
    })

    await mock_agent.handle_chat({"messages": [message]}, MagicMock())

    # Assert that chat_completion was called with use_coding=False (or unset,
    # which is also acceptable since the default is False).
    call_kwargs = mock_agent.context.llm_client.chat_completion.call_args.kwargs
    assert call_kwargs.get("use_coding", False) is False

