import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.core.prompts import DBA_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT, SYSTEM_PROMPT

@pytest.fixture
def mock_context():
    context = MagicMock(spec=GhostContext)
    context.args = MagicMock()
    context.args.temperature = 0.7
    context.args.max_context = 4096
    context.args.use_planning = False
    context.args.smart_memory = 0.0
    context.sandbox_dir = "/tmp/sandbox"
    context.memory_dir = "/tmp/memory"
    context.tor_proxy = None
    context.llm_client = AsyncMock()
    context.llm_client.chat_completion.return_value = {
        "choices": [{"message": {"content": "Test response", "tool_calls": []}}]
    }
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    context.memory_system = MagicMock()
    context.skill_memory = MagicMock()
    context.cached_sandbox_state = None
    return context

@pytest.mark.asyncio
async def test_dba_persona_activation(mock_context):
    """Test that DBA persona is activated when Postgres keywords are present."""
    agent = GhostAgent(mock_context)
    agent.available_tools = {}
    
    # Simulate a user request with DBA keywords
    body = {
        "messages": [{"role": "user", "content": "Can you optimize this slow SQL query using EXPLAIN ANALYZE?"}],
        "model": "gpt-4"
    }
    
    background_tasks = MagicMock()
    
    await agent.handle_chat(body, background_tasks)
    
    # Check the system prompt and transient injection sent to the LLM
    call_args = mock_context.llm_client.chat_completion.call_args
    assert call_args is not None
    payload = call_args[0][0]
    messages = payload["messages"]
    
    # Base system prompt at index 0
    assert messages[0]["role"] == "system"
    assert "You are Ghost" in messages[0]["content"]
    
    # Specialized persona should be in the LAST message (appended)
    last_msg = messages[-1]
    assert "Ghost Principal PostgreSQL Administrator" in last_msg["content"]
    assert "DBA ENGINEERING STANDARDS" in last_msg["content"]

@pytest.mark.asyncio
async def test_python_persona_activation(mock_context):
    """Test that Python persona is activated for coding requests without DBA intent."""
    agent = GhostAgent(mock_context)
    agent.available_tools = {}
    
    body = {
        "messages": [{"role": "user", "content": "Write a Python script to sort a list."}],
        "model": "gpt-4"
    }
    
    background_tasks = MagicMock()
    
    await agent.handle_chat(body, background_tasks)
    
    call_args = mock_context.llm_client.chat_completion.call_args
    payload = call_args[0][0]
    messages = payload["messages"]
    
    # Specialized persona should be in the LAST message (appended)
    last_msg = messages[-1]
    assert "Ghost Advanced Engineering Subsystem" in last_msg["content"]
    assert "Ghost Principal PostgreSQL Administrator" not in last_msg["content"]

@pytest.mark.asyncio
async def test_default_persona_activation(mock_context):
    """Test that default persona is activated for general requests."""
    agent = GhostAgent(mock_context)
    agent.available_tools = {}
    
    body = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "model": "gpt-4"
    }
    
    background_tasks = MagicMock()
    
    await agent.handle_chat(body, background_tasks)
    
    call_args = mock_context.llm_client.chat_completion.call_args
    payload = call_args[0][0]
    messages = payload["messages"]
    
    # Base system prompt at index 0
    assert messages[0]["role"] == "system"
    assert "You are Ghost" in messages[0]["content"]
    
    # Specialized persona should be in the LAST message (appended)
    last_msg = messages[-1]
    assert "Ghost Principal PostgreSQL Administrator" not in last_msg["content"]
    assert "Ghost Advanced Engineering Subsystem" not in last_msg["content"]
    assert "### DYNAMIC SYSTEM STATE" in last_msg["content"]
