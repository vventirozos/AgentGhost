
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
    ctx.skill_memory = MagicMock()
    ctx.llm_client = MagicMock()
    ctx.cached_sandbox_state = None  # Explicitly set to avoid AttributeError
    ctx.sandbox_dir = "/tmp/sandbox"
    
    agent = GhostAgent(context=ctx)
    return agent

@patch("ghost_agent.core.agent.pretty_log")
@pytest.mark.asyncio
async def test_intent_python_specialist(mock_pretty_log, mock_agent):
    # Test strict word boundary for "sh" (should NOT trigger for "short")
    message = {"role": "user", "content": "This is a short message."}
    
    # We mock run_smart_memory_task to avoid bg tasks
    mock_agent.run_smart_memory_task = MagicMock()
    
    # Use AsyncMock for async method
    mock_agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Ok", "tool_calls": []}}]
    })
    
    # 1. Negative Test: "short" should NOT trigger Python mode
    #    If it triggered, "Ghost Python Specialist Activated" would be logged.
    await mock_agent.handle_chat({"messages": [message]}, MagicMock())
    
    # Verify NO Python Specialist log
    logs = [call.args for call in mock_pretty_log.call_args_list]
    python_mode_logs = [l for l in logs if len(l) > 1 and "Ghost Python Specialist Activated" in str(l[1])]
    assert len(python_mode_logs) == 0, f"False positive: 'short' triggered Python mode! Logs: {python_mode_logs}"

    # 2. Positive Test: "run python script" SHOULD trigger
    message2 = {"role": "user", "content": "Please write a python script to calculate pi."}
    await mock_agent.handle_chat({"messages": [message2]}, MagicMock())
    
    logs2 = [call.args for call in mock_pretty_log.call_args_list]
    python_mode_logs2 = [l for l in logs2 if len(l) > 1 and "Ghost Python Specialist Activated" in str(l[1])]
    assert len(python_mode_logs2) > 0, "Failed to detect 'python script' intent!"

@patch("ghost_agent.core.agent.pretty_log")
@pytest.mark.asyncio
async def test_intent_dba_routing(mock_pretty_log, mock_agent):
    # Negative Test: "mysql" shouldn't trigger naive substring matching
    
    message = {"role": "user", "content": "I am working on mysql optimization."}
    mock_agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Ok", "tool_calls": []}}]
    })
    
    await mock_agent.handle_chat({"messages": [message]}, MagicMock())
    
    logs = [call.args for call in mock_pretty_log.call_args_list]
    dba_logs = [l for l in logs if len(l) > 1 and "Ghost PostgreSQL DBA Activated" in str(l[1])]
    
    assert len(dba_logs) == 0

    # Positive Test: "optimize sql query"
    message2 = {"role": "user", "content": "Please explain analyze this postgres query."}
    await mock_agent.handle_chat({"messages": [message2]}, MagicMock())
    
    logs2 = [call.args for call in mock_pretty_log.call_args_list]
    dba_logs2 = [l for l in logs2 if len(l) > 1 and "Ghost PostgreSQL DBA Activated" in str(l[1])]
    assert len(dba_logs2) > 0


@patch("ghost_agent.core.agent.pretty_log")
@pytest.mark.asyncio
async def test_intent_coding_fallback_execute(mock_pretty_log, mock_agent):
    # Negative Test: "execute" alone should NOT trigger coding intent anymore
    message = {"role": "user", "content": "Can you execute that plan?"}
    
    # Mocking internal task
    mock_agent.run_smart_memory_task = MagicMock()
    mock_agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Ok", "tool_calls": []}}]
    })
    
    await mock_agent.handle_chat({"messages": [message]}, MagicMock())
    
    logs = [call.args for call in mock_pretty_log.call_args_list]
    python_mode_logs = [l for l in logs if len(l) > 1 and "Ghost Python Specialist Activated" in str(l[1])]
    
    assert len(python_mode_logs) == 0, "Failed: 'execute' incorrectly activated Python Specialist!"

    # Positive Test: "script" SHOULD still trigger it
    message2 = {"role": "user", "content": "Can you write a bash script for this?"}
    await mock_agent.handle_chat({"messages": [message2]}, MagicMock())
    
    logs2 = [call.args for call in mock_pretty_log.call_args_list]
    python_mode_logs2 = [l for l in logs2 if len(l) > 1 and "Ghost Python Specialist Activated" in str(l[1])]
    assert len(python_mode_logs2) > 0, "Failed: 'script' did not activate Python Specialist!"
    
    # Positive Test: ".py" SHOULD still trigger it
    message3 = {"role": "user", "content": "Look at main.py"}
    await mock_agent.handle_chat({"messages": [message3]}, MagicMock())
    
    logs3 = [call.args for call in mock_pretty_log.call_args_list]
    python_mode_logs3 = [l for l in logs3 if len(l) > 1 and "Ghost Python Specialist Activated" in str(l[1])]
    assert len(python_mode_logs3) > 0, "Failed: '.py' did not activate Python Specialist!"

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
    # Negative Test: conversational message
    message = {"role": "user", "content": "Hello there."}
    
    mock_agent.run_smart_memory_task = MagicMock()
    mock_agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Ok", "tool_calls": []}}]
    })
    
    await mock_agent.handle_chat({"messages": [message]}, MagicMock())
    
    # Assert that chat_completion was called with use_coding=False
    call_kwargs = mock_agent.context.llm_client.chat_completion.call_args.kwargs
    assert call_kwargs.get("use_coding") is False

