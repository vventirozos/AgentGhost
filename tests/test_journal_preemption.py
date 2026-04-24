import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from ghost_agent.core.agent import GhostAgent, GhostContext
import datetime
import asyncio

@pytest.fixture
def mock_context():
    context = MagicMock(spec=GhostContext)
    context.journal = MagicMock()
    context.args = MagicMock()
    context.llm_client = MagicMock()
    context.llm_client.vision_clients = None
    # Idle for 40 seconds (threshold is 30)
    context.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=40)
    return context

@pytest.mark.asyncio
async def test_process_journal_queue_completes(mock_context):
    agent = GhostAgent(context=mock_context)
    agent.run_smart_memory_task = AsyncMock()
    agent._execute_post_mortem = AsyncMock()
    
    mock_context.journal.pop_all.return_value = [
        {"type": "smart_memory", "data": {"text": "hello", "model": "test"}},
        {"type": "post_mortem", "data": {"user": "hi", "tools": [], "ai": "hello", "model": "test"}}
    ]
    
    await agent.process_journal_queue()
    
    assert agent.run_smart_memory_task.call_count == 1
    assert agent._execute_post_mortem.call_count == 1
    assert mock_context.journal.push_front.call_count == 0

@pytest.mark.asyncio
async def test_process_journal_queue_preemption(mock_context):
    agent = GhostAgent(context=mock_context)
    agent.run_smart_memory_task = AsyncMock()
    agent._execute_post_mortem = AsyncMock()
    
    # User interacts recently (less than 30 seconds ago)
    mock_context.last_activity_time = datetime.datetime.now()
    
    items = [
        {"type": "smart_memory", "data": {"text": "hello", "model": "test"}},
        {"type": "post_mortem", "data": {"user": "hi", "tools": [], "ai": "hello", "model": "test"}}
    ]
    mock_context.journal.pop_all.return_value = items
    
    await agent.process_journal_queue()
    
    # Operations should have been aborted
    assert agent.run_smart_memory_task.call_count == 0
    assert agent._execute_post_mortem.call_count == 0
    # Items should be pushed back to the journal
    mock_context.journal.push_front.assert_called_once_with(items)
