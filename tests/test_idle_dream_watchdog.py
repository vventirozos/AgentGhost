import pytest
import asyncio
import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.main import idle_dream_watchdog

@pytest.mark.asyncio
async def test_idle_dream_watchdog_no_context():
    """Test early return when context is missing."""
    with patch("ghost_agent.main.GLOBAL_CONTEXT", None):
        # Should not raise any error and return silently
        await idle_dream_watchdog()

@pytest.mark.asyncio
async def test_idle_dream_watchdog_not_idle_enough():
    """Test that dream doesn't trigger if less than 15 mins idle."""
    mock_context = MagicMock()
    mock_context.memory_system = MagicMock()
    mock_context.llm_client = MagicMock()
    mock_context.llm_client.foreground_tasks = 0
    
    # Only 5 minutes idle
    mock_context.last_activity_time = datetime.datetime.now() - datetime.timedelta(minutes=5)
    
    with patch("ghost_agent.main.GLOBAL_CONTEXT", mock_context):
        await idle_dream_watchdog()
        # Ensure memory_system.collection.get was not called
        assert not mock_context.memory_system.collection.get.called

@pytest.mark.asyncio
async def test_idle_dream_watchdog_triggers_dream():
    """Test that dream triggers given enough idle time and memory entropy."""
    mock_context = MagicMock()
    mock_context.memory_system = MagicMock()
    mock_context.llm_client = MagicMock()
    mock_context.llm_client.foreground_tasks = 0
    mock_context.last_activity_time = datetime.datetime.now() - datetime.timedelta(minutes=20)
    
    # Mock memory DB response with 3 items (enough entropy)
    mock_context.memory_system.collection.get.return_value = {"ids": ["1", "2", "3"]}
    
    async def mock_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)
    
    # Mock Dreamer
    mock_dreamer_instance = MagicMock()
    mock_dreamer_instance.dream = AsyncMock()
    MockDreamer = MagicMock(return_value=mock_dreamer_instance)
    
    with patch("ghost_agent.main.GLOBAL_CONTEXT", mock_context), \
         patch("ghost_agent.main.GLOBAL_AGENT", MagicMock()), \
         patch("asyncio.to_thread", side_effect=mock_to_thread), \
         patch("random.random", return_value=0.1), \
         patch("ghost_agent.core.dream.Dreamer", MockDreamer, create=True):
             
        await idle_dream_watchdog()
        
        # Verify collection.get was called to check entropy at least once
        # Using mock_to_thread means it actually executed the inner .get call
        mock_context.memory_system.collection.get.assert_called_with(where={"type": "auto"}, limit=5)
        
        # Verify Dreamer was initialized and dream was called
        MockDreamer.assert_called_with(mock_context)
        assert mock_dreamer_instance.dream.called
        
        # Verify activity time was reset
        assert (datetime.datetime.now() - mock_context.last_activity_time).total_seconds() < 5

@pytest.mark.asyncio
async def test_lifespan_adds_dream_job():
    """Test that lifespan adds the idle_dream_monitor job to the scheduler."""
    from ghost_agent.main import lifespan, idle_dream_watchdog
    
    mock_app = MagicMock()
    mock_app.state.args = MagicMock()
    mock_app.state.args.no_memory = True # Skip memory init to keep it fast
    mock_context = MagicMock()
    mock_app.state.context = mock_context
    
    with patch("ghost_agent.main.GLOBAL_CONTEXT", None), \
         patch("ghost_agent.main.GLOBAL_AGENT", None), \
         patch("ghost_agent.main.LLMClient") as MockLLMClient, \
         patch("ghost_agent.main.importlib.util.find_spec", return_value=False), \
         patch("ghost_agent.main.ProfileMemory"), \
         patch("ghost_agent.main.AsyncIOScheduler") as MockScheduler, \
         patch("ghost_agent.main.GhostAgent"), \
         patch("ghost_agent.main.SQLAlchemyJobStore"):
             
        mock_scheduler_instance = MagicMock()
        MockScheduler.return_value = mock_scheduler_instance
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.close = AsyncMock()
        MockLLMClient.return_value = mock_llm_instance
        
        # Test the lifespan context manager
        async with lifespan(mock_app):
            # Verify add_job was called correctly
            mock_scheduler_instance.add_job.assert_called_with(
                idle_dream_watchdog,
                'interval',
                minutes=1,
                id="idle_dream_monitor",
                replace_existing=True
            )
