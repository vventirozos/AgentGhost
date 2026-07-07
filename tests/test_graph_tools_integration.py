import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from ghost_agent.tools.memory import tool_remember, tool_recall
from ghost_agent.tools.system import tool_system_utility

@pytest.fixture
def mock_memory_system():
    mem_sys = MagicMock()
    mem_sys.add = MagicMock()
    # Mocking retrieval to return some chunks for generic hits
    mem_sys.search_advanced.return_value = [
        {"source": "VectorDB", "text": "Dummy fact", "score": 1.0}
    ]
    return mem_sys

@pytest.fixture
def mock_graph_memory():
    graph = MagicMock()
    graph.get_neighborhood.return_value = ["- (Alice) -[KNOWS]-> (Bob)", "- (Bob) -[LIKES]-> (Python)"]
    graph.add_triplets = MagicMock()
    return graph

@pytest.fixture
def mock_llm_client():
    llm = AsyncMock()
    # Mock LLM returning triplets JSON for remember extraction
    llm.chat_completion.return_value = {
        "choices": [
            {"message": {"content": '{"graph_triplets": [{"subject": "user", "predicate": "OWNS", "object": "repo"}]}'}}
        ]
    }
    return llm

@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.graph_memory = MagicMock()
    ctx.scheduler = MagicMock()
    ctx.scheduler.running = True
    ctx.scheduler.get_jobs.return_value = []
    # Mock SQLite call strictly
    ctx.graph_memory._lock = MagicMock()
    ctx.graph_memory._lock.__enter__ = MagicMock()
    ctx.graph_memory._lock.__exit__ = MagicMock()
    ctx.graph_memory.db_path = ":memory:"
    return ctx

@pytest.mark.asyncio
async def test_tool_remember_graph_extraction(mock_memory_system, mock_graph_memory, mock_llm_client):
    """Test that tool_remember offloads LLM-based triplet extraction into Graph Memory asynchronously."""
    
    from ghost_agent.utils import logging as _glog
    _glog._BG_TASKS.clear()
    with patch("ghost_agent.tools.memory.asyncio.to_thread") as mock_to_thread:
        # We need asyncio.to_thread patched to actually await logic right away so we can verify effects
        async def passthrough(func, *args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        mock_to_thread.side_effect = passthrough

        result = await tool_remember(
            text="The user owns the repo.",
            memory_system=mock_memory_system,
            graph_memory=mock_graph_memory,
            llm_client=mock_llm_client,
            model_name="test-model"
        )

        # Graph extraction is scheduled via spawn_bg → drain the registry.
        for t in list(_glog._BG_TASKS):
            await t

        assert "Memory stored: 'The user owns the repo.'" in result
        mock_memory_system.add.assert_called_once()
        mock_graph_memory.add_triplets.assert_called_once_with([{"subject": "user", "predicate": "OWNS", "object": "repo"}])

@pytest.mark.asyncio
async def test_tool_recall_neighborhood_injection(mock_memory_system, mock_graph_memory):
    """Test that tool_recall queries the Graph Memory and injects neighborhoods."""
    
    with patch("ghost_agent.tools.memory.asyncio.to_thread") as mock_to_thread:
        async def passthrough(func, *args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        mock_to_thread.side_effect = passthrough
        
        result = await tool_recall(
            query="Alice Python",
            memory_system=mock_memory_system,
            graph_memory=mock_graph_memory,
            top_k=5
        )
        
        assert "### TOPOLOGICAL GRAPH EDGES:" in result
        assert "- (Alice) -[KNOWS]-> (Bob)" in result
        assert "Dummy fact" in result

@pytest.mark.asyncio
async def test_system_utility_graph_healthcheck(mock_context):
    """Test that the system utility queries SQLite graph triplets count."""
    
    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = [42]
        
        result = await tool_system_utility(action="check_health", context=mock_context)
        
        assert "Graph=Active (42 edges)" in result

@pytest.mark.asyncio
async def test_tool_update_profile_graph_injection(mock_graph_memory):
    """Test that profile updates directly instantiate knowledge graph edges without LLM."""
    from ghost_agent.tools.memory import tool_update_profile
    
    with patch("ghost_agent.tools.memory.asyncio.to_thread") as mock_to_thread:
        async def passthrough(func, *args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        mock_to_thread.side_effect = passthrough
        
        # Test updating the profile
        mock_profile = MagicMock()
        mock_memory_system = MagicMock()
        
        result = await tool_update_profile(
            category="preferences",
            key="favorite language",
            value="Python",
            profile_memory=mock_profile,
            memory_system=mock_memory_system,
            graph_memory=mock_graph_memory
        )
        
        assert "SUCCESS: Profile updated." in result
        mock_profile.update.assert_called_once_with("preferences", "favorite language", "Python")
        mock_graph_memory.add_triplets.assert_called_once_with([{"subject": "user", "predicate": "HAS_FAVORITE_LANGUAGE", "object": "python"}])
