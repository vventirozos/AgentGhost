import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from unittest.mock import AsyncMock, patch, MagicMock
from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.memory.vector import VectorMemory
import datetime

@pytest.fixture
def mock_agent():
    context = MagicMock(spec=GhostContext)
    context.args = MagicMock()
    context.args.max_context = 4000
    context.args.smart_memory = 0.5
    context.args.use_planning = False
    context.memory_system = AsyncMock()
    context.llm_client = AsyncMock()
    context.sandbox_dir = MagicMock()
    context.scratchpad = None
    context.profile_memory = None
    context.graph_memory = None
    context.skill_memory = None
    context.memory_bus = None
    context.sandbox_manager = None
    context.biological_task = None
    context.cached_sandbox_state = None
    context.last_activity_time = datetime.datetime.now()
    agent = GhostAgent(context)
    return agent

@pytest.fixture
def mock_vector_memory(tmp_path):
    mem_dir = tmp_path / "memory_db"
    mem_dir.mkdir()
    # We pass a fake upstream url
    mem = VectorMemory(mem_dir, "http://mock-url")
    return mem

@pytest.mark.asyncio
async def test_agent_episodic_archival_logic(mock_agent):
    """Test that _prune_context ingests summaries into the memory system as 'episode'"""
    
    # Mock llm_client to return a valid summary
    mock_agent.context.llm_client.chat_completion.return_value = {
        "choices": [{"message": {"content": "This is a summary of the past."}}]
    }
    
    messages = [
        {"role": "system", "content": "You are AI."},
        {"role": "user", "content": "Goal"},
        {"role": "assistant", "content": "Turn 1"},
        {"role": "user", "content": "Turn 2"},
        {"role": "assistant", "content": "Turn 3"},
        {"role": "user", "content": "Turn 4"},
        {"role": "assistant", "content": "Turn 5"},
        {"role": "user", "content": "Turn 6"},
        {"role": "assistant", "content": "Turn 7"},
        {"role": "user", "content": "Turn 8"},
    ]
    
    # Prune should condense the middle messages into a summary
    # We pass a tiny max_tokens to force pruning
    result = await mock_agent._prune_context(messages, max_tokens=10, model="mock")
    
    # Check that a background task was created to add the memory summary
    # Because asyncio.create_task is used, we need to allow the event loop a tick
    await asyncio.sleep(0.01)
    
    # Check if context.memory_system.add was called
    mock_agent.context.memory_system.add.assert_called_once()
    
    call_args = mock_agent.context.memory_system.add.call_args[0]
    expected_text = "EPISODIC ARCHIVE (Past Conversation Summary):\n[SYSTEM: PREVIOUS TURNS SUMMARIZED]\n\nThis is a summary of the past."
    
    assert call_args[0] == expected_text
    assert call_args[1]["type"] == "episode"
    assert "timestamp" in call_args[1]

def test_vector_memory_episode_and_decay(mock_vector_memory):
    """Test episode scoring and Time-Decayed Salience mathematical calculations"""
    
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    # Give exact control over what chroma returns
    
    old_time = now - datetime.timedelta(days=10)
    old_time_str = old_time.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    
    new_time = now - datetime.timedelta(days=1)
    new_time_str = new_time.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    
    with patch.object(mock_vector_memory.collection, "query") as mock_query:
        mock_query.return_value = {
            'documents': [[
                "Old manual query (-10 points normally)", 
                "New episode query (-12 points normally)", 
                "New manual query (-10 points normally)"
            ]],
            'metadatas': [[
                {"type": "manual", "timestamp": old_time_str},
                {"type": "episode", "timestamp": new_time_str},
                {"type": "manual", "timestamp": new_time_str}
            ]],
            'distances': [[0.2, 0.3, 0.4]],
            'ids': [['1', '2', '3']]
        }
        
        results = mock_vector_memory.search("Test", inject_identity=False)
        
        # Let's hand-calculate the expected scores
        # 1. document: Old manual -> p_score=0 normally? Wait, threshold logic:
        # m_type == manual. distance 0.2 < threshold (0.65). priority_score = 0.
        # dist = 0.2. age_days = 10 -> time_penalty = min(10 * 0.002, 0.15) = 0.02
        # combined_score = (0 * 10) + 0.2 + 0.02 = 0.22
        
        # 2. document: New episode -> p_score: threshold = 0.70. dist 0.3 < 0.70 -> priority_score = -12.
        # dist = 0.3. age_days = 1 -> time_penalty = min(1 * 0.002, 0.15) = 0.002
        # combined_score = (-12 * 10) + 0.3 + 0.002 = -119.698
        
        # 3. document: New manual -> p_score = 0.
        # dist = 0.4. age_days = 1 -> time_penalty = 0.002
        # combined_score = (0 * 10) + 0.4 + 0.002 = 0.402
        
        # Order should be: New episode (1s) -> Old manual (2nd) -> New manual (3rd)
        # The output string format is "[timestamp] (TYPE) **[Prefix]** Doc"
        
        lines = results.split("\n---\n")
        assert len(lines) == 3
        assert "New episode query" in lines[0] # Lowest score, highest rank
        assert "EPISODE" in lines[0].upper()
        assert "Old manual query" in lines[1]
        assert "New manual query" in lines[2]

@pytest.mark.asyncio
async def test_agent_query_expansion_and_bypass(mock_agent):
    """Test should_fetch_memory bypass and Contextual Query Expansion"""
    
    with patch.object(mock_agent, "process_rolling_window", side_effect=lambda msgs, limit: msgs), \
         patch.object(mock_agent, "_prune_context", new_callable=AsyncMock, side_effect=lambda msgs, **kwargs: msgs), \
         patch("ghost_agent.core.agent.GhostAgent._prepare_planning_context", side_effect=Exception("BreakLoop")):
        
        # The MemoryBus now hydrates via the per-item `search_items` entry
        # point (2026-07-07). It's still the vector-search call the expanded
        # query flows into, so it remains the marker for this test.
        mock_agent.context.memory_system.search_items = MagicMock(return_value=[])

        # Scenario 1: Short user text, should expand query
        messages = [
            {"role": "user", "content": "How do I do X?"},
            {"role": "assistant", "content": "You can do X by calling the function run_x(). It is a very cool AI command and does magic."},
            {"role": "user", "content": "run it then"} # 3 words
        ]

        body = {"messages": messages, "model": "mock", "stream": False}

        try:
            await mock_agent.handle_chat(body, {})
        except Exception as e:
            if str(e) != "BreakLoop" and "BreakLoop" not in str(e):
                 print(f"DEBUG EXCEPTION IN HANDLE_CHAT 1: {repr(e)}")
                 pass

        # Verify that should_fetch_memory fired and expanded the query
        mock_agent.context.memory_system.search_items.assert_called()
        called_query = mock_agent.context.memory_system.search_items.call_args[0][0]

        assert "Context: You can do X by calling the function run_x(). It is a very cool AI command and does magic." in called_query
        assert "User intent: run it then" in called_query

        # Scenario 2: Fact-check should skip memory fetch entirely
        mock_agent.context.memory_system.search_items.reset_mock()
        messages_fact = [
            {"role": "user", "content": "Please verify and fact-check this claim."}
        ]
        body_fact = {"messages": messages_fact, "model": "mock", "stream": False}
        try:
            await mock_agent.handle_chat(body_fact, {})
        except Exception as e:
            print(f"DEBUG EXCEPTION IN HANDLE_CHAT 2: {repr(e)}")
            pass
            
        mock_agent.context.memory_system.search_items.assert_not_called()
