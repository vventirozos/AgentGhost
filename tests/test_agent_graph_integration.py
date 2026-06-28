import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def mock_agent():
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.5
    ctx.args.max_context = 4000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    
    ctx.llm_client = AsyncMock()
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all.return_value = ""
    ctx.memory_system = MagicMock()
    ctx.graph_memory = MagicMock()
    ctx.skill_memory = MagicMock()
    ctx.sandbox_dir = "/tmp/sandbox"
    return GhostAgent(ctx)

@pytest.mark.asyncio
async def test_agent_graph_traversal_integration(mock_agent):
    """Test that GraphMemory is queried and topologically injected during handle_chat."""
    agent = mock_agent
    
    messages = [
        {"role": "user", "content": "Tell me about (Neo's) relationship with the Matrix!"}
    ]
    
    # Return edges
    agent.context.graph_memory.get_neighborhood.return_value = [
        "- (Neo) -[HACKS]-> (Matrix)"
    ]
    
    agent.context.llm_client.chat_completion.return_value = {
        "choices": [{"message": {"content": "ok", "tool_calls": []}}]
    }
    
    # Run handle_chat ensuring memory_system is mocked normally
    with patch("ghost_agent.core.agent.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # We need mock_to_thread to return edges for get_neighborhood
        def mock_dispatch(func, *args, **kwargs):
            if func == agent.context.graph_memory.get_neighborhood:
                return ["- (Neo) -[HACKS]-> (Matrix)"]
            if func == agent.context.memory_system.search:
                return "Mocked Vector Memory"
            return ""
                
        mock_to_thread.side_effect = mock_dispatch
        
        await agent.handle_chat({"messages": messages, "model": "test"}, MagicMock())
        
        # Verify get_neighborhood was called with parsed punctuation-stripped words + 'user'
        call_args = [args for args in agent.context.graph_memory.get_neighborhood.call_args_list]
        called_words = mock_to_thread.call_args_list[2].args[1] if len(mock_to_thread.call_args_list) > 2 else []
        
        # We use a comprehensive patch for get_neighborhood to check the exact arguments
        neighborhood_call = next(c for c in mock_to_thread.call_args_list if c.args[0] == agent.context.graph_memory.get_neighborhood)
        extracted_words = neighborhood_call.args[1]
        
        assert "neo's" in extracted_words
        assert "matrix" in extracted_words
        # Redesign #3: graph retrieval no longer auto-seeds the term "user"
        # (that returned the user ego-graph on every turn regardless of topic).
        assert "user" not in extracted_words
        
        # Verify it was injected
        # Check SYSTEM prompt passed to the LLM
        llm_call = agent.context.llm_client.chat_completion.call_args
        messages_sent_to_llm = llm_call.args[0]["messages"]
        user_msg = next(m["content"] for m in reversed(messages_sent_to_llm) if m["role"] == "user")
        
        assert "TOPOLOGICAL KNOWLEDGE GRAPH" in user_msg
        assert "- (Neo) -[HACKS]-> (Matrix)" in user_msg

@pytest.mark.asyncio
async def test_agent_graph_ingestion_background_task(mock_agent):
    """Test that the run_smart_memory_task successfully extracts and ingests Graph triplets."""
    
    agent = mock_agent
    interaction_context = "User stated: I like to eat apples."
    
    agent.context.llm_client.chat_completion.return_value = {
        "choices": [{"message": {"content": '''{
            "score": 0.95,
            "fact": "User likes apples",
            "profile_update": {"category": "likes", "key": "food", "value": "apples"},
            "graph_triplets": [
                {"subject": "User", "predicate": "LIKES", "object": "apples"}
            ]
        }'''}}]
    }
    
    agent.context.graph_memory.add_triplets = MagicMock(return_value=1)
    
    # We execute the private _run_smart_memory_task directly
    with patch("ghost_agent.core.agent.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # Avoid crashing when computing added > 0
        def to_thread_side_effect(func, *args, **kwargs):
            if func == agent.context.graph_memory.add_triplets:
                return 1
            return None
        mock_to_thread.side_effect = to_thread_side_effect
        
        await agent.run_smart_memory_task(interaction_context, "test", 0.5)
        
        # Verify profile_memory was dispatched
        mock_to_thread.assert_any_call(
            agent.context.profile_memory.update,
            "likes", "food", "apples"
        )
        
        # Verify graph_memory was dispatched
        mock_to_thread.assert_any_call(
            agent.context.graph_memory.add_triplets,
            [{"subject": "User", "predicate": "LIKES", "object": "apples"}]
        )
