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
async def test_agent_graph_owner_facts_integration(mock_agent):
    """The OWNER's facts are read from the graph and injected during
    handle_chat; the topical neighbourhood is not (§4NC/§4NM)."""
    agent = mock_agent
    messages = [{"role": "user", "content": "where do I live these days?"}]
    agent.context.llm_client.chat_completion.return_value = {
        "choices": [{"message": {"content": "ok", "tool_calls": []}}]
    }
    with patch("ghost_agent.core.agent.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        def mock_dispatch(func, *args, **kwargs):
            if func == agent.context.graph_memory.owner_facts_matching:
                return ["- (User) -[LIVES_IN]-> (Athens)"]
            return ""
        mock_to_thread.side_effect = mock_dispatch
        with patch("ghost_agent.core.bus.asyncio.to_thread", new=mock_to_thread):
            await agent.handle_chat({"messages": messages, "model": "test"}, MagicMock())

        assert not any(c.args and c.args[0] == agent.context.graph_memory.get_neighborhood
                       for c in mock_to_thread.call_args_list)
        llm_call = agent.context.llm_client.chat_completion.call_args
        messages_sent_to_llm = llm_call.args[0]["messages"]
        user_msg = next(m["content"] for m in reversed(messages_sent_to_llm) if m["role"] == "user")
        assert "TOPOLOGICAL KNOWLEDGE GRAPH" in user_msg
        assert "- (User) -[LIVES_IN]-> (Athens)" in user_msg

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
        # §4QM: the extractor passes the ledger source (the owner's clause) as a kwarg
        # Review 2026-10-31: the extractor's free-form category folds onto the
        # standard drawer ("likes" → preferences), so a fact gets one home
        assert any(c.args[:4] == (agent.context.profile_memory.update, "preferences", "food", "apples")
                   and c.kwargs.get("source", {}).get("via") == "extractor"
                   for c in mock_to_thread.call_args_list), mock_to_thread.call_args_list
        
        # Verify graph_memory was dispatched — §4QM r2: with the ledger source
        # (the drain's request id and the owner's turn for the per-edge clause)
        assert any(c.args[:2] == (agent.context.graph_memory.add_triplets,
                                  [{"subject": "User", "predicate": "LIKES", "object": "apples"}])
                   and c.kwargs.get("source", {}).get("via") == "extractor"
                   and "episode" in c.kwargs.get("source", {})
                   for c in mock_to_thread.call_args_list), mock_to_thread.call_args_list
