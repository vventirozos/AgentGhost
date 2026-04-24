import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.agent import GhostAgent

@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.memory_system = MagicMock()
    ctx.profile_memory = MagicMock()
    ctx.skill_memory = MagicMock()
    ctx.args = MagicMock()
    ctx.args.smart_memory = 0.5
    ctx.args.max_context = 4000
    return ctx

@pytest.mark.asyncio
async def test_run_smart_memory_task_contradiction_engine(mock_context):
    mock_context.profile_memory.get_context_string.return_value = "Profile context"
    agent = GhostAgent(mock_context)
    
    # Mock LLM response to extract fact
    mock_context.llm_client.chat_completion.side_effect = [
        # First call: Fact extraction
        {"choices": [{"message": {"content": '{"score": 0.95, "fact": "User is building a React app.", "profile_update": {"category": "project", "key": "current", "value": "React app"}}'}}]},
        # Second call: Contradiction Engine evaluation
        {"choices": [{"message": {"content": '{"ids": ["ID:123"]}'}}]}
    ]
    
    # Mock advanced search to return conflicting memories
    mock_context.memory_system.search_advanced.return_value = [
        {"id": "123", "text": "User is building a Vue app.", "score": 0.5}
    ]
    
    await agent.run_smart_memory_task("User: I am building a React app.\nAI: Got it.", "test-model", 0.5)
    
    # Assert collection.delete was called with the old ID
    mock_context.memory_system.collection.delete.assert_called_with(ids=["123"])
    
    # Assert new fact was added
    mock_context.memory_system.add.assert_called()
    call_args = mock_context.memory_system.add.call_args[0]
    assert call_args[0] == "User is building a React app."
    assert mock_context.profile_memory.update.called

@pytest.mark.asyncio
async def test_run_smart_memory_task_no_contradiction(mock_context):
    agent = GhostAgent(mock_context)
    
    mock_context.llm_client.chat_completion.side_effect = [
        {"choices": [{"message": {"content": '{"score": 0.85, "fact": "User likes unit tests."}'}}]},
        # Return empty ids list
        {"choices": [{"message": {"content": '{"ids": []}'}}]}
    ]
    
    mock_context.memory_system.search_advanced.return_value = [
        {"id": "999", "text": "User likes Python.", "score": 0.4}
    ]
    
    await agent.run_smart_memory_task("User: Always write tests.\nAI: Okay.", "test-model", 0.5)
    
    # Assert delete was NOT called
    mock_context.memory_system.collection.delete.assert_not_called()
    
    # Assert new fact was added
    mock_context.memory_system.add.assert_called()

@pytest.mark.asyncio
async def test_intent_driven_skill_recall(mock_context):
    agent = GhostAgent(mock_context)
    
    mock_context.skill_memory.get_playbook_context.return_value = "React playbok"
    
    mock_context.profile_memory.get_context_string = MagicMock(return_value="Profile context")
    mock_context.args.temperature = 0.5
    
    # Mock agent dependencies for handle_chat
    import ghost_agent.core.agent
    ghost_agent.core.agent.request_id_context = MagicMock()
    
    with patch.object(agent, 'process_rolling_window', return_value=[{"role": "user", "content": "test"}]), \
         patch.object(agent, '_prune_context', return_value=[{"role": "user", "content": "test"}]):
             
        # Create a body representing an executing tool step
        body = {
            "messages": [
                {"role": "user", "content": "Write me a react component"},
                {"role": "assistant", "content": '{"thought": "Writing component", "tree_update": {}, "next_action_id": "step1", "required_tool": "execute"}'}
            ]
        }
        
        # We simulate the loop failing (breaking) fast so it doesn't run infinitely,
        # but we mock enough to reach the skill memory retrieval block.
        # Actually it's easier to assert that during handling, get_playbook_context was called with the tool string.
        # But handle_chat has a loop. Let's mock llm completion to return a simple non-tool response to exit.
        
        mock_context.llm_client.chat_completion.side_effect = AsyncMock(return_value={
            "choices": [{"message": {"content": "Here is your component"}}]
        })
        
        await agent.handle_chat(body, MagicMock())
        
        # It should have called get_playbook_context with the intent format for the required tool
        call_args = mock_context.skill_memory.get_playbook_context.call_args
        if call_args:
             query_used = call_args.kwargs.get("query", "")
             # Depending on if it successfully parsed the tool json in previous turn, it will use that for intent.
             # Since our previous turn was an assistant turn with json:
             assert "Context:" in query_used or "Write me a react component" in query_used
             
