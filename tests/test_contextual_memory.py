import pytest
import datetime
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.memory.vector import VectorMemory

@pytest.fixture
def mock_context():
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
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.skill_memory = None
    ctx.graph_memory = None
    return ctx

@pytest.mark.asyncio
async def test_time_decayed_salience():
    """Verify that newer memories rank higher than older ones with identical semantic distance."""
    with patch("ghost_agent.memory.vector.chromadb"):
        vm = VectorMemory(Path("/tmp"), "http://mock")
    
    vm.collection = MagicMock()
    
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    # Give the old time an age of 300 days -> penalty maxes out at 0.15
    old_time = (now - datetime.timedelta(days=300)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    new_time = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

    # Both have the same base semantic distance (0.10) and p_score (0 for manual)
    vm.collection.query.return_value = {
        'documents': [['Old Document', 'New Document']],
        'metadatas': [
            [{'type': 'manual', 'timestamp': old_time}, 
             {'type': 'manual', 'timestamp': new_time}]
        ],
        'distances': [[0.10, 0.10]],
        'ids': [['1', '2']]
    }

    results = vm.search("test query", inject_identity=False)
    lines = results.split('\n---\n')
    
    # Due to the time penalty, 'New Document' (0.10 + small penalty) 
    # scores lower than 'Old Document' (0.10 + 0.15). Lowest score wins.
    assert "New Document" in lines[0], "Time-decay failed: Old memory ranked higher!"
    assert "Old Document" in lines[1], "Time-decay failed: Missing old memory!"

@pytest.mark.asyncio
async def test_contextual_query_expansion(mock_context):
    """Verify short user queries are prepended with previous AI context for pronoun resolution."""
    agent = GhostAgent(mock_context)
    
    # Stop the loop early by returning a standard response without tool calls
    mock_context.llm_client.chat_completion.return_value = {
        "choices": [{"message": {"content": "ok", "tool_calls": []}}]
    }
    
    messages = [
        {"role": "user", "content": "Write a python script"},
        {"role": "assistant", "content": "I have created the calculation script."},
        {"role": "user", "content": "run it then"} # Very short, ambiguous query
    ]
    
    with patch("ghost_agent.core.agent.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.return_value = "Mocked Memory"
        await agent.handle_chat({"messages": messages, "model": "test"}, MagicMock())
        
        # Find the memory search call
        print("Mocked calls:", mock_to_thread.call_args_list)
        search_call = next(call for call in mock_to_thread.call_args_list if call.args[0] == mock_context.memory_system.search_items)
        actual_query = search_call.args[1]
        
        assert "Context: I have created the calculation script." in actual_query
        assert "User intent: run it then" in actual_query

@pytest.mark.asyncio
async def test_self_contained_command_is_not_contaminated(mock_context):
    """Regression: a short imperative that already names its subject (ids in
    backticks) must search on the RAW message — NOT get the previous reply
    prepended. Prepending the prior 'difference between projects' answer is
    what made a partial-failure delete re-answer the previous question."""
    agent = GhostAgent(mock_context)

    mock_context.llm_client.chat_completion.return_value = {
        "choices": [{"message": {"content": "ok", "tool_calls": []}}]
    }

    messages = [
        {"role": "user", "content": "what is the difference between the projects"},
        {"role": "assistant", "content": "Here's the comparison: d410 is research-heavy, e5c3 is experiment-heavy."},
        {"role": "user", "content": "delete `ecef207c0d4b` and `516217d294cc`"},
    ]

    with patch("ghost_agent.core.agent.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.return_value = "Mocked Memory"
        await agent.handle_chat({"messages": messages, "model": "test"}, MagicMock())

        search_call = next(call for call in mock_to_thread.call_args_list if call.args[0] == mock_context.memory_system.search_items)
        actual_query = search_call.args[1]

        # The prior comparison must NOT leak into the delete's search query.
        assert "Context:" not in actual_query
        assert "comparison" not in actual_query
        assert actual_query == "delete `ecef207c0d4b` and `516217d294cc`"

@pytest.mark.asyncio
async def test_episodic_archival(mock_context):
    """Verify that pruned context triggers an episodic archival into the vector DB."""
    agent = GhostAgent(mock_context)
    
    # Mock the summarization worker response
    mock_context.llm_client.chat_completion.return_value = {
        "choices": [{"message": {"content": "This is a summary of the dropped messages."}}]
    }
    
    messages = [{"role": "system", "content": "sys"}]
    for i in range(10):
        messages.append({"role": "user", "content": f"msg {i}" * 100})
        
    # Force estimate_tokens to return a huge number so pruning triggers
    with patch('ghost_agent.core.agent.estimate_tokens', return_value=5000):
        await agent._prune_context(messages, max_tokens=1000)
    
    # Give the background asyncio.create_task a moment to execute
    await asyncio.sleep(0.1) 
    
    # Verify memory_system.add was called
    mock_context.memory_system.add.assert_called_once()
    args, kwargs = mock_context.memory_system.add.call_args
    
    assert "EPISODIC ARCHIVE" in args[0]
    assert "This is a summary of the dropped messages." in args[0]
    assert args[1]["type"] == "episode"
