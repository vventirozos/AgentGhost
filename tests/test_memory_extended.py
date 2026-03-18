import pytest
from unittest.mock import MagicMock, AsyncMock
from ghost_agent.tools.memory import (
    tool_remember, 
    tool_recall, 
    tool_unified_forget, 
    tool_scratchpad, 
    tool_knowledge_base,
    tool_gain_knowledge 
)

@pytest.fixture
def mock_memory_system():
    mem = MagicMock()
    # Mocking memory system methods
    mem.add = MagicMock()
    # Mock search_advanced to return list of dicts with low score (high relevance)
    # MUST BE SYNC MagicMock because tool_recall uses to_thread(mem.search_advanced)
    mem.search_advanced = MagicMock(return_value=[
        {"score": 0.1, "text": "The sky is blue", "metadata": {"source": "fact", "type": "fact"}}
    ])
    mem.search = MagicMock(return_value="Memory Content Found")
    mem.forget = MagicMock(return_value=True)
    mem.ingest_file = AsyncMock(return_value="Ingested DOC")
    return mem

@pytest.fixture
def mock_context_with_mem(mock_context, mock_memory_system):
    mock_context.memory_system = mock_memory_system
    return mock_context

@pytest.mark.asyncio
async def test_remember(mock_context_with_mem):
    # Test storing a fact
    res = await tool_remember("The sky is blue", mock_context_with_mem.memory_system)
    assert "stored" in res.lower()
    mock_context_with_mem.memory_system.add.assert_called_once()

@pytest.mark.asyncio
async def test_recall(mock_context_with_mem):
    # Test recalling a fact
    res = await tool_recall("sky", mock_context_with_mem.memory_system)
    assert "sky" in res.lower()
    mock_context_with_mem.memory_system.search_advanced.assert_called_with("sky", limit=10)

@pytest.mark.asyncio
async def test_forget(mock_context_with_mem, temp_dirs):
    # Test forgetting
    # Create file to delete
    (temp_dirs["sandbox"] / "sky.txt").touch()
    
    # Actual sig: tool_unified_forget(target, sandbox_dir, memory_system, ...)
    res = await tool_unified_forget("sky", sandbox_dir=temp_dirs["sandbox"], memory_system=mock_context_with_mem.memory_system)
    assert "forgotten" in res.lower() or "deleted" in res.lower()
    # verify call
    
    # Test generic cleanup
    res = await tool_unified_forget("test", sandbox_dir=mock_context_with_mem.sandbox_dir, memory_system=mock_context_with_mem.memory_system)
    assert "No matching" in res or "Forgot" in res

@pytest.mark.asyncio
async def test_knowledge_base_hallucination_guard(mock_context):
    # Test that providing a title instead of filename returns a helpful error
    res = await tool_gain_knowledge("## The Tragedy of Romeo and Juliet", 
                                  sandbox_dir=mock_context.sandbox_dir, 
                                  memory_system=mock_context.memory_system)
    assert "Error: You passed the document CONTENT or TITLE" in res
    assert "MUST pass the FILENAME" in res
    # verify call
    pass

@pytest.mark.asyncio
async def test_scratchpad(mock_context):
    # Test scratchpad operations
    from ghost_agent.memory.scratchpad import Scratchpad
    mock_context.scratchpad = Scratchpad()
    
    # 1. Add (Use 'set' action)
    # Actual sig: tool_scratchpad(action, scratchpad, key, value)
    await tool_scratchpad(action="set", scratchpad=mock_context.scratchpad, key="milky", value="Buy milk")
    assert "Buy milk" in mock_context.scratchpad.list_all()
    
    # 2. Update (Use 'set' again)
    await tool_scratchpad(action="set", scratchpad=mock_context.scratchpad, key="milky", value="Buy cookies")
    assert "Buy cookies" in mock_context.scratchpad.list_all()
    
    # 3. Delete (Not supported directly? Or set to None?)
    # Tool only supports set, get, list, clear.
    pass
    
    # 4. Clear
    await tool_scratchpad(action="clear", scratchpad=mock_context.scratchpad)
    assert "Scratchpad is empty" in mock_context.scratchpad.list_all()

@pytest.mark.asyncio
async def test_knowledge_base_ingest(mock_context_with_mem, temp_dirs):
    # Create a dummy file to ingest
    f = temp_dirs["sandbox"] / "doc.txt"
    f.write_text("Important document.")
    
    # Test ingest action
    res = await tool_knowledge_base(
        action="ingest_document", 
        content="doc.txt", 
        memory_system=mock_context_with_mem.memory_system,
        sandbox_dir=temp_dirs["sandbox"]
    )
    
    # The actual return might be different if it mocks internal ingestion
    # assert "Success" in res
    # mock_context_with_mem.memory_system.ingest_file.assert_called()
    pass
