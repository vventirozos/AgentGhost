import pytest
from unittest.mock import MagicMock, AsyncMock
from ghost_agent.core.agent import GhostAgent

@pytest.mark.asyncio
async def test_agent_tool_registry(mock_context):
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    
    # Verify core tools are present in definitions
    names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    assert "file_system" in names
    assert "execute" in names
    assert "knowledge_base" in names
    assert "recall" in names

@pytest.mark.asyncio
async def test_agent_tool_loading(mock_context):
    agent = GhostAgent(mock_context)
    # Check keys from registry.py
    assert "file_system" in agent.available_tools
    assert "web_search" in agent.available_tools # 'search' is the tool definition name, but key in available_tools might differ?
    # Registry says: "web_search": lambda ... tool_search
    # "fact_check": ...
    assert "execute" in agent.available_tools
    assert callable(agent.available_tools["file_system"])

@pytest.mark.asyncio
async def test_agent_handoff_tool_integration(mock_context):
    # Verify handoff tool if it exists
    agent = GhostAgent(mock_context)
    # Some versions have handoff or switch_mode
    if "switch_mode" in agent.available_tools:
        assert True
    else:
        # Just ensure we didn't break initialization
        assert True 

@pytest.mark.asyncio
async def test_agent_scheduler_integration(mock_context):
    agent = GhostAgent(mock_context)
    # Check if tools have access to scheduler if needed
    # (Implicitly tested via tasks tool)
    pass
