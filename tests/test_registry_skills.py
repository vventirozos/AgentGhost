import pytest
import json
from unittest.mock import MagicMock, patch

from ghost_agent.tools.registry import get_available_tools, get_active_tool_definitions

def test_get_available_tools_binds_acquired_skills(temp_dirs, mock_context):
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager
    
    # Pre-populate registry with a skill
    manager = AcquiredSkillManager(temp_dirs["sandbox"], MagicMock())
    manager.save_skill("test_dyn_skill", "dyn desc", "{}", "print('a')")
    
    tools = get_available_tools(mock_context)
    
    # Basic tools should be there
    assert "file_system" in tools
    
    # Acquired skill should be bound
    assert "test_dyn_skill" in tools
    assert callable(tools["test_dyn_skill"])

def test_get_active_tool_definitions_with_query(temp_dirs, mock_context):
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager
    
    manager = AcquiredSkillManager(temp_dirs["sandbox"], mock_context.memory_system)
    manager.save_skill("skill_one", "desc 1", "{}", "code")
    manager.save_skill("skill_two", "desc 2", "{}", "code")
    # Simulate a deleted skill not fully erased from DB yet or manual error
    manager.save_skill("skill_deleted", "desc 3", "{}", "code")
    manager.delete_skill("skill_deleted")
    
    # Mock RAG retrieval
    # Collection query returns top 2 results matching skill_two and skill_deleted
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "metadatas": [[{"name": "skill_two"}, {"name": "skill_deleted"}]]
    }
    mock_context.memory_system.collection = mock_collection
    
    with patch("ghost_agent.tools.registry.logger.info") as mock_logger_info, \
         patch("ghost_agent.tools.registry.pretty_log") as mock_pretty_log:
         
        # Retrieve with query
        definitions = get_active_tool_definitions(mock_context, query="find skill two")
        
        # Verify log format: should only load 1 active skill
        mock_logger_info.assert_called_with("Semantic Toolkit Router injected 1 acquired skills.")
        assert mock_pretty_log.call_args[0][0] == "Semantic Routing"
        assert mock_pretty_log.call_args[0][1] == "Loaded 1 skills."
    
    # Verify natural born tools are there (e.g., file_system)
    tool_names = [d["function"]["name"] for d in definitions]
    assert "file_system" in tool_names
    
    # Verify Semantic Tool Retrieval filtered the acquired skills
    assert "skill_two" in tool_names
    assert "skill_one" not in tool_names
    assert "skill_deleted" not in tool_names
    
    # Verify wrapper token
    for d in definitions:
        if d["function"]["name"] == "skill_two":
            assert "[ACQUIRED SKILL]" in d["function"]["description"]

def test_get_active_tool_definitions_without_query(temp_dirs, mock_context):
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager
    
    manager = AcquiredSkillManager(temp_dirs["sandbox"], mock_context.memory_system)
    manager.save_skill("skill_one", "desc 1", "{}", "code")
    
    definitions = get_active_tool_definitions(mock_context, query=None)
    
    tool_names = [d["function"]["name"] for d in definitions]
    assert "skill_one" in tool_names
