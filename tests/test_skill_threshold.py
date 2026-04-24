
import pytest
from unittest.mock import MagicMock
from ghost_agent.memory.skills import SkillMemory

@pytest.fixture
def mock_skill_memory():
    # Mock dependencies
    memory_dir = MagicMock()
    memory_dir.exists.return_value = True
    
    # Patch __init__ to avoid file operations during setup
    with pytest.MonkeyPatch.context() as m:
        m.setattr("ghost_agent.memory.skills.SkillMemory.__init__", lambda self, d: None)
        sm = SkillMemory(memory_dir)
        sm.file_path = MagicMock()
        return sm

def test_get_playbook_context_threshold_filtering(mock_skill_memory):
    # Setup mock memory system
    mock_memory_system = MagicMock()
    mock_memory_system.collection = MagicMock()
    
    # Scenario: 3 results. 
    # 1. High relevance (dist 0.2) -> Should be included
    # 2. Borderline/Low relevance (dist 0.7) -> Should be EXCLUDED (threshold ~0.65)
    # 3. Irrelevant (dist 1.2) -> Should be EXCLUDED
    
    mock_memory_system.collection.query.return_value = {
        'documents': [['Relevant Skill', 'Borderline Skill', 'Irrelevant Skill']],
        'distances': [[0.2, 0.7, 1.2]]
    }
    
    # Run retrieval
    context = mock_skill_memory.get_playbook_context(query="help", memory_system=mock_memory_system)
    
    # Verify
    assert "Relevant Skill" in context
    assert "Borderline Skill" not in context
    assert "Irrelevant Skill" not in context

def test_get_playbook_context_no_match_returns_empty(mock_skill_memory):
    # Setup mock memory system
    mock_memory_system = MagicMock()
    mock_memory_system.collection = MagicMock()
    
    # Scenario: All results define threshold
    mock_memory_system.collection.query.return_value = {
        'documents': [['Bad Skill 1', 'Bad Skill 2']],
        'distances': [[0.8, 0.9]]
    }
    
    # Setup fallback to ensure it DOES NOT fallback if vector search happened but failed threshold
    # The requirement is: "If no relevant skills are found, return an empty string rather than falling back to the recent lessons list."
    # We strip the fallback logic if vector search was attempted.
    
    # To test this, we mock the file_path to return something for the fallback, and ensure it's NOT used.
    mock_skill_memory.file_path.read_text.return_value = '[{"task": "Fallback Task", "mistake": "M", "solution": "S"}]'
    
    # Run retrieval
    context = mock_skill_memory.get_playbook_context(query="help", memory_system=mock_memory_system)
    
    # Verify
    assert context == "", f"Expected empty context, got: {context}"
