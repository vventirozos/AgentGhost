import pytest
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from ghost_agent.tools.acquired_skills import AcquiredSkillManager, tool_create_skill

def test_acquired_skill_manager_init(temp_dirs):
    sandbox_dir = temp_dirs["sandbox"]
    memory_system = MagicMock()
    
    manager = AcquiredSkillManager(sandbox_dir, memory_system)
    
    # Check directory created
    skills_dir = sandbox_dir / "acquired_skills"
    assert skills_dir.exists()
    assert skills_dir.is_dir()
    
    # Check registry created
    registry_file = skills_dir / "skills_registry.json"
    assert registry_file.exists()
    
    with open(registry_file, "r") as f:
        data = json.load(f)
        assert data == {}

def test_acquired_skill_manager_save_and_get(temp_dirs):
    sandbox_dir = temp_dirs["sandbox"]
    memory_system = MagicMock()
    
    manager = AcquiredSkillManager(sandbox_dir, memory_system)
    
    # Save a skill
    manager.save_skill(
        name="test_skill",
        description="A test skill",
        parameters_schema={"type": "object", "properties": {"a": {"type": "string"}}},
        python_code="def run(a):\n    return f'Hello {a}'\n"
    )
    
    # Check physical file
    skill_file = sandbox_dir / "acquired_skills" / "test_skill.py"
    assert skill_file.exists()
    with open(skill_file, "r") as f:
        assert "def run(a):" in f.read()
        
    # Check memory system was called to embed
    memory_system.add.assert_called_once()
    assert memory_system.add.call_args[0][0] == "A test skill"
    
    # Get all skills
    skills = manager.get_all_skills()
    assert "test_skill" in skills
    assert skills["test_skill"]["status"] == "active"
    assert skills["test_skill"]["usage_count"] == 0

def test_acquired_skill_manager_telemetry(temp_dirs):
    sandbox_dir = temp_dirs["sandbox"]
    manager = AcquiredSkillManager(sandbox_dir, MagicMock())
    
    manager.save_skill("fail_skill", "desc", {}, "code")
    
    # Log successes
    manager.log_telemetry("fail_skill", success=True)
    assert manager.get_all_skills()["fail_skill"]["usage_count"] == 1
    assert manager.get_all_skills()["fail_skill"]["failure_count"] == 0
    
    # Log failures to trigger degradation
    for _ in range(5):
        manager.log_telemetry("fail_skill", success=False)
        
    assert manager.get_all_skills()["fail_skill"]["status"] == "degraded"

@pytest.mark.asyncio
async def test_tool_create_skill_success(temp_dirs):
    sandbox_dir = temp_dirs["sandbox"]
    memory_system = MagicMock()
    sandbox_manager = MagicMock()
    
    # Mock tool_execute to simulate successful test
    with patch("ghost_agent.tools.execute.tool_execute", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "EXIT CODE: 0\nSuccess output"
        
        # Test requires parameters_schema and test_payload to be valid JSON strings, not dicts
        result = await tool_create_skill(
            sandbox_dir=sandbox_dir,
            memory_system=memory_system,
            sandbox_manager=sandbox_manager,
            name="new_skill",
            description="new desc",
            parameters_schema='{"type": "object"}',
            python_code="print('works')",
            test_payload='{"test": "data"}'
        )
        
        assert "Success: Skill 'new_skill' acquired" in result
        
        # Check that test_skill.py was cleanup up
        test_file = sandbox_dir / "test_skill.py"
        assert not test_file.exists()
        
        # Check save
        skill_file = sandbox_dir / "acquired_skills" / "new_skill.py"
        assert skill_file.exists()

@pytest.mark.asyncio
async def test_tool_create_skill_failure(temp_dirs):
    sandbox_dir = temp_dirs["sandbox"]
    
    with patch("ghost_agent.tools.execute.tool_execute", new_callable=AsyncMock) as mock_execute:
        # Simulate an error during execution
        mock_execute.return_value = "Error: Traceback (most recent call last):\nSyntaxError"
        
        result = await tool_create_skill(
            sandbox_dir=sandbox_dir,
            memory_system=None,
            sandbox_manager=None,
            name="bad_skill",
            description="desc",
            parameters_schema='{}',
            python_code="bad code",
            test_payload='{}'
        )
        
        assert "Skill creation failed" in result
        assert "SyntaxError" in result
        
        # Ensure it was not saved
        assert not (sandbox_dir / "acquired_skills" / "bad_skill.py").exists()

def test_acquired_skill_manager_delete(temp_dirs):
    sandbox_dir = temp_dirs["sandbox"]
    memory_sys_mock = MagicMock()
    manager = AcquiredSkillManager(sandbox_dir, memory_sys_mock)
    
    manager.save_skill("test_del_skill", "desc", {}, "code")
    assert "test_del_skill" in manager.get_all_skills()
    assert (sandbox_dir / "acquired_skills" / "test_del_skill.py").exists()
    
    # Reset mock after save call
    memory_sys_mock.reset_mock()
    
    # Delete skill
    success = manager.delete_skill("test_del_skill")
    assert success is True
    assert "test_del_skill" not in manager.get_all_skills()
    assert not (sandbox_dir / "acquired_skills" / "test_del_skill.py").exists()
    
    # Verify memory system deletion
    memory_sys_mock.collection.delete.assert_called_once_with(
        where={"name": "test_del_skill", "type": "acquired_skill"}
    )
    
    # Delete non-existent
    success = manager.delete_skill("non_existent")
    assert success is False

@pytest.mark.asyncio
async def test_tool_manage_skills_list(temp_dirs):
    from ghost_agent.tools.acquired_skills import tool_manage_skills
    sandbox_dir = temp_dirs["sandbox"]
    manager = AcquiredSkillManager(sandbox_dir, MagicMock())
    
    # Initially empty
    result = await tool_manage_skills(sandbox_dir=sandbox_dir, memory_system=None, action="list")
    assert "No custom skills have been acquired" in result
    
    manager.save_skill("skill_a", "cool skill", {}, "")
    result = await tool_manage_skills(sandbox_dir=sandbox_dir, memory_system=None, action="list")
    assert "skill_a" in result
    assert "cool skill" in result


@pytest.mark.asyncio
async def test_manage_skills_list_is_complete_and_compact(temp_dirs, tmp_path):
    """The list must be the COMPLETE custom inventory (acquired + composed) and
    steer the model away from re-tabulating every built-in tool (2026-07-15)."""
    from ghost_agent.tools.acquired_skills import tool_manage_skills
    sandbox_dir = temp_dirs["sandbox"]
    mem = tmp_path / "memory"
    (mem / "composed_skills").mkdir(parents=True)
    (mem / "composed_skills" / "composed_skills.json").write_text(json.dumps({
        "deploy_and_verify": {"trigger_description": "deploy then check health",
                              "status": "active"}}))
    AcquiredSkillManager(mem, MagicMock()).save_skill("acq1", "an acquired one", {}, "")

    result = await tool_manage_skills(sandbox_dir=sandbox_dir, memory_dir=mem,
                                      memory_system=None, action="list")
    # Both custom categories present in one call.
    assert "acq1" in result and "an acquired one" in result
    assert "deploy_and_verify" in result and "deploy then check health" in result
    # The footer steers away from the verbose built-in-tool dump.
    assert "BUILT-IN" in result
    assert "full schema" in result

@pytest.mark.asyncio
async def test_tool_manage_skills_delete(temp_dirs):
    from ghost_agent.tools.acquired_skills import tool_manage_skills
    sandbox_dir = temp_dirs["sandbox"]
    memory = MagicMock()
    manager = AcquiredSkillManager(sandbox_dir, memory)
    
    manager.save_skill("skill_b", "bob skill", {}, "")
    
    result = await tool_manage_skills(sandbox_dir=sandbox_dir, memory_system=memory, action="delete", skill_name="skill_b")
    assert "has been deleted" in result
    assert "skill_b" not in manager.get_all_skills()
    
    # Error state
    result_err = await tool_manage_skills(sandbox_dir=sandbox_dir, memory_system=memory, action="delete", skill_name="non_existent")
    assert "not found" in result_err
    
    result_err2 = await tool_manage_skills(sandbox_dir=sandbox_dir, memory_system=memory, action="delete", skill_name=None)
    assert "skill_name is required" in result_err2
