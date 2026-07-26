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
    
    # Verify memory system deletion. Chroma rejects flat multi-key where
    # dicts ("Expected where to have exactly one operator"), so the filter
    # must use $and — and we run the captured filter through chromadb's own
    # validator so a mock can never accept a shape real Chroma rejects.
    memory_sys_mock.collection.delete.assert_called_once_with(
        where={"$and": [{"name": "test_del_skill"}, {"type": "acquired_skill"}]}
    )
    from chromadb.api.types import validate_where
    validate_where(memory_sys_mock.collection.delete.call_args.kwargs["where"])
    
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


def test_purge_orphaned_skill_embeddings(temp_dirs):
    """Embeddings whose skill is gone from the registry are deleted by id;
    registered skills' embeddings are kept. Orphans exist in live stores
    because pre-2026-07-26 vector deletes used a flat multi-key where that
    Chroma rejected."""
    sandbox_dir = temp_dirs["sandbox"]
    memory = MagicMock()
    manager = AcquiredSkillManager(sandbox_dir, memory)
    manager.save_skill("kept_skill", "desc", {}, "code")
    memory.reset_mock()

    memory.collection.get.return_value = {
        "ids": ["id-kept", "id-orphan", "id-nameless"],
        "metadatas": [
            {"type": "acquired_skill", "name": "kept_skill"},
            {"type": "acquired_skill", "name": "naftemporiki_headlines"},
            None,
        ],
    }

    purged = manager.purge_orphaned_skill_embeddings()

    assert purged == 2
    memory.collection.delete.assert_called_once_with(ids=["id-orphan", "id-nameless"])
    # The get filter must be a shape real Chroma accepts.
    from chromadb.api.types import validate_where
    validate_where(memory.collection.get.call_args.kwargs["where"])


def test_purge_orphaned_skill_embeddings_no_orphans(temp_dirs):
    sandbox_dir = temp_dirs["sandbox"]
    memory = MagicMock()
    manager = AcquiredSkillManager(sandbox_dir, memory)
    manager.save_skill("kept_skill", "desc", {}, "code")
    memory.reset_mock()

    memory.collection.get.return_value = {
        "ids": ["id-kept"],
        "metadatas": [{"type": "acquired_skill", "name": "kept_skill"}],
    }

    assert manager.purge_orphaned_skill_embeddings() == 0
    memory.collection.delete.assert_not_called()


def test_purge_orphaned_skill_embeddings_no_memory_system(temp_dirs):
    manager = AcquiredSkillManager(temp_dirs["sandbox"], None)
    assert manager.purge_orphaned_skill_embeddings() == 0


def test_purge_orphaned_skill_embeddings_never_raises(temp_dirs):
    sandbox_dir = temp_dirs["sandbox"]
    memory = MagicMock()
    memory.collection.get.side_effect = RuntimeError("vector store down")
    manager = AcquiredSkillManager(sandbox_dir, memory)
    assert manager.purge_orphaned_skill_embeddings() == 0


@pytest.mark.asyncio
async def test_tool_manage_skills_sweeps_orphaned_embeddings(temp_dirs):
    """tool_manage_skills runs the orphan sweep best-effort on every call
    (same pattern as the degraded-skill retirement sweep)."""
    from ghost_agent.tools.acquired_skills import tool_manage_skills
    sandbox_dir = temp_dirs["sandbox"]
    memory = MagicMock()
    manager = AcquiredSkillManager(sandbox_dir, memory)
    manager.save_skill("live_skill", "desc", {}, "code")
    memory.reset_mock()

    memory.collection.get.return_value = {
        "ids": ["id-live", "id-orphan"],
        "metadatas": [
            {"type": "acquired_skill", "name": "live_skill"},
            {"type": "acquired_skill", "name": "deleted_long_ago"},
        ],
    }

    await tool_manage_skills(sandbox_dir=sandbox_dir, memory_system=memory, action="list")
    memory.collection.delete.assert_called_once_with(ids=["id-orphan"])


def test_save_skill_replaces_previous_embedding(temp_dirs):
    """Editing a skill re-embeds its description; the previous embedding for
    that name must be dropped first (with a Chroma-valid $and filter), or the
    skill holds one routing slot per historical edit."""
    sandbox_dir = temp_dirs["sandbox"]
    memory = MagicMock()
    manager = AcquiredSkillManager(sandbox_dir, memory)

    manager.save_skill("news_headlines", "v1 desc", {}, "code v1")
    manager.save_skill("news_headlines", "v2 desc", {}, "code v2")

    assert memory.collection.delete.call_count == 2
    from chromadb.api.types import validate_where
    for call in memory.collection.delete.call_args_list:
        validate_where(call.kwargs["where"])
    assert memory.add.call_count == 2
    # Each embed must be preceded by the delete of its stale twin.
    ops = [c[0] for c in memory.method_calls if c[0] in ("collection.delete", "add")]
    assert ops == ["collection.delete", "add", "collection.delete", "add"]


def test_save_skill_unchanged_content_skips_embedding_churn(temp_dirs):
    """Identical re-save (same content hash) must touch the vector store
    neither with an add NOR with a delete."""
    sandbox_dir = temp_dirs["sandbox"]
    memory = MagicMock()
    manager = AcquiredSkillManager(sandbox_dir, memory)

    manager.save_skill("stable_skill", "same desc", {}, "same code")
    manager.save_skill("stable_skill", "same desc", {}, "same code")

    assert memory.add.call_count == 1
    assert memory.collection.delete.call_count == 1


def test_purge_collapses_duplicate_embeddings_prefers_current_description(temp_dirs):
    """Same-name duplicates collapse to the embedding matching the registry's
    current description (the live 'injected 3 with 2 skills' shape)."""
    sandbox_dir = temp_dirs["sandbox"]
    memory = MagicMock()
    manager = AcquiredSkillManager(sandbox_dir, memory)
    manager.save_skill("news_headlines", "v2 desc", {}, "code")
    memory.reset_mock()

    memory.collection.get.return_value = {
        "ids": ["id-old", "id-new"],
        "metadatas": [
            {"type": "acquired_skill", "name": "news_headlines"},
            {"type": "acquired_skill", "name": "news_headlines"},
        ],
        "documents": ["v1 desc", "v2 desc"],
    }

    purged = manager.purge_orphaned_skill_embeddings()
    assert purged == 1
    memory.collection.delete.assert_called_once_with(ids=["id-old"])


def test_purge_collapses_duplicates_without_desc_match_keeps_newest(temp_dirs):
    """If no duplicate's document matches the registry description (e.g. the
    description changed without a content-hash change), keep the newest
    insert and drop the rest."""
    sandbox_dir = temp_dirs["sandbox"]
    memory = MagicMock()
    manager = AcquiredSkillManager(sandbox_dir, memory)
    manager.save_skill("drifty_skill", "current desc", {}, "code")
    memory.reset_mock()

    memory.collection.get.return_value = {
        "ids": ["id-a", "id-b"],
        "metadatas": [
            {"type": "acquired_skill", "name": "drifty_skill"},
            {"type": "acquired_skill", "name": "drifty_skill"},
        ],
        "documents": ["old a", "old b"],
    }

    purged = manager.purge_orphaned_skill_embeddings()
    assert purged == 1
    memory.collection.delete.assert_called_once_with(ids=["id-a"])
