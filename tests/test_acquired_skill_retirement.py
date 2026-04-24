"""Tests for acquired skill auto-retirement (#14).

Verifies that:
- Skills with failure_count >= 5 and usage_count < 10 are retired
- Retired skills are moved to retired/ directory
- Active skills are not affected
- Vector store entries are cleaned up
"""

import pytest
import json
from pathlib import Path
from ghost_agent.tools.acquired_skills import AcquiredSkillManager


@pytest.fixture
def skill_manager(tmp_path):
    memory_system = None  # No vector store for these tests
    return AcquiredSkillManager(tmp_path, memory_system)


class TestSkillRetirement:
    def test_retires_degraded_skills(self, skill_manager):
        # Create a degraded skill
        skill_manager.save_skill("bad_tool", "A bad tool", {}, "print('bad')")

        # Manually set high failure count and low usage
        registry = skill_manager._load_registry()
        registry["bad_tool"]["failure_count"] = 5
        registry["bad_tool"]["usage_count"] = 3
        skill_manager._save_registry(registry)

        retired = skill_manager.retire_degraded_skills()

        assert "bad_tool" in retired
        # Should be removed from active registry
        assert "bad_tool" not in skill_manager._load_registry()
        # Should have retired/ directory
        assert (skill_manager.skills_dir / "retired").exists()

    def test_does_not_retire_active_skills(self, skill_manager):
        skill_manager.save_skill("good_tool", "A good tool", {}, "print('good')")

        # Low failure count
        registry = skill_manager._load_registry()
        registry["good_tool"]["failure_count"] = 1
        registry["good_tool"]["usage_count"] = 20
        skill_manager._save_registry(registry)

        retired = skill_manager.retire_degraded_skills()
        assert len(retired) == 0
        assert "good_tool" in skill_manager._load_registry()

    def test_does_not_retire_high_usage_degraded(self, skill_manager):
        # Under the post-#13-audit rules a skill with a CURRENT consecutive
        # failure streak >= 3 is retired regardless of usage count — the
        # `failure_count` field is reset to 0 on every successful telemetry
        # event, so a nonzero value IS the current streak. To assert the
        # "high usage shouldn't trigger retirement", we set failure_count
        # to 0 (no current streak) and a residual large counter elsewhere
        # is irrelevant. Here failure_count=2 means "fewer than 3 in a row"
        # so the skill stays active.
        skill_manager.save_skill("popular_broken", "Popular but broken", {}, "print('x')")

        registry = skill_manager._load_registry()
        registry["popular_broken"]["failure_count"] = 2  # Below the consecutive-3 threshold
        registry["popular_broken"]["usage_count"] = 15
        skill_manager._save_registry(registry)

        retired = skill_manager.retire_degraded_skills()
        assert len(retired) == 0

    def test_moves_code_to_retired_dir(self, skill_manager):
        skill_manager.save_skill("retiring_tool", "Will retire", {}, "print('bye')")

        registry = skill_manager._load_registry()
        registry["retiring_tool"]["failure_count"] = 7
        registry["retiring_tool"]["usage_count"] = 2
        skill_manager._save_registry(registry)

        skill_manager.retire_degraded_skills()

        # Code should be in retired/
        retired_path = skill_manager.skills_dir / "retired" / "retiring_tool.py"
        assert retired_path.exists()
        # Code should not be in active dir
        active_path = skill_manager.skills_dir / "retiring_tool.py"
        assert not active_path.exists()

    def test_retires_multiple_skills(self, skill_manager):
        for name in ["bad1", "bad2", "bad3"]:
            skill_manager.save_skill(name, f"Bad {name}", {}, f"print('{name}')")

        registry = skill_manager._load_registry()
        for name in ["bad1", "bad2", "bad3"]:
            registry[name]["failure_count"] = 6
            registry[name]["usage_count"] = 1
        skill_manager._save_registry(registry)

        retired = skill_manager.retire_degraded_skills()
        assert len(retired) == 3

    def test_empty_registry_returns_empty(self, skill_manager):
        retired = skill_manager.retire_degraded_skills()
        assert retired == []

    def test_retirement_with_vector_store(self, tmp_path):
        mock_memory = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        mock_memory.collection = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()

        mgr = AcquiredSkillManager(tmp_path, mock_memory)
        mgr.save_skill("vec_tool", "Vector tool", {}, "print('vec')")

        registry = mgr._load_registry()
        registry["vec_tool"]["failure_count"] = 5
        registry["vec_tool"]["usage_count"] = 0
        mgr._save_registry(registry)

        mgr.retire_degraded_skills()

        # Vector store delete should have been called
        mock_memory.collection.delete.assert_called()
