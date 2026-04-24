"""Tests for the Tool Composition and Macro Learning module."""

import json
import pytest
import tempfile
from pathlib import Path

from ghost_agent.tools.composed_skills import (
    ComposedSkill, ComposedSkillRegistry, SkillStep,
)


@pytest.fixture
def storage_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    import shutil
    shutil.rmtree(d)


@pytest.fixture
def registry(storage_dir):
    return ComposedSkillRegistry(storage_dir=storage_dir)


class TestSkillStep:
    def test_to_dict(self):
        step = SkillStep(
            tool_name="execute",
            description="Run analysis",
            param_template={"filename": "script.py"},
        )
        d = step.to_dict()
        assert d["tool_name"] == "execute"
        assert d["param_template"]["filename"] == "script.py"

    def test_to_dict_with_branch(self):
        step = SkillStep(
            tool_name="execute",
            description="Try method A",
            branch_condition="error",
            branch_target="method_b",
        )
        d = step.to_dict()
        assert d["branch_condition"] == "error"
        assert d["branch_target"] == "method_b"


class TestComposedSkill:
    def test_success_rate_zero_usage(self):
        skill = ComposedSkill(name="test", trigger_description="test")
        assert skill.success_rate == 0.0

    def test_success_rate_calculation(self):
        skill = ComposedSkill(name="test", trigger_description="test",
                              usage_count=10, success_count=8)
        assert skill.success_rate == 0.8

    def test_to_dict(self):
        skill = ComposedSkill(
            name="csv_analysis",
            trigger_description="analyze CSV file",
            steps=[SkillStep(tool_name="file_system", description="Read CSV")],
            usage_count=5,
            success_count=4,
        )
        d = skill.to_dict()
        assert d["name"] == "csv_analysis"
        assert d["success_rate"] == 0.8
        assert len(d["steps"]) == 1


class TestComposedSkillRegistry:
    def test_register_and_retrieve(self, registry):
        skill = ComposedSkill(
            name="data_pipeline",
            trigger_description="process and analyze data file",
            steps=[
                SkillStep(tool_name="file_system", description="Read input"),
                SkillStep(tool_name="execute", description="Process data"),
                SkillStep(tool_name="file_system", description="Write output"),
            ],
        )
        registry.register(skill)
        assert "data_pipeline" in registry.skills
        assert len(registry.skills["data_pipeline"].steps) == 3

    def test_persistence(self, storage_dir):
        reg1 = ComposedSkillRegistry(storage_dir=storage_dir)
        reg1.register(ComposedSkill(
            name="persistent_skill",
            trigger_description="do something repeatable",
            steps=[SkillStep(tool_name="execute", description="Step 1")],
        ))

        reg2 = ComposedSkillRegistry(storage_dir=storage_dir)
        assert "persistent_skill" in reg2.skills

    def test_find_matching(self, registry):
        registry.register(ComposedSkill(
            name="csv_analysis",
            trigger_description="analyze CSV data file statistics",
        ))
        registry.register(ComposedSkill(
            name="web_scrape",
            trigger_description="scrape website extract data",
        ))

        matches = registry.find_matching("analyze the CSV data")
        assert len(matches) >= 1
        assert matches[0].name == "csv_analysis"

    def test_find_matching_no_results(self, registry):
        registry.register(ComposedSkill(
            name="csv_analysis",
            trigger_description="analyze CSV data file",
        ))
        matches = registry.find_matching("quantum physics simulation")
        assert len(matches) == 0

    def test_find_matching_empty_registry(self, registry):
        assert registry.find_matching("anything") == []

    def test_find_matching_empty_query(self, registry):
        registry.register(ComposedSkill(name="test", trigger_description="test"))
        assert registry.find_matching("") == []

    def test_record_usage_success(self, registry):
        registry.register(ComposedSkill(name="test", trigger_description="test"))
        registry.record_usage("test", success=True)
        assert registry.skills["test"].usage_count == 1
        assert registry.skills["test"].success_count == 1

    def test_record_usage_failure(self, registry):
        registry.register(ComposedSkill(name="test", trigger_description="test"))
        registry.record_usage("test", success=False)
        assert registry.skills["test"].usage_count == 1
        assert registry.skills["test"].success_count == 0

    def test_eviction_on_max(self, storage_dir):
        registry = ComposedSkillRegistry(storage_dir=storage_dir)
        registry.MAX_SKILLS = 3

        for i in range(3):
            registry.register(ComposedSkill(
                name=f"skill_{i}",
                trigger_description=f"skill {i}",
                usage_count=i,
            ))

        # Adding 4th should evict skill_0 (lowest usage)
        registry.register(ComposedSkill(
            name="skill_new",
            trigger_description="new skill",
            usage_count=10,
        ))
        assert "skill_0" not in registry.skills
        assert "skill_new" in registry.skills
        assert len(registry.skills) == 3

    def test_compile_from_pattern(self, registry):
        pattern = [
            {"tool": "file_system", "description": "Read CSV", "params": {"operation": "read"}},
            {"tool": "execute", "description": "Analyze data", "params": {"filename": "analyze.py"}},
            {"tool": "file_system", "description": "Write results", "params": {"operation": "write"}},
        ]
        skill = registry.compile_from_pattern(
            "csv_pipeline", pattern, "Read, analyze, and write CSV data"
        )
        assert skill.name == "csv_pipeline"
        assert len(skill.steps) == 3
        assert skill.steps[0].tool_name == "file_system"

    async def test_execute_linear(self, registry):
        registry.register(ComposedSkill(
            name="pipeline",
            trigger_description="pipeline",
            steps=[
                SkillStep(tool_name="file_system", description="Read",
                         param_template={"operation": "read", "path": "$input_file"}),
                SkillStep(tool_name="execute", description="Process",
                         param_template={"filename": "process.py"}),
            ],
        ))

        results = []

        async def mock_executor(tool_name, tool_args):
            results.append((tool_name, tool_args))
            return f"OK: {tool_name}"

        result = await registry.execute(
            "pipeline", mock_executor, params={"input_file": "data.csv"}
        )
        assert result["success"] is True
        assert result["steps_completed"] == 2
        # Check parameter template resolution
        assert results[0][1]["path"] == "data.csv"

    async def test_execute_with_branch(self, registry):
        registry.register(ComposedSkill(
            name="branch_test",
            trigger_description="test",
            steps=[
                SkillStep(tool_name="execute", description="Try method A",
                         branch_condition="error", branch_target="fallback"),
            ],
            branches={
                "fallback": [
                    SkillStep(tool_name="execute", description="Fallback method B"),
                ],
            },
        ))

        async def mock_executor(tool_name, tool_args):
            return "error: something went wrong"

        result = await registry.execute("branch_test", mock_executor)
        # Should have branched to fallback
        assert result["steps_completed"] >= 1

    async def test_execute_handles_failure(self, registry):
        registry.register(ComposedSkill(
            name="failing",
            trigger_description="fails",
            steps=[
                SkillStep(tool_name="execute", description="Will fail"),
            ],
        ))

        async def failing_executor(tool_name, tool_args):
            raise RuntimeError("boom")

        result = await registry.execute("failing", failing_executor)
        assert result["success"] is False

    async def test_execute_optional_step_failure(self, registry):
        registry.register(ComposedSkill(
            name="optional_test",
            trigger_description="test",
            steps=[
                SkillStep(tool_name="execute", description="Optional step", optional=True),
                SkillStep(tool_name="execute", description="Required step"),
            ],
        ))

        call_count = 0

        async def partial_executor(tool_name, tool_args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("optional fail")
            return "OK"

        result = await registry.execute("optional_test", partial_executor)
        assert result["success"] is True
        assert result["steps_completed"] == 2

    async def test_execute_unknown_skill(self, registry):
        async def noop(t, a):
            return ""
        result = await registry.execute("nonexistent", noop)
        assert result["success"] is False
        assert "not found" in result["error"]
