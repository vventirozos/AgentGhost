"""Tests for the Tool Composition and Macro Learning module."""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path

from ghost_agent.tools.composed_skills import (
    ComposedSkill, ComposedSkillRegistry, SkillStep,
    build_step_executor, make_composed_skill_runner,
    register_composed_skill_runners, tool_manage_composed_skills,
    _format_execution_result, _registry_from_context,
)


class _FakeCtx:
    """Minimal stand-in for the agent context: just the dirs the composed-
    skill registry needs, plus tolerance for the cache attribute setattr."""
    def __init__(self, base):
        self.memory_dir = base
        self.sandbox_dir = base


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


class TestExecutionMode:
    def test_to_dict_includes_execution_mode(self):
        sk = ComposedSkill(name="m", trigger_description="m", execution_mode="parallel")
        assert sk.to_dict()["execution_mode"] == "parallel"

    def test_default_mode_is_sequential(self):
        assert ComposedSkill(name="m", trigger_description="m").execution_mode == "sequential"

    def test_execution_mode_persists(self, storage_dir):
        reg1 = ComposedSkillRegistry(storage_dir=storage_dir)
        reg1.register(ComposedSkill(
            name="m", trigger_description="m", execution_mode="parallel",
            steps=[SkillStep(tool_name="x", description="x")],
        ))
        reg2 = ComposedSkillRegistry(storage_dir=storage_dir)
        assert reg2.skills["m"].execution_mode == "parallel"


class TestParallelExecution:
    async def test_all_succeed_and_resolve_templates(self, registry):
        registry.register(ComposedSkill(
            name="p", trigger_description="p", execution_mode="parallel",
            steps=[
                SkillStep(tool_name="a", description="a", param_template={"k": "$v"}),
                SkillStep(tool_name="b", description="b"),
            ],
        ))
        seen = {}

        async def ex(tool, args):
            seen[tool] = args
            return f"ok:{tool}"

        res = await registry.execute("p", ex, params={"v": "VAL"})
        assert res["mode"] == "parallel"
        assert res["success"] is True
        assert res["steps_completed"] == 2
        assert seen["a"]["k"] == "VAL"  # $v template resolved

    async def test_partial_failure_marks_failed_but_runs_all(self, registry):
        registry.register(ComposedSkill(
            name="pf", trigger_description="pf", execution_mode="parallel",
            steps=[SkillStep(tool_name="a", description="a"),
                   SkillStep(tool_name="b", description="b")],
        ))

        async def ex(tool, args):
            if tool == "b":
                raise RuntimeError("boom")
            return "ok"

        res = await registry.execute("pf", ex)
        assert res["success"] is False
        assert res["steps_completed"] == 2  # the fan-out never short-circuits

    async def test_optional_failure_tolerated(self, registry):
        registry.register(ComposedSkill(
            name="po", trigger_description="po", execution_mode="parallel",
            steps=[SkillStep(tool_name="a", description="a", optional=True),
                   SkillStep(tool_name="b", description="b")],
        ))

        async def ex(tool, args):
            if tool == "a":
                raise RuntimeError("boom")
            return "ok"

        res = await registry.execute("po", ex)
        assert res["success"] is True

    async def test_steps_actually_run_concurrently(self, registry):
        # Step 'first' blocks on a gate that only 'second' can open. If the
        # steps ran sequentially, 'first' would time out (-> failure); the
        # macro succeeding proves they ran concurrently.
        registry.register(ComposedSkill(
            name="conc", trigger_description="c", execution_mode="parallel",
            steps=[SkillStep(tool_name="first", description="first"),
                   SkillStep(tool_name="second", description="second")],
        ))
        gate = asyncio.Event()

        async def ex(tool, args):
            if tool == "first":
                await asyncio.wait_for(gate.wait(), timeout=2.0)
                return "first-done"
            gate.set()
            return "second-done"

        res = await registry.execute("conc", ex)
        assert res["success"] is True
        assert res["steps_completed"] == 2


class TestStepExecutor:
    async def test_blocks_nested_composed_and_unknown(self):
        async def real(**k):
            return "ok"

        ex = build_step_executor({"real": real, "macro": real}, {"macro"})
        assert "blocked" in (await ex("macro", {})).lower()
        assert await ex("real", {"a": 1}) == "ok"
        assert "not available" in (await ex("ghost", {})).lower()
        # None args are tolerated
        assert await ex("real", None) == "ok"


class TestFormatExecutionResult:
    def test_basic(self):
        res = {
            "success": True, "mode": "parallel",
            "steps_completed": 2, "total_steps": 2,
            "results": [
                {"step": "Weather", "tool": "system_utility", "result": "sunny", "success": True},
                {"step": "News", "tool": "web_search", "result": "headline", "success": True},
            ],
        }
        s = _format_execution_result("morning_briefing", res)
        assert "morning_briefing" in s
        assert "sunny" in s and "headline" in s
        assert "parallel" in s

    def test_error_guard(self):
        s = _format_execution_result("ghost", {"success": False, "error": "Skill 'ghost' not found"})
        assert "error" in s.lower()
        assert "ghost" in s

    def test_failed_step_rendered(self):
        res = {
            "success": False, "mode": "parallel", "steps_completed": 1, "total_steps": 1,
            "results": [{"step": "X", "tool": "t", "error": "boom", "success": False}],
        }
        s = _format_execution_result("m", res)
        assert "FAILED" in s and "boom" in s


class TestManageComposedSkillsTool:
    async def test_requires_action(self, tmp_path):
        r = await tool_manage_composed_skills(context=_FakeCtx(tmp_path))
        assert "MANDATORY" in r

    async def test_define_list_delete_round_trip(self, tmp_path):
        ctx = _FakeCtx(tmp_path)
        r = await tool_manage_composed_skills(
            context=ctx, action="define", name="morning_briefing",
            description="weather news diagnostics lessons", mode="parallel",
            steps=[
                {"tool": "system_utility", "description": "Weather", "params": {"action": "check_weather"}},
                {"tool": "web_search", "description": "Headlines", "params": {"query": "latest news headlines"}},
                {"tool": "system_utility", "description": "Diagnostics", "params": {"action": "check_health"}},
                {"tool": "list_lessons", "description": "Lessons", "params": {"scope": "today"}},
            ],
            known_tools={"system_utility", "web_search", "list_lessons"},
        )
        assert "defined" in r.lower()
        assert "WARNING" not in r  # every step tool is known

        sk = _registry_from_context(ctx).skills["morning_briefing"]
        assert sk.execution_mode == "parallel"
        assert len(sk.steps) == 4

        listing = await tool_manage_composed_skills(context=ctx, action="list")
        assert "morning_briefing" in listing

        d = await tool_manage_composed_skills(context=ctx, action="delete", name="morning_briefing")
        assert "deleted" in d.lower()
        assert "morning_briefing" not in _registry_from_context(ctx).skills

    async def test_define_persists_to_disk(self, tmp_path):
        ctx = _FakeCtx(tmp_path)
        await tool_manage_composed_skills(
            context=ctx, action="define", name="brief", description="d",
            steps=[{"tool": "web_search", "params": {"query": "x"}}],
        )
        # A fresh context loads the registry from disk — proves persistence.
        reg2 = _registry_from_context(_FakeCtx(tmp_path))
        assert "brief" in reg2.skills

    async def test_define_warns_on_unknown_tool(self, tmp_path):
        r = await tool_manage_composed_skills(
            context=_FakeCtx(tmp_path), action="define", name="m", description="d",
            steps=[{"tool": "nonexistent_tool", "params": {}}],
            known_tools={"web_search"},
        )
        assert "WARNING" in r
        assert "nonexistent_tool" in r

    async def test_define_rejects_bad_name(self, tmp_path):
        r = await tool_manage_composed_skills(
            context=_FakeCtx(tmp_path), action="define", name="bad name!",
            description="d", steps=[{"tool": "web_search"}],
        )
        assert r.lower().startswith("error")

    async def test_define_rejects_self_reference(self, tmp_path):
        r = await tool_manage_composed_skills(
            context=_FakeCtx(tmp_path), action="define", name="loop",
            description="d", steps=[{"tool": "loop"}],
        )
        assert "recurse" in r.lower() or "itself" in r.lower()

    async def test_delete_missing(self, tmp_path):
        r = await tool_manage_composed_skills(
            context=_FakeCtx(tmp_path), action="delete", name="ghost",
        )
        assert "not found" in r.lower()


class TestRunnerWiring:
    async def test_parallel_fanout_end_to_end(self, tmp_path):
        ctx = _FakeCtx(tmp_path)
        await tool_manage_composed_skills(
            context=ctx, action="define", name="briefing", description="b",
            mode="parallel",
            steps=[{"tool": "weather", "params": {}}, {"tool": "news", "params": {}}],
        )
        calls = []

        async def weather(**k):
            calls.append("weather")
            return "sunny 21C"

        async def news(**k):
            calls.append("news")
            return "top headline"

        tools = {"weather": weather, "news": news}
        n = register_composed_skill_runners(tools, ctx)
        assert n == 1
        assert "briefing" in tools

        out = await tools["briefing"]()
        assert "sunny 21C" in out
        assert "top headline" in out
        assert set(calls) == {"weather", "news"}
        assert "COMPOSED SKILL 'briefing'" in out

    async def test_runner_skips_shadowing_existing_tool(self, tmp_path):
        ctx = _FakeCtx(tmp_path)
        await tool_manage_composed_skills(
            context=ctx, action="define", name="web_search", description="b",
            steps=[{"tool": "x", "params": {}}],
        )

        async def real_web(**k):
            return "real-builtin"

        tools = {"web_search": real_web}
        register_composed_skill_runners(tools, ctx)
        # The real built-in runner must be preserved, not clobbered by the macro.
        assert await tools["web_search"]() == "real-builtin"

    async def test_runner_skips_proposed_macros(self, tmp_path):
        ctx = _FakeCtx(tmp_path)
        reg = _registry_from_context(ctx)
        reg.compile_from_pattern(
            "auto_p", [{"tool": "a", "params": {}}, {"tool": "b", "params": {}}], "proposed",
        )  # default status="proposed"
        tools = {}
        register_composed_skill_runners(tools, ctx)
        assert "auto_p" not in tools  # proposed drafts are not dispatchable
        # After approval it gets wired.
        reg.skills["auto_p"].status = "active"
        tools2 = {}
        register_composed_skill_runners(tools2, ctx)
        assert "auto_p" in tools2


class TestProposedStatusAndApproval:
    def test_status_defaults_active(self):
        assert ComposedSkill(name="m", trigger_description="m").status == "active"

    def test_status_in_to_dict(self):
        sk = ComposedSkill(name="m", trigger_description="m", status="proposed")
        assert sk.to_dict()["status"] == "proposed"

    def test_status_persists(self, storage_dir):
        reg1 = ComposedSkillRegistry(storage_dir=storage_dir)
        reg1.register(ComposedSkill(
            name="m", trigger_description="m", status="proposed",
            steps=[SkillStep(tool_name="x", description="x")],
        ))
        reg2 = ComposedSkillRegistry(storage_dir=storage_dir)
        assert reg2.skills["m"].status == "proposed"

    def test_compile_from_pattern_is_proposed_and_sequential(self, registry):
        sk = registry.compile_from_pattern(
            "auto_x_y", [{"tool": "a", "params": {"k": 1}}, {"tool": "b"}], "desc",
        )
        assert sk.status == "proposed"
        assert sk.execution_mode == "sequential"
        assert registry.skills["auto_x_y"].status == "proposed"

    def test_proposed_hidden_from_tool_definitions(self, registry):
        registry.register(ComposedSkill(
            name="active_one", trigger_description="a",
            steps=[SkillStep(tool_name="web_search", description="s")],
        ))
        registry.compile_from_pattern("auto_a_b", [{"tool": "a"}, {"tool": "b"}], "proposed one")
        names = {d["function"]["name"] for d in registry.to_tool_definitions()}
        assert "active_one" in names
        assert "auto_a_b" not in names  # proposed is hidden from the LLM

    def test_eviction_prefers_proposed_over_active(self, storage_dir):
        reg = ComposedSkillRegistry(storage_dir=storage_dir)
        reg.MAX_SKILLS = 2
        reg.register(ComposedSkill(name="active1", trigger_description="a", usage_count=0))
        reg.compile_from_pattern("auto_p", [{"tool": "a"}, {"tool": "b"}], "proposed")
        # The 3rd registration must evict the PROPOSED draft, not an active macro.
        reg.register(ComposedSkill(name="active2", trigger_description="a", usage_count=0))
        assert "active1" in reg.skills
        assert "active2" in reg.skills
        assert "auto_p" not in reg.skills

    async def test_approve_activates_proposed(self, tmp_path):
        ctx = _FakeCtx(tmp_path)
        reg = _registry_from_context(ctx)
        reg.compile_from_pattern(
            "auto_seq",
            [{"tool": "web_search", "params": {"query": "x"}},
             {"tool": "deep_research", "params": {"query": "y"}}],
            "auto-discovered",
        )
        listing = await tool_manage_composed_skills(context=ctx, action="list")
        assert "Proposed" in listing
        assert "auto_seq" in listing

        r = await tool_manage_composed_skills(context=ctx, action="approve", name="auto_seq")
        assert "approved" in r.lower() or "active" in r.lower()
        assert _registry_from_context(ctx).skills["auto_seq"].status == "active"

    async def test_approve_missing(self, tmp_path):
        r = await tool_manage_composed_skills(
            context=_FakeCtx(tmp_path), action="approve", name="ghost",
        )
        assert "not found" in r.lower()

    async def test_approve_already_active(self, tmp_path):
        ctx = _FakeCtx(tmp_path)
        await tool_manage_composed_skills(
            context=ctx, action="define", name="m", description="d",
            steps=[{"tool": "web_search", "params": {"query": "x"}}],
        )
        r = await tool_manage_composed_skills(context=ctx, action="approve", name="m")
        assert "already active" in r.lower()
