"""§4LS — skills: behaviour pins for the review's fixes."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.tools.composed_skills import (
    _format_execution_result, _registry_from_context, build_step_executor,
    make_composed_skill_runner, tool_manage_composed_skills,
)
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome
from ghost_agent.utils.logging import request_id_context


class _Ctx:
    def __init__(self, base):
        self.memory_dir = base
        self.sandbox_dir = base


@pytest.fixture
def ctx(tmp_path):
    return _Ctx(tmp_path)


def _as(rid):
    return request_id_context.set(rid)


# ── M1: a macro DECLARES how it went ──

@pytest.mark.parametrize("results,success,status", [
    ([{"tool": "a", "step": "s1", "success": True, "result": "ok"},
      {"tool": "browser", "step": "s2", "success": False, "result": "Navigation failed"}],
     False, OutcomeStatus.PARTIAL),
    ([{"tool": "browser", "step": "s1", "success": False, "result": "Navigation failed"}],
     False, OutcomeStatus.FAILED),
    ([{"tool": "a", "step": "s1", "success": True, "result": "ok"}], True, OutcomeStatus.OK),
])
def test_a_macro_result_carries_its_real_status(results, success, status):
    out = _format_execution_result("m", {"success": success, "results": results,
                                         "steps_completed": len(results), "total_steps": len(results),
                                         "mode": "sequential"})
    assert isinstance(out, ToolOutcome) and out.status is status


def test_the_header_counts_only_succeeded_steps():
    out = _format_execution_result("m", {
        "success": False, "mode": "sequential", "steps_completed": 2, "total_steps": 2,
        "results": [{"tool": "a", "step": "s1", "success": True, "result": "ok"},
                    {"tool": "b", "step": "s2", "success": False, "result": "boom"}]})
    assert "1/2 steps succeeded" in out


def test_a_step_still_running_is_not_called_failed(monkeypatch):
    import ghost_agent.sandbox.jobs as J
    monkeypatch.setattr(J, "is_promoted_result", lambda t: "PROMOTED" in str(t))
    out = _format_execution_result("m", {
        "success": False, "mode": "sequential", "steps_completed": 1, "total_steps": 2,
        "results": [{"tool": "execute", "step": "s1", "success": False, "result": "PROMOTED job-1"}]})
    assert out.status is OutcomeStatus.UNRESOLVED and "STILL RUNNING" in out and "FAILED" not in out


# ── M2: a macro cannot run the macro manager (or any meta tool) ──

async def test_define_refuses_a_meta_tool_step(ctx):
    tok = _as("req-owner")
    try:
        out = await tool_manage_composed_skills(
            context=ctx, action="define", name="loop_a", description="d",
            steps=[{"tool": "manage_composed_skills", "params": {"action": "run", "name": "loop_a"}}])
    finally:
        request_id_context.reset(tok)
    assert "cannot be a macro step" in out and "loop_a" not in _registry_from_context(ctx).skills


async def test_the_step_executor_refuses_meta_tools_as_defence_in_depth():
    called = []

    async def manager(**kw):
        called.append(kw)
        return "ran"
    ex = build_step_executor({"manage_composed_skills": manager}, set())
    out = await ex("manage_composed_skills", {"action": "run", "name": "x"})
    assert "[blocked]" in out and called == []


# ── M3: macros are forbidden under replay containment ──

def test_containment_drops_every_macro(tmp_path):
    from ghost_agent.core.isolation import restrict_tool_surface
    ctx = _Ctx(tmp_path)
    reg = _registry_from_context(ctx)
    reg.compile_from_pattern("auto_ws", [{"tool": "web_search", "params": {"query": "$q"}}], "t")
    reg.skills["auto_ws"].status = "active"

    async def macro(**kw):
        return "would call web_search"

    async def read(**kw):
        return "ok"
    agent = SimpleNamespace(context=ctx, available_tools={"auto_ws": macro, "file_system": read},
                            disabled_tools=set())
    keep = restrict_tool_surface(agent)
    assert "auto_ws" not in keep and "auto_ws" not in agent.available_tools
    assert "file_system" in agent.available_tools


# ── M4: probe / background requests never change the stores ──

@pytest.mark.parametrize("rid", ["probe-1", "sched-nightly", "sub-worker"])
async def test_non_owner_requests_cannot_change_the_macro_store(ctx, rid):
    reg = _registry_from_context(ctx)
    reg.compile_from_pattern("keep_me", [{"tool": "web_search", "params": {"query": "$q"}}], "t")
    tok = _as(rid)
    try:
        d = await tool_manage_composed_skills(context=ctx, action="delete", name="keep_me")
        a = await tool_manage_composed_skills(context=ctx, action="approve", name="keep_me")
        n = await tool_manage_composed_skills(
            context=ctx, action="define", name="probe_macro", description="d",
            steps=[{"tool": "web_search", "params": {"query": "$q"}}])
    finally:
        request_id_context.reset(tok)
    skills = _registry_from_context(ctx).skills
    assert all("nothing changed" in x and "STOP" in x for x in (d, a, n))
    assert "keep_me" in skills and skills["keep_me"].status == "proposed" and "probe_macro" not in skills


async def test_a_probe_cannot_delete_or_create_an_acquired_skill(tmp_path):
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager, tool_create_skill, tool_manage_skills
    mgr = AcquiredSkillManager(tmp_path, memory_system=None)
    mgr.save_skill("gen_pw", "desc", {}, "def run(args):\n    return 1\n")
    tok = _as("probe-2")
    try:
        d = await tool_manage_skills(memory_dir=tmp_path, sandbox_dir=tmp_path, action="delete", skill_name="gen_pw")
        c = await tool_create_skill(memory_dir=tmp_path, sandbox_dir=tmp_path, name="x", description="d",
                                    parameters_schema="{}", python_code="x", test_payload="{}")
    finally:
        request_id_context.reset(tok)
    assert "nothing changed" in d and "nothing changed" in c
    assert "gen_pw" in AcquiredSkillManager(tmp_path, memory_system=None).get_all_skills()


async def test_a_probe_list_never_retires_a_skill(tmp_path):
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager, tool_manage_skills
    mgr = AcquiredSkillManager(tmp_path, memory_system=None)
    mgr.save_skill("flaky", "desc", {}, "def run(args):\n    return 1\n")
    for _ in range(3):
        mgr.log_telemetry("flaky", success=False)
    tok = _as("probe-3")
    try:
        await tool_manage_skills(memory_dir=tmp_path, sandbox_dir=tmp_path, action="delete", skill_name="nope")
    finally:
        request_id_context.reset(tok)
    assert "flaky" in AcquiredSkillManager(tmp_path, memory_system=None).get_all_skills()


# ── M5: approve re-runs the mint rules ──

@pytest.mark.parametrize("params", [{"action": "task_update", "project_id": "f36f04d446a6"}, {}])
async def test_approve_refuses_a_step_without_a_runtime_input(ctx, params):
    reg = _registry_from_context(ctx)
    reg.compile_from_pattern("stale", [{"tool": "manage_projects", "params": params}], "t")
    tok = _as("req-owner")
    try:
        out = await tool_manage_composed_skills(context=ctx, action="approve", name="stale")
    finally:
        request_id_context.reset(tok)
    assert "takes no runtime input" in out and _registry_from_context(ctx).skills["stale"].status == "proposed"


# ── m3: re-graduation refreshes the trigger examples ──

def test_regraduation_replaces_probe_examples(tmp_path):
    from ghost_agent.skills_auto.store import GraduatedSkillStore
    st = GraduatedSkillStore(tmp_path)
    c1 = SimpleNamespace(signature_hash="sig", name="n", support=3, confidence=0.8,
                         trigger_examples=["Create step2_check.txt containing exactly DISPATCH-OK-77"])
    st.graduate(c1)
    c2 = SimpleNamespace(signature_hash="sig", name="n", support=11, confidence=0.8,
                         trigger_examples=["write notes.txt and read it back"])
    e = st.graduate(c2)
    assert e["trigger_examples"] == ["write notes.txt and read it back"]


async def test_an_owner_list_still_sweeps_degraded_skills(tmp_path):
    # the tool has only list/delete — a list-exempt sweep never ran (fix review N2)
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager, tool_manage_skills
    mgr = AcquiredSkillManager(tmp_path, memory_system=None)
    mgr.save_skill("flaky", "desc", {}, "def run(args):\n    return 1\n")
    for _ in range(3):
        mgr.log_telemetry("flaky", success=False)
    tok = _as("req-owner")
    try:
        await tool_manage_skills(memory_dir=tmp_path, sandbox_dir=tmp_path, action="list")
    finally:
        request_id_context.reset(tok)
    assert "flaky" not in AcquiredSkillManager(tmp_path, memory_system=None).get_all_skills()


# ── fix review ──

async def test_a_read_only_meta_step_is_still_allowed(ctx):
    tok = _as("req-owner")
    try:
        out = await tool_manage_composed_skills(
            context=ctx, action="define", name="briefing", description="d",
            steps=[{"tool": "introspect", "params": {"action": "summary", "topic": "$topic"}},
                   {"tool": "workspace", "params": {"action": "summary", "q": "$q"}}])
    finally:
        request_id_context.reset(tok)
    assert "briefing" in _registry_from_context(ctx).skills, out


def test_a_macro_shadowing_a_builtin_never_removes_the_builtin(tmp_path):
    from ghost_agent.core.isolation import restrict_tool_surface
    ctx = _Ctx(tmp_path)
    reg = _registry_from_context(ctx)
    reg.compile_from_pattern("file_system", [{"tool": "web_search", "params": {"query": "$q"}}], "t")

    async def read(**kw):
        return "ok"
    agent = SimpleNamespace(context=ctx, available_tools={"file_system": read}, disabled_tools=set())
    restrict_tool_surface(agent)
    assert "file_system" in agent.available_tools


def test_a_hard_failure_outranks_a_step_still_running(monkeypatch):
    import ghost_agent.sandbox.jobs as J
    monkeypatch.setattr(J, "is_promoted_result", lambda t: "PROMOTED" in str(t))
    out = _format_execution_result("m", {
        "success": False, "mode": "parallel", "steps_completed": 2, "total_steps": 2,
        "results": [{"tool": "execute", "step": "s1", "success": False, "result": "PROMOTED job-1"},
                    {"tool": "browser", "step": "s2", "success": False, "result": "Navigation failed"}]})
    assert out.status is OutcomeStatus.PARTIAL and "FAILED" in out


async def test_a_job_wake_turn_resumes_the_owners_work(ctx):
    tok = _as("job-1234")
    try:
        await tool_manage_composed_skills(
            context=ctx, action="define", name="wake_macro", description="d",
            steps=[{"tool": "web_search", "params": {"query": "$q"}}])
    finally:
        request_id_context.reset(tok)
    assert "wake_macro" in _registry_from_context(ctx).skills
