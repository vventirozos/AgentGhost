"""composed_skills — 2026-07-14 audit regressions.

Covers: branch authoring through manage_composed_skills(define) (the executor
honoured branches but nothing could author them), the no-silent-overwrite
guard on define, the explicit save_as truncation marker, the bounded parallel
fan-out, and the file_system hard-failure prefixes in _step_result_ok.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools.composed_skills import (
    MAX_BOUND_VALUE_CHARS,
    _PARALLEL_STEP_CONCURRENCY,
    _registry_from_context,
    _step_result_ok,
    tool_manage_composed_skills,
)

pytestmark = pytest.mark.asyncio


class _FakeCtx:
    def __init__(self, base):
        self.memory_dir = base
        self.sandbox_dir = base


# ------------------------------------------------------------ result gating

async def test_step_result_ok_flags_file_system_hard_failures():
    assert not _step_result_ok("SYSTEM INSTRUCTION: The 'content' parameter is MANDATORY…")
    assert not _step_result_ok("REJECTED: that replace would introduce a syntax error…")
    # …but a SUCCESS that merely mentions the phrase mid-text stays a success.
    assert _step_result_ok("SUCCESS: Applied 2 blocks. SYSTEM INSTRUCTION: 1 blocks failed…")


# ------------------------------------------------------- no silent overwrite

async def test_define_rejects_existing_name(tmp_path):
    ctx = _FakeCtx(tmp_path)
    first = await tool_manage_composed_skills(
        context=ctx, action="define", name="daily", description="v1",
        steps=[{"tool": "web_search", "params": {"query": "x"}}])
    assert "Success" in first

    reg = _registry_from_context(ctx)
    reg.record_usage("daily", success=True)  # tuned macro with stats

    second = await tool_manage_composed_skills(
        context=ctx, action="define", name="daily", description="v2 clobber",
        steps=[{"tool": "web_search", "params": {"query": "y"}}])
    assert second.startswith("Error")
    assert "already exists" in second
    sk = reg.skills["daily"]
    assert sk.trigger_description == "v1"      # original untouched
    assert sk.usage_count == 1                  # stats preserved


# --------------------------------------------------- binding truncation mark

async def test_save_as_truncation_is_marked(tmp_path):
    ctx = _FakeCtx(tmp_path)
    await tool_manage_composed_skills(
        context=ctx, action="define", name="pipe", description="d",
        mode="sequential",
        steps=[
            {"tool": "fetch", "params": {}, "save_as": "body"},
            {"tool": "consume", "params": {"text": "$body"}},
        ])
    reg = _registry_from_context(ctx)

    seen = {}

    async def executor(tool_name, args):
        if tool_name == "fetch":
            return "z" * (MAX_BOUND_VALUE_CHARS + 500)
        seen["consumed"] = args["text"]
        return "ok"

    res = await reg.execute("pipe", executor)
    assert res["success"]
    consumed = seen["consumed"]
    assert len(consumed) <= MAX_BOUND_VALUE_CHARS + 200
    assert "binding truncated" in consumed  # downstream step SEES the cut


# ----------------------------------------------------- bounded parallel fan-out

async def test_parallel_fanout_is_bounded(tmp_path):
    ctx = _FakeCtx(tmp_path)
    n_steps = _PARALLEL_STEP_CONCURRENCY * 2
    await tool_manage_composed_skills(
        context=ctx, action="define", name="wide", description="d",
        mode="parallel",
        steps=[{"tool": "t", "description": f"s{i}", "params": {}}
               for i in range(n_steps)])
    reg = _registry_from_context(ctx)

    live = 0
    peak = 0

    async def executor(tool_name, args):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return "ok"

    res = await reg.execute("wide", executor)
    assert res["success"] and res["steps_completed"] == n_steps
    assert peak <= _PARALLEL_STEP_CONCURRENCY  # all ran, never more at once


# ------------------------------------------------------------ branch authoring

async def test_define_with_branches_and_execute(tmp_path):
    ctx = _FakeCtx(tmp_path)
    r = await tool_manage_composed_skills(
        context=ctx, action="define", name="robust_fetch",
        description="fetch with fallback", mode="sequential",
        steps=[
            {"tool": "primary", "description": "try primary", "params": {},
             "branch_condition": "unreachable", "branch_target": "fallback"},
            {"tool": "after", "description": "normal continuation", "params": {}},
        ],
        branches={
            "fallback": [
                {"tool": "secondary", "description": "use mirror", "params": {}},
            ],
        })
    assert "Success" in r and "fallback" in r

    reg = _registry_from_context(ctx)
    ran = []

    async def executor(tool_name, args):
        ran.append(tool_name)
        if tool_name == "primary":
            return "host unreachable, giving up"
        return "ok"

    res = await reg.execute("robust_fetch", executor)
    assert res["success"]
    assert ran == ["primary", "secondary"]  # branched; 'after' replaced

    # Persistence round-trip keeps the branch.
    reg2 = _registry_from_context(_FakeCtx(tmp_path))
    assert "fallback" in reg2.skills["robust_fetch"].branches


async def test_define_branch_target_must_exist(tmp_path):
    r = await tool_manage_composed_skills(
        context=_FakeCtx(tmp_path), action="define", name="m1",
        description="d", mode="sequential",
        steps=[{"tool": "t", "params": {},
                "branch_condition": "error", "branch_target": "ghost"}])
    assert r.startswith("Error") and "ghost" in r


async def test_define_branch_fields_need_both(tmp_path):
    r = await tool_manage_composed_skills(
        context=_FakeCtx(tmp_path), action="define", name="m2",
        description="d", mode="sequential",
        steps=[{"tool": "t", "params": {}, "branch_condition": "error"}])
    assert r.startswith("Error") and "BOTH" in r


async def test_define_branches_require_sequential(tmp_path):
    r = await tool_manage_composed_skills(
        context=_FakeCtx(tmp_path), action="define", name="m3",
        description="d", mode="parallel",
        steps=[{"tool": "t", "params": {}}],
        branches={"alt": [{"tool": "t2", "params": {}}]})
    assert r.startswith("Error") and "sequential" in r


async def test_branch_steps_get_dataflow_validation(tmp_path):
    r = await tool_manage_composed_skills(
        context=_FakeCtx(tmp_path), action="define", name="m4",
        description="d", mode="sequential",
        steps=[{"tool": "t", "params": {},
                "branch_condition": "x", "branch_target": "alt"}],
        branches={"alt": [
            {"tool": "t2", "params": {"v": "$self_ref"}, "save_as": "self_ref"},
        ]})
    assert r.startswith("Error") and "SAME step" in r


async def test_branch_only_runtime_param_is_advertised(tmp_path):
    ctx = _FakeCtx(tmp_path)
    await tool_manage_composed_skills(
        context=ctx, action="define", name="m5",
        description="d", mode="sequential",
        steps=[{"tool": "t", "params": {},
                "branch_condition": "x", "branch_target": "alt"}],
        branches={"alt": [{"tool": "t2", "params": {"city": "$city"}}]})
    defs = _registry_from_context(ctx).to_tool_definitions()
    entry = next(d for d in defs if d["function"]["name"] == "m5")
    assert "city" in entry["function"]["parameters"]["properties"]
