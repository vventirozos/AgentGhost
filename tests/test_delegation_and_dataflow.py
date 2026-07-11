"""Tests for Feature 2 (2026-07-11):
  * composed-skill DATA-FLOW (`save_as` → `$var` threading between steps)
  * tool-using sub-agent delegation (core.subagent + tools.delegate)
  * the unified background-job registry (core.jobs + the `jobs` tool)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.composed_skills import (
    ComposedSkill, SkillStep, ComposedSkillRegistry, _validate_dataflow,
    MAX_BOUND_VALUE_CHARS,
)
from ghost_agent.core.jobs import (
    JobRegistry, get_job_registry,
    STATUS_RUNNING, STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED,
)
from ghost_agent.core.subagent import (
    resolve_allowed_tools, DEFAULT_ALLOWED_TOOLS, FORBIDDEN_TOOLS,
    MAX_SUBAGENT_TURNS,
)
from ghost_agent.tools.delegate import tool_delegate, tool_jobs


# ══════════════════════════════════════════════════════════════════════
# Composed-skill data flow
# ══════════════════════════════════════════════════════════════════════

def _mgr(tmp_path, skill):
    m = ComposedSkillRegistry(storage_dir=tmp_path)
    m.register(skill)
    return m


class TestResolveArgs:
    def test_whole_value_substitution(self):
        step = SkillStep("t", "d", {"q": "$topic"})
        assert ComposedSkillRegistry._resolve_args(step, {"topic": "otters"}) == \
            {"q": "otters"}

    def test_interpolation_in_text(self):
        step = SkillStep("t", "d", {"prompt": "Summarise this: $body"})
        out = ComposedSkillRegistry._resolve_args(step, {"body": "LONG TEXT"})
        assert out == {"prompt": "Summarise this: LONG TEXT"}

    def test_braced_form(self):
        step = SkillStep("t", "d", {"p": "pre-${x}-post"})
        assert ComposedSkillRegistry._resolve_args(step, {"x": "V"}) == {"p": "pre-V-post"}

    def test_unresolved_becomes_empty_not_literal(self):
        step = SkillStep("t", "d", {"q": "$nope", "r": "a $gone b"})
        out = ComposedSkillRegistry._resolve_args(step, {})
        assert out == {"q": "", "r": "a  b"}

    def test_non_template_values_pass_through(self):
        step = SkillStep("t", "d", {"n": 5, "s": "plain", "b": True})
        assert ComposedSkillRegistry._resolve_args(step, {}) == \
            {"n": 5, "s": "plain", "b": True}


class TestSequentialDataFlow:
    def test_step2_consumes_step1_output(self, tmp_path):
        skill = ComposedSkill(
            name="pipeline", trigger_description="fetch then use",
            execution_mode="sequential",
            steps=[
                SkillStep("fetch", "get it", {"url": "$url"}, save_as="page"),
                SkillStep("summarize", "use it", {"text": "$page"}),
            ],
        )
        m = _mgr(tmp_path, skill)
        seen = []

        async def executor(tool, args):
            seen.append((tool, dict(args)))
            if tool == "fetch":
                return "PAGE BODY HERE"
            return "SUMMARY OK"

        res = asyncio.run(m.execute("pipeline", executor, {"url": "u"}))
        assert res["success"] is True
        assert seen[0] == ("fetch", {"url": "u"})
        # The whole point: step 2 got step 1's RESULT, not "" or "$page".
        assert seen[1] == ("summarize", {"text": "PAGE BODY HERE"})
        assert res["bound"] == ["page"]
        assert res["results"][0]["saved_as"] == "page"

    def test_binding_interpolates_into_a_prompt(self, tmp_path):
        skill = ComposedSkill(
            name="p2", trigger_description="x", execution_mode="sequential",
            steps=[
                SkillStep("a", "", {}, save_as="out"),
                SkillStep("b", "", {"cmd": "echo '$out' > f.txt"}),
            ],
        )
        m = _mgr(tmp_path, skill)
        seen = []

        async def executor(tool, args):
            seen.append(args)
            return "VALUE" if tool == "a" else "ok"

        asyncio.run(m.execute("p2", executor, {}))
        assert seen[1] == {"cmd": "echo 'VALUE' > f.txt"}

    def test_three_stage_chain(self, tmp_path):
        skill = ComposedSkill(
            name="p3", trigger_description="x", execution_mode="sequential",
            steps=[
                SkillStep("s1", "", {}, save_as="a"),
                SkillStep("s2", "", {"in": "$a"}, save_as="b"),
                SkillStep("s3", "", {"in": "$b", "orig": "$a"}),
            ],
        )
        m = _mgr(tmp_path, skill)
        seen = []

        async def executor(tool, args):
            seen.append((tool, dict(args)))
            return {"s1": "A", "s2": "B"}.get(tool, "C")

        res = asyncio.run(m.execute("p3", executor, {}))
        assert res["success"] is True
        assert seen[2] == ("s3", {"in": "B", "orig": "A"})

    def test_initial_params_and_bindings_coexist(self, tmp_path):
        skill = ComposedSkill(
            name="p4", trigger_description="x", execution_mode="sequential",
            steps=[
                SkillStep("s1", "", {"q": "$topic"}, save_as="found"),
                SkillStep("s2", "", {"a": "$found", "b": "$topic"}),
            ],
        )
        m = _mgr(tmp_path, skill)
        seen = []

        async def executor(tool, args):
            seen.append(dict(args))
            return "RESULT"

        asyncio.run(m.execute("p4", executor, {"topic": "T"}))
        assert seen[1] == {"a": "RESULT", "b": "T"}

    def test_caller_params_not_mutated(self, tmp_path):
        skill = ComposedSkill(
            name="p5", trigger_description="x", execution_mode="sequential",
            steps=[SkillStep("s1", "", {}, save_as="x")],
        )
        m = _mgr(tmp_path, skill)
        params = {"topic": "T"}

        async def executor(tool, args):
            return "R"

        asyncio.run(m.execute("p5", executor, params))
        assert params == {"topic": "T"}  # no leaked binding

    def test_bound_value_is_capped(self, tmp_path):
        skill = ComposedSkill(
            name="p6", trigger_description="x", execution_mode="sequential",
            steps=[
                SkillStep("big", "", {}, save_as="blob"),
                SkillStep("use", "", {"v": "$blob"}),
            ],
        )
        m = _mgr(tmp_path, skill)
        seen = []

        async def executor(tool, args):
            seen.append(args)
            return "z" * (MAX_BOUND_VALUE_CHARS * 2) if tool == "big" else "ok"

        asyncio.run(m.execute("p6", executor, {}))
        assert len(seen[1]["v"]) == MAX_BOUND_VALUE_CHARS

    def test_failed_optional_step_still_binds(self, tmp_path):
        skill = ComposedSkill(
            name="p7", trigger_description="x", execution_mode="sequential",
            steps=[
                SkillStep("flaky", "", {}, save_as="r", optional=True),
                SkillStep("next", "", {"v": "$r"}),
            ],
        )
        m = _mgr(tmp_path, skill)
        seen = []

        async def executor(tool, args):
            seen.append(dict(args))
            return "Error: it broke" if tool == "flaky" else "ok"

        res = asyncio.run(m.execute("p7", executor, {}))
        assert res["success"] is True           # optional failure tolerated
        assert seen[1] == {"v": "Error: it broke"}  # error text visible

    def test_save_as_roundtrips_through_disk(self, tmp_path):
        m = ComposedSkillRegistry(storage_dir=tmp_path)
        m.register(ComposedSkill(
            name="p8", trigger_description="x", execution_mode="sequential",
            steps=[SkillStep("a", "", {}, save_as="v"),
                   SkillStep("b", "", {"i": "$v"})],
        ))
        m.save()
        m2 = ComposedSkillRegistry(storage_dir=tmp_path)
        assert m2.skills["p8"].steps[0].save_as == "v"

    def test_schema_omits_step_produced_names(self, tmp_path):
        m = _mgr(tmp_path, ComposedSkill(
            name="p9", trigger_description="x", execution_mode="sequential",
            steps=[
                SkillStep("a", "", {"q": "$topic"}, save_as="found"),
                SkillStep("b", "", {"v": "$found"}),
            ],
        ))
        props = m.to_tool_definitions()[0]["function"]["parameters"]["properties"]
        # `topic` is a runtime param; `found` is produced internally.
        assert set(props) == {"topic"}


class TestDataflowValidation:
    def test_forward_reference_rejected(self):
        steps = [SkillStep("a", "", {"v": "$later"}),
                 SkillStep("b", "", {}, save_as="later")]
        err = _validate_dataflow(steps, "sequential")
        assert err and "only flow forward" in err

    def test_self_reference_rejected(self):
        steps = [SkillStep("a", "", {"v": "$own"}, save_as="own")]
        err = _validate_dataflow(steps, "sequential")
        assert err and "cannot consume its own output" in err

    def test_duplicate_binding_rejected(self):
        steps = [SkillStep("a", "", {}, save_as="x"),
                 SkillStep("b", "", {}, save_as="x")]
        err = _validate_dataflow(steps, "sequential")
        assert err and "re-binds" in err

    def test_parallel_with_save_as_rejected(self):
        steps = [SkillStep("a", "", {}, save_as="x"),
                 SkillStep("b", "", {"v": "$x"})]
        err = _validate_dataflow(steps, "parallel")
        assert err and "requires mode='sequential'" in err

    def test_runtime_param_not_flagged(self):
        steps = [SkillStep("a", "", {"q": "$topic"})]
        assert _validate_dataflow(steps, "sequential") is None

    def test_valid_chain_passes(self):
        steps = [SkillStep("a", "", {"q": "$topic"}, save_as="r"),
                 SkillStep("b", "", {"v": "$r"})]
        assert _validate_dataflow(steps, "sequential") is None


# ══════════════════════════════════════════════════════════════════════
# Job registry
# ══════════════════════════════════════════════════════════════════════

class TestJobRegistry:
    def test_register_and_get(self):
        reg = JobRegistry()
        job = reg.register("subagent", "do a thing")
        assert job.status == STATUS_RUNNING
        assert reg.get(job.id) is job
        assert reg.get("nope") is None

    def test_task_success_lands_result(self):
        async def go():
            reg = JobRegistry()

            async def work():
                return "THE ANSWER"
            job = reg.register("subagent", "x", task=asyncio.create_task(work()))
            await asyncio.sleep(0.05)
            return reg.get(job.id)
        job = asyncio.run(go())
        assert job.status == STATUS_DONE and job.result == "THE ANSWER"

    def test_task_exception_lands_failed(self):
        async def go():
            reg = JobRegistry()

            async def boom():
                raise ValueError("nope")
            job = reg.register("subagent", "x", task=asyncio.create_task(boom()))
            await asyncio.sleep(0.05)
            return reg.get(job.id)
        job = asyncio.run(go())
        assert job.status == STATUS_FAILED and "ValueError: nope" in job.error

    def test_cancel(self):
        async def go():
            reg = JobRegistry()

            async def slow():
                await asyncio.sleep(30)
            job = reg.register("subagent", "x", task=asyncio.create_task(slow()))
            assert reg.cancel(job.id) is True
            await asyncio.sleep(0.05)
            return reg, job.id
        reg, jid = asyncio.run(go())
        assert reg.get(jid).status == STATUS_CANCELLED
        assert reg.cancel(jid) is False  # already finished

    def test_result_capped(self):
        async def go():
            reg = JobRegistry()

            async def big():
                return "x" * 50000
            job = reg.register("subagent", "x", task=asyncio.create_task(big()))
            await asyncio.sleep(0.05)
            return reg.get(job.id)
        from ghost_agent.core.jobs import MAX_RESULT_CHARS
        assert len(asyncio.run(go()).result) == MAX_RESULT_CHARS

    def test_eviction_keeps_running_jobs(self):
        reg = JobRegistry(max_retained=2)
        live = reg.register("subagent", "still going")   # never finishes
        ids = []
        for i in range(4):
            j = reg.register("subagent", f"j{i}")
            reg.finish(j.id, result=f"r{i}")
            ids.append(j.id)
        # Oldest COMPLETED evicted; the running one survives.
        assert reg.get(live.id) is not None
        assert reg.get(ids[0]) is None and reg.get(ids[1]) is None
        assert reg.get(ids[2]) is not None and reg.get(ids[3]) is not None

    def test_finish_is_idempotent(self):
        reg = JobRegistry()
        j = reg.register("swarm", "x")
        reg.finish(j.id, result="first")
        reg.finish(j.id, result="second")
        assert reg.get(j.id).result == "first"

    def test_list_filters(self):
        reg = JobRegistry()
        a = reg.register("subagent", "a")
        b = reg.register("swarm", "b")
        reg.finish(b.id, result="done")
        assert [j.id for j in reg.list(status=STATUS_RUNNING)] == [a.id]
        assert [j.id for j in reg.list(kind="swarm")] == [b.id]

    def test_get_job_registry_is_shared(self):
        ctx = SimpleNamespace()
        assert get_job_registry(ctx) is get_job_registry(ctx)


# ══════════════════════════════════════════════════════════════════════
# Sub-agent tool policy
# ══════════════════════════════════════════════════════════════════════

class TestSubagentToolPolicy:
    def test_default_allowlist(self):
        assert set(resolve_allowed_tools()) == set(DEFAULT_ALLOWED_TOOLS)

    def test_requested_narrows(self):
        assert resolve_allowed_tools(["web_search", "browser"]) == \
            ["browser", "web_search"]

    def test_forbidden_never_granted(self):
        for bad in FORBIDDEN_TOOLS:
            assert bad not in resolve_allowed_tools([bad, "web_search"])

    def test_no_recursive_delegation(self):
        # The headline rail: a sub-agent can never spawn more sub-agents.
        assert "delegate" not in resolve_allowed_tools(["delegate"])
        assert "delegate" not in DEFAULT_ALLOWED_TOOLS
        assert "delegate_to_swarm" not in DEFAULT_ALLOWED_TOOLS

    def test_unknown_tool_dropped(self):
        assert resolve_allowed_tools(["made_up_tool"]) == []

    def test_no_memory_writes_or_scheduling(self):
        allowed = set(resolve_allowed_tools())
        for t in ("update_profile", "manage_tasks", "manage_services",
                  "manage_projects", "self_state"):
            assert t not in allowed


# ══════════════════════════════════════════════════════════════════════
# delegate / jobs tools
# ══════════════════════════════════════════════════════════════════════

class TestDelegateTool:
    def test_requires_task(self):
        out = asyncio.run(tool_delegate(context=SimpleNamespace()))
        assert "'task'" in out and "required" in out

    def test_requires_context(self):
        out = asyncio.run(tool_delegate(task="x"))
        assert "no agent context" in out

    def test_too_many_tasks(self):
        out = asyncio.run(tool_delegate(
            tasks=[f"t{i}" for i in range(9)], context=SimpleNamespace()))
        assert "at most" in out

    def test_spawns_jobs_and_returns_ids(self, monkeypatch):
        import ghost_agent.tools.delegate as dmod

        async def fake_run(context, *, job_id, task, allowed_tools,
                           max_turns, timeout_s):
            return f"ANSWER for {task}"
        monkeypatch.setattr(dmod, "run_subagent", fake_run)

        async def go():
            ctx = SimpleNamespace()
            out = await tool_delegate(task="research otters", context=ctx)
            await asyncio.sleep(0.05)
            return out, get_job_registry(ctx)

        out, reg = asyncio.run(go())
        assert "Delegated 1 task" in out and "job-" in out
        jobs = reg.list()
        assert len(jobs) == 1 and jobs[0].status == STATUS_DONE
        assert jobs[0].result == "ANSWER for research otters"

    def test_wait_true_returns_results_inline(self, monkeypatch):
        import ghost_agent.tools.delegate as dmod

        async def fake_run(context, *, job_id, task, allowed_tools,
                           max_turns, timeout_s):
            return f"RESULT[{task}]"
        monkeypatch.setattr(dmod, "run_subagent", fake_run)
        out = asyncio.run(tool_delegate(
            tasks=["a", "b"], wait=True, context=SimpleNamespace()))
        assert "RESULT[a]" in out and "RESULT[b]" in out

    def test_failed_subagent_surfaces(self, monkeypatch):
        import ghost_agent.tools.delegate as dmod

        async def fake_run(context, **kw):
            raise RuntimeError("upstream died")
        monkeypatch.setattr(dmod, "run_subagent", fake_run)
        out = asyncio.run(tool_delegate(task="x", wait=True,
                                        context=SimpleNamespace()))
        assert "FAILED" in out and "upstream died" in out

    def test_max_turns_clamped(self, monkeypatch):
        import ghost_agent.tools.delegate as dmod
        seen = {}

        async def fake_run(context, *, job_id, task, allowed_tools,
                           max_turns, timeout_s):
            seen["max_turns"] = max_turns
            return "ok"
        monkeypatch.setattr(dmod, "run_subagent", fake_run)
        asyncio.run(tool_delegate(task="x", max_turns=999, wait=True,
                                  context=SimpleNamespace()))
        assert seen["max_turns"] == MAX_SUBAGENT_TURNS


class TestJobsTool:
    def test_status_empty(self):
        out = asyncio.run(tool_jobs(action="status", context=SimpleNamespace()))
        assert "No background jobs" in out

    def test_status_lists_running_and_finished(self):
        ctx = SimpleNamespace()
        reg = get_job_registry(ctx)
        reg.register("subagent", "still going")
        done = reg.register("subagent", "finished one")
        reg.finish(done.id, result="THE RESULT")
        out = asyncio.run(tool_jobs(action="status", context=ctx))
        assert "RUNNING (1)" in out and "FINISHED (1)" in out

    def test_collect_by_id(self):
        ctx = SimpleNamespace()
        reg = get_job_registry(ctx)
        j = reg.register("subagent", "task")
        reg.finish(j.id, result="THE RESULT")
        out = asyncio.run(tool_jobs(action="collect", job_id=j.id, context=ctx))
        assert "THE RESULT" in out

    def test_collect_all_unread(self):
        ctx = SimpleNamespace()
        reg = get_job_registry(ctx)
        for i in range(2):
            j = reg.register("subagent", f"t{i}")
            reg.finish(j.id, result=f"R{i}")
        out = asyncio.run(tool_jobs(action="collect", context=ctx))
        assert "R0" in out and "R1" in out

    def test_collect_running_says_wait(self):
        ctx = SimpleNamespace()
        j = get_job_registry(ctx).register("subagent", "slow")
        out = asyncio.run(tool_jobs(action="collect", job_id=j.id, context=ctx))
        assert "still RUNNING" in out

    def test_collect_unknown_id(self):
        out = asyncio.run(tool_jobs(action="collect", job_id="job-zzzz",
                                    context=SimpleNamespace()))
        assert "no job" in out

    def test_failed_job_shows_error(self):
        ctx = SimpleNamespace()
        reg = get_job_registry(ctx)
        j = reg.register("subagent", "x")
        reg.finish(j.id, status=STATUS_FAILED, error="boom")
        out = asyncio.run(tool_jobs(action="collect", job_id=j.id, context=ctx))
        assert "FAILED" in out and "boom" in out

    def test_unknown_action(self):
        out = asyncio.run(tool_jobs(action="explode", context=SimpleNamespace()))
        assert "unknown action" in out

    def test_action_aliases(self):
        ctx = SimpleNamespace()
        assert "No background jobs" in asyncio.run(
            tool_jobs(action="list", context=ctx))
        assert "No finished jobs" in asyncio.run(
            tool_jobs(action="result", context=ctx))


# ══════════════════════════════════════════════════════════════════════
# Wiring
# ══════════════════════════════════════════════════════════════════════

_SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"


class TestWiring:
    def test_tools_advertised_and_dispatchable(self):
        from ghost_agent.tools.registry import TOOL_DEFINITIONS
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        assert "delegate" in names and "jobs" in names
        src = (_SRC / "tools" / "registry.py").read_text()
        assert '"delegate": lambda' in src and '"jobs": lambda' in src

    def test_composed_schema_documents_save_as(self):
        from ghost_agent.tools.registry import TOOL_DEFINITIONS
        d = next(t for t in TOOL_DEFINITIONS
                 if t["function"]["name"] == "manage_composed_skills")
        step_props = (d["function"]["parameters"]["properties"]["steps"]
                      ["items"]["properties"])
        assert "save_as" in step_props

    def test_swarm_registers_in_job_registry(self):
        src = (_SRC / "tools" / "swarm.py").read_text()
        assert "get_job_registry" in src

    def test_subagent_isolation_contract(self):
        src = (_SRC / "core" / "subagent.py").read_text()
        # The isolation the dream/self-play path proved is necessary.
        for must in ("workspace_model = None", "trajectory_collector = None",
                     "episodic_memory = None", "journal = None",
                     "memory_bus = None", "ReadOnlyVectorMemory",
                     "is_background", "max_turns_override"):
            assert must in src, f"subagent missing isolation: {must}"
