"""Tests for Feature 2 (2026-07-11):
  * (composed-skill DATA-FLOW was removed with the macro engine, §4TA 2026-11-05)
  * (tool-using sub-agent delegation was removed 2026-10-30, §4SI)
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

from ghost_agent.core.jobs import (
    JobRegistry, get_job_registry,
    STATUS_RUNNING, STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED,
)
from ghost_agent.tools.delegate import tool_jobs



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
        res = asyncio.run(go()).result
        # §4NS: capped, and it SAYS it was cut (a 9,014-char result lost its
        # conclusion silently)
        assert res.startswith("x" * MAX_RESULT_CHARS) and "result cut at" in res
        assert len(res) < MAX_RESULT_CHARS + 200

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
# jobs tool
# ══════════════════════════════════════════════════════════════════════

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
        # §4SX D5: `jobs` is no longer advertised (`run` is); its name dispatches to the run alias
        assert "jobs" not in names and "run" in names
        # §4SI: delegate / delegate_to_swarm / dream_mode were removed
        assert not {"delegate", "delegate_to_swarm", "dream_mode"} & set(names)
        src = (_SRC / "tools" / "registry.py").read_text()
        assert '"jobs": lambda **kwargs: jobs_alias' in src and '"delegate": lambda' not in src

