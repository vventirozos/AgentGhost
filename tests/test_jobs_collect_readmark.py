"""Read-marking for ``jobs(action='collect')`` without a job_id (2026-07-22).

Before this, every no-id collect re-dumped ALL retained finished jobs (up to
50 × 8000 chars ≈ 400KB) into the model's context, every time — and the
status text steered the model to keep calling it. Now a collect-all returns
only UN-collected results and read-marks them; an explicit by-id collect
still returns that job's result unconditionally (intentional re-fetch).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
from types import SimpleNamespace

from ghost_agent.core.jobs import (
    JobRegistry, get_job_registry,
    STATUS_RUNNING, STATUS_DONE, STATUS_FAILED,
)
from ghost_agent.tools.delegate import tool_jobs


# ══════════════════════════════════════════════════════════════════════
# Registry-level: the `collected` flag + mark_collected
# ══════════════════════════════════════════════════════════════════════

class TestMarkCollected:
    def test_flag_defaults_false(self):
        reg = JobRegistry()
        j = reg.register("subagent", "t")
        assert j.collected is False
        reg.finish(j.id, result="R")
        assert reg.get(j.id).collected is False  # finishing does not mark

    def test_mark_collected_sets_flag(self):
        reg = JobRegistry()
        j = reg.register("subagent", "t")
        reg.finish(j.id, result="R")
        reg.mark_collected([j.id])
        assert reg.get(j.id).collected is True

    def test_running_jobs_never_marked(self):
        reg = JobRegistry()
        j = reg.register("subagent", "still going")  # never finishes
        reg.mark_collected([j.id])
        assert reg.get(j.id).collected is False

    def test_unknown_ids_skipped(self):
        reg = JobRegistry()
        reg.mark_collected(["job-nope", None])  # must not raise
        reg.mark_collected(None)                # nor this

    def test_to_dict_carries_flag(self):
        reg = JobRegistry()
        j = reg.register("subagent", "t")
        reg.finish(j.id, result="R")
        assert reg.get(j.id).to_dict()["collected"] is False
        reg.mark_collected([j.id])
        assert reg.get(j.id).to_dict()["collected"] is True


# ══════════════════════════════════════════════════════════════════════
# Tool-level: collect-all returns each result ONCE
# ══════════════════════════════════════════════════════════════════════

def _ctx_with_done(n=2):
    ctx = SimpleNamespace()
    reg = get_job_registry(ctx)
    ids = []
    for i in range(n):
        j = reg.register("subagent", f"t{i}")
        reg.finish(j.id, result=f"RESULT-{i}")
        ids.append(j.id)
    return ctx, reg, ids


class TestCollectAllReadmark:
    def test_first_collect_returns_finished_jobs(self):
        ctx, _, _ = _ctx_with_done()
        out = asyncio.run(tool_jobs(action="collect", context=ctx))
        assert "Collected 2 finished job(s)" in out
        assert "RESULT-0" in out and "RESULT-1" in out

    def test_second_collect_says_nothing_new(self):
        ctx, _, ids = _ctx_with_done()
        asyncio.run(tool_jobs(action="collect", context=ctx))
        out = asyncio.run(tool_jobs(action="collect", context=ctx))
        assert "RESULT-0" not in out and "RESULT-1" not in out
        assert "No NEW results" in out
        # …and it hints that already-collected results are re-fetchable by id.
        assert "job_id" in out and ids[-1] in out

    def test_by_id_refetch_works_after_collect_all(self):
        ctx, _, ids = _ctx_with_done()
        asyncio.run(tool_jobs(action="collect", context=ctx))  # marks both
        out = asyncio.run(tool_jobs(action="collect", job_id=ids[0], context=ctx))
        assert "RESULT-0" in out  # explicit re-fetch is never hidden

    def test_by_id_refetch_repeats(self):
        # A by-id fetch may read-mark, but must ALWAYS return the result —
        # even on the second, third… explicit fetch.
        ctx, _, ids = _ctx_with_done(1)
        for _ in range(3):
            out = asyncio.run(tool_jobs(action="collect", job_id=ids[0],
                                        context=ctx))
            assert "RESULT-0" in out

    def test_new_finish_after_collect_is_returned(self):
        ctx, reg, _ = _ctx_with_done()
        asyncio.run(tool_jobs(action="collect", context=ctx))
        late = reg.register("subagent", "late one")
        reg.finish(late.id, result="LATE-RESULT")
        out = asyncio.run(tool_jobs(action="collect", context=ctx))
        assert "Collected 1 finished job(s)" in out
        assert "LATE-RESULT" in out
        assert "RESULT-0" not in out and "RESULT-1" not in out

    def test_empty_registry_message_unchanged(self):
        out = asyncio.run(tool_jobs(action="collect", context=SimpleNamespace()))
        assert "No finished jobs" in out

    def test_failed_jobs_not_swept_into_collect_all(self):
        ctx = SimpleNamespace()
        reg = get_job_registry(ctx)
        j = reg.register("subagent", "x")
        reg.finish(j.id, status=STATUS_FAILED, error="boom")
        out = asyncio.run(tool_jobs(action="collect", context=ctx))
        assert "No finished jobs" in out  # failed ≠ collectible result

    def test_running_job_not_collected_or_marked(self):
        ctx = SimpleNamespace()
        reg = get_job_registry(ctx)
        live = reg.register("subagent", "still going")
        done = reg.register("subagent", "done one")
        reg.finish(done.id, result="D")
        out = asyncio.run(tool_jobs(action="collect", context=ctx))
        assert "D" in out
        assert reg.get(live.id).collected is False
        assert reg.get(live.id).status == STATUS_RUNNING

    def test_status_action_unchanged_by_readmark(self):
        ctx, reg, ids = _ctx_with_done()
        asyncio.run(tool_jobs(action="collect", context=ctx))
        out = asyncio.run(tool_jobs(action="status", context=ctx))
        # Collected jobs still appear in status — read-marking hides them
        # only from collect-all, never from the status listing.
        assert "FINISHED (2)" in out
        assert ids[0] in out and ids[1] in out


# ══════════════════════════════════════════════════════════════════════
# Swarm contract: result_resolver output is what gets collected + marked
# ══════════════════════════════════════════════════════════════════════

class TestResolverStillCollected:
    def test_resolved_result_collected_once_then_marked(self):
        async def go():
            ctx = SimpleNamespace()
            reg = get_job_registry(ctx)

            async def worker():
                return True  # swarm workers return a bool; content elsewhere

            job = reg.register("swarm", "swarm leg")
            job.result_resolver = lambda raw: "SCRATCHPAD CONTENT"
            reg.attach(job.id, asyncio.ensure_future(worker()))
            await asyncio.sleep(0.05)
            first = await tool_jobs(action="collect", context=ctx)
            second = await tool_jobs(action="collect", context=ctx)
            by_id = await tool_jobs(action="collect", job_id=job.id, context=ctx)
            return reg.get(job.id), first, second, by_id

        job, first, second, by_id = asyncio.run(go())
        assert job.status == STATUS_DONE
        assert "SCRATCHPAD CONTENT" in first
        assert "SCRATCHPAD CONTENT" not in second and "No NEW results" in second
        assert "SCRATCHPAD CONTENT" in by_id
        assert job.collected is True
