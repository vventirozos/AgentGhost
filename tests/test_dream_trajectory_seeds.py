"""Dream trajectory seeding (journal §6 2026-07-09).

The REM entropy gate counted only `type:"auto"` vector fragments, which
nothing organically feeds: B3's fact-chat seeding and B4's task seeding both
left the pool at 0 across 12 instrumented arm-runs. Trajectories are the
substrate the agent always produces — `trajectory_dream_fragments` digests
them into REM seed material, `dream()` falls back to it when the auto pool
is thin, and the watchdog eligibility gate mirrors the fallback (else it is
dead code). In trajectory mode the merge/delete consolidation pass is
disabled: `traj:` ids are not vector-store ids.
"""

import inspect
import json
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.dream import Dreamer, trajectory_dream_fragments
from ghost_agent.distill.schema import ToolCall, Trajectory


def _traj(i, outcome="FAILED", error="", reason=""):
    return Trajectory(
        id=f"t{i}",
        user_request=f"Fix the flaky import in module_{i}.py",
        tool_calls=[ToolCall(name="execute", error=error),
                    ToolCall(name="file_system")],
        outcome=outcome,
        failure_reason=reason,
    )


def _ctx_with_trajs(trajs):
    ctx = MagicMock()
    collector = MagicMock()
    collector.iter_trajectories.return_value = iter(trajs)
    ctx.trajectory_collector = collector
    return ctx


# ── the digest builder ───────────────────────────────────────────────────────

def test_fragments_shape_and_content():
    ctx = _ctx_with_trajs([
        _traj(1, "FAILED", error="ModuleNotFoundError: no module named x",
              reason="import loop"),
        _traj(2, "PASSED"),
    ])
    ids, docs = trajectory_dream_fragments(ctx)
    assert ids == ["traj:t1", "traj:t2"]
    assert "TASK: Fix the flaky import" in docs[0]
    assert "OUTCOME: FAILED" in docs[0]
    assert "TOOLS: execute,file_system" in docs[0]
    assert "FIRST_ERROR: ModuleNotFoundError" in docs[0]
    assert "WHY: import loop" in docs[0]
    assert "FIRST_ERROR" not in docs[1]  # clean run has no error segment


def test_fragments_limit_keeps_newest():
    ctx = _ctx_with_trajs([_traj(i) for i in range(50)])
    ids, docs = trajectory_dream_fragments(ctx, limit=10)
    assert len(ids) == 10
    assert ids[-1] == "traj:t49"  # tail of the stream = newest


def test_fragments_never_raise():
    ctx = MagicMock()
    ctx.trajectory_collector = None
    assert trajectory_dream_fragments(ctx) == ([], [])
    ctx2 = MagicMock()
    ctx2.trajectory_collector.iter_trajectories.side_effect = OSError("gone")
    assert trajectory_dream_fragments(ctx2) == ([], [])


# ── dream() fallback behavior ────────────────────────────────────────────────

def _dreamer(auto_docs, trajs):
    context = MagicMock()
    context.memory_system = MagicMock()
    context.memory_system.collection.get.return_value = {
        "ids": [f"a{i}" for i in range(len(auto_docs))],
        "documents": auto_docs,
        "metadatas": [{"type": "auto"}] * len(auto_docs),
        "embeddings": [[0.1]] * len(auto_docs),
    }
    context.skill_memory = MagicMock()
    context.skill_memory._get_lock = lambda: threading.RLock()
    context.skill_memory.file_path.read_text.return_value = "[]"
    context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": json.dumps({
            "consolidations": [{
                "synthesis": "short",
                "merged_ids": ["ID:traj:t1", "ID:traj:t2"],
            }],
            "heuristics": ["Always pin module imports before running them."],
        })}}],
    })
    context._last_dream_fragment_ids = None
    collector = MagicMock()
    collector.iter_trajectories.return_value = iter(trajs)
    context.trajectory_collector = collector
    return Dreamer(context), context


@pytest.mark.asyncio
async def test_thin_auto_pool_falls_back_to_trajectories():
    dreamer, ctx = _dreamer(auto_docs=[], trajs=[_traj(i) for i in range(4)])
    out = await dreamer.dream(model_name="test-model")
    assert "Not enough entropy" not in str(out)
    ctx.llm_client.chat_completion.assert_awaited()          # REM ran
    ctx.skill_memory.learn_lesson.assert_called()            # heuristic saved
    # trajectory mode must never merge/delete: traj ids aren't vector ids
    ctx.memory_system.collection.delete.assert_not_called()


@pytest.mark.asyncio
async def test_starved_both_still_bails():
    dreamer, ctx = _dreamer(auto_docs=[], trajs=[_traj(1)])  # <3 both
    out = await dreamer.dream(model_name="test-model")
    assert "Not enough entropy" in str(out)
    ctx.llm_client.chat_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_healthy_auto_pool_keeps_the_original_path():
    dreamer, ctx = _dreamer(
        auto_docs=["fact one", "fact two", "fact three", "fact four"],
        trajs=[_traj(i) for i in range(4)])
    await dreamer.dream(model_name="test-model")
    # auto path: the trajectory collector must not even be consulted
    ctx.trajectory_collector.iter_trajectories.assert_not_called()


@pytest.mark.asyncio
async def test_trajectory_idempotency_guard():
    dreamer, ctx = _dreamer(auto_docs=[], trajs=[_traj(i) for i in range(4)])
    await dreamer.dream(model_name="test-model")
    # same trajectory set again → REM skipped, no second LLM call
    ctx.trajectory_collector.iter_trajectories.return_value = iter(
        [_traj(i) for i in range(4)])
    out2 = await dreamer.dream(model_name="test-model")
    assert "unchanged" in str(out2)
    assert ctx.llm_client.chat_completion.await_count == 1


# ── the watchdog eligibility gate must mirror the fallback ───────────────────

def test_watchdog_gate_consults_trajectories():
    from ghost_agent.core.agent import GhostAgent
    src = inspect.getsource(GhostAgent)
    assert "trajectory_seed_available" in src, (
        "watchdog dream-eligibility gate lost its trajectory fallback — "
        "dream()'s own fallback would be dead code")


def test_seed_available_counts_lines_cheaply(tmp_path):
    from ghost_agent.core.dream import trajectory_seed_available
    ctx = MagicMock()
    # mock collector root → Path ops fail closed, no iterator poked
    assert trajectory_seed_available(MagicMock()) is False
    # real dir with 3 jsonl lines across day partitions
    day = tmp_path / "2026-07-09"
    day.mkdir()
    (day / "session-a.jsonl").write_text('{"id":1}\n{"id":2}\n\n{"id":3}\n')
    ctx.trajectory_collector.root = tmp_path
    assert trajectory_seed_available(ctx) is True
    (day / "session-a.jsonl").write_text('{"id":1}\n')
    assert trajectory_seed_available(ctx) is False
