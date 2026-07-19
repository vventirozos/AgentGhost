"""Dream self-play seeding (2026-07-19 log eval).

Overnight the ONLY new experience is hourly self-play, which deliberately
detaches the trajectory collector (fake trajectories must not pollute the
distill corpus). The trajectory digest window therefore never changed and
the REM idempotency guard skipped 38/40 cycles while logging
"Auto-memory pool thin (0)" every time. The frontier tracker durably
records every self-play outcome — `selfplay_dream_fragments` digests those
into REM seed material, `dream()` merges them into the thin-pool fallback,
and the watchdog eligibility gate mirrors the fallback (else it is dead
code). Like `traj:` ids, `selfplay:` ids are not vector-store ids and must
never reach the merge/delete consolidation pass.
"""

import inspect
import json
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.dream import (
    Dreamer,
    selfplay_dream_fragments,
    trajectory_dream_fragments,
)
from ghost_agent.distill.schema import ToolCall, Trajectory
from ghost_agent.memory.frontier import FrontierTracker


def _traj(i, outcome="FAILED"):
    return Trajectory(
        id=f"t{i}",
        user_request=f"Fix the flaky import in module_{i}.py",
        tool_calls=[ToolCall(name="execute")],
        outcome=outcome,
    )


def _tracker_with_runs(tmp_path, n=3, cluster="python_general"):
    ft = FrontierTracker(tmp_path)
    for i in range(n):
        ft.record_run(
            cluster,
            f"challenge number {i} — compute averages from csv",
            attempts_used=2 if i == 0 else 1,
            passed=(i != 0),
            description_length=4,
            mistake="assumed sorted order instead of first-appearance order" if i == 0 else "",
        )
    return ft


# ── the digest builder ───────────────────────────────────────────────────────

def test_fragments_shape_and_content(tmp_path):
    ft = _tracker_with_runs(tmp_path, n=2)
    ctx = MagicMock()
    ctx.frontier_tracker = ft
    ids, docs = selfplay_dream_fragments(ctx)
    assert len(ids) == 2
    assert all(i.startswith("selfplay:python_general:") for i in ids)
    assert "SELF-PLAY: python_general" in docs[0]
    assert "OUTCOME: FAILED" in docs[0]
    assert "ATTEMPTS: 2" in docs[0]
    assert "MISTAKE: assumed sorted order" in docs[0]
    assert "OUTCOME: PASSED" in docs[1]
    assert "MISTAKE" not in docs[1]  # clean run has no mistake segment


def test_fragments_limit_keeps_newest(tmp_path):
    # recent_outcomes rings at 10 per cluster; spread across clusters so
    # more than `limit` survive and the newest must win the cut.
    ft = FrontierTracker(tmp_path)
    for c in ("alpha", "beta", "gamma"):
        for i in range(4):
            ft.record_run(c, f"{c} challenge {i}", 1, True, 4)
    ctx = MagicMock()
    ctx.frontier_tracker = ft
    ids, docs = selfplay_dream_fragments(ctx, limit=5)
    assert len(ids) == 5
    # gamma ran last, so its outcomes are the newest timestamps
    assert ids[-1].startswith("selfplay:gamma:")


def test_fragments_isinstance_gate_never_raises():
    # MagicMock context auto-creates attributes — the isinstance gate must
    # refuse them rather than digest mock garbage.
    assert selfplay_dream_fragments(MagicMock()) == ([], [])
    ctx = MagicMock()
    ctx.frontier_tracker = None
    assert selfplay_dream_fragments(ctx) == ([], [])


def test_fragments_survive_corrupt_state(tmp_path):
    ft = FrontierTracker(tmp_path)
    ctx = MagicMock()
    ctx.frontier_tracker = ft
    # empty tracker → no fragments, no exception
    assert selfplay_dream_fragments(ctx) == ([], [])


# ── dream() fallback behavior ────────────────────────────────────────────────

def _dreamer(tmp_path, trajs, tracker=None):
    context = MagicMock()
    context.memory_system = MagicMock()
    context.memory_system.collection.get.return_value = {
        "ids": [], "documents": [], "metadatas": [], "embeddings": [],
    }
    context.skill_memory = MagicMock()
    context.skill_memory._get_lock = lambda: threading.RLock()
    context.skill_memory.file_path.read_text.return_value = "[]"
    context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": json.dumps({
            "consolidations": [],
            "heuristics": ["When output order is unspecified, ask or preserve input order."],
        })}}],
    })
    context._last_dream_fragment_ids = None
    collector = MagicMock()
    collector.iter_trajectories.return_value = iter(trajs)
    context.trajectory_collector = collector
    context.frontier_tracker = tracker
    return Dreamer(context), context


@pytest.mark.asyncio
async def test_thin_pool_with_few_trajectories_seeds_from_selfplay(tmp_path):
    # 1 trajectory alone is below the ≥3 gate; 3 self-play outcomes must
    # push the merged pool over it and REM must run.
    ft = _tracker_with_runs(tmp_path, n=3)
    dreamer, ctx = _dreamer(tmp_path, trajs=[_traj(1)], tracker=ft)
    out = await dreamer.dream(model_name="test-model")
    assert "Not enough entropy" not in str(out)
    ctx.llm_client.chat_completion.assert_awaited()
    # digest mode must never merge/delete: selfplay ids aren't vector ids
    ctx.memory_system.collection.delete.assert_not_called()


@pytest.mark.asyncio
async def test_new_selfplay_outcome_reopens_idempotency_guard(tmp_path):
    ft = _tracker_with_runs(tmp_path, n=3)
    dreamer, ctx = _dreamer(tmp_path, trajs=[_traj(1)], tracker=ft)
    await dreamer.dream(model_name="test-model")
    assert ctx.llm_client.chat_completion.await_count == 1

    # same fragment set again → skip
    ctx.trajectory_collector.iter_trajectories.return_value = iter([_traj(1)])
    out2 = await dreamer.dream(model_name="test-model")
    assert "unchanged" in str(out2)
    assert ctx.llm_client.chat_completion.await_count == 1

    # a fresh self-play outcome lands (the overnight case) → REM runs again
    ft.record_run("python_general", "a brand new challenge", 1, True, 4)
    ctx.trajectory_collector.iter_trajectories.return_value = iter([_traj(1)])
    out3 = await dreamer.dream(model_name="test-model")
    assert "unchanged" not in str(out3)
    assert ctx.llm_client.chat_completion.await_count == 2


@pytest.mark.asyncio
async def test_starved_everything_still_bails(tmp_path):
    ft = _tracker_with_runs(tmp_path, n=1)  # 1 traj + 1 selfplay < 3
    dreamer, ctx = _dreamer(tmp_path, trajs=[_traj(1)], tracker=ft)
    out = await dreamer.dream(model_name="test-model")
    assert "Not enough entropy" in str(out)
    ctx.llm_client.chat_completion.assert_not_awaited()


# ── the watchdog eligibility gate must mirror the fallback ───────────────────

def test_watchdog_gate_consults_selfplay():
    from ghost_agent.core.agent import GhostAgent
    src = inspect.getsource(GhostAgent)
    assert "selfplay_dream_fragments" in src, (
        "watchdog dream-eligibility gate lost its self-play fallback — "
        "dream()'s self-play seeding would be dead code overnight")
