"""Dream-cycle episodic consolidation (closing the built-but-unwired loop).

`EpisodicMemory.get_unconsolidated` / `mark_consolidated` existed since the
module was written but had NO caller — episodes were recorded every turn and
aged out at the 500-cap without ever being generalized, while the B3
postmortem (journal §4A) concluded dream needs a trajectory-shaped seed
source. `Dreamer._consolidate_episodes` closes both: one worker call
generalizes a batch of episodes (trigger → action chain → outcome) into
imperative strategies, gated by `_is_actionable_heuristic`, stored with
source="episode".

Failure contract (mirrors the smart-memory requeue fix, §4C): the batch is
marked consolidated only after a successful worker parse — transient worker
failures leave it queued for the next cycle.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.dream import Dreamer
from ghost_agent.memory.episodes import EpisodicMemory


@pytest.fixture
def epi(tmp_path):
    return EpisodicMemory(tmp_path)


def _record(epi, n=3, success=False):
    ids = []
    for i in range(n):
        ids.append(epi.record_episode(
            trigger=f"deploy service v{i} failed on permissions",
            context="sandbox",
            actions=[
                {"tool": "execute", "args": {"cmd": "cp"}, "result": "EACCES", "success": False},
                {"tool": "execute", "args": {"cmd": "sudo cp"}, "result": "ok", "success": True},
            ],
            outcome="recovered after path fix" if success else "failed on EACCES",
            success=success,
            lesson="check publish path first" if i == 0 else "",
            cluster_id="deploy",
        ))
    return ids


def _ctx(epi, strategies=None, side_effect=None):
    ctx = MagicMock()
    ctx.episodic_memory = epi
    if side_effect is not None:
        ctx.llm_client.chat_completion = AsyncMock(side_effect=side_effect)
    else:
        content = json.dumps({"strategies": strategies or []})
        ctx.llm_client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": content}}]
        })
    return ctx


class TestConsolidateEpisodes:
    async def test_short_circuit_below_minimum(self, epi):
        _record(epi, n=2)
        ctx = _ctx(epi)

        learned = await Dreamer(ctx)._consolidate_episodes("test-model")

        assert learned == 0
        ctx.llm_client.chat_completion.assert_not_awaited()
        assert len(epi.get_unconsolidated()) == 2  # untouched

    async def test_strategies_learned_and_batch_marked(self, epi):
        _record(epi, n=3)
        ctx = _ctx(epi, strategies=[
            "When a sandbox write fails with EACCES, verify the publish path before retrying.",
        ])

        learned = await Dreamer(ctx)._consolidate_episodes("test-model")

        assert learned == 1
        # Lesson reached SkillMemory with episode provenance.
        call = ctx.skill_memory.learn_lesson.call_args
        assert call.kwargs.get("source") == "episode"
        assert "EACCES" in call.args[2]
        # Batch marked consolidated: nothing left to drain.
        assert epi.get_unconsolidated() == []

    async def test_non_actionable_strategies_dropped_but_batch_marked(self, epi):
        _record(epi, n=3)
        ctx = _ctx(epi, strategies=[
            "The agent exhibits a tendency to retry failed deploys.",  # observation
            "The user frequently deploys services.",                    # profile
        ])

        learned = await Dreamer(ctx)._consolidate_episodes("test-model")

        assert learned == 0
        ctx.skill_memory.learn_lesson.assert_not_called()
        # Considered batch is still marked — otherwise the same rows would
        # re-process every cycle forever.
        assert epi.get_unconsolidated() == []

    async def test_worker_failure_keeps_batch_queued(self, epi):
        _record(epi, n=3)
        ctx = _ctx(epi, side_effect=RuntimeError("worker 503"))

        learned = await Dreamer(ctx)._consolidate_episodes("test-model")

        assert learned == 0
        assert len(epi.get_unconsolidated()) == 3  # retry next cycle

    async def test_no_episodic_memory_returns_zero(self):
        ctx = MagicMock()
        ctx.episodic_memory = None

        assert await Dreamer(ctx)._consolidate_episodes("test-model") == 0

    async def test_digest_carries_action_chain_and_outcome(self, epi):
        _record(epi, n=3)
        ctx = _ctx(epi, strategies=[])

        await Dreamer(ctx)._consolidate_episodes("test-model")

        payload = ctx.llm_client.chat_completion.await_args.args[0]
        digest = payload["messages"][1]["content"]
        assert "execute(FAILED) → execute" in digest
        assert "OUTCOME: FAILURE" in digest
        assert "TRIGGER: deploy service v0 failed on permissions" in digest
        assert "LESSON: check publish path first" in digest

    async def test_batch_capped(self, epi, monkeypatch):
        _record(epi, n=5)
        ctx = _ctx(epi, strategies=[])

        await Dreamer(ctx)._consolidate_episodes("test-model", max_episodes=3)

        # Only the drained batch is marked; the rest stay queued.
        assert len(epi.get_unconsolidated()) == 2


class TestDreamIntegration:
    async def test_thin_fragment_pool_still_runs_episode_pass(self, epi, monkeypatch):
        """dream() must run episodic consolidation even when the REM entropy
        gate aborts — otherwise a thin auto-fragment pool starves episodes."""
        import ghost_agent.core.dream as dream_mod
        _record(epi, n=3)
        ctx = _ctx(epi, strategies=[
            "When a sandbox write fails with EACCES, verify the publish path before retrying.",
        ])
        ctx.journal = None
        ctx.memory_system.collection.get = MagicMock(return_value={
            "ids": [], "documents": [], "metadatas": [], "embeddings": [],
        })
        monkeypatch.setattr(dream_mod, "trajectory_dream_fragments", lambda c: ([], []))

        msg = await Dreamer(ctx).dream("test-model")

        assert "Not enough entropy" in msg
        assert "still learned 1 strategy lessons" in msg
        assert epi.get_unconsolidated() == []
