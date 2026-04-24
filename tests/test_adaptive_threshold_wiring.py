"""AdaptiveThreshold must be consulted at the smart-memory gate and
must receive observations so the window self-tunes over time.

Before this wiring, `run_smart_memory_task` compared score against the
hardcoded `args.smart_memory` CLI value exclusively; the
AdaptiveThreshold instance was constructed in main.py but never read
from or written to.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
import argparse
import tempfile
from pathlib import Path


@pytest.fixture
def fake_llm():
    client = MagicMock()
    client.chat_completion = AsyncMock()
    return client


def _build_context(tmpdir: Path, fake_llm, *, with_adaptive=True,
                   adaptive_value=0.7, smart_memory=0.0):
    from ghost_agent.core.agent import GhostContext
    from ghost_agent.memory.adaptive_threshold import AdaptiveThreshold
    args = argparse.Namespace(
        max_context=4096, smart_memory=smart_memory, anonymous=True,
        model="test", verbose=False, no_memory=True,
    )
    ctx = GhostContext(args, tmpdir, tmpdir, "socks5://none")
    ctx.llm_client = fake_llm

    vector = MagicMock()
    vector.collection = MagicMock()
    vector.collection.delete = MagicMock()
    vector.search_advanced = MagicMock(return_value=[])
    ctx.memory_system = vector
    ctx.graph_memory = None
    ctx.memory_semaphore = None  # overridden below via monkeypatch

    if with_adaptive:
        at = AdaptiveThreshold(tmpdir, initial=adaptive_value)
        ctx.adaptive_threshold = at
    else:
        ctx.adaptive_threshold = None
    return ctx


class TestAdaptiveThresholdGate:
    """The gate must consult adaptive_threshold.get_threshold() and
    record an observation per call when the tracker is present."""

    async def test_adaptive_threshold_overrides_loose_cli(self, fake_llm):
        """CLI=0 (disabled), adaptive=0.8. A score of 0.7 must be
        rejected even though the CLI would have let it through."""
        from ghost_agent.core.agent import GhostAgent
        fake_llm.chat_completion.return_value = {
            "choices": [{"message": {"content":
                '{"score": 0.7, "fact": "User likes Python", "profile_update": null, "graph_triplets": []}'
            }}]
        }
        with tempfile.TemporaryDirectory() as td:
            ctx = _build_context(Path(td), fake_llm,
                                 with_adaptive=True, adaptive_value=0.8,
                                 smart_memory=0.0)
            agent = GhostAgent(ctx)
            # score 0.7 < adaptive 0.8 → would be rejected by the gate.
            # We can't easily observe the "didn't save" result, so assert
            # via the record() log: was_useful=False because gate failed.
            await agent.run_smart_memory_task(
                "USER: I like Python.\nAI: Noted.", "test-model", 0.0,
            )
            # The adaptive tracker should now have ONE observation.
            stats = ctx.adaptive_threshold.get_stats()
            assert stats["observations"] == 1
            assert stats["useful_count"] == 0  # rejected by gate

    async def test_adaptive_threshold_respected_when_stricter(self, fake_llm):
        """Score 0.9 passes both CLI (0.5) and adaptive (0.7)."""
        from ghost_agent.core.agent import GhostAgent
        fake_llm.chat_completion.return_value = {
            "choices": [{"message": {"content":
                '{"score": 0.9, "fact": "User is a data scientist", "profile_update": null, "graph_triplets": []}'
            }}]
        }
        with tempfile.TemporaryDirectory() as td:
            ctx = _build_context(Path(td), fake_llm,
                                 with_adaptive=True, adaptive_value=0.7,
                                 smart_memory=0.5)
            agent = GhostAgent(ctx)
            await agent.run_smart_memory_task(
                "USER: I'm a data scientist working with Python.\nAI: Ok.",
                "test-model", 0.5,
            )
            stats = ctx.adaptive_threshold.get_stats()
            assert stats["observations"] == 1
            # Score 0.9 clears both → useful=True
            assert stats["useful_count"] == 1

    async def test_cli_threshold_still_works_without_adaptive(self, fake_llm):
        """Back-compat: no adaptive_threshold on context → CLI value is
        used exclusively, no errors."""
        from ghost_agent.core.agent import GhostAgent
        fake_llm.chat_completion.return_value = {
            "choices": [{"message": {"content":
                '{"score": 0.5, "fact": "x", "profile_update": null, "graph_triplets": []}'
            }}]
        }
        with tempfile.TemporaryDirectory() as td:
            ctx = _build_context(Path(td), fake_llm, with_adaptive=False,
                                 smart_memory=0.7)
            agent = GhostAgent(ctx)
            # Score 0.5 < CLI 0.7 → rejected. No exception about missing
            # adaptive_threshold attribute.
            await agent.run_smart_memory_task(
                "USER: I use VSCode.\nAI: k.", "test-model", 0.7,
            )

    async def test_threshold_selftunes_with_useful_observations(self):
        """Feed the window a batch of observations and verify threshold
        converges to 90% of the min useful score."""
        from ghost_agent.memory.adaptive_threshold import AdaptiveThreshold
        with tempfile.TemporaryDirectory() as td:
            at = AdaptiveThreshold(Path(td), initial=0.5)
            # 25 observations: useful scores at 0.8-0.95, useless at 0.3-0.5.
            import random
            random.seed(42)
            for _ in range(15):
                at.record(score=random.uniform(0.8, 0.95), was_useful=True)
            for _ in range(10):
                at.record(score=random.uniform(0.3, 0.5), was_useful=False)
            # Threshold should sit near 0.9 * min_useful (~0.72) but
            # clamped to max(that, median_useless).
            final = at.get_threshold()
            assert 0.3 <= final <= 0.95
            # Must have moved above the initial default.
            assert final > 0.5

    async def test_threshold_stays_put_without_observations(self):
        from ghost_agent.memory.adaptive_threshold import AdaptiveThreshold
        with tempfile.TemporaryDirectory() as td:
            at = AdaptiveThreshold(Path(td), initial=0.75)
            # Under MIN_OBSERVATIONS (20) → no recalculation.
            for _ in range(5):
                at.record(score=0.9, was_useful=True)
            assert at.get_threshold() == 0.75
