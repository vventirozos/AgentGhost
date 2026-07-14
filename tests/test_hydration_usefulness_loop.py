"""Post-turn hydration usefulness feedback loop (2026-07-14).

`_credit_surfaced` credits items for ENTERING the prompt — circular:
popular memories only get more popular, whether or not they ever help.
The new loop:

1. `hydrate_context` stashes this turn's surviving items (+ intent, ts)
   on `bus.last_hydration`;
2. after the reply is final, the agent spawns
   `judge_hydration_usefulness` (worker-hosted, off the critical path):
   the worker marks which snippets the reply actually drew on;
3. used vector items get `bump_helpful` (double half-life credit — see
   vector ranking), used skills get `record_helpful_retrieval`, and every
   survivor becomes an `(intent, source, used)` line in the observations
   ledger;
4. the dream cycle's `_refit_rrf_weights` fits `rrf_weights.
   fit_intent_weights` on that ledger, persists `rrf/weights.json`, and
   hot-swaps the live bus matrix — the learned fusion weights now track
   real usefulness instead of surfacing.
"""

import json
import time
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.bus import MemoryBus
from ghost_agent.core.dream import Dreamer
from ghost_agent.memory.vector import VectorMemory


def _stash(bus, survivors, intent="contextual", age=0.0):
    bus.last_hydration = {
        "intent": intent,
        "survivors": survivors,
        "ts": time.time() - age,
    }


def _llm(used):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": json.dumps({"used": used})}}]
    })
    return llm


SURVIVORS = [
    {"source": "vector", "text": "The user lives in Athens.", "mem_id": "v1"},
    {"source": "skill", "text": "TRIGGER: deploy fails", "trigger": "deploy fails"},
    {"source": "graph", "text": "user -> LIVES_IN -> athens"},
]


class TestJudge:
    async def test_credits_only_used_items(self, tmp_path):
        vector, skill = MagicMock(), MagicMock()
        bus = MemoryBus(vector_memory=vector, skill_memory=skill,
                        usefulness_ledger_path=tmp_path / "rrf" / "obs.jsonl")
        _stash(bus, SURVIVORS)

        used = await bus.judge_hydration_usefulness("You live in Athens.", _llm([1]))

        assert used == 1
        vector.bump_helpful.assert_called_once_with(["v1"])
        skill.record_helpful_retrieval.assert_not_called()

    async def test_skill_credit_and_ledger(self, tmp_path):
        vector, skill = MagicMock(), MagicMock()
        ledger = tmp_path / "rrf" / "obs.jsonl"
        bus = MemoryBus(vector_memory=vector, skill_memory=skill,
                        usefulness_ledger_path=ledger)
        _stash(bus, SURVIVORS, intent="procedural")

        used = await bus.judge_hydration_usefulness("Fixed via deploy lesson.", _llm([2]))

        assert used == 1
        skill.record_helpful_retrieval.assert_called_once_with("deploy fails")
        rows = [json.loads(l) for l in ledger.read_text().splitlines()]
        assert len(rows) == 3  # ALL survivors observed, used and unused
        assert {r["source"]: r["success"] for r in rows} == {
            "vector": False, "skill": True, "graph": False,
        }
        assert all(r["intent"] == "procedural" for r in rows)

    async def test_stash_is_consumed_once(self, tmp_path):
        bus = MemoryBus(vector_memory=MagicMock())
        _stash(bus, SURVIVORS)
        await bus.judge_hydration_usefulness("reply", _llm([1]))
        assert bus.last_hydration is None
        assert await bus.judge_hydration_usefulness("reply", _llm([1])) == 0

    async def test_stale_stash_skipped(self):
        bus = MemoryBus(vector_memory=MagicMock())
        _stash(bus, SURVIVORS, age=999.0)
        assert await bus.judge_hydration_usefulness("reply", _llm([1]), max_age_s=600) == 0

    async def test_worker_failure_writes_nothing(self, tmp_path):
        ledger = tmp_path / "obs.jsonl"
        bus = MemoryBus(vector_memory=MagicMock(), usefulness_ledger_path=ledger)
        _stash(bus, SURVIVORS)
        llm = MagicMock()
        llm.chat_completion = AsyncMock(side_effect=RuntimeError("worker 503"))

        assert await bus.judge_hydration_usefulness("reply", llm) == 0
        assert not ledger.exists()

    async def test_unparseable_verdict_is_no_credit(self, tmp_path):
        vector = MagicMock()
        bus = MemoryBus(vector_memory=vector)
        _stash(bus, SURVIVORS)
        llm = MagicMock()
        llm.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": "the snippets were all great"}}]
        })
        assert await bus.judge_hydration_usefulness("reply", llm) == 0
        vector.bump_helpful.assert_not_called()

    async def test_hydrate_context_stashes_survivors(self):
        vector = MagicMock()
        vector.search_items.return_value = [
            {"id": "v1", "text": "The user lives in Athens.", "score": 0.3},
        ]
        bus = MemoryBus(vector_memory=vector)

        out = await bus.hydrate_context("where does the user live")

        assert out
        assert bus.last_hydration is not None
        assert bus.last_hydration["survivors"][0]["mem_id"] == "v1"
        assert "intent" in bus.last_hydration and "ts" in bus.last_hydration


class TestVectorBumpHelpful:
    def test_bump_helpful_increments_and_touches(self):
        vm = VectorMemory.__new__(VectorMemory)  # bypass heavy __init__
        vm.collection = MagicMock()
        vm.collection.get.return_value = {
            "ids": ["v1"], "metadatas": [{"helpful_count": 2}],
        }
        vm.bump_helpful(["v1", "v1", None])
        vm.collection.get.assert_called_once_with(ids=["v1"], include=["metadatas"])
        meta = vm.collection.update.call_args.kwargs["metadatas"][0]
        assert meta["helpful_count"] == 3
        assert "last_accessed" in meta

    def test_bump_helpful_empty_is_noop(self):
        vm = VectorMemory.__new__(VectorMemory)
        vm.collection = MagicMock()
        vm.bump_helpful([])
        vm.collection.get.assert_not_called()


class TestRrfRefit:
    def _dreamer(self, tmp_path, n_obs, success_source="skill"):
        ledger = tmp_path / "rrf" / "observations.jsonl"
        ledger.parent.mkdir(parents=True)
        rows = []
        for i in range(n_obs):
            rows.append(json.dumps({
                "intent": "procedural",
                "source": success_source if i % 2 == 0 else "graph",
                "success": i % 2 == 0,
                "ts": "2026-07-14T00:00:00Z",
            }))
        ledger.write_text("\n".join(rows) + "\n")
        bus = MemoryBus(usefulness_ledger_path=ledger)
        ctx = SimpleNamespace(memory_bus=bus, memory_system=MagicMock())
        return Dreamer(ctx), bus, ledger

    def test_too_few_observations_no_refit(self, tmp_path):
        dreamer, bus, _ = self._dreamer(tmp_path, n_obs=10)
        assert dreamer._refit_rrf_weights(min_observations=30) is False
        assert bus._intent_weights is None

    def test_refit_swaps_and_persists(self, tmp_path):
        dreamer, bus, ledger = self._dreamer(tmp_path, n_obs=40)
        assert dreamer._refit_rrf_weights(min_observations=30) is True
        # Hot swap applied: skill (always useful) up, graph (never) down.
        w = bus._intent_weights["procedural"]
        assert w["skill"] > w["graph"]
        # Persisted where main.py loads at boot.
        from ghost_agent.core.rrf_weights import load_intent_weights
        loaded = load_intent_weights(ledger.parent / "weights.json")
        assert loaded is not None
        assert loaded["procedural"]["skill"] == w["skill"]
        # Untouched cells keep the base weights.
        assert bus._intent_weights["factual"]["vector"] == MemoryBus._INTENT_WEIGHTS["factual"]["vector"]

    def test_ledger_trimmed_past_cap(self, tmp_path):
        dreamer, _, ledger = self._dreamer(tmp_path, n_obs=120)
        assert dreamer._refit_rrf_weights(
            min_observations=30, max_ledger_lines=100, keep_lines=50) is True
        assert len(ledger.read_text().splitlines()) == 50

    def test_no_ledger_is_silent(self):
        ctx = SimpleNamespace(memory_bus=MemoryBus(), memory_system=MagicMock())
        assert Dreamer(ctx)._refit_rrf_weights() is False


class TestAgentHook:
    async def test_spawn_helper_is_defensive(self):
        import asyncio
        from ghost_agent.core.agent import GhostAgent

        # No bus / no stash → silent no-op.
        stub = SimpleNamespace(context=SimpleNamespace(memory_bus=None))
        GhostAgent._judge_hydration_safe(stub, "reply")

        # Stashed bus → judge coroutine actually spawned.
        bus = MagicMock()
        bus.last_hydration = {"survivors": [{}], "ts": time.time()}
        judge = AsyncMock(return_value=1)
        bus.judge_hydration_usefulness = judge
        stub = SimpleNamespace(context=SimpleNamespace(
            memory_bus=bus, llm_client=MagicMock(), args=SimpleNamespace(model="m"),
        ))
        GhostAgent._judge_hydration_safe(stub, "reply")
        await asyncio.sleep(0.05)  # let the bg task run
        judge.assert_awaited_once()
