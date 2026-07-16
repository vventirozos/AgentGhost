"""Finalize-burst stagger: hydration-judge defers to the in-flight
verifier verdict (2026-07-16).

Regression target: at end-of-turn the deferred async verify and the
hydration-usefulness judge were dispatched to the single worker node in
the same second. The node shares compute across slots, so the pair
pushed the (7–11s uncontended) verdict past its ceiling —
`Nova: ReadTimeout` on effectively every substantive finalize; one turn
shipped a hallucinated answer because the gate verdict died this way.

The fix: `_attach_late_verdict_handler` publishes its task as
`agent._deferred_verdict_task` (cleared when the verdict lands), and
`_judge_hydration_safe` waits — bounded by `_HYDRATION_JUDGE_STAGGER_S`
— for that task before spawning the judge, so the two worker calls run
back-to-back instead of colliding.
"""

import asyncio
import sys
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.agent import GhostAgent, _HYDRATION_JUDGE_STAGGER_S


def _agent(bus, verdict_task=None):
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace(
        memory_bus=bus,
        llm_client=object(),
        args=SimpleNamespace(model="test-model"),
    )
    if verdict_task is not None:
        agent._deferred_verdict_task = verdict_task
    return agent


def _bus(events, stash=None):
    async def judge(reply, llm_client, model_name="default", turn_id=""):
        events.append("judge")
        return 0

    return SimpleNamespace(
        last_hydration=stash if stash is not None
        else {"survivors": [{"text": "x"}], "ts": 0, "turn_id": "t1"},
        judge_hydration_usefulness=judge,
    )


async def _drain_bg():
    # Let spawned background tasks run to completion.
    for _ in range(20):
        await asyncio.sleep(0)


class TestStagger:
    async def test_judge_waits_for_inflight_verdict(self):
        events = []
        release = asyncio.Event()

        async def slow_verdict():
            events.append("verify-start")
            await release.wait()
            events.append("verify-done")
            return None, None

        verdict_task = asyncio.create_task(slow_verdict())
        await asyncio.sleep(0)  # verdict enters flight
        agent = _agent(_bus(events), verdict_task)

        agent._judge_hydration_safe("final reply", turn_id="t1")
        await _drain_bg()
        # Verdict still in flight → the judge must NOT have run yet.
        assert "judge" not in events

        release.set()
        await verdict_task
        await _drain_bg()
        assert events[-2:] == ["verify-done", "judge"]

    async def test_judge_runs_immediately_when_no_verdict_in_flight(self):
        events = []
        agent = _agent(_bus(events))  # no _deferred_verdict_task at all
        agent._judge_hydration_safe("final reply", turn_id="t1")
        await _drain_bg()
        assert events == ["judge"]

    async def test_judge_runs_immediately_when_verdict_already_done(self):
        events = []

        async def instant_verdict():
            return None, None

        verdict_task = asyncio.create_task(instant_verdict())
        await verdict_task
        agent = _agent(_bus(events), verdict_task)
        agent._judge_hydration_safe("final reply", turn_id="t1")
        await _drain_bg()
        assert events == ["judge"]

    async def test_judge_survives_failing_verdict_task(self):
        """asyncio.wait shields the judge from the verdict's exception —
        the usefulness credit must not die with a broken verifier."""
        events = []

        async def dying_verdict():
            raise RuntimeError("verifier exploded")

        verdict_task = asyncio.create_task(dying_verdict())
        await asyncio.sleep(0)
        agent = _agent(_bus(events), verdict_task)
        agent._judge_hydration_safe("final reply", turn_id="t1")
        await _drain_bg()
        assert events == ["judge"]

    async def test_stagger_bound_is_sane(self):
        """Must exceed the verify worker budget (45s) so the wait actually
        covers a slow verdict, yet stay far inside the judge's 600s
        stash-staleness guard."""
        assert 45.0 < _HYDRATION_JUDGE_STAGGER_S < 600.0


class TestLateVerdictHandlerPublishesTask:
    async def test_task_published_then_cleared_on_completion(self):
        agent = GhostAgent.__new__(GhostAgent)
        agent._record_late_verdict = MagicMock()

        gate = asyncio.Event()

        async def verdict():
            await gate.wait()
            return "vr", "lt"

        task = asyncio.create_task(verdict())
        agent._attach_late_verdict_handler(task, "traj-1", "conv-1")
        assert agent._deferred_verdict_task is task

        gate.set()
        await task
        await _drain_bg()
        # done-callback clears the stagger handle and applies side effects.
        assert agent._deferred_verdict_task is None
        agent._record_late_verdict.assert_called_once_with(
            "vr", "traj-1", "conv-1", last_tool="lt")

    async def test_cancelled_verdict_still_clears_handle(self):
        agent = GhostAgent.__new__(GhostAgent)
        agent._record_late_verdict = MagicMock()

        async def verdict():
            await asyncio.sleep(30)

        task = asyncio.create_task(verdict())
        agent._attach_late_verdict_handler(task, "traj-1", "conv-1")
        task.cancel()
        await _drain_bg()
        assert agent._deferred_verdict_task is None
        agent._record_late_verdict.assert_not_called()
