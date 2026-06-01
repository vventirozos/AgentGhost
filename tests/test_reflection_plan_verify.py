"""Tests for the reflection plan-verification hook (proposal #6).

Reflection was the one learning path with no correctness grounding —
the revised plan went straight to SkillMemory un-checked. The injected
``verify_fn`` lets a backend (LLM soundness judge, or a sandbox re-run
for self-play reflections) gate the lesson: only a verified plan
upgrades the reflection trajectory to PASSED.
"""

from __future__ import annotations

import asyncio

from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome
from ghost_agent.reflection.loop import Reflector


_VALID = (
    "DIAGNOSIS: tried to read from a path that wasn't there\n"
    "REVISED PLAN:\n"
    "1. search the workspace for the target file first\n"
    "2. then read it\n"
)


def _failed_traj(tid="tF"):
    return Trajectory(
        id=tid,
        user_request="parse a logfile",
        failure_reason="access.log not found",
        outcome=Outcome.FAILED.value,
        tool_calls=[ToolCall(name="execute", result="FileNotFoundError")],
    )


async def _critique(_prompt):
    return _VALID


async def test_no_verify_fn_keeps_unknown():
    """Back-compat: without a verify_fn the reflected trajectory stays
    UNKNOWN exactly as before."""
    refl = Reflector(critique_fn=_critique)
    out = await refl.reflect_one(_failed_traj())
    assert out.ok
    assert out.plan_verified is None
    assert out.reflected_trajectory.outcome == Outcome.UNKNOWN.value


async def test_verified_plan_upgrades_to_passed():
    def verify(traj, plan):
        assert plan  # the parsed revised plan is handed to the backend
        return True, "plan lists the file before reading — addresses the cause"

    refl = Reflector(critique_fn=_critique, verify_fn=verify)
    out = await refl.reflect_one(_failed_traj())
    assert out.plan_verified is True
    r = out.reflected_trajectory
    assert r.outcome == Outcome.PASSED.value
    assert r.extra["plan_verified"] is True
    assert "PLAN VERIFIED" in r.final_response


async def test_refuted_plan_stays_unknown():
    def verify(traj, plan):
        return False, "plan ignores the missing-file cause"

    refl = Reflector(critique_fn=_critique, verify_fn=verify)
    out = await refl.reflect_one(_failed_traj())
    assert out.plan_verified is False
    r = out.reflected_trajectory
    assert r.outcome == Outcome.UNKNOWN.value
    assert r.extra["plan_verified"] is False
    assert "PLAN UNVERIFIED" in r.final_response


async def test_async_verify_fn():
    async def verify(traj, plan):
        await asyncio.sleep(0)
        return True, "ok"

    refl = Reflector(critique_fn=_critique, verify_fn=verify)
    out = await refl.reflect_one(_failed_traj())
    assert out.plan_verified is True
    assert out.reflected_trajectory.outcome == Outcome.PASSED.value


async def test_verify_error_is_swallowed():
    def verify(traj, plan):
        raise RuntimeError("judge exploded")

    refl = Reflector(critique_fn=_critique, verify_fn=verify)
    out = await refl.reflect_one(_failed_traj())
    # Reflection still succeeds; outcome stays UNKNOWN, not crashed.
    assert out.ok
    assert out.plan_verified is None
    assert out.reflected_trajectory.outcome == Outcome.UNKNOWN.value
    assert "verify error" in out.reflected_trajectory.extra.get("plan_verify_note", "")


async def test_verify_timeout_is_swallowed():
    async def verify(traj, plan):
        await asyncio.sleep(5)
        return True, "too late"

    refl = Reflector(critique_fn=_critique, verify_fn=verify, verify_timeout_s=0.05)
    out = await refl.reflect_one(_failed_traj())
    assert out.ok
    assert out.reflected_trajectory.outcome == Outcome.UNKNOWN.value
    assert "timed out" in out.reflected_trajectory.extra.get("plan_verify_note", "")
