"""Tests for reflection.loop Reflector."""

import asyncio

import pytest

from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome
from ghost_agent.reflection.loop import (
    Reflector, ReflectionOutcome, ReflectionRunReport,
)


def _failed_trajectory(tid="tF", user_request="do the thing",
                       failure_reason="validator said no"):
    return Trajectory(
        id=tid,
        user_request=user_request,
        failure_reason=failure_reason,
        outcome=Outcome.FAILED.value,
        tool_calls=[ToolCall(name="execute", arguments={"code": "nope"},
                             result="", error="SyntaxError")],
    )


def _passed_trajectory():
    return Trajectory(
        id="tP", user_request="q", outcome=Outcome.PASSED.value,
        final_response="ok",
    )


_VALID_RESPONSE = """DIAGNOSIS: wrong python version
REVISED PLAN:
1. use python3 explicitly
2. re-run in sandbox
"""


async def test_reflector_produces_reflected_trajectory():
    async def critique(_prompt):
        return _VALID_RESPONSE
    refl = Reflector(critique_fn=critique)
    report = await refl.run(failed_source=[_failed_trajectory()])
    assert report.reflected_ok == 1
    assert report.reflected_errors == 0
    assert len(report.outcomes) == 1
    outcome = report.outcomes[0]
    assert outcome.ok
    assert outcome.diagnosis
    assert outcome.revised_plan
    reflected = outcome.reflected_trajectory
    assert reflected.task_kind == "reflection"
    assert reflected.extra["reflected_from"] == "tF"


async def test_reflector_skips_passed_trajectories():
    async def critique(_p):
        return _VALID_RESPONSE
    refl = Reflector(critique_fn=critique)
    report = await refl.run(failed_source=[_passed_trajectory(), _failed_trajectory()])
    assert report.seen_failures == 1  # only the failed one counted
    assert report.reflected_ok == 1


async def test_reflector_skips_already_reflected_ids():
    async def critique(_p):
        return _VALID_RESPONSE
    refl = Reflector(critique_fn=critique)
    already = {"tF"}
    report = await refl.run(
        failed_source=[_failed_trajectory()],
        already_reflected=already,
    )
    assert report.skipped_duplicate == 1
    assert report.reflected_ok == 0


async def test_reflector_respects_max_failures():
    async def critique(_p):
        return _VALID_RESPONSE
    refl = Reflector(critique_fn=critique, max_failures=2)
    fails = [_failed_trajectory(tid=f"f{i}") for i in range(5)]
    report = await refl.run(failed_source=fails)
    assert report.seen_failures == 2  # stops pulling after max
    assert report.reflected_ok == 2


async def test_reflector_persists_via_sink():
    sink_results = []

    async def critique(_p):
        return _VALID_RESPONSE
    refl = Reflector(critique_fn=critique)
    await refl.run(
        failed_source=[_failed_trajectory()],
        sink=sink_results.append,
    )
    assert len(sink_results) == 1
    assert sink_results[0].task_kind == "reflection"


async def test_reflector_sink_raise_non_fatal(tmp_path):
    def raising_sink(_t):
        raise OSError("disk full")

    async def critique(_p):
        return _VALID_RESPONSE
    refl = Reflector(critique_fn=critique)
    # Should not propagate the sink exception.
    report = await refl.run(
        failed_source=[_failed_trajectory()],
        sink=raising_sink,
    )
    assert report.reflected_ok == 1  # reflection succeeded, persistence failed


async def test_reflector_critique_timeout_is_captured():
    async def slow_critique(_p):
        await asyncio.sleep(10.0)
        return _VALID_RESPONSE
    refl = Reflector(critique_fn=slow_critique, per_call_timeout_s=0.05)
    report = await refl.run(failed_source=[_failed_trajectory()])
    assert report.reflected_errors == 1
    assert "timeout" in report.outcomes[0].error


async def test_reflector_critique_exception_captured():
    async def broken_critique(_p):
        raise RuntimeError("LLM fell over")
    refl = Reflector(critique_fn=broken_critique)
    report = await refl.run(failed_source=[_failed_trajectory()])
    assert report.reflected_errors == 1
    assert "RuntimeError" in report.outcomes[0].error


async def test_reflector_unparseable_response_is_error():
    async def bad_critique(_p):
        return "just some plain text with no sections"
    refl = Reflector(critique_fn=bad_critique)
    report = await refl.run(failed_source=[_failed_trajectory()])
    assert report.reflected_errors == 1
    assert "unparseable" in report.outcomes[0].error.lower()


async def test_reflector_accepts_sync_critique():
    def sync_critique(_p):
        return _VALID_RESPONSE
    refl = Reflector(critique_fn=sync_critique)
    report = await refl.run(failed_source=[_failed_trajectory()])
    assert report.reflected_ok == 1


async def test_reflector_callable_source_supported():
    """Callable source (like TrajectoryCollector.iter_trajectories)."""
    def source_factory():
        return iter([_failed_trajectory()])

    async def critique(_p):
        return _VALID_RESPONSE
    refl = Reflector(critique_fn=critique)
    report = await refl.run(failed_source=source_factory)
    assert report.reflected_ok == 1


async def test_already_reflected_set_updated_in_place():
    already = set()

    async def critique(_p):
        return _VALID_RESPONSE
    refl = Reflector(critique_fn=critique)
    await refl.run(
        failed_source=[_failed_trajectory(tid="tF")],
        already_reflected=already,
    )
    assert "tF" in already


async def test_run_skips_candidate_claimed_by_concurrent_reflect_one():
    """A concurrent reflect_one (user-correction path sharing the same
    reflected-ids set) claims candidate B while run() awaits A's critique.
    run() must re-check membership at claim time — vetting at collection
    time is not enough — or B gets reflected twice (duplicate lessons)."""
    already: set = set()
    sink_results = []
    gate = asyncio.Event()
    a = _failed_trajectory(tid="A", user_request="task-A")
    b = _failed_trajectory(tid="B", user_request="task-B")

    async def critique(prompt):
        if "task-A" in prompt:
            await gate.wait()
        return _VALID_RESPONSE

    refl = Reflector(critique_fn=critique)
    run_task = asyncio.create_task(refl.run(
        failed_source=[a, b], sink=sink_results.append, already_reflected=already,
    ))
    for _ in range(5):
        await asyncio.sleep(0)
    assert "A" in already  # run() claimed A and is parked in its critique

    # The concurrent path reflects B during A's await window...
    out_b = await refl.reflect_one(b, sink=sink_results.append, already_reflected=already)
    assert out_b.ok
    # ...and a concurrent attempt on the already-claimed A is refused.
    out_a = await refl.reflect_one(a, already_reflected=already)
    assert out_a.error == "already reflected"

    gate.set()
    report = await run_task
    assert report.reflected_ok == 1        # A only — B was claimed elsewhere
    assert report.skipped_duplicate == 1   # B skipped at claim time
    reflected_from = [t.extra["reflected_from"] for t in sink_results]
    assert sorted(reflected_from) == ["A", "B"]  # exactly once each
    assert "B" in already  # run() never un-claimed the concurrent path's id


async def test_run_error_path_discard_keeps_concurrent_claim():
    """run()'s error-path discard must only un-claim ids THIS run claimed:
    when reflect_one already reflected B, run() (whose critique errors)
    must leave B claimed in the shared set."""
    already: set = set()
    gate = asyncio.Event()
    a = _failed_trajectory(tid="A", user_request="task-A")
    b = _failed_trajectory(tid="B", user_request="task-B")

    async def critique(prompt):
        if "task-A" in prompt:
            await gate.wait()
            raise RuntimeError("LLM fell over")
        return _VALID_RESPONSE

    refl = Reflector(critique_fn=critique)
    run_task = asyncio.create_task(refl.run(
        failed_source=[a, b], already_reflected=already,
    ))
    for _ in range(5):
        await asyncio.sleep(0)
    out_b = await refl.reflect_one(b, already_reflected=already)
    assert out_b.ok

    gate.set()
    report = await run_task
    assert report.reflected_errors == 1  # A's critique raised
    assert "A" not in already            # A un-claimed for a later retry
    assert "B" in already                # the concurrent claim survives


async def test_run_report_summary_string_includes_counts():
    async def critique(_p):
        return _VALID_RESPONSE
    refl = Reflector(critique_fn=critique)
    report = await refl.run(
        failed_source=[_failed_trajectory(tid="a"), _failed_trajectory(tid="b")],
    )
    s = report.summary()
    assert "2/2" in s


async def test_reflected_trajectory_fields_populated_correctly():
    async def critique(_p):
        return _VALID_RESPONSE
    refl = Reflector(critique_fn=critique, model="ghost-model")
    report = await refl.run(failed_source=[_failed_trajectory()])
    r = report.outcomes[0].reflected_trajectory
    assert r.model == "ghost-model"
    assert r.n_steps == 2  # two plan steps
    assert "DIAGNOSIS" in r.final_response
    assert r.extra["source_outcome"] == Outcome.FAILED.value
    assert r.outcome == Outcome.UNKNOWN.value
