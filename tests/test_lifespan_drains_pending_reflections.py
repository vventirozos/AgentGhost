"""Regression: the FastAPI lifespan teardown must drain
``ctx._pending_reflection_tasks`` before the event loop closes.

The user-correction promotion path (``_maybe_promote_prior_turn_via_user_correction``)
schedules ``reflect_one`` via ``loop.create_task`` and adds the task
to ``ctx._pending_reflection_tasks``. Lifespan teardown previously
cancelled ``biological_task``, ``selfplay_loop_task``, the scheduler,
and the LLM/sandbox clients — but never iterated the pending-tasks
set. So an in-flight reflection task at shutdown got destroyed
mid-await, producing a "Task was destroyed but it is pending"
warning AND aborting the upstream LLM round-trip, which can leave
the SkillMemory composite-sink write half-applied (SQLite/Chroma).

Fix: in the ``finally`` block, after cancelling
``biological_task``, iterate ``_pending_reflection_tasks``, cancel
each, and ``asyncio.gather(..., return_exceptions=True)`` with a 5s
timeout so a stuck upstream doesn't pin shutdown.

This test isolates the lifespan teardown logic and pins the
"cancelled-and-awaited" contract without spinning up the full
FastAPI app.
"""
import asyncio
import logging
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock


# A simplified version of the lifespan-teardown drain block. The
# production code lives in main.py:lifespan; tests of the full
# lifespan need a live upstream and a real Chroma store. This
# function pins the contract that the production block must
# implement: cancel each task, gather with timeout, swallow
# CancelledError per task.
async def _drain_pending_reflections(context, timeout: float = 5.0):
    pending = getattr(context, "_pending_reflection_tasks", None)
    if not pending:
        return
    tasks = list(pending)
    for t in tasks:
        t.cancel()
    try:
        _done, still_pending = await asyncio.wait(tasks, timeout=timeout)
        if still_pending:
            logging.getLogger("GhostAgent").warning(
                "Pending reflection drain: %d task(s) did not respond to cancel within %.1fs",
                len(still_pending), timeout,
            )
    except Exception as e:
        logging.getLogger("GhostAgent").warning(f"Pending reflection drain error: {e}")


@pytest.mark.asyncio
async def test_drain_cancels_in_flight_tasks_and_awaits_them():
    """A long-running reflection task must be cancelled AND awaited
    so the event loop has a clean shutdown — no pending warnings,
    no half-applied state."""
    started = asyncio.Event()
    cancel_observed = asyncio.Event()

    async def slow_reflection():
        started.set()
        try:
            await asyncio.sleep(60)  # would-be LLM round-trip
        except asyncio.CancelledError:
            cancel_observed.set()
            raise

    task = asyncio.create_task(slow_reflection())
    pending = {task}
    pending_with_callback = pending  # the production code uses
    task.add_done_callback(pending.discard)
    ctx = SimpleNamespace(_pending_reflection_tasks=pending)

    # Wait for the task to actually be running.
    await started.wait()

    await _drain_pending_reflections(ctx)

    assert task.done(), "Pending task should be done after drain."
    assert task.cancelled() or isinstance(task.exception(), asyncio.CancelledError)
    assert cancel_observed.is_set(), "Cancellation should propagate into the task."


@pytest.mark.asyncio
async def test_drain_completes_already_finished_tasks_cleanly():
    """A reflection task that has already finished by the time the
    drain runs must still be drained cleanly — gathering on a
    completed task is a no-op, no exception."""
    completed = asyncio.Event()

    async def quick_reflection():
        completed.set()
        return "lesson written"

    task = asyncio.create_task(quick_reflection())
    # Let the task actually finish before draining.
    await task

    pending = {task}
    ctx = SimpleNamespace(_pending_reflection_tasks=pending)

    await _drain_pending_reflections(ctx)
    assert task.done()
    assert completed.is_set()
    # cancel() on a done task is a no-op; the task's result is intact.
    assert task.result() == "lesson written"


@pytest.mark.asyncio
async def test_drain_respects_timeout_when_task_ignores_cancellation():
    """An ill-behaved task that swallows CancelledError must NOT
    pin shutdown indefinitely. The drain caps the wait at the
    given timeout."""
    async def stubborn():
        # Pretend to be in a critical section that re-shields itself
        # against cancellation. Bounded so the test exits cleanly
        # even if the drain bug regressed and cancellation is missed.
        for _ in range(3):
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                pass  # bad citizen — swallows and continues

    task = asyncio.create_task(stubborn())
    pending = {task}
    ctx = SimpleNamespace(_pending_reflection_tasks=pending)
    await asyncio.sleep(0)  # let the task start

    start = asyncio.get_event_loop().time()
    await _drain_pending_reflections(ctx, timeout=0.1)
    elapsed = asyncio.get_event_loop().time() - start

    # Drain returned within ~timeout (with a generous slack). The
    # contract: if cancellation isn't honoured, the drain abandons
    # the stragglers and returns — it does NOT wait for them to
    # complete naturally.
    assert elapsed < 0.5, f"Drain took {elapsed:.2f}s, expected ~0.1s"

    # Cleanup: let the stubborn task finish naturally (~1.5s).
    try:
        await asyncio.wait_for(task, timeout=3.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass


@pytest.mark.asyncio
async def test_drain_no_op_when_no_pending_tasks():
    """The drain must be a clean no-op when the set is missing or empty
    — most agent runs never trigger user-correction promotion at all."""
    ctx_missing = SimpleNamespace()
    await _drain_pending_reflections(ctx_missing)  # should not raise

    ctx_empty = SimpleNamespace(_pending_reflection_tasks=set())
    await _drain_pending_reflections(ctx_empty)  # should not raise


def test_main_lifespan_drains_pending_reflections_in_teardown():
    """Source-level pin: the production lifespan in main.py MUST
    contain the drain block. This guards against a future refactor
    that drops it without realising what it protected."""
    from pathlib import Path
    src = Path("src/ghost_agent/main.py").read_text()
    # Must reference the pending set.
    assert "_pending_reflection_tasks" in src, (
        "lifespan must reference ctx._pending_reflection_tasks"
    )
    # Must cancel each task before waiting.
    assert ".cancel()" in src, (
        "lifespan drain must cancel pending reflection tasks"
    )
    # Must use `asyncio.wait` (not `wait_for(gather(...))`). The
    # `wait` form abandons stragglers cleanly when timeout expires;
    # `wait_for(gather(...))` blocks until every gathered task
    # finishes, which a CancelledError-swallowing task can pin
    # indefinitely. Pin the implementation choice so a future
    # refactor doesn't reintroduce the wrong shape.
    assert "asyncio.wait(" in src, (
        "lifespan drain must use asyncio.wait (not wait_for+gather) "
        "so a task that swallows CancelledError can't pin shutdown"
    )
    assert "timeout=" in src, (
        "lifespan drain must bound the wait with a timeout"
    )
