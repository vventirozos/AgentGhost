"""Regression: phase 1 (journal) of `_biological_tick` must use a
cooldown anchor and follow the before-await/finally-reaffirm pattern
the other five phases (2 / 2.5 / 2.6 / 2.7 / 3) already use.

Pre-fix: phase 1 had NO ``_last_journal_at`` anchor and NO
``_JOURNAL_COOLDOWN``. The implicit cooldown was journal-empty
self-disarm — but if ``process_journal_queue`` raised an
exception (consumer bug, transient I/O), the watchdog would
re-fire the failing processor every tick (60 s) until the journal
naturally drained. The anchor pattern adds belt-and-braces
protection: anchor advances even on exception, so refire is
capped at the cooldown.

Mirrors the shape of test_reflection_biological_tick.py.
"""
import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.agent import GhostAgent


def _make_ctx_with_journal(*, idle_secs: float, has_items: bool = True):
    ctx = MagicMock()
    ctx.memory_system = MagicMock()
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=idle_secs)
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    ctx.frontier_tracker = None

    # Journal with controllable item count.
    journal = MagicMock()
    journal._lock = MagicMock()
    journal._lock.__enter__ = MagicMock(return_value=None)
    journal._lock.__exit__ = MagicMock(return_value=None)
    journal.file_path = MagicMock()
    journal.file_path.read_text = MagicMock(
        return_value='[{"type":"smart_memory","data":{}}]' if has_items else '[]'
    )
    ctx.journal = journal

    # Phase 2 will see no candidates, so it short-circuits cleanly.
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    ctx.reflector = None
    ctx.trajectory_collector = None
    return ctx


async def _tick(ctx, *, anchor=None):
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    if anchor is not None:
        agent._last_journal_at = anchor
    await agent._biological_tick()
    return agent


async def test_journal_phase_anchor_advances_on_success():
    """Successful journal-process must advance the anchor."""
    ctx = _make_ctx_with_journal(idle_secs=200, has_items=True)

    # Patch process_journal_queue to a no-op so the test isolates the
    # anchor logic from the consumer's behavior.
    with patch.object(GhostAgent, "process_journal_queue",
                      new_callable=AsyncMock):
        agent = await _tick(ctx)
    assert agent._last_journal_at > datetime.datetime.min, (
        "Successful journal-process must advance _last_journal_at."
    )


async def test_journal_phase_anchor_advances_on_exception():
    """Critical: even when ``process_journal_queue`` raises, the
    anchor must still be advanced. Otherwise the watchdog would
    re-fire the failing processor every tick. This is the
    'before-await + finally-reaffirm' invariant the other five
    phases follow.
    """
    ctx = _make_ctx_with_journal(idle_secs=200, has_items=True)

    async def _boom(self):
        raise RuntimeError("simulated consumer crash")

    with patch.object(GhostAgent, "process_journal_queue", _boom):
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = ctx
        # Should swallow the exception or propagate; either way the
        # anchor must end up advanced. We don't assert which because
        # the production code may evolve.
        try:
            await agent._biological_tick()
        except RuntimeError:
            pass
        assert agent._last_journal_at > datetime.datetime.min, (
            "Anchor MUST be advanced even when process_journal_queue raises. "
            "Without this, the failing phase re-fires every 60s tick."
        )


async def test_journal_phase_respects_cooldown():
    """A second tick within the cooldown window must NOT re-fire the
    journal processor."""
    ctx = _make_ctx_with_journal(idle_secs=200, has_items=True)

    # Pre-set the anchor to "just fired".
    just_fired = datetime.datetime.now()
    with patch.object(GhostAgent, "process_journal_queue",
                      new_callable=AsyncMock) as proc:
        await _tick(ctx, anchor=just_fired)
        proc.assert_not_called()


async def test_journal_phase_skipped_when_no_items():
    """Empty journal short-circuits before touching the anchor or
    calling the processor. (The `return` is gated on `has_items`.)"""
    ctx = _make_ctx_with_journal(idle_secs=200, has_items=False)
    with patch.object(GhostAgent, "process_journal_queue",
                      new_callable=AsyncMock) as proc:
        agent = await _tick(ctx)
        proc.assert_not_called()
        # The anchor should remain at min — no firing happened.
        assert agent._last_journal_at == datetime.datetime.min


async def test_journal_phase_falls_through_to_phase_2_when_no_items():
    """If the journal is empty, phase 1 must NOT block phases 2+ from
    running. (Pre-fix bug check: the `return` must be inside
    `if has_items:`, not at the cooldown level.)"""
    ctx = _make_ctx_with_journal(idle_secs=1200, has_items=False)
    # Wire phase 2 to fire by giving it candidates.
    ctx.memory_system.collection.get = MagicMock(
        return_value={"ids": ["a", "b", "c"]}
    )
    with patch.object(GhostAgent, "process_journal_queue",
                      new_callable=AsyncMock) as proc:
        await _tick(ctx)
        proc.assert_not_called()
        # Phase 2 entry: collection.get must have been called (the
        # phase 2 code reads candidates from memory_system.collection).
        ctx.memory_system.collection.get.assert_called()
