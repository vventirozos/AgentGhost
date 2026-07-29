"""Bounded re-queue for journal items that fail upstream-transiently.

Journal §4C: a 503 from the upstream llama during the smart-memory
consolidation was retried at the HTTP layer (worker failover + one 2s
5xx retry in the client), but on final failure the journal item — which
``pop_all`` had already removed — was swallowed by a bare
``logger.error`` and the consolidation was lost permanently, invisibly.
A main-node TIMEOUT wasn't retried at the HTTP layer at all.

Now: ``run_smart_memory_task`` classifies upstream-transient failures
(5xx / timeout / connection) and raises ``RetryableConsolidationError``
BEFORE anything was stored; ``process_journal_queue`` re-queues the item
with a bounded retry count (``JOURNAL_MAX_RETRIES``) and drops it — with
a visible WARNING — only after the cap.
"""

import datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.memory.journal import (
    JOURNAL_MAX_RETRIES,
    MemoryJournal,
    RetryableConsolidationError,
    is_upstream_transient,
)


def _http_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://127.0.0.1:8088/v1/chat/completions")
    resp = httpx.Response(status, request=req)
    return httpx.HTTPStatusError(f"HTTP {status}", request=req, response=resp)


# ---------------------------------------------------------------------------
# transient classification
# ---------------------------------------------------------------------------

def test_is_upstream_transient_classification():
    assert is_upstream_transient(_http_error(503)) is True
    assert is_upstream_transient(_http_error(500)) is True
    assert is_upstream_transient(_http_error(404)) is False
    assert is_upstream_transient(httpx.ReadTimeout("slow")) is True
    assert is_upstream_transient(httpx.ConnectError("refused")) is True
    assert is_upstream_transient(ValueError("bad json")) is False


# ---------------------------------------------------------------------------
# run_smart_memory_task raises (transient) vs swallows (definitive)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.memory_system = MagicMock()
    ctx.profile_memory = MagicMock()
    ctx.args = MagicMock()
    ctx.args.smart_memory = 0.5
    return ctx


# Passes the fast-abort keyword gate ("always", "my", "use").
_EPISODE = "USER: remember I always use restic for my backups.\nAI: Noted."


@pytest.mark.asyncio
async def test_transient_failure_raises_retryable(mock_context):
    agent = GhostAgent(mock_context)
    mock_context.llm_client.chat_completion.side_effect = _http_error(503)
    with pytest.raises(RetryableConsolidationError):
        await agent.run_smart_memory_task(_EPISODE, "test-model", 0.5)
    mock_context.memory_system.add.assert_not_called()


@pytest.mark.asyncio
async def test_timeout_raises_retryable(mock_context):
    agent = GhostAgent(mock_context)
    mock_context.llm_client.chat_completion.side_effect = httpx.ReadTimeout("slow")
    with pytest.raises(RetryableConsolidationError):
        await agent.run_smart_memory_task(_EPISODE, "test-model", 0.5)


@pytest.mark.asyncio
async def test_definitive_failure_still_swallowed(mock_context):
    # A 4xx would fail identically on re-run: keep the log-and-drop path.
    agent = GhostAgent(mock_context)
    mock_context.llm_client.chat_completion.side_effect = _http_error(400)
    await agent.run_smart_memory_task(_EPISODE, "test-model", 0.5)  # no raise
    mock_context.memory_system.add.assert_not_called()


# ---------------------------------------------------------------------------
# process_journal_queue re-queue / cap / non-transient behavior
# ---------------------------------------------------------------------------

def _drain_agent(mock_context, tmp_path):
    mock_context.journal = MemoryJournal(tmp_path)
    # Idle long enough that respect_idle never suspends the drain.
    mock_context.last_activity_time = (
        datetime.datetime.now() - datetime.timedelta(seconds=600)
    )
    return GhostAgent(mock_context)


@pytest.mark.asyncio
async def test_transient_item_is_requeued_with_retry_count(mock_context, tmp_path):
    agent = _drain_agent(mock_context, tmp_path)
    agent.run_smart_memory_task = AsyncMock(
        side_effect=RetryableConsolidationError("HTTP 503"),
    )
    mock_context.journal.append("smart_memory", {"text": "t", "model": "m"})

    await agent.process_journal_queue()

    # The transient-failed item is requeued to the overflow head (drained
    # first next cycle), so it surfaces via a drain, not len(load()).
    assert mock_context.journal.pending_count() == 1
    remaining = mock_context.journal.pop_all()
    assert len(remaining) == 1
    assert remaining[0]["type"] == "smart_memory"
    assert remaining[0]["retries"] == 1
    assert remaining[0]["data"]["text"] == "t"


@pytest.mark.asyncio
async def test_requeued_item_dropped_after_cap(mock_context, tmp_path):
    agent = _drain_agent(mock_context, tmp_path)
    agent.run_smart_memory_task = AsyncMock(
        side_effect=RetryableConsolidationError("HTTP 503"),
    )
    mock_context.journal.push_front([{
        "type": "smart_memory", "data": {"text": "t", "model": "m"},
        "retries": JOURNAL_MAX_RETRIES,
    }])

    await agent.process_journal_queue()

    assert mock_context.journal.load() == []


@pytest.mark.asyncio
async def test_non_transient_error_not_requeued(mock_context, tmp_path):
    agent = _drain_agent(mock_context, tmp_path)
    agent.run_smart_memory_task = AsyncMock(side_effect=ValueError("boom"))
    mock_context.journal.append("smart_memory", {"text": "t", "model": "m"})

    await agent.process_journal_queue()  # must not raise

    assert mock_context.journal.load() == []


@pytest.mark.asyncio
async def test_successful_item_processed_and_cleared(mock_context, tmp_path):
    agent = _drain_agent(mock_context, tmp_path)
    agent.run_smart_memory_task = AsyncMock(return_value=None)
    mock_context.journal.append("smart_memory", {"text": "t", "model": "m"})

    await agent.process_journal_queue()

    assert mock_context.journal.load() == []
    agent.run_smart_memory_task.assert_awaited_once()


# ---------------------------------------------------------------------------
# TAKE/ACK lifecycle: a COMPLETED drain must not replay on restart
# ---------------------------------------------------------------------------
# 2026-07-29: the drain never acked, so the last batch of a busy period sat
# "in-flight" until the next non-empty pop_all — which the pending_count()
# idle gate may never issue. recover_inflight() then replayed the fully
# consolidated batch at EVERY restart (six deploy restarts re-consolidated
# the same items up to 6x overnight). The drain now acks each item on a
# terminal disposition (consolidated or deliberately dropped); re-queued
# items keep their staged copy as the crash backstop.

import asyncio


@pytest.mark.asyncio
async def test_completed_drain_leaves_no_inflight_replay(mock_context, tmp_path):
    agent = _drain_agent(mock_context, tmp_path)
    agent.run_smart_memory_task = AsyncMock(return_value=None)
    mock_context.journal.append("smart_memory", {"text": "a", "model": "m"})
    mock_context.journal.append("smart_memory", {"text": "b", "model": "m"})

    await agent.process_journal_queue()

    # The staging file is fully acked away — nothing left to "recover".
    assert mock_context.journal.inflight() == []
    # Simulated restart: a fresh journal on the same dir must find nothing.
    revived = MemoryJournal(tmp_path)
    assert revived.pending_count() == 0


@pytest.mark.asyncio
async def test_midbatch_interrupt_replays_only_unprocessed(mock_context, tmp_path):
    agent = _drain_agent(mock_context, tmp_path)
    # First item consolidates; the kill lands on the second (CancelledError
    # is a BaseException — exactly what a deploy SIGTERM produces mid-drain).
    agent.run_smart_memory_task = AsyncMock(
        side_effect=[None, asyncio.CancelledError()],
    )
    mock_context.journal.append("smart_memory", {"text": "done", "model": "m"})
    mock_context.journal.append("smart_memory", {"text": "interrupted", "model": "m"})

    with pytest.raises(asyncio.CancelledError):
        await agent.process_journal_queue()

    # Restart: ONLY the unprocessed item comes back — at-least-once for
    # unfinished work, exactly-once for finished work.
    revived = MemoryJournal(tmp_path)
    assert revived.pending_count() == 1
    remaining = revived.pop_all()
    assert [i["data"]["text"] for i in remaining] == ["interrupted"]


@pytest.mark.asyncio
async def test_terminal_drop_is_acked_not_replayed(mock_context, tmp_path):
    agent = _drain_agent(mock_context, tmp_path)
    agent.run_smart_memory_task = AsyncMock(side_effect=ValueError("poison"))
    mock_context.journal.append("smart_memory", {"text": "t", "model": "m"})

    await agent.process_journal_queue()

    # Deliberately dropped → acked → a restart must not resurrect it forever.
    revived = MemoryJournal(tmp_path)
    assert revived.pending_count() == 0


@pytest.mark.asyncio
async def test_transient_requeue_single_copy_after_restart(mock_context, tmp_path):
    agent = _drain_agent(mock_context, tmp_path)
    agent.run_smart_memory_task = AsyncMock(
        side_effect=RetryableConsolidationError("HTTP 503"),
    )
    mock_context.journal.append("smart_memory", {"text": "t", "model": "m"})

    await agent.process_journal_queue()

    # Re-queued (not acked): the staged copy plus the queue copy must
    # de-duplicate to exactly ONE item across a restart.
    revived = MemoryJournal(tmp_path)
    assert revived.pending_count() == 1
    remaining = revived.pop_all()
    assert len(remaining) == 1
    assert remaining[0]["data"]["text"] == "t"


@pytest.mark.asyncio
async def test_identical_twins_ack_one_at_a_time(mock_context, tmp_path):
    """Byte-identical items must ack ONE-for-ONE, not by value.

    `_dedup_key` is pure content and `append` does not de-duplicate, so a
    batch can stage twins under one key. A set-based partial ack removed
    BOTH staged rows when the first twin was consumed, leaving the second
    twin's ~90 s consolidation with no staging record — a kill in that
    window lost it permanently (worse than the pre-ack behaviour).
    """
    journal = MemoryJournal(tmp_path)
    item = {"type": "smart_memory", "data": {"text": "same", "model": "m"}}
    journal.append("smart_memory", {"text": "same", "model": "m"})
    journal.append("smart_memory", {"text": "same", "model": "m"})

    batch = journal.pop_all()
    assert len(batch) == 2                      # both staged, one key

    journal.ack([batch[0]])                     # first twin consolidated
    assert len(journal.inflight()) == 1         # the SECOND twin survives

    journal.ack([batch[1]])                     # second twin consolidated
    assert journal.inflight() == []


@pytest.mark.asyncio
async def test_twin_drain_loses_nothing_on_midbatch_kill(mock_context, tmp_path):
    """Functional counterpart: identical items, kill after the first."""
    agent = _drain_agent(mock_context, tmp_path)
    agent.run_smart_memory_task = AsyncMock(
        side_effect=[None, asyncio.CancelledError()],
    )
    mock_context.journal.append("smart_memory", {"text": "same", "model": "m"})
    mock_context.journal.append("smart_memory", {"text": "same", "model": "m"})

    with pytest.raises(asyncio.CancelledError):
        await agent.process_journal_queue()

    # Exactly ONE twin comes back: the consolidated one is acked, the
    # interrupted one is recovered.
    revived = MemoryJournal(tmp_path)
    assert revived.pending_count() == 1
