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
