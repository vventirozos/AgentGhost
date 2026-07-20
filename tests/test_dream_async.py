
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.dream import Dreamer

@pytest.fixture
def mock_context():
    context = MagicMock()
    context.memory_system = MagicMock()
    context.memory_system.collection = MagicMock()
    context.llm_client = MagicMock()
    context.skill_memory = MagicMock()
    return context

@pytest.mark.asyncio
async def test_dream_async_db_calls(mock_context):
    # Setup
    dreamer = Dreamer(mock_context)

    # Use longer documents so the consolidation synthesis achieves real
    # compression (>5%). Previous short stubs like "doc1" (4 chars) were
    # expanded by the synthesis, so the compression-metrics gate correctly
    # skipped the write.
    long_doc1 = "The user has been working on a Python-based AI project for several months now"
    long_doc2 = "The user prefers Python and has experience building AI systems with it"
    long_doc3 = "User mentioned they are interested in machine learning and natural language processing"
    mock_context.memory_system.collection.get.return_value = {
        "ids": ["1", "2", "3"],
        "documents": [long_doc1, long_doc2, long_doc3]
    }

    # Mock LLM response with valid JSON in Markdown — the synthesis
    # should be shorter than merged_ids sources combined.
    llm_response_content = """
    ```json
    {
        "consolidations": [
            {
                "synthesis": "User is a Python AI developer",
                "merged_ids": ["ID:1", "ID:2"]
            }
        ],
        "heuristics": []
    }
    ```
    """
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": llm_response_content}}]
    })

    # Mock asyncio.to_thread
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # We need to simulate the return values of the threaded calls
        # 1. collection.get -> returns dict
        # 2. memory.add -> returns None
        # 3. collection.delete -> returns None

        async def side_effect(func, *args, **kwargs):
            if func == mock_context.memory_system.collection.get:
                return {
                    "ids": ["1", "2", "3", "4"],
                    "documents": [long_doc1, long_doc2, long_doc3, "doc4"]
                }
            return None

        mock_to_thread.side_effect = side_effect

        await dreamer.dream()

        # Verify to_thread usage

        # Check get() was wrapped
        # args[0] is the function
        calls = mock_to_thread.call_args_list

        # We expect at least:
        # 1. collection.get
        # 2. memory.add (if synthesis happened)
        # 3. collection.delete (if synthesis happened)

        func_calls = [call.args[0] for call in calls]

        assert mock_context.memory_system.collection.get in func_calls
        assert mock_context.memory_system.add in func_calls
        assert mock_context.memory_system.collection.delete in func_calls

@pytest.mark.asyncio
async def test_dream_robust_json_parsing(mock_context):
    dreamer = Dreamer(mock_context)
    
    # Mock DB get to return enough docs
    mock_context.memory_system.collection.get.return_value = {
        "ids": ["1", "2", "3", "4"],
        "documents": ["d1", "d2", "d3", "d4"]
    }
    
    # Mock LLM response with Markdown code blocks (which json.loads would fail on)
    llm_response_content = "Here is the JSON:\n```json\n{\"consolidations\": []}\n```"
    
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": llm_response_content}}]
    })
    
    # We don't care about async db calls here, just that it doesn't crash on JSON
    # But since we haven't implemented async yet, the code calls collection.get synchronously.
    # That's fine for this test if we Mock correctly.
    
    result = await dreamer.dream()

    # If it failed to parse, it would return "Dream failed: ..."
    assert "Dream failed" not in result
    assert "Dream Complete" in result or "Dream cycle complete" in result


@pytest.mark.asyncio
async def test_dream_passes_timeout_to_background_llm(mock_context):
    """The REM dream LLM call must pass an explicit timeout — unbounded
    `chat_completion` calls were hanging the biological watchdog."""
    dreamer = Dreamer(mock_context)

    mock_context.memory_system.collection.get.return_value = {
        "ids": ["1", "2", "3", "4"],
        "documents": ["d1", "d2", "d3", "d4"]
    }
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": '{"consolidations": [], "heuristics": []}'}}]
    })

    await dreamer.dream()

    mock_context.llm_client.chat_completion.assert_awaited_once()
    kwargs = mock_context.llm_client.chat_completion.await_args.kwargs
    assert kwargs.get("is_background") is True
    assert kwargs.get("use_worker") is True
    # Explicit bounded timeout — the exact value is an implementation
    # detail, but it must be a positive finite number, not None.
    assert isinstance(kwargs.get("timeout"), (int, float))
    assert kwargs["timeout"] > 0


@pytest.mark.asyncio
async def test_dream_logs_completion_success(mock_context):
    """A successful dream must emit a pretty_log so the REM cycle's outcome
    is visible in the console stream, not just returned as a silent string."""
    dreamer = Dreamer(mock_context)

    mock_context.memory_system.collection.get.return_value = {
        "ids": ["1", "2", "3", "4"],
        "documents": ["d1", "d2", "d3", "d4"]
    }
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": '{"consolidations": [], "heuristics": []}'}}]
    })

    with patch("ghost_agent.core.dream.pretty_log") as mock_log:
        result = await dreamer.dream()

    assert "Dream Complete" in result
    completion_logs = [
        call for call in mock_log.call_args_list
        if len(call.args) >= 2 and "Dream Complete" in str(call.args[1])
    ]
    assert completion_logs, "expected a pretty_log call carrying the completion message"


@pytest.mark.asyncio
async def test_dream_logs_error_on_llm_failure(mock_context):
    """A dream whose LLM call raises must surface the error via pretty_log
    instead of returning a silent string."""
    dreamer = Dreamer(mock_context)

    mock_context.memory_system.collection.get.return_value = {
        "ids": ["1", "2", "3", "4"],
        "documents": ["d1", "d2", "d3", "d4"]
    }
    mock_context.llm_client.chat_completion = AsyncMock(
        side_effect=RuntimeError("upstream 500")
    )

    with patch("ghost_agent.core.dream.pretty_log") as mock_log:
        result = await dreamer.dream()

    assert "Dream error" in result
    error_logs = [
        call for call in mock_log.call_args_list
        if len(call.args) >= 2 and "Dream error" in str(call.args[1])
    ]
    assert error_logs, "expected a pretty_log call carrying the error message"
    # At least one error log must be at ERROR level.
    assert any(call.kwargs.get("level") == "ERROR" for call in error_logs)


@pytest.mark.asyncio
async def test_dream_skips_when_fragment_set_unchanged(mock_context):
    """Idempotency: a second REM cycle over the exact same auto-memory set
    must short-circuit and NOT call the LLM. Prevents the observed 30-min
    loop of `Dream Complete. 0/0` when no new memories have arrived."""
    dreamer = Dreamer(mock_context)

    fragment_ids = ["1", "2", "3", "4"]
    mock_context.memory_system.collection.get.return_value = {
        "ids": fragment_ids,
        "documents": ["d1", "d2", "d3", "d4"],
    }
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": '{"consolidations": [], "heuristics": []}'}}]
    })

    # First run — the LLM must be called and the fragment key cached.
    result1 = await dreamer.dream()
    assert "Dream Complete" in result1
    assert mock_context.llm_client.chat_completion.await_count == 1
    assert getattr(mock_context, "_last_dream_fragment_ids", None) == frozenset(fragment_ids)

    # Second run with identical ids — must skip the LLM call entirely.
    result2 = await dreamer.dream()
    assert "Skipping REM" in result2
    assert "fragment set unchanged" in result2
    assert mock_context.llm_client.chat_completion.await_count == 1


@pytest.mark.asyncio
async def test_dream_runs_when_fragment_set_changes(mock_context):
    """The idempotency guard must not block legitimate re-runs once enough
    new auto-memories have arrived. DELTA-AWARE since 2026-07-20: a single
    new fragment is below REDREAM_MIN_NEW_FRAGMENTS (3) and now SKIPS (that
    reopen-on-any-change behavior was the overnight heuristic-churn engine);
    three fresh fragments re-dream."""
    dreamer = Dreamer(mock_context)

    mock_context.memory_system.collection.get.return_value = {
        "ids": ["1", "2", "3"],
        "documents": ["d1", "d2", "d3"],
    }
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": '{"consolidations": [], "heuristics": []}'}}]
    })

    await dreamer.dream()
    assert mock_context.llm_client.chat_completion.await_count == 1

    # ONE new fragment — below the delta threshold → skip, naming the count.
    mock_context.memory_system.collection.get.return_value = {
        "ids": ["1", "2", "3", "NEW"],
        "documents": ["d1", "d2", "d3", "new"],
    }
    result = await dreamer.dream()
    assert "only 1 new fragment" in result
    assert mock_context.llm_client.chat_completion.await_count == 1

    # THREE new fragments vs the last successful cycle → re-dream fires.
    mock_context.memory_system.collection.get.return_value = {
        "ids": ["1", "2", "3", "NEW", "NEW2", "NEW3"],
        "documents": ["d1", "d2", "d3", "new", "new2", "new3"],
    }
    result = await dreamer.dream()
    assert "Dream Complete" in result
    assert mock_context.llm_client.chat_completion.await_count == 2


@pytest.mark.asyncio
async def test_dream_error_does_not_poison_idempotency_cache(mock_context):
    """If the LLM call raises, the fragment key must NOT be cached — the
    next cycle should retry rather than skipping a legitimate dream."""
    dreamer = Dreamer(mock_context)

    mock_context.memory_system.collection.get.return_value = {
        "ids": ["1", "2", "3"],
        "documents": ["d1", "d2", "d3"],
    }
    mock_context.llm_client.chat_completion = AsyncMock(
        side_effect=RuntimeError("upstream down")
    )

    result = await dreamer.dream()
    assert "Dream error" in result
    # Code must not have assigned a frozenset to the cache attribute.
    # (On MagicMock, attribute access autocreates a child mock — so
    # checking `is None` wouldn't work; check the type instead.)
    cached = getattr(mock_context, "_last_dream_fragment_ids", None)
    assert not isinstance(cached, frozenset)


@pytest.mark.asyncio
async def test_dream_logs_low_entropy_skip(mock_context):
    """The <3 auto-memories early-return path must also log its outcome."""
    dreamer = Dreamer(mock_context)

    mock_context.memory_system.collection.get.return_value = {
        "ids": ["1"],
        "documents": ["only one fragment"]
    }

    with patch("ghost_agent.core.dream.pretty_log") as mock_log:
        result = await dreamer.dream()

    assert "Not enough entropy" in result
    skip_logs = [
        call for call in mock_log.call_args_list
        if len(call.args) >= 2 and "Not enough entropy" in str(call.args[1])
    ]
    assert skip_logs, "expected a pretty_log call for the low-entropy skip path"
