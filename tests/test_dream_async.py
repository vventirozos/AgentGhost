"""`Dreamer.dream()` after §4NM: housekeeping only. The REM path (the
consolidation call, heuristics → lessons, the fragment cache) is gone, so
what remains to pin is the pass's outcome surface — the message, the
`last_dream_outcome` stamp, and that no model call is ever made."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.dream import Dreamer


@pytest.fixture
def mock_context():
    context = MagicMock()
    context.memory_system = MagicMock()
    context.memory_system.collection = MagicMock()
    context.llm_client = MagicMock()
    context.llm_client.chat_completion = AsyncMock()
    return context


def _housekeeping_stubs(dreamer, *, manifests=0, maintenance=""):
    return (
        patch.object(dreamer, "_backfill_file_manifests", AsyncMock(return_value=manifests)),
        patch.object(dreamer, "_daily_store_care", AsyncMock(return_value=None)),
        patch.object(dreamer, "_maintenance_due", MagicMock(return_value=bool(maintenance))),
        patch.object(dreamer, "_store_maintenance", AsyncMock(return_value=maintenance)),
    )


@pytest.mark.asyncio
async def test_dream_without_a_memory_system_is_an_error_outcome(mock_context):
    mock_context.memory_system = None
    dreamer = Dreamer(mock_context)

    result = await dreamer.dream()

    assert result == "Memory system not available."
    assert {k: dreamer.last_dream_outcome[k] for k in ("phase", "side_output")} == {"phase": "error", "side_output": False}
    mock_context.llm_client.chat_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_dream_is_housekeeping_only_and_never_calls_the_model(mock_context):
    """The idle cycle must not reach the model."""
    dreamer = Dreamer(mock_context)
    stubs = _housekeeping_stubs(dreamer)
    with stubs[0], stubs[1] as care, stubs[2], stubs[3] as maint, \
            patch("ghost_agent.core.dream.pretty_log"):
        result = await dreamer.dream()

    assert result.startswith("Housekeeping cycle done")
    mock_context.llm_client.chat_completion.assert_not_awaited()
    care.assert_awaited_once()                 # every cycle, no gate
    maint.assert_not_awaited()                 # not due → not run
    assert {k: dreamer.last_dream_outcome[k] for k in ("phase", "side_output")} == {"phase": "skipped", "side_output": False}


@pytest.mark.asyncio
async def test_dream_reports_what_housekeeping_produced(mock_context):
    """`side_output` and the counts in the message follow the housekeeping
    steps, so the ledger can tell a productive cycle from an empty one."""
    dreamer = Dreamer(mock_context)
    stubs = _housekeeping_stubs(dreamer, manifests=2, maintenance=" graph 3 edges pruned")
    with stubs[0], stubs[1], stubs[2], stubs[3] as maint, \
            patch("ghost_agent.core.dream.pretty_log"):
        result = await dreamer.dream()

    maint.assert_awaited_once()
    assert "Described 2 project file(s)." in result
    assert "Store care: graph 3 edges pruned" in result
    assert {k: dreamer.last_dream_outcome[k] for k in ("phase", "side_output")} == {"phase": "skipped", "side_output": True}


@pytest.mark.asyncio
async def test_a_failing_housekeeping_step_does_not_fail_the_cycle(mock_context):
    """Manifest backfill and store maintenance are each wrapped: a raise in
    one still lets the store care run and the cycle conclude."""
    dreamer = Dreamer(mock_context)
    stubs = _housekeeping_stubs(dreamer, maintenance="x")
    with patch.object(dreamer, "_backfill_file_manifests", AsyncMock(side_effect=RuntimeError("boom"))), \
            stubs[1] as care, stubs[2], \
            patch.object(dreamer, "_store_maintenance", AsyncMock(side_effect=RuntimeError("boom"))), \
            patch("ghost_agent.core.dream.pretty_log"):
        result = await dreamer.dream()

    care.assert_awaited_once()
    assert result.startswith("Housekeeping cycle done")
    assert {k: dreamer.last_dream_outcome[k] for k in ("phase", "side_output")} == {"phase": "skipped", "side_output": False}
