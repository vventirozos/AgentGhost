"""Regression: ``tool_update_profile`` legacy path must NOT silently
return "SUCCESS" when a secondary-index write (vector smart_update or
graph triplet) fails.

The legacy direct path (``tool_update_profile`` when ``memory_bus``
is None) had two bare ``except`` blocks:

    if memory_system:
        try: await asyncio.to_thread(memory_system.smart_update, ...)
        except: pass

    if graph_memory:
        try: ... add_triplets ...
        except Exception: pass

    return f"SUCCESS: Profile updated."

Both swallowed real failures (embedding-model timeout, Chroma
connection blip, graph DB write failure) and reported "SUCCESS"
anyway. The agent and user both believed the fact was fully
indexed; the next semantic / graph retrieval missed it.

Fix: log a WARNING for each failure (so a tail of the agent log
shows the failure), AND return ``PARTIAL: ... index(es) lagged``
instead of ``SUCCESS`` so callers can react.
"""
import pytest
from unittest.mock import MagicMock

from ghost_agent.tools.memory import tool_update_profile


def _profile_memory():
    pm = MagicMock()
    pm.update = MagicMock(return_value="JSON updated")
    return pm


@pytest.mark.asyncio
async def test_legacy_path_full_success_returns_success():
    pm = _profile_memory()
    ms = MagicMock()
    ms.smart_update = MagicMock(return_value=None)
    gm = MagicMock()
    gm.add_triplets = MagicMock(return_value=None)
    out = await tool_update_profile(
        category="identity", key="city", value="Athens",
        profile_memory=pm, memory_system=ms, graph_memory=gm,
        memory_bus=None,
    )
    assert out.startswith("SUCCESS"), out


@pytest.mark.asyncio
async def test_legacy_path_vector_failure_returns_partial():
    pm = _profile_memory()
    ms = MagicMock()
    ms.smart_update = MagicMock(side_effect=RuntimeError("embedding upstream down"))
    gm = MagicMock()
    gm.add_triplets = MagicMock(return_value=None)
    out = await tool_update_profile(
        category="identity", key="city", value="Athens",
        profile_memory=pm, memory_system=ms, graph_memory=gm,
        memory_bus=None,
    )
    assert out.startswith("PARTIAL"), (
        f"Vector index failure must surface as PARTIAL, not SUCCESS. Got: {out!r}"
    )
    assert "vector" in out.lower()
    # JSON profile write must still have happened — that's the canonical
    # store and the index lag is best-effort.
    pm.update.assert_called_once()


@pytest.mark.asyncio
async def test_legacy_path_graph_failure_returns_partial():
    pm = _profile_memory()
    ms = MagicMock()
    ms.smart_update = MagicMock(return_value=None)
    gm = MagicMock()
    gm.add_triplets = MagicMock(side_effect=ConnectionError("graph DB unreachable"))
    out = await tool_update_profile(
        category="identity", key="city", value="Athens",
        profile_memory=pm, memory_system=ms, graph_memory=gm,
        memory_bus=None,
    )
    assert out.startswith("PARTIAL"), out
    assert "graph" in out.lower()


@pytest.mark.asyncio
async def test_legacy_path_both_indexes_failing_lists_both():
    pm = _profile_memory()
    ms = MagicMock()
    ms.smart_update = MagicMock(side_effect=RuntimeError("vector down"))
    gm = MagicMock()
    gm.add_triplets = MagicMock(side_effect=RuntimeError("graph down"))
    out = await tool_update_profile(
        category="identity", key="city", value="Athens",
        profile_memory=pm, memory_system=ms, graph_memory=gm,
        memory_bus=None,
    )
    assert out.startswith("PARTIAL"), out
    assert "vector" in out.lower()
    assert "graph" in out.lower()


@pytest.mark.asyncio
async def test_legacy_path_logs_warning_on_failure(caplog):
    """Failures must surface in the log so a tail of the agent log
    shows them. Pre-fix: bare `except: pass` left no trace."""
    import logging as _logging
    pm = _profile_memory()
    ms = MagicMock()
    ms.smart_update = MagicMock(side_effect=RuntimeError("embedding upstream down"))
    with caplog.at_level(_logging.WARNING, logger="GhostAgent"):
        await tool_update_profile(
            category="identity", key="city", value="Athens",
            profile_memory=pm, memory_system=ms, graph_memory=None,
            memory_bus=None,
        )
    msgs = [r.message for r in caplog.records]
    assert any(
        "smart_update" in m and "city" in m for m in msgs
    ), f"Expected a WARNING about smart_update failure for key=city; got {msgs}"
