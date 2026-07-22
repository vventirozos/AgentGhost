"""Sub-agent tool containment must FAIL CLOSED (bug-hunt 2026-07-22).

``run_subagent``'s three-gate restriction block used to sit in one broad
try/except that only ``logger.debug``'d on failure — so a registry import
error or an unexpected defs shape let the delegate run with the FULL
schema + FULL dispatch dict. A containment boundary that trips must abort
the job (``RuntimeError`` → the job registry's done-callback lands
FAILED, same contract as the timeout bound), never proceed open.

Deny-all was rejected as the fallback because it is not constructible
under the very failure being defended against: ``disabled_tools`` is a
DENYLIST (agent.py filters the schema by subtracting names from it), so
hiding everything requires enumerating every advertised name — and the
failure mode is precisely "the registry could not be enumerated".

Happy-path behaviour (successful restriction → same three gates as
before) is covered by tests/test_subagent_containment.py::
TestRunSubagentRestriction and is deliberately not duplicated here.
"""

import sys

import pytest
from unittest.mock import AsyncMock, patch

from ghost_agent.core.subagent import run_subagent

from tests.test_subagent_containment import _fake_context


class TestFailClosed:
    async def test_registry_import_failure_aborts_job(self, tmp_path):
        """If `from ..tools.registry import TOOL_DEFINITIONS` raises, the
        delegate must abort (RuntimeError → FAILED job) — it must NOT run
        with the unrestricted tool surface."""
        ctx = _fake_context(tmp_path)
        handle = AsyncMock(return_value=("done", 0, "sub-jfc1"))
        # None in sys.modules makes the runtime import raise ImportError —
        # simulates the registry being unimportable at restriction time.
        with patch("ghost_agent.core.agent.GhostAgent.handle_chat",
                   new=handle), \
             patch.dict(sys.modules, {"ghost_agent.tools.registry": None}):
            with pytest.raises(RuntimeError,
                               match="containment") as excinfo:
                await run_subagent(ctx, job_id="jfc1", task="do it",
                                   allowed_tools=["recall"], timeout_s=30)
        # The sub-agent never ran at all — no turn with the full registry.
        handle.assert_not_awaited()
        # The original failure is chained for the FAILED-job error string.
        assert isinstance(excinfo.value.__cause__, ImportError)

    async def test_bad_defs_shape_aborts_job(self, tmp_path):
        """If TOOL_DEFINITIONS has an unexpected shape (name extraction
        throws), the delegate must abort, not fail open."""
        ctx = _fake_context(tmp_path)
        handle = AsyncMock(return_value=("done", 0, "sub-jfc2"))
        # Entry without a "name" key → KeyError inside the restriction block.
        with patch("ghost_agent.core.agent.GhostAgent.handle_chat",
                   new=handle), \
             patch("ghost_agent.tools.registry.TOOL_DEFINITIONS",
                   new=[{"function": {}}]):
            with pytest.raises(RuntimeError, match="unrestricted"):
                await run_subagent(ctx, job_id="jfc2", task="do it",
                                   allowed_tools=["recall"], timeout_s=30)
        handle.assert_not_awaited()

    async def test_containment_failure_lands_failed_job(self, tmp_path):
        """End-to-end with the job registry: the raise must land the job as
        FAILED (the same landing the timeout contract uses), with the
        containment message in the error field."""
        import asyncio

        from ghost_agent.core.jobs import JobRegistry, STATUS_FAILED

        ctx = _fake_context(tmp_path)
        reg = JobRegistry()
        job = reg.register("subagent", "do it", tools=1)

        handle = AsyncMock(return_value=("done", 0, f"sub-{job.id}"))
        with patch("ghost_agent.core.agent.GhostAgent.handle_chat",
                   new=handle), \
             patch.dict(sys.modules, {"ghost_agent.tools.registry": None}):
            task = asyncio.ensure_future(
                run_subagent(ctx, job_id=job.id, task="do it",
                             allowed_tools=["recall"], timeout_s=30))
            reg.attach(job.id, task)
            with pytest.raises(RuntimeError):
                await task
            # Done-callbacks run via call_soon — let the loop drain them.
            await asyncio.sleep(0)

        landed = reg.get(job.id)
        assert landed.status == STATUS_FAILED
        assert "containment" in (landed.error or "")
        handle.assert_not_awaited()
