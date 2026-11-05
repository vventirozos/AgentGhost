"""Logging-contract integration tests.

We don't just verify that lines are emitted — we verify the CONTRACT
that monitoring depends on:

  * Every uplift line starts with the ``Metacog <Subsystem>`` title
    so ``grep "metacog "`` reliably catches them all.
  * The validator events carry their primary identifier (tool,
    action) as a key=value pair.
  * Lifetime counters bump on every relevant event.
  * The shutdown summary fires once with the right field set.
  * The bundle counts validator blocks so the summary is accurate.
    (§4PJ: arbitrations and confidence counts went with the arbiter and
    the confidence score; §4PT: host signals and replans with host
    telemetry and the ReplanBridge.)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.core.metacog import MetacogBundle
from ghost_agent.core.metacog_log import Subsystem


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

def _args(**overrides):
    a = argparse.Namespace(
        enable_metacog=True,
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


@pytest.fixture
def bundle(tmp_path):
    ctx = MagicMock()
    ctx.memory_dir = tmp_path
    ctx.args = MagicMock()
    ctx.args.model = "test"
    ctx.llm_client = MagicMock()
    if hasattr(ctx.llm_client, "get_embeddings"):
        del ctx.llm_client.get_embeddings
    ctx.project_store = None
    ctx.current_project_id = None
    return MetacogBundle.from_args(ctx, _args())


# ──────────────────────────────────────────────────────────────────────
# Counter API
# ──────────────────────────────────────────────────────────────────────

class TestCounters:
    def test_count_noop_when_disabled(self):
        b = MetacogBundle()  # enabled=False
        b.count(validator_block=True)
        assert b._ctr_validator_blocks == 0

    def test_count_bumps_when_enabled(self, bundle):
        bundle.count(validator_block=True)
        bundle.count()
        assert bundle._ctr_validator_blocks == 1

# ──────────────────────────────────────────────────────────────────────
# Shutdown summary
# ──────────────────────────────────────────────────────────────────────

class TestShutdownSummary:
    @pytest.mark.asyncio
    async def test_shutdown_emits_summary(self, bundle):
        # Pre-populate some counters
        bundle.count(validator_block=True)

        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            await bundle.shutdown()
        # Find the summary call
        summary_calls = [c for c in pl.call_args_list
                         if c.args and c.args[0] == "Metacog Summary"]
        assert len(summary_calls) == 1
        content = summary_calls[0].args[1]
        assert "arbitrations=" not in content and "conf_" not in content   # §4PJ
        assert content == "validator_blocks=1"      # §4PT: the only field left

    @pytest.mark.asyncio
    async def test_shutdown_summary_robust_to_zero_counters(self, bundle):
        # No activity — summary should still fire (it's the record of
        # an idle session)
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            await bundle.shutdown()
        summary_calls = [c for c in pl.call_args_list
                         if c.args and c.args[0] == "Metacog Summary"]
        assert len(summary_calls) == 1
        content = summary_calls[0].args[1]
        assert "validator_blocks=0" in content


# ──────────────────────────────────────────────────────────────────────
# Validator log contract (smoke — full validator tests live separately)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_shell_validator_block_emits_structured_log(tmp_path):
    from ghost_agent.tools.execute import tool_execute
    bundle = MagicMock()
    bundle.enabled = True
    with patch("ghost_agent.utils.logging.pretty_log") as pl:
        await tool_execute(
            command="rm -rf /",
            sandbox_dir=tmp_path,
            sandbox_manager=MagicMock(),
            memory_dir=tmp_path,
            _metacog_bundle=bundle,
        )
    valid_calls = [c for c in pl.call_args_list
                   if c.args and c.args[0] == "Metacog Valid"]
    assert len(valid_calls) == 1
    content = valid_calls[0].args[1]
    assert "verdict=block" in content
    assert "tool=shell" in content
    # Reason should be present and not empty
    assert "reason=" in content


@pytest.mark.asyncio
async def test_a_real_bundle_counts_a_shell_block_into_the_summary(tmp_path):
    """§4PT: the counter the summary reports, through the real tool and
    the real bundle (the tests above use a MagicMock bundle)."""
    from ghost_agent.tools.execute import tool_execute
    b = MetacogBundle(enabled=True)
    await tool_execute(command="rm -rf /", sandbox_dir=tmp_path, sandbox_manager=MagicMock(),
                       memory_dir=tmp_path, _metacog_bundle=b)
    assert b._ctr_validator_blocks == 1
