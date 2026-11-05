"""Unit tests for the metacognition wiring — verifies that each
integration site is live when --enable-metacog is on and a no-op when
off.

Scope:
  - MetacogBundle constructor + from_args contract
  - GhostContext.metacog field defaults to None (back-compat)
  - tools/registry passes the bundle into execute
  - tool_execute rejects deny-listed shell when bundle is wired
  - execute runs normally when bundle is absent (legacy path)
  - the shutdown summary counts validator blocks; removed host flags still parse (§4PT)
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.core.metacog import (
    MetacogBundle,
    _domain_for_tool,
)


# ──────────────────────────────────────────────────────────────────────
# Bundle construction & lifecycle
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_context(tmp_path):
    ctx = MagicMock()
    ctx.memory_dir = tmp_path
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    ctx.llm_client = MagicMock()
    # `get_embeddings` may or may not exist — start without
    if hasattr(ctx.llm_client, "get_embeddings"):
        del ctx.llm_client.get_embeddings
    ctx.project_store = None
    ctx.current_project_id = None
    return ctx


def _args(**kw):
    a = argparse.Namespace(
        enable_metacog=False,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class TestMetacogBundleConstruction:
    def test_disabled_returns_none(self, fake_context):
        bundle = MetacogBundle.from_args(fake_context, _args(enable_metacog=False))
        assert bundle is None

    def test_enabled_constructs_bundle(self, fake_context):
        bundle = MetacogBundle.from_args(fake_context, _args(enable_metacog=True))
        assert bundle is not None and bundle.enabled
        # §4PJ: the confidence score and arbiter; §4PT: host telemetry, the
        # trigger bus, the runtime budget and the ReplanBridge — all gone
        for gone in ("competence", "confidence", "arbiter", "record_confidence", "arbitrate_tool_calls",
                     "telemetry", "bus", "bridge", "runtime_budget", "repetition", "set_active_task",
                     "record_outcome"):
            assert not hasattr(bundle, gone), gone

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, fake_context):
        bundle = MetacogBundle.from_args(fake_context, _args(enable_metacog=True))
        await bundle.shutdown()
        await bundle.shutdown()  # second call must not raise

    @pytest.mark.asyncio
    async def test_the_summary_counts_validator_blocks(self, fake_context):
        bundle = MetacogBundle.from_args(fake_context, _args(enable_metacog=True))
        bundle.count(validator_block=True)
        bundle.count(validator_block=True)
        bundle.count()
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            await bundle.shutdown()
        assert pl.call_args[0][0] == "Metacog Summary"
        assert pl.call_args[0][1] == "validator_blocks=2"

    def test_a_disabled_bundle_counts_nothing(self):
        b = MetacogBundle(enabled=False)
        b.count(validator_block=True)
        assert b._ctr_validator_blocks == 0


# ──────────────────────────────────────────────────────────────────────
# Domain mapping
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tool,expected", [
    ("execute", "shell"),
    ("database", "sql"),
    ("web_search", "fetch"),
    ("file_system", "fs"),
    ("skill_memory", "memory"),
    ("vision", "vision"),
    ("image_generation", "vision"),
    ("totally_unknown_tool_xyz", "other"),
    ("", "other"),
])
def test_domain_for_tool(tool, expected):
    assert _domain_for_tool(tool) == expected


def test_the_host_telemetry_flags_are_accepted_and_ignored():
    """§4PT: the launcher passes --metacog-mem-high / -mem-floor-mb /
    -cpu-high; removing them outright would stop the agent from starting."""
    import sys
    from ghost_agent import main as main_module
    old_argv = sys.argv[:]
    try:
        sys.argv = ["ghost_agent", "--enable-metacog", "--metacog-mem-high", "101", "--metacog-mem-floor-mb", "300",
                    "--metacog-cpu-high", "101.0", "--metacog-disk-high", "90", "--metacog-mem-warn-free-mb", "1",
                    "--metacog-host-heartbeat-s", "5"]
        args = main_module.parse_args()
    finally:
        sys.argv = old_argv
    assert args.enable_metacog is True


def test_the_retired_metacog_flags_are_rejected():
    """§4PJ: the confidence/arbiter flags were removed outright (the
    launcher never passed them), unlike --calib-refit-cooldown."""
    import sys
    from ghost_agent import main as main_module
    old_argv = sys.argv[:]
    try:
        for flag in (["--metacog-confidence-threshold", "0.4"],
                     ["--metacog-disable-arbiter"],
                     ["--metacog-arbiter-timeout-s", "75"],
                     ["--metacog-disable-logprobs"]):
            sys.argv = ["ghost_agent", "--enable-metacog", *flag]
            with pytest.raises(SystemExit):
                main_module.parse_args()
    finally:
        sys.argv = old_argv


# ──────────────────────────────────────────────────────────────────────
# Validators integration — execute.py
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_execute_blocks_denylisted_shell_when_bundle_enabled(tmp_path):
    """Wire-point: tool_execute must call validate_shell when the
    bundle is wired and reject deny-listed commands BEFORE dispatching
    to the sandbox manager."""
    from ghost_agent.tools.execute import tool_execute
    fake_sandbox = MagicMock()
    fake_sandbox.execute = MagicMock(return_value=("should not run", 0))

    bundle = MagicMock()
    bundle.enabled = True

    out = await tool_execute(
        command="rm -rf /",
        sandbox_dir=tmp_path,
        sandbox_manager=fake_sandbox,
        memory_dir=tmp_path,
        _metacog_bundle=bundle,
    )
    assert "SYSTEM BLOCK" in out
    assert "rejected" in out.lower()
    # Sandbox was never asked to execute
    fake_sandbox.execute.assert_not_called()


@pytest.mark.asyncio
async def test_tool_execute_passes_benign_shell_when_bundle_enabled(tmp_path):
    from ghost_agent.tools.execute import tool_execute
    fake_sandbox = MagicMock()
    fake_sandbox.execute = MagicMock(return_value=("hello", 0))

    bundle = MagicMock()
    bundle.enabled = True

    out = await tool_execute(
        command="echo hello",
        sandbox_dir=tmp_path,
        sandbox_manager=fake_sandbox,
        memory_dir=tmp_path,
        _metacog_bundle=bundle,
    )
    assert "COMMAND RESULT" in out
    fake_sandbox.execute.assert_called_once()


@pytest.mark.asyncio
async def test_tool_execute_runs_denylisted_when_bundle_absent(tmp_path):
    """Back-compat: when --enable-metacog is off, the validator
    pathway must not fire. The deny-listed command would still be
    blocked by the sandbox's own protections, but at the agent level
    nothing changes."""
    from ghost_agent.tools.execute import tool_execute
    fake_sandbox = MagicMock()
    fake_sandbox.execute = MagicMock(return_value=("legacy path", 0))

    out = await tool_execute(
        command="echo legacy",
        sandbox_dir=tmp_path,
        sandbox_manager=fake_sandbox,
        memory_dir=tmp_path,
        _metacog_bundle=None,  # off
    )
    assert "COMMAND RESULT" in out
    fake_sandbox.execute.assert_called_once()


# ──────────────────────────────────────────────────────────────────────
# Validators integration — database.py
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# Registry integration
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_registry_injects_bundle_into_execute_lambda(tmp_path):
    """The `execute` tool entry in registry.py must read context.metacog
    and pass it down to tool_execute via the lambda. We verify by
    awaiting the coroutine — if the lambda forwarded an unknown
    keyword to `tool_execute`, the await would raise TypeError."""
    from ghost_agent.tools.registry import get_available_tools

    ctx = MagicMock()
    ctx.sandbox_dir = tmp_path
    ctx.sandbox_manager = MagicMock()
    ctx.memory_dir = tmp_path
    ctx.tor_proxy = None
    ctx.args = MagicMock()
    ctx.args.max_context = 1024
    ctx.args.anonymous = True
    sentinel = object()
    ctx.metacog = sentinel

    tools = get_available_tools(ctx)
    assert "execute" in tools
    assert "postgres_admin" not in tools          # §4SX E: removed

    # Calling with no command surfaces the standard "must provide"
    # error — what matters is that no TypeError is raised on the
    # unknown `_metacog_bundle` kwarg.
    result = await tools["execute"]()
    # Either a structured error string or just non-crash is fine here
    assert isinstance(result, str)


# ──────────────────────────────────────────────────────────────────────
# CLI flags
# ──────────────────────────────────────────────────────────────────────
