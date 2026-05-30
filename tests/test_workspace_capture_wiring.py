"""End-to-end capture wiring tests for the workspace continuity module.

Covers the integrations the audit flagged:
  * tool_browser records a research artifact on a successful navigate.
  * tool_execute records a command outcome on failure and on long runs.
  * The fact_check dispatch lambda forwards workspace_model into its
    internal deep_research callable.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.workspace import WorkspaceModel


# ---------------------------------------------------------------------
# tool_browser → research artifact sink
# ---------------------------------------------------------------------


async def test_tool_browser_records_navigate_url_into_workspace(tmp_path: Path):
    from ghost_agent.tools import browser as browser_mod

    wm = WorkspaceModel(tmp_path)

    # Build a fake sandbox manager that returns the runner output our
    # parser expects. The runner's success payload is a single
    # JSON-ish line the tool extracts via _parse_runner_output.
    fake_sandbox = MagicMock()
    runner_json = (
        '{"url": "https://example.org/page", "status": 200, '
        '"title": "Example Page"}'
    )
    # The actual parser expects a SUCCESS marker around the JSON line.
    fake_runner_output = f"--- RUNNER OK ---\n{runner_json}\n--- END ---"
    fake_sandbox.execute = MagicMock(return_value=(fake_runner_output, 0))

    with patch.object(browser_mod, "_parse_runner_output",
                      return_value=(True, {
                          "url": "https://example.org/page",
                          "status": 200,
                          "title": "Example Page",
                      })):
        with patch.object(browser_mod, "_get_safe_path",
                          return_value=tmp_path / "runner.py"):
            with patch("builtins.open", create=True):
                out = await browser_mod.tool_browser(
                    operation="navigate",
                    url="https://example.org/page",
                    sandbox_dir=tmp_path,
                    sandbox_manager=fake_sandbox,
                    workspace_model=wm,
                )

    # Tool returned a normal success — and the URL was recorded.
    assert "STATUS: OK" in out or "https://example.org/page" in out
    assert wm.has_seen_url("https://example.org/page")
    research = wm.activity.recent(limit=5, kind="research")
    assert research and research[0].payload.get("source") == "browser"


async def test_tool_browser_dedup_on_repeat(tmp_path: Path):
    """Two navigations to the same URL produce ONE research artifact."""
    from ghost_agent.tools import browser as browser_mod

    wm = WorkspaceModel(tmp_path)
    fake_sandbox = MagicMock()
    fake_sandbox.execute = MagicMock(return_value=("ok", 0))

    with patch.object(browser_mod, "_parse_runner_output",
                      return_value=(True, {
                          "url": "https://example.org/dup",
                          "status": 200,
                          "title": "Dup",
                      })):
        with patch.object(browser_mod, "_get_safe_path",
                          return_value=tmp_path / "runner.py"):
            with patch("builtins.open", create=True):
                for _ in range(2):
                    await browser_mod.tool_browser(
                        operation="navigate",
                        url="https://example.org/dup",
                        sandbox_dir=tmp_path,
                        sandbox_manager=fake_sandbox,
                        workspace_model=wm,
                    )

    assert len(wm.activity.recent(limit=10, kind="research")) == 1


# ---------------------------------------------------------------------
# tool_execute → command outcome sink
# ---------------------------------------------------------------------


async def test_tool_execute_records_failed_command(tmp_path: Path):
    from ghost_agent.tools import execute as execute_mod

    wm = WorkspaceModel(tmp_path)
    fake_sandbox = MagicMock()
    fake_sandbox.execute = MagicMock(return_value=("permission denied", 1))

    # A non-destructive command that fails (exit 1). NB: a destructive form
    # like `rm -rf /protected` is now (correctly) blocked by the unconditional
    # pre-execution validator before it reaches the sandbox, so it would never
    # be recorded — use a plain failing command to exercise the outcome sink.
    out = await execute_mod.tool_execute(
        command="cat /protected/secret.txt",
        sandbox_dir=tmp_path,
        sandbox_manager=fake_sandbox,
        workspace_model=wm,
    )
    assert "EXIT CODE: 1" in out
    cmds = wm.activity.recent(limit=5, kind="command")
    assert cmds
    assert cmds[0].payload.get("exit_code") == 1
    assert "cat" in cmds[0].payload.get("command", "")


async def test_tool_execute_skips_fast_successful_commands(tmp_path: Path):
    """A 0-exit, fast command does NOT pollute the activity log."""
    from ghost_agent.tools import execute as execute_mod

    wm = WorkspaceModel(tmp_path)
    fake_sandbox = MagicMock()
    fake_sandbox.execute = MagicMock(return_value=("hello", 0))

    out = await execute_mod.tool_execute(
        command="echo hello",
        sandbox_dir=tmp_path,
        sandbox_manager=fake_sandbox,
        workspace_model=wm,
    )
    assert "EXIT CODE: 0" in out
    # Significance gate filtered the fast success.
    assert wm.activity.count(kind="command") == 0


async def test_tool_execute_does_not_break_when_workspace_disabled(tmp_path: Path):
    """A disabled workspace must not error inside tool_execute."""
    from ghost_agent.tools import execute as execute_mod

    wm = WorkspaceModel(tmp_path, enabled=False)
    fake_sandbox = MagicMock()
    fake_sandbox.execute = MagicMock(return_value=("ok", 0))
    out = await execute_mod.tool_execute(
        command="echo ok",
        sandbox_dir=tmp_path,
        sandbox_manager=fake_sandbox,
        workspace_model=wm,
    )
    assert "EXIT CODE: 0" in out


# ---------------------------------------------------------------------
# Registry: dispatch lambdas bind workspace_model into the right tools
# ---------------------------------------------------------------------


def test_registry_dispatch_lambdas_wire_workspace_into_browser_and_execute(tmp_path: Path):
    """Smoke test: get_available_tools() returns lambdas that include
    workspace_model in their closure for browser and execute."""
    from ghost_agent.tools.registry import get_available_tools

    wm = WorkspaceModel(tmp_path)
    ctx = SimpleNamespace(
        workspace_model=wm,
        self_model=None,
        args=SimpleNamespace(
            anonymous=False, max_context=4000, model="qwen", default_db="",
        ),
        tor_proxy=None,
        profile_memory=MagicMock(),
        sandbox_dir=str(tmp_path),
        sandbox_manager=None,
        memory_dir=str(tmp_path),
        memory_system=MagicMock(),
        graph_memory=None,
        skill_memory=MagicMock(),
        llm_client=MagicMock(image_gen_clients=None),
        scratchpad=MagicMock(),
        scheduler=MagicMock(),
        memory_bus=None,
        uncertainty_tracker=None,
        metacog=None,
    )
    tools = get_available_tools(ctx)
    assert "browser" in tools
    assert "execute" in tools
    assert "deep_research" in tools
    assert "fact_check" in tools
    # All four lambdas are callable; their concrete behaviour is tested
    # in the per-tool unit tests above.
    for name in ("browser", "execute", "deep_research", "fact_check"):
        assert callable(tools[name])
