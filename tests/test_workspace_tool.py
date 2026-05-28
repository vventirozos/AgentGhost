"""Tests for the workspace and workspace_track tools (read + write
paths into the workspace continuity module)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.workspace import WorkspaceModel
from ghost_agent.tools.workspace import tool_workspace
from ghost_agent.tools.workspace_track import tool_workspace_track


# ---------------------------------------------------------------------
# workspace (read)
# ---------------------------------------------------------------------


async def test_summary_default_action(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    wm.record_task_outcome(job_id="j", task_name="cron", outcome="passed")
    out = await tool_workspace(workspace_model=wm)
    assert "My workspace right now" in out
    assert "Tracked files: 0" in out
    assert "cron" in out or "task" in out.lower()


async def test_summary_matches_explicit(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    wm.record_task_outcome(job_id="j", task_name="hi", outcome="passed")
    default = await tool_workspace(workspace_model=wm)
    explicit = await tool_workspace(action="summary", workspace_model=wm)
    assert default == explicit


async def test_stats_action(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    wm.record_research_artifact(url="https://example.org/a", source="deep_research")
    out = await tool_workspace(action="stats", workspace_model=wm)
    assert "URLs already pulled: 1" in out
    assert "Workspace events on file: 1" in out


async def test_files_action_empty(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    out = await tool_workspace(action="files", workspace_model=wm)
    assert "not watching" in out.lower()


async def test_files_action_with_tracked(tmp_path: Path):
    target = tmp_path / "t.py"
    target.write_text("hi")
    wm = WorkspaceModel(tmp_path)
    wm.track_file(str(target), label="entrypoint")
    out = await tool_workspace(action="files", workspace_model=wm)
    assert "t.py" in out
    assert "(entrypoint)" in out


async def test_changes_action_reports_modification(tmp_path: Path):
    target = tmp_path / "x.py"
    target.write_text("v1")
    wm = WorkspaceModel(tmp_path)
    wm.track_file(str(target))
    # First scan via the changes action — should report "newly tracked".
    out1 = await tool_workspace(action="changes", workspace_model=wm)
    assert "newly tracked" in out1
    # Modify, then re-scan.
    target.write_text("v2 is longer")
    out2 = await tool_workspace(action="changes", workspace_model=wm)
    assert "modified" in out2


async def test_changes_action_no_tracked(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    out = await tool_workspace(action="changes", workspace_model=wm)
    assert "no tracked files" in out.lower()


async def test_changes_action_falls_back_to_activity_log(tmp_path: Path):
    """When the wake-up prefix has already consumed the diff (snapshot
    advanced), the changes action should still surface what changed
    via the activity log."""
    target = tmp_path / "code.py"
    target.write_text("v1")
    wm = WorkspaceModel(tmp_path)
    wm.track_file(str(target))
    # Simulate the wake-up scan that runs at turn start: this advances
    # the snapshot AND logs a file_changed event.
    wm.scan_tracked()
    # Same turn: the explicit changes tool finds no diff because the
    # snapshot is current — but the activity log still holds it.
    out = await tool_workspace(action="changes", workspace_model=wm)
    assert "code.py" in out
    assert "Recent file-change events" in out


async def test_tasks_action(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    wm.record_task_outcome(job_id="J1", task_name="cron-1", outcome="passed")
    wm.record_task_outcome(job_id="J2", task_name="cron-2", outcome="failed", error="boom")
    out = await tool_workspace(action="tasks", workspace_model=wm)
    assert "cron-1" in out
    assert "cron-2" in out


async def test_research_action(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    wm.record_research_artifact(url="https://paper.org/a", source="deep_research")
    out = await tool_workspace(action="research", workspace_model=wm)
    assert "paper.org/a" in out


async def test_commands_action(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    wm.record_command_outcome(command="pytest -k x", exit_code=0, duration_seconds=3.0)
    out = await tool_workspace(action="commands", workspace_model=wm)
    assert "pytest" in out


async def test_narrative_action_empty_and_present(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    out = await tool_workspace(action="narrative", workspace_model=wm)
    assert "No workspace narrative" in out
    wm.narrative.path.parent.mkdir(parents=True, exist_ok=True)
    wm.narrative.path.write_text("the project is humming along.")
    out2 = await tool_workspace(action="narrative", workspace_model=wm)
    assert "humming along" in out2


async def test_recent_action_mixed_kinds(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    wm.record_task_outcome(job_id="j", task_name="t", outcome="passed")
    wm.record_research_artifact(url="https://x.org/a", source="deep_research")
    out = await tool_workspace(action="recent", workspace_model=wm)
    assert "[task_outcome]" in out
    assert "[research]" in out


async def test_recent_limit_clamped(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    for i in range(60):
        wm.note(f"n{i}")
    out = await tool_workspace(action="recent", limit=999, workspace_model=wm)
    # Cap is 50; we should see at most that many lines.
    assert out.count("[note]") <= 50


async def test_disabled_workspace_model_graceful(tmp_path: Path):
    wm = WorkspaceModel(tmp_path, enabled=False)
    out = await tool_workspace(workspace_model=wm)
    assert "unavailable" in out.lower()


async def test_none_workspace_model_graceful():
    out = await tool_workspace(workspace_model=None)
    assert "unavailable" in out.lower()


async def test_invalid_action(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    out = await tool_workspace(action="explode", workspace_model=wm)
    assert "SYSTEM ERROR" in out


# ---------------------------------------------------------------------
# workspace_track (write)
# ---------------------------------------------------------------------


async def test_track_and_untrack(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    target = tmp_path / "f.py"
    target.write_text("hi")
    added = await tool_workspace_track(action="track", path=str(target), label="main", workspace_model=wm)
    assert "Now tracking" in added
    assert len(wm.state.tracked_files()) == 1

    removed = await tool_workspace_track(action="untrack", path=str(target), workspace_model=wm)
    assert "No longer tracking" in removed
    assert wm.state.tracked_files() == []


async def test_track_missing_path_returns_error(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    out = await tool_workspace_track(action="track", workspace_model=wm)
    assert "SYSTEM ERROR" in out


async def test_untrack_unknown_path(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    out = await tool_workspace_track(action="untrack", path="/nope", workspace_model=wm)
    assert "not on the watchlist" in out


async def test_note_records_event(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    out = await tool_workspace_track(action="note", text="started benchmarking", workspace_model=wm)
    assert "Recorded" in out
    assert wm.activity.count(kind="note") == 1


async def test_mark_seen_dedups(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    first = await tool_workspace_track(action="mark_seen", url="https://paper.org/x", workspace_model=wm)
    assert "Marked" in first
    second = await tool_workspace_track(action="mark_seen", url="https://paper.org/x", workspace_model=wm)
    assert "already on the seen list" in second


async def test_unknown_action_returns_error(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    out = await tool_workspace_track(action="blast", workspace_model=wm)
    assert "SYSTEM ERROR" in out


async def test_disabled_workspace_model_track_graceful(tmp_path: Path):
    wm = WorkspaceModel(tmp_path, enabled=False)
    out = await tool_workspace_track(action="track", path="/x", workspace_model=wm)
    assert "unavailable" in out.lower()


# ---------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------


def test_workspace_tools_in_definitions():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    assert "workspace" in names
    assert "workspace_track" in names
    spec = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "workspace")
    enum = spec["function"]["parameters"]["properties"]["action"]["enum"]
    assert set(enum) >= {
        "summary", "stats", "files", "changes",
        "tasks", "research", "commands", "narrative", "recent",
    }


def test_workspace_dispatch_lambdas_bind_workspace_model(tmp_path: Path):
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
    assert "workspace" in tools
    assert "workspace_track" in tools
    assert callable(tools["workspace"])
    assert callable(tools["workspace_track"])
