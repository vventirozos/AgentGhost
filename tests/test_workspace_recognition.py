"""build_workspace_prefix — composition, markers, char cap, strip."""

from pathlib import Path

from ghost_agent.workspace.activity import WorkspaceActivity
from ghost_agent.workspace.recognition import (
    WORKSPACE_PREFIX_CLOSE,
    WORKSPACE_PREFIX_OPEN,
    build_workspace_prefix,
    strip_workspace_prefix,
)
from ghost_agent.workspace.schema import WorkspaceEvent
from ghost_agent.workspace.state import WorkspaceStateThread


def test_empty_inputs_produce_empty_prefix():
    assert build_workspace_prefix(activity=None, state=None) == ""


def test_prefix_contains_markers_and_section_header(tmp_path: Path):
    state = WorkspaceStateThread(tmp_path)
    state.track_file(str(tmp_path / "f.py"))
    state.touch_session()
    text = build_workspace_prefix(activity=None, state=state)
    assert WORKSPACE_PREFIX_OPEN in text
    assert WORKSPACE_PREFIX_CLOSE in text
    assert "WORKSPACE STATE" in text


def test_prefix_surfaces_file_changes(tmp_path: Path):
    state = WorkspaceStateThread(tmp_path)
    state.touch_session()
    changes = [{"path": "/x.py", "label": "main", "change": "modified (+12 bytes)"}]
    text = build_workspace_prefix(activity=None, state=state, file_changes=changes)
    assert "/x.py" in text
    assert "modified" in text


def test_prefix_surfaces_recent_activity(tmp_path: Path):
    act = WorkspaceActivity(tmp_path)
    act.append(WorkspaceEvent(kind="task_outcome", summary="cron-1: passed"))
    text = build_workspace_prefix(activity=act, state=None)
    assert "cron-1: passed" in text
    assert "[task_outcome]" in text


def test_prefix_includes_narrative_first(tmp_path: Path):
    text = build_workspace_prefix(
        activity=None, state=None, narrative="My workspace is small and tidy.",
    )
    # Narrative section header precedes anything else.
    assert "running summary" in text.lower()
    assert "small and tidy" in text


def test_prefix_char_cap_enforced(tmp_path: Path):
    act = WorkspaceActivity(tmp_path)
    for i in range(50):
        act.append(WorkspaceEvent(kind="note", summary="x" * 200))
    text = build_workspace_prefix(activity=act, state=None, max_chars=300)
    assert len(text) < 600  # markers + truncated body


def test_strip_removes_block_cleanly(tmp_path: Path):
    act = WorkspaceActivity(tmp_path)
    act.append(WorkspaceEvent(kind="note", summary="hello"))
    block = build_workspace_prefix(activity=act, state=None)
    full = block + "BASE PROMPT"
    stripped = strip_workspace_prefix(full)
    assert stripped == "BASE PROMPT"
    # Idempotent when no block is present.
    assert strip_workspace_prefix("plain text") == "plain text"
