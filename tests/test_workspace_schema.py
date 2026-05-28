"""Dataclass roundtrip tests for the workspace schema.

Mirrors test_selfhood_schema.py in shape. The on-disk format is the
load-bearing contract — these tests pin it down."""

from pathlib import Path

from ghost_agent.workspace.schema import (
    CommandOutcome,
    FileSnapshot,
    ResearchArtifact,
    SCHEMA_VERSION,
    TaskOutcome,
    TrackedFile,
    WorkspaceEvent,
    WorkspaceState,
)


def test_schema_version_constant():
    assert SCHEMA_VERSION == "v1"


def test_file_snapshot_roundtrip():
    snap = FileSnapshot(path="/x", size=42, mtime_ns=1, digest="abc", exists=True)
    d = snap.to_dict()
    restored = FileSnapshot.from_dict(d)
    assert restored.path == snap.path
    assert restored.size == 42
    assert restored.exists is True


def test_task_outcome_roundtrip():
    t = TaskOutcome(job_id="J1", task_name="hello", outcome="passed", duration_seconds=1.5)
    d = t.to_dict()
    restored = TaskOutcome.from_dict(d)
    assert restored.job_id == "J1"
    assert restored.outcome == "passed"
    assert restored.duration_seconds == 1.5


def test_research_artifact_roundtrip():
    a = ResearchArtifact(url="https://example.org/x", title="Example", source="deep_research")
    d = a.to_dict()
    assert d["url"] == "https://example.org/x"
    restored = ResearchArtifact.from_dict(d)
    assert restored.title == "Example"
    assert restored.source == "deep_research"


def test_command_outcome_roundtrip():
    c = CommandOutcome(command="ls -la", exit_code=0, duration_seconds=0.1)
    d = c.to_dict()
    restored = CommandOutcome.from_dict(d)
    assert restored.command == "ls -la"
    assert restored.exit_code == 0


def test_workspace_event_jsonl_roundtrip():
    ev = WorkspaceEvent(kind="research", payload={"url": "x"}, summary="pulled x")
    line = ev.to_jsonl()
    import json
    d = json.loads(line)
    restored = WorkspaceEvent.from_dict(d)
    assert restored.kind == "research"
    assert restored.payload["url"] == "x"
    assert restored.summary == "pulled x"


def test_workspace_state_roundtrip_with_tracked_files_and_urls():
    state = WorkspaceState(
        tracked_files=[
            TrackedFile(path="/a.py", label="main",
                        last_snapshot=FileSnapshot(path="/a.py", size=1, exists=True)),
            TrackedFile(path="/b.py", label="", last_snapshot=None),
        ],
        last_session_at="2026-05-28T00:00:00Z",
        seen_urls=["https://x.org", "https://y.org"],
    )
    d = state.to_dict()
    restored = WorkspaceState.from_dict(d)
    assert restored.last_session_at == "2026-05-28T00:00:00Z"
    assert len(restored.tracked_files) == 2
    assert restored.tracked_files[0].label == "main"
    assert restored.tracked_files[0].last_snapshot is not None
    assert restored.tracked_files[0].last_snapshot.size == 1
    assert restored.tracked_files[1].last_snapshot is None
    assert "https://x.org" in restored.seen_urls


def test_workspace_state_from_dict_handles_missing_keys():
    # Tolerant of partial / older shapes — should not raise.
    s = WorkspaceState.from_dict({})
    assert s.schema_version == "v1"
    assert s.tracked_files == []
    assert s.seen_urls == []
