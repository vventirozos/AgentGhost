"""WorkspaceModel facade (workspace continuity module) — capture
hooks, dedup, narrative, stats, disabled mode.

Distinct from ``test_workspace_model.py`` which tests the OLDER
``core.workspace_model`` sandbox-state tracker. The two modules
coexist."""

from pathlib import Path

import pytest

from ghost_agent.workspace import WorkspaceModel


def test_disabled_model_no_ops_everywhere(tmp_path: Path):
    wm = WorkspaceModel(tmp_path, enabled=False)
    assert wm.build_wakeup_prefix() == ""
    assert wm.record_task_outcome(job_id="J") is None
    assert wm.record_research_artifact(url="https://x.org") is None
    assert wm.record_command_outcome(command="ls") is None
    assert wm.track_file("/x") is None
    assert wm.untrack_file("/x") is False
    assert wm.scan_tracked() == []
    assert wm.note("x") is None
    assert wm.stats() == {"enabled": False}


def test_record_task_outcome_appends_to_activity(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    t = wm.record_task_outcome(
        job_id="job-1", task_name="hello", outcome="passed",
        duration_seconds=2.0, summary="ran fine",
    )
    assert t is not None
    events = wm.activity.recent(limit=5, kind="task_outcome")
    assert len(events) == 1
    assert "hello" in events[0].summary


def test_record_research_artifact_dedups(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    first = wm.record_research_artifact(url="https://x.org/a", source="deep_research")
    assert first is not None
    second = wm.record_research_artifact(url="https://x.org/a", source="deep_research")
    assert second is None  # dedup
    third = wm.record_research_artifact(url="https://x.org/b", source="deep_research")
    assert third is not None
    research_events = wm.activity.recent(limit=10, kind="research")
    assert len(research_events) == 2


def test_record_command_outcome(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    c = wm.record_command_outcome(
        command="pytest -k smoke", exit_code=1, duration_seconds=12.3, note="flake",
    )
    assert c is not None
    assert c.exit_code == 1
    events = wm.activity.recent(limit=5, kind="command")
    assert events and "pytest" in events[0].summary


def test_track_file_then_scan_emits_change_event(tmp_path: Path):
    target = tmp_path / "watch.txt"
    target.write_text("v1")
    wm = WorkspaceModel(tmp_path)
    wm.track_file(str(target), label="config")
    changes = wm.scan_tracked()
    assert len(changes) == 1
    events = wm.activity.recent(limit=5, kind="file_changed")
    assert events and "watch.txt" in events[0].summary


def test_build_wakeup_prefix_returns_string_with_state(tmp_path: Path):
    target = tmp_path / "w.py"
    target.write_text("hi")
    wm = WorkspaceModel(tmp_path)
    wm.track_file(str(target))
    wm.mark_session_boot()
    prefix = wm.build_wakeup_prefix()
    assert "w.py" in prefix
    assert "WORKSPACE STATE" in prefix


def test_stats_shape(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    wm.record_task_outcome(job_id="j", task_name="x", outcome="passed")
    wm.record_research_artifact(url="https://x.org/a", source="deep_research")
    s = wm.stats()
    assert s["enabled"] is True
    assert s["event_count"] == 2
    assert s["seen_urls"] == 1
    assert s["event_kinds"]["task_outcome"] == 1


async def test_consolidate_narrative_template_path(tmp_path: Path):
    """Without a critique_fn the narrative falls back to the template
    renderer."""
    wm = WorkspaceModel(tmp_path)
    wm.record_task_outcome(job_id="j", task_name="cron", outcome="passed")
    text = await wm.consolidate_narrative()
    assert "cron" in text.lower() or "task" in text.lower()
    assert wm.narrative.latest() == text


async def test_consolidate_narrative_with_critique_fn(tmp_path: Path):
    async def fake_critique(prompt: str) -> str:
        return "polished workspace narrative"
    wm = WorkspaceModel(tmp_path, narrative_critique_fn=fake_critique)
    wm.record_task_outcome(job_id="j", task_name="cron", outcome="passed")
    text = await wm.consolidate_narrative()
    assert text == "polished workspace narrative"


async def test_consolidate_narrative_empty_when_no_state(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    text = await wm.consolidate_narrative()
    assert text == ""


async def test_consolidate_narrative_skips_on_unchanged_input(tmp_path: Path):
    # Idempotency guard (2026-07-13): an unchanged workspace template
    # means an unchanged workspace — no LLM call, no re-persist.
    # Observed live: identical hourly regenerations all night.
    calls = [0]

    async def fake_critique(prompt: str) -> str:
        calls[0] += 1
        return f"narrative v{calls[0]}"

    wm = WorkspaceModel(tmp_path, narrative_critique_fn=fake_critique)
    wm.record_task_outcome(job_id="j", task_name="cron", outcome="passed")

    first = await wm.consolidate_narrative()
    assert first == "narrative v1"

    second = await wm.consolidate_narrative()
    assert second == ""
    assert calls[0] == 1
    assert wm.narrative.latest() == "narrative v1"

    # New workspace activity unblocks the guard.
    wm.record_command_outcome(command="pytest -q", exit_code=0)
    third = await wm.consolidate_narrative()
    assert third == "narrative v2"


def test_note_records_freeform_event(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    ev = wm.note("user reorganised src/ directory")
    assert ev is not None
    notes = wm.activity.recent(limit=5, kind="note")
    assert notes and "reorganised" in notes[0].summary


def test_has_seen_url_after_artifact(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    wm.record_research_artifact(url="https://example.org/paper", source="deep_research")
    assert wm.has_seen_url("https://example.org/paper") is True
    assert wm.has_seen_url("https://example.org/other") is False
