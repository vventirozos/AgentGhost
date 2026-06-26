"""Tests for the workspace changelog (2B) and browser-nav repetition (2C)."""

from ghost_agent.workspace.model import WorkspaceModel
from ghost_agent.workspace.narrative import render_changelog
from ghost_agent.workspace.schema import WorkspaceEvent


# ── 2B: changelog renderer ─────────────────────────────────────────────

def _ev(kind, summary, ts, project_id=""):
    return WorkspaceEvent(kind=kind, summary=summary, timestamp=ts, project_id=project_id)


def test_render_changelog_groups_by_day_descending():
    events = [
        _ev("file_changed", "a.py: modified", "2026-06-24T09:00:00+00:00"),
        _ev("command", "ran pytest exit=0", "2026-06-25T10:00:00+00:00"),
        _ev("file_changed", "b.py: appeared", "2026-06-25T11:00:00+00:00"),
    ]
    out = render_changelog(events)
    assert "# Workspace changelog" in out
    # Most recent day first.
    assert out.index("## 2026-06-25") < out.index("## 2026-06-24")
    assert "[command] ran pytest exit=0" in out
    assert "[file_changed] b.py: appeared" in out


def test_render_changelog_scopes_to_project():
    events = [
        _ev("file_changed", "mine.py: modified", "2026-06-25T09:00:00+00:00", project_id="p1"),
        _ev("file_changed", "theirs.py: modified", "2026-06-25T09:00:00+00:00", project_id="p2"),
    ]
    out = render_changelog(events, active_project_id="p1")
    assert "mine.py" in out
    assert "theirs.py" not in out


def test_render_changelog_empty():
    assert render_changelog([]) == ""


def test_write_changelog_persists_file(tmp_path):
    wm = WorkspaceModel(tmp_path, enabled=True)
    wm.current_project_id = "proj1"
    wm.record_command_outcome(command="pytest", exit_code=0)
    path = wm.write_changelog()
    assert path is not None
    assert path.name == "CHANGELOG.proj1.md"
    assert "pytest" in path.read_text()


def test_write_changelog_noop_when_no_activity(tmp_path):
    wm = WorkspaceModel(tmp_path, enabled=True)
    wm.current_project_id = "empty"
    assert wm.write_changelog() is None


# ── 2C: navigation repetition ──────────────────────────────────────────

def test_record_navigation_suggests_at_threshold(tmp_path):
    wm = WorkspaceModel(tmp_path, enabled=True)
    url = "https://example.com/status"
    assert wm.record_navigation(url) is None   # 1st
    assert wm.record_navigation(url) is None   # 2nd
    suggestion = wm.record_navigation(url)      # 3rd
    assert suggestion is not None
    assert "cache" in suggestion.lower()
    assert url in suggestion


def test_record_navigation_fires_once(tmp_path):
    wm = WorkspaceModel(tmp_path, enabled=True)
    url = "https://example.com/x"
    for _ in range(2):
        wm.record_navigation(url)
    assert wm.record_navigation(url) is not None   # 3rd → fires
    assert wm.record_navigation(url) is None        # 4th → silent


def test_record_navigation_per_url(tmp_path):
    wm = WorkspaceModel(tmp_path, enabled=True)
    wm.record_navigation("https://a.com")
    wm.record_navigation("https://a.com")
    # A different URL is independent — no suggestion yet.
    assert wm.record_navigation("https://b.com") is None


def test_record_navigation_writes_note_event(tmp_path):
    wm = WorkspaceModel(tmp_path, enabled=True)
    url = "https://example.com/loop"
    for _ in range(3):
        wm.record_navigation(url)
    notes = [e for e in wm.activity.recent(limit=20) if e.kind == "note"]
    assert any(url in (n.summary or "") for n in notes)


def test_record_navigation_disabled_is_noop(tmp_path):
    wm = WorkspaceModel(tmp_path, enabled=False)
    for _ in range(5):
        assert wm.record_navigation("https://x.com") is None
