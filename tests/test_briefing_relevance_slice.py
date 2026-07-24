"""Query-conditioned briefing slice (2026-07-24, Phase 3 of project
accessibility).

Fixed top-N briefing sections don't scale: on a large project the model
needs the CORNER of the map matching THIS request. `build_project_briefing`
now accepts `request_text` and injects a RELEVANT TO THIS REQUEST section —
deterministic keyword overlap against the file manifest and a deeper
work_log window, zero LLM cost.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.core.prompts import build_project_briefing


@pytest.fixture
def store(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return ProjectStore(tmp_path / "memory", sandbox_root=sandbox)


@pytest.fixture
def pid(store):
    p = store.create_project("Journal", kind="CODING", goal="training journal")
    store.describe_file(p, "calendar.html", "calendar view of training sessions")
    store.describe_file(p, "weight-tracker.html", "weight logging with canvas chart")
    store.describe_file(p, "server.js", "Node service on :8100", role="entrypoint")
    return p


def test_matching_request_surfaces_relevant_files_first(store, pid):
    text = build_project_briefing(store, pid,
                                  request_text="the calendar does not refresh")
    assert "RELEVANT TO THIS REQUEST" in text
    sec = text.split("RELEVANT TO THIS REQUEST", 1)[1]
    assert "calendar.html" in sec.split("DELIVERABLES")[0]
    # Non-matching files are not in the slice.
    assert "weight-tracker.html" not in sec.split("DELIVERABLES")[0]


def test_no_request_text_no_slice(store, pid):
    text = build_project_briefing(store, pid)
    assert "RELEVANT TO THIS REQUEST" not in text


def test_generic_request_produces_no_false_slice(store, pid):
    """Stopworded phrasing ('resume X and give me a status update') must not
    match everything — that is the exact live request that motivated the
    stopword list."""
    text = build_project_briefing(
        store, pid, request_text="resume the project and give me a status update")
    if "RELEVANT TO THIS REQUEST" in text:  # nothing should overlap
        sec = text.split("RELEVANT TO THIS REQUEST", 1)[1].split("DELIVERABLES")[0]
        assert "calendar.html" not in sec and "server.js" not in sec


def test_slice_includes_last_history_line(store, pid):
    store.add_work_log(pid, request="fix calendar rendering",
                       files=["calendar.html"], outcome="verifier:failed",
                       note="renderCalendar never re-reads DataStore")
    text = build_project_briefing(store, pid,
                                  request_text="calendar still broken")
    sec = text.split("RELEVANT TO THIS REQUEST", 1)[1].split("DELIVERABLES")[0]
    assert "verifier:failed" in sec
    assert "renderCalendar" in sec


def test_deep_journal_matches_surface_beyond_recent_window(store, pid):
    # 6 filler logs push the matching row out of the newest-5 RECENT window.
    store.add_work_log(pid, request="tune chart colors for weight page",
                       files=["weight-tracker.html"], outcome="completed",
                       note="canvas gradient added to weight chart")
    for i in range(6):
        store.add_work_log(pid, request=f"misc step {i}", files=[],
                           outcome="completed", note="routine")
    text = build_project_briefing(
        store, pid, request_text="the weight chart gradient looks wrong")
    sec = text.split("RELEVANT TO THIS REQUEST", 1)[1].split("DELIVERABLES")[0]
    assert "(journal)" in sec and "gradient" in sec
