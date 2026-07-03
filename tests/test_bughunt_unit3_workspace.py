"""Regression tests for bug-hunt unit 3 (workspace/) — see BUGHUNT.md.

Fixed bugs pinned here:
 1. reactions: BOM'd valid .py is not flagged as broken; invalid-byte source
    that the real interpreter rejects is not silently passed
 2. activity: log compacts past a byte cap; iter_events doesn't hold the lock
    across yields (append-during-iteration doesn't deadlock); recent() is O(limit)
 3. narrative: changelog scopes like the prefix path (keeps agnostic + derivable
    legacy events); "undated" bucket sorts last
 4. recognition: strip_workspace_prefix searches CLOSE after OPEN; cross-project
    narrative check is prefix-tolerant
 5. state: track_file expands ~ and absolutises; touch_session preserves the
    prior session timestamp; corrupt state.json is preserved not overwritten;
    from_dict skips a malformed tracked_files entry instead of wiping everything
 6. model: research dedup marks seen only after a successful append; nav counter
    is recency-evicting; TaskOutcome fired_at/finished_at are correct
"""

import json
import os
import threading
import time
from pathlib import Path

import pytest

from ghost_agent.workspace.reactions import check_changed_python_files
from ghost_agent.workspace.activity import WorkspaceActivity, _COMPACT_MAX_BYTES
from ghost_agent.workspace.narrative import render_changelog
from ghost_agent.workspace.recognition import (
    strip_workspace_prefix,
    _narrative_is_cross_project,
    WORKSPACE_PREFIX_OPEN,
    WORKSPACE_PREFIX_CLOSE,
)
from ghost_agent.workspace.state import WorkspaceStateThread, _normalise_path
from ghost_agent.workspace.schema import WorkspaceEvent, WorkspaceState
from ghost_agent.workspace.model import WorkspaceModel


# ──────────────────────────────────────────────────────────────────────
# 1. reactions encoding
# ──────────────────────────────────────────────────────────────────────

class TestReactionsEncoding:
    def test_bom_valid_file_not_flagged(self, tmp_path):
        f = tmp_path / "bom.py"
        f.write_bytes(b"\xef\xbb\xbfx = 1\nprint(x)\n")  # UTF-8 BOM + valid code
        # Pre-fix: read_text kept U+FEFF and ast.parse raised → false warning.
        assert check_changed_python_files([{"path": str(f), "change": "modified"}]) == []

    def test_invalid_byte_broken_file_is_flagged(self, tmp_path):
        f = tmp_path / "bad.py"
        # latin-1 é (0xe9) + a genuine syntax error the interpreter rejects.
        f.write_bytes(b"x = 'caf\xe9'\ny = (\n")
        # Pre-fix: errors="ignore" dropped the byte AND could mask the error.
        assert check_changed_python_files([{"path": str(f), "change": "modified"}])

    def test_clean_ascii_file_not_flagged(self, tmp_path):
        f = tmp_path / "ok.py"
        f.write_text("def f():\n    return 42\n", encoding="utf-8")
        assert check_changed_python_files([{"path": str(f), "change": "modified"}]) == []


# ──────────────────────────────────────────────────────────────────────
# 2. activity log
# ──────────────────────────────────────────────────────────────────────

class TestActivityLog:
    def test_log_compacts_past_cap(self, tmp_path, monkeypatch):
        import ghost_agent.workspace.activity as act_mod
        monkeypatch.setattr(act_mod, "_COMPACT_MAX_BYTES", 4096)
        monkeypatch.setattr(act_mod, "_COMPACT_KEEP_LINES", 20)
        act = WorkspaceActivity(tmp_path)
        for i in range(400):
            act.append(WorkspaceEvent(kind="note", summary=f"event number {i} " + "x" * 40))
        # Pre-fix: file grew without bound. Now it's compacted to the tail.
        assert act.path.stat().st_size <= 4096 * 2
        # The newest events survive compaction; the oldest are dropped.
        recent = act.recent(limit=1)
        assert "event number 399 " in recent[0].summary
        assert not any("event number 0 " in e.summary for e in act.recent(limit=100))

    def test_append_during_iteration_does_not_deadlock(self, tmp_path):
        act = WorkspaceActivity(tmp_path)
        for i in range(5):
            act.append(WorkspaceEvent(kind="note", summary=f"seed {i}"))
        # Pre-fix: iter_events held a non-reentrant lock across yield, so an
        # append from the same thread mid-iteration deadlocked.
        seen = 0
        for _ in act.iter_events():
            seen += 1
            if seen == 1:
                act.append(WorkspaceEvent(kind="note", summary="mid-iteration"))
        assert seen == 5  # snapshot taken before the append

    def test_recent_returns_tail(self, tmp_path):
        act = WorkspaceActivity(tmp_path)
        for i in range(50):
            act.append(WorkspaceEvent(kind="note", summary=f"e{i}"))
        recent = act.recent(limit=3)
        assert [e.summary for e in recent] == ["e47", "e48", "e49"]


# ──────────────────────────────────────────────────────────────────────
# 3. narrative changelog
# ──────────────────────────────────────────────────────────────────────

class TestChangelog:
    def test_scoping_keeps_agnostic_and_derivable_legacy(self):
        active = "991e52bce912"
        events = [
            WorkspaceEvent(kind="note", summary="stamped", project_id=active,
                           timestamp="2026-07-03T10:00:00Z"),
            WorkspaceEvent(kind="note", summary="agnostic note", project_id="",
                           timestamp="2026-07-03T09:00:00Z"),
            WorkspaceEvent(kind="file_changed", summary="legacy edit", project_id="",
                           payload={"path": f"projects/{active}/app.py"},
                           timestamp="2026-07-03T08:00:00Z"),
            WorkspaceEvent(kind="note", summary="OTHER project", project_id="deadbeef99",
                           timestamp="2026-07-03T07:00:00Z"),
        ]
        out = render_changelog(events, active_project_id=active)
        assert "stamped" in out
        assert "agnostic note" in out          # pre-fix: dropped
        assert "legacy edit" in out            # pre-fix: dropped (derivable)
        assert "OTHER project" not in out       # different project stays out

    def test_undated_bucket_sorts_last(self):
        events = [
            WorkspaceEvent(kind="note", summary="today", timestamp="2026-07-03T10:00:00Z"),
            WorkspaceEvent(kind="note", summary="no timestamp", timestamp=""),
        ]
        out = render_changelog(events)
        # Pre-fix: 'undated' ('u' > '2') floated above every real date.
        assert out.index("2026-07-03") < out.index("undated")


# ──────────────────────────────────────────────────────────────────────
# 4. recognition
# ──────────────────────────────────────────────────────────────────────

class TestRecognition:
    def test_strip_searches_close_after_open(self):
        text = (f"{WORKSPACE_PREFIX_CLOSE} stray earlier marker\n"
                f"BEFORE {WORKSPACE_PREFIX_OPEN}\nblock body\n{WORKSPACE_PREFIX_CLOSE}\nAFTER")
        out = strip_workspace_prefix(text)
        # Pre-fix: find(CLOSE) from 0 hit the stray marker → end < start →
        # the block was duplicated instead of removed.
        assert "block body" not in out
        assert "BEFORE" in out and "AFTER" in out

    def test_strip_noop_when_absent(self):
        assert strip_workspace_prefix("no markers here") == "no markers here"

    @pytest.mark.parametrize("path,expected_cross", [
        ("projects/991e52bce912/x.py", False),   # exact
        ("projects/991e52bce9/x.py", False),     # truncated (real prefix)
        ("projects/991e52bce912ab/x.py", False), # extended
        ("projects/deadbeef1234/x.py", True),    # genuinely different
    ])
    def test_cross_project_is_prefix_tolerant(self, path, expected_cross):
        assert _narrative_is_cross_project(f"see {path}", "991e52bce912") is expected_cross


# ──────────────────────────────────────────────────────────────────────
# 5. state
# ──────────────────────────────────────────────────────────────────────

class TestState:
    def test_track_file_expands_and_absolutises(self, tmp_path, monkeypatch):
        st = WorkspaceStateThread(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        tf = st.track_file("~/notes.md")
        # Pre-fix: "~/notes.md" stored verbatim → os.stat never finds it.
        assert tf.path == str(tmp_path / "notes.md")
        assert os.path.isabs(tf.path)

    def test_untrack_matches_after_normalisation(self, tmp_path, monkeypatch):
        st = WorkspaceStateThread(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        st.track_file("~/notes.md")
        assert st.untrack_file("~/notes.md") is True

    def test_relative_path_resolved_against_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = WorkspaceStateThread(tmp_path)
        tf = st.track_file("src/app.py")
        assert tf.path == str(tmp_path / "src" / "app.py")

    def test_touch_session_preserves_prior(self, tmp_path):
        st = WorkspaceStateThread(tmp_path)
        st._state.last_session_at = "2026-07-01T00:00:00Z"  # a prior session
        st.touch_session()
        # last_session_at is now the current boot; prior is preserved.
        assert st._state.prior_session_at == "2026-07-01T00:00:00Z"
        assert st._state.last_session_at != "2026-07-01T00:00:00Z"
        # The prefix surfaces the PRIOR session, not the boot time.
        assert "2026-07-01T00:00:00Z" in st.format_as_prefix()

    def test_corrupt_state_preserved_not_overwritten(self, tmp_path):
        p = tmp_path / "state.json"
        p.write_text("{ this is not valid json", encoding="utf-8")
        st = WorkspaceStateThread(tmp_path)  # _read_or_empty runs
        # Pre-fix: corrupt file silently replaced by empty state on next flush.
        st.track_file(str(tmp_path / "a.py"))  # triggers a flush
        sidecars = list(tmp_path.glob("state.json.corrupt-*"))
        assert sidecars, "corrupt state file was not preserved as a sidecar"

    def test_from_dict_skips_bad_tracked_entry(self):
        data = {
            "tracked_files": ["oops-not-a-dict", {"path": "/real/file.py"}],
            "seen_urls": ["https://a", "https://b"],
            "last_session_at": "2026-07-01T00:00:00Z",
        }
        # Pre-fix: the bad entry threw and the WHOLE state was discarded.
        st = WorkspaceState.from_dict(data)
        assert [tf.path for tf in st.tracked_files] == ["/real/file.py"]
        assert st.seen_urls == ["https://a", "https://b"]
        assert st.last_session_at == "2026-07-01T00:00:00Z"


# ──────────────────────────────────────────────────────────────────────
# 6. model
# ──────────────────────────────────────────────────────────────────────

class TestModel:
    def test_research_dedup_marks_seen_only_after_append(self, tmp_path, monkeypatch):
        wm = WorkspaceModel(tmp_path)
        # Force the activity append to fail once.
        monkeypatch.setattr(wm.activity, "append", lambda ev: None)
        assert wm.record_research_artifact(url="https://example.com/x") is None
        # Pre-fix: the URL was marked seen before the append, so it was
        # permanently deduped despite never being recorded.
        assert wm.has_seen_url("https://example.com/x") is False

        # Restore a working append: the same URL must now record.
        real = WorkspaceModel(tmp_path)
        art = real.record_research_artifact(url="https://example.com/x")
        assert art is not None
        assert real.record_research_artifact(url="https://example.com/x") is None  # now deduped

    def test_nav_counter_stays_warm_when_revisited_among_cold_traffic(self, tmp_path):
        # A URL that keeps being revisited must not have its counter
        # evicted just because lots of OTHER urls were seen in between.
        # Pre-fix (insertion-order eviction) dropped the hot URL's counter
        # even though it was being actively revisited; the recency-aware
        # eviction keeps it warm so the threshold nudge still fires.
        wm = WorkspaceModel(tmp_path)
        hot = "https://hot.example/page"
        assert wm.record_navigation(hot, threshold=3) is None      # visit 1
        for i in range(300):
            wm.record_navigation(f"https://cold.example/a/{i}", threshold=3)
        assert wm.record_navigation(hot, threshold=3) is None      # visit 2 (kept warm)
        for i in range(300):
            wm.record_navigation(f"https://cold.example/b/{i}", threshold=3)
        suggestion = wm.record_navigation(hot, threshold=3)        # visit 3
        assert suggestion is not None
        assert "3 times" in suggestion

    def test_task_outcome_timestamps(self, tmp_path):
        wm = WorkspaceModel(tmp_path)
        t = wm.record_task_outcome(job_id="j1", task_name="nightly",
                                   outcome="passed", duration_seconds=120.0)
        assert t is not None
        # finished_at is set; fired_at is back-dated by the duration.
        assert t.finished_at != ""
        assert t.fired_at != ""
        assert t.fired_at < t.finished_at
