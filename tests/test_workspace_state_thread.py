"""Tests for WorkspaceStateThread — file watcher, URL dedup, and the
state-thread persistence path."""

import json
from pathlib import Path

from ghost_agent.workspace.state import (
    MAX_SEEN_URLS,
    WorkspaceStateThread,
    _normalise_url,
)


def test_normalise_url_strips_fragment_and_trailing_slash():
    assert _normalise_url("https://EX.org/a/#section") == "https://ex.org/a"
    assert _normalise_url("HTTPS://example.com/") == "https://example.com"
    assert _normalise_url("") == ""


def test_track_and_untrack_file(tmp_path: Path):
    st = WorkspaceStateThread(tmp_path)
    p = str(tmp_path / "hello.txt")
    (tmp_path / "hello.txt").write_text("hi")
    tf = st.track_file(p, label="greeting")
    assert tf is not None and tf.label == "greeting"
    assert len(st.tracked_files()) == 1

    # Tracking the same path again is a no-op.
    again = st.track_file(p, label="other")
    assert again.label == "greeting"  # initial label sticks when present
    assert len(st.tracked_files()) == 1

    assert st.untrack_file(p) is True
    assert st.tracked_files() == []
    assert st.untrack_file(p) is False


def test_track_file_empty_path_returns_none(tmp_path: Path):
    st = WorkspaceStateThread(tmp_path)
    assert st.track_file("") is None
    assert st.track_file("   ") is None


def test_scan_tracked_detects_new_modification_and_deletion(tmp_path: Path):
    target = tmp_path / "watch.txt"
    target.write_text("first")
    st = WorkspaceStateThread(tmp_path)
    st.track_file(str(target))

    # First scan is "newly tracked".
    changes = st.scan_tracked()
    assert len(changes) == 1
    assert changes[0]["change"] == "newly tracked"

    # No change on second scan.
    assert st.scan_tracked() == []

    # Modify and re-scan.
    target.write_text("second is longer")
    changes = st.scan_tracked()
    assert len(changes) == 1
    assert "modified" in changes[0]["change"]

    # Delete and re-scan.
    target.unlink()
    changes = st.scan_tracked()
    assert len(changes) == 1
    assert changes[0]["change"] == "deleted"


def test_seen_url_dedup_idempotent(tmp_path: Path):
    st = WorkspaceStateThread(tmp_path)
    added = st.mark_url_seen("https://example.org/a")
    assert added is True
    again = st.mark_url_seen("https://EXAMPLE.org/a/")  # case + slash normalised
    assert again is False
    assert st.has_seen_url("https://example.org/a") is True


def test_seen_urls_cap_fifo(tmp_path: Path):
    st = WorkspaceStateThread(tmp_path)
    for i in range(MAX_SEEN_URLS + 5):
        st.mark_url_seen(f"https://example.org/{i}")
    assert len(st.state.seen_urls) == MAX_SEEN_URLS
    # Earliest URLs evicted; latest still present.
    assert "https://example.org/0" not in st.state.seen_urls
    assert f"https://example.org/{MAX_SEEN_URLS + 4}" in st.state.seen_urls


def test_corrupt_state_file_recovers_silently(tmp_path: Path):
    (tmp_path / "state.json").write_text("{ not json")
    st = WorkspaceStateThread(tmp_path)
    # Falls back to empty state — never raises.
    assert st.tracked_files() == []
    assert st.state.seen_urls == []


def test_state_persists_across_reconstruction(tmp_path: Path):
    st1 = WorkspaceStateThread(tmp_path)
    st1.track_file(str(tmp_path / "x.py"), label="experiment")
    st1.mark_url_seen("https://x.org")
    # Re-open — state should survive.
    st2 = WorkspaceStateThread(tmp_path)
    paths = {tf.path for tf in st2.tracked_files()}
    assert str(tmp_path / "x.py") in paths
    assert st2.has_seen_url("https://x.org")


def test_format_as_prefix_empty_when_no_state(tmp_path: Path):
    st = WorkspaceStateThread(tmp_path)
    assert st.format_as_prefix() == ""


def test_format_as_prefix_surfaces_session_and_tracked(tmp_path: Path):
    st = WorkspaceStateThread(tmp_path)
    st.track_file(str(tmp_path / "a.py"), label="main")
    st.touch_session()
    text = st.format_as_prefix()
    assert "a.py" in text
    assert "(main)" in text
    assert "last touched" in text.lower()
