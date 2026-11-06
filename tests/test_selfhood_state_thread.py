"""Unit tests for the self-state thread (proposal item #3).

§4TB (2026-11-05): the open questions and unfinished threads this file also
held went with the `self_state` tool, their only writer. The tests of their
writers, dedup and caps went with them; what the thread still holds — the
mood and the last-active stamp — keeps every test it had (the ones that used
a question as their vehicle now use the mood).

§4TC (2026-11-06): the mood is a pair of numbers read as a label and an
intensity, not a free-text label — the same tests, driven through the new
writer (`set_owner_mood`: the owner's word puts the mood on a label). The
mood's own behaviour is pinned in tests/test_4tc_mood_engine.py."""

import json
from pathlib import Path

from ghost_agent.selfhood.mood import ANCHOR_DIRECTION
from ghost_agent.selfhood.state import SelfStateThread


def _set(s, label, intensity=2, note=""):
    return s.set_owner_mood(label, ANCHOR_DIRECTION[label], intensity, note=note)


def test_state_mood_setter(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    assert s.mood().neutral and s.state.mood is None
    before, after = _set(s, "curious", note="Elena just asked why Bruno tilts his head")
    assert before.neutral
    assert (after.label, after.intensity) == ("curious", 2)
    assert s.mood().label == "curious"

    # Reload
    s2 = SelfStateThread(tmp_path)
    assert (s2.mood().label, s2.mood().intensity) == ("curious", 2)


def test_state_format_as_prefix_empty(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    assert s.format_as_prefix() == ""


def test_state_format_as_prefix_includes_components(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    _set(s, "calm", 1, note="just consolidated")
    s.set_manner("formal", 3)
    s.touch_session()
    text = s.format_as_prefix()
    assert "My mood: calm, 1 of 3." in text and "My manner (set by the owner): formal, 3 of 3." in text
    assert "I was last active" in text
    assert "just consolidated" not in text          # §4TC: no cause, no note, no time in this line


def test_state_format_as_prefix_has_no_mood_line_for_a_neutral_mood(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    s.touch_session()
    text = s.format_as_prefix()
    assert text.startswith("I was last active") and "mood" not in text.lower()


def test_state_format_as_prefix_truncates(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    _set(s, "irritable")
    s.set_manner("sarcastic", 2)
    s.touch_session()
    text = s.format_as_prefix(max_chars=40)
    assert len(text) <= 40
    assert text.endswith("…")


def test_state_a_reset_with_nothing_on_file_writes_nothing(tmp_path: Path):
    """(Was: an empty label is a no-op.) Nothing to take off → no file."""
    s = SelfStateThread(tmp_path)
    before, after = s.reset_mood()
    assert before.neutral and after.neutral
    before, after = s.set_manner("", 0)
    assert not after.manner
    assert s.state.mood is None
    assert not (tmp_path / "state.json").exists()
    assert not s.mood_history_path.exists()


def test_state_corrupt_json_starts_empty(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text("this is not json", encoding="utf-8")
    s = SelfStateThread(tmp_path)
    assert s.mood().neutral and s.state.mood is None
    assert s.state.last_session_at == ""


def test_state_disabled_no_flush(tmp_path: Path):
    s = SelfStateThread(tmp_path, enabled=False)
    _set(s, "down")
    # In-memory mutation succeeded
    assert s.mood().label == "down"
    # But nothing on disk
    assert not (tmp_path / "state.json").exists()
    assert not s.mood_history_path.exists()


def test_state_touch_session_updates_timestamp(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    assert s.state.last_session_at == ""
    s.touch_session()
    assert s.state.last_session_at != ""

    s2 = SelfStateThread(tmp_path)
    assert s2.state.last_session_at == s.state.last_session_at


def test_state_touch_session_keeps_the_mood(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    _set(s, "tired", 3)
    s.touch_session()
    assert (SelfStateThread(tmp_path).mood().label, SelfStateThread(tmp_path).mood().intensity) == ("tired", 3)


def test_state_atomic_write_no_partial_file(tmp_path: Path):
    """The flush must use rename-from-tmp so a crash mid-write can't
    leave a half-state on disk."""
    s = SelfStateThread(tmp_path)
    _set(s, "content")
    # The tmp file shouldn't exist after a successful flush
    assert not (tmp_path / "state.json.tmp").exists()
    # The real file must be valid JSON
    data = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert data["mood"]["label"] == "content"
    assert (data["mood"]["p"], data["mood"]["e"], data["mood"]["intensity"]) == (0.6, 0.0, 2)
