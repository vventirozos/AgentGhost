"""Unit tests for the self-state thread (proposal item #3)."""

import json
from pathlib import Path

import pytest

from ghost_agent.selfhood.state import MAX_OPEN_QUESTIONS, MAX_UNFINISHED, SelfStateThread


def test_state_persists_open_question(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    q = s.note_open_question("Is consciousness substrate-independent?")
    assert q is not None
    assert q.id

    # Reload from disk
    s2 = SelfStateThread(tmp_path)
    qs = s2.open_questions()
    assert len(qs) == 1
    assert qs[0].text == "Is consciousness substrate-independent?"


def test_state_dedupes_identical_open_questions(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    s.note_open_question("Same question?")
    s.note_open_question("Same question?")
    s.note_open_question("Same question?")
    assert len(s.open_questions()) == 1


def test_state_mark_question_resolved(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    q = s.note_open_question("Will I survive?")
    assert s.mark_question_resolved(q.id) is True
    assert s.open_questions() == []  # filtered out (resolved_at populated)

    # Bogus id is False
    assert s.mark_question_resolved("nope") is False


def test_state_unfinished_threads(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    t = s.add_unfinished("Refactor the parser to handle streaming")
    assert t is not None
    assert len(s.unfinished_threads()) == 1

    assert s.close_unfinished(t.id) is True
    assert s.unfinished_threads() == []


def test_state_mood_setter(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    assert s.mood() is None
    m = s.set_mood("curious", "the user just asked about consciousness")
    assert m is not None
    assert m.label == "curious"
    assert s.mood().label == "curious"

    # Reload
    s2 = SelfStateThread(tmp_path)
    assert s2.mood().label == "curious"


def test_state_capped_open_questions(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    for i in range(MAX_OPEN_QUESTIONS + 5):
        s.note_open_question(f"Question number {i}?")
    qs = s.open_questions()
    assert len(qs) == MAX_OPEN_QUESTIONS
    # most-recent-wins: the LAST one must be present
    assert any(f"Question number {MAX_OPEN_QUESTIONS + 4}" in q.text for q in qs)
    # the first must have been dropped
    assert not any("Question number 0?" in q.text for q in qs)


def test_state_capped_unfinished(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    for i in range(MAX_UNFINISHED + 3):
        s.add_unfinished(f"thread-{i}")
    threads = s.unfinished_threads()
    assert len(threads) == MAX_UNFINISHED


def test_state_format_as_prefix_empty(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    assert s.format_as_prefix() == ""


def test_state_format_as_prefix_includes_components(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    s.note_open_question("Is recursion all there is?")
    s.add_unfinished("finish the consciousness essay")
    s.set_mood("reflective", "just consolidated")
    s.touch_session()
    text = s.format_as_prefix()
    assert "Is recursion all there is?" in text
    assert "consciousness essay" in text
    assert "reflective" in text
    assert "I was last active" in text


def test_state_format_as_prefix_truncates(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    for i in range(8):
        s.note_open_question("x" * 300 + f"-{i}")
    text = s.format_as_prefix(max_chars=200)
    assert len(text) <= 200
    assert text.endswith("…")


def test_state_empty_inputs_no_op(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    assert s.note_open_question("") is None
    assert s.note_open_question("   ") is None
    assert s.add_unfinished("") is None
    assert s.set_mood("") is None


def test_state_corrupt_json_starts_empty(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text("this is not json", encoding="utf-8")
    s = SelfStateThread(tmp_path)
    assert s.open_questions() == []
    assert s.unfinished_threads() == []
    assert s.mood() is None


def test_state_disabled_no_flush(tmp_path: Path):
    s = SelfStateThread(tmp_path, enabled=False)
    s.note_open_question("disabled")
    # In-memory mutation succeeded
    assert len(s.open_questions()) == 1
    # But nothing on disk
    assert not (tmp_path / "state.json").exists()


def test_state_touch_session_updates_timestamp(tmp_path: Path):
    s = SelfStateThread(tmp_path)
    assert s.state.last_session_at == ""
    s.touch_session()
    assert s.state.last_session_at != ""

    s2 = SelfStateThread(tmp_path)
    assert s2.state.last_session_at == s.state.last_session_at


def test_state_atomic_write_no_partial_file(tmp_path: Path):
    """The flush must use rename-from-tmp so a crash mid-write can't
    leave a half-state on disk."""
    s = SelfStateThread(tmp_path)
    s.note_open_question("first")
    # The tmp file shouldn't exist after a successful flush
    assert not (tmp_path / "state.json.tmp").exists()
    # The real file must be valid JSON
    data = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert isinstance(data["open_questions"], list)
