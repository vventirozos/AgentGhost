"""Unit tests for selfhood schema dataclasses."""

import json

from ghost_agent.selfhood.schema import (
    SCHEMA_VERSION,
    Experience,
    Mood,
    OpenQuestion,
    SelfState,
    UnfinishedThread,
)


def test_experience_default_subject_is_self():
    """The continuity tag (proposal item #2) lives on every experience.
    Retrieval layers key off subject=='self' to treat records as 'mine'."""
    e = Experience(summary="I did the thing.")
    assert e.subject == "self"


def test_experience_roundtrip_via_jsonl():
    e = Experience(
        trajectory_id="abc123",
        summary="I worked on parsing logs.",
        tools_used=["execute", "file_system"],
        outcome="passed",
        user_first_words="parse my logs",
    )
    line = e.to_jsonl()
    d = json.loads(line)
    e2 = Experience.from_dict(d)
    assert e2.trajectory_id == "abc123"
    assert e2.summary == "I worked on parsing logs."
    assert e2.tools_used == ["execute", "file_system"]
    assert e2.outcome == "passed"
    assert e2.subject == "self"


def test_experience_from_dict_ignores_unknown_fields():
    e = Experience.from_dict({
        "summary": "hi",
        "future_field_we_dont_know_about": 42,
    })
    assert e.summary == "hi"


def test_self_state_roundtrip():
    s = SelfState(
        open_questions=[OpenQuestion(text="What is consciousness?")],
        unfinished_threads=[UnfinishedThread(descriptor="rewrite the parser")],
        mood=Mood(label="curious", evidence="just saw a new paper"),
        last_session_at="2026-05-11T12:00:00Z",
    )
    d = s.to_dict()
    s2 = SelfState.from_dict(d)
    assert s2.open_questions[0].text == "What is consciousness?"
    assert s2.unfinished_threads[0].descriptor == "rewrite the parser"
    assert s2.mood is not None
    assert s2.mood.label == "curious"
    assert s2.last_session_at == "2026-05-11T12:00:00Z"
    assert s2.schema_version == SCHEMA_VERSION


def test_self_state_handles_null_mood():
    s = SelfState()
    d = s.to_dict()
    assert d["mood"] is None
    s2 = SelfState.from_dict(d)
    assert s2.mood is None


def test_self_state_from_empty_dict():
    s = SelfState.from_dict({})
    assert s.open_questions == []
    assert s.unfinished_threads == []
    assert s.mood is None
    assert s.last_session_at == ""
    assert s.schema_version == SCHEMA_VERSION
