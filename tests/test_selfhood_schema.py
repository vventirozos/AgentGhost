"""Unit tests for selfhood schema dataclasses."""

import json

from ghost_agent.selfhood.schema import (
    SCHEMA_VERSION,
    Experience,
    Mood,
    SelfState,
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
        mood=Mood(p=0.6, e=0.45, updated_at="2026-05-11T12:00:00Z", label="cheerful", intensity=2,
                  cause="the owner gave a thumbs-up", manner="playful", manner_intensity=1),
        last_session_at="2026-05-11T12:00:00Z",
        last_owner_turn_at="2026-05-11T11:58:00Z",
    )
    d = s.to_dict()
    assert set(d) == {"schema_version", "mood", "last_session_at",
                      "last_owner_turn_at",                          # §4TB: no question / thread lists; §4TC: the owner stamp
                      "last_owner_job_at"}                           # §4UD (F9): when a job of the owner's last worked
    s2 = SelfState.from_dict(d)
    assert s2.mood is not None
    assert (s2.mood.label, s2.mood.intensity, s2.mood.p, s2.mood.e) == ("cheerful", 2, 0.6, 0.45)
    assert (s2.mood.manner, s2.mood.manner_intensity, s2.mood.cause) == ("playful", 1, "the owner gave a thumbs-up")
    assert s2.last_session_at == "2026-05-11T12:00:00Z"
    assert s2.last_owner_turn_at == "2026-05-11T11:58:00Z"
    assert s2.schema_version == SCHEMA_VERSION


def test_self_state_handles_null_mood():
    s = SelfState()
    d = s.to_dict()
    assert d["mood"] is None
    s2 = SelfState.from_dict(d)
    assert s2.mood is None


def test_self_state_from_empty_dict():
    s = SelfState.from_dict({})
    assert s.mood is None
    assert s.last_session_at == ""
    assert s.schema_version == SCHEMA_VERSION
