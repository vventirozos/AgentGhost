"""Tests for skills_auto.verifier."""

import pytest

from ghost_agent.skills_auto.extractor import SkillCandidate
from ghost_agent.skills_auto.verifier import verify_candidate


def _c(confidence=0.5):
    return SkillCandidate(
        name="x", cluster="sql", tool_sequence=("a", "b"),
        support=5, exemplar_trajectory_id="t1",
        trigger_examples=[], confidence=confidence, signature_hash="h1",
    )


def test_pass_boosts_confidence_and_keeps():
    c = _c(confidence=0.5)
    result = verify_candidate(c, lambda _c: True, pass_boost=0.1)
    assert result.passed is True
    assert result.action == "keep"
    assert result.updated_confidence == pytest.approx(0.6)


def test_pass_clips_at_one():
    c = _c(confidence=0.95)
    result = verify_candidate(c, lambda _c: True, pass_boost=0.5)
    assert result.updated_confidence == 1.0


def test_fail_with_low_confidence_deprecates():
    c = _c(confidence=0.3)
    result = verify_candidate(
        c, lambda _c: False,
        fail_penalty=0.1, deprecate_below_confidence=0.25,
    )
    assert result.passed is False
    assert result.action == "deprecate"
    assert result.updated_confidence == pytest.approx(0.2)


def test_fail_but_still_above_threshold_retain_monitor():
    c = _c(confidence=0.9)
    result = verify_candidate(
        c, lambda _c: False,
        fail_penalty=0.1, deprecate_below_confidence=0.25,
    )
    assert result.passed is False
    assert result.action == "retain_monitor"
    assert result.updated_confidence == pytest.approx(0.8)


def test_fail_clips_at_zero():
    c = _c(confidence=0.05)
    result = verify_candidate(c, lambda _c: False, fail_penalty=0.5)
    assert result.updated_confidence == 0.0


def test_exception_in_verify_fn_is_treated_as_failure():
    def boom(_c):
        raise RuntimeError("tool missing")
    c = _c(confidence=0.8)
    result = verify_candidate(c, boom, fail_penalty=0.2)
    assert result.passed is False
    assert "RuntimeError" in result.reason
    assert "tool missing" in result.reason


def test_result_carries_candidate_name():
    c = _c(confidence=0.5)
    c.name = "my.skill.name"
    result = verify_candidate(c, lambda _c: True)
    assert result.candidate_name == "my.skill.name"
