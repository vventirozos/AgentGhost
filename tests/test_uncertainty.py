"""Tests for the Metacognitive Monitoring (UncertaintyTracker) module."""

import pytest

from ghost_agent.core.uncertainty import (
    UncertaintyTracker, Unknown, Assumption,
)


@pytest.fixture
def tracker():
    return UncertaintyTracker()


class TestUnknown:
    def test_to_dict(self):
        u = Unknown(what="test", impact=3, resolution="ask user")
        d = u.to_dict()
        assert d["what"] == "test"
        assert d["impact"] == 3
        assert d["resolved"] is False


class TestAssumption:
    def test_to_dict(self):
        a = Assumption(claim="X is true", confidence=0.7, basis="common sense")
        d = a.to_dict()
        assert d["claim"] == "X is true"
        assert d["confidence"] == 0.7
        assert d["verified"] is False


class TestUncertaintyTracker:
    def test_flag_unknown(self, tracker):
        u = tracker.flag_unknown("What database version?", impact=4)
        assert u.what == "What database version?"
        assert u.impact == 4
        assert len(tracker.unknowns) == 1

    def test_flag_unknown_clamps_impact(self, tracker):
        u = tracker.flag_unknown("test", impact=10)
        assert u.impact == 5
        u2 = tracker.flag_unknown("test2", impact=-1)
        assert u2.impact == 1

    def test_resolve_unknown(self, tracker):
        tracker.flag_unknown("What version?", impact=3)
        assert tracker.resolve_unknown(0, "PostgreSQL 15")
        assert tracker.unknowns[0].resolved is True
        assert tracker.unknowns[0].resolved_value == "PostgreSQL 15"

    def test_resolve_unknown_invalid_index(self, tracker):
        assert tracker.resolve_unknown(99, "value") is False

    def test_flag_assumption(self, tracker):
        a = tracker.flag_assumption("User wants Python", confidence=0.8, basis="file extension")
        assert a.claim == "User wants Python"
        assert a.confidence == 0.8

    def test_flag_assumption_clamps_confidence(self, tracker):
        a = tracker.flag_assumption("test", confidence=1.5)
        assert a.confidence == 1.0
        a2 = tracker.flag_assumption("test2", confidence=-0.5)
        assert a2.confidence == 0.0

    def test_verify_assumption(self, tracker):
        tracker.flag_assumption("X is true", confidence=0.5)
        assert tracker.verify_assumption(0, True)
        assert tracker.assumptions[0].verified is True
        assert tracker.assumptions[0].was_correct is True

    def test_get_critical_unknowns(self, tracker):
        tracker.flag_unknown("minor thing", impact=2)
        tracker.flag_unknown("critical thing", impact=5)
        tracker.flag_unknown("resolved critical", impact=4)
        tracker.resolve_unknown(2, "fixed")

        critical = tracker.get_critical_unknowns(min_impact=4)
        assert len(critical) == 1
        assert critical[0].what == "critical thing"

    def test_get_unverified_assumptions(self, tracker):
        tracker.flag_assumption("sure thing", confidence=0.9)
        tracker.flag_assumption("unsure thing", confidence=0.3)
        tracker.flag_assumption("verified thing", confidence=0.2)
        tracker.verify_assumption(2, True)

        unverified = tracker.get_unverified_assumptions(max_confidence=0.5)
        assert len(unverified) == 1
        assert unverified[0].claim == "unsure thing"

    def test_should_ask_user_returns_question(self, tracker):
        tracker.flag_unknown("What is the expected output format?", impact=5, resolution="ask user")
        question = tracker.should_ask_user()
        assert question is not None
        assert "expected output format" in question

    def test_should_ask_user_returns_none_when_no_critical(self, tracker):
        tracker.flag_unknown("minor thing", impact=2, resolution="ask user")
        assert tracker.should_ask_user() is None

    def test_should_ask_user_ignores_non_user_resolutions(self, tracker):
        tracker.flag_unknown("need to search", impact=5, resolution="search web")
        assert tracker.should_ask_user() is None

    def test_risk_summary_empty(self, tracker):
        assert tracker.get_risk_summary() == ""

    def test_risk_summary_with_unknowns_and_assumptions(self, tracker):
        tracker.flag_unknown("unclear requirement", impact=3)
        tracker.flag_assumption("data is CSV", confidence=0.4)
        summary = tracker.get_risk_summary()
        assert "unclear requirement" in summary
        assert "data is CSV" in summary

    def test_reset(self, tracker):
        tracker.flag_unknown("test", impact=3)
        tracker.flag_assumption("test", confidence=0.5)
        tracker.reset()
        assert len(tracker.unknowns) == 0
        assert len(tracker.assumptions) == 0

    def test_to_dict(self, tracker):
        tracker.flag_unknown("test", impact=3)
        tracker.flag_assumption("claim", confidence=0.5)
        d = tracker.to_dict()
        assert len(d["unknowns"]) == 1
        assert len(d["assumptions"]) == 1

    def test_get_context_for_prompt(self, tracker):
        tracker.flag_unknown("unclear thing", impact=4)
        tracker.flag_assumption("risky assumption", confidence=0.3)
        ctx = tracker.get_context_for_prompt()
        assert "UNCERTAINTIES" in ctx
        assert "unclear thing" in ctx
        assert "risky assumption" in ctx

    def test_get_context_empty_when_no_concerns(self, tracker):
        assert tracker.get_context_for_prompt() == ""
