"""Lesson graduation could never fire — 2026-07-27 (later 3).

Measured against the live 50-lesson playbook: 3 lessons met the
``frequency >= 5`` reusability gate, 17 met the mechanizability gate, and
**zero met both** — so ``graduate_skills`` had no candidate to consider,
ever, and "0 graduated" gave no hint why.

Both gates were wrong in opposite directions:

* Reusability at ``frequency >= 5`` was effectively unreachable (38 of 50
  lessons sat at 1), and the three that reached it were behavioural
  guidance rather than procedures.
* Mechanizability was a substring scan whose indicators included
  ``"with "`` and ``"return "`` — ordinary English. Prose like "joining
  all results *with* the exact delimiter" registered as code.

The honest outcome after the fix is still a small number: this playbook is
mostly behavioural heuristics, which are not convertible into Python tools
and should never graduate. The gates now admit the genuinely mechanizable
ones and the telemetry explains the remainder.
"""

import pytest

from ghost_agent.core.dream import (
    _GRADUATION_MIN_FREQUENCY, _looks_mechanizable,
)
from ghost_agent.core.learning_health import _graduation_eligibility


# ──────────────────────────────────────────────────────────────────────
# Mechanizability must not fire on prose
# ──────────────────────────────────────────────────────────────────────

class TestMechanizabilityDetector:
    @pytest.mark.parametrize("prose", [
        # Every one of these matched the OLD substring detector.
        "Ensure the final output is constructed as a single string, joining "
        "all individual results with the exact delimiter specified.",
        "When executing multiple unrelated tool commands, the agent must "
        "explicitly correlate the output of each command with its request.",
        "Always return to the user with a summary before proceeding.",
        "The agent should import context from the previous turn.",
        "Open the discussion with a clarifying question.",
    ])
    def test_prose_is_not_mistaken_for_code(self, prose):
        assert _looks_mechanizable(prose) is False, (
            "English prose must not register as code — this is what made 17 "
            "of 50 live lessons look mechanizable")

    @pytest.mark.parametrize("code", [
        "```python\nimport json\nprint(json.dumps({}))\n```",
        "def solve(payload):\n    return payload['x'] * 2",
        "import subprocess\nsubprocess.run(['ls'])",
        "$ pip install requests",
        "result = compute(sys.argv[1])",
        "Use json.loads on the payload before indexing it.",
    ])
    def test_real_code_is_detected(self, code):
        assert _looks_mechanizable(code) is True

    def test_empty_and_none_are_safe(self):
        assert _looks_mechanizable("") is False
        assert _looks_mechanizable(None) is False


# ──────────────────────────────────────────────────────────────────────
# Reusability floor
# ──────────────────────────────────────────────────────────────────────

class TestReusabilityFloor:
    def test_floor_is_reachable(self):
        """5 was unreachable in practice; the floor must be lower."""
        assert _GRADUATION_MIN_FREQUENCY < 5
        assert _GRADUATION_MIN_FREQUENCY >= 2, "don't graduate one-offs"


# ──────────────────────────────────────────────────────────────────────
# Eligibility telemetry — "0 graduated" must be explainable
# ──────────────────────────────────────────────────────────────────────

def _lesson(**kw):
    base = {"task": "t", "solution": "prose only", "frequency": 1}
    base.update(kw)
    return base


class TestEligibilityBreakdown:
    def test_reports_all_three_counts(self):
        out = _graduation_eligibility([_lesson()])
        assert set(out) == {"graduation_reusable", "graduation_mechanizable",
                            "graduation_eligible"}

    def test_the_live_shape_reports_zero_eligible_with_a_reason(self):
        """Frequent-but-behavioural + mechanizable-but-rare = empty
        intersection. The breakdown makes that visible instead of leaving
        a bare '0 graduated'."""
        pb = (
            [_lesson(frequency=8, solution="the agent must correlate output")] * 3
            + [_lesson(frequency=1, solution="def f():\n    return 1")] * 2
        )
        out = _graduation_eligibility(pb)
        assert out["graduation_reusable"] == 3
        assert out["graduation_mechanizable"] == 2
        assert out["graduation_eligible"] == 0

    def test_verified_lesson_counts_as_reusable(self):
        """A verifier-confirmed lesson is trustworthy even at frequency 1 —
        that is the alternative signal that unblocks graduation."""
        out = _graduation_eligibility(
            [_lesson(frequency=1, verified=True,
                     solution="import json\nprint(json.dumps({}))")])
        assert out["graduation_eligible"] == 1

    def test_frequent_and_mechanizable_is_eligible(self):
        out = _graduation_eligibility(
            [_lesson(frequency=_GRADUATION_MIN_FREQUENCY,
                     solution="def go():\n    return 2")])
        assert out["graduation_eligible"] == 1

    def test_already_graduated_lessons_are_excluded(self):
        out = _graduation_eligibility(
            [_lesson(frequency=9, verified=True, graduated=True,
                     solution="def go():\n    return 2")])
        assert out["graduation_eligible"] == 0

    def test_never_raises_on_junk(self):
        for junk in ([], [{}], [{"solution": None, "frequency": "x"}]):
            _graduation_eligibility(junk)
