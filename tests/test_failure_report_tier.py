"""Tier 2 — failure reports as ground-truth negatives (2026-07-27, later 10).

`classify_user_correction` needs a rebuttal PHRASE plus a REPHRASE of the
original request. Measured over the stored session history: **0 of 246
eligible turns fired**, while 20 (8.1%) unmistakably reported that the
delivered work was broken. The classifier was not malfunctioning — it is
structurally blind to how corrections actually arrive here, which is a
pasted traceback or "it still doesn't work", neither of which contains a
rebuttal phrase or shares tokens with the original request.

Every string in `REAL_FAILURE_REPORTS` and `REAL_NON_REPORTS` below is a
verbatim message from that scan, so this file is a regression corpus, not
invented examples.
"""

import pytest

from ghost_agent.distill.user_correction import (
    classify_failure_report, classify_user_correction,
)


# Verbatim from the session scan — all 20 were real reports of broken work.
REAL_FAILURE_REPORTS = [
    "the pages all show, i enter data , but i have to refresh in order to see my entries, fix that.",
    "the game won't start i don't see a page , just a list of files",
    "game.js:45 Failed to load frame definitions: TypeError: frameData.animations.forEach is not a function",
    "Failed to load resource: the server responded with a status of 404 ()     --- when i click start game nothing happens",
    "Uncaught TypeError: enemyManager.loadLevel is not a function\n    at loadLevel (game.js:59:18)",
    "it still does the same. the game never starts, notify me in slack when you fix it.",
    "now the os is not working properly, get a screenshot to see how it looks and fix it.",
    "Minesweeper right click doesn't work",
    'i see the roms now, but i get "error loading emulator. EMULATOR is not defined"',
    "loader.js:57 Failed to load emulator.min.js",
    "resize works , but moving a window doesn't work",
    "new game in the arkanoid game doesn't work, please fix it.",
    "Uncaught SyntaxError: Identifier 'MAP_EFLG_SOLID' has already been declared (at enemy.js:1:1)",
    "(index):138 Uncaught ReferenceError: ScreenSystem is not defined",
    "screens.js:771 Uncaught TypeError: this.initHighScores is not a function",
]

# Also verbatim — these must NOT be labelled failures.
REAL_NON_REPORTS = [
    # PRAISE that mentions a fix and errors — the inversion trap.
    "menu-bounce fix works — START GAME now clears all overlays, the loop "
    "runs at 60fps with zero console errors.",
    # Ordinary conversation containing "mistake".
    "tell me about one mistake that you have learned from.",
    # Agreement containing "not".
    "you are right, but not just that, the concepts behind your design",
    # The daily-briefing habit — a near-identical RE-ASK, not a complaint.
    "hello ghost, what's new today ?",
    # A feature request containing "not just".
    "add more wallpapers, not just gradients, use real images too.",
    "nope, nothing for now, mark the project as done , if it's not already.",
]


class TestDetectsRealReports:
    @pytest.mark.parametrize("msg", REAL_FAILURE_REPORTS)
    def test_real_reports_are_detected(self, msg):
        v = classify_failure_report(msg)
        assert v.is_failure_report, f"missed a real failure report: {msg[:60]}"
        assert v.signals and v.confidence > 0.0
        assert v.reason.startswith("user-reported failure")

    def test_the_old_classifier_sees_none_of_them(self):
        """Pins WHY this tier exists: the correction classifier is blind to
        every one of these, so they were all being discarded."""
        seen = sum(
            1 for m in REAL_FAILURE_REPORTS
            if classify_user_correction(
                prev_user_request="build me a game",
                prev_assistant_response="Done — the game is ready.",
                current_user_text=m).is_correction)
        assert seen == 0


class TestDirectionGuard:
    @pytest.mark.parametrize("msg", REAL_NON_REPORTS)
    def test_non_reports_are_rejected(self, msg):
        v = classify_failure_report(msg)
        assert not v.is_failure_report, f"false positive on: {msg[:60]}"

    def test_praise_about_a_fix_is_not_a_negative(self):
        """THE inversion risk: 'fix works ... zero console errors' mentions
        a fix and errors while being a compliment. Recording it as a
        negative would flip the label — the same failure mode as a tolerant
        parser that silently returns a wrong answer."""
        v = classify_failure_report(
            "menu-bounce fix works — the loop runs at 60fps with zero console errors.")
        assert not v.is_failure_report

    @pytest.mark.parametrize("msg", [
        "perfect, fix it exactly like that next time",
        "that's right, fix that one the same way",
        "works great, nothing to fix it seems",
    ])
    def test_veto_fires_on_forward_looking_instruction(self, msg):
        """A bare 'fix it' is ambiguous; praise recasts it as instruction
        rather than complaint. This is the ONLY case the veto may claim."""
        v = classify_failure_report(msg)
        assert not v.is_failure_report
        assert "affirmation-veto" in v.signals

    @pytest.mark.parametrize("msg", [
        "the minesweeper right click doesn't work though, but the rest looks good",
        "this is looking good. but the pages need manual reload to get new data",
        "the menu works great now, but: Uncaught TypeError: x is not a function",
    ])
    def test_a_softener_does_not_cancel_a_named_breakage(self, msg):
        """Mixed messages are the COMMON shape. An earlier version of the
        guard vetoed these as praise and silently discarded exactly the
        ground truth this tier exists to capture — an over-aggressive guard
        losing real signal, the same class of bug as the parser that
        inverted a verdict."""
        v = classify_failure_report(msg)
        assert v.is_failure_report, f"lost a real defect report: {msg[:60]}"
        assert "affirmation-veto" not in v.signals

    @pytest.mark.parametrize("msg", [
        "works great now, no errors at all",
        "perfect, that fixed it",
        "no errors now, looks good",
    ])
    def test_success_reports_are_not_negatives(self, msg):
        assert not classify_failure_report(msg).is_failure_report

    def test_a_hard_diagnostic_outranks_praise(self):
        """'it works but here is a traceback' still means something broke —
        the veto must not swallow a pasted stack trace."""
        v = classify_failure_report(
            "the menu works great now, but: Uncaught TypeError: "
            "loadLevel is not a function at game.js:59")
        assert v.is_failure_report
        assert "diagnostic" in v.signals

    def test_bare_error_word_is_not_a_diagnostic(self):
        """'error'/'errors' alone false-positives on success reports; the
        diagnostic signal requires real structure."""
        v = classify_failure_report("there were no errors in the console")
        assert not v.is_failure_report


class TestRobustness:
    @pytest.mark.parametrize("junk", ["", "   ", None, 12345, "\x00\x01"])
    def test_never_raises(self, junk):
        v = classify_failure_report(junk)
        assert v.is_failure_report in (True, False)

    def test_empty_input_is_not_a_report(self):
        assert not classify_failure_report("").is_failure_report
        assert not classify_failure_report("   ").is_failure_report


class TestPrecisionOnTheRealCorpus:
    def test_detection_rate_matches_the_measured_scan(self):
        """The scan found 20 of 246 (8.1%). Guard against a future edit
        that quietly widens the net — a detector that starts firing on a
        quarter of all turns is matching conversation, not failures."""
        detected = sum(1 for m in REAL_FAILURE_REPORTS
                       if classify_failure_report(m).is_failure_report)
        assert detected == len(REAL_FAILURE_REPORTS)
        false_pos = sum(1 for m in REAL_NON_REPORTS
                        if classify_failure_report(m).is_failure_report)
        assert false_pos == 0


class TestCalibrationWiring:
    def test_grade_is_fractional_not_zero(self):
        """An explicit correction earns 0.0; a report is a notch above it
        because attribution to THIS turn is slightly looser."""
        from ghost_agent.core.calibration import _FAILURE_REPORT_GRADE
        assert 0.0 < _FAILURE_REPORT_GRADE < 0.5

    def test_records_a_tagged_fractional_negative(self):
        import tempfile, types
        from pathlib import Path
        from collections import OrderedDict
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.core.calibration import (
            CalibrationTracker, _FAILURE_REPORT_GRADE)

        with tempfile.TemporaryDirectory() as d:
            ct = CalibrationTracker(Path(d), min_samples_for_fit=1)
            ctx = types.SimpleNamespace(
                calibration_tracker=ct,
                _recent_calib_for_correction=OrderedDict({"FP": {
                    "composite": 0.9, "entropy_component": 0.5,
                    "competence_component": 0.9, "uncertainty_pressure": 0.0,
                    "entropy_observed": False, "effort_component": 0.4,
                    "effort_observed": True}}))
            a = GhostAgent.__new__(GhostAgent)
            a.context = ctx
            v = classify_failure_report("Uncaught TypeError: x is not a function")
            assert a._record_failure_report_negative("FP", v) is True
            s = ct._load_samples()[0]
            assert s.outcome == _FAILURE_REPORT_GRADE
            assert s.source == "failure_report"
            # provenance from the stash survives
            assert s.effort_observed is True

    def test_cannot_double_count_one_turn(self):
        """The stash is consumed, so a report AND a correction on the same
        prior turn file at most one negative between them."""
        import tempfile, types
        from pathlib import Path
        from collections import OrderedDict
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.core.calibration import CalibrationTracker

        with tempfile.TemporaryDirectory() as d:
            ct = CalibrationTracker(Path(d), min_samples_for_fit=1)
            ctx = types.SimpleNamespace(
                calibration_tracker=ct,
                _recent_calib_for_correction=OrderedDict({"FP": {
                    "composite": 0.9, "entropy_component": 0.5,
                    "competence_component": 0.9, "uncertainty_pressure": 0.0}}))
            a = GhostAgent.__new__(GhostAgent)
            a.context = ctx
            v = classify_failure_report("it still doesn't work")
            assert a._record_failure_report_negative("FP", v) is True
            assert a._record_failure_report_negative("FP", v) is False
            assert len(ct._load_samples()) == 1

    def test_missing_stash_entry_is_a_clean_noop(self):
        import tempfile, types
        from pathlib import Path
        from collections import OrderedDict
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.core.calibration import CalibrationTracker

        with tempfile.TemporaryDirectory() as d:
            ct = CalibrationTracker(Path(d), min_samples_for_fit=1)
            ctx = types.SimpleNamespace(
                calibration_tracker=ct,
                _recent_calib_for_correction=OrderedDict())
            a = GhostAgent.__new__(GhostAgent)
            a.context = ctx
            v = classify_failure_report("it still doesn't work")
            assert a._record_failure_report_negative("nope", v) is False
            assert ct._load_samples() == []
