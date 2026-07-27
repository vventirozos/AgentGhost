"""Defects found reviewing the same day's own changes — 2026-07-27 (later 6).

Four bugs, three of them introduced earlier the same session. They are the
kind that a green suite does not catch, because each only fires on a path
the tests never construct: a penalised turn that is then corrected, a
correction on an already-failed turn, and an LLM reply whose labels contain
more than one number.

D4 is the important one: a "tolerant" coercion added hours earlier turned a
recoverable parse failure into a SILENT INVERSION of the model's verdict.
"""

import re

import pytest


SRC = (__import__("pathlib").Path(__file__).resolve().parents[1]
       / "src" / "ghost_agent" / "core" / "agent.py").read_text()


# ──────────────────────────────────────────────────────────────────────
# D4 — survivor-index coercion must not fabricate an index
# ──────────────────────────────────────────────────────────────────────

def _coerce(label, n_hypotheses):
    """Mirrors the parsing in agent.py's hypothesis adjudication."""
    out = set()
    m = re.match(r"\D*(\d+)", str(label))
    if m:
        ix = int(m.group(1))
        if 0 <= ix < n_hypotheses:
            out.add(ix)
    return out


class TestSurvivorIndexCoercion:
    @pytest.mark.parametrize("label,expected", [
        ("H0", {0}),
        ("1", {1}),
        ("hypothesis 2", {2}),
        # The regression: stripping ALL non-digits concatenated separate
        # numbers. "H2: retry after 30s" became 230.
        ("H2: retry after 30s", {2}),
        ("H0 — the config at line 42 is wrong", {0}),
        # Out of range must be DISCARDED, not kept as a phantom survivor.
        ("H97", set()),
        ("no numbers here", set()),
        ("", set()),
    ])
    def test_first_digit_run_only_and_range_checked(self, label, expected):
        assert _coerce(label, 3) == expected

    def test_prose_label_no_longer_inverts_the_verdict(self):
        """With the old `re.sub(r"\\D","")`, a prose label produced a
        non-empty out-of-range set — truthy — so the promote/demote loop ran
        and marked EVERY deferred hypothesis refuted. The model said H2
        survived; the code concluded all were refuted."""
        old = {int(re.sub(r"\D", "", "H2: retry after 30s"))}
        assert old == {230}, "documents the old behaviour"
        assert 2 not in old, "the real survivor was lost"
        new = _coerce("H2: retry after 30s", 3)
        assert new == {2}

    def test_source_uses_anchored_match_not_global_strip(self):
        assert 're.match(r"\\D*(\\d+)"' in SRC
        assert 're.sub(r"\\D", "", str(_i))' not in SRC


# ──────────────────────────────────────────────────────────────────────
# D1/D2/D3 — the user-correction stash
# ──────────────────────────────────────────────────────────────────────

class TestCorrectionStash:
    def test_stash_is_gated_on_a_success_outcome(self):
        """D1: a turn already recorded at outcome=0.0 must not be stashed —
        a later correction would record a SECOND 0.0 for the same turn,
        double-weighting it in the Brier/ECE/weight fit."""
        i = SRC.index("_recent_calib_for_correction")
        window = SRC[i:i + 3400]
        # 0.5, not 1.0: under the graded label a clean turn scores the
        # measured prior (0.83) and never reaches 1.0, so a `>= 1.0`
        # gate stops populating the stash entirely — which disables
        # every downstream tier that consumes it.
        assert "if _calib_outcome >= 0.5:" in window

    def test_stash_records_the_pre_penalty_composite(self):
        """D2: the inline record deliberately stores `pre_penalty_composite`
        because the outcome penalty is a function of the label. The stash
        stored the PENALISED value, so a penalised-then-corrected turn
        recorded a ~5x-suppressed prediction against outcome=0.0 — the exact
        optimism bug the inline path documents."""
        i = SRC.index("_recent_calib_for_correction")
        window = SRC[i:i + 3400]
        assert 'getattr(_pending, "pre_penalty_composite"' in window

    def test_stash_carries_both_observation_flags(self):
        """D3: the fit filters samples on `entropy_observed` /
        `effort_observed`. Omitting them meant correction negatives — the
        rarest and most valuable samples — were excluded from the weight
        fits entirely."""
        i = SRC.index("_recent_calib_for_correction")
        window = SRC[i:i + 3400]
        assert '"entropy_observed"' in window
        assert '"effort_observed"' in window
        assert '"effort_component"' in window

    def test_every_stashed_key_is_a_valid_record_kwarg(self):
        """A stray key would raise TypeError at `record(**_comp)` and lose
        the correction negative silently."""
        import inspect
        from ghost_agent.core.calibration import CalibrationTracker
        allowed = set(inspect.signature(CalibrationTracker.record).parameters)
        for key in ("composite", "entropy_component", "competence_component",
                    "uncertainty_pressure", "entropy_observed",
                    "effort_component", "effort_observed"):
            assert key in allowed, f"{key} is not accepted by record()"

    def test_record_accepts_the_stash_shape_end_to_end(self):
        import tempfile
        from pathlib import Path
        from ghost_agent.core.calibration import CalibrationTracker
        stash = {
            "composite": 0.9, "entropy_component": 0.5,
            "competence_component": 0.9, "uncertainty_pressure": 0.0,
            "entropy_observed": True, "effort_component": 0.3,
            "effort_observed": True,
        }
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=1)
            t.record(outcome=0.0, **stash)          # must not raise
            s = t._load_samples()[0]
            assert s.outcome == 0.0
            assert s.effort_observed is True
            assert s.entropy_observed is True
