"""Queue #8 — the calibration WIN/LOSS verdict needs an interval.

`learning_health` called the confidence model a winner whenever `brier_cv`
sat below `brier_base_rate` by more than 1e-6 — a tolerance that makes any
numerical wobble a victory. Measured on the live store 2026-08-21: brier_cv
0.03889 against a base rate of 0.03998, a delta of −0.00108 whose 95%
paired-bootstrap CI is **[−0.00246, +0.00017]**. It straddles zero, so the
honest verdict is "indistinguishable"; the rendered one was "beats the
base-rate predictor".

Same defect as the experiment report's "no difference detected yet" (see
`test_experiments_power_and_coverage.py`): a verdict stated without the power
to support it — here on the instrument every keep/kill decision reads.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
import random
import shutil
from pathlib import Path

import pytest

from ghost_agent.core import calibration as C
from ghost_agent.core.learning_health import render_learning_health


# ──────────────────────────────────────────────────────────────────────
# _cv_delta_ci
# ──────────────────────────────────────────────────────────────────────

def _pairs(n, *, signal, seed=3):
    """`signal` 0.0 = composite carries nothing (model can only match the
    base rate); 1.0 = composite is the outcome (model must win)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        y = 1.0 if rng.random() < 0.8 else 0.0
        c = y * signal + rng.random() * (1.0 - signal)
        out.append((c, y))
    return out


class TestCvDeltaCI:
    def test_the_interval_brackets_its_own_point_estimate(self):
        res = C._cv_delta_ci(_pairs(400, signal=0.0))

        assert res is not None
        delta, lo, hi = res
        assert lo <= delta <= hi

    def test_a_real_signal_puts_the_whole_interval_below_zero(self):
        """The gate must still be able to say "beats" — a fix that only ever
        says "indistinguishable" would be as useless as the one it replaces,
        and would look identical on the live store."""
        res = C._cv_delta_ci(_pairs(600, signal=1.0))

        assert res is not None
        _d, lo, hi = res
        assert hi < 0.0

    def test_no_signal_is_never_credited_with_a_win(self):
        """The discriminating property, stated as what must NOT happen. Note
        the measured behaviour is stronger than "straddles zero": fitting a
        2-parameter Platt map on noise costs a little out-of-sample, so the
        interval can sit entirely ABOVE zero (the model is genuinely worse
        than a constant). Asserting the straddle specifically would pin an
        accident of the fixture rather than the guarantee."""
        res = C._cv_delta_ci(_pairs(600, signal=0.0))

        assert res is not None
        _d, _lo, hi = res
        assert not (hi < 0.0)

    def test_is_deterministic(self):
        p = _pairs(300, signal=0.0)

        assert C._cv_delta_ci(p) == C._cv_delta_ci(p)

    def test_too_few_rows_returns_None_rather_than_a_wide_guess(self):
        assert C._cv_delta_ci(_pairs(5, signal=0.0)) is None


# ──────────────────────────────────────────────────────────────────────
# The params round-trip
# ──────────────────────────────────────────────────────────────────────

class TestParamsRoundTrip:
    def _tracker(self, tmp_path, rows):
        d = tmp_path / "calibration"
        d.mkdir(parents=True, exist_ok=True)
        with (d / "calibration.jsonl").open("w") as f:
            for c, y in rows:
                f.write(json.dumps({
                    "composite": c, "entropy_component": 0.5,
                    "competence_component": c, "uncertainty_pressure": 0.0,
                    "outcome": y, "domain": "",
                    "ts": "2026-08-01T00:00:00.000000Z",
                }) + "\n")
        return C.CalibrationTracker(d)

    def test_a_fit_records_the_interval(self, tmp_path):
        t = self._tracker(tmp_path, _pairs(400, signal=0.0))
        p = t.fit()

        assert p is not None
        assert p.brier_cv_delta_lo is not None
        assert p.brier_cv_delta_hi is not None
        assert p.brier_cv_delta_lo <= p.brier_cv_delta_hi

    def test_the_interval_survives_save_and_load(self, tmp_path):
        t = self._tracker(tmp_path, _pairs(400, signal=0.0))
        written = t.fit()
        loaded = t.load_params()

        assert loaded.brier_cv_delta_lo == written.brier_cv_delta_lo
        assert loaded.brier_cv_delta_hi == written.brier_cv_delta_hi

    def test_a_LEGACY_params_file_loads_as_None_not_zero(self, tmp_path):
        """A zero-width interval at zero would read as "the delta is exactly
        zero" and license a bogus verdict. Absent must stay absent."""
        t = self._tracker(tmp_path, _pairs(400, signal=0.0))
        t.fit()
        raw = json.loads(t.params_path.read_text())
        raw.pop("brier_cv_delta_lo", None)
        raw.pop("brier_cv_delta_hi", None)
        t.params_path.write_text(json.dumps(raw))

        loaded = t.load_params()

        assert loaded.brier_cv_delta_lo is None
        assert loaded.brier_cv_delta_hi is None


# ──────────────────────────────────────────────────────────────────────
# The rendered verdict
# ──────────────────────────────────────────────────────────────────────

class TestRenderedVerdict:
    def _render(self, tmp_path, *, lo, hi, cv=0.0388, base=0.0399):
        sysdir = tmp_path / "system"
        (sysdir / "memory").mkdir(parents=True, exist_ok=True)
        (sysdir / "calibration").mkdir(parents=True, exist_ok=True)
        params = {
            "w_entropy": 0.0, "w_competence": 0.5, "threshold": 0.8,
            "lambda_uncertainty": 0.0, "brier": 0.0387, "n_samples": 1006,
            "fitted_at": "2026-08-21T00:00:00Z", "schema": C.SCHEMA_VERSION,
            "brier_raw": 0.0440, "brier_base_rate": base, "brier_cv": cv,
            "n_negative": 48, "epoch": C.CURRENT_EPOCH,
        }
        if lo is not None:
            params["brier_cv_delta_lo"] = lo
            params["brier_cv_delta_hi"] = hi
        (sysdir / "calibration" / "calibration_params.json").write_text(
            json.dumps(params))
        (sysdir / "calibration" / "calibration.jsonl").write_text("")
        out = render_learning_health(sysdir / "memory")
        return next((l for l in out.splitlines() if "base-rate" in l), "")

    def test_a_straddling_interval_reads_INDISTINGUISHABLE(self, tmp_path):
        """The live shape, 2026-08-21."""
        line = self._render(tmp_path, lo=-0.00250, hi=+0.00019)

        assert "INDISTINGUISHABLE" in line
        assert "straddles zero" in line
        assert "beats" not in line

    def test_a_genuine_win_still_reads_beats(self, tmp_path):
        line = self._render(tmp_path, lo=-0.0120, hi=-0.0031)

        assert "beats" in line
        assert "INDISTINGUISHABLE" not in line

    def test_a_genuine_loss_still_reads_LOSES_TO(self, tmp_path):
        line = self._render(tmp_path, lo=+0.0031, hi=+0.0120)

        assert "LOSES TO" in line

    def test_a_legacy_params_file_labels_its_fallback(self, tmp_path):
        """No CI recorded must not silently reuse the old bare claim — the
        reader has to be able to tell a tested verdict from a point
        comparison."""
        line = self._render(tmp_path, lo=None, hi=None)

        assert "point estimate only" in line
        assert "no CI recorded" in line

    def test_the_interval_is_shown_next_to_the_verdict(self, tmp_path):
        line = self._render(tmp_path, lo=-0.00250, hi=+0.00019)

        assert "95% CI of the delta" in line
        assert "-0.00250" in line and "+0.00019" in line
