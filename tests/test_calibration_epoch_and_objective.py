"""Corpus epochs + delivered-Brier weight search — 2026-08-02.

Both fixes here come from one live symptom: ~291 log lines, one pair per
hourly refit for 26 days, reporting that the composite was "anti-correlated
with outcomes" and that the fit was "WORSE than always predicting the base
rate". The guard was firing correctly and its diagnosis was wrong — it was
measuring the CORPUS, not the signal.

  1. **Epochs.** `calibration.jsonl` is append-only and `max_history` (4000)
     exceeded the whole file, so every refit pooled eras that are not
     comparable: the label scheme changed on 2026-07-27 (binary {0,1}, base
     0.955 → graded, base 0.855), `effort`/`entropy` are observed only from
     then on so 76% of rows collapsed to competence alone, and the
     competence prior itself warms up (mean 0.757 → 0.898) as the label mean
     falls. Pooled, the score rises exactly as the labels fall — a Simpson's
     paradox that forces a negative Platt slope. Measured on the live 1709:
     pooled AUC 0.530 / slope −0.077 / map rejected; current epoch alone
     AUC 0.711 / slope +2.242 / map applied and beating the base rate.

  2. **The search optimised the wrong quantity.** It scored candidates on the
     RAW Brier while the pipeline delivers a Platt-mapped one. The raw-best
     point (0.054164, slope −0.077 → REJECTED) beat the runner-up (0.054204,
     slope +0.388 → ACCEPTED) by 4e-5 — a coin flip in the fourth decimal
     decided whether the agent got a probability map at all.

Fixing (2) alone would have introduced a third bug: scoring on the delivered
Brier removes the level penalty that used to suppress useless features, so
the grid starts buying weight for pure noise. The noise gain is heavy-tailed
(median 0, tail ~1e-3, and it does NOT shrink as 1/n), so a tolerance can't
separate it from signal — hence the separation gate, which is the module's
own "a feature must VARY and SEPARATE" rule applied in the fit.
"""

import json
import random
import tempfile
from pathlib import Path

import pytest

from ghost_agent.core.calibration import (
    CURRENT_EPOCH,
    CalibrationTracker,
    _BRIER_TIE_TOL,
    _MIN_SEPARATION_SIGMAS,
    _separation_sigmas,
    epoch_for_ts,
)


@pytest.fixture()
def tracker():
    with tempfile.TemporaryDirectory() as d:
        yield CalibrationTracker(Path(d), min_samples_for_fit=40)


def _rows(t):
    return [json.loads(line) for line in
            t.history_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_raw(t, **fields):
    """Write a row straight to the JSONL, bypassing `record` — the only way
    to simulate a legacy row that predates a field."""
    t.dir.mkdir(parents=True, exist_ok=True)
    with t.history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(fields) + "\n")


# ──────────────────────────────────────────────────────────────────────
# Epoch tagging
# ──────────────────────────────────────────────────────────────────────

class TestEpochTagging:
    def test_new_rows_carry_the_current_epoch(self, tracker):
        tracker.record(composite=0.8, entropy_component=0.5,
                       competence_component=0.8, outcome=1.0)
        assert _rows(tracker)[0]["epoch"] == CURRENT_EPOCH

    def test_untagged_legacy_rows_are_never_promoted(self, tracker):
        """The whole point of the field. A pre-epoch row silently adopting
        the current tag is exactly the contamination being fixed."""
        _append_raw(tracker, composite=0.9, entropy_component=0.5,
                    competence_component=0.9, outcome=1.0,
                    ts="2026-07-10T00:00:00Z")
        loaded = tracker._load_samples()
        assert loaded[0].epoch == "2026-07-07.binary"
        assert loaded[0].epoch != CURRENT_EPOCH

    @pytest.mark.parametrize("ts,expected", [
        ("2026-07-06T23:59:59Z", "2026-07-07.binary"),
        ("2026-07-26T23:59:59Z", "2026-07-07.binary"),
        ("2026-07-27T00:00:00Z", "2026-07-27.graded"),
        ("2026-08-02T12:00:00Z", "2026-07-27.graded"),
    ])
    def test_boundary_derivation(self, ts, expected):
        assert epoch_for_ts(ts) == expected

    @pytest.mark.parametrize("ts", ["", None, "not-a-timestamp", 12345])
    def test_unparsable_timestamp_lands_in_the_oldest_epoch(self, ts):
        """A row we cannot date is certainly not one we should fit on."""
        assert epoch_for_ts(ts) == "2026-07-07.binary"

    def test_explicit_epoch_survives_the_round_trip(self, tracker):
        tracker.record(composite=0.8, entropy_component=0.5,
                       competence_component=0.8, outcome=1.0,
                       epoch="some.other.epoch")
        assert tracker._load_samples()[0].epoch == "some.other.epoch"


# ──────────────────────────────────────────────────────────────────────
# The fit reads one epoch
# ──────────────────────────────────────────────────────────────────────

class TestFitIsEpochScoped:
    def _mixed(self, t, *, n_legacy=400, n_current=80):
        for i in range(n_legacy):
            _append_raw(t, composite=0.5, entropy_component=0.5,
                        competence_component=0.5,
                        outcome=1.0 if i % 10 else 0.0,
                        ts="2026-07-10T00:00:00Z")
        for i in range(n_current):
            t.record(composite=0.9, entropy_component=0.5,
                     competence_component=0.9 if i % 5 else 0.3,
                     outcome=1.0 if i % 5 else 0.0)

    def test_fit_counts_only_the_current_epoch(self, tracker):
        self._mixed(tracker)
        p = tracker.fit()
        assert p is not None
        assert p.n_samples == 80
        assert p.n_excluded_other_epochs == 400
        assert p.epoch == CURRENT_EPOCH

    def test_fit_bails_rather_than_reaching_across_epochs(self, tracker):
        """400 legacy rows are plenty to clear the floor — and must not be
        allowed to. No fit beats a fit on incomparable rows."""
        self._mixed(tracker, n_legacy=400, n_current=5)
        assert tracker.fit() is None

    def test_diagnostics_are_scoped_like_the_fit(self, tracker):
        """A metric computed over a different population than the fit is how
        the stored-composite AUC came to read 0.676 against a true 0.530."""
        for _ in range(50):
            _append_raw(tracker, composite=0.0, entropy_component=0.5,
                        competence_component=0.5, outcome=1.0,
                        ts="2026-07-10T00:00:00Z")   # Brier 1.0 if counted
        for _ in range(50):
            tracker.record(composite=1.0, entropy_component=0.5,
                           competence_component=1.0, outcome=1.0)
        assert tracker.brier_score() == pytest.approx(0.0)
        assert tracker.ece() == pytest.approx(0.0)
        assert sum(b.count for b in tracker.reliability_table()) == 50

    def test_stats_reports_both_scopes(self, tracker):
        for _ in range(10):
            _append_raw(tracker, composite=0.9, entropy_component=0.5,
                        competence_component=0.9, outcome=1.0,
                        ts="2026-07-10T00:00:00Z")
        for _ in range(4):
            tracker.record(composite=0.9, entropy_component=0.5,
                           competence_component=0.9, outcome=1.0)
        st = tracker.stats()
        assert st["samples"] == 4
        assert st["samples_all_epochs"] == 14
        assert st["epoch"] == CURRENT_EPOCH

    def test_retro_negative_inherits_the_closing_turns_epoch(self, tracker):
        """The retro row REUSES the closing turn's stored features, so
        stamping it current would smuggle an older feature set into the live
        fit with a negative label attached."""
        _append_raw(tracker, composite=0.9, entropy_component=0.5,
                    competence_component=0.9, outcome=1.0,
                    ts="2026-07-10T00:00:00Z", req_id="old-req", source="turn")
        assert tracker.record_task_reopened_negative("old-req") is True
        retro = [s for s in tracker._load_samples() if s.source == "task_reopened"]
        assert len(retro) == 1
        assert retro[0].epoch == "2026-07-07.binary"
        assert retro[0].epoch != CURRENT_EPOCH


# ──────────────────────────────────────────────────────────────────────
# The search optimises what the pipeline delivers
# ──────────────────────────────────────────────────────────────────────

def _informative(t, n=300, seed=11):
    """A corpus where turn EFFORT genuinely predicts the outcome."""
    rng = random.Random(seed)
    for _ in range(n):
        good = rng.random() < 0.75
        t.record(composite=0.85, entropy_component=0.5,
                 competence_component=0.85,
                 effort_component=(rng.uniform(0.6, 1.0) if good
                                   else rng.uniform(0.0, 0.4)),
                 effort_observed=True,
                 outcome=1.0 if good else 0.0)


class TestDeliveredBrierObjective:
    def test_a_real_feature_still_earns_its_weight(self, tracker):
        _informative(tracker)
        p = tracker.fit()
        assert p is not None
        assert p.w_effort > 0.0
        assert p.map_status == "applied"

    def test_the_reported_brier_is_the_one_actually_delivered(self, tracker):
        """`brier` must describe the scale the agent will score on. When the
        map is applied that is the calibrated Brier, not the raw one."""
        _informative(tracker)
        p = tracker.fit()
        assert p.map_status == "applied"
        assert p.brier <= p.brier_raw

    def test_fit_beats_the_base_rate_on_a_learnable_corpus(self, tracker):
        """The second of the two live warnings. `brier_base_rate` is the only
        baseline that matters — a model that loses to it adds nothing."""
        _informative(tracker)
        p = tracker.fit()
        assert p.brier < p.brier_base_rate

    def test_ties_break_toward_the_simpler_model(self, tracker):
        """Competence alone is the null model; a weight must buy more than
        `_BRIER_TIE_TOL` to displace it. Guards the 4e-5 coin flip that put
        the live agent on an inverted map for 26 days."""
        for i in range(200):
            tracker.record(composite=0.9, entropy_component=0.5,
                           competence_component=0.9,
                           effort_component=0.9,   # perfectly redundant
                           effort_observed=True,
                           outcome=1.0 if i % 10 else 0.0)
        p = tracker.fit()
        assert p is not None
        assert p.w_effort == 0.0
        assert p.w_competence == 1.0

    def test_tie_tolerance_is_too_small_to_mask_a_real_gain(self):
        """Sizing check, not behaviour: the tolerance exists to break float
        ties. The live effort weight is worth ~1.0e-2 — two orders of
        magnitude above it — so it can never be swallowed as a tie."""
        assert _BRIER_TIE_TOL < 1.0e-2 / 10


# ──────────────────────────────────────────────────────────────────────
# Noise never earns a weight
# ──────────────────────────────────────────────────────────────────────

class TestSeparationGate:
    def test_pure_noise_earns_no_weight(self, tracker):
        """Scoring on the delivered Brier removes the level penalty that used
        to suppress a useless column, so without this gate the grid buys
        w_entropy = 0.2 for a 4.9e-5 in-sample gain."""
        rng = random.Random(7)
        for _ in range(200):
            tracker.record(composite=0.89, entropy_component=rng.random(),
                           competence_component=0.89,
                           outcome=1.0 if rng.random() < 0.9 else 0.0,
                           entropy_observed=True)
        p = tracker.fit()
        assert p is not None
        assert p.w_entropy == 0.0

    def test_noise_is_rejected_across_many_corpora(self, tracker_factory=None):
        """One seed passing could be luck. The gate false-admits ~4.9% of
        pure-noise corpora by construction (a 2.5σ test), so require the
        large majority to be pinned rather than all of them."""
        pinned = 0
        trials = 20
        for seed in range(trials):
            with tempfile.TemporaryDirectory() as d:
                t = CalibrationTracker(Path(d), min_samples_for_fit=40)
                rng = random.Random(seed)
                for _ in range(150):
                    t.record(composite=0.89, entropy_component=rng.random(),
                             competence_component=0.89,
                             outcome=1.0 if rng.random() < 0.9 else 0.0,
                             entropy_observed=True)
                p = t.fit()
                pinned += int(p is not None and p.w_entropy == 0.0)
        assert pinned >= trials - 3

    def test_gate_admits_a_feature_that_separates(self, tracker):
        _informative(tracker)
        p = tracker.fit()
        assert p.w_effort > 0.0

    def test_separation_is_measured_in_standard_errors(self):
        class S:
            def __init__(self, v, o):
                self.effort_component = v
                self.outcome = o
        # Two tight, well-apart clusters — many σ of separation.
        tight = ([S(0.90 + i * 1e-4, 1.0) for i in range(40)] +
                 [S(0.10 + i * 1e-4, 0.0) for i in range(40)])
        assert _separation_sigmas(tight, "effort_component") > _MIN_SEPARATION_SIGMAS
        # Same means, but spread so wide the difference is within noise.
        rng = random.Random(3)
        wide = ([S(rng.gauss(0.5, 1.0), 1.0) for _ in range(40)] +
                [S(rng.gauss(0.5, 1.0), 0.0) for _ in range(40)])
        assert _separation_sigmas(wide, "effort_component") < _MIN_SEPARATION_SIGMAS

    @pytest.mark.parametrize("rows", [
        [],                                   # nothing
        [("a", 1.0)],                         # one row
        [("a", 1.0), ("b", 1.0)],             # one class only
    ])
    def test_undecidable_separation_pins_rather_than_admits(self, rows):
        class S:
            def __init__(self, v, o):
                self.effort_component = v
                self.outcome = o
        samples = [S(0.5, o) for _, o in rows]
        assert _separation_sigmas(samples, "effort_component") == 0.0

    def test_zero_variance_column_is_not_separable(self):
        class S:
            def __init__(self, o):
                self.effort_component = 0.5   # constant
                self.outcome = o
        samples = [S(1.0) for _ in range(30)] + [S(0.0) for _ in range(30)]
        assert _separation_sigmas(samples, "effort_component") == 0.0


# ──────────────────────────────────────────────────────────────────────
# The live regression, replayed
# ──────────────────────────────────────────────────────────────────────

class TestLiveRegression:
    def test_a_pooled_cross_epoch_corpus_no_longer_inverts_the_map(self, tracker):
        """The exact live shape: a large legacy era whose only live feature
        is anti-correlated, plus a smaller current era that genuinely
        predicts. Pooled, this produced slope −0.077 and a map rejected on
        every refit for 26 days. Scoped, it must fit cleanly."""
        rng = random.Random(23)
        # Legacy era: competence rises while outcomes are near-perfect —
        # the warm-up artefact that drives the global negative slope.
        for i in range(600):
            _append_raw(tracker, composite=0.5 + 0.4 * i / 600,
                        entropy_component=0.5,
                        competence_component=0.5 + 0.4 * i / 600,
                        outcome=1.0 if rng.random() < 0.96 else 0.0,
                        ts="2026-07-10T00:00:00Z")
        # Current era: effort actually predicts.
        for _ in range(300):
            good = rng.random() < 0.8
            tracker.record(composite=0.85, entropy_component=0.5,
                           competence_component=0.85,
                           effort_component=(rng.uniform(0.6, 1.0) if good
                                             else rng.uniform(0.0, 0.4)),
                           effort_observed=True,
                           outcome=1.0 if good else 0.0)
        p = tracker.fit()
        assert p is not None
        assert p.n_samples == 300 and p.n_excluded_other_epochs == 600
        assert p.map_status == "applied", "the map must no longer invert"
        assert p.platt_a > 0.0
        assert p.brier < p.brier_base_rate, "must beat the base-rate baseline"

    def test_params_round_trip_through_disk(self, tracker):
        _informative(tracker)
        p = tracker.fit()
        again = tracker.load_params()
        assert again is not None
        assert again.epoch == p.epoch
        assert again.n_excluded_other_epochs == p.n_excluded_other_epochs
        assert again.map_status == p.map_status

    def test_params_file_without_epoch_still_loads(self, tracker):
        """A params file written before epochs existed must degrade to the
        empty default, not to `None` (which would drop the agent back to the
        hardcoded threshold on restart)."""
        _informative(tracker)
        tracker.fit()
        d = json.loads(tracker.params_path.read_text(encoding="utf-8"))
        d.pop("epoch", None)
        d.pop("n_excluded_other_epochs", None)
        tracker.params_path.write_text(json.dumps(d), encoding="utf-8")
        loaded = tracker.load_params()
        assert loaded is not None
        assert loaded.epoch == ""
        assert loaded.n_excluded_other_epochs == 0


# ──────────────────────────────────────────────────────────────────────
# Self-review pass — defects found by reviewing the changes above
# ──────────────────────────────────────────────────────────────────────

class TestSelfReviewFixes:
    """Every case here is a defect the first pass shipped and the review
    caught. They are all one shape: a new filter applied to SOME of the
    paths that read the filtered data, leaving the rest describing a
    population that no longer exists."""

    def test_a_perfect_separator_is_not_pinned(self):
        """The gate's own worst failure mode. A feature constant WITHIN each
        class and different BETWEEN them has zero within-class variance, so
        the standard error is 0 — and the first version returned 0.0σ for
        that, pinning the single best feature obtainable."""
        class S:
            def __init__(self, v, o):
                self.effort_component = v
                self.outcome = o
        # Values chosen to be EXACTLY representable in binary, so the
        # within-class variance is truly 0.0 and the `se == 0` branch is the
        # one under test. (0.9/0.2 leave ~1e-18 of float residue, which takes
        # the ordinary path and merely yields a huge finite σ — still correct,
        # but it would not exercise the branch.)
        perfect = ([S(0.75, 1.0) for _ in range(20)] +
                   [S(0.25, 0.0) for _ in range(20)])
        assert _separation_sigmas(perfect, "effort_component") == float("inf")
        # What actually matters, and for the float-residue case too:
        for lo, hi in ((0.25, 0.75), (0.2, 0.9)):
            rows = ([S(hi, 1.0) for _ in range(20)] +
                    [S(lo, 0.0) for _ in range(20)])
            assert _separation_sigmas(rows, "effort_component") >= _MIN_SEPARATION_SIGMAS

    def test_a_constant_column_is_still_undecidable(self):
        """Same zero denominator, opposite answer — the two must not be
        conflated. Constant across BOTH classes means no information."""
        class S:
            def __init__(self, o):
                self.effort_component = 0.5
                self.outcome = o
        flat = [S(1.0) for _ in range(20)] + [S(0.0) for _ in range(20)]
        assert _separation_sigmas(flat, "effort_component") == 0.0

    def test_perfect_separator_earns_weight_end_to_end(self, tracker):
        """The unit above, through the real fit."""
        for i in range(120):
            good = bool(i % 4)
            tracker.record(composite=0.8, entropy_component=0.5,
                           competence_component=0.8,
                           effort_component=0.9 if good else 0.2,
                           effort_observed=True,
                           outcome=1.0 if good else 0.0)
        p = tracker.fit()
        assert p is not None and p.w_effort > 0.0


def _health(tmpdir, rows):
    """Build a learning-health report over `rows` (raw dicts)."""
    from ghost_agent.core.learning_health import collect_learning_health
    md = Path(tmpdir) / "memory"
    md.mkdir(parents=True, exist_ok=True)
    cal = Path(tmpdir) / "calibration"
    cal.mkdir(parents=True, exist_ok=True)
    with (cal / "calibration.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            r.setdefault("epoch", CURRENT_EPOCH)
            fh.write(json.dumps(r) + "\n")
    return collect_learning_health(md)


class TestTelemetryMatchesTheMechanism:
    def test_feature_verdict_uses_the_fits_gate_not_a_raw_delta(self, tmp_path):
        """The live contradiction: entropy separated by 0.0421 (over the old
        0.02 delta cut → "live") at 0.63σ (under the fit's 2.5σ → PINNED), so
        two adjacent lines of one report disagreed and the `features: N/4
        live` headline counted a feature the fit refuses to weight."""
        rows = []
        for i in range(400):
            ok = bool(i % 8)
            # Wide spread, tiny mean difference → big delta-vs-sigma gap.
            rows.append({"entropy_component": (0.52 if ok else 0.48)
                         + (i % 37) / 40.0,
                         "entropy_observed": True,
                         "outcome": 1.0 if ok else 0.0})
        cal = _health(tmp_path, rows)["calibration"]
        fh = cal["feature_health"]["entropy_component"]
        assert abs(fh["separation"]) > 0.0
        assert fh["separation_sigmas"] < cal["separation_min_sigmas"]
        assert fh["verdict"] == "dead"
        assert "entropy_component" not in cal["live_features"]
        assert cal["entropy_learnable"] is False   # and the two AGREE

    def test_effort_is_judged_only_on_rows_that_measured_it(self, tmp_path):
        """Fixed for entropy in 2026-07-27, never generalised: effort was
        judged over rows carrying the neutral 0.5 stand-in, so the reported
        separation came from a population the fit never reads."""
        rows = ([{"effort_component": 0.5, "outcome": 1.0}] * 300 +
                [{"effort_component": 0.9, "effort_observed": True,
                  "outcome": 1.0}] * 40 +
                [{"effort_component": 0.2, "effort_observed": True,
                  "outcome": 0.0}] * 40)
        cal = _health(tmp_path, rows)["calibration"]
        fh = cal["feature_health"]["effort_component"]
        assert fh["n"] == 80, "stand-ins must not be judged"
        assert fh["separation"] == pytest.approx(0.7, abs=0.01)
        assert fh["verdict"] == "live"

    def test_graded_labels_are_not_invisible_in_the_outcome_counts(self, tmp_path):
        """`outcome_pos`/`outcome_neg` counted only EXACT 1.0/0.0, so under
        graded labels most of the corpus vanished from the line (155+/7- of
        541 rows). Split at 0.5 like every gate in calibration.py, and keep
        the verifier-checked anchors separately."""
        rows = ([{"competence_component": 0.8, "outcome": 0.83}] * 50 +
                [{"competence_component": 0.8, "outcome": 1.0}] * 10 +
                [{"competence_component": 0.8, "outcome": 0.15}] * 5 +
                [{"competence_component": 0.8, "outcome": 0.0}] * 3)
        cal = _health(tmp_path, rows)["calibration"]
        assert cal["outcome_pos"] == 60 and cal["outcome_neg"] == 8
        assert cal["outcome_pos"] + cal["outcome_neg"] == 68  # nothing lost
        assert cal["outcome_verified_pos"] == 10
        assert cal["outcome_verified_neg"] == 3

    def test_rendered_ratios_share_one_denominator(self, tmp_path):
        """The half-migrated view: an epoch-scoped numerator over a
        whole-file denominator printed a fraction contradicting its own
        percentage (418/1709 shown as 77.3%)."""
        from ghost_agent.core.learning_health import render_learning_health
        rows = [{"entropy_component": 0.4 + (i % 5) / 10.0,
                 "entropy_observed": True,
                 "outcome": 1.0 if i % 6 else 0.0} for i in range(60)]
        md = tmp_path / "memory"
        md.mkdir(parents=True, exist_ok=True)
        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        with (cal_dir / "calibration.jsonl").open("w", encoding="utf-8") as fh:
            for _ in range(200):   # legacy rows, excluded from the fit
                fh.write(json.dumps({"entropy_component": 0.5, "outcome": 1.0,
                                     "ts": "2026-07-10T00:00:00Z"}) + "\n")
            for r in rows:
                r["epoch"] = CURRENT_EPOCH
                fh.write(json.dumps(r) + "\n")
        text = render_learning_health(md)
        line = next(ln for ln in text.splitlines()
                    if "entropy observed on" in ln)
        assert "/60 samples (100.0%)" in line, line
        assert "/260" not in line, "whole-file denominator leaked into a scoped ratio"
