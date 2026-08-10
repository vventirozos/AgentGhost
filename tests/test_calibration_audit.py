"""Calibration + competence: honest numbers, and a dead consumer that says so.

AUDIT 2026-08-10. These instruments fed decisions and had never been checked.
What the audit found, and what each of these tests now pins:

  1. The headline Brier was IN-SAMPLE — a 2-parameter Platt map scored on the
     rows it was fitted to — printed beside `brier_base_rate`, a comparison
     only one side of which paid for its parameters.
  2. `[DEAD]` verdicts came from a difference-of-means gate over **18
     negatives in 694 rows**. It called `competence_component` dead at
     0.05σ while ABLATING it cost 0.021 AUC. The gate cannot resolve what it
     judges, and nobody had ever run the ablation.
  3. Competence cells rendered into the PROMPT with no precision:
     `sql: 74% (n=17)` beside `fetch: 99% (n=847)`. sql's 95% CI is
     [0.50, 0.89] — a coin-flip-wide estimate stated as fact.
  4. ⚠ THE BIG ONE. The threshold's only consumer is metacog arbitration,
     hard-gated by `_METACOG_ARBITER_ENABLED` in core/agent.py. Measured over
     209 metacog summaries in the LIVE log: confidence computed 865 times,
     `below_threshold` fired 118 times, **arbitrations 0**. The stack is a
     measurement with no behaviour attached, and nothing said so.

The through-line: none of these failed loudly. Every one produced a plausible
number. That is why they survived this long.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from ghost_agent.core import learning_health as LH  # noqa: E402
from ghost_agent.core.calibration import (  # noqa: E402
    CalibrationTracker,
    FittedParams,
    _cv_brier,
    _fit_platt,
    apply_platt,
)
from ghost_agent.memory.competence import wilson_interval  # noqa: E402


# ── out-of-sample Brier ─────────────────────────────────────────────────────

def _pairs(n=200, sep=True):
    """Composites that DO carry signal, so a real CV number is produced."""
    out = []
    for i in range(n):
        good = i % 4 != 0
        out.append((0.9 if good else 0.4, 1.0 if good else 0.0))
    return out


def test_cv_brier_is_deterministic():
    """An audit trail that moves on its own is not one — two runs on the same
    rows must give the identical number."""
    p = _pairs()
    assert _cv_brier(p) == _cv_brier(p)


def test_cv_brier_is_out_of_sample_and_so_never_beats_in_sample():
    """⚠ THE DEFECT. In-sample scores the fit on its own rows and is
    optimistic by construction; the honest number cannot be better."""
    p = _pairs()
    a, b = _fit_platt(p)
    in_sample = sum((apply_platt(c, a, b) - y) ** 2 for c, y in p) / len(p)
    assert _cv_brier(p) >= in_sample - 1e-9


def test_cv_brier_declines_to_answer_on_too_few_rows():
    """None, not a confident number off 3 rows."""
    assert _cv_brier([(0.9, 1.0), (0.4, 0.0), (0.8, 1.0)]) is None
    assert _cv_brier([]) is None


# ── the Wilson interval that goes into the PROMPT ───────────────────────────

@pytest.mark.parametrize("p,n", [(0.74, 17), (0.99, 847), (0.98, 44),
                                 (1.0, 10), (0.0, 5), (0.5, 1), (0.87, 1238)])
def test_the_interval_is_always_POSSIBLE(p, n):
    """⚠ MY OWN FIRST FIX FAILED THIS. It rendered "98% ±6%", which reads as
    [92%, 104%] — an interval that cannot exist. Wilson is asymmetric near
    the boundary, so `p ± half` is not the interval. A fix about honest
    precision must not state an impossible one; caught by a sanity check on
    the helper rather than by reading it."""
    lo, hi = wilson_interval(p, n)
    assert 0.0 <= lo <= hi <= 1.0


def test_no_observations_means_no_precision_claim():
    assert wilson_interval(0.5, 0) == (0.0, 1.0)
    assert wilson_interval(0.9, -1) == (0.0, 1.0)


def test_small_n_is_wide_and_large_n_is_tight():
    """The whole point: 17 observations and 847 must not look alike."""
    lo_s, hi_s = wilson_interval(0.74, 17)
    lo_l, hi_l = wilson_interval(0.99, 847)
    assert (hi_s - lo_s) > 0.30, "a 17-sample cell rendered as precise"
    assert (hi_l - lo_l) < 0.05
    # The live values the audit measured.
    assert round(lo_s, 2) == 0.50 and round(hi_s, 2) == 0.89


def test_the_prompt_block_carries_precision_and_flags_provisional(tmp_path):
    """The CI must reach the RENDERED PROMPT, not merely exist in the module.

    ⚠ THE FIRST VERSION OF THIS TEST WAS A FALSE NEGATIVE. It asserted
    `"wilson_interval" in source` — so deleting the CI from the actual output
    line still passed, because the assignment above it survived. Revert-testing
    caught it: the mutation stayed GREEN. A guard that checks a symbol exists
    proves nothing about what the symbol reaches; assert the output.
    """
    from ghost_agent.memory.competence import CompetenceProfile
    prof = CompetenceProfile(tmp_path)
    for _ in range(13):                       # small-n, wide interval
        prof.record("sql", "execute", True)
    for _ in range(4):
        prof.record("sql", "execute", False)
    for i in range(800):                      # large-n, tight interval
        prof.record("fetch", "web_fetch", i % 100 != 0)

    out = prof.get_context_string()
    assert "95% CI" in out, "cells render without precision again"
    assert "provisional" in out, "the small-n cell is no longer flagged"

    sql_line = next(l for l in out.splitlines() if l.strip().startswith("- sql"))
    fetch_line = next(l for l in out.splitlines()
                      if l.strip().startswith("- fetch"))
    assert "provisional" in sql_line, "n=17 rendered as if it were precise"
    assert "provisional" not in fetch_line, (
        "n=800 flagged provisional — the marker is furniture if it is always on")


# ── the verdict is scored on the honest number ──────────────────────────────

def _cal(**kw):
    """A COMPLETE calibration block, shaped like the real collector's.

    ⚠ Hand-rolled first, and it was missing six keys the renderer reads —
    seven tests died on KeyError rather than on the behaviour under test. A
    fixture that does not match the producer tests the fixture.
    """
    base = {
        "samples_on_disk": 1868, "samples_this_epoch": 694,
        "samples_other_epochs": 1168, "samples_superseded": 6,
        "epoch": "2026-07-27.graded", "n_fitted": 694,
        "brier": 0.020541, "brier_cv": 0.020676, "brier_raw": 0.026761,
        "brier_base_rate": 0.02118, "threshold": 0.837,
        "n_negative": 18, "n_samples": 694,
        "feature_contrib": {"effort_component": 0.000665,
                            "competence_component": -0.000049},
        "w_entropy": 0.0, "w_competence": 0.5, "w_effort": 0.5,
        "entropy_observed_samples": 569, "entropy_observed_pct": 82.0,
        "entropy_distinct_values": 305, "entropy_observed_pos": 557,
        "entropy_observed_neg": 12, "entropy_min_samples_gate": 30,
        "entropy_separation_sigmas": 0.48, "effort_separation_sigmas": 3.45,
        "separation_min_sigmas": 2.5, "entropy_learnable": False,
        "outcome_neg": 18, "outcome_pos": 676,
        "outcome_verified_neg": 7, "outcome_verified_pos": 174,
        "platt_a": 1.712932, "effort_observed_samples": 493,
        "label_variance": 0.02118, "label_distinct_values": 8,
        "label_mean": 0.8506, "label_sources": {"turn": 683},
        "feature_health": {}, "live_features": [],
    }
    base.update(kw)
    return base


def test_the_headline_brier_is_the_CROSS_VALIDATED_one(monkeypatch, tmp_path):
    """⚠ The line an operator actually reads. Leading it with an in-sample
    number is where the optimism entered every downstream quote of it."""
    monkeypatch.setattr(LH, "collect_learning_health",
                        lambda md: {"calibration": _cal()})
    out = LH.render_learning_health(tmp_path)
    assert "Brier 0.020676 (CV)" in out
    assert "in-sample 0.020541" in out
    assert "in-sample is NOT performance" in out


def test_an_absent_CV_number_is_labelled_in_sample_not_passed_off(monkeypatch, tmp_path):
    """A params file written before the audit has no CV number. It must say
    'IN-SAMPLE — treat as optimistic', never quote it bare."""
    monkeypatch.setattr(LH, "collect_learning_health",
                        lambda md: {"calibration": _cal(brier_cv=-1.0)})
    out = LH.render_learning_health(tmp_path)
    assert "IN-SAMPLE" in out and "optimistic" in out


def test_the_negative_class_warning_fires_when_starved(monkeypatch, tmp_path):
    """THE NUMBER EVERY OTHER VERDICT RESTS ON."""
    monkeypatch.setattr(LH, "collect_learning_health",
                        lambda md: {"calibration": _cal()})
    out = LH.render_learning_health(tmp_path)
    assert "negative class: 18/694" in out and "UNDER-POWERED" in out


def test_the_warning_does_NOT_fire_on_a_healthy_negative_class(monkeypatch, tmp_path):
    """⚠ OVER-FIRING GUARD. A warning that is always on is furniture, and
    furniture is what the operator learns to skip."""
    monkeypatch.setattr(LH, "collect_learning_health",
                        lambda md: {"calibration": _cal(n_negative=300)})
    out = LH.render_learning_health(tmp_path)
    assert "negative class: 300/694" in out
    assert "UNDER-POWERED" not in out


def test_the_ablation_is_reported_and_outranks_the_sigma_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(LH, "collect_learning_health",
                        lambda md: {"calibration": _cal()})
    out = LH.render_learning_health(tmp_path)
    assert "feature ABLATION" in out
    assert "believe the ablation" in out


# ── the dead consumer ───────────────────────────────────────────────────────

def test_a_DEAD_consumer_is_stated_beside_the_number(monkeypatch, tmp_path):
    """⚠⚠ THE BIG ONE. A reader who sees "threshold 0.837" reasonably assumes
    something happens at 0.837. Measured on the live log: 118 below-threshold
    readings, 0 arbitrations, because `_METACOG_ARBITER_ENABLED` is False."""
    from ghost_agent.core import agent as _agent_mod
    monkeypatch.setattr(_agent_mod, "_METACOG_ARBITER_ENABLED", False,
                        raising=False)
    monkeypatch.setattr(LH, "collect_learning_health",
                        lambda md: {"calibration": _cal()})
    out = LH.render_learning_health(tmp_path)
    assert "CONSUMER DEAD" in out
    assert "_METACOG_ARBITER_ENABLED" in out, "the operator must be told WHERE"


def test_the_dead_notice_DISAPPEARS_when_the_consumer_is_enabled(monkeypatch, tmp_path):
    """⚠ OVER-FIRING GUARD, and the one that keeps this honest: if the notice
    is unconditional it is not evidence about anything."""
    from ghost_agent.core import agent as _agent_mod
    monkeypatch.setattr(_agent_mod, "_METACOG_ARBITER_ENABLED", True,
                        raising=False)
    monkeypatch.setattr(LH, "collect_learning_health",
                        lambda md: {"calibration": _cal()})
    out = LH.render_learning_health(tmp_path)
    assert "CONSUMER DEAD" not in out


def test_the_arbiter_toggle_still_exists_where_the_notice_says_it_does():
    """The notice names a symbol. If it is renamed, the notice becomes a lie
    and this fails rather than the operator chasing a ghost."""
    from ghost_agent.core import agent as _agent_mod
    assert hasattr(_agent_mod, "_METACOG_ARBITER_ENABLED")


# ── params round-trip ───────────────────────────────────────────────────────

def test_new_fields_survive_a_save_load_round_trip(tmp_path):
    t = CalibrationTracker(tmp_path)
    p = FittedParams(w_entropy=0.0, w_competence=0.5, threshold=0.8,
                     lambda_uncertainty=0.0, brier=0.02, n_samples=100,
                     fitted_at="x", brier_cv=0.021, n_negative=7,
                     feature_contrib={"effort_component": 0.001})
    t._save_params(p)
    got = t.load_params()
    assert got.brier_cv == 0.021 and got.n_negative == 7
    assert got.feature_contrib == {"effort_component": 0.001}


def test_a_PRE_AUDIT_params_file_still_loads(tmp_path):
    """Legacy files carry none of these fields. Defaults must read as 'not
    recorded' (-1.0 / 0 / None), never as a measured zero."""
    from ghost_agent.core.calibration import SCHEMA_VERSION
    (tmp_path / "calibration_params.json").write_text(json.dumps({
        "schema": SCHEMA_VERSION, "w_entropy": 0.0, "w_competence": 0.5,
        "threshold": 0.8, "brier": 0.02, "n_samples": 100}))
    got = CalibrationTracker(tmp_path).load_params()
    assert got is not None
    assert got.brier_cv == -1.0 and got.n_negative == 0
    assert got.feature_contrib is None
