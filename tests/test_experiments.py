"""Live randomized arms (core/experiments.py).

Covers the four things that make a randomizer trustworthy: the assignment is
deterministic and fair, a bad config cannot take the hot path down, the
outcome analysis is anytime-valid, and the framework can be killed.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import experiments as ex


@pytest.fixture(autouse=True)
def _clear_cache():
    ex.reset_registry_cache()
    yield
    ex.reset_registry_cache()


# ── assignment ────────────────────────────────────────────────────────

def test_assignment_is_deterministic():
    reg = ex.ExperimentRegistry(ex.DEFAULT_SPECS)
    first = reg.assign("risk_steer", "req-abc")
    assert first in (ex.CONTROL, ex.TREATMENT)
    for _ in range(20):
        assert reg.assign("risk_steer", "req-abc") == first


def test_assignment_is_fair_within_tolerance():
    """A retry landing on a different arm would corrupt the comparison, and a
    skewed split would silently reweight it — so both are pinned."""
    reg = ex.ExperimentRegistry(ex.DEFAULT_SPECS)
    arms = [reg.assign("risk_steer", f"req-{i}") for i in range(4000)]
    treat = arms.count(ex.TREATMENT)
    assert 0.47 < treat / len(arms) < 0.53


def test_salt_reshuffles_assignments():
    a = ex.ExperimentRegistry(ex.DEFAULT_SPECS, salt="epoch-1")
    b = ex.ExperimentRegistry(ex.DEFAULT_SPECS, salt="epoch-2")
    differing = sum(1 for i in range(500)
                    if a.assign("risk_steer", str(i)) != b.assign("risk_steer", str(i)))
    assert differing > 150  # ~50% expected; anything near 0 means salt is inert


def test_traffic_gate_enrolls_a_fraction_and_is_uncorrelated_with_arm():
    spec = ex.ExperimentSpec(name="partial", traffic=0.25)
    reg = ex.ExperimentRegistry([spec])
    assigned = [reg.assign("partial", f"u{i}") for i in range(4000)]
    enrolled = [a for a in assigned if a]
    assert 0.21 < len(enrolled) / 4000 < 0.29
    # The enrollment draw must not bias the arm draw (separate hash prefixes).
    treat = enrolled.count(ex.TREATMENT)
    assert 0.42 < treat / len(enrolled) < 0.58


def test_weighted_arms_respect_weights():
    spec = ex.ExperimentSpec(name="weighted", arms=("a", "b"), weights=(3.0, 1.0))
    reg = ex.ExperimentRegistry([spec])
    picks = [reg.assign("weighted", f"u{i}") for i in range(4000)]
    assert 0.71 < picks.count("a") / len(picks) < 0.79


def test_disabled_and_zero_traffic_yield_no_arm():
    reg = ex.ExperimentRegistry([
        ex.ExperimentSpec(name="off", enabled=False),
        ex.ExperimentSpec(name="zero", traffic=0.0),
    ])
    assert reg.assign("off", "u1") == ""
    assert reg.assign("zero", "u1") == ""
    assert reg.assign("nonexistent", "u1") == ""
    assert reg.assign("off", "") == ""


# ── registry loading ──────────────────────────────────────────────────

def test_load_registry_reads_file_and_caches_on_mtime(tmp_path):
    p = tmp_path / "experiments.json"
    p.write_text(json.dumps({"salt": "s1", "experiments": [
        {"name": "from_file", "arms": ["control", "treatment"]}]}))
    reg = ex.load_registry(p)
    assert "from_file" in reg.specs and reg.salt == "s1"
    assert ex.load_registry(p) is reg  # same mtime → cached object


def test_malformed_file_falls_back_to_defaults(tmp_path):
    p = tmp_path / "experiments.json"
    p.write_text("{not json")
    reg = ex.load_registry(p)
    assert set(reg.specs) == {s.name for s in ex.DEFAULT_SPECS}


def test_missing_file_yields_defaults(tmp_path):
    reg = ex.load_registry(tmp_path / "nope.json")
    assert set(reg.specs) == {s.name for s in ex.DEFAULT_SPECS}


def test_bad_specs_are_skipped_not_fatal(tmp_path):
    p = tmp_path / "experiments.json"
    p.write_text(json.dumps({"experiments": [
        {"name": "Bad Name!"},                       # charset
        {"name": "dupe_arms", "arms": ["a", "a"]},   # duplicate arms
        {"name": "one_arm", "arms": ["a"]},          # < 2 arms
        {"name": "good_one"},
    ]}))
    reg = ex.load_registry(p)
    assert set(reg.specs) == {"good_one"}


def test_redactor_colliding_name_is_rejected(tmp_path):
    """An experiment named like a secret would have its ARM stored as
    '<REDACTED>' in every trajectory — silently unanalysable."""
    p = tmp_path / "experiments.json"
    p.write_text(json.dumps({"experiments": [
        {"name": "auth_token"}, {"name": "safe_name"}]}))
    reg = ex.load_registry(p)
    assert set(reg.specs) == {"safe_name"}


# ── runtime enrollment ────────────────────────────────────────────────

def _ctx(tmp_path):
    return SimpleNamespace(memory_dir=tmp_path / "memory")


def test_enroll_and_read_back(tmp_path, monkeypatch):
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ctx = _ctx(tmp_path)
    arms = ex.enroll_request(ctx, "req-1")
    assert set(arms) == {"risk_steer"}
    assert ex.arm_for(ctx, "risk_steer", "req-1") == arms["risk_steer"]
    assert ex.assignments_for_request(ctx, "req-1") == arms


def test_stale_stash_reads_as_unenrolled(tmp_path):
    ctx = _ctx(tmp_path)
    ex.enroll_request(ctx, "req-1")
    assert ex.arm_for(ctx, "risk_steer", "req-OTHER") == ""
    assert ex.assignments_for_request(ctx, "req-OTHER") == {}


def test_kill_switch_disables_assignment(tmp_path, monkeypatch):
    monkeypatch.setenv(ex.ENV_KILL, "0")
    ctx = _ctx(tmp_path)
    assert ex.enroll_request(ctx, "req-1") == {}
    assert ex.arm_for(ctx, "risk_steer", "req-1") == ""


def test_internal_requests_are_excluded(tmp_path, monkeypatch):
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ctx = _ctx(tmp_path)
    assert ex.enroll_request(ctx, "sub-chess-1234") == {}


def test_enroll_never_raises_on_hostile_context():
    """A context that cannot hold the stash is UNENROLLED, not half-enrolled:
    every consumer reads the arm back by req_id, so an assignment that cannot
    be stashed is one nothing can act on and nothing will record."""
    class Hostile:
        @property
        def memory_dir(self):
            raise RuntimeError("boom")

        def __setattr__(self, k, v):
            raise RuntimeError("boom")

    assert ex.enroll_request(Hostile(), "req-1") == {}
    assert ex.arm_for(Hostile(), "risk_steer") == ""


# ── analysis ──────────────────────────────────────────────────────────

def _traj(arm, outcome="passed", steps=3, dur=1.0, extra_key=ex.EXTRA_KEY):
    return SimpleNamespace(
        extra={extra_key: {"risk_steer": arm}} if arm else {},
        outcome=outcome, n_steps=steps, duration_s=dur,
    )


def test_summarize_counts_arms_and_excludes_unstamped():
    trajs = [_traj("control"), _traj("treatment", outcome="failed"),
             _traj(None), _traj("control", outcome="unknown")]
    summary = ex.summarize_trajectories(trajs)
    assert summary["risk_steer"]["control"].n == 2
    assert summary["risk_steer"]["control"].unknown == 1
    assert summary["risk_steer"]["treatment"].n == 1
    # UNKNOWN contributes to n but NOT to the failure-rate metric.
    assert summary["risk_steer"]["control"].count("failure_rate") == 1


def test_unknown_outcomes_are_excluded_from_failure_rate():
    summary = ex.summarize_trajectories(
        [_traj("control", outcome="unknown") for _ in range(5)])
    assert summary["risk_steer"]["control"].mean("failure_rate") is None


def test_confidence_sequence_actually_covers():
    """The name of the old test claimed coverage and never measured it, which
    is why a 52%-miscoverage variance bug sat behind a green suite. Seeded and
    small, but it is a real coverage check: monitored from the same n the
    verdict gate uses, the true mean must stay inside the running interval on
    the large majority of streams."""
    import random
    rng = random.Random(4242)
    violations = 0
    trials = 120
    for _ in range(trials):
        vals = []
        for _ in range(120):
            vals.append(1.0 if rng.random() < 0.3 else 0.0)
            if len(vals) < ex._MIN_VERDICT_N:
                continue
            r = ex.asymp_cs_radius(vals, alpha=0.05)
            m = sum(vals) / len(vals)
            if not (m - r <= 0.3 <= m + r):
                violations += 1
                break
    # Nominal 5%; measured ~3% at N=200 over 3000 streams. A regression to the
    # unregularised variance takes this to ~58%.
    assert violations / trials < 0.20, f"{violations}/{trials} streams miscovered"


def test_confidence_sequence_shrinks_with_n():
    vals_small = [0.0, 1.0] * 10
    vals_big = [0.0, 1.0] * 500
    r_small = ex.asymp_cs_radius(vals_small)
    r_big = ex.asymp_cs_radius(vals_big)
    assert r_small is not None and r_big is not None
    assert r_big < r_small
    # Anytime-valid intervals are WIDER than the fixed-n normal interval
    # (1.96·σ/√n) — that width is the price of peeking.
    import math
    naive = 1.96 * 0.5 / math.sqrt(len(vals_big))
    assert r_big > naive


def test_confidence_sequence_needs_two_points():
    assert ex.asymp_cs_radius([]) is None
    assert ex.asymp_cs_radius([1.0]) is None
    assert ex.asymp_cs_radius([1.0, 0.0], alpha=0.0) is None


def test_no_difference_verdict_when_arms_match():
    trajs = ([_traj("control", outcome="failed") for _ in range(60)]
             + [_traj("control") for _ in range(60)]
             + [_traj("treatment", outcome="failed") for _ in range(60)]
             + [_traj("treatment") for _ in range(60)])
    summary = ex.summarize_trajectories(trajs)
    cmps = {c.metric: c for c in ex.compare_arms(summary["risk_steer"])}
    assert cmps["failure_rate"].verdict == "no difference detected yet"


def test_large_clean_effect_is_detected_with_the_right_sign():
    # Control fails half the time, treatment never — a real, large effect.
    trajs = ([_traj("control", outcome="failed") for _ in range(200)]
             + [_traj("control") for _ in range(200)]
             + [_traj("treatment") for _ in range(400)])
    summary = ex.summarize_trajectories(trajs)
    cmps = {c.metric: c for c in ex.compare_arms(summary["risk_steer"])}
    assert cmps["failure_rate"].verdict == "TREATMENT BETTER"
    assert cmps["failure_rate"].diff < 0


def test_worse_treatment_is_reported_as_worse():
    trajs = ([_traj("control") for _ in range(400)]
             + [_traj("treatment", outcome="failed") for _ in range(400)])
    summary = ex.summarize_trajectories(trajs)
    cmps = {c.metric: c for c in ex.compare_arms(summary["risk_steer"])}
    assert cmps["failure_rate"].verdict == "TREATMENT WORSE"


def test_report_flags_enrollment_skew():
    trajs = ([_traj("control") for _ in range(90)]
             + [_traj("treatment") for _ in range(10)])
    out = ex.render_report(ex.summarize_trajectories(trajs))
    assert "enrollment skew" in out


def test_report_flags_redacted_arm():
    trajs = [_traj("<REDACTED>") for _ in range(3)]
    out = ex.render_report(ex.summarize_trajectories(trajs))
    assert "<REDACTED>" in out and "redactor" in out


def test_report_handles_empty_corpus():
    assert "No experiment-stamped trajectories" in ex.render_report({})


def test_summarize_survives_malformed_records():
    class Exploding:
        @property
        def extra(self):
            raise RuntimeError("boom")

    summary = ex.summarize_trajectories([Exploding(), _traj("control")])
    assert summary["risk_steer"]["control"].n == 1


# ── guards against wrong verdicts (review findings, 2026-08-05) ───────

def test_constant_arms_never_produce_a_zero_width_interval():
    """The plain sample SD is 0 on constant input, which collapsed the CS to a
    point and "proved" a difference from six Bernoulli observations. At
    p_fail=0.5 that shape occurs 12.2% of the time at 2 turns/arm."""
    r = ex.asymp_cs_radius([1.0] * 50)
    assert r is not None and r > 0.0
    r0 = ex.asymp_cs_radius([0.0] * 50)
    assert r0 is not None and r0 > 0.0


def test_no_verdict_below_the_minimum_sample():
    trajs = ([_traj("control") for _ in range(3)]
             + [_traj("treatment", outcome="failed") for _ in range(3)])
    cmps = {c.metric: c for c in ex.compare_arms(
        ex.summarize_trajectories(trajs)["risk_steer"])}
    fr = cmps["failure_rate"]
    assert fr.diff == 1.0                      # the numbers are still shown
    assert "insufficient data" in fr.verdict   # the conclusion is withheld
    assert fr.diff_lo < 0.0 < fr.diff_hi       # ...and the interval is honest


def test_alpha_is_split_across_metrics():
    """Three intervals get three chances to cross; without the Bonferroni
    split, "stop when ANY excludes zero" runs at ~15% against a nominal 5%."""
    vals = [float(i % 7) for i in range(200)]
    wide = ex.asymp_cs_radius(vals, alpha=0.05 / 3 / 2)
    narrow = ex.asymp_cs_radius(vals, alpha=0.05)
    assert wide > narrow


def test_differential_attrition_is_flagged_not_celebrated():
    """A treatment that makes turns end unverifiable shifts WHICH turns are
    scored. Simulated with zero true effect this manufactured a 14-point
    'improvement' — so failure_rate must carry the confound."""
    trajs = ([_traj("control", outcome="failed") for _ in range(40)]
             + [_traj("control") for _ in range(60)]
             # treatment: the would-be failures resolve UNKNOWN instead
             + [_traj("treatment", outcome="unknown") for _ in range(40)]
             + [_traj("treatment") for _ in range(60)])
    cmps = {c.metric: c for c in ex.compare_arms(
        ex.summarize_trajectories(trajs)["risk_steer"])}
    fr = cmps["failure_rate"]
    assert "differential attrition" in fr.confound
    assert "⚠" in fr.verdict


def test_mechanism_metrics_carry_their_own_caveat():
    cmps = {c.metric: c for c in ex.compare_arms(
        ex.summarize_trajectories(
            [_traj("control", steps=8) for _ in range(40)]
            + [_traj("treatment", steps=2) for _ in range(40)])["risk_steer"])}
    assert "mechanism, not outcome" in cmps["n_steps"].confound
    assert "mechanism, not outcome" in cmps["duration_s"].confound
    assert cmps["failure_rate"].confound == ""


def test_balance_alarm_does_not_fire_on_ordinary_chance():
    """The old fixed ±20% rule falsely accused the stamp 11.9% of the time at
    its own minimum n. 30/20 is a perfectly ordinary coin-flip outcome."""
    trajs = ([_traj("control") for _ in range(30)]
             + [_traj("treatment") for _ in range(20)])
    assert "enrollment skew" not in ex.render_report(
        ex.summarize_trajectories(trajs))


def test_balance_alarm_still_fires_on_a_real_wiring_failure():
    trajs = ([_traj("control") for _ in range(95)]
             + [_traj("treatment") for _ in range(5)])
    assert "enrollment skew" in ex.render_report(ex.summarize_trajectories(trajs))


def test_binomial_two_sided_p_is_exact():
    assert ex._binomial_two_sided_p(25, 50) == pytest.approx(1.0)
    assert ex._binomial_two_sided_p(0, 10) == pytest.approx(2 / 1024)
    assert ex._binomial_two_sided_p(0, 0) == 1.0


# ── triggered subgroup (the powered comparison) ───────────────────────

def _traj_trig(arm, fired, outcome="passed"):
    return SimpleNamespace(
        extra={ex.EXTRA_KEY: {"risk_steer": arm}, "risk_steer_fired": fired},
        outcome=outcome, n_steps=8, duration_s=5.0)


def test_triggered_only_selects_the_turns_the_trigger_touched():
    trajs = [_traj("control"), _traj("treatment"),
             _traj_trig("control", False), _traj_trig("treatment", True)]
    all_turns = ex.summarize_trajectories(trajs)
    triggered = ex.summarize_trajectories(trajs, triggered_only=True)
    assert all_turns["risk_steer"]["control"].n == 2
    assert triggered["risk_steer"]["control"].n == 1
    assert triggered["risk_steer"]["treatment"].n == 1


def test_trigger_fired_reads_presence_not_truth():
    """Presence = the gate tripped (in EITHER arm); the value = whether the
    treatment ran. Conditioning on presence is legitimate because the gate is
    evaluated identically in both arms."""
    assert ex.trigger_fired(_traj_trig("control", False), "risk_steer")
    assert ex.trigger_fired(_traj_trig("treatment", True), "risk_steer")
    assert not ex.trigger_fired(_traj("control"), "risk_steer")
    assert not ex.trigger_fired(_traj_trig("control", False), "other_experiment")


def test_report_points_the_reader_at_the_powered_block():
    trajs = ([_traj("control") for _ in range(50)]
             + [_traj("treatment") for _ in range(50)]
             + [_traj_trig("control", False) for _ in range(5)]
             + [_traj_trig("treatment", True) for _ in range(5)])
    out = ex.render_report(ex.summarize_trajectories(trajs),
                           triggered=ex.summarize_trajectories(
                               trajs, triggered_only=True))
    assert "READ THIS BLOCK" in out
    assert "triggered turns only" in out
    assert "trigger fired on 10/110" in out


def test_report_says_so_when_the_trigger_never_fired():
    out = ex.render_report(ex.summarize_trajectories(
        [_traj("control"), _traj("treatment")]))
    assert "trigger has not fired" in out


# ── late-drain assignment recovery ────────────────────────────────────

def test_a_later_request_does_not_erase_an_earlier_assignment(tmp_path, monkeypatch):
    """A streamed turn's trajectory is written AFTER the next request has
    enrolled. With a single-slot stash that turn lost its arm entirely —
    silently deleting a data point and skewing the balance check."""
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ctx = _ctx(tmp_path)
    first = ex.enroll_request(ctx, "req-streamed")
    ex.enroll_request(ctx, "req-next")          # the cron job that follows
    assert ex.assignments_for_request(ctx, "req-streamed") == first


def test_the_recent_ring_is_bounded(tmp_path, monkeypatch):
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ctx = _ctx(tmp_path)
    for i in range(ex._RECENT_ARMS_MAX + 10):
        ex.enroll_request(ctx, f"req-{i}")
    assert len(ctx._experiment_arms_recent) <= ex._RECENT_ARMS_MAX
    assert ex.assignments_for_request(ctx, "req-0") == {}   # evicted, not leaked


def test_ineligible_turns_are_not_enrolled(tmp_path, monkeypatch):
    """Self-play/dream solver turns write no trajectory, so an arm assigned
    there can never be recorded or analysed — and a coin-flip steer would
    randomize the LESSON KEEP/KILL verdict, not just a reply."""
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ctx = _ctx(tmp_path)
    assert ex.enroll_request(ctx, "req-sim", eligible=False) == {}
    assert ex.arm_for(ctx, "risk_steer", "req-sim") == ""


# ── defects the mutation review proved were untested ──────────────────

def test_binomial_p_survives_a_large_corpus():
    """`2**n` overflows a float64 past n=1023; the OverflowError was swallowed
    into p=1.0, silently disabling the skew alarm at exactly the corpus size
    where it becomes trustworthy."""
    # An ordinary split at n>1023 must NOT read as p=1.0-by-overflow, and an
    # extreme one must still be tiny.
    assert 0.3 < ex._binomial_two_sided_p(500, 1024) < 0.6
    assert ex._binomial_two_sided_p(103, 1030) < 1e-100
    assert ex._binomial_two_sided_p(4000, 10000) < 1e-50
    assert ex._binomial_two_sided_p(5000, 10000) == pytest.approx(1.0)


def test_skew_alarm_still_fires_above_the_old_overflow_point():
    trajs = ([_traj("control") for _ in range(927)]
             + [_traj("treatment") for _ in range(103)])
    assert "enrollment skew" in ex.render_report(ex.summarize_trajectories(trajs))


def test_metric_alpha_split_is_actually_applied():
    """Removing the Bonferroni split changes every radius and no test noticed
    (mutation C). Pin it by comparing compare_arms' interval against one built
    at the un-split alpha."""
    trajs = ([_traj("control", steps=4) for _ in range(200)]
             + [_traj("treatment", steps=4) for _ in range(200)])
    cmp_ = next(c for c in ex.compare_arms(
        ex.summarize_trajectories(trajs)["risk_steer"]) if c.metric == "n_steps")
    vals = [4.0] * 200
    unsplit = ex.asymp_cs_radius(vals, alpha=0.05 / 2)      # arms only
    split = ex.asymp_cs_radius(vals, alpha=0.05 / 3 / 2)    # arms AND metrics
    assert split > unsplit
    # The reported half-width is the sum of the two per-arm radii, and it must
    # be built from the SPLIT alpha.
    assert (cmp_.diff_hi - cmp_.diff_lo) / 2 == pytest.approx(split * 2, rel=1e-6)


def test_report_shows_per_metric_n():
    """A mean over 5 of 200 turns rendered identically to one over 200
    (mutation E) — the sample size has to be on the line."""
    trajs = ([_traj("control", dur=0.0) for _ in range(40)]
             + [_traj("treatment", dur=2.0) for _ in range(40)])
    out = ex.render_report(ex.summarize_trajectories(trajs))
    # duration_s is omitted for the control arm (0.0 is not a measurement),
    # so its per-metric n must show 0/40 rather than silently averaging 40.
    assert "duration_s     n=0/40" in out
    assert "n_steps        n=40/40" in out


def test_registry_cache_invalidates_on_edit(tmp_path):
    """Editing experiments.json is documented as the way to turn an experiment
    off without a deploy; caching was pinned, invalidation was not."""
    import os
    import time
    p = tmp_path / "experiments.json"
    p.write_text(json.dumps({"experiments": [{"name": "first"}]}))
    assert set(ex.load_registry(p).specs) == {"first"}
    time.sleep(0.01)
    p.write_text(json.dumps({"experiments": [{"name": "second"}]}))
    os.utime(p, None)
    assert set(ex.load_registry(p).specs) == {"second"}


def test_malformed_arm_names_are_rejected(tmp_path):
    """Arm names ride into `extra` on every stamped trajectory, and the
    registry is a file this agent can write."""
    p = tmp_path / "experiments.json"
    p.write_text(json.dumps({"experiments": [
        {"name": "huge_arm", "arms": ["control", "x" * 500]},
        {"name": "many_arms", "arms": [f"a{i}" for i in range(20)]},
        {"name": "ok_one", "arms": ["control", "treatment"]},
    ]}))
    assert set(ex.load_registry(p).specs) == {"ok_one"}


# ── request-scoped compliance (the ring made this race WORSE) ─────────

def test_compliance_flag_does_not_leak_to_another_request(tmp_path, monkeypatch):
    """Turn A (control, never steered) must not be stamped as steered because
    turn B steered while A's streamed drain was still pending — that both
    poisons the triggered-only block and drops a clean GEPA fixture."""
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ctx = _ctx(tmp_path)
    ex.enroll_request(ctx, "req-A")
    ex.enroll_request(ctx, "req-B")
    ex.mark_trigger(ctx, "req-B", "risk_steer_fired", True)
    assert ex.trigger_flags(ctx, "req-A") == {}
    assert ex.trigger_flags(ctx, "req-B") == {"risk_steer_fired": True}


def test_compliance_flag_is_not_lost_by_a_later_request(tmp_path, monkeypatch):
    """The mirror failure: A steered, B reset the flags, A's late drain
    stamped no compliance bit — defeating the GEPA isolation on the common
    (streamed) path."""
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ctx = _ctx(tmp_path)
    ex.enroll_request(ctx, "req-A")
    ex.mark_trigger(ctx, "req-A", "risk_steer_fired", True)
    ex.enroll_request(ctx, "req-B")
    assert ex.trigger_flags(ctx, "req-A") == {"risk_steer_fired": True}


def test_streaming_summary_matches_the_two_pass_form():
    trajs = [_traj("control"), _traj("treatment", outcome="failed"),
             _traj_trig("treatment", True), _traj_trig("control", False)]
    all_s, trig_s, coverage = ex.summarize_streaming(trajs)
    assert all_s == ex.summarize_trajectories(trajs)
    assert trig_s == ex.summarize_trajectories(trajs, triggered_only=True)
    assert coverage["stamped"] == 4


def test_zero_stamp_coverage_is_reported_as_a_warning():
    """An empty report that reassures is the "built and silently never ran"
    failure applied to the instrument itself."""
    out = ex.render_report({}, coverage={"user_turns": 412, "stamped": 0})
    assert "⚠" in out and "412 user turn" in out
    quiet = ex.render_report({}, coverage={"user_turns": 0, "stamped": 0})
    assert "⚠" not in quiet


# ── verdict announcement (the "you have an answer" hook) ──────────────

def test_only_decided_verdicts_announce():
    """"no difference detected yet" and "insufficient data" are the NORMAL
    state — announcing them would be pure noise, which is exactly what the
    operator's chat-noise preference forbids."""
    undecided = ex.summarize_trajectories(
        [_traj("control") for _ in range(60)]
        + [_traj("treatment") for _ in range(60)])
    assert ex.pending_announcements(undecided, {}) == []

    decided = ex.summarize_trajectories(
        [_traj("control", outcome="failed") for _ in range(200)]
        + [_traj("treatment") for _ in range(200)])
    fresh = ex.pending_announcements(decided, {})
    assert fresh and "TREATMENT BETTER" in fresh[0][1]


def test_a_verdict_announces_once(tmp_path, monkeypatch):
    decided = ex.summarize_trajectories(
        [_traj("control", outcome="failed") for _ in range(200)]
        + [_traj("treatment") for _ in range(200)])
    first = ex.pending_announcements(decided, {})
    keys = {k for k, _ in first}
    assert ex.pending_announcements(decided, {}, already=keys) == []


def test_triggered_scope_is_labelled_distinctly():
    trig = ex.summarize_trajectories(
        [_traj_trig("control", False, outcome="failed") for _ in range(200)]
        + [_traj_trig("treatment", True) for _ in range(200)])
    lines = [line for _, line in ex.pending_announcements({}, trig)]
    assert lines and "on the turns its trigger fired" in lines[0]


def test_announce_writes_to_the_ledger_and_persists_the_marker(tmp_path):
    from types import SimpleNamespace
    from ghost_agent.core.autonomous_activity import ActivityLog
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.distill.schema import Trajectory

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    collector = TrajectoryCollector(root=tmp_path / "trajectories",
                                    session_id="s")
    for i in range(200):
        collector.append(Trajectory(
            user_request="q", final_response="a", task_kind="user_request",
            outcome="failed", n_steps=3,
            extra={ex.EXTRA_KEY: {"risk_steer": "control"}}))
        collector.append(Trajectory(
            user_request="q", final_response="a", task_kind="user_request",
            outcome="passed", n_steps=3,
            extra={ex.EXTRA_KEY: {"risk_steer": "treatment"}}))

    pushed = []
    log = ActivityLog(tmp_path / "activity.jsonl",
                      on_notify=lambda rec: pushed.append(rec.summary))
    ctx = SimpleNamespace(memory_dir=memory_dir, activity_log=log)

    first = ex.announce_new_verdicts(ctx)
    assert first and any("TREATMENT BETTER" in line for line in first)
    assert pushed, "a decided verdict must reach the notify channel"
    # Idempotent: the marker persisted, so a second tick says nothing.
    assert ex.announce_new_verdicts(ctx) == []
    assert (memory_dir.parent / "experiments_announced.json").exists()


def test_announce_never_raises_without_a_corpus(tmp_path):
    from types import SimpleNamespace
    assert ex.announce_new_verdicts(SimpleNamespace(memory_dir=tmp_path / "nope")) == []
    assert ex.announce_new_verdicts(SimpleNamespace()) == []


# ── §4I Phase 3 — CUPED variance reduction ────────────────────────────

def _traj_cov(arm, outcome, cov):
    return SimpleNamespace(
        extra={ex.EXTRA_KEY: {"risk_steer": arm}, "router_confidence": cov},
        outcome=outcome, n_steps=3, duration_s=1.0)


def test_cuped_theta_is_zero_for_an_unrelated_covariate():
    """A covariate that predicts nothing must be a NO-OP, not a risk — this is
    what makes it safe to run before Phase 2 proves the signal."""
    pairs = [(1.0, 0.5), (0.0, 0.5), (1.0, 0.5), (0.0, 0.5)]
    theta, xbar = ex.cuped_theta(pairs)
    assert theta == 0.0              # zero variance in X
    assert xbar == pytest.approx(0.5)


def test_cuped_theta_recovers_a_known_slope():
    pairs = [(2.0 * x, x) for x in (0.1, 0.2, 0.3, 0.4, 0.5)]
    theta, _ = ex.cuped_theta(pairs)
    assert theta == pytest.approx(2.0)


def test_cuped_adjust_passes_through_rows_without_a_covariate():
    out = ex.cuped_adjust([1.0, 2.0], [None, 0.5], theta=2.0, cov_mean=0.0)
    assert out == [1.0, 1.0]


def test_cuped_narrows_the_interval_when_the_covariate_predicts():
    """The measurable claim: a predictive pre-treatment covariate shrinks the
    interval. Reported, never assumed."""
    import random
    rng = random.Random(99)
    trajs = []
    for arm in (ex.CONTROL, ex.TREATMENT):
        for _ in range(150):
            cov = rng.random()
            # Failure strongly driven by the covariate; no arm effect at all.
            outcome = "failed" if rng.random() < cov else "passed"
            trajs.append(_traj_cov(arm, outcome, cov))
    cmp_ = next(c for c in ex.compare_arms(
        ex.summarize_trajectories(trajs)["risk_steer"])
        if c.metric == "failure_rate")
    assert cmp_.variance_reduction > 0.05, cmp_.variance_reduction
    # The CENTRE MOVES WITH THE WIDTH. The previous version of this test
    # asserted the means were untouched, which locked in a defect worth
    # 12/300 false verdicts: the adjusted series' re-centring IS the variance
    # reduction, so keeping the raw mean left the interval too narrow around
    # an estimator whose variability had not changed.
    assert cmp_.control_mean is not None and cmp_.treatment_mean is not None
    raw_c = sum(ex.summarize_trajectories(trajs)["risk_steer"]
                [ex.CONTROL].values["failure_rate"]) / 150
    assert abs(cmp_.control_mean - raw_c) > 1e-9, (
        "means must be the ADJUSTED ones once CUPED is applied")


def test_cuped_does_not_fire_without_enough_covariate_coverage():
    """The COVERAGE gate specifically — the previous version of this test was
    blocked by the sample-size arm instead, so removing the coverage check
    survived mutation."""
    import random
    rng = random.Random(3)
    trajs = []
    for arm in (ex.CONTROL, ex.TREATMENT):
        for i in range(60):          # well past the per-arm minimum
            cov = rng.random()
            outcome = "failed" if rng.random() < cov else "passed"
            # Only half the rows carry a covariate → 50% coverage, under 0.8.
            trajs.append(_traj_cov(arm, outcome, cov) if i % 2 == 0
                         else _traj(arm, outcome))
    for c in ex.compare_arms(ex.summarize_trajectories(trajs)["risk_steer"]):
        assert c.variance_reduction == 0.0


def test_cuped_needs_the_minimum_sample_in_EACH_arm():
    """A pooled count let CUPED engage at 15/arm and print a variance
    reduction on a row whose verdict was 'insufficient data'."""
    import random
    rng = random.Random(4)
    trajs = []
    for arm, k in ((ex.CONTROL, 45), (ex.TREATMENT, 15)):
        for _ in range(k):
            cov = rng.random()
            trajs.append(_traj_cov(
                arm, "failed" if rng.random() < cov else "passed", cov))
    for c in ex.compare_arms(ex.summarize_trajectories(trajs)["risk_steer"]):
        assert c.variance_reduction == 0.0


def test_cuped_reports_zero_gain_rather_than_a_negative_one():
    """`variance_reduction` is a REPORTED number, so it must never go
    negative even when the adjusted interval is (marginally) wider — the
    adjustment is adopted unconditionally, because min(raw, adjusted) is the
    better-of-two-draws effect and measured consistently anti-conservative."""
    import random
    rng = random.Random(7)
    trajs = []
    for arm in (ex.CONTROL, ex.TREATMENT):
        for _ in range(120):
            trajs.append(_traj_cov(arm, "failed" if rng.random() < 0.4 else "passed",
                                   rng.random()))
    summary = ex.summarize_trajectories(trajs)["risk_steer"]
    for c in ex.compare_arms(summary):
        assert c.variance_reduction >= 0.0


def test_zero_true_effect_with_a_strong_covariate_yields_no_verdict():
    """The regression that matters: keeping the raw mean while adopting the
    adjusted width produced 12/300 false 'TREATMENT BETTER/WORSE' verdicts at
    zero true effect. Re-centring takes it to 0/300."""
    import random
    for rep in range(12):
        rng = random.Random(1000 + rep)
        trajs = []
        for arm in (ex.CONTROL, ex.TREATMENT):
            for _ in range(150):
                cov = rng.random()
                trajs.append(_traj_cov(
                    arm, "failed" if rng.random() < cov else "passed", cov))
        c = next(x for x in ex.compare_arms(
            ex.summarize_trajectories(trajs)["risk_steer"])
            if x.metric == "failure_rate")
        assert not c.verdict.startswith("TREATMENT"), (rep, c.verdict, c.diff)


def test_nan_never_yields_a_zero_width_interval():
    assert ex.asymp_cs_radius([0.5, float("nan"), 0.5] * 20) is None
    assert ex.asymp_cs_radius([0.5, float("inf")] * 20) is None


def test_cuped_adjust_refuses_mismatched_lengths():
    assert ex.cuped_adjust([1.0, 2.0, 3.0], [0.5], theta=2.0, cov_mean=0.0) == \
        [1.0, 2.0, 3.0]


def test_a_bool_is_not_a_covariate():
    assert ex._covariate_of(SimpleNamespace(
        extra={"router_confidence": True})) is None


def test_covariate_is_read_from_the_trajectory_stamp():
    assert ex._covariate_of(_traj_cov("control", "passed", 0.42)) == 0.42
    assert ex._covariate_of(_traj("control")) is None
    assert ex._covariate_of(SimpleNamespace(extra={"router_confidence": "nope"})) is None
    assert ex._covariate_of(SimpleNamespace(extra=None)) is None


# ── review round 5: fixes-of-fixes ────────────────────────────────────

def test_mark_trigger_cannot_evict_a_live_turn(tmp_path, monkeypatch):
    """`_stash_arms` was changed to ring only ENROLLED requests, but
    `mark_trigger` still created an entry for ANY req_id — and sub-agents run
    the same turn loop on a shallow-copied (shared-ring) context, so 40
    sub-turns could evict a user turn whose streamed drain had not yet written
    its trajectory. It would then be recorded with no arm."""
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ctx = _ctx(tmp_path)
    ex.enroll_request(ctx, "user-1")
    for i in range(40):
        ex.mark_trigger(ctx, f"sub-{i}", "risk_steer_fired", True)
    assert ex.assignments_for_request(ctx, "user-1") != {}
    # ...and an unenrolled id never becomes a phantom ring entry.
    assert ex.trigger_flags(ctx, "sub-0") == {}


def test_marker_read_rejects_a_non_list():
    """A marker containing a bare JSON string iterated into single CHARACTERS
    and wrote them back as junk keys."""
    import json as _json
    from types import SimpleNamespace
    from ghost_agent.core.autonomous_activity import ActivityLog

    # Exercised through announce_new_verdicts' loader via a corrupt file.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "memory").mkdir()
        (root / "experiments_announced.json").write_text(_json.dumps("abc"))
        ctx = SimpleNamespace(memory_dir=root / "memory",
                              activity_log=ActivityLog(root / "a.jsonl"))
        # No trajectories → nothing to announce, but the loader must not choke
        # or write junk back.
        assert ex.announce_new_verdicts(ctx) == []


def test_unwritable_marker_bounds_announcements_to_once_per_boot(tmp_path):
    """An unwritable marker (a root-owned file under a UserName launchd
    daemon — a documented failure mode here) used to mean FOUR notify-severity
    pushes every hour, forever."""
    from types import SimpleNamespace
    from ghost_agent.core.autonomous_activity import ActivityLog
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.distill.schema import Trajectory

    ex._ANNOUNCED_THIS_PROCESS.clear()
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    collector = TrajectoryCollector(root=tmp_path / "trajectories",
                                    session_id="s")
    for _ in range(200):
        collector.append(Trajectory(user_request="q", final_response="a",
                                    task_kind="user_request", outcome="failed",
                                    n_steps=3,
                                    extra={ex.EXTRA_KEY: {"risk_steer": "control"}}))
        collector.append(Trajectory(user_request="q", final_response="a",
                                    task_kind="user_request", outcome="passed",
                                    n_steps=3,
                                    extra={ex.EXTRA_KEY: {"risk_steer": "treatment"}}))
    pushed = []
    ctx = SimpleNamespace(
        memory_dir=memory_dir,
        activity_log=ActivityLog(tmp_path / "a.jsonl",
                                 on_notify=lambda r: pushed.append(r.summary)))
    # Make the marker unwritable by putting a DIRECTORY at its path.
    (tmp_path / "experiments_announced.json").mkdir()

    first = ex.announce_new_verdicts(ctx)
    assert first and pushed
    n_first = len(pushed)
    for _ in range(5):                      # five more ticks
        assert ex.announce_new_verdicts(ctx) == []
    assert len(pushed) == n_first           # not once per tick
    ex._ANNOUNCED_THIS_PROCESS.clear()


def test_confounded_metrics_are_never_pushed():
    """`n_steps`/`duration_s` move BY CONSTRUCTION for a treatment that ends
    turns earlier — the first tick measured 4 pushes, 2 of them
    'TREATMENT BETTER — ⚠ mechanism, not outcome'."""
    summary = ex.summarize_trajectories(
        [_traj("control", steps=9, dur=9.0) for _ in range(200)]
        + [_traj("treatment", steps=2, dur=2.0) for _ in range(200)])
    lines = [line for _, line in ex.pending_announcements(summary, {})]
    assert not any("n_steps" in line or "duration_s" in line for line in lines)


def test_no_verdict_is_pushed_when_nothing_can_deliver_it(tmp_path):
    """Marking a verdict announced while no ledger exists loses it for good."""
    from types import SimpleNamespace
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.distill.schema import Trajectory

    ex._ANNOUNCED_THIS_PROCESS.clear()
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    collector = TrajectoryCollector(root=tmp_path / "trajectories",
                                    session_id="s")
    for _ in range(200):
        collector.append(Trajectory(user_request="q", final_response="a",
                                    task_kind="user_request", outcome="failed",
                                    n_steps=3,
                                    extra={ex.EXTRA_KEY: {"risk_steer": "control"}}))
        collector.append(Trajectory(user_request="q", final_response="a",
                                    task_kind="user_request", outcome="passed",
                                    n_steps=3,
                                    extra={ex.EXTRA_KEY: {"risk_steer": "treatment"}}))
    ctx = SimpleNamespace(memory_dir=memory_dir, activity_log=None)
    assert ex.announce_new_verdicts(ctx) == []
    assert not (memory_dir.parent / "experiments_announced.json").exists()
    ex._ANNOUNCED_THIS_PROCESS.clear()


def test_bounded_data_scale_has_no_discontinuity():
    """One out-of-range value used to flip the scale rule and make the
    interval NARROWER than for the same data fully inside [0,1]."""
    inside = ex.asymp_cs_radius([0.3] * 40)
    edge = ex.asymp_cs_radius([-0.001] + [0.3] * 39)
    assert edge >= inside * 0.99
