"""Risk governor + offline triage (core/risk.py).

The live half must fire only on the deep-and-struggling tail, only in the
treatment arm, and only once per request. The offline half must rank the
reflection queue worst-first without ever widening it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ghost_agent.core import risk


# ── depth curve ───────────────────────────────────────────────────────

def test_depth_curve_matches_the_measured_points():
    """These five numbers are measured on this agent's own corpus (§4H).
    Drifting them silently would change a live policy on an unmeasured basis."""
    for step, expected in ((1, 0.178), (4, 0.356), (6, 0.423),
                           (8, 0.520), (12, 0.606)):
        assert risk.depth_failure_prior(step) == pytest.approx(expected)


def test_depth_curve_interpolates_and_is_monotone():
    vals = [risk.depth_failure_prior(s) for s in range(1, 20)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert 0.356 < risk.depth_failure_prior(5) < 0.423


def test_depth_curve_is_flat_outside_the_measured_range():
    """Extrapolating past step 12 would fabricate a number the corpus never
    supported — the curve is held, not extended."""
    assert risk.depth_failure_prior(50) == risk.depth_failure_prior(12)
    assert risk.depth_failure_prior(0) == risk.depth_failure_prior(1)
    assert risk.depth_failure_prior("nonsense") == risk.depth_failure_prior(1)


# ── live reading ──────────────────────────────────────────────────────

def test_shallow_clean_turn_is_low_risk_and_does_not_steer():
    r = risk.turn_risk(step=1, tool_names=["file_system"])
    assert r.band == "low" and not r.should_steer


def test_normal_working_turn_at_the_gate_depth_does_not_steer():
    """Step 6 with five clean calls is a turn doing fine — steering it is
    noise. Pinned because the threshold is the whole design."""
    r = risk.turn_risk(step=6, tool_names=["file_system", "execute",
                                           "file_system", "browser", "execute"])
    assert not r.should_steer


def test_deep_struggling_turn_steers():
    names = ["file_system"] * 4 + ["execute"] * 3 + ["file_system"] * 5
    r = risk.turn_risk(step=12, tool_names=names, execution_failures=2)
    assert r.should_steer and r.band == "high"


def test_depth_alone_below_the_gate_never_steers():
    """A turn can be maximally sprawling and still not be steered before
    step 6 — the gate is AND, not OR."""
    r = risk.turn_risk(step=3, tool_names=["execute"] * 20, execution_failures=5)
    assert not r.should_steer


def test_no_tools_yet_is_not_evidence_of_struggle():
    assert risk.turn_risk(step=8, tool_names=[]).effort_struggle == 0.0
    assert risk.turn_risk(step=8, tool_names=None).effort_struggle == 0.0


def test_turn_risk_never_raises_on_junk():
    r = risk.turn_risk(step="x", tool_names=[None, 5], execution_failures="y")
    assert 0.0 <= r.score <= 1.0


def test_steer_message_quotes_the_measured_prior_not_the_blend():
    r = risk.turn_risk(step=12, tool_names=["execute"] * 12, execution_failures=3)
    msg = risk.risk_steer_message(r)
    assert "61%" in msg              # depth_prior 0.606, not the blended score
    assert "step 12" in msg
    assert "STOP" in msg             # the option the agent never takes alone


def test_kill_switches(monkeypatch):
    monkeypatch.setenv(risk.ENV_STEER_KILL, "0")
    assert not risk.steer_enabled()
    monkeypatch.setenv(risk.ENV_STEER_KILL, "1")
    assert risk.steer_enabled()
    monkeypatch.setenv(risk.ENV_TRIAGE_KILL, "off")
    assert not risk.triage_ranking_enabled()


# ── offline triage ────────────────────────────────────────────────────

def _traj(tid, steps, names=(), errors=0, session_id=""):
    calls = [SimpleNamespace(name=n, arguments={},
                             error="boom" if i < errors else "")
             for i, n in enumerate(names)]
    return SimpleNamespace(id=tid, n_steps=steps, tool_calls=calls,
                           session_id=session_id, task_kind="user_request")


def test_shape_score_ranks_a_grind_above_a_clean_short_turn():
    bad = _traj("bad", 12, ["execute"] * 12, errors=6)
    good = _traj("good", 1, ["file_system"])
    assert risk.shape_triage_score(bad) > risk.shape_triage_score(good)


def test_calibrated_score_wins_over_the_shape_proxy():
    """A turn that LOOKS clean but scored badly must outrank one that looks
    messy — that is the entire point of using the calibrated number."""
    looks_clean = _traj("a", 1, ["file_system"], session_id="req-a")
    looks_messy = _traj("b", 10, ["execute"] * 10, session_id="req-b")
    calibrated = {"req-a": 0.95, "req-b": 0.10}
    ranked = risk.rank_for_triage([looks_messy, looks_clean],
                                  calibrated=calibrated)
    assert [t.id for t in ranked] == ["a", "b"]


def test_calibrated_index_inverts_confidence():
    samples = [SimpleNamespace(req_id="r1", composite=0.9),
               SimpleNamespace(req_id="", composite=0.1)]
    idx = risk.calibrated_risk_index(samples)
    assert idx == {"r1": pytest.approx(0.1)}


def test_ranking_is_stable_for_equal_scores():
    """Equal scores must preserve corpus order, so an all-equal corpus keeps
    exactly today's chronological behaviour."""
    a, b, c = (_traj("a", 3, ["x"]), _traj("b", 3, ["x"]), _traj("c", 3, ["x"]))
    assert [t.id for t in risk.rank_for_triage([a, b, c])] == ["a", "b", "c"]


def test_ranking_limit_and_empty_input():
    trajs = [_traj(str(i), i + 1, ["execute"] * (i + 1)) for i in range(5)]
    assert len(risk.rank_for_triage(trajs, limit=2)) == 2
    assert risk.rank_for_triage([]) == []


def test_ranking_survives_hostile_records():
    class Hostile:
        id = "h"

        @property
        def tool_calls(self):
            raise RuntimeError("boom")

    out = risk.rank_for_triage([Hostile(), _traj("ok", 2, ["x"])])
    assert len(out) == 2  # returned unranked rather than dropped


# ── reflection wiring (§4H item 2) ────────────────────────────────────

@pytest.mark.asyncio
async def test_reflector_reflects_the_worst_of_the_pool_not_the_oldest(monkeypatch):
    from ghost_agent.distill.schema import Outcome, Trajectory
    from ghost_agent.reflection.loop import Reflector

    monkeypatch.setenv(risk.ENV_TRIAGE_KILL, "1")
    # Oldest-first corpus: the grind is LAST, so chronological selection
    # would never reach it.
    corpus = [Trajectory(id=f"clean{i}", outcome=Outcome.FAILED.value,
                         n_steps=1) for i in range(5)]
    corpus.append(Trajectory(id="grind", outcome=Outcome.FAILED.value,
                             n_steps=12))

    seen = []

    async def _critique(prompt):
        return "ROOT CAUSE: x\nREVISED PLAN:\n1. y"

    refl = Reflector(critique_fn=_critique, max_failures=1)

    async def _spy(traj):
        seen.append(traj.id)
        return SimpleNamespace(ok=False, reflected_trajectory=None, error="")

    monkeypatch.setattr(refl, "_reflect_one", _spy)
    await refl.run(failed_source=corpus)
    assert seen == ["grind"]


@pytest.mark.asyncio
async def test_reflector_falls_back_to_chronological_when_disabled(monkeypatch):
    from ghost_agent.distill.schema import Outcome, Trajectory
    from ghost_agent.reflection.loop import Reflector

    monkeypatch.setenv(risk.ENV_TRIAGE_KILL, "0")
    corpus = [Trajectory(id="first", outcome=Outcome.FAILED.value, n_steps=1),
              Trajectory(id="grind", outcome=Outcome.FAILED.value, n_steps=12)]
    seen = []

    async def _critique(prompt):
        return ""

    refl = Reflector(critique_fn=_critique, max_failures=1)

    async def _spy(traj):
        seen.append(traj.id)
        return SimpleNamespace(ok=False, reflected_trajectory=None, error="")

    monkeypatch.setattr(refl, "_reflect_one", _spy)
    await refl.run(failed_source=corpus)
    assert seen == ["first"]


@pytest.mark.asyncio
async def test_ranking_never_widens_the_candidate_set(monkeypatch):
    """Only reflectable (FAILED) trajectories may be selected, ranked or not —
    the pool is bigger, the eligibility rule is not looser."""
    from ghost_agent.distill.schema import Outcome, Trajectory
    from ghost_agent.reflection.loop import Reflector

    monkeypatch.setenv(risk.ENV_TRIAGE_KILL, "1")
    corpus = [Trajectory(id="passed", outcome=Outcome.PASSED.value, n_steps=20),
              Trajectory(id="unknown", outcome=Outcome.UNKNOWN.value, n_steps=20),
              Trajectory(id="failed", outcome=Outcome.FAILED.value, n_steps=2)]
    seen = []

    async def _critique(prompt):
        return ""

    refl = Reflector(critique_fn=_critique, max_failures=3)

    async def _spy(traj):
        seen.append(traj.id)
        return SimpleNamespace(ok=False, reflected_trajectory=None, error="")

    monkeypatch.setattr(refl, "_reflect_one", _spy)
    await refl.run(failed_source=corpus)
    assert seen == ["failed"]


@pytest.mark.asyncio
async def test_reflector_uses_the_injected_calibration_source(monkeypatch):
    from ghost_agent.distill.schema import Outcome, Trajectory
    from ghost_agent.reflection.loop import Reflector

    monkeypatch.setenv(risk.ENV_TRIAGE_KILL, "1")
    corpus = [Trajectory(id="deep", outcome=Outcome.FAILED.value, n_steps=12,
                         session_id="req-deep"),
              Trajectory(id="shallow", outcome=Outcome.FAILED.value, n_steps=1,
                         session_id="req-shallow")]
    # Calibration says the SHALLOW turn was the bad one.
    samples = [SimpleNamespace(req_id="req-deep", composite=0.99),
               SimpleNamespace(req_id="req-shallow", composite=0.01)]
    seen = []

    async def _critique(prompt):
        return ""

    refl = Reflector(critique_fn=_critique, max_failures=1,
                     calibration_source=lambda: samples)

    async def _spy(traj):
        seen.append(traj.id)
        return SimpleNamespace(ok=False, reflected_trajectory=None, error="")

    monkeypatch.setattr(refl, "_reflect_one", _spy)
    await refl.run(failed_source=corpus)
    assert seen == ["shallow"]


@pytest.mark.asyncio
async def test_broken_calibration_source_degrades_to_shape_ranking(monkeypatch):
    from ghost_agent.distill.schema import Outcome, Trajectory
    from ghost_agent.reflection.loop import Reflector

    monkeypatch.setenv(risk.ENV_TRIAGE_KILL, "1")

    def _boom():
        raise RuntimeError("calibration store on fire")

    corpus = [Trajectory(id="shallow", outcome=Outcome.FAILED.value, n_steps=1),
              Trajectory(id="deep", outcome=Outcome.FAILED.value, n_steps=12)]
    seen = []

    async def _critique(prompt):
        return ""

    refl = Reflector(critique_fn=_critique, max_failures=1,
                     calibration_source=_boom)

    async def _spy(traj):
        seen.append(traj.id)
        return SimpleNamespace(ok=False, reflected_trajectory=None, error="")

    monkeypatch.setattr(refl, "_reflect_one", _spy)
    await refl.run(failed_source=corpus)
    assert seen == ["deep"]


def test_gate_selects_a_tail_not_the_whole_corpus():
    """Backtested on the live corpus (2026-08-05): the gate fires on 7.6% of
    user turns and those turns fail at 0.543 vs a 0.277 baseline. Pin the
    shape of that claim — a threshold edit that made this fire on a third of
    all traffic would be a behaviour change disguised as a constant tweak."""
    typical_shallow = [
        risk.turn_risk(step=s, tool_names=["file_system"] * s)
        for s in range(1, 6)
    ]
    assert not any(r.should_steer for r in typical_shallow)
    # A mid-depth turn making steady progress (no repeats, no failures) is
    # still left alone at the gate depth.
    steady = risk.turn_risk(step=6, tool_names=["a", "b", "c", "d", "e", "f"])
    assert not steady.should_steer


# ── single-scale ranking (review findings, 2026-08-05) ────────────────

def test_partial_calibration_coverage_falls_back_to_one_scale():
    """Mixing the calibrated score with the shape proxy measured WORSE than
    either alone (Kendall tau 0.605 vs 0.766/0.763), and on the live corpus it
    demoted 7 of 8 joinable failures — a real 12-step disaster dropped out of
    the reflection queue in favour of a milder turn with no sample."""
    disaster = _traj("disaster", 12, ["execute"] * 12, errors=6,
                     session_id="req-has-sample")
    mild = _traj("mild", 3, ["file_system"] * 3, session_id="req-no-sample")
    # The calibrated value for the disaster is numerically LOW on its own
    # (compressed raw pre-Platt scale) — under the mixed ranking it sank.
    calibrated = {"req-has-sample": 0.20}
    ranked = risk.rank_for_triage([mild, disaster], calibrated=calibrated)
    assert [t.id for t in ranked] == ["disaster", "mild"]


def test_full_calibration_coverage_uses_the_calibrated_scale():
    a = _traj("a", 1, ["file_system"], session_id="req-a")
    b = _traj("b", 10, ["execute"] * 10, session_id="req-b")
    ranked = risk.rank_for_triage([b, a],
                                  calibrated={"req-a": 0.95, "req-b": 0.10})
    assert [t.id for t in ranked] == ["a", "b"]


def test_null_composite_does_not_score_maximum_risk():
    """`_clamp01(None)` is 0.0, so `1 - clamp` made a null read as 1.0 — the
    least informative row in the store jumping ahead of every real disaster."""
    samples = [SimpleNamespace(req_id="r-null", composite=None),
               SimpleNamespace(req_id="r-nan", composite=float("nan")),
               SimpleNamespace(req_id="r-ok", composite=0.8)]
    idx = risk.calibrated_risk_index(samples)
    assert "r-null" not in idx and "r-nan" not in idx
    assert idx["r-ok"] == pytest.approx(0.2)


def test_steer_message_does_not_claim_a_measurement_past_the_curve():
    """Past step 12 the prior is HELD, not measured. Stating it as a
    measurement at the current depth puts a fabricated fact into the model's
    own context, where it becomes a premise."""
    deep = risk.turn_risk(step=40, tool_names=["execute"] * 12,
                          execution_failures=3)
    msg = risk.risk_steer_message(deep)
    assert "at 12 steps or more" in msg
    assert "turns at this depth" not in msg
    # ...and no causal claim survives.
    assert "the cause is almost always" not in msg
    assert "one common pattern" in msg


def test_steer_message_still_says_this_depth_inside_the_curve():
    inside = risk.turn_risk(step=8, tool_names=["execute"] * 8,
                            execution_failures=2)
    assert "at this depth" in risk.risk_steer_message(inside)


# ── the calibration reader (sole producer of the calibrated ranking) ──

def test_recent_samples_window_and_isolation(tmp_path):
    """`grep recent_samples tests/` returned nothing — the only producer of
    the calibrated triage ranking was untested. If it silently returned an
    empty window, triage would fall back to the shape proxy with no signal."""
    import json
    from ghost_agent.core.calibration import CalibrationTracker

    tracker = CalibrationTracker(tmp_path)
    for i in range(40):
        tracker.record(composite=0.5, entropy_component=0.5,
                       competence_component=0.5, uncertainty_pressure=0.0,
                       outcome=1.0, req_id=f"req-{i}")
    assert len(tracker.recent_samples(10)) == 10
    assert len(tracker.recent_samples(1000)) == 40
    assert len(tracker.recent_samples(0)) == 1        # clamped, never empty
    # It is the TAIL (most recent), which is what a ranking window needs.
    assert tracker.recent_samples(5)[-1].req_id == "req-39"


def test_recent_samples_survives_a_torn_store(tmp_path):
    from ghost_agent.core.calibration import CalibrationTracker

    tracker = CalibrationTracker(tmp_path)
    tracker.record(composite=0.5, entropy_component=0.5,
                   competence_component=0.5, uncertainty_pressure=0.0,
                   outcome=1.0, req_id="good")
    with tracker.history_path.open("a", encoding="utf-8") as fh:
        fh.write("{torn line\n")
    assert [s.req_id for s in tracker.recent_samples(10)] == ["good"]
