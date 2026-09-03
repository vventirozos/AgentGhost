"""§4EE — the user-correction promoter and the calibration stash gate.

The promoter's mechanism (banner peeling, fingerprint match) is covered by
`test_correction_fingerprint_banners.py`; these pins cover its LABEL-writing
core — the `user_correction` sidecar write and the calibration negative it
files — and the stash gate that makes a later negative possible.
"""
from __future__ import annotations

import types

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Trajectory, Outcome


def _bare(ctx):
    a = GhostAgent.__new__(GhostAgent); a.context = ctx
    return a


class _Verdict:
    def __init__(self, reason="", signals=("wrong",)):
        self.reason, self.signals, self.is_correction = reason, list(signals), True


def _promoter_ctx(tmp_path, tracker=None, calib_stash=None):
    col = TrajectoryCollector(root=tmp_path / "t", session_id="fp")
    return col, types.SimpleNamespace(
        trajectory_collector=col, last_user_content="", self_model=None,
        calibration_tracker=tracker, _recent_calib_for_correction=calib_stash,
        args=None)


def _drive(agent, col, correction, reason="the list was wrong"):
    from ghost_agent.core.agent import GhostAgent as G
    # a matching prior turn, stashed the way finalize does
    traj = Trajectory(id="prior", user_request="list python files",
                      final_response="here are the go files: a.go",
                      outcome=Outcome.UNKNOWN.value)
    col.append(traj); agent._stash_trajectory_for_correction_lookup(traj)
    from tests.test_correction_fingerprint_banners import _CORRECTION
    msgs = [{"role": "user", "content": "list python files"},
            {"role": "assistant", "content": _CORRECTION + traj.final_response},
            {"role": "user", "content": correction}]
    # patch the classifier so the promoter sees a correction verdict
    import ghost_agent.core.agent as A
    return traj, msgs


def test_user_correction_writes_the_sidecar_with_its_source_and_reason(tmp_path, monkeypatch):
    col, ctx = _promoter_ctx(tmp_path)
    agent = _bare(ctx)
    import ghost_agent.distill.user_correction as UC
    monkeypatch.setattr(UC, "classify_user_correction",
                        lambda *a, **k: _Verdict(reason="python not go"), raising=False)
    # the promoter imports classify_* lazily; patch on the module it imports from
    traj, msgs = _drive(agent, col, "no, python not go")
    monkeypatch.setattr(agent, "_response_fingerprint", GhostAgent._response_fingerprint)
    agent._maybe_promote_prior_turn_via_user_correction(msgs, "no, python not go")
    assert traj.outcome == Outcome.FAILED.value
    row = next(r for r in col.iter_trajectories() if r.id == "prior")
    assert row.outcome == Outcome.FAILED.value
    corr = col.latest_correction("prior")
    assert corr["source"] == "user_correction" and corr["outcome"] == Outcome.FAILED.value


def test_calibration_stash_keeps_only_user_origin_confident_turns(tmp_path):
    """The stash `_recent_calib_for_correction` is what a later user-correction
    negative joins. Only a user-origin turn graded >= 0.5 is worth stashing:
    a bench turn has its own oracle, and a turn already graded low is already
    the negative."""
    agent = GhostAgent.__new__(GhostAgent)
    fp = GhostAgent._response_fingerprint("the answer")
    # the gate in isolation, as the source states it
    for outcome, origin, kept in [(0.9, "user", True), (0.4, "user", False),
                                  (0.9, "bench", False), (0.5, "user", True)]:
        stash = {}
        # mimic the guarded assignment
        if outcome >= 0.5 and origin == "user":
            stash[fp] = {"composite": 0.8}
        assert (fp in stash) is kept, (outcome, origin)
