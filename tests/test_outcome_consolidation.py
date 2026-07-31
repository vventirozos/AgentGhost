"""Step 2 of the post-hunt strategy: consolidate the outcome signal.

Historically a turn's "was this good?" verdict diverged across consumers —
calibration and the selfhood model became verifier-aware, but the trajectory
corpus that feeds the Reflector / PRM / skills-auto saw only the shape
heuristics. So a verifier-caught wrong answer stayed UNKNOWN in the corpus and
never became a lesson or a PRM negative.

`resolve_turn_outcome` is the single combiner, and `_record_turn_trajectory`
now applies it so the corpus outcome matches calibration + selfhood.
"""

import types

import pytest

from ghost_agent.distill.outcome_heuristics import resolve_turn_outcome
from ghost_agent.distill.schema import Outcome

P, F, U = Outcome.PASSED.value, Outcome.FAILED.value, Outcome.UNKNOWN.value


# ══════════════════════════════════════════════════════════════════════
# The pure combiner — priority order
# ══════════════════════════════════════════════════════════════════════

class TestResolveTurnOutcome:
    def test_structural_failure_fails_when_unverified(self):
        assert resolve_turn_outcome(current=U, execution_failed=True) == F

    def test_honest_failure_report_passes(self):
        """PRIORITY REVERSED 2026-07-31 (operator decision): structural
        failure used to beat a verifier PASS, so a turn whose only tool call
        failed and whose answer HONESTLY reported that failure ("that file
        does not exist") was labelled FAILED. That taught the corpus,
        calibration and skills-auto that truthful failure-reporting is bad
        behaviour — the exact incentive that breeds fabricated success. The
        verifier inspected the answer against the evidence; it wins. The
        environment failed, the turn did not."""
        assert resolve_turn_outcome(
            current=U, execution_failed=True, verifier="passed") == P

    def test_refute_still_beats_everything(self):
        """The honest-failure rule must not weaken the refute arm."""
        assert resolve_turn_outcome(
            current=U, execution_failed=True, verifier="failed") == F

    def test_shape_heuristic_failed_not_upgraded_by_honest_report(self):
        """Rule 2 still guards: a turn that thrashed a selector 4× or hit a
        runtime abort marker stays FAILED however honestly it says so —
        that failure is BEHAVIOURAL, not environmental."""
        assert resolve_turn_outcome(
            current=F, execution_failed=True, verifier="passed") == F

    def test_verifier_refuted_fails(self):
        assert resolve_turn_outcome(current=U, verifier="failed") == F

    def test_existing_failed_never_upgraded(self):
        assert resolve_turn_outcome(current=F, verifier="passed") == F
        assert resolve_turn_outcome(current=F, verifier=None) == F

    def test_verifier_supported_passes(self):
        assert resolve_turn_outcome(current=U, verifier="passed") == P

    def test_signal_free_chat_stays_unknown(self):
        assert resolve_turn_outcome(current=U) == U
        assert resolve_turn_outcome(current=U, verifier=None, execution_failed=False) == U

    def test_current_passed_preserved(self):
        assert resolve_turn_outcome(current=P) == P

    def test_none_current_defaults_unknown(self):
        assert resolve_turn_outcome(current=None) == U


# ══════════════════════════════════════════════════════════════════════
# Integration — the corpus outcome is now verifier-aware
# ══════════════════════════════════════════════════════════════════════

class _FakeCollector:
    def __init__(self):
        self.appended = []

    def append(self, traj):
        self.appended.append(traj)


def _agent_with_collector():
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent.__new__(GhostAgent)
    col = _FakeCollector()
    agent.context = types.SimpleNamespace(
        trajectory_collector=col,
        self_model=None,
        profile_memory=None,
        _recent_calib_for_correction=None,
    )
    return agent, col


_MSGS = [
    {"role": "user", "content": "answer this factual question"},
    {"role": "assistant", "content": "here is a confident but wrong answer"},
]


class TestRecordTrajectoryConsolidation:
    def test_verifier_refuted_written_as_failed(self):
        agent, col = _agent_with_collector()
        agent._record_turn_trajectory(
            messages=_MSGS, final_content="here is a confident but wrong answer",
            req_id="r1", model="m", trajectory_id="t1",
            user_request="answer this factual question",
            verifier="failed",
        )
        assert col.appended, "trajectory was not recorded"
        # The corpus record now reflects the verifier verdict (was UNKNOWN).
        assert col.appended[0].outcome == F

    def test_structural_failure_written_as_failed(self):
        agent, col = _agent_with_collector()
        agent._record_turn_trajectory(
            messages=_MSGS, final_content="x", req_id="r2", model="m",
            trajectory_id="t2", user_request="q", execution_failed=True,
        )
        assert col.appended[0].outcome == F

    def test_clean_chat_stays_unknown(self):
        agent, col = _agent_with_collector()
        agent._record_turn_trajectory(
            messages=_MSGS, final_content="a fine answer", req_id="r3", model="m",
            trajectory_id="t3", user_request="q",
        )
        # No verifier verdict, no structural failure, clean shape → UNKNOWN.
        assert col.appended[0].outcome == U

    def test_verifier_passed_written_as_passed(self):
        agent, col = _agent_with_collector()
        agent._record_turn_trajectory(
            messages=_MSGS, final_content="correct answer", req_id="r4", model="m",
            trajectory_id="t4", user_request="q", verifier="passed",
        )
        assert col.appended[0].outcome == P


class TestHonestFailureRecording:
    """End-to-end for the 2026-07-31 honest-failure rule, both delivery
    paths. The sync path folds the verdict in at write time; the ASYNC path
    (production default, GHOST_CRITIC_ASYNC=1) writes FAILED first and must
    let the late CONFIRMED upgrade it — otherwise the two paths disagree on
    identical evidence, which is how this class of mislabel hides."""

    def test_sync_path_writes_passed(self):
        agent, col = _agent_with_collector()
        agent._record_turn_trajectory(
            messages=_MSGS, final_content="that file does not exist",
            req_id="rh1", model="m", trajectory_id="th1",
            user_request="read missing.txt", verifier="passed",
            execution_failed=True,
        )
        assert col.appended[0].outcome == P

    def test_structural_failed_carries_the_upgradable_reason(self):
        """The reason string is the async path's only discriminator between
        'the tool broke' and 'the answer was refuted'."""
        from ghost_agent.distill.outcome_heuristics import (
            STRUCTURAL_FAILURE_REASON,
        )
        agent, col = _agent_with_collector()
        agent._record_turn_trajectory(
            messages=_MSGS, final_content="x", req_id="rh2", model="m",
            trajectory_id="th2", user_request="q", execution_failed=True,
        )
        assert col.appended[0].outcome == F
        assert col.appended[0].failure_reason == STRUCTURAL_FAILURE_REASON

    def test_late_pass_upgrades_structural_failed(self):
        from ghost_agent.distill.outcome_heuristics import (
            STRUCTURAL_FAILURE_REASON,
        )
        agent, _col = _agent_with_collector()
        traj = types.SimpleNamespace(
            id="th3", outcome=F, failure_reason=STRUCTURAL_FAILURE_REASON)
        agent.context._recent_trajectories_for_correction = {"k": traj}
        agent._backfill_trajectory_outcome("th3", P)
        assert traj.outcome == P
        assert traj.failure_reason == ""  # stale reason cleared

    def test_late_pass_does_not_upgrade_a_refute(self):
        agent, _col = _agent_with_collector()
        traj = types.SimpleNamespace(
            id="th4", outcome=F, failure_reason="verifier refuted")
        agent.context._recent_trajectories_for_correction = {"k": traj}
        agent._backfill_trajectory_outcome("th4", P)
        assert traj.outcome == F

    def test_late_pass_does_not_upgrade_a_shape_heuristic_failure(self):
        agent, _col = _agent_with_collector()
        traj = types.SimpleNamespace(
            id="th5", outcome=F,
            failure_reason="browser selector '#go' used 5× in one turn")
        agent.context._recent_trajectories_for_correction = {"k": traj}
        agent._backfill_trajectory_outcome("th5", P)
        assert traj.outcome == F
