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
    def test_structural_failure_is_ground_truth(self):
        assert resolve_turn_outcome(current=U, execution_failed=True) == F
        # structural failure beats even a verifier "passed"
        assert resolve_turn_outcome(current=U, execution_failed=True, verifier="passed") == F

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
