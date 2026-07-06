"""End-to-end pin for the graduation PRODUCER wiring (2026-07-05).

The graduation pipeline (extract → consolidate → verify → store) was
structurally unreachable in production: the only trajectory producer
with real tool_calls (the chat recorder) writes outcome=UNKNOWN, and in
async-critic mode the verifier verdict lands AFTER the record — with no
backfill, the corpus never contained a single passed-with-tools
trajectory (live corpus 2026-07-05: 2058 UNKNOWN / 502 tool-less
reflection PASSED / 0 eligible). The fix routes the late verdict through
``collector.update_outcome`` (sidecar overlay); these tests pin the
chain from the sidecar write to extractor eligibility.
"""

from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Trajectory, ToolCall
from ghost_agent.skills_auto.extractor import extract_candidates
from ghost_agent.skills_auto.consolidator import consolidate


def _chat_traj(request="write and run a script"):
    """A chat-recorder-shaped trajectory: real tool calls, UNKNOWN."""
    return Trajectory(
        task_kind="user_request",
        user_request=request,
        tool_calls=[ToolCall(name="file_system"), ToolCall(name="execute")],
        outcome="unknown",
        final_response="done",
    )


def test_late_passed_overlay_feeds_extraction(tmp_path):
    collector = TrajectoryCollector(root=tmp_path)
    ids = []
    for _ in range(2):
        t = _chat_traj()
        collector.append(t)
        ids.append(t.id)
    # Pre-backfill: UNKNOWN chat turns are invisible to the extractor.
    cands, report = extract_candidates(
        list(collector.iter_trajectories()), min_support=2)
    assert cands == []
    assert report.n_passed_with_tools == 0

    # The late-verdict backfill lands (what _record_late_verdict now does).
    for tid in ids:
        assert collector.update_outcome(
            tid, "passed", source="verifier_late")

    # Post-backfill: the overlay makes both turns extractor-eligible and
    # the recurring sequence becomes a candidate.
    trajs = list(collector.iter_trajectories())
    assert [t.outcome for t in trajs] == ["passed", "passed"]
    cands, report = extract_candidates(trajs, min_support=2)
    assert len(cands) == 1
    assert cands[0].tool_sequence == ("file_system", "execute")
    assert cands[0].support == 2

    consolidated, _ = consolidate(cands)
    assert len(consolidated) == 1


def test_late_failed_overlay_reaches_reflector_input_shape(tmp_path):
    collector = TrajectoryCollector(root=tmp_path)
    t = _chat_traj()
    collector.append(t)
    collector.update_outcome(
        t.id, "failed",
        reason="verifier refuted (late): wrong output",
        source="verifier_late",
    )
    got = next(iter(collector.iter_trajectories()))
    assert got.outcome == "failed"
    assert "verifier refuted" in got.failure_reason


def test_user_correction_beats_late_passed(tmp_path):
    """Ordering: the user's next-turn verdict arrives after the late
    verifier verdict — last-write-wins per id must let it override."""
    collector = TrajectoryCollector(root=tmp_path)
    t = _chat_traj()
    collector.append(t)
    collector.update_outcome(t.id, "passed", source="verifier_late")
    collector.update_outcome(
        t.id, "failed", reason="user-correction", source="user_correction")
    got = next(iter(collector.iter_trajectories()))
    assert got.outcome == "failed"
