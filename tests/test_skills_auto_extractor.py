"""Tests for skills_auto.extractor."""

from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome
from ghost_agent.skills_auto.extractor import (
    extract_candidates, summarize_tool_sequence, SkillCandidate,
)


def _mk(cluster="sql", passed=True, tools=("file_system", "execute"),
        user_request="", traj_id=None):
    calls = [ToolCall(name=n) for n in tools]
    kw = dict(
        cluster=cluster,
        tool_calls=calls,
        user_request=user_request,
        outcome=Outcome.PASSED.value if passed else Outcome.FAILED.value,
    )
    if traj_id is not None:
        kw["id"] = traj_id
    return Trajectory(**kw)


def test_summarize_tool_sequence_names_in_order():
    calls = [ToolCall(name="a"), ToolCall(name="b"), ToolCall(name="c")]
    assert summarize_tool_sequence(calls) == ("a", "b", "c")


def test_summarize_skips_empty_names():
    calls = [ToolCall(name="a"), ToolCall(name="   "), ToolCall(name="b")]
    assert summarize_tool_sequence(calls) == ("a", "b")


def test_summarize_none_safe():
    assert summarize_tool_sequence(None) == ()


def test_extract_requires_min_support():
    trajs = [_mk() for _ in range(2)]
    cands, report = extract_candidates(trajs, min_support=2)
    assert len(cands) == 1
    assert cands[0].support == 2

    trajs_one = [_mk()]
    cands, _ = extract_candidates(trajs_one, min_support=2)
    assert cands == []


def test_extract_drops_trajectories_below_min_tool_calls():
    trajs = [
        _mk(tools=("execute",)),
        _mk(tools=("execute",)),
    ]
    cands, report = extract_candidates(trajs, min_support=2, min_tool_calls=2)
    assert cands == []
    assert report.n_passed_with_tools == 0


def test_extract_only_counts_passed_trajectories_for_support():
    trajs = [
        _mk(passed=True),
        _mk(passed=False),  # counts toward failure tally, not support
        _mk(passed=True),
    ]
    cands, _ = extract_candidates(trajs, min_support=2)
    assert len(cands) == 1
    assert cands[0].support == 2


def test_confidence_penalized_by_failures():
    passed = [_mk() for _ in range(3)]
    both = passed + [_mk(passed=False), _mk(passed=False)]
    only_pass, _ = extract_candidates(passed, min_support=2)
    with_fails, _ = extract_candidates(both, min_support=2)
    assert only_pass[0].support == 3
    assert with_fails[0].support == 3
    assert with_fails[0].confidence < only_pass[0].confidence


def test_different_sequences_are_different_candidates():
    trajs = [
        _mk(tools=("a", "b")),
        _mk(tools=("a", "b")),
        _mk(tools=("c", "d")),
        _mk(tools=("c", "d")),
    ]
    cands, _ = extract_candidates(trajs, min_support=2)
    assert len(cands) == 2
    seqs = {c.tool_sequence for c in cands}
    assert seqs == {("a", "b"), ("c", "d")}


def test_candidates_sorted_by_descending_support():
    trajs = (
        [_mk(tools=("a", "b")) for _ in range(5)]
        + [_mk(tools=("c", "d")) for _ in range(2)]
    )
    cands, _ = extract_candidates(trajs, min_support=2)
    assert cands[0].support >= cands[1].support


def test_exemplar_is_shortest_trajectory():
    trajs = [
        _mk(traj_id="t_long", tools=("a", "b", "c")),
        _mk(traj_id="t_short", tools=("a", "b", "c")),
        _mk(traj_id="t_mid", tools=("a", "b", "c")),
    ]
    # All have the same tool count here — we synthesize a duration
    # difference instead.
    trajs[0].duration_s = 10.0
    trajs[1].duration_s = 1.0
    trajs[2].duration_s = 5.0
    cands, _ = extract_candidates(trajs, min_support=2)
    assert cands[0].exemplar_trajectory_id == "t_short"


def test_trigger_examples_capped_and_non_empty_only():
    trajs = [
        _mk(user_request="", traj_id=f"t{i}") for i in range(2)
    ] + [
        _mk(user_request=f"req{i}", traj_id=f"r{i}") for i in range(5)
    ]
    cands, _ = extract_candidates(trajs, min_support=2, max_trigger_examples=3)
    assert cands
    assert len(cands[0].trigger_examples) == 3
    for tg in cands[0].trigger_examples:
        assert tg.startswith("req")


def test_candidate_name_is_stable_for_same_sequence():
    trajs_a = [_mk() for _ in range(2)]
    trajs_b = [_mk() for _ in range(2)]
    cands_a, _ = extract_candidates(trajs_a, min_support=2)
    cands_b, _ = extract_candidates(trajs_b, min_support=2)
    assert cands_a[0].name == cands_b[0].name


def test_signature_hash_distinguishes_cluster_variants():
    trajs = [
        _mk(cluster="sql", tools=("a", "b")),
        _mk(cluster="sql", tools=("a", "b")),
        _mk(cluster="bash", tools=("a", "b")),
        _mk(cluster="bash", tools=("a", "b")),
    ]
    cands, _ = extract_candidates(trajs, min_support=2)
    hashes = {c.signature_hash for c in cands}
    assert len(hashes) == 2


def test_extraction_report_counts():
    trajs = (
        [_mk() for _ in range(3)]
        + [_mk(tools=("only_one",)) for _ in range(2)]
    )
    cands, report = extract_candidates(trajs, min_support=2)
    assert report.n_trajectories_seen == 5
    assert report.n_passed_with_tools == 3
    assert report.n_candidates_emitted == len(cands)
