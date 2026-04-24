"""Tests for optim.trainset."""

from ghost_agent.distill.schema import Trajectory, Outcome
from ghost_agent.optim.trainset import (
    TrainExample, build_trainset, filter_by_outcome, split_train_eval,
)


def _t(passed=True, steps=3, user_request="do thing", final_response="done",
       batch_id=None, sample_index=0, temperature=0.5):
    return Trajectory(
        user_request=user_request,
        final_response=final_response,
        n_steps=steps,
        outcome=Outcome.PASSED.value if passed else Outcome.FAILED.value,
        batch_id=batch_id,
        sample_index=sample_index,
        temperature=temperature,
    )


def test_filter_by_outcome_keeps_only_passed():
    trajs = [_t(passed=True), _t(passed=False), _t(passed=True)]
    kept = filter_by_outcome(trajs)
    assert len(kept) == 2


def test_filter_by_outcome_min_steps_floor():
    trajs = [_t(steps=0), _t(steps=2), _t(steps=5)]
    kept = filter_by_outcome(trajs, min_steps=2)
    assert len(kept) == 2


def test_filter_by_outcome_max_steps_ceiling():
    trajs = [_t(steps=2), _t(steps=5), _t(steps=15)]
    kept = filter_by_outcome(trajs, max_steps=10)
    assert len(kept) == 2


def test_build_trainset_basic():
    trajs = [
        _t(user_request="req A", final_response="ans A"),
        _t(user_request="req B", final_response="ans B"),
        _t(passed=False, user_request="skip", final_response="bad"),
    ]
    examples = build_trainset(trajs, signature_name="planning.decompose")
    assert len(examples) == 2
    for ex in examples:
        assert ex.signature_name == "planning.decompose"
        assert ex.inputs["user_request"] in ("req A", "req B")
        assert ex.expected_output["final_response"] in ("ans A", "ans B")


def test_build_trainset_dedupes_by_batch_keeps_lowest_temp():
    trajs = [
        _t(batch_id="b1", temperature=0.9, user_request="Q", final_response="hi temp ans"),
        _t(batch_id="b1", temperature=0.2, user_request="Q", final_response="low temp ans"),
        _t(batch_id="b1", temperature=0.5, user_request="Q", final_response="med temp ans"),
    ]
    examples = build_trainset(trajs, signature_name="planning.decompose")
    assert len(examples) == 1
    assert examples[0].expected_output["final_response"] == "low temp ans"


def test_build_trainset_drops_empty_request_or_response():
    trajs = [
        _t(user_request="", final_response="only output"),
        _t(user_request="only request", final_response=""),
        _t(user_request="both", final_response="present"),
    ]
    examples = build_trainset(trajs, signature_name="planning.decompose")
    assert len(examples) == 1
    assert examples[0].inputs["user_request"] == "both"


def test_build_trainset_max_examples_cap():
    trajs = [_t(user_request=f"q{i}", final_response=f"a{i}") for i in range(10)]
    examples = build_trainset(trajs, signature_name="planning.decompose",
                              max_examples=3)
    assert len(examples) == 3


def test_build_trainset_respects_require_passed_false():
    trajs = [_t(passed=False, user_request="failed", final_response="tried")]
    examples = build_trainset(trajs, signature_name="planning.decompose",
                              require_passed=False)
    assert len(examples) == 1


def test_split_train_eval_default_fraction():
    examples = [TrainExample(signature_name="x") for _ in range(10)]
    train, eval_ = split_train_eval(examples, eval_fraction=0.2, random_state=42)
    assert len(train) == 8
    assert len(eval_) == 2


def test_split_train_eval_edge_zero():
    examples = [TrainExample(signature_name="x") for _ in range(5)]
    train, eval_ = split_train_eval(examples, eval_fraction=0.0)
    assert len(train) == 5
    assert len(eval_) == 0


def test_split_train_eval_edge_one():
    examples = [TrainExample(signature_name="x") for _ in range(5)]
    train, eval_ = split_train_eval(examples, eval_fraction=1.0)
    assert len(train) == 0
    assert len(eval_) == 5


def test_split_train_eval_deterministic_with_seed():
    examples = [TrainExample(signature_name="x", inputs={"i": str(i)})
                for i in range(20)]
    a_train, a_eval = split_train_eval(examples, random_state=7)
    b_train, b_eval = split_train_eval(examples, random_state=7)
    assert [e.inputs["i"] for e in a_train] == [e.inputs["i"] for e in b_train]
    assert [e.inputs["i"] for e in a_eval] == [e.inputs["i"] for e in b_eval]


def test_split_train_eval_different_seeds_differ():
    examples = [TrainExample(signature_name="x", inputs={"i": str(i)})
                for i in range(20)]
    a_train, _ = split_train_eval(examples, random_state=1)
    b_train, _ = split_train_eval(examples, random_state=2)
    # Seeds 1 vs 2 should produce a different ordering
    assert [e.inputs["i"] for e in a_train] != [e.inputs["i"] for e in b_train]


def test_split_empty_examples():
    train, eval_ = split_train_eval([], eval_fraction=0.2)
    assert train == []
    assert eval_ == []
