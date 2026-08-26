"""Tests for optim.ab_eval."""

import asyncio

import pytest

from ghost_agent.optim.ab_eval import compare_prompts, PromptComparison
from ghost_agent.optim.trainset import TrainExample


def _ex(i: int) -> TrainExample:
    return TrainExample(
        signature_name="planning.decompose",
        inputs={"user_request": f"request-{i}"},
        expected_output={"final_response": f"answer-{i}"},
        source_trajectory_id=f"tr-{i}",
    )


async def _runner_baseline_always_passes(payload):
    if "BASELINE" in payload["prompt"]:
        return {"passed": True, "output": "base-ok"}
    return {"passed": False, "output": "cand-no"}


async def _runner_candidate_always_passes(payload):
    if "CANDIDATE" in payload["prompt"]:
        return {"passed": True, "output": "cand-ok"}
    return {"passed": False, "output": "base-no"}


async def _runner_both_pass(payload):
    return {"passed": True, "output": "ok"}


async def _runner_slow(payload):
    await asyncio.sleep(5.0)
    return {"passed": True, "output": "late"}


async def _runner_raises(payload):
    raise RuntimeError("model down")


async def test_compare_prompts_reports_candidate_improvement():
    # SIX examples. The test is one-sided now, so a 5-0 sweep (p=0.03125)
    # would already ship; six keeps a margin above the 5-pair floor so this
    # stays a smoke test of "an improvement is reported" rather than a
    # boundary case. The boundary itself is pinned in
    # test_gepa_optim_reaudit.py (4-0 refused, 5-0 shipped).
    examples = [_ex(i) for i in range(6)]
    cmp = await compare_prompts(
        baseline_prompt="BASELINE",
        candidate_prompt="CANDIDATE",
        examples=examples,
        runner=_runner_candidate_always_passes,
    )
    assert cmp.baseline_pass_rate == 0.0
    assert cmp.candidate_pass_rate == 1.0
    assert cmp.delta == 1.0
    assert cmp.candidate_ships is True
    assert cmp.candidate_wins == 6
    assert cmp.p_value is not None and cmp.p_value <= 0.05


async def test_compare_prompts_reports_baseline_better():
    examples = [_ex(i) for i in range(5)]
    cmp = await compare_prompts(
        baseline_prompt="BASELINE",
        candidate_prompt="CANDIDATE",
        examples=examples,
        runner=_runner_baseline_always_passes,
    )
    assert cmp.delta == -1.0
    assert cmp.candidate_ships is False
    assert cmp.baseline_wins == 5


async def test_compare_prompts_ties_when_both_pass():
    examples = [_ex(i) for i in range(3)]
    cmp = await compare_prompts(
        baseline_prompt="BASELINE",
        candidate_prompt="CANDIDATE",
        examples=examples,
        runner=_runner_both_pass,
    )
    assert cmp.delta == 0.0
    assert cmp.ties == 3
    assert cmp.candidate_ships is False


async def test_compare_prompts_runner_timeout_marks_fail():
    examples = [_ex(0)]
    cmp = await compare_prompts(
        baseline_prompt="B",
        candidate_prompt="C",
        examples=examples,
        runner=_runner_slow,
        per_example_timeout_s=0.05,
    )
    assert cmp.baseline_pass_rate == 0.0
    assert cmp.candidate_pass_rate == 0.0


async def test_compare_prompts_runner_exception_marks_fail():
    examples = [_ex(0)]
    cmp = await compare_prompts(
        baseline_prompt="B",
        candidate_prompt="C",
        examples=examples,
        runner=_runner_raises,
    )
    assert cmp.baseline_pass_rate == 0.0
    assert cmp.candidate_pass_rate == 0.0
    # Error detail surfaced in per_example meta.
    assert "RuntimeError" in cmp.per_example[0]["baseline_meta"]["failure_reason"]


async def test_compare_prompts_empty_examples_non_shipping():
    cmp = await compare_prompts(
        baseline_prompt="B", candidate_prompt="C",
        examples=[], runner=_runner_both_pass,
    )
    assert cmp.n_examples == 0
    assert cmp.candidate_ships is False


async def test_min_delta_gates_shipping():
    """Candidate beats baseline by 8pp. min_delta=0.10 → no ship,
    min_delta=0.02 → ship. Uses pp far from the threshold so
    floating-point precision doesn't interfere with the gate.

    ⚠ The margin is what this test is about, so the EVIDENCE must be
    ample enough not to confound it: 8 discordant pairs all one way is
    p=0.0039 one-sided, comfortably past `SHIP_ALPHA`. Written with a 5pp
    gap it produced exactly 5 discordant pairs, which sat on the boundary
    under the two-sided rule this file was written against and would have
    failed for a reason that has nothing to do with `min_delta`."""
    examples = [_ex(i) for i in range(100)]

    async def runner(payload):
        # Baseline 50%, candidate 58% → delta ≈ 0.08, 8 discordant pairs
        idx = int(payload["inputs"]["user_request"].split("-")[1])
        if "BASELINE" in payload["prompt"]:
            return {"passed": idx < 50, "output": ""}
        return {"passed": idx < 58, "output": ""}

    strict = await compare_prompts(
        baseline_prompt="BASELINE",
        candidate_prompt="CANDIDATE",
        examples=examples,
        runner=runner,
        min_delta=0.10,
    )
    assert strict.delta < 0.10
    assert strict.candidate_ships is False

    lenient = await compare_prompts(
        baseline_prompt="BASELINE",
        candidate_prompt="CANDIDATE",
        examples=examples,
        runner=runner,
        min_delta=0.02,
    )
    assert lenient.delta > 0.02
    assert lenient.p_value is not None and lenient.p_value <= 0.05
    assert lenient.candidate_ships is True


async def test_string_returns_are_passed_on_nonempty():
    examples = [_ex(0)]

    async def string_runner(payload):
        return "non-empty"

    cmp = await compare_prompts(
        baseline_prompt="B", candidate_prompt="C",
        examples=examples, runner=string_runner,
    )
    assert cmp.baseline_pass_rate == 1.0
    assert cmp.candidate_pass_rate == 1.0
