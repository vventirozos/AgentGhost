"""Prompt A/B evaluator.

Given two candidate prompts (baseline + candidate) and a callable that
renders+runs them against a set of examples, score both and return a
structured comparison. The eval harness is the scoring backbone so
"did this prompt help?" uses the same pass/fail discipline everything
else in Stage 1 uses.

The runner callable is injected; this module does no LLM work.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from .trainset import TrainExample


RunnerCallable = Callable[..., Union[str, Dict[str, Any], Awaitable[Any]]]


@dataclass
class PromptComparison:
    """Result of `compare_prompts`."""

    baseline_prompt: str
    candidate_prompt: str
    n_examples: int = 0
    baseline_pass_rate: float = 0.0
    candidate_pass_rate: float = 0.0
    delta: float = 0.0
    baseline_wins: int = 0
    candidate_wins: int = 0
    ties: int = 0
    per_example: List[Dict[str, Any]] = field(default_factory=list)
    # True iff the candidate beat the baseline by more than `min_delta`.
    candidate_ships: bool = False


async def _maybe_await(v: Any) -> Any:
    if inspect.isawaitable(v):
        return await v
    return v


async def _run_one(
    runner: RunnerCallable,
    prompt: str,
    example: TrainExample,
    timeout_s: float,
) -> Tuple[bool, Dict[str, Any]]:
    """Invoke the runner once; return (passed, meta). The runner is
    responsible for actually threading `prompt` through the LLM —
    this module is agnostic to how that happens."""
    payload = {
        "prompt": prompt,
        "inputs": example.inputs,
        "expected_output": example.expected_output,
        "signature_name": example.signature_name,
    }
    try:
        result = runner(payload)
        result = await asyncio.wait_for(_maybe_await(result), timeout=timeout_s)
    except asyncio.TimeoutError:
        return False, {"output": "", "failure_reason": f"timeout {timeout_s:.1f}s"}
    except Exception as e:
        return False, {"output": "", "failure_reason": f"{type(e).__name__}: {e}"}

    if isinstance(result, dict):
        passed = bool(result.get("passed"))
        return passed, result
    # Plain string return: non-empty = pass (weak signal; callers
    # generally pass a dict for meaningful eval).
    ok = bool(result and str(result).strip())
    return ok, {"output": result}


async def compare_prompts(
    baseline_prompt: str,
    candidate_prompt: str,
    examples: List[TrainExample],
    runner: RunnerCallable,
    *,
    min_delta: float = 0.02,
    per_example_timeout_s: float = 30.0,
) -> PromptComparison:
    """Run `runner(baseline)` and `runner(candidate)` on every example,
    collect pass rates, report a verdict.

    `min_delta` is the pass-rate improvement required to ship the
    candidate — below it, the candidate is considered noise and does
    not supersede the baseline.
    """
    cmp = PromptComparison(
        baseline_prompt=baseline_prompt,
        candidate_prompt=candidate_prompt,
        n_examples=len(examples),
    )
    if not examples:
        cmp.candidate_ships = False
        return cmp

    base_passes = 0
    cand_passes = 0
    for ex in examples:
        b_pass, b_meta = await _run_one(runner, baseline_prompt, ex, per_example_timeout_s)
        c_pass, c_meta = await _run_one(runner, candidate_prompt, ex, per_example_timeout_s)
        if b_pass:
            base_passes += 1
        if c_pass:
            cand_passes += 1
        if c_pass and not b_pass:
            cmp.candidate_wins += 1
        elif b_pass and not c_pass:
            cmp.baseline_wins += 1
        else:
            cmp.ties += 1
        cmp.per_example.append({
            "signature_name": ex.signature_name,
            "input": ex.inputs,
            "baseline_passed": b_pass,
            "candidate_passed": c_pass,
            "baseline_meta": b_meta,
            "candidate_meta": c_meta,
        })

    cmp.baseline_pass_rate = base_passes / len(examples)
    cmp.candidate_pass_rate = cand_passes / len(examples)
    cmp.delta = cmp.candidate_pass_rate - cmp.baseline_pass_rate
    cmp.candidate_ships = cmp.delta > min_delta
    return cmp
