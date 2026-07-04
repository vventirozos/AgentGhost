"""N-sample self-consistency sampler.

Under the single-LLM constraint there is no teacher. The closest thing
we have to "a better answer than the one we generated" is "the best
validator-passing answer out of N samples from the same model at varied
temperature". This module drives that pattern: run the same prompt N
times, label each sample pass/fail via a validator (when applicable),
emit a batch of labeled Trajectories ready to hand to a
TrajectoryCollector.

The pipeline downstream (rejection-sample SFT in Stage 2) keeps only
the pass-labeled trajectories; the fail-labeled ones still get
recorded because they carry useful negative-example signal.

This module does NOT run the model itself — it takes a runner callable
(same shape as the eval suite's runner) and drives it N times. That
makes it trivially testable and keeps model-wiring concerns in the
caller.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from .schema import Trajectory, Outcome


# Runner shape: async or sync, returns a dict (preferred) or string.
SampleRunner = Callable[..., Union[str, Dict[str, Any], Awaitable[Any]]]

# Validator shape. Returns (passed, reason). May be None if the caller
# has no verifier for this prompt — samples then carry Outcome.UNKNOWN.
SampleValidator = Callable[[str, Dict[str, Any]], Tuple[bool, str]]


@dataclass
class Sample:
    """Output of one run inside a batch. Thin wrapper so callers can
    inspect/filter before handing to a collector."""

    trajectory: Trajectory
    passed: Optional[bool]      # None when validator was not provided
    reason: str = ""


async def _maybe_await(v: Any) -> Any:
    if inspect.isawaitable(v):
        return await v
    return v


class SelfConsistencySampler:
    """Run an N-sample self-consistency batch against an injected runner.

    `temperatures` is the list of temperature values to try (one per
    sample). Length sets N. If `None`, defaults to a reasonable spread
    that encourages diversity without going off-piste:
        [0.2, 0.5, 0.7, 0.9, 1.0]
    """

    DEFAULT_TEMPERATURES: Tuple[float, ...] = (0.2, 0.5, 0.7, 0.9, 1.0)

    def __init__(
        self,
        runner: SampleRunner,
        *,
        temperatures: Optional[List[float]] = None,
        task_kind: str = "self_play",
        model: str = "",
    ):
        self.runner = runner
        self.temperatures = list(temperatures or self.DEFAULT_TEMPERATURES)
        self.task_kind = task_kind
        self.model = model

    async def sample(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        cluster: Optional[str] = None,
        tier: Optional[str] = None,
        session_id: str = "",
        validator: Optional[SampleValidator] = None,
        per_sample_timeout_s: float = 60.0,
    ) -> List[Sample]:
        """Run N samples for a single prompt and return labeled results.

        Errors in individual samples are captured as Outcome.FAILED
        with a failure_reason — the batch as a whole continues so a
        single runtime error doesn't waste the rest of the N samples.
        """
        batch_id = uuid.uuid4().hex[:12]
        out: List[Sample] = []
        for i, temp in enumerate(self.temperatures):
            sample = await self._one(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temp,
                sample_index=i,
                batch_id=batch_id,
                cluster=cluster,
                tier=tier,
                session_id=session_id,
                validator=validator,
                timeout_s=per_sample_timeout_s,
            )
            out.append(sample)
        return out

    async def _one(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        sample_index: int,
        batch_id: str,
        cluster: Optional[str],
        tier: Optional[str],
        session_id: str,
        validator: Optional[SampleValidator],
        timeout_s: float,
    ) -> Sample:
        start = time.monotonic()
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "cluster": cluster,
            "tier": tier,
        }
        output_text = ""
        metrics: Dict[str, Any] = {}
        failure_reason = ""
        try:
            result = self.runner(payload)
            result = await asyncio.wait_for(_maybe_await(result), timeout=timeout_s)
            if isinstance(result, dict):
                output_text = str(result.get("output") or "")
                metrics = result
            else:
                output_text = str(result or "")
        except asyncio.TimeoutError:
            failure_reason = f"timeout after {timeout_s:.1f}s"
        except Exception as e:
            failure_reason = f"runner raised: {type(e).__name__}: {e}"

        duration = time.monotonic() - start
        passed: Optional[bool] = None
        reason = failure_reason
        if failure_reason:
            passed = False
        elif validator is not None:
            try:
                passed, reason = validator(output_text, metrics)
            except Exception as e:
                passed = False
                reason = f"validator raised: {type(e).__name__}: {e}"

        outcome = (
            Outcome.PASSED.value if passed is True
            else Outcome.FAILED.value if passed is False
            else Outcome.UNKNOWN.value
        )

        # Coerce runner-supplied metrics defensively. These conversions live
        # OUTSIDE the runner/validator try above, so a non-numeric value
        # (steps="unknown", duration_s="n/a") would otherwise raise out of
        # _one and abort every sibling sample — defeating the whole point of
        # drawing N samples ("a single runtime error doesn't waste the rest").
        def _num(v, cast, default):
            try:
                return cast(v)
            except (TypeError, ValueError):
                return default
        _md = metrics if isinstance(metrics, dict) else {}
        _vs = _md.get("validator_signal")
        _ex = _md.get("extra")
        traj = Trajectory(
            session_id=session_id,
            task_kind=self.task_kind,
            cluster=cluster,
            tier=tier,
            model=self.model,
            temperature=temperature,
            sample_index=sample_index,
            batch_id=batch_id,
            system_prompt=system_prompt,
            user_request=prompt,
            tool_calls=[],  # Wiring tool-call capture is a caller concern
            n_steps=_num(_md.get("steps"), int, 0),
            tokens_in=_num(_md.get("tokens_in"), int, 0),
            tokens_out=_num(_md.get("tokens_out"), int, 0),
            duration_s=_num(_md.get("duration_s"), float, duration),
            outcome=outcome,
            failure_reason=failure_reason or (reason if passed is False else ""),
            validator_signal=_vs if isinstance(_vs, dict) else {},
            final_response=output_text,
            extra=_ex if isinstance(_ex, dict) else {},
        )
        return Sample(trajectory=traj, passed=passed, reason=reason)


def select_passing(samples: List[Sample]) -> List[Sample]:
    """Return only the samples the validator marked as passed."""
    return [s for s in samples if s.passed is True]


def select_failing(samples: List[Sample]) -> List[Sample]:
    """Return only the samples the validator marked as failed."""
    return [s for s in samples if s.passed is False]


def pairwise_pass_fail(samples: List[Sample]) -> List[Tuple[Sample, Sample]]:
    """Pair each failing sample with the first passing sample from the
    same batch. These are the (bad, good) pairs for preference
    learning / reflection training.

    Returns empty if the batch has no passes or no fails — in which
    case there's no signal to extract.
    """
    passes = select_passing(samples)
    fails = select_failing(samples)
    if not passes or not fails:
        return []
    best_pass = passes[0]  # index 0 = lowest temperature, most conservative
    return [(f, best_pass) for f in fails]
