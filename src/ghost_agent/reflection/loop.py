"""Reflector — the async driver for the reflection phase.

Responsibilities:
  1. Pick up to `max_failures` recent FAILED trajectories from the
     collector / source iterator.
  2. For each, build a reflection prompt, call the LLM, parse the
     response.
  3. Persist a new trajectory representing the reflected attempt
     (task_kind="reflection"), cross-linked to the failed source via
     `extra["reflected_from"]`.
  4. Return a ReflectionRunReport so the caller (biological watchdog)
     can log a single-line summary.

Safety:
  * Never raises. A reflection failure is logged and the run continues.
  * Never mutates input trajectories.
  * Respects per-call and total timeouts so a stalled LLM can't pin
    the watchdog.

The LLM call is abstracted via a callable injected at construction so
tests can pass a deterministic stub and the production caller can wrap
Ghost's LLMClient.
"""

from __future__ import annotations

import asyncio
import datetime
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, List, Optional, Set, Union

from ..distill.schema import Trajectory, Outcome
from .prompts import build_reflection_prompt, parse_reflection_output

logger = logging.getLogger("GhostReflect")


CritiqueCallable = Callable[[str], Union[str, Awaitable[str]]]


@dataclass
class ReflectionOutcome:
    """One failed-trajectory reflection result. Produced per-source."""

    source_trajectory_id: str
    reflected_trajectory: Optional[Trajectory] = None
    diagnosis: str = ""
    revised_plan: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.error == "" and self.reflected_trajectory is not None


@dataclass
class ReflectionRunReport:
    """Aggregate of one reflection tick."""

    seen_failures: int = 0
    reflected_ok: int = 0
    reflected_errors: int = 0
    skipped_duplicate: int = 0
    outcomes: List[ReflectionOutcome] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"reflected {self.reflected_ok}/{self.seen_failures} "
            f"(dup-skipped {self.skipped_duplicate}, errors {self.reflected_errors})"
        )


class Reflector:
    """Drives the reflection phase.

    Usage:
        refl = Reflector(critique_fn=my_llm_caller)
        report = await refl.run(
            failed_source=some_collector.iter_trajectories,
            sink=another_collector.append,
            already_reflected=ctx_attr_or_set,
        )
    """

    def __init__(
        self,
        critique_fn: CritiqueCallable,
        *,
        per_call_timeout_s: float = 45.0,
        max_failures: int = 3,
        model: str = "",
        session_id_prefix: str = "reflect",
        accept_low_novelty_passes: bool = False,
        novelty_threshold: float = 0.15,
    ):
        self.critique_fn = critique_fn
        self.per_call_timeout_s = float(per_call_timeout_s)
        self.max_failures = int(max_failures)
        self.model = model
        self.session_id_prefix = session_id_prefix
        # Proposal F (2026-05-17): when True, the reflector ALSO picks
        # up self-play trajectories that PASSED but with novelty below
        # `novelty_threshold`. These are the cycles where the agent
        # re-emitted a structurally-identical solution to a prior win —
        # technically a pass, but with no learning signal under the
        # new score. Pre-2026-05 these trajectories never reached the
        # reflector because the filter was strict on FAILED.
        self.accept_low_novelty_passes = bool(accept_low_novelty_passes)
        self.novelty_threshold = float(novelty_threshold)

    async def run(
        self,
        *,
        failed_source: Union[
            Iterable[Trajectory],
            Callable[[], Iterable[Trajectory]],
        ],
        sink: Optional[Callable[[Trajectory], Any]] = None,
        already_reflected: Optional[Set[str]] = None,
    ) -> ReflectionRunReport:
        """Iterate failed_source, reflect up to max_failures of them, persist via sink.

        `already_reflected` (if provided) is a set of source trajectory
        ids we've already reflected on — they're skipped and counted
        toward `skipped_duplicate`. Pass a shared set on `context` if
        you want cross-phase idempotency.
        """
        report = ReflectionRunReport()
        reflected_set: Set[str] = already_reflected if already_reflected is not None else set()

        # Resolve the source to an iterator. Accepts either a raw
        # iterable or a zero-arg callable producing one (mirrors how
        # TrajectoryCollector.iter_trajectories is used).
        if callable(failed_source) and not hasattr(failed_source, "__iter__"):
            iterable = failed_source()
        else:
            iterable = failed_source

        candidates: List[Trajectory] = []
        for t in iterable:
            if not self._is_reflectable(t):
                continue
            report.seen_failures += 1
            if t.id in reflected_set:
                report.skipped_duplicate += 1
                continue
            candidates.append(t)
            if len(candidates) >= self.max_failures:
                break

        for traj in candidates:
            outcome = await self._reflect_one(traj)
            report.outcomes.append(outcome)
            reflected_set.add(traj.id)
            if outcome.ok:
                report.reflected_ok += 1
                if sink is not None and outcome.reflected_trajectory is not None:
                    try:
                        sink(outcome.reflected_trajectory)
                    except Exception as e:
                        logger.warning("reflection sink failed: %s", e)
            else:
                report.reflected_errors += 1

        return report

    async def reflect_one(
        self,
        traj: Trajectory,
        *,
        sink: Optional[Callable[[Trajectory], Any]] = None,
        already_reflected: Optional[Set[str]] = None,
    ) -> ReflectionOutcome:
        """Reflect on a SINGLE trajectory, bypassing the iterator.

        Used by the post-turn reflection scheduler in ``handle_chat``:
        once a user-correction promotes the prior trajectory to
        FAILED we want the lesson to land before the user's *next*
        message, which means we can't wait for the biological
        watchdog's 15-60 min idle window. Calling this directly via
        ``asyncio.create_task`` schedules reflection without blocking
        the user-facing turn.

        Honours ``already_reflected`` exactly the same way ``run``
        does: if ``traj.id`` is already in the set, returns a
        no-op outcome flagged with ``error="already reflected"``;
        otherwise adds ``traj.id`` to the set BEFORE awaiting the
        critique so a concurrent ``run`` tick can't double-reflect.
        """
        reflected_set: Set[str] = (
            already_reflected if already_reflected is not None else set()
        )
        if traj.id in reflected_set:
            return ReflectionOutcome(
                source_trajectory_id=traj.id,
                error="already reflected",
            )
        # Add BEFORE await — racing with the watchdog tick is the
        # whole point of having a shared dedup set.
        reflected_set.add(traj.id)

        out = await self._reflect_one(traj)
        if out.ok and sink is not None and out.reflected_trajectory is not None:
            try:
                sink(out.reflected_trajectory)
            except Exception as e:
                logger.warning("reflect_one sink failed: %s", e)
        return out

    def _is_reflectable(self, traj: Trajectory) -> bool:
        """Decide whether a trajectory belongs in the reflection batch.

        Always accept ``FAILED``. When ``accept_low_novelty_passes`` is
        on, ALSO accept self-play trajectories that passed but whose
        recorded novelty score is below the threshold — those cycles
        produced no learning signal under the new score (proposal F).
        Everything else is ignored.
        """
        if traj.outcome == Outcome.FAILED.value:
            return True
        if not self.accept_low_novelty_passes:
            return False
        # Self-play passes with very low novelty count as "boring wins"
        # worth meta-critiquing. Read the novelty signal from the
        # trajectory's `extra` dict where dream.py stashes it.
        extra = getattr(traj, "extra", None) or {}
        if not isinstance(extra, dict):
            return False
        if (extra.get("task_kind") or "").lower() != "self_play" and \
                (getattr(traj, "task_kind", "") or "").lower() != "self_play":
            return False
        novelty = extra.get("solution_novelty")
        if novelty is None:
            return False
        try:
            return float(novelty) < self.novelty_threshold
        except Exception:
            return False

    async def _reflect_one(self, traj: Trajectory) -> ReflectionOutcome:
        out = ReflectionOutcome(source_trajectory_id=traj.id)
        prompt = build_reflection_prompt(traj)
        start = time.monotonic()
        response_text: str = ""
        try:
            call = self.critique_fn(prompt)
            if inspect.isawaitable(call):
                response_text = await asyncio.wait_for(call, timeout=self.per_call_timeout_s)
            else:
                response_text = str(call)
        except asyncio.TimeoutError:
            out.error = f"timeout after {self.per_call_timeout_s:.1f}s"
            return out
        except Exception as e:
            out.error = f"critique_fn raised {type(e).__name__}: {e}"
            return out

        diagnosis, plan = parse_reflection_output(response_text or "")
        if not diagnosis and not plan:
            out.error = "unparseable reflection response"
            return out

        out.diagnosis = diagnosis
        out.revised_plan = plan

        duration = time.monotonic() - start
        reflected = Trajectory(
            id=uuid.uuid4().hex,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            session_id=f"{self.session_id_prefix}:{traj.session_id or 'anon'}",
            task_kind="reflection",
            cluster=traj.cluster,
            tier=traj.tier,
            model=self.model or traj.model,
            temperature=0.3,
            sample_index=None,
            batch_id=None,
            system_prompt="",  # reflection uses its own template, not the chat prompt
            user_request=traj.user_request,
            planning_output="\n".join(f"{i+1}. {s}" for i, s in enumerate(plan)),
            tool_calls=[],
            n_steps=len(plan),
            tokens_in=0,
            tokens_out=0,
            duration_s=float(duration),
            outcome=Outcome.UNKNOWN.value,  # reflection plan isn't executed; unknown
            failure_reason="",
            validator_signal={},
            final_response=f"DIAGNOSIS: {diagnosis}\n"
                           + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan)),
            extra={
                "reflected_from": traj.id,
                "source_failure_reason": traj.failure_reason,
                "source_outcome": traj.outcome,
            },
        )
        out.reflected_trajectory = reflected
        return out
