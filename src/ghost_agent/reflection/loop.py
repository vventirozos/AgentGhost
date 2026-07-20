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
from typing import Any, Awaitable, Callable, Iterable, List, Optional, Set, Tuple, Union

from ..distill.schema import Trajectory, Outcome
from .prompts import build_reflection_prompt, parse_reflection_output

logger = logging.getLogger("GhostReflect")


# User prompts that are system-generated error-recovery scaffolding rather
# than real tasks. Reflecting on them turns a transient tooling failure (a
# malformed one-off diagnostic command, a self-repeating-loop trip) into a
# saved "lesson", polluting the skill playbook with recovery trivia keyed on
# the AUTO-DIAGNOSTIC banner itself — the exact failure where reflection
# learned "when you get this browser error, do X" instead of task knowledge.
# Deliberately NARROWER than selfhood.autobiographical._TEMPLATE_PROMPT_MARKERS:
# synthetic-training and judge-rejection turns ARE legitimate self-play
# learning signal and must still be reflected on.
_RECOVERY_SCAFFOLD_MARKERS = (
    "AUTO-DIAGNOSTIC: DIAGNOSTIC ERROR",
    "SYSTEM ALERT: Your previous turn entered a self-repeating",
)


def _is_recovery_scaffold(traj: Trajectory) -> bool:
    """True when the trajectory's user prompt is agent self-recovery
    scaffolding (see ``_RECOVERY_SCAFFOLD_MARKERS``) rather than a
    substantive task. System banners always lead the user message, so a
    leading-prefix match is sufficient (mirrors autobiographical
    ``_template_marker_for``)."""
    req = getattr(traj, "user_request", "")
    if not isinstance(req, str):
        return False
    head = req.lstrip()[:120]
    return any(head.startswith(m) for m in _RECOVERY_SCAFFOLD_MARKERS)


CritiqueCallable = Callable[[str], Union[str, Awaitable[str]]]
# Verifies a revised plan against the failed trajectory. Returns
# ``(verified, note)``: ``verified`` upgrades the reflection trajectory's
# outcome to PASSED (and tags the lesson verified); ``note`` is a short
# human-readable reason. May be sync or async. Injected (like
# ``critique_fn``) so the backend — an LLM soundness judge, or a sandbox
# re-run for self-play-derived reflections — stays out of the driver.
VerifyCallable = Callable[
    [Trajectory, List[str]], Union[Tuple[bool, str], Awaitable[Tuple[bool, str]]]
]


@dataclass
class ReflectionOutcome:
    """One failed-trajectory reflection result. Produced per-source."""

    source_trajectory_id: str
    reflected_trajectory: Optional[Trajectory] = None
    diagnosis: str = ""
    revised_plan: List[str] = field(default_factory=list)
    error: str = ""
    # None = not verified (no verify_fn, or verification errored). True/
    # False = the verify_fn's verdict on whether the revised plan
    # addresses the diagnosed failure.
    plan_verified: Optional[bool] = None
    verify_note: str = ""

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
        verify_fn: Optional[VerifyCallable] = None,
        verify_timeout_s: float = 60.0,
    ):
        self.critique_fn = critique_fn
        # Optional plan-verification backend (proposal #6: ground the
        # one learning path that previously had zero correctness
        # grounding). When set, each revised plan is checked and the
        # reflection's outcome is upgraded to PASSED only on a verified
        # verdict; otherwise it stays UNKNOWN (current behaviour).
        self.verify_fn = verify_fn
        self.verify_timeout_s = float(verify_timeout_s)
        self.per_call_timeout_s = float(per_call_timeout_s)
        self.max_failures = int(max_failures)
        self.model = model
        self.session_id_prefix = session_id_prefix

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
            # Candidates were vetted at collection time; a concurrent
            # reflect_one — e.g. a user-correction scheduled via
            # create_task, sharing this set — may have claimed and
            # reflected this one while an earlier candidate's critique
            # was awaited. Re-check RIGHT before claiming (check+add with
            # no await between them is atomic on the event loop).
            if traj.id in reflected_set:
                report.skipped_duplicate += 1
                continue
            # Mark BEFORE awaiting (mirrors reflect_one at line ~270).
            # During this await window a concurrent reflect_one(traj)
            # must see the id already claimed, or the same trajectory
            # gets reflected twice (duplicate/contradictory SFT data).
            reflected_set.add(traj.id)
            outcome = await self._reflect_one(traj)
            report.outcomes.append(outcome)
            if outcome.ok:
                report.reflected_ok += 1
                if sink is not None and outcome.reflected_trajectory is not None:
                    try:
                        sink(outcome.reflected_trajectory)
                    except Exception as e:
                        logger.warning("reflection sink failed: %s", e)
            else:
                report.reflected_errors += 1
                # A transient failure (idle-LLM timeout, unparseable critique)
                # must NOT permanently claim the trajectory — un-mark it so a
                # later tick can retry. The id was held only to block a
                # concurrent reflect_one from double-reflecting DURING the
                # await. Safe: the membership check above guarantees this
                # claim was made by THIS iteration, so the discard can never
                # un-claim an id a concurrent path successfully reflected.
                reflected_set.discard(traj.id)

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
        # Same scaffolding gate as the batch path (_is_reflectable); this
        # public entrypoint bypasses it, so re-check here or the post-turn
        # scheduler would still save lessons from AUTO-DIAGNOSTIC turns.
        if _is_recovery_scaffold(traj):
            return ReflectionOutcome(
                source_trajectory_id=traj.id,
                error="recovery-scaffold: not reflected",
            )
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
        if out.ok:
            if sink is not None and out.reflected_trajectory is not None:
                try:
                    sink(out.reflected_trajectory)
                except Exception as e:
                    logger.warning("reflect_one sink failed: %s", e)
        else:
            # Transient failure → un-claim so a later tick / re-schedule can
            # retry (mirrors the batch path in run()). Harmless when
            # `already_reflected` was None (the local set is discarded anyway).
            reflected_set.discard(traj.id)
        return out

    def _is_reflectable(self, traj: Trajectory) -> bool:
        """Decide whether a trajectory belongs in the reflection batch:
        ``FAILED`` trajectories only. (The proposal-F low-novelty-pass
        opt-in was removed 2026-07-20 as dead-by-construction: nothing
        in the live tree writes ``extra["solution_novelty"]`` and the
        dream loop's isolated context carries no trajectory collector,
        so qualifying self-play passes could never reach this filter.)
        """
        # Never learn from the agent's own error-recovery turns — they
        # yield transient tooling trivia, not task knowledge.
        if _is_recovery_scaffold(traj):
            return False
        return traj.outcome == Outcome.FAILED.value

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

        # Plan verification (proposal #6). Reflection was previously the
        # ONE learning path with no correctness grounding — the revised
        # plan was written straight to SkillMemory, executed-or-not,
        # correct-or-not. When a verify_fn is wired, the plan is checked
        # against the failure; only a verified plan upgrades the
        # reflection trajectory to PASSED (and tags the lesson verified).
        # An unverified / errored check leaves outcome=UNKNOWN — same as
        # before, so this never regresses the un-wired path.
        if self.verify_fn is not None and plan:
            try:
                vcall = self.verify_fn(traj, plan)
                if inspect.isawaitable(vcall):
                    verified, note = await asyncio.wait_for(
                        vcall, timeout=self.verify_timeout_s
                    )
                else:
                    verified, note = vcall
                verified = bool(verified)
                out.plan_verified = verified
                out.verify_note = str(note or "")
                reflected.extra["plan_verified"] = verified
                reflected.extra["plan_verify_note"] = out.verify_note
                if verified:
                    reflected.outcome = Outcome.PASSED.value
                    reflected.final_response += (
                        f"\n\nPLAN VERIFIED: {out.verify_note}"
                    )
                else:
                    reflected.final_response += (
                        f"\n\nPLAN UNVERIFIED: {out.verify_note}"
                    )
            except asyncio.TimeoutError:
                logger.debug("reflection plan verify timed out")
                reflected.extra["plan_verify_note"] = "verify timed out"
            except Exception as e:
                logger.debug("reflection plan verify failed: %s", e)
                reflected.extra["plan_verify_note"] = f"verify error: {type(e).__name__}"

        out.reflected_trajectory = reflected
        return out
