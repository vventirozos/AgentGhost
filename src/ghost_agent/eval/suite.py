"""EvalSuite runner.

The harness wires tasks → runner → TaskResult → SuiteResult. The runner
is a callable injected by the caller, which is what keeps the harness
testable in pure Python (inject a mock runner) AND usable in production
(inject an agent-backed runner that actually drives Ghost).

Runner contract:
    runner(task: EvalTask, ctx: Any) -> dict | str | awaitable of either

    If the runner returns a dict, the keys we understand are:
        output        (str|dict)       — fed to task.validate
        steps         (int)            — planning/chat iterations
        tool_calls    (int)            — tool invocations
        tool_errors   (int)            — errored tool results
        tokens_used   (int)            — model tokens consumed
        duration_s    (float)          — wall-clock of the task run
        failure_reason (str)           — override (takes precedence on failure)
        extra         (dict)           — free-form additional metrics

    If the runner returns a plain string, it's treated as `output` only
    (all other metrics default to 0).

The runner MUST be deterministic with respect to the task or the eval
ceases to be useful as a baseline — the harness doesn't enforce that,
but `freeze_baseline` exists precisely to snapshot the day you claim
determinism.
"""

from __future__ import annotations

import asyncio
import datetime
import inspect
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from .metrics import TaskResult, SuiteResult, aggregate
from .tasks import EvalTask


# Runner may be sync or async; returns a string or a dict.
RunnerCallable = Callable[..., Union[str, Dict[str, Any], Awaitable[Any]]]


def _ghost_version() -> str:
    """Best-effort package version. Never fatal."""
    try:
        from .. import __version__  # type: ignore[attr-defined]
        return str(__version__)
    except Exception:
        return "unknown"


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class EvalSuite:
    """Runs a list of EvalTasks against an injected runner and collects
    a SuiteResult.

    Usage:
        suite = EvalSuite("baseline", tasks)
        result = await suite.run(runner=my_runner, ctx=my_ctx)
    """

    def __init__(self, suite_name: str, tasks: List[EvalTask]):
        self.suite_name = suite_name
        self.tasks = list(tasks)

    async def run(
        self,
        runner: Optional[RunnerCallable] = None,
        ctx: Any = None,
        *,
        per_task_timeout_s: float = 60.0,
        stop_on_error: bool = False,
        on_task_end: Optional[Callable[[TaskResult], None]] = None,
    ) -> SuiteResult:
        """Run all tasks sequentially.

        `per_task_timeout_s` is a defense against a rogue runner hanging
        the suite. Timed-out tasks are recorded as failures with reason
        'timeout' — the suite continues.
        """
        results: List[TaskResult] = []
        for task in self.tasks:
            result = await self._run_one(task, runner, ctx, per_task_timeout_s)
            results.append(result)
            if on_task_end is not None:
                try:
                    on_task_end(result)
                except Exception:
                    # Reporter hooks must never break the suite.
                    pass
            if stop_on_error and not result.passed:
                break

        summary = aggregate(results)
        return SuiteResult(
            suite_name=self.suite_name,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            ghost_version=_ghost_version(),
            results=results,
            summary=summary,
        )

    async def _run_one(
        self,
        task: EvalTask,
        runner: Optional[RunnerCallable],
        ctx: Any,
        per_task_timeout_s: float,
    ) -> TaskResult:
        start = time.monotonic()
        output: Any = ""
        metrics: Dict[str, Any] = {}
        failure_reason = ""

        # Regression probes don't need a runner — the probe's own
        # validator is the entire test. Short-circuit.
        if task.category == "regression":
            ok, reason = task.validate(None, ctx)
            duration = time.monotonic() - start
            return TaskResult(
                task_id=task.task_id,
                category=task.category,
                cluster=task.cluster,
                tier=task.tier,
                passed=ok,
                duration_s=duration,
                failure_reason="" if ok else reason,
            )

        # Everything else needs a runner.
        if runner is None:
            duration = time.monotonic() - start
            return TaskResult(
                task_id=task.task_id,
                category=task.category,
                cluster=task.cluster,
                tier=task.tier,
                passed=False,
                duration_s=duration,
                failure_reason="no runner provided",
            )

        try:
            coro_or_val = runner(task, ctx)
            result_val = await asyncio.wait_for(
                _maybe_await(coro_or_val),
                timeout=per_task_timeout_s,
            )
        except asyncio.TimeoutError:
            duration = time.monotonic() - start
            return TaskResult(
                task_id=task.task_id,
                category=task.category,
                cluster=task.cluster,
                tier=task.tier,
                passed=False,
                duration_s=duration,
                failure_reason=f"timeout after {per_task_timeout_s:.1f}s",
            )
        except Exception as e:
            duration = time.monotonic() - start
            return TaskResult(
                task_id=task.task_id,
                category=task.category,
                cluster=task.cluster,
                tier=task.tier,
                passed=False,
                duration_s=duration,
                failure_reason=f"runner raised: {type(e).__name__}: {e}",
            )

        # Unpack runner return shape.
        if isinstance(result_val, dict):
            output = result_val.get("output", "")
            metrics = result_val
        else:
            output = result_val
            metrics = {"output": output}

        try:
            passed, reason = task.validate(output, ctx)
        except Exception as e:
            passed, reason = False, f"validator raised: {type(e).__name__}: {e}"

        if not passed:
            # Runner override takes precedence when present.
            failure_reason = str(metrics.get("failure_reason") or reason)

        duration = float(metrics.get("duration_s") or (time.monotonic() - start))
        return TaskResult(
            task_id=task.task_id,
            category=task.category,
            cluster=task.cluster,
            tier=task.tier,
            passed=passed,
            duration_s=duration,
            steps=int(metrics.get("steps") or 0),
            tool_calls=int(metrics.get("tool_calls") or 0),
            tool_errors=int(metrics.get("tool_errors") or 0),
            tokens_used=int(metrics.get("tokens_used") or 0),
            final_output=str(output) if not isinstance(output, str) else output,
            failure_reason=failure_reason,
            extra=dict(metrics.get("extra") or {}),
        )
