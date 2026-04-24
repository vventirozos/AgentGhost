"""Tests for ghost_agent.eval.suite EvalSuite runner."""

import asyncio

import pytest

from ghost_agent.eval.suite import EvalSuite
from ghost_agent.eval.tasks import (
    CuratedRequestTask, RegressionProbeTask, ChallengeTemplateTask,
)


async def _runner_echo(task, _ctx):
    return {"output": task.prompt, "steps": 3, "tool_calls": 2, "tokens_used": 50}


async def _runner_sync_str(task, _ctx):
    return "sync string output"


async def _runner_raises(task, _ctx):
    raise ValueError("runner failure")


async def _runner_hangs(task, _ctx):
    await asyncio.sleep(10)
    return "never"


async def test_suite_runs_tasks_and_aggregates():
    tasks = [
        CuratedRequestTask(task_id="c1", category="", prompt="hello world", validator=["hello"]),
        CuratedRequestTask(task_id="c2", category="", prompt="unrelated", validator=["42"]),
    ]
    suite = EvalSuite("test", tasks)
    result = await suite.run(runner=_runner_echo)
    assert result.summary["n"] == 2
    assert result.summary["pass_rate"] == 0.5
    assert result.results[0].passed
    assert not result.results[1].passed


async def test_suite_runner_plain_string():
    tasks = [CuratedRequestTask(
        task_id="c", category="", prompt="", validator=["sync"])]
    suite = EvalSuite("test", tasks)
    result = await suite.run(runner=_runner_sync_str)
    assert result.results[0].passed


async def test_suite_runner_exception_marks_failed():
    tasks = [CuratedRequestTask(task_id="c", category="", prompt="")]
    suite = EvalSuite("test", tasks)
    result = await suite.run(runner=_runner_raises)
    assert not result.results[0].passed
    assert "runner raised" in result.results[0].failure_reason
    assert "ValueError" in result.results[0].failure_reason


async def test_suite_runner_timeout_marks_failed():
    tasks = [CuratedRequestTask(task_id="c", category="", prompt="")]
    suite = EvalSuite("test", tasks)
    result = await suite.run(runner=_runner_hangs, per_task_timeout_s=0.05)
    assert not result.results[0].passed
    assert "timeout" in result.results[0].failure_reason


async def test_regression_probe_needs_no_runner():
    probe = RegressionProbeTask(
        task_id="p", category="", prompt="",
        validator=lambda _ctx: (True, ""),
    )
    suite = EvalSuite("probes", [probe])
    result = await suite.run()  # no runner!
    assert result.results[0].passed


async def test_regression_probe_failure_surfaces_reason():
    probe = RegressionProbeTask(
        task_id="p", category="", prompt="",
        validator=lambda _ctx: (False, "anchor missing"),
    )
    suite = EvalSuite("probes", [probe])
    result = await suite.run()
    assert not result.results[0].passed
    assert "anchor missing" in result.results[0].failure_reason


async def test_stop_on_error_halts_suite():
    tasks = [
        CuratedRequestTask(task_id=f"c{i}", category="", prompt="",
                           validator=["never-matches"])
        for i in range(5)
    ]
    suite = EvalSuite("test", tasks)
    result = await suite.run(runner=_runner_echo, stop_on_error=True)
    assert len(result.results) == 1
    assert not result.results[0].passed


async def test_on_task_end_hook_fires():
    seen = []
    probe = RegressionProbeTask(
        task_id="p", category="", prompt="",
        validator=lambda _c: (True, ""),
    )
    suite = EvalSuite("test", [probe])
    await suite.run(on_task_end=lambda r: seen.append(r.task_id))
    assert seen == ["p"]


async def test_on_task_end_hook_exception_doesnt_break_suite():
    def bad_hook(_r):
        raise RuntimeError("hook failure")
    probe = RegressionProbeTask(
        task_id="p", category="", prompt="",
        validator=lambda _c: (True, ""),
    )
    suite = EvalSuite("test", [probe])
    result = await suite.run(on_task_end=bad_hook)
    assert result.results[0].passed  # hook failing doesn't corrupt result


async def test_suite_captures_runner_metrics_into_result():
    async def runner(_t, _c):
        return {
            "output": "something",
            "steps": 7,
            "tool_calls": 4,
            "tool_errors": 1,
            "tokens_used": 500,
            "extra": {"router_confidence": 0.87},
        }
    tasks = [CuratedRequestTask(task_id="c", category="", prompt="")]
    suite = EvalSuite("test", tasks)
    result = await suite.run(runner=runner)
    r = result.results[0]
    assert r.steps == 7
    assert r.tool_calls == 4
    assert r.tool_errors == 1
    assert r.tokens_used == 500
    assert r.extra["router_confidence"] == 0.87


async def test_suite_missing_runner_fails_non_probe_tasks():
    tasks = [CuratedRequestTask(task_id="c", category="", prompt="")]
    suite = EvalSuite("test", tasks)
    result = await suite.run(runner=None)
    assert not result.results[0].passed
    assert "no runner" in result.results[0].failure_reason


async def test_baseline_freeze_and_compare(tmp_path):
    from ghost_agent.eval import freeze_baseline, load_baseline, compare_to_baseline

    async def runner_all_pass(_t, _c):
        return {"output": "ok keyword-matches"}

    tasks = [CuratedRequestTask(
        task_id="c", category="", prompt="", validator=["ok"]
    )]
    suite = EvalSuite("test", tasks)
    r1 = await suite.run(runner=runner_all_pass)

    path = tmp_path / "baseline.json"
    freeze_baseline(r1, path)

    r2 = await suite.run(runner=runner_all_pass)
    loaded = load_baseline(path)
    assert loaded.summary["pass_rate"] == 1.0
    diff = compare_to_baseline(path, r2)
    assert diff["pass_rate_delta"] == 0.0
    assert not diff["regressions"]
