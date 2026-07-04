"""Tests for the execution-grounded behavioral eval (the discriminating suite).

The `capability` baseline is single-turn, zero-tool text Q&A (`mean_tool_calls:
0.0`, `pass_rate: 1.0`) — it stayed green through five live tool-path bugs. A
BehavioralTask instead drives the agent and verifies the REAL side-effect. These
tests exercise the contract with a fake context (no live agent):
  - the runner drives → runs the task's grounded verify → returns a verdict;
  - a BehavioralTask under a non-behavioral runner scores FAIL (unverified),
    never a soft green;
  - real tool metrics flow through to the TaskResult.
"""
from __future__ import annotations

import pytest

from ghost_agent.eval import (
    BehavioralTask,
    EvalSuite,
    agent_behavioral_runner,
    load_behavioral_suite,
)


class FakeCtx:
    """Stand-in for EvalContext: canned agent replies + sandbox/DB/metrics."""

    def __init__(self, reply="done", files=None, db=None, metrics=None, ask_map=None):
        self._reply = reply
        self._files = files or {}
        self._db = db
        self._metrics = metrics or {"steps": 2, "tool_calls": 1, "tool_errors": 0}
        self._ask_map = ask_map or {}
        self.last_metrics = {}
        self.asks = []

    async def ask(self, prompt):
        self.asks.append(prompt)
        for k, v in self._ask_map.items():
            if k in prompt:
                return v
        return self._reply

    def sandbox_read(self, rel):
        return self._files.get(rel)

    async def db_scalar(self, q):
        return self._db

    def trajectory_metrics(self, prompt):
        return dict(self._metrics)


def _task(verify, prompt="do the thing"):
    return BehavioralTask(task_id="beh:test", category="behavioral", prompt=prompt, verify=verify)


# ── validate contract ──────────────────────────────────────────────────

class TestValidateContract:
    def test_dict_verdicts(self):
        t = _task(None)
        assert t.validate({"passed": True}) == (True, "")
        assert t.validate({"passed": False, "failure_reason": "nope"}) == (False, "nope")

    def test_missing_verdict_is_fail(self):
        ok, reason = _task(None).validate({"output": "I did it!"})
        assert ok is False and "no verdict" in reason

    def test_plain_string_is_unverified_fail(self):
        # This is the key anti-false-green: a string (http/stub runner) can't verify.
        ok, reason = _task(None).validate("I totally wrote the file, trust me.")
        assert ok is False and "behavioral runner" in reason


# ── runner drives + grounds ────────────────────────────────────────────

class TestBehavioralRunner:
    async def test_grounded_pass_on_real_sideeffect(self):
        async def verify(out, ctx):
            return (ctx.sandbox_read("x.txt") == "OK", "file wrong/missing")
        res = await agent_behavioral_runner()(_task(verify), FakeCtx(files={"x.txt": "OK"}))
        assert res["passed"] is True
        assert res["tool_calls"] == 1  # real metrics flow through

    async def test_grounded_fail_when_sideeffect_absent(self):
        async def verify(out, ctx):
            return (ctx.sandbox_read("x.txt") == "OK", "file wrong/missing")
        res = await agent_behavioral_runner()(_task(verify), FakeCtx(files={}))
        assert res["passed"] is False
        assert res["failure_reason"] == "file wrong/missing"

    async def test_memory_roundtrip_makes_second_call(self):
        async def verify(out, ctx):
            recall = await ctx.ask("what is the code word?")
            return ("ZEPHYR-7719" in recall, "not recalled")
        ctx = FakeCtx(ask_map={"code word": "the word is ZEPHYR-7719"})
        res = await agent_behavioral_runner()(_task(verify), ctx)
        assert res["passed"] is True
        assert len(ctx.asks) == 2  # drive + recall

    async def test_verify_exception_fails_task_not_suite(self):
        async def verify(out, ctx):
            raise RuntimeError("boom")
        res = await agent_behavioral_runner()(_task(verify), FakeCtx())
        assert res["passed"] is False
        assert "verify raised" in res["failure_reason"]


# ── full suite integration ─────────────────────────────────────────────

class TestSuiteIntegration:
    async def test_verdict_and_metrics_flow_to_result(self):
        async def verify(out, ctx):
            return (True, "")
        suite = EvalSuite("b", [_task(verify)])
        ctx = FakeCtx(metrics={"steps": 3, "tool_calls": 2, "tool_errors": 1})
        result = await suite.run(runner=agent_behavioral_runner(), ctx=ctx, per_task_timeout_s=10)
        r = result.results[0]
        assert r.passed is True
        assert r.tool_calls == 2 and r.tool_errors == 1 and r.steps == 3

    async def test_behavioral_task_under_non_behavioral_runner_is_unverified(self):
        # An http-style runner returns {"output": "..."} with no verdict → the
        # BehavioralTask must FAIL as unverified, never pass on the text.
        async def http_like(task, ctx):
            return {"output": "Sure, I wrote the file and it's perfect!"}
        suite = EvalSuite("b", [_task(lambda o, c: (True, ""))])
        result = await suite.run(runner=http_like, ctx=None, per_task_timeout_s=10)
        assert result.results[0].passed is False
        assert "verdict" in result.results[0].failure_reason


# ── the shipped task set ────────────────────────────────────────────────

class TestBehavioralSuite:
    def test_suite_shape(self):
        tasks = load_behavioral_suite()
        assert tasks and all(t.category == "behavioral" for t in tasks)
        assert all(callable(t.verify) for t in tasks)
        ids = {t.task_id for t in tasks}
        for expected in ("beh:code_exec_output", "beh:file_roundtrip",
                         "beh:memory_roundtrip", "beh:tool_chain_compute_save",
                         "beh:db_grounded_query"):
            assert expected in ids

    def test_every_task_exercises_a_tool_by_design(self):
        # The whole point: unlike `capability`, every behavioral prompt asks for
        # a real side-effect (file / memory / DB), so a 0-tool-call answer fails.
        prompts = " ".join(t.prompt.lower() for t in load_behavioral_suite())
        for surface in ("python", "file", "remember", "sql", "workspace"):
            assert surface in prompts
