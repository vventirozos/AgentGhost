"""Deep-reason wiring: MCTS + HypothesisTester gated behind --deep-reason.

When the flag is off, ``context.mcts_reasoner`` / ``context.hypothesis_tester``
must be None — keeping the worker-pool cost bounded for the default path.
When on, they're instantiated and callable. The System 3 pivot uses the
hypothesis tester as a prelude if present, producing targeted strategies
rather than generic ones.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

from ghost_agent.core.mcts import MCTSReasoner, ActionCandidate
from ghost_agent.core.hypothesis import HypothesisTester, Hypothesis


@pytest.fixture
def fake_llm():
    client = MagicMock()
    client.chat_completion = AsyncMock()
    client.route = AsyncMock()
    return client


class TestMCTSReasonerContract:
    """Basic self-contained contract so future callers know what they get."""

    async def test_expand_returns_empty_without_llm(self):
        r = MCTSReasoner(llm_client=None)
        actions = await r._expand("task", "state", ["execute"], "context")
        assert actions == []

    async def test_select_best_returns_none_when_no_candidates(self, fake_llm):
        fake_llm.chat_completion.return_value = {
            "choices": [{"message": {"content": '{"candidates": []}'}}]
        }
        r = MCTSReasoner(llm_client=fake_llm)
        assert await r.select_best_action("t", "s", ["execute"]) is None

    async def test_backtrack_stack_preserves_alternatives(self, fake_llm):
        r = MCTSReasoner(llm_client=fake_llm)
        r._backtrack_stack = [[
            ActionCandidate(description="alt1", score=0.6),
            ActionCandidate(description="alt2", score=0.4),
        ]]
        assert r.has_alternatives()
        first = await r.backtrack()
        assert first.description == "alt1"
        second = await r.backtrack()
        assert second.description == "alt2"
        assert await r.backtrack() is None


class TestHypothesisTesterContract:
    async def test_generate_returns_empty_without_llm(self):
        t = HypothesisTester(llm_client=None)
        assert await t.generate_hypotheses("broken") == []

    async def test_parallel_test_classifies_consistency(self):
        t = HypothesisTester(llm_client=None)
        h1 = Hypothesis(description="missing file", test_action="ls foo",
                        test_tool="execute")
        h2 = Hypothesis(description="exists", test_action="ls bar",
                        test_tool="execute")

        async def executor(tool, action):
            # "foo" → error path, "bar" → clean
            return "No such file or directory" if "foo" in action else "bar found"

        tested = await t.test_hypotheses_parallel([h1, h2], executor)
        assert tested[0].consistent is False  # error → inconsistent
        assert tested[1].consistent is True


class TestSystem3HypothesisPrelude:
    """When context.hypothesis_tester is set, the System 3 pivot should
    prepend a hypotheses block to its strategy-generation prompt."""

    async def test_pivot_without_tester_runs_unchanged(self, fake_llm):
        """No hypothesis_tester → no prelude; behaviour identical to before."""
        from ghost_agent.core.agent import GhostAgent, GhostContext
        import argparse, tempfile
        from pathlib import Path

        args = argparse.Namespace(max_context=4096, smart_memory=0.0,
                                  anonymous=True, model="test", verbose=False,
                                  no_memory=True)
        with tempfile.TemporaryDirectory() as td:
            sandbox_dir = Path(td)
            ctx = GhostContext(args, sandbox_dir, sandbox_dir, "socks5://none")
            ctx.llm_client = fake_llm
            ctx.hypothesis_tester = None
            fake_llm.chat_completion.side_effect = [
                {"choices": [{"message": {"content": '{"strategies": [{"id":"A","description":"x","steps":[]}]}'}}]},
                {"choices": [{"message": {"content": '{"winning_id":"A","justification":"ok"}'}}]},
            ]
            agent = GhostAgent(ctx)
            result = await agent._run_system_3_pivot(
                "task", "error", "sandbox", "test-model",
            )
            assert result.get("winning_id") == "A"
            # The generator-side call must NOT contain the hypothesis header.
            gen_call_msg = fake_llm.chat_completion.call_args_list[0][0][0]["messages"][1]["content"]
            assert "CANDIDATE ROOT CAUSES" not in gen_call_msg

    async def test_pivot_with_tester_injects_hypotheses(self, fake_llm):
        from ghost_agent.core.agent import GhostAgent, GhostContext
        import argparse, tempfile
        from pathlib import Path

        args = argparse.Namespace(max_context=4096, smart_memory=0.0,
                                  anonymous=True, model="test", verbose=False,
                                  no_memory=True)
        with tempfile.TemporaryDirectory() as td:
            sandbox_dir = Path(td)
            ctx = GhostContext(args, sandbox_dir, sandbox_dir, "socks5://none")
            ctx.llm_client = fake_llm

            tester = HypothesisTester(llm_client=None)

            async def _gen(problem, context="", error_output=""):
                return [
                    Hypothesis(description="missing env var", test_action="x",
                               confidence=0.8),
                ]
            tester.generate_hypotheses = _gen
            ctx.hypothesis_tester = tester

            fake_llm.chat_completion.side_effect = [
                {"choices": [{"message": {"content": '{"strategies": [{"id":"A","description":"x","steps":[]}]}'}}]},
                {"choices": [{"message": {"content": '{"winning_id":"A","justification":"ok"}'}}]},
            ]
            agent = GhostAgent(ctx)
            await agent._run_system_3_pivot(
                "task", "error", "sandbox", "test-model",
            )
            gen_call_msg = fake_llm.chat_completion.call_args_list[0][0][0]["messages"][1]["content"]
            assert "CANDIDATE ROOT CAUSES" in gen_call_msg
            assert "missing env var" in gen_call_msg
