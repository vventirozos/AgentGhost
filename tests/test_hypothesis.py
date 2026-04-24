"""Tests for the Parallel Hypothesis Testing module."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from ghost_agent.core.hypothesis import HypothesisTester, Hypothesis


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.chat_completion = AsyncMock()
    return client


@pytest.fixture
def tester(mock_llm_client):
    return HypothesisTester(llm_client=mock_llm_client)


class TestHypothesis:
    def test_to_dict(self):
        h = Hypothesis(
            description="Missing import",
            test_action="python -c 'import foo'",
            test_tool="execute",
            consistent=True,
            confidence=0.8,
        )
        d = h.to_dict()
        assert d["description"] == "Missing import"
        assert d["consistent"] is True

    def test_defaults(self):
        h = Hypothesis(description="test", test_action="test")
        assert h.consistent is None
        assert h.confidence == 0.5
        assert h.test_tool == ""


class TestHypothesisTester:
    async def test_generate_hypotheses(self, tester, mock_llm_client):
        mock_llm_client.chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "hypotheses": [
                    {"description": "Missing module", "test_action": "pip list | grep foo",
                     "test_tool": "execute", "confidence": 0.7},
                    {"description": "Wrong path", "test_action": "ls /data",
                     "test_tool": "execute", "confidence": 0.5},
                ],
            })}}],
        }
        hypotheses = await tester.generate_hypotheses(
            problem="ImportError: No module named 'foo'",
            error_output="Traceback...",
        )
        assert len(hypotheses) == 2
        assert hypotheses[0].description == "Missing module"
        assert hypotheses[0].confidence == 0.7

    async def test_generate_hypotheses_no_llm(self):
        t = HypothesisTester(llm_client=None)
        result = await t.generate_hypotheses("test")
        assert result == []

    async def test_generate_hypotheses_llm_failure(self, tester, mock_llm_client):
        mock_llm_client.chat_completion.side_effect = Exception("timeout")
        result = await tester.generate_hypotheses("test")
        assert result == []

    async def test_test_hypotheses_parallel(self, tester):
        hypotheses = [
            Hypothesis(description="H1", test_action="test1", test_tool="execute"),
            Hypothesis(description="H2", test_action="test2", test_tool="execute"),
        ]

        async def mock_executor(tool_name, action):
            if "1" in action:
                return "Success: found the issue"
            else:
                return "error: Traceback something failed"

        result = await tester.test_hypotheses_parallel(hypotheses, mock_executor)
        assert len(result) == 2
        assert result[0].consistent is True
        assert result[1].consistent is False

    async def test_test_hypotheses_executor_exception(self, tester):
        hypotheses = [
            Hypothesis(description="H1", test_action="test", test_tool="execute"),
        ]

        async def failing_executor(tool_name, action):
            raise RuntimeError("sandbox down")

        result = await tester.test_hypotheses_parallel(hypotheses, failing_executor)
        assert result[0].consistent is False
        assert "sandbox down" in result[0].result

    async def test_test_hypotheses_empty(self, tester):
        async def noop(t, a):
            return ""
        result = await tester.test_hypotheses_parallel([], noop)
        assert result == []

    async def test_evaluate_results(self, tester, mock_llm_client):
        mock_llm_client.chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "surviving_hypotheses": [0],
                "most_likely": 0,
                "conclusion": "Missing module is the root cause",
                "next_step": "Install the module",
            })}}],
        }
        hypotheses = [
            Hypothesis(description="Missing module", test_action="test",
                       result="module not found", consistent=True),
            Hypothesis(description="Wrong path", test_action="test2",
                       result="path exists", consistent=False),
        ]
        result = await tester.evaluate_results("ImportError", hypotheses)
        assert result["most_likely"] == 0
        assert "Missing module" in result["conclusion"]

    async def test_evaluate_results_fallback_on_failure(self, tester, mock_llm_client):
        mock_llm_client.chat_completion.side_effect = Exception("down")
        hypotheses = [
            Hypothesis(description="H1", test_action="t", consistent=True),
            Hypothesis(description="H2", test_action="t", consistent=False),
        ]
        result = await tester.evaluate_results("problem", hypotheses)
        assert 0 in result["surviving_hypotheses"]
        assert 1 not in result["surviving_hypotheses"]

    def test_get_surviving(self, tester):
        hypotheses = [
            Hypothesis(description="H1", test_action="t", consistent=True),
            Hypothesis(description="H2", test_action="t", consistent=False),
            Hypothesis(description="H3", test_action="t", consistent=True),
            Hypothesis(description="H4", test_action="t", consistent=None),
        ]
        surviving = tester.get_surviving(hypotheses)
        assert len(surviving) == 2
        assert all(h.consistent for h in surviving)
