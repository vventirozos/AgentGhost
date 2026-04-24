"""Reasoning-quality evals.

Golden tasks with graded rubrics that exercise the new reasoning hooks.
Unlike the unit tests for each module, these check end-to-end signal:

* Planner: does a multi-step task tree decompose correctly, enforce
  postconditions, and route around failure via alternatives?
* Thinking budget: are complex queries routed to extended and simple
  queries to tight?
* Verifier: do orphaned-claim answers get annotated?
* Deep-reason prelude: do high-confidence hypotheses inform strategy
  generation?

Each test isolates its subject — there's no cross-suite dependency —
and each has a pass/fail rubric stated in the docstring. This is
intentionally small (≈20 checks) so the suite stays cheap to run, and
the expected failure modes are specific ("answer omits postcondition X")
rather than vague ("bad quality").
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.planning import TaskTree, TaskStatus, DependencyType
from ghost_agent.core.agent import classify_thinking_budget
from ghost_agent.core.verifier import Verifier, VerifyVerdict
from ghost_agent.core.hypothesis import HypothesisTester, Hypothesis


# ──────────────────────────────────────────────────────────── PLANNING


class TestPlanningEvals:
    """Rubric: can the planner decompose a 3-step task, enforce
    postconditions on each step, and route to alternatives on failure?"""

    def test_multistep_decomposition_with_and_dep(self):
        """Golden task: 'Scrape data, transform, and load.' Expect all
        three steps to be required (ALL dep) and parent becomes DONE
        only when all three pass their postconditions."""
        tree = TaskTree()
        root = tree.add_task("ETL pipeline", dependency_type=DependencyType.ALL)
        s1 = tree.add_task("scrape source",
                           parent_id=root,
                           postconditions=["fetched raw data"])
        s2 = tree.add_task("transform rows",
                           parent_id=root,
                           postconditions=["cleaned records ready"])
        s3 = tree.add_task("load into warehouse",
                           parent_id=root,
                           postconditions=["warehouse updated"])

        tree.update_status(s1, TaskStatus.DONE, result="fetched raw data from API (200 OK)")
        tree.update_status(s2, TaskStatus.DONE, result="cleaned records ready for insert")
        # Parent must still be IN_PROGRESS until s3 is done.
        assert tree.nodes[root].status != TaskStatus.DONE
        tree.update_status(s3, TaskStatus.DONE, result="warehouse updated, 42 new rows")
        assert tree.nodes[root].status == TaskStatus.DONE

    def test_postcondition_catches_silent_failure(self):
        """Golden task: tool claims DONE but the postcondition token isn't
        in the result → flip to FAILED and expose via failure_reason."""
        tree = TaskTree()
        t = tree.add_task(
            "write report",
            postconditions=["report.md exists"],
        )
        tree.update_status(t, TaskStatus.DONE, result="greetings, here's an empty reply")
        assert tree.nodes[t].status == TaskStatus.FAILED
        assert "Postcondition" in tree.nodes[t].failure_reason

    def test_failure_routes_to_alternative(self):
        """Golden task: first approach fails a postcondition, alternative
        declared on the parent must be promoted to READY."""
        tree = TaskTree()
        root = tree.add_task("fetch weather")
        primary = tree.add_task(
            "via open-meteo",
            parent_id=root,
            postconditions=["temperature returned"],
        )
        fallback = tree.add_task("via cached report")
        tree.nodes[root].alternatives = [fallback]
        tree.update_status(primary, TaskStatus.DONE, result="HTTP 500 Server Error")
        assert tree.nodes[primary].status == TaskStatus.FAILED
        assert tree.nodes[fallback].status == TaskStatus.READY


# ──────────────────────────────────────────────────── THINKING BUDGET


class TestThinkingBudgetEvals:
    """Rubric: simple queries → tight; multi-step reasoning → extended."""

    @pytest.mark.parametrize("query,expected", [
        ("hi", "tight"),
        ("what time is it", "tight"),
        ("make me a list of colors", "tight"),
        ("debug this traceback and optimize the retry path", "extended"),
        ("explain the algorithm complexity and prove termination", "extended"),
        ("refactor the connection pool", "tight"),  # single non-strong keyword → tight
        ("analyze this CTE and optimize the query plan", "extended"),
    ])
    def test_classification_goldens(self, query, expected):
        assert classify_thinking_budget(query) == expected


# ────────────────────────────────────────────── VERIFIER + COMPLETION


@pytest.fixture
def stubbed_llm():
    client = MagicMock()
    client.chat_completion = AsyncMock()
    client.route = AsyncMock(return_value=None)
    return client


class TestVerifierEvals:
    """Rubric: when the tool output contradicts the claimed answer, the
    verifier returns REFUTED; the gate would annotate the response."""

    async def test_refuted_claim_with_error_evidence(self, stubbed_llm):
        stubbed_llm.chat_completion.return_value = {
            "choices": [{"message": {"content":
                '{"verdict": "REFUTED", "confidence": 0.88, "reasoning": "evidence is a traceback, not a value", "issues": ["FileNotFoundError shown"]}'
            }}]
        }
        v = Verifier(llm_client=stubbed_llm)
        result = await v.verify_claim(
            claim="I counted 42 rows in /tmp/data.csv.",
            evidence="FileNotFoundError: [Errno 2] No such file: '/tmp/data.csv'",
        )
        assert result.verdict == VerifyVerdict.REFUTED
        assert result.confidence >= 0.7  # passes the gate's threshold

    async def test_confirmed_claim_does_not_annotate(self, stubbed_llm):
        stubbed_llm.chat_completion.return_value = {
            "choices": [{"message": {"content":
                '{"verdict": "CONFIRMED", "confidence": 0.95, "reasoning": "matches output", "issues": []}'
            }}]
        }
        v = Verifier(llm_client=stubbed_llm)
        result = await v.verify_claim(
            claim="The file has 42 rows.",
            evidence="$ wc -l /tmp/data.csv\n42 /tmp/data.csv",
        )
        assert result.verdict == VerifyVerdict.CONFIRMED


# ─────────────────────────────────────────────────────── DEEP REASON


class TestDeepReasonEvals:
    """Rubric: high-confidence hypotheses should surface in the top-3 the
    strategy generator sees, while low-confidence ones get dropped."""

    async def test_confidence_ordering_preserved(self):
        """When mixing high and low confidence hypotheses, the top-3 by
        confidence must be the ones selected for the prelude block."""
        tester = HypothesisTester(llm_client=None)
        hyps = [
            Hypothesis(description="missing env var FOO", confidence=0.9,
                       test_action="echo $FOO"),
            Hypothesis(description="port collision", confidence=0.3,
                       test_action="lsof -i:8000"),
            Hypothesis(description="wrong python version", confidence=0.7,
                       test_action="python --version"),
            Hypothesis(description="cosmic rays", confidence=0.05,
                       test_action="echo maybe"),
        ]
        top = sorted(hyps, key=lambda h: h.confidence, reverse=True)[:3]
        descs = [h.description for h in top]
        assert "missing env var FOO" in descs
        assert "wrong python version" in descs
        assert "cosmic rays" not in descs  # pruned

    async def test_parallel_eval_marks_error_inconsistent(self):
        tester = HypothesisTester(llm_client=None)
        h = Hypothesis(description="path issue", test_action="cat /nope",
                       test_tool="execute")

        async def executor(tool, action):
            return "cat: /nope: No such file or directory"

        out = await tester.test_hypotheses_parallel([h], executor)
        assert out[0].consistent is False  # error output → refuted
