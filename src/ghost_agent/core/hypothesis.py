# src/ghost_agent/core/hypothesis.py
"""Parallel Hypothesis Testing.

When the agent is debugging or investigating, this module generates
multiple hypotheses about the root cause and tests them in parallel
rather than sequentially — collapsing what would be 6-8 serial turns
into 2-3 turns through elimination.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("GhostAgent")


@dataclass
class Hypothesis:
    description: str
    test_action: str     # Tool call or code to run that would confirm/refute
    test_tool: str = ""  # Which tool to use (execute, file_system, etc.)
    result: str = ""
    consistent: Optional[bool] = None  # None=untested, True/False after test
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "test_action": self.test_action,
            "test_tool": self.test_tool,
            "result": self.result,
            "consistent": self.consistent,
            "confidence": self.confidence,
        }


# ── Prompts ──────────────────────────────────────────────────────────

_GENERATE_HYPOTHESES_PROMPT = """You are a senior debugging expert. Given a PROBLEM DESCRIPTION and any available CONTEXT, generate 3-5 hypotheses about the root cause. For each hypothesis, specify a minimal test that would confirm or refute it.

PROBLEM:
{problem}

CONTEXT:
{context}

ERROR OUTPUT (if any):
{error_output}

Return ONLY a JSON object:
{{
  "hypotheses": [
    {{
      "description": "What might be wrong",
      "test_action": "A minimal Python script, shell command, or file read that would confirm or refute this",
      "test_tool": "execute or file_system",
      "confidence": 0.0-1.0
    }}
  ]
}}"""

_EVALUATE_RESULTS_PROMPT = """You are a debugging expert analyzing test results. Given the ORIGINAL PROBLEM and the HYPOTHESIS TEST RESULTS, determine which hypotheses are still consistent with the evidence and what the most likely root cause is.

PROBLEM:
{problem}

TEST RESULTS:
{results}

Return ONLY a JSON object:
{{
  "surviving_hypotheses": ["indices of hypotheses still consistent with evidence"],
  "most_likely": "index of the most likely root cause",
  "conclusion": "one sentence summary of findings",
  "next_step": "recommended next action"
}}"""


class HypothesisTester:
    """Generates and tests debugging hypotheses in parallel."""

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client

    async def generate_hypotheses(self, problem: str,
                                  context: str = "",
                                  error_output: str = "") -> List[Hypothesis]:
        """Use the LLM to generate hypotheses about a problem."""
        if not self.llm_client:
            return []

        prompt = _GENERATE_HYPOTHESES_PROMPT.format(
            problem=problem[:2000],
            context=context[:2000],
            error_output=error_output[:2000],
        )

        try:
            result = await self.llm_client.chat_completion({
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1024,
                "stream": False,
            })
            text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            data = self._parse_json(text)
        except Exception as exc:
            logger.warning("Hypothesis generation failed: %s", exc)
            return []

        hypotheses = []
        for h in data.get("hypotheses", []):
            hypotheses.append(Hypothesis(
                description=h.get("description", ""),
                test_action=h.get("test_action", ""),
                test_tool=h.get("test_tool", "execute"),
                confidence=float(h.get("confidence", 0.5)),
            ))
        return hypotheses

    async def test_hypotheses_parallel(
        self,
        hypotheses: List[Hypothesis],
        executor: Callable,
    ) -> List[Hypothesis]:
        """Test all hypotheses in parallel using the provided executor.

        Parameters
        ----------
        hypotheses : list of Hypothesis objects with test_action populated
        executor : async callable(tool_name, action_str) -> result_str
            Typically wraps the sandbox execution or file_system tool.

        Returns the hypotheses with results and consistency updated.
        """
        if not hypotheses:
            return hypotheses

        async def _run_test(h: Hypothesis) -> Hypothesis:
            try:
                result = await executor(h.test_tool or "execute", h.test_action)
                h.result = str(result)[:4000]
                # Simple heuristic: if test ran without error, hypothesis is consistent
                result_lower = h.result.lower()
                has_error = any(kw in result_lower for kw in (
                    "error", "exception", "traceback", "not found",
                    "permission denied", "no such file",
                ))
                # If the test was designed to trigger an error on the hypothesis,
                # we can't auto-classify — leave it for evaluation
                h.consistent = not has_error
            except Exception as exc:
                h.result = f"Test failed: {exc}"
                h.consistent = False
            return h

        await asyncio.gather(*[_run_test(h) for h in hypotheses])
        return hypotheses

    async def evaluate_results(self, problem: str,
                               hypotheses: List[Hypothesis]) -> Dict[str, Any]:
        """Use LLM to evaluate which hypotheses survived testing."""
        if not self.llm_client or not hypotheses:
            return {"surviving_hypotheses": [], "conclusion": "No hypotheses to evaluate"}

        results_text = "\n".join(
            f"H{i}: {h.description}\n"
            f"  Test: {h.test_action[:200]}\n"
            f"  Result: {h.result[:500]}\n"
            f"  Consistent: {h.consistent}"
            for i, h in enumerate(hypotheses)
        )

        prompt = _EVALUATE_RESULTS_PROMPT.format(
            problem=problem[:2000],
            results=results_text[:4000],
        )

        try:
            result = await self.llm_client.chat_completion({
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 512,
                "stream": False,
            })
            text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return self._parse_json(text)
        except Exception as exc:
            logger.warning("Hypothesis evaluation failed: %s", exc)
            # Fallback: return hypotheses that passed their tests
            surviving = [i for i, h in enumerate(hypotheses) if h.consistent]
            return {
                "surviving_hypotheses": surviving,
                "conclusion": f"{len(surviving)} of {len(hypotheses)} hypotheses are consistent with evidence",
            }

    def get_surviving(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Return hypotheses that are still consistent after testing."""
        return [h for h in hypotheses if h.consistent is True]

    @staticmethod
    def _parse_json(text: str) -> dict:
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}
