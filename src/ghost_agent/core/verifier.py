# src/ghost_agent/core/verifier.py
"""Reflective Self-Evaluation Module.

Provides a Verifier that challenges the agent's own conclusions using
a separate LLM call (ideally on a worker node or at a different temperature)
before returning results to the user.

Three capabilities:
1. verify_claim     — Check whether a stated conclusion is supported by evidence.
2. verify_code_output — Check whether code output actually answers the user's question.
3. adversarial_probe — Generate edge cases / counterexamples for stress-testing.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GhostAgent")


class VerifyVerdict(str, Enum):
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class VerifyResult:
    verdict: VerifyVerdict
    confidence: float  # 0.0 – 1.0
    reasoning: str = ""
    issues: List[str] = field(default_factory=list)

    def passed(self) -> bool:
        return self.verdict == VerifyVerdict.CONFIRMED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "issues": self.issues,
        }


# ── Prompts ──────────────────────────────────────────────────────────

_VERIFY_CLAIM_PROMPT = """You are a rigorous fact-checker. Given a CLAIM and supporting EVIDENCE, decide whether the claim is fully supported.

CLAIM:
{claim}

EVIDENCE:
{evidence}

CONTEXT:
{context}

Respond ONLY with a JSON object:
{{
  "verdict": "CONFIRMED" | "REFUTED" | "UNCERTAIN",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence",
  "issues": ["list of specific problems, if any"]
}}"""

_VERIFY_CODE_PROMPT = """You are a code output auditor. Determine whether the CODE and its OUTPUT actually answer the user's INTENT.

USER INTENT:
{intent}

CODE:
{code}

OUTPUT:
{output}

Check for:
1. Does the output contain the information the user asked for?
2. Are the numbers/results plausible (no obvious off-by-one, wrong units, etc.)?
3. Are there silent errors (empty output, truncated results, wrong columns)?

Respond ONLY with a JSON object:
{{
  "verdict": "CONFIRMED" | "REFUTED" | "UNCERTAIN",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence",
  "issues": ["list of specific problems, if any"]
}}"""

_ADVERSARIAL_PROBE_PROMPT = """You are a devil's advocate. Given a PROBLEM and proposed SOLUTION, generate edge cases and counterexamples that could break it.

PROBLEM:
{problem}

SOLUTION:
{solution}

Generate 3-5 specific, testable edge cases. For each, explain what could go wrong.

Respond ONLY with a JSON object:
{{
  "edge_cases": [
    {{"case": "description", "risk": "what could break"}},
    ...
  ]
}}"""


class Verifier:
    """Self-evaluation module that uses LLM introspection to check the agent's
    own work before presenting it to the user."""

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client

    async def _call_llm(self, prompt: str, temperature: float = 0.1) -> dict:
        """Make a verification LLM call, preferring worker nodes for cost.

        Token budget is sized for thinking models (Qwen/DeepSeek-R1 style)
        that emit a <think>...</think> prelude before the JSON — a 512
        cap was getting consumed entirely by the prelude on the default
        qwen-3.5-27b, so every verifier call came back empty.
        """
        if not self.llm_client:
            return {}

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 2048,
            "stream": False,
        }

        # Try routing to worker pool first (cheaper, different perspective).
        # `LLMClient.route()` returns the extracted content string, NOT a
        # full chat-completion dict — the previous `isinstance(result, dict)`
        # check was always False, so the worker path was effectively dead
        # and every verify always fell through to the foreground model.
        route_fn = getattr(self.llm_client, "route", None)
        if route_fn:
            try:
                result = await route_fn(
                    "VERIFY", payload, max_tokens=2048,
                    temperature=temperature, fallback=None,
                )
            except Exception as exc:
                logger.debug("Verifier worker route failed: %s", exc)
                result = None
            if result:
                text = result if isinstance(result, str) else (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                parsed = self._parse_json(text)
                if parsed:
                    return parsed
                # Empty/unparseable worker response → fall through to
                # direct call rather than giving up.

        # Fallback to direct call
        try:
            result = await self.llm_client.chat_completion(payload)
            text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return self._parse_json(text)
        except Exception as exc:
            logger.warning("Verifier LLM call failed: %s", exc)
            return {}

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Robustly extract a JSON object from LLM output."""
        if not text:
            return {}
        import re
        # Strip reasoning-model <think>...</think> preludes (closed OR
        # unclosed — budget exhaustion can leave the block open). The
        # greedy regex fallback below would otherwise match braces
        # INSIDE the thinking block instead of the real JSON verdict.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>[\s\S]*$", "", text).strip()
        if not text:
            return {}
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Walk every `{...}` block from the end — some models emit a
        # final JSON after prose; the last parseable one wins.
        for candidate in reversed(re.findall(r"\{[\s\S]*?\}", text) or []):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        # Last-resort greedy match (multi-line JSON with nested braces).
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def _build_verify_result(self, data: dict) -> Optional[VerifyResult]:
        """Convert a parsed JSON dict into a VerifyResult.

        Returns ``None`` when the verifier LLM produced no usable output
        (worker unavailable, JSON unparseable, upstream error). Callers
        surface that as "skipped" rather than conflating it with a real
        low-confidence UNCERTAIN verdict — the two cases are logged
        identically as "UNCERTAIN (0%)" previously, which hid genuine
        failures of the verifier pipeline itself.
        """
        if not data:
            return None
        verdict_str = data.get("verdict", "UNCERTAIN").upper()
        try:
            verdict = VerifyVerdict(verdict_str)
        except ValueError:
            verdict = VerifyVerdict.UNCERTAIN
        return VerifyResult(
            verdict=verdict,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            issues=data.get("issues", []),
        )

    async def verify_claim(self, claim: str, evidence: str,
                           context: str = "") -> Optional[VerifyResult]:
        """Check whether *claim* is supported by *evidence*."""
        prompt = _VERIFY_CLAIM_PROMPT.format(
            claim=claim[:2000],
            evidence=evidence[:4000],
            context=context[:1000],
        )
        data = await self._call_llm(prompt, temperature=0.1)
        return self._build_verify_result(data)

    async def verify_code_output(self, code: str, output: str,
                                 intent: str) -> Optional[VerifyResult]:
        """Check whether *code* and its *output* actually answer *intent*."""
        prompt = _VERIFY_CODE_PROMPT.format(
            intent=intent[:1000],
            code=code[:4000],
            output=output[:4000],
        )
        data = await self._call_llm(prompt, temperature=0.1)
        return self._build_verify_result(data)

    async def adversarial_probe(self, problem: str,
                                solution: str) -> List[Dict[str, str]]:
        """Generate edge cases that could break *solution*."""
        prompt = _ADVERSARIAL_PROBE_PROMPT.format(
            problem=problem[:2000],
            solution=solution[:4000],
        )
        data = await self._call_llm(prompt, temperature=0.4)
        return data.get("edge_cases", [])
