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

_VERIFY_CODE_PROMPT = """You are a code output auditor. Determine whether the agent's RESPONSE actually answers the user's INTENT — including any explicit constraints in the user's wording.

USER INTENT:
{intent}

CODE THE AGENT RAN:
{code}

TOOL OUTPUT:
{output}

AGENT'S RESPONSE TO THE USER:
{response}

Check, in order:

1. **Constraint satisfaction (highest priority).** Does the user's wording include explicit constraints on the form of the answer? Examples: "just give me the code", "in one sentence", "without using X", "list only the names", "as JSON". If yes, does the AGENT'S RESPONSE satisfy those constraints? If the user asked for code and the agent returned a number / prose / a result, that is a REFUTED — the agent answered a different question than the one asked, even if the tool output is internally consistent.
2. Does the response contain the information the user asked for?
3. Are the numbers/results plausible (no obvious off-by-one, wrong units, etc.)?
4. Are there silent errors (empty output, truncated results, wrong columns)?

Common failure shapes to flag:
- User asks for code/snippet/command → agent returns a result or summary instead of the snippet
- User asks for code AND the agent's RESPONSE does not contain a fenced code block — REFUTED regardless of what the tool output says. "The script ran correctly and prints 1 to 10" is NOT a substitute for the script itself; the user cannot paste a confirmation message into their editor. If `intent` contains verbs like give/show/write/draft + nouns like script/code/function/snippet/query/command, the response MUST include the source in a code fence.
- User asks "how do I X" → agent does X and reports the answer instead of explaining the method
- User asks for a specific format → agent ignores the format
- Tool output is a sandbox-internal artefact the user can't actually use

A verdict of CONFIRMED requires BOTH the tool output to be sound AND the response to match what the user asked for. If only the first holds, return REFUTED.

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
                                 intent: str,
                                 *, response: str = "") -> Optional[VerifyResult]:
        """Check whether the agent's *response* actually answers
        *intent*, given the *code* it ran and the *output* it
        observed.

        ``response`` is the agent's user-facing reply. Defaults to
        empty for back-compat with older callers, but production
        callers should always pass it — without it, the verifier
        falls back to "does the output match the claim" auditing
        which can't catch wrong-question answers (user asks for
        code, agent gives a number; user asks for format X, agent
        replies in format Y). Those failure shapes are the dominant
        wrong-but-confidently-confirmed mode in practice.
        """
        prompt = _VERIFY_CODE_PROMPT.format(
            intent=intent[:1000],
            code=code[:4000],
            output=output[:4000],
            response=(response or "(response not provided to verifier)")[:4000],
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
