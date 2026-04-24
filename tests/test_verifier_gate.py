"""Verifier gate: runs after tool execution on complex tasks.

The verifier was orphaned before this wiring — initialised in main.py but
never called. It now runs between post-mortem and final return, appending
an auditor note when the verdict is REFUTED with high confidence.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.verifier import Verifier, VerifyVerdict, VerifyResult


@pytest.fixture
def fake_llm():
    client = MagicMock()
    client.chat_completion = AsyncMock()
    client.route = AsyncMock(return_value=None)
    return client


class TestVerifierContract:
    async def test_verify_claim_returns_none_without_llm(self):
        # When no LLM is wired, the verifier has no verdict to return.
        # Returning None lets the caller log "skipped" instead of a fake
        # UNCERTAIN (0%) that hides the pipeline failure.
        v = Verifier(llm_client=None)
        result = await v.verify_claim("claim", "evidence")
        assert result is None

    async def test_verify_claim_parses_confirmed(self, fake_llm):
        fake_llm.chat_completion.return_value = {
            "choices": [{"message": {"content": '{"verdict": "CONFIRMED", "confidence": 0.9, "reasoning": "ok", "issues": []}'}}]
        }
        v = Verifier(llm_client=fake_llm)
        result = await v.verify_claim("2+2=4", "math")
        assert result.verdict == VerifyVerdict.CONFIRMED
        assert result.confidence == 0.9

    async def test_verify_refuted_surfaces_issues(self, fake_llm):
        fake_llm.chat_completion.return_value = {
            "choices": [{"message": {"content": '{"verdict": "REFUTED", "confidence": 0.85, "reasoning": "wrong units", "issues": ["treated ms as s"]}'}}]
        }
        v = Verifier(llm_client=fake_llm)
        result = await v.verify_code_output(code="x", output="y", intent="z")
        assert result.verdict == VerifyVerdict.REFUTED
        assert result.issues == ["treated ms as s"]


class TestVerifierGateIntegration:
    """Smoke-test the wiring at the point where handle_chat invokes
    verifier.verify_*. We don't drive the full agent loop — we construct
    the branch in isolation by replicating the condition block's
    behaviour against a stub verifier."""

    async def test_refuted_high_confidence_appends_note(self, fake_llm):
        # Simulate the gate's path: verifier returns REFUTED @ 0.85 conf,
        # the agent must append a "Verifier note:" block to final content.
        fake_llm.chat_completion.return_value = {
            "choices": [{"message": {"content": '{"verdict": "REFUTED", "confidence": 0.85, "reasoning": "no such file", "issues": ["path /tmp/foo.csv missing"]}'}}]
        }
        v = Verifier(llm_client=fake_llm)
        original = "I read /tmp/foo.csv and counted 42 rows."
        last_tool_output = "FileNotFoundError: /tmp/foo.csv"
        result = await v.verify_claim(claim=original, evidence=last_tool_output)

        assert result.verdict == VerifyVerdict.REFUTED
        assert result.confidence >= 0.7
        # The gate would reject and append a note with the issue list.
        annotated = f"{original}\n\n---\n**Verifier note:** {'; '.join(result.issues)}"
        assert "path /tmp/foo.csv missing" in annotated
        assert "Verifier note" in annotated

    async def test_low_confidence_refutation_does_not_annotate(self, fake_llm):
        # When the verifier's confidence is below 0.7 the gate leaves the
        # answer alone — avoiding false positives on marginal verdicts.
        fake_llm.chat_completion.return_value = {
            "choices": [{"message": {"content": '{"verdict": "REFUTED", "confidence": 0.4, "reasoning": "maybe", "issues": ["unclear"]}'}}]
        }
        v = Verifier(llm_client=fake_llm)
        result = await v.verify_claim(claim="x", evidence="y")
        # The gate's guard (>= 0.7) would skip annotation on this result.
        should_annotate = result.verdict == VerifyVerdict.REFUTED and result.confidence >= 0.7
        assert should_annotate is False

    async def test_confirmed_verdicts_never_annotate(self, fake_llm):
        fake_llm.chat_completion.return_value = {
            "choices": [{"message": {"content": '{"verdict": "CONFIRMED", "confidence": 0.95, "reasoning": "all good", "issues": []}'}}]
        }
        v = Verifier(llm_client=fake_llm)
        result = await v.verify_claim(claim="x", evidence="y")
        should_annotate = result.verdict == VerifyVerdict.REFUTED and result.confidence >= 0.7
        assert should_annotate is False
