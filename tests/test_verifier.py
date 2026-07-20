"""Tests for the Reflective Self-Evaluation (Verifier) module."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from ghost_agent.core.verifier import Verifier, VerifyResult, VerifyVerdict


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.chat_completion = AsyncMock()
    client.route = AsyncMock(return_value=None)  # Force fallback to direct call
    return client


@pytest.fixture
def verifier(mock_llm_client):
    return Verifier(llm_client=mock_llm_client)


class TestVerifyResult:
    def test_confirmed_passes(self):
        r = VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.9)
        assert r.passed() is True

    def test_refuted_fails(self):
        r = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.8)
        assert r.passed() is False

    def test_uncertain_fails(self):
        r = VerifyResult(verdict=VerifyVerdict.UNCERTAIN, confidence=0.5)
        assert r.passed() is False

    def test_to_dict(self):
        r = VerifyResult(
            verdict=VerifyVerdict.CONFIRMED,
            confidence=0.95,
            reasoning="All good",
            issues=["minor: formatting"],
        )
        d = r.to_dict()
        assert d["verdict"] == "CONFIRMED"
        assert d["confidence"] == 0.95
        assert d["issues"] == ["minor: formatting"]


class TestVerifier:
    async def test_verify_claim_confirmed(self, verifier, mock_llm_client):
        mock_llm_client.chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "verdict": "CONFIRMED",
                "confidence": 0.9,
                "reasoning": "Claim matches evidence",
                "issues": [],
            })}}],
        }
        result = await verifier.verify_claim(
            claim="The sky is blue",
            evidence="Rayleigh scattering causes the sky to appear blue",
        )
        assert result.verdict == VerifyVerdict.CONFIRMED
        assert result.confidence == 0.9

    async def test_verify_claim_refuted(self, verifier, mock_llm_client):
        mock_llm_client.chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "verdict": "REFUTED",
                "confidence": 0.85,
                "reasoning": "Evidence contradicts claim",
                "issues": ["Incorrect calculation"],
            })}}],
        }
        result = await verifier.verify_claim(
            claim="2 + 2 = 5",
            evidence="Basic arithmetic shows 2 + 2 = 4",
        )
        assert result.verdict == VerifyVerdict.REFUTED
        assert not result.passed()
        assert "Incorrect calculation" in result.issues

    async def test_verify_code_output(self, verifier, mock_llm_client):
        mock_llm_client.chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "verdict": "CONFIRMED",
                "confidence": 0.95,
                "reasoning": "Output correctly answers the question",
                "issues": [],
            })}}],
        }
        result = await verifier.verify_code_output(
            code="print(sum([1,2,3]))",
            output="6",
            intent="Calculate the sum of 1, 2, and 3",
        )
        assert result.passed()

    async def test_verify_with_no_llm(self):
        # Verifier with no LLM client cannot produce a real verdict.
        # Must return None so callers can log "skipped" instead of
        # emitting a misleading UNCERTAIN (0%) that is indistinguishable
        # from a genuine low-confidence result.
        v = Verifier(llm_client=None)
        result = await v.verify_claim("test", "test")
        assert result is None

    async def test_verify_handles_malformed_json(self, verifier, mock_llm_client):
        mock_llm_client.chat_completion.return_value = {
            "choices": [{"message": {"content": "Not valid JSON at all"}}],
        }
        result = await verifier.verify_claim("test", "test")
        assert result is None

    async def test_verify_handles_llm_exception(self, verifier, mock_llm_client):
        mock_llm_client.chat_completion.side_effect = Exception("LLM down")
        result = await verifier.verify_claim("test", "test")
        assert result is None

    def test_parse_json_extracts_from_text(self):
        text = 'Some preamble\n{"verdict": "CONFIRMED", "confidence": 0.8}\nsome trailing'
        result = Verifier._parse_json(text)
        assert result["verdict"] == "CONFIRMED"

    def test_parse_json_empty(self):
        assert Verifier._parse_json("") == {}
        assert Verifier._parse_json(None) == {}

    def test_parse_json_never_returns_non_dict(self):
        # A judge replying with a bare array/string used to escape as-is;
        # callers do `.get(...)` on the result, so that AttributeError'd
        # out of verify_claim and the whole pass was silently skipped.
        assert Verifier._parse_json("[1, 2, 3]") == {}
        assert Verifier._parse_json('"just a string"') == {}
        assert Verifier._parse_json(json.dumps(["a", "b"])) == {}
        # Array of dicts: the fragment walk may salvage one, but the
        # return value must still be a dict.
        assert isinstance(
            Verifier._parse_json('[{"a": 1}, {"b": 2}]'), dict)

    def test_parse_json_salvages_singleton_dict_array(self):
        wrapped = json.dumps([{"verdict": "CONFIRMED", "confidence": 0.8}])
        assert Verifier._parse_json(wrapped) == {
            "verdict": "CONFIRMED", "confidence": 0.8}

    def test_build_verify_result_requires_verdict_key(self, verifier):
        # A parseable dict with no "verdict" key (typically an inner
        # fragment of a truncated reply) is not a verdict — it must read
        # as "skipped" (None), not fabricate UNCERTAIN@0.5.
        assert verifier._build_verify_result(
            {"suspect": 2, "real": True}) is None
        # Key present but null still degrades to UNCERTAIN (unchanged).
        res = verifier._build_verify_result({"verdict": None})
        assert res is not None
        assert res.verdict == VerifyVerdict.UNCERTAIN
