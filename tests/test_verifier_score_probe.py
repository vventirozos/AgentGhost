"""Tests for the logit-expectation score probe (§4F Phase 3a).

One bounded digit-scale call after a two-stage verdict; the expectation
over the digit token distribution blends into `confidence`. Contracts:
verdicts are never changed, UNCERTAIN is excluded, any probe failure
leaves the result untouched, and the feature is env-gated
(GHOST_VERIFY_LOGIT_EXPECT, default OFF, read per call).
"""

import asyncio
import json
import math

import pytest

from ghost_agent.core.verifier import (
    Verifier,
    VerifyVerdict,
    _digit_expectation,
    _logit_expect_enabled,
)


def _lp(token: str, p: float) -> dict:
    return {"token": token, "logprob": math.log(p)}


class TestDigitExpectation:
    def test_expectation_over_digit_mass(self):
        # 8 with p=.6, 7 with p=.3, 2 with p=.1 → E=(8*.6+7*.3+2*.1)/1.0=7.1
        top = [_lp("8", 0.6), _lp("7", 0.3), _lp("2", 0.1)]
        assert _digit_expectation(top) == pytest.approx(7.1 / 9.0)

    def test_non_digit_tokens_ignored_and_renormalized(self):
        top = [_lp("9", 0.4), _lp(" nine", 0.5), _lp("0", 0.1)]
        # digit mass: 9*.4 + 0*.1 over .5 → 7.2/9
        assert _digit_expectation(top) == pytest.approx(7.2 / 9.0)

    def test_no_digit_mass_returns_none(self):
        assert _digit_expectation([_lp("yes", 0.9)]) is None
        assert _digit_expectation([]) is None


class TestEnvGate:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("GHOST_VERIFY_LOGIT_EXPECT", raising=False)
        assert not _logit_expect_enabled()

    def test_enabled(self, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "1")
        assert _logit_expect_enabled()


class _SeqClient:
    """chat_completion fake returning queued responses in order."""

    critic_clients = None

    def __init__(self, responses):
        self._queue = list(responses)
        self.calls = []

    async def chat_completion(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        return self._queue.pop(0)


def _msg(content: str, logprobs: dict = None) -> dict:
    choice = {"message": {"content": content}}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return {"choices": [choice]}


def _stage1() -> dict:
    return _msg(json.dumps({"suspects": [
        {"quote": "34C", "check": "support", "reason": "check temp"}]}))


def _stage2(verdict: str, conf: float = 0.9) -> dict:
    return _msg(json.dumps({
        "checks": [{"suspect": 1, "real": verdict == "REFUTED",
                    "why": "checked"}],
        "extra_problems": [], "verdict": verdict, "confidence": conf,
        "reasoning": "r", "issues": [],
    }))


def _probe(digit: str, p: float) -> dict:
    return _msg(digit, logprobs={"content": [{
        "token": digit,
        "logprob": math.log(p),
        "top_logprobs": [_lp(digit, p), _lp(str(9 - int(digit)), 1 - p)],
    }]})


def _run_two_stage(client) -> "VerifyResult":
    v = Verifier(llm_client=client)
    return asyncio.run(v._verify_claim_two_stage("claim", "evidence", "ctx"))


class TestProbeBlend:
    def test_confirmed_blends_toward_probe(self, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "1")
        # probe: digit 9 p=.9, digit 0 p=.1 → E=8.1/9=0.9
        client = _SeqClient([_stage1(), _stage2("CONFIRMED", 1.0),
                             _probe("9", 0.9)])
        res = _run_two_stage(client)
        assert res.verdict == VerifyVerdict.CONFIRMED
        assert res.probe_score == pytest.approx(0.9)
        assert res.confidence == pytest.approx(0.5 * 1.0 + 0.5 * 0.9)
        assert len(client.calls) == 3

    def test_refuted_uses_inverted_probe(self, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "1")
        # probe says acceptable≈0.9 but verdict REFUTED → aligned = 0.1
        client = _SeqClient([_stage1(), _stage2("REFUTED", 1.0),
                             _probe("9", 0.9)])
        res = _run_two_stage(client)
        assert res.verdict == VerifyVerdict.REFUTED
        assert res.confidence == pytest.approx(0.5 * 1.0 + 0.5 * 0.1)

    def test_verdict_never_changed_by_probe(self, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "1")
        client = _SeqClient([_stage1(), _stage2("REFUTED", 0.8),
                             _probe("9", 0.99)])
        res = _run_two_stage(client)
        assert res.verdict == VerifyVerdict.REFUTED

    def test_disabled_makes_no_probe_call(self, monkeypatch):
        monkeypatch.delenv("GHOST_VERIFY_LOGIT_EXPECT", raising=False)
        client = _SeqClient([_stage1(), _stage2("CONFIRMED", 0.7)])
        res = _run_two_stage(client)
        assert res.confidence == pytest.approx(0.7)
        assert res.probe_score is None
        assert len(client.calls) == 2

    def test_uncertain_skips_probe(self, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "1")
        client = _SeqClient([_stage1(), _stage2("UNCERTAIN", 0.5)])
        res = _run_two_stage(client)
        assert res.verdict == VerifyVerdict.UNCERTAIN
        assert res.probe_score is None
        assert len(client.calls) == 2

    def test_probe_failure_leaves_result_untouched(self, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "1")

        class _Boom(_SeqClient):
            async def chat_completion(self, payload, **kwargs):
                if len(self.calls) >= 2:
                    raise RuntimeError("probe endpoint down")
                return await super().chat_completion(payload, **kwargs)

        client = _Boom([_stage1(), _stage2("CONFIRMED", 0.8)])
        res = _run_two_stage(client)
        assert res.verdict == VerifyVerdict.CONFIRMED
        assert res.confidence == pytest.approx(0.8)
        assert res.probe_score is None

    def test_content_digit_fallback_without_distribution(self, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "1")
        client = _SeqClient([_stage1(), _stage2("CONFIRMED", 1.0),
                             _msg("9")])  # no logprobs at all
        res = _run_two_stage(client)
        assert res.probe_score == pytest.approx(1.0)
        assert res.confidence == pytest.approx(1.0)
