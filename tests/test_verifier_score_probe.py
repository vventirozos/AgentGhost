"""Tests for the logit-expectation score probe (§4F Phase 3a).

One bounded digit-scale call after a two-stage verdict; the expectation
over the digit token distribution blends into `confidence`. Contracts:
verdicts are never changed, UNCERTAIN is excluded, any probe failure
leaves the result untouched, and the feature is env-gated
(GHOST_VERIFY_LOGIT_EXPECT, default OFF, read per call). Since §4BL the
consumer is the verdict-gated CONFIRM cap (T frozen on the corrected
design half; REFUTED readings recorded, never acted on).
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


class TestProbeCap:
    """§4BL: the w-blend is REPLACED by the verdict-gated CONFIRM cap
    (the §4BI foreclosure proved no light symmetric blend can move
    quantized confidences across the 0.7 gate; §4BJ measured the probe
    erring toward "acceptable" on faulty claims → CONFIRM-only)."""

    def test_disbelieved_confirm_is_capped(self, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "1")
        # probe: digit 9 p=.9, digit 0 p=.1 → E=8.1/9=0.9 < T=0.966
        client = _SeqClient([_stage1(), _stage2("CONFIRMED", 1.0),
                             _probe("9", 0.9)])
        res = _run_two_stage(client)
        assert res.verdict == VerifyVerdict.CONFIRMED
        assert res.probe_score == pytest.approx(0.9)
        assert res.confidence == pytest.approx(0.6)   # capped, not blended
        assert "probe-capped" in res.reasoning
        assert len(client.calls) == 3

    def test_believed_confirm_is_untouched(self, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "1")
        monkeypatch.setenv("GHOST_VERIFY_PROBE_CAP_T", "0.85")
        client = _SeqClient([_stage1(), _stage2("CONFIRMED", 1.0),
                             _probe("9", 0.9)])   # 0.9 ≥ 0.85 → no cap
        res = _run_two_stage(client)
        assert res.confidence == pytest.approx(1.0)
        assert "probe-capped" not in (res.reasoning or "")

    def test_refuted_gets_reading_recorded_and_nothing_else(self, monkeypatch):
        # §4BJ: the probe errs toward "acceptable" on faulty claims — a
        # high reading on a REFUTED must neither weaken nor strengthen it.
        monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "1")
        client = _SeqClient([_stage1(), _stage2("REFUTED", 1.0),
                             _probe("9", 0.9)])
        res = _run_two_stage(client)
        assert res.verdict == VerifyVerdict.REFUTED
        assert res.probe_score == pytest.approx(0.9)
        assert res.confidence == pytest.approx(1.0)   # untouched

    def test_cap_never_raises_a_low_confidence(self, monkeypatch):
        # A CONFIRMED already below the cap must stay where it is — the
        # cap is a ceiling, not an assignment.
        monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "1")
        client = _SeqClient([_stage1(), _stage2("CONFIRMED", 0.5),
                             _probe("0", 0.9)])   # probe low → disbelieved
        res = _run_two_stage(client)
        assert res.confidence == pytest.approx(0.5)

    def test_threshold_and_cap_env_junk_safe(self, monkeypatch):
        from ghost_agent.core.verifier import (_probe_cap_threshold,
                                               _probe_conf_cap)
        monkeypatch.setenv("GHOST_VERIFY_PROBE_CAP_T", "junk")
        assert _probe_cap_threshold() == 0.966
        monkeypatch.setenv("GHOST_VERIFY_PROBE_CAP_T", "7")
        assert _probe_cap_threshold() == 1.0
        monkeypatch.setenv("GHOST_VERIFY_PROBE_CONF_CAP", "junk")
        assert _probe_conf_cap() == 0.6
        monkeypatch.setenv("GHOST_VERIFY_PROBE_CONF_CAP", "-3")
        assert _probe_conf_cap() == 0.0

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


class TestProbeCapBoundary:
    def test_reading_exactly_at_threshold_is_not_capped(self, monkeypatch):
        # §4BL R1 MIN-1: strict < is load-bearing — a real design-half
        # trial sits exactly on a candidate boundary, and the one-char
        # mutation `<` → `<=` survived the suite. Pin the boundary with
        # T set to a reachable expectation value.
        monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "1")
        monkeypatch.setenv("GHOST_VERIFY_PROBE_CAP_T", "0.9")
        client = _SeqClient([_stage1(), _stage2("CONFIRMED", 1.0),
                             _probe("9", 0.9)])   # E = 0.9 == T
        res = _run_two_stage(client)
        assert res.probe_score == pytest.approx(0.9)
        assert res.confidence == pytest.approx(1.0)   # NOT capped
        assert "probe-capped" not in (res.reasoning or "")

    def test_default_threshold_is_the_frozen_value(self, monkeypatch):
        from ghost_agent.core.verifier import _probe_cap_threshold
        monkeypatch.delenv("GHOST_VERIFY_PROBE_CAP_T", raising=False)
        assert _probe_cap_threshold() == 0.966

    def test_cap_is_cheap_pass_only(self, monkeypatch):
        # §4BL R1 MAJ-2: under the §4BK kill-switch legacy mix, a MAIN
        # two-stage CONFIRMED must not be capped (unmeasured regime; on
        # the refute-escalation path the probe would weaken refutes).
        import asyncio
        monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "1")
        client = _SeqClient([_stage1(), _stage2("CONFIRMED", 1.0),
                             _probe("9", 0.9)])   # 0.9 < 0.966
        v = Verifier(llm_client=client)
        res = asyncio.run(v._verify_claim_two_stage(
            "claim", "evidence", "ctx", force_main=True))
        assert res.verdict == VerifyVerdict.CONFIRMED
        assert res.probe_score == pytest.approx(0.9)  # reading recorded
        assert res.confidence == pytest.approx(1.0)   # NOT capped
