"""Refute escalation: a CHEAP judge's REFUTED verdict is re-adjudicated on
the MAIN model before it is allowed to act (2026-07-26).

Measured motivation (live): the worker judge (Gemma 4 E4B) refuted a CORRECT
"latest PostgreSQL version is 18.4" answer at 90% and called 49152 bytes
"not exactly 48 KB"; the main 35B judged the same three cases 3/3 correctly.
A false REFUTE scrubs the turn's lessons, files follow-ups, shows the user a
correction and marks the trajectory failed — so it must be confirmed first.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from ghost_agent.core.verifier import (
    Verifier, VerifyResult, VerifyVerdict, _escalate_refute_enabled,
)


def _res(verdict, conf=0.9, issues=None):
    return VerifyResult(verdict=verdict, confidence=conf,
                        reasoning="r", issues=list(issues or []))


class _Client:
    """LLM client stub that advertises a cheap route."""
    def __init__(self, worker=True, critic=False):
        self.worker_clients = [object()] if worker else []
        self.critic_clients = [object()] if critic else []


@pytest.fixture
def v():
    return Verifier(llm_client=_Client())


class TestEscalationBehaviour:
    @pytest.mark.asyncio
    async def test_overturns_cheap_refute_when_main_confirms(self, v, monkeypatch):
        calls = {"n": 0}

        async def _strong(claim, evidence, context, force_main=False):
            calls["n"] += 1
            assert force_main is True          # must escalate to MAIN
            return _res(VerifyVerdict.CONFIRMED, 1.0)

        monkeypatch.setattr(v, "_verify_claim_two_stage", _strong)
        cheap = _res(VerifyVerdict.REFUTED, 0.9, ["derived fact not restated"])
        out = await v._escalate_refute(cheap, "c", "e", "ctx")
        assert out.verdict == VerifyVerdict.CONFIRMED
        assert getattr(out, "escalated_overturn", False) is True
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_keeps_refute_when_main_agrees(self, v, monkeypatch):
        async def _strong(claim, evidence, context, force_main=False):
            return _res(VerifyVerdict.REFUTED, 0.95, ["fabricated version"])
        monkeypatch.setattr(v, "_verify_claim_two_stage", _strong)
        cheap = _res(VerifyVerdict.REFUTED, 0.9, ["fabricated version"])
        out = await v._escalate_refute(cheap, "c", "e", "ctx")
        assert out.verdict == VerifyVerdict.REFUTED
        assert getattr(out, "escalated_overturn", False) is False

    @pytest.mark.asyncio
    async def test_uncertain_is_a_safe_overturn(self, v, monkeypatch):
        """UNCERTAIN is enough: every destructive action (corpus-fail,
        lesson scrub, correction banner) is gated on REFUTED specifically."""
        async def _strong(claim, evidence, context, force_main=False):
            return _res(VerifyVerdict.UNCERTAIN, 0.8)
        monkeypatch.setattr(v, "_verify_claim_two_stage", _strong)
        out = await v._escalate_refute(
            _res(VerifyVerdict.REFUTED, 0.9), "c", "e", "ctx")
        assert out.verdict == VerifyVerdict.UNCERTAIN

    @pytest.mark.asyncio
    async def test_non_refuted_verdicts_are_untouched(self, v, monkeypatch):
        async def _boom(*a, **k):
            raise AssertionError("must not escalate a non-refute")
        monkeypatch.setattr(v, "_verify_claim_two_stage", _boom)
        for verdict in (VerifyVerdict.CONFIRMED, VerifyVerdict.UNCERTAIN):
            out = await v._escalate_refute(_res(verdict), "c", "e", "ctx")
            assert out.verdict == verdict
        assert await v._escalate_refute(None, "c", "e", "ctx") is None

    @pytest.mark.asyncio
    async def test_no_cheap_route_means_nothing_to_escalate_to(self, monkeypatch):
        """When the MAIN model already judged it, escalation is a no-op —
        never a duplicate call."""
        v = Verifier(llm_client=_Client(worker=False, critic=False))

        async def _boom(*a, **k):
            raise AssertionError("must not escalate when main already judged")
        monkeypatch.setattr(v, "_verify_claim_two_stage", _boom)
        cheap = _res(VerifyVerdict.REFUTED, 0.9)
        assert (await v._escalate_refute(cheap, "c", "e", "ctx")) is cheap

    @pytest.mark.asyncio
    async def test_escalation_error_keeps_original_verdict(self, v, monkeypatch):
        """Escalation can only REDUCE false refutes — never make the gate
        less available than before."""
        async def _explode(*a, **k):
            raise RuntimeError("main model down")
        monkeypatch.setattr(v, "_verify_claim_two_stage", _explode)
        monkeypatch.setattr(v, "_call_llm", _explode)
        cheap = _res(VerifyVerdict.REFUTED, 0.9, ["x"])
        out = await v._escalate_refute(cheap, "c", "e", "ctx")
        assert out.verdict == VerifyVerdict.REFUTED

    @pytest.mark.asyncio
    async def test_kill_switch_disables_escalation(self, v, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_ESCALATE_REFUTE", "0")

        async def _boom(*a, **k):
            raise AssertionError("kill switch must prevent escalation")
        monkeypatch.setattr(v, "_verify_claim_two_stage", _boom)
        cheap = _res(VerifyVerdict.REFUTED, 0.9)
        assert (await v._escalate_refute(cheap, "c", "e", "ctx")) is cheap
        assert _escalate_refute_enabled() is False


class TestForceMainRouting:
    @pytest.mark.asyncio
    async def test_force_main_skips_critic_and_worker(self, monkeypatch):
        """force_main must bypass BOTH cheap pools so the escalation is
        actually adjudicated by the strong model."""
        client = MagicMock()
        client.critic_clients = [object()]
        client.route = MagicMock(side_effect=AssertionError("worker route used"))
        client.chat_completion = MagicMock(
            side_effect=AssertionError("critic pool used"))
        v = Verifier(llm_client=client)
        # Direct main call is the only path left; stub it at the httpx layer
        # by making chat_completion raise only for the critic kwarg.
        called = {}

        async def _direct(payload, **kwargs):
            called.update(kwargs)
            return {"choices": [{"message": {"content": '{"verdict":"CONFIRMED","confidence":1.0}'}}]}
        client.chat_completion = _direct
        out = await v._call_llm("p", force_main=True)
        assert out.get("verdict") == "CONFIRMED"
        assert called.get("use_critic") is not True   # critic pool skipped
