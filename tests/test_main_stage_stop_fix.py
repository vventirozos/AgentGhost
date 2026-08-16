"""§4BJ — the two-stage MAIN-leg stop-truncation fix.

The stop:["\\n"] payload discipline was measured on the CHEAP judge
(99.5% minified-compliant) but rode along on force_main stage calls,
where the 35B pretty-prints its JSON — content came back as a lone "{"
at finish=stop on 74% of MAIN adjudications, silently demoting every
escalated re-verify to the classic prompt (flip-i's "48% fallback").
Rule: $GHOST_HOME/system/eval/fallback_fix/DECISION_RULE.md.

Pins: force_main stage payloads carry no stop (but keep the no-think
switch); cheap-leg payloads keep the full discipline; the kill switch
restores legacy; and a pretty-printed MAIN stage answer now parses end
to end through _verify_claim_two_stage.
"""
import json

import pytest

from ghost_agent.core.verifier import Verifier

pytestmark = pytest.mark.asyncio


class _CapturingClient:
    """Records every payload AND its kwargs; answers scripted responses."""

    def __init__(self, responses, critic=False):
        self.responses = list(responses)
        self.payloads = []
        self.kwargs = []
        self.critic_clients = [object()] if critic else []
        self.worker_clients = []

    async def chat_completion(self, payload, **kw):
        self.payloads.append(payload)
        self.kwargs.append(kw)
        text = self.responses.pop(0) if self.responses else "{}"
        return {"choices": [{"message": {"content": text},
                             "finish_reason": "stop"}]}


_PRETTY_STAGE1 = """{
  "suspects": [
    {"quote": "the total is 42", "check": "support",
     "reason": "specific fact worth checking"},
    {"quote": "tests passed", "check": "support", "reason": "unverified"},
    {"quote": "file written", "check": "artifact", "reason": "unverified"}
  ]
}"""

_PRETTY_STAGE2 = """{
  "checks": [{"suspect": 1, "real": false, "why": "evidence supports it"}],
  "verdict": "CONFIRMED",
  "confidence": 0.9,
  "reasoning": "the claim is supported",
  "issues": []
}"""


async def test_force_main_stage_payload_has_no_stop(monkeypatch):
    monkeypatch.delenv("GHOST_VERIFY_MAIN_STAGE_STOP", raising=False)
    client = _CapturingClient([_PRETTY_STAGE1, _PRETTY_STAGE2])
    v = Verifier(llm_client=client)
    res = await v._verify_claim_two_stage("claim", "evidence", "ctx",
                                          force_main=True)
    assert len(client.payloads) == 2
    for p in client.payloads:
        assert "stop" not in p, "force_main stage call carried the stop"
        # The no-think switch stays: the belt came off, not the seatbelt.
        assert p["messages"][0]["content"].endswith("/no_think")
        assert p["chat_template_kwargs"] == {"enable_thinking": False}
    # And the pretty multi-line answers PARSE end to end.
    assert res is not None
    assert res.verdict.value == "CONFIRMED"
    assert res.suspects and len(res.suspects) == 3


async def test_cheap_leg_keeps_the_stop_discipline(monkeypatch):
    # The cheap judge is minified-compliant (measured 99.5%) — its
    # discipline is untouched. This asserts the CRITIC-POOL branch
    # specifically (use_critic=True on the call — R1: without that
    # assert, deleting the branch fell through to the direct path,
    # whose non-force_main payload also carries stop, and the test
    # still passed while pinning nothing).
    monkeypatch.delenv("GHOST_VERIFY_MAIN_STAGE_STOP", raising=False)
    client = _CapturingClient(
        ['{"suspects": [{"quote": "q", "check": "support", "reason": "r"}]}'],
        critic=True)
    v = Verifier(llm_client=client)
    await v._call_llm("prompt", json_only=True, max_tokens=64)
    assert client.payloads, "critic pool was not called"
    assert client.kwargs[0].get("use_critic") is True
    assert client.payloads[0].get("stop") == ["\n"]


async def test_kill_switch_restores_legacy_stop(monkeypatch):
    monkeypatch.setenv("GHOST_VERIFY_MAIN_STAGE_STOP", "1")
    client = _CapturingClient(["{"])
    v = Verifier(llm_client=client)
    await v._call_llm("prompt", json_only=True, force_main=True,
                      max_tokens=64)
    assert client.payloads[0].get("stop") == ["\n"]


async def test_lone_brace_still_falls_back_not_crashes(monkeypatch):
    # The historic failure shape must remain a clean None (classic
    # fallback), never an exception — e.g. under the kill switch.
    monkeypatch.setenv("GHOST_VERIFY_MAIN_STAGE_STOP", "1")
    client = _CapturingClient(["{", "{"])
    v = Verifier(llm_client=client)
    res = await v._verify_claim_two_stage("c", "e", "x", force_main=True)
    assert res is None
