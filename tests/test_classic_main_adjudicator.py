"""§4BK — classic-on-MAIN is the DESIGNED escalation adjudicator.

§4BJ measured the paired comparison: with parsing fixed, two-stage-on-
MAIN upheld cheap-judge false refutes that classic overturned (7-vs-0
clean, p=0.0156 — all 13 gained FPs were lost rescues). Before §4BK the
attempt-then-fallback shape was also a nondeterministic MIX (two-stage
judged an escalation only when the 35B happened to minify, ~26%).
Rule: $GHOST_HOME/system/eval/classic_main_adjudicator/DECISION_RULE.md.

Pins: the refute-escalation re-adjudication goes STRAIGHT to the classic
prompt (no stage payloads on MAIN); the kill switch restores the
attempt; the CHEAP-leg two-stage pipeline is untouched.
"""
import pytest

from ghost_agent.core.verifier import Verifier, VerifyResult, VerifyVerdict

pytestmark = pytest.mark.asyncio


class _Client:
    def __init__(self):
        self.worker_clients = [object()]
        self.critic_clients = []


def _cheap_refute():
    return VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.95,
                        reasoning="r", issues=["fabricated version"])


async def test_escalated_readjudication_is_classic_only(monkeypatch):
    monkeypatch.delenv("GHOST_VERIFY_MAIN_TWO_STAGE", raising=False)
    v = Verifier(llm_client=_Client())
    prompts = []

    async def _llm(prompt, temperature=0.1, max_tokens=2048,
                   json_only=False, force_main=False):
        prompts.append((prompt, json_only, force_main))
        return {"verdict": "CONFIRMED", "confidence": 0.95,
                "reasoning": "the evidence supports the claim",
                "issues": []}

    monkeypatch.setattr(v, "_call_llm", _llm)
    out = await v._escalate_refute(_cheap_refute(), "c", "e", "ctx")
    # Exactly ONE main call — the classic prompt; no stage (json_only)
    # payloads ever reach MAIN.
    assert len(prompts) == 1
    prompt, json_only, force_main = prompts[0]
    assert force_main is True
    assert json_only is False
    assert "rigorous auditor. The agent ran a tool" in prompt
    assert "forced identification" not in prompt
    assert out.verdict == VerifyVerdict.CONFIRMED


async def test_kill_switch_restores_the_two_stage_attempt(monkeypatch):
    monkeypatch.setenv("GHOST_VERIFY_MAIN_TWO_STAGE", "1")
    v = Verifier(llm_client=_Client())
    prompts = []

    async def _llm(prompt, temperature=0.1, max_tokens=2048,
                   json_only=False, force_main=False):
        prompts.append((prompt[:60], json_only, force_main))
        # Unparseable stage answers -> two-stage returns None -> classic.
        if json_only:
            return {}
        return {"verdict": "REFUTED", "confidence": 0.9,
                "reasoning": "holds", "issues": ["x"]}

    monkeypatch.setattr(v, "_call_llm", _llm)
    out = await v._escalate_refute(_cheap_refute(), "c", "e", "ctx")
    # The legacy mix: stage-1 attempted on MAIN first, then classic.
    assert any(j for _, j, _ in prompts), "no stage attempt under the switch"
    assert prompts[0][1] is True and prompts[0][2] is True
    assert out.verdict == VerifyVerdict.REFUTED


async def test_cheap_leg_two_stage_untouched(monkeypatch):
    # The cheap-pass pipeline must still run two-stage regardless of the
    # §4BK gate — FPR control lives there (99.5% parse).
    monkeypatch.delenv("GHOST_VERIFY_MAIN_TWO_STAGE", raising=False)
    v = Verifier(llm_client=_Client())
    calls = []

    async def _llm(prompt, temperature=0.1, max_tokens=2048,
                   json_only=False, force_main=False):
        calls.append((json_only, force_main))
        # Adjudicate marker FIRST — the adjudicate template also
        # contains "forced identification" (the §4BJ cache-classifier
        # lesson, relearned here in miniature).
        if "delivering a final verdict" in prompt:
            return {"verdict": "CONFIRMED", "confidence": 0.9,
                    "reasoning": "fine", "issues": []}
        return {"suspects": [
            {"quote": "q1", "check": "support", "reason": "r"},
            {"quote": "q2", "check": "support", "reason": "r"},
            {"quote": "q3", "check": "artifact", "reason": "r"}]}

    monkeypatch.setattr(v, "_call_llm", _llm)
    res = await v._verify_claim_two_stage("c", "e", "ctx")
    assert res is not None and res.suspects
    # Both stage calls ran on the cheap path (force_main False).
    assert calls == [(True, False), (True, False)]


async def test_confirm_reverify_site_is_classic_only(monkeypatch):
    # R1 MAJ-4: only the refute site was pinned — reverting the gate at
    # the confirm-escalation's _reverify_on_main alone survived the
    # suite (every confirm-escalation plumbing test runs TWO_STAGE=0,
    # where the gated branch is dead regardless).
    monkeypatch.delenv("GHOST_VERIFY_MAIN_TWO_STAGE", raising=False)
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "1")
    v = Verifier(llm_client=_Client())
    calls = []

    async def _llm(prompt, temperature=0.1, max_tokens=2048,
                   json_only=False, force_main=False):
        calls.append((json_only, force_main))
        if json_only and not force_main:
            # Cheap two-stage: adjudicate marker FIRST (it contains the
            # enumerate marker too).
            if "delivering a final verdict" in prompt:
                return {"verdict": "CONFIRMED", "confidence": 1.0,
                        "reasoning": "fine", "issues": []}
            return {"suspects": [
                {"quote": "q1", "check": "support", "reason": "r"},
                {"quote": "q2", "check": "support", "reason": "r"},
                {"quote": "q3", "check": "artifact", "reason": "r"}]}
        # The re-verify on MAIN: §4BK default must send ONLY the classic
        # prompt (json_only False).
        return {"verdict": "REFUTED", "confidence": 0.9,
                "reasoning": "not supported",
                "issues": ["claimed success over a failed tool"]}

    monkeypatch.setattr(v, "_call_llm", _llm)
    out = await v.verify_claim("c", "e", "ctx", high_stakes=True)
    main_calls = [(j, f) for j, f in calls if f]
    assert main_calls, "confirm escalation never re-verified on MAIN"
    assert all(j is False for j, _ in main_calls), \
        "a stage-shaped (json_only) payload reached MAIN under the gate"
    assert out.confirm_withheld is True


async def test_precedence_two_stage_off_beats_main_two_stage_on(monkeypatch):
    # GHOST_VERIFY_TWO_STAGE=0 must disable the escalated attempt even
    # with the §4BK kill switch set — the master switch wins.
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
    monkeypatch.setenv("GHOST_VERIFY_MAIN_TWO_STAGE", "1")
    v = Verifier(llm_client=_Client())
    calls = []

    async def _llm(prompt, temperature=0.1, max_tokens=2048,
                   json_only=False, force_main=False):
        calls.append((json_only, force_main))
        return {"verdict": "CONFIRMED", "confidence": 0.95,
                "reasoning": "supported", "issues": []}

    monkeypatch.setattr(v, "_call_llm", _llm)
    out = await v._escalate_refute(_cheap_refute(), "c", "e", "ctx")
    assert calls == [(False, True)]
    assert out.verdict == VerifyVerdict.CONFIRMED
