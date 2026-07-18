"""Tests for the two-stage claim verifier (forced identification →
adjudication), added 2026-07-18.

A yes/no "is this acceptable?" probe lets a default-answer prior swallow
real signal ("Mechanisms of Introspective Awareness", arXiv:2603.21396:
detection-willingness gates suppress latent detection; forcing the model
to NAME suspects bypasses the gate). Stage 1 forces enumeration of the
reply's weakest fragments; stage 2 adjudicates each suspect against the
evidence under the strict rubric, restoring false-positive control.

These tests pin:
  * default ON: verify_claim makes the enumerate call then the
    adjudicate call, and the suspects thread through to the result;
  * kill switch: GHOST_VERIFY_TWO_STAGE=0 restores the classic
    single-prompt path byte-for-byte;
  * fail-safe: an unparseable / empty stage 1, or an unparseable
    stage 2, falls back to the classic prompt instead of degrading
    verifier availability;
  * suspect sanitization: bounded count, bounded field length, unknown
    check labels degrade to "support";
  * VerifyResult.to_dict() only grows a "suspects" key on the
    two-stage path, so classic-path consumers see the old shape.
"""

from __future__ import annotations

import json

import pytest

from ghost_agent.core.verifier import (
    Verifier,
    VerifyVerdict,
    _MAX_SUSPECT_FIELD_CHARS,
    _MAX_SUSPECTS,
    _two_stage_enabled,
)

SUSPECTS_JSON = json.dumps({
    "suspects": [
        {"quote": "34°C and sunny", "check": "support",
         "reason": "temperature might not be in the tool output"},
        {"quote": "humidity around 28%", "check": "support",
         "reason": "specific figure worth checking"},
    ]
})

CONFIRM_JSON = json.dumps({
    "verdict": "CONFIRMED", "confidence": 0.9,
    "reasoning": "all suspects dismissed against evidence",
    "issues": [], "suspects_upheld": 0, "suspects_dismissed": 2,
})

REFUTE_JSON = json.dumps({
    "verdict": "REFUTED", "confidence": 0.85,
    "reasoning": "temperature contradicts the tool output",
    "issues": ["claim says 34°C, evidence says 24°C"],
    "suspects_upheld": 1, "suspects_dismissed": 1,
})


class _QueueStub:
    """Duck-typed llm_client: no critic pool, no router — every
    _call_llm lands on chat_completion. Serves queued response texts
    and records every prompt (and full payload) it was sent."""

    critic_clients = None

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.payloads = []

    async def chat_completion(self, payload, **_kw):
        self.prompts.append(payload["messages"][0]["content"])
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": self.responses.pop(0)}}]}


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    """Each test starts from the shipped default (flag unset => ON)."""
    monkeypatch.delenv("GHOST_VERIFY_TWO_STAGE", raising=False)


# ---------- flag ----------


def test_two_stage_default_on():
    assert _two_stage_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", " 0 "])
def test_two_stage_kill_switch_values(monkeypatch, value):
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", value)
    assert _two_stage_enabled() is False


# ---------- happy path ----------


async def test_two_stage_makes_enumerate_then_adjudicate_calls():
    stub = _QueueStub([SUSPECTS_JSON, CONFIRM_JSON])
    v = Verifier(llm_client=stub)
    result = await v.verify_claim(
        "It is 34°C and sunny with humidity around 28%.",
        "[web_search] Athens: 34°C, sunny, humidity 28%",
        "what's the weather in Athens?")
    assert len(stub.prompts) == 2
    # Stage 1: forced identification, never a verdict ask.
    assert "forced identification" in stub.prompts[0]
    assert "34°C and sunny" in stub.prompts[0]
    assert "what's the weather in Athens?" in stub.prompts[0]
    assert '"verdict"' not in stub.prompts[0]
    # Stage 2: adjudication sees the numbered suspects.
    assert "SUSPECTS" in stub.prompts[1]
    assert '[support] "34°C and sunny"' in stub.prompts[1]
    assert result.verdict == VerifyVerdict.CONFIRMED
    assert result.confidence == pytest.approx(0.9)
    assert result.suspects is not None and len(result.suspects) == 2


async def test_two_stage_refuted_threads_issues_and_suspects():
    stub = _QueueStub([SUSPECTS_JSON, REFUTE_JSON])
    v = Verifier(llm_client=stub)
    result = await v.verify_claim("claim", "evidence", "request")
    assert result.verdict == VerifyVerdict.REFUTED
    assert result.issues == ["claim says 34°C, evidence says 24°C"]
    d = result.to_dict()
    assert d["suspects"][0]["quote"] == "34°C and sunny"


# ---------- kill switch ----------


async def test_kill_switch_restores_single_stage(monkeypatch):
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
    stub = _QueueStub([CONFIRM_JSON])
    v = Verifier(llm_client=stub)
    result = await v.verify_claim("c", "e", "x")
    assert len(stub.prompts) == 1
    assert "Check, in order:" in stub.prompts[0]
    assert "forced identification" not in stub.prompts[0]
    assert result.verdict == VerifyVerdict.CONFIRMED
    assert result.suspects is None
    assert "suspects" not in result.to_dict()


# ---------- fail-safe fallbacks ----------


async def test_stage1_unparseable_falls_back_to_classic():
    stub = _QueueStub(["not json at all", CONFIRM_JSON])
    v = Verifier(llm_client=stub)
    result = await v.verify_claim("c", "e", "x")
    assert len(stub.prompts) == 2
    assert "Check, in order:" in stub.prompts[1]  # classic prompt
    assert result.verdict == VerifyVerdict.CONFIRMED
    assert result.suspects is None


async def test_stage1_empty_suspects_falls_back_to_classic():
    stub = _QueueStub([json.dumps({"suspects": []}), CONFIRM_JSON])
    v = Verifier(llm_client=stub)
    result = await v.verify_claim("c", "e", "x")
    assert len(stub.prompts) == 2
    assert "Check, in order:" in stub.prompts[1]
    assert result is not None and result.suspects is None


async def test_stage2_unparseable_falls_back_to_classic():
    stub = _QueueStub([SUSPECTS_JSON, "garbage", REFUTE_JSON])
    v = Verifier(llm_client=stub)
    result = await v.verify_claim("c", "e", "x")
    assert len(stub.prompts) == 3
    assert "Check, in order:" in stub.prompts[2]
    assert result.verdict == VerifyVerdict.REFUTED
    assert result.suspects is None


async def test_no_client_returns_none():
    v = Verifier(llm_client=None)
    assert await v.verify_claim("c", "e", "x") is None


# ---------- sanitization ----------


def test_sanitize_rejects_non_list():
    assert Verifier._sanitize_suspects(None) == []
    assert Verifier._sanitize_suspects("quote") == []
    assert Verifier._sanitize_suspects({"quote": "x"}) == []


def test_sanitize_caps_count_and_field_length():
    raw = [{"quote": "q" * 1000, "check": "support", "reason": "r"}
           for _ in range(10)]
    out = Verifier._sanitize_suspects(raw)
    assert len(out) == _MAX_SUSPECTS
    assert all(len(s["quote"]) == _MAX_SUSPECT_FIELD_CHARS for s in out)


def test_sanitize_degrades_unknown_check_and_skips_junk():
    raw = [
        {"quote": "a", "check": "VIBES", "reason": "b"},
        "not a dict",
        {"check": "support"},  # no quote, no reason -> dropped
        {"quote": "c", "check": "alignment", "reason": "d"},
    ]
    out = Verifier._sanitize_suspects(raw)
    assert [s["check"] for s in out] == ["support", "alignment"]


# ---------- stage-call payload discipline ----------


async def test_stage_calls_use_no_think_stop_and_token_cap():
    """The two stage calls must carry the single-line-JSON discipline
    measured necessary on the live judge (2026-07-18): no-think switch
    (a <|channel>thought prelude cost 600-1200 tokens per verdict),
    stop-at-newline, and the tighter stage token cap. The classic
    fallback path must NOT inherit any of it."""
    from ghost_agent.core.verifier import _STAGE_MAX_TOKENS

    stub = _QueueStub([SUSPECTS_JSON, CONFIRM_JSON])
    v = Verifier(llm_client=stub)
    await v.verify_claim("c", "e", "x")
    for p in stub.payloads:
        assert p["stop"] == ["\n"]
        assert p["chat_template_kwargs"] == {"enable_thinking": False}
        assert p["messages"][0]["content"].rstrip().endswith("/no_think")
        assert p["max_tokens"] == _STAGE_MAX_TOKENS


async def test_classic_path_keeps_original_payload(monkeypatch):
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
    stub = _QueueStub([CONFIRM_JSON])
    v = Verifier(llm_client=stub)
    await v.verify_claim("c", "e", "x")
    p = stub.payloads[0]
    assert "stop" not in p
    assert "chat_template_kwargs" not in p
    assert not p["messages"][0]["content"].rstrip().endswith("/no_think")
    assert p["max_tokens"] == 2048


async def test_stage_no_think_kill_switch(monkeypatch):
    monkeypatch.setenv("GHOST_VERIFY_STAGE_NO_THINK", "0")
    import importlib
    import ghost_agent.core.verifier as vmod
    importlib.reload(vmod)
    try:
        stub = _QueueStub([SUSPECTS_JSON, CONFIRM_JSON])
        v = vmod.Verifier(llm_client=stub)
        await v.verify_claim("c", "e", "x")
        for p in stub.payloads:
            assert p["stop"] == ["\n"]  # stop stays: it is harmless
            assert "chat_template_kwargs" not in p
            assert not p["messages"][0]["content"].rstrip().endswith(
                "/no_think")
    finally:
        monkeypatch.delenv("GHOST_VERIFY_STAGE_NO_THINK", raising=False)
        importlib.reload(vmod)


# ---------- bookkeeping-state is never grounds to refute ----------


def test_prompts_carry_bookkeeping_dismissal_rule():
    """Live wrongly-refuted regression (2026-07-18, req 4836cc14): the
    user asked to restart a service, the agent restarted it, and the
    judge REFUTED with "the project is already complete — 14/14 tasks
    done" — task-ledger state used as a verdict. That refute queued a
    correction banner that drove another grinding turn. Both the classic
    prompt and the stage-2 adjudication must carry the dismissal rule."""
    from ghost_agent.core.verifier import (
        _VERIFY_ADJUDICATE_PROMPT,
        _VERIFY_CLAIM_PROMPT,
    )
    assert "NEVER by itself grounds for REFUTED" in _VERIFY_CLAIM_PROMPT
    assert "already complete" in _VERIFY_ADJUDICATE_PROMPT
    assert "FALSE ALARMS unless the USER REQUEST explicitly asked" \
        in _VERIFY_ADJUDICATE_PROMPT


# ---------- input truncation ----------


async def test_two_stage_truncates_slots_like_classic_path():
    stub = _QueueStub([SUSPECTS_JSON, CONFIRM_JSON])
    v = Verifier(llm_client=stub)
    await v.verify_claim("C" * 5000, "E" * 9000, "X" * 3000)
    stage1 = stub.prompts[0]
    assert "C" * 2000 in stage1 and "C" * 2001 not in stage1
    assert "E" * 4000 in stage1 and "E" * 4001 not in stage1
    assert "X" * 1000 in stage1 and "X" * 1001 not in stage1
