"""Tests for the verifier's user-request alignment check.

The previous verifier prompt asked only "does the OUTPUT contain
the information the user asked for?". That gate could not catch the
dominant rubber-stamp failure shape — user asks for code, agent
runs the code itself and returns the *result* — because the result
is internally consistent with the tool output even though it's not
what the user asked for.

The new prompt asks the verifier to first audit constraint
satisfaction (form/format requested by the user) and only then
audit output correctness. These tests pin:

  * the prompt now includes both the agent's response slot and an
    explicit "constraint satisfaction" rubric;
  * the call site passes the response through;
  * back-compat: callers that don't pass `response` still work and
    the prompt fills in a sentinel string;
  * the prompt rendering doesn't crash on edge inputs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.verifier import (
    Verifier,
    VerifyVerdict,
    _VERIFY_CODE_PROMPT,
)


# ---------- prompt content invariants ----------


def test_prompt_includes_response_slot():
    """The new prompt must surface the agent's user-facing response,
    not just the tool output."""
    rendered = _VERIFY_CODE_PROMPT.format(
        intent="x", code="y", output="z", response="r",
    )
    assert "AGENT'S RESPONSE TO THE USER" in rendered
    assert "r" in rendered  # response actually rendered


def test_prompt_includes_constraint_satisfaction_rubric():
    """The first check must be the wording-constraint check; if it
    falls back to bullet 2+ (output correctness), the wrong-question
    failure shape goes unnoticed."""
    rendered = _VERIFY_CODE_PROMPT.format(
        intent="x", code="y", output="z", response="r",
    )
    assert "Constraint satisfaction" in rendered
    # Specific failure-shape examples must be enumerated so the LLM
    # has concrete patterns to match against.
    assert "give me the code" in rendered.lower()
    assert "agent answered a different question" in rendered.lower() or \
           "different question than the one asked" in rendered.lower()


def test_prompt_includes_result_deliverable_exception():
    """When the user asks to "write/run a script ... and tell me the
    integer", the deliverable is the RESULT, not the source. The prompt
    must carry the exception so a correct result without a code fence is
    not REFUTED on the fence alone (the over-eager-repair regression)."""
    rendered = _VERIFY_CODE_PROMPT.format(
        intent="write a script to compute 15! and tell me the integer",
        code="y", output="z", response="r",
    )
    low = rendered.lower()
    assert "exception" in low
    assert "method, not the deliverable" in low
    # The disambiguating guidance must be explicit.
    assert "prefer confirmed" in low


def test_prompt_requires_both_for_confirmed():
    """A CONFIRMED verdict must require BOTH output soundness AND
    response alignment. If either fails, REFUTED."""
    rendered = _VERIFY_CODE_PROMPT.format(
        intent="x", code="y", output="z", response="r",
    )
    # The exact wording: "A verdict of CONFIRMED requires BOTH ..."
    assert "CONFIRMED requires BOTH" in rendered


# ---------- API contract ----------


@pytest.mark.asyncio
async def test_verify_code_output_passes_response_to_prompt():
    """The verifier must thread the `response` kwarg into the LLM
    prompt — without that, the rubric is useless."""
    captured = {}

    class _Stub:
        async def chat_completion(self, payload):
            captured["prompt"] = payload["messages"][0]["content"]
            return {
                "choices": [{"message": {"content":
                    '{"verdict": "REFUTED", "confidence": 0.9, '
                    '"reasoning": "user asked for code, got a number"}'
                }}]
            }

    v = Verifier(llm_client=_Stub())
    result = await v.verify_code_output(
        code="find . -name '*.py' | xargs wc -l",
        output="1623 total",
        intent="just give me the code to count lines",
        response="The project has 1,623 lines of code.",
    )
    assert result is not None
    assert result.verdict == VerifyVerdict.REFUTED
    assert "AGENT'S RESPONSE TO THE USER" in captured["prompt"]
    assert "1,623 lines of code" in captured["prompt"]
    assert "just give me the code" in captured["prompt"]


@pytest.mark.asyncio
async def test_verify_code_output_back_compat_no_response_arg():
    """Older callers that don't pass `response` must still work —
    the prompt fills in a clear sentinel so the verifier can tell
    "we don't have the response" apart from "the response was
    empty"."""
    captured = {}

    class _Stub:
        async def chat_completion(self, payload):
            captured["prompt"] = payload["messages"][0]["content"]
            return {
                "choices": [{"message": {"content":
                    '{"verdict": "UNCERTAIN", "confidence": 0.5, '
                    '"reasoning": "no response provided"}'
                }}]
            }

    v = Verifier(llm_client=_Stub())
    result = await v.verify_code_output(
        code="x",
        output="y",
        intent="z",
    )
    assert result is not None
    assert "(response not provided to verifier)" in captured["prompt"]


@pytest.mark.asyncio
async def test_verify_code_output_truncates_long_response():
    """Defensive: very long responses must not blow the prompt
    budget. Confirm the slot is bounded."""
    captured = {}

    class _Stub:
        async def chat_completion(self, payload):
            captured["prompt"] = payload["messages"][0]["content"]
            return {
                "choices": [{"message": {"content":
                    '{"verdict": "CONFIRMED", "confidence": 0.9, '
                    '"reasoning": "ok"}'
                }}]
            }

    v = Verifier(llm_client=_Stub())
    long_resp = "X" * 50000
    await v.verify_code_output(
        code="c", output="o", intent="i", response=long_resp,
    )
    # The prompt must contain SOME of the response but not all 50k.
    assert "X" in captured["prompt"]
    assert len(captured["prompt"]) < 30000


# ---------- failure-shape regression ----------


@pytest.mark.asyncio
async def test_wrong_question_shape_can_be_refuted():
    """End-to-end shape test: simulate the exact 12:04 trace failure
    and confirm the new prompt format gives the verifier enough
    signal to refute. We use a fake LLM that *honestly applies* the
    rubric — i.e. it returns REFUTED iff the prompt explicitly tells
    it that's the right verdict for "user asks for code, agent gives
    a number". This exercises the prompt format, not LLM
    intelligence."""

    class _RubricFollower:
        """Stand-in for an LLM that follows the constraint-rubric
        instruction literally. Returns REFUTED when the prompt
        mentions the user asked for code AND the agent gave a
        number."""
        async def chat_completion(self, payload):
            text = payload["messages"][0]["content"]
            # Simulate a verifier that reads the rubric and applies it.
            asks_for_code = (
                "give me the code" in text.lower()
                or "just give me the code" in text.lower()
            )
            response_is_number = "1,623 lines of code" in text
            if asks_for_code and response_is_number:
                verdict = "REFUTED"
                conf = 0.9
                reasoning = (
                    "user asked for the code/snippet, agent reported a "
                    "result instead"
                )
            else:
                verdict = "CONFIRMED"
                conf = 0.5
                reasoning = "ok"
            return {
                "choices": [{"message": {"content":
                    f'{{"verdict": "{verdict}", "confidence": {conf}, '
                    f'"reasoning": "{reasoning}"}}'
                }}]
            }

    v = Verifier(llm_client=_RubricFollower())
    result = await v.verify_code_output(
        code="find . -name '*.py' -exec wc -l {} +",
        output="1623 total",
        intent="how can i see how many lines of code is a project ? "
               "just give me the code.",
        response="The project has 1,623 lines of code.",
    )
    assert result is not None
    assert result.verdict == VerifyVerdict.REFUTED, (
        f"new prompt should let the verifier refute the "
        f"wrong-question shape; got {result}"
    )
