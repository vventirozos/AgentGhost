"""Tests for the user-request alignment check in `verify_claim`.

Regression target: a real production trace where the user typed
"stop self play", the model emitted a `system_weather` tool call
instead of `stop_self_play`, and the verifier ran the
non-`execute`-path which routes through `verify_claim`. The old
prompt only asked "is this CLAIM supported by the EVIDENCE?", so it
returned CONFIRMED (100%) — the weather report was internally
consistent with the weather tool output, even though it had nothing
to do with what the user asked for.

These tests pin the new prompt's request-alignment check so that
- the rubric explicitly elevates "does the claim answer the user's
  request" above "is the claim supported by the evidence";
- the verify_claim API still accepts the existing positional/kwarg
  signature so callers don't need a code change;
- empty `context` still works (back-compat with older callers and
  with the `verify_claim` fallback path).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ghost_agent.core.verifier import (
    Verifier,
    VerifyVerdict,
    _VERIFY_CLAIM_PROMPT,
)


# ---------- prompt content invariants ----------


def test_claim_prompt_elevates_request_alignment():
    """The new prompt must put request-alignment FIRST, before
    evidence-support — otherwise an internally-consistent
    weather-report-vs-stop-self-play claim slips through."""
    rendered = _VERIFY_CLAIM_PROMPT.format(
        claim="weather is 20C in Athens",
        evidence="weather: 20C, Athens",
        context="stop self play",
    )
    assert "USER REQUEST" in rendered
    assert "Request alignment" in rendered
    # Wrong-question failure shape must be called out by name, with
    # concrete examples that mirror the production bug.
    rendered_lower = rendered.lower()
    assert "wrong-question" in rendered_lower or "off-topic" in rendered_lower
    # The prompt must explicitly say a true-but-off-topic claim is REFUTED,
    # not CONFIRMED — that's the rubber-stamp pattern we just fixed.
    assert (
        "true-but-off-topic" in rendered_lower
        or "even if the claim is internally consistent" in rendered_lower
    )


def test_claim_prompt_alignment_outranks_evidence_support():
    """Step 1 (alignment) must be the gate. If alignment fails,
    REFUTED regardless of evidence support."""
    rendered = _VERIFY_CLAIM_PROMPT.format(
        claim="x", evidence="y", context="z",
    )
    # Alignment must precede evidence support, structurally.
    align_idx = rendered.find("Request alignment")
    evidence_idx = rendered.find("Evidence support")
    assert align_idx >= 0 and evidence_idx >= 0
    assert align_idx < evidence_idx


def test_claim_prompt_handles_empty_context_for_back_compat():
    """Older callers / fallback paths may pass an empty context.
    The prompt must not blow up and must instruct the LLM to
    skip the alignment check rather than falsely refuting a
    no-context claim."""
    rendered = _VERIFY_CLAIM_PROMPT.format(
        claim="some claim", evidence="some evidence", context="",
    )
    # Must explicitly tell the LLM what to do when context is empty.
    rendered_lower = rendered.lower()
    assert "empty" in rendered_lower or "whitespace" in rendered_lower
    assert "skip this check" in rendered_lower or "proceed to step 2" in rendered_lower


# ---------- API contract ----------


@pytest.mark.asyncio
async def test_verify_claim_threads_user_request_into_prompt():
    """The verifier must put the user request into the prompt as
    USER REQUEST (the field the new alignment check reads)."""
    captured = {}

    class _Stub:
        async def chat_completion(self, payload):
            captured["prompt"] = payload["messages"][0]["content"]
            return {
                "choices": [{"message": {"content":
                    '{"verdict": "REFUTED", "confidence": 0.92, '
                    '"reasoning": "user asked to stop self-play, '
                    'agent reported the weather"}'
                }}]
            }

    v = Verifier(llm_client=_Stub())
    result = await v.verify_claim(
        claim="The current temperature in Athens is 20C.",
        evidence="weather: 20C, sunny, Athens, GR",
        context="stop self play",
    )
    assert result is not None
    assert result.verdict == VerifyVerdict.REFUTED
    # The user's actual request must appear in the rendered prompt
    # under the USER REQUEST slot — that's what the alignment check reads.
    assert "USER REQUEST" in captured["prompt"]
    assert "stop self play" in captured["prompt"]
    # And the agent's reply must be there as the CLAIM.
    assert "Athens is 20C" in captured["prompt"]


@pytest.mark.asyncio
async def test_verify_claim_back_compat_no_context_arg():
    """Two-arg `verify_claim(claim, evidence)` calls (test code,
    older fallback paths) must still work — no signature break."""
    class _Stub:
        async def chat_completion(self, payload):
            return {
                "choices": [{"message": {"content":
                    '{"verdict": "CONFIRMED", "confidence": 0.8, '
                    '"reasoning": "ok"}'
                }}]
            }

    v = Verifier(llm_client=_Stub())
    # Positional, no context — this is the legacy shape used in
    # tests/test_verifier_gate.py and several other suites.
    result = await v.verify_claim("2+2=4", "math")
    assert result is not None
    assert result.verdict == VerifyVerdict.CONFIRMED
