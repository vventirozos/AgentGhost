"""Tests for the 2026-05-17 lesson-extractor redesign (Task #14).

The pre-fix extractor used a single prompt with HARD RULES that
forced `confidence=0` whenever any of four constraints were violated,
producing ~0 lessons saved per 100 cycles. The new design routes by
outcome (STRUGGLED_THEN_WON / FAILED / NOVEL_SHAPE / FIRST_TRY_SUCCESS)
and patches partial LLM responses with a templated fallback so a
genuine-signal cycle never produces nothing.

These tests pin both halves of the redesign:
  * `_build_extractor_prompt` — outcome-specific prompt content
  * `_patch_with_fallback` — templated lesson synthesis when the
    LLM returns partial / empty
  * the dispatcher in `_extract_structured_lesson` — outcome routing
    by (passed, attempt, novelty)
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.dream import (
    _build_extractor_prompt,
    _patch_with_fallback,
    Dreamer,
)


# ---------------------------------------------------------------------------
# _build_extractor_prompt — outcome-specific wording
# ---------------------------------------------------------------------------


class TestBuildExtractorPrompt:
    def _common(self, outcome, **overrides):
        kwargs = dict(
            outcome=outcome,
            cluster_key="data_analysis",
            status_str="SUCCESS (in 2 attempts)" if outcome == "STRUGGLED_THEN_WON" else "FAILURE",
            challenge="parse a CSV file...",
            validation_script="import subprocess\n",
            transcript="agent ran some tools\n",
            attempt=1 if outcome == "STRUGGLED_THEN_WON" else 0,
            solution_novelty=None,
        )
        kwargs.update(overrides)
        return _build_extractor_prompt(**kwargs)

    def test_struggled_then_won_emphasises_debugging_insight(self):
        p = self._common("STRUGGLED_THEN_WON", attempt=1)
        assert "DEBUGGING INSIGHT" in p
        # Quotes the actual attempt number so the LLM grounds the lesson
        # in the real retry, not a generic "you retried".
        assert "attempt 2" in p
        # Sets the no-pre-emptive-empty bias.
        assert "produce a non-empty answer" in p.lower()

    def test_failed_emphasises_error_pattern(self):
        p = self._common("FAILED")
        assert "ERROR PATTERN" in p
        assert "warning sign" in p
        # No "produce non-empty answer" override here — failures are
        # noisier and we DO want the LLM to skip when the transcript is
        # uninformative.

    def test_novel_shape_is_observational(self):
        p = self._common("NOVEL_SHAPE", solution_novelty=0.78)
        assert "ALTERNATIVE APPROACH" in p
        # Novelty score should appear in the header.
        assert "0.78" in p
        # observational tone: anti_pattern allowed empty
        assert "anti_pattern" in p
        assert "can be empty" in p.lower() or "may be empty" in p.lower()

    def test_first_try_success_keeps_strict_bar(self):
        p = self._common("FIRST_TRY_SUCCESS")
        # The strict path: only extract if it really adds something new.
        assert "transferable insight" in p.lower()
        assert "return empty fields" in p.lower()

    def test_hard_rules_language_gone(self):
        """The old prompt's 'HARD RULES ... MUST make you return
        confidence=0' framing was the root cause of mass-empty
        responses. It MUST NOT reappear in any variant."""
        for outcome in ("STRUGGLED_THEN_WON", "FAILED", "NOVEL_SHAPE", "FIRST_TRY_SUCCESS"):
            p = self._common(outcome)
            assert "HARD RULES" not in p
            assert "MUST make you return confidence=0" not in p

    def test_shared_guidelines_phrased_as_preference(self):
        """The four quality bars survive — but as guidance, not gates."""
        p = self._common("STRUGGLED_THEN_WON")
        assert "preference, not gates" in p
        assert "CLASS of task" in p
        assert "fenced" in p  # python code-block recommendation

    def test_novelty_score_surfaces_in_header_when_provided(self):
        p = self._common("NOVEL_SHAPE", solution_novelty=0.42)
        assert "0.42" in p

    def test_no_novelty_omits_header_line(self):
        p = self._common("FAILED", solution_novelty=None)
        assert "Solution novelty" not in p


# ---------------------------------------------------------------------------
# _patch_with_fallback — templated lesson synthesis
# ---------------------------------------------------------------------------


class TestPatchWithFallback:
    def test_no_patch_when_already_viable(self):
        existing = {"trigger": "real trigger", "correct_pattern": "real pattern"}
        out = _patch_with_fallback(
            existing,
            outcome="STRUGGLED_THEN_WON",
            cluster_key="sql",
            challenge="x",
            attempt=1,
            solution_novelty=None,
        )
        assert out["trigger"] == "real trigger"
        assert out["correct_pattern"] == "real pattern"
        assert "fallback_synthesized" not in out

    def test_patches_empty_struggled_then_won(self):
        out = _patch_with_fallback(
            {},
            outcome="STRUGGLED_THEN_WON",
            cluster_key="sql",
            challenge="x",
            attempt=1,
            solution_novelty=None,
        )
        assert out["trigger"]
        assert out["correct_pattern"]
        assert "sql" in out["trigger"]
        assert out["fallback_synthesized"] is True
        # Conservative confidence so real LLM lessons rank higher.
        assert out["confidence"] == 0.30

    def test_patches_empty_failure(self):
        out = _patch_with_fallback(
            {},
            outcome="FAILED",
            cluster_key="regex_parse",
            challenge="x",
            attempt=2,
            solution_novelty=None,
        )
        assert "exhaust" in out["trigger"] or "regex_parse" in out["trigger"]
        assert out["correct_pattern"]
        assert out["confidence"] == 0.30

    def test_patches_empty_novel_shape(self):
        out = _patch_with_fallback(
            {},
            outcome="NOVEL_SHAPE",
            cluster_key="python_general",
            challenge="x",
            attempt=0,
            solution_novelty=0.78,
        )
        assert "0.78" in out["trigger"]
        assert out["correct_pattern"]
        # Novel-shape is observational — anti_pattern may legitimately be empty.
        assert out["anti_pattern"] == ""

    def test_does_not_patch_first_try_success(self):
        out = _patch_with_fallback(
            {},
            outcome="FIRST_TRY_SUCCESS",
            cluster_key="bash",
            challenge="x",
            attempt=0,
            solution_novelty=0.0,
        )
        # First-try wins on familiar shapes really have nothing to teach.
        assert out == {}

    def test_preserves_llm_partial_trigger(self):
        out = _patch_with_fallback(
            {"trigger": "LLM said this"},
            outcome="STRUGGLED_THEN_WON",
            cluster_key="sql",
            challenge="x",
            attempt=1,
            solution_novelty=None,
        )
        assert out["trigger"] == "LLM said this"  # not overwritten
        assert out["correct_pattern"]  # but the missing pattern is filled

    def test_domains_default_from_cluster(self):
        out = _patch_with_fallback(
            {},
            outcome="FAILED",
            cluster_key="sql",
            challenge="x",
            attempt=2,
            solution_novelty=None,
        )
        assert "sql" in out["domains"]

    def test_unknown_cluster_yields_empty_domains(self):
        out = _patch_with_fallback(
            {},
            outcome="FAILED",
            cluster_key="not_a_known_cluster",
            challenge="x",
            attempt=2,
            solution_novelty=None,
        )
        assert out["domains"] == []


# ---------------------------------------------------------------------------
# Dispatcher routing in _extract_structured_lesson
# ---------------------------------------------------------------------------


class TestExtractorRouting:
    """The dispatcher picks an outcome category and builds the right
    prompt before calling the LLM. We mock the LLM to capture the
    prompt and assert routing without touching the network."""

    def _make_dreamer(self, llm_reply: dict):
        ctx = MagicMock()
        ctx.llm_client = MagicMock()
        ctx.llm_client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": __import__("json").dumps(llm_reply)}}]
        })
        ctx.memory_system = MagicMock()
        d = Dreamer(ctx)
        return d, ctx

    def _capture_prompt(self, ctx) -> str:
        """Extract the user prompt sent to chat_completion."""
        payload = ctx.llm_client.chat_completion.await_args.args[0]
        for m in payload["messages"]:
            if m.get("role") == "user":
                return m["content"]
        return ""

    @pytest.mark.asyncio
    async def test_passed_attempt0_low_novelty_routes_to_first_try_success(self):
        d, ctx = self._make_dreamer({"trigger": "", "correct_pattern": ""})
        out = await d._extract_structured_lesson(
            model_name="x", challenge="c", validation_script="v",
            transcript="t", status_str="s", attempt=0, passed=True,
            cluster_key="sql", solution_novelty=0.1,
        )
        prompt = self._capture_prompt(ctx)
        assert "FIRST_TRY_SUCCESS" in prompt
        # First-try with low novelty: no fallback fabricated.
        assert not out.get("fallback_synthesized")

    @pytest.mark.asyncio
    async def test_passed_attempt0_high_novelty_routes_to_novel_shape(self):
        d, ctx = self._make_dreamer({"trigger": "", "correct_pattern": ""})
        out = await d._extract_structured_lesson(
            model_name="x", challenge="c", validation_script="v",
            transcript="t", status_str="s", attempt=0, passed=True,
            cluster_key="sql", solution_novelty=0.8,
        )
        prompt = self._capture_prompt(ctx)
        assert "NOVEL_SHAPE" in prompt
        # Novel-shape: fallback fires because LLM returned empty.
        assert out.get("fallback_synthesized") is True

    @pytest.mark.asyncio
    async def test_passed_attempt1_routes_to_struggled_then_won(self):
        d, ctx = self._make_dreamer({"trigger": "", "correct_pattern": ""})
        out = await d._extract_structured_lesson(
            model_name="x", challenge="c", validation_script="v",
            transcript="t", status_str="s", attempt=1, passed=True,
            cluster_key="data_analysis", solution_novelty=0.0,
        )
        prompt = self._capture_prompt(ctx)
        assert "STRUGGLED_THEN_WON" in prompt
        # Struggled path: fallback fires when LLM returns empty.
        assert out.get("fallback_synthesized") is True
        # Concrete cluster name in the templated lesson.
        assert "data_analysis" in (out.get("trigger") or "")

    @pytest.mark.asyncio
    async def test_failed_routes_to_failed(self):
        d, ctx = self._make_dreamer({"trigger": "", "correct_pattern": ""})
        out = await d._extract_structured_lesson(
            model_name="x", challenge="c", validation_script="v",
            transcript="t", status_str="FAILURE", attempt=2, passed=False,
            cluster_key="regex_parse", solution_novelty=None,
        )
        prompt = self._capture_prompt(ctx)
        assert "FAILED" in prompt
        assert out.get("fallback_synthesized") is True

    @pytest.mark.asyncio
    async def test_real_llm_response_overrides_fallback(self):
        d, ctx = self._make_dreamer({
            "trigger": "real LLM trigger",
            "correct_pattern": "real LLM pattern",
            "anti_pattern": "real LLM anti",
            "domains": ["sql"],
            "confidence": 0.8,
        })
        out = await d._extract_structured_lesson(
            model_name="x", challenge="c", validation_script="v",
            transcript="t", status_str="s", attempt=1, passed=True,
            cluster_key="sql", solution_novelty=0.0,
        )
        assert out["trigger"] == "real LLM trigger"
        assert out["correct_pattern"] == "real LLM pattern"
        # No fabrication needed.
        assert not out.get("fallback_synthesized")
        # Back-compat mirrors still populate.
        assert out["task"] == "real LLM trigger"
        assert out["solution"] == "real LLM pattern"

    @pytest.mark.asyncio
    async def test_llm_exception_still_yields_fallback_on_struggled(self):
        """If the LLM call raises (timeout, malformed JSON, network),
        a struggled-then-won cycle still gets a templated lesson —
        the genuine signal isn't lost just because the meta-LLM
        misbehaved."""
        d, ctx = self._make_dreamer({})  # ignored
        ctx.llm_client.chat_completion = AsyncMock(side_effect=Exception("upstream 503"))
        out = await d._extract_structured_lesson(
            model_name="x", challenge="c", validation_script="v",
            transcript="t", status_str="s", attempt=2, passed=True,
            cluster_key="sql", solution_novelty=0.0,
        )
        assert out.get("fallback_synthesized") is True
        assert out["trigger"]
        assert out["correct_pattern"]
