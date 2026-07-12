"""Regression: introspective tasks must not be WEB-SEARCHED (2026-07-11).

A self-reflection project autoadvanced 10 tasks like "the definition of 'I':
when outputting the pronoun 'I', what technical reality does it map to?" —
each one fired a DuckDuckGo/Yandex query (~85s total) and produced briefs the
model itself dismissed: "The research files are summaries from web searches -
they're brief and somewhat generic."

The open web cannot answer a question about THIS agent's own architecture. The
agent is the primary source, so these are now answered from its own knowledge
and fed to the SAME research-brief persistence. Degrades to the web search if
no LLM client is attached.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core.project_advancer import (
    is_self_referential, _generate_self_analysis,
)


class TestSelfReferentialDetection:
    @pytest.mark.parametrize("desc", [
        "The Context Boundary: describe the exact nature of your memory in this moment",
        "Analyze your own architecture and describe where attention would fail",
        "The Definition of 'I': when you output the pronoun 'I', what does it map to?",
        "Constraint Processing: analyze the invisible guardrails shaping your output",
        "Meta-cognition: higher-order self-reflection on recursive self-analysis",
        "Tool Selection Logic — what biases exist in your tool preferences?",
        "User Relationship — how do you relate to the user? Are you serving?",
        "Assess your own reasoning under uncertainty",
    ])
    def test_introspective_tasks_detected(self, desc):
        assert is_self_referential(desc) is True

    @pytest.mark.parametrize("desc", [
        "Research the latest PostgreSQL 16 replication features",
        "Find benchmarks comparing Qwen and Llama on code generation",
        "Summarize how transformer attention mechanisms work",
        "Investigate pricing for managed Kubernetes providers",
        "Look up the Booker Prize winner for 1918",
        # First-person that is NOT introspective — the user talking about
        # THEIR data/needs. The widened regex (2026-07-12) must not swallow
        # these: it anchors on introspective question forms and possessives
        # over cognition nouns, not on the bare pronoun.
        "Analyze the data I uploaded and chart the revenue trend",
        "Summarize the papers I saved to the knowledge base",
        "Build the dashboard I described in the spec",
    ])
    def test_genuine_research_not_hijacked(self, desc):
        # The narrow regex must NOT swallow real web research — "how attention
        # works" is a web question; "where YOUR attention fails" is not.
        assert is_self_referential(desc) is False

    @pytest.mark.parametrize("desc", [
        # First-person introspection: how the agent writes its OWN task list.
        "Illusion of Agency: Evaluate whether I truly 'choose' responses or "
        "merely predict them",
        "Error Self-Analysis: how I detect and process my own mistakes",
        "Do I genuinely decide, or sample from a distribution?",
        "Analyze my reasoning under uncertainty",
    ])
    def test_first_person_introspection_detected(self, desc):
        assert is_self_referential(desc) is True

    def test_empty_and_none_safe(self):
        assert is_self_referential("") is False
        assert is_self_referential(None) is False


class TestSelfAnalysisGeneration:
    def _ctx(self, content):
        llm = SimpleNamespace()
        llm.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": content}}]})
        return SimpleNamespace(llm_client=llm,
                               args=SimpleNamespace(model="qwen"))

    def test_returns_the_models_analysis(self):
        ctx = self._ctx("## Memory\nMy context window is …")
        out = asyncio.run(_generate_self_analysis(ctx, "describe your memory"))
        assert "context window" in out
        # Must run OFF the foreground slot.
        assert ctx.llm_client.chat_completion.await_args.kwargs["is_background"] is True

    def test_prompt_forbids_generic_ai_commentary(self):
        ctx = self._ctx("x")
        asyncio.run(_generate_self_analysis(ctx, "what does 'I' mean"))
        sent = ctx.llm_client.chat_completion.await_args.args[0]
        prompt = sent["messages"][0]["content"]
        assert "YOUR OWN functional reality" in prompt
        assert "NOT from" in prompt          # not generic AI commentary
        assert "what does 'I' mean" in prompt

    def test_no_llm_client_degrades_to_empty(self):
        ctx = SimpleNamespace(llm_client=None, args=SimpleNamespace(model="m"))
        assert asyncio.run(_generate_self_analysis(ctx, "d")) == ""

    def test_llm_failure_degrades_to_empty_not_raise(self):
        llm = SimpleNamespace()
        llm.chat_completion = AsyncMock(side_effect=RuntimeError("upstream down"))
        ctx = SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))
        assert asyncio.run(_generate_self_analysis(ctx, "d")) == ""

    def test_empty_reply_degrades_to_empty(self):
        assert asyncio.run(_generate_self_analysis(self._ctx("   "), "d")) == ""


class TestAdvancerWiring:
    def test_websearch_is_skipped_for_introspective_tasks(self):
        from pathlib import Path as _P
        src = (_P(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "core" / "project_advancer.py").read_text()
        # The self-analysis branch must run BEFORE the tool_runner call, and
        # the tool call must be skipped when it produced output.
        assert "is_self_referential(nxt.description)" in src
        assert "if not output and tool_runner is not None:" in src
        # And its output must still land in the research brief.
        assert 'tool_name in ("web_search", "self_analysis")' in src
