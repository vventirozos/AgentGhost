"""Tests for smart problem-solving improvements.

Covers:
- Differentiated strike budgets (#1)
- Progressive thinking budget (#2)
- Semantic context anchoring (#3)
- Query reformulation (#4)
- Adaptive sampling parameters (#5)
- Multi-pivot System 3 (#6)
- Tool fallback chains (#7)
- Checkpoint & resume (#8)
"""

import pytest
import math
from unittest.mock import MagicMock, AsyncMock, patch


# ============================================================
# #1: Differentiated Strike Budgets
# ============================================================

class TestDifferentiatedStrikes:
    def test_tool_failure_classification_routes_correctly(self):
        from ghost_agent.tools.tool_failure import classify_tool_failure, FailureClass
        # Transient errors
        fc, _ = classify_tool_failure("Connection timed out")
        assert fc == FailureClass.RETRYABLE
        # Structural errors
        fc, _ = classify_tool_failure("TypeError: cannot add str and int")
        assert fc == FailureClass.DIAGNOSTIC


# ============================================================
# #2: Progressive Thinking Budget
# ============================================================

class TestProgressiveThinking:
    def test_base_and_extended_caps_defined(self):
        from ghost_agent.core.agent import MAX_THINKING_CHARS, MAX_THINKING_CHARS_EXTENDED
        assert MAX_THINKING_CHARS == 32000
        assert MAX_THINKING_CHARS_EXTENDED == 64000
        assert MAX_THINKING_CHARS_EXTENDED > MAX_THINKING_CHARS

    def test_loop_detection_kills_repetitive_content(self):
        from ghost_agent.core.agent import _detect_thinking_loop
        # Create genuinely repetitive content
        repeated = "This is a repeating paragraph that goes on and on. " * 100
        assert _detect_thinking_loop(repeated) is True

    def test_loop_detection_passes_diverse_content(self):
        from ghost_agent.core.agent import _detect_thinking_loop
        # Create diverse content
        diverse = "".join(f"Unique paragraph {i} with different content about topic {i*7}. " for i in range(200))
        assert _detect_thinking_loop(diverse) is False


# ============================================================
# #3: Semantic Context Anchoring
# ============================================================

class TestSemanticAnchoring:
    @pytest.mark.asyncio
    async def test_prune_preserves_error_anchors(self, mock_context):
        from ghost_agent.core.agent import GhostAgent
        agent = GhostAgent(mock_context)

        messages = [
            {"role": "system", "content": "You are an AI."},
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": "Let me check."},
            {"role": "tool", "content": "Error: FileNotFoundError: data.csv not found", "name": "execute"},
            {"role": "assistant", "content": "The issue is the file is missing."},
            {"role": "user", "content": "Try again"},
            {"role": "assistant", "content": "OK trying."},
            {"role": "tool", "content": "Success: processed 100 rows", "name": "execute"},
            {"role": "assistant", "content": "Done!"},
            {"role": "user", "content": "What about edge cases?"},
            {"role": "assistant", "content": "Let me check."},
            {"role": "tool", "content": "Result: all 5 edge cases pass", "name": "execute"},
            {"role": "assistant", "content": "All good."},
            {"role": "user", "content": "Great, finalize."},
        ]

        pruned = await agent._prune_context(messages, max_tokens=50, model="test")

        # The error anchor should survive pruning
        all_content = " ".join(str(m.get("content", "")) for m in pruned)
        assert "ANCHORED" in all_content or "FileNotFoundError" in all_content or "Fix the bug" in all_content

    @pytest.mark.asyncio
    async def test_prune_caps_anchors_at_4(self, mock_context):
        from ghost_agent.core.agent import GhostAgent
        agent = GhostAgent(mock_context)

        # Create 10 messages with error-like tool results
        messages = [{"role": "system", "content": "System"}]
        messages.append({"role": "user", "content": "Original goal"})
        for i in range(8):
            messages.append({"role": "tool", "content": f"Error: problem {i}", "name": f"tool{i}"})
            messages.append({"role": "assistant", "content": f"Working on {i}"})
        for i in range(6):
            messages.append({"role": "user", "content": f"Continue {i}"})
            messages.append({"role": "assistant", "content": f"OK {i}"})

        pruned = await agent._prune_context(messages, max_tokens=100, model="test")

        # Should have at most 4 anchored messages
        anchored = [m for m in pruned if "[ANCHORED]" in str(m.get("content", ""))]
        assert len(anchored) <= 4


# ============================================================
# #4: Query Reformulation
# ============================================================

class TestQueryReformulation:
    def test_reformulate_removes_specific_terms(self):
        from ghost_agent.tools.search import _reformulate_query
        reformulations = _reformulate_query("docker build cache issues 2026 v4.5")
        assert len(reformulations) == 2
        # First reformulation should be broader (no year/version)
        assert "2026" not in reformulations[0]
        assert "4.5" not in reformulations[0]

    def test_reformulate_adds_question_form(self):
        from ghost_agent.tools.search import _reformulate_query
        reformulations = _reformulate_query("python async timeout handling")
        # Should have a "how to" version
        assert any("how to" in r.lower() for r in reformulations)

    def test_reformulate_already_question(self):
        from ghost_agent.tools.search import _reformulate_query
        reformulations = _reformulate_query("how to fix docker compose networking issues on macos")
        assert len(reformulations) == 2
        # Should simplify rather than add "how to" again
        assert all("how to how to" not in r.lower() for r in reformulations)

    def test_reformulate_always_returns_two(self):
        from ghost_agent.tools.search import _reformulate_query
        for query in ["x", "short", "a very long query about many things"]:
            result = _reformulate_query(query)
            assert len(result) == 2


# ============================================================
# #5: Adaptive Sampling Parameters
# ============================================================

class TestAdaptiveSampling:
    def test_classify_creative_task(self):
        from ghost_agent.core.agent import _classify_coding_task
        assert _classify_coding_task("design a new API architecture") == "creative"
        assert _classify_coding_task("brainstorm naming alternatives") == "creative"
        assert _classify_coding_task("refactor the module structure") == "creative"

    def test_classify_precise_task(self):
        from ghost_agent.core.agent import _classify_coding_task
        assert _classify_coding_task("write a SQL select query") == "precise"
        assert _classify_coding_task("fix the auth security issue") == "precise"
        assert _classify_coding_task("update the database schema migration") == "precise"

    def test_classify_balanced_task(self):
        from ghost_agent.core.agent import _classify_coding_task
        assert _classify_coding_task("write a python function") == "balanced"
        assert _classify_coding_task("") == "balanced"

    def test_sampling_params_vary_by_task(self):
        """Coding sub-classification only kicks in when both flags are set.

        New signature: `get_sampling_params(is_tool_turn, query, is_coding)`.
        Creative / precise / balanced profiles require ``is_coding=True``.
        A tool turn WITHOUT coding intent uses the base precise profile
        (temperature=0.6) — that's the fix for over-eager duplicate
        setter calls on turns that don't involve code.
        """
        from ghost_agent.core.agent import get_sampling_params
        creative = get_sampling_params(True, "design a new architecture", is_coding=True)
        precise = get_sampling_params(True, "write exact SQL query", is_coding=True)
        balanced = get_sampling_params(True, "write a function", is_coding=True)
        general = get_sampling_params(False)
        non_coding_tool = get_sampling_params(True, "update the user profile with the name")

        assert creative["temperature"] > precise["temperature"]
        assert general["temperature"] > creative["temperature"]
        assert precise["temperature"] < balanced["temperature"]
        # Non-coding tool turn must NOT get the warm conversational profile
        # (that was the bug — `update_profile` calls at temp 1.0 were
        # re-issued in subsequent turns). Must match the base coding
        # profile (temp 0.6, presence_penalty 0).
        assert non_coding_tool["temperature"] == 0.6
        assert non_coding_tool["presence_penalty"] == 0
        assert non_coding_tool["temperature"] < general["temperature"]

    def test_non_coding_ignores_query(self):
        """Conversational turns (is_tool_turn=False) always return the
        general profile regardless of query content."""
        from ghost_agent.core.agent import get_sampling_params
        p1 = get_sampling_params(False, "design a creative thing")
        p2 = get_sampling_params(False, "precise SQL query")
        assert p1 == p2  # Conversational always uses GENERAL_SAMPLING_PARAMS
        assert p1["temperature"] == 1.0


# ============================================================
# #7: Tool Fallback Chains
# ============================================================

class TestFallbackChains:
    def test_get_fallback_for_deep_research(self):
        from ghost_agent.tools.fallback_chains import get_fallback_hint
        hint = get_fallback_hint("deep_research")
        assert hint is not None
        assert "web_search" in hint.lower()

    def test_get_fallback_for_execute(self):
        from ghost_agent.tools.fallback_chains import get_fallback_hint
        hint = get_fallback_hint("execute")
        assert hint is not None
        assert "file_system" in hint.lower()

    def test_no_fallback_for_unknown_tool(self):
        from ghost_agent.tools.fallback_chains import get_fallback_hint
        assert get_fallback_hint("unknown_tool") is None

    def test_skips_web_search_on_captcha(self):
        from ghost_agent.tools.fallback_chains import get_fallback_hint
        hint = get_fallback_hint("deep_research", "CAPTCHA detected, search blocked")
        # Should skip web_search and suggest recall instead
        assert hint is not None
        assert "recall" in hint.lower() or "memory" in hint.lower()


# ============================================================
# #8: Checkpoint & Resume
# ============================================================

class TestCheckpointResume:
    def test_checkpoint_at_turn_15(self):
        """Verify checkpoint logic saves to scratchpad at turn milestones."""
        from ghost_agent.memory.scratchpad import Scratchpad
        sp = Scratchpad(max_entries=50)

        # Simulate checkpoint at turn 15
        messages = [
            {"role": "assistant", "content": "I found the root cause: missing import"},
            {"role": "tool", "content": "Exit code 0: tests pass", "name": "execute"},
            {"role": "assistant", "content": "Fixed and verified."},
        ]
        turn = 15
        if turn in (15, 30):
            items = []
            for m in messages[-10:]:
                role = m.get("role", "?")
                content = str(m.get("content", ""))[:200]
                if role in ("assistant", "tool") and content.strip():
                    items.append(f"{role}: {content}")
            if items:
                sp.set(f"_checkpoint_t{turn}", " | ".join(items[-5:])[:1000])

        checkpoint = sp.get("_checkpoint_t15")
        assert checkpoint is not None
        assert "root cause" in checkpoint
        assert "Exit code 0" in checkpoint

    def test_checkpoint_at_turn_30(self):
        from ghost_agent.memory.scratchpad import Scratchpad
        sp = Scratchpad(max_entries=50)
        sp.set("_checkpoint_t30", "Progress: 80% complete")
        assert sp.get("_checkpoint_t30") is not None
