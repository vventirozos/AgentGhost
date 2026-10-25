"""Per-task thinking-budget classifier.

The default (tight) is the historical 5-sentence anti-paralysis cap.
Extended fires when the query involves genuine multi-step reasoning —
debugging, algorithm design, proofs, SQL tuning — where the anti-
paralysis rule's "no alternatives" clause prevents legitimate derivation.
§4NM: the SELFPLAY tier (and its content tests) went with synthetic self-play.
"""

from ghost_agent.core.agent import (
    classify_thinking_budget,
    render_think_budget_guidance,
)
from ghost_agent.core.prompts import (
    QWEN_TOOL_PROMPT,
    THINK_BUDGET_TIGHT,
    THINK_BUDGET_EXTENDED,
)


class TestBudgetClassifier:
    def test_empty_query_is_tight(self):
        assert classify_thinking_budget("") == "tight"

    def test_greeting_is_tight(self):
        assert classify_thinking_budget("hi there") == "tight"
        assert classify_thinking_budget("thanks!") == "tight"

    def test_simple_fact_lookup_is_tight(self):
        assert classify_thinking_budget("what time is it") == "tight"
        assert classify_thinking_budget("when was IBM founded") == "tight"

    def test_meta_tasks_stay_tight(self):
        # Title/caption/rename queries — even if they have reasoning words,
        # they're fundamentally single-shot.
        assert classify_thinking_budget(
            "analyze this and explain why",
            is_meta_task=True,
        ) == "tight"

    def test_debug_query_is_extended(self):
        # Two debug-reasoning keywords → extended.
        assert classify_thinking_budget(
            "debug this traceback and explain why the optimize pass is slow"
        ) == "extended"

    def test_sql_tuning_is_extended(self):
        assert classify_thinking_budget(
            "explain analyze this query and optimize the CTE"
        ) == "extended"

    def test_algorithm_design_is_extended(self):
        assert classify_thinking_budget(
            "analyze the complexity of this algorithm and prove correctness"
        ) == "extended"

    def test_coding_with_single_strong_keyword_is_extended(self):
        assert classify_thinking_budget(
            "refactor this for clarity",
            has_coding_intent=True,
        ) == "extended"

    def test_coding_without_strong_keyword_stays_tight(self):
        # Vanilla "write a script" doesn't need extended.
        assert classify_thinking_budget(
            "write a hello world script",
            has_coding_intent=True,
        ) == "tight"


class TestRenderGuidance:
    def test_tight_maps_to_tight_block(self):
        assert render_think_budget_guidance("tight") == THINK_BUDGET_TIGHT

    def test_extended_maps_to_extended_block(self):
        assert render_think_budget_guidance("extended") == THINK_BUDGET_EXTENDED

    def test_selfplay_tier_is_gone_and_falls_through_to_tight(self):
        # §4NM: the SELFPLAY tier left with synthetic self-play; the old
        # tier name now takes the unknown-tier path.
        assert render_think_budget_guidance("selfplay") == THINK_BUDGET_TIGHT

    def test_unknown_falls_back_to_tight(self):
        assert render_think_budget_guidance("whatever") == THINK_BUDGET_TIGHT


class TestQwenPromptPlaceholder:
    """The per-task knob is a placeholder inside QWEN_TOOL_PROMPT. Both
    variants must render through it without leaving the placeholder
    visible in the final prompt."""

    def test_placeholder_is_present_in_raw_prompt(self):
        assert "{think_budget_guidance}" in QWEN_TOOL_PROMPT

    def test_tight_substitution_produces_no_placeholder(self):
        rendered = QWEN_TOOL_PROMPT.replace(
            "{think_budget_guidance}",
            render_think_budget_guidance("tight"),
        )
        assert "{think_budget_guidance}" not in rendered
        assert "DO NOT enumerate future tool calls" in rendered      # §4RK: the cap wording is gone

    def test_extended_substitution_produces_no_placeholder(self):
        rendered = QWEN_TOOL_PROMPT.replace(
            "{think_budget_guidance}",
            render_think_budget_guidance("extended"),
        )
        assert "{think_budget_guidance}" not in rendered
        assert "DO NOT draft runnable code inside <think>" in rendered   # §4RK: the ~15-sentence cap is gone

    def test_anti_paralysis_rule_still_present(self):
        """Sanity: the downstream anti-paralysis rule must survive both
        budgets — it's a hard invariant (tests like test_agent_april_4
        assert its presence)."""
        assert "ANTI-PARALYSIS" in QWEN_TOOL_PROMPT
