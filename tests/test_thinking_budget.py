"""Per-task thinking-budget classifier.

The default (tight) is the historical 5-sentence anti-paralysis cap.
Extended fires when the query involves genuine multi-step reasoning —
debugging, algorithm design, proofs, SQL tuning — where the anti-
paralysis rule's "no alternatives" clause prevents legitimate derivation.
"""

from ghost_agent.core.agent import (
    classify_thinking_budget,
    render_think_budget_guidance,
)
from ghost_agent.core.prompts import (
    QWEN_TOOL_PROMPT,
    THINK_BUDGET_TIGHT,
    THINK_BUDGET_EXTENDED,
    THINK_BUDGET_SELFPLAY,
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

    def test_selfplay_maps_to_selfplay_block(self):
        # SELFPLAY is the tier dream.synthetic_self_play sets on the
        # temp_agent via `thinking_budget_override`. It must resolve to
        # the dedicated SELFPLAY guidance, not fall back to TIGHT or
        # EXTENDED — the wording is what forbids row-by-row
        # recompute in <think>.
        assert render_think_budget_guidance("selfplay") == THINK_BUDGET_SELFPLAY

    def test_unknown_falls_back_to_tight(self):
        assert render_think_budget_guidance("whatever") == THINK_BUDGET_TIGHT


class TestSelfPlayBudgetContent:
    """The SELFPLAY guidance exists specifically to stop the Qwen3.6
    behaviours observed in the 08:46 self-play log: drafting full
    Python inside <think>, and manually recomputing revenue for every
    CSV row after exit code 0. These assertions lock in the key
    phrases so a well-intentioned edit can't accidentally soften
    them back to the EXTENDED tier's permissive wording."""

    def test_forbids_drafting_code_in_think(self):
        assert "drafting runnable code" in THINK_BUDGET_SELFPLAY

    def test_forbids_row_by_row_compute(self):
        body = THINK_BUDGET_SELFPLAY.lower()
        assert "iterating over dataset rows" in body or "row" in body
        assert "by hand" in body

    def test_forbids_redrive_after_exit_zero(self):
        assert "exit code 0" in THINK_BUDGET_SELFPLAY
        assert "re-deriving" in THINK_BUDGET_SELFPLAY or "re-derive" in THINK_BUDGET_SELFPLAY.lower()

    def test_selfplay_is_strictly_shorter_than_extended_cap(self):
        # EXTENDED allows "up to ~15 sentences"; SELFPLAY caps at
        # 6 bullets. If someone relaxes SELFPLAY to 15+ the intent is
        # lost.
        assert "6" in THINK_BUDGET_SELFPLAY and "bullet" in THINK_BUDGET_SELFPLAY.lower()


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
        assert "EXTREMELY CONCISE" in rendered

    def test_extended_substitution_produces_no_placeholder(self):
        rendered = QWEN_TOOL_PROMPT.replace(
            "{think_budget_guidance}",
            render_think_budget_guidance("extended"),
        )
        assert "{think_budget_guidance}" not in rendered
        assert "up to ~15 sentences" in rendered

    def test_anti_paralysis_rule_still_present(self):
        """Sanity: the downstream anti-paralysis rule must survive both
        budgets — it's a hard invariant (tests like test_agent_april_4
        assert its presence)."""
        assert "ANTI-PARALYSIS" in QWEN_TOOL_PROMPT
