"""Per-step result cap for composed-skill execution.

Regression cover for the "morning briefing only shows 2 (then 8) of 10
headlines" bug. The macro fetched all 10 headlines correctly, but the
composed-skill runner truncated each step's body to a fixed 1000 chars
*silently* — so a list-bearing step (10 headlines, well over 1000 chars)
was cut down to ~2 items, and the model, believing it had received the
whole list, delivered a short briefing.

Two properties are pinned here:
  * the cap is large enough to carry a realistic 10-headline step intact,
  * when a step DOES exceed the cap, the truncation is marked EXPLICITLY
    (never silent) so the model/verifier can see content was dropped and
    re-fetch the step standalone.
"""

from __future__ import annotations

import pytest

from ghost_agent.tools.composed_skills import (
    ComposedSkill, ComposedSkillRegistry, SkillStep,
    MAX_STEP_RESULT_CHARS, _cap_step_result,
)


@pytest.fixture
def registry():
    # In-memory registry (no storage_dir) — execution doesn't need disk.
    return ComposedSkillRegistry()


# A realistic naftemporiki-style 10-headline payload: each item is a title
# plus a short summary plus a URL, which is what blew past the old 1000-char
# cap and collapsed the briefing to ~2 visible headlines.
def _ten_headlines() -> str:
    lines = ["Latest 10 headlines from naftemporiki.gr:"]
    for i in range(1, 11):
        lines.append(
            f"{i}. Headline number {i} about markets, economy and shipping "
            f"developments in Greece and abroad — summary sentence {i} giving "
            f"the gist of the story in roughly forty words so the briefing "
            f"reader gets real context. https://www.naftemporiki.gr/story/{i:06d}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# _cap_step_result — the pure helper
# --------------------------------------------------------------------------

class TestCapStepResult:
    def test_short_result_passes_through_unchanged(self):
        s = "weather: Athens, 28C, clear"
        assert _cap_step_result(s) == s

    def test_none_becomes_empty_string(self):
        assert _cap_step_result(None) == ""

    def test_exactly_at_limit_is_untouched(self):
        s = "x" * MAX_STEP_RESULT_CHARS
        assert _cap_step_result(s) == s

    def test_over_limit_is_truncated_with_explicit_marker(self):
        s = "x" * (MAX_STEP_RESULT_CHARS + 500)
        out = _cap_step_result(s)
        # Body is capped...
        assert out.startswith("x" * MAX_STEP_RESULT_CHARS)
        # ...but the truncation is NEVER silent.
        assert "truncated" in out.lower()
        assert "500" in out  # number of dropped chars surfaced
        assert "standalone" in out.lower()  # tells the model how to recover

    def test_custom_limit(self):
        assert _cap_step_result("abcdef", limit=3).startswith("abc")
        assert "truncated" in _cap_step_result("abcdef", limit=3).lower()

    def test_ten_headlines_survive_the_default_cap(self):
        # The core regression: a full 10-headline payload must NOT be
        # truncated by the default per-step cap.
        payload = _ten_headlines()
        assert len(payload) > 1000, "fixture must exceed the OLD 1000-char cap"
        assert len(payload) <= MAX_STEP_RESULT_CHARS
        assert _cap_step_result(payload) == payload
        assert "truncated" not in _cap_step_result(payload).lower()
        # All ten items still present.
        for i in range(1, 11):
            assert f"{i}. Headline number {i}" in _cap_step_result(payload)


# --------------------------------------------------------------------------
# End-to-end through the macro runner — both execution modes
# --------------------------------------------------------------------------

class TestBriefingStepNotTruncated:
    async def test_parallel_macro_keeps_all_ten_headlines(self, registry):
        registry.register(ComposedSkill(
            name="briefing", trigger_description="morning briefing",
            execution_mode="parallel",
            steps=[SkillStep(tool_name="fetch_news", description="latest 10 headlines")],
        ))

        async def ex(tool, args):
            return _ten_headlines()

        res = await registry.execute("briefing", ex)
        body = res["results"][0]["result"]
        assert "truncated" not in body.lower()
        for i in range(1, 11):
            assert f"{i}. Headline number {i}" in body

    async def test_sequential_macro_keeps_all_ten_headlines(self, registry):
        registry.register(ComposedSkill(
            name="briefing_seq", trigger_description="morning briefing",
            execution_mode="sequential",
            steps=[SkillStep(tool_name="fetch_news", description="latest 10 headlines")],
        ))

        async def ex(tool, args):
            return _ten_headlines()

        res = await registry.execute("briefing_seq", ex)
        body = res["results"][0]["result"]
        for i in range(1, 11):
            assert f"{i}. Headline number {i}" in body

    async def test_runaway_step_still_bounded_and_marked(self, registry):
        # A genuinely chatty step (well over the cap) is still bounded — the
        # context-budget guard the cap exists for is preserved — but now it
        # announces the truncation instead of hiding it.
        registry.register(ComposedSkill(
            name="chatty", trigger_description="chatty",
            execution_mode="parallel",
            steps=[SkillStep(tool_name="dump", description="huge dump")],
        ))

        async def ex(tool, args):
            return "y" * (MAX_STEP_RESULT_CHARS * 3)

        res = await registry.execute("chatty", ex)
        body = res["results"][0]["result"]
        assert len(body) < MAX_STEP_RESULT_CHARS * 3
        assert "truncated" in body.lower()
