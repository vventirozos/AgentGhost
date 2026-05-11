"""Regression: the verifier gate must use the STRICT trivial-chat
check, not the loose `is_trivial_greeting` flag.

Pre-fix the gate condition was:

    if verifier is not None
       and last_tool is not None
       and final_ai_content
       and not is_trivial_greeting:

`is_trivial_greeting` is the loose 5-word-conversational check used
to identify "this might be a greeting". The fast-path at the top of
`handle_chat` (around line 2164) AND-combines it with
`_is_strict_trivial_chat(lc)` precisely because the loose flag is
too broad: it fires on correction-shaped prompts like:

    "thanks but wrong answer"     # 4 words, conversational
    "no try harder"               # 3 words, conversational
    "ok do it again"              # 4 words, conversational
    "yeah keep going"             # 3 words, conversational

Those are exactly the highest-leverage cases to verify — the user
just told us we got it wrong. With the loose flag, the verifier
silently skipped them all once the strict-fast-path rejected them
and the full loop ran with tools.

Fix: at the verifier gate, use ``not self._is_strict_trivial_chat(lc)``
so only ACTUAL greetings ("hi", "thanks") skip verification.
"""
from pathlib import Path
import pytest

from ghost_agent.core.agent import GhostAgent


# Source-pin guard. The actual end-to-end test would require booting
# the verifier with a real LLM, which is out of scope. Pin the gate
# expression so a future refactor can't reintroduce the loose flag.
def test_verifier_gate_uses_strict_check_not_loose_flag():
    src = Path("src/ghost_agent/core/agent.py").read_text()
    # The verifier gate is uniquely identifiable by the
    # `verifier is not None` predicate inside an `if (...)` block.
    # That predicate doesn't appear elsewhere.
    marker = "verifier is not None\n"
    idx = src.find(marker)
    assert idx != -1, "Verifier gate predicate not found"
    # The gate condition spans from `if (` to `):`. Look back for the
    # opening `if (` and forward for the closing `):`.
    open_idx = src.rfind("if (", max(0, idx - 200), idx)
    close_idx = src.find("):", idx)
    assert open_idx != -1 and close_idx != -1, "Could not isolate gate block"
    gate_block = src[open_idx:close_idx + 2]

    assert "_is_strict_trivial_chat(lc)" in gate_block, (
        "Verifier gate must use strict trivial-chat check (not loose "
        f"is_trivial_greeting). Gate block:\n{gate_block}"
    )
    # The buggy form (loose flag without strict ALSO) must not appear.
    if "is_trivial_greeting" in gate_block:
        # If the loose flag is referenced at all in the gate, the
        # strict check must be the actual decider.
        assert "_is_strict_trivial_chat" in gate_block, (
            "Verifier gate uses `is_trivial_greeting` without "
            "`_is_strict_trivial_chat` — that's the bug we're fixing."
        )


# Behavioural test of the strict check itself: the prompts that
# previously slipped past must now be classified as NON-trivial.
@pytest.mark.parametrize("correction_prompt", [
    "thanks but wrong",
    "no try again",
    "ok do it again",
    "yeah keep going",
    "no try harder",
    "thanks but that's wrong",
    "ok but the answer is wrong",
])
def test_correction_prompts_are_not_strictly_trivial(correction_prompt):
    """Correction-shaped prompts must NOT be classified as trivial
    by the strict check, so the verifier gate doesn't skip them."""
    assert not GhostAgent._is_strict_trivial_chat(correction_prompt.lower()), (
        f"{correction_prompt!r} was classified as trivial — verifier "
        "would skip and miss a high-leverage correction."
    )


@pytest.mark.parametrize("real_greeting", [
    "hi",
    "hello",
    "thanks",
    "ok",
    "thank you",
    "good morning",
    "got it",
    "no worries",
    "sounds good",
])
def test_actual_greetings_are_strictly_trivial(real_greeting):
    """The strict check must still catch actual greetings — those
    SHOULD bypass verification (no tool output, no claim to verify)."""
    assert GhostAgent._is_strict_trivial_chat(real_greeting.lower()), (
        f"{real_greeting!r} should be classified as trivial, "
        "but the strict check rejected it."
    )
