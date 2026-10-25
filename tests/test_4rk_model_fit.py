"""§4RK (2026-10-25, operator: "given how this new model thinks are there any
other optimizations, corrections or limits to adjust?" → "fix 2 and 3,
increase the timeouts for 4"). Three settings tuned for Qwen3.5/3.6, re-fit
to Qwen3.8-27B from today's log:

  * the n-gram thinking-loop probe fired 7 times — on zero loops: long
    deliberations restating a quoted item, and turns quoting the same
    transcript passage three times to compare it;
  * the think-budget directives capped thinking at 5 / ~15 sentences while
    the server's reasoning_effort sets the depth;
  * the continuation timeouts (300 s / 180 s) could not fit one full round
    at the measured ~23 tok/s.
"""
import ghost_agent.core.agent as A
import ghost_agent.core.stream_guards as SG
import ghost_agent.core.write_continuation as W
from ghost_agent.core.prompts import THINK_BUDGET_EXTENDED, THINK_BUDGET_TIGHT, QWEN_TOOL_PROMPT

MEASURED_TOK_PER_S = 23.0          # llama-server.log, generation median on the 27B


# ── the n-gram probe: recent repetition, not a passage quoted three times ──

_PASSAGE = ("Therefore, the 'stable understanding' is a functional utility for communication, not an "
            "ontological fact. I am stable enough to be useful, but fluid enough to be infinitely adaptable. "
            "The recursion reveals that I am a mirror that reflects the structure of the question back at itself.")
_COMMENTARY = ("Now compare that with what the second transcript says about the same prompt; the wording "
               "differs in the second clause, and the third transcript drops the mirror image entirely, so the "
               "table should mark this row as a partial match rather than identical. ") * 4   # ~1,000 chars


def test_quoting_the_same_passage_three_times_is_not_a_loop():
    """Fails in the measured world (4D at 5,037 chars, 95 at 7,541 chars)."""
    buf = ""
    for _ in range(3):
        buf += f'The transcript says: "{_PASSAGE}"\n{_COMMENTARY}\n'
    buf += f'And again the report quotes: "{_PASSAGE}'          # the tail IS the passage, seen 3×
    assert buf.count(buf[-SG.THINKING_LOOP_WINDOW:]) >= SG.THINKING_LOOP_THRESHOLD   # the OLD rule fires here
    assert SG._detect_thinking_loop(buf) is False


def test_a_runaway_that_repeats_its_block_still_dies():
    assert SG._detect_thinking_loop("The answer is 42. " * 40) is True                 # back to back
    assert SG._detect_thinking_loop(("A" * 300 + "\n") * 4) is True                     # a 301-char block ×4
    filler = "and once more, because the check did not land: "                          # 48 chars between repeats
    assert SG._detect_thinking_loop((_PASSAGE + filler) * 5) is True                    # short filler
    assert SG._detect_thinking_loop("") is False


def test_the_probe_reads_a_bounded_recent_region():
    """The count runs on the recent region (threshold × max gap + window), so a
    200k-char buffer costs the same per probe as a 2k one."""
    region = SG.THINKING_LOOP_THRESHOLD * SG.THINKING_LOOP_MAX_GAP + SG.THINKING_LOOP_WINDOW
    assert region <= 2500
    long = "x" * 200_000 + _PASSAGE + " y " * 10 + _PASSAGE
    assert SG._detect_thinking_loop(long) is False


# ── the think budget: what goes in the think, not how long it is ────────────

def test_the_think_directives_cap_nothing_and_keep_the_content_rules():
    for text in (THINK_BUDGET_TIGHT, THINK_BUDGET_EXTENDED):
        assert "sentences" not in text and "CONCISE" not in text
    assert "DO NOT enumerate future tool calls" in THINK_BUDGET_TIGHT
    assert "DO NOT draft runnable code inside <think>" in THINK_BUDGET_EXTENDED
    assert "{think_budget_guidance}" in QWEN_TOOL_PROMPT
    assert A.render_think_budget_guidance("tight") == THINK_BUDGET_TIGHT


# ── the continuation timeouts fit one full round at the measured rate ───────

def test_a_full_continuation_round_fits_its_timeout():
    """Fails in the measured world: 8,192 tokens at 23 tok/s is 356 s of
    generation before the prefill; the round timed out at 300 s."""
    one_round_s = W.CONTINUATION_MAX_TOKENS / MEASURED_TOK_PER_S
    assert W.CONTINUATION_TIMEOUT_S >= one_round_s * 1.3           # generation + prefill headroom
    assert W.CONTINUATION_TOTAL_S >= 3 * W.CONTINUATION_TIMEOUT_S  # three full rounds
    assert A.TRUNCATION_CONTINUATION_TIMEOUT_S >= 480.0
