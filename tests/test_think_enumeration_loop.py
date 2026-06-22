"""Tests for the anti-enumeration guard + tighter thinking-loop probe.

Regression target (2026-04-19 trace 0B): model spent 64.7s and 3043
tokens in <think> repeating 'I'll write X. Then Y. Then Z.' for 5
planned files. The loop detector fired but only after ~600 chars of
repetition. Two complementary fixes pinned here:

  1. THINK_BUDGET_TIGHT prompt includes an explicit anti-enumeration
     rule so the model is instructed not to enter the pattern.
  2. THINKING_LOOP_WINDOW/THRESHOLD tightened to 150/2 so the
     detector fires at ~300 chars instead of ~600.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.agent import (
    _detect_thinking_loop,
    THINKING_LOOP_WINDOW,
    THINKING_LOOP_THRESHOLD,
    THINKING_LOOP_PROBE_EVERY,
)
from ghost_agent.core.prompts import THINK_BUDGET_TIGHT


# --------------------------------------------------------- conservative thresholds
#
# 2026-06-22: the aggressive (300/150/2) tightening was REVERTED to the
# conservative (500/200/3). It was tuned to kill a weak model's enumeration
# loops fast at an "acceptable false-positive risk"; a strong model rarely
# produces those loops, so that false-positive cost (aborting legitimate
# reasoning that restates a constraint / invariant) now outweighs the benefit.
# The detector still catches a GENUINE loop (≥3 repeats of a 200-char window),
# and the tool_call-collapse probe + the 200K-char ceiling remain fast backstops.

def test_thresholds_are_conservative_for_strong_model():
    assert THINKING_LOOP_WINDOW == 200
    assert THINKING_LOOP_THRESHOLD == 3
    assert THINKING_LOOP_PROBE_EVERY == 500


# --------------------------------------------------------- enumeration pattern

def test_enumeration_pattern_trips_detector():
    """A genuine loop still trips: a >200-char unit repeated 4× (clearly
    repetitive) must be flagged even under the conservative (200/3) detector."""
    enumeration = (
        "I'll write streaming_reader.py. Then aggregator.py. "
        "Then output_report.py. Then cli.py. Then test_parser.py. "
        "Then config_loader.py. Then schema_validator.py. "
        "Then I'll mark the tasks as DONE and move on next. "  # >200 chars/unit
    )
    assert len(enumeration) > 200
    buf = enumeration * 4
    assert _detect_thinking_loop(buf) is True


def test_modestly_repeating_reasoning_is_not_a_loop():
    """Two passes of similar-but-not-identical reasoning (the false-positive
    the aggressive 2× threshold caught) is NOT flagged under (200/3)."""
    a = ("Let me verify the parser handles empty lines, then check the "
         "aggregator sums per-key counts correctly before writing output. ")
    b = ("Now let me verify the parser also handles comment lines, then check "
         "the aggregator orders keys deterministically before writing output. ")
    assert _detect_thinking_loop(a + b) is False


def test_short_nonrepeating_thinking_passes():
    buf = (
        "Let me consider the next step. The parser file exists and the "
        "stream reader task is next. I will write a generator that "
        "reads line by line and yields parsed entries."
    )
    assert _detect_thinking_loop(buf) is False


# --------------------------------------------------------- prompt directive

def test_tight_budget_carries_anti_enumeration_rule():
    """If the rule disappears in a future refactor, the enumeration
    loop returns. Pin the wording."""
    assert "DO NOT enumerate future tool calls" in THINK_BUDGET_TIGHT
    assert "SINGLE next one" in THINK_BUDGET_TIGHT


def test_tight_budget_still_caps_sentences():
    """The original 5-sentence cap must survive the anti-enumeration
    additions — it's the baseline defense."""
    assert "Maximum 5 sentences" in THINK_BUDGET_TIGHT
