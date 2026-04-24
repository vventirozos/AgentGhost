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


# --------------------------------------------------------- tightened thresholds

def test_window_tightened_post_0b_trace():
    """Regression: the 0B trace ran for ~600 chars of visible repetition.
    Post-fix must fire earlier than that."""
    assert THINKING_LOOP_WINDOW <= 150


def test_threshold_tightened_post_0b_trace():
    assert THINKING_LOOP_THRESHOLD <= 2


def test_probe_frequency_tightened():
    assert THINKING_LOOP_PROBE_EVERY <= 300


# --------------------------------------------------------- enumeration pattern

def test_enumeration_pattern_trips_detector():
    """The exact shape from the trace: 'I'll write X. Then Y. Then Z.'
    repeated. Must be flagged by the detector. Three repeats × 150 chars
    each is more than enough to trip the new threshold."""
    enumeration = (
        "I'll write streaming_reader.py. Then aggregator.py. "
        "Then output_report.py. Then cli.py. Then test_parser.py. "
        "Then I'll mark the tasks as DONE and move on. "
    )
    # 150 chars × 4 repeats = 600 chars; detector (150/2) should fire
    buf = enumeration * 4
    assert _detect_thinking_loop(buf) is True


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
