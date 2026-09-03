"""§4EC — `core/stream_guards.py` and `utils/constraints.py` survivors of the §R
re-verification of §4BZ (2026-09-02): the guards' BOUNDS and the hoist parser's
edge arms, each with a world in which it decides."""
import pytest

from ghost_agent.core import stream_guards as sg
from ghost_agent.utils.constraints import parse_start_with_phrase, enforce_start_with


LONG = "this is a long repeated reasoning line of well over forty-eight characters"
FILL = "\n".join(f"unique filler line number {i:04d} padding padding padding padding padding" for i in range(90))


class TestParagraphLoopBounds:
    def test_short_lines_at_the_tail_do_not_consume_the_scan_window(self):
        # L90: short lines are SKIPPED, not counted against PARAGRAPH_LOOP_SCAN_LINES
        buf = FILL + "\n" + "\n".join([LONG] * sg.PARAGRAPH_LOOP_THRESHOLD) + "\n" + "\n".join(["ok"] * 4) + "\nfrag"
        assert len(buf) >= sg.PARAGRAPH_LOOP_MIN_BUF
        assert sg._detect_paragraph_loop(buf) is True

    def test_a_repeat_older_than_the_scan_window_is_not_a_loop(self):
        # L94-96: only the SCAN_LINES newest long lines are checked
        newest = [f"a different long closing thought number {i} that keeps going for a while" for i in range(sg.PARAGRAPH_LOOP_SCAN_LINES + 1)]
        buf = FILL + "\n" + "\n".join([LONG] * sg.PARAGRAPH_LOOP_THRESHOLD) + "\n" + "\n".join(newest) + "\nfrag"
        assert sg._detect_paragraph_loop(buf) is False


def test_stop_marker_at_the_head_of_a_whole_short_buffer_is_real():
    # L156: with idx == 0 there is NO previous char — the mutant reads tail[-1]
    # (a trailing backtick) and dismisses a real transition as a mention
    buf = "<tool_call>abcd`"
    assert sg._tail_has_stop_marker(buf, "") is True


class TestStartWithPhraseParsing:
    def test_a_later_constraint_is_still_scanned_after_a_miss(self):
        assert parse_start_with_phrase(["keep it short", 'Start with "BLUF:"']) == "BLUF:"

    @pytest.mark.parametrize("quoted,expected", [
        ('"Hello!"', "Hello"),   # trailing punctuation dropped on a phrase longer than 4
        ('"Hi!"', "Hi!"),        # kept on a short phrase (the punctuation IS the phrase)
        ('"Hi"', None),          # under 3 chars is no mandate
    ])
    def test_punctuation_and_length_rules(self, quoted, expected):
        assert parse_start_with_phrase([f"Start with {quoted}"]) == expected


class TestEnforceStartWith:
    C = ['Start with "BLUF:"']

    def test_an_already_compliant_reply_is_untouched_even_with_a_second_phrase_led_segment(self):
        reply = "BLUF: first.\n\nBody.\n\nBLUF: again later."
        assert enforce_start_with(reply, self.C) == (reply, 0)

    def test_the_first_matching_segment_after_non_matching_ones_is_hoisted(self):
        reply = "Preamble one.\n\nPreamble two.\n\nBLUF: the point.\n\nDetails."
        tail, dropped = enforce_start_with(reply, self.C)
        assert tail.startswith("BLUF: the point.") and dropped == len("Preamble one.\n\nPreamble two.\n\n")

    def test_a_cut_inside_an_open_fence_is_refused(self):
        reply = "Intro\n\n```\ncode\n\nBLUF: inside the fence\n```\n\nmore"
        assert enforce_start_with(reply, self.C) == (reply, 0)

    def test_a_cut_that_keeps_too_little_is_refused(self):
        big = "x" * 400
        reply = f"{big}\n\n{big}\n\nBLUF: tiny."
        assert enforce_start_with(reply, self.C) == (reply, 0)   # tail < 40% of the reply

    def test_no_phrase_led_segment_means_no_change(self):
        reply = "One.\n\nTwo.\n\nThree."
        assert enforce_start_with(reply, self.C) == (reply, 0)
