"""GAIA scorer + answer extraction (the exactness-critical benchmark piece).

A drifted normalizer silently inflates or deflates the reported score, so the
grading must match the official leaderboard `question_scorer` rules exactly.
These pin: number normalization (units/commas/percent), comma/semicolon list
element-wise compare, string whitespace/punct/case normalization, and the
FINAL-ANSWER extraction contract (last marker wins; absent marker → None, which
the runner scores as a no-answer rather than an empty match).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gaia_scorer import (  # noqa: E402
    GAIA_SYSTEM_PROMPT, extract_final_answer, question_scorer,
)


# ── number ground truth ──────────────────────────────────────────────────────

@pytest.mark.parametrize("answer,gt,ok", [
    ("42", "42", True),
    ("42.0", "42", True),
    ("42", "43", False),
    ("$1,234", "1234", True),        # units + thousands comma stripped
    ("1,234", "1234", True),
    ("17%", "17", True),
    ("  89 ", "89", True),
    ("ninety", "90", False),         # words don't normalize to a number
])
def test_number_scoring(answer, gt, ok):
    assert question_scorer(answer, gt) is ok


# ── string ground truth ──────────────────────────────────────────────────────

@pytest.mark.parametrize("answer,gt,ok", [
    ("Jane Austen", "jane austen", True),      # case
    ("Paris.", "Paris", True),                 # trailing punct stripped
    ("New York", "new york", True),            # internal space collapsed
    ("Einstein", "Newton", False),
    ("the Louvre", "Louvre", False),           # NOTE: article NOT auto-stripped
                                               # by the scorer — the prompt tells
                                               # the model to omit it; we grade
                                               # what the rules actually do.
])
def test_string_scoring(answer, gt, ok):
    assert question_scorer(answer, gt) is ok


# ── list ground truth ────────────────────────────────────────────────────────

def test_list_exact_order_and_membership():
    assert question_scorer("Mercury, Venus, Earth", "Mercury, Venus, Earth") is True
    assert question_scorer("mercury,venus,earth", "Mercury, Venus, Earth") is True
    # wrong order fails (element-wise, positional)
    assert question_scorer("Venus, Mercury, Earth", "Mercury, Venus, Earth") is False
    # length mismatch fails
    assert question_scorer("Mercury, Venus", "Mercury, Venus, Earth") is False


def test_list_mixed_number_and_string():
    assert question_scorer("apple, 3, pear", "apple, 3, pear") is True
    assert question_scorer("apple, 3.0, pear", "apple, 3, pear") is True   # 3.0==3
    assert question_scorer("apple, 4, pear", "apple, 3, pear") is False


def test_semicolon_list():
    assert question_scorer("a; b; c", "a;b;c") is True


# ── extraction ───────────────────────────────────────────────────────────────

def test_extract_basic():
    assert extract_final_answer("reasoning...\nFINAL ANSWER: 74") == "74"


def test_extract_last_marker_wins():
    txt = "FINAL ANSWER: 10\nwait, reconsidering\nFINAL ANSWER: 12"
    assert extract_final_answer(txt) == "12"


def test_extract_case_insensitive_and_strips_wrappers():
    assert extract_final_answer("final answer: [Paris]") == "Paris"
    assert extract_final_answer('FINAL ANSWER: "Jane Austen"') == "Jane Austen"


def test_extract_absent_marker_is_none():
    # absent marker → None; the runner turns None into an incorrect no-answer,
    # never an empty-vs-empty false positive against a "?" test-split GT
    assert extract_final_answer("I think it is 42 but I am not sure.") is None
    assert extract_final_answer("") is None


def test_extract_preserves_list_commas_but_drops_trailing_prose():
    txt = "FINAL ANSWER: Mercury, Venus, Earth\n(those are the inner three)"
    assert extract_final_answer(txt) == "Mercury, Venus, Earth"


# ── end-to-end: extraction → scoring ─────────────────────────────────────────

def test_end_to_end_correct():
    reply = "Let me work through it.\nFINAL ANSWER: 1889"
    assert question_scorer(extract_final_answer(reply), "1889") is True


def test_prompt_is_verbatim_canonical():
    # guardrail: the format rules the scorer assumes must stay in the prompt
    assert "FINAL ANSWER:" in GAIA_SYSTEM_PROMPT
    assert "comma separated list" in GAIA_SYSTEM_PROMPT
    assert "don't use articles" in GAIA_SYSTEM_PROMPT
