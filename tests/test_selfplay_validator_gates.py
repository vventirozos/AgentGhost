"""Self-play validator gate hardening (2026-07-29).

Overnight run: an LLM-generated validator split captured stdout with a
LITERAL backslash-n — ``result.stdout.split('\\\\n')`` in source — so every
solution collapsed to "1 line" and a provably correct solver failed 3/3
attempts ("Expected 10 lines, got 1 lines"), recording a false failure and
a -1.0 frontier delta. Four gate layers missed it: the static gate had no
split lint; the echo self-test probe skipped silently (expected vars built
inside ``def validate()``, out of scope at the module-level insertion
point); the echo verdict only rejected CRASHES, not clean non-zero exits;
and the reference-consistency gate was skipped because no reference
solution shipped.

Covered here:
  * ``_has_literal_backslash_split`` — the static lint, including the
    exact validator recovered from the counterfactual stash.
  * ``validate_challenge_quality`` — rejects a literal-split validator
    with an actionable reason.
  * ``_feedback_shows_joined_actual`` — the solver-side backstop that
    reroutes this failure shape to validator-infra instead of charging
    the agent.
"""

import pytest

from ghost_agent.core.dream import (
    _feedback_shows_joined_actual,
    _has_literal_backslash_split,
    validate_challenge_quality,
)


# Abridged from the actual 2026-07-29 validator (counterfactual stash id
# 8d3567930273): expected computed inside a function, subprocess re-run of
# solution.py, the broken literal-\n split, and — load-bearing for the
# backstop — the two list prints that make the failure self-diagnosing.
REAL_VALIDATOR_ABRIDGED = '''\
import subprocess

def validate():
    expected_output_lines = ["PROD_001:41", "PROD_002:-29"]
    result = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True, timeout=15)
    actual_output_lines = [line.strip() for line in result.stdout.split('\\\\n') if line.strip()]
    if len(actual_output_lines) != len(expected_output_lines):
        print(f"FAIL: Line count mismatch. Expected {len(expected_output_lines)} lines, got {len(actual_output_lines)} lines.")
        print(f"Expected lines: {expected_output_lines}")
        print(f"Actual lines: {actual_output_lines}")
        exit(1)
    print("PASS"); exit(0)

if __name__ == "__main__":
    validate()
'''


# ---------------------------------------------------------------------------
# static lint
# ---------------------------------------------------------------------------

def test_lint_catches_real_overnight_validator():
    assert _has_literal_backslash_split(REAL_VALIDATOR_ABRIDGED) is True


def test_lint_catches_literal_backslash_variants():
    assert _has_literal_backslash_split(r'out.split("\\n")') is True
    assert _has_literal_backslash_split(r"s.split('\\r\\n')") is True
    assert _has_literal_backslash_split(r'x.split(  "\\t"  )') is True


def test_lint_passes_correct_splits():
    assert _has_literal_backslash_split(r'out.split("\n")') is False
    assert _has_literal_backslash_split("out.splitlines()") is False
    assert _has_literal_backslash_split('parts = line.split(":")') is False
    assert _has_literal_backslash_split("") is False
    assert _has_literal_backslash_split(None) is False


def test_quality_gate_rejects_literal_split_validator():
    ok, reason = validate_challenge_quality("", REAL_VALIDATOR_ABRIDGED)
    assert ok is False
    assert "backslash" in reason
    # The reason must carry the concrete fix for the regen feedback loop.
    assert "splitlines" in reason


def test_quality_gate_accepts_correct_split_validator():
    good = REAL_VALIDATOR_ABRIDGED.replace("split('\\\\n')", "split('\\n')")
    ok, reason = validate_challenge_quality("", good)
    assert ok is True, reason


# ---------------------------------------------------------------------------
# solver-side backstop
# ---------------------------------------------------------------------------

# Feedback exactly as the broken validator printed it: the "actual" list has
# ONE element whose repr shows the full correct output, newline-escaped.
_FEEDBACK_BROKEN_SPLIT = (
    "FAIL: Line count mismatch. Expected 3 lines, got 1 lines. "
    r"Expected lines: ['A:1', 'B:2', 'C:3'] Actual lines: ['A:1\nB:2\nC:3']"
)


def test_backstop_detects_joined_actual():
    assert _feedback_shows_joined_actual(_FEEDBACK_BROKEN_SPLIT) is True


def test_backstop_ignores_genuine_mismatch():
    fb = "Expected lines: ['A:1', 'B:2'] Actual lines: ['A:1', 'B:9']"
    assert _feedback_shows_joined_actual(fb) is False


def test_backstop_ignores_wrong_values_even_when_joined():
    fb = r"Expected lines: ['A:1', 'B:2'] Actual lines: ['A:1\nB:9']"
    assert _feedback_shows_joined_actual(fb) is False


def test_backstop_ignores_single_line_challenges():
    # A one-line expected output joined into one line is indistinguishable
    # from a normal pass/fail — must not trigger.
    fb = "Expected lines: ['A:1'] Actual lines: ['A:9']"
    assert _feedback_shows_joined_actual(fb) is False


def test_backstop_safe_on_garbage():
    assert _feedback_shows_joined_actual("") is False
    assert _feedback_shows_joined_actual(None) is False
    assert _feedback_shows_joined_actual("Expected [ Actual [ nonsense") is False


def test_backstop_fires_on_real_validator_output(tmp_path):
    """End-to-end pin: run the recovered validator shape against a CORRECT
    solution and feed its real stdout to the backstop.

    The unit tests above use hand-written feedback strings; this one proves
    the detector matches what the process actually prints — the gap that
    made the 2026-07-29 false failure survive review.
    """
    import subprocess
    import sys

    (tmp_path / ".validator.py").write_text(REAL_VALIDATOR_ABRIDGED)
    # A CORRECT solution: exactly the expected lines, real newlines.
    (tmp_path / "solution.py").write_text(
        'print("PROD_001:41")\nprint("PROD_002:-29")\n'
    )

    proc = subprocess.run(
        [sys.executable, ".validator.py"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )

    # The broken validator rejects a correct solution...
    assert proc.returncode == 1
    # ...and the backstop recognises its output as the joined-actual shape.
    assert _feedback_shows_joined_actual(proc.stdout) is True


# ---------------------------------------------------------------------------
# lint precision: no false rejects that would burn regeneration attempts
# ---------------------------------------------------------------------------

def test_lint_ignores_the_pattern_quoted_in_a_comment():
    # The reject reason is fed back into the regeneration prompt; a model
    # that echoes the constraint as a defensive comment must not be
    # rejected again for it (that burns every attempt → template fallback).
    src = 'out = result.stdout.splitlines()  # NOT .split("\\\\n") — never splits\n'
    assert _has_literal_backslash_split(src) is False


def test_lint_catches_raw_string_form():
    # r"\n" is the SAME literal backslash-n at runtime.
    assert _has_literal_backslash_split(r'out.split(r"\n")') is True


def test_lint_allows_re_split_escape():
    # The `re` engine interprets "\\n" as a real newline, so this is correct.
    assert _has_literal_backslash_split(r'parts = re.split("\\n", text)') is False


def test_quality_gate_does_not_reject_reason_echo():
    ok, reason = validate_challenge_quality("", REAL_VALIDATOR_ABRIDGED)
    assert ok is False
    # The reason must not embed the anti-pattern verbatim, or feeding it to
    # the regen prompt re-teaches the bug.
    assert '.split("\\\\n")' not in reason


# ---------------------------------------------------------------------------
# echo-gate decision matrix
# ---------------------------------------------------------------------------
# The gate rejects a non-zero echo exit ONLY on evidence of an internal
# contradiction — a crash in the validator's own frame, or the joined-actual
# signature. A bare clean failure is INCONCLUSIVE and must pass: the echo
# probe writes the expected variable verbatim, so a validator that requires
# SHAPED stdout (challenge_templates' `FOUND=blob3.txt` shape) fails the echo
# while being perfectly winnable. Rejecting it would forfeit an idle slot on
# a hand-written, provably-correct challenge.

from ghost_agent.core.dream import _looks_like_validator_crash


def _echo_verdict(validator_stdout: str) -> str:
    """Mirror of the gate's decision on a non-zero echo exit."""
    if _looks_like_validator_crash(validator_stdout):
        return "reject"
    if _feedback_shows_joined_actual(validator_stdout):
        return "reject"
    return "allow"


def test_echo_allows_shaped_output_validator():
    # challenge_templates.py "find the blob" shape: expected holds the raw
    # answer, stdout must read `FOUND=blob3.txt`.
    assert _echo_verdict("BAD OUTPUT SHAPE") == "allow"


def test_echo_allows_trailing_newline_mismatch():
    # The probe's dump-extraction strips one trailing newline, so an
    # exact-equality validator can fail its own expected output.
    assert _echo_verdict("FAIL: output did not match expected exactly") == "allow"


def test_echo_rejects_validator_crash():
    tb = (
        'Traceback (most recent call last):\n'
        '  File ".validator.py", line 12, in <module>\n'
        "    v = float('60.00%')\n"
        "ValueError: could not convert string to float: '60.00%'"
    )
    assert _echo_verdict(tb) == "reject"


def test_echo_rejects_joined_actual_signature():
    assert _echo_verdict(_FEEDBACK_BROKEN_SPLIT) == "reject"
