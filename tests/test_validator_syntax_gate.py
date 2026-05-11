"""Coverage for the in-loop syntax gate in `validate_challenge_quality`.

Production trace over a single session showed 37 self-play cycles
terminating after the downstream sandbox pre-flight detected a
SyntaxError in the LLM-generated `<validation_script>` (35 of them
f-string mistakes — embedded `[`, mismatched quotes inside `{...}`,
unescaped `}`). The pre-flight runs OUTSIDE the regen loop, so each
of those failures wasted the entire 3-attempt budget.

The fix calls `ast.parse` on both scripts at the top of
`validate_challenge_quality` so the regen loop receives the line number
and message as `rejection_feedback` and can spend its 3 attempts on
recovery instead of failing terminally on attempt 1.
"""
from __future__ import annotations

import pytest

from ghost_agent.core.dream import validate_challenge_quality


GOOD_SETUP = "open('x.csv','w').write('a\\n1\\n')\n"
GOOD_VALIDATOR = (
    "data = open('x.csv').read()\n"
    "assert '1' in data\n"
)


class TestSyntaxGateAcceptsValid:
    def test_well_formed_pair_passes(self):
        ok, reason = validate_challenge_quality(GOOD_SETUP, GOOD_VALIDATOR)
        assert ok is True, reason

    def test_well_formed_validator_with_complex_fstring_passes(self):
        # Realistic, syntactically valid f-strings should pass — the
        # gate must not over-reject.
        validator = (
            "data = open('x.csv').read()\n"
            "name = 'alice'\n"
            "age = 30\n"
            "print(f'hello {name}, age {age} (len={len(data)})')\n"
        )
        ok, reason = validate_challenge_quality(GOOD_SETUP, validator)
        assert ok is True, reason

    def test_empty_setup_does_not_crash(self):
        # Setup is optional in some challenge shapes (validator-only).
        ok, reason = validate_challenge_quality("", GOOD_VALIDATOR)
        assert ok is True, reason

    def test_none_setup_does_not_crash(self):
        # `_extract_filename_literals` is already called with `setup or ""`,
        # so passing None is a known-safe edge.
        ok, reason = validate_challenge_quality(None, GOOD_VALIDATOR)
        assert ok is True, reason


class TestSyntaxGateRejectsValidatorSyntaxError:
    def test_fstring_mismatched_bracket_rejected_with_line_number(self):
        # The exact failure signature from the production logs:
        # `closing parenthesis '}' does not match opening parenthesis '['`
        validator = (
            "data = {'key': [1, 2, 3]}\n"
            "print(f\"got {data['key'][0}\")\n"
        )
        ok, reason = validate_challenge_quality(GOOD_SETUP, validator)
        assert ok is False
        assert "SyntaxError" in reason
        assert "line 2" in reason

    def test_unterminated_string_rejected(self):
        validator = "x = 'open quote\nprint(x)\n"
        ok, reason = validate_challenge_quality(GOOD_SETUP, validator)
        assert ok is False
        assert "SyntaxError" in reason

    def test_unmatched_paren_rejected(self):
        validator = "print((1 + 2)\n"
        ok, reason = validate_challenge_quality(GOOD_SETUP, validator)
        assert ok is False
        assert "SyntaxError" in reason

    def test_reason_includes_remediation_hint(self):
        # The feedback string is fed verbatim back to the LLM as
        # `rejection_feedback` — it should include the f-string-specific
        # remediation so attempt 2 doesn't repeat the same shape.
        validator = "print(f'val {[0][0}')\n"
        ok, reason = validate_challenge_quality(GOOD_SETUP, validator)
        assert ok is False
        assert "f-string" in reason
        assert "Pre-compute" in reason

    def test_validator_syntax_error_runs_before_marker_check(self):
        # If the syntax check is correctly hoisted to the top, a script
        # that ALSO contains `random.seed(...)` reports the syntax
        # error, not the marker rejection — line 1 already broke parse.
        validator = (
            "import random\n"
            "random.seed(7)\n"
            "print(f\"x {[0}\")\n"
        )
        ok, reason = validate_challenge_quality(GOOD_SETUP, validator)
        assert ok is False
        assert "SyntaxError" in reason
        # Marker text MUST NOT appear — syntax must short-circuit.
        assert "random.seed" not in reason


class TestSyntaxGateRejectsSetupSyntaxError:
    def test_setup_syntax_error_rejected(self):
        bad_setup = "open('x.csv','w'.write('a\\n1\\n')\n"
        ok, reason = validate_challenge_quality(bad_setup, GOOD_VALIDATOR)
        assert ok is False
        assert "setup_script" in reason
        assert "SyntaxError" in reason

    def test_setup_fstring_error_rejected(self):
        bad_setup = "x = 1\nopen('y.csv','w').write(f'val {[x][0}')\n"
        ok, reason = validate_challenge_quality(bad_setup, GOOD_VALIDATOR)
        assert ok is False
        assert "setup_script" in reason

    def test_validator_checked_before_setup(self):
        # Both broken — validator's syntax error wins (it's the script
        # the gate is primarily about and the line number for it is
        # what we want surfaced first).
        bad_setup = "x = (1\n"
        bad_validator = "y = (2\n"
        ok, reason = validate_challenge_quality(bad_setup, bad_validator)
        assert ok is False
        assert "validation_script" in reason


class TestPromptContainsFStringRule:
    """The matching defensive rule lives inside the synthetic-challenge
    prompt assembled in `Dreamer._generate_synthetic_challenge`. Pinning
    the literal here means a future refactor can't silently drop the
    rule the LLM relies on to avoid the failure mode in the first
    place.
    """

    def test_rule_11_present_in_dream_module_source(self):
        from pathlib import Path
        import ghost_agent.core.dream as dmod
        src = Path(dmod.__file__).read_text()
        assert "11. F-STRING SAFETY" in src
        # The remediation pattern referenced by the gate's reason must
        # also be in the prompt (so the LLM knows what to do).
        assert "Pre-compute" in src
