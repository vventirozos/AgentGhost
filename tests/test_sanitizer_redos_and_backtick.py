"""Regression tests for three sanitizer defects found in the deep audit:

1. ReDoS — the looser code-fence patterns used an ambiguous whitespace
   prefix (`[ \\t]*(?:[ \\t]*\\n|[ \\t]+)?`) that backtracked
   catastrophically on an unterminated fence (a ~3KB crafted block hung
   the worker for tens of seconds). Reachable from any code the model was
   asked to write/execute.
2. Backtick corruption — `.strip('`')` on the *matched* fence body
   removed legitimate payload backticks (e.g. shell command substitution).
3. CDATA leak — the fully-wrapped CDATA strip used `rfind` (last `]]>`),
   so a body containing `]]>` leaked the stray marker into the output.
"""

import time

from ghost_agent.utils.sanitizer import (
    extract_code_from_markdown,
    _strip_cdata_envelope,
    _MAX_FENCE_SCAN,
)


# -----------------------------------------------------------------
# 1. ReDoS — unterminated fence with a long whitespace run must be fast
# -----------------------------------------------------------------

def test_unterminated_fence_with_whitespace_run_is_fast():
    # Opening fence + a long same-line whitespace run + NO closing fence.
    # This is the exact shape that drove the old regex into catastrophic
    # backtracking. The input is well under _MAX_FENCE_SCAN so the looser
    # secondary/truncated scans actually RUN (we're testing the regex, not
    # the size cap).
    payload = "```" + (" \t" * 4000)  # ~8KB, no closing ```
    assert len(payload) <= _MAX_FENCE_SCAN
    start = time.perf_counter()
    extract_code_from_markdown(payload)
    elapsed = time.perf_counter() - start
    # Linear matching finishes in milliseconds; the bug took >20s at ~3KB.
    assert elapsed < 5.0, f"extract_code_from_markdown took {elapsed:.1f}s (ReDoS)"


def test_unterminated_fence_mixed_ws_then_text_is_fast():
    payload = "```python" + (" " * 5000) + "\t" * 3000 + "neverclosed"
    start = time.perf_counter()
    extract_code_from_markdown(payload)
    assert (time.perf_counter() - start) < 5.0


def test_oversize_input_skips_looser_scans_and_returns_quickly():
    big = "```" + ("x" * (_MAX_FENCE_SCAN + 100))  # no closing fence
    start = time.perf_counter()
    out = extract_code_from_markdown(big)
    assert (time.perf_counter() - start) < 5.0
    # No complete fence → falls through to returning the (stripped) input.
    assert out.startswith("```") or "x" in out


# -----------------------------------------------------------------
# 2. Backtick preservation — matched fence body keeps its backticks
# -----------------------------------------------------------------

def test_shell_command_substitution_backticks_preserved():
    src = "```sh\necho `date`\n```"
    out = extract_code_from_markdown(src)
    assert out == "echo `date`"          # trailing backtick NOT stripped


def test_matched_block_inner_backticks_preserved():
    src = "```js\nconst x = `template ${y}`;\n```"
    out = extract_code_from_markdown(src)
    assert out == "const x = `template ${y}`;"


def test_normal_python_fence_still_extracted():
    src = "Here is code:\n```python\nx = 1\nprint(x)\n```\nThanks!"
    out = extract_code_from_markdown(src)
    assert out == "x = 1\nprint(x)"


# -----------------------------------------------------------------
# 3. CDATA leak — fully-wrapped strip terminates at the FIRST `]]>`
# -----------------------------------------------------------------

def test_cdata_fully_wrapped_terminates_at_first_marker():
    content = "<![CDATA[print('a')]]>\nprint('b')]]>"
    out = _strip_cdata_envelope(content)
    assert out == "print('a')"
    assert "]]>" not in out               # no stray marker leaks through


def test_cdata_simple_wrap_still_works():
    content = "<![CDATA[x = 1]]>"
    assert _strip_cdata_envelope(content) == "x = 1"


def test_cdata_no_envelope_untouched():
    content = "x = 1\nprint(x)"
    assert _strip_cdata_envelope(content) == content
