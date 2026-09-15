"""Failure classification must read the ERROR, not the advice (2026-09-15).

Tools append a `--- HINT ---` block to failures for the model's benefit,
and the hint enumerates failure modes BY NAME ("If this is a navigation
timeout, …"). `classify_tool_failure` scans patterns in order with first
match winning, so a deterministic

    ValueError: selector '.frag_copy' did not match any element

was booked RETRYABLE on the word "timeout" three paragraphs below it, and
charged to the transient strike budget — a budget no retry can satisfy,
because the selector will never match. Live: req 4b518a82, "Transient
strike 1/4 (timeout)".

Measured across the recorded trajectory corpus: 31 of 53 hinted tool
errors (58.5%) change class once the advice is excluded, all of them
retryable → diagnostic/unknown.
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools.tool_failure import (
    FailureClass,
    classify_tool_failure,
    strip_advisory_sections,
)

_SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"

# The exact live text (req 4b518a82, tool call 22), reproduced from the
# recorded trajectory.
_LIVE_BROWSER_FAILURE = (
    "--- BROWSER RESULT ---\n"
    "STATUS: ERROR\n"
    "Runner failed (exit 1): ValueError: selector '.frag_copy' did not match "
    "any element\n"
    "\n"
    "--- HINT ---\n"
    "If this is a navigation timeout, try wait_until='domcontentloaded' or "
    "raise timeout_ms. If a CLICK timed out or its selector was not found: "
    "each atomic op reloads the page in a fresh context, so elements created "
    "by a previous click (opened windows, menus, dialogs) are GONE — run the "
    "whole flow in one context with operation='interact' and an actions list.\n"
    "-----------"
)


def test_the_live_misclassification_is_fixed():
    """FAILS IF: the hint is classified along with the error.

    The world it fails in is the shipped one: the error is a ValueError
    and the only 'timeout' in the text belongs to the advice.
    """
    cls, match = classify_tool_failure(_LIVE_BROWSER_FAILURE)
    assert cls is FailureClass.DIAGNOSTIC, f"got {cls} on {match!r}"
    assert "timeout" not in match.lower()


def test_a_real_timeout_is_still_retryable():
    """FAILS IF: the stripper eats the error as well as the advice.

    The opposite failure direction, and the more damaging one: a genuine
    transient must keep its retry budget.
    """
    text = ("--- BROWSER RESULT ---\nSTATUS: ERROR\n"
            "Runner failed (exit 1): TimeoutError: Page.goto: Timeout 30000ms "
            "exceeded\n\n--- HINT ---\nTry wait_until='domcontentloaded'.\n-----------")
    cls, _ = classify_tool_failure(text)
    assert cls is FailureClass.RETRYABLE


def test_execute_diagnostic_hint_banner_is_also_stripped():
    """FAILS IF: only the browser's banner shape is handled.

    execute.py writes `--- 💡 DIAGNOSTIC HINT ---` with a different rule
    length and an emoji in the label; a shape-specific stripper misses it.
    """
    text = ("--- EXECUTION RESULT ---\nEXIT CODE: 1\n"
            "NameError: name 'foo' is not defined\n\n"
            "--- 💡 DIAGNOSTIC HINT ---\n"
            "This often means a connection timeout — rate limit? retry.\n"
            "------------------------")
    assert "timeout" not in strip_advisory_sections(text).lower()
    cls, _ = classify_tool_failure(text)
    assert cls is FailureClass.DIAGNOSTIC


def test_text_without_a_hint_is_untouched():
    """FAILS IF: the stripper rewrites ordinary errors."""
    text = "ValueError: selector 'main' did not match any element"
    assert strip_advisory_sections(text) == text


def test_hint_only_text_does_not_become_a_phantom_class():
    """FAILS IF: stripping everything leaves a string that still classifies.

    A result that is nothing BUT advice carries no error to classify;
    returning a confident class there invents a failure.
    """
    text = "\n--- HINT ---\nIf this is a navigation timeout, raise timeout_ms.\n-----------"
    cls, match = classify_tool_failure(text)
    assert cls is FailureClass.UNKNOWN, f"got {cls} on {match!r}"
    # The label must say WHICH kind of unknown — dropping the branch
    # leaves the verdict identical and only the diagnosis poorer, which
    # is exactly the survivor the mutation battery found.
    assert match == "advice-only failure"


def test_content_after_the_hint_block_survives():
    """FAILS IF: the stripper cuts to end-of-text instead of to the rule.

    Nothing appends after a hint TODAY; a cut-to-EOF stripper would
    silently swallow the first thing that ever does.
    """
    text = ("Error: boom\n\n--- HINT ---\nadvice about a timeout\n-----------\n"
            "PermissionError: access denied")
    out = strip_advisory_sections(text)
    assert "PermissionError" in out
    assert "advice about a timeout" not in out
    assert classify_tool_failure(text)[0] is FailureClass.FATAL


_BANNER_RE = re.compile(r"-{2,}\s*[^\n]{0,30}?HINTS?[^\n]{0,20}?-{2,}",
                        re.IGNORECASE)


def _hint_banner_literals():
    """Every `--- … HINT ---` banner a tool appends, found by PARSING the
    tree — `ast.parse` + a walk over string constants, not a text grep.

    Read from the source, never transcribed: a transcribed fixture drifts
    the moment a producer changes its rule length or label. Parsed, not
    grepped, for the reason R4 gives — a text scan also matches the word
    in a COMMENT, so it would report itself covered by prose that
    executes nothing. (It did: the grep version's only `--- HINT ---` hit
    was a comment in tool_failure.py, not browser.py's real literal.)
    """
    import ast

    found = []
    seen = set()
    for path in sorted(_SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:  # pragma: no cover - the suite would be red anyway
            continue
        for node in ast.walk(tree):
            # Plain literals AND the constant pieces of an f-string
            # (`f"\n--- HINT ---\n{hint}"` is a JoinedStr of Constants).
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            m = _BANNER_RE.search(node.value)
            if not m:
                continue
            header = m.group(0).strip()
            if header in seen:
                continue
            seen.add(header)
            found.append((path.name, header))
    return found


def test_every_hint_producer_in_the_tree_is_covered():
    """R1 enumeration — FAILS IF: a new producer invents a banner shape
    the stripper does not match.

    This is the class, not the instance: the defect is 'advice appended
    to an error is classified as the error', and it recurs for every
    banner ever added.
    """
    producers = _hint_banner_literals()
    assert producers, "enumeration found no HINT producers — it has stopped working"
    for filename, header in producers:
        sample = (f"RuntimeError: something deterministic\n\n{header}\n"
                  f"If this is a navigation timeout, raise timeout_ms.\n"
                  f"------------------------")
        stripped = strip_advisory_sections(sample)
        assert "timeout" not in stripped.lower(), (
            f"{filename}: banner {header!r} is not stripped — its advice "
            f"would be classified as the error")


def test_enumeration_actually_fires_on_an_uncovered_shape():
    """R7.2 — FAILS IF: the enumeration above cannot go red.

    Introduce the defect (a banner the regex cannot see) and confirm the
    advice survives. An enumeration never seen to fire is decoration.
    """
    weird = ("RuntimeError: deterministic\n\n== HELPFUL TIP ==\n"
             "If this is a navigation timeout, raise timeout_ms.\n==")
    assert "timeout" in strip_advisory_sections(weird).lower()
    assert classify_tool_failure(weird)[0] is FailureClass.RETRYABLE
