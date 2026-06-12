"""Contextual <think>-tag fragment filter for the thinking display stream.

Production symptom: the word "think" was systematically missing from
every rendered thought ("Let me think about…" displayed as "Let me
about…") because the stream filter dropped ANY token whose stripped form
was "think" or ">" — the model legitimately emits " think" as its own
token. The filter is now contextual: a bare "think"/"think>"/">" token
is suppressed only when the accumulated stream immediately before it
ends with the matching tag opener. The filter is display-only; the
accumulated content was never affected.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.agent import _is_think_tag_fragment


def _render(tokens):
    """Simulate the stream loop: accumulate first, then filter for display
    (both call sites append the token before calling the filter)."""
    acc, shown = "", []
    for tok in tokens:
        acc += tok
        if not _is_think_tag_fragment(tok, acc):
            shown.append(tok)
    return "".join(shown)


def test_word_think_as_own_token_survives():
    """The production regression, verbatim token shape."""
    assert _render(["Let me", " think", " about what would be interesting"]) \
        == "Let me think about what would be interesting"


def test_word_think_at_chunk_start_survives():
    assert _render(["I should ", "think", " of something"]) \
        == "I should think of something"


def test_fragmented_open_tag_suppressed():
    assert _render(["<", "think", ">", "reasoning starts"]) == "reasoning starts"


def test_fragmented_close_tag_suppressed():
    assert _render(["</think", ">", " tail"]) == " tail"


def test_whole_tag_tokens_suppressed():
    assert _is_think_tag_fragment("<think>", "<think>")
    assert _is_think_tag_fragment("</think>", "x</think>")


def test_prose_comparison_operator_survives():
    """Bare '>' is only a fragment after '<think' / '</think'."""
    assert _render(["5 ", ">", " 3 holds"]) == "5 > 3 holds"


def test_think_with_punctuation_survives():
    assert _render(["I ", "think", ", therefore"]) == "I think, therefore"


def test_filter_is_display_only_contract():
    """Both call sites accumulate BEFORE filtering — the helper takes the
    accumulated stream including the token and never mutates anything.
    Pin the signature so a refactor doesn't quietly move it into the
    content path."""
    acc = "Let me think"
    assert _is_think_tag_fragment(" think", acc + " think") is False
    assert acc == "Let me think"
