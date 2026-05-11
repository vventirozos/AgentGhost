"""Regression: the task-tree-regurgitation scrubber must NOT strip
the bracket label from indented markdown links.

The old pattern at ``agent.py`` ``re.sub(r'(?m)^\\s*(?: )\\s*\\[.*?\\].*?\\n?', '', ui_content)``
matched any line beginning with whitespace + space + ``[anything]``.
It was meant to strip task-tree status lines like
``  [task_001] description (IN_PROGRESS)``, but it also clobbered
legitimate markdown such as:

    here are some references:
      [docs](https://example.com/docs)
      [api](https://example.com/api)

    →

    here are some references:
    (https://example.com/docs)
    (https://example.com/api)

Indented markdown is common in deep_research and web-summary
output. Naked `(url)` lines are user-visible damage.

Fix: tighten the regex to require a task-tree-shape token inside
the brackets (``task_NN`` id or one of the status keywords). The
two follow-up regexes (status-keyword line and ``task_NN`` id
line) already cover the canonical task-tree shape, so this one
just needs to stop firing on plain indented markdown.
"""
import re


# The exact regex from agent.py (post-fix).
PATTERN = (
    r'(?m)^\s*\[(?:task_\d+|IN_PROGRESS|READY|PENDING|DONE|FAILED|BLOCKED)\b'
    r'[^\]]*\].*?\n?'
)


def _scrub(text: str) -> str:
    return re.sub(PATTERN, '', text)


# ---------------------------------------------------------------------------
# Negative cases: markdown / prose must survive untouched.
# ---------------------------------------------------------------------------

def test_indented_markdown_link_survives():
    out = _scrub('  [docs](https://example.com/docs) for more')
    assert out == '  [docs](https://example.com/docs) for more'


def test_indented_markdown_link_in_list_survives():
    text = (
        "References:\n"
        "  [docs](https://example.com/docs)\n"
        "  [api](https://example.com/api)\n"
        "  [blog](https://example.com/blog) — recent post\n"
    )
    out = _scrub(text)
    assert "[docs]" in out
    assert "[api]" in out
    assert "[blog]" in out
    assert "(https://example.com/docs)" in out


def test_bullet_list_links_survive():
    text = "* [docs](https://x)\n* [api](https://y)"
    assert _scrub(text) == text


def test_bracket_in_middle_of_line_survives():
    text = "see the [doc] for details"
    assert _scrub(text) == text


def test_footnote_style_brackets_survive():
    """Citation lists like '  [1] Smith et al' shouldn't get
    chopped — '1' is not task_NN nor a status keyword."""
    text = "  [1] Smith et al\n  [2] Jones\n"
    assert _scrub(text) == text


# ---------------------------------------------------------------------------
# Positive cases: real task-tree shapes must still be stripped.
# ---------------------------------------------------------------------------

def test_task_id_line_is_stripped():
    out = _scrub("  [task_001] do the thing (IN_PROGRESS)")
    assert "[task_001]" not in out


def test_top_level_task_line_is_stripped():
    out = _scrub("[task_042] root goal")
    assert "[task_042]" not in out


def test_status_only_bracket_is_stripped():
    """Some task-tree dumps lead with a bracketed status word."""
    out = _scrub("  [DONE] cleanup performed")
    assert "[DONE]" not in out


def test_blocked_status_bracket_is_stripped():
    out = _scrub("  [BLOCKED] waiting on user")
    assert "[BLOCKED]" not in out


def test_mixed_content_keeps_markdown_strips_tasks():
    """The realistic shape: one buffer with both a task-tree dump
    AND a markdown link list. We strip the former, keep the latter."""
    text = (
        "Working on:\n"
        "  [task_001] read inputs (IN_PROGRESS)\n"
        "  [task_002] transform (PENDING)\n"
        "\n"
        "References:\n"
        "  [docs](https://x)\n"
        "  [api](https://y)\n"
    )
    out = _scrub(text)
    assert "[task_001]" not in out
    assert "[task_002]" not in out
    assert "[docs]" in out
    assert "[api]" in out
    assert "(https://x)" in out
