"""Two file_system replace fixes from the GHOSTNET training-session
postmortem (session D4), where a single 1-line fix turned into a ~10-turn
replace→syntax-error→rewrite spiral that lost code.

Bug #1 — flexible-match indentation corruption. The whitespace-flexible
matcher (`r'\s+'.join(words)`) matches starting at the first token, so the
matched span excludes the leading indent of its first line. A raw
`str.replace(match, new_text)` then lands continuation lines at whatever
column the model happened to type, producing "unexpected indent" /
"unindent does not match" — exactly the log's errors. `_reindent_replacement`
re-anchors the block to the matched region's indentation.

Bug #2 — no rollback. A replace that turned a parsing file into a
non-parsing one was persisted anyway (reported SUCCESS + a syntax warning),
destroying the known-good anchor for the next surgical edit. The guard now
refuses such an edit and leaves the file unchanged.

Also covers the upstream root cause: `extract_code_from_markdown` was run
unconditionally and its no-fence `.strip()` dedented the first line of an
indented replacement before the matcher saw it.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import ast

import pytest

from ghost_agent.tools.file_system import (
    tool_replace_text,
    _reindent_replacement,
)


GUARD = (
    "class GhostNet:\n"
    "    def set_input_size(self, input_size):\n"
    "        if self.layer_configs[0].units != input_size:\n"
    "            self.layer_configs[0] = LayerConfig(input_size)\n"
    "            self._initialize_weights()\n"
    "        return self\n"
)
# What the model sends as the SEARCH block (its own approximate indentation).
OLD = (
    "if self.layer_configs[0].units != input_size:\n"
    "    self.layer_configs[0] = LayerConfig(input_size)\n"
    "    self._initialize_weights()"
)


# --------------------------------------------------------- _reindent unit

def test_reindent_anchors_dedented_block():
    file_c = "        x = 1\n        y = 2\n"
    matched = "x = 1\n        y = 2"          # match excludes leading 8 spaces of line 1
    new = "a = 1\nb = 2"                       # model dedented both lines
    out = _reindent_replacement(file_c, matched, new)
    assert out == "a = 1\n        b = 2"       # line 1 bare (file supplies indent), line 2 at 8


def test_reindent_preserves_internal_nesting():
    file_c = "        body()\n"
    matched = "body()"
    new = "if True:\n    body()"
    out = _reindent_replacement(file_c, matched, new)
    # block head sits at the anchor column (8); the model's internal
    # nesting (+4) is preserved, so the nested line lands at 8 + 4 = 12.
    assert out == "if True:\n            body()"


def test_reindent_noop_for_single_line():
    assert _reindent_replacement("    x = 1\n", "x = 1", "y = 2") == "y = 2"


def test_reindent_noop_when_match_not_at_line_start():
    # match begins mid-line (code before it) → not genuine indentation
    file_c = "    z = foo()\n"
    assert _reindent_replacement(file_c, "foo()", "bar()\nbaz()") == "bar()\nbaz()"


# ------------------------------------------------ end-to-end: indentation

async def test_guard_removal_dedented_replacement_parses(tmp_path):
    (tmp_path / "core.py").write_text(GUARD)
    new = ("self.layer_configs[0] = LayerConfig(input_size)\n"
           "self._initialize_weights()")
    res = await tool_replace_text("core.py", OLD, new, tmp_path)
    assert "SUCCESS" in res
    ast.parse((tmp_path / "core.py").read_text())   # must not raise


async def test_guard_removal_absolute_indent_replacement_parses(tmp_path):
    """The model writing the replacement at full absolute indent must also
    work — this is the path the markdown-strip fix protects."""
    (tmp_path / "core.py").write_text(GUARD)
    new = ("        self.layer_configs[0] = LayerConfig(input_size)\n"
           "        self._initialize_weights()")
    res = await tool_replace_text("core.py", OLD, new, tmp_path)
    assert "SUCCESS" in res
    ast.parse((tmp_path / "core.py").read_text())


async def test_replacement_indentation_preserved_without_fences(tmp_path):
    """No code fences → old/new preserved byte-for-byte (no first-line
    dedent from the markdown extractor's no-fence strip)."""
    (tmp_path / "m.py").write_text("def f():\n    a = 1\n    b = 2\n")
    res = await tool_replace_text(
        "m.py", "a = 1\n    b = 2", "a = 10\n    b = 20", tmp_path)
    assert "SUCCESS" in res
    assert (tmp_path / "m.py").read_text() == "def f():\n    a = 10\n    b = 20\n"


# ----------------------------------------------------- end-to-end: rollback

async def test_syntax_breaking_replace_rejected_and_file_unchanged(tmp_path):
    (tmp_path / "core.py").write_text(GUARD)
    res = await tool_replace_text(
        "core.py", "return self", "return self\n      oops()", tmp_path)
    assert "REJECTED" in res and "SUCCESS" not in res
    assert (tmp_path / "core.py").read_text() == GUARD


async def test_clean_replace_still_succeeds(tmp_path):
    (tmp_path / "core.py").write_text(GUARD)
    res = await tool_replace_text(
        "core.py", "return self", "return self  # ok", tmp_path)
    assert "SUCCESS" in res
    assert "# ok" in (tmp_path / "core.py").read_text()


async def test_non_python_replace_not_blocked_by_guard(tmp_path):
    """The in-process regression check only applies to .py/.json; a .txt
    edit is never blocked."""
    (tmp_path / "notes.txt").write_text("hello world\n")
    res = await tool_replace_text("notes.txt", "world", "there", tmp_path)
    assert "SUCCESS" in res
    assert (tmp_path / "notes.txt").read_text() == "hello there\n"
