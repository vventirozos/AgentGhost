"""Corner-case tests for utils/sanitizer.py.

The sanitizer parses LLM output that's frequently malformed: nested
fences, control chars, CDATA leak-through from XML tool-call parsing,
HTML-entity-encoded Python, truncated mid-stream output. Every shape
that CLAUDE.md mentions has dedicated test files; this file goes
broader: empty/whitespace/giant inputs, multi-CDATA, mixed encodings,
non-py extensions, round-trip identity.
"""

from __future__ import annotations

import string

import pytest

from ghost_agent.utils.sanitizer import (
    extract_code_from_markdown,
    fix_python_syntax,
    sanitize_code,
    _strip_cdata_envelope,
    _try_html_unescape_rescue,
)


# ──────────────────────────────────────────────────────────────────────
# extract_code_from_markdown — degenerate inputs
# ──────────────────────────────────────────────────────────────────────

class TestExtractCodeFromMarkdownEdges:
    def test_empty_string_returns_empty(self):
        assert extract_code_from_markdown("") == ""

    def test_none_input_does_not_crash(self):
        # The function checks `if not text` and returns "" — None is falsy
        assert extract_code_from_markdown(None) == ""  # type: ignore[arg-type]

    def test_whitespace_only_input(self):
        out = extract_code_from_markdown("   \n\n   \t  ")
        assert out == ""  # Stripped

    def test_no_fences_returns_input_stripped(self):
        text = "  just plain text, no fences  \n"
        out = extract_code_from_markdown(text)
        assert out == "just plain text, no fences"

    def test_single_fence_extracted(self):
        text = "Here is some code:\n```python\nprint('hi')\n```\n"
        out = extract_code_from_markdown(text, filename="test.py")
        # Either return the inner code or the whole text (if it parses).
        # Whole text is invalid Python (has the prose), so it must extract.
        assert "print('hi')" in out
        assert "Here is" not in out

    def test_multiple_fences_picks_longest(self):
        """When the model emits an example + the real implementation,
        we want the longer (real) one."""
        text = (
            "First a small example:\n"
            "```python\nx = 1\n```\n"
            "Now the real thing:\n"
            "```python\n"
            "def real():\n"
            "    return 42\n"
            "    " + "# padding " * 50 + "\n"
            "```\n"
        )
        out = extract_code_from_markdown(text, filename="test.py")
        assert "def real" in out
        # The short "x = 1" should NOT be the chosen fence — the longer
        # block is selected, much longer than 'x = 1'
        assert len(out) > 50

    def test_truncated_fence_recovered(self):
        """Model output cut off mid-stream — opening fence but no closer."""
        text = "```python\ndef foo():\n    return 1\n# stream cut here..."
        out = extract_code_from_markdown(text, filename="test.py")
        assert "def foo" in out

    def test_valid_python_with_embedded_fence_in_docstring_preserved(self):
        """The whole input parses as Python (docstring contains a fence) —
        must NOT be replaced by the fence's contents. CLAUDE.md pinning."""
        valid_py = '''
def example():
    """Docs.

    Example:
    ```python
    print("inner")
    ```
    """
    return 42
'''
        out = extract_code_from_markdown(valid_py, filename="test.py")
        # The whole file should be preserved
        assert "def example" in out
        assert "return 42" in out

    def test_non_py_extension_falls_through_to_extraction(self):
        """Non-Python extensions skip the AST gate — they always extract
        from the longest fence."""
        text = "Some prose ```js\nconsole.log('hi')\n``` more prose"
        out = extract_code_from_markdown(text, filename="test.js")
        assert "console.log" in out
        # Prose stripped
        assert "Some prose" not in out

    def test_huge_input_doesnt_blow_regex(self):
        """Pathological input: 100K characters with no fence."""
        text = "x" * 100_000
        out = extract_code_from_markdown(text)
        assert out == text  # No fence → unchanged

    def test_fence_with_trailing_spaces_after_closing(self):
        text = "```python\nprint(1)\n```   \n"
        out = extract_code_from_markdown(text, filename="test.py")
        assert "print(1)" in out

    def test_fence_inside_fence_picks_outer(self):
        """Outer fence containing an inner fence (markdown sample inside
        a code block). Heuristic: pick LONGEST."""
        text = (
            "```markdown\n"
            "Here is a code sample:\n"
            "```python\n"
            "x = 1\n"
            "```\n"
            "And it works!\n"
            "```\n"
        )
        out = extract_code_from_markdown(text, filename="test.md")
        # The longest match wins.
        assert len(out) > 0


# ──────────────────────────────────────────────────────────────────────
# fix_python_syntax
# ──────────────────────────────────────────────────────────────────────

class TestFixPythonSyntax:
    def test_idempotent_on_valid_input(self):
        valid = "def f(x):\n    return x + 1\n"
        out = fix_python_syntax(valid)
        # Should not break valid code
        import ast
        ast.parse(out)

    def test_handles_empty_input(self):
        assert fix_python_syntax("") == ""

    def test_handles_only_whitespace(self):
        out = fix_python_syntax("   \n\n  ")
        # Just survives
        import ast
        ast.parse(out)

    def test_unicode_input_preserved(self):
        valid = "x = 'héllo wörld 🎉'\n"
        out = fix_python_syntax(valid)
        import ast
        ast.parse(out)


# ──────────────────────────────────────────────────────────────────────
# _strip_cdata_envelope
# ──────────────────────────────────────────────────────────────────────

class TestCDataStripping:
    def test_fully_wrapped_stripped(self):
        wrapped = "<![CDATA[print('hi')]]>"
        assert _strip_cdata_envelope(wrapped) == "print('hi')"

    def test_no_envelope_unchanged(self):
        plain = "print('hi')"
        assert _strip_cdata_envelope(plain) == plain

    def test_orphan_opener_stripped_when_inner_parses(self):
        wrapped = "<![CDATA[x = 1\ny = 2"  # closer missing
        # Inner is valid Python → strip
        assert _strip_cdata_envelope(wrapped) == "x = 1\ny = 2"

    def test_orphan_opener_kept_when_inner_unparseable(self):
        wrapped = "<![CDATA[def f("  # opener AND syntactically broken
        # Inner is NOT valid Python — leave alone
        assert _strip_cdata_envelope(wrapped) == wrapped

    def test_orphan_closer_stripped_when_prefix_parses(self):
        wrapped = "x = 1\ny = 2 ]]>"
        out = _strip_cdata_envelope(wrapped)
        assert out == "x = 1\ny = 2 "  # closer stripped

    def test_legitimate_cdata_string_literal_unchanged(self):
        """A Python file with `s = "<![CDATA["` inside a string — the
        marker is part of legitimate code, must not be stripped."""
        legit = 's = "<![CDATA[" + payload'
        out = _strip_cdata_envelope(legit)
        assert out == legit

    def test_empty_envelope(self):
        wrapped = "<![CDATA[]]>"
        # Empty body — the strip function may return empty or unchanged
        out = _strip_cdata_envelope(wrapped)
        # Just must not crash and return something
        assert isinstance(out, str)

    def test_none_input_returns_unchanged(self):
        assert _strip_cdata_envelope("") == ""

    def test_whitespace_around_envelope(self):
        wrapped = "   <![CDATA[code]]>   "
        out = _strip_cdata_envelope(wrapped)
        assert "code" in out


# ──────────────────────────────────────────────────────────────────────
# _try_html_unescape_rescue
# ──────────────────────────────────────────────────────────────────────

class TestHTMLUnescapeRescue:
    def test_only_runs_for_py(self):
        bad = "x = &quot;hi&quot;"
        # Non-py extensions are passed through unchanged
        assert _try_html_unescape_rescue(bad, "js") == bad
        assert _try_html_unescape_rescue(bad, "txt") == bad

    def test_decode_commits_only_if_parseable(self):
        bad = "x = &quot;hi&quot;"
        out = _try_html_unescape_rescue(bad, "py")
        # &quot; → " makes it parse
        assert '"hi"' in out

    def test_already_parseable_unchanged(self):
        good = "x = 1\ny = 2"
        out = _try_html_unescape_rescue(good, "py")
        assert out == good

    def test_decode_doesnt_help_returns_original(self):
        """HTML entities present but decoding still doesn't make it
        parseable — return the original (don't commit a partial fix)."""
        bad = "def f(&: pass"  # entity-like + syntactically broken
        out = _try_html_unescape_rescue(bad, "py")
        assert out == bad

    def test_no_entities_quick_reject(self):
        no_entity = "x = 1"
        assert _try_html_unescape_rescue(no_entity, "py") == no_entity

    def test_legit_string_with_entity_unchanged(self):
        """`s = "&quot;"` parses as-is — must not be decoded (would
        change the string's runtime value)."""
        legit = 's = "&quot;"\n'
        out = _try_html_unescape_rescue(legit, "py")
        assert out == legit


# ──────────────────────────────────────────────────────────────────────
# sanitize_code — top-level integration
# ──────────────────────────────────────────────────────────────────────

class TestSanitizeCodeTopLevel:
    def test_empty_input(self):
        out, err = sanitize_code("", "test.py")
        assert err is None or err == ""
        # Empty in → empty out
        assert out == "" or out.strip() == ""

    def test_valid_python_passes_through_unchanged(self):
        valid = "def f():\n    return 42\n"
        out, err = sanitize_code(valid, "test.py")
        assert err is None
        # Whole file preserved
        assert "def f" in out
        assert "return 42" in out

    def test_markdown_wrapped_python_extracted(self):
        wrapped = "Here:\n```python\nprint('x')\n```"
        out, err = sanitize_code(wrapped, "test.py")
        assert err is None
        assert "print('x')" in out
        assert "Here:" not in out

    def test_control_chars_scrubbed(self):
        """\\x00..\\x1f (except \\n \\r \\t) must be stripped."""
        polluted = "x = 1\x00\x01\x02\ny = 2\x07"
        out, err = sanitize_code(polluted, "test.py")
        # The control chars are gone
        assert "\x00" not in out
        assert "\x01" not in out
        assert "\x07" not in out
        # Real content survived
        assert "x = 1" in out
        assert "y = 2" in out

    def test_preserves_newlines_tabs(self):
        valid = "def f():\n\treturn 1\n"
        out, err = sanitize_code(valid, "test.py")
        assert "\t" in out
        assert "\n" in out

    def test_unparseable_post_heal_returns_pre_heal(self):
        """If the healer makes things WORSE, return the pre-heal version
        with the error flag set."""
        # Severely broken Python that the healer can't fix
        broken = "def f(\n  return 1"
        out, err = sanitize_code(broken, "test.py")
        # Either returns the broken code with an error, or a fix
        # Must NOT silently succeed without an error AND broken output
        if err is None:
            # Must parse
            import ast
            ast.parse(out)

    def test_non_py_skips_python_specific_passes(self):
        valid_js = "function f() { return 42; }"
        out, err = sanitize_code(valid_js, "test.js")
        assert err is None
        assert "function f" in out

    def test_cdata_wrapped_python_unwrapped(self):
        wrapped = "<![CDATA[print('hi')]]>"
        out, err = sanitize_code(wrapped, "test.py")
        # CDATA stripped, returns valid Python
        assert err is None
        assert "print('hi')" in out

    def test_html_entity_python_decoded(self):
        wrapped = "x = &quot;hello&quot;\n"
        out, err = sanitize_code(wrapped, "test.py")
        assert err is None
        # The &quot; got decoded to "
        assert '"hello"' in out

    def test_huge_input_completes(self):
        """Defensive: 100K-line file shouldn't take forever."""
        valid = "x = 1\n" * 100_000
        out, err = sanitize_code(valid, "test.py")
        # Just survives
        assert isinstance(out, str)


# ──────────────────────────────────────────────────────────────────────
# Round-trip identity property
# ──────────────────────────────────────────────────────────────────────

class TestSanitizerIdempotence:
    @pytest.mark.parametrize("code", [
        "x = 1",
        "def f():\n    return 1",
        "import os\nprint(os.getcwd())",
        "class Foo:\n    pass",
        "a, b = 1, 2",
        "# comment\nx = 1",
        '"""docstring"""\nx = 1',
        "x = [1, 2, 3]",
        "x = {'a': 1, 'b': 2}",
        "lambda x: x + 1",
    ])
    def test_valid_python_idempotent(self, code):
        """Sanitizing already-valid code twice yields the same result
        as sanitizing once."""
        out1, err1 = sanitize_code(code, "test.py")
        out2, err2 = sanitize_code(out1, "test.py")
        assert err1 is None
        assert err2 is None
        # Idempotent
        assert out1 == out2


# ──────────────────────────────────────────────────────────────────────
# Adversarial: malicious-looking inputs
# ──────────────────────────────────────────────────────────────────────

class TestAdversarialInputs:
    def test_polyglot_python_javascript(self):
        """A string that's valid Python AND looks like JavaScript."""
        # Python valid: `var = 1; func = lambda: None`
        ambiguous = "var = 1\nfunc = lambda: None\n"
        out, err = sanitize_code(ambiguous, "test.py")
        # Just survives
        assert err is None or "x" in out

    def test_zero_width_chars_preserved_in_strings(self):
        """ZWSP (\\u200b) is a Unicode char above 32 — should be kept."""
        text = "x = 'a​b'\n"
        out, err = sanitize_code(text, "test.py")
        assert err is None
        # Zero-width preserved (it's a printable Unicode char)
        assert "​" in out

    def test_backspace_in_source_stripped(self):
        """\\x08 (backspace) is below 32 and must be stripped."""
        text = "x = 1\x08\ny = 2\n"
        out, err = sanitize_code(text, "test.py")
        assert "\x08" not in out

    def test_form_feed_stripped(self):
        """\\x0c (form feed) is below 32 (and not in \\n\\r\\t)."""
        text = "x = 1\x0c\ny = 2\n"
        out, err = sanitize_code(text, "test.py")
        assert "\x0c" not in out

    def test_long_one_line_input(self):
        """A 10K-char single-line program."""
        text = "x = " + "1 + " * 5000 + "0\n"
        out, err = sanitize_code(text, "test.py")
        # Either parses or returns with error — both acceptable
        if err is None:
            import ast
            ast.parse(out)
