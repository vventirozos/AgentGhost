"""Regression test for the docstring-fence mangling bug.

`extract_code_from_markdown` used to always extract the longest
triple-backtick fence content, regardless of whether the input was
"prose wrapping code" or "raw code that happens to contain fenced
examples in a docstring". The second case is common in real Python
— sphinx / mkdocs / rst-style docstrings embed fenced examples
constantly. Before the fix, submitting:

    '''Example:
    ```python
    x = 1
    ```
    '''
    def main():
        return 42

…as a .py file got SILENTLY replaced with just `x = 1` before
execution. The agent then ran the stub, got exit 0, reported
"success", and the actual logic was gone.

Fix: when the target language is Python AND the input parses as
valid Python, extract_code_from_markdown returns the input
unchanged. The prose-wrapping-code path still works because
prose + fences does NOT parse as valid Python.
"""

from ghost_agent.utils.sanitizer import extract_code_from_markdown, sanitize_code


_RAW_PY_WITH_FENCED_DOCSTRING = '''"""Example docstring:
```python
x = 1
```
"""
def main():
    return 42
'''


# ------------------------------------------------------------------
# The bug scenario: raw Python file with fenced docstring examples
# ------------------------------------------------------------------

def test_raw_python_with_fenced_docstring_is_preserved():
    out = extract_code_from_markdown(_RAW_PY_WITH_FENCED_DOCSTRING, filename="foo.py")
    assert "def main()" in out, f"BUG: main() dropped; got {out!r}"
    assert "return 42" in out, "function body dropped"
    assert "Example docstring" in out, "docstring content dropped"


def test_sanitize_code_does_not_mangle_fenced_docstring_py():
    cleaned, err = sanitize_code(_RAW_PY_WITH_FENCED_DOCSTRING, "foo.py")
    assert err is None, f"should parse cleanly; got err={err!r}"
    assert "def main()" in cleaned


def test_complex_sphinx_style_docstring_preserved():
    """A more realistic docstring with multiple embedded code examples."""
    src = '''"""Module-level doc.

Basic usage::
    ```python
    from foo import solve
    solve(42)
    ```

Advanced usage::
    ```python
    result = solve(42, mode="fast")
    assert result > 0
    ```
"""

def solve(x, mode="slow"):
    if mode == "fast":
        return x * 2
    return x
'''
    cleaned, err = sanitize_code(src, "foo.py")
    assert err is None
    assert "def solve(x, mode=" in cleaned
    assert "Advanced usage" in cleaned
    assert "Module-level doc" in cleaned


# ------------------------------------------------------------------
# Normal LLM patterns still work
# ------------------------------------------------------------------

def test_prose_wrapping_python_still_extracted():
    md = "Here is the code:\n```python\ndef hello():\n    return 'hi'\n```\nDone."
    out = extract_code_from_markdown(md, filename="foo.py")
    assert out.strip() == "def hello():\n    return 'hi'"


def test_prose_wrapping_still_extracted_without_filename():
    """Backwards compatibility: existing callers that don't pass
    filename get the old behaviour."""
    md = "Here:\n```python\nprint(1)\n```"
    out = extract_code_from_markdown(md)
    assert out == "print(1)"


def test_javascript_still_extracts_without_ast_gate():
    """The ast-parse gate only applies to .py. Non-python files
    extract fences as before."""
    md = "// comment\n```js\nconsole.log(1)\n```"
    out = extract_code_from_markdown(md, filename="foo.js")
    assert out == "console.log(1)"


def test_unfenced_python_unchanged():
    plain = "def f(x):\n    return x + 1\n"
    out = extract_code_from_markdown(plain, filename="foo.py")
    assert out.strip() == "def f(x):\n    return x + 1"


# ------------------------------------------------------------------
# Syntax-broken python: the fence IS the intended payload → extract
# ------------------------------------------------------------------

def test_syntax_broken_python_falls_through_to_fence_extraction():
    """When the outer text is NOT valid Python, the fenced content
    is probably the real payload the caller meant. Keep extracting."""
    broken = "def broken(\n```python\nfixed_code = 1\n```"
    out = extract_code_from_markdown(broken, filename="foo.py")
    assert out == "fixed_code = 1"


def test_llm_style_prefix_and_single_fence():
    """The canonical LLM output shape: a short preamble, then one
    fenced code block. Must extract the fence."""
    llm_out = (
        "Here's the solution:\n\n"
        "```python\n"
        "def solve(n):\n"
        "    return sum(range(n))\n"
        "```\n\n"
        "This is O(n)."
    )
    out = extract_code_from_markdown(llm_out, filename="foo.py")
    assert "def solve(n)" in out
    assert "Here's the solution" not in out
    assert "This is O(n)" not in out


# ------------------------------------------------------------------
# Edge: Python that HAS fences AND is valid — must preserve
# ------------------------------------------------------------------

def test_valid_python_with_top_level_fenced_comment_preserved():
    """A module-level string (common in notebooks → scripts conversion)
    that contains fences inside it. Must not be mangled."""
    src = '''_DOC = """
Example:
```python
x = 1
```
"""
print(_DOC[:10])
'''
    cleaned, err = sanitize_code(src, "foo.py")
    assert err is None
    assert "print(_DOC[:10])" in cleaned
    assert "_DOC =" in cleaned


def test_non_python_extension_still_extracts_even_if_looks_like_py():
    """A .sh / .md file that contains fence patterns must still extract.
    The ast-parse gate is Python-only."""
    src = '''# My shell script
```bash
echo "hello"
```
'''
    out = extract_code_from_markdown(src, filename="foo.sh")
    # No ast gate for .sh → extracts
    assert 'echo "hello"' in out
