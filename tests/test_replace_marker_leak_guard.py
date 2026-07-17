"""SEARCH/REPLACE marker-leak guards + js/html syntax rollback + search exit-1.

Live incident 2026-07-14 (WebOS index.html): the model packed TWO edits into
one `<<<< SEARCH … ==== … >>>>` envelope. Only the FIRST `====` is the
separator, so the second edit's `====` lines and text were pasted into the
file literally — and the tool reported SUCCESS. The syntax-rollback guard
only covered .py/.json (html passed through), node was invisible under
launchd's minimal PATH (so the post-write JS check silently skipped), and
`file_system search` reported rg's exit-1 "no matches" as a SYSTEM ERROR —
so the agent could neither prevent, detect, nor verify the damage.

Guards under test:
  1. per-block multi-edit envelope rejection in `tool_replace_text`
  2. `_marker_leak` backstop in `_write_replace_guarded` (all replace paths)
  3. `_syntax_regression_js_html` — node-backed rollback for .js/.html
  4. `_find_node` — node resolution that survives launchd's minimal PATH
  5. `tool_file_search` exit-1 normalization ("no matches", not an error)
"""
import pytest

from ghost_agent.tools.file_system import (
    tool_replace_text,
    tool_file_search,
    _find_node,
    _marker_leak,
    _syntax_regression_js_html,
)


@pytest.fixture
def sandbox(tmp_path):
    d = tmp_path / "sandbox"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# 1. Multi-edit envelope rejection (the exact live corruption shape)
# ---------------------------------------------------------------------------

async def test_multi_edit_envelope_rejected_file_unchanged(sandbox):
    target = sandbox / "app.js"
    original = "const a = 1;\nconst b = 2;\nconst c = 3;\n"
    target.write_text(original)
    # Two edits packed into ONE envelope — everything after the first ====
    # (including the second ==== and both texts) is one replace_str.
    payload = (
        "<<<< SEARCH\n"
        "const a = 1;\n"
        "====\n"
        "const a = 10;\n"
        "====\n"
        "const b = 2;\n"
        "====\n"
        "const b = 20;\n"
        ">>>>"
    )
    res = await tool_replace_text("app.js", payload, None, sandbox)
    assert "REJECTED" in res
    assert "envelope" in res
    assert target.read_text() == original, "file must be untouched"
    assert "====" not in target.read_text()


async def test_good_and_bad_envelopes_mixed(sandbox):
    target = sandbox / "app.js"
    target.write_text("const a = 1;\nconst b = 2;\n")
    payload = (
        "<<<< SEARCH\nconst a = 1;\n====\nconst a = 10;\n>>>>\n"
        "<<<< SEARCH\nconst b = 2;\n====\nconst b = 20;\n====\nstray\n>>>>"
    )
    res = await tool_replace_text("app.js", payload, None, sandbox)
    # The clean envelope applies; the malformed one is named as failed.
    assert "SUCCESS" in res and "1 blocks failed" in res
    after = target.read_text()
    assert "const a = 10;" in after
    assert "const b = 2;" in after  # bad envelope NOT applied
    assert "====" not in after


async def test_canonical_single_envelope_still_works(sandbox):
    target = sandbox / "app.js"
    target.write_text("let x = 1;\n")
    payload = "<<<< SEARCH\nlet x = 1;\n====\nlet x = 2;\n>>>>"
    res = await tool_replace_text("app.js", payload, None, sandbox)
    assert res.startswith("SUCCESS")
    assert target.read_text() == "let x = 2;\n"


# ---------------------------------------------------------------------------
# 2. _marker_leak backstop (two-argument replace path)
# ---------------------------------------------------------------------------

async def test_two_arg_replace_adding_marker_line_rejected(sandbox):
    target = sandbox / "notes.txt"
    original = "alpha\nbeta\n"
    target.write_text(original)
    res = await tool_replace_text("notes.txt", "beta", "beta\n====\ngamma", sandbox)
    assert "REJECTED" in res and "marker" in res
    assert target.read_text() == original


async def test_cleanup_edit_removing_markers_is_allowed(sandbox):
    # Count-aware: a file already corrupted with ==== stays editable —
    # the fix REMOVES markers and must not be blocked by its own guard.
    target = sandbox / "broken.txt"
    target.write_text("good line\n====\nmore\n")
    res = await tool_replace_text("broken.txt", "====\nmore", "more", sandbox)
    assert res.startswith("SUCCESS")
    assert "====" not in target.read_text()


def test_marker_leak_ignores_rst_style_underlines():
    # A 5+ equals line (RST/Markdown underline) is NOT a parser separator
    # and must not trip the guard.
    assert _marker_leak("Title\n", "Title\n=====\n") == ""
    assert _marker_leak("a\n", "a\n====\n") != ""


def test_marker_leak_count_aware():
    prev = "x\n====\ny\n"
    assert _marker_leak(prev, prev) == ""          # unchanged count
    assert _marker_leak(prev, "x\ny\n") == ""      # markers removed
    assert _marker_leak(prev, prev + "====\n") != ""  # one MORE added


# ---------------------------------------------------------------------------
# 3. node-backed syntax rollback for .js / .html
# ---------------------------------------------------------------------------

_HAS_NODE = _find_node() is not None
needs_node = pytest.mark.skipif(not _HAS_NODE, reason="node binary not found")


@needs_node
async def test_js_regression_rejected(sandbox):
    target = sandbox / "logic.js"
    original = "function ok() { return 1; }\n"
    target.write_text(original)
    res = await tool_replace_text(
        "logic.js", "return 1;", "return 1; }", sandbox)  # extra brace
    assert "REJECTED" in res and "syntax" in res.lower()
    assert target.read_text() == original


@needs_node
async def test_html_inline_script_regression_rejected(sandbox):
    target = sandbox / "index.html"
    original = "<html><script>\nconst n = 1;\n</script></html>\n"
    target.write_text(original)
    res = await tool_replace_text(
        "index.html", "const n = 1;", "const n = ;", sandbox)
    assert "REJECTED" in res
    assert target.read_text() == original


@needs_node
async def test_html_already_broken_still_editable(sandbox):
    # Regression semantics: a partial fix on an already-broken file must
    # not be blocked just because the file still doesn't parse afterwards.
    target = sandbox / "index.html"
    target.write_text(
        "<html><script>\nconst a = ;\nconst b = ;\n</script></html>\n")
    res = await tool_replace_text(
        "index.html", "const a = ;", "const a = 1;", sandbox)
    assert res.startswith("SUCCESS")
    assert "const a = 1;" in target.read_text()


@needs_node
async def test_js_html_regression_helper_direct():
    assert await _syntax_regression_js_html(
        "const x = 1;\n", "const x = ;\n", "f.js") != ""
    assert await _syntax_regression_js_html(
        "const x = 1;\n", "const x = 2;\n", "f.js") == ""
    # not a regression when prev was already broken
    assert await _syntax_regression_js_html(
        "const x = ;\n", "const y = ;\n", "f.js") == ""


def test_find_node_resolves_on_this_box():
    # On dev boxes node exists at a homebrew path launchd can't see;
    # _find_node must resolve it regardless of PATH.
    if not _HAS_NODE:
        pytest.skip("node genuinely absent")
    assert _find_node() is not None


# ---------------------------------------------------------------------------
# 4. file_system search exit-1 → "no matches", not SYSTEM ERROR
# ---------------------------------------------------------------------------

class _FakeSandboxManager:
    def __init__(self, output, exit_code):
        self._ret = (output, exit_code)
        self.last_cmd = None

    def execute(self, cmd, timeout=None):
        self.last_cmd = cmd
        return self._ret


async def test_search_exit1_sentinel_reports_no_matches(sandbox):
    (sandbox / "f.txt").write_text("hello\n")
    mgr = _FakeSandboxManager(
        "[SYSTEM ERROR]: Process failed (Exit 1) with no output.", 1)
    res = await tool_file_search("====", sandbox, "f.txt", mgr)
    assert "No matches found" in res
    assert "SYSTEM ERROR" not in res
    assert "NOT FOUND" in res


async def test_search_exit1_empty_output_reports_no_matches(sandbox):
    (sandbox / "f.txt").write_text("hello\n")
    mgr = _FakeSandboxManager("", 1)
    res = await tool_file_search("missing", sandbox, "f.txt", mgr)
    assert "No matches found" in res


async def test_search_real_error_exit2_passes_through(sandbox):
    (sandbox / "f.txt").write_text("hello\n")
    mgr = _FakeSandboxManager("rg: regex parse error", 2)
    res = await tool_file_search("[bad", sandbox, "f.txt", mgr)
    assert "regex parse error" in res


async def test_search_matches_still_returned(sandbox):
    (sandbox / "f.txt").write_text("hello\n")
    mgr = _FakeSandboxManager("1:hello", 0)
    res = await tool_file_search("hello", sandbox, "f.txt", mgr)
    assert "1:hello" in res


# ---------------------------------------------------------------------------
# 4. Syntax rejection carries the WOULD-BE lines (2026-07-17, req A3)
# ---------------------------------------------------------------------------

async def test_py_syntax_rejection_shows_would_be_snippet(sandbox):
    """Three opaque rejections in a row sent the model to an unguarded
    patch script that corrupted the file. The rejection must show the
    rejected content around the error line so the mistake is visible,
    and the log line carries the error too (operator-side opacity)."""
    target = sandbox / "parser.py"
    original = ("def parse(x):\n"
                "    if x:\n"
                "        return 1\n"
                "    return 0\n")
    target.write_text(original)
    payload = (
        "<<<< SEARCH\n"
        "        return 1\n"
        "====\n"
        "return 99\n"   # dedented out of the if-block → IndentationError
        ">>>>"
    )
    res = await tool_replace_text("parser.py", payload, None, sandbox)
    assert "REJECTED" in res and "syntax error" in res
    assert "(line 3" in res                       # error location named
    assert "would have read" in res               # snippet header
    assert ">    3 | return 99" in res   # the offending line, marked
    assert "    2 |" in res                       # context line present
    assert target.read_text() == original


def test_would_be_snippet_edge_cases():
    from ghost_agent.tools.file_system import _would_be_snippet
    assert _would_be_snippet("a\nb", "no line info here") == ""
    assert _would_be_snippet("a\nb", "boom (line 99, col 1)") == ""
    out = _would_be_snippet("l1\nl2\nl3\nl4\nl5", "x (line 1, col 1)")
    assert out.splitlines()[0].startswith(">    1 | l1")
