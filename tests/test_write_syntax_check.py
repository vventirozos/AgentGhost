"""Tests for the post-write syntax check on sandbox file writes.

Regression target (req EA → req 70): the agent wrote data.js / game.js
with unescaped apostrophes inside single-quoted strings, every write
returned a bare SUCCESS, the verifier confirmed the build without loading
it, and the user discovered the parse error by clicking a dead button.
Write time is the cheapest place to catch this: the same tool result that
says SUCCESS must also name the broken line.
"""
import shutil

import pytest

from ghost_agent.tools.file_system import (
    _syntax_feedback,
    tool_replace_text,
    tool_write_file,
)

_NODE = shutil.which("node")


# ── helper-level checks ──────────────────────────────────────────────
async def test_python_syntax_error_is_reported(tmp_path):
    p = tmp_path / "broken.py"
    p.write_text("def f(:\n    pass\n")
    note = await _syntax_feedback(p, "broken.py")
    assert "SYNTAX CHECK FAILED" in note
    assert "line 1" in note


async def test_python_clean_file_is_silent(tmp_path):
    p = tmp_path / "ok.py"
    p.write_text("def f():\n    return 1\n")
    assert await _syntax_feedback(p, "ok.py") == ""


async def test_json_syntax_error_is_reported(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text('{"a": 1,}')
    note = await _syntax_feedback(p, "broken.json")
    assert "SYNTAX CHECK FAILED" in note


async def test_unknown_extension_is_skipped(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("anything 'doesn't matter' here")
    assert await _syntax_feedback(p, "notes.md") == ""


@pytest.mark.skipif(_NODE is None, reason="node not on PATH")
async def test_js_apostrophe_bug_is_reported_with_line(tmp_path):
    # The exact req-70 failure shape: a contraction inside a
    # single-quoted string ends the string early.
    p = tmp_path / "data.js"
    p.write_text("const POOL = [\n  'every decision you didn't make',\n];\n")
    note = await _syntax_feedback(p, "data.js")
    assert "SYNTAX CHECK FAILED" in note
    assert "data.js:2" in note or "Unexpected identifier" in note


@pytest.mark.skipif(_NODE is None, reason="node not on PATH")
async def test_js_clean_file_is_silent(tmp_path):
    p = tmp_path / "ok.js"
    p.write_text("const x = \"didn't break\";\nconsole.log(x);\n")
    assert await _syntax_feedback(p, "ok.js") == ""


# ── tool-level integration ───────────────────────────────────────────
async def test_write_file_appends_warning_for_broken_python(tmp_path):
    res = await tool_write_file("bad.py", "def f(:\n    pass\n", tmp_path)
    assert res.startswith("SUCCESS")
    assert "SYNTAX CHECK FAILED" in res
    # the file IS written — the agent must be able to read & fix it
    assert (tmp_path / "bad.py").exists()


async def test_write_file_clean_python_has_no_warning(tmp_path):
    res = await tool_write_file("good.py", "x = 1\n", tmp_path)
    assert res.startswith("SUCCESS")
    assert "SYNTAX CHECK FAILED" not in res


async def test_replace_text_exact_match_reports_introduced_breakage(tmp_path):
    (tmp_path / "mod.py").write_text("value = 1\n")
    res = await tool_replace_text(
        "mod.py", "value = 1", "value = (1\n", tmp_path,
    )
    assert "SUCCESS" in res
    assert "SYNTAX CHECK FAILED" in res


async def test_replace_text_clean_edit_has_no_warning(tmp_path):
    (tmp_path / "mod2.py").write_text("value = 1\n")
    res = await tool_replace_text("mod2.py", "value = 1", "value = 2", tmp_path)
    assert "SUCCESS" in res
    assert "SYNTAX CHECK FAILED" not in res
