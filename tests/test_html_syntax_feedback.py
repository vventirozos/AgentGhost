"""Inline-<script> syntax checking on write.

A single-file browser-OS deliverable hides all its JS inside one inline
<script> block. Before this, _syntax_feedback only parsed .py/.json/.js
files, so a one-character typo in inline JS (`content=` instead of
`content:`) wrote SUCCESS, the browser loaded a blank page, and the model
spent the whole turn re-reading 32KB chunks unable to localise it (observed
live: ~760s + a thinking-loop abort on exactly this bug).

These guard that an inline JS error is now named with its HTML line at
write time, and that clean / module / external scripts do NOT false-positive.
"""

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools.file_system import (
    _inline_js_blocks,
    _remap_node_diag,
    _syntax_feedback,
)

requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node binary required for JS --check"
)


def _write_html(body: str) -> Path:
    t = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8")
    t.write(body)
    t.close()
    return Path(t.name)


# ----------------------------------------------------------- block extraction

def test_inline_blocks_report_html_start_line():
    html = "<html>\n<head></head>\n<body>\n<script>\nvar a=1;\n</script>\n</body>"
    blocks = _inline_js_blocks(html)
    assert len(blocks) == 1
    start_line, src = blocks[0]
    # body group starts on the <script> line; node's own line count includes
    # the leading newline, so a typo on "var a=1;" remaps to its real line 5.
    assert start_line == 4
    assert "var a=1;" in src


def test_inline_blocks_skip_external_and_module_and_nonjs():
    html = (
        '<script src="app.js"></script>'
        '<script type="module">import x from "y";</script>'
        '<script type="application/json">{"a":1}</script>'
        '<script type="text/babel">const x = <div/>;</script>'
        '<script></script>'                       # empty
    )
    assert _inline_js_blocks(html) == []


# --------------------------------------------------------------- remap helper

def test_remap_adds_html_offset_and_keeps_message():
    diag = (
        "/tmp/x.js:12\n"
        "  {content='# Games'},\n"
        "          ^^^^^^^^^^\n"
        "SyntaxError: Invalid shorthand property initializer\n"
        "    at wrapSafe (node:internal/modules/cjs/loader:1)\n"
    )
    out = _remap_node_diag(diag, start_line=89)   # 89 + 12 - 1 == 100
    assert out.startswith("line 100: SyntaxError: Invalid shorthand property initializer")
    assert "{content='# Games'}" in out
    assert "wrapSafe" not in out                  # loader noise dropped


# ----------------------------------------------------- end-to-end on write

@requires_node
def test_inline_js_typo_is_caught_with_line():
    # the exact shape that stumped the agent live: `=` instead of `:`
    html = (
        "<html><body>\n<script>\n"
        "const fs = {\n"
        "  '/Games/README.md': {type:'file', content='# Games'},\n"
        "};\n"
        "</script></body></html>"
    )
    path = _write_html(html)
    try:
        note = asyncio.run(_syntax_feedback(path, "index.html"))
    finally:
        os.unlink(path)
    assert "SYNTAX CHECK FAILED" in note
    assert "Invalid shorthand property initializer" in note
    assert "line 4" in note                       # the content= line


@requires_node
def test_clean_inline_js_is_silent():
    html = (
        "<html><body><div id=x></div>\n<script>\n"
        "const fs={'/a':{type:'dir',children:[]}};\n"
        "function boot(){document.getElementById('x').textContent='ok';}\n"
        "boot();\n</script></body></html>"
    )
    path = _write_html(html)
    try:
        assert asyncio.run(_syntax_feedback(path, "ok.html")) == ""
    finally:
        os.unlink(path)


@requires_node
def test_module_script_not_false_flagged():
    # import/export trips `node --check` as a script; we must skip it
    html = '<html><body><script type="module">import a from "./a.js"; export const z=1;</script></body></html>'
    path = _write_html(html)
    try:
        assert asyncio.run(_syntax_feedback(path, "m.html")) == ""
    finally:
        os.unlink(path)
