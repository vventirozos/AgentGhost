"""Tests for console-error source location (file:line:col) capture.

Regression target (req 70): a script PARSE error fires ``pageerror`` with
NO stack frames — "Unexpected identifier 't'" arrived with no file/line,
the agent blamed em-dashes, and the real bug (an unescaped apostrophe
inside a single-quoted string) survived an 807-second debugging session.
The file:line:col for parse errors only ever arrives on the ``console``
event's ``msg.location``, so the runner must capture it and the
LLM-facing formatter must render it next to the message.
"""
import sys
import types

from ghost_agent.tools.browser import (
    _format_js_diagnostics,
    _runner_script,
)


def _load_runner_ns():
    """Exec the runner source with a stub playwright module so we can
    unit-test `_attach_diagnostics` (which lives inside the sandbox
    runner string) without a real browser."""
    if "playwright" not in sys.modules:
        pw = types.ModuleType("playwright")
        aa = types.ModuleType("playwright.async_api")
        aa.async_playwright = lambda *a, **k: None
        pw.async_api = aa
        sys.modules["playwright"] = pw
        sys.modules["playwright.async_api"] = aa
    ns: dict = {}
    exec(compile(_runner_script(), "<runner>", "exec"), ns)
    return ns


class _FakePage:
    def __init__(self):
        self.handlers = {}

    def on(self, event, cb):
        self.handlers[event] = cb


class _FakeConsole:
    def __init__(self, mtype, text, location=None):
        self.type = mtype
        self.text = text
        if location is not None:
            self.location = location


# ── capture ──────────────────────────────────────────────────────────
def test_console_location_is_captured_as_file_line_col():
    ns = _load_runner_ns()
    page = _FakePage()
    _, console_msgs = ns["_attach_diagnostics"](page)
    page.handlers["console"](_FakeConsole(
        "error", "Uncaught SyntaxError: Unexpected identifier 't'",
        location={
            "url": "file:///workspace/projects/00cb/data.js",
            "lineNumber": 34,   # playwright is 0-based
            "columnNumber": 46,
        },
    ))
    assert len(console_msgs) == 1
    # 0-based → 1-based, and the url is shortened to its last 2 segments
    assert console_msgs[0]["loc"] == "00cb/data.js:35:47"


def test_console_without_location_still_captures_text():
    ns = _load_runner_ns()
    page = _FakePage()
    _, console_msgs = ns["_attach_diagnostics"](page)
    # No .location attribute at all (older playwright / synthetic msg)
    page.handlers["console"](_FakeConsole("error", "boom"))
    assert len(console_msgs) == 1
    assert console_msgs[0]["text"] == "boom"
    assert "loc" not in console_msgs[0]


def test_console_with_empty_url_omits_loc():
    ns = _load_runner_ns()
    page = _FakePage()
    _, console_msgs = ns["_attach_diagnostics"](page)
    page.handlers["console"](_FakeConsole(
        "error", "boom", location={"url": "", "lineNumber": 0,
                                   "columnNumber": 0},
    ))
    assert "loc" not in console_msgs[0]


# ── LLM-facing rendering ─────────────────────────────────────────────
def test_format_renders_location_before_message():
    out = _format_js_diagnostics({
        "console": [
            {"type": "error",
             "text": "Uncaught SyntaxError: Unexpected identifier 't'",
             "loc": "00cb/data.js:35:47"},
        ],
    })
    assert "00cb/data.js:35:47" in out
    # location precedes the message so the agent reads file:line FIRST
    assert out.index("data.js:35:47") < out.index("Unexpected identifier")


def test_format_without_location_is_unchanged():
    out = _format_js_diagnostics({
        "console": [{"type": "error", "text": "ReferenceError: foo"}],
    })
    assert "ReferenceError: foo" in out
    assert "— " not in out.split("ReferenceError")[0].split("•")[-1]
