"""Tests for browser JS console / uncaught-exception capture.

Regression target: the browser tool used to expose only navigate /
extract_text / screenshot / click — with NO `page.on("console")` or
`page.on("pageerror")` hook. A silent `init()` crash (e.g. a TypeError
referencing an undefined `player.spawnPos`) therefore left only a frozen
loading screen, invisible to the agent's channels. It would then burn
tens of minutes guessing at a "performance" problem. These tests lock in
the capture + the LLM-facing rendering so the agent can SEE the throw.
"""
import sys
import types

import pytest

from ghost_agent.tools.browser import (
    _format_js_diagnostics,
    _runner_script,
)


# ── runner-string structure ──────────────────────────────────────────
def test_runner_has_console_and_pageerror_hooks():
    src = _runner_script()
    assert 'page.on("pageerror"' in src or "page.on('pageerror'" in src
    assert 'page.on("console"' in src or "page.on('console'" in src
    # The diagnostics must be merged into the op result inside the shared
    # context helper so EVERY op benefits, not just one.
    assert "_attach_diagnostics" in src
    assert "js_errors" in src and "console" in src


# ── exercise the in-sandbox capture logic directly ───────────────────
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


class _FakeErr:
    def __init__(self, message, stack=""):
        self.message = message
        self.stack = stack


class _FakeConsole:
    def __init__(self, mtype, text):
        self.type = mtype
        self.text = text


def test_attach_captures_uncaught_exception():
    ns = _load_runner_ns()
    page = _FakePage()
    js_errors, console_msgs = ns["_attach_diagnostics"](page)
    page.handlers["pageerror"](
        _FakeErr(
            "TypeError: Cannot read properties of undefined (reading 'y')",
            "at init (file:///x/index.html:132:14)",
        )
    )
    assert len(js_errors) == 1
    assert "TypeError" in js_errors[0]
    # stack is folded in so the agent gets the line number
    assert "index.html:132" in js_errors[0]


def test_attach_captures_console_errors_and_warnings():
    ns = _load_runner_ns()
    page = _FakePage()
    js_errors, console_msgs = ns["_attach_diagnostics"](page)
    page.handlers["console"](_FakeConsole("error", "boom"))
    page.handlers["console"](_FakeConsole("warning", "careful"))
    page.handlers["console"](_FakeConsole("log", "just noise"))
    types_seen = {c["type"] for c in console_msgs}
    assert "error" in types_seen and "warning" in types_seen


def test_attach_caps_runaway_errors():
    ns = _load_runner_ns()
    page = _FakePage()
    js_errors, console_msgs = ns["_attach_diagnostics"](page)
    for i in range(200):
        page.handlers["pageerror"](_FakeErr(f"err {i}"))
        page.handlers["console"](_FakeConsole("log", f"chatter {i}"))
    assert len(js_errors) <= ns["_MAX_JS_ERRORS"]
    # console is bounded too (a console.log-in-a-loop page can't flood ctx)
    assert len(console_msgs) <= ns["_MAX_CONSOLE_MSGS"] * 2


# ── LLM-facing rendering ─────────────────────────────────────────────
def test_format_clean_page_is_empty():
    assert _format_js_diagnostics({"url": "x", "title": "t"}) == ""
    assert _format_js_diagnostics({}) == ""
    assert _format_js_diagnostics(None) == ""


def test_format_surfaces_uncaught_exception_prominently():
    out = _format_js_diagnostics({
        "js_errors": ["TypeError: x is undefined\nat init (index.html:132)"],
    })
    assert "UNCAUGHT JS EXCEPTIONS" in out
    assert "TypeError" in out
    # Tells the agent to fix the throw before blaming performance.
    assert "performance" in out.lower() or "BEFORE" in out


def test_format_surfaces_console_errors():
    out = _format_js_diagnostics({
        "console": [
            {"type": "error", "text": "ReferenceError: foo"},
            {"type": "log", "text": "ignored noise"},
        ],
    })
    assert "CONSOLE" in out
    assert "ReferenceError" in out
    assert "ignored noise" not in out  # plain logs are filtered out
