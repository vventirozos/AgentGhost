"""Live log console (2026-07-13) — source pins.

The pretty-log stream already reaches the browser over the WebSocket
(it drives the face + planner monologue); the console makes it readable:
a header button toggles a bottom drawer fed from a ring buffer that
collects even while closed. Verified live via Playwright (marker line
appended to the agent log appeared in the open drawer; history survived
close/reopen); these pins keep the wiring from silently regressing.
"""

import re
from pathlib import Path

import pytest

_STATIC = Path(__file__).resolve().parent.parent / "interface" / "static"


@pytest.fixture(scope="module")
def html() -> str:
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js() -> str:
    return (_STATIC / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css() -> str:
    return (_STATIC / "style.css").read_text(encoding="utf-8")


def test_markup_present(html):
    for el in ("logs-btn", "log-console", "log-console-body",
               "log-clear", "log-close", "log-resume"):
        assert f'id="{el}"' in html, f"missing #{el}"


def test_stream_feeds_console(js):
    # The ws 'log' branch must feed the ring buffer...
    log_branch = js[js.index("if (data.type === 'log')"):][:800]
    assert "pushLogEntry(data.content, data.is_error)" in log_branch
    # ...which collects even while the drawer is closed (push happens
    # unconditionally; only RENDERING is gated on visibility).
    push_fn = js[js.index("function pushLogEntry"):][:900]
    assert "logBuffer.push(entry)" in push_fn
    assert "classList.contains('hidden')" in push_fn


def test_buffer_and_dom_are_capped(js):
    assert "LOG_BUFFER_CAP = 500" in js
    assert "logBuffer.shift()" in js
    assert "removeChild(logConsoleBody.firstChild)" in js


def test_ansi_is_stripped(js):
    # The server broadcasts raw pretty-log lines WITH ANSI escapes.
    assert re.search(r"ANSI_ESCAPE_RE\s*=", js)
    assert "replace(ANSI_ESCAPE_RE" in js


def test_scroll_pinning_contract(js):
    # Follows the tail until the user scrolls up; resume pill returns.
    assert "logPinned" in js
    assert "scrollTop = logConsoleBody.scrollHeight" in js
    assert "log-resume" in js or "logResumeBtn" in js


def test_console_styles_present(css):
    body_rule = css[css.index("#log-console-body"):][:600]
    assert "overflow-y: auto" in body_rule
    assert "position: fixed" in css[css.index("#log-console {"):][:400]


def test_css_version_bumped_with_markup(html):
    # The drawer markup shipped with new CSS — a stale cached stylesheet
    # would render it unstyled over the chat.
    assert "style.css?v=3.0" not in html
