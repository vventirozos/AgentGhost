"""Tests for the browser tool's Tor-reliability + diagnostics fixes.

Changes addressing the browser-over-Tor failures seen in production traces:

  1. wait_until default — when going through a proxy (Tor) and the caller
     hasn't pinned wait_until, the navigation milestone defaults to
     `domcontentloaded` instead of `load`. The full `load` event often
     never fires within the timeout on JS-heavy pages over a slow exit,
     so a page that had already delivered its content would time out.

  2. failure visibility — the operator's live stream now shows the ACTUAL
     runner error cause (truncated), not a bare "runner exit 1".

  3. profile-lock serialization — concurrent Chromium launches on the
     shared persistent profile dir crash each other; browser ops are
     serialized so they queue instead.
"""
import asyncio
import json
import shlex

import pytest
from unittest.mock import patch, MagicMock

from ghost_agent.tools.browser import _build_op_payload, tool_browser


def _make_sandbox_stub(output: str, exit_code: int = 0):
    stub = MagicMock()
    stub.last_command = None
    stub.tor_proxy = None

    def _execute(cmd, timeout=300, **kwargs):
        stub.last_command = cmd
        return output, exit_code

    stub.execute = _execute
    return stub


def _payload_from_command(cmd: str) -> dict:
    """Extract the JSON op-dict the runner was invoked with."""
    # cmd looks like:  python3 -u .browser_runner.py '<json>'
    tokens = shlex.split(cmd)
    return json.loads(tokens[-1])


# --------------------------------------------------------------------------
# 1. wait_until default (unit, via _build_op_payload)
# --------------------------------------------------------------------------
def test_proxy_defaults_wait_until_to_domcontentloaded():
    payload = _build_op_payload(
        "navigate", "http://x.com", None, None, None, None, None,
        30000, "socks5h://127.0.0.1:9050",
    )
    assert payload["wait_until"] == "domcontentloaded"
    # socks5h:// is rewritten to socks5:// for Chromium.
    assert payload["proxy"] == "socks5://127.0.0.1:9050"


def test_explicit_wait_until_always_wins_over_proxy_default():
    payload = _build_op_payload(
        "navigate", "http://x.com", None, None, "networkidle", None, None,
        30000, "socks5://127.0.0.1:9050",
    )
    assert payload["wait_until"] == "networkidle"


def test_no_proxy_leaves_wait_until_unset():
    # Without a proxy (e.g. local file:// fixture) we don't override; the
    # runner keeps its own `load` default.
    payload = _build_op_payload(
        "navigate", "http://x.com", None, None, None, None, None,
        30000, None,
    )
    assert "wait_until" not in payload


# --------------------------------------------------------------------------
# 1b. wait_until default (integration, through tool_browser)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tool_browser_navigate_over_tor_uses_domcontentloaded(tmp_path):
    ok = {"status": 200, "url": "https://example.com/", "title": "Example"}
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(ok)}\n")
    await tool_browser(
        operation="navigate", url="https://example.com/",
        sandbox_dir=tmp_path, sandbox_manager=stub,
        tor_proxy="socks5://127.0.0.1:9050",
    )
    payload = _payload_from_command(stub.last_command)
    assert payload["wait_until"] == "domcontentloaded"


# --------------------------------------------------------------------------
# 2. failure visibility — real cause reaches the operator stream
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_runner_failure_cause_is_logged_to_stream(tmp_path):
    err = "[BROWSER_ERR] TimeoutError: page.goto: Timeout 30000ms exceeded\n"
    stub = _make_sandbox_stub(err, exit_code=1)

    with patch("ghost_agent.tools.browser.pretty_log") as mock_log:
        result = await tool_browser(
            operation="navigate", url="https://slow-spa.example/",
            sandbox_dir=tmp_path, sandbox_manager=stub,
            tor_proxy="socks5://127.0.0.1:9050",
        )

    # The agent-facing return still carries the full cause + a hint.
    assert "TimeoutError" in result
    assert "Timeout 30000ms exceeded" in result

    # The operator's live stream (pretty_log WARNING) now names the cause,
    # not just "runner exit 1".
    warning_msgs = [
        c.args[1] for c in mock_log.call_args_list
        if len(c.args) > 1 and "runner exit" in str(c.args[1])
    ]
    assert warning_msgs, "expected a 'runner exit' warning to be logged"
    assert any("TimeoutError" in m for m in warning_msgs)


@pytest.mark.asyncio
async def test_logged_cause_is_truncated_and_single_line(tmp_path):
    # A long multi-line traceback must be flattened + capped in the stream
    # (the full text still goes to the agent).
    long_tb = "[BROWSER_ERR] RuntimeError: boom\n" + ("x" * 500 + "\n") * 5
    stub = _make_sandbox_stub(long_tb, exit_code=1)

    with patch("ghost_agent.tools.browser.pretty_log") as mock_log:
        await tool_browser(
            operation="navigate", url="https://x.example/",
            sandbox_dir=tmp_path, sandbox_manager=stub,
            tor_proxy="socks5://127.0.0.1:9050",
        )

    warning_msgs = [
        c.args[1] for c in mock_log.call_args_list
        if len(c.args) > 1 and "runner exit" in str(c.args[1])
    ]
    assert warning_msgs
    msg = warning_msgs[0]
    # Flattened to one line (newlines replaced with a visible glyph).
    assert "\n" not in msg
    # Capped — the raw cause was ~2500 chars; the logged line stays bounded.
    assert len(msg) < 400


# --------------------------------------------------------------------------
# 3. Concurrent browser ops are serialized on the shared profile
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_concurrent_browser_ops_are_serialized(tmp_path):
    """Three browser navigations fired at once (as the agent does in a
    single turn) must NOT run their Chromium launches concurrently — that
    corrupts the shared persistent profile and SIGSEGVs the losers. The
    _BROWSER_PROFILE_LOCK must funnel them through one at a time."""
    import threading
    import time

    active = {"now": 0, "max": 0}
    guard = threading.Lock()

    def _execute(cmd, timeout=300, **kwargs):
        with guard:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.05)  # hold the "launch" open so any overlap is observable
        with guard:
            active["now"] -= 1
        return f"[BROWSER_OK] {json.dumps({'status': 200, 'url': 'http://x/', 'title': 't'})}\n", 0

    stub = MagicMock()
    stub.execute = _execute
    stub.tor_proxy = None

    await asyncio.gather(*[
        tool_browser(
            operation="navigate", url=f"https://e{i}.example/",
            sandbox_dir=tmp_path, sandbox_manager=stub,
            tor_proxy="socks5://127.0.0.1:9050",
        )
        for i in range(3)
    ])

    assert active["max"] == 1, f"expected serialized launches, saw {active['max']} concurrent"
