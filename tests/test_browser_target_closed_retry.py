"""Chromium launch-race retry (2026-07-31, probe req 68033190).

The first atomic browser op after recent browser activity can hit
`TargetClosedError: BrowserType.launch_persistent_context: Target page,
context or browser has been closed` — the previous call's Chromium is still
tearing down on the SHARED profile dir when the new launch grabs it. The
profile lock serialises our runner subprocesses, not Chromium's own async
shutdown. Live cost: the agent self-healed (navigate → screenshot again)
but burned ~2 turns and an error result doing so.

The tool now retries ONCE after a short settle when the runner error
mentions TargetClosedError. Any other error must NOT retry (a genuine
failure repeated twice doubles latency for nothing).
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from unittest.mock import MagicMock

from ghost_agent.tools.browser import tool_browser

_TCE_OUTPUT = (
    "[BROWSER_ERR] TargetClosedError: BrowserType.launch_persistent_context: "
    "Target page, context or browser has been closed\n"
)


def _sequenced_stub(outputs):
    """Sandbox stub returning each (output, exit_code) in order; repeats the
    last one if called again."""
    stub = MagicMock()
    stub.calls = 0

    def _execute(cmd, timeout=300, **kwargs):
        idx = min(stub.calls, len(outputs) - 1)
        stub.calls += 1
        return outputs[idx]

    stub.execute = _execute
    return stub


async def test_target_closed_error_retries_once_and_succeeds(tmp_path):
    ok_payload = {"status": 200, "url": "file:///workspace/x.html",
                  "title": "X", "path": "/workspace/shot.png"}
    stub = _sequenced_stub([
        (_TCE_OUTPUT, 1),
        (f"[BROWSER_OK] {json.dumps(ok_payload)}\n", 0),
    ])
    result = await tool_browser(
        operation="screenshot", url="file:///workspace/x.html",
        out_path="shot.png", sandbox_dir=tmp_path, sandbox_manager=stub)
    assert stub.calls == 2
    assert "STATUS: OK" in result
    assert "TargetClosedError" not in result


async def test_target_closed_error_retry_is_bounded():
    """Persistent TargetClosedError → exactly ONE retry, then the error
    surfaces (no retry loop)."""
    stub = _sequenced_stub([(_TCE_OUTPUT, 1)])
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        result = await tool_browser(
            operation="navigate", url="file:///workspace/x.html",
            sandbox_dir=Path(td), sandbox_manager=stub)
    assert stub.calls == 2
    assert "STATUS: ERROR" in result
    assert "TargetClosedError" in result


async def test_other_errors_do_not_retry(tmp_path):
    stub = _sequenced_stub([("[BROWSER_ERR] net::ERR_FILE_NOT_FOUND\n", 1)])
    result = await tool_browser(
        operation="navigate", url="file:///workspace/x.html",
        sandbox_dir=tmp_path, sandbox_manager=stub)
    assert stub.calls == 1
    assert "STATUS: ERROR" in result
