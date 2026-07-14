"""browser tool — 2026-07-14 audit regressions.

Covers:
  * navigate/click formatters now RENDER the runner's text preview (the
    runner computed and shipped it since the nav-preview feature, but the
    host formatter silently dropped it — forcing the follow-up extract_text
    relaunch the preview exists to eliminate);
  * click now surfaces js diagnostics (a click that crashed page JS looked
    identical to one that worked);
  * settle_ms/nav_text_chars/post_click_ms are coerced + plumbed (a
    settle_ms="2s" used to raise a raw ValueError out of the tool);
  * analyze_screenshot_render no longer false-flags white-background TEXT
    pages as BLANK (uniform now requires few distinct colours too);
  * interact screenshots get the same objective RENDER_CHECK as the atomic
    screenshot op (previously bypassed entirely).
"""

import json
import os
import shlex
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import MagicMock

from ghost_agent.tools.browser import (
    _build_op_payload,
    analyze_screenshot_render,
    tool_browser,
)


def _make_sandbox_stub(output: str, exit_code: int = 0):
    stub = MagicMock()
    stub.last_command = None

    def _execute(cmd, timeout=300, **kwargs):
        stub.last_command = cmd
        return output, exit_code

    stub.execute = _execute
    return stub


def _payload_from_cmd(cmd: str) -> dict:
    """The runner is invoked as `python3 -u .browser_runner.py '<json>'`."""
    return json.loads(shlex.split(cmd)[-1])


# ----------------------------------------------------- formatter: text preview

@pytest.mark.asyncio
async def test_navigate_formatter_renders_text_preview(tmp_path):
    payload = {"status": 200, "url": "http://x/", "title": "T",
               "text": "Hello from the page body", "length": 24,
               "truncated": False}
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(payload)}\n")
    result = await tool_browser(operation="navigate", url="http://x/",
                                sandbox_dir=tmp_path, sandbox_manager=stub)
    assert "PAGE TEXT" in result
    assert "Hello from the page body" in result
    assert "LENGTH: 24" in result


@pytest.mark.asyncio
async def test_navigate_formatter_without_text_has_no_text_block(tmp_path):
    payload = {"status": 200, "url": "http://x/", "title": "T"}
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(payload)}\n")
    result = await tool_browser(operation="navigate", url="http://x/",
                                sandbox_dir=tmp_path, sandbox_manager=stub)
    assert "PAGE TEXT" not in result


@pytest.mark.asyncio
async def test_click_formatter_renders_text_and_js_diagnostics(tmp_path):
    payload = {"url": "http://x/", "title": "T",
               "text": "state after click", "length": 17, "truncated": False,
               "js_errors": ["TypeError: boom at init()"]}
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(payload)}\n")
    result = await tool_browser(operation="click", url="http://x/",
                                selector="#go",
                                sandbox_dir=tmp_path, sandbox_manager=stub)
    assert "state after click" in result
    assert "UNCAUGHT JS EXCEPTIONS" in result
    assert "TypeError: boom" in result


# --------------------------------------------------- numeric coercion/plumbing

@pytest.mark.asyncio
async def test_garbage_settle_ms_does_not_raise(tmp_path):
    payload = {"path": "/workspace/s.png", "url": "http://x/"}
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(payload)}\n")
    result = await tool_browser(operation="screenshot", url="http://x/",
                                out_path="s.png", settle_ms="2s",
                                sandbox_dir=tmp_path, sandbox_manager=stub)
    assert "STATUS: OK" in result
    assert _payload_from_cmd(stub.last_command)["settle_ms"] == 0


@pytest.mark.asyncio
async def test_nav_text_chars_plumbed_and_clamped(tmp_path):
    payload = {"status": 200, "url": "http://x/", "title": "T"}
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(payload)}\n")
    await tool_browser(operation="navigate", url="http://x/",
                       nav_text_chars=999_999_999,
                       sandbox_dir=tmp_path, sandbox_manager=stub)
    sent = _payload_from_cmd(stub.last_command)
    assert sent["nav_text_chars"] == 64 * 1024  # clamped to _MAX_TEXT_CHARS

    await tool_browser(operation="navigate", url="http://x/",
                       nav_text_chars=0,
                       sandbox_dir=tmp_path, sandbox_manager=stub)
    assert _payload_from_cmd(stub.last_command)["nav_text_chars"] == 0


@pytest.mark.asyncio
async def test_post_click_ms_plumbed(tmp_path):
    payload = {"path": "/workspace/s.png", "url": "http://x/"}
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(payload)}\n")
    await tool_browser(operation="screenshot", url="http://x/",
                       out_path="s.png", click_center=True, post_click_ms=1500,
                       sandbox_dir=tmp_path, sandbox_manager=stub)
    sent = _payload_from_cmd(stub.last_command)
    assert sent["click_center"] is True
    assert sent["post_click_ms"] == 1500


def test_build_op_payload_defaults_do_not_include_new_keys():
    p = _build_op_payload(op="navigate", url="http://x/", selector=None,
                          out_path=None, wait_until=None, full_page=None,
                          max_chars=None, timeout_ms=30000, tor_proxy=None)
    assert "settle_ms" not in p and "nav_text_chars" not in p and "post_click_ms" not in p


# --------------------------------------------------------- render-check tuning

def test_render_check_text_page_is_not_flagged_blank(tmp_path):
    """A white-background docs page is >80% white but its anti-aliased text
    spans many colour buckets — it must NOT be called BLANK/uniform."""
    from PIL import Image
    img = Image.new("RGB", (120, 90), (255, 255, 255))
    # Simulate anti-aliased glyphs: 30 distinct gray levels sprinkled in.
    for i in range(30):
        g = i * 8
        for j in range(10):
            img.putpixel(((i * 4 + j) % 120, (i * 3 + j) % 90), (g, g, g))
    p = tmp_path / "docs.png"
    img.save(p)
    r = analyze_screenshot_render(p)
    assert r is not None
    assert r["dominant_pct"] >= 0.80          # white really dominates…
    assert r["verdict"] == "has_content"      # …but it is NOT blank


def test_render_check_solid_frame_still_flagged(tmp_path):
    from PIL import Image
    p = tmp_path / "sky.png"
    Image.new("RGB", (120, 90), (135, 206, 235)).save(p)
    r = analyze_screenshot_render(p)
    assert r["verdict"] == "uniform"


# ------------------------------------------------ interact screenshot evidence

@pytest.mark.asyncio
async def test_interact_screenshot_gets_render_check(tmp_path):
    from PIL import Image
    # The PNG the (stubbed) runner "wrote" — solid colour → uniform verdict.
    Image.new("RGB", (100, 80), (10, 10, 40)).save(tmp_path / "shot.png")

    runner_payload = {
        "actions": [{"index": 0, "action": "screenshot", "ok": True,
                     "path": "/workspace/shot.png"}],
        "final_url": "http://x/", "final_title": "T",
        "used_last_url": False, "aborted": False,
    }
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(runner_payload)}\n")
    result = await tool_browser(
        operation="interact", url="http://x/",
        actions=[{"action": "screenshot", "out_path": "shot.png"}],
        sandbox_dir=tmp_path, sandbox_manager=stub)
    assert "RENDER_CHECK: UNIFORM" in result
