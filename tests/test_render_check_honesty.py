"""RENDER_CHECK must not certify a render it cannot see (2026-09-15).

The probe answered "has_content — the frame contains visual content" for
any frame that was not near-uniform, and the agent believed it. Live (req
4b518a82) three screenshots of one Telegram post captured the share
overlay — a gradient with a 3-button pill and nothing else — each scored
13% dominant / 182 colours → HAS_CONTENT, and the turn spent three
screenshot+vision rounds re-confirming an empty frame before filing the
answer it was looking for as "not obtained".

No pixel statistic separates the two populations. Measured on labelled
captures from the live sandbox: the empty overlay's edge density (0.0111)
is HIGHER than a correctly rendered desktop UI's (0.0085), and its
dominant-colour share (13%) is lower than a correctly rendered text
page's (87%). So the probe reports its numbers and states what it does
not know, and the DOM-side text count is carried alongside as a separate
instrument.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import MagicMock

from ghost_agent.tools.browser import analyze_screenshot_render, tool_browser


def _make_sandbox_stub(output: str, exit_code: int = 0):
    stub = MagicMock()
    stub.last_command = None

    def _execute(cmd, timeout=300, **kwargs):
        stub.last_command = cmd
        return output, exit_code

    stub.execute = _execute
    return stub


def _overlay_like(path):
    """A frame with the live overlay's shape: a wide smooth gradient with
    a small white pill on it. Non-uniform by every pixel measure, and
    carrying no page content whatsoever."""
    from PIL import Image, ImageDraw
    w, h = 320, 180
    img = Image.new("RGB", (w, h))
    for x in range(w):           # green→yellow gradient, like t.me's
        for y in range(h):
            img.putpixel((x, y), (90 + x // 4, 150 + y // 8, 90))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([110, 78, 210, 102], radius=12, fill=(255, 255, 255))
    d.rectangle([140, 86, 180, 94], fill=(40, 120, 200))
    img.save(path)
    return path


def test_a_content_free_overlay_is_not_called_content(tmp_path):
    """FAILS IF: the verdict/note asserts the frame has content.

    The shipped world: this exact frame shape returned
    'has_content — the frame contains visual content'.
    """
    r = analyze_screenshot_render(_overlay_like(tmp_path / "overlay.png"))
    assert r is not None
    assert r["verdict"] != "has_content"
    assert r["verdict"] == "indeterminate"
    assert "contains visual content" not in r["note"]


def test_the_note_states_its_own_limit_and_names_the_recapture_route(tmp_path):
    """FAILS IF: the note is shortened back to a bare verdict.

    The note is the only place the model is told (a) this check cannot
    confirm its content arrived and (b) what to do about a lazy page —
    the route the 2026-09-14 run found by accident and the 09-15 run
    never did.
    """
    r = analyze_screenshot_render(_overlay_like(tmp_path / "o2.png"))
    note = r["note"].lower()
    assert "cannot confirm" in note
    assert "settle_ms" in note
    assert "interact" in note


def test_a_blank_frame_is_still_called_blank(tmp_path):
    """FAILS IF: loosening the claim also loosened the one real verdict.

    'uniform' is the narrow thing this instrument CAN detect; it must
    survive untouched.
    """
    from PIL import Image
    p = tmp_path / "sky.png"
    Image.new("RGB", (120, 90), (120, 170, 230)).save(p)
    r = analyze_screenshot_render(p)
    assert r["verdict"] == "uniform"
    assert "BLANK" in r["note"]


def test_numbers_are_still_reported(tmp_path):
    """FAILS IF: dropping the claim also dropped the evidence.

    Removing a gate means KEEPING the signal — the operator and the
    verifier both read these two numbers.
    """
    r = analyze_screenshot_render(_overlay_like(tmp_path / "o3.png"))
    assert isinstance(r["dominant_pct"], float)
    assert isinstance(r["distinct_colors"], int)
    assert r["distinct_colors"] > 6


@pytest.mark.asyncio
async def test_screenshot_reports_dom_text_chars_with_advice(tmp_path):
    """FAILS IF: the DOM count is not surfaced, or surfaced without the
    route when it is tiny.

    38 characters is what `t.me/investigations/363` actually returned
    while the pixels said HAS_CONTENT — the one number that separated the
    empty capture from the loaded one.
    """
    from PIL import Image
    Image.new("RGB", (100, 80), (10, 10, 40)).save(tmp_path / "shot.png")
    payload = {"path": "/workspace/shot.png", "url": "https://t.me/x/363",
               "used_last_url": False, "dom_text_chars": 38}
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(payload)}\n")
    out = await tool_browser(operation="screenshot", url="https://t.me/x/363",
                             out_path="shot.png", sandbox_dir=tmp_path,
                             sandbox_manager=stub)
    assert "DOM_TEXT_CHARS: 38" in out
    assert "settle_ms" in out
    assert "interact" in out


@pytest.mark.asyncio
async def test_a_text_rich_page_gets_the_count_without_the_nag(tmp_path):
    """FAILS IF: the advice fires on every capture.

    A loaded page must not be told to re-take it; noise on the success
    path is how a real warning gets ignored.
    """
    from PIL import Image
    Image.new("RGB", (100, 80), (10, 10, 40)).save(tmp_path / "shot.png")
    payload = {"path": "/workspace/shot.png", "url": "https://example.com/",
               "used_last_url": False, "dom_text_chars": 9175}
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(payload)}\n")
    out = await tool_browser(operation="screenshot", url="https://example.com/",
                             out_path="shot.png", sandbox_dir=tmp_path,
                             sandbox_manager=stub)
    assert "DOM_TEXT_CHARS: 9175" in out
    assert "settle_ms" not in out.split("DOM_TEXT_CHARS")[1]


@pytest.mark.asyncio
async def test_a_runner_without_the_field_omits_the_line(tmp_path):
    """FAILS IF: the host assumes the field exists.

    A sandbox provisioned before this change ships the OLD runner; the
    formatter must degrade to no line, never to 'DOM_TEXT_CHARS: None'.
    """
    from PIL import Image
    Image.new("RGB", (100, 80), (10, 10, 40)).save(tmp_path / "shot.png")
    payload = {"path": "/workspace/shot.png", "url": "https://example.com/",
               "used_last_url": False}
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(payload)}\n")
    out = await tool_browser(operation="screenshot", url="https://example.com/",
                             out_path="shot.png", sandbox_dir=tmp_path,
                             sandbox_manager=stub)
    assert "DOM_TEXT_CHARS" not in out
