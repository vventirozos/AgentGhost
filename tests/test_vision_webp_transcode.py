"""Vision node format compatibility — 2026-07-29 regression.

A webp target was sniffed correctly and shipped to the vision node as
`image/webp`, but llama.cpp-based nodes decode images with stb_image (no
webp/tiff support) → the node answered an HTTP error and the whole path
surfaced as "vision node offline". Fix: `_normalize_for_node` re-encodes
anything outside the node-safe set (png/jpeg/gif/bmp) to PNG before
shipping, falling back to the original bytes if Pillow can't decode them.
Companion fix in core/llm.py: `_node_error_detail` puts the HTTP status +
body snippet in the node-failure log instead of the bare class name.
"""

import base64
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import AsyncMock

Image = pytest.importorskip("PIL.Image")

from ghost_agent.tools import vision as vision_mod
from ghost_agent.tools.vision import tool_vision_analysis, _normalize_for_node

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _make_image_bytes(fmt, mode="RGB", size=(8, 8), color=(200, 30, 30)):
    img = Image.new(mode, size, color if mode == "RGB" else None)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _llm(content="vision says hi"):
    client = AsyncMock()
    client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": content}}]})
    return client


def _payload_image_urls(llm):
    payload = llm.chat_completion.await_args[0][0]
    return [c["image_url"]["url"] for c in payload["messages"][1]["content"]
            if c["type"] == "image_url"]


# ------------------------------------------------------- _normalize_for_node

def test_node_safe_formats_pass_through_untouched():
    png = _make_image_bytes("PNG")
    jpeg = _make_image_bytes("JPEG")
    assert _normalize_for_node("image/png", png) == ("image/png", png)
    assert _normalize_for_node("image/jpeg", jpeg) == ("image/jpeg", jpeg)


def test_webp_is_transcoded_to_png():
    webp = _make_image_bytes("WEBP")
    mime, data = _normalize_for_node("image/webp", webp)
    assert mime == "image/png"
    assert data.startswith(PNG_MAGIC)
    with Image.open(io.BytesIO(data)) as img:
        assert img.size == (8, 8)


def test_palette_tiff_is_transcoded():
    img = Image.new("P", (4, 4))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    mime, data = _normalize_for_node("image/tiff", buf.getvalue())
    assert mime == "image/png"
    assert data.startswith(PNG_MAGIC)


def test_cmyk_tiff_is_flattened_to_rgb_png():
    # PNG cannot hold CMYK — without the mode flatten, img.save raises and
    # the transcode silently falls back to bytes the node can't decode.
    img = Image.new("CMYK", (4, 4))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    mime, data = _normalize_for_node("image/tiff", buf.getvalue())
    assert mime == "image/png"
    with Image.open(io.BytesIO(data)) as out:
        assert out.mode == "RGB"


def test_oversized_transcode_is_downscaled():
    """The output is bounded too: a webp decoding past the pixel budget
    would otherwise re-encode to a huge PNG payload for the node."""
    from ghost_agent.tools.vision import _MAX_PDF_PAGE_PIXELS
    webp = _make_image_bytes("WEBP", size=(3000, 3000))
    mime, data = _normalize_for_node("image/webp", webp)
    assert mime == "image/png"
    with Image.open(io.BytesIO(data)) as img:
        assert img.size[0] * img.size[1] <= _MAX_PDF_PAGE_PIXELS


def test_undecodable_bytes_fall_back_to_original():
    """A webp-labelled body Pillow can't decode ships unchanged (pre-guard
    behavior) instead of erroring inside the tool."""
    fake_webp = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 32
    assert _normalize_for_node("image/webp", fake_webp) == ("image/webp", fake_webp)


# ------------------------------------------------------------- end to end

@pytest.mark.asyncio
async def test_local_webp_file_reaches_node_as_png(tmp_path):
    (tmp_path / "shot.webp").write_bytes(_make_image_bytes("WEBP"))
    llm = _llm("a red square")
    out = await tool_vision_analysis(
        action="describe_picture", target="shot.webp",
        llm_client=llm, sandbox_dir=tmp_path)
    assert "a red square" in out
    urls = _payload_image_urls(llm)
    assert urls and urls[0].startswith("data:image/png;base64,")
    assert base64.b64decode(urls[0].split(",", 1)[1]).startswith(PNG_MAGIC)


@pytest.mark.asyncio
async def test_url_webp_reaches_node_as_png(monkeypatch, tmp_path):
    from tests.test_vision_hardening import _FakeResp, _fake_client_factory
    webp = _make_image_bytes("WEBP")
    monkeypatch.setattr(vision_mod.httpx, "AsyncClient", _fake_client_factory([
        _FakeResp(200, headers={"content-type": "image/webp"}, body=webp),
    ]))
    llm = _llm("described")
    out = await tool_vision_analysis(
        action="describe_picture", target="http://example.com/pic.webp",
        llm_client=llm, sandbox_dir=tmp_path)
    assert "described" in out
    urls = _payload_image_urls(llm)
    assert urls and urls[0].startswith("data:image/png;base64,")


# ------------------------------------------------------- node failure logs

def test_node_error_detail_surfaces_http_status_and_body():
    import httpx
    from ghost_agent.core.llm import _node_error_detail

    resp = httpx.Response(400, content=b"  unsupported\n image  format ")
    err = httpx.HTTPStatusError(
        "boom", request=httpx.Request("POST", "http://eva/v1/chat/completions"),
        response=resp)
    assert _node_error_detail(err) == "HTTP 400 — unsupported image format"

    long_body = httpx.Response(500, content=b"x" * 1000)
    err_long = httpx.HTTPStatusError(
        "boom", request=httpx.Request("POST", "http://eva"), response=long_body)
    assert len(_node_error_detail(err_long)) <= len("HTTP 500 — ") + 160

    assert _node_error_detail(TimeoutError()) == "TimeoutError"
