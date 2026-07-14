"""vision_analysis hardening — 2026-07-14 regressions.

Covers: per-hop SSRF validation on URL redirects (the hole already closed in
tool_download_file on 2026-07-07 that vision never got), PDF rasterisation
gated on the file actually being a PDF (extract_text_pdf on an image used to
throw away the working image data), the explicit 10-page PDF cap note,
magic-byte typing of local files (a .txt was previously guessed image/jpeg
and shipped to the vision model), and prompt-alias healing.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import AsyncMock

from ghost_agent.tools import vision as vision_mod
from ghost_agent.tools.vision import tool_vision_analysis, _sniff_image_mime

# Tiny valid PNG header + padding — enough for the magic-byte sniffer.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 64


def _llm(content="vision says hi"):
    client = AsyncMock()
    client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": content}}]})
    return client


# ------------------------------------------------------------- URL redirects

class _FakeResp:
    def __init__(self, status_code, headers=None, body=b""):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aiter_bytes(self):
        yield self._body


class _FakeStreamCM:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *a):
        return False


def _fake_client_factory(responses):
    """Build an httpx.AsyncClient stand-in serving `responses` in order."""
    it = iter(responses)

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url):
            return _FakeStreamCM(next(it))

    return _FakeClient


@pytest.mark.asyncio
async def test_redirect_to_internal_host_is_blocked(monkeypatch, tmp_path):
    """A public URL 302-redirecting to loopback must be refused — with
    follow_redirects=True this bypassed the original-URL SSRF check."""
    monkeypatch.setattr(vision_mod.httpx, "AsyncClient", _fake_client_factory([
        _FakeResp(302, headers={"location": "http://127.0.0.1:8000/secret.png"}),
    ]))
    out = await tool_vision_analysis(
        action="describe_picture", target="http://example.com/pic.png",
        llm_client=_llm(), sandbox_dir=tmp_path)
    assert out.startswith("Error")
    assert "SSRF" in out or "blocked" in out.lower()


@pytest.mark.asyncio
async def test_redirect_loop_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(vision_mod.httpx, "AsyncClient", _fake_client_factory([
        _FakeResp(302, headers={"location": "http://example.com/hop"})
        for _ in range(10)
    ]))
    out = await tool_vision_analysis(
        action="describe_picture", target="http://example.com/pic.png",
        llm_client=_llm(), sandbox_dir=tmp_path)
    assert "too many redirects" in out.lower()


@pytest.mark.asyncio
async def test_safe_redirect_is_followed(monkeypatch, tmp_path):
    llm = _llm("a nice picture")
    monkeypatch.setattr(vision_mod.httpx, "AsyncClient", _fake_client_factory([
        _FakeResp(302, headers={"location": "http://example.com/real.png"}),
        _FakeResp(200, headers={"content-type": "image/png"}, body=PNG_BYTES),
    ]))
    out = await tool_vision_analysis(
        action="describe_picture", target="http://example.com/pic.png",
        llm_client=llm, sandbox_dir=tmp_path)
    assert "a nice picture" in out


@pytest.mark.asyncio
async def test_garbage_content_length_does_not_crash(monkeypatch, tmp_path):
    llm = _llm("ok")
    monkeypatch.setattr(vision_mod.httpx, "AsyncClient", _fake_client_factory([
        _FakeResp(200, headers={"content-type": "image/png",
                                "content-length": "garbage"}, body=PNG_BYTES),
    ]))
    out = await tool_vision_analysis(
        action="describe_picture", target="http://example.com/pic.png",
        llm_client=llm, sandbox_dir=tmp_path)
    assert "ok" in out and not out.startswith("Error")


# ---------------------------------------------------------- local-file typing

@pytest.mark.asyncio
async def test_extract_text_pdf_on_image_keeps_the_image(tmp_path):
    """The action used to force the fitz branch for ANY target, replacing the
    already-extracted image data with a doomed PDF parse."""
    (tmp_path / "shot.png").write_bytes(PNG_BYTES)
    llm = _llm("OCR text here")
    out = await tool_vision_analysis(
        action="extract_text_pdf", target="shot.png",
        llm_client=llm, sandbox_dir=tmp_path)
    assert "OCR text here" in out
    assert "Error processing PDF" not in out
    payload = llm.chat_completion.await_args[0][0]
    imgs = [c for c in payload["messages"][1]["content"] if c["type"] == "image_url"]
    assert imgs and imgs[0]["image_url"]["url"].startswith("data:image/png")


@pytest.mark.asyncio
async def test_extensionless_image_is_sniffed(tmp_path):
    (tmp_path / "noext").write_bytes(JPEG_BYTES)
    llm = _llm("described")
    out = await tool_vision_analysis(
        action="describe_picture", target="noext",
        llm_client=llm, sandbox_dir=tmp_path)
    assert "described" in out
    payload = llm.chat_completion.await_args[0][0]
    imgs = [c for c in payload["messages"][1]["content"] if c["type"] == "image_url"]
    assert imgs[0]["image_url"]["url"].startswith("data:image/jpeg")


@pytest.mark.asyncio
async def test_text_file_is_refused_not_hallucinated_over(tmp_path):
    (tmp_path / "notes.txt").write_text("just some text")
    llm = _llm()
    out = await tool_vision_analysis(
        action="describe_picture", target="notes.txt",
        llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("Error")
    assert "file_system" in out
    llm.chat_completion.assert_not_awaited()


def test_sniffer_signatures():
    assert _sniff_image_mime(PNG_BYTES) == "image/png"
    assert _sniff_image_mime(JPEG_BYTES) == "image/jpeg"
    assert _sniff_image_mime(b"GIF89a...") == "image/gif"
    assert _sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert _sniff_image_mime(b"plain text here") is None
    assert _sniff_image_mime(b"") is None


# ----------------------------------------------------------------- prompts

@pytest.mark.asyncio
async def test_prompt_aliases_are_healed(tmp_path):
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    llm = _llm("answer")
    await tool_vision_analysis(
        action="describe_picture", target="img.png",
        llm_client=llm, sandbox_dir=tmp_path,
        question="How many cats are in this picture?")
    payload = llm.chat_completion.await_args[0][0]
    texts = [c["text"] for c in payload["messages"][1]["content"] if c["type"] == "text"]
    assert texts[0] == "How many cats are in this picture?"


# ----------------------------------------------------------------- PDF cap

@pytest.mark.asyncio
async def test_pdf_page_cap_is_announced(tmp_path):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for _ in range(12):
        doc.new_page(width=72, height=72)
    pdf_path = tmp_path / "big.pdf"
    doc.save(str(pdf_path))
    doc.close()

    llm = _llm("pages analysed")
    out = await tool_vision_analysis(
        action="extract_text_pdf", target="big.pdf",
        llm_client=llm, sandbox_dir=tmp_path)
    assert "pages analysed" in out
    assert "12 pages" in out and "first 10" in out  # explicit, not silent
