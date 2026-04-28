"""URL-on-file_system error message routes to `browser`, not just `knowledge_base`.

Bug from production T7 trace: user said "Open https://example.com using
the browser tool and tell me the H1." The model first tried
`file_system.read_file('https://example.com')`. The old error message
read: "Use knowledge_base(action='ingest_document') instead." That
single hint MIGRATED the model away from `browser` (which the user had
explicitly named) and into a `knowledge_base(ingest_document, url=...)`
call — `browser` was never invoked, and the agent returned the literal
error string as its final answer.

The error-surface text PRIMES the model's next tool choice. Lead with
`browser` for live page reads; mention `knowledge_base` only as the
ingest-into-memory path.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from ghost_agent.tools.file_system import tool_read_file, tool_read_document_chunked


@pytest.mark.asyncio
async def test_read_file_on_url_suggests_browser_first(tmp_path: Path):
    res = await tool_read_file("https://example.com", sandbox_dir=tmp_path)
    assert "browser" in res, (
        "URL error must mention the `browser` tool — that's the right "
        "answer for live page reads. The previous one-line message "
        "only mentioned knowledge_base, which silently misrouted every "
        "browser request."
    )
    # browser comes BEFORE knowledge_base in the message — order matters
    # for prompt salience.
    assert res.index("browser") < res.index("knowledge_base"), (
        "browser must appear BEFORE knowledge_base in the error so the "
        "model's first read of the message biases toward browser, the "
        "common-case answer."
    )


@pytest.mark.asyncio
async def test_read_file_on_url_uses_new_filename_param(tmp_path: Path):
    """The error string is a hint the model copies verbatim. It must NOT
    use the legacy `content=` param — that field was removed from the
    schema in the Layer 1 rename and would advertise a foot-gun."""
    res = await tool_read_file("https://example.com/page", sandbox_dir=tmp_path)
    if "knowledge_base" in res:
        assert "content=" not in res, (
            "Error message must use the new `filename=` parameter, not "
            "the deprecated `content=` alias."
        )
        assert "filename=" in res


@pytest.mark.asyncio
async def test_read_chunked_on_url_suggests_browser_first(tmp_path: Path):
    """The same URL guard exists on the chunked-read path. Must mirror."""
    res = await tool_read_document_chunked("https://example.com", sandbox_dir=tmp_path)
    assert "browser" in res
    assert res.index("browser") < res.index("knowledge_base")
    assert "content=" not in res


@pytest.mark.asyncio
async def test_pdf_error_uses_new_filename_param(tmp_path: Path):
    """The PDF guard message also pre-fills the next tool call. Must
    use `filename=`, not the legacy `content=`."""
    # Create a fake PDF (just touch — the guard fires on extension)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    res = await tool_read_file("doc.pdf", sandbox_dir=tmp_path)
    assert "knowledge_base" in res
    assert "content=" not in res, (
        "PDF redirect must use `filename=`, not the deprecated "
        "`content=` alias removed in the Layer 1 schema rename."
    )
    assert "filename=" in res


@pytest.mark.asyncio
async def test_url_error_includes_the_url_for_copy_paste(tmp_path: Path):
    """The error should include the actual URL string in both
    suggestions so the model can copy-paste rather than re-derive it
    (re-derivation is where typos creep in)."""
    target = "https://example.com/some/page"
    res = await tool_read_file(target, sandbox_dir=tmp_path)
    # URL appears at least twice (once in browser suggestion, once in kb).
    assert res.count(target) >= 2, (
        f"Expected the URL to appear in both the browser and "
        f"knowledge_base suggestions; got: {res}"
    )
