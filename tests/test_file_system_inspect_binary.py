"""`file_system(operation="inspect")` on a binary file (req 2422eb25, 2026-09-06).

`tool_inspect_file` opened the target in text mode with `errors="replace"`
and returned its first ten newline-delimited "lines" — for a JPEG that was
~1.3 KB of JFIF/Exif/ICC bytes as replacement characters, injected into the
context of every image request that peeks the file before vision (a hydrated
lesson tells the model to do exactly that). `read` and `read_chunked` had a
binary sniff; `inspect` did not.

The sniffed reply is deliberately NOT an "Error:" line: the strike counter
treats that prefix as an execution failure, and a peek at an image that
EXISTS is a success — the model wanted to know the file is there.

Worlds where these fail: remove the sniff (JFIF bytes come back); make the
sniffed reply start with "Error" (a strike per image request); drop the kind
or size (the summary stops answering "what is this file").
"""

import pytest

from ghost_agent.tools.file_system import tool_inspect_file

# A JPEG head (SOI, APP0 "JFIF", APP1) followed by bytes that a text decode
# turns into replacement characters, with a newline inside so the old
# line-based peek would have returned two "lines" of garbage.
JPEG = (b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00H\x00H\x00\x00\xff\xe1"
        + bytes(range(256)) * 4 + b"\n" + bytes(range(256)) * 2)


@pytest.mark.asyncio
async def test_inspect_does_not_dump_a_jpegs_bytes(temp_dirs):
    sandbox = temp_dirs["sandbox"]
    (sandbox / "photo-20260906-171445.jpg").write_bytes(JPEG)
    out = await tool_inspect_file("photo-20260906-171445.jpg", sandbox)
    assert "JFIF" not in out and "�" not in out, out[:200]
    assert "JPEG image" in out
    assert f"{len(JPEG):,} bytes" in out
    assert "vision_analysis" in out
    assert not out.lstrip().startswith("Error"), (
        "a peek at an existing image must not read as a tool failure")


@pytest.mark.asyncio
async def test_inspect_names_other_kinds_without_the_vision_hint(temp_dirs):
    sandbox = temp_dirs["sandbox"]
    (sandbox / "doc.pdf").write_bytes(b"%PDF-1.7\n\x00\x00binary" * 20)
    (sandbox / "blob.bin").write_bytes(b"\x00\x01\x02\x03" * 100)
    pdf = await tool_inspect_file("doc.pdf", sandbox)
    assert "PDF document" in pdf and "vision_analysis" not in pdf
    blob = await tool_inspect_file("blob.bin", sandbox)
    assert "binary file" in blob and "\x00" not in blob
    assert not blob.lstrip().startswith("Error")


@pytest.mark.asyncio
async def test_inspect_still_peeks_text_files(temp_dirs):
    """Control: the sniff must not touch the text path."""
    sandbox = temp_dirs["sandbox"]
    (sandbox / "notes.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")
    out = await tool_inspect_file("notes.txt", sandbox, lines=2)
    assert out == "line1\nline2"


@pytest.mark.asyncio
async def test_inspect_on_a_missing_file_is_unchanged(temp_dirs):
    """Control: the reconciling missing-file message still comes first."""
    out = await tool_inspect_file("photo-20260906-171609.jpg", temp_dirs["sandbox"])
    assert "does not exist" in out or "not found" in out
