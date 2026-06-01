"""report_pdf source_files: compile a detailed report from sandbox files
directly, so the full content reaches the PDF without the model having to
re-transcribe it (which collapsed into a thin 1-page summary on the
meta-cognitive report)."""

import re

import pytest

from ghost_agent.tools.report_pdf import _sections_from_files, tool_generate_pdf


def test_sections_from_files_splits_each(tmp_path):
    (tmp_path / "a.md").write_text("# A\nintro\n## A1\nbody1\n## A2\nbody2")
    (tmp_path / "b.md").write_text("# B\n## B1\nx\n## B2\ny")
    secs, missing = _sections_from_files(tmp_path, ["a.md", "b.md"])
    headings = [s["heading"] for s in secs]
    assert "A1" in headings and "A2" in headings and "B1" in headings and "B2" in headings
    assert missing == []


def test_sections_from_files_missing_reported(tmp_path):
    (tmp_path / "a.md").write_text("# A\n## A1\nx")
    secs, missing = _sections_from_files(tmp_path, ["a.md", "nope.md"])
    assert missing == ["nope.md"]
    assert secs  # a.md still produced sections


def test_no_header_file_titled_from_filename(tmp_path):
    (tmp_path / "notes_file.md").write_text("just flat content, no headers")
    secs, _ = _sections_from_files(tmp_path, ["notes_file.md"])
    assert len(secs) == 1
    assert secs[0]["heading"] == "Notes File"
    assert "flat content" in secs[0]["body"]


def test_path_coercion_comma_and_json(tmp_path):
    (tmp_path / "a.md").write_text("# A\n## A1\nx")
    (tmp_path / "b.md").write_text("# B\n## B1\ny")
    s_comma, _ = _sections_from_files(tmp_path, "a.md, b.md")
    s_json, _ = _sections_from_files(tmp_path, '["a.md", "b.md"]')
    assert len(s_comma) == len(s_json) >= 4


async def test_pdf_from_source_files_is_detailed(tmp_path):
    (tmp_path / "t1.md").write_text("# Task 1\n" + ("detail " * 200) + "\n## Sub\n" + ("more " * 200))
    (tmp_path / "t2.md").write_text("# Task 2\n" + ("findings " * 200))
    out = await tool_generate_pdf(
        title="Detailed Report", source_files=["t1.md", "t2.md"], sandbox_dir=tmp_path,
    )
    assert out.startswith("SUCCESS")
    m = re.search(r"/api/download/(report_[A-Za-z0-9]+\.pdf)", out)
    assert m
    pdf = tmp_path / m.group(1)
    assert pdf.exists()
    import fitz
    doc = fitz.open(pdf)
    txt = "".join(p.get_text() for p in doc)
    # The full file content made it into the PDF — NOT a thin summary.
    assert len(txt) > 1500
    assert "Task 1" in txt and "Task 2" in txt


async def test_source_files_only_satisfies_required(tmp_path):
    (tmp_path / "x.md").write_text("# X\n## X1\nbody text here")
    out = await tool_generate_pdf(title="R", source_files=["x.md"], sandbox_dir=tmp_path)
    assert out.startswith("SUCCESS")  # source_files alone is enough; no 'sections' needed


async def test_missing_source_files_noted_in_success(tmp_path):
    (tmp_path / "x.md").write_text("# X\n## X1\nbody")
    out = await tool_generate_pdf(
        title="R", source_files=["x.md", "gone.md"], sandbox_dir=tmp_path,
    )
    assert out.startswith("SUCCESS")
    assert "gone.md" in out and "skipped" in out.lower()
