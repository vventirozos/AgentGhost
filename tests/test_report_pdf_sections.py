"""A whole-report-in-one-markdown-string should auto-split into sections
on its top-level headers, so a "dumped" report renders as a structured
multi-section PDF instead of one flat blob (the report-regeneration
trace produced sections=1 from 5 detailed source files)."""

import json

from ghost_agent.tools.report_pdf import _normalise_sections, _split_markdown_into_sections


def test_single_string_with_headers_auto_splits():
    md = "# Report\nIntro line\n\n## Task 1\nFindings A\n\n## Task 2\nFindings B"
    out = _normalise_sections(md, None, None)
    assert [s["heading"] for s in out] == ["Report", "Task 1", "Task 2"]
    assert "Findings A" in out[1]["body"]


def test_single_string_no_headers_stays_one_section():
    out = _normalise_sections("just a flat blob of text", None, None)
    assert len(out) == 1
    assert out[0]["body"] == "just a flat blob of text"


def test_single_header_not_split():
    out = _normalise_sections("# Only One\nbody text here", None, None)
    assert len(out) == 1  # < 2 top-level headers → not split


def test_subheaders_stay_in_parent_body():
    md = "## Top\nintro\n### Sub-detail\nmore\n## Two\nx"
    out = _normalise_sections(md, None, None)
    assert [s["heading"] for s in out] == ["Top", "Two"]
    assert "### Sub-detail" in out[0]["body"]


def test_explicit_list_unchanged():
    secs = [{"heading": "H1", "body": "b1"}, {"heading": "H2", "body": "b2"}]
    out = _normalise_sections(secs, None, None)
    assert [s["heading"] for s in out] == ["H1", "H2"]


def test_json_string_list_still_decoded():
    s = json.dumps([{"heading": "A", "body": "x"}, {"heading": "B", "body": "y"}])
    out = _normalise_sections(s, None, None)
    assert [x["heading"] for x in out] == ["A", "B"]


def test_body_kwarg_fallback_also_splits():
    out = _normalise_sections(None, "## S1\na\n## S2\nb", None)
    assert [s["heading"] for s in out] == ["S1", "S2"]


def test_split_helper_returns_empty_without_two_headers():
    assert _split_markdown_into_sections("no headers at all") == []
    assert _split_markdown_into_sections("# one\nbody") == []
