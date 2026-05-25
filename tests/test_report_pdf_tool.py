"""Unit tests for tools/report_pdf.py + its registry wiring."""
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.report_pdf import (
    tool_generate_pdf,
    _normalise_sections,
    _build_html,
)
from ghost_agent.tools.registry import (
    TOOL_DEFINITIONS,
    get_available_tools,
    get_active_tool_definitions,
)


# --- happy path -------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_pdf_success_writes_valid_pdf(tmp_path):
    res = await tool_generate_pdf(
        title="Top 5 Tech Trends 2027",
        subtitle="Market Analysis",
        author="Ghost Agent",
        sections=[
            {"heading": "Intro", "body": "**Bold** and _italic_ markdown.\n\n- one\n- two"},
            {"heading": "Trend 1", "body": "Lorem ipsum.\n\nSecond paragraph."},
        ],
        sandbox_dir=tmp_path,
    )

    assert res.startswith("SUCCESS")
    assert "(/api/download/report_" in res
    assert ".pdf)" in res

    pdfs = list(tmp_path.glob("report_*.pdf"))
    assert len(pdfs) == 1
    # PDF magic header
    assert pdfs[0].read_bytes()[:5] == b"%PDF-"


@pytest.mark.asyncio
async def test_generate_pdf_long_body_spans_multiple_pages(tmp_path):
    """A large body should overflow into a second physical page rather
    than being truncated. Regression guard for the Story/DocumentWriter
    pagination loop."""
    big_body = ("Lorem ipsum dolor sit amet. " * 800).strip()
    res = await tool_generate_pdf(
        title="Long Report",
        sections=[{"heading": "Body", "body": big_body}],
        sandbox_dir=tmp_path,
    )
    assert res.startswith("SUCCESS")

    import fitz
    pdfs = list(tmp_path.glob("report_*.pdf"))
    doc = fitz.open(pdfs[0])
    try:
        assert doc.page_count >= 2
    finally:
        doc.close()


# --- validation -------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_pdf_missing_title(tmp_path):
    res = await tool_generate_pdf(
        title="",
        sections=[{"heading": "x", "body": "y"}],
        sandbox_dir=tmp_path,
    )
    assert res.startswith("SYSTEM ERROR")
    assert "title" in res.lower()
    assert not list(tmp_path.glob("*.pdf"))


@pytest.mark.asyncio
async def test_generate_pdf_missing_sections(tmp_path):
    res = await tool_generate_pdf(
        title="Hello",
        sections=None,
        sandbox_dir=tmp_path,
    )
    assert res.startswith("SYSTEM ERROR")
    assert "sections" in res.lower()


@pytest.mark.asyncio
async def test_generate_pdf_missing_sandbox_dir():
    res = await tool_generate_pdf(
        title="Hello",
        sections=[{"heading": "x", "body": "y"}],
        sandbox_dir=None,
    )
    assert res.startswith("SYSTEM ERROR")
    assert "sandbox" in res.lower()


@pytest.mark.asyncio
async def test_generate_pdf_oversize_body_rejected(tmp_path):
    from ghost_agent.tools import report_pdf as rp
    huge = "a" * (rp._MAX_BODY_CHARS + 1)
    res = await tool_generate_pdf(
        title="Big",
        sections=[{"heading": "x", "body": huge}],
        sandbox_dir=tmp_path,
    )
    assert res.startswith("SYSTEM ERROR")
    assert "cap" in res.lower() or "exceed" in res.lower()


# --- parameter healing ------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_pdf_string_sections_promoted(tmp_path):
    """LLMs sometimes pass `sections='one big markdown blob'` instead
    of a list. The healer should wrap it in a single section so the
    call still succeeds."""
    res = await tool_generate_pdf(
        title="Plain",
        sections="# Hello\n\nSingle-string body that should be auto-wrapped.",
        sandbox_dir=tmp_path,
    )
    assert res.startswith("SUCCESS")
    assert list(tmp_path.glob("report_*.pdf"))


@pytest.mark.asyncio
async def test_generate_pdf_uses_body_alias(tmp_path):
    """If the LLM passes `body=...` instead of `sections=...`, accept it."""
    res = await tool_generate_pdf(
        title="Alias",
        body="A body passed under the wrong kwarg name.",
        sandbox_dir=tmp_path,
    )
    assert res.startswith("SUCCESS")


def test_normalise_sections_handles_dicts_with_aliases():
    out = _normalise_sections(
        [{"title": "h1", "content": "c1"}, {"name": "h2", "text": "c2"}],
        None,
        None,
    )
    assert out == [
        {"heading": "h1", "body": "c1"},
        {"heading": "h2", "body": "c2"},
    ]


def test_normalise_sections_string_in_list_becomes_section():
    out = _normalise_sections(["just a string"], None, None)
    assert out == [{"heading": "Section 1", "body": "just a string"}]


def test_normalise_sections_empty_returns_empty():
    assert _normalise_sections(None, None, None) == []
    assert _normalise_sections([], None, None) == []


# --- HTML escaping (don't let injected angle brackets break the doc) --------

def test_build_html_escapes_title_and_subtitle():
    full = _build_html(
        title="<script>alert(1)</script>",
        subtitle="<b>",
        author="</style>",
        sections=[{"heading": "h", "body": "ok"}],
    )
    assert "<script>" not in full
    assert "&lt;script&gt;" in full
    assert "&lt;b&gt;" in full
    assert "&lt;/style&gt;" in full


# --- registry wiring --------------------------------------------------------

def test_report_pdf_in_static_definitions():
    names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    assert "report_pdf" in names


def test_report_pdf_active_for_any_context(tmp_path):
    """Unlike image_generation, report_pdf has no remote-service gate —
    it should appear in the active definitions for every context."""
    ctx = MagicMock()
    ctx.llm_client.image_gen_clients = []
    ctx.sandbox_dir = tmp_path
    ctx.memory_dir = tmp_path
    ctx.memory_system = None
    ctx.args.default_db = None

    defs = get_active_tool_definitions(ctx)
    assert any(t["function"]["name"] == "report_pdf" for t in defs)


def test_report_pdf_runtime_lambda_registered(tmp_path):
    ctx = MagicMock()
    ctx.sandbox_dir = tmp_path
    ctx.memory_dir = tmp_path
    ctx.memory_system = None
    ctx.llm_client.image_gen_clients = []
    ctx.args.default_db = None

    tools = get_available_tools(ctx)
    assert "report_pdf" in tools
    assert callable(tools["report_pdf"])


@pytest.mark.asyncio
async def test_report_pdf_callable_through_registry(tmp_path):
    """Calling the registry-wired lambda should produce a real PDF in
    the context's sandbox_dir, end-to-end."""
    ctx = MagicMock()
    ctx.sandbox_dir = tmp_path
    ctx.memory_dir = tmp_path
    ctx.memory_system = None
    ctx.llm_client.image_gen_clients = []
    ctx.args.default_db = None

    tools = get_available_tools(ctx)
    res = await tools["report_pdf"](
        title="Via Registry",
        sections=[{"heading": "x", "body": "y"}],
    )
    assert res.startswith("SUCCESS")
    assert list(tmp_path.glob("report_*.pdf"))
