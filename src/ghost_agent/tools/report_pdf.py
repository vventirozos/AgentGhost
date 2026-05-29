"""Native PDF report generator.

Mirrors the shape of ``tool_generate_image``: produce a binary asset
inside ``sandbox_dir`` and return a single string telling the LLM the
exact markdown to emit so the file is offered to the user via the
existing ``/api/download/<filename>`` route.

Engine: PyMuPDF (``fitz``) + the pure-Python ``markdown`` library.
PyMuPDF is already a hard dependency (it backs ``vision_analysis`` /
``read_chunked``), so no new system libraries are required — unlike
WeasyPrint which would pull in Pango/Cairo. ``markdown`` is a small
pure-Python package added to ``requirements.txt``.

Input shape (defensive — see ``_normalise_sections``):

  tool_generate_pdf(
      title="Top Tech Trends 2027",
      subtitle="Market Analysis",          # optional
      author="Ghost Agent",                # optional
      sections=[
          {"heading": "Introduction",
           "body": "Markdown _body_ with **bold**, lists, etc."},
          {"heading": "Trend #1: ...",
           "body": "..."},
          ...
      ],
  )

``sections`` may also be a single markdown string — it is auto-wrapped
into a one-section report so the LLM cannot trivially mis-call the
tool.
"""

import asyncio
import html
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from ..utils.logging import Icons, pretty_log


_PAGE_W, _PAGE_H = 595, 842            # A4 portrait, points
_MARGIN = 50
_CONTENT_RECT = (_MARGIN, _MARGIN + 30, _PAGE_W - _MARGIN, _PAGE_H - _MARGIN)
_MAX_BODY_CHARS = 200_000              # ~80 pages — hard cap on a single report

_CSS = """
* { font-family: sans-serif; color: #1a1a1a; }
h1 { font-size: 22pt; color: #111; margin-bottom: 6pt; }
h2 { font-size: 16pt; color: #222; margin-top: 14pt; margin-bottom: 4pt;
     border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 13pt; color: #333; margin-top: 10pt; margin-bottom: 3pt; }
p  { font-size: 10.5pt; line-height: 1.45; margin: 4pt 0; }
li { font-size: 10.5pt; line-height: 1.4; }
code { font-family: monospace; background: #f3f3f3; padding: 0 3px; font-size: 9.5pt; }
pre  { font-family: monospace; background: #f3f3f3; padding: 6pt; font-size: 9pt;
       white-space: pre-wrap; }
blockquote { border-left: 3px solid #888; padding-left: 8pt; color: #444; }
table { border-collapse: collapse; margin: 6pt 0; }
th, td { border: 1px solid #ccc; padding: 3pt 6pt; font-size: 10pt; text-align: left; }
.subtitle { font-size: 12pt; color: #555; margin-top: -2pt; }
.meta { font-size: 9pt; color: #777; margin-bottom: 10pt; }
hr { border: 0; border-top: 1px solid #ddd; margin: 8pt 0; }
"""


def _normalise_sections(sections: Any, body: Any, content: Any) -> list[dict]:
    """Coerce whatever the LLM passed into ``[{heading, body}, ...]``.

    Why so forgiving: in self-play logs the agent regularly hallucinates
    alternate parameter names (``body=``, ``content=``, ``text=``) and
    sometimes passes a single string when it should have passed a list.
    Mirrors the healing block at the top of ``tool_generate_image``.
    """
    if isinstance(sections, list) and sections:
        out = []
        for i, s in enumerate(sections):
            if isinstance(s, dict):
                heading = str(s.get("heading") or s.get("title") or s.get("name") or "").strip()
                bdy = s.get("body") or s.get("content") or s.get("text") or ""
                if not isinstance(bdy, str):
                    bdy = str(bdy)
                out.append({"heading": heading, "body": bdy})
            elif isinstance(s, str):
                out.append({"heading": f"Section {i+1}", "body": s})
        if out:
            return out

    # Single-string fallback: maybe the LLM passed `body=` or `content=`
    # or dumped the whole report into `sections="..."`.
    fallback = None
    for cand in (sections, body, content):
        if isinstance(cand, str) and cand.strip():
            fallback = cand
            break
    if fallback:
        return [{"heading": "", "body": fallback}]
    return []


def _md_to_html(md_text: str) -> str:
    """Render markdown → HTML. Degrades to ``<pre>``-wrapped escaped
    text if the ``markdown`` package isn't importable (so the tool
    still works on a minimal install — output just won't be as pretty)."""
    try:
        import markdown as _md
        return _md.markdown(
            md_text,
            extensions=["extra", "sane_lists", "tables", "nl2br"],
            output_format="html5",
        )
    except ImportError:
        return f"<pre>{html.escape(md_text)}</pre>"


def _build_html(title: str, subtitle: str, author: str, sections: list[dict]) -> str:
    """Assemble the full HTML document fed to fitz."""
    parts = [f"<style>{_CSS}</style>"]
    safe_title = html.escape(title or "Report")
    parts.append(f"<h1>{safe_title}</h1>")
    if subtitle:
        parts.append(f"<div class='subtitle'>{html.escape(subtitle)}</div>")
    if author:
        parts.append(f"<div class='meta'>By {html.escape(author)}</div>")
    parts.append("<hr/>")

    for sec in sections:
        heading = sec.get("heading", "").strip()
        body_md = sec.get("body", "")
        if heading:
            parts.append(f"<h2>{html.escape(heading)}</h2>")
        parts.append(_md_to_html(body_md))

    return "".join(parts)


def _render_to_pdf(full_html: str, out_path: Path) -> int:
    """Render ``full_html`` into a multi-page A4 PDF at ``out_path``.

    Uses ``fitz.Story`` + ``fitz.DocumentWriter`` to flow arbitrary
    HTML across as many pages as needed. Returns the page count.
    Raises on a hard render failure — the caller wraps the exception
    into a SYSTEM ERROR string for the LLM."""
    import fitz  # PyMuPDF

    mediabox = fitz.Rect(0, 0, _PAGE_W, _PAGE_H)
    where = fitz.Rect(*_CONTENT_RECT)
    story = fitz.Story(html=full_html)

    writer = fitz.DocumentWriter(str(out_path))
    pages = 0
    more = True
    try:
        while more:
            dev = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()
            pages += 1
            if pages > 200:            # safety cap
                break
    finally:
        writer.close()
    return pages


_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _sanitize_filename(raw: Optional[str]) -> Optional[str]:
    """Return a safe, sandbox-relative filename or None if invalid.

    Rules:
      * Strip whitespace; reject empty.
      * Strip any leading ``./`` and `path.basename` to defeat traversal
        like ``../etc/passwd`` or ``/tmp/x.pdf``.
      * Must match `_SAFE_FILENAME_RE`: starts alphanumeric, body
        alphanumeric / dot / underscore / dash, length ≤ 128.
      * If no `.pdf` suffix, append one (the tool only writes PDFs).
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Defeat traversal regardless of separator.
    s = s.replace("\\", "/").split("/")[-1]
    if not s.lower().endswith(".pdf"):
        s = f"{s}.pdf"
    if not _SAFE_FILENAME_RE.match(s):
        return None
    return s


async def tool_generate_pdf(
    title: str = "",
    sections: Any = None,
    subtitle: str = "",
    author: str = "",
    filename: str = "",
    sandbox_dir: Path = None,
    **kwargs,
) -> str:
    """Native PDF report tool. See module docstring for input shape.

    Returns either:
      - ``"SUCCESS: ... [Open PDF](/api/download/report_xxx.pdf)"``
      - ``"SYSTEM ERROR: ..."``  (validation or render failure)
    """
    # --- PARAMETER HALLUCINATION HEALING (mirrors tool_generate_image) ---
    title = (title or kwargs.get("name") or kwargs.get("report_title") or "").strip()
    subtitle = (subtitle or kwargs.get("subheading") or "").strip()
    author = (author or kwargs.get("by") or "").strip()

    secs = _normalise_sections(
        sections,
        kwargs.get("body"),
        kwargs.get("content") or kwargs.get("markdown") or kwargs.get("text"),
    )

    if not title:
        return (
            "SYSTEM ERROR: 'title' is REQUIRED for report_pdf. Pass the "
            "report's display title as a string (e.g. title='Top 5 Tech "
            "Trends for 2027')."
        )
    if not secs:
        return (
            "SYSTEM ERROR: 'sections' is REQUIRED for report_pdf. Pass a "
            "list of {heading, body} dicts (body is markdown), or a single "
            "markdown string if you only have one section."
        )

    total_chars = sum(len(s.get("body", "")) for s in secs)
    if total_chars > _MAX_BODY_CHARS:
        return (
            f"SYSTEM ERROR: report bodies total {total_chars:,} chars, which "
            f"exceeds the {_MAX_BODY_CHARS:,}-char per-report cap. Split the "
            "report into multiple calls or trim the bodies."
        )

    if sandbox_dir is None:
        return "SYSTEM ERROR: report_pdf requires a sandbox_dir (none was wired by the registry)."

    sandbox_dir = Path(sandbox_dir)
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    # Caller may supply a `filename` (or hallucinate it as `name`/
    # `output`/`path`). Sanitize it through `_sanitize_filename` to
    # block traversal and bad shapes; if nothing usable is supplied,
    # fall back to a generated `report_<8hex>.pdf`.
    requested_name = (
        filename
        or kwargs.get("output")
        or kwargs.get("path")
        or kwargs.get("file")
    )
    safe_name = _sanitize_filename(requested_name)
    if requested_name and not safe_name:
        return (
            "SYSTEM ERROR: requested filename "
            f"{requested_name!r} is not safe — must start with "
            "alphanumeric and contain only letters/digits/./_/-"
        )
    final_name = safe_name or f"report_{uuid.uuid4().hex[:8]}.pdf"
    out_path = sandbox_dir / final_name

    pretty_log("PDF Report", f"Title: {title[:60]} | sections={len(secs)}", icon=Icons.REPORT_PDF)

    full_html = _build_html(title, subtitle, author, secs)
    try:
        pages = await asyncio.to_thread(_render_to_pdf, full_html, out_path)
    except ImportError as e:
        return (
            f"SYSTEM ERROR: PDF engine unavailable ({e}). PyMuPDF (fitz) is "
            "a hard dependency — reinstall via `pip install -r requirements.txt`."
        )
    except Exception as e:
        # Wipe a half-written file so the link below never points at
        # a corrupt artefact.
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        return f"SYSTEM ERROR: PDF render failed: {type(e).__name__}: {e}"

    size_kb = out_path.stat().st_size / 1024
    return (
        f"SUCCESS: PDF report generated ({pages} page(s), {size_kb:.1f} KB). "
        "DO NOT CALL THIS TOOL AGAIN with the same title. Respond DIRECTLY "
        "to the user by including this exact markdown so the file is "
        f"offered for download:\n\n"
        f"[📄 {title} (PDF)](/api/download/{final_name})"
    )
