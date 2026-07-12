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
import json
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


def _split_markdown_into_sections(md: str) -> list[dict]:
    """Split a single markdown document into sections on its top-level
    (``#`` / ``##``) ATX headers.

    The agent frequently dumps a whole multi-part report into ONE markdown
    string (``sections="# Report\\n## Task 1 …\\n## Task 2 …"``); rendered
    as a single section that becomes a flat wall of text and reads "thin"
    even when it's long. Splitting on the top-level headers turns it into a
    properly structured multi-section PDF — one section per ``##`` — with
    no extra work from the model. Deeper headers (``###`` +) stay inside
    their parent section's body. Returns ``[]`` when there's nothing to
    split on (< 2 top-level headers), so single-section reports are left
    to the plain single-string fallback."""
    header_re = re.compile(r"^\s{0,3}(#{1,2})\s+(.+?)\s*#*\s*$")
    sections: list[dict] = []
    heading = ""
    body_lines: list[str] = []

    def _flush() -> None:
        if heading or "\n".join(body_lines).strip():
            sections.append({"heading": heading, "body": "\n".join(body_lines).strip()})

    found = False
    for ln in md.splitlines():
        m = header_re.match(ln)
        if m:
            found = True
            _flush()
            heading = m.group(2).strip()
            body_lines = []
        else:
            body_lines.append(ln)
    _flush()
    if not found or len(sections) < 2:
        return []
    return sections


_HINT_MAX_FILES = 40


def _available_files_hint(sandbox_dir: Path) -> str:
    """List the text files that DO exist, so a caller that guessed a filename
    can fix it in ONE retry instead of guessing again.

    Mirrors ``file_system._missing_file_message``, which exists for exactly
    this reason. Without it, ``report_pdf`` said only "24 source file(s) could
    not be read and were skipped: [...]" — naming the misses but never what was
    actually there. Observed live 2026-07-11: the model had invented filenames
    from task descriptions, then regenerated the PDF THREE times and listed the
    sandbox tree TWICE (~50s of a user-facing turn) to discover the real files
    lived under ``research/`` with different names. Never raises.
    """
    try:
        root = Path(sandbox_dir).resolve()
        if not root.is_dir():
            return ""
        rels = []
        for p in sorted(root.rglob("*")):
            if len(rels) >= _HINT_MAX_FILES:
                break
            if not p.is_file() or p.suffix.lower() not in (".md", ".txt", ".rst"):
                continue
            if any(part.startswith(".") for part in p.parts):
                continue
            rels.append(str(p.relative_to(root)))
        if not rels:
            return ""
        more = ("\n  …(list truncated)" if len(rels) >= _HINT_MAX_FILES else "")
        return (
            "\n\nFILES THAT DO EXIST in this workspace (use these EXACT paths — "
            "do not guess names from task descriptions):\n  "
            + "\n  ".join(rels) + more
        )
    except Exception:  # noqa: BLE001 — a hint must never break the tool
        return ""


def _sections_from_files(sandbox_dir: Path, files: Any) -> tuple[list[dict], list[str]]:
    """Build report sections by reading sandbox markdown/text files.

    The robust path for "generate a detailed report from all the task
    files": the agent passes ``source_files=[...]`` and the bulk content
    flows file → PDF directly, instead of the model having to
    re-transcribe it into the tool-call argument — which made small models
    either summarise it down to a thin blurb or stall emitting a giant
    argument (observed repeatedly on the meta-cognitive report). Each file
    is split on its top-level headers; a header-less file becomes one
    section titled from its filename. Missing files are skipped and
    returned so the caller can surface them."""
    paths: list[str] = []
    if isinstance(files, str):
        s = files.strip()
        if s[:1] == "[":
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    paths = [str(x) for x in parsed]
            except (ValueError, TypeError):
                paths = []
        if not paths:
            paths = [p.strip() for p in re.split(r"[,\n]", s) if p.strip()]
    elif isinstance(files, (list, tuple)):
        paths = [str(x).strip() for x in files if str(x).strip()]

    out: list[dict] = []
    missing: list[str] = []
    try:
        from .file_system import _get_safe_path
    except Exception:
        _get_safe_path = None
    # (see _available_files_hint — a bare "skipped: [...]" list made the model
    # GUESS filenames across three PDF regenerations)
    _MAX_SOURCE_FILE_BYTES = 25 * 1024 * 1024  # 25 MB per source file
    _sb_root = Path(sandbox_dir).resolve()
    for p in paths:
        try:
            if _get_safe_path is not None:
                fp = _get_safe_path(Path(sandbox_dir), p)
            else:
                # Fallback containment check: without _get_safe_path (a
                # near-dead path — file_system essentially always imports),
                # still refuse a path that escapes the sandbox root.
                fp = (Path(sandbox_dir) / str(p).lstrip("/")).resolve()
                if fp != _sb_root and not str(fp).startswith(str(_sb_root) + "/"):
                    missing.append(p)
                    continue
            if not fp.exists() or not fp.is_file():
                missing.append(p)
                continue
            # Reject an oversized file BEFORE read_text pulls it all into
            # memory (the total-chars cap downstream can't help post-read).
            if fp.stat().st_size > _MAX_SOURCE_FILE_BYTES:
                missing.append(f"{p} (too large, >{_MAX_SOURCE_FILE_BYTES // (1024*1024)} MB)")
                continue
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            missing.append(p)
            continue
        if not text.strip():
            continue
        split = _split_markdown_into_sections(text)
        if split:
            out.extend(split)
        else:
            stem = Path(p).stem.replace("_", " ").replace("-", " ").strip().title()
            out.append({"heading": stem, "body": text.strip()})
    return out, missing


def _normalise_sections(sections: Any, body: Any, content: Any) -> list[dict]:
    """Coerce whatever the LLM passed into ``[{heading, body}, ...]``.

    Why so forgiving: in self-play logs the agent regularly hallucinates
    alternate parameter names (``body=``, ``content=``, ``text=``) and
    sometimes passes a single string when it should have passed a list.
    Mirrors the healing block at the top of ``tool_generate_image``.
    """
    # The section list frequently arrives as a JSON-encoded *string* rather
    # than a real list — e.g. ``sections='[{"heading": "Intro", "body": ...}]'``.
    # This happens when the model stringifies the argument, or when the XML
    # tool-call parser stores a ``<parameter name="sections">[...]</parameter>``
    # body verbatim. Without this decode the call collapses into ONE section
    # whose body is the raw JSON text — the agent then sees a 1-page report and
    # retries the call forever. Recover the structure before anything else.
    if isinstance(sections, str):
        stripped = sections.strip()
        if stripped[:1] in ("[", "{"):
            try:
                parsed = json.loads(stripped)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                sections = parsed
            elif isinstance(parsed, dict):
                sections = [parsed]

    # A native dict (single section) arrives directly under a JSON transport —
    # wrap it so it isn't rejected as "missing" when the string form is handled.
    if isinstance(sections, dict):
        sections = [sections]

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
        # Auto-structure a whole-report-in-one-string into sections on its
        # top-level headers, so a "dumped" report still renders as a proper
        # multi-section PDF rather than one flat blob. Falls back to a
        # single section when there are no headers to split on.
        split = _split_markdown_into_sections(fallback)
        if split:
            return split
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


def _render_to_pdf(full_html: str, out_path: Path) -> "tuple[int, bool]":
    """Render ``full_html`` into a multi-page A4 PDF at ``out_path``.

    Uses ``fitz.Story`` + ``fitz.DocumentWriter`` to flow arbitrary
    HTML across as many pages as needed. Returns ``(page_count, truncated)``
    where ``truncated`` is True when the 200-page safety cap cut content off.
    Raises on a hard render failure — the caller wraps the exception
    into a SYSTEM ERROR string for the LLM."""
    import fitz  # PyMuPDF

    mediabox = fitz.Rect(0, 0, _PAGE_W, _PAGE_H)
    where = fitz.Rect(*_CONTENT_RECT)
    story = fitz.Story(html=full_html)

    writer = fitz.DocumentWriter(str(out_path))
    pages = 0
    more = True
    truncated = False
    try:
        while more:
            dev = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()
            pages += 1
            if pages > 200:            # safety cap
                truncated = more  # content remained → the PDF is cut short
                break
    finally:
        writer.close()
    return pages, truncated


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
    source_files: Any = None,
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

    # Build sections directly from sandbox source files when asked. This is
    # the scalable path for a detailed report compiled from many task
    # files: the model lists the files (cheap) and the tool reads + splits
    # them, so the full content reaches the PDF without being squeezed
    # through the model's output tokens (which produced a thin summary /
    # a stall). Any explicit sections (e.g. an intro) lead; file sections
    # follow.
    source_files = (
        source_files or kwargs.get("files") or kwargs.get("from_files")
        or kwargs.get("source_file") or kwargs.get("source")
    )
    file_missing: list[str] = []
    if source_files and sandbox_dir is not None:
        file_secs, file_missing = _sections_from_files(Path(sandbox_dir), source_files)
        secs = secs + file_secs

    if not title:
        return (
            "SYSTEM ERROR: 'title' is REQUIRED for report_pdf. Pass the "
            "report's display title as a string (e.g. title='Top 5 Tech "
            "Trends for 2027')."
        )
    if not secs:
        _miss = (f" (source_files given but none readable: {file_missing})"
                 if file_missing else "")
        return (
            "SYSTEM ERROR: 'sections' is REQUIRED for report_pdf. Pass a "
            "list of {heading, body} dicts (body is markdown), a single "
            "markdown string, OR — to compile a detailed report from files "
            "you already wrote — source_files=['a.md','b.md', …] and the "
            "tool will read and section them for you." + _miss
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

    # The /api/download/<path> route resolves against the sandbox ROOT, but
    # when a project is active sandbox_dir is scoped to <root>/projects/<id>.
    # Build the download link RELATIVE TO THE ROOT so the file is reachable
    # (the route accepts a sub-path and containment-checks it).
    from .file_system import project_download_prefix
    download_rel = f"{project_download_prefix(sandbox_dir)}{final_name}"

    pretty_log("PDF Report", f"Title: {title[:60]} | sections={len(secs)}", icon=Icons.REPORT_PDF)

    full_html = _build_html(title, subtitle, author, secs)
    try:
        pages, _pdf_truncated = await asyncio.to_thread(_render_to_pdf, full_html, out_path)
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

    # stat() is outside the render try/except — guard it so a "successful"
    # render that produced no file surfaces a clean error, not a raw crash.
    try:
        size_kb = out_path.stat().st_size / 1024
    except OSError as e:
        return f"SYSTEM ERROR: PDF render reported success but the file is missing ({e})."
    _miss_note = (
        f"\n\n(Note: {len(file_missing)} source file(s) could not be read and "
        f"were skipped: {file_missing}.)"
        + _available_files_hint(Path(sandbox_dir))
        if file_missing else ""
    )
    _trunc_note = (
        "\n\n⚠️ The document exceeded the 200-page safety cap and was TRUNCATED — "
        "content is missing. Split it into multiple reports or shorten the input."
        if _pdf_truncated else ""
    )
    return (
        f"SUCCESS: PDF report generated ({pages} page(s), {size_kb:.1f} KB). "
        "DO NOT CALL THIS TOOL AGAIN with the same title. Respond DIRECTLY "
        "to the user by including this exact markdown so the file is "
        f"offered for download:\n\n"
        f"[📄 {title} (PDF)](/api/download/{download_rel})"
        f"{_miss_note}{_trunc_note}"
    )
