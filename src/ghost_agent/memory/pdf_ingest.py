"""Streaming, structure-aware PDF ingest (2026-07-13).

Built for the real target: the PostgreSQL manual — ~3,000 pages, ~15 MB,
~9-10 M chars of extracted text. The old path could not touch it:

  * ``MAX_PDF_PAGES = 1000``      → hard refusal before doing anything.
  * ``MAX_INGEST_TEXT_CHARS = 5M``→ silent loss of ~half the manual.
  * whole-document materialisation → the full text, then the full chunk
    list, then an enriched COPY of the chunk list, all in RAM at once.
  * ``[Source: file.pdf]`` was the ONLY context on a chunk — PDF text has
    no markdown headers, so ``semantic_split_text``'s header-prepending
    never fired and a chunk about ``wal_level`` had no idea it lived under
    "19.5. Write Ahead Log".

This module fixes all four:

  1. **Streaming.** Pages are read one at a time and flushed to the store
     in batches (``BATCH_CHUNKS``); peak memory is one batch, not one
     document. A 3,000-page manual ingests in bounded RAM.
  2. **TOC breadcrumbs.** ``fitz`` exposes the PDF outline
     (``doc.get_toc()``), which maps *page → chapter/section*. Every chunk
     is prefixed with its breadcrumb, so the embedded text carries the
     structure the raw page text lost. This is the single biggest
     retrieval win on a reference manual, where the same identifier
     recurs in twenty different contexts.
  3. **Bigger chunks.** 1,200 chars (vs 600) with 150 overlap — a manual's
     unit of meaning is a parameter description or a syntax block, and
     600 chars shredded those. Still well inside the embedder's window.
  4. **Progress + cancellation-safe.** The caller gets a per-batch
     callback for the live log; the doc handle is always closed.

Pure/synchronous by design — the caller runs it in a thread. It never
raises for per-page problems (a corrupt page is skipped, not fatal).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

from ..utils.helpers import semantic_split_text

logger = logging.getLogger("GhostAgent")


# Ingest sizing. These are the ceilings the OLD code set far too low for a
# real manual; they are now high enough for the PostgreSQL docs (~3k pages,
# ~10M chars) with headroom, while still refusing something absurd that
# would OOM the box.
MAX_PDF_PAGES = 6000
MAX_TEXT_CHARS = 40_000_000        # ~40 MB of extracted text
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
BATCH_CHUNKS = 256                 # chunks flushed to the store per batch

# Page text that is pure boilerplate (running headers/footers, bare page
# numbers) adds noise to every embedding. Cheap, conservative cleanup.
_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")
_WS_RE = re.compile(r"[ \t]{2,}")


@dataclass
class IngestStats:
    pages: int = 0
    chars: int = 0
    chunks: int = 0
    sections: int = 0
    truncated: bool = False
    skipped_pages: int = 0
    errors: List[str] = field(default_factory=list)
    #: The document's own table of contents, ``[(level, title, page), ...]``
    #: in document order — the STRUCTURE, kept rather than thrown away once
    #: the breadcrumbs are built. Before 2026-09-09 this was computed and
    #: discarded, so "how many chapters does the manual have?" had no
    #: answerable route and the agent ground through 20+ semantic queries
    #: (request e0f4a8bd). Empty for a PDF with no outline.
    outline: List[Tuple[int, str, int]] = field(default_factory=list)
    #: Pages in the FILE (``st.pages`` counts pages that yielded text).
    pages_total: int = 0


# ──────────────────────────────────────────────────────────────────────
# Table of contents → per-page breadcrumb
# ──────────────────────────────────────────────────────────────────────

def normalise_toc(toc) -> List[Tuple[int, str, int]]:
    """PyMuPDF's ``[[level, title, page_1based], ...]`` → clean, sorted
    ``[(level, title, page), ...]``.

    ONE authority for the two readers — the breadcrumb builder below and the
    stored outline (`IngestStats.outline`). They disagreed about nothing
    while one of them existed; a second private normaliser is how the
    §4FN compound-age bug happened, so there is only ever this one.
    A malformed row is dropped, never fatal.
    """
    entries: List[Tuple[int, str, int]] = []
    for row in toc or ():
        try:
            level = int(row[0])
            title = " ".join(str(row[1]).split())
            page = int(row[2])
        except (TypeError, ValueError, IndexError):
            continue
        if page < 1 or not title:
            continue
        entries.append((max(1, level), title, page))
    entries.sort(key=lambda e: (e[2], e[0]))
    return entries


def build_page_breadcrumbs(toc, page_count: int) -> List[str]:
    """Map every page index → a "Chapter › Section › Subsection" string.

    ``toc`` is PyMuPDF's ``[[level, title, page_1based], ...]``. Entries are
    in document order, so we sweep forward maintaining a level stack: each
    page inherits the most recent heading path that started at or before it.
    Pages before the first TOC entry (title/copyright/TOC itself) get "".

    Returns a list of length ``page_count`` (index 0 = page 1).
    """
    crumbs = [""] * max(0, int(page_count or 0))
    if not toc or not crumbs:
        return crumbs

    entries = normalise_toc(toc)
    if not entries:
        return crumbs

    stack: List[str] = []
    idx = 0
    for page_i in range(len(crumbs)):
        page_1 = page_i + 1
        # Apply every TOC entry that starts on or before this page.
        while idx < len(entries) and entries[idx][2] <= page_1:
            level, title, _ = entries[idx]
            del stack[level - 1:]          # pop deeper/equal levels
            while len(stack) < level - 1:  # tolerate skipped levels
                stack.append("")
            stack.append(title)
            idx += 1
        crumbs[page_i] = " › ".join(s for s in stack if s)
    return crumbs


def _clean_page_text(text: str) -> str:
    if not text:
        return ""
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or _PAGE_NUMBER_RE.match(s):
            continue
        lines.append(_WS_RE.sub(" ", s))
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Streaming chunk producer
# ──────────────────────────────────────────────────────────────────────

def iter_pdf_chunks(
    file_path: Path,
    filename: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    max_pages: int = MAX_PDF_PAGES,
    max_chars: int = MAX_TEXT_CHARS,
    stats: Optional[IngestStats] = None,
) -> Iterator[str]:
    """Yield breadcrumb-prefixed chunks from a PDF, one section at a time.

    Accumulates text per TOC SECTION (not per page): a section is the
    natural retrieval unit, and chunking at section boundaries stops a
    parameter description from being split across two unrelated headings.
    A section longer than the chunk size is split internally by
    ``semantic_split_text`` — every resulting piece keeps the breadcrumb.
    """
    import fitz  # PyMuPDF — imported here so the module is import-safe

    st = stats if stats is not None else IngestStats()
    doc = fitz.open(str(file_path))
    try:
        page_count = len(doc)
        if page_count > max_pages:
            raise ValueError(
                f"PDF has {page_count} pages; ingest refuses more than "
                f"{max_pages}. Split it first."
            )
        st.pages_total = page_count
        try:
            _toc = doc.get_toc()
            crumbs = build_page_breadcrumbs(_toc, page_count)
            st.sections = len({c for c in crumbs if c})
            # KEEP the structure, do not just consume it (request e0f4a8bd).
            st.outline = normalise_toc(_toc)
        except Exception as e:  # noqa: BLE001 — outline is optional
            logger.debug("PDF TOC unavailable (%s); ingesting without breadcrumbs", e)
            crumbs = [""] * page_count

        buf: List[str] = []
        buf_crumb = ""
        buf_len = 0

        def _flush() -> Iterator[str]:
            nonlocal buf, buf_len
            if not buf:
                return
            body = "\n".join(buf).strip()
            buf, buf_len = [], 0
            if not body:
                return
            header = f"[{filename}]"
            if buf_crumb:
                header += f" {buf_crumb}"
            # Chunk the section, then stamp the breadcrumb on every piece so
            # the EMBEDDED text carries the structure (not just the metadata).
            for piece in semantic_split_text(body, chunk_size, chunk_overlap):
                piece = piece.strip()
                if piece:
                    st.chunks += 1
                    yield f"{header}\n{piece}"

        for page_i in range(page_count):
            if st.chars >= max_chars:
                st.truncated = True
                break
            try:
                text = doc[page_i].get_text()
            except Exception as pe:  # noqa: BLE001 — skip the page, not the doc
                st.skipped_pages += 1
                st.errors.append(f"page {page_i + 1}: {pe}")
                continue
            text = _clean_page_text(text)
            if not text:
                continue

            st.pages += 1
            st.chars += len(text)
            crumb = crumbs[page_i] if page_i < len(crumbs) else ""

            # Section boundary → flush what we have under the OLD breadcrumb.
            if crumb != buf_crumb and buf:
                yield from _flush()
            buf_crumb = crumb
            buf.append(text)
            buf_len += len(text)

            # Guard: a section with no TOC entries (or a huge one) must not
            # grow unbounded — flush once it exceeds a few chunks' worth.
            if buf_len >= chunk_size * 8:
                yield from _flush()

        yield from _flush()
    finally:
        try:
            doc.close()
        except Exception:  # noqa: BLE001
            pass


def ingest_pdf_streaming(
    file_path: Path,
    filename: str,
    memory_system,
    *,
    progress: Optional[Callable[[IngestStats], None]] = None,
    **kwargs,
) -> IngestStats:
    """Stream a PDF into the vector store in bounded memory.

    Chunks are flushed to ``memory_system.ingest_document`` in batches of
    ``BATCH_CHUNKS``, so peak RAM is one batch — not one document. Returns
    the ``IngestStats``; raises only on a fatal condition (unreadable PDF,
    page-cap breach), never on a single bad page.
    """
    stats = IngestStats()
    batch: List[str] = []
    flushed = 0

    for chunk in iter_pdf_chunks(file_path, filename, stats=stats, **kwargs):
        batch.append(chunk)
        if len(batch) >= BATCH_CHUNKS:
            ok, msg = memory_system.ingest_document(filename, batch, _batch=True)
            if not ok:
                raise RuntimeError(f"embedding failed at chunk {flushed}: {msg}")
            flushed += len(batch)
            batch = []
            if progress:
                try:
                    progress(stats)
                except Exception:  # noqa: BLE001 — progress must never break ingest
                    pass

    if batch:
        ok, msg = memory_system.ingest_document(filename, batch, _batch=True)
        if not ok:
            raise RuntimeError(f"embedding failed at chunk {flushed}: {msg}")
        flushed += len(batch)
        if progress:
            try:
                progress(stats)
            except Exception:  # noqa: BLE001
                pass

    return stats
