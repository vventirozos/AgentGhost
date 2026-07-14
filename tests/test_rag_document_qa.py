"""RAG overhaul (2026-07-13): streaming PDF ingest, TOC breadcrumbs,
document-scoped QA, and the embedder-fingerprint guard.

Motivating target: load the full PostgreSQL manual (~3k pages, ~15 MB,
~10 M chars) and ask it questions. The old pipeline could not:

  * ingest REFUSED >1000 pages, then silently truncated at 5 M chars;
  * PDF text has no markdown headers, so chunks carried no section
    context at all — a chunk about `wal_level` did not know it lived
    under "19.5. Write Ahead Log";
  * there was NO document-scoped retrieval: the only path to the model
    was ambient hydration, where doc chunks competed with episodes and
    skills for a shared 6-12k char budget, capped at 12 fragments from
    the WHOLE store.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.memory.pdf_ingest import (
    IngestStats,
    build_page_breadcrumbs,
    ingest_pdf_streaming,
)
from ghost_agent.memory.vector import (
    EMBED_MODEL_NAME,
    _embedder_sidecar_mismatch,
)


# ──────────────────────────────────────────────────────────────────────
# TOC breadcrumbs — the biggest retrieval win on a reference manual
# ──────────────────────────────────────────────────────────────────────

_PG_TOC = [
    [1, "Chapter 19. Server Configuration", 500],
    [2, "19.5. Write Ahead Log", 520],
    [3, "19.5.1. Settings", 521],
    [2, "19.6. Replication", 540],
    [1, "Chapter 20. Client Authentication", 560],
]


def test_breadcrumbs_nest_and_pop_by_level():
    crumbs = build_page_breadcrumbs(_PG_TOC, 570)
    # Before the first entry: no breadcrumb (title / TOC pages).
    assert crumbs[0] == ""
    # Inside a subsection: the FULL path.
    assert crumbs[520] == (
        "Chapter 19. Server Configuration › 19.5. Write Ahead Log › "
        "19.5.1. Settings"
    )
    # A sibling at level 2 POPS the level-3 entry (not append).
    assert crumbs[539] == "Chapter 19. Server Configuration › 19.6. Replication"
    # A new level-1 chapter pops everything below it.
    assert crumbs[560] == "Chapter 20. Client Authentication"


def test_breadcrumbs_survive_a_malformed_outline():
    # A junk outline must not kill a 3,000-page ingest.
    bad = [["x", "no page", None], [1, "", 5], [2, "Real", 3]]
    crumbs = build_page_breadcrumbs(bad, 5)
    assert len(crumbs) == 5
    assert crumbs[4] == "Real"


def test_breadcrumbs_empty_toc_is_safe():
    assert build_page_breadcrumbs([], 3) == ["", "", ""]
    assert build_page_breadcrumbs(None, 2) == ["", ""]


# ──────────────────────────────────────────────────────────────────────
# Streaming ingest — bounded memory, breadcrumb-stamped chunks
# ──────────────────────────────────────────────────────────────────────

class _FakePage:
    def __init__(self, text):
        self._t = text

    def get_text(self):
        return self._t


class _FakeDoc:
    """Minimal fitz.Document stand-in."""

    def __init__(self, pages, toc):
        self._pages = [_FakePage(p) for p in pages]
        self._toc = toc
        self.closed = False

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, i):
        return self._pages[i]

    def get_toc(self):
        return self._toc

    def close(self):
        self.closed = True


def _fake_fitz(pages, toc):
    mod = MagicMock()
    doc = _FakeDoc(pages, toc)
    mod.open.return_value = doc
    return mod, doc


def test_streaming_ingest_stamps_breadcrumbs_on_every_chunk(tmp_path):
    pages = [
        "front matter",                       # p1 — before any TOC entry
        "wal_level determines how much information is written to the WAL. "
        * 30,                                  # p2 — inside 19.5
    ]
    toc = [[1, "Chapter 19. Server Configuration", 2],
           [2, "19.5. Write Ahead Log", 2]]
    fake_mod, doc = _fake_fitz(pages, toc)

    mem = MagicMock()
    mem.ingest_document.return_value = (True, "ok")

    with patch.dict("sys.modules", {"fitz": fake_mod}):
        stats = ingest_pdf_streaming(
            tmp_path / "pg.pdf", "pg.pdf", mem)

    assert stats.pages == 2
    assert stats.chunks >= 1
    assert doc.closed is True

    # Every chunk written carries the file + its section breadcrumb.
    written = [c for call in mem.ingest_document.call_args_list
               for c in call.args[1]]
    assert written, "nothing was written to the store"
    wal_chunks = [c for c in written if "wal_level" in c]
    assert wal_chunks, "the WAL page produced no chunk"
    for c in wal_chunks:
        assert c.startswith("[pg.pdf]")
        assert "19.5. Write Ahead Log" in c.splitlines()[0]

    # Batch mode: the streamer owns enrichment, so ingest_document must be
    # told NOT to re-enrich.
    for call in mem.ingest_document.call_args_list:
        assert call.kwargs.get("_batch") is True


def test_streaming_ingest_flushes_in_bounded_batches(tmp_path):
    # 40 dense pages → several batches; peak memory is one batch, not the doc.
    pages = ["lorem ipsum dolor sit amet " * 120 for _ in range(40)]
    fake_mod, _ = _fake_fitz(pages, [])
    mem = MagicMock()
    mem.ingest_document.return_value = (True, "ok")

    with patch.dict("sys.modules", {"fitz": fake_mod}):
        stats = ingest_pdf_streaming(tmp_path / "big.pdf", "big.pdf", mem)

    from ghost_agent.memory.pdf_ingest import BATCH_CHUNKS
    assert stats.chunks > 0
    for call in mem.ingest_document.call_args_list:
        assert len(call.args[1]) <= BATCH_CHUNKS


def test_streaming_ingest_skips_a_bad_page_not_the_document(tmp_path):
    class _BoomPage:
        def get_text(self):
            raise RuntimeError("corrupt page")

    fake_mod, doc = _fake_fitz(["good page text " * 40], [])
    doc._pages.insert(0, _BoomPage())
    mem = MagicMock()
    mem.ingest_document.return_value = (True, "ok")

    with patch.dict("sys.modules", {"fitz": fake_mod}):
        stats = ingest_pdf_streaming(tmp_path / "x.pdf", "x.pdf", mem)

    assert stats.skipped_pages == 1
    assert stats.chunks >= 1          # the good page still made it


def test_streaming_ingest_refuses_an_absurd_page_count(tmp_path):
    fake_mod, _ = _fake_fitz(["p"] * 3, [])
    mem = MagicMock()
    with patch.dict("sys.modules", {"fitz": fake_mod}):
        with pytest.raises(ValueError, match="refuses more than"):
            ingest_pdf_streaming(tmp_path / "x.pdf", "x.pdf", mem, max_pages=2)


def test_page_cap_is_high_enough_for_a_real_manual():
    # The old cap (1000) hard-refused the PostgreSQL manual (~3k pages).
    from ghost_agent.memory import pdf_ingest as pi
    assert pi.MAX_PDF_PAGES >= 3000
    assert pi.MAX_TEXT_CHARS >= 10_000_000
    # Chunks big enough to hold a parameter description (was 600).
    assert pi.CHUNK_SIZE >= 1000


# ──────────────────────────────────────────────────────────────────────
# Document-scoped QA
# ──────────────────────────────────────────────────────────────────────

def _vm_with_docs(monkeypatch, docs, dists):
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory.__new__(VectorMemory)
    vm.collection = MagicMock()
    vm.collection.query.return_value = {
        "documents": [docs],
        "distances": [dists],
        "ids": [[f"id{i}" for i in range(len(docs))]],
    }
    vm.embedding_fn = MagicMock(return_value=[[0.1] * 384])
    return vm


def test_search_document_is_scoped_to_one_file(monkeypatch):
    vm = _vm_with_docs(monkeypatch, ["[pg.pdf] 19.5\nwal_level ..."], [0.3])
    hits = vm.search_document("pg.pdf", "what does wal_level do?", k=5)
    assert hits and "wal_level" in hits[0]["text"]
    # The Chroma query MUST filter by source — this is the whole point:
    # the ambient path searched the entire memory soup.
    kwargs = vm.collection.query.call_args.kwargs
    assert kwargs["where"] == {"source": "pg.pdf"}
    # Deep pool for a manual with thousands of chunks.
    assert kwargs["n_results"] >= 40


def test_search_document_uses_the_bge_query_instruction(monkeypatch):
    vm = _vm_with_docs(monkeypatch, ["chunk"], [0.2])
    vm.search_document("pg.pdf", "how do I vacuum?", k=3)
    # BGE is ASYMMETRIC: the query gets an instruction prefix, passages do
    # not. Chroma's query_texts= path can't express that (it reuses the doc
    # embedder), so the query must be embedded here and passed as vectors.
    assert "query_embeddings" in vm.collection.query.call_args.kwargs
    embedded = vm.embedding_fn.call_args.args[0][0]
    assert embedded.startswith("Represent this sentence for searching")
    assert "how do I vacuum?" in embedded


def test_search_document_returns_nothing_for_a_blank_question():
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory.__new__(VectorMemory)
    vm.collection = MagicMock()
    assert vm.search_document("pg.pdf", "") == []
    assert vm.search_document("", "q") == []


@pytest.mark.asyncio
async def test_query_action_rejects_an_uningested_document():
    from ghost_agent.tools.memory import tool_query_document
    mem = MagicMock()
    mem.get_library.return_value = ["other.pdf"]
    out = await tool_query_document("pg.pdf", "wal?", memory_system=mem)
    assert "not in the knowledge base" in out
    assert "other.pdf" in out          # tells the model what IS available


@pytest.mark.asyncio
async def test_query_action_returns_ranked_passages_with_guidance():
    from ghost_agent.tools.memory import tool_query_document
    mem = MagicMock()
    mem.get_library.return_value = ["pg.pdf"]
    mem.search_document.return_value = [
        {"text": "[pg.pdf] 19.5\nwal_level ...", "id": "a", "score": 0.11},
        {"text": "[pg.pdf] 19.6\narchive_mode ...", "id": "b", "score": 0.22},
    ]
    out = await tool_query_document("pg.pdf", "wal_level?", memory_system=mem)
    assert "wal_level" in out and "archive_mode" in out
    assert "[1]" in out and "[2]" in out           # ranked
    assert "breadcrumb" in out.lower()             # told to cite the section
    assert "query again" in out.lower()            # told it may iterate


@pytest.mark.asyncio
async def test_query_action_tolerates_a_stem_filename():
    from ghost_agent.tools.memory import tool_query_document
    mem = MagicMock()
    mem.get_library.return_value = ["postgresql.pdf"]
    mem.search_document.return_value = [
        {"text": "chunk", "id": "a", "score": 0.1}]
    out = await tool_query_document("postgresql", "q", memory_system=mem)
    assert "PASSAGES FROM 'postgresql.pdf'" in out


def test_query_action_is_registered_in_the_tool_schema():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    kb = next(t for t in TOOL_DEFINITIONS
              if t["function"]["name"] == "knowledge_base")
    props = kb["function"]["parameters"]["properties"]
    assert "query" in props["action"]["enum"]
    assert "question" in props
    # The model must be steered from `recall` to the scoped tool for manuals.
    recall = next(t for t in TOOL_DEFINITIONS
                  if t["function"]["name"] == "recall")
    assert "knowledge_base(action='query'" in recall["function"]["description"]


# ──────────────────────────────────────────────────────────────────────
# Embedder fingerprint — a silent-wrongness guard
# ──────────────────────────────────────────────────────────────────────

def test_embedder_default_is_bge_small():
    assert EMBED_MODEL_NAME == "BAAI/bge-small-en-v1.5"


def test_sidecar_matching_model_is_fine(tmp_path):
    p = tmp_path / "embedder.json"
    p.write_text(json.dumps({"model": EMBED_MODEL_NAME, "dim": 384}))
    assert _embedder_sidecar_mismatch(p, EMBED_MODEL_NAME, 500) is None


def test_sidecar_different_model_is_a_mismatch(tmp_path):
    # Both models are 384-d and L2-normalised, so NOTHING errors — the
    # vectors just silently stop meaning anything. This guard is the only
    # thing standing between that and plausible-looking garbage retrieval.
    p = tmp_path / "embedder.json"
    p.write_text(json.dumps({"model": "all-MiniLM-L6-v2", "dim": 384}))
    reason = _embedder_sidecar_mismatch(p, EMBED_MODEL_NAME, 160)
    assert reason and "all-MiniLM-L6-v2" in reason


def test_legacy_store_without_a_sidecar_is_a_mismatch(tmp_path):
    reason = _embedder_sidecar_mismatch(
        tmp_path / "absent.json", EMBED_MODEL_NAME, 160)
    assert reason and "no embedder fingerprint" in reason


def test_fresh_empty_store_without_a_sidecar_is_fine(tmp_path):
    assert _embedder_sidecar_mismatch(
        tmp_path / "absent.json", EMBED_MODEL_NAME, 0) is None
