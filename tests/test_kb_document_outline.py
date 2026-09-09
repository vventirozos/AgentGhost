"""Pins for the document OUTLINE route (request e0f4a8bd, 2026-09-08).

THE LIVE FAILURE. "How many chapters does the postgresql 19 manual in your
knowledgebase have?" ran for 5+ minutes across 20+ turns and never got an
answer. The question is about a document's SHAPE; the only retrieval on
offer was semantic, so `query` returned eight passages at relevance 0.08
(text-search headline options, EXPLAIN output) under a footer that said "if
they do not contain the answer, say so and query again with different
wording" — an instruction that cannot terminate for a structural question.
The model obeyed it ten times, each adding ~10 KB of irrelevant passages to
a context that reached 170 K chars (turns went 6 s → 30 s), and in between
spent six turns writing probe.py, pip-installing pypdf and running `find /`
for a PDF that is not in the sandbox.

The structure was never missing. `pdf_ingest` reads the PDF's own table of
contents to build its breadcrumbs — and then dropped it.

These tests run the REAL streaming ingest over a REAL PDF (PyMuPDF builds
one with a real TOC) and the REAL persistence/derive code on a real
`VectorMemory` instance, so a pin here fails when the shipped path breaks,
not when a copy of it does.
"""
import json
import os
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from ghost_agent.memory import pdf_ingest
from ghost_agent.memory.pdf_ingest import (build_page_breadcrumbs,
                                           ingest_pdf_streaming, normalise_toc)
from ghost_agent.memory.vector import VectorMemory
from ghost_agent.tools.memory import (_outline_labels, _outline_level_counts,
                                      render_document_outline,
                                      tool_gain_knowledge, tool_knowledge_base)

#: The shape of a real manual: parts → chapters → sections. Three chapters,
#: which is what "how many chapters" must return.
TOC = [
    [1, "Part I. Tutorial", 1],
    [2, "Chapter 1. Getting Started", 2],
    [3, "1.1. Installation", 2],
    [2, "Chapter 2. The SQL Language", 4],
    [1, "Part II. Internals", 5],
    [2, "Chapter 3. Storage", 5],
]
CHAPTERS = 3
PARTS = 2


def _make_pdf(path, toc=TOC, pages=6):
    import fitz
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text(
            (72, 100),
            f"Body text of page {i + 1}. It discusses wal_level, "
            f"pg_stat_activity and other identifiers at length.")
    if toc:
        doc.set_toc(toc)
    doc.save(str(path))
    doc.close()
    return path


class FakeStore(VectorMemory):
    """A real `VectorMemory` with only what the outline path touches.

    `__init__` is bypassed on purpose: the embedding model and Chroma client
    are irrelevant here, and `_get_lock` is documented to cope. Every method
    under test is the SHIPPED one.
    """

    def __init__(self, tmp_path):
        self.chroma_dir = tmp_path
        self.outlines_file = tmp_path / "document_outlines.json"
        self.library_file = tmp_path / "library_index.json"
        self.library_file.write_text("[]")
        self._lock = threading.RLock()
        self.chunks = []            # what the ingest actually wrote
        self.rows = []              # (text, type) — the store's real shape
        self.collection = self      # `derive_*` reads self.collection.get

    # --- the ingest sink -------------------------------------------------
    def add(self, text, meta=None):          # the doc-level summary sink
        """The ingest writes ONE `document_summary` row under the SAME
        `source` as the chunks; it is stored here the same way, so the
        outline's type filter is exercised rather than assumed."""
        self.summaries = getattr(self, "summaries", [])
        self.summaries.append(text)
        self.rows.append((text, (meta or {}).get("type", "auto")))

    def ingest_document(self, filename, chunks, _batch=False):
        self.rows.extend((c, "document") for c in chunks)
        self.chunks.extend(chunks)
        lib = json.loads(self.library_file.read_text())
        if filename not in lib:
            lib.append(filename)
            self.library_file.write_text(json.dumps(lib))
        return True, "ok"

    def get_library(self):
        return json.loads(self.library_file.read_text())

    # --- the Chroma surface `derive_document_outline` uses ---------------
    def get(self, where=None, include=None, limit=None, offset=0):
        want = None
        for clause in (where or {}).get("$and", []):
            if "type" in clause:
                want = clause["type"]
        pool = [t for t, ty in self.rows if want is None or ty == want]
        return {"documents": pool[offset:offset + (limit or len(pool))]}


@pytest.fixture
def store(tmp_path):
    return FakeStore(tmp_path)


# --- the writer keeps the structure -------------------------------------

def test_the_ingest_keeps_the_table_of_contents(tmp_path, store):
    """World where it fails: the TOC is read, used for breadcrumbs and
    dropped — which is exactly what shipped until 2026-09-09, and why
    request e0f4a8bd could not be answered by any tool call."""
    pdf = _make_pdf(tmp_path / "manual.pdf")
    stats = ingest_pdf_streaming(pdf, "manual.pdf", store)
    assert stats.chunks > 0
    assert stats.pages_total == 6
    assert [list(e) for e in stats.outline] == [list(e) for e in TOC]


def test_the_breadcrumbs_and_the_outline_share_one_normaliser():
    """Two private normalisers of the same table is the §4FN compound-age
    defect. `build_page_breadcrumbs` must go through `normalise_toc`, so a
    fix to one is a fix to both."""
    calls = {"n": 0}
    real = pdf_ingest.normalise_toc

    def counting(toc):
        calls["n"] += 1
        return real(toc)

    pdf_ingest.normalise_toc = counting
    try:
        build_page_breadcrumbs(TOC, 6)
    finally:
        pdf_ingest.normalise_toc = real
    assert calls["n"] == 1, "build_page_breadcrumbs normalises the TOC itself"


def test_a_malformed_toc_row_is_dropped_not_fatal():
    entries = normalise_toc([[1, "Good", 3], ["x", "bad level", 1],
                             [1, "", 5], [1, "no page", 0], None])
    assert entries == [(1, "Good", 3)]


# --- the reader answers the question ------------------------------------

@pytest.mark.asyncio
async def test_the_outline_action_answers_how_many_chapters(tmp_path, store):
    """THE REGRESSION. One call, no semantic query, and the count is in the
    output. World where it fails: `outline` is not wired, or reports only
    titles (a model would have to count them itself, which is what it was
    doing from memory when the request stalled)."""
    pdf = _make_pdf(tmp_path / "manual.pdf")
    stats = ingest_pdf_streaming(pdf, "manual.pdf", store)
    store.set_document_outline("manual.pdf", {
        "filename": "manual.pdf", "source": "toc",
        "entries": [list(e) for e in stats.outline],
        "pages": stats.pages_total, "chunks": stats.chunks})

    out = await tool_knowledge_base(action="outline", filename="manual.pdf",
                                    memory_system=store)
    # THE ANSWER, in the document's own words — not the level total. On the
    # live manual level 2 holds 91 entries and exactly 70 of them are
    # chapters, so a level count reported as a chapter count answers 91 to a
    # question whose true answer is 70.
    assert f"{CHAPTERS} \u00d7 Chapter" in out, out
    assert f"{PARTS} \u00d7 Part" in out, out
    assert f"level 2: {CHAPTERS}" in out and f"level 1: {PARTS}" in out
    assert "Chapter 1. Getting Started" in out and "Part II. Internals" in out
    # …and the deepest level is NOT printed at the default depth, while its
    # count still is — the counts are complete at every depth.
    assert "1.1. Installation" not in out and "level 3: 1" in out


@pytest.mark.asyncio
async def test_the_outline_is_rebuilt_for_a_document_ingested_before_the_fix(tmp_path, store):
    """The LIVE manual's path: ingested 2026-09-04, so no record exists. The
    breadcrumbs on its own chunks carry the structure; the rebuild must find
    the same counts as the table of contents did, and cache them.

    World where it fails: `outline` only reads the cache, so every document
    ingested before this change stays as unanswerable as it was."""
    pdf = _make_pdf(tmp_path / "manual.pdf")
    ingest_pdf_streaming(pdf, "manual.pdf", store)
    assert store.get_document_outline("manual.pdf") == {}   # nothing stored

    out = await tool_knowledge_base(action="outline", filename="manual.pdf",
                                    memory_system=store)
    assert f"{CHAPTERS} \u00d7 Chapter" in out, out
    assert f"{PARTS} \u00d7 Part" in out, out
    assert "Rebuilt from the stored section breadcrumbs" in out
    # cached, so the second call is free
    rec = store.get_document_outline("manual.pdf")
    assert rec.get("source") == "breadcrumbs"
    assert [r[1] for r in rec["entries"]][:3] == [
        "Part I. Tutorial", "Chapter 1. Getting Started", "1.1. Installation"]


def test_the_rebuild_reads_the_store_in_pages(tmp_path, store):
    """Peak memory is one page, not one manual — the PostgreSQL document is
    7.6 M chars and materialising it to count headings is how the reset_all
    path once stalled every concurrent request."""
    pdf = _make_pdf(tmp_path / "manual.pdf")
    ingest_pdf_streaming(pdf, "manual.pdf", store)
    seen = []
    real_get = store.get

    def spy(**kwargs):
        seen.append(kwargs.get("limit"))
        return real_get(**kwargs)

    store.get = spy
    try:
        rec = store.derive_document_outline("manual.pdf", page=2)
    finally:
        store.get = real_get
    assert rec["entries"], rec
    assert seen and all(n == 2 for n in seen), seen
    assert len(seen) > 1, "a single unbounded read is not paging"


def test_the_rebuild_orders_chapters_numerically(store):
    """"Chapter 10" after "Chapter 9": a plain string sort puts 10 first,
    and an outline in the wrong order is a wrong answer to "what is the
    last chapter"."""
    store.rows = [
        (f"[m.pdf] Part I. X › Chapter {n}. T{n}\nbody", "document")
        for n in (10, 9, 1)]
    rec = store.derive_document_outline("m.pdf")
    assert [r[1] for r in rec["entries"]] == [
        "Part I. X", "Chapter 1. T1", "Chapter 9. T9", "Chapter 10. T10"]


def test_a_document_with_no_outline_says_so(store):
    """Refuse to invent structure: a plain-text ingest has none, and the
    honest answer names the route that does work."""
    out = render_document_outline({"filename": "notes.txt", "chunks": 4,
                                   "entries": []})
    assert "NO table of contents" in out and "action='query'" in out


def test_a_huge_outline_is_capped_but_its_counts_are_not(store):
    """The PostgreSQL manual has ~6,200 outline entries. The tree is capped;
    the counts — the answer — never are."""
    entries = [[2, f"Chapter {i}. T", i] for i in range(1, 501)]
    out = render_document_outline({"filename": "big.pdf", "entries": entries})
    assert "level 2: 500" in out
    assert "more entries at this depth" in out
    assert len(out.splitlines()) < 200, "the tree is not capped"


# --- the library tells the model what it holds --------------------------

@pytest.mark.asyncio
async def test_list_docs_reports_shape_not_just_a_name(tmp_path, store):
    """`LIBRARY CONTENTS (1 files): - postgresql-19-A4.pdf` was the entire
    answer the model got before asking 20 structural questions."""
    pdf = _make_pdf(tmp_path / "manual.pdf")
    stats = ingest_pdf_streaming(pdf, "manual.pdf", store)
    bare = await tool_knowledge_base(action="list_docs", memory_system=store)
    assert "manual.pdf" in bare and "action='outline'" in bare

    store.set_document_outline("manual.pdf", {
        "filename": "manual.pdf", "source": "toc",
        "entries": [list(e) for e in stats.outline],
        "pages": stats.pages_total, "chunks": stats.chunks})
    rich = await tool_knowledge_base(action="list_docs", memory_system=store)
    assert "6 pages" in rich and "chunks" in rich
    assert f"{PARTS}/{CHAPTERS}/1" in rich, rich


def test_forgetting_a_document_forgets_its_structure(store, monkeypatch):
    """A stored outline for a deleted document is a claim about something
    that no longer exists."""
    store.set_document_outline("m.pdf", {"filename": "m.pdf", "entries": []})
    monkeypatch.setattr(store, "collection", type("C", (), {"delete": lambda *a, **k: None})())
    monkeypatch.setattr(store, "_update_library_index", lambda *a, **k: None)
    store.delete_document_by_name("m.pdf")
    assert store.get_document_outline("m.pdf") == {}


def test_the_outline_index_survives_a_corrupt_file(store):
    """A corrupt sidecar must not take the knowledge base down with it."""
    store.outlines_file.write_text("{not json")
    assert store.get_document_outline("m.pdf") == {}
    store.set_document_outline("m.pdf", {"filename": "m.pdf", "entries": [[1, "A", 1]]})
    assert store.get_document_outline("m.pdf")["entries"] == [[1, "A", 1]]


def test_two_documents_do_not_overwrite_each_other(store):
    store.set_document_outline("a.pdf", {"filename": "a.pdf", "entries": [[1, "A", 1]]})
    store.set_document_outline("b.pdf", {"filename": "b.pdf", "entries": [[1, "B", 1]]})
    assert store.get_document_outline("a.pdf")["entries"] == [[1, "A", 1]]
    assert store.get_document_outline("b.pdf")["entries"] == [[1, "B", 1]]


def test_level_counts_ignore_malformed_rows():
    assert _outline_level_counts([[1, "a", 1], ["x", "b", 2], [2, "c", 3]]) == {1: 1, 2: 1}


@pytest.mark.asyncio
async def test_outline_of_an_unknown_document_names_what_is_available(store):
    out = await tool_knowledge_base(action="outline", filename="nope.pdf",
                                    memory_system=store)
    assert "not in the knowledge base" in out
    missing = await tool_knowledge_base(action="outline", memory_system=store)
    assert "'filename' is MANDATORY" in missing


@pytest.mark.asyncio
async def test_the_ingest_TOOL_persists_the_outline_it_computed(tmp_path, store):
    """The whole ingest path, end to end: `knowledge_base(action=
    'ingest_document')` on a real PDF must leave the structure ON DISK, so
    the next question about shape is one cached call and not a rebuild.

    World where it fails: `tool_gain_knowledge` computes `stats.outline` and
    returns without storing it — which mutation-tested as a SURVIVOR while
    the only pins set the record by hand (2026-09-09)."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    _make_pdf(sandbox / "manual.pdf")
    res = await tool_gain_knowledge("manual.pdf", sandbox, store)
    assert res.startswith("SUCCESS"), res

    rec = store.get_document_outline("manual.pdf")
    assert rec.get("source") == "toc", rec          # exact, not rebuilt
    assert rec["pages"] == 6 and rec["chunks"] > 0
    assert [r[1] for r in rec["entries"]] == [t[1] for t in TOC]

    out = await tool_knowledge_base(action="outline", filename="manual.pdf",
                                    memory_system=store)
    assert f"level 2: {CHAPTERS}" in out
    assert "Rebuilt from" not in out, "an ingested document should not need a rebuild"


# --- the counts have to be the DOCUMENT's, not the index's ---------------

@pytest.mark.parametrize("title,label", [
    ("Part I. Tutorial", "Part"),
    ("Chapter 12. Full Text Search", "Chapter"),
    ("Appendix F. Additional Supplied Modules", "Appendix"),   # LETTERED
    ("Appendix A", "Appendix"),
    ("Section 3: Introduction", "Section"),
    ("Book IV) Notes", "Book"),
    # …and prose is not a division, whatever it starts with
    ("See Also", None), ("DROP TABLE", None), ("Return Value", None),
    ("Note A brief summary", None), ("F.29. pg_overexplain", None),
    ("12.3. Controlling Text Search", None), ("Bibliography", None),
])
def test_a_division_label_is_a_shape_not_a_word_list(title, label):
    """World where it fails: the label is matched from a list of accepted
    words (stale the first time a manual says "Annex"), or the enumerator
    admits only digits and romans — which counted 5 of the live manual's 15
    lettered appendices and called the result exact."""
    got = _outline_labels([[2, title, 0]])
    assert (list(got.get(2, {})) or [None])[0] == label, got


def test_label_counts_equal_a_recount_of_the_titles():
    """Identity, not a property: the reported count must equal what a reader
    gets by counting the titles themselves."""
    entries = ([[2, f"Chapter {i}. T", 0] for i in range(1, 71)]
               + [[2, f"Appendix {c}. T", 0] for c in "ABCDEFGHIJKLMNO"]
               + [[2, "Preface", 0], [2, "See Also", 0]])
    labels = _outline_labels(entries)[2]
    assert labels["Chapter"] == sum(
        1 for e in entries if e[1].startswith("Chapter "))
    assert labels["Appendix"] == sum(
        1 for e in entries if e[1].startswith("Appendix "))
    assert labels["Chapter"] == 70 and labels["Appendix"] == 15
    # the level total is BIGGER than any label — and both are reported
    out = render_document_outline({"filename": "m.pdf", "entries": entries})
    assert "70 \u00d7 Chapter" in out and "15 \u00d7 Appendix" in out
    assert "level 2: 87" in out


@pytest.mark.asyncio
async def test_the_document_summary_row_is_not_an_outline_entry(tmp_path, store):
    """MEASURED ON THE LIVE STORE (2026-09-09). The ingest writes one
    `document_summary` row under the same `source` as the chunks
    ("Reference document: 3083 pages, 1936 sections…"), and reading rows by
    source alone put that sentence into the PostgreSQL manual's outline as a
    top-level heading — a fabricated division, in the one output whose whole
    job is to be exact."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    _make_pdf(sandbox / "manual.pdf")
    await tool_gain_knowledge("manual.pdf", sandbox, store)
    assert any("Reference document:" in t for t, _ in store.rows), \
        "the summary row is not even present — the pin would be vacuous"

    rec = store.derive_document_outline("manual.pdf")
    assert not any("Reference document" in r[1] for r in rec["entries"]), rec["entries"]
    assert [r[1] for r in rec["entries"]] == [t[1] for t in TOC]


# --- what the model is told about where documents live ------------------

def _kb_description():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    return next(t for t in TOOL_DEFINITIONS
                if t["function"]["name"] == "knowledge_base")["function"]["description"]


def test_the_tool_says_an_ingested_document_is_not_a_file_on_disk():
    """Request e0f4a8bd spent SIX turns on probe.py, `pip install pypdf` and
    five `find /` runs, hunting a PDF that is not in the sandbox and never
    was. The old description said "do NOT write Python scripts to read PDFs"
    but never said WHY, so the model read the file as merely missing and
    kept looking.

    World where it fails: the rule is restored without its reason, or the
    reason is written somewhere the model never reads."""
    desc = _kb_description()
    assert "AN INGESTED DOCUMENT IS NOT A FILE YOU CAN OPEN" in desc
    assert "find" in desc and "never install a PDF library" in desc
    # …and it names what DOES work, in one sentence, or the model is left
    # with a prohibition and no route — the shape that produced the five
    # `find` calls. (Asserting the bare words "query"/"outline" would be
    # vacuous: they occur in the actions list regardless.)
    assert ("These actions ARE your access to it: query for its prose, "
            "outline for its structure, list_docs for what you hold." in desc)


def test_the_reason_is_pinned_against_the_description_optimiser():
    """§4FJ: a promoted `tool_description.knowledge_base.json` rewrites this
    text toward a reward and can drop any sentence. The rule survived the
    last optimiser round; the REASON is what the model needed, so both are
    pinned and a candidate that sheds them is rejected."""
    from ghost_agent.tools import registry as R
    base = _kb_description()
    pins = R._pinned_sentences("knowledge_base", base)
    assert len(pins) == len(R.TOOL_DESC_PINNED["knowledge_base"]), \
        "a knowledge_base pin drifted out of the live baseline"
    # The REASON is pinned, not only the rule. The rule alone survived the
    # last optimiser round and the model still hunted the file for six
    # turns — a prohibition it cannot explain is one it treats as a hint.
    assert any("NOT A FILE YOU CAN OPEN" in s for s in pins), pins
    assert any("never install a PDF library" in s for s in pins), pins
    assert any("action='outline', NOT query" in s for s in pins), pins
    assert R._validate_tool_description("knowledge_base", base, base) is True
    for sent in pins:
        shed = base.replace(sent, "")
        assert R._validate_tool_description("knowledge_base", base, shed) is False, sent[:40]


def test_the_structural_question_is_routed_in_the_description_itself():
    """The model chose `query` for "how many chapters" because nothing told
    it otherwise; the schema is where that choice is made."""
    desc = _kb_description()
    assert "how many chapters" in desc
    assert "action='outline', NOT query" in desc
