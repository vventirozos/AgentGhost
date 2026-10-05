"""§4LT — knowledge base: behaviour pins for the review's fixes."""
from __future__ import annotations

import random
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.outcome import OutcomeStatus


def _run_with_deadline(fn, seconds=10):
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("out", fn()), daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), "the splitter did not terminate"
    return box["out"]


# ── CRIT: the splitter always terminates, and every chunk fits ──

def test_the_shape_that_froze_the_agent_splits_at_once():
    from ghost_agent.utils.helpers import recursive_split_text, semantic_split_text
    page = ("Acme Hardware - Opening hours and contact. "
            + " | ".join(f"Category {i}: drills, saws, sanders, clamps" for i in range(25))
            + ". Contact us at the store.")
    for split in (recursive_split_text, semantic_split_text):
        out = _run_with_deadline(lambda: split(page, 600, 100))
        assert out and all(len(c) <= 600 for c in out)


def test_random_texts_always_terminate_within_size():
    from ghost_agent.utils.helpers import recursive_split_text
    rnd = random.Random(7)
    seps = ["\n\n", "\n", ". ", "? ", "; ", ", ", " ", "x", "word"]

    def fuzz():
        for _ in range(800):
            s = "".join(rnd.choice(seps) * rnd.randint(1, 3) for _ in range(rnd.randint(50, 700)))
            cs, ov = rnd.choice([50, 120, 600]), rnd.choice([0, 20, 100])
            assert all(len(c) <= cs for c in recursive_split_text(s, cs, ov))
        return True
    assert _run_with_deadline(fuzz, 30)


async def test_the_ingest_splits_off_the_event_loop(tmp_path, monkeypatch):
    import ghost_agent.tools.memory as M
    main = threading.get_ident()
    where = []

    def split(text, *a, **k):
        where.append(threading.get_ident())
        return ["one chunk"]
    monkeypatch.setattr(M, "semantic_split_text", split)
    (tmp_path / "n.txt").write_text("Some text " * 30)
    ms = MagicMock()
    ms.get_library.return_value = []
    ms.ingest_document.return_value = (True, "ok")
    await M.tool_gain_knowledge("n.txt", tmp_path, ms)
    assert where and where[0] != main


# ── M1: repeated identical chunks ingest ──

def test_duplicate_chunks_never_reach_one_upsert_twice(tmp_path):
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory.__new__(VectorMemory)
    vm._lock = threading.RLock()
    vm._get_lock = lambda: vm._lock
    vm.collection = MagicMock()
    vm.get_library = lambda: []
    vm._update_library_index = lambda *a, **k: True
    ok, _ = vm.ingest_document("log.txt", ["Traceback A", "middle", "Traceback A"])
    ids = vm.collection.upsert.call_args.kwargs["ids"]
    assert ok and len(ids) == len(set(ids)) == 2


# ── M2: a PDF that fails part-way leaves nothing behind ──

def test_a_failed_pdf_ingest_is_rolled_back(monkeypatch):
    import ghost_agent.memory.pdf_ingest as P
    monkeypatch.setattr(P, "iter_pdf_chunks",
                        lambda fp, fn, stats=None, **k: iter([f"c{i}" for i in range(P.BATCH_CHUNKS * 2 + 1)]))
    calls = {"n": 0}

    def ingest(fn, batch, _batch=False):
        calls["n"] += 1
        return (calls["n"] == 1, "embedder down")
    ms = SimpleNamespace(ingest_document=ingest, delete_document_by_name=MagicMock())
    with pytest.raises(RuntimeError):
        P.ingest_pdf_streaming("x.pdf", "x.pdf", ms)
    ms.delete_document_by_name.assert_called_once_with("x.pdf")


# ── M3: a failed ingest is DECLARED failed ──

@pytest.mark.parametrize("text,status", [
    ("Embedding Error: Expected IDs to be unique", OutcomeStatus.FAILED),
    ("Web Error: timed out", OutcomeStatus.FAILED),
    ("Disk Error: no such file", OutcomeStatus.FAILED),
    ("Error: Extracted text is empty.", OutcomeStatus.FAILED),          # the shared classifier's call
    ("Error fetching https://x after 3 retries: timeout", OutcomeStatus.FAILED),
    ("SUCCESS (PARTIAL): Ingested the first 5 MB of 'a.txt'", OutcomeStatus.PARTIAL),
    ("SUCCESS: Ingested 'a.txt'.", OutcomeStatus.OK),
])
def test_ingest_results_declare_their_status(text, status):
    from ghost_agent.tools.memory import _declare_ingest
    assert _declare_ingest(text).status is status


async def test_the_tool_returns_the_declared_status(monkeypatch):
    import ghost_agent.tools.memory as M

    async def gain(*a, **k):
        return "Embedding Error: boom"
    monkeypatch.setattr(M, "tool_gain_knowledge", gain)
    out = await M.tool_knowledge_base(action="ingest_document", filename="a.txt",
                                      sandbox_dir=None, memory_system=MagicMock())
    assert out.status is OutcomeStatus.FAILED


# ── M4: breadcrumbs follow the document's own TOC order ──

def test_a_subsection_is_never_nested_under_the_next_section():
    from ghost_agent.memory.pdf_ingest import normalise_toc
    toc = [[2, "5.8 Modifying Tables", 120], [3, "5.8.8 Renaming a Table", 125],
           [2, "5.9 Privileges", 125]]
    assert [t for _, t, _ in normalise_toc(toc)] == [
        "5.8 Modifying Tables", "5.8.8 Renaming a Table", "5.9 Privileges"]


# ── minors: one name per file; the skip and truncation messages are honest ──

async def test_dot_slash_names_the_same_document_and_the_skip_says_how_to_refresh(tmp_path):
    import ghost_agent.tools.memory as M
    ms = MagicMock()
    ms.get_library.return_value = ["notes.txt"]
    out = await M.tool_gain_knowledge("./notes.txt", tmp_path, ms)
    assert "Skipped: 'notes.txt'" in out and "NOT 'all'" in out


async def test_a_truncated_ingest_says_it_is_partial(tmp_path, monkeypatch):
    import ghost_agent.tools.memory as M
    monkeypatch.setattr(M, "semantic_split_text", lambda text, *a, **k: ["one chunk"])
    (tmp_path / "big.txt").write_text("word " * 1_050_000)          # > 5 MB of text
    ms = MagicMock()
    ms.get_library.return_value = []
    ms.ingest_document.return_value = (True, "ok")
    out = await M.tool_gain_knowledge("big.txt", tmp_path, ms)
    assert "PARTIAL" in out and "first 5 MB" in out



def test_a_containment_refusal_is_never_ok():
    from ghost_agent.tools.memory import _declare_ingest
    assert _declare_ingest("Security Error: Path '../x' escapes the sandbox").status is not OutcomeStatus.OK


def test_a_second_concurrent_ingest_of_one_pdf_is_refused_and_rolls_nothing_back(monkeypatch):
    import ghost_agent.memory.pdf_ingest as P
    ms = SimpleNamespace(ingest_document=lambda *a, **k: (True, "ok"), delete_document_by_name=MagicMock())
    P._IN_FLIGHT.add("x.pdf")
    try:
        with pytest.raises(RuntimeError):
            P.ingest_pdf_streaming("x.pdf", "x.pdf", ms)
    finally:
        P._IN_FLIGHT.discard("x.pdf")
    ms.delete_document_by_name.assert_not_called()


@pytest.mark.parametrize("name,want", [("./notes.txt", "notes.txt"), (".//notes.txt", "notes.txt"),
                                       ("notes.txt", "notes.txt"), ("/abs/x.txt", "/abs/x.txt"),
                                       ("https://a.b/./c", "https://a.b/./c")])
def test_one_name_per_document_everywhere(name, want):
    from ghost_agent.tools.memory import _norm_doc_name
    assert _norm_doc_name(name) == want


async def test_a_query_by_dot_slash_name_finds_the_document(monkeypatch):
    import ghost_agent.tools.memory as M
    seen = []

    async def q(filename=None, question=None, memory_system=None):
        seen.append(filename)
        return "ok"
    monkeypatch.setattr(M, "tool_query_document", q)
    await M.tool_knowledge_base(action="query", filename="./notes.txt", question="x",
                                sandbox_dir=None, memory_system=MagicMock())
    assert seen == ["notes.txt"]


async def test_quoting_the_marker_is_not_truncation_but_a_cut_url_is(tmp_path, monkeypatch):
    import ghost_agent.tools.memory as M
    monkeypatch.setattr(M, "semantic_split_text", lambda text, *a, **k: ["one chunk"])
    ms = MagicMock()
    ms.get_library.return_value = []
    ms.ingest_document.return_value = (True, "ok")
    (tmp_path / "note.md").write_text("The marker reads [... INGEST TRUNCATED at 5 MB of extracted text ...] in logs.\nMore.")
    assert "PARTIAL" not in await M.tool_gain_knowledge("note.md", tmp_path, ms)

    async def fetch(url):
        return "page text " * 50 + "\n[... TRUNCATED at 5 MB ceiling ...]"
    monkeypatch.setattr(M, "helper_fetch_url_content", fetch)
    ms.get_library.return_value = []
    out = await M.tool_gain_knowledge("https://example.org/big", tmp_path, ms)
    assert "PARTIAL" in out
