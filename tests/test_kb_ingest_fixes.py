"""Regression tests for the knowledge-base ingest pipeline audit.

Coverage map:
  KB-1  Library index: atomic write
  KB-2  Library index update inside the vector lock (no race)
  KB-3  delete_document_by_name / delete_by_query / unified_forget take the lock
  KB-4  PDF page cap + extracted-text cap on ingest
  KB-5  PyMuPDF doc handle is closed even on per-page error
  KB-6  Local text-file ingest is bounded
  KB-7  Unified forget disk sweep: recursive, prefers exact, sandbox-confined
  KB-8  Unified forget profile sweep: key match only, no destructive value match
  KB-9  Chunk IDs hash the FULL chunk (not just first 20 chars)
  KB-10 Concurrent ingest: dedup early-return inside the lock
  KB-11 reset_all error-trapping
"""
import asyncio
import hashlib
import json
import os
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.memory.vector import VectorMemory
from ghost_agent.tools.memory import (
    tool_gain_knowledge, tool_unified_forget, tool_knowledge_base,
)


def _make_vm_skeleton(tmp_path):
    """Build a VectorMemory whose collection is a MagicMock but with a real
    library_file + lock + _get_lock helper. Skips heavy chromadb init."""
    vm = VectorMemory.__new__(VectorMemory)
    vm.library_file = tmp_path / "library_index.json"
    vm.library_file.write_text("[]")
    vm.collection = MagicMock()
    vm.collection.get = MagicMock(return_value={"ids": []})
    vm.collection.add = MagicMock()
    vm.collection.upsert = MagicMock()
    vm.collection.delete = MagicMock()
    vm.collection.query = MagicMock(return_value={"ids": [], "distances": [], "documents": [], "metadatas": []})
    vm._lock = threading.RLock()
    return vm


# =====================================================================
# KB-1 — atomic library index write
# =====================================================================


def test_library_index_write_uses_atomic_replace(tmp_path, monkeypatch):
    vm = _make_vm_skeleton(tmp_path)
    seen_ops = []
    real_replace = os.replace

    def tracking_replace(src, dst):
        seen_ops.append(("replace", str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr("ghost_agent.memory.vector.os.replace", tracking_replace)
    vm._update_library_index("doc1.pdf", "add")
    assert seen_ops, "atomic os.replace was not used to commit the library index"
    assert seen_ops[0][2].endswith("library_index.json")


def test_library_index_recovers_from_corrupt_file(tmp_path, caplog):
    import logging
    vm = _make_vm_skeleton(tmp_path)
    vm.library_file.write_text("not valid json !!!{{{")
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        vm._update_library_index("doc.pdf", "add")
    # After the corrupt-recovery branch, the index contains exactly the new entry.
    assert json.loads(vm.library_file.read_text()) == ["doc.pdf"]
    assert any("corrupt" in r.message.lower() for r in caplog.records)


def test_library_index_concurrent_writers_no_lost_entries(tmp_path):
    """Hammer the index from multiple threads; every entry must survive."""
    vm = _make_vm_skeleton(tmp_path)

    def worker(prefix):
        for i in range(20):
            vm._update_library_index(f"{prefix}_doc_{i}.pdf", "add")

    threads = [threading.Thread(target=worker, args=(p,)) for p in ("A", "B", "C", "D")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = set(json.loads(vm.library_file.read_text()))
    expected = {f"{p}_doc_{i}.pdf" for p in "ABCD" for i in range(20)}
    assert final == expected, f"lost entries: {expected - final}"


# =====================================================================
# KB-2 — index update inside ingest's lock window
# =====================================================================


def test_ingest_document_holds_lock_through_index_update(tmp_path):
    vm = _make_vm_skeleton(tmp_path)
    # If the lock were released before _update_library_index, a concurrent
    # writer could slip in. We assert the index has ONE clean entry after
    # the call regardless.
    ok, _ = vm.ingest_document("only.pdf", ["chunk one", "chunk two"])
    assert ok
    assert json.loads(vm.library_file.read_text()) == ["only.pdf"]


# =====================================================================
# KB-3 — vector lock now covers all delete entry points
# =====================================================================


def test_delete_document_by_name_takes_the_lock(tmp_path):
    vm = _make_vm_skeleton(tmp_path)
    vm.library_file.write_text(json.dumps(["target.pdf"]))
    in_critical = [0]
    max_seen = [0]
    sentinel_lock = threading.Lock()

    def tracking_delete(*a, **k):
        with sentinel_lock:
            in_critical[0] += 1
            max_seen[0] = max(max_seen[0], in_critical[0])
        import time as _t; _t.sleep(0.005)
        with sentinel_lock:
            in_critical[0] -= 1
        return None

    vm.collection.delete = tracking_delete

    threads = [
        threading.Thread(target=lambda: vm.delete_document_by_name("target.pdf"))
        for _ in range(4)
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    # If the lock works, max concurrent entries == 1.
    assert max_seen[0] == 1


def test_delete_by_query_locks_around_query_and_delete(tmp_path):
    vm = _make_vm_skeleton(tmp_path)
    vm.collection.query = MagicMock(return_value={
        "ids": [["m1"]], "distances": [[0.1]],
        "documents": [["doc"]], "metadatas": [[{"type": "auto"}]],
    })
    in_critical = [0]
    max_seen = [0]
    sentinel_lock = threading.Lock()

    def tracking(*a, **k):
        with sentinel_lock:
            in_critical[0] += 1
            max_seen[0] = max(max_seen[0], in_critical[0])
        import time as _t; _t.sleep(0.003)
        with sentinel_lock:
            in_critical[0] -= 1
        return None

    vm.collection.delete = tracking

    threads = [threading.Thread(target=lambda: vm.delete_by_query("doc")) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert max_seen[0] == 1


# =====================================================================
# KB-4 / KB-5 / KB-6 — ingest size + page caps + close on error
# =====================================================================


@pytest.mark.asyncio
async def test_ingest_refuses_oversized_local_file(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    huge = sandbox / "huge.txt"
    # Sparse 200 MB file via seek
    with open(huge, "wb") as f:
        f.seek(200 * 1024 * 1024)
        f.write(b"x")
    mem = MagicMock()
    mem.get_library.return_value = []
    res = await tool_gain_knowledge("huge.txt", sandbox, mem)
    assert "Error" in res
    assert "MB" in res
    mem.ingest_document.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_truncates_text_at_5mb(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = sandbox / "long.txt"
    target.write_text("A" * (8 * 1024 * 1024))  # 8 MB of A's

    mem = MagicMock()
    mem.get_library.return_value = []
    mem.ingest_document.return_value = (True, "ok")
    res = await tool_gain_knowledge("long.txt", sandbox, mem)
    assert "SUCCESS" in res
    # The chunks passed to ingest_document represent the TRUNCATED text — not 8 MB.
    chunks_arg = mem.ingest_document.call_args.args[1]
    total_chars = sum(len(c) for c in chunks_arg)
    assert total_chars < 6 * 1024 * 1024, f"truncation cap missing, got {total_chars}"


# =====================================================================
# KB-7 — unified_forget disk sweep is recursive + safe-path bounded
# =====================================================================


@pytest.mark.asyncio
async def test_unified_forget_disk_sweeps_subdirectories(tmp_path):
    """Files in subdirectories must be reachable by the recursive sweep."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    sub = sandbox / "subdir"
    sub.mkdir()
    target_file = sub / "target_doc.txt"
    target_file.write_text("dummy")
    other_file = sandbox / "unrelated.txt"
    other_file.write_text("keep me")

    mem = MagicMock()
    mem.get_library.return_value = []
    mem.collection.query.return_value = {
        "ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]
    }
    mem._get_lock = MagicMock()
    mem._get_lock.return_value.__enter__ = MagicMock(return_value=None)
    mem._get_lock.return_value.__exit__ = MagicMock(return_value=False)

    report = await tool_unified_forget("target_doc", sandbox, mem)
    assert not target_file.exists(), "subdirectory file should have been deleted"
    assert other_file.exists(), "unrelated file must NOT have been touched"
    assert "Disk: Deleted" in report


@pytest.mark.asyncio
async def test_unified_forget_rejects_short_target(tmp_path):
    res = await tool_unified_forget("ab", tmp_path, MagicMock())
    assert "Error" in res
    assert "3 characters" in res or "specific" in res


@pytest.mark.asyncio
async def test_unified_forget_disk_prefers_exact_over_substring(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "report.txt").write_text("exact")
    (sandbox / "report_v2.txt").write_text("substring")
    (sandbox / "extra_report_notes.txt").write_text("substring")

    mem = MagicMock()
    mem.get_library.return_value = []
    mem.collection.query.return_value = {
        "ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]
    }
    mem._get_lock = MagicMock()
    mem._get_lock.return_value.__enter__ = MagicMock(return_value=None)
    mem._get_lock.return_value.__exit__ = MagicMock(return_value=False)

    await tool_unified_forget("report.txt", sandbox, mem)
    # Exact match wins — the substring matches stay alive.
    assert not (sandbox / "report.txt").exists()
    assert (sandbox / "report_v2.txt").exists()
    assert (sandbox / "extra_report_notes.txt").exists()


# =====================================================================
# KB-8 — profile sweep key matching (exact preferred over substring)
#
# NB: an earlier test here asserted the sweep was KEY-ONLY (a value
# mentioning the target must never be deleted). That invariant was
# intentionally superseded by the VALUE SWEEP in
# `tools/memory.py:tool_unified_forget` (`_value_mentions_target`), which
# removes a forgotten entity even when it lives in a VALUE (e.g. a pet
# stored in `assets.pets`). The stale key-only test was removed rather
# than reverting that feature. See memory:kb-forget-value-sweep-test-conflict.
# =====================================================================


@pytest.mark.asyncio
async def test_unified_forget_profile_sweep_prefers_exact_key_match(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    profile = MagicMock()
    profile.load.return_value = {
        "root": {"location": "Athens", "located_in": "EU"}
    }
    mem = MagicMock()
    mem.get_library.return_value = []
    mem.collection.query.return_value = {
        "ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]
    }
    mem._get_lock = MagicMock()
    mem._get_lock.return_value.__enter__ = MagicMock(return_value=None)
    mem._get_lock.return_value.__exit__ = MagicMock(return_value=False)

    await tool_unified_forget("location", sandbox, mem, profile_memory=profile)
    delete_calls = [c.args for c in profile.delete.call_args_list]
    # Exact match beats substring — only `location` should be deleted.
    assert ("root", "location") in delete_calls
    assert ("root", "located_in") not in delete_calls


# =====================================================================
# KB-9 — chunk IDs hash the FULL chunk (not first 20 chars)
# =====================================================================


def test_ingest_document_chunk_ids_use_full_chunk_hash(tmp_path):
    vm = _make_vm_skeleton(tmp_path)
    # Two chunks that share the first 20 characters but diverge afterwards.
    chunks = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox runs across the field at noon",
    ]
    vm.ingest_document("animals.txt", chunks)
    upsert_call = vm.collection.upsert.call_args
    ids = upsert_call.kwargs.get("ids") or upsert_call.args[2]
    # The two ids must be DIFFERENT — under the old "first 20 chars" hash
    # they would have collided.
    assert len(set(ids)) == 2

    # The id hashes the filename + the FULL ENRICHED chunk text. The per-batch
    # index was dropped (2026-07-13, streaming ingest): `enumerate()` restarts
    # at 0 on every batch, so an index-based id COLLIDED across the batches of
    # a streamed document. Hashing the text keeps ids globally unique and
    # stable on re-ingest, and makes identical chunks dedup by construction.
    enriched_0 = "[Source: animals.txt]\nthe quick brown fox jumps over the lazy dog"
    expected_id_0 = hashlib.md5(
        f"animals.txt|{enriched_0}".encode("utf-8")).hexdigest()
    assert ids[0] == expected_id_0


def test_ingest_document_batch_mode_ids_are_stable_across_batches(tmp_path):
    """Streaming ingest calls ingest_document once per BATCH. Ids must not
    collide across batches (the old per-batch `enumerate` index did), and
    batch chunks must NOT be re-enriched — pdf_ingest already stamps each
    chunk with its `[file] breadcrumb` header."""
    vm = _make_vm_skeleton(tmp_path)
    batch1 = ["[manual.pdf] Ch 1 › A\nalpha text", "[manual.pdf] Ch 1 › B\nbeta text"]
    batch2 = ["[manual.pdf] Ch 2 › C\ngamma text"]

    vm.ingest_document("manual.pdf", batch1, _batch=True)
    ids1 = vm.collection.upsert.call_args.kwargs["ids"]
    docs1 = vm.collection.upsert.call_args.kwargs["documents"]
    vm.ingest_document("manual.pdf", batch2, _batch=True)
    ids2 = vm.collection.upsert.call_args.kwargs["ids"]

    assert not set(ids1) & set(ids2), "batch ids collided across batches"
    # No double-enrichment: the chunk keeps its own breadcrumb header.
    assert docs1[0] == batch1[0]
    assert not docs1[0].startswith("[Source:")


# =====================================================================
# KB-10 — concurrent ingest: dedup early-return inside the lock
# =====================================================================


def test_ingest_document_skips_when_already_in_library(tmp_path):
    vm = _make_vm_skeleton(tmp_path)
    vm.library_file.write_text(json.dumps(["already.pdf"]))
    ok, msg = vm.ingest_document("already.pdf", ["any chunk"])
    assert ok
    assert "Skipped" in msg
    # The collection must NOT have been touched on a second ingest.
    vm.collection.upsert.assert_not_called()


# =====================================================================
# KB-11 — reset_all error trapping
# =====================================================================


@pytest.mark.asyncio
async def test_reset_all_handles_partial_batch_failure(tmp_path):
    mem = MagicMock()
    mem.collection.get.return_value = {"ids": [f"id_{i}" for i in range(1200)]}

    call_n = {"n": 0}
    def maybe_fail(*a, **k):
        call_n["n"] += 1
        if call_n["n"] == 2:  # second batch fails
            raise RuntimeError("DB lock contention")
        return None
    mem.collection.delete = maybe_fail

    lib = tmp_path / "library_index.json"
    lib.write_text(json.dumps(["a", "b"]))
    mem.library_file = lib

    res = await tool_knowledge_base("reset_all", sandbox_dir=tmp_path, memory_system=mem)
    assert "Partial" in res or "failed" in res.lower()
    # Successful batches still ran, the failure was logged but not raised.


@pytest.mark.asyncio
async def test_reset_all_clears_library_atomically(tmp_path):
    mem = MagicMock()
    mem.collection.get.return_value = {"ids": ["a", "b"]}
    mem.collection.delete = MagicMock(return_value=None)
    lib = tmp_path / "library_index.json"
    lib.write_text(json.dumps(["doc.pdf"]))
    mem.library_file = lib
    res = await tool_knowledge_base("reset_all", sandbox_dir=tmp_path, memory_system=mem)
    assert "Success" in res or "Wiped" in res
    assert lib.read_text() == "[]"
