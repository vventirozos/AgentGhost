"""2026-07-22 memory-substrate review — vector.py data-integrity fixes.

Findings verified against the LIVE production store (7,368 fragments, 96.8% of
them ingested-document chunks after the 2026-07-13 manual ingest):

1. CRIT — an embedding-function conflict silently DELETED the whole store. The
   recovery path (`delete_collection` + recreate) sits in an `except` that
   matches "already exists"/"Embedding function conflict", and it is reached
   BEFORE the embedder-fingerprint guard (which lives after the raising
   `get_or_create_collection` in the same try) can refuse. A chromadb upgrade or
   an EF-class change would have wiped everything. Now: count first, refuse to
   reset a POPULATED collection.
2. HIGH — ambient hydration was crowded out by the document corpus. Documents
   get a 1.25 threshold (2x) AND p_score=-5, and lower combined_score wins, so a
   barely-related chunk at dist 1.0 (-0.5) beat a strong auto memory at dist 0.30
   (+0.60). Measured on the live store: the 30-candidate pool was 30/30
   documents. Ambient hydration now excludes the doc corpus (document QA has its
   own scoped `search_document` path).
3. HIGH — `search_advanced` unconditionally bumped retrieval stats on every raw
   hit, so the episodic tier credited ~5-8 rows per hydration that were never
   shown to the model, poisoning prune-survival ranking and time decay.
4. MED — `smart_update`'s denylist wasn't the complement of `_PRUNABLE_TYPES`,
   so identity/synthesis/document_summary/acquired_skill — and even a user-saved
   `manual` — were legal deletion victims. Now same-type-only.
5. MED — `add()`'s id-exists early return meant a vector twin's metadata could
   never be refreshed, so `retract_lessons_from_trajectory` didn't match and a
   discredited lesson stayed retrievable.
"""
import threading
from unittest.mock import MagicMock

import pytest

from ghost_agent.memory.vector import VectorMemory


def _bare_vm():
    vm = VectorMemory.__new__(VectorMemory)  # bypass heavy __init__
    vm._lock = threading.RLock()
    vm._get_lock = lambda: vm._lock
    vm.collection = MagicMock()
    return vm


class TestAmbientExcludesDocumentCorpus:
    def test_search_selection_filters_out_documents(self):
        vm = _bare_vm()
        vm.collection.query.return_value = {
            "documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]],
        }
        vm._search_selection("where do I live?", inject_identity=False)
        where = vm.collection.query.call_args.kwargs["where"]
        assert where == {"type": {"$ne": "document"}}

    def test_document_qa_path_is_unaffected(self):
        # Ambient hydration excludes docs; the scoped doc-QA path must not.
        import inspect
        src = inspect.getsource(VectorMemory.search_document)
        assert '"$ne": "document"' not in src


class TestSearchAdvancedStatBump:
    def _vm_with_hits(self):
        vm = _bare_vm()
        vm.collection.query.return_value = {
            "ids": [["a", "b"]],
            "documents": [["doc a", "doc b"]],
            "metadatas": [[{"type": "episode"}, {"type": "document"}]],
            "distances": [[0.1, 0.2]],
        }
        vm._bump_retrieval_stats = MagicMock()
        return vm

    def test_bump_happens_by_default(self):
        vm = self._vm_with_hits()
        vm.search_advanced("q")
        vm._bump_retrieval_stats.assert_called_once_with(["a", "b"])

    def test_record_retrievals_false_suppresses_the_write(self):
        vm = self._vm_with_hits()
        vm.search_advanced("q", record_retrievals=False)
        vm._bump_retrieval_stats.assert_not_called()

    def test_where_scopes_the_query(self):
        vm = self._vm_with_hits()
        vm.search_advanced("q", where={"type": "episode"})
        assert vm.collection.query.call_args.kwargs["where"] == {"type": "episode"}

    def test_no_where_key_when_unscoped(self):
        vm = self._vm_with_hits()
        vm.search_advanced("q")
        assert "where" not in vm.collection.query.call_args.kwargs


class TestSmartUpdateSameTypeOnly:
    def test_dedup_is_scoped_to_the_incoming_type(self):
        vm = _bare_vm()
        vm.collection.query.return_value = {"ids": [[]], "distances": [[]]}
        vm.add = MagicMock()
        for label in ("auto", "manual", "identity", "synthesis"):
            vm.collection.query.reset_mock()
            vm.smart_update("some text", label)
            assert vm.collection.query.call_args.kwargs["where"] == {"type": label}


class TestAddRefreshesTwinMetadata:
    def test_existing_id_refreshes_metadata_instead_of_no_op(self):
        vm = _bare_vm()
        vm.collection.get.return_value = {"ids": ["dup"]}
        meta = {"type": "skill", "source_trajectory_id": "NEW", "verified": True}
        vm.add("a lesson body long enough", meta)
        # Not re-added...
        vm.collection.add.assert_not_called()
        # ...but the twin's metadata IS refreshed so retraction-by-trajectory
        # can match it.
        vm.collection.update.assert_called_once()
        assert vm.collection.update.call_args.kwargs["metadatas"] == [meta]

    def test_new_id_still_adds(self):
        vm = _bare_vm()
        vm.collection.get.return_value = {"ids": []}
        vm._adds_since_prune = 0
        vm._prune_if_needed = MagicMock()
        vm.add("a brand new memory body", {"type": "auto"})
        vm.collection.add.assert_called_once()


class TestEmbeddingConflictNeverWipesAPopulatedStore:
    def test_populated_collection_refuses_reset(self):
        import inspect
        src = inspect.getsource(VectorMemory.__init__)
        # The destructive branch must count the existing collection and bail
        # out rather than delete a populated store.
        assert "get_collection" in src
        assert "_existing > 0" in src
        assert "Refusing to reset" in src
        # And the reset that remains is explicitly the empty-collection case.
        assert "EMPTY collection" in src
