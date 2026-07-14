"""Read-only memory façades actually block writes (bug-hunt 2026-07-14).

The old wrappers no-op'd GUESSED method names (`add_memory`, `delete_memory`,
`add_triplet`, `insert_fact`) that don't exist on the real stores, while the
REAL mutators (`add`, `ingest_document`, `delete_fragment`, `add_triplets`,
`delete_by_target`, `remove_by_trigger`, …) fell straight through
`__getattr__` and executed against the operator's live memory. Plus `.real`
and the chromadb `.collection`/`.client` handles were reachable, so a caller
could write around the façade entirely.

These tests pin the contract: every real mutator is a no-op, reads pass
through, and the raw writable handles are blocked. The mutator lists are
checked against the ACTUAL store method names so a store growing a new writer
that isn't blocked will surface here.
"""

import pytest
from unittest.mock import MagicMock

from ghost_agent.memory.readonly import (
    ReadOnlyVectorMemory, ReadOnlySkillMemory, ReadOnlyGraphMemory,
)


class TestVectorReadOnly:
    def test_real_mutators_are_noops(self):
        real = MagicMock()
        ro = ReadOnlyVectorMemory(real)
        for m in ("add", "smart_update", "ingest_document", "process_batch",
                  "bump_retrievals", "bump_helpful", "forget_episode",
                  "delete_document_by_name", "correct_fragment",
                  "delete_fragment", "delete_by_query"):
            assert getattr(ro, m)("x", {"y": 1}) is None, f"{m} not blocked"
        real.add.assert_not_called()
        real.ingest_document.assert_not_called()
        real.delete_fragment.assert_not_called()

    def test_reads_pass_through(self):
        real = MagicMock()
        real.search_advanced.return_value = [{"text": "hit"}]
        real.get_library.return_value = ["doc.pdf"]
        ro = ReadOnlyVectorMemory(real)
        assert ro.search_advanced("q") == [{"text": "hit"}]
        assert ro.get_library() == ["doc.pdf"]

    def test_search_does_not_record_retrievals(self):
        real = MagicMock()
        real.search.return_value = "result"
        ro = ReadOnlyVectorMemory(real)
        assert ro.search("q") == "result"
        # The stat-recording write must be forced off.
        assert real.search.call_args.kwargs.get("record_retrievals") is False

    def test_raw_handles_blocked(self):
        real = MagicMock()
        real.collection = MagicMock()
        real.client = MagicMock()
        ro = ReadOnlyVectorMemory(real)
        assert ro.real is None          # the writable store itself
        assert ro.collection is None    # chromadb handle
        assert ro.client is None
        assert ro.is_read_only is True

    def test_absent_store_degrades(self):
        ro = ReadOnlyVectorMemory(None)
        assert ro.add("x") is None        # mutator no-op works without a store
        assert ro.search_advanced is None  # read attr degrades to None


class TestSkillReadOnly:
    def test_real_mutators_are_noops(self):
        real = MagicMock()
        ro = ReadOnlySkillMemory(real)
        for m in ("learn_lesson", "save_playbook",
                  "retract_lessons_from_trajectory", "record_retrieval",
                  "record_helpful_retrieval", "credit_recent_retrievals",
                  "record_retrievals_bulk", "prune_low_utility",
                  "mark_verified", "remove_by_trigger"):
            assert getattr(ro, m)("x") is None, f"{m} not blocked"
        real.learn_lesson.assert_not_called()
        real.remove_by_trigger.assert_not_called()

    def test_reads_pass_through(self):
        real = MagicMock()
        real.get_playbook_items.return_value = [{"trigger": "t"}]
        ro = ReadOnlySkillMemory(real)
        assert ro.get_playbook_items("q") == [{"trigger": "t"}]

    def test_real_handle_blocked(self):
        ro = ReadOnlySkillMemory(MagicMock())
        assert ro.real is None


class TestGraphReadOnly:
    def test_real_mutators_are_noops(self):
        real = MagicMock()
        ro = ReadOnlyGraphMemory(real)
        for m in ("add_triplets", "delete_by_target", "wipe_all",
                  "prune_stale_edges", "execute_graph_compression", "bump",
                  "initialize_graph"):
            assert getattr(ro, m)([{"subject": "a"}]) is None, f"{m} not blocked"
        real.add_triplets.assert_not_called()
        real.delete_by_target.assert_not_called()
        real.wipe_all.assert_not_called()

    def test_reads_pass_through(self):
        real = MagicMock()
        real.get_neighborhood.return_value = ["a -> REL -> b"]
        ro = ReadOnlyGraphMemory(real)
        assert ro.get_neighborhood(["a"], 15) == ["a -> REL -> b"]

    def test_raw_handles_blocked(self):
        real = MagicMock()
        real.nx_graph = MagicMock()
        ro = ReadOnlyGraphMemory(real)
        assert ro.real is None
        assert ro.nx_graph is None


class TestMutatorListsMatchRealStores:
    """Guard: every mutating method the real store exposes is in the
    façade's block set. If a store grows a new writer, this fails loudly."""

    # Method-name prefixes/exact names that WRITE. Reads (search*/get*/
    # embed*/find*/list*/propose*/to_*) are allowed to pass through.
    _WRITE_HINTS = ("add", "delete", "remove", "insert", "update", "save",
                    "learn", "record", "bump", "prune", "wipe", "forget",
                    "correct", "ingest", "smart_update", "mark_", "retract",
                    "credit", "execute_graph", "process_batch", "initialize")

    def _mutating_methods(self, cls):
        out = set()
        for name in dir(cls):
            if name.startswith("_"):
                continue
            if any(name == h or name.startswith(h) for h in self._WRITE_HINTS):
                out.add(name)
        return out

    def test_vector_writers_all_blocked(self):
        from ghost_agent.memory.vector import VectorMemory
        writers = self._mutating_methods(VectorMemory)
        blocked = ReadOnlyVectorMemory._MUTATORS | {"search"}  # search forced RO
        missing = writers - blocked
        assert not missing, f"unblocked VectorMemory writers: {missing}"

    def test_graph_writers_all_blocked(self):
        from ghost_agent.memory.graph import GraphMemory
        writers = self._mutating_methods(GraphMemory)
        missing = writers - ReadOnlyGraphMemory._MUTATORS
        assert not missing, f"unblocked GraphMemory writers: {missing}"

    def test_skill_writers_all_blocked(self):
        from ghost_agent.memory.skills import SkillMemory
        writers = self._mutating_methods(SkillMemory)
        missing = writers - ReadOnlySkillMemory._MUTATORS
        assert not missing, f"unblocked SkillMemory writers: {missing}"
