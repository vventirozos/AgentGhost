"""Tests for the memory subsystem audit fixes.

Each test focuses on a single fix from the audit so a regression points
straight at the broken behaviour rather than tangling unrelated paths.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from ghost_agent.memory.profile import ProfileMemory
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.memory.journal import MemoryJournal
from ghost_agent.memory.vector import VectorMemory
from ghost_agent.tools import memory as memory_tools


# ---------------------------------------------------------------------------
# Fix 1: vector.py smart_update threshold
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vector_memory():
    memory_dir = MagicMock()
    memory_dir.exists.return_value = True
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            "ghost_agent.memory.vector.VectorMemory.__init__",
            lambda self, a, b, c=None: None,
        )
        vm = VectorMemory(memory_dir, "http://localhost:11434")
        vm.collection = MagicMock()
        vm.add = MagicMock()
        # Provide the lock helper — a fresh instance bypassed __init__.
        import threading
        vm._lock = threading.RLock()
        return vm


def test_smart_update_threshold_relaxed_to_050(mock_vector_memory):
    """Distance 0.45 is now treated as a duplicate (was kept under 0.30)."""
    existing_id = "existing_abc"
    mock_vector_memory.collection.query.return_value = {
        "ids": [[existing_id]],
        "distances": [[0.45]],
        "documents": [["paraphrase one"]],
        "metadatas": [[{"timestamp": "old"}]],
    }
    mock_vector_memory.smart_update("paraphrase two")
    mock_vector_memory.collection.delete.assert_called_with(ids=[existing_id])
    mock_vector_memory.add.assert_called()


def test_smart_update_threshold_still_rejects_distant(mock_vector_memory):
    """A distance of 0.80 is unrelated; nothing should be deleted."""
    mock_vector_memory.collection.query.return_value = {
        "ids": [["existing_xyz"]],
        "distances": [[0.80]],
        "documents": [["unrelated text"]],
        "metadatas": [[{"timestamp": "old"}]],
    }
    mock_vector_memory.smart_update("totally different fact")
    mock_vector_memory.collection.delete.assert_not_called()
    mock_vector_memory.add.assert_called()


def test_smart_update_source_threshold_value():
    """Belt-and-braces: confirm the literal threshold in the source is 0.50."""
    src = Path(__file__).resolve().parents[1] / "src/ghost_agent/memory/vector.py"
    text = src.read_text()
    # Find the smart_update method body and confirm the < 0.50 literal lives in it.
    start = text.index("def smart_update(")
    # End of method = next "def " at the same indentation.
    rest = text[start:]
    end_rel = rest.index("\n    def ", 1)
    body = rest[:end_rel]
    assert "dist < 0.50" in body, "smart_update threshold must be 0.50"


# ---------------------------------------------------------------------------
# Fix 2: vector.py search() candidate pool widened to 30
# ---------------------------------------------------------------------------


def test_search_uses_wider_candidate_pool():
    """The chroma query inside `search()` must request 30 candidates."""
    src = Path(__file__).resolve().parents[1] / "src/ghost_agent/memory/vector.py"
    text = src.read_text()
    start = text.index("def search(")
    rest = text[start:]
    end_rel = rest.index("\n    def ", 1)
    body = rest[:end_rel]
    assert "n_results=30" in body, "search() must query 30 candidates"
    assert "n_results=10" not in body, "stale n_results=10 left in search()"


# ---------------------------------------------------------------------------
# Fix 3: profile.py merge semantics
# ---------------------------------------------------------------------------


@pytest.fixture
def profile(tmp_path):
    return ProfileMemory(tmp_path)


def test_profile_merge_keeps_both_values(profile):
    profile.update("interests", "language", "python")
    profile.update("interests", "language", "rust")
    data = profile.load()
    assert data["interests"]["language"] == ["python", "rust"]


def test_profile_merge_dedups_repeats(profile):
    profile.update("interests", "language", "python")
    profile.update("interests", "language", "rust")
    profile.update("interests", "language", "python")  # duplicate of head
    data = profile.load()
    # Should not introduce a third "python"
    assert data["interests"]["language"] == ["python", "rust"]


def test_profile_noop_on_identical_scalar(profile):
    # Use a fresh category/key so we aren't tripping over the default
    # "{name: User}" seed in __init__.
    profile.update("interests", "sport", "tennis")
    profile.update("interests", "sport", "tennis")
    data = profile.load()
    # Stays a scalar — no spurious list promotion on identical writes.
    assert data["interests"]["sport"] == "tennis"


def test_profile_get_context_string_handles_lists(profile):
    profile.update("interests", "language", "python")
    profile.update("interests", "language", "rust")
    s = profile.get_context_string()
    assert "language: python, rust" in s


# ---------------------------------------------------------------------------
# Fix 4: scratchpad TTL on get()
# ---------------------------------------------------------------------------


def test_scratchpad_get_returns_none_after_ttl_in_memory():
    sp = Scratchpad(max_entries=10, ttl=1)
    sp.set("ephemeral", "value")
    assert sp.get("ephemeral") == "value"
    time.sleep(1.2)
    assert sp.get("ephemeral") is None
    # And the cache has been pruned
    assert sp.count() == 0


def test_scratchpad_get_returns_none_after_ttl_persistent(tmp_path):
    db_path = tmp_path / "scratchpad.db"
    sp = Scratchpad(max_entries=10, persist_path=db_path, ttl=1)
    sp.set("ephemeral", "value")
    time.sleep(1.2)
    assert sp.get("ephemeral") is None


# ---------------------------------------------------------------------------
# Fix 5 / 6: journal.push_front and journal.drain
# ---------------------------------------------------------------------------


def test_journal_push_front_preserves_requeued_when_under_capacity(tmp_path):
    journal = MemoryJournal(tmp_path, max_capacity=5)
    journal.append("a", {"i": 1})
    journal.append("b", {"i": 2})
    journal.append("c", {"i": 3})
    requeued = [{"type": "r1", "data": {}}, {"type": "r2", "data": {}}]
    journal.push_front(requeued)
    loaded = journal.load()
    # All requeued items present at the head, all original items kept too.
    assert [e["type"] for e in loaded] == ["r1", "r2", "a", "b", "c"]


def test_journal_push_front_drops_tail_when_over_capacity(tmp_path):
    journal = MemoryJournal(tmp_path, max_capacity=4)
    journal.append("a", {"i": 1})
    journal.append("b", {"i": 2})
    journal.append("c", {"i": 3})
    requeued = [{"type": "r1", "data": {}}, {"type": "r2", "data": {}}]
    journal.push_front(requeued)
    loaded = journal.load()
    # Requeued items preserved; oldest tail dropped.
    assert [e["type"] for e in loaded] == ["r1", "r2", "a", "b"]
    assert all(e["type"] != "c" for e in loaded)


def test_journal_push_front_handles_oversized_items_list(tmp_path):
    journal = MemoryJournal(tmp_path, max_capacity=3)
    journal.append("orig", {"i": 0})
    requeued = [{"type": f"r{i}", "data": {}} for i in range(5)]
    journal.push_front(requeued)
    loaded = journal.load()
    # When items alone exceed capacity, keep the most-recent items only.
    assert len(loaded) == 3
    assert [e["type"] for e in loaded] == ["r2", "r3", "r4"]


def test_journal_drain_returns_and_clears(tmp_path):
    journal = MemoryJournal(tmp_path, max_capacity=10)
    journal.append("a", {"i": 1})
    journal.append("b", {"i": 2})
    drained = journal.drain()
    assert [e["type"] for e in drained] == ["a", "b"]
    # Persistence is updated atomically.
    assert journal.load() == []
    # Subsequent drain on an empty journal is harmless.
    assert journal.drain() == []


def test_journal_drain_persists_to_file(tmp_path):
    journal = MemoryJournal(tmp_path, max_capacity=10)
    journal.append("a", {"i": 1})
    journal.drain()
    on_disk = json.loads(journal.file_path.read_text())
    assert on_disk == []


# ---------------------------------------------------------------------------
# Fix 8: tools/memory.py invokes get_library as a function
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unified_forget_invokes_get_library(tmp_path):
    """tool_unified_forget must actually CALL memory_system.get_library()."""
    memory_system = MagicMock()
    memory_system.get_library = MagicMock(return_value=["doc1.txt"])
    # collection.query returns no ids so the semantic sweep is a no-op.
    memory_system.collection.query.return_value = {
        "ids": [[]],
        "distances": [[]],
        "documents": [[]],
        "metadatas": [[]],
    }
    # delete_document_by_name shouldn't crash even though stem won't match
    memory_system.delete_document_by_name = MagicMock()

    await memory_tools.tool_unified_forget(
        target="doc1",
        sandbox_dir=tmp_path,
        memory_system=memory_system,
    )

    memory_system.get_library.assert_called_once()
    # And because doc1 stem matches doc1.txt, the document deleter ran.
    memory_system.delete_document_by_name.assert_called_with("doc1.txt")
