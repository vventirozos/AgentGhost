"""Regression tests for two vector-memory data-loss defects.

1. ``smart_update`` template over-match — a close neighbour that merely
   shares a surface template ("User's favorite <X> is <Y>") but is a
   DIFFERENT fact must NOT be deleted; a genuine restatement of the SAME
   fact must still collapse into one entry.
2. ``correct_fragment`` id-collision loss — the replacement must persist
   even when its text hashes to an id already occupied in the collection
   (Chroma's ``add`` silently no-ops on an existing id).
"""

import hashlib
import threading

import pytest
from unittest.mock import MagicMock

from ghost_agent.memory.vector import VectorMemory, _subject_key


# ---------------------------------------------------------------------------
# _subject_key — the pure guard helper
# ---------------------------------------------------------------------------

def test_subject_key_normalises_template_subject():
    # Same attribute, different phrasing / possessive → same key.
    assert _subject_key("User's favorite color is blue") == "favorite color"
    assert _subject_key("My favorite color is actually blue") == "favorite color"
    # Different attribute → different key (this is the erasure the guard stops).
    assert _subject_key("User's favorite food is blue cheese") == "favorite food"
    assert (_subject_key("User's favorite color is blue")
            != _subject_key("User's favorite food is blue cheese"))


def test_subject_key_none_without_copula():
    # No "is/are/was/were" → no extractable key; caller falls back to distance.
    assert _subject_key("paraphrase two") is None
    assert _subject_key("") is None
    assert _subject_key(None) is None


# ---------------------------------------------------------------------------
# smart_update — template over-match guard
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vm():
    """A VectorMemory with a mocked collection + add (init bypassed)."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            "ghost_agent.memory.vector.VectorMemory.__init__",
            lambda self, a, b, c=None: None,
        )
        vm = VectorMemory(MagicMock(), "http://localhost:11434")
        vm.collection = MagicMock()
        vm.add = MagicMock()
        vm._lock = threading.RLock()
        return vm


def test_smart_update_keeps_distinct_same_template_fact(mock_vm):
    """"favorite food is blue cheese" must NOT erase "favorite color is blue"
    even though they embed under the 0.50 dedup threshold."""
    mock_vm.collection.query.return_value = {
        "ids": [["color_fact_id"]],
        "distances": [[0.30]],  # well under threshold — old code would delete
        "documents": [["User's favorite color is blue"]],
        "metadatas": [[{"type": "auto"}]],
    }
    mock_vm.smart_update("User's favorite food is blue cheese")
    # The distinct fact survives; only the new fact is added.
    mock_vm.collection.delete.assert_not_called()
    mock_vm.add.assert_called_once()


def test_smart_update_still_replaces_genuine_restatement(mock_vm):
    """A true restatement of the SAME fact (same subject key) still collapses
    into a single canonical entry — the intended dedup behaviour is intact."""
    mock_vm.collection.query.return_value = {
        "ids": [["color_fact_id"]],
        "distances": [[0.20]],
        "documents": [["User's favorite color is blue"]],
        "metadatas": [[{"type": "auto"}]],
    }
    mock_vm.smart_update("User's favorite color is actually blue")
    mock_vm.collection.delete.assert_called_once_with(ids=["color_fact_id"])
    mock_vm.add.assert_called_once()


def test_smart_update_paraphrase_without_template_still_dedups(mock_vm):
    """No copula on either side → no key to compare → distance-only decision,
    preserving dedup for genuine paraphrases that don't fit the template."""
    mock_vm.collection.query.return_value = {
        "ids": [["para_id"]],
        "distances": [[0.15]],
        "documents": [["paraphrase one"]],
        "metadatas": [[{"type": "auto"}]],
    }
    mock_vm.smart_update("paraphrase two")
    mock_vm.collection.delete.assert_called_once_with(ids=["para_id"])


# ---------------------------------------------------------------------------
# correct_fragment — id-collision loss
# ---------------------------------------------------------------------------

class _FaithfulChromaFake:
    """Reproduces the two Chroma semantics that create the loss bug:
    ``add()`` SILENTLY IGNORES an id that already exists, while ``upsert()``
    always writes. A store whose ``add`` overwrote would hide the defect."""

    def __init__(self):
        self.rows = {}  # id -> (document, metadata)

    def get(self, ids=None, include=None):
        if ids is not None:
            items = [(i, self.rows[i][0], self.rows[i][1])
                     for i in ids if i in self.rows]
        else:
            items = [(i, d, m) for i, (d, m) in self.rows.items()]
        return {"ids": [x[0] for x in items],
                "documents": [x[1] for x in items],
                "metadatas": [x[2] for x in items]}

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)

    def add(self, documents, metadatas, ids):
        if ids[0] in self.rows:  # Chroma no-ops on an existing id
            return
        self.rows[ids[0]] = (documents[0], metadatas[0])

    def upsert(self, documents, metadatas, ids):
        self.rows[ids[0]] = (documents[0], metadatas[0])


def _vm_with(collection):
    vm = object.__new__(VectorMemory)  # bypass heavy __init__
    vm.collection = collection
    return vm


def test_fake_models_chroma_add_noop():
    """Sanity: the fake's add() must ignore an existing id, or the collision
    test below would pass even against the buggy code."""
    col = _FaithfulChromaFake()
    col.add(documents=["first"], metadatas=[{}], ids=["dup"])
    col.add(documents=["second"], metadatas=[{}], ids=["dup"])
    assert col.rows["dup"][0] == "first"  # second add was a no-op


def test_correct_fragment_persists_under_id_collision():
    """The replacement text hashes to an id already occupied by a DIFFERENT
    fragment. Under the old delete-then-add, the correction was lost (old
    deleted, add() no-op). upsert must land it."""
    col = _FaithfulChromaFake()
    old_text = "User wants a single-file chess game against the assistant."
    replacement = "User wants a turn-based chess game against the assistant."

    old_id = hashlib.md5(old_text.encode()).hexdigest()
    new_id = hashlib.md5(replacement.encode()).hexdigest()
    assert old_id != new_id

    # Seed the fragment being corrected...
    col.rows[old_id] = (old_text, {"type": "auto", "timestamp": "t0"})
    # ...and force a hash collision: a squatter already occupies new_id.
    col.rows[new_id] = ("unrelated squatter fragment", {"type": "manual"})

    vm = _vm_with(col)
    ok, detail = vm.correct_fragment(old_text, replacement)

    assert ok, detail
    assert old_id not in col.rows                      # old fragment removed
    assert col.rows[new_id][0] == replacement          # correction persisted
    assert col.rows[new_id][1]["type"] == "auto"       # original metadata kept


def test_correct_fragment_normal_path_still_works():
    """No collision → still a clean delete-then-write with metadata carried."""
    col = _FaithfulChromaFake()
    old_text = "User prefers the dark theme in the editor."
    replacement = "User prefers the light theme in the editor."
    old_id = hashlib.md5(old_text.encode()).hexdigest()
    new_id = hashlib.md5(replacement.encode()).hexdigest()
    col.rows[old_id] = (old_text, {"type": "manual", "timestamp": "t1"})

    vm = _vm_with(col)
    ok, _ = vm.correct_fragment(old_text, replacement)

    assert ok
    assert old_id not in col.rows
    assert col.rows[new_id][0] == replacement
    assert col.rows[new_id][1]["type"] == "manual"
