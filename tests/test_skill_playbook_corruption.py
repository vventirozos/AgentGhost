"""Regression: ``_load_playbook`` must NOT silently mask a corrupt
JSON file as an empty playbook — that path lets the next
``learn_lesson`` overwrite every prior lesson with a single new entry.

Trace shape:
1. ``skills_playbook.json`` becomes corrupt (partial write from an
   external process, transient I/O blip during disk pressure, etc.).
2. Agent fires the next ``learn_lesson``.
3. Old behaviour: ``_load_playbook`` does ``except Exception: return []``,
   so ``learn_lesson`` computes ``playbook = [new_lesson] + []`` and
   atomically saves a single-element list. **Every prior lesson is gone.**
   Both the JSON playbook (canonical) AND any later vector retrievals
   that fall through to JSON snapshots are lost.

Fix: distinguish the cases.
- Missing file → ``[]`` (first run, expected).
- Empty file → ``[]`` (first run, half-initialised, expected).
- Corrupt JSON → rename to ``skills_playbook.json.corrupt-<ts>`` to
  preserve the bytes for human recovery, THEN return ``[]``.
- OSError (disk full, permission) → re-raise. Treating these as ``[]``
  produces the same data-loss path on a perfectly-readable file the
  OS just refused to read once.
"""
import json
import re
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ghost_agent.memory.skills import SkillMemory


def _make_skill_memory(tmp_path):
    sm = SkillMemory.__new__(SkillMemory)
    sm.file_path = tmp_path / "skills_playbook.json"
    sm._lock = MagicMock()
    sm._lock.__enter__ = MagicMock(return_value=None)
    sm._lock.__exit__ = MagicMock(return_value=None)
    return sm


def test_missing_file_returns_empty_list(tmp_path):
    sm = _make_skill_memory(tmp_path)
    assert not sm.file_path.exists()
    assert sm._load_playbook() == []


def test_empty_file_returns_empty_list(tmp_path):
    sm = _make_skill_memory(tmp_path)
    sm.file_path.write_text("")
    assert sm._load_playbook() == []


def test_valid_playbook_returns_data(tmp_path):
    sm = _make_skill_memory(tmp_path)
    payload = [{"task": "x", "mistake": "y", "solution": "z"}]
    sm.file_path.write_text(json.dumps(payload))
    assert sm._load_playbook() == payload


def test_non_list_json_returns_empty_list(tmp_path):
    """Defensive: an object at the top level (rather than a list)
    falls back to []. Pre-existing behavior, pinned here so the
    refactor doesn't change it."""
    sm = _make_skill_memory(tmp_path)
    sm.file_path.write_text(json.dumps({"not": "a list"}))
    assert sm._load_playbook() == []


def test_corrupt_json_renames_file_and_returns_empty(tmp_path):
    """The critical fix: corrupt JSON must NOT silently look like
    an empty playbook to the caller. The corrupt bytes are moved
    aside so a human can recover, and the next save creates a
    fresh file rather than wiping the recovered data."""
    sm = _make_skill_memory(tmp_path)
    corrupt_bytes = '{"task": "partial-write-truncated", "mistake": '
    sm.file_path.write_text(corrupt_bytes)

    result = sm._load_playbook()
    assert result == []

    # The corrupt file must be moved aside, not deleted, not left in place.
    assert not sm.file_path.exists(), (
        "Corrupt file should have been renamed, not left at the original path."
    )
    # A timestamped sidecar must exist.
    sidecars = list(tmp_path.glob("skills_playbook.json.corrupt-*"))
    assert len(sidecars) == 1, (
        f"Expected exactly one .corrupt-<ts> sidecar; found {sidecars}"
    )
    assert sidecars[0].read_text() == corrupt_bytes, (
        "The corrupt file's bytes must be preserved verbatim in the sidecar."
    )
    # The sidecar timestamp shape must be ISO-ish.
    assert re.search(r"\.corrupt-\d{8}T\d{6}$", sidecars[0].name)


def test_subsequent_learn_lesson_does_not_wipe_corrupt_sidecar(tmp_path):
    """End-to-end: after a corrupt-file recovery, a learn_lesson
    call creates a fresh playbook with the new lesson — and the
    .corrupt-* sidecar is still on disk for human recovery."""
    sm = _make_skill_memory(tmp_path)
    sm.file_path.write_text('{"task": "broken')  # corrupt

    # First load triggers the rename.
    assert sm._load_playbook() == []
    sidecars = list(tmp_path.glob("skills_playbook.json.corrupt-*"))
    assert len(sidecars) == 1
    sidecar_path = sidecars[0]
    sidecar_bytes = sidecar_path.read_text()

    # Now simulate a save (what `learn_lesson` would do).
    sm._save_playbook_unlocked = SkillMemory._save_playbook_unlocked.__get__(sm)
    sm._save_playbook_unlocked([{"task": "new", "mistake": "m", "solution": "s"}])

    # Fresh playbook exists; sidecar still preserved.
    assert sm.file_path.exists()
    assert json.loads(sm.file_path.read_text())[0]["task"] == "new"
    assert sidecar_path.exists()
    assert sidecar_path.read_text() == sidecar_bytes


def test_oserror_propagates(tmp_path):
    """A real OSError (permission denied, disk-level fault) must NOT
    be silently swallowed as ``[]``. The old behavior masked it,
    leading the next ``learn_lesson`` to overwrite a perfectly-
    intact file with a 1-element playbook (since the load lied).
    """
    sm = _make_skill_memory(tmp_path)
    sm.file_path.write_text("[]")

    def _raise_perm(*a, **kw):
        raise PermissionError("simulated EACCES")

    with patch.object(Path, "read_text", _raise_perm):
        with pytest.raises(PermissionError):
            sm._load_playbook()
