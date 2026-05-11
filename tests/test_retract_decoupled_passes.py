"""Regression: ``retract_lessons_from_trajectory`` must clean up
stale vector entries even when the JSON playbook has zero matches,
AND must NOT run the vector scrub when the JSON pass itself failed
(that would create asymmetric drift instead of fixing it).

The original code gated the vector scrub on ``removed > 0``, which
meant prior-drift state (JSON cleaned, vector still has the entry)
never got reconciled by a subsequent retract call — the stale
vector entry kept costing a retrieval slot until the next REM
prune. Pre-investigation E3 framing also suggested running the
vector scrub *despite* a JSON failure; that's actually wrong —
``_save_playbook_unlocked`` is atomic via ``os.replace``, so a
JSON save failure leaves the original file intact. Vector-scrubbing
in that path would CREATE drift (JSON has lesson, vector
doesn't), so we explicitly DO NOT do it.

The fix:
- JSON pass success + matches: scrub vector (unchanged).
- JSON pass success + no matches: STILL scrub vector (drift cleanup).
- JSON pass failure: DO NOT scrub vector (preserve consistency).
"""
import pytest
from unittest.mock import MagicMock, patch

from ghost_agent.memory.skills import SkillMemory


def _make_skill_memory(tmp_path, initial_playbook=None):
    sm = SkillMemory.__new__(SkillMemory)
    sm.file_path = tmp_path / "skills_playbook.json"
    if initial_playbook is not None:
        import json as _j
        sm.file_path.write_text(_j.dumps(initial_playbook))
    return sm


def _memory_system_with_collection():
    ms = MagicMock()
    ms.collection = MagicMock()
    ms.collection.delete = MagicMock(return_value=None)
    return ms


def test_vector_scrub_runs_when_json_match_found(tmp_path):
    """Baseline: with a real JSON match, both passes run."""
    sm = _make_skill_memory(tmp_path, initial_playbook=[
        {"task": "x", "mistake": "y", "solution": "z",
         "source_trajectory_id": "tid-1"},
        {"task": "a", "mistake": "b", "solution": "c",
         "source_trajectory_id": "tid-2"},
    ])
    ms = _memory_system_with_collection()

    removed = sm.retract_lessons_from_trajectory("tid-1", memory_system=ms)
    assert removed == 1
    ms.collection.delete.assert_called_once_with(
        where={"source_trajectory_id": "tid-1"}
    )


def test_vector_scrub_runs_even_when_json_has_zero_matches(tmp_path):
    """The fix: prior-drift cleanup. JSON had no matches (already
    scrubbed previously), but the vector store may still have a
    stale entry — we should still try to scrub it."""
    sm = _make_skill_memory(tmp_path, initial_playbook=[
        {"task": "a", "mistake": "b", "solution": "c",
         "source_trajectory_id": "tid-other"},
    ])
    ms = _memory_system_with_collection()

    removed = sm.retract_lessons_from_trajectory("tid-stale", memory_system=ms)
    assert removed == 0  # JSON had no matches
    # Vector scrub MUST still have run (drift cleanup).
    ms.collection.delete.assert_called_once_with(
        where={"source_trajectory_id": "tid-stale"}
    )


def test_vector_scrub_skipped_when_json_pass_raised(tmp_path):
    """Symmetric to the drift-cleanup case: when JSON pass fails,
    JSON is unchanged (atomic save). Scrubbing only the vector
    would create the opposite drift. So we DON'T scrub vector
    in that path."""
    sm = _make_skill_memory(tmp_path)
    ms = _memory_system_with_collection()

    # Simulate a JSON load failure by raising from read_text.
    # _load_playbook re-raises OSError (per E1 fix), so the JSON
    # try/except in retract catches it and sets json_failed=True.
    with patch.object(type(sm.file_path), 'read_text',
                      lambda self: (_ for _ in ()).throw(PermissionError("EACCES"))):
        removed = sm.retract_lessons_from_trajectory("tid-x", memory_system=ms)

    assert removed == 0
    # Vector scrub MUST NOT have run (JSON pass failed).
    ms.collection.delete.assert_not_called()


def test_empty_trajectory_id_no_op_no_vector_call(tmp_path):
    """The legacy-protection sentinel: empty/non-string ids return 0
    without touching anything. Pin the contract."""
    sm = _make_skill_memory(tmp_path, initial_playbook=[])
    ms = _memory_system_with_collection()

    assert sm.retract_lessons_from_trajectory("", memory_system=ms) == 0
    assert sm.retract_lessons_from_trajectory(None, memory_system=ms) == 0
    ms.collection.delete.assert_not_called()


def test_no_memory_system_skips_vector_pass_cleanly(tmp_path):
    """When memory_system is None, the vector-scrub block is a no-op.
    The JSON pass should still run normally."""
    sm = _make_skill_memory(tmp_path, initial_playbook=[
        {"task": "x", "mistake": "y", "solution": "z",
         "source_trajectory_id": "tid-1"},
    ])
    removed = sm.retract_lessons_from_trajectory("tid-1", memory_system=None)
    assert removed == 1


def test_vector_scrub_failure_logged_and_swallowed(tmp_path, caplog):
    """When the vector pass itself raises, the JSON pass result
    must still be returned and the failure must be logged. JSON
    is canonical; a logged vector failure is acceptable."""
    import logging as _logging
    sm = _make_skill_memory(tmp_path, initial_playbook=[
        {"task": "x", "mistake": "y", "solution": "z",
         "source_trajectory_id": "tid-1"},
    ])
    ms = MagicMock()
    ms.collection = MagicMock()
    ms.collection.delete = MagicMock(side_effect=RuntimeError("chroma down"))

    with caplog.at_level(_logging.WARNING, logger="GhostAgent"):
        removed = sm.retract_lessons_from_trajectory("tid-1", memory_system=ms)
    assert removed == 1  # JSON pass succeeded
    assert any("vector pass failed" in r.message for r in caplog.records)
