"""Retraction scrubs the vector twin even when it lacks the trajectory id
(2026-09-15, §4HB).

Live: the false lesson from trajectory 97b402e8 had `source_trajectory_id`
set on its playbook row and "" on its vector twin. A retraction by
trajectory removed the JSON lesson and left the embedded copy retrievable —
the drift the code called "recoverable on a later rebuild" is a false lesson
still surfacing on recall until then. The JSON pass knows exactly which
TRIGGERS it removed; the vector pass now scrubs by trigger too.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import MagicMock

from ghost_agent.memory.skills import SkillMemory


def _mem_with_lesson(tmp_path, trigger="Find the Revolut notification screenshot"):
    mem = SkillMemory(tmp_path)
    mem.learn_lesson(task=trigger, mistake="m", solution="s")
    pb = mem._load_playbook()
    for r in pb:
        r["source_trajectory_id"] = "traj-97b4"
    mem.save_playbook(pb)
    return mem


def test_the_vector_twin_is_scrubbed_by_trigger(tmp_path):
    """FAILS IF: the vector pass deletes only by source_trajectory_id.

    The live world: a twin with an EMPTY id survives that delete.
    """
    mem = _mem_with_lesson(tmp_path)
    coll = MagicMock()
    ms = MagicMock(); ms.collection = coll
    assert mem.retract_lessons_from_trajectory("traj-97b4", memory_system=ms) == 1
    wheres = [c.kwargs.get("where") for c in coll.delete.call_args_list]
    assert {"source_trajectory_id": "traj-97b4"} in wheres
    by_trigger = [w for w in wheres if w and "trigger" in w]
    assert by_trigger, "no delete by trigger — the id-less twin survives"
    assert "Find the Revolut notification screenshot" in by_trigger[0]["trigger"]["$in"]


def test_no_trigger_delete_when_nothing_was_removed(tmp_path):
    """FAILS IF: an empty retraction issues a `$in: []` delete."""
    mem = SkillMemory(tmp_path)
    coll = MagicMock()
    ms = MagicMock(); ms.collection = coll
    assert mem.retract_lessons_from_trajectory("nobody", memory_system=ms) == 0
    wheres = [c.kwargs.get("where") for c in coll.delete.call_args_list]
    assert not [w for w in wheres if w and "trigger" in w]


def test_a_failed_json_pass_still_skips_the_vector_pass(tmp_path, monkeypatch):
    """FAILS IF: the trigger scrub runs on a JSON failure — the drift in the
    other direction, and the pre-existing contract."""
    mem = _mem_with_lesson(tmp_path)
    monkeypatch.setattr(mem, "_load_playbook", MagicMock(side_effect=RuntimeError("disk")))
    coll = MagicMock()
    ms = MagicMock(); ms.collection = coll
    mem.retract_lessons_from_trajectory("traj-97b4", memory_system=ms)
    coll.delete.assert_not_called()


def test_a_vector_error_does_not_undo_the_json_retraction(tmp_path):
    """FAILS IF: the new delete can raise past the best-effort guard."""
    mem = _mem_with_lesson(tmp_path)
    coll = MagicMock(); coll.delete.side_effect = RuntimeError("chroma down")
    ms = MagicMock(); ms.collection = coll
    assert mem.retract_lessons_from_trajectory("traj-97b4", memory_system=ms) == 1
    assert all(r.get("source_trajectory_id") != "traj-97b4" for r in mem._load_playbook())
