"""Tests for M18: a removed/trimmed/pruned skill-lesson's embedded vector twin
is deleted (no orphan vectors drifting from the capped JSON playbook)."""

import tempfile
from pathlib import Path

import pytest

from ghost_agent.memory.skills import SkillMemory, _delete_lesson_twin


class _FakeColl:
    def __init__(self):
        self.deletes = []

    def delete(self, where=None):
        self.deletes.append(where)


class _FakeMem:
    def __init__(self):
        self.collection = _FakeColl()

    def add(self, text, meta):  # learn_lesson calls this; record nothing
        pass


class TestDeleteTwinHelper:
    def test_keys_by_trigger(self):
        m = _FakeMem()
        _delete_lesson_twin(m, {"trigger": "parse CSV with quoted commas", "solution": "s"})
        assert m.collection.deletes == [{"type": "skill", "trigger": "parse CSV with quoted commas"}]

    def test_falls_back_to_task_then_source(self):
        m = _FakeMem()
        _delete_lesson_twin(m, {"task": "fix encoding"})
        assert m.collection.deletes[-1] == {"type": "skill", "trigger": "fix encoding"}
        m2 = _FakeMem()
        _delete_lesson_twin(m2, {"source_trajectory_id": "tj-9"})  # no trigger/task
        assert m2.collection.deletes[-1] == {"source_trajectory_id": "tj-9"}

    def test_none_memory_is_noop(self):
        # Must not raise.
        _delete_lesson_twin(None, {"trigger": "x"})


class TestRemoveByTriggerDeletesTwin:
    def test_remove_threads_twin_delete(self):
        sm = SkillMemory(Path(tempfile.mkdtemp()))
        sm.save_playbook([{"trigger": "lesson A", "task": "lesson A", "solution": "s", "timestamp": "2026-01-01"}])
        m = _FakeMem()
        assert sm.remove_by_trigger("lesson A", memory_system=m) is True
        assert m.collection.deletes == [{"type": "skill", "trigger": "lesson A"}]

    def test_remove_without_memory_still_works(self):
        sm = SkillMemory(Path(tempfile.mkdtemp()))
        sm.save_playbook([{"trigger": "B", "task": "B", "solution": "s", "timestamp": "2026-01-01"}])
        assert sm.remove_by_trigger("B") is True  # no memory_system → no twin delete, no error
        assert sm.remove_by_trigger("nope") is False


class TestPruneDeletesTwins:
    def test_prune_deletes_twins_of_pruned(self):
        sm = SkillMemory(Path(tempfile.mkdtemp()))
        # 12 lessons, low-utility + retrieved enough + unverified → eligible to prune.
        pb = []
        for i in range(12):
            pb.append({
                "trigger": f"lesson {i}", "task": f"lesson {i}", "solution": "s",
                "timestamp": f"2026-01-{i+1:02d}", "retrievals": 9, "helpful_retrievals": 0,
                "verified": False, "frequency": 1,
            })
        sm.save_playbook(pb)
        m = _FakeMem()
        removed = sm.prune_low_utility(min_retrievals=5, max_drop_fraction=0.25, memory_system=m)
        # Every pruned lesson's twin should have been deleted (one delete per prune).
        assert removed >= 1
        assert len(m.collection.deletes) == removed
        assert all(d.get("type") == "skill" for d in m.collection.deletes)
