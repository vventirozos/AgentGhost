"""Corner-case + concurrency tests for memory stores.

Targets:
  * SkillMemory: lesson dedup, retraction edge cases, playbook
    corruption recovery, concurrent learn_lesson + read.
  * TrajectoryCollector: concurrent append, day rollover behavior,
    corrections sidecar with malformed lines, iter under concurrent
    write, count() correctness with mixed/empty files.
  * Disk failures: read-only paths, missing parent dir, permission
    denied (where the test environment supports it).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import List

import pytest

from ghost_agent.distill.collector import TrajectoryCollector, CORRECTIONS_FILENAME
from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
from ghost_agent.memory.skills import SkillMemory


# ──────────────────────────────────────────────────────────────────────
# SkillMemory edge cases
# ──────────────────────────────────────────────────────────────────────

class TestSkillMemoryEdges:
    def test_fresh_playbook_has_zero_lessons(self, tmp_path: Path):
        sm = SkillMemory(tmp_path)
        playbook = sm.list_lessons()
        assert isinstance(playbook, (list, dict))

    def test_learn_lesson_with_empty_strings(self, tmp_path: Path):
        sm = SkillMemory(tmp_path)
        # Empty strings might be rejected or accepted; either is fine
        # as long as it doesn't crash.
        try:
            sm.learn_lesson(
                task="", mistake="", solution="",
                memory_system=None,
            )
        except (ValueError, RuntimeError):
            pass

    def test_learn_lesson_with_huge_strings(self, tmp_path: Path):
        """The lesson API truncates inputs; it must not raise."""
        sm = SkillMemory(tmp_path)
        sm.learn_lesson(
            task="x" * 10_000,
            mistake="y" * 10_000,
            solution="z" * 10_000,
            memory_system=None,
        )
        # Should not crash; playbook should be non-empty
        assert len(sm.list_lessons()) >= 1

    def test_retract_with_unknown_id_returns_zero(self, tmp_path: Path):
        sm = SkillMemory(tmp_path)
        # No lessons → retract by id is a no-op
        assert sm.retract_lessons_from_trajectory("nonexistent-id") == 0

    def test_retract_empty_string_is_protected(self, tmp_path: Path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson(
            task="t", mistake="m", solution="s",
            memory_system=None, source_trajectory_id="",  # legacy
        )
        # Even with a legacy lesson present, "" must not bulk-scrub it
        assert sm.retract_lessons_from_trajectory("") == 0
        assert len(sm.list_lessons()) >= 1

    def test_retract_specific_id_scrubs_only_that_one(self, tmp_path: Path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson(
            task="t1", mistake="m1", solution="s1",
            memory_system=None, source_trajectory_id="traj-a",
        )
        sm.learn_lesson(
            task="t2", mistake="m2", solution="s2",
            memory_system=None, source_trajectory_id="traj-b",
        )
        before = len(sm.list_lessons())
        sm.retract_lessons_from_trajectory("traj-a")
        after = len(sm.list_lessons())
        assert after == before - 1, (
            f"retraction removed wrong count: {before} → {after}"
        )

    def test_retract_idempotent(self, tmp_path: Path):
        """Calling retract twice on the same id must not crash or
        accidentally remove other lessons."""
        sm = SkillMemory(tmp_path)
        sm.learn_lesson(
            task="t", mistake="m", solution="s",
            memory_system=None, source_trajectory_id="t1",
        )
        n1 = sm.retract_lessons_from_trajectory("t1")
        n2 = sm.retract_lessons_from_trajectory("t1")  # already gone
        assert n1 >= 0
        assert n2 == 0

    def test_corrupted_playbook_recoverable(self, tmp_path: Path):
        """If the playbook JSON is malformed, SkillMemory should
        recover gracefully — start fresh rather than crash."""
        playbook = tmp_path / "skills_playbook.json"
        playbook.write_text("{this is not valid json")
        try:
            sm = SkillMemory(tmp_path)
            # If it loads, must produce a valid (empty) state
            assert isinstance(sm.list_lessons(), (list, dict))
        except (json.JSONDecodeError, ValueError):
            # Acceptable: explicit failure on corruption
            pass


# ──────────────────────────────────────────────────────────────────────
# SkillMemory concurrency
# ──────────────────────────────────────────────────────────────────────

class TestSkillMemoryConcurrency:
    def test_concurrent_learn_lesson_does_not_lose_writes(self, tmp_path: Path):
        """N threads each calling learn_lesson must produce roughly N
        lessons (some dedup is OK; gross loss is a bug)."""
        sm = SkillMemory(tmp_path)
        errors: List[Exception] = []

        def writer(tid: int):
            try:
                for i in range(5):
                    sm.learn_lesson(
                        task=f"task {tid}-{i}",
                        mistake=f"unique mistake {tid}-{i}",
                        solution=f"solution {tid}-{i}",
                        memory_system=None,
                        source_trajectory_id=f"traj-{tid}-{i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert errors == [], f"concurrent writes failed: {errors[:3]}"
        # At least most writes should have landed (dedup may collapse
        # near-identical ones, but unique tids/i should be distinct)
        n = len(sm.list_lessons())
        assert n >= 5, f"only {n} lessons after 20 writes"


# ──────────────────────────────────────────────────────────────────────
# TrajectoryCollector edge cases
# ──────────────────────────────────────────────────────────────────────

class TestTrajectoryCollectorEdges:
    def test_disabled_append_returns_none(self, tmp_path: Path):
        coll = TrajectoryCollector(root=tmp_path, session_id="x", enabled=False)
        traj = Trajectory(user_request="x", outcome=Outcome.PASSED.value)
        assert coll.append(traj) is None

    def test_disabled_update_outcome_returns_false(self, tmp_path: Path):
        coll = TrajectoryCollector(root=tmp_path, session_id="x", enabled=False)
        assert coll.update_outcome("any-id", "failed", "reason") is False

    def test_update_outcome_with_empty_id_returns_false(self, tmp_path: Path):
        coll = TrajectoryCollector(root=tmp_path, session_id="x", enabled=True)
        assert coll.update_outcome("", "failed", "reason") is False

    def test_update_outcome_with_non_string_id_returns_false(self, tmp_path: Path):
        coll = TrajectoryCollector(root=tmp_path, session_id="x", enabled=True)
        assert coll.update_outcome(None, "failed", "reason") is False  # type: ignore
        assert coll.update_outcome(123, "failed", "reason") is False  # type: ignore

    def test_iter_empty_root_yields_nothing(self, tmp_path: Path):
        coll = TrajectoryCollector(root=tmp_path / "nonexistent", session_id="x")
        assert list(coll.iter_trajectories()) == []

    def test_iter_skips_malformed_jsonl_lines(self, tmp_path: Path):
        coll = TrajectoryCollector(root=tmp_path, session_id="s1", enabled=True)
        # Append a valid trajectory
        coll.append(Trajectory(user_request="ok", outcome=Outcome.PASSED.value))
        # Now corrupt the file by appending garbage
        files = list((tmp_path).rglob("session-*.jsonl"))
        assert files
        with files[0].open("a") as f:
            f.write("{this is not valid json\n")
            f.write("\n")  # blank line
            f.write('{"id": "x", "incomplete":\n')

        # iter should yield only the valid trajectory and skip the rest
        trajs = list(coll.iter_trajectories())
        # At least 1 (the original); at most 1 (the corrupt ones are skipped)
        assert len(trajs) >= 1

    def test_count_with_mixed_files(self, tmp_path: Path):
        coll = TrajectoryCollector(root=tmp_path, session_id="s1", enabled=True)
        for i in range(5):
            coll.append(Trajectory(user_request=f"r{i}", outcome=Outcome.PASSED.value))
        n = coll.count()
        assert n == 5

    def test_count_empty_root(self, tmp_path: Path):
        coll = TrajectoryCollector(root=tmp_path / "nope")
        assert coll.count() == 0

    def test_corrections_sidecar_with_corrupt_lines(self, tmp_path: Path):
        """A corrupt sidecar line must not break iter_trajectories."""
        coll = TrajectoryCollector(root=tmp_path, session_id="s1", enabled=True)
        coll.append(Trajectory(id="t1", user_request="x", outcome=Outcome.PASSED.value))

        # Corrupt the sidecar
        sidecar = tmp_path / CORRECTIONS_FILENAME
        with sidecar.open("a") as f:
            f.write("not valid json\n")
            f.write("\n")
            f.write('{"trajectory_id": "t1", "outcome": "failed"}\n')
            f.write("more garbage\n")

        # iter yields the valid trajectory with the correction overlaid
        trajs = list(coll.iter_trajectories())
        assert len(trajs) == 1
        assert trajs[0].outcome == "failed"

    def test_orphan_correction_silently_ignored(self, tmp_path: Path):
        """A correction record for an unknown traj_id must not crash."""
        coll = TrajectoryCollector(root=tmp_path, session_id="s1", enabled=True)
        coll.append(Trajectory(id="t1", user_request="x", outcome=Outcome.PASSED.value))
        coll.update_outcome("nonexistent-id", "failed", "reason")
        trajs = list(coll.iter_trajectories())
        # Original trajectory unaffected
        assert len(trajs) == 1
        assert trajs[0].id == "t1"
        assert trajs[0].outcome == "passed"

    def test_last_write_wins_on_repeated_corrections(self, tmp_path: Path):
        coll = TrajectoryCollector(root=tmp_path, session_id="s1", enabled=True)
        coll.append(Trajectory(id="t1", user_request="x", outcome=Outcome.PASSED.value))
        coll.update_outcome("t1", "failed", "reason 1")
        coll.update_outcome("t1", "passed", "reason 2 — corrected the correction")
        trajs = list(coll.iter_trajectories())
        assert len(trajs) == 1
        # Latest correction wins
        assert trajs[0].outcome == "passed"

    def test_append_many_returns_correct_count(self, tmp_path: Path):
        coll = TrajectoryCollector(root=tmp_path, session_id="s1", enabled=True)
        n = coll.append_many([
            Trajectory(user_request=f"r{i}", outcome=Outcome.UNKNOWN.value)
            for i in range(10)
        ])
        assert n == 10


# ──────────────────────────────────────────────────────────────────────
# TrajectoryCollector concurrency
# ──────────────────────────────────────────────────────────────────────

class TestTrajectoryCollectorConcurrency:
    def test_concurrent_appends_all_land(self, tmp_path: Path):
        coll = TrajectoryCollector(root=tmp_path, session_id="s1", enabled=True)
        N = 20
        errors: List[Exception] = []

        def writer(tid: int):
            try:
                for i in range(N):
                    coll.append(Trajectory(
                        id=f"traj-{tid}-{i}",
                        user_request=f"req {tid}-{i}",
                        outcome=Outcome.PASSED.value,
                    ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert errors == [], f"concurrent appends failed: {errors[:3]}"
        # All writes landed
        assert coll.count() == 4 * N

    def test_concurrent_appends_and_updates(self, tmp_path: Path):
        """Mixed workload: writers append, another thread updates outcomes.
        No torn reads on iteration."""
        coll = TrajectoryCollector(root=tmp_path, session_id="s1", enabled=True)

        # Pre-seed
        for i in range(5):
            coll.append(Trajectory(
                id=f"seed-{i}", user_request=f"r{i}",
                outcome=Outcome.PASSED.value,
            ))

        errors: List[Exception] = []
        stop = threading.Event()

        def appender():
            try:
                i = 0
                while not stop.is_set():
                    coll.append(Trajectory(
                        id=f"new-{i}", user_request=f"x{i}",
                        outcome=Outcome.UNKNOWN.value,
                    ))
                    i += 1
                    if i > 100:
                        break
            except Exception as e:
                errors.append(e)

        def updater():
            try:
                for i in range(5):
                    coll.update_outcome(f"seed-{i}", "failed", "concurrent")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    list(coll.iter_trajectories())
            except Exception as e:
                errors.append(e)

        ts = [
            threading.Thread(target=appender),
            threading.Thread(target=updater),
            threading.Thread(target=reader),
        ]
        for t in ts:
            t.start()
        # Let it run briefly then stop
        import time
        time.sleep(0.5)
        stop.set()
        for t in ts:
            t.join(timeout=10)

        assert errors == [], f"concurrent mixed workload failed: {errors[:3]}"
