"""Tests for post-mortem deduplication in skills.py (#8).

Verifies that:
- _find_duplicate_lesson detects near-identical lessons via vector search
- _find_duplicate_lesson falls back to JSON substring matching
- learn_lesson merges duplicates instead of creating new entries
- learn_lesson creates new entries when no duplicate exists
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from ghost_agent.memory.skills import SkillMemory


@pytest.fixture
def skill_memory(tmp_path):
    return SkillMemory(tmp_path)


class TestFindDuplicateLesson:
    def test_detects_vector_duplicate(self, skill_memory):
        mock_memory = MagicMock()
        mock_memory.collection = MagicMock()
        mock_memory.collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["SITUATION: Parse CSV\nMISTAKE: wrong encoding\nSOLUTION: use latin-1"]],
            "distances": [[0.10]],  # Very similar
        }

        result = skill_memory._find_duplicate_lesson(
            "Parse CSV files", "wrong encoding", "use latin-1 encoding",
            memory_system=mock_memory
        )

        assert result is not None
        assert result["source"] == "vector"
        assert result["distance"] < 0.15

    def test_no_duplicate_when_distance_too_high(self, skill_memory):
        mock_memory = MagicMock()
        mock_memory.collection = MagicMock()
        mock_memory.collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["Something completely different"]],
            "distances": [[0.80]],  # Not similar
        }

        result = skill_memory._find_duplicate_lesson(
            "Parse CSV files", "wrong encoding", "use latin-1",
            memory_system=mock_memory
        )

        assert result is None

    def test_falls_back_to_json_matching(self, skill_memory):
        # Pre-populate playbook
        skill_memory.save_playbook([{
            "timestamp": "2025-01-01T00:00:00",
            "task": "Parse CSV files",
            "mistake": "wrong encoding",
            "solution": "use latin-1"
        }])

        result = skill_memory._find_duplicate_lesson(
            "Parse CSV files", "encoding error", "use utf-8",
            memory_system=None
        )

        assert result is not None
        assert result["source"] == "json"

    def test_no_duplicate_in_empty_playbook(self, skill_memory):
        result = skill_memory._find_duplicate_lesson(
            "Brand new task", "new mistake", "new solution",
            memory_system=None
        )
        assert result is None


class TestLearnLessonDedup:
    def test_merges_json_duplicate(self, skill_memory):
        # Pre-populate with existing lesson
        skill_memory.save_playbook([{
            "timestamp": "2025-01-01T00:00:00",
            "task": "Parse CSV files",
            "mistake": "encoding",
            "solution": "short fix",
            "frequency": 1
        }])

        skill_memory.learn_lesson(
            "Parse CSV files", "encoding", "longer better fix explanation"
        )

        playbook = json.loads(skill_memory.file_path.read_text())
        assert len(playbook) == 1  # No new entry added
        assert playbook[0]["frequency"] == 2
        # Longer solution should replace shorter one
        assert "longer better fix" in playbook[0]["solution"]

    def test_creates_new_for_unique_lesson(self, skill_memory):
        skill_memory.learn_lesson(
            "Totally new task", "new mistake", "new solution"
        )

        playbook = json.loads(skill_memory.file_path.read_text())
        assert len(playbook) == 1
        assert playbook[0]["task"] == "Totally new task"
        assert playbook[0]["frequency"] == 1

    def test_skips_vector_duplicate(self, skill_memory):
        mock_memory = MagicMock()
        mock_memory.collection = MagicMock()
        mock_memory.collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["SITUATION: Same task\nMISTAKE: same\nSOLUTION: same"]],
            "distances": [[0.05]],  # Near-identical
        }

        initial_playbook = [{
            "timestamp": "2025-01-01T00:00:00",
            "task": "Other task",
            "mistake": "other",
            "solution": "other",
            "frequency": 1
        }]
        skill_memory.save_playbook(initial_playbook)

        skill_memory.learn_lesson(
            "Same task", "same mistake", "same solution",
            memory_system=mock_memory
        )

        # Playbook should not have a new entry
        playbook = json.loads(skill_memory.file_path.read_text())
        assert len(playbook) == 1
        assert playbook[0]["task"] == "Other task"

    def test_respects_50_lesson_limit(self, skill_memory):
        # Fill with 50 lessons
        for i in range(50):
            skill_memory.learn_lesson(f"Task {i}", f"Mistake {i}", f"Solution {i}")

        playbook = json.loads(skill_memory.file_path.read_text())
        assert len(playbook) == 50
        # Most recent should be first
        assert playbook[0]["task"] == "Task 49"


class TestLearnLessonReturnContract:
    """learn_lesson's truthy/falsy return (2026-07-20): "written" for a new
    entry, "reinforced" for a dedup frequency bump, None on every drop path
    — so callers (e.g. the failure distiller) can tell a real write from a
    silent drop. Existing callers ignore the return value."""

    def _vector_hit(self, dist=0.05):
        mock_memory = MagicMock()
        mock_memory.collection = MagicMock()
        mock_memory.collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["SITUATION: Same task\nMISTAKE: same\nSOLUTION: same"]],
            "distances": [[dist]],
        }
        return mock_memory

    def test_new_lesson_returns_written(self, skill_memory):
        assert skill_memory.learn_lesson(
            "Totally new task", "new mistake", "new solution") == "written"

    def test_json_merge_returns_reinforced(self, skill_memory):
        skill_memory.learn_lesson("Parse CSV files", "encoding", "short fix")
        assert skill_memory.learn_lesson(
            "Parse CSV files", "encoding", "longer better fix") == "reinforced"

    def test_vector_dedup_with_json_twin_returns_reinforced(self, skill_memory):
        # Vector hit AND a playbook row with the same normalized trigger →
        # the vector branch bumps frequency.
        skill_memory.save_playbook([{
            "timestamp": "2025-01-01T00:00:00",
            "task": "Parse CSV files",
            "mistake": "wrong encoding",
            "solution": "use latin-1",
            "frequency": 1,
        }])
        result = skill_memory.learn_lesson(
            "Parse CSV files", "wrong encoding", "use latin-1 encoding",
            memory_system=self._vector_hit())
        assert result == "reinforced"
        playbook = json.loads(skill_memory.file_path.read_text())
        assert playbook[0]["frequency"] == 2

    def test_vector_dedup_without_json_twin_returns_none(self, skill_memory):
        # Same setup as test_skips_vector_duplicate: near-identical vector
        # hit, but no playbook row to bump → silent drop → falsy.
        skill_memory.save_playbook([{
            "timestamp": "2025-01-01T00:00:00",
            "task": "Other task",
            "mistake": "other",
            "solution": "other",
            "frequency": 1,
        }])
        result = skill_memory.learn_lesson(
            "Parse CSV files", "wrong encoding", "use latin-1 encoding",
            memory_system=self._vector_hit())
        assert result is None

    def test_quality_drop_returns_none(self, skill_memory):
        result = skill_memory.learn_lesson(
            "pref", "none", "The user prefers ripgrep over grep",
            source="dream")
        assert result is None
        assert skill_memory._load_playbook() == []

    def test_swallowed_exception_returns_none(self, skill_memory, monkeypatch):
        monkeypatch.setattr(
            skill_memory, "_find_duplicate_lesson",
            MagicMock(side_effect=RuntimeError("boom")))
        assert skill_memory.learn_lesson("Task", "mistake", "solution") is None


class TestMistakeLessWideThreshold:
    """2026-07-20: mistake-less rules ("When X, do Y" dream/reflection
    heuristics) dedup at 0.25 instead of 0.15 — measured on the live
    embedder, rewordings of the SAME rule land at 0.07-0.17 while distinct
    rules stay >= 0.29, so the 0.15 default let reworded re-saves through
    all night (10+ copies of two rules)."""

    def _mem_at_distance(self, dist):
        mock_memory = MagicMock()
        mock_memory.collection = MagicMock()
        mock_memory.collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["SITUATION: When querying weather, name the "
                           "location\nMISTAKE: none\nSOLUTION: name the "
                           "location explicitly"]],
            "distances": [[dist]],
        }
        return mock_memory

    def test_mistakeless_rule_deduped_in_wide_band(self, skill_memory):
        # dist 0.20: past the 0.15 default, inside the 0.25 rule cutoff.
        mem = self._mem_at_distance(0.20)
        skill_memory.learn_lesson(
            "When querying weather, ensure the query names the exact "
            "requested location to avoid ambiguity.",
            "none",
            "When querying weather, ensure the query names the exact "
            "requested location to avoid ambiguity.",
            memory_system=mem, source="dream",
        )
        playbook = json.loads(skill_memory.file_path.read_text())
        assert playbook == []  # deduped — no new entry written

    def test_mistakeful_lesson_keeps_tight_threshold(self, skill_memory):
        # Same 0.20 distance, but a REAL mistake → 0.15 cutoff applies and
        # the lesson is written as new.
        mem = self._mem_at_distance(0.20)
        skill_memory.learn_lesson(
            "Parse the weather API response",
            "read the wrong station's data",
            "Always match the station id against the requested city "
            "before reading values.",
            memory_system=mem,
        )
        playbook = json.loads(skill_memory.file_path.read_text())
        assert len(playbook) == 1

    def test_env_override_restores_old_behavior(self, skill_memory, monkeypatch):
        monkeypatch.setenv("GHOST_RULE_DEDUP_DIST", "0.15")
        mem = self._mem_at_distance(0.20)
        skill_memory.learn_lesson(
            "When querying weather, ensure the query names the exact "
            "requested location to avoid ambiguity.",
            "none",
            "When querying weather, ensure the query names the exact "
            "requested location to avoid ambiguity.",
            memory_system=mem, source="dream",
        )
        playbook = json.loads(skill_memory.file_path.read_text())
        assert len(playbook) == 1  # old threshold → treated as new
