"""Tests for cross-episode pattern detection (#15).

Verifies that:
- detect_tool_patterns finds recurring tool sequences
- Patterns need 3+ occurrences to be reported
- Patterns include example references
"""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock
from ghost_agent.core.dream import detect_tool_patterns
from ghost_agent.memory.skills import SkillMemory


@pytest.fixture
def skill_memory_with_patterns(tmp_path):
    sm = SkillMemory(tmp_path)
    # Create lessons with recurring tool patterns
    lessons = []
    for i in range(4):
        lessons.append({
            "timestamp": f"2025-01-0{i+1}T00:00:00",
            "task": f"Data analysis task {i}",
            "mistake": "forgot to check file",
            "solution": f"Use file_system to read, then execute Python script to process, then recall from memory",
            "frequency": 1,
        })
    # Add a different pattern
    for i in range(3):
        lessons.append({
            "timestamp": f"2025-02-0{i+1}T00:00:00",
            "task": f"Web research task {i}",
            "mistake": "incomplete research",
            "solution": f"Use web_search to find sources, then deep_research for details",
            "frequency": 1,
        })
    sm.save_playbook(lessons)
    return sm


@pytest.fixture
def skill_memory_no_patterns(tmp_path):
    sm = SkillMemory(tmp_path)
    # Create lessons with no recurring patterns
    lessons = [
        {"timestamp": "2025-01-01", "task": "Task 1", "mistake": "m1", "solution": "Just use execute", "frequency": 1},
        {"timestamp": "2025-01-02", "task": "Task 2", "mistake": "m2", "solution": "Just use recall", "frequency": 1},
    ]
    sm.save_playbook(lessons)
    return sm


class TestDetectToolPatterns:
    def test_finds_recurring_patterns(self, skill_memory_with_patterns):
        patterns = detect_tool_patterns(skill_memory_with_patterns)
        assert len(patterns) >= 1
        # Should find the file_system + execute + recall pattern
        pattern_names = [p["pattern_name"] for p in patterns]
        assert any("execute" in name and "file_system" in name for name in pattern_names)

    def test_patterns_have_minimum_frequency(self, skill_memory_with_patterns):
        patterns = detect_tool_patterns(skill_memory_with_patterns)
        for p in patterns:
            assert p["frequency"] >= 3

    def test_no_patterns_in_sparse_playbook(self, skill_memory_no_patterns):
        patterns = detect_tool_patterns(skill_memory_no_patterns)
        assert len(patterns) == 0

    def test_none_skill_memory(self):
        patterns = detect_tool_patterns(None)
        assert patterns == []

    def test_empty_playbook(self, tmp_path):
        sm = SkillMemory(tmp_path)
        patterns = detect_tool_patterns(sm)
        assert patterns == []

    def test_patterns_include_examples(self, skill_memory_with_patterns):
        patterns = detect_tool_patterns(skill_memory_with_patterns)
        for p in patterns:
            assert "description" in p
            assert len(p["description"]) > 0
            assert "pattern_name" in p
            assert p["pattern_name"].startswith("strategy:")

    def test_own_pattern_lessons_do_not_inflate_counts(self, skill_memory_with_patterns):
        """Regression (2026-07-20): previously the detector counted its
        own previously-saved "[Pattern] ..." lessons as instances (their
        task/description contain the tool keywords), so every REM cycle
        the count self-reinforced by one. Pattern-writer output — the
        "[Pattern]" task prefix and the source="dream_pattern" tag —
        must be excluded."""
        sm = skill_memory_with_patterns
        baseline = detect_tool_patterns(sm)
        base_freq = {p["pattern_name"]: p["frequency"] for p in baseline}

        playbook = sm._load_playbook()
        # Simulate two prior REM cycles having saved the detected pattern.
        for i in range(2):
            playbook.append({
                "timestamp": f"2025-03-0{i+1}T00:00:00",
                "task": "[Pattern] strategy:execute → file_system → recall",
                "mistake": "none",
                "solution": ("Recurring tool pattern (execute → file_system "
                             "→ recall) seen in 4 lessons. Examples: ..."),
                "source": "dream_pattern",
                "frequency": 1,
            })
        sm.save_playbook(playbook)

        again = detect_tool_patterns(sm)
        again_freq = {p["pattern_name"]: p["frequency"] for p in again}
        assert again_freq == base_freq, (
            "pattern-writer output must not count as pattern instances"
        )
