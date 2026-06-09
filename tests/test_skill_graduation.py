"""Tests for skill graduation pipeline (#4).

Verifies that:
- Lessons qualify for graduation based on frequency and code patterns
- LLM generates tool code from lesson
- Graduated lessons are marked in the playbook
"""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.dream import Dreamer


@pytest.fixture(autouse=True)
def _mock_create_skill():
    """Graduation now actually creates the skill via the TDD-gated
    tool_create_skill. Patch it to succeed by default so the candidate-
    selection tests don't need a live sandbox; individual tests override
    the return value to exercise the failure path."""
    with patch(
        "ghost_agent.tools.acquired_skills.tool_create_skill",
        new=AsyncMock(return_value="Success: Skill 'csv_parser' acquired and tested successfully."),
    ) as m:
        yield m


@pytest.fixture
def mock_dreamer_for_graduation():
    context = MagicMock()
    context.memory_system = MagicMock()
    context.skill_memory = MagicMock()
    context.skill_memory._get_lock = lambda: __import__("threading").RLock()

    # Playbook with a graduatable lesson
    playbook = [
        {
            "timestamp": "2025-01-01",
            "task": "Parse mixed-encoding CSV",
            "mistake": "Used utf-8 which failed",
            "solution": "def parse_csv(path): import csv; return csv.reader(open(path, encoding='latin-1'))",
            "frequency": 6,  # Above threshold
        },
        {
            "timestamp": "2025-01-02",
            "task": "Simple greeting",
            "mistake": "none",
            "solution": "Just say hello",
            "frequency": 10,  # High freq but no code
        },
        {
            "timestamp": "2025-01-03",
            "task": "Already graduated",
            "mistake": "none",
            "solution": "def already(): return True",
            "frequency": 8,
            "graduated": True,
        },
    ]
    context.skill_memory.file_path = MagicMock()
    context.skill_memory.file_path.read_text.return_value = json.dumps(playbook)
    context.skill_memory.save_playbook = MagicMock()
    # graduate_lessons now re-loads the playbook under the lock right before
    # saving (the stale-snapshot write-back destroyed concurrently-learned
    # lessons) and saves via _save_playbook_unlocked.
    context.skill_memory._load_playbook = MagicMock(return_value=json.loads(json.dumps(playbook)))
    context.skill_memory._save_playbook_unlocked = MagicMock()

    context.llm_client = MagicMock()
    context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{
            "message": {
                "content": json.dumps({
                    "name": "csv_parser",
                    "description": "Parse CSV with encoding detection",
                    "parameters_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                    "python_code": "import sys\nprint('parsed')",
                    "test_payload": '{\"path\": \"test.csv\"}'
                })
            }
        }]
    })

    return Dreamer(context)


class TestSkillGraduation:
    @pytest.mark.asyncio
    async def test_graduates_high_frequency_code_lesson(self, mock_dreamer_for_graduation):
        result = await mock_dreamer_for_graduation.graduate_lessons()

        assert "1 lessons graduated" in result
        # The freshly re-loaded playbook must be saved with graduated=True
        # flipped on the matching lesson (not a stale snapshot write-back).
        sm = mock_dreamer_for_graduation.context.skill_memory
        sm._save_playbook_unlocked.assert_called()
        saved = sm._save_playbook_unlocked.call_args[0][0]
        graduated_tasks = [l["task"] for l in saved if l.get("graduated")]
        assert "Parse mixed-encoding CSV" in graduated_tasks

    @pytest.mark.asyncio
    async def test_skips_already_graduated(self, mock_dreamer_for_graduation):
        # The "Already graduated" lesson should be skipped
        result = await mock_dreamer_for_graduation.graduate_lessons()
        # Only 1 should graduate (the CSV parser, not the already graduated one)
        assert "1 lessons graduated" in result

    @pytest.mark.asyncio
    async def test_skips_no_code_lessons(self, mock_dreamer_for_graduation):
        # "Simple greeting" has high freq but no code indicators
        # Should not be graduated
        result = await mock_dreamer_for_graduation.graduate_lessons()
        assert "1 lessons graduated" in result  # Only the CSV one

    @pytest.mark.asyncio
    async def test_failed_creation_does_not_graduate(self, mock_dreamer_for_graduation, _mock_create_skill):
        # When TDD/creation fails, the lesson must NOT be marked graduated
        # (the bug this fix closes: lessons were burned without a skill).
        _mock_create_skill.return_value = "Skill creation failed: test exited 1."
        result = await mock_dreamer_for_graduation.graduate_lessons()
        assert "0 lessons graduated" in result
        # The playbook must NOT be saved to flip graduated=True
        mock_dreamer_for_graduation.context.skill_memory.save_playbook.assert_not_called()
        mock_dreamer_for_graduation.context.skill_memory._save_playbook_unlocked.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_candidates(self):
        context = MagicMock()
        context.skill_memory = MagicMock()
        context.skill_memory._get_lock = lambda: __import__("threading").RLock()
        context.skill_memory.file_path = MagicMock()
        context.skill_memory.file_path.read_text.return_value = json.dumps([
            {"task": "low freq", "mistake": "m", "solution": "s", "frequency": 1}
        ])
        context.memory_system = MagicMock()

        dreamer = Dreamer(context)
        result = await dreamer.graduate_lessons()
        assert "No lessons ready" in result

    @pytest.mark.asyncio
    async def test_no_skill_memory(self):
        context = MagicMock()
        context.skill_memory = None
        context.memory_system = MagicMock()

        dreamer = Dreamer(context)
        result = await dreamer.graduate_lessons()
        assert "No skill memory" in result
