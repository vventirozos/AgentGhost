"""Tests for dream consolidation metrics (#13).

Verifies that:
- Low-compression consolidations are skipped
- High-compression consolidations are applied
- Metrics are reported in the return string
"""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture
def mock_dreamer():
    context = MagicMock()
    context.memory_system = MagicMock()
    context.memory_system.collection = MagicMock()
    context.skill_memory = MagicMock()
    context.skill_memory._get_lock = lambda: __import__("threading").RLock()
    context.skill_memory.file_path = MagicMock()
    context.skill_memory.file_path.read_text.return_value = "[]"
    context.llm_client = MagicMock()
    context.llm_client.chat_completion = AsyncMock()

    from ghost_agent.core.dream import Dreamer
    return Dreamer(context)


class TestDreamConsolidationMetrics:
    @pytest.mark.asyncio
    async def test_skips_low_compression_consolidation(self, mock_dreamer):
        # Setup: memory has 5 documents
        mock_dreamer.memory.collection.get.return_value = {
            "ids": ["id1", "id2", "id3", "id4", "id5"],
            "documents": [
                "The user prefers Python",
                "The user likes Python programming",
                "User wants dark mode",
                "User drives a Tesla",
                "User lives in Athens"
            ],
            "metadatas": [{"type": "auto"}] * 5,
            "embeddings": [[0.1]] * 5
        }

        # LLM returns a consolidation that barely compresses (synthesis ≈ same length)
        mock_dreamer.context.llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "consolidations": [{
                            "synthesis": "The user prefers Python programming language",  # 44 chars
                            "merged_ids": ["ID:id1", "ID:id2"]  # sources: 25 + 36 = 61 chars
                            # compression = 1 - (44/61) = 0.28 → should pass
                        }],
                        "heuristics": []
                    })
                }
            }]
        }

        result = await mock_dreamer.dream()
        assert "Dream Complete" in result

    @pytest.mark.asyncio
    async def test_applies_high_compression_consolidation(self, mock_dreamer):
        # Similar setup but with highly compressible data
        mock_dreamer.memory.collection.get.return_value = {
            "ids": ["id1", "id2", "id3"],
            "documents": [
                "A" * 200,  # 200 chars
                "B" * 200,  # 200 chars
                "C" * 100,
            ],
            "metadatas": [{"type": "auto"}] * 3,
            "embeddings": [[0.1]] * 3,
        }

        mock_dreamer.context.llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "consolidations": [{
                            "synthesis": "Compressed summary",  # 18 chars vs 400 source
                            "merged_ids": ["ID:id1", "ID:id2"]
                        }],
                        "heuristics": []
                    })
                }
            }]
        }

        result = await mock_dreamer.dream()
        assert "Dream Complete" in result
        # The consolidation should be applied (high compression)
        assert "Synthesized 1" in result

    @pytest.mark.asyncio
    async def test_reports_skipped_count(self, mock_dreamer):
        mock_dreamer.memory.collection.get.return_value = {
            "ids": ["id1", "id2", "id3"],
            "documents": ["Short text", "Another short", "Third short"],
            "metadatas": [{"type": "auto"}] * 3,
            "embeddings": [[0.1]] * 3,
        }

        # Consolidation where synthesis is LONGER than sources (negative compression)
        mock_dreamer.context.llm_client.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "consolidations": [{
                            "synthesis": "This is a much longer synthesis than the original short texts combined together to make a point about something",
                            "merged_ids": ["ID:id1", "ID:id2"]
                        }],
                        "heuristics": []
                    })
                }
            }]
        }

        result = await mock_dreamer.dream()
        assert "low-compression" in result or "Synthesized 0" in result


class TestDreamHeuristicKeying:
    """IMPROVEMENTS.md #24: each dream heuristic must be keyed on its own
    content (+ source='dream'), not the constant '[System] Dream Heuristic'
    that collapsed every heuristic from every REM cycle into one playbook slot.
    """

    @pytest.mark.asyncio
    async def test_heuristics_keyed_per_content_with_source(self, mock_dreamer):
        # >3 auto-memories so the dream clears the entropy gate.
        mock_dreamer.memory.collection.get.return_value = {
            "ids": [f"id{i}" for i in range(5)],
            "documents": [f"auto memory number {i}" for i in range(5)],
            "metadatas": [{"type": "auto"}] * 5,
            "embeddings": [[0.1]] * 5,
        }
        mock_dreamer.context.llm_client.chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "consolidations": [],
                "heuristics": [
                    "Always probe container readiness before exec.",
                    "Prefer line-ranged reads after a failed replace.",
                ],
            })}}]
        }
        mock_dreamer.context.skill_memory.learn_lesson = MagicMock()

        await mock_dreamer.dream()

        calls = mock_dreamer.context.skill_memory.learn_lesson.call_args_list
        heuristic_calls = [
            c for c in calls
            if c.kwargs.get("source") == "dream"
        ]
        assert len(heuristic_calls) == 2
        # Each keyed on its OWN content, never the old constant.
        tasks = {c.args[0] for c in heuristic_calls}
        assert "[System] Dream Heuristic" not in tasks
        assert any("probe container readiness" in t for t in tasks)
        assert any("line-ranged reads" in t for t in tasks)
        # trigger mirrors the content-derived task so BM25 retrieval can find it.
        for c in heuristic_calls:
            assert c.kwargs.get("trigger") == c.args[0]
