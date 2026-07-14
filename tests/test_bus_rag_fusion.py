"""Tests for the RAG-Fusion upgrades to the Cognitive Event Bus."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from ghost_agent.core.bus import MemoryBus


@pytest.fixture
def bus():
    return MemoryBus(
        vector_memory=MagicMock(),
        graph_memory=MagicMock(),
        skill_memory=MagicMock(),
        episodic_memory=MagicMock(),
    )


class TestQueryDecomposition:
    async def test_short_query_not_decomposed(self, bus):
        sub = await bus._decompose_query("hello world", llm_client=None)
        assert sub == ["hello world"]

    async def test_long_query_decomposed_heuristic(self, bus):
        query = "How should I handle the authentication migration and also update the database schema"
        sub = await bus._decompose_query(query, llm_client=None)
        assert len(sub) >= 2
        assert query in sub  # Original always included

    async def test_no_decomposition_on_internal_expanded_query(self, bus):
        """Expanded queries with 'Context:' or '|' markers should not be split."""
        query = "Context: cool AI command and does magic. | User intent: run it"
        sub = await bus._decompose_query(query, llm_client=None)
        assert sub == [query]  # Only the original, no split

    async def test_llm_decomposition(self, bus):
        mock_llm = MagicMock()
        mock_llm.route = AsyncMock(return_value={
            "choices": [{"message": {"content": json.dumps([
                "authentication middleware architecture",
                "database schema migration patterns",
                "compliance requirements",
            ])}}],
        })
        query = "How should I handle the authentication migration given compliance requirements"
        sub = await bus._decompose_query(query, llm_client=mock_llm)
        assert len(sub) >= 3  # Original + decomposed
        assert query in sub

    async def test_llm_decomposition_fallback_on_failure(self, bus):
        mock_llm = MagicMock()
        mock_llm.route = AsyncMock(side_effect=Exception("worker down"))
        query = "How should I handle the auth migration and also fix the database"
        sub = await bus._decompose_query(query, llm_client=mock_llm)
        # Should fall back to heuristic
        assert len(sub) >= 2

    async def test_max_sub_queries_capped(self, bus):
        mock_llm = MagicMock()
        mock_llm.route = AsyncMock(return_value={
            "choices": [{"message": {"content": json.dumps([
                "q1", "q2", "q3", "q4", "q5", "q6",
            ])}}],
        })
        query = "a long query with many words to trigger decomposition via LLM"
        sub = await bus._decompose_query(query, llm_client=mock_llm)
        assert len(sub) <= 4  # Original + max 3


class TestDedupItems:
    def test_dedup_removes_duplicates(self):
        items = [
            {"source": "vector", "text": "fact A"},
            {"source": "vector", "text": "fact A"},
            {"source": "vector", "text": "fact B"},
        ]
        result = MemoryBus._dedup_items(items)
        assert len(result) == 2

    def test_dedup_preserves_order(self):
        items = [
            {"source": "vector", "text": "first"},
            {"source": "vector", "text": "second"},
            {"source": "vector", "text": "first"},
        ]
        result = MemoryBus._dedup_items(items)
        assert result[0]["text"] == "first"
        assert result[1]["text"] == "second"

    def test_dedup_empty(self):
        assert MemoryBus._dedup_items([]) == []


class TestHydrateContextRAGFusion:
    async def test_hydrate_with_llm_client(self, bus):
        """Test that hydrate_context accepts llm_client parameter."""
        bus.vector.search = MagicMock(return_value="Memory: test fact")
        bus.graph.get_neighborhood = MagicMock(return_value=["(User) -[LIKES]-> (Python)"])
        bus.skill.get_playbook_context = MagicMock(return_value="")
        bus.episodic.search_similar = MagicMock(return_value=[])

        result = await bus.hydrate_context("test query", llm_client=None)
        # Should work without LLM client (single query mode)
        assert isinstance(result, str)

    async def test_hydrate_empty_query(self, bus):
        result = await bus.hydrate_context("", llm_client=None)
        assert result == ""

    async def test_hydrate_whitespace_query(self, bus):
        result = await bus.hydrate_context("   ", llm_client=None)
        assert result == ""


class TestFetchAllTiers:
    async def test_fetch_all_tiers_returns_five_lists(self, bus):
        bus.vector.search = MagicMock(return_value="fact1")
        bus.graph.get_neighborhood = MagicMock(return_value=["edge1"])
        bus.skill.get_playbook_context = MagicMock(return_value="lesson1")
        bus.episodic.search_similar = MagicMock(return_value=[])
        bus.episodic.format_for_context = MagicMock(return_value="")

        result = await bus._fetch_all_tiers("test query")
        assert len(result) == 5  # vector, graph, skill, episodic, session


class TestExistingBusFunctionality:
    """Ensure existing bus functionality is preserved."""

    def test_classify_query_intent_factual(self):
        assert MemoryBus._classify_query_intent("who works at Google") == "factual"

    def test_classify_query_intent_procedural(self):
        assert MemoryBus._classify_query_intent("how to fix this error") == "procedural"

    def test_classify_query_intent_contextual(self):
        assert MemoryBus._classify_query_intent("tell me about the project") == "contextual"

    async def test_publish_fact_dedup(self, bus):
        bus.vector.add = MagicMock()
        fact = {"text": "test fact", "triplets": []}
        r1 = await bus.publish_fact("auto", fact)
        r2 = await bus.publish_fact("auto", fact)
        assert r2["vector"] == "dedup"

    def test_extract_query_terms(self):
        terms = MemoryBus._extract_query_terms("What is the capital of France?")
        assert "france" in [t.lower() for t in terms] or "capital" in [t.lower() for t in terms]

    def test_rrf_fusion_basic(self):
        list1 = [{"source": "vector", "text": "A"}, {"source": "vector", "text": "B"}]
        list2 = [{"source": "graph", "text": "B"}, {"source": "graph", "text": "C"}]
        fused = MemoryBus._reciprocal_rank_fusion([list1, list2])
        texts = [item["text"] for item, _ in fused]
        # B should rank higher (appears in both lists)
        assert "B" in texts
