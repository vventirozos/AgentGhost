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
        episodic_memory=MagicMock(),
    )


class TestQuerySplit:
    """§4NQ: the LLM decomposition is gone; only the keyword split stays."""

    def test_short_query_not_split(self, bus):
        assert bus._split_query("hello world") == ["hello world"]

    def test_and_also_splits_into_parts(self, bus):
        query = "How should I handle the authentication migration and also update the database schema"
        sub = bus._split_query(query)
        assert len(sub) == 3
        assert sub[0] == query  # Original always included

    def test_a_message_with_a_pipe_is_split_too(self, bus):
        """§4NQ: the '|'/'Context:' skip only guarded the removed expansion."""
        query = "run ps aux | grep python on the box and also show the disk usage per folder"
        assert bus._split_query(query) == [query, "run ps aux | grep python on the box",
                                           "show the disk usage per folder"]

    def test_hydration_never_calls_a_model(self, bus):
        import inspect
        assert "llm_client" not in inspect.signature(bus.hydrate_context).parameters


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
    async def test_hydrate_returns_a_string(self, bus):
        """hydrate_context runs with the tiers mocked (no model call, §4NQ)."""
        bus.vector.search = MagicMock(return_value="Memory: test fact")
        bus.graph.get_neighborhood = MagicMock(return_value=["(User) -[LIKES]-> (Python)"])
        bus.episodic.search_similar = MagicMock(return_value=[])

        result = await bus.hydrate_context("test query")
        # Should work without LLM client (single query mode)
        assert isinstance(result, str)

    async def test_hydrate_empty_query(self, bus):
        result = await bus.hydrate_context("")
        assert result == ""

    async def test_hydrate_whitespace_query(self, bus):
        result = await bus.hydrate_context("   ")
        assert result == ""


class TestFetchAllTiers:
    async def test_fetch_all_tiers_returns_the_three_hydrating_tiers(self, bus):
        """§4NM: the owner's facts (graph); §4QL added the human-graded
        experience tier and §4QM the owner's fact history (both empty here:
        no store, switch off). §4TD removed the lesson tier."""
        bus.graph.owner_facts_matching = MagicMock(return_value=["edge1"])
        result = await bus._fetch_all_tiers("test query")
        assert len(result) == 3  # graph (owner facts), experience, history (§4QM)
        assert result[1] == [] and result[2] == []


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

    def test_rrf_fusion_basic(self):
        list1 = [{"source": "vector", "text": "A"}, {"source": "vector", "text": "B"}]
        list2 = [{"source": "graph", "text": "B"}, {"source": "graph", "text": "C"}]
        fused = MemoryBus._reciprocal_rank_fusion([list1, list2])
        texts = [item["text"] for item, _ in fused]
        # B should rank higher (appears in both lists)
        assert "B" in texts
