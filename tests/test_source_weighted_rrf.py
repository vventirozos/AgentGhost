"""Tests for source-weighted RRF in MemoryBus (#6).

Verifies that:
- Intent classification works correctly
- Weighted RRF boosts appropriate sources per intent
- Factual queries boost graph, procedural boost skill
"""

import pytest
from ghost_agent.core.bus import MemoryBus


class TestIntentClassification:
    def test_factual_intent(self):
        assert MemoryBus._classify_query_intent("who is the user") == "factual"
        assert MemoryBus._classify_query_intent("what is the name") == "factual"
        assert MemoryBus._classify_query_intent("where does the user live") == "factual"

    def test_procedural_intent(self):
        assert MemoryBus._classify_query_intent("how to fix this error") == "procedural"
        assert MemoryBus._classify_query_intent("solve this debug issue") == "procedural"
        assert MemoryBus._classify_query_intent("steps to avoid this mistake") == "procedural"
        assert MemoryBus._classify_query_intent("how should I never do this") == "procedural"

    def test_contextual_intent(self):
        assert MemoryBus._classify_query_intent("tell me about the project") == "contextual"
        assert MemoryBus._classify_query_intent("general information") == "contextual"

    def test_empty_query(self):
        assert MemoryBus._classify_query_intent("") == "contextual"


class TestWeightedRRF:
    def _make_items(self, source, texts):
        return [{"source": source, "text": t} for t in texts]

    def test_factual_boosts_graph(self):
        vector = self._make_items("vector", ["User likes Python"])
        graph = self._make_items("graph", ["(User) -[LIKES]-> (Python)"])
        skill = self._make_items("skill", ["Use Python for scripting"])

        fused = MemoryBus._reciprocal_rank_fusion(
            [vector, graph, skill], intent="factual"
        )

        # Graph should score highest with factual intent (weight=2.0 vs 1.0/0.5)
        top_source = fused[0][0]["source"]
        assert top_source == "graph"

    def test_procedural_boosts_skill(self):
        vector = self._make_items("vector", ["Some memory"])
        graph = self._make_items("graph", ["Some edge"])
        skill = self._make_items("skill", ["## LESSON: Always check encoding"])

        fused = MemoryBus._reciprocal_rank_fusion(
            [vector, graph, skill], intent="procedural"
        )

        top_source = fused[0][0]["source"]
        assert top_source == "skill"

    def test_contextual_is_balanced(self):
        vector = self._make_items("vector", ["Memory 1"])
        graph = self._make_items("graph", ["Graph 1"])
        skill = self._make_items("skill", ["Skill 1"])

        fused = MemoryBus._reciprocal_rank_fusion(
            [vector, graph, skill], intent="contextual"
        )

        # Vector has highest weight (1.5) in contextual
        top_source = fused[0][0]["source"]
        assert top_source == "vector"

    def test_empty_lists_handled(self):
        fused = MemoryBus._reciprocal_rank_fusion(
            [[], [], []], intent="factual"
        )
        assert fused == []

    def test_single_source(self):
        vector = self._make_items("vector", ["Only vector"])
        fused = MemoryBus._reciprocal_rank_fusion(
            [vector, [], []], intent="factual"
        )
        assert len(fused) == 1

    def test_default_intent_is_contextual(self):
        vector = self._make_items("vector", ["test"])
        fused1 = MemoryBus._reciprocal_rank_fusion([vector, [], []])
        fused2 = MemoryBus._reciprocal_rank_fusion([vector, [], []], intent="contextual")
        # Scores should be equal
        assert abs(fused1[0][1] - fused2[0][1]) < 0.001
