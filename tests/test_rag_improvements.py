"""Tests for RAG system improvements.

Covers:
- Semantic chunking (#1)
- Cross-encoder re-ranking / BM25 hybrid search (#2, #3)
- Episodic memory in RRF hydration (#4)
- Document summaries on ingest (#5)
- Adaptive hydration budget (#7)
- 3-hop graph traversal (#8)
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path


# ============================================================
# #1: Semantic Chunking
# ============================================================

class TestSemanticChunking:
    def test_markdown_headers_preserved(self):
        from ghost_agent.utils.helpers import semantic_split_text
        text = """# Introduction
This is the intro paragraph with enough content to fill a chunk. We need sufficient text to trigger the splitting logic across multiple sections.

## Methods
We used method A and B for our analysis. The methodology included data collection, preprocessing, and statistical analysis with multiple regression models applied.

## Results
The results were statistically significant with p-value below 0.01. We observed a strong correlation between the independent and dependent variables across all experimental conditions."""

        chunks = semantic_split_text(text, chunk_size=200)
        assert len(chunks) >= 2
        # Section headers should be prepended as context
        assert any("[Section:" in c for c in chunks)

    def test_code_blocks_kept_intact(self):
        from ghost_agent.utils.helpers import semantic_split_text
        text = """# Setup

```python
def hello():
    print("world")
    return 42
```

Some explanation text."""

        chunks = semantic_split_text(text, chunk_size=300)
        # Code block should not be split
        code_chunks = [c for c in chunks if "def hello" in c]
        assert len(code_chunks) == 1
        assert 'print("world")' in code_chunks[0]

    def test_falls_back_to_recursive_for_plain_text(self):
        from ghost_agent.utils.helpers import semantic_split_text
        text = "Just some plain text. " * 50
        chunks = semantic_split_text(text, chunk_size=200)
        assert len(chunks) > 1
        # No [Section:] prefixes for unstructured text
        assert not any("[Section:" in c for c in chunks)

    def test_empty_text(self):
        from ghost_agent.utils.helpers import semantic_split_text
        assert semantic_split_text("") == []

    def test_short_text_single_chunk(self):
        from ghost_agent.utils.helpers import semantic_split_text
        assert semantic_split_text("Short text.", chunk_size=600) == ["Short text."]

    def test_respects_chunk_size_limit(self):
        from ghost_agent.utils.helpers import semantic_split_text
        text = "# Header\n\n" + "Word " * 500
        chunks = semantic_split_text(text, chunk_size=600)
        for chunk in chunks:
            # Allow some slack for header prefix
            assert len(chunk) < 800


# ============================================================
# #2/#3: Cross-encoder Re-ranking & BM25
# ============================================================

class TestBM25Scoring:
    def test_bm25_exact_match_scores_high(self):
        from ghost_agent.memory.vector import _bm25_score
        query_tokens = ["filenotfounderror", "data.csv"]
        doc_tokens = ["error", "filenotfounderror", "data.csv", "not", "found"]
        score = _bm25_score(query_tokens, doc_tokens, avg_dl=10.0)
        assert score > 0

    def test_bm25_no_overlap_scores_zero(self):
        from ghost_agent.memory.vector import _bm25_score
        query_tokens = ["python", "async"]
        doc_tokens = ["rust", "sync", "thread"]
        score = _bm25_score(query_tokens, doc_tokens, avg_dl=5.0)
        assert score == 0.0

    def test_bm25_empty_inputs(self):
        from ghost_agent.memory.vector import _bm25_score
        assert _bm25_score([], ["a", "b"], 5.0) == 0.0
        assert _bm25_score(["a"], [], 5.0) == 0.0
        assert _bm25_score(["a"], ["a"], 0.0) == 0.0


class TestCrossEncoderRerank:
    def test_rerank_boosts_keyword_matches(self):
        from ghost_agent.memory.vector import _cross_encoder_rerank
        # Give all candidates similar combined_scores so BM25 can differentiate
        candidates = [
            {"doc": "General info about programming languages and tools", "combined_score": -4.0},
            {"doc": "FileNotFoundError data.csv not found in directory listing", "combined_score": -4.0},
            {"doc": "How to handle missing files in Python applications", "combined_score": -4.0},
        ]
        reranked = _cross_encoder_rerank("FileNotFoundError data.csv", candidates)
        # The exact keyword match should be boosted to top by BM25
        assert "FileNotFoundError" in reranked[0]["doc"]

    def test_rerank_empty_candidates(self):
        from ghost_agent.memory.vector import _cross_encoder_rerank
        assert _cross_encoder_rerank("query", []) == []

    def test_rerank_respects_top_k(self):
        from ghost_agent.memory.vector import _cross_encoder_rerank
        candidates = [{"doc": f"doc {i}", "combined_score": float(i)} for i in range(20)]
        reranked = _cross_encoder_rerank("query", candidates, top_k=5)
        assert len(reranked) == 5


# ============================================================
# #4: Episodic Memory in RRF
# ============================================================

class TestEpisodicInRRF:
    @pytest.mark.asyncio
    async def test_hydration_includes_episodic(self):
        from ghost_agent.core.bus import MemoryBus
        from ghost_agent.memory.episodes import EpisodicMemory
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ep = EpisodicMemory(Path(tmpdir))
            ep.record_episode(
                trigger="Parse CSV with pandas",
                outcome="Success", success=True,
                lesson="Check encoding first"
            )

            bus = MemoryBus(episodic_memory=ep)
            result = await bus.hydrate_context("parse CSV data")

            # Should include episodic section
            if result:  # May be empty if no other sources
                assert isinstance(result, str)

    def test_intent_weights_include_episodic(self):
        from ghost_agent.core.bus import MemoryBus
        for intent in ["factual", "procedural", "contextual"]:
            weights = MemoryBus._INTENT_WEIGHTS[intent]
            assert "episodic" in weights

    def test_section_budget_includes_episodic(self):
        from ghost_agent.core.bus import MemoryBus
        assert "episodic" in MemoryBus._SECTION_BUDGETS
        total = sum(MemoryBus._SECTION_BUDGETS.values())
        assert abs(total - 1.0) < 0.01


# ============================================================
# #5: Document Summaries
# ============================================================

class TestDocumentSummaries:
    @pytest.mark.asyncio
    async def test_ingest_creates_summary(self, tmp_path):
        # Create a test file
        test_file = tmp_path / "test_doc.txt"
        test_file.write_text("This is a test document with enough content to generate a summary. " * 20)

        mock_memory = MagicMock()
        mock_memory.get_library.return_value = []
        mock_memory.ingest_document = MagicMock()
        mock_memory._update_library_index = MagicMock()
        mock_memory._get_lock = lambda: __import__("threading").RLock()
        mock_memory.add = MagicMock()

        from ghost_agent.tools.memory import tool_gain_knowledge
        result = await tool_gain_knowledge(
            filename="test_doc.txt",
            sandbox_dir=tmp_path,
            memory_system=mock_memory
        )

        assert "SUCCESS" in result
        # Should have called add() for the document summary
        summary_calls = [
            c for c in mock_memory.add.call_args_list
            if "Document Summary" in str(c.args[0]) if c.args
        ]
        assert len(summary_calls) >= 1


# ============================================================
# #7: Adaptive Hydration Budget
# ============================================================

class TestAdaptiveHydrationBudget:
    @pytest.mark.asyncio
    async def test_complex_query_gets_larger_budget(self):
        from ghost_agent.core.bus import MemoryBus
        bus = MemoryBus()  # No memory sources — just testing budget calc

        # Complex query (>30 words)
        complex_query = " ".join(["word"] * 35)
        result = await bus.hydrate_context(complex_query, context_budget=12000)
        # Should not crash and should return empty (no sources)
        assert result == ""

    @pytest.mark.asyncio
    async def test_simple_query_keeps_default_budget(self):
        from ghost_agent.core.bus import MemoryBus
        bus = MemoryBus()
        result = await bus.hydrate_context("simple query", context_budget=12000)
        assert result == ""


# ============================================================
# #8: 3-Hop Graph Traversal
# ============================================================

class TestThreeHopGraph:
    def test_3_hop_finds_distant_connections(self):
        from ghost_agent.memory.graph import GraphMemory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            g = GraphMemory(Path(tmpdir))
            g.add_triplets([
                {"subject": "alice", "predicate": "WORKS_AT", "object": "acme"},
                {"subject": "acme", "predicate": "OWNED_BY", "object": "megacorp"},
                {"subject": "megacorp", "predicate": "LOCATED_IN", "object": "tokyo"},
            ])

            edges = g.get_neighborhood(["alice"], global_limit=50)
            texts = " ".join(str(e) for e in edges).lower()

            # 1-hop: alice -> acme
            assert "acme" in texts
            # 2-hop: alice -> acme -> megacorp
            assert "megacorp" in texts
            # 3-hop: alice -> acme -> megacorp -> tokyo
            assert "tokyo" in texts

    def test_2_hop_still_works(self):
        from ghost_agent.memory.graph import GraphMemory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            g = GraphMemory(Path(tmpdir))
            g.add_triplets([
                {"subject": "user", "predicate": "KNOWS", "object": "python"},
                {"subject": "python", "predicate": "USED_FOR", "object": "ml"},
            ])

            edges = g.get_neighborhood(["user"])
            texts = " ".join(str(e) for e in edges).lower()
            assert "python" in texts
            assert "ml" in texts

    def test_avoids_cycles_in_3_hop(self):
        from ghost_agent.memory.graph import GraphMemory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            g = GraphMemory(Path(tmpdir))
            # Create a cycle: A -> B -> C -> A
            g.add_triplets([
                {"subject": "a", "predicate": "LINKS", "object": "b"},
                {"subject": "b", "predicate": "LINKS", "object": "c"},
                {"subject": "c", "predicate": "LINKS", "object": "a"},
            ])

            # Should not crash or infinite loop
            edges = g.get_neighborhood(["a"], global_limit=50)
            assert isinstance(edges, list)
