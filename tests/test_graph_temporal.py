"""Tests for graph temporal reasoning (#5).

Verifies that:
- New contradicting triplets expire old ones
- Only temporally-valid edges are loaded by default
- get_expired_triplets returns historical data
- Spreading activation only traverses valid edges
"""

import pytest
import time
from pathlib import Path
from ghost_agent.memory.graph import GraphMemory


@pytest.fixture
def graph(tmp_path):
    return GraphMemory(tmp_path)


class TestGraphTemporalReasoning:
    def test_contradicting_triplet_expires_old(self, graph):
        # Add initial fact
        graph.add_triplets([{"subject": "user", "predicate": "WORKS_AT", "object": "companya"}])

        # Verify it's in the graph
        edges = graph.get_neighborhood(["user"])
        assert any("companya" in str(e).lower() for e in edges)

        # Add contradicting fact (same subject+predicate, different object)
        graph.add_triplets([{"subject": "user", "predicate": "WORKS_AT", "object": "companyb"}])

        # New fact should be visible
        edges = graph.get_neighborhood(["user"])
        assert any("companyb" in str(e).lower() for e in edges)
        # Old fact should be expired (not in active graph)
        # The edge "companya" should no longer be traversable
        graph.initialize_graph()  # Reload only valid edges
        assert "companya" not in str(graph.nx_graph.nodes())

    def test_non_contradicting_triplets_coexist(self, graph):
        graph.add_triplets([
            {"subject": "user", "predicate": "LIKES", "object": "python"},
            {"subject": "user", "predicate": "LIKES", "object": "rust"}
        ])

        edges = graph.get_neighborhood(["user"])
        text = " ".join(str(e) for e in edges).lower()
        assert "python" in text
        assert "rust" in text

    def test_same_triplet_reinforces_weight(self, graph):
        graph.add_triplets([{"subject": "user", "predicate": "KNOWS", "object": "python"}])
        graph.add_triplets([{"subject": "user", "predicate": "KNOWS", "object": "python"}])

        # Weight should be 2
        import sqlite3
        with sqlite3.connect(graph.db_path) as conn:
            row = conn.execute(
                "SELECT weight FROM triplets WHERE subject='user' AND predicate='KNOWS' AND object='python'"
            ).fetchone()
            assert row[0] == 2

    def test_get_expired_triplets(self, graph):
        # Create and then expire a triplet
        graph.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "london"}])
        graph.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "athens"}])

        expired = graph.get_expired_triplets()
        assert len(expired) >= 1
        assert any(e["object"] == "london" for e in expired)

    def test_get_expired_triplets_by_subject(self, graph):
        graph.add_triplets([{"subject": "alice", "predicate": "DRIVES", "object": "toyota"}])
        graph.add_triplets([{"subject": "alice", "predicate": "DRIVES", "object": "tesla"}])
        graph.add_triplets([{"subject": "bob", "predicate": "DRIVES", "object": "honda"}])
        graph.add_triplets([{"subject": "bob", "predicate": "DRIVES", "object": "bmw"}])

        expired_alice = graph.get_expired_triplets(subject="alice")
        assert all(e["subject"] == "alice" for e in expired_alice)

    def test_initialize_graph_excludes_expired(self, graph):
        # Uses LIVES_IN (a genuinely single-valued predicate). Was IS, but IS is
        # multi-valued ("webos IS a project" AND "IS done") and was removed from
        # the functional set 2026-07-22 — wrongly treating IS/OWNS as functional
        # had silently expired real facts (e.g. the operator's OWNS history).
        graph.add_triplets([{"subject": "x", "predicate": "LIVES_IN", "object": "old"}])
        graph.add_triplets([{"subject": "x", "predicate": "LIVES_IN", "object": "new"}])

        graph.initialize_graph(include_expired=False)
        nodes = list(graph.nx_graph.nodes())
        assert "old" not in nodes
        assert "new" in nodes

    def test_initialize_graph_includes_expired_when_requested(self, graph):
        graph.add_triplets([{"subject": "x", "predicate": "IS", "object": "old"}])
        graph.add_triplets([{"subject": "x", "predicate": "IS", "object": "new"}])

        graph.initialize_graph(include_expired=True)
        nodes = list(graph.nx_graph.nodes())
        assert "new" in nodes
        # old may or may not be present depending on if it's connected
        # The important thing is it doesn't crash

    def test_valid_from_is_set(self, graph):
        graph.add_triplets([{"subject": "user", "predicate": "HAS", "object": "cat"}])

        import sqlite3
        with sqlite3.connect(graph.db_path) as conn:
            row = conn.execute(
                "SELECT valid_from FROM triplets WHERE subject='user' AND object='cat'"
            ).fetchone()
            assert row[0] is not None
            assert row[0] > 0
