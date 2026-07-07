"""Graph forgetting + node-cache (IMPROVEMENTS.md #27c).

The knowledge graph is the only UNCAPPED memory tier — non-functional predicates
accumulate forever and weight-1 stale edges dilute retrieval, while
`_map_words_to_seeds` rebuilt `list(nodes())` on every query word (O(nodes)/turn).
`prune_stale_edges` gives it a decay story (weight is the signal); a cached node
list removes the per-query materialization.
"""
import sqlite3

import pytest

from ghost_agent.memory.graph import GraphMemory


@pytest.fixture
def graph(tmp_path):
    return GraphMemory(tmp_path)


def _age_edge(graph, subject, days):
    """Backdate an edge's timestamp so the age filter can bite deterministically."""
    with sqlite3.connect(graph.db_path) as conn:
        conn.execute(
            f"UPDATE triplets SET timestamp = datetime('now', '-{days} days') "
            f"WHERE subject = ?", (subject,))
        conn.commit()


def test_prune_drops_old_weight1_edges(graph):
    graph.add_triplets([{"subject": "alice", "predicate": "LIKES", "object": "tea"}])
    _age_edge(graph, "alice", 90)
    removed = graph.prune_stale_edges(max_age_days=45)
    assert removed == 1
    # Gone from both DB and the in-memory mirror.
    with sqlite3.connect(graph.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM triplets").fetchone()[0] == 0
    assert "alice" not in graph.nx_graph


def test_prune_keeps_reinforced_edges(graph):
    # Reinforce the same edge twice → weight 2.
    graph.add_triplets([{"subject": "bob", "predicate": "OWNS", "object": "dog"}])
    graph.add_triplets([{"subject": "bob", "predicate": "OWNS", "object": "dog"}])
    _age_edge(graph, "bob", 90)
    removed = graph.prune_stale_edges(max_age_days=45, keep_min_weight=1)
    assert removed == 0  # weight 2 > threshold → kept regardless of age
    assert graph.nx_graph.has_edge("bob", "dog")


def test_prune_keeps_recent_edges(graph):
    graph.add_triplets([{"subject": "carol", "predicate": "SAW", "object": "movie"}])
    # Fresh (default timestamp = now) → not old enough to prune.
    removed = graph.prune_stale_edges(max_age_days=45)
    assert removed == 0
    assert graph.nx_graph.has_edge("carol", "movie")


def test_prune_is_idempotent(graph):
    graph.add_triplets([{"subject": "dave", "predicate": "HAS", "object": "hat"}])
    _age_edge(graph, "dave", 90)
    assert graph.prune_stale_edges(max_age_days=45) == 1
    assert graph.prune_stale_edges(max_age_days=45) == 0


def test_node_cache_used_and_invalidated(graph):
    graph.add_triplets([{"subject": "eve", "predicate": "KNOWS", "object": "frank"}])
    # First seed-map builds the cache.
    seeds = graph._map_words_to_seeds(["eve"])
    assert "eve" in seeds
    assert graph._node_list_cache is not None
    # A new edge invalidates the cache so a new node is visible.
    graph.add_triplets([{"subject": "grace", "predicate": "KNOWS", "object": "heidi"}])
    assert graph._node_list_cache is None
    seeds2 = graph._map_words_to_seeds(["grace"])
    assert "grace" in seeds2


def test_neighborhood_still_works_after_prune(graph):
    graph.add_triplets([
        {"subject": "root", "predicate": "LINKS", "object": "keepme"},
        {"subject": "root", "predicate": "LINKS", "object": "dropme"},
    ])
    # Reinforce keepme so it survives; age both.
    graph.add_triplets([{"subject": "root", "predicate": "LINKS", "object": "keepme"}])
    _age_edge(graph, "root", 90)
    graph.prune_stale_edges(max_age_days=45)
    nb = graph.get_neighborhood(["root"])
    joined = " ".join(nb).lower()  # neighborhood renderer title-cases nodes
    assert "keepme" in joined
    assert "dropme" not in joined
