"""Unit tests for the NetworkX-backed Spreading Activation GraphRAG layer."""
import sqlite3

import networkx as nx
import pytest

from ghost_agent.memory.graph import GraphMemory


@pytest.fixture
def gm(tmp_path):
    return GraphMemory(tmp_path)


@pytest.fixture
def populated_gm(tmp_path):
    g = GraphMemory(tmp_path)
    g.add_triplets([
        {"subject": "user", "predicate": "OWNS", "object": "dog"},
        {"subject": "dog", "predicate": "NAMED", "object": "max"},
        {"subject": "user", "predicate": "LIVES_IN", "object": "berlin"},
        {"subject": "berlin", "predicate": "CAPITAL_OF", "object": "germany"},
        {"subject": "max", "predicate": "BREED", "object": "husky"},
        # Disconnected component to ensure it is not surfaced.
        {"subject": "moon", "predicate": "ORBITS", "object": "earth"},
    ])
    return g


# ---------------------------------------------------------------- mirroring


def test_init_creates_empty_nx_graph(gm):
    assert isinstance(gm.nx_graph, nx.MultiDiGraph)
    assert gm.nx_graph.number_of_nodes() == 0


def test_add_triplets_mirrors_into_nx_graph(gm):
    gm.add_triplets([
        {"subject": "Alice", "predicate": "knows", "object": "Bob"},
    ])
    assert "alice" in gm.nx_graph
    assert "bob" in gm.nx_graph
    assert gm.nx_graph.has_edge("alice", "bob")
    edge = list(gm.nx_graph["alice"]["bob"].values())[0]
    assert edge["predicate"] == "KNOWS"
    assert edge["weight"] == 1


def test_add_triplets_reinforces_weight_in_mirror(gm):
    triplet = [{"subject": "alice", "predicate": "KNOWS", "object": "bob"}]
    gm.add_triplets(triplet)
    gm.add_triplets(triplet)
    edge = list(gm.nx_graph["alice"]["bob"].values())[0]
    assert edge["weight"] == 2


def test_initialize_graph_hydrates_from_existing_sqlite(tmp_path):
    """A fresh GraphMemory pointing at an existing DB rebuilds the mirror."""
    first = GraphMemory(tmp_path)
    first.add_triplets([
        {"subject": "rome", "predicate": "CAPITAL_OF", "object": "italy"},
        {"subject": "italy", "predicate": "IN", "object": "europe"},
    ])

    # Brand-new instance — exercises initialize_graph() against persisted rows.
    second = GraphMemory(tmp_path)
    assert second.nx_graph.number_of_nodes() == 3
    assert second.nx_graph.has_edge("rome", "italy")
    assert second.nx_graph.has_edge("italy", "europe")


def test_delete_by_target_updates_both_stores(gm):
    gm.add_triplets([
        {"subject": "lois lane", "predicate": "WORKS_AT", "object": "daily planet"},
        {"subject": "superman", "predicate": "LIKES", "object": "lois lane"},
        {"subject": "batman", "predicate": "WORKS_IN", "object": "gotham"},
    ])
    deleted = gm.delete_by_target("lois")
    assert deleted == 2
    # SQLite
    with sqlite3.connect(gm.db_path) as conn:
        rows = conn.execute("SELECT subject, object FROM triplets").fetchall()
    assert rows == [("batman", "gotham")]
    # NetworkX mirror
    assert "lois lane" not in gm.nx_graph
    assert "daily planet" not in gm.nx_graph
    assert gm.nx_graph.has_edge("batman", "gotham")


def test_wipe_all_clears_nx_graph(gm):
    gm.add_triplets([{"subject": "a", "predicate": "IS", "object": "b"}])
    gm.wipe_all()
    assert gm.nx_graph.number_of_nodes() == 0
    assert gm.get_neighborhood(["alpha"]) == []


def test_execute_graph_compression_rebuilds_mirror(gm):
    gm.add_triplets([
        {"subject": "ny", "predicate": "IN", "object": "usa"},
        {"subject": "new york", "predicate": "ALIAS", "object": "ny"},
    ])
    gm.execute_graph_compression([{"old_node": "ny", "new_node": "new york"}])
    # The 'ny' node should be gone from the mirror after compression.
    assert "ny" not in gm.nx_graph
    assert "new york" in gm.nx_graph


# ------------------------------------------------------ spreading activation


def test_get_neighborhood_returns_2hop_chain(populated_gm):
    edges = populated_gm.get_neighborhood(["user"])
    # Direct edge
    assert "- (User) -[OWNS]-> (Dog)" in edges
    # 2-hop chain in the example shape from the task description.
    assert "- (User) -[OWNS]-> (Dog) -[NAMED]-> (Max)" in edges
    # Also surfaces the other branch
    assert any("Berlin" in e for e in edges)


def test_get_neighborhood_skips_disconnected_components(populated_gm):
    edges = populated_gm.get_neighborhood(["user"])
    joined = " ".join(edges)
    assert "Moon" not in joined and "Earth" not in joined


def test_get_neighborhood_fuzzy_seed_mapping(populated_gm):
    """A misspelled query should still map onto the closest exact node."""
    edges = populated_gm.get_neighborhood(["berln"])  # missing 'i'
    assert any("Berlin" in e for e in edges)
    assert any("Germany" in e for e in edges)


def test_get_neighborhood_substring_seed_mapping(populated_gm):
    edges = populated_gm.get_neighborhood(["germ"])
    assert any("Germany" in e for e in edges)


def test_get_neighborhood_weights_rank_paths(gm):
    gm.add_triplets([{"subject": "user", "predicate": "LIKES", "object": "tea"}])
    gm.add_triplets([{"subject": "user", "predicate": "LIKES", "object": "tea"}])
    gm.add_triplets([{"subject": "user", "predicate": "LIKES", "object": "tea"}])
    gm.add_triplets([{"subject": "user", "predicate": "LIKES", "object": "coffee"}])
    edges = gm.get_neighborhood(["user"])
    # The reinforced edge ranks first.
    assert edges[0].endswith("(Tea) [Score 3]") or "[Score 3]" in edges[0]
    assert any("(Coffee)" in e for e in edges)


def test_get_neighborhood_global_limit_enforced(gm):
    triplets = [
        {"subject": "user", "predicate": "TAGGED", "object": f"item{i}"}
        for i in range(10)
    ]
    gm.add_triplets(triplets)
    edges = gm.get_neighborhood(["user"], global_limit=4)
    assert len(edges) == 4


def test_get_neighborhood_no_seeds_returns_empty(populated_gm):
    assert populated_gm.get_neighborhood(["xyznotanything"]) == []


def test_get_neighborhood_empty_graph_safe(gm):
    assert gm.get_neighborhood(["anything"]) == []


def test_get_neighborhood_short_words_skipped(populated_gm):
    # Short words must be filtered before any node lookup happens.
    assert populated_gm.get_neighborhood(["a", "of", "in"]) == []


def test_through_chain_via_seed(gm):
    """X -> seed -> Y chains should be enumerated."""
    gm.add_triplets([
        {"subject": "alice", "predicate": "KNOWS", "object": "bob"},
        {"subject": "bob", "predicate": "LIKES", "object": "coding"},
    ])
    edges = gm.get_neighborhood(["bob"])
    assert "- (Alice) -[KNOWS]-> (Bob) -[LIKES]-> (Coding)" in edges


def test_backward_chain_terminates_at_seed(gm):
    """Z -> Y -> seed chains should also be enumerated."""
    gm.add_triplets([
        {"subject": "alice", "predicate": "MANAGES", "object": "team"},
        {"subject": "team", "predicate": "BUILDS", "object": "product"},
    ])
    edges = gm.get_neighborhood(["product"])
    assert any("Alice" in e and "Team" in e and "Product" in e for e in edges)
