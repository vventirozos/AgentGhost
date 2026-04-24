import pytest
import sqlite3
from pathlib import Path
from ghost_agent.memory.graph import GraphMemory

@pytest.fixture
def temp_graph(tmp_path):
    graph = GraphMemory(tmp_path)
    yield graph

def test_graph_initialization(temp_graph):
    """Test that the DB initializes with the correct schema."""
    db_path = temp_graph.db_path
    assert db_path.exists()
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='triplets'")
        assert cursor.fetchone() is not None

def test_add_triplets_normalizes_data(temp_graph):
    """Test that entities are lowercased and predicates are uppercased."""
    triplets = [
        {"subject": "User", "predicate": "LIKES", "object": "Python"},
        {"subject": " GHOST_AGENT ", "predicate": " uses ", "object": "SQLite "}
    ]
    added = temp_graph.add_triplets(triplets)
    assert added == 2
    
    with sqlite3.connect(temp_graph.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT subject, predicate, object FROM triplets ORDER BY subject")
        rows = cursor.fetchall()
        
        # ' ghost_agent ' -> 'ghost_agent'
        assert rows[0] == ("ghost_agent", "USES", "sqlite")
        assert rows[1] == ("user", "LIKES", "python")

def test_add_triplets_corner_cases(temp_graph):
    """Test empty arrays, invalid structures, and partial keys."""
    assert temp_graph.add_triplets([]) == 0
    assert temp_graph.add_triplets([{"subject": "foo"}]) == 0
    assert temp_graph.add_triplets([{"predicate": "IS", "object": "bar"}]) == 0
    # Missing subject/object completely
    assert temp_graph.add_triplets([{"subject": "", "predicate": "IS", "object": "bar"}]) == 0
    
def test_add_triplets_deduplication(temp_graph):
    """Test that UNIQUE constraints prevent duplicate edges and instead increase weight."""
    triplets = [{"subject": "Alice", "predicate": "KNOWS", "object": "Bob"}]
    # First addition -> rowcount 1
    assert temp_graph.add_triplets(triplets) == 1
    # Second addition -> DO UPDATE sets weight + 1, returning rowcount 1
    assert temp_graph.add_triplets(triplets) == 1
    
    with sqlite3.connect(temp_graph.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT weight FROM triplets WHERE subject='alice'")
        assert cursor.fetchone()[0] == 2

def test_get_neighborhood_basic(temp_graph):
    """Test that spreading activation surfaces both 1-hop edges and the 2-hop chain through the seed."""
    temp_graph.add_triplets([
        {"subject": "alice", "predicate": "KNOWS", "object": "bob"},
        {"subject": "bob", "predicate": "LIKES", "object": "coding"},
        {"subject": "charlie", "predicate": "KNOWS", "object": "david"}
    ])

    edges = temp_graph.get_neighborhood(["bob"])

    # Disconnected (charlie/david) component must not appear.
    assert all("Charlie" not in e and "David" not in e for e in edges)
    assert "- (Alice) -[KNOWS]-> (Bob)" in edges
    assert "- (Bob) -[LIKES]-> (Coding)" in edges
    # 2-hop through-chain produced by spreading activation
    assert "- (Alice) -[KNOWS]-> (Bob) -[LIKES]-> (Coding)" in edges

def test_get_neighborhood_short_words_ignored(temp_graph):
    """Test that words under 3 characters are completely skipped in traversal."""
    temp_graph.add_triplets([
        {"subject": "it", "predicate": "IS", "object": "true"},
        {"subject": "he", "predicate": "GOES", "object": "there"}
    ])
    edges = temp_graph.get_neighborhood(["it", "he"])
    assert len(edges) == 0

def test_get_neighborhood_fuzzy_node_mapping(temp_graph):
    """Query words map to the closest matching exact node in the graph."""
    temp_graph.add_triplets([{"subject": "superman", "predicate": "FLIES", "object": "fast"}])
    # 'super' is not an exact node, but fuzzy/substring mapping lands on 'superman'.
    edges = temp_graph.get_neighborhood(["SUPER"])
    assert len(edges) == 1
    assert "- (Superman) -[FLIES]-> (Fast)" in edges

def test_delete_by_target(temp_graph):
    """Test dropping edges based on loose target matching."""
    temp_graph.add_triplets([
        {"subject": "superman", "predicate": "LIKES", "object": "lois lane"},
        {"subject": "batman", "predicate": "WORKS_IN", "object": "gotham"},
        {"subject": "lois lane", "predicate": "WORKS_AT", "object": "daily planet"},
    ])
    
    # Target "lois" should match 'lois lane' as a subject and object
    deleted = temp_graph.delete_by_target("Lois")
    assert deleted == 2
    
    # Verify remaining edges
    all_edges = temp_graph.get_neighborhood(["superman", "batman", "gotham", "lois lane", "daily planet"])
    assert len(all_edges) == 1
    assert "Batman" in all_edges[0]

def test_delete_by_target_short_word(temp_graph):
    """Test short words are rejected."""
    temp_graph.add_triplets([{"subject": "it", "predicate": "IS", "object": "true"}])
    deleted = temp_graph.delete_by_target("it")
    assert deleted == 0

def test_wipe_all(temp_graph):
    """Test obliterating the entire graph."""
    temp_graph.add_triplets([
        {"subject": "a", "predicate": "IS", "object": "b"},
        {"subject": "b", "predicate": "IS", "object": "c"}
    ])
    
    temp_graph.wipe_all()
    
    # Verify via neighborhood which scans the DB
    edges = temp_graph.get_neighborhood(["a", "b", "c"])
    assert len(edges) == 0
