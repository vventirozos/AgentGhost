"""Data-loss regressions in the knowledge graph (verified against the live DB).

Seven defects, all of which silently destroyed operator knowledge:

1. `OWNS`/`IS` were treated as FUNCTIONAL predicates, so every new
   `user OWNS X` expired every other `user OWNS *` row — 19 of the 21 expired
   rows in the production graph were OWNS (BMW, Ducati, evolmonkey, webos,
   piggybag, nova, chess coach).
2. `delete_by_target` hard-DELETEd on an unanchored `LIKE '%t%'`:
   `forget("tin")` removed 83 rows, `forget("game")` ~182 (13% of the graph)
   once `get_connected_entities` re-amplified it through generic hubs.
3. Genuinely single-valued OPERATIONAL predicates (`HAS_STATUS`, `HAS_PID`)
   were missing from the whitelist — `chess-v4 HAS_STATUS dead` and `running`
   were both current.
4. Node compression never re-applied functional-predicate expiry, minting two
   mutually exclusive "current" facts on every subject-side merge.
5. A merge turned an edge BETWEEN the two merged nodes into a self-loop.
6. `_map_words_to_seeds`' substring rule re-seeded the ego hub
   ('aiohttp' -> 'ai', 'username' -> 'user', degree 234).
7. Expiry was completely silent.
"""
import logging
import sqlite3

import pytest

from ghost_agent.memory.graph import GraphMemory


@pytest.fixture
def gm(tmp_path):
    return GraphMemory(tmp_path)


def _objects(gm, subject, predicate, current_only=True):
    """Current objects for (subject, predicate), straight from SQLite."""
    q = ("SELECT object FROM triplets WHERE subject=? AND predicate=?"
         + (" AND valid_until IS NULL" if current_only else ""))
    with sqlite3.connect(gm.db_path) as conn:
        return sorted(r[0] for r in conn.execute(q, (subject, predicate)).fetchall())


def _insert(gm, subject, predicate, obj, weight=1, valid_from=None, valid_until=None):
    with sqlite3.connect(gm.db_path) as conn:
        conn.execute(
            "INSERT INTO triplets (subject, predicate, object, weight, valid_from, valid_until) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (subject, predicate, obj, weight, valid_from, valid_until))
        conn.commit()


# ---------------------------------------------------------------- 1. OWNS/IS


def test_owns_accumulates_instead_of_expiring(gm):
    """You own many things — a second OWNS must not expire the first."""
    gm.add_triplets([{"subject": "user", "predicate": "OWNS", "object": "bmw 118i"}])
    gm.add_triplets([{"subject": "user", "predicate": "OWNS", "object": "ducati streetfighter v4s"}])
    gm.add_triplets([{"subject": "user", "predicate": "OWNS", "object": "evolmonkey"}])

    assert _objects(gm, "user", "OWNS") == [
        "bmw 118i", "ducati streetfighter v4s", "evolmonkey"]
    assert gm.get_expired_triplets(subject="user") == []
    # ...and all three survive a reload (mirror only loads current edges).
    gm.initialize_graph()
    for obj in ("bmw 118i", "ducati streetfighter v4s", "evolmonkey"):
        assert gm.nx_graph.has_edge("user", obj)


def test_is_is_multi_valued(gm):
    """X IS many things — IS must not be functional either."""
    gm.add_triplets([{"subject": "webos", "predicate": "IS", "object": "a project"}])
    gm.add_triplets([{"subject": "webos", "predicate": "IS", "object": "done"}])
    assert _objects(gm, "webos", "IS") == ["a project", "done"]


def test_has_pet_and_other_multivalued_has_predicates_accumulate(gm):
    gm.add_triplets([{"subject": "user", "predicate": "HAS_PET", "object": "hanzo"}])
    gm.add_triplets([{"subject": "user", "predicate": "HAS_PET", "object": "mortimer"}])
    assert _objects(gm, "user", "HAS_PET") == ["hanzo", "mortimer"]


# ------------------------------------------------- 3. functional whitelist


def test_functional_predicate_still_expires(gm):
    """The genuinely one-to-one predicates keep their supersede semantics."""
    gm.add_triplets([{"subject": "bob", "predicate": "WORKS_AT", "object": "google"}])
    gm.add_triplets([{"subject": "bob", "predicate": "WORKS_AT", "object": "meta"}])
    assert _objects(gm, "bob", "WORKS_AT") == ["meta"]
    assert any(e["object"] == "google" for e in gm.get_expired_triplets(subject="bob"))


@pytest.mark.parametrize("predicate,old,new", [
    ("HAS_STATUS", "dead", "running"),
    ("STATUS", "stopped", "running"),
    ("HAS_PID", "595", "719"),
])
def test_operational_predicates_are_functional(gm, predicate, old, new):
    """Live contradiction: chess-v4 was both `dead` and `running`, pid 595 and 719."""
    gm.add_triplets([{"subject": "chess-v4", "predicate": predicate, "object": old}])
    gm.add_triplets([{"subject": "chess-v4", "predicate": predicate, "object": new}])
    assert _objects(gm, "chess-v4", predicate) == [new]


def test_initialize_graph_excludes_expired(gm):
    """Coverage carried over from `test_graph_temporal.py`, which asserted this
    using `IS` — a predicate that is (correctly) no longer functional. Same
    intent, re-pinned on a genuinely single-valued predicate.

    NOTE for the coordinator: `tests/test_graph_temporal.py::
    TestGraphTemporalReasoning::test_initialize_graph_excludes_expired` still
    uses `IS` and now fails by design; swapping its predicate to `LIVES_IN`
    (as here) restores it.
    """
    gm.add_triplets([{"subject": "x", "predicate": "LIVES_IN", "object": "old"}])
    gm.add_triplets([{"subject": "x", "predicate": "LIVES_IN", "object": "new"}])
    gm.initialize_graph(include_expired=False)
    nodes = list(gm.nx_graph.nodes())
    assert "old" not in nodes
    assert "new" in nodes


def test_multivalued_has_predicates_are_not_functional():
    """Guard the whitelist itself: nothing multi-valued may sneak back in."""
    forbidden = {"OWNS", "IS", "HAS_PET", "HAS_TASK", "HAS_FEATURE", "HAS_NAME",
                 "HAS_STATE", "HAS_FEN", "LIKES", "KNOWS", "HAS", "USES",
                 "CONTAINS", "HAS_PROJECT", "HAS_DESCRIPTION"}
    assert not (forbidden & GraphMemory._FUNCTIONAL_PREDICATES)


# ------------------------------------------------------- 7. expiry is logged


def test_expiry_is_logged(gm, caplog):
    gm.add_triplets([{"subject": "bob", "predicate": "LIVES_IN", "object": "london"}])
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        gm.add_triplets([{"subject": "bob", "predicate": "LIVES_IN", "object": "athens"}])
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "expiry" in msg and "london" in msg and "athens" in msg


# --------------------------------------------------------- 2. delete_by_target


def test_delete_by_target_does_not_nuke_substring_matches(gm):
    """`forget("tin")` must not eat `testing`, `printing`, `tinyai`, `martin`."""
    gm.add_triplets([
        {"subject": "user", "predicate": "RUNS", "object": "testing"},
        {"subject": "printer", "predicate": "DOES", "object": "printing"},
        {"subject": "tinyai", "predicate": "HAS_STATUS", "object": "archived"},
        {"subject": "martin", "predicate": "KNOWS", "object": "user"},
        {"subject": "user", "predicate": "USES", "object": "tin"},   # the real target
    ])
    deleted = gm.delete_by_target("tin")
    assert deleted == 1                       # exactly the real target
    with sqlite3.connect(gm.db_path) as conn:
        rows = conn.execute("SELECT subject, object FROM triplets").fetchall()
    assert ("user", "tin") not in rows
    assert sorted(rows) == [
        ("martin", "user"), ("printer", "printing"),
        ("tinyai", "archived"), ("user", "testing")]


def test_delete_by_target_still_forgets_the_real_entity(gm):
    """Forget is not a no-op: whole-token hits on either side still go."""
    gm.add_triplets([
        {"subject": "mortimer", "predicate": "IS_A", "object": "iguana"},
        {"subject": "user", "predicate": "HAS_PET", "object": "mortimer"},
        {"subject": "user", "predicate": "HAS_PET", "object": "hanzo"},
        {"subject": "vet", "predicate": "TREATED", "object": "mortimer the iguana"},
        {"subject": "user", "predicate": "LIKES", "object": "mortimers"},  # plural form
    ])
    deleted = gm.delete_by_target("mortimer")
    assert deleted == 4
    assert "mortimer" not in gm.nx_graph
    assert gm.nx_graph.has_edge("user", "hanzo")


def test_delete_by_target_short_target_rejected(gm):
    gm.add_triplets([{"subject": "it", "predicate": "IS", "object": "true"}])
    assert gm.delete_by_target("it") == 0


def test_huge_forget_expires_instead_of_hard_deleting(gm):
    """Blast-radius guard: a forget big enough to gut the graph is made undoable."""
    gm.add_triplets(
        [{"subject": "chess", "predicate": "HAS_MOVE", "object": f"move{i}"}
         for i in range(60)]
        + [{"subject": "user", "predicate": "LIKES", "object": "python"}]
    )
    n = gm.delete_by_target("chess")
    assert n == 60
    with sqlite3.connect(gm.db_path) as conn:
        # Rows are retained but expired => invisible to every read path...
        assert conn.execute("SELECT COUNT(*) FROM triplets").fetchone()[0] == 61
        assert conn.execute(
            "SELECT COUNT(*) FROM triplets WHERE valid_until IS NULL").fetchone()[0] == 1
    assert "chess" not in gm.nx_graph
    assert gm.get_neighborhood(["chess"]) == []
    assert not any(t["subject"] == "chess" for t in gm.get_recent_triplets())
    # ...and recoverable.
    assert len(gm.get_expired_triplets(subject="chess", limit=100)) == 60


# ------------------------------------------- 2b. neighbour-expansion amplifier


def test_expansion_still_reaches_a_tight_alias(gm):
    gm.add_triplets([
        {"subject": "user", "predicate": "HAS_PET", "object": "mortimer"},
        {"subject": "mortimer", "predicate": "IS_A", "object": "iguana"},
    ])
    related = gm.get_connected_entities("mortimer")
    assert "iguana" in related
    assert "user" not in related and "mortimer" not in related


def test_expansion_skips_generic_hubs(gm):
    """`assistant`/`project`/`system`/`done` are hubs in the live graph; a
    forget must never expand through them."""
    gm.add_triplets([
        {"subject": "assistant", "predicate": "BUILT", "object": "chessbot"},
        {"subject": "chessbot", "predicate": "PART_OF", "object": "project"},
        {"subject": "chessbot", "predicate": "HAS_STATUS", "object": "done"},
        {"subject": "system", "predicate": "HOSTS", "object": "chessbot"},
        {"subject": "chessbot", "predicate": "IS_A", "object": "engine"},
    ])
    related = gm.get_connected_entities("chessbot")
    assert related == ["engine"]


def test_expansion_skips_high_degree_neighbours(gm):
    """Hub-by-measurement, not just by name: forgetting a satellite must not
    drag in a node the whole graph hangs off."""
    gm.add_triplets(
        [{"subject": "webos", "predicate": "HAS_FEATURE", "object": f"feat{i}"}
         for i in range(12)]
        + [{"subject": "sidecar", "predicate": "TALKS_TO", "object": "webos"},
           {"subject": "sidecar", "predicate": "IS_A", "object": "daemon"}]
    )
    related = gm.get_connected_entities("sidecar")
    assert "webos" not in related
    assert "daemon" in related


def test_expansion_ignores_substring_neighbours(gm):
    gm.add_triplets([
        {"subject": "testing", "predicate": "USES", "object": "pytest"},
        {"subject": "tin", "predicate": "IS_A", "object": "metal"},
    ])
    assert gm.get_connected_entities("tin") == ["metal"]


def test_expansion_ignores_expired_edges(gm):
    gm.add_triplets([{"subject": "bob", "predicate": "LIVES_IN", "object": "london"}])
    gm.add_triplets([{"subject": "bob", "predicate": "LIVES_IN", "object": "athens"}])
    assert gm.get_connected_entities("bob") == ["athens"]


# ---------------------------------------- 4./5. compression: conflicts + loops


def test_merge_reconciles_functional_conflict(gm):
    """bobby -> bob must not leave `bob WORKS_AT google` AND `meta` current."""
    _insert(gm, "bob", "WORKS_AT", "google", valid_from=100.0)
    _insert(gm, "bobby", "WORKS_AT", "meta", valid_from=200.0)
    gm.execute_graph_compression([{"old_node": "bobby", "new_node": "bob"}])

    assert _objects(gm, "bob", "WORKS_AT") == ["meta"]           # newest wins
    assert _objects(gm, "bob", "WORKS_AT", current_only=False) == ["google", "meta"]
    assert any(e["object"] == "google" for e in gm.get_expired_triplets(subject="bob"))
    assert not gm.nx_graph.has_edge("bob", "google")
    assert gm.nx_graph.has_edge("bob", "meta")


def test_merge_leaves_multivalued_predicates_alone(gm):
    """The reconciliation must not become a second OWNS-style shredder."""
    _insert(gm, "bob", "OWNS", "bmw", valid_from=100.0)
    _insert(gm, "bobby", "OWNS", "ducati", valid_from=200.0)
    gm.execute_graph_compression([{"old_node": "bobby", "new_node": "bob"}])
    assert _objects(gm, "bob", "OWNS") == ["bmw", "ducati"]


def test_merge_drops_edge_between_the_merged_nodes(gm):
    """`new-york SAME_AS new york` must not survive as a self-loop."""
    gm.add_triplets([
        {"subject": "new-york", "predicate": "SAME_AS", "object": "new york"},
        {"subject": "new york", "predicate": "IN", "object": "usa"},
    ])
    gm.execute_graph_compression([{"old_node": "new-york", "new_node": "new york"}])

    with sqlite3.connect(gm.db_path) as conn:
        rows = conn.execute("SELECT subject, predicate, object FROM triplets").fetchall()
    assert ("new york", "SAME_AS", "new york") not in rows
    assert rows == [("new york", "IN", "usa")]
    assert not gm.nx_graph.has_edge("new york", "new york")
    # No neighbourhood duplication from a phantom loop.
    assert gm.get_neighborhood(["new york"]) == ["- (New York) -[IN]-> (Usa)"]


def test_merge_keeps_genuine_self_loop(gm):
    """A real old->old self-loop still migrates to new->new (pre-existing rule)."""
    _insert(gm, "ny", "same_as", "ny", weight=2)
    gm.execute_graph_compression([{"old_node": "ny", "new_node": "new york"}])
    with sqlite3.connect(gm.db_path) as conn:
        rows = conn.execute("SELECT subject, predicate, object FROM triplets").fetchall()
    assert rows == [("new york", "same_as", "new york")]


# ------------------------------------------------------------ 6. seed mapping


def test_seed_mapping_does_not_reseed_ego_hub(gm):
    gm.add_triplets([
        {"subject": "ai", "predicate": "IS_A", "object": "field"},
        {"subject": "user", "predicate": "LIKES", "object": "python"},
        {"subject": "system", "predicate": "HAS_STATUS", "object": "healthy"},
    ])
    assert gm._map_words_to_seeds(["aiohttp"]) == []
    assert gm._map_words_to_seeds(["username"]) == []
    assert gm._map_words_to_seeds(["systemd"]) == []
    # An exact query word still seeds normally.
    assert gm._map_words_to_seeds(["user"]) == ["user"]


def test_seed_mapping_keeps_narrowing_substring_matches(gm):
    """The safe direction (word inside a longer node name) is untouched."""
    gm.add_triplets([
        {"subject": "berlin", "predicate": "CAPITAL_OF", "object": "germany"},
        {"subject": "superman", "predicate": "FLIES", "object": "fast"},
    ])
    assert "germany" in gm._map_words_to_seeds(["germ"])
    assert "superman" in gm._map_words_to_seeds(["super"])
    assert "berlin" in gm._map_words_to_seeds(["berln"])   # difflib fallback


def test_seed_mapping_allows_close_fragment(gm):
    """A fragment that is nearly the whole word is still a legitimate seed."""
    gm.add_triplets([{"subject": "postgres", "predicate": "IS_A", "object": "database"}])
    assert gm._map_words_to_seeds(["postgresql"]) == ["postgres"]
