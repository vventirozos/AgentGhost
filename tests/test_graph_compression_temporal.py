"""Graph compression must preserve temporal state and self-loops.

`execute_graph_compression` merges a node `old_node -> new_node` (entity dedup).
The prior version SELECTed rows regardless of `valid_until` and re-INSERTed them
with `valid_from = now` and no `valid_until`, so a *superseded* (expired) fact
came back as CURRENT; and an `old -> old` self-loop was rewritten subject-only to
`new -> old`, then deleted by the `object = old_node` sweep. These tests pin the
corrected behaviour: temporal state carries through, self-loops survive as
`new -> new`, weights sum without double-counting, and a temporal merge is
current-wins. The method is currently unwired (only a no-op stub calls it in
dream.py) — this locks correctness in before it is wired.
"""
import sqlite3

import pytest

from ghost_agent.memory.graph import GraphMemory


@pytest.fixture
def gm(tmp_path):
    return GraphMemory(tmp_path)


def _insert(gm, subject, predicate, obj, weight=1, valid_from=None, valid_until=None):
    with sqlite3.connect(gm.db_path) as conn:
        conn.execute(
            "INSERT INTO triplets (subject, predicate, object, weight, valid_from, valid_until) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (subject, predicate, obj, weight, valid_from, valid_until))
        conn.commit()


def _row(gm, subject, predicate, obj):
    with sqlite3.connect(gm.db_path) as conn:
        cur = conn.execute(
            "SELECT weight, valid_from, valid_until FROM triplets "
            "WHERE subject=? AND predicate=? AND object=?",
            (subject, predicate, obj))
        return cur.fetchone()


def test_expired_fact_not_resurrected(gm):
    """A superseded fact under old_node stays superseded under new_node."""
    _insert(gm, "bob", "was_ceo_of", "acme", valid_from=1.0, valid_until=1000.0)
    gm.execute_graph_compression([{"old_node": "bob", "new_node": "robert"}])
    row = _row(gm, "robert", "was_ceo_of", "acme")
    assert row is not None, "fact should migrate to the new node"
    assert row[2] == 1000.0, "expired fact must NOT come back as current"
    assert _row(gm, "bob", "was_ceo_of", "acme") is None


def test_current_fact_stays_current(gm):
    _insert(gm, "bob", "lives_in", "berlin", valid_from=1.0, valid_until=None)
    gm.execute_graph_compression([{"old_node": "bob", "new_node": "robert"}])
    row = _row(gm, "robert", "lives_in", "berlin")
    assert row is not None and row[2] is None


def test_self_loop_survives_as_new_new(gm):
    """old -> old must migrate to new -> new, not be dropped."""
    _insert(gm, "ny", "same_as", "ny", weight=2, valid_until=None)
    gm.execute_graph_compression([{"old_node": "ny", "new_node": "new york"}])
    assert _row(gm, "new york", "same_as", "new york") is not None
    assert _row(gm, "ny", "same_as", "ny") is None
    assert _row(gm, "new york", "same_as", "ny") is None
    assert "new york" in gm.nx_graph
    assert "ny" not in gm.nx_graph


def test_weight_merges_without_double_count(gm):
    """Source current + pre-existing target current -> weights sum exactly once."""
    _insert(gm, "ny", "in", "usa", weight=3)
    _insert(gm, "new york", "in", "usa", weight=5)
    gm.execute_graph_compression([{"old_node": "ny", "new_node": "new york"}])
    row = _row(gm, "new york", "in", "usa")
    assert row is not None and row[0] == 8
    assert _row(gm, "ny", "in", "usa") is None


def test_current_wins_when_merging_expired_into_current(gm):
    """Existing target current, incoming source expired -> result stays current."""
    _insert(gm, "new york", "in", "usa", weight=1, valid_until=None)   # current
    _insert(gm, "ny", "in", "usa", weight=1, valid_until=2000.0)        # expired source
    gm.execute_graph_compression([{"old_node": "ny", "new_node": "new york"}])
    row = _row(gm, "new york", "in", "usa")
    assert row is not None
    assert row[2] is None, "current wins over an expired merge partner"
    assert row[0] == 2


def test_both_expired_keeps_later_expiry_and_earliest_from(gm):
    _insert(gm, "new york", "in", "usa", weight=1, valid_from=1.0, valid_until=1000.0)
    _insert(gm, "ny", "in", "usa", weight=1, valid_from=2.0, valid_until=3000.0)
    gm.execute_graph_compression([{"old_node": "ny", "new_node": "new york"}])
    row = _row(gm, "new york", "in", "usa")
    assert row is not None
    assert row[2] == 3000.0, "later expiry kept when both expired"
    assert row[1] == 1.0, "earliest valid_from kept"


def test_object_side_rewrite_preserves_expiry(gm):
    """old_node appearing as the OBJECT of an expired edge also stays expired."""
    _insert(gm, "alice", "worked_at", "ny", valid_from=1.0, valid_until=500.0)
    gm.execute_graph_compression([{"old_node": "ny", "new_node": "new york"}])
    row = _row(gm, "alice", "worked_at", "new york")
    assert row is not None and row[2] == 500.0
    assert _row(gm, "alice", "worked_at", "ny") is None
