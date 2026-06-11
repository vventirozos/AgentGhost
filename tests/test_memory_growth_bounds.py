"""Bounded-growth fixes from the 2026-06-11 memory audit.

1. VectorMemory now caps the open-ended prunable tiers (`auto` / `manual`)
   at MAX_PRUNABLE_MEMORIES, evicting the lowest-utility entries (fewest
   retrievals, then oldest). Documents / skills / episodes are exempt.
2. EpisodicMemory reaps orphaned `episode_actions` rows UNCONDITIONALLY on
   every record_episode — not only when an insert breaches the cap — so
   orphans from non-cap deletes can't accumulate.
"""

import sqlite3

from ghost_agent.memory.episodes import EpisodicMemory
from ghost_agent.memory.vector import VectorMemory

# --- 1. VectorMemory prune -------------------------------------------------


class _FakeCollection:
    """Minimal in-memory stand-in for a Chroma collection: enough surface
    (add/get-with-where/delete) for the prune path."""

    def __init__(self):
        self.docs = {}  # id -> (text, meta)

    def add(self, documents, metadatas, ids):
        for d, m, i in zip(documents, metadatas, ids):
            self.docs[i] = (d, dict(m))

    def get(self, ids=None, where=None, include=None):
        if ids is not None:
            keep = [i for i in ids if i in self.docs]
        else:
            keep = list(self.docs)
            if where and "type" in where and "$in" in where["type"]:
                allowed = set(where["type"]["$in"])
                keep = [i for i in keep if self.docs[i][1].get("type") in allowed]
        return {"ids": keep, "metadatas": [self.docs[i][1] for i in keep]}

    def delete(self, ids):
        for i in ids:
            self.docs.pop(i, None)

    def count(self):
        return len(self.docs)


def _bare_vector():
    vm = VectorMemory.__new__(VectorMemory)
    vm.collection = _FakeCollection()
    return vm


def test_prune_evicts_lowest_utility_over_cap(monkeypatch):
    monkeypatch.setattr(VectorMemory, "MAX_PRUNABLE_MEMORIES", 10)
    vm = _bare_vector()
    # 15 auto memories with ascending retrieval_count 0..14
    for n in range(15):
        vm.collection.add(
            documents=[f"memory number {n}"],
            metadatas=[
                {
                    "type": "auto",
                    "retrieval_count": n,
                    "last_accessed": f"2026-06-{n+1:02d}T00:00:00Z",
                }
            ],
            ids=[f"id{n}"],
        )
    with vm._get_lock():
        deleted = vm._prune_if_needed()
    assert deleted == 5
    survivors = set(vm.collection.get()["ids"])
    # the 10 highest retrieval_count survive (ids 5..14); 0..4 evicted
    assert survivors == {f"id{n}" for n in range(5, 15)}


def test_prune_never_touches_exempt_types(monkeypatch):
    monkeypatch.setattr(VectorMemory, "MAX_PRUNABLE_MEMORIES", 2)
    vm = _bare_vector()
    for n in range(5):
        vm.collection.add(
            documents=[f"doc {n}"],
            metadatas=[{"type": "document", "retrieval_count": 0}],
            ids=[f"doc{n}"],
        )
    for n in range(5):
        vm.collection.add(
            documents=[f"ep {n}"],
            metadatas=[{"type": "episode", "retrieval_count": 0}],
            ids=[f"ep{n}"],
        )
    with vm._get_lock():
        deleted = vm._prune_if_needed()
    assert deleted == 0
    assert vm.collection.count() == 10  # nothing exempt was evicted


def test_prune_noop_under_cap(monkeypatch):
    monkeypatch.setattr(VectorMemory, "MAX_PRUNABLE_MEMORIES", 100)
    vm = _bare_vector()
    vm.collection.add(documents=["x y z"], metadatas=[{"type": "auto"}], ids=["a"])
    with vm._get_lock():
        assert vm._prune_if_needed() == 0


def test_add_triggers_prune_after_interval(monkeypatch):
    monkeypatch.setattr(VectorMemory, "MAX_PRUNABLE_MEMORIES", 5)
    monkeypatch.setattr(VectorMemory, "_PRUNE_CHECK_EVERY", 3)
    vm = _bare_vector()
    # silence the pretty_log save line
    import ghost_agent.memory.vector as vec

    monkeypatch.setattr(vec, "pretty_log", lambda *a, **k: None)
    for n in range(12):
        vm.add(f"distinct memory text {n}")
    # cap is 5, and prune fires every 3 adds → population stays bounded
    assert vm.collection.count() <= 5


# --- 2. EpisodicMemory orphan reaping --------------------------------------


def _orphan_count(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM episode_actions WHERE episode_id NOT IN "
            "(SELECT id FROM episodes)"
        ).fetchone()[0]


def test_orphans_reaped_after_non_cap_delete(tmp_path):
    em = EpisodicMemory(tmp_path)
    eid = em.record_episode(
        trigger="do a thing",
        actions=[
            {"tool": "execute", "result": "ok", "success": True},
            {"tool": "browser", "result": "ok", "success": True},
        ],
    )
    # delete the parent episode directly (a non-cap delete path), leaving
    # its action rows orphaned
    with sqlite3.connect(em.db_path) as conn:
        conn.execute("DELETE FROM episodes WHERE id = ?", (eid,))
        conn.commit()
    assert _orphan_count(em.db_path) == 2  # orphans exist before next write

    # the very next record_episode must reap them, even though the table is
    # nowhere near MAX_EPISODES
    em.record_episode(
        trigger="another thing", actions=[{"tool": "execute", "result": "ok"}]
    )
    assert _orphan_count(em.db_path) == 0


def test_action_index_created(tmp_path):
    em = EpisodicMemory(tmp_path)
    with sqlite3.connect(em.db_path) as conn:
        idx = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
    assert "idx_ea_episode" in idx


def test_episode_cap_still_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(EpisodicMemory, "MAX_EPISODES", 10)
    em = EpisodicMemory(tmp_path)
    for n in range(25):
        em.record_episode(
            trigger=f"task {n}", actions=[{"tool": "execute", "result": "ok"}]
        )
    with sqlite3.connect(em.db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    assert count <= 10
    assert _orphan_count(em.db_path) == 0
