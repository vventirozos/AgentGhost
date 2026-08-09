"""F8 — episodic usage credit + value-weighted eviction.

Problem this pins (from the §4M Lens-D F8 finding): episodic usage credit
landed ONLY on the vector twin's retrieval counter, which lives in the vector
store. Episodic eviction is plain SQL over the episodes table and structurally
cannot read that counter, so it evicted by pure AGE — a spent episode surfaced
and useful a dozen times was discarded at exactly the same priority as one
never retrieved.

The fix adds `access_count` / `last_accessed` columns to the episodes table
(authoritative for EVICTION; the vector counter stays authoritative for
retrieval stats), writes them from every path that surfaces an episode, and
orders eviction least-valuable-first.

Each test is red-on-revert of a specific piece: drop the ORDER BY change and
`test_used_old_episode_survives_unused_newer_one` fails; drop the
credit call on either path and that path's credit test fails; drop the
INSERT's last_accessed seeding and `test_new_episode_not_born_evict_first`
fails; drop the backfill and `test_migration_backfills_last_accessed` fails.
"""

import sqlite3
import time
from contextlib import closing

import pytest

from ghost_agent.memory.episodes import EpisodicMemory


def _cols(db_path):
    with closing(sqlite3.connect(db_path)) as c:
        return {r[1] for r in c.execute("PRAGMA table_info(episodes)")}


def _rows(db_path, sql="SELECT id, timestamp, access_count, last_accessed "
                       "FROM episodes ORDER BY id"):
    with closing(sqlite3.connect(db_path)) as c:
        return c.execute(sql).fetchall()


def _make_legacy_db(tmp_path, n=3):
    """Create a PRE-MIGRATION episodes DB (old schema, no usage columns)."""
    db = tmp_path / "episodic_memory.db"
    with closing(sqlite3.connect(db)) as c:
        c.execute(
            """CREATE TABLE episodes (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   trigger TEXT NOT NULL, context TEXT DEFAULT '',
                   outcome TEXT DEFAULT '', outcome_success INTEGER DEFAULT 0,
                   lesson TEXT DEFAULT '', cluster_id TEXT DEFAULT '',
                   timestamp REAL NOT NULL, consolidated INTEGER DEFAULT 0)"""
        )
        now = time.time()
        for i in range(n):
            c.execute(
                "INSERT INTO episodes (trigger, timestamp, consolidated) "
                "VALUES (?, ?, 1)", (f"legacy episode {i}", now - 10000 + i))
        c.commit()
    return db


# ── migration ─────────────────────────────────────────────────────────────

def test_migration_adds_columns_to_existing_store(tmp_path):
    db = _make_legacy_db(tmp_path)
    assert "access_count" not in _cols(db)
    EpisodicMemory(tmp_path)
    assert {"access_count", "last_accessed"} <= _cols(db)


def test_migration_is_idempotent(tmp_path):
    _make_legacy_db(tmp_path)
    EpisodicMemory(tmp_path)
    # A second construction re-runs _init_db; the PRAGMA guard must make the
    # ALTERs a no-op rather than raising "duplicate column name".
    EpisodicMemory(tmp_path)
    em = EpisodicMemory(tmp_path)
    assert em.count_episodes() == 3 if hasattr(em, "count_episodes") else True


def test_migration_backfills_last_accessed(tmp_path):
    """Pre-migration rows must seed last_accessed from timestamp. Left at the
    0 default they would ALL sort below every new row on eviction's second
    key — the migration itself would make the whole existing store
    evict-first."""
    db = _make_legacy_db(tmp_path)
    EpisodicMemory(tmp_path)
    rows = _rows(db)
    assert rows, "expected legacy rows to survive migration"
    for _id, ts, access, last in rows:
        assert access == 0
        assert last == pytest.approx(ts), "last_accessed not backfilled from timestamp"


def test_new_episode_not_born_evict_first(tmp_path):
    """A brand-new episode must seed last_accessed at insert time, not 0."""
    em = EpisodicMemory(tmp_path)
    eid = em.record_episode("fresh episode about ingress controllers")
    row = em.get_episode(eid)
    assert row["access_count"] == 0
    assert row["last_accessed"] > 0
    assert row["last_accessed"] == pytest.approx(row["timestamp"])


# ── credit recording ──────────────────────────────────────────────────────

def test_substring_path_credits_surfaced_episodes(tmp_path):
    """The substring fallback surfaces episodes too — episodes reachable ONLY
    by substring (missing vector twin) must not stay at zero credit."""
    em = EpisodicMemory(tmp_path)
    hit = em.record_episode("kubernetes ingress deployment rollout")
    miss = em.record_episode("completely unrelated pasta recipe")
    results = em.search_similar("kubernetes ingress", limit=5)  # no vector_memory
    assert any(r["id"] == hit for r in results)
    assert em.get_episode(hit)["access_count"] == 1
    assert em.get_episode(miss)["access_count"] == 0, "credited a non-surfaced episode"


def test_vector_path_credits_and_is_not_gated_on_scoped_query(tmp_path):
    """Credit must fire on the search_advanced FALLBACK path too. The
    `scoped_query` gate exists only to stop the VECTOR counter being double
    bumped; nothing else writes episodes.access_count, so inheriting that gate
    would leave fallback-path episodes evict-first forever."""
    em = EpisodicMemory(tmp_path)
    eid = em.record_episode("deploy the ingress controller")
    other = em.record_episode("unrelated topic entirely")

    class FakeVec:  # no .collection → scoped path returns None → fallback
        def search_advanced(self, q, limit=10, record_retrievals=True):
            return [{"id": "v1",
                     "metadata": {"type": "episode", "episode_id": eid},
                     "score": 0.1}]

    results = em.search_similar("ingress", limit=5, vector_memory=FakeVec())
    assert any(r["id"] == eid for r in results)
    assert em.get_episode(eid)["access_count"] == 1
    assert em.get_episode(other)["access_count"] == 0


def test_credit_helper_never_raises_on_bad_input(tmp_path):
    em = EpisodicMemory(tmp_path)
    for bad in (None, [], ["not-an-int"], [None]):
        em._credit_surfaced_episodes(bad)  # must not raise


def test_one_bad_id_does_not_drop_credit_for_the_rest(tmp_path):
    """Per-item coercion: a malformed id in the batch must not turn a partial
    failure into a total one (the other episodes still earn their credit)."""
    em = EpisodicMemory(tmp_path)
    good = em.record_episode("a perfectly good episode")
    em._credit_surfaced_episodes([None, good, "nonsense"])
    assert em.get_episode(good)["access_count"] == 1


# ── value-weighted eviction (the point of F8) ─────────────────────────────

def _spend_all(db_path):
    """Mark every row consolidated + lesson-free so it lands in the
    'spent' eviction tier."""
    with closing(sqlite3.connect(db_path)) as c:
        c.execute("UPDATE episodes SET consolidated = 1, lesson = ''")
        c.commit()


def test_used_old_episode_survives_unused_newer_one(tmp_path):
    """THE F8 inversion. Under pure-age eviction the OLDEST row is always the
    victim; with usage credit the oldest-but-repeatedly-useful row survives
    and the least-valuable (oldest zero-credit) row is evicted instead."""
    em = EpisodicMemory(tmp_path)
    em.MAX_EPISODES = 4
    ids = []
    for i in range(4):
        ids.append(em.record_episode(f"spent episode number {i}"))
        time.sleep(0.01)
    _spend_all(tmp_path / "episodic_memory.db")

    for _ in range(5):                       # the OLDEST one keeps proving useful
        em._credit_surfaced_episodes([ids[0]])

    em.record_episode("a fresh new episode")  # breaches the cap → evict one

    with closing(sqlite3.connect(tmp_path / "episodic_memory.db")) as c:
        alive = {r[0] for r in c.execute("SELECT id FROM episodes")}
    assert ids[0] in alive, "the oldest-but-USED episode was evicted (pure-age behaviour)"
    assert ids[1] not in alive, "expected the oldest zero-credit episode to be the victim"


def test_cap_still_hard_enforced_when_every_row_is_hot(tmp_path):
    """Value ordering must not defeat the cap: when every row has credit the
    fallback tier still evicts (the §4M unbounded-growth fix stays intact)."""
    em = EpisodicMemory(tmp_path)
    em.MAX_EPISODES = 3
    ids = [em.record_episode(f"hot episode {i}") for i in range(3)]
    for eid in ids:                          # everything is used
        em._credit_surfaced_episodes([eid])
    em.record_episode("one more episode")

    with closing(sqlite3.connect(tmp_path / "episodic_memory.db")) as c:
        count = c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    assert count <= em.MAX_EPISODES, f"cap not enforced: {count} rows"


def test_fallback_tier_also_orders_by_value(tmp_path):
    """The cap block has TWO tiers: spent-rows-first, then an any-kind
    fallback that guarantees the cap is enforced. Both must order by value —
    pinning only the spent tier would leave the fallback evicting by pure age
    for every store whose rows aren't consolidated (the default shape).

    Unconsolidated rows never match the spent tier, so this store evicts
    entirely through the fallback path.
    """
    em = EpisodicMemory(tmp_path)
    em.MAX_EPISODES = 3
    ids = []
    for i in range(3):
        ids.append(em.record_episode(f"unconsolidated episode {i}"))
        time.sleep(0.01)
    # Nothing is consolidated → the spent tier finds no victims at all.
    with closing(sqlite3.connect(tmp_path / "episodic_memory.db")) as c:
        spent = c.execute("SELECT COUNT(*) FROM episodes "
                          "WHERE lesson = '' AND consolidated = 1").fetchone()[0]
    assert spent == 0, "fixture no longer exercises the fallback tier"

    for _ in range(5):                       # oldest row is repeatedly useful
        em._credit_surfaced_episodes([ids[0]])
    em.record_episode("newcomer")            # breaches cap → fallback tier evicts

    with closing(sqlite3.connect(tmp_path / "episodic_memory.db")) as c:
        alive = {r[0] for r in c.execute("SELECT id FROM episodes")}
    assert ids[0] in alive, "fallback tier evicted the oldest-but-USED episode (pure age)"
    assert ids[1] not in alive, "expected the oldest zero-credit row as fallback victim"


def test_eviction_prefers_least_recently_surfaced_at_equal_credit(tmp_path):
    """Tie-break: equal access_count → the one surfaced LONGEST ago goes."""
    em = EpisodicMemory(tmp_path)
    em.MAX_EPISODES = 3
    ids = [em.record_episode(f"episode {i}") for i in range(3)]
    _spend_all(tmp_path / "episodic_memory.db")
    # Same credit count, different recency: credit ids[2] then ids[1] (ids[0]
    # stays the least-recently-surfaced of the three).
    for eid in (ids[0], ids[2], ids[1]):
        em._credit_surfaced_episodes([eid])
        time.sleep(0.01)
    em.record_episode("newcomer")

    with closing(sqlite3.connect(tmp_path / "episodic_memory.db")) as c:
        alive = {r[0] for r in c.execute("SELECT id FROM episodes")}
    assert ids[0] not in alive, "expected the least-recently-surfaced to be evicted"
    assert ids[1] in alive and ids[2] in alive
