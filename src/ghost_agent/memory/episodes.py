"""Episodic memory — structured episode storage.

Unlike vector memory (facts) and graph memory (relationships), episodic
memory stores *sequences of actions with their outcomes*. This lets the
agent recall "last time I tried X, I did Y and it failed because Z, so
I switched to W" — causal chains, not just isolated facts.

Episodes are indexed by trigger similarity (vector search on trigger +
context) and can be consolidated by the dream cycle into generalized
strategies while keeping the best exemplar intact.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.helpers import get_utc_timestamp

logger = logging.getLogger("GhostAgent")


class EpisodicMemory:
    """SQLite-backed episodic memory with vector-searchable triggers.

    Each episode is a structured record:
    - trigger: what initiated the episode (user request / event)
    - context: relevant state at the time
    - actions: sequence of tool calls with results
    - outcome: success/failure + why
    - lesson: extracted heuristic (if any)
    - cluster_id: for similarity grouping and consolidation
    """

    MAX_EPISODES = 500
    MAX_ACTIONS_PER_EPISODE = 20

    # Action truncation keeps the HEAD *and* the TAIL. The whole point of this
    # module is "I tried X, it failed because Z, so I switched to W" — the
    # RESOLUTION lives in the last actions. Keeping `actions[:20]` amputated
    # exactly that (16 of 139 live episodes sat at exactly 20 actions), so the
    # stored record showed only the failed exploration while the episode row
    # still asserted outcome_success=1. We keep the first
    # TRUNCATION_HEAD_ACTIONS (the opening approach), one marker row saying how
    # much was dropped, and the rest of the budget from the tail.
    TRUNCATION_HEAD_ACTIONS = 5
    TRUNCATION_MARKER_TOOL = "(actions elided)"

    # How deep the substring fallback scans. Was a hardcoded 100 against a
    # MAX_EPISODES=500 store, so everything older than the newest 100 episodes
    # was unreachable by the fallback and — when its vector twin was missing —
    # unreachable by ANY path (12 such episodes live).
    FALLBACK_SCAN_LIMIT = MAX_EPISODES

    # Minimum distance-derived relevance an episode vector hit must clear to be
    # returned. Previously every vector hit was stamped relevance_score = 1.0
    # regardless of distance, so semantically unrelated episodes were injected
    # as if they were perfect matches. relevance = clamp(1.0 - distance, 0, 1);
    # hits below this floor are dropped. Tunable.
    MIN_VECTOR_RELEVANCE = 0.2

    # Recency shaping, applied identically to BOTH recall paths so their
    # `relevance_score` values are on one comparable [0, 1] scale (consumers
    # rank across paths). The base score dominates; recency only breaks ties:
    #   score = base * (1 - RECENCY_WEIGHT + RECENCY_WEIGHT * decay)
    # with decay = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS) and decay = 1.0
    # when the row carries no usable timestamp.
    RECENCY_HALF_LIFE_DAYS = 14.0
    RECENCY_WEIGHT = 0.15

    # Textual tells that a SUCCESSFUL episode got there by recovering from
    # trouble. Used by `search_recoveries` for episodes whose action rows carry
    # no failure flag (see `_recovery_evidence`).
    RECOVERY_OUTCOME_MARKERS = (
        "recover", "retry", "retried", "retrying", "instead", "fallback",
        "fell back", "workaround", "worked around", "root cause", "turned out",
        "second attempt", "eventually", "after failing", "the bug", "the fix",
        "fixed", "broke", "broken", "failed", "error",
    )

    def __init__(self, memory_dir: Path):
        self.db_path = memory_dir / "episodic_memory.db"
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS episodes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trigger TEXT NOT NULL,
                        context TEXT DEFAULT '',
                        outcome TEXT DEFAULT '',
                        outcome_success INTEGER DEFAULT 0,
                        lesson TEXT DEFAULT '',
                        cluster_id TEXT DEFAULT '',
                        timestamp REAL NOT NULL,
                        consolidated INTEGER DEFAULT 0,
                        access_count INTEGER NOT NULL DEFAULT 0,
                        last_accessed REAL NOT NULL DEFAULT 0
                    )
                ''')
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS episode_actions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        episode_id INTEGER NOT NULL,
                        action_order INTEGER NOT NULL,
                        tool_name TEXT NOT NULL,
                        tool_args TEXT DEFAULT '',
                        result TEXT DEFAULT '',
                        success INTEGER DEFAULT 1,
                        FOREIGN KEY (episode_id) REFERENCES episodes(id)
                    )
                ''')
                # F8 usage-credit migration (2026-08-08). The columns above are
                # in the CREATE TABLE for fresh stores; an EXISTING store
                # predates them, and SQLite has no `ADD COLUMN IF NOT EXISTS`
                # — so read the live column set and add only what's missing.
                # `ADD COLUMN` with a constant DEFAULT is metadata-only in
                # SQLite (no table rewrite), so this is safe on the live store.
                # Idempotent: the PRAGMA check makes a second _init_db a no-op.
                _cols = {r[1] for r in conn.execute("PRAGMA table_info(episodes)")}
                if "access_count" not in _cols:
                    conn.execute("ALTER TABLE episodes ADD COLUMN "
                                 "access_count INTEGER NOT NULL DEFAULT 0")
                if "last_accessed" not in _cols:
                    conn.execute("ALTER TABLE episodes ADD COLUMN "
                                 "last_accessed REAL NOT NULL DEFAULT 0")
                    # BACKFILL, and why it matters: eviction orders by
                    # (access_count, last_accessed, timestamp). Leaving
                    # pre-migration rows at last_accessed=0 would rank EVERY
                    # existing episode below every new one on the second key —
                    # i.e. the migration itself would make the whole existing
                    # store evict-first. Seeding from `timestamp` means "never
                    # surfaced since we started counting" sorts as old, which
                    # is exactly what it means.
                    conn.execute("UPDATE episodes SET last_accessed = timestamp")
                conn.execute('CREATE INDEX IF NOT EXISTS idx_ep_trigger ON episodes(trigger)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_ep_cluster ON episodes(cluster_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_ep_ts ON episodes(timestamp)')
                # Backs the value-weighted eviction ORDER BY (F8). Created
                # AFTER the migration above so the columns exist on an
                # upgraded store.
                conn.execute('CREATE INDEX IF NOT EXISTS idx_ep_value '
                             'ON episodes(access_count, last_accessed)')
                # Supports the unconditional orphan-reap anti-join in
                # record_episode (episode_actions has no enforced FK cascade).
                conn.execute('CREATE INDEX IF NOT EXISTS idx_ea_episode ON episode_actions(episode_id)')
                conn.commit()

    def _truncate_actions(self, actions: List[Dict[str, Any]],
                          episode_id: int) -> List[Dict[str, Any]]:
        """Clamp an action sequence to MAX_ACTIONS_PER_EPISODE, keeping the
        HEAD *and* the TAIL.

        The old ``actions[:MAX]`` kept the opening exploration and dropped the
        recovery — the exact half the episode exists to remember. We keep
        ``TRUNCATION_HEAD_ACTIONS`` from the front (how the agent came at the
        problem), one marker row recording how many were dropped, and the rest
        of the budget from the END (what actually resolved it).

        The marker row is stored with ``success = 1`` on purpose so it never
        counts as a failed action in the recovery signal
        (:meth:`_recovery_evidence`).
        """
        total = len(actions)
        if total <= self.MAX_ACTIONS_PER_EPISODE:
            return list(actions)
        head_n = min(self.TRUNCATION_HEAD_ACTIONS, self.MAX_ACTIONS_PER_EPISODE - 2)
        if head_n < 0:
            head_n = 0
        tail_n = self.MAX_ACTIONS_PER_EPISODE - head_n - 1  # 1 slot for the marker
        dropped = total - head_n - tail_n
        logger.warning(
            "Episode %s: %d actions exceed the cap of %d — keeping the first %d "
            "and the last %d, eliding %d in the middle (the tail holds the "
            "resolution, so it is never dropped).",
            episode_id, total, self.MAX_ACTIONS_PER_EPISODE, head_n, tail_n, dropped,
        )
        marker = {
            "tool": self.TRUNCATION_MARKER_TOOL,
            "args": {"elided": dropped, "total": total},
            "result": f"... {dropped} intermediate actions elided ...",
            "success": True,
        }
        return list(actions[:head_n]) + [marker] + list(actions[-tail_n:])

    def record_episode(self, trigger: str, context: str = "",
                       actions: List[Dict[str, Any]] = None,
                       outcome: str = "", success: bool = False,
                       lesson: str = "", cluster_id: str = "",
                       vector_memory=None) -> int:
        """Store a new episode. Returns the episode ID.

        When *vector_memory* is provided, the episode's trigger (and lesson,
        if any) is also indexed into the vector store with
        ``{"type":"episode","episode_id":...}`` metadata so the semantic
        recall path in :meth:`search_similar` / :meth:`_vector_search` can
        map vector hits back to episode rows. Episodes evicted by the
        capacity cap have their vector entries removed too, so the index
        stays bounded.
        """
        evicted_ids: List[int] = []
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                # `last_accessed` is seeded to the insert timestamp, NOT left at
                # its 0 default (F8): eviction's second sort key is
                # last_accessed, so a brand-new never-surfaced episode born at 0
                # would rank below every pre-existing row and be evicted first —
                # the exact inversion this feature exists to remove.
                _now = time.time()
                cursor = conn.execute(
                    '''INSERT INTO episodes (trigger, context, outcome, outcome_success,
                       lesson, cluster_id, timestamp, last_accessed)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (trigger[:500], context[:2000], outcome[:1000],
                     1 if success else 0, lesson[:500], cluster_id, _now, _now)
                )
                episode_id = cursor.lastrowid

                if actions:
                    kept = self._truncate_actions(actions, episode_id)
                    for i, action in enumerate(kept):
                        conn.execute(
                            '''INSERT INTO episode_actions
                               (episode_id, action_order, tool_name, tool_args, result, success)
                               VALUES (?, ?, ?, ?, ?, ?)''',
                            (episode_id, i,
                             action.get("tool", "unknown")[:100],
                             json.dumps(action.get("args", {}), default=str)[:500],
                             str(action.get("result", ""))[:1000],
                             1 if action.get("success", True) else 0)
                        )

                # Enforce max capacity. We SELECT the victims before deleting
                # (rather than DELETE … WHERE id IN (subquery)) so their ids can
                # be handed to the vector store for orphan removal below.
                count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                if count > self.MAX_EPISODES:
                    # Prefer deleting oldest SPENT episodes: consolidated,
                    # no lesson attached. The old preference was inverted —
                    # it deleted `consolidated = 0` rows, i.e. the dream
                    # loop's PENDING INPUT (live: 163/165 consolidated, so
                    # at cap every insert would have evicted the freshest
                    # experience while weeks-old spent rows fossilized, and
                    # the pending pool could never reach _consolidate's
                    # min_episodes=3 again).
                    # F8: LEAST-VALUABLE-FIRST within the tier, not pure age.
                    # Usage credit used to land only on the vector twin's
                    # counter, which this SQL cannot read — so a spent episode
                    # surfaced (and useful) a dozen times was evicted at exactly
                    # the same priority as one never retrieved. Ordering by
                    # (access_count, last_accessed, timestamp) is a TOTAL order
                    # that keeps the cap hard-enforceable while letting a proven
                    # old episode outlive a never-used newer one. Deliberately
                    # lexicographic rather than a decay score: no half-life
                    # constant to tune (and mis-tune).
                    victims = [r[0] for r in conn.execute(
                        '''SELECT id FROM episodes
                           WHERE lesson = '' AND consolidated = 1
                           ORDER BY access_count ASC, last_accessed ASC,
                                    timestamp ASC LIMIT ?''',
                        (count - self.MAX_EPISODES,)
                    ).fetchall()]
                    if victims:
                        conn.execute(
                            f"DELETE FROM episodes WHERE id IN ({','.join('?' * len(victims))})",
                            victims,
                        )
                        evicted_ids.extend(victims)
                    # Fallback: if lesson-bearing / consolidated rows STILL keep
                    # the table over the cap, delete the oldest of any kind so
                    # the cap is actually enforced. (The old code only deleted
                    # lesson-empty/unconsolidated rows, so once those filled the
                    # table the cap silently stopped enforcing — unbounded growth.)
                    still = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                    if still > self.MAX_EPISODES:
                        victims2 = [r[0] for r in conn.execute(
                            '''SELECT id FROM episodes
                               ORDER BY access_count ASC, last_accessed ASC,
                                        timestamp ASC LIMIT ?''',
                            (still - self.MAX_EPISODES,)
                        ).fetchall()]
                        if victims2:
                            conn.execute(
                                f"DELETE FROM episodes WHERE id IN ({','.join('?' * len(victims2))})",
                                victims2,
                            )
                            evicted_ids.extend(victims2)
                # Reap action rows orphaned by ANY delete (the cap-enforcement
                # deletes above, but also consolidation / external deletes that
                # never breach the cap). The FK is declared but not enforced (no
                # PRAGMA / cascade), so without an UNCONDITIONAL reap here
                # episode_actions grew unboundedly with orphans — the previous
                # version only reaped inside the `count > MAX` branch, so orphans
                # from non-cap deletes survived until an insert happened to breach
                # the cap. Cheap: indexed anti-join (idx_ea_episode) at ≤500 rows.
                conn.execute(
                    "DELETE FROM episode_actions WHERE episode_id NOT IN "
                    "(SELECT id FROM episodes)"
                )

                conn.commit()

        # Vector index maintenance happens OUTSIDE the sqlite lock so the
        # embedding call (potentially slow) doesn't serialise other writers.
        if vector_memory is not None:
            self._ingest_episode_vector(
                episode_id, trigger, lesson, vector_memory,
            )
            for vid in evicted_ids:
                forget = getattr(vector_memory, "forget_episode", None)
                if callable(forget):
                    try:
                        forget(vid)
                    except Exception:
                        pass
        # Episode commits were silent (audit A8). Episodes land every turn, so
        # the ordinary commit stays DEBUG; a LESSON-BEARING episode is a real
        # learning event and gets durable INFO — named, with the outcome.
        if lesson:
            logger.info(
                "episode #%d committed with lesson: %r (trigger: %s, %s)",
                episode_id, lesson[:100], trigger[:60],
                "success" if success else "failure",
            )
        else:
            logger.debug(
                "episode #%d committed (trigger: %s, %s, %d action(s))",
                episode_id, trigger[:60],
                "success" if success else "failure", len(actions or []),
            )
        return episode_id

    def _ingest_episode_vector(self, episode_id: int, trigger: str,
                               lesson: str, vector_memory) -> None:
        """Index one episode's trigger (+lesson) into the vector store with
        ``{"type":"episode","episode_id":...}`` metadata. Best-effort — a
        failure here must never sink the episode write.

        The vector store dedups on the document text (md5), so two episodes
        with an identical trigger share a single vector entry pointing at the
        first; this trades a small recall edge case for not flooding the index
        with near-duplicate embeddings, matching the store's own dedup
        contract in :meth:`VectorMemory.add`.
        """
        add_fn = getattr(vector_memory, "add", None)
        if not callable(add_fn):
            return
        text = (trigger or "").strip()
        if lesson:
            text = f"{text} :: {lesson}".strip(" :")
        if len(text) < 5:
            return
        try:
            add_fn(text, {
                "type": "episode",
                "episode_id": int(episode_id),
                # ISO-8601 like every OTHER writer into this store. A float
                # epoch here rendered as `[1783488280.73]` in the prompt
                # (VectorMemory._render_item prints the raw value) next to
                # `[2026-07-22T...Z]` neighbours — unreadable as a date for
                # exactly the memory type where recency matters most.
                "timestamp": get_utc_timestamp(),
            })
        except Exception as exc:
            # Was `debug`, i.e. invisible: a failed embed leaves an episode
            # that is permanently unreachable by semantic recall while
            # record_episode still returns a valid id. The operator watches
            # the live stream — this must be visible there (and it is what
            # `reconcile_vector_index` exists to repair).
            logger.warning(
                "Episode %s vector ingest FAILED — episode is invisible to "
                "semantic recall until reconciled: %s", episode_id, exc,
            )

    @staticmethod
    def _episode_document(trigger: str, lesson: str) -> str:
        """The exact document text `_ingest_episode_vector` embeds — kept in
        sync so the reconcile can tell a dedup-covered episode from a
        genuinely-missing one."""
        text = (trigger or "").strip()
        if lesson:
            text = f"{text} :: {lesson}".strip(" :")
        return text

    def reconcile_vector_index(self, vector_memory, limit: int = None) -> int:
        """Re-ingest episodes whose vector twin GENUINELY failed to embed —
        and, crucially, SKIP episodes that only lack an own-id twin because
        the vector store dedup'd their (identical) trigger onto another
        episode's entry. Returns the number of genuine re-ingests.

        The store dedups on document text (md5), so N episodes with the same
        trigger share ONE entry carrying ONE episode_id; the other N-1 have
        no own-id twin but are STILL reachable by semantic recall via the
        shared entry. The old reconcile counted those as "missing" and
        re-`add`d them every boot — the add() no-op'd, so the number never
        dropped (and ticked up as new same-trigger episodes accrued),
        alarming the operator and doing pointless work. Now we compare each
        episode's would-be document against the documents ALREADY indexed:
        present → dedup-covered (skip, report separately); absent → a real
        hole (re-ingest).

        Not called from inside this module — intended for a boot / maintenance
        hook owned by the caller (agent.py).
        """
        if vector_memory is None:
            return 0
        if limit is None:
            limit = self.MAX_EPISODES
        collection = getattr(vector_memory, "collection", None)
        get_fn = getattr(collection, "get", None) if collection is not None else None
        if not callable(get_fn):
            return 0
        try:
            existing = get_fn(where={"type": "episode"},
                              include=["metadatas", "documents"])
        except Exception as exc:
            logger.warning("Episode vector reconcile: index read failed: %s", exc)
            return 0
        if not isinstance(existing, dict):
            return 0
        indexed_ids = set()
        for meta in (existing.get("metadatas") or []):
            try:
                indexed_ids.add(int((meta or {}).get("episode_id")))
            except (TypeError, ValueError):
                continue
        indexed_docs = {str(d).strip() for d in (existing.get("documents") or []) if d}
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                rows = conn.execute(
                    """SELECT id, trigger, lesson FROM episodes
                       ORDER BY timestamp DESC LIMIT ?""",
                    (int(limit),),
                ).fetchall()
        no_own_twin = [r for r in rows if int(r[0]) not in indexed_ids]
        genuine, dedup_covered = [], 0
        for ep_id, trigger, lesson in no_own_twin:
            doc = self._episode_document(trigger or "", lesson or "")
            # A too-short doc is never embedded (see _ingest_episode_vector's
            # len<5 guard), so it's not a repairable hole either.
            if len(doc) < 5 or doc in indexed_docs:
                dedup_covered += 1
            else:
                genuine.append((ep_id, trigger, lesson))
        if not genuine:
            if dedup_covered:
                logger.info(
                    "Episode vector reconcile: all %d without an own-id twin "
                    "are dedup-covered (reachable via a shared-trigger entry) "
                    "— nothing to re-ingest.", dedup_covered)
            return 0
        logger.warning(
            "Episode vector reconcile: %d of %d episodes are genuinely "
            "missing a vector twin (invisible to semantic recall) — "
            "re-ingesting (%d others are dedup-covered).",
            len(genuine), len(rows), dedup_covered,
        )
        for ep_id, trigger, lesson in genuine:
            self._ingest_episode_vector(
                ep_id, trigger or "", lesson or "", vector_memory)
        return len(genuine)

    def search_similar(self, trigger: str, limit: int = 5,
                       vector_memory=None) -> List[Dict]:
        """Find episodes with similar triggers.

        Uses vector-based semantic search when *vector_memory* is provided,
        falling back to substring matching otherwise. The vector search
        embeds the trigger and compares against stored episode embeddings,
        catching semantically similar episodes that share no keywords.

        Both paths stamp ``relevance_score`` on the SAME [0, 1] scale
        (semantic: ``1 - distance``; substring: fraction of query words
        matched), each shaped by the same mild recency multiplier, so callers
        that rank results coming from either path compare like with like. The
        substring path additionally keeps the raw overlap in ``match_count``.
        """
        if not trigger:
            return []

        # Try vector-based search first (much better recall)
        if vector_memory is not None:
            try:
                results = self._vector_search(trigger, limit, vector_memory)
                if results:
                    return results
            except Exception as exc:
                logger.debug("Episodic vector search failed, falling back: %s", exc)

        # Fallback: keyword-based substring matching
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                words = [w.strip().lower() for w in trigger.split() if len(w.strip()) > 3]
                if not words:
                    surfaced = self._get_recent(conn, limit)
                else:
                    results = []
                    # Scan depth = the capacity cap, not a hardcoded 100: with
                    # MAX_EPISODES=500 the old LIMIT 100 made every older episode
                    # unreachable by this path, and unreachable FULL STOP for the
                    # ones whose vector twin is missing.
                    cursor = conn.execute(
                        "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?",
                        (self.FALLBACK_SCAN_LIMIT,),
                    )
                    for row in cursor:
                        row_dict = dict(row)
                        trigger_lower = row_dict["trigger"].lower()
                        overlap = sum(1 for w in words if w in trigger_lower)
                        if overlap > 0:
                            # Fraction of the query matched → [0, 1], directly
                            # comparable with the vector path's 1 - distance
                            # (the old value was a raw integer count, so callers
                            # ranking across both paths compared 3 against 0.9).
                            row_dict["match_count"] = overlap
                            row_dict["relevance_score"] = self._shape_by_recency(
                                overlap / len(words), row_dict.get("timestamp"),
                            )
                            results.append(row_dict)

                    results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                    surfaced = results[:limit]
        # Episode-row usage credit for the SUBSTRING path too (F8). Wiring the
        # credit sink into only the semantic path would leave every episode that
        # is reachable only by substring — precisely the ones whose vector twin
        # is missing — permanently at zero credit, i.e. evict-first forever.
        # Deliberately OUTSIDE both `with` blocks: this opens its own write
        # connection, and issuing that write while the read cursor above is
        # still open risks "database is locked", which the helper would swallow
        # — credit silently lost, the failure mode that is hardest to notice.
        self._credit_surfaced_episodes([e.get("id") for e in surfaced])
        return surfaced

    @classmethod
    def _recency_decay(cls, timestamp) -> float:
        """Exponential recency factor in (0, 1] — 1.0 for "just now" and for
        rows with no usable timestamp (so recency never penalises a caller
        that hands us a partial dict)."""
        try:
            ts = float(timestamp)
        except (TypeError, ValueError):
            return 1.0
        if ts <= 0:
            return 1.0
        age_days = max(0.0, (time.time() - ts) / 86400.0)
        try:
            return max(0.0, min(1.0, 0.5 ** (age_days / cls.RECENCY_HALF_LIFE_DAYS)))
        except (OverflowError, ZeroDivisionError):
            return 0.0

    @classmethod
    def _shape_by_recency(cls, base: float, timestamp) -> float:
        """Blend a mild recency signal into a [0, 1] relevance score.

        Relevance still dominates (recency moves the score by at most
        RECENCY_WEIGHT), but a fresh episode now outranks a stale one at equal
        match quality — previously neither path had ANY recency signal."""
        decay = cls._recency_decay(timestamp)
        shaped = base * (1.0 - cls.RECENCY_WEIGHT + cls.RECENCY_WEIGHT * decay)
        return max(0.0, min(1.0, shaped))

    def _scoped_episode_hits(self, trigger: str, k: int,
                             vector_memory) -> Optional[List[Dict]]:
        """Query the vector store for ``type == "episode"`` rows ONLY.

        Returns parsed hits (possibly empty), or ``None`` when this vector
        store exposes no scoped-query path — the caller then falls back to
        ``search_advanced``.

        Why this exists: ``search_advanced`` has no type filter and bumps
        ``retrieval_count`` / ``last_accessed`` on EVERY hit before we filter
        to episodes, so each hydration landed ~5-8 phantom retrievals on
        document / identity rows the model never saw — poisoning the exact
        fields that drive prune survival and time decay (936 document chunks
        live carry non-zero retrieval counts). Chroma supports
        ``where={"type": "episode"}``; ``VectorMemory.search_document`` and
        ``forget_episode`` already drive ``collection`` that way, so this
        stays inside an established pattern without touching vector.py.
        A ``where=`` argument on ``search_advanced`` itself would be the
        cleaner fix — that file belongs to another owner.
        """
        collection = getattr(vector_memory, "collection", None)
        query_fn = getattr(collection, "query", None) if collection is not None else None
        if not callable(query_fn):
            return None
        lock = None
        lock_fn = getattr(vector_memory, "_get_lock", None)
        if callable(lock_fn):
            try:
                lock = lock_fn()
            except Exception:
                lock = None
        try:
            if lock is not None and hasattr(lock, "__enter__"):
                with lock:
                    res = query_fn(query_texts=[trigger], n_results=max(1, int(k)),
                                   where={"type": "episode"})
            else:
                res = query_fn(query_texts=[trigger], n_results=max(1, int(k)),
                               where={"type": "episode"})
        except Exception as exc:
            logger.debug("Scoped episode vector query unavailable (%s); "
                         "falling back to search_advanced", exc)
            return None
        # Strict shape check — anything unexpected (including a test double
        # that returns a mock) means "no scoped path", not "no episodes".
        if not isinstance(res, dict):
            return None
        ids = res.get("ids")
        if not isinstance(ids, list):
            return None
        if not ids or not isinstance(ids[0], list):
            return []
        metas = res.get("metadatas") if isinstance(res.get("metadatas"), list) else [[]]
        dists = res.get("distances") if isinstance(res.get("distances"), list) else [[]]
        metas0 = metas[0] if metas and isinstance(metas[0], list) else []
        dists0 = dists[0] if dists and isinstance(dists[0], list) else []
        hits: List[Dict] = []
        for i, mem_id in enumerate(ids[0]):
            meta = metas0[i] if i < len(metas0) else {}
            try:
                dist = float(dists0[i])
            except (IndexError, TypeError, ValueError):
                dist = 1.0
            hits.append({"id": mem_id, "metadata": meta or {}, "score": dist})
        return hits

    def _credit_surfaced_episodes(self, episode_ids) -> None:
        """Record that these episodes were SURFACED (returned to a caller, i.e.
        injected into a prompt) — the eviction-facing half of usage credit.

        AUTHORITY SPLIT (F8, 2026-08-08). Two counters now exist and they are
        deliberately NOT the same one:

        * the vector twin's ``retrieval_count`` / ``last_accessed`` (bumped via
          ``bump_retrievals``) stays authoritative for retrieval stats, RRF and
          vector prune survival;
        * ``episodes.access_count`` / ``episodes.last_accessed`` — written ONLY
          here — is authoritative for EPISODIC EVICTION, which is plain SQL over
          this table and structurally cannot read the vector store.

        Both are driven from the same surfacing event, so they cannot drift in
        DIRECTION (the twin-divergence lesson); they are allowed to differ in
        magnitude, because they answer different questions.

        This is the single writer, called from EVERY path that surfaces an
        episode (semantic and substring) — a credit sink wired into only one of
        two paths would quietly make the other path's episodes evict-first,
        which is the same partial-coverage defect this feature exists to fix.

        Never raises: usage accounting must not break recall.
        """
        # Coerce PER ITEM, not as one comprehension: a single malformed id
        # (a row dict without "id", a None) would otherwise raise out of the
        # whole list and silently drop credit for every OTHER episode in the
        # batch — a partial-failure-becomes-total-failure shape.
        ids = []
        for raw in (episode_ids or []):
            try:
                ids.append(int(raw))
            except (TypeError, ValueError):
                continue
        if not ids:
            return
        try:
            now = time.time()
            with self._lock:
                with closing(sqlite3.connect(self.db_path)) as conn:
                    conn.execute(
                        "UPDATE episodes SET access_count = access_count + 1, "
                        f"last_accessed = ? WHERE id IN ({','.join('?' * len(ids))})",
                        [now, *ids],
                    )
                    conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.debug("episode usage-credit skipped: %s", exc)

    def _vector_search(self, trigger: str, limit: int,
                       vector_memory) -> List[Dict]:
        """Semantic search using the vector memory's embedding model."""
        # Episodes are ingested into the vector store with
        # {"type":"episode","episode_id":...} metadata by record_episode (when
        # given a vector_memory), so this semantic path is live; it still falls
        # back to substring matching when no episode-typed hits come back (e.g.
        # a cold store, or an episode whose trigger was deduped under another
        # episode's vector entry).
        #
        # Preferred: a type-SCOPED query, which never touches non-episode rows
        # (and therefore never bumps their retrieval stats). Fallback for
        # stores without a queryable collection: search_advanced, which
        # retrieves everything and is filtered by type below.
        scoped = self._scoped_episode_hits(trigger, limit * 2, vector_memory)
        if scoped is None:
            search_fn = getattr(vector_memory, "search_advanced", None)
            if not callable(search_fn):
                return []
            try:
                # §4M (Lens B MINOR, latent): probe purity — this fallback
                # read must not bump retrieval stats (search_advanced
                # records by default). Signature-guarded so stubs without
                # the parameter keep working.
                import inspect as _insp
                kw = {"limit": limit * 2}
                try:
                    _params = _insp.signature(search_fn).parameters
                    if ("record_retrievals" in _params
                            or any(p.kind is _insp.Parameter.VAR_KEYWORD
                                   for p in _params.values())):
                        kw["record_retrievals"] = False
                except (TypeError, ValueError):
                    kw["record_retrievals"] = False  # mock/builtin — safe
                hits = search_fn(trigger, **kw)
            except Exception:
                return []
            scoped_query = False
        else:
            hits = scoped
            scoped_query = True
        if not hits:
            return []
        # Defensive even on the scoped path: keep episode hits only.
        hits = [h for h in hits if (h.get("metadata") or {}).get("type") == "episode"]
        # Map vector hits back to episode records, keeping each hit's distance
        # (search_advanced exposes it as "score" — lower is closer). We retain
        # the BEST (smallest) distance per episode if it appears more than once.
        ep_dist: Dict[int, float] = {}
        ep_vec_id: Dict[int, Any] = {}
        ordered_ids: List[int] = []
        for hit in hits:
            meta = hit.get("metadata", {})
            ep_id = meta.get("episode_id")
            if ep_id is None:
                continue
            try:
                ep_id = int(ep_id)
            except (ValueError, TypeError):
                continue
            try:
                dist = float(hit.get("score"))
            except (TypeError, ValueError):
                dist = 1.0
            if ep_id not in ep_dist:
                ordered_ids.append(ep_id)
                ep_dist[ep_id] = dist
                ep_vec_id[ep_id] = hit.get("id")
            elif dist < ep_dist[ep_id]:
                ep_dist[ep_id] = dist
                ep_vec_id[ep_id] = hit.get("id")
        if not ordered_ids:
            return []
        results = []
        surfaced_vec_ids: List[Any] = []
        surfaced_ep_ids: List[int] = []
        for ep_id in ordered_ids:
            # Distance-derived relevance instead of a flat 1.0. Drop hits below
            # the floor so semantically irrelevant episodes aren't injected.
            relevance = 1.0 - ep_dist[ep_id]
            if relevance < 0.0:
                relevance = 0.0
            elif relevance > 1.0:
                relevance = 1.0
            if relevance < self.MIN_VECTOR_RELEVANCE:
                continue
            ep = self.get_episode(ep_id)
            if ep:
                # Same [0, 1] scale and same recency shaping as the substring
                # path (a row with no timestamp is left untouched).
                ep["relevance_score"] = self._shape_by_recency(
                    relevance, ep.get("timestamp"))
                results.append(ep)
                surfaced_ep_ids.append(ep_id)
                if ep_vec_id.get(ep_id):
                    surfaced_vec_ids.append(ep_vec_id[ep_id])
            if len(results) >= limit:
                break
        # The scoped path bypasses search_advanced's blanket stats bump, so
        # credit reinforcement HERE — and only to the episode rows that were
        # actually surfaced, which is what the spaced-repetition fields are
        # supposed to mean.
        if scoped_query and surfaced_vec_ids:
            bump = getattr(vector_memory, "bump_retrievals", None)
            if callable(bump):
                try:
                    bump(surfaced_vec_ids)
                except Exception as exc:
                    logger.debug("episode retrieval bump failed: %s", exc)
        # Episode-row credit (F8) is NOT gated on `scoped_query`. That gate
        # exists solely to stop the VECTOR counter being double-bumped on the
        # path where search_advanced already counted; nothing else writes
        # episodes.access_count, so inheriting the gate would simply leave the
        # fallback path's episodes permanently at zero credit — evict-first
        # forever despite being surfaced. Same reason it is outside the
        # `surfaced_vec_ids` check: an episode with no vector twin id is still
        # a surfaced episode.
        self._credit_surfaced_episodes(surfaced_ep_ids)
        return results

    def search_by_outcome(self, success: bool, limit: int = 10) -> List[Dict]:
        """Find episodes by outcome (success/failure). Useful for System 3
        crisis pivots that need to find past recovery strategies."""
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """SELECT * FROM episodes WHERE outcome_success = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (1 if success else 0, limit),
                )
                return [dict(row) for row in cursor]

    def _recovery_evidence(self, ep: Dict) -> Tuple[str, Optional[List[Dict]]]:
        """Classify HOW strongly a successful episode looks like a recovery.

        Returns ``(evidence, actions_or_None)`` where evidence is one of:

        * ``"lesson"``        — an explicit distilled lesson (strongest, but the
          production writer never sets one, see :meth:`search_recoveries`)
        * ``"failed_action"`` — the episode ended successfully yet contains at
          least one FAILED tool call: the literal "tried X, it broke, switched
          to W" signature, derived from data the live schema actually has
        * ``"outcome_text"``  — the outcome narrates trouble (error / fixed /
          retried / fell back …), for episodes whose action rows are absent
        * ``"success_only"``  — a relevant success with no trouble signal

        The action rows are returned when we had to fetch them, so the caller
        can attach the recovery chain instead of re-querying.
        """
        if (ep.get("lesson") or "").strip():
            return "lesson", None
        actions = ep.get("actions")
        fetched = None
        if actions is None:
            # The substring path returns bare `episodes` rows (SELECT *), so
            # the action chain — where the recovery signal lives — isn't there.
            try:
                full = self.get_episode(int(ep.get("id")))
            except (TypeError, ValueError):
                full = None
            if full:
                actions = full.get("actions")
                fetched = actions
        for a in actions or []:
            if a.get("tool_name") == self.TRUNCATION_MARKER_TOOL:
                continue
            if not a.get("success", 1):
                return "failed_action", fetched
        text = f"{ep.get('outcome', '')} {ep.get('context', '')}".lower()
        if any(m in text for m in self.RECOVERY_OUTCOME_MARKERS):
            return "outcome_text", fetched
        return "success_only", fetched

    def search_recoveries(self, error_context: str, limit: int = 5,
                          vector_memory=None) -> List[Dict]:
        """Find past episodes where the agent recovered from a similar failure.

        Used by System 3 crisis pivots to inform strategy generation with
        historical context.

        This used to require a non-empty ``lesson``, which made it
        structurally dead: the only production writer
        (``core/agent.py::_record_episode_safe``) never passes one, so 0 of
        145 live episodes qualified and every crisis pivot silently got ``[]``.
        The recovery signal is therefore derived from fields the live schema
        DOES carry (see :meth:`_recovery_evidence`) — chiefly "ended
        successfully but contains a failed tool call", which matches 38 of the
        145 live episodes. Each returned episode is tagged with
        ``recovery_evidence`` so the caller can weigh it.

        Ordering: strong evidence first (in relevance order), and only if
        nothing qualifies do we fall back to plain relevant successes —
        candidates are already similarity-filtered against the error context,
        so a weak answer beats the previous guaranteed-empty one.
        """
        candidates = self.search_similar(
            error_context, limit=limit * 3, vector_memory=vector_memory,
        )
        strong: List[Dict] = []
        weak: List[Dict] = []
        for ep in candidates:
            if not ep.get("outcome_success"):
                continue
            evidence, actions = self._recovery_evidence(ep)
            enriched = dict(ep)
            enriched["recovery_evidence"] = evidence
            if actions is not None and "actions" not in enriched:
                enriched["actions"] = actions
            if evidence == "success_only":
                weak.append(enriched)
            else:
                strong.append(enriched)
        return (strong or weak)[:limit]

    def _get_recent(self, conn, limit: int) -> List[Dict]:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor]

    def get_episode(self, episode_id: int) -> Optional[Dict]:
        """Retrieve a full episode with its actions."""
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM episodes WHERE id = ?", (episode_id,)
                ).fetchone()
                if not row:
                    return None
                episode = dict(row)
                actions = conn.execute(
                    '''SELECT tool_name, tool_args, result, success
                       FROM episode_actions WHERE episode_id = ?
                       ORDER BY action_order''',
                    (episode_id,)
                ).fetchall()
                episode["actions"] = [dict(a) for a in actions]
                return episode

    def get_recent_episodes(self, limit: int = 10) -> List[Dict]:
        """Return the most recent episodes (without action details)."""
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                return self._get_recent(conn, limit)

    def get_episodes_by_cluster(self, cluster_id: str, limit: int = 10) -> List[Dict]:
        """Return episodes in a specific cluster."""
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    '''SELECT * FROM episodes WHERE cluster_id = ?
                       ORDER BY timestamp DESC LIMIT ?''',
                    (cluster_id, limit)
                )
                return [dict(row) for row in cursor]

    def mark_consolidated(self, episode_ids: List[int]):
        """Mark episodes as consolidated by the dream cycle."""
        if not episode_ids:
            return
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                placeholders = ",".join("?" * len(episode_ids))
                conn.execute(
                    f"UPDATE episodes SET consolidated = 1 WHERE id IN ({placeholders})",
                    episode_ids
                )
                conn.commit()

    def get_unconsolidated(self, limit: int = 50) -> List[Dict]:
        """Return episodes that haven't been processed by the dream cycle."""
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    '''SELECT * FROM episodes WHERE consolidated = 0
                       ORDER BY timestamp DESC LIMIT ?''',
                    (limit,)
                )
                return [dict(row) for row in cursor]

    @staticmethod
    def _relative_age(timestamp) -> str:
        """Render an epoch timestamp as a compact relative age ("3d ago").

        Returns "" for a missing / unparseable timestamp so renderers degrade
        to their previous output instead of printing junk."""
        try:
            ts = float(timestamp)
        except (TypeError, ValueError):
            return ""
        if ts <= 0:
            return ""
        delta = max(0.0, time.time() - ts)
        minutes = delta / 60.0
        if minutes < 1:
            return "just now"
        if minutes < 60:
            return f"{int(minutes)}m ago"
        hours = minutes / 60.0
        if hours < 24:
            return f"{int(hours)}h ago"
        days = hours / 24.0
        if days < 30:
            return f"{int(days)}d ago"
        months = days / 30.44
        if months < 12:
            return f"{int(months)}mo ago"
        return f"{days / 365.25:.1f}y ago"

    @staticmethod
    def format_episode(ep: Dict) -> str:
        """Render ONE episode as a single compact line.

        Shared by `format_for_context` (blob rendering, legacy callers) and
        the MemoryBus per-item fetcher — one episode per RRF item so fusion
        can rank episodes individually instead of treating the whole tier
        as a single rank-1 blob.

        Includes a RELATIVE date (`[3d ago]`) — without it the model could not
        tell an episode from yesterday from one six months old, which is the
        difference between "this is how the system behaves" and "this is how
        the system used to behave"."""
        age = EpisodicMemory._relative_age(ep.get("timestamp"))
        entry = (
            f"- [{ep.get('cluster_id') or 'general'}] "
            f"{('[' + age + '] ') if age else ''}"
            f"Trigger: {ep['trigger'][:100]} | "
            f"Outcome: {'SUCCESS' if ep.get('outcome_success') else 'FAILURE'} — "
            f"{ep.get('outcome', 'unknown')[:100]}"
        )
        if ep.get("lesson"):
            entry += f" | Lesson: {ep['lesson'][:100]}"
        return entry

    def format_for_context(self, episodes: List[Dict], max_chars: int = 2000) -> str:
        """Format episodes for injection into LLM context."""
        if not episodes:
            return ""
        lines = ["### RELEVANT PAST EPISODES (from prior sessions — for reference only, NOT events in the current conversation):"]
        used = 0
        for ep in episodes:
            entry = self.format_episode(ep)
            if used + len(entry) > max_chars:
                lines.append(f"[... +{len(episodes) - len(lines) + 1} more episodes]")
                break
            lines.append(entry)
            used += len(entry)
        return "\n".join(lines)

    def count(self) -> int:
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
