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
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    def __init__(self, memory_dir: Path):
        self.db_path = memory_dir / "episodic_memory.db"
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
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
                        consolidated INTEGER DEFAULT 0
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
                conn.execute('CREATE INDEX IF NOT EXISTS idx_ep_trigger ON episodes(trigger)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_ep_cluster ON episodes(cluster_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_ep_ts ON episodes(timestamp)')
                # Supports the unconditional orphan-reap anti-join in
                # record_episode (episode_actions has no enforced FK cascade).
                conn.execute('CREATE INDEX IF NOT EXISTS idx_ea_episode ON episode_actions(episode_id)')
                conn.commit()

    def record_episode(self, trigger: str, context: str = "",
                       actions: List[Dict[str, Any]] = None,
                       outcome: str = "", success: bool = False,
                       lesson: str = "", cluster_id: str = "") -> int:
        """Store a new episode. Returns the episode ID."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    '''INSERT INTO episodes (trigger, context, outcome, outcome_success,
                       lesson, cluster_id, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (trigger[:500], context[:2000], outcome[:1000],
                     1 if success else 0, lesson[:500], cluster_id, time.time())
                )
                episode_id = cursor.lastrowid

                if actions:
                    for i, action in enumerate(actions[:self.MAX_ACTIONS_PER_EPISODE]):
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

                # Enforce max capacity.
                count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                if count > self.MAX_EPISODES:
                    # Prefer deleting oldest non-lesson, unconsolidated episodes.
                    conn.execute(
                        '''DELETE FROM episodes WHERE id IN (
                            SELECT id FROM episodes
                            WHERE lesson = '' AND consolidated = 0
                            ORDER BY timestamp ASC
                            LIMIT ?
                        )''',
                        (count - self.MAX_EPISODES,)
                    )
                    # Fallback: if lesson-bearing / consolidated rows STILL keep
                    # the table over the cap, delete the oldest of any kind so
                    # the cap is actually enforced. (The old code only deleted
                    # lesson-empty/unconsolidated rows, so once those filled the
                    # table the cap silently stopped enforcing — unbounded growth.)
                    still = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                    if still > self.MAX_EPISODES:
                        conn.execute(
                            '''DELETE FROM episodes WHERE id IN (
                                SELECT id FROM episodes ORDER BY timestamp ASC LIMIT ?
                            )''',
                            (still - self.MAX_EPISODES,)
                        )
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
        return episode_id

    def search_similar(self, trigger: str, limit: int = 5,
                       vector_memory=None) -> List[Dict]:
        """Find episodes with similar triggers.

        Uses vector-based semantic search when *vector_memory* is provided,
        falling back to substring matching otherwise. The vector search
        embeds the trigger and compares against stored episode embeddings,
        catching semantically similar episodes that share no keywords.
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
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                words = [w.strip().lower() for w in trigger.split() if len(w.strip()) > 3]
                if not words:
                    return self._get_recent(conn, limit)

                results = []
                cursor = conn.execute(
                    "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT 100"
                )
                for row in cursor:
                    row_dict = dict(row)
                    trigger_lower = row_dict["trigger"].lower()
                    score = sum(1 for w in words if w in trigger_lower)
                    if score > 0:
                        row_dict["relevance_score"] = score
                        results.append(row_dict)

                results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                return results[:limit]

    def _vector_search(self, trigger: str, limit: int,
                       vector_memory) -> List[Dict]:
        """Semantic search using the vector memory's embedding model."""
        # Search for episode-type memories in the vector store
        # Use search_advanced (the real raw-hits API). The previous
        # `search_raw` attribute existed NOWHERE, so this path was always a
        # no-op and recall silently fell back to substring matching. NOTE:
        # activating true semantic recall also needs episodes ingested into
        # the vector store with {"type":"episode","episode_id":...} metadata
        # (a separate ingestion gap); until that's wired this still falls
        # back — but it now calls a method that actually exists.
        search_fn = getattr(vector_memory, "search_advanced", None)
        if not callable(search_fn):
            return []
        try:
            hits = search_fn(trigger, limit=limit * 2)
        except Exception:
            return []
        if not hits:
            return []
        # search_advanced doesn't filter by type — keep episode hits only.
        hits = [h for h in hits if (h.get("metadata") or {}).get("type") == "episode"]
        # Map vector hits back to episode records
        episode_ids = []
        for hit in hits:
            meta = hit.get("metadata", {})
            ep_id = meta.get("episode_id")
            if ep_id is not None:
                try:
                    episode_ids.append(int(ep_id))
                except (ValueError, TypeError):
                    pass
        if not episode_ids:
            return []
        results = []
        for ep_id in episode_ids[:limit]:
            ep = self.get_episode(ep_id)
            if ep:
                ep["relevance_score"] = 1.0  # Vector matches are high relevance
                results.append(ep)
        return results

    def search_by_outcome(self, success: bool, limit: int = 10) -> List[Dict]:
        """Find episodes by outcome (success/failure). Useful for System 3
        crisis pivots that need to find past recovery strategies."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """SELECT * FROM episodes WHERE outcome_success = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (1 if success else 0, limit),
                )
                return [dict(row) for row in cursor]

    def search_recoveries(self, error_context: str, limit: int = 5,
                          vector_memory=None) -> List[Dict]:
        """Find past episodes where the agent recovered from a similar failure.

        Used by System 3 crisis pivots to inform strategy generation with
        historical context.
        """
        # Search for failure episodes that eventually succeeded
        candidates = self.search_similar(
            error_context, limit=limit * 3, vector_memory=vector_memory,
        )
        # Filter to episodes that had a lesson (implying recovery)
        recoveries = [
            ep for ep in candidates
            if ep.get("lesson") and ep.get("outcome_success")
        ]
        return recoveries[:limit]

    def _get_recent(self, conn, limit: int) -> List[Dict]:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor]

    def get_episode(self, episode_id: int) -> Optional[Dict]:
        """Retrieve a full episode with its actions."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
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
            with sqlite3.connect(self.db_path) as conn:
                return self._get_recent(conn, limit)

    def get_episodes_by_cluster(self, cluster_id: str, limit: int = 10) -> List[Dict]:
        """Return episodes in a specific cluster."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
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
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ",".join("?" * len(episode_ids))
                conn.execute(
                    f"UPDATE episodes SET consolidated = 1 WHERE id IN ({placeholders})",
                    episode_ids
                )
                conn.commit()

    def get_unconsolidated(self, limit: int = 50) -> List[Dict]:
        """Return episodes that haven't been processed by the dream cycle."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    '''SELECT * FROM episodes WHERE consolidated = 0
                       ORDER BY timestamp DESC LIMIT ?''',
                    (limit,)
                )
                return [dict(row) for row in cursor]

    def format_for_context(self, episodes: List[Dict], max_chars: int = 2000) -> str:
        """Format episodes for injection into LLM context."""
        if not episodes:
            return ""
        lines = ["### RELEVANT PAST EPISODES (from prior sessions — for reference only, NOT events in the current conversation):"]
        used = 0
        for ep in episodes:
            entry = (
                f"- [{ep.get('cluster_id', 'general')}] "
                f"Trigger: {ep['trigger'][:100]} | "
                f"Outcome: {'SUCCESS' if ep.get('outcome_success') else 'FAILURE'} — "
                f"{ep.get('outcome', 'unknown')[:100]}"
            )
            if ep.get("lesson"):
                entry += f" | Lesson: {ep['lesson'][:100]}"
            if used + len(entry) > max_chars:
                lines.append(f"[... +{len(episodes) - len(lines) + 1} more episodes]")
                break
            lines.append(entry)
            used += len(entry)
        return "\n".join(lines)

    def count(self) -> int:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
