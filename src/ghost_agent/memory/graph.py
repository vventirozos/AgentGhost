import sqlite3
import threading
import difflib
import logging
from pathlib import Path
from typing import List, Dict, Iterable, Tuple

import networkx as nx

logger = logging.getLogger("GhostAgent")


class GraphMemory:
    """Knowledge graph with SQLite persistence + in-memory NetworkX routing.

    SQLite remains the source of truth on disk. A `nx.MultiDiGraph` mirror is
    held in memory and used for spreading-activation traversal in
    `get_neighborhood`. Both stores are kept in sync by `add_triplets`,
    `delete_by_target`, `wipe_all`, and `execute_graph_compression`.
    """

    def __init__(self, memory_dir: Path):
        self.db_path = memory_dir / "knowledge_graph.db"
        self._lock = threading.RLock()
        self.nx_graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._init_db()
        self.initialize_graph()

    # ------------------------------------------------------------------ setup

    def _init_db(self):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS triplets (
                        subject TEXT,
                        predicate TEXT,
                        object TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(subject, predicate, object)
                    )
                ''')
                try:
                    conn.execute('ALTER TABLE triplets ADD COLUMN weight INTEGER DEFAULT 1')
                except Exception:
                    pass
                # Temporal columns: valid_from marks when the fact became true,
                # valid_until marks when it was superseded (NULL = still current).
                try:
                    conn.execute('ALTER TABLE triplets ADD COLUMN valid_from REAL')
                except Exception:
                    pass
                try:
                    conn.execute('ALTER TABLE triplets ADD COLUMN valid_until REAL')
                except Exception:
                    pass
                conn.execute('CREATE INDEX IF NOT EXISTS idx_subj ON triplets(subject)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_obj ON triplets(object)')
                conn.commit()

    def initialize_graph(self, include_expired: bool = False):
        """Hydrate the in-memory NetworkX graph from the SQLite triplets table.

        By default only loads temporally-valid edges (valid_until IS NULL).
        Pass ``include_expired=True`` to load the full history."""
        with self._lock:
            self.nx_graph = nx.MultiDiGraph()
            with sqlite3.connect(self.db_path) as conn:
                if include_expired:
                    query = 'SELECT subject, predicate, object, weight FROM triplets'
                else:
                    query = 'SELECT subject, predicate, object, weight FROM triplets WHERE valid_until IS NULL'
                cursor = conn.execute(query)
                for s, p, o, w in cursor:
                    self._upsert_edge(s, p, o, int(w or 1))

    # ----------------------------------------------------------- graph mirror

    def _upsert_edge(self, s: str, p: str, o: str, weight: int):
        """Add or reinforce a single edge in the in-memory graph."""
        existing_key = None
        if self.nx_graph.has_edge(s, o):
            for k, data in self.nx_graph[s][o].items():
                if data.get("predicate") == p:
                    existing_key = k
                    break
        if existing_key is not None:
            self.nx_graph[s][o][existing_key]["weight"] = weight
        else:
            self.nx_graph.add_edge(s, o, predicate=p, weight=weight)

    def _remove_edge(self, s: str, p: str, o: str):
        if not self.nx_graph.has_edge(s, o):
            return
        kill = [k for k, data in self.nx_graph[s][o].items()
                if data.get("predicate") == p]
        for k in kill:
            self.nx_graph.remove_edge(s, o, key=k)
        for n in (s, o):
            if n in self.nx_graph and self.nx_graph.degree(n) == 0:
                self.nx_graph.remove_node(n)

    # ------------------------------------------------------------ public CRUD

    def add_triplets(self, triplets: List[Dict[str, str]]):
        if not triplets:
            return 0
        added = 0
        import time as _time
        now = _time.time()
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                for t in triplets:
                    s = t.get("subject", "")
                    p = t.get("predicate", "")
                    o = t.get("object", "")
                    if not (s and p and o):
                        continue
                    sn = str(s).lower().strip()
                    pn = str(p).upper().strip()
                    on = str(o).lower().strip()
                    if not (sn and pn and on):
                        continue
                    try:
                        # Temporal conflict resolution: if the same subject+predicate
                        # exists with a DIFFERENT object, expire the old edge instead
                        # of accumulating contradictions. Only applies to "functional"
                        # predicates (one-to-one) like WORKS_AT, LIVES_IN, DRIVES.
                        # Multi-valued predicates like LIKES, KNOWS, HAS are excluded.
                        _FUNCTIONAL_PREDICATES = {
                            "WORKS_AT", "LIVES_IN", "DRIVES", "MARRIED_TO",
                            "IS", "BORN_IN", "STUDIES_AT", "LOCATED_IN",
                            "EMPLOYED_BY", "OWNS",
                        }
                        conflicting = []
                        if pn in _FUNCTIONAL_PREDICATES:
                            conflicting = conn.execute(
                                '''SELECT object FROM triplets
                                   WHERE subject = ? AND predicate = ? AND object != ?
                                   AND valid_until IS NULL''',
                                (sn, pn, on)
                            ).fetchall()
                        if conflicting:
                            conn.execute(
                                '''UPDATE triplets SET valid_until = ?
                                   WHERE subject = ? AND predicate = ? AND object != ?
                                   AND valid_until IS NULL''',
                                (now, sn, pn, on)
                            )
                            # Remove expired edges from the in-memory graph
                            for (old_obj,) in conflicting:
                                self._remove_edge(sn, pn, old_obj)

                        cursor = conn.execute(
                            '''INSERT INTO triplets (subject, predicate, object, weight, timestamp, valid_from)
                               VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP, ?)
                               ON CONFLICT(subject, predicate, object)
                               DO UPDATE SET weight = weight + 1, timestamp = CURRENT_TIMESTAMP, valid_until = NULL''',
                            (sn, pn, on, now)
                        )
                        if cursor.rowcount > 0:
                            added += 1
                        # Read back the resulting weight so the mirror stays accurate
                        row = conn.execute(
                            'SELECT weight FROM triplets WHERE subject=? AND predicate=? AND object=?',
                            (sn, pn, on)
                        ).fetchone()
                        weight = int(row[0]) if row and row[0] is not None else 1
                        self._upsert_edge(sn, pn, on, weight)
                    except Exception:
                        pass
                conn.commit()
        return added

    def delete_by_target(self, target: str) -> int:
        if not target or len(target.strip()) < 3:
            return 0
        t_norm = target.lower().strip()
        like = f"%{t_norm}%"
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Collect matched rows first so we can mirror the deletion.
                doomed = conn.execute(
                    '''SELECT subject, predicate, object FROM triplets
                       WHERE subject LIKE ? OR object LIKE ?''',
                    (like, like)
                ).fetchall()
                cursor = conn.execute(
                    '''DELETE FROM triplets
                       WHERE subject LIKE ? OR object LIKE ?''',
                    (like, like)
                )
                deleted = cursor.rowcount
                conn.commit()
            for s, p, o in doomed:
                self._remove_edge(s, p, o)
        return deleted

    #: Generic hub nodes that link to nearly everything; expanding a
    #: `forget` to these would wipe unrelated knowledge, so they are never
    #: returned as connected entities.
    _ENTITY_EXPANSION_STOPLIST = {
        "user", "me", "i", "you", "it", "this", "that", "they", "them",
        "he", "she", "we", "thing", "things",
    }

    def get_connected_entities(self, target: str, limit: int = 8) -> List[str]:
        """Return distinct entity names directly (1 hop) connected to any node
        whose name contains ``target``.

        Lets ``forget`` expand an entity wipe to its tightly-coupled
        neighbours: forgetting ``mortimer`` surfaces ``iguana`` (from a
        ``mortimer IS_A iguana`` edge) so the alias tombstone goes too.
        Generic hub nodes (``user``, pronouns) are filtered out so the
        expansion can't snowball into unrelated memory.
        """
        if not target or len(target.strip()) < 3:
            return []
        t = target.lower().strip()
        like = f"%{t}%"
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    '''SELECT subject, object FROM triplets
                       WHERE subject LIKE ? OR object LIKE ?''',
                    (like, like)
                ).fetchall()
        out: List[str] = []
        seen = set()
        for s, o in rows:
            for node in (s, o):
                n = (node or "").lower().strip()
                # Skip the target's own variants, hub nodes, and tiny tokens.
                if not n or len(n) < 3 or t in n or n in t:
                    continue
                if n in self._ENTITY_EXPANSION_STOPLIST:
                    continue
                if n not in seen:
                    seen.add(n)
                    out.append(n)
                    if len(out) >= limit:
                        return out
        return out

    def wipe_all(self):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM triplets')
                conn.commit()
            self.nx_graph = nx.MultiDiGraph()

    def get_recent_triplets(self, limit: int = 100) -> List[Dict[str, str]]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT subject, predicate, object FROM triplets ORDER BY timestamp DESC LIMIT ?',
                    (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]

    def execute_graph_compression(self, merges: List[Dict[str, str]]) -> int:
        ops = 0
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for m in merges:
                    old_node = m.get("old_node", "").lower().strip()
                    new_node = m.get("new_node", "").lower().strip()
                    if old_node and new_node and old_node != new_node:
                        try:
                            # Rather than UPDATE OR IGNORE (which silently
                            # drops merges where the target row exists and
                            # loses the source weight), explicitly merge
                            # weights by summing them into the target row.
                            # Subject-side rewrite.
                            cursor.execute(
                                '''SELECT predicate, object, weight FROM triplets
                                   WHERE subject = ?''', (old_node,))
                            src_rows = cursor.fetchall()
                            for pred, obj, w in src_rows:
                                cursor.execute(
                                    '''INSERT INTO triplets (subject, predicate, object, weight, timestamp, valid_from)
                                       VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                                       ON CONFLICT(subject, predicate, object)
                                       DO UPDATE SET weight = weight + excluded.weight,
                                                     timestamp = CURRENT_TIMESTAMP''',
                                    (new_node, pred, obj, int(w or 1)))
                            # Object-side rewrite.
                            cursor.execute(
                                '''SELECT subject, predicate, weight FROM triplets
                                   WHERE object = ? AND subject != ?''', (old_node, old_node))
                            obj_rows = cursor.fetchall()
                            for subj, pred, w in obj_rows:
                                cursor.execute(
                                    '''INSERT INTO triplets (subject, predicate, object, weight, timestamp, valid_from)
                                       VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                                       ON CONFLICT(subject, predicate, object)
                                       DO UPDATE SET weight = weight + excluded.weight,
                                                     timestamp = CURRENT_TIMESTAMP''',
                                    (subj, pred, new_node, int(w or 1)))
                            # Now delete all rows that still reference old_node.
                            cursor.execute(
                                "DELETE FROM triplets WHERE subject = ? OR object = ?",
                                (old_node, old_node))
                            ops += 1
                        except Exception as e:
                            logger.debug(f"Graph compression merge failed: {type(e).__name__}: {e}")
                conn.commit()
            # Rebuild mirror after a structural rewrite — simpler than diffing.
            self.initialize_graph()
        return ops

    def get_expired_triplets(self, subject: str = None, limit: int = 50) -> List[Dict]:
        """Return expired (superseded) triplets, optionally filtered by subject."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if subject:
                    cursor = conn.execute(
                        '''SELECT subject, predicate, object, valid_from, valid_until
                           FROM triplets WHERE valid_until IS NOT NULL AND subject = ?
                           ORDER BY valid_until DESC LIMIT ?''',
                        (subject.lower().strip(), limit)
                    )
                else:
                    cursor = conn.execute(
                        '''SELECT subject, predicate, object, valid_from, valid_until
                           FROM triplets WHERE valid_until IS NOT NULL
                           ORDER BY valid_until DESC LIMIT ?''',
                        (limit,)
                    )
                return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------- spreading act'n

    def _map_words_to_seeds(self, words: Iterable[str]) -> List[str]:
        """Map free-form query words to exact node names in the graph.

        Strategy per word: exact match → substring containment → difflib
        fuzzy fallback. Words shorter than 3 chars are skipped.
        """
        if not self.nx_graph or self.nx_graph.number_of_nodes() == 0:
            return []
        seeds: List[str] = []
        seen = set()
        all_nodes = list(self.nx_graph.nodes())
        for w in words:
            wl = str(w).lower().strip()
            if len(wl) < 3:
                continue
            matches: List[str] = []
            if wl in self.nx_graph:
                matches = [wl]
            else:
                substr = [n for n in all_nodes if wl in n or n in wl]
                if substr:
                    # Prefer the closest length match for stable ordering
                    matches = sorted(substr, key=lambda n: (abs(len(n) - len(wl)), n))[:3]
                else:
                    matches = difflib.get_close_matches(wl, all_nodes, n=3, cutoff=0.7)
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    seeds.append(m)
        return seeds

    def _out_edges(self, node: str) -> List[Tuple[str, str, str, int]]:
        return [(node, nb, d.get("predicate", ""), int(d.get("weight", 1) or 1))
                for _, nb, _, d in self.nx_graph.out_edges(node, keys=True, data=True)]

    def _in_edges(self, node: str) -> List[Tuple[str, str, str, int]]:
        return [(pr, node, d.get("predicate", ""), int(d.get("weight", 1) or 1))
                for pr, _, _, d in self.nx_graph.in_edges(node, keys=True, data=True)]

    def _spreading_activation(self, seed: str,
                              path_scores: Dict[Tuple, int],
                              max_hops: int = 3) -> None:
        """Multi-hop BFS from `seed`, recording every directed chain of
        length 1 to `max_hops` that is naturally readable.
        Score = sum of edge weights along the chain.

        3-hop enables complex reasoning like:
        "A works_at B, B is_owned_by C, C is_located_in D"
        """
        if seed not in self.nx_graph:
            return
        out1 = self._out_edges(seed)
        in1 = self._in_edges(seed)

        def bump(chain: Tuple[Tuple[str, str, str], ...], score: int):
            prev = path_scores.get(chain)
            if prev is None or prev < score:
                path_scores[chain] = score

        # 1-hop forward and backward
        for s, o, p, w in out1:
            bump(((s, p, o),), w)
        for s, o, p, w in in1:
            bump(((s, p, o),), w)

        # 2-hop forward: seed -> Y -> Z
        for s1, y, p1, w1 in out1:
            for _, z, p2, w2 in self._out_edges(y):
                if z == seed:
                    continue
                bump(((s1, p1, y), (y, p2, z)), w1 + w2)

                # 3-hop forward: seed -> Y -> Z -> W
                if max_hops >= 3:
                    for _, w_node, p3, w3 in self._out_edges(z):
                        if w_node == seed or w_node == y:
                            continue
                        bump(((s1, p1, y), (y, p2, z), (z, p3, w_node)), w1 + w2 + w3)

        # 2-hop backward: Z -> Y -> seed   (Y is the in-neighbour of seed)
        for x, o1, p1, w1 in in1:
            for z, _, p2, w2 in self._in_edges(x):
                if z == seed:
                    continue
                bump(((z, p2, x), (x, p1, o1)), w1 + w2)

                # 3-hop backward: W -> Z -> Y -> seed
                if max_hops >= 3:
                    for w_node, _, p3, w3 in self._in_edges(z):
                        if w_node == seed or w_node == x:
                            continue
                        bump(((w_node, p3, z), (z, p2, x), (x, p1, o1)), w1 + w2 + w3)

        # Through-chain: X -> seed -> Y
        for x, _, p1, w1 in in1:
            for _, y, p2, w2 in out1:
                if x == y:
                    continue
                bump(((x, p1, seed), (seed, p2, y)), w1 + w2)

    @staticmethod
    def _format_path(chain: Tuple[Tuple[str, str, str], ...], score: int) -> str:
        first = chain[0][0]
        parts = [f"({first.title()})"]
        for s, p, o in chain:
            parts.append(f"-[{p}]->")
            parts.append(f"({o.title()})")
        line = "- " + " ".join(parts)
        if score > len(chain):
            line += f" [Score {score}]"
        return line

    def get_neighborhood(self, words: List[str], global_limit: int = 25) -> List[str]:
        """Spreading-activation GraphRAG over the in-memory NetworkX graph.

        1. Map query words to exact graph nodes (fuzzy → exact matching).
        2. Run 2-hop BFS from each seed, scoring chains by edge-weight sum.
        3. Return the highest-scoring directed paths formatted for the LLM.
        """
        with self._lock:
            seeds = self._map_words_to_seeds(words)
            if not seeds:
                return []
            path_scores: Dict[Tuple, int] = {}
            for seed in seeds:
                self._spreading_activation(seed, path_scores)
            if not path_scores:
                return []
            sorted_paths = sorted(
                path_scores.items(),
                key=lambda item: (item[1], len(item[0])),
                reverse=True
            )[:global_limit]
            return [self._format_path(chain, score) for chain, score in sorted_paths]
