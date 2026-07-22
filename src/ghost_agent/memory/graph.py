import sqlite3
import threading
import difflib
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Iterable, Optional, Tuple

import networkx as nx

logger = logging.getLogger("GhostAgent")


@lru_cache(maxsize=512)
def _entity_pattern(target: str) -> "re.Pattern":
    """Whole-word/whole-phrase matcher for an entity name.

    ``target`` must match as a complete token (or token sequence) inside the
    node name — optionally pluralised — never as a bare substring:
    ``lois`` matches ``lois lane`` and ``lois's dog``, ``game`` matches
    ``games`` and ``chess game``, but ``tin`` does NOT match ``testing`` and
    ``game`` does NOT match ``gamedev``. ``[^\\W_]`` is "unicode alphanumeric,
    underscore excluded", so ``chess_game`` still matches ``game``.
    """
    return re.compile(r"(?<![^\W_])" + re.escape(target) + r"s?(?![^\W_])")


class GraphMemory:
    """Knowledge graph with SQLite persistence + in-memory NetworkX routing.

    SQLite remains the source of truth on disk. A `nx.MultiDiGraph` mirror is
    held in memory and used for spreading-activation traversal in
    `get_neighborhood`. Both stores are kept in sync by `add_triplets`,
    `delete_by_target`, `wipe_all`, and `execute_graph_compression`.
    """

    #: Predicates that are SINGLE-VALUED: a new object supersedes the old one,
    #: which gets a `valid_until` stamp instead of accumulating a contradiction.
    #: Every entry here MUST be one-to-one *for any subject* — a multi-valued
    #: predicate in this set silently expires real knowledge on every new
    #: extraction. `OWNS`/`IS` used to live here and destroyed 19 of the
    #: operator's ownership facts (you own many things; X IS many things);
    #: likewise `HAS_PET`, `HAS_TASK`, `HAS_FEATURE`, `HAS_NAME` (written with
    #: the generic subject `project`) and `HAS_STATE`/`HAS_FEN` (position logs)
    #: are deliberately absent. Comparison is case-insensitive (uppercased).
    _FUNCTIONAL_PREDICATES = {
        # Biographical
        "WORKS_AT", "LIVES_IN", "DRIVES", "MARRIED_TO",
        "BORN_IN", "STUDIES_AT", "LOCATED_IN", "EMPLOYED_BY",
        "HAS_AGE", "HAS_LOCATION",
        # Operational — written by the agent itself, single-valued by
        # construction (a process has one status/one pid at a time).
        "HAS_STATUS", "STATUS", "HAS_PID",
    }

    def __init__(self, memory_dir: Path):
        self.db_path = memory_dir / "knowledge_graph.db"
        self._lock = threading.RLock()
        self.nx_graph: nx.MultiDiGraph = nx.MultiDiGraph()
        # Cached node-name list for _map_words_to_seeds. Rebuilding
        # list(nx_graph.nodes()) on EVERY query word was O(nodes) per turn
        # (the graph is the only uncapped memory tier); this snapshots it and
        # invalidates on any edge mutation.
        self._node_list_cache: Optional[List[str]] = None
        self._init_db()
        self.initialize_graph()

    def _invalidate_node_cache(self):
        self._node_list_cache = None

    def _nodes_snapshot(self) -> List[str]:
        if self._node_list_cache is None:
            self._node_list_cache = list(self.nx_graph.nodes())
        return self._node_list_cache

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
            self._invalidate_node_cache()
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
            self._invalidate_node_cache()  # new nodes may have appeared

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
                self._invalidate_node_cache()

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
                        # predicates (one-to-one) like WORKS_AT, LIVES_IN, DRIVES —
                        # see `_FUNCTIONAL_PREDICATES`. Multi-valued predicates
                        # (LIKES, KNOWS, HAS_*, OWNS, IS) are excluded: expiring
                        # those is silent data loss, not an update.
                        conflicting = []
                        if pn in self._FUNCTIONAL_PREDICATES:
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
                            # Expiry is destructive-by-retrieval (expired edges
                            # vanish from every read path), so it is NEVER silent.
                            for (old_obj,) in conflicting:
                                logger.warning(
                                    "graph expiry: %s %s '%s' superseded by '%s'",
                                    sn, pn, old_obj, on,
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

    @staticmethod
    def _entity_matches(node: str, target: str) -> bool:
        """True when ``node`` names the entity ``target`` (already normalised).

        Whole-token match, NOT substring: `forget("tin")` must not reach
        `testing`/`printing`, and `forget("game")` must not reach `gamedev`.
        Multi-word names still match on any complete token run, so `lois`
        reaches `lois lane` and `mortimer` reaches `mortimer the iguana`.
        """
        n = (node or "").strip().lower()
        if not n or not target:
            return False
        if n == target:
            return True
        return _entity_pattern(target).search(n) is not None

    #: Blast-radius guard for `delete_by_target`. Whole-token matching alone
    #: does not bound a forget: `game` is a legitimate token of 108 rows (8%)
    #: of the live graph. A forget at or above this many rows is therefore
    #: downgraded to a temporal expiry — it disappears from every read path
    #: (mirror, neighborhood, recent triplets) exactly like a delete, but
    #: stays recoverable via `get_expired_triplets`. Small, surgical forgets
    #: keep the original hard-delete semantics.
    _FORGET_SOFT_EXPIRE_MIN_ROWS = 50

    def delete_by_target(self, target: str) -> int:
        if not target or len(target.strip()) < 3:
            return 0
        t_norm = target.lower().strip()
        like = f"%{t_norm}%"
        import time as _time
        now = _time.time()
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # LIKE is only a cheap prefilter — the authoritative test is
                # `_entity_matches`, which requires a whole-token hit. The old
                # code deleted on the raw LIKE, so `forget("tin")` hard-deleted
                # 83 unrelated rows of the production graph with no undo.
                candidates = conn.execute(
                    '''SELECT rowid, subject, predicate, object FROM triplets
                       WHERE subject LIKE ? OR object LIKE ?''',
                    (like, like)
                ).fetchall()
                doomed = [
                    row for row in candidates
                    if self._entity_matches(row[1], t_norm)
                    or self._entity_matches(row[3], t_norm)
                ]
                if not doomed:
                    return 0
                live_rows = conn.execute(
                    'SELECT COUNT(*) FROM triplets WHERE valid_until IS NULL'
                ).fetchone()[0] or 0
                if len(doomed) >= self._FORGET_SOFT_EXPIRE_MIN_ROWS:
                    logger.warning(
                        "graph forget '%s': %d/%d live rows matched — EXPIRING "
                        "instead of deleting (recoverable via get_expired_triplets)",
                        t_norm, len(doomed), live_rows,
                    )
                    conn.executemany(
                        'UPDATE triplets SET valid_until = ? WHERE rowid = ?',
                        [(now, row[0]) for row in doomed],
                    )
                else:
                    conn.executemany(
                        'DELETE FROM triplets WHERE rowid = ?',
                        [(row[0],) for row in doomed],
                    )
                deleted = len(doomed)
                conn.commit()
            for _, s, p, o in doomed:
                self._remove_edge(s, p, o)
        return deleted

    #: Generic hub nodes that link to nearly everything; expanding a
    #: `forget` to these would wipe unrelated knowledge, so they are never
    #: returned as connected entities. Pronouns are not enough — the live
    #: graph's real hubs are operational nouns (`assistant` deg 54, `project`
    #: 49, `system` 43, `done` 14). Also used as a guard in
    #: `_map_words_to_seeds` so a longer query word cannot re-seed a hub.
    _ENTITY_EXPANSION_STOPLIST = {
        "user", "me", "i", "you", "it", "this", "that", "they", "them",
        "he", "she", "we", "thing", "things",
        "assistant", "agent", "ghost", "system", "project", "projects",
        "task", "tasks", "todo", "done", "bug", "bugs", "issue", "issues",
        "file", "files", "code", "error", "errors", "service", "services",
        "status", "test", "tests", "data", "session", "model", "tool",
        "tools", "memory", "goal", "goals", "feature", "features",
    }

    #: Dynamic companion to the stoplist: any neighbour with a degree above
    #: this is a hub by measurement, whatever its name, and is never followed
    #: by the forget expansion (forgetting a chess bot must not reach `webos`).
    _EXPANSION_MAX_DEGREE = 8

    def get_connected_entities(self, target: str, limit: int = 8) -> List[str]:
        """Return distinct entity names directly (1 hop) connected to
        ``target``.

        Lets ``forget`` expand an entity wipe to its tightly-coupled
        neighbours: forgetting ``mortimer`` surfaces ``iguana`` (from a
        ``mortimer IS_A iguana`` edge) so the alias tombstone goes too.

        Every returned name is fed back into ``delete_by_target`` by the
        caller, so this is the amplifier of any forget: it is bounded on
        three axes — the anchor must match a WHOLE token of the node name
        (not a substring), only CURRENT edges are followed, and hub
        neighbours (by stoplist or by measured degree) are never returned.
        """
        if not target or len(target.strip()) < 3:
            return []
        t = target.lower().strip()
        like = f"%{t}%"
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    '''SELECT subject, object FROM triplets
                       WHERE (subject LIKE ? OR object LIKE ?)
                       AND valid_until IS NULL''',
                    (like, like)
                ).fetchall()
            out: List[str] = []
            seen = set()
            for s, o in rows:
                # Only follow an edge the target genuinely anchors, and only
                # to the OTHER endpoint.
                if self._entity_matches(s, t):
                    neighbours = (o,)
                elif self._entity_matches(o, t):
                    neighbours = (s,)
                else:
                    continue
                for node in neighbours:
                    n = (node or "").lower().strip()
                    # Skip the target's own variants, hub nodes, and tiny tokens.
                    if not n or len(n) < 3 or t in n or n in t:
                        continue
                    if n in self._ENTITY_EXPANSION_STOPLIST:
                        continue
                    if n in self.nx_graph and \
                            self.nx_graph.degree(n) > self._EXPANSION_MAX_DEGREE:
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
            self._invalidate_node_cache()

    def prune_stale_edges(self, max_age_days: int = 45, keep_min_weight: int = 1) -> int:
        """Forget low-signal stale edges — the graph's decay story.

        The graph is the only uncapped memory tier (vector 5000, episodes 500,
        skills 50); non-functional predicates (LIKES/HAS/KNOWS/…) accumulate
        forever, and weight-1 edges older than a threshold are almost always
        one-off extractor noise that dilutes retrieval. This deletes currently-
        valid edges with ``weight <= keep_min_weight`` and a ``timestamp``
        older than ``max_age_days`` from BOTH the DB and the in-memory mirror.
        Reinforced edges (weight > threshold) are kept regardless of age —
        weight IS the decay signal, previously stored but never used for
        forgetting. Returns the number of edges pruned. Idempotent, best-effort.
        Intended to run from the dream cycle."""
        removed = 0
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cutoff_expr = f"datetime('now', '-{int(max_age_days)} days')"
                    rows = conn.execute(
                        f"""SELECT subject, predicate, object FROM triplets
                            WHERE valid_until IS NULL
                              AND COALESCE(weight, 1) <= ?
                              AND timestamp < {cutoff_expr}""",
                        (int(keep_min_weight),),
                    ).fetchall()
                    for s, p, o in rows:
                        conn.execute(
                            "DELETE FROM triplets WHERE subject=? AND predicate=? AND object=?",
                            (s, p, o),
                        )
                        self._remove_edge(s, p, o)
                        removed += 1
                    conn.commit()
                if removed:
                    self._invalidate_node_cache()
        except Exception as e:
            logger.warning("graph prune_stale_edges failed: %s", e)
        return removed

    def get_recent_triplets(self, limit: int = 100) -> List[Dict[str, str]]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                # Only CURRENT facts (valid_until IS NULL) — otherwise a
                # superseded/expired triplet ("bob WORKS_AT google", later
                # replaced by "meta") is returned alongside the live one and,
                # if surfaced into context, contradicts the current fact.
                cursor = conn.execute(
                    'SELECT subject, predicate, object FROM triplets '
                    'WHERE valid_until IS NULL ORDER BY timestamp DESC LIMIT ?',
                    (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]

    def propose_merge_candidates(self, max_candidates: int = 12,
                                 neighbor_window: int = 5,
                                 fuzzy_cutoff: float = 0.90) -> List[Dict[str, str]]:
        """Deterministic near-duplicate node pairs for dream-time compression.

        Two tiers, distinguished by ``kind``:

        - ``"safe"`` — names identical after stripping punctuation/whitespace
          ("new-york" vs "new york"): mergeable without confirmation.
        - ``"fuzzy"`` — high-similarity lexicographic neighbors (plural forms,
          trailing typos): the caller must get an LLM same-entity confirmation
          before merging — string similarity alone conflates distinct entities
          ("new"/"news").

        Fuzzy comparison is bounded to each node's ``neighbor_window`` sorted
        neighbors (near-duplicates share prefixes) so the pass stays
        ~O(n log n) on the only uncapped memory tier. Canonical direction: the
        higher-degree node survives as ``new_node`` (ties: the longer name).
        Read-only — the merge policy lives in core/dream.py."""
        import re

        def _norm(name: str) -> str:
            return re.sub(r"[\s\-_./'\"]+", "", name)

        with self._lock:
            nodes = [n for n in self._nodes_snapshot()
                     if isinstance(n, str) and len(n) > 2 and not n.isdigit()]
            deg = {n: self.nx_graph.degree(n) for n in nodes}

        out: List[Dict[str, str]] = []
        seen: set = set()

        def _add(a: str, b: str, kind: str):
            key = tuple(sorted((a, b)))
            if key in seen:
                return
            seen.add(key)
            # Higher-degree node survives; tie broken toward the longer name.
            if (deg.get(a, 0), len(a)) >= (deg.get(b, 0), len(b)):
                old, new = b, a
            else:
                old, new = a, b
            out.append({"old_node": old, "new_node": new, "kind": kind})

        by_norm: Dict[str, List[str]] = {}
        for n in nodes:
            by_norm.setdefault(_norm(n), []).append(n)
        for variants in by_norm.values():
            for other in variants[1:]:
                _add(variants[0], other, "safe")

        ordered = sorted(nodes)
        for i, a in enumerate(ordered):
            if len(out) >= max_candidates:
                break
            for b in ordered[i + 1:i + 1 + neighbor_window]:
                if _norm(a) == _norm(b):
                    continue  # tier-1 pair (or already merged direction)
                if difflib.SequenceMatcher(None, a, b).ratio() >= fuzzy_cutoff:
                    _add(a, b, "fuzzy")

        return out[:max_candidates]

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
                            # Snapshot every triple that touches old_node on
                            # EITHER side (a self-loop old->old matches once),
                            # then rewrite old_node -> new_node wherever it
                            # appears and merge each rewritten triple into the
                            # target. We carry valid_from/valid_until through so
                            # a superseded (expired) fact does NOT come back as
                            # current, and rewrite both endpoints so an
                            # old->old self-loop survives as new->new instead of
                            # being deleted. Weights sum; temporal state merges
                            # current-wins (see _merge_triplet_row).
                            cursor.execute(
                                '''SELECT subject, predicate, object, weight, valid_from, valid_until
                                   FROM triplets WHERE subject = ? OR object = ?''',
                                (old_node, old_node))
                            src_rows = cursor.fetchall()
                            for subj, pred, obj, w, vfrom, vuntil in src_rows:
                                n_subj = new_node if subj == old_node else subj
                                n_obj = new_node if obj == old_node else obj
                                if n_subj == n_obj and subj != obj:
                                    # The edge ran BETWEEN the two merged nodes
                                    # ("new-york SAME_AS new york"). Rewriting it
                                    # mints a self-loop that duplicates the merged
                                    # node's whole neighbourhood in every
                                    # spreading-activation chain. Drop it — a
                                    # genuine old->old self-loop (subj == obj)
                                    # still migrates to new->new below.
                                    continue
                                self._merge_triplet_row(
                                    cursor, n_subj, pred, n_obj,
                                    int(w or 1), vfrom, vuntil)
                            # Delete the migrated source rows. Rewritten targets
                            # never reference old_node, so this only removes the
                            # originals, never the freshly-merged copies.
                            cursor.execute(
                                "DELETE FROM triplets WHERE subject = ? OR object = ?",
                                (old_node, old_node))
                            # A subject-side merge unions two different objects of
                            # the same functional predicate ("bob WORKS_AT google"
                            # + "bobby WORKS_AT meta"), minting two mutually
                            # exclusive CURRENT facts. add_triplets' conflict
                            # resolution never runs here, so re-apply it.
                            self._reconcile_functional_conflicts(cursor, new_node)
                            ops += 1
                        except Exception as e:
                            logger.debug(f"Graph compression merge failed: {type(e).__name__}: {e}")
                conn.commit()
            # Rebuild mirror after a structural rewrite — simpler than diffing.
            self.initialize_graph()
        return ops

    def _reconcile_functional_conflicts(self, cursor, subject: str) -> int:
        """Re-apply single-valued-predicate expiry for one subject.

        `add_triplets` resolves functional conflicts at write time, but node
        compression rewrites rows straight into SQLite and can leave two
        CURRENT objects under the same functional predicate. Newest wins
        (latest ``valid_from``, then ``timestamp``, then insertion order);
        the losers get a ``valid_until`` stamp. Returns the number expired."""
        import time as _time
        now = _time.time()
        expired = 0
        rows = cursor.execute(
            '''SELECT rowid, predicate, object, COALESCE(valid_from, 0), timestamp
               FROM triplets WHERE subject = ? AND valid_until IS NULL''',
            (subject,)).fetchall()
        by_pred: Dict[str, List] = {}
        for rid, pred, obj, vfrom, ts in rows:
            pred_u = str(pred or "").upper()
            if pred_u in self._FUNCTIONAL_PREDICATES:
                by_pred.setdefault(pred_u, []).append((vfrom, ts or "", rid, pred, obj))
        for pred_u, items in by_pred.items():
            if len(items) < 2:
                continue
            items.sort(reverse=True)  # newest valid_from / timestamp / rowid first
            winner = items[0]
            for vfrom, ts, rid, pred, obj in items[1:]:
                cursor.execute(
                    'UPDATE triplets SET valid_until = ? WHERE rowid = ?', (now, rid))
                expired += 1
                logger.warning(
                    "graph expiry (merge): %s %s '%s' superseded by '%s'",
                    subject, pred, obj, winner[4],
                )
        return expired

    @staticmethod
    def _merge_triplet_row(cursor, subject: str, predicate: str, obj: str,
                           weight: int, valid_from, valid_until) -> None:
        """Insert (subject, predicate, obj) or, if it already exists, merge into
        it: weights sum, valid_from keeps the earliest, and valid_until is
        current-wins — if EITHER the incoming or existing row is current
        (valid_until IS NULL) the result is current; if both are expired we keep
        the later (max) expiry. Used by graph compression so a node merge never
        resurrects a superseded fact and never double-counts weight."""
        cursor.execute(
            '''SELECT weight, valid_from, valid_until FROM triplets
               WHERE subject = ? AND predicate = ? AND object = ?''',
            (subject, predicate, obj))
        existing = cursor.fetchone()
        if existing is None:
            cursor.execute(
                '''INSERT INTO triplets (subject, predicate, object, weight, timestamp, valid_from, valid_until)
                   VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)''',
                (subject, predicate, obj, int(weight or 1), valid_from, valid_until))
            return
        ex_w, ex_from, ex_until = existing
        merged_w = int(ex_w or 1) + int(weight or 1)
        if ex_until is None or valid_until is None:
            merged_until = None
        else:
            merged_until = max(ex_until, valid_until)
        froms = [v for v in (ex_from, valid_from) if v is not None]
        merged_from = min(froms) if froms else None
        cursor.execute(
            '''UPDATE triplets SET weight = ?, timestamp = CURRENT_TIMESTAMP,
                                   valid_from = ?, valid_until = ?
               WHERE subject = ? AND predicate = ? AND object = ?''',
            (merged_w, merged_from, merged_until, subject, predicate, obj))

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

    #: Minimum length / minimum length-ratio for a node name that is a
    #: FRAGMENT of the query word ('ai' from 'aiohttp', 'user' from
    #: 'username'). That direction re-seeds generic ego hubs from unrelated
    #: words and blows the ego graph into every turn's context — the exact
    #: hydration that `bus._STOPWORDS` exists to prevent. The opposite
    #: direction (word inside a longer node: 'germ' -> 'germany') narrows
    #: instead of widening and stays unrestricted.
    _SEED_FRAGMENT_MIN_LEN = 4
    _SEED_FRAGMENT_MIN_RATIO = 0.75

    def _seed_containment_ok(self, node: str, word: str) -> bool:
        """Guard for the substring tier of `_map_words_to_seeds`."""
        if node == word:
            return True
        # Never reach a hub node except by an exact query word.
        if node in self._ENTITY_EXPANSION_STOPLIST:
            return False
        if node in word:  # node is a fragment of the word — the risky direction
            if len(node) < self._SEED_FRAGMENT_MIN_LEN:
                return False
            if len(node) < self._SEED_FRAGMENT_MIN_RATIO * len(word):
                return False
        return True

    def _map_words_to_seeds(self, words: Iterable[str]) -> List[str]:
        """Map free-form query words to exact node names in the graph.

        Strategy per word: exact match → substring containment → difflib
        fuzzy fallback. Words shorter than 3 chars are skipped.
        """
        if not self.nx_graph or self.nx_graph.number_of_nodes() == 0:
            return []
        seeds: List[str] = []
        seen = set()
        all_nodes = self._nodes_snapshot()
        for w in words:
            wl = str(w).lower().strip()
            if len(wl) < 3:
                continue
            matches: List[str] = []
            if wl in self.nx_graph:
                matches = [wl]
            else:
                substr = [n for n in all_nodes
                          if (wl in n or n in wl) and self._seed_containment_ok(n, wl)]
                if substr:
                    # Prefer the closest length match for stable ordering
                    matches = sorted(substr, key=lambda n: (abs(len(n) - len(wl)), n))[:3]
                else:
                    # Same hub protection on the fuzzy tier: 'systemd' is a
                    # 0.92 difflib match for the 'system' hub.
                    matches = [
                        m for m in difflib.get_close_matches(wl, all_nodes, n=3, cutoff=0.7)
                        if m not in self._ENTITY_EXPANSION_STOPLIST
                    ]
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
